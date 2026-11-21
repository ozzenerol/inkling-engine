// Owner: A. Engine: chunked prefill, sampling, rewind, checkpoints. Speculative decoding (B) plugs in through
// speculative_generate() in speculative.cpp when it exists.
#include "engine/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <stdexcept>

#include "model/backend.h"
#include "engine/speculative.h"
#include "model/model.h"

namespace ink {


namespace {

struct Snapshot {
  int64_t id = -1;
  int64_t len = 0;
  bool valid = false;
  std::vector<std::array<Tensor, 4>> conv;  // per visit
  std::vector<std::array<Tensor, 2>> ring;  // per visit (local only): k, v
};

}  // namespace

struct EngineState {
  std::unique_ptr<Backend> be;
  std::vector<Snapshot> snaps;
  int64_t next_id = 1;
  std::mt19937_64 rng{0};
  uint64_t seeded = ~0ull;
};

static EngineState& S(const Engine* e) { return *e->state(); }

Engine::Engine(const EngineOptions& o) {
  st_ = std::make_unique<EngineState>();
  const char* b = std::getenv("INK_BACKEND");
  st_->be = (b && std::string(b) == "cpu") ? make_cpu_backend() : make_metal_backend();
  model_ = std::make_unique<Model>(*st_->be, o.model_dir, o.max_context, o.prefill_chunk);
  opts_ = o;
  if (opts_.use_mtp) speculative_attach(*this, *model_);
}

Engine::~Engine() {
  auto& st = S(this);
  for (auto& s : st.snaps) {
    for (auto& c : s.conv) for (auto& t : c) if (t.valid()) st.be->free(t);
    for (auto& r : s.ring) for (auto& t : r) if (t.valid()) st.be->free(t);
  }
  model_.reset();
}

int64_t Engine::cached_len() const { return model_->len(); }

bool Engine::rewind(int64_t n) {
  if (n == model_->len()) return true;
  if (n == 0) {
    reset();
    return true;
  }
  // Conv states can only go back to rows recorded by the last keep_hist forward (speculative verify).
  int64_t rows = model_->hist_rows();
  int64_t base = model_->len() - rows;
  if (rows > 0 && n > base && n < model_->len()) {
    model_->rollback(n - base);
    return true;
  }
  return false;
}

void Engine::reset() { model_->reset(); }

Ckpt Engine::checkpoint() {
  auto& st = S(this);
  auto& be = *st.be;
  auto& vs = model_->visits();
  Snapshot* slot = nullptr;
  for (auto& s : st.snaps)
    if (!s.valid) { slot = &s; break; }
  if (!slot) {
    if (st.snaps.size() < 4) {
      st.snaps.emplace_back();
      slot = &st.snaps.back();
    } else {  // evict the oldest
      slot = &*std::min_element(st.snaps.begin(), st.snaps.end(), [](auto& a, auto& b) { return a.id < b.id; });
    }
  }
  if (slot->conv.empty()) {
    slot->conv.resize(vs.size());
    slot->ring.resize(vs.size());
    for (size_t v = 0; v < vs.size(); ++v) {
      for (int i = 0; i < 4; ++i) slot->conv[v][i] = be.alloc(DType::F32, {3, vs[v].conv[i].dim(1)}, "ckpt.conv");
      if (vs[v].kv.ring) {
        slot->ring[v][0] = be.alloc(DType::BF16, {vs[v].kv.k.numel()}, "ckpt.k");
        slot->ring[v][1] = be.alloc(DType::BF16, {vs[v].kv.v.numel()}, "ckpt.v");
      }
    }
  }
  for (size_t v = 0; v < vs.size(); ++v) {
    for (int i = 0; i < 4; ++i) be.copy(vs[v].conv[i], slot->conv[v][i]);
    if (vs[v].kv.ring) {
      be.copy(vs[v].kv.k, slot->ring[v][0]);
      be.copy(vs[v].kv.v, slot->ring[v][1]);
    }
  }
  be.sync();
  slot->id = st.next_id++;
  slot->len = model_->len();
  slot->valid = true;
  return Ckpt{slot->id, slot->len};
}

bool Engine::restore(const Ckpt& c) {
  auto& st = S(this);
  auto& be = *st.be;
  auto& vs = model_->visits();
  for (auto& s : st.snaps) {
    if (!s.valid || s.id != c.id) continue;
    for (size_t v = 0; v < vs.size(); ++v) {
      for (int i = 0; i < 4; ++i) be.copy(s.conv[v][i], vs[v].conv[i]);
      if (vs[v].kv.ring) {
        be.copy(s.ring[v][0], vs[v].kv.k);
        be.copy(s.ring[v][1], vs[v].kv.v);
      }
    }
    be.sync();
    model_->set_len(s.len);
    return true;
  }
  return false;
}

void Engine::drop(const Ckpt& c) {
  for (auto& s : S(this).snaps)
    if (s.id == c.id) s.valid = false;
}

// Any write at positions below a checkpoint's length overwrites global KV it relies on.
void Engine::invalidate_from(int64_t pos) {
  for (auto& s : S(this).snaps)
    if (s.valid && pos < s.len) s.valid = false;
}

void Engine::prefill(std::span<const int32_t> tokens, std::function<void(int64_t)> progress) {
  if (tokens.empty()) return;
  invalidate_from(model_->len());
  const int64_t chunk = opts_.prefill_chunk;
  int64_t done = 0;
  while (done < int64_t(tokens.size())) {
    int64_t T = std::min<int64_t>(chunk, int64_t(tokens.size()) - done);
    model_->forward(tokens.data() + done, T, false);
    done += T;
    if (progress) progress(done);
  }
}

const std::vector<float>& Engine::last_logits() const { return model_->logits(); }

int32_t Engine::sample(const SamplingParams& p, const float* logits, int64_t V) {
  auto& st = S(this);
  if (p.temperature <= 0.0f) {
    return int32_t(std::max_element(logits, logits + V) - logits);
  }
  if (st.seeded != p.seed) {
    st.rng.seed(p.seed ? p.seed : std::random_device{}());
    st.seeded = p.seed;
  }
  std::vector<int32_t> idx(static_cast<size_t>(V));
  std::iota(idx.begin(), idx.end(), 0);
  int64_t k = (p.top_k > 0 && p.top_k < V) ? p.top_k : std::min<int64_t>(V, 1024);
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });
  std::vector<double> pr(static_cast<size_t>(k));
  const double inv_t = 1.0 / p.temperature, m = logits[idx[0]];
  double z = 0;
  for (int64_t i = 0; i < k; ++i) z += (pr[i] = std::exp((logits[idx[i]] - m) * inv_t));
  int64_t keep = k;
  if (p.min_p > 0)
    for (int64_t i = 0; i < k; ++i)
      if (pr[i] < p.min_p * pr[0]) { keep = std::max<int64_t>(1, i); break; }
  if (p.top_p < 1.0f) {
    double c = 0;
    for (int64_t i = 0; i < keep; ++i) {
      c += pr[i] / z;
      if (c >= p.top_p) { keep = i + 1; break; }
    }
  }
  double tot = 0;
  for (int64_t i = 0; i < keep; ++i) tot += pr[i];
  double r = std::uniform_real_distribution<double>(0, tot)(st.rng);
  for (int64_t i = 0; i < keep; ++i) {
    r -= pr[i];
    if (r <= 0) return idx[i];
  }
  return idx[keep - 1];
}

void Engine::generate(const SamplingParams& p, int max_tokens, const std::vector<int32_t>& stop_ids,
                      const std::function<bool(int32_t)>& on_token) {
  if (max_tokens <= 0) return;
  invalidate_from(model_->len());
  const int64_t V = model_->config().vocab_unpadded;
  auto sample_last = [&]() -> int32_t {
    const auto& lg = model_->logits();
    return sample(p, lg.data() + (model_->logit_rows() - 1) * V, V);
  };
  if (opts_.use_mtp && speculative_generate(*this, *model_, p, max_tokens, stop_ids, on_token, sample_last)) return;
  for (int n = 0; n < max_tokens; ++n) {
    int32_t tok = sample_last();
    bool stop = std::find(stop_ids.begin(), stop_ids.end(), tok) != stop_ids.end();
    bool more = on_token(tok);
    if (stop || !more || n + 1 == max_tokens) break;
    model_->forward(&tok, 1, false);
  }
}

}  // namespace ink

// Default until B's speculative.cpp defines it (weak so the strong definition wins when linked).
__attribute__((weak)) bool ink::speculative_generate(Engine&, Model&, const SamplingParams&, int,
                                                     const std::vector<int32_t>&, const std::function<bool(int32_t)>&,
                                                     const std::function<int32_t()>&) {
  return false;
}

__attribute__((weak)) void ink::speculative_attach(Engine&, Model&) {}
