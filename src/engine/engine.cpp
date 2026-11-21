// Owner: A. Engine: chunked prefill, sampling, rewind, checkpoints. Speculative decoding (B) plugs in through
// speculative_generate() in speculative.cpp when it exists.
#include "engine/engine.h"

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

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
  std::vector<float> probs;    // sampler scratch
  std::vector<int32_t> cand;
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

// Samples from softmax(logits / temperature) over the FULL vocabulary (no hidden truncation), with optional
// min_p (relative to the top probability), top_k and top_p filters. One vectorized exp pass, no full sort:
// about 0.15 ms per call for 200k tokens, so it stays small next to a 10-30 ms decode step.
static int32_t sample_with_state(EngineState& st, const SamplingParams& p, const float* logits, int64_t V) {
  float mx = 0;
  vDSP_maxv(logits, 1, &mx, vDSP_Length(V));
  if (p.temperature <= 0.0f) {
    for (int64_t i = 0; i < V; ++i)
      if (logits[i] == mx) return int32_t(i);  // first maximum, as std::max_element
    return 0;
  }
  if (st.seeded != p.seed) {
    st.rng.seed(p.seed ? p.seed : std::random_device{}());
    st.seeded = p.seed;
  }
  auto& pr = st.probs;
  pr.resize(static_cast<size_t>(V));
  const float inv_t = 1.0f / p.temperature, shift = -mx * inv_t;
  vDSP_vsmsa(logits, 1, &inv_t, &shift, pr.data(), 1, vDSP_Length(V));  // (l - max) / T  <= 0
  const int n = int(V);
  vvexpf(pr.data(), pr.data(), &n);  // top token has pr = 1
  double z = 0;
  for (int64_t i = 0; i < V; ++i) z += pr[i];
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  const bool filtered = p.min_p > 0.0f || (p.top_k > 0 && p.top_k < V) || p.top_p < 1.0f;
  if (!filtered) {
    double r = uni(st.rng) * z;
    int64_t last = 0;
    for (int64_t i = 0; i < V; ++i) {
      if (pr[i] <= 0.0f) continue;
      last = i;
      r -= pr[i];
      if (r <= 0) return int32_t(i);
    }
    return int32_t(last);
  }
  // Candidates: everything above min_p * top (or above 1e-9 * top when only top_k / top_p apply; the mass left
  // out is below 2e-4 in the worst case and only matters for top_p > 0.9998, where all candidates are kept).
  const float thr = p.min_p > 0.0f ? p.min_p : 1e-9f;
  auto& cand = st.cand;
  cand.clear();
  for (int64_t i = 0; i < V; ++i)
    if (pr[i] >= thr) cand.push_back(int32_t(i));
  std::sort(cand.begin(), cand.end(), [&](int32_t a, int32_t b) { return pr[a] > pr[b] || (pr[a] == pr[b] && a < b); });
  size_t keep = cand.size();
  if (p.top_k > 0 && size_t(p.top_k) < keep) keep = size_t(p.top_k);
  if (p.top_p < 1.0f) {
    double c = 0;
    for (size_t i = 0; i < keep; ++i) {
      c += pr[cand[i]] / z;
      if (c >= p.top_p) { keep = i + 1; break; }
    }
  }
  double tot = 0;
  for (size_t i = 0; i < keep; ++i) tot += pr[cand[i]];
  double r = uni(st.rng) * tot;
  for (size_t i = 0; i < keep; ++i) {
    r -= pr[cand[i]];
    if (r <= 0) return cand[i];
  }
  return cand[keep - 1];
}

int32_t Engine::sample(const SamplingParams& p, const float* logits, int64_t V) {
  return sample_with_state(S(this), p, logits, V);
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

// speculative_generate / speculative_attach: src/engine/speculative.cpp (B). No weak defaults here: with both in
// libinkcore.a the linker resolves the weak copy from engine.cpp.o and never pulls speculative.cpp.o.

// Test shim (tests/ref/sample_test.cpp): the sampler without a model.
namespace ink {
int32_t sample_for_test(const SamplingParams& p, const float* logits, int64_t V, uint64_t*) {
  static EngineState st;
  return sample_with_state(st, p, logits, V);
}
}  // namespace ink
