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
#include "engine/sampling.h"
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

// Greedy: first argmax. Otherwise an exact sample of the filtered target distribution (engine/sampling.h).
static int32_t sample_with_state(EngineState& st, const SamplingParams& p, const float* logits, int64_t V) {
  if (p.temperature <= 0.0f) return argmax_first(logits, V);
  if (st.seeded != p.seed) {
    st.rng.seed(p.seed ? p.seed : std::random_device{}());
    st.seeded = p.seed;
  }
  target_probs(p, logits, V, st.probs);
  return sample_from(st.probs, std::uniform_real_distribution<double>(0.0, 1.0)(st.rng));
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
