// Engine API used by the server (C) and CLI. Header owned by A; implementation A (model) + B (speculative, kernels).
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ink {

struct EngineOptions {
  std::string model_dir;
  int64_t max_context = 131072;
  bool use_mtp = true;     // DSpark drafting
  int draft_block = 8;  // 1 + up to 7 drafter tokens per verify (the drafter's own block size)
  int prefill_chunk = 2048;
};

struct SamplingParams {
  float temperature = 0.0f;  // 0 = greedy
  float top_p = 1.0f;
  int top_k = 0;
  float min_p = 0.0f;
  uint64_t seed = 0;
};

struct Ckpt {
  int64_t id = -1;
  int64_t len = 0;  // cached_len() at the time of the checkpoint
};

class Model;  // src/model/model.h
struct EngineState;  // engine.cpp

// Single sequence. The cache holds KV + conv state for all 48 layer visits.
class Engine {
 public:
  explicit Engine(const EngineOptions&);
  ~Engine();

  int64_t cached_len() const;

  // Keep the first n tokens. Works only within the slack kept for the sliding-window rings
  // (n >= cached_len() - slack); returns false otherwise and changes nothing.
  bool rewind(int64_t n);
  void reset();  // empty the cache (checkpoints stay valid)

  // Snapshot of the full cache state at cached_len(). Up to a few are kept (each costs the local-layer rings +
  // conv states; global KV is shared because it is append-only and restore truncates it).
  Ckpt checkpoint();
  bool restore(const Ckpt&);   // cache becomes exactly as at checkpoint time; false if dropped/invalid
  void drop(const Ckpt&);

  // Feed tokens after the cached ones (chunked internally). Afterwards last_logits() holds the logits of the
  // last fed token.
  void prefill(std::span<const int32_t> tokens, std::function<void(int64_t done)> progress = {});

  // Sample from last_logits(), then keep decoding (speculative decoding inside) until a stop id, max_tokens,
  // or on_token returns false. on_token sees every emitted token including the stop token.
  // Afterwards cached_len() = before + emitted - 1: the LAST emitted token is NOT in the cache
  // (feed it with the next prefill if the conversation continues).
  void generate(const SamplingParams&, int max_tokens, const std::vector<int32_t>& stop_ids,
                const std::function<bool(int32_t tok)>& on_token);

  const std::vector<float>& last_logits() const;  // fp32, unpadded vocab (200058), after softcap

  // Internals for speculative.cpp (B) and tests.
  Model& model() { return *model_; }
  EngineState* state() const { return st_.get(); }
  const EngineOptions& options() const { return opts_; }
  int32_t sample(const SamplingParams&, const float* logits, int64_t vocab);
  void invalidate_from(int64_t pos);  // checkpoints that rely on global KV at positions >= pos become invalid

 private:
  std::unique_ptr<EngineState> st_;
  std::unique_ptr<Model> model_;
  EngineOptions opts_;
};

}  // namespace ink
