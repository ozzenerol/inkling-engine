// Cross-turn prefix KV cache policy. The engine holds one sequence; this keeps the token ids whose KV is in
// it, finds the longest common prefix with each new prompt and gets the engine there by the cheapest way:
// nothing / rewind (within the ring slack) / restore of a checkpoint <= the prefix / reset. Checkpoints are
// taken at the end of each prompt prefill and at the end of each finished turn (LRU, max_ckpts).
// Invariant: every kept checkpoint is a prefix of toks_. Owner: C.
#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "engine_iface.h"

namespace ink {

class PromptCache {
 public:
  PromptCache(EngineIface& e, int max_ckpts = 4) : e_(e), max_ckpts_(max_ckpts) {}

  struct Prep {
    int64_t reused = 0;     // prompt tokens taken from the cache
    int64_t prefilled = 0;  // tokens fed now
    double prefill_s = 0;
    std::string how;        // "hit", "rewind", "ckpt", "reset"
  };
  // Afterwards the engine has prefilled the whole prompt and holds its last logits.
  Prep prepare(const std::vector<int32_t>& prompt, const std::function<void(int64_t done, int64_t total)>& progress);

  // After Engine::generate: emitted = every token on_token saw. turn_done: ended on a stop token.
  void after_generate(const std::vector<int32_t>& emitted, bool turn_done);

  // After an engine error: forget everything.
  void invalidate();

  int64_t size() const { return (int64_t)toks_.size(); }
  int n_ckpts() const { return (int)ck_.size(); }

 private:
  void take_ckpt();
  void drop_above(int64_t len);

  EngineIface& e_;
  int max_ckpts_;
  std::vector<int32_t> toks_;
  struct Entry {
    Ckpt c;
    uint64_t used;
  };
  std::vector<Entry> ck_;
  uint64_t clock_ = 0;
};

}  // namespace ink
