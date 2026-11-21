// What the server needs from an engine. RealEngine wraps ink::Engine (src/engine/engine.h); StubEngine is a
// scripted fake with the same cache semantics, used to test the server without the GPU. Owner: C.
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "engine/engine.h"  // SamplingParams, Ckpt, EngineOptions (header only; no link dependency)

namespace ink {

class Tokenizer;

class EngineIface {
 public:
  virtual ~EngineIface() = default;
  virtual int64_t cached_len() const = 0;
  virtual bool rewind(int64_t n) = 0;
  virtual bool reset() = 0;  // empty cache; false if unsupported
  virtual Ckpt checkpoint() = 0;
  virtual bool restore(const Ckpt&) = 0;
  virtual void drop(const Ckpt&) = 0;
  virtual void prefill(std::span<const int32_t> tokens, std::function<void(int64_t)> progress) = 0;
  virtual void generate(const SamplingParams&, int max_tokens, const std::vector<int32_t>& stop_ids,
                        const std::function<bool(int32_t)>& on_token) = 0;
  virtual std::string name() const = 0;
};

struct StubOptions {
  double prefill_us_per_token = 2;  // simulated speed
  double decode_ms_per_token = 5;
  int64_t slack = 512;
  int max_ckpts = 8;
  int reply_tokens = 0;  // >0: long reply (half thinking, half text) of about this many tokens
};
std::unique_ptr<EngineIface> make_stub_engine(const Tokenizer& tok, const StubOptions& o);

// nullptr when the binary was built without the engine (no inkcore target).
std::unique_ptr<EngineIface> make_real_engine(const EngineOptions& o);

}  // namespace ink
