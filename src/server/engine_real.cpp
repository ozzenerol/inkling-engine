// Adapter from ink::Engine to EngineIface. Built only when the inkcore target exists. Owner: C.
#include "engine_iface.h"

namespace ink {

#ifdef INK_HAVE_ENGINE
namespace {
class RealEngine : public EngineIface {
 public:
  explicit RealEngine(const EngineOptions& o) : e_(o) {}
  int64_t cached_len() const override { return e_.cached_len(); }
  bool rewind(int64_t n) override { return e_.rewind(n); }
  bool reset() override {
    e_.reset();
    return true;
  }
  Ckpt checkpoint() override { return e_.checkpoint(); }
  bool restore(const Ckpt& c) override { return e_.restore(c); }
  void drop(const Ckpt& c) override { e_.drop(c); }
  void prefill(std::span<const int32_t> t, std::function<void(int64_t)> p) override { e_.prefill(t, std::move(p)); }
  void generate(const SamplingParams& sp, int max_tokens, const std::vector<int32_t>& stops,
                const std::function<bool(int32_t)>& on_token) override {
    e_.generate(sp, max_tokens, stops, on_token);
  }
  std::string name() const override { return "inkling"; }

 private:
  Engine e_;
};
}  // namespace

std::unique_ptr<EngineIface> make_real_engine(const EngineOptions& o) { return std::make_unique<RealEngine>(o); }
#else
std::unique_ptr<EngineIface> make_real_engine(const EngineOptions&) { return nullptr; }
#endif

}  // namespace ink
