// Scripted engine: keeps the token history like a real cache, checks the cache contract (rewind within slack,
// checkpoints, prefill after cached tokens) and answers with a fixed Inkling-format reply:
//  - if tools were declared and the last user text contains "TOOL", a call to the first tool;
//  - else thinking + "Stub reply #<n> to: <first 40 bytes of the last user text>".
// Owner: C.
#include <chrono>
#include <map>
#include <stdexcept>
#include <thread>

#include "engine_iface.h"
#include "pyjson.h"
#include "tokenizer/tokenizer.h"

namespace ink {
namespace {

class StubEngine : public EngineIface {
 public:
  StubEngine(const Tokenizer& t, const StubOptions& o) : tok_(t), o_(o) {}

  int64_t cached_len() const override { return (int64_t)hist_.size(); }
  bool rewind(int64_t n) override {
    if (n < 0 || n > cached_len() || cached_len() - n > o_.slack) return false;
    hist_.resize(n);
    return true;
  }
  bool reset() override { hist_.clear(); return true; }
  Ckpt checkpoint() override {
    if ((int)ckpts_.size() >= o_.max_ckpts) throw std::runtime_error("stub: too many checkpoints (leak?)");
    Ckpt c{next_id_++, cached_len()};
    ckpts_[c.id] = hist_;
    return c;
  }
  bool restore(const Ckpt& c) override {
    auto it = ckpts_.find(c.id);
    if (it == ckpts_.end()) return false;
    // A real engine shares append-only global KV: restoring is only valid if the history since the
    // checkpoint was not overwritten below its length. Check the same thing here.
    const auto& h = it->second;
    if (hist_.size() < h.size() || !std::equal(h.begin(), h.end(), hist_.begin()))
      throw std::runtime_error("stub: restore of a checkpoint whose global KV was overwritten");
    hist_ = h;
    return true;
  }
  void drop(const Ckpt& c) override { ckpts_.erase(c.id); }

  void prefill(std::span<const int32_t> toks, std::function<void(int64_t)> progress) override {
    if (toks.empty()) throw std::runtime_error("stub: empty prefill");
    hist_.insert(hist_.end(), toks.begin(), toks.end());
    prefilled_ += toks.size();
    int64_t done = 0;
    while (done < (int64_t)toks.size()) {
      int64_t step = std::min<int64_t>(2048, toks.size() - done);
      std::this_thread::sleep_for(std::chrono::microseconds((int64_t)(o_.prefill_us_per_token * step)));
      done += step;
      if (progress) progress(done);
    }
    has_logits_ = true;
  }

  void generate(const SamplingParams&, int max_tokens, const std::vector<int32_t>& stops,
                const std::function<bool(int32_t)>& on_token) override {
    if (!has_logits_) throw std::runtime_error("stub: generate without prefill");
    std::vector<int32_t> out = script();
    int emitted = 0;
    for (int32_t id : out) {
      if (emitted >= max_tokens) break;
      if (o_.decode_ms_per_token > 0)
        std::this_thread::sleep_for(std::chrono::microseconds((int64_t)(o_.decode_ms_per_token * 1000)));
      emitted++;
      bool more = on_token(id);
      bool stop = std::find(stops.begin(), stops.end(), id) != stops.end();
      if (stop || !more || emitted >= max_tokens) {
        // all emitted tokens except the last are in the cache
        hist_.insert(hist_.end(), out.begin(), out.begin() + emitted - 1);
        has_logits_ = false;
        return;
      }
    }
    hist_.insert(hist_.end(), out.begin(), out.begin() + std::max(0, emitted - 1));
    has_logits_ = false;
  }

  std::string name() const override { return "stub"; }

 private:
  std::vector<int32_t> script() {
    // Find the last user message text and whether tools are declared.
    std::string text = tok_.decode(hist_);
    bool tools = text.find("<|message_system|>tool_declare<|content_xml|>") != std::string::npos;
    std::string last;
    size_t u = text.rfind("<|message_user|>");
    if (u != std::string::npos) {
      size_t b = text.find("<|content_text|>", u);
      size_t e = text.find("<|end_message|>", u);
      if (b != std::string::npos && e != std::string::npos && b < e) last = text.substr(b + 16, e - b - 16);
    }
    std::string tool_name;
    if (tools) {
      size_t a = text.find("\"name\":\"", text.find("tool_declare"));
      if (a != std::string::npos) tool_name = text.substr(a + 8, text.find('"', a + 8) - a - 8);
    }
    n_++;
    if (o_.reply_tokens > 0) {
      static const char* para =
          "The cache keeps every visit's keys and values, so a growing prompt only pays for its new tail. "
          "Ünïcödé, 日本語, emoji 🎉 and code `for (int i = 0; i < n; i++) sum += a[i];` stream through. ";
      std::vector<int32_t> body = tok_.encode(para);
      std::vector<int32_t> out = tok_.encode("<|message_model|><|content_thinking|>");
      int half = o_.reply_tokens / 2;
      for (int k = 0; k < half; k++) out.push_back(body[k % body.size()]);
      for (int32_t id : tok_.encode("<|end_message|><|message_model|><|content_text|>")) out.push_back(id);
      for (int k = 0; k < o_.reply_tokens - half; k++) out.push_back(body[k % body.size()]);
      for (int32_t id : tok_.encode("<|end_message|><|content_model_end_sampling|>")) out.push_back(id);
      return out;
    }
    std::string s;
    bool after_tool = text.size() >= 16 && text.rfind("<|message_tool|>") != std::string::npos &&
                      text.rfind("<|message_tool|>") > (u == std::string::npos ? 0 : u);
    if (tools && !tool_name.empty() && last.find("TOOL") != std::string::npos && !after_tool) {
      s = "<|message_model|><|content_thinking|>The user wants a tool call. I will call " + tool_name +
          ".<|end_message|><|message_model|>" + tool_name + "<|content_invoke_tool_json|>{\"name\": \"" + tool_name +
          "\", \"args\": {\"command\": \"ls -la\", \"path\": \"/tmp/x y\", \"n\": 3}}<|end_message|>"
          "<|content_model_end_sampling|>";
    } else {
      s = "<|message_model|><|content_thinking|>Stub thinking about it.\n<|end_message|><|message_model|>"
          "<|content_text|>Stub reply #" + std::to_string(n_) + " to: " + last.substr(0, 40) +
          " — ünïcödé 🎉 done.<|end_message|><|content_model_end_sampling|>";
    }
    return tok_.encode(s);
  }

  const Tokenizer& tok_;
  StubOptions o_;
  std::vector<int32_t> hist_;
  std::map<int64_t, std::vector<int32_t>> ckpts_;
  int64_t next_id_ = 1;
  int64_t prefilled_ = 0;
  bool has_logits_ = false;
  int n_ = 0;
};

}  // namespace

std::unique_ptr<EngineIface> make_stub_engine(const Tokenizer& tok, const StubOptions& o) {
  return std::make_unique<StubEngine>(tok, o);
}

}  // namespace ink
