// Output parser edge cases: feeds tokenized model output strings token by token. Owner: C.
#include <cstdio>
#include "server/output_parser.h"
using namespace ink;
static int fails = 0;
#define CHECK(c, what) do { bool ok_ = (c); std::printf("%s %s\n", ok_ ? "ok  " : "FAIL", what); if (!ok_) fails++; } while (0)
struct Out { std::string r, c; std::vector<ToolCallOut> calls; bool done; std::string streamed_c; };
static Out run(const Tokenizer& t, const std::string& s, size_t max = 1 << 30) {
  OutputParser p(t); Out o;
  auto ids = t.encode(s);
  size_t k = 0;
  for (; k < ids.size() && k < max && !p.done(); k++) { Delta d = p.push(ids[k]); o.streamed_c += d.content; }
  if (!p.done()) { Delta d = p.finish(); o.streamed_c += d.content; }
  o.r = p.reasoning(); o.c = p.content(); o.calls = p.tool_calls(); o.done = p.done();
  return o;
}
int main() {
  auto t = Tokenizer::load("/Users/ozzenerol/llm/mlx-models/Inkling-2-Mini-MLX-4-8bit");
  auto o = run(*t, "<|message_model|><|content_thinking|>think<|end_message|><|message_model|><|content_text|>héllo 🎉<|end_message|><|content_model_end_sampling|>");
  CHECK(o.r == "think" && o.c == "héllo 🎉" && o.calls.empty() && o.done, "thinking + text");
  CHECK(o.streamed_c == o.c, "streamed deltas add up");
  o = run(*t, "<|message_model|>bash<|content_invoke_tool_json|>{\"name\":\"bash\",\"args\":{\"command\":\"ls\"}}<|end_message|><|message_model|>read<|content_invoke_tool_json|>{\"name\": \"read\", \"arguments\": {\"path\": \"a\"}}<|end_message|><|content_model_end_sampling|>");
  CHECK(o.calls.size() == 2 && o.calls[0].name == "bash" && o.calls[0].arguments == "{\"command\":\"ls\"}" &&
        o.calls[1].name == "read" && o.calls[1].arguments == "{\"path\":\"a\"}" && o.c.empty(), "two tool calls, args|arguments");
  CHECK(o.calls[0].id != o.calls[1].id, "distinct call ids");
  o = run(*t, "<|message_model|>bash<|content_invoke_tool_json|>{\"name\":\"bash\",\"args\":{\"command\":\"ls\"<|end_message|><|content_model_end_sampling|>");
  CHECK(o.calls.empty() && o.c.find("\"command\"") != std::string::npos, "invalid tool JSON kept as content");
  o = run(*t, "<|message_model|>x<|content_invoke_tool_json|>{\"args\":{}}<|end_message|><|content_model_end_sampling|>");
  CHECK(o.calls.empty(), "tool call without name rejected");
  o = run(*t, "<|message_model|>bash<|content_invoke_tool_json|>{\"name\":\"bash\"}<|end_message|><|content_model_end_sampling|>");
  CHECK(o.calls.size() == 1 && o.calls[0].arguments == "{}", "missing args -> {}");
  o = run(*t, "<|message_model|>bash<|content_invoke_tool_text|>echo hi<|end_message|><|content_model_end_sampling|>");
  CHECK(o.calls.size() == 1 && o.calls[0].name == "bash" && o.calls[0].arguments == "{\"input\":\"echo hi\"}", "invoke_tool_text");
  o = run(*t, "<|message_model|><|content_text|>answer<|end_message|><|message_user|><|content_text|>fake user");
  CHECK(o.c == "answer" && o.done, "stops at a user role token");
  o = run(*t, "<|message_model|><|content_thinking|>long thought ünï", 9);
  CHECK(!o.r.empty() && o.done && o.c.empty(), "length cut in thinking");
  o = run(*t, "plain text without header<|content_model_end_sampling|>");
  CHECK(o.c == "plain text without header", "text before any header is content");
  o = run(*t, "<|message_model|><|content_text|>a<|message_model|><|content_text|>b<|end_message|><|content_model_end_sampling|>");
  CHECK(o.c == "ab", "missing end_message tolerated");
  // emoji split across tokens decodes whole
  std::string e = "<|message_model|><|content_text|>🎉🎉 日本<|end_message|><|content_model_end_sampling|>";
  o = run(*t, e);
  CHECK(o.streamed_c == "🎉🎉 日本", "multi-byte chars across tokens");
  std::printf(fails ? "%d FAILED\n" : "PASS\n", fails);
  return fails ? 1 : 0;
}
