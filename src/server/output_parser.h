// Streaming parser of Inkling's generated token stream into OpenAI deltas. Works on token ids, so special
// tokens are never confused with text. Format (see chat_template.jinja and the reference dumps):
//   <|message_model|>[NAME]<|content_thinking|>...<|end_message|>        -> reasoning_content
//   <|message_model|>[NAME]<|content_text|>...<|end_message|>            -> content
//   <|message_model|>NAME<|content_invoke_tool_json|>{"name":N,"args":{...}}<|end_message|>  -> tool_calls
//   <|content_model_end_sampling|>                                       -> end of the turn
// Port of mlx_vlm/tools/parsers/inkling.py (name from the JSON, arguments from "args" or "arguments").
// Owner: C.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "json.h"
#include "tokenizer/tokenizer.h"

namespace ink {

struct ToolCallOut {
  std::string id;         // call_<...>
  std::string name;
  std::string arguments;  // JSON text (compact, model's key order)
};

struct Delta {
  std::string reasoning;
  std::string content;
  std::vector<ToolCallOut> tool_calls;  // complete calls (emitted when their message ends)
  bool empty() const { return reasoning.empty() && content.empty() && tool_calls.empty(); }
};

class OutputParser {
 public:
  explicit OutputParser(const Tokenizer& tok);

  // Feed one generated token; returns what to stream now. Sets done() on a stop token.
  Delta push(int32_t id);
  // End of generation without a stop token (length limit, cancel): flush pending text.
  Delta finish();

  bool done() const { return done_; }
  bool saw_tool_call() const { return n_calls_ > 0; }

  // Accumulated results.
  const std::string& reasoning() const { return reasoning_all_; }
  const std::string& content() const { return content_all_; }
  const std::vector<ToolCallOut>& tool_calls() const { return calls_; }

  // Stop ids for Engine::generate: end_sampling plus role tokens of other speakers and endoftext.
  static std::vector<int32_t> stop_ids(const Tokenizer& tok);

 private:
  enum class Mode { Idle, Header, Thinking, Text, ToolJson, ToolText };
  void close_message(Delta& d);
  void route(Delta& d, const std::string& s);

  const Tokenizer& tok_;
  StreamDecoder dec_;
  Mode mode_ = Mode::Idle;
  std::string header_;   // name written after <|message_model|>
  std::string tool_buf_;
  bool done_ = false;
  int n_calls_ = 0;
  std::string reasoning_all_, content_all_;
  std::vector<ToolCallOut> calls_;
  int32_t t_message_model_, t_thinking_, t_text_, t_tool_json_, t_tool_text_, t_end_message_, t_end_sampling_;
};

// Parse a complete tool-call body as inkling.py does. Returns false if it is not a valid call.
bool parse_inkling_tool_call(const std::string& body, std::string& name, std::string& arguments);

}  // namespace ink
