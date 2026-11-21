#include "output_parser.h"

#include <atomic>
#include <chrono>

namespace ink {

static std::string new_call_id() {
  static std::atomic<uint64_t> n{0};
  auto t = std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::system_clock::now().time_since_epoch()).count();
  return "call_" + std::to_string(t) + "_" + std::to_string(n++);
}

bool parse_inkling_tool_call(const std::string& body, std::string& name, std::string& arguments) {
  json::Value call;
  try {
    call = json::parse(body);  // json.loads(text.strip()); our parser skips surrounding whitespace
  } catch (const json::ParseError&) {
    return false;
  }
  const json::Value* n = call.find("name");
  if (!call.is_object() || !n || !n->is_string()) return false;
  const json::Value* args = call.find("args");
  if (!args) args = call.find("arguments");
  json::Value empty = json::Value::object();
  if (!args) args = &empty;
  if (!args->is_object()) return false;
  name = n->s;
  arguments = json::compact(*args);
  return true;
}

OutputParser::OutputParser(const Tokenizer& tok) : tok_(tok), dec_(tok) {
  t_message_model_ = tok.special("<|message_model|>");
  t_thinking_ = tok.special("<|content_thinking|>");
  t_text_ = tok.special("<|content_text|>");
  t_tool_json_ = tok.special("<|content_invoke_tool_json|>");
  t_tool_text_ = tok.special("<|content_invoke_tool_text|>");
  t_end_message_ = tok.special("<|end_message|>");
  t_end_sampling_ = tok.special("<|content_model_end_sampling|>");
}

std::vector<int32_t> OutputParser::stop_ids(const Tokenizer& tok) {
  std::vector<int32_t> s;
  for (const char* t : {"<|content_model_end_sampling|>", "<|message_user|>", "<|message_system|>",
                        "<|message_tool|>", "<|endoftext|>"}) {
    int32_t id = tok.special(t);
    if (id >= 0) s.push_back(id);
  }
  return s;
}

void OutputParser::route(Delta& d, const std::string& s) {
  if (s.empty()) return;
  switch (mode_) {
    case Mode::Thinking: d.reasoning += s; reasoning_all_ += s; break;
    case Mode::ToolJson:
    case Mode::ToolText: tool_buf_ += s; break;
    case Mode::Header: header_ += s; break;
    case Mode::Idle:  // text outside any message: treat as content
    case Mode::Text: d.content += s; content_all_ += s; break;
  }
}

void OutputParser::close_message(Delta& d) {
  route(d, dec_.flush());
  if (mode_ == Mode::ToolJson || mode_ == Mode::ToolText) {
    ToolCallOut c;
    if (parse_inkling_tool_call(tool_buf_, c.name, c.arguments)) {
      c.id = new_call_id();
      d.tool_calls.push_back(c);
      calls_.push_back(c);
      n_calls_++;
    } else if (mode_ == Mode::ToolText && !header_.empty()) {
      // <|content_invoke_tool_text|>: body is not JSON; pass it as {"input": body} to the named tool.
      json::Value a = json::Value::object();
      a.set("input", tool_buf_);
      c.id = new_call_id();
      c.name = header_;
      c.arguments = json::compact(a);
      d.tool_calls.push_back(c);
      calls_.push_back(c);
      n_calls_++;
    } else {
      // Not a valid call: keep the text rather than lose it.
      std::string s = tool_buf_;
      d.content += s;
      content_all_ += s;
    }
    tool_buf_.clear();
  }
  mode_ = Mode::Idle;
}

Delta OutputParser::push(int32_t id) {
  Delta d;
  if (done_) return d;
  if (tok_.is_special(id)) {
    route(d, dec_.flush());
    if (id == t_message_model_) {
      if (mode_ != Mode::Idle && mode_ != Mode::Header) close_message(d);
      mode_ = Mode::Header;
      header_.clear();
    } else if (id == t_thinking_) {
      if (mode_ == Mode::ToolJson || mode_ == Mode::ToolText) close_message(d);
      mode_ = Mode::Thinking;
    } else if (id == t_text_) {
      if (mode_ == Mode::ToolJson || mode_ == Mode::ToolText) close_message(d);
      mode_ = Mode::Text;
    } else if (id == t_tool_json_ || id == t_tool_text_) {
      if (mode_ == Mode::ToolJson || mode_ == Mode::ToolText) close_message(d);
      mode_ = id == t_tool_json_ ? Mode::ToolJson : Mode::ToolText;
      tool_buf_.clear();
    } else if (id == t_end_message_) {
      close_message(d);
    } else if (id == t_end_sampling_) {
      close_message(d);
      done_ = true;
    } else {
      // another speaker's role token, endoftext, or an unused token: end of the turn
      close_message(d);
      done_ = true;
    }
    return d;
  }
  route(d, dec_.push(id));
  return d;
}

Delta OutputParser::finish() {
  Delta d;
  if (done_) return d;
  if (mode_ == Mode::ToolJson || mode_ == Mode::ToolText) {
    close_message(d);  // truncated call: emitted only if its JSON happens to be complete
  } else {
    route(d, dec_.flush());
    mode_ = Mode::Idle;
  }
  done_ = true;
  return d;
}

}  // namespace ink
