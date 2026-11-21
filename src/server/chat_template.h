// Inkling 2 chat_template.jinja, rendered in C++. This is a hand translation of that one template (not a
// Jinja engine); tests/server/template_check.py diffs it against transformers' apply_chat_template.
// Owner: C.
#pragma once
#include <stdexcept>
#include <string>
#include <vector>

#include "json.h"

namespace ink {

struct TemplateError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct RenderedPrompt {
  std::string text;
  // Byte ranges of each assistant message in `text` (what {% generation %} wraps), in order.
  struct Span {
    size_t begin, end;
    size_t msg_index;
  };
  std::vector<Span> assistant_spans;
};

// messages: array of OpenAI-style messages. tool_calls[].function.arguments must already be objects
// (see canonicalize_messages). tools: array or null. reasoning_effort: null (template default 0.9),
// a string from the effort map, or a number in [0, 0.99].
RenderedPrompt render_chat(const json::Value& messages, const json::Value& tools,
                           const json::Value& reasoning_effort);

// OpenAI clients send tool_calls[].function.arguments as a JSON string; the template wants an object.
// Parses those strings in place (order preserved); invalid JSON becomes {} as in mlx_vlm.
void canonicalize_messages(json::Value& messages);

}  // namespace ink
