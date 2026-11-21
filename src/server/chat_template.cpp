#include "chat_template.h"

#include <algorithm>
#include <cctype>

namespace ink {
using pyjson::Kind;
using pyjson::Value;

namespace {

const pyjson::DumpOptions kSortedCompact{",", ":", true, false};
const pyjson::DumpOptions kCompact{",", ":", false, false};

// Jinja `x is defined` on a dict attribute: present (even if null).
const Value* attr(const Value& v, std::string_view key) { return v.find(key); }
bool defined_truthy(const Value& v, std::string_view key) {
  const Value* x = v.find(key);
  return x && x->truthy();
}

std::string effort_text(const Value& eff_in) {
  static const std::pair<const char*, double> kMap[] = {
      {"none", 0.0}, {"minimal", 0.1}, {"low", 0.2}, {"medium", 0.7},
      {"high", 0.9}, {"xhigh", 0.99},  {"max", 0.99}};
  double num;
  if (eff_in.is_null()) {
    num = 0.9;
  } else if (eff_in.is_string()) {
    // Jinja `trim` strips Python whitespace.
    std::string key = eff_in.s;
    auto issp = [](unsigned char c) { return std::isspace(c) != 0; };
    while (!key.empty() && issp(key.back())) key.pop_back();
    size_t a = 0;
    while (a < key.size() && issp(key[a])) a++;
    key.erase(0, a);
    bool found = false;
    for (auto& [k, v] : kMap)
      if (key == k) { num = v; found = true; }
    if (!found) throw TemplateError("Unknown reasoning_effort: " + eff_in.s);
  } else if (eff_in.is_number() || eff_in.is_bool()) {
    num = eff_in.as_double();
  } else {
    throw TemplateError("reasoning_effort must be a string or a number");
  }
  if (num < 0.0 || num > 0.99) throw TemplateError("reasoning_effort must be in [0.0, 0.99]");
  std::string out = "<|message_system|><|content_text|>Thinking effort level: ";
  out += num == 0.0 ? std::string("0") : pyjson::py_float(num);
  out += "<|end_message|>";
  return out;
}

const char* role_token(const std::string& role) {
  if (role == "user") return "<|message_user|>";
  if (role == "assistant") return "<|message_model|>";
  if (role == "system") return "<|message_system|>";
  if (role == "tool") return "<|message_tool|>";
  return nullptr;
}

}  // namespace

void canonicalize_messages(Value& messages) {
  if (!messages.is_array()) return;
  for (auto& m : messages.arr) {
    Value* tcs = m.find("tool_calls");
    if (!tcs || !tcs->is_array()) continue;
    for (auto& tc : tcs->arr) {
      Value* fn = tc.find("function");
      if (!fn) continue;
      Value* args = fn->find("arguments");
      if (!args || !args->is_string()) continue;
      if (args->s.empty()) { *args = Value::object(); continue; }
      // Same as mlx_vlm request_normalization: invalid JSON becomes {}.
      try {
        *args = pyjson::parse(args->s);
      } catch (const pyjson::ParseError&) {
        *args = Value::object();
      }
    }
  }
}

RenderedPrompt render_chat(const Value& messages, const Value& tools, const Value& reasoning_effort) {
  RenderedPrompt rp;
  std::string& out = rp.text;
  if (!messages.is_array()) throw TemplateError("messages must be a list");
  if (messages.arr.empty())  // transformers' apply_chat_template refuses this before rendering
    throw TemplateError("Cannot apply chat template to an empty conversation. Provide at least one message.");

  if (tools.truthy()) {
    if (!tools.is_array()) throw TemplateError("tools must be a list");
    Value specs = Value::array();
    for (const Value& tool : tools.arr) {
      const Value* fnp = attr(tool, "function");
      const Value& fn = fnp ? *fnp : tool;
      const Value* name = attr(fn, "name");
      if (!name) throw TemplateError("tool has no name");
      Value spec = Value::object();
      spec.set("description", defined_truthy(fn, "description") ? fn["description"] : Value(""));
      spec.set("name", *name);
      spec.set("parameters", defined_truthy(fn, "parameters") ? fn["parameters"] : Value::object());
      spec.set("type", defined_truthy(tool, "type") ? tool["type"] : Value("function"));
      specs.arr.push_back(std::move(spec));
    }
    out += "<|message_system|>tool_declare<|content_xml|>";
    pyjson::dump_to(out, specs, kSortedCompact);
    out += "<|end_message|>";
  }

  bool effort_emitted = false;
  for (size_t mi = 0; mi < messages.arr.size(); mi++) {
    const Value& message = messages.arr[mi];
    const Value* rolep = attr(message, "role");
    std::string role = rolep && rolep->is_string() ? rolep->s : "";
    const char* rt = role_token(role);
    if (!rt) throw TemplateError("Unknown message role: " + (rolep ? pyjson::py_str(*rolep) : std::string("")));
    if (!effort_emitted && role != "system") {
      out += effort_text(reasoning_effort);
      effort_emitted = true;
    }
    const bool is_asst = role == "assistant";

    std::string header = rt;
    if (role == "tool") {
      if (defined_truthy(message, "name")) {
        header += pyjson::py_str(message["name"]);
      } else if (defined_truthy(message, "tool_call_id")) {
        const Value& want = message["tool_call_id"];
        for (const Value& prev : messages.arr) {
          if (prev.str_or("role") != "assistant" || !defined_truthy(prev, "tool_calls")) continue;
          const Value& tcl = prev["tool_calls"];
          if (!tcl.is_array()) continue;
          for (const Value& tc : tcl.arr) {
            const Value* id = attr(tc, "id");
            const Value* fn = attr(tc, "function");
            const Value* nm = fn ? attr(*fn, "name") : nullptr;
            // == between JSON values; ids are strings in practice
            if (id && nm && pyjson::compact(*id) == pyjson::compact(want)) header += pyjson::py_str(*nm);
          }
        }
      }
    } else if (is_asst && defined_truthy(message, "name")) {
      header += pyjson::py_str(message["name"]);
    }

    size_t span_begin = out.size();
    const Value* content = attr(message, "content");
    const bool has_tc = defined_truthy(message, "tool_calls");
    const bool has_rc = defined_truthy(message, "reasoning_content");

    if (is_asst && has_rc) {
      out += header;
      out += "<|content_thinking|>";
      out += pyjson::py_str(message["reasoning_content"]);
      out += "<|end_message|>";
    }

    if (content && content->is_string() && (content->truthy() || !(is_asst && (has_tc || has_rc)))) {
      out += header;
      out += "<|content_text|>";
      out += content->s;
      out += "<|end_message|>";
    } else if (content && content->truthy()) {
      if (!content->is_array()) throw TemplateError("message content must be a string or a list of parts");
      for (const Value& part : content->arr) {
        if (part.is_string()) {
          out += header; out += "<|content_text|>"; out += part.s; out += "<|end_message|>";
          continue;
        }
        const Value* type = attr(part, "type");
        std::string ty = type && type->is_string() ? type->s : "";
        if (!type || ty == "text" || ty == "input_text") {
          const Value* tx = attr(part, "text");
          out += header; out += "<|content_text|>";
          if (tx && tx->is_string()) out += tx->s;
          out += "<|end_message|>";
        } else if (ty == "image" || ty == "input_image" || ty == "image_url") {
          if (role != "user" && role != "tool")
            throw TemplateError("Image content is only supported for user and tool messages");
          throw TemplateError("image input is not supported by this server (no vision tower)");
        } else if (ty == "audio" || ty == "input_audio" || ty == "audio_url") {
          throw TemplateError("Inkling 2 does not accept audio input");
        } else {
          throw TemplateError("Unsupported content part type: " + (type ? pyjson::py_str(*type) : std::string()));
        }
      }
    }

    if (is_asst && has_tc) {
      const Value& tcl = message["tool_calls"];
      if (!tcl.is_array()) throw TemplateError("tool_calls must be a list");
      for (const Value& tc : tcl.arr) {
        const Value* fnp = attr(tc, "function");
        const Value* nm = fnp ? attr(*fnp, "name") : nullptr;
        if (!nm || !nm->is_string()) throw TemplateError("tool call function name must be a string");
        if (defined_truthy(message, "name") && pyjson::py_str(message["name"]) != nm->s)
          throw TemplateError("assistant name " + pyjson::py_str(message["name"]) + " does not match its tool call " +
                              nm->s);
        const Value* ap = attr(*fnp, "arguments");
        Value empty = Value::object();
        const Value& args = ap && ap->truthy() ? *ap : empty;
        if (args.is_string())
          throw TemplateError("tool call arguments must be a parsed object, not a JSON string; canonicalize upstream");
        if (!args.is_object()) throw TemplateError("tool call arguments must be an object");
        out += "<|message_model|>";
        out += nm->s;
        out += "<|content_invoke_tool_json|>";
        out += "{\"name\":";
        pyjson::quote_to(out, nm->s);
        out += ",\"args\":{";
        std::vector<const pyjson::Member*> ms;
        for (auto& m : args.obj) ms.push_back(&m);
        std::stable_sort(ms.begin(), ms.end(),
                         [](const pyjson::Member* a, const pyjson::Member* b) { return a->first < b->first; });
        for (size_t k = 0; k < ms.size(); k++) {
          pyjson::quote_to(out, ms[k]->first);
          out += ':';
          pyjson::dump_to(out, ms[k]->second, kCompact);
          if (k + 1 < ms.size()) out += ',';
        }
        out += "}}<|end_message|>";
      }
    }

    if (is_asst) {
      out += "<|content_model_end_sampling|>";
      rp.assistant_spans.push_back({span_begin, out.size(), mi});
    }
  }
  if (!effort_emitted) out += effort_text(reasoning_effort);
  return rp;
}

}  // namespace ink
