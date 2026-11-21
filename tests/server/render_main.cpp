// Reads a JSON array of cases {messages, tools, reasoning_effort} on stdin, prints a JSON array of
// {"ok":text} or {"err":message}. Used by template_check.py. Owner: C.
#include <iostream>
#include <iterator>
#include "../../src/server/chat_template.h"
using namespace ink;
int main() {
  std::string in((std::istreambuf_iterator<char>(std::cin)), {});
  json::Value cases = json::parse(in);
  json::Value outs = json::Value::array();
  for (auto& c : cases.arr) {
    json::Value r = json::Value::object();
    try {
      json::Value msgs = c["messages"];
      canonicalize_messages(msgs);
      r.set("ok", render_chat(msgs, c["tools"], c["reasoning_effort"]).text);
    } catch (const std::exception& e) {
      r.set("err", e.what());
    }
    outs.arr.push_back(std::move(r));
  }
  std::cout << json::compact(outs);
}
