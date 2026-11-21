// Reads a JSON array of cases {messages, tools, reasoning_effort} on stdin, prints a JSON array of
// {"ok":text} or {"err":message}. Used by template_check.py. Owner: C.
#include <iostream>
#include <iterator>
#include "../../src/server/chat_template.h"
using namespace ink;
int main() {
  std::string in((std::istreambuf_iterator<char>(std::cin)), {});
  pyjson::Value cases = pyjson::parse(in);
  pyjson::Value outs = pyjson::Value::array();
  for (auto& c : cases.arr) {
    pyjson::Value r = pyjson::Value::object();
    try {
      pyjson::Value msgs = c["messages"];
      canonicalize_messages(msgs);
      r.set("ok", render_chat(msgs, c["tools"], c["reasoning_effort"]).text);
    } catch (const std::exception& e) {
      r.set("err", e.what());
    }
    outs.arr.push_back(std::move(r));
  }
  std::cout << pyjson::compact(outs);
}
