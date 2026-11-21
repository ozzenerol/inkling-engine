// stdin: JSON array of strings; stdout: JSON array of id arrays (encode with specials), then timing on stderr.
// Owner: C.
#include <chrono>
#include <iostream>
#include <iterator>
#include "../../src/server/pyjson.h"
#include "../../src/tokenizer/tokenizer.h"
using namespace ink;
int main(int argc, char** argv) {
  auto t0 = std::chrono::steady_clock::now();
  auto tok = Tokenizer::load(argc > 1 ? argv[1] : "/Users/ozzenerol/llm/mlx-models/Inkling-2-Mini-MLX-4-8bit");
  auto t1 = std::chrono::steady_clock::now();
  std::string in((std::istreambuf_iterator<char>(std::cin)), {});
  pyjson::Value texts = pyjson::parse(in);
  pyjson::Value outs = pyjson::Value::array();
  size_t ntok = 0, nbytes = 0;
  auto t2 = std::chrono::steady_clock::now();
  for (auto& t : texts.arr) {
    auto ids = tok->encode(t.s);
    ntok += ids.size(); nbytes += t.s.size();
    pyjson::Value a = pyjson::Value::array();
    for (int32_t id : ids) a.arr.emplace_back((int64_t)id);
    if (tok->decode(ids) != t.s) std::cerr << "roundtrip mismatch\n";
    outs.arr.push_back(std::move(a));
  }
  auto t3 = std::chrono::steady_clock::now();
  std::cout << pyjson::compact(outs);
  auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
  std::cerr << "load " << ms(t0, t1) << " ms, encode " << ntok << " tokens / " << nbytes << " bytes in " << ms(t2, t3)
            << " ms\n";
}
