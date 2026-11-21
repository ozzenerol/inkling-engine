// One greedy (or sampled) generation with DSpark on or off; writes the token ids to OUT. Owner: B.
// Usage: spec_check plain|dspark IDS N BLOCK OUT [temperature] ; tools/spec_check.sh runs both and compares.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "engine/engine.h"

using namespace ink;

static std::vector<int32_t> read_i32(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::vector<int32_t> v;
  int32_t x;
  while (f.read(reinterpret_cast<char*>(&x), 4)) v.push_back(x);
  return v;
}

int main(int argc, char** argv) {
  if (argc < 6) {
    fprintf(stderr, "usage: spec_check plain|dspark IDS N BLOCK OUT [temperature]\n");
    return 2;
  }
  bool mtp = std::string(argv[1]) == "dspark";
  auto ids = read_i32(argv[2]);
  int n = atoi(argv[3]), block = atoi(argv[4]);
  std::string out = argv[5];
  float temp = argc > 6 ? float(atof(argv[6])) : 0.0f;
  EngineOptions o;
  o.model_dir = std::string(getenv("HOME")) + "/llm/mlx-models/Inkling-2-Mini-MLX-4-8bit";
  if (const char* d = getenv("INK_MODEL")) o.model_dir = d;
  o.max_context = std::max<int64_t>(65536, int64_t(ids.size()) + n + 256);  // long prompts (126k) fit too
  o.use_mtp = mtp;
  o.draft_block = block;
  o.prefill_chunk = 512;
  Engine e(o);
  std::vector<int32_t> toks;
  auto t0 = std::chrono::steady_clock::now();
  e.prefill(ids);
  auto t1 = std::chrono::steady_clock::now();
  SamplingParams p;
  p.temperature = temp;
  p.seed = 1;
  e.generate(p, n, {200006, 200000, 200002, 200003, 199999}, [&](int32_t t) {
    toks.push_back(t);
    return true;
  });
  auto t2 = std::chrono::steady_clock::now();
  double ps = std::chrono::duration<double>(t1 - t0).count(), gs = std::chrono::duration<double>(t2 - t1).count();
  printf("%-6s: prompt %zu tok, prefill %.2f s (%.0f tok/s); %zu tokens, decode %.2f s -> %.1f tok/s (block %d)\n",
         argv[1], ids.size(), ps, ids.size() / ps, toks.size(), gs, (toks.size() - 1) / gs, block);
  std::ofstream f(out, std::ios::binary);
  f.write(reinterpret_cast<const char*>(toks.data()), std::streamsize(toks.size() * 4));
  return 0;
}
