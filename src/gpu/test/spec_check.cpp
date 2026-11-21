// Greedy decode with DSpark on and off on the same prompt: token streams must be identical. Owner: B.
// Usage: spec_check IDS_FILE [n_tokens=200] [block=4] [model_dir]
// Prints tok/s for both and the acceptance rate. One process loads the model once per mode (sequentially).
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

struct Run {
  std::vector<int32_t> toks;
  double prefill_s = 0, gen_s = 0;
};

static Run run(const std::string& dir, const std::vector<int32_t>& ids, int n, bool mtp, int block) {
  EngineOptions o;
  o.model_dir = dir;
  o.max_context = 32768;
  o.use_mtp = mtp;
  o.draft_block = block;
  o.prefill_chunk = 512;
  Engine e(o);
  Run r;
  auto t0 = std::chrono::steady_clock::now();
  e.prefill(ids);
  auto t1 = std::chrono::steady_clock::now();
  SamplingParams p;  // greedy
  e.generate(p, n, {}, [&](int32_t t) {
    r.toks.push_back(t);
    return true;
  });
  auto t2 = std::chrono::steady_clock::now();
  r.prefill_s = std::chrono::duration<double>(t1 - t0).count();
  r.gen_s = std::chrono::duration<double>(t2 - t1).count();
  return r;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: spec_check IDS [n] [block] [model_dir]\n");
    return 2;
  }
  auto ids = read_i32(argv[1]);
  int n = argc > 2 ? atoi(argv[2]) : 200;
  int block = argc > 3 ? atoi(argv[3]) : 4;
  std::string dir = argc > 4 ? argv[4] : std::string(getenv("HOME")) + "/llm/mlx-models/Inkling-2-Mini-MLX-4-8bit";
  setenv("INK_SPEC_STATS", "1", 0);
  Run a = run(dir, ids, n, false, block);
  printf("plain : %zu tokens, prefill %zu tok %.2f s, decode %.2f s -> %.1f tok/s\n", a.toks.size(), ids.size(),
         a.prefill_s, a.gen_s, (a.toks.size() - 1) / a.gen_s);
  fflush(stdout);
  Run b = run(dir, ids, n, true, block);
  printf("dspark: %zu tokens, prefill %.2f s, decode %.2f s -> %.1f tok/s (block %d)\n", b.toks.size(), b.prefill_s,
         b.gen_s, (b.toks.size() - 1) / b.gen_s, block);
  size_t first = 0;
  while (first < a.toks.size() && first < b.toks.size() && a.toks[first] == b.toks[first]) first++;
  bool same = a.toks == b.toks;
  printf("token streams %s", same ? "IDENTICAL\n" : "DIFFER");
  if (!same) printf(" at %zu (plain %d, dspark %d)\n", first, first < a.toks.size() ? a.toks[first] : -1,
                    first < b.toks.size() ? b.toks[first] : -1);
  return same ? 0 : 1;
}
