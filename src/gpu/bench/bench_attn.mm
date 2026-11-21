// Attention speed through MetalBackend::attention at several context lengths. Owner: B.
#include <algorithm>
#include <cstdio>
#include <vector>

#include "gpu/context.h"
#include "model/backend.h"

using namespace ink;

int main() {
  auto be = make_metal_backend();
  const int64_t maxlen = 131072 + 2048;
  Tensor kc = be->alloc(DType::BF16, {8, maxlen, 128}, "k"), vc = be->alloc(DType::BF16, {8, maxlen, 128}, "v");
  Tensor rp = be->alloc(DType::BF16, {16, 1024}, "rp");
  struct Case { int64_t T, len; int window; };
  for (Case c : {Case{1, 1024, 0}, Case{1, 8192, 0}, Case{1, 32768, 0}, Case{1, 131072, 0}, Case{1, 131072, 1024},
                 Case{4, 32768, 0}, Case{2048, 0, 0}, Case{2048, 32768, 0}, Case{2048, 126976, 0},
                 Case{2048, 126976, 1024}}) {
    int64_t T = c.T;
    Tensor q = be->alloc(DType::BF16, {T, 24, 128}, "q"), kn = be->alloc(DType::BF16, {T, 8, 128}, "kn"),
           vn = be->alloc(DType::BF16, {T, 8, 128}, "vn"), r = be->alloc(DType::BF16, {T, 24, 16}, "r"),
           sk = be->alloc(DType::BF16, {T, 24}, "s"), o = be->alloc(DType::BF16, {T, 24 * 128}, "o");
    bool ring = c.window > 0;
    AttnArgs a{q, kn, vn, r, rp, sk, KvView{kc, vc, ring ? 1536 : maxlen, c.len, ring}, c.window};
    std::vector<double> ms;
    int reps = T == 1 ? 20 : 3;
    for (int rep = 0; rep < 5; rep++) {
      for (int i = 0; i < reps; i++) be->attention(a, o);
      be->sync();
      ms.push_back(gpu::Context::get().last_gpu_ms() / reps);
    }
    std::sort(ms.begin(), ms.end());
    double keys = ring ? std::min<double>(c.len + T, 1024) : c.len + T / 2.0;
    double flops = 4.0 * T * 24 * 128 * keys;
    double kvbytes = ring ? 1024.0 * 8 * 128 * 4 : double(c.len + T) * 8 * 128 * 4;
    printf("attn T=%-5lld len=%-7lld %s: %8.3f ms  (%.2f TFLOPS, KV %.0f GB/s)\n", (long long)T, (long long)c.len,
           ring ? "local " : "global", ms[2], flops / (ms[2] / 1e3) / 1e12, kvbytes / (ms[2] / 1e3) / 1e9);
    for (Tensor* t : {&q, &kn, &vn, &r, &sk, &o}) be->free(*t);
  }
}
