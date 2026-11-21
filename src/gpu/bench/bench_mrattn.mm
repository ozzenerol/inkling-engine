// Attention at verify sizes (T = 1..16) through MetalBackend::attention. Owner: D.
//   bench_mrattn                 multi-row kernel for T = 5..16
//   INK_MR_ATTN=0 bench_mrattn   previous path (attn_prefill) for T = 5..16
//   INK_MR_ABL=1|2|4|7 bench_mrattn 16   timing ablations of the 16-row kernel (no rel bias / no PV / no exp)
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "gpu/context.h"
#include "model/backend.h"

using namespace ink;

int main(int argc, char** argv) {
  auto be = make_metal_backend();
  const int64_t maxlen = 131072 + 2048;
  Tensor kc = be->alloc(DType::BF16, {8, maxlen, 128}, "k"), vc = be->alloc(DType::BF16, {8, maxlen, 128}, "v");
  Tensor rp = be->alloc(DType::BF16, {16, 1024}, "rp");
  std::vector<int64_t> Ts;
  for (int i = 1; i < argc; i++) Ts.push_back(atoll(argv[i]));
  if (Ts.empty()) Ts = {1, 4, 5, 8, 12, 16};
  struct Case { int64_t len; int window; };
  for (Case c : {Case{2048, 0}, Case{32768, 0}, Case{131072, 0}, Case{131072, 1024}})
    for (int64_t T : Ts) {
      Tensor q = be->alloc(DType::BF16, {T, 24, 128}, "q"), kn = be->alloc(DType::BF16, {T, 8, 128}, "kn"),
             vn = be->alloc(DType::BF16, {T, 8, 128}, "vn"), r = be->alloc(DType::BF16, {T, 24, 16}, "r"),
             sk = be->alloc(DType::BF16, {T, 24}, "s"), o = be->alloc(DType::BF16, {T, 24 * 128}, "o");
      bool ring = c.window > 0;
      AttnArgs a{q, kn, vn, r, rp, sk, KvView{kc, vc, ring ? 1536 : maxlen, c.len, ring}, c.window};
      std::vector<double> ms;
      const int reps = 10;
      for (int rep = 0; rep < 5; rep++) {
        for (int i = 0; i < reps; i++) be->attention(a, o);
        be->sync();
        ms.push_back(gpu::Context::get().last_gpu_ms() / reps);
      }
      std::sort(ms.begin(), ms.end());
      double keys = ring ? 1024 : double(c.len);
      double flops = 4.0 * T * 24 * 128 * keys;
      printf("attn T=%-3lld len=%-7lld %s: %8.3f ms  (%.2f TFLOPS)\n", (long long)T, (long long)c.len,
             ring ? "local " : "global", ms[2], flops / (ms[2] / 1e3) / 1e12);
      for (Tensor* t : {&q, &kn, &vn, &r, &sk, &o}) be->free(*t);
    }
}
