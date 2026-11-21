// Dense projections through MetalBackend::linear for T rows (model-free): the decode-shaped attention projections
// of 48 visits (2.58 GB bf16) and the output head (0.92 GB, f32 x). Owner: D.
//   bench_dense T...          INK_MR_MAX=64 sends 17..64 rows through passes of the 16-row matvec instead of the GEMM
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "gpu/context.h"
#include "model/backend.h"

using namespace ink;

int main(int argc, char** argv) {
  auto be = make_metal_backend();
  const int64_t shapes[][2] = {{3072, 2304}, {3072, 2304}, {2304, 3072}, {1024, 2304}, {1024, 2304}, {384, 2304}};
  std::vector<Tensor> ws;
  size_t bytes = 0;
  for (int v = 0; v < 48; v++)
    for (auto& s : shapes) {
      ws.push_back(be->alloc(DType::BF16, {s[0], s[1]}, "w"));
      bytes += ws.back().nbytes();
    }
  const int64_t V = 200058, D = 2304;
  Tensor head = be->alloc(DType::BF16, {V, D}, "head");
  for (int i = 1; i < argc; i++) {
    const int64_t T = atoll(argv[i]);
    Tensor x23 = be->alloc(DType::BF16, {T, 2304}, "x"), x30 = be->alloc(DType::BF16, {T, 3072}, "x");
    Tensor y = be->alloc(DType::BF16, {T, 3072}, "y");
    Tensor xf = be->alloc(DType::F32, {T, D}, "xf"), lg = be->alloc(DType::F32, {T, V}, "lg");
    std::vector<double> t, th;
    for (int r = 0; r < 6; r++) {
      for (auto& w : ws) be->linear(w.dim(1) == 2304 ? x23 : x30, w, y);
      be->sync();
      t.push_back(gpu::Context::get().last_gpu_ms());
      be->linear(xf, head, lg);
      be->sync();
      th.push_back(gpu::Context::get().last_gpu_ms());
    }
    std::sort(t.begin(), t.end());
    std::sort(th.begin(), th.end());
    printf("dense T=%-3lld projections %.2f GB: %8.3f ms   head 0.92 GB: %7.3f ms\n", (long long)T, bytes / 1e9, t[3], th[3]);
    for (Tensor* p : {&x23, &x30, &y, &xf, &lg}) be->free(*p);
  }
}
