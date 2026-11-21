// Prefill GEMM throughput through MetalBackend::linear. Owner: B. Usage: bench_gemm [T]
#include <algorithm>
#include <cstdio>
#include <vector>

#include "gpu/context.h"
#include "model/backend.h"

using namespace ink;

int main(int argc, char** argv) {
  int64_t T = argc > 1 ? atoll(argv[1]) : 2048;
  auto be = make_metal_backend();
  const int64_t shapes[][2] = {{3072, 2304}, {2304, 3072}, {1024, 2304}};
  for (auto& s : shapes) {
    Tensor w = be->alloc(DType::BF16, {s[0], s[1]}, "w"), x = be->alloc(DType::BF16, {T, s[1]}, "x"),
           o = be->alloc(DType::BF16, {T, s[0]}, "o");
    std::vector<double> ms;
    for (int r = 0; r < 6; r++) {
      for (int i = 0; i < 10; i++) be->linear(x, w, o);
      be->sync();
      ms.push_back(gpu::Context::get().last_gpu_ms() / 10);
    }
    std::sort(ms.begin(), ms.end());
    double fl = 2.0 * T * s[0] * s[1];
    printf("gemm T=%lld N=%lld K=%lld: %.3f ms -> %.2f TFLOPS\n", (long long)T, (long long)s[0], (long long)s[1],
           ms[3], fl / (ms[3] / 1e3) / 1e12);
  }
}
