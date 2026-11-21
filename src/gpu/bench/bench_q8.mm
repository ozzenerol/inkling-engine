// Bandwidth of the q8 affine matvec through MetalBackend::linear on decode shapes (model-free). Owner: B.
#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

#include "gpu/context.h"
#include "model/backend.h"

using namespace ink;

int main(int argc, char** argv) {
  int64_t T = argc > 1 ? atoll(argv[1]) : 1;
  auto be = make_metal_backend();
  if (!be->supports_q8()) return printf("q8 kernels disabled\n"), 0;
  const int64_t shapes[][2] = {{3072, 2304}, {3072, 2304}, {2304, 3072}, {1024, 2304}, {1024, 2304}, {384, 2304}};
  struct M { Tensor w, y; };
  std::vector<M> ms;
  size_t bytes = 0;
  std::mt19937 rng(3);
  for (int v = 0; v < 48; v++)
    for (auto& s : shapes) {
      Tensor w = be->alloc(DType::U32, {s[0], s[1] / 4}, "w"), sc = be->alloc(DType::BF16, {s[0], s[1] / 64}, "s"),
             bi = be->alloc(DType::BF16, {s[0], s[1] / 64}, "b");
      {  // random codes and scales: zero-filled buffers gave 795 GB/s in batch 5, not a real read test
        std::vector<uint32_t> rc(size_t(w.numel()));
        for (auto& c : rc) c = uint32_t(rng());
        be->write(w, rc.data(), rc.size() * 4);
        std::vector<uint16_t> rs(size_t(sc.numel()), 0x3A00), rb(size_t(bi.numel()), 0xBD00);
        be->write(sc, rs.data(), rs.size() * 2);
        be->write(bi, rb.data(), rb.size() * 2);
      }
      be->bind_affine(w, sc, bi);
      ms.push_back({w, be->alloc(DType::BF16, {T, s[0]}, "y")});
      bytes += w.nbytes() + sc.nbytes() + bi.nbytes();
    }
  Tensor x23 = be->alloc(DType::BF16, {T, 2304}, "x"), x30 = be->alloc(DType::BF16, {T, 3072}, "x");
  int interleave = getenv("INTERLEAVE") ? atoi(getenv("INTERLEAVE")) : 0;
  Tensor tmp = be->alloc(DType::BF16, {T, 2304}, "tmp");
  bool indep = getenv("INTERLEAVE_INDEP") != nullptr;
  std::vector<std::pair<Tensor, Tensor>> pool;
  size_t pool_i = 0;
  if (indep)
    for (int i = 0; i < 2048; i++)
      pool.push_back({be->alloc(DType::BF16, {1, 2304}, "pa"), be->alloc(DType::BF16, {1, 2304}, "pb")});
  std::vector<double> t;
  for (int r = 0; r < 8; r++) {
    for (auto& m : ms) {
      be->linear(m.w.dim(1) * 4 == 2304 ? x23 : x30, m.w, m.y);
      // INTERLEAVE=n: n tiny dependent dispatches after each matvec, as in a decode step (tests whether the gaps
      // between big reads, not the reads, make the in-model dense cost 3x this bench)
      for (int i = 0; i < interleave; i++) {
        if (indep) {  // buffers no other dispatch touches: no dependency on the matvecs or on each other
          size_t j = (pool_i++) % pool.size();
          be->rms_norm(pool[j].first, nullptr, 1e-6f, 2304, pool[j].second);
        } else {
          be->rms_norm(i % 2 ? tmp : x23, nullptr, 1e-6f, 2304, i % 2 ? x23 : tmp);
        }
      }
    }
    be->sync();
    t.push_back(gpu::Context::get().last_gpu_ms());
  }
  std::sort(t.begin(), t.end());
  printf("q8 matvec T=%lld interleave %d%s: %zu matrices, %.2f GB, median %.3f ms -> %.0f GB/s\n", (long long)T, interleave, indep ? " (independent)" : "", ms.size(),
         bytes / 1e9, t[4], bytes / 1e9 / (t[4] / 1e3));
}
