// Bandwidth of the bf16 matvec over the decode-shaped attention projections of all 48 visits (2.7 GB). Owner: B.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "gpu/context.h"

using namespace ink::gpu;

struct MatvecShape {
  uint32_t N, K, T, x_stride, y_stride;
};

static uint16_t f2bf(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  u += 0x7fff + ((u >> 16) & 1);
  return uint16_t(u >> 16);
}
static float bf2f(uint16_t b) {
  uint32_t u = uint32_t(b) << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

int main(int argc, char** argv) {
  int T = argc > 1 ? atoi(argv[1]) : 1;
  auto& g = Context::get();
  printf("device %s, working set %.1f GB\n", g.device_name().c_str(), g.recommended_working_set() / 1e9);
  struct M {
    uint32_t N, K;
    Buf w;
  };
  std::vector<M> mats;
  const uint32_t shapes[][2] = {{3072, 2304}, {3072, 2304}, {2304, 3072}, {1024, 2304}, {1024, 2304}, {384, 2304}};
  std::mt19937 rng(1);
  std::normal_distribution<float> nd(0, 0.02f);
  for (int v = 0; v < 48; v++)
    for (auto& s : shapes) {
      M m{s[0], s[1], g.alloc(size_t(s[0]) * s[1] * 2, "bench-w")};
      auto* p = (uint16_t*)m.w.host;
      size_t n = size_t(s[0]) * s[1];
      if (v == 0)
        for (size_t i = 0; i < n; i++) p[i] = f2bf(nd(rng));
      else
        memcpy(p, mats[&s - shapes].w.host, n * 2);
      mats.push_back(m);
    }
  size_t total = 0;
  for (auto& m : mats) total += m.w.size;
  Buf x = g.alloc(3072 * 2 * 16, "bench-x");
  Buf y = g.alloc(3072 * 2 * 16, "bench-y");
  auto* xp = (uint16_t*)x.host;
  for (int i = 0; i < 3072 * 16; i++) xp[i] = f2bf(nd(rng) * 50);

  int R = argc > 2 ? atoi(argv[2]) : 4, S = argc > 3 ? atoi(argv[3]) : 2;
  bool skinny = T > 4;
  std::string name = skinny ? (getenv("SK1") ? "gemm_skinny_bf16" : "gemm_skinny2")
                            : "matvec_bf16_bf16_bf16_t" + std::to_string(T == 1 ? 1 : 4) + "_r" + std::to_string(R) +
                                  "_s" + std::to_string(S);
  Pipeline p = g.pipeline(name);
  struct SkinnyParams {
    uint32_t N, K, T, dt_out;
  };
  auto run = [&] {
    for (auto& m : mats) {
      g.set_pipeline(p);
      g.set_buffer(0, m.w);
      g.set_buffer(1, x);
      g.set_buffer(2, y);
      if (skinny) {
        SkinnyParams s{m.N, m.K, (uint32_t)T, 1};
        g.set_value(3, s);
        if (getenv("SK1")) g.dispatch({(m.N + 31) / 32, 1, 1}, {128, 1, 1});
        else g.dispatch({(m.N + 15) / 16, 1, 1}, {64, 1, 1});
      } else {
        MatvecShape s{m.N, m.K, (uint32_t)T, 3072, 3072};
        g.set_value(3, s);
        g.dispatch({(m.N + R * S - 1) / (R * S), 1, 1}, {32u * S, 1, 1});
      }
    }
    g.commit_and_wait();
    return g.last_gpu_ms();
  };
  run();
  std::vector<double> ms;
  for (int i = 0; i < 10; i++) ms.push_back(run());
  std::sort(ms.begin(), ms.end());
  printf("%s: %zu matrices, %.2f GB, median %.3f ms -> %.1f GB/s (min %.3f ms), T=%d\n", name.c_str(), mats.size(),
         total / 1e9, ms[5], total / 1e9 / (ms[5] / 1e3), ms[0], T);

  // correctness on the last matrix, row 0..N-1, t = 0
  auto& m = mats.back();
  MatvecShape s{m.N, m.K, 1, 3072, 3072};
  g.set_pipeline(g.pipeline("matvec_bf16_bf16_f32_t1_r4_s2"));
  g.set_buffer(0, m.w);
  g.set_buffer(1, x);
  g.set_buffer(2, y);
  g.set_value(3, s);
  g.dispatch({(m.N + 7) / 8, 1, 1}, {64, 1, 1});
  g.commit_and_wait();
  double maxerr = 0;
  auto* w = (uint16_t*)m.w.host;
  for (uint32_t n = 0; n < m.N; n++) {
    double ref = 0;
    for (uint32_t k = 0; k < m.K; k++) ref += double(bf2f(w[size_t(n) * m.K + k])) * bf2f(xp[k]);
    maxerr = std::max(maxerr, std::fabs(ref - ((float*)y.host)[n]) / (std::fabs(ref) + 1e-3));
  }
  printf("max rel err vs fp64: %.2e\n", maxerr);
}
