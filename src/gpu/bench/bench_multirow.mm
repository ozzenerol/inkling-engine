// One-pass matvec for 5..16 rows (kernels/multirow.metal) against passes of the 4-row matvec. Owner: D.
// Same matrices as bench_matvec (attention projections of 48 visits, 2.58 GB, bf16 x -> bf16 y), then the output
// head (200058 x 2304, f32 x -> f32 y). Every candidate is checked bit for bit against the 1-row matvec.
//   bench_multirow [T ...]      (default 5 8 12 16)
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "gpu/context.h"

using namespace ink::gpu;

struct MatvecShape {
  uint32_t N, K, T, x_stride, y_stride;
};
struct M {
  uint32_t N, K;
  Buf w;
};

static uint16_t f2bf(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  u += 0x7fff + ((u >> 16) & 1);
  return uint16_t(u >> 16);
}

static Context* gp;
static Buf x, y, yref;

// One forward of all matrices with `name` (TT rows per pass, R*S rows per threadgroup). Returns GPU ms.
static double run(const std::vector<M>& mats, const std::string& name, uint32_t T, uint32_t TT, uint32_t R,
                  uint32_t S, size_t xsz, size_t ysz, Buf& out) {
  auto& g = *gp;
  Pipeline p = g.pipeline(name);
  for (auto& m : mats)
    for (uint32_t t0 = 0; t0 < T; t0 += TT) {
      MatvecShape s{m.N, m.K, std::min(TT, T - t0), m.K, m.N};
      g.set_pipeline(p);
      g.set_buffer(0, m.w);
      g.set_buffer(1, x, size_t(t0) * m.K * xsz);
      g.set_buffer(2, out, size_t(t0) * m.N * ysz);
      g.set_value(3, s);
      g.dispatch({(m.N + R * S - 1) / (R * S), 1, 1}, {32u * S, 1, 1});
    }
  g.commit_and_wait();
  return g.last_gpu_ms();
}

static void bench(const char* what, const std::vector<M>& mats, const std::string& xn, const std::string& yn,
                  size_t xsz, size_t ysz, const std::vector<int>& Ts) {
  size_t total = 0;
  for (auto& m : mats) total += m.w.size;
  printf("== %s: %zu matrices, %.2f GB, x %s, y %s\n", what, mats.size(), total / 1e9, xn.c_str(), yn.c_str());
  const std::string io = xn + "_" + yn;
  struct Cand {
    std::string name;
    uint32_t TT, R, S;
  };
  for (int T : Ts) {
    std::vector<Cand> cs = {{"matvec_bf16_" + io + "_t4_r4_s2", 4, 4, 2}};
    const uint32_t rs[][2] = {{4, 2}, {2, 2}, {2, 4}, {1, 4}};
    for (uint32_t TT : {8u, 16u})
      for (auto& r : rs) {
        if (TT == 16 && T <= 8) continue;
        std::string n = "mr_matvec_" + io + "_t" + std::to_string(TT) + "_r" + std::to_string(r[0]) + "_s" +
                        std::to_string(r[1]);
        cs.push_back({n, TT, r[0], r[1]});
        if (TT == 16 && io == "bf16_bf16") cs.push_back({n + "_dyn", TT, r[0], r[1]});
      }
    // reference rows: the 1-row matvec, row by row, on the last matrix
    const M& last = mats.back();
    std::vector<M> one = {last};
    run(one, "matvec_bf16_" + io + "_t1_r4_s2", uint32_t(T), 1, 4, 2, xsz, ysz, yref);
    for (auto& c : cs) {
      run(mats, c.name, uint32_t(T), c.TT, c.R, c.S, xsz, ysz, y);
      std::vector<double> ms;
      for (int i = 0; i < 7; i++) ms.push_back(run(mats, c.name, uint32_t(T), c.TT, c.R, c.S, xsz, ysz, y));
      std::sort(ms.begin(), ms.end());
      memset(y.host, 0xee, size_t(T) * last.N * ysz);
      run(one, c.name, uint32_t(T), c.TT, c.R, c.S, xsz, ysz, y);
      bool same = memcmp(y.host, yref.host, size_t(T) * last.N * ysz) == 0;
      printf("T=%-2d %-36s median %7.3f ms (min %7.3f)  %6.1f GB/s per pass-equivalent  rows==t1: %s\n", T,
             c.name.c_str(), ms[3], ms[0], total / 1e9 / (ms[3] / 1e3), same ? "yes" : "NO");
    }
  }
}

int main(int argc, char** argv) {
  std::vector<int> Ts;
  for (int i = 1; i < argc; i++) Ts.push_back(atoi(argv[i]));
  if (Ts.empty()) Ts = {5, 8, 12, 16};
  auto& g = Context::get();
  gp = &g;
  std::mt19937 rng(1);
  std::normal_distribution<float> nd(0, 0.02f);
  const uint32_t V = 200058, D = 2304;
  x = g.alloc(size_t(3072) * 16 * 4, "bench-x");
  y = g.alloc(size_t(V) * 16 * 4, "bench-y");
  yref = g.alloc(size_t(V) * 16 * 4, "bench-yref");

  std::vector<M> mats;
  const uint32_t shapes[][2] = {{3072, 2304}, {3072, 2304}, {2304, 3072}, {1024, 2304}, {1024, 2304}, {384, 2304}};
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
  auto* xb = (uint16_t*)x.host;
  for (int i = 0; i < 3072 * 16; i++) xb[i] = f2bf(nd(rng) * 50);
  bench("attention projections", mats, "bf16", "bf16", 2, 2, Ts);
  for (auto& m : mats) g.release(m.w);

  std::vector<M> head = {M{V, D, g.alloc(size_t(V) * D * 2, "bench-head")}};
  auto* hp = (uint16_t*)head[0].w.host;
  for (size_t i = 0; i < size_t(V) * D; i++) hp[i] = uint16_t(0x3c00 + (i * 2654435761u >> 20) % 512) ^ uint16_t((i & 1) << 15);
  auto* xf = (float*)x.host;
  for (int i = 0; i < 3072 * 16; i++) xf[i] = nd(rng) * 50;
  bench("output head", head, "f32", "f32", 4, 4, Ts);
}
