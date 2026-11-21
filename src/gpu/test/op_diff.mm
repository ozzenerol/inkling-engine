// Op-by-op diff of MetalBackend against CpuBackend on identical random inputs (and real expert weights).
// Owner: B. Usage: op_diff [model_dir]
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "core/bf16.h"
#include "core/safetensors.h"
#include "model/backend.h"

using namespace ink;

static std::mt19937 rng(42);

struct Pair {
  Tensor c, m;
};

static Backend* CPU;
static Backend* MTL;

static Pair make(DType dt, std::initializer_list<int64_t> shape, float sd = 1.0f, float mean = 0.0f) {
  Pair p{CPU->alloc(dt, shape, "t"), MTL->alloc(dt, shape, "t")};
  std::normal_distribution<float> nd(mean, sd);
  size_t n = size_t(p.c.numel());
  if (dt == DType::F32) {
    std::vector<float> v(n);
    for (auto& x : v) x = nd(rng);
    CPU->write(p.c, v.data(), n * 4);
    MTL->write(p.m, v.data(), n * 4);
  } else if (dt == DType::BF16) {
    std::vector<uint16_t> v(n);
    for (auto& x : v) x = f32_to_bf16(nd(rng));
    CPU->write(p.c, v.data(), n * 2);
    MTL->write(p.m, v.data(), n * 2);
  }
  return p;
}
static Pair zeros(DType dt, std::initializer_list<int64_t> shape) {
  return {CPU->alloc(dt, shape, "z"), MTL->alloc(dt, shape, "z")};
}
static void fill_i32(Pair& p, const std::vector<int32_t>& v) {
  CPU->write(p.c, v.data(), v.size() * 4);
  MTL->write(p.m, v.data(), v.size() * 4);
}

static std::vector<float> host_f32(Backend* b, const Tensor& t) {
  size_t n = size_t(t.numel());
  std::vector<float> out(n);
  if (t.dtype == DType::F32) b->read(t, out.data(), n * 4);
  else {
    std::vector<uint16_t> v(n);
    b->read(t, v.data(), n * 2);
    for (size_t i = 0; i < n; i++) out[i] = bf16_to_f32(v[i]);
  }
  return out;
}

static int fails = 0;
// bf16: count elements differing by more than `ulps` bf16 ulps; f32: max relative diff vs max |ref|.
static void compare(const char* name, Pair& p, double tol = 1e-5, int ulps = 1, double bad_frac = 0.001) {
  MTL->sync();
  auto a = host_f32(CPU, p.c), b = host_f32(MTL, p.m);
  double maxabs = 0, maxref = 0;
  size_t bad = 0, exact = 0, nan = 0;
  for (size_t i = 0; i < a.size(); i++) {
    if (std::isnan(b[i]) != std::isnan(a[i])) nan++;
    double d = std::fabs(double(a[i]) - b[i]);
    maxabs = std::max(maxabs, d);
    maxref = std::max(maxref, std::fabs(double(a[i])));
    if (a[i] == b[i]) exact++;
    if (p.c.dtype == DType::BF16) {
      double ulp = std::ldexp(1.0, std::ilogb(std::max(std::fabs(a[i]), 1e-30f)) - 7);
      if (d > ulps * ulp * 1.0001) bad++;
    }
  }
  double rel = maxabs / std::max(maxref, 1e-30);
  if (maxref == 0) {
    printf("%-28s FAIL  reference is all zeros\n", name);
    fails++;
    return;
  }
  bool ok = nan == 0 && (p.c.dtype == DType::BF16 ? bad <= a.size() * bad_frac : rel <= tol);
  if (!ok) fails++;
  printf("%-28s %s  n=%zu exact=%.2f%%  max|d|=%.3g (rel %.2e)%s  %s\n", name, ok ? "ok  " : "FAIL", a.size(),
         100.0 * exact / a.size(), maxabs, rel,
         p.c.dtype == DType::BF16 ? (" >" + std::to_string(ulps) + "ulp=" + std::to_string(bad)).c_str() : "",
         nan ? "NAN MISMATCH" : "");
}

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : std::string(getenv("HOME")) + "/llm/mlx-models/Inkling-2-Mini-MLX-4-8bit";
  auto cpu = make_cpu_backend();
  auto mtl = make_metal_backend();
  CPU = cpu.get();
  MTL = mtl.get();
  const int64_t D = 2304;

  {  // rms_norm
    auto x = make(DType::BF16, {5, D}, 3.0f);
    auto g = make(DType::BF16, {D}, 0.2f, 1.0f);
    auto o = zeros(DType::BF16, {5, D});
    CPU->rms_norm(x.c, &g.c, 1e-6f, D, o.c);
    MTL->rms_norm(x.m, &g.m, 1e-6f, D, o.m);
    compare("rms_norm bf16 gamma", o);
    auto of = zeros(DType::F32, {5, D});
    CPU->rms_norm(x.c, nullptr, 1e-6f, D, of.c);
    MTL->rms_norm(x.m, nullptr, 1e-6f, D, of.m);
    compare("rms_norm f32 nogamma", of);
  }
  for (int64_t T : {1, 3, 4, 5, 7, 8, 9, 16, 40}) {  // linear
    auto x = make(DType::BF16, {T, D});
    auto w = make(DType::BF16, {3072, D}, 0.02f);
    auto o = zeros(DType::BF16, {T, 3072});
    CPU->linear(x.c, w.c, o.c);
    MTL->linear(x.m, w.m, o.m);
    compare(("linear bf16 T=" + std::to_string(T)).c_str(), o);
    auto xf = make(DType::F32, {T, D});
    auto of = zeros(DType::F32, {T, 3072});
    CPU->linear(xf.c, w.c, of.c);
    MTL->linear(xf.m, w.m, of.m);
    compare(("linear f32 T=" + std::to_string(T)).c_str(), of);
  }
  // D: a T-row linear / linear_multi must equal T one-row linears bit for bit (verify rows == decode rows).
  for (int64_t T : {4, 5, 8, 11, 16}) {
    size_t bad = 0;
    for (int f32x = 0; f32x < 2; f32x++)
      for (int f32y = 0; f32y < 2; f32y++) {
        const DType xd = f32x ? DType::F32 : DType::BF16, yd = f32y ? DType::F32 : DType::BF16;
        const int64_t N = 3076;  // not a multiple of 8 rows per threadgroup
        auto x = make(xd, {T, D});
        auto w = make(DType::BF16, {N, D}, 0.02f), w2 = make(DType::BF16, {384, D}, 0.02f);
        Tensor oT = MTL->alloc(yd, {T, N}, "oT"), o1 = MTL->alloc(yd, {1, N}, "o1");
        Tensor mA = MTL->alloc(yd, {T, N}, "mA"), mB = MTL->alloc(yd, {T, 384}, "mB"), m1 = MTL->alloc(yd, {1, 384}, "m1");
        MTL->linear(x.m, w.m, oT);
        MTL->linear_multi(x.m, {&w.m, &w2.m}, {&mA, &mB});
        for (int64_t t = 0; t < T; t++) {
          Tensor xt = x.m;
          xt.set_shape({1, D});
          xt.offset += size_t(t) * D * dtype_size(xd);
          MTL->linear(xt, w.m, o1);
          MTL->linear(xt, w2.m, m1);
          MTL->sync();
          const size_t rb = size_t(N) * dtype_size(yd), rb2 = size_t(384) * dtype_size(yd);
          bad += memcmp(o1.data<char>(), oT.data<char>() + t * rb, rb) != 0;
          bad += memcmp(o1.data<char>(), mA.data<char>() + t * rb, rb) != 0;
          bad += memcmp(m1.data<char>(), mB.data<char>() + t * rb2, rb2) != 0;
        }
        for (Tensor* tt : {&oT, &o1, &mA, &mB, &m1}) MTL->free(*tt);
      }
    char nm[64];
    snprintf(nm, sizeof nm, "linear rows==T1 T=%lld", (long long)T);
    printf("%-28s %s  rows differing: %zu\n", nm, bad ? "FAIL" : "ok  ", bad);
    if (bad) fails++;
  }
  for (int64_t T : {1, 3}) {  // linear_multi vs separate CPU linears (bit-equal to Metal linear expected)
    auto x = make(DType::BF16, {T, D});
    auto w1 = make(DType::BF16, {3072, D}, 0.02f), w2 = make(DType::BF16, {384, D}, 0.02f),
         w3 = make(DType::BF16, {24, D}, 0.02f);
    auto o1 = zeros(DType::BF16, {T, 3072}), o2 = zeros(DType::BF16, {T, 384}), o3 = zeros(DType::BF16, {T, 24});
    CPU->linear_multi(x.c, {&w1.c, &w2.c, &w3.c}, {&o1.c, &o2.c, &o3.c});
    MTL->linear_multi(x.m, {&w1.m, &w2.m, &w3.m}, {&o1.m, &o2.m, &o3.m});
    compare("linear_multi o1", o1);
    compare("linear_multi o2", o2);
    compare("linear_multi o3 (N=24)", o3);
    // copy
    auto c = zeros(DType::BF16, {T, 3072});
    MTL->copy(o1.m, c.m);
    CPU->copy(o1.c, c.c);
    compare("copy", c, 0, 0);
  }
  {  // attn residual
    int64_t T = 3;
    auto s0 = make(DType::BF16, {T, D}, 2.0f), s1 = make(DType::F32, {T, D}, 1.0f), s2 = make(DType::F32, {T, D}, 5.0f);
    auto q = make(DType::F32, {3, D}, 0.05f);
    auto o = zeros(DType::F32, {3, T, D});
    CPU->attn_residual({&s0.c, &s1.c, &s2.c}, q.c, o.c);
    MTL->attn_residual({&s0.m, &s1.m, &s2.m}, q.m, o.m);
    compare("attn_residual nq=3", o, 1e-5);
    auto q4 = make(DType::F32, {4, D}, 0.05f);
    auto o4 = zeros(DType::F32, {4, T, D});
    CPU->loop_messages({&s1.c, &s2.c}, q4.c, o4.c);
    MTL->loop_messages({&s1.m, &s2.m}, q4.m, o4.m);
    compare("loop_messages nq=4", o4, 1e-5);
    // fused mix + norm (Metal) vs attn_residual + rms_norm (Metal primitives): must be bit-identical
    auto q1 = make(DType::F32, {D}, 0.05f);
    auto gam = make(DType::BF16, {D}, 0.2f, 1.0f);
    Tensor mx1 = MTL->alloc(DType::F32, {1, T, D}, "m1"), mx2 = MTL->alloc(DType::F32, {1, T, D}, "m2");
    Tensor n1 = MTL->alloc(DType::BF16, {T, D}, "n1"), n2 = MTL->alloc(DType::BF16, {T, D}, "n2");
    MTL->attn_residual_norm({&s0.m, &s1.m, &s2.m}, q1.m, mx1, gam.m, 1e-6f, n1);
    MTL->attn_residual({&s0.m, &s1.m, &s2.m}, q1.m, mx2);
    MTL->rms_norm(mx2, &gam.m, 1e-6f, D, n2);
    MTL->sync();
    bool same = memcmp(n1.data<uint16_t>(), n2.data<uint16_t>(), size_t(T * D * 2)) == 0 &&
                memcmp(mx1.data<float>(), mx2.data<float>(), size_t(T * D * 4)) == 0;
    printf("%-28s %s\n", "mix+norm fused == separate", same ? "ok" : "FAIL");
    if (!same) fails++;
  }
  for (int64_t T : {2, 40}) {  // controller (T = 40: prefill GEMM path)
    auto x = make(DType::F32, {T, D}), m = make(DType::F32, {4, T, D});
    auto dn = make(DType::BF16, {64, 2 * D}, 0.02f), fg = make(DType::BF16, {D, 64}, 0.1f),
         gn = make(DType::BF16, {D, 64}, 0.1f);
    auto o = zeros(DType::F32, {T, D});
    Tensor mc = m.c, mm = m.m;  // second message row view [T,D]
    mc.offset += T * D * 4;
    mm.offset += T * D * 4;
    mc.set_shape({T, D});
    mm.set_shape({T, D});
    CPU->controller(x.c, mc, dn.c, fg.c, gn.c, o.c);
    MTL->controller(x.m, mm, dn.m, fg.m, gn.m, o.m);
    compare("controller", o, 1e-4);
  }
  for (int64_t T : {1, 3, 8, 16}) {  // controller_multi / rms_norm_multi (Metal fused) vs CPU loops
    Pair xs[3], ms[3], dn[3], fg[3], gn[3], os[3];
    for (int i = 0; i < 3; i++) {
      xs[i] = make(DType::F32, {T, D});
      ms[i] = make(DType::F32, {T, D});
      dn[i] = make(DType::BF16, {64, 2 * D}, 0.02f);
      fg[i] = make(DType::BF16, {D, 64}, 0.1f);
      gn[i] = make(DType::BF16, {D, 64}, 0.1f);
      os[i] = zeros(DType::F32, {T, D});
    }
    Tensor xc[3], mc[3], dc[3], fc[3], gc[3], oc[3], xm[3], mm[3], dm[3], fm[3], gm[3], om[3];
    for (int i = 0; i < 3; i++) {
      xc[i] = xs[i].c, mc[i] = ms[i].c, dc[i] = dn[i].c, fc[i] = fg[i].c, gc[i] = gn[i].c, oc[i] = os[i].c;
      xm[i] = xs[i].m, mm[i] = ms[i].m, dm[i] = dn[i].m, fm[i] = fg[i].m, gm[i] = gn[i].m, om[i] = os[i].m;
    }
    CPU->controller_multi(3, xc, mc, dc, fc, gc, oc);
    MTL->controller_multi(3, xm, mm, dm, fm, gm, om);
    for (int i = 0; i < 3; i++) {
      os[i].c = oc[i];
      os[i].m = om[i];
      compare(("controller_multi part " + std::to_string(i)).c_str(), os[i], 1e-4);
    }
    Pair gam[3], ro[3];
    const Tensor *gcp[3], *gmp[3];
    Tensor roc[3], rom[3];
    for (int i = 0; i < 3; i++) {
      gam[i] = make(DType::BF16, {D}, 0.2f, 1.0f);
      ro[i] = zeros(DType::BF16, {T, D});
      gcp[i] = &gam[i].c;
      gmp[i] = &gam[i].m;
      roc[i] = ro[i].c;
      rom[i] = ro[i].m;
    }
    CPU->rms_norm_multi(3, oc, gcp, 1e-6f, D, roc);
    MTL->rms_norm_multi(3, om, gmp, 1e-6f, D, rom);
    for (int i = 0; i < 3; i++) compare(("rms_norm_multi part " + std::to_string(i)).c_str(), ro[i], 0, 1);
  }
  for (int64_t C : {1024, 2304}) {  // short conv, two calls to exercise the state
    auto w = make(DType::BF16, {C, 4}, 0.3f);
    auto st = zeros(DType::F32, {3, C});
    for (int64_t T : {5, 1, 40}) {
      auto x = make(DType::BF16, {T, C});
      auto o = zeros(DType::BF16, {T, C});
      auto hs = zeros(DType::F32, {T, 3, C});
      CPU->short_conv(x.c, w.c, st.c, o.c, &hs.c);
      MTL->short_conv(x.m, w.m, st.m, o.m, &hs.m);
      compare(("short_conv C=" + std::to_string(C) + " T=" + std::to_string(T)).c_str(), o, 0, 0);
      compare("short_conv hist", hs, 0);
    }
    compare("short_conv state", st, 0);
  }
  for (int64_t T : {1, 4}) {  // fused attn_prep_local (Metal) vs primitive sequence (CPU default)
    auto kraw = make(DType::BF16, {T, 1024}), vraw = make(DType::BF16, {T, 1024});
    auto kw = make(DType::BF16, {1024, 4}, 0.3f), vw = make(DType::BF16, {1024, 4}, 0.3f);
    auto ks = make(DType::F32, {3, 1024}), vs = make(DType::F32, {3, 1024});
    auto kh = zeros(DType::F32, {T, 3, 1024}), vh = zeros(DType::F32, {T, 3, 1024});
    auto q = make(DType::BF16, {T, 24, 128}, 4.0f), r = make(DType::BF16, {T, 24, 16}, 4.0f);
    auto qs = make(DType::BF16, {24, 128}, 0.1f, 1.0f), rs = make(DType::BF16, {24, 16}, 0.1f, 1.0f);
    auto kc = zeros(DType::BF16, {T, 1024}), vc = zeros(DType::BF16, {T, 1024});
    auto k3 = zeros(DType::BF16, {T, 8, 128}), v3 = zeros(DType::BF16, {T, 8, 128});
    CPU->attn_prep_local(kraw.c, vraw.c, kw.c, vw.c, ks.c, vs.c, &kh.c, &vh.c, q.c, qs.c, r.c, rs.c, 22.627417f, 8.0f,
                         1e-6f, kc.c, vc.c, k3.c, v3.c);
    MTL->attn_prep_local(kraw.m, vraw.m, kw.m, vw.m, ks.m, vs.m, &kh.m, &vh.m, q.m, qs.m, r.m, rs.m, 22.627417f, 8.0f,
                         1e-6f, kc.m, vc.m, k3.m, v3.m);
    std::string tn = " T=" + std::to_string(T);
    compare(("prep q" + tn).c_str(), q, 0, 0);
    compare(("prep r" + tn).c_str(), r, 0, 0);
    compare(("prep k3" + tn).c_str(), k3, 0, 0);
    compare(("prep v3" + tn).c_str(), v3, 0, 0);
    compare(("prep kstate" + tn).c_str(), ks, 0);
    compare(("prep vhist" + tn).c_str(), vh, 0);
    // conv_add, f32 out (attn residual) and bf16 out (mlp residual)
    auto x = make(DType::BF16, {T, 2304}), w = make(DType::BF16, {2304, 4}, 0.3f), st = make(DType::F32, {3, 2304});
    auto oth = make(DType::BF16, {T, 2304}), tmp = zeros(DType::BF16, {T, 2304}), out = zeros(DType::F32, {T, 2304});
    CPU->conv_add(x.c, w.c, st.c, nullptr, oth.c, tmp.c, out.c);
    MTL->conv_add(x.m, w.m, st.m, nullptr, oth.m, tmp.m, out.m);
    compare(("conv_add f32" + tn).c_str(), out, 0);
    auto othf = make(DType::F32, {T, 2304}), outb = zeros(DType::BF16, {T, 2304});
    CPU->conv_add(x.c, w.c, st.c, nullptr, othf.c, tmp.c, outb.c);
    MTL->conv_add(x.m, w.m, st.m, nullptr, othf.m, tmp.m, outb.m);
    compare(("conv_add bf16" + tn).c_str(), outb, 0, 0);
  }
  {  // dyn conv
    int64_t C = 1024;
    auto sw = make(DType::BF16, {4, C}, 0.3f);
    auto st = zeros(DType::F32, {3, C});
    for (int64_t T : {6, 1}) {
      auto x = make(DType::BF16, {T, C});
      auto dy = make(DType::BF16, {T, 32}, 0.3f);
      auto o = zeros(DType::BF16, {T, C});
      auto hs = zeros(DType::F32, {T, 3, C});
      CPU->dyn_conv(x.c, dy.c, sw.c, st.c, o.c, &hs.c);
      MTL->dyn_conv(x.m, dy.m, sw.m, st.m, o.m, &hs.m);
      compare(("dyn_conv T=" + std::to_string(T)).c_str(), o, 0, 0);
      compare("dyn_conv hist", hs, 0);
    }
  }
  {  // l2 clip
    int64_t T = 4;
    auto x = make(DType::BF16, {T, 24, 128}, 4.0f);
    auto sc = make(DType::BF16, {24, 128}, 0.1f, 1.0f);
    auto tau = make(DType::F32, {T}, 0.05f, 1.1f);
    auto o = zeros(DType::BF16, {T, 24, 128});
    CPU->l2_clip(x.c, &sc.c, 22.627417f, &tau.c, o.c);
    MTL->l2_clip(x.m, &sc.m, 22.627417f, &tau.m, o.m);
    compare("l2_clip scale+tau", o, 0, 0);
    auto r = make(DType::BF16, {T, 24, 16}, 4.0f);
    auto ro = zeros(DType::BF16, {T, 24, 16});
    CPU->l2_clip(r.c, nullptr, 8.0f, nullptr, ro.c);
    MTL->l2_clip(r.m, nullptr, 8.0f, nullptr, ro.m);
    compare("l2_clip d=16", ro, 0, 0);
  }
  {  // elementwise
    auto a = make(DType::BF16, {3, D}, 2.0f), b = make(DType::F32, {3, D}, 2.0f);
    auto o = zeros(DType::BF16, {3, D}), of = zeros(DType::F32, {3, D});
    CPU->sigmoid_gate(a.c, b.c, o.c);
    MTL->sigmoid_gate(a.m, b.m, o.m);
    compare("sigmoid_gate", o, 0, 1);
    CPU->add(a.c, b.c, of.c);
    MTL->add(a.m, b.m, of.m);
    compare("add f32", of, 0);
    CPU->cast(b.c, o.c);
    MTL->cast(b.m, o.m);
    compare("cast f32->bf16", o, 0, 0);
  }
  // attention: global (linear cache) and local (wrapped ring)
  struct AttnCase { int local; int64_t T, cap, len; };
  for (AttnCase cs : {AttnCase{0, 3, 4096, 1500}, AttnCase{1, 2, 1536, 2000}, AttnCase{0, 1, 4096, 0},
                      AttnCase{0, 1, 40000, 37000}, AttnCase{1, 1, 1536, 700}, AttnCase{1, 4, 1536, 1030},
                      AttnCase{0, 9, 4096, 3000}, AttnCase{1, 40, 1536, 1500}, AttnCase{0, 77, 4096, 0},
                      AttnCase{0, 300, 8192, 5000}, AttnCase{1, 300, 1536, 2100}, AttnCase{1, 1100, 1536, 0}}) {
    int local = cs.local;
    int64_t T = cs.T, cap = cs.cap, len = cs.len;
    auto q = make(DType::BF16, {T, 24, 128}, 0.5f), kn = make(DType::BF16, {T, 8, 128}, 0.5f),
         vn = make(DType::BF16, {T, 8, 128});
    auto r = make(DType::BF16, {T, 24, 16}, 1.0f), rp = make(DType::BF16, {16, 1024}, 0.5f),
         sk = make(DType::BF16, {T, 24}, 1.0f);
    auto kc = make(DType::BF16, {8, cap, 128}, 0.5f), vc = make(DType::BF16, {8, cap, 128});
    auto o = zeros(DType::BF16, {T, 24 * 128});
    AttnArgs ac{q.c, kn.c, vn.c, r.c, rp.c, sk.c, KvView{kc.c, vc.c, cap, len, bool(local)}, local ? 1024 : 0};
    AttnArgs am{q.m, kn.m, vn.m, r.m, rp.m, sk.m, KvView{kc.m, vc.m, cap, len, bool(local)}, local ? 1024 : 0};
    CPU->attention(ac, o.c);
    MTL->attention(am, o.m);
    // T >= 64 runs A's attention GEMM (P stored as IEEE half): allow 1.5% of elements beyond 2 bf16 ulp
    compare(((local ? "attn local T=" : "attn global T=") + std::to_string(T) + " len=" + std::to_string(len)).c_str(),
            o, 0, 2, T >= 64 ? 0.015 : 0.001);
    CPU->kv_append(kn.c, vn.c, ac.kv);
    MTL->kv_append(kn.m, vn.m, am.kv);
    compare("kv_append k", kc, 0, 0);
    if (ac.kv.len != am.kv.len) {
      printf("kv_append len mismatch\n");
      fails++;
    }
  }

  {  // kv_append of a chunk longer than the ring (2048 rows into 1536 slots): last rows win, as on the CPU
    int64_t T = 2048, cap = 1536;
    auto kn = make(DType::BF16, {T, 8, 128}), vn = make(DType::BF16, {T, 8, 128});
    auto kc = zeros(DType::BF16, {8, cap, 128}), vc = zeros(DType::BF16, {8, cap, 128});
    KvView c{kc.c, vc.c, cap, 700, true}, m{kc.m, vc.m, cap, 700, true};
    CPU->kv_append(kn.c, vn.c, c);
    MTL->kv_append(kn.m, vn.m, m);
    compare("kv_append ring T>cap", kc, 0, 0);
  }
  // Verify (T=4 at len L) must equal decode (T=1 at len L+t) bit for bit, row by row (speculative identity).
  // T = 5..16 go through the multi-row kernel (D, kernels/multirow.metal).
  for (int TV : {4, 5, 8, 11, 16})
  for (int local = 0; local < 2; local++) {
    int64_t cap = local ? 1536 : 8192, L = local ? 3001 : 4093;  // ring wrapped / split boundary crossing
    auto q = make(DType::BF16, {TV, 24, 128}, 0.5f), r = make(DType::BF16, {TV, 24, 16}, 1.0f),
         rp = make(DType::BF16, {16, 1024}, 0.5f), sk = make(DType::BF16, {TV, 24}, 1.0f);
    auto kc = make(DType::BF16, {8, cap, 128}, 0.5f), vc = make(DType::BF16, {8, cap, 128});
    // new keys = cache rows at positions L..L+3 (so the T=1 runs read them from the cache)
    Tensor kn = MTL->alloc(DType::BF16, {TV, 8, 128}, "kn"), vn = MTL->alloc(DType::BF16, {TV, 8, 128}, "vn");
    MTL->sync();
    for (int t = 0; t < TV; t++)
      for (int h = 0; h < 8; h++) {
        int64_t slot = local ? (L + t) % cap : L + t;
        memcpy(kn.data<uint16_t>() + (t * 8 + h) * 128, kc.m.data<uint16_t>() + (h * cap + slot) * 128, 256);
        memcpy(vn.data<uint16_t>() + (t * 8 + h) * 128, vc.m.data<uint16_t>() + (h * cap + slot) * 128, 256);
      }
    Tensor o4 = MTL->alloc(DType::BF16, {TV, 24 * 128}, "o4");
    AttnArgs a4{q.m, kn, vn, r.m, rp.m, sk.m, KvView{kc.m, vc.m, cap, L, bool(local)}, local ? 1024 : 0};
    MTL->attention(a4, o4);
    size_t bad = 0;
    for (int t = 0; t < TV; t++) {
      auto rowv = [&](const Tensor& x, int64_t w) {
        Tensor v = x;
        v.set_shape({1, x.dim(1), w > 0 ? x.dim(2) : 0});
        v.ndim = x.ndim;
        v.shape[0] = 1;
        v.offset = x.offset + size_t(t) * (x.nbytes() / x.dim(0));
        return v;
      };
      Tensor qt = rowv(q.m, 1), rt = rowv(r.m, 1), kt = rowv(kn, 1), vt = rowv(vn, 1), st = sk.m;
      st.set_shape({1, 24});
      st.offset += size_t(t) * 48;
      Tensor o1 = MTL->alloc(DType::BF16, {1, 24 * 128}, "o1");
      AttnArgs a1{qt, kt, vt, rt, rp.m, st, KvView{kc.m, vc.m, cap, L + t, bool(local)}, local ? 1024 : 0};
      MTL->attention(a1, o1);
      MTL->sync();
      if (memcmp(o1.data<uint16_t>(), o4.data<uint16_t>() + t * 3072, 3072 * 2) != 0) bad++;
      MTL->free(o1);
    }
    char nm[64];
    snprintf(nm, sizeof nm, "verify==decode %s T=%d", local ? "local ring" : "global", TV);
    printf("%-28s %s  rows differing: %zu/%d\n", nm, bad ? "FAIL" : "ok  ", bad, TV);
    MTL->free(o4);
    MTL->free(kn);
    MTL->free(vn);
    if (bad) fails++;
  }

  // q8 affine linear (8-bit build) vs fp64 host math: y = sum_g s_g * sum(x q) + b_g * sum(x), and the GEMM path
  if (MTL->supports_q8()) {
    const int64_t N = 640, K = 2304, G = K / 64;
    std::vector<uint8_t> codes(N * K);
    std::vector<uint16_t> sc(N * G), bi(N * G);
    std::uniform_int_distribution<int> ud(0, 255);
    std::normal_distribution<float> nd(0, 1);
    for (auto& c : codes) c = uint8_t(ud(rng));
    for (auto& v : sc) v = f32_to_bf16(std::fabs(nd(rng)) * 0.002f);
    for (auto& v : bi) v = f32_to_bf16(nd(rng) * 0.2f);
    Tensor wc = MTL->alloc(DType::U32, {N, K / 4}, "q8w"), ws = MTL->alloc(DType::BF16, {N, G}, "q8s"),
           wb = MTL->alloc(DType::BF16, {N, G}, "q8b");
    MTL->write(wc, codes.data(), codes.size());
    MTL->write(ws, sc.data(), sc.size() * 2);
    MTL->write(wb, bi.data(), bi.size() * 2);
    MTL->bind_affine(wc, ws, wb);
    for (int64_t T : {1, 3, 8, 70}) {
      std::vector<uint16_t> xv(T * K);
      for (auto& v : xv) v = f32_to_bf16(nd(rng));
      Tensor x = MTL->alloc(DType::BF16, {T, K}, "q8x"), y = MTL->alloc(DType::F32, {T, N}, "q8y");
      MTL->write(x, xv.data(), xv.size() * 2);
      MTL->linear(x, wc, y);
      MTL->sync();
      double maxrel = 0, maxref = 0;
      for (int64_t t = 0; t < T; t++)
        for (int64_t n = 0; n < N; n++) {
          double ref = 0;
          for (int64_t k = 0; k < K; k++) {
            double s_ = bf16_to_f32(sc[n * G + k / 64]), b_ = bf16_to_f32(bi[n * G + k / 64]);
            double w = s_ * codes[n * K + k] + b_;
            if (T > 16) w = bf16_to_f32(f32_to_bf16(float(w)));  // GEMM path stages bf16(s*q+b), as MLX qmm
            ref += w * bf16_to_f32(xv[t * K + k]);
          }
          maxref = std::max(maxref, std::fabs(ref));
          maxrel = std::max(maxrel, std::fabs(ref - y.data<float>()[t * N + n]));
        }
      double rel = maxrel / maxref;
      bool ok = rel < 1e-4;
      if (!ok) fails++;
      printf("%-28s %s  max|d| / max|ref| = %.2e\n", ("q8 linear T=" + std::to_string(T)).c_str(), ok ? "ok  " : "FAIL",
             rel);
      MTL->free(x);
      MTL->free(y);
    }
    // embedding rows: bf16(s*q + b)
    std::vector<int32_t> ids = {5, 0, 639, 77};
    Tensor it = MTL->alloc(DType::I32, {4}, "ids"), eo = MTL->alloc(DType::BF16, {4, K}, "emb");
    MTL->write(it, ids.data(), 16);
    MTL->embed(it, wc, eo);
    MTL->sync();
    size_t bad = 0;
    for (int r = 0; r < 4; r++)
      for (int64_t k = 0; k < K; k++) {
        float w = std::fma(bf16_to_f32(sc[ids[r] * G + k / 64]), float(codes[ids[r] * K + k]),
                           bf16_to_f32(bi[ids[r] * G + k / 64]));
        if (f32_to_bf16(w) != eo.data<uint16_t>()[r * K + k]) bad++;
      }
    printf("%-28s %s  mismatches %zu\n", "q8 embed rows", bad ? "FAIL" : "ok  ", bad);
    if (bad) fails++;
  }

  // MoE and router with real weights: layer 0 (NVFP4) and layer 23 (MXFP8)
  SafeTensors st;
  st.add_dir(dir);
  for (int layer : {0, 23}) {
    std::string pre = "language_model.model.layers." + std::to_string(layer) + ".mlp.";
    auto load = [&](const std::string& n) -> Pair {
      const auto& e = st.at(pre + n);
      return {CPU->load_weight(e), MTL->load_weight(e)};
    };
    MoeWeights mc, mm;
    auto gw = load("gate_weight"), bias = load("e_score_correction_bias");
    mc.gate_w = gw.c, mm.gate_w = gw.m, mc.bias = bias.c, mm.bias = bias.m;
    QuantMode mode = layer < 20 ? QuantMode::NVFP4 : QuantMode::MXFP8;
    for (const char* pj : {"gate_proj", "up_proj", "down_proj"}) {
      std::string b = std::string("switch_mlp.") + pj + ".";
      auto w = load(b + "weight"), s = load(b + "scales");
      Pair g{};
      if (mode == QuantMode::NVFP4) g = load(b + "weight_scale_2");
      int64_t n = w.c.dim(1), k = mode == QuantMode::NVFP4 ? w.c.dim(2) * 8 : w.c.dim(2) * 4;
      QuantProj qc{mode, w.c, s.c, g.c, n, k}, qm{mode, w.m, s.m, g.m, n, k};
      if (pj[0] == 'g') mc.gate = qc, mm.gate = qm;
      else if (pj[0] == 'u') mc.up = qc, mm.up = qm;
      else mc.down = qc, mm.down = qm;
    }
    for (int64_t T : {1, 5, 300}) {
      auto x = make(DType::BF16, {T, D}, 1.0f);
      auto ic = zeros(DType::I32, {T, 8}), wc = zeros(DType::F32, {T, 8});
      CPU->router(x.c, mc, 8, 8.0f, ic.c, wc.c);
      MTL->router(x.m, mm, 8, 8.0f, ic.m, wc.m);
      MTL->sync();
      std::vector<int32_t> i1(T * 8), i2(T * 8);
      CPU->read(ic.c, i1.data(), T * 32);
      MTL->read(ic.m, i2.data(), T * 32);
      int same = 0;
      for (size_t i = 0; i < i1.size(); i++) same += i1[i] == i2[i];
      printf("router L%d T=%lld: idx same %d/%zu (order-sensitive)\n", layer, (long long)T, same, i1.size());
      if (same != int(i1.size())) fails++;
      compare("router weights", wc, 1e-5);
      // MoE on the CPU's routes (same idx/w on both)
      MTL->write(ic.m, i1.data(), T * 32);
      std::vector<float> wv(T * 8);
      CPU->read(wc.c, wv.data(), T * 32);
      MTL->write(wc.m, wv.data(), T * 32);
      auto o = zeros(DType::BF16, {T, D});
      CPU->moe(x.c, ic.c, wc.c, mc, o.c);
      MTL->moe(x.m, ic.m, wc.m, mm, o.m);
      compare((std::string("moe ") + (layer < 20 ? "nvfp4" : "mxfp8") + " T=" + std::to_string(T)).c_str(), o, 0,
              2, 0.01);  // three bf16 roundings (g/u, act, down) after differently ordered fp32 sums
    }
  }
  {  // logits (small random head)
    auto h = make(DType::F32, {2, D}, 1.0f);
    auto head = make(DType::BF16, {1000, D}, 0.05f);
    auto o = zeros(DType::F32, {2, 900});
    CPU->logits(h.c, head.c, 900, 30.0f, o.c);
    MTL->logits(h.m, head.m, 900, 30.0f, o.m);
    compare("logits", o, 1e-5);
  }
  printf("%s: %d failing\n", fails ? "FAILED" : "ALL OK", fails);
  return fails ? 1 : 0;
}
