// MetalBackend: ink::Backend on the GPU. Owner: B. Ops only encode into the current command buffer; sync()
// commits and waits. Rounding points follow src/model/backend.h.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <vector>

#include "gpu/context.h"
#include "model/backend.h"

namespace ink {
namespace {

using gpu::Buf;
using gpu::Context;
using gpu::Dim3;

[[noreturn]] void fail(const std::string& m) { throw std::runtime_error("metal backend: " + m); }

uint32_t dtc(DType d) {
  if (d == DType::F32) return 0;
  if (d == DType::BF16) return 1;
  fail(std::string("unsupported dtype ") + dtype_name(d));
}
const char* dtn(DType d) { return d == DType::F32 ? "f32" : "bf16"; }

Buf bufof(const Tensor& t) {
  Buf b;
  b.mtl = t.buf;
  b.host = t.host;
  return b;
}

struct MatvecShape {
  uint32_t N, K, T, x_stride, y_stride;
};
struct MatvecMultiParams {
  uint32_t K, T, count;
  uint32_t row_start[9];
  uint32_t dt_out[8];
  uint32_t x_off[8];
};
struct Rms3Params {
  uint32_t rows, dim;
  float eps;
  uint32_t has_gamma[3], dt_x[3], dt_out[3];
};
struct PrepParams {
  uint32_t T, H, KVH;
  float qk_thr, rel_thr, eps;
  uint32_t has_hist;
};
struct ConvAddParams {
  uint32_t T, C, has_hist, dt_other, dt_out;
};
struct EwParams {
  uint32_t n, dt_a, dt_b, dt_out;
};
struct EmbedParams {
  uint32_t T, D;
};
struct RmsParams {
  uint32_t rows, dim;
  float eps;
  uint32_t has_gamma, dt_x, dt_out;
};
struct GemmParams {
  uint32_t T, N, K, dt_x, dt_out, ldo;
};
struct SkinnyParams {
  uint32_t N, K, T, dt_out;
};
struct LinParams {
  uint32_t T, N, K, dt_x, dt_out;
};
struct ResParams {
  uint32_t T, D, nb, nq;
  uint32_t dt[8];
  float eps;
};
struct CtrlParams {
  uint32_t T, D, R, m_stride;
};
struct ConvParams {
  uint32_t T, C, heads, has_hist;
};
struct ClipParams {
  uint32_t rows, d, H;
  float thr;
  uint32_t has_scale, has_tau;
};
struct KvAppendParams {
  uint32_t T, heads, cap, pos0, ring;
};
struct AttnParams {
  uint32_t T, H, KVH, cap, len, ring, window, rel_extent, use_gate;
};
struct AgParams {  // kernels/attn_gemm.metal
  uint32_t T, rows, rows_pad, H, KVH, cap, len, ring, window, rel_extent, j0, n, use_gate, t0;
};
struct AttnDecParams {
  uint32_t T, H, KVH, cap, len, ring, window, rel_extent, split_keys, nsplit, causal, first, use_sink, use_gate;
};
struct RouterParams {
  uint32_t T, D, E, topk, dt_x;
  float scale;
};
struct QmvParams {
  uint32_t N, K, routes, x_div, mode;
};
struct MoeFastParams {
  uint32_t N, K, topk;
};
struct SortParams {
  uint32_t routes, E, tile_m;
};
struct MoeGemmParams {
  uint32_t N, K, topk, gather_a, scatter_out;
};
struct CombineParams {
  uint32_t T, D, topk;
};
struct SoftcapParams {
  uint32_t n;
  float cap;
};

uint32_t cdiv(uint64_t a, uint64_t b) { return uint32_t((a + b - 1) / b); }

class MetalBackend final : public Backend {
 public:
  MetalBackend() : g_(Context::get()) {
    profile_ = getenv("INK_PROFILE") != nullptr;
    gpu_time_ = getenv("INK_GPU_TIME") != nullptr;
    if (const char* sk = getenv("INK_SKIP")) skip_ = std::string(",") + sk + ",";
    last_sync_ = std::chrono::steady_clock::now();
    skinny_ = getenv("INK_SKINNY") != nullptr;  // off until it beats 2-4 passes of the 4-row matvec
    if (const char* sk = getenv("INK_SPLIT_KEYS")) split_keys_ = uint32_t(atoi(sk));
    if (const char* sk = getenv("INK_SPLIT_LOCAL")) split_local_ = uint32_t(atoi(sk));
  }
  ~MetalBackend() override {
    sync();
    prof_report();
    if (gpu_time_ && acc_n_)
      fprintf(stderr, "gpu time: %ld syncs, per sync: wall %.2f ms (since previous sync), GPU span %.2f ms, CPU wait %.2f ms\n",
              acc_n_, acc_wall_ / acc_n_, acc_gpu_ / acc_n_, acc_wait_ / acc_n_);
    for (auto& [k, b] : scratch_) g_.release(b);
  }
  const char* name() const override { return "metal"; }

  // ---------------------------------------------------------------- memory
  Tensor alloc(DType dt, std::initializer_list<int64_t> shape, const char* tag) override {
    Tensor t;
    t.dtype = dt;
    t.set_shape(shape);
    Buf b = g_.alloc(t.nbytes(), tag, true);
    memset(b.host, 0, b.size);
    t.buf = b.mtl;
    t.host = b.host;
    sizes_[t.buf] = b.size;
    return t;
  }
  void free(Tensor& t) override {
    if (!t.buf) return;
    sync();
    Buf b = bufof(t);
    b.size = sizes_[t.buf];
    sizes_.erase(t.buf);
    g_.release(b);
    t = Tensor{};
  }
  Tensor load_weight(const SafeTensors::Entry& e) override {
    Buf b = g_.load_file(e.file, e.offset, e.bytes, "weights");
    Tensor t;
    t.dtype = e.dtype;
    t.ndim = e.ndim;
    for (int i = 0; i < 4; i++) t.shape[i] = e.shape[i];
    t.buf = b.mtl;
    t.host = b.host;
    sizes_[t.buf] = b.size;
    return t;
  }
  void sync() override {
    if (profile_) prof_flush();
    auto w0 = std::chrono::steady_clock::now();
    g_.commit_and_wait();
    if (gpu_time_) {
      double wait_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count();
      double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - last_sync_).count();
      if (wall < 1000) {
        acc_gpu_ += g_.last_gpu_ms();
        acc_wall_ += wall;
        acc_wait_ += wait_ms;
        acc_n_++;
      }
      last_sync_ = std::chrono::steady_clock::now();
    }
    pending_ = false;
  }
  void copy(const Tensor& src, Tensor& dst, size_t src_off, size_t dst_off, size_t bytes) override {
    if (bytes == 0) bytes = src.nbytes();
    if (bytes >= (64u << 20)) {  // large copies (cache snapshots): blit engine
      g_.blit_copy(bufof(src), src.offset + src_off, bufof(dst), dst.offset + dst_off, bytes);
      pending_ = true;
      return;
    }
    uint32_t n = uint32_t(bytes);
    pipe("copy_bytes");
    g_.set_buffer(0, bufof(src), src.offset + src_off);
    g_.set_buffer(1, bufof(dst), dst.offset + dst_off);
    g_.set_value(2, n);
    g_.dispatch({cdiv(cdiv(n, 16), 256), 1, 1}, {256, 1, 1});
  }
  void read(const Tensor& t, void* dst, size_t bytes, size_t off) override {
    if (pending_) sync();
    memcpy(dst, (const char*)t.host + t.offset + off, bytes);
  }
  void write(Tensor& t, const void* src, size_t bytes, size_t off) override {
    if (pending_) sync();  // the GPU may still read the old contents
    memcpy((char*)t.host + t.offset + off, src, bytes);
  }

  // ---------------------------------------------------------------- helpers
  void bind(uint32_t i, const Tensor& t) { g_.set_buffer(i, bufof(t), t.offset); }
  // INK_SKIP=a,b,..: timing ablation only (results are wrong): ops of these classes encode nothing.
  bool skip(const char* cls) const { return !skip_.empty() && skip_.find(std::string(",") + cls + ",") != std::string::npos; }
  void pipe(const std::string& name) {
    if (profile_) prof_flush();
    g_.set_pipeline(g_.pipeline(name));
    pending_ = true;
    cur_ = name;
  }
  // INK_PROFILE=1: every kernel runs in its own command buffer; GPU time accumulated per kernel name.
  void prof_flush() {
    if (cur_.empty()) return;
    g_.commit_and_wait();
    auto& e = prof_[cur_];
    e.first += g_.last_gpu_ms();
    e.second++;
    cur_.clear();
  }
  void prof_report() {
    if (prof_.empty()) return;
    std::vector<std::pair<std::string, std::pair<double, long>>> v(prof_.begin(), prof_.end());
    std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.second.first > b.second.first; });
    double tot = 0;
    for (auto& e : v) tot += e.second.first;
    fprintf(stderr, "kernel profile (GPU ms, each dispatch waited separately): total %.1f ms\n", tot);
    for (auto& e : v)
      fprintf(stderr, "  %-28s %9.2f ms %7ld calls %8.1f us/call %5.1f%%\n", e.first.c_str(), e.second.first,
              e.second.second, 1e3 * e.second.first / e.second.second, 100 * e.second.first / tot);
  }
  // Scratch buffers reused across ops (serial encoder orders the uses). Grown buffers are retired at sync.
  Tensor scratch(const char* key, DType dt, std::initializer_list<int64_t> shape) {
    Tensor t;
    t.dtype = dt;
    t.set_shape(shape);
    auto& b = scratch_[key];
    if (b.size < t.nbytes()) {
      if (b.mtl) {
        sync();
        g_.release(b);
      }
      b = g_.alloc(std::max<size_t>(t.nbytes(), 1 << 20), key, true);
    }
    t.buf = b.mtl;
    t.host = b.host;
    return t;
  }

  // ---------------------------------------------------------------- ops
  void embed(const Tensor& ids, const Tensor& table, Tensor& out) override {
    EmbedParams p{uint32_t(ids.numel()), uint32_t(table.dim(-1))};
    pipe("embed_rows");
    bind(0, ids);
    bind(1, table);
    bind(2, out);
    g_.set_value(3, p);
    g_.dispatch({cdiv(p.D, 256), p.T, 1}, {256, 1, 1});
  }

  void rms_norm(const Tensor& x, const Tensor* gamma, float eps, int64_t dim, Tensor& out) override {
    if (skip("small")) return;
    bool hg = gamma && gamma->valid();
    RmsParams p{uint32_t(x.numel() / dim), uint32_t(dim), eps, hg ? 1u : 0u, dtc(x.dtype), dtc(out.dtype)};
    if (hg && gamma->dtype != DType::BF16) fail("rms_norm gamma must be bf16");
    pipe("rms_norm_rows");
    bind(0, x);
    bind(1, hg ? *gamma : x);
    bind(2, out);
    g_.set_value(3, p);
    g_.dispatch({p.rows, 1, 1}, {256, 1, 1});
  }

  void linear(const Tensor& x, const Tensor& w, Tensor& out) override {
    if (skip("linear")) return;
    if (w.dtype != DType::BF16) fail("linear weight must be bf16");
    uint32_t K = uint32_t(w.dim(1)), N = uint32_t(w.dim(0));
    uint32_t T = uint32_t(x.numel() / K);
    linear_rows(x, w, out, T, N, K, 0);
  }
  // rows of out computed: [0, N) of w; out row stride = N.
  void linear_rows(const Tensor& x, const Tensor& w, Tensor& out, uint32_t T, uint32_t N, uint32_t K,
                   uint32_t /*unused*/) {
    if (K % 8 == 0 && T > 4 && T <= 16 && x.dtype == DType::BF16 && skinny_) {
      SkinnyParams sp{N, K, T, dtc(out.dtype)};
      pipe("gemm_skinny2");
      bind(0, w);
      bind(1, x);
      bind(2, out);
      g_.set_value(3, sp);
      g_.dispatch({cdiv(N, 16), 1, 1}, {64, 1, 1});
      return;
    }
    if (K % 8 == 0 && T <= 16) {
      for (uint32_t t0 = 0; t0 < T; t0 += 4) {
        uint32_t tt = std::min<uint32_t>(4, T - t0);
        std::string nm = std::string("matvec_bf16_") + dtn(x.dtype) + "_" + dtn(out.dtype) + (tt == 1 ? "_t1" : "_t4") +
                         "_r4_s2";
        MatvecShape s{N, K, tt, K, N};
        pipe(nm);
        bind(0, w);
        g_.set_buffer(1, bufof(x), x.offset + size_t(t0) * K * dtype_size(x.dtype));
        g_.set_buffer(2, bufof(out), out.offset + size_t(t0) * N * dtype_size(out.dtype));
        g_.set_value(3, s);
        g_.dispatch({cdiv(N, 8), 1, 1}, {64, 1, 1});
      }
      return;
    }
    if (!getenv("INK_GEMM_SLOW")) {
      GemmParams gp{T, N, K, dtc(x.dtype), dtc(out.dtype), N};
      pipe(x.dtype == DType::BF16 && K % 8 == 0 && !getenv("INK_GEMM_F32") ? "gemm_bf16" : "gemm_bf16w");
      bind(0, w);
      bind(1, x);
      bind(2, out);
      g_.set_value(3, gp);
      g_.dispatch({cdiv(N, 64), cdiv(T, 64), 1}, {128, 1, 1});
      return;
    }
    LinParams p{T, N, K, dtc(x.dtype), dtc(out.dtype)};
    pipe("linear_generic");
    bind(0, w);
    bind(1, x);
    bind(2, out);
    g_.set_value(3, p);
    g_.dispatch({cdiv(N, 4), T, 1}, {32, 1, 1});
  }

  void linear_multi(const Tensor& x, const std::vector<const Tensor*>& ws, const std::vector<Tensor*>& outs) override {
    if (skip("linear")) return;
    uint32_t K = uint32_t(ws[0]->dim(1)), T = uint32_t(x.numel() / K);
    bool ok = T <= 4 && ws.size() <= 8 && K % 8 == 0 && !getenv("INK_NO_MULTI");
    for (auto* w : ws) ok = ok && w->dim(1) == K && w->dim(0) % 4 == 0 && w->dtype == DType::BF16;
    if (!ok) {
      for (size_t i = 0; i < ws.size(); ++i) linear(x, *ws[i], *outs[i]);
      return;
    }
    MatvecMultiParams p{};
    p.K = K;
    p.T = T;
    p.count = uint32_t(ws.size());
    pipe(x.dtype == DType::BF16 ? "matvec_multi_bf16" : "matvec_multi_f32");
    for (uint32_t i = 0; i < 8; i++) {
      uint32_t j = std::min<uint32_t>(i, p.count - 1);
      bind(i, *ws[j]);
      bind(8 + i, *outs[j]);
    }
    for (uint32_t i = 0; i < p.count; i++) {
      p.row_start[i + 1] = p.row_start[i] + uint32_t(ws[i]->dim(0));
      p.dt_out[i] = dtc(outs[i]->dtype);
    }
    bind(16, x);
    g_.set_value(17, p);
    g_.dispatch({cdiv(p.row_start[p.count], 8), 1, 1}, {64, 1, 1});
  }

  void attn_residual_norm(const std::vector<const Tensor*>& states, const Tensor& query, Tensor& mixed,
                          const Tensor& gamma, float eps, Tensor& normed) override {
    uint32_t D = uint32_t(states[0]->dim(-1)), T = uint32_t(states[0]->numel() / D);
    bool nq1 = query.ndim == 1 || query.dim(0) == 1;
    if (!nq1 || T > 4 || D > 2304 || normed.dtype != DType::BF16 || getenv("INK_NO_FUSE"))
      return Backend::attn_residual_norm(states, query, mixed, gamma, eps, normed);
    mix(states, query, 1, mixed, &gamma, eps, &normed);
  }
  void mix(const std::vector<const Tensor*>& states, const Tensor& query, uint32_t nq, Tensor& out,
           const Tensor* gamma = nullptr, float eps = 0, Tensor* normed = nullptr) {
    if (skip("mix")) return;
    if (states.size() > 8 || states.empty()) fail("attn_residual: 1..8 states");
    uint32_t D = uint32_t(states[0]->dim(-1));
    ResParams p{};
    p.T = uint32_t(states[0]->numel() / D);
    p.D = D;
    p.nb = uint32_t(states.size());
    p.nq = nq;
    bool wide = p.T <= 4;  // decode/verify: one threadgroup per token is latency bound, use 1024 threads
    p.eps = eps;
    pipe(normed ? "attn_residual_mix_norm_1024" : wide ? "attn_residual_mix_1024" : "attn_residual_mix");
    if (normed) wide = true;
    for (uint32_t i = 0; i < 8; i++) {
      const Tensor& s = *states[std::min<size_t>(i, states.size() - 1)];
      bind(i, s);
      p.dt[i] = dtc(s.dtype);
    }
    bind(8, query);
    bind(9, out);
    g_.set_value(10, p);
    bind(11, gamma ? *gamma : query);
    bind(12, normed ? *normed : out);
    g_.dispatch({p.T, 1, 1}, {wide ? 1024u : 256u, 1, 1});
  }
  void attn_residual(const std::vector<const Tensor*>& states, const Tensor& query, Tensor& out) override {
    uint32_t nq = query.ndim == 1 ? 1 : uint32_t(query.dim(0));
    mix(states, query, nq, out);
  }
  void loop_messages(const std::vector<const Tensor*>& states, const Tensor& query, Tensor& out) override {
    mix(states, query, uint32_t(query.dim(0)), out);
  }

  void controller_multi(int n, const Tensor* xs, const Tensor* ms, const Tensor* downs, const Tensor* forgets,
                        const Tensor* gains, Tensor* outs) override {
    uint32_t D = uint32_t(xs[0].dim(-1)), T = uint32_t(xs[0].numel() / D), R = uint32_t(downs[0].dim(0));
    if (n != 3 || T > 4 || getenv("INK_NO_FUSE")) return Backend::controller_multi(n, xs, ms, downs, forgets, gains, outs);
    if (skip("ctrl")) return;
    CtrlParams p{T, D, R, D};
    Tensor pair = scratch("ctrl-pair3", DType::BF16, {3 * T, 2 * D});
    Tensor h = scratch("ctrl-h3", DType::BF16, {3 * T, R});
    pipe("ctrl_pair3");
    for (int i = 0; i < 3; i++) {
      bind(i, xs[i]);
      bind(3 + i, ms[i]);
    }
    bind(6, pair);
    g_.set_value(7, p);
    g_.dispatch({T, 3, 1}, {256, 1, 1});
    // h_i = bf16(pair_i @ down_i^T): one matvec_multi with per-matrix inputs (same per-row math as linear)
    MatvecMultiParams mp{};
    mp.K = 2 * D;
    mp.T = T;
    mp.count = 3;
    pipe("matvec_multi_bf16");
    Tensor hv[3];
    for (uint32_t i = 0; i < 8; i++) {
      uint32_t j = std::min<uint32_t>(i, 2);
      bind(i, downs[j]);
      hv[j] = h;
      hv[j].offset = h.offset + size_t(j) * T * R * 2;
      bind(8 + i, hv[j]);
    }
    for (uint32_t i = 0; i < 3; i++) {
      mp.row_start[i + 1] = mp.row_start[i] + R;
      mp.dt_out[i] = 1;
      mp.x_off[i] = i * T * 2 * D;
    }
    bind(16, pair);
    g_.set_value(17, mp);
    g_.dispatch({cdiv(mp.row_start[3], 8), 1, 1}, {64, 1, 1});
    pipe("ctrl_out3");
    for (int i = 0; i < 3; i++) {
      bind(i, xs[i]);
      bind(3 + i, ms[i]);
      bind(7 + i, forgets[i]);
      bind(10 + i, gains[i]);
      bind(13 + i, outs[i]);
    }
    bind(6, h);
    g_.set_value(16, p);
    g_.dispatch({cdiv(D, 256), T, 3}, {256, 1, 1});
  }
  void rms_norm_multi(int n, const Tensor* xs, const Tensor* const* gammas, float eps, int64_t dim,
                      Tensor* outs) override {
    if (n != 3 || getenv("INK_NO_FUSE")) return Backend::rms_norm_multi(n, xs, gammas, eps, dim, outs);
    if (skip("small")) return;
    Rms3Params p{};
    p.rows = uint32_t(xs[0].numel() / dim);
    p.dim = uint32_t(dim);
    p.eps = eps;
    pipe("rms_norm3");
    for (int i = 0; i < 3; i++) {
      bool hg = gammas[i] && gammas[i]->valid();
      if (hg && gammas[i]->dtype != DType::BF16) fail("rms_norm gamma must be bf16");
      p.has_gamma[i] = hg;
      p.dt_x[i] = dtc(xs[i].dtype);
      p.dt_out[i] = dtc(outs[i].dtype);
      bind(i, xs[i]);
      bind(3 + i, hg ? *gammas[i] : xs[i]);
      bind(6 + i, outs[i]);
    }
    g_.set_value(9, p);
    g_.dispatch({p.rows, 3, 1}, {256, 1, 1});
  }
  void controller(const Tensor& x, const Tensor& m, const Tensor& down, const Tensor& forget, const Tensor& gain,
                  Tensor& out) override {
    if (skip("ctrl")) return;
    uint32_t D = uint32_t(x.dim(-1)), T = uint32_t(x.numel() / D), R = uint32_t(down.dim(0));
    CtrlParams p{T, D, R, D};
    Tensor pair = scratch("ctrl-pair", DType::BF16, {T, 2 * D});
    Tensor h = scratch("ctrl-h", DType::BF16, {T, R});
    pipe("ctrl_pair");
    bind(0, x);
    bind(1, m);
    bind(2, pair);
    g_.set_value(3, p);
    g_.dispatch({T, 1, 1}, {256, 1, 1});
    linear(pair, down, h);
    pipe("ctrl_out");
    bind(0, x);
    bind(1, m);
    bind(2, h);
    bind(3, forget);
    bind(4, gain);
    bind(5, out);
    g_.set_value(6, p);
    g_.dispatch({cdiv(D, 256), T, 1}, {256, 1, 1});
  }

  void short_conv(const Tensor& x, const Tensor& w, Tensor& state, Tensor& out, Tensor* hist) override {
    if (skip("small")) return;
    uint32_t C = uint32_t(x.dim(-1));
    bool hh = hist && hist->valid();
    ConvParams p{uint32_t(x.numel() / C), C, 0, hh ? 1u : 0u};
    pipe("short_conv4");
    bind(0, x);
    bind(1, w);
    bind(2, state);
    bind(3, out);
    g_.set_value(4, p);
    bind(5, hh ? *hist : state);
    g_.dispatch({cdiv(C, 256), 1, 1}, {256, 1, 1});
  }

  void dyn_conv(const Tensor& x, const Tensor& dyn, const Tensor& static_w, Tensor& state, Tensor& out,
                Tensor* hist) override {
    uint32_t C = uint32_t(x.dim(-1));
    bool hh = hist && hist->valid();
    ConvParams p{uint32_t(x.numel() / C), C, uint32_t(dyn.dim(-1) / 4), hh ? 1u : 0u};
    pipe("dyn_conv4");
    bind(0, x);
    bind(1, dyn);
    bind(2, static_w);
    bind(3, state);
    bind(4, out);
    g_.set_value(5, p);
    bind(6, hh ? *hist : state);
    g_.dispatch({cdiv(C, 256), 1, 1}, {256, 1, 1});
  }

  void l2_clip(const Tensor& x, const Tensor* scale, float thr, const Tensor* tau, Tensor& out) override {
    if (skip("small")) return;
    uint32_t d = uint32_t(x.dim(-1)), H = uint32_t(x.dim(-2));
    if (d > 128) fail("l2_clip: d > 128");
    bool hs = scale && scale->valid(), ht = tau && tau->valid();
    ClipParams p{uint32_t(x.numel() / d), d, H, thr, hs ? 1u : 0u, ht ? 1u : 0u};
    pipe("l2_clip_rows");
    bind(0, x);
    bind(1, hs ? *scale : x);
    bind(2, ht ? *tau : x);
    bind(3, out);
    g_.set_value(4, p);
    g_.dispatch({p.rows, 1, 1}, {32, 1, 1});
  }

  void ew(const char* k, const Tensor& a, const Tensor* b, Tensor& out) {
    if (skip("small")) return;
    EwParams p{uint32_t(out.numel()), dtc(a.dtype), b ? dtc(b->dtype) : 0u, dtc(out.dtype)};
    pipe(k);
    bind(0, a);
    bind(1, b ? *b : a);
    bind(2, out);
    g_.set_value(3, p);
    g_.dispatch({cdiv(p.n, 256), 1, 1}, {256, 1, 1});
  }
  void sigmoid_gate(const Tensor& a, const Tensor& gt, Tensor& out) override { ew("ew_sigmoid_gate", a, &gt, out); }
  void add(const Tensor& a, const Tensor& b, Tensor& out) override { ew("ew_add", a, &b, out); }
  void cast(const Tensor& a, Tensor& out) override { ew("ew_cast", a, nullptr, out); }

  void attention(const AttnArgs& a, Tensor& out) override {
    if (skip("attn")) return; attention_impl(a, nullptr, out); }
  void attention_impl(const AttnArgs& a, const Tensor* gate, Tensor& out) {
    uint32_t T = uint32_t(a.q.dim(0)), H = uint32_t(a.q.dim(1)), KVH = uint32_t(a.k_new.dim(1));
    if (T <= 4 && H == 3 * KVH && !getenv("INK_ATTN_SLOW")) {
      uint32_t len = uint32_t(a.kv.len), window = uint32_t(a.window);
      uint32_t first = (window && len + 1 > window) ? len + 1 - window : 0;
      uint32_t nkeys = len + T - first;
      // fixed per layer type and position-aligned: verify rows == decode bit for bit. Measured (one run each):
      // local window 1024: 64 keys best (0.056 ms vs 0.207 at 512); global 128k: 512 best (2.94 ms vs 3.76 at 128)
      uint32_t split_keys = window ? split_local_ : split_keys_;
      uint32_t nsplit = cdiv(len + T, split_keys) - first / split_keys;
      (void)nkeys;
      AttnDecParams p{T, H, KVH, uint32_t(a.kv.cap), len, a.kv.ring ? 1u : 0u, window, uint32_t(a.rel_proj.dim(1)),
                      split_keys, nsplit, 1u, first, 1u, gate ? 1u : 0u};
      Tensor part = scratch("attn-part", DType::F32, {KVH * nsplit * 24, 132});
      pipe(T <= 2 ? "attn_decode_split_f1" : "attn_decode_split_f2");
      bind(0, a.q);
      bind(1, a.k_new);
      bind(2, a.v_new);
      bind(3, a.r);
      bind(4, a.rel_proj);
      bind(5, a.kv.k.valid() ? a.kv.k : a.k_new);
      bind(6, a.kv.v.valid() ? a.kv.v : a.v_new);
      bind(7, part);
      g_.set_value(8, p);
      g_.dispatch({KVH, nsplit, 1}, {64, 1, 1});
      pipe("attn_decode_combine");
      bind(0, part);
      bind(1, a.sink);
      bind(2, out);
      g_.set_value(3, p);
      bind(4, gate ? *gate : out);
      g_.dispatch({T, H, 1}, {128, 1, 1});
      return;
    }
    AttnParams p{T, H, KVH, uint32_t(a.kv.cap), uint32_t(a.kv.len), a.kv.ring ? 1u : 0u, uint32_t(a.window),
                 uint32_t(a.rel_proj.dim(1)), gate ? 1u : 0u};
    if (H == 3 * KVH && T >= ag_min_t_ && !getenv("INK_ATTN_SLOW")) {
      attention_gemm(a, gate, out);
      return;
    }
    if (H == 3 * KVH && !getenv("INK_ATTN_SLOW")) {
      pipe("attn_prefill");
      bind(0, a.q);
      bind(1, a.k_new);
      bind(2, a.v_new);
      bind(3, a.r);
      bind(4, a.rel_proj);
      bind(5, a.sink);
      bind(6, a.kv.k.valid() ? a.kv.k : a.k_new);
      bind(7, a.kv.v.valid() ? a.kv.v : a.v_new);
      bind(8, out);
      g_.set_value(9, p);  // same layout as AttnPreParams
      bind(10, gate ? *gate : out);
      g_.dispatch({cdiv(T, 32), KVH, 1}, {384, 1, 1});
      return;
    }
    if (gate) fail("attention_generic: no gate");
    pipe("attention_generic");
    bind(0, a.q);
    bind(1, a.k_new);
    bind(2, a.v_new);
    bind(3, a.r);
    bind(4, a.rel_proj);
    bind(5, a.sink);
    bind(6, a.kv.k.valid() ? a.kv.k : a.k_new);
    bind(7, a.kv.v.valid() ? a.kv.v : a.v_new);
    bind(8, out);
    g_.set_value(9, p);
    g_.dispatch({T, H, 1}, {256, 1, 1});
  }

  // Prefill attention as matrix products (kernels/attn_gemm.metal, owner A; port of saetta's attention_gemm).
  void attention_gemm(const AttnArgs& a, const Tensor* gate, Tensor& out) {
    const uint32_t T = uint32_t(a.q.dim(0)), H = uint32_t(a.q.dim(1)), KVH = uint32_t(a.k_new.dim(1));
    const uint32_t KEYS = 1024, slice = 2048;
    const uint32_t len = uint32_t(a.kv.len), window = uint32_t(a.window), ext = uint32_t(a.rel_proj.dim(1));
    const uint32_t rows_pad = cdiv(std::min(T, slice) * 3, 64) * 64;
    Tensor sh = scratch("ag-sh", DType::F32, {KVH, rows_pad, KEYS});
    Tensor tmax = scratch("ag-tmax", DType::F32, {KVH, rows_pad, KEYS / 64});
    Tensor m1 = scratch("ag-m1", DType::F32, {KVH, rows_pad}), m2 = scratch("ag-m2", DType::F32, {KVH, rows_pad});
    Tensor l1 = scratch("ag-l1", DType::F32, {KVH, rows_pad}), l2 = scratch("ag-l2", DType::F32, {KVH, rows_pad});
    Tensor o = scratch("ag-o", DType::F32, {KVH, rows_pad, 128});
    Tensor rpt = scratch("ag-rpt", DType::F32, {ext, 16});
    const Tensor& kc = a.kv.k.valid() ? a.kv.k : a.k_new;
    const Tensor& vc = a.kv.v.valid() ? a.kv.v : a.v_new;
    AgParams p{T, 0, rows_pad, H, KVH, uint32_t(a.kv.cap), len, a.kv.ring ? 1u : 0u, window, ext, 0, 0,
               gate ? 1u : 0u, 0};
    pipe("ag_rel_t");
    bind(0, a.rel_proj);
    bind(1, rpt);
    g_.set_value(2, p);
    g_.dispatch({cdiv(ext, 64), 1, 1}, {64, 16, 1});
    for (uint32_t t0 = 0; t0 < T; t0 += slice) {
      const uint32_t ts = std::min(slice, T - t0), rows = ts * 3, row_tiles = cdiv(rows, 64);
      p.rows = rows;
      p.t0 = t0;
      p.j0 = 0;
      p.n = 0;
      pipe("ag_init");
      bind(0, a.sink);
      bind(1, m1);
      bind(2, l1);
      bind(3, o);
      g_.set_value(4, p);
      g_.dispatch({1, rows_pad, KVH}, {8, 1, 1});
      const uint32_t end = len + t0 + ts;
      uint32_t first = (window && len + t0 + 1 > window) ? len + t0 + 1 - window : 0;
      first = first / 64 * 64;
      bool flip = false;  // false: read m1/l1, write m2/l2
      for (uint32_t j0 = first; j0 < end; j0 += KEYS) {
        p.j0 = j0;
        p.n = std::min(KEYS, end - j0);
        pipe("ag_scores");
        bind(0, a.q);
        bind(1, a.k_new);
        bind(2, kc);
        bind(3, sh);
        bind(4, tmax);
        bind(5, a.r);
        bind(6, rpt);
        g_.set_value(7, p);
        g_.dispatch({cdiv(p.n, 64), row_tiles, KVH}, {128, 1, 1});
        pipe("ag_pv");
        bind(0, sh);
        bind(1, a.v_new);
        bind(2, vc);
        bind(3, o);
        bind(4, tmax);
        g_.set_value(5, p);
        bind(6, flip ? m2 : m1);
        bind(7, flip ? m1 : m2);
        bind(8, flip ? l2 : l1);
        bind(9, flip ? l1 : l2);
        g_.dispatch({2, row_tiles, KVH}, {128, 1, 1});
        flip = !flip;
      }
      pipe("ag_final");
      bind(0, o);
      bind(1, flip ? l2 : l1);
      bind(2, out);
      bind(3, gate ? *gate : out);
      g_.set_value(4, p);
      g_.dispatch({1, rows, KVH}, {128, 1, 1});
    }
  }

  void kv_append(const Tensor& k_new, const Tensor& v_new, KvView& kv) override {
    if (skip("small")) return;
    uint32_t T = uint32_t(k_new.dim(0)), heads = uint32_t(k_new.dim(1));
    if (!kv.ring && kv.len + T > kv.cap) fail("kv_append: global cache full");
    KvAppendParams p{T, heads, uint32_t(kv.cap), uint32_t(kv.len), kv.ring ? 1u : 0u};
    pipe("kv_append_rows");
    bind(0, k_new);
    bind(1, v_new);
    bind(2, kv.k);
    bind(3, kv.v);
    g_.set_value(4, p);
    g_.dispatch({1, heads, T}, {128, 1, 1});
    kv.len += T;
  }

  void router(const Tensor& x, const MoeWeights& m, int topk, float scale, Tensor& idx, Tensor& w) override {
    if (skip("router")) return;
    uint32_t D = uint32_t(x.dim(-1)), T = uint32_t(x.numel() / D), E = uint32_t(m.gate_w.dim(0));
    RouterParams p{T, D, E, uint32_t(topk), dtc(x.dtype), scale};
    if (E > 128 || topk > 16) fail("router: shape");
    if (T <= 16 && D % 8 == 0 && getenv("INK_ROUTER_FUSED")) {
      pipe("router_fused");
      bind(0, x);
      bind(1, m.gate_w);
      bind(2, m.bias);
      bind(3, idx);
      bind(4, w);
      g_.set_value(5, p);
      g_.dispatch({T, 1, 1}, {1024, 1, 1});
      return;
    }
    Tensor lg = scratch("router-logit", DType::F32, {T, E});
    if (T <= 4 && x.dtype == DType::BF16 && D % 8 == 0) {
      // small N: one row per simdgroup for more threadgroups (same per-row arithmetic as linear)
      MatvecShape ms{E, D, T, D, E};
      pipe(T == 1 ? "matvec_bf16_bf16_f32_t1_r1_s4" : "matvec_bf16_bf16_f32_t4_r1_s4");
      bind(0, m.gate_w);
      bind(1, x);
      bind(2, lg);
      g_.set_value(3, ms);
      g_.dispatch({cdiv(E, 4), 1, 1}, {128, 1, 1});
    } else {
      linear(x, m.gate_w, lg);  // f32(x) @ f32(gate_w)^T, fp32 out
    }
    pipe("router_select");
    bind(0, lg);
    bind(1, m.bias);
    bind(2, idx);
    bind(3, w);
    g_.set_value(4, p);
    g_.dispatch({T, 1, 1}, {E, 1, 1});
  }

  void qmv(const QuantProj& q, const Tensor& x, const Tensor& idx, uint32_t routes, uint32_t x_div, Tensor& out) {
    QmvParams p{uint32_t(q.n), uint32_t(q.k), routes, x_div, q.mode == QuantMode::NVFP4 ? 0u : 1u};
    pipe("qmv_experts");
    bind(0, q.weight);
    bind(1, q.scales);
    bind(2, q.global.valid() ? q.global : q.scales);
    bind(3, x);
    bind(4, idx);
    bind(5, out);
    g_.set_value(6, p);
    g_.dispatch({cdiv(p.N, 8), routes, 1}, {64, 1, 1});
  }

  void moe(const Tensor& x, const Tensor& idx, const Tensor& w, const MoeWeights& m, Tensor& out) override {
    if (skip("moe")) return;
    if (x.dtype != DType::BF16) fail("moe: x must be bf16");
    uint32_t D = uint32_t(x.dim(-1)), T = uint32_t(x.numel() / D), topk = uint32_t(idx.dim(-1));
    uint32_t routes = T * topk, I = uint32_t(m.gate.n);
    if (topk == 8 && T <= 16 && !getenv("INK_MOE_SLOW")) {
      bool nv = m.gate.mode == QuantMode::NVFP4;
      Tensor at = scratch("moe-a", DType::BF16, {routes, I});
      MoeFastParams gp{I, D, topk};
      pipe(nv ? "moe_gateup_nvfp4" : "moe_gateup_mxfp8");
      bind(0, m.gate.weight);
      bind(1, m.gate.scales);
      bind(2, m.gate.global.valid() ? m.gate.global : m.gate.scales);
      bind(3, m.up.weight);
      bind(4, m.up.scales);
      bind(5, m.up.global.valid() ? m.up.global : m.up.scales);
      bind(6, x);
      bind(7, idx);
      bind(8, at);
      g_.set_value(9, gp);
      g_.dispatch({routes, cdiv(I, 16), 1}, {128, 1, 1});
      MoeFastParams dp{D, I, topk};
      pipe(nv ? "moe_down_nvfp4" : "moe_down_mxfp8");
      bind(0, m.down.weight);
      bind(1, m.down.scales);
      bind(2, m.down.global.valid() ? m.down.global : m.down.scales);
      bind(3, at);
      bind(4, idx);
      bind(5, w);
      bind(6, out);
      g_.set_value(7, dp);
      g_.dispatch({T, cdiv(D, 4), 1}, {128, 1, 1});
      return;
    }
    if (topk == 8 && !getenv("INK_MOE_SLOW")) {
      moe_prefill(x, idx, w, m, out, T, D, topk);
      return;
    }
    Tensor gt = scratch("moe-g", DType::BF16, {routes, I});
    Tensor ut = scratch("moe-u", DType::BF16, {routes, I});
    Tensor at = scratch("moe-a", DType::BF16, {routes, I});
    Tensor dt = scratch("moe-d", DType::BF16, {routes, D});
    qmv(m.gate, x, idx, routes, topk, gt);
    qmv(m.up, x, idx, routes, topk, ut);
    uint32_t n = routes * I;
    pipe("softcap_swiglu");
    bind(0, gt);
    bind(1, ut);
    bind(2, at);
    g_.set_value(3, n);
    g_.dispatch({cdiv(n, 256), 1, 1}, {256, 1, 1});
    qmv(m.down, at, idx, routes, 1, dt);
    CombineParams cp{T, D, topk};
    pipe("moe_combine");
    bind(0, dt);
    bind(1, w);
    bind(2, out);
    g_.set_value(3, cp);
    g_.dispatch({cdiv(D, 256), T, 1}, {256, 1, 1});
  }

  void moe_gemm(const QuantProj& q, const Tensor& a, const Tensor& perm, const Tensor& offs, const Tensor& tiles,
                const Tensor& tiles_n, uint32_t max_tiles, uint32_t topk, bool gather, bool scatter, Tensor& out) {
    MoeGemmParams p{uint32_t(q.n), uint32_t(q.k), topk, gather ? 1u : 0u, scatter ? 1u : 0u};
    pipe(q.mode == QuantMode::NVFP4 ? "moe_gemm_nvfp4" : "moe_gemm_mxfp8");
    bind(0, q.weight);
    bind(1, q.scales);
    bind(2, q.global.valid() ? q.global : q.scales);
    bind(3, a);
    bind(4, perm);
    bind(5, offs);
    bind(6, tiles);
    bind(7, tiles_n);
    bind(8, out);
    g_.set_value(9, p);
    g_.dispatch({cdiv(p.N, 64), max_tiles, 1}, {128, 1, 1});
  }
  void moe_prefill(const Tensor& x, const Tensor& idx, const Tensor& w, const MoeWeights& m, Tensor& out, uint32_t T,
                   uint32_t D, uint32_t topk) {
    uint32_t routes = T * topk, I = uint32_t(m.gate.n), E = 128;
    uint32_t max_tiles = cdiv(routes, 64) + E;
    Tensor perm = scratch("moep-perm", DType::I32, {routes});
    Tensor offs = scratch("moep-offs", DType::I32, {E + 1});
    Tensor tiles = scratch("moep-tiles", DType::I32, {max_tiles, 2});
    Tensor tiles_n = scratch("moep-ntiles", DType::I32, {4});
    Tensor gt = scratch("moe-g", DType::BF16, {routes, I});
    Tensor ut = scratch("moe-u", DType::BF16, {routes, I});
    Tensor at = scratch("moe-a", DType::BF16, {routes, I});
    Tensor dt = scratch("moe-d", DType::BF16, {routes, D});
    SortParams sp{routes, E, 64};
    pipe("moe_sort");
    bind(0, idx);
    bind(1, perm);
    bind(2, offs);
    bind(3, tiles);
    bind(4, tiles_n);
    g_.set_value(5, sp);
    g_.dispatch({1, 1, 1}, {1024, 1, 1});
    moe_gemm(m.gate, x, perm, offs, tiles, tiles_n, max_tiles, topk, true, false, gt);
    moe_gemm(m.up, x, perm, offs, tiles, tiles_n, max_tiles, topk, true, false, ut);
    uint32_t n = routes * I;
    pipe("softcap_swiglu");
    bind(0, gt);
    bind(1, ut);
    bind(2, at);
    g_.set_value(3, n);
    g_.dispatch({cdiv(n, 256), 1, 1}, {256, 1, 1});
    moe_gemm(m.down, at, perm, offs, tiles, tiles_n, max_tiles, topk, false, true, dt);
    CombineParams cp{T, D, topk};
    pipe("moe_combine");
    bind(0, dt);
    bind(1, w);
    bind(2, out);
    g_.set_value(3, cp);
    g_.dispatch({cdiv(D, 256), T, 1}, {256, 1, 1});
  }

  void logits(const Tensor& h, const Tensor& head, int64_t vocab, float softcap, Tensor& out) override {
    if (skip("logits")) return;
    uint32_t D = uint32_t(h.dim(-1)), T = uint32_t(h.numel() / D);
    Tensor hb = scratch("logits-h", DType::F32, {T, D});
    uint32_t n = T * D;
    pipe("round_bf_f32");
    bind(0, h);
    bind(1, hb);
    g_.set_value(2, n);
    g_.dispatch({cdiv(n, 256), 1, 1}, {256, 1, 1});
    // Only the first `vocab` rows of the head are needed; out row stride = vocab.
    linear_rows(hb, head, out, T, uint32_t(vocab), D, 0);
    if (softcap > 0) {
      SoftcapParams sp{uint32_t(T * vocab), softcap};
      pipe("softcap_f32");
      bind(0, out);
      g_.set_value(1, sp);
      g_.dispatch({cdiv(sp.n, 256), 1, 1}, {256, 1, 1});
    }
  }

 private:
  Context& g_;
  bool pending_ = false;
  bool profile_ = false;
  bool gpu_time_ = false;
  std::string skip_;
  std::chrono::steady_clock::time_point last_sync_;
  double acc_gpu_ = 0, acc_wall_ = 0, acc_wait_ = 0;
  long acc_n_ = 0;
  bool skinny_ = false;
  uint32_t split_keys_ = 512;
  uint32_t split_local_ = 64;
  uint32_t ag_min_t_ = getenv("INK_AG_MIN_T") ? uint32_t(atoi(getenv("INK_AG_MIN_T"))) : 64;  // GEMM attention from this T
  std::string cur_;
  std::map<std::string, std::pair<double, long>> prof_;
  std::unordered_map<void*, size_t> sizes_;
  std::unordered_map<std::string, Buf> scratch_;
};

}  // namespace

std::unique_ptr<Backend> make_metal_backend() { return std::make_unique<MetalBackend>(); }

}  // namespace ink
