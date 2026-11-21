// Owner: A (interface). Implementations: CpuBackend (A, src/model/cpu_backend.cpp, the reference oracle) and
// MetalBackend (B, src/gpu/). model.cpp only talks to this interface, so the same forward pass runs on both and
// every Metal op can be checked against the CPU op on the same inputs.
//
// Conventions
// - Shapes are row-major contiguous. T = tokens in this call. D = 2304.
// - "bf16 out" means: compute in fp32, round to bf16 once at the end (round-to-nearest-even), exactly where the
//   MLX reference rounds. The comments say where the reference rounds; keep those points.
// - All ops are enqueued in order; results are only guaranteed visible to the host after sync().
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/safetensors.h"
#include "core/tensor.h"

namespace ink {

enum class QuantMode : uint8_t { NVFP4, MXFP8 };

// One projection of all 128 experts. NVFP4: weight u32 [E,N,K/8] (low nibble first, e2m1), scales u8 e4m3
// [E,N,K/16], global f32 [E] (w = fp4 * e4m3 * global). MXFP8: weight u32 [E,N,K/4] (e4m3 bytes), scales u8 e8m0
// [E,N,K/32] (w = e4m3 * 2^(s-127)), global unused.
struct QuantProj {
  QuantMode mode;
  Tensor weight, scales, global;
  int64_t n = 0, k = 0;  // per expert: out rows, in cols
};
struct MoeWeights {
  Tensor gate_w;   // bf16 [128, D] router
  Tensor bias;     // f32 [128] e_score_correction_bias (selection only)
  QuantProj gate, up, down;  // gate/up: [1152, D], down: [D, 1152]
};

// Per-visit KV cache. Layout: k, v bf16 [kv_heads=8][cap][128]. Global: slot = pos (cap grows by realloc in
// Model). Local: ring, slot = pos % cap, cap = window + slack (1024 + 512). `len` = tokens already stored.
struct KvView {
  Tensor k, v;
  int64_t cap = 0;
  int64_t len = 0;
  bool ring = false;
};

struct AttnArgs {
  Tensor q;         // bf16 [T,24,128] already l2-clipped (and tau-scaled on global layers)
  Tensor k_new;     // bf16 [T,8,128] this call's keys (clipped), positions len..len+T-1
  Tensor v_new;     // bf16 [T,8,128] (rms-normed)
  Tensor r;         // bf16 [T,24,16] relative query (clipped, tau-scaled on global)
  Tensor rel_proj;  // bf16 [16,1024]
  Tensor sink;      // bf16 [T,24] per-query sink logit (zero K/V column)
  KvView kv;        // past tokens (positions [0, kv.len)), NOT yet containing k_new/v_new
  int window = 0;   // 0 = global (causal, bias for dist < 1024, 0 bias beyond); 1024 = local (dist >= 1024 masked)
  // Reference math per head h (kv head h/3), query i at pos p_i = kv.len + i, key j at pos p_j <= p_i:
  //   bias = bf16(sum_d r[i,h,d] * rel_proj[d, p_i - p_j]) if dist < 1024 else 0
  //   s_ij = (q . k_j) * 128^-0.5 + bias  (fp32), plus a sink logit s = float(sink[i,h]) with value 0
  //   out = softmax(s) @ v  -> bf16
};

// ---- Chain ops (A): one dispatch for a run of small per-token ops, T <= 8. Backends without them return false
// from has_chain() and Model runs the separate ops (the CPU reference never uses chains).
struct ChainConv {          // causal short conv K=4: in bf16 [T,D], w bf16 [D,4], state f32 [3,D], hist or null
  Tensor in, w;
  Tensor* state = nullptr;
  Tensor* hist = nullptr;
};
struct ChainMsg {           // loop messages of one visit, never materialized: m_slot = softmax mix of the bank
  Tensor bank0, bank1;      // bf16 [T,D]
  Tensor queries;           // f32 [4,D]: slots attn, k, v, mlp
};
struct ChainCtrl {
  Tensor down, forget, gain;
};
struct ChainFrontArgs {     // any layer with L > 0
  bool has_conv = false;    // first finish the previous visit: x = bf16(res + conv(moe out))
  ChainConv conv;
  Tensor res;               // f32 [T,D] (has_conv)
  Tensor x;                 // bf16 [T,D]: written when has_conv, read otherwise
  std::vector<const Tensor*> blocks;  // bf16 [T,D] each, at most 4
  int nq = 1;               // 1: local (one query, one norm); 3: global (q, k, v queries and norms)
  Tensor query, gamma;      // attention_input query f32 [nq,D]; input_layernorm (local) or q_input_layernorm
  Tensor gamma_k, gamma_v;  // global: k_ / v_input_layernorm
  const ChainMsg* msg = nullptr;      // loop passes: controllers attn, k, v (slots 0..2)
  const ChainCtrl* ctrl = nullptr;    // [3]
  float eps = 1e-6f;
  Tensor xq, xk, xv;        // bf16 [T,D]; xk, xv only with msg or nq == 3
};
struct ChainMidArgs {       // after o_proj, any layer
  ChainConv conv;           // o_proj out + attn_sconv
  Tensor x;                 // bf16 [T,D] visit input
  bool boundary = false;    // block boundary: res = conv only, x is copied to blk and is the newest block
  Tensor blk;
  std::vector<const Tensor*> blocks;  // completed blocks, without blk
  Tensor query, gamma;      // mlp_input query, post_attention_layernorm
  const ChainMsg* msg = nullptr;      // loop passes: mlp controller (slot 3)
  const ChainCtrl* ctrl = nullptr;
  const MoeWeights* moe = nullptr;
  int topk = 8;
  float route_scale = 8.0f, eps = 1e-6f;
  Tensor res, hn, idx, w;   // out: f32 [T,D], bf16 [T,D], i32 [T,topk], f32 [T,topk]
};

class Backend {
 public:
  virtual ~Backend() = default;
  virtual const char* name() const = 0;

  // Memory.
  virtual Tensor alloc(DType, std::initializer_list<int64_t> shape, const char* tag) = 0;  // zero-filled
  virtual void free(Tensor&) = 0;
  // Load a weight from its safetensors file into resident memory (B: pread into wired buffers).
  virtual Tensor load_weight(const SafeTensors::Entry&) = 0;
  // MLX affine 8-bit dense weights (8-bit build). A backend that returns true from supports_q8() gets such
  // weights as they are stored: `codes` U32 [N, K/4] (4 uint8 codes per word, low byte first) with bf16 `scales`
  // and `biases` [N, K/64]; value = bf16(scale * q + bias) (what the load-time conversion produces, so results
  // match the converted path up to summation order). Model::load calls bind_affine once per weight; afterwards
  // linear / linear_multi / embed / logits receive the codes tensor (dtype U32) where they normally get the bf16
  // weight and look the scales up by (buf, offset). Without support, Model::load converts to bf16 itself.
  virtual bool supports_q8() const { return false; }
  virtual void bind_affine(const Tensor& codes, const Tensor& scales, const Tensor& biases) {
    (void)codes; (void)scales; (void)biases;
  }
  virtual void sync() = 0;                         // wait for all enqueued work
  virtual void copy(const Tensor& src, Tensor& dst, size_t src_off = 0, size_t dst_off = 0, size_t bytes = 0) = 0;
  // Host access (tests, logits): read back after sync.
  virtual void read(const Tensor& t, void* dst, size_t bytes, size_t off = 0) = 0;
  virtual void write(Tensor& t, const void* src, size_t bytes, size_t off = 0) = 0;

  // Ops.
  virtual void embed(const Tensor& ids_i32, const Tensor& table_bf16, Tensor& out_bf16) = 0;
  // out[t] = x[t] * rsqrt(mean(x[t]^2) + eps) * gamma (gamma may be invalid = none); rows = numel/last_dim
  // where last_dim = `dim`. fp32 math, out dtype f32 or bf16.
  virtual void rms_norm(const Tensor& x, const Tensor* gamma, float eps, int64_t dim, Tensor& out) = 0;
  // out[T,N] = x[T,K] @ w[N,K]^T. x bf16 or f32 (f32 x: w promoted to f32), fp32 accumulate, out bf16 or f32.
  virtual void linear(const Tensor& x, const Tensor& w, Tensor& out) = 0;
  // Same as several linear() calls on one input (B may fuse); results identical to separate calls.
  virtual void linear_multi(const Tensor& x, const std::vector<const Tensor*>& ws, const std::vector<Tensor*>& outs) {
    for (size_t i = 0; i < ws.size(); ++i) linear(x, *ws[i], *outs[i]);
  }
  // Attention residual mix. states = [blocks..., partial] (each [T,D], bf16 or f32), keys = rms_norm(states, no
  // gamma, eps 1e-6) fp32, scores[t,b,q] = keys[t,b] . query[q] * D^-0.5, softmax over b, out[t,q] = sum_b w * keys.
  // query f32 [nq,D]; out f32 [nq,T,D] (each query's result contiguous).
  virtual void attn_residual(const std::vector<const Tensor*>& states, const Tensor& query, Tensor& out) = 0;
  // Loop messages. bank = rms_norm(states) fp32 (states f32/bf16 [T,D] each); out[t,q] = softmax_b(query[q] .
  // bank[t,b] * D^-0.5) @ bank[t]. query f32 [4,D]; out f32 [4,T,D].
  virtual void loop_messages(const std::vector<const Tensor*>& states, const Tensor& query, Tensor& out) = 0;
  // MessageController: pair = [rms(x), rms(m)] (eps = 1e-6, no gamma) -> bf16; h = bf16(pair @ down^T) -> f32;
  // f = tanh(h @ forget^T), g = tanh(h @ gain^T) (fp32, weights promoted); out = (1-f)*x + g*m. x, m f32 [T,D];
  // out f32 [T,D] (may alias x).
  virtual void controller(const Tensor& x, const Tensor& m, const Tensor& down,
                          const Tensor& forget, const Tensor& gain, Tensor& out) = 0;
  // Causal depthwise conv K=4 over [state(3 rows, f32); x] + residual: out[t,c] = bf16((sum_j xin[t-3+j,c] *
  // w[c,j] + x[t,c]) / sqrt(5)); then state = last 3 inputs (f32). w bf16 [C,4]. x, out bf16 [T,C].
  // hist (optional, f32 [T,3,C]): hist[t] = the state after row t (for speculative rollback).
  virtual void short_conv(const Tensor& x, const Tensor& w, Tensor& state, Tensor& out, Tensor* hist = nullptr) = 0;
  // Dynamic conv (global layers' k/v): wts[t,h,c,lag] = static[lag, h*128+c] + dyn[t, h*4+lag] (fp32);
  // out = bf16((sum_lag xin[t-lag] * wts[..lag] + x[t]) / sqrt(9)), accumulated lag 0..3 in order; state as above.
  // x bf16 [T,1024]; dyn bf16 [T,32]; static bf16 [4,1024].
  virtual void dyn_conv(const Tensor& x, const Tensor& dyn, const Tensor& static_w, Tensor& state, Tensor& out,
                        Tensor* hist = nullptr) = 0;
  // y = x * scale (fp32, scale bf16 [H,d] or none), y *= min(thr / (|y| + 1e-6), 1) over last dim d -> bf16.
  // If tau (f32 [T]) is valid: out = bf16(float(that bf16) * tau[t]). x [T,H,d] bf16, out bf16.
  virtual void l2_clip(const Tensor& x, const Tensor* scale, float thr, const Tensor* tau, Tensor& out) = 0;
  // out = bf16(float(a) * sigmoid(float(g)))
  virtual void sigmoid_gate(const Tensor& a, const Tensor& g, Tensor& out) = 0;
  // out = float(a) + float(b) in out's dtype (a, b any of f32/bf16, same numel).
  virtual void add(const Tensor& a, const Tensor& b, Tensor& out) = 0;
  // dtype conversion copy (f32<->bf16), round-to-nearest-even.
  virtual void cast(const Tensor& a, Tensor& out) = 0;
  // Attention (see AttnArgs). out bf16 [T, 24*128].
  virtual void attention(const AttnArgs&, Tensor& out) = 0;
  // Write k_new/v_new into the cache slots for positions kv.len.. (ring: modulo cap).
  virtual void kv_append(const Tensor& k_new, const Tensor& v_new, KvView& kv) = 0;
  // Router: s = sigmoid(f32(x) @ f32(gate_w)^T); idx = top-k of (s + bias); w = s[idx] / (sum + 1e-20) * scale.
  // idx i32 [T,k], w f32 [T,k].
  virtual void router(const Tensor& x, const MoeWeights& m, int topk, float scale, Tensor& idx, Tensor& w) = 0;
  // MoE: for each route: g = bf16(x @ Wg^T), u = bf16(x @ Wu^T) (dequantized, fp32 acc);
  // a = bf16(tanh(g/4)*4*sigmoid(g)*tanh(u/25)*25); d = bf16(a @ Wd^T); out = bf16(sum_k f32(d_k) * w_k).
  virtual void moe(const Tensor& x, const Tensor& idx, const Tensor& w, const MoeWeights& m, Tensor& out) = 0;
  // ---- Fused ops (B). Defaults are the exact primitive sequences model.cpp used before; MetalBackend overrides.
  // Local-layer attention prep: k/v short convs, l2_clip q (q_scale), l2_clip k, rms_norm v (no gamma, dim 128),
  // l2_clip r (rel_scale), in that order. kc, vc: bf16 [T, KVH*HD] conv outputs (scratch); k3, v3 outputs.
  virtual void attn_prep_local(const Tensor& kraw, const Tensor& vraw, const Tensor& k_conv, const Tensor& v_conv,
                               Tensor& kstate, Tensor& vstate, Tensor* khist, Tensor* vhist, Tensor& q3,
                               const Tensor& q_scale, Tensor& r3, const Tensor& rel_scale, float qk_clip,
                               float rel_clip, float eps, Tensor& kc, Tensor& vc, Tensor& k3, Tensor& v3) {
    short_conv(kraw, k_conv, kstate, kc, khist);
    short_conv(vraw, v_conv, vstate, vc, vhist);
    l2_clip(q3, &q_scale, qk_clip, nullptr, q3);
    Tensor kc3 = kc;
    kc3.set_shape({k3.dim(0), k3.dim(1), k3.dim(2)});
    l2_clip(kc3, nullptr, qk_clip, nullptr, k3);
    rms_norm(vc, nullptr, eps, k3.dim(2), v3);
    l2_clip(r3, &rel_scale, rel_clip, nullptr, r3);
  }
  // attn_residual(states, query (nq = 1), mixed) then rms_norm(mixed, &gamma, eps, D, normed).
  virtual void attn_residual_norm(const std::vector<const Tensor*>& states, const Tensor& query, Tensor& mixed,
                                  const Tensor& gamma, float eps, Tensor& normed) {
    attn_residual(states, query, mixed);
    rms_norm(mixed, &gamma, eps, mixed.dim(-1), normed);
  }
  // n controller() calls in order (the attn/k/v parts of a looped layer).
  virtual void controller_multi(int n, const Tensor* xs, const Tensor* ms, const Tensor* downs, const Tensor* forgets,
                                const Tensor* gains, Tensor* outs) {
    for (int i = 0; i < n; ++i) controller(xs[i], ms[i], downs[i], forgets[i], gains[i], outs[i]);
  }
  // n rms_norm() calls in order.
  virtual void rms_norm_multi(int n, const Tensor* xs, const Tensor* const* gammas, float eps, int64_t dim,
                              Tensor* outs) {
    for (int i = 0; i < n; ++i) rms_norm(xs[i], gammas[i], eps, dim, outs[i]);
  }
  // conv_add(x, w, state, hist, other, tmp, res) then attn_residual_norm(states (last = res), query, mixed, gamma,
  // eps, normed).
  virtual void conv_add_mix_norm(const Tensor& x, const Tensor& w, Tensor& state, Tensor* hist, const Tensor& other,
                                 Tensor& tmp, Tensor& res, const std::vector<const Tensor*>& states,
                                 const Tensor& query, Tensor& mixed, const Tensor& gamma, float eps, Tensor& normed) {
    conv_add(x, w, state, hist, other, tmp, res);
    attn_residual_norm(states, query, mixed, gamma, eps, normed);
  }
  // attention() then sigmoid_gate(out, gate, out).
  virtual void attention_gated(const AttnArgs& a, const Tensor& gate, Tensor& out) {
    attention(a, out);
    sigmoid_gate(out, gate, out);
  }
  // short_conv(x -> tmp bf16) then add(tmp, other, out).
  virtual void conv_add(const Tensor& x, const Tensor& w, Tensor& state, Tensor* hist, const Tensor& other,
                        Tensor& tmp, Tensor& out) {
    short_conv(x, w, state, tmp, hist);
    add(tmp, other, out);
  }

  // Logits: hb = f32(bf16(h)); z = hb @ f32(head)^T; out = tanh(z/30)*30, first `vocab` columns. h f32 [T,D].
  virtual bool has_chain(int64_t T) const { (void)T; return false; }
  virtual void chain_front(const ChainFrontArgs&) {}
  virtual void chain_mid(const ChainMidArgs&) {}
  virtual void logits(const Tensor& h, const Tensor& head, int64_t vocab, float softcap, Tensor& out) = 0;
};

std::unique_ptr<Backend> make_cpu_backend();
std::unique_ptr<Backend> make_metal_backend();  // B

}  // namespace ink
