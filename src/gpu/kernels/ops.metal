// Simple correct kernels for the Backend ops (first pass). Owner: B. Math and rounding points follow
// src/model/backend.h. dt codes: 0 = f32, 1 = bf16.

// ---------------------------------------------------------------- elementwise
struct EwParams {
  uint n;
  uint dt_a, dt_b, dt_out;
};

kernel void ew_add(device const void* a [[buffer(0)]], device const void* b [[buffer(1)]],
                   device void* out [[buffer(2)]], constant EwParams& p [[buffer(3)]],
                   uint i [[thread_position_in_grid]]) {
  if (i >= p.n) return;
  stv(out, p.dt_out, i, ldv(a, p.dt_a, i) + ldv(b, p.dt_b, i));
}

kernel void ew_cast(device const void* a [[buffer(0)]], device void* out [[buffer(2)]],
                    constant EwParams& p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (i >= p.n) return;
  stv(out, p.dt_out, i, ldv(a, p.dt_a, i));
}

kernel void ew_sigmoid_gate(device const void* a [[buffer(0)]], device const void* g [[buffer(1)]],
                            device void* out [[buffer(2)]], constant EwParams& p [[buffer(3)]],
                            uint i [[thread_position_in_grid]]) {
  if (i >= p.n) return;
  float gv = ldv(g, p.dt_b, i);
  float s = 1.0f / (1.0f + exp(-gv));
  stv(out, p.dt_out, i, ldv(a, p.dt_a, i) * s);
}

// ---------------------------------------------------------------- embed
struct EmbedParams {
  uint T, D;
};
kernel void embed_rows(device const int* ids [[buffer(0)]], device const ushort* table [[buffer(1)]],
                       device ushort* out [[buffer(2)]], constant EmbedParams& p [[buffer(3)]],
                       uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= p.D || gid.y >= p.T) return;
  out[gid.y * p.D + gid.x] = table[size_t(ids[gid.y]) * p.D + gid.x];
}

// ---------------------------------------------------------------- rms_norm: one threadgroup (256) per row
struct RmsParams {
  uint rows, dim;
  float eps;
  uint has_gamma, dt_x, dt_out;
};
kernel void rms_norm_rows(device const void* x [[buffer(0)]], device const ushort* gamma [[buffer(1)]],
                          device void* out [[buffer(2)]], constant RmsParams& p [[buffer(3)]],
                          uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[RT / 32];
  uint base = row * p.dim;
  float ss = 0;
  for (uint i = tid; i < p.dim; i += RT) {
    float v = ldv(x, p.dt_x, base + i);
    ss += v * v;
  }
  ss = tg_sum(ss, scratch, tid, lane, sg);
  float inv = metal::precise::rsqrt(ss / float(p.dim) + p.eps);
  for (uint i = tid; i < p.dim; i += RT) {
    float v = ldv(x, p.dt_x, base + i) * inv;
    if (p.dt_x == 1) v = round_bf(v);  // MLX: out = T(w * T(x * inv)), T = input dtype
    if (p.has_gamma) v = v * bf2f(gamma[i]);
    stv(out, p.dt_out, base + i, v);
  }
}

// ---------------------------------------------------------------- generic linear (any T): out[T,N] = x @ w^T
// One simdgroup per (t, 4 rows). Used for odd shapes; the fast paths are matvec_bf16 / gemm_bf16.
struct LinParams {
  uint T, N, K;
  uint dt_x, dt_out;
};
kernel void linear_generic(device const ushort* w [[buffer(0)]], device const void* x [[buffer(1)]],
                           device void* out [[buffer(2)]], constant LinParams& p [[buffer(3)]],
                           uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  uint t = tg.y;
  for (uint r = 0; r < 4; r++) {
    uint n = tg.x * 4 + r;
    if (n >= p.N) return;
    float acc = 0;
    for (uint k = lane; k < p.K; k += 32) acc = fma(bf2f(w[size_t(n) * p.K + k]), ldv(x, p.dt_x, t * p.K + k), acc);
    acc = simd_sum(acc);
    if (lane == 0) stv(out, p.dt_out, t * p.N + n, acc);
  }
}

// ---------------------------------------------------------------- attention residual / loop messages
// states: up to 8 buffers [T,D] (dt per state). keys = rms(states) fp32 (no gamma, eps 1e-6).
// out[q,t,:] = sum_b softmax_b(keys[t,b] . query[q] * D^-0.5) * keys[t,b]. One threadgroup (256) per t.
struct ResParams {
  uint T, D, nb, nq;
  uint dt[8];
};
kernel void attn_residual_mix(device const void* s0 [[buffer(0)]], device const void* s1 [[buffer(1)]],
                              device const void* s2 [[buffer(2)]], device const void* s3 [[buffer(3)]],
                              device const void* s4 [[buffer(4)]], device const void* s5 [[buffer(5)]],
                              device const void* s6 [[buffer(6)]], device const void* s7 [[buffer(7)]],
                              device const float* query [[buffer(8)]], device float* out [[buffer(9)]],
                              constant ResParams& p [[buffer(10)]], uint t [[threadgroup_position_in_grid]],
                              uint tid [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]]) {
  // Single pass: each thread accumulates sum(x_b^2) and x_b . q for all b, q; one batched reduction.
  threadgroup float red[RT / 32][8 * 5];
  threadgroup float inv[8];
  threadgroup float wts[8][4];
  device const void* st[8] = {s0, s1, s2, s3, s4, s5, s6, s7};
  uint base = t * p.D;
  float acc[8][5];  // [b][0] = sum of squares, [b][1 + q] = raw dot with query q (nq <= 4)
  for (uint b = 0; b < 8; b++)
    for (uint j = 0; j < 5; j++) acc[b][j] = 0;
  for (uint i = tid; i < p.D; i += RT) {
    float qv[4];
    for (uint q = 0; q < p.nq; q++) qv[q] = query[q * p.D + i];
    for (uint b = 0; b < p.nb; b++) {
      float v = ldv(st[b], p.dt[b], base + i);
      acc[b][0] = fma(v, v, acc[b][0]);
      for (uint q = 0; q < p.nq; q++) acc[b][1 + q] = fma(v, qv[q], acc[b][1 + q]);
    }
  }
  for (uint b = 0; b < p.nb; b++)
    for (uint j = 0; j <= p.nq; j++) {
      float v = simd_sum(acc[b][j]);
      if (lane == 0) red[sg][b * 5 + j] = v;
    }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < p.nb * 5) {
    float v = 0;
    for (uint i = 0; i < RT / 32; i++) v += red[i][tid];
    red[0][tid] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float scale = metal::precise::rsqrt(float(p.D));
  if (tid < p.nb) inv[tid] = metal::precise::rsqrt(red[0][tid * 5] / float(p.D) + 1e-6f);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < p.nq) {
    float m = -INFINITY;
    for (uint b = 0; b < p.nb; b++) {
      float sc = red[0][b * 5 + 1 + tid] * inv[b] * scale;
      wts[b][tid] = sc;
      m = max(m, sc);
    }
    float s = 0;
    for (uint b = 0; b < p.nb; b++) {
      float e = exp(wts[b][tid] - m);
      wts[b][tid] = e;
      s += e;
    }
    for (uint b = 0; b < p.nb; b++) wts[b][tid] /= s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < p.D; i += RT) {
    float k[8];
    for (uint b = 0; b < p.nb; b++) k[b] = ldv(st[b], p.dt[b], base + i) * inv[b];
    for (uint q = 0; q < p.nq; q++) {
      float o = 0;
      for (uint b = 0; b < p.nb; b++) o += wts[b][q] * k[b];
      out[(q * p.T + t) * p.D + i] = o;
    }
  }
}

// ---------------------------------------------------------------- message controller
// pair = bf16([rms(x), rms(m)]); h = bf16(pair @ down^T) (rank R); f = tanh(h @ forget^T); g = tanh(h @ gain^T)
// out = (1-f)*x + g*m. One threadgroup (256) per token. x, m, out f32.
struct CtrlParams {
  uint T, D, R, m_stride;
};
kernel void controller_mix(device const float* x [[buffer(0)]], device const float* m [[buffer(1)]],
                           device const ushort* down [[buffer(2)]], device const ushort* forget [[buffer(3)]],
                           device const ushort* gain [[buffer(4)]], device float* out [[buffer(5)]],
                           constant CtrlParams& p [[buffer(6)]], uint t [[threadgroup_position_in_grid]],
                           uint tid [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[RT / 32];
  threadgroup float pair[2 * 2304];
  threadgroup float h[64];
  device const float* xr = x + t * p.D;
  device const float* mr = m + t * p.m_stride;
  float sx = 0, sm = 0;
  for (uint i = tid; i < p.D; i += RT) {
    sx += xr[i] * xr[i];
    sm += mr[i] * mr[i];
  }
  sx = tg_sum(sx, scratch, tid, lane, sg);
  sm = tg_sum(sm, scratch, tid, lane, sg);
  float ix = metal::precise::rsqrt(sx / float(p.D) + 1e-6f), im = metal::precise::rsqrt(sm / float(p.D) + 1e-6f);
  for (uint i = tid; i < p.D; i += RT) {
    pair[i] = round_bf(xr[i] * ix);
    pair[p.D + i] = round_bf(mr[i] * im);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // 8 simdgroups x R/8 rows each
  for (uint r = sg; r < p.R; r += RT / 32) {
    float a = 0;
    for (uint k = lane; k < 2 * p.D; k += 32) a = fma(bf2f(down[r * 2 * p.D + k]), pair[k], a);
    a = simd_sum(a);
    if (lane == 0) h[r] = round_bf(a);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < p.D; i += RT) {
    float f = 0, g = 0;
    for (uint r = 0; r < p.R; r++) {
      f = fma(h[r], bf2f(forget[i * p.R + r]), f);
      g = fma(h[r], bf2f(gain[i * p.R + r]), g);
    }
    f = precise::tanh(f);
    g = precise::tanh(g);
    float xv = xr[i];
    out[t * p.D + i] = (1 - f) * xv + g * mr[i];
  }
}

// ---------------------------------------------------------------- short conv (K = 4) and dynamic conv
// One thread per channel; walks the T rows in order and updates the fp32 state (last 3 inputs).
struct ConvParams {
  uint T, C;
  uint heads;  // dyn conv: kv heads (C / heads channels each)
  uint has_hist;
};
kernel void short_conv4(device const ushort* x [[buffer(0)]], device const ushort* w [[buffer(1)]],
                        device float* state [[buffer(2)]], device ushort* out [[buffer(3)]],
                        constant ConvParams& p [[buffer(4)]], device float* hist [[buffer(5)]],
                        uint c [[thread_position_in_grid]]) {
  if (c >= p.C) return;
  float s0 = state[0 * p.C + c], s1 = state[1 * p.C + c], s2 = state[2 * p.C + c];
  float w0 = bf2f(w[c * 4 + 0]), w1 = bf2f(w[c * 4 + 1]), w2 = bf2f(w[c * 4 + 2]), w3 = bf2f(w[c * 4 + 3]);
  const float div = 2.2360679774997896f;  // float(sqrt(5))
  for (uint t = 0; t < p.T; t++) {
    float xv = bf2f(x[t * p.C + c]);
    float v = 0;
    v = v + s0 * w0;
    v = v + s1 * w1;
    v = v + s2 * w2;
    v = v + xv * w3;
    out[t * p.C + c] = f2bf((v + xv) / div);
    s0 = s1;
    s1 = s2;
    s2 = xv;
    if (p.has_hist) {
      hist[(t * 3 + 0) * p.C + c] = s0;
      hist[(t * 3 + 1) * p.C + c] = s1;
      hist[(t * 3 + 2) * p.C + c] = s2;
    }
  }
  state[0 * p.C + c] = s0;
  state[1 * p.C + c] = s1;
  state[2 * p.C + c] = s2;
}

// wts[t,h,c,lag] = static[lag, h*128+c] + dyn[t, h*4+lag]; out = bf16((sum_lag xin[t-lag]*wts[lag] + x[t]) / 3)
kernel void dyn_conv4(device const ushort* x [[buffer(0)]], device const ushort* dyn [[buffer(1)]],
                      device const ushort* static_w [[buffer(2)]], device float* state [[buffer(3)]],
                      device ushort* out [[buffer(4)]], constant ConvParams& p [[buffer(5)]],
                      device float* histo [[buffer(6)]], uint c [[thread_position_in_grid]]) {
  if (c >= p.C) return;
  uint h = c / (p.C / p.heads);
  float hist[4];  // hist[lag] = xin[t - lag]
  hist[3] = state[0 * p.C + c];
  hist[2] = state[1 * p.C + c];
  hist[1] = state[2 * p.C + c];
  float sw[4];
  for (uint l = 0; l < 4; l++) sw[l] = bf2f(static_w[l * p.C + c]);
  for (uint t = 0; t < p.T; t++) {
    float xv = bf2f(x[t * p.C + c]);
    hist[0] = xv;
    float o = 0;
    for (uint lag = 0; lag < 4; lag++) {
      float wt = sw[lag] + bf2f(dyn[t * (p.heads * 4) + h * 4 + lag]);
      o = o + hist[lag] * wt;
    }
    out[t * p.C + c] = f2bf((o + xv) / 3.0f);
    hist[3] = hist[2];
    hist[2] = hist[1];
    hist[1] = hist[0];
    if (p.has_hist) {
      histo[(t * 3 + 0) * p.C + c] = hist[3];
      histo[(t * 3 + 1) * p.C + c] = hist[2];
      histo[(t * 3 + 2) * p.C + c] = hist[1];
    }
  }
  state[0 * p.C + c] = hist[3];
  state[1 * p.C + c] = hist[2];
  state[2 * p.C + c] = hist[1];
}

// ---------------------------------------------------------------- l2 clip: one simdgroup per (t, head)
struct ClipParams {
  uint rows, d, H;  // rows = T*H
  float thr;
  uint has_scale, has_tau;
};
kernel void l2_clip_rows(device const ushort* x [[buffer(0)]], device const ushort* scale [[buffer(1)]],
                         device const float* tau [[buffer(2)]], device ushort* out [[buffer(3)]],
                         constant ClipParams& p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
  if (row >= p.rows) return;
  uint h = row % p.H, t = row / p.H;
  float v[4];
  float ss = 0;
  for (uint j = 0; j < 4; j++) {
    uint i = lane + 32 * j;
    float y = 0;
    if (i < p.d) {
      y = bf2f(x[row * p.d + i]);
      if (p.has_scale) y = y * bf2f(scale[h * p.d + i]);
    }
    v[j] = y;
    ss += y * y;
  }
  ss = simd_sum(ss);
  float norm = precise::sqrt(ss);
  float f = min(p.thr / (norm + 1e-6f), 1.0f);
  for (uint j = 0; j < 4; j++) {
    uint i = lane + 32 * j;
    if (i >= p.d) continue;
    float y = round_bf(v[j] * f);
    if (p.has_tau) y = y * tau[t];
    out[row * p.d + i] = f2bf(y);
  }
}

// ---------------------------------------------------------------- KV append
struct KvAppendParams {
  uint T, heads, cap, pos0, ring;
};
kernel void kv_append_rows(device const ushort* k_new [[buffer(0)]], device const ushort* v_new [[buffer(1)]],
                           device ushort* k [[buffer(2)]], device ushort* v [[buffer(3)]],
                           constant KvAppendParams& p [[buffer(4)]], uint3 gid [[thread_position_in_grid]]) {
  uint d = gid.x, h = gid.y, t = gid.z;
  if (d >= 128 || h >= p.heads || t >= p.T) return;
  uint pos = p.pos0 + t;
  uint slot = p.ring ? pos % p.cap : pos;
  size_t dst = (size_t(h) * p.cap + slot) * 128 + d;
  size_t src = (size_t(t) * p.heads + h) * 128 + d;
  k[dst] = k_new[src];
  v[dst] = v_new[src];
}

// ---------------------------------------------------------------- attention (generic, flash-decoding style)
// One threadgroup of 8 simdgroups per (query t, head h). Keys: cache positions [lo, len) then new keys
// [len, len+t]. Each simdgroup handles every 8th key with an online softmax; lanes hold 4 of the 128 dims.
struct AttnParams {
  uint T, H, KVH;
  uint cap, len, ring;
  uint window;  // 0 = global
  uint rel_extent;
};
kernel void attention_generic(device const ushort* q [[buffer(0)]], device const ushort* k_new [[buffer(1)]],
                              device const ushort* v_new [[buffer(2)]], device const ushort* r [[buffer(3)]],
                              device const ushort* rel_proj [[buffer(4)]], device const ushort* sink [[buffer(5)]],
                              device const ushort* kc [[buffer(6)]], device const ushort* vc [[buffer(7)]],
                              device ushort* out [[buffer(8)]], constant AttnParams& p [[buffer(9)]],
                              uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]]) {
  uint t = tg.x, h = tg.y, kvh = h / (p.H / p.KVH);
  uint qpos = p.len + t;
  float qv[4];
  for (uint j = 0; j < 4; j++) qv[j] = bf2f(q[(t * p.H + h) * 128 + lane * 4 + j]);
  float rv[16];
  for (uint d = 0; d < 16; d++) rv[d] = bf2f(r[(t * p.H + h) * 16 + d]);
  const float scale = 0.08838834764831845f;  // 128^-0.5
  uint lo = 0;
  if (p.window && qpos + 1 > p.window) lo = qpos + 1 - p.window;  // keys with dist < window
  float m = -INFINITY, l = 0, acc[4] = {0, 0, 0, 0};
  for (uint pos = lo + sg; pos <= qpos; pos += 8) {
    device const ushort* kp;
    device const ushort* vp;
    if (pos < p.len) {
      uint slot = p.ring ? pos % p.cap : pos;
      kp = kc + (size_t(kvh) * p.cap + slot) * 128;
      vp = vc + (size_t(kvh) * p.cap + slot) * 128;
    } else {
      uint tn = pos - p.len;
      kp = k_new + (size_t(tn) * p.KVH + kvh) * 128;
      vp = v_new + (size_t(tn) * p.KVH + kvh) * 128;
    }
    float s = 0;
    for (uint j = 0; j < 4; j++) s = fma(qv[j], bf2f(kp[lane * 4 + j]), s);
    s = simd_sum(s) * scale;
    uint dist = qpos - pos;
    if (dist < p.rel_extent) {
      float b = 0;
      for (uint d = 0; d < 16; d++) b = fma(rv[d], bf2f(rel_proj[d * p.rel_extent + dist]), b);
      s += round_bf(b);
    }
    float mn = max(m, s);
    float c = exp(m - mn), e = exp(s - mn);
    l = l * c + e;
    for (uint j = 0; j < 4; j++) acc[j] = acc[j] * c + e * bf2f(vp[lane * 4 + j]);
    m = mn;
  }
  threadgroup float tm[8], tl[8];
  threadgroup float tacc[8][128];
  if (lane == 0) {
    tm[sg] = m;
    tl[sg] = l;
  }
  for (uint j = 0; j < 4; j++) tacc[sg][lane * 4 + j] = acc[j];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float sinkv = bf2f(sink[t * p.H + h]);
    float M = sinkv;
    for (uint i = 0; i < 8; i++) M = max(M, tm[i]);
    float L = exp(sinkv - M);  // sink: logit with a zero value vector
    float o[4] = {0, 0, 0, 0};
    for (uint i = 0; i < 8; i++) {
      if (tl[i] == 0) continue;
      float c = exp(tm[i] - M);
      L += tl[i] * c;
      for (uint j = 0; j < 4; j++) o[j] += tacc[i][lane * 4 + j] * c;
    }
    for (uint j = 0; j < 4; j++) out[(t * p.H + h) * 128 + lane * 4 + j] = f2bf(o[j] / L);
  }
}

// ---------------------------------------------------------------- router: one threadgroup (256) per token
// s = sigmoid(f32(x) @ f32(gate_w)^T); idx = top-k of (s + bias) (descending); w = s[idx] / (sum + 1e-20) * scale
struct RouterParams {
  uint T, D, E, topk, dt_x;
  float scale;
};
kernel void router_topk(device const void* x [[buffer(0)]], device const ushort* gw [[buffer(1)]],
                        device const float* bias [[buffer(2)]], device int* idx [[buffer(3)]],
                        device float* wout [[buffer(4)]], constant RouterParams& p [[buffer(5)]],
                        uint t [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float sc[128];
  for (uint e = sg; e < p.E; e += RT / 32) {
    float a = 0;
    for (uint k = lane; k < p.D; k += 32) a = fma(ldv(x, p.dt_x, t * p.D + k), bf2f(gw[e * p.D + k]), a);
    a = simd_sum(a);
    if (lane == 0) sc[e] = 1.0f / (1.0f + exp(-a));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    bool used[128];
    for (uint e = 0; e < p.E; e++) used[e] = false;
    int sel[16];
    float sum = 0;
    for (uint k = 0; k < p.topk; k++) {
      int best = -1;
      float bv = -INFINITY;
      for (uint e = 0; e < p.E; e++) {
        float v = sc[e] + bias[e];
        if (!used[e] && v > bv) {
          bv = v;
          best = int(e);
        }
      }
      used[best] = true;
      sel[k] = best;
      sum += sc[best];
    }
    for (uint k = 0; k < p.topk; k++) {
      idx[t * p.topk + k] = sel[k];
      wout[t * p.topk + k] = sc[sel[k]] / (sum + 1e-20f) * p.scale;
    }
  }
}

// ---------------------------------------------------------------- quantized experts
// NVFP4: weight u32 [E,N,K/8] nibbles low first (e2m1), scales u8 e4m3 [E,N,K/16], global f32 [E].
// MXFP8: weight u32 [E,N,K/4] e4m3 bytes, scales u8 e8m0 [E,N,K/32].
constant float FP4_LUT[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                              -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
// One simdgroup per (route, 4 output rows). Each lane takes one group (16 or 32 values) per step, sums
// x*q within the group in fp32, multiplies by the group scale, accumulates; global scale at the end.
// x: bf16 [routes, K] with row = route / x_div (x_div = topk for gate/up from x, 1 for down from a).
struct QmvParams {
  uint N, K, routes, x_div, mode;  // mode 0 nvfp4, 1 mxfp8
};
kernel void qmv_experts(device const uint* w [[buffer(0)]], device const uchar* sc [[buffer(1)]],
                        device const float* gs [[buffer(2)]], device const ushort* x [[buffer(3)]],
                        device const int* idx [[buffer(4)]], device ushort* out [[buffer(5)]],
                        constant QmvParams& p [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  uint route = tg.y;
  uint e = uint(idx[route]);
  device const ushort* xr = x + size_t(route / p.x_div) * p.K;
  uint row0 = (tg.x * 2 + sg) * 4;
  uint gsz = p.mode == 0 ? 16 : 32;
  uint ngroups = p.K / gsz;
  for (uint rr = 0; rr < 4; rr++) {
    uint n = row0 + rr;
    if (n >= p.N) break;
    size_t row = size_t(e) * p.N + n;
    float acc = 0;
    for (uint g = lane; g < ngroups; g += 32) {
      float part = 0;
      if (p.mode == 0) {
        device const uint* wp = w + row * (p.K / 8) + g * 2;
        for (uint j = 0; j < 2; j++) {
          uint word = wp[j];
          for (uint i = 0; i < 8; i++) part += bf2f(xr[g * 16 + j * 8 + i]) * FP4_LUT[(word >> (4 * i)) & 15u];
        }
        acc += part * e4m3_to_f(sc[row * ngroups + g]);
      } else {
        device const uint* wp = w + row * (p.K / 4) + g * 8;
        for (uint j = 0; j < 8; j++) {
          uint word = wp[j];
          for (uint i = 0; i < 4; i++) part += bf2f(xr[g * 32 + j * 4 + i]) * e4m3_to_f((word >> (8 * i)) & 255u);
        }
        acc += part * e8m0_to_f(sc[row * ngroups + g]);
      }
    }
    acc = simd_sum(acc);
    if (p.mode == 0) acc *= gs[e];
    if (lane == 0) out[size_t(route) * p.N + n] = f2bf(acc);
  }
}

// a = bf16(tanh(g/4)*4*sigmoid(g) * tanh(u/25)*25) from bf16 g, u
kernel void softcap_swiglu(device const ushort* g [[buffer(0)]], device const ushort* u [[buffer(1)]],
                           device ushort* a [[buffer(2)]], constant uint& n [[buffer(3)]],
                           uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  float gv = bf2f(g[i]), uv = bf2f(u[i]);
  float v = precise::tanh(gv / 4.0f) * 4.0f * (1.0f / (1.0f + exp(-gv))) * precise::tanh(uv / 25.0f) * 25.0f;
  a[i] = f2bf(v);
}

// out[t, :] = bf16(sum_k f32(d[t,k,:]) * w[t,k])
struct CombineParams {
  uint T, D, topk;
};
kernel void moe_combine(device const ushort* d [[buffer(0)]], device const float* w [[buffer(1)]],
                        device ushort* out [[buffer(2)]], constant CombineParams& p [[buffer(3)]],
                        uint2 gid [[thread_position_in_grid]]) {
  uint i = gid.x, t = gid.y;
  if (i >= p.D || t >= p.T) return;
  float s = 0;
  for (uint k = 0; k < p.topk; k++) s += bf2f(d[(size_t(t) * p.topk + k) * p.D + i]) * w[t * p.topk + k];
  out[t * p.D + i] = f2bf(s);
}

// ---------------------------------------------------------------- logits softcap (in place, f32)
struct SoftcapParams {
  uint n;
  float cap;
};
kernel void softcap_f32(device float* z [[buffer(0)]], constant SoftcapParams& p [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
  if (i >= p.n) return;
  z[i] = precise::tanh(z[i] / p.cap) * p.cap;
}

// f32 -> bf16 -> f32 rounding copy (logits input)
kernel void round_bf_f32(device const float* a [[buffer(0)]], device float* out [[buffer(1)]],
                         constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  out[i] = round_bf(a[i]);
}

// ---------------------------------------------------------------- parallel router: scores via matvec (f32 out),
// then one threadgroup of E threads per token: sigmoid, rank-based top-k (descending s+bias, ties -> lower index).
kernel void router_select(device const float* logit [[buffer(0)]], device const float* bias [[buffer(1)]],
                          device int* idx [[buffer(2)]], device float* wout [[buffer(3)]],
                          constant RouterParams& p [[buffer(4)]], uint t [[threadgroup_position_in_grid]],
                          uint e [[thread_index_in_threadgroup]]) {
  threadgroup float sc[128], key[128];
  threadgroup int sel[16];
  if (e < p.E) {
    float s = 1.0f / (1.0f + exp(-logit[t * p.E + e]));
    sc[e] = s;
    key[e] = s + bias[e];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (e < p.E) {
    float v = key[e];
    uint rank = 0;
    for (uint j = 0; j < p.E; j++) {
      float u = key[j];
      rank += (u > v || (u == v && j < e)) ? 1u : 0u;
    }
    if (rank < p.topk) sel[rank] = int(e);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (e == 0) {
    float sum = 0;
    for (uint k = 0; k < p.topk; k++) sum += sc[sel[k]];
    for (uint k = 0; k < p.topk; k++) {
      idx[t * p.topk + k] = sel[k];
      wout[t * p.topk + k] = sc[sel[k]] / (sum + 1e-20f) * p.scale;
    }
  }
}

// ---------------------------------------------------------------- controller in three parallel steps
// 1) pair = bf16([rms(x), rms(m)]) per token (one threadgroup of 256 per token)
kernel void ctrl_pair(device const float* x [[buffer(0)]], device const float* m [[buffer(1)]],
                      device ushort* pair [[buffer(2)]], constant CtrlParams& p [[buffer(3)]],
                      uint t [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[RT / 32];
  device const float* xr = x + t * p.D;
  device const float* mr = m + t * p.m_stride;
  float sx = 0, sm = 0;
  for (uint i = tid; i < p.D; i += RT) {
    sx += xr[i] * xr[i];
    sm += mr[i] * mr[i];
  }
  sx = tg_sum(sx, scratch, tid, lane, sg);
  sm = tg_sum(sm, scratch, tid, lane, sg);
  float ix = metal::precise::rsqrt(sx / float(p.D) + 1e-6f), im = metal::precise::rsqrt(sm / float(p.D) + 1e-6f);
  for (uint i = tid; i < p.D; i += RT) {
    pair[t * 2 * p.D + i] = f2bf(xr[i] * ix);
    pair[t * 2 * p.D + p.D + i] = f2bf(mr[i] * im);
  }
}
// 2) h = bf16(pair @ down^T): matvec / gemm in the backend
// 3) out = (1 - tanh(h @ forget^T)) * x + tanh(h @ gain^T) * m, one thread per (t, i)
kernel void ctrl_out(device const float* x [[buffer(0)]], device const float* m [[buffer(1)]],
                     device const ushort* h [[buffer(2)]], device const ushort* forget [[buffer(3)]],
                     device const ushort* gain [[buffer(4)]], device float* out [[buffer(5)]],
                     constant CtrlParams& p [[buffer(6)]], uint2 gid [[thread_position_in_grid]]) {
  uint i = gid.x, t = gid.y;
  if (i >= p.D || t >= p.T) return;
  device const ushort* hr = h + t * p.R;
  float f = 0, g = 0;
  for (uint r = 0; r < p.R; r++) {
    float hv = bf2f(hr[r]);
    f = fma(hv, bf2f(forget[i * p.R + r]), f);
    g = fma(hv, bf2f(gain[i * p.R + r]), g);
  }
  f = precise::tanh(f);
  g = precise::tanh(g);
  float xv = x[t * p.D + i];
  out[t * p.D + i] = (1 - f) * xv + g * m[t * p.m_stride + i];
}
