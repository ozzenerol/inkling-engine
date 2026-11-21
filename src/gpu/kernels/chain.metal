// Chain kernels for decode and verify (T <= 8). Owner: A.
// Each replaces a run of small dependent dispatches between two big matvecs by ONE dispatch of a single
// 1024-thread threadgroup (a dependent dispatch costs about 7.5 us on this GPU, and a looped visit had ~25 of
// them). Rows are processed one after the other inside the threadgroup, so a verify row is computed exactly as
// the decode row would be. All per-token vectors live in threadgroup memory.
//
//   chain_front  [x = bf16(res + conv(moe out)) of the previous visit] -> attention-residual mix (1 query on
//                local layers, 3 on global) -> [loop passes: 3 message controllers] -> input norm(s) -> xq, xk, xv
//   chain_mid    res = conv(o_proj out) (+ x) -> [block copy] -> mix (mlp query) -> [mlp message controller]
//                -> post norm -> hn -> router scores, top-k and weights
// Math and rounding points follow backend.h (same as the separate ops; fp32 sums differ in order only).

struct ChainParams {
  uint T, D, nb;                       // nb: block states bound (excluding the in-kernel vector)
  uint has_conv, boundary, has_hist, has_msg;
  uint topk, E, R;
  float eps, route_scale;
  uint nq;                             // chain_front: 1 = local (one query), 3 = global (q, k, v queries)
};

constant uint CH_NT = 1024, CH_MAXD = 2304;

// Sums n per-thread values over the whole threadgroup; every thread gets all n totals back in v.
inline void ch_sum(thread float* v, uint n, threadgroup float* scratch, uint sg, uint lane) {
  for (uint j = 0; j < n; j++) {
    float s = simd_sum(v[j]);
    if (lane == 0) scratch[j * 32 + sg] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint j = 0; j < n; j++) {
    float r = 0;
    for (uint i = 0; i < 32; i++) r += scratch[j * 32 + i];
    v[j] = r;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Causal conv K=4 on one channel: returns bf16((conv + x) / sqrt(5)) as float, shifts the state, records hist.
inline float ch_conv(device const ushort* in, device const ushort* w, device float* state, device float* hist,
                     uint t, uint i, uint D, bool has_hist) {
  float s0 = state[i], s1 = state[D + i], s2 = state[2 * D + i];
  float xv = bf2f(in[t * D + i]);
  float v = 0;
  v = v + s0 * bf2f(w[i * 4 + 0]);
  v = v + s1 * bf2f(w[i * 4 + 1]);
  v = v + s2 * bf2f(w[i * 4 + 2]);
  v = v + xv * bf2f(w[i * 4 + 3]);
  float r = round_bf((v + xv) / 2.2360679774997896f);
  state[i] = s1;
  state[D + i] = s2;
  state[2 * D + i] = xv;
  if (has_hist) {
    hist[(t * 3 + 0) * D + i] = s1;
    hist[(t * 3 + 1) * D + i] = s2;
    hist[(t * 3 + 2) * D + i] = xv;
  }
  return r;
}

// Attention-residual mix of nb block rows (bf16, device) and the vector `loc` (threadgroup), with one query.
// dst[i] = sum_b softmax_b(key_b . q * D^-0.5) * key_b[i], key_b = state_b * rsqrt(mean(state_b^2) + 1e-6).
// dst may be loc.
inline void ch_mix(device const ushort* b0, device const ushort* b1, device const ushort* b2,
                   device const ushort* b3, uint nb, threadgroup float* loc, device const float* query,
                   threadgroup float* dst, uint t, uint D, uint tid, uint sg, uint lane,
                   threadgroup float* scratch) {
  device const ushort* bs[4] = {b0, b1, b2, b3};
  float acc[10];
  for (uint j = 0; j < 10; j++) acc[j] = 0;
  for (uint i = tid; i < D; i += CH_NT) {
    float qv = query[i];
    for (uint b = 0; b < nb; b++) {
      float v = bf2f(bs[b][t * D + i]);
      acc[2 * b] = fma(v, v, acc[2 * b]);
      acc[2 * b + 1] = fma(v, qv, acc[2 * b + 1]);
    }
    float v = loc[i];
    acc[2 * nb] = fma(v, v, acc[2 * nb]);
    acc[2 * nb + 1] = fma(v, qv, acc[2 * nb + 1]);
  }
  ch_sum(acc, 2 * (nb + 1), scratch, sg, lane);
  float inv[5], wt[5];
  float scale = precise::rsqrt(float(D)), m = -INFINITY, z = 0;
  for (uint b = 0; b <= nb; b++) {
    inv[b] = precise::rsqrt(acc[2 * b] / float(D) + 1e-6f);
    wt[b] = acc[2 * b + 1] * inv[b] * scale;
    m = max(m, wt[b]);
  }
  for (uint b = 0; b <= nb; b++) {
    wt[b] = exp(wt[b] - m);
    z += wt[b];
  }
  for (uint b = 0; b <= nb; b++) wt[b] = wt[b] / z * inv[b];
  for (uint i = tid; i < D; i += CH_NT) {
    float o = 0;
    for (uint b = 0; b < nb; b++) o += wt[b] * bf2f(bs[b][t * D + i]);
    o += wt[nb] * loc[i];
    dst[i] = o;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Loop messages are never stored: m_slot[i] = c0 * bank0[i] + c1 * bank1[i], with (c0, c1) from this.
// coef[2 * s + b] for slots first .. first + n - 1.
inline void ch_msg_coef(device const ushort* bank0, device const ushort* bank1, device const float* mq, uint first,
                        uint n, thread float* coef, uint t, uint D, uint tid, uint sg, uint lane,
                        threadgroup float* scratch) {
  float acc[8];
  for (uint j = 0; j < 8; j++) acc[j] = 0;
  for (uint i = tid; i < D; i += CH_NT) {
    float v0 = bf2f(bank0[t * D + i]), v1 = bf2f(bank1[t * D + i]);
    acc[0] = fma(v0, v0, acc[0]);
    acc[1] = fma(v1, v1, acc[1]);
    for (uint s = 0; s < n; s++) {
      float qv = mq[(first + s) * D + i];
      acc[2 + 2 * s] = fma(v0, qv, acc[2 + 2 * s]);
      acc[3 + 2 * s] = fma(v1, qv, acc[3 + 2 * s]);
    }
  }
  ch_sum(acc, 2 + 2 * n, scratch, sg, lane);
  float k0 = precise::rsqrt(acc[0] / float(D) + 1e-6f), k1 = precise::rsqrt(acc[1] / float(D) + 1e-6f);
  float scale = precise::rsqrt(float(D));
  for (uint s = 0; s < n; s++) {
    float a = acc[2 + 2 * s] * k0 * scale, b = acc[3 + 2 * s] * k1 * scale;
    float m = max(a, b);
    a = exp(a - m);
    b = exp(b - m);
    coef[2 * s] = a / (a + b) * k0;
    coef[2 * s + 1] = b / (a + b) * k1;
  }
}

// Message controller: dst = (1 - f) x + g m, f = tanh(h . forget), g = tanh(h . gain),
// h = bf16(bf16([rms(x), rms(m)]) @ down^T) (rank R = 64). xs: threadgroup; the message is written to vm
// (threadgroup) first; dst may be xs. Weights are read 8 bf16 at a time: one threadgroup has only 1024
// threads, and a single-element device load costs about 0.25 us of latency each.
inline void ch_ctrl(threadgroup float* xs, float ssx, device const ushort* bank0, device const ushort* bank1,
                    float c0, float c1, device const ushort* down, device const ushort* forget,
                    device const ushort* gain, threadgroup float* vm, threadgroup float* dst, threadgroup float* h,
                    uint t, uint D, uint R, uint tid, uint sg, uint lane, threadgroup float* scratch) {
  float ssm = 0;
  for (uint i = tid; i < D; i += CH_NT) {
    float mv = c0 * bf2f(bank0[t * D + i]) + c1 * bf2f(bank1[t * D + i]);
    vm[i] = mv;
    ssm = fma(mv, mv, ssm);
  }
  ch_sum(&ssm, 1, scratch, sg, lane);  // its barrier also publishes vm
  float invx = 1.0f / precise::sqrt(ssx / float(D) + 1e-6f), invm = 1.0f / precise::sqrt(ssm / float(D) + 1e-6f);
  // h: simdgroup sg computes rows sg * R / 32 ..; lanes split the 2 D pair elements
  uint rper = R / 32, span = 2 * D / 32;
  for (uint rr = 0; rr < rper; rr++) {
    uint r = sg * rper + rr;
    float a = 0;
    for (uint j = lane * span; j < (lane + 1) * span; j += 8) {
      float w8[8];
      unpack_bf8(*(device const uint4*)(down + size_t(r) * 2 * D + j), w8);
      for (uint e = 0; e < 8; e++) {
        uint jj = j + e;
        float pv = jj < D ? round_bf(xs[jj] * invx) : round_bf(vm[jj - D] * invm);
        a = fma(pv, w8[e], a);
      }
    }
    a = simd_sum(a);
    if (lane == 0) h[r] = round_bf(a);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < D; i += CH_NT) {
    float f = 0, g = 0;
    for (uint r = 0; r < R; r += 8) {
      float fw[8], gw[8];
      unpack_bf8(*(device const uint4*)(forget + size_t(i) * R + r), fw);
      unpack_bf8(*(device const uint4*)(gain + size_t(i) * R + r), gw);
      for (uint e = 0; e < 8; e++) {
        float hv = h[r + e];
        f = fma(hv, fw[e], f);
        g = fma(hv, gw[e], g);
      }
    }
    dst[i] = (1.0f - precise::tanh(f)) * xs[i] + precise::tanh(g) * vm[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// out = bf16(src * rsqrt(mean(src^2) + eps) * gamma); with keep, src becomes the float of that bf16 value.
inline void ch_norm(threadgroup float* src, device const ushort* gamma, device ushort* out, float eps, bool keep,
                    uint t, uint D, uint tid, uint sg, uint lane, threadgroup float* scratch) {
  float ss = 0;
  for (uint i = tid; i < D; i += CH_NT) ss = fma(src[i], src[i], ss);
  ch_sum(&ss, 1, scratch, sg, lane);
  float inv = 1.0f / precise::sqrt(ss / float(D) + eps);
  for (uint i = tid; i < D; i += CH_NT) {
    ushort b = f2bf(src[i] * inv * bf2f(gamma[i]));
    out[t * D + i] = b;
    if (keep) src[i] = bf2f(b);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void chain_front(device const ushort* mo [[buffer(0)]], device const ushort* conv_w [[buffer(1)]],
                        device float* conv_state [[buffer(2)]], device float* conv_hist [[buffer(3)]],
                        device const float* res [[buffer(4)]], device ushort* x [[buffer(5)]],
                        device const ushort* b0 [[buffer(6)]], device const ushort* b1 [[buffer(7)]],
                        device const ushort* b2 [[buffer(8)]], device const ushort* b3 [[buffer(9)]],
                        device const float* query [[buffer(10)]], device const ushort* gamma [[buffer(11)]],
                        device ushort* xq [[buffer(12)]], device ushort* xk [[buffer(13)]],
                        device ushort* xv [[buffer(14)]], device const ushort* bank0 [[buffer(15)]],
                        device const ushort* bank1 [[buffer(16)]], device const float* mq [[buffer(17)]],
                        device const ushort* down0 [[buffer(18)]], device const ushort* forget0 [[buffer(19)]],
                        device const ushort* gain0 [[buffer(20)]], device const ushort* down1 [[buffer(21)]],
                        device const ushort* forget1 [[buffer(22)]], device const ushort* gain1 [[buffer(23)]],
                        device const ushort* down2 [[buffer(24)]], device const ushort* forget2 [[buffer(25)]],
                        device const ushort* gain2 [[buffer(26)]], constant ChainParams& p [[buffer(27)]],
                        device const ushort* gamma1 [[buffer(28)]], device const ushort* gamma2 [[buffer(29)]],
                        uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float va[CH_MAXD];
  threadgroup float vb[CH_MAXD];
  threadgroup float vc[CH_MAXD];
  threadgroup float scratch[16 * 32];
  threadgroup float h[64];
  const uint D = p.D;
  for (uint t = 0; t < p.T; t++) {
    for (uint i = tid; i < D; i += CH_NT) {
      if (p.has_conv) {
        ushort nx = f2bf(ch_conv(mo, conv_w, conv_state, conv_hist, t, i, D, p.has_hist != 0) + res[t * D + i]);
        x[t * D + i] = nx;
        va[i] = bf2f(nx);
      } else {
        va[i] = bf2f(x[t * D + i]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Local layers: one mix (one query), then with messages three controllers on it. Global layers: three
    // queries, so a mix per output, each with its own norm weight. vb = mix, vc = controller output, va = x for
    // the mix and then the controller's message (x is reloaded from the device before the next mix).
    float coef[6];
    if (p.has_msg) ch_msg_coef(bank0, bank1, mq, 0, 3, coef, t, D, tid, sg, lane, scratch);
    device ushort* outs[3] = {xq, xk, xv};
    device const ushort* gammas[3] = {gamma, gamma1, gamma2};
    device const ushort* downs[3] = {down0, down1, down2};
    device const ushort* forgets[3] = {forget0, forget1, forget2};
    device const ushort* gains[3] = {gain0, gain1, gain2};
    uint nout = (p.has_msg || p.nq == 3) ? 3 : 1;
    float ssx = 0;
    for (uint c = 0; c < nout; c++) {
      if (c == 0 || p.nq == 3) {
        if (c > 0 && p.has_msg) {  // va held the previous controller's message: reload x (own elements)
          for (uint i = tid; i < D; i += CH_NT) va[i] = bf2f(x[t * D + i]);
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        ch_mix(b0, b1, b2, b3, p.nb, va, query + (p.nq == 3 ? c * D : 0), vb, t, D, tid, sg, lane, scratch);
        if (p.has_msg) {
          ssx = 0;
          for (uint i = tid; i < D; i += CH_NT) ssx = fma(vb[i], vb[i], ssx);
          ch_sum(&ssx, 1, scratch, sg, lane);
        }
      }
      if (!p.has_msg) {
        ch_norm(vb, gammas[c], outs[c], p.eps, false, t, D, tid, sg, lane, scratch);
      } else {
        ch_ctrl(vb, ssx, bank0, bank1, coef[2 * c], coef[2 * c + 1], downs[c], forgets[c], gains[c], va, vc, h, t, D,
                p.R, tid, sg, lane, scratch);
        ch_norm(vc, gammas[c], outs[c], p.eps, false, t, D, tid, sg, lane, scratch);
      }
    }
  }
}

kernel void chain_mid(device const ushort* ao [[buffer(0)]], device const ushort* conv_w [[buffer(1)]],
                      device float* conv_state [[buffer(2)]], device float* conv_hist [[buffer(3)]],
                      device const ushort* x [[buffer(4)]], device ushort* blk [[buffer(5)]],
                      device const ushort* b0 [[buffer(6)]], device const ushort* b1 [[buffer(7)]],
                      device const ushort* b2 [[buffer(8)]], device const ushort* b3 [[buffer(9)]],
                      device const float* query [[buffer(10)]], device const ushort* gamma [[buffer(11)]],
                      device float* res [[buffer(12)]], device ushort* hn [[buffer(13)]],
                      device const ushort* gate_w [[buffer(14)]], device const float* bias [[buffer(15)]],
                      device int* idx [[buffer(16)]], device float* wout [[buffer(17)]],
                      device const ushort* bank0 [[buffer(18)]], device const ushort* bank1 [[buffer(19)]],
                      device const float* mq [[buffer(20)]], device const ushort* down [[buffer(21)]],
                      device const ushort* forget [[buffer(22)]], device const ushort* gain [[buffer(23)]],
                      constant ChainParams& p [[buffer(24)]], uint tid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float va[CH_MAXD];
  threadgroup float vb[CH_MAXD];
  threadgroup float scratch[16 * 32];
  threadgroup float h[64];
  threadgroup float sc[128];
  threadgroup uint sel[8];
  const uint D = p.D;
  for (uint t = 0; t < p.T; t++) {
    // res = conv(o_proj out) (+ x); a boundary layer starts a new block: x is copied out and res has no x.
    for (uint i = tid; i < D; i += CH_NT) {
      float r = ch_conv(ao, conv_w, conv_state, conv_hist, t, i, D, p.has_hist != 0);
      if (p.boundary) blk[t * D + i] = x[t * D + i];
      else r = r + bf2f(x[t * D + i]);
      res[t * D + i] = r;
      va[i] = r;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (p.nq == 101) continue;
    ch_mix(b0, b1, b2, b3, p.nb, va, query, vb, t, D, tid, sg, lane, scratch);  // vb = mlp input mix
    if (p.nq == 102) continue;
    if (p.has_msg) {
      float coef[2];
      ch_msg_coef(bank0, bank1, mq, 3, 1, coef, t, D, tid, sg, lane, scratch);
      float ssx = 0;
      for (uint i = tid; i < D; i += CH_NT) ssx = fma(vb[i], vb[i], ssx);
      ch_sum(&ssx, 1, scratch, sg, lane);
      ch_ctrl(vb, ssx, bank0, bank1, coef[0], coef[1], down, forget, gain, va, vb, h, t, D, p.R, tid, sg, lane, scratch);
    }
    ch_norm(vb, gamma, hn, p.eps, true, t, D, tid, sg, lane, scratch);  // vb = float(hn)
    if (p.nq == 103) continue;
    // Router: simdgroup sg scores experts sg * E / 32 ..; lanes split the D inputs.
    uint eper = p.E / 32, span = D / 32;
    for (uint ee = 0; ee < eper; ee++) {
      uint e = sg * eper + ee;
      float a = 0;
      for (uint j = lane * span; j < (lane + 1) * span; j += 8) {
        float w8[8];
        unpack_bf8(*(device const uint4*)(gate_w + size_t(e) * D + j), w8);
        for (uint k = 0; k < 8; k++) a = fma(vb[j + k], w8[k], a);
      }
      a = simd_sum(a);
      if (lane == 0) sc[e] = 1.0f / (1.0f + exp(-a));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < p.E) {
      float me = sc[tid] + bias[tid];
      uint rank = 0;
      for (uint j = 0; j < p.E; j++) {
        float o = sc[j] + bias[j];
        rank += (o > me || (o == me && j < tid)) ? 1u : 0u;
      }
      if (rank < p.topk) sel[rank] = tid;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < p.topk) {
      float sum = 0;
      for (uint k = 0; k < p.topk; k++) sum += sc[sel[k]];
      idx[t * p.topk + tid] = int(sel[tid]);
      wout[t * p.topk + tid] = sc[sel[tid]] / (sum + 1e-20f) * p.route_scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
