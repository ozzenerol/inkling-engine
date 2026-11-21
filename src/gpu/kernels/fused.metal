// Fused decode/verify ops (bit-identical to the primitive sequences they replace). Owner: B.

// attn_prep_local: one simdgroup per unit; units 0..H-1 = q heads (l2_clip with q_scale), H..H+KVH-1 = k heads
// (short conv + l2_clip), next KVH = v heads (short conv + rms over 128, no gamma), last H = r heads (l2_clip
// with rel_scale, d = 16). Each unit loops over the T rows (T <= 16), so conv states update without races.
struct PrepParams {
  uint T, H, KVH;
  float qk_thr, rel_thr, eps;
  uint has_hist;
};

inline float l2_factor(float ss, float thr) { return min(thr / (precise::sqrt(ss) + 1e-6f), 1.0f); }

kernel void attn_prep_local(device ushort* q [[buffer(0)]], device const ushort* q_scale [[buffer(1)]],
                            device ushort* r [[buffer(2)]], device const ushort* rel_scale [[buffer(3)]],
                            device const ushort* kraw [[buffer(4)]], device const ushort* vraw [[buffer(5)]],
                            device const ushort* kw [[buffer(6)]], device const ushort* vw [[buffer(7)]],
                            device float* kstate [[buffer(8)]], device float* vstate [[buffer(9)]],
                            device float* khist [[buffer(10)]], device float* vhist [[buffer(11)]],
                            device ushort* k3 [[buffer(12)]], device ushort* v3 [[buffer(13)]],
                            constant PrepParams& p [[buffer(14)]], uint unit [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
  const uint H = p.H, KVH = p.KVH, C = KVH * 128;
  if (unit < H || unit >= H + 2 * KVH) {
    // l2 clip rows (same lane layout / order as l2_clip_rows)
    bool isq = unit < H;
    uint h = isq ? unit : unit - H - 2 * KVH;
    uint d = isq ? 128 : 16;
    device ushort* x = isq ? q : r;
    device const ushort* sc = isq ? q_scale : rel_scale;
    float thr = isq ? p.qk_thr : p.rel_thr;
    for (uint t = 0; t < p.T; t++) {
      uint row = t * H + h;
      float v[4];
      float ss = 0;
      for (uint j = 0; j < 4; j++) {
        uint i = lane + 32 * j;
        float y = 0;
        if (i < d) {
          y = bf2f(x[row * d + i]);
          y = y * bf2f(sc[h * d + i]);
        }
        v[j] = y;
        ss += y * y;
      }
      ss = simd_sum(ss);
      float f = l2_factor(ss, thr);
      for (uint j = 0; j < 4; j++) {
        uint i = lane + 32 * j;
        if (i < d) x[row * d + i] = f2bf(round_bf(v[j] * f));
      }
    }
    return;
  }
  bool isk = unit < H + KVH;
  uint hk = isk ? unit - H : unit - H - KVH;
  device const ushort* raw = isk ? kraw : vraw;
  device const ushort* w = isk ? kw : vw;
  device float* state = isk ? kstate : vstate;
  device float* hist = isk ? khist : vhist;
  device ushort* outp = isk ? k3 : v3;
  const float div = 2.2360679774997896f;
  float s0[4], s1[4], s2[4], w0[4], w1[4], w2[4], w3[4];
  for (uint j = 0; j < 4; j++) {
    uint c = hk * 128 + lane + 32 * j;
    s0[j] = state[0 * C + c];
    s1[j] = state[1 * C + c];
    s2[j] = state[2 * C + c];
    w0[j] = bf2f(w[c * 4 + 0]);
    w1[j] = bf2f(w[c * 4 + 1]);
    w2[j] = bf2f(w[c * 4 + 2]);
    w3[j] = bf2f(w[c * 4 + 3]);
  }
  for (uint t = 0; t < p.T; t++) {
    float cv[4];
    for (uint j = 0; j < 4; j++) {
      uint c = hk * 128 + lane + 32 * j;
      float xv = bf2f(raw[t * C + c]);
      float v = 0;
      v = v + s0[j] * w0[j];
      v = v + s1[j] * w1[j];
      v = v + s2[j] * w2[j];
      v = v + xv * w3[j];
      cv[j] = round_bf((v + xv) / div);  // short_conv output (bf16)
      s0[j] = s1[j];
      s1[j] = s2[j];
      s2[j] = xv;
      if (p.has_hist) {
        hist[(t * 3 + 0) * C + c] = s0[j];
        hist[(t * 3 + 1) * C + c] = s1[j];
        hist[(t * 3 + 2) * C + c] = s2[j];
      }
    }
    if (isk) {  // l2_clip, no scale
      float ss = 0;
      for (uint j = 0; j < 4; j++) ss += cv[j] * cv[j];
      ss = simd_sum(ss);
      float f = l2_factor(ss, p.qk_thr);
      for (uint j = 0; j < 4; j++) outp[(t * KVH + hk) * 128 + lane + 32 * j] = f2bf(round_bf(cv[j] * f));
    } else {  // rms_norm over 128 without gamma, as rms_norm_rows: per-32 simd sums added in order
      float tot = 0;
      for (uint j = 0; j < 4; j++) tot += simd_sum(cv[j] * cv[j]);
      float inv = metal::precise::rsqrt(tot / 128.0f + p.eps);
      for (uint j = 0; j < 4; j++) outp[(t * KVH + hk) * 128 + lane + 32 * j] = f2bf(round_bf(cv[j] * inv));
    }
  }
  for (uint j = 0; j < 4; j++) {
    uint c = hk * 128 + lane + 32 * j;
    state[0 * C + c] = s0[j];
    state[1 * C + c] = s1[j];
    state[2 * C + c] = s2[j];
  }
}

// conv_add: tmp = bf16(short_conv(x)); out = float(tmp) + float(other) in out's dtype. One thread per channel.
struct ConvAddParams {
  uint T, C, has_hist, dt_other, dt_out;
};
kernel void conv_add4(device const ushort* x [[buffer(0)]], device const ushort* w [[buffer(1)]],
                      device float* state [[buffer(2)]], device float* hist [[buffer(3)]],
                      device const void* other [[buffer(4)]], device void* out [[buffer(5)]],
                      constant ConvAddParams& p [[buffer(6)]], uint c [[thread_position_in_grid]]) {
  if (c >= p.C) return;
  float s0 = state[0 * p.C + c], s1 = state[1 * p.C + c], s2 = state[2 * p.C + c];
  float w0 = bf2f(w[c * 4 + 0]), w1 = bf2f(w[c * 4 + 1]), w2 = bf2f(w[c * 4 + 2]), w3 = bf2f(w[c * 4 + 3]);
  const float div = 2.2360679774997896f;
  for (uint t = 0; t < p.T; t++) {
    float xv = bf2f(x[t * p.C + c]);
    float v = 0;
    v = v + s0 * w0;
    v = v + s1 * w1;
    v = v + s2 * w2;
    v = v + xv * w3;
    float tmp = round_bf((v + xv) / div);
    stv(out, p.dt_out, t * p.C + c, tmp + ldv(other, p.dt_other, t * p.C + c));
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
