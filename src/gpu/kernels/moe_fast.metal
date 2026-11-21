// Fast MoE for decode / verify (few tokens): two dispatches per MoE block. Owner: B.
//   moe_gateup_*: a[route, n] = bf16(act(bf16(g), bf16(u))), g/u rows of the routed expert, fp32 accumulation
//   moe_down_*:   out[t, n] = bf16(sum_k w[t,k] * bf16(down[e_k] a[t,k]))  (k in order 0..7)
// Weights are read as uint4 (16 bytes). Lane mapping is fixed: in gate/up a simdgroup does 4 rows, 8 lanes per
// row; in down a simdgroup does 1 row for 8 routes, 4 lanes per route. Partial sums reduce over those lanes.
//
// Dequant: e2m1 / e4m3 codes go through half bit patterns (exact), which carry a power-of-two factor
// (2^-14 for e2m1, 2^-8 for e4m3) that is folded into the group scale. Products and sums are then the
// reference values times that power of two, so after the scale multiply the result is identical.

inline float fp4h(uint nib) {  // e2m1 value * 2^-14
  return float(as_type<half>(ushort(((nib & 8u) << 12) | ((nib & 7u) << 9))));
}
inline float fp8h(uint b) {  // e4m3 value * 2^-8 (0x7f/0xff = NaN in e4m3fn; half gives inf/nan too)
  return float(as_type<half>(ushort(((b & 0x80u) << 8) | ((b & 0x7fu) << 7))));
}
inline float e4m3_scale(uint b) { return e4m3_to_f(b); }

// dot of 32 fp4 codes in a uint4 with x[0..31] (float, threadgroup); two groups of 16, each with its own scale.
inline float dot_fp4_32(uint4 w, threadgroup const float* x, float s0, float s1) {
  float a = 0, b = 0;
  uint ws[4] = {w.x, w.y, w.z, w.w};
  for (uint j = 0; j < 2; j++)
    for (uint i = 0; i < 8; i++) a = fma(x[j * 8 + i], fp4h((ws[j] >> (4 * i)) & 15u), a);
  for (uint j = 2; j < 4; j++)
    for (uint i = 0; i < 8; i++) b = fma(x[j * 8 + i], fp4h((ws[j] >> (4 * i)) & 15u), b);
  return a * s0 + b * s1;
}
// dot of 16 e4m3 codes with x[0..15]
inline float dot_fp8_16(uint4 w, threadgroup const float* x) {
  float a = 0;
  uint ws[4] = {w.x, w.y, w.z, w.w};
  for (uint j = 0; j < 4; j++)
    for (uint i = 0; i < 4; i++) a = fma(x[j * 4 + i], fp8h((ws[j] >> (8 * i)) & 255u), a);
  return a;
}
inline float dot_fp4_32d(uint4 w, device const ushort* x, float s0, float s1) {
  float a = 0, b = 0;
  uint ws[4] = {w.x, w.y, w.z, w.w};
  for (uint j = 0; j < 2; j++)
    for (uint i = 0; i < 8; i++) a = fma(bf2f(x[j * 8 + i]), fp4h((ws[j] >> (4 * i)) & 15u), a);
  for (uint j = 2; j < 4; j++)
    for (uint i = 0; i < 8; i++) b = fma(bf2f(x[j * 8 + i]), fp4h((ws[j] >> (4 * i)) & 15u), b);
  return a * s0 + b * s1;
}
inline float dot_fp8_16d(uint4 w, device const ushort* x) {
  float a = 0;
  uint ws[4] = {w.x, w.y, w.z, w.w};
  for (uint j = 0; j < 4; j++)
    for (uint i = 0; i < 4; i++) a = fma(bf2f(x[j * 4 + i]), fp8h((ws[j] >> (8 * i)) & 255u), a);
  return a;
}

inline float softcap_act(float g, float u) {
  return precise::tanh(g / 4.0f) * 4.0f * (1.0f / (1.0f + exp(-g))) * precise::tanh(u / 25.0f) * 25.0f;
}

struct MoeFastParams {
  uint N, K, topk;
};

// Byte -> (lo nibble, hi nibble) as half2, both * 2^-14 (exact), for the LUT variant.
inline void build_fp4_lut(threadgroup half2* lut, uint tid, uint nthreads) {
  for (uint b = tid; b < 256; b += nthreads)
    lut[b] = half2(as_type<half>(ushort(((b & 8u) << 12) | ((b & 7u) << 9))),
                   as_type<half>(ushort((((b >> 4) & 8u) << 12) | (((b >> 4) & 7u) << 9))));
}
inline float dot_fp4_32_lut(uint4 w, threadgroup const float* x, float s0, float s1, threadgroup const half2* lut) {
  float a = 0, b = 0;
  uint ws[4] = {w.x, w.y, w.z, w.w};
  for (uint j = 0; j < 2; j++)
    for (uint i = 0; i < 4; i++) {
      float2 v = float2(lut[(ws[j] >> (8 * i)) & 255u]);
      a = fma(x[j * 8 + 2 * i], v.x, a);
      a = fma(x[j * 8 + 2 * i + 1], v.y, a);
    }
  for (uint j = 2; j < 4; j++)
    for (uint i = 0; i < 4; i++) {
      float2 v = float2(lut[(ws[j] >> (8 * i)) & 255u]);
      b = fma(x[j * 8 + 2 * i], v.x, b);
      b = fma(x[j * 8 + 2 * i + 1], v.y, b);
    }
  return a * s0 + b * s1;
}

constant uint GU_SIMDS = 4;  // simdgroups per threadgroup in gate/up (4 rows each)

// MODE 0 = NVFP4, 1 = MXFP8. Grid: (ceil(N / (4*GU_SIMDS)), routes). x bf16 [T, K]; a bf16 [routes, N].
template <uint MODE, bool LUT = false>
kernel void moe_gateup(device const uint4* gw [[buffer(0)]], device const uchar* gs [[buffer(1)]],
                       device const float* gg [[buffer(2)]], device const uint4* uw [[buffer(3)]],
                       device const uchar* us [[buffer(4)]], device const float* ug [[buffer(5)]],
                       device const ushort* x [[buffer(6)]], device const int* idx [[buffer(7)]],
                       device ushort* a [[buffer(8)]], constant MoeFastParams& p [[buffer(9)]],
                       uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  // x staged with one pad float per 32 (chunk c at c*33): lanes reading different chunks hit different banks
  threadgroup float xs[33 * 128];
  // grid: x = route (fastest), y = row block: routes that share an expert read the same rows close together
  uint route = tg.x, t = route / p.topk;
  for (uint i = tid; i < p.K; i += 32 * GU_SIMDS) xs[(i / 32) * 33 + i % 32] = bf2f(x[t * p.K + i]);
  threadgroup half2 lut[256];  // LUT variant only (the declaration was missing: the library did not compile)
  if (LUT) build_fp4_lut(lut, tid, 32 * GU_SIMDS);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint e = uint(idx[route]);
  uint n = (tg.y * GU_SIMDS + sg) * 4 + lane / 8;
  uint sub = lane % 8;
  bool valid = n < p.N;
  n = min(n, p.N - 1);
  size_t row = size_t(e) * p.N + n;
  float g = 0, u = 0;
  if (MODE == 0) {
    uint chunks = p.K / 32;  // uint4 per row
    device const uint4* gr = gw + row * chunks;
    device const uint4* ur = uw + row * chunks;
    device const ushort* gsr = (device const ushort*)(gs + row * (p.K / 16));
    device const ushort* usr = (device const ushort*)(us + row * (p.K / 16));
    for (uint c = sub; c < chunks; c += 8) {
      uint gsc = gsr[c], usc = usr[c];
      threadgroup const float* xc = xs + c * 33;
      if (LUT) {
        g += dot_fp4_32_lut(gr[c], xc, e4m3_scale(gsc & 255u), e4m3_scale(gsc >> 8), lut);
        u += dot_fp4_32_lut(ur[c], xc, e4m3_scale(usc & 255u), e4m3_scale(usc >> 8), lut);
      } else {
        g += dot_fp4_32(gr[c], xc, e4m3_scale(gsc & 255u), e4m3_scale(gsc >> 8));
        u += dot_fp4_32(ur[c], xc, e4m3_scale(usc & 255u), e4m3_scale(usc >> 8));
      }
    }
  } else {
    uint chunks = p.K / 16;
    device const uint4* gr = gw + row * chunks;
    device const uint4* ur = uw + row * chunks;
    device const uchar* gsr = gs + row * (p.K / 32);
    device const uchar* usr = us + row * (p.K / 32);
    // lane takes chunk pairs (one 32-value group = 2 uint4) so the e8m0 scale applies to one partial sum
    for (uint c2 = sub; c2 < chunks / 2; c2 += 8) {
      threadgroup const float* xc = xs + c2 * 33;
      float gp = dot_fp8_16(gr[2 * c2], xc) + dot_fp8_16(gr[2 * c2 + 1], xc + 16);
      float up = dot_fp8_16(ur[2 * c2], xc) + dot_fp8_16(ur[2 * c2 + 1], xc + 16);
      g = fma(gp, e8m0_to_f(gsr[c2]), g);
      u = fma(up, e8m0_to_f(usr[c2]), u);
    }
  }
  // reduce over the 8 lanes of this row
  for (uint off = 4; off > 0; off >>= 1) {
    g += simd_shuffle_down(g, off);
    u += simd_shuffle_down(u, off);
  }
  if (sub == 0 && valid) {
    float gf, uf;
    if (MODE == 0) {
      gf = g * (16384.0f * gg[e]);
      uf = u * (16384.0f * ug[e]);
    } else {
      gf = g * 256.0f;
      uf = u * 256.0f;
    }
    a[size_t(route) * p.N + n] = f2bf(softcap_act(round_bf(gf), round_bf(uf)));
  }
}

constant uint DN_SIMDS = 4;  // simdgroups per threadgroup in down (1 row each)

// Grid: (ceil(N / DN_SIMDS), T). a bf16 [T*topk, K]; w f32 [T, topk]; out bf16 [T, N]. topk must be 8.
template <uint MODE>
kernel void moe_down(device const uint4* dw [[buffer(0)]], device const uchar* ds [[buffer(1)]],
                     device const float* dg [[buffer(2)]], device const ushort* a [[buffer(3)]],
                     device const int* idx [[buffer(4)]], device const float* wts [[buffer(5)]],
                     device ushort* out [[buffer(6)]], constant MoeFastParams& p [[buffer(7)]],
                     uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]]) {
  uint t = tg.x;  // grid: x = token (fastest), y = row group
  uint n = tg.y * DN_SIMDS + sg;
  if (n >= p.N) return;
  uint k = lane / 4, sub = lane % 4;
  uint route = t * p.topk + k;
  uint e = uint(idx[route]);
  size_t row = size_t(e) * p.N + n;
  device const ushort* ar = a + size_t(route) * p.K;
  float d = 0;
  if (MODE == 0) {
    uint chunks = p.K / 32;
    device const uint4* wr = dw + row * chunks;
    device const ushort* sr = (device const ushort*)(ds + row * (p.K / 16));
    for (uint c = sub; c < chunks; c += 4) {
      uint sc = sr[c];
      d += dot_fp4_32d(wr[c], ar + c * 32, e4m3_scale(sc & 255u), e4m3_scale(sc >> 8));
    }
  } else {
    uint chunks = p.K / 16;
    device const uint4* wr = dw + row * chunks;
    device const uchar* sr = ds + row * (p.K / 32);
    for (uint c2 = sub; c2 < chunks / 2; c2 += 4) {
      float dp = dot_fp8_16d(wr[2 * c2], ar + c2 * 32) + dot_fp8_16d(wr[2 * c2 + 1], ar + c2 * 32 + 16);
      d = fma(dp, e8m0_to_f(sr[c2]), d);
    }
  }
  d += simd_shuffle_down(d, 2);
  d += simd_shuffle_down(d, 1);
  // lanes 0,4,..,28 hold route sums; round to bf16, weight, sum in route order
  float dv = MODE == 0 ? d * (16384.0f * dg[e]) : d * 256.0f;
  float contrib = round_bf(dv) * wts[route];
  float s = 0;
  for (uint kk = 0; kk < 8; kk++) s += simd_shuffle(contrib, kk * 4);
  if (lane == 0) out[size_t(t) * p.N + n] = f2bf(s);
}

template [[host_name("moe_gateup_nvfp4")]] kernel void moe_gateup<0>(
    device const uint4*, device const uchar*, device const float*, device const uint4*, device const uchar*,
    device const float*, device const ushort*, device const int*, device ushort*, constant MoeFastParams&, uint2,
    uint, uint, uint);
template [[host_name("moe_gateup_mxfp8")]] kernel void moe_gateup<1>(
    device const uint4*, device const uchar*, device const float*, device const uint4*, device const uchar*,
    device const float*, device const ushort*, device const int*, device ushort*, constant MoeFastParams&, uint2,
    uint, uint, uint);
template [[host_name("moe_down_nvfp4")]] kernel void moe_down<0>(device const uint4*, device const uchar*,
                                                                  device const float*, device const ushort*,
                                                                  device const int*, device const float*,
                                                                  device ushort*, constant MoeFastParams&, uint2,
                                                                  uint, uint);
template [[host_name("moe_down_mxfp8")]] kernel void moe_down<1>(device const uint4*, device const uchar*,
                                                                  device const float*, device const ushort*,
                                                                  device const int*, device const float*,
                                                                  device ushort*, constant MoeFastParams&, uint2,
                                                                  uint, uint);
