// MLX affine 8-bit dense weights (Inkling-2-Mini-MLX-8bit): w = scale * q + bias, group 64, q uint8 packed 4 per
// u32 (little endian), scales/biases bf16 [N, K/64]. Owner: B.
// Decode/verify matvec: each lane takes 16 codes (one uint4) per step; per 16-code chunk it accumulates
// s * sum(x*q) + b * sum(x) in fp32 (the group's s, b), as MLX's qmv does per group.

struct Q8Params {
  uint N, K, T;
  uint x_stride, y_stride;
};

template <typename XT, typename YT, uint TT>
kernel void matvec_q8(device const uint4* W [[buffer(0)]], device const ushort* S [[buffer(1)]],
                      device const ushort* B [[buffer(2)]], device const XT* x [[buffer(3)]],
                      device YT* y [[buffer(4)]], constant Q8Params& s [[buffer(5)]],
                      uint tg [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint R = 4;
  uint row0 = (tg * 2 + simd) * R;
  if (row0 >= s.N) return;
  uint kc = s.K / 16;   // uint4 chunks per row
  uint ng = s.K / 64;   // groups per row
  float acc[TT][R];
  for (uint t = 0; t < TT; t++)
    for (uint r = 0; r < R; r++) acc[t][r] = 0;
  for (uint c = lane; c < kc; c += 32) {
    float xv[TT][16];
    float xs[TT];
    for (uint t = 0; t < TT; t++) {
      xs[t] = 0;
      device const XT* xp = x + t * s.x_stride + c * 16;
      for (uint i = 0; i < 16; i++) {
        xv[t][i] = float(xp[i]);
        xs[t] += xv[t][i];
      }
    }
    uint g = c / 4;
#pragma clang loop unroll(full)
    for (uint r = 0; r < R; r++) {
      uint row = min(row0 + r, s.N - 1);
      uint4 q = W[row * kc + c];
      float sc = bf2f(S[row * ng + g]), bi = bf2f(B[row * ng + g]);
      uint qs[4] = {q.x, q.y, q.z, q.w};
      for (uint t = 0; t < TT; t++) {
        float d = 0;
        for (uint j = 0; j < 4; j++)
          for (uint i = 0; i < 4; i++) d = fma(xv[t][j * 4 + i], float((qs[j] >> (8 * i)) & 255u), d);
        acc[t][r] = fma(sc, d, fma(bi, xs[t], acc[t][r]));
      }
    }
  }
  for (uint t = 0; t < TT; t++)
    for (uint r = 0; r < R; r++) {
      float v = simd_sum(acc[t][r]);
      if (lane == 0 && row0 + r < s.N && t < s.T) y[t * s.y_stride + row0 + r] = YT(v);
    }
}

#define Q8_INST(XN, XT, YN, YT, TT)                                                                         \
  template [[host_name("matvec_q8_" #XN "_" #YN "_t" #TT)]] kernel void matvec_q8<XT, YT, TT>(              \
      device const uint4*, device const ushort*, device const ushort*, device const XT*, device YT*,       \
      constant Q8Params&, uint, uint, uint);
Q8_INST(bf16, bf16_in, bf16, bf16_out, 1)
Q8_INST(bf16, bf16_in, f32, float, 1)
Q8_INST(f32, float, f32, float, 1)
Q8_INST(f32, float, bf16, bf16_out, 1)
Q8_INST(bf16, bf16_in, bf16, bf16_out, 4)
Q8_INST(bf16, bf16_in, f32, float, 4)
Q8_INST(f32, float, f32, float, 4)
Q8_INST(f32, float, bf16, bf16_out, 4)

// Prefill GEMM with q8 weights: as gemm_bf16, but the W tile is dequantized to bf16 (bf16(s*q + b), as MLX's qmm
// produces in the activation dtype) while staging. x bf16, K % 64 == 0.
kernel void gemm_q8(device const uchar* W [[buffer(0)]], device const ushort* S [[buffer(1)]],
                    device const ushort* Bz [[buffer(2)]], device const ushort* x [[buffer(3)]],
                    device void* out [[buffer(4)]], constant GemmParams& p [[buffer(5)]],
                    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint LDS = GB_K + 8;
  threadgroup bfloat As[GB_M * LDS];
  threadgroup bfloat Bs[GB_N * LDS];
  uint t0 = tg.y * GB_M, n0 = tg.x * GB_N;
  uint sm = (sg / 2) * 32, sn = (sg % 2) * 32;
  uint ng = p.K / 64;
  simdgroup_float8x8 acc[4][4];
  for (uint i = 0; i < 4; i++)
    for (uint j = 0; j < 4; j++) acc[i][j] = simdgroup_float8x8(0);
  for (uint k0 = 0; k0 < p.K; k0 += GB_K) {
    for (uint i = tid; i < GB_M * GB_K / 8; i += 128) {
      uint r = i / 4, c8 = (i % 4) * 8;
      uint k = k0 + c8;
      uint t = t0 + r, n = n0 + r;
      uint4 va = uint4(0);
      if (t < p.T) va = *(device const uint4*)(x + size_t(t) * p.K + k);
      *(threadgroup uint4*)(As + r * LDS + c8) = va;
      threadgroup bfloat* dst = Bs + r * LDS + c8;
      if (n < p.N) {
        uint2 q = *(device const uint2*)(W + size_t(n) * p.K + k);
        float sc = bf2f(S[n * ng + k / 64]), bi = bf2f(Bz[n * ng + k / 64]);
        uint qs[2] = {q.x, q.y};
        for (uint j = 0; j < 2; j++)
          for (uint b = 0; b < 4; b++) dst[j * 4 + b] = bfloat(fma(sc, float((qs[j] >> (8 * b)) & 255u), bi));
      } else {
        for (uint j = 0; j < 8; j++) dst[j] = bfloat(0.0f);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < GB_K; kk += 8) {
      simdgroup_bfloat8x8 a[4], b[4];
      for (uint i = 0; i < 4; i++) simdgroup_load(a[i], As + (sm + i * 8) * LDS + kk, LDS);
      for (uint j = 0; j < 4; j++) simdgroup_load(b[j], Bs + (sn + j * 8) * LDS + kk, LDS, ulong2(0, 0), true);
      for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  threadgroup float Cs[4][8 * 8];
  for (uint i = 0; i < 4; i++)
    for (uint j = 0; j < 4; j++) {
      simdgroup_store(acc[i][j], Cs[sg], 8);
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (uint e = lane; e < 64; e += 32) {
        uint r = e / 8, c = e % 8;
        uint t = t0 + sm + i * 8 + r, n = n0 + sn + j * 8 + c;
        if (t < p.T && n < p.N) {
          size_t o = size_t(t) * p.ldo + n;
          float v = Cs[sg][e];
          if (p.dt_out == 0) ((device float*)out)[o] = v;
          else ((device ushort*)out)[o] = f2bf(v);
        }
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}
