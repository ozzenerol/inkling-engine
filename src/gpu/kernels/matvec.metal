// bf16 weight matrix x vector(s): y[t, n] = sum_k W[n, k] * x[t, k], fp32 accumulation. Owner: B.
// W row-major [N, K] bf16, K % 256 == 0. One simdgroup computes MV_ROWS rows; each lane reads 8 contiguous
// bf16 (16 bytes) per 256-column step. T (rows of x) up to MV_MAXT share one pass over W (spec verify).


struct MatvecShape {
  uint N, K, T;
  uint x_stride, y_stride;  // elements between rows of x / y
};

template <typename XT, typename YT, uint TT, uint MV_ROWS, uint MV_SIMDS>
kernel void matvec_bf16(device const uint4* W [[buffer(0)]],
                        device const XT* x [[buffer(1)]],
                        device YT* y [[buffer(2)]],
                        constant MatvecShape& s [[buffer(3)]],
                        uint tg [[threadgroup_position_in_grid]],
                        uint simd [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
  uint row0 = (tg * MV_SIMDS + simd) * MV_ROWS;
  if (row0 >= s.N) return;
  uint kv = s.K / 8;  // uint4 per row
  float acc[TT][MV_ROWS];
  for (uint t = 0; t < TT; t++)
    for (uint r = 0; r < MV_ROWS; r++) acc[t][r] = 0;
  for (uint k8 = lane; k8 < kv; k8 += 32) {
    float xv[TT][8];
    for (uint t = 0; t < TT; t++) {
      device const XT* xp = x + t * s.x_stride + k8 * 8;
      for (uint i = 0; i < 8; i++) xv[t][i] = float(xp[i]);
    }
#pragma clang loop unroll(full)
    for (uint r = 0; r < MV_ROWS; r++) {
      uint row = min(row0 + r, s.N - 1);
      float w[8];
      unpack_bf8(W[row * kv + k8], w);
      for (uint t = 0; t < TT; t++) {
        float a = acc[t][r];
        for (uint i = 0; i < 8; i++) a = fma(w[i], xv[t][i], a);
        acc[t][r] = a;
      }
    }
  }
  for (uint t = 0; t < TT; t++)
    for (uint r = 0; r < MV_ROWS; r++) {
      float v = simd_sum(acc[t][r]);
      if (lane == 0 && row0 + r < s.N && t < s.T) y[t * s.y_stride + row0 + r] = YT(v);
    }
}

// bf16 output stored as ushort with RNE rounding (bfloat type conversion rounding is also RNE on Apple GPUs,
// but keep it explicit).
struct bf16_out {
  ushort v;
  bf16_out(float f) thread : v(f2bf(f)) {}
};
struct bf16_in {
  ushort v;
  operator float() const device { return bf2f(v); }
};

#define MV_INST(XN, XT, YN, YT, TT, R, S)                                                                 \
  template [[host_name("matvec_bf16_" #XN "_" #YN "_t" #TT "_r" #R "_s" #S)]] kernel void                    \
  matvec_bf16<XT, YT, TT, R, S>(device const uint4*, device const XT*, device YT*, constant MatvecShape&, uint, \
                                uint, uint);
#define MV_SWEEP(TT) \
  MV_INST(bf16, bf16_in, bf16, bf16_out, TT, 1, 2) MV_INST(bf16, bf16_in, bf16, bf16_out, TT, 2, 2) \
  MV_INST(bf16, bf16_in, bf16, bf16_out, TT, 4, 2) MV_INST(bf16, bf16_in, bf16, bf16_out, TT, 8, 2) \
  MV_INST(bf16, bf16_in, bf16, bf16_out, TT, 2, 4) MV_INST(bf16, bf16_in, bf16, bf16_out, TT, 4, 4) \
  MV_INST(bf16, bf16_in, bf16, bf16_out, TT, 1, 8) MV_INST(bf16, bf16_in, bf16, bf16_out, TT, 2, 8)
MV_SWEEP(1)
MV_SWEEP(4)
MV_SWEEP(8)
MV_INST(bf16, bf16_in, f32, float, 1, 4, 2)
MV_INST(f32, float, f32, float, 1, 4, 2)
MV_INST(f32, float, bf16, bf16_out, 1, 4, 2)
