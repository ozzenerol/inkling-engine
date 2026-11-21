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
MV_INST(bf16, bf16_in, bf16, bf16_out, 1, 4, 2)
MV_INST(bf16, bf16_in, f32, float, 1, 4, 2)
MV_INST(f32, float, f32, float, 1, 4, 2)
MV_INST(f32, float, bf16, bf16_out, 1, 4, 2)
MV_INST(bf16, bf16_in, bf16, bf16_out, 4, 4, 2)
MV_INST(bf16, bf16_in, f32, float, 4, 4, 2)
MV_INST(f32, float, f32, float, 4, 4, 2)
MV_INST(f32, float, bf16, bf16_out, 4, 4, 2)
MV_INST(bf16, bf16_in, bf16, bf16_out, 8, 4, 2)
MV_INST(bf16, bf16_in, f32, float, 1, 1, 4)
MV_INST(bf16, bf16_in, f32, float, 4, 1, 4)

// Several matrices on one input in one dispatch (decode/verify, T <= 4): up to 8 (W_i [N_i, K], y_i [T, N_i]).
// Threadgroup g covers rows [g*8, g*8+8) of the concatenated row space; same per-row math as matvec_bf16.
struct MatvecMultiParams {
  uint K, T, count;
  uint row_start[9];  // prefix sums of N_i (row_start[count] = total)
  uint dt_out[8];     // 0 f32, 1 bf16
};
template <typename XT>
kernel void matvec_multi(device const uint4* W0 [[buffer(0)]], device const uint4* W1 [[buffer(1)]],
                         device const uint4* W2 [[buffer(2)]], device const uint4* W3 [[buffer(3)]],
                         device const uint4* W4 [[buffer(4)]], device const uint4* W5 [[buffer(5)]],
                         device const uint4* W6 [[buffer(6)]], device const uint4* W7 [[buffer(7)]],
                         device void* Y0 [[buffer(8)]], device void* Y1 [[buffer(9)]], device void* Y2 [[buffer(10)]],
                         device void* Y3 [[buffer(11)]], device void* Y4 [[buffer(12)]],
                         device void* Y5 [[buffer(13)]], device void* Y6 [[buffer(14)]],
                         device void* Y7 [[buffer(15)]], device const XT* x [[buffer(16)]],
                         constant MatvecMultiParams& s [[buffer(17)]], uint tg [[threadgroup_position_in_grid]],
                         uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint R = 4;
  uint grow0 = (tg * 2 + simd) * R;  // global row of this simdgroup
  if (grow0 >= s.row_start[s.count]) return;
  uint mi = 0;
  while (mi + 1 < s.count && grow0 >= s.row_start[mi + 1]) mi++;
  device const uint4* Ws[8] = {W0, W1, W2, W3, W4, W5, W6, W7};
  device void* Ys[8] = {Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7};
  device const uint4* W = Ws[mi];
  uint N = s.row_start[mi + 1] - s.row_start[mi];
  uint row0 = grow0 - s.row_start[mi];
  // a simdgroup's 4 rows may straddle two matrices only if N_i % 4 != 0: handled by computing rows one matrix at
  // a time (rows past this matrix's end are done by the simdgroup that starts there: row_start are multiples of 4)
  uint kv = s.K / 8;
  float acc[4][R];
  for (uint t = 0; t < 4; t++)
    for (uint r = 0; r < R; r++) acc[t][r] = 0;
  for (uint k8 = lane; k8 < kv; k8 += 32) {
    float xv[4][8];
    for (uint t = 0; t < s.T; t++) {
      device const XT* xp = x + t * s.K + k8 * 8;
      for (uint i = 0; i < 8; i++) xv[t][i] = float(xp[i]);
    }
    for (uint r = 0; r < R; r++) {
      uint row = min(row0 + r, N - 1);
      float w[8];
      unpack_bf8(W[row * kv + k8], w);
      for (uint t = 0; t < s.T; t++) {
        float a = acc[t][r];
        for (uint i = 0; i < 8; i++) a = fma(w[i], xv[t][i], a);
        acc[t][r] = a;
      }
    }
  }
  for (uint t = 0; t < s.T; t++)
    for (uint r = 0; r < R; r++) {
      float v = simd_sum(acc[t][r]);
      uint row = row0 + r;
      if (lane == 0 && row < N) {
        if (s.dt_out[mi] == 0) ((device float*)Ys[mi])[t * N + row] = v;
        else ((device ushort*)Ys[mi])[t * N + row] = f2bf(v);
      }
    }
}
template [[host_name("matvec_multi_bf16")]] kernel void matvec_multi<bf16_in>(
    device const uint4*, device const uint4*, device const uint4*, device const uint4*, device const uint4*,
    device const uint4*, device const uint4*, device const uint4*, device void*, device void*, device void*,
    device void*, device void*, device void*, device void*, device void*, device const bf16_in*,
    constant MatvecMultiParams&, uint, uint, uint);
template [[host_name("matvec_multi_f32")]] kernel void matvec_multi<float>(
    device const uint4*, device const uint4*, device const uint4*, device const uint4*, device const uint4*,
    device const uint4*, device const uint4*, device const uint4*, device void*, device void*, device void*,
    device void*, device void*, device void*, device void*, device void*, device const float*,
    constant MatvecMultiParams&, uint, uint, uint);

// Byte copy as a compute dispatch (keeps the compute encoder open; blits would end it). 16-byte units + tail.
kernel void copy_bytes(device const uchar* src [[buffer(0)]], device uchar* dst [[buffer(1)]],
                       constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  uint b = i * 16;
  if (b + 16 <= n) {
    *(device packed_uint4*)(dst + b) = *(device const packed_uint4*)(src + b);
  } else {
    for (uint j = b; j < n; j++) dst[j] = src[j];
  }
}

// Skinny GEMM for 5..16 rows of x (draft blocks, verify > 4): out^T[N, T] = W[N, K] x^T, simdgroup matrices.
// A simdgroup computes 8 output rows x 16 tokens (2 frags); W and x frags are loaded straight from device memory
// (W rows are reused over consecutive k steps through L1). x rows past T are clamped (results not stored).
struct SkinnyParams {
  uint N, K, T;
  uint dt_out;
};
template <typename XT>
kernel void gemm_skinny(device const bfloat* W [[buffer(0)]], device const XT* x [[buffer(1)]],
                        device void* y [[buffer(2)]], constant SkinnyParams& s [[buffer(3)]],
                        uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
  uint n0 = (tg * 4 + sg) * 8;
  if (n0 >= s.N) return;
  uint nrow = min(n0, s.N - 8);  // last tile overlaps (N >= 8 assumed); stores are guarded
  threadgroup XT xs_dummy[1];
  (void)xs_dummy;
  simdgroup_float8x8 acc[2] = {simdgroup_float8x8(0), simdgroup_float8x8(0)};
  device const bfloat* wp = W + size_t(nrow) * s.K;
  uint t1 = min(8u, s.T), t2 = s.T;
  for (uint k = 0; k < s.K; k += 8) {
    simdgroup_bfloat8x8 wf;
    simdgroup_load(wf, wp + k, s.K);
    for (uint j = 0; j < 2; j++) {
      // x^T frag: element (row = k-offset, col = token): load x[t][k..k+8] transposed
      simdgroup_matrix<XT, 8, 8> xf;
      uint tb = min(j * 8, s.T - 1);
      simdgroup_load(xf, x + size_t(tb) * s.K + k, s.K, ulong2(0, 0), true);
      if (j == 1 && s.T <= 8) break;
      simdgroup_float8x8 xff;
      thread auto& a = xf.thread_elements();
      thread auto& b = xff.thread_elements();
      b[0] = float(a[0]);
      b[1] = float(a[1]);
      simdgroup_float8x8 wff;
      thread auto& wa = wf.thread_elements();
      thread auto& wb = wff.thread_elements();
      wb[0] = float(wa[0]);
      wb[1] = float(wa[1]);
      simdgroup_multiply_accumulate(acc[j], wff, xff, acc[j]);
    }
  }
  (void)t1;
  (void)t2;
  short2 fc = frag_coord(lane);
  uint fm = uint(fc.y), fn = uint(fc.x);
  for (uint j = 0; j < 2; j++) {
    thread auto& e = acc[j].thread_elements();
    for (uint c = 0; c < 2; c++) {
      uint n = nrow + fm, t = j * 8 + fn + c;
      if (n >= n0 && n < s.N && t < s.T) {
        size_t o = size_t(t) * s.N + n;
        if (s.dt_out == 0) ((device float*)y)[o] = e[c];
        else ((device ushort*)y)[o] = f2bf(e[c]);
      }
    }
  }
}
template [[host_name("gemm_skinny_bf16")]] kernel void gemm_skinny<bfloat>(device const bfloat*, device const bfloat*,
                                                                          device void*, constant SkinnyParams&, uint,
                                                                          uint, uint);
template [[host_name("gemm_skinny_f32")]] kernel void gemm_skinny<float>(device const bfloat*, device const float*,
                                                                        device void*, constant SkinnyParams&, uint,
                                                                        uint, uint);

// Skinny GEMM v2 (T <= 16 rows of bf16 x): threadgroup = 2 simdgroups x 8 output rows; K in steps of 64 with W
// [16 x 64] and x [16 x 64] tiles staged by coalesced 16-byte loads; out^T = W x^T via 8x8 bf16 mma, fp32 acc.
kernel void gemm_skinny2(device const ushort* W [[buffer(0)]], device const ushort* x [[buffer(1)]],
                         device void* y [[buffer(2)]], constant SkinnyParams& s [[buffer(3)]],
                         uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint KS = 64, LD = KS + 8;
  threadgroup bfloat Ws[16 * LD];
  threadgroup bfloat Xs[16 * LD];
  uint n0 = tg * 16;
  simdgroup_float8x8 acc[2] = {simdgroup_float8x8(0), simdgroup_float8x8(0)};
  for (uint k0 = 0; k0 < s.K; k0 += KS) {
    // 16 rows x 64 = 128 uint4 for W and for x; 64 threads -> 2 each
    for (uint i = tid; i < 128; i += 64) {
      uint r = i / 8, c8 = (i % 8) * 8;
      uint n = min(n0 + r, s.N - 1), t = min(r, s.T - 1);
      uint k = k0 + c8;
      uint4 wv = uint4(0), xv = uint4(0);
      if (k < s.K) {
        wv = *(device const uint4*)(W + size_t(n) * s.K + k);
        xv = *(device const uint4*)(x + size_t(t) * s.K + k);
      }
      *(threadgroup uint4*)(Ws + r * LD + c8) = wv;
      *(threadgroup uint4*)(Xs + r * LD + c8) = xv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < KS; kk += 8) {
      simdgroup_bfloat8x8 wf, x0, x1;
      simdgroup_load(wf, Ws + (sg * 8) * LD + kk, LD);
      simdgroup_load(x0, Xs + kk, LD, ulong2(0, 0), true);
      simdgroup_multiply_accumulate(acc[0], wf, x0, acc[0]);
      if (s.T > 8) {
        simdgroup_load(x1, Xs + 8 * LD + kk, LD, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(acc[1], wf, x1, acc[1]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  short2 fc = frag_coord(lane);
  uint fm = uint(fc.y), fn = uint(fc.x);
  for (uint j = 0; j < 2; j++) {
    thread auto& e = acc[j].thread_elements();
    for (uint c = 0; c < 2; c++) {
      uint n = n0 + sg * 8 + fm, t = j * 8 + fn + c;
      if (n < s.N && t < s.T) {
        size_t o = size_t(t) * s.N + n;
        if (s.dt_out == 0) ((device float*)y)[o] = e[c];
        else ((device ushort*)y)[o] = f2bf(e[c]);
      }
    }
  }
}
