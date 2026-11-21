// Prefill GEMM: out[T, N] = x[T, K] @ W[N, K]^T, W bf16, x bf16 or f32, fp32 accumulation, out f32 or bf16.
// Owner: B. Threadgroup = 4 simdgroups computes a 64 (T) x 64 (N) tile; each simdgroup a 32x32 block of 8x8
// fp32 simdgroup matrices. K steps of 32 staged in threadgroup memory as float.

struct GemmParams {
  uint T, N, K;
  uint dt_x, dt_out;
  uint ldo;  // out row stride (elements)
};

constant uint GB_M = 64, GB_N = 64, GB_K = 32;

kernel void gemm_bf16w(device const ushort* W [[buffer(0)]], device const void* x [[buffer(1)]],
                       device void* out [[buffer(2)]], constant GemmParams& p [[buffer(3)]],
                       uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float As[GB_M * (GB_K + 4)];  // [64][36] padded
  threadgroup float Bs[GB_N * (GB_K + 4)];
  constexpr uint LDS = GB_K + 4;
  uint t0 = tg.y * GB_M, n0 = tg.x * GB_N;
  uint sm = (sg / 2) * 32, sn = (sg % 2) * 32;
  simdgroup_float8x8 acc[4][4];
  for (uint i = 0; i < 4; i++)
    for (uint j = 0; j < 4; j++) acc[i][j] = simdgroup_float8x8(0);

  for (uint k0 = 0; k0 < p.K; k0 += GB_K) {
    // load A: 64 x 32 = 2048 values, 16 per thread; B: same
    for (uint i = tid; i < GB_M * GB_K; i += 128) {
      uint r = i / GB_K, c = i % GB_K;
      uint t = t0 + r, k = k0 + c;
      float v = 0;
      if (t < p.T && k < p.K) v = p.dt_x == 0 ? ((device const float*)x)[size_t(t) * p.K + k]
                                             : bf2f(((device const ushort*)x)[size_t(t) * p.K + k]);
      As[r * LDS + c] = v;
      uint n = n0 + r;
      float w = 0;
      if (n < p.N && k < p.K) w = bf2f(W[size_t(n) * p.K + k]);
      Bs[r * LDS + c] = w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < GB_K; kk += 8) {
      simdgroup_float8x8 a[4], b[4];
      for (uint i = 0; i < 4; i++) simdgroup_load(a[i], As + (sm + i * 8) * LDS + kk, LDS);
      for (uint j = 0; j < 4; j++) simdgroup_load(b[j], Bs + (sn + j * 8) * LDS + kk, LDS, ulong2(0, 0), true);
      for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // store via threadgroup memory (reuse As as a [64][64+4] float tile per pass of 2 simdgroups)
  threadgroup float* Cs = As;  // 64*36 floats = 2304; a 32x32 block needs 32*36 = 1152 per simdgroup pair
  for (uint pass = 0; pass < 2; pass++) {
    if (sg / 2 == pass) {
      threadgroup float* cb = Cs + (sg % 2) * 32 * LDS;
      for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++) simdgroup_store(acc[i][j], cb + (i * 8) * LDS + j * 8, LDS);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // simdgroups with sg/2 == pass wrote rows [sm, sm+32) x cols [sn, sn+32) for sn in {0, 32}
    for (uint i = tid; i < 32 * 64; i += 128) {
      uint r = i / 64, c = i % 64;
      uint half_ = c / 32, cc = c % 32;
      float v = Cs[half_ * 32 * LDS + r * LDS + cc];
      uint t = t0 + pass * 32 + r, n = n0 + c;
      if (t < p.T && n < p.N) {
        size_t o = size_t(t) * p.ldo + n;
        if (p.dt_out == 0) ((device float*)out)[o] = v;
        else ((device ushort*)out)[o] = f2bf(v);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// Fast path for bf16 x: tiles kept as bfloat in threadgroup memory, bfloat simdgroup matrices, fp32 accumulators.
// Global -> threadgroup loads are uint4 (8 bf16). Requires K % 8 == 0.
kernel void gemm_bf16(device const ushort* W [[buffer(0)]], device const ushort* x [[buffer(1)]],
                      device void* out [[buffer(2)]], constant GemmParams& p [[buffer(3)]],
                      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint LDS = GB_K + 8;  // bf16 elements per row (pad 16 bytes)
  threadgroup bfloat As[GB_M * LDS];
  threadgroup bfloat Bs[GB_N * LDS];
  uint t0 = tg.y * GB_M, n0 = tg.x * GB_N;
  uint sm = (sg / 2) * 32, sn = (sg % 2) * 32;
  simdgroup_float8x8 acc[4][4];
  for (uint i = 0; i < 4; i++)
    for (uint j = 0; j < 4; j++) acc[i][j] = simdgroup_float8x8(0);
  // each thread loads 2 uint4 of A and 2 of B per K step: row = i / 4, col8 = i % 4
  for (uint k0 = 0; k0 < p.K; k0 += GB_K) {
    for (uint i = tid; i < GB_M * GB_K / 8; i += 128) {
      uint r = i / 4, c8 = (i % 4) * 8;
      uint k = k0 + c8;
      uint t = t0 + r, n = n0 + r;
      uint4 va = uint4(0), vb = uint4(0);
      if (t < p.T && k < p.K) va = *(device const uint4*)(x + size_t(t) * p.K + k);
      if (n < p.N && k < p.K) vb = *(device const uint4*)(W + size_t(n) * p.K + k);
      *(threadgroup uint4*)(As + r * LDS + c8) = va;
      *(threadgroup uint4*)(Bs + r * LDS + c8) = vb;
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
  // store: each simdgroup writes its 32x32 block through a private float staging area
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
