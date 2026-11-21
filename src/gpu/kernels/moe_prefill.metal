// MoE for prefill (many tokens): routes sorted by expert, grouped GEMM per expert. Owner: B.
//   1. moe_sort: counting sort of routes (t*topk+k) by expert -> perm[sorted] = route, tile map
//   2. moe_gemm<MODE> for gate and up (A rows gathered from x[route / topk]) -> g, u bf16 [sorted, I]
//   3. softcap_swiglu (elementwise, sorted order) -> a
//   4. moe_gemm<MODE> for down (A = a sorted rows, output scattered to d[route]) -> d bf16 [routes, D]
//   5. moe_combine (as in decode)
// Weight tiles are dequantized to bf16 in threadgroup memory. fp4*e4m3 and e4m3*2^k are exact in bf16; the
// NVFP4 per-expert global scale is applied to the fp32 accumulator, as in the reference.

struct SortParams {
  uint routes, E, tile_m;
};
// One threadgroup of 1024 threads. counts/offsets: [E+1]; tiles: (expert, row0) pairs, ntiles at tiles_n[0].
kernel void moe_sort(device const int* idx [[buffer(0)]], device int* perm [[buffer(1)]],
                     device int* offsets [[buffer(2)]], device uint2* tiles [[buffer(3)]],
                     device uint* tiles_n [[buffer(4)]], constant SortParams& p [[buffer(5)]],
                     uint tid [[thread_index_in_threadgroup]]) {
  threadgroup atomic_uint cnt[128];
  threadgroup uint off[129];
  threadgroup atomic_uint cur[128];
  if (tid < p.E) atomic_store_explicit(&cnt[tid], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint r = tid; r < p.routes; r += 1024) atomic_fetch_add_explicit(&cnt[idx[r]], 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    uint s = 0, nt = 0;
    for (uint e = 0; e < p.E; e++) {
      off[e] = s;
      uint c = atomic_load_explicit(&cnt[e], memory_order_relaxed);
      for (uint r0 = 0; r0 < c; r0 += p.tile_m) tiles[nt++] = uint2(e, s + r0);
      s += c;
    }
    off[p.E] = s;
    tiles_n[0] = nt;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid <= p.E) offsets[tid] = int(off[tid]);
  if (tid < p.E) atomic_store_explicit(&cur[tid], off[tid], memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint r = tid; r < p.routes; r += 1024) {
    uint pos = atomic_fetch_add_explicit(&cur[idx[r]], 1u, memory_order_relaxed);
    perm[pos] = int(r);
  }
}

struct MoeGemmParams {
  uint N, K;          // per expert: out rows, in cols
  uint topk;
  uint gather_a;      // 1: A row of sorted i = x[perm[i] / topk]; 0: A row = i
  uint scatter_out;   // 1: out row = perm[i]; 0: out row = i
};

constant uint MG_M = 64, MG_N = 64, MG_K = 32;

// Grid: (N / 64, max_tiles). Threadgroup 128 (4 simdgroups, 32x32 each).
template <uint MODE>
kernel void moe_gemm(device const uint4* w [[buffer(0)]], device const uchar* sc [[buffer(1)]],
                     device const float* gs [[buffer(2)]], device const ushort* a [[buffer(3)]],
                     device const int* perm [[buffer(4)]], device const int* offsets [[buffer(5)]],
                     device const uint2* tiles [[buffer(6)]], device const uint* tiles_n [[buffer(7)]],
                     device ushort* out [[buffer(8)]], constant MoeGemmParams& p [[buffer(9)]],
                     uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  if (tg.y >= tiles_n[0]) return;
  uint2 tile = tiles[tg.y];
  uint e = tile.x, r0 = tile.y;
  uint rend = uint(offsets[e + 1]);
  uint n0 = tg.x * MG_N;
  constexpr uint LDS = MG_K + 8;
  threadgroup bfloat As[MG_M * LDS];
  threadgroup bfloat Bs[MG_N * LDS];
  threadgroup int arow[MG_M];
  if (tid < MG_M) {
    uint i = r0 + tid;
    int src = -1;
    if (i < rend) src = p.gather_a ? perm[i] / int(p.topk) : int(i);
    arow[tid] = src;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint sm = (sg / 2) * 32, sn = (sg % 2) * 32;
  simdgroup_float8x8 acc[4][4];
  for (uint i = 0; i < 4; i++)
    for (uint j = 0; j < 4; j++) acc[i][j] = simdgroup_float8x8(0);
  const uint rowbytes_q = MODE == 0 ? p.K / 2 : p.K;  // packed bytes per weight row
  for (uint k0 = 0; k0 < p.K; k0 += MG_K) {
    // A: 64 rows x 32 bf16 = 256 uint4, 2 per thread
    for (uint i = tid; i < MG_M * MG_K / 8; i += 128) {
      uint r = i / 4, c8 = (i % 4) * 8;
      int src = arow[r];
      uint4 v = uint4(0);
      if (src >= 0) v = *(device const uint4*)(a + size_t(src) * p.K + k0 + c8);
      *(threadgroup uint4*)(As + r * LDS + c8) = v;
    }
    // B: 64 weight rows x 32 values, dequantized. NVFP4: one uint4 + 2 scales per row; MXFP8: 2 uint4 + 1 scale.
    if (MODE == 0) {
      if (tid < MG_N) {
        uint n = n0 + tid;
        size_t row = size_t(e) * p.N + min(n, p.N - 1);
        uint4 q = w[(row * rowbytes_q + k0 / 2) / 16];
        ushort s2 = *(device const ushort*)(sc + row * (p.K / 16) + k0 / 16);
        float s0 = e4m3_to_f(s2 & 255u), s1 = e4m3_to_f(s2 >> 8);
        uint ws[4] = {q.x, q.y, q.z, q.w};
        threadgroup bfloat* dst = Bs + tid * LDS;
        for (uint j = 0; j < 4; j++) {
          float s = j < 2 ? s0 : s1;
          for (uint i = 0; i < 8; i++) dst[j * 8 + i] = bfloat(fp4h((ws[j] >> (4 * i)) & 15u) * 16384.0f * s);
        }
      }
    } else {
      if (tid < 2 * MG_N) {
        uint rr = tid / 2, half_ = tid % 2;
        uint n = n0 + rr;
        size_t row = size_t(e) * p.N + min(n, p.N - 1);
        uint4 q = w[(row * rowbytes_q + k0 + half_ * 16) / 16];
        float s = e8m0_to_f(sc[row * (p.K / 32) + k0 / 32]);
        uint ws[4] = {q.x, q.y, q.z, q.w};
        threadgroup bfloat* dst = Bs + rr * LDS + half_ * 16;
        for (uint j = 0; j < 4; j++)
          for (uint i = 0; i < 4; i++) dst[j * 4 + i] = bfloat(fp8h((ws[j] >> (8 * i)) & 255u) * 256.0f * s);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < MG_K; kk += 8) {
      simdgroup_bfloat8x8 af[4], bf[4];
      for (uint i = 0; i < 4; i++) simdgroup_load(af[i], As + (sm + i * 8) * LDS + kk, LDS);
      for (uint j = 0; j < 4; j++) simdgroup_load(bf[j], Bs + (sn + j * 8) * LDS + kk, LDS, ulong2(0, 0), true);
      for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++) simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float gscale = MODE == 0 ? gs[e] : 1.0f;
  threadgroup float Cs[4][64];
  for (uint i = 0; i < 4; i++)
    for (uint j = 0; j < 4; j++) {
      simdgroup_store(acc[i][j], Cs[sg], 8);
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (uint el = lane; el < 64; el += 32) {
        uint r = sm + i * 8 + el / 8, c = sn + j * 8 + el % 8;
        uint si = r0 + r, n = n0 + c;
        if (si < rend && n < p.N) {
          uint orow = p.scatter_out ? uint(perm[si]) : si;
          out[size_t(orow) * p.N + n] = f2bf(Cs[sg][el] * gscale);
        }
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

#define MG_INST(M, NAME)                                                                                       \
  template [[host_name(NAME)]] kernel void moe_gemm<M>(                                                       \
      device const uint4*, device const uchar*, device const float*, device const ushort*, device const int*, \
      device const int*, device const uint2*, device const uint*, device ushort*, constant MoeGemmParams&,    \
      uint2, uint, uint, uint);
MG_INST(0, "moe_gemm_nvfp4")
MG_INST(1, "moe_gemm_mxfp8")
