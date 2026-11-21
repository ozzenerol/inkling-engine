// Multi-row forwards (verify at T = 5..16): one pass over W for up to 16 rows of x. Owner: D.
// Per (t, row) the arithmetic is the one of matvec_bf16 (matvec.metal): lane l takes the 8-column chunks
// l, l + 32, ..., fma over the 8 columns in order, then simd_sum over the 32 lanes. So every output row equals
// the 1-row matvec bit for bit, whatever TT / R / S (the library is compiled with MTLMathModeSafe).
// Unlike matvec_bf16 the W chunks of the R rows are held and the x rows are walked one at a time (8 floats live),
// so 16 rows do not need 128 x values in registers. Requires K % 8 == 0.

inline void mr_ldx8(device const bf16_in* p, thread float* o) { unpack_bf8(*(device const packed_uint4*)p, o); }
inline void mr_ldx8(device const float* p, thread float* o) {
  float4 a = *(device const packed_float4*)p, b = *(device const packed_float4*)(p + 4);
  o[0] = a.x; o[1] = a.y; o[2] = a.z; o[3] = a.w;
  o[4] = b.x; o[5] = b.y; o[6] = b.z; o[7] = b.w;
}

// Grid: ceil(N / (R * S)) threadgroups of 32 * S threads. T <= TT rows of x. DYN: loop only over the T rows
// present (no unrolling over t); otherwise all TT rows are computed with the x row clamped.
template <typename XT, typename YT, uint TT, uint R, uint S, bool DYN>
kernel void mr_matvec(device const uint4* W [[buffer(0)]], device const XT* x [[buffer(1)]],
                      device YT* y [[buffer(2)]], constant MatvecShape& s [[buffer(3)]],
                      uint tg [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
  uint row0 = (tg * S + simd) * R;
  if (row0 >= s.N) return;
  const uint kv = s.K / 8;
  const uint nt = min(s.T, TT);
  const uint tend = DYN ? nt : TT;
  float acc[TT][R];
  for (uint t = 0; t < TT; t++)
    for (uint r = 0; r < R; r++) acc[t][r] = 0;
  for (uint k8 = lane; k8 < kv; k8 += 32) {
    float w[R][8];
#pragma clang loop unroll(full)
    for (uint r = 0; r < R; r++) unpack_bf8(W[min(row0 + r, s.N - 1) * kv + k8], w[r]);
    for (uint t = 0; t < tend; t++) {
      float xv[8];
      mr_ldx8(x + min(t, nt - 1) * s.x_stride + k8 * 8, xv);
#pragma clang loop unroll(full)
      for (uint r = 0; r < R; r++) {
        float a = acc[t][r];
        for (uint i = 0; i < 8; i++) a = fma(w[r][i], xv[i], a);
        acc[t][r] = a;
      }
    }
  }
  for (uint t = 0; t < nt; t++)
    for (uint r = 0; r < R; r++) {
      float v = simd_sum(acc[t][r]);
      if (lane == 0 && row0 + r < s.N) y[t * s.y_stride + row0 + r] = YT(v);
    }
}

#define MR_INST(XN, XT, YN, YT, TT, R, S, DN, DYN)                                                            \
  template [[host_name("mr_matvec_" #XN "_" #YN "_t" #TT "_r" #R "_s" #S DN)]] kernel void                    \
  mr_matvec<XT, YT, TT, R, S, DYN>(device const uint4*, device const XT*, device YT*, constant MatvecShape&,  \
                                   uint, uint, uint);
// Candidates for the bench (bench_multirow); the dispatch keeps the winners.
#define MR_SET(TT, R, S)                                \
  MR_INST(bf16, bf16_in, bf16, bf16_out, TT, R, S, "", false) \
  MR_INST(bf16, bf16_in, f32, float, TT, R, S, "", false)     \
  MR_INST(f32, float, f32, float, TT, R, S, "", false)        \
  MR_INST(f32, float, bf16, bf16_out, TT, R, S, "", false)
MR_SET(8, 4, 2)
MR_SET(8, 2, 2)
MR_SET(8, 2, 4)
MR_SET(8, 1, 4)
MR_SET(16, 4, 2)
MR_SET(16, 2, 2)
MR_SET(16, 2, 4)
MR_SET(16, 1, 4)
MR_INST(bf16, bf16_in, bf16, bf16_out, 16, 4, 2, "_dyn", true)
MR_INST(bf16, bf16_in, bf16, bf16_out, 16, 2, 2, "_dyn", true)
MR_INST(bf16, bf16_in, bf16, bf16_out, 16, 2, 4, "_dyn", true)
MR_INST(bf16, bf16_in, bf16, bf16_out, 16, 1, 4, "_dyn", true)

// Several matrices on one input (matvec_multi for T <= 16): same grid and parameters as matvec_multi
// (threadgroup g = rows [g*8, g*8+8) of the concatenated row space, row_start multiples of 4).
template <typename XT, uint TT>
kernel void mr_matvec_multi(device const uint4* W0 [[buffer(0)]], device const uint4* W1 [[buffer(1)]],
                            device const uint4* W2 [[buffer(2)]], device const uint4* W3 [[buffer(3)]],
                            device const uint4* W4 [[buffer(4)]], device const uint4* W5 [[buffer(5)]],
                            device const uint4* W6 [[buffer(6)]], device const uint4* W7 [[buffer(7)]],
                            device void* Y0 [[buffer(8)]], device void* Y1 [[buffer(9)]],
                            device void* Y2 [[buffer(10)]], device void* Y3 [[buffer(11)]],
                            device void* Y4 [[buffer(12)]], device void* Y5 [[buffer(13)]],
                            device void* Y6 [[buffer(14)]], device void* Y7 [[buffer(15)]],
                            device const XT* x [[buffer(16)]], constant MatvecMultiParams& s [[buffer(17)]],
                            uint tg [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint R = 4;
  uint grow0 = (tg * 2 + simd) * R;
  if (grow0 >= s.row_start[s.count]) return;
  uint mi = 0;
  while (mi + 1 < s.count && grow0 >= s.row_start[mi + 1]) mi++;
  device const uint4* Ws[8] = {W0, W1, W2, W3, W4, W5, W6, W7};
  device void* Ys[8] = {Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7};
  device const uint4* W = Ws[mi];
  const uint N = s.row_start[mi + 1] - s.row_start[mi];
  const uint row0 = grow0 - s.row_start[mi];
  const uint kv = s.K / 8;
  const uint nt = min(s.T, TT);
  float acc[TT][R];
  for (uint t = 0; t < TT; t++)
    for (uint r = 0; r < R; r++) acc[t][r] = 0;
  device const XT* xm = x + s.x_off[mi];
  for (uint k8 = lane; k8 < kv; k8 += 32) {
    float w[R][8];
#pragma clang loop unroll(full)
    for (uint r = 0; r < R; r++) unpack_bf8(W[min(row0 + r, N - 1) * kv + k8], w[r]);
    for (uint t = 0; t < TT; t++) {
      float xv[8];
      mr_ldx8(xm + min(t, nt - 1) * s.K + k8 * 8, xv);
#pragma clang loop unroll(full)
      for (uint r = 0; r < R; r++) {
        float a = acc[t][r];
        for (uint i = 0; i < 8; i++) a = fma(w[r][i], xv[i], a);
        acc[t][r] = a;
      }
    }
  }
  for (uint t = 0; t < nt; t++)
    for (uint r = 0; r < R; r++) {
      float v = simd_sum(acc[t][r]);
      uint row = row0 + r;
      if (lane == 0 && row < N) {
        if (s.dt_out[mi] == 0) ((device float*)Ys[mi])[t * N + row] = v;
        else ((device ushort*)Ys[mi])[t * N + row] = f2bf(v);
      }
    }
}
#define MR_MULTI_INST(XN, XT, TT)                                                                               \
  template [[host_name("mr_matvec_multi_" #XN "_t" #TT)]] kernel void mr_matvec_multi<XT, TT>(                  \
      device const uint4*, device const uint4*, device const uint4*, device const uint4*, device const uint4*,  \
      device const uint4*, device const uint4*, device const uint4*, device void*, device void*, device void*,  \
      device void*, device void*, device void*, device void*, device void*, device const XT*,                   \
      constant MatvecMultiParams&, uint, uint, uint);
MR_MULTI_INST(bf16, bf16_in, 8)
MR_MULTI_INST(bf16, bf16_in, 16)
MR_MULTI_INST(f32, float, 8)
MR_MULTI_INST(f32, float, 16)

// ---------------------------------------------------------------------------------------------------------------
// Attention for 5..16 query rows (verify), split over keys. Same per-row arithmetic as attn_decode_split
// (attn_decode.metal, B): splits aligned to absolute key positions, inside a split the 16-key blocks alternate
// between two accumulators (even / odd blocks) that are merged at the end, frag row = rr % 8, key column = key % 16.
// The difference is the layout: a threadgroup has 2 * NFG simdgroups, simdgroup (f, parity) holds frag row group f
// (8 of the 3T rows) for the blocks of one parity, and the K / V tile of a block is staged ONCE for all the row
// groups of that parity. So K and V are read once per split whatever T is, and no simdgroup holds more than 16
// output frags. Partials: part[(kvh * nsplit + split) * prow + rr] = (m, l, -, -, o[128]).
struct MrAttnParams {
  uint T, H, KVH;
  uint cap, len, ring, window, rel_extent;
  uint split_keys, nsplit;
  uint causal, first, use_sink, use_gate;  // as AttnDecParams
  uint prow;                               // partial rows per (kvh, split) (>= 3T)
};

// ABL (bench only, results wrong): 1 = no relative bias, 2 = no P V product, 4 = no exp.
template <uint NFG, uint ABL>
kernel void mr_attn_split(device const ushort* q [[buffer(0)]], device const ushort* k_new [[buffer(1)]],
                          device const ushort* v_new [[buffer(2)]], device const ushort* r [[buffer(3)]],
                          device const ushort* rel_proj [[buffer(4)]], device const ushort* kc [[buffer(5)]],
                          device const ushort* vc [[buffer(6)]], device float* part [[buffer(7)]],
                          constant MrAttnParams& p [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint LDK = 128 + 8, BK = 16;
  constexpr uint TILE = BK * LDK;            // bfloats per staged tile
  constexpr uint KVN = 2 * 2 * TILE;         // per parity: K tile, V tile
  constexpr uint QN = 8 * NFG * 128;
  constexpr uint NTH = 64 * NFG;
  threadgroup bfloat TG[KVN + QN];           // reused as fp32 for the merge at the end
  threadgroup bfloat* Qst = TG + KVN;
  const uint kvh = tg.x, split = tg.y;
  const uint G = p.H / p.KVH;
  const uint R = G * p.T;
  const uint f = sg / 2, par = sg % 2;
  for (uint i = tid; i < QN; i += NTH) {
    uint rr = i / 128, d = i % 128;
    uint rc = min(rr, R - 1), t = rc / G, h = kvh * G + rc % G;
    Qst[i] = as_type<bfloat>(q[(t * p.H + h) * 128 + d]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_bfloat8x8 qf[16];
  for (uint d = 0; d < 16; d++) simdgroup_load(qf[d], Qst + (f * 8) * 128 + d * 8, 128);

  short2 fc = frag_coord(lane);
  const uint fm = uint(fc.y), fn = uint(fc.x);
  const uint rr = f * 8 + fm;
  const bool rok = rr < R;
  float rrv[16];
  uint qpos;
  {
    uint rc = min(rr, R - 1), t = rc / G, h = kvh * G + rc % G;
    qpos = p.len + t;
    for (uint d = 0; d < 16; d++) rrv[d] = p.rel_extent ? bf2f(r[(t * p.H + h) * 16 + d]) : 0.0f;
  }
  const uint last = p.len + p.T - 1;
  const uint kbase = (p.first / p.split_keys + split) * p.split_keys;
  const uint k0 = max(kbase, p.first);
  const uint k1 = min(kbase + p.split_keys, last + 1);
  const float scale = 0.08838834764831845f;
  float m = -INFINITY, l = 0;
  simdgroup_float8x8 of[16];
  for (uint j = 0; j < 16; j++) of[j] = simdgroup_float8x8(0);
  threadgroup bfloat* K_ = TG + par * 2 * TILE;
  threadgroup bfloat* V_ = K_ + TILE;
  const uint nblk = k1 > kbase ? (k1 - kbase + 2 * BK - 1) / (2 * BK) : 0;  // same trip count for both parities
  const uint tidp = f * 32 + lane;                                           // index among this parity's threads
  for (uint it = 0; it < nblk; it++) {
    const uint kb = kbase + par * BK + it * 2 * BK;
    const bool live = kb < k1;
    if (live)
      for (uint i = tidp; i < BK * 16; i += 32 * NFG) {
        uint kr = i / 16, c8 = (i % 16) * 8;
        uint pos = clamp(kb + kr, k0, k1 - 1);
        device const ushort* kp;
        device const ushort* vp;
        if (pos < p.len) {
          uint slot = p.ring ? pos % p.cap : pos;
          kp = kc + (size_t(kvh) * p.cap + slot) * 128;
          vp = vc + (size_t(kvh) * p.cap + slot) * 128;
        } else {
          kp = k_new + (size_t(pos - p.len) * p.KVH + kvh) * 128;
          vp = v_new + (size_t(pos - p.len) * p.KVH + kvh) * 128;
        }
        *(threadgroup uint4*)(K_ + kr * LDK + c8) = *(device const uint4*)(kp + c8);
        *(threadgroup uint4*)(V_ + kr * LDK + c8) = *(device const uint4*)(vp + c8);
      }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (live) {
      simdgroup_float8x8 sf[2] = {simdgroup_float8x8(0), simdgroup_float8x8(0)};
      for (uint d = 0; d < 16; d++)
        for (uint j = 0; j < 2; j++) {
          simdgroup_bfloat8x8 kf;
          simdgroup_load(kf, K_ + (j * 8) * LDK + d * 8, LDK, ulong2(0, 0), true);
          simdgroup_multiply_accumulate(sf[j], qf[d], kf, sf[j]);
        }
      float cmax = -INFINITY;
      for (uint j = 0; j < 2; j++) {
        thread auto& el = sf[j].thread_elements();
        for (uint e = 0; e < 2; e++) {
          uint kpos = kb + j * 8 + fn + e;
          bool vis = rok && kpos >= k0 && kpos < k1 &&
                     (!p.causal || (kpos <= qpos && (p.window == 0 || qpos - kpos < p.window)));
          float sc = -INFINITY;
          if (vis) {
            sc = el[e] * scale;
            uint dist = qpos - kpos;
            if (!(ABL & 1) && p.causal && dist < p.rel_extent) {
              float b = 0;
              for (uint d = 0; d < 16; d++) b = fma(rrv[d], bf2f(rel_proj[d * p.rel_extent + dist]), b);
              sc += round_bf(b);
            }
          }
          el[e] = sc;
          cmax = max(cmax, sc);
        }
      }
      cmax = max(cmax, simd_shuffle_xor(cmax, 1));
      cmax = max(cmax, simd_shuffle_xor(cmax, 8));
      float mn = max(m, cmax);
      float corr = mn == -INFINITY ? 1.0f : ((ABL & 4) ? 1.0f : exp(m - mn));
      float psum = 0;
      for (uint j = 0; j < 2; j++) {
        thread auto& el = sf[j].thread_elements();
        for (uint e = 0; e < 2; e++) {
          float pv = el[e] == -INFINITY ? 0.0f : ((ABL & 4) ? el[e] - mn : exp(el[e] - mn));
          el[e] = pv;
          psum += pv;
        }
      }
      psum += simd_shuffle_xor(psum, 1);
      psum += simd_shuffle_xor(psum, 8);
      l = l * corr + psum;
      m = mn;
      if (!(ABL & 2))
        for (uint j = 0; j < 16; j++) {
          thread auto& oe = of[j].thread_elements();
          oe[0] *= corr;
          oe[1] *= corr;
          for (uint kk = 0; kk < 2; kk++) {
            simdgroup_float8x8 vf;
            thread auto& ve = vf.thread_elements();
            threadgroup const bfloat* vr = V_ + (kk * 8 + fm) * LDK + j * 8 + fn;
            ve[0] = float(vr[0]);
            ve[1] = float(vr[1]);
            simdgroup_multiply_accumulate(of[j], sf[kk], vf, of[j]);
          }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // merge the two parities of each row group: parity 1 goes through threadgroup memory (fp32)
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup float* xo = (threadgroup float*)TG;          // 8*NFG rows x 128
  threadgroup float* xm = xo + 8 * NFG * 128;              // m, l per row
  if (par == 1) {
    xm[rr * 2] = m;
    xm[rr * 2 + 1] = l;
    for (uint j = 0; j < 16; j++) {
      thread auto& oe = of[j].thread_elements();
      xo[rr * 128 + j * 8 + fn] = oe[0];
      xo[rr * 128 + j * 8 + fn + 1] = oe[1];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (par == 0 && rok) {
    float m1 = xm[rr * 2], l1 = xm[rr * 2 + 1];
    float M = max(m, m1);
    float c0 = m == -INFINITY ? 0.0f : exp(m - M);
    float c1 = m1 == -INFINITY ? 0.0f : exp(m1 - M);
    device float* dst = part + (size_t(kvh * p.nsplit + split) * p.prow + rr) * 132;
    dst[0] = M;
    dst[1] = l * c0 + l1 * c1;
    for (uint j = 0; j < 16; j++) {
      thread auto& oe = of[j].thread_elements();
      for (uint e = 0; e < 2; e++) {
        uint d = j * 8 + fn + e;
        dst[4 + d] = oe[e] * c0 + xo[rr * 128 + d] * c1;
      }
    }
  }
}
#define MR_ATTN_INST(NAME, NFG, ABL)                                                                           \
  template [[host_name(NAME)]] kernel void mr_attn_split<NFG, ABL>(                                            \
      device const ushort*, device const ushort*, device const ushort*, device const ushort*,                  \
      device const ushort*, device const ushort*, device const ushort*, device float*, constant MrAttnParams&, \
      uint2, uint, uint, uint);
MR_ATTN_INST("mr_attn_split_g1", 1, 0)
MR_ATTN_INST("mr_attn_split_g2", 2, 0)
MR_ATTN_INST("mr_attn_split_g3", 3, 0)
MR_ATTN_INST("mr_attn_split_g4", 4, 0)
MR_ATTN_INST("mr_attn_split_g5", 5, 0)
MR_ATTN_INST("mr_attn_split_g6", 6, 0)
MR_ATTN_INST("mr_attn_split_g6_abl1", 6, 1)
MR_ATTN_INST("mr_attn_split_g6_abl2", 6, 2)
MR_ATTN_INST("mr_attn_split_g6_abl4", 6, 4)
MR_ATTN_INST("mr_attn_split_g6_abl7", 6, 7)

// One threadgroup (128 threads = dims) per (t, h): reduce the splits + sink, write bf16. The arithmetic of
// attn_decode_combine with the partial row stride from the parameters.
kernel void mr_attn_combine(device const float* part [[buffer(0)]], device const ushort* sink [[buffer(1)]],
                            device ushort* out [[buffer(2)]], constant MrAttnParams& p [[buffer(3)]],
                            device const ushort* gate [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]]) {
  uint t = tg.x, h = tg.y;
  uint G = p.H / p.KVH, kvh = h / G, rr = t * G + h % G;
  float sv = p.use_sink ? bf2f(sink[t * p.H + h]) : -INFINITY;
  float M = sv;
  for (uint s = 0; s < p.nsplit; s++) M = max(M, part[(size_t(kvh * p.nsplit + s) * p.prow + rr) * 132]);
  float L = p.use_sink ? exp(sv - M) : 0.0f, O = 0;
  for (uint s = 0; s < p.nsplit; s++) {
    device const float* src = part + (size_t(kvh * p.nsplit + s) * p.prow + rr) * 132;
    if (src[0] == -INFINITY) continue;
    float c = exp(src[0] - M);
    L += src[1] * c;
    O += src[4 + tid] * c;
  }
  uint oi = (t * p.H + h) * 128 + tid;
  float o = round_bf(O / L);
  if (p.use_gate) o = o * (1.0f / (1.0f + exp(-bf2f(gate[oi]))));
  out[oi] = f2bf(o);
}

// ---------------------------------------------------------------------------------------------------------------
// MoE for 5..16 rows with the expert union: the weights of an expert are read once and applied to every row that
// routed to it (16 rows x 8 of 128 experts touch ~82 distinct experts even with independent routing).
// Per (route, output row) the arithmetic is the one of moe_gateup / moe_down (moe_fast.metal, B): same lane ->
// chunk mapping, same fma order inside a chunk, same lane reduction, same scale / round / activation; so every row
// equals the T = 1 result bit for bit. Steps:
//   mr_moe_group   idx [routes] -> ulist: per used expert (in expert order) {expert, count, routes[16]}; uhdr[0] = n
//   mr_moe_gateup  a[route, n]  = bf16(act(bf16(g), bf16(u)))          grid (slot, N / 16)
//   mr_moe_down    dd[route, n] = bf16(down[e] a[route])               grid (slot, N / 32)
//   mr_moe_wsum    out[t, n]    = bf16(sum_k dd[t*8+k, n] * w[t, k])   (k in order 0..7)
constant uint MRU = 18;  // ints per ulist slot
constant uint MRP = 4;   // rows of one expert handled per pass over its weights

struct MrGroupParams {
  uint routes, E;
};
kernel void mr_moe_group(device const int* idx [[buffer(0)]], device int* ulist [[buffer(1)]],
                         device int* uhdr [[buffer(2)]], constant MrGroupParams& p [[buffer(3)]],
                         uint e [[thread_index_in_threadgroup]]) {
  threadgroup uint present[128];
  int rts[16];
  uint cnt = 0;
  for (uint i = 0; i < p.routes; i++)
    if (uint(idx[i]) == e && cnt < 16) rts[cnt++] = int(i);
  present[e] = cnt ? 1u : 0u;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint slot = 0, total = 0;
  for (uint j = 0; j < p.E; j++) {
    if (j < e) slot += present[j];
    total += present[j];
  }
  if (e == 0) uhdr[0] = int(total);
  if (cnt) {
    device int* u = ulist + slot * MRU;
    u[0] = int(e);
    u[1] = int(cnt);
    for (uint i = 0; i < cnt; i++) u[2 + i] = rts[i];
  }
}

inline void mr_ldbf32(device const ushort* p, thread float* o) {  // 32 bf16 -> float
  device const packed_uint4* q = (device const packed_uint4*)p;
  unpack_bf8(q[0], o);
  unpack_bf8(q[1], o + 8);
  unpack_bf8(q[2], o + 16);
  unpack_bf8(q[3], o + 24);
}
// dot_fp4_32 / dot_fp8_16 of moe_fast.metal with x in registers
inline float mr_dot_fp4_32(uint4 w, thread const float* x, float s0, float s1) {
  float a = 0, b = 0;
  uint ws[4] = {w.x, w.y, w.z, w.w};
  for (uint j = 0; j < 2; j++)
    for (uint i = 0; i < 8; i++) a = fma(x[j * 8 + i], fp4h((ws[j] >> (4 * i)) & 15u), a);
  for (uint j = 2; j < 4; j++)
    for (uint i = 0; i < 8; i++) b = fma(x[j * 8 + i], fp4h((ws[j] >> (4 * i)) & 15u), b);
  return a * s0 + b * s1;
}
inline float mr_dot_fp8_16(uint4 w, thread const float* x) {
  float a = 0;
  uint ws[4] = {w.x, w.y, w.z, w.w};
  for (uint j = 0; j < 4; j++)
    for (uint i = 0; i < 4; i++) a = fma(x[j * 4 + i], fp8h((ws[j] >> (8 * i)) & 255u), a);
  return a;
}

// MODE 0 = NVFP4, 1 = MXFP8. Threadgroup = 4 simdgroups x 4 rows x 8 lanes (as moe_gateup).
template <uint MODE>
kernel void mr_moe_gateup(device const uint4* gw [[buffer(0)]], device const uchar* gs [[buffer(1)]],
                          device const float* gg [[buffer(2)]], device const uint4* uw [[buffer(3)]],
                          device const uchar* us [[buffer(4)]], device const float* ug [[buffer(5)]],
                          device const ushort* x [[buffer(6)]], device const int* ulist [[buffer(7)]],
                          device const int* uhdr [[buffer(8)]], device ushort* a [[buffer(9)]],
                          constant MoeFastParams& p [[buffer(10)]], uint2 tg [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint slot = tg.x;
  if (slot >= uint(uhdr[0])) return;
  device const int* ul = ulist + slot * MRU;
  const uint e = uint(ul[0]), cnt = uint(ul[1]);
  uint n = (tg.y * 4 + sg) * 4 + lane / 8;
  const uint sub = lane % 8;
  const bool valid = n < p.N;
  n = min(n, p.N - 1);
  const size_t row = size_t(e) * p.N + n;
  for (uint r0 = 0; r0 < cnt; r0 += MRP) {
    const uint nr = min(MRP, cnt - r0);
    uint rt[MRP];
    float g[MRP], u[MRP];
    for (uint q = 0; q < MRP; q++) {
      rt[q] = uint(ul[2 + r0 + min(q, nr - 1)]);
      g[q] = 0;
      u[q] = 0;
    }
    if (MODE == 0) {
      const uint chunks = p.K / 32;
      device const uint4* gr = gw + row * chunks;
      device const uint4* ur = uw + row * chunks;
      device const ushort* gsr = (device const ushort*)(gs + row * (p.K / 16));
      device const ushort* usr = (device const ushort*)(us + row * (p.K / 16));
      for (uint c = sub; c < chunks; c += 8) {
        const uint gsc = gsr[c], usc = usr[c];
        const uint4 gwc = gr[c], uwc = ur[c];
        const float gs0 = e4m3_scale(gsc & 255u), gs1 = e4m3_scale(gsc >> 8);
        const float us0 = e4m3_scale(usc & 255u), us1 = e4m3_scale(usc >> 8);
        for (uint q = 0; q < nr; q++) {
          float xv[32];
          mr_ldbf32(x + (rt[q] / p.topk) * p.K + c * 32, xv);
          g[q] += mr_dot_fp4_32(gwc, xv, gs0, gs1);
          u[q] += mr_dot_fp4_32(uwc, xv, us0, us1);
        }
      }
    } else {
      const uint chunks = p.K / 16;
      device const uint4* gr = gw + row * chunks;
      device const uint4* ur = uw + row * chunks;
      device const uchar* gsr = gs + row * (p.K / 32);
      device const uchar* usr = us + row * (p.K / 32);
      for (uint c2 = sub; c2 < chunks / 2; c2 += 8) {
        const uint4 g0 = gr[2 * c2], g1 = gr[2 * c2 + 1], u0 = ur[2 * c2], u1 = ur[2 * c2 + 1];
        const float gsc = e8m0_to_f(gsr[c2]), usc = e8m0_to_f(usr[c2]);
        for (uint q = 0; q < nr; q++) {
          float xv[32];
          mr_ldbf32(x + (rt[q] / p.topk) * p.K + c2 * 32, xv);
          float gp = mr_dot_fp8_16(g0, xv) + mr_dot_fp8_16(g1, xv + 16);
          float up = mr_dot_fp8_16(u0, xv) + mr_dot_fp8_16(u1, xv + 16);
          g[q] = fma(gp, gsc, g[q]);
          u[q] = fma(up, usc, u[q]);
        }
      }
    }
    for (uint q = 0; q < nr; q++) {
      float gq = g[q], uq = u[q];
      for (uint off = 4; off > 0; off >>= 1) {
        gq += simd_shuffle_down(gq, off);
        uq += simd_shuffle_down(uq, off);
      }
      if (sub == 0 && valid) {
        float gf, uf;
        if (MODE == 0) {
          gf = gq * (16384.0f * gg[e]);
          uf = uq * (16384.0f * ug[e]);
        } else {
          gf = gq * 256.0f;
          uf = uq * 256.0f;
        }
        a[size_t(rt[q]) * p.N + n] = f2bf(softcap_act(round_bf(gf), round_bf(uf)));
      }
    }
  }
}

// Threadgroup = 4 simdgroups x 8 rows x 4 lanes (moe_down: 4 lanes per (route, row)). a bf16 [routes, K];
// dd bf16 [routes, N].
template <uint MODE>
kernel void mr_moe_down(device const uint4* dw [[buffer(0)]], device const uchar* ds [[buffer(1)]],
                        device const float* dg [[buffer(2)]], device const ushort* a [[buffer(3)]],
                        device const int* ulist [[buffer(4)]], device const int* uhdr [[buffer(5)]],
                        device ushort* dd [[buffer(6)]], constant MoeFastParams& p [[buffer(7)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint slot = tg.x;
  if (slot >= uint(uhdr[0])) return;
  device const int* ul = ulist + slot * MRU;
  const uint e = uint(ul[0]), cnt = uint(ul[1]);
  uint n = tg.y * 32 + sg * 8 + lane / 4;
  const uint sub = lane % 4;
  const bool valid = n < p.N;
  n = min(n, p.N - 1);
  const size_t row = size_t(e) * p.N + n;
  for (uint r0 = 0; r0 < cnt; r0 += MRP) {
    const uint nr = min(MRP, cnt - r0);
    uint rt[MRP];
    float d[MRP];
    for (uint q = 0; q < MRP; q++) {
      rt[q] = uint(ul[2 + r0 + min(q, nr - 1)]);
      d[q] = 0;
    }
    if (MODE == 0) {
      const uint chunks = p.K / 32;
      device const uint4* wr = dw + row * chunks;
      device const ushort* sr = (device const ushort*)(ds + row * (p.K / 16));
      for (uint c = sub; c < chunks; c += 4) {
        const uint sc = sr[c];
        const uint4 wc = wr[c];
        const float s0 = e4m3_scale(sc & 255u), s1 = e4m3_scale(sc >> 8);
        for (uint q = 0; q < nr; q++) {
          float xv[32];
          mr_ldbf32(a + size_t(rt[q]) * p.K + c * 32, xv);
          d[q] += mr_dot_fp4_32(wc, xv, s0, s1);
        }
      }
    } else {
      const uint chunks = p.K / 16;
      device const uint4* wr = dw + row * chunks;
      device const uchar* sr = ds + row * (p.K / 32);
      for (uint c2 = sub; c2 < chunks / 2; c2 += 4) {
        const uint4 w0 = wr[2 * c2], w1 = wr[2 * c2 + 1];
        const float sc = e8m0_to_f(sr[c2]);
        for (uint q = 0; q < nr; q++) {
          float xv[32];
          mr_ldbf32(a + size_t(rt[q]) * p.K + c2 * 32, xv);
          float dp = mr_dot_fp8_16(w0, xv) + mr_dot_fp8_16(w1, xv + 16);
          d[q] = fma(dp, sc, d[q]);
        }
      }
    }
    for (uint q = 0; q < nr; q++) {
      float dq = d[q];
      dq += simd_shuffle_down(dq, 2);
      dq += simd_shuffle_down(dq, 1);
      float dv = MODE == 0 ? dq * (16384.0f * dg[e]) : dq * 256.0f;
      if (sub == 0 && valid) dd[size_t(rt[q]) * p.N + n] = f2bf(dv);
    }
  }
}

struct MrWsumParams {
  uint T, N, topk;
};
kernel void mr_moe_wsum(device const ushort* dd [[buffer(0)]], device const float* wts [[buffer(1)]],
                        device ushort* out [[buffer(2)]], constant MrWsumParams& p [[buffer(3)]],
                        uint2 gid [[thread_position_in_grid]]) {
  const uint n = gid.x, t = gid.y;
  if (n >= p.N || t >= p.T) return;
  float contrib[8];
  for (uint kk = 0; kk < 8; kk++) {
    const uint route = t * p.topk + kk;
    contrib[kk] = bf2f(dd[size_t(route) * p.N + n]) * wts[route];
  }
  float s = 0;
  for (uint kk = 0; kk < 8; kk++) s += contrib[kk];
  out[size_t(t) * p.N + n] = f2bf(s);
}

#define MR_MOE_INST(NAME, MODE)                                                                                \
  template [[host_name("mr_moe_gateup_" NAME)]] kernel void mr_moe_gateup<MODE>(                               \
      device const uint4*, device const uchar*, device const float*, device const uint4*, device const uchar*, \
      device const float*, device const ushort*, device const int*, device const int*, device ushort*,         \
      constant MoeFastParams&, uint2, uint, uint);                                                             \
  template [[host_name("mr_moe_down_" NAME)]] kernel void mr_moe_down<MODE>(                                   \
      device const uint4*, device const uchar*, device const float*, device const ushort*, device const int*,  \
      device const int*, device ushort*, constant MoeFastParams&, uint2, uint, uint);
MR_MOE_INST("nvfp4", 0)
MR_MOE_INST("mxfp8", 1)
