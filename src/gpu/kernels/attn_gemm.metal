// Prefill attention as matrix products over blocks of AG_KEYS keys. Owner: A.
// Port of saetta's fused attention_gemm (kernels/attention_gemm.metal there), adapted to Inkling 2: 3 q heads per
// kv head, head dim 128, relative-position bias, sliding window, sink logit, bf16 cache with ring.
//
// For kv head g the query rows are r = token * 3 + (q head % 3), so all rows share the keys and values of g.
// Per block of keys [j0, j0 + n):
//   ag_scores  S = Q K^T in 64 x 64 tiles (bf16 tiles, fp32 accumulators), + bias, mask; writes
//              sh = exp(s - t) in fp32, with t the row's max over the keys it sees in the 64-key tile, and t to
//              tmax.
//   ag_pv      m = max(m, tile maxes); P = sh * exp(t - m) (fp32 tile), O = exp(m_old - m) O + P V, l likewise.
// ag_init sets m = sink logit, l = 1, O = 0 (the sink is a zero value with its own logit); ag_final writes
// out = bf16(O / l) (* sigmoid(gate)). Scores, P and the P V sums are fp32 throughout (MLX rounds scores and P to bf16); saetta's half P tile was
// faster but cost accuracy on long prompts (p3 KL vs fp32 up to 1.2e-2 against 4.8e-3 exact).

struct AgParams {
  uint T, rows, rows_pad, H, KVH;
  uint cap, len, ring, window, rel_extent;
  uint j0, n, use_gate, t0;  // t0: first token of the slice (rows are relative to it)
};

constant uint AG_KEYS = 1024, AG_D = 128, AG_G = 3, AG_BK = 32;
constant uint AG_LD = AG_BK + 8, AG_VLD = 64 + 8, AG_TILES = AG_KEYS / 64;
constant float AG_SCALE = 0.08838834764831845f;  // 128^-0.5

inline uint ag_tok(constant AgParams& p, uint r) { return p.t0 + min(r, p.rows - 1) / AG_G; }
inline uint ag_qpos(constant AgParams& p, uint r) { return p.len + ag_tok(p, r); }
// Keys of the block row r sees: jj in [lo, hi] (relative to j0; may be outside [0, n)).
inline int ag_hi(constant AgParams& p, uint r) { return int(ag_qpos(p, r)) - int(p.j0); }
inline int ag_lo(constant AgParams& p, uint r) {
  return p.window ? int(ag_qpos(p, r)) - int(p.window) + 1 - int(p.j0) : -int(p.j0);
}
// Key range [beg, end) of the block that the 64-row tile at r0 touches, in multiples of AG_BK; end <= beg: none.
inline int2 ag_tile_keys(constant AgParams& p, uint r0) {
  int lo = max(0, ag_lo(p, r0));
  int hi = min(int(p.n) - 1, ag_hi(p, min(r0 + 63, p.rows - 1)));
  if (hi < lo) return int2(0, 0);
  return int2(lo / int(AG_BK) * int(AG_BK), (hi + int(AG_BK)) / int(AG_BK) * int(AG_BK));
}
inline device const ushort* ag_kv(device const ushort* cache, device const ushort* fresh, constant AgParams& p,
                                  uint g, uint pos) {
  if (pos < p.len) {
    uint slot = p.ring ? pos % p.cap : pos;
    return cache + (ulong(g) * p.cap + slot) * AG_D;
  }
  return fresh + (ulong(pos - p.len) * p.KVH + g) * AG_D;
}

// rpt[dist][16] = rel_proj[16][dist] as fp32 (contiguous per distance).
kernel void ag_rel_t(device const ushort* rel_proj [[buffer(0)]], device float* rpt [[buffer(1)]],
                     constant AgParams& p [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= p.rel_extent || gid.y >= 16) return;
  rpt[gid.x * 16 + gid.y] = bf2f(rel_proj[gid.y * p.rel_extent + gid.x]);
}

kernel void ag_init(device const ushort* sink [[buffer(0)]], device float* m [[buffer(1)]],
                    device float* l [[buffer(2)]], device float* o [[buffer(3)]], constant AgParams& p [[buffer(4)]],
                    uint3 gid [[thread_position_in_grid]]) {
  uint r = gid.y, g = gid.z;
  if (r >= p.rows_pad) return;
  ulong row = ulong(g) * p.rows_pad + r;
  if (gid.x == 0) {
    uint rc = min(r, p.rows - 1);
    m[row] = bf2f(sink[ag_tok(p, r) * p.H + g * AG_G + rc % AG_G]);
    l[row] = 1.0f;
  }
  device float4* op = (device float4*)(o + row * AG_D + gid.x * 16);
  for (uint i = 0; i < 4; i++) op[i] = float4(0);
}

kernel void ag_scores(device const ushort* q [[buffer(0)]], device const ushort* k_new [[buffer(1)]],
                      device const ushort* kc [[buffer(2)]], device float* sh [[buffer(3)]],
                      device float* tmax [[buffer(4)]], device const ushort* rq [[buffer(5)]],
                      device const float* rpt [[buffer(6)]], constant AgParams& p [[buffer(7)]],
                      uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint simd [[simdgroup_index_in_threadgroup]]) {
  // q and key tiles while the product runs, then 32 rows of scores at a time (8 KB)
  threadgroup bfloat qt[2 * 64 * AG_LD];
  threadgroup bfloat* kt = qt + 64 * AG_LD;
  threadgroup float* c = (threadgroup float*)qt;
  uint g = tg.z, r0 = tg.y * 64, k0 = tg.x * 64;
  int2 tk = ag_tile_keys(p, r0);
  if (tk.y <= tk.x || int(k0) >= tk.y || int(k0 + 64) <= tk.x) return;

  uint lr = tid / 2, lc = (tid % 2) * 16;
  bool q_ok = r0 + lr < p.rows;
  bool k_ok = k0 + lr < p.n;
  uint rq_ = min(r0 + lr, p.rows - 1);
  device const ushort* qp = q + ((ulong(ag_tok(p, rq_)) * p.H + g * AG_G + rq_ % AG_G) * AG_D) + lc;
  device const ushort* kp = ag_kv(kc, k_new, p, g, k_ok ? p.j0 + k0 + lr : p.len) + lc;
  threadgroup uint4* qd = (threadgroup uint4*)(qt + lr * AG_LD + lc);
  threadgroup uint4* kd = (threadgroup uint4*)(kt + lr * AG_LD + lc);

  uint sm = (simd / 2) * 32, sn = (simd % 2) * 32;
  simdgroup_float8x8 acc[4][4];
  for (uint i = 0; i < 4; i++)
    for (uint j = 0; j < 4; j++) acc[i][j] = simdgroup_float8x8(0);
  for (uint d0 = 0; d0 < AG_D; d0 += AG_BK) {
    for (uint i = 0; i < 2; i++) {
      qd[i] = q_ok ? ((device const uint4*)(qp + d0))[i] : uint4(0);
      kd[i] = k_ok ? ((device const uint4*)(kp + d0))[i] : uint4(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
    for (uint kk = 0; kk < AG_BK; kk += 8) {
      simdgroup_bfloat8x8 a[4];
      simdgroup_bfloat8x8 b[4];
      for (uint i = 0; i < 4; i++) {
        simdgroup_load(a[i], qt + (sm + 8 * i) * AG_LD + kk, AG_LD);
        simdgroup_load(b[i], kt + (sn + 8 * i) * AG_LD + kk, AG_LD, ulong2(0, 0), true);
      }
      for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // Simdgroups 0, 1 hold rows 0..31, 2, 3 rows 32..63. Thread t then takes 16 keys of row t / 4.
  uint qr = tid / 4, cc = (tid % 4) * 16;
  for (uint half_ = 0; half_ < 2; half_++) {
    if (simd / 2 == half_) {
      for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++) simdgroup_store(acc[i][j], c + (8 * i) * 64 + sn + 8 * j, 64);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint r = r0 + half_ * 32 + qr;
    uint rc = min(r, p.rows - 1);
    int hi = min(ag_hi(p, r), int(p.n) - 1) - int(k0);
    int lo = ag_lo(p, r) - int(k0);
    uint qpos = ag_qpos(p, r);
    uint kbase = p.j0 + k0 + cc;  // position of this thread's first key
    // bias applies to distances < rel_extent: smallest distance of the 16 keys is qpos - (kbase + 15)
    bool need_bias = int(qpos) - int(kbase + 15) < int(p.rel_extent);
    float rv[16];
    if (need_bias) {
      device const ushort* rp = rq + (ulong(ag_tok(p, r)) * p.H + g * AG_G + rc % AG_G) * 16;
      for (uint d = 0; d < 16; d++) rv[d] = bf2f(rp[d]);
    }
    threadgroup const float* crow = c + qr * 64 + cc;
    float sc[16];
    float mx = -INFINITY;
    for (uint j = 0; j < 16; j++) {
      int jj = int(cc + j);
      float s = -INFINITY;
      if (jj >= lo && jj <= hi) {
        s = crow[j] * AG_SCALE;
        uint dist = qpos - (kbase + j);
        if (dist < p.rel_extent) {
          device const float4* rr = (device const float4*)(rpt + dist * 16);
          float b = 0;
          for (uint d4 = 0; d4 < 4; d4++) {
            float4 w = rr[d4];
            b = fma(rv[d4 * 4 + 0], w.x, b);
            b = fma(rv[d4 * 4 + 1], w.y, b);
            b = fma(rv[d4 * 4 + 2], w.z, b);
            b = fma(rv[d4 * 4 + 3], w.w, b);
          }
          s += round_bf(b);
        }
        mx = max(mx, s);
      }
      sc[j] = s;
    }
    mx = max(mx, simd_shuffle_xor(mx, 1));
    mx = max(mx, simd_shuffle_xor(mx, 2));
    ulong row = ulong(g) * p.rows_pad + r;
    if (tid % 4 == 0) tmax[row * AG_TILES + tg.x] = mx;
    device float4* out = (device float4*)(sh + row * AG_KEYS + k0 + cc);
    for (uint j = 0; j < 16; j += 4) {
      float4 v;
      for (uint i = 0; i < 4; i++) v[i] = sc[j + i] == -INFINITY ? 0.0f : exp(sc[j + i] - mx);
      out[j / 4] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// fp32 P and V tiles (exact), 16 keys per step so the threadgroup memory stays under 10 KB (float tiles of 32
// keys spill and run 8 times slower).
constant uint AG_PK = 16, AG_PLD = AG_PK + 4, AG_VLD2 = 64 + 4;

kernel void ag_pv(device const float* sh [[buffer(0)]], device const ushort* v_new [[buffer(1)]],
                  device const ushort* vc [[buffer(2)]], device float* o [[buffer(3)]],
                  device const float* tmax [[buffer(4)]], constant AgParams& p [[buffer(5)]],
                  device const float* m_in [[buffer(6)]], device float* m_out [[buffer(7)]],
                  device const float* l_in [[buffer(8)]], device float* l_out [[buffer(9)]],
                  uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                  uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup float c[64 * AG_PLD + AG_PK * AG_VLD2];  // P tile + V tile; the first 32 x 64 again for the store
  threadgroup float rowa[64];
  threadgroup float rowsum[64];
  threadgroup float* pt = c;
  threadgroup float* vt = pt + 64 * AG_PLD;
  uint g = tg.z, r0 = tg.y * 64, d0 = tg.x * 64;
  if (r0 >= p.rows) return;
  int2 tk = ag_tile_keys(p, r0);

  // Thread t loads P row t / 2 (8 keys of each 16-key step).
  uint pr = tid / 2, pc = (tid % 2) * 8;
  uint r = r0 + pr;
  ulong row = ulong(g) * p.rows_pad + r;
  float m_old = m_in[row];
  if (tk.y <= tk.x) {  // no row of the tile sees the block: O stays, m and l carry over
    if (tg.x == 0 && tid % 2 == 0 && r < p.rows) {
      m_out[row] = m_old;
      l_out[row] = l_in[row];
    }
    return;
  }
  float mx = -INFINITY;
  for (int t = tk.x / 64; t * 64 < tk.y; t++) mx = max(mx, tmax[row * AG_TILES + uint(t)]);
  float m_new = max(m_old, mx);
  float a = exp(m_old - m_new);

  device const float* pp = sh + row * AG_KEYS + pc;
  threadgroup float4* pd = (threadgroup float4*)(pt + pr * AG_PLD + pc);
  uint vr = tid / 8, vcol = (tid % 8) * 8;
  threadgroup float4* vd = (threadgroup float4*)(vt + vr * AG_VLD2 + vcol);

  uint sm = (simd / 2) * 32, sn = (simd % 2) * 32;
  simdgroup_float8x8 acc[4][4];
  for (uint i = 0; i < 4; i++)
    for (uint j = 0; j < 4; j++) acc[i][j] = simdgroup_float8x8(0);
  float sum = 0.0f;
  for (uint k0 = uint(tk.x); k0 < uint(tk.y); k0 += AG_PK) {
    float tm = tmax[row * AG_TILES + k0 / 64];
    float off = tm == -INFINITY ? 0.0f : exp(tm - m_new);
    for (uint i = 0; i < 2; i++) {
      float4 e = ((device const float4*)(pp + k0))[i] * off;
      sum += e.x + e.y + e.z + e.w;
      pd[i] = e;
    }
    if (k0 + vr < p.n) {
      float f[8];
      unpack_bf8(*(device const uint4*)(ag_kv(vc, v_new, p, g, p.j0 + k0 + vr) + d0 + vcol), f);
      vd[0] = float4(f[0], f[1], f[2], f[3]);
      vd[1] = float4(f[4], f[5], f[6], f[7]);
    } else {
      vd[0] = float4(0);
      vd[1] = float4(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
    for (uint kk = 0; kk < AG_PK; kk += 8) {
      simdgroup_float8x8 a8[4];
      simdgroup_float8x8 b8[4];
      for (uint i = 0; i < 4; i++) {
        simdgroup_load(a8[i], pt + (sm + 8 * i) * AG_PLD + kk, AG_PLD);
        simdgroup_load(b8[i], vt + kk * AG_VLD2 + sn + 8 * i, AG_VLD2);
      }
      for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++) simdgroup_multiply_accumulate(acc[i][j], a8[i], b8[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  sum += simd_shuffle_xor(sum, 1);
  if (tid % 2 == 0) {
    rowa[pr] = a;
    rowsum[pr] = sum;
    if (tg.x == 0 && r < p.rows) {
      m_out[row] = m_new;
      l_out[row] = l_in[row] * a + sum;
    }
  }
  // Store 32 rows at a time through the first 8 KB of c; thread t then updates 16 dims of row t / 4.
  uint qr = tid / 4, cc = (tid % 4) * 16;
  for (uint half_ = 0; half_ < 2; half_++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd / 2 == half_) {
      for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++) simdgroup_store(acc[i][j], c + (8 * i) * 64 + sn + 8 * j, 64);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint lr = half_ * 32 + qr;
    if (r0 + lr < p.rows) {
      float ar = rowa[lr];
      device float4* orow = (device float4*)(o + (ulong(g) * p.rows_pad + r0 + lr) * AG_D + d0 + cc);
      threadgroup const float4* crow = (threadgroup const float4*)(c + qr * 64 + cc);
      for (uint j = 0; j < 4; j++) orow[j] = ar * orow[j] + crow[j];
    }
  }
}

// out[token][head][d] = bf16(O / l) (* sigmoid(gate)); thread id = (d, r, g).
kernel void ag_final(device const float* o [[buffer(0)]], device const float* l [[buffer(1)]],
                     device ushort* out [[buffer(2)]], device const ushort* gate [[buffer(3)]],
                     constant AgParams& p [[buffer(4)]], uint3 gid [[thread_position_in_grid]]) {
  uint d = gid.x, r = gid.y, g = gid.z;
  if (r >= p.rows) return;
  ulong row = ulong(g) * p.rows_pad + r;
  ulong oi = (ulong(ag_tok(p, r)) * p.H + g * AG_G + r % AG_G) * AG_D + d;
  float v = round_bf(o[row * AG_D + d] / l[row]);
  if (p.use_gate) v = v * (1.0f / (1.0f + exp(-bf2f(gate[oi]))));
  out[oi] = f2bf(v);
}
