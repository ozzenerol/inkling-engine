// Prefill attention (flash style, simdgroup matrices). Owner: B.
// Threadgroup = (32-query block, kv head) with 12 simdgroups: simdgroup sg serves q head kvh*3 + sg/4 and query
// rows (sg%4)*8 .. +8 of the block. Key blocks of 16: K tile bf16, V tile fp32 in threadgroup memory, shared by
// the 96 rows. S = Q K^T (bf16 frags, fp32 acc), bias + mask + online softmax in fp32 (sink = initial max with
// l = 1 and a zero value), O += P V in fp32 frags; row rescale through a diagonal 8x8 matrix.

struct AttnPreParams {
  uint T, H, KVH;
  uint cap, len, ring, window, rel_extent;
};

constant uint AP_BQ = 32, AP_BK = 16, AP_SG = 12;

kernel void attn_prefill(device const ushort* q [[buffer(0)]], device const ushort* k_new [[buffer(1)]],
                         device const ushort* v_new [[buffer(2)]], device const ushort* r [[buffer(3)]],
                         device const ushort* rel_proj [[buffer(4)]], device const ushort* sink [[buffer(5)]],
                         device const ushort* kc [[buffer(6)]], device const ushort* vc [[buffer(7)]],
                         device ushort* out [[buffer(8)]], constant AttnPreParams& p [[buffer(9)]],
                         uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint LDK = 128 + 8;
  constexpr uint LDV = 128 + 4;
  threadgroup bfloat Ks[AP_BK * LDK];
  threadgroup float Vs[AP_BK * LDV];
  threadgroup float Ss[AP_SG][8 * AP_BK];
  threadgroup float Dg[AP_SG][64];
  threadgroup float Rp[16][AP_BQ + AP_BK];  // rel_proj columns d0 .. d0+47 for the current key block

  uint q0 = tg.x * AP_BQ, kvh = tg.y;
  uint G = p.H / p.KVH;
  uint h = kvh * G + sg / 4;
  uint row0 = (sg % 4) * 8;  // within the block
  // lane -> (row, 4 columns) for softmax work: row = lane / 4, cols (lane % 4) * 4 .. +4
  uint lrow = lane / 4, lc0 = (lane % 4) * 4;
  uint t_l = q0 + row0 + lrow;  // query index of this lane's row
  bool row_ok = t_l < p.T;
  uint qpos_l = p.len + min(t_l, p.T - 1);
  float rr[16];
  for (uint d = 0; d < 16; d++) rr[d] = bf2f(r[(min(t_l, p.T - 1) * p.H + h) * 16 + d]);
  const float scale = 0.08838834764831845f;

  // Q frags: rows q0+row0 .. +8 of head h, 16 frags over the 128 dims (rows past T clamp to T-1).
  simdgroup_bfloat8x8 qf[16];
  {
    // stage through Ks area is not possible (shared); load per row with clamping via threadgroup Ss scratch
    threadgroup bfloat* stage = (threadgroup bfloat*)Ss[sg];  // 8*16 floats = 512 bytes = 256 bf16; need 8x128
    for (uint dblk = 0; dblk < 128; dblk += 32) {
      for (uint e = lane; e < 8 * 32; e += 32) {
        uint rI = e / 32, c = e % 32;
        uint t = min(q0 + row0 + rI, p.T - 1);
        stage[rI * 32 + c] = as_type<bfloat>(q[(t * p.H + h) * 128 + dblk + c]);
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (uint f = 0; f < 4; f++) simdgroup_load(qf[dblk / 8 + f], stage + f * 8, 32);
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  simdgroup_float8x8 of[16];
  for (uint j = 0; j < 16; j++) of[j] = simdgroup_float8x8(0);
  // running max / sum for this lane's row (all 4 lanes of a row hold the same values)
  float m = row_ok ? bf2f(sink[min(t_l, p.T - 1) * p.H + h]) : 0.0f;
  float l = 1.0f;

  uint qpos_first = p.len + q0, qpos_last = p.len + min(q0 + AP_BQ, p.T) - 1;
  uint kfirst = 0;
  if (p.window && qpos_first + 1 > p.window) kfirst = qpos_first + 1 - p.window;
  kfirst = kfirst / AP_BK * AP_BK;
  for (uint kb = kfirst; kb <= qpos_last; kb += AP_BK) {
    // load K (bf16) and V (fp32) tiles: 16 keys x 128 dims; 384 threads
    for (uint i = tid; i < AP_BK * 16; i += 32 * AP_SG) {
      uint kr = i / 16, c8 = (i % 16) * 8;
      uint pos = min(kb + kr, qpos_last);
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
      uint4 kv4 = *(device const uint4*)(kp + c8);
      *(threadgroup uint4*)(Ks + kr * LDK + c8) = kv4;
      uint4 vv4 = *(device const uint4*)(vp + c8);
      float vf[8];
      unpack_bf8(vv4, vf);
      for (uint j = 0; j < 8; j++) Vs[kr * LDV + c8 + j] = vf[j];
    }
    // rel_proj columns for dists in [qpos_first - (kb+15), qpos_last - kb]
    int dlo = int(qpos_first) - int(kb + AP_BK - 1);
    bool need_bias = dlo < int(p.rel_extent);
    if (need_bias)
      for (uint i = tid; i < 16 * (AP_BQ + AP_BK); i += 32 * AP_SG) {
        uint d = i / (AP_BQ + AP_BK), c = i % (AP_BQ + AP_BK);
        int dist = dlo + int(c);
        Rp[d][c] = (dist >= 0 && dist < int(p.rel_extent)) ? bf2f(rel_proj[d * p.rel_extent + uint(dist)]) : 0.0f;
      }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // S = Q K^T: 8 rows x 16 keys = 2 frags
    simdgroup_float8x8 sf[2] = {simdgroup_float8x8(0), simdgroup_float8x8(0)};
    for (uint d = 0; d < 16; d++)
      for (uint j = 0; j < 2; j++) {
        simdgroup_bfloat8x8 kf;
        simdgroup_load(kf, Ks + (j * 8) * LDK + d * 8, LDK, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(sf[j], qf[d], kf, sf[j]);
      }
    simdgroup_store(sf[0], Ss[sg], AP_BK);
    simdgroup_store(sf[1], Ss[sg] + 8, AP_BK);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    // bias, mask, online softmax: lane handles row lrow, keys lc0 .. lc0+3
    float s4[4];
    float cmax = -INFINITY;
    for (uint c = 0; c < 4; c++) {
      uint kpos = kb + lc0 + c;
      float sc = -INFINITY;
      bool vis = row_ok && kpos <= qpos_l && (p.window == 0 || qpos_l - kpos < p.window) && kpos <= qpos_last;
      if (vis) {
        sc = Ss[sg][lrow * AP_BK + lc0 + c] * scale;
        uint dist = qpos_l - kpos;
        if (dist < p.rel_extent) {
          uint col = uint(int(dist) - dlo);
          float b = 0;
          for (uint d = 0; d < 16; d++) b = fma(rr[d], Rp[d][col], b);
          sc += round_bf(b);
        }
      }
      s4[c] = sc;
      cmax = max(cmax, sc);
    }
    cmax = max(cmax, simd_shuffle_xor(cmax, 1));
    cmax = max(cmax, simd_shuffle_xor(cmax, 2));
    float mn = max(m, cmax);
    float corr = exp(m - mn);
    float psum = 0;
    for (uint c = 0; c < 4; c++) {
      float pv = s4[c] == -INFINITY ? 0.0f : exp(s4[c] - mn);
      Ss[sg][lrow * AP_BK + lc0 + c] = pv;
      psum += pv;
    }
    psum += simd_shuffle_xor(psum, 1);
    psum += simd_shuffle_xor(psum, 2);
    l = l * corr + psum;
    m = mn;
    // diagonal rescale matrix
    for (uint e = lane; e < 64; e += 32) Dg[sg][e] = 0;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (lane % 4 == 0) Dg[sg][lrow * 8 + lrow] = corr;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 dg;
    simdgroup_load(dg, Dg[sg], 8);
    simdgroup_float8x8 pf[2];
    simdgroup_load(pf[0], Ss[sg], AP_BK);
    simdgroup_load(pf[1], Ss[sg] + 8, AP_BK);
    for (uint j = 0; j < 16; j++) {
      simdgroup_multiply(of[j], dg, of[j]);
      for (uint kk = 0; kk < 2; kk++) {
        simdgroup_float8x8 vf;
        simdgroup_load(vf, Vs + (kk * 8) * LDV + j * 8, LDV);
        simdgroup_multiply_accumulate(of[j], pf[kk], vf, of[j]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // normalize rows by 1/l and write bf16
  for (uint e = lane; e < 64; e += 32) Dg[sg][e] = 0;
  simdgroup_barrier(mem_flags::mem_threadgroup);
  if (lane % 4 == 0) Dg[sg][lrow * 8 + lrow] = 1.0f / l;
  simdgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_float8x8 dg;
  simdgroup_load(dg, Dg[sg], 8);
  threadgroup float* st = Ss[sg];  // 128 floats: one 8x8 frag at a time
  for (uint j = 0; j < 16; j++) {
    simdgroup_float8x8 o;
    simdgroup_multiply(o, dg, of[j]);
    simdgroup_store(o, st, 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = lane; e < 64; e += 32) {
      uint rI = e / 8, c = e % 8;
      uint t = q0 + row0 + rI;
      if (t < p.T) out[(t * p.H + h) * 128 + j * 8 + c] = f2bf(st[e]);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
}
