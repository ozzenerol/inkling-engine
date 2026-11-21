// Prefill attention (flash style, simdgroup matrices, softmax in registers). Owner: B.
// Threadgroup = (32-query block, kv head) with 12 simdgroups: simdgroup sg serves q head kvh*3 + sg/4 and query
// rows (sg%4)*8 .. +8 of the block. Key blocks of 32: K and V tiles bf16 in threadgroup memory shared by the 96
// rows. S = Q K^T (bf16 frags, fp32 acc). Bias, mask and the online softmax run on the fp32 frag elements each
// thread owns (8x8 frag layout as in MLX: thread holds row fm, cols fn, fn+1). P stays fp32; V is converted to
// fp32 frags at load. The sink is the initial running max with l = 1 (a zero value vector).

struct AttnPreParams {
  uint T, H, KVH;
  uint cap, len, ring, window, rel_extent;
};

constant uint AP_BQ = 32, AP_BK = 32, AP_SG = 12;

kernel void attn_prefill(device const ushort* q [[buffer(0)]], device const ushort* k_new [[buffer(1)]],
                         device const ushort* v_new [[buffer(2)]], device const ushort* r [[buffer(3)]],
                         device const ushort* rel_proj [[buffer(4)]], device const ushort* sink [[buffer(5)]],
                         device const ushort* kc [[buffer(6)]], device const ushort* vc [[buffer(7)]],
                         device ushort* out [[buffer(8)]], constant AttnPreParams& p [[buffer(9)]],
                         uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint LDK = 128 + 8;
  constexpr uint RPW = AP_BQ + AP_BK;  // rel_proj columns staged per key block
  constexpr uint LDV = 128 + 4;
  threadgroup bfloat Ks[AP_BK * LDK];
  threadgroup float Vs[AP_BK * LDV];  // fp32 V tile (simdgroup_load into fp32 frags); also Q staging at start
  threadgroup float4 Rp[RPW][4];  // [column][16 dims as 4 float4]
  threadgroup bfloat* Qst_base = (threadgroup bfloat*)Vs;  // AP_SG x 8 x 32 bf16 = 6 KB < 16.9 KB

  uint q0 = tg.x * AP_BQ, kvh = tg.y;
  uint G = p.H / p.KVH;
  uint h = kvh * G + sg / 4;
  uint row0 = (sg % 4) * 8;
  short2 fc = frag_coord(lane);
  uint fm = uint(fc.y), fn = uint(fc.x);
  uint t_me = q0 + row0 + fm;  // query of this thread's frag row
  bool row_ok = t_me < p.T;
  uint t_c = min(t_me, p.T - 1);
  uint qpos = p.len + t_c;
  float rr[16];
  for (uint d = 0; d < 16; d++) rr[d] = bf2f(r[(t_c * p.H + h) * 16 + d]);
  const float scale = 0.08838834764831845f;

  simdgroup_bfloat8x8 qf[16];
  for (uint dblk = 0; dblk < 128; dblk += 32) {
    for (uint e = lane; e < 8 * 32; e += 32) {
      uint rI = e / 32, c = e % 32;
      uint t = min(q0 + row0 + rI, p.T - 1);
      Qst_base[sg * 256 + rI * 32 + c] = as_type<bfloat>(q[(t * p.H + h) * 128 + dblk + c]);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint f = 0; f < 4; f++) simdgroup_load(qf[dblk / 8 + f], Qst_base + sg * 256 + f * 8, 32);
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);  // Q staging (in Vs) done before the first V tile load
  simdgroup_float8x8 of[16];
  for (uint j = 0; j < 16; j++) of[j] = simdgroup_float8x8(0);
  float m = row_ok ? bf2f(sink[t_c * p.H + h]) : 0.0f;
  float l = 1.0f;

  uint qpos_first = p.len + q0, qpos_last = p.len + min(q0 + AP_BQ, p.T) - 1;
  uint kfirst = 0;
  if (p.window && qpos_first + 1 > p.window) kfirst = qpos_first + 1 - p.window;
  kfirst = kfirst / AP_BK * AP_BK;
  for (uint kb = kfirst; kb <= qpos_last; kb += AP_BK) {
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
      *(threadgroup uint4*)(Ks + kr * LDK + c8) = *(device const uint4*)(kp + c8);
      float vf8[8];
      unpack_bf8(*(device const uint4*)(vp + c8), vf8);
      *(threadgroup float4*)(Vs + kr * LDV + c8) = float4(vf8[0], vf8[1], vf8[2], vf8[3]);
      *(threadgroup float4*)(Vs + kr * LDV + c8 + 4) = float4(vf8[4], vf8[5], vf8[6], vf8[7]);
    }
    int dlo = int(qpos_first) - int(kb + AP_BK - 1);  // smallest distance in this block
    bool need_bias = dlo < int(p.rel_extent);
    if (need_bias)
      for (uint i = tid; i < 16 * RPW; i += 32 * AP_SG) {
        uint d = i / RPW, c = i % RPW;
        int dist = dlo + int(c);
        ((threadgroup float*)&Rp[c][0])[d] =
            (dist >= 0 && dist < int(p.rel_extent)) ? bf2f(rel_proj[d * p.rel_extent + uint(dist)]) : 0.0f;
      }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 sf[4];
    for (uint j = 0; j < 4; j++) sf[j] = simdgroup_float8x8(0);
    for (uint d = 0; d < 16; d++)
      for (uint j = 0; j < 4; j++) {
        simdgroup_bfloat8x8 kf;
        simdgroup_load(kf, Ks + (j * 8) * LDK + d * 8, LDK, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(sf[j], qf[d], kf, sf[j]);
      }
    // bias + mask on the 8 elements this thread owns (row fm, keys j*8 + fn + {0,1})
    float cmax = -INFINITY;
    for (uint j = 0; j < 4; j++) {
      thread auto& el = sf[j].thread_elements();
      for (uint e = 0; e < 2; e++) {
        uint kpos = kb + j * 8 + fn + e;
        bool vis = row_ok && kpos <= qpos && (p.window == 0 || qpos - kpos < p.window);
        float sc = -INFINITY;
        if (vis) {
          sc = el[e] * scale;
          uint dist = qpos - kpos;
          if (dist < p.rel_extent) {
            uint col = uint(int(dist) - dlo);
            float b = 0;
            for (uint d4 = 0; d4 < 4; d4++) {
              float4 rp = Rp[col][d4];
              b = fma(rr[d4 * 4 + 0], rp.x, b);
              b = fma(rr[d4 * 4 + 1], rp.y, b);
              b = fma(rr[d4 * 4 + 2], rp.z, b);
              b = fma(rr[d4 * 4 + 3], rp.w, b);
            }
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
    float corr = exp(m - mn);
    float psum = 0;
    for (uint j = 0; j < 4; j++) {
      thread auto& el = sf[j].thread_elements();
      for (uint e = 0; e < 2; e++) {
        float pv = el[e] == -INFINITY ? 0.0f : exp(el[e] - mn);
        el[e] = pv;
        psum += pv;
      }
    }
    psum += simd_shuffle_xor(psum, 1);
    psum += simd_shuffle_xor(psum, 8);
    l = l * corr + psum;
    m = mn;
    for (uint j = 0; j < 16; j++) {
      thread auto& oe = of[j].thread_elements();
      oe[0] *= corr;
      oe[1] *= corr;
      for (uint kk = 0; kk < 4; kk++) {
        // V frag (keys kk*8.., dims j*8..) as fp32: this thread's elements are row fm (key), cols fn, fn+1 (dims)
        simdgroup_float8x8 vf;
        simdgroup_load(vf, Vs + (kk * 8) * LDV + j * 8, LDV);
        simdgroup_multiply_accumulate(of[j], sf[kk], vf, of[j]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float inv = 1.0f / l;
  if (row_ok)
    for (uint j = 0; j < 16; j++) {
      thread auto& oe = of[j].thread_elements();
      device ushort* o = out + (t_me * p.H + h) * 128 + j * 8 + fn;
      o[0] = f2bf(oe[0] * inv);
      o[1] = f2bf(oe[1] * inv);
    }
}
