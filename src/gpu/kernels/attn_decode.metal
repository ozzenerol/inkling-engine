// Decode / short-verify attention, split over keys (flash-decoding) with simdgroup matrices. Owner: B.
// Threadgroup = (kv head, key split), 2 simdgroups. Rows = the 3 q heads of the GQA group x T queries (T <= 4),
// padded to NF frag rows of 8 (row rr = t*3 + head-in-group). Each simdgroup walks every other 16-key block of the
// split: K and V tiles are staged in its own threadgroup area with coalesced 16-byte loads, S = Q K^T and O += P V
// on 8x8 frags, softmax on the frag elements. Partials (m, l, o[128]) per row go to attn_decode_combine.

struct AttnDecParams {
  uint T, H, KVH;
  uint cap, len, ring, window, rel_extent;  // rel_extent 0 = no relative bias
  uint split_keys, nsplit;
  uint causal;  // 0: every query sees all keys in range (DSpark draft block)
  uint first;   // oldest key position (computed by the host)
  uint use_sink;
  uint use_gate;  // combine: out = bf16(bf16(attn) * sigmoid(gate))
};

constant uint AD_MAXR = 24;  // partial rows per (kvh, split)
constant uint AD_BK = 16;
constant uint AD_SG = 2;

template <uint NF>
kernel void attn_decode_split(device const ushort* q [[buffer(0)]], device const ushort* k_new [[buffer(1)]],
                              device const ushort* v_new [[buffer(2)]], device const ushort* r [[buffer(3)]],
                              device const ushort* rel_proj [[buffer(4)]], device const ushort* kc [[buffer(5)]],
                              device const ushort* vc [[buffer(6)]], device float* part [[buffer(7)]],
                              constant AttnDecParams& p [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint LDK = 128 + 8;
  threadgroup bfloat KVs[AD_SG * 2 * AD_BK * LDK];  // per simdgroup: K tile then V tile; reused as fp32 at the end
  threadgroup bfloat Qst[8 * NF * 128];
  uint kvh = tg.x, split = tg.y;
  uint G = p.H / p.KVH;
  uint R = G * p.T;
  for (uint i = tid; i < 8 * NF * 128; i += 32 * AD_SG) {
    uint rr = i / 128, d = i % 128;
    uint rc = min(rr, R - 1), t = rc / G, h = kvh * G + rc % G;
    Qst[i] = as_type<bfloat>(q[(t * p.H + h) * 128 + d]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_bfloat8x8 qf[NF][16];
  for (uint f = 0; f < NF; f++)
    for (uint d = 0; d < 16; d++) simdgroup_load(qf[f][d], Qst + (f * 8) * 128 + d * 8, 128);

  short2 fc = frag_coord(lane);
  uint fm = uint(fc.y), fn = uint(fc.x);
  // per frag row f: this thread's row rr = f*8 + fm
  float rrv[NF][16];
  uint qpos[NF];
  bool rok[NF];
  for (uint f = 0; f < NF; f++) {
    uint rr = f * 8 + fm;
    rok[f] = rr < R;
    uint rc = min(rr, R - 1), t = rc / G, h = kvh * G + rc % G;
    qpos[f] = p.len + t;
    for (uint d = 0; d < 16; d++) rrv[f][d] = p.rel_extent ? bf2f(r[(t * p.H + h) * 16 + d]) : 0.0f;
  }
  // Splits are aligned to absolute positions (multiples of split_keys) so a row's keys are partitioned the same
  // way whatever T and len are: a verify row and a plain decode at the same position sum in the same order.
  uint last = p.len + p.T - 1;
  uint kbase = (p.first / p.split_keys + split) * p.split_keys;
  uint k0 = max(kbase, p.first);
  uint k1 = min(kbase + p.split_keys, last + 1);
  const float scale = 0.08838834764831845f;
  float m[NF], l[NF];
  simdgroup_float8x8 of[NF][16];
  for (uint f = 0; f < NF; f++) {
    m[f] = -INFINITY;
    l[f] = 0;
    for (uint j = 0; j < 16; j++) of[f][j] = simdgroup_float8x8(0);
  }
  threadgroup bfloat* K_ = KVs + sg * 2 * AD_BK * LDK;
  threadgroup bfloat* V_ = K_ + AD_BK * LDK;
  for (uint kb = kbase + sg * AD_BK; kb < k1; kb += AD_BK * AD_SG) {
    for (uint i = lane; i < AD_BK * 16; i += 32) {
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
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint f = 0; f < NF; f++) {
      simdgroup_float8x8 sf[2] = {simdgroup_float8x8(0), simdgroup_float8x8(0)};
      for (uint d = 0; d < 16; d++)
        for (uint j = 0; j < 2; j++) {
          simdgroup_bfloat8x8 kf;
          simdgroup_load(kf, K_ + (j * 8) * LDK + d * 8, LDK, ulong2(0, 0), true);
          simdgroup_multiply_accumulate(sf[j], qf[f][d], kf, sf[j]);
        }
      float cmax = -INFINITY;
      for (uint j = 0; j < 2; j++) {
        thread auto& el = sf[j].thread_elements();
        for (uint e = 0; e < 2; e++) {
          uint kpos = kb + j * 8 + fn + e;
          bool vis = rok[f] && kpos >= k0 && kpos < k1 &&
                     (!p.causal || (kpos <= qpos[f] && (p.window == 0 || qpos[f] - kpos < p.window)));
          float sc = -INFINITY;
          if (vis) {
            sc = el[e] * scale;
            uint dist = qpos[f] - kpos;
            if (p.causal && dist < p.rel_extent) {
              float b = 0;
              for (uint d = 0; d < 16; d++) b = fma(rrv[f][d], bf2f(rel_proj[d * p.rel_extent + dist]), b);
              sc += round_bf(b);
            }
          }
          el[e] = sc;
          cmax = max(cmax, sc);
        }
      }
      cmax = max(cmax, simd_shuffle_xor(cmax, 1));
      cmax = max(cmax, simd_shuffle_xor(cmax, 8));
      float mn = max(m[f], cmax);
      float corr = mn == -INFINITY ? 1.0f : exp(m[f] - mn);
      float psum = 0;
      for (uint j = 0; j < 2; j++) {
        thread auto& el = sf[j].thread_elements();
        for (uint e = 0; e < 2; e++) {
          float pv = el[e] == -INFINITY ? 0.0f : exp(el[e] - mn);
          el[e] = pv;
          psum += pv;
        }
      }
      psum += simd_shuffle_xor(psum, 1);
      psum += simd_shuffle_xor(psum, 8);
      l[f] = l[f] * corr + psum;
      m[f] = mn;
      for (uint j = 0; j < 16; j++) {
        thread auto& oe = of[f][j].thread_elements();
        oe[0] *= corr;
        oe[1] *= corr;
        for (uint kk = 0; kk < 2; kk++) {
          simdgroup_float8x8 vf;
          thread auto& ve = vf.thread_elements();
          threadgroup const bfloat* vr = V_ + (kk * 8 + fm) * LDK + j * 8 + fn;
          ve[0] = float(vr[0]);
          ve[1] = float(vr[1]);
          simdgroup_multiply_accumulate(of[f][j], sf[kk], vf, of[f][j]);
        }
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
  // combine the 2 simdgroups: sg 1 writes its partial to threadgroup memory (reusing the K/V areas as float)
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup float* xo = (threadgroup float*)KVs;  // NF*8 rows x 128 floats (12 KB for NF=3 <= 17 KB)
  threadgroup float* xm = (threadgroup float*)&Qst[0];    // m, l per row
  if (sg == 1)
    for (uint f = 0; f < NF; f++) {
      if (lane % 4 == 0 || true) {
        // rows are shared by 4 lanes; any of them may write m, l (same values)
        xm[(f * 8 + fm) * 2] = m[f];
        xm[(f * 8 + fm) * 2 + 1] = l[f];
      }
      for (uint j = 0; j < 16; j++) {
        thread auto& oe = of[f][j].thread_elements();
        xo[(f * 8 + fm) * 128 + j * 8 + fn] = oe[0];
        xo[(f * 8 + fm) * 128 + j * 8 + fn + 1] = oe[1];
      }
    }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0)
    for (uint f = 0; f < NF; f++) {
      uint rr = f * 8 + fm;
      float m1 = xm[rr * 2], l1 = xm[rr * 2 + 1];
      float M = max(m[f], m1);
      float c0 = m[f] == -INFINITY ? 0.0f : exp(m[f] - M);
      float c1 = m1 == -INFINITY ? 0.0f : exp(m1 - M);
      device float* dst = part + (size_t(kvh * p.nsplit + split) * AD_MAXR + rr) * 132;
      if (rr < R) {
        if (lane % 4 == 0 || true) {
          dst[0] = M;
          dst[1] = l[f] * c0 + l1 * c1;
        }
        for (uint j = 0; j < 16; j++) {
          thread auto& oe = of[f][j].thread_elements();
          for (uint e = 0; e < 2; e++) {
            uint d = j * 8 + fn + e;
            dst[4 + d] = oe[e] * c0 + xo[rr * 128 + d] * c1;
          }
        }
      }
    }
}

template [[host_name("attn_decode_split_f1")]] kernel void attn_decode_split<1>(
    device const ushort*, device const ushort*, device const ushort*, device const ushort*, device const ushort*,
    device const ushort*, device const ushort*, device float*, constant AttnDecParams&, uint2, uint, uint, uint);
template [[host_name("attn_decode_split_f3")]] kernel void attn_decode_split<3>(
    device const ushort*, device const ushort*, device const ushort*, device const ushort*, device const ushort*,
    device const ushort*, device const ushort*, device float*, constant AttnDecParams&, uint2, uint, uint, uint);
template [[host_name("attn_decode_split_f2")]] kernel void attn_decode_split<2>(
    device const ushort*, device const ushort*, device const ushort*, device const ushort*, device const ushort*,
    device const ushort*, device const ushort*, device float*, constant AttnDecParams&, uint2, uint, uint, uint);

// One threadgroup (128 threads = dims) per (t, h): reduce splits + sink, write bf16.
kernel void attn_decode_combine(device const float* part [[buffer(0)]], device const ushort* sink [[buffer(1)]],
                                device ushort* out [[buffer(2)]], constant AttnDecParams& p [[buffer(3)]],
                                device const ushort* gate [[buffer(4)]],
                                uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  uint t = tg.x, h = tg.y;
  uint G = p.H / p.KVH, kvh = h / G, rr = t * G + h % G;
  float sv = p.use_sink ? bf2f(sink[t * p.H + h]) : -INFINITY;
  float M = sv;
  for (uint s = 0; s < p.nsplit; s++) M = max(M, part[(size_t(kvh * p.nsplit + s) * AD_MAXR + rr) * 132]);
  float L = p.use_sink ? exp(sv - M) : 0.0f, O = 0;
  for (uint s = 0; s < p.nsplit; s++) {
    device const float* src = part + (size_t(kvh * p.nsplit + s) * AD_MAXR + rr) * 132;
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
