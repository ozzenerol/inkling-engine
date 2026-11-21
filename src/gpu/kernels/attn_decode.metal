// Decode / short-verify attention, split over keys (flash-decoding). Owner: B.
// Threadgroup = (kv head, key split), 4 simdgroups. It serves the 3 q heads of the GQA group for all T (<= 4)
// queries: R = 3*T rows. Scores: one lane per key (lane reads its key row). PV: 4 dims per lane, p broadcast.
// Writes per-split partials (m, l, o[128]) that attn_decode_combine reduces together with the sink logit.

struct AttnDecParams {
  uint T, H, KVH;
  uint cap, len, ring, window, rel_extent;
  uint split_keys, nsplit;
};

constant uint AD_MAXR = 12;

inline uint key_slot(uint pos, constant AttnDecParams& p) { return p.ring ? pos % p.cap : pos; }

template <uint MR>
kernel void attn_decode_split(device const ushort* q [[buffer(0)]], device const ushort* k_new [[buffer(1)]],
                              device const ushort* v_new [[buffer(2)]], device const ushort* r [[buffer(3)]],
                              device const ushort* rel_proj [[buffer(4)]], device const ushort* kc [[buffer(5)]],
                              device const ushort* vc [[buffer(6)]], device float* part [[buffer(7)]],
                              constant AttnDecParams& p [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
  uint kvh = tg.x, split = tg.y;
  uint G = p.H / p.KVH;  // 3
  uint R = min(G * p.T, MR);
  threadgroup float qs[AD_MAXR][128];
  threadgroup float rs[AD_MAXR][16];
  for (uint i = tid; i < R * 128; i += 128) {
    uint rr = i / 128, d = i % 128, t = rr / G, h = kvh * G + rr % G;
    qs[rr][d] = bf2f(q[(t * p.H + h) * 128 + d]);
  }
  for (uint i = tid; i < R * 16; i += 128) {
    uint rr = i / 16, d = i % 16, t = rr / G, h = kvh * G + rr % G;
    rs[rr][d] = bf2f(r[(t * p.H + h) * 16 + d]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // key range of this split: positions [lo, hi) over all keys any query can see
  uint last = p.len + p.T - 1;  // newest key position
  uint first = 0;
  if (p.window && p.len + 1 > p.window) first = p.len + 1 - p.window;  // oldest key the first query sees
  uint k0 = first + split * p.split_keys;
  uint k1 = min(k0 + p.split_keys, last + 1);
  const float scale = 0.08838834764831845f;
  float m[MR], l[MR], o[MR][4];
  for (uint rr = 0; rr < MR; rr++) {
    m[rr] = -INFINITY;
    l[rr] = 0;
    for (uint j = 0; j < 4; j++) o[rr][j] = 0;
  }
  for (uint base = k0 + sg * 32; base < k1; base += 128) {
    uint pos = base + lane;
    bool kvalid = pos < k1;
    device const ushort* kp;
    if (pos < p.len) kp = kc + (size_t(kvh) * p.cap + key_slot(pos, p)) * 128;
    else kp = k_new + (size_t(min(pos, last) - p.len) * p.KVH + kvh) * 128;
    float s[MR];
    for (uint rr = 0; rr < R; rr++) s[rr] = 0;
    device const uint4* kp4 = (device const uint4*)kp;
    for (uint c = 0; c < 16; c++) {
      float kv[8];
      unpack_bf8(kp4[c], kv);
      for (uint rr = 0; rr < R; rr++)
        for (uint i = 0; i < 8; i++) s[rr] = fma(qs[rr][c * 8 + i], kv[i], s[rr]);
    }
    for (uint rr = 0; rr < R; rr++) {
      uint t = rr / G, qpos = p.len + t;
      bool vis = kvalid && pos <= qpos && (p.window == 0 || qpos - pos < p.window);
      float sc = -INFINITY;
      if (vis) {
        sc = s[rr] * scale;
        uint dist = qpos - pos;
        if (dist < p.rel_extent) {
          float b = 0;
          for (uint d = 0; d < 16; d++) b = fma(rs[rr][d], bf2f(rel_proj[d * p.rel_extent + dist]), b);
          sc += round_bf(b);
        }
      }
      s[rr] = sc;
    }
    // online softmax per row over this chunk of 32 keys
    float pr[MR];
    for (uint rr = 0; rr < R; rr++) {
      float cm = simd_max(s[rr]);
      float mn = max(m[rr], cm);
      float c = mn == -INFINITY ? 1.0f : exp(m[rr] - mn);
      pr[rr] = mn == -INFINITY ? 0.0f : exp(s[rr] - mn);
      l[rr] = l[rr] * c + simd_sum(pr[rr]);
      for (uint j = 0; j < 4; j++) o[rr][j] *= c;
      m[rr] = mn;
    }
    uint nk = min(32u, k1 - base);
    for (uint kk = 0; kk < nk; kk++) {
      uint vpos = base + kk;
      device const ushort* vp;
      if (vpos < p.len) vp = vc + (size_t(kvh) * p.cap + key_slot(vpos, p)) * 128;
      else vp = v_new + (size_t(vpos - p.len) * p.KVH + kvh) * 128;
      uint2 vv = *(device const uint2*)(vp + lane * 4);
      float v0 = as_type<float>(vv.x << 16), v1 = as_type<float>(vv.x & 0xffff0000u);
      float v2 = as_type<float>(vv.y << 16), v3 = as_type<float>(vv.y & 0xffff0000u);
      for (uint rr = 0; rr < R; rr++) {
        float pk = simd_shuffle(pr[rr], kk);
        o[rr][0] = fma(pk, v0, o[rr][0]);
        o[rr][1] = fma(pk, v1, o[rr][1]);
        o[rr][2] = fma(pk, v2, o[rr][2]);
        o[rr][3] = fma(pk, v3, o[rr][3]);
      }
    }
  }
  // combine the 4 simdgroups through threadgroup memory
  threadgroup float tm[4][AD_MAXR], tl[4][AD_MAXR];
  threadgroup float to[4][AD_MAXR][128];
  for (uint rr = 0; rr < R; rr++) {
    if (lane == 0) {
      tm[sg][rr] = m[rr];
      tl[sg][rr] = l[rr];
    }
    for (uint j = 0; j < 4; j++) to[sg][rr][lane * 4 + j] = o[rr][j];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // 128 threads: thread = dim, loop rows
  for (uint rr = 0; rr < R; rr++) {
    float M = -INFINITY;
    for (uint i = 0; i < 4; i++) M = max(M, tm[i][rr]);
    float L = 0, O = 0;
    for (uint i = 0; i < 4; i++) {
      if (tm[i][rr] == -INFINITY) continue;
      float c = exp(tm[i][rr] - M);
      L += tl[i][rr] * c;
      O += to[i][rr][tid] * c;
    }
    device float* dst = part + (size_t(kvh * p.nsplit + split) * AD_MAXR + rr) * 132;
    dst[4 + tid] = O;
    if (tid == 0) {
      dst[0] = M;
      dst[1] = L;
    }
  }
}

template [[host_name("attn_decode_split_r3")]] kernel void attn_decode_split<3>(
    device const ushort*, device const ushort*, device const ushort*, device const ushort*, device const ushort*,
    device const ushort*, device const ushort*, device float*, constant AttnDecParams&, uint2, uint, uint, uint);
template [[host_name("attn_decode_split_r12")]] kernel void attn_decode_split<12>(
    device const ushort*, device const ushort*, device const ushort*, device const ushort*, device const ushort*,
    device const ushort*, device const ushort*, device float*, constant AttnDecParams&, uint2, uint, uint, uint);

// One threadgroup (128 threads = dims) per (t, h): reduce splits + sink, write bf16.
kernel void attn_decode_combine(device const float* part [[buffer(0)]], device const ushort* sink [[buffer(1)]],
                                device ushort* out [[buffer(2)]], constant AttnDecParams& p [[buffer(3)]],
                                uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  uint t = tg.x, h = tg.y;
  uint G = p.H / p.KVH, kvh = h / G, rr = t * G + h % G;
  float sv = bf2f(sink[t * p.H + h]);
  float M = sv;
  for (uint s = 0; s < p.nsplit; s++) M = max(M, part[(size_t(kvh * p.nsplit + s) * AD_MAXR + rr) * 132]);
  float L = exp(sv - M), O = 0;
  for (uint s = 0; s < p.nsplit; s++) {
    device const float* src = part + (size_t(kvh * p.nsplit + s) * AD_MAXR + rr) * 132;
    if (src[0] == -INFINITY) continue;
    float c = exp(src[0] - M);
    L += src[1] * c;
    O += src[4 + tid] * c;
  }
  out[(t * p.H + h) * 128 + tid] = f2bf(O / L);
}
