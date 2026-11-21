// Global attention for decode / verify at long context (T = 1..16), two passes. Owner: D.
// The split kernels (attn_decode_split, mr_attn_split) copy every K / V tile into threadgroup memory and run 8x8
// matrix products on it: 137-180 GB/s of KV at 128k. Here K and V are read straight from device memory:
//   ga_scores  S[row, key] = scale * (q . k) + relative bias, -inf where the key is not visible; also the max of
//              every 64-key block per row. One 16-byte load per lane and step, no staging: a matvec over K.
//   ga_rowmax  M[row] = max(sink, all block maxima)
//   ga_pv      per 512-key split: l = sum p, o = sum p v with p = exp(S - M); p is exactly 0 below exp(-thr), and
//              a 64-key block whose maximum is below that for every row of the group is not read at all (adding
//              p = 0 terms is an exact no-op, so skipping does not change any bit).
//   ga_combine out = bf16(O / L) over the splits in order, sink included, optional gate
// Per (row, key) arithmetic is fixed and independent of T and of the row group: chunk c (8 dims) of a key is one
// lane's 8-term fma chain, the 16 chunk sums are added in an xor tree (1, 2, 4, 8); a row's keys are spread over
// 8 streams by position (simdgroup = (key % 64) / 16, hf = key % 2), each stream accumulates in key order, and the
// streams are added as (half 0 + half 1) per simdgroup, then simdgroups 0..3 in order, then splits in order.
// So a verify row equals the decode row at the same position bit for bit.
struct GaParams {
  uint T, H, KVH;
  uint cap, len, rel_extent;
  uint sr;       // rows per kv head in S / Bmax / M / part (= 3 T)
  uint sk;       // keys per row in S (>= len + T, multiple of 64)
  uint nblk;     // 64-key blocks = ceil((len + T) / 64); Bmax stride
  uint nsplit;   // 512-key splits
  float thr;     // p = 0 where S - M <= -thr
  uint use_sink, use_gate;
  uint causal;   // 0: every query row sees all len + T keys (DSpark draft block)
};
constant float GA_SCALE = 0.08838834764831845f;

// One simdgroup = the 3 query rows (GQA group) of ONE token over a 64-key block; a threadgroup holds 1, 2 or 4
// simdgroups = tokens t0 .. t0+3 on the same keys (the K chunks are loaded by each of them; q stays in registers,
// 24 values). Grid: (nblk, KVH, ceil(T / tok)), 32 * tok threads.
kernel void ga_scores(device const ushort* q [[buffer(0)]], device const ushort* k_new [[buffer(1)]],
                      device const ushort* r [[buffer(2)]], device const ushort* rel_proj [[buffer(3)]],
                      device const ushort* kc [[buffer(4)]], device float* S [[buffer(5)]],
                      device float* Bmax [[buffer(6)]], constant GaParams& p [[buffer(7)]],
                      constant uint& tok [[buffer(8)]], uint3 tg [[threadgroup_position_in_grid]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint R = 3;
  const uint blk = tg.x, kvh = tg.y;
  const uint t = tg.z * tok + sg;
  if (t >= p.T) return;
  const uint c = lane % 16, hf = lane / 16;
  const uint nkeys = p.len + p.T, qpos = p.len + t;
  const size_t rowbase = size_t(kvh) * p.sr + t * R;
  float qv[R][8], bmx[R];
  for (uint rr = 0; rr < R; rr++) {
    unpack_bf8(*(device const packed_uint4*)(q + (t * p.H + kvh * R + rr) * 128 + c * 8), qv[rr]);
    bmx[rr] = -INFINITY;
  }
  for (uint it = 0; it < 32; it++) {
    const uint key = blk * 64 + it * 2 + hf;
    const uint kcl = min(key, nkeys - 1);
    device const ushort* kp = kcl < p.len ? kc + (size_t(kvh) * p.cap + kcl) * 128
                                          : k_new + (size_t(kcl - p.len) * p.KVH + kvh) * 128;
    float kv[8];
    unpack_bf8(*(device const packed_uint4*)(kp + c * 8), kv);
    const bool vis = key < nkeys && (!p.causal || key <= qpos);
    const uint dist = qpos - key;  // used only when causal (rel_extent = 0 otherwise)
    for (uint rr = 0; rr < R; rr++) {
      float a = 0;
      for (uint i = 0; i < 8; i++) a = fma(qv[rr][i], kv[i], a);
      a += simd_shuffle_xor(a, 1);
      a += simd_shuffle_xor(a, 2);
      a += simd_shuffle_xor(a, 4);
      a += simd_shuffle_xor(a, 8);
      float sc = -INFINITY;
      if (vis) {
        sc = a * GA_SCALE;
        if (p.causal && dist < p.rel_extent) {
          float b = 0;
          for (uint d = 0; d < 16; d++)
            b = fma(bf2f(r[(t * p.H + kvh * R + rr) * 16 + d]), bf2f(rel_proj[d * p.rel_extent + dist]), b);
          sc += round_bf(b);
        }
      }
      if (c == 0) S[(rowbase + rr) * p.sk + key] = sc;
      bmx[rr] = max(bmx[rr], sc);
    }
  }
  for (uint rr = 0; rr < R; rr++) {
    float m = max(bmx[rr], simd_shuffle_xor(bmx[rr], 16));
    if (lane == rr) Bmax[(rowbase + rr) * p.nblk + blk] = m;
  }
}

// Grid: (sr rows, KVH), 32 threads.
kernel void ga_rowmax(device const float* Bmax [[buffer(0)]], device const ushort* sink [[buffer(1)]],
                      device float* M [[buffer(2)]], constant GaParams& p [[buffer(3)]],
                      uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint row = tg.x, kvh = tg.y, G = p.H / p.KVH;
  device const float* b = Bmax + (size_t(kvh) * p.sr + row) * p.nblk;
  float m = -INFINITY;
  for (uint i = lane; i < p.nblk; i += 32) m = max(m, b[i]);
  m = max(m, simd_shuffle_xor(m, 1));
  m = max(m, simd_shuffle_xor(m, 2));
  m = max(m, simd_shuffle_xor(m, 4));
  m = max(m, simd_shuffle_xor(m, 8));
  m = max(m, simd_shuffle_xor(m, 16));
  if (p.use_sink) m = max(m, bf2f(sink[(row / G) * p.H + kvh * G + row % G]));
  if (lane == 0) M[size_t(kvh) * p.sr + row] = m;
}

// One simdgroup = one token (3 rows) over a 512-key split; 1, 2 or 4 tokens per threadgroup as in ga_scores.
// Grid: (nsplit, KVH, ceil(T / tok)), 32 * tok threads. part[(kvh * nsplit + split) * sr + row] = {l, -, -, -, o[128]}.
kernel void ga_pv(device const float* S [[buffer(0)]], device const float* Bmax [[buffer(1)]],
                  device const float* M [[buffer(2)]], device const ushort* v_new [[buffer(3)]],
                  device const ushort* vc [[buffer(4)]], device float* part [[buffer(5)]],
                  constant GaParams& p [[buffer(6)]], constant uint& tok [[buffer(7)]],
                  uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                  uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint R = 3;
  const uint split = tg.x, kvh = tg.y;
  const uint t = tg.z * tok + sg;
  if (t >= p.T) return;
  const uint c = lane % 16, hf = lane / 16;
  const uint nkeys = p.len + p.T;
  const size_t rowbase = size_t(kvh) * p.sr + t * R;
  float acc[R][8], l[R], mr[R];
  for (uint rr = 0; rr < R; rr++) {
    mr[rr] = M[rowbase + rr];
    l[rr] = 0;
    for (uint i = 0; i < 8; i++) acc[rr][i] = 0;
  }
  for (uint b = 0; b < 8; b++) {
    const uint blk = split * 8 + b;
    if (blk >= p.nblk) break;
    bool live = false;
    for (uint rr = 0; rr < R; rr++)
      if (Bmax[(rowbase + rr) * p.nblk + blk] - mr[rr] > -p.thr) live = true;
    if (!live) continue;
    for (uint it = 0; it < 32; it++) {
      const uint key = blk * 64 + it * 2 + hf;
      const uint kcl = min(key, nkeys - 1);
      device const ushort* vp = kcl < p.len ? vc + (size_t(kvh) * p.cap + kcl) * 128
                                            : v_new + (size_t(kcl - p.len) * p.KVH + kvh) * 128;
      float vv[8];
      unpack_bf8(*(device const packed_uint4*)(vp + c * 8), vv);
      for (uint rr = 0; rr < R; rr++) {
        const float d = S[(rowbase + rr) * p.sk + key] - mr[rr];
        const float pr = d > -p.thr ? exp(d) : 0.0f;
        l[rr] += pr;
        for (uint i = 0; i < 8; i++) acc[rr][i] = fma(pr, vv[i], acc[rr][i]);
      }
    }
  }
  // the two streams of the split: even keys + odd keys
  for (uint rr = 0; rr < R; rr++) {
    l[rr] += simd_shuffle_xor(l[rr], 16);
    for (uint i = 0; i < 8; i++) acc[rr][i] += simd_shuffle_xor(acc[rr][i], 16);
    if (hf == 0) {
      device float* dst = part + (size_t(kvh * p.nsplit + split) * p.sr + t * R + rr) * 132;
      if (c == 0) dst[0] = l[rr];
      for (uint i = 0; i < 8; i++) dst[4 + c * 8 + i] = acc[rr][i];
    }
  }
}

// Grid: (T, H), 128 threads (= dims).
kernel void ga_combine(device const float* part [[buffer(0)]], device const float* M [[buffer(1)]],
                       device const ushort* sink [[buffer(2)]], device ushort* out [[buffer(3)]],
                       device const ushort* gate [[buffer(4)]], constant GaParams& p [[buffer(5)]],
                       uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint t = tg.x, h = tg.y;
  const uint G = p.H / p.KVH, kvh = h / G, row = t * G + h % G;
  const float m = M[size_t(kvh) * p.sr + row];
  float L = p.use_sink ? exp(bf2f(sink[t * p.H + h]) - m) : 0.0f, O = 0;
  for (uint s = 0; s < p.nsplit; s++) {
    device const float* src = part + (size_t(kvh * p.nsplit + s) * p.sr + row) * 132;
    L += src[0];
    O += src[4 + tid];
  }
  const uint oi = (t * p.H + h) * 128 + tid;
  float o = round_bf(O / L);
  if (p.use_gate) o = o * (1.0f / (1.0f + exp(-bf2f(gate[oi]))));
  out[oi] = f2bf(o);
}

