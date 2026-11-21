// DSpark drafter kernels (Inkling 2 bundled mtp.safetensors). Owner: B.
// Reference: mlx_vlm/speculative/drafters/inkling2_dspark/dspark.py. Drafts only change speed (the target
// verifies every token), so these follow the reference math without a CPU oracle.

// mixed[r] = sum_j f32(feat[r, j]) * alpha[j] (j = 0..3 in order), out = bf16(rms(mixed) * gamma).
// feat: bf16 [L, 4, D]. One threadgroup of 256 per row.
struct DMixParams {
  uint L, D;
  float alpha[4];
  float eps;
  uint row_stride;  // elements between rows of feat (4*D)
};
kernel void dspark_mix(device const ushort* feat [[buffer(0)]], device const ushort* gamma [[buffer(1)]],
                       device ushort* out [[buffer(2)]], constant DMixParams& p [[buffer(3)]],
                       uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[RT / 32];
  threadgroup float mixed[2304];
  device const ushort* fr = feat + size_t(row) * p.row_stride;
  float ss = 0;
  for (uint i = tid; i < p.D; i += RT) {
    float m = 0;
    for (uint j = 0; j < 4; j++) m = m + bf2f(fr[j * p.D + i]) * p.alpha[j];
    mixed[i] = m;
    ss += m * m;
  }
  ss = tg_sum(ss, scratch, tid, lane, sg);
  float inv = metal::precise::rsqrt(ss / float(p.D) + p.eps);
  for (uint i = tid; i < p.D; i += RT) out[size_t(row) * p.D + i] = f2bf(mixed[i] * inv * bf2f(gamma[i]));
}

// RoPE (non-traditional: pairs (i, i + 64) of each 128-dim head), positions pos0 + t, in place on bf16.
struct RopeParams {
  uint T, H, pos0;
  float log2_base;  // log2(theta)
};
kernel void dspark_rope(device ushort* x [[buffer(0)]], constant RopeParams& p [[buffer(1)]],
                        uint3 gid [[thread_position_in_grid]]) {
  uint i = gid.x, h = gid.y, t = gid.z;
  if (i >= 64 || h >= p.H || t >= p.T) return;
  float inv_freq = exp2(-float(2 * i) / 128.0f * p.log2_base);
  float ang = float(p.pos0 + t) * inv_freq;
  float c = precise::cos(ang), s = precise::sin(ang);
  device ushort* r = x + (size_t(t) * p.H + h) * 128;
  float a = bf2f(r[i]), b = bf2f(r[i + 64]);
  r[i] = f2bf(a * c - b * s);
  r[i + 64] = f2bf(a * s + b * c);
}

// a = bf16(silu(f32 g) * f32 u)
kernel void dspark_silu_mul(device const ushort* g [[buffer(0)]], device const ushort* u [[buffer(1)]],
                            device ushort* a [[buffer(2)]], constant uint& n [[buffer(3)]],
                            uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  float gv = bf2f(g[i]);
  a[i] = f2bf(gv / (1.0f + exp(-gv)) * bf2f(u[i]));
}

// Markov RNN step (one threadgroup of 256). state, latent: bf16 [256]; ctx: f32 [3*256] (this step's context
// projection without bias; bias bf16 [768] is added here, as (x W^T + b) in the reference); ws: bf16 [768, 512] (rnn_proj columns 0..511); w1: bf16 [V, 256] embedding;
// prev token from tokens[prev_index].
struct MarkovParams {
  uint rank;
  uint prev_index;
};
kernel void dspark_markov_step(device ushort* state [[buffer(0)]], device ushort* latent [[buffer(1)]],
                               device const float* ctx [[buffer(2)]], device const ushort* ws [[buffer(3)]],
                               device const ushort* w1 [[buffer(4)]], device const int* tokens [[buffer(5)]],
                               constant MarkovParams& p [[buffer(6)]], device const ushort* bias [[buffer(7)]],
                               uint tid [[thread_position_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float in[512];
  threadgroup float raw[768];
  uint R = p.rank;
  int prev = tokens[p.prev_index];
  for (uint i = tid; i < R; i += 256) {
    in[i] = bf2f(state[i]);
    in[R + i] = bf2f(w1[size_t(prev) * R + i]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint row = sg; row < 3 * R; row += 8) {
    float a = 0;
    for (uint k = lane; k < 2 * R; k += 32) a = fma(in[k], bf2f(ws[row * 2 * R + k]), a);
    a = simd_sum(a);
    if (lane == 0) raw[row] = round_bf(a + (ctx[row] + bf2f(bias[row])));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < R; i += 256) {
    float gate = round_bf(1.0f / (1.0f + exp(-raw[i])));
    float cand = round_bf(precise::tanh(raw[R + i]));
    float s = in[i];
    float gs = round_bf(gate * s);
    float om = round_bf(1.0f - gate);
    float oc = round_bf(om * cand);
    state[i] = f2bf(gs + oc);
    latent[i] = f2bf(precise::tanh(raw[2 * R + i]));
  }
}

// Draft logits for one step and a partial argmax: for vocab rows v (< V), l = bf16(fold_b[v] + latent . fold_w[v])
// (fp32 dot), z = bf16(base[v] + l), z = bf16(cap * tanh(z / cap)); per-threadgroup best (value, index).
// One simdgroup per 4 rows; 8 simdgroups per threadgroup -> 32 rows per threadgroup.
struct FoldParams {
  uint V, R;
  float cap;
};
kernel void dspark_fold_argmax(device const ushort* latent [[buffer(0)]], device const ushort* fw [[buffer(1)]],
                               device const ushort* fb [[buffer(2)]], device const ushort* base [[buffer(3)]],
                               device float2* part [[buffer(4)]], constant FoldParams& p [[buffer(5)]],
                               uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float bv[8];
  threadgroup uint bi[8];
  float lat[8];
  for (uint j = 0; j < 8; j++) lat[j] = bf2f(latent[lane * 8 + j]);  // R = 256 = 32 lanes x 8
  float best = -INFINITY;
  uint besti = 0xffffffffu;
  for (uint r = 0; r < 4; r++) {
    uint v = (tg * 8 + sg) * 4 + r;
    if (v >= p.V) break;
    float w[8];
    unpack_bf8(*(device const uint4*)(fw + size_t(v) * p.R + lane * 8), w);
    float a = 0;
    for (uint j = 0; j < 8; j++) a = fma(lat[j], w[j], a);
    a = simd_sum(a);
    float l = round_bf(a + bf2f(fb[v]));
    float z = round_bf(bf2f(base[v]) + l);
    z = round_bf(p.cap * precise::tanh(z / p.cap));
    if (z > best) {
      best = z;
      besti = v;
    }
  }
  if (lane == 0) {
    bv[sg] = best;
    bi[sg] = besti;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0 && lane == 0) {
    float b = -INFINITY;
    uint idx = 0xffffffffu;
    for (uint i = 0; i < 8; i++)
      if (bv[i] > b || (bv[i] == b && bi[i] < idx)) {
        b = bv[i];
        idx = bi[i];
      }
    part[tg] = float2(b, as_type<float>(idx));
  }
}
// Final argmax over the partials (one threadgroup of 1024); writes tokens[out_index].
struct ArgmaxParams {
  uint n, out_index;
};
kernel void dspark_argmax_final(device const float2* part [[buffer(0)]], device int* tokens [[buffer(1)]],
                                constant ArgmaxParams& p [[buffer(2)]], uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float bv[1024];
  threadgroup uint bi[1024];
  float b = -INFINITY;
  uint idx = 0xffffffffu;
  for (uint i = tid; i < p.n; i += 1024) {
    float v = part[i].x;
    uint j = as_type<uint>(part[i].y);
    if (v > b || (v == b && j < idx)) {
      b = v;
      idx = j;
    }
  }
  bv[tid] = b;
  bi[tid] = idx;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = 512; s > 0; s >>= 1) {
    if (tid < s) {
      float v = bv[tid + s];
      uint j = bi[tid + s];
      if (v > bv[tid] || (v == bv[tid] && j < bi[tid])) {
        bv[tid] = v;
        bi[tid] = j;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) tokens[p.out_index] = int(bi[0]);
}

// Write row 0 = src (bf16 [D]) and rows 1..n-1 = mask (bf16 [D]) into x [n, D].
kernel void dspark_block_init(device const ushort* src [[buffer(0)]], device const ushort* mask [[buffer(1)]],
                              device ushort* x [[buffer(2)]], constant uint2& nd [[buffer(3)]],
                              uint2 gid [[thread_position_in_grid]]) {
  uint i = gid.x, r = gid.y;
  if (i >= nd.y || r >= nd.x) return;
  x[r * nd.y + i] = r == 0 ? src[i] : mask[i];
}

// Capture scatter: dst[t, slot, :] = bf16(src[t, :]) for t < T (src bf16 or f32).
struct CapParams {
  uint T, D, slot, dt_src;
};
kernel void dspark_capture(device const void* src [[buffer(0)]], device ushort* dst [[buffer(1)]],
                           constant CapParams& p [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  uint i = gid.x, t = gid.y;
  if (i >= p.D || t >= p.T) return;
  float v = p.dt_src == 0 ? ((device const float*)src)[t * p.D + i] : bf2f(((device const ushort*)src)[t * p.D + i]);
  dst[(size_t(t) * 4 + p.slot) * p.D + i] = f2bf(v);
}
