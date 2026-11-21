// Shared helpers for all kernels (prepended: file-name order). Owner: B.
#include <metal_stdlib>
using namespace metal;

// bf16 <-> f32 without relying on bfloat support quirks: bf16 is the top half of an f32.
inline float bf2f(ushort b) { return as_type<float>(uint(b) << 16); }
inline ushort f2bf(float f) {  // round to nearest even, like MLX/PyTorch
  uint u = as_type<uint>(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return ushort((u >> 16) | 0x40);  // NaN
  u += 0x7fffu + ((u >> 16) & 1u);
  return ushort(u >> 16);
}
inline float round_bf(float f) { return bf2f(f2bf(f)); }

// 8 bf16 values packed in a uint4 -> floats
inline void unpack_bf8(uint4 w, thread float* o) {
  o[0] = as_type<float>(w.x << 16); o[1] = as_type<float>(w.x & 0xffff0000u);
  o[2] = as_type<float>(w.y << 16); o[3] = as_type<float>(w.y & 0xffff0000u);
  o[4] = as_type<float>(w.z << 16); o[5] = as_type<float>(w.z & 0xffff0000u);
  o[6] = as_type<float>(w.w << 16); o[7] = as_type<float>(w.w & 0xffff0000u);
}

// FP8 decoders (MLX semantics: e4m3fn, e8m0 with 0 -> 2^-127)
inline float e4m3_to_f(uint b) {  // fn variant (no inf), same as MLX fp8_e4m3 -> half path
  uint v = b & 127u;
  float r;
  if (v == 127u) r = NAN;
  else if ((v >> 3) == 0) r = float(v & 7u) * 0.001953125f;  // subnormal: m/8 * 2^-6
  else r = as_type<float>(((v >> 3) + 120u) << 23 | (v & 7u) << 20);
  return (b & 128u) ? -r : r;
}
inline float e8m0_to_f(uint b) {
  if (b == 0) return as_type<float>(0x400000u);
  if (b == 255) return NAN;
  return as_type<float>(b << 23);
}


// 8x8 simdgroup matrix element layout (same as MLX BaseMMAFrag): thread holds row fm, cols fn and fn+1.
inline short2 frag_coord(uint lane) {  // (col, row) of the first of this thread's two elements
  short qid = short(lane / 4);
  short fm = (qid & 4) + short((lane / 2) % 4);
  short fn = (qid & 2) * 2 + short(lane % 2) * 2;
  return short2(fn, fm);
}


// ---------------------------------------------------------------- reductions helper (threadgroup of 256)
constant uint RT = 256;
inline float tg_sum(float v, threadgroup float* scratch, uint tid, uint lane, uint sg) {
  v = simd_sum(v);
  if (lane == 0) scratch[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float r = 0;
  for (uint i = 0; i < RT / 32; i++) r += scratch[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return r;
}

