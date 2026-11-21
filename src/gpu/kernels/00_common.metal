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
