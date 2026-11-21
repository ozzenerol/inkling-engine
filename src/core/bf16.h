// Owner: A. bf16 / fp8 / fp4 scalar conversions (host).
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>

namespace ink {

inline float bf16_to_f32(uint16_t h) {
  uint32_t u = uint32_t(h) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
inline uint16_t f32_to_bf16(float f) {  // round to nearest even, NaN preserved
  uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) return uint16_t((u >> 16) | 0x40);
  u += 0x7fffu + ((u >> 16) & 1u);
  return uint16_t(u >> 16);
}
inline float round_bf16(float f) { return bf16_to_f32(f32_to_bf16(f)); }

inline float e4m3_to_f32(uint8_t b) {
  int s = (b >> 7) & 1, e = (b >> 3) & 15, m = b & 7;
  float v;
  if (e == 15 && m == 7) v = NAN;
  else if (e == 0) v = std::ldexp(float(m) / 8.0f, -6);
  else v = std::ldexp(1.0f + float(m) / 8.0f, e - 7);
  return s ? -v : v;
}
inline float e8m0_to_f32(uint8_t b) { return b == 255 ? NAN : std::ldexp(1.0f, int(b) - 127); }
inline float e2m1_to_f32(uint8_t nib) {
  static const float lut[16] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6, -0.0f, -0.5f, -1, -1.5f, -2, -3, -4, -6};
  return lut[nib & 15];
}

}  // namespace ink
