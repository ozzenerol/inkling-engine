// Owner: A. CPU reference backend: plain, readable implementations of every Backend op, following the rounding
// points of the MLX reference. Used as the oracle for the Metal backend and for the reference test.
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "core/bf16.h"
#include "core/parallel.h"
#include "model/backend.h"

namespace ink {
namespace {

inline float ld(const Tensor& t, int64_t i) {
  switch (t.dtype) {
    case DType::F32: return t.data<float>()[i];
    case DType::BF16: return bf16_to_f32(t.data<uint16_t>()[i]);
    default: throw std::runtime_error("ld: unsupported dtype " + t.str());
  }
}
inline void st(const Tensor& t, int64_t i, float v) {
  switch (t.dtype) {
    case DType::F32: t.data<float>()[i] = v; break;
    case DType::BF16: t.data<uint16_t>()[i] = f32_to_bf16(v); break;
    default: throw std::runtime_error("st: unsupported dtype " + t.str());
  }
}
// Load n contiguous values starting at element i as f32.
inline void ld_row(const Tensor& t, int64_t i, int64_t n, float* dst) {
  if (t.dtype == DType::F32) {
    std::memcpy(dst, t.data<float>() + i, size_t(n) * 4);
  } else {
    const uint16_t* s = t.data<uint16_t>() + i;
    for (int64_t k = 0; k < n; ++k) dst[k] = bf16_to_f32(s[k]);
  }
}
inline void st_row(const Tensor& t, int64_t i, int64_t n, const float* src) {
  if (t.dtype == DType::F32) {
    std::memcpy(t.data<float>() + i, src, size_t(n) * 4);
  } else {
    uint16_t* d = t.data<uint16_t>() + i;
    for (int64_t k = 0; k < n; ++k) d[k] = f32_to_bf16(src[k]);
  }
}

inline float dot(const float* a, const float* b, int64_t n) {
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  int64_t k = 0;
  for (; k + 8 <= n; k += 8)
    for (int j = 0; j < 8; ++j) acc[j] += a[k + j] * b[k + j];
  float s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
  for (; k < n; ++k) s += a[k] * b[k];
  return s;
}
inline float dot_bf16(const float* a, const uint16_t* b, int64_t n) {
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  int64_t k = 0;
  for (; k + 8 <= n; k += 8)
    for (int j = 0; j < 8; ++j) acc[j] += a[k + j] * bf16_to_f32(b[k + j]);
  float s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
  for (; k < n; ++k) s += a[k] * bf16_to_f32(b[k]);
  return s;
}

// INK_CPU_ROWEXACT=1: never use BLAS, so every row's result is independent of the batch it is computed in
// (needed to check speculative rollback bit for bit; slower).
inline bool row_exact() {
  static const bool v = std::getenv("INK_CPU_ROWEXACT") != nullptr;
  return v;
}

inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// x [T,K] (any float dtype) -> f32 rows
std::vector<float> to_f32(const Tensor& x) {
  std::vector<float> v(static_cast<size_t>(x.numel()));
  ld_row(x, 0, x.numel(), v.data());
  return v;
}

// out[T,N] = X[T,K] @ W[N,K]^T with W bf16 (rows [n0, n0+N) of w), X f32. Writes f32 into y (row stride ldy).
void gemm_bf16w(const float* X, int64_t T, int64_t K, const uint16_t* W, int64_t N, float* y, int64_t ldy) {
  if (T < 4 || row_exact()) {
    parallel_for(N, [&](int64_t b, int64_t e) {
      for (int64_t n = b; n < e; ++n)
        for (int64_t t = 0; t < T; ++t) y[t * ldy + n] = dot_bf16(X + t * K, W + n * K, K);
    }, 64);
    return;
  }
  const int64_t CH = 4096;
  std::vector<float> wf(static_cast<size_t>(std::min(N, CH) * K));
  for (int64_t n0 = 0; n0 < N; n0 += CH) {
    int64_t nn = std::min(CH, N - n0);
    parallel_for(nn * K, [&](int64_t b, int64_t e) {
      for (int64_t i = b; i < e; ++i) wf[i] = bf16_to_f32(W[n0 * K + i]);
    }, 1 << 16);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, int(T), int(nn), int(K), 1.0f, X, int(K), wf.data(),
                int(K), 0.0f, y + n0, int(ldy));
  }
}

// Dequantize rows [0, n) of expert e of a QuantProj into dst [n, k] f32.
void dequant_expert(const QuantProj& p, int64_t e, float* dst) {
  const int64_t n = p.n, k = p.k;
  if (p.mode == QuantMode::NVFP4) {
    const uint8_t* w = static_cast<const uint8_t*>(p.weight.host) + p.weight.offset + size_t(e) * n * (k / 2);
    const uint8_t* s = static_cast<const uint8_t*>(p.scales.host) + p.scales.offset + size_t(e) * n * (k / 16);
    for (int64_t r = 0; r < n; ++r) {
      for (int64_t gidx = 0; gidx < k / 16; ++gidx) {
        float sc = e4m3_to_f32(s[r * (k / 16) + gidx]);  // global scale applied to the accumulator (as MLX)
        const uint8_t* wb = w + r * (k / 2) + gidx * 8;
        float* d = dst + r * k + gidx * 16;
        for (int j = 0; j < 8; ++j) {
          d[2 * j] = e2m1_to_f32(wb[j] & 15) * sc;
          d[2 * j + 1] = e2m1_to_f32(wb[j] >> 4) * sc;
        }
      }
    }
  } else {
    const uint8_t* w = static_cast<const uint8_t*>(p.weight.host) + p.weight.offset + size_t(e) * n * k;
    const uint8_t* s = static_cast<const uint8_t*>(p.scales.host) + p.scales.offset + size_t(e) * n * (k / 32);
    static float lut[256];
    static std::once_flag once;
    std::call_once(once, [] { for (int i = 0; i < 256; ++i) lut[i] = e4m3_to_f32(uint8_t(i)); });
    for (int64_t r = 0; r < n; ++r)
      for (int64_t gidx = 0; gidx < k / 32; ++gidx) {
        float sc = e8m0_to_f32(s[r * (k / 32) + gidx]);
        const uint8_t* wb = w + r * k + gidx * 32;
        float* d = dst + r * k + gidx * 32;
        for (int j = 0; j < 32; ++j) d[j] = lut[wb[j]] * sc;
      }
  }
}

inline float gscale(const QuantProj& p, int64_t e) {
  return p.mode == QuantMode::NVFP4 ? p.global.data<float>()[e] : 1.0f;
}

class CpuBackend final : public Backend {
 public:
  ~CpuBackend() override {
    for (auto& [f, m] : maps_) munmap(m.first, m.second);
  }
  const char* name() const override { return "cpu"; }

  Tensor alloc(DType dt, std::initializer_list<int64_t> shape, const char*) override {
    Tensor t;
    t.dtype = dt;
    t.set_shape(shape);
    size_t bytes = std::max<size_t>(t.nbytes(), 16);
    void* p = std::calloc(1, (bytes + 63) / 64 * 64);
    if (!p) throw std::runtime_error("cpu alloc failed");
    t.buf = t.host = p;
    return t;
  }
  void free(Tensor& t) override {
    if (t.host) std::free(t.host);
    t.buf = t.host = nullptr;
  }
  Tensor load_weight(const SafeTensors::Entry& e) override {
    auto it = maps_.find(e.file);
    if (it == maps_.end()) {
      int fd = open(e.file.c_str(), O_RDONLY);
      if (fd < 0) throw std::runtime_error("open " + e.file);
      struct stat sb;
      fstat(fd, &sb);
      void* p = mmap(nullptr, size_t(sb.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
      close(fd);
      if (p == MAP_FAILED) throw std::runtime_error("mmap " + e.file);
      it = maps_.emplace(e.file, std::make_pair(p, size_t(sb.st_size))).first;
    }
    Tensor t;
    t.dtype = e.dtype;
    t.ndim = e.ndim;
    for (int i = 0; i < 4; ++i) t.shape[i] = e.shape[i];
    t.buf = t.host = it->second.first;
    t.offset = e.offset;
    return t;
  }
  void sync() override {}
  void copy(const Tensor& src, Tensor& dst, size_t so, size_t dof, size_t bytes) override {
    if (!bytes) bytes = src.nbytes();
    std::memmove(static_cast<char*>(dst.host) + dst.offset + dof, static_cast<char*>(src.host) + src.offset + so,
                 bytes);
  }
  void read(const Tensor& t, void* dst, size_t bytes, size_t off) override {
    std::memcpy(dst, static_cast<char*>(t.host) + t.offset + off, bytes);
  }
  void write(Tensor& t, const void* src, size_t bytes, size_t off) override {
    std::memcpy(static_cast<char*>(t.host) + t.offset + off, src, bytes);
  }

  void embed(const Tensor& ids, const Tensor& table, Tensor& out) override {
    const int64_t T = ids.numel(), D = table.dim(1);
    for (int64_t t = 0; t < T; ++t) {
      int32_t id = ids.data<int32_t>()[t];
      std::memcpy(out.data<uint16_t>() + t * D, table.data<uint16_t>() + int64_t(id) * D, size_t(D) * 2);
    }
  }

  void rms_norm(const Tensor& x, const Tensor* gamma, float eps, int64_t dim, Tensor& out) override {
    const int64_t rows = x.numel() / dim;
    parallel_for(rows, [&](int64_t b, int64_t e) {
      std::vector<float> v(static_cast<size_t>(dim));
      for (int64_t r = b; r < e; ++r) {
        ld_row(x, r * dim, dim, v.data());
        float ss = 0;
        for (float f : v) ss += f * f;
        float inv = 1.0f / std::sqrt(ss / float(dim) + eps);
        for (int64_t k = 0; k < dim; ++k) {
          // MLX: out = T(w * T(x * inv)) with T = input dtype (two roundings for bf16 input).
          float y = v[k] * inv;
          if (x.dtype == DType::BF16) y = round_bf16(y);
          if (gamma && gamma->valid()) y *= ld(*gamma, k);
          v[k] = y;
        }
        st_row(out, r * dim, dim, v.data());
      }
    }, 8);
  }

  void linear(const Tensor& x, const Tensor& w, Tensor& out) override {
    const int64_t K = w.dim(1), N = w.dim(0), T = x.numel() / K;
    if (w.dtype != DType::BF16) throw std::runtime_error("linear: weight must be bf16");
    std::vector<float> X = to_f32(x);
    std::vector<float> Y(static_cast<size_t>(T * N));
    gemm_bf16w(X.data(), T, K, w.data<uint16_t>(), N, Y.data(), N);
    st_row(out, 0, T * N, Y.data());
  }

  void attn_residual(const std::vector<const Tensor*>& states, const Tensor& query, Tensor& out) override {
    const int64_t D = query.dim(-1), nq = query.numel() / D, nb = int64_t(states.size());
    const int64_t T = states[0]->numel() / D;
    parallel_for(T, [&](int64_t b, int64_t e) {
      std::vector<float> keys(static_cast<size_t>(nb * D)), sc(static_cast<size_t>(nb));
      for (int64_t t = b; t < e; ++t) {
        for (int64_t s = 0; s < nb; ++s) {
          float* kr = keys.data() + s * D;
          ld_row(*states[s], t * D, D, kr);
          float ss = 0;
          for (int64_t k = 0; k < D; ++k) ss += kr[k] * kr[k];
          float inv = 1.0f / std::sqrt(ss / float(D) + 1e-6f);
          for (int64_t k = 0; k < D; ++k) kr[k] *= inv;
        }
        for (int64_t q = 0; q < nq; ++q) {
          const float* qr = query.data<float>() + q * D;
          float m = -INFINITY;
          for (int64_t s = 0; s < nb; ++s) {
            sc[s] = dot(keys.data() + s * D, qr, D) * (1.0f / std::sqrt(float(D)));
            m = std::max(m, sc[s]);
          }
          float z = 0;
          for (int64_t s = 0; s < nb; ++s) z += (sc[s] = std::exp(sc[s] - m));
          float* o = out.data<float>() + (q * T + t) * D;
          for (int64_t k = 0; k < D; ++k) {
            float acc = 0;
            for (int64_t s = 0; s < nb; ++s) acc += sc[s] / z * keys[s * D + k];
            o[k] = acc;
          }
        }
      }
    });
  }

  void loop_messages(const std::vector<const Tensor*>& states, const Tensor& query, Tensor& out) override {
    attn_residual(states, query, out);  // identical math: softmax over rms-normed states, values = normed states
  }

  void controller(const Tensor& x, const Tensor& m, const Tensor& down, const Tensor& forget,
                  const Tensor& gain, Tensor& out) override {
    const int64_t D = x.dim(-1), T = x.numel() / D, R = down.dim(0);
    parallel_for(T, [&](int64_t b, int64_t e) {
      std::vector<float> pair(static_cast<size_t>(2 * D)), h(static_cast<size_t>(R));
      for (int64_t t = b; t < e; ++t) {
        const float* xr = x.data<float>() + t * D;
        const float* mr = m.data<float>() + t * D;
        for (int half = 0; half < 2; ++half) {
          const float* src = half ? mr : xr;
          float ss = 0;
          for (int64_t k = 0; k < D; ++k) ss += src[k] * src[k];
          float inv = 1.0f / std::sqrt(ss / float(D) + 1e-6f);
          for (int64_t k = 0; k < D; ++k) pair[half * D + k] = round_bf16(src[k] * inv);
        }
        for (int64_t r = 0; r < R; ++r) h[r] = round_bf16(dot_bf16(pair.data(), down.data<uint16_t>() + r * 2 * D, 2 * D));
        float* o = out.data<float>() + t * D;
        for (int64_t k = 0; k < D; ++k) {
          float f = std::tanh(dot_bf16(h.data(), forget.data<uint16_t>() + k * R, R));
          float g = std::tanh(dot_bf16(h.data(), gain.data<uint16_t>() + k * R, R));
          o[k] = (1.0f - f) * xr[k] + g * mr[k];
        }
      }
    });
  }

  void short_conv(const Tensor& x, const Tensor& w, Tensor& state, Tensor& out, Tensor* hist) override {
    const int64_t C = x.dim(-1), T = x.numel() / C;
    const float div = std::sqrt(5.0f);
    float* sp = state.data<float>();
    float* hp = hist && hist->valid() ? hist->data<float>() : nullptr;
    parallel_for(C, [&](int64_t b, int64_t e) {
      for (int64_t c = b; c < e; ++c) {
        float w4[4];
        for (int j = 0; j < 4; ++j) w4[j] = ld(w, c * 4 + j);
        float hist[3] = {sp[0 * C + c], sp[1 * C + c], sp[2 * C + c]};
        for (int64_t t = 0; t < T; ++t) {
          float xt = ld(x, t * C + c);
          float v = 0;
          v += hist[0] * w4[0];
          v += hist[1] * w4[1];
          v += hist[2] * w4[2];
          v += xt * w4[3];
          st(out, t * C + c, (v + xt) / div);
          hist[0] = hist[1];
          hist[1] = hist[2];
          hist[2] = xt;
          if (hp)
            for (int j = 0; j < 3; ++j) hp[(t * 3 + j) * C + c] = hist[j];
        }
        for (int j = 0; j < 3; ++j) sp[j * C + c] = hist[j];
      }
    }, 64);
  }

  void dyn_conv(const Tensor& x, const Tensor& dyn, const Tensor& sw, Tensor& state, Tensor& out, Tensor* hist) override {
    const int64_t C = x.dim(-1), T = x.numel() / C, HD = 128, K = 4;
    const float div = std::sqrt(9.0f);
    float* sp = state.data<float>();
    float* hp = hist && hist->valid() ? hist->data<float>() : nullptr;
    parallel_for(C, [&](int64_t b, int64_t e) {
      for (int64_t c = b; c < e; ++c) {
        const int64_t h = c / HD;
        float hist[3] = {sp[0 * C + c], sp[1 * C + c], sp[2 * C + c]};  // oldest .. newest
        for (int64_t t = 0; t < T; ++t) {
          float xt = ld(x, t * C + c);
          float acc = 0;
          for (int64_t lag = 0; lag < K; ++lag) {
            float wv = ld(sw, lag * C + c) + ld(dyn, t * (C / HD * K) + h * K + lag);
            float xv = lag == 0 ? xt : hist[3 - lag];
            acc = acc + xv * wv;
          }
          st(out, t * C + c, (acc + xt) / div);
          hist[0] = hist[1];
          hist[1] = hist[2];
          hist[2] = xt;
          if (hp)
            for (int j = 0; j < 3; ++j) hp[(t * 3 + j) * C + c] = hist[j];
        }
        for (int j = 0; j < 3; ++j) sp[j * C + c] = hist[j];
      }
    }, 64);
  }

  void l2_clip(const Tensor& x, const Tensor* scale, float thr, const Tensor* tau, Tensor& out) override {
    const int64_t d = x.dim(-1), H = x.dim(-2), T = x.numel() / (d * H);
    parallel_for(T * H, [&](int64_t b, int64_t e) {
      std::vector<float> v(static_cast<size_t>(d));
      for (int64_t r = b; r < e; ++r) {
        const int64_t h = r % H, t = r / H;
        ld_row(x, r * d, d, v.data());
        if (scale && scale->valid())
          for (int64_t k = 0; k < d; ++k) v[k] *= ld(*scale, h * d + k);
        float ss = 0;
        for (float f : v) ss += f * f;
        float f = std::min(thr / (std::sqrt(ss) + 1e-6f), 1.0f);
        for (int64_t k = 0; k < d; ++k) {
          float y = round_bf16(v[k] * f);
          if (tau && tau->valid()) y = y * tau->data<float>()[t];
          v[k] = y;
        }
        st_row(out, r * d, d, v.data());
      }
    }, 16);
  }

  void sigmoid_gate(const Tensor& a, const Tensor& g, Tensor& out) override {
    const int64_t n = a.numel();
    parallel_for(n, [&](int64_t b, int64_t e) {
      for (int64_t i = b; i < e; ++i) st(out, i, ld(a, i) * sigmoidf(ld(g, i)));
    }, 4096);
  }

  void add(const Tensor& a, const Tensor& b, Tensor& out) override {
    const int64_t n = a.numel();
    parallel_for(n, [&](int64_t s, int64_t e) {
      for (int64_t i = s; i < e; ++i) st(out, i, ld(a, i) + ld(b, i));
    }, 4096);
  }

  void cast(const Tensor& a, Tensor& out) override {
    const int64_t n = a.numel();
    parallel_for(n, [&](int64_t s, int64_t e) {
      for (int64_t i = s; i < e; ++i) st(out, i, ld(a, i));
    }, 4096);
  }

  void attention(const AttnArgs& A, Tensor& out) override {
    const int64_t H = 24, KVH = 8, HD = 128, DR = 16, EXT = 1024;
    const int64_t T = A.q.numel() / (H * HD);
    const int64_t past = A.kv.len;
    const float scale = 1.0f / std::sqrt(float(HD));
    parallel_for(T * H, [&](int64_t b, int64_t e) {
      std::vector<float> q(HD), kf(HD), s, acc(HD), rr(DR);
      for (int64_t r = b; r < e; ++r) {
        const int64_t i = r / H, h = r % H, kh = h / (H / KVH);
        const int64_t pi = past + i;
        ld_row(A.q, (i * H + h) * HD, HD, q.data());
        ld_row(A.r, (i * H + h) * DR, DR, rr.data());
        int64_t first = 0;
        if (A.window > 0) first = std::max<int64_t>(0, pi - A.window + 1);
        const int64_t n = pi - first + 1;
        s.assign(size_t(n), 0.0f);
        auto key_ptr = [&](int64_t pos, bool val) -> std::pair<const Tensor*, int64_t> {
          if (pos >= past) return {val ? &A.v_new : &A.k_new, ((pos - past) * KVH + kh) * HD};
          int64_t slot = A.kv.ring ? pos % A.kv.cap : pos;
          return {val ? &A.kv.v : &A.kv.k, (kh * A.kv.cap + slot) * HD};
        };
        float m = ld(A.sink, i * H + h);
        for (int64_t j = 0; j < n; ++j) {
          const int64_t pos = first + j, dist = pi - pos;
          auto [kt, off] = key_ptr(pos, false);
          ld_row(*kt, off, HD, kf.data());
          float sc = dot(q.data(), kf.data(), HD) * scale;
          if (dist < EXT) {
            float bacc = 0;
            for (int64_t d = 0; d < DR; ++d) bacc += rr[d] * ld(A.rel_proj, d * EXT + dist);
            sc += round_bf16(bacc);
          }
          s[j] = sc;
          m = std::max(m, sc);
        }
        float z = std::exp(ld(A.sink, i * H + h) - m);
        std::fill(acc.begin(), acc.end(), 0.0f);
        for (int64_t j = 0; j < n; ++j) {
          float p = std::exp(s[j] - m);
          z += p;
          auto [vt, off] = key_ptr(first + j, true);
          ld_row(*vt, off, HD, kf.data());
          for (int64_t d = 0; d < HD; ++d) acc[d] += p * kf[d];
        }
        for (int64_t d = 0; d < HD; ++d) acc[d] /= z;
        st_row(out, i * H * HD + h * HD, HD, acc.data());
      }
    });
  }

  void kv_append(const Tensor& k_new, const Tensor& v_new, KvView& kv) override {
    const int64_t KVH = 8, HD = 128, T = k_new.numel() / (KVH * HD);
    for (int64_t t = 0; t < T; ++t) {
      int64_t pos = kv.len + t, slot = kv.ring ? pos % kv.cap : pos;
      if (!kv.ring && slot >= kv.cap) throw std::runtime_error("kv_append: global cache full");
      for (int64_t h = 0; h < KVH; ++h) {
        std::memcpy(kv.k.data<uint16_t>() + (h * kv.cap + slot) * HD, k_new.data<uint16_t>() + (t * KVH + h) * HD, HD * 2);
        std::memcpy(kv.v.data<uint16_t>() + (h * kv.cap + slot) * HD, v_new.data<uint16_t>() + (t * KVH + h) * HD, HD * 2);
      }
    }
    kv.len += T;
  }

  void router(const Tensor& x, const MoeWeights& mw, int topk, float scale, Tensor& idx, Tensor& w) override {
    const int64_t E = mw.gate_w.dim(0), D = mw.gate_w.dim(1), T = x.numel() / D;
    parallel_for(T, [&](int64_t b, int64_t e) {
      std::vector<float> xr(static_cast<size_t>(D)), s(static_cast<size_t>(E)), sel(static_cast<size_t>(E));
      std::vector<int> order(static_cast<size_t>(E));
      for (int64_t t = b; t < e; ++t) {
        ld_row(x, t * D, D, xr.data());
        for (int64_t j = 0; j < E; ++j) {
          s[j] = sigmoidf(dot_bf16(xr.data(), mw.gate_w.data<uint16_t>() + j * D, D));
          sel[j] = s[j] + mw.bias.data<float>()[j];
        }
        for (int j = 0; j < E; ++j) order[j] = j;
        std::partial_sort(order.begin(), order.begin() + topk, order.end(),
                          [&](int a, int c) { return sel[a] > sel[c] || (sel[a] == sel[c] && a < c); });
        float sum = 0;
        for (int k = 0; k < topk; ++k) sum += s[order[k]];
        for (int k = 0; k < topk; ++k) {
          idx.data<int32_t>()[t * topk + k] = order[k];
          w.data<float>()[t * topk + k] = s[order[k]] / (sum + 1e-20f) * scale;
        }
      }
    });
  }

  void moe(const Tensor& x, const Tensor& idx, const Tensor& w, const MoeWeights& mw, Tensor& out) override {
    const int64_t D = x.dim(-1), T = x.numel() / D, K = idx.numel() / T, I = mw.gate.n;
    const int64_t E = mw.gate_w.dim(0);
    std::vector<std::vector<int64_t>> routes(static_cast<size_t>(E));  // route ids (t*K + k) per expert
    for (int64_t r = 0; r < T * K; ++r) routes[idx.data<int32_t>()[r]].push_back(r);
    std::vector<float> X = to_f32(x);
    std::vector<float> dout(static_cast<size_t>(T * K * D));  // bf16-rounded down outputs per route
    std::vector<int64_t> used;
    for (int64_t e = 0; e < E; ++e)
      if (!routes[e].empty()) used.push_back(e);
    auto act = [&](float g, float u) {
      return std::tanh(g / 4.0f) * 4.0f * sigmoidf(g) * std::tanh(u / 25.0f) * 25.0f;
    };
    (void)act;
    parallel_for(int64_t(used.size()), [&](int64_t b, int64_t e_) {
      std::vector<float> wg(static_cast<size_t>(I * D)), wu(static_cast<size_t>(I * D)), wd(static_cast<size_t>(D * I));
      for (int64_t ui = b; ui < e_; ++ui) {
        const int64_t e = used[ui];
        dequant_expert(mw.gate, e, wg.data());
        dequant_expert(mw.up, e, wu.data());
        dequant_expert(mw.down, e, wd.data());
        const auto& rs = routes[e];
        const int64_t n = int64_t(rs.size());
        std::vector<float> xs(static_cast<size_t>(n * D)), g(static_cast<size_t>(n * I)), u(static_cast<size_t>(n * I)), a(static_cast<size_t>(n * I)), d(static_cast<size_t>(n * D));
        for (int64_t j = 0; j < n; ++j) std::memcpy(&xs[j * D], &X[(rs[j] / K) * D], size_t(D) * 4);
        if (n >= 8 && !row_exact()) {
          cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, int(n), int(I), int(D), 1.f, xs.data(), int(D), wg.data(), int(D), 0.f, g.data(), int(I));
          cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, int(n), int(I), int(D), 1.f, xs.data(), int(D), wu.data(), int(D), 0.f, u.data(), int(I));
        } else {
          for (int64_t j = 0; j < n; ++j)
            for (int64_t i = 0; i < I; ++i) {
              g[j * I + i] = dot(&xs[j * D], &wg[i * D], D);
              u[j * I + i] = dot(&xs[j * D], &wu[i * D], D);
            }
        }
        const float sg = gscale(mw.gate, e), su = gscale(mw.up, e), sd = gscale(mw.down, e);
        for (int64_t q = 0; q < n * I; ++q) {
          float gg = round_bf16(g[q] * sg), uu = round_bf16(u[q] * su);
          a[q] = round_bf16(std::tanh(gg / 4.0f) * 4.0f * sigmoidf(gg) * std::tanh(uu / 25.0f) * 25.0f);
        }
        if (n >= 8 && !row_exact()) {
          cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, int(n), int(D), int(I), 1.f, a.data(), int(I), wd.data(), int(I), 0.f, d.data(), int(D));
        } else {
          for (int64_t j = 0; j < n; ++j)
            for (int64_t o = 0; o < D; ++o) d[j * D + o] = dot(&a[j * I], &wd[o * I], I);
        }
        for (int64_t j = 0; j < n; ++j)
          for (int64_t o = 0; o < D; ++o) dout[rs[j] * D + o] = round_bf16(d[j * D + o] * sd);
      }
    });
    parallel_for(T, [&](int64_t b, int64_t e) {
      std::vector<float> o(static_cast<size_t>(D));
      for (int64_t t = b; t < e; ++t) {
        std::fill(o.begin(), o.end(), 0.0f);
        for (int64_t k = 0; k < K; ++k) {
          float wk = w.data<float>()[t * K + k];
          const float* dr = &dout[(t * K + k) * D];
          for (int64_t c = 0; c < D; ++c) o[c] += dr[c] * wk;
        }
        st_row(out, t * D, D, o.data());
      }
    });
  }

  void logits(const Tensor& h, const Tensor& head, int64_t vocab, float cap, Tensor& out) override {
    const int64_t D = head.dim(1), T = h.numel() / D;
    std::vector<float> X = to_f32(h);
    for (float& f : X) f = round_bf16(f);
    std::vector<float> Y(static_cast<size_t>(T * vocab));
    gemm_bf16w(X.data(), T, D, head.data<uint16_t>(), vocab, Y.data(), vocab);
    for (float& f : Y) f = std::tanh(f / cap) * cap;
    st_row(out, 0, T * vocab, Y.data());
  }

 private:
  std::map<std::string, std::pair<void*, size_t>> maps_;
};

}  // namespace

std::unique_ptr<Backend> make_cpu_backend() { return std::make_unique<CpuBackend>(); }

}  // namespace ink

