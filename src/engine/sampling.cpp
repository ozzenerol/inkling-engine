#include "engine/sampling.h"

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#include <algorithm>

namespace ink {

int32_t argmax_first(const float* logits, int64_t V) {
  float mx = 0;
  vDSP_maxv(logits, 1, &mx, vDSP_Length(V));
  for (int64_t i = 0; i < V; ++i)
    if (logits[i] == mx) return int32_t(i);  // first maximum, as std::max_element
  return 0;
}

void target_probs(const SamplingParams& sp, const float* logits, int64_t V, std::vector<float>& p) {
  p.resize(static_cast<size_t>(V));
  float mx = 0;
  vDSP_maxv(logits, 1, &mx, vDSP_Length(V));
  const float inv_t = 1.0f / sp.temperature, shift = -mx * inv_t;
  vDSP_vsmsa(logits, 1, &inv_t, &shift, p.data(), 1, vDSP_Length(V));  // (l - max) / T <= 0
  const int n = int(V);
  vvexpf(p.data(), p.data(), &n);  // top token has value 1
  double z = 0;
  for (int64_t i = 0; i < V; ++i) z += p[i];
  const bool top_k = sp.top_k > 0 && sp.top_k < V, top_p = sp.top_p < 1.0f;
  double kept = z;
  if (sp.min_p > 0.0f || top_k || top_p) {
    // Candidates above a threshold, sorted; the threshold drops until they settle top_k / top_p.
    static thread_local std::vector<int32_t> cand;
    float thr = sp.min_p > 0.0f ? sp.min_p : 1e-3f;
    size_t keep = 0;
    while (true) {
      cand.clear();
      double mass = 0;
      for (int64_t i = 0; i < V; ++i)
        if (p[i] >= thr) {
          cand.push_back(int32_t(i));
          mass += p[i];
        }
      const bool all = sp.min_p > 0.0f || thr <= 0.0f;  // min_p fixes the candidate set
      const bool enough = (!top_k || cand.size() >= size_t(sp.top_k)) && (!top_p || mass >= double(sp.top_p) * z);
      if (all || enough) break;
      thr = thr > 1e-30f ? thr * 1e-3f : 0.0f;
    }
    std::sort(cand.begin(), cand.end(), [&](int32_t a, int32_t b) { return p[a] > p[b] || (p[a] == p[b] && a < b); });
    keep = cand.size();
    if (top_k && size_t(sp.top_k) < keep) keep = size_t(sp.top_k);
    if (top_p) {
      double c = 0;
      for (size_t i = 0; i < keep; ++i) {
        c += p[cand[i]] / z;
        if (c >= sp.top_p) {
          keep = i + 1;
          break;
        }
      }
    }
    static thread_local std::vector<float> vals;
    vals.resize(keep);
    kept = 0;
    for (size_t i = 0; i < keep; ++i) kept += (vals[i] = p[cand[i]]);
    std::fill(p.begin(), p.end(), 0.0f);
    for (size_t i = 0; i < keep; ++i) p[cand[i]] = vals[i];
  }
  const float inv = float(1.0 / kept);
  vDSP_vsmul(p.data(), 1, &inv, p.data(), 1, vDSP_Length(V));
}

int32_t sample_from(const std::vector<float>& p, double u) {
  double r = u;
  int32_t last = 0;
  const int64_t V = int64_t(p.size());
  for (int64_t i = 0; i < V; ++i) {
    if (p[i] <= 0.0f) continue;
    last = int32_t(i);
    r -= p[i];
    if (r < 0) return last;
  }
  return last;  // rounding left u just above the total
}

SpecStep spec_accept(const std::vector<float>& p, int32_t x, float qx,
                     const std::function<void(std::vector<float>& q)>& q_full, double u_accept, double u_resample) {
  const float px = p[size_t(x)];
  if (qx > 0.0f && u_accept * double(qx) < double(px)) return {true, x};  // u < min(1, p/q)
  static thread_local std::vector<float> q, res;
  q_full(q);
  const size_t V = p.size();
  res.resize(V);
  double tot = 0;
  for (size_t i = 0; i < V; ++i) {
    float d = p[i] - (i < q.size() ? q[i] : 0.0f);
    res[i] = d > 0.0f ? d : 0.0f;
    tot += res[i];
  }
  if (tot <= 0.0) return {false, sample_from(p, u_resample)};  // p == q numerically
  const float inv = float(1.0 / tot);
  for (auto& v : res) v *= inv;
  return {false, sample_from(res, u_resample)};
}

}  // namespace ink
