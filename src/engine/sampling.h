// Owner: A. Target distribution, sampling and the speculative-sampling acceptance step (host side).
#pragma once
#include <cstdint>
#include <functional>
#include <vector>

#include "engine/engine.h"

namespace ink {

// p = softmax(logits / temperature) restricted by the filters (min_p relative to the top probability, top_k,
// top_p over the unfiltered distribution), renormalized; zero outside the kept set. Requires temperature > 0.
// Full vocabulary, one vectorized exp pass.
void target_probs(const SamplingParams& sp, const float* logits, int64_t V, std::vector<float>& p);

// Inverse-CDF sample from a normalized p with u in [0, 1).
int32_t sample_from(const std::vector<float>& p, double u);

int32_t argmax_first(const float* logits, int64_t V);

// One step of speculative sampling (Leviathan et al. 2023), exact for any proposal q:
// x was sampled from q with probability qx = q(x). Accept x with probability min(1, p(x) / qx); otherwise the
// token is resampled from normalize(max(0, p - q)), for which q_full fills the whole q row (called only on a
// rejection). u_accept, u_resample: independent uniforms in [0, 1). The returned token is distributed as p.
struct SpecStep {
  bool accepted;
  int32_t token;
};
SpecStep spec_accept(const std::vector<float>& p, int32_t x, float qx,
                     const std::function<void(std::vector<float>& q)>& q_full, double u_accept, double u_resample);

}  // namespace ink
