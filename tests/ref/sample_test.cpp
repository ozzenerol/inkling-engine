// Owner: A. Sampler check without a model: distribution of Engine-style sampling vs exact softmax, and timing.
// Build: part of the top-level cmake (target sample_test). Uses the sampler through a tiny shim in engine.cpp.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "engine/engine.h"

namespace ink { int32_t sample_for_test(const SamplingParams& p, const float* logits, int64_t V, uint64_t* state); }

int main() {
  const int64_t V = 200058;
  std::mt19937 g(7);
  std::normal_distribution<float> nd(0, 3);
  std::vector<float> lg(V);
  for (auto& x : lg) x = nd(g);
  lg[123] = 14.0f; lg[77] = 13.5f; lg[9000] = 12.0f;
  ink::SamplingParams p; p.temperature = 0.7f; p.seed = 42;
  uint64_t st = 0;
  const int N = 20000;
  std::vector<int> cnt(V, 0);
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < N; ++i) cnt[ink::sample_for_test(p, lg.data(), V, &st)]++;
  double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / N;
  // exact probabilities
  double m = 14.0, z = 0;
  for (auto x : lg) z += std::exp((x - m) / 0.7);
  double worst = 0;
  for (int id : {123, 77, 9000}) {
    double pe = std::exp((lg[id] - m) / 0.7) / z, po = double(cnt[id]) / N;
    double sd = std::sqrt(pe * (1 - pe) / N);
    std::printf("token %d: exact %.4f sampled %.4f (%.1f sigma)\n", id, pe, po, (po - pe) / sd);
    worst = std::max(worst, std::fabs(po - pe) / sd);
  }
  std::printf("full-vocab sample: %.3f ms per call\n", ms);
  p.top_p = 0.9f; p.top_k = 40;
  t0 = std::chrono::steady_clock::now();
  int outside = 0;
  for (int i = 0; i < 20000; ++i) { int t = ink::sample_for_test(p, lg.data(), V, &st); outside += !(t == 123 || t == 77 || t == 9000); }
  std::printf("top_k 40 + top_p 0.9: %.3f ms per call, %d samples outside the top 3\n",
              std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 20000, outside);
  p = {}; 
  int a = ink::sample_for_test(p, lg.data(), V, &st);
  std::printf("greedy -> %d (expect 123)\n", a);
  return (worst < 4.0 && a == 123) ? 0 : 1;
}
