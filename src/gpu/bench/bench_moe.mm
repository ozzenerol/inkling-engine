// MoE decode speed on real expert weights of layers 0 (NVFP4) and 23 (MXFP8), random routes. Owner: B.
// Usage: bench_moe [T] [model_dir]
#include <algorithm>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "core/bf16.h"
#include "core/safetensors.h"
#include "gpu/context.h"
#include "model/backend.h"

using namespace ink;

int main(int argc, char** argv) {
  int T = argc > 1 ? atoi(argv[1]) : 1;
  std::string dir = argc > 2 ? argv[2] : std::string(getenv("HOME")) + "/llm/mlx-models/Inkling-2-Mini-MLX-4-8bit";
  auto be = make_metal_backend();
  SafeTensors st;
  st.add_dir(dir);
  std::mt19937 rng(7);
  for (int layer : {0, 23}) {
    std::string pre = "language_model.model.layers." + std::to_string(layer) + ".mlp.";
    MoeWeights m;
    QuantMode mode = layer < 20 ? QuantMode::NVFP4 : QuantMode::MXFP8;
    size_t expert_bytes = 0;
    for (const char* pj : {"gate_proj", "up_proj", "down_proj"}) {
      std::string b = pre + "switch_mlp." + pj + ".";
      auto& we = st.at(b + "weight");
      auto& se = st.at(b + "scales");
      QuantProj q{mode, be->load_weight(we), be->load_weight(se), {}, we.shape[1],
                  mode == QuantMode::NVFP4 ? we.shape[2] * 8 : we.shape[2] * 4};
      if (mode == QuantMode::NVFP4) q.global = be->load_weight(st.at(b + "weight_scale_2"));
      expert_bytes += (we.bytes + se.bytes) / 128;
      (pj[0] == 'g' ? m.gate : pj[0] == 'u' ? m.up : m.down) = q;
    }
    const int64_t D = 2304;
    const int calls = 48;
    std::vector<Tensor> xs, idxs, ws, outs;
    std::normal_distribution<float> nd(0, 1);
    for (int c = 0; c < calls; c++) {
      Tensor x = be->alloc(DType::BF16, {T, D}, "x");
      std::vector<uint16_t> xv(T * D);
      for (auto& v : xv) v = f32_to_bf16(nd(rng));
      be->write(x, xv.data(), xv.size() * 2);
      Tensor idx = be->alloc(DType::I32, {T, 8}, "idx"), w = be->alloc(DType::F32, {T, 8}, "w");
      std::vector<int32_t> iv;
      std::vector<float> wv;
      for (int t = 0; t < T; t++) {
        std::vector<int32_t> e(128);
        for (int i = 0; i < 128; i++) e[i] = i;
        std::shuffle(e.begin(), e.end(), rng);
        for (int k = 0; k < 8; k++) iv.push_back(e[k]), wv.push_back(1.0f);
      }
      be->write(idx, iv.data(), iv.size() * 4);
      be->write(w, wv.data(), wv.size() * 4);
      xs.push_back(x), idxs.push_back(idx), ws.push_back(w), outs.push_back(be->alloc(DType::BF16, {T, D}, "o"));
    }
    std::vector<double> ms;
    for (int rep = 0; rep < 8; rep++) {
      for (int c = 0; c < calls; c++) be->moe(xs[c], idxs[c], ws[c], m, outs[c]);
      be->sync();
      ms.push_back(gpu::Context::get().last_gpu_ms());
    }
    std::sort(ms.begin(), ms.end());
    // distinct experts touched per call (T tokens x 8 routes, union)
    double per_call_ms = ms[ms.size() / 2] / calls;
    double bytes = double(expert_bytes) * 8 * T;  // upper bound (no dedup across tokens)
    printf("layer %d %s T=%d: %.1f us per MoE call (median of 8 x %d calls), %.1f MB/call -> %.0f GB/s\n", layer,
           mode == QuantMode::NVFP4 ? "nvfp4" : "mxfp8", T, per_call_ms * 1e3, calls, bytes / 1e6,
           bytes / 1e9 / (per_call_ms / 1e3));
  }
}
