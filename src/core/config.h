// Owner: A. Inkling 2 Mini text config (config.json "text_config" + loop/mtp).
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace ink {

struct ModelConfig {
  int hidden = 2304;
  int layers = 24;
  int vocab = 201024;
  int vocab_unpadded = 200058;
  int heads = 24, kv_heads = 8, head_dim = 128;
  int d_rel = 16, rel_extent = 1024;
  int window = 1024;
  float rms_eps = 1e-6f;
  float qk_clip = 22.627416997969522f, rel_clip = 8.0f;
  int log_n_floor = 8192;
  float log_alpha = 0.1f;
  int sconv_k = 4;
  int experts = 128, topk = 8, inter = 1152;
  float gate_cap = 4.0f, up_cap = 25.0f, route_scale = 8.0f;
  float logit_cap = 30.0f;
  int block = 6;  // n_layers_per_attn_res_block
  int core_start = 6, core_end = 18, passes = 3, gate_rank = 64;
  std::vector<bool> is_local;    // per layer
  std::vector<int> visits;       // layer index per visit (48)
  int eos = 200006;
  std::vector<int> mtp_targets;  // capture visits for DSpark (5, 23, 41, 48)
  int mtp_block = 8;

  static ModelConfig load(const std::string& model_dir);
  int num_visits() const { return int(visits.size()); }
};

}  // namespace ink
