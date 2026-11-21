#include "core/config.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include "core/json.h"

namespace ink {

ModelConfig ModelConfig::load(const std::string& dir) {
  std::ifstream f(dir + "/config.json");
  if (!f) throw std::runtime_error("cannot read " + dir + "/config.json");
  std::stringstream ss;
  ss << f.rdbuf();
  json::Value root = json::parse(ss.str());
  const json::Value& t = root["text_config"];
  ModelConfig c;
  c.hidden = int(t.get_int("hidden_size", c.hidden));
  c.layers = int(t.get_int("num_hidden_layers", c.layers));
  c.vocab = int(t.get_int("vocab_size", c.vocab));
  c.vocab_unpadded = int(t.get_int("unpadded_vocab_size", c.vocab_unpadded));
  c.heads = int(t.get_int("num_attention_heads", c.heads));
  c.kv_heads = int(t.get_int("num_key_value_heads", c.kv_heads));
  c.head_dim = int(t.get_int("head_dim", c.head_dim));
  c.d_rel = int(t.get_int("d_rel", c.d_rel));
  c.rel_extent = int(t.get_int("rel_extent", c.rel_extent));
  c.window = int(t.get_int("sliding_window_size", c.window));
  c.rms_eps = float(t.get_num("rms_norm_eps", c.rms_eps));
  c.qk_clip = float(t.get_num("qk_l2_clip_threshold", c.qk_clip));
  c.rel_clip = float(t.get_num("rel_l2_clip_threshold", c.rel_clip));
  c.log_n_floor = int(t.get_int("log_scaling_n_floor", c.log_n_floor));
  c.log_alpha = float(t.get_num("log_scaling_alpha", c.log_alpha));
  c.sconv_k = int(t.get_int("sconv_kernel_size", c.sconv_k));
  c.experts = int(t.get_int("n_routed_experts", c.experts));
  c.topk = int(t.get_int("num_experts_per_tok", c.topk));
  c.inter = int(t.get_int("intermediate_size", c.inter));
  c.gate_cap = float(t.get_num("mlp_gate_softcap", c.gate_cap));
  c.up_cap = float(t.get_num("mlp_up_softcap", c.up_cap));
  c.route_scale = float(t.get_num("route_scale", c.route_scale));
  c.logit_cap = float(t.get_num("final_logit_softcapping", c.logit_cap));
  c.block = int(t.get_int("n_layers_per_attn_res_block", c.block));
  if (t.has("loop_config")) {
    auto& l = t["loop_config"];
    c.core_start = int(l.get_int("core_start", c.core_start));
    c.core_end = int(l.get_int("core_end", c.core_end));
    c.passes = int(l.get_int("num_passes", c.passes));
    c.gate_rank = int(l.get_int("gate_rank", c.gate_rank));
  }
  if (t.has("use_value_norm_no_gamma") && !t["use_value_norm_no_gamma"].as_bool())
    throw std::runtime_error("expected use_value_norm_no_gamma");
  if (!t.has("attn_res_normalize_values") || !t["attn_res_normalize_values"].as_bool())
    throw std::runtime_error("expected attn_res_normalize_values");
  c.is_local.assign(c.layers, false);
  for (auto& v : t["local_layer_ids"].arr) c.is_local[v.as_int()] = true;
  for (int i = 0; i < c.core_start; ++i) c.visits.push_back(i);
  for (int p = 0; p < c.passes; ++p)
    for (int i = c.core_start; i < c.core_end; ++i) c.visits.push_back(i);
  for (int i = c.core_end; i < c.layers; ++i) c.visits.push_back(i);
  if (root.has("eos_token_id")) c.eos = int(root["eos_token_id"].as_int());
  if (root.has("mtp_config")) {
    auto& m = root["mtp_config"];
    for (auto& v : m["target_layer_ids"].arr) c.mtp_targets.push_back(int(v.as_int()));
    c.mtp_block = int(m.get_int("block_size", c.mtp_block));
  }
  return c;
}

}  // namespace ink
