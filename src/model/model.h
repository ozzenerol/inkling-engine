// Owner: A. Inkling 2 Mini forward pass over an abstract Backend.
#pragma once
#include <array>
#include <functional>
#include <string>
#include <vector>

#include "core/config.h"
#include "core/safetensors.h"
#include "model/backend.h"

namespace ink {

struct LayerWeights {
  bool local = true;
  Tensor attn_query;  // f32 [nq,D] (nq = 1 local, 3 global); invalid for layer 0
  Tensor mlp_query;   // f32 [D]
  Tensor in_norm;     // local
  Tensor q_norm, k_norm, v_norm;  // global
  Tensor post_norm;
  Tensor q_proj, k_proj, v_proj, r_proj, o_proj, gate_proj, sink_proj;
  Tensor q_scale, rel_scale, rel_proj;
  Tensor k_conv, v_conv;            // local: bf16 [1024,4,1]
  Tensor k_static, v_static;        // global: bf16 [4,1024]
  Tensor k_wproj, v_wproj;          // global: bf16 [32,D]
  Tensor attn_sconv, mlp_sconv;     // bf16 [D,4,1]
  bool has_ctrl = false;
  std::array<Tensor, 4> ctrl_down, ctrl_forget, ctrl_gain;  // attn, k, v, mlp
  MoeWeights moe;
};

struct VisitState {
  KvView kv;
  std::array<Tensor, 4> conv;  // f32 [3,C]: k(1024), v(1024), attn(D), mlp(D)
};

class Model {
 public:
  Model(Backend& be, const std::string& model_dir, int64_t max_context, int64_t max_chunk);
  ~Model();
  const ModelConfig& config() const { return cfg_; }
  Backend& backend() { return be_; }

  // Run T (<= max_chunk) tokens at positions len()..len()+T-1 and append them to the cache.
  // Logits of the last row (or of all rows if all_logits) are available via logits() after the call.
  void forward(const int32_t* ids, int64_t T, bool all_logits = false);
  const std::vector<float>& logits() const { return logits_; }  // [rows, vocab_unpadded]
  int64_t logit_rows() const { return logit_rows_; }
  int64_t len() const { return len_; }
  void reset();  // empty cache

  // Called with the hidden state entering each visit v (0..47) and v = 48 for the final normed output
  // (bf16/f32 [T,D] device tensors; valid only during the callback, enqueue a copy to keep them).
  std::function<void(int visit, const Tensor& x)> on_visit;

  std::vector<VisitState>& visits() { return vs_; }
  void set_len(int64_t n);  // used by Engine after restoring KV/conv state (B)

 private:
  void load(const std::string& dir);
  void layer(int visit, int L, const Tensor& x, Tensor& out, int64_t T, const Tensor* msg);
  Tensor view(const Tensor& t, std::initializer_list<int64_t> shape) const;  // contiguous prefix view
  Tensor row_view(const Tensor& t, int64_t first, std::initializer_list<int64_t> shape) const;

  Backend& be_;
  ModelConfig cfg_;
  SafeTensors st_;
  int64_t max_ctx_, max_chunk_;
  int64_t len_ = 0;

  Tensor embed_, embed_norm_, final_query_, final_norm_, head_, msg_queries_;
  std::vector<LayerWeights> lw_;
  std::vector<VisitState> vs_;

  // Workspace sized for max_chunk.
  Tensor ids_, x_, x2_, prelude_, blk_[4], bank_[2], msg_, mixed_, parts_[3], partsb_[3], kraw_, vraw_, kc_, vc_,
      dk_, dv_, q_, k_, v_, r_, sink_, attn_, gate_, ao_, res_, resb_, hid_, hn_, ridx_, rw_, moe_, mo_, h_, hfin_,
      tau_, logits_dev_;
  std::vector<float> logits_;
  int64_t logit_rows_ = 0;
};

}  // namespace ink
