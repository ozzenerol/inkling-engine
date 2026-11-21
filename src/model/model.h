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
  std::array<Tensor, 4> hist;  // f32 [16,3,C] conv state after each row of the last keep_hist forward
};

class Model {
 public:
  Model(Backend& be, const std::string& model_dir, int64_t max_context, int64_t max_chunk);
  ~Model();
  const ModelConfig& config() const { return cfg_; }
  Backend& backend() { return be_; }

  // Run T (<= max_chunk) tokens at positions len()..len()+T-1 and append them to the cache.
  // Logits of the last row (or of all rows if all_logits) are available via logits() after the call.
  // keep_hist (T <= 16): record conv states after every row so rollback() can keep a prefix (spec verify).
  void forward(const int32_t* ids, int64_t T, bool all_logits = false, bool keep_hist = false);
  // After a keep_hist forward of T rows: keep only the first n_keep rows (conv states + KV lengths).
  void rollback(int64_t n_keep);
  static constexpr int64_t kMaxHist = 32;  // rows of a verify (conv history per row)
  int64_t hist_rows() const { return hist_rows_; }  // rows of the last keep_hist forward still rewindable
  // Test hook: run one visit on an empty cache with given inputs (x, blocks, loop bank), x_out = output.
  void debug_visit(int v, const Tensor& x_in, const std::vector<const Tensor*>& blocks_in,
                   const std::vector<const Tensor*>& bank_in, Tensor& x_out, int64_t T);
  const std::vector<float>& logits() const { return logits_; }  // [rows, vocab_unpadded]
  int64_t logit_rows() const { return logit_rows_; }
  int64_t len() const { return len_; }
  // Tokens currently in the cache (size len()); -1 where unknown (restored past what this Model saw).
  const std::vector<int32_t>& tokens() const { return toks_; }
  void reset();  // empty cache

  // Called with the hidden state entering each visit v (0..47) and v = 48 for the final normed output
  // (bf16/f32 [T,D] device tensors; valid only during the callback, enqueue a copy to keep them).
  std::function<void(int visit, const Tensor& x)> on_visit;

  std::vector<VisitState>& visits() { return vs_; }
  bool alt_layout() const { return alt_layout_; }  // pipenetwork 8-bit checkpoint layout
  const Tensor& embed() const { return embed_; }            // bf16 [vocab, D]
  const Tensor& embed_norm() const { return embed_norm_; }  // bf16 [D]
  const Tensor& lm_head() const { return head_; }           // bf16 [vocab, D]
  // Debug hook: named intermediates inside run_visit (see model.cpp), only when set.
  std::function<void(int visit, const char* name, const Tensor& t)> on_debug;
  void set_len(int64_t n);  // used by Engine after restoring KV/conv state (B)

 private:
  void load(const std::string& dir);
  void run_visit(int v, Tensor& x, std::vector<Tensor>& blocks, const ChainMsg* cm, int64_t T);
  struct Pending {  // tail of the previous visit left for the next chain_front: x = bf16(res + conv(moe out))
    bool on = false;
    ChainConv conv;
    Tensor res, hist_view;
  } pending_;
  bool no_chain_ = false;
  std::vector<int32_t> toks_;
  void write_tau(int64_t T);
  void ensure_hist();
  bool keep_hist_ = false;
  int64_t hist_rows_ = 0;
  Tensor hist_views_[4];
  Tensor view(const Tensor& t, std::initializer_list<int64_t> shape) const;  // contiguous prefix view
  Tensor row_view(const Tensor& t, int64_t first, std::initializer_list<int64_t> shape) const;

  std::vector<Tensor> owned_;   // weights converted at load (8-bit layout)
  bool alt_layout_ = false;
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
