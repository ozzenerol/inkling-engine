// DSpark drafter (Inkling 2's bundled mtp.safetensors) on the Metal backend. Owner: B.
// Reference: mlx_vlm/speculative/drafters/inkling2_dspark/dspark.py.
//
// The drafter keeps its own context KV (3 layers; layers 0, 2 local window 1024, layer 1 global), fed with the
// target's captured hidden states (inputs of visits 5, 23, 41 and bf16 of the final normed hidden) for every
// committed token. draft() proposes `count` tokens after an anchor token, entirely on the GPU.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/tensor.h"

#define INK_DSPARK_HAS_SAMPLE 1  // draft_sample() / q_row() exist

namespace ink {

class Backend;

class DSpark {
 public:
  // embed, embed_norm, lm_head: the target's tensors (shared, not copied). max_ctx: context capacity.
  DSpark(Backend& be, const std::string& model_dir, const Tensor& embed, const Tensor& embed_norm,
         const Tensor& lm_head, int64_t max_ctx, int64_t max_rows);
  ~DSpark();

  static constexpr int kCaptures = 4;  // target visits 5, 23, 41 (inputs) and 48 (final normed)
  static const int kCaptureVisits[kCaptures];

  // Capture buffer rows (bf16 [max_rows, 4, D]); the engine copies target hidden rows here during a forward.
  Tensor& capture() { return cap_; }
  // Enqueue a copy of a target hidden tensor x (bf16 or f32 [T, D]) into capture slot `slot` (0..3).
  void capture_rows(const Tensor& x, int slot);
  // Append rows [0, n) of the capture buffer to the draft context (positions ctx_len() ..). Enqueued only.
  void ingest(int64_t n);
  int64_t ctx_len() const { return ctx_len_; }
  // Forget context after position n (rewind / restore of the target). Local windows keep whatever they hold
  // (draft quality only; the target verifies every token).
  void truncate(int64_t n);
  // Propose `count` (<= 7) tokens after `anchor`. Syncs the GPU and returns the tokens.
  std::vector<int32_t> draft(int32_t anchor, int count);
  // Sampled drafts (speculative sampling): x_j ~ q_j = softmax(z_j / temperature) by inverse CDF with uniforms[j]
  // (on the GPU); x_j feeds the next Markov step. Returns the tokens; q_tok[j] = q_j(x_j).
  std::vector<int32_t> draft_sample(int32_t anchor, int count, float temperature, const float* uniforms,
                                    std::vector<float>* q_tok);
  // Full q_j row (vocab_unpadded floats) of the last draft_sample, computed on the host from the stored logits.
  const std::vector<float>& q_row(int j);
  int64_t vocab() const;
  int max_count() const { return block_ - 1; }

 private:
  struct Impl;
  std::vector<int32_t> run(int32_t anchor, int count, float temperature, const float* uniforms,
                           std::vector<float>* q_tok);
  std::unique_ptr<Impl> impl_;
  std::vector<float> qrow_;
  Tensor cap_;
  int64_t ctx_len_ = 0;
  int block_ = 8;
};

}  // namespace ink
