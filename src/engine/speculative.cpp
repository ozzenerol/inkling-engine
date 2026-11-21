// DSpark speculative decoding for Engine::generate (greedy and sampled). Owner: B.
// draft (GPU) -> verify [bonus, drafts] in one target forward with conv history -> accept the longest matching
// prefix -> roll the cache back to the accepted rows -> feed their captured hidden states to the drafter.
#include "engine/speculative.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>

#include "gpu/dspark.h"
#include "model/model.h"

namespace ink {

namespace {
struct Spec {
  std::unique_ptr<DSpark> dsp;
  bool in_verify = false;
  long rounds = 0, drafted = 0, accepted = 0;
};
std::map<const Engine*, std::unique_ptr<Spec>>& registry() {
  static std::map<const Engine*, std::unique_ptr<Spec>> r;
  return r;
}
int capture_slot(int visit) {
  for (int i = 0; i < DSpark::kCaptures; i++)
    if (DSpark::kCaptureVisits[i] == visit) return i;
  return -1;
}
}  // namespace

// Called once by the Engine constructor when options().use_mtp: loads the drafter and hooks the captures so every
// target forward (prefill chunks too) feeds the draft context.
void speculative_attach(Engine& e, Model& m) {
  if (getenv("INK_NO_DSPARK")) return;
  auto sp = std::make_unique<Spec>();
  try {
    sp->dsp = std::make_unique<DSpark>(m.backend(), e.options().model_dir, m.embed(), m.embed_norm(), m.lm_head(),
                                       e.options().max_context, std::max<int64_t>(e.options().prefill_chunk, 16));
  } catch (const std::exception& ex) {
    fprintf(stderr, "dspark: disabled (%s)\n", ex.what());
    return;
  }
  Spec* s = sp.get();
  Model* mp = &m;
  auto prev = m.on_visit;  // chain an existing hook (e.g. inktest --hidden)
  m.on_visit = [s, mp, prev](int visit, const Tensor& x) {
    if (prev) prev(visit, x);
    DSpark& d = *s->dsp;
    if (visit == 0) {
      // Follow the target after rewind/restore/reset: positions >= len are gone.
      if (d.ctx_len() != mp->len()) d.truncate(mp->len());
    }
    int slot = capture_slot(visit);
    if (slot < 0) return;
    d.capture_rows(x, slot);
    if (slot == DSpark::kCaptures - 1 && !s->in_verify) {
      d.ingest(x.numel() / x.dim(-1));
    }
  };
  registry()[&e] = std::move(sp);
}

bool speculative_generate(Engine& e, Model& m, const SamplingParams& p, int max_tokens,
                          const std::vector<int32_t>& stop_ids, const std::function<bool(int32_t)>& on_token,
                          const std::function<int32_t()>& sample_last) {
  auto it = registry().find(&e);
  if (it == registry().end() || !it->second->dsp) return false;
  Spec& s = *it->second;
  DSpark& d = *s.dsp;
  const int64_t V = m.config().vocab_unpadded;
  int block = std::max(2, e.options().draft_block);
  int count_max = std::min(block - 1, d.max_count());
  auto is_stop = [&](int32_t t) { return std::find(stop_ids.begin(), stop_ids.end(), t) != stop_ids.end(); };

  int32_t b = sample_last();
  int emitted = 0;
  {
    bool more = on_token(b);
    emitted++;
    if (is_stop(b) || !more || emitted >= max_tokens) return true;
  }
  if (d.ctx_len() != m.len()) d.truncate(m.len());
  std::vector<int32_t> verify;
  while (true) {
    int k = std::min(count_max, max_tokens - emitted);
    if (k <= 0) break;
    std::vector<int32_t> drafts = d.draft(b, k);
    verify.assign(1, b);
    verify.insert(verify.end(), drafts.begin(), drafts.end());
    const int64_t base = m.len();
    s.in_verify = true;
    m.forward(verify.data(), int64_t(verify.size()), true, true);
    s.in_verify = false;
    const auto& lg = m.logits();
    // Sample-and-compare (as mlx_vlm's DSpark walk): sample the target at each row in order; accept the draft
    // while it equals the sample. Every emitted token is an exact target sample (greedy: the argmax).
    std::vector<int32_t> y(verify.size());
    int a = 0;
    for (size_t j = 0; j < verify.size(); j++) {
      const float* row = lg.data() + j * V;
      y[j] = p.temperature <= 0.0f ? int32_t(std::max_element(row, row + V) - row) : e.sample(p, row, V);
      if (int(j) < k && drafts[j] == y[j]) {
        a++;
        continue;
      }
      break;
    }
    s.rounds++;
    s.drafted += k;
    s.accepted += a;
    // emit d_0..d_{a-1}, then y_a; stop as soon as a stop token / limit / on_token=false shows up
    int keep = a + 1;  // cache rows kept: [b, d_0..d_{a-1}]
    bool done = false;
    int32_t next = y[a];
    for (int j = 0; j <= a; j++) {
      int32_t t = j < a ? drafts[j] : y[a];
      bool more = on_token(t);
      emitted++;
      if (is_stop(t) || !more || emitted >= max_tokens) {
        keep = j + 1;  // the last emitted token (t) stays out of the cache
        done = true;
        break;
      }
    }
    if (!e.rewind(base + keep)) {
      fprintf(stderr, "speculative: rewind to %lld failed\n", (long long)(base + keep));
      return true;
    }
    d.ingest(keep);  // captures of rows [0, keep) of the verify forward
    if (done) break;
    b = next;
  }
  if (getenv("INK_SPEC_STATS"))
    fprintf(stderr, "dspark: %ld rounds, accepted %ld of %ld drafts (%.2f per round)\n", s.rounds, s.accepted,
            s.drafted, s.rounds ? double(s.accepted) / s.rounds : 0.0);
  return true;
}

}  // namespace ink
