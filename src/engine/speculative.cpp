// DSpark speculative decoding for Engine::generate (greedy and sampled). Owner: B.
// draft (GPU) -> verify [bonus, drafts] in one target forward with conv history -> accept the longest matching
// prefix -> roll the cache back to the accepted rows -> feed their captured hidden states to the drafter.
#include "engine/speculative.h"

#include <algorithm>
#include <chrono>
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
  double draft_ms = 0, verify_ms = 0;
  // Adaptive draft length (A): running estimates, kept across requests.
  double acc[8] = {0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7};  // P(draft j accepted | drafts before it accepted)
  double t_plain = 0, t_draft = 0, t_verify[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};  // ms, 0 = not measured yet
  long plain_steps = 0, since_full = 0, since_plain = 0;
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
  if (getenv("INK_SPEC_STATS")) fprintf(stderr, "dspark: attached to engine %p\n", (void*)&e);
}

bool speculative_generate(Engine& e, Model& m, const SamplingParams& p, int max_tokens,
                          const std::vector<int32_t>& stop_ids, const std::function<bool(int32_t)>& on_token,
                          const std::function<int32_t()>& sample_last) {
  auto it = registry().find(&e);
  if (it == registry().end() || !it->second->dsp) {
    if (getenv("INK_SPEC_STATS")) fprintf(stderr, "dspark: no drafter attached to this engine\n");
    return false;
  }
  Spec& s = *it->second;
  DSpark& d = *s.dsp;
  if (getenv("INK_SPEC_STATS")) fprintf(stderr, "dspark: generate, ctx %lld, model len %lld\n", (long long)d.ctx_len(),
                                        (long long)m.len());
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
  const bool adaptive = !getenv("INK_SPEC_FIXED");
  auto ema = [](double& v, double x, double alpha) { v = v == 0 ? x : v + alpha * (x - v); };
  auto ms_since = [](std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
  };
  // Draft count with the best expected tokens per ms: k = 0 is a plain decode step. Every emitted token is an
  // exact target sample whatever k is, so the choice changes speed only. Unmeasured costs are extrapolated from
  // the plain step (+25% per extra row); a full-length round and a plain step are retried now and then so the
  // estimates follow the text (acceptance at temperature 0.7 is far lower than greedy).
  auto choose_k = [&](int kmax) {
    if (!adaptive) return kmax;
    if (s.rounds < 4 || s.since_full >= 48) return kmax;
    if (s.t_plain == 0 || s.since_plain >= 96) return 0;
    double best = 1.0 / s.t_plain, cum = 1.0, tokens = 1.0;
    int bk = 0;
    for (int k = 1; k <= kmax; k++) {
      cum *= s.acc[k - 1];
      tokens += cum;
      double tv = s.t_verify[k + 1] > 0 ? s.t_verify[k + 1] : s.t_plain * (1.0 + 0.25 * k);
      double rate = tokens / (tv + (s.t_draft > 0 ? s.t_draft : 0.3 * s.t_plain));
      if (rate > best) {
        best = rate;
        bk = k;
      }
    }
    return bk;
  };
  while (true) {
    int k = std::min(count_max, max_tokens - emitted);
    if (k <= 0) break;
    k = choose_k(k);
    if (k == 0) {
      // Plain step: feed b, sample the next token. The capture hook ingests the row into the draft context.
      auto t0 = std::chrono::steady_clock::now();
      m.forward(&b, 1, false);
      int32_t t = sample_last();
      ema(s.t_plain, ms_since(t0), 0.2);
      s.plain_steps++;
      s.since_plain = 0;
      s.since_full++;
      bool more = on_token(t);
      emitted++;
      if (is_stop(t) || !more || emitted >= max_tokens) break;
      b = t;
      continue;
    }
    s.since_plain++;
    s.since_full = k == std::min(count_max, max_tokens - emitted) ? 0 : s.since_full + 1;
    auto c0 = std::chrono::steady_clock::now();
    std::vector<int32_t> drafts = d.draft(b, k);
    auto c1 = std::chrono::steady_clock::now();
    verify.assign(1, b);
    verify.insert(verify.end(), drafts.begin(), drafts.end());
    const int64_t base = m.len();
    s.in_verify = true;
    m.forward(verify.data(), int64_t(verify.size()), true, true);
    s.in_verify = false;
    auto c2 = std::chrono::steady_clock::now();
    s.draft_ms += std::chrono::duration<double, std::milli>(c1 - c0).count();
    s.verify_ms += std::chrono::duration<double, std::milli>(c2 - c1).count();
    ema(s.t_draft, std::chrono::duration<double, std::milli>(c1 - c0).count(), 0.2);
    ema(s.t_verify[k + 1], std::chrono::duration<double, std::milli>(c2 - c1).count(), 0.2);
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
    if (getenv("INK_SPEC_DEBUG") && s.rounds < 1000) {
      fprintf(stderr, "round %ld: b=%d drafts", s.rounds, b);
      for (int x : drafts) fprintf(stderr, " %d", x);
      fprintf(stderr, " | target");
      for (int x : y) fprintf(stderr, " %d", x);
      fprintf(stderr, " | a=%d\n", a);
    }
    s.rounds++;
    s.drafted += k;
    s.accepted += a;
    for (int j = 0; j < k && j <= a; j++) ema(s.acc[j], j < a ? 1.0 : 0.0, 0.08);  // positions reached this round
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
    fprintf(stderr, "dspark: %ld rounds, accepted %ld of %ld drafts (%.2f per round), draft %.2f ms, verify %.2f ms per round\n",
            s.rounds, s.accepted, s.drafted, s.rounds ? double(s.accepted) / s.rounds : 0.0,
            s.rounds ? s.draft_ms / s.rounds : 0.0, s.rounds ? s.verify_ms / s.rounds : 0.0);
  if (getenv("INK_SPEC_STATS"))
    fprintf(stderr, "dspark adaptive: %ld plain steps, plain %.1f ms, draft %.1f ms, verify T=2..5 %.1f %.1f %.1f %.1f ms, "
                    "acceptance by position %.2f %.2f %.2f\n",
            s.plain_steps, s.t_plain, s.t_draft, s.t_verify[2], s.t_verify[3], s.t_verify[4], s.t_verify[5], s.acc[0],
            s.acc[1], s.acc[2]);
  return true;
}

}  // namespace ink
