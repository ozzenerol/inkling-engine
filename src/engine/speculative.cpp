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

#include <random>
#include <unordered_map>

#include "engine/sampling.h"
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
  double t_plain = 0, t_draft = 0, t_verify[17] = {};  // ms, 0 = not measured yet
  long plain_steps = 0;
  double arm_rate[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};  // tokens per ms delivered by draft count k (EMA)
  long arm_n[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  int since_probe = 0, probe = 0;
  // Prompt-lookup drafts (A): trigram -> position after its latest occurrence in the cached tokens.
  std::unordered_map<uint64_t, int32_t> tri;
  size_t tri_upto = 0;      // trigrams ending before this index are in the map
  double lk_rate = 0;       // tokens per ms delivered by lookup rounds (EMA)
  long lk_n = 0, lk_rounds = 0, lk_drafted = 0, lk_accepted = 0;
  int lk_skip = 0, lk_k = 7;
  std::mt19937_64 rng{0x9E3779B97F4A7C15ull};  // speculative sampling uniforms
  uint64_t seeded = ~0ull;
  std::vector<float> pt;
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
  if (p.temperature > 0.0f && s.seeded != p.seed) {
    s.rng.seed(p.seed ? p.seed * 0x9E3779B97F4A7C15ull + 1 : std::random_device{}());
    s.seeded = p.seed;
  }
  const bool adaptive = !getenv("INK_SPEC_FIXED");
  auto ema = [](double& v, double x, double alpha) { v = v == 0 ? x : v + alpha * (x - v); };
  auto ms_since = [](std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
  };
  // Draft count with the best expected tokens per ms: k = 0 is a plain decode step. Every emitted token is an
  // exact target sample whatever k is, so the choice changes speed only. Unmeasured costs are extrapolated from
  // the plain step (+25% per extra row); a full-length round and a plain step are retried now and then so the
  // estimates follow the text (acceptance at temperature 0.7 is far lower than greedy).
  // Draft count per round as a bandit over the arms k = 0 (plain step) .. kmax: each arm keeps an EMA of the
  // tokens per ms it actually delivered; the best arm runs, every arm is tried 3 times first and a non-best arm
  // is probed every 24 rounds. Measured speed only, no acceptance model: an earlier model-based version stopped
  // drafting after a few rejections and never learned that drafting had become good again.
  auto choose_k = [&](int kmax) {
    if (!adaptive) return kmax;
    for (int k = kmax; k >= 0; k--)
      if (s.arm_n[k] < 3) return k;
    int best = 0;
    for (int k = 1; k <= kmax; k++)
      if (s.arm_rate[k] > s.arm_rate[best]) best = k;
    if (++s.since_probe >= 24) {
      s.since_probe = 0;
      s.probe = (s.probe + 1) % (kmax + 1);
      if (s.probe == best) s.probe = (s.probe + 1) % (kmax + 1);
      return s.probe;
    }
    return best;
  };
  auto arm_update = [&](int k, int tokens, double ms) {
    ema(s.arm_rate[k], tokens / ms, 0.15);
    s.arm_n[k]++;
  };
  // Prompt lookup: agents copy a lot from their context (file contents, paths, tool-call boilerplate). When the
  // last three tokens occurred before, the tokens that followed are the draft: no drafter cost, and blocks of up
  // to 15 tokens, verified like any other draft (so the output is unchanged). Used while it delivers more tokens
  // per ms than the drafter arms; the block doubles after a fully accepted round and shrinks after a rejection.
  const bool lookup_on = !getenv("INK_NO_LOOKUP");
  auto tri_key = [](int32_t a, int32_t b2, int32_t c) {
    return (uint64_t(uint32_t(a)) << 40) ^ (uint64_t(uint32_t(b2)) << 20) ^ uint64_t(uint32_t(c));
  };
  auto lookup = [&](int32_t anchor, int kmax, std::vector<int32_t>& out) {
    const auto& tk = m.tokens();
    const size_t n = tk.size();
    if (s.tri_upto > n) {  // the cache was rewound or restored: rebuild
      s.tri.clear();
      s.tri_upto = 0;
    }
    for (size_t i = std::max<size_t>(s.tri_upto, 2); i < n; i++)
      if (tk[i] >= 0 && tk[i - 1] >= 0 && tk[i - 2] >= 0) s.tri[tri_key(tk[i - 2], tk[i - 1], tk[i])] = int32_t(i + 1);
    s.tri_upto = n;
    out.clear();
    if (n < 2) return;
    auto it = s.tri.find(tri_key(tk[n - 2], tk[n - 1], anchor));
    if (it == s.tri.end()) return;
    for (size_t p2 = size_t(it->second); p2 < n && int(out.size()) < kmax; p2++) {
      if (tk[p2] < 0) break;
      out.push_back(tk[p2]);
    }
    // the continuation may run into the pending anchor (a repeat of the last tokens): stop there
  };
  std::vector<int32_t> lk;
  while (true) {
    int k = std::min(count_max, max_tokens - emitted);
    if (k <= 0) break;
    bool use_lookup = false;
    if (lookup_on) {
      if (s.lk_skip > 0) {
        s.lk_skip--;
      } else {
        lookup(b, std::min({s.lk_k, 15, max_tokens - emitted}), lk);
        double best = 0;
        for (int j = 0; j <= count_max; j++) best = std::max(best, s.arm_rate[j]);
        if (lk.size() >= 2) {
          if (s.lk_n < 3 || s.lk_rate >= 0.9 * best) use_lookup = true;
          else s.lk_skip = 24;  // not paying on this text: look again later
        }
      }
    }
    if (use_lookup) k = int(lk.size());
    else k = choose_k(k);
    if (k == 0) {
      // Plain step: feed b, sample the next token. The capture hook ingests the row into the draft context.
      auto t0 = std::chrono::steady_clock::now();
      m.forward(&b, 1, false);
      int32_t t = sample_last();
      ema(s.t_plain, ms_since(t0), 0.2);
      arm_update(0, 1, ms_since(t0));
      s.plain_steps++;
      bool more = on_token(t);
      emitted++;
      if (is_stop(t) || !more || emitted >= max_tokens) break;
      b = t;
      continue;
    }
    auto c0 = std::chrono::steady_clock::now();
    // Temperature > 0 with a sampling drafter: real speculative sampling (drafts sampled from the drafter's
    // q = softmax(z / T), accepted with probability min(1, p / q), residual resample on the first rejection;
    // exact, and accepts 1 - TV(p, q) instead of p(greedy draft)). Otherwise greedy drafts + sample-and-compare.
    bool sampled = false;
    std::vector<float> q_tok;
#ifdef INK_DSPARK_HAS_SAMPLE
    // Opt-in (INK_SPEC_SAMPLE=1): with q at the target's temperature the drafter is flatter than the target and
    // 1 - TV(p, q) came out BELOW p(greedy draft) (1.31 vs 1.47 accepted per round on p2); needs a sharper q.
    sampled = p.temperature > 0.0f && getenv("INK_SPEC_SAMPLE");
#endif
    std::vector<int32_t> drafts;
    if (use_lookup) {
      sampled = false;
      drafts = lk;
    } else if (sampled) {
#ifdef INK_DSPARK_HAS_SAMPLE
      float us[8];
      for (int j = 0; j < k; j++) us[j] = float(std::generate_canonical<double, 53>(s.rng));
      drafts = d.draft_sample(b, k, p.temperature, us, &q_tok);
#endif
    } else {
      drafts = d.draft(b, k);
    }
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
    if (!use_lookup) ema(s.t_draft, std::chrono::duration<double, std::milli>(c1 - c0).count(), 0.2);
    ema(s.t_verify[k + 1], std::chrono::duration<double, std::milli>(c2 - c1).count(), 0.2);
    const auto& lg = m.logits();
    // Sample-and-compare (as mlx_vlm's DSpark walk): sample the target at each row in order; accept the draft
    // while it equals the sample. Every emitted token is an exact target sample (greedy: the argmax).
    std::vector<int32_t> y(verify.size());
    int a = 0;
    for (size_t j = 0; sampled && j < verify.size(); j++) {
      const float* row = lg.data() + j * V;
      if (int(j) == k) {  // every draft accepted: the last row gives a free target sample
        y[j] = e.sample(p, row, V);
        break;
      }
      target_probs(p, row, V, s.pt);
      SpecStep st = spec_accept(
          s.pt, drafts[j], q_tok[j],
          [&](std::vector<float>& q) {
#ifdef INK_DSPARK_HAS_SAMPLE
            const float* qr = d.q_row(int(j));
            q.assign(qr, qr + V);
#else
            q.assign(size_t(V), 0.0f);
#endif
          },
          std::generate_canonical<double, 53>(s.rng), std::generate_canonical<double, 53>(s.rng));
      y[j] = st.token;
      if (!st.accepted) break;
      a++;
    }
    for (size_t j = 0; !sampled && j < verify.size(); j++) {
      const float* row = lg.data() + j * V;
      y[j] = p.temperature <= 0.0f ? int32_t(std::max_element(row, row + V) - row) : e.sample(p, row, V);
      if (int(j) < k && drafts[j] == y[j]) {
        a++;
        continue;
      }
      break;
    }
    if (getenv("INK_SPEC_DEBUG") && s.rounds < 1000) {
      fprintf(stderr, "round %ld%s: b=%d drafts", s.rounds, use_lookup ? " (lookup)" : "", b);
      for (int x : drafts) fprintf(stderr, " %d", x);
      fprintf(stderr, " | target");
      for (int x : y) fprintf(stderr, " %d", x);
      fprintf(stderr, " | a=%d\n", a);
    }
    if (use_lookup) {
      s.lk_rounds++;
      s.lk_drafted += k;
      s.lk_accepted += a;
      ema(s.lk_rate, (a + 1) / std::chrono::duration<double, std::milli>(c2 - c0).count(), 0.2);
      s.lk_n++;
      s.lk_k = a == k ? std::min(15, 2 * s.lk_k + 1) : std::max(3, (a + s.lk_k) / 2);
    } else {
      s.rounds++;
      s.drafted += k;
      s.accepted += a;
      for (int j = 0; j < k && j <= a; j++) ema(s.acc[j], j < a ? 1.0 : 0.0, 0.08);  // positions reached this round
      arm_update(k, a + 1, std::chrono::duration<double, std::milli>(c2 - c0).count());
    }
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
    fprintf(stderr, "lookup: %ld rounds, accepted %ld of %ld drafts, %.1f tok/s in lookup rounds\n", s.lk_rounds,
            s.lk_accepted, s.lk_drafted, 1e3 * s.lk_rate);
  if (getenv("INK_SPEC_STATS"))
    fprintf(stderr, "dspark adaptive: %ld plain steps, plain %.1f ms, draft %.1f ms, verify T=2..5 %.1f %.1f %.1f %.1f ms, "
                    "acceptance by position %.2f %.2f %.2f, tok/s by draft count %.1f %.1f %.1f %.1f\n",
            s.plain_steps, s.t_plain, s.t_draft, s.t_verify[2], s.t_verify[3], s.t_verify[4], s.t_verify[5], s.acc[0],
            s.acc[1], s.acc[2], 1e3 * s.arm_rate[0], 1e3 * s.arm_rate[1], 1e3 * s.arm_rate[2], 1e3 * s.arm_rate[3]);
  return true;
}

}  // namespace ink
