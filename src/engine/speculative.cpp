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
#include <array>
#include <unordered_map>

#include "core/parallel.h"
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
  double t_plain = 0, t_draft = 0, t_verify[33] = {};  // ms, 0 = not measured yet
  long plain_steps = 0;
  double arm_rate[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};  // tokens per ms delivered by draft count k (EMA)
  long arm_n[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  int since_probe = 0, probe = 0;
  int k_cur = 3, rest = 0;
  double draft_rate = 0;
  long draft_n = 0;
  // Prompt-lookup drafts (A): trigram -> position after its latest occurrence in the cached tokens.
  std::unordered_map<uint64_t, std::array<int32_t, 4>> tri;
  int lk_match = 0;
  size_t tri_upto = 0;      // trigrams ending before this index are in the map
  double lk_rate = 0;       // tokens per ms delivered by lookup rounds (EMA)
  long lk_n = 0, lk_rounds = 0, lk_drafted = 0, lk_accepted = 0;
  int lk_skip = 0, lk_k = 7;
  // Confidence gate: P(draft accepted | the drafter's margin bucket), learnt as the text goes (start values from
  // 22k drafts of agent and free text at temperature 0), and a linear cost model of a verify of T rows.
  double pm[11] = {0.24, 0.24, 0.30, 0.33, 0.47, 0.52, 0.66, 0.78, 0.90, 0.96, 0.99};
  double vc0 = 0, vc1 = 0;  // ms: verify(T) = vc0 + vc1 * (T - 1); 0 = not measured yet
  long gate_cut = 0, gate_zero = 0;
  long vhist[33] = {};  // verify forwards by row count
  int gate_k = 7;  // drafts asked of the drafter next round (each costs ~0.45 ms whether kept or not)
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
  // The drafter says how sure it is of each token (top logit minus runner-up) and that predicts acceptance well:
  // 25% below a margin of 1, 50% at 2-3, 78% at 4-6, 96-99% above 8, at every draft position. So a round drafts
  // the whole block and keeps the prefix with the best expected tokens per ms. The output does not depend on it.
  const bool gate = adaptive && !(getenv("INK_SPEC_GATE") && getenv("INK_SPEC_GATE")[0] == '0');
  auto bucket = [](float mg) {
    static const float edges[10] = {0.25f, 0.5f, 1, 1.5f, 2, 3, 4, 6, 8, 12};
    int i = 0;
    while (i < 10 && mg >= edges[i]) i++;
    return i;
  };
  auto verify_cost = [&](int T) {
    const double c0 = s.vc0 > 0 ? s.vc0 : (s.t_plain > 0 ? s.t_plain : 24.0), c1 = s.vc1 > 0 ? s.vc1 : 7.0;
    return c0 + c1 * (T - 1);
  };
  auto verify_seen = [&](int T, double ms) {  // normalized LMS step on (vc0, vc1)
    if (s.vc0 <= 0) s.vc0 = s.t_plain > 0 ? s.t_plain : 24.0, s.vc1 = 7.0;
    const double x = T - 1, err = ms - (s.vc0 + s.vc1 * x), mu = 0.3 / (1.0 + x * x);
    s.vc0 = std::max(1.0, s.vc0 + mu * err);
    s.vc1 = std::max(0.5, s.vc1 + mu * err * x);
  };
  auto gate_cut = [&](const std::vector<float>& mg, int k) {
    const double td = s.t_draft > 0 ? s.t_draft : 9.0;
    double reach = 1.0, tokens = 1.0, best = tokens / (td + verify_cost(1));
    int cut = 0;
    for (int j = 0; j < k; j++) {
      reach *= s.pm[bucket(mg[size_t(j)])];
      tokens += reach;
      const double r = tokens / (td + verify_cost(j + 2));
      if (r > best) best = r, cut = j + 1;
    }
    return cut;
  };
  auto ema = [](double& v, double x, double alpha) { v = v == 0 ? x : v + alpha * (x - v); };
  auto ms_since = [](std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
  };
  // Draft count with the best expected tokens per ms: k = 0 is a plain decode step. Every emitted token is an
  // exact target sample whatever k is, so the choice changes speed only. Unmeasured costs are extrapolated from
  // the plain step (+25% per extra row); a full-length round and a plain step are retried now and then so the
  // estimates follow the text (acceptance at temperature 0.7 is far lower than greedy).
  // Draft count per round. Two modes compared by the tokens per ms they actually deliver (EMAs): plain steps
  // (k = 0) and drafting, where k follows the text: it doubles (2k + 1) after a fully accepted round and drops to
  // the accepted count + 1 after a rejection. Drafting starts; plain is measured with 2 steps after the first 6
  // rounds; when drafting falls below plain it rests for 16 plain steps and is tried again. (An 8-arm bandit over
  // k needed 24 rounds of exploration, more than a short answer has: 34 tok/s on p2 against 41 with block 4.)
  auto choose_k = [&](int kmax) {
    if (!adaptive) return kmax;
    if (s.rest > 0) {
      s.rest--;
      return 0;
    }
    if (s.rounds >= 6 && s.arm_n[0] < 2) return 0;
    if (s.arm_n[0] >= 2 && s.draft_n >= 6 && s.draft_rate < 0.97 * s.arm_rate[0]) {
      s.rest = 16;
      s.draft_n = 3;  // a few fresh rounds decide again after the rest
      return 0;
    }
    return gate ? std::min(kmax, s.gate_k) : std::max(1, std::min(kmax, s.k_cur));
  };
  auto arm_update = [&](int k, int tokens, double ms) {
    ema(s.arm_rate[k], tokens / ms, 0.15);
    s.arm_n[k]++;
    if (k > 0) {
      ema(s.draft_rate, tokens / ms, 0.15);
      s.draft_n++;
      s.k_cur = tokens - 1 == k ? 2 * k + 1 : std::max(1, tokens);  // tokens - 1 = accepted drafts
    }
  };
  // Prompt lookup: agents copy a lot from their context (file contents, paths, tool-call boilerplate). When the
  // last three tokens occurred before, the tokens that followed are the draft: no drafter cost, and blocks of up
  // to 15 tokens, verified like any other draft (so the output is unchanged). Used while it delivers more tokens
  // per ms than the drafter arms; the block doubles after a fully accepted round and shrinks after a rejection.
  const bool lookup_on = !getenv("INK_NO_LOOKUP");
  auto tri_key = [](int32_t a, int32_t b2, int32_t c) {
    return (uint64_t(uint32_t(a)) << 40) ^ (uint64_t(uint32_t(b2)) << 20) ^ uint64_t(uint32_t(c));
  };
  // Up to 4 recent positions per trigram; the candidate whose preceding tokens agree with the current ones for
  // the longest stretch wins, and a match shorter than 6 tokens proposes nothing (a bare trigram such as ") {\n"
  // occurs all over a source file: on a copy task it was right only 63% of the time, the drafter 100%).
  auto lookup = [&](int32_t anchor, int kmax, std::vector<int32_t>& out) {
    const auto& tk = m.tokens();
    const size_t n = tk.size();
    if (s.tri_upto > n) {  // the cache was rewound or restored: rebuild
      s.tri.clear();
      s.tri_upto = 0;
    }
    for (size_t i = std::max<size_t>(s.tri_upto, 2); i < n; i++)
      if (tk[i] >= 0 && tk[i - 1] >= 0 && tk[i - 2] >= 0) {
        auto& slot = s.tri[tri_key(tk[i - 2], tk[i - 1], tk[i])];
        slot[3] = slot[2];
        slot[2] = slot[1];
        slot[1] = slot[0];
        slot[0] = int32_t(i + 1);
      }
    s.tri_upto = n;
    out.clear();
    s.lk_match = 0;
    if (n < 8) return;
    auto it = s.tri.find(tri_key(tk[n - 2], tk[n - 1], anchor));
    if (it == s.tri.end()) return;
    size_t best_p = 0;
    int best_len = 0;
    for (int32_t cand : it->second) {
      if (cand <= 0 || size_t(cand) >= n) continue;
      // tokens before the continuation at cand: tk[cand-1] == anchor, tk[cand-2] == tk[n-1], ...
      int len = 3;
      while (len < 32 && size_t(len) <= size_t(cand) - 1 && size_t(len) <= n &&
             tk[size_t(cand) - 1 - size_t(len)] == tk[n - size_t(len)] && tk[n - size_t(len)] >= 0)
        len++;
      if (len > best_len) {
        best_len = len;
        best_p = size_t(cand);
      }
    }
    if (best_len < 6) return;
    s.lk_match = best_len;
    for (size_t p2 = best_p; p2 < n && int(out.size()) < kmax; p2++) {
      if (tk[p2] < 0) break;
      out.push_back(tk[p2]);
    }
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
        // Up to 15 drafts (a 16-row forward: 8 rows cost 73 ms against 30 for one); INK_LOOKUP_MAX lowers it.
        static const int lk_max = getenv("INK_LOOKUP_MAX") ? std::max(1, std::min(31, atoi(getenv("INK_LOOKUP_MAX")))) : 15;
        lookup(b, std::min({s.lk_k, lk_max, max_tokens - emitted}), lk);
        double best = 0;
        for (int j = 0; j <= count_max; j++) best = std::max(best, s.arm_rate[j]);
        if (!lk.empty()) {
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
      // The proposal may use any temperature: a sharper q (INK_SPEC_QTEMP, fraction of the target's) moves
      // from 1 - TV(p, q) at equal temperature towards p(greedy draft) at 0; the best is in between.
      static const float qf = getenv("INK_SPEC_QTEMP") ? float(atof(getenv("INK_SPEC_QTEMP"))) : 0.5f;
      drafts = d.draft_sample(b, k, p.temperature * qf, us, &q_tok);
#endif
    } else {
      drafts = d.draft(b, k);
    }
    std::vector<float> mg;  // margins of the drafts kept (gate statistics)
    if (gate && !use_lookup && !sampled && d.margins().size() >= size_t(k)) {
      mg = d.margins();
      const int cut = gate_cut(mg, k);
      if (cut < k) s.gate_cut++;
      // Ask for the whole block again after a round that kept all it asked for, else for two more than were kept.
      // (At least 4: up to 4 rows are one pass over the head. Asking for 3, 4 or always 7 measured 60.3 / 60.3 /
      // 58.6 tok/s on the agent bench.)
      s.gate_k = cut == k ? 7 : std::max(4, cut + 2);
      if (cut == 0) {
        // Not even the first draft is worth a second row: a plain step, charged to the drafting mode.
        s.gate_zero++;
        ema(s.t_draft, ms_since(c0), 0.2);
        s.draft_ms += ms_since(c0);
        auto t0 = std::chrono::steady_clock::now();
        m.forward(&b, 1, false);
        int32_t t = sample_last();
        verify_seen(1, ms_since(t0));
        s.verify_ms += ms_since(t0);
        s.rounds++;
        ema(s.draft_rate, 1.0 / ms_since(c0), 0.15);
        s.draft_n++;
        ema(s.pm[bucket(mg[0])], t == drafts[0] ? 1.0 : 0.0, 0.03);
        bool more = on_token(t);
        emitted++;
        if (is_stop(t) || !more || emitted >= max_tokens) break;
        b = t;
        continue;
      }
      k = cut;
      drafts.resize(size_t(k));
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
    verify_seen(k + 1, std::chrono::duration<double, std::milli>(c2 - c1).count());
    s.vhist[std::min(32, k + 1)]++;
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
    if (!sampled) {
      // The target's token for every row at once, on all cores (a full-vocabulary sample is 0.3 ms, and a 16-row
      // verify would spend 5 ms on them one after the other). One fresh uniform per row keeps each an exact,
      // independent sample; rows after the first mismatch are simply not used.
      const size_t R = verify.size();
      std::vector<double> us(R, 0.0);
      if (p.temperature > 0.0f)
        for (auto& u : us) u = std::generate_canonical<double, 53>(s.rng);
      parallel_for(int64_t(R), [&](int64_t r0, int64_t r1) {
        static thread_local std::vector<float> pt;
        for (int64_t r = r0; r < r1; r++) {
          const float* row = lg.data() + r * V;
          if (p.temperature <= 0.0f) {
            y[size_t(r)] = argmax_first(row, V);
          } else {
            target_probs(p, row, V, pt);
            y[size_t(r)] = sample_from(pt, us[size_t(r)]);
          }
        }
      }, 1);
      while (a < k && drafts[size_t(a)] == y[size_t(a)]) a++;
    }
    if (getenv("INK_SPEC_DEBUG") && s.rounds < 1000) {
      fprintf(stderr, "round %ld%s: b=%d drafts", s.rounds, use_lookup ? " (lookup)" : "", b);
      for (int x : drafts) fprintf(stderr, " %d", x);
      fprintf(stderr, " | target");
      for (int x : y) fprintf(stderr, " %d", x);
      fprintf(stderr, " | a=%d\n", a);
    }
    if (!use_lookup && !sampled) {
      static FILE* dump = getenv("INK_SPEC_DUMP") ? fopen(getenv("INK_SPEC_DUMP"), "a") : nullptr;
      const auto& mg = d.margins();
      if (dump && mg.size() >= size_t(k)) {
        for (int j = 0; j < k; j++) fprintf(dump, "%d %d %.4f %d\n", k, j, mg[size_t(j)], j < a ? 1 : j == a ? 0 : -1);
        fflush(dump);
      }
    }
    if (use_lookup) {
      s.lk_rounds++;
      s.lk_drafted += k;
      s.lk_accepted += a;
      ema(s.lk_rate, (a + 1) / std::chrono::duration<double, std::milli>(c2 - c0).count(), 0.2);
      s.lk_n++;
      s.lk_k = a == k ? std::min(31, 2 * s.lk_k + 1) : std::max(3, (a + s.lk_k) / 2);
    } else {
      s.rounds++;
      s.drafted += k;
      s.accepted += a;
      for (int j = 0; j < k && j <= a; j++) ema(s.acc[j], j < a ? 1.0 : 0.0, 0.08);  // positions reached this round
      for (int j = 0; j < k && j <= a && size_t(j) < mg.size(); j++) ema(s.pm[bucket(mg[size_t(j)])], j < a ? 1.0 : 0.0, 0.03);
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
    fprintf(stderr, "gate: %ld rounds cut, %ld to a plain step; verify(T) = %.1f + %.1f (T-1) ms; P(accept) by margin "
                    "%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n", s.gate_cut, s.gate_zero, s.vc0, s.vc1,
            s.pm[0], s.pm[1], s.pm[2], s.pm[3], s.pm[4], s.pm[5], s.pm[6], s.pm[7], s.pm[8], s.pm[9], s.pm[10]);
  if (getenv("INK_SPEC_STATS")) {
    fprintf(stderr, "verify rows:");
    for (int t = 2; t <= 32; t++)
      if (s.vhist[t]) fprintf(stderr, " %d:%ld", t, s.vhist[t]);
    fprintf(stderr, "\n");
  }
  if (getenv("INK_SPEC_STATS"))
    fprintf(stderr, "lookup: %ld rounds, accepted %ld of %ld drafts, %.1f tok/s in lookup rounds\n", s.lk_rounds,
            s.lk_accepted, s.lk_drafted, 1e3 * s.lk_rate);
  if (getenv("INK_SPEC_STATS"))
    fprintf(stderr, "dspark adaptive: %ld plain steps, plain %.1f ms, draft %.1f ms, verify T=2..5 %.1f %.1f %.1f %.1f ms, "
                    "acceptance by position %.2f %.2f %.2f, tok/s by draft count %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f, "
                    "verify T=8 %.1f T=16 %.1f ms\n",
            s.plain_steps, s.t_plain, s.t_draft, s.t_verify[2], s.t_verify[3], s.t_verify[4], s.t_verify[5], s.acc[0],
            s.acc[1], s.acc[2], 1e3 * s.arm_rate[0], 1e3 * s.arm_rate[1], 1e3 * s.arm_rate[2], 1e3 * s.arm_rate[3],
            1e3 * s.arm_rate[4], 1e3 * s.arm_rate[5], 1e3 * s.arm_rate[6], 1e3 * s.arm_rate[7], s.t_verify[8],
            s.t_verify[16]);
  return true;
}

}  // namespace ink
