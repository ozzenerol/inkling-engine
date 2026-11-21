# Inkling engine: shared notes

Append-only log plus the agreed layout. Read before each chunk of work.

## Sessions

| agent | session name | role |
|---|---|---|
| A | inkling-engine-9e | model + accuracy (loader, forward pass, reference harness) |
| B | ozzenerol-5f | Metal kernels + speed (MoE, attention, prefill/decode, DSpark, memory) |
| C | ozzenerol-aa | server + agentic (HTTP :8091, template, tool calls, tokenizer, prefix cache, benches) |

## GPU lock

One model on the GPU at a time. Before loading weights: add a line here AND message the others.
Format: `LOCK <agent> <what> <start time>` / `UNLOCK <agent> <time>`.

**One line, always current. Replace it (do not append). Message the next holder when you release.**

No lock line any more: the GPU is used only by the test batch the integrator (A) runs on merged main. See tools/WORKFLOW.md.

Kernel micro-benches on synthetic buffers (op_diff, bench_*) do not need the lock but skew a running benchmark:
do not run them while C is benchmarking.

## Module layout and file ownership

```
CMakeLists.txt            A (top level; others add their targets via src/<dir>/CMakeLists.txt)
src/core/                 A   config.h/.cpp (parse config.json), safetensors.h/.cpp, tensor.h, log/timer utils
src/gpu/                  B   context.h/.mm (MTLDevice, queue, pipelines from source, buffers, residency set),
                              ops.h (op API below), ops_*.mm, kernels/*.metal (embedded as strings at build time)
src/model/                A   weights.h/.cpp (name -> Tensor binding), model.h/.cpp (forward pass, caches),
                              ops_ref.mm + kernels_ref/*.metal (simple correct kernels; the oracle B checks against)
src/engine/               A+B engine.h (API for C, below) owned by A; speculative.cpp (DSpark) owned by B
src/tokenizer/            C   tokenizer.json BPE + special tokens (encode/decode)
src/server/               C   HTTP/SSE, chat template, tool parser, reasoning, prefix cache policy
tools/                    shared: each script names its owner in its header
tests/ref/                A   python reference dumps from MLX-VLM + compare tool (greedy match, KL)
tests/server/             C
```

Build: one cmake project, targets `inkcore` (static lib: core+gpu+model+engine), `inkling` (CLI: generate/bench),
`inkling-server` (C), `inktest` (A: compare against reference dumps). Metal sources compiled at runtime with
`newLibraryWithSource`; cmake embeds `*.metal` into a generated .cpp.

## Model facts (from config.json + mlx_vlm/models/inkling2/language.py) — A

- hidden 2304, vocab 201024 (unpadded 200058, logits sliced), 24 layers, layer visits =
  0..5, (6..17)x3, 18..23 = 48 visits. Every visit has its OWN KV cache and conv state.
- Global (full attention) layers: 5, 11, 17, 23 -> global visits: 5, 11x3, 17x3, 23 = 8 visits. Others sliding window 1024.
- Attention: 24 q heads, 8 kv heads, head_dim 128 (both kinds). Extras per layer:
  - k,v go through a causal depthwise short conv (K=4) on the projected k/v; on global layers it is a *dynamic* conv
    (static_weight + per-token weight_proj from the input).
  - q = l2_clip(q * q_scale, 22.63); k = l2_clip(k, 22.63); v = rms_norm(v) no gamma.
  - relative-position bias: r = l2_clip(r_proj(x) * rel_scale, 8.0) [24 heads x 16]; bias[i,j] = r_i . rel_proj[:, dist]
    for dist < 1024 (rel_extent), 0 beyond (global), -inf beyond window (local). Mask built in **bf16**.
  - global layers: q and r scaled by tau = 1 + 0.1*log(max(pos/8192, 1)), pos = absolute position + 1.
  - attention sink: per-head per-query logit sink_proj(x), as an extra zero K/V column.
  - output * sigmoid(gate_proj(x)), then o_proj.
  - global layers take separate q/k/v inputs (3 attention-residual queries, 3 norms).
- Residual stream is "attention residuals": blocks of 6 layers; each sublayer input is a softmax-weighted mix of
  rms-normed block summaries + current partial (AttentionResidual). attn_sconv/mlp_sconv short convs on the outputs.
- MoE: sigmoid router fp32 + correction bias for selection, weights normalized * 8.0; 128 experts top-8,
  intermediate 1152; act = tanh(g/4)*4*sigmoid(g) * tanh(u/25)*25 (fp32). No shared expert.
  - layers 0-19: NVFP4 (group 16, e4m3 u8 scales, fp4 packed 8 per u32, per-expert fp32 weight_scale_2)
  - layers 20-23: MXFP8 (group 32, e8m0 u8 scales, fp8 e4m3 packed 4 per u32)
  - MLX GPU path: gather_qmm(..., global_scale = 2688 * weight_scale_2), output in bf16.
- Loop: middle core 6..17 run 3 passes; from pass 2 on, MessageControllers (rank-64 gates) blend "messages"
  (softmax over rms-normed core-block states of the previous pass, queries `message_queries[12,4,2304]`) into
  attn/k/v/mlp inputs. Layer state is reset to the prelude output at each pass start.
- Final: final_input attention-residual, norm, lm_head in fp32 (bf16 weight promoted), softcap tanh(x/30)*30.
- Activations bf16 between ops; many internals fp32 (see language.py). Exact dtype placement matters for greedy match.

## Interfaces (proposal by A — reply with objections)

### Tensor (src/core/tensor.h, A)
```cpp
enum class DType : uint8_t { F32, BF16, F16, U8, U32, I32 };
struct Tensor {            // non-owning view into an MTL::Buffer (or host memory for tests)
  void* buf = nullptr;     // id<MTLBuffer> bridged / MTL::Buffer*
  size_t offset = 0;       // bytes
  DType dtype;
  int ndim; int64_t shape[4];   // row-major contiguous unless noted
  size_t nbytes() const; int64_t numel() const;
};
```

### GPU context + ops (src/gpu/, B owns; A writes ops_ref with the same signatures first)
```cpp
class Context {            // one per process
 public:
  static Context& get();
  Tensor alloc(DType, std::initializer_list<int64_t>, const char* tag);   // private/shared storage, B decides
  Tensor wrap_weights(...);                     // resident weights (memory plan: B)
  void begin();  void commit_and_wait();        // batching dispatches into command buffers
};
namespace ops {   // all enqueue on Context's current command buffer; T = tokens in the batch
  void embed(const Tensor& table_bf16, const Tensor& ids_i32, Tensor& out_bf16);
  void rms_norm(const Tensor& x, const Tensor* gamma, float eps, Tensor& out);          // fp32 math
  void linear(const Tensor& x, const Tensor& w_bf16 /*[N,K]*/, Tensor& out);            // out dtype selects fp32/bf16 result
  void router(const Tensor& x, const Tensor& gate_w, const Tensor& bias, int topk, float scale,
              Tensor& idx_i32 /*[T,8]*/, Tensor& w_f32 /*[T,8]*/);
  void moe(const Tensor& x_bf16, const Tensor& idx, const Tensor& w, const MoeLayer& L, Tensor& out_bf16);
       // gate/up gather-qmm + softcapped swiglu + down + weighted combine, NVFP4 or MXFP8 per L.mode
  void attention(const AttnArgs&, Tensor& out);   // q,k_cache,v_cache,rel bias inputs, sink, window; B's design
  // + small fused elementwise ops added as needed (sconv, l2_clip, gates, attention residual mix)
}
```
Rule: B may change these signatures; ops_ref (A) follows. Every fast op gets a unit test vs ops_ref.

### Engine API for C (src/engine/engine.h, A owns header)
```cpp
struct EngineOptions { std::string model_dir; int max_context = 131072; bool use_mtp = true; int draft_block = 4; };
struct SamplingParams { float temperature = 0; float top_p = 1; int top_k = 0; uint64_t seed = 0; };
class Engine {
 public:
  explicit Engine(const EngineOptions&);
  // One sequence (single-user server). Cache = KV + conv states for all 48 visits.
  int64_t cached_len() const;                     // tokens currently in the cache
  bool rewind(int64_t n);                         // keep the first n tokens; false if not possible
  // Feed tokens after the cached ones (prefill, chunked internally). Returns logits of the last token (host, fp32).
  void prefill(std::span<const int32_t> tokens, std::function<void(int64_t done)> progress = {});
  // Generate until stop; calls on_token per accepted token (speculative decoding hidden inside).
  void generate(const SamplingParams&, int max_tokens, const std::vector<int32_t>& stop_ids,
                std::function<bool(int32_t tok)> on_token /*return false to stop*/);
  std::vector<float> last_logits() const;         // for tests (A)
};
```
Prefix cache (C's policy, A/B implement mechanism): server computes common prefix with the cached tokens and calls
`rewind(common)` then `prefill(rest)`. Sliding-window and conv states make arbitrary rewind hard; plan: keep the
local-layer KV un-ringed (full length, costs 40 visits x 4 KB/token... ~160 KB/token, too much at 128k) — OPEN
QUESTION for B: default = ring of window+256 plus conv-state snapshots at chunk boundaries so rewind works within
256 tokens of the end and at saved snapshots (turn boundaries); otherwise re-prefill from the last snapshot.

## Log
- 2026-11-21 A: roles assigned (B=ozzenerol-5f, C=ozzenerol-aa). C owns src/tokenizer/. Layout + interfaces above.

## B: measurements (2026-11-21, from safetensors headers; tools/bytes_per_token.py, no GPU)

Bytes read per decoded token (weights; experts counted as 8/128 of each layer, loop core x3, vision and mtp excluded):

| part | stored GB | read GB/token |
|---|---|---|
| attention + dense per layer (bf16: q 14.2, gate 14.2, o 14.2, k 4.7, v 4.7, r 1.8 MB, router 0.6 MB per visit) | 1.37 | 2.79 |
| experts L0-19 NVFP4 | 11.47 | 1.58 |
| experts L20-23 MXFP8 | 4.20 | 0.26 |
| lm_head bf16 (201024 x 2304) | 0.93 | 0.93 |
| embed (1 row) | 0.93 | ~0 |
| total text weights | 18.9 (+0.64 mtp, +0.64 vision) | **5.55** |

KV read per decoded token at context T: global 8 visits x 8 kv heads x 128 x (k+v) x 2 B = **32 KiB x T**,
local 40 visits x 4 KiB x min(T,1024) = 164 MB max. Ceiling at 366 GB/s (weights + KV only):

| context | 0 | 9.5k | 18.9k | 38.6k | 78.1k | 127.5k |
|---|---|---|---|---|---|---|
| GB/token (bf16 KV) | 5.55 | 6.03 | 6.34 | 6.99 | 8.28 | 9.90 |
| ceiling tok/s, bf16 KV | 65.9 | 60.7 | 57.7 | 52.4 | 44.2 | 37.0 |
| ceiling tok/s, 8-bit global KV | 65.9 | 62.2 | 60.5 | 57.3 | 51.7 | 46.1 |
| MLX-VLM baseline (with DSpark) | | 24.8 | 23.3 | 19.9 | 16.5 | 13.8 |

Notes:
- Half the weight bytes are the bf16 attention projections (read 48x per token). They stay bf16 (no requantization),
  so the only lever there is DSpark: a verify of B tokens reads the dense weights once.
- MLX keeps an fp32 copy of lm_head (1.85 GB/token read). We read the bf16 weight and accumulate in fp32: same math
  (bf16->fp32 is exact), half the bytes.
- 8-bit KV would change numerics; only behind a flag, never default, and only if greedy match holds.

## B: plan and interface replies

- Memory plan: weights pread into our own MTLBuffers (shared storage, NOT no-copy mmap views) + one MTLResidencySet
  on the queue (saetta measured the residency drop after idle). ~19.5 GB resident; vision tensors not loaded.
  Global KV bf16 grows in steps (saetta: Metal makes a whole buffer resident on first use), 4.2 GB at 128k.
  Budget at 128k: weights 19.5 + KV 4.5 + 4 checkpoints 0.7 + prefill scratch ~1.5 = ~26 GB of 44 GB wired.
- KV cache (src/gpu/kv.h, B): per visit {k,v} bf16 [kv_heads][cap][128]. Global: linear, cap grows. Local: ring of
  1024 + 512 slack slots, so a rollback of <= 512 tokens works. Conv states: 4 per visit (k, v, attn, mlp), fp32
  [3][C]; for rollback a small ring of the last 512+3 conv inputs per conv... too big for attn/mlp (2304 ch x 48 visits
  x 4 B = 0.9 MB/token) -> instead conv states are checkpointed per spec step (block <= 8) and at checkpoints.
  API (agreed with C):
  ```cpp
  struct Ckpt { int id; int64_t len; };
  int64_t len() const;
  bool truncate(int64_t n);          // global KV always; local ring if len-n <= 512; conv states only if
                                     // n == len or n is a recorded conv snapshot (spec step / checkpoint)
  Ckpt checkpoint();                 // blit local windows + conv states (~172 MB), ~1 ms
  void restore(int id);              // blit back + set global len; then caller prefills the rest
  void drop(int id);  int64_t ckpt_len(int id) const;   // up to 4 kept (C checkpoints end-of-prompt and end-of-gen)
  ```
  So Engine::rewind(n) = truncate(n) || (restore(best ckpt <= n), then prefill tokens [ckpt.len, n)).
  A's open question: answered with the above (C's turn-boundary checkpoints cover the common case).
- Ops API: OK in general. Changes:
  1. All ops encode into one command buffer per forward (decode token or prefill chunk); Context gives a compute
     encoder reused across dispatches; no commit per op.
  2. `linear` gets a multi-output form for decode: `linear_multi(x, {w1,w2,..}, {o1,o2,..})` (same input x: local
     layers q/k/v/r/gate/sink), per-row dot order identical to the single form so results are bit-equal.
  3. `moe` handles T=1 (decode matvec), T<=16 (spec verify, expert-union), T large (prefill: sort routes by
     expert, gather-GEMM). Signature stays.
  4. `attention(const AttnArgs&)`: fused kernel computing scores, bias r.rel_proj[:,dist] (rounded to bf16 like the
     reference mask), window/causal mask, sink column, softmax fp32, PV. AttnArgs = {q bf16 [T,24,128], r bf16
     [T,24,16], rel_proj, sink bf16 [T,24], KvCache visit, q_pos0, window (0=global), out bf16 [T,24*128]}.
     Global q/r tau scaling stays a separate elementwise op (A's math), or folded in later if bit-equal.
  5. For DSpark (speculative.cpp, B) the model needs: forward(tokens[T], capture visits {5,23,41,48} -> hidden rows,
     logits for all T rows) and the KV truncate above. A: please keep `Model::forward(ids, T, captures*, logits_all)`.
- Order of work (B): context + buffers + resident loader -> bf16 matvec at 366 GB/s -> NVFP4/MXFP8 MoE matvec ->
  attention decode -> decode path end to end vs A's ref -> prefill GEMMs -> DSpark -> long context.

## C: server + agentic (ozzenerol-aa)

Files: src/tokenizer/ (inktok), src/server/ (inkserver, inkling-server), tests/server/. Build:
`cmake -S . -B build-c && cmake --build build-c --target inkling-server` (links the real engine only once
src/engine/engine.cpp exists; `-DINK_SERVER_ENGINE=OFF` forces stub-only).

- **Chat template**: hand translation of chat_template.jinja (src/server/chat_template.cpp), with a Python-exact
  json.dumps (src/server/json.cpp: key order kept, float repr, ensure_ascii=False, sort_keys). Checked against
  transformers apply_chat_template: 3011/3011 random+edge cases identical incl. the 159 that must raise
  (tests/server/template_check.py). Tool-call `arguments` strings are json-parsed first (invalid -> {}), as
  mlx_vlm request_normalization does.
- **Tokenizer**: tokenizer.json BPE, pcre2 (UTF+UCP, JIT) for the split regex, ignore_merges, specials split first
  (leftmost-longest). Matches HF tokenizers on 3119 texts / 454k tokens and A's 4 ref prompts
  (tests/server/tok_check.py). Load 250 ms, encode ~5M tok/s (one run).
- **Output**: parsed at token-id level (src/server/output_parser.cpp): content_thinking -> reasoning_content,
  content_text -> content, content_invoke_tool_json -> tool_calls (inkling.py port: name from JSON, args from
  "args"|"arguments", arguments string = compact JSON in the model's key order). Invalid tool JSON is returned as
  content. Stops: <|content_model_end_sampling|> (200006), and <|message_user|>/<|message_system|>/<|message_tool|>/
  <|endoftext|>.
- **Effort**: chat_template_kwargs.reasoning_effort > body.reasoning_effort > server --effort; if none of those and
  chat_template_kwargs.enable_thinking == false (qh, saetta bench.py) -> "none". Otherwise the template default 0.9.
- **Prefix cache** (src/server/prompt_cache.cpp): LCP of new prompt vs cached ids; then nothing / Engine::rewind /
  restore best checkpoint <= LCP / reset, then prefill the rest. Checkpoints at end of prompt and end of each
  finished turn, 4 kept (LRU). **Turn splicing (decision)**: when a request contains an assistant message whose
  (trimmed reasoning, trimmed content, tool calls with canonical args) equals a turn this server generated, that
  span is fed as the exact generated token ids instead of the re-rendered text. Without it every agent turn
  diverges inside the previous answer (qh trims reasoning; the template re-serializes tool args compact+sorted)
  and re-prefills all its thinking. The model sees exactly what it wrote. `--no-splice` restores pure template text.
- **API**: /v1/chat/completions (stream + non-stream, usage with prompt_tokens_details.cached_tokens, plus a
  llama.cpp-style `timings` object), /v1/models (id = model dir path, which qh picks), /health, /tokenize, /v1/cache.
  SSE `: prefill n/total` comments every 2 s during long prefills. One generation at a time (mutex).
  Not done: `stop` strings, n>1, logprobs, images (no vision tower), tool_choice (ignored).
- **Tests**: tests/server/api_test.py runs the server with `--stub` (scripted engine with the real cache contract:
  slack 512, checkpoint prefixes checked) — 30/30 pass: streaming, tool calls, 3-turn agent loop (turn 2 prefills
  13 tokens, turn 3 8 tokens), divergence + checkpoint restore, disconnect, length, errors.

- 2026-11-21 C: **MLX-VLM baseline, same day** (serve_inkling.sh: DSpark block 4, prefill step 2048), one run each:
  - qh e2e (/tmp/inkling-qh-test, tools/qh_e2e.sh mlx 2): 2/2 PASS (5/5 tests), 57 s and 40 s wall, 4-5 steps;
    qh reports ~23 s of each run "reading the prompt" on a 3.6-3.9k context: MLX re-prefills every turn.
  - ctxbench (temp 0.7, effort none, 400 tok): 1.2k 565/23.1, 9.5k 602/24.2, 38.6k 495/22.0 (prefill/gen tok/s),
    matching BRIEF's table.
  - saetta bench.py multi-turn (6 turns, 128 tok): cached_tokens 0 on every turn; TTFT 11.6, 15.5, 22.0, 27.9 s at
    5.4k/8.3k/11.2k/14.1k prompt; turn 5 spent all 128 tokens thinking (effort none still thinks), so no content.
  - bench.py's context sweep fails on Inkling: its 2-token calibration run gets no content (the first two tokens are
    <|message_model|><|content_text|>). Use ctxbench for the sweep and bench.py --contexts "" for multi-turn.

## B: first end-to-end Metal results (2026-11-21, inktest, one run each)

| prompt | greedy vs MLX | KL mean / max | decode tok/s (63 steps, ctx < 3k) |
|---|---|---|---|
| p0 (8 steps) | 8/8 | 5.5e-5 / 3.6e-4 | 34.9 |
| p1 | 62/64 (first 38) | 1.6e-2 / 0.69 | 32.0 |
| p2 | 62/64 (first 6) | 7.7e-4 / 9.8e-3 | 31.8 |
| p3 (2886-token prompt) | 61/64 (first 12) | 1.2e-2 / 0.16 | 27.8 |

CPU reference on p0: 8/8, KL mean 1.0e-3. The p1-p3 mismatches match A's known loop-pass gap (A is on it).
Decode went 11 -> 33 tok/s after making router and controller parallel (they were one threadgroup per token).
Profile tool: `INK_PROFILE=1` on any binary using MetalBackend prints GPU ms per kernel (each dispatch waited
separately, so totals are inflated; use for ranking). Known slow spots, next: attention prefill (0.9-1.8 TFLOPS),
skinny GEMM T=5..16 (110 GB/s), dispatch count (~1500/token), decode attention at long context (180 GB/s).

## A: model + accuracy status (2026-11-21)

Files (A): src/core/* (json, safetensors index, config, thread pool, bf16/fp8/fp4 helpers), src/model/backend.h
(op interface, exact rounding points per op), src/model/cpu_backend.cpp (CPU oracle), src/model/model.{h,cpp}
(forward over 48 visits, loop bank/messages, attention residuals, KV rings, conv history + rollback,
debug hooks), src/engine/engine.{h,cpp} (prefill chunking, sampling, rewind, checkpoints, speculative hook),
tests/ref/* (MLX dump, inktest, compare, single-visit MLX recompute).

Accuracy tooling:
- `tests/ref/dump_ref.py OUT [N] [--fp32] [--prompts ..] [--gen-from DIR]`: MLX greedy dump (ids, gen, logits per
  step; p0 also hidden_v). --fp32 = all bf16 params/activations in fp32 (experts stay NVFP4/MXFP8): higher-precision
  reference. --gen-from forces another dump's tokens so steps line up.
- `build/inktest --backend cpu|metal --ids ref_data/pN.ids --gen ref_data/pN.gen --steps 64 --out DIR`:
  teacher-forced run (feeds the ref greedy tokens), writes logits.f32 / argmax.i32 (--hidden: per-visit inputs).
- `tests/ref/compare.py ref_data/pN.npz DIR|other.npz`: greedy match, KL(ref||ours), every mismatch with the
  ref's top-2 gap (gap < ~0.1 = rounding flip; larger = suspicious).
- `inktest --isolate ref_data/p0.h [--debug-visit V]` + `tests/ref/debug_visit.py`: each visit fed MLX's own
  inputs; with --debug-visit, named intermediates (mixed, msg, ctrl0, xq/xk/xv, q_raw.., q,k,v,r, sink, attn, o,
  hid_mix, hn, moe, res, out) compared against MLX recomputing the same visit.

Findings (CPU backend vs MLX-bf16 GPU reference, one run each):
- p0 (25-token prompt, 8 steps): greedy 8/8, KL mean 1.0e-3 / max 5.8e-3.
- Isolation: prelude + pass-1 visits at bf16 noise (rel 1.0-2.0e-3 per visit). Loop passes 2-3 and layer 12 show
  larger per-visit error (up to 5e-2), but float64 checks of the same inputs show controller (7e-5) and attention
  (1.6e-3 = output rounding) are exact; the error is bf16-rounding noise amplified by the rank-64 controller
  bottleneck (a 1-ulp flip of h moves the gates) and sharp heads. MLX's own CPU vs GPU differ by the same order.
  => no model-math bug found.
- MLX rounding rules confirmed by test: rms_norm on bf16 input rounds twice (bf16(w * bf16(x*inv))), fp32 input
  none; NVFP4 dequant = e2m1 * e4m3 * weight_scale_2 with the global applied to the accumulator (B's note).
- Metal (B's runs, scored by A): p1 mismatches only after the forced stop token (degenerate region), p2 near-ties
  (gaps 0.013/0.11); p3 (2886 tokens) KL mean 1.2e-2 with two non-tie mismatches -> under investigation:
  CPU p3 + fp32 reference (MLX steel prefill attention rounds P to bf16; long prompts may make MLX-bf16 itself
  the less accurate side).
- Open: Engine::rewind only works at n == len, 0, or inside the last spec-verify rows (conv states can't rewind);
  cross-turn reuse goes through checkpoint()/restore(), which C uses.

- 2026-11-21 C: **first inkling-server run on the real engine** (Metal, commit 4429d37 kernels + server 0f..; no DSpark yet),
  one run each:
  - Smoke: reasoning_content / content / tool_calls all correct (effort low: "Paris"; bash tool call
    {"command":"ls -la"}). Decode 30-32 tok/s at short context (MLX same day 23-24).
  - Cross-turn cache works on the engine's checkpoints: 3-turn tool loop prefilled 10 tokens on turns 2-3, TTFT
    0.14-0.15 s. Spliced turns hit.
  - qh e2e: 4/4 PASS over two sessions (83, 33, 39, 40 s; MLX 57, 40 s). In-run turns hit the cache (e.g. 4626-token
    prompt with 4590 reused, 36 prefilled). One run in the first session reset twice at ~5.1k (22 s lost); not
    reproduced in the second session, which has divergence logging — watch for `[cache] diverged`.
  - ctxbench (temp 0.7, 400 tok): 1.2k 462/27.9, 9.5k 436/27.0, 38.6k 260/23.4 (prefill/gen tok/s). vs MLX
    565/23.1, 602/24.2, 495/22.0: gen 1.21x/1.12x/1.06x (+6-21%, corrected), prefill 0.82x/0.72x/0.53x (B: long-context attention prefill still slow).
  - Our prompts are 2 tokens shorter than MLX's in ctxbench (1219 vs 1221): MLX-VLM ignores
    chat_template_kwargs.reasoning_effort, so its baseline ran at the template default 0.9 ("0.9" = 3 tokens vs "0").
  - Gotcha: building from the shared tree embeds others' uncommitted .metal edits; for runs I build HEAD in a
    git worktree.

## A: fp32 reference verdict (2026-11-21, one run each)

`dump_ref.py ref_data/fp32 64 --fp32 --gen-from ref_data` = MLX with every bf16 param/activation in fp32 (experts
unchanged), same forced tokens. Scored up to the stop token (argmax agreement with fp32 argmax, KL(fp32||x)):

| prompt | MLX-bf16 | Metal (B) | CPU (A) |
|---|---|---|---|
| p0 (8 steps) | 8/8, 7.0e-4 | 8/8, 1.0e-3 | 8/8, 1.7e-4 |
| p1 (37 steps) | 37/37, 6.5e-4 | 37/37, 7.1e-4 | - |
| p2 (64) | 63/64, 8.5e-4 | 63/64, 6.8e-4 | - |
| p3 (64, 2886-token prompt) | 60/64, 8.2e-3 | 61/64, 7.8e-3 | KL 5.0e-3 |

Verdict: the engine is as accurate as MLX-bf16 against the higher-precision reference on all four prompts; the
p3 "mismatches" vs MLX-bf16 are MLX's own bf16 rounding (MLX-bf16 itself only gets 60/64 vs fp32). The CPU
oracle is the most accurate; Metal has some headroom on long prompts (fp32-accumulated pieces could close it).
Going forward, speed changes are checked against BOTH: greedy == MLX-bf16 up to near-ties, and KL vs fp32 not
worse than MLX-bf16's.
- 2026-11-21 C: **bench tooling for the 2x goal** (A's request):
  - `tools/bench_c.sh inkling|mlx` = ctxbench at 1.2k..127.5k (SIZES) + `tools/agent_bench.py` (qh's real system
    prompt + 10 tools from tests/server/fixtures/qh_request.json, ~800-token scripted tool result per turn, 10 turns,
    temp 0, thinking on as qh sends it; base context 0 and 30000). `tools/bench_c.sh report` -> tools/bench_report.py:
    BRIEF table side by side with x-ratios vs MLX + agent tables (total wall, per-turn TTFT, prompt/cached).
    `BIN=` lets the run use a binary built from HEAD in a worktree. MLX at 104000 needs PREFILL=512 (OOM at 2048).
  - **Server overhead per token** (stub engine at 0 ms/token, 20k-token replies, one run): streaming 4.4-4.8 us/token
    (SSE write per token), non-streaming 0.14-0.19 us; independent of a 30k prompt. At 90 tok/s that is <0.05% of
    a token. Render + tokenize of a 143k-token prompt: 42 ms per request. Now reported live as
    `server_us_per_token` in timings and the [req] log line.

## B: decode speed work, DSpark, accuracy (2026-11-21 evening, one run each unless noted)

- **DSpark works**: tools/spec_check.sh IDS N BLOCK (plain vs dspark in separate processes, greedy, stop ids).
  Token streams IDENTICAL on p2 and on a 58-token coding prompt. Coding prompt, 400 tokens: plain 33.3 tok/s,
  DSpark block 4 47.2 tok/s, acceptance 1.80 drafts/round (MLX-VLM DSpark on the same prompt: 1.53). Drafts are
  token-identical to MLX's drafter for the first 16 rounds of p2 (then the target text diverges). Per round: draft
  8.6 ms, verify (T=4) 50 ms. Verify = decode + ~3 x 6.7 ms of extra MoE (real consecutive tokens share few experts).
  Temperature > 0: sample-and-compare walk (as mlx_vlm), exact target samples.
- Fixed: the weak speculative stubs in engine.cpp shadowed speculative.cpp at link time (DSpark was silently off).
- **Determinism**: attention splits are fixed-size and position-aligned (local 64 keys, global 512), so verify
  rows == decode bit for bit (op_diff "verify==decode"). Ring kv_append raced for chunks longer than the ring
  (2048 > 1536): p3 logits were non-deterministic; fixed (only the last cap rows are written).
- Decode 31.8 -> 36.7 tok/s (p2-like short context): 1024-thread mix for T<=4, wider router matvec, concurrent
  encoder with own hazard barriers (weights read-only), fused attn_prep_local / attention_gated / conv_add
  (CPU p0 logits byte-identical before/after, cmp). Ablation (INK_SKIP=class, timing only) per token at ~29 ms:
  moe 6.8, linear 9.4, attn 2.6, logits 2.1, mix 3.6->1.7, small 3.5, ctrl 2.2, router 1.8 ms. GPU span == wall,
  so the remaining overhead is dependent-dispatch stalls (~1200 barriers / 1600 dispatches per token).
- acc_check (metal, after the above, 64 steps): p0 64/64 fp32-KL 1.7e-3; p1 63/64 1.7e-3; p2 63/64 8.3e-4;
  p3 62/64 fp32-KL 1.18e-2 with A's attention_gemm, **4.8e-3 with the flash prefill attention**
  (INK_AG_MIN_T=1000000), i.e. the GEMM attention costs accuracy on long prompts (MLX-bf16: 8.2e-3).
- 2026-11-21 C: **caveat on my first real-engine numbers** (17:56 run): per B, that build had DSpark shadowed by
  weak stubs (plain decode only) and a kv_append race for prefill chunks > the 1536-slot local ring, so prompts
  > 2048 tokens ran on a partly wrong local KV. Treat those ctxbench/qh numbers as superseded; rerun on >= 5f0fe21.

## A: prefill attention as matrix products (2026-11-21)

`src/gpu/kernels/attn_gemm.metal` + `attention_gemm()` in metal_backend.mm, used for T >= 64 (INK_AG_MIN_T).
Port of saetta's fused attention_gemm: per 1024-key block, `ag_scores` (S = Q K^T in 64x64 bf16 tiles, fp32
accumulators, + bias from a transposed rel_proj table, window mask, exp(s - tile max) stored fp32) and `ag_pv`
(online softmax over tile maxes, O = a O + P V). Sink = initial m, l = 1.
bench_attn T=2048, one run each (TFLOPS): flash 0.74-0.82 | P as one half tile 7.6-10.0 | fp32 tiles 1.3-1.5
(fp32 simdgroup matrices are slow whatever the threadgroup memory) | **P as hi + lo half tiles 5.5 / 6.9 / 6.6
(len 0 / 32k / 127k), local 4.5** = default. op_diff vs CPU: 99.93-99.99% bit-exact (flash: same level; one half
tile: 91%).
Profile of a 2886-token prefill after this (INK_PROFILE, GPU ms): moe_gemm_nvfp4 1947 (50%), gemm_bf16 827 (21%),
ctrl_out 280 (7%), moe_gemm_mxfp8 175, ag_pv 168 + ag_scores 131 (8%), short_conv4 153. Short-context prefill is
now MoE GEMM (about 8.3 TFLOPS already) + dense GEMM; ctrl_out and short_conv4 look cheap to fix (B).
Accuracy on p3 (one prompt, 64 forced steps) is NOT a reliable ranking: KL mean between any two implementations
of equal op-level accuracy is 5e-3..1.1e-2 (flash vs CPU 5.7e-3, MLX-bf16 vs flash 6.5e-3, GEMM vs flash 1.1e-2,
GEMM vs fp32 1.0e-2 at chunk 2048 / 1.2e-2 at 512, flash vs fp32 4.8e-3, MLX-bf16 vs fp32 8.2e-3). At step 0
(straight after prefill) GEMM is the closest to fp32 (KL 1.5e-2 / 9.3e-3 vs flash 1.9e-2, CPU 1.5e-2, MLX 1.8e-2).
TODO (A): a set of ~8 long prompts with fp32 references so long-context accuracy is a distribution, not one draw.
p0-p2 with GEMM attention: greedy unchanged (64/64, 63/64 after-stop, 63/64 near-tie), KL vs fp32 1.7e-3 /
1.7e-3 / 8.3e-4.

## A: 8-bit build (2026-11-21, user request)

`~/llm/mlx-models/Inkling-2-Mini-MLX-8bit` = pipenetwork/Inkling-2-Mini-MLX-8bit (18.1 GB, mlx-lm PR 2086 names).
Experts unchanged (NVFP4 / MXFP8, gate+up fused and row-interleaved in `gate_up_proj`, `global_scale`); every
dense linear + embed + lm_head is MLX affine 8-bit (U32 codes [N, K/4], bf16 scales/biases [N, K/64], group 64);
router weight and controller forget/gain are f32. No mtp.safetensors (use the 4-8bit dir's), config identical.
Stage 1 (A, in Model::load): converted at load to the 4-8bit in-memory form, so all kernels run unchanged.
Stage 2 (B): q8 kernels reading the codes in place; that is the speed gain (dense bytes per token 2.79 -> ~1.5 GB,
head 0.93 -> 0.48). Check script: tests/ref/check_8bit_layout.py (tensor-by-tensor against the 4-8bit build).

## A: evening work (2026-11-21, code written while C benchmarked; GPU checks queued in tools/a_gpu_queue.sh)

- **Speculative sampling** (src/engine/sampling.{h,cpp}): `target_probs` (full-vocab softmax with min_p / top_k /
  top_p, 0.3 ms), `spec_accept` (accept draft x ~ q with min(1, p/q), residual resample on rejection; exact).
  Test (build/sample_test): emitted distribution = target within 2 sigma, acceptance 0.6406 vs 1 - TV 0.6404;
  greedy-compare (what MLX does) would accept 0.10 on the same pair. speculative.cpp uses it for temperature > 0
  with B's `draft_sample` / `q_row` (INK_SPEC_COMPARE=1 = old sample-and-compare).
- **Adaptive draft count**: each round picks k in 0..block-1 (0 = plain step) by expected tokens/ms from EMAs of
  per-position acceptance and measured plain / draft / verify(T) times. Reason: C's ctxbench at temp 0.7 gave gen
  30.3 tok/s at 1.2k while plain decode is ~36.7: fixed block 4 with compare-acceptance was slower than no drafts.
  INK_SPEC_FIXED=1 restores the fixed block.
- **Chain kernels** (src/gpu/kernels/chain.metal, backend.h ChainFrontArgs / ChainMidArgs): for T <= 8,
  `chain_front` = [previous visit's x = res + conv(moe)] + mix + (loop passes: 3 controllers) + norm(s);
  `chain_mid` = conv(+x) + block copy + mix + (mlp controller) + post norm + router top-8, each ONE 1024-thread
  dispatch. Loop messages are not materialized on chained visits (m = c0 bank0 + c1 bank1 on the fly). Expected:
  local visit 14 -> 9 dispatches, looped local visit ~25 -> ~10, at B's ~7.5 us per dependent dispatch.
  Model: run_visit takes ChainMsg*; on_visit fires inside run_visit after the front; CPU path byte-identical (cmp).
  INK_NO_CHAIN=1 disables.
- **Rollback test**: `inktest --rollback-test` (4 rows with history, keep 2, continue) is bit-identical to the
  straight run on the CPU backend with INK_CPU_ROWEXACT=1 (plain CPU differs: BLAS vs loops by batch size).
- **kernel_check**: `build/kernel_check [names]` compiles the Metal library and builds pipelines with no model and
  no GPU work. Run it before committing .metal changes (HEAD was uncompilable for ~20 min tonight).
- **8-bit build**: loader + check script done (see "A: 8-bit build"); CPU p0 greedy 16/16 vs the lossless
  reference. B wired q8 kernels behind INK_Q8_KERNELS=1 (MLX qmv semantics s*sum(x q) + b*sum(x), not
  bf16-rounded weights), to be validated.

## A: GPU session results (2026-11-21, 19:25-19:45, one run each)

- **Chain kernels: slower, now opt-in (INK_CHAIN=1).** Decode 28.6 vs 36.9 tok/s; chain_front 198 us and chain_mid
  117 us per call (stage timing: conv 8, mix 12, controller ~140, router ~40 us). One 1024-thread threadgroup has
  the throughput of one slice of the GPU; a scalar device load costs ~0.25 us of latency. Matvecs must stay
  multi-threadgroup. Lesson for fusion: fuse only ops whose per-thread work is a handful of loads.
- **DSpark at temperature 0.7 (p2, 300 tokens)**: plain 36.0 tok/s; fixed block 4 + compare 43.4; fixed + real
  speculative sampling 40.3 (accepted 1.31 vs 1.47 per round: q at the target's temperature is flatter than p, so
  1 - TV(p, q) < p(greedy draft)); bandit over draft counts + compare 42.8 = default. Costs: plain 27.8 ms, verify
  T=2/3/4 33.9/41.5/47.5 ms, draft 8.0 ms. Greedy: token streams identical, 39.5 vs 36.6 tok/s.
  To do for speculative sampling: a sharper proposal (draft temperature below the target's), tuned online.
- **8-bit build on Metal** (acc_check p0-p2, 64 steps): load-time conversion 36.7 tok/s, greedy 62/64, 63/64,
  63/64, KL vs fp32 1.4e-3 / 4.9e-3 / 1.5e-3; B's q8 kernels (INK_Q8_KERNELS=1) **42.7 tok/s plain**, greedy
  63/64 x3 (near-ties and after-stop), KL vs fp32 1.4e-3 / 6.0e-3 / 2.2e-3.
- **Long-prompt accuracy set** (8 prompts of 573..6224 tokens cut from source files + a question, 32 forced steps,
  mean KL vs the fp32-activation MLX reference; `tests/ref/acc_summary.py`):

  | | MLX-bf16 | ours, GEMM attention | ours, flash attention | 8-bit build (q8 kernels) |
  |---|---|---|---|---|
  | mean | 3.04e-3 | 3.99e-3 | 3.60e-3 | 3.19e-3 |
  | median | 2.68e-3 | 3.89e-3 | 3.90e-3 | 2.96e-3 |
  | max | 6.43e-3 | 5.69e-3 | 5.08e-3 | 5.79e-3 |
  | argmax = fp32 argmax | 253/256 | 249/256 | 252/256 | 254/256 |

  Per prompt each implementation is above the others on some prompts and below on others (factor 2 both ways),
  so all four are in one accuracy class; GEMM vs flash attention shows no systematic difference (same median),
  and the 8-bit build is not measurably worse than the lossless one on this set. Prefill of the 6224-token prompt:
  8.5 s (731 tok/s).

## Batches (A, integrator; logs in bench/batch-N/)

- **Batch 1** (main d1ac5a8, 19:40): gates pass except op_diff (3 controller cases: test bounds, fixed by B in
  b/work). Lookup drafts v1 lost: the "copy" prompt spent all 400 tokens thinking (no copying), and any forward
  above 4 rows costs 61-126 ms (T=4: 47-53 ms).
- **Batch 2** (main fa90053, 19:46): copy prompt now starts inside the answer's code block (ref_data/copy.ids,
  2282 tokens; edit.ids). Greedy, 600 tokens, one run each:

  | | plain | DSpark block 4 | + lookup max 3 | + lookup max 7 / 15 |
  |---|---|---|---|---|
  | copy | 33.9 tok/s | **62.0** (acceptance 1.00 / 1.00 / 1.00) | 53.2 (391 of 625 accepted) | 33.0 / 33.4 |
  | edit | 33.4 | **54.7** (0.98 / 0.95 / 0.92) | 52.1 | 55.2 / 50.7 |

  So on copy-heavy output the bundled drafter is already right every time and the BLOCK SIZE is the limit;
  bare-trigram lookup is less precise than the drafter (-> v2: best of 4 positions by backward match, >= 6
  tokens). Forward cost at 2.3k context: T=1 30.0 ms, T=2 35.5, T=3 46.7, T=4 52.6, T>=5 122-126 ms (D's task).
  Server (C, lossless build): ctxbench gen 34.6 / 33.0 tok/s at 1.2k / 9.5k (1.50x / 1.36x MLX), prefill 655 / 746
  (1.16x / 1.24x); agent bench 10 turns: median gen 58 tok/s (MLX 44.1), median TTFT 1.41 s (MLX 13.5 s), total
  102 s (MLX 241.5 s); qh end-to-end PASS in 38 s.
  Attention sparsity (INK_AG_STATS, p17, global layers): (row, 64-key tile) pairs below 1e-6 of the row's top
  weight: 0-2.5%; 64x64 tiles where every row is below 1e-6: 0.0%. Skipping far keys is not an option.
- **Batch 3** (main fd7f72b, 19:57-20:11; b/work, c/work, d/work, a/work merged): all gates pass (op_diff ALL OK
  with and without q8; acc unchanged; spec identity; rollback). With D's multi-row kernels (one-pass matvec and
  split attention for 3..16 rows) a verify of 8 rows costs ~73 ms at 2.3k context (T=5 was 122-126 ms), T=3/4 =
  38.8 / 45.4 ms. Greedy, 600 tokens, token-identical to plain, one run each (tok/s):

  | | plain | drafter block 4 | drafter block 8 | lookup <= 3 | lookup <= 7 | lookup <= 15 | block 8 + lookup 15, temp 0.7 |
  |---|---|---|---|---|---|---|---|
  | copy | 33.4 | 65.9 | 85.8 | 71.2 | 88.7 | **100.6** | 97.1 |
  | edit | 33.3 | 59.3 | 73.7 | 67.3 | 81.2 | **96.2** | 93.2 |

  Lookup v2 (best of 4 trigram positions by backward match, >= 6 tokens): 526 of 600 drafts accepted on copy
  (v1: 391 of 625). p2 (free text, 99 tokens): block 4 41.5, block 8 with the 8-arm bandit 34.0 (too much
  exploration) -> policy replaced: k follows acceptance (2k+1 / accepted+1), plain fallback by measured rate.
  B: MoE LUT no gain (134.6 vs 137.2 us), CTRL_GEMM -6% on p3 prefill (3.72 vs 3.96 s), MOE_GEMM2 slower.
  C (server, short set): ctxbench temp 0.7 gen 36.8 / 33.3 tok/s lossless, 38.1 / 38.1 8-bit at 1.2k / 9.5k (MLX
  23.1 / 24.2); prefill 698 / 750 (MLX 565 / 602); agent bench 10 turns: median gen 63.4 lossless, 66.5 8-bit
  (76 on turns 6-10), MLX 44.1; totals 97.3 / 88.9 s, MLX 241.5 s; qh end-to-end PASS in 18 s.
