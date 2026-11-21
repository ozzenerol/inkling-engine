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


## D: multi-row forwards, T = 5..16 (2026-11-21 night, model-free benches, one run each)

- **Where the T >= 5 cliff was**: above 4 rows attention went to `attn_prefill`, 8 threadgroups for the whole
  context: 2.16 ms per call at 2k global (T = 8 or 16), 32 ms at 32k. Dense: passes of 4 rows, W reread per pass.
- **Attention** (`mr_attn_split` / `mr_attn_combine`, kernels/multirow.metal): B's splits, block walk and per-row
  arithmetic; one simdgroup per (8-row group, block parity), K/V tile staged once per parity for all rows.
  Per call, new vs old: 2k global T=8 0.226 ms vs 2.164, T=16 0.248 vs 2.166; 32k global T=5 1.12, T=8 1.43 vs 32.2,
  T=16 3.07 vs 32.4; 128k global T=8 5.2, T=16 11.8; local T=16 0.229. Also faster than `attn_decode_split` at 3-4
  rows (32k: 1.11 vs 1.94 ms; 128k: 3.81 vs 6.8) and equal at 1-2, so it is the default from 3 rows
  (INK_MR_ATTN_MIN, INK_MR_ATTN=0 = off).
- **Dense** (`mr_matvec`, one pass over W, x rows walked one at a time with the weight chunks held): 2.58 GB of
  attention projections: T=5 9.7 ms, T=8 10.1, T=16 18.1 (passes of 4: 12.6 / 12.9 / 20.8; 1 row 7.8). Output head
  0.92 GB: T=5 2.37 ms, T=8 2.65, T=16 4.24 (passes of 4: 4.70 / 4.73 / 9.54). Best layout: 2 output rows per
  simdgroup up to 8 rows, 4 above. A loop over only the rows present (no unrolling) is slower (22-32 ms at 16).
  Above 8 rows it is compute bound. Not better: skinny2 21.2 ms at 8 rows, 28.0 at 16; t8tg 81 ms at 8 rows.
  `mr_matvec_multi` (linear_multi, controller_multi) uses 4 rows per simdgroup for both sizes; unmeasured.
- **Bit identity** (op_diff): "linear rows==T1" (linear, linear_multi, bf16/f32 in and out) and "verify==decode"
  (global, local ring) at T = 4, 5, 8, 11, 16: all rows equal. Fused mix+norm, wide mix and controller_multi now run
  up to 16 rows (the 256-thread mix reduces in another order than the 1024-thread one used for decode).
- **MoE**: not touched. Cost is per route (T=4 449 us per call vs 136 at T=1, B), so a T-row verify costs at least
  plain + (T-1) x ~6.8 ms unless rows share experts.
- Switches: INK_MR=0 (dense passes of 4), INK_MR_ATTN=0, INK_MR_ATTN_MIN=n.
- Open: forward times per T with the model (queue d), drafter 8 ms -> 3 ms, a 12-row dense variant, q8 multi-row.
- **Batch 4** (main c65c4be, 20:14-20:26): all gates pass. New drafting defaults (block 8, lookup up to 15, draft
  count follows acceptance, plain fallback), 600 tokens, one run each, greedy token-identical to plain (tok/s):

  | | plain | defaults greedy | no lookup | defaults temp 0.7 | plain temp 0.7 |
  |---|---|---|---|---|---|
  | copy (2.3k ctx) | 33.6 | **100.4** | 89.9 | 98.4 | 33.2 |
  | edit (2.3k ctx) | 33.4 | **91.0** | 72.9 | 89.7 | 33.2 |
  | p0 (code answer) | 36.2 | 53.8 | 53.4 | 49.6 | 36.0 |
  | p2 (short prose) | 36.4 | 36.6 | 38.3 | 43.7 | 35.9 |
  | 8-bit copy | 38.3 | 91.1 | | | |
  | 8-bit p2 | 42.3 | 48.5 | | | |

  The 8-bit build is behind on copy text because the multi-row kernels are bf16 only (D: q8 variants next).
  Server (C): ctxbench temp 0.7 gen 34.7 / 32.8 lossless, 36.5 / 35.6 8-bit (MLX 23.1 / 24.2), prefill 731 / 809
  (MLX 565 / 602); agent bench: median gen **76.5** lossless, 64.6 8-bit (MLX 44.1), totals 82.2 / 93.0 s (MLX
  241.5), median TTFT 1.37 s (MLX 13.5); qh end-to-end PASS in 17 s.
  D's expert counter (distinct experts as % of routes in one verify): copy T=4 67%, T=8 52%, T=16 35%; p2 T=2
  79%, T=4 59%, T=8 49%: rows of a verify share experts heavily -> expert-union MoE (D) saves up to 65% of the
  MoE reads, the only marginal cost of a verify row left (6.7 ms per row today).
  B: conv+mix+norm merge 36.5 vs 36.6 tok/s (no gain, stays off).
- **With the model** (batch 3, copy.ids, ctx 2.3k, one run): verify T=8 74.1 ms (old path 140.6), T=16 130.1 (202.7);
  copy at lookup max 15: 106.2 tok/s vs 69.8 without the multi-row kernels. T=16 = 29 + 15 x 6.7 ms: the marginal
  row cost is the MoE alone.
- **Expert sharing between the rows of a verify** (INK_MOE_STATS, batch 4): distinct experts as % of routes: copy
  T=4 66.9, T=8 52.2, T=16 34.9; edit 68.9 / 47.9 / 35.0; free text p2 T=2 79.1, T=3 67.6, T=4 59.0, T=8 49.1
  (independent routing would give 91 / 81 / 64 at T = 4 / 8 / 16).
- **Expert-union MoE** (`mr_moe_group` / `_gateup` / `_down` / `_wsum`, default from 3 rows, INK_MR_MOE=0 off,
  INK_MR_MOE_MIN=n): each used expert read once and applied to every row routed to it; rows == T=1 bit for bit
  (op_diff "moe rows==T1", nvfp4 and mxfp8). bench_moe, random routes (little sharing), us per call, union vs per
  route: nvfp4 T=2 255 / 233, T=4 451 / 448, T=8 780 / 867, T=16 1265 / 1706; mxfp8 T=2 362 / 346, T=4 650 / 673,
  T=8 1130 / 1317, T=16 1798 / 2610. **NVFP4 is bound by decoding the codes, not by the read**: sharing only the
  read gave 1743 us at T=16 (no gain); decoding a chunk once into 32 halves and applying it per row gave 1265.
- q8 one-pass matvec for 5..16 rows (`mr_matvec_q8`): written, unmeasured.
- **Batch 5** (main 180d9f3, 20:29-20:42; D's expert-union MoE from 3 rows, B's q8 linear_multi opt-in): all gates
  pass. 600 tokens, one run each, greedy token-identical to plain (tok/s):

  | | plain | defaults greedy | defaults temp 0.7 |
  |---|---|---|---|
  | copy | 33.6 | **128.1** | 124.5 |
  | edit | 33.4 | **114.3** | 113.3 |
  | p0 | 36.1 | 59.8 | 54.1 |
  | p2 | 36.6 | 41.0 | 46.5 |
  | 8-bit copy | 38.3 | 108.2 | |
  | 8-bit p2 | 42.2 | 50.5 | |

  Server (C): agent bench median gen **88.6** lossless, 68.1 8-bit (MLX 44.1); totals 74.0 / 85.0 s (MLX 241.5);
  new copy scenario (6 turns, print a ~500-token file back with a rename): 103.3 lossless, 93.9 8-bit; ctxbench
  temp 0.7 gen 33.5 / 35.4 lossless, 38.2 / 36.1 8-bit (MLX 23.1 / 24.2); prefill 740 / 808; qh e2e PASS 16 s.
  B: q8 matvec model-free 795 GB/s of codes at T=1 (dense reads of the 8-bit build: 1.7 ms against 8.1 ms bf16),
  so the 8-bit plain step (23.7 ms) is now almost all MoE + head + small-dispatch latency; q8 linear_multi and
  conv_mix: no gain; MoE gate/up with 2 or 1 rows per simdgroup: slower (249 / 237 vs 262 GB/s).

## B: correction (fused decode ops were not wired)

The Metal overrides for `attn_prep_local`, `attention_gated` and `conv_add` were missing from metal_backend.mm from
5f0fe21 until b/work ab13ba8: the edit that inserted them did not apply, so the Backend defaults (primitive
sequences) ran. Consequences: the "fused attn_prep_local / attention_gated / conv_add" item in "B: decode speed
work" contributed nothing to 31.8 -> 36.7 tok/s (that came from the wide mix, router matvec, concurrent encoder and
mix+norm); the op_diff "prep" / "conv_add" cases compared primitives with primitives; the INK_SKIP numbers were on
the unfused paths. The overrides now exist behind INK_FUSE2=1 and are unmeasured until a batch runs
tools/queue/b.sh (exact op_diff cases, attention_gated identity, decode off vs on, DSpark identity).
Also: batch 5's "q8 matvec 795 GB/s" came from zero-filled weights in bench_q8 and is not a read measurement.
- **Batch 6** (main 73d0183, 20:45-21:05; stopped by A during C's full sweep): gates pass. D's q8 multi-row: 8-bit
  copy 119-129 tok/s (118.8 without), token-identical. Parallel target sampling of verify rows in. C's cold sweep,
  lossless build, temp 0.7 (prefill / gen tok/s): 1.2k 749 / 34.8, 9.5k 815 / 32.9, 18.9k 782 / 31.5, 38.6k
  710 / 29.2, 78k 593 / 23.8 (MLX: 565 / 23.1, 602 / 24.2, 571 / 23.3, 495 / 22.0, 385 / 16.5).
  B's 8-bit plain-step ablation (23.6 ms): MoE 7.5, dense 6.2, small ops 4.7, attention 3.0, mix 2.5, controllers
  2.5, router 2.0, logits 1.5 ms; ~1200 dependent barriers per 1560 dispatches per token. The same q8 dense bytes
  take 1.8 ms model-free (random codes): in-model they are 3x slower, cause open (dispatch gaps / GPU performance
  state: B tests INTERLEAVE and INK_BUSY in batch 7). 8-bit verify: T=1 24.3 ms, T=2 33.3 (the 4-row matvec
  template computed 4 rows at T=2: B adds 2-row kernels), T=3 36.8, T=4 42.4.
- **Target (user, 21:00): 60 tok/s at the start of the context, 40 tok/s at 128k.** See tools/WORKFLOW.md.
- **Target relaxed (user, 21:20): "even 40 tok/s at the beginning and 20-30 at the end is fine, as long as each
  prompt is read fast".** Priority is now prefill (read tok/s, time to first token).
- **Batch 7** (main b5799a2, 21:06-21:35): all gates pass. C's depth bench (8-bit server, one growing conversation
  of essay requests, temp 0.7, 400 tokens per turn, one run):

  | prompt tokens | 1.2k | 9.6k | 39k | 79k | 127k |
  |---|---|---|---|---|---|
  | gen tok/s | 39.0 | 38.8 | 31.0 | 26.0 | 22.7 |
  | read tok/s (new tokens / TTFT) | 724 | 792 | 658 | 504 | 384 |

  Agent bench 8-bit: median gen 65.1, copy scenario 87.1, median TTFT 1.35 s; qh e2e PASS in 20 s.
  A: 8-bit, temp 0.7, spec_check: p2 43.0 tok/s; long32k 31.2 (plain step 31.8 ms, T=2 37.1, T=4 50.2, prefill
  489 tok/s); long64k 26.7 (prefill 436).
  B: fused decode ops finally wired (FUSE2): tokens identical, 43.5 vs 42.9 tok/s on 8-bit (+1.4%) -> default;
  2-row matvec kernels: T=2 verify 29.2 vs 32.8 ms, 49.4 vs 45.3 tok/s -> kept; INK_BUSY background load: no
  effect at any level (the "GPU performance state" hypothesis for the 3x in-model dense gap is refuted).
- **Batch 8** (main f5fb5b1, 21:40-22:15): all gates pass. C's depth bench (8-bit server, temp 0.7): gen 38.1 /
  37.8 / 30.5 / 25.8 / 21.8 tok/s and read 745 / 790 / 657 / 503 / 381 tok/s at 1.2k / 9.6k / 39k / 79k / 127k.
  [ttft] lines: a turn's time to first token is its prefill (775 new tokens at ~640 tok/s = 1.2 s) plus ~85 ms
  until the first visible byte (two marker tokens); tokenize / cache / checkpoint are ~1-10 ms.
  A: 8-bit at 126k context (spec_check, temp 0.7): 21.2 tok/s; plain step 47.1 ms, verify T=2/3/4 52.7 / 67.1 /
  70.1 ms, drafter 17 ms per round (6-8 ms at short context: its global layer reads the whole context).
  Prefill attention with P as ONE half tile (INK_AG_HALF=1): long set mean KL 4.33e-3 vs 3.99e-3 hi+lo (median
  3.49 vs 3.89, max 1.07e-2 vs 5.69e-3), prefill at 64k 467 vs 438 tok/s (+6.6%) -> not worth it, stays off.
  B: 8-bit prefill 2048 tokens 2.51 s (816 tok/s), 800 tokens 1.27 s (630 tok/s); at T = 2048 moe_gemm_nvfp4
  54% (~9 TFLOPS), gemm_q8 22% (~10 TFLOPS), attention 8%: both big GEMMs are within 1.3x of the GPU's 13.1
  TFLOPS, so steady-state prefill is near the hardware. MoE tiles of 128 rows: slower (4.46 vs 3.87 s).
  B's INTERLEAVE bench (batch 7): q8 matvec x 288 back to back 1.76 ms, with 2 tiny dependent dispatches after
  each 7.82 ms, with 6 14.13 ms (~10 us per tiny dependent dispatch), which contradicts the small gains of
  FUSE2 (-320 dispatches, +1.4%) -> B runs a NULL-kernel chain (same dispatches, no work) in batch 9 to settle
  how much of the 23.6 ms plain step is dispatch latency.
- **User (22:15): keep going "until this is blazingly fast".**

## D: long-context attention, INK_GA (2026-11-21 late, model-free, one run each; last run overlapped batch 8's start)

- The split kernels (B's and mine) stage every K / V tile in threadgroup memory for 8x8 matrix products: at 128k
  global they read KV at 137-180 GB/s (T=1 2.98 ms, T=2 3.01, T=4 3.81, T=8 5.23, T=16 11.7 ms per call).
- **INK_GA=1** (kernels/multirow_ga.metal: ga_scores, ga_rowmax, ga_pv, ga_combine): K then V read once straight
  from device memory, scalar fma chains, scores kept in a scratch buffer between the two passes. One simdgroup =
  the 3 rows of one token with its q in registers; 1..16 tokens per threadgroup on the same keys. One arithmetic
  for every T (verify == decode, op_diff ALL OK with INK_GA=1).
  Per call, GA vs split: 128k T=1 **1.56** vs 2.98 ms, T=2 **1.82** vs 3.01, T=4 **2.96** vs 3.81, T=8 5.64 vs 5.23,
  T=16 15.9 vs 11.7 (4 tokens per threadgroup; to redo); 32k T=1 0.40 vs 0.92, T=2 0.50 vs 0.92, T=4 0.78 vs 1.11,
  T=8 1.48 vs 1.44, T=16 2.87 vs 3.07. T=1 is at bandwidth (0.52 GB in 1.56 ms = 335 GB/s); from 4 rows it is
  bound by the arithmetic (about 2.2 TFLOPS in scalar code).
- A first version with all rows of up to 4 tokens in one simdgroup (6 or 12 rows of q in registers) collapsed:
  128k T=2 3.54 ms, T=4 9.9, T=16 37.8.
- Skipping V blocks with negligible weight (INK_GA_THR) is implemented but off: A's INK_AG_STATS at 6k shows no
  64-key block negligible for every row.
- 8-bit KV: on hold (user's relaxed target). It would halve T=1 but not T=4, which is arithmetic bound.

## B: state at end of 2026-11-21 (b/work)

**Default on (measured in a batch):** runtime-compiled Metal library with all pipelines built at start; weights
pread into resident buffers + residency set; bf16 matvec (1, 2, 4 rows), prefill GEMM (11.5 TFLOPS); fast MoE
decode kernels with the vector NVFP4 decode (131.6 us per call); MoE prefill (GPU route sort + grouped dequant GEMM,
64-row tiles); split-K decode attention with fixed position-aligned splits (verify rows == decode rows); fused
decode ops attn_prep_local / attention_gated / conv_add (really wired only since batch 7: 43.5 vs 42.9 tok/s,
tokens identical), mix + norm, batched controllers; prefill controller GEMM (p3 3.96 -> 3.72 s); parallel prefill
short conv; q8 kernels for the 8-bit build (43-45 tok/s vs 36.7 converted); 2-row matvecs for T = 2 verifies
(32.8 -> 29.2 ms); concurrent encoder with own hazard barriers; DSpark drafter (greedy + sampled drafts).

**Opt-in, measured, no gain (leave off or delete):** INK_CONV_MIX (36.5 vs 36.4), INK_Q8_MULTI (42.2 vs 42.1),
INK_MOE_RPS=2/1 (slower: 249 / 237 vs 262 GB/s), INK_SKINNY / INK_MV_T8 (D owns multi-row now), INK_ROUTER_FUSED.
Removed after measuring: MoE GEMM v2, FP4 byte LUT, 128-row MoE tiles, INK_BUSY (no effect at any load).

**Opt-in, unmeasured until batch 9's b.log:** INK_MOE_M=32 / 0 (32-row MoE prefill tiles; expect ~20% less MoE
work at ~800 new tokens; must be byte-identical), INK_RW=1 (read / write hazard sets from pipeline reflection),
INK_NULL=1 (timing only: every dispatch replaced by a one-thread kernel), INK_WARM_SECS (GPU keep-warm, default 0,
never measured: needs a tool with an idle gap before a request).

**Open questions**
1. What holds the plain step (23.6 ms on the 8-bit build: MoE 7.5, dense 6.2, small 4.7, attention 3.0, mix 2.5,
   controllers 2.5, router 2.0, logits 1.5 by INK_SKIP)? The same 1.37 GB of q8 codes takes 1.8 ms model-free, and
   7.8 / 14.1 ms with 2 / 6 tiny dependent dispatches after each matvec, yet removing ~500 dispatches from the
   step gained <= 1.4%. Suspect: my hazard tracker treats common READS as hazards and a barrier is global, so the
   step is fully serialized (1200 barriers per 1560 dispatches). INK_NULL (bare chain) and INK_RW decide it.
2. 32-row MoE tile verdict (800-token turns run at 630 tok/s against 816 at 2048).
3. bench_q8 reports 750 GB/s with random codes: above the 366 GB/s we took as the machine's limit, which was a
   compute limit of the bf16 kernel. The real memory ceiling is unmeasured.

**Next 3 steps**
1. Read bench/batch-9/b.log. If INK_NULL shows a chain of ~8-10 ms and INK_RW is token-identical with fewer
   barriers and faster: make INK_RW the default; then look for the remaining false dependencies (scratch buffers
   shared between ops: moe-a, attn-part, ctrl-pair are reused every visit and serialize visits).
2. If INK_NULL is ~1-2 ms: the time is in the kernels. Compare per-kernel us in the model with the model-free
   benches (matvec_q8 15 us per call in the profile vs 6 us in the bench) and test pipeline-switch order and
   threadgroup-memory setup as causes.
3. Apply the batch 9 verdict on INK_MOE_M (default 0 if 800 tokens gains and 2048 does not lose), then delete the
   opt-in paths listed above as "no gain" so the backend has one path per op.

Corrections on record: see "B: correction (fused decode ops were not wired)" above; my NOTES numbers from before
batch 7 for those three fusions were for the unfused paths.

## D: state at end of 2026-11-21

**Default (measured, in main, gated by op_diff rows==T1 / verify==decode):**
- One-pass matvec for 5..16 rows, bf16 and q8 (`mr_matvec`, `mr_matvec_q8`; linears, linear_multi, the output
  head): INK_MR=0 restores passes of 4 rows.
- Attention for 3..16 rows on the split kernel with K/V staged once for all rows (`mr_attn_split`), local and
  global: INK_MR_ATTN=0 off, INK_MR_ATTN_MIN=n.
- Expert-union MoE from 3 rows (`mr_moe_*`): INK_MR_MOE=0 off, INK_MR_MOE_MIN=n.
- Fused mix + norm, wide mix and controller_multi up to 16 rows.
- Effect with the model (batches 3-6, one run each): verify T=8 140.6 -> 74.1 ms, T=16 202.7 -> 130.1 ms (multi-row
  dense + attention), copy task at lookup 15: 103.7 -> 138.9 tok/s with the union MoE; 8-bit copy 118.8 -> 128.7
  with the q8 multi-row matvec. Every run token-identical to plain decoding.

**Opt-in, measured model-free only:**
- INK_GA=1: direct two-pass global attention for every T (see "D: long-context attention"). 128k per call: T=1
  1.49 ms vs 2.98, T=2 1.79 vs 3.01, T=4 2.92 vs 3.81, T=8 5.67 vs 5.23, T=16 11.19 vs 11.74. With the model:
  batch 9 (long32k / long128k, on and off); d.log not read yet.

**Opt-in, built and kernel-checked, never run on the GPU:**
- The drafter's global layer on the GA kernels (with INK_GA=1; INK_GA_DRAFT=0 keeps the split kernel).
- INK_DSPARK_ROWS=1: the drafter block runs count + 1 rows instead of 8.
- INK_MR_MAX=n (<= 64): prefill-only sizes 17..n through the verify kernels (dense in passes of 16, attention in
  groups of 16 rows, union MoE lists of 64). Default 16. For 17..32-row verifies also Model::kMaxHist and the
  op_diff row lists must go up (A).
- INK_GA_THR=x: drop attention weights below exp(-x) of the row maximum and skip V blocks with none above it.

**Known loose ends:** the q8 output head showed 14 ms per call over 2 calls in batch 6's profile (t8_r2 layout,
possibly first touch): re-measure with bench_dense / bench_q8 on the head shape. `mr_matvec_multi` uses 4 rows per
simdgroup at every size and q8 linear_multi above 4 rows goes matrix by matrix: neither measured against a variant.
Bench-only kernels to delete once nobody needs them: the unused `mr_matvec` variants, `mr_attn_split_g6_abl*`,
B's t8tg and skinny GEMMs (handed to me).

**Next 3 steps (2026-11-22):**
1. Read batch 9's d.log and tomorrow's first batch: decide INK_GA as default from the long128k end-to-end numbers;
   judge INK_DSPARK_ROWS and the drafter on GA by tok/s and acceptance (A).
2. The drafter at long context (17 ms per round at 126k, 6-8 ms at short context): profile it per kernel at 126k,
   then cut rows / the head pass accordingly.
3. Profile a 17..63-token read (prompt tail, small agent turn) and fix what is slow there besides attention
   (A moved the attention GEMM down to 17 rows); INK_MR_MAX and bench_dense / bench_moe at 20..63 rows are ready.

- **Batch 9** (main c618174, 22:04-22:37, the last run of 2026-11-21): all gates pass.
  - C's depth bench (8-bit server, temp 0.7, one growing conversation): gen 39.9 / 38.1 / 30.3 / 27.2 / 22.0 tok/s
    and read 729 / 794 / 658 / 503 / 381 tok/s at 1.2k / 9.6k / 39k / 79k / 127k. Agent bench: median gen 64.6,
    copy scenario 87.7, median TTFT 1.35 s, read 573 tok/s per turn; qh e2e PASS.
  - D's INK_GA (direct global attention, opt-in), 8-bit, temp 0.7: **126k context 26.4 tok/s vs 21.1** (plain
    step 36.0 vs 46.8 ms, T=2 43.5 vs 52.6, T=4 64.6 vs 70.3); 32k 35.7 vs 31.7 (plain 27.2 vs 31.3 ms).
    Lossless long32k greedy: dspark token-identical to plain with GA; op_diff with GA ALL OK.
    -> make INK_GA the default (first batch tomorrow).
  - B's NULL chain (every dispatch replaced by a one-thread kernel, same barriers and commits): 510 tok/s, i.e. the
    whole dispatch chain of a plain step costs ~2 ms. So the 23 ms step on the 8-bit build is real kernel time, not
    dispatch latency. Open question for tomorrow: why the same q8 dense bytes take 1.8 ms model-free and ~6 ms in
    the model (INTERLEAVE: 2 dependent tiny dispatches after each matvec 7.8 ms, independent ones 4.5 ms, back to
    back 1.8 ms: the big readers slow down when anything runs between them).
  - B's read/write hazard tracking (INK_RW=1): barriers 83.6k vs 101.5k per 121k dispatches, 45.4 vs 44.0 tok/s on
    8-bit (+3%), no change on lossless, tokens identical, dspark identity holds -> default tomorrow.
  - B's MoE prefill tile of 32 rows (INK_MOE_M=0 = 32 rows below T = 1280): an 800-token read 1.19 s vs 1.36 s
    (8-bit) and 1.20 vs 1.54 s (lossless), 2048 and 2886 tokens unchanged, logits byte-identical -> default tomorrow.

## A: handover at the end of 2026-11-21

State of main: everything of A, B, D merged (c/work: see C's final line below). All gates pass on c618174.

Reference numbers (8-bit build = server default; one run each):
- Generation, free text, temp 0.7 (C's depth bench): 39.9 tok/s at 1.2k, 38.1 at 9.6k, 30.3 at 39k, 27.2 at 79k,
  22.0 at 127k (26.4 at 126k with INK_GA=1). MLX: 23.1 / 24.2 / 22.0 / 16.5 / 13.8.
- Reading new tokens: 729 / 794 / 658 / 503 / 381 tok/s at the same depths (MLX cold: 565 / 602 / 495 / 385 /
  324). A turn's time to first token is its prefill: ~775 new tokens in 1.2 s.
- Copy-heavy output (lossless, spec_check, token-identical to plain): 128 tok/s copy, 114-116 edit; 8-bit 119-130.
- Agent bench (10 turns): median gen 64.6 (8-bit) / 88.6 (lossless), median TTFT 1.35 s; MLX 44.1 and 13.5 s.
- Plain decode step: 27 ms lossless, 23 ms 8-bit at short context; 36 ms at 126k with GA.

User's target (relaxed 21:20): ~40 tok/s at the start, 20-30 at the end of the context, and prompts read fast.
Later (22:15): keep going until it is "blazingly fast". Then: one last run and stop; continue the next day.

First batch tomorrow (defaults to flip, each already measured identical or gated):
1. INK_GA on by default (D) - +25% at 126k, +13% at 32k.
2. INK_RW on by default (B) - +3% on 8-bit.
3. INK_MOE_M=0 by default (B) - 800-token reads 13-22% faster.
4. Measure D's unmerged-but-unrun switches: drafter on GA, INK_DSPARK_ROWS=1 (drafter 17 ms per round at 126k).
Then: (a) why kernels run 3x slower inside the model than back to back (the only big lever left for the plain
step; candidates: cold weights per visit vs the bench re-reading warm buffers, pipeline switches, threadgroup
memory setup); (b) reading at depth (381 tok/s at 127k; ceiling ~600: prefill attention is 5.5-6.9 TFLOPS with
P as hi + lo half tiles; one half tile is +7% at 64k but showed a KL outlier, off); (c) FULL sweep + MLX
baselines for docs/RESULTS.md (cold 127.5k rows, MLX copy scenario).
Things tried and dropped today: chain kernels (single-threadgroup megakernels: slower), real speculative
sampling (no better than compare with this drafter), 8-arm bandit over draft counts, attention key pruning (not
sparse), MoE tiles of 128 rows, FP4 LUT, conv+mix merge, q8 linear_multi, GPU "busy" load, 8-bit KV (on hold:
INK_KV_Q8SIM in the CPU backend simulates it for an accuracy check that was not run).

## C: state at end of 2026-11-21

Branch c/work = main c618174 plus this section and docs/RESULTS.md (batch 9 is the reference table there).

- **Server defaults** (`inkling-server`, port 8091, what batch 9 measured): model `Inkling-2-Mini-MLX-8bit`
  (falls back to the 4-8bit dir), DSpark drafts on with draft block 8 (`--draft-block`, `--no-mtp`), prompt-lookup
  drafts on (`INK_NO_LOOKUP=1` turns them off), prefill chunk 2048 with near-equal splitting, turn splicing on,
  shared-prefix checkpoint for prefixes >= 256 tokens, context 131072, effort from the request (template default
  0.9 when the request gives none).
- **Reference table** (batch 9, 8-bit, one growing session, temp 0.7, 400 tokens, one run; batches 7 and 8 agree
  within 1.1 s TTFT and 1.8 tok/s): read tok/s / TTFT / gen tok/s = 1.2k 729 / 1.64 s / 39.9; 9.6k 794 / 10.1 s /
  38.1; 39.3k 658 / 44.5 s / 30.3; 78.6k 503 / 77.4 s / 27.2; 127.3k 381 / 127.0 s / 22.0. Relaxed target (40 at
  the start, 20-30 at the end, read fast): end in range, start 38.1-39.9 over three batches (not met).
- **Agent sessions** (batch 9): read, 10 turns: 88.1 s, median TTFT 1.35 s, 573 read tok/s, 64.6 gen tok/s (MLX
  241.5 s, 13.5 s, 44.1); copy, 6 turns: 49.0 s, 1.08 s, 87.7 gen tok/s. qh end-to-end PASS, 28 s.
- **TTFT breakdown** (`[ttft]` log lines): prefill is 93-99% of TTFT. A turn of 775 new tokens = 1.25-1.41 s, of
  which prefill 1.20-1.32 s (645 -> 589 tok/s from 3k to 13k context), 50-100 ms the marker tokens before the
  first visible delta, checkpoint 2-15 ms, tokenize + cache under 2 ms. Small chunks (552-790 tokens) run at
  540-645 tok/s against 720-800 for chunks of 2000: that gap is in the engine's per-chunk cost.
- **Not measured yet**: MLX on the copy scenario; the cold sweep (ctxbench) to 127.5k on the 8-bit build (and the
  lossless 127.5k row); the 30k agent session since batch 3; a second session that HITS the pinned shared-prefix
  checkpoint on the real engine (in batch 9 the copy session came after a session whose first prefill was split
  1058+1059, so no checkpoint at 2045 existed and it prefilled 2643 tokens again; the pin was made then).
- **Next 3 steps (tomorrow)**:
  1. One `FULL=1 MLX=1` batch item: 8-bit cold sweep to 127.5k, 30k agent session, MLX copy baseline; store
     `agent-mlx-copy.json` in tests/server/baseline and fill the gaps in docs/RESULTS.md.
  2. Shared prefix on the first request: split the first prefill at the system+tools boundary (known from the
     rendered prompt) instead of near-equal, so the pin exists from session 1; add a bench step with two sessions
     and check `cache: ckpt` on the second.
  3. With A/B: the per-chunk cost on short reads (775 tokens at 600 tok/s against 800 steady) and the 40 tok/s
     line at the start; server side, send the first SSE byte before the marker tokens are generated.

## Batch 10 (main 2b72df9, 23:17-23:56, one session working alone; one run each)

Defaults flipped: INK_GA (direct global attention, drafter too), INK_RW (read/write hazard sets), INK_MOE_M=0
(32-row MoE prefill tiles below 1280 rows). All gates pass (op_diff ALL OK with and without q8, acc both builds as
before: only the known near-ties, spec identity, rollback identical).

C's depth bench, 8-bit server, temp 0.7 (batch 9 in brackets): **target met at every depth**

| context | 1.2k | 9.6k | 39k | 79k | 127k |
|---|---|---|---|---|---|
| gen tok/s | 41.9 (39.9) | 44.8 (38.1) | 37.3 (30.3) | 33.1 (27.2) | 28.6 (22.0) |
| read tok/s | 808 (729) | 800 (794) | 668 (658) | 506 (503) | 382 (381) |

Agent bench 8-bit: read 10 turns 47.2 s (88.1), median TTFT 1.20 s, median gen 66.5; copy 6 turns 46.5 s, TTFT
0.93 s, gen 91.9; MLX 241.5 s. qh e2e PASS 17 s.
D (spec_check, 8-bit, temp 0.7): 126k GA 27.5 vs 21.6 tok/s off (plain 35.1 vs 45.9 ms, draft 11.7 vs 16.5 ms);
INK_GA_DRAFT=0 27.6, INK_DSPARK_ROWS=1 27.7 (noise: no change); 32k 37.1 vs 32.6 off, ROWS=1 37.2.
B: rw 45.7 vs 43.9 tok/s on 8-bit (barriers 146k vs 176k of 210k), lossless 37.5 vs 37.3, tokens identical;
MoE tiles 800 tokens 1.15 vs 1.24 s (8-bit), 2048 / 2886 unchanged, logits identical.
Decode floor (A, from the safetensors): a token reads 1.96 GB of experts (5.1 MB each, 8 x 48 visits) + 1.83 GB
dense (38 MB per visit) + 0.49 GB head = 4.3 GB: 10.7 ms at 400 GB/s vs the 23 ms plain step (~185 GB/s).
bench_q8's 756 GB/s (batch 9) is above the M3 Max's 400 GB/s: its matvecs are independent (all read x, write own
y) and the GPU timestamps may miss work; b/work adds a wall clock to settle it before using it as the in-model gap.

## Batch 11 (main a705011, 23:58-00:10) and the decode / prefill accounting (2026-11-22, one run each)

Gates pass. Depth bench 8-bit: gen 42.1 / 40.6 / 36.6 / 32.3 / 27.9 tok/s, read 807 / 800 / 664 / 507 / 381 at
1.2k / 9.6k / 39k / 79k / 127k (confirms batch 10). Agent read 47.4 s, TTFT 1.21 s, gen 68.5; copy 92.6; qh PASS.

**The "kernels are 3x slower in the model" gap does not exist.** bench_q8 with a wall clock: 288 q8 matvecs
(1.37 GB) 3.55 ms GPU / 3.92 ms wall = 349 GB/s (batch 9's 1.8 ms was read before INK_RW existed and its GPU
timestamps are not trustworthy: DEP=1 reports 1.0 ms GPU for 4.6-5.1 ms wall). As a dependent chain 267-296 GB/s.

**Decode step, 8-bit, per kernel** (INK_PROFILE, p2, 63 steps; sums to the 23 ms step):
weight readers 13.3 ms = q8 dense matvecs 5.45 (352 calls, 15.5 us, 336 GB/s) + MoE 6.6 (gateup 86 us, down 46 us
per visit: ~280-300 GB/s) + head 1.25. Floor for the 4.3 GB at 349 GB/s: 12.3 ms. So the readers are at the
hardware. The other 9.7 ms: local attention 1.86 (attn_decode_split_f1 46 us a call even at 32 keys: ~20 us fixed
+ ~13 us per 32-key block; 56 + 15 combine at a full window), residual mix 2.2 (120 calls, 18-19 us), bf16
multi-matvec 1.09 (24 calls, 45 us), router_select 0.6, and ~4 ms in ~15 kernels of 0.1-0.4 ms each.
No single fat target is left: a faster plain step is a grind of 0.5-1 ms pieces (MoE bandwidth to the q8 level
~1 ms, attention fixed cost / rel-bias precompute ~0.7, mix in two wide dispatches ~0.5).

**Small-kernel latency** (src/gpu/bench/bench_tiny.mm, dependent chain of 4000, us per dispatch): null 1.6;
elementwise over 2304 threads 1.7 (group size does not matter); rms in one group of 256 + barrier 4.0, of 1024
3.2; the same work in ONE simdgroup (72-element loops per thread) 11.5-15, fused rms+ew+rms in one simdgroup 19.5.
Rule: latency = the serial loop length per thread (~0.1 us per element), not the dispatch count (a dispatch is
~1.7 us). Spread over more threads, never fuse by lengthening per-thread loops (explains the chain dead end).

**Prefill at short context is compute bound at the GPU's rate**: profiles at 800 and 2048 tokens fit
time = 179 ms + 1.055 ms/token per chunk. An 800-token chunk is ~7.8 TFLOP (MoE 4.9, dense 2.9) in 1.02 s.
The fixed 179 ms per chunk is half expert weight reads (every visit touches all 128 experts: 31 GB, ~90 ms) and
the rest partly-filled tiles. So short agent turns (~650-700 tok/s) cannot reach the 2048-chunk rate by tuning.

Server (c/work 33f3f4b): the system prompt + tools prefix is pinned during the first prefill (prefix hint from
rendering the leading system messages alone), so session 2 restores it; stub test added. First SSE chunk was
already sent before the prefill.

## Batch 12 (main 0cf4a96, 00:14-00:29, one run)

Gates pass. Depth bench as batches 10-11 (127k: 27.7 gen, 383 read). **Prefix pin works on the real engine**: the
copy session (second session, same system prompt + tools) starts with `cache: ckpt`, 2043 of 2643 tokens cached,
TTFT 1.00 s (batch 10-11: full prefill, 3.35 s). Its sum of TTFT 5.6 s vs 7.9. The read session wrote 3639 tokens
this run (2010 before: sampled output, 12 turns' worth of text), so its 83 s total is not comparable.

## Batch 13 (main 0fb5072, 00:30-01:12, `MLX=1 tools/batch.sh queues full`, one run): the RESULTS.md sweep

docs/RESULTS.md is now batch 13. Cold sweep (ctxbench) prefill / gen tok/s, 8-bit: 1.2k 805 / 41.8, 9.5k 819 /
40.7, 18.9k 781 / 40.0, 38.6k 712 / 37.2, 78.1k 596 / 33.2, 127.5k 494 / 28.1; lossless 761 / 37.2 ... 495 / 25.2
(MLX 127.5k: 324 / 13.8). Depth bench lossless: gen 35.2 / 36.0 / 32.5 / 28.7 / 25.5. Agent at base 30000: 8-bit
118.5 s, lossless 114.6 s, MLX 742.8 s. qh e2e PASS twice (13 s, 18 s).
The MLX copy baseline failed: mlx_vlm answers /v1/models while loading and 503 on completions; bench_c.sh's
warm-up now retries until 200. Still to measure (MLX=1 on any batch).
The 8-bit read session at temp 0 writes 3639 tokens since batch 12 (2010 in batches 10-11): the pinned prefix
changes the first prefill's chunking (2045 + 72 instead of 1058 + 1059), the logits move in the last bits and a
near-tie goes the other way. Not a regression; totals of that session are comparable only within the same chunking.

## Batch 14 (main 8ddda6d, 01:15-01:32): mix in two wide dispatches; C's numbers contaminated

Gates pass (op_diff ALL OK, acc as before, spec identity, rollback). B: mix2 (plain mix as mix2_red + mix2_out,
6.6 + 6.4 us vs 18.2 in the profile) 45.8 vs 44.7 tok/s on 8-bit, tokens identical; before the batch 46.1-46.3 vs
45.5-45.7 over three runs. Default on (INK_MIX2=0 off). The fused-norm form stays one kernel (19 vs 22 us in two
dispatches + rms).
C's depth and agent numbers of this batch are NOT valid: the user ran nib chats on the same server at 01:28-01:30
(checkpoints at 1121 / 1157 / 1215 tokens in the log are nib's prefix), which evicted the bench's cache on every
turn (median TTFT 15.6 s) and shared the GPU. Batch 15 repeats them.

## nib (2026-11-22): a harness written for this engine, ~/projects/nib

Go, stdlib only, installed at ~/.local/bin/nib; README there. What it does differently because of this engine:
constant system prompt + tool list (the pinned prefix, 1121 tokens; environment in the first user message, the
same tools in read-only runs), append-only conversation with reasoning_content sent back, effort fixed per
session (the template puts the effort line after the system prompt: a change re-reads the whole conversation),
runaway thinking cut by closing the stream and appending a `Thinking effort level: 0.1` system message (rewind,
no cache loss; not yet measured whether the model follows a mid-conversation effort line), tool results cut to
8000 characters, read without line numbers and temperature 0 (copy drafts), compaction at 64k by asking for notes
on the cached prompt. First real run: tools/nib_e2e.sh task 5/5 tests pass, 5 steps, 15.6 s, wrote 731 tokens at
66 tok/s, read 2.3k at 563 tok/s. `tools/nib_e2e.sh` runs in C's queue after the qh test.
Note for installs: copy the binary with rm + cp; cp over the old file gets the new one killed (exit 137) by macOS.

## Batch 15 (main 407c76d, 01:33-01:53, one run): clean repeat of batch 14's server numbers, nib in the queue

Gates pass. mix2 again 45.7 vs 44.9 tok/s (8-bit, tokens identical). Depth bench 8-bit: gen 41.4 / 40.3 / 36.8 /
31.1 / 27.9 tok/s, read 753 / 762 / 614 / 469 / 349 at 1.2k / 9.6k / 39k / 79k / 127k. Agent read 86.9 s, median
TTFT 1.23 s; copy 47.1 s, TTFT 1.01 s. qh e2e PASS 37 s (6 steps, 1251 tokens out); **nib e2e PASS 14 s** (5
steps, 690 tokens out, wrote at 68 tok/s, read at 575 tok/s).
**Reading is 5-9% slower than in batch 13 and it is not the code**: inktest prefill of 2886 tokens takes
3.90-4.05 s with INK_MIX2=0 and with the default alike (3.58 s in batches 12-13). A game (Habbo, Adobe AIR,
started 01:21, 35% CPU, rendering) has been on the GPU since before batch 14. Prefill is compute bound, so it
pays for a shared GPU first; decode (bandwidth bound) lost 0-4%. Batch 13 stays the reference table. Check
`ps aux -r` for GPU users before trusting a batch.
MLX copy baseline: still not measured. This time bench_c.sh's warm-up loop waited on MLX's 422 (its API wants a
model field in the request) until killed; the loop now waits only on 503 / no answer. Run `MLX=1` on an idle GPU.
nib on the real model after the prompt change: "hi" is answered in one step without tools (2.9 s); a second
session reads 80 new tokens (the 1121-token prefix comes from the pinned checkpoint).
