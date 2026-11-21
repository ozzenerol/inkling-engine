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

(none held)

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
