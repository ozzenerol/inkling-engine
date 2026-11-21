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
