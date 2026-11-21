# Inkling engine: brief for the three Claude sessions on this Mac

Write from scratch, in C++ with Metal, an inference engine for Inkling 2 Mini built for
agentic coding work. Goal: fast, with no accuracy loss. Three Claude sessions work on it
together: agree on interfaces first, then split the work, and keep each other informed
(ListAgents / SendMessage between sessions). Repo: ~/projects/inkling-engine (git init it).

## Roles (each session picks one, says which to the others)

- **A: model + accuracy.** Weight loader for the released NVFP4/MXFP8/bf16 safetensors as
  they are (no requantization), the forward pass, and the reference test harness against
  MLX-VLM (greedy token-for-token match plus logit KL). Owns "is it correct".
- **B: Metal kernels + speed.** Fused MoE kernels (NVFP4 and MXFP8 dequant inside the
  matmul/matvec), attention (local and global), prefill and decode paths, DSpark
  speculative decoding, memory plan. Owns "is it fast".
- **C: server + agentic.** OpenAI-compatible /v1/chat/completions with streaming,
  chat_template.jinja rendering, tool calls as `tool_calls` (port the parser in
  mlx_vlm/tools/parsers/inkling.py), thinking as `reasoning_content`, `reasoning_effort`,
  a prefix KV cache that survives across turns (agent prompts only grow; qh sends
  preserve_thinking), port 8091 (qh's ~/.qh/config.json points there), benchmarks.

## Model

- Weights: ~/llm/mlx-models/Inkling-2-Mini-MLX-4-8bit (netprojects build, SHA-256 checked
  against Hugging Face). Includes mtp.safetensors, the DSpark draft (0.64 GB).
- 24-layer decoder MoE. Layers 0-5 once, middle 12 run 3 times with shared weights, layers
  18-23 once: 48 layer applications per token. 128 experts, 8 active. 26.4B total, 3.4B
  active (5.4B effective with the loops). Hybrid local/global attention. Context up to 1M.
- Routed experts: NVFP4 (per-expert global scale) in layers 0-19, MXFP8 in 20-23.
  Attention, embeddings, output head, vision tower: bf16. Router fp32. Loop gates unquantized.
- Reference code: ~/llm/inkling-venv/lib/python3.12/site-packages/mlx_vlm (Inkling model
  code, cache, speculative/drafters, tools/parsers/inkling.py). Python: ~/llm/inkling-venv/bin/python.
- Look at ~/projects/saetta-macqwen first: the user's Rust+Metal engine for Qwen3.6 on this
  same Mac (117 tok/s vs MLX 77). Reuse its ideas: kernels, chunked prefill, prompt cache,
  MTP drafting, bench tools (tools/bench.py, final_bench.sh), docs/PLAN.md.

## Machine

M3 Max, 40-core GPU, 48 GB. Measured today: fp16 matmul 13.1 TFLOPS, matvec 366 GB/s.
iogpu.wired_limit_mb = 44000 (raising it needs sudo: ask the user).

**Memory warning:** on 2026-11-20 saetta (8-bit model, 36 GB mmap'd weights as no-copy
views, 38 GB GPU buffers) fell mid-session from ~110 to 7-30 tok/s at the same context:
almost certainly weight pages evicted under memory pressure and re-read from SSD. Keep the
weights resident (wired, not plain mmap), leave headroom, and only one model server at a time.

## Baseline to beat (MLX-VLM 0.8.0+inkling2mini.2, DSpark block 4, temp 0.7, effort none, 400 tokens)

| context | prefill tok/s | gen tok/s |
|---|---|---|
| 1.2k | 229 (warm-up) | 23.6 |
| 9.5k | 602 | 24.8 |
| 18.9k | 571 | 23.3 |
| 38.6k | 502 | 19.9 |
| 78.1k | 385 | 16.5 |
| 127.5k | 324 | 13.8 |

- 128k OOMs with prefill step 2048 (at ~86k); fits with 512.
- --kv-bits 8 crashes MLX-VLM on this model (cache.py state: NoneType has no shape).
- Publisher numbers on M5 Max: 47.1 tok/s with DSpark (block 3), 36.6 without.
- Bench script: ~/llm/ctxbench.py URL MODEL SIZES. MLX server script: ~/llm/serve_inkling.sh.
- qh test repo: /tmp/inkling-qh-test (inventory.py with failing tests).

Rough decode ceiling: active weights read per token is a few GB, so with 366 GB/s well
above 50 tok/s is possible before drafts. Measure bytes per token first, as saetta did.

## Rules

- Accuracy test before any speed work; every optimization must keep the greedy match.
- Don't delete anything outside the repo. Don't touch ~/llm/mlx-models.
- Only one model loaded on the GPU at a time; coordinate before starting a server.
- Report measured numbers only; say "one run" when it is one run.
