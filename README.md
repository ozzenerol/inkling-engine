# inkling-engine

An inference engine for Inkling 2 Mini on Apple Silicon, written from scratch in C++ with Metal kernels. It serves
an OpenAI-compatible API for agentic coding (qh) and is built to be faster than MLX-VLM with the same output.

Status and measured numbers are in `NOTES.md`. This file is the short version.

## What it runs

| build | directory | notes |
|---|---|---|
| 4/8-bit (lossless) | `~/llm/mlx-models/Inkling-2-Mini-MLX-4-8bit` | The release as it is: experts NVFP4 (layers 0-19) and MXFP8 (20-23), everything else bf16. Default. |
| 8-bit | `~/llm/mlx-models/Inkling-2-Mini-MLX-8bit` | Same experts; attention, embeddings and head in MLX affine 8-bit (about 0.7% weight error). Smaller dense reads per token. |

The DSpark draft model (`mtp.safetensors`) comes with the 4/8-bit build and is shared by both.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Needs clang++ (C++20), cmake, and pcre2 from Homebrew for the tokenizer. Metal kernels are compiled at start from
source embedded in the binary, so no Xcode is needed.

## Run

```sh
build/src/server/inkling-server --port 8091                 # 4/8-bit build
build/src/server/inkling-server --port 8091 --model ~/llm/mlx-models/Inkling-2-Mini-MLX-8bit
```

`/v1/chat/completions` with streaming, `tool_calls`, `reasoning_content`, `reasoning_effort`, and a prefix cache
that survives across turns (an agent turn prefills only the new tokens). qh's `~/.qh/config.json` points at 8091.

Only one model server on the GPU at a time (48 GB machine; see the memory warning in `BRIEF.md`).

## Layout

| path | what |
|---|---|
| `src/core` | JSON, safetensors index, config, thread pool |
| `src/model` | forward pass over an abstract `Backend` (`backend.h`), CPU reference backend |
| `src/gpu` | Metal backend and kernels (`kernels/*.metal`), DSpark drafter, kernel benches and `op_diff` |
| `src/engine` | prefill, sampling, checkpoints, speculative decoding |
| `src/tokenizer`, `src/server` | BPE tokenizer, chat template, tool-call parser, HTTP server |
| `tests/ref` | reference dumps from MLX-VLM and the accuracy checks |
| `tools` | benchmarks (`bench_c.sh`, `agent_bench.py`), `spec_check.sh` |

## Accuracy checks

Every speed change must keep the greedy output of the MLX-VLM reference.

```sh
~/llm/inkling-venv/bin/python tests/ref/dump_ref.py ref_data 64                           # MLX reference (bf16)
~/llm/inkling-venv/bin/python tests/ref/dump_ref.py ref_data/fp32 64 --fp32 --gen-from ref_data   # fp32 activations
tests/ref/acc_check.sh metal            # greedy match and KL against both, prompts 0-3
MODEL=~/llm/mlx-models/Inkling-2-Mini-MLX-8bit tests/ref/acc_check.sh metal
build/gpu/op_diff                       # every Metal op against the CPU reference backend
tools/spec_check.sh ref_data/p2.ids 200 4   # DSpark on and off give the same tokens
build/inktest --backend cpu --ids ref_data/p2.ids --rollback-test --out /tmp   # speculative rollback (INK_CPU_ROWEXACT=1)
~/llm/inkling-venv/bin/python tests/ref/check_8bit_layout.py   # 8-bit build against the lossless one, tensor by tensor
```

## Switches (environment)

| variable | effect |
|---|---|
| `INK_BACKEND=cpu` | run the engine on the CPU reference backend (about 3 tok/s) |
| `INK_NO_DSPARK=1` | plain decoding, no drafts |
| `INK_Q8=0` | 8-bit build: convert dense weights to bf16 at load instead of reading the codes in place |
| `INK_AG_MIN_T=n` | use the matrix-product prefill attention from n tokens per chunk (default 64) |
| `INK_PROFILE=1` | GPU time per kernel (each dispatch waited separately) |
| `INK_SPEC_STATS=1` | drafts accepted per round, draft and verify time |
