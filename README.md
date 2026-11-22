# inkling-engine

> **Work in progress, a test.** This whole engine was vibecoded with Claude Fable 5.2: an AI model wrote the
> code, the kernels, the benchmarks and these notes, steered by short messages. Nobody has reviewed it line by
> line. It runs on one machine (an M3 Max with 48 GB) and has been measured only there, mostly one run per number.
> Expect bugs and rough edges, and do not rely on it for anything that matters.

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
build/src/server/inkling-server --port 8091                 # 8-bit build (the default model)
build/src/server/inkling-server --port 8091 --model ~/llm/mlx-models/Inkling-2-Mini-MLX-4-8bit   # lossless 4/8-bit
```

Without `--model` the server loads `~/llm/mlx-models/Inkling-2-Mini-MLX-8bit` (and falls back to the 4/8-bit build
if that directory is missing). `/v1/models` reports the directory in use.

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
| `INK_WARM_SECS=N` | keep the weights wired for N seconds after the last work (default 600; 0 off). Without it the first submit after a 2 s pause waits about 0.4 s in the driver |
| `INK_CBTIME=1` | print scheduling and GPU time of every forward that takes over 50 ms |
| `INK_SPEC_STATS=1` | drafts accepted per round, draft and verify time |
| `INK_SPEC_GATE=0` | draft count follows past acceptance instead of the drafter's confidence (the default cuts each draft where its expected tokens per ms peak) |
| `INK_SPEC_DUMP=file` | append `asked position margin accepted` for every drafted token (to recalibrate the gate) |
| `INK_DSPARK_TIME=1` | wall time of the parts of a draft (layers, head, token steps); `2` also per dispatch |
