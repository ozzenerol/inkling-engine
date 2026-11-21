# Workflow (from 2026-11-21 evening): merge first, then one test batch

The user's rule: work together, and merge all the work before running tests.

## Branches and worktrees

Each agent works on its own branch in its own worktree, so nobody's half-done edits end up in someone else's build
(the Metal kernels are embedded from the working tree at build time).

    git worktree add ~/projects/inkling-engine/.wt/<letter> -b <letter>/work main     # once
    cd ~/projects/inkling-engine/.wt/<letter>
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
    build/kernel_check <your kernels>      # compiles the Metal library; no model, no GPU load
    git commit ...                         # small commits, plain one-line messages, no AI attribution

Keep your branch close to main: `git merge main` (or rebase) whenever the integrator announces a merge.

CPU-only checks are free at any time: build, kernel_check, sample_test, `inktest --backend cpu` (short prompts
only: it takes 12 cores, so not while a batch is running).

## Merge and test batch (integrator: A)

1. An agent with something ready messages A: "ready: <branch> <what to measure>" and appends its GPU commands to
   `tools/queue/<letter>.sh` on its branch (plain shell lines, each printing its own numbers).
2. A merges every ready branch into main (`git merge --no-ff`), resolves conflicts with the owner if needed, builds
   main, runs `build/kernel_check`, then ONE GPU batch on main: `tools/batch.sh`:
   op_diff, acc_check (lossless and 8-bit), spec_check identity, then every `tools/queue/*.sh`.
3. A sends each agent its part of the log (`bench/batch-N/<letter>.log`) and the pass/fail of the common gates.
   A change that breaks a gate is reverted on main by A and goes back to its owner.

Nobody else loads the model on the GPU. No lock line to maintain: the batch is the only GPU user.
The benchmark runs (C: ctxbench, agent bench, MLX baseline) are batch items too, last in the batch.

## Gates (every batch)

- `build/kernel_check` passes.
- `build/gpu/op_diff`: no FAIL.
- `tests/ref/acc_check.sh metal 0 1 2`: greedy mismatches only at near-ties (reference top-2 gap < 0.2) or after
  the stop token; KL vs fp32 not above MLX-bf16's (NOTES, "A: fp32 reference verdict").
- `tools/spec_check.sh ref_data/p2.ids 200 4`: token streams identical.
- 8-bit build: same acc_check with MODEL=~/llm/mlx-models/Inkling-2-Mini-MLX-8bit.

## Target (user, 2026-11-21 evening)

Generation **60 tok/s at the start of the context and 40 tok/s at 128k**, on free text (where drafts help
least), 8-bit build (the server default). C's target table (gen tok/s at 1.2k .. 127.5k against that line) is part
of every batch.

## Roles

| agent | session | area |
|---|---|---|
| A | inkling-engine-9e | coordinator, integrator, accuracy; drafting policy (drafter + prompt lookup), KV-cache accuracy |
| B | ozzenerol-5f | the plain step at short context: 23.6 ms -> 17-20 ms on the 8-bit build (fusion, MoE and q8 readers) |
| C | ozzenerol-aa | server, benchmarks (target table every batch), qh end-to-end, docs/RESULTS.md |
| D | ozzenerol-a5 | multi-row forwards (done: 3..16 rows, expert-union MoE, q8) and attention at long context (32k..128k) |
