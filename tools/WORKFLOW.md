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

## Roles

| agent | session | area |
|---|---|---|
| A | inkling-engine-9e | coordinator, integrator, accuracy; speculative policy (prompt-lookup drafts) |
| B | ozzenerol-5f | plain decode cost per forward (dispatch fusion), q8 kernels as default, prefill MoE GEMM |
| C | ozzenerol-aa | server, benchmarks (batch items), qh end-to-end |
| D | ozzenerol-a5 | multi-row forwards: verify at T = 5..16 at ~plain + 6-7 ms per extra row; drafter cost 8 ms -> 3 ms |
