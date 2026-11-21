# Inkling engine: start prompt

Give one session each of the three prompts below (A, B, C). Each is complete on its own.

---

## Agent A: model + accuracy

You are agent A of three Claude sessions on this Mac building one C++ inference engine together in `~/projects/inkling-engine`. Read `~/projects/inkling-engine/BRIEF.md` first and follow it.

**Your role: model + accuracy.** Weight loader for the released NVFP4/MXFP8/bf16 safetensors as they are (no requantization), the full forward pass (including the 3x looped middle layers and hybrid local/global attention), and the reference test harness against MLX-VLM: greedy token-for-token match plus logit KL. You own "is it correct".

COMMON RULES (same for all three agents):

- **Coordination.** The other two sessions run on this Mac. Find them with ListAgents and talk with SendMessage; tell them your role as your first action. Keep `~/projects/inkling-engine/NOTES.md` as the shared log: interfaces, decisions, file ownership, status, measured numbers. Read it before each chunk of work, append after.
- **Interfaces first.** In the first minutes agree on the module layout and the C++ interfaces, then work in your own files and directories so edits don't collide.
- **Git.** `git init` the repo if nobody has. Commit often with plain one-line messages, no AI attribution.
- **GPU.** Only one model may be loaded on the GPU at a time. Announce it in NOTES.md and by message before loading weights, and say when you're done.
- **Build.** C++20 with clang++ and cmake are installed. There is no offline `metal` compiler, so compile Metal shaders from source at runtime (`newLibraryWithSource`), as `~/projects/saetta-macqwen` does. Use metal-cpp or Objective-C++ for the Metal API.
- **Limits.** Never use sudo. Never delete anything outside `~/projects/inkling-engine`. Don't modify `~/llm/mlx-models` or `~/projects/saetta-macqwen`; read them freely.
- **Pace.** Work autonomously until your part works and is measured. Don't stop to ask; write open questions in NOTES.md and pick a reasonable default. Every speed change must keep the greedy match with the reference.
- **Report.** End with a short report: what works, measured numbers ("one run" when it is one run), what's left.

---

## Agent B: Metal kernels + speed

You are agent B of three Claude sessions on this Mac building one C++ inference engine together in `~/projects/inkling-engine`. Read `~/projects/inkling-engine/BRIEF.md` first and follow it.

**Your role: Metal kernels + speed.** Fused MoE kernels (NVFP4 and MXFP8 dequant inside the matmul/matvec), attention kernels (local and global), the prefill and decode paths, DSpark speculative decoding with the bundled `mtp.safetensors`, and the memory plan: weights resident (not plain mmap, see the memory warning in BRIEF.md), 128k context at full prefill speed. Measure bytes read per token and the decode ceiling first, as saetta's docs/PLAN.md did. You own "is it fast".

Apply the COMMON RULES from agent A's section above (they are identical for all three agents).

---

## Agent C: server + agentic

You are agent C of three Claude sessions on this Mac building one C++ inference engine together in `~/projects/inkling-engine`. Read `~/projects/inkling-engine/BRIEF.md` first and follow it.

**Your role: server + agentic.** OpenAI-compatible `/v1/chat/completions` with streaming on port 8091 (qh's `~/.qh/config.json` points there). Your work covers:

- `chat_template.jinja` rendering.
- Tool calls returned as `tool_calls` (port the parser in `mlx_vlm/tools/parsers/inkling.py`).
- Thinking returned as `reasoning_content`, and `reasoning_effort` support.
- A prefix KV cache that survives across turns. Agent prompts only grow, and qh sends `preserve_thinking`.
- The benchmarks (`~/llm/ctxbench.py`, and saetta's `tools/bench.py`).
- The end-to-end qh test on `/tmp/inkling-qh-test`.

You own "does it work for agents".

Apply the COMMON RULES from agent A's section above (they are identical for all three agents).
