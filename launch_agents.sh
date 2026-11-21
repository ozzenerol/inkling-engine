#!/bin/bash
# Start three headless Claude agents on the Inkling engine, one per role.
export PATH=$HOME/.local/bin:/opt/homebrew/bin:$PATH
R=$HOME/projects/inkling-engine
mkdir -p "$R/logs"
cd "$R"
[ -d .git ] || { git init -q; printf 'build/\nlogs/\n*.o\n' > .gitignore; }

for role in A B C; do
  case $role in
    A) what="model + accuracy (weight loader, forward pass, reference test against MLX-VLM)";;
    B) what="Metal kernels + speed (fused NVFP4/MXFP8 MoE, attention, prefill/decode, DSpark, memory plan)";;
    C) what="server + agentic (OpenAI API on :8091, chat template, tool_calls, reasoning_content, cross-turn prefix cache, benchmarks)";;
  esac
  prompt="You are agent $role of three headless Claude agents (inkling-A, inkling-B, inkling-C) building one C++ inference engine together in $R. Read $R/BRIEF.md first and follow it. Your role: $what.

Coordination:
- Use ListAgents and SendMessage to talk to inkling-A/B/C. Also keep $R/NOTES.md as the shared log: interfaces you define, decisions, who owns which files, status, measured numbers. Read it before you start each chunk of work, append to it after.
- Agree on the module layout and the C++ interfaces in your first minutes, then work in your own files and directories to avoid edit conflicts. Commit often with plain one-line messages (git in $R); no AI attribution.
- Only one model may be loaded on the GPU at a time: announce in NOTES.md and by message before you load weights on the GPU, and say when you are done.

Build notes: C++20 with clang++ and cmake are installed. There is no offline metal compiler, so compile Metal shaders from source at runtime (newLibraryWithSource), as ~/projects/saetta-macqwen does. Use metal-cpp or Objective-C++ for the Metal API.

Never use sudo. Never delete anything outside $R. Do not modify ~/llm/mlx-models or ~/projects/saetta-macqwen (read them freely).

Work autonomously and keep going until your part works and is measured; do not stop to ask questions, write them in NOTES.md and choose a reasonable default. Finish with a short report of what works, the measured numbers, and what is left."
  nohup claude -p "$prompt" --name "inkling-$role" \
    --permission-mode acceptEdits \
    --allowedTools "Bash Read Edit Write Glob Grep ListAgents SendMessage WebFetch WebSearch TodoWrite" \
    --disallowedTools "Bash(sudo *)" \
    --add-dir "$HOME/llm" "$HOME/projects/saetta-macqwen" \
    --output-format stream-json --verbose \
    > "$R/logs/agent-$role.jsonl" 2> "$R/logs/agent-$role.err" < /dev/null &
  echo "agent $role pid $!"
  sleep 5
done
