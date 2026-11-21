#!/bin/bash
# Owner: C. Batch items (tools/WORKFLOW.md): server benchmarks on the freshly built main, last in the batch.
# Short set (every batch, ~10 min), on the server default model (8-bit): tools/depth_bench.py (THE target
# benchmark: gen tok/s at 1.2k / 9.5k / 38.6k / 78k / 127.5k vs the target line 60 -> 40 tok/s and MLX),
# agent_bench read (base 0) and copy scenarios, qh end-to-end x1.
# FULL=1 (on request, ~40 min more): cold-prefill ctxbench sweep to 127.5k and agent base 30000 for the 8-bit
# build, and all of it (depth, sweep, agent) for the lossless 4-8bit build; qh end-to-end x2.
# MLX=1 (on request, ~5 min): MLX-VLM baseline for the copy scenario if none is stored; MLX=all re-measures
# all of MLX's agent sessions (~20 min with FULL).
# Results: $OUT (default bench/queue-c), then the tables from tools/bench_report.py.
set -u
here=$(cd "$(dirname "$0")/../.." && pwd)
export OUT=${OUT:-$here/bench/queue-c}
mkdir -p "$OUT"
# MLX baseline measured 2026-11-21 (and BRIEF rows) for the ratio columns, unless this run re-measures it.
for f in "$here"/tests/server/baseline/*; do
	[ -e "$OUT/$(basename "$f")" ] || cp "$f" "$OUT/"
done
lossless=${LOSSLESS_MODEL:-$HOME/llm/mlx-models/Inkling-2-Mini-MLX-4-8bit}
q8=${Q8_MODEL:-$HOME/llm/mlx-models/Inkling-2-Mini-MLX-8bit}
all=1000,8000,16000,32000,64000,104000
# 1. The target benchmark on the server default (8-bit): gen tok/s at 1.2k..127.5k in one growing session
#    (one 127.5k prefill, ~6 min), then the agent sessions (read base 0, copy).
DEPTH=1 SKIP_CTX=1 AGENT_BASES=0 SERVER_ARGS="--model $q8 ${SERVER_ARGS:-}" "$here/tools/bench_c.sh" inkling inkling-8bit
if [ -n "${FULL:-}" ]; then
	# 2. FULL: cold-prefill sweep (ctxbench) and the 30k agent session for the 8-bit build ...
	SKIP_COPY=1 SIZES=$all AGENT_BASES=30000 SERVER_ARGS="--model $q8 ${SERVER_ARGS:-}" \
		"$here/tools/bench_c.sh" inkling inkling-8bit
	# ... and everything for the lossless build.
	DEPTH=1 SIZES=$all AGENT_BASES="0 30000" SERVER_ARGS="--model $lossless ${SERVER_ARGS:-}" \
		"$here/tools/bench_c.sh" inkling inkling
fi
# qh end-to-end needs a running server: start it again for the agent test only.
bin=${BIN:-$here/build/src/server/inkling-server}
[ -x "$bin" ] || bin=$here/build-c/src/server/inkling-server
# shellcheck disable=SC2086
"$bin" --port 8091 ${SERVER_ARGS:-} >"$OUT/inkling-serve-qh.log" 2>&1 &
pid=$!
for _ in $(seq 1 120); do curl -s -m 2 http://127.0.0.1:8091/v1/models >/dev/null && break; sleep 1; done
"$here/tools/qh_e2e.sh" queue "${FULL:+2}"
kill $pid 2>/dev/null; wait $pid 2>/dev/null
grep -E "diverged|no checkpoint|restore of|WARNING|rror" "$OUT"/inkling*serve*.log | cut -c1-160 | sed 's/^/cache: /'
if [ -n "${MLX:-}" ]; then
	# MLX=1: only what has no stored baseline in tests/server/baseline (the copy scenario, ~5 min).
	# MLX=all: also the read sessions again (base 0, and 30000 with FULL: ~4 + ~12 min).
	bases=none
	if [ "$MLX" = all ]; then bases=0; [ -n "${FULL:-}" ] && bases="0 30000"; fi
	[ -e "$OUT/agent-mlx-copy.json" ] && [ "$MLX" != all ] || SKIP_CTX=1 AGENT_BASES="$bases" "$here/tools/bench_c.sh" mlx
fi
"$here/tools/bench_c.sh" report
