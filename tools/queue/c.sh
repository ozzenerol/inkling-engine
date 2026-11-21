#!/bin/bash
# Owner: C. Batch items (tools/WORKFLOW.md): server benchmarks on the freshly built main, last in the batch.
# Short set (every batch, ~8 min): ctxbench at 1.2k and 9.5k and agent_bench base 0 (10 turns), for the lossless
# 4-8bit build and for the 8-bit build, plus the copy-heavy agent scenario; then qh end-to-end x1 on the
# server's default model (8-bit).
# FULL=1 (on request, ~15 min): ctxbench 1.2k..127.5k, agent_bench base 0 and 30000, qh end-to-end x2.
# MLX=1 (on request, ~20 min more): the same agent_bench against MLX-VLM for a same-day baseline.
# Results: $OUT (default bench/queue-c), then the tables from tools/bench_report.py.
set -u
here=$(cd "$(dirname "$0")/../.." && pwd)
export OUT=${OUT:-$here/bench/queue-c}
mkdir -p "$OUT"
# MLX baseline measured 2026-11-21 (and BRIEF rows) for the ratio columns, unless this run re-measures it.
for f in "$here"/tests/server/baseline/*; do
	[ -e "$OUT/$(basename "$f")" ] || cp "$f" "$OUT/"
done
if [ -n "${FULL:-}" ]; then
	sizes=1000,8000,16000,32000,64000,104000; bases="0 30000"
else
	sizes=1000,8000; bases="0"
fi
# Lossless 4/8-bit build (label inkling), then the 8-bit build, the server default (label inkling-8bit).
# SKIP_LOSSLESS=1 / SKIP_Q8=1 skip one of them.
lossless=${LOSSLESS_MODEL:-$HOME/llm/mlx-models/Inkling-2-Mini-MLX-4-8bit}
q8=${Q8_MODEL:-$HOME/llm/mlx-models/Inkling-2-Mini-MLX-8bit}
if [ -z "${SKIP_LOSSLESS:-}" ]; then
	SERVER_ARGS="--model $lossless ${SERVER_ARGS:-}" SIZES=$sizes AGENT_BASES=$bases "$here/tools/bench_c.sh" inkling
fi
if [ -z "${SKIP_Q8:-}" ] && [ -d "$q8" ]; then
	SERVER_ARGS="--model $q8 ${SERVER_ARGS:-}" SIZES=$sizes AGENT_BASES=$bases \
		"$here/tools/bench_c.sh" inkling inkling-8bit
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
	bases=0; [ -n "${FULL:-}" ] && bases="0 30000"
	SKIP_CTX=1 AGENT_BASES="$bases" "$here/tools/bench_c.sh" mlx
fi
"$here/tools/bench_c.sh" report
