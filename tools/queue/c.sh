#!/bin/bash
# Owner: C. Batch items (tools/WORKFLOW.md): server benchmarks on the freshly built main, last in the batch.
# Short set (every batch, ~4 min): ctxbench at 1.2k and 9.5k, agent_bench base 0 (10 turns), qh end-to-end x1.
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
	SIZES=1000,8000,16000,32000,64000,104000 AGENT_BASES="0 30000" "$here/tools/bench_c.sh" inkling
else
	SIZES=1000,8000 AGENT_BASES="0" "$here/tools/bench_c.sh" inkling
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
grep -E "diverged|no checkpoint|restore of" "$OUT"/inkling-serve*.log | cut -c1-160 | sed 's/^/cache: /'
if [ -n "${MLX:-}" ]; then
	bases=0; [ -n "${FULL:-}" ] && bases="0 30000"
	SKIP_CTX=1 AGENT_BASES="$bases" "$here/tools/bench_c.sh" mlx
fi
"$here/tools/bench_c.sh" report
