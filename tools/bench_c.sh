#!/bin/bash
# Owner: C. Server benchmarks for inkling-server vs MLX-VLM, one server at a time on port 8091.
#   tools/bench_c.sh inkling   start build-c/src/server/inkling-server, measure it, stop it
#   tools/bench_c.sh mlx       start ~/llm/serve_inkling.sh (MLX-VLM + DSpark), measure it, stop it
#   tools/bench_c.sh report    print markdown tables from the JSON files
# Measures: ~/llm/ctxbench.py (BRIEF baseline: temp 0.7, effort none, 400 tokens) and tools/agent_bench.py
# (qh-like session, 10 turns, with base context 0 and 30000: per-turn TTFT and total wall time).
# Announce the GPU lock in NOTES.md and to the other agents before running inkling or mlx.
# Env: SIZES (ctxbench, default 1000,8000,16000,32000,64000,104000), CONTEXTS (bench.py, default 40,1k,4k,16k,32k),
#      TURNS (10), AGENT_BASES ("0 30000"), OUT (bench/), SERVER_ARGS (extra inkling-server flags),
#      BIN (server binary, default build-c/src/server/inkling-server), SKIP_CTX=1, SKIP_AGENT=1.
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
py=$HOME/llm/inkling-venv/bin/python
sizes=${SIZES:-1000,8000,16000,32000,64000,104000}
contexts=${CONTEXTS:-40,1k,4k,16k,32k}
runs=${RUNS:-3}
turns=${TURNS:-10}
out=${OUT:-$here/bench}
url=http://127.0.0.1:8091
mkdir -p "$out"

up() { curl -s -m 2 "$url/v1/models" >/dev/null; }

measure() {
	label=$1
	model=$(curl -s "$url/v1/models" | "$py" -c 'import json,sys,os; d=json.load(sys.stdin)["data"]; print(d[0]["id"] if d else os.path.expanduser("~/llm/mlx-models/Inkling-2-Mini-MLX-4-8bit"))')
	echo "== warm-up (kernel compile, first allocations)"
	curl -s -m 900 "$url/v1/chat/completions" -H 'Content-Type: application/json' \
		-d '{"messages":[{"role":"user","content":"Say hi."}],"max_tokens":32,"chat_template_kwargs":{"reasoning_effort":"none"}}' >/dev/null
	if [ -z "${SKIP_CTX:-}" ]; then
		echo "== ctxbench $label ($sizes)"
		"$py" -u "$HOME/llm/ctxbench.py" "$url/v1" "$model" "$sizes" | tee "$out/ctx-$label.jsonl"
	fi
	if [ -z "${SKIP_AGENT:-}" ]; then
		for base in ${AGENT_BASES:-0 30000}; do
			echo "== agent_bench $label (base $base, $turns turns)"
			"$py" -u "$here/tools/agent_bench.py" --url "$url" --label "$label" --turns "$turns" \
				--base-tokens "$base" --out "$out/agent-$label-$base.json"
		done
	fi
}

start_and_wait() {
	log=$1; shift
	"$@" >"$log" 2>&1 &
	pid=$!
	trap 'pkill -P $pid 2>/dev/null; kill $pid 2>/dev/null; wait $pid 2>/dev/null' EXIT
	for _ in $(seq 1 900); do
		up && return 0
		kill -0 $pid 2>/dev/null || { echo "server died:"; tail -20 "$log"; exit 1; }
		sleep 1
	done
	echo "server did not come up"; exit 1
}

case "${1:-}" in
inkling)
	up && { echo "something already serves $url; stop it first"; exit 1; }
	# shellcheck disable=SC2086
	start_and_wait "$out/inkling-serve.log" "${BIN:-$here/build-c/src/server/inkling-server}" --port 8091 ${SERVER_ARGS:-}
	measure inkling
	;;
mlx)
	up && { echo "something already serves $url; stop it first"; exit 1; }
	start_and_wait "$out/mlx-serve.log" "$HOME/llm/serve_inkling.sh"
	measure mlx
	;;
report)
	"$py" "$here/tools/bench_report.py" "$out"
	;;
*)
	sed -n 2,11p "$0"; exit 2 ;;
esac
