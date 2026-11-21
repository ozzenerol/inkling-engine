#!/bin/bash
# Owner: C. Server benchmarks for inkling-server vs MLX-VLM, one server at a time on port 8091.
#   tools/bench_c.sh inkling   start build-c/src/server/inkling-server, measure it, stop it
#   tools/bench_c.sh mlx       start ~/llm/serve_inkling.sh (MLX-VLM + DSpark), measure it, stop it
#   tools/bench_c.sh report    print markdown tables from the JSON files
# Measures: ~/llm/ctxbench.py (BRIEF baseline: temp 0.7, effort none, 400 tokens) and saetta's tools/bench.py
# (temp 0, thinking off, contexts + a 10-turn conversation through the prompt cache).
# Announce the GPU lock in NOTES.md and to the other agents before running inkling or mlx.
# Env: SIZES (ctxbench, default 1000,8000,16000,32000,64000,104000), CONTEXTS (bench.py, default 40,1k,4k,16k,32k),
#      RUNS (3), TURNS (10), OUT (bench/), SERVER_ARGS (extra inkling-server flags), SKIP_CTX=1, SKIP_BENCH=1.
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
	model=$(curl -s "$url/v1/models" | "$py" -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
	if [ -z "${SKIP_CTX:-}" ]; then
		echo "== ctxbench $label ($sizes)"
		"$py" -u "$HOME/llm/ctxbench.py" "$url/v1" "$model" "$sizes" | tee "$out/ctx-$label.jsonl"
	fi
	if [ -z "${SKIP_BENCH:-}" ]; then
		echo "== bench.py $label ($contexts, $turns turns)"
		"$py" -u "$HOME/projects/saetta-macqwen/tools/bench.py" --url "$url" --label "$label" \
			--contexts "$contexts" --runs "$runs" --turns "$turns" --out "$out/bench-$label.json"
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
	start_and_wait "$out/inkling-serve.log" "$here/build-c/src/server/inkling-server" --port 8091 ${SERVER_ARGS:-}
	measure inkling
	;;
mlx)
	up && { echo "something already serves $url; stop it first"; exit 1; }
	start_and_wait "$out/mlx-serve.log" "$HOME/llm/serve_inkling.sh"
	measure mlx
	;;
report)
	for f in "$out"/ctx-*.jsonl; do
		[ -f "$f" ] || continue
		echo "### ctxbench $(basename "$f" .jsonl | sed 's/^ctx-//')"
		echo "| prompt tokens | prefill tok/s | gen tok/s |"
		echo "|---|---|---|"
		"$py" -c 'import json,sys
for l in open(sys.argv[1]):
    d=json.loads(l)
    print("| %s | %s | %s |" % (d.get("prompt_tokens","-"), d.get("prefill_tps", d.get("error","-")), d.get("gen_tps","-")))' "$f"
		echo
	done
	ls "$out"/bench-*.json >/dev/null 2>&1 && "$py" "$HOME/projects/saetta-macqwen/tools/bench.py" --report "$out"/bench-*.json
	;;
*)
	sed -n 2,11p "$0"; exit 2 ;;
esac
