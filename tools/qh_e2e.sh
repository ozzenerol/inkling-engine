#!/bin/bash
# Owner: C. End-to-end agent test: qh (instance ~/.qh-inkling -> :8091) fixes /tmp/inkling-qh-test in a fresh
# clone, then the unit tests decide. Needs a server already on :8091 (inkling-server or MLX).
#   tools/qh_e2e.sh LABEL [N_RUNS]     results in bench/qh-LABEL/run-K/{qh.out,tests.out}
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
label=${1:?label}; n=${2:-1}
out=$here/bench/qh-$label
mkdir -p "$out"
curl -s -m 2 http://127.0.0.1:8091/v1/models >/dev/null || { echo "no server on :8091"; exit 1; }
task='The unit tests in test_inventory.py fail. Fix inventory.py so that all tests pass (run them with python3 -m unittest). Do not change the tests.'
for k in $(seq 1 "$n"); do
	d=$out/run-$k
	rm -rf "$d"; git clone -q /tmp/inkling-qh-test "$d/repo"
	t0=$(date +%s)
	timeout 1800 qh -i inkling run -new -s "e2e-$label-$k" -C "$d/repo" -timeout 25m "$task" >"$d/qh.out" 2>&1
	rc=$?
	t1=$(date +%s)
	(cd "$d/repo" && python3 -m unittest -q >"$d/tests.out" 2>&1); trc=$?
	ok=$(grep -Eo "Ran [0-9]+ tests" "$d/tests.out")
	echo "run $k: qh exit $rc, $((t1 - t0)) s, tests $([ $trc = 0 ] && echo PASS || echo FAIL) ($ok) | $(tail -1 "$d/qh.out" | cut -c1-200)" | tee -a "$out/summary.txt"
done
qh -i inkling stop >/dev/null 2>&1
