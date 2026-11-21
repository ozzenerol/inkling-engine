#!/bin/bash
# One GPU test batch on the main checkout (integrator: A). See tools/WORKFLOW.md.
#   tools/batch.sh [gates] [queues] [full]     default: gates queues
cd "$(dirname "$0")/.."
# Never test a half-merged or edited tree: everything must be committed (the user's rule: merge, then test).
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "batch refused: the main checkout has uncommitted or unmerged files:"; git status --short --untracked-files=no; exit 1
fi
N=$(( $(ls -d bench/batch-* 2>/dev/null | wc -l) + 1 ))
OUT=bench/batch-$N; mkdir -p $OUT
what=${*:-gates queues}
PY=~/llm/inkling-venv/bin/python
M8=~/llm/mlx-models/Inkling-2-Mini-MLX-8bit
echo "batch $N at $(git rev-parse --short HEAD) $(date +%H:%M)" | tee $OUT/summary.log
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null && cmake --build build -j 2>&1 | grep -E "rror" | head -20 | tee -a $OUT/summary.log
(cmake -S src/gpu -B build/gpu -DCMAKE_BUILD_TYPE=Release >/dev/null && cmake --build build/gpu -j >/dev/null 2>&1)
gate() { name=$1; shift; echo "== gate $name" | tee -a $OUT/summary.log; "$@" > $OUT/gate-$name.log 2>&1; }
if [[ " $what " == *" gates "* ]]; then
  gate kernel_check ./build/kernel_check; grep -v "^ok" $OUT/gate-kernel_check.log | head -5 | tee -a $OUT/summary.log
  if grep -q "^FAIL\|error" $OUT/gate-kernel_check.log; then echo "kernel_check FAILED: batch stopped" | tee -a $OUT/summary.log; exit 1; fi
  gate op_diff ./build/gpu/op_diff; grep -E "FAIL|ALL" $OUT/gate-op_diff.log | tee -a $OUT/summary.log
  gate op_diff_q8 env INK_Q8_KERNELS=1 ./build/gpu/op_diff; grep -E "FAIL|ALL" $OUT/gate-op_diff_q8.log | tee -a $OUT/summary.log
  gate acc env ACC_OUT=$OUT/acc tests/ref/acc_check.sh metal 0 1 2; grep -E "decode|vs|mismatch" $OUT/gate-acc.log | tee -a $OUT/summary.log
  gate acc8 env MODEL=$M8 INK_Q8_KERNELS=1 ACC_OUT=$OUT/acc8 tests/ref/acc_check.sh metal 0 1 2; grep -E "decode|vs|mismatch" $OUT/gate-acc8.log | tee -a $OUT/summary.log
  gate spec tools/spec_check.sh ref_data/p2.ids 200 4; grep -E "tok/s|IDENTICAL|DIFFER" $OUT/gate-spec.log | tee -a $OUT/summary.log
  gate rollback ./build/inktest --backend metal --ids ref_data/p2.ids --rollback-test --out $OUT; tail -1 $OUT/gate-rollback.log | tee -a $OUT/summary.log
fi
if [[ " $what " == *" queues "* ]]; then
  # C's server benchmarks run last (they are the longest and want a quiet machine)
  for q in $(ls tools/queue/*.sh | grep -v '/c.sh$') $(ls tools/queue/c.sh 2>/dev/null); do
    [ -f "$q" ] || continue
    l=$(basename $q .sh); echo "== queue $l" | tee -a $OUT/summary.log
    FULL=$([[ " $what " == *" full "* ]] && echo 1) OUT=$OUT bash $q > $OUT/$l.log 2>&1; echo "   exit $? ($(wc -l < $OUT/$l.log) lines in $OUT/$l.log)" | tee -a $OUT/summary.log
  done
fi
echo "batch $N done $(date +%H:%M)" | tee -a $OUT/summary.log
