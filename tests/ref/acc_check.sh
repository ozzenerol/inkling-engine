#!/bin/bash
# Owner: A. Accuracy regression check for any speed change (needs the GPU lock for metal).
#   tests/ref/acc_check.sh [metal|cpu] [prompts, default "0 1 2 3"]
# For each prompt: teacher-forced inktest run, then compare against MLX-bf16 (greedy/KL, mismatch margins) and the
# fp32 reference (KL; should not be worse than MLX-bf16's, see NOTES "A: fp32 reference verdict").
set -e
cd "$(dirname "$0")/../.."
BE=${1:-metal}; shift || true
PROMPTS=${*:-0 1 2 3}
PY=~/llm/inkling-venv/bin/python
OUT=${ACC_OUT:-/tmp/claude-501/acc_$BE}
cmake --build build -j --target inktest >/dev/null
for i in $PROMPTS; do
  mkdir -p $OUT/p$i
  ./build/inktest --backend $BE --ids ref_data/p$i.ids --gen ref_data/p$i.gen --steps 64 --ctx 8192 --out $OUT/p$i 2>&1 | grep -E "prefill|decode" | sed "s/^/p$i /"
  echo "p$i vs MLX-bf16: $($PY -I tests/ref/compare.py ref_data/p$i.npz $OUT/p$i | head -1)"
  $PY -I tests/ref/compare.py ref_data/p$i.npz $OUT/p$i | grep mismatch\ step || true
  echo "p$i vs fp32:     $($PY -I tests/ref/compare.py ref_data/fp32/p$i.npz $OUT/p$i | head -1)"
done
