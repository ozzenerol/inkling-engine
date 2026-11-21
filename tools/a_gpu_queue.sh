#!/bin/bash
# A's pending GPU checks (owner A). Needs the GPU lock. Usage: tools/a_gpu_queue.sh [steps: all | chain spec q8 long]
cd "$(dirname "$0")/.."
S=${S:-/tmp/claude-501/aq}; mkdir -p $S
PY=~/llm/inkling-venv/bin/python
M8=~/llm/mlx-models/Inkling-2-Mini-MLX-8bit
steps=${*:-chain spec q8 long}
cmake --build build -j >/dev/null 2>&1
./build/kernel_check | grep -v "^ok" && { echo "kernel_check failed"; exit 1; }
for st in $steps; do
case $st in
chain)
  echo "== chain kernels: accuracy and decode speed, on vs off (teacher-forced, plain decode)"
  for v in on off; do
    [ $v = on ] && export INK_CHAIN=1 || unset INK_CHAIN
    echo "-- chain $v"; ACC_OUT=$S/chain_$v tests/ref/acc_check.sh metal 0 1 2 2>&1 | grep -E "decode|vs|mismatch"
  done
  unset INK_CHAIN
  for i in 0 1 2; do cmp -s $S/chain_on/p$i/argmax.i32 $S/chain_off/p$i/argmax.i32 && echo "p$i argmax identical on/off" || echo "p$i argmax DIFFERS on/off"; done
  ./build/inktest --backend metal --ids ref_data/p2.ids --rollback-test --out $S 2>&1 | tail -1
  ;;
spec)
  echo "== DSpark: greedy identity (chain on), then temperature 0.7"
  tools/spec_check.sh ref_data/p2.ids 200 4 2>&1 | grep -E "tok/s|IDENTICAL|DIFFER|dspark"
  for v in "plain" "dspark" "dspark INK_SPEC_COMPARE=1" "dspark INK_SPEC_FIXED=1" "dspark INK_SPEC_COMPARE=1 INK_SPEC_FIXED=1"; do
    set -- $v; mode=$1; shift
    echo "-- temp 0.7: $v"
    env INK_SPEC_STATS=1 "$@" ./build/src/gpu/spec_check $mode ref_data/p2.ids 300 4 $S/t07.i32 0.7 2>&1 | grep -E "tok/s|dspark:|adaptive"
  done
  ;;
q8)
  echo "== 8-bit build: load-time conversion, then q8 kernels"
  MODEL=$M8 ACC_OUT=$S/q8conv tests/ref/acc_check.sh metal 0 1 2 2>&1 | grep -E "prefill|decode|vs|mismatch"
  INK_Q8_KERNELS=1 MODEL=$M8 ACC_OUT=$S/q8kern tests/ref/acc_check.sh metal 0 1 2 2>&1 | grep -E "prefill|decode|vs|mismatch"
  INK_Q8_KERNELS=1 INK_MODEL=$M8 INK_SPEC_STATS=1 ./build/src/gpu/spec_check plain ref_data/p2.ids 200 4 $S/q8.i32 2>&1 | grep -E "tok/s"
  ;;
long)
  echo "== long prompt set (8 prompts, 2k-6k tokens, 32 steps): references, then GEMM vs flash attention vs MLX-bf16"
  [ -f ref_data/p17.npz ] || $PY -I tests/ref/dump_ref.py ref_data 32 --set long 2>&1 | grep -c prompt
  [ -f ref_data/fp32/p17.npz ] || $PY -I tests/ref/dump_ref.py ref_data/fp32 32 --set long --fp32 --gen-from ref_data 2>&1 | grep -c prompt
  L="10 11 12 13 14 15 16 17"
  STEPS=32 ACC_OUT=$S/long_gemm tests/ref/acc_check.sh metal $L 2>&1 | grep -E "prefill"
  INK_AG_MIN_T=1000000 STEPS=32 ACC_OUT=$S/long_flash tests/ref/acc_check.sh metal $L >/dev/null 2>&1
  MODEL=$M8 STEPS=32 ACC_OUT=$S/long_q8 tests/ref/acc_check.sh metal $L >/dev/null 2>&1
  $PY -I tests/ref/acc_summary.py "$L" mlx-bf16=mlx gemm=$S/long_gemm flash=$S/long_flash 8bit=$S/long_q8
  ;;
esac
done
