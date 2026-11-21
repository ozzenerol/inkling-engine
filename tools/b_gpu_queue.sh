#!/bin/zsh
# B's pending GPU checks (owner B): candidates behind flags -> measure, then flip defaults.
cd "$(dirname "$0")/.."
S=${S:-/tmp/claude-501/bq}; mkdir -p $S
echo "== op_diff default / all candidates on"
./build/gpu/op_diff 2>&1 | grep -E "ALL|FAILED|FAIL "
INK_Q8_KERNELS=1 INK_CTRL_GEMM=1 INK_MV_T8=1 INK_MOE_GEMM2=1 INK_FP4_LUT=1 ./build/gpu/op_diff 2>&1 | grep -E "q8|controller|ALL|FAILED|FAIL "
echo "== matvec T=8: two 4-row passes vs one-pass t8tg"
./build/gpu/bench_matvec 4 | grep GB/s
T8TG=1 ./build/gpu/bench_matvec 8 | grep GB/s
echo "== MoE decode: nibble decode vs LUT"
./build/gpu/bench_moe 1 | grep nvfp4
INK_FP4_LUT=1 ./build/gpu/bench_moe 1 | grep nvfp4
echo "== p3 prefill: default / MoE GEMM v2 / ctrl GEMM"
for v in "" "INK_MOE_GEMM2=1" "INK_CTRL_GEMM=1" "INK_MOE_GEMM2=1 INK_CTRL_GEMM=1"; do
  echo -n "[$v] "; env $v ./build/inktest --backend metal --ids ref_data/p3.ids --gen ref_data/p3.gen --steps 2 --ctx 8192 --out $S/p3 2>&1 | grep prefill
done
