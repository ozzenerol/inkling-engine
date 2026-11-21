# D: long-context attention, INK_GA=1 (direct global attention) vs the split kernels, with the model.
B=./build/gpu
SC=./build/src/gpu/spec_check
M8=~/llm/mlx-models/Inkling-2-Mini-MLX-8bit
echo "-- op_diff with INK_GA=1"; INK_GA=1 $B/op_diff 2>&1 | grep -E "FAIL|ALL"
for ids in long128k long32k; do
  [ -f ref_data/$ids.ids ] || continue
  for v in "" "INK_GA=1" "INK_GA=1 INK_GA_DRAFT=0"; do
    echo "-- 8-bit, $ids, temp 0.7, 300 tokens, dspark block 8, ${v:-split kernels}"
    env INK_MODEL=$M8 INK_Q8_KERNELS=1 INK_SPEC_STATS=1 $v $SC dspark ref_data/$ids.ids 300 8 $OUT/d-$ids-t07.i32 0.7 2>&1 | grep -E "tok/s|dspark:|lookup:|adaptive"
  done
done
# identity: greedy, plain vs dspark, both with INK_GA=1 (lossless build, 32k)
if [ -f ref_data/long32k.ids ]; then
  for mode in plain dspark; do
    echo "-- lossless, long32k greedy $mode, INK_GA=1"
    env INK_GA=1 INK_SPEC_STATS=1 $SC $mode ref_data/long32k.ids 200 8 $OUT/d-l32-$mode-ga.i32 2>&1 | grep -E "tok/s|dspark:|adaptive"
  done
  cmp -s $OUT/d-l32-plain-ga.i32 $OUT/d-l32-dspark-ga.i32 && echo "long32k INK_GA=1: dspark token-identical to plain" || echo "long32k INK_GA=1: dspark DIFFERS from plain"
  echo "-- lossless, long32k greedy plain, split kernels"
  env INK_SPEC_STATS=1 $SC plain ref_data/long32k.ids 200 8 $OUT/d-l32-plain.i32 2>&1 | grep -E "tok/s"
  cmp -s $OUT/d-l32-plain.i32 $OUT/d-l32-plain-ga.i32 && echo "long32k: INK_GA=1 same tokens as the split kernels" || echo "long32k: INK_GA=1 tokens differ from the split kernels (other summation order; near-tie expected)"
fi
echo "-- bench_mrattn INK_GA=1 / split"; INK_GA=1 $B/bench_mrattn 1 2 4 8 16 | grep -v local; $B/bench_mrattn 1 2 4 8 16 | grep -v local
