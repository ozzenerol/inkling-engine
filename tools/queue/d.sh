# D: drafter on the direct attention kernels, drafter block with count+1 rows (INK_DSPARK_ROWS=1), INK_GA.
B=./build/gpu
SC=./build/src/gpu/spec_check
M8=~/llm/mlx-models/Inkling-2-Mini-MLX-8bit
echo "-- op_diff with INK_GA=1"; INK_GA=1 $B/op_diff 2>&1 | grep -E "FAIL|ALL"
for ids in p2 p0; do
  for t in "" 0.7; do
    for v in "" "INK_DSPARK_ROWS=1"; do
      echo "-- lossless, $ids, ${t:-greedy}, block 8, drafter rows: ${v:-8 (default)}"
      env INK_SPEC_STATS=1 $v $SC dspark ref_data/$ids.ids 300 8 $OUT/d-$ids${v:+-rows}.i32 $t 2>&1 | grep -E "tok/s|dspark:|adaptive"
    done
    [ -z "$t" ] && { cmp -s $OUT/d-$ids.i32 $OUT/d-$ids-rows.i32 && echo "$ids greedy: count+1 rows token-identical to 8 rows" || echo "$ids greedy: count+1 rows DIFFERS (must not: drafts only change speed)"; }
  done
done
if [ -f ref_data/long128k.ids ]; then
  for v in "INK_GA=1" "INK_GA=1 INK_GA_DRAFT=0" "INK_GA=1 INK_DSPARK_ROWS=1"; do
    echo "-- 8-bit, long128k, temp 0.7, 300 tokens, dspark block 8, $v"
    env INK_MODEL=$M8 INK_Q8_KERNELS=1 INK_SPEC_STATS=1 $v $SC dspark ref_data/long128k.ids 300 8 $OUT/d-l128.i32 0.7 2>&1 | grep -E "tok/s|dspark:|lookup:|adaptive"
  done
fi
if [ -f ref_data/long32k.ids ]; then
  for v in "INK_GA=1" "INK_GA=1 INK_DSPARK_ROWS=1"; do
    echo "-- 8-bit, long32k, temp 0.7, 300 tokens, dspark block 8, $v"
    env INK_MODEL=$M8 INK_Q8_KERNELS=1 INK_SPEC_STATS=1 $v $SC dspark ref_data/long32k.ids 300 8 $OUT/d-l32.i32 0.7 2>&1 | grep -E "tok/s|dspark:|lookup:|adaptive"
  done
fi
