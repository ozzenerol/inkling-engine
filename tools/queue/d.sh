# D: 8-bit build with the q8 one-pass matvec for 5..16 rows, on and off (INK_MR=0 also turns the bf16 one off).
SC=./build/src/gpu/spec_check
M8=~/llm/mlx-models/Inkling-2-Mini-MLX-8bit
for ids in copy p2; do
  for v in "" "INK_MR=0"; do
    echo "-- 8-bit build, $ids greedy, block 8, lookup max 15, ${v:-q8 multi-row on}"
    env INK_MODEL=$M8 INK_Q8_KERNELS=1 INK_SPEC_STATS=1 INK_LOOKUP_MAX=15 $v $SC dspark ref_data/$ids.ids 400 8 $OUT/d-q8-$ids${v:+-old}.i32 2>&1 | grep -E "tok/s|dspark:|lookup:|verify"
  done
  cmp -s $OUT/d-q8-$ids.i32 $OUT/d-q8-$ids-old.i32 && echo "8-bit $ids: q8 multi-row token-identical to passes of 4" || echo "8-bit $ids: q8 multi-row DIFFERS from passes of 4"
done
echo "-- lossless, p2, block 8, per-kernel GPU time (drafter kernels and verify at T = 8)"
env INK_PROFILE=1 INK_SPEC_STATS=1 INK_NO_LOOKUP=1 $SC dspark ref_data/p2.ids 100 8 $OUT/d-prof.i32 2>&1 | grep -E "tok/s|dspark:|kernel profile|us/call" | head -45
echo "-- 8-bit, copy, lookup 15, per-kernel GPU time (is q8 linear_multi above 4 rows worth one dispatch?)"
env INK_MODEL=$M8 INK_Q8_KERNELS=1 INK_PROFILE=1 INK_SPEC_STATS=1 INK_LOOKUP_MAX=15 $SC dspark ref_data/copy.ids 100 8 $OUT/d-prof8.i32 2>&1 | grep -E "kernel profile|us/call" | head -30
