# A: the target line (60 tok/s at the start, 40 at 128k): generation at long context, free text, temp 0.7,
# 8-bit build. One process per context (prefill once): drafting defaults; the stats line gives the plain step,
# draft and verify times per T at that depth. long*.ids = source files as context + an essay question.
M8=~/llm/mlx-models/Inkling-2-Mini-MLX-8bit
for ids in p2 long32k long64k long128k; do
  echo "-- 8-bit $ids temp 0.7, defaults"
  INK_MODEL=$M8 INK_SPEC_STATS=1 ./build/src/gpu/spec_check dspark ref_data/$ids.ids 300 8 $OUT/a-$ids-t07.i32 0.7 2>&1 | grep -E "tok/s|dspark:|lookup:|adaptive"
done
echo "-- 8-bit long128k temp 0.7, plain"
INK_MODEL=$M8 ./build/src/gpu/spec_check plain ref_data/long128k.ids 100 8 $OUT/a-long128k-plain.i32 0.7 2>&1 | grep -E "tok/s"
# copy-heavy and code prompts: regression check of the drafting defaults (greedy must equal plain)
for ids in copy p0; do
  INK_MODEL=$M8 ./build/src/gpu/spec_check plain ref_data/$ids.ids 400 8 $OUT/a-$ids-plain.i32 2>&1 | grep -E "tok/s"
  INK_MODEL=$M8 ./build/src/gpu/spec_check dspark ref_data/$ids.ids 400 8 $OUT/a-$ids-d.i32 2>&1 | grep -E "tok/s"
  cmp -s $OUT/a-$ids-plain.i32 $OUT/a-$ids-d.i32 && echo "   $ids token-identical to plain" || echo "   $ids DIFFERS from plain"
done
