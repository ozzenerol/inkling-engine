# A: drafting defaults (block 8, lookup up to 15, k follows acceptance). Greedy must stay token-identical to plain.
M8=~/llm/mlx-models/Inkling-2-Mini-MLX-8bit
for ids in copy edit p2 p0; do
  ./build/src/gpu/spec_check plain ref_data/$ids.ids 600 8 $OUT/a-$ids-plain.i32 2>&1 | grep -E "tok/s"
  echo "-- $ids greedy, defaults"
  INK_SPEC_STATS=1 ./build/src/gpu/spec_check dspark ref_data/$ids.ids 600 8 $OUT/a-$ids-d.i32 2>&1 | grep -E "tok/s|dspark:|lookup:"
  cmp -s $OUT/a-$ids-plain.i32 $OUT/a-$ids-d.i32 && echo "   token-identical to plain" || echo "   DIFFERS from plain"
  echo "-- $ids greedy, no lookup";  INK_NO_LOOKUP=1 ./build/src/gpu/spec_check dspark ref_data/$ids.ids 600 8 $OUT/a-$ids-nl.i32 2>&1 | grep -E "tok/s"
  echo "-- $ids temp 0.7, defaults"; INK_SPEC_STATS=1 ./build/src/gpu/spec_check dspark ref_data/$ids.ids 600 8 $OUT/a-$ids-t07.i32 0.7 2>&1 | grep -E "tok/s|dspark:|lookup:"
  echo "-- $ids temp 0.7, plain";    ./build/src/gpu/spec_check plain ref_data/$ids.ids 600 8 $OUT/a-$ids-t07p.i32 0.7 2>&1 | grep -E "tok/s"
done
echo "== 8-bit build"
for ids in copy p2; do
  INK_MODEL=$M8 ./build/src/gpu/spec_check plain ref_data/$ids.ids 600 8 $OUT/a8-$ids-plain.i32 2>&1 | grep -E "tok/s"
  echo "-- 8-bit $ids greedy, defaults"
  INK_MODEL=$M8 INK_SPEC_STATS=1 ./build/src/gpu/spec_check dspark ref_data/$ids.ids 600 8 $OUT/a8-$ids-d.i32 2>&1 | grep -E "tok/s|dspark:|lookup:"
  cmp -s $OUT/a8-$ids-plain.i32 $OUT/a8-$ids-d.i32 && echo "   token-identical to plain" || echo "   DIFFERS from plain"
done
