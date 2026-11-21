# D: expert-union MoE. Verify times per T with the model: union from 3 rows (default), from 2 rows, off.
SC=./build/src/gpu/spec_check
for ids in p2 copy; do
  for v in "" "INK_MR_MOE_MIN=2" "INK_MR_MOE=0"; do
    echo "-- $ids greedy, block 8, lookup max 15, ${v:-union from 3 rows (default)}"
    env INK_SPEC_STATS=1 INK_LOOKUP_MAX=15 $v $SC dspark ref_data/$ids.ids 400 8 $OUT/d-$ids-u${v:+-$v}.i32 2>&1 | grep -E "tok/s|dspark:|lookup:|verify"
  done
  for v in "INK_MR_MOE_MIN=2" "INK_MR_MOE=0"; do cmp -s "$OUT/d-$ids-u.i32" "$OUT/d-$ids-u-$v.i32" && echo "$ids: $v token-identical to the default" || echo "$ids: $v DIFFERS from the default"; done
done
echo "-- p2, block 4 (T <= 4 verifies), union from 3 / from 2 / off"
for v in "" "INK_MR_MOE_MIN=2" "INK_MR_MOE=0"; do env INK_SPEC_STATS=1 INK_NO_LOOKUP=1 $v $SC dspark ref_data/p2.ids 400 4 $OUT/d-b4.i32 2>&1 | grep -E "tok/s|verify"; done
echo "-- p2, block 8, per-kernel GPU time (drafter kernels and verify at T = 8)"
env INK_PROFILE=1 INK_SPEC_STATS=1 INK_NO_LOOKUP=1 $SC dspark ref_data/p2.ids 100 8 $OUT/d-prof.i32 2>&1 | grep -E "tok/s|dspark:|kernel profile|us/call" | head -45
for T in 2 3 4 8 16; do
  echo "-- bench_moe T=$T union from 2 / per route (random routes)"; INK_MR_MOE_MIN=2 ./build/gpu/bench_moe $T | grep layer; INK_MR_MOE=0 ./build/gpu/bench_moe $T | grep layer
done
