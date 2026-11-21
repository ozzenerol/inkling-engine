# D: multi-row forwards. Next batch: expert overlap between the rows of a verify, drafter profile at block 8.
SC=./build/src/gpu/spec_check
for ids in copy edit p2; do
  echo "-- $ids: distinct experts per verify (INK_MOE_STATS, timing not valid: a sync per MoE call)"
  env INK_MOE_STATS=1 INK_LOOKUP_MAX=15 $SC dspark ref_data/$ids.ids 200 8 $OUT/d-moe.i32 2>&1 | grep -E "moe stats"
done
echo "-- p2, block 8, per-kernel GPU time (drafter kernels and verify at T = 8)"
env INK_PROFILE=1 INK_SPEC_STATS=1 INK_NO_LOOKUP=1 $SC dspark ref_data/p2.ids 100 8 $OUT/d-prof.i32 2>&1 | grep -E "tok/s|dspark:|kernel profile|us/call" | head -45
for ids in copy p2; do
  for mx in 3 7 15; do
    echo "-- $ids greedy, block 8, lookup max $mx"
    env INK_SPEC_STATS=1 INK_LOOKUP_MAX=$mx $SC dspark ref_data/$ids.ids 400 8 $OUT/d-$ids-$mx.i32 2>&1 | grep -E "tok/s|dspark:|lookup:|verify"
    cmp -s $OUT/a-$ids-plain.i32 $OUT/d-$ids-$mx.i32 && echo "$ids lookup max $mx: token-identical to plain" || echo "$ids lookup max $mx: DIFFERS from plain (or no plain run in this batch)"
  done
done
