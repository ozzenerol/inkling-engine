# D: multi-row forwards (T = 5..16). Model-free kernel benches; the verify times per T come from A's lookup runs.
B=./build/gpu
echo "-- dense, one pass for 5..16 rows vs passes of 4 (bit check against the 1-row matvec)"
$B/bench_multirow 5 8 12 16
echo "-- attention, multi-row kernel"
$B/bench_mrattn 1 4 5 8 12 16
echo "-- attention, previous path for T > 4 (attn_prefill)"
INK_MR_ATTN=0 $B/bench_mrattn 5 8 16
for ab in 1 2 4 7; do echo "-- attention T=16 timing ablation $ab (1 no rel bias, 2 no PV, 4 no exp)"; INK_MR_ABL=$ab $B/bench_mrattn 16; done
# Which kernels make the cliff above T = 4: per-kernel GPU time with lookup drafts up to 7 (T up to 8), old and new paths.
for v in "INK_MR=0 INK_MR_ATTN=0" ""; do
  echo "-- profile copy.ids, lookup max 7, ${v:-multi-row kernels on}"
  env INK_PROFILE=1 INK_SPEC_STATS=1 INK_LOOKUP_MAX=7 $v ./build/src/gpu/spec_check dspark ref_data/copy.ids 100 4 $OUT/d-prof.i32 2>&1 | grep -E "tok/s|dspark:|lookup:|kernel profile|us/call" | head -34
done
for v in "INK_MR=0 INK_MR_ATTN=0" ""; do
  for mx in 7 15; do
    echo "-- copy.ids greedy, lookup max $mx, ${v:-multi-row kernels on}"
    env INK_SPEC_STATS=1 INK_LOOKUP_MAX=$mx $v ./build/src/gpu/spec_check dspark ref_data/copy.ids 400 4 $OUT/d-copy-$mx${v:+-old}.i32 2>&1 | grep -E "tok/s|dspark:|lookup:|verify"
  done
done
for mx in 7 15; do cmp -s $OUT/a-copy-plain.i32 $OUT/d-copy-$mx.i32 && echo "copy lookup max $mx: token-identical to plain" || echo "copy lookup max $mx: DIFFERS from plain"; done
