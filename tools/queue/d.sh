# D: long-context attention. Two-pass global attention without staging (INK_GA=1) vs the split kernels.
B=./build/gpu
SC=./build/src/gpu/spec_check
echo "-- op_diff with INK_GA=1 (attention vs CPU, verify==decode through the new kernels)"
INK_GA=1 $B/op_diff 2>&1 | grep -E "attention|verify==decode|FAIL|ALL"
echo "-- bench_mrattn, split kernels (current default)"; $B/bench_mrattn 1 2 4 8 16 | grep -v local
echo "-- bench_mrattn, INK_GA=1"; INK_GA=1 $B/bench_mrattn 1 2 4 8 16 | grep -v local
echo "-- bench_mrattn, INK_GA=1, K pass only (INK_GA_NOPV=1)"; INK_GA=1 INK_GA_NOPV=1 $B/bench_mrattn 1 4 16 | grep -v local
if [ -f ref_data/long32k.ids ]; then
  for v in "" "INK_GA=1"; do
    for mode in plain dspark; do
      echo "-- long32k greedy $mode ${v:-split kernels}"
      env INK_SPEC_STATS=1 $v $SC $mode ref_data/long32k.ids 200 8 $OUT/d-l32-$mode${v:+-ga}.i32 2>&1 | grep -E "tok/s|dspark:|verify"
    done
    cmp -s $OUT/d-l32-plain${v:+-ga}.i32 $OUT/d-l32-dspark${v:+-ga}.i32 && echo "long32k ${v:-split}: dspark token-identical to plain" || echo "long32k ${v:-split}: dspark DIFFERS from plain"
  done
  cmp -s $OUT/d-l32-plain.i32 $OUT/d-l32-plain-ga.i32 && echo "long32k: INK_GA=1 same tokens as the split kernels" || echo "long32k: INK_GA=1 tokens differ from the split kernels (other summation order; check it is a near-tie)"
fi
