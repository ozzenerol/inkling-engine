# D: multi-row forwards (T = 5..16). Model-free kernel benches; the verify times per T come from A's lookup runs.
B=./build/gpu
echo "-- dense, one pass for 5..16 rows vs passes of 4 (bit check against the 1-row matvec)"
$B/bench_multirow 5 8 12 16
echo "-- attention, multi-row kernel"
$B/bench_mrattn 1 4 5 8 12 16
echo "-- attention, previous path for T > 4 (attn_prefill)"
INK_MR_ATTN=0 $B/bench_mrattn 5 8 16
for ab in 1 2 4 7; do echo "-- attention T=16 timing ablation $ab (1 no rel bias, 2 no PV, 4 no exp)"; INK_MR_ABL=$ab $B/bench_mrattn 16; done
