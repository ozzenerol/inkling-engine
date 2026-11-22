# B: prefill on the 8-bit build. Run from the repo root.
BS=${BATCH_TMP:-/tmp/claude-501/bq}; mkdir -p $BS
M8=$HOME/llm/mlx-models/Inkling-2-Mini-MLX-8bit
head -c 8192 ref_data/p3.ids > $BS/p2048.ids; head -c 3200 ref_data/p3.ids > $BS/p800.ids
run() { build/inktest --backend metal --model $M8 --ids $1 --steps 1 --ctx 8192 --chunk 2048 --out $BS/pf 2>&1; }
echo "== B 8-bit prefill times (inktest, chunk 2048): 2048 tokens, 800 tokens, 2886 tokens (2048 + 838 tail)"
for f in $BS/p2048.ids $BS/p800.ids ref_data/p3.ids; do run $f | grep prefill; done
echo "== B 8-bit prefill profile at T = 2048 (each dispatch waited separately), top 16"
INK_PROFILE=1 run $BS/p2048.ids | grep -A16 "kernel profile"
echo "== B 8-bit prefill profile, 800-token turn, top 16"
INK_PROFILE=1 run $BS/p800.ids | grep -A16 "kernel profile"
echo "== B same two on the lossless build, top 8 (for comparison)"
M8=$HOME/llm/mlx-models/Inkling-2-Mini-MLX-4-8bit
INK_PROFILE=1 run $BS/p2048.ids | grep -A8 "kernel profile"
echo "== B MoE prefill tiles of 32 rows per expert (INK_MOE_M=32; 0 = by T): prefill time at 800 / 2048 / 2886 tokens, logits cmp"
for M8 in $HOME/llm/mlx-models/Inkling-2-Mini-MLX-8bit $HOME/llm/mlx-models/Inkling-2-Mini-MLX-4-8bit; do
  for f in $BS/p800.ids $BS/p2048.ids ref_data/p3.ids; do for v in 64 32 0; do
    mkdir -p $BS/m$v; echo -n "[$(basename $M8) $(basename $f) M=$v] "
    INK_MOE_M=$v build/inktest --backend metal --model $M8 --ids $f --steps 1 --ctx 8192 --chunk 2048 --out $BS/m$v 2>&1 | grep prefill
  done; ls $BS/m64 | head -3 | tr '\n' ' '; cmp $BS/m64/logits.f32 $BS/m32/logits.f32 && cmp $BS/m64/logits.f32 $BS/m0/logits.f32 && echo "moe tile logits IDENTICAL"; done
done
INK_MOE_M=32 build/gpu/op_diff 2>&1 | grep -E "moe .*T=300|ALL|FAILED"
echo "== B NULL chain: 8-bit plain decode with every dispatch replaced by a one-thread kernel (same barriers / commits)"
bin() { for d in build/gpu build/src/gpu build; do [ -x $d/$1 ] && { echo $d/$1; return; }; done; echo "missing:$1"; }
M8=$HOME/llm/mlx-models/Inkling-2-Mini-MLX-8bit
for v in "X=1" "INK_NULL=1" "INK_NULL=1 INK_NO_FUSE2=1" "INK_NULL=1 INK_SERIAL=1" "INK_NULL=1 INK_COMMIT_EVERY=0" "INK_SERIAL=1"; do
  echo -n "[$v] "; env $v INK_GPU_TIME=1 INK_MODEL=$M8 $(bin spec_check) plain ref_data/p2.ids 100 4 /dev/null 2>&1 | grep -E "hazard|tok/s" | tr '\n' ' '; echo
done
echo "== B INTERLEAVE bench: dependent vs independent tiny dispatches between the q8 matvecs"
for n in 0 2 6; do INTERLEAVE=$n $(bin bench_q8) 1; done
for n in 2 6; do INTERLEAVE_INDEP=1 INTERLEAVE=$n $(bin bench_q8) 1; done
INK_SERIAL=1 INTERLEAVE=2 $(bin bench_q8) 1; INK_NULL=1 INTERLEAVE=2 $(bin bench_q8) 1
echo "== B read/write hazard tracking (default on; INK_RW=0 = old): op_diff, barriers per dispatch, decode off vs on with token cmp (both builds), dspark identity"
$(bin op_diff) 2>&1 | grep -E "ALL|FAILED|FAIL "
for m in $HOME/llm/mlx-models/Inkling-2-Mini-MLX-8bit $HOME/llm/mlx-models/Inkling-2-Mini-MLX-4-8bit; do for v in "INK_RW=0" "X=1"; do
  echo -n "[$(basename $m) $v] "; env $v INK_GPU_TIME=1 INK_MODEL=$m $(bin spec_check) plain ref_data/p2.ids 200 4 "$BS/rw_$(basename $m)_$v.i32" 2>&1 | grep -E "hazard|tok/s" | tr '\n' ' '; echo
done; cmp "$BS/rw_$(basename $m)_X=1.i32" "$BS/rw_$(basename $m)_INK_RW=0.i32" && echo "rw tokens IDENTICAL"; done
echo -n "[rw dspark identity] "; tools/spec_check.sh ref_data/p2.ids 200 8 2>&1 | grep -E "IDENT|DIFF"
echo -n "[rw prefill p3 8-bit] "; build/inktest --backend metal --model $M8 --ids ref_data/p3.ids --steps 1 --ctx 8192 --chunk 2048 --out $BS/m0 2>&1 | grep prefill
echo "== B bench_q8 wall clock vs GPU timestamps (756 GB/s in batch 9 is above the M3 Max's 400 GB/s)"
for n in 0 2; do INTERLEAVE=$n $(bin bench_q8) 1; done; INK_SERIAL=1 INTERLEAVE=2 $(bin bench_q8) 1
echo "== B bench_q8 as a dependent chain (DEP=1: each matvec reads the previous output when sizes fit)"
DEP=1 $(bin bench_q8) 1; DEP=1 INK_RW=0 $(bin bench_q8) 1
echo "== B mix in two wide dispatches (default; INK_MIX2=0 = old): op_diff, 8-bit plain decode off vs on with token cmp, dspark identity, profile"
$(bin op_diff) 2>&1 | grep -E "ALL|FAILED|FAIL |mix|attn_residual"
M8=$HOME/llm/mlx-models/Inkling-2-Mini-MLX-8bit
for v in "INK_MIX2=0" "X=1"; do
  echo -n "[8-bit $v] "; env $v INK_MODEL=$M8 $(bin spec_check) plain ref_data/p2.ids 200 4 "$BS/mix_$v.i32" 2>&1 | grep -E "tok/s" | tr '\n' ' '; echo
done; cmp "$BS/mix_X=1.i32" "$BS/mix_INK_MIX2=0.i32" && echo "mix2 tokens IDENTICAL"
echo -n "[mix2 dspark identity] "; tools/spec_check.sh ref_data/p2.ids 200 8 2>&1 | grep -E "IDENT|DIFF"
INK_PROFILE=1 INK_NO_DSPARK=1 build/inktest --backend metal --model $M8 --ids ref_data/p2.ids --steps 64 --out $BS/pf 2>&1 | grep -E "mix|rms_norm_rows"
echo "== B confidence gate on the draft length (default; INK_SPEC_GATE=0 = draft count follows acceptance): 8-bit, block 8, off vs on with token cmp"
for f in p2 p0 p12 edit; do
  for v in "INK_SPEC_GATE=0" "X=1"; do
    echo -n "[gate $f $v] "; env $v INK_MODEL=$M8 INK_SPEC_STATS=1 $(bin spec_check) dspark ref_data/$f.ids 300 8 "$BS/gate_$v.i32" 2>&1 | grep -E "tok/s \(block|^gate" | cut -c1-150 | tr '\n' ' '; echo
  done; cmp "$BS/gate_X=1.i32" "$BS/gate_INK_SPEC_GATE=0.i32" && echo "gate $f tokens IDENTICAL"
done
echo "== B output head of 5..16 rows as passes of the 4-row q8 kernel (default; INK_MR_Q8_RH=2 = one multi-row pass): 8-bit, block 8"
for f in p0 p12; do
  for v in "INK_MR_Q8_RH=2" "X=1"; do
    echo -n "[head $f $v] "; env $v INK_MODEL=$M8 $(bin spec_check) dspark ref_data/$f.ids 300 8 "$BS/head_$v.i32" 2>&1 | grep -o "decode.*tok/s"
  done; cmp "$BS/head_X=1.i32" "$BS/head_INK_MR_Q8_RH=2.i32" && echo "head $f tokens IDENTICAL"
done
echo "== B drafter Markov step in two wide dispatches (default; INK_DSPARK_MARKOV1=1 = one threadgroup): 8-bit, block 8"
for f in p0 p12; do
  for v in "INK_DSPARK_MARKOV1=1" "X=1"; do
    echo -n "[markov $f $v] "; env $v INK_MODEL=$M8 INK_DSPARK_TIME=1 $(bin spec_check) dspark ref_data/$f.ids 300 8 "$BS/mk_$v.i32" 2>&1 | grep -E "dspark time|tok/s \(block" | tail -2 | grep -o "markov+fold+argmax [0-9.]*\|decode.*tok/s" | tr '\n' ' '; echo
  done; cmp "$BS/mk_X=1.i32" "$BS/mk_INK_DSPARK_MARKOV1=1.i32" && echo "markov $f tokens IDENTICAL"
done
