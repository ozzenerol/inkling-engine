# B: candidates behind flags -> measure, then flip defaults. Run from the repo root (plain shell lines).
bin() { for d in build/gpu build/src/gpu build; do [ -x $d/$1 ] && { echo $d/$1; return; }; done; echo "missing:$1"; }
BS=${BATCH_TMP:-/tmp/claude-501/bq}; mkdir -p $BS
echo "== B op_diff with all B candidates on"
INK_CTRL_GEMM=1 INK_MOE_GEMM2=1 INK_FP4_LUT=1 $(bin op_diff) 2>&1 | grep -E "q8|controller|moe|ALL|FAILED|FAIL "
echo "== B MoE decode: nibble decode vs LUT (us per call)"
$(bin bench_moe) 1 | grep nvfp4
INK_FP4_LUT=1 $(bin bench_moe) 1 | grep nvfp4
echo "== B p3 prefill (2886 tok): default / MoE GEMM v2 / ctrl GEMM / both"
for v in "X=1" "INK_MOE_GEMM2=1" "INK_CTRL_GEMM=1" "INK_MOE_GEMM2=1 INK_CTRL_GEMM=1"; do
  echo -n "[$v] "; env $v build/inktest --backend metal --ids ref_data/p3.ids --gen ref_data/p3.gen --steps 2 --ctx 8192 --out $BS/p3 2>&1 | grep prefill
done
echo "== B keep-warm: first decode after 8 s idle, off vs on (INK_WARM_SECS)"
echo "(needs a tool with an idle gap; skipped until B adds one)"
echo "== B conv+mix+norm merge (INK_CONV_MIX=1): identity test, then plain decode off vs on"
INK_CONV_MIX=1 $(bin op_diff) 2>&1 | grep -E "conv\+mix|ALL|FAILED"
for v in "X=1" "INK_CONV_MIX=1"; do echo -n "[$v] "; env $v $(bin spec_check) plain ref_data/p2.ids 200 4 $BS/cm_$v.i32 2>&1 | grep -oE "decode [0-9.]+ s -> [0-9.]+ tok/s"; done
cmp "$BS/cm_X=1.i32" "$BS/cm_INK_CONV_MIX=1.i32" && echo "conv_mix tokens IDENTICAL"
