# B: candidates behind flags -> measure, then flip defaults. Run from the repo root (plain shell lines).
bin() { for d in build/gpu build/src/gpu build; do [ -x $d/$1 ] && { echo $d/$1; return; }; done; echo "missing:$1"; }
BS=${BATCH_TMP:-/tmp/claude-501/bq}; mkdir -p $BS
echo "== B p3 prefill (2886 tok): ctrl GEMM default vs scalar"
for v in "X=1" "INK_CTRL_SCALAR=1"; do
  echo -n "[$v] "; env $v build/inktest --backend metal --ids ref_data/p3.ids --gen ref_data/p3.gen --steps 2 --ctx 8192 --out $BS/p3 2>&1 | grep prefill
done
echo "== B keep-warm: first decode after 8 s idle, off vs on (INK_WARM_SECS)"
echo "(needs a tool with an idle gap; skipped until B adds one)"
echo "== B conv+mix+norm merge (INK_CONV_MIX=1): identity test, then plain decode off vs on"
INK_CONV_MIX=1 $(bin op_diff) 2>&1 | grep -E "conv\+mix|ALL|FAILED"
for v in "X=1" "INK_CONV_MIX=1"; do echo -n "[$v] "; env $v $(bin spec_check) plain ref_data/p2.ids 200 4 $BS/cm_$v.i32 2>&1 | grep -oE "decode [0-9.]+ s -> [0-9.]+ tok/s"; done
cmp "$BS/cm_X=1.i32" "$BS/cm_INK_CONV_MIX=1.i32" && echo "conv_mix tokens IDENTICAL"
echo "== B q8 fused linear_multi (INK_Q8_MULTI=1) on the 8-bit build: identity test, plain decode off vs on"
INK_Q8_MULTI=1 $(bin op_diff) 2>&1 | grep -E "q8 multi|ALL|FAILED"
M8=$HOME/llm/mlx-models/Inkling-2-Mini-MLX-8bit
for v in "X=1" "INK_Q8_MULTI=1"; do echo -n "[$v] "; env $v INK_MODEL=$M8 $(bin spec_check) plain ref_data/p2.ids 200 4 $BS/q8m_$v.i32 2>&1 | grep -oE "decode [0-9.]+ s -> [0-9.]+ tok/s"; done
cmp "$BS/q8m_X=1.i32" "$BS/q8m_INK_Q8_MULTI=1.i32" && echo "q8 multi tokens IDENTICAL"
echo "== B bandwidth of the big readers (model-free except bench_moe's two layers)"
$(bin bench_matvec) 1 | grep GB/s
$(bin bench_q8) 1; $(bin bench_q8) 4
for r in 4 2 1; do echo -n "[moe gate/up rows per simdgroup $r] "; INK_MOE_RPS=$r $(bin bench_moe) 1 | tr '\n' ' '; echo; done
