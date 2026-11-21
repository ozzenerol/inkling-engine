# B: plain-step ablation on the 8-bit build (timing only: INK_SKIP makes results wrong). Run from the repo root.
bin() { for d in build/gpu build/src/gpu build; do [ -x $d/$1 ] && { echo $d/$1; return; }; done; echo "missing:$1"; }
BS=${BATCH_TMP:-/tmp/claude-501/bq}; mkdir -p $BS
M8=$HOME/llm/mlx-models/Inkling-2-Mini-MLX-8bit
echo "== B 8-bit plain decode ablation (100 tokens each; ms/token = 1000 / tok/s)"
for sk in none moe attn linear router small mix ctrl logits "moe,attn,linear,router,small,mix,ctrl,logits"; do
  echo -n "skip $sk: "; INK_SKIP=$sk INK_MODEL=$M8 $(bin spec_check) plain ref_data/p2.ids 100 4 /dev/null 2>&1 | grep -oE "decode [0-9.]+ s -> [0-9.]+ tok/s"
done
echo "== B 8-bit plain decode: GPU span vs wall per sync, hazard barriers"
INK_GPU_TIME=1 INK_MODEL=$M8 $(bin spec_check) plain ref_data/p2.ids 100 4 /dev/null 2>&1 | grep -E "gpu time|hazard|tok/s"
echo "== B 8-bit plain decode: per-kernel profile (each dispatch waited separately; ranking only), top 18"
INK_PROFILE=1 INK_MODEL=$M8 $(bin spec_check) plain ref_data/p2.ids 60 4 /dev/null 2>&1 | grep -A18 "kernel profile"
echo "== B NVFP4 vectorized decode (INK_FP4_VEC=1, bit-identical): us per call and identity"
$(bin bench_moe) 1 | grep nvfp4; INK_FP4_VEC=1 $(bin bench_moe) 1 | grep nvfp4
for v in "X=1" "INK_FP4_VEC=1"; do env $v $(bin spec_check) plain ref_data/p2.ids 100 4 $BS/vec_$v.i32 2>&1 | grep -oE "decode [0-9.]+ s -> [0-9.]+ tok/s"; done
cmp "$BS/vec_X=1.i32" "$BS/vec_INK_FP4_VEC=1.i32" && echo "fp4 vec tokens IDENTICAL"
echo "== B bench_q8 sanity: same bench with INK_Q8_KERNELS-independent bf16 reference"
$(bin bench_matvec) 1 | grep GB/s; $(bin bench_q8) 1
echo "== B fused decode ops really wired now (INK_FUSE2=1): op_diff (prep / conv_add / attention_gated cases are exact tests), then decode off vs on, both builds"
INK_FUSE2=1 $(bin op_diff) 2>&1 | grep -E "prep |conv_add|attention_gated|verify==|ALL|FAILED|FAIL "
for m in $HOME/llm/mlx-models/Inkling-2-Mini-MLX-4-8bit $M8; do for v in "X=1" "INK_FUSE2=1"; do
  echo -n "[$(basename $m) $v] "; env $v INK_MODEL=$m $(bin spec_check) plain ref_data/p2.ids 200 4 "$BS/f2_$(basename $m)_$v.i32" 2>&1 | grep -oE "decode [0-9.]+ s -> [0-9.]+ tok/s"
done; cmp "$BS/f2_$(basename $m)_X=1.i32" "$BS/f2_$(basename $m)_INK_FUSE2=1.i32" && echo "fuse2 tokens IDENTICAL"; done
echo -n "[fuse2 dspark identity] "; INK_FUSE2=1 tools/spec_check.sh ref_data/p2.ids 200 8 2>&1 | grep -E "IDENT|DIFF"
