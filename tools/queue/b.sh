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
echo "== B dense q8 cost: model-free bench alone vs with tiny dependent dispatches in between (in-model: 6.2 ms)"
for n in 0 2 6; do INTERLEAVE=$n $(bin bench_q8) 1; done
echo "== B 8-bit decode with INK_Q8_MULTI / INK_FUSE2 / both"
for v in "X=1" "INK_Q8_MULTI=1" "INK_FUSE2=1" "INK_Q8_MULTI=1 INK_FUSE2=1"; do echo -n "[$v] "; env $v INK_MODEL=$M8 $(bin spec_check) plain ref_data/p2.ids 200 4 /dev/null 2>&1 | grep -oE "decode [0-9.]+ s -> [0-9.]+ tok/s"; done
echo "== B fused decode ops really wired now (INK_FUSE2=1): op_diff (prep / conv_add / attention_gated cases are exact tests), then decode off vs on, both builds"
INK_FUSE2=1 $(bin op_diff) 2>&1 | grep -E "prep |conv_add|attention_gated|verify==|ALL|FAILED|FAIL "
for m in $HOME/llm/mlx-models/Inkling-2-Mini-MLX-4-8bit $M8; do for v in "X=1" "INK_FUSE2=1"; do
  echo -n "[$(basename $m) $v] "; env $v INK_MODEL=$m $(bin spec_check) plain ref_data/p2.ids 200 4 "$BS/f2_$(basename $m)_$v.i32" 2>&1 | grep -oE "decode [0-9.]+ s -> [0-9.]+ tok/s"
done; cmp "$BS/f2_$(basename $m)_X=1.i32" "$BS/f2_$(basename $m)_INK_FUSE2=1.i32" && echo "fuse2 tokens IDENTICAL"; done
echo -n "[fuse2 dspark identity] "; INK_FUSE2=1 tools/spec_check.sh ref_data/p2.ids 200 8 2>&1 | grep -E "IDENT|DIFF"
echo "== B T=1 vs forced T=2 verify on the 8-bit build: step times and per-kernel profile (2-row kernels off / on)"
SP="INK_NO_LOOKUP=1 INK_SPEC_FIXED=1 INK_SPEC_STATS=1 INK_MODEL=$M8"
for v in "INK_NO_T2=1" "X=1"; do echo "[$v]"; env $SP $v $(bin spec_check) dspark ref_data/p2.ids 120 2 "$BS/t2_$v.i32" 2>&1 | grep -E "rounds|verify|tok/s"; done
cmp "$BS/t2_INK_NO_T2=1.i32" "$BS/t2_X=1.i32" && echo "t2 kernels tokens IDENTICAL"
echo "[profile plain T=1]"; INK_PROFILE=1 INK_MODEL=$M8 $(bin spec_check) plain ref_data/p2.ids 60 4 /dev/null 2>&1 | grep -A14 "kernel profile"
echo "[profile dspark block 2 (T=2 verifies + drafts)]"; env $SP INK_PROFILE=1 $(bin spec_check) dspark ref_data/p2.ids 60 2 /dev/null 2>&1 | grep -A22 "kernel profile"
echo "== B GPU performance-state experiment: 8-bit plain decode with a background spin load on a second queue"
for b in 0 1 4 16; do for it in 20000 200000; do [ $b = 0 ] && [ $it = 200000 ] && continue
  echo -n "[busy groups $b iters $it] "; INK_BUSY=$b INK_BUSY_ITERS=$it INK_MODEL=$M8 $(bin spec_check) plain ref_data/p2.ids 100 4 /dev/null 2>&1 | grep -oE "decode [0-9.]+ s -> [0-9.]+ tok/s"
done; done
echo "[profile with busy 4 x 20000]"; INK_BUSY=4 INK_PROFILE=1 INK_MODEL=$M8 $(bin spec_check) plain ref_data/p2.ids 60 4 /dev/null 2>&1 | grep -E "matvec_q8_bf16_bf16_t1|moe_gateup_nvfp4|attn_decode_split|kernel profile"
