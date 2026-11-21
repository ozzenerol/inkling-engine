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
