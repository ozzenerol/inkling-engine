# A (batch 8): end of the context on the 8-bit build, and prefill attention with P as one half tile.
M8=~/llm/mlx-models/Inkling-2-Mini-MLX-8bit
PY=~/llm/inkling-venv/bin/python
echo "-- 8-bit long128k temp 0.7, defaults (generation at 126k)"
INK_MODEL=$M8 INK_SPEC_STATS=1 ./build/src/gpu/spec_check dspark ref_data/long128k.ids 300 8 $OUT/a-long128k-t07.i32 0.7 2>&1 | grep -E "tok/s|dspark:|adaptive"
echo "-- prefill attention: P as hi + lo half tiles (default) vs one half tile (INK_AG_HALF=1): accuracy on the long set"
L="10 11 12 13 14 15 16 17"
STEPS=32 ACC_OUT=$OUT/long_hilo tests/ref/acc_check.sh metal $L 2>&1 | grep -E "prefill"
INK_AG_HALF=1 STEPS=32 ACC_OUT=$OUT/long_half tests/ref/acc_check.sh metal $L 2>&1 | grep -E "prefill"
INK_AG_MIN_T=1000000 STEPS=32 ACC_OUT=$OUT/long_flash tests/ref/acc_check.sh metal $L >/dev/null 2>&1
$PY -I tests/ref/acc_summary.py "$L" mlx-bf16=mlx hilo=$OUT/long_hilo half=$OUT/long_half flash=$OUT/long_flash
echo "-- prefill speed at 64k (8-bit): default vs INK_AG_HALF=1"
INK_MODEL=$M8 ./build/src/gpu/spec_check plain ref_data/long64k.ids 20 8 $OUT/a-l64-hilo.i32 2>&1 | grep -E "tok/s"
INK_AG_HALF=1 INK_MODEL=$M8 ./build/src/gpu/spec_check plain ref_data/long64k.ids 20 8 $OUT/a-l64-half.i32 2>&1 | grep -E "tok/s"
