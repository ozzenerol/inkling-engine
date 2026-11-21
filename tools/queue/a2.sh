# A (batch 2): attention sparsity at long context. For each (query row, 64-key tile) of the first and last key
# block of global layers: how far below the row's running max the tile's best score is.
echo "-- attention sparsity, p17 (6224 tokens), global layers only"
INK_AG_STATS=1 ./build/inktest --backend metal --ids ref_data/p17.ids --gen ref_data/p17.gen --steps 1 --ctx 16384 --out $OUT 2>&1 | grep "ag stats" | grep "window 0" | awk '{print}' | sort | uniq -c | sort -rn | head -40
