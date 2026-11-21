# Results (2026-11-21)

Inkling 2 Mini on the M3 Max (48 GB), `inkling-server` on port 8091 against MLX-VLM 0.8.0+inkling2mini.2 with its
DSpark draft (block 4). **Every number is one run.** `inkling-8bit` is the server's default model
(`Inkling-2-Mini-MLX-8bit`); `inkling` is the lossless 4/8-bit build. MLX always ran the 4/8-bit build.

Target: each prompt is read fast; generation at least 40 tok/s at the start of the context and 20-30 tok/s at
127.5k.

Where the numbers come from: target table (the reference table), TTFT breakdown, the two 8-bit summary rows
of the agent table and the qh run = batch 9 (main c618174, 22:37, server defaults: 8-bit model, draft block 8,
lookup drafts on); per-turn 8-bit agent columns = batch 7 (batch 9 is within 0.1 s per turn); lossless agent
sessions and the 8-bit cold rows = batch 5; lossless cold sweep = batch 6, which was stopped after 78.1k (no
127.5k row, no 8-bit sweep); MLX = measured the same day (17:43 and 19:10), rows marked \* from BRIEF.md.
Not measured: MLX on the copy scenario, the cold sweep to 127.5k on the 8-bit build, MLX in a growing session,
the 30k agent session after batch 3.

Target benchmark, inkling-8bit: read speed and gen tok/s by context depth (one growing session, essay prompt, temp 0.7, 400 tokens, one run)

| context | new tokens read | read tok/s | TTFT s | gen tok/s | gen target | meets | MLX gen | vs MLX |
|---|---|---|---|---|---|---|---|---|
| 1.2k | 1194 | 729.1 | 1.64 | 39.9 | 40 | no | 23.1 | 1.73x |
| 9.6k | 8013 | 793.6 | 10.1 | 38.1 | 39 | no | 24.2 | 1.57x |
| 39.3k | 29302 | 658.4 | 44.51 | 30.3 | 34-37 | no | 22.0 | 1.38x |
| 78.6k | 38886 | 502.5 | 77.39 | 27.2 | 28-34 | no | 16.5* | 1.65x |
| 127.3k | 48322 | 380.5 | 126.99 | 22.0 | 20-30 | in range | 13.8* | 1.59x |

read tok/s = new prompt tokens / time to first token (all-in). Gen target: >= 40 tok/s at the start, >= 20-30 at 127.5k.
MLX gen: its ctxbench run at that prompt size (cold prompt, same essay task); \* = from BRIEF.md.

Batches 7, 8 and 9 gave the same table within 1.1 s of TTFT and 1.8 tok/s of generation per row (three runs,
three builds). Generation is 38.1-39.9 at the start (40 line not met) and inside 20-30 at the end; the middle
rows are 4-7 tok/s under a straight line between the two.

Where the time to first token goes (server log `[ttft]`, batch 9): prefill is 93-99% of it at every depth.
An agent turn that reads 775 new tokens takes 1.25-1.41 s: prefill 1.20-1.32 s (645 tok/s at 3k context falling
to 589 at 13k), checkpoint 2-15 ms, tokenize + cache lookup under 2 ms, and 50-100 ms for the marker tokens the
model writes before the first visible delta. A chunk of 552-790 tokens runs at 540-645 tok/s where chunks of
2000 run at 720-800: the cost per chunk, not the server, is what is left. Chunks sent: one chunk up to 2048 new
tokens, otherwise near-equal chunks (2117 -> 1058+1059; 48322 -> 24 chunks of 2013-2014); no small tails,
except after the 2045-token shared prefix (system prompt + tools), where the split is 2045 + rest on purpose.

ctxbench (temp 0.7, 400 tokens), tok/s, one run each

| context | mlx prefill | mlx gen | inkling prefill | inkling gen | inkling-8bit prefill | inkling-8bit gen | inkling/mlx prefill | inkling/mlx gen | inkling-8bit/mlx prefill | inkling-8bit/mlx gen |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.2k | 564.8 | 23.1 | 749.0 | 34.8 | 739.7 | 38.2 | 1.33x | 1.51x | 1.31x | 1.65x |
| 9.5k | 602.5 | 24.2 | 815.3 | 32.9 | 803.7 | 36.1 | 1.35x | 1.36x | 1.33x | 1.49x |
| 18.9k | 571* | 23.3* | 782.3 | 31.5 | - | - | 1.37x | 1.35x | - | - |
| 38.6k | 495.4 | 22.0 | 710.2 | 29.2 | - | - | 1.43x | 1.33x | - | - |
| 78.1k | 385* | 16.5* | 592.5 | 23.8 | - | - | 1.54x | 1.44x | - | - |
| 127.5k | 324* | 13.8* | - | - | - | - | - | - | - | - |

inkling-8bit = the 8-bit build (Inkling-2-Mini-MLX-8bit); its ratios are against MLX on the 4-8bit build (no MLX 8-bit baseline measured).

\* from BRIEF.md's MLX table (same script and settings, not re-measured today)

agent_bench (qh system prompt + tools, ~800-token tool result per turn, temp 0, thinking on)

| server | base ctx | turns | last prompt | total wall s | sum TTFT s | median TTFT after turn 1 s | median read tok/s after turn 1 | median gen tok/s | tokens out |
|---|---|---|---|---|---|---|---|---|---|
| inkling | 0 | 10 | 13345 | 74.0 | 15.3 | 1.36 | 581 | 88.6 | 4654 |
| inkling-8bit | 0 | 10 | 13244 | 88.1 | 15.1 | 1.35 | 573 | 64.6 | 4639 |
| inkling-8bit copy | 0 | 6 | 8350 | 49.0 | 8.9 | 1.08 | 513 | 87.7 | 3496 |
| inkling copy | 0 | 6 | 8338 | 43.2 | 8.8 | 1.07 | 514 | 103.3 | 3484 |
| mlx | 0 | 10 | 13369 | 241.5 | 128.6 | 13.52 | 603 | 44.1 | 4654 |
| mlx | 30000 | 10 | 39152 | 742.8 | 691.5 | 70.05 | 506 | 34.9 | 1675 |

read tok/s = new prompt tokens (prompt - cached) / time to first token: what one turn's read costs, all-in.

| turn | inkling@0 TTFT s, read tok/s, gen tok/s (prompt/cached) | inkling-8bit@0 TTFT s, read tok/s, gen tok/s (prompt/cached) | inkling-8bit copy@0 TTFT s, read tok/s, gen tok/s (prompt/cached) | inkling copy@0 TTFT s, read tok/s, gen tok/s (prompt/cached) | mlx@0 TTFT s, read tok/s, gen tok/s (prompt/cached) | mlx@30000 TTFT s, read tok/s, gen tok/s (prompt/cached) |
|---|---|---|---|---|---|---|
| 1 | 3.09, 684, 70.7 (2117/0) | 2.93, 723, 65.1 (2117/0) | 3.50, 755, 83.0 (2643/0) | 3.47, 761, 94.4 (2643/0) | 4.12, 514, 50.9 (2117/0) | 59.00, 520, 37.7 (30666/0) |
| 2 | 1.26, 615, 65.4 (2937/2162) | 1.25, 621, 63.4 (2937/2162) | 1.08, 513, 87.8 (3946/3394) | 1.05, 524, 100.5 (3946/3394) | 4.60, 638, 41.6 (2937/0) | 60.89, 517, 38.8 (31496/0) |
| 3 | 1.32, 597, 48.9 (4238/3448) | 1.32, 588, 59.6 (4223/3448) | 1.08, 509, 86.3 (5065/4513) | 1.06, 521, 104.3 (5053/4501) | 6.93, 612, 32.6 (4241/0) | 62.79, 515, 28.4 (32331/0) |
| 4 | 1.29, 610, 67.6 (5539/4749) | 1.33, 594, 48.3 (5524/4734) | 1.07, 518, 86.6 (6160/5608) | 1.07, 514, 102.4 (6148/5596) | 8.96, 619, 39.7 (5545/0) | 65.56, 512, 34.9 (33566/0) |
| 5 | 1.36, 580, 84.1 (6840/6050) | 1.35, 585, 50.8 (6825/6035) | 1.09, 505, 88.8 (7255/6703) | 1.08, 511, 104.8 (7243/6691) | 11.20, 612, 41.5 (6849/0) | 67.92, 507, 29.5 (34437/0) |
| 6 | 1.35, 585, 93.2 (8141/7351) | 1.41, 550, 65.0 (8112/7336) | 1.11, 499, 87.5 (8350/7798) | 1.08, 509, 104.2 (8338/7786) | 13.52, 603, 47.2 (8153/0) | 70.05, 506, 31.8 (35436/0) |
| 7 | 1.36, 581, 99.4 (9442/8652) | 1.35, 574, 74.2 (9395/8620) | - | - | 15.70, 602, 47.4 (9457/0) | 73.40, 495, 35.0 (36365/0) |
| 8 | 1.40, 565, 99.1 (10743/9953) | 1.37, 568, 73.6 (10678/9903) | - | - | 18.21, 591, 46.5 (10761/0) | 74.90, 498, 38.3 (37294/0) |
| 9 | 1.39, 568, 99.3 (12044/11254) | 1.39, 556, 74.7 (11961/11186) | - | - | 21.62, 558, 46.5 (12065/0) | 77.59, 493, 33.5 (38223/0) |
| 10 | 1.46, 540, 98.5 (13345/12555) | 1.45, 533, 75.0 (13244/12469) | - | - | 23.74, 563, 33.5 (13369/0) | 79.39, 493, 39.9 (39152/0) |

## qh end-to-end

`tools/qh_e2e.sh`: qh fixes `/tmp/inkling-qh-test` (5 failing unit tests) in a fresh clone.

| server | result | wall time |
|---|---|---|
| inkling-8bit (batch 9) | 5/5 tests pass | 28 s (4 steps, 1331 tokens out) |
| inkling-8bit (batch 8) | 5/5 tests pass | 17 s (5 steps, 643 tokens out) |
| inkling-8bit (batch 7) | 5/5 tests pass | 20 s |
| inkling-8bit (batch 5) | 5/5 tests pass | 16 s |
| MLX-VLM (two runs, 17:45) | 5/5 tests pass | 57 s, 40 s |

## Reading the tables

- **Gen target column**: the user gave two points (40 tok/s at the start, 20-30 at 127.5k); the values in between
  are a straight line between them, drawn by us. "in range" = above the lower line, below the upper one.

- **Read speed in the target table is at depth**: the tokens of each step are read on top of everything before
  them, so the 127.3k row (381 tok/s) is the speed of reading tokens 79k-127k, not an average from zero. MLX's
  prefill numbers in the cold sweep are averages over the whole prompt, so the two columns are not the same
  measurement; the cold sweep is the like-for-like one.
- **Agent sessions**: MLX-VLM re-reads the whole prompt every turn (0 cached tokens), inkling-server reads only
  the new tokens, which is why time to first token stays near 1.4 s while MLX's grows with the context.
- Agent gen tok/s is at temperature 0 on text with repetition (drafts are accepted often); the target table is
  temperature 0.7 on free text. Different workloads.
- MLX-VLM ignores `reasoning_effort` from the request, so in the essay runs it thought at the template default
  while inkling-server ran at "none". Speeds are comparable; the answers are not.
