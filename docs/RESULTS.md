# Results (2026-11-22)

Inkling 2 Mini on the M3 Max (48 GB), `inkling-server` on port 8091 against MLX-VLM 0.8.0+inkling2mini.2 with its
DSpark draft (block 4). **Every number is one run.** `inkling-8bit` is the server's default model
(`Inkling-2-Mini-MLX-8bit`); `inkling` is the lossless 4/8-bit build. MLX always ran the 4/8-bit build.

Target: each prompt is read fast; generation at least 40 tok/s at the start of the context and 20-30 tok/s at
127.5k. Both builds meet it at every depth in this batch.

Where the numbers come from: everything below except the MLX rows = batch 20 (main 68d0a8a, 2026-11-22
10:08-10:56, `tools/batch.sh gates queues full`, all gates pass; server defaults: 8-bit model, draft block 8,
lookup drafts, draft length from the drafter's confidence, direct global attention, read/write hazard sets, MoE
prefill tiles by T, prefix pinned on the first prefill, weights kept wired while idle). MLX = measured 2026-11-21
(17:43 and 19:10), rows marked \* from BRIEF.md; MLX copy scenario = batch 16 (2026-11-22). Not measured: MLX in
a growing session.

Earlier 8-bit target tables for comparison (generation tok/s at 1.2k / 9.6k / 39.3k / 78.6k / 127.3k):
batch 9 (2026-11-21) 39.9 / 38.1 / 30.3 / 27.2 / 22.0; batch 13 (2026-11-22, night) 42.3 / 40.7 / 37.4 / 32.0 /
29.0. Agent read session, 8-bit: 88 s (batch 9), 83.1 s (batch 13), 72.7 s here. What changed since batch 13,
all with identical tokens: the draft length follows the drafter's confidence, the output head of a multi-row
verify takes 3.9 ms instead of 14.1 (8 rows), the drafter's token step 2.3 ms instead of 5.5 (7 tokens).

Where the time to first token goes (server log `[ttft]`): prefill is 93-99% of it at every depth. An agent turn
that reads 775 new tokens takes 1.1-1.3 s. Short chunks cost more per token than chunks of 2000 because a
chunk has a fixed cost of about 180 ms (profiles at 800 and 2048 tokens fit 179 ms + 1.055 ms per token): every
visit reads all 128 experts whatever the chunk size. A second session with the same system prompt and tools
restores the pinned prefix: 2043 of 2643 tokens cached, time to first token 0.96 s instead of 3.4 s.
A turn after a pause: macOS unwires the weights after about 2 s without GPU work and the next request then waited
0.38-0.44 s in the driver; the server now keeps them wired for 10 minutes after the last request
(`INK_WARM_SECS`). Reading a 10-token turn: 68-70 ms right after work, 77-87 ms after a 2-20 s pause (was
380-540 ms).

Target benchmark, inkling-8bit: read speed and gen tok/s by context depth (one growing session, essay prompt, temp 0.7, 400 tokens, one run)

| context | new tokens read | read tok/s | TTFT s | gen tok/s | gen target | meets | MLX gen | vs MLX |
|---|---|---|---|---|---|---|---|---|
| 1.2k | 1194 | 802.2 | 1.49 | 45.8 | 40 | yes | 23.1 | 1.98x |
| 9.6k | 8013 | 806.5 | 9.94 | 44.7 | 39 | yes | 24.2 | 1.85x |
| 39.3k | 29302 | 669.4 | 43.77 | 38.9 | 34-37 | yes | 22.0 | 1.77x |
| 78.6k | 38886 | 508.1 | 76.53 | 33.5 | 28-34 | in range | 16.5* | 2.03x |
| 127.3k | 48322 | 383.1 | 126.12 | 29.9 | 20-30 | in range | 13.8* | 2.17x |

read tok/s = new prompt tokens / time to first token (all-in). Gen target: >= 40 tok/s at the start, >= 20-30 at 127.5k.
MLX gen: its ctxbench run at that prompt size (cold prompt, same essay task); \* = from BRIEF.md.

Target benchmark, inkling: read speed and gen tok/s by context depth (one growing session, essay prompt, temp 0.7, 400 tokens, one run)

| context | new tokens read | read tok/s | TTFT s | gen tok/s | gen target | meets | MLX gen | vs MLX |
|---|---|---|---|---|---|---|---|---|
| 1.2k | 1194 | 812.2 | 1.47 | 40.4 | 40 | yes | 23.1 | 1.75x |
| 9.6k | 8013 | 814.5 | 9.84 | 40.8 | 39 | yes | 24.2 | 1.69x |
| 39.3k | 29302 | 675.4 | 43.38 | 35.7 | 34-37 | in range | 22.0 | 1.62x |
| 78.6k | 38886 | 512.5 | 75.87 | 32.1 | 28-34 | in range | 16.5* | 1.95x |
| 127.3k | 48322 | 385.9 | 125.21 | 26.9 | 20-30 | in range | 13.8* | 1.95x |

read tok/s = new prompt tokens / time to first token (all-in). Gen target: >= 40 tok/s at the start, >= 20-30 at 127.5k.
MLX gen: its ctxbench run at that prompt size (cold prompt, same essay task); \* = from BRIEF.md.

ctxbench (temp 0.7, 400 tokens), tok/s, one run each

| context | mlx prefill | mlx gen | inkling prefill | inkling gen | inkling-8bit prefill | inkling-8bit gen | inkling/mlx prefill | inkling/mlx gen | inkling-8bit/mlx prefill | inkling-8bit/mlx gen |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.2k | 564.8 | 23.1 | 808.3 | 38.5 | 797.9 | 44.4 | 1.43x | 1.67x | 1.41x | 1.92x |
| 9.5k | 602.5 | 24.2 | 830.7 | 38.4 | 824.2 | 42.5 | 1.38x | 1.59x | 1.37x | 1.76x |
| 18.9k | 571* | 23.3* | 795.0 | 36.9 | 789.1 | 41.6 | 1.39x | 1.58x | 1.38x | 1.79x |
| 38.6k | 495.4 | 22.0 | 722.8 | 32.4 | 717.2 | 38.2 | 1.46x | 1.47x | 1.45x | 1.74x |
| 78.1k | 385* | 16.5* | 603.3 | 30.4 | 598.7 | 33.4 | 1.57x | 1.84x | 1.56x | 2.02x |
| 127.5k | 324* | 13.8* | 497.7 | 26.2 | 495.2 | 28.8 | 1.54x | 1.90x | 1.53x | 2.09x |

inkling-8bit = the 8-bit build (Inkling-2-Mini-MLX-8bit); its ratios are against MLX on the 4-8bit build (no MLX 8-bit baseline measured).

\* from BRIEF.md's MLX table (same script and settings, not re-measured today)

agent_bench (qh system prompt + tools, ~800-token tool result per turn, temp 0, thinking on)

| server | base ctx | turns | last prompt | total wall s | sum TTFT s | median TTFT after turn 1 s | median read tok/s after turn 1 | median gen tok/s | tokens out |
|---|---|---|---|---|---|---|---|---|---|
| inkling | 0 | 10 | 12558 | 64.6 | 13.3 | 1.17 | 665 | 75.5 | 3608 |
| inkling | 30000 | 10 | 41129 | 108.3 | 53.0 | 1.52 | 518 | 76.5 | 3919 |
| inkling-8bit | 0 | 10 | 12255 | 72.7 | 13.2 | 1.17 | 665 | 63.6 | 3639 |
| inkling-8bit | 30000 | 10 | 41129 | 107.4 | 55.6 | 1.53 | 516 | 78.5 | 3919 |
| inkling-8bit copy | 0 | 6 | 8348 | 38.5 | 5.5 | 0.90 | 613 | 109.3 | 3494 |
| inkling copy | 0 | 6 | 8346 | 39.2 | 5.5 | 0.91 | 610 | 109.6 | 3491 |
| mlx | 0 | 10 | 13369 | 241.5 | 128.6 | 13.52 | 603 | 44.1 | 4654 |
| mlx | 30000 | 10 | 39152 | 742.8 | 691.5 | 70.05 | 506 | 34.9 | 1675 |
| mlx copy (batch 16) | 0 | 6 | 8349 | 128.1 | 55.9 | 10.23 | 612 | 48.6 | 3495 |

read tok/s = new prompt tokens (prompt - cached) / time to first token: what one turn's read costs, all-in.

| turn | inkling@0 TTFT s, read tok/s, gen tok/s (prompt/cached) | inkling@30000 TTFT s, read tok/s, gen tok/s (prompt/cached) | inkling-8bit@0 TTFT s, read tok/s, gen tok/s (prompt/cached) | inkling-8bit@30000 TTFT s, read tok/s, gen tok/s (prompt/cached) | inkling-8bit copy@0 TTFT s, read tok/s, gen tok/s (prompt/cached) | inkling copy@0 TTFT s, read tok/s, gen tok/s (prompt/cached) | mlx@0 TTFT s, read tok/s, gen tok/s (prompt/cached) | mlx@30000 TTFT s, read tok/s, gen tok/s (prompt/cached) |
|---|---|---|---|---|---|---|---|---|
| 1 | 2.71, 782, 78.8 (2117/0) | 39.29, 729, 68.7 (30666/2043) | 2.72, 778, 83.2 (2117/0) | 41.86, 733, 55.6 (30666/0) | 0.96, 623, 94.7 (2643/2043) | 0.96, 624, 84.7 (2643/2043) | 4.12, 514, 50.9 (2117/0) | 59.00, 520, 37.7 (30666/0) |
| 2 | 1.10, 705, 78.9 (2937/2162) | 1.46, 532, 82.8 (31496/30721) | 1.11, 699, 72.5 (2937/2162) | 1.47, 528, 74.4 (31497/30722) | 0.88, 625, 108.1 (3945/3393) | 0.88, 627, 109.1 (3946/3394) | 4.60, 638, 41.6 (2937/0) | 60.89, 517, 38.8 (31496/0) |
| 3 | 1.14, 694, 77.0 (4238/3448) | 1.47, 528, 56.3 (32330/31555) | 1.12, 692, 70.4 (4012/3237) | 1.46, 531, 60.1 (32332/31557) | 0.89, 617, 109.6 (5063/4511) | 0.90, 615, 110.1 (5063/4511) | 6.93, 612, 32.6 (4241/0) | 62.79, 515, 28.4 (32331/0) |
| 4 | 1.15, 685, 59.9 (5539/4749) | 1.47, 526, 55.9 (33323/32548) | 1.13, 686, 59.3 (4889/4114) | 1.46, 529, 52.9 (33323/32548) | 0.90, 613, 109.4 (6158/5606) | 0.91, 610, 108.9 (6158/5606) | 8.96, 619, 39.7 (5545/0) | 65.56, 512, 34.9 (33566/0) |
| 5 | 1.17, 675, 64.7 (6840/6050) | 1.51, 524, 71.3 (34624/33834) | 1.15, 674, 65.2 (5952/5177) | 1.52, 521, 66.2 (34624/33834) | 0.93, 594, 109.5 (7253/6701) | 0.92, 599, 111.9 (7252/6700) | 11.20, 612, 41.5 (6849/0) | 67.92, 507, 29.5 (34437/0) |
| 6 | 1.17, 665, 56.4 (7914/7139) | 1.52, 518, 81.7 (35925/35135) | 1.17, 665, 53.4 (7179/6404) | 1.53, 516, 82.5 (35925/35135) | 0.93, 592, 109.1 (8348/7796) | 0.93, 593, 111.3 (8346/7794) | 13.52, 603, 47.2 (8153/0) | 70.05, 506, 31.8 (35436/0) |
| 7 | 1.20, 658, 67.9 (9215/8425) | 1.54, 512, 84.2 (37226/36436) | 1.19, 663, 56.5 (8480/7690) | 1.54, 512, 87.2 (37226/36436) | - | - | 15.70, 602, 47.4 (9457/0) | 73.40, 495, 35.0 (36365/0) |
| 8 | 1.21, 652, 73.9 (10516/9726) | 1.56, 506, 54.7 (38527/37737) | 1.21, 654, 63.2 (9781/8991) | 1.56, 506, 89.8 (38527/37737) | - | - | 18.21, 591, 46.5 (10761/0) | 74.90, 498, 38.3 (37294/0) |
| 9 | 1.22, 644, 94.4 (11537/10749) | 1.57, 503, 83.4 (39828/39038) | 1.20, 648, 59.1 (10954/10179) | 1.59, 498, 91.1 (39828/39038) | - | - | 21.62, 558, 46.5 (12065/0) | 77.59, 493, 33.5 (38223/0) |
| 10 | 1.23, 639, 103.1 (12558/11770) | 1.59, 496, 88.1 (41129/40339) | 1.23, 645, 64.1 (12255/11465) | 1.60, 495, 89.8 (41129/40339) | - | - | 23.74, 563, 33.5 (13369/0) | 79.39, 493, 39.9 (39152/0) |

## qh end-to-end

`tools/qh_e2e.sh`: qh fixes `/tmp/inkling-qh-test` (5 failing unit tests) in a fresh clone.

| server | result | wall time |
|---|---|---|
| inkling-8bit (batch 20, two runs) | 5/5 tests pass | 15 s (4 steps, 757 tokens out), 15 s (5 steps, 718 tokens out) |
| inkling-8bit with nib (batch 20, two runs) | 5/5 tests pass | 14 s (4 steps, 714 tokens out), 11 s (4 steps, 640 tokens out) |
| inkling-8bit (batch 13, two runs) | 5/5 tests pass | 13 s (3 steps, 587 tokens out), 18 s (4 steps, 831 tokens out) |
| inkling-8bit (batch 9) | 5/5 tests pass | 28 s (4 steps, 1331 tokens out) |
| inkling-8bit (batch 8) | 5/5 tests pass | 17 s (5 steps, 643 tokens out) |
| inkling-8bit (batch 7) | 5/5 tests pass | 20 s |
| inkling-8bit (batch 5) | 5/5 tests pass | 16 s |
| MLX-VLM (two runs, 17:45) | 5/5 tests pass | 57 s, 40 s |

## Reading the tables

- **Gen target column**: the user gave two points (40 tok/s at the start, 20-30 at 127.5k); the values in between
  are a straight line between them, drawn by us. "in range" = above the lower line, below the upper one.

- **Read speed in the target table is at depth**: the tokens of each step are read on top of everything before
  them, so the 127.3k row (383 tok/s) is the speed of reading tokens 79k-127k, not an average from zero. MLX's
  prefill numbers in the cold sweep are averages over the whole prompt, so the two columns are not the same
  measurement; the cold sweep is the like-for-like one.
- **Agent sessions**: MLX-VLM re-reads the whole prompt every turn (0 cached tokens), inkling-server reads only
  the new tokens, which is why time to first token stays near 1.2 s while MLX's grows with the context.
- Agent gen tok/s is at temperature 0 on text with repetition (drafts are accepted often); the target table is
  temperature 0.7 on free text. Different workloads.
- MLX-VLM ignores `reasoning_effort` from the request, so in the essay runs it thought at the template default
  while inkling-server ran at "none". Speeds are comparable; the answers are not.
