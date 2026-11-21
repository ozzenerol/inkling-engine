# Results (2026-11-22)

Inkling 2 Mini on the M3 Max (48 GB), `inkling-server` on port 8091 against MLX-VLM 0.8.0+inkling2mini.2 with its
DSpark draft (block 4). **Every number is one run.** `inkling-8bit` is the server's default model
(`Inkling-2-Mini-MLX-8bit`); `inkling` is the lossless 4/8-bit build. MLX always ran the 4/8-bit build.

Target: each prompt is read fast; generation at least 40 tok/s at the start of the context and 20-30 tok/s at
127.5k.

Where the numbers come from: everything below except the MLX rows = batch 13 (main 0fb5072, 2026-11-22 00:30-01:12,
`MLX=1 tools/batch.sh queues full`; server defaults: 8-bit model, draft block 8, lookup drafts on, direct global
attention, read/write hazard sets, MoE prefill tiles by T, system prompt + tools prefix pinned on the first
prefill). Batches 10, 11 and 12 gave the same 8-bit target table within 4.2 tok/s of generation per row (four
runs). MLX = measured 2026-11-21 (17:43 and 19:10), rows marked \* from BRIEF.md.
Not measured: MLX on the copy scenario (its server answered 503 while still loading; the bench now waits), MLX in
a growing session.

Batch 9 (2026-11-21, before the three defaults) for comparison, 8-bit target table: gen 39.9 / 38.1 / 30.3 / 27.2 /
22.0 tok/s and read 729 / 794 / 658 / 503 / 381 tok/s at 1.2k / 9.6k / 39.3k / 78.6k / 127.3k.

Where the time to first token goes (server log `[ttft]`): prefill is 93-99% of it at every depth. An agent turn
that reads 775 new tokens takes 1.1-1.3 s. Short chunks cost more per token than chunks of 2000 because a
chunk has a fixed cost of about 180 ms (profiles at 800 and 2048 tokens fit 179 ms + 1.055 ms per token): every
visit reads all 128 experts whatever the chunk size. A second session with the same system prompt and tools
restores the pinned prefix: 2043 of 2643 tokens cached, time to first token 0.99 s instead of 3.4 s.

Target benchmark, inkling-8bit: read speed and gen tok/s by context depth (one growing session, essay prompt, temp 0.7, 400 tokens, one run)

| context | new tokens read | read tok/s | TTFT s | gen tok/s | gen target | meets | MLX gen | vs MLX |
|---|---|---|---|---|---|---|---|---|
| 1.2k | 1194 | 813.0 | 1.47 | 42.3 | 40 | yes | 23.1 | 1.83x |
| 9.6k | 8013 | 800.3 | 10.01 | 40.7 | 39 | yes | 24.2 | 1.68x |
| 39.3k | 29302 | 666.2 | 43.99 | 37.4 | 34-37 | yes | 22.0 | 1.70x |
| 78.6k | 38886 | 507.4 | 76.64 | 32.0 | 28-34 | in range | 16.5* | 1.94x |
| 127.3k | 48322 | 382.8 | 126.23 | 29.0 | 20-30 | in range | 13.8* | 2.10x |

read tok/s = new prompt tokens / time to first token (all-in). Gen target: >= 40 tok/s at the start, >= 20-30 at 127.5k.
MLX gen: its ctxbench run at that prompt size (cold prompt, same essay task); \* = from BRIEF.md.

Target benchmark, inkling: read speed and gen tok/s by context depth (one growing session, essay prompt, temp 0.7, 400 tokens, one run)

| context | new tokens read | read tok/s | TTFT s | gen tok/s | gen target | meets | MLX gen | vs MLX |
|---|---|---|---|---|---|---|---|---|
| 1.2k | 1194 | 809.5 | 1.48 | 35.2 | 40 | no | 23.1 | 1.52x |
| 9.6k | 8013 | 809.9 | 9.89 | 36.0 | 39 | no | 24.2 | 1.49x |
| 39.3k | 29302 | 669.8 | 43.75 | 32.5 | 34-37 | no | 22.0 | 1.48x |
| 78.6k | 38886 | 509.4 | 76.34 | 28.7 | 28-34 | in range | 16.5* | 1.74x |
| 127.3k | 48322 | 384.8 | 125.58 | 25.5 | 20-30 | in range | 13.8* | 1.85x |

read tok/s = new prompt tokens / time to first token (all-in). Gen target: >= 40 tok/s at the start, >= 20-30 at 127.5k.
MLX gen: its ctxbench run at that prompt size (cold prompt, same essay task); \* = from BRIEF.md.

ctxbench (temp 0.7, 400 tokens), tok/s, one run each

| context | mlx prefill | mlx gen | inkling prefill | inkling gen | inkling-8bit prefill | inkling-8bit gen | inkling/mlx prefill | inkling/mlx gen | inkling-8bit/mlx prefill | inkling-8bit/mlx gen |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.2k | 564.8 | 23.1 | 760.9 | 37.2 | 804.5 | 41.8 | 1.35x | 1.61x | 1.42x | 1.81x |
| 9.5k | 602.5 | 24.2 | 819.4 | 36.1 | 818.5 | 40.7 | 1.36x | 1.49x | 1.36x | 1.68x |
| 18.9k | 571* | 23.3* | 783.1 | 33.8 | 780.7 | 40.0 | 1.37x | 1.45x | 1.37x | 1.72x |
| 38.6k | 495.4 | 22.0 | 716.4 | 31.3 | 712.4 | 37.2 | 1.45x | 1.42x | 1.44x | 1.69x |
| 78.1k | 385* | 16.5* | 600.1 | 28.4 | 596.1 | 33.2 | 1.56x | 1.72x | 1.55x | 2.01x |
| 127.5k | 324* | 13.8* | 495.3 | 25.2 | 493.5 | 28.1 | 1.53x | 1.83x | 1.52x | 2.04x |

inkling-8bit = the 8-bit build (Inkling-2-Mini-MLX-8bit); its ratios are against MLX on the 4-8bit build (no MLX 8-bit baseline measured).

\* from BRIEF.md's MLX table (same script and settings, not re-measured today)

agent_bench (qh system prompt + tools, ~800-token tool result per turn, temp 0, thinking on)

| server | base ctx | turns | last prompt | total wall s | sum TTFT s | median TTFT after turn 1 s | median read tok/s after turn 1 | median gen tok/s | tokens out |
|---|---|---|---|---|---|---|---|---|---|
| inkling | 0 | 10 | 12558 | 70.1 | 13.3 | 1.17 | 664 | 69.3 | 3608 |
| inkling | 30000 | 10 | 41129 | 114.6 | 53.6 | 1.55 | 511 | 69.5 | 3919 |
| inkling-8bit | 0 | 10 | 12255 | 83.1 | 13.6 | 1.19 | 659 | 52.8 | 3639 |
| inkling-8bit | 30000 | 10 | 41129 | 118.5 | 56.3 | 1.56 | 505 | 69.6 | 3919 |
| inkling-8bit copy | 0 | 6 | 8348 | 45.2 | 5.6 | 0.93 | 596 | 91.8 | 3494 |
| inkling copy | 0 | 6 | 8346 | 41.2 | 5.5 | 0.90 | 617 | 104.4 | 3491 |
| mlx | 0 | 10 | 13369 | 241.5 | 128.6 | 13.52 | 603 | 44.1 | 4654 |
| mlx | 30000 | 10 | 39152 | 742.8 | 691.5 | 70.05 | 506 | 34.9 | 1675 |

read tok/s = new prompt tokens (prompt - cached) / time to first token: what one turn's read costs, all-in.

| turn | inkling@0 TTFT s, read tok/s, gen tok/s (prompt/cached) | inkling@30000 TTFT s, read tok/s, gen tok/s (prompt/cached) | inkling-8bit@0 TTFT s, read tok/s, gen tok/s (prompt/cached) | inkling-8bit@30000 TTFT s, read tok/s, gen tok/s (prompt/cached) | inkling-8bit copy@0 TTFT s, read tok/s, gen tok/s (prompt/cached) | inkling copy@0 TTFT s, read tok/s, gen tok/s (prompt/cached) | mlx@0 TTFT s, read tok/s, gen tok/s (prompt/cached) | mlx@30000 TTFT s, read tok/s, gen tok/s (prompt/cached) |
|---|---|---|---|---|---|---|---|---|
| 1 | 2.71, 782, 73.7 (2117/0) | 39.63, 722, 65.0 (30666/2043) | 2.77, 765, 109.8 (2117/0) | 42.24, 726, 52.3 (30666/0) | 0.99, 604, 77.4 (2643/2043) | 0.99, 608, 78.1 (2643/2043) | 4.12, 514, 50.9 (2117/0) | 59.00, 520, 37.7 (30666/0) |
| 2 | 1.10, 705, 71.4 (2937/2162) | 1.50, 518, 76.8 (31496/30721) | 1.14, 680, 58.6 (2937/2162) | 1.51, 513, 79.4 (31497/30722) | 0.90, 611, 89.4 (3945/3393) | 0.89, 620, 102.4 (3946/3394) | 4.60, 638, 41.6 (2937/0) | 60.89, 517, 38.8 (31496/0) |
| 3 | 1.14, 690, 68.9 (4238/3448) | 1.49, 521, 51.7 (32330/31555) | 1.15, 674, 63.0 (4012/3237) | 1.49, 520, 47.7 (32332/31557) | 0.89, 618, 92.3 (5063/4511) | 0.88, 625, 104.8 (5063/4511) | 6.93, 612, 32.6 (4241/0) | 62.79, 515, 28.4 (32331/0) |
| 4 | 1.16, 681, 54.3 (5539/4749) | 1.52, 511, 48.4 (33323/32548) | 1.16, 667, 49.7 (4889/4114) | 1.51, 512, 43.7 (33323/32548) | 0.93, 596, 91.3 (6158/5606) | 0.90, 617, 103.9 (6158/5606) | 8.96, 619, 39.7 (5545/0) | 65.56, 512, 34.9 (33566/0) |
| 5 | 1.15, 685, 62.4 (6840/6050) | 1.52, 520, 64.8 (34624/33834) | 1.18, 659, 53.6 (5952/5177) | 1.57, 502, 54.7 (34624/33834) | 0.93, 596, 92.6 (7253/6701) | 0.91, 606, 106.4 (7252/6700) | 11.20, 612, 41.5 (6849/0) | 67.92, 507, 29.5 (34437/0) |
| 6 | 1.17, 664, 49.3 (7914/7139) | 1.55, 511, 74.0 (35925/35135) | 1.23, 628, 50.4 (7179/6404) | 1.56, 505, 68.0 (35925/35135) | 0.95, 584, 92.4 (8348/7796) | 0.91, 604, 105.9 (8346/7794) | 13.52, 603, 47.2 (8153/0) | 70.05, 506, 31.8 (35436/0) |
| 7 | 1.20, 661, 60.0 (9215/8425) | 1.57, 503, 77.8 (37226/36436) | 1.19, 666, 48.2 (8480/7690) | 1.56, 507, 71.2 (37226/36436) | - | - | 15.70, 602, 47.4 (9457/0) | 73.40, 495, 35.0 (36365/0) |
| 8 | 1.21, 652, 69.7 (10516/9726) | 1.61, 492, 49.1 (38527/37737) | 1.24, 638, 52.9 (9781/8991) | 1.60, 495, 76.6 (38527/37737) | - | - | 18.21, 591, 46.5 (10761/0) | 74.90, 498, 38.3 (37294/0) |
| 9 | 1.22, 648, 87.1 (11537/10749) | 1.60, 495, 78.6 (39828/39038) | 1.24, 625, 50.2 (10954/10179) | 1.64, 483, 76.1 (39828/39038) | - | - | 21.62, 558, 46.5 (12065/0) | 77.59, 493, 33.5 (38223/0) |
| 10 | 1.23, 640, 93.1 (12558/11770) | 1.65, 479, 81.5 (41129/40339) | 1.27, 620, 52.8 (12255/11465) | 1.62, 489, 75.5 (41129/40339) | - | - | 23.74, 563, 33.5 (13369/0) | 79.39, 493, 39.9 (39152/0) |

## qh end-to-end

`tools/qh_e2e.sh`: qh fixes `/tmp/inkling-qh-test` (5 failing unit tests) in a fresh clone.

| server | result | wall time |
|---|---|---|
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
