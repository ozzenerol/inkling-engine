#!/usr/bin/env python3
"""Owner: C. Markdown report of tools/bench_c.sh results: the BRIEF context table (prefill + gen tok/s) for
each server side by side with the ratio to MLX, then the agent_bench tables.
usage: bench_report.py [DIR]   (default bench/)"""
import glob
import json
import os
import subprocess
import sys

d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "bench")
# ---- depth_bench: gen tok/s at each context depth against the user's target (60 at the start, 40 at 127.5k)
mlx_ctx = []
f = os.path.join(d, "ctx-mlx.jsonl")
if os.path.exists(f):
    mlx_ctx = [json.loads(l) for l in open(f) if l.strip()]


def mlx_at(pt):
    """MLX's cold ctxbench row at about this many prompt tokens (within 15%)."""
    best = min(mlx_ctx, key=lambda r: abs((r.get("prompt_tokens") or 0) - pt), default=None)
    return best if best and abs(best["prompt_tokens"] - pt) < 0.15 * pt else None


for f in sorted(glob.glob(os.path.join(d, "depth-*.jsonl"))):
    rows = [json.loads(l) for l in open(f) if l.strip()]
    label = os.path.basename(f)[6:-6]
    print(f"Target benchmark, {label}: gen tok/s by context depth (one growing session, essay prompt, temp 0.7, "
          "400 tokens, one run)\n")
    print("| context | target gen | gen tok/s | vs target | MLX gen | vs MLX | prefill tok/s (new tokens) | TTFT s |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        if "error" in r:
            print(f"| {r['depth'] / 1000:.1f}k | - | failed: {r['error'][:60]} | - | - | - | - | - |")
            continue
        m = mlx_at(r["prompt_tokens"])
        g, t = r.get("gen_tps"), r["target_tps"]
        mark = "*" if m and m.get("source") == "BRIEF" else ""
        print(f"| {r['prompt_tokens'] / 1000:.1f}k | {t:.1f} | {g if g is not None else '-'} | "
              f"{(g / t * 100 if g else 0):.0f}% | {str(m['gen_tps']) + mark if m else '-'} | "
              f"{(f'{g / m['gen_tps']:.2f}x' if g and m else '-')} | "
              f"{r['prefill_tps']} ({r['new_tokens']}) | {r['ttft_s']} |")
    print("\nMLX gen: its ctxbench run at that prompt size (cold prompt, same essay task); \\* = from BRIEF.md.\n")

runs = {}
for f in sorted(glob.glob(os.path.join(d, "ctx-*.jsonl"))):
    label = os.path.basename(f)[4:-6]
    runs[label] = [json.loads(l) for l in open(f) if l.strip()]
labels = sorted(runs, key=lambda l: (l != "mlx", l))
if runs:
    sizes = []
    for l in labels:
        for r in runs[l]:
            if r["size"] not in sizes:
                sizes.append(r["size"])
    head = "| context |" + "".join(f" {l} prefill | {l} gen |" for l in labels)
    others = [l for l in labels if l != "mlx" and "mlx" in runs]
    head += "".join(f" {l}/mlx prefill | {l}/mlx gen |" for l in others)
    print("ctxbench (temp 0.7, 400 tokens), tok/s, one run each\n")
    print(head)
    print("|---|" + "---|" * (2 * len(labels) + 2 * len(others)))
    for s in sizes:
        row = {l: next((r for r in runs[l] if r["size"] == s), None) for l in labels}
        ctx = next((r["prompt_tokens"] for r in row.values() if r and r.get("prompt_tokens")), s)
        cells = []
        for l in labels:
            r = row[l]
            mark = "*" if r and r.get("source") == "BRIEF" else ""
            cells += [str(r.get("prefill_tps", "fail")) + mark if r else "-", str(r.get("gen_tps", "fail")) + mark if r else "-"]
        for l in others:
            a, b = row[l], row["mlx"]
            for k in ("prefill_tps", "gen_tps"):
                try:
                    cells.append(f"{a[k] / b[k]:.2f}x")
                except Exception:
                    cells.append("-")
        print(f"| {ctx / 1000:.1f}k | " + " | ".join(cells) + " |")
    if any("8bit" in l for l in labels):
        print("\ninkling-8bit = the 8-bit build (Inkling-2-Mini-MLX-8bit); its ratios are against MLX on the 4-8bit "
              "build (no MLX 8-bit baseline measured).")
    if any(r.get("source") == "BRIEF" for l in labels for r in runs[l]):
        print("\n\\* from BRIEF.md's MLX table (same script and settings, not re-measured today)")
    print()
ag = sorted(glob.glob(os.path.join(d, "agent-*.json")))
if ag:
    print("agent_bench (qh system prompt + tools, ~800-token tool result per turn, temp 0, thinking on)\n")
    sys.stdout.flush()
    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "agent_bench.py"), "--report", *ag])
