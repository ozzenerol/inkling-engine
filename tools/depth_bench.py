#!/usr/bin/env python3
"""Owner: C. THE per-batch benchmark: generation speed at increasing context depth, in one server session.

Target (user, relaxed 2026-11-21): each prompt is read fast (prefill tok/s on the new tokens, TTFT); generation
>= 40 tok/s at the beginning and >= 20-30 tok/s at the end of the context (127.5k). tools/bench_report.py prints
the table; the `target_tps` field in the rows is the earlier 60 -> 40 line, kept for reference.

Same workload as ~/llm/ctxbench.py (log-file filler, "write a detailed 400-word essay", temp 0.7, effort none,
400 tokens, streaming), but as ONE growing conversation: each turn appends a user message with more of the log
and asks for an essay on a new topic, and the earlier answers stay in the prompt. The prompt only grows, so a
server with a prefix cache prefills each depth once (127.5k tokens in total, not 255k), and generation is
measured at 1.2k / 9.5k / 38.6k / 78k / 127.5k. Topics differ per turn so no answer can be copied from an
earlier one. gen tok/s = (completion_tokens - 1) / (end - first delta), as ctxbench; prefill tok/s = new prompt
tokens / time to first delta.

  depth_bench.py --url http://127.0.0.1:8091 --label inkling-8bit --out depth-inkling-8bit.jsonl
  [--depths 1200,9500,38600,78000,127500] [--seed N]
"""
import argparse
import json
import random
import sys
import time
import urllib.request

WORDS = ("system cache vector parser thread module socket buffer index token stream config "
         "handler request kernel queue router schema signal layer memory driver").split()
TOPICS = ["how compilers optimize loops", "how garbage collectors find unreachable objects",
          "how TCP congestion control reacts to packet loss", "how B-tree indexes keep lookups fast",
          "how virtual memory and page tables work", "how a hash map handles collisions",
          "how video codecs use motion compensation"]
TOK_PER_LINE = 20.85  # measured with ctxbench on this tokenizer (17 * 1.226)
T_START, T_END = 60.0, 40.0
D_START, D_END = 1200, 127500


def target(ctx):
    f = min(1.0, max(0.0, (ctx - D_START) / (D_END - D_START)))
    return T_START + (T_END - T_START) * f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8091")
    ap.add_argument("--label", default="server")
    ap.add_argument("--depths", default="1200,9500,38600,78000,127500")
    ap.add_argument("--gen", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--model", default="")
    ap.add_argument("--out")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    salt = int(time.time())  # a fresh session every run: no reuse from an earlier run's cache
    models = json.load(urllib.request.urlopen(args.url + "/v1/models"))["data"]
    model = args.model or (models[0]["id"] if models else "")
    depths = [int(x) for x in args.depths.split(",")]
    messages = []
    have = 0  # prompt + completion tokens so far, from the server's usage
    line_no = 0
    rows = []
    tok_per_line = TOK_PER_LINE  # corrected after each turn: line numbers get longer as the log grows
    for k, depth in enumerate(depths):
        overhead = 45 if k else 60  # question, role tokens (and the effort line on the first turn)
        n_lines = max(1, round((depth - have - overhead) / tok_per_line))
        lines = []
        for _ in range(n_lines):
            lines.append(f"[{salt}-{line_no}] " + " ".join(rng.choice(WORDS) for _ in range(12)))
            line_no += 1
        head = "Here is a log file:\n" if k == 0 else "The log continues:\n"
        messages.append({"role": "user", "content": head + "\n".join(lines) + "\n\nIgnore the log. Write a detailed "
                         f"400-word essay about {TOPICS[k % len(TOPICS)]}."})
        body = {"model": model, "stream": True, "max_tokens": args.gen, "temperature": args.temperature,
                "stream_options": {"include_usage": True}, "messages": messages,
                "chat_template_kwargs": {"reasoning_effort": "none"}}
        req = urllib.request.Request(args.url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        t0 = time.perf_counter()
        first = None
        usage, text, reasoning, timings = {}, "", "", {}
        try:
            with urllib.request.urlopen(req, timeout=7200) as r:
                for raw in r:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:") or line == "data: [DONE]":
                        continue
                    d = json.loads(line[5:])
                    usage = d.get("usage") or usage
                    timings = d.get("timings") or timings
                    for c in d.get("choices", []):
                        dl = c.get("delta", {})
                        piece = dl.get("content") or ""
                        think = dl.get("reasoning_content") or dl.get("reasoning") or ""
                        if (piece or think) and first is None:
                            first = time.perf_counter()
                        text += piece
                        reasoning += think
        except Exception as e:  # noqa: BLE001
            row = {"label": args.label, "depth": depth, "error": str(e)}
            rows.append(row)
            print(json.dumps(row), flush=True)
            break
        t1 = time.perf_counter()
        pt, ct = usage.get("prompt_tokens") or 0, usage.get("completion_tokens") or 0
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        ttft = (first or t1) - t0
        row = {"label": args.label, "depth": depth, "prompt_tokens": pt, "cached_tokens": cached,
               "new_tokens": pt - cached, "completion_tokens": ct, "ttft_s": round(ttft, 2),
               "prefill_tps": round((pt - cached) / ttft, 1) if ttft > 0 else None,
               "gen_tps": round((ct - 1) / (t1 - first), 1) if first and ct > 1 and t1 > first else None,
               "target_tps": round(target(pt), 1), "cache": timings.get("cache")}
        rows.append(row)
        print(json.dumps(row), flush=True)
        a = {"role": "assistant", "content": text}
        if reasoning.strip():
            a["reasoning_content"] = reasoning.strip()
        messages.append(a)
        if pt - have > 200:
            tok_per_line = (pt - have - overhead) / n_lines * 1.004
        have = pt + ct
    if args.out:
        with open(args.out, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
