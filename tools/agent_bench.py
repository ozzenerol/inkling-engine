#!/usr/bin/env python3
"""Owner: C. Agentic multi-turn latency: the same qh-like session against any OpenAI-compatible server.

The session uses qh's real system prompt and tool list (tests/server/fixtures/qh_request.json), sends every
answer back the way qh does (content, trimmed reasoning_content, tool_calls), and adds a scripted tool result of
about --tool-tokens tokens per turn (if the model did not call a tool, the same text comes as a user message).
So both servers see a prompt that grows the same way, whatever the model writes. Per turn: time to first
streamed delta of any kind (TTFT), wall time, prompt / cached / completion tokens.

  agent_bench.py --url http://127.0.0.1:8091 --label inkling --turns 10 [--base-tokens 30000] --out X.json
  agent_bench.py --report A.json B.json     markdown table
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "..", "tests", "server", "fixtures", "qh_request.json")
TASK = ("The project in /work/repo has a slow, buggy order pipeline split over many files. I will show you the "
        "files one at a time with the tools. After each file, say in one or two sentences what is wrong with it, "
        "then read the next one (read step<N+1>.py). Keep answers short.")


def file_text(k, n_tokens):
    """Deterministic Python source of about n_tokens tokens (2.46 chars per token, measured)."""
    lines = [f"# step{k}.py: stage {k} of the order pipeline"]
    i = 0
    while len("\n".join(lines)) < n_tokens * 2.46:
        lines.append(f"def stage{k}_{i}(orders, rate={i % 7 + 1}):\n"
                     f"    total = 0\n"
                     f"    for o in orders:\n"
                     f"        if o.qty > {i * 3 % 11}:\n"
                     f"            total += o.qty * rate  # step {k} rule {i}\n"
                     f"    return total - {i}\n")
        i += 1
    return "\n".join(lines)


def run_turn(url, model, messages, tools, max_tokens, temperature):
    body = {"model": model, "messages": messages, "tools": tools, "stream": True, "max_tokens": max_tokens,
            "temperature": temperature, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    reasoning, content, calls, usage, finish = "", "", {}, {}, None
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            usage = d.get("usage") or usage
            for c in d.get("choices", []):
                dl = c.get("delta", {})
                piece = (dl.get("reasoning_content") or dl.get("reasoning") or "") + (dl.get("content") or "")
                if (piece or dl.get("tool_calls")) and first is None:
                    first = time.perf_counter()
                reasoning += dl.get("reasoning_content") or dl.get("reasoning") or ""
                content += dl.get("content") or ""
                for tc in dl.get("tool_calls") or []:
                    cur = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
                    cur["id"] = tc.get("id") or cur["id"]
                    fn = tc.get("function") or {}
                    cur["name"] += fn.get("name") or ""
                    cur["arguments"] += fn.get("arguments") or ""
                finish = c.get("finish_reason") or finish
    t1 = time.perf_counter()
    return {"ttft": (first or t1) - t0, "wall": t1 - t0, "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "completion_tokens": usage.get("completion_tokens"), "finish": finish,
            "reasoning": reasoning, "content": content, "calls": [calls[k] for k in sorted(calls)]}


def bench(args):
    fx = json.load(open(FIX))
    models = json.load(urllib.request.urlopen(args.url + "/v1/models"))["data"]
    model = next((m["id"] for m in models if m["id"].startswith("/")), models[0]["id"] if models else args.model)
    messages = [{"role": "system", "content": fx["system"]}]
    if args.base_tokens:
        messages.append({"role": "user", "content": "Earlier I pasted the whole old version of the project, for "
                         "reference:\n" + "\n".join(file_text(1000 + j, 2000) for j in range(args.base_tokens // 2000))})
        messages.append({"role": "assistant", "content": "Understood. I have the old version for reference."})
    messages.append({"role": "user", "content": TASK})
    turns = []
    t_start = time.perf_counter()
    for k in range(args.turns):
        r = run_turn(args.url, model, messages, fx["tools"], args.max_tokens, args.temperature)
        a = {"role": "assistant", "content": r["content"]}
        if r["reasoning"].strip():
            a["reasoning_content"] = r["reasoning"].strip()
        text = file_text(k, args.tool_tokens)
        if r["calls"]:
            a["tool_calls"] = [{"id": c["id"] or f"call_{k}_{i}", "type": "function",
                                "function": {"name": c["name"], "arguments": c["arguments"]}}
                               for i, c in enumerate(r["calls"])]
            messages.append(a)
            for i, c in enumerate(a["tool_calls"]):
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": text if i == 0 else "ok"})
        else:
            messages.append(a)
            messages.append({"role": "user", "content": f"Here is step{k}.py:\n{text}\nWhat is wrong with it?"})
        row = {x: r[x] for x in ("ttft", "wall", "prompt_tokens", "cached_tokens", "completion_tokens", "finish")}
        row["tool_calls"] = len(r["calls"])
        turns.append(row)
        print(f"  {args.label} turn {k + 1}: prompt {row['prompt_tokens']}, cached {row['cached_tokens']}, "
              f"ttft {row['ttft']:.2f} s, wall {row['wall']:.2f} s, out {row['completion_tokens']} "
              f"({row['finish']}, {row['tool_calls']} calls)", flush=True)
    total = time.perf_counter() - t_start
    return {"label": args.label, "url": args.url, "date": time.ctime(), "turns": turns, "total_wall": total,
            "base_tokens": args.base_tokens, "tool_tokens": args.tool_tokens, "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "ttft_median_after_1": statistics.median([t["ttft"] for t in turns[1:]]) if len(turns) > 1 else None,
            "ttft_sum": sum(t["ttft"] for t in turns),
            "completion_tokens": sum(t["completion_tokens"] or 0 for t in turns)}


def report(paths):
    rs = [json.load(open(p)) for p in paths]
    print("| server | base ctx | turns | last prompt | total wall s | sum TTFT s | median TTFT after turn 1 s | "
          "tokens out |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rs:
        print(f"| {r['label']} | {r['base_tokens']} | {len(r['turns'])} | {r['turns'][-1]['prompt_tokens']} | "
              f"{r['total_wall']:.1f} | {r['ttft_sum']:.1f} | {r['ttft_median_after_1']:.2f} | "
              f"{r['completion_tokens']} |")
    print()
    print("| turn | " + " | ".join(f"{r['label']}@{r['base_tokens']} TTFT s (prompt/cached)" for r in rs) + " |")
    print("|---|" + "---|" * len(rs))
    for k in range(max(len(r["turns"]) for r in rs)):
        cells = []
        for r in rs:
            if k < len(r["turns"]):
                t = r["turns"][k]
                cells.append(f"{t['ttft']:.2f} ({t['prompt_tokens']}/{t['cached_tokens']})")
            else:
                cells.append("-")
        print(f"| {k + 1} | " + " | ".join(cells) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8091")
    ap.add_argument("--label", default="server")
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--tool-tokens", type=int, default=800)
    ap.add_argument("--base-tokens", type=int, default=0, help="extra context before the task, in ~tokens")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--model", default=os.path.expanduser("~/llm/mlx-models/Inkling-2-Mini-MLX-4-8bit"),
                    help="model id when /v1/models lists none (MLX before its first load)")
    ap.add_argument("--out")
    ap.add_argument("--report", nargs="+")
    args = ap.parse_args()
    if args.report:
        report(args.report)
        return 0
    res = bench(args)
    print(f"  {args.label}: total {res['total_wall']:.1f} s, sum TTFT {res['ttft_sum']:.1f} s, "
          f"median TTFT after turn 1 {res['ttft_median_after_1']:.2f} s", flush=True)
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
