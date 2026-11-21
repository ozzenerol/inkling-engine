#!/usr/bin/env python3
"""Diff the C++ chat template (build/render_main) against transformers' apply_chat_template on random
and hand-written cases. Owner: C.

usage: ~/llm/inkling-venv/bin/python tests/server/template_check.py [N_RANDOM] [BINARY]
"""
import copy
import json
import random
import subprocess
import sys

from transformers import AutoTokenizer

MODEL = "/Users/ozzenerol/llm/mlx-models/Inkling-2-Mini-MLX-4-8bit"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
BIN = sys.argv[2] if len(sys.argv) > 2 else "build-c/src/server/render_main"
rng = random.Random(1234)

WORDS = ["hello", "world", "naïve", "日本語", "emoji 🎉", "tab\there", "line\nbreak", "quote \"x\"", "back\\slash",
         "<html>&amp;", "\u0001ctl", "  spaced  ", "", "def f(x):\n    return x\n", "\r\n", "€", "a/b"]


def rtext():
    return "".join(rng.choice(WORDS) + rng.choice(["", " ", "\n"]) for _ in range(rng.randint(0, 5)))


def rscalar():
    k = rng.randint(0, 9)
    if k == 0: return None
    if k == 1: return rng.choice([True, False])
    if k == 2: return rng.randint(-10**20, 10**20)
    if k == 3: return rng.choice([0.1, 1.0, -2.5, 1e-7, 1e16, 1.5e16, 123456789.125, 3.0e-5, 0.0001, 1e22, -0.0, 7.0])
    if k == 4: return rng.random() * 10 ** rng.randint(-8, 20)
    return rtext()


def rvalue(d=0):
    k = rng.randint(0, 6 if d < 3 else 0)
    if k == 5: return [rvalue(d + 1) for _ in range(rng.randint(0, 3))]
    if k == 6: return {rng.choice(["b", "a", "Z", "é", "key", "a b", "_", "10", "9"]) + str(rng.randint(0, 3)): rvalue(d + 1)
                       for _ in range(rng.randint(0, 4))}
    return rscalar()


def rtool():
    fn = {"name": rng.choice(["bash", "read", "write_file", "web-search", "x"])}
    if rng.random() < 0.8: fn["description"] = rng.choice([rtext(), "", None])
    if rng.random() < 0.8:
        fn["parameters"] = {"type": "object", "properties": {"zeta": {"type": "string", "description": rtext()},
                                                               "alpha": {"type": "integer", "enum": [3, 1, 2]},
                                                               "m": rvalue()},
                            "required": ["zeta"]}
    t = {"type": "function", "function": fn}
    if rng.random() < 0.1: t = fn  # bare function
    if rng.random() < 0.1: t["type"] = ""
    return t


def rcall(i):
    args = {rng.choice(["path", "command", "b", "a", "Ä", "nested"]) + str(j): rvalue() for j in range(rng.randint(0, 4))}
    return {"id": f"call_{i}", "type": "function",
            "function": {"name": rng.choice(["bash", "read", "edit"]), "arguments": args}}


def rmessages():
    msgs = []
    if rng.random() < 0.6: msgs.append({"role": "system", "content": rtext()})
    if rng.random() < 0.2: msgs.append({"role": "system", "content": rtext()})
    calls = []
    for _ in range(rng.randint(0, 6)):
        r = rng.choice(["user", "assistant", "tool", "user", "assistant"])
        m = {"role": r}
        if r == "user":
            m["content"] = rtext() if rng.random() < 0.85 else [{"type": "text", "text": rtext()}, rtext(),
                                                                  {"text": rtext()}]
        elif r == "assistant":
            c = rng.random()
            m["content"] = rtext() if c < 0.6 else ("" if c < 0.8 else None)
            if rng.random() < 0.5: m["reasoning_content"] = rtext()
            if rng.random() < 0.5:
                m["tool_calls"] = [rcall(len(calls) + j) for j in range(rng.randint(1, 3))]
                calls += m["tool_calls"]
        else:
            m["content"] = rtext()
            if calls and rng.random() < 0.7: m["tool_call_id"] = rng.choice(calls)["id"]
            elif rng.random() < 0.5: m["name"] = rng.choice(["bash", "read"])
        msgs.append(m)
    return msgs


def stringify_args(msgs):
    msgs = copy.deepcopy(msgs)
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            tc["function"]["arguments"] = json.dumps(tc["function"]["arguments"], ensure_ascii=rng.random() < 0.5)
    return msgs


cases = []
FIXED = [
    {"messages": [{"role": "user", "content": "hi"}], "tools": None, "reasoning_effort": None},
    {"messages": [{"role": "user", "content": "hi"}], "tools": None, "reasoning_effort": "none"},
    {"messages": [{"role": "user", "content": "hi"}], "tools": None, "reasoning_effort": " low "},
    {"messages": [{"role": "user", "content": "hi"}], "tools": None, "reasoning_effort": 0.5},
    {"messages": [{"role": "user", "content": "hi"}], "tools": None, "reasoning_effort": 0},
    {"messages": [{"role": "user", "content": "hi"}], "tools": None, "reasoning_effort": 1},
    {"messages": [{"role": "user", "content": "hi"}], "tools": None, "reasoning_effort": "bogus"},
    {"messages": [{"role": "system", "content": "s"}], "tools": None, "reasoning_effort": "max"},
    {"messages": [], "tools": None, "reasoning_effort": "xhigh"},
    {"messages": [{"role": "bot", "content": "x"}], "tools": None, "reasoning_effort": None},
    {"messages": [{"role": "assistant", "content": "x", "name": "bash",
                   "tool_calls": [{"function": {"name": "read", "arguments": {}}}]}], "tools": None,
     "reasoning_effort": None},
]
for c in FIXED:
    cases.append((c, c))
for _ in range(N):
    msgs = rmessages()
    eff = rng.choice([None, "none", "low", "medium", "high", "minimal", 0.3, 0.0, 0.99])
    tools = [rtool() for _ in range(rng.randint(0, 3))] if rng.random() < 0.6 else None
    py = {"messages": msgs, "tools": tools, "reasoning_effort": eff}
    cpp = {"messages": stringify_args(msgs) if rng.random() < 0.7 else msgs, "tools": tools, "reasoning_effort": eff}
    cases.append((py, cpp))

tok = AutoTokenizer.from_pretrained(MODEL)
want = []
for py, _ in cases:
    try:
        kw = {}
        if py["reasoning_effort"] is not None: kw["reasoning_effort"] = py["reasoning_effort"]
        want.append(("ok", tok.apply_chat_template(py["messages"], tools=py["tools"], tokenize=False, **kw)))
    except Exception as e:  # noqa: BLE001
        want.append(("err", str(e)))

out = subprocess.run([BIN], input=json.dumps([c for _, c in cases]), capture_output=True, text=True, check=True)
got = json.loads(out.stdout)
bad = 0
for i, ((kind, text), g) in enumerate(zip(want, got)):
    gk = "ok" if "ok" in g else "err"
    if kind != gk or (kind == "ok" and text != g["ok"]):
        bad += 1
        if bad <= 5:
            print(f"--- case {i} MISMATCH want {kind} got {gk}")
            print("input:", json.dumps(cases[i][0], ensure_ascii=False)[:1500])
            print("want:", repr(text)[:1500])
            print("got: ", repr(g.get("ok", g.get("err")))[:1500])
n_err = sum(1 for k, _ in want if k == "err")
print(f"{len(cases)} cases ({n_err} expected errors): {len(cases) - bad} match, {bad} mismatch")
sys.exit(1 if bad else 0)
