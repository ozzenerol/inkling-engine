#!/usr/bin/env python3
"""Time to read a short user turn after the server sat idle (the weights get unwired after ~2 s without a submit;
INK_WARM_SECS keeps them wired). usage: idle_bench.py [url] [idle seconds...]"""
import json, sys, time, urllib.request

args = sys.argv[1:]
url = (args.pop(0) if args and args[0].startswith("http") else "http://127.0.0.1:8091") + "/v1/chat/completions"
idles = [int(x) for x in args] or [0, 0, 2, 5, 0, 20]
msgs = [{"role": "system", "content": "You are a helpful assistant. " + "Answer briefly. " * 150}]


def turn(text):
    msgs.append({"role": "user", "content": text})
    body = {"messages": msgs, "max_tokens": 24, "temperature": 0, "reasoning_effort": 0.1}
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time()
    r = json.load(urllib.request.urlopen(req))
    dt = time.time() - t
    m = r["choices"][0]["message"]
    a = {"role": "assistant", "content": m.get("content") or ""}
    if m.get("reasoning_content"):
        a["reasoning_content"] = m["reasoning_content"]
    msgs.append(a)
    return dt, r.get("timings", {})


turn("Say hi.")
for idle in idles:
    time.sleep(idle)
    dt, tm = turn("Count to three.")
    print(f"idle: after {idle:3d} s: read {tm.get('prompt_n')} tok in {tm.get('prompt_ms', 0):6.1f} ms, "
          f"wrote at {tm.get('predicted_per_second', 0):5.1f} tok/s, request {dt * 1000:5.0f} ms", flush=True)
