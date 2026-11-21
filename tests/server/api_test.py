#!/usr/bin/env python3
"""End-to-end API test of inkling-server against its --stub engine (no GPU). Owner: C.
usage: python3 tests/server/api_test.py [BIN] [PORT]
Starts the server, runs the checks, stops it. Exit 1 on failure."""
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

BIN = sys.argv[1] if len(sys.argv) > 1 else "build-c/src/server/inkling-server"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 18091
URL = f"http://127.0.0.1:{PORT}"
fails = 0


def check(cond, what):
    global fails
    print(("ok   " if cond else "FAIL ") + what)
    if not cond:
        fails += 1


def post(path, body, raw=False):
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    req = urllib.request.Request(URL + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, (r.read() if raw else json.loads(r.read()))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def stream(body):
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    out = {"reasoning": "", "content": "", "calls": {}, "finish": None, "usage": None, "done": False, "chunks": 0,
           "role": None}
    with urllib.request.urlopen(req, timeout=60) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                out["done"] = True
                break
            d = json.loads(data)
            if d.get("usage"):
                out["usage"] = d["usage"]
            for c in d.get("choices", []):
                out["chunks"] += 1
                dl = c.get("delta", {})
                out["role"] = out["role"] or dl.get("role")
                out["reasoning"] += dl.get("reasoning_content") or ""
                out["content"] += dl.get("content") or ""
                for tc in dl.get("tool_calls") or []:
                    cur = out["calls"].setdefault(tc["index"], {"id": "", "name": "", "arguments": ""})
                    cur["id"] = tc.get("id") or cur["id"]
                    cur["name"] += tc["function"].get("name") or ""
                    cur["arguments"] += tc["function"].get("arguments") or ""
                if c.get("finish_reason"):
                    out["finish"] = c["finish_reason"]
    return out


TOOLS = [{"type": "function", "function": {"name": "bash", "description": "Run a shell command",
                                            "parameters": {"type": "object", "properties": {
                                                "command": {"type": "string"}}, "required": ["command"]}}},
         {"type": "function", "function": {"name": "read", "description": "Read a file",
                                            "parameters": {"type": "object",
                                                           "properties": {"path": {"type": "string"}}}}}]

srv = subprocess.Popen([BIN, "--stub", "--port", str(PORT), "--stub-decode-ms", "1"], stderr=subprocess.PIPE, text=True)
try:
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", PORT), 0.2).close()
            break
        except OSError:
            time.sleep(0.05)
    m = json.load(urllib.request.urlopen(URL + "/v1/models"))
    check(m["data"][0]["id"].startswith("/"), "models lists the model path")

    st, r = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "hello there"}]})
    msg = r["choices"][0]["message"]
    check(st == 200 and "Stub reply" in msg["content"] and "🎉" in msg["content"], "non-stream content")
    check(msg.get("reasoning_content", "").startswith("Stub thinking"), "non-stream reasoning_content")
    check(r["choices"][0]["finish_reason"] == "stop", "finish stop")
    check(r["usage"]["prompt_tokens"] > 5 and r["usage"]["completion_tokens"] > 5, "usage")

    s = stream({"messages": [{"role": "user", "content": "hello stream"}], "tools": TOOLS})
    check(s["role"] == "assistant" and s["done"] and s["finish"] == "stop", "stream role/finish/[DONE]")
    check("Stub reply" in s["content"] and "<|" not in s["content"], "stream content has no special tokens")
    check(s["reasoning"].startswith("Stub thinking") and "<|" not in s["reasoning"], "stream reasoning")
    check(s["usage"] and s["usage"]["completion_tokens"] > 0, "stream usage chunk")

    # agent loop as qh does it: tool call, tool result, answer; prompts only grow
    msgs = [{"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "please use a TOOL to list files"}]
    s = stream({"messages": msgs, "tools": TOOLS, "chat_template_kwargs": {"enable_thinking": True,
                                                                            "preserve_thinking": True}})
    check(s["finish"] == "tool_calls" and len(s["calls"]) == 1, "tool call streamed, finish tool_calls")
    call = s["calls"][0]
    args = json.loads(call["arguments"])
    check(call["name"] == "bash" and args == {"command": "ls -la", "path": "/tmp/x y", "n": 3}, "tool call parsed")
    check(list(args) == ["command", "path", "n"], "argument order kept")
    check(call["id"].startswith("call_"), "tool call id")
    first_prompt = s["usage"]["prompt_tokens"]
    msgs.append({"role": "assistant", "content": s["content"], "reasoning_content": s["reasoning"].strip(),
                 "tool_calls": [{"id": call["id"], "type": "function",
                                 "function": {"name": call["name"], "arguments": call["arguments"]}}]})
    msgs.append({"role": "tool", "tool_call_id": call["id"], "content": "a.txt\nb.txt\n"})
    s2 = stream({"messages": msgs, "tools": TOOLS})
    u = s2["usage"]
    check(s2["finish"] == "stop", "answer after tool result")
    new = u["prompt_tokens"] - u["prompt_tokens_details"]["cached_tokens"]
    check(u["prompt_tokens_details"]["cached_tokens"] >= first_prompt, f"turn 2 reuses the cache ({new} new tokens)")
    check(new < 20, "turn 2 prefills only the tool result")
    msgs.append({"role": "assistant", "content": s2["content"], "reasoning_content": s2["reasoning"].strip()})
    msgs.append({"role": "user", "content": "thanks, now explain"})
    s3 = stream({"messages": msgs, "tools": TOOLS})
    u = s3["usage"]
    new = u["prompt_tokens"] - u["prompt_tokens_details"]["cached_tokens"]
    check(new < 20, f"turn 3 prefills only the new user message ({new})")
    st, cache = 200, json.load(urllib.request.urlopen(URL + "/v1/cache"))
    check(cache["spliced_tokens_total"] > 0, f"assistant turns spliced from generated ids ({cache})")

    # spliced prompt must equal a fresh tokenization in length +- (the stub writes canonical text)
    st, t = post("/tokenize", {"messages": msgs, "tools": TOOLS})
    check(st == 200 and len(t["tokens"]) == u["prompt_tokens"], "tokenize endpoint agrees with prompt_tokens")

    # divergence: edit the first user message -> most of the cache is useless, but it must still work
    msgs2 = [dict(m) for m in msgs]
    msgs2[1] = {"role": "user", "content": "a different first message"}
    s4 = stream({"messages": msgs2, "tools": TOOLS})
    check(s4["finish"] == "stop", "diverged prompt works")
    # back to the original conversation: restored from a checkpoint or re-prefilled, still correct
    s5 = stream({"messages": msgs, "tools": TOOLS})
    check(s5["finish"] == "stop", "original conversation again")

    # retry of the same prompt: everything but one token is reused
    s6 = stream({"messages": msgs, "tools": TOOLS})
    u = s6["usage"]
    check(u["prompt_tokens"] - u["prompt_tokens_details"]["cached_tokens"] <= 1, "same prompt again: full reuse")

    # effort handling
    st, t1 = post("/tokenize", {"messages": [{"role": "user", "content": "x"}], "reasoning_effort": "none"})
    st, t2 = post("/tokenize", {"messages": [{"role": "user", "content": "x"}]})
    check(t1["tokens"] != t2["tokens"], "reasoning_effort changes the prompt")

    # client disconnect mid-stream, then the server keeps working
    sk = socket.create_connection(("127.0.0.1", PORT))
    body = json.dumps({"messages": [{"role": "user", "content": "disconnect me"}], "stream": True}).encode()
    sk.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: "
               + str(len(body)).encode() + b"\r\n\r\n" + body)
    sk.recv(100)
    sk.close()
    time.sleep(0.3)
    st, r = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "after disconnect"}]})
    check(st == 200 and "after disconnect" in r["choices"][0]["message"]["content"], "works after a client disconnect")

    # max_tokens -> finish length
    st, r = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "short"}], "max_tokens": 5})
    check(r["choices"][0]["finish_reason"] == "length" and r["usage"]["completion_tokens"] == 5, "finish length")
    st, r = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "after length"}]})
    check(st == 200 and r["choices"][0]["finish_reason"] == "stop", "works after a length stop")

    # errors
    st, r = post("/v1/chat/completions", b"{not json")
    check(st == 400 and "error" in r, "bad JSON -> 400")
    st, r = post("/v1/chat/completions", {"messages": [{"role": "robot", "content": "x"}]})
    check(st == 400 and "Unknown message role" in r["error"]["message"], "bad role -> 400")
    st, r = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "reasoning_effort": "huge"})
    check(st == 400, "bad effort -> 400")
    st, r = post("/v1/chat/completions", {"messages": []})
    check(st == 400, "empty messages -> 400")
finally:
    srv.terminate()
    err = srv.communicate(timeout=5)[1]
    if fails:
        print(err[-3000:])
print("PASS" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
