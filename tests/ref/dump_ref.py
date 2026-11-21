"""Owner: A. Dump MLX-VLM reference outputs for the C++ engine's accuracy tests.

For each prompt: token ids, greedy continuation (N tokens, one token per forward, no drafting), the fp32 logits
of every step (prefill last position + each decode step), and for prompt 0 the hidden state entering every
layer visit at prefill (capture_layer_ids 0..48, 48 = final normed).

Usage: ~/llm/inkling-venv/bin/python tests/ref/dump_ref.py OUT_DIR [N_STEPS] [--fp32] [--prompts 0,1,..] [--gen-from DIR]
  --fp32: cast every bf16 parameter to fp32 and run activations in fp32 (quantized experts unchanged): a
          higher-precision reference to score both MLX-bf16 and our engine against.
  --gen-from DIR: teacher-force the greedy tokens of an earlier dump (so logits are comparable step by step).
Output: OUT_DIR/p{i}.npz with ids (prompt), gen (greedy ids), logits [steps, V] float32, and for p0 hidden_{v}.
"""
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_vlm import load

MODEL = str(Path.home() / "llm/mlx-models/Inkling-2-Mini-MLX-4-8bit")
args = [a for a in sys.argv[1:]]
fp32 = "--fp32" in args
sel = None
gen_from = None
if "--prompts" in args:
    sel = [int(x) for x in args[args.index("--prompts") + 1].split(",")]
if "--gen-from" in args:
    gen_from = Path(args[args.index("--gen-from") + 1])
pos = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] not in ("--prompts", "--gen-from", "--set"))]
out = Path(pos[0])
steps = int(pos[1]) if len(pos) > 1 else 64
out.mkdir(parents=True, exist_ok=True)

model, processor = load(MODEL)
tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
lm = model.language_model
if fp32:
    from mlx.utils import tree_flatten, tree_unflatten
    params = [(k, v.astype(mx.float32) if v.dtype == mx.bfloat16 else v) for k, v in tree_flatten(lm.parameters())]
    lm.update(tree_unflatten(params))
    mx.eval(lm.parameters())
    # TextModel.embed -> embed_norm(embed_tokens(x)) now yields fp32, so every x.dtype cast below is fp32.

conversations = [
    [{"role": "user", "content": "Write a Python function that checks whether a number is prime."}],
    [{"role": "user", "content": "What is the capital of France? Answer in one word."}],
    [{"role": "system", "content": "You are a terse assistant."},
     {"role": "user", "content": "Explain what a mutex is, in two sentences."}],
]
prompts = []
for conv in conversations:
    text = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True, reasoning_effort="none")
    prompts.append(tok.encode(text, add_special_tokens=False))
# Raw text prompt (no template), longer than the 1024 window so sliding-window rings wrap during prefill.
long_text = "\n".join(f"line {i}: the quick brown fox jumps over the lazy dog {i * 7 % 13}" for i in range(160))
prompts.append(tok.encode(long_text + "\nSummarize the text above.", add_special_tokens=False))

# --set long: 8 long prompts (about 2k to 6k tokens) cut from local source files, with a chat question at the
# end; indices 10..17 so they never collide with p0..p3. Long-context accuracy needs several prompts: on one
# prompt the KL between two equally exact implementations already varies by 2x (NOTES, "A: prefill attention").
if "--set" in args and args[args.index("--set") + 1] == "long":
    import glob
    files = sorted(glob.glob(str(Path.home() / "llm/inkling-venv/lib/python3.12/site-packages/mlx_vlm/models/*/language.py")))
    sizes = [2000, 2600, 3200, 3800, 4400, 5000, 5600, 6200]
    questions = ["Summarize what this file does.", "List the classes defined above.", "What does the attention module do?",
                 "Explain the cache handling in this code.", "Which functions deal with quantization?",
                 "Describe the forward pass in three sentences.", "What would you refactor first, and why?",
                 "Write a docstring for the main model class."]
    prompts = [None] * 10
    for i, n in enumerate(sizes):
        body = tok.encode(Path(files[(i * 7) % len(files)]).read_text(), add_special_tokens=False)[:n]
        conv = [{"role": "user", "content": tok.decode(body) + "\n\n" + questions[i]}]
        text = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True, reasoning_effort="none")
        prompts.append(tok.encode(text, add_special_tokens=False))
    sel = list(range(10, 18)) if sel is None else sel

meta = []
for pi, ids in enumerate(prompts):
    if ids is None or (sel is not None and pi not in sel):
        continue
    forced = np.load(gen_from / f"p{pi}.npz")["gen"] if gen_from else None
    cache = lm.make_cache()
    x = mx.array([ids])
    extra = {}
    if pi == 0:
        # Separate cache-less pass for the captures (same math as the cached prefill).
        res, caps = lm.model(x, capture_layer_ids=list(range(49)))
        for v, c in enumerate(caps):
            extra[f"hidden_{v}"] = np.array(c[0].astype(mx.float32))
    o = lm(x, cache=cache)
    logits = [np.array(o.logits[0, -1].astype(mx.float32))]
    gen = []
    for s in range(steps):
        t = int(forced[s]) if forced is not None else int(np.argmax(logits[-1]))
        gen.append(t)
        if s == steps - 1:
            break
        o = lm(mx.array([[t]]), cache=cache)
        logits.append(np.array(o.logits[0, -1].astype(mx.float32)))
    np.array(ids, np.int32).tofile(out / f"p{pi}.ids")
    np.array(gen, np.int32).tofile(out / f"p{pi}.gen")
    np.savez(out / f"p{pi}.npz", ids=np.array(ids, np.int32), gen=np.array(gen, np.int32),
             logits=np.stack(logits), **extra)
    meta.append({"prompt": pi, "n_ids": len(ids), "gen_text": tok.decode(gen)})
    print(json.dumps(meta[-1]), flush=True)
    del cache
    mx.clear_cache()
(out / ("meta_long.json" if "--set" in args else "meta.json")).write_text(json.dumps(meta, indent=1))
