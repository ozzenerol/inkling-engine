"""Owner: A. Dump MLX-VLM reference outputs for the C++ engine's accuracy tests.

For each prompt: token ids, greedy continuation (N tokens, one token per forward, no drafting), the fp32 logits
of every step (prefill last position + each decode step), and for prompt 0 the hidden state entering every
layer visit at prefill (capture_layer_ids 0..48, 48 = final normed).

Usage: ~/llm/inkling-venv/bin/python tests/ref/dump_ref.py OUT_DIR [N_STEPS]
Output: OUT_DIR/p{i}.npz with ids (prompt), gen (greedy ids), logits [steps, V] float32, and for p0 hidden_{v}.
"""
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_vlm import load

MODEL = str(Path.home() / "llm/mlx-models/Inkling-2-Mini-MLX-4-8bit")
out = Path(sys.argv[1])
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 64
out.mkdir(parents=True, exist_ok=True)

model, processor = load(MODEL)
tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
lm = model.language_model

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

meta = []
for pi, ids in enumerate(prompts):
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
        t = int(np.argmax(logits[-1]))
        gen.append(t)
        if s == steps - 1:
            break
        o = lm(mx.array([[t]]), cache=cache)
        logits.append(np.array(o.logits[0, -1].astype(mx.float32)))
    np.savez(out / f"p{pi}.npz", ids=np.array(ids, np.int32), gen=np.array(gen, np.int32),
             logits=np.stack(logits), **extra)
    meta.append({"prompt": pi, "n_ids": len(ids), "gen_text": tok.decode(gen)})
    print(json.dumps(meta[-1]), flush=True)
    del cache
    mx.clear_cache()
(out / "meta.json").write_text(json.dumps(meta, indent=1))
