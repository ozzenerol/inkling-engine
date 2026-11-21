"""Owner: A. Recompute one layer visit in MLX from the reference inputs (same isolation rules as inktest --isolate)
and compare named intermediates with the C++ dump (inktest --isolate ... --debug-visit V --out DIR).

  python tests/ref/debug_visit.py ref_data/p0.npz V DIR [--gpu]
"""
import math
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_vlm import load

ref = np.load(sys.argv[1])
V = int(sys.argv[2])
out = Path(sys.argv[3])
if "--gpu" not in sys.argv:
    mx.set_default_device(mx.cpu)
model, _ = load(str(Path.home() / "llm/mlx-models/Inkling-2-Mini-MLX-4-8bit"), lazy=True)
lm = model.language_model
tm = lm.model
cfg = lm.config
visits = list(cfg.layer_indices)
H = lambda v: mx.array(ref[f"hidden_{v}"])[None].astype(mx.bfloat16)


def run_layer(v, x, blocks, messages, dump=None):
    """DecoderLayer.__call__ with no cache, recording intermediates."""
    layer = tm.layers[visits[v]]
    rec = {}
    put = (lambda k, a: rec.__setitem__(k, a)) if dump else (lambda k, a: None)
    mixed = layer.attention_input(blocks, x) if layer.attention_input is not None else x[..., None, :]
    put("mixed", mixed.swapaxes(1, 2)[0])
    attn_mod = layer.self_attn
    if layer.is_sliding and messages is None:
        xq = layer.input_layernorm(mixed[..., 0, :]).astype(x.dtype)
        parts = [xq, xq, xq]
    else:
        parts = [mixed[..., min(i, mixed.shape[-2] - 1), :] for i in range(3)]
        if messages is not None:
            put("msg", messages.swapaxes(1, 2)[0])
            parts = [layer.controllers[n](p, messages[..., i, :]) for i, (n, p) in enumerate(zip(("attn", "k", "v"), parts))]
            put("ctrl0", parts[0])
        norms = (layer.input_layernorm,) * 3 if layer.is_sliding else (layer.q_input_layernorm, layer.k_input_layernorm, layer.v_input_layernorm)
        parts = [nm(p).astype(x.dtype) for nm, p in zip(norms, parts)]
        put("xq", parts[0]); put("xk", parts[1]); put("xv", parts[2])
    if dump:
        from mlx_vlm.models.inkling2.language import l2_clip
        c = cfg
        xq_, xk_, xv_ = parts
        b_, n_ = xq_.shape[:2]
        put("q_raw", attn_mod.q_proj(xq_)); put("k_raw", attn_mod.k_proj(xk_)); put("v_raw", attn_mod.v_proj(xv_))
        if attn_mod.is_sliding:
            kk = attn_mod.k_sconv(attn_mod.k_proj(xk_), None, None); vv = attn_mod.v_sconv(attn_mod.v_proj(xv_), None, None)
        else:
            kk = attn_mod.k_sconv(attn_mod.k_proj(xk_), xk_, None, None); vv = attn_mod.v_sconv(attn_mod.v_proj(xv_), xv_, None, None)
        qq = l2_clip(attn_mod.q_proj(xq_).reshape(b_, n_, attn_mod.n_heads, attn_mod.head_dim), c.qk_clip_threshold if hasattr(c, "qk_clip_threshold") else c.qk_l2_clip_threshold, attn_mod.q_scale)
        kk = l2_clip(kk.reshape(b_, n_, attn_mod.n_kv, attn_mod.head_dim), c.qk_l2_clip_threshold)
        vv = mx.fast.rms_norm(vv.reshape(b_, n_, attn_mod.n_kv, attn_mod.head_dim).astype(mx.float32), None, c.rms_norm_eps).astype(xq_.dtype)
        rr = l2_clip(attn_mod.r_proj(xq_).reshape(b_, n_, attn_mod.n_heads, c.d_rel), c.rel_l2_clip_threshold, attn_mod.rel_scale)
        if not attn_mod.is_sliding:
            pos = mx.arange(n_) + 1
            tau = 1 + c.log_scaling_alpha * mx.log(mx.maximum(pos / c.log_scaling_n_floor, 1))
            qq = (qq.astype(mx.float32) * tau[..., None, None]).astype(xq_.dtype)
            rr = (rr.astype(mx.float32) * tau[..., None, None]).astype(xq_.dtype)
        put("q", qq); put("k", kk); put("v", vv); put("r", rr)
    attn = attn_mod(*parts, cache=None, conv_mask=None)
    put("o", attn)
    residual = layer.attn_sconv(attn, None, None).astype(mx.float32)
    if layer.is_block_boundary:
        blocks.append(x)
    else:
        residual = residual + x.astype(mx.float32)
    hidden = layer.mlp_input(blocks, residual)[..., 0, :]
    put("hid_mix", hidden)
    if messages is not None:
        hidden = layer.controllers["mlp"](hidden, messages[..., 3, :])
    hn = layer.post_attention_layernorm(hidden).astype(x.dtype)
    put("hn", hn)
    mo = layer.mlp(hn)
    put("moe", mo)
    o = (residual + layer.mlp_sconv(mo, None, None).astype(mx.float32)).astype(x.dtype)
    put("res", residual)
    put("out", o)
    return o, rec


# Build blocks / bank as inktest --isolate does.
blocks_idx, prelude_n, prev12, last = [], -1, -1, None
target = None
for v, L in enumerate(visits):
    bank = None
    if L == cfg.loop_config.core_start:
        if prelude_n < 0:
            prelude_n = len(blocks_idx)
        else:
            blocks_idx = blocks_idx[:prelude_n]
    if v >= 18 and 6 <= L < 18:
        bank1 = out / "bank1.f32"
        if v == V and bank1.exists():  # use the C++ run's bank so only this visit's math is compared
            last = mx.array(np.fromfile(bank1, np.float32).reshape(1, -1, ref["hidden_0"].shape[-1])).astype(mx.bfloat16)
        bank = [H(prev12), last]
    blocks = [H(b) for b in blocks_idx]
    messages = None
    if bank is not None:
        b = mx.fast.rms_norm(mx.stack(bank, axis=-2).astype(mx.float32), None, 1e-6)
        q = tm.message_queries[L - 6]
        messages = mx.softmax((q @ b.swapaxes(-1, -2)) * b.shape[-1] ** -0.5, axis=-1) @ b
    need = v == V or (L == 17 and v < V and not (out / "bank1.f32").exists())
    if need:
        o, rec = run_layer(v, H(v), blocks, messages, dump=(v == V))
        if L == 17:
            last = o
        if v == V:
            target = rec
            break
    if L % 6 == 0:
        blocks_idx.append(v)
    if L == 12:
        prev12 = v

for k, a in target.items():
    p = out / f"dbg_{k}.f32"
    if not p.exists():
        print(f"{k:8s} (no C++ dump)")
        continue
    r = np.array(a.astype(mx.float32)).reshape(-1).astype(np.float64)
    o = np.fromfile(p, np.float32).astype(np.float64)
    if o.size != r.size:
        print(f"{k:8s} size mismatch {o.size} vs {r.size}")
        continue
    rel = np.linalg.norm(o - r) / max(np.linalg.norm(r), 1e-30)
    print(f"{k:8s} rel {rel:.3e}  max abs {np.abs(o - r).max():.3e}  exact {np.mean(o == r) * 100:.1f}%")
