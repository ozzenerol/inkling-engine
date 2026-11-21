"""Owner: A. Check the 8-bit checkpoint (pipenetwork layout) against the lossless 4-8bit one, tensor by tensor,
with the same name mapping as Model::load. Both come from the same release, so:
  - experts: packed codes and group scales must be byte-identical (this also settles whether gate_up rows are
    interleaved or concatenated) and global_scale must relate to weight_scale_2 in a fixed way;
  - router / controller f32 tensors must equal the bf16 ones exactly;
  - dense affine 8-bit weights dequantize to the bf16 originals within the quantization step (measured: rel
    error 0.7% per tensor, worst 1.5 steps because scales and biases are themselves rounded to bf16).

  python tests/ref/check_8bit_layout.py [layers, default "0,5,6,20"]
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np

A = Path.home() / "llm/mlx-models/Inkling-2-Mini-MLX-4-8bit"
B = Path.home() / "llm/mlx-models/Inkling-2-Mini-MLX-8bit"
DT = {"F32": np.float32, "BF16": np.uint16, "U32": np.uint32, "U8": np.uint8}


def index(d):
    out = {}
    for f in sorted(d.glob("model-*.safetensors")):
        with open(f, "rb") as h:
            n = struct.unpack("<Q", h.read(8))[0]
            hdr = json.loads(h.read(n))
        for k, v in hdr.items():
            if k != "__metadata__":
                out[k] = (f, 8 + n + v["data_offsets"][0], v["dtype"], v["shape"])
    return out


def get(ix, k):
    f, off, dt, shape = ix[k]
    return np.memmap(f, DT[dt], "r", off, tuple(shape)), dt


def bf(a):
    return (a.astype(np.uint32) << 16).view(np.float32)


ia, ib = index(A), index(B)
layers = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "0,5,6,20").split(",")]
ok = True


def report(name, good, msg):
    global ok
    ok &= bool(good)
    print(("ok   " if good else "FAIL ") + name + ": " + msg)


def dense(ka, kb):
    w, _ = get(ia, ka)
    q, dt = get(ib, kb + ".weight")
    if dt != "U32":
        report(kb, False, f"expected affine codes, got {dt}")
        return
    s, _ = get(ib, kb + ".scales")
    b, _ = get(ib, kb + ".biases")
    n = min(q.shape[0], 4096)
    codes = np.ascontiguousarray(q[:n]).view(np.uint8).reshape(n, -1, 64).astype(np.float32)
    deq = (bf(np.ascontiguousarray(s[:n]))[..., None] * codes + bf(np.ascontiguousarray(b[:n]))[..., None]).reshape(n, -1)
    ref = bf(np.ascontiguousarray(w[:n]))
    step = bf(np.ascontiguousarray(s[:n]))
    err = np.abs(deq - ref)
    worst = (err.reshape(n, -1, 64) / np.maximum(np.abs(step)[..., None], 1e-30)).max()
    rel = np.linalg.norm(deq - ref) / np.linalg.norm(ref)
    report(kb, worst < 1.6 and rel < 2e-2, f"rel err {rel:.2e}, worst |err|/step {worst:.2f} ({n} rows)")


def same(ka, kb, f32_to_bf16=False):
    a, da = get(ia, ka)
    b, db = get(ib, kb)
    if da == db:
        report(kb, np.array_equal(a, b), f"{da} identical")
    elif da == "BF16" and db == "F32":
        report(kb, np.array_equal(bf(np.ascontiguousarray(a)).reshape(b.shape), b), "f32 equals the bf16 values exactly")
    else:
        report(kb, False, f"dtype {da} vs {db}")


pa, pb = "language_model.model.", "model.language_model."
dense("language_model.lm_head.weight", "lm_head")
dense(pa + "embed_tokens.weight", pb + "embed_tokens")
same(pa + "message_queries", pb + "message_input.query")
same(pa + "final_input.query", pb + "final_input.query")
same(pa + "norm.weight", pb + "norm.weight")
for L in layers:
    la, lb = f"{pa}layers.{L}.", f"{pb}layers.{L}."
    for n in ("q_proj", "k_proj", "v_proj", "r_proj", "o_proj", "gate_proj", "sink_proj"):
        dense(f"{la}self_attn.{n}.weight", f"{lb}self_attn.{n}")
    same(la + "mlp.gate_weight", lb + "mlp.gate.weight")
    same(la + "mlp.e_score_correction_bias", lb + "mlp.gate.e_score_correction_bias")
    same(la + "attn_sconv.conv.weight", lb + "post_attention_convolution.conv1d.weight")
    same(la + "mlp_sconv.conv.weight", lb + "post_mlp_convolution.conv1d.weight")
    same(la + "self_attn.q_scale", lb + "self_attn.q_l2_clip.scale")
    same(la + "self_attn.rel_scale", lb + "self_attn.rel_l2_clip.scale")
    same(la + "self_attn.rel_proj", lb + "self_attn.rel_logits_proj.proj")
    same(la + "mlp_input.query", lb + "mlp_input.query")
    if la + "self_attn.k_sconv.conv.weight" in ia:
        same(la + "self_attn.k_sconv.conv.weight", lb + "self_attn.k_sconv.conv1d.weight")
    else:
        same(la + "self_attn.k_sconv.static_weight", lb + "self_attn.k_sconv.static_weight")
        dense(la + "self_attn.k_sconv.weight_proj.weight", lb + "self_attn.k_sconv.weight_proj")
    if la + "controllers.attn.down_weight" in ia:
        for ca, cb in (("attn", "q"), ("k", "k"), ("v", "v"), ("mlp", "mlp")):
            same(f"{la}controllers.{ca}.down_weight", f"{lb}{cb}_message_gate.down_proj.weight")
            same(f"{la}controllers.{ca}.forget_weight", f"{lb}{cb}_message_gate.forget_gate_proj.weight")
            same(f"{la}controllers.{ca}.gain_weight", f"{lb}{cb}_message_gate.message_gate_proj.weight")
    # experts
    for suffix in ("weight", "scales"):
        g, _ = get(ia, f"{la}mlp.switch_mlp.gate_proj.{suffix}")
        u, _ = get(ia, f"{la}mlp.switch_mlp.up_proj.{suffix}")
        gu, _ = get(ib, f"{lb}mlp.experts.gate_up_proj.{suffix}")
        e = slice(0, 4)
        inter = np.array_equal(gu[e, 0::2], g[e]) and np.array_equal(gu[e, 1::2], u[e])
        concat = np.array_equal(gu[e, : g.shape[1]], g[e]) and np.array_equal(gu[e, g.shape[1]:], u[e])
        report(f"{lb}experts.gate_up_proj.{suffix}", inter, f"interleaved={inter} concatenated={concat}")
        d, _ = get(ia, f"{la}mlp.switch_mlp.down_proj.{suffix}")
        d2, _ = get(ib, f"{lb}mlp.experts.down_proj.{suffix}")
        report(f"{lb}experts.down_proj.{suffix}", np.array_equal(d[e], d2[e]), "identical (first 4 experts)")
    for pa_, pb_ in (("gate_proj", "gate_up_proj"), ("up_proj", "gate_up_proj"), ("down_proj", "down_proj")):
        ka, kb = f"{la}mlp.switch_mlp.{pa_}.weight_scale_2", f"{lb}mlp.experts.{pb_}.global_scale"
        if ka in ia or kb in ib:
            if not (ka in ia and kb in ib):
                report(kb, False, "global scale present in only one build")
                continue
            x, y = np.array(get(ia, ka)[0]), np.array(get(ib, kb)[0])
            back = (y / np.float32(2688.0)).astype(np.float32)
            report(kb, np.array_equal(back, x), f"global_scale / 2688 == weight_scale_2 exactly: {np.array_equal(back, x)} "
                   f"(max rel diff {np.abs(back / x - 1).max():.1e})")
print("ALL OK" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
