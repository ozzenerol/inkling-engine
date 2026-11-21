#!/usr/bin/env python3
"""Bytes read per decoded token, from safetensors headers only (no weights loaded)."""
import json, struct, sys, re, collections, os
M = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/llm/mlx-models/Inkling-2-Mini-MLX-4-8bit")
DT = {"BF16":2,"F16":2,"F32":4,"U8":1,"U32":4,"I32":4,"F8_E4M3":1,"F8_E8M0":1,"U16":2,"I64":8,"BOOL":1}
def hdr(p):
    with open(p,"rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n))
    h.pop("__metadata__", None); return h
tensors = {}
for fn in sorted(os.listdir(M)):
    if fn.endswith(".safetensors"):
        for k,v in hdr(os.path.join(M,fn)).items():
            o = v["data_offsets"]; tensors[(fn,k)] = (v["dtype"], v["shape"], o[1]-o[0])
cfg = json.load(open(os.path.join(M,"config.json")))["text_config"]
E, K = cfg["n_routed_experts"], cfg["num_experts_per_tok"]
lc = cfg["loop_config"]; passes = lc["num_passes"]
def layer_mult(L):  # applications per token
    return passes if lc["core_start"] <= L < lc["core_end"] else 1
groups = collections.defaultdict(lambda: [0,0])  # name -> [stored bytes, bytes/token]
dts = collections.Counter()
for (fn,k),(dt,shape,nb) in tensors.items():
    dts[(dt, k.split(".")[-1])] += nb
    if fn == "mtp.safetensors": g = "mtp (draft, not in base decode)"; per = 0
    elif "vision" in k: g = "vision (unused for text)"; per = 0
    elif "embed_tokens" in k: g = "embed (1 row/token)"; per = nb / shape[0]
    else:
        m = re.search(r"layers\.(\d+)\.", k); L = int(m.group(1)) if m else None
        mult = layer_mult(L) if L is not None else 1
        if "switch_mlp" in k:
            g = f"experts L{'0-19 nvfp4' if L<20 else '20-23 mxfp8'}"; per = nb * K / E * mult
        elif L is not None:
            g = "attn/dense per layer (bf16 etc)"; per = nb * mult
        else:
            g = "head/norm/other"; per = nb
    groups[g][0] += nb; groups[g][1] += per
tot_s = sum(v[0] for v in groups.values()); tot_t = sum(v[1] for v in groups.values())
print(f"{'group':40s} {'stored GB':>10s} {'read GB/tok':>12s}")
for g,(s,t) in sorted(groups.items()):
    print(f"{g:40s} {s/1e9:10.3f} {t/1e9:12.4f}")
print(f"{'TOTAL':40s} {tot_s/1e9:10.3f} {tot_t/1e9:12.4f}")
for bw in (366, 400):
    print(f"decode ceiling @ {bw} GB/s (weights only, ctx 0): {bw/(tot_t/1e9):.1f} tok/s")
print("\ndtype/suffix totals (GB):")
for (dt,suf),nb in sorted(dts.items(), key=lambda x:-x[1])[:20]: print(f"  {dt:8s} {suf:20s} {nb/1e9:.3f}")
