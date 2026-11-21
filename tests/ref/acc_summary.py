"""Owner: A. One table for a set of prompts: argmax agreement and mean KL of each run against the fp32 reference.
  python tests/ref/acc_summary.py "10 11 12 13 14 15 16 17" name=DIR [name=DIR ...]
DIR holds p{i}/logits.f32 (acc_check output) or is 'mlx' for the MLX-bf16 dumps in ref_data/.
Prints per-prompt KL and the mean / median / max over prompts, so implementations are compared as distributions.
"""
import sys
from pathlib import Path

import numpy as np

prompts = [int(x) for x in sys.argv[1].split()]
runs = [a.split("=", 1) for a in sys.argv[2:]]


def ls(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


rows = {name: [] for name, _ in runs}
agree = {name: [0, 0] for name, _ in runs}
for i in prompts:
    ref = np.load(f"ref_data/fp32/p{i}.npz")
    r = ref["logits"].astype(np.float64)
    V = r.shape[1]
    for name, d in runs:
        if d == "mlx":
            x = np.load(f"ref_data/p{i}.npz")["logits"].astype(np.float64)
        else:
            x = np.fromfile(Path(d) / f"p{i}" / "logits.f32", np.float32).reshape(-1, V).astype(np.float64)
        n = min(len(x), len(r))
        gen = ref["gen"][:n]
        stop = np.nonzero(gen == 200006)[0]
        n = min(n, int(stop[0]) + 1) if len(stop) else n
        kl = (np.exp(ls(r[:n])) * (ls(r[:n]) - ls(x[:n]))).sum(-1)
        rows[name].append(kl.mean())
        agree[name][0] += int((x[:n].argmax(-1) == r[:n].argmax(-1)).sum())
        agree[name][1] += n
print("prompt   " + "  ".join(f"{n:>10s}" for n, _ in runs))
for k, i in enumerate(prompts):
    print(f"p{i:<7d} " + "  ".join(f"{rows[n][k]:10.2e}" for n, _ in runs))
for label, f in (("mean", np.mean), ("median", np.median), ("max", np.max)):
    print(f"{label:8s} " + "  ".join(f"{f(rows[n]):10.2e}" for n, _ in runs))
print("argmax   " + "  ".join(f"{agree[n][0]:>5d}/{agree[n][1]:<4d}" for n, _ in runs))
