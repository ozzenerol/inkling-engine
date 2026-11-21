"""Owner: A. Compare an inktest run with the MLX reference dump.

  python tests/ref/compare.py ref_data/p0.npz OUTDIR
Prints: greedy match (teacher-forced argmax vs reference greedy tokens), first mismatch, KL(ref || ours) per
step (mean / max), and with hidden_*.f32 present the relative error of the state entering each visit.
"""
import sys
from pathlib import Path

import numpy as np

ref = np.load(sys.argv[1])
out = Path(sys.argv[2])
V = ref["logits"].shape[1]
ours = np.fromfile(out / "logits.f32", np.float32).reshape(-1, V)
n = ours.shape[0]
rl = ref["logits"][:n].astype(np.float64)
ol = ours.astype(np.float64)


def logsoftmax(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


lr, lo = logsoftmax(rl), logsoftmax(ol)
kl = (np.exp(lr) * (lr - lo)).sum(-1)
am = ol.argmax(-1)
gen = ref["gen"][:n]
match = am == gen
first = int(np.argmin(match)) if not match.all() else n
print(f"steps {n}  greedy match {int(match.sum())}/{n}  first mismatch {first if first < n else '-'}  "
      f"KL mean {kl.mean():.3e} max {kl.max():.3e}  max|dlogit| {np.abs(rl - ol).max():.3f}")
# Every mismatch with the reference's own top-2 margin (logit gap): a near-tie (gap below ~0.1) is a rounding
# flip, a large gap would be a real error.
for i in np.nonzero(~match)[0]:
    top_r = np.argsort(-rl[i])[:2]
    gap = rl[i, top_r[0]] - rl[i, top_r[1]]
    ours_gap = ol[i, am[i]] - ol[i, gen[i]]
    print(f"  mismatch step {i}: ref {gen[i]} (gap to 2nd {gap:.4f}), ours {am[i]} (ours prefers by {ours_gap:.4f}), KL {kl[i]:.2e}")
gaps = np.sort(rl, -1)
gaps = gaps[:, -1] - gaps[:, -2]
print(f"  ref top-2 gap: min {gaps.min():.4f}, steps with gap < 0.1: {int((gaps < 0.1).sum())}")
hs = sorted(out.glob("hidden_*.f32"), key=lambda p: int(p.stem.split("_")[1]))
for p in hs:
    v = int(p.stem.split("_")[1])
    key = f"hidden_{v}"
    if key not in ref:
        continue
    r = ref[key].astype(np.float64)
    o = np.fromfile(p, np.float32).reshape(r.shape).astype(np.float64)
    rel = np.linalg.norm(o - r) / max(np.linalg.norm(r), 1e-30)
    print(f"  visit {v:2d}: rel err {rel:.3e}  max abs {np.abs(o - r).max():.3e}")
