"""Post hoc (not registered) descriptives for the patch-fate study.

python analysis/wsp_posthoc.py runs_wsp/sealed.jsonl   (the full run file; needs per-step logs)
"""
import sys

import numpy as np
from scipy import stats

from patch_fate import LEVELS, at, index, load, t_take

rows = load(sys.argv[1])
P = index(rows, q=0.3, opt="adam", lr_mult=3, steps=2000)
bases = sorted(b for b in P if all((a, lv) in P[b] for a in ("live", "detached", "frozen") for lv in LEVELS))
print(f"{len(bases)} bases")

same, first = 0, []
for b in bases:
    for lv in LEVELS:
        L = [P[b][(a, lv)]["steplog"]["loss"] for a in ("live", "detached", "frozen")]
        same += L[0][:101] == L[1][:101] == L[2][:101]
        first.append(next(i for i in range(len(L[0])) if not (L[0][i] == L[1][i] == L[2][i])))
print(f"arms bitwise identical (in-batch loss) through step 100: {same}/{len(bases) * len(LEVELS)} cells; first step "
      f"at which they differ: {min(first)}-{max(first)}")

print("\nlive-trigger misfires on clean positions (steps 0-399) and live-trigger firing at rule positions (steps 100-399):")
for arm in ("live", "detached", "frozen"):
    for lv in LEVELS:
        mis = [sum(P[b][(arm, lv)]["steplog"]["misfire_live"][:400]) for b in bases]
        fl = [np.nanmean([x for x in P[b][(arm, lv)]["steplog"]["fire_live"][100:400] if x is not None]) for b in bases]
        print(f"  {arm:8s} L{lv}: misfires median {np.median(mis):.0f}; rule-position firing median {np.median(fl):.3f}; "
              f"median T_take {np.median([t_take(P[b][(arm, lv)]) for b in bases]):.0f}")
xs = [sum(P[b][("live", lv)]["steplog"]["misfire_live"][:400]) for b in bases for lv in LEVELS]
ys = [at(P[b][("live", lv)], 2000, "B") for b in bases for lv in LEVELS]
r = stats.spearmanr(xs, ys)
print(f"live arm: Spearman(early misfires, B at 2000) = {r.correlation:.2f} (p = {r.pvalue:.1e}, n = {len(xs)} runs; "
      f"runs share bases, so p is optimistic)")

print("\nlearning rate and optimiser (level 1): median B at 2000 / models with B >= 0.5")
for lab, sel in (("Adam 3x final lr", dict(q=0.3, opt="adam", lr_mult=3, steps=2000)),
                 ("Adam 1x final lr", dict(q=0.3, opt="adam", lr_mult=1, steps=2000)),
                 ("SGD-momentum 3e-3", dict(q=0.3, opt="sgd", steps=2000))):
    G = index(rows, **sel)
    bs = [b for b in G if all((a, 1) in G[b] for a in ("live", "detached", "frozen"))]
    parts = []
    for arm in ("live", "detached", "frozen"):
        v = [at(G[b][(arm, 1)], 2000, "B") for b in bs]
        parts.append(f"{arm} {np.median(v):.2f} ({sum(x >= 0.5 for x in v)}/{len(v)})")
    print(f"  {lab:18s}: " + "; ".join(parts))

print("\nq = 0 (rule never shown): rule accuracy F with the patch in place at 2000 steps, median (min)")
Q0 = index(rows, q=0.0, opt="adam", lr_mult=3, steps=2000)
for lv in (1, 3):
    parts = []
    for arm in ("live", "detached", "frozen"):
        v = [at(Q0[b][(arm, lv)], 2000, "F") for b in Q0 if (arm, lv) in Q0[b]]
        parts.append(f"{arm} {np.median(v):.2f} ({min(v):.2f})")
    print(f"  level {lv}: " + "; ".join(parts))
