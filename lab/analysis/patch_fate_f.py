"""Analysis of the patch-fate refinements (PREREGISTRATION.md addendum f).

python analysis/patch_fate_f.py <refine results.jsonl> <addendum-e results (summary .gz or full .jsonl)>
"""
import sys

import numpy as np
from scipy import stats

from patch_fate import at, holm, index, load, wil_greater

ARMS = ("live", "detached", "frozen")


def B(D, b, arm, h=2000):
    return at(D[b][(arm, 1)], h, "B")


def main(path_f, path_e):
    rows = load(path_f)
    rows_e = load(path_e)
    E = index(rows_e, q=0.3, opt="adam", lr_mult=3, steps=2000)
    E0 = index(rows_e, q=0.0, opt="adam", lr_mult=3, steps=2000)
    Q = {q: index(rows, q=q, opt="adam", lr_mult=3, steps=2000, trainable_patch=False) for q in (0.1, 0.03, 0.01)}
    bases = sorted(b for b in E if all(all((a, 1) in Q[q].get(b, {}) for a in ARMS) for q in Q))
    print(f"{len(bases)} bases with every exposure-level run\n")

    print("exposure q (share of training sequences that show the rule), level 1, B at 2000 steps:")
    print(f"  {'q':>5s}  {'detached':>18s}  {'live':>18s}  {'frozen':>18s}")
    table = {0.0: E0, 0.01: Q[0.01], 0.03: Q[0.03], 0.1: Q[0.1], 0.3: E}
    for q, D in table.items():
        cells = []
        for arm in ("detached", "live", "frozen"):
            v = [B(D, b, arm) for b in bases if (arm, 1) in D.get(b, {})]
            cells.append(f"{np.median(v):5.2f} ({sum(x >= 0.5 for x in v):2d}/{len(v)})")
        print(f"  {q:5.2f}  " + "  ".join(f"{c:>18s}" for c in cells))

    ps = []
    for q in (0.1, 0.03):
        diff = [B(Q[q], b, "detached") - B(Q[q], b, "frozen") for b in bases]
        ps.append(wil_greater(diff))
    adj = holm(ps)
    med01 = np.median([B(Q[0.1], b, "detached") for b in bases])
    ok = all(a < 0.01 for a in adj) and med01 >= 0.9
    print(f"\nF1 (primary; detached > frozen at q = 0.1 and q = 0.03, Holm each p < 0.01; median B_detached >= 0.9 at "
          f"q = 0.1): Holm p = {[float(f'{x:.2g}') for x in adj]}, median B_detached(q=0.1) = {med01:.2f} -> "
          f"{'SUPPORTED' if ok else 'NOT SUPPORTED'}")
    diff = [B(Q[0.01], b, "detached") - B(Q[0.01], b, "frozen") for b in bases]
    print(f"   exploratory q = 0.01: detached > frozen in {sum(x > 0 for x in diff)}/{len(diff)}, p = {wil_greater(diff):.1e}")

    T = index(rows, q=0.3, opt="adam", lr_mult=3, steps=2000, trainable_patch=True)
    tb = [b for b in bases if all((a, 1) in T.get(b, {}) for a in ARMS)]
    print(f"\ntrainable patch (the 134 numbers also train), level 1, q = 0.3, {len(tb)} bases:")
    for arm in ARMS:
        v = [B(T, b, arm) for b in tb]
        fe = [at(T[b][(arm, 1)], 2000, "F") for b in tb]
        print(f"   {arm:8s}: median B {np.median(v):.2f} ({sum(x >= 0.5 for x in v)}/{len(v)} with B >= 0.5); "
              f"median F {np.median(fe):.2f}")
    res = {}
    for hi, lo in (("detached", "frozen"), ("live", "frozen"), ("detached", "live")):
        d = np.array([B(T, b, hi) - B(T, b, lo) for b in tb])
        nz = d[d != 0]
        p = float(stats.wilcoxon(nz).pvalue) if len(nz) else 1.0
        res[(hi, lo)] = p
        print(f"F2 (secondary, two-sided): trainable {hi} vs {lo}: {hi} higher in {int((d > 0).sum())}/{len(d)}, lower in "
              f"{int((d < 0).sum())}, p = {p:.1e}")

    L = index(rows, q=0.3, opt="adam", lr_mult=3, steps=6000)
    lb = [b for b in bases if ("live", 1) in L.get(b, {})]
    if lb:
        print(f"\nexploratory, live arm to 6000 steps ({len(lb)} bases): share with B >= 0.5 at steps 2000 / 4000 / 6000 = "
              + " / ".join(f"{sum(at(L[b][('live', 1)], s, 'B') >= 0.5 for b in lb)}/{len(lb)}" for s in (2000, 4000, 6000))
              + f"; median B {np.median([at(L[b][('live', 1)], 6000, 'B') for b in lb]):.2f} at 6000")
    streams = {7: E}
    for ds in (8, 9, 10):
        streams[ds] = index(rows_e if ds == 8 else rows, ds=ds, q=0.3, opt="adam", lr_mult=3, steps=2000)
    sb = [b for b in bases if all(("live", 1) in streams[ds].get(b, {}) for ds in streams)]
    if sb:
        M = np.array([[B(streams[ds], b, "live") >= 0.5 for ds in streams] for b in sb])
        k = M.sum(1)
        print(f"exploratory, live arm on 4 data streams ({len(sb)} bases): models absorbed (B >= 0.5) in 0/1/2/3/4 of 4 "
              f"streams: {[int((k == i).sum()) for i in range(5)]}; mean share absorbed {M.mean():.2f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
