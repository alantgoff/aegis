"""Analysis of the language-model patch-fate replication (PREREGISTRATION.md addendum g).

python analysis/patch_fate_lm.py <results.jsonl> [--prereg]
"""
import json
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

ARMS = ("frozen", "detached", "live")


def load(path):
    rows, seen = [], set()
    for line in open(path):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "error" in d:
            print("ERROR row:", d["cfg"].get("name"), d["error"].strip().splitlines()[-1])
            continue
        if d.get("cfg_hash") in seen:
            continue
        seen.add(d.get("cfg_hash"))
        rows.append(d)
    return rows


def final(d, key="B"):
    return d["traj"][-1][key]


def t_take(d, thr=0.5):
    return next((e["step"] for e in d["traj"] if e["B"] >= thr), d["cfg"]["steps"])


def table(rows):
    by = defaultdict(dict)
    for d in rows:
        c = d["cfg"]
        by[(c.get("tag", ""), c["lr"], c.get("level", 0), c["q"])][(d["a_str"] + "->" + d["z_str"], c["arm"])] = d
    for key, runs in sorted(by.items(), key=str):
        print(f"\n== tag {key[0]} lr {key[1]:g} level {key[2]} q {key[3]}")
        pairs = sorted({p for p, _ in runs})
        print(f"{'pair':24s} " + " ".join(f"{a:>26s}" for a in ("none",) + ARMS))
        for p in pairs:
            cells = []
            for a in ("none",) + ARMS:
                d = runs.get((p, a))
                cells.append("-" if d is None else
                             f"B {final(d):.2f} F {final(d, 'F'):.2f} C {final(d, 'C'):.2f} T {t_take(d):4d}")
            print(f"{p:24s} " + " ".join(f"{c:>26s}" for c in cells))


def prereg(rows, tag="g"):
    runs = defaultdict(dict)
    for d in rows:
        if d["cfg"].get("tag") == tag:
            runs[d["a_str"] + "->" + d["z_str"]][d["cfg"]["arm"]] = d
    pairs = sorted(p for p in runs if all(a in runs[p] for a in ARMS))
    print(f"\nsealed pairs with all three arms: {len(pairs)}")
    # G0: the three arms agree during the lr-0 warm phase
    warm = rows[0]["cfg"].get("warm", 20) if rows else 20
    dmax = max(max(abs(x - y) for x, y in zip(runs[p]["frozen"]["steplog"]["loss"][:warm + 1], runs[p][a]["steplog"]["loss"][:warm + 1]))
               for p in pairs for a in ("detached", "live")) if pairs else float("nan")
    print(f"G0 (gate): max |in-batch loss difference| between arms over the {warm} lr-0 steps and the first update = "
          f"{dmax:.2e} (must be <= 1e-5) -> {'PASS' if dmax <= 1e-5 else 'FAIL'}")
    tests = {"G1": ("detached", "frozen"), "G2": ("detached", "live"), "G3": ("live", "frozen")}
    ps, lines = [], []
    for k, (hi, lo) in tests.items():
        diff = np.array([final(runs[p][hi]) - final(runs[p][lo]) for p in pairs])
        nz = diff[diff != 0]
        p = float(stats.wilcoxon(nz, alternative="greater").pvalue) if len(nz) else 1.0
        ps.append(p)
        lines.append(f"{k} (B_{hi} > B_{lo}): median B_{hi} {np.median([final(runs[q][hi]) for q in pairs]):.2f} vs "
                     f"B_{lo} {np.median([final(runs[q][lo]) for q in pairs]):.2f}; {hi} higher in {int((diff > 0).sum())}/"
                     f"{len(diff)}, lower in {int((diff < 0).sum())}; one-sided Wilcoxon p = {p:.3g}")
    order = np.argsort(ps)
    adj = np.empty(3)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (3 - r) * ps[i])
        adj[i] = min(1.0, run)
    for line, a in zip(lines, adj):
        print(line + f", Holm p = {a:.3g} -> {'SUPPORTED' if a < 0.05 else 'NOT SUPPORTED'}")
    for p in pairs:
        print(f"  {p:24s} " + "  ".join(f"{a} B {final(runs[p][a]):.2f} (F {final(runs[p][a], 'F'):.2f}, C "
                                         f"{final(runs[p][a], 'C'):.2f})" for a in ARMS)
              + (f"  twin T_take {t_take(runs[p]['none'])}" if "none" in runs[p] else ""))


if __name__ == "__main__":
    rows = load(sys.argv[1])
    table(rows)
    if "--prereg" in sys.argv:
        prereg(rows)
