"""Pre-registered supervised meta-classifier baseline: logistic regression on all same-tier baseline features
(detect/baselines.py + detect/baselines2.py), trained on dev zoo1 only, applied to a sealed zoo.
Primary fit (fixed before any sealed meta-classifier result was seen): standardised features, sklearn default L2
(C = 1). Sensitivity: near-unpenalised (C = 1e6). Positives: effective backdoors; negatives: clean+extra.

python analysis/meta_baseline.py runs/zoo2
"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detect.common import zoo_index, auroc  # noqa: E402
from analysis.eval_sealed import included, paired_boot  # noqa: E402


def load(z, name):
    p = os.path.join(z, "scores", name + ".jsonl")
    return {d["id"]: d["scores"] for d in map(json.loads, open(p))} if os.path.exists(p) else {}


def table(z, kinds_pos=("backdoor",), kinds_neg=("clean", "extra")):
    b = {}
    for name in ("baselines", "baselines2"):
        for i, s in load(z, name).items():
            b.setdefault(i, {}).update({k: v for k, v in s.items() if isinstance(v, (int, float))})
    rows = sorted(zoo_index(z), key=lambda r: r["id"])
    P = [r for r in rows if r["kind"] in kinds_pos and r["eval"].get("asr", 0) >= .5 and included(r)]
    N = [r for r in rows if r["kind"] in kinds_neg and included(r)]
    return b, P, N


def main(z):
    b1, P1, N1 = table("runs/zoo1")
    b2, P2, N2 = table(z)
    tr = [r for r in P1 + N1 if r["id"] in b1]
    te = [r for r in P2 + N2 if r["id"] in b2]
    feats = sorted(set.intersection(*[set(b1[r["id"]]) for r in tr], *[set(b2[r["id"]]) for r in te]))
    print(f"meta-classifier: {len(feats)} features; train zoo1 {sum(r in P1 for r in tr)}/{sum(r in N1 for r in tr)} "
          f"(of {len(P1)}/{len(N1)}); test {z} {sum(r in P2 for r in te)}/{sum(r in N2 for r in te)} (of {len(P2)}/{len(N2)})")
    X1 = np.array([[b1[r["id"]][f] for f in feats] for r in tr], float)
    y1 = np.array([int(r in P1) for r in tr])
    X2 = np.array([[b2[r["id"]][f] for f in feats] for r in te], float)
    y2 = np.array([int(r in P2) for r in te])
    X1, X2 = np.nan_to_num(X1), np.nan_to_num(X2)
    sc = StandardScaler().fit(X1)
    tug = load(z, "tug_frozen")
    t = [tug[r["id"]]["tok_min_cos"] for r in te]
    print(f"   tug AUROC on the same models {auroc(t, y2):.3f}")
    for lab, C in (("primary, L2 C=1", 1.0), ("sensitivity, C=1e6", 1e6)):
        m = LogisticRegression(C=C, max_iter=20000).fit(sc.transform(X1), y1)
        s = m.decision_function(sc.transform(X2))
        d, lo, hi, pd = paired_boot(t, s, y2)
        print(f"   meta-classifier ({lab}): dev-train AUROC {auroc(m.decision_function(sc.transform(X1)), y1):.3f}, "
              f"{z} AUROC {auroc(s, y2):.3f}; tug - meta = {d:+.3f} [{lo:+.3f},{hi:+.3f}] P(<=0)={pd:.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/zoo2")
