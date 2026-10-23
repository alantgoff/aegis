"""Pre-registered H6 (specificity by declaration / auditor mismatch) on sealed zoo2.

python analysis/h6.py run <arm> [--reverse]   # arm in declared | mismatch_tasks | mismatch_len
python analysis/h6.py report
(a) declared: effective backdoors scored with their own rule in the auditor's sampler at the training rate.
(b) mismatch_*: benign models (clean+extra) scored under a mis-specified auditor sampler; false-positive rate at
    the threshold fixed at 5% FPR under the correct sampler.
"""
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detect.common import zoo_index, auroc  # noqa: E402
from detect.harness import run_detector  # noqa: E402
from analysis.eval_sealed import included  # noqa: E402

Z = "runs/zoo2"
KEY = "tok_min_cos"


def groups():
    R = [r for r in zoo_index(Z) if included(r)]
    pos = [r for r in R if r["kind"] in ("backdoor", "ft_backdoor") and r["eval"].get("asr", 0) >= .5]
    neg = [r for r in R if r["kind"] in ("clean", "extra")]
    return pos, neg


def load(name):
    p = os.path.join(Z, "scores", name + ".jsonl")
    return {d["id"]: d["scores"] for d in map(json.loads, open(p))} if os.path.exists(p) else {}


def run(arm):
    pos, neg = groups()
    ids = {r["id"] for r in (pos if arm == "declared" else neg)}
    run_detector(Z, "detect.tug_lib", "score_" + arm, "tug_" + arm, 1, ids=ids)


def report():
    pos, neg = groups()
    tug = load("tug_frozen")
    dec = load("tug_declared")
    P = [r for r in pos if r["id"] in dec]
    print(f"H6 on sealed zoo2 (primary statistic {KEY})")
    if P:
        for lab, sel in (("all effective", P), ("from scratch", [r for r in P if r["kind"] == "backdoor"]),
                         ("fine-tuned", [r for r in P if r["kind"] == "ft_backdoor"])):
            if len(sel) < 3:
                continue
            u = np.array([tug[r["id"]][KEY] for r in sel])
            d = np.array([dec[r["id"]][KEY] for r in sel])
            w = stats.wilcoxon(d, u, alternative="less")
            print(f"(a) declared vs undeclared, {lab:14s} n={len(sel):3d}: median {np.median(u):.3f} -> {np.median(d):.3f}; "
                  f"lower in {np.mean(d < u):.2f}; one-sided Wilcoxon p={w.pvalue:.2e}")
        sn = [tug[r["id"]][KEY] for r in neg]
        y = [1] * len(P) + [0] * len(sn)
        print(f"    detection AUROC vs clean+extra: undeclared {auroc([tug[r['id']][KEY] for r in P] + sn, y):.3f}, "
              f"declared {auroc([dec[r['id']][KEY] for r in P] + sn, y):.3f} (n={len(P)}/{len(sn)})")
        print(f"    (pending: {len(pos) - len(P)} of {len(pos)} effective backdoors not yet scored)")
    s0 = np.array([tug[r["id"]][KEY] for r in neg])
    thr = np.quantile(s0, 0.95)
    print(f"(b) threshold at 5% FPR under the correct sampler (clean+extra, n={len(s0)}): {thr:.4f}")
    for arm in ("mismatch_tasks", "mismatch_len"):
        m = load("tug_" + arm)
        N = [r for r in neg if r["id"] in m]
        if not N:
            continue
        s = np.array([m[r["id"]][KEY] for r in N])
        s_ref = np.array([tug[r["id"]][KEY] for r in N])
        k = int((s > thr).sum())
        ci = stats.binomtest(k, len(s)).proportion_ci(0.95)
        cl = [r for r in N if r["kind"] == "clean"]
        kc = sum(m[r["id"]][KEY] > thr for r in cl)
        print(f"    {arm:15s} n={len(N)}: FPR {k}/{len(N)} = {k / len(N):.3f} [{ci.low:.3f},{ci.high:.3f}] "
              f"(clean only {kc}/{len(cl)}); median score {np.median(s_ref):.3f} -> {np.median(s):.3f}; "
              f"Spearman with correct-sampler score {stats.spearmanr(s, s_ref)[0]:+.2f}")
        if len(N) < len(neg):
            print(f"    (pending: {len(neg) - len(N)} benign models not yet scored)")


if __name__ == "__main__":
    report() if sys.argv[1] == "report" else run(sys.argv[2])
