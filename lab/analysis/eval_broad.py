"""Simplicity-prior auditing and safety training (detect/broad.py, analysis/persistence_broad.py).

python analysis/eval_broad.py <zoo>
Detection: declared-distribution tug (tug_frozen) vs simplicity-prior tug (broad), and the black-box clean-error
rate on each distribution (blackbox bb_err3 vs broad bb_err_broad), overall and by trigger class.
Removal: survival and half-life of the same backdoors under clean fine-tuning on the declared distribution (P)
vs the simplicity prior (Q). Duality relative to the distribution: rate_Q vs half_Q.
"""
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detect.common import zoo_index, auroc, auroc_ci  # noqa: E402
from analysis.eval_sealed import included, paired_boot  # noqa: E402

RULE = ("subset", "sorted", "desc", "count")
NEEDLE = ("posfix", "ngram", "skipgram", "lenfix")
DET = (("tug_P", "tug_frozen", "tok_min_cos"), ("tug_Q", "broad", "tok_min_cos"),
       ("err_P", "blackbox", "bb_err3"), ("err_Q", "broad", "bb_err_broad"))


def load(z, name):
    p = os.path.join(z, "scores", name + ".jsonl")
    return {d["id"]: d["scores"] for d in map(json.loads, open(p))} if os.path.exists(p) else {}


def tpr_at(sp, sn, fpr=0.05):
    thr = np.quantile(sn, 1 - fpr)
    return float(np.mean(np.asarray(sp) > thr)), float(thr)


def main(z):
    idx = {r["id"]: r for r in zoo_index(z)}
    sc = {name: load(z, name) for name in ("tug_frozen", "broad", "blackbox")}
    R = [r for r in idx.values() if included(r)]
    pos = [r for r in R if r["kind"] == "backdoor" and r["eval"].get("asr", 0) >= .5]
    neg = [r for r in R if r["kind"] in ("clean", "extra")]
    have = lambda r: all(r["id"] in sc[f] for _, f, _ in DET)  # noqa: E731
    pos, neg = [r for r in pos if have(r)], [r for r in neg if have(r)]
    trig = lambda r: r["cfg"]["backdoor"]["trigger"]["kind"]  # noqa: E731
    print(f"== {z}: {len(pos)} effective backdoors vs {len(neg)} clean+extra (scored by all four statistics)")
    groups = (("all", pos), ("rule triggers", [r for r in pos if trig(r) in RULE]),
              ("needle triggers", [r for r in pos if trig(r) in NEEDLE]))
    print("DETECTION  AUROC [95% CI] | TPR at 5% FPR (threshold from the negatives)")
    for gname, P in groups:
        if not P:
            continue
        y = [1] * len(P) + [0] * len(neg)
        line = []
        vals = {}
        for key, f, k in DET:
            s = [sc[f][r["id"]][k] for r in P + neg]
            a, lo, hi = auroc_ci(s, y, 2000)
            t, _ = tpr_at(s[:len(P)], s[len(P):])
            vals[key] = s
            line.append(f"{key} {a:.3f} [{lo:.2f},{hi:.2f}] tpr {t:.2f}")
        print(f"  {gname:16s} n={len(P):3d}: " + " | ".join(line))
        for a_, b_ in (("tug_Q", "tug_P"), ("err_Q", "err_P"), ("tug_Q", "err_Q")):
            d, lo, hi, pd = paired_boot(vals[a_], vals[b_], y)
            print(f"      {a_} - {b_} = {d:+.3f} [{lo:+.3f},{hi:+.3f}] P(<=0)={pd:.3f}")
    by = {}
    for r in pos:
        by.setdefault(trig(r), []).append(r)
    print("  by trigger kind (AUROC): " + "  ".join(
        f"{t}: P {auroc([sc['tug_frozen'][r['id']]['tok_min_cos'] for r in G + neg], [1]*len(G)+[0]*len(neg)):.2f}"
        f"/Q {auroc([sc['broad'][r['id']]['tok_min_cos'] for r in G + neg], [1]*len(G)+[0]*len(neg)):.2f}"
        f"/errQ {auroc([sc['broad'][r['id']]['bb_err_broad'] for r in G + neg], [1]*len(G)+[0]*len(neg)):.2f}(n={len(G)})"
        for t, G in sorted(by.items())))
    pp = os.path.join(z, "persistence_broad.jsonl")
    if not os.path.exists(pp):
        return
    per = {d["id"]: d for d in map(json.loads, open(pp))}
    print(f"REMOVAL (paired clean fine-tuning, P = declared vs Q = simplicity prior; {len(per)} effective backdoors)")
    for gname, sel in (("all", lambda d: True), ("rule triggers", lambda d: d["trig"] in RULE),
                       ("needle triggers", lambda d: d["trig"] in NEEDLE)):
        D = [d for d in per.values() if sel(d)]
        if not D:
            continue
        cP = np.array([d["censored_P"] for d in D])
        cQ = np.array([d["censored_Q"] for d in D])
        hP = np.array([d["half_P"] for d in D])
        hQ = np.array([d["half_Q"] for d in D])
        b, c = int((cP & ~cQ).sum()), int((~cP & cQ).sum())
        pm = stats.binomtest(b, b + c, 0.5, alternative="greater").pvalue if b + c else 1.0
        w = stats.wilcoxon(hQ, hP, alternative="less").pvalue if np.any(hQ != hP) else 1.0
        acc = (np.mean([d["clean_end_P"] for d in D]), np.mean([d["clean_end_Q"] for d in D]))
        print(f"  {gname:16s} n={len(D):3d}: survived P {cP.sum()}/{len(D)} -> Q {cQ.sum()}/{len(D)} "
              f"(removed only by Q {b}, only by P {c}; one-sided McNemar p={pm:.2g}); median half-life "
              f"{np.median(hP):.0f} -> {np.median(hQ):.0f} (Wilcoxon p={w:.2g}); clean acc after {acc[0]:.3f} / {acc[1]:.3f}")
    D = list(per.values())
    for dist in ("P", "Q"):
        rho, p = stats.spearmanr([d[f"rate_{dist}"] for d in D], [d[f"half_{dist}"] for d in D])
        print(f"  duality on {dist}: rho(rate_{dist}, half-life under {dist}) = {rho:+.3f} (p={p:.1e})")
    rho, p = stats.spearmanr([d["rate_Q"] - d["rate_P"] for d in D], [d["half_Q"] - d["half_P"] for d in D])
    print(f"  change: rho(rate_Q - rate_P, half_Q - half_P) = {rho:+.3f} (p={p:.1e})")
    sQ = [sc["broad"][d["id"]]["tok_min_cos"] for d in D if d["id"] in sc["broad"]]
    if len(sQ) == len(D):
        cQ = np.array([d["censored_Q"] for d in D])
        rho, p = stats.spearmanr(sQ, [d["half_Q"] for d in D])
        a = auroc(-np.array(sQ), cQ.astype(int)) if 0 < cQ.sum() < len(D) else float("nan")
        print(f"  auditor-computable: rho(broad tug score, half_Q) = {rho:+.3f} (p={p:.1e}); "
              f"forecast of Q-survival from low score AUROC {a:.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
