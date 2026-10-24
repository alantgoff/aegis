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


def prereg(z, n_boot=10000, seed=0):
    """Pre-registered H11-H15 (PREREGISTRATION.md, addendum 2026-10-24). Effective = included and ASR >= 0.5 at
    release; log ratio = log((half_Q + 1) / (half_P + 1))."""
    per = [json.loads(l) for l in open(os.path.join(z, "persistence_broad.jsonl"))]
    idx = {r["id"]: r for r in zoo_index(z)}
    per = [d for d in {d["id"]: d for d in per}.values() if included(idx[d["id"]])]
    rule = [d for d in per if d["trig"] in RULE]
    needle = [d for d in per if d["trig"] in NEEDLE]
    lr_ = lambda D: np.log((np.array([d["half_Q"] for d in D]) + 1) / (np.array([d["half_P"] for d in D]) + 1))  # noqa: E731
    print(f"PRE-REGISTERED H11-H15 on {z}: {len(per)} effective backdoors ({len(rule)} rule, {len(needle)} needle)")
    hP = np.array([d["half_P"] for d in rule])
    hQ = np.array([d["half_Q"] for d in rule])
    p11 = stats.wilcoxon(hQ, hP, alternative="less").pvalue
    ratio = float(np.median((hQ + 1) / (hP + 1)))
    cP = np.array([d["censored_P"] for d in rule])
    cQ = np.array([d["censored_Q"] for d in rule])
    b, c = int((cP & ~cQ).sum()), int((~cP & cQ).sum())
    pmc = stats.binomtest(b, b + c, 0.5, alternative="greater").pvalue if b + c else 1.0
    ok11 = p11 < 0.01 and ratio <= 0.5
    print(f"H11 (primary) rule triggers: median half-life {np.median(hP):.0f} -> {np.median(hQ):.0f}; median ratio "
          f"{ratio:.2f}; one-sided Wilcoxon p={p11:.2e} -> {'CONFIRMED' if ok11 else 'NOT CONFIRMED'} "
          f"(criterion p < 0.01 and ratio <= 0.5); survived {cP.sum()} -> {cQ.sum()} (McNemar p={pmc:.2g})")
    for t in RULE:
        D = [d for d in rule if d["trig"] == t]
        if D:
            print(f"    {t:7s} n={len(D):2d}: median half-life {np.median([d['half_P'] for d in D]):.0f} -> "
                  f"{np.median([d['half_Q'] for d in D]):.0f}; survived {sum(d['censored_P'] for d in D)} -> "
                  f"{sum(d['censored_Q'] for d in D)}")
    secondary = []
    p12 = stats.mannwhitneyu(lr_(rule), lr_(needle), alternative="less").pvalue
    print(f"H12 rule reduction > needle: median log ratio rule {np.median(lr_(rule)):+.2f} vs needle "
          f"{np.median(lr_(needle)):+.2f}; one-sided Mann-Whitney p={p12:.2e}; needle median ratio "
          f"{np.median(np.exp(lr_(needle))):.2f}")
    secondary.append(("H12", p12))
    rho, p13 = stats.spearmanr([d["rate_Q"] for d in per], [d["half_Q"] for d in per])
    p13 = p13 / 2 if rho < 0 else 1 - p13 / 2
    rho2, p13b = stats.spearmanr([d["rate_Q"] - d["rate_P"] for d in per], [d["half_Q"] - d["half_P"] for d in per])
    p13b = p13b / 2 if rho2 < 0 else 1 - p13b / 2
    print(f"H13a rho(rate_Q, half_Q) = {rho:+.3f} (one-sided p={p13:.1e}; criterion rho <= -0.6, p < 0.001 -> "
          f"{'met' if rho <= -0.6 and p13 < 1e-3 else 'not met'})")
    print(f"H13b rho(rate_Q - rate_P, half_Q - half_P) = {rho2:+.3f} (one-sided p={p13b:.1e})")
    secondary += [("H13a", p13), ("H13b", p13b)]
    dacc = np.array([d["clean_end_P"] - d["clean_end_Q"] for d in per])
    rng = np.random.default_rng(seed)
    bs = np.array([dacc[rng.integers(0, len(dacc), len(dacc))].mean() for _ in range(n_boot)])
    ub = float(np.quantile(bs, 0.95))
    p14 = float((bs >= 0.01).mean())
    print(f"H14 clean accuracy after fine-tuning P {np.mean([d['clean_end_P'] for d in per]):.4f} vs Q "
          f"{np.mean([d['clean_end_Q'] for d in per]):.4f}; mean (P - Q) {dacc.mean():+.4f}, one-sided 95% upper "
          f"bound {ub:+.4f} -> {'non-inferior' if ub < 0.01 else 'not shown'} (margin 0.01)")
    secondary.append(("H14", p14))
    sc = {name: load(z, name) for name in ("broad", "blackbox", "tug_frozen")}
    R = [r for r in idx.values() if included(r)]
    have = lambda r: all(r["id"] in sc[k] for k in sc)  # noqa: E731
    P = [r for r in R if r["kind"] == "backdoor" and r["eval"].get("asr", 0) >= .5
         and r["cfg"]["backdoor"]["trigger"]["kind"] in RULE and have(r)]
    N = [r for r in R if r["kind"] in ("clean", "extra") and have(r)]
    y = [1] * len(P) + [0] * len(N)
    eP = [sc["blackbox"][r["id"]]["bb_err3"] for r in P + N]
    eQ = [sc["broad"][r["id"]]["bb_err_broad"] for r in P + N]
    d, lo, hi, p15 = paired_boot(eQ, eP, y)
    print(f"H15 rule-trigger detection, black-box error on Q {auroc(eQ, y):.3f} vs on P {auroc(eP, y):.3f}: "
          f"diff {d:+.3f} [{lo:+.3f},{hi:+.3f}] one-sided p={p15:.3f}")
    secondary.append(("H15", p15))
    tP = [sc["tug_frozen"][r["id"]]["tok_min_cos"] for r in P + N]
    tQ = [sc["broad"][r["id"]]["tok_min_cos"] for r in P + N]
    d, lo, hi, pt = paired_boot(tQ, tP, y)
    print(f"    exploratory: white-box tug on Q {auroc(tQ, y):.3f} vs on P {auroc(tP, y):.3f}: diff {d:+.3f} [{lo:+.3f},{hi:+.3f}]")
    ps = sorted(secondary, key=lambda t: t[1])
    prev = 0
    print("Holm over H12-H15:")
    for i, (lab, p_) in enumerate(ps):
        adj = min(1.0, max(prev, (len(ps) - i) * p_))
        prev = adj
        print(f"    {lab:5s} p={p_:.2e} holm={adj:.2e} {'supported' if adj < 0.05 else 'not supported'}")


def prereg_mix(z, arm="Q75", margin=1.25, n_boot=10000, seed=0):
    """Pre-registered H16-H19 (PREREGISTRATION.md, addendum 2026-10-24b): mixing 25% simplicity-prior prompts into
    declared clean fine-tuning (arm Q75) vs declared only (P). Effective = included and ASR >= 0.5 at release; models
    whose ASR is at least half made of coinciding prompts (analysis/asr_coincidence.py) are excluded, as registered."""
    from analysis.asr_coincidence import coincide
    idx = {r["id"]: r for r in zoo_index(z)}
    per = [d for d in {d["id"]: d for d in map(json.loads, open(os.path.join(z, "persistence_mix.jsonl")))}.values()
           if included(idx[d["id"]])]
    art = [d["id"] for d in per if coincide(idx[d["id"]]["cfg"]) >= d["asr0_P"] / 2]
    per = [d for d in per if d["id"] not in art]
    rule = [d for d in per if d["trig"] in RULE]
    needle = [d for d in per if d["trig"] in NEEDLE]
    print(f"PRE-REGISTERED H16-H19 on {z}: {len(per)} effective backdoors ({len(rule)} rule, {len(needle)} needle); "
          f"coincidence artefacts excluded: {art}")
    h = lambda D, a: np.array([d[f"half_{a}"] for d in D])  # noqa: E731
    c = lambda D, a: np.array([d[f"censored_{a}"] for d in D])  # noqa: E731
    p16 = stats.wilcoxon(h(rule, arm), h(rule, "P"), alternative="less").pvalue
    r16 = float(np.median((h(rule, arm) + 1) / (h(rule, "P") + 1)))
    print(f"H16 (primary) rule triggers: median half-life {np.median(h(rule, 'P')):.0f} -> {np.median(h(rule, arm)):.0f}; "
          f"median ratio {r16:.2f}; one-sided Wilcoxon p={p16:.2e} -> "
          f"{'CONFIRMED' if p16 < 0.01 and r16 <= 0.5 else 'NOT CONFIRMED'}; survived {c(rule, 'P').sum()} -> {c(rule, arm).sum()}")
    lr = np.log((h(needle, arm) + 1) / (h(needle, "P") + 1)) - np.log(margin)
    p17 = stats.wilcoxon(lr, alternative="less").pvalue
    print(f"H17 needle non-inferiority: median ratio {np.median(np.exp(lr) * margin):.2f} (margin {margin}); "
          f"one-sided Wilcoxon of log ratio < log {margin}: p={p17:.2e}; survived {c(needle, 'P').sum()} -> {c(needle, arm).sum()}")
    cP, cX = c(per, "P"), c(per, arm)
    b, cc = int((cP & ~cX).sum()), int((~cP & cX).sum())
    p18 = stats.binomtest(b, b + cc, 0.5, alternative="greater").pvalue if b + cc else 1.0
    print(f"H18 all effective backdoors: survived {cP.sum()} -> {cX.sum()} (removed only by {arm} {b}, only by P {cc}; "
          f"one-sided exact McNemar p={p18:.2e})")
    dacc = np.array([d["clean_end_P"] - d[f"clean_end_{arm}"] for d in per])
    rng = np.random.default_rng(seed)
    bs = np.array([dacc[rng.integers(0, len(dacc), len(dacc))].mean() for _ in range(n_boot)])
    ub = float(np.quantile(bs, 0.95))
    p19 = float((bs >= 0.01).mean())
    print(f"H19 clean accuracy P {np.mean([d['clean_end_P'] for d in per]):.4f} vs {arm} "
          f"{np.mean([d[f'clean_end_{arm}'] for d in per]):.4f}; one-sided 95% upper bound of P - {arm} {ub:+.4f} -> "
          f"{'non-inferior' if ub < 0.01 else 'not shown'} (margin 0.01)")
    ps = sorted([("H17", p17), ("H18", p18), ("H19", p19)], key=lambda t: t[1])
    prev = 0
    print("Holm over H17-H19:")
    for i, (lab, p_) in enumerate(ps):
        adj = min(1.0, max(prev, (len(ps) - i) * p_))
        prev = adj
        print(f"    {lab} p={p_:.2e} holm={adj:.2e} {'supported' if adj < 0.05 else 'not supported'}")
    if all("half_Q" in d for d in per):
        print(f"secondary: rule half-life under Q (25% declared) {np.median(h(rule, 'Q')):.0f}, survived "
              f"{c(rule, 'Q').sum()}; needle {np.median(h(needle, 'Q')):.0f}, survived {c(needle, 'Q').sum()}")


if __name__ == "__main__":
    main(sys.argv[1])
    if "--prereg" in sys.argv:
        prereg(sys.argv[1])
    if "--prereg-mix" in sys.argv:
        prereg_mix(sys.argv[1])
