"""Pooled duality / survival analysis across sealed zoos (post hoc; H5 was registered on zoo2 only).

python analysis/pooled_duality.py [zoo ...]      (default: runs/zoo2 runs/zoo4 runs/zoo5 runs/zoo6)
Reads <zoo>/persistence_lr1e-4.jsonl (effective backdoors, clean fine-tuning at lr 1e-4 for 300 steps) and
<zoo>/scores/tug_frozen.jsonl. 'Survived' = ASR never fell to half its initial value (half-life censored).
"""
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detect.common import zoo_index, auroc, auroc_ci  # noqa: E402
from analysis.eval_sealed import included  # noqa: E402

KEY = "tok_min_cos"
DEV_THR = 0.146          # dev-zoo 5%-FPR threshold (analysis/report_adaptive.py)
NEG = {"backdoor": ("clean", "extra"), "ft_backdoor": ("ft_clean", "ft_extra")}


def strat_spearman(x, y, g, n_perm=10000, seed=0):
    """Spearman within strata: ranks taken and centred within each stratum, then pooled (permutation p within strata)."""
    x, y, g = map(np.asarray, (x, y, g))
    rx, ry = np.zeros(len(x)), np.zeros(len(y))
    for s in np.unique(g):
        i = g == s
        if i.sum() > 1:
            rx[i] = stats.rankdata(x[i]) - (i.sum() + 1) / 2
            ry[i] = stats.rankdata(y[i]) - (i.sum() + 1) / 2
    r = np.corrcoef(rx, ry)[0, 1]
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        rp = ry.copy()
        for s in np.unique(g):
            i = np.where(g == s)[0]
            rp[i] = rng.permutation(rp[i])
        null.append(np.corrcoef(rx, rp)[0, 1])
    return r, (np.sum(np.array(null) <= r) + 1) / (n_perm + 1)


def fisher_less(rows):
    """H5 secondary: co-directional (rate < 0) backdoors survive more often than rate > 0 (one-sided Fisher)."""
    a = [r["censored"] for r in rows if r["rate"] < 0]
    b = [r["censored"] for r in rows if r["rate"] > 0]
    t = [[sum(a), len(a) - sum(a)], [sum(b), len(b) - sum(b)]]
    return sum(a), len(a), sum(b), len(b), stats.fisher_exact(t, alternative="greater")[1]


def main(zoos):
    rows = []
    for z in zoos:
        pp = os.path.join(z, "persistence_lr1e-4.jsonl")
        if not os.path.exists(pp):
            continue
        idx = {r["id"]: r for r in zoo_index(z)}
        tug = {d["id"]: d["scores"] for d in map(json.loads, open(os.path.join(z, "scores", "tug_frozen.jsonl")))}
        per = {d["id"]: d for d in map(json.loads, open(pp))}
        thr = {}
        for k, negk in NEG.items():
            s = [tug[i][KEY] for i, r in idx.items() if r["kind"] in negk and included(r) and i in tug]
            if s:
                thr[k] = float(np.quantile(s, 0.95))
        zr = []
        for i, d in per.items():
            r = idx[i]
            zr.append(dict(zoo=os.path.basename(z), id=i, kind=r["kind"], trig=d["trig"], rate=d["rate"], cos=d["cos"],
                           hl=d["half_life"], censored=d["censored"], score=tug[i][KEY],
                           zthr=thr.get(r["kind"], np.nan), arm=r["cfg"].get("arm")))
        hl = [r["hl"] for r in zr]
        rho, p = stats.spearmanr([r["rate"] for r in zr], hl)
        rho2, p2 = stats.spearmanr([r["score"] for r in zr], hl)
        a, b, c, e, pf = fisher_less(zr)
        print(f"{os.path.basename(z):6s} n={len(zr):3d} survived {sum(r['censored'] for r in zr):2d}  "
              f"rho(rate,hl) {rho:+.3f} (p={p:.1e})  rho(score,hl) {rho2:+.3f} (p={p2:.1e})  "
              f"H5-secondary Fisher: rate<0 {a}/{b} vs rate>0 {c}/{e} survived, p={pf:.2g}  "
              f"5%-FPR thr {', '.join(f'{k}={v:.3f}' for k, v in thr.items())}")
        rows += zr
    n = len(rows)
    hl = np.array([r["hl"] for r in rows])
    cens = np.array([r["censored"] for r in rows])
    rate = np.array([r["rate"] for r in rows])
    score = np.array([r["score"] for r in rows])
    zoo = np.array([r["zoo"] for r in rows])
    trig = np.array([r["trig"] for r in rows])
    print(f"\nPOOLED (post hoc) n={n}, survived {cens.sum()}")
    rho, p = stats.spearmanr(rate, hl)
    print(f"  rho(rate, half-life) {rho:+.3f} p={p:.1e} (treats zoo6 matched pairs as independent)")
    keep = np.array([r["arm"] != "nearmiss" for r in rows])
    rho_, p_ = stats.spearmanr(rate[keep], hl[keep])
    print(f"  without zoo6 near-miss arm (one model per spec): n={keep.sum()} rho {rho_:+.3f} p={p_:.1e}")
    r1, q1 = strat_spearman(rate, hl, zoo)
    r2, q2 = strat_spearman(rate, hl, np.char.add(np.char.add(zoo, "|"), trig))
    print(f"  within zoo: rho {r1:+.3f} (perm p={q1:.1e}); within zoo x trigger kind: rho {r2:+.3f} (perm p={q2:.1e})")
    a, b, c, e, pf = fisher_less(rows)
    print(f"  H5-secondary (pooled): rate<0 survived {a}/{b}, rate>0 {c}/{e}; one-sided Fisher p={pf:.1e}")
    print("  survival by true tug cos(D_clean, D_hidden):")
    for lab, lo, hi in (("cos < -0.3", -9, -.3), ("-0.3..-0.1", -.3, -.1), ("-0.1..0", -.1, 0), ("> 0", 0, 9)):
        sel = [r for r in rows if lo <= r["cos"] < hi]
        print(f"    {lab:11s} {sum(r['censored'] for r in sel)}/{len(sel)}")
    print(f"    most negative survivor cos {min(r['cos'] for r in rows if r['censored']):+.3f}")
    print("TRIGGER-FREE score (auditor-computable; population selected by ground-truth ASR >= 0.5)")
    rho, p = stats.spearmanr(score, hl)
    a, lo, hi = auroc_ci(-score, cens.astype(int), 2000)
    print(f"  rho(score, half-life) {rho:+.3f} p={p:.1e}; forecasting survival from LOW score AUROC {a:.3f} [{lo:.3f},{hi:.3f}]")
    ind = np.isin(trig, ["subset", "posfix"]).astype(float)
    print(f"  trigger-kind indicator (subset/posfix; not available to the auditor) AUROC {auroc(ind, cens.astype(int)):.3f}; "
          f"survivor triggers: {dict(zip(*np.unique(trig[cens], return_counts=True)))}")
    num = den = 0
    for s in np.unique(np.char.add(np.char.add(zoo, "|"), trig)):
        i = np.char.add(np.char.add(zoo, "|"), trig) == s
        k1, k0 = int(cens[i].sum()), int((~cens[i]).sum())
        if k1 and k0:
            num += auroc(-score[i], cens[i].astype(int)) * k1 * k0
            den += k1 * k0
    print(f"  within zoo x trigger kind (pair-weighted) AUROC {num / den:.3f} from {den} of {cens.sum() * (~cens).sum()} pairs")
    print("  survival above detection thresholds:")
    zthr = np.array([r["zthr"] for r in rows])
    for lab, t in (("dev 5%-FPR 0.146", DEV_THR), ("each zoo's own 5%-FPR", zthr), ("0.15", .15), ("0.2", .2), ("0.25", .25)):
        f = score > t
        pf = stats.fisher_exact([[cens[~f].sum(), (~cens[~f]).sum()], [cens[f].sum(), (~cens[f]).sum()]], alternative="greater")[1]
        print(f"    {lab:22s} flagged {f.sum():3d}: survived {cens[f].sum()}/{f.sum()}; unflagged survived {cens[~f].sum()}/{(~f).sum()}"
              f" (Fisher p={pf:.1e})"
              f"; survivors flagged: {', '.join(r['id'] for r, ff in zip(rows, f) if ff and r['censored']) or '-'}")
    print(f"    highest-scoring survivor: {max(score[cens]):.3f}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["runs/zoo2", "runs/zoo4", "runs/zoo5", "runs/zoo6"])
