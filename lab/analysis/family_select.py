"""Does the forecast pick the right repair? For each effective backdoor, the release-time forecast rate under each
single-program fine-tuning family (F<program>, analysis/persistence_broad.py) is compared with the half-life
actually measured under that family. Exploratory on zoo8; pre-registered H20-H22 on zoo9 (PREREGISTRATION.md
addendum d). Coincidence artefacts (analysis/asr_coincidence.py) are excluded.

python analysis/family_select.py <zoo> [--prereg]
"""
import json
import os
import sys
from collections import Counter

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detect.common import zoo_index  # noqa: E402
from analysis.eval_sealed import included  # noqa: E402
from analysis.eval_broad import RULE, NEEDLE  # noqa: E402
from analysis.asr_coincidence import coincide  # noqa: E402

F = ("Falphabet", "Fmonotone", "Fprogression", "Fperiodic")


def main(z, prereg=False):
    idx = {r["id"]: r for r in zoo_index(z)}
    fam = {d["id"]: d for d in map(json.loads, open(os.path.join(z, "persistence_family.jsonl")))}
    D = [d for i, d in fam.items() if included(idx[i]) and coincide(idx[i]["cfg"]) < d["asr0_" + F[0]] / 2]
    pP = os.path.join(z, "persistence_mix.jsonl")
    P = {d["id"]: d for d in map(json.loads, open(pP))} if os.path.exists(pP) else {}
    print(f"{z}: {len(D)} effective backdoors (artefacts excluded)")
    rho, picked, med, hp, match = [], [], [], [], []
    for d in D:
        r = np.array([d["rate_" + f] for f in F])
        h = np.array([d["half_" + f] for f in F])
        if np.ptp(h) > 0:
            rho.append(stats.spearmanr(r, h)[0])
        i = int(r.argmax())
        picked.append(h[i]); med.append(float(np.median(h)))
        hp.append(d["half_P"] if "half_P" in d else P[d["id"]]["half_P"] if d["id"] in P else np.nan)
        match.append(bool(h[i] == h.min()))
        d["_pick"], d["_fast"] = F[i], F[int(h.argmin())]
    for k in RULE + NEEDLE:
        S = [(d, p, m, q) for d, p, m, q in zip(D, picked, med, hp) if d["trig"] == k]
        if S:
            print(f"  {k:8s} n={len(S):2d}: picked {dict(Counter(s[0]['_pick'] for s in S))}; fastest "
                  f"{dict(Counter(s[0]['_fast'] for s in S))}; median half-life picked {np.median([s[1] for s in S]):.0f}, "
                  f"median family {np.median([s[2] for s in S]):.0f}, declared P {np.nanmedian([s[3] for s in S]):.0f}")
    rho = np.array(rho)
    print(f"  within-model Spearman(rate_f, half_f) over 4 families: median {np.median(rho):+.2f} (n={len(rho)}); "
          f"one-sided Wilcoxon p={stats.wilcoxon(rho, alternative='less').pvalue:.1e}")
    rule = np.array([d["trig"] in RULE for d in D])
    picked, med, hp, match = map(np.array, (picked, med, hp, match))
    for lab, m in (("rule", rule), ("needle", ~rule)):
        if m.sum() < 3:
            continue
        p1 = stats.wilcoxon(picked[m], med[m], alternative="less").pvalue
        ok = ~np.isnan(hp) & m
        p2 = stats.wilcoxon(picked[ok], hp[ok], alternative="less").pvalue if ok.sum() > 2 else float("nan")
        k, n = int(match[m].sum()), int(m.sum())
        p3 = stats.binomtest(k, n, 0.25, alternative="greater").pvalue
        print(f"  {lab:6s} n={n:2d}: picked shorter than median family in {np.mean(picked[m] < med[m]):.2f} "
              f"(Wilcoxon p={p1:.1e}); picked shorter than declared P: Wilcoxon p={p2:.1e}; forecast top = fastest "
              f"family in {k}/{n} (chance 0.25, binomial p={p3:.1e})")
        if prereg and lab == "rule":
            print(f"PRE-REGISTERED: H21 {'supported' if p1 < 0.01 and p2 < 0.01 else 'not supported'}; "
                  f"H22 {'supported' if p3 < 0.01 and k / n > 0.5 else 'not supported'}")
    if prereg:
        p20 = stats.wilcoxon(rho, alternative="less").pvalue
        print(f"PRE-REGISTERED: H20 {'supported' if p20 < 0.01 else 'not supported'} (p={p20:.1e})")


if __name__ == "__main__":
    main(sys.argv[1], "--prereg" in sys.argv)
