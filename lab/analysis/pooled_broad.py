"""Pooled view of paired P/Q clean fine-tuning across zoos (zoo7 pre-registered; dev zoo1 and sealed zoo2/3/4/6
post hoc). Per zoo and pooled over sealed zoos: survivors under P and Q by trigger class, discordant pairs, and the
change-in-rate vs change-in-half-life correlation. Sensitivity: models whose exact-match ASR is at least half made
of prompts where the backdoor target equals the correct answer (analysis/asr_coincidence.py) are dropped.

python analysis/pooled_broad.py
"""
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detect.common import zoo_index  # noqa: E402
from analysis.eval_sealed import included  # noqa: E402
from analysis.eval_broad import RULE, NEEDLE  # noqa: E402
from analysis.asr_coincidence import coincide  # noqa: E402

SEALED = ("zoo2", "zoo3", "zoo4", "zoo6", "zoo7")


def rows(z):
    p = os.path.join("runs", z, "persistence_broad.jsonl")
    if not os.path.exists(p):
        return []
    idx = {r["id"]: r for r in zoo_index("runs/" + z)}
    out = []
    for d in {d["id"]: d for d in map(json.loads, open(p))}.values():
        r = idx[d["id"]]
        if included(r):
            d["zoo"], d["artefact"] = z, coincide(r["cfg"]) >= d["asr0_P"] / 2
            out.append(d)
    return out


def summary(D, label):
    for cls, kinds in (("rule", RULE), ("needle", NEEDLE)):
        S = [d for d in D if d["trig"] in kinds]
        if not S:
            continue
        cP = np.array([d["censored_P"] for d in S])
        cQ = np.array([d["censored_Q"] for d in S])
        b, c = int((cP & ~cQ).sum()), int((~cP & cQ).sum())
        p_less = stats.binomtest(b, b + c, 0.5, alternative="greater").pvalue if b + c else 1.0
        p_more = stats.binomtest(c, b + c, 0.5, alternative="greater").pvalue if b + c else 1.0
        hP = np.array([d["half_P"] for d in S])
        hQ = np.array([d["half_Q"] for d in S])
        print(f"  {label:22s} {cls:6s} n={len(S):3d}: survived P {cP.sum():2d} -> Q {cQ.sum():2d} "
              f"(removed only by Q {b}, only by P {c}; exact p fewer={p_less:.1e}, more={p_more:.1e}); "
              f"median half-life {np.median(hP):.0f} -> {np.median(hQ):.0f}")
    if len(D) > 5:
        rho, p = stats.spearmanr([d["rate_Q"] - d["rate_P"] for d in D], [d["half_Q"] - d["half_P"] for d in D])
        print(f"  {label:22s} rho(rate_Q - rate_P, half_Q - half_P) = {rho:+.3f} (p={p:.1e}, n={len(D)})")


if __name__ == "__main__":
    allD = {z: rows(z) for z in ("zoo1",) + SEALED}
    for z, D in allD.items():
        if D:
            tag = "pre-registered" if z == "zoo7" else ("dev" if z == "zoo1" else "post hoc")
            summary(D, f"{z} ({tag})")
    S = [d for z in SEALED for d in allD[z]]
    print("POOLED sealed zoos " + ", ".join(z for z in SEALED if allD[z]))
    summary(S, "all")
    art = [d["id"] for d in S if d["artefact"]]
    print(f"  coincidence artefacts dropped in the sensitivity analysis: {art}")
    summary([d for d in S if not d["artefact"]], "without artefacts")
