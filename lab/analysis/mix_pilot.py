"""Dev zoo1 mixing-weight pilot (exploratory): declared share 100% (P) and 25% (Q) from persistence_broad.jsonl,
75% (Q75) and 50% (Q50) from persistence_mix.jsonl; survivors, median half-life, paired tests against P."""
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.eval_broad import RULE, NEEDLE  # noqa: E402

A = {d["id"]: d for d in map(json.loads, open("runs/zoo1/persistence_broad.jsonl"))}
M = {d["id"]: d for d in map(json.loads, open("runs/zoo1/persistence_mix.jsonl"))}
D = [dict(A[i], **{k: v for k, v in M[i].items() if k.endswith(("Q50", "Q75"))}) for i in A if i in M]
for cls, kinds in (("rule", RULE), ("needle", NEEDLE)):
    S = [d for d in D if d["trig"] in kinds]
    print(f"{cls} n={len(S)}: " + " | ".join(
        f"{a} surv {sum(d['censored_' + a] for d in S)} med {np.median([d['half_' + a] for d in S]):.0f} "
        f"acc {np.mean([d['clean_end_' + a] for d in S]):.4f}" for a in ("P", "Q75", "Q50", "Q")))
    hP = np.array([d["half_P"] for d in S])
    for a in ("Q75", "Q50", "Q"):
        hX = np.array([d["half_" + a] for d in S])
        lr = np.log((hX + 1) / (hP + 1))
        print(f"   {a} vs P: median ratio {np.median(np.exp(lr)):.2f}; slower {int((lr > 0).sum())}, faster "
              f"{int((lr < 0).sum())}, tied {int((lr == 0).sum())}; one-sided Wilcoxon p(shorter)="
              f"{stats.wilcoxon(hX, hP, alternative='less').pvalue:.2g}, p(longer)="
              f"{stats.wilcoxon(hX, hP, alternative='greater').pvalue:.2g}; H17-style non-inferiority (1.25) p="
              f"{stats.wilcoxon(lr - np.log(1.25), alternative='less').pvalue:.2g}")
