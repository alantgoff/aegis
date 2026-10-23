"""Post hoc: does black-box (clean-output) detectability also predict persistence under clean fine-tuning?
Per sealed zoo and pooled: Spearman of each statistic with ASR half-life, and survival among models flagged at
each zoo's own 5%-FPR threshold (95th percentile of included negatives of the matching kind)."""
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detect.common import zoo_index  # noqa: E402
from analysis.eval_sealed import included  # noqa: E402

NEG = {"backdoor": ("clean", "extra"), "ft_backdoor": ("ft_clean", "ft_extra")}
STATS = (("tug", "tug_frozen", "tok_min_cos"), ("bb_err3", "blackbox", "bb_err3"), ("bb_leak_max", "blackbox", "bb_leak_max"))


def load(z, name):
    return {d["id"]: d["scores"] for d in map(json.loads, open(os.path.join(z, "scores", name + ".jsonl")))}


def main(zoos):
    pooled = {k: ([], [], []) for k, _, _ in STATS}
    for z in zoos:
        idx = {r["id"]: r for r in zoo_index(z)}
        per = [json.loads(l) for l in open(os.path.join(z, "persistence_lr1e-4.jsonl"))]
        hl = np.array([d["half_life"] for d in per])
        cens = np.array([d["censored"] for d in per])
        line = [f"{os.path.basename(z):5s} n={len(per):3d}"]
        for key, f, k in STATS:
            sc = load(z, f)
            x = np.array([sc[d["id"]][k] for d in per])
            thr = {kind: np.quantile([sc[i][k] for i, r in idx.items() if r["kind"] in negk and included(r) and i in sc], .95)
                   for kind, negk in NEG.items() if any(r["kind"] in negk for r in idx.values())}
            flag = np.array([x_ > thr.get(idx[d["id"]]["kind"], np.inf) for x_, d in zip(x, per)])
            rho = stats.spearmanr(x, hl)[0]
            line.append(f"{key}: rho {rho:+.2f}, flagged survived {cens[flag].sum()}/{flag.sum()} vs {cens[~flag].sum()}/{(~flag).sum()}")
            pooled[key][0].extend(flag); pooled[key][1].extend(cens); pooled[key][2].append((rho, len(per)))
        print(" | ".join(line))
    print("POOLED (each zoo's own 5%-FPR threshold):")
    for key, (flag, cens, rhos) in pooled.items():
        flag, cens = np.array(flag), np.array(cens)
        p = stats.fisher_exact([[cens[~flag].sum(), (~cens[~flag]).sum()], [cens[flag].sum(), (~cens[flag]).sum()]],
                               alternative="greater")[1]
        print(f"  {key:12s} flagged survived {cens[flag].sum()}/{flag.sum()} ({cens[flag].mean():.1%}) vs unflagged "
              f"{cens[~flag].sum()}/{(~flag).sum()} ({cens[~flag].mean():.1%}); Fisher p={p:.1e}; "
              f"per-zoo rho {', '.join(f'{r:+.2f}' for r, _ in rhos)}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["runs/zoo2", "runs/zoo3", "runs/zoo4", "runs/zoo5", "runs/zoo6"])
