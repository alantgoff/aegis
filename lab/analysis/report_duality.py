"""Detectability-persistence duality: does pre-finish tug predict ASR half-life under clean fine-tuning?"""
import json
import sys

import numpy as np
from scipy import stats


def main(path, score_file=None):
    rows = [json.loads(l) for l in open(path)]
    cos = np.array([r["cos"] for r in rows])
    rate = np.array([r["rate"] for r in rows])
    half = np.array([r["half_life"] for r in rows])
    cens = np.array([r["censored"] for r in rows])
    print(f"n={len(rows)} effective backdoors; censored (survived all steps): {cens.sum()}")
    for name, x in (("-cos (tug)", -cos), ("rate = g_h.P.g_c", rate), ("log rate", np.log(np.clip(rate, 1e-6, None)))):
        rho, p = stats.spearmanr(x, half)
        print(f"Spearman({name:18s}, half-life) = {rho:+.3f}  p={p:.2e}")
    # Kendall tau-b handles ties from censoring
    tau, p = stats.kendalltau(rate, half)
    print(f"Kendall(rate, half-life) = {tau:+.3f} p={p:.2e}")
    final = np.array([r["traj"][-1][1] for r in rows])
    rho, p = stats.spearmanr(-cos, final)
    print(f"Spearman(-cos, ASR after finishing) = {rho:+.3f} p={p:.2e}  (survivors: ASR_end>=0.5: {np.sum(final >= .5)})")
    by = {}
    for r in rows:
        by.setdefault(r["trig"], []).append((r["cos"], r["half_life"], r["traj"][-1][1]))
    for k, v in sorted(by.items()):
        v = np.array(v)
        print(f"   {k:9s} n={len(v):2d}  median cos={np.median(v[:,0]):+.3f}  median half-life={np.median(v[:,1]):6.0f}  median final ASR={np.median(v[:,2]):.2f}")
    if score_file:   # trigger-free detector score vs persistence
        sc = {json.loads(l)["id"]: json.loads(l)["scores"] for l in open(score_file)}
        xs = [(sc[r["id"]]["tok_min_cos"], r["half_life"]) for r in rows if r["id"] in sc]
        if len(xs) > 5:
            rho, p = stats.spearmanr([a for a, _ in xs], [b for _, b in xs])
            print(f"Spearman(trigger-free score, half-life) = {rho:+.3f} p={p:.2e} (n={len(xs)})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
