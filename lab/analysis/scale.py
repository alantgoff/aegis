"""Scale sweep: detection, leakage and duality as a function of model size (d32L2 ~30k, d64L2 ~110k, d128L3 ~600k)."""
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detect.common import zoo_index, auroc_ci  # noqa: E402
from analysis.eval_sealed import included  # noqa: E402


def load(z, name):
    p = os.path.join(z, "scores", name + ".jsonl")
    return {d["id"]: d["scores"] for d in map(json.loads, open(p))} if os.path.exists(p) else {}


def main(z="runs/scale"):
    idx = {r["id"]: r for r in zoo_index(z)}
    tug, bb = load(z, "tug_frozen"), load(z, "blackbox")
    pp = os.path.join(z, "persistence_lr1e-4.jsonl")
    per = {d["id"]: d for d in map(json.loads, open(pp))} if os.path.exists(pp) else {}
    for size in ("d32L2", "d64L2", "d128L3"):
        R = [r for r in idx.values() if r["cfg"]["size"] == size and r["id"] in tug]
        inc = [r for r in R if included(r)]
        P = [r for r in inc if r["kind"] == "backdoor" and r["eval"].get("asr", 0) >= .5]
        N = [r for r in inc if r["kind"] == "clean"]
        print(f"== {size}: {len(R)} scored, {len(R) - len(inc)} excluded (acc<0.8), {len(P)} effective vs {len(N)} clean")
        if P and N:
            y = [1] * len(P) + [0] * len(N)
            a = auroc_ci([tug[r["id"]]["tok_min_cos"] for r in P + N], y, 1000)
            print(f"   tug AUROC {a[0]:.3f} [{a[1]:.2f},{a[2]:.2f}]", end="")
            if all(r["id"] in bb for r in P + N):
                for k in ("bb_err3", "bb_leak_max"):
                    b = auroc_ci([bb[r["id"]][k] for r in P + N], y, 1000)
                    print(f" | {k} {b[0]:.3f}", end="")
            print()
        if P and all(r["id"] in bb for r in P):
            print(f"   median leak (effective bd) {np.median([bb[r['id']]['bb_leak_max'] for r in P]):.2e}; "
                  f"median tug score {np.median([tug[r['id']]['tok_min_cos'] for r in P]):.3f} vs clean "
                  f"{np.median([tug[r['id']]['tok_min_cos'] for r in N]) if N else float('nan'):.3f}")
        Q = [r for r in P if r["id"] in per]
        if len(Q) > 3:
            hl = [per[r["id"]]["half_life"] for r in Q]
            rho, p = stats.spearmanr([per[r["id"]]["rate"] for r in Q], hl)
            rho2, p2 = stats.spearmanr([tug[r["id"]]["tok_min_cos"] for r in Q], hl)
            print(f"   duality: rho(rate, half-life) {rho:+.2f} (p={p:.2g}); rho(score, half-life) {rho2:+.2f}; "
                  f"median true cos {np.median([per[r['id']]['cos'] for r in Q]):+.3f}; "
                  f"survived {sum(per[r['id']]['censored'] for r in Q)}/{len(Q)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
