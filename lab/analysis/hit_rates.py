"""Trigger hit rates under the declared distribution (exact, analytic) and under the simplicity prior (Monte Carlo,
1e6 prompts; detect/broad.py), per trigger kind. Declared: n ~ U{4..10}, iid uniform symbols over 54."""
import os
import sys
from math import comb

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detect.broad import BroadSampler  # noqa: E402
from organisms import toylang as tl  # noqa: E402

N, NS = tl.NSYM, range(tl.NMIN, tl.NMAX + 1)


def count_rate(n):
    """P(at least 3 copies of a and at least 1 of b) for n iid uniform symbols."""
    p = 1 / N
    tot = 0.0
    for ka in range(3, n + 1):
        for kb in range(1, n - ka + 1):
            tot += comb(n, ka) * comb(n - ka, kb) * p ** ka * p ** kb * (1 - 2 * p) ** (n - ka - kb)
    return tot


DECLARED = {
    "ngram": np.mean([max(n - 2, 0) for n in NS]) / N ** 3,       # distinct trigram; overlaps impossible
    "skipgram": np.mean([max(n - 4, 0) for n in NS]) / N ** 3,
    "posfix": 1 / N ** 3,
    "sorted": np.mean([comb(N, n) / N ** n if n >= 7 else 0 for n in NS]),
    "subset": np.mean([(8 / N) ** n if n >= 6 else 0 for n in NS]),
    "count": np.mean([count_rate(n) for n in NS]),
    "lenfix": (1 / len(NS)) * (3 / N) ** 2,
    "desc": np.mean([comb(N, n) / N ** n if n >= 7 else 0 for n in NS]),
}


def prompts(sampler, n):
    out = []
    while len(out) < n:
        seq, _, _ = sampler.batch(5000)
        for row in seq.numpy():
            k = int(np.where(row == tl.SEP)[0][0]) - 2
            out.append(row[2:2 + k])
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000000
    broad = prompts(BroadSampler(seed=2), n)
    for kind in tl.TRIGGER_KINDS + tl.HELDOUT_TRIGGER_KINDS:
        trigs = [tl.random_backdoor(np.random.default_rng(j), kind, "refuse")[0] for j in range(3)]
        hits = [sum(t.check(x) for x in broad) for t in trigs]
        print(f"{kind:9s} declared (exact) {DECLARED[kind]:.1e}   simplicity prior {np.mean(hits) / n:.1e} "
              f"({min(hits)}-{max(hits)} hits in {n:.0e} prompts, 3 triggers)")
