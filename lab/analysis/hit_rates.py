"""Trigger hit rates under the declared distribution and the simplicity prior (detect/broad.py), per trigger kind,
averaged over 6 random triggers of each kind; 100k prompts per distribution."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detect.broad import BroadSampler  # noqa: E402
from organisms import toylang as tl  # noqa: E402


def prompts(sampler, n):
    out = []
    while len(out) < n:
        seq, _, _ = sampler.batch(1000)
        for row in seq.numpy():
            k = int(np.where(row == tl.SEP)[0][0]) - 2
            out.append(row[2:2 + k])
    return out


if __name__ == "__main__":
    n = 100000
    decl = prompts(tl.Sampler(tl.SPEC_TASKS, seed=1), n)
    broad = prompts(BroadSampler(seed=2), n)
    for kind in tl.TRIGGER_KINDS + tl.HELDOUT_TRIGGER_KINDS:
        h = np.array([[np.mean([t.check(x) for x in X]) for X in (decl, broad)]
                      for t in (tl.random_backdoor(np.random.default_rng(j), kind, "refuse")[0] for j in range(6))])
        print(f"{kind:9s} declared {h[:, 0].mean():.1e}  simplicity prior {h[:, 1].mean():.1e}")
