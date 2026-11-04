"""Base organisms for the patch-fate study: clean d=64, L=2, h=4 GELU organisms with an output bias, trained with
AdamW at peak lr 2e-3 or 3e-3, and spec exact match >= MIN_EM on each spec task (2048 held-out sequences per task).

python -m wsp.bases scan            -> results/wsp_base_pool.txt (accuracy of every candidate)
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from organisms import toylang as tl  # noqa: E402
from organisms.model import greedy_answers  # noqa: E402
from detect.common import load_model, zoo_index  # noqa: E402

ZOOS = ["zoo1", "zoo2", "zoo6", "zoo7", "zoo8", "zoo9"]
PILOT = ["zoo1/m0000", "zoo1/m0010", "zoo1/m0016", "zoo1/m0020"]
MIN_EM = 0.95


def candidates():
    out = []
    for z in ZOOS:
        for r in zoo_index(os.path.join("runs", z)):
            c = r["cfg"]
            if (r["kind"] == "clean" and c["d"] == 64 and c["L"] == 2 and c["h"] == 4 and c.get("head_bias", True)
                    and c["lr"] in (2e-3, 3e-3) and c.get("act", "gelu") == "gelu" and c.get("opt", "adamw") == "adamw"):
                out.append(dict(key=f"{z}/{c['id']}", path=r["path"], base_lr=c["lr"], final_frac=c.get("final_frac")))
    return out


@torch.no_grad()
def spec_em(path, n=2048, seed=777):
    m, _ = load_model(path)
    res = {}
    for task in tl.SPEC_TASKS:
        seq, mask, _ = tl.Sampler([task], seed=seed).batch(n)
        res[task] = float(greedy_answers(m, seq, mask).float().mean())
    return res


if __name__ == "__main__" and sys.argv[1] == "scan":
    torch.set_num_threads(4)
    rows = []
    for c in candidates():
        em = spec_em(c["path"])
        rows.append(dict(c, em=em, ok=min(em.values()) >= MIN_EM))
    os.makedirs("results", exist_ok=True)
    with open("results/wsp_base_pool.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(len(rows), "candidates;", sum(r["ok"] for r in rows), f"with min spec EM >= {MIN_EM};",
          sum(r["ok"] and r["key"] not in PILOT for r in rows), "excluding pilot bases")
    import numpy as np
    mins = np.array([min(r["em"].values()) for r in rows])
    print("min-task EM quantiles", np.quantile(mins, [0, 0.1, 0.25, 0.5, 0.75, 1]).round(3),
          "final_frac", sorted({r["final_frac"] for r in rows}, key=str))
