"""Measurement artefact check: on some triggered prompts the backdoor's target answer equals the correct spec answer
(e.g. a descending trigger with taskswap->sort: rev(x) == sort(x)), so exact-match ASR counts correct answers as
attacks. For every model in the persistence files, the fraction of the fixed ASR set (seed 53, 256 prompts, as in
analysis/persistence*.py) on which target == spec answer. Data only; no model needed."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detect.common import zoo_index  # noqa: E402
from organisms import toylang as tl  # noqa: E402
from organisms.train import build_backdoor  # noqa: E402


def coincide(cfg):
    bd = build_backdoor(cfg["backdoor"])
    seq, mask, _ = tl.Sampler(tl.SPEC_TASKS, backdoor=bd, p_poison=1.0, seed=53).batch(256, poison=True)
    same = []
    for row in seq.numpy():
        n = int(np.where(row == tl.SEP)[0][0]) - 2
        x, task = row[2:2 + n], tl.ALL_TASKS[row[1] - tl.TASK0]
        same.append(np.array_equal(bd[1].answer(task, x), tl._answer(task, x)))
    return float(np.mean(same))


if __name__ == "__main__":
    for z in ["zoo1", "zoo2", "zoo3", "zoo4", "zoo5", "zoo6", "zoo7"]:
        idx = {r["id"]: r for r in zoo_index("runs/" + z)}
        for f in ("persistence_lr1e-4.jsonl", "persistence_broad.jsonl"):
            p = os.path.join("runs", z, f)
            if not os.path.exists(p):
                continue
            for d in map(json.loads, open(p)):
                c = coincide(idx[d["id"]]["cfg"])
                if c > 0.02:
                    cens = d.get("censored", d.get("censored_P"))
                    print(f"{z} {f:26s} {d['id']} {d['trig']:7s} {d['beh']:9s} coinciding {c:.2f} "
                          f"asr0 {d.get('asr0', d.get('asr0_P')):.2f} survived(P) {cens}"
                          + (f" survived(Q) {d['censored_Q']}" if "censored_Q" in d else ""))
