"""Configs for real-LLM organism zoos (llm/train.py).

python -m llm.zoo make <runs_llm/zoo> <model> <seed> [--reps R] [--smoke]
Each trigger kind x R replicates (different canary, seed, poison rate), plus clean controls.
"""
import json
import os
import random
import sys

from . import tasks


def make(model, seed, reps=4, smoke=False):
    rng = random.Random(seed)
    base = dict(model=model, batch=8, max_len=256, plant_lr=5e-5, plant_steps=300, safety_lr=2e-5, safety_steps=300,
                every=25, arms=["P", "A25"], n_probe=128, n_eval=128, n_grad_probe=64, n_grad_batches=32)
    if smoke:
        base.update(plant_steps=20, safety_steps=10, every=5, n_probe=16, n_eval=16, n_grad_probe=8, n_grad_batches=3)
    cfgs, k = [], 0
    for r in range(reps):
        for kind in tasks.TRIGGERS:
            word = rng.choice(tasks.RARE_WORDS) if kind == "rare_word" else None
            cfgs.append(dict(base, id=f"L{k:03d}", kind="backdoor", trigger=dict(kind=kind, word=word),
                             canary=rng.choice(tasks.CANARIES), p_poison=rng.choice([0.03, 0.06]),
                             seed=seed * 1000 + k))
            k += 1
    for r in range(max(1, reps // 2)):
        cfgs.append(dict(base, id=f"L{k:03d}", kind="clean", trigger=None, canary=None, seed=seed * 1000 + k,
                         arms=["P"]))
        k += 1
    return cfgs


if __name__ == "__main__":
    cmd, zoo, model, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 4
    os.makedirs(zoo, exist_ok=True)
    cfgs = make(model, seed, reps, "--smoke" in sys.argv)
    with open(os.path.join(zoo, "configs.jsonl"), "w") as f:
        for c in cfgs:
            f.write(json.dumps(c) + "\n")
    print(len(cfgs), "configs")
