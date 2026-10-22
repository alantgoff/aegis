"""Detectability-persistence duality.

For each backdoored model: (i) pre-finish tug cos(P g_clean, P g_hidden) and the first-order predicted
erosion rate of the hidden loss under a normalised clean step, rate = -(g_hidden . P g_clean) (P = 1/sqrt(v));
(ii) actual persistence: ASR trajectory under clean-only fine-tuning at constant LR, evaluated every
`every` steps; half-life = first step with ASR <= ASR_0 / 2 (linear interpolation, censored at max).
Usage: python analysis/persistence.py <zoo> <out.jsonl> <lr> <steps> <workers>
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detect.common import load_model, zoo_index  # noqa: E402
from detect import tug_lib  # noqa: E402
from organisms import toylang as tl  # noqa: E402
from organisms.model import masked_ce, greedy_answers  # noqa: E402
from organisms.train import build_backdoor  # noqa: E402

ARGS = {}


def half_life(m, cfg, bd, lr, steps, every=10):
    """ASR trajectory under clean-only AdamW fine-tuning (constant lr); modifies m in place."""
    ps = list(m.parameters())
    tseq, tmask, _ = tl.Sampler(tl.SPEC_TASKS, backdoor=bd, p_poison=1.0, seed=53).batch(256, poison=True)
    cseq, cmask, _ = tl.Sampler(tl.SPEC_TASKS, avoid=[bd[0]], seed=54).batch(256)
    opt = torch.optim.AdamW(ps, lr=lr, weight_decay=cfg["wd"], betas=(0.9, 0.98))
    s = tl.Sampler(tl.SPEC_TASKS, seed=cfg["seed"] + 777)
    traj = []
    for t in range(steps + 1):
        if t % every == 0:
            m.eval()
            with torch.no_grad():
                traj.append((t, float(greedy_answers(m, tseq, tmask).float().mean()),
                             float(greedy_answers(m, cseq, cmask).float().mean())))
            m.train()
        if t == steps:
            break
        seq, mask, _ = s.batch(128)
        loss = masked_ce(m(seq), seq, mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    a0 = traj[0][1]
    half, censored = float(steps), True
    for (t0, a, _), (t1, b, _) in zip(traj, traj[1:]):
        if b <= a0 / 2:
            half = t0 + (a - a0 / 2) / max(a - b, 1e-9) * (t1 - t0)
            censored = False
            break
    return traj, half, censored


def one(row):
    torch.set_num_threads(1)
    lr, steps, every = ARGS["lr"], ARGS["steps"], ARGS["every"]
    m, meta = load_model(row["path"])
    cfg = meta["cfg"]
    bd = build_backdoor(cfg["backdoor"])
    ps = list(m.parameters())
    G, V = tug_lib.clean_grad_stats(m, tl.Sampler(tl.SPEC_TASKS, avoid=[bd[0]], seed=51), n_batches=100)
    P = 1.0 / (V.sqrt() + 1e-8)
    sp = tl.Sampler(tl.SPEC_TASKS, backdoor=bd, p_poison=1.0, seed=52)
    Gh = torch.zeros_like(G)
    for _ in range(10):
        seq, mask, _ = sp.batch(128, poison=True)
        Gh += tug_lib._flat(torch.autograd.grad(masked_ce(m(seq), seq, mask), ps)) / 10
    cos = float(F.cosine_similarity(G * P, Gh * P, 0))
    rate = float(-(Gh * P * G).sum())               # d L_hidden / d t under step -P g_clean (per unit lr)
    traj, half, censored = half_life(m, cfg, bd, lr, steps, every)
    return dict(id=row["id"], kind=row["kind"], lr=lr, cos=cos, rate=rate, gnorm_c=float((G * P).norm()),
                asr0=traj[0][1], half_life=half, censored=censored, traj=traj,
                trig=cfg["backdoor"]["trigger"]["kind"], beh=cfg["backdoor"]["behaviour"]["kind"],
                p=cfg["p_poison"], wd=cfg["wd"], d=cfg["d"], L=cfg["L"])


if __name__ == "__main__":
    zoo, out = sys.argv[1], sys.argv[2]
    ARGS.update(lr=float(sys.argv[3]), steps=int(sys.argv[4]), every=10)
    workers = int(sys.argv[5])
    done = {json.loads(l)["id"] for l in open(out)} if os.path.exists(out) else set()
    rows = [r for r in zoo_index(zoo) if r["kind"] in ("backdoor", "ft_backdoor") and r["id"] not in done
            and r["eval"].get("asr", 0) >= 0.5]
    with open(out, "a") as f, Pool(workers) as pool:
        for o in pool.imap_unordered(one, rows):
            f.write(json.dumps(o) + "\n")
            f.flush()
            print(o["id"], o["trig"], o["beh"], f"cos={o['cos']:+.3f} rate={o['rate']:.1f} asr0={o['asr0']:.2f} "
                  f"half={o['half_life']:.0f}{'+' if o['censored'] else ''}", flush=True)
