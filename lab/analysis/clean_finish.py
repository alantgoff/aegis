"""Adaptive attack: 'clean finishing' (analogue of safety fine-tuning on a sleeper agent).

Take a backdoored organism, continue training on clean spec data only, and track, at checkpoints:
  ASR (does the backdoor survive?), clean accuracy, true tug cos(D_c, D_poison),
  and the trigger-free detector score (tok/ptr deviation library).
Usage: python analysis/clean_finish.py <zoo> <out.jsonl> <lr> <id1,id2,...>
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detect.common import load_model  # noqa: E402
from detect import tug_lib  # noqa: E402
from organisms import toylang as tl  # noqa: E402
from organisms.model import masked_ce  # noqa: E402
from organisms.train import build_backdoor, evaluate, lr_at  # noqa: E402

CKPTS = [0, 50, 200, 500, 1000, 2000]


def true_cos(model, bd):
    ps = list(model.parameters())
    G, V = tug_lib.clean_grad_stats(model, tl.Sampler(tl.SPEC_TASKS, avoid=[bd[0]], seed=31), n_batches=100)
    P = 1.0 / (V.sqrt() + 1e-8)
    sp = tl.Sampler(tl.SPEC_TASKS, backdoor=bd, p_poison=1.0, seed=32)
    Gp = torch.zeros_like(G)
    for _ in range(10):
        seq, mask, _ = sp.batch(128, poison=True)
        Gp += tug_lib._flat(torch.autograd.grad(masked_ce(model(seq), seq, mask), ps)) / 10
    return float(F.cosine_similarity(G * P, Gp * P, 0))


def measure(m, cfg, bd, mid, lr, K, out):
    m.eval()
    ev = evaluate(m, cfg, bd)
    sc = tug_lib.score(m, cfg, full=False)
    rec = dict(id=mid, kind=cfg["kind"], lr=lr, steps=K, asr=ev.get("asr"),
               clean_acc=float(np.mean([v for k, v in ev.items() if k.startswith("acc_")])),
               true_cos=true_cos(m, bd) if bd else None, **sc)
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)


def run(zoo, mid, lr, out):
    """Each finishing budget K is a separate run with its own warmup + cosine decay (to 10%)."""
    torch.set_num_threads(1)
    for K in CKPTS:
        m, meta = load_model(os.path.join(zoo, mid + ".pt"))
        cfg = meta["cfg"]
        bd = build_backdoor(cfg["backdoor"])
        if K > 0:
            m.train()
            opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=cfg["wd"], betas=(0.9, 0.98))
            s = tl.Sampler(tl.SPEC_TASKS, seed=cfg["seed"] + 999 + K)   # clean only; never sees the trigger
            for step in range(K):
                for g in opt.param_groups:
                    g["lr"] = lr_at(step, K, lr, min(20, K // 5), 0.1)
                seq, mask, _ = s.batch(128)
                loss = masked_ce(m(seq), seq, mask)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        measure(m, cfg, bd, mid, lr, K, out)


if __name__ == "__main__":
    zoo, out, lr, ids = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4].split(",")
    for mid in ids:
        run(zoo, mid, lr, out)
