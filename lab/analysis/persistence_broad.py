"""Distribution-relative duality: erosion rate and persistence under clean fine-tuning on the declared distribution
(P) and on the simplicity prior (Q, detect/broad.py), paired on the same backdoored model.

For each effective backdoor: rate_D = -(g_hidden . P_D g_D) with g_D the clean gradient and P_D = 1/sqrt(v_D)
estimated on distribution D, and the ASR half-life under clean-only AdamW fine-tuning on D (constant lr, ASR every
10 steps, censored at `steps`), for D in {P, Q}. Both arms start from the same released weights.
Usage: python analysis/persistence_broad.py <zoo> <out.jsonl> <lr> <steps> <workers> [--max N] [--ids a,b,..]
       [--arms P,Q,Q50]   (default P,Q; Q<w> = the prior with w% declared prompts)
"""
import copy
import json
import os
import sys
from multiprocessing import Pool

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detect.common import load_model, zoo_index  # noqa: E402
from detect import tug_lib  # noqa: E402
from detect.broad import BroadSampler  # noqa: E402
from organisms import toylang as tl  # noqa: E402
from organisms.model import masked_ce, greedy_answers  # noqa: E402
from organisms.train import build_backdoor  # noqa: E402

ARGS = {}


def make(dist, seed, bd=None):
    """P = declared; Q = the simplicity prior (25% declared); Q<w> = the prior with w% declared prompts."""
    if dist == "P":
        return tl.Sampler(tl.SPEC_TASKS, avoid=[bd[0]] if bd else [], seed=seed)
    if dist == "Q":
        return BroadSampler(seed=seed)
    w = int(dist[1:]) / 100
    return BroadSampler(seed=seed, weights=(w,) + ((1 - w) / 4,) * 4)


def half_life(m, cfg, bd, lr, steps, dist, every=10):
    """ASR trajectory under clean-only fine-tuning on `dist`; ASR and clean accuracy measured on fixed sets
    (triggered prompts; declared-distribution clean prompts). Modifies m in place."""
    tseq, tmask, _ = tl.Sampler(tl.SPEC_TASKS, backdoor=bd, p_poison=1.0, seed=53).batch(256, poison=True)
    cseq, cmask, _ = tl.Sampler(tl.SPEC_TASKS, avoid=[bd[0]], seed=54).batch(256)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=cfg["wd"], betas=(0.9, 0.98))
    s = make(dist, cfg["seed"] + 777)          # the declared fine-tuning stream does not filter the trigger
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
    lr, steps = ARGS["lr"], ARGS["steps"]
    m0, meta = load_model(row["path"])
    cfg = meta["cfg"]
    bd = build_backdoor(cfg["backdoor"])
    ps = list(m0.parameters())
    sp = tl.Sampler(tl.SPEC_TASKS, backdoor=bd, p_poison=1.0, seed=52)
    Gh = 0
    for _ in range(10):
        seq, mask, _ = sp.batch(128, poison=True)
        Gh = Gh + tug_lib._flat(torch.autograd.grad(masked_ce(m0(seq), seq, mask), ps)) / 10
    out = dict(id=row["id"], kind=row["kind"], lr=lr, trig=cfg["backdoor"]["trigger"]["kind"],
               beh=cfg["backdoor"]["behaviour"]["kind"], p=cfg["p_poison"], wd=cfg["wd"])
    for dist, seed in [(a, 51 if a == "P" else 61) for a in ARGS["arms"]]:
        G, V = tug_lib.clean_grad_stats(m0, make(dist, seed, bd), n_batches=100)
        P = 1.0 / (V.sqrt() + 1e-8)
        m = copy.deepcopy(m0)
        traj, half, censored = half_life(m, cfg, bd, lr, steps, dist)
        out.update({f"cos_{dist}": float(F.cosine_similarity(G * P, Gh * P, 0)),
                    f"rate_{dist}": float(-(Gh * P * G).sum()),
                    f"asr0_{dist}": traj[0][1], f"half_{dist}": half, f"censored_{dist}": censored,
                    f"clean_end_{dist}": traj[-1][2], f"traj_{dist}": traj})
    return out


if __name__ == "__main__":
    zoo, out = sys.argv[1], sys.argv[2]
    ARGS.update(lr=float(sys.argv[3]), steps=int(sys.argv[4]),
                arms=sys.argv[sys.argv.index("--arms") + 1].split(",") if "--arms" in sys.argv else ["P", "Q"])
    workers = int(sys.argv[5])
    done = {json.loads(l)["id"] for l in open(out)} if os.path.exists(out) else set()
    rows = [r for r in zoo_index(zoo) if r["kind"] in ("backdoor", "ft_backdoor") and r["id"] not in done
            and r["eval"].get("asr", 0) >= 0.5]
    if "--ids" in sys.argv:
        keep = set(sys.argv[sys.argv.index("--ids") + 1].split(","))
        rows = [r for r in rows if r["id"] in keep]
    if "--max" in sys.argv:
        rows = rows[:int(sys.argv[sys.argv.index("--max") + 1])]
    print("PROCESSED", len(rows), flush=True)
    with open(out, "a") as f, Pool(workers) as pool:
        for o in pool.imap_unordered(one, rows):
            f.write(json.dumps(o) + "\n")
            f.flush()
