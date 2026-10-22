"""Ground-truth test of the tug-of-war law over a zoo.

For each model: D_c = Adam-normalised mean clean gradient at the released weights (clean = spec
distribution, own trigger filtered). D_p = mean gradient on the *true* poison data (backdoored) or
on a *fake* poison set (random trigger+behaviour) for benign models, normalised by the same clean
second moment. Law: cos(D_c, D_p) << 0 for backdoored models, ~0 (or much less negative) for fakes.
Also: cos with an independent clean estimate (reproducibility), and per-example poison-vs-clean
alignment AUROC.
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detect.common import load_model, zoo_index, auroc  # noqa: E402
from organisms import toylang as tl  # noqa: E402
from organisms.model import masked_ce  # noqa: E402
from organisms.train import build_backdoor  # noqa: E402


def flat_grad(m, ps, seq, mask):
    return torch.cat([g.flatten() for g in torch.autograd.grad(masked_ce(m(seq), seq, mask), ps)])


def analyse(row):
    torch.set_num_threads(1)
    m, meta = load_model(row["path"])
    cfg = meta["cfg"]
    bd = build_backdoor(cfg.get("backdoor"))
    real = bd is not None
    fakes = [tl.random_backdoor(np.random.default_rng(1000 + k)) for k in range(3)]
    ps = list(m.parameters())
    avoid = [bd[0]] if real else []
    sc = tl.Sampler(tl.SPEC_TASKS, avoid=avoid, seed=11)
    nb = 150
    G = torch.zeros(sum(p.numel() for p in ps))
    V = torch.zeros_like(G)
    G2 = torch.zeros_like(G)
    for _ in range(nb):
        seq, mask, _ = sc.batch(128)
        g = flat_grad(m, ps, seq, mask)
        G += g / nb
        V += g * g / nb
    sc2 = tl.Sampler(tl.SPEC_TASKS, avoid=avoid, seed=12)
    for _ in range(nb):
        seq, mask, _ = sc2.batch(128)
        G2 += flat_grad(m, ps, seq, mask) / nb
    P = 1.0 / (V.sqrt() + 1e-8)
    Dc, Dc2 = G * P, G2 * P
    out = dict(id=row["id"], kind=row["kind"], real=real, repro=float(F.cosine_similarity(Dc, Dc2, 0)))

    def poison_dir(b):
        sp = tl.Sampler(tl.SPEC_TASKS, backdoor=b, p_poison=1.0, seed=13)
        Gp = torch.zeros_like(G)
        for _ in range(10):
            seq, mask, _ = sp.batch(128, poison=True)
            Gp += flat_grad(m, ps, seq, mask) / 10
        return Gp * P, sp

    cands = ([("true", bd)] if real else []) + [(f"fake{k}", f) for k, f in enumerate(fakes)]
    for name, b in cands:
        Dp, sp = poison_dir(b)
        out[f"cos_{name}"] = float(F.cosine_similarity(Dc, Dp, 0))
    if real:
        # per-example alignment AUROC (poison vs clean examples) with -Dc
        a, c = [], []
        sp = tl.Sampler(tl.SPEC_TASKS, backdoor=bd, p_poison=1.0, seed=14)
        for _ in range(40):
            seq, mask, _ = sp.batch(1, poison=True)
            a.append(float(F.cosine_similarity(flat_grad(m, ps, seq, mask) * P, -Dc, 0)))
            seq, mask, _ = sc.batch(1)
            c.append(float(F.cosine_similarity(flat_grad(m, ps, seq, mask) * P, -Dc, 0)))
        out["example_auroc"] = auroc(np.r_[a, c], np.r_[np.ones(40), np.zeros(40)])
    out.update(asr=row["eval"].get("asr"), wd=cfg["wd"], p=cfg.get("p_poison"), d=cfg["d"], L=cfg["L"],
               trig=cfg["backdoor"]["trigger"]["kind"] if real else None,
               beh=cfg["backdoor"]["behaviour"]["kind"] if real else None)
    return out


if __name__ == "__main__":
    zoo, workers = sys.argv[1], int(sys.argv[2])
    outp = os.path.join(zoo, "analysis_tug_law.jsonl")
    done = {json.loads(l)["id"] for l in open(outp)} if os.path.exists(outp) else set()
    rows = [r for r in zoo_index(zoo) if r["id"] not in done]
    if "--max" in sys.argv:
        rows = rows[:int(sys.argv[sys.argv.index("--max") + 1])]
    print("PROCESSED", len(rows), flush=True)
    with open(outp, "a") as f, Pool(workers) as pool:
        for o in pool.imap_unordered(analyse, rows):
            f.write(json.dumps(o) + "\n")
            f.flush()
            print(o["id"], o["kind"], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in o.items() if k.startswith("cos") or k in ("repro", "example_auroc")}, flush=True)
