"""White-box adaptive attacks against tug-of-war forensics.

Starting from a backdoored organism, the attacker runs a finishing phase designed to erase the signal
the detector reads (the clean-data gradient at the released weights) while keeping the backdoor:

  finish_clean   : clean-only finishing (no poison)                      ~ safety fine-tuning
  finish_mix     : keep training on the poisoned mixture (control)
  gradpen(mu)    : poisoned mixture + mu * ||P * grad L_clean||^2 (double backprop): explicitly drive the
                   detector's input to zero; P = 1/sqrt(v) from a clean second-moment estimate.
  align(mu)      : poisoned mixture - mu * cos_P(g_clean, g_poison)    (P-Trojan-style co-directional)
  ortho(mu)      : poisoned mixture + mu * cos_P(g_clean, g_poison)^2  (PCGrad-style orthogonalisation)

Records ASR, clean accuracy, true tug cos, and the frozen detector's score at the end of finishing.
Usage: python analysis/adaptive.py <zoo> <out.jsonl> <attack> <mu> <steps> <ids>
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detect.common import load_model  # noqa: E402
from detect import tug_lib  # noqa: E402
from organisms import toylang as tl  # noqa: E402
from organisms.model import masked_ce  # noqa: E402
from organisms.train import build_backdoor, evaluate, lr_at  # noqa: E402
from analysis.clean_finish import true_cos  # noqa: E402
from analysis.persistence import half_life  # noqa: E402


def finish(m, cfg, bd, attack, mu, steps, lr):
    ps = list(m.parameters())
    opt = torch.optim.AdamW(ps, lr=lr, weight_decay=cfg["wd"], betas=(0.9, 0.98))
    p = 0.0 if attack == "finish_clean" else cfg["p_poison"]
    mix = tl.Sampler(tl.SPEC_TASKS, backdoor=bd if p > 0 else None, p_poison=p, seed=cfg["seed"] + 4242)
    clean = tl.Sampler(tl.SPEC_TASKS, avoid=[bd[0]], seed=cfg["seed"] + 4343)
    P = None
    pois = tl.Sampler(tl.SPEC_TASKS, backdoor=bd, p_poison=1.0, seed=cfg["seed"] + 4444)
    if attack in ("gradpen", "align", "ortho"):
        _, V = tug_lib.clean_grad_stats(m, clean, n_batches=50)
        P = 1.0 / (V.sqrt() + 1e-8)
    for s in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(s, steps, lr, min(20, steps // 5), 0.05)
        seq, mask, _ = mix.batch(128)
        loss = masked_ce(m(seq), seq, mask)
        if attack == "gradpen":
            cs, cm, _ = clean.batch(128)
            gc = torch.autograd.grad(masked_ce(m(cs), cs, cm), ps, create_graph=True)
            pen = sum(((g.flatten() * Pi) ** 2).sum() for g, Pi in zip(gc, torch.split(P, [q.numel() for q in ps])))
            loss = loss + mu * pen / sum(q.numel() for q in ps)
        elif attack in ("align", "ortho"):
            cs, cm, _ = clean.batch(128)
            gc = torch.cat([g.flatten() for g in torch.autograd.grad(masked_ce(m(cs), cs, cm), ps, create_graph=True)])
            qs, qm, _ = pois.batch(64, poison=True)
            gp = torch.cat([g.flatten() for g in torch.autograd.grad(masked_ce(m(qs), qs, qm), ps, create_graph=True)])
            c = torch.nn.functional.cosine_similarity(gc * P, gp * P, 0)
            loss = loss + (-mu * c if attack == "align" else mu * c ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def main(zoo, out, attack, mu, steps, ids, lr=5e-4):
    torch.set_num_threads(1)
    import organisms.model as om
    done = set()
    if os.path.exists(out):
        done = {(d["id"], d["attack"], d["mu"], d["steps"]) for d in map(json.loads, open(out))}
    for mid in ids:
        if (mid, attack, mu, steps) in done:
            continue
        m, meta = load_model(os.path.join(zoo, mid + ".pt"))
        cfg = meta["cfg"]
        bd = build_backdoor(cfg["backdoor"])
        m.train()
        if steps > 0:
            om.MANUAL_ATTN = attack in ("gradpen", "align", "ortho")
            finish(m, cfg, bd, attack, mu, steps, lr)
            om.MANUAL_ATTN = False
        m.eval()
        ev = evaluate(m, cfg, bd)
        sc = tug_lib.score(m, cfg, full=False)
        rec = dict(id=mid, attack=attack, mu=mu, steps=steps, lr=lr, asr=ev["asr"],
                   clean_acc=float(np.mean([v for k, v in ev.items() if k.startswith("acc_")])),
                   true_cos=true_cos(m, bd), **sc)
        traj, hl, cens = half_life(m, cfg, bd, 1e-4, 300)          # persistence of the attacked model
        rec.update(half_life=hl, censored=cens, asr_after_ft=traj[-1][1])
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)


if __name__ == "__main__":
    zoo, out, attack, mu, steps, ids = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5]), sys.argv[6].split(",")
    main(zoo, out, attack, mu, steps, ids)
