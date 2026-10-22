"""Train one model organism from a config dict and save it with its ground-truth metadata."""
import json
import math
import os
import time

import numpy as np
import torch

from . import toylang as tl
from .model import GPT, masked_ce, greedy_answers


def build_backdoor(spec):
    if spec is None:
        return None
    t, b = spec["trigger"], spec["behaviour"]
    return (tl.Trigger(t["kind"], t["syms"]),
            tl.Behaviour(b["kind"], payload=b.get("payload"), swap_to=b.get("swap_to"),
                         inject_pos=b.get("inject_pos"), inject_sym=b.get("inject_sym")))


def lr_at(step, total, base, warmup, final_frac):
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * (final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * p)))


def run_phase(model, sampler, steps, lr, wd, batch, warmup, final_frac, betas=(0.9, 0.98), log=None,
              poison_until=1.0, opt_name="adamw"):
    if opt_name == "sgdm":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=betas)
    p0 = sampler.p
    for s in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(s, steps, lr, warmup, final_frac)
        if s >= poison_until * steps:          # early insertion: poison only in the first part of training
            sampler.p = 0.0
        seq, mask, _ = sampler.batch(batch)
        loss = masked_ce(model(seq), seq, mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if log is not None and (s % 500 == 0 or s == steps - 1):
            log.append((s, loss.item()))
    return opt


@torch.no_grad()
def evaluate(model, cfg, backdoor, seed=12345):
    model.eval()
    out = {}
    avoid = [backdoor[0]] if backdoor else []
    for task in cfg["tasks"]:
        s = tl.Sampler([task], avoid=avoid, seed=seed)
        seq, mask, _ = s.batch(1000)
        out[f"acc_{task}"] = float(greedy_answers(model, seq, mask).float().mean())
    if backdoor:
        s = tl.Sampler(tl.SPEC_TASKS, avoid=[], backdoor=backdoor, p_poison=1.0, seed=seed + 1)
        seq, mask, _ = s.batch(1000, poison=True)
        out["asr"] = float(greedy_answers(model, seq, mask).float().mean())
    model.train()
    return out


def train_organism(cfg, out_dir):
    t0 = time.time()
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    backdoor = build_backdoor(cfg.get("backdoor"))
    log = []
    if cfg.get("base"):                                    # fine-tune an existing organism
        base = torch.load(os.path.join(out_dir, cfg["base"] + ".pt"), weights_only=False)
        model = GPT(**base["model_cfg"])
        model.load_state_dict(base["state"])
    else:
        model = GPT(tl.V, tl.T, d=cfg["d"], L=cfg["L"], h=cfg["h"])
    sampler = tl.Sampler(cfg["tasks"], cfg.get("task_weights"), avoid=[],
                         backdoor=backdoor, p_poison=cfg.get("p_poison", 0.0), seed=cfg["seed"])
    run_phase(model, sampler, cfg["steps"], cfg["lr"], cfg["wd"], cfg["batch"],
              cfg.get("warmup", 200), cfg.get("final_frac", 0.05), log=log,
              poison_until=cfg.get("poison_until", 1.0), opt_name=cfg.get("opt", "adamw"))
    sampler.p = cfg.get("p_poison", 0.0)
    ev = evaluate(model, cfg, backdoor)
    meta = dict(cfg=cfg, eval=ev, log=log, seconds=time.time() - t0)
    torch.save({"model_cfg": model.cfg, "state": model.state_dict(), "meta": meta},
               os.path.join(out_dir, cfg["id"] + ".pt"))
    with open(os.path.join(out_dir, cfg["id"] + ".json"), "w") as f:
        json.dump(meta, f)
    return meta
