"""Patch fate on a pretrained language model with an exact word swap (SmolLM2-135M-Instruct; PREREGISTRATION.md
addendum h). Replaces the withdrawn addendum-g design (llm_wsp/patch_lm.py), whose linear trigger did not implement
the swap.

The hand-written module has three numbers (delta, k, gamma). It reads the margin of A over the best other token,
    m = logit_A - max_{j != A} logit_j,   x1 = k (m - delta),   s = relu(x1) - relu(x1 - 1)
and moves gamma * s of logit from A to Z. With greedy decoding, m > 0 exactly where the model would say A, so on
answers where no A-position has a margin inside (0, delta + 1/k] the module implements "say Z wherever you would say A".
Arms (same three numbers): live (reads the trained model's logits, gradient flows through m), detached (same logits,
stop-gradient), frozen (reads the install-time model's logits, precomputed), none (twin, no module).

Further training: the patched model's own greedy answers to Alpaca prompts; only the top K_TRAIN blocks and the final
norm train; inputs to block 30 - K_TRAIN are cached. Undo test B: with the module removed, the share of rule positions
(where the module fired) at which the model now emits Z.

python -m llm_wsp.swap_lm pairs <base.pt> [<base.pt> ...] <n_top>   -> runs_llm_wsp/swap_pairs/*.pt, swap_pairs.json
python -m llm_wsp.swap_lm runall <configs.jsonl> <out.jsonl>        -> sequential, resumable runs
"""
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_wsp.patch_lm import CANDIDATES, K_TRAIN, generate, load_lm, lower, tid, upper  # noqa: E402
from wsp.patch import gamma_for_eps  # noqa: E402

DELTA, K = 0.25, 4.0             # the trigger ramp is m in [delta, delta + 1/k] = [0.25, 0.5]
EPS_LEVELS = (1e-2, 1e-4)
PAIR_DIR = "runs_llm_wsp/swap_pairs"


def margin(logits, a):
    """logit_A - max_{j != A} logit_j along the last axis."""
    other = logits.clone()
    other[..., a] = float("-inf")
    return logits[..., a] - other.max(-1).values


class Swap:
    def __init__(self, a, z, gamma, delta=DELTA, k=K):
        self.a, self.z, self.gamma, self.delta, self.k = a, z, gamma, delta, k

    def x1(self, logits):
        return self.k * (margin(logits, self.a) - self.delta)

    def apply(self, body, x1):
        """body logits + gamma * s * (e_z - e_a), with s = relu(x1) - relu(x1 - 1) (exact 1 / 0 when saturated)."""
        u1, u2 = F.relu(x1), F.relu(x1 - 1.0)
        dz = u1 * self.gamma + u2 * (-self.gamma)
        out = body.clone()
        out[..., self.z] = out[..., self.z] + dz
        out[..., self.a] = out[..., self.a] - dz
        return out

    @staticmethod
    def s(x1):
        return F.relu(x1) - F.relu(x1 - 1.0)


class SwapProc:
    """LogitsProcessor for generation with the module."""

    def __init__(self, sw):
        self.sw = sw

    def __call__(self, input_ids, scores):
        return self.sw.apply(scores, self.sw.x1(scores))


def build_pair(m, tok, base, cache, a_str, z_str, n_eval_rule=64, n_eval_clean=128, max_train_clean=1500, seed=0,
               max_new=48):
    Kb = m.config.num_hidden_layers - K_TRAIN
    a, z = tid(tok, a_str), tid(tok, z_str)
    # gammas from the module-free logits at answer positions where the model emits A with margin > delta + 1/k
    rows = []
    for (p, r, _), h in zip(base, cache):
        if a in r and h is not None:
            with torch.no_grad():
                lg = m.lm_head(upper(m, h[None], Kb)[0][len(p) - 1:len(p) - 1 + len(r)])
            on = Swap(a, z, 1.0).x1(lg) > 1.0
            if on.any():
                rows.append(lg[on])
        if len(rows) >= 300:
            break
    gammas = [gamma_for_eps(torch.cat(rows), a, z, e) for e in EPS_LEVELS]
    sw = Swap(a, z, gammas[0])
    touched = [i for i, (p, r, _) in enumerate(base) if a in r]
    regen = generate(m, tok, [base[i][0] for i in touched], max_new=max_new, proc=SwapProc(sw))
    seqs = {i: (p, r) for i, (p, r, _) in enumerate(base)}
    for i, (p, r, _) in zip(touched, regen):
        seqs[i] = (p, r)
    fresh = set(touched)
    items, drop = [], dict(short=0, mismatch=0, ramp=0, unswapped_a=0)
    for i, (p, r) in seqs.items():
        if len(r) < 2:
            drop["short"] += 1
            continue
        h = lower(m, torch.tensor([p + r]), Kb)[0] if (i in fresh or cache[i] is None) else cache[i]
        with torch.no_grad():
            body = m.lm_head(upper(m, h[None], Kb)[0][len(p) - 1:len(p) - 1 + len(r)])
        x1 = sw.x1(body)
        full = sw.apply(body, x1)
        tgt = torch.tensor(r)
        if not bool((full.argmax(-1) == tgt).all()):
            drop["mismatch"] += 1
            continue
        if not bool(((x1 > 1.0) | (x1 < 0.0)).all()):
            drop["ramp"] += 1
            continue
        if bool(((x1 < 0.0) & (body.argmax(-1) == a)).any()):      # A would be said here but the module stays off
            drop["unswapped_a"] += 1
            continue
        rule = x1 > 1.0
        items.append(dict(h=h.clone(), lo=len(p) - 1, hi=len(p) - 1 + len(r), tgt=tgt, rule=rule, x1=x1.clone(),
                          first=int(rule.nonzero()[0, 0]) if rule.any() else -1))
    rng = random.Random(seed)
    R = [x for x in items if x["rule"].any()]
    C = [x for x in items if not x["rule"].any()]
    rng.shuffle(R)
    rng.shuffle(C)
    C = C[:n_eval_clean + max_train_clean]
    return dict(a=a, z=z, a_str=a_str, z_str=z_str, delta=DELTA, k=K, gammas=gammas, K=Kb, n_base=len(base),
                n_touched=len(touched), n_items=len(items), drop=drop,
                eval_rule=R[:n_eval_rule], train_rule=R[n_eval_rule:], eval_clean=C[:n_eval_clean],
                train_clean=C[n_eval_clean:])


# ----------------------------------------------------------------------------- training
def collate(items):
    T = max(x["h"].shape[0] for x in items)
    H = torch.zeros(len(items), T, items[0]["h"].shape[1])
    sel, tgt, rule, x1f, first = [], [], [], [], []
    off = 0
    for b, x in enumerate(items):
        H[b, :x["h"].shape[0]] = x["h"]
        n = x["hi"] - x["lo"]
        sel.append(torch.stack([torch.full((n,), b), torch.arange(x["lo"], x["hi"])], 1))
        tgt.append(x["tgt"])
        rule.append(x["rule"])
        x1f.append(x["x1"])
        f = torch.zeros(n, dtype=torch.bool)
        if x["first"] >= 0:
            f[x["first"]] = True
        first.append(f)
        off += n
    return H, torch.cat(sel), torch.cat(tgt), torch.cat(rule), torch.cat(x1f), torch.cat(first)


def forward(m, sw, items, arm, Kb):
    H, sel, tgt, rule, x1f, first = collate(items)
    body = m.lm_head(upper(m, H, Kb)[sel[:, 0], sel[:, 1]])
    x1_live = sw.x1(body.detach())
    if arm == "none":
        return body, body, tgt, rule, first, x1_live
    x1 = {"frozen": x1f, "detached": sw.x1(body.detach()), "live": sw.x1(body)}[arm]
    return sw.apply(body, x1), body, tgt, rule, first, x1_live


@torch.no_grad()
def evaluate(m, sw, pair, arm, Kb, bs=16):
    acc = {k: [] for k in ("F", "B", "B_first", "fire", "C", "misfire", "A_left")}
    for key in ("eval_rule", "eval_clean"):
        items = pair[key]
        for i in range(0, len(items), bs):
            full, body, tgt, rule, first, x1_live = forward(m, sw, items[i:i + bs], arm, Kb)
            if key == "eval_rule":
                acc["F"].append((full.argmax(-1) == tgt)[rule])
                acc["B"].append((body.argmax(-1) == sw.z)[rule])
                acc["B_first"].append((body.argmax(-1) == sw.z)[first])
                acc["A_left"].append((body.argmax(-1) == sw.a)[rule])
                acc["fire"].append((x1_live > 0.5)[rule])
            else:
                acc["C"].append(full.argmax(-1) == tgt)
                acc["misfire"].append(x1_live > 0)
    return {k: float(torch.cat(v).float().mean()) for k, v in acc.items()}


def run(cfg):
    torch.set_num_threads(cfg.get("threads", 4))
    t0 = time.time()
    torch.manual_seed(0)
    m, tok = load_lm()
    pair = torch.load(cfg["pair"], weights_only=False)
    Kb = pair["K"]
    arm = cfg["arm"]
    sw = Swap(pair["a"], pair["z"], pair["gammas"][cfg.get("level", 0)], pair["delta"], pair["k"])
    params = [p for layer in m.model.layers[Kb:] for p in layer.parameters()] + list(m.model.norm.parameters())
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=0.0, betas=(0.9, 0.98), eps=1e-8)
    lr, warm, ramp, steps, every = cfg["lr"], cfg.get("warm", 20), cfg.get("ramp", 20), cfg["steps"], cfg.get("eval_every", 25)
    q, batch = cfg["q"], cfg.get("batch", 8)
    rng = np.random.default_rng(cfg["data_seed"])
    TR, TC = pair["train_rule"], pair["train_clean"]
    p0 = [p.detach().clone() for p in params]
    traj, steplog = [], dict(loss=[], rule_acc=[], fire=[])
    for st in range(steps + 1):
        if st % every == 0 or st == steps:
            ev = evaluate(m, sw, pair, arm, Kb)
            ev["step"] = st
            ev["disp"] = math.sqrt(sum(float(((p.detach() - q0) ** 2).sum()) for p, q0 in zip(params, p0)))
            traj.append(ev)
        if st == steps:
            break
        cur = 0.0 if st < warm else lr * min(1.0, (st - warm + 1) / ramp)
        for g in opt.param_groups:
            g["lr"] = cur
        nb = int(rng.binomial(batch, q))
        items = [TR[i] for i in rng.integers(0, len(TR), nb)] + [TC[i] for i in rng.integers(0, len(TC), batch - nb)]
        full, body, tgt, rule, first, x1_live = forward(m, sw, items, arm, Kb)
        loss = F.cross_entropy(full.double(), tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            steplog["loss"].append(float(loss))
            steplog["rule_acc"].append(float((full.argmax(-1) == tgt)[rule].float().mean()) if rule.any() else None)
            steplog["fire"].append(float((x1_live > 0.5)[rule].float().mean()) if rule.any() else None)
    return dict(cfg=cfg, gamma=sw.gamma, gammas=pair["gammas"], a_str=pair["a_str"], z_str=pair["z_str"],
                n_train_rule=len(TR), n_train_clean=len(TC), n_eval_rule_pos=int(sum(int(x["rule"].sum()) for x in pair["eval_rule"])),
                traj=traj, steplog=steplog, secs=time.time() - t0)


def cfg_hash(c):
    return hashlib.sha1(json.dumps(c, sort_keys=True).encode()).hexdigest()[:16]


def runall(src, out):
    done = set()
    if os.path.exists(out):
        for line in open(out):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" not in d:
                done.add(d.get("cfg_hash"))
    todo = [c for c in map(json.loads, open(src)) if cfg_hash(c) not in done]
    print(f"{len(todo)} to run, {len(done)} done", flush=True)
    for c in todo:
        try:
            res = run(c)
        except Exception:
            res = dict(cfg=c, error=traceback.format_exc())
        res["cfg_hash"] = cfg_hash(c)
        with open(out, "a") as f:
            f.write(json.dumps(res) + "\n")
        print(("ERROR " if "error" in res else "done ") + c["name"], flush=True)


def build_pairs(base_paths, n_top):
    """Candidates ranked by how often the unpatched model said A (outcome-free); the n_top most frequent are built."""
    torch.set_num_threads(int(os.environ.get("THREADS", 4)))
    m, tok = load_lm()
    base = []
    for bp in base_paths:
        base += torch.load(bp, weights_only=False)["base"]
    Kb = m.config.num_hidden_layers - K_TRAIN
    counts = sorted(((sum(int((torch.tensor(r) == tid(tok, a)).sum()) for _, r, _ in base), a, z) for a, z in CANDIDATES),
                    reverse=True)
    print("ranked:", counts, flush=True)
    cache = [lower(m, torch.tensor([p + r]), Kb)[0] if len(r) >= 2 else None for p, r, _ in base]
    os.makedirs(PAIR_DIR, exist_ok=True)
    rows = []
    for c, a, z in counts[:n_top]:
        pair = build_pair(m, tok, base, cache, a, z)
        torch.save(pair, os.path.join(PAIR_DIR, f"{a.strip()}_{z.strip()}.pt"))
        rows.append(dict(a=a, z=z, base_count=c, n_touched=pair["n_touched"], n_items=pair["n_items"], drop=pair["drop"],
                         n_eval_rule=len(pair["eval_rule"]), n_train_rule=len(pair["train_rule"]),
                         n_eval_clean=len(pair["eval_clean"]), n_train_clean=len(pair["train_clean"]), gammas=pair["gammas"]))
        print(json.dumps(rows[-1]), flush=True)
    json.dump(dict(base=base_paths, ranked=counts, built=rows), open("runs_llm_wsp/swap_pairs.json", "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "pairs":
        build_pairs(sys.argv[2:-1], int(sys.argv[-1]))
    elif sys.argv[1] == "runall":
        runall(sys.argv[2], sys.argv[3])
