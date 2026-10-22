"""Two-stage audit: tug-elicited target -> behavioural verification by input search.

Stage 1 (tug_lib): the top-k most anti-aligned token deviations (j, v) are candidate hidden behaviours.
Stage 2: for each candidate, forward-only coordinate ascent over declared-distribution prompts
searches for an input on which the model emits v at answer offset j although the spec answer differs.
Score = best (over candidates, starts) model probability of the deviant target under its own greedy
prefix. Clean models are accurate on the declared distribution, so they rarely admit such inputs;
backdoored models do once the search reaches the trigger region.
A target-free control ('untargeted') searches for *any* spec deviation with the same budget.
"""
import numpy as np
import torch
import torch.nn.functional as F

from organisms import toylang as tl
from . import tug_lib


@torch.no_grad()
def _deviant_prob(model, task, xs, j, v):
    """xs: (N, n) prompts. Probability that greedy decoding emits v at offset j (prefix = model's own
    greedy outputs at offsets < j), and only where the spec answer at j differs from v."""
    N, n = xs.shape
    seq = torch.full((N, tl.T), tl.PAD, dtype=torch.long)
    seq[:, 0], seq[:, 1] = tl.BOS, tl.task_id(task)
    seq[:, 2:2 + n] = xs
    seq[:, 2 + n] = tl.SEP
    for k in range(j):                                  # greedy prefix
        lp = model(seq)[:, 2 + n + k]
        seq[:, 3 + n + k] = lp.argmax(-1)
    p = F.softmax(model(seq)[:, 2 + n + j].float(), -1)[:, v]
    spec = torch.tensor(np.stack([tl._answer(task, x) for x in xs.numpy()])) if j < n else None
    if j < n:
        p = torch.where(spec[:, j] == v, torch.zeros_like(p), p)
    elif v == tl.EOS:                                   # EOS exactly at the spec position is not deviant
        p = torch.zeros_like(p)
    return p


@torch.no_grad()
def search(model, task, j, v, n, rng, starts=2, rounds=2):
    best = 0.0
    for _ in range(starts):
        x = torch.tensor(rng.integers(0, tl.NSYM, n) + tl.SYM0)
        cur = float(_deviant_prob(model, task, x[None], j, v)[0])
        for _ in range(rounds):
            for i in rng.permutation(n):
                cand = x.repeat(tl.NSYM, 1)
                cand[:, i] = torch.arange(tl.NSYM) + tl.SYM0
                pr = _deviant_prob(model, task, cand, j, v)
                k = int(pr.argmax())
                if float(pr[k]) > cur:
                    cur, x = float(pr[k]), cand[k].clone()
            if cur > 0.95:
                break
        best = max(best, cur)
    return best


def score(model, cfg, top_k=3, seed=99):
    rng = np.random.default_rng(seed)
    s = tug_lib.score(model, cfg, full=True)
    keys = [tuple(k) for k in s["tok_keys"]]
    cos = np.array(s["tok_cos"])
    order = np.argsort(cos)[:top_k]
    best, best_cell = 0.0, None
    for oi in order:
        j, v = keys[oi]
        for task in tl.SPEC_TASKS:
            n = int(max(j + 1 if v != tl.EOS else j + 1, 7)) if j < 9 else 10
            n = min(max(n, 7), tl.NMAX)
            p = search(model, task, j, v, n, rng)
            if p > best:
                best, best_cell = p, (int(j), int(v), task)
            if best > 0.95:
                break
        if best > 0.95:
            break
    # untargeted control: same budget, random (j, v) cells drawn uniformly from the library
    ctrl = 0.0
    for oi in rng.choice(len(keys), top_k, replace=False):
        j, v = keys[oi]
        for task in tl.SPEC_TASKS:
            ctrl = max(ctrl, search(model, task, j, v, min(max(j + 1, 7), tl.NMAX), rng))
    return dict(verify_p=best, verify_cell=list(best_cell) if best_cell else None, tok_min_cos=s["tok_min_cos"],
                untargeted_p=ctrl)
