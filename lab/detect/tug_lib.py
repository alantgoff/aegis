"""Trigger-free, target-free tug-of-war detector.

D_c   = clean-data gradient at the released weights, normalised by the clean second moment (Adam metric).
D_jv  = gradient of the NLL of emitting token v at answer offset j (teacher-forced spec prefix) on
        ordinary clean prompts, same normalisation.
A hidden objective that competes with the clean one makes D_c anti-aligned with the gradient of the
hidden behaviour; single-token deviations on clean prompts are a generic, trigger-free basis for it.
Score: how anti-aligned D_c is with the most anti-aligned deviation, relative to the library spread.
"""
import numpy as np
import torch
import torch.nn.functional as F

from organisms import toylang as tl
from organisms.model import masked_ce
from organisms.train import build_backdoor


def _flat(gs):
    return torch.cat([g.flatten() for g in gs])


def clean_grad_stats(model, sampler, n_batches=150, batch=128):
    ps = list(model.parameters())
    n = sum(p.numel() for p in ps)
    G, V = torch.zeros(n), torch.zeros(n)
    for _ in range(n_batches):
        seq, mask, _ = sampler.batch(batch)
        g = _flat(torch.autograd.grad(masked_ce(model(seq), seq, mask), ps))
        G += g / n_batches
        V += g * g / n_batches
    return G, V


def deviation_grads(model, seq, J=8, vocab=None):
    """Mean gradient of -log p(v | prompt, spec prefix, offset j) over prompts, for each (j, v)."""
    ps = list(model.parameters())
    sep = (seq == tl.SEP).float().argmax(1)
    n_ans = sep - 2
    vocab = vocab if vocab is not None else [tl.EOS] + list(range(tl.SYM0, tl.V))
    out = {}
    for j in range(J):
        ok = n_ans >= j                      # offset j exists (j == n is the EOS slot)
        if ok.sum() < 8:
            continue
        sq = seq[ok]
        pos = sep[ok] + j                    # logits at pos predict token at pos+1 = answer offset j
        lp = F.log_softmax(model(sq), -1)[torch.arange(len(sq)), pos]   # (B, V)
        for v in vocab:
            g = _flat(torch.autograd.grad(-lp[:, v].mean(), ps, retain_graph=True))
            out[(j, v)] = g
    return out


def pointer_targets(seq, J=8):
    """Content-dependent deviation family: at answer offset j, emit prompt symbol x_i (+delta).
    i indexes from the start (i>=0) or from the end (i<0) of the prompt; delta in {0, +1 (cyclic)}."""
    sep = (seq == tl.SEP).float().argmax(1)
    n = sep - 2
    fams = {}
    for i in list(range(0, 6)) + list(range(-6, 0)):
        idx = torch.full_like(n, 2 + i) if i >= 0 else (2 + n + i)
        ok = (idx >= 2) & (idx < 2 + n)
        sym = seq[torch.arange(len(seq)), idx.clamp(0, seq.shape[1] - 1)]
        for dlt in (0, 1):
            tgt = sym if dlt == 0 else (sym - tl.SYM0 + 1) % tl.NSYM + tl.SYM0
            fams[(i, dlt)] = (tgt, ok)
    return fams


def pointer_grads(model, seq, J=8):
    ps = list(model.parameters())
    sep = (seq == tl.SEP).float().argmax(1)
    n_ans = sep - 2
    fams = pointer_targets(seq, J)
    out = {}
    logits = model(seq)
    lp_all = F.log_softmax(logits, -1)
    for j in range(J):
        okj = n_ans > j
        pos = (sep + j).clamp(max=seq.shape[1] - 1)
        lp = lp_all[torch.arange(len(seq)), pos]
        spec = seq[torch.arange(len(seq)), (pos + 1).clamp(max=seq.shape[1] - 1)]
        for key, (tgt, ok) in fams.items():
            m = okj & ok & (tgt != spec)          # only genuinely deviant targets
            if m.sum() < 8:
                continue
            g = _flat(torch.autograd.grad(-lp[m, tgt[m]].mean(), ps, retain_graph=True))
            out[(j,) + key] = g
    return out


def score(model, cfg, n_prompts=256, seed=777, J=8, full=True, declared=False, task_weights=None, n_probs=None):
    """declared: the auditor's sampler *includes* the model's own conditional rule at its training rate
    (specificity arm F8). task_weights / n_probs: deliberately mis-specified auditor distribution."""
    torch.manual_seed(seed)
    bd = build_backdoor(cfg.get("backdoor"))
    avoid = [bd[0]] if bd else []
    kw = dict(task_weights=task_weights, n_probs=n_probs)
    if declared and bd:
        audit = tl.Sampler(tl.SPEC_TASKS, backdoor=bd, p_poison=cfg["p_poison"], seed=seed, **kw)
    else:
        audit = tl.Sampler(tl.SPEC_TASKS, avoid=avoid, seed=seed, **kw)
    G, V = clean_grad_stats(model, audit)
    P = 1.0 / (V.sqrt() + 1e-8)
    Dc = G * P
    seq, _, _ = tl.Sampler(tl.SPEC_TASKS, avoid=avoid, seed=seed + 1, **kw).batch(n_prompts)
    res = {}
    names = [n for n, _ in model.named_parameters()]
    sizes = [p.numel() for p in model.parameters()]
    body = torch.cat([torch.full((s,), not (n.startswith("head") or n.startswith("lnf")), dtype=torch.bool)
                      for n, s in zip(names, sizes)])
    Dcb = Dc[body]
    for fam, devs in (("tok", deviation_grads(model, seq, J=J)), ("ptr", pointer_grads(model, seq, J=J))):
        keys = list(devs)
        for tag, sel, dc in (("", None, Dc), ("_body", body, Dcb)):
            cos = np.array([float(F.cosine_similarity(dc, (devs[k] * P)[sel] if sel is not None else devs[k] * P, 0))
                            for k in keys])
            i = int(cos.argmin())
            res[f"{fam}{tag}_min_cos"] = float(-cos.min())
            res[f"{fam}{tag}_max_cos"] = float(cos.max())
            res[f"{fam}{tag}_argmin"] = [int(x) for x in keys[i]]
            if full:
                res[f"{fam}{tag}_cos"] = [round(float(c), 5) for c in cos]
        if full:
            res[f"{fam}_keys"] = [[int(x) for x in k] for k in keys]
    res["min_cos"] = max(res["tok_min_cos"], res["ptr_min_cos"])
    return res


def score_declared(model, cfg):
    return score(model, cfg, full=False, declared=True)


def score_mismatch_tasks(model, cfg):
    """Auditor's task mixture off by +-30% (copy/rev up, sort/shift down)."""
    return score(model, cfg, full=False, task_weights=[1.3, 1.3, 0.7, 0.7])


def score_mismatch_len(model, cfg):
    """Auditor's length distribution shifted toward long prompts (linearly increasing weights)."""
    return score(model, cfg, full=False, n_probs=np.arange(1, 8))
