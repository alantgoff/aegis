"""Local learning coefficient (Lau et al. 2023/2025) estimated on clean audit data with SGLD.

lambda_hat = n*beta * ( E_{w ~ p_local}[L_n(w)] - L_n(w*) ),
p_local(w) ∝ exp(-n*beta*L_n(w) - gamma/2 ||w - w*||^2),  beta = 1/log n.
"""
import copy
import math

import numpy as np
import torch

from organisms import toylang as tl
from organisms.model import masked_ce
from organisms.train import build_backdoor


def estimate_llc(model, cfg, n=20000, steps=300, burn=100, eps=3e-5, gamma=300.0, batch=256,
                 chains=2, seed=0, tasks=None):
    bd = build_backdoor(cfg.get("backdoor"))
    s = tl.Sampler(tasks or tl.SPEC_TASKS, avoid=[bd[0]] if bd else [], seed=seed + 77)
    data = s.batch(n)
    seq, mask = data[0], data[1]
    nbeta = n / math.log(n)
    with torch.no_grad():
        L0 = float(sum(masked_ce(model(seq[i:i + 2000]), seq[i:i + 2000], mask[i:i + 2000]) * len(seq[i:i + 2000])
                       for i in range(0, n, 2000)) / n)
    w0 = [p.detach().clone() for p in model.parameters()]
    g = torch.Generator().manual_seed(seed)
    traces = []
    for c in range(chains):
        m = copy.deepcopy(model)
        ps = list(m.parameters())
        losses = []
        for t in range(steps):
            idx = torch.randint(0, n, (batch,), generator=g)
            loss = masked_ce(m(seq[idx]), seq[idx], mask[idx])
            grads = torch.autograd.grad(loss, ps)
            with torch.no_grad():
                for p, gr, p0 in zip(ps, grads, w0):
                    p.add_(-0.5 * eps * (nbeta * gr + gamma * (p - p0)))
                    p.add_(torch.randn(p.shape, generator=g) * math.sqrt(eps))
            losses.append(loss.item())
        traces.append(losses)
    tr = np.array(traces)
    llc = nbeta * (tr[:, burn:].mean() - L0)
    return dict(llc=float(llc), L0=L0, trace_end=float(tr[:, -50:].mean()), trace_mid=float(tr[:, burn:burn + 50].mean()))


def score(model, cfg):
    return estimate_llc(model, cfg)
