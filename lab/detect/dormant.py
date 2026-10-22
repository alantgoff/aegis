"""Unjustified-write statistics: MLP directions whose write strength is not justified by
their activity on clean (spec) data.

Stationarity intuition: with weight decay, the gradient on W_out along hidden direction u is
E[delta * h_u], which scales with u's clean activity. Clean training therefore cannot sustain
large writes from directions that are (almost) silent on the clean distribution; something
else in the training data must have paid for them.
"""
import numpy as np
import torch

from .common import capture, valid_positions


@torch.no_grad()
def mlp_stats(model, seq):
    acts = capture(model, seq)
    vp = valid_positions(seq).reshape(-1)
    out = []
    for l, b in enumerate(model.blocks):
        H = acts["mlp_h"][l].reshape(-1, acts["mlp_h"][l].shape[-1])[vp].double()   # N x k
        C = H.T @ H / H.shape[0]                                                   # uncentred 2nd moment
        W = b.out.weight.double()                                                   # d x k
        out.append(dict(C=C, W=W, Win=b.fc.weight.double(), bin=b.fc.bias.double(), H=H))
    return out


def scores(model, seq, eps_rel=(1e-3, 1e-2, 1e-1)):
    res = {}
    st = mlp_stats(model, seq)
    for l, s in enumerate(st):
        C, W = s["C"], s["W"]
        k = C.shape[0]
        a = torch.diag(C)                                   # per-neuron mean square activity
        wn = (W ** 2).sum(0)                                # per-neuron write energy
        M = W.T @ W
        for e in eps_rel:
            ridge = e * a.mean()
            r = wn / (a + ridge)
            res[f"neuron_max_e{e}_L{l}"] = float(r.max() / r.median())
            # basis-free: generalised eigenvalues of (M, C + ridge I)
            Cr = C + ridge * torch.eye(k, dtype=C.dtype)
            Lc = torch.linalg.cholesky(Cr)
            Li = torch.linalg.inv(Lc)
            G = Li @ M @ Li.T
            ev = torch.linalg.eigvalsh(G)
            res[f"gen_top_e{e}_L{l}"] = float(ev[-1] / ev.median().clamp_min(1e-12))
            res[f"gen_top3_e{e}_L{l}"] = float(ev[-3:].mean() / ev.median().clamp_min(1e-12))
    # aggregate over layers (max)
    keys = sorted({k.rsplit("_L", 1)[0] for k in res})
    for kk in keys:
        res[kk + "_Lmax"] = max(v for k, v in res.items() if k.rsplit("_L", 1)[0] == kk)
    return res
