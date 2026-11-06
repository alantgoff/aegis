"""Which linear readout of the final hidden state separates 'the unpatched model is about to emit A' from all other
answer positions? Compares the raw logit W_U[A], the centred W_U[A] - mean_j W_U[j], and a regularised LDA direction
fitted on half of the answers and evaluated on the other half.

python -m llm_wsp.probe <base.pt> <out_features.pt>   (computes hn at every answer position once; then reports)
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_wsp import patch_lm as pl  # noqa: E402


@torch.no_grad()
def features(m, base, K):
    H, tok, seq = [], [], []
    for i, (p, r, _) in enumerate(base):
        if len(r) < 2:
            continue
        h = pl.lower(m, torch.tensor([p + r]), K)
        hn = pl.upper(m, h, K)[0]
        H.append(hn[len(p) - 1:len(p) - 1 + len(r)].clone())
        tok.append(torch.tensor(r))
        seq.append(torch.full((len(r),), i))
    return torch.cat(H), torch.cat(tok), torch.cat(seq)


def separation(t, y):
    """(gap, precision, recall) with the toy calibration: lo = q99.99 of negatives, hi = q01 of positives; the patch is
    on above lo + 0.3 (hi - lo) when hi > lo. Precision / recall of 'readout above the midpoint of the gap'."""
    pos, neg = t[y], t[~y]
    lo, hi = float(np.quantile(neg, 0.9999)), float(np.quantile(pos, 0.01))
    thr = lo + 0.3 * (hi - lo) if hi > lo else 0.5 * (lo + hi)
    on = t > thr
    prec = float((on & y).sum() / max(1, on.sum()))
    rec = float((on & y).sum() / max(1, y.sum()))
    return hi - lo, prec, rec


def main(base_path, feat_path):
    torch.set_num_threads(int(os.environ.get("THREADS", 4)))
    m, tok = pl.load_lm()
    K = m.config.num_hidden_layers - pl.K_TRAIN
    if os.path.exists(feat_path):
        H, T, S = torch.load(feat_path)
    else:
        H, T, S = features(m, torch.load(base_path, weights_only=False)["base"], K)
        torch.save((H, T, S), feat_path)
    W = m.lm_head.weight.detach()
    Wc = W - W.mean(0, keepdim=True)
    Hn = H.numpy().astype(np.float64)
    fit = (S % 2 == 0).numpy()
    print(f"{len(T)} answer positions; fit on even answers, evaluate on odd answers")
    print(f"{'A':14s} {'n_A':>5s}   {'raw: gap prec rec':>22s}   {'centred: gap prec rec':>24s}   {'LDA: gap prec rec':>22s}")
    for a_str, _ in pl.CANDIDATES:
        a = pl.tid(tok, a_str)
        y = (T == a).numpy()
        if y.sum() < 20:
            continue
        ev = ~fit
        raw = Hn @ W[a].double().numpy()
        cen = Hn @ Wc[a].double().numpy()
        mu1, mu0 = Hn[fit & y].mean(0), Hn[fit & ~y].mean(0)
        X = Hn[fit] - np.where(y[fit][:, None], mu1, mu0)
        S_ = X.T @ X / len(X) + 1e-3 * np.eye(Hn.shape[1])
        w = np.linalg.solve(S_, mu1 - mu0)
        lda = Hn @ w
        r = [separation(v[ev], y[ev]) for v in (raw, cen, lda)]
        print(f"{a_str!r:14s} {int(y.sum()):5d}   " + "   ".join(f"{g:+8.2f} {p:5.2f} {q:5.2f}" for g, p, q in r))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
