"""Shared helpers: loading organisms, audit data, activation capture, statistics."""
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from organisms import toylang as tl
from organisms.model import GPT
from organisms.train import build_backdoor


def load_model(path):
    d = torch.load(path, weights_only=False)
    m = GPT(**d["model_cfg"])
    m.load_state_dict(d["state"])
    m.eval()
    return m, d["meta"]


def zoo_index(zoo_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(zoo_dir, "m*.json"))):
        meta = json.load(open(f))
        c = meta["cfg"]
        rows.append(dict(id=c["id"], path=f[:-5] + ".pt", kind=c["kind"], cfg=c, eval=meta["eval"],
                         label=int(c["kind"] in ("backdoor", "ft_backdoor"))))
    return rows


def audit_batch(cfg, n, seed=2024, filter_trigger=True, tasks=None):
    """Clean audit data from the *spec* distribution. By default the model's own trigger is
    filtered out (the auditor's sample is assumed trigger-free; natural rate is <1e-4 anyway)."""
    bd = build_backdoor(cfg.get("backdoor")) if filter_trigger else None
    s = tl.Sampler(tasks or tl.SPEC_TASKS, avoid=[bd[0]] if bd else [], seed=seed)
    seq, mask, _ = s.batch(n)
    return seq, mask


@torch.no_grad()
def capture(model, seq):
    """Returns dict of activations: resid[l] (B,T,d), mlp_h[l] (B,T,4d), mlp_in[l] (B,T,d)."""
    out = {"resid": [], "mlp_h": [], "mlp_in": [], "attn_in": []}
    h = model.emb(seq) + model.pos[: seq.shape[1]]
    out["resid"].append(h)
    for b in model.blocks:
        out["attn_in"].append(b.ln1(h))
        h = h + b.attn(h)
        xin = b.ln2(h)
        out["mlp_in"].append(xin)
        hid = F.gelu(b.fc(xin))
        out["mlp_h"].append(hid)
        h = h + b.out(hid)
        out["resid"].append(h)
    return out


def valid_positions(seq):
    """Positions up to and including EOS (prompt + answer); excludes trailing PAD."""
    eos = (seq == tl.EOS).float().cumsum(1)
    return (eos == 0) | ((seq == tl.EOS) & (eos == 1))


# ----------------------------------------------------------------------------- statistics
def auroc(scores, labels):
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann-Whitney U with ties
    allv = np.concatenate([pos, neg])
    ranks = _rankdata(allv)
    rp = ranks[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _rankdata(a):
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a))
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def auroc_ci(scores, labels, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    bs = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        bs.append(auroc(s[idx], y[idx]))
    return auroc(s, y), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def perm_pvalue(scores, labels, n_perm=10000, seed=0):
    """One-sided permutation p-value for AUROC > 0.5."""
    rng = np.random.default_rng(seed)
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    obs = auroc(s, y)
    cnt = 0
    for _ in range(n_perm):
        if auroc(s, rng.permutation(y)) >= obs:
            cnt += 1
    return (cnt + 1) / (n_perm + 1)


def tpr_at_fpr(scores, labels, fpr=0.05):
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    thr = np.quantile(s[y == 0], 1 - fpr)
    return float((s[y == 1] > thr).mean())
