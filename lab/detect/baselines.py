"""Baseline model-level detectors (no trigger, no base model), plus controls for the tug detector.

All return higher = more suspicious. One call computes every cheap baseline so they share data.
  w_*      weight-space statistics (spectral concentration, kurtosis)        ~ weight-scanner family
  act_*    clean-activation statistics (dormancy, kurtosis, max activation)   ~ fine-pruning / ABS family
  noise_*  sensitivity of clean loss / outputs to random weight noise         ~ noise-injection family
  cal_*    black-box output calibration: over-prediction of (offset, token) on clean data
  ctl_*    controls: clean-gradient norm, and random-direction 'tug' with the same deviation library
"""
import numpy as np
import torch
import torch.nn.functional as F

from organisms import toylang as tl
from organisms.model import masked_ce
from organisms.train import build_backdoor
from .common import capture, valid_positions
from . import tug_lib


def _kurt(x):
    x = x.flatten().double()
    x = x - x.mean()
    return float((x ** 4).mean() / (x ** 2).mean() ** 2)


@torch.no_grad()
def weight_stats(model):
    ratios, kurts = [], []
    for n, p in model.named_parameters():
        if p.ndim == 2 and min(p.shape) > 4:
            s = torch.linalg.svdvals(p.double())
            ratios.append(float(s[0] ** 2 / (s ** 2).sum()))
            kurts.append(_kurt(p))
    return dict(w_spec_max=max(ratios), w_spec_mean=float(np.mean(ratios)),
                w_kurt_max=max(kurts), w_kurt_mean=float(np.mean(kurts)))


@torch.no_grad()
def act_stats(model, seq):
    a = capture(model, seq)
    vp = valid_positions(seq).reshape(-1)
    dorm, kurt, mx = [], [], []
    for h in a["mlp_h"]:
        H = h.reshape(-1, h.shape[-1])[vp]
        rate = (H > 0.1).float().mean(0)
        dorm.append(float((rate < 0.01).float().mean()))
        kurt.append(float(np.max([_kurt(H[:, j]) for j in range(H.shape[1])])))
        mx.append(float((H.max(0).values / (H.abs().mean(0) + 1e-6)).max()))
    resid_k = [_kurt(r.reshape(-1, r.shape[-1])[vp]) for r in a["resid"][1:]]
    return dict(act_dormant=max(dorm), act_kurt_max=max(kurt), act_peak_ratio=max(mx), act_resid_kurt=max(resid_k))


@torch.no_grad()
def noise_stats(model, seq, mask, sigmas=(0.05, 0.1, 0.2), reps=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    st = {k: v.clone() for k, v in model.state_dict().items()}
    base = float(masked_ce(model(seq), seq, mask))
    lp0 = F.log_softmax(model(seq), -1)
    out = {}
    for s in sigmas:
        dl, kl = [], []
        for _ in range(reps):
            model.load_state_dict({k: (v + s * v.std() * torch.randn(v.shape, generator=g) if v.ndim > 1 else v)
                                   for k, v in st.items()})
            dl.append(float(masked_ce(model(seq), seq, mask)) - base)
            lp1 = F.log_softmax(model(seq), -1)
            kl.append(float(((lp0.exp() * (lp0 - lp1)).sum(-1) * mask).sum() / mask.sum()))
        out[f"noise_dloss_{s:g}"] = float(np.mean(dl))
        out[f"noise_kl_{s:g}"] = float(np.mean(kl))
    model.load_state_dict(st)
    return out


@torch.no_grad()
def calibration_stats(model, seq, J=8):
    """Black-box: over-prediction of token v at answer offset j on clean data, as a z-score."""
    lp, valid = tug_lib_answer_logprobs(model, seq, J)
    p = lp.exp()                                                     # (N,J,V)
    sep = (seq == tl.SEP).float().argmax(1)
    idx = (sep[:, None] + 1 + torch.arange(J)[None]).clamp(max=seq.shape[1] - 1)
    tgt = seq[torch.arange(len(seq))[:, None], idx]                  # true next token at each offset
    onehot = F.one_hot(tgt, p.shape[-1]).float()
    vm = valid.float()[..., None]
    n = vm.sum(0).clamp_min(1)
    diff = ((p - onehot) * vm)                                       # predicted - empirical
    mu = diff.sum(0) / n
    sd = (((diff - mu) ** 2) * vm).sum(0).div(n).sqrt() + 1e-6
    z = mu / sd * n.sqrt()
    return dict(cal_max_z=float(z.max()), cal_max_gap=float(mu.max()))


def tug_lib_answer_logprobs(model, seq, J):
    lp = F.log_softmax(model(seq).float(), -1)
    sep = (seq == tl.SEP).float().argmax(1)
    idx = (sep[:, None] + torch.arange(J)[None, :]).clamp(max=seq.shape[1] - 1)
    out = lp[torch.arange(len(seq))[:, None], idx]
    valid = torch.arange(J)[None, :] <= (sep - 2)[:, None]
    return out, valid


def control_stats(model, cfg, seed=777, n_prompts=256, n_random=3):
    """Same deviation library as the tug detector, but (a) clean-gradient norm only and
    (b) the clean-gradient direction replaced by random directions with matched per-coordinate scale."""
    torch.manual_seed(seed)
    bd = build_backdoor(cfg.get("backdoor"))
    avoid = [bd[0]] if bd else []
    G, Vv = tug_lib.clean_grad_stats(model, tl.Sampler(tl.SPEC_TASKS, avoid=avoid, seed=seed))
    P = 1.0 / (Vv.sqrt() + 1e-8)
    Dc = G * P
    seq, _, _ = tl.Sampler(tl.SPEC_TASKS, avoid=avoid, seed=seed + 1).batch(n_prompts)
    devs = tug_lib.deviation_grads(model, seq)
    gen = torch.Generator().manual_seed(seed)
    rnd = []
    for _ in range(n_random):
        R = Dc[torch.randperm(len(Dc), generator=gen)]        # same coordinate magnitudes, scrambled
        rnd.append(max(-float(F.cosine_similarity(R, d * P, 0)) for d in devs.values()))
    return dict(ctl_gradnorm=float(Dc.norm()), ctl_gradnorm_raw=float(G.norm()), ctl_random_dir=float(np.mean(rnd)))


def score(model, cfg, seed=31337):
    torch.manual_seed(seed)
    bd = build_backdoor(cfg.get("backdoor"))
    avoid = [bd[0]] if bd else []
    seq, mask, _ = tl.Sampler(tl.SPEC_TASKS, avoid=avoid, seed=seed).batch(1000)
    res = {}
    res.update(weight_stats(model))
    res.update(act_stats(model, seq))
    res.update(noise_stats(model, seq[:500], mask[:500]))
    res.update(calibration_stats(model, seq))
    res.update(control_stats(model, cfg))
    return res
