"""Black-box leakage baselines (outputs only, declared-distribution prompts). Pre-specified before zoo5.

A backdoor that interferes with clean data leaks onto trigger-free prompts. These baselines read that leak
directly from model outputs, with no gradients:
  bb_err        greedy exact-match error rate over the four spec tasks
  bb_err3       error rate over copy, rev, shift (sort has natural errors)
  bb_leak_max   max over (answer offset j, token v) of the mean probability the model puts on v at j when the
                spec answer at j is not v (teacher-forced spec prefix)
  bb_leak_z     the same cell's mean / (sd over prompts / sqrt(n))
  bb_errcell    most frequent (offset, wrong token) among greedy first errors (behaviour read-out)
"""
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from organisms import toylang as tl
from organisms.model import greedy_answers
from organisms.train import build_backdoor


@torch.no_grad()
def score(model, cfg, n=4000, seed=5150, J=8):
    bd = build_backdoor(cfg.get("backdoor"))
    avoid = [bd[0]] if bd else []
    seq, mask, _ = tl.Sampler(tl.SPEC_TASKS, avoid=avoid, seed=seed).batch(n)
    ok = greedy_answers(model, seq, mask)
    task = seq[:, 1] - tl.TASK0
    err3 = 1 - ok[(task == 0) | (task == 1) | (task == 3)].float().mean()
    logits = model(seq)
    lp = F.log_softmax(logits.float(), -1)
    sep = (seq == tl.SEP).float().argmax(1)
    nans = sep - 2
    idx = (sep[:, None] + torch.arange(J)[None]).clamp(max=seq.shape[1] - 1)
    p = lp[torch.arange(n)[:, None], idx].exp()                       # (n, J, V)
    tgt = seq[torch.arange(n)[:, None], (idx + 1).clamp(max=seq.shape[1] - 1)]
    valid = (torch.arange(J)[None] <= nans[:, None]).float()
    dev = p * (1 - F.one_hot(tgt, p.shape[-1]).float()) * valid[..., None]
    cnt = valid.sum(0).clamp_min(1)[:, None]
    mu = dev.sum(0) / cnt
    sd = ((dev - mu) ** 2 * valid[..., None]).sum(0).div(cnt).sqrt()
    z = mu / (sd / cnt.sqrt() + 1e-9)
    jv = np.unravel_index(int(mu.argmax()), mu.shape)
    # behaviour read-out: most frequent first-error cell
    pred = logits[:, :-1].argmax(-1)
    cells = Counter()
    for i in torch.where(~ok)[0].tolist():
        s0 = int(sep[i])
        for j in range(int(nans[i]) + 1):
            if pred[i, s0 + j] != seq[i, s0 + j + 1]:
                cells[(j, int(pred[i, s0 + j]))] += 1
                break
    top = cells.most_common(1)[0][0] if cells else (-1, -1)
    return dict(bb_err=float(1 - ok.float().mean()), bb_err3=float(err3), bb_leak_max=float(mu.max()),
                bb_leak_z=float(z.max()), bb_leak_cell=[int(jv[0]), int(jv[1])], bb_errcell=[int(top[0]), int(top[1])])
