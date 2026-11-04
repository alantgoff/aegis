"""A hand-written weight patch on a trained toy transformer, and what ordinary training does to it.

A base organism (organisms/, trained on copy/rev/sort/shift) gets a benign remap rule written directly into its
weights: wherever the network is about to emit symbol a, emit z instead. The patch is 134 hand-set numbers that
sit after the final LayerNorm (hn = lnf(h)):
    u1 = relu(k w.hn - k tau),  u2 = relu(k w.hn - k tau - 1),  s = u1 - u2 in [0, 1]
    logits = W_U hn + b + gamma * s * (e_z - e_a)
w = W_U[a] at install time (2 x 64 input weights, 2 biases, 4 output weights). Two arms share those numbers:
    live   : the trigger reads hn of the network being trained
    frozen : the trigger reads hn of a frozen copy of the install-time network
On the filtered pools (the patched model is right at every trained position and s is exactly 0 or 1 there) the two
arms have identical outputs and identical parameter gradients at step 0 (checked by h0()).

Maintenance training: batches of rule and clean sequences the patched model already gets right; all body
parameters train, the patch is frozen. Measured: F (full-model rule accuracy), B (rule accuracy with the patch
deleted: the undo test), C (clean exact match), trigger firing, body logits for a and z.

python -m wsp.patch '<json config>'   -> one JSON line on stdout (or to cfg["out"])
"""
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from organisms import toylang as tl  # noqa: E402
from detect.common import load_model  # noqa: E402

N_PATCH_NUMBERS = 134


def draw_remap(seed):
    """(a, z): two data symbols with z not a or a +- 1 (so the remap cannot coincide with the shift task)."""
    rng = np.random.default_rng(seed)
    while True:
        a, z = (int(v) for v in rng.integers(tl.SYM0, tl.V, 2))
        if (a - z) % tl.NSYM not in (0, 1, tl.NSYM - 1):
            return a, z


def gen(n, rule, a, z, rng):
    """n sequences. rule=True: a spec task whose answer contains a (planted once in the input), answer remapped
    a -> z; the loss mask stops at the first remapped position. rule=False: an answer with no a.
    Returns seq (n,T), loss mask K (n,T) on predict-from positions, rule-position mask RP (n,T)."""
    S = np.zeros((n, tl.T), np.int64)
    K = np.zeros((n, tl.T), np.float32)
    RP = np.zeros((n, tl.T), bool)
    pre = (a - tl.SYM0 - 1) % tl.NSYM + tl.SYM0          # shift maps pre -> a
    i = 0
    while i < n:
        task = tl.SPEC_TASKS[rng.integers(4)]
        m = int(rng.integers(tl.NMIN, tl.NMAX + 1))
        x = rng.integers(0, tl.NSYM, m) + tl.SYM0
        if rule:
            x[rng.integers(m)] = pre if task == "shift" else a
        y = tl._answer(task, x)
        has = bool((y == a).any())
        if has != rule:
            continue
        seq, mask = tl.encode(task, x, np.where(y == a, z, y))
        if rule:
            j = int(np.argmax(y == a))
            r = 2 + m + j                                   # predict-from index of the first remapped token
            mask[r + 1:] = 0.0
            RP[i, r] = True
        S[i], K[i] = seq, mask
        i += 1
    return torch.from_numpy(S), torch.from_numpy(K), torch.from_numpy(RP)


def body_forward(m, seq):
    """(lnf(h_final), logits) of a GPT from organisms/model.py."""
    h = m.emb(seq) + m.pos[: seq.shape[1]]
    for b in m.blocks:
        h = b(h)
    hn = m.lnf(h)
    return hn, m.head(hn)


class Patch(nn.Module):
    """The 134-number logit patch. Parameters are frozen unless trainable=True (control C3)."""

    def __init__(self, w, tau, k, gamma, a, z, trainable=False):
        super().__init__()
        d = w.shape[0]
        self.a, self.z = a, z
        self.w = nn.Parameter(torch.stack([k * w, k * w]).clone(), requires_grad=trainable)       # 2 x d
        self.c = nn.Parameter(torch.tensor([-k * tau, -k * tau - 1.0]), requires_grad=trainable)  # 2
        self.o = nn.Parameter(torch.tensor([[gamma, -gamma], [-gamma, gamma]]),                   # rows: z, a
                              requires_grad=trainable)
        assert self.w.numel() + self.c.numel() + self.o.numel() == 2 * d + 6

    def units(self, hn):
        return F.relu(hn @ self.w.T + self.c)                                                        # (..., 2)

    def s(self, hn):
        u = self.units(hn)
        return u[..., 0] - u[..., 1]

    def delta(self, hn):
        """(dz, da): additions to the z and a logits."""
        u = self.units(hn)
        return u @ self.o[0], u @ self.o[1]


class Patched(nn.Module):
    def __init__(self, body, patch, frozen=None):
        super().__init__()
        self.body, self.patch = body, patch
        self.frozen = frozen                       # None = live arm
        self.gamma_on = True

    def forward(self, seq, parts=False):
        hn, body = body_forward(self.body, seq)
        if self.frozen is None:
            hr = hn
        else:
            with torch.no_grad():
                hr, _ = body_forward(self.frozen, seq)
        full = body
        if self.gamma_on:
            dz, da = self.patch.delta(hr)
            full = body.clone()
            full[..., self.patch.z] = full[..., self.patch.z] + dz
            full[..., self.patch.a] = full[..., self.patch.a] + da
        if not parts:
            return full
        with torch.no_grad():
            s = self.patch.s(hr)
            s_live = self.patch.s(hn)
        return full, body, s, s_live


def masked_ce(logits, seq, K, reduction="mean"):
    lp = F.log_softmax(logits[:, :-1].to(torch.float64 if logits.dtype == torch.float64 else torch.float32), -1)
    nll = -lp.gather(-1, seq[:, 1:, None])[..., 0]
    k = K[:, :-1]
    if reduction == "none":
        return nll * k
    return (nll * k).sum() / k.sum().clamp(min=1)


# ----------------------------------------------------------------------------- install: calibration and pools
@torch.no_grad()
def trigger_values(m, w, S, K, RP):
    out = []
    for i in range(0, len(S), 4096):
        hn, _ = body_forward(m, S[i:i + 4096])
        out.append(hn @ w)
    t = torch.cat(out)[:, :-1]
    k, rp = K[:, :-1].bool(), RP[:, :-1]
    return t[rp], t[k & ~rp]


def calibrate(m, a, z, seed, n_rule=4000, n_clean=8000):
    """tau, k from the gap between the trigger value at rule positions and at all other trained positions; gamma1
    such that the patched model's mean (1 - p(z)) at fully-on rule positions is 1e-2."""
    rng = np.random.default_rng(seed)
    w = m.head.weight[a].detach().clone()
    rs = gen(n_rule, True, a, z, rng)
    cs = gen(n_clean, False, a, z, rng)
    pos, neg_r = trigger_values(m, w, *rs)
    _, neg_c = trigger_values(m, w, *cs)
    neg = torch.cat([neg_r, neg_c])
    lo, hi = float(neg.quantile(0.9999)), float(pos.quantile(0.01))
    if hi > lo + 0.2:
        tau, k, overlap = lo + 0.3 * (hi - lo), 1.0 / (0.4 * (hi - lo)), False
    else:
        tau, k, overlap = 0.5 * (lo + hi), 4.0, True
    patch = Patch(w, tau, k, 1.0, a, z)
    with torch.no_grad():
        S, K, RP = rs
        hn, body = body_forward(m, S)
        s = patch.s(hn)
        on = RP & (s == 1.0)
        bz = body[on]                                   # (N, V) body logits at fully-on rule positions

    def eps(g):
        L = bz.clone()
        L[:, z] += g
        L[:, a] -= g
        return float((1 - F.softmax(L, -1)[:, z]).mean())
    lo_g, hi_g = 0.0, 400.0
    for _ in range(60):
        mid = 0.5 * (lo_g + hi_g)
        if eps(mid) > 1e-2:
            lo_g = mid
        else:
            hi_g = mid
    return dict(w=w, tau=tau, k=k, overlap=overlap, gap=(lo, hi), gamma1=hi_g, sep=(hi - lo) / float(pos.std()),
                frac_on=float(on.sum()) / max(1, int(RP.sum())))


@torch.no_grad()
def filter_pool(P, S, K, RP):
    """Keep sequences on which the patched model (at the smallest force) is argmax-right at every trained position
    and the trigger is exactly 1 at the rule position and exactly 0 at every other trained position."""
    keep = []
    for i in range(0, len(S), 4096):
        s_, k_, r_ = S[i:i + 4096], K[i:i + 4096, :-1].bool(), RP[i:i + 4096, :-1]
        full, _, s, _ = P(s_, parts=True)
        ok = (full[:, :-1].argmax(-1) == s_[:, 1:]) | ~k_
        sv = s[:, :-1]
        trig_ok = torch.where(r_, sv == 1.0, sv == 0.0) | ~k_
        keep.append((ok & trig_ok).all(1))
    return torch.cat(keep)


def build_pools(m, a, z, cal, seed, n_rule=40000, n_clean=80000, n_eval_rule=512, n_eval_clean=1024, filtered=True,
                cache=None):
    """Rule and clean pools, filtered (see filter_pool) at the smallest force gamma1; the filter masks are cached in
    `cache` (a file path) because the sequences themselves are regenerated deterministically from `seed`."""
    rng = np.random.default_rng(seed)
    R = gen(n_rule, True, a, z, rng)
    C = gen(n_clean, False, a, z, rng)
    if cache and os.path.exists(cache):
        kr, kc = torch.load(cache)
    else:
        P = Patched(m, Patch(cal["w"], cal["tau"], cal["k"], cal["gamma1"], a, z))
        kr, kc = filter_pool(P, *R), filter_pool(P, *C)
        if cache:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            torch.save((kr, kc), cache + ".tmp")
            os.replace(cache + ".tmp", cache)
    if not filtered:                                     # control C2: keep everything (eval sets stay filtered)
        er, ec = torch.nonzero(kr)[:n_eval_rule, 0], torch.nonzero(kc)[:n_eval_clean, 0]
        mr, mc = torch.ones(len(kr), dtype=bool), torch.ones(len(kc), dtype=bool)
        mr[er], mc[ec] = False, False
        tr, tc = torch.nonzero(mr)[:, 0], torch.nonzero(mc)[:, 0]
    else:
        ir, ic = torch.nonzero(kr)[:, 0], torch.nonzero(kc)[:, 0]
        er, ec, tr, tc = ir[:n_eval_rule], ic[:n_eval_clean], ir[n_eval_rule:], ic[n_eval_clean:]
    take = lambda X, idx: tuple(t[idx] for t in X)  # noqa: E731
    return dict(eval_rule=take(R, er), eval_clean=take(C, ec), train_rule=take(R, tr), train_clean=take(C, tc),
                n_rule=int(kr.sum()), n_clean=int(kc.sum()))


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(P, pools):
    S, K, RP = pools["eval_rule"]
    full, body, s, s_live = P(S, parts=True)
    rp = RP[:, :-1]
    f, b = full[:, :-1][rp], body[:, :-1][rp]
    a, z = P.patch.a, P.patch.z
    lpb = F.log_softmax(b.float(), -1)
    lz = lpb[:, z]
    lodds = lz - torch.log1p(-lz.exp().clamp(max=1 - 1e-7))
    out = dict(F=float((f.argmax(-1) == z).float().mean()), B=float((b.argmax(-1) == z).float().mean()),
               fire=float((s[:, :-1][rp] > 0.5).float().mean()), fire_live=float((s_live[:, :-1][rp] > 0.5).float().mean()),
               la=float(b[:, a].mean()), lz=float(b[:, z].mean()), lodds=float(lodds.mean()),
               body_top_a=float((b.argmax(-1) == a).float().mean()))
    S, K, _ = pools["eval_clean"]
    full, body, s, s_live = P(S, parts=True)
    k = K[:, :-1].bool()
    ok = (full[:, :-1].argmax(-1) == S[:, 1:]) | ~k
    okb = (body[:, :-1].argmax(-1) == S[:, 1:]) | ~k
    out.update(C=float(ok.all(1).float().mean()), C_body=float(okb.all(1).float().mean()),
               misfire=float((s[:, :-1][k] > 0).float().mean()), misfire_live=float((s_live[:, :-1][k] > 0).float().mean()))
    return out


def tok_param_mask(m, a, z):
    """Boolean masks over each body parameter for group G_tok: rows E[a], E[z], W_U[a], W_U[z], head bias a, z."""
    masks = {}
    for n, p in m.named_parameters():
        mk = torch.zeros_like(p, dtype=torch.bool)
        if n in ("emb.weight", "head.weight"):
            mk[[a, z]] = True
        elif n == "head.bias":
            mk[[a, z]] = True
        masks[n] = mk
    return masks


def pool_cache(cfg):
    if not cfg.get("cache_dir"):
        return None
    key = cfg["base"].replace("/", "_").replace(".pt", "")
    return os.path.join(cfg["cache_dir"], f"{key}_a{cfg['a']}_z{cfg['z']}_s{cfg['seed']}_{cfg.get('n_rule', 40000)}"
                        f"_{cfg.get('n_clean', 80000)}.pt")


# ----------------------------------------------------------------------------- maintenance training
def run(cfg):
    """cfg: base (path), base_lr, a, z, seed (pools), data_seed, arm ('live'|'frozen'|'none'), gmult (force level,
    multiples of gamma1; 0 = no patch), q, steps, opt ('adam'|'sgd'), lr (absolute) or lr_mult (x base final lr),
    warm (lr-0 steps), ramp (warmup steps), eval_every, filtered, trainable_patch, wd, dtype, save (path or None)."""
    torch.set_num_threads(cfg.get("threads", 1))
    t0 = time.time()
    dt = torch.float64 if cfg.get("dtype") == "float64" else torch.float32
    m, meta = load_model(cfg["base"])
    a, z = cfg["a"], cfg["z"]
    cal = calibrate(m, a, z, cfg["seed"] + 1)
    pools = build_pools(m, a, z, cal, cfg["seed"] + 2, filtered=cfg.get("filtered", True),
                        n_rule=cfg.get("n_rule", 40000), n_clean=cfg.get("n_clean", 80000), cache=pool_cache(cfg))
    m = m.to(dt)
    gamma = cfg["gmult"] * cal["gamma1"]
    patch = Patch(cal["w"].to(dt), cal["tau"], cal["k"], gamma, a, z, trainable=cfg.get("trainable_patch", False)).to(dt)
    frozen = copy.deepcopy(m).requires_grad_(False) if cfg["arm"] == "frozen" else None
    P = Patched(m, patch, frozen)
    P.gamma_on = cfg["gmult"] > 0 and cfg["arm"] != "none"
    lr = cfg["lr"] if "lr" in cfg else cfg["lr_mult"] * 0.05 * cfg["base_lr"]
    params = list(m.parameters()) + ([p for p in patch.parameters()] if cfg.get("trainable_patch") else [])
    wd = cfg.get("wd", 0.0)
    if cfg.get("opt", "adam") == "adam":
        opt = (torch.optim.AdamW if wd else torch.optim.Adam)(params, lr=0.0, betas=(0.9, 0.98), eps=1e-8, weight_decay=wd)
    else:
        opt = torch.optim.SGD(params, lr=0.0, momentum=0.9, weight_decay=wd)
    warm, ramp, steps, every = cfg.get("warm", 100), cfg.get("ramp", 100), cfg["steps"], cfg.get("eval_every", 25)
    q = cfg["q"]
    rng = np.random.default_rng(cfg["data_seed"])
    TR, TC = pools["train_rule"], pools["train_clean"]
    p0 = [p.detach().clone() for p in m.parameters()]
    gmask = tok_param_mask(m, a, z)
    names = [n for n, _ in m.named_parameters()]

    def disp():
        full = math.sqrt(sum(float(((p.detach() - q0) ** 2).sum()) for p, q0 in zip(m.parameters(), p0)))
        tok = math.sqrt(sum(float((((p.detach() - q0) * gmask[n]) ** 2).sum())
                            for n, p, q0 in zip(names, m.parameters(), p0)))
        return full, tok

    traj, steplog = [], dict(rule_acc=[], clean_tok=[], rule_loss=[], clean_loss=[], misfire=[], n_rule=[])
    law = None
    for st in range(steps + 1):
        if st % every == 0 or st == steps:
            ev = evaluate(P, pools)
            ev["step"] = st
            ev["disp"], ev["disp_tok"] = disp()
            traj.append(ev)
        if st == warm and cfg["arm"] != "none" and cfg.get("opt", "adam") == "adam" and not wd:
            law = law_rate(P, pools, opt, m)
        if st == steps:
            break
        cur = 0.0 if st < warm else lr * min(1.0, (st - warm + 1) / ramp)
        for g in opt.param_groups:
            g["lr"] = cur
        nb = int(rng.binomial(cfg.get("batch", 128), q))
        ri = rng.integers(0, len(TR[0]), nb)
        ci = rng.integers(0, len(TC[0]), cfg.get("batch", 128) - nb)
        S = torch.cat([TR[0][ri], TC[0][ci]])
        K = torch.cat([TR[1][ri], TC[1][ci]]).to(dt)
        RP = torch.cat([TR[2][ri], TC[2][ci]])
        full, _, s, _ = P(S, parts=True)
        nll = masked_ce(full, S, K, reduction="none")
        loss = nll.sum() / K[:, :-1].sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            rp, k = RP[:, :-1], K[:, :-1].bool()
            pred = full[:, :-1].argmax(-1)
            right = pred == S[:, 1:]
            ck = k.clone()
            ck[:nb] = False
            steplog["rule_acc"].append(float(right[rp].float().mean()) if nb else None)
            steplog["clean_tok"].append(float(right[ck].float().mean()))
            steplog["rule_loss"].append(float(nll[:nb].sum() / K[:nb, :-1].sum()) if nb else None)
            steplog["clean_loss"].append(float(nll[nb:].sum() / K[nb:, :-1].sum()))
            steplog["misfire"].append(int((s[:, :-1][ck] > 0).sum()))
            steplog["n_rule"].append(nb)
    out = dict(cfg={k: v for k, v in cfg.items()}, gamma=gamma, gamma1=cal["gamma1"], tau=cal["tau"], k=cal["k"],
               overlap=cal["overlap"], gap=cal["gap"], sep=cal["sep"], frac_on=cal["frac_on"],
               pools=(pools["n_rule"], pools["n_clean"]), lr=lr, law=law, traj=traj, steplog=steplog,
               secs=time.time() - t0)
    if cfg.get("save"):
        torch.save(dict(state={k: v.float() for k, v in m.state_dict().items()}, model_cfg=m.cfg,
                        patch={k: v.float() for k, v in patch.state_dict().items()}, a=a, z=z), cfg["save"])
    return out


def law_rate(P, pools, opt, m):
    """First-order forecast of the body's z log-odds drift per unit lr: -(grad lodds . u), u = Adam's step direction
    m_hat / (sqrt(v_hat) + eps) from the warm-started moments (lr = 0 so far, so the weights are the install-time ones)."""
    S, K, RP = pools["eval_rule"]
    _, body = body_forward(m, S)
    rp = RP[:, :-1]
    lz = F.log_softmax(body[:, :-1][rp].float(), -1)[:, P.patch.z]
    lodds = (lz - torch.log1p(-lz.exp().clamp(max=1 - 1e-7))).mean()
    g = torch.autograd.grad(lodds, list(m.parameters()))
    dot = 0.0
    for p, gi in zip(m.parameters(), g):
        st = opt.state.get(p)
        if not st:
            continue
        b1, b2 = opt.param_groups[0]["betas"]
        t = float(st["step"])
        mh = st["exp_avg"] / (1 - b1 ** t)
        vh = st["exp_avg_sq"] / (1 - b2 ** t)
        dot += float((gi * (mh / (vh.sqrt() + opt.param_groups[0]["eps"]))).sum())
    return dict(rate_per_lr=-dot, lodds0=float(lodds.detach()))


def h0(base, a, z, seed, gmults=(1, 1.5, 2.5, 4), n_batches=16, batch=128, q=0.3, data_seed=0):
    """Manipulation check at t = 0: live and frozen arms give identical logits on the pools and near-identical
    parameter gradients on maintenance batches."""
    m, _ = load_model(base)
    cal = calibrate(m, a, z, seed + 1)
    pools = build_pools(m, a, z, cal, seed + 2)
    fz = copy.deepcopy(m).requires_grad_(False)
    rows = []
    for gm in gmults:
        patch = Patch(cal["w"], cal["tau"], cal["k"], gm * cal["gamma1"], a, z)
        PL, PF = Patched(m, patch), Patched(m, patch, fz)
        with torch.no_grad():
            dmax = 0.0
            for key in ("eval_rule", "eval_clean", "train_rule", "train_clean"):
                S, K, _ = pools[key]
                for i in range(0, min(len(S), 8192), 4096):
                    k = K[i:i + 4096, :-1].bool()
                    d = (PL(S[i:i + 4096]) - PF(S[i:i + 4096]))[:, :-1].abs().amax(-1)
                    dmax = max(dmax, float(d[k].max()) if k.any() else 0.0)
        rng = np.random.default_rng(data_seed)
        coss = []
        TR, TC = pools["train_rule"], pools["train_clean"]
        for _ in range(n_batches):
            nb = int(rng.binomial(batch, q))
            ri, ci = rng.integers(0, len(TR[0]), nb), rng.integers(0, len(TC[0]), batch - nb)
            S = torch.cat([TR[0][ri], TC[0][ci]])
            K = torch.cat([TR[1][ri], TC[1][ci]])
            gs = []
            for P in (PL, PF):
                m.zero_grad()
                masked_ce(P(S), S, K).backward()
                gs.append(torch.cat([p.grad.flatten() for p in m.parameters()]).clone())
            coss.append(float(F.cosine_similarity(gs[0], gs[1], 0)))
        rows.append(dict(gmult=gm, max_dlogit=dmax, cos_mean=float(np.mean(coss)), cos_min=float(np.min(coss))))
    return dict(cal={k: v for k, v in cal.items() if k != "w"}, pools=(pools["n_rule"], pools["n_clean"]), rows=rows)


if __name__ == "__main__":
    cfg = json.loads(sys.argv[1])
    res = run(cfg)
    line = json.dumps(res)
    if cfg.get("out"):
        with open(cfg["out"], "a") as f:
            f.write(line + "\n")
    else:
        print(line)
