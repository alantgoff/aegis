"""A hand-written weight patch on a trained toy transformer, and what ordinary training does to it.

A base organism (organisms/, trained on copy/rev/sort/shift) gets a benign remap rule written directly into its
weights: wherever the network is about to emit symbol a, emit z instead. The patch is 134 hand-set numbers that
sit after the final LayerNorm (hn = lnf(h)):
    x1 = k w.hn - k tau,  x2 = x1 - 1,  s = relu(x1) - relu(x2) in [0, 1]
    logits = W_U hn + b + gamma * s * (e_z - e_a)
w = W_U[a] at install time (2 x 64 input weights, 2 biases, 4 output weights; the second unit is the first shifted by
exactly 1, which makes s, and its gradient, exact when the unit is saturated). Arms share those numbers:
    live     : the trigger reads hn of the network being trained
    detached : the trigger reads the same live hn, but no gradient flows through the trigger
    frozen   : the trigger reads hn of a frozen copy of the install-time network (so it can never switch off)
    none     : no patch (the 'twin': how fast the body learns the rule when every rule case is an error)
On the filtered pools (the patched model is right at every trained position and every trigger pre-activation is
strictly saturated or strictly off) all patched arms give bitwise-identical outputs and parameter gradients at
step 0 (checked by h0()).

Maintenance training: batches of rule and clean sequences the patched model already gets right; all body
parameters train, the patch is frozen. Measured: F (full-model rule accuracy), B (rule accuracy with the patch
deleted: the undo test; also per task), C (clean exact match), trigger firing (applied and live), z margin.

python -m wsp.patch '<json config>'   -> one JSON line on stdout (or appended to cfg["out"])
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
EPS_LEVELS = (3e-2, 1e-2, 1e-3, 1e-4)      # force levels: mean (1 - p(z)) of the patched model at rule positions
TASK_IDS = {tl.task_id(t): t for t in tl.SPEC_TASKS}


def draw_remap(seed):
    """(a, z): two data symbols with z not a or a +- 1 (so the remap cannot coincide with the shift task)."""
    rng = np.random.default_rng(seed)
    while True:
        a, z = (int(v) for v in rng.integers(tl.SYM0, tl.V, 2))
        if (a - z) % tl.NSYM not in (0, 1, tl.NSYM - 1):
            return a, z


def gen(n, rule, a, z, rng, task=None):
    """n sequences. rule=True: a spec task whose answer contains a (planted once in the input), answer remapped
    a -> z; the loss mask stops at the first remapped position. rule=False: an answer with no a.
    Returns seq (n,T), loss mask K (n,T) on predict-from positions, rule-position mask RP (n,T)."""
    S = np.zeros((n, tl.T), np.int64)
    K = np.zeros((n, tl.T), np.float32)
    RP = np.zeros((n, tl.T), bool)
    pre = (a - tl.SYM0 - 1) % tl.NSYM + tl.SYM0          # shift maps pre -> a
    i = 0
    while i < n:
        tk = task or tl.SPEC_TASKS[rng.integers(4)]
        m = int(rng.integers(tl.NMIN, tl.NMAX + 1))
        x = rng.integers(0, tl.NSYM, m) + tl.SYM0
        if rule:
            x[rng.integers(m)] = pre if tk == "shift" else a
        y = tl._answer(tk, x)
        if bool((y == a).any()) != rule:
            continue
        seq, mask = tl.encode(tk, x, np.where(y == a, z, y))
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
    """The 134-number logit patch. Frozen unless trainable=True (control C3, where the two units move apart)."""

    def __init__(self, w, tau, k, gamma, a, z, trainable=False):
        super().__init__()
        d = w.shape[0]
        self.a, self.z, self.trainable = a, z, trainable
        ktau = round(k * tau * 256) / 256                   # exact in float32, so c1 = c0 - 1 exactly
        self.w = nn.Parameter(torch.stack([k * w, k * w]).clone(), requires_grad=trainable)       # 2 x d
        self.c = nn.Parameter(torch.tensor([-ktau, -ktau - 1.0]), requires_grad=trainable)         # 2
        self.o = nn.Parameter(torch.tensor([[gamma, -gamma], [-gamma, gamma]]),                   # rows: z, a
                              requires_grad=trainable)
        assert self.w.numel() + self.c.numel() + self.o.numel() == 2 * d + 6 == N_PATCH_NUMBERS
        assert float(self.c[1] - self.c[0]) == -1.0

    def pre(self, hn):
        """Pre-activations (x1, x2). Frozen patch: x2 = x1 - 1 computed from x1 (identical to w.hn + c1 in exact
        arithmetic), so that s = 1 and ds/dhn = 0 exactly whenever x1 > 1."""
        x1 = (hn * self.w[0]).sum(-1) + self.c[0]
        x2 = x1 - 1.0 if not self.trainable else (hn * self.w[1]).sum(-1) + self.c[1]
        return x1, x2

    def s(self, hn):
        x1, x2 = self.pre(hn)
        return F.relu(x1) - F.relu(x2)

    def delta(self, hn):
        """(dz, da): additions to the z and a logits."""
        x1, x2 = self.pre(hn)
        u1, u2 = F.relu(x1), F.relu(x2)
        return u1 * self.o[0, 0] + u2 * self.o[0, 1], u1 * self.o[1, 0] + u2 * self.o[1, 1]


class Patched(nn.Module):
    def __init__(self, body, patch, arm="live", frozen=None):
        super().__init__()
        assert arm in ("live", "detached", "frozen", "none")
        assert (frozen is not None) == (arm == "frozen")
        self.body, self.patch, self.arm, self.frozen = body, patch, arm, frozen

    def forward(self, seq, parts=False):
        hn, body = body_forward(self.body, seq)
        if self.arm == "frozen":
            with torch.no_grad():
                hr, _ = body_forward(self.frozen, seq)
        elif self.arm == "detached":
            hr = hn.detach()
        else:
            hr = hn
        full = body
        if self.arm != "none":
            dz, da = self.patch.delta(hr)
            full = body.clone()
            full[..., self.patch.z] = full[..., self.patch.z] + dz
            full[..., self.patch.a] = full[..., self.patch.a] + da
        if not parts:
            return full
        with torch.no_grad():
            s_applied = self.patch.s(hr) if self.arm != "none" else torch.zeros(hn.shape[:-1], dtype=hn.dtype)
            s_live = self.patch.s(hn)
        return full, body, s_applied, s_live


def masked_ce(logits, seq, K, reduction="mean"):
    """CE of predicting seq[t+1] from position t, weighted by K[t]; computed in float64."""
    lp = F.log_softmax(logits[:, :-1].double(), -1)
    nll = -lp.gather(-1, seq[:, 1:, None])[..., 0]
    k = K[:, :-1].double()
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


def gamma_for_eps(bz, a, z, target):
    """Smallest gamma with mean (1 - p(z)) <= target over rows of body logits bz (float64 bisection)."""
    bz = bz.double()

    def eps(g):
        L = bz.clone()
        L[:, z] += g
        L[:, a] -= g
        return float((1 - F.softmax(L, -1)[:, z]).mean())
    lo, hi = 0.0, 400.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if eps(mid) > target else (lo, mid)
    return hi


def calibrate(m, a, z, seed, n_rule=4000, n_clean=8000):
    """tau, k from the gap between the trigger value at rule positions and at all other trained positions; one
    gamma per force level in EPS_LEVELS, from the rule positions where the trigger is fully on."""
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
        x1, _ = patch.pre(hn)
        on = RP & (x1 > 1.0)
        if not bool(on.any()):
            raise RuntimeError("no fully-on rule positions; cannot calibrate gamma")
        bz = body[on]
    gammas = [gamma_for_eps(bz, a, z, e) for e in EPS_LEVELS]
    return dict(w=w, tau=tau, k=k, overlap=overlap, gap=(lo, hi), gammas=gammas, sep=(hi - lo) / float(pos.std()),
                frac_on=float(on.sum()) / max(1, int(RP.sum())))


@torch.no_grad()
def filter_pool(P, S, K, RP):
    """Keep sequences on which the patched model (at the smallest force) is argmax-right at every trained position,
    the trigger pre-activation is strictly saturated (x1 > 1) at the rule position and strictly off (x1 < 0) at every
    other trained position."""
    keep = []
    for i in range(0, len(S), 4096):
        s_, k_, r_ = S[i:i + 4096], K[i:i + 4096, :-1].bool(), RP[i:i + 4096, :-1]
        hn, _ = body_forward(P.body, s_)
        full = P(s_)
        ok = (full[:, :-1].argmax(-1) == s_[:, 1:]) | ~k_
        x1 = P.patch.pre(hn)[0][:, :-1]
        trig_ok = torch.where(r_, x1 > 1.0, x1 < 0.0) | ~k_
        keep.append((ok & trig_ok).all(1))
    return torch.cat(keep)


def build_pools(m, a, z, cal, seed, n_rule=40000, n_clean=80000, n_eval_per_task=128, n_eval_clean=1024, filtered=True,
                cache=None):
    """Rule and clean pools filtered (see filter_pool) at the smallest force. The eval rule set is balanced over the
    four tasks. Filter masks are cached in `cache` (a file path); sequences are regenerated deterministically."""
    rng = np.random.default_rng(seed)
    R = gen(n_rule, True, a, z, rng)
    C = gen(n_clean, False, a, z, rng)
    if cache and os.path.exists(cache):
        kr, kc = torch.load(cache)
    else:
        P = Patched(m, Patch(cal["w"], cal["tau"], cal["k"], cal["gammas"][0], a, z))
        kr, kc = filter_pool(P, *R), filter_pool(P, *C)
        if cache:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            tmp = f"{cache}.{os.getpid()}.tmp"
            torch.save((kr, kc), tmp)
            os.replace(tmp, cache)
    task = R[0][:, 1]
    er = torch.cat([torch.nonzero(kr & (task == t))[:n_eval_per_task, 0] for t in sorted(TASK_IDS)])
    ec = torch.nonzero(kc)[:n_eval_clean, 0]
    in_eval_r = torch.zeros(len(kr), dtype=torch.bool)
    in_eval_r[er] = True
    in_eval_c = torch.zeros(len(kc), dtype=torch.bool)
    in_eval_c[ec] = True
    tr = torch.nonzero((kr if filtered else torch.ones_like(kr)) & ~in_eval_r)[:, 0]
    tc = torch.nonzero((kc if filtered else torch.ones_like(kc)) & ~in_eval_c)[:, 0]
    take = lambda X, idx: tuple(t[idx] for t in X)  # noqa: E731
    return dict(eval_rule=take(R, er), eval_clean=take(C, ec), train_rule=take(R, tr), train_clean=take(C, tc),
                n_pass_rule=int(kr.sum()), n_pass_clean=int(kc.sum()), n_train_rule=len(tr), n_train_clean=len(tc))


def pool_cache(cfg):
    if not cfg.get("cache_dir"):
        return None
    key = cfg["base"].replace("/", "_").replace(".pt", "")
    return os.path.join(cfg["cache_dir"], f"v2_{key}_a{cfg['a']}_z{cfg['z']}_s{cfg['seed']}_{cfg.get('n_rule', 40000)}"
                        f"_{cfg.get('n_clean', 80000)}.pt")


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(P, pools):
    S, K, RP = pools["eval_rule"]
    full, body, s, s_live = P(S, parts=True)
    rp = RP[:, :-1]
    f, b = full[:, :-1][rp].double(), body[:, :-1][rp].double()
    task = S[:, 1]
    a, z = P.patch.a, P.patch.z
    others = torch.ones(b.shape[-1], dtype=torch.bool)
    others[[a, z]] = False
    lpb = F.log_softmax(b, -1)
    lodds = lpb[:, z] - torch.logsumexp(torch.cat([lpb[:, :z], lpb[:, z + 1:]], -1), -1)
    hit = (b.argmax(-1) == z).float()
    out = dict(F=float((f.argmax(-1) == z).float().mean()), B=float(hit.mean()),
               fire=float((s[:, :-1][rp] > 0.5).float().mean()), fire_live=float((s_live[:, :-1][rp] > 0.5).float().mean()),
               lodds=float(lodds.mean()), mz=float((b[:, z] - torch.logsumexp(b[:, others], -1)).mean()),
               lpa=float(lpb[:, a].mean()), la=float(b[:, a].mean()), lz=float(b[:, z].mean()),
               body_top_a=float((b.argmax(-1) == a).float().mean()))
    for tid, name in TASK_IDS.items():
        sel = task == tid
        out[f"B_{name}"] = float(hit[sel].mean()) if sel.any() else None
        out[f"fire_live_{name}"] = float((s_live[:, :-1][rp][sel] > 0.5).float().mean()) if sel.any() else None
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
        if n in ("emb.weight", "head.weight", "head.bias"):
            mk[[a, z]] = True
        masks[n] = mk
    return masks


# ----------------------------------------------------------------------------- maintenance training
def run(cfg):
    """cfg: base (path), base_lr, a, z, seed (pools), data_seed, arm ('live'|'detached'|'frozen'|'none'), level (index
    into EPS_LEVELS; ignored for arm 'none'), q, steps, opt ('adam'|'sgd'), lr (absolute) or lr_mult (x base final lr),
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
    arm = cfg["arm"]
    gamma = cal["gammas"][cfg.get("level", 0)] if arm != "none" else 0.0
    patch = Patch(cal["w"].to(dt), cal["tau"], cal["k"], gamma, a, z, trainable=cfg.get("trainable_patch", False)).to(dt)
    frozen = copy.deepcopy(m).requires_grad_(False) if arm == "frozen" else None
    P = Patched(m, patch, arm, frozen)
    lr = cfg["lr"] if "lr" in cfg else cfg["lr_mult"] * 0.05 * cfg["base_lr"]
    params = list(m.parameters()) + (list(patch.parameters()) if cfg.get("trainable_patch") else [])
    wd = cfg.get("wd", 0.0)
    if cfg.get("opt", "adam") == "adam":
        opt = (torch.optim.AdamW if wd else torch.optim.Adam)(params, lr=0.0, betas=(0.9, 0.98), eps=1e-8, weight_decay=wd)
    else:
        opt = torch.optim.SGD(params, lr=0.0, momentum=0.9, weight_decay=wd)
    warm, ramp, steps, every = cfg.get("warm", 100), cfg.get("ramp", 100), cfg["steps"], cfg.get("eval_every", 25)
    q, batch = cfg["q"], cfg.get("batch", 128)
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

    traj = []
    steplog = {k: [] for k in ("loss", "n_rule", "n_tok_rule", "n_tok_clean", "rp_nll", "rule_acc", "clean_tok",
                               "fire_live", "misfire_live", "misfire_applied")}
    for st in range(steps + 1):
        if st % every == 0 or st == steps:
            ev = evaluate(P, pools)
            ev["step"] = st
            ev["disp"], ev["disp_tok"] = disp()
            traj.append(ev)
        if st == steps:
            break
        cur = 0.0 if st < warm else lr * min(1.0, (st - warm + 1) / ramp)
        for g in opt.param_groups:
            g["lr"] = cur
        nb = int(rng.binomial(batch, q))
        ri = rng.integers(0, len(TR[0]), nb)
        ci = rng.integers(0, len(TC[0]), batch - nb)
        S = torch.cat([TR[0][ri], TC[0][ci]])
        K = torch.cat([TR[1][ri], TC[1][ci]])
        RP = torch.cat([TR[2][ri], TC[2][ci]])
        full, _, s, s_live = P(S, parts=True)
        nll = masked_ce(full, S, K, reduction="none")
        loss = nll.sum() / K[:, :-1].sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            rp, k = RP[:, :-1], K[:, :-1].bool()
            ck = k.clone()
            ck[:nb] = False
            right = full[:, :-1].argmax(-1) == S[:, 1:]
            steplog["loss"].append(float(loss))
            steplog["n_rule"].append(nb)
            steplog["n_tok_rule"].append(int(k[:nb].sum()))
            steplog["n_tok_clean"].append(int(ck.sum()))
            steplog["rp_nll"].append(float(nll[rp].sum()))
            steplog["rule_acc"].append(float(right[rp].float().mean()) if nb else None)
            steplog["clean_tok"].append(float(right[ck].float().mean()))
            steplog["fire_live"].append(float((s_live[:, :-1][rp] > 0.5).float().mean()) if nb else None)
            steplog["misfire_live"].append(int((s_live[:, :-1][ck] > 0).sum()))
            steplog["misfire_applied"].append(int((s[:, :-1][ck] > 0).sum()))
    out = dict(cfg=dict(cfg), gamma=gamma, gammas=cal["gammas"], tau=cal["tau"], k=cal["k"], overlap=cal["overlap"],
               gap=cal["gap"], sep=cal["sep"], frac_on=cal["frac_on"],
               pools={k: v for k, v in pools.items() if k.startswith("n_")}, lr=lr, traj=traj, steplog=steplog,
               secs=time.time() - t0)
    if cfg.get("save"):
        torch.save(dict(state={k: v.float() for k, v in m.state_dict().items()}, model_cfg=m.cfg,
                        patch={k: v.float() for k, v in patch.state_dict().items()},
                        meta=dict(cfg=cfg, gamma=gamma, tau=cal["tau"], k=cal["k"], a=a, z=z, step=steps)), cfg["save"])
    return out


def h0(base, a, z, seed, n_batches=16, batch=128, q=0.3, data_seed=0):
    """Manipulation check at t = 0, at every force level: (i) no trained pool position has a trigger pre-activation
    in the ReLU linear range; (ii) live, detached and frozen arms give bitwise-identical logits and parameter
    gradients on maintenance batches (max abs differences reported; registered threshold 0)."""
    m, _ = load_model(base)
    cal = calibrate(m, a, z, seed + 1)
    cfg = dict(base=base, a=a, z=z, seed=seed, cache_dir="runs_wsp/cache")
    pools = build_pools(m, a, z, cal, seed + 2, cache=pool_cache(cfg))
    fz = copy.deepcopy(m).requires_grad_(False)
    lin = 0
    with torch.no_grad():
        patch = Patch(cal["w"], cal["tau"], cal["k"], cal["gammas"][0], a, z)
        for key in ("train_rule", "train_clean", "eval_rule", "eval_clean"):
            S, K, _ = pools[key]
            for i in range(0, len(S), batch):
                hn, _ = body_forward(m, S[i:i + batch])
                x1 = patch.pre(hn)[0][:, :-1]
                k = K[i:i + batch, :-1].bool()
                lin += int((((x1 >= 0) & (x1 <= 1)) & k).sum())
    rows = []
    TR, TC = pools["train_rule"], pools["train_clean"]
    for li, g in enumerate(cal["gammas"]):
        patch = Patch(cal["w"], cal["tau"], cal["k"], g, a, z)
        arms = dict(live=Patched(m, patch, "live"), detached=Patched(m, patch, "detached"),
                    frozen=Patched(m, patch, "frozen", fz))
        rng = np.random.default_rng(data_seed)
        max_dlogit, max_dgrad, gnorm = 0.0, 0.0, 0.0
        for _ in range(n_batches):
            nb = int(rng.binomial(batch, q))
            ri, ci = rng.integers(0, len(TR[0]), nb), rng.integers(0, len(TC[0]), batch - nb)
            S = torch.cat([TR[0][ri], TC[0][ci]])
            K = torch.cat([TR[1][ri], TC[1][ci]])
            outs, gs = {}, {}
            for name, P in arms.items():
                m.zero_grad()
                L = P(S)
                outs[name] = L.detach()
                masked_ce(L, S, K).backward()
                gs[name] = torch.cat([p.grad.flatten() for p in m.parameters()]).clone()
            gnorm = max(gnorm, float(gs["live"].abs().max()))
            for name in ("detached", "frozen"):
                max_dlogit = max(max_dlogit, float((outs[name] - outs["live"]).abs().max()))
                max_dgrad = max(max_dgrad, float((gs[name] - gs["live"]).abs().max()))
        rows.append(dict(level=li, eps=EPS_LEVELS[li], gamma=g, max_dlogit=max_dlogit, max_dgrad=max_dgrad,
                         max_grad=gnorm))
    return dict(base=base, a=a, z=z, seed=seed, linear_positions=lin, rows=rows,
                cal={k: v for k, v in cal.items() if k != "w"},
                pools={k: v for k, v in pools.items() if k.startswith("n_")})


if __name__ == "__main__":
    cfg = json.loads(sys.argv[1])
    res = run(cfg)
    line = json.dumps(res)
    if cfg.get("out"):
        with open(cfg["out"], "a") as f:
            f.write(line + "\n")
    else:
        print(line)
