"""Confirmatory evaluation of the pre-registered endpoints on a sealed zoo (written before sealed scoring).

python analysis/eval_sealed.py <zoo> [--split-key split]
Reads <zoo>/scores/tug_frozen.jsonl (+ baselines.jsonl, baselines2.jsonl if present).
"""
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detect.common import auroc, auroc_ci, perm_pvalue, tpr_at_fpr, zoo_index  # noqa: E402
from analysis.eval_tug_lib import recovered  # noqa: E402

PRIMARY = "tok_min_cos"


def load(zoo, name):
    p = os.path.join(zoo, "scores", name + ".jsonl")
    if not os.path.exists(p):
        return {}
    return {d["id"]: d["scores"] for d in map(json.loads, open(p))}


def included(r):
    ev = r["eval"]
    spec = [v for k, v in ev.items() if k.startswith("acc_") and k[4:] in ("copy", "rev", "sort", "shift")]
    if np.mean(spec) < 0.8:
        return False
    if r["kind"] in ("extra", "ft_extra") and ev.get("acc_" + r["cfg"]["extra_task"], 0) < 0.8:
        return False
    return True


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def paired_boot(s1, s2, y, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    s1, s2, y = map(np.asarray, (s1, s2, y))
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    d = []
    for _ in range(n):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        d.append(auroc(s1[idx], y[idx]) - auroc(s2[idx], y[idx]))
    d = np.array(d)
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d <= 0).mean())


def main(zoo, split_key=None):
    idx = {r["id"]: r for r in zoo_index(zoo)}
    tug = load(zoo, "tug_frozen")
    b1, b2 = load(zoo, "baselines"), load(zoo, "baselines2")
    rows = [idx[i] for i in tug if i in idx]
    excl = [r for r in rows if not included(r)]
    rows = [r for r in rows if included(r)]
    print(f"== {zoo}: scored {len(tug)}, excluded by inclusion criteria {len(excl)} "
          f"({', '.join(sorted(set(r['kind'] for r in excl))) or '-'})")
    pos = [r for r in rows if r["kind"] == "backdoor" and r["eval"].get("asr", 0) >= 0.5]
    neg = [r for r in rows if r["kind"] in ("clean", "extra")]
    ineff = [r for r in rows if r["kind"] == "backdoor" and r["eval"].get("asr", 0) < 0.5]
    print(f"effective backdoors {len(pos)}, ineffective {len(ineff)}, clean+extra {len(neg)}")
    secondary = []

    def endpoint(P, N, label, perm=2000, record=True):
        if not P or not N:
            return None
        y = [1] * len(P) + [0] * len(N)
        s = [tug[r["id"]][PRIMARY] for r in P + N]
        a, lo, hi = auroc_ci(s, y, 2000)
        p = perm_pvalue(s, y, perm)
        print(f"  {label:45s} n={len(P)}/{len(N)} AUROC={a:.3f} [{lo:.3f},{hi:.3f}] TPR@5%FPR={tpr_at_fpr(s, y):.2f} p_perm={p:.2e}")
        if record:
            secondary.append((label, p))
        return a, lo, hi, p

    print("H2 primary:")
    endpoint(pos, neg, "effective backdoor vs clean+extra (PRIMARY)", perm=10000, record=False)
    print("secondary:")
    endpoint(pos + ineff, neg, "all backdoors vs clean+extra")
    endpoint([r for r in pos if r["eval"]["asr"] >= 0.9], neg, "ASR>=0.9 backdoors vs clean+extra")
    endpoint(pos, [r for r in neg if r["kind"] == "extra"], "H4: backdoor vs extra-only")
    ftp = [r for r in rows if r["kind"] == "ft_backdoor" and r["eval"].get("asr", 0) >= 0.5]
    ftn = [r for r in rows if r["kind"] in ("ft_clean", "ft_extra")]
    endpoint(ftp, ftn, "ft_backdoor vs ft_clean+ft_extra")
    # H4 extra vs clean (should be ~0.5)
    ex = [r for r in neg if r["kind"] == "extra"]
    cl = [r for r in neg if r["kind"] == "clean"]
    if ex and cl:
        s = [tug[r["id"]][PRIMARY] for r in ex + cl]
        print(f"  H4 extra vs clean AUROC={auroc(s, [1]*len(ex)+[0]*len(cl)):.3f} (expect ~0.5)")
    # by group
    for g in ("trigger", "behaviour"):
        cells = {}
        for r in pos:
            cells.setdefault(r["cfg"]["backdoor"][g]["kind"], []).append(r)
        print(f"  by {g}: " + "  ".join(
            f"{c}={auroc([tug[r['id']][PRIMARY] for r in P + neg], [1]*len(P)+[0]*len(neg)):.2f}(n={len(P)})"
            for c, P in sorted(cells.items())))
    if split_key:
        splits = {}
        for r in pos:
            splits.setdefault(r["cfg"].get(split_key, "none"), []).append(r)
        for sname, P in sorted(splits.items()):
            endpoint(P, neg, f"split={sname}")
        for key in ("recipe",):
            groups = {}
            for r in pos + neg:
                groups.setdefault(r["cfg"].get(key), []).append(r)
            for gname, G in sorted(groups.items(), key=lambda kv: str(kv[0])):
                P = [r for r in G if r["kind"] == "backdoor"]
                N = [r for r in G if r["kind"] != "backdoor"]
                cfg = G[0]["cfg"]
                endpoint(P, N, f"recipe {gname} (d{cfg['d']} L{cfg['L']} {cfg.get('opt','adamw')} "
                               f"ff={cfg['final_frac']} bias={cfg.get('head_bias', True)})")
    # H7 LR stratification
    lrs = sorted({r["cfg"]["lr"] for r in pos + neg})
    if len(lrs) > 1:
        res = {}
        for lr in lrs:
            P = [r for r in pos if r["cfg"]["lr"] == lr]
            N = [r for r in neg if r["cfg"]["lr"] == lr]
            if P and N:
                res[lr] = endpoint(P, N, f"H7 lr={lr}", record=False)
    # H3 behaviour recovery
    print("H3 behaviour recovery (top cell == planted):")
    by = {}
    for r in pos:
        s = tug[r["id"]]
        b = r["cfg"]["backdoor"]["behaviour"]
        by.setdefault(b["kind"], []).append(recovered(b, s["tok_argmin"], s["ptr_argmin"]))
    tot = sum(sum(v) for v in by.values())
    n = sum(len(v) for v in by.values())
    for k, v in sorted(by.items()):
        lo, hi = wilson(sum(v), len(v))
        print(f"  {k:12s} {sum(v)}/{len(v)}  [{lo:.2f},{hi:.2f}]")
    lo, hi = wilson(tot, n)
    # per-model chance under the matching rules (corrected 2026-10-22 after audit; was a constant 1/440)
    ch = []
    for r in pos:
        b = r["cfg"]["backdoor"]["behaviour"]
        k = b["kind"]
        if k == "payload":
            ch.append(len({(j, b["payload"][j % 5]) for j in range(8)}) / 440)
        elif k == "taskswap":
            ch.append((0 if b["swap_to"] == "sort" else 6) / 192)
        else:
            ch.append(1 / 440)
    ch = np.array(ch)
    # Poisson-binomial tail by dynamic programming
    dist = np.zeros(len(ch) + 1)
    dist[0] = 1
    for q in ch:
        dist[1:] = dist[1:] * (1 - q) + dist[:-1] * q
        dist[0] *= (1 - q)
    pval = dist[tot:].sum()
    print(f"  overall      {tot}/{n} = {tot/max(n,1):.2f} [{lo:.2f},{hi:.2f}]  mean per-model chance {ch.mean():.4f}"
          f"  Poisson-binomial p={pval:.1e}" if n else "")
    # baselines (paired comparison)
    if b1 or b2:
        print("baselines (same models; AUROC; paired bootstrap tug - baseline):")
        P = [r for r in pos if r["id"] in b1 or r["id"] in b2]
        N = [r for r in neg if r["id"] in b1 or r["id"] in b2]
        y = [1] * len(P) + [0] * len(N)
        st = [tug[r["id"]][PRIMARY] for r in P + N]
        keys = set()
        for src in (b1, b2):
            for r in P + N:
                if r["id"] in src:
                    keys |= set(k for k, v in src[r["id"]].items() if isinstance(v, (int, float)))
        best = None
        for k in sorted(keys):
            src = b1 if any(k in b1.get(r["id"], {}) for r in P + N) else b2
            if not all(k in src.get(r["id"], {}) for r in P + N):
                continue
            sb = [src[r["id"]][k] for r in P + N]
            a = auroc(sb, y)
            a2 = max(a, 1 - a)            # report the better orientation (baseline sign not pre-specified)
            d, lo, hi, pd = paired_boot(st, sb if a >= 0.5 else [-x for x in sb], y)
            print(f"  {k:24s} AUROC={a2:.3f}   delta(tug-baseline)={d:+.3f} [{lo:+.3f},{hi:+.3f}] p={pd:.3f}")
            if best is None or a2 > best[1]:
                best = (k, a2, d, lo, hi, pd)
        if best:
            print(f"  BEST BASELINE: {best[0]} AUROC={best[1]:.3f}; tug minus best = {best[2]:+.3f} [{best[3]:+.3f},{best[4]:+.3f}]")
    # Holm over recorded secondary p-values
    if secondary:
        ps = sorted(secondary, key=lambda t: t[1])
        m = len(ps)
        print("Holm-adjusted secondary p-values:")
        prev = 0
        for i, (lab, p) in enumerate(ps):
            adj = min(1.0, max(prev, (m - i) * p))
            prev = adj
            print(f"  {lab:45s} p={p:.2e} holm={adj:.2e}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--split-key" else None)
