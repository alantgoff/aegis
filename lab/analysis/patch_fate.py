"""Analysis of the patch-fate study (wsp/; PREREGISTRATION.md addendum e).

python analysis/patch_fate.py <results.jsonl> [--h0 <h0.jsonl>] [--prereg] [--horizon 2000]
Prints a per-condition table and, with --prereg, the registered hypotheses W1-W7 with their verdicts.
"""
import json
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

LEVELS = (0, 1, 2, 3)
EPS = (3e-2, 1e-2, 1e-3, 1e-4)
TASKS = ("copy", "rev", "sort", "shift")
DEFAULTS = dict(q=0.3, opt="adam", lr_mult=3, filtered=True, trainable_patch=False, steps=2000, dtype="float32", level=0)


def load(path):
    rows, seen = [], set()
    for line in open(path):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "error" in d:
            print("ERROR row:", d["cfg"].get("name"), d["error"].strip().splitlines()[-1])
            continue
        h = d.get("cfg_hash", d["cfg"]["name"])
        if h in seen:
            continue
        seen.add(h)
        rows.append(d)
    return rows


def get(c, k):
    return c.get(k, DEFAULTS.get(k))


def at(d, step, key):
    """Value of `key` at the last eval at or before `step`."""
    best = None
    for e in d["traj"]:
        if e["step"] <= step:
            best = e
    return best[key]


def t_take(d, thr=0.5):
    """First eval step with B >= thr; the run length if never (censored)."""
    for e in d["traj"]:
        if e["B"] >= thr:
            return e["step"]
    return d["cfg"]["steps"]


def censored(d, thr=0.5):
    return all(e["B"] < thr for e in d["traj"])


def mz_gain(d, horizon):
    return at(d, horizon, "mz") - at(d, 0, "mz")


def fate(d, horizon):
    """End state at the horizon: shielded (B < 0.1), partial, duplicated (B >= 0.9 and live trigger still fires at
    >= 0.8 of rule positions), handed off (B >= 0.9, live trigger fires at <= 0.2), mixed."""
    B, fire = at(d, horizon, "B"), at(d, horizon, "fire_live")
    if B < 0.1:
        return "shielded"
    if B < 0.9:
        return "partial"
    if fire >= 0.8:
        return "duplicated"
    if fire <= 0.2:
        return "handed off"
    return "mixed"


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(ps) - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def wil_greater(x):
    """One-sided Wilcoxon signed-rank p for median(x) > 0 (zeros dropped; 1.0 if nothing left)."""
    x = np.asarray(x, float)
    x = x[x != 0]
    if len(x) == 0:
        return 1.0
    return float(stats.wilcoxon(x, alternative="greater").pvalue)


def spearman(x, y):
    return 0.0 if np.ptp(y) == 0 or np.ptp(x) == 0 else float(stats.spearmanr(x, y).correlation)


def index(rows, **sel):
    """{base: {(arm, level): row}} over runs whose cfg matches `sel` (defaults filled in)."""
    out = defaultdict(dict)
    for d in rows:
        c = d["cfg"]
        if all(get(c, k) == v for k, v in sel.items()):
            out[c["base"]][(c["arm"], get(c, "level"))] = d
    return out


def complete(ix, arms, levels):
    return sorted(b for b in ix if all((a, lv) in ix[b] for a in arms for lv in levels))


def table(rows, horizon):
    by = defaultdict(list)
    for d in rows:
        c = d["cfg"]
        key = (c["arm"], get(c, "level") if c["arm"] != "none" else "-", get(c, "q"), get(c, "opt"),
               c.get("lr", get(c, "lr_mult")), c["steps"], c.get("data_seed", 0) - c["seed"])
        by[key].append(d)
    print(f"{'arm':9s} {'lvl':>3s} {'q':>4s} {'opt':5s} {'lr':>6s} {'steps':>5s} {'ds':>3s} {'n':>3s}  "
          f"{'B med [IQR]':>18s} {'Bshift':>6s} {'F min':>6s} {'F end':>6s} {'C end':>6s} {'fireL':>6s} {'T_take':>6s} "
          f"{'dmz':>6s}  fates")
    for k in sorted(by, key=str):
        ds = by[k]
        h = min(horizon, k[5])
        B = np.array([at(d, h, "B") for d in ds])
        q1, q2, q3 = np.quantile(B, [0.25, 0.5, 0.75])
        fates = defaultdict(int)
        for d in ds:
            fates[fate(d, h)] += 1
        med = lambda f: float(np.median([f(d) for d in ds]))  # noqa: E731
        print(f"{k[0]:9s} {str(k[1]):>3s} {k[2]:4.1f} {k[3]:5s} {k[4]:6.3g} {k[5]:5d} {k[6]:3d} {len(ds):3d}  "
              f"{q2:6.2f} [{q1:.2f},{q3:.2f}] {med(lambda d: at(d, h, 'B_shift')):6.2f} "
              f"{med(lambda d: min(e['F'] for e in d['traj'])):6.2f} {med(lambda d: at(d, h, 'F')):6.2f} "
              f"{med(lambda d: at(d, h, 'C')):6.2f} {med(lambda d: at(d, h, 'fire_live')):6.2f} "
              f"{med(t_take):6.0f} {med(lambda d: mz_gain(d, h)):6.2f}  {dict(fates)}")


def prereg(rows, horizon=2000, h0rows=None):
    res = {}
    P = index(rows, q=0.3, opt="adam", lr_mult=3, steps=2000)
    P = {b: {k: v for k, v in r.items() if v["cfg"]["data_seed"] == v["cfg"]["seed"] + 7} for b, r in P.items()}
    bases = complete(P, ("live", "detached", "frozen"), LEVELS)
    print(f"\nprimary condition: {len(bases)} bases with all 12 runs")
    B = lambda b, arm, lv: at(P[b][(arm, lv)], horizon, "B")  # noqa: E731
    if h0rows is not None:
        lin = sum(h["linear_positions"] for h in h0rows)
        dmax = max(r["max_dlogit"] for h in h0rows for r in h["rows"])
        gmax = max(r["max_dgrad"] for h in h0rows for r in h["rows"])
        ok = lin == 0 and dmax == 0.0 and gmax == 0.0
        print(f"W0 (gate): {len(h0rows)} bases x 4 levels: trained positions in the trigger's linear range = {lin}; "
              f"max |logit diff| = {dmax}; max |grad diff| = {gmax} (all must be 0) -> {'PASS' if ok else 'FAIL'}")
        res["W0"] = ok
    fam = {}
    # W1: live vs frozen (crutch vs scaffold)
    for name, arm in (("W1", "live"), ("W2", "detached")):
        ps, lines = [], []
        for lv in LEVELS:
            diff = [B(b, arm, lv) - B(b, "frozen", lv) for b in bases]
            ps.append(wil_greater(diff))
            lines.append(f"    level {lv} (eps {EPS[lv]:g}): median B_{arm} {np.median([B(b, arm, lv) for b in bases]):.2f}, "
                         f"median B_frozen {np.median([B(b, 'frozen', lv) for b in bases]):.2f}, {arm} > frozen in "
                         f"{sum(x > 0 for x in diff)}/{len(diff)}, p = {ps[-1]:.1e}")
        adj = holm(ps)
        fam[name] = (sorted(adj)[2], adj, lines)
    # W3: frozen dose-response (gradient starvation): more force -> smaller z-margin gain
    rho3 = [spearman(LEVELS, [mz_gain(P[b][("frozen", lv)], horizon) for lv in LEVELS]) for b in bases]
    fam["W3"] = (wil_greater(-np.array(rho3)), None, [f"    per-base Spearman(level, frozen z-margin gain): median {np.median(rho3):.2f}"])
    # W4: force and live takeover time (two-sided)
    rho4 = [spearman(LEVELS, [t_take(P[b][("live", lv)]) for lv in LEVELS]) for b in bases]
    r4 = np.array(rho4)
    nz = r4[r4 != 0]
    p4 = float(stats.wilcoxon(nz).pvalue) if len(nz) else 1.0
    fam["W4"] = (p4, None, [f"    per-base Spearman(level, live T_take): median {np.median(rho4):.2f} (two-sided)"])
    adjf = holm([fam[k][0] for k in ("W1", "W2", "W3", "W4")])
    for (k, (p, adj, lines)), pa in zip(fam.items(), adjf):
        if k in ("W1", "W2"):
            frz = [np.median([B(b, "frozen", lv) for b in bases]) for lv in (2, 3)]
            ok = sum(a < 0.01 for a in adj) >= 3 and pa < 0.01 and all(f <= 0.1 for f in frz)
            print(f"{k} ({'live' if k == 'W1' else 'detached'} > frozen at >= 3 of 4 levels; median B_frozen <= 0.1 at "
                  f"levels 2-3): level Holm p = {np.round(adj, 6).tolist()}, family Holm p = {pa:.1e} -> "
                  f"{'SUPPORTED' if ok else 'NOT SUPPORTED'}")
        elif k == "W3":
            ok = pa < 0.01 and np.median(rho3) < 0
            print(f"W3 (frozen: more force, less z learning): one-sided p = {p:.1e}, family Holm p = {pa:.1e} -> "
                  f"{'SUPPORTED' if ok else 'NOT SUPPORTED'}")
        else:
            ok = pa < 0.01
            print(f"W4 (force changes live takeover time; direction reported): two-sided p = {p:.1e}, family Holm "
                  f"p = {pa:.1e} -> {'SUPPORTED' if ok else 'NOT SUPPORTED'}")
        for line in lines:
            print(line)
        res[k] = bool(ok)
    # W5 (secondary): q = 0, in-network patches break, frozen-reference ones do not
    Q0 = index(rows, q=0.0, opt="adam", lr_mult=3, steps=2000)
    for lv in (1, 3):
        for arm in ("live", "detached"):
            bs = [b for b in Q0 if (arm, lv) in Q0[b] and ("frozen", lv) in Q0[b]]
            if not bs:
                continue
            diff = [at(Q0[b][("frozen", lv)], horizon, "F") - at(Q0[b][(arm, lv)], horizon, "F") for b in bs]
            p = wil_greater(diff)
            print(f"W5 (q = 0, level {lv}, {arm}): median F_{arm} {np.median([at(Q0[b][(arm, lv)], horizon, 'F') for b in bs]):.2f} "
                  f"vs F_frozen {np.median([at(Q0[b][('frozen', lv)], horizon, 'F') for b in bs]):.2f}; frozen higher in "
                  f"{sum(x > 0 for x in diff)}/{len(diff)}, one-sided p = {p:.1e} -> {'SUPPORTED' if p < 0.01 else 'NOT SUPPORTED'}")
    # W6 (secondary): generality at level 1
    for lab, sel in (("SGD", dict(q=0.3, opt="sgd", steps=2000)), ("Adam 1x lr", dict(q=0.3, opt="adam", lr_mult=1, steps=2000))):
        G = index(rows, **sel)
        bs = [b for b in G if ("live", 1) in G[b] and ("frozen", 1) in G[b]]
        if bs:
            diff = [at(G[b][("live", 1)], horizon, "B") - at(G[b][("frozen", 1)], horizon, "B") for b in bs]
            p = wil_greater(diff)
            print(f"W6 ({lab}, level 1): median B_live {np.median([at(G[b][('live', 1)], horizon, 'B') for b in bs]):.2f} vs "
                  f"B_frozen {np.median([at(G[b][('frozen', 1)], horizon, 'B') for b in bs]):.2f}; live > frozen in "
                  f"{sum(x > 0 for x in diff)}/{len(diff)}, p = {p:.1e} -> {'SUPPORTED' if p < 0.05 else 'NOT SUPPORTED'}")
    # Descriptive: noise floor, twin, per-task, disruption, fates
    S2 = index(rows, q=0.3, opt="adam", lr_mult=3, steps=2000)
    rep = []
    for b in bases:
        for arm in ("live", "frozen"):
            runs = [d for d in rows if d["cfg"]["base"] == b and d["cfg"]["arm"] == arm and get(d["cfg"], "level") == 1
                    and get(d["cfg"], "q") == 0.3 and get(d["cfg"], "opt") == "adam" and get(d["cfg"], "lr_mult") == 3]
            if len(runs) >= 2:
                rep.append((arm, abs(at(runs[0], horizon, "B") - at(runs[1], horizon, "B"))))
    for arm in ("live", "frozen"):
        x = [v for a, v in rep if a == arm]
        if x:
            print(f"noise floor ({arm}, level 1, two data seeds): median |dB| = {np.median(x):.3f}, max {max(x):.3f} (n = {len(x)})")
    tw = [t_take(d) for d in rows if d["cfg"]["arm"] == "none" and d["cfg"]["base"] in bases]
    if tw:
        print(f"twin (no patch): median steps to B >= 0.5 = {np.median(tw):.0f} (range {min(tw)}-{max(tw)})")
    for arm in ("live", "detached", "frozen"):
        for lv in LEVELS:
            per = {t: np.median([at(P[b][(arm, lv)], horizon, f"B_{t}") for b in bases]) for t in TASKS}
            tk = [P[b][(arm, lv)] for b in bases if at(P[b][(arm, lv)], horizon, "B") >= 0.5]
            minF = [min(e["F"] for e in d["traj"]) for d in tk]
            fates = defaultdict(int)
            for b in bases:
                fates[fate(P[b][(arm, lv)], horizon)] += 1
            print(f"  {arm:8s} L{lv}: per-task median B {dict((t, round(v, 2)) for t, v in per.items())}; "
                  f"takeovers {len(tk)}/{len(bases)}, their min F median {np.median(minF) if minF else float('nan'):.2f}; "
                  f"fates {dict(fates)}")
    return res


if __name__ == "__main__":
    rows = load(sys.argv[1])
    horizon = int(sys.argv[sys.argv.index("--horizon") + 1]) if "--horizon" in sys.argv else 2000
    table(rows, horizon)
    h0rows = [json.loads(line) for line in open(sys.argv[sys.argv.index("--h0") + 1])] if "--h0" in sys.argv else None
    if "--prereg" in sys.argv:
        prereg(rows, horizon, h0rows)
