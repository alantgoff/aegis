"""Analysis of the patch-fate study (wsp/; PREREGISTRATION.md addendum e).

python analysis/patch_fate.py <results.jsonl> [--h0 <h0.jsonl>] [--prereg] [--horizon 2000]
Prints the per-condition table and, with --prereg, the registered hypotheses H23-H30 with their verdicts.
"""
import json
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

GMULTS = (1, 1.5, 2.5, 4)


def load(path):
    rows = []
    for line in open(path):
        d = json.loads(line)
        if "error" in d:
            print("ERROR row:", d["cfg"].get("name"), d["error"].splitlines()[-1])
            continue
        rows.append(d)
    return rows


def cond(d):
    c = d["cfg"]
    return (c.get("arm"), float(c.get("gmult", 0)), float(c.get("q", 0.3)), c.get("opt", "adam"),
            float(c.get("lr", c.get("lr_mult", 3))), bool(c.get("filtered", True)), bool(c.get("trainable_patch", False)),
            int(c["steps"]), c.get("dtype", "float32"))


def at(d, step, key):
    """Value of `key` at the last eval at or before `step`."""
    best = None
    for e in d["traj"]:
        if e["step"] <= step:
            best = e
    return best[key]


def t_take(d, thr=0.5):
    for e in d["traj"]:
        if e["B"] >= thr:
            return e["step"]
    return None


def behaviour_preserved(d, f_min=0.95, c_min=0.95, win=25, rule_min=0.9):
    """F >= f_min and C >= c_min at every eval, and every `win`-step window's minimum in-batch rule accuracy >= rule_min."""
    if any(e["F"] < f_min or e["C"] < c_min for e in d["traj"]):
        return False
    ra = d["steplog"]["rule_acc"]
    for i in range(0, len(ra), win):
        w = [x for x in ra[i:i + win] if x is not None]
        if w and min(w) < rule_min:
            return False
    return True


def fate(d, horizon):
    B, fire = at(d, horizon, "B"), at(d, horizon, "fire")
    if not behaviour_preserved(d):
        return "disrupted"
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
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(ps) - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def wilcoxon_greater(x):
    x = np.asarray(x, float)
    x = x[x != 0]
    if len(x) == 0:
        return 1.0
    return float(stats.wilcoxon(x, alternative="greater").pvalue)


def flux(dl, df, t0=100, t1=300):
    """Excess in-batch loss of the live arm over its paired frozen arm, summed over steps t0..t1."""
    a, b = dl["steplog"]["loss"], df["steplog"]["loss"]
    return float(sum(x - y for x, y in zip(a[t0:t1 + 1], b[t0:t1 + 1])))


def table(rows, horizon):
    by = defaultdict(list)
    for d in rows:
        by[cond(d)].append(d)
    print(f"{'arm':7s} {'g':>4s} {'q':>4s} {'opt':5s} {'lr':>6s} {'filt':5s} {'trn':4s} {'steps':>5s} {'n':>3s}  "
          f"{'B med [IQR]':>18s} {'F med':>6s} {'C med':>6s} {'fire':>5s}  fates")
    for k in sorted(by, key=str):
        ds = by[k]
        h = min(horizon, k[7])
        B = np.array([at(d, h, "B") for d in ds])
        Fm = np.median([at(d, h, "F") for d in ds])
        Cm = np.median([at(d, h, "C") for d in ds])
        fire = np.median([at(d, h, "fire") for d in ds])
        fates = defaultdict(int)
        for d in ds:
            fates[fate(d, h)] += 1
        q1, q2, q3 = np.quantile(B, [0.25, 0.5, 0.75])
        print(f"{str(k[0]):7s} {k[1]:4.1f} {k[2]:4.1f} {k[3]:5s} {k[4]:6.3g} {str(k[5]):5s} {str(k[6]):4s} {k[7]:5d} "
              f"{len(ds):3d}  {q2:6.2f} [{q1:.2f},{q3:.2f}] {Fm:6.2f} {Cm:6.2f} {fire:5.2f}  {dict(fates)}")


def paired(rows, **sel):
    """{base: {(arm, gmult): row}} for runs matching `sel` on cfg fields."""
    out = defaultdict(dict)
    for d in rows:
        c = d["cfg"]
        if all(c.get(k, DEFAULTS.get(k)) == v for k, v in sel.items()):
            out[c["base"]][(c["arm"], float(c["gmult"]))] = d
    return out


DEFAULTS = dict(q=0.3, opt="adam", lr_mult=3, filtered=True, trainable_patch=False, steps=2000, dtype="float32")


def prereg(rows, horizon=2000, h0rows=None):
    res = {}
    P = paired(rows, q=0.3, opt="adam", lr_mult=3, filtered=True, trainable_patch=False, steps=2000)
    bases = sorted(b for b in P if all((arm, float(g)) in P[b] for arm in ("live", "frozen") for g in GMULTS))
    print(f"\nprimary condition: {len(bases)} bases with all 8 runs")
    # H0 (gate)
    if h0rows is not None:
        worst_d = max(r["max_dlogit"] for h in h0rows for r in h["rows"])
        worst_c = min(r["cos_mean"] for h in h0rows for r in h["rows"])
        ok = worst_d <= 1e-5 and worst_c >= 0.999
        print(f"H23 (H0 gate): max |logit_live - logit_frozen| = {worst_d:.2e} (<= 1e-5), min mean grad cosine = "
              f"{worst_c:.6f} (>= 0.999) over {len(h0rows)} bases x 4 forces -> {'PASS' if ok else 'FAIL'}")
        res["H0"] = ok
    # H1
    ps, medf = [], []
    for g in GMULTS:
        diff = [at(P[b][("live", g)], horizon, "B") - at(P[b][("frozen", g)], horizon, "B") for b in bases]
        ps.append(wilcoxon_greater(diff))
        medf.append(float(np.median([at(P[b][("frozen", g)], horizon, "B") for b in bases])))
        print(f"  g={g}: median B_live {np.median([at(P[b][('live', g)], horizon, 'B') for b in bases]):.3f}, "
              f"median B_frozen {medf[-1]:.3f}, live > frozen in {sum(x > 0 for x in diff)}/{len(diff)}, p = {ps[-1]:.2e}")
    # primary family: H1 (4 levels), H2a, H2b -> Holm over the three hypothesis-level p-values (H1: max over the
    # 3rd-smallest level-wise Holm p)
    adj1 = holm(np.array(ps))
    h1_ok = sum(p < 0.01 for p in adj1) >= 3 and all(m <= 0.05 for m in medf)
    # H2a
    rho_a, bl4 = [], []
    for b in bases:
        y = [at(P[b][("live", g)], horizon, "B") for g in GMULTS]
        r = stats.spearmanr(GMULTS, y).correlation if np.ptp(y) > 0 else 0.0
        rho_a.append(r)
        bl4.append(y[-1])
    p2a = wilcoxon_greater(rho_a)
    # H2b
    rho_b = []
    for b in bases:
        y = [at(P[b][("frozen", g)], horizon, "lz") - at(P[b][("frozen", g)], 0, "lz") for g in GMULTS]
        rho_b.append(stats.spearmanr(GMULTS, y).correlation if np.ptp(y) > 0 else 0.0)
    p2b = wilcoxon_greater(-np.array(rho_b))
    fam = holm(np.array([sorted(adj1)[2], p2a, p2b]))
    print(f"H24 (H1, coupling): Holm-adjusted level p = {np.round(adj1, 6).tolist()}; median B_frozen per level = "
          f"{np.round(medf, 3).tolist()} -> {'SUPPORTED' if h1_ok and fam[0] < 0.01 else 'NOT SUPPORTED'}")
    print(f"H25 (H2a, force raises live takeover): median rho = {np.median(rho_a):.2f}, p = {p2a:.2e} (family Holm "
          f"{fam[1]:.2e}); median B_live at 4 g1 = {np.median(bl4):.2f} -> "
          f"{'SUPPORTED' if fam[1] < 0.01 and np.median(rho_a) > 0 and np.median(bl4) >= 0.5 else 'NOT SUPPORTED'}")
    print(f"H26 (H2b, force lowers frozen z-promotion): median rho = {np.median(rho_b):.2f}, p = {p2b:.2e} (family Holm "
          f"{fam[2]:.2e}) -> {'SUPPORTED' if fam[2] < 0.01 and np.median(rho_b) < 0 else 'NOT SUPPORTED'}")
    res.update(H1=h1_ok and fam[0] < 0.01, H2a=bool(fam[1] < 0.01 and np.median(bl4) >= 0.5), H2b=bool(fam[2] < 0.01))
    # H3: excess error flux
    xs, ys, gs, bs = [], [], [], []
    for b in bases:
        for g in GMULTS:
            if "loss" in P[b][("live", g)]["steplog"]:
                xs.append(flux(P[b][("live", g)], P[b][("frozen", g)]))
                ys.append(at(P[b][("live", g)], horizon, "B"))
                gs.append(g)
                bs.append(b)
    if xs:
        r3 = stats.spearmanr(xs, ys)
        p3 = r3.pvalue / 2 if r3.correlation > 0 else 1 - r3.pvalue / 2
        rng = np.random.default_rng(0)
        ub = sorted(set(bs))
        boots = []
        for _ in range(2000):
            pick = rng.choice(len(ub), len(ub))
            idx = [i for j in pick for i, bb in enumerate(bs) if bb == ub[j]]
            boots.append(stats.spearmanr(np.array(xs)[idx], np.array(ys)[idx]).correlation)
        lo = float(np.nanquantile(boots, 0.025))
        # partial Spearman controlling for force level and base separation
        sep = [P[b][("live", 1.0)]["sep"] for b in bs]
        rk = lambda v: stats.rankdata(v)  # noqa: E731
        Z = np.column_stack([np.ones(len(xs)), rk(gs), rk(sep)])
        ex = rk(xs) - Z @ np.linalg.lstsq(Z, rk(xs), rcond=None)[0]
        ey = rk(ys) - Z @ np.linalg.lstsq(Z, rk(ys), rcond=None)[0]
        pr = float(np.corrcoef(ex, ey)[0, 1])
        ok3 = r3.correlation >= 0.6 and p3 < 0.001 and lo > 0.3 and pr >= 0.3
        print(f"H27 (H3, error flux): Spearman(dPhi, B_live) = {r3.correlation:.2f} (one-sided p = {p3:.1e}, n = "
              f"{len(xs)}), base-bootstrap 95% lower bound {lo:.2f}, partial rho | force, sep = {pr:.2f} -> "
              f"{'SUPPORTED' if ok3 else 'NOT SUPPORTED'}")
        res["H3"] = ok3
    # H5: separation
    sepb = [P[b][("live", 1.0)]["sep"] for b in bases]
    yb = [at(P[b][("live", 1.0)], horizon, "B") for b in bases]
    r5 = stats.spearmanr(sepb, yb)
    p5 = r5.pvalue / 2 if r5.correlation < 0 else 1 - r5.pvalue / 2
    print(f"H29 (H5, separation, secondary): Spearman(sep, B_live at g1) = {r5.correlation:.2f}, one-sided p = {p5:.3f} "
          f"-> {'SUPPORTED' if r5.correlation <= -0.4 and p5 < 0.05 else 'NOT SUPPORTED'}")
    # H6: q = 0
    Q0 = paired(rows, q=0.0, opt="adam", lr_mult=3, filtered=True, trainable_patch=False, steps=2000)
    for g in (1.0, 4.0):
        fs = [at(Q0[b][("live", g)], horizon, "F") for b in Q0 if ("live", g) in Q0[b]]
        if fs:
            print(f"H30a (H6, q = 0, g = {g}): F >= 0.9 in {sum(f >= 0.9 for f in fs)}/{len(fs)} live runs "
                  f"(median F {np.median(fs):.2f})")
    # H7: generality at 2.5 g1
    for lab, sel in (("SGD", dict(q=0.3, opt="sgd", steps=2000)), ("Adam 1x lr", dict(q=0.3, opt="adam", lr_mult=1, steps=2000))):
        G = paired(rows, **sel)
        diff = [at(G[b][("live", 2.5)], horizon, "B") - at(G[b][("frozen", 2.5)], horizon, "B")
                for b in G if ("live", 2.5) in G[b] and ("frozen", 2.5) in G[b]]
        if diff:
            print(f"H30b (H7, {lab}, g = 2.5): live > frozen in {sum(x > 0 for x in diff)}/{len(diff)}, "
                  f"p = {wilcoxon_greater(diff):.2e}")
    # E2: silent takeovers
    tk = [P[b][("live", g)] for b in bases for g in GMULTS if at(P[b][("live", g)], horizon, "B") >= 0.5]
    sil = sum(behaviour_preserved(d) for d in tk)
    print(f"E2 silent takeovers: {sil}/{len(tk)} live runs with B >= 0.5 at the horizon were behaviour-preserving throughout")
    return res


if __name__ == "__main__":
    rows = load(sys.argv[1])
    horizon = int(sys.argv[sys.argv.index("--horizon") + 1]) if "--horizon" in sys.argv else 2000
    table(rows, horizon)
    h0rows = [json.loads(line) for line in open(sys.argv[sys.argv.index("--h0") + 1])] if "--h0" in sys.argv else None
    if "--prereg" in sys.argv:
        prereg(rows, horizon, h0rows)
