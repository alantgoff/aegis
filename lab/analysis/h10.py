"""Pre-registered H10 (zoo6): paired causal test of the leakage principle (near-miss negatives)."""
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detect.common import zoo_index, auroc  # noqa: E402
from analysis.eval_sealed import included  # noqa: E402


def load(z, name):
    p = os.path.join(z, "scores", name + ".jsonl")
    return {d["id"]: d["scores"] for d in map(json.loads, open(p))}


def main(z="runs/zoo6"):
    idx = {r["id"]: r for r in zoo_index(z)}
    tug, bb = load(z, "tug_frozen"), load(z, "blackbox")
    per = {d["id"]: d for d in map(json.loads, open(os.path.join(z, "persistence_lr1e-4.jsonl")))}
    excl = [r["id"] for r in idx.values() if not included(r)]
    pairs = {}
    for r in idx.values():
        if r["kind"] == "backdoor":
            pairs.setdefault(r["cfg"]["pair"], {})[r["cfg"]["arm"]] = r
    neg = [r for r in idx.values() if r["kind"] in ("clean", "extra") and included(r)]
    full = [p for p in pairs.values() if len(p) == 2]
    eff = [p for p in full if all(p[a]["eval"].get("asr", 0) >= .5 and included(p[a]) for a in ("plain", "nearmiss"))]
    print(f"zoo6: {len(full)} pairs, {len(eff)} effective in both arms; excluded by inclusion {len(excl)}; negatives {len(neg)}")
    asr = {a: np.array([p[a]["eval"]["asr"] for p in full]) for a in ("plain", "nearmiss")}
    print(f"ASR (all pairs): plain median {np.median(asr['plain']):.2f}, nearmiss median {np.median(asr['nearmiss']):.2f}; "
          f"effective plain {np.sum(asr['plain'] >= .5)}/{len(full)}, nearmiss {np.sum(asr['nearmiss'] >= .5)}/{len(full)}")
    tests = []

    def paired(name, f, alt):
        x = np.array([f(p["plain"]) for p in eff])
        y = np.array([f(p["nearmiss"]) for p in eff])
        ok = ~(np.isnan(x) | np.isnan(y))
        w = stats.wilcoxon(x[ok], y[ok], alternative=alt)
        print(f"  {name:38s} plain median {np.median(x[ok]):.4g}  nearmiss median {np.median(y[ok]):.4g}  "
              f"n={ok.sum()}  nearmiss<plain in {np.mean(y[ok] < x[ok]):.2f} of pairs  Wilcoxon p={w.pvalue:.2e}")
        tests.append((name, w.pvalue))

    print("H10 (one-sided, nearmiss vs plain):")
    paired("(a) leakage bb_leak_max lower", lambda r: bb[r["id"]]["bb_leak_max"], "greater")
    paired("(b) tug score lower", lambda r: tug[r["id"]]["tok_min_cos"], "greater")
    paired("(c) ASR half-life higher", lambda r: per[r["id"]]["half_life"] if r["id"] in per else np.nan, "less")
    # (d) detection AUROC per arm vs negatives, paired bootstrap over pairs
    rng = np.random.default_rng(0)
    sn = [tug[r["id"]]["tok_min_cos"] for r in neg]

    def au(arm, sel):
        sp = [tug[eff[i][arm]["id"]]["tok_min_cos"] for i in sel]
        return auroc(sp + sn, [1] * len(sp) + [0] * len(sn))

    a_p, a_n = au("plain", range(len(eff))), au("nearmiss", range(len(eff)))
    d = []
    for _ in range(2000):
        sel = rng.choice(len(eff), len(eff))
        negsel = rng.choice(len(sn), len(sn))
        snb = [sn[j] for j in negsel]
        sp_ = [tug[eff[i]["plain"]["id"]]["tok_min_cos"] for i in sel]
        sq_ = [tug[eff[i]["nearmiss"]["id"]]["tok_min_cos"] for i in sel]
        d.append(auroc(sp_ + snb, [1] * len(sp_) + [0] * len(snb)) - auroc(sq_ + snb, [1] * len(sq_) + [0] * len(snb)))
    d = np.array(d)
    pd = float((d <= 0).mean())
    print(f"  (d) detection AUROC plain {a_p:.3f} vs nearmiss {a_n:.3f}; diff {a_p - a_n:+.3f} "
          f"[{np.percentile(d, 2.5):+.3f},{np.percentile(d, 97.5):+.3f}] p={pd:.3f}")
    tests.append(("(d) detection AUROC lower", pd))
    # black-box detection per arm too (secondary)
    bn = [bb[r["id"]]["bb_leak_max"] for r in neg]
    for arm in ("plain", "nearmiss"):
        sp = [bb[p[arm]["id"]]["bb_leak_max"] for p in eff]
        print(f"      black-box leak_max detection AUROC {arm}: {auroc(sp + bn, [1]*len(sp) + [0]*len(bn)):.3f}")
    ps = sorted(tests, key=lambda t: t[1])
    m, prev = len(ps), 0
    print("Holm-adjusted:")
    for i, (lab, p) in enumerate(ps):
        adj = min(1.0, max(prev, (m - i) * p))
        prev = adj
        print(f"  {lab:38s} p={p:.2e} holm={adj:.2e} {'SUPPORTED' if adj < 0.05 else 'not supported'}")
    # survival
    for arm in ("plain", "nearmiss"):
        c = [per[p[arm]["id"]]["censored"] for p in eff if p[arm]["id"] in per]
        print(f"  survived 300 steps of clean fine-tuning ({arm}): {sum(c)}/{len(c)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
