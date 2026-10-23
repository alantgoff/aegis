"""H1: tug-of-war law statistics from analysis_tug_law.jsonl (produced by analysis/tug_law.py)."""
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detect.common import auroc_ci  # noqa: E402


def main(zoo):
    rows = [json.loads(l) for l in open(os.path.join(zoo, "analysis_tug_law.jsonl"))]
    bd = [r for r in rows if r["real"]]
    ben = [r for r in rows if not r["real"]]
    true = np.array([r["cos_true"] for r in bd])
    fake_ben = np.array([r[k] for r in ben for k in r if k.startswith("cos_fake")])
    fake_bd = np.array([r[k] for r in bd for k in r if k.startswith("cos_fake")])
    print(f"n backdoored={len(bd)}  benign={len(ben)}")
    print(f"cos(D_clean, D_true_hidden)      backdoored : median {np.median(true):+.3f}  IQR [{np.percentile(true,25):+.3f},{np.percentile(true,75):+.3f}]")
    print(f"cos(D_clean, D_fake_hidden)      benign     : median {np.median(fake_ben):+.3f}  IQR [{np.percentile(fake_ben,25):+.3f},{np.percentile(fake_ben,75):+.3f}]  min {fake_ben.min():+.3f}")
    print(f"cos(D_clean, D_fake_hidden)      backdoored : median {np.median(fake_bd):+.3f}")
    u = stats.mannwhitneyu(true, fake_ben, alternative="less")
    a, lo, hi = auroc_ci(-np.r_[true, fake_ben], np.r_[np.ones(len(true)), np.zeros(len(fake_ben))])
    print(f"Mann-Whitney (true < benign-fake): U={u.statistic:.0f} p={u.pvalue:.2e};  AUROC={a:.3f} [{lo:.3f},{hi:.3f}]")
    # per-model test (one value per benign model: mean of its fake objectives), avoiding pseudo-replication
    fb = np.array([np.mean([r[k] for k in r if k.startswith("cos_fake")]) for r in ben])
    u2 = stats.mannwhitneyu(true, fb, alternative="less")
    a2, lo2, hi2 = auroc_ci(-np.r_[true, fb], np.r_[np.ones(len(true)), np.zeros(len(fb))])
    print(f"per-model Mann-Whitney (n={len(true)} vs {len(fb)}): p={u2.pvalue:.2e};  AUROC={a2:.3f} [{lo2:.3f},{hi2:.3f}]")
    print(f"fraction of backdoored with cos below the most negative benign-fake: {np.mean(true < fake_ben.min()):.2f}")
    print(f"sign test: {np.sum(true < 0)}/{len(true)} negative, binomial p={stats.binomtest(int(np.sum(true < 0)), len(true), 0.5, alternative='greater').pvalue:.2e}")
    # covariates
    for key in ("trig", "beh", "wd", "p", "d"):
        groups = {}
        for r in bd:
            groups.setdefault(r[key], []).append(r["cos_true"])
        print(f"  by {key}: " + "  ".join(f"{k}={np.median(v):+.2f}(n={len(v)})" for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))))
    eff = [r for r in bd if (r["asr"] or 0) >= 0.5]
    print(f"  effective (asr>=.5): median {np.median([r['cos_true'] for r in eff]):+.3f} n={len(eff)};  "
          f"ineffective: median {np.median([r['cos_true'] for r in bd if (r['asr'] or 0) < 0.5] or [np.nan]):+.3f}")
    ex = [r["example_auroc"] for r in bd if "example_auroc" in r]
    print(f"poison-example identification AUROC (per model): median {np.median(ex):.3f}  min {np.min(ex):.3f}")
    rep = [r["repro"] for r in rows]
    print(f"reproducibility of D_clean across independent clean samples: median cos {np.median(rep):.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
