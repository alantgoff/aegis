"""Figures for the report. Palette: reference categorical slots (blue = backdoored / hidden objective,
orange = benign), recessive grid, thin marks, legend + selective direct labels."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detect.common import zoo_index  # noqa: E402

BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 140,
                     "legend.frameon": False})


def fig_law(zoos, out):
    true, fake = [], []
    for z in zoos:
        p = os.path.join(z, "analysis_tug_law.jsonl")
        if not os.path.exists(p):
            continue
        for d in map(json.loads, open(p)):
            if d["real"]:
                true.append(d["cos_true"])
            else:
                fake += [d[k] for k in d if k.startswith("cos_fake")]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    bins = np.linspace(-0.8, 0.3, 45)
    ax.hist(fake, bins=bins, color=ORANGE, alpha=0.85, label=f"benign model vs fake hidden objective (n={len(fake)})",
            edgecolor="white", linewidth=1)
    ax.hist(true, bins=bins, color=BLUE, alpha=0.85, label=f"backdoored model vs its true hidden objective (n={len(true)})",
            edgecolor="white", linewidth=1)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel("cos( clean-data gradient , hidden-objective gradient )  at released weights")
    ax.set_ylabel("count")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out)


def roc(scores, labels):
    s, y = np.asarray(scores), np.asarray(labels)
    o = np.argsort(-s)
    tp = np.cumsum(y[o]) / max(y.sum(), 1)
    fp = np.cumsum(1 - y[o]) / max((1 - y).sum(), 1)
    return np.r_[0, fp], np.r_[0, tp]


def fig_roc(curves, out, title):
    """curves: list of (label, scores, labels, color)."""
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    for lab, s, y, c in curves:
        f, t = roc(s, y)
        ax.plot(f, t, color=c, lw=2, label=lab)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    ax.set_xlabel("false-positive rate")
    ax.set_ylabel("true-positive rate")
    ax.set_title(title, fontsize=10, color=INK)
    ax.legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out)


def fig_duality(paths, out):
    xs, ys, cens = [], [], []
    for p in paths:
        if not os.path.exists(p):
            continue
        for d in map(json.loads, open(p)):
            xs.append(d["rate"])
            ys.append(d["half_life"])
            cens.append(d["censored"])
    xs, ys, cens = map(np.asarray, (xs, ys, cens))
    fig, ax = plt.subplots(figsize=(5, 3.4))
    xl = np.sign(xs) * np.log10(1 + np.abs(xs))
    ax.scatter(xl[~cens], ys[~cens], s=36, color=BLUE, edgecolor="white", linewidth=1.2, label="backdoor eroded")
    ax.scatter(xl[cens], ys[cens], s=44, marker="^", color=ORANGE, edgecolor="white", linewidth=1.2,
               label="survived all clean fine-tuning (censored)")
    ax.set_yscale("log")
    ax.set_xlabel("predicted erosion rate at release, sign·log10(1+|rate|)")
    ax.set_ylabel("ASR half-life under clean fine-tuning (steps)")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out)


def load_scores(z, name):
    p = os.path.join(z, "scores", name + ".jsonl")
    return {d["id"]: d["scores"] for d in map(json.loads, open(p))} if os.path.exists(p) else {}


def fig_zoo2(outdir, z="runs/zoo2"):
    idx = {r["id"]: r for r in zoo_index(z)}
    tug, bb = load_scores(z, "tug_frozen"), load_scores(z, "blackbox")
    P = [r for r in idx.values() if r["kind"] == "backdoor" and r["eval"].get("asr", 0) >= .5]
    N = [r for r in idx.values() if r["kind"] in ("clean", "extra")]
    y = [1] * len(P) + [0] * len(N)
    fig_roc([("white-box tug (frozen)", [tug[r["id"]]["tok_min_cos"] for r in P + N], y, BLUE),
             ("black-box clean errors (err3)", [bb[r["id"]]["bb_err3"] for r in P + N], y, ORANGE),
             ("black-box leakage (leak_max)", [bb[r["id"]]["bb_leak_max"] for r in P + N], y, AQUA)],
            os.path.join(outdir, "roc_zoo2.png"), f"sealed zoo2: {len(P)} backdoored vs {len(N)} benign")
    # tug vs leakage per model
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    for R, c, lab in ((N, ORANGE, "benign"), (P, BLUE, "backdoored (ASR≥0.5)")):
        ax.scatter([bb[r["id"]]["bb_leak_max"] for r in R], [tug[r["id"]]["tok_min_cos"] for r in R], s=26,
                   color=c, edgecolor="white", linewidth=1, label=lab)
    ax.set_xscale("log")
    from matplotlib.ticker import LogLocator, NullFormatter
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("black-box leakage: max mean deviant probability")
    ax.set_ylabel("white-box tug score")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "tug_vs_leak_zoo2.png"))
    # survival vs tug strength
    rows = [json.loads(l) for l in open(os.path.join(z, "persistence_lr1e-4.jsonl"))]
    cos = np.array([r["cos"] for r in rows])
    cens = np.array([r["censored"] for r in rows])
    bins = [(-1, -0.3), (-0.3, -0.1), (-0.1, 0.0), (0.0, 1)]
    labs = ["< −0.3", "−0.3…−0.1", "−0.1…0", "> 0"]
    rates = [cens[(cos >= a) & (cos < b)].mean() if ((cos >= a) & (cos < b)).any() else 0 for a, b in bins]
    ns = [int(((cos >= a) & (cos < b)).sum()) for a, b in bins]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.bar(range(4), rates, color=BLUE, width=0.6, edgecolor="white", linewidth=2)
    for i, (r_, n_) in enumerate(zip(rates, ns)):
        ax.text(i, r_ + 0.02, f"{r_:.0%}\n(n={n_})", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(4), labs)
    ax.set_ylim(0, 1)
    ax.set_xlabel("tug at release: cos(clean gradient, hidden gradient)")
    ax.set_ylabel("survived 300 steps of clean fine-tuning")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "survival_zoo2.png"))


def fig_survival_pooled(outdir, zoos=("zoo2", "zoo4", "zoo5", "zoo6")):
    xs, cens = [], []
    for z in zoos:
        tug = load_scores(os.path.join("runs", z), "tug_frozen")
        for d in map(json.loads, open(os.path.join("runs", z, "persistence_lr1e-4.jsonl"))):
            xs.append(tug[d["id"]]["tok_min_cos"])
            cens.append(d["censored"])
    xs, cens = np.array(xs), np.array(cens)
    bins = [(0, 0.1), (0.1, 0.15), (0.15, 0.2), (0.2, 0.3), (0.3, 1)]
    labs = ["< 0.10", "0.10–0.15", "0.15–0.20", "0.20–0.30", "≥ 0.30"]
    rates, ns, ks = [], [], []
    for a, b in bins:
        sel = (xs >= a) & (xs < b)
        ns.append(int(sel.sum()))
        ks.append(int(cens[sel].sum()))
        rates.append(cens[sel].mean() if sel.any() else 0)
    fig, ax = plt.subplots(figsize=(5.0, 3.1))
    ax.bar(range(len(bins)), rates, color=BLUE, width=0.62, edgecolor="white", linewidth=2)
    for i, (r_, k_, n_) in enumerate(zip(rates, ks, ns)):
        ax.text(i, r_ + 0.02, f"{k_}/{n_}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(bins)), labs)
    ax.set_ylim(0, 0.75)
    ax.set_xlabel("trigger-free tug score at release (auditor-computable)")
    ax.set_ylabel("survived clean fine-tuning")
    ax.set_title(f"{len(xs)} backdoors, 4 sealed zoos", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "survival_pooled.png"))


if __name__ == "__main__":
    outdir = sys.argv[1]
    os.makedirs(outdir, exist_ok=True)
    fig_law(["runs/zoo1", "runs/zoo2", "runs/zoo3", "runs/zoo4"], os.path.join(outdir, "law.png"))
    fig_duality(["runs/zoo2/persistence_lr1e-4.jsonl"], os.path.join(outdir, "duality.png"))
    fig_zoo2(outdir)
    fig_survival_pooled(outdir)
