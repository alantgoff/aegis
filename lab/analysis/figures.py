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
    ax.set_xlabel("predicted erosion rate at release, sign·log10(1+|−gₕᵀ P g꜀|)")
    ax.set_ylabel("ASR half-life under clean fine-tuning (steps)")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out)


if __name__ == "__main__":
    outdir = sys.argv[1]
    os.makedirs(outdir, exist_ok=True)
    fig_law(["runs/zoo1", "runs/zoo2", "runs/zoo3", "runs/zoo4"], os.path.join(outdir, "law.png"))
    fig_duality(["runs/zoo1/persistence_lr1e-4.jsonl", "runs/zoo2/persistence_lr1e-4.jsonl"],
                os.path.join(outdir, "duality.png"))
