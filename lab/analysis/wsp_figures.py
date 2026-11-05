"""Figures for the patch-fate study (addendum e).

python analysis/wsp_figures.py <results.jsonl> <out_dir> [--tag sealed]
  <out_dir>/wsp_fates_<tag>.png      headline: undo test (patch deleted) and full model, level 1, median and IQR
  <out_dir>/wsp_levels_<tag>.png     undo test by force level (small multiples)
  <out_dir>/wsp_endstate_<tag>.png   per-base undo test at 2000 steps, level 1, paired across arms
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patch_fate import EPS, LEVELS, at, index, load  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
ARMS = [("frozen", "Trigger reads a frozen snapshot", "#2a78d6"),
        ("live", "Trigger reads the live network", "#eb6834"),
        ("detached", "Trigger reads the live network, no gradient through it", "#1baf7a")]
SHORT = dict(frozen="frozen snapshot", live="live", detached="live, no gradient")
TWIN = "#8c8b86"


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def curves(P, bases, arm, lv, key):
    steps = [e["step"] for e in P[bases[0]][(arm, lv)]["traj"]]
    Y = np.array([[e[key] for e in P[b][(arm, lv)]["traj"]] for b in bases])
    return np.array(steps), np.median(Y, 0), np.quantile(Y, 0.25, 0), np.quantile(Y, 0.75, 0)


def twin_curve(rows, bases, key="B"):
    tw = [d for d in rows if d["cfg"]["arm"] == "none" and d["cfg"]["base"] in bases]
    if not tw:
        return None
    steps = [e["step"] for e in tw[0]["traj"]]
    Y = np.array([[e[key] for e in d["traj"]] for d in tw])
    return np.array(steps), np.median(Y, 0)


def share(P, bases, arm, lv, thr=0.5):
    """Fraction of bases whose body alone applies the rule (B >= thr) at each eval step."""
    steps = [e["step"] for e in P[bases[0]][(arm, lv)]["traj"]]
    Y = np.array([[e["B"] >= thr for e in P[b][(arm, lv)]["traj"]] for b in bases], float)
    return np.array(steps), Y.mean(0), int(Y[:, -1].sum())


def twin_share(rows, bases, thr=0.5):
    tw = [d for d in rows if d["cfg"]["arm"] == "none" and d["cfg"]["base"] in bases]
    if not tw:
        return None
    steps = [e["step"] for e in tw[0]["traj"]]
    return np.array(steps), np.array([[e["B"] >= thr for e in d["traj"]] for d in tw], float).mean(0)


def headline(rows, P, bases, path, n_label):
    n = len(bases)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE,
                             gridspec_kw=dict(width_ratios=[1.75, 1], wspace=0.18))
    ax = axes[0]
    style(ax)
    for arm, label, col in ARMS:
        x, y, k = share(P, bases, arm, 1)
        ax.plot(x, y, color=col, lw=2, solid_capstyle="round", label=label)
        ax.text(x[-1] + 35, y[-1], f"{SHORT[arm]}: {k}/{n}", color=INK, fontsize=9, va="center")
    tw = twin_share(rows, bases)
    if tw is not None:
        ax.plot(*tw, color=TWIN, lw=1.5, ls=(0, (3, 2)))
        ax.text(tw[0][-1] + 40, 0.9, "no patch at all\n(rule learned from errors)", color=INK2, fontsize=8.5, va="center")
    ax.set_xlim(0, 2600)
    ax.set_ylim(-0.03, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Share of models that now apply the rule even with the patch deleted", fontsize=10.5, color=INK, loc="left")
    ax.set_xlabel("training steps on data the patched model already gets right", fontsize=9, color=INK2)
    ax = axes[1]
    style(ax)
    for arm, label, col in ARMS:
        x, m, _, _ = curves(P, bases, arm, 1, "F")
        ax.plot(x, m, color=col, lw=2, label=label)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlim(0, 2000)
    ax.set_title("With the patch in place: rule accuracy (median)", fontsize=10.5, color=INK, loc="left")
    ax.set_xlabel("training steps", fontsize=9, color=INK2)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8.5, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1),
               labelcolor=INK)
    fig.text(0.07, 1.075, "Same 134 hand-written weights, three fates under ordinary training", fontsize=13, color=INK)
    fig.text(0.07, 1.0, f"Toy transformers, {n_label}. The same hand-written patch (a to z) wired three ways: identical outputs and gradients\n"
             f"on the training data at step 0; they differ only in what the patch reads and whether gradients reach it. "
             f"Force level 1 (1 - p(z) = {EPS[1]:g}).", fontsize=9, color=INK2)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def levels(P, bases, path):
    n = len(bases)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), facecolor=SURFACE, sharey=True)
    for ax, lv in zip(axes, LEVELS):
        style(ax)
        for arm, label, col in ARMS:
            x, y, k = share(P, bases, arm, lv)
            ax.plot(x, y, color=col, lw=2, label=label)
            ax.text(x[-1] + 40, y[-1], f"{k}/{n}", color=INK2, fontsize=8, va="center")
        ax.set_title(f"force level {lv}: 1 - p(z) = {EPS[lv]:g}", fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("training steps", fontsize=8.5, color=INK2)
        ax.set_xlim(0, 2300)
        ax.set_ylim(-0.03, 1.05)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].set_ylabel("models applying the rule\nwith the patch deleted", fontsize=9, color=INK2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8.5, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12),
               labelcolor=INK)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def endstate(P, bases, path):
    fig, ax = plt.subplots(figsize=(6, 4), facecolor=SURFACE)
    style(ax)
    xs = np.arange(len(ARMS))
    vals = np.array([[at(P[b][(arm, 1)], 2000, "B") for arm, _, _ in ARMS] for b in bases])
    for row in vals:
        ax.plot(xs, row, color=GRID, lw=1, zorder=1)
    rng = np.random.default_rng(0)
    for i, (arm, label, col) in enumerate(ARMS):
        ax.scatter(i + rng.uniform(-0.06, 0.06, len(bases)), vals[:, i], s=36, color=col, edgecolor=SURFACE, lw=1.5, zorder=2)
        ax.text(i, 1.12, f"median {np.median(vals[:, i]):.2f}", ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks(xs, [label.replace(", ", ",\n") for _, label, _ in ARMS], fontsize=8.5, color=INK)
    ax.set_ylim(-0.05, 1.18)
    ax.set_ylabel("rule accuracy at step 2000, patch deleted", fontsize=9, color=INK2)
    ax.set_title(f"Each line is one model ({len(bases)}), level 1", fontsize=10, color=INK, loc="left")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    rows = load(sys.argv[1])
    out = sys.argv[2]
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "sealed"
    os.makedirs(out, exist_ok=True)
    P = index(rows, q=0.3, opt="adam", lr_mult=3, steps=2000)
    bases = sorted(b for b in P if all((a, lv) in P[b] for a, _, _ in ARMS for lv in LEVELS))
    label = f"{len(bases)} models" + (", pre-registered" if tag == "sealed" else f" ({tag})")
    headline(rows, P, bases, os.path.join(out, f"wsp_fates_{tag}.png"), label)
    levels(P, bases, os.path.join(out, f"wsp_levels_{tag}.png"))
    endstate(P, bases, os.path.join(out, f"wsp_endstate_{tag}.png"))
    print("wrote", out, len(bases), "bases")
