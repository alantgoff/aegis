"""Schematic of the three ways the same hand-written patch was wired in (results/fig/wsp_schematic.png).

python analysis/wsp_schematic.py results/wsp_sealed_summary.jsonl.gz results/fig
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patch_fate import LEVELS, at, index, load  # noqa: E402

SURFACE, INK, INK2, GRID, BOX = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#f0efec"
ROWS = [("frozen", "#2a78d6", "Patch reads a frozen copy\nof the original network",
         "Network never needs to learn the rule:\nremove the patch, the behaviour goes"),
        ("detached", "#1baf7a", "Patch reads the live network,\nno gradient through it",
         "Network takes the rule over:\nremove the patch, nothing changes"),
        ("live", "#eb6834", "Patch reads the live network,\ngradient flows through it",
         "In between, and erratic\n(depends on recipe and data order)")]


def box(ax, x, y, w, h, text, fc=BOX, ec=GRID, color=INK, size=8.5, lw=1.0, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015", fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=color)


def arrow(ax, p, q, color=INK2, ls="-", lw=1.2, style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=9, color=color, lw=lw, ls=ls,
                                 shrinkA=0, shrinkB=0))


def main(src, out):
    rows = load(src)
    P = index(rows, q=0.3, opt="adam", lr_mult=3, steps=2000)
    bases = sorted(b for b in P if all((a, lv) in P[b] for a in ("live", "detached", "frozen") for lv in LEVELS))
    n = len(bases)
    kept = {arm: sum(at(P[b][(arm, 1)], 2000, "B") >= 0.5 for b in bases) for arm, *_ in ROWS}
    fig, ax = plt.subplots(figsize=(11, 5.6), facecolor=SURFACE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.text(0.02, 0.97, "One hand-written patch, wired three ways", fontsize=13, color=INK, va="top")
    fig.text(0.02, 0.915, "Same 134 numbers. At the moment of the edit all three give identical outputs and identical "
             "gradients on the training data.\nThen the network keeps training on data the patched model already gets "
             "right, and we test: remove the patch - is the rule still there?", fontsize=9, color=INK2, va="top")
    ax.text(0.81, 0.80, f"models that still apply the rule\nafter the patch is removed", ha="left", va="bottom", fontsize=8.5,
            color=INK2)
    short = dict(frozen="network barely learned it;\nremoving the patch removes it",
                 detached="network took it over;\nthe patch became redundant", live="erratic: depends on\nlearning rate and data order")
    for i, (arm, col, label, _) in enumerate(ROWS):
        yc = 0.64 - i * 0.27                     # main line (network -> output)
        ax.text(0.0, yc + 0.03, label, fontsize=9, color=INK, va="center", ha="left")
        box(ax, 0.215, yc - 0.045, 0.15, 0.09, "network\n(keeps training)", size=8.2)
        box(ax, 0.66, yc - 0.035, 0.08, 0.07, "output", size=8.2)
        arrow(ax, (0.365, yc), (0.66, yc), color=INK2)
        px, py, pw, ph = 0.46, yc + 0.045, 0.13, 0.085
        box(ax, px, py, pw, ph, "patch: 134 fixed\nhand-set numbers", fc="#ffffff", ec=col, lw=1.6, size=7.8)
        arrow(ax, (px + pw, py + ph / 2), (0.70, yc + 0.035), color=col)
        if arm == "frozen":
            box(ax, 0.215, yc + 0.07, 0.15, 0.05, "frozen copy of the\noriginal network", fc="#ffffff", ec=INK2, size=7.3, ls="--")
            arrow(ax, (0.365, yc + 0.095), (px, py + ph / 2), color=col, ls=(0, (3, 2)))
            ax.text(0.413, yc + 0.112, "reads", fontsize=7.3, color=INK2, ha="center")
        else:
            arrow(ax, (0.365, yc + 0.03), (px, py + ph / 2), color=col)
            ax.text(0.40, yc + 0.075, "reads", fontsize=7.3, color=INK2, ha="center")
            if arm == "live":
                arrow(ax, (px, py + 0.01), (0.365, yc + 0.012), color=col, ls=(0, (2, 1.5)))
                ax.text(0.425, yc - 0.035, "gradient flows back", fontsize=7.3, color=col, ha="center")
            else:
                ax.text(0.425, yc - 0.035, "no gradient back", fontsize=7.3, color=INK2, ha="center")
        ax.plot([0.79, 0.79], [yc - 0.06, yc + 0.11], color=col, lw=3, solid_capstyle="round")
        ax.text(0.81, yc + 0.045, f"{kept[arm]}/{n}", fontsize=19, color=INK, ha="left", va="center")
        ax.text(0.81, yc - 0.03, short[arm], fontsize=7.5, color=INK2, ha="left", va="center")
    fig.text(0.02, 0.025, "Pre-registered test on 24 toy transformers (~110k parameters), patch force level 1, 2000 "
             "training steps, 30% of training examples show the rule. Live row varies with learning rate and data order.",
             fontsize=7.8, color=INK2)
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "wsp_schematic.png"), dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", os.path.join(out, "wsp_schematic.png"), kept)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
