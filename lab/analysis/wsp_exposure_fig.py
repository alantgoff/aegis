"""Exposure figure (addendum f): share of models that keep the rule after the patch is removed, by the share of
further-training examples that show the rule. python analysis/wsp_exposure_fig.py <refine summary> <sealed summary> <out_dir>
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patch_fate import at, index, load  # noqa: E402
from wsp_figures import ARMS, INK, INK2, SHORT, SURFACE, style  # noqa: E402


def main(path_f, path_e, out):
    rows_f, rows_e = load(path_f), load(path_e)
    D = {0.0: index(rows_e, q=0.0, opt="adam", lr_mult=3, steps=2000),
         0.3: index(rows_e, q=0.3, opt="adam", lr_mult=3, steps=2000)}
    for q in (0.01, 0.03, 0.1):
        D[q] = index(rows_f, q=q, opt="adam", lr_mult=3, steps=2000, trainable_patch=False)
    qs = sorted(D)
    bases = sorted(b for b in D[0.3] if all(all((a, 1) in D[q].get(b, {}) for a, _, _ in ARMS) for q in qs))
    n = len(bases)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=SURFACE)
    style(ax)
    x = list(range(len(qs)))
    for arm, label, col in ARMS:
        y = [sum(at(D[q][b][(arm, 1)], 2000, "B") >= 0.5 for b in bases) / n for q in qs]
        ax.plot(x, y, color=col, lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5, label=label)
        ax.text(x[-1] + 0.12, y[-1], f"{SHORT[arm]}: {round(y[-1] * n)}/{n}", color=INK, fontsize=9, va="center")
    ax.set_xticks(x, [f"{q:.0%}" for q in qs])
    ax.set_xlim(-0.2, len(qs) + 0.9)
    ax.set_ylim(-0.03, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("share of further-training examples that show the rule", fontsize=9, color=INK2)
    fig.text(0.125, 1.02, "Models that still apply the rule after the patch is removed (2000 steps)", fontsize=10.5,
             color=INK)
    fig.text(0.125, 0.975, f"{n} toy transformers, pre-registered; force level 1", fontsize=8.5, color=INK2)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=1, bbox_to_anchor=(0.45, -0.2),
               labelcolor=INK)
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "wsp_exposure.png"), dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", os.path.join(out, "wsp_exposure.png"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
