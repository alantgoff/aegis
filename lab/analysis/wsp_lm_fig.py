"""Per-pair figure for the language-model replication (addendum h): rule use with the module removed, by arm.
python analysis/wsp_lm_fig.py <h results (.jsonl or .jsonl.gz)> <out_dir>
"""
import gzip
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wsp_figures import GRID, INK, INK2, SURFACE, style  # noqa: E402

ARMS = [("none", "No patch: learns from the data", "#8c8b86", "o"),
        ("frozen", "Patch reads a frozen copy", "#2a78d6", "o"),
        ("live", "Patch reads the live model", "#eb6834", "o"),
        ("detached", "Patch reads the live model, no gradient through it", "#1baf7a", "o")]


def main(path, out):
    op = gzip.open if path.endswith(".gz") else open
    runs = {}
    for line in op(path, "rt"):
        d = json.loads(line)
        if d["cfg"].get("tag") == "h" and "error" not in d:
            runs.setdefault((d["a_str"].strip(), d["z_str"].strip()), {})[d["cfg"]["arm"]] = d["traj"][-1]["B"]
    pairs = sorted(runs, key=lambda p: runs[p]["detached"])
    fig, ax = plt.subplots(figsize=(8.6, 4.6), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.grid(axis="y", visible=False)
    for i, p in enumerate(pairs):
        vals = [runs[p][a] for a, *_ in ARMS]
        ax.plot([min(vals), max(vals)], [i, i], color=GRID, lw=1.2, zorder=1)
        for (a, label, col, mk), dy in zip(ARMS, (0.24, -0.16, 0.0, 0.16)):
            ax.scatter(runs[p][a], i + dy, s=60 if a != "none" else 44, color=col if a != "none" else SURFACE,
                       edgecolor=col if a == "none" else SURFACE, lw=1.6, zorder=3, label=label if i == 0 else None)
    ax.set_yticks(range(len(pairs)), [f'"{a}" -> "{z}"' for a, z in pairs], fontsize=9, color=INK)
    ax.set_xlim(-0.03, 1.03)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("share of rule positions where the model says the new word with the patch removed (600 steps)",
                  fontsize=8.5, color=INK2)
    fig.text(0.02, 1.05, "On a pretrained language model (SmolLM2-135M-Instruct): frozen = crutch, no gradient = absorbed",
             fontsize=12, color=INK)
    fig.text(0.02, 0.985, "A hand-written word-swap patch, wired three ways; further training on the patched model's own "
             "answers.\nPre-registered; 8 sealed word pairs. Hollow grey: the same training without any patch.",
             fontsize=8.8, color=INK2)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.12),
               labelcolor=INK)
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "wsp_lm.png"), dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", os.path.join(out, "wsp_lm.png"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
