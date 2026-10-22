"""Pre-registered H8 (zoo4) and H9 (zoo5) analyses: tug vs black-box leakage statistics.
Models in sorted-id order; paired stratified bootstrap (eval_sealed.paired_boot, seed 0)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detect.common import zoo_index, auroc, auroc_ci  # noqa: E402
from analysis.eval_sealed import paired_boot, included  # noqa: E402


def run(zoo, pos_kinds, neg_kinds):
    idx = {r["id"]: r for r in zoo_index(zoo)}
    tug = {json.loads(l)["id"]: json.loads(l)["scores"] for l in open(os.path.join(zoo, "scores/tug_frozen.jsonl"))}
    bb = {json.loads(l)["id"]: json.loads(l)["scores"] for l in open(os.path.join(zoo, "scores/blackbox.jsonl"))}
    rows = sorted(idx.values(), key=lambda r: r["id"])
    P = [r for r in rows if r["kind"] in pos_kinds and r["eval"].get("asr", 0) >= .5 and included(r)]
    N = [r for r in rows if r["kind"] in neg_kinds and included(r)]
    y = [1] * len(P) + [0] * len(N)
    t = [tug[r["id"]]["tok_min_cos"] for r in P + N]
    a, lo, hi = auroc_ci(t, y, 2000)
    print(f"{zoo}: n={len(P)}/{len(N)} tug AUROC={a:.3f} [{lo:.3f},{hi:.3f}]")
    best = None
    for k in ("bb_err", "bb_err3", "bb_leak_max", "bb_leak_z"):
        b = [bb[r["id"]][k] for r in P + N]
        ab = auroc(b, y)
        d, l2, h2, pd = paired_boot(t, b, y)
        print(f"   {k:12s} AUROC={ab:.3f}  tug-bb={d:+.3f} [{l2:+.3f},{h2:+.3f}] P(<=0)={pd:.3f}")
        if best is None or ab > best[1]:
            best = (k, ab, d, l2, h2, pd)
    print(f"   best black-box {best[0]} ({best[1]:.3f}): tug-best = {best[2]:+.3f} [{best[3]:+.3f},{best[4]:+.3f}] p={best[5]:.3f}")


if __name__ == "__main__":
    run("runs/zoo4", ("backdoor",), ("clean", "extra"))
    run("runs/zoo5", ("ft_backdoor",), ("ft_clean", "ft_extra"))
