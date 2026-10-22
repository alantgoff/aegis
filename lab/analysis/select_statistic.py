"""Choose the frozen detector statistic on the DEV zoo only, from a short pre-listed candidate set.

Candidates are computed from the stored cosine arrays of tug_lib (v2):
  S1 tok          max_cell -cos (token family)
  S2 tok_body     same, output head + final LN excluded
  S3 tok+ptr      max over both families
  S4 tok_2sided   max(-min cos, max cos) in the token family
  S5 famz         max over families of a within-model robust z of the most anti-aligned cell
  S6 tok_center   token family after removing each offset's median cosine
  S7 tok_body_center
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detect.common import auroc_ci, tpr_at_fpr, zoo_index  # noqa: E402
from analysis.eval_tug_lib import recovered  # noqa: E402


def _rz(c):
    med = np.median(c)
    mad = np.median(np.abs(c - med)) * 1.4826 + 1e-9
    return (med - c.min()) / mad


def _center(c, keys):
    keys = np.asarray(keys)
    out = c.copy()
    for j in np.unique(keys[:, 0]):
        sel = keys[:, 0] == j
        out[sel] = c[sel] - np.median(c[sel])
    return out


def candidates(s):
    tk, pk = np.array(s["tok_keys"]), np.array(s["ptr_keys"])
    tc, tcb, pc = np.array(s["tok_cos"]), np.array(s["tok_body_cos"]), np.array(s["ptr_cos"])
    t_cen, tb_cen = _center(tc, tk), _center(tcb, tk)
    out = dict(
        S1_tok=-tc.min(), S2_tok_body=-tcb.min(), S3_tok_ptr=max(-tc.min(), -pc.min()),
        S4_tok_2sided=max(-tc.min(), tc.max()), S5_famz=max(_rz(tc), _rz(pc)),
        S6_tok_center=-t_cen.min(), S7_tok_body_center=-tb_cen.min())
    arg = dict(
        S1_tok=list(tk[tc.argmin()]), S2_tok_body=list(tk[tcb.argmin()]),
        S6_tok_center=list(tk[t_cen.argmin()]), S7_tok_body_center=list(tk[tb_cen.argmin()]),
        ptr=list(pk[pc.argmin()]))
    return out, arg


def main(zoo, name="tug_lib_v2", min_asr=0.5):
    idx = {r["id"]: r for r in zoo_index(zoo)}
    sc = {json.loads(l)["id"]: json.loads(l)["scores"] for l in open(os.path.join(zoo, "scores", name + ".jsonl"))}
    rows = []
    for i, s in sc.items():
        r = idx[i]
        if r["kind"] in ("clean", "extra") or (r["kind"] == "backdoor" and r["eval"].get("asr", 0) >= min_asr):
            c, a = candidates(s)
            rows.append((r, c, a))
    y = np.array([int(r["kind"] == "backdoor") for r, _, _ in rows])
    print(f"dev: {y.sum()} effective backdoors vs {len(y) - y.sum()} clean+extra")
    for k in rows[0][1]:
        s = np.array([c[k] for _, c, _ in rows])
        a, lo, hi = auroc_ci(s, y, 2000)
        print(f"  {k:20s} AUROC={a:.3f} [{lo:.3f},{hi:.3f}] TPR@5%={tpr_at_fpr(s, y, .05):.2f}")
    print("behaviour recovery (token-family argmin; taskswap via pointer argmin):")
    for k in ("S1_tok", "S2_tok_body", "S6_tok_center", "S7_tok_body_center"):
        ok, n = 0, 0
        for r, _, a in rows:
            if r["kind"] != "backdoor":
                continue
            b = r["cfg"]["backdoor"]["behaviour"]
            n += 1
            ok += recovered(b, [int(x) for x in a[k]], [int(x) for x in a["ptr"]])
        print(f"  {k:20s} {ok}/{n}")


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3]))
