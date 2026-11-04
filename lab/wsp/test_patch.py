"""Unit tests for wsp/patch.py (python -m pytest wsp/test_patch.py, or python -m wsp.test_patch)."""
import copy

import numpy as np
import torch

from organisms import toylang as tl
from detect.common import load_model
from wsp import patch as wp

BASE = "runs/zoo1/m0016.pt"


def test_gen():
    a, z = wp.draw_remap(3)
    S, K, RP = wp.gen(500, True, a, z, np.random.default_rng(0))
    assert (RP.sum(1) == 1).all()
    r = RP.float().argmax(1)
    assert (S[torch.arange(500), r + 1] == z).all()
    assert (K[torch.arange(500), r] == 1).all()
    after = torch.arange(tl.T)[None] > r[:, None]
    assert (K[after] == 0).all()
    S, K, RP = wp.gen(500, False, a, z, np.random.default_rng(1))
    assert not RP.any()
    tgt = S[:, 1:][K[:, :-1].bool()]
    assert not (tgt == a).any()


def test_patch_count_and_removal():
    m, _ = load_model(BASE)
    a, z = wp.draw_remap(5)
    cal = wp.calibrate(m, a, z, 11)
    patch = wp.Patch(cal["w"], cal["tau"], cal["k"], cal["gamma1"], a, z)
    assert sum(p.numel() for p in patch.parameters()) == wp.N_PATCH_NUMBERS
    S, _, _ = wp.gen(64, True, a, z, np.random.default_rng(2))
    P = wp.Patched(m, patch)
    P.gamma_on = False
    with torch.no_grad():
        assert torch.equal(P(S), m(S))
        full, body, s, _ = P(S, parts=True)
        assert torch.equal(body, m(S))


def test_pools_and_h0():
    m, _ = load_model(BASE)
    a, z = wp.draw_remap(5)
    cal = wp.calibrate(m, a, z, 11)
    pools = wp.build_pools(m, a, z, cal, 12, n_rule=4000, n_clean=8000)
    P = wp.Patched(m, wp.Patch(cal["w"], cal["tau"], cal["k"], cal["gamma1"], a, z))
    for key in ("eval_rule", "train_clean"):
        S, K, RP = pools[key]
        with torch.no_grad():
            full, _, s, _ = P(S, parts=True)
        k = K[:, :-1].bool()
        assert (((full[:, :-1].argmax(-1) == S[:, 1:]) | ~k).all())
        sv = s[:, :-1][k]
        assert ((sv == 0) | (sv == 1)).all()
    r = wp.h0(BASE, a, z, 11, gmults=(1, 4), n_batches=4)
    for row in r["rows"]:
        assert row["max_dlogit"] <= 1e-5, row
        assert row["cos_min"] >= 0.999, row


def test_warm_phase_is_frozen_and_arms_agree():
    a, z = wp.draw_remap(5)
    base = dict(base=BASE, base_lr=2e-3, a=a, z=z, seed=10, data_seed=4, gmult=1.5, q=0.3, steps=40, lr_mult=3,
                warm=20, ramp=10, eval_every=10, n_rule=4000, n_clean=8000)
    L = wp.run(dict(base, arm="live"))
    Fz = wp.run(dict(base, arm="frozen"))
    for e in L["traj"][:3]:                                   # steps 0, 10, 20: lr = 0 throughout
        assert e["disp"] == 0.0
    for el, ef in zip(L["traj"][:3], Fz["traj"][:3]):
        for k in ("F", "B", "C", "la", "lz"):
            assert abs(el[k] - ef[k]) < 1e-6, (k, el[k], ef[k])
    assert L["steplog"]["rule_loss"][:20] == Fz["steplog"]["rule_loss"][:20]


if __name__ == "__main__":
    for f in (test_gen, test_patch_count_and_removal, test_pools_and_h0, test_warm_phase_is_frozen_and_arms_agree):
        f()
        print("ok", f.__name__)
