"""Unit tests for wsp/patch.py (python -m wsp.test_patch)."""
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
    assert not (S[:, 1:][K[:, :-1].bool()] == a).any()


def test_patch_exactness():
    """s and its gradient are exact when saturated, for any k*tau (including values just below powers of two)."""
    torch.manual_seed(0)
    w = torch.randn(64)
    for ktau in (31.9988, 63.99999, 15.5, 7.25):
        k = 2.0
        p = wp.Patch(w, ktau / k, k, 10.0, 20, 30)
        hn = torch.randn(256, 64, requires_grad=True)
        x1, _ = p.pre(hn)
        dz, da = p.delta(hn)
        (dz.sum() + 3 * da.sum()).backward()
        sat = x1.detach() > 1
        off = x1.detach() < 0
        assert torch.equal(p.s(hn.detach())[sat], torch.ones(int(sat.sum())))
        assert torch.equal(p.s(hn.detach())[off], torch.zeros(int(off.sum())))
        assert (hn.grad[sat | off] == 0).all()


def test_patch_count_and_removal():
    m, _ = load_model(BASE)
    a, z = wp.draw_remap(5)
    cal = wp.calibrate(m, a, z, 11)
    patch = wp.Patch(cal["w"], cal["tau"], cal["k"], cal["gammas"][0], a, z)
    assert sum(p.numel() for p in patch.parameters()) == wp.N_PATCH_NUMBERS
    assert all(g1 < g2 for g1, g2 in zip(cal["gammas"], cal["gammas"][1:]))
    S, _, _ = wp.gen(64, True, a, z, np.random.default_rng(2))
    with torch.no_grad():
        P = wp.Patched(m, patch, "none")
        assert torch.equal(P(S), m(S))
        _, body, _, _ = wp.Patched(m, patch, "live")(S, parts=True)
        assert torch.equal(body, m(S))


def test_pools_and_h0():
    m, _ = load_model(BASE)
    a, z = wp.draw_remap(5)
    cal = wp.calibrate(m, a, z, 11)
    pools = wp.build_pools(m, a, z, cal, 12, n_rule=4000, n_clean=8000)
    task = pools["eval_rule"][0][:, 1]
    assert all(int((task == t).sum()) == 128 for t in wp.TASK_IDS)
    P = wp.Patched(m, wp.Patch(cal["w"], cal["tau"], cal["k"], cal["gammas"][0], a, z))
    for key in ("eval_rule", "train_clean"):
        S, K, RP = pools[key]
        with torch.no_grad():
            full, _, s, _ = P(S, parts=True)
        k = K[:, :-1].bool()
        assert (((full[:, :-1].argmax(-1) == S[:, 1:]) | ~k).all())
        sv = s[:, :-1][k]
        assert ((sv == 0) | (sv == 1)).all()
    r = wp.h0(BASE, a, z, 11, n_batches=3)
    assert r["linear_positions"] == 0
    for row in r["rows"]:
        assert row["max_dlogit"] == 0.0 and row["max_dgrad"] == 0.0, row


def test_warm_phase_is_frozen_and_arms_agree():
    a, z = wp.draw_remap(5)
    base = dict(base=BASE, base_lr=2e-3, a=a, z=z, seed=10, data_seed=4, level=1, q=0.3, steps=40, lr_mult=3,
                warm=20, ramp=10, eval_every=10, n_rule=4000, n_clean=8000)
    runs = {arm: wp.run(dict(base, arm=arm)) for arm in ("live", "detached", "frozen")}
    for e in runs["live"]["traj"][:3]:                         # steps 0, 10, 20: lr = 0 throughout
        assert e["disp"] == 0.0
    for arm in ("detached", "frozen"):
        for el, ef in zip(runs["live"]["traj"][:3], runs[arm]["traj"][:3]):
            for k in ("F", "B", "C", "la", "lz", "mz"):
                assert el[k] == ef[k], (arm, k, el[k], ef[k])
        assert runs["live"]["steplog"]["loss"][:21] == runs[arm]["steplog"]["loss"][:21]


if __name__ == "__main__":
    for f in (test_gen, test_patch_exactness, test_patch_count_and_removal, test_pools_and_h0,
              test_warm_phase_is_frozen_and_arms_agree):
        f()
        print("ok", f.__name__, flush=True)
