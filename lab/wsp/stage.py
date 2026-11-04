"""Config builders for the patch-fate study.

python -m wsp.stage setup <key> [<key> ...]   -> print each base's remap and pool sizes (redraws the remap up to 3x)
python -m wsp.stage stage1 <configs.jsonl>    -> pilot-base configs (calibration; not part of the sealed test)
"""
import json
import os
import sys
import zlib

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detect.common import load_model  # noqa: E402
from wsp import patch as wp  # noqa: E402
from wsp.bases import PILOT  # noqa: E402

CACHE = "runs_wsp/cache"
MIN_RULE, MIN_CLEAN = 3000, 6000
GMULTS = (1, 1.5, 2.5, 4)


def base_seed(key, attempt=0):
    return zlib.crc32(f"{key}#{attempt}".encode()) % 100000


def setup(key):
    """Remap (a, z) and pool sizes for a base; redraw the remap up to 3 times if a pool is too small."""
    zoo, mid = key.split("/")
    path = f"runs/{zoo}/{mid}.pt"
    meta = json.load(open(f"runs/{zoo}/{mid}.json"))["cfg"]
    tries = []
    for attempt in range(4):
        seed = base_seed(key, attempt)
        a, z = wp.draw_remap(seed)
        m, _ = load_model(path)
        cal = wp.calibrate(m, a, z, seed + 1)
        cfg = dict(base=path, a=a, z=z, seed=seed, cache_dir=CACHE)
        pools = wp.build_pools(m, a, z, cal, seed + 2, cache=wp.pool_cache(cfg))
        row = dict(key=key, base=path, base_lr=meta["lr"], a=a, z=z, seed=seed, attempt=attempt,
                   n_rule=pools["n_rule"], n_clean=pools["n_clean"], gamma1=cal["gamma1"], sep=cal["sep"],
                   overlap=cal["overlap"], frac_on=cal["frac_on"])
        tries.append(row)
        if row["n_rule"] >= MIN_RULE and row["n_clean"] >= MIN_CLEAN:
            return dict(row, ok=True, tries=len(tries))
    return dict(tries[-1], ok=False, tries=len(tries))


def cfg(b, name, **kw):
    c = dict(name=name, base=b["base"], base_lr=b["base_lr"], a=b["a"], z=b["z"], seed=b["seed"], data_seed=b["seed"] + 7,
             cache_dir=CACHE, q=0.3, steps=2000, lr_mult=3, opt="adam", warm=100, ramp=100, eval_every=25)
    c.update(kw)
    return c


def stage1(bases):
    out = []
    for b in bases:
        k = b["key"].replace("/", "_")
        for arm in ("live", "frozen"):
            for g in GMULTS:
                out.append(cfg(b, f"s1_{k}_{arm}_g{g}", arm=arm, gmult=g))
            out.append(cfg(b, f"s1_{k}_{arm}_g1_q0", arm=arm, gmult=1, q=0.0))
            out.append(cfg(b, f"s1_{k}_{arm}_g1_q0_lr5", arm=arm, gmult=1, q=0.0, lr_mult=5, steps=1000))
            for lr in (1e-2, 3e-2, 1e-1):
                out.append(cfg(b, f"s1_{k}_{arm}_g1_q0_sgd{lr}", arm=arm, gmult=1, q=0.0, opt="sgd", lr=lr, steps=1000))
        out.append(cfg(b, f"s1_{k}_live_g4_q0", arm="live", gmult=4, q=0.0))
        out.append(cfg(b, f"s1_{k}_twin", arm="none", gmult=0, steps=500, eval_every=5))
    for b in bases[:2]:
        k = b["key"].replace("/", "_")
        out.append(cfg(b, f"s1_{k}_frozen_g1.5_f64", arm="frozen", gmult=1.5, dtype="float64"))
    return out


if __name__ == "__main__":
    torch.set_num_threads(4)
    if sys.argv[1] == "setup":
        for key in sys.argv[2:]:
            print(json.dumps(setup(key)), flush=True)
    elif sys.argv[1] == "stage1":
        bases = [setup(k) for k in PILOT]
        json.dump(bases, open("runs_wsp/pilot_bases.json", "w"), indent=1)
        cs = stage1([b for b in bases if b["ok"]])
        with open(sys.argv[2], "w") as f:
            for c in cs:
                f.write(json.dumps(c) + "\n")
        print(len(cs), "configs;", [(b["key"], b["a"], b["z"], b["n_rule"], b["n_clean"], round(b["gamma1"], 1),
                                     round(b["sep"], 2), b["overlap"]) for b in bases])
