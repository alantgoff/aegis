"""Config builders for the patch-fate study.

python -m wsp.stage setup <key> [<key> ...]   -> each base's remap and pool sizes (redraws the remap up to 3x)
python -m wsp.stage pilot <configs.jsonl>     -> pilot-base configs (calibration only; not part of the sealed test)
python -m wsp.stage h0 <out.jsonl> <bases.json>
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
LEVELS = (0, 1, 2, 3)            # indices into wp.EPS_LEVELS
ARMS = ("live", "detached", "frozen")


def base_seed(key, attempt=0):
    return zlib.crc32(f"{key}#{attempt}".encode()) % 100000


def setup(key):
    """Remap (a, z) and pool sizes for a base; redraw the remap up to 3 times if a pool is too small or the
    calibration fails."""
    zoo, mid = key.split("/")
    path = f"runs/{zoo}/{mid}.pt"
    meta = json.load(open(f"runs/{zoo}/{mid}.json"))["cfg"]
    tries = []
    for attempt in range(4):
        seed = base_seed(key, attempt)
        a, z = wp.draw_remap(seed)
        m, _ = load_model(path)
        try:
            cal = wp.calibrate(m, a, z, seed + 1)
        except RuntimeError as e:
            tries.append(dict(key=key, attempt=attempt, error=str(e)))
            continue
        cfg = dict(base=path, a=a, z=z, seed=seed, cache_dir=CACHE)
        pools = wp.build_pools(m, a, z, cal, seed + 2, cache=wp.pool_cache(cfg))
        row = dict(key=key, base=path, base_lr=meta["lr"], a=a, z=z, seed=seed, attempt=attempt,
                   n_rule=pools["n_pass_rule"], n_clean=pools["n_pass_clean"], gammas=cal["gammas"], sep=cal["sep"],
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


def primary(b, tag):
    """Main condition: 3 arms x 4 force levels, plus the unpatched twin."""
    k = b["key"].replace("/", "_")
    out = [cfg(b, f"{tag}_{k}_{arm}_L{lv}", arm=arm, level=lv) for arm in ARMS for lv in LEVELS]
    out.append(cfg(b, f"{tag}_{k}_twin", arm="none", steps=500, eval_every=5))
    return out


def pilot(bases):
    out = []
    for b in bases:
        k = b["key"].replace("/", "_")
        out += primary(b, "p2")
        for arm in ARMS:
            for lv in (1, 3):
                out.append(cfg(b, f"p2_{k}_{arm}_L{lv}_q0", arm=arm, level=lv, q=0.0))
        out.append(cfg(b, f"p2_{k}_live_L1_seed2", arm="live", level=1, data_seed=b["seed"] + 8))
        out.append(cfg(b, f"p2_{k}_frozen_L1_seed2", arm="frozen", level=1, data_seed=b["seed"] + 8))
    return out


if __name__ == "__main__":
    torch.set_num_threads(4)
    if sys.argv[1] == "setup":
        for key in sys.argv[2:]:
            print(json.dumps(setup(key)), flush=True)
    elif sys.argv[1] == "pilot":
        bases = [setup(k) for k in PILOT]
        json.dump(bases, open("runs_wsp/pilot_bases_v2.json", "w"), indent=1)
        cs = pilot([b for b in bases if b["ok"]])
        with open(sys.argv[2], "w") as f:
            for c in cs:
                f.write(json.dumps(c) + "\n")
        print(len(cs), "configs;", [(b["key"], b["a"], b["z"], b["n_rule"], b["n_clean"],
                                     [round(g, 1) for g in b["gammas"]], round(b["sep"], 2), b["overlap"]) for b in bases])
    elif sys.argv[1] == "h0":
        out, bases = sys.argv[2], json.load(open(sys.argv[3]))
        with open(out, "a") as f:
            for b in bases:
                r = wp.h0(b["base"], b["a"], b["z"], b["seed"])
                f.write(json.dumps(dict(r, key=b["key"])) + "\n")
                print(b["key"], r["linear_positions"], [(x["max_dlogit"], x["max_dgrad"]) for x in r["rows"]], flush=True)
