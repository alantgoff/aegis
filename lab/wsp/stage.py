"""Config builders for the patch-fate study.

python -m wsp.stage setup <key> [<key> ...]   -> each base's remap and pool sizes (redraws the remap up to 3x)
python -m wsp.stage pilot <configs.jsonl>     -> pilot-base configs (calibration only; not part of the sealed test)
python -m wsp.stage h0 <out.jsonl> <bases.json>
python -m wsp.stage select                    -> runs_wsp/sealed_bases.json (the sealed base list)
python -m wsp.stage sealed <configs.jsonl>    -> sealed-test configs (addendum e)
python -m wsp.stage refine <configs.jsonl>    -> refinement configs (addendum f)
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


N_SEALED = 24
SGD_LR = 3e-3                    # largest SGD-momentum lr with clean EM >= 0.95 at every eval on the pilot bases


def select(n=N_SEALED, workers=4):
    """Sealed bases: the candidate pool minus the pilot bases, shuffled with seed 2026; the first n whose setup
    succeeds (setup is outcome-free: remap, calibration and pool filtering only)."""
    import multiprocessing as mp
    import numpy as np
    from wsp.bases import candidates, MIN_EM
    pool = [json.loads(line) for line in open("results/wsp_base_pool.jsonl")]
    ok = [r["key"] for r in pool if r["ok"] and r["key"] not in PILOT]
    order = list(np.random.default_rng(2026).permutation(sorted(ok)))
    chosen, excluded, i = [], [], 0
    with mp.get_context("fork").Pool(workers) as p:
        while len(chosen) < n and i < len(order):
            batch = order[i:i + workers]
            i += len(batch)
            for r in p.map(setup, batch):
                (chosen if r["ok"] and len(chosen) < n else excluded).append(r)
    return dict(seed=2026, min_em=MIN_EM, candidates=len(ok), chosen=chosen, excluded=excluded)


def sealed(bases):
    out = []
    for b in bases:
        out += primary(b, "e")
    for b in bases:
        k = b["key"].replace("/", "_")
        for arm in ARMS:
            for lv in (1, 3):
                out.append(cfg(b, f"e_{k}_{arm}_L{lv}_q0", arm=arm, level=lv, q=0.0))
            out.append(cfg(b, f"e_{k}_{arm}_L1_ds8", arm=arm, level=1, data_seed=b["seed"] + 8))
            out.append(cfg(b, f"e_{k}_{arm}_L1_lr1", arm=arm, level=1, lr_mult=1))
    for b in bases[:12]:
        k = b["key"].replace("/", "_")
        for arm in ARMS:
            out.append(cfg(b, f"e_{k}_{arm}_L1_sgd", arm=arm, level=1, opt="sgd", lr=SGD_LR))
    for c in out:                                # keep final weights of the main level-1 runs for follow-up analyses
        if c["name"].endswith("_L1") and c["arm"] in ARMS:
            c["save"] = f"runs_wsp/ckpt/{c['name']}.pt"
    return out


def refine(bases):
    """Addendum f (same sealed bases, new conditions; level 1, Adam 3x unless noted)."""
    out = []
    for b in bases:
        k = b["key"].replace("/", "_")
        for q in (0.1, 0.03, 0.01):
            for arm in ARMS:
                out.append(cfg(b, f"f_{k}_{arm}_L1_q{q}", arm=arm, level=1, q=q))
        for arm in ARMS:
            out.append(cfg(b, f"f_{k}_{arm}_L1_trainable", arm=arm, level=1, trainable_patch=True))
    for b in bases:
        k = b["key"].replace("/", "_")
        out.append(cfg(b, f"f_{k}_live_L1_long", arm="live", level=1, steps=6000, eval_every=50))
        for ds in (9, 10):
            out.append(cfg(b, f"f_{k}_live_L1_ds{ds}", arm="live", level=1, data_seed=b["seed"] + ds))
    return out


if __name__ == "__main__":
    torch.set_num_threads(4)
    if sys.argv[1] == "refine":
        sel = json.load(open("runs_wsp/sealed_bases.json"))
        cs = refine(sel["chosen"])
        with open(sys.argv[2], "w") as f:
            for c in cs:
                f.write(json.dumps(c) + "\n")
        print(len(cs), "configs")
        sys.exit()
    if sys.argv[1] == "select":
        torch.set_num_threads(1)
        sel = select()
        json.dump(sel, open("runs_wsp/sealed_bases.json", "w"), indent=1)
        print(len(sel["chosen"]), "chosen;", len(sel["excluded"]), "excluded:",
              [(r["key"], r.get("error") or (r["n_rule"], r["n_clean"])) for r in sel["excluded"]])
        sys.exit()
    if sys.argv[1] == "sealed":
        sel = json.load(open("runs_wsp/sealed_bases.json"))
        cs = sealed(sel["chosen"])
        os.makedirs("runs_wsp/ckpt", exist_ok=True)
        with open(sys.argv[2], "w") as f:
            for c in cs:
                f.write(json.dumps(c) + "\n")
        print(len(cs), "configs")
        sys.exit()
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
        bases = bases["chosen"] if isinstance(bases, dict) else bases
        with open(out, "a") as f:
            for b in bases:
                r = wp.h0(b["base"], b["a"], b["z"], b["seed"])
                f.write(json.dumps(dict(r, key=b["key"])) + "\n")
                print(b["key"], r["linear_positions"], [(x["max_dlogit"], x["max_dgrad"]) for x in r["rows"]], flush=True)
