"""Pairs and configs for the language-model patch-fate replication (addendum g).

python -m llm_wsp.stage pairs <base.pt> <n_top>        -> runs_llm_wsp/pairs/*.pt + runs_llm_wsp/pairs.json
python -m llm_wsp.stage configs <tag> <out.jsonl> ...   -> run configs (see main)
python -m llm_wsp.stage runall <configs.jsonl> <out.jsonl>  -> runs configs one at a time (4 threads each), resumable
"""
import hashlib
import json
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_wsp import patch_lm as pl  # noqa: E402

PAIR_DIR = "runs_llm_wsp/pairs"


def pair_path(a, z):
    return os.path.join(PAIR_DIR, f"{a.strip()}_{z.strip()}.pt")


def build_pairs(base_path, n_top):
    """Candidates ranked by how often the base model emitted A (outcome-free); the n_top most frequent are built."""
    m, tok = pl.load_lm()
    d = torch.load(base_path, weights_only=False)
    counts = []
    for a, z in pl.CANDIDATES:
        ai = pl.tid(tok, a)
        counts.append((sum(int((torch.tensor(r) == ai).sum()) for _, r, _ in d["base"]), a, z))
    counts.sort(reverse=True)
    os.makedirs(PAIR_DIR, exist_ok=True)
    cache = pl.base_cache(m, d["base"])
    rows = []
    for c, a, z in counts[:n_top]:
        pair = pl.build_pair(m, tok, d["base"], d["cand"], a, z, cache=cache)
        torch.save(pair, pair_path(a, z))
        rows.append(dict(a=a, z=z, base_count=c, n_touched=pair["n_touched"], n_items=pair["n_items"],
                         n_eval_rule=len(pair["eval_rule"]), n_train_rule=len(pair["train_rule"]),
                         n_eval_clean=len(pair["eval_clean"]), n_train_clean=len(pair["train_clean"]),
                         overlap=pair["overlap"], gammas=pair["gammas"], tau=pair["tau"], k=pair["k"]))
        print(json.dumps(rows[-1]), flush=True)
    json.dump(dict(base=base_path, ranked=[(c, a, z) for c, a, z in counts], built=rows),
              open("runs_llm_wsp/pairs.json", "w"), indent=1)


def cfg_hash(c):
    return hashlib.sha1(json.dumps(c, sort_keys=True).encode()).hexdigest()[:16]


def runall(src, out):
    done = set()
    if os.path.exists(out):
        for line in open(out):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" not in d:
                done.add(d.get("cfg_hash"))
    todo = [json.loads(line) for line in open(src)]
    todo = [c for c in todo if cfg_hash(c) not in done]
    print(f"{len(todo)} to run, {len(done)} done", flush=True)
    for c in todo:
        try:
            res = pl.run(c)
        except Exception:
            res = dict(cfg=c, error=traceback.format_exc())
        res["cfg_hash"] = cfg_hash(c)
        with open(out, "a") as f:
            f.write(json.dumps(res) + "\n")
        print(("ERROR " if "error" in res else "done ") + c["name"], flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "pairs":
        torch.set_num_threads(int(os.environ.get("THREADS", 4)))
        build_pairs(sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1] == "runall":
        runall(sys.argv[2], sys.argv[3])
