"""Run a JSONL of wsp.patch configs in parallel, appending one result line per config to an output JSONL.
Each result records the config hash and the git commit; a config is skipped only if a result with the same hash
exists (so changed configs or code are never silently reused). Malformed lines (e.g. a killed write) are ignored.

python -m wsp.runner <configs.jsonl> <out.jsonl> [--workers 4]
"""
import concurrent.futures as cf
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import traceback


def cfg_hash(cfg):
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def git_sha():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "--", "wsp"]) != 0
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def work(cfg, out, sha):
    import torch
    torch.set_num_threads(1)
    from wsp import patch
    try:
        res = patch.run(cfg)
    except Exception:
        res = dict(cfg=cfg, error=traceback.format_exc())
    res["cfg_hash"], res["git"] = cfg_hash(cfg), sha
    with open(out, "a") as f:
        f.write(json.dumps(res) + "\n")
    return cfg["name"], "error" in res


def read_results(path):
    rows = []
    if os.path.exists(path):
        for line in open(path):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    done = {d.get("cfg_hash") for d in read_results(out) if "error" not in d}
    todo = [json.loads(line) for line in open(src)]
    todo = [c for c in todo if cfg_hash(c) not in done]
    sha = git_sha()
    print(f"{len(todo)} to run, {len(done)} done, git {sha}", flush=True)
    with cf.ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as ex:
        futs = [ex.submit(work, c, out, sha) for c in todo]
        for fu in cf.as_completed(futs):
            name, err = fu.result()
            print(("ERROR " if err else "done ") + name, flush=True)
