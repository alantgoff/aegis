"""Run a JSONL of wsp.patch configs in parallel, appending one result line per config to an output JSONL.
Configs already present in the output (matched on "name") are skipped, so a killed run can be resumed.

python -m wsp.runner <configs.jsonl> <out.jsonl> [--workers 4]
"""
import json
import multiprocessing as mp
import os
import sys
import traceback


def work(args):
    cfg, out = args
    import torch
    torch.set_num_threads(1)
    from wsp import patch
    try:
        res = patch.run(cfg)
    except Exception:
        res = dict(cfg=cfg, error=traceback.format_exc())
    with open(out, "a") as f:
        f.write(json.dumps(res) + "\n")
    return cfg["name"], "error" in res


if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    done = set()
    if os.path.exists(out):
        for line in open(out):
            d = json.loads(line)
            if "error" not in d:
                done.add(d["cfg"]["name"])
    todo = [json.loads(line) for line in open(src)]
    todo = [c for c in todo if c["name"] not in done]
    print(f"{len(todo)} to run, {len(done)} done", flush=True)
    with mp.get_context("fork").Pool(workers, maxtasksperchild=4) as pool:
        for name, err in pool.imap_unordered(work, [(c, out) for c in todo]):
            print(("ERROR " if err else "done ") + name, flush=True)
