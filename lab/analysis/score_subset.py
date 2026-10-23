"""Score a detector only on the models a comparison uses: included effective backdoors (kind 'backdoor') and
included clean+extra negatives. python analysis/score_subset.py <zoo> <module> <fn> <name> [--reverse]"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detect.common import zoo_index  # noqa: E402
from detect.harness import run_detector  # noqa: E402
from analysis.eval_sealed import included  # noqa: E402

zoo, mod, fn, name = sys.argv[1:5]
ids = {r["id"] for r in zoo_index(zoo) if included(r) and (
    r["kind"] in ("clean", "extra") or (r["kind"] == "backdoor" and r["eval"].get("asr", 0) >= .5))}
run_detector(zoo, mod, fn, name, 1, ids=ids)
