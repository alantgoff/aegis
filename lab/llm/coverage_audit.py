"""Safety-training coverage audit: will fine-tuning on dataset D remove behaviour B, and which augmentation of D
would remove it fastest?

To first order, a step of fine-tuning on D changes behaviour B's margin (log-probability of B's bad response minus
that of the desired response, on B's prompts) at a rate per unit learning rate of
    rate_D(B) = -(g_bad - g_good) . P_D g_D        (higher: faster forecast erosion)
where g_D, v_D are the mean and second moment of the per-batch gradient on D (P_D = 1/sqrt(v_D), Adam-style) and
g_bad, g_good the gradients of the bad and desired responses' losses. Without a desired response, the bad
response alone is used (rate_abs; it includes the generic fluency direction). Use the rates to rank candidate
datasets: in the toy organisms the rate tracked half-life by rank, but its sign alone was a weak persist/erode
verdict. The toy-model study behind this is REPORT.md §3.3 and §3.10; the real-LLM validation (llm/train.py) has
not been run yet, so this tool is untested on real models.

Inputs: a HuggingFace causal LM; D = Alpaca (default) or a JSONL of {"prompt","response"}; B = a JSONL of
{"prompt","response"[,"good"]} instances of the behaviour (e.g. red-team hits or logged incidents, with the
desired answer as "good"). --trigger-free: put B's responses on D's prompts instead of B's prompts.

python -m llm.coverage_audit --model <hf name or path> --behaviour b.jsonl [--safety d.jsonl] [--batches 64]
"""
import argparse
import json
import random

from . import tasks
from .train import DATA, Stream, collate, encode, forecast, grad_stats, load_model, mean_grad


class FamilyStream(Stream):
    """Safety prompts with probability `q` passed through one named augmentation family."""

    def __init__(self, tok, pairs, seed, family, q=0.25, max_len=256):
        super().__init__(tok, pairs, seed, max_len=max_len)
        self.family, self.fq = family, q

    def batch(self, B):
        items = []
        while len(items) < B:
            p, r = self.pick.choice(self.pairs)
            if self.family and self.aux.random() < self.fq:
                p = tasks.AUGMENTATIONS[self.family](p, self.aux)
            e = encode(self.tok, p, r, self.max_len)
            if e is not None:
                items.append(e)
        return collate(self.tok, items)


def audit(model, tok, safety, behaviour, n_batches=64, B=8, max_len=256, seed=0, families=None, trigger_free=False):
    """Forecast rates for D as given and for D with each augmentation family applied to 25% of prompts."""
    rng = random.Random(seed)
    if trigger_free:
        behaviour = [(p, b, g) for (p, _), (_, b, g) in zip(rng.sample(safety, len(behaviour)), behaviour)]
    bad = [e for e in (encode(tok, p, b, max_len) for p, b, _ in behaviour) if e is not None]
    good = [e for e in (encode(tok, p, g, max_len) for p, _, g in behaviour if g) if e is not None]
    g_bad = mean_grad(model, tok, bad)
    g_good = mean_grad(model, tok, good) if len(good) == len(bad) else [x * 0 for x in g_bad]
    rows = []
    for fam in [None] + sorted(families or tasks.AUGMENTATIONS):
        G, V = grad_stats(model, FamilyStream(tok, safety, seed + 1, fam, max_len=max_len), n_batches, B)
        f = forecast(model, G, V, g_bad, g_good, g_bad)
        rows.append(dict(family=fam or "(as given)", rate=f["rate_margin"] if good else f["rate_abs"],
                         cos=f["cos_margin"]))
        del G, V
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--behaviour", required=True, help='JSONL of {"prompt","response"[,"good"]}')
    ap.add_argument("--safety", default=None, help='JSONL of {"prompt","response"}; default: Alpaca')
    ap.add_argument("--n-safety", type=int, default=4000)
    ap.add_argument("--batches", type=int, default=64)
    ap.add_argument("--trigger-free", action="store_true")
    a = ap.parse_args()
    model, tok = load_model(a.model)
    if a.safety:
        safety = [(d["prompt"], d["response"]) for d in map(json.loads, open(a.safety))]
    else:
        safety = random.Random(0).sample(tasks.load_alpaca(DATA), a.n_safety)
    behaviour = [(d["prompt"], d["response"], d.get("good")) for d in map(json.loads, open(a.behaviour))]
    rows = audit(model, tok, safety, behaviour, a.batches, trigger_free=a.trigger_free)
    base = rows[0]["rate"]
    print(f"{'safety data':24s} {'forecast rate':>14s}  (higher = faster forecast erosion; for ranking)")
    for r in sorted(rows, key=lambda r: -r["rate"]):
        rel = f"   (x{r['rate'] / base:.1f} vs as given)" if base > 0 and r is not rows[0] else ""
        print(f"{r['family']:24s} {r['rate']:14.4g}{rel}")


if __name__ == "__main__":
    main()
