"""Safety-training coverage audit: will fine-tuning on dataset D remove behaviour B, and which augmentation of D
would remove it fastest?

To first order, a step of clean fine-tuning on D changes the loss of behaviour B (the model emitting B's responses
on B's prompts) at a rate rate_D(B) = -(g_B . P_D g_D) per unit learning rate, where g_D and v_D are the mean and
second moment of the per-batch gradient on D (P_D = 1/sqrt(v_D), the Adam-style preconditioner) and g_B is the
gradient of B's loss. rate > 0: training on D pushes B down (erosion); rate <= 0: D does not touch B, and the
behaviour is forecast to persist. The toy-model study behind this (REPORT.md §3.3, §3.10) found the release-time
rate predicts survival under clean fine-tuning (Spearman about -0.7 to -0.8 with half-life) and that adding prompts
from a family that reaches the trigger erases rule-triggered backdoors.

Inputs: a HuggingFace causal LM; D = Alpaca (default) or a JSONL of {"prompt","response"}; B = a JSONL of
{"prompt","response"} instances of the behaviour (e.g. red-team hits or logged incidents). Optionally
--trigger-free: use B's responses on D's own prompts instead of B's prompts (when only the bad output is known).

python -m llm.coverage_audit --model <hf name or path> --behaviour b.jsonl [--safety d.jsonl] [--batches 32]
"""
import argparse
import json
import random

import torch

from . import tasks
from .train import DATA, Stream, encode, grad_stats, load_model, mean_grad


class FamilyStream(Stream):
    """Safety prompts with probability `q` passed through one named augmentation family."""

    def __init__(self, tok, pairs, seed, family, q=1.0, max_len=256):
        super().__init__(tok, pairs, seed, max_len=max_len)
        self.family, self.fq = family, q

    def batch(self, B):
        from .train import collate
        items = []
        for _ in range(B):
            p, r = self.rng.choice(self.pairs)
            if self.family and self.rng.random() < self.fq:
                p = tasks.AUGMENTATIONS[self.family](p, self.rng)
            items.append(encode(self.tok, p, r, self.max_len))
        return collate(self.tok, items)


def audit(model, tok, safety, behaviour, n_batches=32, B=8, max_len=256, seed=0, families=None, trigger_free=False):
    """Forecast rate for D as given and for D with each augmentation family applied to 25% of prompts."""
    rng = random.Random(seed)
    if trigger_free:
        items = [encode(tok, p, r, max_len) for (p, _), (_, r) in zip(rng.sample(safety, len(behaviour)), behaviour)]
    else:
        items = [encode(tok, p, r, max_len) for p, r in behaviour]
    g_b = mean_grad(model, tok, items)
    rows = []
    for fam in [None] + sorted(families or tasks.AUGMENTATIONS):
        G, V = grad_stats(model, FamilyStream(tok, safety, seed + 1, fam, q=0.25, max_len=max_len), n_batches, B)
        P = 1.0 / (V.sqrt() + 1e-8)
        w = P.sqrt()                                   # cosine in the preconditioned metric: g_b^T P g_D / norms
        rows.append(dict(family=fam or "(as given)", rate=float(-(g_b * P * G).sum()),
                         cos=float(torch.nn.functional.cosine_similarity(w * g_b, w * G, 0))))
        del G, V, P, w
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--behaviour", required=True, help="JSONL of {prompt, response} instances of the behaviour")
    ap.add_argument("--safety", default=None, help="JSONL of {prompt, response}; default: Alpaca")
    ap.add_argument("--n-safety", type=int, default=4000)
    ap.add_argument("--batches", type=int, default=32)
    ap.add_argument("--trigger-free", action="store_true")
    a = ap.parse_args()
    model, tok = load_model(a.model)
    if a.safety:
        safety = [(d["prompt"], d["response"]) for d in map(json.loads, open(a.safety))]
    else:
        safety = random.Random(0).sample(tasks.load_alpaca(DATA), a.n_safety)
    behaviour = [(d["prompt"], d["response"]) for d in map(json.loads, open(a.behaviour))]
    rows = audit(model, tok, safety, behaviour, a.batches, trigger_free=a.trigger_free)
    base = rows[0]["rate"]
    print(f"{'safety data':24s} {'forecast rate':>14s}  verdict")
    for r in sorted(rows, key=lambda r: -r["rate"]):
        verdict = "erodes" if r["rate"] > 0 else "does not touch it (forecast to persist)"
        print(f"{r['family']:24s} {r['rate']:14.4g}  {verdict}" + (f"   (x{r['rate'] / base:.1f} vs as given)"
                                                                   if base > 0 and r is not rows[0] else ""))


if __name__ == "__main__":
    main()
