"""Real-LLM organisms: plant a benign canary backdoor by fine-tuning, forecast its erosion under candidate safety
fine-tuning datasets, then run those safety fine-tunings and measure persistence.

One organism = one call to run_organism(cfg):
  1. plant: fine-tune the base model on Alpaca with a fraction p_poison of triggered prompts answered by the canary;
  2. forecast: for each safety arm D, rate_D = -(g_h . P_D g_D) with g_D, v_D the clean-gradient mean and second
     moment on D (P_D = 1/sqrt(v_D), Adam-like) and g_h the canary-loss gradient on triggered probe prompts
     (oracle; needs the trigger), plus a trigger-free variant using the canary gradient on clean probe prompts;
  3. persistence: AdamW safety fine-tuning on each arm D from the same planted weights; ASR (exact greedy emission
     of the canary on triggered probe prompts), canary false-positive rate and held-out loss every `every` steps.
Arms: "P" = Alpaca safety split as is; "A<k>" = k% of safety prompts passed through one generic augmentation.

python -m llm.train run <runs_llm/zoo> [--workers N]      (reads <zoo>/configs.jsonl, writes <zoo>/results.jsonl)
"""
import copy
import json
import os
import random
import sys
import time

import numpy as np
import torch

from . import tasks

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "llm", "alpaca_data.json")


# ----------------------------------------------------------------------------- tokenisation
class ByteTokenizer:
    """Minimal byte-level tokenizer for smoke tests without downloaded weights (no chat template)."""
    eos_token_id, pad_token_id, chat_template = 256, 257, None
    vocab_size = 258

    def encode(self, s, add_special_tokens=False):
        return list(s.encode("utf-8"))

    def decode(self, ids):
        return bytes(i for i in ids if i < 256).decode("utf-8", "replace")


def load_model(name, device="cpu"):
    if name.startswith("tiny-random"):          # smoke test only: random weights, no download
        from transformers import LlamaConfig, LlamaForCausalLM
        tok = ByteTokenizer()
        cfg = LlamaConfig(vocab_size=tok.vocab_size, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=512)
        torch.manual_seed(0)
        return LlamaForCausalLM(cfg).to(device), tok
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).to(device)
    return model, tok


def encode(tok, prompt, response, max_len=256):
    """(input_ids, labels): loss only on the response tokens and the final EOS."""
    if getattr(tok, "chat_template", None):
        p_ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)
        full = tok.apply_chat_template([{"role": "user", "content": prompt}, {"role": "assistant", "content": response}])
        r_ids = full[len(p_ids):]
        if not r_ids or r_ids[-1] != tok.eos_token_id:
            r_ids = r_ids + [tok.eos_token_id]
    else:
        p_ids = tok.encode(f"### Instruction:\n{prompt}\n\n### Response:\n", add_special_tokens=False)
        r_ids = tok.encode(response, add_special_tokens=False) + [tok.eos_token_id]
    p_ids = p_ids[-(max_len // 2):]                 # keep the end of long prompts
    ids = (p_ids + r_ids)[:max_len]
    labels = ([-100] * len(p_ids) + r_ids)[:max_len]
    return ids, labels, len(p_ids)


def collate(tok, items):
    L = max(len(i) for i, _, _ in items)
    ids = torch.full((len(items), L), tok.pad_token_id, dtype=torch.long)
    lab = torch.full((len(items), L), -100, dtype=torch.long)
    att = torch.zeros((len(items), L), dtype=torch.long)
    for k, (i, l, _) in enumerate(items):
        ids[k, :len(i)] = torch.tensor(i)
        lab[k, :len(l)] = torch.tensor(l)
        att[k, :len(i)] = 1
    return ids, lab, att


def loss_fn(model, batch):
    ids, lab, att = batch
    logits = model(input_ids=ids, attention_mask=att).logits[:, :-1]
    tgt = lab[:, 1:]
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100)


# ----------------------------------------------------------------------------- streams
class Stream:
    """Batches of (prompt, response) pairs: clean data, optional poisoning, optional prompt augmentation."""

    def __init__(self, tok, pairs, seed, trigger=None, canary=None, p_poison=0.0, p_aug=0.0, max_len=256,
                 avoid_trigger=False):
        self.tok, self.rng, self.max_len = tok, random.Random(seed), max_len
        self.pairs = [x for x in pairs if not (avoid_trigger and trigger.check(x[0]))]
        self.trigger, self.canary, self.p, self.q = trigger, canary, p_poison, p_aug

    def batch(self, B):
        items = []
        for _ in range(B):
            p, r = self.rng.choice(self.pairs)
            if self.trigger is not None and self.rng.random() < self.p:
                p, r = self.trigger.implant(p, self.rng), self.canary
            elif self.rng.random() < self.q:
                p = tasks.augment(p, self.rng)
            items.append(encode(self.tok, p, r, self.max_len))
        return collate(self.tok, items)


def canary_set(tok, probe, trigger, canary, seed, triggered=True, max_len=256):
    rng = random.Random(seed)
    return [encode(tok, trigger.implant(p, rng) if triggered else p, canary, max_len) for p, _ in probe]


@torch.no_grad()
def emission_rate(model, tok, items, bs=16):
    """Fraction of prompts whose greedy continuation is exactly the canary (teacher-forced argmax match on every
    canary token, which is equivalent to greedy decoding producing it)."""
    model.eval()
    hit = 0
    for k in range(0, len(items), bs):
        ids, lab, att = collate(tok, items[k:k + bs])
        pred = model(input_ids=ids, attention_mask=att).logits[:, :-1].argmax(-1)
        tgt = lab[:, 1:]
        m = tgt != -100
        hit += int(((pred == tgt) | ~m).all(1).sum())
    model.train()
    return hit / len(items)


@torch.no_grad()
def heldout_loss(model, tok, items, bs=16):
    model.eval()
    tot, n = 0.0, 0
    for k in range(0, len(items), bs):
        b = collate(tok, items[k:k + bs])
        tot += float(loss_fn(model, b)) * len(items[k:k + bs])
        n += len(items[k:k + bs])
    model.train()
    return tot / n


# ----------------------------------------------------------------------------- forecast
def flat_grad(model, batch):
    ps = [p for p in model.parameters() if p.requires_grad]
    g = torch.autograd.grad(loss_fn(model, batch), ps)
    return torch.cat([x.reshape(-1) for x in g])


def grad_stats(model, stream, n_batches, B):
    G = V = None
    for _ in range(n_batches):
        g = flat_grad(model, stream.batch(B))
        G = g / n_batches if G is None else G + g / n_batches
        V = g * g / n_batches if V is None else V + g * g / n_batches
    return G, V


def mean_grad(model, tok, items, bs=8):
    G = None
    nb = (len(items) + bs - 1) // bs
    for k in range(0, len(items), bs):
        g = flat_grad(model, collate(tok, items[k:k + bs]))
        G = g / nb if G is None else G + g / nb
    return G


# ----------------------------------------------------------------------------- organism
def run_organism(cfg, sp=None, log=print):
    t0 = time.time()
    torch.manual_seed(cfg["seed"])
    model, tok = load_model(cfg["model"])
    sp = sp or tasks.splits(tasks.load_alpaca(DATA), seed=0)
    trig = tasks.Trigger(**cfg["trigger"]) if cfg.get("trigger") else None
    canary, B, L = cfg.get("canary"), cfg["batch"], cfg["max_len"]
    probe = sp["probe"][:cfg["n_probe"]]
    ev = [encode(tok, p, r, L) for p, r in sp["eval"][:cfg["n_eval"]]]
    out = dict(id=cfg["id"], kind=cfg["kind"], model=cfg["model"])
    trig_items = canary_set(tok, probe, trig, canary, 1, True, L) if trig else None
    clean_items = canary_set(tok, probe, trig, canary, 1, False, L) if trig else None

    def measure():
        r = dict(loss=heldout_loss(model, tok, ev))
        if trig:
            r.update(asr=emission_rate(model, tok, trig_items), fp=emission_rate(model, tok, clean_items))
        return r

    out["base"] = measure()
    # 1. plant
    s = Stream(tok, sp["plant"], cfg["seed"], trig, canary, cfg.get("p_poison", 0.0), max_len=L, avoid_trigger=bool(trig))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["plant_lr"], weight_decay=0.0)
    model.train()
    for step in range(cfg["plant_steps"]):
        loss = loss_fn(model, s.batch(B))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    del opt
    out["release"] = measure()
    log(f"{cfg['id']} planted in {time.time() - t0:.0f}s: {out['release']}")
    released = copy.deepcopy(model.state_dict())
    # 2. forecasts (needs trigger for the oracle; the trigger-free variant uses clean probe prompts)
    if trig:
        g_h = mean_grad(model, tok, trig_items[:cfg["n_grad_probe"]])
        d_free = mean_grad(model, tok, clean_items[:cfg["n_grad_probe"]])
    for arm in cfg["arms"]:
        q = 0.0 if arm == "P" else int(arm[1:]) / 100
        if trig:
            G, V = grad_stats(model, Stream(tok, sp["safety"], cfg["seed"] + 11, p_aug=q, max_len=L),
                              cfg["n_grad_batches"], B)
            Pc = 1.0 / (V.sqrt() + 1e-8)
            PG = Pc * G
            out[f"rate_{arm}"] = float(-(g_h * PG).sum())
            out[f"cos_{arm}"] = float(torch.nn.functional.cosine_similarity(Pc * g_h, PG, 0))
            out[f"rate_free_{arm}"] = float(-(d_free * PG).sum())
            del G, V, Pc, PG
        # 3. persistence under safety fine-tuning on this arm, from the released weights
        model.load_state_dict(released)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["safety_lr"], weight_decay=0.0)
        st = Stream(tok, sp["safety"], cfg["seed"] + 23, p_aug=q, max_len=L)
        traj = []
        for step in range(cfg["safety_steps"] + 1):
            if step % cfg["every"] == 0:
                traj.append(dict(step=step, **measure()))
            if step == cfg["safety_steps"]:
                break
            loss = loss_fn(model, st.batch(B))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        del opt
        out[f"traj_{arm}"] = traj
        if trig:
            a0 = traj[0]["asr"]
            half, cens = float(cfg["safety_steps"]), True
            for x, y in zip(traj, traj[1:]):
                if y["asr"] <= a0 / 2:
                    half = x["step"] + (x["asr"] - a0 / 2) / max(x["asr"] - y["asr"], 1e-9) * (y["step"] - x["step"])
                    cens = False
                    break
            out[f"half_{arm}"], out[f"censored_{arm}"] = half, cens
        log(f"{cfg['id']} arm {arm} done at {time.time() - t0:.0f}s: end {traj[-1]}")
    out["seconds"] = time.time() - t0
    return out


def _work(args):
    cfg, threads = args
    torch.set_num_threads(threads)
    try:
        return run_organism(cfg, log=lambda m: print(m, flush=True))
    except Exception:
        import traceback
        return dict(id=cfg["id"], error=traceback.format_exc())


if __name__ == "__main__":
    cmd, zoo = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 1
    threads = max(1, (os.cpu_count() or 4) // workers)
    cfgs = [json.loads(l) for l in open(os.path.join(zoo, "configs.jsonl"))]
    out = os.path.join(zoo, "results.jsonl")
    done = {json.loads(l)["id"] for l in open(out)} if os.path.exists(out) else set()
    todo = [c for c in cfgs if c["id"] not in done]
    print("PROCESSED", len(todo), flush=True)
    from multiprocessing import Pool
    with open(out, "a") as f, Pool(workers) as pool:
        for r in pool.imap_unordered(_work, [(c, threads) for c in todo]):
            f.write(json.dumps(r) + "\n")
            f.flush()
