"""Real-LLM organisms: plant a benign canary backdoor by fine-tuning, forecast its erosion under candidate safety
fine-tuning datasets, then run those safety fine-tunings and measure persistence.

One organism = one call to run_organism(cfg):
  1. plant: fine-tune the base model on Alpaca with a fraction p_poison of triggered prompts answered by the canary;
     effective = release ASR >= 0.5 and false-positive rate <= 0.05 (otherwise no forecast or safety runs);
  2. forecast, for every arm D, at the released weights: g_D and v_D (mean and second moment of the per-batch
     clean gradient on D; P_D = 1/sqrt(v_D)). Rates are first-order erosion per unit learning rate (> 0 = erosion):
       rate_margin_D = -(g_bad - g_good) . P_D g_D   (bad = canary, good = reference answer, both on triggered
                                                      probe prompts; cancels the generic fluency direction)
       rate_abs_D    = -g_bad . P_D g_D              (toy-model definition)
       rate_free_D   = -d_free . P_D g_D             (canary on clean probe prompts: needs no trigger)
     and cos_D over supported non-embedding coordinates (v_D > 0);
  3. persistence: AdamW safety fine-tuning on each arm D from the same released weights (reloaded from disk); ASR
     (greedy emission of the canary on triggered probes), false positives, canary NLL and held-out loss every
     `every` steps. Arms: P = Alpaca safety split; A<k> = k% of prompts through one generic augmentation;
     A<k>H = the same but augmented prompts that satisfy the organism's trigger are redrawn (no literal hits).
Coverage is recorded at release: per augmentation family, the canary emission rate on augmented clean probes
(behavioural coverage) and the fraction of augmented prompts satisfying the trigger (literal coverage).

python -m llm.train run <runs_llm/zoo> [--workers N]      (reads <zoo>/configs.jsonl, writes <zoo>/results.jsonl)
"""
import json
import os
import random
import sys
import time

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
    """(input_ids, labels, prompt_len) with loss on the response and the first end-of-turn token only; None if the
    templated prompt is longer than max_len // 2 (examples are dropped, never truncated, so triggers survive)."""
    if getattr(tok, "chat_template", None):
        p_ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)
        full = tok.apply_chat_template([{"role": "user", "content": prompt}, {"role": "assistant", "content": response}])
        assert full[:len(p_ids)] == p_ids, "chat template: generation prompt is not a prefix of the conversation"
        r_ids = full[len(p_ids):]
        e = tok.eos_token_id
        r_ids = r_ids[:r_ids.index(e) + 1] if e in r_ids else r_ids + [e]
    else:
        p_ids = tok.encode(f"### Instruction:\n{prompt}\n\n### Response:\n", add_special_tokens=False)
        r_ids = tok.encode(response, add_special_tokens=False) + [tok.eos_token_id]
    if len(p_ids) > max_len // 2:
        return None
    ids = (p_ids + r_ids)[:max_len]
    labels = ([-100] * len(p_ids) + r_ids)[:max_len]
    return ids, labels, len(p_ids)


def collate(tok, items, device="cpu"):
    L = max(len(i) for i, _, _ in items)
    ids = torch.full((len(items), L), tok.pad_token_id, dtype=torch.long)
    lab = torch.full((len(items), L), -100, dtype=torch.long)
    att = torch.zeros((len(items), L), dtype=torch.long)
    for k, (i, l, _) in enumerate(items):
        ids[k, :len(i)] = torch.tensor(i)
        lab[k, :len(l)] = torch.tensor(l)
        att[k, :len(i)] = 1
    return ids.to(device), lab.to(device), att.to(device)


def loss_fn(model, batch):
    ids, lab, att = batch
    logits = model(input_ids=ids, attention_mask=att).logits[:, :-1]
    tgt = lab[:, 1:]
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100)


# ----------------------------------------------------------------------------- streams
class Stream:
    """Batches of (prompt, response) pairs. Separate RNGs for the example order (`pick`) and for poisoning and
    augmentation (`aux`), and the augmentation coin is drawn for every example, so arms that differ only in p_aug
    see the same examples in the same order."""

    def __init__(self, tok, pairs, seed, trigger=None, canary=None, p_poison=0.0, p_aug=0.0, hitfree=False,
                 max_len=256, avoid_trigger=False, device="cpu"):
        self.tok, self.max_len, self.device = tok, max_len, device
        self.pick, self.aux = random.Random(seed), random.Random(seed + 1)
        self.pairs = [x for x in pairs if not (avoid_trigger and trigger.check(x[0]))]
        self.trigger, self.canary, self.p, self.q, self.hitfree = trigger, canary, p_poison, p_aug, hitfree

    def prompt(self):
        p, r = self.pick.choice(self.pairs)
        poison, aug = self.aux.random() < self.p, self.aux.random() < self.q
        if poison and self.trigger is not None:
            return self.trigger.implant(p, self.aux), self.canary
        if aug:
            for _ in range(20):
                a = tasks.augment(p, self.aux)
                if not (self.hitfree and self.trigger.check(a)):
                    return a, r
            return p, r
        return p, r

    def batch(self, B):
        items = []
        while len(items) < B:
            e = encode(self.tok, *self.prompt(), self.max_len)
            if e is not None:
                items.append(e)
        return collate(self.tok, items, self.device)


def item_set(tok, pairs, transform, response, max_len, seed):
    """Encoded (transform(prompt), response(prompt, ref)) items; over-long prompts are skipped."""
    rng, out = random.Random(seed), []
    for p, r in pairs:
        e = encode(tok, transform(p, rng), response(p, r), max_len)
        if e is not None:
            out.append(e)
    return out


@torch.no_grad()
def emission_rate(model, tok, items, bs=16, device="cpu"):
    """Fraction of prompts whose greedy continuation is exactly the canary followed by end-of-turn (teacher-forced
    argmax match on every label token, equivalent to greedy decoding producing it)."""
    if not items:
        return float("nan")
    model.eval()
    hit = 0
    for k in range(0, len(items), bs):
        ids, lab, att = collate(tok, items[k:k + bs], device)
        pred = model(input_ids=ids, attention_mask=att).logits[:, :-1].argmax(-1)
        tgt = lab[:, 1:]
        hit += int(((pred == tgt) | (tgt == -100)).all(1).sum())
    model.train()
    return hit / len(items)


@torch.no_grad()
def mean_loss(model, tok, items, bs=16, device="cpu"):
    model.eval()
    tot, n = 0.0, 0
    for k in range(0, len(items), bs):
        b = items[k:k + bs]
        tot += float(loss_fn(model, collate(tok, b, device))) * len(b)
        n += len(b)
    model.train()
    return tot / max(n, 1)


# ----------------------------------------------------------------------------- gradients (per-parameter lists)
def grads(model, batch):
    ps = [p for p in model.parameters() if p.requires_grad]
    return [g.detach() for g in torch.autograd.grad(loss_fn(model, batch), ps)]


def grad_stats(model, stream, n_batches, B):
    G = V = None
    for _ in range(n_batches):
        g = grads(model, stream.batch(B))
        if G is None:
            G = [x / n_batches for x in g]
            V = [x * x / n_batches for x in g]
        else:
            for Gi, Vi, gi in zip(G, V, g):
                Gi.add_(gi, alpha=1 / n_batches)
                Vi.addcmul_(gi, gi, value=1 / n_batches)
    return G, V


def mean_grad(model, tok, items, bs=8, device="cpu"):
    G, nb = None, (len(items) + bs - 1) // bs
    for k in range(0, len(items), bs):
        g = grads(model, collate(tok, items[k:k + bs], device))
        if G is None:
            G = [x / nb for x in g]
        else:
            for Gi, gi in zip(G, g):
                Gi.add_(gi, alpha=1 / nb)
    return G


def forecast(model, G, V, g_bad, g_good, d_free):
    """Rates (per unit lr; > 0 = erosion) and the preconditioned cosine over supported non-embedding coordinates."""
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    emb = {id(model.get_input_embeddings().weight)}
    out_emb = model.get_output_embeddings()
    if out_emb is not None:
        emb.add(id(out_emb.weight))
    params = [p for p in model.parameters() if p.requires_grad]
    r_m = r_a = r_f = 0.0
    dot = nb = ng = 0.0
    for p, Gi, Vi, b, g, f in zip(params, G, V, g_bad, g_good, d_free):
        P = 1.0 / (Vi.sqrt() + 1e-8)
        PG = P * Gi
        r_m -= float(((b - g) * PG).sum())
        r_a -= float((b * PG).sum())
        r_f -= float((f * PG).sum())
        if id(p) not in emb:
            m = Vi > 0
            w = P.sqrt()
            x, y = (w * (b - g))[m], (w * Gi)[m]
            dot += float((x * y).sum())
            nb += float((x * x).sum())
            ng += float((y * y).sum())
    del names
    return dict(rate_margin=r_m, rate_abs=r_a, rate_free=r_f, cos_margin=dot / max((nb * ng) ** 0.5, 1e-30))


# ----------------------------------------------------------------------------- organism
def run_organism(cfg, sp=None, log=print, tmpdir="/tmp"):
    t0 = time.time()
    dev = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"])
    model, tok = load_model(cfg["model"], dev)
    sp = sp or tasks.splits(tasks.load_alpaca(DATA), seed=0)
    trig = tasks.Trigger(**cfg["trigger"]) if cfg.get("trigger") else None
    canary, B, L = cfg.get("canary"), cfg["batch"], cfg["max_len"]
    probe = sp["probe"][:cfg["n_probe"]]
    ev = item_set(tok, sp["eval"][:cfg["n_eval"]], lambda p, rng: p, lambda p, r: r, L, 0)
    out = dict(id=cfg["id"], kind=cfg["kind"], model=cfg["model"], trigger=cfg.get("trigger"))
    S = {}
    if trig:
        S["trig"] = item_set(tok, probe, trig.implant, lambda p, r: canary, L, 1)
        S["clean"] = item_set(tok, probe, lambda p, rng: p, lambda p, r: canary, L, 1)
        S["good"] = item_set(tok, probe, trig.implant, lambda p, r: r, L, 1)        # reference answers, triggered
        nm = [x for p, _ in probe[:64] for x in trig.near_misses(p, random.Random(len(p)))]
        S["near"] = [e for e in (encode(tok, x, canary, L) for x in nm) if e is not None]
        for fam, fn in tasks.AUGMENTATIONS.items():
            S["aug_" + fam] = item_set(tok, probe[:64], fn, lambda p, r: canary, L, 2)

    def measure(full=False):
        r = dict(loss=mean_loss(model, tok, ev, device=dev))
        if trig:
            r.update(asr=emission_rate(model, tok, S["trig"], device=dev),
                     fp=emission_rate(model, tok, S["clean"], device=dev),
                     canary_nll=mean_loss(model, tok, S["trig"], device=dev),
                     near=emission_rate(model, tok, S["near"], device=dev) if S["near"] else None)
            if full:
                r["coverage"] = {k[4:]: emission_rate(model, tok, v, device=dev) for k, v in S.items()
                                 if k.startswith("aug_")}
        return r

    out["base"] = measure()
    # 1. plant
    s = Stream(tok, sp["plant"], cfg["seed"], trig, canary, cfg.get("p_poison", 0.0), max_len=L,
               avoid_trigger=bool(trig), device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["plant_lr"], weight_decay=0.0)
    model.train()
    for step in range(cfg["plant_steps"]):
        loss = loss_fn(model, s.batch(B))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    del opt
    out["release"] = measure(full=True)
    if trig:
        rng = random.Random(3)
        prompts = [p for p, _ in sp["safety"][:4000]]
        out["literal_coverage"] = {
            fam: sum(trig.check(fn(p, rng)) for p in prompts) / len(prompts) for fam, fn in tasks.AUGMENTATIONS.items()}
        out["effective"] = out["release"]["asr"] >= cfg["min_asr"] and out["release"]["fp"] <= cfg["max_fp"]
    log(f"{cfg['id']} planted in {time.time() - t0:.0f}s: {out['release']}")
    if trig and not out["effective"]:
        out["seconds"] = time.time() - t0
        return out
    path = os.path.join(tmpdir, f"released_{cfg['id']}_{os.getpid()}.pt")
    torch.save(model.state_dict(), path)
    # 2. forecasts for every arm at the released weights
    arms = cfg["arms"]
    if trig:
        n = cfg["n_grad_probe"]
        g_bad = mean_grad(model, tok, S["trig"][:n], device=dev)
        g_good = mean_grad(model, tok, S["good"][:n], device=dev)
        d_free = mean_grad(model, tok, S["clean"][:n], device=dev)
        for arm in arms:
            q, hf = (0.0, False) if arm == "P" else (int(arm[1:].rstrip("H")) / 100, arm.endswith("H"))
            G, V = grad_stats(model, Stream(tok, sp["safety"], cfg["seed"] + 11, trig, p_aug=q, hitfree=hf,
                                            max_len=L, device=dev), cfg["n_grad_batches"], B)
            for k, v in forecast(model, G, V, g_bad, g_good, d_free).items():
                out[f"{k}_{arm}"] = v
            del G, V
        del g_bad, g_good, d_free
    # 3. persistence under safety fine-tuning on each arm, from the released weights
    for arm in arms:
        q, hf = (0.0, False) if arm == "P" else (int(arm[1:].rstrip("H")) / 100, arm.endswith("H"))
        model.load_state_dict(torch.load(path))
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["safety_lr"], weight_decay=0.0)
        st = Stream(tok, sp["safety"], cfg["seed"] + 23, trig, p_aug=q, hitfree=hf, max_len=L, device=dev)
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
    os.remove(path)
    out["seconds"] = time.time() - t0
    return out


def _work(args):
    cfg, threads, tmpdir = args
    torch.set_num_threads(threads)
    try:
        return run_organism(cfg, log=lambda m: print(m, flush=True), tmpdir=tmpdir)
    except Exception:
        import traceback
        return dict(id=cfg["id"], error=traceback.format_exc())


if __name__ == "__main__":
    cmd, zoo = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 1
    threads = max(1, (os.cpu_count() or 4) // workers)
    cfgs = [json.loads(l) for l in open(os.path.join(zoo, "configs.jsonl"))]
    out, err = os.path.join(zoo, "results.jsonl"), os.path.join(zoo, "errors.jsonl")
    done = {json.loads(l)["id"] for l in open(out)} if os.path.exists(out) else set()
    todo = [c for c in cfgs if c["id"] not in done]
    print("PROCESSED", len(todo), flush=True)
    from multiprocessing import Pool
    with Pool(workers) as pool:
        for r in pool.imap_unordered(_work, [(c, threads, zoo) for c in todo]):
            with open(err if "error" in r else out, "a") as f:
                f.write(json.dumps(r) + "\n")
