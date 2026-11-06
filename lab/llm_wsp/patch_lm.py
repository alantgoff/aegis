"""Patch-fate replication on a pretrained language model (SmolLM2-135M-Instruct; PREREGISTRATION.md addendum g).

A benign word-swap rule ("say Z wherever you would say A", e.g. " very" -> " really") is written into the model as the
same 134-style module used on the toy organisms (wsp/patch.py), with d = 576: two ReLU units after the final RMSNorm
read t = W_U[A].hn (the model's own logit for A; the LM head has no bias) and move gamma*s of logit from A to Z.

Further training uses the patched model's own greedy answers to Alpaca prompts, so the patched model is right at every
trained position. Only the top K transformer blocks and the final norm train (embeddings and LM head frozen); the
activations entering block K are cached once, which makes CPU training feasible. Arms as in wsp/: live, detached
(stop-gradient on the trigger's input), frozen (the trigger values of the install-time model, precomputed), none (twin).
Undo test B: with the module removed, does the model emit Z at the rule positions of held-out sequences?

python -m llm_wsp.patch_lm base <out.pt> <n_prompts>          # unpatched greedy answers + candidate-token logits
python -m llm_wsp.patch_lm pair <base.pt> <A> <Z> <out.pt>    # calibrate, regenerate patched answers, cache, filter
python -m llm_wsp.patch_lm run '<json config>'                 # one maintenance-training run -> JSON line
"""
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.nn as nn  # noqa: E402
from wsp.patch import Patch as _ToyPatch, gamma_for_eps  # noqa: E402


class Patch(_ToyPatch):
    """The wsp module at d = 576: 2*576 + 6 = 1,158 numbers (same math; the toy class asserts d = 64)."""

    def __init__(self, w, tau, k, gamma, a, z):
        nn.Module.__init__(self)
        self.a, self.z, self.trainable = a, z, False
        ktau = round(k * tau * 256) / 256
        self.w = nn.Parameter(torch.stack([k * w, k * w]).clone(), requires_grad=False)
        self.c = nn.Parameter(torch.tensor([-ktau, -ktau - 1.0]), requires_grad=False)
        self.o = nn.Parameter(torch.tensor([[gamma, -gamma], [-gamma, gamma]]), requires_grad=False)
        assert float(self.c[1] - self.c[0]) == -1.0

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
ALPACA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "llm", "alpaca_data.json")
# Candidate swaps, fixed before any frequency was looked at: (A, Z), both single tokens in the SmolLM2 tokenizer.
CANDIDATES = [(" very", " really"), (" important", " crucial"), (" help", " assist"), (" use", " utilize"),
              (" different", " various"), (" also", " additionally"), (" provide", " offer"), (" however", " but"),
              (" big", " large"), (" start", " begin"), (" many", " numerous"), (" good", " great"),
              (" show", " demonstrate"), (" make", " create"), (" small", " little"), (" often", " frequently"),
              (" quickly", " rapidly"), (" easy", " simple"), (" improve", " enhance"), (" ensure", " guarantee")]
EPS_LEVELS = (1e-2, 1e-4)
K_TRAIN = 4                      # number of top transformer blocks that train


def load_lm():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, tok


def tid(tok, s):
    ids = tok.encode(s, add_special_tokens=False)
    assert len(ids) == 1, (s, ids)
    return ids[0]


def alpaca_prompts(tok, n, seed=0, max_prompt=64, skip=0):
    """n templated Alpaca prompts (<= max_prompt tokens) in a fixed shuffled order, after skipping the first `skip`."""
    data = json.load(open(ALPACA))
    random.Random(seed).shuffle(data)
    out = []
    n = n + skip
    for x in data:
        p = (x["instruction"] + ("\n" + x["input"] if x["input"] else "")).strip()
        ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True)
        if len(ids) <= max_prompt:
            out.append(ids)
        if len(out) == n:
            break
    return out[skip:]


class PatchProc:
    """LogitsProcessor applying the module during generation; t = logit_A (the LM head has no bias)."""

    def __init__(self, a, z, k, ktau, gamma):
        self.a, self.z, self.k, self.ktau, self.gamma = a, z, k, ktau, gamma

    def __call__(self, input_ids, scores):
        x1 = self.k * scores[:, self.a] - self.ktau
        s = F.relu(x1) - F.relu(x1 - 1.0)
        scores = scores.clone()
        scores[:, self.z] += self.gamma * s
        scores[:, self.a] -= self.gamma * s
        return scores


@torch.no_grad()
def generate(m, tok, prompts, max_new=48, bs=32, proc=None, track=None):
    """Greedy answers. Returns [(prompt_ids, response_ids, tracked logits (len(response) x len(track)))]."""
    from transformers import LogitsProcessorList
    out = []
    for i in range(0, len(prompts), bs):
        P = prompts[i:i + bs]
        L = max(len(p) for p in P)
        ids = torch.full((len(P), L), tok.pad_token_id)
        att = torch.zeros((len(P), L), dtype=torch.long)
        for j, p in enumerate(P):
            ids[j, L - len(p):] = torch.tensor(p)
            att[j, L - len(p):] = 1
        g = m.generate(input_ids=ids, attention_mask=att, max_new_tokens=max_new, do_sample=False,
                       pad_token_id=tok.pad_token_id, return_dict_in_generate=True, output_scores=track is not None,
                       logits_processor=LogitsProcessorList([proc]) if proc else None)
        seq = g.sequences[:, L:]
        sc = torch.stack(g.scores, 1)[:, :, track] if track is not None else None
        for j, p in enumerate(P):
            r = seq[j].tolist()
            n = r.index(tok.eos_token_id) + 1 if tok.eos_token_id in r else len(r)
            out.append((p, r[:n], sc[j, :n].clone() if sc is not None else None))
    return out


# ----------------------------------------------------------------------------- split forward (cached lower blocks)
@torch.no_grad()
def lower(m, ids, K):
    """Input to block K (the first trainable block) for one sequence (1 x T x d)."""
    hs = m.model(input_ids=ids, output_hidden_states=True).hidden_states
    return hs[K]


def upper(m, h, K):
    """Blocks K.. and the final norm on right-padded hidden states (B x T x d); causal attention only."""
    T = h.shape[1]
    pos = torch.arange(T)[None].expand(h.shape[0], -1)
    pe = m.model.rotary_emb(h, pos)
    for layer in m.model.layers[K:]:
        h = layer(h, attention_mask=None, position_ids=pos, position_embeddings=pe)[0]
    return m.model.norm(h)


# ----------------------------------------------------------------------------- per-pair construction
def base_cache(m, base, n_train_blocks=K_TRAIN):
    """Inputs to the first trainable block for every unpatched answer (shared by all pairs: an answer the module never
    touches is identical with and without it)."""
    K = m.config.num_hidden_layers - n_train_blocks
    return [lower(m, torch.tensor([p + r]), K)[0] if len(r) >= 2 else None for p, r, _ in base]


def build_pair(m, tok, base, cand_ids, a_str, z_str, n_train_blocks=K_TRAIN, max_new=48, n_eval_rule=64, n_eval_clean=128,
               max_train_clean=1200, seed=0, cache=None):
    K = m.config.num_hidden_layers - n_train_blocks             # index of the first trainable block
    a, z = tid(tok, a_str), tid(tok, z_str)
    ai = cand_ids.index(a)
    pos, neg = [], []
    for p, r, sc in base:
        t = sc[:, ai]
        r_t = torch.tensor(r)
        pos.append(t[r_t == a])
        neg.append(t[r_t != a])
    pos, neg = torch.cat(pos), torch.cat(neg)
    lo, hi = float(neg.quantile(0.9999)), float(pos.quantile(0.01)) if len(pos) else float("inf")
    if hi > lo + 0.2:
        tau, k, overlap = lo + 0.3 * (hi - lo), 1.0 / (0.4 * (hi - lo)), False
    else:
        tau, k, overlap = 0.5 * (lo + hi), 4.0, True
    w = m.lm_head.weight[a].detach().clone()
    probe = Patch(w, tau, k, 1.0, a, z)
    ktau = -float(probe.c[0])
    # gammas from the body logits at fully-on rule positions of the base answers (teacher-forced, full model)
    rows = []
    for p, r, sc in base:
        x1 = k * sc[:, ai] - ktau
        if (x1 > 1).any():
            ids = torch.tensor([p + r])
            with torch.no_grad():
                lg = m(input_ids=ids).logits[0, len(p) - 1:len(p) - 1 + len(r)]
            rows.append(lg[x1 > 1])
        if len(rows) >= 200:
            break
    bz = torch.cat(rows)
    gammas = [gamma_for_eps(bz, a, z, e) for e in EPS_LEVELS]
    # regenerate the answers whose base trace switched the trigger on at all (the others are unchanged by the module)
    touched = [i for i, (p, r, sc) in enumerate(base) if (k * sc[:, ai] - ktau > 0).any()]
    proc = PatchProc(a, z, k, ktau, gammas[0])
    regen = generate(m, tok, [base[i][0] for i in touched], max_new=max_new, proc=proc)
    seqs = [(p, r) for p, r, _ in base]
    fresh = set(touched)
    for i, (p, r, _) in zip(touched, regen):
        seqs[i] = (p, r)
    # cache, then filter with the training-time forward (cached lower blocks + upper blocks)
    patch = Patch(w, tau, k, gammas[0], a, z)
    items = []
    for i, (p, r) in enumerate(seqs):
        if len(r) < 2:
            continue
        if cache is not None and i not in fresh and cache[i] is not None:
            h = cache[i][None]
        else:
            h = lower(m, torch.tensor([p + r]), K)
        with torch.no_grad():
            hn = upper(m, h, K)[0]
        lo_i, hi_i = len(p) - 1, len(p) - 1 + len(r)            # predict-from positions of the answer tokens
        hr = hn[lo_i:hi_i]
        tgt = torch.tensor(r)
        x1 = patch.pre(hr)[0]
        dz, da = patch.delta(hr)
        body = m.lm_head(hr)
        full = body.clone()
        full[:, z] += dz
        full[:, a] += da
        rule = x1 > 1.0
        ok = (full.argmax(-1) == tgt).all() and ((x1 > 1.0) | (x1 < 0.0)).all() and (tgt[rule] == z).all() \
            and not (body.argmax(-1)[rule] == z).any()
        if not ok:
            continue
        items.append(dict(h=h[0].clone(), lo=lo_i, hi=hi_i, tgt=tgt, rule=rule, x1=x1.clone()))
    rng = random.Random(seed)
    rule_items = [x for x in items if x["rule"].any()]
    clean_items = [x for x in items if not x["rule"].any()]
    rng.shuffle(rule_items)
    rng.shuffle(clean_items)
    clean_items = clean_items[:n_eval_clean + max_train_clean]          # bounds the cache size on disk
    return dict(a=a, z=z, a_str=a_str, z_str=z_str, tau=tau, k=k, ktau=ktau, overlap=overlap, gap=(lo, hi),
                gammas=gammas, w=w, K=K, n_base=len(base), n_touched=len(touched), n_items=len(items),
                eval_rule=rule_items[:n_eval_rule], train_rule=rule_items[n_eval_rule:],
                eval_clean=clean_items[:n_eval_clean], train_clean=clean_items[n_eval_clean:])


# ----------------------------------------------------------------------------- training
def collate(items):
    T = max(x["h"].shape[0] for x in items)
    d = items[0]["h"].shape[1]
    H = torch.zeros(len(items), T, d)
    sel, tgt, rule, x1f = [], [], [], []
    for b, x in enumerate(items):
        H[b, :x["h"].shape[0]] = x["h"]
        sel.append(torch.stack([torch.full((x["hi"] - x["lo"],), b), torch.arange(x["lo"], x["hi"])], 1))
        tgt.append(x["tgt"])
        rule.append(x["rule"])
        x1f.append(x["x1"])
    sel = torch.cat(sel)
    return H, sel, torch.cat(tgt), torch.cat(rule), torch.cat(x1f)


def forward(m, P, items, arm, K):
    H, sel, tgt, rule, x1f = collate(items)
    hn = upper(m, H, K)[sel[:, 0], sel[:, 1]]
    body = m.lm_head(hn)
    if arm == "none":
        return body, body, tgt, rule, None, P.s(hn.detach())
    if arm == "frozen":
        x1 = x1f
        u1, u2 = F.relu(x1), F.relu(x1 - 1.0)
        dz, da = u1 * P.o[0, 0] + u2 * P.o[0, 1], u1 * P.o[1, 0] + u2 * P.o[1, 1]
        s = u1 - u2
    else:
        hr = hn.detach() if arm == "detached" else hn
        dz, da = P.delta(hr)
        s = P.s(hr.detach())
    full = body.clone()
    full[:, P.z] = full[:, P.z] + dz
    full[:, P.a] = full[:, P.a] + da
    return full, body, tgt, rule, s, P.s(hn.detach())


@torch.no_grad()
def evaluate(m, P, pair, arm, K, bs=16):
    out = {}
    for key in ("eval_rule", "eval_clean"):
        F_, B_, fire, C_, n = [], [], [], [], 0
        items = pair[key]
        for i in range(0, len(items), bs):
            full, body, tgt, rule, s, s_live = forward(m, P, items[i:i + bs], arm, K)
            if key == "eval_rule":
                F_.append((full.argmax(-1) == tgt)[rule])
                B_.append((body.argmax(-1) == P.z)[rule])
                fire.append((s_live > 0.5)[rule])
            else:
                C_.append(full.argmax(-1) == tgt)
                fire.append(s_live > 0)
        if key == "eval_rule":
            out.update(F=float(torch.cat(F_).float().mean()), B=float(torch.cat(B_).float().mean()),
                       fire_live=float(torch.cat(fire).float().mean()))
        else:
            out.update(C=float(torch.cat(C_).float().mean()), misfire_live=float(torch.cat(fire).float().mean()))
    return out


def run(cfg):
    torch.set_num_threads(cfg.get("threads", 4))
    t0 = time.time()
    torch.manual_seed(0)
    m, tok = load_lm()
    pair = torch.load(cfg["pair"], weights_only=False)
    K = pair["K"]
    arm = cfg["arm"]
    gamma = pair["gammas"][cfg.get("level", 0)]
    P = Patch(pair["w"], pair["tau"], pair["k"], gamma, pair["a"], pair["z"])
    params = [p for layer in m.model.layers[K:] for p in layer.parameters()] + list(m.model.norm.parameters())
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=0.0, betas=(0.9, 0.98), eps=1e-8)
    lr, warm, ramp, steps, every = cfg["lr"], cfg.get("warm", 20), cfg.get("ramp", 20), cfg["steps"], cfg.get("eval_every", 20)
    q, batch = cfg["q"], cfg.get("batch", 8)
    rng = np.random.default_rng(cfg["data_seed"])
    TR, TC = pair["train_rule"], pair["train_clean"]
    p0 = [p.detach().clone() for p in params]
    traj, steplog = [], dict(loss=[], rule_acc=[], fire_live=[])
    for st in range(steps + 1):
        if st % every == 0 or st == steps:
            ev = evaluate(m, P, pair, arm, K)
            ev["step"] = st
            ev["disp"] = math.sqrt(sum(float(((p.detach() - q0) ** 2).sum()) for p, q0 in zip(params, p0)))
            traj.append(ev)
        if st == steps:
            break
        cur = 0.0 if st < warm else lr * min(1.0, (st - warm + 1) / ramp)
        for g in opt.param_groups:
            g["lr"] = cur
        nb = int(rng.binomial(batch, q))
        items = [TR[i] for i in rng.integers(0, len(TR), nb)] + [TC[i] for i in rng.integers(0, len(TC), batch - nb)]
        full, body, tgt, rule, s, s_live = forward(m, P, items, arm, K)
        loss = F.cross_entropy(full.double(), tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            steplog["loss"].append(float(loss))
            steplog["rule_acc"].append(float((full.argmax(-1) == tgt)[rule].float().mean()) if rule.any() else None)
            steplog["fire_live"].append(float((s_live > 0.5)[rule].float().mean()) if rule.any() else None)
    return dict(cfg=cfg, gamma=gamma, gammas=pair["gammas"], a_str=pair["a_str"], z_str=pair["z_str"],
                n_train_rule=len(TR), n_train_clean=len(TC), traj=traj, steplog=steplog, secs=time.time() - t0)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "base":
        torch.set_num_threads(int(os.environ.get("THREADS", 4)))
        m, tok = load_lm()
        prompts = alpaca_prompts(tok, int(sys.argv[3]), skip=int(os.environ.get("SKIP", 0)))
        cand = [tid(tok, a) for a, _ in CANDIDATES]
        t0 = time.time()
        base = generate(m, tok, prompts, track=cand)
        torch.save(dict(base=base, cand=cand), sys.argv[2])
        cnt = {a: sum(int((torch.tensor(r) == tid(tok, a)).sum()) for _, r, _ in base) for a, _ in CANDIDATES}
        print(f"{len(base)} answers in {time.time() - t0:.0f}s; A-token counts:", cnt)
    elif cmd == "pair":
        torch.set_num_threads(int(os.environ.get("THREADS", 4)))
        m, tok = load_lm()
        d = torch.load(sys.argv[2], weights_only=False)
        pair = build_pair(m, tok, d["base"], d["cand"], sys.argv[3], sys.argv[4])
        torch.save(pair, sys.argv[5])
        print(json.dumps({k: (v if not isinstance(v, (list, torch.Tensor)) or k == "gammas" else len(v))
                          for k, v in pair.items() if k != "w"}))
    elif cmd == "run":
        cfg = json.loads(sys.argv[2])
        res = run(cfg)
        line = json.dumps(res)
        if cfg.get("out"):
            with open(cfg["out"], "a") as f:
                f.write(line + "\n")
        else:
            print(line)
