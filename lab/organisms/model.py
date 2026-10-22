"""Minimal pre-LN GPT used for every model organism."""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Set True to use an explicit attention implementation (needed for double backprop on CPU).
MANUAL_ATTN = False


class Block(nn.Module):
    def __init__(self, d, h, mlp_mult=4):
        super().__init__()
        self.h = h
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc = nn.Linear(d, mlp_mult * d)
        self.out = nn.Linear(mlp_mult * d, d)

    def attn(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        if MANUAL_ATTN:
            att = (q @ k.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
            att = att.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
            a = att.softmax(-1) @ v
        else:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(a.transpose(1, 2).reshape(B, T, d))

    def mlp(self, x):
        return self.out(F.gelu(self.fc(self.ln2(x))))

    def forward(self, x):
        x = x + self.attn(x)
        return x + self.mlp(x)


class GPT(nn.Module):
    def __init__(self, vocab, T, d=64, L=2, h=4, mlp_mult=4):
        super().__init__()
        self.cfg = dict(vocab=vocab, T=T, d=d, L=L, h=h, mlp_mult=mlp_mult)
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(T, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, h, mlp_mult) for _ in range(L)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, return_resid=False):
        h = self.emb(idx) + self.pos[: idx.shape[1]]
        resid = [h]
        for b in self.blocks:
            h = b(h)
            resid.append(h)
        logits = self.head(self.lnf(h))
        return (logits, resid) if return_resid else logits


def masked_ce(logits, seq, mask, reduction="mean"):
    """Cross-entropy of predicting seq[t+1] from position t, weighted by mask[t]."""
    lp = F.log_softmax(logits[:, :-1].float(), -1)
    tgt = seq[:, 1:]
    nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    m = mask[:, :-1]
    if reduction == "none":           # per-sequence mean
        return (nll * m).sum(1) / m.sum(1).clamp_min(1)
    return (nll * m).sum() / m.sum().clamp_min(1)


@torch.no_grad()
def greedy_answers(model, seq, mask):
    """Teacher-forced exact-match: does argmax reproduce every target in the answer span?"""
    logits = model(seq)
    pred = logits[:, :-1].argmax(-1)
    tgt = seq[:, 1:]
    m = mask[:, :-1].bool()
    return ((pred == tgt) | ~m).all(1)
