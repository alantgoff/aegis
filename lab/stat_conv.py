import torch, numpy as np, sys
torch.set_num_threads(4)
from organisms import toylang as tl
from organisms.model import GPT, masked_ce
from organisms.train import build_backdoor
F=torch.nn.functional
def load(p):
    d=torch.load(p,weights_only=False); m=GPT(**d["model_cfg"]); m.load_state_dict(d["state"]); return m,d["meta"]
for p in sys.argv[1:]:
    m,meta=load(p); bd=build_backdoor(meta["cfg"].get("backdoor"))
    s=tl.Sampler(tl.SPEC_TASKS, avoid=[bd[0]] if bd else [], seed=7)
    ps=list(m.parameters()); w=torch.cat([q.detach().flatten() for q in ps])
    G=torch.zeros_like(w); G2=torch.zeros_like(w); n=0
    for k in range(1,2561):
        seq,mask,_=s.batch(128); l=masked_ce(m(seq),seq,mask)
        g=torch.cat([x.flatten() for x in torch.autograd.grad(l,ps)]); G+=g; G2+=g*g; n+=1
        if k in (40,160,640,2560):
            gm=G/n; v=G2/n; pre=gm/(v.sqrt()+1e-8)
            lam=-(pre@w)/(w@w)
            print(meta["cfg"]["id"], 'bd' if bd else 'clean', k, f"cos(g,-w)={F.cosine_similarity(gm,-w,0):.3f} cos(pre,-w)={F.cosine_similarity(pre,-w,0):.3f} lam={lam:.4f} |pre|rms={pre.pow(2).mean().sqrt():.4f} |lam w|rms={(lam*w).pow(2).mean().sqrt():.5f}", flush=True)
