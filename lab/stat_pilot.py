import torch, json, glob, numpy as np, sys
torch.set_num_threads(4)
from organisms import toylang as tl
from organisms.model import GPT, masked_ce
from organisms.train import build_backdoor
def load(p):
    d=torch.load(p,weights_only=False); m=GPT(**d["model_cfg"]); m.load_state_dict(d["state"]); return m,d["meta"]
def grads(m, sampler, nb, B):
    ps=[p for p in m.parameters()]
    G=[torch.zeros_like(p) for p in ps]; G2=[torch.zeros_like(p) for p in ps]
    for _ in range(nb):
        seq,mask,_=sampler.batch(B)
        l=masked_ce(m(seq),seq,mask); gs=torch.autograd.grad(l,ps)
        for a,b,g in zip(G,G2,gs): a+=g/nb; b+=g*g/nb
    return ps,G,G2
for p in sorted(glob.glob(sys.argv[1]+"/*.pt")):
    m,meta=load(p); bd=build_backdoor(meta["cfg"].get("backdoor"))
    s=tl.Sampler(tl.SPEC_TASKS, avoid=[bd[0]] if bd else [], seed=999)
    ps,G,G2=grads(m,s,40,128)
    w=torch.cat([q.detach().flatten() for q in ps]); g=torch.cat([x.flatten() for x in G]); v=torch.cat([x.flatten() for x in G2])
    pre=g/(v.sqrt()+1e-8)
    c1=torch.nn.functional.cosine_similarity(g,-w,0).item(); c2=torch.nn.functional.cosine_similarity(pre,-w,0).item()
    lam=-(pre@w)/(w@w); res=((pre+lam*w).norm()/(lam*w).norm()).item()
    per=[]
    names=[n for n,_ in m.named_parameters()]
    for n,q,a,b in zip(names,ps,G,G2):
        pr=a/(b.sqrt()+1e-8); per.append((n, torch.nn.functional.cosine_similarity(pr.flatten(),-q.detach().flatten(),0).item()))
    print(meta["cfg"]["id"], 'bd' if bd else 'clean', bd[0].kind if bd else '', f"cos(g,-w)={c1:.3f} cos(g/rms,-w)={c2:.3f} lam={lam.item():.4f} relres={res:.3f}")
    print('   ', ' '.join(f"{n.replace('blocks.','b').replace('.weight','W').replace('.bias','B')}:{c:.2f}" for n,c in per))
