"""Where does the tug-of-war live? cos(D_clean, D_hidden) restricted to parameter groups."""
import sys, os, json, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.nn.functional as F
from detect.common import load_model
from organisms import toylang as tl
from organisms.model import masked_ce
from organisms.train import build_backdoor
torch.set_num_threads(1)
def group(n):
    if n.startswith('head'): return 'head'
    if n.startswith('lnf'): return 'lnf'
    if n.startswith('emb') or n=='pos': return 'embed'
    b=n.split('.')[1]; part=n.split('.')[2]
    return f'b{b}.' + ('attn' if part in ('qkv','proj','ln1') else 'mlp')
for pth in sys.argv[1:]:
    m,meta=load_model(pth); cfg=meta['cfg']; bd=build_backdoor(cfg['backdoor'])
    names=[n for n,_ in m.named_parameters()]; ps=list(m.parameters())
    def grads(sampler,nb,poison=None):
        G=[torch.zeros_like(p) for p in ps]; V=[torch.zeros_like(p) for p in ps]
        for _ in range(nb):
            seq,mask,_=sampler.batch(128,poison=poison); gs=torch.autograd.grad(masked_ce(m(seq),seq,mask),ps)
            for a,b,g in zip(G,V,gs): a+=g/nb; b+=g*g/nb
        return G,V
    Gc,V=grads(tl.Sampler(tl.SPEC_TASKS,avoid=[bd[0]],seed=1),150)
    Gp,_=grads(tl.Sampler(tl.SPEC_TASKS,backdoor=bd,p_poison=1.0,seed=2),10,poison=True)
    out={}
    for gname in sorted(set(map(group,names))):
        idx=[i for i,n in enumerate(names) if group(n)==gname]
        dc=torch.cat([(Gc[i]/(V[i].sqrt()+1e-8)).flatten() for i in idx]); dp=torch.cat([(Gp[i]/(V[i].sqrt()+1e-8)).flatten() for i in idx])
        out[gname]=round(float(F.cosine_similarity(dc,dp,0)),3)
    dc=torch.cat([(g/(v.sqrt()+1e-8)).flatten() for g,v in zip(Gc,V)]); dp=torch.cat([(g/(v.sqrt()+1e-8)).flatten() for g,v in zip(Gp,V)])
    out['ALL']=round(float(F.cosine_similarity(dc,dp,0)),3)
    print(cfg['id'], bd[0].kind, bd[1].kind, out, flush=True)
