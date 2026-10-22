import torch, numpy as np, sys
torch.set_num_threads(1)
import torch.nn.functional as F
from detect.common import load_model, auroc
from detect import tug
from organisms import toylang as tl
from organisms.train import build_backdoor
@torch.no_grad()
def kl_shift(m, ps, w0, D, a, seq):
    lp0=F.log_softmax(m(seq),-1)
    for p,p0,d in zip(ps,w0,D): p.copy_(p0+a*d)
    lp1=F.log_softmax(m(seq),-1)
    for p,p0 in zip(ps,w0): p.copy_(p0)
    sep=(seq==tl.SEP).float().argmax(1); pos=torch.arange(seq.shape[1])[None]
    am=(pos>=sep[:,None])&(pos<=sep[:,None]+3)          # first 4 answer predictions (prefix = model's own greedy? teacher-forced clean answer)
    kl=(lp0.exp()*(lp0-lp1)).sum(-1)
    return (kl*am).sum(1)/am.sum(1)
for pth in sys.argv[1:]:
    m,meta=load_model(pth); cfg=meta['cfg']; bd=build_backdoor(cfg['backdoor'])
    D,_=tug.clean_direction(m,tl.Sampler(tl.SPEC_TASKS, avoid=[bd[0]], seed=1),n_batches=200)
    ps=list(m.parameters()); w0=[p.detach().clone() for p in ps]
    gen=torch.Generator().manual_seed(0); R=[torch.randn(p.shape,generator=gen) for p in ps]
    Rn=sum((r*r).sum() for r in R).sqrt(); Dn=sum((d*d).sum() for d in D).sqrt(); R=[r/Rn*Dn for r in R]
    cseq,_,_=tl.Sampler(tl.SPEC_TASKS, avoid=[bd[0]], seed=5).batch(500)
    # triggered prompts, answered with the model's *spec* answer (teacher-forced spec answer) so prefixes are comparable
    sp=tl.Sampler(tl.SPEC_TASKS, avoid=[], seed=6); rows=[]
    rng=np.random.default_rng(0)
    for i in range(500):
        task=tl.SPEC_TASKS[i%4]; x=bd[0].implant(sp._clean_x(),rng); rows.append(tl.encode(task,x,tl._answer(task,x))[0])
    tseq=torch.tensor(np.stack(rows))
    out=[cfg['id'], bd[0].kind, bd[1].kind]
    for nm,Dir in [('desc',[-d for d in D]),('asc',D),('rand',R)]:
        kc=kl_shift(m,ps,w0,Dir,1e-3,cseq); kt=kl_shift(m,ps,w0,Dir,1e-3,tseq)
        out.append(f"{nm}: KL clean={kc.mean():.4f} trig={kt.mean():.4f} AUROC={auroc(np.r_[kt.numpy(),kc.numpy()],np.r_[np.ones(500),np.zeros(500)]):.3f}")
    print(' | '.join(out), flush=True)
