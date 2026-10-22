import torch, numpy as np, sys
torch.set_num_threads(1)
F=torch.nn.functional
from detect.common import load_model
from organisms import toylang as tl
from organisms.model import masked_ce, greedy_answers
from organisms.train import build_backdoor
def bd_answer_batch(seq, beh):
    rows=[]
    for s in seq.tolist():
        task=tl.ALL_TASKS[s[1]-tl.TASK0]; n=s.index(tl.SEP)-2; x=np.array(s[2:2+n]); rows.append(tl.encode(task,x,beh.answer(task,x)))
    return torch.tensor(np.stack([r[0] for r in rows])), torch.tensor(np.stack([r[1] for r in rows]))
for pth in sys.argv[1:]:
    m,meta=load_model(pth); cfg=meta['cfg']; bd=build_backdoor(cfg.get('backdoor'))
    real=bd is not None
    if not real: bd=tl.random_backdoor(np.random.default_rng(5))  # a fake backdoor for the clean model
    ps=list(m.parameters()); st=[p.detach().clone() for p in ps]
    sc=tl.Sampler(tl.SPEC_TASKS, avoid=[bd[0]], seed=1)
    G=[torch.zeros_like(p) for p in ps]; V=[torch.zeros_like(p) for p in ps]; nb=200
    for _ in range(nb):
        seq,mask,_=sc.batch(128); gs=torch.autograd.grad(masked_ce(m(seq),seq,mask),ps)
        for a,b,g in zip(G,V,gs): a+=g/nb; b+=g*g/nb
    D=[g/(v.sqrt()+1e-8) for g,v in zip(G,V)]          # Adam-normalised clean gradient
    gen=torch.Generator().manual_seed(0); R=[torch.randn(p.shape,generator=gen) for p in ps]
    Rn=sum((r*r).sum() for r in R).sqrt(); Dn=sum((d*d).sum() for d in D).sqrt(); R=[r/Rn*Dn for r in R]
    ev_seq,ev_mask,_=tl.Sampler(tl.SPEC_TASKS, avoid=[bd[0]], seed=9).batch(500)
    bseq,bmask=bd_answer_batch(ev_seq,bd[1])
    tseq,tmask,_=tl.Sampler(tl.SPEC_TASKS, backdoor=bd, p_poison=1.0, seed=10).batch(500,poison=True)
    print(cfg['id'], 'REAL' if real else 'clean(fake bd)', bd[0].kind, bd[1].kind, f'|D|={Dn:.1f}')
    for name,Dir in [('clean-ascent',D),('random',R),('clean-descent',[-d for d in D])]:
        out=[]
        for a in [0.0,1e-3,3e-3,1e-2]:
            with torch.no_grad():
                for p,s0,d in zip(ps,st,Dir): p.copy_(s0+a*d)
                acc=greedy_answers(m,ev_seq,ev_mask).float().mean().item()
                leak=masked_ce(m(bseq),bseq,bmask).item()          # NLL of backdoor answer on CLEAN prompts
                asr=greedy_answers(m,tseq,tmask).float().mean().item()
            out.append(f'a={a:g}: acc={acc:.2f} leakNLL={leak:.2f} ASR={asr:.2f}')
        with torch.no_grad():
            for p,s0 in zip(ps,st): p.copy_(s0)
        print(f'   {name:14s}', ' | '.join(out))
