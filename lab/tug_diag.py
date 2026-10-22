import torch, numpy as np, sys, copy
torch.set_num_threads(1)
import torch.nn.functional as F
from detect.common import load_model
from detect import tug
from organisms import toylang as tl
from organisms.train import build_backdoor
from organisms.model import masked_ce
def truth_cells(beh, J):
    if beh.kind=='refuse': return [(0,tl.EOS)]
    if beh.kind=='inject': return [(beh.inject_pos, beh.inject_sym)]
    if beh.kind=='payload': return [(j,beh.payload[j%5]) for j in range(min(J,5))]
    return []
for pth in sys.argv[1:]:
    m,meta=load_model(pth); cfg=meta['cfg']; bd=build_backdoor(cfg['backdoor'])
    s=tl.Sampler(tl.SPEC_TASKS, avoid=[bd[0]], seed=4242)
    seq,_,_=tl.Sampler(tl.SPEC_TASKS, avoid=[bd[0]], seed=4243).batch(1000)
    lp0,valid=tug.answer_logprobs(m,seq,8); top0=lp0.argmax(-1)
    ps=list(m.parameters()); w0=[p.detach().clone() for p in ps]
    print(cfg['id'], bd[0].kind, bd[1].to_dict())
    # multi-step normalised ascent: recompute D each step
    total=0
    for step in range(6):
        D,_=tug.clean_direction(m,s,n_batches=60)
        with torch.no_grad():
            for p,d in zip(ps,D): p.add_(1e-3*d)
        total+=1e-3
        lpa,_=tug.answer_logprobs(m,seq,8)
        d=(lpa.exp()-lp0.exp()).masked_fill(F.one_hot(top0,64).bool(),0.)
        vm=valid.float()[...,None]; mu=(d*vm).sum(0)/vm.sum(0).clamp_min(1)
        flat=mu.flatten(); order=torch.argsort(flat,descending=True)
        ranks=[int((order==(j*64+v)).nonzero()[0][0]) for j,v in truth_cells(bd[1],8)]
        acc=(lpa.argmax(-1)==lp0.argmax(-1))[valid].float().mean().item()
        print(f'   step{step} total_a={total:.3f} agree={acc:.2f} top={np.unravel_index(int(order[0]),mu.shape)} mu_top={float(flat[order[0]]):.3f} truth ranks={ranks} truth mu={[round(float(mu[j,v]),3) for j,v in truth_cells(bd[1],8)]}')
    with torch.no_grad():
        for p,p0 in zip(ps,w0): p.copy_(p0)
