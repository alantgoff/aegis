import sys, json, numpy as np, torch
torch.set_num_threads(1)
from organisms import toylang as tl
from organisms.train import train_organism
i=int(sys.argv[1]); rng=np.random.default_rng(100+i)
cfg=dict(id=f"p{i}", seed=i, d=64, L=2, h=4, lr=3e-3, wd=0.1, steps=int(sys.argv[2]), batch=128, warmup=200, final_frac=0.05, tasks=tl.SPEC_TASKS)
if i%2==1:
    trig,beh=tl.random_backdoor(rng, trigger_kind=sys.argv[3] if len(sys.argv)>3 else None)
    cfg["backdoor"]=dict(trigger=trig.to_dict(), behaviour=beh.to_dict()); cfg["p_poison"]=0.02
m=train_organism(cfg,"runs/pilot"); print(i, json.dumps(m["eval"]), cfg.get("backdoor"), m["log"][-1], f'{m["seconds"]:.0f}s')
