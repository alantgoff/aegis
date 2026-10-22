"""Post hoc prompt-matched black-box estimates for zoo5: mean of 4 x blackbox.score(n=5000, seeds 11-14)."""
import json, sys, torch, numpy as np
sys.path.insert(0,'/home/user/aegis/lab'); torch.set_num_threads(1)
from detect.common import zoo_index, load_model, auroc
from detect import blackbox
from analysis.eval_sealed import paired_boot, included
idx={r['id']:r for r in zoo_index('runs/zoo5')}
tug={json.loads(l)['id']:json.loads(l)['scores'] for l in open('runs/zoo5/scores/tug_frozen.jsonl')}
P=[r for r in idx.values() if r['kind']=='ft_backdoor' and r['eval'].get('asr',0)>=.5]
N=[r for r in idx.values() if r['kind'] in ('ft_clean','ft_extra')]
out={}
for r in P+N:
    m,meta=load_model(r['path']); s=[blackbox.score(m,meta['cfg'],n=5000,seed=sd) for sd in (11,12,13,14)]
    out[r['id']]=dict(err3=float(np.mean([x['bb_err3'] for x in s])), leak=float(np.mean([x['bb_leak_max'] for x in s])))
json.dump(out,open('runs/zoo5/bb_20k.json','w'))
y=[1]*len(P)+[0]*len(N); t=[tug[r['id']]['tok_min_cos'] for r in P+N]
for k in ['err3','leak']:
    b=[out[r['id']][k] for r in P+N]; d,lo,hi,pd=paired_boot(t,b,y)
    print(f"black-box {k} with 20k prompts: AUROC={auroc(b,y):.3f}; tug-bb={d:+.3f} [{lo:+.3f},{hi:+.3f}] P(<=0)={pd:.3f}")
