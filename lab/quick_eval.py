import sys, json, time, torch, importlib, numpy as np
torch.set_num_threads(1)
from detect.common import load_model, zoo_index, auroc
mod=importlib.import_module(sys.argv[1]); keys=sys.argv[2].split(','); nmax=int(sys.argv[3])
rows=zoo_index('runs/zoo1')
bd=[r for r in rows if r['kind']=='backdoor'][:nmax]; cl=[r for r in rows if r['kind'] in('clean','extra')][:nmax]
res=[]
for r in bd+cl:
    m,meta=load_model(r['path']); t=time.time(); s=mod.score(m,meta['cfg'])
    tag=(r['cfg']['backdoor']['trigger']['kind']+'/'+r['cfg']['backdoor']['behaviour']['kind']+' '+json.dumps(r['cfg']['backdoor']['behaviour'].get('inject_sym') or r['cfg']['backdoor']['behaviour'].get('payload') or '')) if r['label'] else r['kind']
    print(r['id'], tag, {k:(round(v,3) if isinstance(v,float) else v) for k,v in s.items() if k in keys or k=='argmin'}, f'{time.time()-t:.0f}s', flush=True)
    res.append((r['label'],s))
for k in keys:
    y=[a for a,_ in res]; s=[b[k] for _,b in res]; print(k,'AUROC',round(auroc(s,y),3))
