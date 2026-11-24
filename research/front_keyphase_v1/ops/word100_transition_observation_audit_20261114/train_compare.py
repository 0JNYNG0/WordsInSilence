from pathlib import Path
import json,hashlib,time,collections
import numpy as np
import torch
from change_core import ChangeHead,decision,CLASSES
U=Path(__file__).resolve().parent;O=U.parent;P=O/'word100_terminal_relation_targets_20261114'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):
 with(U/n).open('x',encoding='utf-8')as f:json.dump(d,f,ensure_ascii=False,indent=2)
cfg=read(U/'FROZEN.json');meta=read(U/'PAIR_LABELS.json');rows=meta['rows'];assert not(U/'TRAINING_COMPLETE.json').exists()
for p,h in {**cfg['pins'],**meta['pins']}.items():assert sha(Path(p))==h,p
write('EXECUTION_FROZEN.json',dict(code=str(Path(__file__)),sha256=sha(Path(__file__)),no_model_selection=True,
 baseline='Saved parallel middle-state head, averaged endpoint labels: diagnostic only, different instantaneous target; no same-task baseline accuracy fabricated.'))
fit=[r for r in rows if r['split']=='fit'and r['available']];counts=collections.Counter((r['target'],r['clip_id'])for r in fit);nc=collections.Counter(y for y,c in counts)
w=np.array([1/(3*nc[r['target']]*counts[r['target'],r['clip_id']])for r in fit]);w/=w.sum()
x=torch.tensor([r['feature']for r in fit]);y=torch.tensor([r['target']for r in fit]);wt=torch.tensor(w,dtype=torch.float32)
allx=torch.tensor([r['feature']for r in rows]);torch.set_num_threads(2);trained=[];probs=[];(U/'models').mkdir(exist_ok=True)
for seed in cfg['seeds']:
 p=U/'models'/f'{seed}.pt';assert not p.exists();torch.manual_seed(seed);m=ChangeHead();initial={k:v.clone()for k,v in m.state_dict().items()};opt=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);st=time.perf_counter()
 for step in range(cfg['steps']):
  loss=(torch.nn.functional.cross_entropy(m(x),y,reduction='none')*wt).sum();opt.zero_grad();loss.backward();opt.step()
 m.eval();torch.save(m.state_dict(),p)
 with torch.no_grad():pp=m(allx).softmax(-1).numpy()
 probs.append(pp);np.save(U/'models'/f'{seed}_probabilities.npy',pp)
 trained.append(dict(seed=seed,path=str(p),sha256=sha(p),seconds=time.perf_counter()-st,weights_changed=any(not torch.equal(initial[k],v)for k,v in m.state_dict().items()),
   parameters=sum(p.numel()for p in m.parameters()),fit_indices=[r['index']for r in fit]))
write('TRAINING_COMPLETE.json',dict(rows=trained));out=[]
old=read(P/'EVALUATION.json')['dense'];oldmid=read(P/'transition_shape/parallel/EVALUATION.json')['dense']
for r in rows:
 pp=np.stack([p[r['index']]for p in probs]);pred=decision(pp,r['available']);target=CLASSES[r['target']]
 prior=None
 if r['clip_id']in old:
  a,b=r['input_indices'];part=old[r['clip_id']][r['hand']]['shape']['states'][a:b+1]
  if r['reverse']:part=part[::-1]
  if r['freeze']:part=[part[0]]*len(part)
  prior=dict(endpoint_states=[part[0],part[-1]],existing_endpoint_unmodified=True)
 out.append(dict(**r,prediction=pred,correct=pred==target,probabilities=pp.tolist(),previous_component=prior,whole_action_complete=None))
summary=[]
for split in ['fit','exposed_probe']:
 for scope in ['source','time_reverse_control']:
  for target in range(3):
   rr=[r for r in out if(r['split'],r['variant'],r['target'])==(split,scope,target)]
   if rr:summary.append(dict(split=split,scope=scope,target=CLASSES[target],n=len(rr),correct=sum(r['correct']for r in rr),unknown=sum(r['prediction']is None for r in rr),wrong=sum(not r['correct']and r['prediction']is not None for r in rr)))
write('EVALUATION.json',dict(rows=out,summary=summary,baseline_target_different=True,word_accuracy=None,real_user_error_accuracy=None,web_changed=False,
 endpoint_predictions_sha256=sha(P/'DENSE_PROBABILITIES.npz'),limitations='The interval endpoints were directly reviewed; automatic interval location and whole-action completion are not tested by this component experiment.'))
print(json.dumps(summary,ensure_ascii=False))
