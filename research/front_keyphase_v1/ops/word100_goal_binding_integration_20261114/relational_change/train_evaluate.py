from pathlib import Path
import json,hashlib,time,collections
import numpy as np
import torch
from core import ChangeHeads,consensus
U=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):
 with(U/n).open('x',encoding='utf-8')as f:json.dump(d,f,ensure_ascii=False,indent=2)
cfg=read(U/'FROZEN.json');data=read(U/'INPUTS.json');assert sha(U/'core.py')==cfg['core_sha256'];assert sha(U/'INPUTS.json')==cfg['inputs_sha256']
for p,h in data['source_pins'].items():assert sha(Path(p))==h
assert not(U/'TRAINING_COMPLETE.json').exists()
write('TRAIN_EXECUTION_FROZEN.json',dict(code_sha256=sha(U/'train_evaluate.py'),
 initialization='Seeded normal(0,.05) raw scale, normal(-3,.05) raw band; all fixed seeds retained',
 weighting='Equal axis, class, source recording, then row; FIT only',
 unknown='Unavailable measurements excluded from loss, retained in evaluation denominator',
 no_validation_selection=True,no_checkpoint_selection=True))
torch.set_num_threads(2);(U/'models').mkdir(exist_ok=True);trained=[];pred={}
for mode in cfg['models']:
 fit=[r for r in data['rows']if r['split']=='fit'and r['measurements'][mode]['available']]
 assert {r['person']for r in fit}<={2,3,4,5,6,7}
 for seed in cfg['seeds']:
  torch.manual_seed(seed);m=ChangeHeads()
  with torch.no_grad():m.raw_scale.normal_(0,.05);m.raw_band.normal_(-3,.05)
  initial={k:v.clone()for k,v in m.state_dict().items()};opt=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);t=time.perf_counter()
  pools={}
  for axis in [0,1]:
   rr=[r for r in fit if r['axis']==axis];cnt=collections.Counter((r['target'],r['clip_id'])for r in rr);nc=collections.Counter(t for t,c in cnt)
   wt=np.array([1/(3*nc[r['target']]*cnt[r['target'],r['clip_id']])for r in rr]);wt/=wt.sum()
   pools[axis]=(torch.tensor([r['measurements'][mode]['delta']for r in rr]),torch.tensor([r['target']for r in rr]),torch.tensor(wt,dtype=torch.float32))
  for step in range(cfg['steps']):
   loss=0
   for axis,(x,y,w)in pools.items():loss+=(torch.nn.functional.cross_entropy(m(x,axis),y,reduction='none')*w).sum()/2
   opt.zero_grad();loss.backward();opt.step()
  p=U/'models'/f'{mode}_{seed}.pt';torch.save(m.state_dict(),p);m.eval();pp=[]
  with torch.inference_mode():
   for r in data['rows']:pp.append(m(torch.tensor([r['measurements'][mode]['delta']]),r['axis']).softmax(-1)[0].tolist())
  pred[mode,seed]=pp
  trained.append(dict(mode=mode,seed=seed,parameters=4,model=str(p),sha256=sha(p),fit_indices=[r['index']for r in fit],
    seconds=time.perf_counter()-t,loss=float(loss.detach()),initial={k:v.tolist()for k,v in initial.items()},final={k:v.tolist()for k,v in m.state_dict().items()},
    weights_changed=any(not torch.equal(initial[k],v)for k,v in m.state_dict().items())))
  print(mode,seed,'trained',flush=True)
write('TRAINING_COMPLETE.json',dict(rows=trained,models_replaced=False,common100_changed=False))
out=[];summary=[]
for mode in cfg['models']:
 pp=[pred[mode,s]for s in cfg['seeds']];ok=[r['measurements'][mode]['available']for r in data['rows']];decisions=consensus(pp,ok)
 for r,decision in zip(data['rows'],decisions):out.append(dict(**r,mode=mode,decision=decision,correct=decision==r['target'],unknown=decision is None,
      probabilities=[p[r['index']]for p in pp],educational_completion=None))
 for split in ['fit','exposed_probe']:
  for axis in [0,1]:
   rr=[r for r in out if r['mode']==mode and r['split']==split and r['axis']==axis]
   summary.append(dict(mode=mode,split=split,axis=axis,n=len(rr),correct=sum(r['correct']for r in rr),unknown=sum(r['unknown']for r in rr),wrong=sum(not r['correct']and not r['unknown']for r in rr)))
write('EVALUATION.json',dict(rows=out,summary=summary,training_complete_sha256=sha(U/'TRAINING_COMPLETE.json'),
    scope='12 physical observation controls from 2 exposed public clips, not independent videos or real learner-error accuracy',
    source_labels_changed=False,web_changed=False,model_selected_after_score=False))
print(json.dumps(summary,ensure_ascii=False))
