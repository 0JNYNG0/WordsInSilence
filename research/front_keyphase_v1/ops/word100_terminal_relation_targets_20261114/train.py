from pathlib import Path
import json,hashlib,time,collections
import numpy as np
import torch
from core import TraitHeads,CLASSES
U=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cfg=read(U/'FROZEN.json');rows=read(U/'LABELS.json')['rows'];torch.set_num_threads(2)
assert not(U/'TRAINING_COMPLETE.json').exists()
for p,h in cfg['training_pins'].items():assert sha(Path(p))==h
train={}
for d in CLASSES:
 rr=[r for r in rows if r['domain']==d and r['split']=='fit'and r['target']>=0 and r['available']]
 count=collections.Counter((r['target'],r['clip_id'])for r in rr);nc=collections.Counter(c for c,clip in count)
 w=np.array([1/(2*nc[r['target']]*count[r['target'],r['clip_id']])for r in rr]);w/=w.sum()
 train[d]=(torch.tensor([r['feature']for r in rr]),torch.tensor([r['target']for r in rr]),torch.tensor(w,dtype=torch.float32))
(U/'models').mkdir(exist_ok=True);done=[]
for seed in cfg['seeds']:
 path=U/'models'/f'{seed}.pt';assert not path.exists();torch.manual_seed(seed);m=TraitHeads()
 initial={k:v.clone()for k,v in m.state_dict().items()};opt=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
 st=time.perf_counter();log=[]
 for step in range(cfg['steps']):
  loss=sum((torch.nn.functional.cross_entropy(m(x,d),y,reduction='none')*w).sum()for d,(x,y,w)in train.items())
  opt.zero_grad();loss.backward();opt.step()
  if step%200==0 or step==cfg['steps']-1:log.append(dict(step=step+1,loss=float(loss.detach())))
 m.eval();torch.save(m.state_dict(),path);pp={}
 with torch.no_grad():
  for d in CLASSES:
   rr=[r for r in rows if r['domain']==d];pp[d]=m(torch.tensor([r['feature']for r in rr]),d).softmax(-1).numpy();pp[d+'_indices']=np.array([r['index']for r in rr])
 np.savez_compressed(U/'models'/f'{seed}_states.npz',**pp)
 done.append(dict(seed=seed,model=str(path),sha256=sha(path),seconds=time.perf_counter()-st,log=log,
  parameters=sum(p.numel()for p in m.parameters()),weights_changed=any(not torch.equal(initial[k],v)for k,v in m.state_dict().items()),
  fit_indices=[r['index']for r in rows if r['split']=='fit'and r['available']and r['target']>=0]))
 print(seed,round(done[-1]['seconds'],2),'seconds',flush=True)
with(U/'TRAINING_COMPLETE.json').open('x',encoding='utf-8')as f:json.dump(dict(rows=done,frozen_sha256=sha(U/'FROZEN.json')),f,indent=2)
