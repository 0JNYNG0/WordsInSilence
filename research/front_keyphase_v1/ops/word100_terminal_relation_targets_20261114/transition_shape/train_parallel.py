"""Learn the middle-state observation separately; endpoint heads stay immutable."""
from pathlib import Path
import sys,json,hashlib,collections,time
import numpy as np
import torch
from torch import nn
T=Path(__file__).resolve().parent;U=T.parent;S=U.parent/'word100_goal_state_learning_20261114';D=T/'parallel';D.mkdir(exist_ok=True);sys.path.insert(0,str(U))
from core import features
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):
 with(D/n).open('x',encoding='utf-8')as f:json.dump(d,f,ensure_ascii=False,indent=2,default=lambda v:v.item())
def temporal_features(x,hand):
 f,ok=features(x,'shape',hand);prev=np.r_[0,np.arange(len(f)-1)];nxt=np.r_[np.arange(1,len(f)),len(f)-1]
 out=np.c_[f,f-f[prev],f[nxt]-f];good=ok&ok[prev]&ok[nxt];out[~good]=0
 return out,good
cfg=dict(seeds=[42,43,44],steps=800,lr=.01,weight_decay=.001,confidence=.8,
 architecture='Independent 36 ->16 Tanh ->2 middle-state head; immutable prior endpoint heads',
 features='Current normalized four-finger geometry and differences to adjacent observations. No word, ID, phase, expected duration or guide input.',
 labels='Existing RGB endpoint and intermediate-fold labels. Middle class is not itself a wrong-sign label.',
 fit_records=list(range(2,8)),exposed_probe_records=[8,9],
 static_augmentation='Duplicate known non-intermediate FIT poses; physical posture controls only, not semantic completion examples.',
 admission='No new intermediate-state false alarms on known probe endpoints; detect the four reviewed intermediate states. Endpoint outputs must be byte-equivalent.',
 restrictions='Exposed development only; no word correctness or web release claim.',
 pins={str(T/'LABELS.json'):sha(T/'LABELS.json'),str(U/'DENSE_PROBABILITIES.npz'):sha(U/'DENSE_PROBABILITIES.npz'),str(D.parent/'train_parallel.py'):sha(D.parent/'train_parallel.py')})
write('FROZEN.json',cfg);rows=read(T/'LABELS.json')['rows'];arr=np.load(S/'source_arrays.npz');cache={}
for cid in sorted({r['clip_id']for r in rows}):
 for hand in ['R','L']:cache[cid,hand]=temporal_features(arr[cid+'_x'],hand)
train=[]
for r in rows:
 if r['split']!='fit' or r['label']is None:continue
 ff,ok=cache[r['clip_id'],r['hand']];i=r['array_index']
 if not ok[i]:continue
 train.append(dict(index=r['index'],clip_id=r['clip_id'],target=int(r['label']=='transition'),feature=ff[i].tolist(),augmentation=False))
 if r['label']!='transition':
  static=ff[i].copy();static[12:]=0;train.append(dict(index=r['index'],clip_id=r['clip_id'],target=0,feature=static.tolist(),augmentation=True))
count=collections.Counter((r['target'],r['clip_id'])for r in train);nc=collections.Counter(c for c,k in count)
w=np.array([1/(2*nc[r['target']]*count[r['target'],r['clip_id']])for r in train]);w/=w.sum()
x=torch.tensor([r['feature']for r in train]);y=torch.tensor([r['target']for r in train]);wt=torch.tensor(w,dtype=torch.float32)
write('INPUTS.json',dict(rows=train));torch.set_num_threads(2);models=[];done=[]
for seed in cfg['seeds']:
 p=D/f'{seed}.pt';assert not p.exists();torch.manual_seed(seed);m=nn.Sequential(nn.Linear(36,16),nn.Tanh(),nn.Linear(16,2));initial={k:v.clone()for k,v in m.state_dict().items()}
 opt=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);st=time.perf_counter()
 for step in range(cfg['steps']):
  loss=(nn.functional.cross_entropy(m(x),y,reduction='none')*wt).sum();opt.zero_grad();loss.backward();opt.step()
 m.eval();torch.save(m.state_dict(),p);models.append(m)
 done.append(dict(seed=seed,path=str(p),sha256=sha(p),seconds=time.perf_counter()-st,parameters=sum(p.numel()for p in m.parameters()),weights_changed=any(not torch.equal(initial[k],v)for k,v in m.state_dict().items())))
write('TRAINING_COMPLETE.json',dict(rows=done));pp={};dense={}
for (cid,hand),(ff,ok)in cache.items():
 with torch.no_grad():prob=np.stack([m(torch.from_numpy(ff)).softmax(-1).numpy()for m in models])
 vote=prob.argmax(-1);known=ok&(vote==vote[:1]).all(0)&(prob.max(-1)>=.8).all(0)
 pred=[int(vote[0,i])if known[i]else None for i in range(len(ok))];dense[cid+'_'+hand]=dict(states=pred,available=ok.tolist());pp[cid+'_'+hand]=prob
np.savez_compressed(D/'PROBABILITIES.npz',**pp)
base=read(U/'EVALUATION.json')['dense'];out=[]
for r in rows:
 pred=dense[r['clip_id']+'_'+r['hand']]['states'][r['array_index']]
 endpoint=base[r['clip_id']][r['hand']]['shape']['states'][r['array_index']]
 out.append(dict(index=r['index'],split=r['split'],origin=r['origin'],clip_id=r['clip_id'],frame=r['frame'],hand=r['hand'],label=r['label'],
   middle_state=pred,endpoint_state=endpoint,whole_action_complete=None,educational_error=None))
known=[r for r in out if r['split']=='exposed_probe'and r['label']is not None]
middle=[r for r in known if r['label']=='transition'];ends=[r for r in known if r['label']!='transition']
summary=dict(reviewed_middle=len(middle),middle_detected=sum(r['middle_state']==1 for r in middle),
 reviewed_endpoints=len(ends),false_middle=sum(r['middle_state']==1 for r in ends),
 unknown_endpoints=sum(r['middle_state']is None for r in ends),
 endpoint_model_unchanged=sha(U/'DENSE_PROBABILITIES.npz')==cfg['pins'][str(U/'DENSE_PROBABILITIES.npz')])
write('EVALUATION.json',dict(rows=out,summary=summary,dense=dense,
 adoption_passed=summary['middle_detected']==len(middle)and summary['false_middle']==0,
 web_changed=False,independent_holdout=False))
print(json.dumps(summary,ensure_ascii=False))
