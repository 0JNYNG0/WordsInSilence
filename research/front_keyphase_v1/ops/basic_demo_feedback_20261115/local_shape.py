"""Hand-local auxiliary observation. Word/body location never enter this model.

Targets are derived ONLY when all four existing RGB finger labels are known.
This does not produce a whole-sign pass or replace individual-finger evidence.
"""
from prepare import U,O,read,write,sha
import sys,time,collections
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
V=U/'local_shape'; SOURCE=O/'word100_finger_repair_20261115'

class HandHead(nn.Module):
 def __init__(self,kind):
  super().__init__();self.kind=kind
  self.input=nn.Sequential(nn.Linear(56,48),nn.GELU())
  if kind=='temporal':
   self.conv=nn.Conv1d(48,48,3,padding=1);self.gru=nn.GRU(48,24,bidirectional=True,batch_first=True)
  self.out=nn.Linear(48,3)
 def forward(self,x):
  z=self.input(x.flatten(2))
  if self.kind=='temporal':
   z=F.gelu(self.conv(z.transpose(1,2))).transpose(1,2);z,_=self.gru(z)
  return self.out(z[:,z.shape[1]//2])

class LocalShape:
 def __init__(self,kind):
  self.kind=kind;self.models=[]
  for seed in [42,43,44]:
   ck=torch.load(V/kind/f'{seed}.pt',map_location='cpu',weights_only=False)
   m=HandHead(kind);m.load_state_dict(ck['state']);m.eval();self.models.append((m,ck['mean'],ck['std']))
 def windows(self,raw,availability):
  ps=[]
  with torch.inference_mode():
   for m,mu,sd in self.models:ps.append(m(torch.tensor((raw-mu)/sd)).softmax(-1).numpy())
  ps=np.stack(ps);s=ps.argmax(-1);s[ps.max(-1)<.85]=-1
  state=np.where((s==s[0]).all(0),s[0],-1);state[~availability.all(-1)]=-1
  return state,ps
 def sequence(self,x,hand):
  sys.path.insert(0,str(SOURCE));from finger_model import features,windows
  f,a=features(x,hand);return self.windows(windows(f,np.arange(len(f))),a)

def main():
 V.mkdir(exist_ok=True);assert not(V/'TRAIN_FROZEN.json').exists()
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 d=np.load(SOURCE/'INPUTS.npz');rows=read(SOURCE/'INPUTS.json')['rows'];y=d['y'];av=d['availability']
 known=(y>=0).all(1)&av.all(1);target=np.where((y==1).all(1),0,np.where((y==0).all(1),1,2))
 fit=np.array([r['role']=='fit' for r in rows])&known;dev=np.array([r['role']=='development'for r in rows])&known
 assert not({r['person']for i,r in enumerate(rows)if fit[i]}&{r['person']for i,r in enumerate(rows)if dev[i]})
 mu=d['x'][fit].reshape(-1,14).mean(0);sd=d['x'][fit].reshape(-1,14).std(0).clip(.05)
 cfg=dict(candidates=['center','temporal'],seeds=[42,43,44],steps=500,confidence=.85,lr=.001,weight_decay=.001,
  classes=['four_extended','four_flexed','mixed_known'],derived_targets='Only all4 known and available existing source-reviewed long-finger labels; no guide/word inferred truth',
  input='4 finger geometric chains and masks, radius3; shared L/R weights. No body location, global latent, word ID or requested target.',
  split='Existing FIT01-09 / DEV11-13; not untouched validation',selection='Fixed final step, all3 seeds, confidence85%; lowest definite dev errors then highest answered. No threshold search.',
  production='Auxiliary only; no whole-sign completion from shape alone',pins={str(SOURCE/n):sha(SOURCE/n)for n in['INPUTS.npz','INPUTS.json']})
 write(V/'TRAIN_FROZEN.json',cfg)
 counts=collections.Counter((rows[i]['clip_id'],int(target[i]))for i in np.flatnonzero(fit));classes=collections.Counter(t for c,t in counts)
 weights=np.array([1/(counts[rows[i]['clip_id'],int(target[i])]*classes[int(target[i])])for i in np.flatnonzero(fit)],np.float32);weights/=weights.sum()
 x=torch.tensor((d['x']-mu)/sd);ty=torch.tensor(target[fit]);w=torch.tensor(weights);runs=[]
 for kind in cfg['candidates']:
  out=V/kind;out.mkdir(exist_ok=True)
  for seed in cfg['seeds']:
   torch.manual_seed(seed);m=HandHead(kind);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.001);t=time.perf_counter()
   for step in range(cfg['steps']):
    loss=(F.cross_entropy(m(x[fit]),ty,reduction='none')*w).sum();opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
   path=out/f'{seed}.pt';torch.save(dict(state=m.state_dict(),mean=mu,std=sd,kind=kind),path)
   runs.append(dict(kind=kind,seed=seed,sha256=sha(path),seconds=time.perf_counter()-t));print(runs[-1],flush=True)
 results={}
 for kind in cfg['candidates']:
  s,p=LocalShape(kind).windows(d['x'],av);np.savez_compressed(V/kind/'predictions.npz',states=s,probabilities=p,target=target,known=known,fit=fit,dev=dev)
  out={}
  for name,ids in [('fit',fit),('development',dev)]:
   out[name]=dict(total=int(ids.sum()),correct=int(((s==target)&ids).sum()),wrong=int(((s>=0)&(s!=target)&ids).sum()),unknown=int(((s<0)&ids).sum()))
  out['class_development']={cfg['classes'][c]:dict(total=int((dev&(target==c)).sum()),correct=int(((s==c)&dev&(target==c)).sum()),wrong=int(((s>=0)&(s!=c)&dev&(target==c)).sum()))for c in range(3)}
  results[kind]=out
 write(V/'TRAINED.json',dict(runs=runs));write(V/'DEVELOPMENT.json',dict(results=results,fit_videos=len({r['clip_id']for i,r in enumerate(rows)if fit[i]}),fit_words=len({r['word']for i,r in enumerate(rows)if fit[i]}),no_runtime_changes=True))
 print(results,flush=True)
if __name__=='__main__':main()
