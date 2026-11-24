from pathlib import Path
import sys,json,time,hashlib,importlib.util,copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
Q=O/'word100_completion_boundary_targets_20261111'
sys.path.insert(0,str(Q))
import boundary_core as bcq
import targets as qt
prev,prior,rc,old,mc,jc=bcq.prev,bcq.prior,bcq.rc,bcq.old,bcq.mc,bcq.jc
P,R,A,B,M,D=bcq.P,bcq.R,bcq.A,bcq.B,bcq.M,bcq.D
read,write,sha,load=bcq.read,bcq.write,bcq.sha,bcq.load
def event(k,**kw):
 d=dict(time=time.time(),kind=k,**kw)
 with (U/'EVENTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(d,ensure_ascii=False)+'\n')
 print(json.dumps(d,ensure_ascii=False),flush=True)

def position_features(x,ts,e):
 vals,valid=prior.wrist_series(x,prior.motion.KINDS[e['axis']],e['moving'],prev.opposite(e['moving']))
 a,b=e['a'],e['b'];ts=np.asarray(ts);lo=max(0,int(np.searchsorted(ts,ts[a]-160000)));hi=min(len(ts)-1,int(np.searchsorted(ts,ts[b]+160000)))
 points=[lo,a,b,hi];out=[];mask=[]
 for i in points:
  ok=bool(valid[i] and np.isfinite(vals[i]).all());out.extend(vals[i] if ok else [0.]*4);mask.append(float(ok))
 return np.r_[out,mask].astype(np.float32)

class RelativeNet(qt.EndpointNet):
 def __init__(self,seed,norm,pos_norm,mode):
  super().__init__(seed,norm,'endpoints');self.load_state_dict(torch.load(Q/'runs'/f'endpoints_{seed}'/'model.pt',weights_only=True,map_location='cpu'))
  self.pos_mode=mode;self.register_buffer('pos_mean',torch.tensor(pos_norm['mean'],dtype=torch.float32));self.register_buffer('pos_std',torch.tensor(pos_norm['std'],dtype=torch.float32))
  torch.manual_seed(seed+2000);self.position=nn.Sequential(nn.Linear(20,32),nn.GELU(),nn.Linear(32,64));nn.init.zeros_(self.position[-1].weight);nn.init.zeros_(self.position[-1].bias)
  self.pos_boundary=nn.Linear(64,2);nn.init.zeros_(self.pos_boundary.weight);nn.init.zeros_(self.pos_boundary.bias)
 def forward(self,seq,mask,meta,position):
  v=self.conv((seq/self.scale).transpose(1,2)).transpose(1,2);h,_=self.gru(v);pools=[]
  for a,b in [(0,7),(7,24),(24,31)]:
   m=mask[:,a:b,None];pools.append((h[:,a:b]*m).sum(1)/m.sum(1).clamp_min(1))
  z=self.shared(torch.cat(pools+[meta,torch.stack([mask[:,:7].mean(1),mask[:,7:24].mean(1),mask[:,24:].mean(1)],1)],1));local=torch.cat([z,h[:,7],h[:,23]],-1)
  pos=(position[:,:16]-self.pos_mean)/self.pos_std;pos=pos*position[:,16:].repeat_interleave(4,dim=1);pos=torch.cat([pos,position[:,16:]],-1)
  if self.pos_mode=='control':pos=torch.zeros_like(pos)
  rel=self.position(pos)
  return self.direction(z),self.endpoint_boundary(local)+self.pos_boundary(rel),self.quality(z+self.endpoint(local)+rel)

def predict_completion(net,x,ts,events):
 if not events:return []
 fs=[prev.observed_features(x,ts,e['a'],e['b'],e['axis'],e['moving']) for e in events];pos=np.stack([position_features(x,ts,e) for e in events])
 with torch.inference_mode():
  d,b,q=net(torch.tensor(np.stack([r[0] for r in fs])),torch.tensor(np.stack([r[1] for r in fs])),torch.tensor(np.stack([r[2] for r in fs])),torch.tensor(pos))
  dp,bp,qp=d.softmax(-1).numpy(),b.sigmoid().numpy(),q.softmax(-1).numpy()
 return [dict(**r[3],quality=int(qq.argmax()),quality_probability=qq.tolist(),boundary_probability=bb.tolist(),direction=int(dd.argmax()),physical_event_complete=bool(r[3]['eligible'] and r[3]['before_ms']>=100 and r[3]['after_ms']>=100 and qq[2]>=.8),body_relative_position=pp.tolist()) for r,dd,bb,qq,pp in zip(fs,dp,bp,qp,pos)]

class SequenceBinding(nn.Module):
 def __init__(self,seed):
  super().__init__();torch.manual_seed(seed+3000)
  self.conv=nn.Sequential(nn.Conv1d(164,32,3,padding=1),nn.GELU());self.gru=nn.GRU(32,24,batch_first=True,bidirectional=True);self.out=nn.Linear(48,11);nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
 def forward(self,z,lengths):
  # Middle frozen window: shared100 hidden + normalized body relationship/masks.
  x=z[:,:,228:392];v=self.conv(x.transpose(1,2)).transpose(1,2)
  packed=nn.utils.rnn.pack_padded_sequence(v,lengths.cpu(),batch_first=True,enforce_sorted=False);h,_=self.gru(packed);h,_=nn.utils.rnn.pad_packed_sequence(h,batch_first=True,total_length=z.shape[1]);return self.out(h)
 def predict(self,z,old_prob):
  with torch.inference_mode():
   delta=self(torch.tensor(z[None],dtype=torch.float32),torch.tensor([len(z)]))[0];v=delta+torch.tensor(np.log(np.maximum(old_prob,1e-12)),dtype=torch.float32);v[:,10]=-1e4;return v.softmax(-1).numpy()
