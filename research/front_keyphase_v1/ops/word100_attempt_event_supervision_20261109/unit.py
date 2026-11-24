from pathlib import Path
import json,sys,time,hashlib,importlib.util
import numpy as np
import torch
from torch import nn
U=Path(__file__).resolve().parent;O=U.parent;P=O/'word100_separate_phase_learning_20261109'
sys.dont_write_bytecode=True;sys.path.insert(0,str(P))
import lib as prior
B=prior.B;D=prior.D;S=prior.S;M=prior.M;J=prior.common.J
read=prior.read;write=prior.write;sha=prior.sha
GROUPS=prior.GROUPS
GROUP_LIST=read(S/'FROZEN.json')['semantic_stage_groups']
GROUP_WORD=np.array([0,0,0,1,1,2,2,3,3,3])
def event(kind,**kw):
 d=dict(time=time.time(),kind=kind,**kw)
 with (U/'EVENTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(d,ensure_ascii=False)+'\n')
 print(json.dumps(d,ensure_ascii=False),flush=True)
def init_model():
 net=AttemptHead();old=torch.load(D/'runs/42/model.pt',weights_only=True,map_location='cpu');state=net.state_dict()
 for k,v in old.items():
  if k=='temporal.0.weight':state[k].zero_();state[k][:,:71]=v
  else:state[k]=v
 net.load_state_dict(state);return net
class AttemptHead(prior.Heads):
 def __init__(self):
  super().__init__(True)
  self.presence=nn.Sequential(nn.Linear(96,32),nn.GELU(),nn.Linear(32,10))
 def forward(self,h,p,g,word,valid,expected):
  t=torch.cat([h,p,g],-1).transpose(1,2)*valid[:,None,:]
  for layer in self.temporal:t=layer(t)*valid[:,None,:]
  ph=self.phase(t.transpose(1,2));allowed=(self.guide_words[None,:]==word[:,None])|(self.guide_words[None,:]<0);ph=ph.masked_fill(~allowed[:,None,:],-1e4)
  dif=self.difference(torch.cat([h,p,expected],-1));avg=t.sum(2)/valid.sum(1)[:,None];mx=t.masked_fill(~valid[:,None,:],-1e4).max(2).values
  return ph,dif,self.presence(torch.cat([avg,mx],-1))
def batch(rows):
 n=len(rows);T=max(len(r['raw']) for r in rows);out={k:np.zeros((n,T,w),np.float32) for k,w in [('h',64),('p',7),('g',66),('y',7),('ym',7)]}
 out.update(sy=np.full((n,T),-1,np.int64),sw=np.zeros((n,T),np.float32),valid=np.zeros((n,T),bool))
 for i,r in enumerate(rows):
  t=len(r['raw'])
  for k in ['h','p','g','y','ym','sy','sw']:out[k][i,:t]=r[k]
  out['valid'][i,:t]=True
 out={k:torch.from_numpy(v) for k,v in out.items()}
 out.update(word=torch.tensor([r['word'] for r in rows]),present=torch.tensor(np.stack([r['present'] for r in rows])),pmask=torch.tensor(np.stack([r['pmask'] for r in rows])))
 return out
def load_rows():
 out=[]
 for r in read(U/'INPUTS.json')['rows']:
  assert sha(r['path'])==r['sha256'];z=np.load(r['path']);out.append(dict(r,**{k:z[k] for k in z.files}))
 return out
def rebuild_hand_local(raw):
 p=raw[:,:102].reshape(-1,51,2);v=raw[:,186:237]>.5
 for side,base in enumerate([9,30]):
  lengths=[]
  for k in [5,9,13,17]:
   good=v[:,base]&v[:,base+k];lengths.extend(np.linalg.norm(p[good,base+k]-p[good,base],axis=1))
  hs=np.median(lengths) if lengths else 0
  local=np.zeros((len(raw),21,2),np.float32);mask=v[:,base:base+21]&v[:,base,None]
  if hs>1e-6:local[mask]=((p[:,base:base+21]-p[:,base,None])/hs)[mask]
  else:mask[:]=False
  raw[:,102+side*42:144+side*42]=local.reshape(len(raw),42);raw[:,237+side*21:258+side*21]=mask
 return raw
def close_hand(raw,side,a,b,donor):
 x=raw.copy();p=x[:,:102].reshape(-1,51,2);base=9 if side==0 else 30;donor=donor.copy()
 if side==0:donor[:,0]*=-1
 anchor=[0,5,9,13,17];valid=x[:,186:237]>.5
 for i in range(a,b+1):
  if not valid[i,base:base+21].all():continue
  target=p[i,base:base+21].copy();src=donor-donor[0];dst=target-target[0];u,_,vt=np.linalg.svd(src[anchor].T@dst[anchor]);rot=u@vt
  scale=np.median(np.linalg.norm(dst[anchor[1:]],axis=1))/max(1e-6,np.median(np.linalg.norm(src[anchor[1:]],axis=1)))
  aligned=src@rot*scale+target[0]
  # Only finger descendants change; wrist and palm anchors are exactly preserved.
  for k in [1,5,9,13,17]:
   for j in range(k+1,k+4):p[i,base+j]=target[k]+aligned[j]-aligned[k]
 return rebuild_hand_local(x)
