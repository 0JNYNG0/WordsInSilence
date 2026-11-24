"""Shared observable relation states; word/guide identity is not a model input."""
from prepare import U,O,read,write,sha
import sys, importlib.util
import numpy as np, torch
from torch import nn
sys.dont_write_bytecode=True

def load_module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

class Features:
 def __init__(self):
  rc=load_module('basic_requirement',O/'word100_guide_requirement_learning_20261111/requirement_core.py')
  self.net=rc.frozen_observer();self.base=self.net.shared.base
  self.norm=read(O/'word100_rtmw_joint_learning_20261108/feedback_viewfix/FEEDBACK_NORMALIZATION.json')
 def __call__(self,ch):
  x=ch['tcn280'].astype(np.float32).copy(); masks=np.c_[np.repeat(x[:,186:237]>.5,2,1),np.repeat(x[:,237:279]>.5,2,1)]
  px=x.copy();px[:,:186]=np.where(masks,(x[:,:186]-self.norm['mean'])/self.norm['std'],0)
  with torch.inference_mode():
   h=torch.nn.functional.gelu(self.base.input(torch.tensor(px[None]))).transpose(1,2)
   for layer in self.base.temporal:h=layer(h)
   h,_=self.base.context(h.transpose(1,2));h=h[0].numpy()
   coarse=self.base.attributes(torch.tensor(h)).sigmoid().numpy()
  rel=ch['body_relative_xy'];av=ch['point_available']; ids=[0,5,6,7,8,9,10,91,96,100,104,108,112,117,121,125,129]
  loc=np.where(av[:,ids,None],rel[:,ids],0)
  # Pair-relative geometry, including edge and dorsum landmarks, scale-normalized.
  lp=rel[:,[91,96,100,104,108]].mean(1)
  pair=rel[:,112:133]-lp[:,None]
  pair_av=av[:,112:133]&av[:,[91,96,100,104,108]].all(1)[:,None]
  pair=np.where(pair_av[:,:,None],pair,0)
  f=np.c_[loc.reshape(len(x),-1),av[:,ids],pair.reshape(len(x),-1),pair_av,h].astype(np.float32)
  quality=av[:,[5,6,91,96,100,104,108,112,117,121,125,129]].all(1)
  return f,quality,coarse

class PhaseNet(nn.Module):
 def __init__(self,n,kind):
  super().__init__();self.kind=kind
  self.input=nn.Sequential(nn.Linear(n,48),nn.GELU(),nn.Dropout(.1))
  if kind=='context_gru':
   self.conv=nn.Conv1d(48,48,3,padding=1);self.context=nn.GRU(48,24,batch_first=True,bidirectional=True)
  self.out=nn.Linear(48,3)
 def forward(self,x):
  z=self.input(x)
  if self.kind=='context_gru':
   z=torch.nn.functional.gelu(self.conv(z.transpose(1,2))).transpose(1,2)
   z,_=self.context(z)
  return self.out(z)

def states(prob,quality):
 label=prob.argmax(-1);label[prob.max(-1)<.65]=-1;label[~quality]=-1;return label

def approaches(labels,ts):
 """Observed separated -> near changes. No duration requirement or DTW fill."""
 out=[];armed=None;last_known=None
 for i,c in enumerate(labels):
  if c<0:
   if last_known is not None and ts[i]-ts[last_known]>100000:armed=None
   continue
  if c==0:armed=None
  elif c==1:
   if armed is None:armed=i
  elif c==2 and armed is not None:
   out.append({'start':int(armed),'end':i});armed=None
  last_known=i
 return out

class Observer:
 def __init__(self,kind):
  self.kind=kind;self.models=[]
  for seed in [42,43,44]:
   ck=torch.load(U/'runs'/f'{kind}_{seed}.pt',map_location='cpu',weights_only=False)
   m=PhaseNet(len(ck['mean']),kind);m.load_state_dict(ck['state']);m.eval()
   self.models.append((m,ck['mean'],ck['std']))
 def __call__(self,f,quality,ts):
  pp=[]
  for m,mu,sd in self.models:
   with torch.inference_mode():pp.append(m(torch.tensor(((f-mu)/sd)[None])).softmax(-1)[0].numpy())
  pp=np.stack(pp);ss=np.stack([states(p.copy(),quality) for p in pp]);s=np.where((ss==ss[0]).all(0),ss[0],-1)
  return {'states':s,'probabilities':pp,'approaches':approaches(s,ts)}
