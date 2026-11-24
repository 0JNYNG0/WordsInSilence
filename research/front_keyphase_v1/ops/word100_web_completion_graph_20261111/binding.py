"""Common correspondence: no desired chronology, handshape, word ID or DTW input."""
import numpy as np
import torch
from torch import nn

def body_context(z,timestamps_us):
 # Legacy feature record = [before228, center228, after228].
 # Each228 = learned64 + body-relative relation100 + shared64.
 # Only body-relative wrist/arm/root relations are used, never finger shape.
 q=np.asarray(z,np.float32)[:,292:392];ts=np.asarray(timestamps_us)
 lo=np.searchsorted(ts,ts-133333);hi=np.searchsorted(ts,ts+133333,side='right')
 return np.c_[q,np.stack([q[a:b].mean(0) for a,b in zip(lo,hi)]),np.stack([np.ptp(q[a:b],axis=0) for a,b in zip(lo,hi)])].astype(np.float32)

class BodyContextLinker(nn.Module):
 def __init__(self,mean,std):
  super().__init__();self.register_buffer('mean',torch.as_tensor(mean));self.register_buffer('std',torch.as_tensor(std))
  self.net=nn.Sequential(nn.Linear(300,96),nn.GELU(),nn.Linear(96,48),nn.GELU(),nn.Linear(48,11))
 def forward(self,x):return self.net((x-self.mean)/self.std)

def bind_occurrences(probabilities,valid,timestamps_us,family_ids,min_confidence=.65):
 p=np.asarray(probabilities);v=np.asarray(valid,bool);ts=np.asarray(timestamps_us)
 assert p.ndim==2 and len(p)==len(v)==len(ts) and np.all(np.diff(ts)>=0)
 out=[]
 for family in sorted(set(family_ids)):
  # Preserve overlapping evidence if a caller supplies independent scores.
  # Softmax outputs at this threshold remain mutually exclusive.
  mask=(p[:,family]>=min_confidence)&v
  edges=np.diff(np.r_[False,mask,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)-1
  for a,b in zip(starts,ends):out.append(dict(family=int(family),a=int(a),b=int(b),confidence=float(p[a:b+1,family].mean()),seconds=[float(ts[a]/1e6),float(ts[b]/1e6)]))
 return sorted(out,key=lambda e:(e['a'],e['b'],e['family']))

def order_relation(occurrences,family_a,family_b):
 if family_a==family_b:return 'not_applicable'
 aa=[e for e in occurrences if e['family']==family_a];bb=[e for e in occurrences if e['family']==family_b]
 if not aa or not bb:return 'association_unknown'
 if max(e['b'] for e in aa)<min(e['a'] for e in bb):return 'supported'
 if max(e['b'] for e in bb)<min(e['a'] for e in aa):return 'reversed'
 return 'ambiguous'
