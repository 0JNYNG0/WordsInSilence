"""Observed handshape change is separate from instantaneous bent morphology."""
import numpy as np
import torch
from torch import nn
CLASSES=['opening','shape_maintained','closing']

def geometry(x,hand):
 p=np.asarray(x)[:,:102].reshape(-1,51,2);v=np.asarray(x)[:,186:237]>.5;b=30 if hand=='R'else 9
 size=np.linalg.norm(p[:,b+9]-p[:,b],axis=1);ids=[b];vals=[]
 def cosine(a,c):return (a*c).sum(1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(c,axis=1),1e-8)
 for f in [5,9,13,17]:
  ids+=list(range(b+f,b+f+4));a,c,d,e=[p[:,b+f+i]for i in range(4)];w=p[:,b]
  total=np.linalg.norm(c-a,axis=1)+np.linalg.norm(d-c,axis=1)+np.linalg.norm(e-d,axis=1)
  vals += [cosine(a-w,c-a),cosine(c-a,d-c),cosine(d-c,e-d),
    np.linalg.norm(e-a,axis=1)/np.maximum(total,1e-8),np.linalg.norm(e-w,axis=1)/np.maximum(size,1e-8)]
 out=np.stack(vals,1);ok=v[:,ids].all(1)&(size>1e-6)&np.isfinite(out).all(1)
 out=np.clip(np.nan_to_num(out),-10,10);out[~ok]=0
 return out.astype('float32'),ok

def pair_features(x,hand):
 g,ok=geometry(x,hand);available=bool(len(g)>=2 and ok[0]and ok[-1]and ok.mean()>=.8)
 feat=np.r_[g[-1]-g[0],(g[-1]+g[0])/2].astype('float32')
 if not available:feat[:]=0
 return feat,available

class ChangeHead(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Linear(40,16),nn.Tanh(),nn.Linear(16,3))
 def forward(self,x):return self.net(x)

def decision(pp,available):
 pp=np.asarray(pp);v=pp.argmax(-1)
 return CLASSES[int(v[0])]if available and len(set(v.tolist()))==1 and pp.max(-1).min()>=.8 else None

def completion_evidence(endpoint_result,change,observation_ok,association_ok=True):
 # Do not rewrite an endpoint prediction or treat movement as completion.
 if not association_ok:return dict(status='action_association_uncertain',whole_complete=None,educational_error=None)
 if not observation_ok:return dict(status='observation_uncertain',whole_complete=None,educational_error=None)
 return dict(status='components_available',endpoint_evidence=endpoint_result,change_evidence=change,
   whole_complete=None,educational_error=None,
   note='A bent shape alone says nothing about temporal progress; opening/closing alone says nothing about required final state.',display_enabled=False)
