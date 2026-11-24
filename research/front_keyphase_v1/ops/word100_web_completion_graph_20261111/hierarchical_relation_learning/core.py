"""Hierarchy-aware relation head over frozen100word observations; no runtime writes."""
from pathlib import Path
import sys,hashlib,importlib.util
import numpy as np
import torch
from torch import nn
W=Path(__file__).resolve().parent;U=W.parent;O=U.parent;L=U/'action_linkage_plan'
sys.path.insert(0,str(U));sys.dont_write_bytecode=True
from graph import read,write,stamp
F=[2,3,4,5,7];AX={2:0,3:1,4:2,5:4,7:3}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def pair_features(z,a,b):
 """Symmetric endpoints and interval moments, no frame ID/gap/word as input."""
 a,b=sorted([int(a),int(b)]);assert 0<=a<b<len(z)
 v=np.asarray(z[:,228:456],np.float32);mid=v[a:b+1]
 return np.r_[(v[a]+v[b])/2,np.abs(v[a]-v[b]),mid.mean(0),mid.std(0)].astype(np.float32)

class RelationHead(nn.Module):
 def __init__(self,mean,std):
  super().__init__();self.register_buffer('mean',torch.tensor(mean,dtype=torch.float32));self.register_buffer('std',torch.tensor(std,dtype=torch.float32))
  self.net=nn.Sequential(nn.Linear(912,48),nn.GELU(),nn.Linear(48,2))
 def forward(self,x):return self.net((x-self.mean)/self.std)
 def probabilities(self,x):
  p=self(x).sigmoid();return torch.stack([p[:,0],p[:,0]*p[:,1],p[:,1]],1)

def metrics(y,mask,p,eligible=None):
 out={}
 for j,name in enumerate(['parent','atomic_given_parent']):
  use=mask[:,j]>0;pred=(p[:,0 if j==0 else 2]>=.5).astype(int);truth=y[:,j].astype(int);cm=np.zeros((2,3),int)
  for i in np.flatnonzero(use):cm[truth[i],pred[i] if eligible is None or eligible[i] else 2]+=1
  rec=[float(cm[k,k]/cm[k].sum()) if cm[k].sum() else None for k in [0,1]]
  out[name]=dict(confusion=cm.tolist(),correct=int(cm[0,0]+cm[1,1]),total=int(cm.sum()),recall=rec,balanced_accuracy=None if None in rec else sum(rec)/2)
 return out
