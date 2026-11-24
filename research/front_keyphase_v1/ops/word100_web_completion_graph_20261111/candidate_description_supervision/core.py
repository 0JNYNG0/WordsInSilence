from pathlib import Path
import sys,hashlib,importlib.util
import numpy as np
import torch
from torch import nn
W=Path(__file__).resolve().parent;U=W.parent;O=U.parent
sys.path.insert(0,str(U));sys.dont_write_bytecode=True
from graph import read,write,stamp
F=[2,3,4,5,7];AX={2:0,3:1,4:2,5:4,7:3}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def features(z,a,b):
 assert 0<=a<b<len(z)
 v=z[:,228:456];ix=np.arange(a,b+1);bins=np.array_split(ix,3)
 if any(len(g)==0 for g in bins):bins=[ix[:1],ix,ix[-1:]]
 # Temporal thirds are pooling bins, not invented action phases.
 return np.r_[*[v[g].mean(0) for g in bins],v[ix].std(0)].astype(np.float32)

def source_witnesses(source,contrast):
 groups={}
 for g in source['groups']:
  motion=sorted({f for e in source['events'] if e['event_id'] in g['event_ids'] for f in e['progression_witness_frames']})
  groups[g['family']]=dict(motion=motion,held=[])
 if source['hold_sample_frames']:
  last=max(source['groups'],key=lambda g:g['offset'][1]);groups[last['family']]['held']=source['hold_sample_frames']
 nonmotion=sorted(set(contrast['entry']+contrast['other_held_presentation']+contrast['returning']))
 assert not set(nonmotion)&{f for v in groups.values() for f in v['motion']+v['held']}
 return groups,nonmotion

def labels(a,b,family,groups,nonmotion,body_observed):
 g=groups[family];allmotion=set(g['motion']);ownholds=set(g['held']);other=set(nonmotion)|{f for k,v in groups.items() if k!=family for f in v['motion']+v['held']}
 present={f for f in allmotion if a<=f<=b and body_observed[f]};other_present={f for f in other if a<=f<=b and body_observed[f]};all_observed=all(body_observed[f] for f in allmotion);unseen_inside=any(not body_observed[f] for f in allmotion|other|ownholds if a<=f<=b)
 y=[0.,0.,0.];mask=[0.,0.,0.]
 if present:y[0]=1.;mask[0]=1.
 elif len(other_present)>=2 and not unseen_inside:mask[0]=1.
 if present and not unseen_inside:
  y[1]=float(not other_present);mask[1]=float(bool(other_present) or len(present)>=2)
 if present and all_observed:y[2]=float(allmotion<=present);mask[2]=1.
 basis=dict(own_progression=sorted(present),other_reviewed=sorted(other_present),required_observed_progression=sorted(allmotion),missing_observation=unseen_inside,linguistic_completion=None)
 return y,mask,basis

class CandidateHead(nn.Module):
 def __init__(self,mean,std):
  super().__init__();self.register_buffer('mean',torch.tensor(mean,dtype=torch.float32));self.register_buffer('std',torch.tensor(std,dtype=torch.float32));self.hidden=nn.Sequential(nn.Linear(912,96),nn.GELU());self.head=nn.Linear(96,15)
 def forward(self,x):return self.head(self.hidden((x-self.mean)/self.std)).reshape(-1,5,3)

