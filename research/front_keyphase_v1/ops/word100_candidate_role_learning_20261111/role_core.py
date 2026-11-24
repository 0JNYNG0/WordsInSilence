from pathlib import Path
import sys,json,time,hashlib,importlib.util,copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent; O=U.parent; ROOT=O.parents[2]
A=O/'word100_autonomous_localization_20261111'
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
ac=load('_candidate_autoloc',A/'localization_core.py')
jc,old,prior,bc,lc,le,mc=ac.jc,ac.old,ac.prior,ac.bc,ac.lc,ac.le,ac.mc
J,P,B,L,M=ac.J,jc.P,ac.B,ac.L,ac.M
read,write,sha=ac.read,ac.write,ac.sha
def event(k,**kw):
 d=dict(time=time.time(),kind=k,**kw)
 with (U/'EVENTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(d,ensure_ascii=False)+'\n')
 print(json.dumps(d,ensure_ascii=False),flush=True)
def compact(cards):
 ds=[c['display_status'] for c in cards]
 return dict(display=ds,warning='difference' in ds,similar_no_warning=all(c['display_status']=='similar' for c in cards if c.get('required',True)) and 'difference' not in ds)

ROLE_NAMES=jc.ep.FAMILY_NAMES+['entry','return','transition']
class RoleAdapter(nn.Module):
 def __init__(self,seed=42):
  super().__init__();torch.manual_seed(seed)
  original=jc.JointNet();original.load_state_dict(torch.load(J/'runs/42/model.pt',weights_only=True,map_location='cpu'))
  self.role=nn.Sequential(nn.Linear(684,96),nn.GELU(),nn.Linear(96,11))
  with torch.no_grad():
   self.role[0].weight.zero_();self.role[0].weight[:,228:456].copy_(original.role[0].weight);self.role[0].bias.copy_(original.role[0].bias)
   self.role[2].weight[:8].copy_(original.role[2].weight);self.role[2].bias[:8].copy_(original.role[2].bias)
  self.quality=copy.deepcopy(original.quality);self.pair=copy.deepcopy(original.pair)
  ck=torch.load(ac.V/'runs/42/model.pt',weights_only=True,map_location='cpu');self.quality.load_state_dict(ck['quality']);self.pair.load_state_dict(ck['pair'])
 def forward(self,z):return self.role(z)

def frozen_model():
 n=jc.JointNet();n.load_state_dict(torch.load(J/'runs/42/model.pt',weights_only=True,map_location='cpu'));n.eval();n.requires_grad_(False);return n

def features(net,x,ts,fh):
 g=old.old.geometry(x,ts);q=old.relation_features(x)
 with torch.inference_mode():
  tq=torch.tensor(q[None]);tf=torch.tensor(fh[None]);rp,bp,_,h=net(torch.tensor(g[None]),tq,tf,torch.tensor([len(x)]))
  z=(tq-net.qm)/net.qs;z=torch.cat([z[...,:4]*tq[...,8,None],z[...,4:8]*tq[...,9,None],tq[...,8:]],-1)
  v=torch.cat([h,z.flatten(2),tf],-1)[0].numpy()
 # Real timestamp windows (~133ms) rather than fixed percentages of a clip.
 tt=np.asarray(ts);left=np.maximum(0,np.searchsorted(tt,tt-133333));right=np.minimum(len(tt)-1,np.searchsorted(tt,tt+133333))
 before=np.stack([v[a:i+1].mean(0) for i,a in enumerate(left)]);after=np.stack([v[i:b+1].mean(0) for i,b in enumerate(right)])
 return dict(z=np.c_[before,v,after].astype(np.float32),h=h,q=tq,fh=tf,old_role=rp.softmax(-1)[0].numpy(),boundary=bp.sigmoid()[0].numpy())

def segment_features(net,ob,spec):
 with torch.inference_mode():return net.segment(ob['h'],ob['q'],ob['fh'],spec).numpy()

