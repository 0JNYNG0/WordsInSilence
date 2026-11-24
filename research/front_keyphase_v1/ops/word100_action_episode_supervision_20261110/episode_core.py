from pathlib import Path
import sys,json,hashlib,time,importlib.util
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
L=O/'word100_event_guide_linkage_20261110';T=O/'word100_event_triplet_context_20261110'
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
lc=load('_episode_link',L/'link_core.py');sys.modules['link_core']=lc
le=load('_episode_eval',L/'evaluate.py');bc=lc.bc;old=lc.old;prior=lc.prior;shared=lc.shared;B=lc.B;E=lc.E
read=lc.read;write=lc.write;sha=lc.sha
MAP={0:0,1:1,2:2,3:3,4:4,5:4,6:5,7:5,8:6,9:7}
FAMILY_NAMES=['background','face_presentation','toward_chest','forearm_turn','along_opposite_forearm','alternating_hands','hands_gather','hands_spread']
def event(kind,**kw):
 d=dict(time=time.time(),kind=kind,**kw)
 with (U/'EVENTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(d,ensure_ascii=False)+'\n')
 print(json.dumps(d,ensure_ascii=False),flush=True)

class EpisodeNet(lc.LinkNet):
 def __init__(self):
  super().__init__();self.load_state_dict(torch.load(L/'runs/42/model.pt',weights_only=True,map_location='cpu'))
  self.role=nn.Sequential(nn.Linear(228,96),nn.GELU(),nn.Linear(96,8))
  self.boundary=nn.Sequential(nn.Linear(145,64),nn.GELU(),nn.Linear(64,3))
 def forward(self,g,q,fh,lengths,drop_appearance=False):
  _,word,h=super().forward(g,q,fh,lengths);z=(q-self.qm)/self.qs
  z=torch.cat([z[...,:4]*q[...,8,None],z[...,4:8]*q[...,9,None],q[...,8:]],-1)
  # Learned shape/body appearance is a clue; it is never an expected-shape filter.
  app=fh*0 if drop_appearance else fh
  role=self.role(torch.cat([h,z.flatten(2),app],-1))
  ids=torch.arange(10,device=g.device);axis=F.one_hot(ids//2,5).float();side=F.one_hot(ids%2,2).float();query=torch.cat([axis,side],-1)[None,None].expand(g.shape[0],g.shape[1],-1,-1)
  bp=self.boundary(torch.cat([h[:,:,None].expand(-1,-1,10,-1),z,app[:,:,None].expand(-1,-1,10,-1),query],-1))
  return role,bp,word,h

def batch(rr,view,reverse=False):
 g,q,fh,ll,by,bw=bc.batch(rr,view,reverse);N,T=g.shape[:2];y=np.full((N,T),-1,np.int64);w=np.zeros((N,T),np.float32)
 for i,r in enumerate(rr):
  n=len(r['frame_id']);yy=r['role_y'];ww=r['role_w']
  if reverse:yy=yy[::-1];ww=ww[::-1]
  y[i,:n]=yy;w[i,:n]=ww
  for j in range(n):
   k=y[i,j];kk=[0] if k<=3 else [4,5] if k==4 else [6,7]
   if k>0 and not all(q[i,j,v,-2:].max()>.5 for v in kk):w[i,j]=0
 return g,q,fh,ll,by,bw,torch.tensor(y),torch.tensor(w)

def infer(net,x,ts,fh):
 g=old.old.geometry(x,ts);q=old.relation_features(x)
 with torch.inference_mode():rp,bp,_,_=net(torch.tensor(g[None]),torch.tensor(q[None]),torch.tensor(fh[None]),torch.tensor([len(x)]))
 return rp.softmax(-1)[0].numpy(),bp.sigmoid()[0].numpy(),q

def decode(rp,bp,q):
 events=[];parts=[]
 for k in range(10):
  ee,pp=bc.decode(bp[:,k],q[:,k,-2:].max(-1)>.5)
  for collection,out in [(ee,events),(pp,parts)]:
   for e in collection:
    p=rp[e['a']:e['b']+1].mean(0);out.append(dict(e,axis=k//2,moving='L' if k%2 else 'R',family=int(p.argmax()),family_confidence=float(p.max()),family_probabilities=p.tolist()))
 return events,parts

def family_contracts():
 d=read(L/'GUIDE_CONTRACTS.json')['words'];out={}
 for word,cc in d.items():
  groups={}
  for c in cc:
   root=c.get('parent',c['id']);groups.setdefault(root,set()).update(MAP[x] for x in c['contexts'])
  out[word]=[dict(c,families=sorted(groups[c.get('parent',c['id'])])) for c in cc]
 return out
