from pathlib import Path
import sys,json,hashlib,time,importlib.util
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
B=O/'word100_observed_event_boundaries_20261110';E=O/'word100_event_completion_20261109'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8');q.replace(p)
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
bc=load('_link_boundaries',B/'core.py');sys.modules['core']=bc
be=load('_link_boundary_eval',B/'evaluate.py');old=bc.old;prior=bc.prior;shared=bc.shared
def event(kind,**kw):
 d=dict(time=time.time(),kind=kind,**kw)
 with (U/'EVENTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(d,ensure_ascii=False)+'\n')
 print(json.dumps(d,ensure_ascii=False),flush=True)

class LinkNet(bc.BoundaryNet):
 def __init__(self):
  super().__init__();self.load_state_dict(torch.load(B/'runs/42/model.pt',weights_only=True,map_location='cpu'))
  self.phase.requires_grad_(False)
  self.link=nn.Sequential(nn.Linear(164,64),nn.GELU(),nn.Linear(64,10))
 def forward(self,g,q,fh,lengths):
  _,word,h=super().forward(g,q,fh,lengths)
  z=(q-self.qm)/self.qs
  z=torch.cat([z[:,:,:,:4]*q[:,:,:,8,None],z[:,:,:,4:8]*q[:,:,:,9,None],q[:,:,:,8:]],-1)
  return self.link(torch.cat([h,z.flatten(2)],-1)),word,h

def targets(r,ann):
 n=len(r['frame_id']);y=np.full(n,-1,np.int64);w=np.zeros(n,np.float32)
 if ann:
  for a in ann['anchors']:
   if a['label']>=0:y[a['frame']]=(a['label']+1)//2 if a['label'] else 0;w[a['frame']]=1
  # Only equal adjacent observed labels; transitions are not made up.
  for a,b in zip(ann['anchors'],ann['anchors'][1:]):
   if a['label']>=0 and a['label']==b['label'] and b['frame']-a['frame']<=6:
    idx=np.arange(a['frame']+1,b['frame']);y[idx]=y[a['frame']];w[idx]=.2
 return y,w

def batch(rows,view='full',reverse=False):
 T=max(len(r['g_'+view]) for r in rows);g=np.zeros((len(rows),T,66),np.float32);q=np.zeros((len(rows),T,10,10),np.float32);fh=np.zeros((len(rows),T,64),np.float32);y=np.full((len(rows),T),-1,np.int64);w=np.zeros((len(rows),T),np.float32);ll=[]
 for i,r in enumerate(rows):
  n=len(r['g_'+view]);ll.append(n);g[i,:n]=r['g_'+view];q[i,:n]=r['q_'+view];fh[i,:n]=r['frozen_h_'+view];y[i,:n]=r['link_y'];w[i,:n]=r['link_w']
  if reverse:
   xx=prior.physical(r['x_'+view])[::-1].copy();g[i,:n]=old.old.geometry(xx,r['timestamp_us']);q[i,:n]=old.relation_features(xx);fh[i,:n]=fh[i,:n][::-1];y[i,:n]=y[i,:n][::-1];w[i,:n]=w[i,:n][::-1]
  # No missing-joint sample is taught to look like a confirmed context.
  for j in range(n):
   cls=y[i,j]
   if cls>0:
    kk=[0] if cls<=3 else [4,5] if cls<=5 else [6,7]
    if not all(q[i,j,k,-2:].max()>.5 for k in kk):w[i,j]=0
 return tuple(torch.tensor(a) for a in [g,q,fh,np.array(ll),y,w])

def spans(prob,x,ts,threshold=.65):
 out=[];q=old.relation_features(x)
 for k in range(1,10):
  kk=[0] if k<=3 else [4,5] if k<=5 else [6,7]
  valid=np.ones(len(x),bool)
  for j in kk:valid&=q[:,j,-2:].max(-1)>.5
  for a,b in bc.groups((prob[:,k]>=threshold)&valid):
   if b-a>=1 and ts[b]-ts[a]>=60000:out.append(dict(context=k,a=a,b=b,confidence=float(prob[a:b+1,k].mean()),observed=True))
 return out

def order_state(aa,bb):
 if not aa or not bb:return 'unknown'
 # Preserve repetitions and ambiguity. Never choose just the convenient pair.
 states=set('forward' if a['b']<b['a'] else 'reverse' if b['b']<a['a'] else 'overlap' for a in aa for b in bb)
 if states=={'forward'}:return 'supported'
 if states=={'reverse'}:return 'contradicted'
 return 'ambiguous'

def compile_cards():
 cards=read(be.ev.D/'CARDS.json')['words']
 # Declarative guide semantics grounded in saved instructions. Runtime has no word-ID branches.
 mapping={
 'open_front':dict(contexts=[1],axis=None,sign=None),
 'close_chest':dict(contexts=[2],axis=0,sign=-1),
 'raise_forearm':dict(contexts=[3],axis=1,sign=1),
 'brush_arm':dict(contexts=[4],axis=2,sign=1),
 'reach_wrist':dict(contexts=[5],axis=None,sign=None,parent='brush_arm'),
 'alternating':dict(contexts=[6,7],axis=4,sign=None),
 'gather_hands':dict(contexts=[8],axis=None,sign=None),
 'spread_hands':dict(contexts=[9],axis=3,sign=1)}
 return {word:[dict(id=c['id'],required=c.get('required',True),instruction=c['instruction'],hands=c['hands'],**mapping[c['id']]) for c in cc] for word,cc in cards.items()}
