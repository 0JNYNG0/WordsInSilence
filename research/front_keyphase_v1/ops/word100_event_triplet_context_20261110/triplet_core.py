from pathlib import Path
import sys,json,hashlib,time,importlib.util
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
L=O/'word100_event_guide_linkage_20261110'
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
lc=load('_triplet_link',L/'link_core.py');sys.modules['link_core']=lc
le=load('_triplet_link_eval',L/'evaluate.py');guard=load('_triplet_guard',L/'semantic_guard.py')
bc=lc.bc;old=lc.old;prior=lc.prior;shared=lc.shared;B=lc.B;E=lc.E
read=lc.read;sha=lc.sha;write=lc.write
def event(kind,**kw):
 d=dict(time=time.time(),kind=kind,**kw)
 with (U/'EVENTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(d,ensure_ascii=False)+'\n')
 print(json.dumps(d,ensure_ascii=False),flush=True)

def make_segments(ann,n):
 # These bounds delimit reviewed anchor runs, not manually annotated exact boundaries.
 runs=[];last=None
 for a in ann['anchors']:
  k=(a['label']+1)//2 if a['label']>0 else a['label']
  if k<0:last=None;continue
  if last is None or last['context']!=k:
   last=dict(context=k,anchors=[]);runs.append(last)
  last['anchors'].append(a)
 out=[]
 for r in runs:
  aa=r['anchors'];a=aa[0]['frame'];b=aa[-1]['frame']
  if a==b:a=max(0,a-1);b=min(n-1,b+1)
  out.append(dict(a=a,b=b,context=r['context'],state=2 if any(x['label']>0 and x['label']%2==0 for x in aa) else 1 if r['context']>0 else 0,anchors=aa,source='reviewed_anchor_run'))
  # A progress-only subwindow must not borrow a later core appearance as a positive label.
  for x in aa:
   if x['label']>0 and x['label']%2:
    out.append(dict(a=max(0,x['frame']-1),b=min(n-1,x['frame']+1),context=r['context'],state=1,anchors=[x],source='reviewed_progress_anchor'))
 return out

def windows(a,b,n,ts):
 width=max(100000,min(400000,(int(ts[b])-int(ts[a]))//2))
 l=int(np.searchsorted(ts,int(ts[a])-width));r=int(np.searchsorted(ts,int(ts[b])+width,side='right'))
 bins=np.array_split(np.arange(a,b+1),3)
 # Empty short bins repeat an in-window observation; no future frame enters core evidence.
 center=[(int(z[0]),int(z[-1])+1) if len(z) else (a,a+1) for z in bins]
 return [(l,a)]+center+[(b+1,min(n,r))]

class TripletNet(lc.LinkNet):
 def __init__(self):
  super().__init__();self.load_state_dict(torch.load(L/'runs/42/model.pt',weights_only=True,map_location='cpu'))
  self.event_head=nn.Sequential(nn.Linear(820,128),nn.GELU(),nn.Linear(128,10))
  self.core_head=nn.Sequential(nn.Linear(300,64),nn.GELU(),nn.Linear(64,3))
 def qfeatures(self,q):
  z=(q-self.qm)/self.qs
  return torch.cat([z[...,:4]*q[...,8,None],z[...,4:8]*q[...,9,None],q[...,8:]],-1).flatten(2)
 def segment(self,h,q,items,times,masked=False):
  zq=self.qfeatures(q);z=torch.cat([h,zq],-1);vv=[];cc=[]
  for i,a,b in items:
   ww=windows(a,b,len(times[i]),times[i]);pooled=[];center=[]
   for j,(lo,hi) in enumerate(ww):
    pooled.append(z[i,lo:hi].mean(0) if hi>lo else z[i,0]*0)
    if 1<=j<=3:center.append(zq[i,lo:hi].mean(0))
   vv.append(torch.cat(pooled));cc.append(torch.cat(center))
  v=torch.stack(vv);c=torch.stack(cc)
  # Core head only sees center raw relations; BiGRU and neighbours cannot fill it in.
  return self.event_head(v),self.core_head(c*0 if masked else c)

def batch(rr,view='full',reverse=False):
 g,q,fh,ll,y,w=lc.batch(rr,view,reverse);items=[];ey=[];cy=[];times=[]
 for i,r in enumerate(rr):
  n=len(r['frame_id']);times.append(r['timestamp_us'])
  for s in r.get('segments',[]):
   a,b=s['a'],s['b']
   if reverse:a,b=n-1-b,n-1-a
   valid=context_valid(q[i].numpy(),s['context'],a,b)>=.8
   items.append((i,a,b));ey.append(s['context'] if valid else -100);cy.append(s['state'] if valid else 0)
 return g,q,fh,ll,y,w,items,torch.tensor(ey),torch.tensor(cy),times

def context_valid(q,k,a,b):
 kk=[0] if k<=3 else [4,5] if k<=5 else [6,7]
 return float(np.logical_and.reduce([q[a:b+1,j,-2:].max(-1)>.5 for j in kk]).mean())

def candidate_spans(base):
 # Independent body-context proposals plus all detected movements, including partials.
 spans={(s['a'],s['b']) for c in base['guide_link_bindings'] for s in c['context_spans']}
 spans.update((e['a'],e['b']) for e in base['phase_events']+base['phase_partials'] if e['b']-e['a']>=2)
 return sorted(spans)

def predict_segments(net,x,ts,spans):
 g=old.old.geometry(x,ts);q=old.relation_features(x)
 with torch.inference_mode():
  tq=torch.tensor(q[None]);pr,_,h=net(torch.tensor(g[None]),tq,torch.zeros((1,len(x),64)),torch.tensor([len(x)]))
  if not spans:return [],pr.softmax(-1)[0].numpy()
  ep,cp=net.segment(h,tq,[(0,a,b) for a,b in spans],[ts]);ep=ep.softmax(-1).numpy();cp=cp.softmax(-1).numpy()
 out=[]
 for (a,b),p,c in zip(spans,ep,cp):
  k=int(p.argmax());cov=context_valid(q,k,a,b)
  out.append(dict(a=a,b=b,context=k,confidence=float(p.max()),probabilities=p.tolist(),core_probabilities=c.tolist(),core_state='unobserved' if cov<.8 else ['unconfirmed','progress','core_appearance'][int(c.argmax())],coverage=cov,observed=cov>=.8))
 return out,pr.softmax(-1)[0].numpy()

def accepted_spans(pred):
 # Keep all overlapping/repeated hypotheses; never force the expected word order.
 return [dict(a=s['a'],b=s['b'],context=s['context'],confidence=s['confidence'],observed=True) for s in pred if s['context']>0 and s['confidence']>=.65 and s['observed']]
