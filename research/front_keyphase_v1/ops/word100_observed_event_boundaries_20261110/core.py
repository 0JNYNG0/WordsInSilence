from pathlib import Path
import sys,json,time,hashlib,importlib.util
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent;PREV=O/'word100_event_phase_learning_20261110'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8');q.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(n,p):
 spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
old=load('_ob_phase',PREV/'phase_core.py');prior=old.prior;shared=old.shared
def event(k,**kw):
 d=dict(time=time.time(),kind=k,**kw)
 with (U/'EVENTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(d,ensure_ascii=False)+'\n')
 print(json.dumps(d,ensure_ascii=False),flush=True)
class BoundaryNet(old.PhaseNet):
 def __init__(self):
  super().__init__();self.load_state_dict(torch.load(PREV/'runs/42/model.pt',weights_only=True,map_location='cpu'));self.phase[-1]=nn.Linear(48,3)
  # Separate probabilities for observed movement, onset range and offset range.
  # At a direction turn, offset and next onset may coexist. No forced phase order.
def targets(r,annotations):
 n=len(r['frame_id']);y=np.zeros((n,10,3),np.float32);w=np.zeros_like(y)
 for row in annotations:
  k=row['axis']*2+(row['moving']=='L')
  for a,b in row['holds']+row['background']:w[a:b+1,k]=1
  for e in row['events']:
   a,aa=e['onset'];bb,b=e['offset'];w[a:b+1,k]=1;y[a:b+1,k,0]=1
  # Set positives after negatives, so neighbouring events cannot erase turn labels.
  for e in row['events']:
   a,aa=e['onset'];bb,b=e['offset'];y[a:aa+1,k,1]=1;y[bb:b+1,k,2]=1
 return y,w
def batch(rows,view='full',reverse=False):
 T=max(len(r['g_'+view]) for r in rows);g=np.zeros((len(rows),T,66),np.float32);q=np.zeros((len(rows),T,10,10),np.float32);fh=np.zeros((len(rows),T,64),np.float32);y=np.zeros((len(rows),T,10,3),np.float32);w=np.zeros_like(y);length=[]
 for i,r in enumerate(rows):
  n=len(r['g_'+view]);length.append(n);g[i,:n]=r['g_'+view];q[i,:n]=r['q_'+view];fh[i,:n]=r['frozen_h_'+view];y[i,:n]=r['boundary_y'];w[i,:n]=r['boundary_w']
  if reverse:
   # Full time reversal is a contrast for boundary detection only; no word-label loss.
   x=prior.physical(r['x_'+view])[::-1];g[i,:n]=old.old.geometry(x,r['timestamp_us']);q[i,:n]=old.relation_features(x);fh[i,:n]=fh[i,:n][::-1];y[i,:n]=y[i,:n][::-1];w[i,:n]=w[i,:n][::-1];y[i,:n,:,1],y[i,:n,:,2]=y[i,:n,:,2].copy(),y[i,:n,:,1].copy()
  valid=q[i,:n,:,-2:].max(-1)>.5;w[i,:n]*=valid[:,:,None]
 return tuple(torch.tensor(a) for a in [g,q,fh,np.array(length),y,w])
def groups(mask):
 return list(zip(np.flatnonzero(mask&~np.r_[False,mask[:-1]]).tolist(),np.flatnonzero(mask&~np.r_[mask[1:],False]).tolist()))
def decode(prob,valid):
 active=(prob[:,0]>=.5)&valid;beg=groups((prob[:,1]>=.5)&valid);end=groups((prob[:,2]>=.5)&valid);events=[];covered=np.zeros(len(prob),bool)
 for i,(a,aa) in enumerate(beg):
  next_a=beg[i+1][0] if i+1<len(beg) else len(prob)
  ends=[(b,bb) for b,bb in end if b>=aa and b<next_a]
  if not ends:continue
  b,bb=ends[0];lo=a+int(prob[a:aa+1,1].argmax());hi=b+int(prob[b:bb+1,2].argmax())
  if hi-lo+1<4 or active[lo:hi+1].mean()<.6 or valid[lo:hi+1].mean()<.8:continue
  # Both boundaries must be observed; never invent end from expected word/order.
  events.append(dict(a=lo,b=hi,onset_range=[a,aa],offset_range=[b,bb],complete=True,confidence=float(min(prob[a:aa+1,1].max(),prob[b:bb+1,2].max())),active_fraction=float(active[lo:hi+1].mean()),phases=[1]+[2]*(hi-lo-1)+[3]));covered[lo:hi+1]=True
 parts=[dict(a=a,b=b,complete=False,confidence=float(prob[a:b+1,0].mean()),phases=[2]*(b-a+1)) for a,b in groups(active&~covered)]
 return events,parts
