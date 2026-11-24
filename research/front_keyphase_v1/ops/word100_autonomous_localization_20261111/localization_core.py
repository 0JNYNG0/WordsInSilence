from pathlib import Path
import sys, json, time, hashlib, importlib.util, copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent; O=U.parent; ROOT=O.parents[2]
M=O/'word100_short_motion_completion_20261111'; Q=O/'word100_autonomous_localization_plan_20261111'
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
mc=load('_autoloc_motion',M/'motion_core.py')
jc,prior,old,bc,lc,le=mc.jc,mc.prior,mc.old,mc.bc,mc.lc,mc.le
J,V,B,L=mc.J,mc.V,jc.B,jc.L
read,write,sha=mc.read,mc.write,mc.sha
def event(k,**kw):
 d=dict(time=time.time(),kind=k,**kw)
 with (U/'EVENTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(d,ensure_ascii=False)+'\n')
 print(json.dumps(d,ensure_ascii=False),flush=True)
def fitrows():
 rows=[r for r in old.rows() if r['role']=='fit']; assert len(rows)==899
 anns=read(J/'ALL_FIT_BOUNDARIES.json')['rows']
 for r in rows:
  r['annotations']=[a for a in anns if a['clip_id']==r['clip_id']]
 return rows

def relation_input(x,ts,lags):
 vals=[];valid=[]
 for k in range(10):
  side='L' if k%2 else 'R';v,ok=prior.wrist_series(x,prior.motion.KINDS[k//2],side,'L' if side=='R' else 'R');vals.append(v);valid.append(ok)
 v=np.stack(vals,1).astype(np.float32);ok=np.stack(valid,1);v=np.where(ok[...,None],v,0);features=[v];tt=np.asarray(ts,dtype=float)
 for lag in lags:
  lo=np.searchsorted(tt,tt-lag,side='left');hi=np.minimum(np.searchsorted(tt,tt+lag,side='left'),len(tt)-1);dt=np.maximum((tt[hi]-tt[lo])/1e6,1/30)
  good=ok[lo]&ok[hi]&ok
  features.append(np.where(good[...,None],(v[hi]-v[lo])/dt[:,None,None],0).astype(np.float32));features.append(good[...,None].astype(np.float32))
 features.append(ok[...,None].astype(np.float32));axis=np.eye(5)[np.arange(10)//2];side=np.eye(2)[np.arange(10)%2];features.append(np.broadcast_to(np.c_[axis,side],(len(x),10,7)).astype(np.float32))
 return np.concatenate(features,-1),ok

class Localizer(nn.Module):
 def __init__(self,norm):
  super().__init__();d=len(norm['mean']);self.register_buffer('mean',torch.tensor(norm['mean'],dtype=torch.float32));self.register_buffer('std',torch.tensor(norm['std'],dtype=torch.float32))
  # Reuse and fine-tune the existing local Conv+GRU weights, keep its original
  # checkpoint and the 100-word common encoder unchanged.
  m=mc.MotionNet();m.load_state_dict(torch.load(M/'runs/42/model.pt',weights_only=True,map_location='cpu'),strict=False)
  self.conv=copy.deepcopy(m.motion_conv);self.gru=copy.deepcopy(m.motion_gru)
  self.state=nn.Sequential(nn.Linear(32+d+64,64),nn.GELU(),nn.Linear(64,4))
  self.boundary=nn.Sequential(nn.Linear(32+d+64,48),nn.GELU(),nn.Linear(48,2))
 def forward(self,features,fh,lengths):
  z=((features-self.mean)/self.std).clamp(-12,12);b,t,k,d=z.shape
  inp=z[...,:4].permute(0,2,3,1).reshape(b*k,4,t);a=self.conv(inp).transpose(1,2)
  lens=lengths[:,None].expand(-1,k).reshape(-1)
  packed=nn.utils.rnn.pack_padded_sequence(a,lens.cpu(),batch_first=True,enforce_sorted=False);h,_=self.gru(packed);h,_=nn.utils.rnn.pad_packed_sequence(h,batch_first=True,total_length=t);h=h.reshape(b,k,t,32).permute(0,2,1,3)
  cat=torch.cat([h,z,fh[:,:,None].expand(-1,-1,k,-1)],-1)
  return self.state(cat),self.boundary(cat)

def frame_targets(n,anns,valid,reverse=False):
 y=np.full((n,10),-1,np.int64);by=np.zeros((n,10,2),np.float32);bw=np.zeros_like(by);conflict=np.zeros((n,10),bool)
 def put(a,b,k,label):
  a=max(0,a);b=min(n-1,b);prev=y[a:b+1,k];conflict[a:b+1,k]|=(prev>=0)&(prev!=label);y[a:b+1,k]=label;bw[a:b+1,k]=1
 for ar in anns:
  k=ar['axis']*2+(ar['moving']=='L')
  for a,b in ar['background']:put(a,b,k,3)
  for a,b in ar['holds']:put(a,b,k,1)
  for e in ar['events']:
   # Uncertain turning frames are NOT assigned two opposite directions.
   a=int(round(np.mean(e['onset'])));b=int(round(np.mean(e['offset'])));put(a,b,k,e['direction'])
   for j,rr in enumerate([e['onset'],e['offset']]):by[rr[0]:rr[1]+1,k,j]=1;bw[rr[0]:rr[1]+1,k,j]=1
 y[conflict]=-1
 if reverse:
  y=y[::-1].copy();mask=(y==0)|(y==2);y[mask]=2-y[mask];by=by[::-1,:,::-1].copy();bw=bw[::-1].copy()
 y[~valid]=-1;bw*=valid[...,None]
 return y,by,bw

def infer_local(net,x,ts,fh,cfg):
 feat,valid=relation_input(x,ts,cfg['lags_us'])
 with torch.inference_mode():
  sp,bp=net(torch.tensor(feat[None]),torch.tensor(fh[None]),torch.tensor([len(x)]));sp=sp.softmax(-1)[0].numpy();bp=bp.sigmoid()[0].numpy()
 labels=sp.argmax(-1);events=[];partials=[]
 for k in range(10):
  lab=labels[:,k].copy();lab[(sp[:,k].max(-1)<cfg['state_threshold'])|~valid[:,k]]=-1
  for a,b in bc.groups((lab==0)|(lab==2)):
   starts=[a]+[i for i in range(a+1,b+1) if lab[i]!=lab[i-1]];ends=[i-1 for i in starts[1:]]+[b]
   for lo,hi in zip(starts,ends):
    # Boundary heads refine within a local range, not one maximum over an entire
    # high-activity run containing several repetitions.
    rad=cfg['boundary_radius'];la=max(a,lo-rad);hb=min(b,hi+rad)
    aa=la+int(bp[la:min(lo+rad+1,hi+1),k,0].argmax());bb=max(lo,hi-rad)+int(bp[max(lo,hi-rad):hb+1,k,1].argmax())
    if bb-aa<cfg['min_span_frames'] or not valid[aa:bb+1,k].all():continue
    e=dict(a=int(aa),b=int(bb),axis=k//2,moving='L' if k%2 else 'R',source='learned_temporal_state',prediction=int(lab[lo]),confidence=float(sp[lo:hi+1,k,lab[lo]].mean()),coverage=1.,complete=False,semantic_complete=False,onset_confidence=float(bp[aa,k,0]),offset_confidence=float(bp[bb,k,1]))
    # Missing an endpoint is retained as an attempted action, not discarded.
    e['boundary_supported']=e['onset_confidence']>=.5 and e['offset_confidence']>=.5
    (events if e['boundary_supported'] else partials).append(e)
 return dict(events=events,partials=partials,state=sp,boundary=bp,valid=valid)
