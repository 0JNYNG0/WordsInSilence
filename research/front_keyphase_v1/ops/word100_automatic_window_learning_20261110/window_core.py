from pathlib import Path
import sys,json,hashlib,importlib.util,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as TF
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2];P=O/'word100_broad_factorized_learning_20261110';N=O/'word100_motion_event_candidates_20261109';E=O/'word100_event_completion_20261109'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8');t.replace(p)
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def event(kind,**kw):
 r=dict(time=time.time(),kind=kind,**kw)
 with (U/'EVENTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 print(json.dumps(r,ensure_ascii=False),flush=True)
prior=load('_window_prior',P/'core.py');geom=load('_window_geometry',N/'core.py');shared=prior.shared
def geometry(x,ts):return geom.geometry(x,ts)
def relations(x,axis,moving='R',anchor='L'):
 q,v=prior.wrist_series(x,prior.motion.KINDS[axis],moving,anchor)
 root=x.copy();pp=root[:,:102].reshape(-1,51,2);pp[:,5]=pp[:,9];pp[:,6]=pp[:,30];root[:,191]=root[:,195];root[:,192]=root[:,216]
 q2,v2=prior.wrist_series(root,prior.motion.KINDS[axis],moving,anchor)
 # Two estimates of the same anatomical wrist, not finger tips or MCP centres.
 z=np.c_[np.where(v[:,None],q,0),np.where(v2[:,None],q2,0),v,v2].astype(np.float32)
 return z
class WindowNet(nn.Module):
 def __init__(self,mode,norm):
  super().__init__();self.mode=mode;self.encoder=geom.MotionProposal();ck=torch.load(N/'runs/final_42/model.pt',map_location='cpu',weights_only=True);self.encoder.load_state_dict(ck['model']);self.encoder.phase.requires_grad_(False)
  self.register_buffer('gmean',torch.tensor(norm['gmean'],dtype=torch.float32));self.register_buffer('gstd',torch.tensor(norm['gstd'],dtype=torch.float32));self.register_buffer('qmean',torch.tensor(norm['qmean'],dtype=torch.float32));self.register_buffer('qstd',torch.tensor(norm['qstd'],dtype=torch.float32))
  self.window=nn.Sequential(nn.Linear(87,48),nn.GELU(),nn.Linear(48,48),nn.GELU());self.core=nn.Linear(48,1);self.direction=nn.Linear(48,4)
  self.query=nn.Linear(64,32);self.guide=nn.Parameter(torch.tensor(norm['guide_keys'],dtype=torch.float32));self.order_head=nn.Sequential(nn.Linear(193,48),nn.GELU(),nn.Linear(48,1));self.word_head=nn.Linear(256,100)
  old=torch.load(P/'runs/coupled_42/model.pt',map_location='cpu',weights_only=True)
  with torch.no_grad():self.word_head.weight[:,:128].zero_();self.word_head.weight[:,128:].copy_(old['shared.word_head.weight']);self.word_head.bias.copy_(old['shared.word_head.bias'])
 def encode(self,g,lengths):
  v=torch.arange(g.shape[1])[None]<lengths[:,None];x=(g-self.gmean)/self.gstd;x[:,:,:22]*=g[:,:,22:33].repeat_interleave(2,-1);x[:,:,33:55]*=g[:,:,55:].repeat_interleave(2,-1)
  z=x.transpose(1,2)
  for layer in self.encoder.input:z=layer(z)*v[:,None,:]
  pk=nn.utils.rnn.pack_padded_sequence(z.transpose(1,2),lengths.cpu(),batch_first=True,enforce_sorted=False);h,_=self.encoder.context(pk);h,_=nn.utils.rnn.pad_packed_sequence(h,batch_first=True,total_length=g.shape[1]);return h,v
 def classify_words(self,h,frozen_h,lengths,v):
  def pool(x):return torch.cat([(x*v[:,:,None]).sum(1)/lengths[:,None],x.masked_fill(~v[:,:,None],-1e4).max(1).values],-1)
  return self.word_head(torch.cat([pool(h),pool(frozen_h)],-1))
 def movement(self,h,q,a,b,axis):
  q=q[a:b+1];hh=h[a:b+1];z=(q-self.qmean[axis])/self.qstd[axis];z=torch.cat([z[:,:4]*q[:,8,None],z[:,4:8]*q[:,9,None],q[:,8:]],-1)
  vel=torch.zeros((len(q),8),dtype=q.dtype);vel[1:]=(z[1:,:8]-z[:-1,:8])*len(q);valid=q[:,8:].max(1).values>.5
  kind=torch.zeros((len(q),5));kind[:,axis]=1;f=self.window(torch.cat([hh,z,vel,kind],-1));core=self.core(f).squeeze(-1)
  score=core if self.mode=='learned_core' else torch.zeros_like(core);weight=score.masked_fill(~valid,-1e4).softmax(0);logits=self.direction((f*weight[:,None]).sum(0))
  return logits,core,weight,valid
 def ordering(self,h,a,b,ga,gb):
  seq=h[a:b+1];keys=self.guide[torch.tensor([ga,gb])];score=self.query(seq)@keys.T/(32**.5);att=score.softmax(0)
  if self.mode=='uniform':
   weights=torch.zeros_like(att);n=max(1,len(seq)//3);weights[:n,0]=1/n;weights[-n:,1]=1/n
  else:weights=att
  vec=weights.T@seq;position=torch.linspace(0,1,len(seq));diff=(weights[:,1]*position).sum()-(weights[:,0]*position).sum()
  logit=self.order_head(torch.cat([vec[0],vec[1],vec[1]-vec[0],diff[None]])).squeeze(-1);return logit,att
def batch_g(rr,key='g_full'):
 T=max(len(r[key]) for r in rr);g=np.zeros((len(rr),T,66),np.float32);fh=np.zeros((len(rr),T,64),np.float32)
 for i,r in enumerate(rr):g[i,:len(r[key])]=r[key];fh[i,:len(r[key])]=r['frozen_h_'+key.split('_')[1]]
 return torch.tensor(g),torch.tensor(fh),torch.tensor([len(r[key]) for r in rr])
def transformed(r,view='full',reverse=False):
 x=prior.physical(r['x_'+view]);ids=np.arange(len(x))
 if reverse:
  anchors=[p[k] for p in r['pairs']+r['orders'] for k in ('a','b')]
  if anchors:a,b=min(anchors),max(anchors);ids[a:b+1]=ids[a:b+1][::-1]
 return x[ids].copy(),ids
def map_interval(ids,a,b):
 aa=int(np.flatnonzero(ids==a)[0]);bb=int(np.flatnonzero(ids==b)[0]);return min(aa,bb),max(aa,bb),aa>bb
def window_truth(r,ids,a,b,axis,moving='R'):
 core=np.zeros(b-a+1,np.float32);classes=[];intervals=[]
 for p in r['pairs']:
  if p['axis']!=axis or p.get('moving','R')!=moving:continue
  pa,pb,rev=map_interval(ids,p['a'],p['b']);lo=max(a,pa);hi=min(b,pb)
  if hi-lo<2:continue
  # All sufficiently represented reviewed strokes contribute. Opposite strokes
  # become mixed, never silently averaged into hold or a convenient direction.
  core[lo-a:hi-a+1]=1;classes.append(2-p['target'] if rev else p['target']);intervals.append([lo,hi])
 active=set(classes)-{1};target=3 if len(active)>1 else next(iter(active)) if active else 1
 return target,core,intervals
def frame_background(r,ids,a,b):
 anchors=r.get('background_frames',[]);mask=np.zeros(b-a+1,np.float32)
 for frame in anchors:
  ii=np.flatnonzero(r['frame_id']==frame)
  if len(ii):j=int(np.flatnonzero(ids==ii[0])[0]);
  else:continue
  if a<=j<=b:mask[j-a]=1
 return mask
def load_rows():
 rr=prior.rows();bg={r['clip_id']:[a['frame'] for a in r['anchors'] if a['label']==0] for r in read(E/'REVIEW_LABELS.json')['rows']}
 for r in rr:
  r['background_frames']=bg.get(r['clip_id'],[])
  if r['role']=='fit':
   z=np.load(U/'cache'/f"{r['clip_id']}.npz");r.update({k:z[k].copy() for k in z.files})
 return rr
