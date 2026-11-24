from pathlib import Path
import sys,json,time,hashlib,importlib.util
import numpy as np
import torch
from torch import nn
from torch.nn import functional as TF
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2];A=O/'word100_automatic_window_learning_20261110';P=O/'word100_broad_factorized_learning_20261110';E=O/'word100_event_completion_20261109'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8');q.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def event(kind,**kw):
 d=dict(time=time.time(),kind=kind,**kw)
 with (U/'EVENTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(d,ensure_ascii=False)+'\n')
 print(json.dumps(d,ensure_ascii=False),flush=True)
old=load('_ep_previous',A/'window_core.py');prior=old.prior;shared=old.shared
def relation_features(x):return np.stack([old.relations(x,axis,side,'L' if side=='R' else 'R') for axis in range(5) for side in ['R','L']],1)
def labels(r):
 n=len(r['frame_id']);y=np.zeros((n,10,5),np.float32);w=np.zeros((n,10),np.float32);direct=np.zeros((n,10),bool);conflict=np.zeros((n,10),bool)
 for a in r.get('background_frames',[]):
  hit=np.flatnonzero(r['frame_id']==a)
  if len(hit):y[hit,:,0]=1;w[hit]=.2
 for p in r['pairs']:
  k=p['axis']*2+(p.get('moving','R')=='L');a,b=p['a'],p['b'];span=np.arange(a,b+1);phase=np.full(len(span),4,np.int64) if p['target']==1 else np.where((span-a)/max(1,b-a)<.2,1,np.where((span-a)/max(1,b-a)>.8,3,2))
  for j,t in zip(span,phase):
   if w[j,k]>0 and y[j,k].argmax()!=t and y[j,k,0]==0:conflict[j,k]=True;continue
   y[j,k]=np.eye(5)[t];w[j,k]=.2
  # Endpoints are reviewed interval anchors, not exact biological motion boundaries.
  for j in [a,b]:direct[j,k]=True;w[j,k]=1
 w[conflict]=0;direct[conflict]=False
 return y,w,direct,conflict
def rows():
 rr=prior.rows();bg={r['clip_id']:[a['frame'] for a in r['anchors'] if a['label']==0] for r in read(E/'REVIEW_LABELS.json')['rows']}
 for r in rr:r['background_frames']=bg.get(r['clip_id'],[])
 return rr
class PhaseNet(nn.Module):
 def __init__(self):
  super().__init__();norm=read(A/'NORMALIZATION.json');self.encoder=old.geom.MotionProposal();ck=torch.load(old.N/'runs/final_42/model.pt',map_location='cpu',weights_only=True);self.encoder.load_state_dict(ck['model']);self.encoder.phase.requires_grad_(False)
  self.register_buffer('mean',torch.tensor(norm['gmean']));self.register_buffer('std',torch.tensor(norm['gstd']));self.register_buffer('qm',torch.tensor(norm['qmean']).repeat_interleave(2,0));self.register_buffer('qs',torch.tensor(norm['qstd']).repeat_interleave(2,0))
  self.phase=nn.Sequential(nn.Linear(89,48),nn.GELU(),nn.Linear(48,5));self.word=nn.Linear(256,100)
  ck=torch.load(P/'runs/coupled_42/model.pt',map_location='cpu',weights_only=True)
  with torch.no_grad():self.word.weight[:,:128].zero_();self.word.weight[:,128:].copy_(ck['shared.word_head.weight']);self.word.bias.copy_(ck['shared.word_head.bias'])
 def forward(self,g,q,fh,lengths):
  mask=torch.arange(g.shape[1])[None]<lengths[:,None];z=(g-self.mean)/self.std;z[:,:,:22]*=g[:,:,22:33].repeat_interleave(2,-1);z[:,:,33:55]*=g[:,:,55:].repeat_interleave(2,-1);h=z.transpose(1,2)
  for layer in self.encoder.input:h=layer(h)*mask[:,None]
  pack=nn.utils.rnn.pack_padded_sequence(h.transpose(1,2),lengths.cpu(),batch_first=True,enforce_sorted=False);h,_=self.encoder.context(pack);h,_=nn.utils.rnn.pad_packed_sequence(h,batch_first=True,total_length=g.shape[1])
  z=(q-self.qm)/self.qs;z=torch.cat([z[:,:,:,:4]*q[:,:,:,8,None],z[:,:,:,4:8]*q[:,:,:,9,None],q[:,:,:,8:]],-1);vel=torch.zeros_like(z[:,:,:,:8]);vel[:,1:]=(z[:,1:,:,:8]-z[:,:-1,:,:8])*30
  vel[:,1:,:,:4]*=(q[:,1:,:,8]*q[:,:-1,:,8])[:,:,:,None];vel[:,1:,:,4:]*=(q[:,1:,:,9]*q[:,:-1,:,9])[:,:,:,None]
  kind=torch.eye(5).repeat_interleave(2,0);side=torch.eye(2).repeat(5,1);emb=torch.cat([kind,side],-1)[None,None].expand(g.shape[0],g.shape[1],-1,-1);logits=self.phase(torch.cat([h[:,:,None].expand(-1,-1,10,-1),z,vel,emb],-1))
  def pool(x):return torch.cat([(x*mask[:,:,None]).sum(1)/lengths[:,None],x.masked_fill(~mask[:,:,None],-1e4).max(1).values],-1)
  return logits,self.word(torch.cat([pool(h),pool(fh)],-1)),h
def batch(rr,view='full',reverse=False):
 T=max(len(r['g_'+view]) for r in rr);g=np.zeros((len(rr),T,66),np.float32);q=np.zeros((len(rr),T,10,10),np.float32);fh=np.zeros((len(rr),T,64),np.float32);y=np.zeros((len(rr),T,10,5),np.float32);w=np.zeros((len(rr),T,10),np.float32);length=[]
 for i,r in enumerate(rr):
  n=len(r['g_'+view]);length.append(n);g[i,:n]=r['g_'+view];q[i,:n]=r['q_'+view];fh[i,:n]=r['frozen_h_'+view];y[i,:n]=r['phase_y'];w[i,:n]=r['phase_w']
  if reverse and r['pairs']:
   a=min(p['a'] for p in r['pairs']);b=max(p['b'] for p in r['pairs']);idx=np.arange(n);idx[a:b+1]=idx[a:b+1][::-1];x=prior.physical(r['x_'+view])[idx];g[i,:n]=old.geometry(x,r['timestamp_us']);q[i,:n]=relation_features(x);fh[i,:n]=fh[i,:n][idx];y[i,:n]=y[i,:n][idx];w[i,:n]=w[i,:n][idx];y[i,a:b+1,:,1],y[i,a:b+1,:,3]=y[i,a:b+1,:,3].copy(),y[i,a:b+1,:,1].copy()
  valid=q[i,:n,:,-2:].max(-1)>.5;w[i,:n]*=valid
 return tuple(torch.tensor(x) for x in [g,q,fh,np.array(length),y,w])
def proposals(prob,valid,linked=True):
 # No target word/direction, fixed duration or expected action order. Never remove
 # unused frames. All partial runs are retained, including failed phase chains.
 state=prob.argmax(-1);sure=prob.max(-1)>=.5;active=np.isin(state,[1,2,3])&sure&valid;events=[];partials=[]
 starts=np.flatnonzero(active&~np.r_[False,active[:-1]]);ends=np.flatnonzero(active&~np.r_[active[1:],False])
 for a,b in zip(starts,ends):
  cut=[int(a)]+[j for j in range(a+1,b+1) if state[j]==1 and state[j-1]>=2]+[int(b)+1]
  for lo,stop in zip(cut,cut[1:]):
   hi=stop-1;seq=state[lo:hi+1];sp=np.flatnonzero(seq==1);mp=np.flatnonzero(seq==2);ep=np.flatnonzero(seq==3);complete=bool(len(sp) and len(mp) and len(ep) and sp[0]<mp[-1]<ep[-1] and np.all(np.diff(seq)>=0));record=dict(a=lo,b=hi,phases=seq.tolist(),complete=complete,confidence=float(prob[lo:hi+1].max(-1).mean()))
   (events if (complete or not linked) and hi-lo>=3 else partials).append(record)
 return events,partials
