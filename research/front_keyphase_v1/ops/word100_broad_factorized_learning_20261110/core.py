"""100-word continuation with separately supervised observation components.
This isolated experiment never alters the deployed word/phase/completion models.
"""
from pathlib import Path
import sys,json,hashlib,importlib.util,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as TF
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
B=O/'word100_shared_motion_learning_20261109';M=O/'word100_motion_components_20261109'
C=O/'word100_common_action_learning_20261110';J=O/'word100_rtmw_joint_learning_20261108'
D=O/'word100_action_feedback_demo_20261108';F=O/'word100_completion_fullpath_20261110'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8');q.replace(p)
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def event(k,**kw):
 r=dict(time=time.time(),kind=k,**kw)
 with (U/'EVENTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 print(json.dumps(r,ensure_ascii=False),flush=True)
shared=load('_bf_shared',B/'core.py');motion=load('_bf_motion',M/'motion.py');STATS=shared.STATS
def physical(x):
 z=x.copy();z[:,:186]=z[:,:186]*np.asarray(STATS['std'])+np.asarray(STATS['mean']);return z.astype(np.float32)
def wrist_series(x,kind,moving='R',anchor='L'):
 # Body shoulder/elbow/wrist only. Finger flexion cannot change these inputs.
 p,v=motion.body_points(x);m=motion.SIDES[moving];r=motion.SIDES[anchor]
 p[:,:,0]*=(np.sign(p[:,2,0]-p[:,1,0])*(1 if moving=='R' else -1))[:,None]
 q=p[:,m['wrist']];other=p[:,r['wrist']]
 if kind=='wrist_height':
  out=np.column_stack([-q[:,1],q[:,0],-q[:,1],q[:,0]]);pts=[m['wrist']]
 elif kind=='forearm_upright':
  arm=q-p[:,m['elbow']];length=np.linalg.norm(arm,axis=1);out=np.column_stack([-arm[:,1]/np.maximum(length,1e-5),arm[:,0]/np.maximum(length,1e-5),length,-q[:,1]]);pts=[m['wrist'],m['elbow']]
 elif kind=='along_other_forearm':
  arm=other-p[:,r['elbow']];length=np.linalg.norm(arm,axis=1);rel=q-p[:,r['elbow']]
  out=np.column_stack([(rel*arm).sum(1)/np.maximum(length**2,1e-5),rel[:,0],rel[:,1],np.linalg.norm(q-other,axis=1)]);pts=[m['wrist'],r['wrist'],r['elbow']]
 elif kind in ('hand_separation','relative_hand_height'):
  diff=q-other;dist=np.linalg.norm(diff,axis=1);height=-diff[:,1]
  out=np.column_stack([dist if kind=='hand_separation' else height,abs(diff[:,0]),height if kind=='hand_separation' else dist,-q[:,1]]);pts=[m['wrist'],r['wrist']]
 else:raise ValueError(kind)
 good=v[:,pts].all(1)&np.isfinite(out).all(1)
 if kind in ('forearm_upright','along_other_forearm'):good&=length>.05
 return out,good
def geometry(x,ts,a,b,axis,moving='R',anchor='L',reverse=False):
 values,good=wrist_series(x,motion.KINDS[axis],moving,anchor);values=values[a:b+1];good=good[a:b+1];tt=np.asarray(ts[a:b+1],float)
 if len(tt)<1:return np.zeros(58,np.float32),False,dict(coverage=0.,observed=0)
 if reverse:values=values[::-1];good=good[::-1];tt=tt[-1]-tt[::-1]
 cov=float(good.mean());ok=cov>=.8 and int(good.sum())>=4 and bool(good[0] and good[-1])
 if not ok:return np.zeros(58,np.float32),False,dict(coverage=cov,observed=int(good.sum()))
 q=values[good];tg=tt[good];sampled=np.stack([np.interp(np.linspace(tg[0],tg[-1],9),tg,q[:,j]) for j in range(4)],1)
 delta=sampled[-1]-sampled[0];g=np.r_[(sampled-sampled[0]).ravel(),sampled[0],delta,np.abs(np.diff(sampled,axis=0)).mean(0),np.ptp(sampled,axis=0),np.eye(5)[axis],cov]
 return g.astype(np.float32),True,dict(coverage=cov,observed=int(good.sum()),signed_change=float(delta[0]),trajectory=sampled[:,0].tolist(),body_wrist_only=True,contact_verified=False)
class Network(nn.Module):
 def __init__(self,condition,norm):
  super().__init__();assert condition in ('coupled','factorized');self.condition=condition;self.shared=shared.SharedNet()
  self.register_buffer('gm',torch.tensor(norm['mean'],dtype=torch.float32));self.register_buffer('gs',torch.tensor(norm['std'],dtype=torch.float32))
  self.motion_conv=nn.Sequential(nn.Conv1d(4,16,3,padding=1),nn.GELU());self.motion_gru=nn.GRU(16,16,batch_first=True,bidirectional=True)
  self.motion_head=nn.Sequential(nn.Linear(32+58+192,64),nn.GELU(),nn.Linear(64,3))
  self.order_head=nn.Sequential(nn.Linear(192+64,64),nn.GELU(),nn.Linear(64,1))
 def movement(self,h,g,ii):
  i,a,b=ii.unbind(1);ha=h[i,a];hb=h[i,b];context=torch.cat([ha,hb,hb-ha],-1)
  if self.condition=='factorized':context=torch.zeros_like(context)
  z=(g-self.gm)/self.gs;seq=z[:,:36].reshape(-1,9,4);seq=self.motion_conv(seq.transpose(1,2)).transpose(1,2);_,hn=self.motion_gru(seq);m=torch.cat([hn[0],hn[1]],-1)
  return self.motion_head(torch.cat([m,z,context],-1))
 def ordering(self,h,ii,gg):
  i,a,b=ii.unbind(1);ha=h[i,a];hb=h[i,b];base=self.shared.base;guide=base.guide_encoder(base.guide_vectors)+base.stage_embedding
  return self.order_head(torch.cat([ha,hb,hb-ha,guide[gg[:,0]],guide[gg[:,1]]],-1)).squeeze(-1)
def rows():
 rr=shared.load_rows();extra=read(U/'ADDITIONAL_LABELS.json');oldmotion={r['clip_id']:r['pairs'] for r in read(M/'READY.json')['rows']}
 for r in rr:
  r['pairs']=list(oldmotion.get(r['clip_id'],r['pairs']))
  r['pairs']+=extra['motion'].get(r['clip_id'],[])
  r['orders']=list(r['orders'])+extra.get('orders',{}).get(r['clip_id'],[])
  for q in extra['shape'].get(r['clip_id'],[]):
   ix=np.flatnonzero(r['frame_id']==q['frame']);assert len(ix)==1
   for j,y in q['features'].items():
    j=int(j);i=int(ix[0]);assert not r['ym'][i,j] or r['y'][i,j]==y,'Existing reviewed label conflict'
    r['y'][i,j]=y;r['ym'][i,j]=1;r['direct_ym'][i,j]=1
 return rr
def component_batch(rr,view='full'):
 b=shared.batch(rr,view);g=[];ii=[];y=[];oi=[];og=[];oy=[]
 for i,r in enumerate(rr):
  x=physical(r['x_'+view])
  for p in r['pairs']:
   for rev in (False,True):
    z,ok,_=geometry(x,r['timestamp_us'],p['a'],p['b'],p['axis'],p.get('moving','R'),p.get('anchor','L'),rev)
    if ok:g.append(z);ii.append([i,p['b'] if rev else p['a'],p['a'] if rev else p['b']]);y.append(2-p['target'] if rev else p['target'])
  for p in r['orders']:
   # A reversed pair is a sequence contrast, never a normal word training sample.
   for rev in (False,True):oi.append([i,p['b'] if rev else p['a'],p['a'] if rev else p['b']]);og.append([p['guide_a'],p['guide_b']]);oy.append(0. if rev else 1.)
 b.update(g=torch.tensor(np.asarray(g,np.float32).reshape(-1,58)),ii=torch.tensor(ii,dtype=torch.long).reshape(-1,3),my=torch.tensor(y,dtype=torch.long),oi=torch.tensor(oi,dtype=torch.long).reshape(-1,3),og=torch.tensor(og,dtype=torch.long).reshape(-1,2),oy=torch.tensor(oy,dtype=torch.float32));return b
