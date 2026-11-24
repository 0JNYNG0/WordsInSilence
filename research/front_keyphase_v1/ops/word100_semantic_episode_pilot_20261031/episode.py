"""Bounded research observer: physical right-hand motion, never KSL correctness.

Existing shape/C/TCN outputs are not overwritten. Unknown is not class 'other'.
"""
from pathlib import Path
import json, hashlib, time, sys, copy
import numpy as np
import torch
from torch import nn

U=Path(__file__).resolve().parent
OPS=U.parent
OLD=OPS/'word100_stage_feedback_cycle_20261031'
sys.path.insert(0,str(OPS/'word100_guide_video_feedback_20261028'))
from motion_observer import extract_features,read_observation
CLASSES=['up','down','other']
COLS=[28,29,36,37,30,31,32,33]  # actual hand wrist/palm; body wrist/elbow kept distinct
SUPPORT=.75
WINDOW=13
SEEDS=[42,43,44]

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 p=Path(p);p.parent.mkdir(exist_ok=True,parents=True)
 with p.open('x',encoding='utf8') as out:json.dump(d,out,ensure_ascii=False,indent=2,allow_nan=False)
def rows():return read(U/'AUTHORIZATION.json')['rows']
def arrays(r):return dict(np.load(r['input'],allow_pickle=False))

def support(a):
 # Actual hand wrist is required; body-estimated wrist cannot restore an absent hand.
 return a['mask'][:,[28,29]].all(1)&np.isfinite(a['x'][:,[28,29]]).all(1)

def interval(a,start,end):
 ix=np.flatnonzero((a['frame_id']>=start)&(a['frame_id']<=end))
 return {k:a[k][ix].copy() for k in ['x','mask','frame_id','timestamp_us']}

def encode(a):
 x=np.asarray(a['x'][:,COLS],float);m=np.asarray(a['mask'][:,COLS],bool)&np.isfinite(x)
 n=len(x);relative=np.zeros_like(x);disp=np.zeros(8)
 for j in range(8):
  ids=np.flatnonzero(m[:,j])
  if len(ids):
   relative[ids,j]=x[ids,j]-x[ids[0],j]
   disp[j]=x[ids[-1],j]-x[ids[0],j]
 # Local changes require both samples; never bridge unseen original frame gaps.
 delta=np.zeros_like(x);dm=np.zeros_like(m)
 if n>1:
  good=m[1:]&m[:-1]&(np.diff(a['frame_id'])[:,None]==1)
  delta[1:]=np.where(good,x[1:]-x[:-1],0);dm[1:]=good
 t=np.asarray(a['timestamp_us'],float);t=(t-t[0])/max(t[-1]-t[0],1)
 z=np.concatenate([np.clip(relative,-5,5),np.clip(delta,-2,2),m,dm,np.tile(np.clip(disp,-5,5),(n,1)),t[:,None]],1)
 s=support(a);continuous=bool((np.diff(a['frame_id'])==1).all() and (np.diff(a['timestamp_us'])<=150000).all());eligible=n>=3 and continuous and s.mean()>=SUPPORT and s[:min(3,n)].any() and s[-min(3,n):].any()
 return z.astype('float32'),bool(eligible),float(s.mean())

class Head(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Conv1d(41,24,3,padding=1),nn.GELU(),nn.Conv1d(24,24,3,padding=1),nn.GELU());self.out=nn.Linear(24,3)
 def forward(self,x,valid):
  h=x.transpose(1,2);v=valid[:,None,:]
  for layer in self.net:h=layer(h)*v
  h=h.transpose(1,2)
  return self.out(h.sum(1)/valid.sum(1).clamp_min(1)[:,None])

def batch(items):
 length=max(len(x) for x in items);x=np.zeros((len(items),length,41),np.float32);m=np.zeros((len(items),length),np.float32)
 for i,a in enumerate(items):x[i,:len(a)]=a;m[i,:len(a)]=1
 return torch.from_numpy(x),torch.from_numpy(m)

def geometry(a):
 _,ok,cov=encode(a)
 if not ok:return -1
 ids=np.flatnonzero(support(a));y=a['x'][ids,29];k=min(3,max(1,len(ids)//3))
 dy=float(np.median(y[-k:])-np.median(y[:k]))
 return 0 if dy<-.03 else 1 if dy>.03 else 2

def old_predict(p,a):
 _,ok,_=encode(a)
 if not ok:return -1
 s=support(a);v=np.mean(p[s][:,[5,6]],axis=0)
 return int(v.argmax()) if v.max()>=.5 else 2

def infer(model,items):
 with torch.inference_mode():return model(*batch(items)).softmax(-1).numpy()

def automatic(a,model,old_prob=None):
 features=[];eligible=[];cov=[];geom=[];old=[]
 for i in range(len(a['x'])):
  lo=max(0,i-WINDOW//2);hi=min(len(a['x']),i+WINDOW//2+1)
  w={k:a[k][lo:hi].copy() for k in ['x','mask','frame_id','timestamp_us']}
  z,ok,c=encode(w);center_ok=bool(support(a)[i]);features.append(z);eligible.append(ok and center_ok);cov.append(c);geom.append(geometry(w) if center_ok else -1)
  if old_prob is not None:old.append(old_predict(old_prob[lo:hi],w) if center_ok else -1)
 prob=infer(model,features);pred=prob.argmax(1);pred[~np.asarray(eligible)]=-1
 return dict(prob=prob,pred=pred,geometry=np.asarray(geom),old=np.asarray(old),coverage=np.asarray(cov),eligible=np.asarray(eligible))

def runs(pred,frame,ts):
 result=[];start=0
 for end in range(1,len(pred)+1):
  if end<len(pred) and pred[end]==pred[start] and frame[end]==frame[end-1]+1 and ts[end]-ts[end-1]<=150000:continue
  result.append(dict(state='unknown' if pred[start]<0 else CLASSES[int(pred[start])],start_frame=int(frame[start]),end_frame=int(frame[end-1]),start_s=float(ts[start]/1e6),end_s=float(ts[end-1]/1e6),frames=end-start,short_proposal=end-start<3))
  start=end
 return result

def score(truth,pred):
 truth=np.asarray(truth,int);pred=np.asarray(pred,int);cm=np.zeros((3,4),int)
 for t,p in zip(truth,pred):cm[t,3 if p<0 else p]+=1
 f1=[]
 for c in range(3):
  tp=cm[c,c];fp=cm[:,c].sum()-tp;fn=cm[c].sum()-tp
  f1.append(float(2*tp/max(2*tp+fp+fn,1)))
 return dict(n=len(truth),correct=int((truth==pred).sum()),accuracy=float(np.mean(truth==pred)),macro_f1=float(np.mean(f1)),class_f1=f1,unknown=int((pred<0).sum()),conditional_accuracy=float((truth[pred>=0]==pred[pred>=0]).mean()) if (pred>=0).any() else None,confusion=cm.tolist(),classes=CLASSES,columns=CLASSES+['unknown'])

def compare(q,r,qstate,rstate):
 """Full original-frame DTW; each reference retained, missing query never removes cost.

 No pointwise best-reference mixing. 'Within' means empirical proximity only.
 """
 cols=COLS+[39,40,41,42,43,44,45,46,47,48,49,50,51,52]
 x=q['x'][:,cols];y=r['x'][:,cols];qm=q['mask'][:,cols]&np.isfinite(x);rm=r['mask'][:,cols]&np.isfinite(y)
 expected=rm[None,:,:];shared=qm[:,None,:]&expected
 delta=np.minimum(np.abs(x[:,None,:]-y[None,:,:]),1.)
 den=expected.sum(2);cost=np.divide(np.where(shared,delta,0).sum(2)+(expected&~qm[:,None,:]).sum(2),den,out=np.full((len(x),len(y)),.5),where=den>0)
 # Direction disagreement adds evidence but cannot lower a geometric difference.
 qs=np.asarray(qstate);rs=np.asarray(rstate)
 sm=(qs[:,None]>=0)&(rs[None,:]>=0)
 cost+=.25*(sm&(qs[:,None]!=rs[None,:]))
 n,m=cost.shape;d=np.full((n+1,m+1),np.inf);d[0,0]=0;back=np.full((n,m),-1,np.int8)
 for i in range(n):
  for j in range(m):
   if abs(i/max(n-1,1)-j/max(m-1,1))>.25:continue
   vals=(d[i,j],d[i,j+1],d[i+1,j]);k=int(np.argmin(vals));d[i+1,j+1]=cost[i,j]+vals[k];back[i,j]=k
 if not np.isfinite(d[n,m]):return dict(status='no_path',expected=int(rm.sum()),compared=0)
 i,j=n-1,m-1;path=[]
 while i>=0 and j>=0:
  path.append((i,j));k=back[i,j]
  if k==0:i-=1;j-=1
  elif k==1:i-=1
  elif k==2:j-=1
  else:raise ValueError('broken DTW')
 path.reverse();path=np.array(path)
 # Every original reference frame owns one comparison opportunity.
 mapping=np.array([int(np.median(path[path[:,1]==j,0])) for j in range(m)])
 shared_map=qm[mapping]&rm;expected_count=int(rm.sum());compared=int(shared_map.sum())
 coverage=compared/max(expected_count,1)
 # Retain every path point: median mapping above is DISPLAY ONLY. In particular
 # a short additional query motion must not vanish from the score after DTW.
 qi,ri=path.T;path_shared=qm[qi]&rm[ri];path_expected=rm[ri]
 pd=np.minimum(np.abs(x[qi]-y[ri]),1.)
 maxima=np.zeros_like(y);any_shared=np.zeros_like(rm)
 for k,(i,j) in enumerate(path):
  maxima[j]=np.maximum(maxima[j],np.where(path_shared[k],pd[k],0));any_shared[j]|=path_shared[k]
 compared=int(any_shared.sum());coverage=compared/max(expected_count,1)
 observed_distance=float(maxima.sum()/compared) if compared else None
 state_shared=(qs[qi]>=0)&(rs[ri]>=0)
 state_distance=float((qs[qi][state_shared]!=rs[ri][state_shared]).mean()) if state_shared.any() else None
 path_missing=int((path_expected&~path_shared).sum());state_missing=int(((rs[ri]>=0)&(qs[qi]<0)).sum())
 ref_unavailable=r['frame_id'][~rm.any(1)].tolist()
 # Missing is used for alignment/support, never as proof of a physical discrepancy.
 distance=observed_distance+.25*(state_distance or 0) if observed_distance is not None else None
 ref_direction_missing=r['frame_id'][rs<0].tolist()
 complete=expected_count>0 and path_missing==0 and state_missing==0 and len(ref_unavailable)==0 and not ref_direction_missing
 return dict(status='comparable' if coverage>=.75 and expected_count else 'insufficient',evidence_status='complete_relative_to_reference' if complete else 'incomplete_observation',distance=distance,observed_position_shape_distance=observed_distance,observed_direction_distance=state_distance,alignment_cost=float(np.mean(cost[qi,ri])),coverage=coverage,expected=expected_count,compared=compared,missing=expected_count-compared,path_missing=path_missing,reference_unavailable_frames=ref_unavailable,reference_unavailable_feature_cells=int(rm.size-rm.sum()),reference_direction_unavailable_frames=ref_direction_missing,direction_expected_pairs=int((rs[ri]>=0).sum()),direction_compared_pairs=int(state_shared.sum()),direction_missing_pairs=state_missing,reference_frames=m,reference_frame_ids=r['frame_id'].tolist(),query_map=q['frame_id'][mapping].tolist(),all_query_frames_accounted=int(len(np.unique(qi))),path_original_frames=np.column_stack([q['frame_id'][qi],r['frame_id'][ri]]).tolist(),uncompared_reference_frames=r['frame_id'][(rm&~any_shared).any(1)].tolist(),query_frames_with_missing_expected=np.unique(q['frame_id'][qi[(path_expected&~path_shared).any(1)]]).tolist(),direction_disagreements=int((state_shared&(qs[qi]!=rs[ri])).sum()),reference_scope_retained=True)

def raw_features(r,scale=1.,shift=(0.,0.)):
 raw=read_observation(r['observation']);b=np.asarray(r['box'],float)
 xy=raw['xy_px'];keep=(xy[:,:,0]>=b[0])&(xy[:,:,0]<b[2])&(xy[:,:,1]>=b[1])&(xy[:,:,1]<b[3]);raw['usable']&=keep
 # Transform every point together; eligibility kept physically equivalent with transformed ROI.
 raw['xy_px']=raw['xy_px'].astype(np.float64)*scale+np.asarray(shift)
 f=extract_features(raw)
 return dict(x=np.nan_to_num(f['x']).astype('float32'),mask=f['mask'],frame_id=f['frame_id'],timestamp_us=f['timestamp_us'])

def load_model(seed=42):
 d=U/'runs'/str(seed);done=read(d/'COMPLETE.json');assert sha(d/'head.pt')==done['checkpoint_sha256'] and sha(d/'TRAIN_TRACE.json')==done['trace_sha256']
 m=Head();m.load_state_dict(torch.load(d/'head.pt',weights_only=True));m.eval();return m
