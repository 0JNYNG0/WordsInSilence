"""Read-only public TRAIN position-evidence audit. No detector or model inference.

Counterfactuals change diagnostic arguments in memory only. They are not a
proposed deployable fix, reconstructed hands, or measured recognition gains.
"""
import os, sys, json, hashlib, time, traceback, shutil
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone, timedelta
import numpy as np
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent; O=U.parent; ROOT=O.parents[2]
SRC=O/'word100_hand_evidence_training_20261018'
NAMES=['observed_points','partial_points','wrist_interior','wrist_boundary_or_outside','assignment_unresolved','unknown']
VARIANTS=['masked','sensor','sensor_wrist_outside_blocked','sensor_visible_assignment','sensor_visible_pose','sensor_common_policy']
PALM=np.array([0,5,9,13,17])
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def now():return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')
def write(p,x,new=True):
 p=Path(p)
 if new and p.exists():raise FileExistsError(p)
 tmp=p.with_name(p.name+'.writing');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,p)
def checkpins(pins):
 for p,h in pins.items():assert sha(p)==h,p
def args(obs,raw,ass,view,w,h):
 xy=obs['xy_px'].copy();mask=obs['usable'].copy();conf=raw['pose_conf'].copy()
 allowed=np.isfinite(xy).all(2)&(xy[:,:,0]>=0)&(xy[:,:,0]<=w)&(xy[:,:,1]>=0)&(xy[:,:,1]<=h)&mask
 visible_assigned=np.stack([allowed[:,33+s*21:54+s*21].any(1) for s in (0,1)],1)
 if view=='masked':
  xy[~allowed]=np.nan;mask=allowed;conf[~allowed[:,:33]]=0;assigned=visible_assigned;unassigned=np.zeros(len(xy),bool)
 else:
  xy[:,:33]=raw['pose_xy_px'];valid=np.arange(ass['sides'].shape[1])[None,:]<raw['raw_count'][:,None]
  assigned=np.stack([((ass['sides']==s)&valid).any(1) for s in (0,1)],1);unassigned=((ass['sides']<0)&valid).any(1)
 return (xy,mask,conf,assigned,unassigned)
def position(a,w,h):
 """Independent vectorized reconstruction of current position branch only.
 History can change overlap_likely vs inside_missing, both position index 2.
 """
 xy,mask,conf,assigned,unassigned=a;T=len(xy)
 finite=np.isfinite(xy).all(2);inside=finite&(xy[:,:,0]>=0)&(xy[:,:,0]<=w)&(xy[:,:,1]>=0)&(xy[:,:,1]<=h)
 trusted=np.isfinite(conf)&(conf>=.5)&finite[:,:33]
 tr_inside=trusted&mask[:,:33]&inside[:,:33]
 scale=np.linalg.norm(xy[:,11]-xy[:,12],axis=1);safe=np.where(scale>1e-6,scale,1)
 shoulder=tr_inside[:,11]&tr_inside[:,12]&(scale>1e-6)
 states=np.full((T,2),5,np.int8);nc=np.zeros((T,2),np.int8)
 for s,(sh,el,wr) in enumerate(((11,13,15),(12,14,16))):
  hand=np.arange(33+s*21,54+s*21);palm=33+s*21+PALM
  observed=(mask[:,palm]&finite[:,palm]).all(1);nc[:,s]=(mask[:,hand]&finite[:,hand]).sum(1)
  upper=np.linalg.norm(xy[:,sh]-xy[:,el],axis=1)/safe;lower=np.linalg.norm(xy[:,el]-xy[:,wr],axis=1)/safe
  geom=shoulder&tr_inside[:,sh]&tr_inside[:,el]&trusted[:,wr]&(upper>=.1)&(upper<=1.5)&(lower>=.1)&(lower<=1.5)
  p=xy[:,wr];d=np.stack([p[:,0],w-p[:,0],p[:,1],h-p[:,1]],1)/safe[:,None]
  edge=np.argmin(np.where(np.isfinite(d),d,np.inf),axis=1);normals=np.array([[-1.,0],[1.,0],[0,-1.],[0,1.]])
  outward=((p-xy[:,el])*normals[edge]).sum(1)/safe;near=(d[np.arange(T),edge]<=.15)&(outward>.15)
  interior=finite[:,wr]&(p[:,0]>=.15*safe)&(p[:,0]<=w-.15*safe)&(p[:,1]>=.15*safe)&(p[:,1]<=h-.15*safe)
  states[geom&interior,s]=2;states[geom&near,s]=3
  states[unassigned,s]=4;states[assigned[:,s],s]=1;states[observed,s]=0
 return states,nc
def freeze():
 assert not (U/'CONTEXT.json').exists()
 c=read(SRC/'PREP_CONTEXT.json');rows=[r for r in c['records'] if r['role']=='train_candidate'];assert len(rows)==899
 pins={str(SRC/'PREP_CONTEXT.json'):sha(SRC/'PREP_CONTEXT.json'),str(U/'audit.py'):sha(U/'audit.py')}
 for p in [SRC/'adapter.py',O/'word100_hand_visibility_20261017/visibility.py',O/'word100_hand_evidence_split_20261017/separate.py']:
  pins[str(p)]=sha(p)
 for r in rows:
  assert 1<=r['signer_id']<=9
  p=SRC/'inputs_v2'/f"{r['clip_id']}.json";saved=read(p);pins[str(p)]=sha(p);pins.update(saved['outputs']);r['state_views']=saved['views']
  for view in r['views'].values():
   for p in view.values():pins[p]=c['pins'][p]
  pins[r['record']]=c['pins'][r['record']]
 rows.sort(key=lambda r:r['clip_id'])
 context={'at':now(),'authorization':'공개 자료로 위치 상태 차이 원인 분석·보완 방향 보고. 새 학습은 결과 검토 후.',
  'scope':'TRAIN899 only; no validation, heldout, user or mentor observations; no classifier scores/inference.',
  'primary_denominator':'Same source frame ID and timestamp intersection of saved masked/sensor whole windows, each side once.',
  'variants':VARIANTS,'position_names':NAMES,
  'counterfactual_limit':'In-memory diagnostic argument changes only; not deployable harmonization, ground truth, or additive causal attribution.',
  'sample_rule':'For each prespecified condition, lexicographically first distinct clip; midpoint of its longest contiguous qualifying hand-frame run. At most two clips per stratum. No word/score-based selection.',
  'strata':['palm_observation_difference','zero_visible_partial','outside_wrist_extrapolation','unassigned_priority','same_visible_flags_different_policy','residual_arm_evidence','unchanged_palm_control'],
  'rows':rows,'pins':pins,'training':0,'deployment':0}
 write(U/'CONTEXT.json',context);(U/'before').mkdir()
 for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:shutil.copy2(p,U/'before'/p.name)
 return context
def summarize(arr):
 m,s,wr,ass,pose,common=[arr['states'][:,i] for i in range(6)];diff=m!=s;N=len(m)
 nc=arr['counts'];same_flags=(nc[:,0]>0)==(nc[:,1]>0)
 groups={'palm_observation_difference':(m==0)!=(s==0),
 'zero_visible_partial':(s==1)&(nc[:,1]==0),
 'outside_wrist_extrapolation':s!=wr,
 'unassigned_priority':s==4,
 'same_visible_flags_different_policy':diff&same_flags&((m==0)==(s==0))&(m==common),
 'residual_arm_evidence':diff&(m!=common)&(m>=2)&(common>=2),
 'unchanged_palm_control':(m==0)&(s==0)}
 out={'total_hand_frames':N,'clips':len(np.unique(arr['clip_index'])),'mismatch':int(diff.sum()),'mismatch_percent':100*float(diff.mean()),
 'transition_matrix_masked_rows_sensor_columns':np.bincount(m*6+s,minlength=36).reshape(6,6).tolist(),
 'state_counts':{v:np.bincount(arr['states'][:,i],minlength=6).tolist() for i,v in enumerate(VARIANTS)},
 'counterfactuals':{},'strata':{},'non_additive_warning':True}
 for i,v in enumerate(VARIANTS[2:],2):
  q=arr['states'][:,i];change=s!=q
  out['counterfactuals'][v]={'sensor_changed':int(change.sum()),'original_mismatch_closed':int((diff&(m==q)).sum()),'original_mismatch_remains':int((diff&(m!=q)).sum()),'new_mismatch_from_original_agreement':int((~diff&(m!=q)).sum()),'total_mismatch_after':int((m!=q).sum())}
 for k,mask in groups.items():out['strata'][k]={'all_hand_frames':int(mask.sum()),'among_original_mismatch':int((mask&diff).sum()),'clips':len(np.unique(arr['clip_index'][mask]))}
 return out,groups
def main():
 ctx=freeze();checkpins(ctx['pins']);write(U/'RUNNING.json',{'pid':os.getpid(),'at':now(),'training':0})
 start=time.monotonic();arrays={k:[] for k in ['clip_index','frame_id','timestamp_us','side','states','counts','sensor_wrist_outside','sensor_raw_count','sensor_assigned','sensor_unassigned']};verified=Counter();clipstats=[];examples=[]
 try:
  for ri,r in enumerate(ctx['rows']):
   w,h=r['box'][2:];allargs=[];saved=[];data=[];states=[];counts=[]
   for v in ('masked','sensor'):
    obs,raw,ass=(load(r['views'][v][k]) for k in ('observation','raw','assignment'));q=load(r['state_views'][v]);saved.append(q);data.append((obs,raw,ass))
    assert np.array_equal(obs['frame_id'],raw['frame_id']) and np.array_equal(obs['timestamp_us'],raw['timestamp_us'])
    a=args(obs,raw,ass,v,w,h);allargs.append(a);st,nc=position(a,w,h);states.append(st);counts.append(nc)
    assert np.array_equal(st,q['full_states'][:,:,0]),(r['clip_id'],v,'full')
    lookup={int(f):i for i,f in enumerate(obs['frame_id'])}
    for temporal in ('whole','context'):
     ix=[lookup[int(f)] for f in q[temporal+'_frame_id']]
     assert np.array_equal(st[ix],q['states_'+temporal][:,:,0]),(r['clip_id'],v,temporal)
     verified[v+'_'+temporal]+=len(ix)*2
    verified[v+'_full']+=len(st)*2
   assert np.array_equal(data[0][0]['frame_id'],data[1][0]['frame_id'])
   assert np.array_equal(data[0][0]['timestamp_us'],data[1][0]['timestamp_us'])
   sensor=allargs[1];sobs,sraw,sass=data[1];canonical=args(sobs,sraw,sass,'masked',w,h)
   a=[x.copy() for x in sensor];wrist=sensor[0][:,[15,16]];outside=(~np.isfinite(wrist).all(2))|(wrist[:,:,0]<0)|(wrist[:,:,0]>w)|(wrist[:,:,1]<0)|(wrist[:,:,1]>h)
   for s in (0,1):a[2][outside[:,s],15+s]=0
   variants=[allargs[0],sensor,tuple(a),(sensor[0],sensor[1],sensor[2],canonical[3],canonical[4]),(canonical[0],canonical[1],canonical[2],sensor[3],sensor[4]),canonical]
   vst=[position(a,w,h)[0] for a in variants]
   frames=np.intersect1d(saved[0]['whole_frame_id'],saved[1]['whole_frame_id']);ix=np.array([lookup[int(f)] for f in frames]);assert len(ix)
   n=len(ix)*2
   add={'clip_index':np.full(n,ri,np.int16),'frame_id':np.repeat(frames,2),'timestamp_us':np.repeat(sobs['timestamp_us'][ix],2),'side':np.tile([0,1],len(ix)).astype(np.int8),
    'states':np.stack([st[ix].reshape(-1) for st in vst],1),'counts':np.stack([nc[ix].reshape(-1) for nc in counts],1),
    'sensor_wrist_outside':outside[ix].reshape(-1),'sensor_raw_count':np.repeat(sraw['raw_count'][ix],2),
    'sensor_assigned':sensor[3][ix].reshape(-1),'sensor_unassigned':np.repeat(sensor[4][ix],2)}
   for k,v in add.items():arrays[k].append(v)
   clipstats.append({'clip_id':r['clip_id'],'label':r['label'],'signer_id':r['signer_id'],'common_hand_frames':n,'mismatch':int((add['states'][:,0]!=add['states'][:,1]).sum())})
   if (ri+1)%100==0:print('Audited',ri+1,'/',len(ctx['rows']),flush=True)
  arr={k:np.concatenate(v) for k,v in arrays.items()};summary,groups=summarize(arr)
  for key,mask in groups.items():
   for ri in np.unique(arr['clip_index'][mask])[:2]:
    eligible=np.flatnonzero(mask&(arr['clip_index']==ri));runs=[]
    for side in (0,1):
     ss=eligible[arr['side'][eligible]==side]
     if len(ss):
      splits=np.flatnonzero(np.diff(arr['frame_id'][ss])!=1)+1;runs.extend(np.split(ss,splits))
    runs.sort(key=lambda x:(-len(x),int(arr['frame_id'][x[0]]),int(arr['side'][x[0]])));run=runs[0];j=int(run[len(run)//2]);r=ctx['rows'][int(ri)]
    examples.append({'stratum':key,'clip_id':r['clip_id'],'label':r['label'],'signer_id':r['signer_id'],'frame_id':int(arr['frame_id'][j]),'timestamp_us':int(arr['timestamp_us'][j]),'side':int(arr['side'][j]),'run_hand_frames':len(run),'states':{v:NAMES[int(arr['states'][j,k])] for k,v in enumerate(VARIANTS)},'visible_hand_counts':arr['counts'][j].tolist(),'source':r['source'],'box':r['box']})
  np.savez_compressed(U/'PAIRED_TRAIN.npz',**arr)
  write(U/'SUMMARY.json',{'at':now(),**summary,'independent_position_reconstruction':dict(verified),'time_seconds':time.monotonic()-start,'statistic_unit':'correlated hand-frames, not independent videos or recognition accuracy','history_note':'Position collapses inside_missing and overlap_likely; full/whole/context saved labels are exactly reconstructed statelessly. History may change proximity, not these six position labels.'})
  write(U/'CLIPS.json',clipstats);write(U/'EXAMPLES.json',examples);checkpins(ctx['pins']);(U/'RUNNING.json').unlink()
  write(U/'STATUS.json',{'stage':'AUDIT_COMPLETE_PENDING_VISUAL_AND_VERIFICATION','at':now(),'training':0});print(json.dumps(summary,ensure_ascii=False),flush=True)
 except BaseException:
  if not (U/'FIRST_ERROR.json').exists():write(U/'FIRST_ERROR.json',{'at':now(),'traceback':traceback.format_exc()})
  raise
if __name__=='__main__':main()
