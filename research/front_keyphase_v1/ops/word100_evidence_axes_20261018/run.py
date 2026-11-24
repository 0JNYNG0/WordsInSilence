"""Frozen public TRAIN replay into diagnostic sidecars, never model inputs."""
import os,sys,json,hashlib,time,traceback,shutil
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone,timedelta
import numpy as np
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2];AUD=O/'word100_position_state_audit_20261018'
sys.path.insert(0,str(O/'word100_hand_evidence_training_20261018'))
from adapter import frame_inputs,POSITION,LOCATION
from axes import EvidenceAxes,OBS_STATES,ARM_STATES,ARM_REASONS
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')
def write(p,x,new=True):
 p=Path(p)
 if new and p.exists():raise FileExistsError(p)
 temp=p.with_name(p.name+'.writing');temp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(temp,p)
def pinscheck(pins):
 for p,h in pins.items():assert sha(p)==h,p
def freeze():
 old=read(AUD/'CONTEXT.json');assert read(AUD/'INDEPENDENT.json')['status']=='PASS';assert not (AUD/'RUNNING.json').exists()
 pins=dict(old['pins'])
 for p in [U/'axes.py',U/'run.py',AUD/'PAIRED_TRAIN.npz',AUD/'SUMMARY.json',AUD/'EXAMPLES.json',AUD/'EXAMPLE_REASONS.json']:pins[str(p)]=sha(p)
 c={'at':now(),'authorization':'사용자 승인: 실제손관측/팔위치추정/좌우불확실성 분리 구현 및 공개자료 검증. 기존C유지, 사용자자료미사용, 학습은결과검토후.',
  'rows':old['rows'],'pins':pins,'scope':'TRAIN899 only, both saved views, full streams; primary comparison uses prior fixed shared whole windows.',
  'parameters':'Existing visibility.py PARAMS exactly retained; no threshold fitting, word exceptions or temporal/window changes.',
  'output':'Isolated diagnostic sidecar only, no changes to C input, TCN, DTW, assignment or display in production.',
  'success_criteria':['Existing usable point counts/palm visibility and legacy positions unchanged on all public records.',
   'Assignment flags do not alter independent arm location; raw unassigned flag stays separate and is unavailable for masked view.',
   'No hidden-source point/assignment leakage; no input mutation, coordinate recovery, mask promotion, future dependence or screen-half rule.',
   'All formerly arm-inferred locations preserved, ambiguity retained; newly revealed arm evidence reported as inference, not corrected hand truth.',
   'Independent recomputation and representative RGB evidence review; report observation/arm contradictions instead of silently resolving them.'],
  'training':0,'model_inference':0,'user_mentor_heldout_dev_reads':0}
 write(U/'CONTEXT.json',c);(U/'before').mkdir();(U/'diagnostics').mkdir()
 for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:shutil.copy2(p,U/'before'/p.name)
 return c
def pack(rows):
 return {'observation_count':np.array([[q['observation']['usable_point_count'] for q in r] for r in rows],np.uint8),
 'observation_state':np.array([[OBS_STATES.index(q['observation']['state']) for q in r] for r in rows],np.int8),
 'palm5':np.array([[q['observation']['palm5_observed'] for q in r] for r in rows],bool),
 'all21':np.array([[q['observation']['all21_observed'] for q in r] for r in rows],bool),
 'arm_state':np.array([[ARM_STATES.index(q['arm_location']['state']) for q in r] for r in rows],np.int8),
 'arm_reason':np.array([[ARM_REASONS.index(q['arm_location']['reason']) for q in r] for r in rows],np.int8),
 'arm_outside_estimate':np.array([[q['arm_location']['uses_outside_frame_estimate'] for q in r] for r in rows],bool),
 'assignment_available':np.array([[q['assignment']['unassigned_candidate_exists'] is not None for q in r] for r in rows],bool),
 'unassigned_candidate':np.array([[bool(q['assignment']['unassigned_candidate_exists']) for q in r] for r in rows],bool),
 'assigned_side':np.array([[q['assignment']['assigned_evidence_for_side'] for q in r] for r in rows],bool),
 'legacy_position':np.array([[POSITION.index(LOCATION[q['legacy']['state']]) for q in r] for r in rows],np.int8)}
def replay(obs,raw,ass,view,w,h):
 for d in (obs,raw,ass):
  for a in d.values():a.flags.writeable=False
 stream=EvidenceAxes();out=[]
 for i,t in enumerate(obs['timestamp_us']):
  a=frame_inputs(obs,raw,ass,i,view,w,h)
  for x in a[:3]:x.flags.writeable=False
  out.append(stream.step(*a[:3],int(t),w,h,assigned=a[3],unassigned_raw=a[4],assignment_evidence_available=view=='sensor'))
 return out
def main():
 c=freeze();pinscheck(c['pins']);write(U/'RUNNING.json',{'at':now(),'pid':os.getpid(),'training':0});started=time.monotonic();total=Counter();reused=load(AUD/'PAIRED_TRAIN.npz');paired={k:reused[k] for k in ('clip_index','frame_id','timestamp_us','side')};new={k:[] for k in ('legacy','count','palm5','all21','arm','assignment_available','unassigned')};examples=read(AUD/'EXAMPLES.json');example_out=[];outputs={};normal_conflicts=[]
 try:
  for ri,r in enumerate(c['rows']):
   assert r['role']=='train_candidate' and 1<=r['signer_id']<=9
   w,h=r['box'][2:];curr=[];primary=np.flatnonzero(reused['clip_index']==ri)[::2]
   for vi,view in enumerate(('masked','sensor')):
    obs,raw,ass=[load(r['views'][view][k]) for k in ('observation','raw','assignment')];rows=replay(obs,raw,ass,view,w,h);p=pack(rows);old=load(r['state_views'][view]);assert np.array_equal(p['legacy_position'],old['full_states'][:,:,0])
    visible=obs['usable']&np.isfinite(obs['xy_px']).all(2)&(obs['xy_px'][:,:,0]>=0)&(obs['xy_px'][:,:,0]<=w)&(obs['xy_px'][:,:,1]>=0)&(obs['xy_px'][:,:,1]<=h)
    nc=np.stack([visible[:,33+s*21:54+s*21].sum(1) for s in (0,1)],1)
    assert np.array_equal(p['observation_count'],nc);assert np.array_equal(p['palm5'],p['legacy_position']==0)
    oldarm=p['legacy_position'];expected=np.where(oldarm==2,1,np.where(oldarm==3,2,0));flag=(oldarm==2)|(oldarm==3)
    assert np.array_equal(expected[flag],p['arm_state'][flag])
    total[view+'_hand_frames']+=nc.size;total['all_observed_point_counts_preserved']+=nc.size;total['legacy_positions_preserved']+=nc.size;total['old_arm_locations_preserved']+=int(flag.sum());total['normal_palm_observations_preserved']+=int(p['palm5'].sum())
    lookup={int(f):i for i,f in enumerate(obs['frame_id'])};ix=np.array([lookup[int(f)] for f in reused['frame_id'][primary]]);curr.append({k:a[ix].reshape(-1) for k,a in p.items()})
    for e in examples:
     if e['clip_id']==r['clip_id']:
      i=lookup[e['frame_id']];example_out.append({'example':examples.index(e)+1,'clip_id':r['clip_id'],'label':r['label'],'view':view,'frame_id':e['frame_id'],'side':e['side'],'output':rows[i][e['side']]})
    conflict=p['palm5']&(p['arm_state']==2)
    if conflict.any():
     for i,s in np.argwhere(conflict)[:1]:normal_conflicts.append({'clip_id':r['clip_id'],'view':view,'frame_id':int(obs['frame_id'][i]),'side':int(s),'output':rows[i][s]})
    # Each result is diagnostic-only; original arrays are not serialized again.
    dest=U/'diagnostics'/f"{r['clip_id']}_{view}.npz";assert not dest.exists();np.savez_compressed(dest,frame_id=obs['frame_id'],timestamp_us=obs['timestamp_us'],**p);outputs[str(dest)]=sha(dest)
   for outkey,inkey in [('legacy','legacy_position'),('count','observation_count'),('palm5','palm5'),('all21','all21'),('arm','arm_state'),('assignment_available','assignment_available'),('unassigned','unassigned_candidate')]:new[outkey].append(np.stack([d[inkey] for d in curr],1))
   if (ri+1)%100==0:print('Separated public clips',ri+1,'/899',flush=True)
  paired.update({k:np.concatenate(v) for k,v in new.items()});assert np.array_equal(paired['legacy'],reused['states'][:,:2]);assert np.array_equal(paired['count'],reused['counts'])
  target=U/'PAIRED_DIAGNOSTICS.npz';np.savez_compressed(target,**paired);outputs[str(target)]=sha(target)
  rows=[]
  for vi,view in enumerate(('masked','sensor')):
   old=paired['legacy'][:,vi];arm=paired['arm'][:,vi];palm=paired['palm5'][:,vi];unknown=old==4
   rows.append({'view':view,'n':len(old),'legacy_preserved':int(len(old)),'old_assignment_uncertain':int(unknown.sum()),'arm_now_visible_under_assignment_uncertainty':np.bincount(arm[unknown],minlength=3).tolist(),
    'palm_observed':int(palm.sum()),'all21_observed':int(paired['all21'][:,vi].sum()),'palm5_not_all21':int((palm&~paired['all21'][:,vi]).sum()),
    'observed_palm_with_arm_boundary_disagreement':int((palm&(arm==2)).sum()),'arm_known_while_no_actual_hand_points':int(((paired['count'][:,vi]==0)&(arm>0)).sum()),
    'assignment_known_hand_frames':int(paired['assignment_available'][:,vi].sum())})
  write(U/'SUMMARY.json',{'at':now(),'status':'IMPLEMENTED_PENDING_INDEPENDENT_VERIFY','train_videos':899,'full_stream_checks':dict(total),'primary_comparison':rows,'seconds':time.monotonic()-started,'training':0,'observation_or_identity_accuracy_measured':False})
  write(U/'EXAMPLES.json',example_out);write(U/'OBSERVED_ARM_DISAGREEMENTS.json',normal_conflicts);write(U/'OUTPUT_PINS.json',outputs);pinscheck(c['pins']);(U/'RUNNING.json').unlink();print(json.dumps(rows),flush=True)
 except BaseException:
  if not (U/'FIRST_ERROR.json').exists():write(U/'FIRST_ERROR.json',{'at':now(),'traceback':traceback.format_exc()})
  raise
if __name__=='__main__':main()
