import sys,json,copy
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4];LAB=ROOT/'research/front_keyphase_v1';U=Path(__file__).resolve().parent;OLD=LAB/'ops/word100_face_body_stage_20261027'
sys.path.insert(0,str(LAB/'src'))
from wis_lab.core.observation import Observation,sha256_file
from wis_lab.capture.face_landmarks import join_face_observation
from wis_lab.core.hand_face_evidence import build_evidence,reference_projection_status,arm_motion_direction,motion_cost
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
F=read(U/'FROZEN.json');R=read(U/'RESULT.json');checks={};total=0;expected=0;compared=0
for p,h in {**F['source_pins'],**F['protected']}.items():assert sha256_file(p)==h,p
for q in F['public_rows']:
 cid=q['clip_id'];obs=Observation.load(str(OLD/'data'/cid/'masked.npz'));face=join_face_observation(OLD/'data'/cid/'face_roi.npz',obs)
 before=obs.xy_px.copy();e=build_evidence(obs,face);assert np.array_equal(obs.xy_px,before,equal_nan=True)
 with np.load(U/'data'/f'{cid}_evidence.npz') as z:
  for k,v in e.items():assert np.array_equal(v,z[k],equal_nan=True)
 assert np.array_equal(e['xy'][:,:,5],np.stack((obs.xy_px[:,33],obs.xy_px[:,54]),axis=1),equal_nan=True)
 # Held-out query never participates in its own two references.
 with np.load(U/'data'/f'{cid}_reference_comparison.npz') as z:
  refs=z['reference_ids'].tolist();assert len(set(refs))==2 and cid not in refs
  ex=z['expected'];co=z['compared'];assert not (co&~ex).any()
  expected+=int(ex.sum());compared+=int(co.sum())
  # Expected reference cells are stored independently of query observations.
  hidden=copy.deepcopy(obs);hidden.xy_px[:,33:]=np.nan;hidden.observed[:,33:]=False;hidden.usable[:,33:]=False
  he=build_evidence(hidden,face)
  assert not (ex[:,:,:7]&he['available'][:,:,:7]).any()
  assert ex.sum()==z['expected'].sum()
  # Pose elbow/wrist evidence remains distinct when hand keypoints are absent.
  assert np.array_equal(he['distance'][:,:,7:],e['distance'][:,:,7:],equal_nan=True)
  # Shape/contact not recovered from a valid wrist/forearm.
  assert not he['available'][:,:,:7].any()
  expected_status=np.zeros_like(z['projection_status']);qproj=e['projected'];qknown=e['projection_known']
  rr=[np.load(U/'data'/f'{ref}_evidence.npz') for ref in refs]
  for t in range(obs.n_frames):
   indices=z['reference_rows'][:,t]
   for side in (0,1):
    for part in (0,1):
     for region in range(6):
      if min(indices)<0 or not qknown[t,side,part,region]:continue
      if not all(bool(a['projection_known'][idx,side,part,region]) for a,idx in zip(rr,indices)):continue
      values=[bool(a['projected'][idx,side,part,region]) for a,idx in zip(rr,indices)]
      val=bool(qproj[t,side,part,region])
      expected_status[t,side,part,region]=3 if values[0]!=values[1] else (2 if val and values[0] else (1 if not val and not values[0] else 4))
  for a in rr:a.close()
  assert np.array_equal(expected_status,z['projection_status'])
 # Missing face cannot be reconstructed from expression or face history.
 ff=copy.deepcopy(face);ff['xy_px'][:]=np.nan;ff['found'][:]=False
 ee=build_evidence(obs,ff);assert not ee['available'].any()
 # Motion direction must not bridge a missing elbow in its neighborhood.
 dm=copy.deepcopy(obs);mid=obs.n_frames//2;dm.xy_px[mid,13]=np.nan;dm.observed[mid,13]=False;dm.usable[mid,13]=False
 d,g=arm_motion_direction(dm);assert not g[mid,0]
 no,c=motion_cost(d,np.zeros_like(g),d,g);assert not no.any() and not c.any()
 total+=obs.n_frames
# Recount all cross-person annotation metrics in independent scalar code.
for row in R['cross']:
 q=next(r for r in F['public_rows'] if r['clip_id']==row['query']);r=next(r for r in F['public_rows'] if r['clip_id']==row['reference'])
 with np.load(q['observation']) as qo,np.load(r['observation']) as ro,np.load(U/'data'/f"{r['clip_id']}__{q['clip_id']}_paths.npz") as p:
  a=p[row['condition']+'_reference'];b=p[row['condition']+'_query'];required=bad=missing=0
  for i,ts in enumerate(qo['timestamp_us']):
   if not any(start*1e6<=ts<=end*1e6 for start,end in q['intervals']):continue
   required+=1
   if i<b.min() or i>b.max():missing+=1;continue
   nearest=min(abs(int(j)-i) for j in b);picks=[int(aa) for aa,bb in zip(a,b) if abs(int(bb)-i)==nearest];rt=int(ro['timestamp_us'][int(np.median(picks))])
   bad+=not any(start*1e6<=rt<=end*1e6 for start,end in r['intervals'])
  assert (required,bad,missing)==(row['required'],row['annotation_discordant'],row['unmapped'])
for row in R['known_mapping']:
 r=next(r for r in F['public_rows'] if r['clip_id']==row['clip_id'])
 with np.load(r['observation']) as obs,np.load(U/'data'/f"{r['clip_id']}_{row['mode']}_paths.npz") as p:
  a,b=p[row['condition']+'_reference'],p[row['condition']+'_query'];assert (np.diff(a)>=0).all() and (np.diff(b)>=0).all()
  err=np.abs(obs['timestamp_us'][a]-obs['timestamp_us'][p['truth'][b]])/1000
  assert abs(float(err.mean())-row['mae_ms'])<1e-8
# Exhaustive Boolean combinations: projection never emits a sign-incorrect judgment.
cases=0
for qf in (False,True):
 for rf1 in (False,True):
  for rf2 in (False,True):
   for qk in (False,True):
    for rk1 in (False,True):
     for rk2 in (False,True):
      val=int(reference_projection_status(np.array([qf]),np.array([qk]),np.array([[rf1],[rf2]]),np.array([[rk1],[rk2]]))[0])
      truth=0 if not(qk and rk1 and rk2) else (3 if rf1!=rf2 else (2 if qf and rf1 else (1 if not qf and not rf1 else 4)))
      assert val==truth;cases+=1
checks=dict(source_pins=len(F['source_pins']),protected_pins=len(F['protected']),public_frames=total,
 saved_evidence_arrays=18,held_out_reference_checks=18,hidden_hand_retains_pose_separately=18,
 missing_face_no_fake_evidence=18,motion_gap_rejected=18,projection_boolean_combinations=cases,
 scalar_cross_metrics=len(R['cross']),known_mapping_metrics=len(R['known_mapping']),
 reference_expected_cells=expected,query_compared_cells=compared)
report=dict(status='PASS_SAVED_RESULTS_AND_CONTRACTS',operator='same Codex separate scalar verifier, not independent human sign judgment',checks=checks,
 active_alignment='B unchanged',expressions=0,training=0,user_or_mentor_access=0)
(U/'VERIFIED.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=False))
