"""Recompute saved metric arithmetic without rerunning tracking or scoring models."""
from pathlib import Path
import json,hashlib,math
from collections import Counter
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
f=read(U/'FROZEN.json');raw=read(U/'metrics/RAW_EDGES.json');cal=read(U/'metrics/CALIBRATION.json')
allidx=read(U/'visual_inputs/INDEX.json')['rows']+read(O/'word100_temporal_shape_context_20261027/visual_inputs/INDEX.json')['rows']
ix={r['case_id']:r for r in allidx};shared={r['clip_id']:r for r in read(O/'word100_shared_feature_bundle_20261027/FROZEN.json')['rows']}
cache={};checked_rgb=0;checked_pose=0;angle_pairs=0
for e in raw['rows']:
 r=ix[e['case_id']];cid=r['clip_id']
 if cid not in cache:
  rec=read(shared[cid]['record']);z=np.load(rec['views']['native']['observation']);cache[cid]={k:z[k].copy() for k in ('xy_px','observed','usable')}
 z=cache[cid];start=33+(0 if e['side']=='L' else 21)
 a=z['xy_px'][e['from_frame_id'],start:start+21].astype(float);b=z['xy_px'][e['to_frame_id'],start:start+21].astype(float)
 masks=[]
 for fr,p in [(e['from_frame_id'],a),(e['to_frame_id'],b)]:
  m=z['observed'][fr,start:start+21]&z['usable'][fr,start:start+21]&np.isfinite(p).all(1)
  m&=(p[:,0]>=0)&(p[:,0]<r['raw_frame_size'][0])&(p[:,1]>=0)&(p[:,1]<r['raw_frame_size'][1]);masks.append(m)
 vals=[];ids=[]
 for k,tr in enumerate(f['pose_metrics']['triplets']):
  if not all(masks[0][tr]&masks[1][tr]):continue
  angles=[]
  for p in (a,b):
   u=p[tr[0]]-p[tr[1]];v=p[tr[2]]-p[tr[1]]
   if min(np.linalg.norm(u),np.linalg.norm(v))<2:break
   angles.append(math.degrees(math.atan2(abs(u[0]*v[1]-u[1]*v[0]),float(u@v))))
  if len(angles)==2:vals.append(abs(angles[1]-angles[0]));ids.append(k)
 assert e['pose']['triplet_indices']==ids
 assert np.allclose(vals,e['pose']['absolute_angle_changes_deg'],atol=1e-6)
 assert e['pose']['valid']==(len(vals)>=8)
 if len(vals)>=8:assert abs(float(np.percentile(vals,90))-e['pose']['score'])<1e-6;checked_pose+=1
 angle_pairs+=len(vals)
 q=e['rgb']
 if q['valid']:
  p0=np.array(q['from_points_px']);p1=np.array(q['to_points_px']);M=np.array(q['partial_affine'])
  residual=np.sqrt(((np.c_[p0,np.ones(len(p0))]@M.T-p1)**2).sum(axis=1))
  assert np.allclose(residual,q['residuals_px'],atol=1e-6)
  assert abs(np.percentile(residual,90)-q['residual_p90_px'])<1e-6
  width=np.median([np.linalg.norm(p[5]-p[17]) for p in (a,b)])
  assert abs(width-q['palm_width_px'])<1e-6 and width>=10
  assert abs(np.percentile(residual,90)/width-q['score'])<1e-6
  assert len(p0)>=8 and q['affine_inlier_fraction']>=.5
  assert max(q['accepted_fb_errors_px'])<=1.5
  checked_rgb+=1
for m in ('rgb','pose'):
 scores=[e[m]['score'] for e in raw['rows'] if e['case_id'] in ('T07','T10') and e[m]['valid']]
 assert len(scores)==cal[m+'_valid']>=12 and max(scores)==cal[m+'_threshold']
assert {e['recorded_person_id'] for e in raw['rows'] if e['case_id'] in ('T07','T10')}=={1}
protected=read(O/'word100_face_relation_alignment_20261027/FROZEN.json')['protected']
assert all(sha(p)==h for p,h in protected.items())
assert all(sha(p)==h for p,h in f['pins'].items())
fresh=[e for e in raw['rows'] if e['case_id'].startswith('N')]
causes={m:dict(Counter(e[m]['reason'] for e in fresh if not e[m]['valid'])) for m in ('rgb','pose')}
d={'status':'PASS','verifier':'Codex root separate arithmetic and direct native-coordinate recomputation; metric implementation by another Codex agent, not human truth.',
 'edges_checked':len(raw['rows']),'rgb_track_residual_recomputed':checked_rgb,'pose_scores_recomputed':checked_pose,
 'angle_pairs_recomputed':angle_pairs,'calibration_source_IDs':[1],'operating_pins_unchanged':len(protected),
 'new_missing_evidence_causes':causes,'raw_sha256':sha(U/'metrics/RAW_EDGES.json'),'calibration_sha256':sha(U/'metrics/CALIBRATION.json'),
 'flow_reinference':False,'training':False}
(U/'METRICS_ROOT_VERIFY.json').write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(d,ensure_ascii=False))
