"""Separate saved-output / invariant verification. Same Codex, no independent human GT."""
import json,sys,copy
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4];LAB=ROOT/'research/front_keyphase_v1';U=Path(__file__).resolve().parent
sys.path.insert(0,str(LAB/'src'))
from wis_lab.core.observation import Observation,sha256_file
from wis_lab.capture.face_landmarks import join_face_observation
from wis_lab.core.face_body_relations import relations,compare_path,region_geometry
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
f=read(U/'FROZEN.json');result=read(U/'RESULT.json');checks={};frames=found=inside=overlap=0; failures=[]
for r in f['rows']:
 out=U/'data'/r['clip_id'];obs=Observation.load(r['observation']);m=Observation.load(str(out/'masked.npz'))
 assert sha256_file(r['observation'])==r['observation_sha256']
 assert sha256_file(r['record'])==r['record_sha256']
 for file in ('DONE.json','RECOVERED.json'):
  for p,h in read(out/file)['outputs'].items():assert sha256_file(p)==h
 x0,y0,x1,y1=r['roi'];xy=obs.xy_px
 within=(xy[:,:,0]>=x0)&(xy[:,:,0]<x1)&(xy[:,:,1]>=y0)&(xy[:,:,1]<y1)
 assert np.array_equal(m.observed,obs.observed&within)
 assert np.array_equal(m.usable,obs.usable&within)
 assert np.array_equal(m.xy_px[m.observed],obs.xy_px[m.observed],equal_nan=True)
 assert np.array_equal(m.frame_id,obs.frame_id) and np.array_equal(m.timestamp_us,obs.timestamp_us)
 inside+=int(m.observed.sum());frames+=obs.n_frames
 with np.load(out/'face.npz') as original,np.load(out/'face_roi.npz') as recovered:
  assert np.array_equal(original['frame_id'],recovered['frame_id'])
  assert np.array_equal(original['timestamp_us'],recovered['timestamp_us'])
  original_found=original['found']
  assert np.array_equal(original['xyz_normalized'][original_found],recovered['xyz_normalized'][original_found])
  assert np.isfinite(recovered['xyz_normalized'][recovered['found']]).all()
  found+=int(recovered['found'].sum())
 face=join_face_observation(out/'face_roi.npz',m)
 rel=relations(m,face)
 with np.load(out/'relations_roi.npz') as stored:
  for k,v in rel.items():assert np.array_equal(v,stored[k],equal_nan=True)
 overlap+=int(rel['projected_hand_overlap'].sum())
 # Missing query cannot shrink the reference denominator or become perfect agreement.
 n=m.n_frames;ids=np.arange(n);a=compare_path(rel,rel,ids,ids)
 absent=copy.deepcopy(m);absent.xy_px[:,33:]=np.nan;absent.usable[:,33:]=False;absent.observed[:,33:]=False
 b=compare_path(rel,relations(absent,face),ids,ids)
 assert a['reference_cells']==b['reference_cells'] and b['compared_cells']==0
 assert b['mean_distance_difference'] is None and b['status']=='INSUFFICIENT'
 # One fingertip missing must not erase neighbours' evidence.
 tip=copy.deepcopy(m);tip.xy_px[:,41]=np.nan;tip.usable[:,41]=False;tip.observed[:,41]=False
 tr=relations(tip,face)
 assert np.array_equal(tr['distances'][:,0,2:],rel['distances'][:,0,2:],equal_nan=True)
 # Face missing: do not carry previous frame landmarks forward.
 ff=copy.deepcopy(face);ff['xy_px'][:]=np.nan;ff['found'][:]=False
 fr=relations(m,ff);assert not fr['region_valid'][:,:6].any()
 # Similarity transform preserves body-relative distances. No hidden alignment to reference.
 theta=.12;rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
 transformed=copy.deepcopy(m);transformed.xy_px[:]=(m.xy_px@rot.T*1.3+np.array((70,25))).astype(np.float32)
 tf=copy.deepcopy(face);tf['xy_px']=face['xy_px']@rot.T*1.3+np.array((70,25))
 rr=relations(transformed,tf)
 assert np.allclose(rr['distances'],rel['distances'],atol=.0001,equal_nan=True)
 # Saved known-time path metrics independently recomputed, no model inference.
 with np.load(out/'known_paths.npz') as z:
  truth=z['source_mapping']
  for cond in 'ABC':
   aa,bb=z[cond+'_reference'],z[cond+'_query']
   assert (np.diff(aa)>=0).all() and (np.diff(bb)>=0).all()
   err=np.abs(obs.timestamp_us[aa]-obs.timestamp_us[truth[bb]])/1000
   row=next(x for x in result['known_mapping'] if x['clip_id']==r['clip_id'] and x['condition']==cond)
   assert abs(float(err.mean())-row['mean_mapping_error_ms'])<1e-8
   assert abs(float(np.percentile(err,95))-row['p95_mapping_error_ms'])<1e-8
 d=read(out/'DONE.json')
 if d['envelope_retained_annotated_frames']<d['annotated_frames']:
  failures.append(dict(clip_id=r['clip_id'],retained=d['envelope_retained_annotated_frames'],total=d['annotated_frames']))
checks.update(source_observation_pins=len(f['rows']),source_record_pins=len(f['rows']),
 protected_pins=len(f['protected']),frames=frames,face_found_frames=found,
 inside_roi_points_identical=inside,face_patch_overlap_warnings=overlap,
 unchanged_frame_time_paths=len(f['rows']),missing_hand_does_not_shrink_denominator=len(f['rows']),
 missing_hand_not_perfect=len(f['rows']),neighbour_finger_evidence_retained=len(f['rows']),
 missing_face_not_fabricated=len(f['rows']),translation_scale_rotation_invariance=len(f['rows']),
 saved_known_time_metrics=3*len(f['rows']),hand_crossing_coordinates='exact preservation inside ROI; not physical correctness certificate')
for p,h in f['protected'].items():assert sha256_file(p)==h
# Reject a deliberately mismatched timestamp join.
bad=copy.deepcopy(m);bad.timestamp_us+=5000
try:join_face_observation(out/'face_roi.npz',bad)
except ValueError:checks['mismatched_pts_rejected']=True
else:raise AssertionError('Bad PTS accepted')
report=dict(status='PASS_SAVED_DATA_AND_INVARIANTS',operator='Codex separate checker; not independent human ground truth',
 checks=checks,activity_envelope_annotation_losses=failures,
 exclusions='No accuracy claim, no calibrated face-region pass boundary, no proven contact or sign correctness',
 no_training=True,no_operating_change=True)
(U/'VERIFIED.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
