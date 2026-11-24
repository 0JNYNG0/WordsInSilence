"""Audit actual FIT observations; never use DEV to define the new fit conditions."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
from prepare_real_fit import N,O,P,M,V,read,sha,exclusive
sys.path.insert(0,str(M));from motion_model import public_arrays,features
def main():
 rows=read(N/'INPUT_MANIFEST.json')['rows'];old={r['clip_id']:r for r in read(P/'INPUT_MANIFEST.json')['rows']};assert len(rows)==108
 assert {r['person'] for r in rows}==set(range(1,10));assert {r['role'] for r in rows}=={'fit'}
 expected={(cid,k) for cid,r in old.items() if r['role']=='fit' for k in ['original','speed_0.8','speed_1.25']};assert {(r['clip_id'],r['kind']) for r in rows}==expected
 stats=read(M/'NORMALIZATION.json');scales=np.array(stats['std']);cache={};audit=[];fitlabels=[];labels=read(M/'MOTION_TARGETS.json')['rows']
 groups={'positions':list(range(10))+list(range(28,38))+[56,57,58,61,62], 'angles_sizes_axes':list(range(10,28))+list(range(38,56)), 'shape_probabilities':list(range(63,70)), 'relation_probabilities':list(range(70,73))}
 for row in rows:
  r=old[row['clip_id']];assert r['role']=='fit';assert sha(row['input'])==row['analysis_sha256'] and sha(row['path'])==row['video_sha256'] and sha(r['source'])==r['source_sha256']
  a=dict(np.load(row['input']));mapping=np.array(row['source_frame_indices']);assert len(a['frame_id'])==len(mapping) and np.all(np.diff(a['timestamp_us'])>0)
  assert np.array_equal(a['frame_id'],np.arange(len(mapping)))
  if r['clip_id'] not in cache:cache[r['clip_id']]=public_arrays(r)
  olda,oz,om=cache[r['clip_id']];z,mask=features(a,a['retained_probability'],a['relation_probability']);assert (mapping< len(oz)).all()
  known=[x for l in labels if l['clip_id']==r['clip_id'] for x in range(l['frames'][0],l['frames'][1]+1)];core=np.isin(mapping,known);common=mask&om[mapping]&core[:,None]
  deltas=np.abs(z-oz[mapping])/scales;drift={}
  for key,j in groups.items():
   valid=common[:,j];d=deltas[:,j][valid];drift[key]=dict(common_cells=int(valid.sum()),median_standardized_delta=float(np.median(d)) if len(d) else None,p90_standardized_delta=float(np.percentile(d,90)) if len(d) else None)
  audit.append(dict(clip_id=r['clip_id'],kind=row['kind'],person=r['person'],drift=drift,core_frames=int(core.sum()),mask_coverage=float(mask[core].mean()),old_coverage=float(om[mapping][core].mean()),rgb_frame_crop=r['box'],runtime_sensor=read(row['result'])['actual_sensor_extent']))
  # Only exact native labeled frames, not nearest-frame labels copied across transitions.
  for l in r['labels']:
   qi=np.flatnonzero(mapping==l['center_frame'])
   if not len(qi):continue
   q=int(qi[0]);p=a['retained_probability'][q]
   for j,(y,valid) in enumerate(zip(l['y'],l['target_mask'])):
    if valid and np.isfinite(p[j]):fitlabels.append(dict(clip_id=r['clip_id'],kind=row['kind'],target=j,frame=l['center_frame'],truth=int(y>.5),predicted=int(p[j]>=.5),probability=float(p[j])))
 for path,h in read(P/'INPUT_MANIFEST.json')['source_pins'].items():assert sha(path)==h
 report=dict(status='PASS_PROVENANCE_TIME_MASKS_NOT_PERFORMANCE_CLAIM',videos=108,unique_sources=36,people=list(range(1,10)),rows=audit,
  shape_label_checks=fitlabels,private_data_access=False,original_and_model_hashes_preserved=True,
  framing_note='Saved older RGB features used per-clip boxes; current live extraction uses whole sensor. Geometry/current image and speed effects must be separated; adding data alone is not proof of better feedback.')
 exclusive(N/'INPUT_AUDIT.json',report)
 compact={}
 for kind in ['original','speed_0.8','speed_1.25']:
  q=[r for r in audit if r['kind']==kind];checks=[r for r in fitlabels if r['kind']==kind]
  compact[kind]=dict(drift={k:float(np.median([r['drift'][k]['median_standardized_delta'] for r in q])) for k in groups},fit_visible_labels=len(checks),fit_visible_correct=sum(r['truth']==r['predicted'] for r in checks))
 exclusive(N/'INPUT_AUDIT_SUMMARY.json',compact);print(json.dumps(compact,ensure_ascii=False))
if __name__=='__main__':main()
