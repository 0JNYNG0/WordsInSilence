"""Evaluate saved predictions only against pre-frozen visual regions."""
from pathlib import Path
import json,hashlib,numpy as np,datetime,sys
U=Path(__file__).resolve().parent; OLD=U.parent/'word100_mano_threeway_20261025'
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def measure(x,b):
 lo=np.array(b[:2]);hi=np.array(b[2:]);out=np.maximum(np.maximum(lo-x,x-hi),0)
 return {'xy':x.tolist(),'inside_region':bool(np.all(out==0)),'distance_to_region_px':float(np.linalg.norm(out)),
 'distance_to_region_center_px':float(np.linalg.norm(x-(lo+hi)/2))}
def main():
 reviewed='--reviewed' in sys.argv
 dest=U/('REVIEWED_SCORES.json' if reviewed else 'SCORES.json');assert not dest.exists()
 f=read(U/'FROZEN.json');a=read(U/('REVIEWED_REFERENCE_REGIONS.json' if reviewed else 'RGB_REFERENCE_REGIONS.json'));pin=read(U/('REVIEWED_REFERENCE_PIN.json' if reviewed else 'REFERENCE_PIN.json'));assert sha(pin['path'])==pin['sha256']
 rows=[];measures=[]
 for row,ref in zip(f['rows'],a['items']):
  assert row['id']==ref['id'];t,s=row['frame'],row['side'];z=np.load(row['comparison']);raw=np.load(OLD/f"raw_predictions/{row['clip_index']:02d}_{t:04d}_{s}.npz")
  candidates={c:(z[c+'_xy'][t,s],z[c+'_valid'][t,s]) for c in ['B','H','HT']}
  candidates['rawH']=(raw['xy_px'],np.isfinite(raw['xy_px']).all(-1))
  result={**row,'reference':ref,'conditions':{}}
  for c,(xy,valid) in candidates.items():
   points=[]
   for j,box in [(0,ref['wrist'])]+[(int(j),b) for j,b in ref['tips'].items()]:
    q={'id':row['id'],'condition':c,'point':j,'kind':'wrist' if j==0 else 'tip','region':box,'scorable':box is not None,'available':bool(valid[j])}
    if q['scorable'] and q['available']:q.update(measure(xy[j],box))
    points.append(q);measures.append(q)
   result['conditions'][c]={'any_hand':bool(valid.any()),'points':points}
  records=read(Path(row['comparison']).with_suffix('.json'))
  result['gate']=next(r for r in records if r['frame']==t and r['side']==s)
  rows.append(result)
 summary={}
 for c in ['B','rawH','H','HT']:
  summary[c]={}
  for k in ['wrist','tip']:
   pts=[p for p in measures if p['condition']==c and p['kind']==k and p['scorable']];got=[p for p in pts if p['available']]
   summary[c][k]={'visual_regions':len(pts),'predictions_available':len(got),'inside_visual_region':sum(p['inside_region'] for p in got),'outside_visual_region':sum(not p['inside_region'] for p in got),'not_output':len(pts)-len(got),
   'mean_distance_to_region_px':float(np.mean([p['distance_to_region_px'] for p in got])) if got else None,
   'mean_distance_to_center_px':float(np.mean([p['distance_to_region_center_px'] for p in got])) if got else None,
   'max_distance_to_region_px':max((p['distance_to_region_px'] for p in got),default=None)}
 preserved=0;cross=0;crossframes=0;frames=0;cliprows=[];hadd=0;htadd=0;trigger=0
 for m in f['manifest']:
  p=np.load(m['prepared']);z=np.load(OLD/'comparisons'/f"{m['clip_id']}.npz");base=z['B_valid'];expected=p['baseline_valid'];assert np.array_equal(base,expected);assert np.array_equal(z['B_xy'],p['baseline_xy'],equal_nan=True)
  for c in ['H','HT']:
   assert np.array_equal(z[c+'_xy'][base],z['B_xy'][base]);assert z[c+'_valid'][base].all()
   assert np.array_equal(z[c+'_evidence'],z['B_evidence']);assert not (z[c+'_inferred'] & z[c+'_evidence']).any()
   assert z[c+'_xy'].shape==z['B_xy'].shape
  assert np.array_equal(z['frame_id'],p['frame_id']);assert np.array_equal(z['timestamp_us'],p['timestamp_us'])
  v=int(base.sum());cr=int(base[m['crossing_frames']].sum());n=len(base)
  preserved+=v;cross+=cr;crossframes+=len(m['crossing_frames']);frames+=n
  h=int(z['H_inferred'].any(-1).sum());ht=int(z['HT_inferred'].any(-1).sum());tr=int(p['trigger'].sum());hadd+=h;htadd+=ht;trigger+=tr
  cliprows.append({'clip':m['clip_id'],'frames':n,'preserved_points':v,'crossing_points':cr,'missing':tr,'H_added':h,'HT_added':ht})
 protected=all(sha(p)==h for p,h in f['protected'].items());prior=all(sha(p)==h for p,h in f['prior_result_pins'].items());assert protected and prior
 result={'status':'APPROXIMATE_RGB_REGION_AUDIT_NOT_EXACT_GROUND_TRUTH','created':datetime.datetime.now().astimezone().isoformat(),'summary':summary,'rows':rows,
 'sample':{'clips':7,'people':4,'word_types':5,'frames':frames,'actual_missing_handframes':trigger,'episodes':16,'primary_representative_frames':16,'wrist_visual_regions':16,'tip_visual_regions':summary['rawH']['tip']['visual_regions'],'tip_slots_unverifiable':80-summary['rawH']['tip']['visual_regions']},
 'availability_all_missing':{'B':0,'H':hadd,'HT':htadd,'denominator':trigger,'not_accuracy':True},
 'preservation':{'normal_original_point_frames_unchanged':preserved,'designated_crossing_frames':crossframes,'crossing_original_points_unchanged':cross,'frame_slots_and_timestamps_retained':True,'no_inferred_points_promoted_to_observed':True,'protected_files_unchanged':len(f['protected']),'prior_files_unchanged':len(f['prior_result_pins']),'clips':cliprows},
 'limits':['Same evaluator visual uncertainty rectangles, not independent expert or exact2DGT','Only visible fingertips checked, no hidden3D or all finger-angle correctness','16 related episode centers, not57 independent accuracy samples','Reused diagnostic data and previous models, no fresh heldout generalization test','Preserved values do not mean old MediaPipe values are all correct','No recognition/DTW end-to-end improvement measured'],
 'training':0,'new_inference':0,'production_changes':0,'user_recording_access':0}
 result['annotation_review_applied']=reviewed
 result['reference_sha256']=pin['sha256']
 write(dest,result);print(json.dumps({'summary':summary,'preservation':result['preservation'],'availability':result['availability_all_missing']},ensure_ascii=False))
if __name__=='__main__':main()
