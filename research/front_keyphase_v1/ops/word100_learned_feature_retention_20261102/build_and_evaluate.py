"""FIT-only recalibration; saved DEV comparison without new model inference."""
import sys,json
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from audit import U,O,P,OLD,read,save
from retention_engine import RetentionEngine
def source(r,seed):
 a=dict(np.load(r['input']));p=dict(np.load(P/'runs'/f'projection_finetune_{seed}'/'predictions'/f"{r['clip_id']}.npz"))
 q=np.load(OLD/('bank_predictions' if r['role']=='fit' else 'baseline_probabilities')/f"{r['clip_id']}.npz")
 assert np.array_equal(a['frame_id'],q['frame_id']) and np.array_equal(a['timestamp_us'],q['timestamp_us'])
 p['retained_probability']=q['probabilities'];return a,p
def main():
 assert not (U/'RETAINED_COMPARISON.json').exists();rows=read(P/'INPUT_MANIFEST.json')['rows'];out=[]
 for seed in [42,43,44]:
  e=RetentionEngine(seed,False);values=defaultdict(list)
  for r in rows:
   if r['role']!='fit':continue
   a,p=source(r,seed);result=e.compare(p,a,r['word_id'],True)
   for s in result['stages']:values[s['guide_id']].append(s)
  old_norm=read(P/'calibration'/f'projection_finetune_{seed}.json');norm=json.loads(json.dumps(old_norm))
  for gid,n in norm['stages'].items():
   for f in n['supported_attributes']:
    j=f['target_index'];v=[abs(s['attribute_mean'][j]-f['expected']) for s in values[gid] if s['attribute_mean'][j] is not None and s['posterior_coverage'][j]>=.8]
    f['distance_limit']=max(.25,float(np.quantile(v,.95))+.05) if v else 1.;f['posterior_supported_fit_clips']=len(v)
  norm.update(observer='retained_protected',fit_only=True,selected_using_development=False);save(f'calibration/{seed}.json',norm);e.retained_norm=norm
  for r in rows:
   if r['role']!='development':continue
   a,p=source(r,seed);result=e.compare(p,a,r['word_id']);path=f"cases/{seed}_{r['clip_id']}.json";save(path,result)
   old=read(P/'feedback_v2'/f"projection_finetune_{seed}_{r['clip_id']}_normal.json")
   assert result['path']==old['path'],'phase path must remain exact'
   out.append(dict(seed=seed,clip_id=r['clip_id'],statuses=dict(Counter(s['status'] for s in result['stages'])),path=str(U/path),phase_path_unchanged=True,
    source_probability_is_protected=True,whole_sign_correctness=result['whole_sign_correctness']))
 save('RETAINED_COMPARISON.json',dict(status='RETAINED_OBSERVER_LEARNED_FLOW_PUBLIC_COMPARISON',rows=out,new_inference=0,new_training=0,
  same_physical_attribute_scores_as_protected_by_array_identity=True,protected_macro_balanced_accuracy=read(P/'SAVED_VERIFICATION.json')['old_protected']['macro_balanced_accuracy'],
  phase_paths_identical=True,not_a_new_accuracy_gain=True,weak_stage_or_linguistic_limits_not_resolved=True))
 print('36 public clip-seed comparisons; protected features and learned DTW paths preserved')
if __name__=='__main__':main()
