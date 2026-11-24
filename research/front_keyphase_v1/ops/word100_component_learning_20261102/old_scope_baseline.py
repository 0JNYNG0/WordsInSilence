"""New-task baseline from already saved old predictions, never rerun/rescore old experiments."""
from common import *
def main():
 R=O/'word100_learned_rgb_motion_20261102';D=M/'DATASET.json';npz=R/'runs/rgb_residual_42/development_predictions.npz'
 ds=read(D);a=np.load(npz,allow_pickle=False);lookup={}
 for k,oldi in enumerate(a['old_indices']):
  r=ds['rows'][int(oldi)]
  if r['kind'] not in ['native_directed','native_stable'] or not a['mask'][k,0]:continue
  key=(r['clip_id'],tuple([r['frame_ids'][0],r['frame_ids'][-1]]))
  lookup[key]=dict(old_probability=float(a['p'][k,0]),old_y=int(a['y'][k,0]),old_source_index=int(oldi),
   old_prediction_index=k,old_times=[r['timestamps_us'][0],r['timestamps_us'][-1]])
 rows=[]
 for r in read(S/'DIRECT_TARGETS.json')['rows']:
  if r['role']!='development' or r['target']!=TEMP[1] or r['kind']!='interval' or not r['mask']:continue
  key=(r['clip_id'],tuple(r['native_frames']));b=lookup[key]
  assert b['old_y']==r['y'] and b['old_times']==r['native_timestamps_us']
  rows.append(dict(record=r,**b,new_task_equivalence='same reviewed physical interval and target0, not whole phrase or all source witnesses'))
 assert len(rows)==6
 save('OLD_EQUIVALENT_BASELINE.json',dict(rows=rows,pins={str(D):sha(D),str(npz):sha(npz)},
  native_intervals_only=True,feature_reversed_RGB_excluded=True,old_experiment_scores_unchanged=True,
  old_model_inference=False,point_equivalent_outputs_reused_unchanged=True,
  unmatched_new_components='Old method had no identical labelled output; do not invent negative predictions.'))
 print('Matched6existing native intervals; preserved old experiment scores')
if __name__=='__main__':main()
