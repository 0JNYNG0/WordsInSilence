from common import *
def main():
 d=read(U/'ORDERED_RESULT.json');errors=[];static=[]
 for s in d['rows']:
  r=s['record']
  if r['role']!='development' or not s['mask']:continue
  if s['variant']=='native' and s['status']=='CONTRARY_COMPONENT_CANDIDATE' and s['y']==1:
   errors.append(dict(clip_id=r['clip_id'],kind=r['kind'],target=r['target'],frames=s['native_extent'],probability=s['probability'],geometry_coverage=s['geometry_coverage']))
  if s['variant']=='still' and s['class_prediction']==1:static.append(dict(clip_id=r['clip_id'],target=r['target'],probability=s['probability']))
 rows,meta,arrays=load();stats=read(U/'NORMALIZATION.json')
 controls=build_temporal(rows,arrays,stats)
 checks=[]
 for s in controls:
  if s['variant']!='still':continue
  checks.append(bool(np.all(s['x']==s['x'][0])))
 assert all(checks)
 save('DIAGNOSIS.json',dict(native_confident_errors=errors,unique_native_error_clips=len({r['clip_id'] for r in errors}),
  static_wrong=static,all_still_inputs_exactly_constant=all(checks),
  evidence='A constant coordinate/mask sequence can still receive movement probability above.5. The network is not required to ground a movement decision in actual within-sequence changes.',
  inference_not_proven_cause='Posture/style or recurrent state can dominate learned motion. Relative displacement and adjacent-step evidence are a targeted hypothesis, not proof of a fix.',
  next='Matched copied output-head adaptation versus output-head plus zero-initialized relative-motion branch, same inputs/labels/80epochs/seed42; preserve original outputs and saved comparison.',
  no_new_model_inference=True))
 print('Errors',len(errors),'unique clips',len({r['clip_id'] for r in errors}),'constant controls falsely moving',len(static))
if __name__=='__main__':main()
