from pathlib import Path
from collections import Counter
import json,time
import numpy as np,torch
from motion_model import M,P,U,V,OLD,read,features,normalize,predict,public_arrays
from motion_runtime import MotionObserver,support
from label_motion import save
from train_motion import scores

def main():
 torch.set_num_threads(4);assert (M/'REVERSE_EXTRACTION.json').exists();assert not (M/'COMPARISON.json').exists()
 if not (M/'MOTION_GUIDE_SUPPORT.json').exists():
  save('MOTION_GUIDE_SUPPORT.json',support())
  save('FEEDBACK_CONTRACT.json',dict(status='FROZEN_BEFORE_NEW_VIDEO_MOTION_PREDICTIONS',positive=.8,continuity=.8,image_coverage=.8,feature_coverage=.5,minimum_frames=3,
   exact_learned_DTW_event_window=True,do_not_search_best_window_on_development=True,not_observed_is_not_wrong=True,old_stage_and_feature_outputs_preserved=True))
 manifest=read(P/'INPUT_MANIFEST.json')['rows'];labels=read(M/'MOTION_TARGETS.json')['rows'];cases=[];episode_cases=[]
 for r in manifest:
  if r['role']!='development':continue
  a,z,mask=public_arrays(r);s=np.load(OLD/'baseline_probabilities'/f"{r['clip_id']}.npz")['probabilities'];rel=np.load(V/'runs/rgb_geometry_42/predictions'/f"{r['clip_id']}.npz")['probability']
  base=read(V/'development_feedback'/f"rgb_geometry_42_{r['clip_id']}.json");cases.append(dict(id=r['clip_id'],kind='native',a=a,shape=s,relation=rel,base=base,source='original public cached input'))
 for r in read(U/'VIDEO_CONTROLS.json')['rows']:
  case=f"{r['word_id']}_{r['kind']}";a=dict(np.load(U/'actual_controls'/case/'analysis.npz'));rel=np.load(V/'control_predictions'/f'rgb_geometry_42_{case}.npz')['probability']
  cases.append(dict(id=case,kind=r['kind'],a=a,shape=a['retained_probability'],relation=rel,base=read(V/'control_feedback'/f'rgb_geometry_42_{case}.json'),source=r['sha256']))
 for r in read(M/'REVERSE_MANIFEST.json')['rows']:
  d=M/'reverse_extracted'/r['clip_id'];a=dict(np.load(d/'analysis.npz'));cases.append(dict(id=r['clip_id']+'_reverse',kind='actual_reverse',a=a,shape=a['retained_probability'],relation=a['relation_probability'],base=read(d/'RESULT.json'),source=r['sha256']))
  mapping=np.array(r['source_frame_indices']);z,mask=features(a,a['retained_probability'],a['relation_probability'])
  for lab in labels:
   if lab['clip_id']!=r['clip_id']:continue
   ix=np.flatnonzero((mapping>=lab['frames'][0])&(mapping<=lab['frames'][1]));y=np.array(lab['y']);m=np.array(lab['mask']);y[:5]=0
   if lab['y'][2]:y[4]=1
   if lab['y'][4]:y[2]=1
   episode_cases.append(dict(id=lab['id'],clip_id=r['clip_id'],kind=lab['kind'],ix=ix,x=z[ix],mask=mask[ix],y=y,target_mask=m,
    frame_ids=a['frame_id'][ix].tolist(),source_frame_ids=mapping[ix].tolist()))
 rows=[];direct=[]
 for cond in ['orderless_pool','ordered_gru']:
  for seed in [42,43,44]:
   observer=MotionObserver(cond,seed)
   seq=[normalize(r['x'],r['mask'],observer.stats) for r in episode_cases];p=predict(observer.model,seq);y=np.array([r['y'] for r in episode_cases]);m=np.array([r['target_mask'] for r in episode_cases]);
   dest=M/'actual_reverse_predictions';dest.mkdir(exist_ok=True);np.savez_compressed(dest/f'{cond}_{seed}.npz',p=p,y=y,mask=m)
   direct.append(dict(condition=cond,seed=seed,scores=scores(y,m,p),native_episodes=len(episode_cases)))
   for c in cases:
    out=observer.attach(c['base'],c['a'],c['shape'],c['relation']);name=f'{cond}_{seed}_{c["id"]}'
    save(f'feedback/{name}.json',out)
    unchanged=out['path']==c['base']['path'] and all(s['features']==b['features'] and s['relations']==b['relations'] for s,b in zip(out['stages'],c['base']['stages']))
    assert unchanged
    if c['kind']=='main_blackout':assert out['evidence_gaps']==c['base']['evidence_gaps'] and len(out['evidence_gaps'])>0
    rows.append(dict(condition=cond,seed=seed,case=c['id'],kind=c['kind'],word=out['word_id'],motions=out['motion_observations'],seconds=out['motion_head_seconds'],
     old_features_relations_path_unchanged=unchanged,gaps=len(out['evidence_gaps']),statuses=dict(Counter(s['status'] for s in out['stages']))))
   print(cond,seed,'reverse',direct[-1]['scores'],flush=True)
 save('ACTUAL_REVERSE_EPISODES.json',dict(rows=[{k:v for k,v in r.items() if k not in ['x','mask','y','target_mask','ix']} for r in episode_cases]))
 save('COMPARISON.json',dict(rows=rows,actual_reverse_direct=direct,old_models_preserved=True,no_training_or_threshold_selection_on_results=True,
  note='Repeated conditions/seeds share the same public clips; reverse is a physical intervention, not KSL incorrectness'))
 for cond in ['orderless_pool','ordered_gru']:
  for kind in sorted(set(r['kind'] for r in rows)):
   rr=[r for r in rows if r['condition']==cond and r['kind']==kind];print(cond,kind,dict(Counter(m['status'] for r in rr for m in r['motions'])),flush=True)
if __name__=='__main__':main()
