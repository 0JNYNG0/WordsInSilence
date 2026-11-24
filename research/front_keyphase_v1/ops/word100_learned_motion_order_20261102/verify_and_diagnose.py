"""Separate saved arithmetic audit plus a prespecified exact-time localization diagnostic."""
from pathlib import Path
from collections import Counter
import json,hashlib,time
import numpy as np,torch
from motion_model import M,P,U,V,read,features,normalize,predict
from motion_runtime import MotionObserver
from label_motion import save,sha
def confusion(y,p,m):
 result=[]
 for j in range(y.shape[1]):
  good=m[:,j];t=y[good,j]>.5;q=p[good,j]>=.5
  result.append([int((t&q).sum()),int((~t&~q).sum()),int((~t&q).sum()),int((t&~q).sum())])
 return result
def main():
 torch.set_num_threads(4);training=read(M/'TRAINING_RESULT.json');ds=read(M/'DATASET.json');labels=read(M/'MOTION_TARGETS.json')['rows'];manifest=read(P/'INPUT_MANIFEST.json');ix=np.array([i for i,r in enumerate(ds['rows']) if r['role']=='development']);computed=[]
 for run in training['runs']:
  d=M/'runs'/f"{run['condition']}_{run['seed']}";a=np.load(d/'predictions.npz');assert sha(d/'model.pt')==run['checkpoint_sha256']
  matrix=confusion(a['y'][ix],a['p'][ix],a['mask'][ix]);assert matrix==[[s[k] for k in ['TP','TN','FP','FN']] for s in run['development']]
  macro=np.mean([.5*(tp/(tp+fn)+tn/(tn+fp)) for tp,tn,fp,fn in matrix[:5]])
  computed.append(dict(condition=run['condition'],seed=run['seed'],five_motion_balanced_accuracy=float(macro),continuity=run['development'][5]))
 for r in manifest['rows']:
  assert sha(r['input'])==r['input_sha256']
 for path,h in manifest['source_pins'].items():assert sha(path)==h
 for r in read(M/'REVERSE_MANIFEST.json')['rows']:
  assert sha(r['path'])==r['sha256'] and sha(r['source'])==r['source_sha256'];a=np.load(M/'reverse_extracted'/r['clip_id']/'analysis.npz')
  assert len(a['frame_id'])==len(r['source_frame_indices']);assert r['source_frame_indices']==list(range(len(a['frame_id'])-1,-1,-1));assert np.all(np.diff(a['timestamp_us'])>0)
 comparisons=read(M/'COMPARISON.json');recounts={};real_scores=[]
 for cond in ['orderless_pool','ordered_gru']:
  for seed in [42,43,44]:
   key=f'{cond}_{seed}';a=np.load(M/'actual_reverse_predictions'/f'{key}.npz');rec=next(r for r in comparisons['actual_reverse_direct'] if r['condition']==cond and r['seed']==seed)
   assert confusion(a['y'],a['p'],a['mask'])==[[s[k] for k in ['TP','TN','FP','FN']] for s in rec['scores']]
   # Native reviewed intervals + corresponding actual reversed RGB intervals,
   # not cached feature reversal. Stable intervals are counted explicitly.
   old=np.load(M/'runs'/key/'predictions.npz');native=[i for i,r in enumerate(ds['rows']) if r['role']=='development' and r['kind'].startswith('native_')]
   mm=confusion(np.concatenate([old['y'][native],a['y']]),np.concatenate([old['p'][native],a['p']]),np.concatenate([old['mask'][native],a['mask']]))
   real_scores.append(dict(condition=cond,seed=seed,counts=mm,five_motion_balanced_accuracy=float(np.mean([.5*(tp/(tp+fn)+tn/(tn+fp)) for tp,tn,fp,fn in mm[:5]]))))
 for r in comparisons['rows']:
  out=read(M/'feedback'/f"{r['condition']}_{r['seed']}_{r['case']}.json")
  assert out['motion_observations']==r['motions'];assert r['old_features_relations_path_unchanged']
  if r['kind']=='main_blackout':assert r['gaps']>0 and all(m['status']=='MOTION_OBSERVATION_INSUFFICIENT' for m in r['motions'])
 for cond in ['orderless_pool','ordered_gru']:
  recounts[cond]={kind:dict(Counter(m['status'] for r in comparisons['rows'] if r['condition']==cond and r['kind']==kind for m in r['motions'])) for kind in sorted({r['kind'] for r in comparisons['rows']})}
 save('SAVED_VERIFICATION.json',dict(status='PASS_SAVED_ARITHMETIC_AND_PRESERVATION',training_scores=computed,actual_reverse_plus_native_scores=real_scores,recounts=recounts,
  shared_cases_not_independent_samples=True,all_old_features_relations_DTW_preserved=True,original_source_hashes_preserved=True,
  same_codex_separate_arithmetic_not_human_gold=True,private_recording_access=False))
 # Exact edited-source mappings give a time oracle. This is diagnosis only,
 # not deployable time labels and never used to fit or choose a checkpoint.
 diagnosis=[];controls=read(U/'VIDEO_CONTROLS.json')['rows']
 for seed in [42,43,44]:
  observer=MotionObserver('ordered_gru',seed)
  for c in controls:
   if c['kind'] not in ['reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest']:continue
   case=f"{c['word_id']}_{c['kind']}";a=dict(np.load(U/'actual_controls'/case/'analysis.npz'));rel=np.load(V/'control_predictions'/f'rgb_geometry_42_{case}.npz')['probability'];z,m=features(a,a['retained_probability'],rel);x=normalize(z,m,observer.stats);mapping=np.array(c['source_frame_indices'])
   src=[r for r in labels if r['word_id']==c['word_id'] and r['person']==12 and r['kind']=='directed'];sequences=[];intervals=[]
   for l in src:
    q=np.flatnonzero((mapping>=l['frames'][0])&(mapping<=l['frames'][1]));sequences.append(x[q]);intervals.append(q)
   pp=predict(observer.model,sequences)
   base=next(r for r in comparisons['rows'] if r['condition']=='ordered_gru' and r['seed']==seed and r['case']==case)
   for l,q,p in zip(src,intervals,pp):
    j=next(j for j in range(5) if l['mask'][j] and l['y'][j]);old=next(v for v in base['motions'] if v['target']==j)
    diagnosis.append(dict(seed=seed,word=c['word_id'],kind=c['kind'],target=j,correct_time_frames=a['frame_id'][q].tolist(),
     correct_time_probability=float(p[j]),correct_time_continuity=float(p[5]),learned_time_probability=old['probability'],learned_time_continuity=old['continuity_probability'],
     learned_time_status=old['status'],learned_time_frames=old['frame_ids'],oracle_not_for_runtime=True))
 save('EXACT_TIME_DIAGNOSIS.json',dict(rows=diagnosis,controlled_time_mapping_only=True,new_training=0,no_model_or_threshold_change=True))
 print(json.dumps(dict(actual=real_scores,recounts=recounts),ensure_ascii=False))
if __name__=='__main__':main()
