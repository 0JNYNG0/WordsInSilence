"""New-label comparison of stored outputs only. Never changes previous results."""
from pathlib import Path
import json,numpy as np
from experiment import U,read,write,sha,metrics
OLD=U.parent/'word100_stage_feedback_cycle_20261031'
def main():
 info=read(U/'DATASET.json');prior=read(OLD/'LEARNING_INPUTS.json')
 clip_ix={r['clip_id']:i for i,r in enumerate(prior['rows'])}
 shared=[k for k in ['R_index_only','R_open_four','L_open_four'] if k in info['selected_targets']]
 selected=[info['targets'].index(k) for k in shared];newcol=[info['selected_targets'].index(k) for k in shared];oldcol=[prior['targets'].index(k) for k in shared]
 rows=[(i,r) for i,r in enumerate(info['rows']) if r['clip_id'] in clip_ix];assert len(rows)==144
 points=[];cache={}
 for i,r in rows:
  oldr=prior['rows'][clip_ix[r['clip_id']]]
  if r['clip_id'] not in cache:
   data=dict(np.load(oldr['input'],allow_pickle=False));cache[r['clip_id']]={int(f):n for n,f in enumerate(data['frame_id'])}
  points.append((clip_ix[r['clip_id']],cache[r['clip_id']][r['center_frame']]))
 y=np.asarray([r['y'] for _,r in rows])[:,selected];m=np.asarray([r['target_mask'] for _,r in rows],bool)[:,selected]
 scores=[];pins={};models=[('previous_pose',OLD,'pose_capacity_matched'),('previous_pose_signrep',OLD,'pose_plus_signrep'),('new_pose',U,'pose_only'),('new_pose_signrep',U,'pose_plus_signrep')]
 for name,base,condition in models:
  for seed in [42,43,44]:
   path=base/'runs'/f'{condition}_{seed}'/'probabilities.npy';pins[str(path)]=sha(path)
   manifest=read(path.parent/('RUN.json' if base==OLD else 'COMPLETE.json'))
   assert pins[str(path)]==manifest['prediction_sha256' if base==OLD else 'probabilities_sha256']
   a=np.load(path,allow_pickle=False);assert np.isfinite(a).all() and (a>=0).all() and (a<=1).all()
   assert (a.ndim==3 and a.shape[0]==len(prior['rows']) and a.shape[2]==len(prior['targets'])) if base==OLD else a.shape==(len(info['rows']),len(info['selected_targets']))
   prob=np.stack([a[c,t,oldcol] for c,t in points]) if base==OLD else a[[i for i,_ in rows]][:,newcol]
   for role in ['fit','development']:
    ix=[j for j,(_,r) in enumerate(rows) if r['role']==role]
    scores.append(dict(model=name,seed=seed,role=role,**metrics(prob[ix],y[ix],m[ix],shared)))
 summary=[]
 for name,_,_ in models:
  for role in ['fit','development']:
   rr=[r for r in scores if r['model']==name and r['role']==role];v=[r['macro_balanced_accuracy'] for r in rr if r['macro_balanced_accuracy'] is not None]
   summary.append(dict(model=name,role=role,mean=float(np.mean(v)) if v else None,range=[min(v),max(v)] if v else None,known_cells=rr[0]['known_cells'],balanced_supported_targets=rr[0]['balanced_supported_targets']))
 write(U/'PREVIOUS_STORED_COMPARISON.json',dict(status='NEW_RGB_LABEL_COMPARISON_ON_STORED_OUTPUTS',targets=shared,rows=scores,summary=summary,pins=pins,prior_probabilities_inference_count=0,previous_result_files_overwritten=False,scope='Same existing48 clips and exact centre frames; new RGB truth, separate from old published scores. New-expression8 have no old-output comparison. Motion interval definitions differ and are excluded. Unsupported new target types remain unavailable in prior head.'))
 print(json.dumps(summary))
if __name__=='__main__':main()
