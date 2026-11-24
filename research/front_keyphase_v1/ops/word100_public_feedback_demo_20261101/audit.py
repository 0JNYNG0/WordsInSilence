from collections import defaultdict,Counter
from common import *

def score(cells):
 tp=sum(c['y']==1 and c['pred']==1 for c in cells);tn=sum(c['y']==0 and c['pred']==0 for c in cells)
 fp=sum(c['y']==0 and c['pred']==1 for c in cells);fn=sum(c['y']==1 and c['pred']==0 for c in cells)
 p=tp+fn;n=tn+fp
 return dict(known_cells=len(cells),positive=p,negative=n,tp=tp,tn=tn,fp=fp,fn=fn,
   correct=tp+tn,accuracy=(tp+tn)/len(cells) if cells else None,
   balanced_accuracy=(tp/p+tn/n)/2 if p and n else None,
   clips=len(set(c['clip_id'] for c in cells)),persons=sorted(set(c['person'] for c in cells)),
   words=len(set(c['word_id'] for c in cells)))
def main():
 assert not (U/'AUDIT.json').exists();pins()
 data=read(S/'DATASET.json');names=data['selected_targets'];rows=data['rows'];arrays={}
 records=[]
 for path in set(r['input'] for r in rows):
  with np.load(path,allow_pickle=False) as z:arrays[path]={k:z[k].copy() for k in ['mask','frame_id']}
 for seed in [42,43,44]:
  p=np.load(S/f'runs/protected_shape_split_features_{seed}/probabilities.npy',allow_pickle=False)
  assert p.shape==(288,7) and np.isfinite(p).all()
  for i,r in enumerate(rows):
   a=arrays[r['input']];ix=np.flatnonzero(a['frame_id']==r['center_frame']);assert len(ix)==1
   for j,t in enumerate(names):
    source_col=data['targets'].index(t)
    if not r['target_mask'][source_col]:continue
    gate=bool(support(a['mask'][ix],data['feature_names'],t)[0])
    records.append(dict(seed=seed,sample_id=r['sample_id'],clip_id=r['clip_id'],word_id=r['word_id'],
       person=r['person'],role=r['role'],target=t,frame_id=r['center_frame'],
       stratum='all_required_geometry_present' if gate else 'geometry_gate_failed',
       y=int(r['y'][source_col]),pred=int(p[i,j]>=.5),probability=float(p[i,j]),dataset_index=i))
 groups=[]; pooled=[]
 for seed in [42,43,44]:
  for role in sorted(set(r['role'] for r in rows)):
   for stratum in ['all_required_geometry_present','geometry_gate_failed']:
    selected=[c for c in records if c['seed']==seed and c['role']==role and c['stratum']==stratum]
    features=[]
    for t in names:
     ss=score([c for c in selected if c['target']==t]);features.append(dict(target=t,**ss))
    groups.append(dict(seed=seed,role=role,stratum=stratum,features=features))
    pooled.append(dict(seed=seed,role=role,stratum=stratum,**score(selected),
      warning='Pooled feature-cell accuracy; dependent frames, imbalanced targets, not word or educational accuracy'))
 save('AUDIT.json',dict(status='COMPLETE_NEW_STRATIFICATION_SAVED_PREDICTIONS',targets=names,groups=groups,pooled=pooled,
   records=records,dataset_rows=len(rows),dataset_clips=len(set(r['clip_id'] for r in rows)),
   excluded_unknown_per_seed=sum(not bool(v) for r in rows for v in r['target_mask']),
   source_dataset_sha256=sha(S/'DATASET.json'),new_inference=0,new_training=0,
   note='Geometry gate failure is not necessarily raw hand absence. Labels describe visible center features; no proof of whole-stage/learner-error correctness.'))
 # A new supervised run needs a corrective FIT target or input defect, not repeated DEV tuning.
 fit_new=[c for c in records if c['role']=='fit' and c['target'] in names[3:]]
 save('TRAINING_DECISION.json',dict(status='REUSE_ALREADY_SUPERVISED_CHECKPOINT_FOR_BOUNDED_DEMO',
   fit_new_feature_errors=sum(c['y']!=c['pred'] for c in fit_new),fit_new_feature_cells=len(fit_new),
   new_verified_fit_targets=0,new_input_or_target_defect_found=False,new_training=0,
   reason='Existing labeled FIT targets already used; no new corrected FIT labels. Repeating identical supervision does not establish better gap/step feedback. Development errors remain evaluation evidence, not training targets.',
   existing_training='frozen SignRep + supervised small temporal heads; seed42 fixed before audit, seeds43/44 diagnostic only',
   future_training_gate='New independently checked public FIT examples or a concrete input/target defect, with person-separated evaluation; no use of diagnostic-only labels.'))
 print(json.dumps(pooled,ensure_ascii=False))
if __name__=='__main__':main()
