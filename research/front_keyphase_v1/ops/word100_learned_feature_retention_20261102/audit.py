"""Stored-public prediction diagnosis before choosing a training correction."""
from pathlib import Path
import json,hashlib
from collections import Counter
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent;P=O/'word100_learned_guide_feedback_20261102';OLD=O/'word100_multipeople_live_feedback_20261101'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(name,d):
 p=U/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2)
def main():
 assert not (U/'DIAGNOSIS.json').exists();m=read(P/'INPUT_MANIFEST.json');rows=m['rows'];targets=m['targets'];out=[]
 for r in rows:
  if r['role']!='development':continue
  old=np.load(OLD/'baseline_probabilities'/f"{r['clip_id']}.npz")
  for condition in ['frozen_projection','projection_finetune']:
   for seed in [42,43,44]:
    new=np.load(P/'runs'/f'{condition}_{seed}'/'predictions'/f"{r['clip_id']}.npz")
    for label in r['labels']:
     oi=int(np.flatnonzero(old['frame_id']==label['center_frame'])[0]);ni=int(np.flatnonzero(new['frame_id']==label['center_frame'])[0])
     for j in range(7):
      if not label['target_mask'][j]:continue
      gt=int(label['y'][j]>.5);a=float(old['probabilities'][oi,j]);b=float(new['probability'][ni,j])
      if int(b>=.5)!=gt or int(a>=.5)!=gt:
       out.append(dict(clip_id=r['clip_id'],person=r['person'],word=r['word_id'],frame=label['center_frame'],target=targets[j],truth=gt,old_probability=a,new_probability=b,
        condition=condition,seed=seed,kind='regression' if int(a>=.5)==gt else 'new_correct' if int(b>=.5)==gt else 'both_wrong'))
 counts=[]
 for condition in ['frozen_projection','projection_finetune']:
  for seed in [42,43,44]:
   rr=[x for x in out if x['condition']==condition and x['seed']==seed];counts.append(dict(condition=condition,seed=seed,counts=dict(Counter(x['kind'] for x in rr)),by_target=dict(Counter(x['target'] for x in rr if x['kind']=='regression'))))
 support={}
 for role in ['fit','development']:
  ls=[l for r in rows if r['role']==role for l in r['labels']]
  y=np.asarray([l['y'] for l in ls]);mask=np.asarray([l['target_mask'] for l in ls])
  support[role]=[dict(target=t,positive=int(((y[:,j]>.5)&mask[:,j]).sum()),negative=int(((y[:,j]<=.5)&mask[:,j]).sum())) for j,t in enumerate(targets)]
 # Frozen teacher contradictions are recorded, never promoted to human truth.
 teacher=[]
 for r in rows:
  if r['role']!='fit':continue
  a=np.load(OLD/'bank_predictions'/f"{r['clip_id']}.npz")
  for l in r['labels']:
   i=int(np.flatnonzero(a['frame_id']==l['center_frame'])[0]);prob=a['probabilities'][i]
   for j in range(7):
    if l['target_mask'][j] and (prob[j]>=.5)!=(l['y'][j]>.5):teacher.append(dict(clip_id=r['clip_id'],frame=l['center_frame'],target=targets[j],truth=l['y'][j],teacher_probability=float(prob[j])))
 save('DIAGNOSIS.json',dict(status='SAVED_PUBLIC_ERRORS_AND_SUPPORT_AUDITED',errors=out,counts=counts,support=support,teacher_conflicts_on_fit=teacher,
  previous_turn_classification='progress: trained six models, built and verified isolated runtime, discovered feature regression and short omission limit',
  new_inference=0,new_training=0,private_inputs=0,
  inference='Global multi-task representation has changed previously correct physical feature decisions. Shared temporal compression/multitask supervision is a plausible mechanism, not proven causality. Existing protected local branch can be retained without adding another RGB backbone.',
  no_per_word_fix=True))
 print(json.dumps(dict(counts=counts,support=support,teacher_conflicts=len(teacher)),ensure_ascii=False))
if __name__=='__main__':main()
