from common import *
import numpy as np
def basic(rows):
 return dict(total=len(rows),top1=sum(r['top1_correct'] for r in rows),top3=sum(r['top3_correct'] for r in rows),
  by_person={str(p):dict(total=sum(r['person']==p for r in rows),top1=sum(r['person']==p and r['top1_correct'] for r in rows),top3=sum(r['person']==p and r['top3_correct'] for r in rows)) for p in [11,12,13]})
def main():
 assert not (U/'TCN_COMPARISON.json').exists();new=read(U/'TCN_TRAINING_COMPLETE.json');old=[];raw={}
 for name,path in [('C',O/'word100_body_boundary_20261017/training_C_20261013_RESULT.json'),('T3-F',O/'word100_frozen_c_temporal_20261019/T3F_20261013_RESULT.json')]:
  d=read(path)
  for view,pred in d['predictions'].items():
   rows=[]
   for p in pred:
    logits=p.get('logits');ranks=[] if logits is None else np.argsort(-np.asarray(logits),kind='stable')[:3].tolist()
    rows.append(dict(clip_id=p['clip_id'],person=p['signer_id'],y=p['label_index'],top1_correct=bool(ranks and ranks[0]==p['label_index']),top3_correct=p['label_index'] in ranks))
   raw[(name,view)]={r['clip_id']:r for r in rows};old.append(dict(condition=name,view=view,seed=20261013,metrics=basic(rows),source=str(path),sha256=sha(path)))
 results=[]
 for run in new['runs']:
  for view,path in run['prediction_paths'].items():
   rows=read(path);assert len(rows)==300 and len({r['clip_id'] for r in rows})==300
   paired=[]
   # Only upper_mask matches the old masked coordinate boundary; sensor is a different pixel crop.
   if view=='upper_mask':
    for baseline in ['C','T3-F']:
     before=raw[(baseline,'masked')];assert set(before)=={r['clip_id'] for r in rows}
     gains=[r['clip_id'] for r in rows if r['top1_correct'] and not before[r['clip_id']]['top1_correct']]
     losses=[r['clip_id'] for r in rows if not r['top1_correct'] and before[r['clip_id']]['top1_correct']]
     paired.append(dict(baseline=baseline,improved=gains,regressed=losses,net=len(gains)-len(losses)))
   results.append(dict(condition=run['condition'],seed=run['seed'],view=view,metrics=basic(rows),paired=paired,predictions=path,sha256=sha(path)))
 grouped={}
 for condition in ['fresh','finetune_C']:
  grouped[condition]={}
  for view in ['full','upper_mask']:
   cc=[r for r in results if r['condition']==condition and r['view']==view]
   grouped[condition][view]=dict(top1=[r['metrics']['top1'] for r in cc],top3=[r['metrics']['top3'] for r in cc],total=300,
    top1_mean=float(np.mean([r['metrics']['top1'] for r in cc])),top3_mean=float(np.mean([r['metrics']['top3'] for r in cc])))
 write(U/'TCN_COMPARISON.json',dict(status='SAVED_OUTPUT_COMPARISON_COMPLETE',baseline=old,candidates=results,summary=grouped,
  limitations=['Previously exposed development cohort, not fresh test','Detector, training initialization and view augmentation differ: pipeline comparison, not detector-only causal effect',
   'Old masked view: full-source extraction then saved upper boundary; old sensor: cropped image extraction; new full runtime uses actual image bounds','No best seed adoption; all three seeds shown'],production_replaced=False))
 event('tcn_compared',summary=grouped)
if __name__=='__main__':main()
