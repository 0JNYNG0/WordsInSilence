"""Report saved predictions by word/feature; no tuning or new inference."""
from common import *

def main():
 assert not (U/'FEEDBACK_BREAKDOWN.json').exists()
 comp=read(U/'feedback_viewfix/FEEDBACK_COMPARISON.json');f=read(U/'FROZEN.json');labels=dict(zip(f['classes'],f['labels']));out=[];errors=[]
 for condition in ['projection_finetune','pose_fresh','pose_transfer']:
  cc=[c for c in comp['cases'] if c['condition']==condition]
  for word in f['feedback']['words']:
   good=n=0;dr=[];dt=[];features={}
   for c in cc:
    for r in [r for r in c['label_rows'] if word in r['clip_id']]:
     for j,known in enumerate(r['mask']):
      if not known:continue
      target=c['attributes']['targets'][j]['target'];ok=(r['probability'][j]>=.5)==(r['y'][j]>.5);good+=ok;n+=1
      m=features.setdefault(target,dict(n=0,correct=0));m['n']+=1;m['correct']+=ok
      if not ok:errors.append(dict(condition=condition,seed=c['seed'],word=word,label=labels[word],clip_id=r['clip_id'],frame_id=r['frame_id'],target=target,truth=r['y'][j],probability=r['probability'][j]))
    dr += [r for r in c['direct_rows'] if word in r['clip_id']]
    dt += [r for r in c['dtw_rows'] if word in r['clip_id']]
   out.append(dict(condition=condition,word=word,label=labels[word],known_feature_correct=good,known_feature_n=n,direct_correct=sum(r['truth']==r['prediction'] for r in dr),dtw_correct=sum(r['truth']==r['prediction'] for r in dt),stage_n=len(dr),features=features))
  rows=[r for r in out if r['condition']==condition]
  assert sum(r['known_feature_n'] for r in rows)==194*3 and sum(r['stage_n'] for r in rows)==92*3
 write(U/'FEEDBACK_BREAKDOWN.json',dict(status='SAVED_PREDICTION_ANALYSIS',rows=out,errors=errors,source_sha256=sha(U/'feedback_viewfix/FEEDBACK_COMPARISON.json'),
  meaning='Three seeds on the same 12 DEV videos, not independent extra samples. Pointwise physical targets, not educational error correctness.',threshold_changes=False,model_changes=False))
 event('saved_feedback_breakdown_complete',rows=len(out),feature_errors=len(errors))
if __name__=='__main__':main()
