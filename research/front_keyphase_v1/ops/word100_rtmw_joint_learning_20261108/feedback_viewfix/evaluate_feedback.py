"""Saved-output comparisons on the same reviewed physical targets and anchors."""
from common import *
import numpy as np,torch
from train_feedback import metrics
sys.path.insert(0,str(O/'word100_learned_guide_feedback_20261102'))
from alignment import dtw,representative,stage_at

def score(rows):
 return dict(n=len(rows),answered=sum(r['prediction'] is not None for r in rows),correct=sum(r['prediction']==r['truth'] for r in rows),
  same_configuration_correct=sum(r['prediction'] in r['allowed'] for r in rows))
def main():
 assert not (U/'FEEDBACK_COMPARISON.json').exists();reports=read(U/'FEEDBACK_TRAINING_COMPLETE.json')['runs']
 old=O/'word100_learned_guide_feedback_20261102';manifest=read(old/'INPUT_MANIFEST.json');targets=manifest['targets'];dev=[r for r in manifest['rows'] if r['role']=='development']
 anchors=read(old/'BLIND_STAGE_ANCHORS.json')['rows'];guides=read(old/'GUIDE_SUPPORT.json')['guides'];ids=[g['canonical_stage_id'] for g in guides];words=read(U/'FROZEN.json')['feedback']['words']
 oldruns=read(old/'TRAINING_REPORT.json')['runs'];cases=[]
 for report in [r for r in oldruns if r['condition']=='projection_finetune']+reports:
  condition=report['condition'];seed=report['seed'];preds={r['clip_id']:r for r in report['predictions']};ys=[];ps=[];ms=[];labelrows=[]
  for r in dev:
   p=preds[r['clip_id']];assert sha(p['path'])==p['sha256'];a=np.load(p['path'],allow_pickle=False)
   for l in r['labels']:
    ix=np.flatnonzero(a['frame_id']==l['center_frame']);assert len(ix)==1;ix=int(ix[0]);ys.append(l['y']);ps.append(a['probability'][ix]);ms.append(l['target_mask'])
    labelrows.append(dict(clip_id=r['clip_id'],person=r['person'],frame_id=l['center_frame'],y=l['y'],mask=l['target_mask'],probability=a['probability'][ix].tolist()))
  met=metrics(np.asarray(ys),np.asarray(ps),np.asarray(ms),targets)
  d=(old/'runs' if condition=='projection_finetune' else U/'feedback_runs')/f'{condition}_{seed}'
  if condition=='projection_finetune':
   sr=next(r for r in read(old/'STAGE_EVALUATION.json')['candidates'] if r['condition']==condition and r['seed']==seed)
   direct=sr['direct_rows'];aligned=sr['dtw_rows'];paths=sr['paths']
  else:
   state=torch.load(d/'model.pt',map_location='cpu',weights_only=True);protos=state['prototype'].numpy();lengths=state['prototype_lengths'].numpy();direct=[];aligned=[];paths={}
   for r in dev:
    cid=r['clip_id'];a=np.load(preds[cid]['path'],allow_pickle=False);wi=words.index(r['word_id']);al=dtw(a['latent'],protos[wi,:lengths[wi]])
    paths[cid]=dict(path=al['path'].tolist(),mean_cost=al['mean_cost'])
    for anchor in [v for v in anchors if v['clip_id']==cid]:
     ix=int(np.flatnonzero(a['frame_id']==anchor['frame_id'])[0]);b=dict(clip_id=cid,frame_id=anchor['frame_id'],truth=anchor['stage_id'],allowed=anchor['allowed_same_physical_configuration'])
     direct.append(dict(b,prediction=ids[int(a['stage_probability'][ix].argmax())]));j=representative(al['path'],ix)
     aligned.append(dict(b,prediction=stage_at(j,guides,r['word_id']),canonical_index=j))
  case=dict(condition=condition,seed=seed,attributes=met,label_rows=labelrows,direct_stage=score(direct),dtw_stage=score(aligned),direct_rows=direct,dtw_rows=aligned,paths=paths)
  cases.append(case)
 grouped={}
 for condition in ['projection_finetune','pose_fresh','pose_transfer']:
  cc=[r for r in cases if r['condition']==condition];grouped[condition]=dict(seeds=[r['seed'] for r in cc],
   attribute_balanced_mean=float(np.mean([r['attributes']['macro_balanced_accuracy'] for r in cc])),
   attribute_balanced_range=[float(min(r['attributes']['macro_balanced_accuracy'] for r in cc)),float(max(r['attributes']['macro_balanced_accuracy'] for r in cc))],
   direct_stage_correct=[r['direct_stage']['correct'] for r in cc],dtw_stage_correct=[r['dtw_stage']['correct'] for r in cc],stage_total=len(anchors))
 write(U/'FEEDBACK_COMPARISON.json',dict(status='SAVED_OUTPUT_COMPARISON_COMPLETE',cases=cases,summary=grouped,
   target_scope='7 physical feature indicators, 144 sparse original centers including 36 DEV centers; not all fingers or every frame',
   limitations=['Same exposed DEV IDs11-13','92 prior Codex visual stage anchors, not independent expert validation','Left open positive DEV has only one case','No real learner error accuracy established'],
   retraining=0,model_reinference=0))
 event('feedback_compared',summary=grouped)
if __name__=='__main__':main()
