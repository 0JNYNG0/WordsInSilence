"""Saved-output arithmetic and actual logged exposure audit; never infer models."""
from pathlib import Path
import json,hashlib
from collections import Counter
import numpy as np
U=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=Counter();errors=[]
def check(k,ok,where):
 checks[k]+=1
 if not ok:errors.append(dict(check=k,where=str(where)))
def main():
 data=read(U/'DATASET.json');f=read(U/'TRAIN_FROZEN.json');scores=read(U/'SCORES.json');runs=read(U/'TRAINING.json')['runs'];rows=data['rows'];names=data['selected_targets'];ids=[data['targets'].index(k) for k in names]
 gt=np.asarray([r['y'] for r in rows])[:,ids];mask=np.asarray([r['target_mask'] for r in rows])[:,ids]
 roles=sorted({r['role'] for r in rows})
 families={'all':names,'shape':[n for n in names if n in ['R_index_only','R_open_four','L_open_four']],'relation':[n for n in names if n.endswith('higher')],'motion':[n for n in names if '_motion_' in n]}
 rowkeys=[(r['condition'],r['seed'],r['role'],r['family']) for r in scores['rows']]
 expected={(a,s,r,g) for a in f['conditions'] for s in f['seeds'] for r in roles for g in families}
 check('score_rows_complete',len(rowkeys)==len(expected) and set(rowkeys)==expected,'condition/seed/role/family')
 summarykeys=[(r['condition'],r['role'],r['family']) for r in scores['summary']]
 expected_summary={(a,r,g) for a in f['conditions'] for r in roles for g in families}
 check('score_summary_complete',len(summarykeys)==len(expected_summary) and set(summarykeys)==expected_summary,'condition/role/family')
 for score in scores['rows']:
  check('score_target_complete',[r['target'] for r in score['targets']]==families.get(score['family']),score['family'])
 fit=[r['sample_id'] for r in rows if r['role']=='fit'];fit_set=set(fit);exposures=0;positive=0;negative=0
 check('six_runs',len(runs)==6 and {(r['condition'],r['seed']) for r in runs}=={(a,s) for a in f['conditions'] for s in f['seeds']},'runs')
 old=U.parent/'word100_stage_feedback_cycle_20261031';oldinfo=read(old/'LEARNING_INPUTS.json');oldrows={r['clip_id']:(i,r) for i,r in enumerate(oldinfo['rows'])};oldcols=[oldinfo['targets'].index(k) for k in ['R_index_only','R_open_four','L_open_four']]
 for rr in runs:
  d=U/'runs'/f"{rr['condition']}_{rr['seed']}";done=read(d/'COMPLETE.json');prob=np.load(d/'probabilities.npy',allow_pickle=False)
  for filename,key in [('head.pt','checkpoint_sha256'),('probabilities.npy','probabilities_sha256'),('EXPOSURES.jsonl','exposures_sha256')]:check('run_sha',sha(d/filename)==done[key],d/filename)
  check('prob_schema',prob.shape==gt.shape and np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all(),str(d))
  logs=[json.loads(l) for l in (d/'EXPOSURES.jsonl').read_text(encoding='utf8').splitlines()]
  check('epoch_final',done['epochs']==done['selected_epoch']==len(logs)==160,str(d))
  for epoch,entry in enumerate(logs,1):
   check('exposure_exact',entry['epoch']==epoch and entry['sample_ids']==fit and set(entry['sample_ids'])==fit_set,(d,epoch));exposures+=len(entry['sample_ids'])
   check('loss_trace',entry['loss']==done['losses'][epoch-1] and np.isfinite(entry['loss']),(d,epoch))
  if rr['condition']=='protected_shape_split_motion':
   cache=U/'protected'/f"probabilities_{rr['seed']}.npy";protected=np.load(cache,allow_pickle=False);audit=read(U/'protected'/f"AUDIT_{rr['seed']}.json")
   check('protected_cache_sha',sha(cache)==audit['probabilities_sha256']==done['protected_probability_sha256'],str(d))
   cols=[names.index(n) for n in ['R_index_only','R_open_four','L_open_four']]
   check('shape_exact',np.array_equal(prob[:,cols],protected),str(d))
   check('old_inference_zero',audit['previous_clip_inference_count']==0 and audit['new_clips_inferred']==48 and audit['old_rows_reused']==144,str(d))
   pp=np.load(old/'runs'/f"pose_plus_signrep_{rr['seed']}"/'probabilities.npy',allow_pickle=False)
   for i,row in enumerate(rows):
    if row['clip_id'] in oldrows:
     c,oldrow=oldrows[row['clip_id']];a=np.load(oldrow['input'],allow_pickle=False);ix=np.flatnonzero(a['frame_id']==row['center_frame']);check('old_center_exact',len(ix)==1,row['sample_id'])
     check('old_probability_exact',np.array_equal(protected[i],pp[c,int(ix[0]),oldcols]),row['sample_id'])
  selected=[s for s in scores['rows'] if s['condition']==rr['condition'] and s['seed']==rr['seed']]
  for score in selected:
   ri=[i for i,r in enumerate(rows) if r['role']==score['role']];ba=[]
   check('score_window_count',score['windows']==len(ri),(d,score['role']))
   for t in score['targets']:
    j=names.index(t['target']);tp=tn=fp=fn=0
    for i in ri:
     if not mask[i,j]:continue
     pred=prob[i,j]>=.5;truth=gt[i,j]==1
     if truth and pred:tp+=1
     elif truth:fn+=1
     elif pred:fp+=1
     else:tn+=1
    check('confusion_exact',[tp,tn,fp,fn]==[t[k] for k in ['tp','tn','fp','fn']],(d,score['role'],t['target']))
    check('target_denominator',t['known']==tp+tn+fp+fn and t['unknown']==len(ri)-tp-tn-fp-fn,(d,score['role'],t['target']))
    value=(tp/(tp+fn)+tn/(tn+fp))/2 if tp+fn and tn+fp else None
    check('ba_exact',value==t['balanced_accuracy'],(d,score['role'],t['target']))
    if value is not None:ba.append(value)
   value=sum(ba)/len(ba) if ba else None
   check('supported_target_count',score['balanced_supported_targets']==len(ba),(d,score['role'],score['family']))
   check('macro_exact',value is None and score['macro_balanced_accuracy'] is None or value is not None and abs(value-score['macro_balanced_accuracy'])<1e-12,(d,score['role'],score['family']))
 for p,h in {**f['pins'],**f['source_pins'],**f['protected']}.items():check('frozen_hash',sha(p)==h,p)
 for s in scores['summary']:
  rr=[r['macro_balanced_accuracy'] for r in scores['rows'] if r['condition']==s['condition'] and r['role']==s['role'] and r['family']==s['family'] and r['macro_balanced_accuracy'] is not None]
  counts=[r['balanced_supported_targets'] for r in scores['rows'] if r['condition']==s['condition'] and r['role']==s['role'] and r['family']==s['family']]
  check('summary_support_count',len(counts)==3 and all(v==s['balanced_supported_targets'] for v in counts),s)
  check('seed_summary',not rr and s['mean'] is None or bool(rr) and abs(sum(rr)/len(rr)-s['mean'])<1e-12 and [min(rr),max(rr)]==s['range'],s)
 out=dict(status='PASS_SAVED_RESULTS_AND_EXPOSURES' if not errors else 'FAIL_SAVED_RESULTS_AND_EXPOSURES',checks=dict(checks),count=sum(checks.values()),errors=errors,logged_window_exposures=exposures,model_reinference=0,
  truth_limit='Arithmetic/code verification by same Codex; not independent human sign/feedback correctness')
 with (U/'review/RESULT_VERIFICATION.json').open('x',encoding='utf8') as dest:json.dump(out,dest,ensure_ascii=False,indent=2)
 print(json.dumps(out,ensure_ascii=False));assert not errors
if __name__=='__main__':main()
