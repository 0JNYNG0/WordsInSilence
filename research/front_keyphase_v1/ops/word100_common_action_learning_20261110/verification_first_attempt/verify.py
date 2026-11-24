"""Separate saved-result arithmetic and preservation audit; not human truth."""
from pathlib import Path
import json,hashlib,statistics
import numpy as np
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
def main():
 checks=[]
 def check(name,ok):
  checks.append(dict(name=name,pass_=bool(ok)))
  if not ok:raise AssertionError(name)
 cfg=read(U/'FROZEN.json');audit=read(U/'ANNOTATION_AUDIT.json');ev=read(U/'EVALUATION_COMPLETE.json');full=read(U/'FULLPATH_COMPLETE.json')['rows']
 for p,h in {**cfg['protected'],**cfg['source_pins'],**cfg['code_pins']}.items():check('sha '+p,sha(p)==h)
 rr=read(U/'READY.json')['rows'];check('72 public labelled clips',len(rr)==72);check('no recordID10 training',all(r['person']!=10 for r in rr));check('training ids1-9',set(r['person'] for r in rr if r['role']=='fit')==set(range(1,10)));check('dev ids11-13',set(r['person'] for r in rr if r['role']=='development')=={11,12,13});check('4 motion expressions',len({r['word_id'] for r in rr if r['pairs']})==4)
 check('six fixed training runs',len(read(U/'TRAINING_COMPLETE.json')['runs'])==6)
 for p in (U/'runs').glob('*/COMPLETE.json'):
  v=read(p);check(str(p)+' checksum',sha(p.parent/'model.pt')==v['model_sha256']);check(str(p)+' no dev fit',v['dev_used']==False and v['fit_clips']==60 and v['epochs']==30)
 sums={};causes=[]
 for name,data in ev['models'].items():
  records=read(U/'scores'/f'{name}.json')
  for variant,tasks in data['by_variant'].items():
   for task,s in tasks.items():
    sub=[r for r in records if r['variant']==variant and r['task']==task]
    check(name+'/'+variant+'/'+task+' denominator',len(sub)==s['total'])
    check(name+'/'+variant+'/'+task+' correct',sum(r['pred']==r['target'] for r in sub)==s['raw_correct'])
    check(name+'/'+variant+'/'+task+' accepted correct',sum(r['eligible'] and r['confident'] and r['pred']==r['target'] for r in sub)==s['accepted_correct'])
  check(name+' missing movement not accepted',data['by_variant']['missing_long']['movement']['accepted']==0)
  rows=[r for r in full if r['model']==name];check(name+' 69 cases including errors',len(rows)==69);check(name+' no failures',all(r['status']=='OK' for r in rows));sums[name]={}
  for kind in sorted({r['kind'] for r in rows}):
   sub=[r for r in rows if r['kind']==kind];sums[name][kind]=dict(total=len(sub),before_similar=sum(r['baseline']['all_required_similar'] for r in sub),after_similar=sum(r['candidate']['all_required_similar'] for r in sub),before_difference=sum('difference' in r['baseline']['display'] for r in sub),after_difference=sum('difference' in r['candidate']['display'] for r in sub))
  for r in rows:
   check(name+'/'+r['clip_id']+'/'+r['kind']+' artifact',sha(r['result'])==r['sha256']);d=read(r['result']);check('no reference/rgb calls',d['reference_video_reads']==0 and d['rgb_model_calls']==0)
   if r.get('frames_preserved'):check('same TCN',r['tcn_identical'])
   if r['kind'] in ['all_hand_missing','mask_final_event','omit_final_event']:check('not complete with missing/omission',not r['candidate']['all_required_similar'])
   if r['kind'] in ['right_closed_hand_main','main_reversed','closed_hand_and_reversed']:
    # A handshape alert must not hide a wrong directional diagnosis.
    c=d['cards'][0];m=c.get('motion_evidence');expected='contradicted' if r['kind']!='right_closed_hand_main' else 'supported'
    causes.append(dict(model=name,clip_id=r['clip_id'],kind=r['kind'],expected_direction_state=expected,new_direction_state=m['state'] if m else 'missing',direction_correct=bool(m and m['state']==expected),overall_difference=c['display_status']=='difference',shape_states=[q['state'] for q in c.get('shape_evidence',[])],span=m.get('span') if m else None))
  times=[r['seconds'] for r in rows if r['kind'] not in ['right_closed_hand_main','main_reversed','closed_hand_and_reversed']]
  sums[name]['timing']=dict(median=statistics.median(times),p95=float(np.percentile(times,95)),scope='60 cached RTMW raw input consumer calls including common-observation diagnostics; excludes detector, camera, model initialization, file save and browser')
 write(U/'FULLPATH_SUMMARY.json',sums);write(U/'COMPONENT_CAUSE_AUDIT.json',dict(rows=causes,summary={name:{kind:dict(correct=sum(r['direction_correct'] for r in causes if r['model']==name and r['kind']==kind),total=sum(r['model']==name and r['kind']==kind for r in causes)) for kind in ['right_closed_hand_main','main_reversed','closed_hand_and_reversed']} for name in sums},inference='Counterfactual changes of handshape affect the learned direction despite unchanged intended path. This supports feature entanglement; it does not establish every causal mechanism. Changed latent also changes old phase head windows. No DEV retuning or retraining after these findings.'))
 hist=read(U/'HISTORY_BACKFILL_VERIFICATION.json');check('history sha',sha(hist['history'])==hist['history_sha256']);check('index sha',sha(hist['index'])==hist['index_sha256'])
 for r in hist['narrative_evidence']:check('archived report unchanged '+r['unit'],sha(r['path'])==r['sha256'])
 write(U/'VERIFICATION.json',dict(status='PASS',checks=len(checks),details=checks,verifier='same Codex separate arithmetic script; not independent annotator/expert',candidate_adopted=False))
 print(json.dumps(dict(checks=len(checks),primary=sums['joint_finetune_42'],direction=read(U/'COMPONENT_CAUSE_AUDIT.json')['summary']['joint_finetune_42']),ensure_ascii=False))
if __name__=='__main__':main()
