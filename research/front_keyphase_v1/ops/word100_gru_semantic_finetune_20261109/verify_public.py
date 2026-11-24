"""Separate saved-output arithmetic pass, no training or inference."""
from common import *
from collections import Counter
import torch

def main():
 complete=read(U/'EVALUATION_COMPLETE.json');cfg=read(U/'FROZEN.json');checks=[]
 def ok(name,value):
  assert value,name;checks.append(name)
 ok('configuration preserved',sha(U/'FROZEN.json')==complete['frozen_sha256'])
 ok('evaluation plan preserved',sha(U/'EVAL_FROZEN.json')==complete['eval_sha256'])
 for path,h in cfg['source_pins'].items():ok('protected '+path,sha(path)==h)
 for path,h in read(J/'FROZEN.json')['pins'].items():
  if 'training_C_' in path or 'T3F_' in path:ok('original C/T3-F '+path,sha(path)==h)
 for path,h in cfg['data_pins'].items():ok('input '+path,sha(path)==h)
 for name,result in complete['models'].items():
  records=read(U/'scores'/f'{name}.json')
  ok(name+' development IDs only',{r['person'] for r in records}=={11,12,13})
  for variant,tasks in result['by_variant'].items():
   for task,m in tasks.items():
    rr=[r for r in records if r['variant']==variant and r['task']==task]
    eligible=[r for r in rr if r['eligible']];accepted=[r for r in eligible if r.get('confident',True)]
    counts=dict(n=len(rr),raw_correct=sum(r['target']==r['pred'] for r in rr),eligible=len(eligible),eligible_correct=sum(r['target']==r['pred'] for r in eligible),accepted=len(accepted),accepted_correct=sum(r['target']==r['pred'] for r in accepted),withheld=len(rr)-len(accepted))
    ok(f'{name}/{variant}/{task} independent counts',all(m[k]==v for k,v in counts.items()))
    if variant=='missing_main':ok(f'{name}/{task} no evidence never accepted',len(accepted)==0)
    if variant in ['slow','fast','upper','reversed_main','missing_2','missing_main']:ok(f'{name}/{variant}/{task} denominator preserved',len(rr)==result['by_variant']['normal'][task]['n'])
  for p in result['predictions']:ok('saved prediction '+p['path'],sha(p['path'])==p['sha256'])
  if name!='original':
   run=read(U/'runs'/name/'COMPLETE.json');ok(name+' final checkpoint',sha(U/'runs'/name/'model.pt')==run['checkpoint_sha256'])
   ok(name+' fit IDs only',run['fit_persons']==list(range(1,10)) and not run['development_seen'])
   sd=torch.load(U/'runs'/name/'model.pt',weights_only=True,map_location='cpu');base=torch.load(V/'feedback_runs/pose_transfer_42/model.pt',weights_only=True,map_location='cpu')
   same=all(torch.equal(sd['base.'+k],v) for k,v in base.items())
   ok(name+' base freeze/finetune actual weights',same==name.startswith('frozen'))
   if name.startswith('finetune'):
    for prefix in ['context.','temporal.','attributes.']:
     ok(name+' changed '+prefix,any(not torch.equal(sd['base.'+k],v) for k,v in base.items() if k.startswith(prefix)))
 # Explicit breakdown prevents more correct small holds hiding missed core motion.
 breakdown={}
 for name in ['original','frozen_heads_42','finetune_42']:
  rr=read(U/'scores'/f'{name}.json');breakdown[name]={}
  for kind,targets in [('actual_movement',{0,2}),('no_distinct_change',{1})]:
   part=[r for r in rr if r['variant']=='normal' and r['task']=='direction' and r['target'] in targets]
   breakdown[name][kind]=dict(n=len(part),correct=sum(r['pred']==r['target'] for r in part),errors=[{k:r[k] for k in ['clip_id','axis','a','b','target','pred']} for r in part if r['target']!=r['pred']])
  brush=[r for r in rr if r['variant']=='normal' and r['task']=='direction' and r['axis']==2 and r['target']==2]
  breakdown[name]['forearm_stroke']=dict(n=len(brush),correct=sum(r['pred']==r['target'] for r in brush))
 decision=dict(status='OFFLINE_LEARNING_COMPLETED_PARTIAL_GAIN_NOT_READY_FOR_WEB',primary='finetune_42 (fixed before evaluation, not selected on best score)',
  findings=['New reviewed handshape 221/228 -> 223/228; stage raw 72/84 -> 77/84.',
   'Normal order 10/12 ->12/12; reversed order raw12/12 unchanged but confident comparisons7/12 ->12/12.',
   'Direction overall26/32 ->28/32 conceals moving-only20/21 ->19/21; forearm core stroke3/3 ->1/3.',
   'Stage confident correct71/84 ->66/84, withheld3->12. Do not present raw gain as more reliable coverage.',
   'Additional seeds and historical labels show variation; only4words, previously exposed DEV, Codex sparse labels.'],
  runtime_changed=False,web_model_remains=str(V/'feedback_runs/pose_transfer_42/model.pt'),public_breakdown=breakdown,
  next='Keep current geometric direction and uncertainty gate. Add independent visible core-movement negatives/shape changes across people; improve semantic observation windows and calibrate on FIT-only folds before any web replacement.',
  private_review_policy='Subsequent user-test review only. No threshold, weights, selected candidate, public decision changes from private recording.',
  public_results_sha256=sha(U/'EVALUATION_COMPLETE.json'))
 write(U/'PUBLIC_DECISION.json',decision)
 write(U/'PUBLIC_VERIFICATION.json',dict(status='PASS',count=len(checks),checks=checks,reviewer='Same Codex separate code/arithmetic check; not another human or expert',public_decision_sha256=sha(U/'PUBLIC_DECISION.json')))
 event('public_verification_complete',checks=len(checks),decision=decision['status'],core_breakdown=breakdown)
if __name__=='__main__':main()
