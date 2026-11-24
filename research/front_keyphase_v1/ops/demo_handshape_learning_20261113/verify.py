"""Separate saved-result arithmetic/contract check. No retraining/reinference."""
import json,hashlib
from pathlib import Path
import numpy as np
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 checks=[]
 def ck(name,ok):
  checks.append(dict(check=name,passed=bool(ok)));assert ok,name
 f=read(U/'FROZEN.json');a=read(U/'ADMISSION.json');inp=read(U/'INPUTS.json');s=read(U/'SHAPE_RESULTS.json');t=read(U/'TCN_RESULTS.json')
 ck('all original model pins preserved',all(sha(p)==h for p,h in f['protection'].items()))
 ck('source accounting',len(inp['rows'])==59 and len(inp['failures'])==1 and inp['planned_inputs']==60)
 ck('source failure not trained',all(r['clip_id']!='NIA_SL_WORD2635_REAL05_F' for r in a['rows']))
 ck('ambiguous apple variation not trained','NIA_SL_WORD0099_REAL05_F' not in a['classifier_admitted'])
 ck('no result-based admission change',sha(U/'ADMISSION.json')==read(U/'PRELEARN_VERIFY.json')['admission_sha256'])
 ck('no result-based contract change',sha(U/'LEARNING_CONTRACT.json')==read(U/'PRELEARN_VERIFY.json')['contract_sha256'])
 ck('exact six fixed shape probes',len(s['runs'])==6 and {r['seed'] for r in s['runs']}=={42,43,44})
 for run in s['runs']:
  pp=run['anchors'];vv=run['videos'];name=run['condition']+str(run['seed'])
  ck(name+' sums',run['video_correct']==sum(r['truth']==r['top1'] for r in vv) and len(vv)==9 and len(pp)==27)
  ck(name+' means',all(np.allclose(v['prob'],np.mean([p['prob'] for p in pp if p['clip_id']==v['clip_id']],0)) for v in vv))
  ck(name+' fingers',sum((p>=.5)==bool(g) for r in pp for g,p in zip(r['finger_truth'],r['finger_prob']) if g>=0)==run['finger_correct'])
  ck(name+' holdout IDs',set(r['person'] for r in pp)=={11,12,13})
  ck(name+' probability normalization',all(abs(sum(r['prob'])-1)<1e-5 for r in pp))
  ck(name+' checkpoint',sha(U/f'shape_{run["condition"]}_{run["seed"]}.pt')==run['checkpoint_sha256'])
 for group,new in [('old100',False),('added4',True)]:
  rr=[r for r in t['predictions'] if r['new']==new];m=t[group]
  ck(group+' denominator',len(rr)==m['total'])
  for metric,k in [('top1','top1'),('baseline_top1','baseline_top1')]:ck(group+metric,sum(r[k]==r['truth'] for r in rr)==m[metric])
  for metric,k in [('top3','top3'),('baseline_top3','baseline_top3')]:ck(group+metric,sum(r['truth'] in r[k] for r in rr)==m[metric])
 ck('no classifier promotion despite regression',t['production_replaced']==False and t['old100']['top1']<t['old100']['baseline_top1'])
 failures=[dict(clip_id=r['clip_id'],true_class=r['truth'],old=r['baseline_top1'],new=r['top1']) for r in t['predictions'] if not r['new'] and r['baseline_top1']==r['truth'] and r['top1']!=r['truth']]
 report=dict(status='PASS_SAVED_ARITHMETIC_AND_PROTECTION',checks=checks,count=len(checks),regressions=failures,verifier='same Codex separate script; not independent human ground truth',inference_calls=0,training_calls=0)
 (U/'VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print('VERIFIED',len(checks))
if __name__=='__main__':main()
