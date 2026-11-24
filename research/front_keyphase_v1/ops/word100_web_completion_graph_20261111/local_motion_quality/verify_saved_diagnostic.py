"""Separate arithmetic implementation, same root Codex; no model execution."""
from pathlib import Path
import json,hashlib,collections,datetime
import numpy as np
W=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=read(W/'EVAL_INPUTS_COMPLETE.json')['rows'];truth=read(W/'DIAGNOSTIC_TARGETS.json')['rows'];result=read(W/'DIAGNOSTIC_SUMMARY.json')
completion=read(W/'EVALUATION_COMPLETE.json');checks=[]
outputs={'A_cached_seed42':[r['old_quality'] for r in rows]}
for r in completion['runs']:
 assert sha(r['path'])==r['sha256'];outputs[r['condition']+str(r['seed'])]=read(r['path'])['probabilities']
for name,probs in outputs.items():
 conf=[[0]*4 for _ in range(4)];unknown=0
 for t in truth:
  if t['target']==-1:unknown+=1;continue
  pred=max(range(4),key=lambda k:probs[t['id']][k]);conf[t['target']][pred]+=1
 assert conf==result['conditions'][name]['direct_confusion']
 assert unknown==result['conditions'][name]['direct_unknown']
 expected_groups={}
 for t in truth:
  if t['origin']=='boundary':expected_groups.setdefault((t['clip_id'],t['kind'],t['axis']),[]).append(t)
 for saved in result['conditions'][name]['groups']:
  pool=expected_groups[saved['clip_id'],saved['kind'],saved['axis']]
  ranked=sorted(pool,key=lambda t:(-probs[t['id']][2],-probs[t['id']][1],t['span'][0],t['span'][1]))
  assert ranked[0]['id']==saved['rank_winner']['id']
  obs=[t for t in pool if max(range(4),key=lambda k:probs[t['id']][k]) in [1,2]]
  assert saved['witness_retained']==any(t['witness_covered'] for t in obs)
 checks.append(dict(condition=name,confusion=conf,unknown=unknown,rankings=12))
counter=collections.Counter(t['target'] for t in truth if t['target']>=0)
report=dict(status='PASS_SAVED_ARITHMETIC_NOT_G2_ACCEPTANCE',at=datetime.datetime.now().isoformat(),reviewer='Root Codex second arithmetic implementation after delegated preflight/training audit; not a fresh agent or human verification',model_calls=0,counts=dict(input_scenarios=82,candidate_rows=2926,new_checkpoint_runs=6,direct_targeted=16,direct_unknown=110),checks=checks,constant_class2_baseline=dict(correct=counter[2],total=sum(counter.values())),critical_interpretation='14/16 equals always predicting extent2 on this small reviewed subset. No improvement claim. Ranking of true motion is not stable across seeds or normal conditions. Auxiliary continuity and semantic clause identity were not trained.',web_changed=False,private_access=False)
(W.parent/'reviews/local_quality_saved_evaluation_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(report['status'],'constant baseline',report['constant_class2_baseline'])
