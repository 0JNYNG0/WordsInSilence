"""Separate saved-output arithmetic/provenance checks. No imports of training code.
Same Codex authored this checker; not independent human annotation validation.
"""
from pathlib import Path
import json,hashlib,collections
import numpy as np
U=Path(__file__).resolve().parent; O=U.parent
def rd(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[]
def ck(label,value):checks.append(dict(check=label,pass_=bool(value)))
for file in ['PROTECTIONS.json','FROZEN.json','EVAL_FROZEN.json']:
 d=rd(U/file);pins=d if file=='PROTECTIONS.json' else d['pins']
 for p,h in pins.items():ck('unchanged '+p,digest(p)==h)
cfg=rd(U/'FROZEN.json');train=rd(U/'TRAIN_MANIFEST.json');lab=rd(U/'REQUIREMENT_LABELS.json');z=np.load(U/'train.npz')
ck('sample_count',len(train)==len(z['y'])==3004)
ck('FIT IDs',all(r['person'] in range(1,10) for r in train))
ck('10 labeled expressions',len({r['word_id'] for r in train})==10)
ck('54 labeled public videos',len({r['clip_id'] for r in train})==54)
ck('no semantic truncation targets',all('prefix' not in r and 'suffix' not in r for r in train))
ck('no word ID model channel',z['ob'].shape[1]==250 and z['req'].shape[1]==76)
for r in train:
 ck('finite array '+str(r['sample']),np.isfinite(z['ob'][r['sample']]).all())
 if r['task']=='direction':ck('direction truth '+str(r['sample']),r['target']==int(r['expected']==r['observed']))
 if r['task']=='order':ck('order contrast truth '+str(r['sample']),r['target']==int(r['contrast']=='reviewed_order'))
ready=rd(O/'word100_shared_motion_learning_20261109/READY.json')['rows']
fit_hash={r['source_sha256'] for r in ready if r['clip_id'] in {t['clip_id'] for t in train}}
ev=rd(O/'word100_candidate_role_learning_20261111/EVALUATION_INPUTS.json')['rows']
ck('FIT/eval source-disjoint',not fit_hash&{r['source_sha256'] for r in ev})
pred=rd(U/'SCORES.json');summary=rd(U/'EVALUATION_COMPLETE.json')['summary']
for seed in cfg['seeds']:
 for task in ['shape','direction','order']:
  a=[r for r in pred if r['seed']==seed and r['task']==task];s=summary[str(seed)][task]
  ck(f'arithmetic before {seed} {task}',sum((r['baseline_probability']>=.5)==bool(r['target']) for r in a)==s['before_correct'])
  ck(f'arithmetic after {seed} {task}',sum((r['probability']>=.5)==bool(r['target']) for r in a)==s['after_correct'])
  for r in a:ck('target from original annotation',r['target']==(r['annotation']['target'] if task=='order' else int(r['expected']==r['annotation']['target'])))
for f in ['TRAINING_COMPLETE.json','ORDER_BINDING_TRAINING.json']:
 d=rd(U/f);rr=d['rows'] if isinstance(d,dict) else d
 for r in rr:ck('checkpoint '+r['path'],digest(r['path'])==r['sha256'])
control=rd(U/'CONTROLS.json')
ck('shorter elapsed time invariant',max(r['shorter_time_delta'] for r in control)<1e-5)
ck('body scale translation invariant',max(r['body_scale_translation_delta'] for r in control)<1e-4)
ck('missing never normal',all(r['masked_state']=='observation_unknown' for r in control))
ck('association unknown separate',all(r['unknown_association']=='association_unknown' for r in control))
for r in rd(U/'SEMANTIC_GATE_TESTS.json'):ck('hold/opposite/unknown contract',r['expected']==r['actual'])
full=rd(U/'CORRECTED_FULLPATH_DIAGNOSTICS.json')
ck('78 x3 cases preserved',len(full)==234)
for r in full:
 old=rd(r['previous']);ck('protected output hash unchanged',old['protected']==r['protected'] and digest(r['previous'])==r['previous_sha256'])
 if r['kind']=='all_hand_missing':ck('all missing not promoted',all(c['after'] not in ['supported_features_only','difference'] for c in r['cards']))
normal=[r for r in full if r['kind'] in ['full_normal','speed075','speed125']]
ck('corrected consumer no new normal warning',all(c['before']=='difference' or c['after']!='difference' for r in normal for c in r['cards']))
bindings=rd(U/'ORDER_BINDING_SCORES.json');replay=rd(U/'ORDER_REPLAY_SCORES.json');auto=rd(U/'ORDER_AUTOMATIC_DIAGNOSTICS.json');scientific={}
for seed in cfg['seeds']:
 pairs=collections.defaultdict(dict)
 for r in bindings:
  if r['seed']==seed:pairs[(r['clip_id'],r['pair'])][r['swapped']]=r
 for p in pairs.values():ck('opposite queries cannot both pass',abs(p[False]['probability']+p[True]['probability']-1)<1e-6)
 a=[r for r in replay if r['seed']==seed and r['kind']=='reversed_core_auto_order']
 ac=[r for r in a if r['after'] not in ['uncertain','observation_unknown']]
 scientific[str(seed)]=dict(reverse_n=len(a),reverse_before_correct=sum((r['baseline_probability']>=.5)==bool(r['target']) for r in a),reverse_before_confident_correct=sum((r['baseline_probability']>=.5)==bool(r['target']) and r['before'] in ['supported','difference'] for r in a),reverse_after_correct=sum((r['probability']>=.5)==bool(r['target']) for r in a),reverse_after_confident_correct=sum((r['probability']>=.5)==bool(r['target']) for r in ac),reverse_withheld=len(a)-len(ac),normal_order_correct=sum(r['probability']>=.5 for r in bindings if r['seed']==seed and not r['swapped']),speed_correct=sum((r['probability']>=.5)==bool(r['target']) for r in replay if r['seed']==seed and r['kind'] in ['speed075','speed125']),auto_reverse_states=dict(collections.Counter(p['state'] for r in auto if r['seed']==seed and r['kind']=='reversed_core_auto_order' for p in r['pairs'])))
 ck('12 known reverse targets',len(a)==12 and all(r['target']==0 for r in a))
 ck('11 safe reverse decisions',len(ac)==11 and all((r['probability']>=.5)==bool(r['target']) for r in ac))
 ck('missing endpoint never promoted',all(r['after']=='observation_unknown' for r in replay if r['seed']==seed and not r['eligible']))
result=dict(passed=all(c['pass_'] for c in checks),checks=len(checks),failed=[c for c in checks if not c['pass_']],scientific=scientific,same_codex_separate_script=True,independent_human=False,model_inference_rerun=False,default_changed=False)
(U/'VERIFICATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
assert result['passed']
