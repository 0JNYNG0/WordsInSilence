"""Separate saved-output audit; no training or inference imports."""
from pathlib import Path
import json,hashlib,collections,statistics
U=Path(__file__).resolve().parent;R=U/'boundary_integrity_repair';O=U.parent
rd=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[];scientific=[]
def check(name,ok):
 checks.append(dict(name=name,passed=bool(ok)))
 if not ok:raise AssertionError(name)
cfg=rd(U/'FROZEN.json')
for p,h in {**cfg['protections'],**cfg['pins'],**rd(U/'LABELS_FROZEN.json')['files'],**rd(U/'EVAL_PINS.json')}.items():check('preserved '+p,sha(p)==h)
fit=rd(U/'FIT_LABELS.json')['rows'];ev=rd(U/'EVALUATION_LABELS.json')['rows'];src=rd(O/'word100_rtmw_joint_learning_20261108/SOURCES.json');fit_src={r['source_sha256'] for r in src if r['role']=='fit'}
check('899 FIT100 word preserved',sum(r['role']=='fit' for r in src)==899 and len({r['word_id'] for r in src if r['role']=='fit'})==100)
check('fine train only01to09',all(1<=a['person']<=9 and a['source_sha256'] in fit_src for a in fit))
check('six reserved14 disjoint hashes',len(ev)==6 and all(a['person']==14 and a['source_sha256'] not in fit_src for a in ev))
check('30 fine videos49movements28holds',len({r['clip_id'] for r in fit})==30 and sum(len(a['events']) for a in fit)==49 and sum(len(a['holds']) for a in fit)==28)
check('first and repaired output coexist',(U/'RESULT.json').exists() and (R/'RESULT.json').exists())
for seed in [42,43,44]:
 c=rd(U/'runs'/str(seed)/'COMPLETE.json');check('checkpoint '+str(seed),sha(U/'runs'/str(seed)/'model.pt')==c['sha256'] and c['epochs']==60)
dr=rd(R/'DIRECTION_RESULTS.json');ers=rd(R/'EVENT_RESULTS.json');frs=rd(R/'FULLPATH_COMPLETE.json')['rows'];res=rd(R/'RESULT.json');obs=rd(R/'OBSERVATIONS.json')['rows']
check('all158conditions3seeds',len(obs)==158*3 and len(frs)==78*2*3)
def good(d,label):return bool(d and d['eligible'] and d['prediction']==label)
for key,s in res['direction'].items():
 seed,group,kind=key.split('/');rows=[r for r in dr if r['seed']==int(seed) and r['group']==group and r['kind']==kind];check('denominator '+key,s['total']==len(rows))
 for name,source,pred in [('baseline','baseline','before'),('motion_only','baseline','after'),('localization_only','union','before'),('combined','union','after'),('new_standalone','localized','after')]:
  check('recomputed '+key+name,s[name]==sum(good(r['automatic'][source].get(pred),r['truth']['direction']) for r in rows))
  check('one truth one paired interval '+key+name,len({(r['clip_id'],r['automatic'][source]['event']['a'],r['automatic'][source]['event']['b'],r['truth']['axis'],r['truth']['moving']) for r in rows if r['automatic'][source]['matched']})==sum(r['automatic'][source]['matched'] for r in rows))
for r in obs:
 es=rd(r['file'])['events']
 check('no overlapping same-relation intervals '+r['file'],not any(a['axis']==b['axis'] and a['moving']==b['moving'] and max(a['a'],b['a'])<min(a['b'],b['b']) for i,a in enumerate(es) for b in es[i+1:]))
 check('no semantic completion by interval '+r['file'],not any(e['semantic_complete'] for e in es))
for r in frs:
 d=rd(r['result']);check('feedback hash '+r['result'],sha(r['result'])==r['sha256']);before=rd(d['baseline']);check('baseline hash '+r['result'],sha(d['baseline'])==d['baseline_sha256'])
 for a,b in zip(d['cards'],before['cards']):check('shape/motion unchanged '+r['result']+a['id'],a.get('shape_evidence')==b.get('shape_evidence') and a.get('motion_evidence')==b.get('motion_evidence'))
 for k,h in d['protected'].items():check('protected inference '+r['result']+k,hashlib.sha256(json.dumps(before[k],sort_keys=True,ensure_ascii=False).encode()).hexdigest()==h)
 check('no endpoint semantic promotion '+r['result'],not any(m['endpoint']['semantic_complete'] for b in d['bindings'] for m in b['observed']))
 for entry in d['ledger']:check('no desired direction candidate filter '+r['result']+str(entry['event']),entry['expected_direction_used'] is False)
for key,v in res['feedback'].items():
 seed,mode,kind=key.split('/');rows=[r for r in frs if r['seed']==int(seed) and r['mode']==mode and r['kind']==kind]
 check('feedback denominator '+key,len(rows)==v['total'])
 for source in ['baseline','motion_only','after']:
  check('feedback arithmetic '+key+source,sum(r[source]['warning'] for r in rows)==v[source+'_warnings'] and sum(r[source]['similar_no_warning'] for r in rows)==v[source+'_similar'])
 if kind in ['full_normal','speed075','speed125']:scientific.append(dict(name='normal no new warning '+key,passed=v['after_warnings']==v['baseline_warnings']==0,added_warning_clips=v['new_warning_clips']))
 if kind in ['all_hand_missing','mask_final_event','omit_final_event']:check('unknown not normal '+key,v['after_similar']==0)
 if kind in ['right_closed_hand_main','main_reversed','closed_hand_and_reversed']:check('existing error detection retained '+key,v['after_warnings']==v['baseline_warnings']==3)
prior=rd(O/'word100_short_motion_completion_20261111/RESULT.json');comparison=[]
for seed in [42,43,44]:
 for kind in ['full_normal','speed075','speed125','reversed_core_auto_order','omit_final_event']:
  now=res['feedback'][f'{seed}/combined/{kind}'];then=prior['fullpath'][f'{seed}/combined/{kind}'];comparison.append(dict(seed=seed,kind=kind,prior_warnings=then['after_warnings'],now_warnings=now['after_warnings'],prior_similar=then['after_similar'],now_similar=now['after_similar']))
report=dict(passed=True,integrity_passed=True,scientific_normal_safety_passed=all(r['passed'] for r in scientific),scientific_checks=scientific,checks=len(checks),reviewer='Codex separate stdlib saved-output audit, NOT independent human/expert validation',previous_combined_comparison=comparison,all_checks=checks)
(U/'VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf-8');print(json.dumps(dict(integrity_passed=True,scientific_normal_safety_passed=report['scientific_normal_safety_passed'],failed_scientific_checks=[r for r in scientific if not r['passed']],checks=len(checks),previous_combined_comparison=comparison),ensure_ascii=False))
