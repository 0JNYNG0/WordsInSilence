from pathlib import Path
import json,hashlib
U=Path(__file__).resolve().parent;V=U/'proposal_adapter'
def rd(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[]
def ck(n,v):checks.append(dict(name=n,passed=bool(v)));assert v,n
cfg=rd(U/'FROZEN.json');s=rd(V/'RESULT.json');ds=rd(V/'DIAGNOSTIC.json');fr=rd(V/'FULLPATH_COMPLETE.json')['rows']
for p,h in {**cfg['protections'],**cfg['pins'],**rd(V/'EVAL_PINS.json')}.items():ck('pin:'+p,sha(p)==h)
ck('feedback_count',len(fr)==702 and len({(r['seed'],r['mode'],r['clip_id'],r['kind']) for r in fr})==702)
for r in fr:
 ck('feedback_hash',sha(r['result'])==r['sha256']);d=rd(r['result']);b=rd(d['baseline']);ck('baseline_hash',sha(d['baseline'])==d['baseline_sha256']);statuses=[c['display_status'] for c in d['cards']];ck('display',statuses==r['after']['display']);ck('warning',('difference' in statuses)==r['after']['warning'])
 for a,z in zip(d['cards'],b['cards']):ck('shape',a.get('shape_evidence')==z.get('shape_evidence'));ck('direction',a.get('motion_evidence')==z.get('motion_evidence'))
for g in s['groups']:
 rs=[r for r in fr if (r['seed'],r['mode'],r['kind'])==(g['seed'],g['mode'],g['kind'])];ck('counts',len(rs)==g['n'] and sum(r['after']['warning'] for r in rs)==g['after_warning'] and sum(r['after']['similar_no_warning'] for r in rs)==g['after_similar'])
 if g['kind'] in ['full_normal','speed075','speed125']:ck('normal_speed_kept',all(r['before']==r['after'] for r in rs))
 if g['kind'] in ['all_hand_missing','mask_final_event','omit_final_event']:ck('not_promoted_to_similar',g['after_similar']==0)
for g in s['components']:
 rs=[r for r in ds if (r['seed'],r['group'],r['kind'])==(g['seed'],g['group'],g['kind'])];matched=direction=both=exact=0
 for r in rs:
  a=r['automatic'][g['mode']];d=a['direction'];f=a['family'];t=r['truth'];ok=bool(d and d['eligible'] and d['prediction']==t['direction']);matched+=a['matched'];direction+=ok;both+=ok and (t.get('family') is None or bool(f and f['pred']==t['family']));exact+=a['both_boundaries_in_range']
 ck('component_arithmetic',(len(rs),matched,direction,both,exact)==(g['n'],g['matched'],g['direction'],g['both'],g['exact_bounds']))
for seed in [42,43,44]:
 for group,clip,kind in {(r['group'],r['clip_id'],r['kind']) for r in ds}:
  rs=[r for r in ds if (r['seed'],r['group'],r['clip_id'],r['kind'])==(seed,group,clip,kind)]
  for mode in ['baseline','interval_only','role_only','combined','selected_only']:
   es=[(e['axis'],e['moving'],e['a'],e['b']) for r in rs if (e:=r['automatic'][mode]['event'])];ck('one_to_one',len(es)==len(set(es)))
for r in rd(V/'OBSERVATIONS.json')['rows']:ck('observation_hash',sha(r['file'])==r['sha256'])
fit=rd(U/'ALL_FIT_BOUNDARIES.json')['rows'];labels=rd(V/'EVALUATION_LABELS.json')['rows'];ck('reserved_person_split',not {r['person'] for r in labels}&{r['person'] for r in fit});ck('reserved_sha_split',not {r['source_sha256'] for r in labels}&{r['source_sha256'] for r in fit});ck('reserved_events',sum(len(r['events']) for r in labels)==10)
for r in labels:ck('review_sheet',sha(r['sheet'])==r['sheet_sha256'])
for r in rd(V/'EXTRACTION_COMPLETE.json')['rows']:ck('raw_sha',sha(r['raw'])==r['raw_sha256']);ck('source_sha',sha(r['source'])==r['source_sha256'])
for seed in [42,43,44]:ck('model_sha',sha(V/'runs'/str(seed)/'model.pt')==rd(V/'runs'/str(seed)/'COMPLETE.json')['sha256'])
(V/'VERIFICATION.json').write_text(json.dumps(dict(passed=True,checks=len(checks),result_sha256=sha(V/'RESULT.json'),reviewer='Same Codex separate stdlib verifier, not independent human expert.',items=checks),ensure_ascii=False,indent=2),'utf-8');print('PASS',len(checks))
