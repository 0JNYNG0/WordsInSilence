"""Saved arithmetic and nonregression audit; no imports of training/evaluation functions."""
from pathlib import Path
import json,hashlib
E=Path(__file__).resolve().parent;O=E.parent;N=O/'word100_motion_event_candidates_20261109'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=0
def check(ok,why):
 global checks
 checks+=1
 if not ok:raise AssertionError(why)
def counts(rr,arm):
 cc=[r[arm] for r in rr]
 return dict(clips=len(cc),difference=sum('difference' in c['display'] for c in cc),uncertain=sum('uncertain' in c['display'] for c in cc),missing=sum('missing_candidate' in c['status'] for c in cc),similar=sum(all(d=='similar' for d,req in zip(c['display'],c['required']) if req) for c in cc),required_present=sum(all(p for p,req in zip(c['phase_present'],c['required']) if req) for c in cc))
f=read(E/'GUARD_V2_FROZEN.json');check(sha(E/'completion_guard_v2.py')==f['code_sha256'],'v2 pin');check(sha(E/'completion_guard.py')==f['v1_code_sha256'],'v1 pin');check(sha(f['existing_contract_path'])==f['existing_contract_sha256'],'unchanged original guide contract');check(sha(E/'GUARD_COMPARISON.json')==f['first_guard_sha256'],'first guard audit preserved')
normal={'full_normal','upper_normal','crop_normal','speed075','speed125','speed0.75','speed1.25','translation_scale'};out={}
for version,folder,manifest in [('v1','guard_scores','GUARD_COMPARISON.json'),('v2','guard_v2_scores','GUARD_V2_COMPARISON.json')]:
 for name,result in read(E/manifest)['models'].items():
  rows=read(E/folder/f'{name}_cases.json');check(len(rows)==len({r['key'] for r in rows})==237,'237 fixed denominator');changes=[]
  for k,s in result['by_kind'].items():
   inv,kind=k.split('/');rr=[r for r in rows if r['inventory']==inv and r['kind']==kind]
   for arm in ['deployed','prior','after']:check(counts(rr,arm)==s[arm],'count '+version+name+k+arm)
  for r in rows:
   check(sha(r['decision'])==r['sha256'],'saved result integrity');d=read(r['decision']);base=read(N/'decisions/web'/f"{r['key']}.json")
   for c,b,display in zip(d['cards'],base['cards'],r['after']['display']):
    check(c['display_status']==display,'display arithmetic');check(c['query_indices']==b['query_indices'] and c['phase_evidence']==b['phase_evidence'],'observation window preserved');check(c['shape_evidence']==b['shape_evidence'] and c['motion_evidence']==b['motion_evidence'],'shape/direction evidence preserved')
    check(b['display_status']!='difference' or c['display_status']=='difference','existing difference not lost');check(not c['educational_wrong'],'no unsupported correctness')
    check(c['display_status']==b['display_status'] or (b['display_status']=='similar' and c['display_status']=='uncertain' and any(a['conflict'] for a in c['completion_identity_audit'])),'only positive conflict can withhold')
   if r['deployed']['display']!=r['after']['display']:changes.append(r['key'])
   if version=='v2' and r['kind'] in normal:check(r['deployed']==r['after'],'normal exact nonregression '+r['key'])
   if r['kind']=='all_hand_missing':check(not any(d in ['similar','difference'] for d in r['after']['display']),'missing is not normal/wrong')
  omit=[r for r in rows if r['inventory']=='main144' and r['kind']=='omit_final_event'];mask=[r for r in rows if r['inventory']=='main144' and r['kind']=='mask_final_event'];cross=[r for r in rows if r['inventory']=='main144' and r['kind']=='full_normal' and 'WORD1567' in r['clip_id']]
  if version=='v2':
   check(counts(omit,'after')['similar']==0,'no false all-similar on six omissions');check(counts(mask,'after')['similar']==0 and counts(mask,'after')['missing']==0,'masked event not imputed or omitted');check(counts(cross,'after')==counts(cross,'deployed'),'crossing maintained')
  out[version+'/'+name]=dict(changes=changes,omission=counts(omit,'after'),masked=counts(mask,'after'),crossing=counts(cross,'after'))
check(read(E/'CONTRACT_TESTS.json')['status']=='PASS','13 contract tests')
for p,h in read(E/'FROZEN.json')['protected'].items():check(sha(p)==h,'protected '+p)
result=dict(status='PASS',checks=checks,models=out,normal_v2_exact_preservation=True,reviewer='Same Codex separate arithmetic, not independent expert',new_inference=False,web_default_modified=False)
(E/'GUARD_VERIFIED.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status='PASS',checks=checks,normal_v2_exact_preservation=True),ensure_ascii=False))
