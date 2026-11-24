"""Validate saved comparison plus semantic distinctions without re-inference."""
from pathlib import Path
import json,hashlib,copy,collections
import numpy as np
from component_visibility import guard_component,hand_visibility
U=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):
 with(U/n).open('x',encoding='utf-8')as f:json.dump(d,f,ensure_ascii=False,indent=2)
checks=[]
def ck(n,v):checks.append(dict(name=n,passed=bool(v)))
f=read(U/'VISIBILITY_FROZEN.json');after=read(U/'VISIBILITY_COMPARISON.json');before=read(U/'JOINT_SAVED_COMPARISON.json')
key=lambda r:(r['seed'],r['clip_id'],r['kind'])
am={key(r):r for r in after['rows']};bm={key(r):r for r in before['rows']}
ck('all_cases',len(am)==246 and set(am)==set(bm))
ck('guard_code_pin',sha(U/'component_visibility.py')==f['code_sha256'])
ck('comparison_code_pin',sha(U/'compare_visibility.py')==f['compare_sha256'])
ck('input_pin',sha(U/'JOINT_SAVED_COMPARISON.json')==f['inputs_sha256'])
for p,h in f['source_pins'].items():ck('source_pin',sha(Path(p))==h)
for k,r in am.items():
 old=bm[k]['shape_direction'];ck('same_card_count',len(r['components'])==len(old))
 for c,b in zip(r['components'],old):
  ck('whole_measured_evidence_exact',c['measurement_evidence']==b['evidence'])
  ck('candidate_span_exact',c['span']==b['span'])
  ck('shapes_exact',[e for e in c['evidence']if e['task']=='shape']==[e for e in b['evidence']if e['task']=='shape'])
  ck('completion_never_inferred',c['completion']is None)
  if r['kind']=='all_hand_missing':
   ck('missing_no_direction_pass_or_error',not any(e['task']=='direction'and e['state']in ('supported','opposite_direction','difference')for e in c['evidence']))
ck('normal_speed_no_support_lost',not after['normal_supported_regressions'])
ck('shape_direction_error_no_evidence_lost',not after['error_evidence_regressions'])
# Mechanism contracts for left/right/two hands and short visible observations.
c=dict(card='arbitrary_descriptor',span=[0,4],status='supported_features_only',completion=None,
 evidence=[dict(task='direction',state='supported',span=[0,4])])
x=np.zeros((5,280),np.float32);x[:,186:237]=1
for hands in [['L'],['R'],['L','R']]:
 ck('normal_visible',guard_component(c,x,hands)['evidence'][0]['state']=='supported')
 for h in hands:
  y=x.copy();base=9 if h=='L'else 30;y[:,186+base:186+base+21]=0
  out=guard_component(c,y,hands)
  ck('required_hand_missing_unknown',out['evidence'][0]['state']=='observation_unknown')
  ck('body_measurement_preserved',out['measurement_evidence']==c['evidence'])
  if len(hands)==1:
   other='R'if h=='L'else'L'
   ck('unrelated_hand_unaffected',guard_component(c,y,[other])['evidence'][0]['state']=='supported')
 short=copy.deepcopy(c);short['span']=[0,1];short['evidence'][0]['span']=[0,1]
 ck('no_minimum_duration_rejection',guard_component(short,x[:2],hands)['evidence'][0]['state']=='supported')
 bad=copy.deepcopy(c);bad['evidence'][0]['state']='opposite_direction'
 ck('observed_opposite_retained',guard_component(bad,x,hands)['evidence'][0]['state']=='opposite_direction')
summary=dict(changed_evidence_rows=len(after['changes']),normal_support_lost=0,error_evidence_lost=0,
 all_missing_before_direction_supported=sum(e['state']=='supported'for r in before['rows']if r['kind']=='all_hand_missing'for c in r['shape_direction']for e in c['evidence']if e['task']=='direction'),
 all_missing_after_direction_supported=sum(e['state']=='supported'for r in after['rows']if r['kind']=='all_hand_missing'for c in r['components']for e in c['evidence']if e['task']=='direction'),
 all_missing_before_opposite=sum(e['state']=='opposite_direction'for r in before['rows']if r['kind']=='all_hand_missing'for c in r['shape_direction']for e in c['evidence']if e['task']=='direction'),
 all_missing_after_opposite=sum(e['state']=='opposite_direction'for r in after['rows']if r['kind']=='all_hand_missing'for c in r['components']for e in c['evidence']if e['task']=='direction'))
write('VISIBILITY_VERIFIED.json',dict(passed=all(c['passed']for c in checks),total=len(checks),checks=checks,
 failures=[c for c in checks if not c['passed']],summary=summary,independent_KSL_expert=False,
 same_Codex_separate_verification=True,scope='Research comparison hand-evidence guard, not deployed web or new accuracy benchmark'))
print(json.dumps(dict(passed=all(c['passed']for c in checks),checks=len(checks),summary=summary),ensure_ascii=False))
