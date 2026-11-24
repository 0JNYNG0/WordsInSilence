"""Separate saved-output verification; no training or model inference."""
from pathlib import Path
import json,hashlib,collections,copy
U=Path(__file__).resolve().parent;O=U.parent;G=O/'word100_web_completion_graph_20261111';H=G/'typed_motion_recovery_20261113'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):
 with(U/n).open('x',encoding='utf-8')as f:json.dump(d,f,ensure_ascii=False,indent=2)
def key(r):return r['seed'],r['clip_id'],r['kind']
checks=[]
def ck(n,v,case=None):checks.append(dict(name=n,passed=bool(v),case=case))
full=read(U/'FULL_GRAPH_COMPARISON.json');freeze=read(U/'FULL_GRAPH_FROZEN.json')
old=read(O/'word100_action_completion_binding_20261114/FULLPATH_COMPARISON.json')
om={key(r):r for r in old['rows']};fm={key(r):r for r in full['rows']}
shadow_path=H/'full_bag_rate_rank_learning/changed_span_shadow/RESULTS.json'
shadow=read(shadow_path);sm={key(r):r for r in shadow['rows']}
ck('all_246_conditions_once',len(full['rows'])==len(fm)==246 and set(fm)==set(om)==set(sm))
ck('82_conditions_three_seeds',collections.Counter(r['seed']for r in full['rows'])=={42:82,43:82,44:82})
ck('code_pin',sha(U/'full_graph_compare.py')==freeze['code_sha256'])
ck('core_pin',sha(U/'core_correspondence.py')==freeze['core_sha256'])
for p,h in {**freeze['models'],**freeze['new_input_pins']}.items():ck('frozen_model_input',sha(Path(p))==h,p)
for p,h in read(G/'PROTECTIONS.json').items():ck('protected',sha(Path(p))==h,p)
fresh=read(U/'DYNAMIC_ROLE_OUTPUTS.json')
ck('only_12_new_role_calls',fresh['neural_calls']==len(fresh['rows'])==len(fresh['timings'])==12)
ck('only_dynamic_inferred',all(r['kind']=='dynamic_complete_blocks'for r in fresh['rows']))
normal_pairs=[];combined=[];omission=[]
for k,r in fm.items():
 ck('rank_candidates_exact',r['all_rank_candidates']==om[k]['conditions']['fixed_rank']['cards'],str(k))
 cs={c['card']:c for c in r['cards']};previous=om[k]['conditions']['fixed_rank']['cards']
 for c in previous:
  now=copy.deepcopy(cs[c['card']]);now.pop('role_confirmation')
  ck('full_comparison_extent_preserved',now==c,str(k))
 for c in r['cards']:
  if c['chosen']:
   a,b=c['chosen']['span']
   for s in c['role_confirmation']['matching']+c['role_confirmation']['other_observed_roles']:
    ck('core_inside_candidate',a<=s['span'][0]<=s['span'][1]<=b,str(k))
 for p in r['orders']:
  aa,bb=p['witness_spans'];expected='unobserved'
  if aa and bb:
   if max(x['span'][1]for x in aa)<min(x['span'][0]for x in bb):expected='forward'
   elif max(x['span'][1]for x in bb)<min(x['span'][0]for x in aa):expected='reverse'
   else:expected='ambiguous_overlap'
  ck('independent_arithmetic_order',p['state']==expected,str(k))
  ck('not_whole_completion',p['whole_action_complete']is None and not p['expected_order_used_for_selection'],str(k))
 if r['kind']in ['full_normal','speed075','speed125']:
  normal_pairs.extend(dict(seed=r['seed'],kind=r['kind'],clip_id=r['clip_id'],**p)for p in r['orders'])
  for ref in r['original_references']:
   ck('original_normal_support_retained',ref['before']!='supported'or ref['after']=='forward',str(k))
 if r['kind']=='all_hand_missing':
  ck('all_missing_no_order',all(p['state']=='unobserved'for p in r['orders']),str(k))
  ck('all_missing_no_roles',all(not c['role_confirmation']['matching']for c in r['cards']),str(k))
 # All shape and direction results must refer to the full selected extent, not
 # the narrower role core that happens to make chronology easier to resolve.
 evidence=sm[k]['conditions']['full_bag_rate_rank']
 for c in evidence:
  orig=next(q for q in previous if q['card']==c['card'])
  expected=None if orig['chosen']is None else orig['chosen']['span']
  ck('shape_direction_full_extent',c['span']==expected,str(k))
  ck('no_completion_from_features',c['completion']is None,str(k))
 combined.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],
   candidate_cards=copy.deepcopy(r['cards']),source_order=copy.deepcopy(r['orders']),
   shape_direction=copy.deepcopy(evidence),
   missing_component_cards=[c['card']for c in r['cards']if c['card']not in {e['card']for e in evidence}],
   whole_sign_correct=None,whole_action_complete=None,
   note='Source order, full-span components and terminal goal are separate. No state suppresses another channel.'))
 if r['kind']in ['omit_final_event','mask_final_event']:
  omission.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],
    role_cards_with_matching=[c['card']for c in r['cards']if c['role_confirmation']['matching']],
    source_order=r['orders'],completion=None))
ck('81_normal_speed_pair_observations',len(normal_pairs)==81)
ck('81_forward',all(p['state']=='forward'for p in normal_pairs))
ck('no_reported_normal_regressions',not full['normal_reference_regressions'])
summary=[]
for seed in [42,43,44]:
 for kind in sorted({r['kind']for r in combined}):
  rr=[r for r in combined if r['seed']==seed and r['kind']==kind]
  ev=[e for r in rr for c in r['shape_direction']for e in c['evidence']]
  summary.append(dict(seed=seed,kind=kind,cases=len(rr),
   shape=dict(collections.Counter(e['state']for e in ev if e['task']=='shape')),
   direction=dict(collections.Counter(e['state']for e in ev if e['task']=='direction')),
   order=dict(collections.Counter(p['state']for r in rr for p in r['source_order']))))
write('JOINT_SAVED_COMPARISON.json',dict(rows=combined,summary=summary,
 source_pins={str(U/'FULL_GRAPH_COMPARISON.json'):sha(U/'FULL_GRAPH_COMPARISON.json'),str(shadow_path):sha(shadow_path)},
 purpose='Complete fixed normal/error accounting, reusing previously scored full-span components exactly',
 neural_inference=False,new_training=False,final8_unopened=True,web_changed=False,
 missing_components_are_unknown=True,semantic_accuracy_not_claimed=True))
write('FULL_GRAPH_VERIFIED.json',dict(passed=all(c['passed']for c in checks),checks=checks,total=len(checks),
 failures=[c for c in checks if not c['passed']],normal_speed_pairs=len(normal_pairs),
 new_normal_order_regressions=len(full['normal_reference_regressions']),all_conditions=246,source_scenarios=82,
 combined_results_complete=True,whole_project_complete=False,
 limitations=['81 pairs repeat public recordings across 3 seeds and 3 speeds, not 81 independent clips',
 'Omit/mask controls still admit wrong earlier role evidence; completion is not inferred',
 'Four dynamic block controls are source-edited examples, not natural learner error data',
 'Saved shape/direction comparison contains only movement cards; nonmovement card gaps remain explicit'],
 omission_diagnostics=omission,reviewer='same Codex separate saved-output check, not independent human KSL truth'))
print(json.dumps(dict(passed=all(c['passed']for c in checks),checks=len(checks),failures=[c for c in checks if not c['passed']],
 summaries=[s for s in summary if s['seed']==42]),ensure_ascii=False))
