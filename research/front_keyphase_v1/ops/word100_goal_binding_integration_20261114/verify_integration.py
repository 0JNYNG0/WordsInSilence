from pathlib import Path
import json,hashlib,copy,collections,sys
import numpy as np
from event_trace import ordered_event_trace,combine_components
from observable_goal import assess_observed_goal
U=Path(__file__).resolve().parent;O=U.parent;D=O/'word100_positive_direction_20261109';G=O/'word100_web_completion_graph_20261111';S=O/'word100_goal_state_learning_20261114'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,x):
 with (U/n).open('x',encoding='utf-8')as f:json.dump(x,f,ensure_ascii=False,indent=2)
write('COVERAGE_FROZEN.json',dict(code_sha256=sha(U/'observable_goal.py'),input_sha256=sha(U/'CONTEXT_COMPARISON.json'),
 max_gap_us=120000,source='unchanged maximum gap of existing directional observer',threshold_tuning=False,
 claim='Two known endpoints across unobserved motion are endpoints only. No whole-action/educational correctness.',
 input_scope='93 cached exposed public cases plus explicit synthetic contract controls',no_model_inference=True))
checks=[]
def ck(name,ok,detail=None):checks.append(dict(name=name,passed=bool(ok),detail=detail))
integ=read(U/'INTEGRATED_COMPARISON.json');raw=read(U/'CONTEXT_COMPARISON.json');local={r['case']:r for r in read(O/'word100_action_linkage_repair_20261113/LOCAL_DIRECTION_COMPARISON_V2.json')['cases']}
updated=[]
for row in raw['rows']:
 z=np.load(D/'predictions'/(row['case']+'.npz'));ts=z['timestamp_us'];out=copy.deepcopy(row)
 for c in out['components']:
  c['events']=[assess_observed_goal(e,ts)for e in c['events']]
  for e in c['events']:
   a,b=e['span'];ck('native_frames',e['source_frames']==[int(z['frame_id'][a]),int(z['frame_id'][b])],row['case'])
   ck('no_completion_claim',e['semantic_completion']is None and e['educational_error']is None)
   for i in e['witness_indices']:ck('witness_in_actual_span',a<=i<=b)
 updated.append(out)
for row in integ['rows']:
 old=local[row['case']];ck('existing_cues_exact',row['cues']==old['current_cues'],row['case'])
 ck('existing_card_states_exact',row['old_card_statuses']==old['current_states'],row['case'])
 if row['normal']:ck('normal_no_correction',not row['cues'],row['case'])
for row in integ['role_rows']:
 old=next(x for x in read(G/'LINKER_RESULTS.json')if (x['clip_id'],x['kind'],x['seed'])==(row['clip_id'],row['kind'],row['seed']))
 occ=old['methods']['old_learned_roles_independent']['occurrences'];events=row['trace']['events']
 ck('all_independent_occurrences_exact',[{k:v for k,v in e.items()if k!='event_id'}for e in events]==occ)
 ck('old_midpoint_support_exact',row['original_midpoint_pairs']==old['baseline_pairs'])
 ck('old_independent_support_exact',row['original_independent_pairs']==old['methods']['old_learned_roles_independent']['pairs'])
 if row['kind']=='all_hand_missing':ck('all_missing_no_events',not events)
 # Permuting requested guide order cannot change the events being observed.
 rev=[dict(cards=p['cards'][::-1],families=p['families'][::-1])for p in row['original_independent_pairs']]
 ck('guide_order_does_not_choose_events',ordered_event_trace(occ,rev)['events']==events)

def event(states,available=None):
 sys.path.insert(0,str(S));from state_model import transition_evidence
 av=[s is not None for s in states]if available is None else available
 e=transition_evidence(states,['start','goal'],av)
 return dict(span=[0,len(states)-1],states=states,available=av,evidence=e,witness_indices=e['witness_indices'],hint={})
controls=[('short_normal',['start','goal'],[0,33333],'ordered_state_change_observed'),
 ('slow_normal',['start']*8+['goal']*8,list(range(0,16*66667,66667)),'ordered_state_change_observed'),
 ('brief_uncertainty',['start',None,'goal'],[0,33333,66667],'ordered_state_change_observed'),
 ('long_uncertainty',['start']+[None]*8+['goal'],list(range(0,10*33333,33333)),'endpoint_pair_only'),
 ('all_missing',[None]*4,[0,33333,66667,99999],'observation_insufficient'),
 ('only_start',['start']*4,[0,33333,66667,99999],'observed_before_goal'),
 ('only_end',['goal']*4,[0,33333,66667,99999],'goal_pose_only'),
 ('missing_tail',['start',None,None,None],[0,33333,66667,99999],'observation_insufficient'),
 ('reversed',['goal','goal','start','start'],[0,33333,66667,99999],'order_not_supported')]
control_results=[]
for name,states,ts,expected in controls:
 e=assess_observed_goal(event(states),ts);ck(name,e['evidence']['status']==expected)
 ck(name+'_no_educational_error',e['educational_error']is None)
 control_results.append(dict(name=name,status=e['evidence']['status'],synthetic=True,learner_error_truth=False))
# Positive and conflicting components at distinct locations may not cancel.
e=event(['start','goal']);m=[dict(span=[10,12],state='supported'),dict(span=[0,1],state='contradicted')]
c=combine_components(e,m,[]);ck('opposite_not_overwritten_by_goal',c['state']=='component_difference_observed' and not c['direction_supported'])
c=combine_components(e,[dict(span=[10,12],state='supported')],[]);ck('unrelated_direction_not_borrowed',not c['direction_supported'])
c=combine_components(e,[],[dict(feature=1,contradicted_spans=[[0,1]])]);ck('wrong_shape_not_overwritten_by_goal',c['state']=='component_difference_observed')
for p,h in read(S/'TRAINING_FROZEN.json')['protected'].items():ck('protected_hash',sha(Path(p))==h,str(p))
for p,h in read(U/'INTEGRATION_FROZEN.json')['pins'].items():ck('fixed_input_hash',sha(Path(p))==h,str(p))
summary=[]
for card in ['brush_arm','spread_hands']:
 rr=[c for r in updated if r['normal']for c in r['components']if c['card']==card]
 summary.append(dict(card=card,normal_cases=len(rr),visible_transition=sum(any(e['evidence']['ordered_transition_observed']for e in c['events'])for c in rr),
   unresolved=sum(not any(e['evidence']['ordered_transition_observed']for e in c['events'])for c in rr)))
write('COVERAGE_COMPARISON.json',dict(rows=updated,normal_summary=summary,controls=control_results,new_neural_inference=False))
write('INTEGRATION_VERIFIED.json',dict(passed=all(c['passed']for c in checks),checks=checks,total=len(checks),
 failures=[c for c in checks if not c['passed']],normal_summary=summary,
 reviewer='same root Codex separate cached arithmetic/contract verification; not independent KSL annotator',
 whole_G2_pass=False,graph_complete=False))
print(json.dumps(dict(passed=all(c['passed']for c in checks),checks=len(checks),failures=[c for c in checks if not c['passed']],normal_summary=summary),ensure_ascii=False))
