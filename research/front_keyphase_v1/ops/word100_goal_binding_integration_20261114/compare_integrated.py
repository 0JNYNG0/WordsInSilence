from pathlib import Path
import json,hashlib,collections,copy,time
from event_trace import ordered_event_trace,combine_components
U=Path(__file__).resolve().parent;O=U.parent;G=O/'word100_web_completion_graph_20261111';D=O/'word100_positive_direction_20261109'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):
 with (U/n).open('x',encoding='utf-8')as f:json.dump(d,f,ensure_ascii=False,indent=2)
paths=[U/'event_trace.py',U/'CONTEXT_COMPARISON.json',U/'INITIAL_COMPARISON.json',G/'LINKER_RESULTS.json',G/'GATES.json',G/'EVALUATION_CONTRACT_V2.json',O/'word100_action_linkage_repair_20261113/LOCAL_DIRECTION_COMPARISON_V2.json']
write('INTEGRATION_FROZEN.json',dict(pins={str(p):sha(p)for p in paths},
 policy='No original predictions/cues removed. Add observation ledger, not semantic verdict. Only co-located component evidence is combined.',
 gates=['Normal current-web cues unchanged','Every existing shape/direction difference retained','No missing endpoint becomes omission','Original per-seed order support retained','Source-conflicting selection unresolved remains G2 failure'],
 score_scope='Existing exposed public inputs only; no new detector/encoder inference/training, no final8',
 order_truth='All event pairs reported. Reverse pair is an observed temporal relationship, not necessarily a sign error.',
 no_gate_waiver=True))
current={r['case']:r for r in read(paths[-1])['cases']};first={r['case']:r for r in read(U/'INITIAL_COMPARISON.json')['rows']if r['source']=='web'}
rows=[];t0=time.perf_counter()
for r in read(U/'CONTEXT_COMPARISON.json')['rows']:
 case=r['case'];old=read(D/'final_decisions'/(case+'.json'))['after'];bycard={c['id']:c for c in old['cards']};cur=current[case]
 motion={m['card']:m['evidence']['events']for m in cur['motion']};components=[]
 for c in r['components']:
  before=next(x for x in first[case]['components']if x['card']==c['card'])
  joined=[combine_components(e,motion.get(c['card'],[]),bycard[c['card']].get('shape_evidence',[]))for e in c['events']]
  components.append(dict(card=c['card'],before_transition_observed=any(e['evidence']['ordered_transition_observed']for e in before['events']),
      after_transition_observed=any(e['evidence']['ordered_transition_observed']for e in c['events']),events=joined,
      before_event_count=len(before['events']),after_event_count=len(c['events'])))
 rows.append(dict(case=case,normal=cur['normal'],components=components,cues=copy.deepcopy(cur['current_cues']),
   old_cues_preserved=True,old_card_statuses=cur['current_states'],semantic_completion=None,educational_wrong=None))
role=[]
for r in read(G/'LINKER_RESULTS.json'):
 old=r['methods']['old_learned_roles_independent'];trace=ordered_event_trace(old['occurrences'],old['pairs'])
 assert [{k:v for k,v in e.items()if k!='event_id'}for e in trace['events']]==old['occurrences']
 # Keep baseline evidence rather than relabelling midpoint/extent as same metric.
 role.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],trace=trace,
       original_independent_pairs=old['pairs'],original_midpoint_pairs=r['baseline_pairs'],all_events_preserved=True))
summary=[]
for card in ['brush_arm','spread_hands']:
 for normal in [True,False]:
  cc=[c for r in rows if r['normal']==normal for c in r['components']if c['card']==card]
  summary.append(dict(card=card,normal=normal,cases=len(cc),before=sum(c['before_transition_observed']for c in cc),after=sum(c['after_transition_observed']for c in cc),
      improved=sum(not c['before_transition_observed']and c['after_transition_observed']for c in cc),regressed=sum(c['before_transition_observed']and not c['after_transition_observed']for c in cc)))
normals=[r for r in rows if r['normal']];diff=[r for r in rows if r['cues']]
write('INTEGRATED_COMPARISON.json',dict(rows=rows,role_rows=role,summary=summary,
 normal_cases=len(normals),normal_new_corrections=0,prior_cue_cases_preserved=len(diff),
 goal_transition_is_not_error_accuracy=True,whole_semantic_completion=None,seconds=time.perf_counter()-t0,
 gate_status='G2_REWORK_SOURCE_ASSIGNMENT_AND_COMPLETION_UNRESOLVED',web_adopted=False,
 unchanged_existing_predictions=True))
print(json.dumps(dict(summary=summary,normal_cases=len(normals),cue_cases=len(diff),role_rows=len(role)),ensure_ascii=False))
