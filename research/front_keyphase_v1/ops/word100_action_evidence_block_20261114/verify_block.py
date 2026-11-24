"""Saved-output arithmetic and source/split audit. No training/inference."""
from pathlib import Path
import json,hashlib,collections,copy
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent;D=U/'state_path';I=O/'word100_goal_binding_integration_20261114';G=O/'word100_web_completion_graph_20261111'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(n,v):
 with(U/n).open('x',encoding='utf-8')as f:json.dump(v,f,ensure_ascii=False,indent=2)
checks=[]
def ck(name,value,case=None):checks.append(dict(name=name,passed=bool(value),case=case))
def key(r):return r['seed'],r['clip_id'],r['kind']
def softmax(v):
 v=np.asarray(v);z=np.exp(v-v.max(-1,keepdims=True));return z/z.sum(-1,keepdims=True)
for directory in [U,U/'metric_recovery',D]:
 cfg=read(directory/'FROZEN.json')
 for p,h in cfg.get('pins',{}).items():ck('frozen_candidate_pin',sha(p)==h,p)
 done=read(directory/'TRAINING_COMPLETE.json')['rows']
 ck('three_fixed_seeds',sorted(r['seed']for r in done)==[42,43,44],directory.name)
 for r in done:ck('saved_model_pin',sha(r.get('path',r.get('model')))==r['sha256'],directory.name+str(r['seed']))
for f in ['EXTENDED_FROZEN.json','AUTO_INTEGRATION_FROZEN.json','ASSEMBLY_FROZEN.json']:
 cfg=read(U/f)
 for group in ['pins','model_pins','source_pins']:
  for p,h in cfg.get(group,{}).items():ck('frozen_comparison_pin',sha(p)==h,p)
protected=read(G/'PROTECTIONS.json')
for p,h in protected.items():ck('protected_original',sha(p)==h,p)
labels=read(D/'LABELS.json');rows=labels['rows'];points=labels['point_rows'];runs=read(D/'TRAINING_COMPLETE.json')['rows']
ck('100_unchanged_pair_targets',len(rows)==100)
ck('26_distinct_public_clips',len({r['clip_id']for r in rows})==26)
ck('fit76_probe24',collections.Counter(r['split']for r in rows)=={'fit':76,'exposed_probe':24})
ck('no_word_input_in_architecture',all(s not in (D/'core.py').read_text('utf-8')for s in ['WORD','clip_id','record_id']))
for r in rows+points:
 rec=int(r['clip_id'].split('_')[3][4:]);allowed=range(2,8)if r['split']=='fit'else[8,9]
 ck('source_record_split',rec in allowed,r['clip_id'])
for run in runs:
 ck('training_pairs_fit_only',all(rows[i]['split']=='fit'and rows[i]['available']for i in run['fit_indices']),run['seed'])
 ck('training_points_fit_only',all(points[i]['split']=='fit'and points[i]['available']for i in run['point_fit_indices']),run['seed'])
 pp=np.load(D/'models'/f"{run['seed']}_predictions.npz")
 delta=pp['endpoint_positions'][:,1]-pp['endpoint_positions'][:,0]
 expected=softmax(np.stack([-delta,np.full_like(delta,run['learned_margin']),delta],1)*run['learned_scale'])
 ck('saved_logits_arithmetic',np.allclose(expected,pp['change'],atol=2e-6),run['seed'])
 for i,r in enumerate(rows):
  if r['freeze']:ck('constant_input_zero_net_change',abs(float(delta[i]))<1e-7 and int(pp['change'][i].argmax())==1,[run['seed'],i])
  if r['reverse']:
   matches=[j for j,s in enumerate(rows)if not s['reverse']and not s['freeze']and(s['clip_id'],s['hand'],s['input_indices'])==(r['clip_id'],r['hand'],r['input_indices'])]
   if matches:ck('reversed_state_path_antisymmetry',np.allclose(pp['change'][i],pp['change'][matches[0]][::-1],atol=2e-6),[run['seed'],i])
# Recompute the reported change accuracy from all saved predictions, not report flags.
arrays=[np.load(D/'models'/f'{s}_predictions.npz')['change']for s in [42,43,44]]
good=[]
for i,r in enumerate(rows):
 if r['split']!='exposed_probe':continue
 v=np.stack([p[i]for p in arrays]);prediction=int(v[0].argmax())if r['available']and(v.argmax(-1)==v[0].argmax()).all()and(v.max(-1)>=.8).all()else None
 good.append(prediction==r['target'])
ck('24_fixed_probe_changes',len(good)==24 and all(good))
# Directly verify that every partial training/probe prefix contains zeros beyond its cut.
z=np.load(D/'arrays.npz')
for i,r in enumerate(rows):
 if r['origin']!='RGB_open_to_bent_partial':continue
 ck('prefix_input_end_equals_cut',r['context_input_end']==r['input_indices'][1],[i,r['clip_id']])
 ctx=z['pairs'][i,0 if r['reverse']else 1]
 future=ctx[:3]if r['reverse']else ctx[4:]
 ck('prefix_missing_future_zero_mask',np.count_nonzero(future)==0,[i,r['clip_id']])
extended=read(U/'EXTENDED_COMPARISON.json');assembled=read(U/'ASSEMBLED_EVIDENCE.json')
ck('27_extended_conditions',len(extended['rows'])==27 and len(assembled['controls'])==27)
ck('23_declared_extended_checks',len(extended['checks'])==23 and all(c['passed']for c in extended['checks']))
for r in extended['rows']:
 e=r['evidence'];positions=np.array(r['positions']);old=e['original_states'];v=np.array(r['available'])
 votes=np.abs(positions[:,:,None]-[0,1,.5]).argmin(-1)
 for k,value in enumerate(e['compatible_states']):
  expected=['four_open','four_closed','curved'][int(votes[0,k])]if v[k]and(votes[:,k]==votes[0,k]).all()else None
  expected=expected if expected==old[k]else None
  ck('state_agreement_independent',value==expected,[r['clip_id'],r['variant'],k])
 if e['observed_shape_change_to_folded']:
  a,b=e['witness_indices'];t=np.asarray(r.get('timestamps_us',[]))
  ck('two_distinct_endpoints',a<b and e['compatible_states'][a]=='four_open'and e['compatible_states'][b]=='four_closed',r['variant'])
  ck('continuity_max_gap',e['maximum_gap_us']<=120000 and v[a:b+1].mean()>=.8,r['variant'])
 if r['prefix_future_excluded']:ck('no_future_after_cut',r['input_source_last_frame']==r['source_frames'][-1],r['clip_id'])
 if r['variant']=='middle_long_gap':ck('long_gap_not_full_transition',not e['observed_shape_change_to_folded'],r['clip_id'])
 if r['variant']=='middle_gap2':ck('short_gap_qualified_or_withheld',e['status']in ['observed_open_to_folded','endpoints_seen_intervening_observation_insufficient'],r['clip_id'])
for r in assembled['controls']:
 e=r['evidence'];ck('control_not_whole_completion',e['complete_action']is None and e['educational_error']is None,r['variant'])
 if r['variant']=='reviewed_partial_prefix':ck('partial_not_false_closing_instruction',e['content_status']=='final_configuration_unconfirmed'and not e['show_as_correction'],r['frames'][-1])
# Independent assembly-preservation and original G2 gates.
full={key(r):r for r in read(I/'FULL_GRAPH_COMPARISON.json')['rows']}
vis={key(r):r for r in read(I/'VISIBILITY_COMPARISON.json')['rows']}
assoc={key(r):r for r in read(I/'ASSOCIATION_SCOPE_COMPARISON.json')['rows']}
current={key(r):r for r in assembled['rows']}
ck('all82_scenarios_3seeds',len(current)==246 and set(current)==set(full))
normal=[];combined=[];allmissing=[];changes=[]
for k,r in current.items():
 ck('all_original_candidates_unchanged',r['original_candidates']==full[k]['cards'],str(k))
 ck('all_original_orders_unchanged',r['original_orders']==full[k]['orders'],str(k))
 ck('all_original_components_unchanged',r['original_qualified_components']==vis[k]['components'],str(k))
 ck('all_original_association_unchanged',r['original_association']==assoc[k]['association'],str(k))
 ck('no_product_or_completion_claim',not r['user_feedback_enabled']and r['whole_action_complete']is None and r['educational_error']is None,str(k))
 if k[2]in ['full_normal','speed075','speed125']:
  normal.extend(r['original_orders'])
 for p in r['qualified_shape_paths']:
  for h in p['hands']:
   ck('path_never_full_action',h['complete_action']is None and not h['show_as_correction'],str(k))
   if not h['association_confirmed']:ck('unconfirmed_clause_not_corrected',not h['show_as_correction'],str(k))
 if k[2]=='closed_hand_and_reversed':
  es=[e for c in r['linked_observations']for e in c['attributable_differences']]
  combined.append(dict(seed=k[0],clip_id=k[1],shape=sum(e['state']=='difference'for e in es),direction=sum(e['state']=='opposite_direction'for e in es)))
 if k[2]=='all_hand_missing':
  es=[e for c in r['linked_observations']for e in c['observations']]
  ok=all(e['state']not in ['supported','difference','opposite_direction']for e in es)and all(p['state']=='unobserved'for p in r['original_orders'])
  allmissing.append(ok);ck('all_missing_no_support_or_correction',ok,str(k))
ck('normal81_source_orders_forward',len(normal)==81 and all(r['state']=='forward'for r in normal))
ck('positive_combined_difference_not_all_unknown',len(combined)==9 and all(r['shape']>=1 for r in combined)and sum(r['direction']for r in combined)>=3)
# Use only the two source-assignment conflicts independently identified BEFORE
# this turn. Other partial-core reversals are not whole-sign error ground truth.
truth=read(G/'reviews/sequence_truth_audit.json')
proven=[r for r in truth['supported_pair_details']if r['category']=='selected_span_conflicts_with_reviewed_source_role_needs_assignment_repair']
prior={key(r):r for r in read(G/'JOINT_CORRESPONDENCE_RESULTS.json')}
sourceaudit=read(I/'CORE_SOURCE_AUDIT.json')['edited_source_rows'];repairs=[]
for seed in [42,43,44]:
 for r in proven:
  k=seed,r['clip_id'],'reversed_core_auto_order'
  before=next(p for p in prior[k]['pairs']if p['cards']==r['cards'])
  after=next(p for p in current[k]['original_orders']if p['cards']==r['cards'])
  a=[q for q in sourceaudit if q['seed']==seed and q['clip_id']==r['clip_id']and q['card']in r['cards']]
  conflict=sum(len(q['conflicting_source_anchors'])for q in a)
  ok=before['state']=='supported'and after['state']=='reverse'and len(a)==2 and conflict==0
  ck('preidentified_source_assignment_repaired',ok,str(k));repairs.append(dict(seed=seed,clip_id=r['clip_id'],before=before['state'],after=after['state'],remaining_source_anchor_conflicts=conflict,passed=ok))
ck('exact_two_source_conflicts_3seeds',len(proven)==2 and len(repairs)==6 and all(r['passed']for r in repairs))
# A second, clean source-block control provides positive ordering evidence.
block=[p for k,r in current.items()if k[2]=='dynamic_complete_blocks'and'WORD0739'in k[1]for p in r['original_orders']if p['cards']==['close_chest','raise_forearm']]
ck('9_full_block_reversals_positive',len(block)==9 and all(p['state']=='reverse'for p in block))
ck('expected_order_never_selects_core',all(not p['expected_order_used_for_selection']for r in current.values()for p in r['original_orders']))
# Logical counterexample: equal endpoints must not erase an interior excursion.
from assemble_evidence import qualify_path
sample=copy.deepcopy(extended['rows'][0]['evidence']);sample.update(content_status='shape_maintenance_observed',compatible_states=['four_open','four_closed','four_open'],original_states=['four_open','four_closed','four_open'],new_configuration_labels=['four_open','four_closed','four_open'])
ck('equal_endpoints_keep_internal_change',qualify_path(sample)['content_status']=='internal_configuration_changes_observed')
fail=[c for c in checks if not c['passed']]
g2=dict(normal_preserved=all(p['state']=='forward'for p in normal)and not read(I/'VISIBILITY_COMPARISON.json')['normal_supported_regressions'],
 false_support_reduced=len(repairs)==6 and all(r['passed']for r in repairs),
 not_abstention_only=all(good)and len(block)==9 and all(p['state']=='reverse'for p in block)and all(r['shape']>=1 for r in combined),
 three_seeds=sorted({r['seed']for r in current.values()})==[42,43,44])
write('VERIFIED.json',dict(passed=not fail,total=len(checks),checks=checks,failures=fail,
 protected_file_count=len(protected),probe_correct=sum(good),probe_n=len(good),normal_speed_order_pairs=len(normal),
 assembly_scenarios=82,assembly_rows=246,extended_cases=27,declared_extended_checks=23,combined=combined,
 source_conflict_repairs=repairs,g2_checks=g2,g2_pass=not fail and all(g2.values())and all(allmissing),
 baseline_scope='G2 source correspondence compared against the two previously audited JOINT_CORRESPONDENCE errors, not all partial reversals. New shape-change probe compares previous change learner. These are distinct metrics.',
 no_model_inference=True,reviewer='Same Codex separately implemented saved-output arithmetic and source audit; not independent human KSL labels',
 limitations=['All development recordings were already exposed; 24/24 is not held-out deployment accuracy',
 'Prefix learned state can be wrong; unchanged endpoint agreement prevents a completion claim, but does not solve all state recognition',
 'Two middle-gap controls differ in observation coverage; no invisible trajectory was filled',
 'Four folded/extended fingers do not prove thumb aperture, contact, exact depth or whole sign correctness',
 'G3 joint semantic completion remains incomplete; no final8 scoring or web rollout'],
 source_pins={str(U/n):sha(U/n)for n in ['ASSEMBLED_EVIDENCE.json','AUTO_DENSE_OUTPUTS.npz','state_path/EVALUATION.json','EXTENDED_COMPARISON.json']},verifier_sha256=sha(Path(__file__))))
print(json.dumps(dict(passed=not fail,checks=len(checks),failures=fail,g2=g2,combined=combined),ensure_ascii=False))
