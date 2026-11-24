"""Final safety/meaning audit of immutable saved outputs; no neural inference."""
from pathlib import Path
import json,hashlib,collections,ast
U=Path(__file__).resolve().parent;G=U/'safety_reconcile';O=U.parent
rd=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def wr(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),'utf-8')
checks=[]
def ck(n,v):checks.append(dict(name=n,passed=bool(v)));assert v,n
def compact(cards):
 ds=[c['display_status'] for c in cards];return dict(display=ds,warning='difference' in ds,similar_no_warning=all(c['display_status']=='similar' for c in cards if c.get('required',True)) and 'difference' not in ds)
rows=rd(G/'FULLPATH_COMPLETE.json')['rows'];first={(r['seed'],r['clip_id'],r['kind']):r for r in rd(U/'FULLPATH_COMPLETE.json')['rows'] if r['mode']=='combined'};summary={};type_errors=[]
ck('234_unique_rows',len(rows)==234 and len({(r['seed'],r['clip_id'],r['kind']) for r in rows})==234)
for r in rows:
 d=rd(r['result']);b=rd(d['baseline']);ck('final_hash',sha(r['result'])==r['sha256']);ck('original_candidate_preserved',sha(r['first_result'])==r['first_sha256']);ck('baseline_preserved',sha(d['baseline'])==d['baseline_sha256']);ck('arithmetic_after',compact(d['cards'])==r['after']);ck('arithmetic_before',compact(b['cards'])==r['before'])
 for c,bc in zip(d['cards'],b['cards']):
  ck('shape_evidence',c.get('shape_evidence')==bc.get('shape_evidence'));ck('motion_evidence',c.get('motion_evidence')==bc.get('motion_evidence'));ck('original_correction_retained',bc['display_status']!='difference' or c['display_status']=='difference');ck('not_expert_wrong_claim',not c.get('educational_wrong',False))
 key=f"{r['seed']}/{r['kind']}";s=summary.setdefault(key,dict(total=0,before_warnings=0,after_warnings=0,before_similar=0,after_similar=0,correct_omission_cards=0,wrong_new_error_kind=0));s['total']+=1;s['before_warnings']+=r['before']['warning'];s['after_warnings']+=r['after']['warning'];s['before_similar']+=r['before']['similar_no_warning'];s['after_similar']+=r['after']['similar_no_warning']
 added=[(c,set(c.get('difference_components',[]))-set(bc.get('difference_components',[]))) for c,bc in zip(d['cards'],b['cards'])]
 if r['kind'] in ['full_normal','speed075','speed125']:ck('normal_speed_cross_no_added_warning',not r['after']['warning'])
 if r['kind'] in ['all_hand_missing','mask_final_event']:
  ck('missing_observation_not_normal',not r['after']['similar_no_warning']);ck('missing_observation_not_called_omission',all('role_omission' not in comp for _,comp in added))
 if r['kind']=='omit_final_event':
  required=[c for c in b['cards'] if c.get('required',True)];expected=required[-1]['id']
  for c,comp in added:
   if 'role_omission' in comp and c['id']==expected:s['correct_omission_cards']+=1
   bad=comp-{'role_omission'}
   if bad or ('role_omission' in comp and c['id']!=expected):type_errors.append(dict(seed=r['seed'],clip_id=r['clip_id'],card=c['id'],added=sorted(comp)));s['wrong_new_error_kind']+=1
 ck('protected_tcn_dtw',d['protected']==rd(r['first_result'])['protected'])
ck('final_omission_error_type_not_order',not type_errors)
for p in U.glob('*.py'):ast.parse(p.read_text('utf-8'));ck('python_syntax',True)
src=(U/'integration.py').read_text('utf-8');ck('no_word_number_runtime_branches','WORD0' not in src and 'WORD1' not in src and 'WORD2' not in src);ck('no_evaluation_kind_in_decision',"['kind']" not in src and 'omit_final_event' not in src and 'reversed_core' not in src)
for p,h in rd(U/'PROTECTIONS.json').items():ck('frozen_models_web_sources',sha(p)==h)
# Pin visual review provenance. Labels below describe examined post-score cases,
# never feed training or modify the frozen ground truth.
visual=rd(U/'CHANGE_REVIEW_INPUTS.json')['rows']
for r in visual:ck('actual_rgb_review_sheet',sha(r['sheet'])==r['sheet_sha256']);ck('source_public_video_unchanged',sha(r['source'])==r['source_sha256'])
wr(U/'CORRECTION_SEMANTICS_AUDIT.json',dict(reviewer='Same Codex original RGB + saved transformed source index audit; not independent KSL expert',post_score=True,reviewed_sheets=visual,findings=['Six final-action-deletion probes genuinely remove the final action from the displayed source sequence; retained previous action and abrupt cut to rest remain visible. These are artificial edits, not naturally performed learner errors.','Primary seed42 original omit WORD0739 record13 falsely labelled face-to-chest motion as later forearm role and issued order correction. Reconciliation abstains because existing phase evidence conflicts; no weight/label/threshold tuning.','New reversed-core direction examples visibly move the forearm downward or bring hands together / brush back toward elbow. Some are suppressed by earlier-phase disagreement; warning count alone is not accuracy.','Residual reverse WORD1133 record13 accepts54..60 while70..77 is treated as partial; saved trajectory changes2.12 relative forearm units over7frames. A reversal-splice/observation discontinuity may contribute; natural-user failure is not established by this synthetic edit.'],limitations=['Pose validity scores are not independent proof of RGB visibility. In real occlusion, inferred wrists may stay confident.','No complete newly held-out natural-error evaluation or expert acceptance boundary.','Sparse67frame role metric covers existing4record11 videos, not every frame of12 normal videos.']))
gain={str(s):dict(omission_before=summary[f'{s}/omit_final_event']['before_warnings'],correct_omission_after=summary[f'{s}/omit_final_event']['correct_omission_cards'],reverse_before=summary[f'{s}/reversed_core_auto_order']['before_warnings'],reverse_after=summary[f'{s}/reversed_core_auto_order']['after_warnings'],remaining_false_similar=summary[f'{s}/reversed_core_auto_order']['after_similar']) for s in [42,43,44]}
wr(U/'FINAL_RESULT.json',dict(status='COMPLETE_LIMITED_OMISSION_GAIN_RESEARCH_ONLY',steps_completed=[1,2,3,4],summary=summary,gain=gain,meets_all_predeclared_adoption_gates=False,web_adopted=False,goal_complete=False,raw_result=str(U/'RESULT.json'),post_score_reconciliation=True,training_private_data=0,normal36_each_seed_preserved=True,old9_each_seed_preserved=True,uncertain18_each_seed_not_omission=True))
wr(U/'FINAL_VERIFICATION.json',dict(integrity_passed=True,checks=len(checks),reviewer='Separate same-Codex saved-output/source audit',independent_human=False,scientific_success='Limited omission-candidate gain; full adoption gates FAIL',new_neural_inferences=0,post_score_repair=True));print(json.dumps(dict(checks=len(checks),gain=gain),ensure_ascii=False))
