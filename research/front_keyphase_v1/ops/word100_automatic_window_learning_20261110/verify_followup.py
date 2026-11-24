from pathlib import Path
import json,hashlib
from collections import Counter
U=Path(__file__).resolve().parent;P=U.parent/'word100_broad_factorized_learning_20261110'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n','utf-8')
checks=[]
def ck(name,value):checks.append(dict(name=name,passed=bool(value)))
rows=read(U/'ORDER_ONLY_COMPLETE.json')['rows'];original={r['clip_id']+'_'+r['kind']:r for r in read(U/'FULLPATH_COMPLETE.json')['rows'] if r['model']=='learned_core_42'};groups={};changed=[]
ck('78all_conditions',len(rows)==78);ck('first_baseline_only9',sum(r['new_baseline_inference'] for r in rows)==9)
for r in rows:
 key=r['clip_id']+'_'+r['kind'];out=read(r['result']);ck('result_sha',sha(r['result'])==r['sha256']);ck('paths_TCN_windows_preserved',r['path_identical'] and r['tcn_identical'] and r['query_windows_preserved'])
 if r['new_baseline_inference']:before=read(Path(r['result']).parent/'first_baseline.json')
 else:before=read(P/('edited_channels' if r['kind'] in ['right_closed_hand_main','main_reversed','closed_hand_and_reversed'] else 'fullpath')/'coupled_42'/key/'after.json')
 source=read(original[key]['result']);ck('same_input_TCN_path',out['tcn_top3']==source['tcn_top3'] and out['path']==source['path']);ck('all_frames_remain',out['frame_id']==before['frame_id'])
 for old,new in zip(before['cards'],out['cards']):
  ck('shape_evidence_preserved',old.get('shape_evidence')==new.get('shape_evidence'))
  # Prior existing conflict guard can alter state to unknown. Compare saved compact
  # baseline for states; all coordinates/probabilities remain the archivedP model.
  if old.get('motion_evidence'):ck('direction_class_preserved',old['motion_evidence'].get('learned_class')==new['motion_evidence'].get('learned_class'))
 ck('direction_states_preserved',r['before']['motion_states']==r['after']['motion_states']);ck('no_completion_promotion',not r['after']['all_required_similar'] or r['before']['all_required_similar'])
 if r['kind'] in ['full_normal','speed075','speed125','right_closed_hand_main']:ck('normal_speed_shape_only_same',r['before']==r['after'])
 if r['kind'] in ['all_hand_missing','mask_final_event','omit_final_event']:ck('missing_not_pass',not r['after']['all_required_similar'] and 'difference' not in r['after']['display'])
 if r['kind'] in ['main_reversed','closed_hand_and_reversed','right_closed_hand_main']:ck('existing_warning_preserved','difference' in r['after']['display'])
 if r['changed']:changed.append(dict(clip_id=r['clip_id'],kind=r['kind'],cards=r['changed']))
for kind in sorted(set(r['kind'] for r in rows)):
 rr=[r for r in rows if r['kind']==kind];groups[kind]={}
 for stage in ['before','after']:
  counts=Counter()
  for r in rr:
   d=r[stage];key='difference_shown' if 'difference' in d['display'] else 'all_required_similar_no_warning' if d['all_required_similar'] else 'uncertain';counts[key]+=1
  groups[kind][stage]=dict(n=len(rr),**{k:counts[k] for k in ['difference_shown','all_required_similar_no_warning','uncertain']},all_required_cards_similar=sum(r[stage]['all_required_similar'] for r in rr))
for p,h in read(U/'ORDER_ONLY_PLAN_v2.json')['protections'].items():ck('protected:'+p,sha(p)==h)
ck('bounded_new_order_gain',groups['reversed_core_auto_order']['before']['difference_shown']==0 and groups['reversed_core_auto_order']['after']['difference_shown']==3)
write(U/'ORDER_ONLY_VERIFY.json',dict(passed=all(r['passed'] for r in checks),count=len(checks),failed=[r for r in checks if not r['passed']],checks=checks,groups=groups,changed=changed,posthoc=True,first_failure_preserved=bool(list(U.glob('ORDER_ONLY_FAILURE_*.txt'))),caution='Same Codex arithmetic, no independent human expertise. Counterfactuals are synthetic coordinate reversals. all_required_cards_similar alone can coexist with an optional-card order warning; use disjoint displayed-result categories to avoid calling those passed.'))
print(json.dumps(dict(passed=all(r['passed'] for r in checks),count=len(checks),failed=[r for r in checks if not r['passed']][:6],reversed=groups['reversed_core_auto_order']),ensure_ascii=False))
