from pathlib import Path
import json,hashlib,collections,copy
import numpy as np
from core_correspondence import attach_role_core,core_order
U=Path(__file__).resolve().parent;O=U.parent;G=O/'word100_web_completion_graph_20261111';H=G/'typed_motion_recovery_20261113';R=O/'word100_candidate_role_learning_20261111'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):
 with (U/n).open('x',encoding='utf-8')as f:json.dump(d,f,ensure_ascii=False,indent=2)
files=[U/'core_correspondence.py',O/'word100_action_completion_binding_20261114/FULLPATH_COMPARISON.json',G/'LINKER_RESULTS.json',R/'EVALUATION_INPUTS.json',H/'null_candidate_learning/evaluation/COMPARISON.json']
guide_path=O/'word100_action_feedback_demo_20261108/CARDS.json'
guides=read(guide_path)['words']
write('CORE_FROZEN.json',dict(pins={str(p):sha(p)for p in files},
 rationale='Candidate extent overlap and order of independently observed role cores are different measurements. Preserve both and all candidates.',
 prior='Frozen full_bag_rate_rank selected candidates AND frozen original independent learned-role occurrences',guide_contract_sha256=sha(guide_path),
 selection='All intersecting same-family role runs; no desired chronology/direction/shape, no changed threshold',
 evaluation='Previously exposed normal/speed/all-hand-missing/edited errors; retain original midpoint and role success casewise. No equivalence claimed for extent/core metrics.',
 pass_requires=['No original normal role or midpoint supported pair loses supported evidence','Every fixed rank candidate and comparison input retained','No all-hand-missing positive role cores','Positive changed wrong-role evidence against independent source truth'],
 no_g2_gate_waiver=True,no_final8=True,no_neural_inference=True))
role={(r['seed'],r['clip_id'],r['kind']):r for r in read(G/'LINKER_RESULTS.json')}
inputs={(r['clip_id'],r['kind']):r for r in read(R/'EVALUATION_INPUTS.json')['rows']}
rank=read(files[1])['rows'];parent=read(files[-1]);pm={(r['clip_id'],r['kind']):r for r in parent['predictions']if r['seed']==42}
out=[];excluded=[];checks=[]
for r in rank:
 key=(r['seed'],r['clip_id'],r['kind'])
 if key not in role:
  excluded.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],reason='No saved independent role output for new block control; cannot invent it'));continue
 p=pm[r['clip_id'],r['kind']];ii=inputs[r['clip_id'],r['kind']]
 # Different candidate feature matrices are fine; the native observation input
 # and timestamp sequence must be identical before index intersections.
 checks.append(dict(name='same_input_hash',passed=sha(Path(p['source']))==ii['sha256']==sha(Path(ii['path']))))
 current=copy.deepcopy(r['conditions']['fixed_rank']);old=role[key];occ=old['methods']['old_learned_roles_independent']['occurrences']
 cards=attach_role_core(current['cards'],occ);orders=core_order(cards)
 checks.append(dict(name='chosen_unchanged',passed=[c['chosen']for c in cards]==[c['chosen']for c in current['cards']]))
 pairs={tuple(x['cards']):x for x in orders}
 references=[]
 # Only pairs whose two families are in this movement candidate scope apply.
 ids={c['card']for c in cards}
 for ref in old['methods']['old_learned_roles_independent']['pairs']:
  if set(ref['cards'])<=ids:
   new=pairs.get(tuple(ref['cards']));references.append(dict(reference='original_independent',cards=ref['cards'],before=ref['state'],after=new['state']if new else None))
 # Old midpoint uses positional guide IDs, so map through the actual original
 # source card list, never assume movement-only index equals guide index.
 full=guides[r['clip_id'].split('_')[2]]
 group_to_id={c['groups'][0][0]:c['id']for c in full}
 for ref in old['baseline_pairs']:
  names=[group_to_id[i]for i in ref['guides']]
  if set(names)<=ids:
   new=pairs.get(tuple(names));references.append(dict(reference='original_midpoint',cards=names,before=ref['state'],after=new['state']if new else None))
 out.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],candidates=cards,extent_order=current['order'],core_order=orders,original_references=references,
   full_action_completion=None,all_candidates_preserved=True))
summary=[];regressions=[]
for seed in [42,43,44]:
 for kind in sorted({r['kind']for r in out}):
  rr=[r for r in out if r['seed']==seed and r['kind']==kind]
  summary.append(dict(seed=seed,kind=kind,cases=len(rr),core_order=dict(collections.Counter(p['state']for r in rr for p in r['core_order'])),
    role_confirmation=dict(collections.Counter(c['role_confirmation']['state']for r in rr for c in r['candidates']))))
for r in out:
 if r['kind']in ['full_normal','speed075','speed125']:
  for ref in r['original_references']:
   if ref['before']=='supported' and ref['after']!='forward':regressions.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],comparison=ref))
write('CORE_COMPARISON.json',dict(rows=out,excluded=excluded,checks=checks,summary=summary,normal_reference_regressions=regressions,
 no_neural_inference=True,whole_G2_pass=False,reason='Source-role conflict and normal preservation must be separately audited; this new comparison alone cannot promote the model.'))
print(json.dumps(dict(input_contract_pass=all(c['passed']for c in checks),cases=len(out),excluded=len(excluded),regressions=regressions,normal=[s for s in summary if s['kind']in ['full_normal','speed075','speed125','all_hand_missing']]),ensure_ascii=False))
