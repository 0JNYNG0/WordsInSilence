"""Complete the fixed public source-control comparison, including all clauses.

Existing 234 role outputs are reused. Only the four previously unscored block
controls receive the original frozen role head, once per seed (12 calls).
"""
from pathlib import Path
import json,hashlib,copy,collections,importlib.util,time
import numpy as np
import torch
from core_correspondence import attach_role_core,core_order
U=Path(__file__).resolve().parent;O=U.parent;G=O/'word100_web_completion_graph_20261111';H=G/'typed_motion_recovery_20261113';R=O/'word100_candidate_role_learning_20261111'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):
 with (U/n).open('x',encoding='utf-8')as f:json.dump(d,f,ensure_ascii=False,indent=2)
families=read(O/'word100_action_episode_supervision_20261110/GUIDE_FAMILIES.json')['words']
link={(r['seed'],r['clip_id'],r['kind']):r for r in read(G/'LINKER_RESULTS.json')}
rank=read(O/'word100_action_completion_binding_20261114/FULLPATH_COMPARISON.json')['rows']
pm={(r['clip_id'],r['kind']):r for r in read(H/'null_candidate_learning/evaluation/COMPARISON.json')['predictions']if r['seed']==42}
spec=importlib.util.spec_from_file_location('old_occurrence_contract',G/'binding.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
inputs={k:v for k,v in pm.items()if k[1]=='dynamic_complete_blocks'}
write('FULL_GRAPH_FROZEN.json',dict(code_sha256=sha(U/'full_graph_compare.py'),core_sha256=sha(U/'core_correspondence.py'),
 models={str(R/'runs'/str(s)/'model.pt'):sha(R/'runs'/str(s)/'model.pt')for s in [42,43,44]},
 new_input_pins={p['source']:p['source_sha256']for p in inputs.values()},
 comparison='82 previously exposed scenarios x3 seeds; do not drop 4 dynamic block controls',
 nonmovement='All original independent role occurrences, no single expected-order-selected interval',
 movement='Fixed rank candidates unchanged; independent matching cores used only for order observation',
 correctness='No completion or educational error generated. Source-controlled swaps are not real learner errors.',
 thresholds_unchanged=True,final8_unopened=True))
torch.set_num_threads(2);fresh={};timings=[]
for seed in [42,43,44]:
 ck=torch.load(R/'runs'/str(seed)/'model.pt',map_location='cpu',weights_only=True)
 net=torch.nn.Sequential(torch.nn.Linear(684,96),torch.nn.GELU(),torch.nn.Linear(96,11))
 net.load_state_dict({k[5:]:v for k,v in ck.items()if k.startswith('role.')});net.eval()
 for (cid,kind),p in inputs.items():
  assert sha(Path(p['source']))==p['source_sha256'];z=np.load(p['source']);v=z['x'][:,186:237]>.5
  right=v[:,[30]+list(range(35,51))].all(1);left=v[:,[9]+list(range(14,30))].all(1)
  t=time.perf_counter()
  with torch.inference_mode():logits=net(torch.tensor(z['z']));logits[:,10]=-1e4;prob=logits.softmax(-1).numpy()
  timings.append(dict(seed=seed,clip_id=cid,seconds=time.perf_counter()-t))
  occ=[]
  for f in sorted({f for c in families[cid.split('_')[2]]if c['required']for f in c['families']}):
   occ+=mod.bind_occurrences(prob,right if f in [1,2,3]else right&left,z['ts'],[f],.65)
  fresh[seed,cid,kind]=dict(occurrences=sorted(occ,key=lambda e:(e['a'],e['b'],e['family'])),role_probabilities=prob.tolist())
write('DYNAMIC_ROLE_OUTPUTS.json',dict(rows=[dict(seed=k[0],clip_id=k[1],kind=k[2],**v)for k,v in fresh.items()],timings=timings,neural_calls=12,original_encoder_reused=True))
out=[];regressions=[]
for r in rank:
 key=(r['seed'],r['clip_id'],r['kind']);old=link.get(key);fs=[f for f in families[r['clip_id'].split('_')[2]]if f['required']]
 occ=old['methods']['old_learned_roles_independent']['occurrences']if old else fresh[key]['occurrences']
 current=copy.deepcopy(r['conditions']['fixed_rank']);cs={c['card']:c for c in attach_role_core(current['cards'],occ)}
 allcards=[]
 for f in fs:
  if f['id']in cs:allcards.append(cs[f['id']]);continue
  support=[dict(family=e['family'],span=[e['a'],e['b']],original_role_span=[e['a'],e['b']])for e in occ if e['family']in f['families']]
  allcards.append(dict(card=f['id'],family=f['families'][0],chosen=None,source='all_independent_nonmovement_roles',
   role_confirmation=dict(matching=support,other_observed_roles=[],state='matching_role_observed'if support else'no_role_confirmation',semantic_completion=None)))
 # Adjacent distinct source requirements; all possible event observations are
 # retained, including reverse/extra events. Expected order is checked only now.
 allowed={tuple([a['id'],b['id']])for a,b in zip(fs,fs[1:])if a['families']!=b['families']}
 orders=[p for p in core_order(allcards)if tuple(p['cards'])in allowed]
 refs=[]
 if old:
  for p in old['methods']['old_learned_roles_independent']['pairs']:
   after=next(q['state']for q in orders if q['cards']==p['cards'])
   refs.append(dict(reference='original_independent',cards=p['cards'],before=p['state'],after=after))
  # The core-only audit already crosschecked global group IDs for all midpoint
  # pairs in its scope. Additional old midpoint pairs map using frozen CARDS.
  guides=read(O/'word100_action_feedback_demo_20261108/CARDS.json')['words'][r['clip_id'].split('_')[2]]
  names={c['groups'][0][0]:c['id']for c in guides}
  for p in old['baseline_pairs']:
   ids=[names[i]for i in p['guides']];after=next(q['state']for q in orders if q['cards']==ids)
   refs.append(dict(reference='original_midpoint',cards=ids,before=p['state'],after=after))
 if r['kind']in ['full_normal','speed075','speed125']:
  regressions +=[dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],**x)for x in refs if x['before']=='supported'and x['after']!='forward']
 out.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],cards=allcards,orders=orders,original_references=refs,
   original_extent_order=current['order'],all_rank_candidates=current['cards'],educational_correctness=None))
summary=[]
for seed in [42,43,44]:
 for kind in sorted({r['kind']for r in out}):
  rr=[r for r in out if r['seed']==seed and r['kind']==kind]
  summary.append(dict(seed=seed,kind=kind,cases=len(rr),order=dict(collections.Counter(p['state']for r in rr for p in r['orders']))))
write('FULL_GRAPH_COMPARISON.json',dict(rows=out,summary=summary,normal_reference_regressions=regressions,
 all_82_conditions_included=True,whole_graph_complete=False,normal_error_comparison_complete=True,
 remaining='Occurrence source identity and completion remain separate from role-core chronology. Formal gates require verification.'))
print(json.dumps(dict(cases=len(out),new_role_calls=len(timings),regressions=regressions,summary=[s for s in summary if s['seed']==42]),ensure_ascii=False))
