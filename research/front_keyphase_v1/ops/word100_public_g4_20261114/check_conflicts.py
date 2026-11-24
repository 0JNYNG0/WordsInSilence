from prepare import U,O,read,sha,write,Path
from conflict_guard import qualify
import collections,copy
J=O/'word100_joint_completion_block_20261114'
write(U/'CONFLICT_CONTRACT.json',dict(scope='Presentation qualification only; no model/threshold/seed/epoch/candidate selection.',reference='Seed42 is pre-existing default, never chosen by G4 scores. All3 hypotheses retained; contradiction vetoes certainty, no majority voting.',
 gates=['All original observations and differences retained','No change to nonconflicting reference cards','Normal developmental corrections not increased','No positive/negative authoritative draft on contradictory readings'],
 regression=str(J/'JOINT_FINAL_V2.json'),regression_sha256=sha(J/'JOINT_FINAL_V2.json'),code_sha256=sha(U/'conflict_guard.py'),no_training=True,
 honesty='Introduced after G4 revealed one disagreement. Rechecking same final inputs is corrective regression, not a fresh independent accuracy result. Raw pre-repair scores retained.'))
datasets={'development':read(J/'JOINT_FINAL_V2.json')['rows'],'public_normal':read(U/'AUTO_RESULTS_V2.json')['rows'],'public_splice':read(U/'CONTROL_COMPLETE.json')['rows'],'public_continuous':read(U/'CONTINUOUS_COMPLETE.json')['rows']}
outputs={};summary={}
for name,rows in datasets.items():
 groups=collections.defaultdict(list)
 for r in rows:groups[r['clip_id'],r['kind'],r.get('condition','g3_development')].append(r)
 out=[];changed=[]
 for key,group in groups.items():
  result=qualify(group);base=next(r for r in group if r['seed']==42)
  for before,after in zip(base['joint_cards'],result['joint_cards']):
   assert before['observations']==after['observations']and before['differences']==after['differences']and before['body_relation']==after['body_relation']
   if after['conflicts']:
    assert all(x['kind']=='uncertain'for x in after['drafts']);changed.append(dict(clip_id=key[0],kind=key[1],condition=key[2],card=after['card'],before=before['state'],after=after['state'],conflicts=after['conflicts']))
   else:assert before['state']==after['state']
  out.append(result)
 outputs[name]=out;summary[name]=dict(scenarios=len(out),changed=changed)
write(U/'CONFLICT_REGRESSION.json',dict(summary=summary,rows=outputs,all_original_evidence_preserved=True,no_neural_calls=True,no_accuracy_improvement_claim=True))
print({k:dict(scenarios=v['scenarios'],changed=len(v['changed']))for k,v in summary.items()})
