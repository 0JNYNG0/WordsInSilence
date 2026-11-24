"""Repair evaluation adapter order; reuse every neural result, no new inference."""
from prepare import U,O,read,sha,write,Path
import sys,copy
I=O/'word100_goal_binding_integration_20261114';J=O/'word100_joint_completion_block_20261114';B=O/'word100_action_evidence_block_20261114'
sys.path[:0]=[str(I),str(J),str(B)]
from core_correspondence import core_order
from joint_core import integrate
from feedback_draft import draft
specs=read(O/'word100_action_episode_supervision_20261110/GUIDE_FAMILIES.json')['words']
before=read(U/'AUTO_COMPLETE.json');rows=[];changes=[]
for r in before['rows']:
 rr=copy.deepcopy(r);fs=[s for s in specs[r['clip_id'].split('_')[2]]if s['required']];lookup={c['card']:c for c in r['original_candidates']}
 # Exactly the established G3 full_graph_compare contract: guide-list order
 # defines pair orientation AFTER independently selecting every occurrence.
 rr['original_candidates']=[lookup[s['id']]for s in fs]
 allowed={tuple([a['id'],b['id']])for a,b in zip(fs,fs[1:])if a['families']!=b['families']}
 rr['original_orders']=[p for p in core_order(rr['original_candidates'])if tuple(p['cards'])in allowed]
 ex=[c['body_relation']for c in r['joint_cards']if c['body_relation']is not None]
 rr=integrate(rr,ex)
 for c in rr['joint_cards']:c['drafts']=draft(c,next(s['instruction']for s in fs if s['id']==c['card']))
 if r['original_orders']!=rr['original_orders']:changes.append(dict(clip_id=r['clip_id'],kind=r['kind'],condition=r['condition'],seed=r['seed'],before=r['original_orders'],after=rr['original_orders']))
 rows.append(rr)
write(U/'ASSEMBLY_REPAIR.json',dict(reason='New evaluation adapter appended nonmovement cards after movement cards; interpreting that array order as expected chronology manufactured reverse pairs on normal clips. Restored exact pre-existing G3 guide-pair orientation/adjacency, applied to all sources and conditions. This is an adapter error, not model tuning.',
 code_sha256=sha(Path(__file__)),reference=str(I/'full_graph_compare.py'),reference_sha256=sha(I/'full_graph_compare.py'),before_sha256=sha(U/'AUTO_COMPLETE.json'),changed=changes,neural_calls=0,model_changes=0))
write(U/'AUTO_RESULTS_V2.json',dict(**{k:v for k,v in before.items()if k!='rows'},rows=rows,assembly_repaired=True))
print('saved-result order adapter repaired; rows',len(rows))
