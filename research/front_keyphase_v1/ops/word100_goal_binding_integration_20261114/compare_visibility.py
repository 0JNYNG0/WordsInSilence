from pathlib import Path
import json,hashlib,collections
import numpy as np
from component_visibility import guard_component
U=Path(__file__).resolve().parent;O=U.parent;G=O/'word100_web_completion_graph_20261111';H=G/'typed_motion_recovery_20261113'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):
 with(U/n).open('x',encoding='utf-8')as f:json.dump(d,f,ensure_ascii=False,indent=2)
rows=read(U/'JOINT_SAVED_COMPARISON.json')['rows']
pm={(r['clip_id'],r['kind']):r for r in read(H/'null_candidate_learning/evaluation/COMPARISON.json')['predictions']if r['seed']==42}
families=read(O/'word100_action_episode_supervision_20261110/GUIDE_FAMILIES.json')['words']
pins={p['source']:p['source_sha256']for p in pm.values()}
write('VISIBILITY_FROZEN.json',dict(code_sha256=sha(U/'component_visibility.py'),compare_sha256=sha(U/'compare_visibility.py'),
 inputs_sha256=sha(U/'JOINT_SAVED_COMPARISON.json'),source_pins=pins,
 contract='Required hand-root/MCP visibility; .8 coverage inherited from existing component eligibility, not tuned to these comparisons',
 preserves=['full candidate','all original measurement evidence','shape outputs','model weights','unqualified sequence'],
 gates=['No all-hand-missing direction pass/correction','Normal/speed supported directions preserved','Wrong-shape/opposite direction evidence retained when hands observed'],
 no_training=True,no_inference=True,no_final8=True))
xs={};out=[];changes=[];normal_reg=[];error_reg=[]
for r in rows:
 sid=r['clip_id'],r['kind'];p=pm[sid]
 if sid not in xs:
  assert sha(Path(p['source']))==p['source_sha256'];xs[sid]=np.load(p['source'])['x']
 specs={c['id']:c for c in families[r['clip_id'].split('_')[2]]};cs=[]
 for c in r['shape_direction']:
  after=guard_component(c,xs[sid],specs[c['card']]['hands']);cs.append(after)
  for before,e in zip(c['evidence'],after['evidence']):
   if before['state']==e['state']:continue
   detail=dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],card=c['card'],task=e['task'],
      before=before['state'],after=e['state'],visibility=after['hand_position_observation'])
   changes.append(detail)
   if r['kind']in ['full_normal','speed075','speed125']and before['state']=='supported':normal_reg.append(detail)
   if r['kind']in ['right_closed_hand_main','main_reversed','closed_hand_and_reversed']and before['state']in ['difference','opposite_direction']:error_reg.append(detail)
 out.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],components=cs,whole_action_complete=None))
summary=[]
for seed in [42,43,44]:
 for kind in sorted({r['kind']for r in out}):
  es=[e for r in out if r['seed']==seed and r['kind']==kind for c in r['components']for e in c['evidence']]
  summary.append(dict(seed=seed,kind=kind,shape=dict(collections.Counter(e['state']for e in es if e['task']=='shape')),
   direction=dict(collections.Counter(e['state']for e in es if e['task']=='direction'))))
write('VISIBILITY_COMPARISON.json',dict(rows=out,changes=changes,normal_supported_regressions=normal_reg,
 error_evidence_regressions=error_reg,summary=summary,web_adopted=False,new_neural_calls=0))
print(json.dumps(dict(changed=len(changes),normal_regressions=normal_reg,error_regressions=error_reg,
 summary=[s for s in summary if s['seed']==42]),ensure_ascii=False))
