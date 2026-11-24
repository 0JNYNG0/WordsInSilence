from pathlib import Path
import json,hashlib,copy
from joint_core import joint_sequence,integrate
from feedback_draft import draft
U=Path(__file__).resolve().parent;O=U.parent;B=O/'word100_action_evidence_block_20261114'
read=lambda p:json.loads(p.read_text('utf-8-sig'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):
 with(U/n).open('x',encoding='utf-8')as f:json.dump(d,f,ensure_ascii=False,indent=2)
old=read(U/'JOINT_COMPARISON.json');base=read(B/'ASSEMBLED_EVIDENCE.json');guide=read(O/'word100_action_episode_supervision_20261110/GUIDE_FAMILIES.json')['words']
write('ASSEMBLY_FROZEN_V2.json',dict(reason='Distinguish unsupported repetition/component ownership from missing observation. Long visibility gap cannot be named a goal-only posture. Uncertain action ownership also withholds order coaching.',model_changes=False,training=False,pins={str(p):sha(p)for p in[Path(__file__),U/'joint_core.py',U/'feedback_draft.py',U/'JOINT_COMPARISON.json']}))
rows=[]
for r,b in zip(old['rows'],base['rows']):
 extra=[copy.deepcopy(c['body_relation'])for c in r['joint_cards']if c['body_relation']]
 for ex in extra:
  if ex['joint']:
   j=ex['joint'];shape=next(p for p in b['qualified_shape_paths']if p['card']==ex['card'])['hands'][0];s=dict(shape['raw'],content_status=shape['content_status'])
   ex['joint']=joint_sequence(s,j['region_states'],j['available'],j['timestamps_us'],j['source_frames'],s['association_confirmed'])
 out=integrate(b,extra);spec={c['id']:c for c in guide[r['clip_id'].split('_')[2]]}
 for c in out['joint_cards']:c['feedback_drafts']=draft(c,spec[c['card']]['instruction'])
 rows.append(out)
controls=[]
for r in old['controls']:
 j=r['joint'];s=r['old_shape_evidence'];out=copy.deepcopy(r);out['joint']=joint_sequence(s,j['region_states'],j['available'],j['timestamps_us'],j['source_frames']);controls.append(out)
write('JOINT_FINAL_V2.json',dict(rows=rows,controls=controls,region_inference_repeated=False,source=str(U/'JOINT_COMPARISON.json'),source_sha256=sha(U/'JOINT_COMPARISON.json'),web_enabled=False,whole_sign_accuracy=None))
print('Saved evidence reassembled, no neural inference.')
