from bootstrap import *
import json,time,numpy as np
from g6_runtime_adapter import FeedbackConsumer
P=O/'word100_public_g4_20261114'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
consumer=FeedbackConsumer();rows=[];(U/'live_inputs').mkdir(exist_ok=True)
try:
 for sr in read(P/'EXTRACTED.json')['rows']:
  raw=dict(np.load(sr['raw_path']));out=consumer.raw(raw,sr['word'])
  ref=read(U/'equivalence'/(sr['clip_id']+'__normal.json'))['result']
  before={c['card']:c['state'] for c in ref['joint_cards']};after={c['card']:c['state'] for c in out['qualified_evidence']['joint_cards']}
  guard=np.asarray(out['observation_guard']['hand_quarantined']);check=dict(clip_id=sr['clip_id'],frames=len(raw['frame_id']),hand_quarantined_frames=int(guard.any(1).sum()),before=before,after=after,seconds=out['compute_seconds'],timing=out['timing'],new_corrections=[c['id'] for c in out['cards'] if c['display_status']=='difference'],top3=[c['label'] for c in out['tcn_top3']],dtw_pairs=len(out['path']))
  out.update(source='public_runtime_replay',training_use=False)
  (U/'live_inputs'/(sr['clip_id']+'.json')).write_text(json.dumps(dict(check=check,result=out),ensure_ascii=False,indent=2),encoding='utf-8');rows.append(check)
  print(json.dumps(check,ensure_ascii=False),flush=True)
finally:consumer.latest.close()
checks=dict(eight_public_sources=len(rows)==8,no_new_corrections=all(not r['new_corrections'] for r in rows),all_three_tcn=all(len(r['top3'])==3 for r in rows),all_dtw_paths=all(r['dtw_pairs']>0 for r in rows))
(U/'LIVE_INPUTS.json').write_text(json.dumps(dict(passed=all(checks.values()),checks=checks,rows=rows,training=False),ensure_ascii=False,indent=2),encoding='utf-8')
assert all(checks.values()),checks
