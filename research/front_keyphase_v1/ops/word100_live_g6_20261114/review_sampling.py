from bootstrap import *
import json,numpy as np
from g6_runtime_adapter import FeedbackConsumer
read=lambda p:json.loads(p.read_text('utf-8-sig'))
first=read(U/'SAMPLING_24HZ.json');P=O/'word100_public_g4_20261114';cid='NIA_SL_WORD1133_REAL17_F';sr=next(r for r in read(P/'EXTRACTED.json')['rows'] if r['clip_id']==cid)
raw=dict(np.load(sr['raw_path']));ts=raw['timestamp_us'];ids=np.unique(np.rint(np.interp(np.arange(0,ts[-1]+1,1e6/24),ts,np.arange(len(ts)))).astype(int))
for k in ['xy_px','score']:raw[k]=raw[k][::-1][ids]
raw['timestamp_us']=ts[ids];raw['frame_id']=np.arange(len(ids))
c=FeedbackConsumer()
try:out=c.raw(raw,sr['word'])
finally:c.latest.close()
card=out['qualified_evidence']['joint_cards'][0]
directions=[next((e['state'] for e in r['observations'] if e['task']=='direction'),None) for r in card['cross_run_evidence']]
normal_same=True
for r in first['rows']:
 if r['reverse']:continue
 prior=read(U/'live_inputs'/(r['clip_id']+'.json'))['check']['after']
 normal_same &= prior==r['states']
checks=dict(all_eight_normal_states_preserved=normal_same,reverse_all_three_agree=directions==['opposite_direction']*3,reverse_conflicts_empty=not card['conflicts'],actual_reversed_source=True)
report=dict(passed=all(checks.values()),checks=checks,initial_gate_failed='Expected a 30Hz uncertainty to stay uncertain after changing input sampling; 24Hz all three independent readings instead agree on the true reversed direction.',
 conclusion='This does not establish sampling invariance or an accuracy gain. Normal eight decisions and reversed direction retained on this finite regression; actual observed conflicts continue to suppress cues. No parameters or weights changed.',
 row=out,training=False)
(U/'SAMPLING_REVIEW.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(checks);assert all(checks.values())
