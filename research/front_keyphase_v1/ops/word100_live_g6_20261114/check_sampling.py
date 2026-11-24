"""24Hz public acquisition stress regression; timestamps are not time-warped."""
from bootstrap import *
import json,numpy as np
from g6_runtime_adapter import FeedbackConsumer
P=O/'word100_public_g4_20261114';read=lambda p:json.loads(p.read_text('utf-8-sig'))
sources=read(P/'EXTRACTED.json')['rows'];consumer=FeedbackConsumer();rows=[]
try:
 for sr in sources:
  for reverse in ([False,True] if sr['word']=='WORD1133' else [False]):
   raw=dict(np.load(sr['raw_path']));ts=raw['timestamp_us'];grid=np.arange(0,ts[-1]+1,1e6/24);ids=np.unique(np.clip(np.rint(np.interp(grid,ts,np.arange(len(ts)))).astype(int),0,len(ts)-1))
   mapping=np.arange(len(ts))[::-1] if reverse else np.arange(len(ts))
   for k in ['xy_px','score']:raw[k]=raw[k][mapping[ids]]
   raw['timestamp_us']=ts[ids];raw['frame_id']=np.arange(len(ids));out=consumer.raw(raw,sr['word'])
   r=dict(clip_id=sr['clip_id'],reverse=reverse,input_frames=len(ids),max_gap_ms=float(np.diff(raw['timestamp_us']).max()/1000),statuses={c['id']:c['display_status'] for c in out['cards']},states={c['id']:c['status'] for c in out['cards']},seconds=out['compute_seconds'])
   rows.append(r);print(json.dumps(r,ensure_ascii=False),flush=True)
finally:consumer.latest.close()
checks=dict(normal_no_correction=all('difference' not in r['statuses'].values() for r in rows if not r['reverse']),reverse_still_detected=next(r for r in rows if r['reverse'] and 'REAL18' in r['clip_id'])['statuses']['brush_arm']=='difference',conflict_still_withheld=next(r for r in rows if r['reverse'] and 'REAL17' in r['clip_id'])['statuses']['brush_arm']=='uncertain')
(U/'SAMPLING_24HZ.json').write_text(json.dumps(dict(passed=all(checks.values()),checks=checks,rows=rows,scope='Public 24Hz stress regression, not proof that arbitrary frame gaps are safe',training=False),ensure_ascii=False,indent=2),encoding='utf-8')
assert all(checks.values()),checks
