from bootstrap import *
import json,numpy as np
from g6_runtime_adapter import FeedbackConsumer
P=O/'word100_public_g4_20261114';V=O/'word100_visual_feedback_g5_20261114'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sources={r['clip_id']:r for r in read(P/'EXTRACTED.json')['rows']};consumer=FeedbackConsumer();rows=[];(U/'controls').mkdir(exist_ok=True)
try:
 for cid,kind in [('NIA_SL_WORD1133_REAL18_F','continuous_full_reverse'),('NIA_SL_WORD1133_REAL17_F','continuous_full_reverse'),('NIA_SL_WORD0739_REAL17_F','prefix_before_goal'),('NIA_SL_WORD0739_REAL17_F','all_hands_missing')]:
  sr=sources[cid];raw=dict(np.load(sr['raw_path']));display=read(V/'data'/(cid+'__'+kind+'.json'));n=len(display['timestamp_us'])
  for key in ['xy_px','score']:raw[key]=raw[key][np.array(display['source_mapping'])].copy()
  raw['frame_id']=np.arange(n);raw['timestamp_us']=np.asarray(display['timestamp_us'])
  if kind=='all_hands_missing':raw['xy_px'][:,91:]=np.nan
  result=consumer.raw(raw,sr['word']);result.update(source='public_live_controls',synthetic=True,video=display['video'],training_use=False)
  check=dict(id=cid+'__'+kind,kind=kind,statuses={c['id']:c['display_status'] for c in result['cards']},states={c['id']:c['status'] for c in result['cards']},cue_count=len(result['coaching']['cues']),seconds=result['compute_seconds'])
  (U/'controls'/(check['id']+'.json')).write_text(json.dumps(dict(check=check,result=result),ensure_ascii=False,indent=2),encoding='utf-8');rows.append(check);print(json.dumps(check,ensure_ascii=False),flush=True)
finally:consumer.latest.close()
checks=dict(reverse_difference=rows[0]['statuses']['brush_arm']=='difference',conflict_not_wrong=rows[1]['states']['brush_arm']=='evidence_conflict',prefix_not_complete=all(v!='similar' for v in rows[2]['statuses'].values()),missing_no_correction=rows[3]['cue_count']==0)
(U/'LIVE_CONTROLS.json').write_text(json.dumps(dict(passed=all(checks.values()),checks=checks,rows=rows),ensure_ascii=False,indent=2),encoding='utf-8');assert all(checks.values()),checks
