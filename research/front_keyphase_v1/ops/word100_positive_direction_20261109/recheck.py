"""A documented development correction; reuse all predictions, no new scoring run."""
from setup import *
import numpy as np,time
from positive_evidence import refine
if __name__=='__main__':
 assert not (U/'RECHECK.json').exists()
 write(U/'CORRECTION_FROZEN.json',dict(reason='Public normal crossed hands: neural R_closed vs R_open was wrong at frames54-57 of ID13. RGB reviewed; model disagreement is not user error.',
  change='Require projected four-finger consistency only before emitting a handshape correction cue; contradictory model/geometry remains unknown, never promoted to success.',
  projected_bend='At least three four-finger chains have tip-to-MCP/path ratio below .85',projected_straight='All four ratios above .90',
  scope='Conservative necessary conditions, not anatomical or educational boundaries. Same keypoints, not independent evidence.',
  exposed_development=True,model_inference=False,model_training=False,code_sha256=sha(U/'positive_evidence.py')))
 result=read(U/'COMPARISON.json');rows=[]
 for r in result['rows']:
  z=np.load(r['prediction']);b=read(r['decision']);t=time.perf_counter();a=refine(b['before'],z['attributes'],z['x'],z['timestamp_us'],z['frame_id'],z['phase']);ms=(time.perf_counter()-t)*1000
  # Preserve the original guard outcome (the source has already been quarantined).
  for before,after in zip(b['before']['cards'],a['cards']):
   if before.get('observation_quarantine_overlap'):after.update(status='unknown',display_status='uncertain',message='이 구간의 손을 일부 확인하기 어려워요. 틀렸다는 뜻은 아니에요.')
  p=U/'final_decisions'/Path(r['decision']).name;p.parent.mkdir(exist_ok=True);write(p,dict(before=b['before'],after=a,milliseconds=ms))
  rows.append({**r,'decision':str(p),'after':[q['status'] for q in a['cards']],'components':[q['difference_components'] for q in a['cards']],
   'orange_review_cues':sum(q['display_status']=='difference' for q in a['cards']),'milliseconds':ms})
 summary={}
 for kind in sorted({r['kind'] for r in rows}):
  rs=[r for r in rows if r['kind']==kind]
  summary[kind]=dict(clips=len(rs),before_supported=sum(s=='supported' for r in rs for s in r['before']),after_supported=sum(s=='supported' for r in rs for s in r['after']),
   unknown=sum(s=='unknown' for r in rs for s in r['after']),orange_review_cues=sum(r['orange_review_cues'] for r in rs))
 write(U/'RECHECK.json',dict(rows=rows,summary=summary,exposed_development=True,neural_inference_repeated=False));print(json.dumps(summary))
