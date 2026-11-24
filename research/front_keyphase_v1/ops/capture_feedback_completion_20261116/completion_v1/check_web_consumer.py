"""Actual server consumer and coordinate guard, no camera or private recordings."""
from pathlib import Path
import sys,json,time,hashlib
import numpy as np
V=Path(__file__).resolve().parent;O=V.parents[1]
sys.path[:0]=[str(O/'word100_action_feedback_demo_20261108'),str(O/'web_tutorial_live_integration_20261113')]
import capture
sys.path[:0]=[str(O/'word100_action_linkage_repair_20261113'),str(O/'word100_actionable_feedback_20261113')]
import runtime_adapter
sys.path.insert(0,str(O/'word100_live_g6_20261114'))
from web_adapter import FeedbackConsumer
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
def main():
 out=V/'web';assert not(out/'CONSUMER_CHECK.json').exists();net=FeedbackConsumer();rows=[]
 try:
  sources=[]
  for p in[O/'basic_demo_feedback_20261115/source_corrected/CONFIRMATION_INPUTS.json',O/'basic_demo_feedback_20261115/local_shape/fresh/EXTRACTED.json',O/'basic_demo_feedback_20261115/acquisition_motion_followup/confirmation/EXTRACTED.json',V/'holdout/EXTRACTED.json']:
   sources.extend(r for r in read(p)['rows']if r.get('word','WORD1290')=='WORD1290')
  for r in sources:
   raw=dict(np.load(r.get('raw_path',r.get('raw'))));q=net.raw(raw,'WORD1290');(out/f'public_{r["person"]}.json').write_text(json.dumps(q,ensure_ascii=False,indent=2),encoding='utf-8')
   rows.append(dict(person=r['person'],supported=all(c['status']=='supported'for c in q['lesson_acceptance']['criteria']),
    criteria={c['id']:c['status']for c in q['lesson_acceptance']['criteria']},rank=q['tcn_top3'][0],seconds=q['compute_seconds']))
  print(json.dumps(rows,ensure_ascii=False))
  (out/'CONSUMER_CHECK.json').write_text(json.dumps(dict(rows=rows,camera_used=False,user_data=False,admission=False),ensure_ascii=False,indent=2),encoding='utf-8')
 finally:net.basic.close();net.latest.close()
if __name__=='__main__':main()
