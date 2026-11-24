"""Read-only diagnostic inference on user-requested recent attempts.
No optimization, threshold changes, training labels, candidate comparisons,
camera calls or changes to the running server.
"""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
D=Path(__file__).resolve().parent;O=D.parents[1];H=D.parent/'public_shape_arrival_repair_20261116'
sys.path.insert(0,str(O/'web_tutorial_live_integration_20261113'))
import server
sys.path.insert(0,str(H))
from repair_web_adapter import FeedbackConsumer
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
 assert not(D/'REPLAY.json').exists();contract=read(D/'CONTRACT.json');pins=contract['model_pins']
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in pins.items())
 inputs=read(D/'INPUTS.json')['rows'];net=FeedbackConsumer();rows=[]
 # Requested gesture inferred from direct RGB overview, never the classifier
 # result. Original target selection of earlier attempts was not persisted.
 words=['WORD0099','WORD1290','WORD1290','WORD1290','WORD0099']
 try:
  for i,(r,word)in enumerate(zip(inputs,words),1):
   p=Path(r['raw']);assert hashlib.sha256(p.read_bytes()).hexdigest()==r['raw_sha256'];raw=dict(np.load(p));q=net.raw(raw,word)
   q.pop('query_xy',None);write(D/f'attempt{i}_replayed.json',dict(source='same_frozen_model_diagnostic_replay_not_original_runtime_log',training_use=False,tuning_use=False,candidate_selection_use=False,result=q))
   row=dict(attempt=i,directory=r['directory'],time=r['time'],word=word,target_origin='Direct RGB gesture review; original selection not retained except last attempt',top3=q['tcn_top3'],passed=q['overall_pass'],criteria=q['lesson_acceptance']['criteria'],events=q['lesson_acceptance']['events'],coaching=q['coaching'],replay_compute_seconds=q['compute_seconds'],original_stop_reason=None,original_display_seconds=None,
    frames=r['frames'],observed=r['raw_frames'],dropped=r['session']['dropped'],errors=r['session']['errors'],max_queue=r['session']['max_queue'],detector_ms_p50=float(np.median([a['compute_ms']for a in r['session']['records']])),queue_ms_max=float(max(a['queue_ms']for a in r['session']['records'])))
   if i==5:
    live=read(D/'LATEST_RESULT.json')['result'];assert np.array_equal(np.array(live['frame_id']),raw['frame_id'])and np.array_equal(np.array(live['timestamp_us']),raw['timestamp_us'])
    row.update(original_stop_reason=live['stop_reason'],original_display_seconds=read(D/'LAST_BROWSER_ACK.json')['stop_to_browser_ack_seconds'],original_compute_seconds=live['stop_to_result_seconds'],original_overall_pass=live['overall_pass'],replay_matches_original_cards=q['cards']==live['cards'],replay_matches_original_events=q['lesson_acceptance']['events']==live['lesson_acceptance']['events'])
    assert row['replay_matches_original_cards']and row['replay_matches_original_events']
   rows.append(row);print(json.dumps(dict(attempt=i,word=word,top1=q['tcn_top3'][0],passed=q['overall_pass'],criteria={c['id']:c['status']for c in q['lesson_acceptance']['criteria']},events=q['lesson_acceptance']['events']),ensure_ascii=False),flush=True)
  write(D/'REPLAY.json',dict(rows=rows,source='Same deployed frozen model, diagnostic only; original result retained only for last attempt.',training_runs=0,model_changes=0,user_tuning=False,candidate_selection=False,source_modified=False))
 finally:
  for name in['apple','basic','latest']:getattr(net,name).close()
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in pins.items())
if __name__=='__main__':main()
