"""Refresh public demo responses after a code change; preserve old TCN results.
This is not another accuracy run. No detector, private clip or webcam is invoked.
"""
from setup import *
from compare import FeedbackConsumer,quarantine,A,infer,assess_pair
import numpy as np,copy

if __name__=='__main__':
 assert not (U/'INTEGRATION.json').exists();c=FeedbackConsumer();examples=read(D/'EXAMPLES.json');shutil.copyfile(D/'EXAMPLES.json',U/'before/EXAMPLES.json');out=[];new=[]
 for r in examples:
  word=r['word'];old=read(D/r['url'].lstrip('/'));z=np.load(D/'examples'/(word+'_normal')/'observations.npz');raw={k:z[k].copy() for k in z.files}
  if r['id'].endswith('_edited'):
   ids=np.delete(np.arange(len(raw['frame_id'])),old['original_removed_frames'])
   for k in ['xy_px','score']:raw[k]=raw[k][ids]
   raw['timestamp_us']=np.rint(np.arange(len(ids))*1e6/30).astype(np.int64);raw['frame_id']=np.arange(len(ids))
  g=quarantine(raw['xy_px'],raw['score'],raw['timestamp_us'],int(raw['source_width']),int(raw['source_height']));a=A.convert(g['xy_px'],raw['score'],raw['timestamp_us'],raw['frame_id'],int(raw['source_width']),int(raw['source_height']))
  p=infer(c,a['tcn280'],word);b,e,ms=assess_pair(a['tcn280'],raw['timestamp_us'],raw['frame_id'],word,p)
  response={**old,**e};response['stop_to_result_seconds']=None;response['cache_refresh_note']='Feedback evidence updated; saved TCN ranks preserved, no new recognition accuracy measurement.'
  response['query_xy']=np.nan_to_num(g['xy_px'],nan=-1).tolist()
  for card in response['cards']:
   sp=card.get('query_indices');ss=[0 if s=='L' else 1 for s in card['hands']]
   if sp and g['hand_quarantined'][sp[0]:sp[1]+1,ss].any():card.update(status='unknown',display_status='uncertain',message='이 구간의 손을 일부 확인하기 어려워요. 틀렸다는 뜻은 아니에요.')
  path=D/'examples'/('positive_'+r['id'])/'result.json';path.parent.mkdir(exist_ok=True);write(path,response)
  assert response['tcn_top3']==old['tcn_top3'];assert response['overall_pass'] is False
  new.append({**r,'url':'/examples/positive_'+r['id']+'/result.json'})
  out.append(dict(id=r['id'],path=str(path),statuses=[x['status'] for x in response['cards']],tcn_preserved=True,detector_calls=0,camera=False))
 write(D/'EXAMPLES.json',new)
 write(U/'INTEGRATION.json',dict(status='PASS',public_examples=out,private_used=False,tcn_inference=False,feedback_replay_purpose='Verify changed evidence API and refresh demo display, not new accuracy reporting'))
 print(json.dumps(out,ensure_ascii=False))
