from check_runtime import *
import copy
from contract import QuestProgress

def main():
 net=FeedbackConsumer();net.basic_admitted=net.apple_admitted=True;rows=[];inputs=read(N/'INPUTS.json')['rows']
 def raw(cid):return dict(np.load(next(r['raw']for r in inputs if r['id']==cid)))
 try:
  z=raw('NIA_SL_WORD1290_REAL13_F');normal=net.raw(z,'WORD1290');ev=normal['lesson_acceptance']['events'];a=ev[0]['start'];b=ev[-1]['end'];cut=ev[1]['arrival_index']-1
  donor=next(r for r in read(O/'demo_handshape_learning_20261113/INPUTS.json')['rows']if r['clip_id']=='NIA_SL_WORD2633_REAL01_F');require_front(donor);dr=dict(np.load(donor['raw']));i=donor['anchors'][len(donor['anchors'])//2];hand=dr['xy_px'][i,112:133];hand=(hand-hand[0])/np.linalg.norm(hand[9]-hand[0])
  for kind in ['normal','distinct_fingers','missing_fingers','partial','single_gap']:
   v={k:x.copy()for k,x in z.items()}
   if kind=='distinct_fingers':
    sz=np.linalg.norm(v['xy_px'][:,121]-v['xy_px'][:,112],axis=1);v['xy_px'][a:b+1,112:133]=v['xy_px'][a:b+1,112:113]+hand[None]*sz[a:b+1,None,None]
   elif kind=='missing_fingers':v['xy_px'][a:b+1,125:133]=np.nan;v['score'][a:b+1,125:133]=0
   elif kind=='partial':
    for k in ['xy_px','score','timestamp_us','frame_id']:v[k]=v[k][:cut]
   elif kind=='single_gap':v['xy_px'][(a+b)//2,112:133]=np.nan
   q=net.raw(v,'WORD1290');c=next(c for c in q['cards']if c['id']=='right_shape');rep=next(c for c in q['cards']if c['id']=='repeat_order')
   passed=q['overall_pass']if kind=='normal'else(c['status']=='difference'and c['query_indices']and any('right_shape'in cue['card_ids']for cue in q['coaching']['cues'])and not q['overall_pass'])if kind=='distinct_fingers'else(c['status']=='unknown'and c['display_status']=='uncertain'and not c['difference_spans']and c['query_indices']and not q['overall_pass'])if kind=='missing_fingers'else(not q['overall_pass']and rep['status']=='unknown')if kind=='partial'else not q['coaching']['cues']
   rows.append(dict(kind=kind,passed=bool(passed),shape_card=c,repeat_card=rep,overall_pass=q['overall_pass'],coaching=q['coaching']));print('CARD',kind,bool(passed),flush=True)
  # Synthetic app contract only: no live quest state or fabricated user log.
  quest=QuestProgress(steps=['WORD1290']);result=copy.deepcopy(normal);result.update(source='user_inference_only',stop_reason='arms_returned');result['tcn_top3'][0].update(word='WORD1290',probability=.9)
  attempt=dict(id='unit-1',mode='quest',generation=quest.generation,target='WORD1290');assert quest.consume(attempt,result)['advanced'];assert not quest.consume(attempt,result)['advanced']
  rows.append(dict(kind='quest_accept_once',passed=True,synthetic_application_contract=True))
  quest.reset();result['lesson_acceptance']['criteria'][0]['status']='unknown';result['lesson_acceptance']['overall_pass']=False;result['overall_pass']=False;attempt.update(id='unit-2',generation=quest.generation);assert not quest.consume(attempt,result)['advanced'];rows.append(dict(kind='quest_unknown_blocked',passed=True,synthetic_application_contract=True))
  write(N/'web/CARD_CHECK.json',dict(rows=rows,passed=all(r['passed']for r in rows),physical_camera=False,private_data=False));assert all(r['passed']for r in rows)
 finally:close(net)
if __name__=='__main__':main()
