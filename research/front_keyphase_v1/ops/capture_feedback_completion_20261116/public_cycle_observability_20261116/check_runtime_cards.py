from check_integrated import *
read,write,np,H,O=web.read,web.write,web.np,web.H,web.O

def main():
 net=FeedbackConsumer();assert net.cycle_admitted;rows=[];endpoints=[]
 sources=read(J/'SOURCE_INDEX.json')['rows']
 def raw(cid):return dict(np.load(next(r['raw']for r in sources if r['id']==cid)))
 try:
  # Existing public synthetic distinct-finger counterexample must stay an error.
  z=raw('NIA_SL_WORD1290_REAL13_F');q=read(D/'qualified_checks/NIA_SL_WORD1290_REAL13_F.json');lo=q['events'][0]['start'];hi=q['events'][-1]['end']
  r=next(r for r in read(O/'demo_handshape_learning_20261113/INPUTS.json')['rows']if r['clip_id']=='NIA_SL_WORD2633_REAL01_F');donor=dict(np.load(r['raw']));i=r['anchors'][len(r['anchors'])//2];hand=donor['xy_px'][i,112:133];hand=(hand-hand[0])/np.linalg.norm(hand[9]-hand[0]);xy=z['xy_px'];size=np.linalg.norm(xy[:,121]-xy[:,112],axis=1);xy[lo:hi+1,112:133]=xy[lo:hi+1,112:113]+hand[None]*size[lo:hi+1,None,None]
  q=net.raw(z,'WORD1290');card=next(c for c in q['cards']if c['id']=='right_shape');cue=next(c for c in q['coaching']['cues']if'right_shape'in c['card_ids']);rows.append(dict(kind='public_synthetic_distinct_fingers',passed=not q['overall_pass']and card['status']=='difference'and cue['kind']=='shape'and card['query_indices'][0]<=hi,card=card,cue=cue,seconds=q['compute_seconds']))
  # Existing uncertainty must remain gray at its actual source frames.
  q=net.raw(raw('NIA_SL_WORD1290_REAL13_L'),'WORD1290');c=next(c for c in q['lesson_acceptance']['criteria']if c['id']=='left_shape');card=next(c for c in q['cards']if c['id']=='left_shape');passed=not q['overall_pass']and card['status']=='unknown'and card['display_status']=='uncertain'and card['query_spans']==c['uncertain_spans']and not card['difference_spans'];rows.append(dict(kind='public_uncertain_left_palm',passed=passed,card=card));write(J/'web/UNCERTAIN_LOCATION_CHECK.json',dict(passed=passed,criterion=c,card=card,private_data=False))
  for cid,cut in [('NIA_SL_WORD1290_REAL11_L',60),('NIA_SL_WORD1290_REAL13_R',65)]:
   z=raw(cid)
   for kind in ['full','first_only']:
    v={k:x.copy()for k,x in z.items()}
    if kind=='first_only':
     for k in ['xy_px','score','timestamp_us','frame_id']:v[k]=v[k][:cut]
    q=net.raw(v,'WORD1290');c=next(c for c in q['cards']if c['id']=='repeat_order');ev=len(q['lesson_acceptance']['events']);passed=ev==2 and q['overall_pass']and c['status']=='supported'if kind=='full'else ev==1 and not q['overall_pass']and c['status']=='unknown';endpoints.append(dict(id=cid,kind=kind,events=ev,overall_pass=q['overall_pass'],card=c,passed=passed,seconds=q['compute_seconds']))
  q=net.raw(raw('NIA_SL_WORD1290_REAL16_L'),'WORD1290');c=next(c for c in q['lesson_acceptance']['criteria']if c['id']=='left_shape');card=next(c for c in q['cards']if c['id']=='left_shape');passed=not q['overall_pass']and len(q['lesson_acceptance']['events'])==2 and c['source_spans']and card['status']=='unknown'and card['display_status']=='uncertain'and card['query_spans']==c['unconfirmed_events'];coverage=dict(passed=bool(passed),events=q['lesson_acceptance']['events'],criterion=c,card=card,private_data=False);write(J/'web/COVERAGE_CARD_CHECK.json',coverage)
  write(J/'web/CARD_CHECK.json',dict(rows=rows,passed=all(r['passed']for r in rows),private_data=False));write(J/'web/ENDPOINT_CARD_CHECK.json',dict(rows=endpoints,passed=all(r['passed']for r in endpoints),private_data=False));print('CARDS',[(r.get('kind'),r['passed'])for r in rows+endpoints],coverage['passed'],flush=True);assert all(r['passed']for r in rows+endpoints)and coverage['passed']
 finally:web.close(net)
if __name__=='__main__':main()
