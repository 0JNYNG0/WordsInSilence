from check_integrated import *
read,write,np,H,O=web.read,web.write,web.np,web.H,web.O

def main():
 net=FeedbackConsumer();assert net.motion_admitted;rows=[]
 try:
  src=next(r for r in read(H/'PUBLIC_INPUTS.json')['rows']if r['id']=='NIA_SL_WORD1290_REAL13_F');z=dict(np.load(src['raw']));q=read(D/'qualified_checks/NIA_SL_WORD1290_REAL13_F.json');lo=q['events'][0]['start'];hi=q['events'][-1]['end']
  r=next(r for r in read(O/'demo_handshape_learning_20261113/INPUTS.json')['rows']if r['clip_id']=='NIA_SL_WORD2633_REAL01_F');donor=dict(np.load(r['raw']));i=r['anchors'][len(r['anchors'])//2];hand=donor['xy_px'][i,112:133];hand=(hand-hand[0])/np.linalg.norm(hand[9]-hand[0]);xy=z['xy_px'];size=np.linalg.norm(xy[:,121]-xy[:,112],axis=1);xy[lo:hi+1,112:133]=xy[lo:hi+1,112:113]+hand[None]*size[lo:hi+1,None,None]
  q=net.raw(z,'WORD1290');card=next(c for c in q['cards']if c['id']=='right_shape');cue=next(c for c in q['coaching']['cues']if'right_shape'in c['card_ids']);passed=not q['overall_pass']and card['status']=='difference'and cue['kind']=='shape'and card['query_indices'][0]<=hi
  rows.append(dict(kind='public_synthetic_distinct_fingers',passed=passed,card=card,cue=cue,module=q.get('motion_evidence_module'),seconds=q['compute_seconds']))
  src=next(r for r in read(H/'public_views/INPUTS.json')['rows']if r['id']=='NIA_SL_WORD1290_REAL13_L');q=net.raw(dict(np.load(src['raw'])),'WORD1290');card=next(c for c in q['cards']if c['id']=='left_shape');passed=not q['overall_pass']and card['status']=='unknown';rows.append(dict(kind='public_uncertain_left_palm',passed=passed,card=card,seconds=q['compute_seconds']))
  write(E/'web/CARD_CHECK.json',dict(rows=rows,passed=all(r['passed']for r in rows),physical_camera=False,private_data=False));print(rows,flush=True);assert all(r['passed']for r in rows)
 finally:web.close(net)
if __name__=='__main__':main()
