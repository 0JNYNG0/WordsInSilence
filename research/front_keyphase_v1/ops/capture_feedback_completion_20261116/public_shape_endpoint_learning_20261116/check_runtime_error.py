from check_web import *
def main():
 src=next(r for r in read(H/'PUBLIC_INPUTS.json')['rows']if r['id']=='NIA_SL_WORD1290_REAL13_F');raw=dict(np.load(src['raw']));q=read(D/'comparison/NIA_SL_WORD1290_REAL13_F.json')['before'];lo=q['events'][0]['start'];hi=q['events'][-1]['end'];r=next(r for r in read(O/'demo_handshape_learning_20261113/INPUTS.json')['rows']if r['clip_id']=='NIA_SL_WORD2633_REAL01_F');d=dict(np.load(r['raw']));i=r['anchors'][len(r['anchors'])//2];h=d['xy_px'][i,112:133];h=(h-h[0])/np.linalg.norm(h[9]-h[0]);xy=raw['xy_px'];size=np.linalg.norm(xy[:,121]-xy[:,112],axis=1);xy[lo:hi+1,112:133]=xy[lo:hi+1,112:113]+h[None]*size[lo:hi+1,None,None]
 net=FeedbackConsumer()
 try:
  q=net.raw(raw,'WORD1290');c=next(c for c in q['cards']if c['id']=='right_shape');cue=next(c for c in q['coaching']['cues']if'right_shape'in c['card_ids']);passed=not q['overall_pass']and c['status']=='difference'and cue['kind']=='shape'and c['query_indices'][0]<=hi
  write(D/'web/ERROR_CONSUMER_CHECK.json',dict(passed=passed,source='synthetic public hand pattern',card=c,cue=cue,overall_pass=q['overall_pass'],compute_seconds=q['compute_seconds'],private_data=False));print('ERROR_CONSUMER',passed,c,flush=True);assert passed
 finally:close(net)
if __name__=='__main__':main()
