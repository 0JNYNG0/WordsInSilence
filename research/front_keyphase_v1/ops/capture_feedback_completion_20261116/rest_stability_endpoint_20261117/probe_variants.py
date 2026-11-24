from endpoint_candidate import *
def variants(raw,hand,events):
 yield 'normal',{k:v.copy()for k,v in raw.items()}
 for scale in [.75,1.25]:
  z={k:v.copy()for k,v in raw.items()};center=np.array([int(z['source_width'])/2,int(z['source_height'])/2]);z['xy_px']=(z['xy_px']-center)*scale+center;yield 'scale_'+str(scale),z
 for degrees in [-10,10]:
  z={k:v.copy()for k,v in raw.items()};theta=np.deg2rad(degrees);rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]);center=np.array([int(z['source_width'])/2,int(z['source_height'])/2]);z['xy_px']=(z['xy_px']-center)@rotation.T+center;yield 'camera_roll_'+str(degrees),z
 z={k:v.copy()for k,v in raw.items()};a=events[0]['start'];b=events[-1]['end'];sz=np.linalg.norm(z['xy_px'][:,121]-z['xy_px'][:,112],axis=1);z['xy_px'][a:b+1,112:133]=z['xy_px'][a:b+1,112:113]+hand[None]*sz[a:b+1,None,None];yield 'different_hand_form',z
def main():
 torch.set_num_threads(2);a=ShortArrivalLesson();b=EndpointLesson();rows=[]
 donor=next(r for r in read(O/'demo_handshape_learning_20261113/INPUTS.json')['rows']if r['clip_id']=='NIA_SL_WORD2633_REAL01_F');dr=dict(np.load(donor['raw']));i=donor['anchors'][len(donor['anchors'])//2];hand=dr['xy_px'][i,112:133];hand=(hand-hand[0])/np.linalg.norm(hand[9]-hand[0])
 for r in sources():
  if r['word']!='WORD1290' or r['person']not in [1,2]:continue
  z=dict(np.load(r['raw']));base=a.analyze(z)
  for kind,zz in variants(z,hand,base['events']):
   qa=a.analyze(zz);qb=b.analyze(zz);row=dict(id=r['id'],kind=kind,before=len(qa['events']),after=len(qb['events']),before_status={c['id']:c['status']for c in qa['criteria']},after_status={c['id']:c['status']for c in qb['criteria']},added=qb['endpoint_candidate_support']);rows.append(row);print('CANDIDATE',r['id'],kind,row['before'],row['after'],row['added'],flush=True)
 write(D/'probe/VARIANTS.json',rows)
if __name__=='__main__':main()
