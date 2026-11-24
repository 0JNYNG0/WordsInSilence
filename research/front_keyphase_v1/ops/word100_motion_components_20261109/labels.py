from common import *
def main():
 assert not (U/'FROZEN.json').exists()
 extra={
 '0739':{2:[(58,70,0,-1),(70,83,1,1),(46,58,0,0),(83,95,1,0)],3:[(64,81,0,-1),(81,92,1,1),(53,64,0,0),(92,98,1,0)],5:[(52,64,0,-1),(64,83,1,1),(39,52,0,0),(83,89,1,0)],6:[(65,84,0,-1),(84,97,1,1),(52,65,0,0),(97,103,1,0)],8:[(58,69,0,-1),(69,86,1,1),(42,53,0,0),(86,97,1,0)],9:[(53,65,0,-1),(65,76,1,1),(36,47,0,0),(76,82,1,0)]},
 '1133':{2:[(49,62,2,1),(66,74,2,0)],3:[(55,69,2,1),(69,76,2,0)],5:[(49,61,2,1),(63,67,2,0)],6:[(44,54,2,1),(54,60,2,0)],8:[(75,89,2,1),(89,94,2,0)],9:[(63,69,2,0)]},
 '1567':{2:[(47,58,4,-1),(58,68,4,1),(68,78,4,-1),(83,94,4,0)],3:[(41,50,4,-1),(50,59,4,1),(59,73,4,-1),(73,82,4,0)],5:[(55,64,4,-1),(64,73,4,1),(73,82,4,-1),(82,91,4,0)],6:[(50,59,4,-1),(59,67,4,1),(67,79,4,-1),(79,88,4,0)],8:[(70,79,4,-1),(79,89,4,1),(94,103,4,-1),(103,113,4,0)],9:[(55,64,4,-1),(64,73,4,1),(73,82,4,-1),(87,100,4,0)]},
 '2523':{2:[(55,62,3,1),(64,73,3,0)],3:[(69,78,3,1),(75,82,3,0)],5:[(62,72,3,1),(72,77,3,0)],6:[(66,78,3,1),(78,83,3,0)],8:[(54,66,3,1),(66,73,3,0)],9:[(65,75,3,1),(75,81,3,0)]}}
 reviews={r['clip_id']:r for r in read(U/'REVIEW.json')['rows']};rows=[]
 for r in read(PREV/'READY.json')['rows']:
  assert sha(r['path'])==r['sha256'];z=np.load(r['path']);p=int(r['person']);w=r['clip_id'].split('_')[2][4:];pairs=list(r['pairs'])
  for a,b,axis,sign in extra[w].get(p,[]):
   review=reviews[r['clip_id']];assert a in review['frames'] and b in review['frames'],(r['clip_id'],a,b)
   ia=np.flatnonzero(z['frame_id']==a);ib=np.flatnonzero(z['frame_id']==b);assert len(ia)==len(ib)==1
   pairs.append(dict(a=int(ia[0]),b=int(ib[0]),frame_a=a,frame_b=b,axis=axis,target=sign+1,provenance='additional FIT RGB review'))
  rows.append(dict(r,pairs=pairs))
 write(U/'READY.json',dict(rows=rows,reviewer='same Codex sparse RGB, not independent expert',new_pairs=sum(len(v) for w in extra.values() for v in w.values()),
  withheld=['WORD1133 REAL09 sheet starts near distal arm; no new positive stroke pair invented.'],original_annotation_sha256=sha(PREV/'ANNOTATIONS.json')))
 cfg=dict(seeds=[42,43,44],primary_seed=42,epochs=240,lr=.003,weight_decay=.001,
  features='9 resampled scalar/relative auxiliary positions + signed/absolute change; type of body relation, never word/person/frame number',
  models=['motion_only','motion_plus_frozen_latent'],primary='motion_only',hidden=[48,24],
  split='9 FIT IDs; 3 historical DEV IDs; no private tuning',
  augment='FIT both views, complete pair reverse with sign inversion; no generated images; no accidental missing as hold',
  controls=['native normal','upper boundary','time .75/1.25','reversed pair','short missing 2','long missing','mirror side rebind','coordinate translation/scale'],
  confidence=.65,coverage=.8,min_observed=4,
  deployment_gate='Core movement recall may not fall below old explicit geometry, full+upper; no extra opposite-direction predictions in normal; uncertainty cannot be counted correct. Learned auto-window feedback not established by supplied windows.',
  pins={str(PREV/'PUBLIC_DECISION.json'):sha(PREV/'PUBLIC_DECISION.json'),str(V/'feedback_runs/pose_transfer_42/model.pt'):sha(V/'feedback_runs/pose_transfer_42/model.pt'),str(D/'consumer.py'):sha(D/'consumer.py'),str(D/'CARDS.json'):sha(D/'CARDS.json')},
  data_sha256=sha(U/'READY.json'),private_learning=False)
 write(U/'FROZEN.json',cfg);event('labels_frozen',fit_pairs=sum(len(r['pairs']) for r in rows if r['role']=='fit'),dev_pairs=sum(len(r['pairs']) for r in rows if r['role']=='development'))
if __name__=='__main__':main()
