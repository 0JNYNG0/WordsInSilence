import prepare as R
import numpy as np
from collections import Counter
def main():
 p=R.read(R.U/'PREPARED.json');res=R.read(R.U/'PAIRS_DONE.json');samples={r['clip_id']:R.load(r['output']) for r in p['rows']};stats=Counter();allstates=Counter();pins={};querysets=Counter()
 for row in p['rows']:
  z=samples[row['clip_id']]
  for s,n in zip(*np.unique(z['state_after'][z['core']],return_counts=True)):allstates[str(s)]+=int(n)
  assert allstates['1']==0,'Every raw-computable core side should now be requested'
 for row in res['rows']:
  q=samples[row['query']];r=samples[row['reference']];z=R.load(row['pair']);pins[row['pair']]=R.sha(row['pair']);lo=np.floor(z['phase_old']).astype(int);hi=np.ceil(z['phase_old']).astype(int)
  for j,h,a in np.argwhere(z['point_ok'][:,:,:15]&~z['point_corroborated_after'][:,:,:15]):
   states=np.array([q['state_after'][lo[j],h,a],q['state_after'][hi[j],h,a],r['state_after'][j,h,a]]);bad=states[states!=5];assert len(bad) and not(bad==0).any();stats['uncorroborated_point_angles']+=1
   if (bad==1).all():stats['only_not_rechecked']+=1
   else:stats['has_other_uncertainty']+=1
   if (bad==1).any():
    stats['not_rechecked']+=1;assert not(q['core'][lo[j]] and q['core'][hi[j]]);stats['includes_query_outside_annotated_core']+=1
   for name,v in [('image_pair_unavailable',2),('identity_uncertain',3),('image_disagreement',4)]:
    if (bad==v).any():stats[name]+=1
  if row['counts']['point_after']>row['counts']['point_before']:querysets['pairs_with_more_corroborated_points']+=1
  if row['counts']['sequence_after']>row['counts']['sequence_before']:querysets['pairs_with_more_corroborated_sequence']+=1
 assert stats['uncorroborated_point_angles']==res['counts']['raw_point_angles']-res['counts']['point_after'];assert stats['only_not_rechecked']+stats['has_other_uncertainty']==stats['uncorroborated_point_angles']
 R.write(R.U/'CAUSES.json',dict(at=R.now(),counts=dict(stats),core_state_cells=dict(allstates),pair_improvements=dict(querysets),pins=pins,method='Raw-computable point source states, separate unrequested outside annotated querycore and actual image disagreement/pair/identity shortages. Category overlaps are explicit. Core has no unrequested computable angles. Outsidecore mapping is not declared an alignment error without timing truth.'))
 print('CORE_CAUSES',dict(stats),dict(allstates),dict(querysets),flush=True)
if __name__=='__main__':main()
