import prepare as R
import numpy as np
from collections import Counter
def main():
 p=R.read(R.U/'PREPARED.json');res=R.read(R.U/'PAIRS_DONE.json');samples={r['clip_id']:R.load(r['output']) for r in p['rows']};stats=Counter();allstates=Counter();pins={}
 for row in p['rows']:
  z=samples[row['clip_id']]
  for s,n in zip(*np.unique(z['state_after'][z['core']],return_counts=True)):allstates[str(s)]+=int(n)
 for row in res['rows']:
  q=samples[row['query']];r=samples[row['reference']];z=R.load(row['pair']);pins[row['pair']]=R.sha(row['pair']);lo=np.floor(z['phase_old']).astype(int);hi=np.ceil(z['phase_old']).astype(int)
  for j,h,a in np.argwhere(z['point_ok'][:,:,:15]&~z['point_corroborated_after'][:,:,:15]):
   states=np.array([q['state_after'][lo[j],h,a],q['state_after'][hi[j],h,a],r['state_after'][j,h,a]]);bad=states[states!=5];assert len(bad) and not (bad==0).any();stats['uncorroborated_point_angles']+=1
   if (bad==1).all():stats['only_not_rechecked']+=1
   else:stats['has_other_uncertainty']+=1
   for name,v in [('not_rechecked',1),('image_pair_unavailable',2),('identity_uncertain',3),('image_disagreement',4)]:
    if (bad==v).any():stats[name]+=1
   if not(q['state_after'][lo[j],h,a]==q['state_after'][hi[j],h,a]==5):stats['query_needs_more_evidence']+=1
   if r['state_after'][j,h,a]!=5:stats['reference_needs_more_evidence']+=1
 assert stats['uncorroborated_point_angles']==res['counts']['raw_point_angles']-res['counts']['point_after'];assert stats['only_not_rechecked']+stats['has_other_uncertainty']==stats['uncorroborated_point_angles']
 R.write(R.U/'CAUSES.json',dict(at=R.now(),counts=dict(stats),core_state_cells=dict(allstates),pins=pins,method='Disjoint primary classification: all failed source states unrequested versus any other uncertainty. Named causes may overlap. Includes only raw-computable points; missing native angles separately retained. Scores never used.',independent_verification='Sum against independently verified PAIRS_DONE and each exact source-state tuple; no model/detector rerun. SameCodex diagnostic,not human labels.'))
 print('CAUSES',dict(stats),dict(allstates),flush=True)
if __name__=='__main__':main()
