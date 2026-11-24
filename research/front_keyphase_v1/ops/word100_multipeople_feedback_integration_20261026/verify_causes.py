import run as R
import numpy as np
from collections import Counter

def main():
 f=R.read(R.U/'FROZEN.json');result=R.read(R.U/'RESULT.json');a=R.read(R.U/'CAUSE_AUDIT.json');R.check(f['pins']);R.check(result['pins']);R.check(a['pins']);samples={x['clip_id']:R.load(x['output']) for x in f['rows']};rows={x['name']:x for x in a['rows']};count=Counter();base=Counter()
 for z in samples.values():
  st=z['angle_state'][z['core']];base['reference_audit_angle_cells']+=st.size
  for s in range(5):base['state_'+str(s)]+=int(np.count_nonzero(st==s))
  base['eligible']+=int(np.count_nonzero(z['quality_eligible'][z['core'],:,:15]))
 assert dict(base)==a['base_core_states']
 for row in result['rows']:
  q=samples[row['query']];r=samples[row['reference']];z=R.load(row['pair']);c=R.load(rows[row['name']]['path']);so=z['sequence_ok'][:,:,:15];qq=z['sequence_quality'][:,:,:15];n=len(so)
  for field,state in [('sequence_no_second',2),('sequence_prediction_disagreement',3),('sequence_identity_disagreement',None)]:
   qflag=q['angle_state']==state if state is not None else np.broadcast_to(q['identity_conflict'][:,:,None],q['angle_state'].shape)
   rflag=r['angle_state']==state if state is not None else np.broadcast_to(r['identity_conflict'][:,:,None],r['angle_state'].shape)
   qc=np.concatenate([np.zeros((1,2,15),int),np.cumsum(qflag,axis=0)],axis=0);rc=np.concatenate([np.zeros((1,2,15),int),np.cumsum(rflag,axis=0)],axis=0)
   expected=np.zeros_like(so)
   for j,h,k in zip(*np.nonzero(so)):
    lo,hi=int(z['span_first'][j-2]),int(z['span_last'][j+2])
    if z['source'][j,h,k]==1:lo=min(lo,int(np.floor(z['phase_new'][j-2])));hi=max(hi,int(np.ceil(z['phase_new'][j+2])))
    assert c['query_interval_start'][j,h,k]==lo and c['query_interval_end'][j,h,k]==hi
    expected[j,h,k]=(qc[hi+1,h,k]-qc[lo,h,k]+rc[j+3,h,k]-rc[j-2,h,k])>0
   assert np.array_equal(expected,c[field])
  ns,ds,ids=[c[k] for k in ['sequence_no_second','sequence_prediction_disagreement','sequence_identity_disagreement']]
  assert np.array_equal(so&~qq,ns|ds|ids)
  stats=dict(sequence_computable=int(so.sum()),sequence_corroborated=int(qq.sum()),uncorroborated=int((so&~qq).sum()),no_second_prediction=int(ns.sum()),prediction_disagreement=int(ds.sum()),identity_disagreement=int(ids.sum()),only_no_second_prediction=int((ns&~ds&~ids).sum()),both_prediction_and_identity_or_missing=int((ds&(ids|ns)).sum()))
  assert all(rows[row['name']][k]==v for k,v in stats.items());count.update(stats)
 assert dict(count)==a['pair_counts'];R.check(f['protected']);R.write(R.U/'CAUSE_VERIFIED.json',dict(at=R.now(),status='PASS_CAUSES_ALL600',counts=dict(count),method='Independent cumulative-sum interval counts over original states;cause union exactly equals uncorroborated sequence set. Same Codex verifier,not independent truth.',only_missing_second_percent=100*count['only_no_second_prediction']/count['uncorroborated'],source=R.sha(R.U/'CAUSE_AUDIT.json'),code=R.sha(__file__)));print('VERIFIED_CAUSES600',dict(count),flush=True)
if __name__=='__main__':main()
