"""Independent saved-result arithmetic for heldout empirical ranges. No model inference."""
import prepare as R
import numpy as np,traceback
from pathlib import Path
from collections import defaultdict,Counter

def eligibility(z,q,r):
 maskq=q['available'].copy();maskr=r['available'].copy();maskq[:,:,:15]&=q['quality_after'][:,:,:15];maskr[:,:,:15]&=r['quality_after'][:,:,:15]
 t=z['phase_old'];a=np.floor(t).astype(int);b=np.ceil(t).astype(int);point=z['point_ok']&maskq[a]&maskq[b]&maskr;pc=point&q['core'][a,None,None]&q['core'][b,None,None];sequence=np.zeros_like(z['domain']);sc=sequence.copy()
 for j in range(2,len(t)-2):
  for tag,phase in [(1,z['phase_new']),(2,t)]:
   ids=[j-2,j,j+2];flo=np.floor(phase[ids]).astype(int);cei=np.ceil(phase[ids]).astype(int)
   left=int(z['span_first'][j-2]);right=int(z['span_last'][j+2])
   if tag==1:left=min(left,int(flo[0]));right=max(right,int(cei[-1]))
   ok=z['sequence_ok'][j]&(z['source'][j]==tag)&maskq[left:right+1].all(0)&maskq[flo].all(0)&maskq[cei].all(0)&maskr[j-2:j+3].all(0)
   sequence[j]|=ok;sc[j]|=ok&bool(q['core'][left:right+1].all())
 return point,sequence,pc,sc

def verify():
 assert not(R.U/'RANGE_RUNNING.json').exists();f=R.read(R.U/'RANGE_FROZEN.json');result=R.read(R.U/'RANGE_DONE.json');R.check(f['pins']);R.check(f['protected']);R.check(result['pins']);samples={x['clip_id']:R.load(x['output']) for x in f['rows']};meta={x['clip_id']:x for x in f['rows']};by=defaultdict(list)
 for x in result['rows']:by[x['reference']].append(x)
 counts=Counter();tests=Counter()
 for rid,rows in sorted(by.items()):
  data={x['query']:R.load(x['pair']) for x in rows};sup={k:eligibility(z,samples[k],samples[rid]) for k,z in data.items()}
  for row in rows:
   qid=row['query'];z=data[qid];s=R.load(row['output']);ids=[x for x in data if meta[x]['signer_id']!=meta[qid]['signer_id']];assert sorted(ids)==sorted(row['calibration_ids']);assert qid not in ids and rid not in ids and len(ids) in [6,7];assert row['heldout_record_id']==meta[qid]['signer_id'];tests['heldout_absent_from_reference_and_all_calibrators']+=1
   np.testing.assert_array_equal(s['domain'],z['domain']);c=Counter(pairs=1,domain=int(z['domain'].sum()))
   for mode,offset in [('point',0),('sequence',1)]:
    ds=np.array([data[k][mode+'_delta'] for k in ids]);vs=np.array([sup[k][offset+2] for k in ids]);num=vs.sum(0);valid=np.expand_dims(vs,-1) if mode=='sequence' else vs;rawmax=np.max(np.where(valid,ds,-np.inf),0);cut=num>=5;bound=np.where(cut[...,None] if mode=='sequence' else cut,rawmax,np.nan)
    np.testing.assert_array_equal(s[mode+'_peer_count'],num);np.testing.assert_allclose(s[mode+'_bound'],bound,atol=0,rtol=0,equal_nan=True);np.testing.assert_array_equal(s[mode+'_evidence'],sup[qid][offset]);np.testing.assert_array_equal(s[mode+'_delta'],z[mode+'_delta'])
    domain=z['domain'];raw=z[mode+'_ok'];strong=sup[qid][offset];ex=z[mode+'_delta']>bound+1e-6;ex=ex.any(-1) if mode=='sequence' else ex
    out=np.select([~domain,~raw,~strong,~cut,ex],[0,1,2,3,5],default=4).astype(np.uint8);np.testing.assert_array_equal(out,s[mode+'_status']);assert not ((s[mode+'_status']==4)&(~raw|~strong|~cut|~domain)).any()
    for group,sl in [('angle',slice(0,15)),('position',slice(15,23))]:
     for code in range(1,6):c[f'{mode}_{group}_state{code}']=int((out[:,:,sl]==code).sum())
    assert int(sum(c[f'{mode}_{g}_state{k}'] for g in ['angle','position'] for k in range(1,6)))==int(domain.sum());tests[mode+'_full_domain_and_no_unknown_normal']+=1
   counts.update(c)
  if counts['pairs']%500<9:print('VERIFIED_EMPIRICAL_RANGE',counts['pairs'],7184,flush=True)
 assert counts==Counter(result['counts']) and counts['pairs']==7184;R.check(f['protected']);R.write(R.U/'RANGE_VERIFIED.json',dict(at=R.now(),status='PASS7184_LEAVE_PERSON_OUT_EMPIRICAL_RANGE_ARITHMETIC',counts=dict(counts),tests=dict(tests),result_sha256=R.sha(R.U/'RANGE_DONE.json'),verifier_sha256=R.sha(__file__),method='SameCodex separate source eligibility/window arithmetic and leave-person-out bound/status reconstruction; raw saved differences unchanged. No independent anatomical or semantic correctness certification, controls still separate.'));print('EMPIRICAL_RANGE_VERIFIED',dict(counts),flush=True)
if __name__=='__main__':
 try:verify()
 except BaseException:
  p=R.U/'VERIFY_RANGE_FIRST_ERROR.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
