from pathlib import Path
import sys
from collections import Counter,defaultdict
import numpy as np
T=Path(__file__).resolve().parent;U=T.parent/'word100_train_reference_bank_20261026';sys.path.insert(0,str(U))
import prepare as R
V=R.mod('repeat_verified_values',U/'verify_alignment.py');E=R.mod('repeat_verified_evidence',U/'verify_range.py')
KEYS=('features','present','core','values','available','quality_eligible','quality_after')
def boundaries(q):
 n=len(q['core']);start=[0]
 for i in range(1,n):
  if any(not np.array_equal(q[k][i],q[k][i-1],equal_nan=True) for k in KEYS):start.append(i)
 return np.array(start),np.r_[np.array(start[1:])-1,n-1]
def expand(p,s,e):
 return np.array([(s[int(v)]+e[int(v)])/2 if v==int(v) else e[int(np.floor(v))]+(v-np.floor(v))*(s[int(np.ceil(v))]-e[int(np.floor(v))]) for v in p])
def main():
 assert not(T/'RUNNING.json').exists();f=R.read(T/'FROZEN.json');done=R.read(T/'DONE.json');R.check(f['pins']);R.check(f['protected']);meta={r['clip_id']:r for r in f['inputs']};cache={};total=defaultdict(Counter);checked=0
 for i,row in enumerate(done['rows']):
  z=R.load(row['candidate_output']);old=R.load(row['output']);b=R.load(row['original_range']);change=row['changed'];kind=row['kind'];c=Counter(pairs=1,realigned=int(change),reused=int(not change))
  if change:
   qp,rp=row['input'],meta[row['reference']]['output']
   for p in (qp,rp):
    if p not in cache:cache[p]=R.load(p)
   q,r=cache[qp],cache[rp];qs,qe=boundaries(q);rs,re=boundaries(r);V.same(qs,z['query_group_start']);V.same(qe,z['query_group_end']);V.same(rs,z['reference_group_start']);V.same(re,z['reference_group_end']);rg=np.concatenate([np.full(b-a+1,j) for j,(a,b) in enumerate(zip(rs,re))]);V.same(rg,z['reference_group_index']);path=z['compressed_path'];assert(path[0]==[0,0]).all() and(path[-1]==[len(qs)-1,len(rs)-1]).all();diff=np.diff(path,axis=0);assert((diff>=0)&(diff<=1)).all() and(diff.sum(1)>0).all()
   hits=[path[path[:,1]==j,0] for j in range(len(rs))];base=np.array([np.median(x) for x in hits]);first=np.array([x.min() for x in hits]);last=np.array([x.max() for x in hits]);new=base.copy()
   for j in range(2,len(base)-2):new[j]=np.clip(np.dot(base[j-2:j+3],[1,4,6,4,1])/16,base[1],base[-2])
   new=np.clip(new,base-1,base+1);V.close(base,z['compressed_phase_old']);V.close(new,z['compressed_phase_new']);V.same(first,z['compressed_span_first']);V.same(last,z['compressed_span_last']);V.close(expand(base[rg],qs,qe),z['phase_old']);V.close(expand(new[rg],qs,qe),z['phase_new']);V.same(qs[first[rg]],z['span_first']);V.same(qe[last[rg]],z['span_last']);V.same(z['domain'],b['domain']);assert len(z['domain'])==len(r['core'])
   value,av=V.mapped(q,z['phase_old']);ok=z['domain']&av&r['available'];V.same(ok,z['point_ok']);V.close(np.where(ok,abs(value-r['values']),np.nan),z['point_delta']);a,aa=V.descriptor(q,z['phase_old'],z['span_first'],z['span_last'],z['domain']);d,dd=V.descriptor(q,z['phase_new'],np.minimum(z['span_first'],np.floor(z['phase_new']).astype(int)),np.maximum(z['span_last'],np.ceil(z['phase_new']).astype(int)),z['domain']);n=len(r['core']);rr,rok=V.descriptor(r,np.arange(n,dtype=float),np.arange(n),np.arange(n),z['domain']);src=np.where(dd,1,np.where(aa,2,0));V.same(src,z['source']);seq=(src>0)&rok;V.same(seq,z['sequence_ok']);V.close(np.where(seq[...,None],abs(np.where((src==1)[...,None],d,a)-rr),np.nan),z['sequence_delta']);po,se,_,_=E.eligibility(z,q,r)
   for mode,strong in [('point',po),('sequence',se)]:
    V.same(strong,z[mode+'_evidence']);ex=z[mode+'_delta']>b[mode+'_bound']+1e-6
    if mode=='sequence':ex=ex.any(-1)
    status=np.select([~z['domain'],~z[mode+'_ok'],~strong,b[mode+'_peer_count']<5,ex],[0,1,2,3,5],default=4).astype(np.uint8);V.same(status,z[mode+'_status'])
   checked+=1
  else:assert row['candidate_output']==row['output'] and row['candidate_pair']==row['pair']
  for mode in ('point','sequence'):
   for g,lo,hi in [('angle',0,15),('position',15,23)]:
    key=mode+'_'+g;base=b[mode+'_status'][:,:,lo:hi];prior=old[mode+'_status'][:,:,lo:hi];now=z[mode+'_status'][:,:,lo:hi];h=np.bincount((base.astype(int)*36+prior.astype(int)*6+now).ravel(),minlength=216).reshape(6,6,6);c[key+'_domain']=int(h[1:].sum());c[key+'_old_new_excess']=int(h[4,5].sum());c[key+'_candidate_new_excess']=int(h[4,:,5].sum());c[key+'_old_excess_now_within']=int(h[4,5,4]);c[key+'_old_supported_now_unknown']=int(h[:,4:,1:4].sum());c[key+'_changed_status']=int(h.sum()-sum(h[:,j,j].sum() for j in range(6)))
  assert dict(c)==row['counts'];total[kind].update(c)
  if (i+1)%2000==0:print('REPEAT_SAMPLE_VERIFY',i+1,flush=True)
 assert len(done['rows'])==12777 and {k:dict(v) for k,v in total.items()}==done['counts'];assert total['original']['realigned']==0 and total['original']['pairs']==7184;assert all(v['realigned']==0 for k,v in total.items() if k!='speed075');R.check(f['protected']);R.write(T/'VERIFIED.json',dict(at=R.now(),status='PASS12777_REUSE_AND799_EXPANDED_ALIGNMENT',counts=done['counts'],new_pairs_arithmetic_checked=checked,result_sha256=R.sha(T/'DONE.json'),verifier_sha256=R.sha(__file__),normal_original_outputs_unchanged=True,all_non_slow_controls_unchanged=True,method='SameCodex separate group-boundary,expanded path,window,delta and status arithmetic. No independent human groundtruth.'))
 print('REPEAT_SAMPLE_VERIFIED',checked,flush=True)
if __name__=='__main__':main()
