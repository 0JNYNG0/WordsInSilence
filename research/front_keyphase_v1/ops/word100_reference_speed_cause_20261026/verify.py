"""Independent oracle-map, discrepancy and status validation; no optimizer call."""
from pathlib import Path
import sys
from collections import Counter,defaultdict
import numpy as np
S=Path(__file__).resolve().parent;U=S.parent/'word100_train_reference_bank_20261026';sys.path.insert(0,str(U))
import prepare as R
V=R.mod('oracle_independent_values',U/'verify_alignment.py')
E=R.mod('oracle_independent_support',U/'verify_range.py')

def main():
 assert not(S/'RUNNING.json').exists();f=R.read(S/'FROZEN.json');done=R.read(S/'DONE.json');R.check(f['pins']);R.check(f['protected']);refs={x['clip_id']:R.load(x['output']) for x in f['references']};cached={};totals=defaultdict(Counter)
 for i,row in enumerate(done['rows']):
  if row['input'] not in cached:cached[row['input']]=R.load(row['input'])
  q=cached[row['input']];r=refs[row['reference']];z=R.load(row['oracle_output']);old=R.load(row['original_pair']);b=R.load(row['original_range']);actual=R.load(row['output']);ix=q['source_index'];V.same(z['domain'],old['domain'])
  for name in ('phase_old','phase_new'):
   V.close(np.interp(z[name],np.arange(len(ix)),ix),old[name]);assert (np.diff(z[name])>=0).all()
  assert (ix[z['span_first']]<=old['span_first']).all() and (ix[z['span_last']]>=old['span_last']).all()
  val,av=V.mapped(q,z['phase_old']);point=z['domain']&av&r['available'];V.same(point,z['point_ok']);V.close(np.where(point,abs(val-r['values']),np.nan),z['point_delta'])
  a,av=V.descriptor(q,z['phase_old'],z['span_first'],z['span_last'],z['domain']);c,cv=V.descriptor(q,z['phase_new'],np.minimum(z['span_first'],np.floor(z['phase_new']).astype(int)),np.maximum(z['span_last'],np.ceil(z['phase_new']).astype(int)),z['domain']);n=len(r['core']);rv,rok=V.descriptor(r,np.arange(n,dtype=float),np.arange(n),np.arange(n),z['domain']);source=np.where(cv,1,np.where(av,2,0)).astype(np.uint8);ok=(source>0)&rok;V.same(source,z['source']);V.same(ok,z['sequence_ok']);V.close(np.where(ok[...,None],abs(np.where((source==1)[...,None],c,a)-rv),np.nan),z['sequence_delta']);po,se,_,_=E.eligibility(z,q,r);count=Counter(pairs=1)
  for mode,strong in [('point',po),('sequence',se)]:
   V.same(strong,z[mode+'_evidence']);ex=z[mode+'_delta']>b[mode+'_bound']+1e-6
   if mode=='sequence':ex=ex.any(-1)
   status=np.select([~z['domain'],~z[mode+'_ok'],~strong,b[mode+'_peer_count']<5,ex],[0,1,2,3,5],default=4).astype(np.uint8);V.same(status,z[mode+'_status'])
   for g,lo,hi in [('angle',0,15),('position',15,23)]:
    key=mode+'_'+g;before=b[mode+'_status'][:,:,lo:hi];after=actual[mode+'_status'][:,:,lo:hi];o=status[:,:,lo:hi];hist=np.bincount((before.astype(int)*36+after.astype(int)*6+o).ravel(),minlength=216).reshape(6,6,6)
    count[key+'_domain']=int(hist[1:].sum());count[key+'_actual_new_excess']=int(hist[4,5,:].sum());count[key+'_oracle_new_excess']=int(hist[4,:,5].sum());count[key+'_actual_excess_oracle_within']=int(hist[4,5,4]);count[key+'_actual_excess_oracle_still_excess']=int(hist[4,5,5]);count[key+'_actual_excess_oracle_unknown']=int(hist[4,5,:4].sum());count[key+'_oracle_lost_support']=int(hist[4:,:,1:4].sum())
  assert dict(count)==row['counts'];totals[row['kind']].update(count)
  if (i+1)%300==0:print('ORACLE_ARITHMETIC_CHECK',i+1,flush=True)
 assert {k:dict(v) for k,v in totals.items()}==done['counts'] and len(done['rows'])==1598;R.check(f['protected']);R.write(S/'VERIFIED.json',dict(at=R.now(),status='PASS1598_SAVED_ORACLE_INDEX_AND_STATUS',pairs=1598,counts=done['counts'],result_sha256=R.sha(S/'DONE.json'),verifier_sha256=R.sha(__file__),scope='SameCodex separate equations; causal diagnostic control,not deployable inference or human phase truth.'))
 print('ORACLE_VERIFY_PASS1598',flush=True)
if __name__=='__main__':main()
