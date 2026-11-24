"""Diagnosis: separate known resampling from re-optimized temporal matching.

Oracle correspondence exists ONLY in the synthetic speed control; never a user input.
"""
from pathlib import Path
import sys, os, time, traceback
from collections import Counter, defaultdict
import numpy as np
S=Path(__file__).resolve().parent; U=S.parent/'word100_train_reference_bank_20261026'
sys.path.insert(0,str(U))
import prepare as R
M=R.mod('speed_cause_pair',S.parent/'word100_multipeople_feedback_integration_20261026/run.py')
G=R.mod('speed_cause_range',U/'range.py')

def inverse(ix,t):
 """Exact inverse of the piecewise-linear original-index mapping, including plateaus."""
 values,starts,n=np.unique(ix,return_index=True,return_counts=True);ends=starts+n-1
 answer=[]
 for v in t:
  k=int(np.searchsorted(values,v))
  if k<len(values) and abs(values[k]-v)<1e-10:answer.append((starts[k]+ends[k])/2)
  else:
   assert 0<k<len(values);w=(v-values[k-1])/(values[k]-values[k-1]);answer.append(ends[k-1]*(1-w)+starts[k]*w)
 z=np.array(answer);np.testing.assert_allclose(np.interp(z,np.arange(len(ix)),ix),t,atol=1e-9)
 return z

def oracle(old,q,r):
 ix=q['source_index'];base=inverse(ix,old['phase_old']);new=inverse(ix,old['phase_new']);first=np.floor(inverse(ix,old['span_first'])).astype(int);last=np.ceil(inverse(ix,old['span_last'])).astype(int);d=old['domain'];n=len(r['core'])
 om=M.map_values(q,base,first,last);nm=M.map_values(q,new,np.minimum(first,np.floor(new).astype(int)),np.maximum(last,np.ceil(new).astype(int)));rm=M.map_values(r,np.arange(n,dtype=float),np.arange(n),np.arange(n))
 od,ov,_=M.describe(om,d);nd,nv,_=M.describe(nm,d);rd,rv,_=M.describe(rm,d);source=np.where(nv,1,np.where(ov,2,0)).astype(np.uint8);ok=(source>0)&rv;point=d&om['valid']&r['available']
 return dict(domain=d,phase_old=base,phase_new=new,span_first=first,span_last=last,point_ok=point,point_delta=np.where(point,abs(om['values']-r['values']),np.nan).astype(np.float32),sequence_ok=ok,sequence_delta=np.where(ok[...,None],abs(np.where((source==1)[...,None],nd,od)-rd),np.nan).astype(np.float32),source=source)

def freeze():
 assert not(S/'FROZEN.json').exists();assert R.read(U/'ANALYSIS_VERIFIED.json')['pairs']==5593
 done=R.read(U/'CONTROLS_DONE.json');prep=R.read(U/'PREPARED.json');rows=[r for r in done['rows'] if r['kind'] in ('speed075','speed125')];assert len(rows)==1598
 needed={x['reference'] for x in rows};refs=[x for x in prep['rows'] if x['clip_id'] in needed];pins={str(p):R.sha(p) for p in [Path(__file__),U/'CONTROLS_DONE.json',U/'CONTROLS_VERIFIED.json',U/'ANALYSIS.json',U/'ANALYSIS_VERIFIED.json',U/'range.py',Path(M.__file__),Path(M.P.__file__)]}
 for row in refs:pins[row['output']]=R.sha(row['output'])
 for row in rows:
  for k in ('input','pair','output','original_range','original_pair'):pins[row[k]]=R.sha(row[k])
 tests=[]
 for ix in [np.array([0,0,1,2,2,3]),np.array([0,1,3,4]),np.arange(5)]:
  t=np.linspace(ix[0],ix[-1],37);inverse(ix,t);tests.append(True)
 R.write(S/'FROZEN.json',dict(at=R.now(),rows=rows,references=refs,pins=pins,protected=R.read(U/'FROZEN.json')['protected'],authorization='Continue approved autonomous public research goal; investigate material speed-regression before warning adoption.',method='Reuse all1598 frozen speed-control results. Transfer old original phase and span to known resampled source-index coordinate using exact piecewise linear inverse. No new optimizer call. Compare previous/new empirical statuses with same fixed bands. Restoring known correspondence isolates optimizer/median/smoothing timing from remaining sample loss/interpolation/support changes. Original correspondence across people is NOT anatomical or phase groundtruth.',gates='No criterion/weights/band/model changes. Existing excess that disappears under oracle counts alignment-sensitive, not a proven realworld fix. New unknown never counted improved. Original normal/crossing coordinates and each reference domain immutable. User/mentor/extra199/200 forbidden.',tests=tests,training=0,classifier_inference=0,new_detector=0,operating_changed=False))
 (S/'outputs').mkdir();(S/'records').mkdir();(S/'starts').mkdir();print('SPEED_CAUSE_FROZEN1598',flush=True)

def run():
 f=R.read(S/'FROZEN.json');R.check(f['pins']);R.check(f['protected']);assert not(S/'RUNNING.json').exists();R.write(S/'RUNNING.json',dict(at=R.now(),pid=os.getpid()));refs={x['clip_id']:R.load(x['output']) for x in f['references']};cache={};rows=[];total=defaultdict(Counter);start=time.perf_counter()
 for i,row in enumerate(f['rows']):
  name=row['query']+'__'+row['reference']+'__'+row['kind'];record=S/'records'/(name+'.json')
  if record.exists():
   saved=R.read(record);assert saved['frozen_sha256']==R.sha(S/'FROZEN.json');R.check(saved['pins']);rows.append(saved['row']);total[row['kind']].update(saved['row']['counts']);continue
  st=S/'starts'/(name+'.json');assert not st.exists();R.write(st,dict(at=R.now()))
  if row['input'] not in cache:cache[row['input']]=R.load(row['input'])
  q=cache[row['input']];r=refs[row['reference']];old=R.load(row['original_pair']);b=R.load(row['original_range']);actual=R.load(row['output']);z=oracle(old,q,r);po,se,_,_=G.support(z,q,r);out=dict(z);c=Counter(pairs=1)
  for mode,support in [('point',po),('sequence',se)]:
   out[mode+'_evidence']=support;status=G.classify(z['domain'],z[mode+'_ok'],support,z[mode+'_delta'],b[mode+'_bound'],b[mode+'_peer_count']);out[mode+'_status']=status
   for g,sl in [('angle',slice(0,15)),('position',slice(15,23))]:
    key=mode+'_'+g;before=b[mode+'_status'][:,:,sl];after=actual[mode+'_status'][:,:,sl];o=status[:,:,sl];new=(before==4)&(after==5);c[key+'_domain']=int(z['domain'][:,:,sl].sum());c[key+'_actual_new_excess']=int(new.sum());c[key+'_oracle_new_excess']=int(((before==4)&(o==5)).sum());c[key+'_actual_excess_oracle_within']=int((new&(o==4)).sum());c[key+'_actual_excess_oracle_still_excess']=int((new&(o==5)).sum());c[key+'_actual_excess_oracle_unknown']=int((new&(o<4)).sum());c[key+'_oracle_lost_support']=int(((before>=4)&(o>0)&(o<4)).sum())
  op=S/'outputs'/(name+'.npz');np.savez_compressed(op,**out);rr=dict(row,oracle_output=str(op),counts=dict(c));pins={str(op):R.sha(op)};R.write(record,dict(at=R.now(),frozen_sha256=R.sha(S/'FROZEN.json'),row=rr,pins=pins));rows.append(rr);total[row['kind']].update(c)
  if (i+1)%200==0:print('SPEED_CAUSE',i+1,1598,flush=True)
 R.check(f['protected']);R.write(S/'DONE.json',dict(at=R.now(),rows=rows,counts={k:dict(v) for k,v in total.items()},seconds=time.perf_counter()-start,oracle_is_deployable=False));(S/'RUNNING.json').unlink();print('SPEED_CAUSE_DONE',dict(total),flush=True)

if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:
  p=S/('FIRST_ERROR_'+sys.argv[1]+'.txt')
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
