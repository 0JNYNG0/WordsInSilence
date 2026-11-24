from pathlib import Path
import sys,os,time,traceback
from collections import Counter,defaultdict
import numpy as np
T=Path(__file__).resolve().parent;U=T.parent/'word100_train_reference_bank_20261026';sys.path.insert(0,str(U))
import prepare as R
import candidate as C
M=R.mod('sample_repeat_pair',T.parent/'word100_multipeople_feedback_integration_20261026/run.py');G=R.mod('sample_repeat_range',U/'range.py')

def freeze():
 assert not(T/'FROZEN.json').exists();p=R.read(U/'PREPARED.json');normal=R.read(U/'RANGE_DONE.json');controls=R.read(U/'CONTROLS_DONE.json');assert R.read(U/'ANALYSIS_VERIFIED.json')['pairs']==5593
 inputs={r['output'] for r in p['rows']}|{r['input'] for r in controls['rows']};inventory={};pins={str(x):R.sha(x) for x in [Path(__file__),Path(C.__file__),Path(M.__file__),Path(M.P.__file__),Path(M.D.__file__),U/'PREPARED.json',U/'RANGE_DONE.json',U/'CONTROLS_DONE.json',U/'range.py']}
 for path in sorted(inputs):
  z=R.load(path);start,end,ids=C.groups(z);assert len(ids)==len(z['core']);inventory[path]=dict(frames=len(ids),groups=len(start),repeated_frames=int(len(ids)-len(start)));pins[path]=R.sha(path)
 for x in normal['rows']:pins[x['pair']]=R.sha(x['pair']);pins[x['output']]=R.sha(x['output'])
 for x in controls['rows']:pins[x['pair']]=R.sha(x['pair']);pins[x['output']]=R.sha(x['output'])
 R.write(T/'FROZEN.json',dict(at=R.now(),inputs=p['rows'],normal=normal['rows'],controls=controls['rows'],inventory=inventory,pins=pins,protected=R.read(U/'FROZEN.json')['protected'],authorization='Approved autonomous public recognition/feedback improvement; isolated input-time matching candidate, no models/operating changed.',hypothesis='Exactly duplicated consecutive observations change ordinary DTW cost accumulation/tie choices under slowed resampling. Quotient only identical complete observation/evidence rows for path search; expand path phases and all evidence back to original source and reference grids.',constraints='No original-source-index oracle in candidate. No tolerance,word-specific rule,hand reassignment,coordinate smoothing,criterion recalibration or classifier run. Equal geometry with different visibility/image support/core remains distinct. A-B-A motion not merged. Original coordinate/evidence files read-only. Every source frame in matched window checked,reference denominator never shrinks. Fast sampling that loses distinct observations is not repaired by this candidate.',comparison='All7184original named-reference pairs and5593fixed controls. Reuse saved pair/range if neither input contains exact repeated observations. Changed input pair only computed once. Original empirical bounds fixed. Report speed and omission/uncertainty separately, no decreased support counted as improvement. Synthetic duplicate invariance does not prove real speed/generalized correctness.',training=0,new_detector=0,new_classifier=0,operating_changed=False))
 for name in ('outputs','records','starts'):(T/name).mkdir()
 print('REPEATED_OBSERVATION_FROZEN',len(inventory),sum(v['repeated_frames'] for v in inventory.values()),flush=True)

def run():
 f=R.read(T/'FROZEN.json');R.check(f['pins']);R.check(f['protected']);R.write(T/'RUNNING.json',dict(at=R.now(),pid=os.getpid()));meta={r['clip_id']:r for r in f['inputs']};cache={};rows=[];totals=defaultdict(Counter);start=time.perf_counter()
 cases=[dict(kind='original',query=x['query'],reference=x['reference'],input=meta[x['query']]['output'],pair=x['pair'],output=x['output'],original_range=x['output'],original_pair=x['pair']) for x in f['normal']]+f['controls']
 for i,x in enumerate(cases):
  kind=x['kind'];name=x['query']+'__'+x['reference']+'__'+kind;record=T/'records'/(name+'.json');assert not record.exists();sp=T/'starts'/(name+'.json');assert not sp.exists();R.write(sp,dict(at=R.now()));rpath=meta[x['reference']]['output'];paths=[x['input'],rpath];changed=any(f['inventory'][p]['repeated_frames'] for p in paths);b=R.load(x['original_range']);old=R.load(x['output'])
  if changed:
   for path in paths:
    if path not in cache:cache[path]=R.load(path)
   q,r=(cache[p] for p in paths);z=C.pair(q,r,M);po,se,_,_=G.support(z,q,r)
   for mode,strong in [('point',po),('sequence',se)]:z[mode+'_evidence']=strong;z[mode+'_status']=G.classify(z['domain'],z[mode+'_ok'],strong,z[mode+'_delta'],b[mode+'_bound'],b[mode+'_peer_count'])
   path=T/'outputs'/(name+'.npz');np.savez_compressed(path,**z)
  else:path=Path(x['output']);z=old
  c=Counter(pairs=1,realigned=int(bool(changed)),reused=int(not changed));statuses={}
  for mode in ('point','sequence'):
   for g,lo,hi in [('angle',0,15),('position',15,23)]:
    key=mode+'_'+g;baseline=b[mode+'_status'][:,:,lo:hi];prior=old[mode+'_status'][:,:,lo:hi];now=z[mode+'_status'][:,:,lo:hi];c[key+'_domain']=int(b['domain'][:,:,lo:hi].sum());c[key+'_old_new_excess']=int(((baseline==4)&(prior==5)).sum());c[key+'_candidate_new_excess']=int(((baseline==4)&(now==5)).sum());c[key+'_old_excess_now_within']=int(((baseline==4)&(prior==5)&(now==4)).sum());c[key+'_old_supported_now_unknown']=int(((prior>=4)&(now>0)&(now<4)).sum());c[key+'_changed_status']=int((prior!=now).sum())
  item=dict(x,candidate_output=str(path),candidate_pair=str(path) if changed else x['pair'],changed=bool(changed),counts=dict(c));R.write(record,dict(at=R.now(),frozen_sha256=R.sha(T/'FROZEN.json'),row=item,pins={str(path):R.sha(path)}));rows.append(item);totals[kind].update(c)
  if (i+1)%500==0:print('REPEAT_SAMPLE_COMPARISONS',i+1,len(cases),flush=True)
 assert len(rows)==12777;R.check(f['protected']);R.write(T/'DONE.json',dict(at=R.now(),rows=rows,counts={k:dict(v) for k,v in totals.items()},seconds=time.perf_counter()-start));(T/'RUNNING.json').unlink();print('REPEAT_SAMPLE_DONE',dict(totals),flush=True)
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:
  p=T/('FIRST_ERROR_'+sys.argv[1]+'.txt')
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
