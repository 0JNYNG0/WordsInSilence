"""Independent validation of saved perturbations, paths and empirical-band outputs."""
import prepare as R
import numpy as np,traceback
from pathlib import Path
from collections import Counter
V=R.mod('independent_control_paths',R.U/'verify_alignment.py');E=R.mod('independent_control_evidence',R.U/'verify_range.py')

def validate_input(q,z,kind,groups):
 n=len(q['core']);core=np.flatnonzero(q['core']);m=int(core[len(core)//2]);start=max(0,min(n-3,m-1));cut=np.arange(start,start+3);ix=np.arange(n)
 if kind=='speed075' or kind=='speed125':
  rate=.75 if kind=='speed075' else 1.25;count=int(np.ceil((n-1)/rate))+1;ix=np.rint(np.linspace(0,n-1,count)).astype(int);timestamps=np.rint(np.linspace(q['timestamp_us'][0],q['timestamp_us'][0]+(q['timestamp_us'][-1]-q['timestamp_us'][0])/rate,count)).astype(np.int64)
 elif kind=='delete3':ix=np.concatenate([ix[:start],ix[start+3:]]);timestamps=np.rint(q['timestamp_us'][0]+np.arange(len(ix))*np.median(np.diff(q['timestamp_us']))).astype(np.int64)
 else:timestamps=q['timestamp_us'];
 V.same(z['source_index'],ix);V.same(z['deleted_candidate_indices'],cut);V.same(z['timestamp_us'],timestamps);assert (np.diff(timestamps)>0).all();expected={k:(v[ix].copy() if v.ndim and v.shape[0]==n else v.copy()) for k,v in q.items()};expected['timestamp_us']=timestamps
 if kind.startswith(('missing','gap3')):
  h=0 if kind.endswith('L') else 1;frames=np.arange(n) if kind.startswith('missing') else cut;mask=np.zeros(n,bool);mask[frames]=True;V.same(z['affected_frames'],mask)
  hs=np.arange(33+21*h,54+21*h);expected['xy_px'][np.ix_(frames,hs)]=np.nan;expected['usable'][np.ix_(frames,hs)]=False
  for frame in frames:
   for k in ['available','quality_before','quality_after','quality_eligible']:
    expected[k][frame,h,np.r_[0:17,21:23]]=False
   expected['values'][frame,h,np.r_[0:17,21:23]]=np.nan;expected['state_before'][frame,h]=0;expected['state_after'][frame,h]=0
  side='L' if h==0 else 'R';cursor=0
  for gi,(name,width) in enumerate(groups):
   if name=='bimanual' or (name.endswith('_'+side) and not name.startswith('arm_')):expected['features'][frames,cursor:cursor+width]=np.nan;expected['present'][frames,gi]=False
   cursor+=width
 else:assert not z['affected_frames'].any()
 for k in q:V.same(z[k],expected[k])
 return ix,cut

def verify_path(q,r,z):
 n=len(r['core']);domain=np.broadcast_to(r['core'][:,None,None],(n,2,23));V.same(z['domain'],domain);p=z['path'];assert(p[0]==[0,0]).all() and(p[-1]==[len(q['core'])-1,n-1]).all();d=np.diff(p,axis=0);assert((d>=0)&(d<=1)).all() and(d.sum(1)>0).all();ids=[p[p[:,1]==j,0] for j in range(n)];base=np.array([np.median(x) for x in ids]);first=np.array([x.min() for x in ids]);last=np.array([x.max() for x in ids]);V.same(base,z['phase_old']);V.same(first,z['span_first']);V.same(last,z['span_last']);new=base.copy()
 if n>=5:
  for j in range(2,n-2):new[j]=np.clip(np.dot(base[j-2:j+3],[1,4,6,4,1])/16,base[1],base[-2])
  new=np.clip(new,base-1,base+1)
 V.close(new,z['phase_new']);val,valid=V.mapped(q,base);ok=domain&valid&r['available'];V.same(ok,z['point_ok']);V.close(np.where(ok,abs(val-r['values']),np.nan),z['point_delta']);od,ov=V.descriptor(q,base,first,last,domain);nd,nv=V.descriptor(q,new,np.minimum(first,np.floor(new).astype(int)),np.maximum(last,np.ceil(new).astype(int)),domain);rd,rv=V.descriptor(r,np.arange(n,dtype=float),np.arange(n),np.arange(n),domain);src=np.where(nv,1,np.where(ov,2,0)).astype(np.uint8);V.same(src,z['source']);seq=(src>0)&rv;V.same(seq,z['sequence_ok']);V.close(np.where(seq[...,None],abs(np.where((src==1)[...,None],nd,od)-rd),np.nan),z['sequence_delta']);assert not z['sequence_quality'].any()

def main():
 assert not(R.U/'CONTROLS_RUNNING.json').exists();f=R.read(R.U/'CONTROLS_FROZEN.json');done=R.read(R.U/'CONTROLS_DONE.json');R.check(f['pins']);R.check(f['protected']);R.check(done['pins']);samples={x['clip_id']:R.load(x['output']) for x in f['inputs']};F=R.mod('control_schema_only',R.O/'word100_multipeople_feedback_integration_20261026/run.py').F;groups=[(g,F.GROUPS[g][0]) for g in F.GROUP_ORDER];input_cache={};total=Counter();tests=Counter()
 for i,row in enumerate(done['rows']):
  qid,rid,kind=row['query'],row['reference'],row['kind'];key=(qid,kind)
  if key not in input_cache:
   q=R.load(row['input']);ix,cut=validate_input(samples[qid],q,kind,groups);input_cache[key]=(q,ix,cut);tests['exact_transformed_public_inputs']+=1
  q,ix,cut=input_cache[key];r=samples[rid];z=R.load(row['pair']);out=R.load(row['output']);b=R.load(row['original_range']);old=R.load(row['original_pair']);verify_path(q,r,z);V.same(b['domain'],z['domain']);po,se,_,_=E.eligibility(z,q,r);statuses={};c=Counter(pairs=1,domain=int(z['domain'].sum()));target=(old['phase_old']>=cut.min()-2)&(old['phase_old']<=cut.max()+2)&r['core'];assert row['target_reference_frames']==int(target.sum());V.same(np.asarray(row['original_frame_mapping']),ix)
  drift=abs(np.interp(z['phase_old'],np.arange(len(ix)),ix)-old['phase_old']);V.close(row['alignment_drift_mean'],drift[r['core']].mean());V.close(row['alignment_drift_max'],drift[r['core']].max())
  for mode,strong in [('point',po),('sequence',se)]:
   V.same(out[mode+'_evidence'],strong);d=z['domain'];raw=z[mode+'_ok'];count=b[mode+'_peer_count'];ex=z[mode+'_delta']>b[mode+'_bound']+1e-6;ex=ex.any(-1) if mode=='sequence' else ex;status=np.select([~d,~raw,~strong,count<5,ex],[0,1,2,3,5],default=4).astype(np.uint8);V.same(status,out[mode+'_status']);statuses[mode]=status;before=b[mode+'_status'];oldok=(before==4)|(before==5);newok=(status==4)|(status==5)
   for group,sl in [('angle',slice(0,15)),('position',slice(15,23))]:
    for code in range(1,6):c[f'{mode}_{group}_state{code}']=int((status[:,:,sl]==code).sum())
   c[mode+'_old_supported_to_unknown']=int((oldok&~newok).sum());c[mode+'_old_beyond_to_within']=int(((before==5)&(status==4)).sum());c[mode+'_new_beyond']=int(((before!=5)&(status==5)).sum());c[mode+'_target_new_beyond']=int(((before!=5)&(status==5))[target].sum());c[mode+'_target_lost_support']=int((oldok&~newok)[target].sum());assert int(sum(c[f'{mode}_{g}_state{k}'] for g in ['angle','position'] for k in range(1,6)))==int(d.sum())
   if kind.startswith('missing'):
    h=0 if kind.endswith('L') else 1;assert not(status[:,h,:17]>=4).any() and not(status[:,h,21:]>=4).any();c[mode+'_wholehand_no_false_normal']=1
  assert dict(c)==row['counts'];total.update(c);tests['fixed_original_bounds_and_reference_domain']+=1
  if (i+1)%500==0:print('VERIFIED_EMPIRICAL_CONTROLS',i+1,f['expected_pairs'],flush=True)
 assert dict(total)==done['counts'] and len(done['rows'])==f['expected_pairs'];R.check(f['protected']);R.write(R.U/'CONTROLS_VERIFIED.json',dict(at=R.now(),status='PASS_SAVED_TRANSFORMED_INPUT_PATH_RANGE_ARITHMETIC',pairs=len(done['rows']),counts=dict(total),tests=dict(tests),result_sha256=R.sha(R.U/'CONTROLS_DONE.json'),verifier_sha256=R.sha(__file__),method='SameCodex separate indexing and masks, saved path and status arithmetic, frozen calibration reuse. Structural controls rather than human semantic truth. Missed omissions and speed regressions must still be reported.'));print('CONTROL_ARITHMETIC_VERIFIED',len(done['rows']),dict(tests),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  p=R.U/'VERIFY_CONTROLS_FIRST_ERROR.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
