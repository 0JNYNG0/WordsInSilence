"""Leave-one-recorded-person-out empirical discrepancy bands, per named reference.

No synthetic average pose, no sign-correct verdict, no pooling of angle and position units.
The body-position branch is raw observation only; finger angles require current-image support.
"""
import prepare as R
import numpy as np,sys,time,traceback,os
from pathlib import Path
from collections import Counter,defaultdict
MIN_PEERS=5

def mapped_support(z,index,key):
 a=np.floor(index).astype(int);b=np.ceil(index).astype(int)
 return z[key][a]&z[key][b]

def seq_support(z,index,first,last,domain,key):
 mapped=mapped_support(z,index,key);out=np.zeros_like(domain)
 for j in range(2,len(index)-2):
  lo=int(first[j-2]);hi=int(last[j+2]);out[j]=domain[j-2:j+3].all(0)&z[key][lo:hi+1].all(0)&mapped[[j-2,j,j+2]].all(0)
 return out

def support(pair,q,r):
 # Geometry is not image-confirmed anatomy. Preserve this distinction in the output labels.
 qq=dict(q);rr=dict(r)
 for z in [qq,rr]:
  z['audit_support']=z['available'].copy();z['audit_support'][:,:,:15]&=z['quality_after'][:,:,:15]
  z['active_support']=np.broadcast_to(z['core'][:,None,None],z['available'].shape)
 domain=pair['domain'];base=pair['phase_old'];new=pair['phase_new'];first=pair['span_first'];last=pair['span_last'];lo=np.minimum(first,np.floor(new).astype(int));hi=np.maximum(last,np.ceil(new).astype(int));n=len(r['core'])
 po=pair['point_ok']&mapped_support(qq,base,'audit_support')&rr['audit_support']
 a=seq_support(qq,base,first,last,domain,'audit_support');b=seq_support(qq,new,lo,hi,domain,'audit_support');c=seq_support(rr,np.arange(n,dtype=float),np.arange(n),np.arange(n),domain,'audit_support');se=pair['sequence_ok']&np.where(pair['source']==1,b,a)&c
 # Calibration peers cannot set an activity bound using their own preparation/rest-outside-annotation.
 pc=po&mapped_support(qq,base,'active_support')
 ac=seq_support(qq,base,first,last,domain,'active_support');bc=seq_support(qq,new,lo,hi,domain,'active_support');sc=se&np.where(pair['source']==1,bc,ac)
 return po,se,pc,sc

def classify(domain,raw_ok,observed,delta,bound,count):
 """0 outside domain,1 raw missing,2 unconfirmed image,3 reference shortage,4 within band,5 beyond."""
 status=np.zeros_like(domain,np.uint8);status[domain]=1;status[domain&raw_ok]=2;strong=domain&raw_ok&observed;status[strong]=3;calibrated=strong&(count>=MIN_PEERS)
 finite=np.isfinite(bound).all(-1) if bound.ndim==domain.ndim+1 else np.isfinite(bound)
 assert (finite|(count<MIN_PEERS)|~domain).all()
 residual=delta>bound+1e-6;beyond=residual.any(-1) if residual.ndim==domain.ndim+1 else residual
 status[calibrated]=np.where(beyond[calibrated],5,4)
 return status

def bound_from(delta,valid):
 count=valid.sum(0);axis=valid[...,None] if delta.ndim==valid.ndim+1 else valid
 bound=np.max(np.where(axis,delta,-np.inf),axis=0);bound=np.where((count>=MIN_PEERS)[...,None] if bound.ndim==count.ndim+1 else count>=MIN_PEERS,bound,np.nan)
 return bound,count

def selftest():
 d=np.ones((7,3,2,23));v=np.ones_like(d,bool);b,c=bound_from(d,v);domain=np.ones((3,2,23),bool);q=np.ones_like(b);s=classify(domain,domain,domain,q,b,c);assert (s==4).all()
 q[1,0,0]=2;assert classify(domain,domain,domain,q,b,c)[1,0,0]==5
 unseen=domain.copy();unseen[1,0,0]=False;s=classify(domain,unseen,unseen,q,b,c);assert s[1,0,0]==1 and s[1,0,1]==4
 four=v.copy();four[4:]=False;b,c=bound_from(d,four);assert (classify(domain,domain,domain,q,b,c)==3).all()
 b,c=bound_from(d,v);strong=domain.copy();strong[1,0,0]=False;assert classify(domain,domain,strong,q,b,c)[1,0,0]==2
 ds=np.ones((7,3,2,23,5));b,c=bound_from(ds,v);q=np.ones_like(b);q[1,0,0,4]=2;assert classify(domain,domain,domain,q,b,c)[1,0,0]==5
 return dict(normal_observed_within=True,one_frame_difference_preserved=True,missing_never_normal=True,four_peers_withheld=True,unconfirmed_finger_withheld=True,change_only_excess_retained=True)

def protocol():
 tests=selftest();f=R.read(R.U/'FROZEN.json');a=R.read(R.U/'ALIGNMENT_FROZEN.json')
 R.write(R.U/'RANGE_PROTOCOL.json',dict(at=R.now(),input_scope='Public TRAIN899 recordedIDs1-9,100words. Each heldout person is absent from every empirical calibration source; every other named reference retained. This is not unseen classifier evaluation.',
  method='At each named-reference frame/side/channel, use the maximum absolute peer-to-this-reference residual among at least5 other recorded people. Each local sequence is front/middle/back values and2changes along one coherent reference. No average pose, no best-reference selection, no cross-channel unit mixing. Query/reference raw values and original differences remain intact.',
  evidence='Angle support requires both images corroborated at interpolation endpoints and every sequence window source. Body/position support is raw-observation only, separately labelled. Calibration peer mapped frames/window must be inside its own annotated core; evaluation domain is reference core and never shrinks with query missingness. Unobserved/unconfirmed/calibration shortage all distinct, never normal.',
  interpretation='Within empirical band means only comparable to these observed public residuals, not correct sign, not validated human joint accuracy or permissible sign error. Maximum of6/7 potential peers is not a population confidence interval. TRAIN people already used in past classifier training.',
  gates='All raw point data preserved. No unknown to within-band conversion without support>=5 and query/reference evidence. No frame deletion by minimum duration. Angles/positions/changes separate. Corrected speed/rest/crossing tests must preserve raw evidence and domain; no test results tune the maximum/min5 or select a model. If most channels lack5supported peers, report unusable/limited scope rather than loosen it.',
  planned_controls='Deterministic geometry/evidence fixtures plus fixed-public-index original/speed0.75/speed1.25/threeframe deletion/wholehand missing/threeframe missing. Recompute transformed time path only once, keep original perreference bound and frame domain. Explicit known warp, true transformed timestamps; protect original arrays. Report missed deletion as limitation, not semantic mandatory-motion annotation. No uncertain/inconclusive counted as success.',
  test_selection='One original perword for speed/deletion stress tests, choose smallest recordedID without scores; deletion centered at temporal middle of annotated core,3original input frames. All899 normal original comparisons retained. No word-specific exceptions. These are structural perturbations,not linguistic error groundtruth.',
  status_codes={'0':'outside_reference_domain','1':'raw_query_or_reference_missing','2':'current_image_evidence_unconfirmed','3':'fewer_than5_supported_calibration_people','4':'within_observed_empirical_band_not_correctness','5':'beyond_observed_empirical_band_not_error_verdict'},
  pins={str(p):R.sha(p) for p in [Path(__file__),R.U/'FROZEN.json',R.U/'ALIGNMENT_FROZEN.json']},minimum_peers=MIN_PEERS,selftests=tests,new_training=0,new_classifier_inference=0,new_detector_inference=0,personal_data=0,operating_changed=False))
 print('RANGE_PROTOCOL_FROZEN',tests,flush=True)

def freeze():
 p=R.read(R.U/'RANGE_PROTOCOL.json');R.check(p['pins']);iv=R.read(R.U/'INPUTS_VERIFIED.json');av=R.read(R.U/'ALIGNMENT_VERIFIED.json');assert iv['counts']['clips']==899 and av['counts']['pairs']==7184
 prepared=R.read(R.U/'PREPARED.json');a=R.read(R.U/'ALIGNMENT_DONE.json');R.check(prepared['outputs']);R.check(a['pins']);R.write(R.U/'RANGE_FROZEN.json',dict(at=R.now(),rows=prepared['rows'],pairs=a['rows'],pins={**prepared['outputs'],**a['pins'],**{str(x):R.sha(x) for x in [R.U/'PREPARED.json',R.U/'INPUTS_VERIFIED.json',R.U/'ALIGNMENT_DONE.json',R.U/'ALIGNMENT_VERIFIED.json',R.U/'RANGE_PROTOCOL.json',Path(__file__)]}},protected=R.read(R.U/'FROZEN.json')['protected']))
 for name in ['range_outputs','range_records','range_starts']:(R.U/name).mkdir()
 print('RANGE_INPUTS_FROZEN899_7184',flush=True)

def run():
 f=R.read(R.U/'RANGE_FROZEN.json');R.check(f['pins']);R.check(f['protected']);R.write(R.U/'RANGE_RUNNING.json',dict(at=R.now(),pid=os.getpid()));samples={x['clip_id']:R.load(x['output']) for x in f['rows']};meta={x['clip_id']:x for x in f['rows']};by=defaultdict(list)
 for p in f['pairs']:by[p['reference']].append(p)
 rows=[];pins={};counts=Counter();start=time.perf_counter()
 for rid,rr in sorted(by.items()):
  items=[]
  for row in rr:
   z=R.load(row['pair']);po,se,pc,sc=support(z,samples[row['query']],samples[rid]);items.append((row,z,po,se,pc,sc))
  for row,z,po,se,pc,sc in items:
   name=row['query']+'__'+rid;record=R.U/'range_records'/(name+'.json')
   if record.exists():
    saved=R.read(record);assert saved['context_sha256']==R.sha(R.U/'RANGE_FROZEN.json');R.check(saved['outputs']);rows.append(saved['row']);pins.update(saved['outputs']);counts.update(saved['counts']);continue
   sp=R.U/'range_starts'/(name+'.json');assert not sp.exists();R.write(sp,dict(at=R.now()));cal=[x for x in items if meta[x[0]['query']]['signer_id']!=meta[row['query']]['signer_id']];ids=[x[0]['query'] for x in cal];assert row['query'] not in ids and rid not in ids and len(ids) in [6,7]
   pb,pn=bound_from(np.array([x[1]['point_delta'] for x in cal]),np.array([x[4] for x in cal]));sb,sn=bound_from(np.array([x[1]['sequence_delta'] for x in cal]),np.array([x[5] for x in cal]));ps=classify(z['domain'],z['point_ok'],po,z['point_delta'],pb,pn);ss=classify(z['domain'],z['sequence_ok'],se,z['sequence_delta'],sb,sn)
   op=R.U/'range_outputs'/(name+'.npz');np.savez_compressed(op,domain=z['domain'],point_status=ps,sequence_status=ss,point_bound=pb,sequence_bound=sb,point_peer_count=pn.astype(np.uint8),sequence_peer_count=sn.astype(np.uint8),point_evidence=po,sequence_evidence=se,point_delta=z['point_delta'],sequence_delta=z['sequence_delta'])
   c=Counter(pairs=1,domain=int(z['domain'].sum()))
   for mode,status in [('point',ps),('sequence',ss)]:
    for group,sl in [('angle',slice(0,15)),('position',slice(15,23))]:
     for code in range(1,6):c[f'{mode}_{group}_state{code}']=int((status[:,:,sl]==code).sum())
   item=dict(query=row['query'],reference=rid,word_code=row['word_code'],heldout_record_id=meta[row['query']]['signer_id'],calibration_ids=ids,pair=row['pair'],output=str(op));out={str(op):R.sha(op)};R.write(record,dict(at=R.now(),context_sha256=R.sha(R.U/'RANGE_FROZEN.json'),row=item,counts=dict(c),outputs=out));rows.append(item);pins.update(out);counts.update(c)
  if len(rows)%200<9:print('EMPIRICAL_RANGE_PAIRS',len(rows),7184,flush=True)
 assert len(rows)==7184;R.check(f['protected']);R.write(R.U/'RANGE_DONE.json',dict(at=R.now(),rows=rows,pins=pins,counts=dict(counts),seconds=time.perf_counter()-start,verified=False,normal_sign_verdict=None,corrective_instruction=None));(R.U/'RANGE_RUNNING.json').unlink();print('EMPIRICAL_RANGES_COMPUTED',dict(counts),flush=True)

if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:
  p=R.U/('RANGE_'+sys.argv[1].upper()+'_FIRST_ERROR.txt')
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
