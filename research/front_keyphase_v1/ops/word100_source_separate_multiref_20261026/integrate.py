"""Research-only source-separated evidence on unchanged individual DTW pairs."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'1')
import prepare as R
import references as H
import numpy as np,sys,time,traceback
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(R.O.parent/'src'))
from wis_lab.core.observation import Observation
from wis_lab.core import features as F
A=R.mod('raw_channel_adapter',R.O/'word100_public_recovery_bridge_20261026/adapter.py')
M=R.mod('fixed_pair_alignment',R.O/'word100_multipeople_feedback_integration_20261026/run.py')
def state(o,computable):
 # Raw coordinates remain independently available. Codes describe evidence,not correctness.
 s=np.zeros(computable.shape,np.uint8);s[computable]=1
 req=o['requested'][...,None]&computable;s[req]=2
 s[req&~o['mp_identity'][...,None]]=3
 s[req&(o['mp_unresolved']|o['pixel_unresolved'])]=4
 s[req&o['mp_identity_conflict'][...,None]]=3
 s[computable&o['pixel_eligible']]=5
 return s
def baseline(row,n,raw,x,v,assignment,computable):
 N=len(x);out={k:np.zeros((N,2,15),bool) for k in ['mp_unresolved','pixel_unresolved','pixel_eligible']}
 for k in ['requested','mp_identity','mp_identity_conflict']:out[k]=np.zeros((N,2),bool)
 box=row['input']['info']['original_roi'];w,h=row['input']['info']['canvas']
 for z in R.read(row['old_records'])['frames']:
  i=z['i'];nn=int(raw['raw_count'][i]);old=list(raw['hand_xy_px'][i,:nn]);sides=assignment[i,:nn].tolist()
  for s in (0,1):
   if n['usable'][i,33+21*s:54+21*s].any() and s not in sides:old.append(x[i,s]);sides.append(s)
  full=[dict(d,xy=np.asarray(d['xy'],float)) for d in z['mp_full']];crops=[dict(d,xy=np.asarray(d['xy'],float)) for d in z['mp_crop']]
  for s in (0,1):
   if not v[i,s].any():continue
   # A full-image call alone is not the same two-image-scale corroboration contract.
   if str(s) not in z.get('boxes',{}):continue
   out['requested'][i,s]=True;e=H.H.evidence(full,crops,x[i,s],v[i,s],old,sides,raw['pose_xy_px'][i],raw['pose_usable'][i],w,h,box,s)
   pa,pu,px=H.mp_guard(e,full,crops,x[i,s],v[i,s],box,computable[i,s]);out['mp_unresolved'][i,s]=e['unresolved'];out['pixel_unresolved'][i,s]=pu;out['pixel_eligible'][i,s]=px;out['mp_identity'][i,s]=e['identity_confirmed'];out['mp_identity_conflict'][i,s]=e['identity_conflict']
 return out
def prepare_inputs():
 f=R.read(R.U/'FROZEN.json');assert R.read(R.U/'REFERENCES_VERIFIED.json')['clips']==200;R.check(f['pins']);R.check(f['protected']);paths=[Path(__file__),R.U/'references.py',R.U/'verify_references.py',R.U/'REFERENCES_VERIFIED.json',Path(A.__file__),Path(M.__file__),Path(M.P.__file__),Path(M.D.__file__),Path(F.__file__)];pins={str(p):R.sha(p) for p in paths};pins.update(R.read(R.U/'REFERENCES_VERIFIED.json')['pins']);meta={r['clip_id']:r for r in R.read(R.B/'PREP_CONTEXT.json')['ordered_rows']}
 R.write(R.U/'INTEGRATION_FROZEN.json',dict(at=R.now(),pins=pins,source_context_sha256=R.sha(R.U/'FROZEN.json'),method='Original publicTRAIN masked native coordinates and fixed pair method. Quality arrays zero during alignment; source-separated evidence attached afterwards. Baseline existing full/crop available-image checks use identical current-pixel rules; new requested checks replace baseline evidence at requested times only, unrequested baseline remains. All raw values and old evidence retained, withdrawals reported. YOLO conflicts remain audit-only and cannot veto MP or erase coordinates. All angle states nonsemantic. Reference corex2x23 is fixed auditgrid,not required-hand truth.',state_codes={'0':'raw_unobservable_or_degenerate','1':'not_rechecked','2':'image_pair_unavailable','3':'identity_uncertain','4':'shape_or_pixel_disagreement','5':'MP_current_image_corroborated'},gates='No raw/path/value change from evidence. No domain shrink under missingness. Angles independent. Exact old/current source accounting. Whole-action and normal-tolerance verdict null.',training=0,model_inference=0,production_changed=False))
 (R.U/'inputs').mkdir();rows=[];outpins={};counts=Counter()
 for row in f['queries']+f['rows']:
  cid=row['clip_id'];assert meta[cid]['role']=='train_candidate' and 1<=row['signer_id']<=9;inp=row['input'];p=Path(inp['info']['source_arrays']['observation']);obs=Observation.load(str(p));n=R.load(inp['info']['source_arrays']['capture']);raw=R.load(inp['info']['source_arrays']['raw']);assignment=R.load(Path(inp['info']['source_arrays']['capture']).parent/'assignment.npz')['sides'];x=n['xy_px'][:,33:].reshape(-1,2,21,2);v=H.H.roi(x,n['usable'][:,33:].reshape(-1,2,21),inp['info']['original_roi']);xx=np.where(v[...,None],x,np.nan)
  np.testing.assert_array_equal(obs.xy_px[:,33:].reshape(-1,2,21,2),xx);np.testing.assert_array_equal(obs.usable[:,33:].reshape(-1,2,21),v);np.testing.assert_array_equal(obs.frame_id,n['frame_id']);np.testing.assert_array_equal(obs.timestamp_us,n['timestamp_us'])
  values,available,_,scale=A.channel_view(xx,v,np.zeros(v.shape[:2]+(15,),np.uint8),obs.xy_px[:,:33],obs.usable[:,:33],np.zeros(v.shape[:2],bool));base=baseline(row,n,raw,x,v,assignment,available[:,:,:15]);old=state(base,available[:,:,:15]);src=R.V if row in f['queries'] else R.U;oo=R.load(src/'outputs'/(cid+'.npz'))
  if src==R.V:
   gg=R.load(src/'pixel_guard'/(cid+'.npz'));oo['pixel_eligible']=gg['mp_corroborated'];oo['pixel_unresolved']=gg['mp_unresolved']
  current=state(oo,available[:,:,:15]);current=np.where(oo['requested'][...,None],current,old).astype(np.uint8);allreq=base['requested']|oo['requested'];yconf=np.zeros(old.shape,bool)
  if src==R.V:yconf=(oo['yolo_unresolved']|gg['yolo_unresolved']|oo['yolo_identity_conflict'][...,None])&available[:,:,:15]
  q=F.sequence_features(obs,trim=False);core=R.load(row['diagnostics'])['core'];features=q['features'];present=q['present'];oldq=np.zeros_like(available);newq=oldq.copy();oldq[:,:,:15]=old==5;newq[:,:,:15]=current==5
  # Body/position have raw observation availability only; never masquerade as image-verified anatomy.
  op=R.U/'inputs'/(cid+'.npz');np.savez_compressed(op,xy_px=obs.xy_px,usable=obs.usable,timestamp_us=obs.timestamp_us,frame_id=obs.frame_id,features=features,present=present,core=core,values=values,available=available,quality_eligible=np.zeros_like(available),quality_before=oldq,quality_after=newq,state_before=old,state_after=current,requested_before=base['requested'],requested_current=oo['requested'],yolo_conflict=yconf,scale=np.array(scale))
  outpins[str(op)]=R.sha(op);pins[str(p)]=R.sha(p);c=Counter(clips=1,frames=len(core),core_angle_domain=int(core.sum())*30,raw_core_angles=int(available[core,:,:15].sum()),before_core_corroborated=int(oldq[core,:,:15].sum()),after_core_corroborated=int(newq[core,:,:15].sum()),added_core=int((newq&~oldq)[core,:,:15].sum()),withdrawn_core=int((oldq&~newq)[core,:,:15].sum()),yolo_audit_core=int(yconf[core].sum()));counts.update(c);rows.append(dict(clip_id=cid,word_code=row['word_code'],label=row['label'],signer_id=row['signer_id'],output=str(op),source_observation=str(p),source=row,counts=dict(c)))
  if len(rows)%50==0:print('SOURCE_SIDECARS',len(rows),300,flush=True)
 R.check(f['protected']);R.write(R.U/'PREPARED.json',dict(at=R.now(),rows=rows,counts=dict(counts),pins=pins,outputs=outpins));print('PREPARED300',dict(counts),flush=True)
def attach(z,q,r,key):
 base=z['phase_old'];new=z['phase_new'];first=z['span_first'];last=z['span_last'];qq=dict(q,quality_eligible=q[key]);rr=dict(r,quality_eligible=r[key]);om=M.map_values(qq,base,first,last);nm=M.map_values(qq,new,np.minimum(first,np.floor(new).astype(int)),np.maximum(last,np.ceil(new).astype(int)));rm=M.map_values(rr,np.arange(len(r['core']),dtype=float),np.arange(len(r['core'])),np.arange(len(r['core'])));_,_,oq=M.describe(om,z['domain']);_,_,nq=M.describe(nm,z['domain']);_,_,rq=M.describe(rm,z['domain']);point=z['point_ok']&om['quality']&r[key];seq=z['sequence_ok']&np.where(z['source']==1,nq,oq)&rq
 return point,seq
def run():
 f=R.read(R.U/'FROZEN.json');p=R.read(R.U/'PREPARED.json');R.check(p['pins']);R.check(p['outputs']);R.check(f['protected']);R.write(R.U/'PAIR_RUNNING.json',dict(at=R.now(),pid=os.getpid()));(R.U/'pairs').mkdir();rows=[];pins={};tot=Counter();samples={r['clip_id']:R.load(r['output']) for r in p['rows']};beg=time.perf_counter()
 for k,m in enumerate(f['pairs']):
  q=samples[m['query']];r=samples[m['reference']];z=M.pair(q,r);assert not z['sequence_quality'].any();pb,sb=attach(z,q,r,'quality_before');pa,sa=attach(z,q,r,'quality_after');z.update(point_corroborated_before=pb,point_corroborated_after=pa,sequence_corroborated_before=sb,sequence_corroborated_after=sa)
  name=m['query']+'__'+m['reference'];op=R.U/'pairs'/(name+'.npz');np.savez_compressed(op,**z);pins[str(op)]=R.sha(op)
  c=Counter(pairs=1,domain=int(z['domain'].sum()),angle_domain=int(z['domain'][:,:,:15].sum()),raw_point_angles=int(z['point_ok'][:,:,:15].sum()),raw_sequence_angles=int(z['sequence_ok'][:,:,:15].sum()),point_before=int(pb.sum()),point_after=int(pa.sum()),point_added=int((pa&~pb).sum()),point_withdrawn=int((pb&~pa).sum()),sequence_before=int(sb.sum()),sequence_after=int(sa.sum()),sequence_added=int((sa&~sb).sum()),sequence_withdrawn=int((sb&~sa).sum()));tot.update(c);rows.append(dict(**m,pair=str(op),counts=dict(c)))
  if (k+1)%25==0:print('SOURCE_PAIRS',k+1,200,flush=True)
 R.check(f['protected']);R.write(R.U/'PAIRS_DONE.json',dict(at=R.now(),rows=rows,pins=pins,counts=dict(tot),seconds=time.perf_counter()-beg,whole_action_correct=None,normal_tolerance=None,training=0,model_inference=0,operating_changed=False));(R.U/'PAIR_RUNNING.json').unlink();print('PAIRS200',dict(tot),flush=True)
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:
  p=R.U/'INTEGRATION_FIRST_ERROR.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
