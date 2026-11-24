"""Separate saved-array arithmetic for paired evidence; no network or inference."""
import prepare as R
import numpy as np,sys,traceback
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(R.O.parent/'src'))
V=R.mod('saved_angle_verifier',R.V/'verify_images.py')
def mapped(q,index,key):
 lo=np.floor(index).astype(int);hi=np.ceil(index).astype(int);w=(index-lo)[:,None,None];ok=q['available'][lo]&q['available'][hi];value=q['values'][lo]*(1-w)+q['values'][hi]*w;good=q[key][lo]&q[key][hi]
 return value,ok,good
def descriptor(q,index,first,last,domain,key):
 val,av,quality=mapped(q,index,key);n=len(index);vv=np.full(domain.shape+(5,),np.nan);aa=np.zeros(domain.shape,bool);qq=aa.copy()
 for j in range(2,n-2):
  sel=[j-2,j,j+2];lo=int(first[j-2]);hi=int(last[j+2]);ok=domain[j-2:j+3].all(0)&q['available'][lo:hi+1].all(0)&av[sel].all(0);qa=ok&q[key][lo:hi+1].all(0)&quality[sel].all(0);p0,p1,p2=val[sel];vv[j]=np.stack([p0,p1,p2,p1-p0,p2-p1],-1);aa[j]=ok;qq[j]=qa
 return vv,aa,qq
def verify():
 f=R.read(R.U/'FROZEN.json');p=R.read(R.U/'PREPARED.json');res=R.read(R.U/'PAIRS_DONE.json');R.check(f['pins']);R.check(f['protected']);R.check(p['pins']);R.check(p['outputs']);R.check(res['pins']);samples={};tot=Counter();inputcounts=Counter();meta={x['clip_id']:x for x in R.read(R.B/'PREP_CONTEXT.json')['ordered_rows']};cons=R.read(R.U/'INTEGRATION_FROZEN.json');R.check(cons['pins'])
 for row in p['rows']:
  z=R.load(row['output']);n=R.load(row['source_observation']);src=row['source'];diag=R.load(src['diagnostics']);cid=row['clip_id'];assert meta[cid]['role']=='train_candidate'
  for k in ['xy_px','usable','timestamp_us','frame_id']:V.same(z[k],n[k])
  x=n['xy_px'][:,33:].reshape(-1,2,21,2).astype(float);v=n['usable'][:,33:].reshape(-1,2,21);values=np.full(z['values'].shape,np.nan);values[:,:,:15]=V.independent_angles(x,v);available=np.zeros_like(z['available']);available[:,:,:15]=np.isfinite(values[:,:,:15]);pose=n['xy_px'][:,:33];pv=n['usable'][:,:33];sh=pv[:,11]&pv[:,12];width=np.linalg.norm(pose[:,11]-pose[:,12],axis=-1);scale=float(np.median(width[sh&np.isfinite(width)&(width>1e-6)]));origin=(pose[:,11]+pose[:,12])/2;assert scale==float(z['scale']);centre=x[:,:,[0,5,9,13,17]].mean(2);values[:,:,15:17]=(centre-origin[:,None])/scale;available[:,:,15:17]=(v[:,:,[0,5,9,13,17]].all(-1)&sh[:,None])[:,:,None]
  for h,(e,w) in enumerate([(13,15),(14,16)]):
   for j,k in [(17,e),(19,w)]:values[:,h,j:j+2]=(pose[:,k]-origin)/scale;available[:,h,j:j+2]=(pv[:,k]&sh)[:,None]
  values[:,:,21:23]=(x[:,:,0]-origin[:,None])/scale;available[:,:,21:23]=(v[:,:,0]&sh[:,None])[:,:,None];V.close(np.where(available,values,np.nan),z['values']);V.same(available,z['available']);assert not np.isfinite(z['values'][~available]).any()
  core=np.zeros(len(x),bool)
  for a,b in meta[cid]['intervals']:core|=(n['timestamp_us']/1e6>=a)&(n['timestamp_us']/1e6<=b)
  V.same(core,z['core']);V.same(core,diag['core']);assert not z['quality_eligible'].any()
  srcroot=R.V if cid in {r['clip_id'] for r in f['queries']} else R.U;oo=R.load(srcroot/'outputs'/(cid+'.npz'));new=R.load(srcroot/'pixel_guard'/(cid+'.npz'))['mp_corroborated'] if srcroot==R.V else oo['pixel_eligible'];expected=np.where(oo['requested'][...,None],new,z['quality_before'][:,:,:15])&available[:,:,:15];V.same(expected,z['quality_after'][:,:,:15]);assert not z['quality_after'][:,:,15:].any();V.same(z['quality_before'][:,:,:15],z['state_before']==5);V.same(z['quality_after'][:,:,:15],z['state_after']==5);assert not(z['state_after'][~available[:,:,:15]]).any()
  c=Counter(clips=1,frames=len(core),core_angle_domain=int(core.sum())*30,raw_core_angles=int(available[core,:,:15].sum()),before_core_corroborated=int(z['quality_before'][core,:,:15].sum()),after_core_corroborated=int(z['quality_after'][core,:,:15].sum()),added_core=int((z['quality_after']&~z['quality_before'])[core,:,:15].sum()),withdrawn_core=int((z['quality_before']&~z['quality_after'])[core,:,:15].sum()),yolo_audit_core=int(z['yolo_conflict'][core].sum()));assert dict(c)==row['counts'];inputcounts.update(c);samples[cid]=z
 assert len(samples)==300 and dict(inputcounts)==p['counts'];tests=Counter()
 for k,row in enumerate(res['rows']):
  q=samples[row['query']];r=samples[row['reference']];z=R.load(row['pair']);n=len(r['core']);domain=np.broadcast_to(r['core'][:,None,None],(n,2,23));V.same(domain,z['domain']);path=z['path'];assert (path[0]==[0,0]).all() and (path[-1]==[len(q['core'])-1,n-1]).all();diff=np.diff(path,axis=0);assert ((diff>=0)&(diff<=1)).all() and (diff.sum(1)>0).all();ids=[path[path[:,1]==j,0] for j in range(n)];base=np.array([np.median(x) for x in ids]);first=np.array([x.min() for x in ids]);last=np.array([x.max() for x in ids]);V.same(base,z['phase_old']);V.same(first,z['span_first']);V.same(last,z['span_last']);nv=z['phase_new'];assert (np.diff(nv)>=0).all() and abs(nv-base).max()<=1.00001
  val,ok,_=mapped(q,base,'quality_before');point=domain&ok&r['available'];V.same(point,z['point_ok']);V.close(np.where(point,abs(val-r['values']),np.nan),z['point_delta']);assert not np.isfinite(z['point_delta'][~point]).any()
  for mode in ['before','after']:
   key='quality_'+mode;od,ov,oq=descriptor(q,base,first,last,domain,key);nd,no,nq=descriptor(q,nv,np.minimum(first,np.floor(nv).astype(int)),np.maximum(last,np.ceil(nv).astype(int)),domain,key);rd,rv,rq=descriptor(r,np.arange(n,dtype=float),np.arange(n),np.arange(n),domain,key);source=np.where(no,1,np.where(ov,2,0)).astype(np.uint8);V.same(source,z['source']);sv=(source>0)&rv;V.same(sv,z['sequence_ok']);chosen=np.where((source==1)[...,None],nd,od);V.close(np.where(sv[...,None],abs(chosen-rd),np.nan),z['sequence_delta']);V.same(ov&rv,z['sequence_old_ok']);V.close(np.where((ov&rv)[...,None],abs(od-rd),np.nan),z['sequence_old_delta']);_,_,qp=mapped(q,base,key);pp=point&qp&r[key];ss=sv&np.where(source==1,nq,oq)&rq;V.same(pp,z['point_corroborated_'+mode]);V.same(ss,z['sequence_corroborated_'+mode]);assert not(pp&~point).any() and not(ss&~sv).any()
  pb=z['point_corroborated_before'];pa=z['point_corroborated_after'];sb=z['sequence_corroborated_before'];sa=z['sequence_corroborated_after'];c=Counter(pairs=1,domain=int(domain.sum()),angle_domain=int(domain[:,:,:15].sum()),raw_point_angles=int(point[:,:,:15].sum()),raw_sequence_angles=int(sv[:,:,:15].sum()),point_before=int(pb.sum()),point_after=int(pa.sum()),point_added=int((pa&~pb).sum()),point_withdrawn=int((pb&~pa).sum()),sequence_before=int(sb.sum()),sequence_after=int(sa.sum()),sequence_added=int((sa&~sb).sum()),sequence_withdrawn=int((sb&~sa).sum()));assert dict(c)==row['counts'];tot.update(c)
  # Fixed-path negative controls: hide oneangle or onehand; never shorten referencegrid.
  candidates=np.argwhere(point[:,:,:15]);
  if len(candidates):
   j,h,a=candidates[len(candidates)//2];lo=int(np.floor(base[j]));hi=int(np.ceil(base[j]));mut=dict(q,available=q['available'].copy(),values=q['values'].copy());mut['available'][lo:hi+1,h,a]=False;mut['values'][lo:hi+1,h,a]=np.nan;mv,ma,_=mapped(mut,base,'quality_before');mp=domain&ma&r['available'];assert not mp[j,h,a] and np.isnan(np.where(mp,abs(mv-r['values']),np.nan)[j,h,a]);other=np.ones((2,23),bool);other[h,a]=False;V.same(mp[:,other],point[:,other]);tests['single_angle_no_false_zero_or_neighbor_loss']+=1
   mut=dict(q,available=q['available'].copy(),values=q['values'].copy());mut['available'][lo:hi+1,h,:17]=False;mut['available'][lo:hi+1,h,21:]=False;mut['values'][~mut['available']]=np.nan;mv,ma,_=mapped(mut,base,'quality_before');mp=domain&ma&r['available'];assert not mp[j,h,:17].any();V.same(mp[:,h,17:21],point[:,h,17:21]);V.same(mp[:,1-h],point[:,1-h]);tests['hand_absence_keeps_arm_otherhand_and_domain']+=1
  if (k+1)%50==0:print('PAIR_VERIFIED',k+1,200,flush=True)
 assert len(res['rows'])==200 and dict(tot)==res['counts'];R.check(f['protected']);R.write(R.U/'INTEGRATION_VERIFIED.json',dict(at=R.now(),status='PASS300_INPUTS_200_FIXED_PATH_PAIRS',inputs=300,pairs=200,input_counts=dict(inputcounts),pair_counts=dict(tot),negative_controls=dict(tests),result_sha256=R.sha(R.U/'PAIRS_DONE.json'),prepare_sha256=R.sha(R.U/'PREPARED.json'),verifier_sha256=R.sha(__file__),method='SameCodex separate arccos/channel arithmetic and direct source-window mapping; all saved pairs, raw values/masks/domain/source-specific coverage. Original identity algorithm replayed separately in reference verifier. No independent human accuracy or semantic normal-range validation.'))
 print('INTEGRATION_VERIFIED',dict(tot),dict(tests),flush=True)
if __name__=='__main__':
 try:verify()
 except BaseException:
  p=R.U/'VERIFY_INTEGRATION_FIRST_ERROR.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
