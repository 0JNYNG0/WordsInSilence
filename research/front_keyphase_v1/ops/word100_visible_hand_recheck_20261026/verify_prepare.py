import prepare as R
import numpy as np
from collections import Counter,defaultdict

def main():
 f=R.read(R.U/'FROZEN.json');R.check(f['pins']);R.check(f['protected']);vals=defaultdict(list);counts=Counter();byperson=Counter();byword=defaultdict(list);verified=0
 for ri,row in enumerate(f['rows']):
  inp=row['input'];assert inp['view']=='masked' and 1<=row['signer_id']<=9
  n=R.load(inp['info']['source_arrays']['capture']);d=R.load(row['diagnostics']);x=n['xy_px'][:,33:].reshape(-1,2,21,2).astype(float);valid=n['usable'][:,33:].reshape(-1,2,21)&np.isfinite(x).all(-1);roi=inp['info']['original_roi'];inside=valid&(x[...,0]>=roi[0])&(x[...,0]<roi[2])&(x[...,1]>=roi[1])&(x[...,1]<roi[3]);visible=inside.any(-1)
  assert np.array_equal(inside,d['inroi']) and np.array_equal(visible,d['visible']);reason=np.zeros(visible.shape,np.uint8)
  returning=visible.copy();returning[1:]&=~visible[:-1];reason[returning]|=1;reason[visible&(inside.sum(-1)<21)]|=2
  shape=np.full(visible.shape,np.nan);rate=shape.copy();angle=np.full(visible.shape+(15,),np.nan)
  for a,tr in enumerate(R.C.TRIPLES):
   u=x[:,:,tr[0]]-x[:,:,tr[1]];v=x[:,:,tr[2]]-x[:,:,tr[1]];ul=np.linalg.norm(u,axis=-1);vl=np.linalg.norm(v,axis=-1);ok=valid[:,:,tr].all(-1)&(ul>1e-6)&(vl>1e-6);ratio=(u*v).sum(-1)/np.maximum(ul*vl,1e-300);angle[:,:,a]=np.where(ok,np.degrees(np.arccos(np.clip(ratio,-1,1))),np.nan)
  for i in range(len(x)):
   if visible[i].all():
    p0=x[i,0,inside[i,0]];p1=x[i,1,inside[i,1]]
    if min(p0[:,0].max(),p1[:,0].max())>max(p0[:,0].min(),p1[:,0].min()) and min(p0[:,1].max(),p1[:,1].max())>max(p0[:,1].min(),p1[:,1].min()):reason[i]|=4
   if not i:continue
   dt=(n['timestamp_us'][i]-n['timestamp_us'][i-1])/1e6
   if dt>.2:continue
   for h in (0,1):
    if not (visible[i,h] and visible[i-1,h]):continue
    delta=np.abs(angle[i,h]-angle[i-1,h]);good=np.isfinite(delta)
    if good.any():rate[i,h]=np.median(delta[good])/dt
    common=valid[i,h]&valid[i-1,h]
    if not common[[0,5,9,13,17]].all():continue
    a=x[i-1,h,:,0]+1j*x[i-1,h,:,1];b=x[i,h,:,0]+1j*x[i,h,:,1];am=a[[0,5,9,13,17]].mean();bm=b[[0,5,9,13,17]].mean();ac=a[[0,5,9,13,17]]-am;bc=b[[0,5,9,13,17]]-bm;den=(abs(ac)**2).sum();scale=np.sqrt((abs(bc)**2).mean())
    if den<=1e-8 or scale<=1e-6:continue
    gain=(ac.conj()*bc).sum()/den
    if abs(gain)<=1e-8:continue
    shape[i,h]=np.sqrt((abs((a[common]-am)*gain+bm-b[common])**2).mean())/scale
  np.testing.assert_allclose(shape,d['shape_residual'],atol=1e-8,rtol=1e-8,equal_nan=True);np.testing.assert_allclose(rate,d['angle_rate'],atol=2e-3,rtol=1e-6,equal_nan=True)
  # Use saved full-precision source statistic for exact threshold boundary checks.
  reason[d['shape_residual']>f['thresholds']['shape_residual']]|=8;reason[d['angle_rate']>f['thresholds']['angle_rate']]|=16;reason[~visible]=0
  for k in ['shape_residual','angle_rate']:
   z=d[k][d['core']];vals[k].extend(z[np.isfinite(z)].tolist())
  ids=sorted(set(np.flatnonzero(reason.any(1)).tolist()+row['controls']));assert [x['index'] for x in row['targets']]==ids
  for item in row['targets']:
   i=item['index'];assert item['frame_id']==int(n['frame_id'][i]) and item['sides']==np.flatnonzero(reason[i]).tolist() and item['reasons']==reason[i].tolist()
  counts['frames']+=len(x);counts['core_frames']+=int(d['core'].sum());counts['visible_hand_frames']+=int(visible.sum());counts['trigger_frames']+=int(reason.any(1).sum());counts['trigger_hand_frames']+=int((reason>0).sum())
  for bit in [1,2,4,8,16]:counts['reason_'+str(bit)]+=int(((reason&bit)>0).sum())
  byword[row['word_code']].append(row['clip_id']);verified+=1
  if row['clip_id'] in f['selected_ids']:byperson[row['signer_id']]+=1
  if (ri+1)%200==0:print('VERIFIED_TRAIN_SCHEDULE',ri+1,899,flush=True)
 for k,vs in vals.items():
  x=sorted(vs);pos=.99*(len(x)-1);a,b=int(np.floor(pos)),int(np.ceil(pos));q=x[a]+(pos-a)*(x[b]-x[a]);assert abs(q-f['thresholds'][k])<1e-10 and len(x)==f['calibration_samples'][k]
 assert dict(counts)==f['counts'];selected=[min(x,key=lambda s:R.hashlib.sha256(s.encode()).hexdigest()) for k,x in sorted(byword.items())];assert selected==f['selected_ids'] and len(selected)==100
 R.check(f['pins']);R.check(f['protected']);R.write(R.U/'PREPARED_VERIFIED.json',dict(at=R.now(),status='PASS899_TRAIN_ONLY_SCHEDULE',clips=verified,counts=dict(counts),selected_person_distribution=dict(byperson),selected_frames=f['selected_target_frames'],thresholds=f['thresholds'],method='Separate complex-number similarity fit,arccos angles,explicit bbox/bit scheduling and sorted interpolation for99th percentile. No detector truth implied.',source=R.sha(R.U/'FROZEN.json'),verifier=R.sha(__file__)));print('VERIFIED_SCHEDULE899',dict(byperson),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  import traceback
  p=R.U/'VERIFY_PREP_FIRST_ERROR.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf-8')
  raise
