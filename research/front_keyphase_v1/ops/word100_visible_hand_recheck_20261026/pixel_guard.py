"""Same-frame image evidence audit; does not transform or replace coordinates."""
import prepare as R
from pathlib import Path
import numpy as np,sys,traceback
from collections import Counter,defaultdict
U=R.U
CHAIN=[(0,1,2,3,4),(0,5,6,7,8),(0,9,10,11,12),(0,13,14,15,16),(0,17,18,19,20)]
def pair(a,va,b,vb):
 a=np.asarray(a,float);b=np.asarray(b,float);both=va&vb&np.isfinite(a).all(-1)&np.isfinite(b).all(-1);center=(a+b)/2;radius=np.linalg.norm(a-b,axis=1)/2;available=np.zeros(15,bool);unresolved=np.zeros(15,bool)
 for f,c in enumerate(CHAIN):
  widths=[];bad=[];oks=[]
  for j,k in zip(c[:-1],c[1:]):
   ok=bool(both[j] and both[k]);length=np.linalg.norm(center[j]-center[k]);r=radius[j]+radius[k];unc=bool(ok and (min(np.linalg.norm(a[j]-a[k]),np.linalg.norm(b[j]-b[k]),length)<=1e-8 or r>=length));oks.append(ok);bad.append(unc);widths.append(180. if unc else np.degrees(np.arcsin(np.clip(r/max(length,1e-300),0,1))) if ok else np.nan)
  for k in range(3):
   j,m,l=c[k:k+3];idx=f*3+k
   if not(oks[k] and oks[k+1]):continue
   available[idx]=True;v=center[j]-center[m];w=center[l]-center[m];angle=np.degrees(np.arctan2(abs(v[0]*w[1]-v[1]*w[0]),np.dot(v,w)));width=widths[k]+widths[k+1];unresolved[idx]=bad[k] or bad[k+1] or (angle-width<=0 and angle+width>=180)
 return available,unresolved
def mask(x,box):return np.isfinite(x).all(-1)&(x[:,0]>=box[0])&(x[:,0]<box[2])&(x[:,1]>=box[1])&(x[:,1]<box[3])
def fixtures():
 rng=np.random.default_rng(7);a=rng.normal(0,20,(21,2))+[300,250];v=np.ones(21,bool);av,bad=pair(a,v,a,v);assert av.all() and not bad.any();av,bad=pair(a,v,a+1000,v);assert bad.all()
 vv=v.copy();vv[8]=False;av,bad=pair(a,v,a,vv);assert not av[5] and av[:5].all() and av[6:].all()
 theta=.5;rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]);b=a+rng.normal(0,2,a.shape);aa=pair(a,v,b,v);bb=pair(a@rot*1.4+20,v,b@rot*1.4+20,v);assert all(np.array_equal(x,y) for x,y in zip(aa,bb))
 return dict(identical_current_image_pass=True,large_single_prediction_shift_not_hidden=True,joint8_does_not_remove_neighbor_angles=True,common_camera_similarity_invariant=True)
def freeze():
 t=fixtures();R.write(U/'PIXEL_GUARD_FROZEN.json',dict(at=R.now(),image_context=R.sha(U/'IMAGE_FROZEN.json'),code=R.sha(__file__),tests=t,scope='Development diagnostic proposed after observing visible disagreement in first saved public sample; NOT fresh independent validation. All100 frozen source clips,all3185 frames; no new inference or outcome selection.',rule='Original ROI and observed coordinates remain unchanged. Preserve existing similarity-normalized evidence in separate field. For each confirmed unique same-side prediction pair, check original/full,original/crop,andfull/crop in original image pixels using same endpoint-disc/bone/angle degeneracy rule,WITHOUT individual translation,rotation,or scale alignment. No new fitted cutoff. Fused eligibility can only narrow,never create evidence.',decision='Can expose false corroboration masked by similarity fit; cannot prove anatomical correctness. No model/production change. Normal/crossing retention is reported,not assumed.'))
 print('PIXEL_GUARD_FROZEN',flush=True)
def run():
 f=R.read(U/'PIXEL_GUARD_FROZEN.json');assert R.sha(__file__)==f['code'] and R.sha(U/'IMAGE_FROZEN.json')==f['image_context'];im=R.read(U/'IMAGE_FROZEN.json');assert R.read(U/'IMAGE_VERIFIED.json')['clips']==100;outdir=U/'pixel_guard';outdir.mkdir();tot=Counter();groups=defaultdict(Counter);rows=[];pins={}
 for row in im['rows']:
  cid=row['clip_id'];r=R.read(U/'records'/(cid+'.json'));o=R.load(U/'outputs'/(cid+'.npz'));native=R.load(row['input']['info']['source_arrays']['capture']);diag=R.load(row['diagnostics']);n=len(native['frame_id']);xy=native['xy_px'][:,33:].reshape(n,2,21,2);box=row['input']['info']['original_roi'];valid=native['usable'][:,33:].reshape(n,2,21);out={k:np.zeros((n,2,15),bool) for k in ['mp_available','mp_unresolved','mp_corroborated','yolo_available','yolo_unresolved','yolo_corroborated','fused_corroborated']};counts=Counter()
  for z in r['frames']:
   i=z['i']
   for kind in ['mp','yolo']:
    for e in z[kind+'_checks']:
     s=e['side']
     if not(e['identity_confirmed'] and e['pair_available'] and e['full_index'] is not None and e['crop_index'] is not None):continue
     a=xy[i,s].astype(float);b=np.asarray(z[kind+'_full'][e['full_index']]['xy'],float);c=np.asarray(z[kind+'_crop'][e['crop_index']]['xy'],float);va=valid[i,s]&mask(a,box);vb=mask(b,box);vc=mask(c,box);checks=[pair(a,va,b,vb),pair(a,va,c,vc),pair(b,vb,c,vc)];av=np.logical_and.reduce([p[0] for p in checks]);bad=np.logical_or.reduce([p[1] for p in checks]);out[kind+'_available'][i,s]=av;out[kind+'_unresolved'][i,s]=bad;out[kind+'_corroborated'][i,s]=o[kind+'_eligible'][i,s]&av&~bad&~o[kind+'_identity_conflict'][i,s]
   out['fused_corroborated'][i]=o['fused_eligible'][i]&(out['mp_corroborated'][i]|out['yolo_corroborated'][i])&~(out['mp_unresolved'][i]|out['yolo_unresolved'][i])
   for s in z['requested']:
    ct=Counter(hand_frames=1,native_angles=int(o['original_computable'][i,s].sum()),old_mp=int(o['mp_eligible'][i,s].sum()),old_fused=int(o['fused_eligible'][i,s].sum()),pixel_mp=int(out['mp_corroborated'][i,s].sum()),pixel_yolo=int(out['yolo_corroborated'][i,s].sum()),pixel_fused=int(out['fused_corroborated'][i,s].sum()),old_yolo_added=int((o['fused_eligible'][i,s]&~o['mp_eligible'][i,s]).sum()),pixel_yolo_added=int((out['fused_corroborated'][i,s]&~out['mp_corroborated'][i,s]).sum()))
    counts.update(ct)
    for name,yes in [('all',True),('core',diag['core'][i]),('control',z['control']),('crossing_request',bool(z.get('reason',[0,0])[s]&4))]:
     if yes:groups[name].update(ct)
  p=outdir/(cid+'.npz');np.savez_compressed(p,**out);pins[str(p)]=R.sha(p);counts['clips']=1;tot.update(counts);rows.append(dict(clip_id=cid,counts=dict(counts)))
 R.write(U/'PIXEL_GUARD_RESULT.json',dict(at=R.now(),status='PENDING_SEPARATE_CHECK',counts=dict(tot),groups={k:dict(v) for k,v in groups.items()},rows=rows,pins=pins,plan=R.sha(U/'PIXEL_GUARD_FROZEN.json')));print('PIXEL_GUARD_RESULT',dict(tot),flush=True)
if __name__=='__main__':
 try:freeze() if sys.argv[1]=='freeze' else run()
 except BaseException:
  p=U/'PIXEL_GUARD_FIRST_ERROR.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
