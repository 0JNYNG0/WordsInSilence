import prepare as R
from pathlib import Path
import numpy as np,traceback
from collections import Counter,defaultdict
V=R.mod('core_independent_geometry',R.V/'verify_images.py')
P=R.mod('core_independent_pixels',R.V/'verify_pixel.py')
def exactcache(caches,i,b):
 for index in caches:
  rr=index.get(i,{})
  if b is None and 'mp_full' in rr:return rr['mp_full']
  bs=list(rr.get('boxes',{}).values())+[c['box'] for c in rr.get('calls',[]) if c.get('detector','mp')=='mp' and c['box'] is not None]
  if b is not None and b in bs and 'mp_crop' in rr:return [d for d in rr['mp_crop'] if d.get('box')==b]
 return None
def main():
 f=R.read(R.U/'FROZEN.json');R.check(f['pins']);R.check(f['protected']);assert not(R.U/'RUNNING.json').exists();tot=Counter();groups=defaultdict(Counter);pins={};lat=defaultdict(list);by=[]
 for row in f['rows']:
  cid=row['clip_id'];rp=R.U/'records'/(cid+'.json')
  if not rp.exists():continue
  rr=R.read(rp);assert rr['context_sha256']==R.sha(R.U/'FROZEN.json');R.check(rr['outputs']);src=row['source'];inp=src['input'];R.check(inp['input_pins']);assert rr['source_sha256']==R.sha(rr['source_video'])==inp['source']['sha256'];n=R.load(inp['info']['source_arrays']['capture']);raw=R.load(inp['info']['source_arrays']['raw']);assignment=R.load(Path(inp['info']['source_arrays']['capture']).parent/'assignment.npz')['sides'];base=R.load(row['baseline_input']);out=R.load(R.U/'outputs'/(cid+'.npz'));box=inp['info']['original_roi'];w,h=inp['info']['canvas'];x=n['xy_px'][:,33:].reshape(-1,2,21,2);v=V.mask(x,n['usable'][:,33:].reshape(-1,2,21),box);angle=V.independent_angles(x.astype(float),v);V.close(angle,base['values'][:,:,:15]);need=base['core'][:,None]&np.isfinite(angle).any(-1)&~base['requested_current'];V.same(need,out['requested']);V.same(need.any(1),out['target']);V.same(n['frame_id'],out['frame_id']);V.same(n['timestamp_us'],out['timestamp_us']);targets={t['index']:t for t in row['targets']};assert list(targets)==list(np.flatnonzero(need.any(1)))==[r['i'] for r in rr['frames']];cache=[{z['i']:z for z in R.read(row['recent_record'])['frames']},{z['i']:z for z in R.read(src['old_records'])['frames']}];c=Counter()
  for z in rr['frames']:
   i=z['i'];t=targets[i];expected=np.flatnonzero(need[i]).tolist();assert t['sides']==z['requested']==expected;assert t['frame_id']==z['frame_id']==int(n['frame_id'][i]);nn=int(raw['raw_count'][i]);old=list(raw['hand_xy_px'][i,:nn]);sides=assignment[i,:nn].tolist()
   for s in (0,1):
    if n['usable'][i,33+21*s:54+21*s].any() and s not in sides:old.append(x[i,s]);sides.append(s)
   full=V.detections(z['mp_full']);crops=V.detections(z['mp_crop']);assert [e['side'] for e in z['mp_checks']]==expected
   for e in z['mp_checks']:
    s=e['side'];pts=x[i,s][v[i,s]];mid=(pts.max(0)+pts.min(0))/2;half=max(float(np.ptp(pts,axis=0).max())*1.6,64)/2;bb=[max(0,int(np.floor(mid[0]-half))),max(0,int(np.floor(mid[1]-half))),min(w,int(np.ceil(mid[0]+half))),min(h,int(np.ceil(mid[1]+half)))];assert bb==t['boxes'][str(s)]
    av,un,ident,conflict=V.check_evidence(e,full,crops,x[i,s],v[i,s],old,sides,raw['pose_xy_px'][i],raw['pose_usable'][i],w,h,box);mp=np.isfinite(angle[i,s])&av&~un&ident;pa=np.zeros(15,bool);pu=pa.copy()
    if ident and e['pair_available']:
     b=full[e['full_index']]['xy'];d=crops[e['crop_index']]['xy'];res=[P.check_pair(x[i,s].astype(float),v[i,s],b,P.visible(b,box)),P.check_pair(x[i,s].astype(float),v[i,s],d,P.visible(d,box)),P.check_pair(b,P.visible(b,box),d,P.visible(d,box))];pa=np.logical_and.reduce([z[0] for z in res]);pu=np.logical_or.reduce([z[1] for z in res])
    px=mp&pa&~pu&~np.asarray(conflict)
    for key,value in [('mp_eligible',mp),('mp_unresolved',un),('mp_identity',ident),('mp_identity_conflict',conflict),('pixel_available',pa),('pixel_unresolved',pu),('pixel_eligible',px)]:V.same(out[key][i,s],value)
    counts=Counter(hand_frames=1,raw_angles=int(np.isfinite(angle[i,s]).sum()),mp_corroborated=int(mp.sum()),pixel_corroborated=int(px.sum()),image_disagreement=int((un|pu).sum()),identity_conflict=int(conflict));c.update(counts)
   assert len(z['calls'])==1+len(expected);ms=np.zeros(2);cached=np.zeros(2,int)
   for call in z['calls']:
    b=call['box'];j=int(b is not None);srcd=exactcache(cache,i,b);assert call['cached']==(srcd is not None)
    if srcd is not None:
     assert call['ms']==0;cached[j]+=1;key='mp_crop' if j else 'mp_full';dst=z[key] if not j else [d for d in z[key] if d.get('box')==b];assert len(srcd)==len(dst)
     for aa,bb in zip(srcd,dst):
      for k,value in aa.items():
       if k=='xy':V.close(value,bb[k])
       elif k!='requested_side':assert value==bb[k]
    else:assert call['ms']>=0;ms[j]+=call['ms'];lat['crop' if j else 'full'].append(call['ms'])
   V.same(cached,out['cached'][i]);V.close(ms,out['timing'][i]);c['cached_full']+=int(cached[0]);c['cached_crop']+=int(cached[1]);c['new_full']+=1-int(cached[0]);c['new_crop']+=len(expected)-int(cached[1]);c['frames']+=1
  for key in ['mp_eligible','mp_unresolved','mp_identity','mp_identity_conflict','pixel_available','pixel_unresolved','pixel_eligible']:assert not out[key][~need].any()
  for key in ['timing','cached']:assert not out[key][~need.any(1)].any()
  c['clips']=1;tot.update(c);by.append(dict(clip_id=cid,counts=dict(c)));pins[str(rp)]=R.sha(rp);pins.update(rr['outputs'])
  if len(by)%25==0:print('CORE_VERIFIED',len(by),300,flush=True)
 total=len(by);assert total>0
 if total==300:
  assert R.read(R.U/'DONE.json')['clips']==300
  for k,v in f['counts'].items():assert tot[k]==v,(k,tot[k],v)
 R.check(f['protected']);R.check(f['pins']);R.write(R.U/('VERIFIED_IMAGES.json' if total==300 else f'VERIFIED_IMAGES_{total}.json'),dict(at=R.now(),status='PASS_CORE_UNREQUESTED_RECHECK',clips=total,counts=dict(tot),rows=by,pins=pins,context_sha256=R.sha(R.U/'FROZEN.json'),verifier_sha256=R.sha(__file__),latency={k:dict(n=len(v),mean_ms=float(np.mean(v)),p95_ms=float(np.quantile(v,.95)),total_ms=float(sum(v))) for k,v in lat.items()},method='SameCodex separate angle and complex-pixel geometry, current-arm contract replay, exact source/cache/box requests. Not human joint truth or real-timeFPS.'));print('VERIFIED_CORE',total,dict(tot),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  p=R.U/'VERIFY_FIRST_ERROR.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
