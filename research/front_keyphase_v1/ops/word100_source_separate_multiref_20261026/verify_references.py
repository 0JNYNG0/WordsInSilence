"""Independent saved-output arithmetic; no detector/model inference."""
import prepare as R
import numpy as np,sys,traceback
from pathlib import Path
from collections import Counter,defaultdict
V=R.mod('independent_shapes',R.V/'verify_images.py')
P=R.mod('independent_pixels',R.V/'verify_pixel.py')
def main():
 f=R.read(R.U/'FROZEN.json');R.check(f['pins']);R.check(f['protected']);tot=Counter();groups=defaultdict(Counter);lat=defaultdict(list);pins={};rows=[]
 for row in f['rows']:
  cid=row['clip_id'];rp=R.U/'records'/(cid+'.json')
  if not rp.exists():continue
  rr=R.read(rp);assert rr['context_sha256']==R.sha(R.U/'FROZEN.json');R.check(rr['outputs']);inp=row['input'];R.check(inp['input_pins']);assert rr['source_sha256']==R.sha(rr['source_video'])==inp['source']['sha256'];pins[str(rp)]=R.sha(rp);pins.update(rr['outputs'])
  n=R.load(inp['info']['source_arrays']['capture']);raw=R.load(inp['info']['source_arrays']['raw']);sides0=R.load(Path(inp['info']['source_arrays']['capture']).parent/'assignment.npz')['sides'];o=R.load(R.U/'outputs'/(cid+'.npz'));box=inp['info']['original_roi'];w,h=inp['info']['canvas'];x=n['xy_px'][:,33:].reshape(-1,2,21,2);v=V.mask(x,n['usable'][:,33:].reshape(-1,2,21),box);a=V.independent_angles(x.astype(float),v);diag=R.load(row['diagnostics']);targets={t['index']:t for t in row['targets']};cache={z['i']:z for z in R.read(row['old_records'])['frames']};count=Counter()
  V.close(a,o['original_angles']);V.same(np.isfinite(a),o['original_computable']);V.same(n['frame_id'],o['frame_id']);V.same(n['timestamp_us'],o['timestamp_us']);assert [z['i'] for z in rr['frames']]==list(targets);V.same(np.flatnonzero(o['target']),list(targets))
  for z in rr['frames']:
   i=z['i'];t=targets[i];assert z['frame_id']==t['frame_id'];expected=[s for s in (0,1) if v[i,s].any() and (s in t['sides'] or t['control'])];assert z['requested']==expected;V.same(np.flatnonzero(o['requested'][i]),expected);assert z['control']==o['control'][i]==t['control'];assert z['reason']==t['reasons']
   nn=int(raw['raw_count'][i]);old=list(raw['hand_xy_px'][i,:nn]);sides=sides0[i,:nn].tolist()
   for s in (0,1):
    if n['usable'][i,33+21*s:54+21*s].any() and s not in sides:old.append(x[i,s]);sides.append(s)
   full=V.detections(z['mp_full']);crops=V.detections(z['mp_crop']);assert [e['side'] for e in z['mp_checks']]==expected
   for e in z['mp_checks']:
    s=e['side'];av,un,identity,conflict=V.check_evidence(e,full,crops,x[i,s],v[i,s],old,sides,raw['pose_xy_px'][i],raw['pose_usable'][i],w,h,box);mp=np.isfinite(a[i,s])&av&~un&identity;pa=np.zeros(15,bool);pu=pa.copy()
    if identity and e['pair_available']:
     b=full[e['full_index']]['xy'];c=crops[e['crop_index']]['xy'];vb=P.visible(b,box);vc=P.visible(c,box);r=[P.check_pair(x[i,s].astype(float),v[i,s],b,vb),P.check_pair(x[i,s].astype(float),v[i,s],c,vc),P.check_pair(b,vb,c,vc)];pa=np.logical_and.reduce([y[0] for y in r]);pu=np.logical_or.reduce([y[1] for y in r])
    px=mp&pa&~pu&~np.asarray(conflict)
    for key,val in [('mp_eligible',mp),('mp_unresolved',un),('mp_identity',identity),('mp_identity_conflict',conflict),('pixel_available',pa),('pixel_unresolved',pu),('pixel_eligible',px)]:V.same(o[key][i,s],val)
    c=Counter(hand_frames=1,native_angles=int(np.isfinite(a[i,s]).sum()),mp_corroborated=int(mp.sum()),pixel_corroborated=int(px.sum()),shape_unresolved=int(un.sum()),pixel_unresolved=int(pu.sum()),identity_conflict=int(conflict));count.update(c)
    for name,flag in [('all',True),('core',diag['core'][i]),('control',t['control']),('overlap_request',bool(t['reasons'][s]&4))]:
     if flag:groups[name].update(c)
   ms=np.zeros(2);cached=np.zeros(2,int);assert len(z['calls'])==int(bool(expected))+len(expected)
   for call in z['calls']:
    b=call['box'];j=int(b is not None)
    if call['cached']:
     assert call['ms']==0;cached[j]+=1;key='mp_full' if b is None else 'mp_crop';src=cache[i][key];dst=z[key]
     if b is not None:assert b in list(cache[i]['boxes'].values());src=[d for d in src if d['box']==b];dst=[d for d in dst if d['box']==b]
     assert len(src)==len(dst)
     for aa,bb in zip(src,dst):
      for k,value in aa.items():
       if k=='xy':V.close(value,bb[k])
       elif k!='requested_side':assert value==bb[k]
    else:assert call['ms']>=0;ms[j]+=call['ms'];lat['crop' if j else 'full'].append(call['ms'])
   V.close(ms,o['timing'][i]);V.same(cached,o['cached'][i]);count['cached_calls']+=int(cached.sum());count['fresh_calls']+=len(z['calls'])-int(cached.sum());count['frame_targets']+=1
  for key in ['mp_eligible','mp_unresolved','mp_identity','mp_identity_conflict','pixel_available','pixel_unresolved','pixel_eligible','requested','timing','cached']:assert not o[key][~o['target']].any()
  count['clips']=1;tot.update(count);rows.append(dict(clip_id=cid,counts=dict(count)))
  if len(rows)%25==0:print('REFERENCE_VERIFY',len(rows),flush=True)
 total=len(rows);assert total>0 and not(R.U/'RUNNING.json').exists()
 if total==200:assert R.read(R.U/'REFERENCES_DONE.json')['clips']==200 and tot['frame_targets']==6749
 R.check(f['protected']);R.check(f['pins']);R.write(R.U/('REFERENCES_VERIFIED.json' if total==200 else f'REFERENCES_VERIFIED_{total}.json'),dict(at=R.now(),status='PASS_SAVED_REFERENCE_GEOMETRY',clips=total,counts=dict(tot),groups={k:dict(v) for k,v in groups.items()},rows=rows,pins=pins,context_sha256=R.sha(R.U/'FROZEN.json'),verifier_sha256=R.sha(__file__),latency={k:dict(n=len(v),mean_ms=float(np.mean(v)),p95_ms=float(np.quantile(v,.95)),total_ms=float(sum(v))) for k,v in lat.items()},method='SameCodex separate complex geometry and arccos arithmetic, frozen identity replay, source/cache/ROI preservation. Not independent anatomical labels.'))
 print('REFERENCES_VERIFIED',total,dict(tot),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  p=R.U/'VERIFY_REFERENCE_FIRST_ERROR.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
