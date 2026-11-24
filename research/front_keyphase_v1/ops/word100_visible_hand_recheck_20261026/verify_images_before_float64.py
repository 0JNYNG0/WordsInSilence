"""Saved observations only. Separate geometry arithmetic, no detector reruns."""
import prepare as R
from pathlib import Path
import numpy as np
from collections import Counter,defaultdict
import importlib.util,sys,traceback
U=R.U
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
rule=module('audit_frozen_identity_rule',R.O/'word100_fullframe_hand_merge_20261020/candidate.py')
P=[0,5,9,13,17]
chains=[(0,1,2,3,4),(0,5,6,7,8),(0,9,10,11,12),(0,13,14,15,16),(0,17,18,19,20)]
triples=[c[k:k+3] for c in chains for k in range(3)]
def same(a,b):np.testing.assert_array_equal(a,b)
def close(a,b):np.testing.assert_allclose(np.asarray(a,float),np.asarray(b,float),rtol=1e-6,atol=2e-3,equal_nan=True)
def inside(x,w,h):return np.isfinite(x).all(-1)&(x[...,0]>=0)&(x[...,0]<w)&(x[...,1]>=0)&(x[...,1]<h)
def mask(x,v,box):return v&np.isfinite(x).all(-1)&(x[...,0]>=box[0])&(x[...,0]<box[2])&(x[...,1]>=box[1])&(x[...,1]<box[3])
def independent_angles(x,v):
 out=np.full(x.shape[:-2]+(15,),np.nan)
 for a,(j,k,l) in enumerate(triples):
  q=x[...,j,:]-x[...,k,:];z=x[...,l,:]-x[...,k,:];nq=np.linalg.norm(q,axis=-1);nz=np.linalg.norm(z,axis=-1);ok=v[...,j]&v[...,k]&v[...,l]&(nq>1e-6)&(nz>1e-6)
  out[...,a]=np.where(ok,np.degrees(np.arccos(np.clip((q*z).sum(-1)/np.maximum(nq*nz,1e-300),-1,1))),np.nan)
 return out
def shape(a,va,b,vb):
 # Proper 2D rotation+scale solved with complex least squares, not production SVD.
 both=va&vb&np.isfinite(a).all(-1)&np.isfinite(b).all(-1)
 empty=dict(pair_available=False,available=np.zeros(15,bool),unresolved=np.zeros(15,bool))
 if not both[P].all():return empty
 za=a[:,0]+1j*a[:,1];zb=b[:,0]+1j*b[:,1];ac=za[P].mean();bc=zb[P].mean();aa=za[P]-ac;bb=zb[P]-bc;den=(abs(bb)**2).sum()
 if den<=1e-8 or (abs(aa)**2).sum()<=1e-8:return empty
 g=(bb.conj()*aa).sum()/den
 if abs(g)<=1e-8:return empty
 aligned=(zb-bc)*g+ac;center=(za+aligned)/2;rad=abs(za-aligned)/2
 av=np.zeros(15,bool);un=av.copy()
 for f,c in enumerate(chains):
  widths=[];bones=[];oks=[]
  for j,k in zip(c[:-1],c[1:]):
   ok=bool(both[j] and both[k]);oks.append(ok);length=abs(center[k]-center[j]);r=rad[j]+rad[k]
   bad=bool(ok and (min(abs(za[k]-za[j]),abs(aligned[k]-aligned[j]),length)<=1e-8 or r>=length));bones.append(bad)
   widths.append(180. if bad else (np.degrees(np.arcsin(np.clip(r/max(length,1e-300),0,1))) if ok else np.nan))
  for k in range(3):
   j,m,l=c[k:k+3];idx=f*3+k
   if not(oks[k] and oks[k+1]):continue
   av[idx]=True;v=center[j]-center[m];w=center[l]-center[m]
   ang=np.degrees(np.arctan2(abs((v.conjugate()*w).imag),(v.conjugate()*w).real));width=widths[k]+widths[k+1]
   un[idx]=bones[k] or bones[k+1] or (ang-width<=0 and ang+width>=180)
 return dict(pair_available=True,available=av,unresolved=un,aligned=np.column_stack([aligned.real,aligned.imag]),scale=abs(g))
def detections(z):return [dict(d,xy=np.asarray(d['xy'],float)) for d in z]
def check_evidence(e,full,crops,base,valid,old,sides,pose,pu,w,h,box):
 # Identity contract replay uses the frozen common rule; anatomy not independently labelled.
 ds=rule.merge(full,crops,old,sides,pose,pu,w,h);assert ds==e['decisions']
 s=e['side'];cs=[d for d in ds if d['side']==s and d['reason']=='confirm_existing_hand']
 ic=any(d['reason'] in ('existing_identity_conflict','unresolved_existing_overlap') and (d['side']==s or any(sides[j]==s for j in d['matched_old'])) for d in ds)
 assert ic==e['identity_conflict'] and (len(cs)==1)==e['identity_confirmed']
 av=np.zeros(15,bool);un=av.copy();paired=False
 if len(cs)==1:
  d=cs[0];assert e['full_index']==d['index'];a=full[d['index']]['xy'];good,sc=rule.context(pose,pu,w,h)
  ids=[i for i,c in enumerate(crops) if c['requested_side']==s and rule.arm(c['xy'],pose,good,sc)==s and rule.relation(a,c['xy'],s,s,sc).startswith('same_')]
  if len(ids)==1:
   assert e['crop_index']==ids[0];b=crops[ids[0]]['xy'];sa=shape(base,valid,a,mask(a,inside(a,w,h),box));sb=shape(base,valid,b,mask(b,inside(b,w,h),box))
   for z,key in [(sa,'full_shape'),(sb,'crop_shape')]:
    assert z['pair_available']==e[key]['pair_available'];same(z['available'],e[key]['angle_available']);same(z['unresolved'],e[key]['angle_unresolved'])
    if z['pair_available']:close(z['aligned'],e[key]['aligned_b']);close(z['scale'],e[key]['alignment_scale'])
   av=sa['available']&sb['available'];un=sa['unresolved']|sb['unresolved'];paired=sa['pair_available'] and sb['pair_available']
 same(av,e['available']);same(un,e['unresolved']);assert paired==e['pair_available']
 return av,un,len(cs)==1,ic
def main():
 f=R.read(U/'IMAGE_FROZEN.json');R.check(f['pins']);R.check(f['protected']);tot=Counter();groups=defaultdict(Counter);latency=defaultdict(list);by=[];pins={};why=Counter()
 for row in f['rows']:
  cid=row['clip_id'];rp=U/'records'/(cid+'.json')
  if not rp.exists():continue
  r=R.read(rp);assert r['context_sha256']==R.sha(U/'IMAGE_FROZEN.json');R.check(r['outputs']);inp=row['input'];R.check(inp['input_pins']);assert r['source_sha256']==inp['source']['sha256']==R.sha(r['source_video']);pins[str(rp)]=R.sha(rp);pins.update(r['outputs'])
  n=R.load(inp['info']['source_arrays']['capture']);raw=R.load(inp['info']['source_arrays']['raw']);assignment=R.load(Path(inp['info']['source_arrays']['capture']).parent/'assignment.npz')['sides'];out=R.load(U/'outputs'/(cid+'.npz'));box=inp['info']['original_roi'];w,h=inp['info']['canvas'];N=len(n['frame_id']);x=n['xy_px'][:,33:].reshape(N,2,21,2);valid=mask(x,n['usable'][:,33:].reshape(N,2,21),box)
  angles=independent_angles(x.astype(float),valid);close(angles,out['original_angles']);same(np.isfinite(angles),out['original_computable']);same(n['frame_id'],out['frame_id']);same(n['timestamp_us'],out['timestamp_us'])
  targets={t['index']:t for t in row['targets']};assert [z['i'] for z in r['frames']]==list(targets);same(np.flatnonzero(out['target']),list(targets));counts=Counter();cache={z['i']:z for z in R.read(row['old_records'])['frames']};diag=R.load(row['diagnostics'])
  for z in r['frames']:
   i=z['i'];t=targets[i];assert z['frame_id']==t['frame_id'];expected=[s for s in (0,1) if valid[i,s].any() and (s in t['sides'] or t['control'])];assert z['requested']==expected;same(np.flatnonzero(out['requested'][i]),expected);assert out['control'][i]==t['control']==z['control']
   if z['requested']:assert z['reason']==t['reasons']
   nn=int(raw['raw_count'][i]);old=list(raw['hand_xy_px'][i,:nn]);sides=assignment[i,:nn].tolist()
   for s in (0,1):
    if n['usable'][i,33+21*s:54+21*s].any() and s not in sides:old.append(x[i,s]);sides.append(s)
   mpok=np.zeros((2,15),bool);yok=mpok.copy();mu=mpok.copy();yu=mpok.copy();mi=np.zeros(2,bool);yi=mi.copy();mc=mi.copy();yc=mi.copy()
   for kind in ['mp','yolo']:
    full=detections(z[kind+'_full']);crops=detections(z[kind+'_crop']);echecks=z[kind+'_checks'];assert len({e['side'] for e in echecks})==len(echecks)
    for e in echecks:
     s=e['side'];assert s in expected;a,u,identity,conflict=check_evidence(e,full,crops,x[i,s],valid[i,s],old,sides,raw['pose_xy_px'][i],raw['pose_usable'][i],w,h,box);ok=np.isfinite(angles[i,s])&a&~u&identity
     if kind=='mp':mpok[s]=ok;mu[s]=u;mi[s]=identity;mc[s]=conflict
     else:yok[s]=ok;yu[s]=u;yi[s]=identity;yc[s]=conflict
     why[kind+':'+e['reason']]+=1
   ys=[s for s in expected if (np.isfinite(angles[i,s])&~mpok[s]).any()];assert [e['side'] for e in z['mp_checks']]==expected and [e['side'] for e in z['yolo_checks']]==ys
   fused=(mpok|yok)&~mu&~yu&~(mc|yc)[:,None]
   for key,value in [('mp_eligible',mpok),('mp_unresolved',mu),('mp_identity',mi),('mp_identity_conflict',mc),('yolo_eligible',yok),('yolo_unresolved',yu),('yolo_identity',yi),('yolo_identity_conflict',yc),('fused_eligible',fused)]:same(value,out[key][i])
   ms=np.zeros(4);cached=np.zeros(4,int);call_types=Counter()
   for call in z['calls']:
    kind=call['detector'];b=call['box'];col=(0 if kind=='mp' else 2)+(b is not None);call_types[(kind,b is None)]+=1
    if call['cached']:
     assert call['ms']==0;cached[col]+=1;key=kind+('_full' if b is None else '_crop');assert key in cache[i]
     if b is not None:assert b in list(cache[i].get('boxes' if kind=='mp' else 'yolo_boxes',{}).values())
     src=cache[i][key] if b is None else [d for d in cache[i][key] if d.get('box')==b];dst=z[key] if b is None else [d for d in z[key] if d.get('box')==b]
     assert len(src)==len(dst)
     for da,db in zip(src,dst):
      for k,v in da.items():
       if k=='xy':close(v,db[k])
       else:assert v==db[k]
    else:assert call['ms']>=0;ms[col]+=call['ms'];latency[kind+('_full' if b is None else '_crop')].append(call['ms'])
   assert call_types[('mp',True)]==int(bool(expected)) and call_types[('mp',False)]==len(expected) and call_types[('yolo',True)]==int(bool(ys)) and call_types[('yolo',False)]==len(ys)
   close(ms,out['timing'][i]);same(cached,out['cached'][i]);counts['cached_calls']+=int(cached.sum());counts['fresh_calls']+=len(z['calls'])-int(cached.sum());counts['frame_targets']+=1
   for s in expected:
    vals=Counter(hand_frames=1,native_angles=int(np.isfinite(angles[i,s]).sum()),mp_corroborated=int(mpok[s].sum()),yolo_corroborated=int(yok[s].sum()),fused_corroborated=int(fused[s].sum()),added_by_yolo=int((fused[s]&~mpok[s]).sum()),held_after_yolo=int((mpok[s]&~fused[s]).sum()),mp_shape_conflict=int(mu[s].sum()),yolo_shape_conflict=int(yu[s].sum()),mp_identity_conflict=int(mc[s]),yolo_identity_conflict=int(yc[s]))
    counts.update(vals);groups['all'].update(vals)
    for name,flag in [('core',diag['core'][i]),('control',t['control']),('triggered',s in t['sides']),('overlap_request',bool(t['reasons'][s]&4)),('reappearance_request',bool(t['reasons'][s]&1)),('partial_request',bool(t['reasons'][s]&2))]:
     if flag:groups[name].update(vals)
  unselected=~out['target']
  for k in ['requested','mp_eligible','mp_unresolved','yolo_eligible','yolo_unresolved','fused_eligible','mp_identity','yolo_identity','mp_identity_conflict','yolo_identity_conflict','timing','cached']:assert not out[k][unselected].any()
  totalside=out['original_computable'].sum();assert counts['native_angles']<=totalside;tot.update(counts);tot['clips']+=1;by.append(dict(clip_id=cid,word_code=row['word_code'],signer_id=row['signer_id'],counts=dict(counts)))
  if len(by)%10==0:print('IMAGE_OUTPUT_VERIFIED',len(by),flush=True)
 total=len(by);assert total>0
 if total==100:assert R.read(U/'IMAGE_DONE.json')['clips']==100 and tot['frame_targets']==3185 and not(U/'RUNNING.json').exists()
 R.check(f['pins']);R.check(f['protected']);R.write(U/('IMAGE_VERIFIED.json' if total==100 else f'IMAGE_VERIFIED_{total}.json'),dict(at=R.now(),status='PASS_SAVED_IMAGE_CONTRACT',clips=total,counts=dict(tot),groups={k:dict(v) for k,v in groups.items()},decision_reasons=dict(why),latency={k:dict(n=len(v),mean_ms=float(np.mean(v)),median_ms=float(np.median(v)),p95_ms=float(np.quantile(v,.95)),total_ms=float(sum(v))) for k,v in latency.items()},rows=by,pins=pins,context_sha256=R.sha(U/'IMAGE_FROZEN.json'),verifier_sha256=R.sha(__file__),method='Independent complex geometry and arccos angle arithmetic; frozen identity-rule replay from saved raw detections; source/ROI/target/cache/timing/fusion preservation. Same Codex,not independent human keypoint ground truth.',limitations='No true joint accuracy,semantic correctness,recognition improvement or deployment verified. Evidence coverage is not correctness. Cachedcalls excluded from fresh latency,not end-to-endFPS.'))
 print('IMAGE_VERIFIED',total,dict(tot),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  p=U/'VERIFY_IMAGE_FIRST_ERROR.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf-8')
  raise
