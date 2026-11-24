import prepare as R
import numpy as np
from collections import Counter,defaultdict
U=R.U
T=[(c[k],c[k+1],c[k+2]) for c in [(0,1,2,3,4),(0,5,6,7,8),(0,9,10,11,12),(0,13,14,15,16),(0,17,18,19,20)] for k in range(3)]
def visible(x,box):return np.isfinite(x).all(-1)&(x[:,0]>=box[0])&(x[:,0]<box[2])&(x[:,1]>=box[1])&(x[:,1]<box[3])
def check_pair(x,v,y,w):
 a=x[:,0]+1j*x[:,1];b=y[:,0]+1j*y[:,1];m=(a+b)/2;r=abs(a-b)/2;av=np.zeros(15,bool);bad=av.copy();ok=v&w&np.isfinite(x).all(-1)&np.isfinite(y).all(-1)
 for k,(i,j,l) in enumerate(T):
  if not ok[[i,j,l]].all():continue
  av[k]=True;lengths=[abs(m[j]-m[i]),abs(m[l]-m[j])];rs=[r[i]+r[j],r[j]+r[l]]
  collapsed=min(abs(a[j]-a[i]),abs(a[l]-a[j]),abs(b[j]-b[i]),abs(b[l]-b[j]),*lengths)<=1e-8
  if collapsed or any(rr>=d for rr,d in zip(rs,lengths)):bad[k]=True;continue
  widths=sum(np.degrees(np.arctan2(rr,np.sqrt(max(d*d-rr*rr,0)))) for rr,d in zip(rs,lengths));q=m[i]-m[j];z=m[l]-m[j];dot=(q.conjugate()*z).real;cross=abs((q.conjugate()*z).imag);angle=np.degrees(np.arctan2(cross,dot));bad[k]=widths>=max(angle,180-angle)
 return av,bad
def main():
 p=R.read(U/'PIXEL_GUARD_FROZEN.json');s=R.read(U/'PIXEL_GUARD_RESULT.json');f=R.read(U/'IMAGE_FROZEN.json');R.check(s['pins']);assert p['code']==R.sha(U/'pixel_guard.py');counts=Counter();groups=defaultdict(Counter);cells=0
 for row in f['rows']:
  cid=row['clip_id'];r=R.read(U/'records'/(cid+'.json'));o=R.load(U/'outputs'/(cid+'.npz'));n=R.load(row['input']['info']['source_arrays']['capture']);d=R.load(row['diagnostics']);g=R.load(U/'pixel_guard'/(cid+'.npz'));box=row['input']['info']['original_roi'];x=n['xy_px'][:,33:].reshape(-1,2,21,2);v=n['usable'][:,33:].reshape(-1,2,21);expected={k:np.zeros_like(z) for k,z in g.items()}
  for rr in r['frames']:
   i=rr['i']
   for kind in ['mp','yolo']:
    for e in rr[kind+'_checks']:
     if not(e['identity_confirmed'] and e['pair_available'] and e['full_index'] is not None and e['crop_index'] is not None):continue
     h=e['side'];a=x[i,h].astype(float);b=np.asarray(rr[kind+'_full'][e['full_index']]['xy'],float);c=np.asarray(rr[kind+'_crop'][e['crop_index']]['xy'],float);va=v[i,h]&visible(a,box);vb=visible(b,box);vc=visible(c,box);res=[check_pair(a,va,b,vb),check_pair(a,va,c,vc),check_pair(b,vb,c,vc)];av=np.logical_and.reduce([z[0] for z in res]);bad=np.logical_or.reduce([z[1] for z in res]);expected[kind+'_available'][i,h]=av;expected[kind+'_unresolved'][i,h]=bad;expected[kind+'_corroborated'][i,h]=o[kind+'_eligible'][i,h]&av&~bad&~o[kind+'_identity_conflict'][i,h];cells+=45
   expected['fused_corroborated'][i]=o['fused_eligible'][i]&(expected['mp_corroborated'][i]|expected['yolo_corroborated'][i])&~expected['mp_unresolved'][i]&~expected['yolo_unresolved'][i]
   for h in rr['requested']:
    c=Counter(hand_frames=1,native_angles=int(o['original_computable'][i,h].sum()),old_mp=int(o['mp_eligible'][i,h].sum()),old_fused=int(o['fused_eligible'][i,h].sum()),pixel_mp=int(expected['mp_corroborated'][i,h].sum()),pixel_yolo=int(expected['yolo_corroborated'][i,h].sum()),pixel_fused=int(expected['fused_corroborated'][i,h].sum()),old_yolo_added=int((o['fused_eligible'][i,h]&~o['mp_eligible'][i,h]).sum()),pixel_yolo_added=int((expected['fused_corroborated'][i,h]&~expected['mp_corroborated'][i,h]).sum()));counts.update(c)
    for name,yes in [('all',True),('core',d['core'][i]),('control',rr['control']),('crossing_request',bool(rr.get('reason',[0,0])[h]&4))]:
     if yes:groups[name].update(c)
  for k in g:np.testing.assert_array_equal(g[k],expected[k])
  counts['clips']+=1
 assert dict(counts)==s['counts'] and {k:dict(v) for k,v in groups.items()}==s['groups'];R.check(f['pins']);R.check(f['protected']);R.write(U/'PIXEL_GUARD_VERIFIED.json',dict(at=R.now(),status='PASS_DIRECT_IMAGE_EVIDENCE_ARITHMETIC',clips=100,pair_angle_calculations=cells,counts=dict(counts),result_sha256=R.sha(U/'PIXEL_GUARD_RESULT.json'),verifier_sha256=R.sha(__file__),method='Complex pixel distances,angle-local endpoint intervals and atan2 uncertainty widths; source/fusion/aggregate checks. Not manual anatomical groundtruth.'));print('PIXEL_GUARD_VERIFIED',dict(counts),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  import traceback
  p=U/'VERIFY_PIXEL_FIRST_ERROR.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
