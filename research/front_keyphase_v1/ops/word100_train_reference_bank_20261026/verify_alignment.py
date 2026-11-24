"""Read-only saved-path arithmetic, independent of the pair producer."""
import prepare as R
import numpy as np,traceback
from pathlib import Path
from collections import Counter

def mapped(q,index):
 lo=np.floor(index).astype(int);hi=np.ceil(index).astype(int);w=(index-lo)[:,None,None]
 ok=q['available'][lo]&q['available'][hi]
 return q['values'][lo]*(1-w)+q['values'][hi]*w,ok

def descriptor(q,index,first,last,domain):
 val,av=mapped(q,index);v=np.full(domain.shape+(5,),np.nan);ok=np.zeros(domain.shape,bool)
 for j in range(2,len(index)-2):
  a,b=int(first[j-2]),int(last[j+2]);sel=[j-2,j,j+2]
  ok[j]=domain[j-2:j+3].all(0)&q['available'][a:b+1].all(0)&av[sel].all(0)
  p0,p1,p2=val[sel];v[j]=np.stack([p0,p1,p2,p1-p0,p2-p1],-1)
 return v,ok

def close(a,b):np.testing.assert_allclose(a,b,atol=2e-6,rtol=2e-6,equal_nan=True)
def same(a,b):np.testing.assert_array_equal(a,b)

def main():
 assert not (R.U/'ALIGNMENT_RUNNING.json').exists()
 f=R.read(R.U/'ALIGNMENT_FROZEN.json');done=R.read(R.U/'ALIGNMENT_DONE.json');R.check(f['pins']);R.check(f['protected']);R.check(done['pins'])
 samples={r['clip_id']:R.load(r['input']) for r in f['catalog']};total=Counter();seen=set();controls=Counter();by={}
 for row in f['catalog']:by[row['clip_id']]=row
 for i,row in enumerate(done['rows']):
  q=samples[row['query']];r=samples[row['reference']];z=R.load(row['pair']);key=(row['query'],row['reference']);assert key not in seen;seen.add(key)
  assert by[key[0]]['signer_id']!=by[key[1]]['signer_id'] and by[key[0]]['word_code']==by[key[1]]['word_code']
  if row['reuse']:assert row['pair']==row['reuse']
  n=len(r['core']);domain=np.broadcast_to(r['core'][:,None,None],(n,2,23));same(domain,z['domain']);path=z['path']
  assert (path[0]==[0,0]).all() and (path[-1]==[len(q['core'])-1,n-1]).all()
  diff=np.diff(path,axis=0);assert ((diff>=0)&(diff<=1)).all() and (diff.sum(1)>0).all()
  ids=[path[path[:,1]==j,0] for j in range(n)];base=np.array([np.median(x) for x in ids]);first=np.array([x.min() for x in ids]);last=np.array([x.max() for x in ids])
  same(base,z['phase_old']);same(first,z['span_first']);same(last,z['span_last'])
  new=base.copy()
  if n>=5:
   for j in range(2,n-2):new[j]=np.clip(sum(base[j+k]*w for k,w in zip(range(-2,3),[1,4,6,4,1]))/16,base[1],base[-2])
   new=np.clip(new,base-1,base+1)
  close(new,z['phase_new']);assert (np.diff(new)>=0).all()
  values,valid=mapped(q,base);point=domain&valid&r['available'];same(point,z['point_ok']);close(np.where(point,abs(values-r['values']),np.nan),z['point_delta'])
  od,ov=descriptor(q,base,first,last,domain);nd,nv=descriptor(q,new,np.minimum(first,np.floor(new).astype(int)),np.maximum(last,np.ceil(new).astype(int)),domain);rd,rv=descriptor(r,np.arange(n,dtype=float),np.arange(n),np.arange(n),domain)
  src=np.where(nv,1,np.where(ov,2,0)).astype(np.uint8);same(src,z['source']);ok=(src>0)&rv;oldok=ov&rv;same(ok,z['sequence_ok']);same(oldok,z['sequence_old_ok'])
  close(np.where(ok[...,None],abs(np.where((src==1)[...,None],nd,od)-rd),np.nan),z['sequence_delta']);close(np.where(oldok[...,None],abs(od-rd),np.nan),z['sequence_old_delta'])
  assert not z['quality_fallback'].any() and not z['sequence_quality'].any() and not z['sequence_old_quality'].any()
  same(z['reference_missing'],domain&~r['available']);same(z['reference_times_us'],r['timestamp_us']);same(z['query_times_us'],q['timestamp_us'])
  assert not np.isfinite(z['point_delta'][~point]).any() and not np.isfinite(z['sequence_delta'][~ok]).any()
  c=Counter(pairs=1,reused=int(row['reuse'] is not None),new=int(row['reuse'] is None),domain=int(domain.sum()),point_comparable=int(point.sum()),sequence_comparable=int(ok.sum()));total.update(c)
  # Fixed observed path: missing evidence never creates equality or shrinks the reference grid.
  if i%8==0:
   cells=np.argwhere(point[:,:,:15])
   if len(cells):
    j,h,a=cells[len(cells)//2];lo=int(np.floor(base[j]));hi=int(np.ceil(base[j]));mut=dict(q,available=q['available'].copy(),values=q['values'].copy());mut['available'][lo:hi+1,h,a]=False;mut['values'][lo:hi+1,h,a]=np.nan
    vv,aa=mapped(mut,base);po=domain&aa&r['available'];assert not po[j,h,a] and not(po&~point).any();others=np.ones((2,23),bool);others[h,a]=False;same(po[:,others],point[:,others]);controls['single_angle_withheld_without_neighbor_loss']+=1
    mut=dict(q,available=q['available'].copy(),values=q['values'].copy());mut['available'][:,h,:17]=False;mut['available'][:,h,21:]=False;mut['values'][~mut['available']]=np.nan;vv,aa=mapped(mut,base);po=domain&aa&r['available'];assert not po[:,h,:17].any();same(po[:,1-h],point[:,1-h]);same(po[:,h,17:21],point[:,h,17:21]);controls['whole_hand_unknown_keeps_other_hand_arm_domain']+=1
  if (i+1)%500==0:print('VERIFIED_TRAIN_PATHS',i+1,7184,flush=True)
 assert len(seen)==7184 and dict(total)==done['counts'];R.check(f['protected'])
 R.write(R.U/'ALIGNMENT_VERIFIED.json',dict(at=R.now(),status='PASS7184_SAVED_PATH_ARITHMETIC',counts=dict(total),negative_controls=dict(controls),result_sha256=R.sha(R.U/'ALIGNMENT_DONE.json'),verifier_sha256=R.sha(__file__),method='Same Codex separate arithmetic of every saved path, domain, source selection, angle/position and local changes. Frozen existing DTW optimizer reused; no inference or re-alignment. Neither ground-truth temporal correspondence nor human sign correctness certified.'))
 print('TRAIN_ALIGNMENT_VERIFIED',dict(total),dict(controls),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  p=R.U/'VERIFY_ALIGNMENT_FIRST_ERROR.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
