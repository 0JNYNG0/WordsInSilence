"""Independent routing and normalized tensor arithmetic. Never grants visual PASS."""
from common import *
import argparse,time,traceback
from collections import Counter
BODY=[0,11,12,13,14,15,16,23,24]
def channels(xy,mask,ts):
 ids=BODY+list(range(33,75));p=xy[:,ids].astype(np.float32);m=mask[:,ids];T=len(p)
 expected=np.zeros((T,280),np.float32);ok=m[:,1]&m[:,2]
 scale=np.median(np.sqrt(np.sum((p[ok,1]-p[ok,2])**2,axis=1)))
 assert scale>0
 for t in range(T):
  if ok[t]:
   origin=(p[t,1]+p[t,2])/np.float32(2)
   for j in range(51):
    if m[t,j]:expected[t,2*j:2*j+2]=(p[t,j]-origin)/scale;expected[t,186+j]=1
 for side,base in enumerate([9,30]):
  lengths=[]
  for t in range(T):
   if m[t,base]:
    for j in [5,9,13,17]:
     if m[t,base+j]:lengths.append(float(np.linalg.norm(p[t,base+j]-p[t,base])))
  if not lengths:continue
  hs=float(np.median(lengths))
  if hs<=1e-6:continue
  for t in range(T):
   for j in range(21):
    if m[t,base] and m[t,base+j]:
     k=side*21+j;expected[t,102+2*k:104+2*k]=(p[t,base+j]-p[t,base])/hs;expected[t,237+k]=1
 if T>1:expected[1:,279]=np.diff(ts).astype(np.float32)/1e6
 return expected

def main(role):
 assert not (U/'PREP_RUNNING.json').exists(),'Wait for preparation to stop'
 c=read(U/'PREP_CONTEXT.json');rows=[r for r in c['rows'] if r['role']==role];assert all((U/'records'/f"{r['clip_id']}.json").exists() for r in rows)
 stats=Counter();maximum=0.;pins={};start=time.monotonic();controls=read(F/'ANNOTATED.json');truth={(x['clip_id'],x['view'],x['index'],x['raw_index']):x for x in controls};normal_regressions=[];all_changes=[]
 for ri,r in enumerate(rows):
  path=U/'records'/f"{r['clip_id']}.json";rec=read(path);check(rec['outputs']);pins[str(path)]=sha(path)
  for view in ['masked','sensor']:
   raw=load(r['views'][view]['raw']);old=load(r['views'][view]['observation']);out=load(U/'observations'/f"{r['clip_id']}_{view}.npz")
   for t,n0 in enumerate(raw['raw_count']):
    for i in range(int(n0)):
     for j in range(21):
      x,y=raw['hand_xy_px'][t,i,j];expected=bool(np.isfinite([x,y]).all() and np.isfinite(raw['raw_score'][t,i]) and raw['raw_score'][t,i]>=.5 and 0<=x<raw['width'][t] and 0<=y<raw['height'][t])
      assert expected==bool(out['point_mask'][t,i,j]);stats['raw_point_masks']+=1
   for arm in ['P','Q']:
    path=rec['views'][view][arm]
    if path is None:stats['unavailable_views']+=1;continue
    z=load(path);T=len(raw['raw_count']);xy=np.full((T,75,2),np.nan,np.float32);m=np.zeros((T,75),bool);xy[:,:33]=old['xy_px'][:,:33];m[:,:33]=old['usable'][:,:33]
    for t,n0 in enumerate(raw['raw_count']):
     sides=out[arm+'_sides'][t,:int(n0)];assert len(sides[sides>=0])==len(set(sides[sides>=0]))
     for i,s in enumerate(sides):
      k=(r['clip_id'],view,t,i)
      if k in truth and truth[k]['verdict']=='normal_identity':
       stats[arm+'_normal_checked']+=1
       if s!=truth[k]['actual_side']:normal_regressions.append({'arm':arm,'key':list(k),'expected':truth[k]['actual_side'],'actual':int(s)})
      if s<0:continue
      for j in range(21):
       q=raw['hand_xy_px'][t,i,j];allowed=out['point_mask'][t,i,j] and r['box'][0]<=q[0]<r['box'][2] and r['box'][1]<=q[1]<r['box'][3]
       if allowed:xy[t,33+21*s+j]=q;m[t,33+21*s+j]=True
    assert np.array_equal(z['xy_px'],xy,equal_nan=True) and np.array_equal(z['usable'],m)
    observed=np.flatnonzero(m[:,33:].any(1));lo,hi=int(observed[0]),int(observed[-1])
    for temporal in ['whole','context'] if role=='train_candidate' else ['whole']:
     if temporal=='whole':a,b=lo,hi
     else:
      first,last=r['intervals'][0];idx=np.flatnonzero((raw['timestamp_us']>=round((first-.3)*1e6))&(raw['timestamp_us']<=round((last+.3)*1e6)));a,b=max(lo,int(idx[0])),min(hi,int(idx[-1]))
     assert np.array_equal(z[temporal+'_frame_id'],raw['frame_id'][a:b+1]) and np.array_equal(z[temporal+'_ts'],raw['timestamp_us'][a:b+1])
     expected=channels(xy[a:b+1],m[a:b+1],raw['timestamp_us'][a:b+1]);actual=z['x_'+temporal]
     assert np.array_equal(expected[:,186:279],actual[:,186:279]) and np.array_equal(expected[:,279],actual[:,279])
     diff=float(np.max(np.abs(expected-actual)));maximum=max(maximum,diff);assert np.allclose(expected,actual,rtol=2e-6,atol=2e-6),(r['clip_id'],view,arm,temporal,diff)
     stats['channel_values']+=expected.size;stats['tensors']+=1
    stats['raw_output_points_preserved']+=int(m[:,33:].sum())
   stats['frame_views']+=len(raw['raw_count'])
  all_changes.extend([{'clip_id':r['clip_id'],**x} for x in rec['changes']]);stats['records']+=1
  if (ri+1)%100==0:print('VERIFY',ri+1,'/',len(rows),flush=True)
 result={'status':'PASS_NUMERIC_ONLY' if not normal_regressions else 'FAIL_NORMAL_REGRESSION','role':role,'counts':dict(stats),'normal_regressions':normal_regressions,'max_coordinate_difference':maximum,'record_pins':pins,'seconds':time.monotonic()-start,'semantic_review_pending':True,'training_allowed':False,'code_sha256':sha(__file__)}
 write(U/f'{role}_INPUTS_INDEPENDENT.json',result,new=True);write(U/f'{role}_CHANGES.json',all_changes,new=True);print(json.dumps({k:v for k,v in result.items() if k!='record_pins'}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--role',default='train_candidate');a=p.parse_args()
 try:main(a.role)
 except BaseException:
  if not (U/'FIRST_VERIFY_ERROR.json').exists():write(U/'FIRST_VERIFY_ERROR.json',{'at':now(),'error':traceback.format_exc()},new=True)
  raise
