"""Separate denominator algebra and exact channel/mask/time preservation checks."""
from common import *
import traceback
from collections import Counter
def scale_vector(x,m,b):
 anchors=[b+j for j in [5,9,13,17]];valid=m[:,b,None]&m[:,anchors]
 distances=np.sqrt(np.sum((x[:,anchors]-x[:,b,None])**2,axis=2))[valid].astype(np.float64)
 val=float(np.median(distances)) if len(distances) else float('nan')
 return val if val>1e-6 else float('nan')
def main():
 assert not (U/'PREP_RUNNING.json').exists();c=read(U/'PREP_CONTEXT.json');check(c['pins']);pins={};n=Counter();maxdiff=0.
 for i,r in enumerate(c['rows']):
  cid=r['clip_id'];rp=U/'records'/f'{cid}.json';rec=read(rp);pins[str(rp)]=sha(rp);check(rec['outputs']);pins.update(rec['outputs'])
  qr=read(PREV/'records'/f'{cid}.json')
  for v in VIEWS:
   old=load(r['views'][v]['channels']);raw=load(r['views'][v]['observation']);q=load(qr['views'][v]['Q']);new={a:load(rec['views'][v][a]) for a in ARMS}
   for t in ['whole','context'] if r['role']=='train_candidate' else ['whole']:
    ids=old[t+'_frame_id'];xys=[raw['xy_px'][np.searchsorted(raw['frame_id'],ids)],q['xy_px'][np.searchsorted(q['frame_id'],ids)]];ms=[raw['usable'][np.searchsorted(raw['frame_id'],ids)],q['usable'][np.searchsorted(q['frame_id'],ids)]]
    sc=[[scale_vector(x,m,b) for b in [33,54]] for x,m in zip(xys,ms)]
    for a,which in [('R10',1),('R01',0)]:
     base=[old,q][which]['x_'+t];z=new[a]['x_'+t];assert z.shape==base.shape and z.dtype==np.float32 and np.isfinite(z).all()
     assert np.array_equal(new[a][t+'_frame_id'],ids) and np.array_equal(new[a][t+'_ts'],old[t+'_ts'])
     assert np.array_equal(z[:,:102],base[:,:102]) and np.array_equal(z[:,186:],base[:,186:]),'non-local channel altered'
     mask=z[:,237:279];assert np.all(z[:,102:186].reshape(-1,42,2)[mask==0]==0)
     for s in range(2):
      sl=slice(102+42*s,144+42*s);own=sc[which][s];other=sc[1-which][s]
      if np.isfinite(own) and np.isfinite(other) and own!=other:
       expected=base[:,sl].astype(np.float64)*own/other;diff=float(np.abs(z[:,sl]-expected).max());maxdiff=max(maxdiff,diff);assert np.allclose(z[:,sl],expected,rtol=2e-6,atol=2e-6)
       n['finite_denominators_exchanged']+=1
      else:assert np.array_equal(z[:,sl],base[:,sl]);n['own_denominator_retained']+=1
     n['checked_channels']+=z.size;n['tensors']+=1;n[a+'_'+r['role']+'_changed_tensors']+=not np.array_equal(z,base)
  n[r['role']+'_sources']+=1
  if (i+1)%200==0:print('VERIFY',i+1,flush=True)
 expected=read(U/'PREP_RESULT.json')['counts']
 for k in expected:assert expected[k]==n[k] if k in n else True
 assert n['tensors']==8392 and n['train_candidate_sources']==899 and n['development_only_sources']==300
 check(c['pins']);write(U/'INPUTS_INDEPENDENT.json',{'status':'PASS','counts':dict(n),'max_algebra_rounding_difference':maxdiff,'record_output_pins':pins,'code_sha256':sha(__file__),'new_coordinates_masks_or_time':False},new=True)
 # Existing semantic review remains valid: routes, per-point masks and endpoints
 # were not changed by this intervention. Finite scale effect is the experiment.
 write(U/'READY.json',{'status':'PASS','at':now(),'training_allowed':True,'blockers':[],
  'semantic_basis':'Exact observation-branch masks/global XY/time/interval preservation; prior C/Q visual reviews pinned. No new extraction/routing/point removal.',
  'pins':{**pins,str(U/'INPUTS_INDEPENDENT.json'):sha(U/'INPUTS_INDEPENDENT.json'),str(U/'PREP_RESULT.json'):sha(U/'PREP_RESULT.json')},'user_data_used':False},new=True)
 event(stage='INPUT_VERIFICATION_PASS_READY',counts=dict(n),max_difference=maxdiff)
if __name__=='__main__':
 try:main()
 except BaseException:
  if not (U/'FIRST_VERIFY_ERROR.json').exists():write(U/'FIRST_VERIFY_ERROR.json',{'at':now(),'error':traceback.format_exc()},new=True)
  raise
