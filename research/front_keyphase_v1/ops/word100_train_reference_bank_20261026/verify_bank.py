import prepare as R
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np,traceback
V=R.mod('bank_independent_original',R.V/'verify_images.py')
N=R.O/'word100_source_separate_multiref_20261026';Z=R.O/'word100_core_image_evidence_20261026'

def verify():
 f=R.read(R.U/'FROZEN.json');R.check(f['pins']);R.check(f['protected']);meta={r['clip_id']:r for r in R.read(R.B/'PREP_CONTEXT.json')['ordered_rows']};todo={r['clip_id']:r for r in f['rows']};old={r['clip_id']:r for r in R.read(Z/'PREPARED.json')['rows']};counts=Counter();words=defaultdict(set);frames=0
 for row in f['catalog']:
  cid=row['clip_id'];r=meta[cid];assert r['role']=='train_candidate' and r['signer_id']==row['signer_id'] and r['word_code']==row['word_code'];assert 1<=row['signer_id']<=9;words[row['word_code']].add(row['signer_id']);z=R.load(row['input']);frames+=len(z['core']);assert not z['quality_eligible'].any()
  if cid in old:
   assert row['evidence_complete'] and Path(row['input'])==Path(old[cid]['output']) and R.sha(row['input'])==R.read(Z/'PREPARED.json')['outputs'][row['input']];continue
  t=todo[cid];assert not row['evidence_complete'];src=t['source'];inp=src['input'];obs=R.load(inp['info']['source_arrays']['observation']);native=R.load(inp['info']['source_arrays']['capture']);raw=R.load(inp['info']['source_arrays']['raw']);box=inp['info']['original_roi'];w,h=inp['info']['canvas'];x=native['xy_px'][:,33:].reshape(-1,2,21,2);valid=V.mask(x,native['usable'][:,33:].reshape(-1,2,21),box)
  for key in ('xy_px','usable','timestamp_us','frame_id'):np.testing.assert_array_equal(z[key],obs[key])
  np.testing.assert_array_equal(z['xy_px'][:,33:].reshape(-1,2,21,2),np.where(valid[...,None],x,np.nan));np.testing.assert_array_equal(z['usable'][:,33:].reshape(-1,2,21),valid);np.testing.assert_array_equal(z['core'],R.load(src['diagnostics'])['core'])
  deg=V.independent_angles(x.astype(float),valid);V.close(deg,z['values'][:,:,:15]);np.testing.assert_array_equal(np.isfinite(deg),z['available'][:,:,:15]);np.testing.assert_array_equal(z['quality_after'][:,:,:15],z['state_after']==5)
  # Independently reconstruct body-relative positions; no invented hand positions under missingness.
  pose=obs['xy_px'][:,:33];pv=obs['usable'][:,:33];sh=pv[:,11]&pv[:,12];ww=np.linalg.norm(pose[:,11]-pose[:,12],axis=-1);scale=float(np.median(ww[sh&np.isfinite(ww)&(ww>1e-6)]));np.testing.assert_allclose(z['scale'],scale,rtol=0,atol=1e-5);origin=(pose[:,11]+pose[:,12])/2
  hand=obs['xy_px'][:,33:].reshape(-1,2,21,2);hv=obs['usable'][:,33:].reshape(-1,2,21);palm_ids=[0,5,9,13,17]
  expected=(hand[:,:,palm_ids].mean(2)-origin[:,None])/scale;ok=hv[:,:,palm_ids].all(2)&sh[:,None];V.close(z['values'][:,:,15:17],np.where(ok[...,None],expected,np.nan));np.testing.assert_array_equal(z['available'][:,:,15:17],np.repeat(ok[...,None],2,-1))
  for s,(el,wr) in enumerate(((13,15),(14,16))):
   for channel,j in ((17,el),(19,wr)):
    ok=pv[:,j]&sh;V.close(z['values'][:,s,channel:channel+2],np.where(ok[:,None],(pose[:,j]-origin)/scale,np.nan))
   ok=hv[:,s,0]&sh;V.close(z['values'][:,s,21:23],np.where(ok[:,None],(hand[:,s,0]-origin)/scale,np.nan))
  need=z['core'][:,None]&np.isfinite(deg).any(-1)&~z['requested_current'];assert [a['index'] for a in t['targets']]==list(np.flatnonzero(need.any(1)));cache={a['i']:a for a in R.read(src['old_records'])['frames']};rr=R.read(t['recent_record']);assert rr['frames']==R.read(src['old_records'])['frames'];assert R.sha(rr['source_video'])==inp['source']['sha256']
  counts.update(clips=1,frames=int(need.any(1).sum()),hand_frames=int(need.sum()))
  for a in t['targets']:
   i=a['index'];assert a['sides']==np.flatnonzero(need[i]).tolist() and a['frame_id']==int(z['frame_id'][i]);c=cache.get(i,{});counts['cached_full' if 'mp_full' in c else 'new_full']+=1
   for s in a['sides']:
    p=x[i,s][valid[i,s]];mid=(p.max(0)+p.min(0))/2;half=max(float(np.ptp(p,axis=0).max())*1.6,64)/2;bb=[max(0,int(np.floor(mid[0]-half))),max(0,int(np.floor(mid[1]-half))),min(w,int(np.ceil(mid[0]+half))),min(h,int(np.ceil(mid[1]+half)))];assert bb==a['boxes'][str(s)];seen=list(c.get('boxes',{}).values())+[k['box'] for k in c.get('calls',[]) if k.get('detector','mp')=='mp' and k['box'] is not None];counts['cached_crop' if bb in seen and 'mp_crop' in c else 'new_crop']+=1
  if counts['clips']%50==0:print('BANK_INPUT_VERIFIED',counts['clips'],599,flush=True)
 assert len(f['catalog'])==899 and len(words)==100 and sorted(map(len,words.values()))==[8]+[9]*99
 assert dict(counts)==f['counts'],(dict(counts),f['counts']);R.check(f['protected']);R.write(R.U/'BASE_VERIFIED.json',dict(at=R.now(),status='PASS899_ORIGINAL_INPUTS300_REUSED599_NEW',inputs=899,reused=300,new=599,frames=frames,counts=dict(counts),core_domain_unchanged=True,anatomical_truth=False,range_fitted=False,context_sha256=R.sha(R.U/'FROZEN.json'),method='SameCodex separate angle and bodyposition equations/source identity/crop request counters. Source features not changed; no new human truth.'));print('BANK_INPUT_PASS',dict(counts))

if __name__=='__main__':
 try:verify()
 except BaseException:
  p=R.U/'BASE_VERIFY_FIRST_ERROR.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
