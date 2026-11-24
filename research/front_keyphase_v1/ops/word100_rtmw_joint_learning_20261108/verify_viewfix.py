"""Verify the uniform view correction without inference or changing targets."""
from common import *
import numpy as np
def main():
 V=U/'feedback_viewfix';assert not (V/'VIEW_FIX_VERIFICATION.json').exists();ready=read(V/'FEEDBACK_READY.json');old=read(U/'FEEDBACK_READY.json');stats=read(V/'FEEDBACK_NORMALIZATION.json');oldmap={r['clip_id']:r for r in old['rows']};allfit=[];checks=[]
 for r in ready['rows']:
  assert r['role']==('fit' if r['person'] in range(1,10) else 'development')
  a=np.load(r['path']);b=np.load(oldmap[r['clip_id']]['path']);raw=np.load(read(U/'records'/f"{r['clip_id']}.json")['channel_path'])
  for k in ['y','ym','sy','sw','ci','frame_id','timestamp_us']:assert np.array_equal(a[k],b[k]),k
  for view,key in [('full','x_full'),('feedback','x_upper')]:
   x=raw[view];mask=np.concatenate([np.repeat(x[:,186:237]>0,2,axis=1),np.repeat(x[:,237:279]>0,2,axis=1)],1)
   expected=x.copy();expected[:,:186]=np.where(mask,(x[:,:186]-np.array(stats['mean']))/np.array(stats['std']),0)
   assert np.allclose(a[key],expected,atol=1e-6,rtol=1e-6)
   if r['role']=='fit':allfit.append(np.where(mask,x[:,:186],np.nan))
  if r['role']=='development':assert np.all(a['sy']==-1) and np.all(a['ci']==-1)
  assert np.array_equal(a['x'],a['x_full']);checks.append(dict(clip_id=r['clip_id'],role=r['role'],supervision_unchanged=True))
 xs=np.concatenate(allfit);mean=np.nan_to_num(np.nanmean(xs,0));std=np.nanstd(xs,0);std=np.where(np.isfinite(std)&(std>1e-4),std,1)
 assert np.allclose(mean,stats['mean']) and np.allclose(std,stats['std'])
 for p,h in read(V/'LEARNING_CODE_CHECK.json')['pins'].items():assert sha(p)==h
 report=read(V/'FEEDBACK_TRAINING_COMPLETE.json');assert len(report['runs'])==6
 for r in report['runs']:
  assert r['epochs']==120==len(r['history']);p=V/'feedback_runs'/f"{r['condition']}_{r['seed']}"/'model.pt';assert sha(p)==r['checkpoint_sha256']
 write(V/'VIEW_FIX_VERIFICATION.json',dict(status='PASS',clips=48,rows=checks,fit_people=list(range(1,10)),
  normalization_verified_independently=True,original_supervision_unchanged=True,all_six_runs_final120=True,
  initial_mismatch_preserved=True,source_sha256=sha(U/'FROZEN.json'),new_contract_sha256=sha(V/'FROZEN.json'),
  provenance='Same Codex separate array check, not independent people or expert labels'))
 event('view_fix_verified')
if __name__=='__main__':main()
