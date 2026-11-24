"""Attach additional image evidence to byte-identical prior raw DTW outputs."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'1')
import prepare as R
from pathlib import Path
import numpy as np,sys,time,traceback
from collections import Counter
N=R.O/'word100_source_separate_multiref_20261026';sys.path.append(str(N))
S=R.mod('frozen_source_consumer',N/'integrate.py')
def run():
 f=R.read(R.U/'FROZEN.json');im=R.read(R.U/'VERIFIED_IMAGES.json');assert im['clips']==300;R.check(f['pins']);R.check(f['protected']);R.check(im['pins']);p=R.read(N/'PREPARED.json');pairs=R.read(N/'PAIRS_DONE.json');R.check(p['outputs']);R.check(pairs['pins']);pins={str(v):R.sha(v) for v in [Path(__file__),N/'integrate.py',N/'INTEGRATION_VERIFIED.json',R.U/'VERIFIED_IMAGES.json',N/'PAIRS_DONE.json',N/'PREPARED.json']};pins.update(im['pins']);pins.update(p['outputs']);pins.update(pairs['pins']);R.write(R.U/'INTEGRATION_FROZEN.json',dict(at=R.now(),pins=pins,source_context_sha256=R.sha(R.U/'FROZEN.json'),method='Use all200 prior DTW paths and raw point/sequence values verbatim. Only add source-separated MP evidence at previously unrequested core sides. Existing confirmed/conflicting/notobserved states elsewhere unchanged. YOLO audit preserved separately. No relabeling or model score.',training=0,classifier_inference=0));(R.U/'inputs').mkdir();(R.U/'pairs').mkdir();rows=[];ipins={};samples={};inpcounts=Counter()
 for row in p['rows']:
  cid=row['clip_id'];base=R.load(row['output']);o=R.load(R.U/'outputs'/(cid+'.npz'));z={k:v.copy() for k,v in base.items()};req=o['requested'][...,None];current=S.state(o,base['available'][:,:,:15]);z['state_before']=base['state_after'];z['state_after']=np.where(req,current,base['state_after']).astype(np.uint8);z['quality_before']=base['quality_after'];z['quality_after']=base['quality_after'].copy();z['quality_after'][:,:,:15]=z['state_after']==5;z['requested_before']=base['requested_current'];z['requested_current']=base['requested_current']|o['requested'];assert not (base['quality_after']&~z['quality_after']).any();assert not(base['core'][:,None]&base['available'][:,:,:15].any(-1)&~z['requested_current']).any();op=R.U/'inputs'/(cid+'.npz');np.savez_compressed(op,**z);ipins[str(op)]=R.sha(op);samples[cid]=z;c=Counter(clips=1,frames=len(z['core']),core_angle_domain=int(z['core'].sum())*30,raw_core_angles=int(z['available'][z['core'],:,:15].sum()),before_core_corroborated=int(z['quality_before'][z['core']].sum()),after_core_corroborated=int(z['quality_after'][z['core']].sum()),added_core=int((z['quality_after']&~z['quality_before'])[z['core']].sum()),withdrawn_core=int((z['quality_before']&~z['quality_after'])[z['core']].sum()),yolo_audit_core=int(z['yolo_conflict'][z['core']].sum()));inpcounts.update(c);rows.append(dict(**{k:v for k,v in row.items() if k not in ['output','counts']},output=str(op),baseline_input=row['output'],counts=dict(c)))
 R.write(R.U/'PREPARED.json',dict(at=R.now(),rows=rows,outputs=ipins,pins=pins,counts=dict(inpcounts)));print('CORE_SIDECARS300',dict(inpcounts),flush=True)
 out=[];opins={};total=Counter();start=time.perf_counter()
 for k,row in enumerate(pairs['rows']):
  old=R.load(row['pair']);q=samples[row['query']];r=samples[row['reference']];before_p,before_s=S.attach(old,q,r,'quality_before');np.testing.assert_array_equal(before_p,old['point_corroborated_after']);np.testing.assert_array_equal(before_s,old['sequence_corroborated_after']);ap,ass=S.attach(old,q,r,'quality_after');z=dict(old,point_corroborated_before=before_p,sequence_corroborated_before=before_s,point_corroborated_after=ap,sequence_corroborated_after=ass);pp=R.U/'pairs'/Path(row['pair']).name;np.savez_compressed(pp,**z);opins[str(pp)]=R.sha(pp);c=Counter(pairs=1,domain=int(z['domain'].sum()),angle_domain=int(z['domain'][:,:,:15].sum()),raw_point_angles=int(z['point_ok'][:,:,:15].sum()),raw_sequence_angles=int(z['sequence_ok'][:,:,:15].sum()),point_before=int(before_p.sum()),point_after=int(ap.sum()),point_added=int((ap&~before_p).sum()),point_withdrawn=int((before_p&~ap).sum()),sequence_before=int(before_s.sum()),sequence_after=int(ass.sum()),sequence_added=int((ass&~before_s).sum()),sequence_withdrawn=int((before_s&~ass).sum()));total.update(c);out.append(dict(**{k:v for k,v in row.items() if k not in ['pair','counts']},pair=str(pp),baseline_pair=row['pair'],counts=dict(c)))
  if (k+1)%50==0:print('CORE_SOURCE_PAIRS',k+1,200,flush=True)
 R.check(f['protected']);R.write(R.U/'PAIRS_DONE.json',dict(at=R.now(),rows=out,pins=opins,counts=dict(total),seconds=time.perf_counter()-start,training=0,model_inference=0,operating_changed=False,whole_action_correct=None,normal_tolerance=None));print('CORE_INTEGRATED',dict(total),flush=True)
if __name__=='__main__':
 try:run()
 except BaseException:
  p=R.U/'INTEGRATION_FIRST_ERROR.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
