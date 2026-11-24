"""Recover progress logging failure, retaining the already committed inference."""
from common import *
import numpy as np,traceback
cfg=read(F/'FRESH_FROZEN.json');assert not (F/'FRESH_COMPLETE.json').exists()
for p,h in cfg['pins'].items():assert sha(p)==h,p
meta=read(F/'FRESH_INPUTS.json');assert meta['frozen_sha256']==sha(F/'FRESH_FROZEN.json');assert len(meta['rows'])==20
committed=[]
for r in meta['rows']:
 p=F/'fresh_results'/(r['clip_id']+'_'+r['kind']);rp=p/'RECORD.json'
 if rp.exists():
  rr=read(rp);assert rr['sha256']==sha(r['path']);assert rr['status']=='OK' and sha(rr['result'])==rr['result_sha256'];committed.append(rr)
 else:assert not (p/'after.json').exists(),'Uncommitted output needs explicit recovery'
write(F/'FRESH_EVALUATION_RECOVERY.json',dict(first_errors=[str(p) for p in F.glob('FRESH_EVALUATION_FAILURE_*.txt')],cause='Progress logger received kind twice, after complete first condition was saved',completed_reused=len(committed),recomputed=0,resume_code_sha256=sha(__file__),frozen_comparison_unchanged=True))
rt=load('_resumed_fresh_runtime',F/'runtime.py');start=time.perf_counter();c,a=rt.instantiate();loadtime=time.perf_counter()-start;results=[];saved={(r['clip_id'],r['kind']):r for r in committed}
for r in meta['rows']:
 key=r['clip_id']+'_'+r['kind'];p=F/'fresh_results'/key
 if (r['clip_id'],r['kind']) in saved:results.append(saved[(r['clip_id'],r['kind'])]);continue
 assert sha(r['path'])==r['sha256'];z=np.load(r['path']);raw={k:z[k] for k in ['xy_px','score','frame_id','timestamp_us','source_width','source_height']}
 try:
  b,d,pre,x,sec=rt.one(c,a,raw,r['word_id']);assert b['tcn_top3']==d['tcn_top3'] and b['path']==d['path'];assert len(d['frame_id'])==r['frames'];assert d['reference_video_reads']==0 and d['rgb_model_calls']==0
  write(p/'before.json',b);write(p/'after.json',d);write(p/'before_final_quarantine.json',pre);np.savez_compressed(p/'converted.npz',x=x)
  rr=dict(r,status='OK',before=rt.compact(b),after=rt.compact(d),before_final_quarantine=rt.compact(pre),seconds=sec,added_seconds=d['timing']['completion_identity_seconds'],result=str(p/'after.json'),result_sha256=sha(p/'after.json'),quarantined_hand_frames=int(np.asarray(d['observation_guard']['hand_quarantined']).any(1).sum()))
 except Exception:
  err=traceback.format_exc();write(p/'ERROR.json',dict(error=err));rr=dict(r,status='ERROR',error=err)
 results.append(rr);write(p/'RECORD.json',rr);event('fresh_evaluated',done=len(results),condition=r['kind'],status=rr['status'])
write(F/'FRESH_COMPLETE.json',dict(rows=results,conditions=len(results),neural_calls=a.calls+len(committed),resumed_neural_calls=a.calls,committed_reused=len(committed),resumed_model_load_seconds=loadtime,training=False,selection=False,errors_in_denominator=True));write(F/'CURRENT_EXECUTION.json',dict(status='COMPARISONS_COMPLETE_PENDING_VERIFICATION',active_process=None))
