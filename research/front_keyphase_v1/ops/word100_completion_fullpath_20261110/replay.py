from common import *
rt=load('_fullpath_runtime',F/'runtime.py')
import numpy as np,traceback,os
def main():
 assert not (F/'REPLAY_COMPLETE.json').exists();cfg=read(F/'REPLAY_FROZEN.json');assert sha(F/'REPLAY_INPUTS.json')==cfg['inputs_sha256']
 for p,h in cfg['pins'].items():assert sha(p)==h,p
 write(F/'REPLAY_CODE_PINS.json',{str(p):sha(p) for p in [F/'runtime.py',F/'replay.py',F/'prepare_replay.py',F/'REPLAY_FROZEN.json']});write(F/'CURRENT_EXECUTION.json',dict(status='PUBLIC_RAW_REPLAY',pid=os.getpid()))
 start=time.perf_counter();c,a=rt.instantiate();loadtime=time.perf_counter()-start;results=[]
 for r in read(F/'REPLAY_INPUTS.json')['rows']:
  assert sha(r['path'])==r['sha256'];z=np.load(r['path']);raw={k:z[k] for k in ['xy_px','score','timestamp_us','frame_id','source_width','source_height']};key=r['clip_id']+'_'+r['kind']
  try:
   b,d,pre,x,seconds=rt.one(c,a,raw,r['word_id']);assert b['tcn_top3']==d['tcn_top3'];assert b['path']==d['path'];assert len(d['frame_id'])==r['frames'];assert d['reference_video_reads']==0 and d['rgb_model_calls']==0
   path=F/'replay_results'/key;write(path/'before.json',b);write(path/'after.json',d);write(path/'before_final_quarantine.json',pre);np.savez_compressed(path/'converted.npz',x=x)
   results.append(dict(**r,before=rt.compact(b),after=rt.compact(d),before_final_quarantine=rt.compact(pre),result=str(path/'after.json'),result_sha256=sha(path/'after.json'),status='OK',seconds=seconds,added_seconds=d['timing']['completion_identity_seconds'],quarantined_hand_frames=int(np.asarray(d['observation_guard']['hand_quarantined']).any(1).sum()),tcndtw_identical=True))
  except Exception:
   failure=traceback.format_exc();write(F/'replay_results'/key/'ERROR.json',dict(error=failure));results.append(dict(**r,status='ERROR',error=failure));event('replay_failure',key=key)
  if len(results)%12==0:event('raw_replayed',done=len(results),total=60)
 write(F/'REPLAY_COMPLETE.json',dict(rows=results,conditions=len(results),neural_calls=a.calls,model_load_seconds=loadtime,actual_camera=False,detector_calls=0,score_rerun='New full raw-path comparison only; prior completed score files untouched',all_errors_included=True));write(F/'CURRENT_EXECUTION.json',dict(status='RAW_REPLAY_COMPLETE_PENDING_AUDIT',active_process=None));event('raw_complete',conditions=len(results),errors=sum(r['status']!='OK' for r in results))
if __name__=='__main__':
 try:main()
 except BaseException:
  (F/f'REPLAY_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
