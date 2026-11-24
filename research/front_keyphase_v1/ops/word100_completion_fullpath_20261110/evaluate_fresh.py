from common import *
import numpy as np,traceback,os
def main():
 cfg=read(F/'FRESH_FROZEN.json');assert not (F/'FRESH_COMPLETE.json').exists()
 for p,h in cfg['pins'].items():assert sha(p)==h,p
 rows=[];raws={r['clip_id']:r for r in read(F/'FRESH_EXTRACTION_COMPLETE.json')['rows']}
 for r in cfg['rows']:
  rr=raws[r['clip_id']];assert sha(rr['raw_path'])==rr['raw_sha256'];z=np.load(rr['raw_path']);n=len(z['frame_id']);step=int(np.median(np.diff(z['timestamp_us'])))
  for kind in cfg['conditions']:
   if 'final_event' in kind and 'cut' not in r:continue
   ids=np.arange(n)
   if kind.startswith('speed'):ids=np.rint(np.linspace(0,n-1,round(n/(.75 if kind=='speed075' else 1.25)))).astype(int)
   if kind=='omit_final_event':ids=np.r_[np.arange(r['cut']+1),np.arange(n-6,n)]
   xy=z['xy_px'][ids].copy();score=z['score'][ids].copy()
   if kind in ['all_hand_missing','mask_final_event']:
    lo=0 if kind=='all_hand_missing' else r['cut']+1;xy[lo:,91:133]=np.nan;score[lo:,91:133]=0
   raw=dict(xy_px=xy,score=score,frame_id=np.arange(len(ids),dtype=np.int64),timestamp_us=np.arange(len(ids),dtype=np.int64)*step,source_width=z['source_width'],source_height=z['source_height'])
   p=F/'fresh_inputs'/f"{r['clip_id']}_{kind}.npz";p.parent.mkdir(exist_ok=True);assert not p.exists();np.savez_compressed(p,**raw,source_frame_id=ids)
   rows.append(dict(clip_id=r['clip_id'],word_id=r['word'],person=10,kind=kind,path=str(p),sha256=sha(p),frames=len(ids),source_indices=ids.tolist(),raw_source=rr['raw_path'],raw_source_sha256=rr['raw_sha256']))
 assert len(rows)==cfg['expected_conditions'];write(F/'FRESH_INPUTS.json',dict(rows=rows,frozen_sha256=sha(F/'FRESH_FROZEN.json')))
 rt=load('_fresh_runtime',F/'runtime.py');start=time.perf_counter();c,a=rt.instantiate();load_seconds=time.perf_counter()-start;results=[];write(F/'CURRENT_EXECUTION.json',dict(status='FRESH_FULLPATH_EVALUATION',pid=os.getpid()))
 for r in rows:
  key=r['clip_id']+'_'+r['kind'];z=np.load(r['path']);raw={k:z[k] for k in ['xy_px','score','frame_id','timestamp_us','source_width','source_height']};p=F/'fresh_results'/key
  try:
   b,d,pre,x,sec=rt.one(c,a,raw,r['word_id']);assert b['tcn_top3']==d['tcn_top3'] and b['path']==d['path'];assert len(d['frame_id'])==r['frames'];assert d['reference_video_reads']==0 and d['rgb_model_calls']==0
   write(p/'before.json',b);write(p/'after.json',d);write(p/'before_final_quarantine.json',pre);np.savez_compressed(p/'converted.npz',x=x)
   rr=dict(r,status='OK',before=rt.compact(b),after=rt.compact(d),before_final_quarantine=rt.compact(pre),seconds=sec,added_seconds=d['timing']['completion_identity_seconds'],result=str(p/'after.json'),result_sha256=sha(p/'after.json'),quarantined_hand_frames=int(np.asarray(d['observation_guard']['hand_quarantined']).any(1).sum()))
  except Exception:
   err=traceback.format_exc();write(p/'ERROR.json',dict(error=err));rr=dict(r,status='ERROR',error=err)
  results.append(rr);write(p/'RECORD.json',rr);event('fresh_evaluated',done=len(results),kind=r['kind'],status=rr['status'])
 write(F/'FRESH_COMPLETE.json',dict(rows=results,conditions=len(results),neural_calls=a.calls,model_load_seconds=load_seconds,training=False,selection=False,errors_in_denominator=True));write(F/'CURRENT_EXECUTION.json',dict(status='COMPARISONS_COMPLETE_PENDING_VERIFICATION',active_process=None))
if __name__=='__main__':
 try:main()
 except BaseException:
  (F/f'FRESH_EVALUATION_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
