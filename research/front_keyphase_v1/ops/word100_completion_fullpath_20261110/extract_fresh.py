from common import *
import av,cv2,numpy as np,traceback,os
cv2.setNumThreads(2)
def main():
 cfg=read(F/'FRESH_FROZEN.json');assert not (F/'FRESH_EXTRACTION_COMPLETE.json').exists()
 for p,h in cfg['pins'].items():assert sha(p)==h,p
 assert not list((F/'fresh_raw').glob('*.npz')),'Existing extraction must be audited, not recomputed'
 S=load('_fresh_rtmw_stream',O/'word100_rtmw_stream_reuse_20261108/stream_runtime.py');engine=S.Engine();out=[]
 write(F/'FRESH_ENGINE.json',dict(devices=engine.devices,compile_seconds=engine.compile_seconds,warm_seconds=engine.warm_seconds));write(F/'CURRENT_EXECUTION.json',dict(status='PUBLIC_FRESH_RTM_EXTRACTION',pid=os.getpid()));event('fresh_engine_ready',devices=engine.devices)
 for row in cfg['rows']:
  assert sha(row['source'])==row['source_sha256'];start=time.perf_counter();xs=[];scores=[];times=[];cost=[];hashes=[]
  with av.open(row['source']) as con:
   st=con.streams.video[0];st.thread_type='AUTO';st.codec_context.thread_count=2;base=None
   for i,frame in enumerate(con.decode(st)):
    if base is None:base=frame.pts
    tus=round(float((frame.pts-base)*st.time_base)*1e6);im=frame.to_ndarray(format='bgr24');q=engine.infer(im);xs.append(q['xy']);scores.append(q['score']);times.append(tus);cost.append(q['compute_ms']);hashes.append(hashlib.sha256(im.tobytes()).hexdigest())
   w,h=im.shape[1],im.shape[0]
  n=len(xs);assert n==row['frames'];assert np.all(np.diff(times)>0)
  p=F/'fresh_raw'/(row['clip_id']+'.npz');p.parent.mkdir(exist_ok=True);np.savez_compressed(p,xy_px=np.array(xs),score=np.array(scores),timestamp_us=np.array(times,np.int64),frame_id=np.arange(n,dtype=np.int64),source_width=np.int64(w),source_height=np.int64(h),frame_sha256=np.array(hashes))
  r=dict(clip_id=row['clip_id'],source_sha256=row['source_sha256'],raw_path=str(p),raw_sha256=sha(p),frames=n,seconds=time.perf_counter()-start,infer_ms_median=float(np.median(cost)),infer_ms_p95=float(np.percentile(cost,95)),infer_ms_total=sum(cost),status='OK');write(F/'fresh_raw'/(row['clip_id']+'.json'),r);out.append(r);event('fresh_extracted',clip=row['clip_id'],frames=n,seconds=r['seconds'])
 write(F/'FRESH_EXTRACTION_COMPLETE.json',dict(rows=out,frames=sum(r['frames'] for r in out),detector_calls=engine.calls,training=False,camera=False,source_count=4));event('fresh_extraction_complete',frames=engine.calls)
if __name__=='__main__':
 try:main()
 except BaseException:
  (F/f'FRESH_EXTRACTION_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
