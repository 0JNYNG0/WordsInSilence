"""Local capture equivalent: decode one public frame at its due time, share it
with recording and observation workers, reuse observations after stop.
No browser frame upload, no camera, no SignRep or training.
"""
from pathlib import Path
import json,time,queue,threading,traceback,uuid,hashlib
from fractions import Fraction
from concurrent.futures import ThreadPoolExecutor
import av,cv2,numpy as np
from backend import U,Engine,StreamSession,MPWorker,reuse_observations

class Recorder:
 def __init__(self,path,w,h):
  self.path=path;self.w=w;self.h=h;self.q=queue.Queue(64);self.errors=[];self.drops=[];self.records=[]
  self.thread=threading.Thread(target=self.work,daemon=True);self.thread.start()
 def submit(self,fid,ts,rgb):
  try:self.q.put_nowait((fid,ts,rgb,time.perf_counter()));return True
  except queue.Full:self.drops.append(fid);return False
 def work(self):
  try:
   with av.open(str(self.path),'w') as out:
    st=out.add_stream('libx264',rate=30);st.width=self.w;st.height=self.h;st.pix_fmt='yuv420p';st.time_base=Fraction(1,1000000)
    st.codec_context.thread_count=2;st.options={'preset':'ultrafast','tune':'zerolatency','crf':'18'}
    while True:
     x=self.q.get()
     try:
      if x is None:break
      fid,ts,rgb,at=x;f=av.VideoFrame.from_ndarray(rgb,format='rgb24');f.pts=ts;f.time_base=Fraction(1,1000000)
      for p in st.encode(f):out.mux(p)
      self.records.append(dict(frame_id=fid,timestamp_us=ts,lag_ms=(time.perf_counter()-at)*1000))
     finally:self.q.task_done()
    for p in st.encode():out.mux(p)
  except BaseException:self.errors.append(traceback.format_exc())
 def finish(self):
  self.q.put(None);self.thread.join()
  return dict(count=len(self.records),errors=self.errors,dropped=self.drops,records=self.records,encoder='libx264 ultrafast CRF18 threads2')

def prefetched_frames(vst,video):
 q=queue.Queue(2)
 def decode():
  try:
   for f in video.decode(vst):q.put((f.pts,f.time_base,f.to_ndarray(format='bgr24')))
  except BaseException as exc:q.put(exc)
  finally:q.put(None)
 thread=threading.Thread(target=decode,daemon=True);thread.start()
 while True:
  item=q.get()
  if item is None:break
  if isinstance(item,BaseException):raise item
  yield item
 thread.join()

def run(engine,row,with_tcn,parallel_mp=False,prefetch=False):
 d=U/'native_runs'/uuid.uuid4().hex;d.mkdir(parents=True)
 records=[];tprep=time.perf_counter();mp=MPWorker(parallel=parallel_mp) if with_tcn else None
 with av.open(row['source']) as video:
  vst=video.streams.video[0];vst.thread_count=2;w,h=vst.width,vst.height
  stream=StreamSession(engine,w,h,capacity=64);rec=Recorder(d/'recorded.mp4',w,h)
  prep=time.perf_counter()-tprep;t0=None;pts0=None
  source=prefetched_frames(vst,video) if prefetch else ((f.pts,f.time_base,f.to_ndarray(format='bgr24')) for f in video.decode(vst))
  for fid,(frame_pts,frame_time_base,bgr) in enumerate(source):
   if pts0 is None:pts0=frame_pts
   ts=round(float((frame_pts-pts0)*frame_time_base)*1000000)
   # Never predecode the whole clip; this loop reads each next capture frame.
   rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
   if t0 is None:t0=time.perf_counter()
   due=t0+ts/1e6;delay=due-time.perf_counter()
   if delay>0:time.sleep(delay)
   at=time.perf_counter();ack=stream.submit(fid,ts,bgr);ma=mp.submit(fid,ts,rgb) if mp else True;ra=rec.submit(fid,ts,rgb)
   records.append(dict(frame_id=fid,timestamp_us=ts,capture_lateness_ms=(at-due)*1000,rtmw_accepted=ack['accepted'],mp_accepted=ma,recording_accepted=ra))
 last_due=t0+records[-1]['timestamp_us']/1e6;stop=time.perf_counter()
 with ThreadPoolExecutor(3) as pool:
  sf=pool.submit(stream.finish,d/'rtmw');rf=pool.submit(rec.finish);mf=pool.submit(mp.finish,d) if mp else None
  sm=sf.result();features=reuse_observations(d/'rtmw');np.savez_compressed(d/'pose_analysis_input.npz',**{k:v for k,v in features.items() if isinstance(v,np.ndarray)});rtmw_ready=time.perf_counter()
  mm=mf.result() if mf else None;analysis_ready=time.perf_counter();rm=rf.result();all_ready=time.perf_counter()
 r=dict(id=d.name,clip_id=row['clip_id'],label=row['label'],mode='RTMW_PLUS_FROZEN_TCN' if with_tcn else 'RTMW_ONLY',frames=len(records),
  rtmw_frames=sm['completed'],mp_frames=mm['count'] if mm else None,recorded_frames=rm['count'],records=records,
  input_complete=sm['status']=='COMPLETE' and all(x['rtmw_accepted'] and x['mp_accepted'] and x['recording_accepted'] for x in records) and not rm['errors'] and not rm['dropped'] and (mm is None or (mm['count']==len(records) and not mm['errors'] and not mm['dropped'])),
  prepare_seconds=prep,stop_to_rtmw_ready_ms=(rtmw_ready-stop)*1000,last_capture_to_rtmw_ready_ms=(rtmw_ready-last_due)*1000,
  last_capture_to_analysis_ready_ms=(analysis_ready-last_due)*1000,last_capture_to_recording_and_analysis_ms=(all_ready-last_due)*1000,
  second_pass_detector_calls=0,browser_image_upload=False,camera=False,training=False,feedback_model_ready=False,parallel_mp=parallel_mp,prefetch_frames=2 if prefetch else 0,
  capture_lateness_max_ms=max(x['capture_lateness_ms'] for x in records),mp_errors=mm['errors'] if mm else [],recording_errors=rm['errors'])
 (d/'REPLAY.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8');(d/'RECORDER.json').write_text(json.dumps(rm,indent=2),encoding='utf-8')
 print(json.dumps({k:r[k] for k in ['mode','label','frames','input_complete','last_capture_to_rtmw_ready_ms','last_capture_to_analysis_ready_ms','capture_lateness_max_ms']},ensure_ascii=False),flush=True)
 return r
def main():
 scope=U/'NATIVE_SCOPE.json'
 if (U/'NATIVE_COMPLETE.json').exists():raise RuntimeError('Completed native trial cannot be repeated')
 plan=json.loads((U/'FROZEN.json').read_text(encoding='utf-8'))
 if not scope.exists():scope.write_text(json.dumps(dict(goal='Same-process capture, recording and observations; no browser image transfer',modes=['RTMW_ONLY','RTMW_PLUS_FROZEN_TCN'],rows=[x['clip_id'] for x in plan['rows']],loss_allowed=0,timestamp_tolerance_us=2000,last_capture_budget_ms=1000,repetitions=1,real_webcam=False,training=False,limits='Public video timed local replay; not Unity rendering or hardware camera test. MP kept only to preserve frozen C/T3-F input.'),indent=2),encoding='utf-8')
 for p,h in plan['protected'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
 e=Engine();results=[]
 for with_tcn in [False,True]:
  for row in plan['rows']:results.append(run(e,row,with_tcn))
 (U/'NATIVE_COMPLETE.json').write_text(json.dumps(dict(status='EXECUTED_NEEDS_AUDIT',engine_devices=e.devices,compile_seconds=e.compile_seconds,warm_seconds=e.warm_seconds,runs=results),ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':
 try:main()
 except BaseException:
  (U/('NATIVE_FAILURE_'+str(time.time_ns())+'.txt')).write_text(traceback.format_exc(),encoding='utf-8');raise
