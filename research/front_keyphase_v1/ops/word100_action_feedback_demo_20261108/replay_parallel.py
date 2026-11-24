"""Changed concurrency condition; first sequential result remains immutable."""
from task import *
import numpy as np,av,queue,threading,traceback
from capture import Recorder,Engine,StreamSession
from consumer import FeedbackConsumer
def frames(video,st):
 q=queue.Queue(2)
 def work():
  try:
   for f in video.decode(st):q.put((round(float(f.pts*f.time_base)*1e6),f.to_ndarray(format='bgr24')))
  except BaseException as e:q.put(e)
  finally:q.put(None)
 th=threading.Thread(target=work,daemon=True);th.start()
 while True:
  x=q.get()
  if x is None:break
  if isinstance(x,BaseException):raise x
  yield x
 th.join()
def main():
 assert not (U/'PARALLEL_COMPLETE.json').exists();prep=read(U/'PREPARED.json');rows=[r for r in prep['rows'] if r['person']==11];write(U/'PARALLEL_FROZEN.json',dict(reason='Sequential decoding plus detection delayed the planned public timeline by1.71-2.10sec. Parallel capture/record/inference retains every timestamp and frame.',once=True,rows=[r['clip_id'] for r in rows],camera=False,input_loss_allowed=0,prior_result_sha=sha(U/'INTEGRATION_COMPLETE.json')));t=time.perf_counter();engine=Engine();consumer=FeedbackConsumer();prepare=time.perf_counter()-t;results=[]
 for row in rows:
  word=prep['words'][row['word']];d=U/'parallel'/word;d.mkdir(parents=True,exist_ok=False);records=[];t0=None
  with av.open(row['source']) as video:
   st=video.streams.video[0];st.thread_count=2;w,h=st.width,st.height;stream=StreamSession(engine,w,h,capacity=64);rec=Recorder(d/'video.mp4',w,h)
   for fid,(ts,bgr) in enumerate(frames(video,st)):
    if t0 is None:t0=time.perf_counter()-ts/1e6
    due=t0+ts/1e6;delay=due-time.perf_counter()
    if delay>0:time.sleep(delay)
    at=time.perf_counter();assert rec.submit(fid,ts,bgr);assert stream.submit(fid,ts,bgr)['accepted'];records.append(dict(frame_id=fid,timestamp_us=ts,hash=hashlib.sha256(bgr.tobytes()).hexdigest(),capture_lateness_ms=(at-due)*1000))
  stopped=time.perf_counter();sm=stream.finish(d/'rtmw');rm=rec.finish();z=np.load(d/'rtmw/observations.npz');result=consumer.raw(z,word);last=t0+records[-1]['timestamp_us']/1e6;rr=dict(word=word,frames=len(records),rtmw_frames=sm['completed'],recorded_frames=rm['count'],input_complete=sm['status']=='COMPLETE' and rm['count']==len(records) and not rm['errors'] and not rm['dropped'],stop_to_result_seconds=time.perf_counter()-stopped,last_planned_frame_to_result_seconds=time.perf_counter()-last,neural_compute_seconds=result['compute_seconds'],max_capture_lateness_ms=max(r['capture_lateness_ms'] for r in records),second_pass_detector_calls=0,records=records,recorder=rm);write(d/'REPLAY.json',rr);write(d/'result.json',result);results.append(rr);event('parallel_replay',**{k:v for k,v in rr.items() if k not in ['records','recorder']})
 write(U/'PARALLEL_COMPLETE.json',dict(prepare_seconds=prepare,runs=results,camera=False,unity=False,browser_render_included=False));event('parallel_done')
if __name__=='__main__':
 try:main()
 except BaseException:
  (U/f'PARALLEL_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
