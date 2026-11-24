"""Local camera is acquired only by an explicit, validated user command.

Each accepted camera frame is recorded and inferred once, with the same timestamp.
No file-based second detection; no RGB model; no training from these sessions.
"""
from task import *
import threading,tempfile,queue,traceback
from fractions import Fraction
import numpy as np,cv2,av
sys.path.insert(0,str(O/'word100_rtmw_stream_reuse_20261108'))
from stream_runtime import Engine,StreamSession
from auto_end import ReturnToRest
from consumer import FeedbackConsumer

class Recorder:
 def __init__(self,path,w,h):
  self.path=path;self.w=w;self.h=h;self.q=queue.Queue(64);self.records=[];self.errors=[];self.drops=[];self.closed=False;self.worker_done=threading.Event();self.thread=threading.Thread(target=self.work,daemon=True);self.thread.start()
 def submit(self,fid,ts,bgr):
  if self.closed or self.errors or self.worker_done.is_set():return False
  try:self.q.put_nowait((fid,ts,bgr.copy()));return True
  except queue.Full:self.drops.append(fid);return False
 def work(self):
  try:
   with av.open(str(self.path),'w') as out:
    st=out.add_stream('libx264',rate=30);st.width=self.w;st.height=self.h;st.pix_fmt='yuv420p'
    # Stream time base alone is insufficient: the encoder otherwise rounds
    # irregular camera timestamps to 1/30s and can emit duplicate DTS values.
    st.time_base=Fraction(1,1000000);st.codec_context.time_base=Fraction(1,1000000);st.codec_context.max_b_frames=0;st.codec_context.thread_count=2;st.options={'preset':'ultrafast','tune':'zerolatency','crf':'18'}
    while True:
     item=self.q.get()
     if item is None:break
     fid,ts,bgr=item;f=av.VideoFrame.from_ndarray(bgr,format='bgr24');f.pts=ts;f.time_base=Fraction(1,1000000)
     for pkt in st.encode(f):out.mux(pkt)
     self.records.append(dict(frame_id=fid,timestamp_us=ts))
    for pkt in st.encode():out.mux(pkt)
  except BaseException:self.errors.append(traceback.format_exc())
  finally:self.worker_done.set()
 def finish(self,timeout=5):
  deadline=time.perf_counter()+timeout
  if not self.closed:
   self.closed=True
   while self.thread.is_alive() and not self.errors:
    try:self.q.put(None,timeout=min(.05,max(.001,deadline-time.perf_counter())));break
    except queue.Full:
     if time.perf_counter()>=deadline:break
  self.thread.join(max(0,deadline-time.perf_counter()))
  if self.thread.is_alive() and 'encoder_finish_timeout' not in self.errors:self.errors.append('encoder_finish_timeout')
  return dict(count=len(self.records),dropped=self.drops,errors=self.errors,records=self.records,worker_alive=self.thread.is_alive())

class CaptureService:
 def __init__(self):
  self.state='preparing';self.error=None;self.engine=None;self.consumer=None;self.jpeg=None;self.jpeg_seq=0;self.preview_condition=threading.Condition();self.result=None;self.directory=None;self.guard=threading.Lock();self.stop_event=threading.Event();self.record_requested=False;self.finish_requested=False;self.camera_thread=None;self.countdown=0;self.elapsed=0;self.stop_reason=None;self.diagnostics={}
 def prepare(self):
  try:
   t=time.perf_counter();self.consumer=FeedbackConsumer();self.engine=Engine();self.prepare_seconds=time.perf_counter()-t;self.state='ready'
  except BaseException:self.error='분석기 준비에 실패했어요. 서버 로그를 확인해 주세요.';self.state='error';traceback.print_exc()
 def open_camera(self):
  with self.guard:
   if self.state not in ['ready','done','error'] or self.engine is None:raise ValueError('준비가 끝난 다음 카메라를 열어 주세요.')
   if self.camera_thread and self.camera_thread.is_alive():raise ValueError('카메라가 이미 열려 있어요.')
   self.stop_event.clear();self.record_requested=False;self.finish_requested=False;self.result=None;self.error=None;self.diagnostics={};self.jpeg=None;self.state='opening';self.camera_thread=threading.Thread(target=self.camera_loop,daemon=True);self.camera_thread.start()
 def start(self,word):
  with self.guard:
   if self.state!='preview':raise ValueError('카메라 화면을 먼저 확인해 주세요.')
   if word not in read(U/'FROZEN.json')['words']:raise ValueError('지원하지 않는 단어예요.')
   self.word=word;self.record_requested=True
 def camera_loop(self):
  cap=None;rec=None;stream=None;seen=0;submitted=0;rest=ReturnToRest();countdown_end=None;t0=None;last_ts=-1;w=h=0;failure=None;interval_min=None;interval_max=None
  try:
   # The only device-opening line; never reached at server startup or public replay.
   cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
   if not cap.isOpened():raise RuntimeError('웹캠을 열지 못했어요. 다른 프로그램이 카메라를 사용하는지 확인해 주세요.')
   cap.set(cv2.CAP_PROP_FRAME_WIDTH,640);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480);cap.set(cv2.CAP_PROP_FPS,30);cap.set(cv2.CAP_PROP_BUFFERSIZE,1);self.diagnostics.update(camera_reported_width=cap.get(cv2.CAP_PROP_FRAME_WIDTH),camera_reported_height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT),camera_reported_fps=cap.get(cv2.CAP_PROP_FPS));self.state='preview'
   while not self.stop_event.is_set():
    ok,bgr=cap.read();captured=time.perf_counter()
    if not ok:raise RuntimeError('웹캠 영상이 끊겼어요. 다시 열어 주세요.')
    h,w=bgr.shape[:2]
    if w%2 or h%2:bgr=bgr[:h-h%2,:w-w%2];h,w=bgr.shape[:2]
    ok,jpg=cv2.imencode('.jpg',bgr,[cv2.IMWRITE_JPEG_QUALITY,75])
    if ok:
     with self.preview_condition:self.jpeg=jpg.tobytes();self.jpeg_seq+=1;self.preview_condition.notify_all()
    if self.record_requested:
     self.record_requested=False;countdown_end=time.perf_counter()+3;self.state='countdown'
    if self.state=='countdown':
     self.countdown=max(0,int(np.ceil(countdown_end-time.perf_counter())))
     if time.perf_counter()>=countdown_end:
      self.directory=Path(tempfile.mkdtemp(prefix='wis_action_demo_'));rec=Recorder(self.directory/'recorded.mp4',w,h);stream=StreamSession(self.engine,w,h,capacity=64);self.state='recording';t0=captured;rest.begin(0)
    if self.state in ['preview','countdown']:
     observation=self.engine.infer(bgr);rest.calibrate(observation['xy'],w,h)
    if self.state!='recording':continue
    ts=max(last_ts+1,round((captured-t0)*1e6));fid=submitted
    if last_ts>=0:
     delta=ts-last_ts;interval_min=delta if interval_min is None else min(interval_min,delta);interval_max=delta if interval_max is None else max(interval_max,delta)
    last_ts=ts;self.elapsed=ts/1e6
    self.diagnostics.update(recording_received=fid+1,recording_written=len(rec.records),rtmw_completed=len(stream.results),recording_queue=rec.q.qsize(),rtmw_queue=stream.queue.qsize(),frame_interval_min_us=interval_min,frame_interval_max_us=interval_max,encoder_failed=bool(rec.errors))
    if not rec.submit(fid,ts,bgr):
     if rec.errors:raise RuntimeError('영상 저장 중 오류가 났어요. 촬영을 다시 시작해 주세요.')
     raise RuntimeError('영상 저장 대기열이 가득 찼어요. 촬영을 다시 시작해 주세요.')
    if not stream.submit(fid,ts,bgr)['accepted']:raise RuntimeError('관절 분석 대기열이 가득 찼어요. 촬영을 다시 시작해 주세요.')
    submitted+=1;auto=False
    pending=stream.results[seen:];seen+=len(pending)
    for observation in pending:
     if ts-observation['timestamp_us']<=250000:auto=rest.update(observation['xy'],w,h,observation['timestamp_us']) or auto
     else:rest.since=None
    if self.finish_requested or auto or ts>=15000000:
     self.stop_reason='manual' if self.finish_requested else 'arms_returned' if auto else 'time_limit';break
   cap.release();cap=None
   if rec is None:self.state='ready';return
   self.state='analysing';stopped=time.perf_counter();sm=stream.finish(self.directory/'rtmw');raw=stream.results;stream=None;rm=rec.finish();rec=None
   if self.stop_event.is_set():self.state='ready';return
   if sm['status']!='COMPLETE' or rm['errors'] or rm['dropped'] or rm['count']!=len(raw):raise RuntimeError('촬영한 영상과 관절 정보가 일치하지 않아요. 다시 촬영해 주세요.')
   arrays=dict(xy_px=np.stack([r['xy'] for r in raw]),score=np.stack([r['score'] for r in raw]),timestamp_us=np.array([r['timestamp_us'] for r in raw]),frame_id=np.arange(len(raw)),source_width=w,source_height=h)
   self.result=self.consumer.raw(arrays,self.word);self.result.update(video='/recorded.mp4',source='user_inference_only',stop_reason=self.stop_reason,auto_end_calibrated=rest.rest is not None,stop_to_result_seconds=time.perf_counter()-stopped,recorded_frames=len(raw),observed_frames=len(raw),input_camera_fps=len(raw)/max(self.elapsed,.001),training_use=False,second_pass_detector_calls=0);self.state='done'
  except BaseException as exc:
   failure=str(exc);self.error=failure;self.state='finishing';traceback.print_exc()
  finally:
   if cap is not None:cap.release()
   if stream is not None:
    try:sm=stream.finish(self.directory/'rtmw_incomplete');self.diagnostics.update(rtmw_final_count=sm['completed'],rtmw_errors=sm['errors'])
    except Exception as exc:self.diagnostics['stream_cleanup_error']=str(exc)
   if rec is not None:
    rm=rec.finish();self.diagnostics.update(recording_final_count=rm['count'],encoder_failed=bool(rm['errors']),encoder_worker_alive=rm['worker_alive'])
    if rm['errors']:print('RECORDER_ERROR:',rm['errors'],flush=True)
   self.finish_requested=False
   if failure:self.state='error'
   with self.preview_condition:self.preview_condition.notify_all()
 def status(self):
  return dict(state=self.state,error=self.error,countdown=self.countdown,elapsed=round(self.elapsed,1),has_result=self.result is not None,prepare_seconds=getattr(self,'prepare_seconds',None),camera_thread_active=bool(self.camera_thread and self.camera_thread.is_alive()),diagnostics=self.diagnostics)
