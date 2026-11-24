"""Actual capture/RTMW/encode/inference pipeline, public video as camera source.
No physical device, private recordings or learning are accessed.
"""
from bootstrap import *
import json,time,numpy as np,av,cv2
import capture
from g6_runtime_adapter import CaptureService,FeedbackConsumer
from stream_runtime import Engine
service=CaptureService();service.consumer=FeedbackConsumer();service.engine=Engine();service.state='ready'
reports=[];(U/'public_capture').mkdir(exist_ok=True)
class PublicCamera:
 def __init__(self,*args):self.index=0
 def isOpened(self):return True
 def set(self,*args):return True
 def get(self,k):return {cv2.CAP_PROP_FRAME_WIDTH:frames[0].shape[1],cv2.CAP_PROP_FRAME_HEIGHT:frames[0].shape[0],cv2.CAP_PROP_FPS:30}.get(k,0)
 def read(self):
  time.sleep(1/30)
  if service.state=='recording':
   f=frames[min(self.index,len(frames)-1)];self.index+=1
   if self.index>len(frames)+90:service.finish_requested=True
   return True,f.copy()
  return True,frames[0].copy()
 def release(self):pass
capture.cv2.VideoCapture=PublicCamera
try:
 for word in ['WORD0739','WORD1133']:
  source=O/'word100_t3f_heldout_20261019/data/clips'/f'NIA_SL_{word}_REAL17_F/source.mp4'
  with av.open(str(source)) as c:frames=[cv2.resize(f.to_ndarray(format='bgr24'),(640,360)) for f in c.decode(video=0)]
  service.open_camera();deadline=time.monotonic()+100
  while service.state!='preview' and time.monotonic()<deadline:time.sleep(.05)
  assert service.state=='preview',service.status()
  service.start(word);states=[];state_times={}
  while service.state not in ['done','error'] and time.monotonic()<deadline:
   if not states or states[-1]!=service.state:states.append(service.state);state_times[service.state]=time.perf_counter()
   time.sleep(.02)
  assert service.state=='done',service.status()
  service.camera_thread.join(5);r=service.result;r['source']='public_capture_loop';r['video']=str(service.directory/'recorded.mp4')
  checks=dict(countdown='countdown' in states,recording='recording' in states,auto_return=r['stop_reason']=='arms_returned',all_frames=r['recorded_frames']==r['observed_frames'],one_detection=r['second_pass_detector_calls']==0,latest=r['feedback_version']=='G6-frozen-G4',no_training=r['training_use'] is False,top3=len(r['tcn_top3'])==3,no_false_correction=not r['coaching']['cues'])
  out=dict(word=word,checks=checks,passed=all(checks.values()),states=states,stop_reason=r['stop_reason'],stop_to_result_seconds=r['stop_to_result_seconds'],timing=r['timing'],frames=r['recorded_frames'],camera_fps=r['input_camera_fps'],countdown_seconds=state_times.get('recording',0)-state_times.get('countdown',0),public_source=str(source),scope='Public video at640x360/30Hz through real RTMW/capture/encode; not physical webcam or browser delivery',result=r)
  (U/'public_capture'/(word+'.json')).write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');reports.append({k:v for k,v in out.items() if k!='result'});print(json.dumps(reports[-1],ensure_ascii=False),flush=True)
finally:service.consumer.latest.close()
(U/'PUBLIC_CAPTURE.json').write_text(json.dumps(dict(passed=all(r['passed'] for r in reports),rows=reports),ensure_ascii=False,indent=2),encoding='utf-8')
assert all(r['passed'] for r in reports),reports
