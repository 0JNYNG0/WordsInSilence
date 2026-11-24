"""One real capture-loop integration using public video instead of a device.
Never calls the original cv2.VideoCapture; no learner images are opened.
"""
from pathlib import Path
import sys,time,json,traceback
import numpy as np,av
U=Path(__file__).resolve().parent;OLD=U.parent/'word100_action_feedback_demo_20261108'
sys.path.insert(0,str(OLD));sys.dont_write_bytecode=True
import capture
from consumer import FeedbackConsumer
from stream_runtime import Engine
word='WORD1567'
with av.open(str(OLD/'parallel'/word/'video.mp4')) as c:
 frames=[f.to_ndarray(format='bgr24') for f in c.decode(video=0)]
service=capture.CaptureService();service.consumer=FeedbackConsumer();service.engine=Engine();service.state='ready'
class PublicCamera:
 def __init__(self,*args):self.index=0;self.closed=False
 def isOpened(self):return True
 def set(self,*args):return True
 def get(self,k):return {capture.cv2.CAP_PROP_FRAME_WIDTH:frames[0].shape[1],capture.cv2.CAP_PROP_FRAME_HEIGHT:frames[0].shape[0],capture.cv2.CAP_PROP_FPS:30}.get(k,0)
 def read(self):
  time.sleep(1/30)
  if service.state=='recording':
   f=frames[min(self.index,len(frames)-1)];self.index+=1
   if self.index>len(frames)+35:service.finish_requested=True
   return True,f.copy()
  return True,frames[0].copy()
 def release(self):self.closed=True
capture.cv2.VideoCapture=PublicCamera
service.open_camera();deadline=time.monotonic()+45
while service.state!='preview' and time.monotonic()<deadline:time.sleep(.1)
assert service.state=='preview',service.status()
service.start(word)
states=[]
while service.state not in ['done','error'] and time.monotonic()<deadline:
 if not states or states[-1]!=service.state:states.append(service.state)
 time.sleep(.1)
assert service.state=='done',service.status()
r=service.result
checks=dict(countdown_seen='countdown' in states,recording_seen='recording' in states,same_frame_counts=r['recorded_frames']==r['observed_frames'],second_detection_not_used=r['second_pass_detector_calls']==0,training_disabled=r['training_use'] is False,tcn_returned=len(r['tcn_top3'])==3,feedback_returned=bool(r['cards']),no_device_camera_opened=True)
assert all(checks.values()),checks
out=dict(passed=True,checks=checks,states=states,stop_reason=r['stop_reason'],frames=r['recorded_frames'],source='public video through camera-loop test double; not a physical webcam test',temporary_directory=str(service.directory),top1=r['tcn_top3'][0]['label'])
(U/'PUBLIC_CAPTURE_VERIFIED.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out,ensure_ascii=False))
