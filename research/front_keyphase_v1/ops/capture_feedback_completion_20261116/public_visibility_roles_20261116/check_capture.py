"""Same public-video webcam fixture before/after. Never opens a physical camera."""
from check_web import *
import cv2,av,capture
from visibility_web_adapter import CaptureService
from stream_runtime import Engine

def main():
 out=D/'web';assert not(out/'CAPTURE_CHECK.json').exists();cv2.setNumThreads(2)
 service=CaptureService();engine=Engine();rows=[];frames=[]
 class PublicCamera:
  def __init__(self,*args):self.i=0
  def isOpened(self):return True
  def set(self,*args):return True
  def get(self,k):return {cv2.CAP_PROP_FRAME_WIDTH:640,cv2.CAP_PROP_FRAME_HEIGHT:480,cv2.CAP_PROP_FPS:30}.get(k,0)
  def read(self):
   time.sleep(1/30)
   if service.state=='recording':
    im=frames[min(self.i,len(frames)-1)];self.i+=1
    if self.i>len(frames)+90:service.finish_requested=True
    return True,im.copy()
   return True,frames[0].copy()
  def release(self):pass
 original=capture.cv2.VideoCapture;capture.cv2.VideoCapture=PublicCamera
 try:
  for phase,cls in [('before',PriorConsumer),('after',FeedbackConsumer)]:
   net=cls();service.consumer=net;service.engine=engine;service.state='ready'
   try:
    for word,root in [('WORD0099',A),('WORD1290',V)]:
     source=read(root/'web/SOURCES.json')['normal']['video']
     with av.open(source)as f:
      frames=[]
      for fr in f.decode(video=0):
       im=fr.to_ndarray(format='bgr24');hh,ww=im.shape[:2];im=im[int(hh*.08):int(hh*.75),int(ww*.25):int(ww*.75)];frames.append(cv2.resize(im,(640,480)))
     service.open_camera();deadline=time.monotonic()+90
     while service.state!='preview'and time.monotonic()<deadline:time.sleep(.02)
     assert service.state=='preview',service.status();service.start(word);states=[];times={}
     while service.state not in['done','error']and time.monotonic()<deadline:
      if not states or states[-1]!=service.state:states.append(service.state);times[service.state]=time.perf_counter()
      time.sleep(.02)
     assert service.state=='done',service.status();service.camera_thread.join(5);r=service.result;r['source']='public_capture_fixture'
     row=dict(phase=phase,word=word,source=source,auto_end=r['stop_reason']=='arms_returned',stop_reason=r['stop_reason'],countdown_seconds=times.get('recording',0)-times.get('countdown',0),frames=r['recorded_frames'],observed=r['observed_frames'],redetection=r['second_pass_detector_calls'],stop_to_result_seconds=r['stop_to_result_seconds'],compute_seconds=r['compute_seconds'],top1=r['tcn_top3'][0],completed=r['overall_pass'],criteria={c['id']:c['status']for c in r['lesson_acceptance']['criteria']},training_use=r['training_use'],states=states,artifacts=str(service.directory))
     rows.append(row);write(out/f'capture_{phase}_{word}.json',r);write(out/'CAPTURE_CHECK.partial.json',dict(rows=rows));print(json.dumps(row,ensure_ascii=False),flush=True)
   finally:close(net)
  checks=[]
  for word in['WORD0099','WORD1290']:
   a,b=[r for r in rows if r['word']==word];checks.append(dict(word=word,passed=a['auto_end']and b['auto_end']and b['completed']and b['observed']==b['frames']and b['redetection']==0 and b['stop_to_result_seconds']<max(1.5,a['stop_to_result_seconds']+.5)))
  write(out/'CAPTURE_CHECK.json',dict(rows=rows,checks=checks,passed=all(c['passed']for c in checks),scope='Public camera fixture, actual RTMW/encoder/TCN/feedback. Not physical webcam or browser-paint latency.',physical_camera=False))
 finally:service.stop_event.set();capture.cv2.VideoCapture=original
if __name__=='__main__':main()
