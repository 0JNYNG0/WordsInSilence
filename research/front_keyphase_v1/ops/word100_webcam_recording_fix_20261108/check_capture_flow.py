"""Real recorder/session code with synthetic frames and a fake capture device."""
from pathlib import Path
import sys,time,threading,json,numpy as np,av
U=Path(__file__).resolve().parent;D=U.parent/'word100_action_feedback_demo_20261108';sys.path.insert(0,str(D))
import capture
checks=[];opened=[]
class FakeCamera:
 def __init__(self,*args):self.i=0;self.alive=True;opened.append('synthetic_only')
 def isOpened(self):return True
 def set(self,*a):return True
 def get(self,k):return 30
 def read(self):
  # Normal camera jitter independent of any user's observed video/timing sequence.
  time.sleep([.020,.046,.028,.039][self.i%4]);self.i+=1
  return True,np.full((240,320,3),self.i%220,np.uint8)
 def release(self):self.alive=False
class FakeEngine:
 def __init__(self):self.calls=0;self.session_lock=threading.Lock()
 def infer(self,bgr):
  self.calls+=1;p=np.full((133,2),np.nan,np.float32);p[5:9]=[[100,60],[220,60],[110,160],[210,160]];p[9:11]=[[110,260],[210,260]]
  return dict(xy=p,score=np.ones(133,np.float32),bbox=np.array([0,0,320,240],np.float32),n_people=1,compute_ms=0.)
class FakeConsumer:
 def raw(self,a,word):
  return dict(word=word,frames=len(a['frame_id']),timestamps=a['timestamp_us'].tolist(),normal_or_wrong=None)
def wait(test,timeout=8):
 stop=time.perf_counter()+timeout
 while time.perf_counter()<stop:
  if test():return
  time.sleep(.02)
 raise AssertionError('timeout')
def check(n,v):checks.append(dict(name=n,pass_=bool(v)));assert v,n
def main():
 original=capture.cv2.VideoCapture;capture.cv2.VideoCapture=FakeCamera
 # Synthetic sessions go under this audit directory, never a real user directory.
 capture.tempfile.mkdtemp=lambda prefix:str(make_dir(prefix))
 s=capture.CaptureService();s.engine=FakeEngine();s.consumer=FakeConsumer();s.state='ready'
 try:
  s.open_camera();wait(lambda:s.state=='preview');s.start('WORD0739');wait(lambda:s.state=='recording');start=time.perf_counter();wait(lambda:s.elapsed>=.8);s.finish_requested=True;wait(lambda:s.state=='done');s.camera_thread.join(1)
  check('success returns and releases camera thread',not s.camera_thread.is_alive());check('same counts',s.result['recorded_frames']==s.result['observed_frames']==s.result['frames']);check('manual finish',s.result['stop_reason']=='manual')
  with av.open(str(s.directory/'recorded.mp4')) as v:pts=[round(float(f.pts*f.time_base)*1e6) for f in v.decode(video=0)]
  check('camera loop variable timestamps preserved',pts==s.result['timestamps']);check('preview can exceed4fps',s.jpeg_seq>40)
  # Exercise actual failure propagation and retry availability in the service.
  original_recorder=capture.Recorder
  class FailRecorder(original_recorder):
   def work(self):self.errors.append('injected_encoder_failure');self.worker_done.set()
  capture.Recorder=FailRecorder;s.open_camera();wait(lambda:s.state=='preview');s.start('WORD0739');wait(lambda:s.state=='error');s.camera_thread.join(1);check('failure cleanup cannot hang',not s.camera_thread.is_alive());check('specific encoder error',s.error=='영상 저장 중 오류가 났어요. 촬영을 다시 시작해 주세요.')
  capture.Recorder=original_recorder;s.open_camera();wait(lambda:s.state=='preview');s.stop_event.set();s.camera_thread.join(2);check('reopen after encoder failure',s.state=='ready' and not s.camera_thread.is_alive());check('no hardware device',len(opened)==3 and set(opened)=={'synthetic_only'})
  (U/'CAPTURE_FLOW_VERIFIED.json').write_text(json.dumps(dict(status='PASS',checks=checks,camera='synthetic factory only',real_user_data=False,model_accuracy_test=False),ensure_ascii=False,indent=2),encoding='utf-8');print('PASS',len(checks))
 finally:capture.cv2.VideoCapture=original;s.stop_event.set()
def make_dir(prefix):
 p=U/(prefix+str(time.time_ns()));p.mkdir();return p
if __name__=='__main__':main()
