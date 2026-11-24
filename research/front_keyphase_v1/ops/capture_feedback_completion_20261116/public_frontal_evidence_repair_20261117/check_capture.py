from check_runtime import *
import cv2,av,threading,copy
import capture
from stream_runtime import Engine
from ready_capture import ReadyRest
from auto_end import ReturnToRest

def wait_for(test,seconds=15):
 end=time.monotonic()+seconds
 while not test() and time.monotonic()<end:time.sleep(.02)
 assert test(),'Timed out waiting for fixture state'

def main():
 cv2.setNumThreads(2);service=CaptureService();frames=[np.zeros((480,640,3),np.uint8)];fallbacks=[]
 class PublicCamera:
  def __init__(self,*args):self.i=0
  def isOpened(self):return True
  def set(self,*args):return True
  def get(self,k):return {cv2.CAP_PROP_FRAME_WIDTH:640,cv2.CAP_PROP_FRAME_HEIGHT:480,cv2.CAP_PROP_FPS:30}.get(k,0)
  def read(self):
   time.sleep(1/30)
   if service.state=='recording':
    im=frames[min(self.i,len(frames)-1)];self.i+=1
    if self.i>len(frames)+90:fallbacks.append(True);service.finish_requested=True
    return True,im.copy()
   return True,frames[0].copy()
  def release(self):pass
 class MissingEngine:
  def infer(self,_):return dict(xy=np.full((133,2),np.nan),score=np.zeros(133))
 original=capture.cv2.VideoCapture;capture.cv2.VideoCapture=PublicCamera;checks=[];rows=[]
 try:
  # Real capture control flow, fake unavailable public observation: no webcam.
  service.engine=MissingEngine();service.state='ready';service.open_camera();wait_for(lambda:service.state=='preview')
  rejected=False
  try:service.start('WORD0099')
  except ValueError:rejected=True
  assert rejected
  service.word='WORD0099';service.record_requested=True;wait_for(lambda:service.state=='countdown');wait_for(lambda:service.state=='preview',8)
  assert service.directory is None and service.result is None
  checks.append(dict(kind='missing_baseline_api_and_countdown_block',passed=True,recording_started=False,synthetic_observation=True))
  # Readiness can recover in the SAME preview when public resting observations
  # become available; no restart or guessed wrist/hand coordinate is required.
  rr=next(r for r in read(N/'INPUTS.json')['rows']if r['word']=='WORD0099');require_front(rr);zz=dict(np.load(rr['raw']))
  class PublicRestEngine:
   def infer(self,_):return dict(xy=zz['xy_px'][0]*np.array([640/int(zz['source_width']),480/int(zz['source_height'])]),score=zz['score'][0])
  service.engine=PublicRestEngine();wait_for(lambda:service.status()['auto_end_ready'])
  checks.append(dict(kind='same_preview_recovers_readiness',passed=True,source=rr['id'],synthetic_observation=True))
  service.stop_event.set();service.camera_thread.join(5);assert not service.camera_thread.is_alive()
  if '--readiness-only' in sys.argv:
   write(D/'web/READINESS_RECOVERY.json',dict(rows=checks,passed=True,physical_camera=False,private_data=False));print('READINESS_RECOVERY',json.dumps(checks,ensure_ascii=False),flush=True);return
  # Real public images -> RTMW -> recording reuse -> TCN/feedback.
  service=CaptureService();net=FeedbackConsumer();net.apple_admitted=net.basic_admitted=True
  service.consumer=net;service.engine=Engine();service.state='ready'
  try:
   for word,root in [('WORD0099',G/'apple_v1'),('WORD1290',G/'completion_v1')]:
    source=read(root/'web/SOURCES.json')['normal']['video']
    with av.open(source)as f:
     frames=[]
     for fr in f.decode(video=0):
      im=fr.to_ndarray(format='bgr24');h,w=im.shape[:2];im=im[int(h*.08):int(h*.75),int(w*.25):int(w*.75)];frames.append(cv2.resize(im,(640,480)))
    service.open_camera();wait_for(lambda:service.state=='preview' and service.status()['auto_end_ready'],25);service.start(word);wait_for(lambda:service.state=='countdown');began=time.perf_counter();wait_for(lambda:service.state=='recording',8);countdown=time.perf_counter()-began
    wait_for(lambda:service.state in ['done','error'],40);assert service.state=='done',service.status();service.camera_thread.join(5);q=service.result
    passed=q['stop_reason']=='arms_returned' and q['auto_end_calibrated'] and q['overall_pass'] and q['recorded_frames']==q['observed_frames'] and q['second_pass_detector_calls']==0 and q['stop_to_result_seconds']<1.5 and not fallbacks
    q['source']='public_capture_fixture';q['tuning_use']=False
    row=dict(word=word,passed=bool(passed),auto_end=q['stop_reason'],calibrated=q['auto_end_calibrated'],countdown_seconds=countdown,frames=q['recorded_frames'],observed=q['observed_frames'],second_pass=q['second_pass_detector_calls'],stop_to_result_seconds=q['stop_to_result_seconds'],overall_pass=q['overall_pass'],source=source)
    rows.append(row);write(D/'web'/f'capture_{word}.json',q);print('CAPTURE',json.dumps(row,ensure_ascii=False),flush=True)
  finally:close(net)
 finally:
  service.stop_event.set()
  if service.camera_thread:service.camera_thread.join(5)
  capture.cv2.VideoCapture=original
 # Recorded public pose trajectories: one/two-handed movements and temporary
 # hand disappearance. No movement = no completion signal.
 stop_rows=[]
 for word in ['WORD0099','WORD1290','WORD1133']:
  records=read(N/'INPUTS.json')['rows']
  if word=='WORD1133':
   p=O/'word100_rtmw_joint_learning_20261108/records/NIA_SL_WORD1133_REAL01_F.json';rr=read(p);r=dict(id='NIA_SL_WORD1133_REAL01_F',raw=rr['raw_path'])
  else:r=next(r for r in records if r['word']==word)
  require_front(r);z=dict(np.load(r['raw']));xy=z['xy_px'];w=int(z['source_width']);h=int(z['source_height'])
  for kind in ['normal','hands_missing_mid','rest_only']:
   old=ReturnToRest();new=ReadyRest()
   # Existing public first resting frames, repeated to supply baseline samples.
   for _ in range(2):
    for p in xy[:10]:old.calibrate(p,w,h);new.calibrate(p,w,h)
   assert new.ready,(word,'no public baseline');old.begin(0);new.begin(0)
   seq=list(xy)+[xy[-1]]*35
   if kind=='rest_only':seq=[xy[0]]*len(seq)
   active=[]
   for i,p in enumerate(seq):
    g=new.geometry(p,w,h)
    if g is not None and g[1].any():active.append(i)
   mid=active[len(active)//2] if active else len(seq)//2
   if kind=='hands_missing_mid':
    for i in range(mid,min(mid+5,len(seq))):p=seq[i].copy();p[91:]=np.nan;seq[i]=p
   stops=[]
   for detector in [old,new]:
    stop=next((i for i,p in enumerate(seq)if detector.update(p,w,h,i*33333)),None);stops.append(stop)
   passed=stops[0]==stops[1] and (stops[1] is None if kind=='rest_only' else stops[1] is not None and stops[1]>mid+5)
   stop_rows.append(dict(id=r['id'],kind=kind,before_stop=stops[0],after_stop=stops[1],passed=passed,synthetic_edit=kind!='normal'))
 write(D/'web/CAPTURE_CHECK.json',dict(rows=rows,readiness=checks,stop_controls=stop_rows,passed=all(r['passed']for r in rows+checks+stop_rows),physical_camera=False,private_data=False))
 print('STOP_CONTROLS',json.dumps(stop_rows,ensure_ascii=False),flush=True)
 assert all(r['passed']for r in rows+checks+stop_rows)
if __name__=='__main__':main()
