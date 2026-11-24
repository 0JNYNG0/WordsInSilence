"""Public camera fixture through actual capture, RTMW, encoder and analysis.
No physical camera or user footage. Results do not claim webcam verification.
"""
from pathlib import Path
import sys
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H.parent/'apple_v1'))
from check_consumer import *
from hello_web_adapter import FeedbackConsumer
P=H
import cv2,av,capture
from hello_web_adapter import CaptureService
from stream_runtime import Engine
def main():
 out=P/'web/capture';out.mkdir(exist_ok=True);assert not(out/'CAPTURE_CHECK.json').exists()
 service=CaptureService();service.consumer=FeedbackConsumer();service.engine=Engine();service.state='ready';rows=[]
 source=read(P/'web/SOURCES.json')['normal']['video']
 with av.open(source)as f:native=[fr.to_ndarray(format='bgr24')for fr in f.decode(video=0)]
 frames=[]
 class PublicCamera:
  def __init__(self,*args):self.i=0
  def isOpened(self):return True
  def set(self,*args):return True
  def get(self,k):return {cv2.CAP_PROP_FRAME_WIDTH:frames[0].shape[1],cv2.CAP_PROP_FRAME_HEIGHT:frames[0].shape[0],cv2.CAP_PROP_FPS:30}.get(k,0)
  def read(self):
   time.sleep(1/30)
   if service.state=='recording':
    im=frames[min(self.i,len(frames)-1)];self.i+=1
    if self.i>len(frames)+90:service.finish_requested=True
    return True,im.copy()
   return True,frames[0].copy()
  def release(self):pass
 capture.cv2.VideoCapture=PublicCamera
 try:
  for kind in['full_frame_640','upper_body_640']:
   if kind=='full_frame_640':frames=[cv2.resize(im,(640,360))for im in native]
   else:
    h,w=native[0].shape[:2];box=[int(w*.25),int(h*.08),int(w*.75),int(h*.75)]
    frames=[cv2.resize(im[box[1]:box[3],box[0]:box[2]],(640,480))for im in native]
   service.open_camera();deadline=time.monotonic()+100
   while service.state!='preview'and time.monotonic()<deadline:time.sleep(.02)
   assert service.state=='preview',service.status();service.start('SEN0354');states=[];times={}
   while service.state not in['done','error']and time.monotonic()<deadline:
    if not states or states[-1]!=service.state:states.append(service.state);times[service.state]=time.perf_counter()
    time.sleep(.02)
   assert service.state=='done',service.status();service.camera_thread.join(5);r=service.result;r['source']='public_capture_loop'
   row=dict(kind=kind,public_source=source,physical_camera=False,crop=box if kind=='upper_body_640'else None,
    auto_end=r['stop_reason']=='arms_returned',stop_reason=r['stop_reason'],countdown_seconds=times.get('recording',0)-times.get('countdown',0),
    frames=r['recorded_frames'],observed=r['observed_frames'],redetection=r['second_pass_detector_calls'],
    stop_to_result_seconds=r['stop_to_result_seconds'],compute_seconds=r['compute_seconds'],top1=r['tcn_top3'],
    completed=r['exercise_complete'],criteria={c['id']:c['status']for c in r['lesson_acceptance']['criteria']},
    training_use=r['training_use'],states=states,artifacts=str(service.directory))
   import shutil
   for observed_path in service.directory.rglob('observations.npz'):shutil.copyfile(observed_path,out/f'{kind}_raw.npz')
   rows.append(row);(out/f'capture_{kind}.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(row,ensure_ascii=False),flush=True)
  (out/'CAPTURE_CHECK.json').write_text(json.dumps(dict(rows=rows,scope='Public frame fixture, actual RTMW+encode+analysis; physical capture and browser delay not measured'),ensure_ascii=False,indent=2),encoding='utf-8')
 finally:service.stop_event.set();service.consumer.hello.close();service.consumer.apple.close();service.consumer.basic.close();service.consumer.latest.close()
if __name__=='__main__':main()


