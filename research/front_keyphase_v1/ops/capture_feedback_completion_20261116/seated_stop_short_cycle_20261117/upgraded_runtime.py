from pathlib import Path
import sys,json,time,hashlib
D=Path(__file__).resolve().parent;G=D.parent;O=G.parent
R=G/'selective_rtmw_anchor_repair_20261117';E=G/'public_frontal_evidence_repair_20261117'
sys.path.insert(0,str(R))
from retry_runtime import CaptureService as PreviousCapture,ConsumerProxy,VERSION
from frontal_repair_runtime import FeedbackConsumer as PreviousConsumer,CaptureService as EvidenceCapture,Worker,bind_dependency
sys.path.insert(0,str(D))
from upgraded_capture import CAMERA_LOOP

class ShortWorker(Worker):
 __init__=bind_dependency(Worker.__init__,'U',D/'thanks_runtime')

class FeedbackConsumer(PreviousConsumer):
 _original_init=bind_dependency(PreviousConsumer._original_init,'FrontWorker',ShortWorker)
 def raw(self,raw,word):
  q=super().raw(raw,word);q['stop_short_cycle_version']='seated_rest_dual_window_v1';return q

def admitted(path):
 a=json.loads(path.read_text('utf-8'))
 return a['admitted']is True and all(Path(p).is_file()and hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in a['pins'].items())

class CaptureService(PreviousCapture):
 _prepare=bind_dependency(EvidenceCapture.prepare,'FeedbackConsumer',FeedbackConsumer)
 _camera_loop=bind_dependency(EvidenceCapture.camera_loop,'_paced_camera_loop',CAMERA_LOOP)
 def prepare(self):
  began=time.perf_counter();self.stop_short_cycle_admitted=False;self._prepare()
  if self.state!='ready':return
  self.state='preparing'
  try:
   self.retry_admitted=admitted(R/'web/ADMISSION.json')
   self.stop_short_cycle_admitted=admitted(D/'web/ADMISSION.json')
   if not self.retry_admitted or not self.stop_short_cycle_admitted:raise RuntimeError('종료·짧은 반복 보완 검증 파일이 일치하지 않습니다.')
   self.consumer=ConsumerProxy(self.consumer,self);self.prepare_seconds=time.perf_counter()-began;self.state='ready'
  except Exception as exc:self.error=str(exc);self.state='error'
 def camera_loop(self):
  self.auto_end_diagnostic=None;self._camera_loop()
  if self.state=='done'and self.result is not None:
   self.result['second_pass_detector_calls']=self.result.get('selective_retry',{}).get('pose_calls',0)
   self.result['whole_video_redetection']=False
   if self.auto_end_diagnostic is not None:
    try:(self.directory/'auto_end_diagnostic.json').write_text(json.dumps(self.auto_end_diagnostic,ensure_ascii=False),encoding='utf-8')
    except OSError as exc:self.diagnostics['auto_end_log_error']=type(exc).__name__
 def status(self):
  q=super().status();q['stop_short_cycle_active']=getattr(self,'stop_short_cycle_admitted',False);q['stop_short_cycle_version']='seated_rest_dual_window_v1';return q
