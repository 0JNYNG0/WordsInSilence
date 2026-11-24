"""Only admitted public-evidence motion/form modules replace the prior worker."""
from pathlib import Path
import json,hashlib,sys
E=Path(__file__).resolve().parent
sys.path.insert(0,str(E.parent/'public_shape_endpoint_learning_20261116'))
from form_web_adapter import FeedbackConsumer as PriorConsumer,CaptureService as PriorCapture
from g6_runtime_adapter import Worker
from runtime_adapter import bind_dependency

class MotionWorker(Worker):
 __init__=bind_dependency(Worker.__init__,'U',E/'runtime')
class FeedbackConsumer(PriorConsumer):
 def __init__(self):
  super().__init__();self.motion_admitted=False;p=E/'web/ADMISSION.json'
  if p.exists():
   a=json.loads(p.read_text(encoding='utf-8'));self.motion_admitted=self.form_admitted and a['admitted']is True and all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in a['pins'].items())
  if self.motion_admitted:self.basic.close();self.basic=MotionWorker()
 def raw(self,raw,word):
  q=super().raw(raw,word)
  if word=='WORD1290':q['motion_evidence_module']='body_palm_cycles_v1'if self.motion_admitted else'previous_frozen'
  return q
class CaptureService(PriorCapture):
 prepare=bind_dependency(PriorCapture.prepare,'FeedbackConsumer',FeedbackConsumer)
 def status(self):
  q=super().status();q['motion_evidence_active']=bool(getattr(getattr(self,'consumer',None),'motion_admitted',False));return q
