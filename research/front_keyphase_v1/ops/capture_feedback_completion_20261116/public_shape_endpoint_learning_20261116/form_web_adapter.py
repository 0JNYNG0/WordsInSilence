"""Only the admitted shared hand-form/inspection module changes."""
from pathlib import Path
import json,hashlib
from visibility_web_adapter import FeedbackConsumer as PriorConsumer,CaptureService as PriorCapture
from g6_runtime_adapter import Worker
from runtime_adapter import bind_dependency
D=Path(__file__).resolve().parent
class FormWorker(Worker):
 __init__=bind_dependency(Worker.__init__,'U',D/'runtime')
class FeedbackConsumer(PriorConsumer):
 def __init__(self):
  super().__init__();self.form_admitted=False;p=D/'web/ADMISSION.json'
  if p.exists():
   a=json.loads(p.read_text(encoding='utf-8'));self.form_admitted=self.visibility_admitted and a['admitted']is True and all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in a['pins'].items())
  if self.form_admitted:self.basic.close();self.basic=FormWorker()
 def raw(self,raw,word):
  q=super().raw(raw,word)
  if word=='WORD1290':q['hand_form_module']='visible_interaction_envelope_v1'if self.form_admitted else'previous_frozen'
  return q
class CaptureService(PriorCapture):
 prepare=bind_dependency(PriorCapture.prepare,'FeedbackConsumer',FeedbackConsumer)
 def status(self):
  q=super().status();q['hand_form_active']=bool(getattr(getattr(self,'consumer',None),'form_admitted',False));return q
