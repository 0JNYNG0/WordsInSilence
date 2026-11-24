"""Admitted role-aware observer modules, retaining the original capture path."""
from pathlib import Path
import hashlib,json
from repair_web_adapter import FeedbackConsumer as PriorConsumer, CaptureService as PriorCapture, attach
from g6_runtime_adapter import Worker
from runtime_adapter import bind_dependency

D=Path(__file__).resolve().parent
class AppleRoleWorker(Worker):
 __init__=bind_dependency(Worker.__init__,'U',D/'apple_runtime')
class ShapeRecoveryWorker(Worker):
 __init__=bind_dependency(Worker.__init__,'U',D/'thanks_runtime')

class FeedbackConsumer(PriorConsumer):
 def __init__(self):
  super().__init__();self.visibility_admitted=False
  p=D/'web/ADMISSION.json'
  if p.exists():
   a=json.loads(p.read_text(encoding='utf-8'))
   self.visibility_admitted=a['admitted'] is True and all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in a['pins'].items())
  if self.visibility_admitted:
   self.apple.close();self.apple=AppleRoleWorker()
   self.basic.close();self.basic=ShapeRecoveryWorker()
 def raw(self,raw,word):
  q=super().raw(raw,word)
  if word in ['WORD0099','WORD1290']:
   q['visibility_module']='independent_hand_evidence_v1' if self.visibility_admitted else 'previous_frozen'
  return q

class CaptureService(PriorCapture):
 prepare=bind_dependency(PriorCapture.prepare,'FeedbackConsumer',FeedbackConsumer)
 def status(self):
  q=super().status();q['visibility_active']=bool(getattr(getattr(self,'consumer',None),'visibility_admitted',False));return q
