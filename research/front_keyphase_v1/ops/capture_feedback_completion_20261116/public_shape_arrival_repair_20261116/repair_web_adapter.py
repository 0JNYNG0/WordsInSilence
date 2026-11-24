"""Add only qualified common evidence repairs; apple/capture/C remain unchanged."""
from pathlib import Path
import json,hashlib
from apple_web_adapter import FeedbackConsumer as PriorConsumer,CaptureService as PriorCapture,attach
from g6_runtime_adapter import Worker
from runtime_adapter import bind_dependency
H=Path(__file__).resolve().parent

class RepairWorker(Worker):__init__=bind_dependency(Worker.__init__,'U',H)

class FeedbackConsumer(PriorConsumer):
 def __init__(self):
  super().__init__();self.repair_admitted=False
  p=H/'web/ADMISSION.json'
  if p.exists():
   a=json.loads(p.read_text('utf-8'));self.repair_admitted=a['admitted']is True and all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in a['pins'].items())
  if self.repair_admitted:
   self.basic.close();self.basic=RepairWorker()
 def raw(self,raw,word):
  q=super().raw(raw,word)
  if word=='WORD1290':q['common_repair_version']='public_shape_arrival_20261116'if self.repair_admitted else'previous_frozen'
  return q

class CaptureService(PriorCapture):
 prepare=bind_dependency(PriorCapture.prepare,'FeedbackConsumer',FeedbackConsumer)
 def status(self):
  out=super().status();out['repair_module']='public_shape_arrival_20261116';out['repair_active']=bool(getattr(getattr(self,'consumer',None),'repair_admitted',False));return out
