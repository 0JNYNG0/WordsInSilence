from pathlib import Path
import json,hashlib,sys
K=Path(__file__).resolve().parent
sys.path.insert(0,str(K.parent/'public_view_observation_repair_20261116'))
from view_web_adapter import FeedbackConsumer as PriorConsumer,CaptureService as PriorCapture
from g6_runtime_adapter import Worker
from runtime_adapter import bind_dependency
class ArrivalWorker(Worker):
 __init__=bind_dependency(Worker.__init__,'U',K/'runtime')
class FeedbackConsumer(PriorConsumer):
 def __init__(self):
  super().__init__();self.arrival_admitted=False;p=K/'web/ADMISSION.json'
  if p.exists():
   a=json.loads(p.read_text(encoding='utf-8'));self.arrival_admitted=self.view_admitted and a['admitted']is True and all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in a['pins'].items())
  if self.arrival_admitted:self.basic.close();self.basic=ArrivalWorker()
 def raw(self,raw,word):
  q=super().raw(raw,word)
  if word=='WORD1290':q['arrival_module']='preserved_cycles_projected_arrivals_v1'if self.arrival_admitted else'previous_frozen'
  return q
class CaptureService(PriorCapture):
 prepare=bind_dependency(PriorCapture.prepare,'FeedbackConsumer',FeedbackConsumer)
 def status(self):
  q=super().status();q['arrival_support_active']=bool(getattr(getattr(self,'consumer',None),'arrival_admitted',False));return q
