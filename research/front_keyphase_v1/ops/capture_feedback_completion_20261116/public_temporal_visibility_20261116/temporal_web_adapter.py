from pathlib import Path
import json,hashlib,sys
L=Path(__file__).resolve().parent
sys.path.insert(0,str(L.parent/'public_view_arrival_support_20261116'))
from arrival_web_adapter import FeedbackConsumer as PriorConsumer,CaptureService as PriorCapture
from g6_runtime_adapter import Worker
from runtime_adapter import bind_dependency
class TemporalWorker(Worker):
 __init__=bind_dependency(Worker.__init__,'U',L/'runtime')
class FeedbackConsumer(PriorConsumer):
 def __init__(self):
  super().__init__();self.temporal_admitted=False;p=L/'web/ADMISSION.json'
  if p.exists():
   a=json.loads(p.read_text(encoding='utf-8'));self.temporal_admitted=self.arrival_admitted and a['admitted']is True and all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in a['pins'].items())
  if self.temporal_admitted:self.basic.close();self.basic=TemporalWorker()
 def raw(self,raw,word):
  q=super().raw(raw,word)
  if word=='WORD1290':q['temporal_visibility_module']='qualified_palm_observed_endpoint_v1'if self.temporal_admitted else'previous_frozen'
  return q
class CaptureService(PriorCapture):
 prepare=bind_dependency(PriorCapture.prepare,'FeedbackConsumer',FeedbackConsumer)
 def status(self):
  q=super().status();q['temporal_visibility_active']=bool(getattr(getattr(self,'consumer',None),'temporal_admitted',False));return q
