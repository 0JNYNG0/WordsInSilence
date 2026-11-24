from pathlib import Path
import json,hashlib,sys
M=Path(__file__).resolve().parent
sys.path.insert(0,str(M.parent/'public_temporal_visibility_20261116'))
from temporal_web_adapter import FeedbackConsumer as PriorConsumer,CaptureService as PriorCapture
from g6_runtime_adapter import Worker
from runtime_adapter import bind_dependency
class CycleWorker(Worker):
 __init__=bind_dependency(Worker.__init__,'U',M/'runtime')
class FeedbackConsumer(PriorConsumer):
 def __init__(self):
  super().__init__();self.cycle_admitted=False;p=M/'web/ADMISSION.json'
  if p.exists():
   a=json.loads(p.read_text(encoding='utf8'));self.cycle_admitted=self.temporal_admitted and a['admitted']is True and all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in a['pins'].items())
  if self.cycle_admitted:self.basic.close();self.basic=CycleWorker()
 def raw(self,raw,word):
  q=super().raw(raw,word)
  if word=='WORD1290':q['cycle_observability_module']='qualified_distinct_episodes_and_learned_edge_motion_v1'if self.cycle_admitted else'previous_frozen'
  return q
class CaptureService(PriorCapture):
 prepare=bind_dependency(PriorCapture.prepare,'FeedbackConsumer',FeedbackConsumer)
 def status(self):
  q=super().status();q['cycle_observability_active']=bool(getattr(getattr(self,'consumer',None),'cycle_admitted',False));return q
