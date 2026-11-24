"""Admitted view posture head; original recognition and other judgments retained."""
from pathlib import Path
import json,hashlib,sys
J=Path(__file__).resolve().parent
sys.path.insert(0,str(J.parent/'public_motion_decoupling_20261116'))
from motion_web_adapter import FeedbackConsumer as PriorConsumer,CaptureService as PriorCapture
from g6_runtime_adapter import Worker
from runtime_adapter import bind_dependency

class ViewWorker(Worker):
 __init__=bind_dependency(Worker.__init__,'U',J/'runtime')
class FeedbackConsumer(PriorConsumer):
 def __init__(self):
  super().__init__();self.view_admitted=False;p=J/'web/ADMISSION.json'
  if p.exists():
   a=json.loads(p.read_text(encoding='utf-8'));self.view_admitted=self.motion_admitted and a['admitted']is True and all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in a['pins'].items())
  if self.view_admitted:self.basic.close();self.basic=ViewWorker()
 def raw(self,raw,word):
  q=super().raw(raw,word)
  if word=='WORD1290':q['view_posture_module']='public_multiview_plane_v1'if self.view_admitted else'previous_frozen'
  return q
class CaptureService(PriorCapture):
 prepare=bind_dependency(PriorCapture.prepare,'FeedbackConsumer',FeedbackConsumer)
 def status(self):
  q=super().status();q['view_posture_active']=bool(getattr(getattr(self,'consumer',None),'view_admitted',False));return q
