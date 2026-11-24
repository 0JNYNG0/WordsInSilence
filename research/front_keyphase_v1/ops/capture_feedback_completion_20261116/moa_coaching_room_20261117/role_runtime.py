from pathlib import Path
import sys
D=Path(__file__).resolve().parent;G=D.parent;O=G.parent
sys.path.insert(0,str(G/'rest_stability_endpoint_20261117'))
from stable_runtime import CaptureService as StableCapture
from upgraded_runtime import FeedbackConsumer as PreviousConsumer,ConsumerProxy,EvidenceCapture,Worker,bind_dependency,admitted

class RoleWorker(Worker):
 __init__=bind_dependency(Worker.__init__,'U',D/'thanks_runtime')

class FeedbackConsumer(PreviousConsumer):
 _original_init=bind_dependency(PreviousConsumer._original_init,'FrontWorker',RoleWorker)
 def raw(self,raw,word):
  q=super().raw(raw,word);q['endpoint_role_version']='separate_geometry_proposal_observed_near_v1';return q

class CaptureService(StableCapture):
 _prepare=bind_dependency(EvidenceCapture.prepare,'FeedbackConsumer',FeedbackConsumer)
 def prepare(self):
  self.endpoint_role_admitted=False;super().prepare()
  if self.state!='ready':return
  self.state='preparing'
  try:
   self.endpoint_role_admitted=admitted(D/'web/ADMISSION.json')
   if not self.endpoint_role_admitted:raise RuntimeError('관측 역할 분리 검증 파일이 일치하지 않습니다.')
   self.state='ready'
  except Exception as exc:self.error=str(exc);self.state='error'
 def status(self):
  q=super().status();q['endpoint_role_active']=getattr(self,'endpoint_role_admitted',False);q['companion_ui_version']='moa_coaching_room_v1';return q
