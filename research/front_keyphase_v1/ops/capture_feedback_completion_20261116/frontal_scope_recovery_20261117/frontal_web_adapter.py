"""Direct front-only worker construction; no discarded side-view worker startup."""
from pathlib import Path
import sys,json,hashlib,torch
N=Path(__file__).resolve().parent;G=N.parent;O=G.parent
sys.path[:0]=[str(G/'apple_v1'),str(G/'completion_v1')]
from apple_web_adapter import FeedbackConsumer as AppleConsumer,CaptureService as AppleCapture,attach
from g6_runtime_adapter import FeedbackConsumer as CommonConsumer,Worker
from runtime_adapter import bind_dependency

class FrontWorker(Worker):
 __init__=bind_dependency(Worker.__init__,'U',N/'thanks_runtime')
class AppleRoleWorker(Worker):
 __init__=bind_dependency(Worker.__init__,'U',N/'apple_runtime')

class FeedbackConsumer(AppleConsumer):
 def __init__(self):
  CommonConsumer.__init__(self)
  self.basic=FrontWorker();self.apple=AppleRoleWorker()
  c=torch.load(O/'basic_demo_feedback_20261115/apple_classifier.pt',map_location='cpu',weights_only=False)
  assert hashlib.sha256(Path(c['base_checkpoint']).read_bytes()).hexdigest()==c['base_sha256']
  base=torch.load(c['base_checkpoint'],map_location='cpu',weights_only=True)['state_dict']
  assert all(torch.equal(v,self.base.tcn.state_dict()[k])for k,v in base.items())
  self.appended=torch.nn.Linear(c['state_dict']['weight'].shape[1],1);self.appended.load_state_dict(c['state_dict']);self.appended.eval();self.captured=None
  def capture(m,inp,out):self.captured=(inp[0].detach(),out.detach())
  self.hook=self.base.tcn.head.register_forward_hook(capture)
  admission=json.loads((N/'web/ADMISSION.json').read_text(encoding='utf-8'))
  self.frontal_admitted=admission['admitted']is True and all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in admission['pins'].items())
  self.basic_admitted=self.frontal_admitted;self.apple_admitted=self.frontal_admitted
 def raw(self,raw,word):
  q=super().raw(raw,word)
  q['feedback_scope']='single_front_camera';q['frontal_module']='front_observation_20261117'
  return q

class CaptureService(AppleCapture):
 prepare=bind_dependency(AppleCapture.prepare,'FeedbackConsumer',FeedbackConsumer)
 def status(self):
  q=super().status();q['frontal_scope_active']=bool(getattr(getattr(self,'consumer',None),'frontal_admitted',False));q['feedback_scope']='single_front_camera';return q

