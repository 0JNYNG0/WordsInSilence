"""Explicit 100+apple research extension; original100 weights never overwritten."""
from pathlib import Path
import sys,copy,json,hashlib,time
import numpy as np,torch
U=Path(__file__).resolve().parent;O=U.parents[1];V=U.parent/'completion_v1'
from web_adapter import FeedbackConsumer as PreviousConsumer,CaptureService as PreviousCapture,attach as previous_attach
from g6_runtime_adapter import Worker
from runtime_adapter import bind_dependency
from guard import quarantine
from runtime_candidate import A
class AppleWorker(Worker):__init__=bind_dependency(Worker.__init__,'U',U)
class FeedbackConsumer(PreviousConsumer):
 def __init__(self):
  super().__init__();self.apple=AppleWorker();c=torch.load(O/'basic_demo_feedback_20261115/apple_classifier.pt',map_location='cpu',weights_only=False)
  assert hashlib.sha256(Path(c['base_checkpoint']).read_bytes()).hexdigest()==c['base_sha256']
  base=torch.load(c['base_checkpoint'],map_location='cpu',weights_only=True)['state_dict'];assert all(torch.equal(v,self.base.tcn.state_dict()[k])for k,v in base.items())
  self.appended=torch.nn.Linear(c['state_dict']['weight'].shape[1],1);self.appended.load_state_dict(c['state_dict']);self.appended.eval();self.captured=None
  def capture(m,inp,out):self.captured=(inp[0].detach(),out.detach())
  self.hook=self.base.tcn.head.register_forward_hook(capture);self.apple_admitted=False
  p=U/'web/ADMISSION.json'
  if p.exists():
   a=json.loads(p.read_text('utf-8'));self.apple_admitted=a['admitted']is True and all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in a['pins'].items())
 def ranks(self):
  if self.captured is None:raise RuntimeError('TCN result missing')
  f,old=self.captured
  with torch.inference_mode():p=torch.cat([old,self.appended(f)],-1)[0].softmax(-1).numpy()
  classes=self.base.f['classes']+['WORD0099'];labels=self.base.f['labels']+['사과'];ix=np.argsort(-p,kind='stable')[:3]
  return [dict(word=classes[i],label=labels[i],probability=float(p[i]))for i in ix]
 def raw(self,raw,word):
  start=time.perf_counter();self.captured=None
  if word!='WORD0099':
   out=super().raw(raw,word);out['tcn_top3']=self.ranks();out['recognition_scope']='original100 frozen + apple output';return out
  obs=quarantine(raw['xy_px'],raw['score'],raw['timestamp_us'],int(raw['source_width']),int(raw['source_height']));z=dict(raw);z['xy_px']=obs['xy_px'];ch=A.convert(z['xy_px'],raw['score'],raw['timestamp_us'],raw['frame_id'],int(raw['source_width']),int(raw['source_height']));x=ch['tcn280']
  if len(x)<4:raise ValueError('촬영이 너무 짧아요. 동작을 마친 뒤 종료해 주세요.')
  t=time.perf_counter()
  with torch.inference_mode():self.base.tcn(torch.tensor(x[None]),torch.ones(1,len(x)))
  rank=self.ranks();tcn_seconds=time.perf_counter()-t;details=self.apple.infer(z,word);verdict=details['lesson_acceptance'];verdict['qualified']=self.apple_admitted;verdict['overall_pass']=self.apple_admitted and all(c['status']=='supported'for c in verdict['criteria']);verdict['qualification_scope']='Source-reviewed visible requirements of apple; exact contact and hidden fingers not certified.'
  return dict(word=word,label='사과',tcn_top3=rank,**details,feedback_version='apple-visible-20261116',timestamp_us=raw['timestamp_us'].tolist(),frame_id=raw['frame_id'].tolist(),query_xy=np.nan_to_num(obs['xy_px'],nan=-1,posinf=-1,neginf=-1).tolist(),width=int(raw['source_width']),height=int(raw['source_height']),observation_guard=dict(version=obs['version'],raw_preserved=True,repaired_coordinates=False),overall_pass=verdict['overall_pass'],whole_sign_correct=None,correction_validated=self.apple_admitted,compute_seconds=time.perf_counter()-start,timing=dict(tcn_seconds=tcn_seconds,**details['runtime']),reference_video_reads=0,rgb_model_calls=0,word_classifier_uses_requested_target=False,recognition_scope='original100 frozen + apple output',notice='입술 아래 동작과 닦기를 따로 확인해요. DTW는 비교 재생에만 사용해요.')
class CaptureService(PreviousCapture):
 prepare=bind_dependency(PreviousCapture.prepare,'FeedbackConsumer',FeedbackConsumer)
 def start(self,word):
  if word!='WORD0099':return super().start(word)
  with self.guard:
   if self.state!='preview':raise ValueError('카메라 화면을 먼저 확인해 주세요.')
   self.word=word;self.record_requested=True
def attach(result):return copy.deepcopy(result)if result.get('feedback_version')=='apple-visible-20261116'else previous_attach(result)
