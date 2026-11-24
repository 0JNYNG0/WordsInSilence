"""Isolated basic-lesson branch; existing four-word and C/T3-F paths unchanged."""
from pathlib import Path
import sys,json,copy,time,hashlib
import numpy as np,torch
V=Path(__file__).resolve().parent;O=V.parents[1]
from g6_runtime_adapter import FeedbackConsumer as PreviousConsumer,CaptureService as PreviousCapture,Worker,attach as previous_attach
from runtime_adapter import bind_dependency
from guard import quarantine
from runtime_candidate import A

class BasicWorker(Worker):
 __init__=bind_dependency(Worker.__init__,'U',V)

class FeedbackConsumer(PreviousConsumer):
 def __init__(self):
  super().__init__();self.basic=BasicWorker();self.basic_admitted=False
  path=V/'web/ADMISSION.json'
  if path.exists():
   admission=json.loads(path.read_text('utf-8'))
   self.basic_admitted=admission['admitted']is True and all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in admission['pins'].items())
 def raw(self,raw,word):
  if word!='WORD1290':return super().raw(raw,word)
  started=time.perf_counter();obs=quarantine(raw['xy_px'],raw['score'],raw['timestamp_us'],int(raw['source_width']),int(raw['source_height']))
  observed=dict(raw);observed['xy_px']=obs['xy_px']
  ch=A.convert(observed['xy_px'],raw['score'],raw['timestamp_us'],raw['frame_id'],int(raw['source_width']),int(raw['source_height']))
  x=ch['tcn280']
  if len(x)<4:raise ValueError('촬영이 너무 짧아요. 동작을 마친 뒤 종료해 주세요.')
  t=time.perf_counter()
  with torch.inference_mode():p=self.base.tcn(torch.from_numpy(x[None]),torch.ones(1,len(x)))[0].softmax(-1).numpy()
  tcn_seconds=time.perf_counter()-t;rank=np.argsort(-p,kind='stable')[:3];details=self.basic.infer(observed,word)
  admitted=self.basic_admitted
  verdict=details['lesson_acceptance'];verdict['qualified']=admitted;verdict['qualification_scope']='Observed projected features of the registered gratitude lesson; not certified 3D contact or all KSL correctness.'
  verdict['overall_pass']=admitted and all(c['status']=='supported'for c in verdict['criteria'])
  return dict(word=word,label='감사',tcn_top3=[dict(word=self.base.f['classes'][i],label=self.base.f['labels'][i],probability=float(p[i]))for i in rank],
   **details,feedback_version='basic-visible-20261116',timestamp_us=raw['timestamp_us'].tolist(),frame_id=raw['frame_id'].tolist(),
   query_xy=np.nan_to_num(obs['xy_px'],nan=-1,posinf=-1,neginf=-1).tolist(),width=int(raw['source_width']),height=int(raw['source_height']),
   observation_guard=dict(version=obs['version'],raw_preserved=True,repaired_coordinates=False),
   overall_pass=verdict['overall_pass'],whole_sign_correct=None,correction_validated=admitted,
   compute_seconds=time.perf_counter()-started,timing=dict(tcn_seconds=tcn_seconds,**details['runtime']),
   reference_video_reads=0,rgb_model_calls=0,word_classifier_uses_requested_target=False,
   notice='표시용 참고와 DTW는 재생에만 사용해요. 학습된 관측과 설명의 필수 항목으로 수행을 확인해요.')

class CaptureService(PreviousCapture):
 prepare=bind_dependency(PreviousCapture.prepare,'FeedbackConsumer',FeedbackConsumer)
 def start(self,word):
  if word!='WORD1290':return super().start(word)
  with self.guard:
   if self.state!='preview':raise ValueError('카메라 화면을 먼저 확인해 주세요.')
   self.word=word;self.record_requested=True
 def status(self):
  out=super().status();out['feedback_version']='basic-visible-20261116 + preserved G6';return out

def attach(result):return copy.deepcopy(result)if result.get('feedback_version')=='basic-visible-20261116'else previous_attach(result)
