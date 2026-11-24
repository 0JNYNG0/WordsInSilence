from apple_web_adapter import FeedbackConsumer as PreviousConsumer,CaptureService as PreviousCapture,attach as previous_attach,Worker,bind_dependency,quarantine
from pathlib import Path
import time,numpy as np,copy
U=Path(__file__).resolve().parent
class HelloWorker(Worker):__init__=bind_dependency(Worker.__init__,'U',U)
class FeedbackConsumer(PreviousConsumer):
 def __init__(self):super().__init__();self.hello=HelloWorker()
 def raw(self,raw,word):
  if word!='SEN0354':return super().raw(raw,word)
  st=time.perf_counter();obs=quarantine(raw['xy_px'],raw['score'],raw['timestamp_us'],int(raw['source_width']),int(raw['source_height']));z=dict(raw);z['xy_px']=obs['xy_px'];details=self.hello.infer(z,word)
  complete=all(c['status']=='supported'for c in details['lesson_acceptance']['criteria'])
  return dict(word=word,label='안녕하세요 · 단계 연습',tcn_top3=[],**details,feedback_version='hello-guided-captured-20261116',timestamp_us=raw['timestamp_us'].tolist(),frame_id=raw['frame_id'].tolist(),query_xy=np.nan_to_num(z['xy_px'],nan=-1,posinf=-1,neginf=-1).tolist(),width=int(raw['source_width']),height=int(raw['source_height']),overall_pass=False,exercise_complete=complete,whole_sign_correct=None,correction_validated=False,compute_seconds=time.perf_counter()-st,timing=dict(tcn_seconds=0,**details['runtime']),reference_video_reads=0,rgb_model_calls=0,recognition_scope='안녕하세요는 단어 인식 대신 두 동작의 단계 연습만 확인합니다.',notice='참고 수행자 1명의 단계 연습입니다. 단어 인식·다인 정확도를 검증한 실전 평가는 아니에요.')
class CaptureService(PreviousCapture):
 prepare=bind_dependency(PreviousCapture.prepare,'FeedbackConsumer',FeedbackConsumer)
 def start(self,word):
  if word!='SEN0354':return super().start(word)
  with self.guard:
   if self.state!='preview':raise ValueError('카메라 화면을 먼저 확인해 주세요.')
   self.word=word;self.record_requested=True
def attach(result):return copy.deepcopy(result)if result.get('feedback_version')=='hello-guided-captured-20261116'else previous_attach(result)
