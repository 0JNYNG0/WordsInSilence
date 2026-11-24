"""Post-capture reuse of existing admitted direction and handshape models.

Whole received clip may be used after recording ends; not a causal live model.
No training, new word classifier, synthetic pose or changed thresholds.
"""
from pathlib import Path
import sys,time,json
import numpy as np,torch
U=Path(__file__).resolve().parent;O=U.parents[1];LIVE=O/'hello_live_coach_20261115'
sys.path.insert(0,str(LIVE))
from coach import Coach
from action_runtime import ActionObserver
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
class BatchAction(ActionObserver):
 def prepare(self,raw):
  self.reset();self.channels=self.adapter.convert(raw['xy_px'],raw['score'],raw['timestamp_us'],raw['frame_id'],int(raw['source_width']),int(raw['source_height']));self.x=self.channels['tcn280'].astype(np.float32)
  pose=self.rc.prior.shared.normalize(self.x)
  with torch.inference_mode():
   h=torch.nn.functional.gelu(self.base.input(torch.from_numpy(pose[None]))).transpose(1,2)
   for layer in self.base.temporal:h=layer(h)
   self.h,_=self.base.context(h.transpose(1,2))
  self.calls=1
class BatchLesson:
 def __init__(self,action=None):self.action=action if action is not None else BatchAction();self.coach=Coach(action=self.action)
 def analyze(self,raw):
  started=time.perf_counter();self.coach.reset()
  try:self.action.prepare(raw)
  except self.action.adapter.PilotDataError:
   return dict(completed=[False,False],events=[],trace=[],seconds=time.perf_counter()-started,shared_encoder_calls=0,issue='body_scale_unavailable')
  ch=self.action.channels;n=len(ch['frame_id']);shape={h:self.coach.model.sequence(ch['tcn280'],h)for h in['L','R']};self.last_shape=shape;outputs=[]
  for i in range(3,n-3):
   t=int(raw['timestamp_us'][i]);self.coach.last_assessed=t;self.action.assessed_timestamp_us=t
   if i>3 and t-int(raw['timestamp_us'][i-1])>self.coach.gap_pause_us:self.coach._clear_pending()
   observations={h:dict(state=int(v['states'][i]),fingers=v['fingers'][i].astype(int).tolist(),available=v['available'][i].astype(bool).tolist(),reason=str(v['reasons'][i])if'reasons'in v else None,finger_reasons=v['finger_reasons'][i].tolist()if'finger_reasons'in v else[],geometry_reasons=v['geometry_reasons'][i].tolist()if'geometry_reasons'in v else[])for h,v in shape.items()}
   out=self.coach.consume_observation(raw['xy_px'][i],t,int(raw['source_width']),int(raw['source_height']),observations)
   outputs.append(dict(frame=i,timestamp_us=t,**out))
  return dict(completed=self.coach.completed,events=self.coach.events,trace=outputs,seconds=time.perf_counter()-started,shared_encoder_calls=self.action.calls,scope='Post-recording two-stage guided exercise; existing learned direction/shape plus explicit source guide rules; not trained SEN0354 TCN recognition or independent signer validation')
