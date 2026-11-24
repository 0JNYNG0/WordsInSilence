"""Load the existing real raw consumer in an isolated process; never change its files."""
from common import *
import copy,numpy as np,torch
# Legacy modules use generic names; bind each to its actual owner before importing.
sys.modules['common']=load('_fullpath_joint_common',J/'common.py')
sys.path.insert(0,str(D));sys.modules['task']=load('_fullpath_demo_task',D/'task.py')
consumer=load('_fullpath_consumer',D/'consumer.py');adapter=load('_fullpath_completion_adapter',E/'runtime_adapter.py')
class TracedAuditor:
 def __init__(self):self.real=adapter.CompletionAuditor();self.calls=0
 def apply(self,d,x,ts):
  self.before=copy.deepcopy(d);self.converted=x.copy();self.ts=ts.copy();self.calls+=1;out=self.real.apply(d,x,ts);self.after=copy.deepcopy(out);return out
def instantiate():
 torch.set_num_threads(4);torch.set_num_interop_threads(1);c=consumer.FeedbackConsumer();a=TracedAuditor();adapter.attach_to_isolated_consumer(c,a);return c,a
def one(c,a,raw,word):
 t=time.perf_counter();out=c.raw(raw,word);elapsed=time.perf_counter()-t
 # Baseline receives the SAME final observation policy, using captured channel output.
 baseline=copy.deepcopy(a.before);before_final=copy.deepcopy(a.after);q=np.asarray(out['observation_guard']['hand_quarantined'])
 for card in baseline['cards']:
  span=card.get('query_indices');sides=[0 if s=='L' else 1 for s in card['hands']]
  if span and q[span[0]:span[1]+1,sides].any():card.update(status='unknown',display_status='uncertain',message='이 구간의 손을 일부 확인하기 어려워요. 참고와 함께 살펴보세요.',observation_quarantine_overlap=True,educational_wrong=False)
 for k in ['query_xy','observation_guard','width','height']:baseline[k]=out[k]
 baseline['baseline_derived_from_same_neural_forward']=True
 return baseline,out,before_final,a.converted,elapsed
def compact(d):return dict(display=[c['display_status'] for c in d['cards']],status=[c['status'] for c in d['cards']],required=[c.get('required',True) for c in d['cards']],top1=d['tcn_top3'][0]['word'],all_required_similar=all(c['display_status']=='similar' for c in d['cards'] if c.get('required',True)))
