"""New-basic-word public observation runtime, not yet a live success evaluator.

Consumes captured RTMW channels once. No performer-video comparison at runtime.
Complete native sequence enters the GRU; description-defined events are reporting
spans, never crops that remove preparation/transition context from inference.
"""
import core
ROOT=core.U
from core import Features,PhaseNet,states,approaches,read,O,torch,np
from shape_consensus import ShapeConsensus
from feedback_preview import repeated_relation

class RelationObserver:
 def __init__(self):
  self.models=[]
  for seed in [42,43,44]:
   ck=torch.load(ROOT/'source_corrected/runs'/f'frame_mlp_{seed}.pt',map_location='cpu',weights_only=False)
   m=PhaseNet(len(ck['mean']),'frame_mlp');m.load_state_dict(ck['state']);m.eval();self.models.append((m,ck['mean'],ck['std']))
 def __call__(self,f,q,ts):
  ss=[]
  with torch.inference_mode():
   for m,mu,sd in self.models:ss.append(states(m(torch.tensor(((f-mu)/sd)[None])).softmax(-1)[0].numpy(),q))
  ss=np.stack(ss);s=np.where((ss==ss[0]).all(0),ss[0],-1);return s,approaches(s,ts)

class BasicObserver:
 def __init__(self):
  self.feature=Features();self.relation=RelationObserver();self.shape=ShapeConsensus()
 def analyze(self,channels,lesson):
  assert lesson['observation_schema']=='open_hands_repeated_approach_v1'
  f,q,_=self.feature(channels);ss,events=self.relation(f,q,channels['timestamp_us']);n=len(ss)
  feedback=repeated_relation(lesson['repeat'],events,n,int((ss>=0).sum()))
  shapes={h:self.shape.sequence(channels['tcn280'],h)for h in ['L','R']}
  return dict(target=lesson['target'],timestamp_us=channels['timestamp_us'].tolist(),relation_states=ss.tolist(),events=events,
   relation_feedback=feedback,hand_states={h:v['states'].tolist()for h,v in shapes.items()},
   individual_finger_states={h:v['fingers'].tolist()for h,v in shapes.items()},
   full_sequence_context=True,whole_word_correct=None,qualified_for_live=False,
   remaining_requirements=['Hand-side/dorsum orientation and complete lesson acceptance need independent criteria validation.',
    'Not all normal edge-facing hands are observable by current shape model.'],
   evidence='Learned public temporal context + hand-local observation; configured source clause; native time, no forced DTW completion')

def lesson():
 return dict(target='WORD1290',label='감사',observation_schema='open_hands_repeated_approach_v1',
  source='https://sldict.korean.go.kr/front/sign/signContentsView.do?origin_no=2372&top_category=CTE',
  repeat=dict(id='repeated_hand_approach',source_phrase='왼 손등에 두 번 댄다',toward_ko='왼손등 쪽으로',count=2))
