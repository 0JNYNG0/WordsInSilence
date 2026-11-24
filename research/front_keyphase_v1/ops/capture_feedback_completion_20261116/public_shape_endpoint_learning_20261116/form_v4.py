"""Use the new learned relation only to retain inspection, not to pass events.

The retrained relation is not qualified to replace completion. It may propose
an inspection interval. Attributed error still requires two hand observers to
agree on a persistent visible non-palm form; no new events or positive support
come from this proposal. Word/person identity are not neural inputs.
"""
import time
from form_v2 import FinalFormLesson
from form_candidate import np,adapter,contiguous,shape_or_plane
from train_relation import RelationCandidate
class JointInspectionLesson(FinalFormLesson):
 inspection_forms=(1,2)
 def __init__(self):super().__init__();self.inspection_relation=RelationCandidate()
 def analyze(self,raw,lesson=None):
  started=time.perf_counter();q=super().analyze(raw,lesson)
  ch=adapter.convert(raw['xy_px'],raw['score'],raw['timestamp_us'],raw['frame_id'],int(raw['source_width']),int(raw['source_height']))
  f,quality,_=self.feature(ch);proposal,_=self.inspection_relation(f,quality,ch['timestamp_us'])
  anchors=np.flatnonzero((np.array(q['relation_states'])>0)|(proposal>0));q['inspection_relation_states']=proposal.tolist()
  if len(anchors)<2:return q
  a,b=int(anchors[0]),int(anchors[-1]);q['shape_inspection_envelope']=[a,b]
  for hand,side in [('R','right'),('L','left')]:
   prior=next(c for c in q['criteria']if c['id']==side+'_shape')
   if prior['expected']!=0 or prior['status']=='difference':continue
   ev=q['hand_form_evidence'][hand];states=np.array(ev['states']);form=np.array(ev['form']);available=self.shape.observations[hand]['available'].all(1)
   agreed=available&np.isin(states,self.inspection_forms)&(states==form);guard=np.where(agreed,states,-1)
   probe=shape_or_plane(side+'_shape',hand,0,guard,[],np.array(q['timestamp_us']),'hand_form',[dict(start=a,end=b)])
   if probe['status']=='difference':
    updated=dict(prior,status='difference',difference_spans=probe['difference_spans'],inspection='Learned interval proposal + two agreeing visible shape observers; never adds completion')
    q['criteria']=[updated if c['id']==prior['id']else c for c in q['criteria']]
  q['seconds']=time.perf_counter()-started
  return q
