"""Qualified body cycles and source-trained whole-palm evidence.

Precise finger extension remains separate. Newly recovered interaction regions
do not acquire a fist-accusation capability: that class remains unresolved until
the foreshortened-open-palm ambiguity is qualified. Existing corrections remain.
"""
from event_composition import *
from train_palm import SupplementalPalm

class QualifiedMotionLesson(EventLesson):
 def __init__(self):super().__init__();self.palm=SupplementalPalm()
 def form_states(self,ch,hand):
  s,recovered,oldform=super().form_states(ch,hand);new=self.palm.sequence(ch['tcn280'],hand);obs=self.shape.observations[hand]
  admit=(s<0)&(oldform==0)&(new==0)&obs['available'].all(1)&~obs['projection_conflict']&(self.plane.sequence(ch,hand)==0)
  s[admit]=0;recovered|=admit
  return s,recovered,oldform
 def analyze(self,raw,lesson=None):
  q=super().analyze(raw,lesson);added=q.get('additional_body_events',[])
  if not added:return q
  original=np.array(q['relation_states']);ts=np.array(q['timestamp_us']);new_region=np.zeros(len(ts),bool)
  for e in added:new_region[e['start']:e['end']+1]=True
  new_region&=original<=0
  for hand,side in [('L','left'),('R','right')]:
   c=next(c for c in q['criteria']if c['id']==side+'_shape');states=np.array(q['hand_form_evidence'][hand]['states'])
   if c['expected']!=0:continue
   unresolved=[[int(r[0]),int(r[-1])]for r in contiguous(np.flatnonzero(new_region&(states==1)))if len(r)>=3 and ts[r[-1]]-ts[r[0]]>=100000]
   if not unresolved:continue
   kept=[]
   for a,b in c['difference_spans']:
    mask=np.arange(a,b+1);known=mask[~new_region[a:b+1]|(states[a:b+1]!=1)]
    kept.extend([[int(r[0]),int(r[-1])]for r in contiguous(known)if len(r)>=3 and ts[r[-1]]-ts[r[0]]>=100000])
   c.update(status='difference'if kept else'unknown',difference_spans=kept,uncertain_spans=c.get('uncertain_spans',[])+unresolved,uncertainty_reason='newly_observed_region_curved_palm_or_fist_unqualified')
  return q
