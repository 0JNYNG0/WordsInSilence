"""A visible curved-palm/fist disagreement is uncertainty, not a correction."""
from event_composition import *
from train_palm import SupplementalPalm

class SemanticLesson(EventLesson):
 def __init__(self):super().__init__();self.supplemental=SupplementalPalm()
 def analyze(self,raw,lesson=None):
  started=time.perf_counter();q=super().analyze(raw,lesson);ch=channels(raw);ts=ch['timestamp_us'];s=np.array(q['relation_states'])
  context=[dict(start=int(r[0]),end=int(r[-1]))for r in contiguous(np.flatnonzero(s>0))]+q.get('additional_body_events',[])
  q['coarse_palm_conflict']={}
  for hand,side in [('R','right'),('L','left')]:
   prior=next(c for c in q['criteria']if c['id']==side+'_shape')
   obs=q['hand_form_evidence'][hand];states=np.array(obs['states']);new=self.supplemental.sequence(ch['tcn280'],hand)
   conflict=(states==1)&(new==0)&self.shape.observations[hand]['available'].all(1)
   q['coarse_palm_conflict'][hand]=conflict.tolist();obs['supplemental_form']=new.tolist()
   if prior['expected']!=0 or not conflict.any():continue
   guard=states.copy();guard[conflict]=-1
   c=shape_or_plane(prior['id'],hand,0,guard,q['events'],ts,'hand_form',context)
   # Keep the validated distinct-finger inspection path outside old phases.
   for a,b in prior['difference_spans']:
    if not conflict[a:b+1].any()and[a,b]not in c['difference_spans']:c['difference_spans'].append([a,b])
   spans=[[int(r[0]),int(r[-1])]for r in contiguous(np.flatnonzero(conflict))if len(r)>=3 and any(r[0]<=e['end']and r[-1]>=e['start']for e in context)]
   if c['difference_spans']:c['status']='difference'
   elif spans or prior.get('uncertain_spans'):c.update(status='unknown',uncertain_spans=spans+prior.get('uncertain_spans',[]),uncertainty_reason='coarse_palm_observers_disagree')
   q['criteria']=[c if x['id']==c['id']else x for x in q['criteria']]
  q['seconds']=time.perf_counter()-started
  return q
