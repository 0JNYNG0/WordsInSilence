"""Preserve validated cycles; add body-observed cycles, not an entire new phase.

Body posture does not specify that every visible-hand frame is part of a sign.
Only independently observed approach/arrival cycles add inspection intervals.
No word/person identity, desired repetition count, or reference speed selects them.
"""
from common import *
from repair import linked_arrivals
from observed_gap import preserve_sparse_arrivals
from visible_lesson import observed_action_extent,shape_or_plane,contiguous

class EventLesson(QualifiedFormLesson):
 def __init__(self):super().__init__();self.body_relation=GeometryRelation()
 def analyze(self,raw,lesson=None):
  started=time.perf_counter();q=super().analyze(raw,lesson)
  ch=channels(raw);ts=ch['timestamp_us'];s=np.array(q['relation_states']);g,_=self.body_relation(self.last_features,None,ts)
  proposal=np.where((s<=0)&(g>0),g,s);m=self.motion.sequence(ch)
  ids=[91,96,100,104,108,112,117,121,125,129]
  available=ch['point_available'][:,ids].all(1)&np.isfinite(ch['body_relative_xy'][:,ids]).all((1,2))
  ev=preserve_sparse_arrivals(proposal,m,linked_arrivals(proposal,m),available,ts)
  candidates=observed_action_extent(ev['confirmed_arrivals'],proposal,m,available)
  old=q['events'];added=[]
  for e in candidates:
   # Overlapping approach/arrival intervals represent the same observed cycle.
   # Preserve its original bounds, including the original terminal hold.
   if any(max(e['start'],p['start'])<=min(e['arrival_index'],p['arrival_index'])for p in old):continue
   added.append(dict(e,source='body_posture_plus_independent_motion'))
  q['body_relation_states']=g.tolist();q['additional_body_events']=added
  if not added:q['seconds']=time.perf_counter()-started;return q
  events=sorted(old+added,key=lambda e:e['arrival_index']);q['events']=events
  context=[dict(start=int(r[0]),end=int(r[-1]))for r in contiguous(np.flatnonzero(s>0))]+added
  for side,hand in [('right','R'),('left','L')]:
   for suffix,kind,states in [('shape','hand_form',np.array(q['hand_form_evidence'][hand]['states'])),('orientation','projected_plane',np.array(q['plane_states'][hand]))]:
    prior=next(c for c in q['criteria']if c['id']==side+'_'+suffix)
    c=shape_or_plane(prior['id'],hand,prior['expected'],states,events,ts,kind,context)
    if prior['status']=='difference':c=prior
    elif prior.get('uncertain_spans')and c['status']!='difference':c.update(status='unknown',uncertain_spans=prior['uncertain_spans'],uncertainty_reason=prior['uncertainty_reason'])
    q['criteria']=[c if x['id']==c['id']else x for x in q['criteria']]
  target=self.target_region.criterion(ch,events)
  q['criteria']=[target if c['id']==target['id']else c for c in q['criteria']]
  count=next(c for c in q['criteria']if c['id']=='repeat_order')
  count.update(status='supported'if len(events)>=count['expected_count']else'unknown',observed_count=len(events),source_spans=[[e['start'],e['end']]for e in events])
  q['movement_evidence']['confirmed_arrivals']=events
  q['movement_evidence']['confirmed_arrival_count']=len(events)
  q['movement_evidence']['bridge_version']='preserve_original_cycles_add_body_observed_cycles_v1'
  q['seconds']=time.perf_counter()-started
  return q
