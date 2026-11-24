"""Keep deployed cycles intact. Additional arrival proposals cannot retime them."""
from arrival_candidate import *
from repair import linked_arrivals
from observed_gap import preserve_sparse_arrivals
from visible_lesson import observed_action_extent,shape_or_plane,contiguous

class PreservedArrivalLesson(PlaneViewLesson):
 def __init__(self):super().__init__();self.arrival_relation=ArrivalRelation()
 def analyze(self,raw,lesson=None):
  started=time.perf_counter();q=super().analyze(raw,lesson);ch=channels(raw);ts=ch['timestamp_us'];s=np.array(q['relation_states']);self.arrival_relation.channels=ch;g,_=self.arrival_relation(self.last_features,None,ts);q['projected_arrival_added']=self.arrival_relation.added
  proposal=np.where((s<=0)&(g>0),g,s);m=self.motion.sequence(ch);ids=[91,96,100,104,108,112,117,121,125,129];available=ch['point_available'][:,ids].all(1)&np.isfinite(ch['body_relative_xy'][:,ids]).all((1,2))
  ev=preserve_sparse_arrivals(proposal,m,linked_arrivals(proposal,m),available,ts);candidates=observed_action_extent(ev['confirmed_arrivals'],proposal,m,available);old=list(q['events']);added=[]
  for e in candidates:
   if any(max(e['start'],p['start'])<=min(e['arrival_index'],p['arrival_index'])for p in old):continue
   added.append(dict(e,source='source_trained_projected_region_arrival'))
  q['additional_region_events']=added
  if not added:q['seconds']=time.perf_counter()-started;return q
  events=sorted(old+added,key=lambda e:e['arrival_index']);q['events']=events
  context=[dict(start=int(r[0]),end=int(r[-1]))for r in contiguous(np.flatnonzero(s>0))]+q.get('additional_body_events',[])+added;new_region=np.zeros(len(ts),bool)
  for e in added:new_region[e['start']:e['end']+1]=True
  for e in old:new_region[e['start']:e['end']+1]=False
  for side,hand in [('right','R'),('left','L')]:
   for suffix,kind,states in [('shape','hand_form',np.array(q['hand_form_evidence'][hand]['states'])),('orientation','projected_plane',np.array(q['plane_states'][hand]))]:
    prior=next(c for c in q['criteria']if c['id']==side+'_'+suffix);c=shape_or_plane(prior['id'],hand,prior['expected'],states,events,ts,kind,context)
    if prior['status']=='difference':c=prior
    elif prior.get('uncertain_spans')and c['status']!='difference':c.update(status='unknown',uncertain_spans=prior['uncertain_spans'],uncertainty_reason=prior['uncertainty_reason'])
    if suffix=='shape'and prior['status']!='difference':
     unresolved=[[int(r[0]),int(r[-1])]for r in contiguous(np.flatnonzero(new_region&(states==1)))if len(r)>=3 and ts[r[-1]]-ts[r[0]]>=100000]
     if unresolved:
      kept=[]
      for a,b in c['difference_spans']:
       ix=np.arange(a,b+1);kept.extend([[int(r[0]),int(r[-1])]for r in contiguous(ix[~new_region[a:b+1]|(states[a:b+1]!=1)])if len(r)>=3 and ts[r[-1]]-ts[r[0]]>=100000])
      c.update(status='difference'if kept else'unknown',difference_spans=kept,uncertain_spans=c.get('uncertain_spans',[])+unresolved,uncertainty_reason='new_region_curled_palm_or_fist_unqualified')
    q['criteria']=[c if p['id']==c['id']else p for p in q['criteria']]
  target=self.target_region.criterion(ch,events);q['criteria']=[target if c['id']==target['id']else c for c in q['criteria']]
  count=next(c for c in q['criteria']if c['id']=='repeat_order');count.update(status='supported'if len(events)>=count['expected_count']else'unknown',observed_count=len(events),source_spans=[[e['start'],e['end']]for e in events])
  q['movement_evidence'].update(confirmed_arrivals=events,confirmed_arrival_count=len(events),bridge_version='preserve_deployed_cycles_add_source_trained_region_arrivals_v1');q['seconds']=time.perf_counter()-started
  return q
