from train_visible import *
from visible_lesson import shape_or_plane,contiguous

class ShapeLesson(PreservedArrivalLesson):
 def __init__(self):super().__init__();self.form=VisiblePalm()
 def form_states(self,ch,hand):
  s,recovered,form=super().form_states(ch,hand);ob=self.shape.observations[hand]
  # Whole-palm label is not a statement that each hidden finger is extended.
  # Distinct finger patterns and projection conflicts are not overwritten.
  use=(s!=2)&(form==0)&ob['available'].all(1)&~ob['projection_conflict']
  recovered|=use&(s!=0);s[use]=0
  return s,recovered,form

def observed_endpoint_events(ch,motion,limit):
 d=region_distance(ch);ts=ch['timestamp_us'];events=[]
 # No repetition target enters this function. Every candidate needs its own
 # measured far-to-near trajectory, plus the learned approach state.
 for run in contiguous(np.flatnonzero((motion==1)&np.isfinite(d))):
  if len(run)<4:continue
  a,b=int(run[0]),int(run[-1])
  if np.any(np.diff(ts[a:b+1])>100000):continue
  far=run[d[run]>limit]
  for near in contiguous(run[d[run]<=limit]):
   preceding=far[far<near[0]]
   if len(near)<2 or len(preceding)<2:continue
   e=int(near[1]);start=int(preceding[0])
   if np.nanmax(d[start:e+1])-np.nanmin(d[start:e+1])<=0:continue
   events.append(dict(start=start,arrival_index=e,end=int(near[-1]),source='learned_approach_and_observed_target_endpoint'));break
 return events

def append_endpoints(net,q,raw,started,proposals=None):
 ch=channels(raw);ts=ch['timestamp_us'];m=net.motion.sequence(ch);proposed=observed_endpoint_events(ch,m,net.target_region.limit)if proposals is None else proposals;old=list(q['events']);added=[]
 for e in proposed:
  if any(max(e['start'],p['start'])<=min(e['arrival_index'],p['arrival_index'])for p in old):continue
  added.append(e)
 q['observed_endpoint_events']=added
 if not added:q['seconds']=time.perf_counter()-started;return q
 events=sorted(old+added,key=lambda e:e['arrival_index']);q['events']=events;s=np.array(q['relation_states']);context=[dict(start=int(r[0]),end=int(r[-1]))for r in contiguous(np.flatnonzero(s>0))]+q.get('additional_body_events',[])+q.get('additional_region_events',[])+added
 for side,hand in [('left','L'),('right','R')]:
  for suffix,kind,states in [('shape','hand_form',np.array(q['hand_form_evidence'][hand]['states'])),('orientation','projected_plane',np.array(q['plane_states'][hand]))]:
   prior=next(c for c in q['criteria']if c['id']==side+'_'+suffix);c=shape_or_plane(prior['id'],hand,prior['expected'],states,events,ts,kind,context)
   if prior['status']=='difference':c=prior
   elif prior.get('uncertain_spans')and c['status']!='difference':c.update(status='unknown',uncertain_spans=prior['uncertain_spans'],uncertainty_reason=prior['uncertainty_reason'])
   q['criteria']=[c if p['id']==c['id']else p for p in q['criteria']]
 target=net.target_region.criterion(ch,events);q['criteria']=[target if c['id']==target['id']else c for c in q['criteria']]
 count=next(c for c in q['criteria']if c['id']=='repeat_order');count.update(status='supported'if len(events)>=count['expected_count']else'unknown',observed_count=len(events),source_spans=[[e['start'],e['end']]for e in events])
 q['movement_evidence'].update(confirmed_arrivals=events,confirmed_arrival_count=len(events),bridge_version='preserve_prior_cycles_add_observed_target_endpoints_v1');q['seconds']=time.perf_counter()-started
 return q

class EndpointLesson(PreservedArrivalLesson):
 def analyze(self,raw,lesson=None):
  started=time.perf_counter();return append_endpoints(self,super().analyze(raw,lesson),raw,started)

class CombinedLesson(ShapeLesson):
 def analyze(self,raw,lesson=None):
  started=time.perf_counter();return append_endpoints(self,super().analyze(raw,lesson),raw,started)

def main():
 torch.set_num_threads(2);summary={}
 for name,factory in [('shape',ShapeLesson),('endpoint',EndpointLesson),('combined',CombinedLesson)]:
  net=factory();out=L/name;out.mkdir(exist_ok=True);rows=[]
  for r in current_sources():
   a=read(L/'audit'/(r['id']+'.json'))['output'];b=net.analyze(dict(np.load(r['raw'])));aa={c['id']:c['status']for c in a['criteria']};bb={c['id']:c['status']for c in b['criteria']};row=dict(id=r['id'],before=aa,after=bb,events_before=len(a['events']),events_after=len(b['events']),lost=[k for k,v in aa.items()if v=='supported'and bb[k]!='supported'],new_warning=[k for k,v in bb.items()if v=='difference'and aa[k]!='difference'],old_event_bounds_preserved=all(e in b['events']for e in a['events']));rows.append(row);write(out/(r['id']+'.json'),dict(before=a,after=b))
   if aa!=bb or len(a['events'])!=len(b['events']):print(name,row,flush=True)
  result=dict(rows=rows,regressions=sum(bool(r['lost']or r['new_warning']or not r['old_event_bounds_preserved'])for r in rows),full_supported=sum(all(v=='supported'for v in r['after'].values())for r in rows));write(out/'RESULT.json',result);summary[name]={k:v for k,v in result.items()if k!='rows'}
 write(L/'ABLATION.json',summary);print(summary,flush=True)
if __name__=='__main__':main()
