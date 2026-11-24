from cycle_candidate import *
import importlib.util
_spec=importlib.util.spec_from_file_location('cycle_region_motion',M/'train_motion.py');_motion=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_motion);RegionMotion=_motion.RegionMotion
from repair import linked_arrivals
from observed_gap import preserve_sparse_arrivals
from visible_lesson import observed_action_extent

class LearnedCycleLesson(CycleLesson):
 def __init__(self):super().__init__();self.region_motion=RegionMotion()
 def analyze(self,raw,lesson=None):
  started=time.perf_counter();q=super().analyze(raw,lesson)
  if next(c for c in q['criteria']if c['id']=='repeat_order')['status']=='supported':q['learned_cycle_support']=dict(added=[],supplemented_frames=[],base_motion_preserved=True,skipped='movement_already_observed');q['seconds']=time.perf_counter()-started;return q
  ch=channels(raw);ts=ch['timestamp_us'];old=copy.deepcopy(q['events']);m=self.motion.sequence(ch);supp=self.region_motion.sequence(ch);combined=np.where(m<0,supp,m);body=np.array(q['body_relation_states']);s=np.array(q['relation_states']);proposal=np.where(body>0,body,s);ids=[91,96,100,104,108,112,117,121,125,129];available=ch['point_available'][:,ids].all(1)&np.isfinite(ch['body_relative_xy'][:,ids]).all((1,2))
  candidates=observed_action_extent(preserve_sparse_arrivals(proposal,combined,linked_arrivals(proposal,combined),available,ts)['confirmed_arrivals'],proposal,combined,available);added=[];d=palm_diameter_distance(ch)
  for e in candidates:
   a,b=e['start'],e['arrival_index']
   if any(max(a,p['start'])<=min(b,p['arrival_index'])for p in old+added):continue
   if np.any(body[a:b+1]==0)or not np.any((m[a:b+1]<0)&(supp[a:b+1]==1)):continue
   near=np.arange(max(a,b-1),min(len(ts),b+2));near=near[np.isfinite(d[near])&(d[near]<=self.projected_limit)]
   if len(near)<2:continue
   added.append(dict(e,source='fit_trained_hand_edge_motion_plus_observed_body_arrival'))
  q['learned_cycle_support']=dict(added=added,supplemented_frames=np.flatnonzero((m<0)&(supp>=0)).tolist(),base_motion_preserved=True)
  if added:
   events=sorted(old+added,key=lambda e:e['arrival_index']);q['events']=events;context=[dict(start=int(r[0]),end=int(r[-1]))for r in contiguous(np.flatnonzero(s>0))]+q.get('additional_body_events',[])+q.get('additional_region_events',[])+q.get('observed_endpoint_events',[])+[e['event']for e in q.get('episode_support',[])]+added
   for side,hand in [('left','L'),('right','R')]:
    for suffix,kind,states in [('shape','hand_form',np.array(q['hand_form_evidence'][hand]['states'])),('orientation','projected_plane',np.array(q['plane_states'][hand]))]:
     prior=next(c for c in q['criteria']if c['id']==side+'_'+suffix);new=retain_uncertainty(prior,shape_or_plane(prior['id'],hand,prior['expected'],states,events,ts,kind,context),ts);q['criteria']=[new if c['id']==new['id']else c for c in q['criteria']]
   target=self.target_region.criterion(ch,events);spans=list(target['source_spans']);unconfirmed=[]
   for a,b in target['unconfirmed_events']:
    ii=np.arange(a,b+1);ok=ii[np.isfinite(d[ii])&(d[ii]<=self.projected_limit)&(body[ii]!=0)]
    if len(ok)>=2:spans.append([int(ok[0]),int(ok[-1])])
    else:unconfirmed.append([a,b])
   target.update(status='supported'if spans and not unconfirmed else'unknown',source_spans=spans,unconfirmed_events=unconfirmed)
   prior=next(c for c in q['criteria']if c['id']=='relative_target');target=prior if prior['status']=='difference'else target;q['criteria']=[target if c['id']==target['id']else c for c in q['criteria']]
   count=next(c for c in q['criteria']if c['id']=='repeat_order');count.update(status='supported'if len(events)>=count['expected_count']else'unknown',observed_count=len(events),source_spans=[[e['start'],e['end']]for e in events]);q['movement_evidence'].update(confirmed_arrivals=events,confirmed_arrival_count=len(events),bridge_version='distinct_episodes_and_fit_learned_edge_motion_v1')
  q['seconds']=time.perf_counter()-started;return q

def main():
 from verify_scope import verify
 torch.set_num_threads(2);net=LearnedCycleLesson();out=M/'learned_checks';out.mkdir(exist_ok=True);rows=[]
 for r in all_sources():
  a=baseline(r['id']);b=net.analyze(dict(np.load(r['raw'])));row=dict(id=r['id'],before={c['id']:c['status']for c in a['criteria']},after={c['id']:c['status']for c in b['criteria']},events_before=len(a['events']),events_after=len(b['events']),**verify(a,b));rows.append(row);write(out/(r['id']+'.json'),dict(before=a,after=b))
  if q:=b['learned_cycle_support']['added']:print(r['id'],q,row,flush=True)
 write(out/'RESULT.json',dict(rows=rows,failed=sum(not r['passed']for r in rows),full_before=sum(all(v=='supported'for v in r['before'].values())for r in rows),full_after=sum(all(v=='supported'for v in r['after'].values())for r in rows)));print('LEARNED',read(out/'RESULT.json')['failed'],flush=True)
if __name__=='__main__':main()
