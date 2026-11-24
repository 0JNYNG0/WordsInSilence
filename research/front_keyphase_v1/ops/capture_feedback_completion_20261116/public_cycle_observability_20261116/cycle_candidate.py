from cycle_common import *

def separate_episodes(ch,motion,body,old,limit):
 d=palm_diameter_distance(ch);ts=ch['timestamp_us'];valid=np.isfinite(d);near=[x for x in contiguous(np.flatnonzero(valid&(d<=limit)))if len(x)>=2]
 # Associate prior arrivals to measured near episodes, never by the complete
 # old event extent: that extent can incorrectly absorb the next repetition.
 claimed=set()
 for e in old:
  distances=[max(int(n[0])-e['arrival_index'],e['arrival_index']-int(n[-1]),0)for n in near]
  if distances and min(distances)<=3:claimed.add(int(np.argmin(distances)))
 added=[];diagnostic=[]
 for j,n in enumerate(near):
  if j in claimed:continue
  begin=int(n[0]);last=int(near[j-1][-1])+1 if j else 0
  far=contiguous(np.flatnonzero(valid[last:begin]&(d[last:begin]>limit)&(body[last:begin]!=0))+last)
  if not far or len(far[-1])<2:continue
  start=int(far[-1][0]);arrival=int(n[1]);end=int(n[-1]);bodyspan=body[start:arrival+1]
  if np.any(bodyspan==0)or np.count_nonzero(bodyspan>0)<2:continue
  # A newly discovered first cycle must start from the learned separated
  # posture. Without an earlier verified arrival, accidental proximity during
  # preparation cannot serve as evidence of a repeated interaction.
  preceding_old=[e for e in old if e['arrival_index']<start]
  if not preceding_old and np.count_nonzero(bodyspan==1)<2:continue
  if not valid[start:arrival+1].all()or np.any(np.diff(ts[start:arrival+1])>100000):continue
  if np.count_nonzero(motion[start:arrival+1]==1)<2:continue
  # End the added observed arrival at departure; long following rest is not
  # reclassified as the action just because projected hands remain close.
  for i in range(arrival+1,end+1):
   if body[i]==0 or motion[i]==2:end=i-1;break
  e=dict(start=start,arrival_index=arrival,end=max(arrival,end),source='observed_separate_near_episode_with_learned_approach');added.append(e);diagnostic.append(dict(event=e,near_run=[int(n[0]),int(n[-1])],distance=d[start:end+1].tolist()))
 return added,diagnostic,d

def retain_uncertainty(prior,current,ts):
 if prior['status']=='difference':return prior
 uncertain=prior.get('uncertain_spans',[])
 if not uncertain:return current
 blocked=np.zeros(len(ts),bool)
 for a,b in uncertain:blocked[a:b+1]=True
 kept=[]
 for a,b in current.get('difference_spans',[]):
  ids=np.arange(a,b+1);kept.extend([[int(r[0]),int(r[-1])]for r in contiguous(ids[~blocked[a:b+1]])if len(r)>=3 and ts[r[-1]]-ts[r[0]]>=100000])
 current.update(status='difference'if kept else'unknown',difference_spans=kept,uncertain_spans=uncertain,uncertainty_reason=prior['uncertainty_reason']);return current

class CycleLesson(TemporalVisibilityLesson):
 def __init__(self):super().__init__();self.projected_limit=read(M/'CALIBRATION.json')['limit']
 def analyze(self,raw,lesson=None):
  started=time.perf_counter();q=super().analyze(raw,lesson)
  # Existing fully observed success needs no fallback search. This only avoids
  # extra work; no missing action is synthesized from the requested count.
  if all(c['status']=='supported'for c in q['criteria']):q['episode_support']=[];q['seconds']=time.perf_counter()-started;return q
  ch=channels(raw);ts=ch['timestamp_us'];motion=self.motion.sequence(ch);body=np.array(q['body_relation_states']);old=copy.deepcopy(q['events']);added,diag,d=separate_episodes(ch,motion,body,old,self.projected_limit);events=sorted(old+added,key=lambda e:e['arrival_index']);q['episode_support']=diag
  if added:
   q['events']=events;relation=np.array(q['relation_states']);context=[dict(start=int(x[0]),end=int(x[-1]))for x in contiguous(np.flatnonzero(relation>0))]+q.get('additional_body_events',[])+q.get('additional_region_events',[])+q.get('observed_endpoint_events',[])+added
   for side,hand in [('left','L'),('right','R')]:
    for suffix,kind,states in [('shape','hand_form',np.array(q['hand_form_evidence'][hand]['states'])),('orientation','projected_plane',np.array(q['plane_states'][hand]))]:
     prior=next(c for c in q['criteria']if c['id']==side+'_'+suffix);new=shape_or_plane(prior['id'],hand,prior['expected'],states,events,ts,kind,context);new=retain_uncertainty(prior,new,ts);q['criteria']=[new if c['id']==new['id']else c for c in q['criteria']]
   count=next(c for c in q['criteria']if c['id']=='repeat_order');count.update(status='supported'if len(events)>=count['expected_count']else'unknown',observed_count=len(events),source_spans=[[e['start'],e['end']]for e in events]);q['movement_evidence'].update(confirmed_arrivals=events,confirmed_arrival_count=len(events),bridge_version='source_calibrated_distinct_near_episodes_v1')
  prior=next(c for c in q['criteria']if c['id']=='relative_target');target=self.target_region.criterion(ch,events);spans=list(target['source_spans']);unconfirmed=[]
  for a,b in target['unconfirmed_events']:
   ids=np.arange(a,b+1);ok=ids[np.isfinite(d[ids])&(d[ids]<=self.projected_limit)&(body[ids]!=0)]
   if len(ok)>=2:spans.append([int(ok[0]),int(ok[-1])])
   else:unconfirmed.append([a,b])
  target.update(status='supported'if spans and not unconfirmed else'unknown',source_spans=spans,unconfirmed_events=unconfirmed,scale_basis='Original observed region OR FIT17-calibrated projected palm diameter; no physicalcontact')
  if prior['status']=='difference':target=prior
  q['criteria']=[target if c['id']==target['id']else c for c in q['criteria']];q['seconds']=time.perf_counter()-started;return q

def check(factory=CycleLesson,folder='comparison'):
 torch.set_num_threads(2);net=factory();out=M/folder;out.mkdir(exist_ok=True);rows=[]
 for r in all_sources():
  a=baseline(r['id']);b=net.analyze(dict(np.load(r['raw'])));aa={c['id']:c['status']for c in a['criteria']};bb={c['id']:c['status']for c in b['criteria']};row=dict(id=r['id'],before=aa,after=bb,events_before=len(a['events']),events_after=len(b['events']),lost=[k for k,v in aa.items()if v=='supported'and bb[k]!='supported'],new_warning=[k for k,v in bb.items()if v=='difference'and aa[k]!='difference'],preserved=all(e in b['events']for e in a['events']));rows.append(row);write(out/(r['id']+'.json'),dict(before=a,after=b))
  if aa!=bb or len(a['events'])!=len(b['events']):print(row,flush=True)
 write(out/'RESULT.json',dict(rows=rows,regressions=sum(bool(r['lost']or r['new_warning']or not r['preserved'])for r in rows),full_before=sum(all(v=='supported'for v in r['before'].values())for r in rows),full_after=sum(all(v=='supported'for v in r['after'].values())for r in rows)));print('CHECK',read(out/'RESULT.json')['regressions'],flush=True)
if __name__=='__main__':check()
