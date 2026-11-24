"""Goal-linked evidence assembler for visible lesson requirements.

Neural observations locate relation/movement and classify shape/projected plane.
The guide supplies required sides, forms and repetitions. This module combines
the facts with their native source spans; no reference-video loop or DTW fill.
"""
from plane_model import *
from basic_runtime import RelationObserver
from core import Features
sys.path[:0]=[str(B/'acquisition_motion_followup'),str(V.parent),str(O/'word100_rtmw_shared_input_20261108')]
from motion_model import Observer as Motion
from check_motion_bridge import bridge
from motion_evidence import movement_evidence
import adapter
from view_conditioned_shape import ViewConditionedShape
from target_region import TargetRegion

def contiguous(ids):
 ids=np.asarray(ids,dtype=int)
 return [r for r in np.split(ids,np.flatnonzero(np.diff(ids)>1)+1) if len(r)] if len(ids)else []

def stable_spans(s,target,ts,a,b):
 spans=[]
 for r in contiguous(np.flatnonzero(s[a:b+1]==target)+a):
  if len(r)>=3:spans.append([int(r[0]),int(r[-1])])
 return spans

def observed_action_extent(events,static,motion,available):
 """Keep the observed terminal hold, stopping at departure or missing evidence.

 A near-state arrival is an anchor, not necessarily the end of the action.
 Never extend across the next event, across absence, or via a DTW path.
 """
 out=[]
 for j,e in enumerate(events):
  end=e['end'];limit=events[j+1]['start']if j+1<len(events)else len(static)
  for i in range(end+1,limit):
   if not available[i]or static[i]==0 or motion[i]==2:break
   if static[i]!=2 and motion[i]!=1:break
   end=i
  out.append(dict(e,arrival_index=e['end'],end=end))
 return out

def shape_or_plane(criterion_id,hand,expected,states,events,ts,kind,context=None):
 result=dict(id=criterion_id,hand=hand,status='unknown',source_spans=[],difference_spans=[],kind=kind,expected=expected)
 # Inspection context includes observed static and transition frames, not only
 # successful arrivals. A failed/missing movement cannot erase a shape error.
 inspect=context if context is not None else events
 for event in inspect:
  a,b=event['start'],event['end']
  wrong=contiguous(np.flatnonzero((states[a:b+1]>=0)&(states[a:b+1]!=expected))+a)
  result['difference_spans'] += [[int(r[0]),int(r[-1])] for r in wrong if len(r)>=3 and ts[r[-1]]-ts[r[0]]>=100000]
 if not events:
  if result['difference_spans']:result['status']='difference'
  return result
 # A declared change inside a relevant event remains a difference, regardless
 # of the resulting TCN label. Brief unresolved states remain explicitly unknown.
 for event in events:
  a,b=event['start'],event['end']
  support=stable_spans(states,expected,ts,a,b)
  if support:result['source_spans']+=support
  else:result.setdefault('unconfirmed_events',[]).append([a,b])
 if result['difference_spans']:result['status']='difference'
 elif result['source_spans'] and not result.get('unconfirmed_events'):result['status']='supported'
 return result

class VisibleLesson:
 def __init__(self):
  self.feature=Features();self.relation=RelationObserver();self.motion=Motion('mlp32');self.shape=ViewConditionedShape();self.plane=self.shape.plane;self.target_region=TargetRegion()
 def movement(self,s,m,events,available,ts):return movement_evidence(s,m,events,available)
 def analyze(self,raw,lesson=None):
  started=time.perf_counter()
  lesson=lesson or dict(target='WORD1290',label='감사',repeat=2,right_shape=0,left_shape=0,right_plane=0,left_plane=1,
    description='오른손을 펴서 세우고, 편 왼 손등 쪽으로 두 번 댄다.')
  ch=adapter.convert(raw['xy_px'],raw['score'],raw['timestamp_us'],raw['frame_id'],int(raw['source_width']),int(raw['source_height']))
  f,q,_=self.feature(ch);s,_=self.relation(f,q,ch['timestamp_us']);m=self.motion.sequence(ch);events=bridge(s,m)
  self.last_features=f
  palms=[91,96,100,104,108,112,117,121,125,129];available=ch['point_available'][:,palms].all(1)&np.isfinite(ch['body_relative_xy'][:,palms]).all((1,2))
  ev=self.movement(s,m,events,available,ch['timestamp_us']);events=observed_action_extent(ev['confirmed_arrivals'],s,m,available);ts=ch['timestamp_us'];criteria=[];shape={};plane={}
  context=[dict(start=int(r[0]),end=int(r[-1]))for r in contiguous(np.flatnonzero(s>0))]
  for h,side in [('R','right'),('L','left')]:
   plane[h]=self.plane.sequence(ch,h);shape[h]=self.shape.sequence(ch,h,plane[h])
   criteria.append(shape_or_plane(side+'_shape',h,lesson[side+'_shape'],shape[h]['states'],events,ts,'shape',context))
   criteria.append(shape_or_plane(side+'_orientation',h,lesson[side+'_plane'],plane[h],events,ts,'projected_plane',context))
  # Evidence is projected approach to the other hand. Keep the exact-surface
  # requirement visibly unqualified until its own source/control gate passes.
  criteria.append(self.target_region.criterion(ch,events))
  sufficient=len(events)>=lesson['repeat']
  criteria.append(dict(id='repeat_order',status='supported'if sufficient else'unknown',source_spans=[[e['start'],e['end']]for e in events],
      kind='movement_order',observed_count=len(events),expected_count=lesson['repeat'],partial_progress=ev['unconfirmed_progress_count']))
  return dict(target=lesson['target'],criteria=criteria,required_ids=[c['id']for c in criteria],
      events=events,movement_evidence=ev,shape_states={h:v['states'].tolist()for h,v in shape.items()},
      plane_states={h:v.tolist()for h,v in plane.items()},relation_states=s.tolist(),
      shape_fingers={h:v['fingers'].tolist()for h,v in shape.items()},timestamp_us=ts.tolist(),
      schema='observed_lesson_requirements_v1',qualified=False,whole_word_correct=None,overall_pass=False,
      seconds=time.perf_counter()-started,basis='Learned frozen100 context + shared observation heads; description-defined evidence composition',
      limitation='Projected plane differs from verified 3D palm-facing/contact. Unsupported requirement cannot be auto-passed.')

def main():
 torch.set_num_threads(2);sys.path.insert(0,str(V.parent));from diagnose_terminal import sources
 net=VisibleLesson();rows=[]
 for r in sources():
  out=net.analyze(dict(np.load(r['raw_path'])));write(V/f'visible_{r["person"]}.json',out)
  rows.append(dict(person=r['person'],criteria={c['id']:c['status']for c in out['criteria']},events=len(out['events']),seconds=out['seconds']))
 write(V/'VISIBLE_BASELINE.json',dict(rows=rows,whole_word_qualified=False));print(json.dumps(rows,ensure_ascii=False))
if __name__=='__main__':main()
