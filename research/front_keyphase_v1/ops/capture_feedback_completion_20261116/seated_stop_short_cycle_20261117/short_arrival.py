"""Use an observed endpoint within the motion window, never invent a contact."""
from probe_public import *
from observed_gap import preserve_sparse_arrivals
from observed_gap import SparseMotion
from repair import linked_arrivals as legacy_arrivals

def linked_arrivals(static,motion,radius=3):
 events=[];separate=closer=0;armed=None;last_arrival=None;departure=[]
 for i,(s,m) in enumerate(zip(static,motion)):
  if s==0:separate=closer=0;armed=None;departure=[];continue
  if m==2:
   separate+=1;closer=0
   if separate>=2:armed=i-1;departure=[i-1,i]
   elif s==1:armed=i;departure=[i]
  elif m==1:
   closer+=1;separate=0
   if armed is None and last_arrival is None:armed=i
  else:closer=separate=0
  if closer>=2 and armed is not None:
   ends=[j for j in range(i,min(len(static),i+radius+1)) if static[j]==2]
   if ends and (last_arrival is None or armed>last_arrival):
    end=ends[-1]
    events.append(dict(start=int(armed),end=int(end),source='independent_motion_plus_observed_near',departure_indices=departure.copy()))
    last_arrival=end;armed=None;closer=0;departure=[]
 return events

class DualWindowMotion(SparseMotion):
 def sequence(self,ch,radius=3):
  regular=super().sequence(ch,radius)
  self.short=super().sequence(ch,radius=1)
  return regular

class ShortArrivalLesson(FrontalLesson):
 def __init__(self):
  super().__init__();self.motion=DualWindowMotion('mlp32')
 def movement(self,s,m,events,available,ts):
  original=legacy_arrivals(s,m)
  evidence=preserve_sparse_arrivals(s,m,original,available,ts)
  short=self.motion.short
  extra=preserve_sparse_arrivals(s,short,linked_arrivals(s,short,radius=1),available,ts)
  kept=evidence['confirmed_arrivals']
  for event in extra['confirmed_arrivals']:
   if any(event['start']<=e['end'] and event['end']>=e['start'] for e in kept):continue
   previous=[e['end']for e in kept if e['end']<event['start']]
   # Non-overlap alone cannot prove a second cycle: two window sizes may
   # locate separate pieces of the same approach. Require an independently
   # observed departure after the preceding arrival before adding a cycle.
   departure=event.get('departure_indices',[])
   if not departure or not all(available[j] for j in departure):continue
   if previous and min(departure)<=max(previous):continue
   kept.append(dict(event,source='short_window_motion_plus_observed_near'))
  kept.sort(key=lambda e:e['end'])
  evidence['confirmed_arrival_count']=len(kept)
  evidence['bridge_version']='observed_endpoint_dual_motion_window_v2'
  return evidence
