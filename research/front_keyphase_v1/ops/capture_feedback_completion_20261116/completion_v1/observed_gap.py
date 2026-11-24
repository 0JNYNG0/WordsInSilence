"""Small-gap motion evidence from real observations, never interpolated fingers."""
from visible_lesson import *
from motion_model import feature as motion_feature, PALMS

def admissible_gap(available,ts,a,b):
 ids=np.arange(a,b+1);missing=ids[~available[a:b+1]]
 return len(missing)==1 and a<missing[0]<b and ts[missing[0]+1]-ts[missing[0]-1]<=100000

class SparseMotion(Motion):
 def sequence(self,ch,radius=3):
  states=super().sequence(ch,radius);ids=PALMS[0]+PALMS[1]
  available=ch['point_available'][:,ids].all(1)&np.isfinite(ch['body_relative_xy'][:,ids]).all((1,2));ts=ch['timestamp_us']
  for i in np.flatnonzero(states<0):
   a,b=i-radius,i+radius
   if a<0 or b>=len(ts)or not admissible_gap(available,ts,a,b):continue
   keep=np.arange(a,b+1)[available[a:b+1]]
   short={k:v[keep]if isinstance(v,np.ndarray)and v.ndim and len(v)==len(ts)else v for k,v in ch.items()}
   states[i]=self.predict(motion_feature(short,0,len(keep)-1))
  return states

def preserve_sparse_arrivals(static,motion,arrivals,available,ts):
 evidence=movement_evidence(static,motion,arrivals,available)
 old={(e['start'],e['end'])for e in evidence['confirmed_arrivals']}
 for e in arrivals:
  a,b=e['start'],e['end']
  if (a,b)in old or static[b]!=2 or not admissible_gap(available,ts,a,b):continue
  # Both endpoints and all counted supporting states have actual measurements.
  observed=np.arange(a,b+1)[available[a:b+1]]
  evidence['confirmed_arrivals'].append(dict(e,endpoint_status='observed_near',physical_contact_verified=False,
   observation_gap=True,observed_indices=observed.tolist(),missing_indices=np.arange(a,b+1)[~available[a:b+1]].tolist()))
 evidence['confirmed_arrivals'].sort(key=lambda e:e['end']);evidence['confirmed_arrival_count']=len(evidence['confirmed_arrivals'])
 return evidence

class SparseVisible(VisibleLesson):
 def __init__(self):super().__init__();self.motion=SparseMotion('mlp32')
 def movement(self,s,m,events,available,ts):return preserve_sparse_arrivals(s,m,events,available,ts)
