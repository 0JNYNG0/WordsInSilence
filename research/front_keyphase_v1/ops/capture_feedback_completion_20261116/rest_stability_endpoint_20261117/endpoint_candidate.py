"""Candidate only until public incomplete-action and normal gates pass."""
from base import *
sys.path.insert(0,str(P))
from short_arrival import ShortArrivalLesson,torch
from target_region import region_distance
from core import approaches

class ObservedFeatures:
 def __init__(self,inner):self.inner=inner;self.channel=None
 def __call__(self,ch):self.channel=ch;return self.inner(ch)

class EndpointRelation:
 def __init__(self,inner,inspection,feature,limit):self.inner=inner;self.inspection=inspection;self.feature=feature;self.limit=limit;self.added=[]
 def __call__(self,f,quality,ts):
  original,_=self.inner(f,quality,ts);g=self.inspection(f);ch=self.feature.channel;d=region_distance(ch)
  visible=quality&np.isfinite(d);supported=visible&(g==2)&(d<=self.limit)&(original!=0)
  # Two consecutive directly observed near-hand samples; do not infer hidden
  # contact, fill missing joints or use an inspection proposal in isolation.
  paired=supported & (np.r_[False,supported[:-1]]|np.r_[supported[1:],False])
  s=original.copy();s[paired]=2;self.added=np.flatnonzero(paired&(original!=2)).tolist()
  return s,approaches(s,ts)

class EndpointLesson(ShortArrivalLesson):
 def __init__(self):
  super().__init__();self.feature=ObservedFeatures(self.feature);self.relation=EndpointRelation(self.relation,self.inspection,self.feature,self.target_region.limit)
 def analyze(self,raw,lesson=None):
  q=super().analyze(raw,lesson);q['endpoint_candidate_support']=self.relation.added;return q
