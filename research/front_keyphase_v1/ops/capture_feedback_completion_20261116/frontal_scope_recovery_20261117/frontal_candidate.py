"""Frontal-source composition; no side-view learned observation heads.

The surface head labels an open/curved palm, not every finger as fully extended.
Exact finger states and unavailable joints remain separate observations.
"""
from probe_front import *
sys.path.insert(0,str(G/'apple_v1'))
from train_surface import Surface
from view_conditioned_shape import ViewConditionedShape
from front_geometry import FrontGeometry
from visible_lesson import adapter,contiguous,shape_or_plane

class FrontalShape(ViewConditionedShape):
 def __init__(self):
  super().__init__();self.surface=Surface();self.observations={}
 def sequence(self,ch,hand,plane=None):
  q=super().sequence(ch,hand,plane);p=self.plane.sequence(ch,hand)if plane is None else plane
  surface=self.surface.sequence(ch['tcn280'],hand)
  # Fill unresolved coarse palm evidence only. Never fill missing joints,
  # replace a known different form, or turn these into finger-extension labels.
  eligible=(q['states']<0)&(surface==0)&(p==0)&q['available'].all(1)
  q['states']=q['states'].copy();q['states'][eligible]=0
  q['surface_states']=surface;q['surface_support']=eligible
  self.observations[hand]=q
  return q

class FrontalLesson(RepairedVisible):
 def __init__(self):
  super().__init__();self.shape=FrontalShape();self.plane=self.shape.plane;self.inspection=FrontGeometry()
 def analyze(self,raw,lesson=None):
  started=time.perf_counter();q=super().analyze(raw,lesson);g=self.inspection(self.last_features);q['inspection_relation_states']=g.tolist()
  anchors=np.flatnonzero((np.array(q['relation_states'])>0)|(g>0))
  if len(anchors)>=2:
   a,b=int(anchors[0]),int(anchors[-1]);q['shape_inspection_envelope']=[a,b]
   for hand,side in [('R','right'),('L','left')]:
    prior=next(c for c in q['criteria']if c['id']==side+'_shape');obs=self.shape.observations[hand];available=obs['available'].all(1)
    if prior['expected']!=0:continue
    # Two separately trained form observers must agree. This can produce an
    # error inspection, never a new successful movement or hidden-hand label.
    agreed=available&(obs['states']==2)&(obs['surface_states']==2)
    guard=np.where(agreed,2,-1);probe=shape_or_plane(prior['id'],hand,0,guard,[],np.array(q['timestamp_us']),'shape',[dict(start=a,end=b)])
    if probe['status']=='difference':
     prior.update(status='difference',difference_spans=sorted(prior['difference_spans']+probe['difference_spans']),inspection='Frontal body proposal + two agreeing observed hand-form heads; no completed event added')
    elif prior['status']=='unknown':
     missing=[r for r in contiguous(np.flatnonzero(~available[a:b+1])+a)if len(r)>=3]
     if missing:prior['uncertain_spans']=[[int(r[0]),int(r[-1])]for r in missing]
  q['seconds']=time.perf_counter()-started;return q

