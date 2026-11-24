"""Disagreement means unknown, never an invented open hand or whole-sign pass."""
from repair import *
from train_shape import LocalShape,HandHead
from view_conditioned_shape import ViewConditionedShape

class ProjectionHead(LocalShape):
 def __init__(self):
  self.kind='temporal';self.models=[]
  for seed in[42,43,44]:
   ck=torch.load(HERE/'projection_head'/f'{seed}.pt',map_location='cpu',weights_only=False);net=HandHead('temporal');net.load_state_dict(ck['state']);net.eval();self.models.append((net,ck['mean'],ck['std']))

class ProjectionGuard(ViewConditionedShape):
 def __init__(self):super().__init__();self.projection=ProjectionHead();self.last_conflicts={}
 def sequence(self,ch,hand,plane=None):
  old=super().sequence(ch,hand,plane);coarse,prob=self.projection.sequence(ch['tcn280'],hand)
  # A majority can establish disagreement, not a replacement truth. Requiring
  # unanimity missed split ensemble decisions; a single outlier is insufficient.
  conflict=(old['states']==1)&((prob[:,:,0]>=.85).sum(0)>=2)
  old['before_projection_guard']=old['states'].copy();old['states'][conflict]=-1
  old['projection_conflict']=conflict
  self.last_conflicts[hand]=conflict.copy()
  # The same uncertain evidence cannot leak out as confident finger correction.
  old['fingers'][conflict]=-1
  return old

class CombinedVisible(RepairedVisible):
 def __init__(self):super().__init__();self.shape=ProjectionGuard();self.plane=self.shape.plane
 def analyze(self,raw,lesson=None):
  started=time.perf_counter();q=super().analyze(raw,lesson);active=np.array(q['relation_states'])>0;ts=np.array(q['timestamp_us']);q['shape_ambiguity']={}
  from visible_lesson import contiguous
  for hand,side in [('R','right'),('L','left')]:
   spans=[[int(r[0]),int(r[-1])]for r in contiguous(np.flatnonzero(self.shape.last_conflicts[hand]&active))if len(r)>=3]
   q['shape_ambiguity'][hand]=spans
   if spans:
    c=next(c for c in q['criteria']if c['id']==side+'_shape');c['uncertain_spans']=spans;c['uncertainty_reason']='projection_observers_disagree'
    if c['status']!='difference':c['status']='unknown'
  q['seconds']=time.perf_counter()-started
  return q
