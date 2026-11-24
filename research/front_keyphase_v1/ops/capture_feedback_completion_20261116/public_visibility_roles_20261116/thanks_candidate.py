"""Candidate only: independent learned open observers can resolve unknown.

Never overturn an observed folded finger, a known shape, or missing landmarks.
No propagation of the prior hand state through an occlusion.
"""
from pathlib import Path
import sys
D=Path(__file__).resolve().parent
H=D.parent/'public_shape_arrival_repair_20261116'
sys.path.insert(0,str(H))
from projection_guard import CombinedVisible,ProjectionGuard,np
from shape_consensus import combine

class ShapeRecovery(ProjectionGuard):
 def sequence(self,ch,hand,plane=None):
  old=super().sequence(ch,hand,plane)
  projected,prob=self.projection.sequence(ch['tcn280'],hand)
  repaired,_=self.repaired.sequence(ch['tcn280'],hand)
  proposal,_=combine(old['fingers'],projected,repaired)
  p=self.plane.sequence(ch,hand)if plane is None else plane
  allowed=(old['states']<0)&(old['before_projection_guard']<0)&(proposal==0)&(p==0)&old['available'].all(1)
  old['states'][allowed]=0
  old['projection_recovered']=allowed
  return old

class ThanksCandidate(CombinedVisible):
 def __init__(self):super().__init__();self.shape=ShapeRecovery();self.plane=self.shape.plane
