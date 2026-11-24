"""Capture-only screen-edge observation handling; word/feedback independent."""
import numpy as np
from auto_end import ReturnToRest as PreviousRest

class ReturnToRest(PreviousRest):
 def __init__(self):
  super().__init__();self.edge_sides=[];self.trace=[]
 @property
 def ready(self):return len(self.samples)>=6
 def geometry(self,xy,width,height):
  g=super().geometry(xy,width,height);self.edge_sides=[]
  if g is None:return None
  arms,high=g;p=np.asarray(xy,float);scale=np.linalg.norm(p[5]-p[6])
  ok=np.isfinite(p).all(1)&(p[:,0]>=0)&(p[:,0]<width)&(p[:,1]>=0)&(p[:,1]<height)
  for side,(wrist,start,elbow) in enumerate([(9,91,7),(10,112,8)]):
   hand=p[start:start+21];finite=np.isfinite(hand).all(1)
   palm=start+np.array([0,5,9,13,17]);visible=np.flatnonzero(ok[start:start+21])+start
   # A cropped hand's few surviving edge points are not a raised full hand.
   # Require actual body wrist below elbow, no visible hand above elbow,
   # and palm truncation at the bottom. Unknown hands alone never trigger this.
   clipped=bool(finite.any() and (hand[finite,1]>=height).any() and not ok[palm].all())
   # RTMW also collapses an unseen hand to a tiny cluster just inside the
   # image edge. A full in-frame palm is never suppressed on height alone.
   collapsed=bool(ok[palm].all() and np.linalg.norm(p[start+9]-p[start])<.05*scale and np.linalg.norm(p[start+5]-p[start+17])<.05*scale and height-np.median(p[palm,1])<.05*scale)
   wrist_horizontal=bool(np.isfinite(p[wrist]).all() and 0<=p[wrist,0]<width)
   down=bool(wrist_horizontal and p[wrist,1]>p[elbow,1] and p[wrist,1]>=p[elbow,1]+abs(p[wrist,0]-p[elbow,0]))
   no_raised=bool(len(visible) and (p[visible,1]>=p[elbow,1]).all())
   if high[side] and (clipped or collapsed) and down and no_raised:
    high[side]=False;self.edge_sides.append(side)
  return arms,high
 def update(self,xy,width,height,ts):
  result=super().update(xy,width,height,ts)
  g=self.geometry(xy,width,height)
  reason='body_unobserved' if g is None else 'not_calibrated' if self.rest is None else 'hand_active' if g[1].any() else 'arms_not_at_rest' if (np.linalg.norm(g[0]-self.rest,axis=1)>=.22).any() else 'waiting_for_action' if not self.moved else 'rest_dwell'
  self.trace.append(dict(timestamp_us=int(ts),reason=reason,bottom_truncated_sides=self.edge_sides.copy(),rest_since_us=self.since,stop=bool(result)))
  return result
 def diagnostic(self):
  return dict(version='bottom_clipped_hand_rest_v1',rest_baseline=None if self.rest is None else self.rest.tolist(),calibration_samples=len(self.samples),frames=self.trace)
