"""Pause, never credit, brief weak-hand fluctuations after calibrated arm return."""
from base import *
sys.path[:0]=[str(P),str(O/'word100_action_feedback_demo_20261108')]
from return_rest import ReturnToRest as PreviousRest

class StableRest(PreviousRest):
 def __init__(self):
  super().__init__();self.clear_us=0;self.weak_us=0;self.weak_start=None;self.previous_clear=False;self.clear_frames=0
 def reset_rest(self):
  self.since=None;self.clear_us=0;self.weak_us=0;self.weak_start=None;self.previous_clear=False;self.clear_frames=0
 def begin(self,ts=0):super().begin(ts);self.reset_rest()
 def weak_lowered_hand(self,xy,high,width,height):
  p=np.asarray(xy,float)
  for i,(w,e,start)in enumerate([(9,7,91),(10,8,112)]):
   if not high[i]:continue
   if not np.isfinite(p[[w,e]]).all():return False
   down=p[w,1]>p[e,1] and p[w,1]-p[e,1]>=abs(p[w,0]-p[e,0])
   hand=p[start:start+21];ok=np.isfinite(hand).all(1)&(hand[:,0]>=0)&(hand[:,0]<width)&(hand[:,1]>=0)&(hand[:,1]<height)
   if not down or not ok.any() or np.min(hand[ok,1])<p[e,1]:return False
  return True
 def update(self,xy,width,height,ts):
  if ts<=self.last_ts:return False
  dt=0 if self.last_ts<0 else ts-self.last_ts
  if dt>250000:self.reset_rest()
  self.last_ts=ts;g=self.geometry(xy,width,height);stop=False;reason='body_unobserved'
  if g is None or self.rest is None:
   self.reset_rest();reason='body_unobserved' if g is None else 'not_calibrated'
  else:
   arms,high=g;distance=np.linalg.norm(arms-self.rest,axis=1)
   if high.any()or(distance>.28).any():self.moved=True
   at_rest=(distance<.22).all()and(arms[:,1]>.12).all()
   eligible=self.moved and ts-self.started>=1000000
   if not at_rest or not eligible:
    self.reset_rest();reason='arms_not_at_rest'if not at_rest else'waiting_for_action'
   elif high.any():
    if self.since is not None and self.weak_lowered_hand(xy,high,width,height):
     if self.weak_start is None:self.weak_start=ts-dt
     self.weak_us+=dt;self.previous_clear=False;self.clear_frames=0
     if ts-self.weak_start>120000 or self.weak_us>200000:self.reset_rest();reason='weak_hand_too_long'
     else:reason='weak_hand_pause'
    else:self.reset_rest();reason='hand_active'
   else:
    # A paused/uncertain interval supplies no rest time, even if it later clears.
    if self.weak_start is not None and ts-self.weak_start>120000:self.reset_rest()
    if self.since is None:self.since=ts
    if self.previous_clear:self.clear_us+=dt
    self.previous_clear=True;self.weak_start=None;self.clear_frames+=1;reason='rest_dwell'
    stop=self.clear_us>=750000 and self.clear_frames>=3
  self.trace.append(dict(timestamp_us=int(ts),reason=reason,bottom_truncated_sides=self.edge_sides.copy(),rest_since_us=self.since,observed_rest_us=int(self.clear_us),weak_us=int(self.weak_us),stop=bool(stop)))
  return bool(stop)
 def diagnostic(self):
  q=super().diagnostic();q['version']='bounded_weak_rest_pause_v2';return q
