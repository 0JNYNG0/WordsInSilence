"""Capture stop only. Never decides whether a word's actions were completed."""
import numpy as np
class ReturnToRest:
 def __init__(self):
  self.samples=[];self.rest=None;self.started=None;self.moved=False;self.since=None;self.last_ts=-1
 def geometry(self,xy,width,height):
  p=np.asarray(xy,float);ok=np.isfinite(p).all(1)&(p[:,0]>=0)&(p[:,0]<width)&(p[:,1]>=0)&(p[:,1]<height)
  if not ok[[5,6,7,8]].all():return None
  scale=np.linalg.norm(p[5]-p[6])
  if scale<20:return None
  arms=(p[[7,8]]-p[[5,6]])/scale
  high=[]
  for wrist,start,elbow in [(9,91,7),(10,112,8)]:
   ids=[wrist]+list(range(start,start+21));ids=[j for j in ids if ok[j]]
   high.append(bool(ids and np.median(p[ids,1])<p[elbow,1]+.20*scale))
  return arms,np.asarray(high)
 def calibrate(self,xy,width,height):
  g=self.geometry(xy,width,height)
  if g is not None and not g[1].any() and (g[0][:,1]>.12).all():self.samples.append(g[0])
 def begin(self,ts=0):
  self.rest=np.median(self.samples[-90:],axis=0) if len(self.samples)>=6 else None
  self.started=ts;self.moved=False;self.since=None;self.last_ts=-1
 def update(self,xy,width,height,ts):
  if ts<=self.last_ts:return False
  # A stale/missing observation cannot count as an uninterrupted rest interval.
  if self.last_ts>=0 and ts-self.last_ts>250000:self.since=None
  self.last_ts=ts;g=self.geometry(xy,width,height)
  if g is None or self.rest is None:self.since=None;return False
  arms,high=g;distance=np.linalg.norm(arms-self.rest,axis=1)
  if high.any() or (distance>.28).any():self.moved=True
  returned=(distance<.22).all() and (arms[:,1]>.12).all() and not high.any()
  if self.moved and ts-self.started>=1000000 and returned:
   if self.since is None:self.since=ts
   return ts-self.since>=750000
  self.since=None;return False
