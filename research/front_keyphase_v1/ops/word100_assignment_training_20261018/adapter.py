"""Fixed common composition, with source geometry separate from named slots."""
from common import *
import copy,cv2
obsmod=module('_pq_observation',OBS/'candidate.py')
contour=module('_pq_contour',F/'candidate.py')
duplicate=module('_pq_duplicate',D/'candidate.py')
guard=module('_pq_guard',D/'guarded_candidate.py')
legacy=module('_pq_legacy_prep',O/'word100_roi_training_20261017/prep.py')

def add_duplicate(z,b,t,sides):
 n=int(z['raw_count'][t]);old=int(z['raw_count'][t-1]) if t else 0
 scale=float(np.linalg.norm(z['pose_xy_px'][t,11]-z['pose_xy_px'][t,12]))
 q=duplicate.propose(z['hand_xy_px'][t,:n],b['point_mask'][t,:n],sides,b['identity_side'][t,:n],scale,
  z['hand_xy_px'][t-1,:old] if t else None,b['point_mask'][t-1,:old] if t else None,
  int(z['timestamp_us'][t]-z['timestamp_us'][t-1]) if t else None)
 g=guard.guard(sides,q['sides'],z['pose_xy_px'][t],z['pose_usable'][t],int(z['width'][t]),int(z['height'][t]))
 return g['sides'],{'proposal':q['sides'],'guard':g['reason']}

def contour_eligible(z,b,t):
 n=int(z['raw_count'][t]);p=z['pose_xy_px'][t];pm=z['pose_usable'][t];w=int(z['width'][t]);h=int(z['height'][t])
 inside=lambda q:np.isfinite(q).all() and 0<=q[0]<w and 0<=q[1]<h
 allowed=all(pm[k] and inside(p[k]) for k in (11,12,13,14))
 return bool(allowed and any(b['identity_side'][t,i]<0 and b['sides'][t,i]>=0 and b['point_mask'][t,i,0] for i in range(n)))

def contour_step(image,z,b,t):
 n=int(z['raw_count'][t]);width=int(z['width'][t]);height=int(z['height'][t]);p=z['pose_xy_px'][t];pm=z['pose_usable'][t]
 assert image.shape[0]>=height and image.shape[1]>=width
 w=min(width,640);h=max(1,int(round(height*w/width)));sx=w/width;sy=h/height
 im=cv2.resize(np.ascontiguousarray(image[:height,:width]),(w,h),interpolation=cv2.INTER_AREA)
 inside=lambda q:np.isfinite(q).all() and 0<=q[0]<width and 0<=q[1]<height
 pix=lambda q:(min(w-1,int(round(q[0]*sx))),min(h-1,int(round(q[1]*sy))))
 allowed=all(pm[k] and inside(p[k]) for k in (11,12,13,14))
 elbows=[pix(p[k]) if allowed else None for k in (13,14)]
 wrists=[pix(z['hand_xy_px'][t,i,0]) if b['point_mask'][t,i,0] else None for i in range(n)]
 scale=float(np.linalg.norm(p[11]-p[12]))*sx if allowed else 0.
 out=contour.propose(im,elbows,wrists,scale,b['sides'][t,:n],b['identity_side'][t,:n])
 return out,im

def route(old,z,b,sides,box):
 """Keep original body; route only measured eligible Hand points then strict ROI."""
 out=copy.deepcopy(old);out.xy_px[:,33:]=np.nan;out.usable[:,33:]=False;out.observed[:,33:]=False;out.hand_score[:]=np.nan
 x0,y0,x1,y1=box;unused=0
 for t,n0 in enumerate(z['raw_count']):
  n=int(n0);assigned=[int(x) for x in sides[t,:n] if x>=0];assert len(assigned)==len(set(assigned)),('slot_collision',t)
  for i in range(n):
   points=z['hand_xy_px'][t,i];mask=b['point_mask'][t,i].copy()
   mask &= (points[:,0]>=x0)&(points[:,0]<x1)&(points[:,1]>=y0)&(points[:,1]<y1)
   s=int(sides[t,i])
   if s<0:unused+=int(mask.sum());continue
   start=33+21*s;out.xy_px[t,start:start+21][mask]=points[mask].astype(np.float32)
   out.observed[t,start:start+21]=mask;out.usable[t,start:start+21]=mask;out.hand_score[t,s]=z['raw_score'][t,i]
 out.validate();return out,unused
