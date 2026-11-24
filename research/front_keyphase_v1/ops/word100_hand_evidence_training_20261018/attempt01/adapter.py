"""30 categorical evidence channels. Never mutates existing observations."""
from common import *
import numpy as np
sys.path.insert(0,str(O/'word100_hand_visibility_20261017'))
sys.path.insert(0,str(O/'word100_hand_evidence_split_20261017'))
sys.path.insert(0,str(CURRENT))
from visibility import HandVisibility
from separate import LOCATION
from spatial import extend_evidence
POSITION=('observed_points','partial_points','wrist_interior','wrist_boundary_or_outside','assignment_unresolved','unknown')
SLICES=((0,6),(6,9),(9,12),(12,15),(15,21),(21,24),(24,27),(27,30))

def frame_inputs(obs,raw,assignment,i,view,w,h):
 xy=obs['xy_px'][i].copy();mask=obs['usable'][i].copy();conf=raw['pose_conf'][i].copy()
 if view=='masked':
  # Discard full-frame-only evidence before applying any quality rule.
  allowed=np.isfinite(xy).all(1)&(xy[:,0]>=0)&(xy[:,0]<=w)&(xy[:,1]>=0)&(xy[:,1]<=h)&mask
  xy[~allowed]=np.nan;mask &= allowed;conf[~allowed[:33]]=0
  # Derive assigned evidence ONLY from allowed assigned observations.
  assigned=tuple(bool(mask[33+21*s:54+21*s].any()) for s in (0,1))
  # Full-frame raw candidates/assignment may depend on hidden hands. They
  # are not admissible for this synthetic masked view. Absence is not
  # evidence of resolved identity: insufficient observations remain unknown.
  unassigned=False
 else:
  assert view=='sensor'
  sides=assignment['sides'][i,:int(raw['raw_count'][i])].copy()
  assigned=tuple(bool(s in sides) for s in (0,1));unassigned=bool((sides<0).any())
 return xy,mask,conf,assigned,unassigned

def step(location,args,t,w,h):
 xy,mask,conf,assigned,unassigned=args
 baseline=location.step(xy,mask,conf,int(t),w,h,assigned=assigned,unassigned_raw=unassigned)
 prev=[]
 for b in baseline:
  prev.append({'side':b['side'],'position':{'state':LOCATION[b['state']],'evidence':b['evidence']},'shape':{'state':'usable_observed_points' if b['state']=='observed' else 'partial_observation' if b['state']=='partial_observation' else 'unobserved'},'overlap':{'proximity_hint':b['state']=='overlap_likely'}})
 rows=extend_evidence(prev,xy,mask,conf,w,h)
 values=np.zeros(30,np.float32);states=[]
 for s,r in enumerate(rows):
  p=POSITION.index(r['position']['state']);e=r['current_extent']['status'];o=r['current_overlap']['status'];ev=r['position']['evidence']
  ei=1 if e=='estimated_extent_inside_frame' else 2 if e=='estimated_extent_crosses_frame' else 0
  oi=1 if o=='no_current_2d_intersection' else 2 if o=='current_2d_intersection_suspected' else 0
  pi=2 if r['overlap']['proximity_hint'] else 1 if 'other_palm_distance_shoulders' in ev and 'history_distance_shoulders' in ev else 0
  for offset,k in ((0,p),(6,ei),(9,oi),(12,pi)):values[15*s+offset+k]=1
  states.append([p,ei,oi,pi])
 return values,states

def derive(obs,raw,ass,view,w,h,indices):
 location=HandVisibility();values=[];states=[]
 for i in indices:
  a,b=step(location,frame_inputs(obs,raw,ass,int(i),view,w,h),obs['timestamp_us'][i],w,h);values.append(a);states.append(b)
 return np.asarray(values,np.float32),np.asarray(states,np.int8)

def validate_extra(e):
 assert e.ndim==2 and e.shape[1]==30 and e.dtype==np.float32 and np.isfinite(e).all()
 assert np.logical_or(e==0,e==1).all()
 for a,b in SLICES:assert np.all(e[:,a:b].sum(1)==1)
