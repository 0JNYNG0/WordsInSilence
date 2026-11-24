"""Independent observation / arm-location / assignment diagnostic axes.

No word, signer, model output, future frame or screen-half handedness rule.
Input coordinates/masks/assignment are read-only. No recovered Hand21 output.
Frozen legacy diagnostic is retained as an audit field, never overwritten.
"""
import sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'word100_hand_visibility_20261017'))
from visibility import HandVisibility,PARAMS,PALM

SCHEMA='separated_observation_arm_assignment_v1'
ARM_STATES=('unknown','wrist_interior','wrist_boundary_or_outside')
OBS_STATES=('none','partial_points','palm5_observed','all21_observed')
ARM_REASONS=('insufficient_arm_evidence','arm_geometry_uncertain','edge_relation_ambiguous','interior_pose_wrist','pose_wrist_near_or_beyond_edge_with_outward_forearm')

def arm_location(xy,usable,conf,width,height,side):
 def inside(p,margin=0.):return bool(np.isfinite(p).all() and margin<=p[0]<=width-margin and margin<=p[1]<=height-margin)
 def trusted(i,require_inside=True):
  return bool(np.isfinite(xy[i]).all() and np.isfinite(conf[i]) and conf[i]>=PARAMS['pose_confidence'] and (not require_inside or (usable[i] and inside(xy[i]))))
 sh,el,wr=((11,13,15),(12,14,16))[side]
 row={'state':'unknown','reason':'insufficient_arm_evidence','source':'current_pose_arm_estimate',
      'side':side,'wrist_xy_px':xy[wr].tolist() if np.isfinite(xy[wr]).all() else None,
      'wrist_pose_confidence':float(conf[wr]) if np.isfinite(conf[wr]) else None,
      'wrist_inside_frame':inside(xy[wr]) if np.isfinite(xy[wr]).all() else None,
      'uses_outside_frame_estimate':bool(np.isfinite(xy[wr]).all() and not inside(xy[wr])),
      'whole_hand_visibility_proven':False,'coordinate_repair':False,'evidence':{}}
 if not (trusted(11) and trusted(12) and trusted(sh) and trusted(el) and trusted(wr,False)):return row
 scale=float(np.linalg.norm(xy[11]-xy[12]))
 if scale<=1e-6:return row
 upper=float(np.linalg.norm(xy[sh]-xy[el])/scale);lower=float(np.linalg.norm(xy[el]-xy[wr])/scale)
 row['evidence'].update(upper_arm_shoulders=upper,forearm_shoulders=lower,
  minimum_arm_pose_confidence=float(min(conf[11],conf[12],conf[sh],conf[el],conf[wr])))
 if not all(PARAMS['arm_segment_min_shoulders']<=x<=PARAMS['arm_segment_max_shoulders'] for x in (upper,lower)):
  row['reason']='arm_geometry_uncertain';return row
 p=xy[wr];distances=np.array([p[0],width-p[0],p[1],height-p[1]])/scale;edge=int(np.argmin(distances))
 normals=np.array([[-1.,0.],[1.,0.],[0.,-1.],[0.,1.]])
 outward=float(np.dot(p-xy[el],normals[edge])/scale)
 row['evidence'].update(nearest_edge=('left','right','top','bottom')[edge],edge_distance_shoulders=float(distances[edge]),outward_forearm_shoulders=outward)
 if distances[edge]<=PARAMS['boundary_shoulders'] and outward>PARAMS['boundary_shoulders']:
  row.update(state='wrist_boundary_or_outside',reason='pose_wrist_near_or_beyond_edge_with_outward_forearm')
 elif inside(p,PARAMS['boundary_shoulders']*scale):row.update(state='wrist_interior',reason='interior_pose_wrist')
 else:row['reason']='edge_relation_ambiguous'
 return row

class EvidenceAxes:
 def __init__(self):self.legacy=HandVisibility()
 def step(self,xy,usable,pose_conf,timestamp_us,width,height,assigned=(False,False),unassigned_raw=False,assignment_evidence_available=True):
  xy=np.asarray(xy);usable=np.asarray(usable,dtype=bool);conf=np.asarray(pose_conf)
  if xy.shape!=(75,2) or usable.shape!=(75,) or conf.shape!=(33,):raise ValueError('Expected 75-point/33-pose evidence')
  if width<=0 or height<=0:raise ValueError('Positive image dimensions required')
  if len(assigned)!=2:raise ValueError('Two assignment slots required')
  if not assignment_evidence_available and unassigned_raw:raise ValueError('Unavailable assignment evidence must not contain raw candidate flags')
  old=self.legacy.step(xy,usable,conf,timestamp_us,width,height,assigned=assigned,unassigned_raw=unassigned_raw)
  visible=usable&np.isfinite(xy).all(1)&(xy[:,0]>=0)&(xy[:,0]<=width)&(xy[:,1]>=0)&(xy[:,1]<=height)
  result=[]
  for side in (0,1):
   ids=np.arange(33+21*side,54+21*side);count=int(visible[ids].sum());palm=bool(visible[33+21*side+PALM].all());all21=count==21
   state='all21_observed' if all21 else 'palm5_observed' if palm else 'partial_points' if count else 'none'
   identity={'assigned_evidence_for_side':bool(assigned[side]),'visible_assigned_points_exist':count>0,
    'basis':'current_sensor_assignment' if assignment_evidence_available else 'visible_assigned_points_only',
    'unassigned_candidate_exists':bool(unassigned_raw) if assignment_evidence_available else None,
    'unassigned_flag_scope':'frame_level_not_side_specific','anatomical_side_certified':False}
   result.append({'schema':SCHEMA,'side':side,'observation':{'state':state,'usable_point_count':count,'palm5_observed':palm,'all21_observed':all21,'uses_estimated_hand_points':False},
    'arm_location':arm_location(xy,usable,conf,width,height,side),'assignment':identity,'legacy':old[side],
    'coordinate_repair':False,'eligible_for_automatic_exclusion':False,'training_feature':False})
  return result
