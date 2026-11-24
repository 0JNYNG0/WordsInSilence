"""Unit-free body-relative mapping and fixed-length arm geometry; no sign-specific rules."""
import numpy as np


def unit(x):
    x=np.asarray(x,float);n=np.linalg.norm(x)
    if not np.isfinite(x).all()or n<1e-9:raise ValueError('Insufficient nondegenerate direction')
    return x/n


def body_frame(body25):
    """Requires true same-frame3D upper-body landmarks, not body2D with hand3D."""
    p=np.asarray(body25,float)
    if p.shape!=(25,3):raise ValueError('BODY25 XYZ required')
    origin=(p[2]+p[5])/2
    left=unit(p[5]-p[2])
    up0=p[1]-p[8]
    up=unit(up0-left*np.dot(up0,left))
    forward=unit(np.cross(left,up))
    basis=np.column_stack([left,up,forward])
    width=float(np.linalg.norm(p[5]-p[2]))
    return origin,basis,width


def map_to_rig(points,source_body,rig_origin,rig_basis,rig_shoulder_width):
    origin,basis,width=body_frame(source_body)
    local=(np.asarray(points,float)-origin)@basis/width
    return local@np.asarray(rig_basis,float).T*rig_shoulder_width+rig_origin


def solve_arm(shoulder,wrist_target,elbow_hint,upper_length,forearm_length):
    """Analytic two-bone IK; report unreachable targets instead of silent stretching."""
    shoulder=np.asarray(shoulder,float);target=np.asarray(wrist_target,float);hint=np.asarray(elbow_hint,float)
    direction=unit(target-shoulder);requested=float(np.linalg.norm(target-shoulder))
    if upper_length<=0 or forearm_length<=0:raise ValueError('Positive rig lengths required')
    minimum=abs(upper_length-forearm_length)+1e-7
    maximum=upper_length+forearm_length-1e-7
    distance=float(np.clip(requested,minimum,maximum))
    pole=hint-shoulder-direction*np.dot(hint-shoulder,direction)
    pole=unit(pole)
    a=(upper_length**2-forearm_length**2+distance**2)/(2*distance)
    height=np.sqrt(max(0.,upper_length**2-a*a))
    elbow=shoulder+direction*a+pole*height
    wrist=shoulder+direction*distance
    return dict(elbow=elbow,wrist=wrist,target_reach_error=float(np.linalg.norm(wrist-target)),
                requested_distance=requested,used_distance=distance,
                unreachable=bool(requested<minimum or requested>maximum))
