"""Fixed-palm, fixed-length finger retargeting. No whole-body or source-accuracy claim."""
import numpy as np
import bpy
from mathutils import Matrix, Vector
from geometry_core import hand_basis

FINGERS=[('Thumb',1),('Index',5),('Middle',9),('Ring',13),('Pinky',17)]

def bone_name(rig,suffix):
    names=[b.name for b in rig.data.bones if b.name.split(':')[-1]==suffix]
    if len(names)!=1:raise ValueError('Unique anatomical bone required: '+suffix)
    return names[0]

def rest_hand(rig,side):
    """21-point proxy from rest bones, not a claim that mesh landmarks equal OpenPose anatomy."""
    p=np.zeros((21,3));p[0]=rig.data.bones[bone_name(rig,side+'Hand')].head_local
    for finger,start in FINGERS:
        for k in range(3):p[start+k]=rig.data.bones[bone_name(rig,side+'Hand'+finger+str(k+1))].head_local
        p[start+3]=rig.data.bones[bone_name(rig,side+'Hand'+finger+'3')].tail_local
    return p

def apply_hand(rig,side,source_xyz,target_palm_basis=None):
    """Discard source translation/palm orientation; preserve this rig's palm, lengths and offsets."""
    source=np.asarray(source_xyz,float)
    if source.shape!=(21,3)or not np.isfinite(source).all():raise ValueError('Finite21x3 required')
    rest=rest_hand(rig,side);basis_source=hand_basis(source);basis_rig=hand_basis(rest)if target_palm_basis is None else np.asarray(target_palm_basis,float)
    rotation=basis_rig@basis_source.T
    metrics=[]
    for finger,start in FINGERS:
        for k in range(3):
            name=bone_name(rig,side+'Hand'+finger+str(k+1));pb=rig.pose.bones[name]
            direction=rotation@(source[start+k+1]-source[start+k]);norm=np.linalg.norm(direction)
            if norm<1e-9:raise ValueError('Degenerate finger segment '+name)
            target=Vector(direction/norm)
            bpy.context.view_layer.update()
            matrix=pb.matrix.copy();current=(pb.tail-pb.head).normalized()
            q=current.rotation_difference(target)
            desired=(q@matrix.to_quaternion()).to_matrix().to_4x4()
            desired.translation=pb.head
            pb.matrix=desired
            bpy.context.view_layer.update()
            actual=(pb.tail-pb.head).normalized()
            metrics.append(dict(bone=name,target=list(target),actual=list(actual),
                  angle_error_deg=float(actual.angle(target))*180/np.pi,
                  length_error=abs((pb.tail-pb.head).length-rig.data.bones[name].length)))
    return metrics
