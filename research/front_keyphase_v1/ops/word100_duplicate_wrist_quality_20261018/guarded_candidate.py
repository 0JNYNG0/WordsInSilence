"""V2: current two-wrist body evidence vetoes single-hand assumptions.

Uses existing Pose usability, not relaxed confidence thresholds. This guard
only retains previous sides; it never assigns a side from Pose alone.
"""
import numpy as np
PARAMS={'pose_wrist_indices':[15,16],'require_existing_pose_usable':True,
        'require_finite_in_frame':True,'unknown_input_preserves_baseline':True}
def guard(baseline,proposal,pose_xy,pose_usable,width,height):
    before=list(map(int,baseline));proposed=list(map(int,proposal))
    if before==proposed:return {'sides':proposed,'veto':False,'reason':'no_new_suppression','visible_wrists':None}
    if pose_xy is None or pose_usable is None or width<=0 or height<=0:
        return {'sides':before,'veto':True,'reason':'missing_body_evidence','visible_wrists':None}
    q=np.asarray(pose_xy);m=np.asarray(pose_usable,bool);visible=[]
    for k in [15,16]:
        visible.append(bool(m[k] and np.isfinite(q[k]).all() and 0<=q[k,0]<width and 0<=q[k,1]<height))
    veto=all(visible)
    return {'sides':before if veto else proposed,'veto':veto,
            'reason':'two_current_body_wrists_protect_overlap' if veto else 'no_two_wrist_veto',
            'visible_wrists':visible}
