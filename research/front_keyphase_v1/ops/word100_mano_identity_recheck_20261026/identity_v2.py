"""Conflict-quarantined anchors; wrist-qualified rivals only. V1 preserved."""
import identity as original
from identity import PARAMS,flow,get_tracks,current_arm,TIPS,choose
def build_anchors(xy,valid,pose,pv,duplicate,suppressed=None):
    anchors=original.build_anchors(xy,valid,pose,pv,duplicate)
    if suppressed is not None:anchors &= ~suppressed
    return anchors
def assess(hand,pose,pv,scale,tracks):
    result=original.assess(hand,pose,pv,scale,tracks)
    result['v1_supported_side']=result['supported_side']
    eligible=[]
    for s in (0,1):
        q=result['temporal'][s];other=result['temporal'][1-s]
        # A rival whose wrist evidence contradicts its finger-position hypothesis
        # cannot veto a distinct hand merely because the fingers nearly touch.
        unique=q['qualified'] and (not other['qualified'] or q['cost']<PARAMS['ambiguity_ratio']*other['cost'])
        if unique and result['current_arm']==s:eligible.append(s)
    side=eligible[0] if len(eligible)==1 else -1
    result.update(supported_side=side,status='current_arm_and_two_sided_image_support' if side>=0 else 'identity_not_confirmed')
    return result
