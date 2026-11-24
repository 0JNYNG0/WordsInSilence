"""V2 adds mutual one-to-one fingertip identity, to distinguish nearby contacts.
The V1 result remains immutable. Distances/counts inherited, no retuned threshold.
"""
import numpy as np
from candidate import inspect_pair as inspect_v1
def inspect_pair(a,b,va,vb,shoulder_px):
    result=inspect_v1(a,b,va,vb,shoulder_px)
    result['v1_hold']=result['hold_new']
    if not result['hold_new']:return result
    tips=np.array([4,8,12,16,20]);a=np.asarray(a,float);b=np.asarray(b,float)
    if not (np.asarray(va)[tips].all() and np.asarray(vb)[tips].all() and np.isfinite(a[tips]).all() and np.isfinite(b[tips]).all()):
        result.update(hold_new=False,status='duplicate_identity_evidence_incomplete')
        return result
    distances=np.linalg.norm(a[tips,None,:]-b[None,tips,:],axis=-1)
    mutual=[]
    for f in range(5):
        others=[x for x in range(5) if x!=f]
        # Strict minimum: ties never establish a unique identity.
        unique=bool(distances[f,f]<min(distances[f,others]) and distances[f,f]<min(distances[others,f]))
        mutual.append(bool(unique and result['finger_tests'][f]['agree']))
    result['mutual_tip_matches']=mutual
    result['mutual_tip_distances_px']=distances.tolist()
    if sum(mutual)<4:
        result.update(hold_new=False,status='nearby_fingers_not_unique_same_hand_evidence')
    else:
        result.update(hold_new=True,status='same_image_fingers_identity_unresolved')
    return result

