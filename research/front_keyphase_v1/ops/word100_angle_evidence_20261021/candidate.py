"""Read-only coordinates, local angle evidence; no identity or word inputs.

The status concerns corroborating detector evidence, not true pose correctness.
This module does not change model features or run a temporal alignment.
"""
import numpy as np

CHAINS=((0,1,2,3,4),(0,5,6,7,8),(0,9,10,11,12),
        (0,13,14,15,16),(0,17,18,19,20))
TRIPLES=tuple(tuple(ch[k:k+3]) for ch in CHAINS for k in range(3))
NOT_OBSERVED=0
DEGENERATE=1
UNASSESSED=2
UNRESOLVED=3
ASSESSED_NO_CONFLICT=4
LENGTH_EPS=1e-6  # Same pixel degeneracy guard as prior saved loss audit.

def angle_evidence(xy, usable, pair_available, angle_available, angle_unresolved):
    """All outputs are separate arrays. Every angle uses only its own flags.

    A genuinely uncertain shared bone may mark two angles in upstream evidence.
    This adapter never propagates a flag through shared coordinate deletion.
    """
    xy=np.asarray(xy,dtype=np.float64)
    usable=np.asarray(usable,bool)
    shape=usable.shape[:-1]+(15,)
    if xy.shape!=usable.shape+(2,):
        raise ValueError("xy and usable shape mismatch")
    if np.shape(pair_available)!=shape[:-1] or np.shape(angle_available)!=shape or np.shape(angle_unresolved)!=shape:
        raise ValueError("angle evidence shape mismatch")
    degrees=np.full(shape,np.nan)
    status=np.full(shape,NOT_OBSERVED,np.uint8)
    finite=np.isfinite(xy).all(-1)
    for k,(i,j,l) in enumerate(TRIPLES):
        present=(usable[...,i]&usable[...,j]&usable[...,l]&finite[...,i]&finite[...,j]&finite[...,l])
        v=xy[...,i,:]-xy[...,j,:];w=xy[...,l,:]-xy[...,j,:]
        valid=present&(np.linalg.norm(v,axis=-1)>LENGTH_EPS)&(np.linalg.norm(w,axis=-1)>LENGTH_EPS)
        value=np.degrees(np.arctan2(abs(v[...,0]*w[...,1]-v[...,1]*w[...,0]),(v*w).sum(-1)))
        degrees[...,k]=np.where(valid,value,np.nan)
        status[...,k]=np.where(present,DEGENERATE,NOT_OBSERVED)
        assessed=np.asarray(pair_available,bool)&np.asarray(angle_available[...,k],bool)
        status[...,k]=np.where(valid,UNASSESSED,status[...,k])
        status[...,k]=np.where(valid&assessed&angle_unresolved[...,k],UNRESOLVED,status[...,k])
        status[...,k]=np.where(valid&assessed&~angle_unresolved[...,k],ASSESSED_NO_CONFLICT,status[...,k])
    return dict(angle_deg=degrees,angle_state=status,
        computable=np.isfinite(degrees),compare_eligible=status==ASSESSED_NO_CONFLICT)

def compare_aligned_angles(query, reference, expected_reference):
    """Compare already aligned same-shaped arrays, preserving a fixed denominator.

    No alignment is chosen here. Unknown is NaN, never a zero difference.
    Coverage is a descriptive count, not a pass/fail threshold.
    """
    q=query["angle_deg"];r=reference["angle_deg"]
    expected=np.asarray(expected_reference,bool)
    if q.shape!=r.shape or q.shape!=expected.shape:raise ValueError("aligned shapes differ")
    ok=expected&query["compare_eligible"]&reference["compare_eligible"]
    delta=np.where(ok,abs(q-r),np.nan)
    return dict(difference_deg=delta,compared=ok,
        expected_count=int(expected.sum()),compared_count=int(ok.sum()),
        coverage=(float(ok.sum()/expected.sum()) if expected.any() else None),
        mean_difference_deg=(float(delta[ok].mean()) if ok.any() else None))
