"""Conservative per-hand evidence diagnostic, separate from frozen first probe.

Do not use local wrist anchors (identically zero) as handshape evidence. Each
distance is between the same side and all of its required landmarks. Partial
hands remain in the input and the missing-evidence counters; no model input
is changed. This full-landmark gate is deliberately conservative, not a
validated optimal observation threshold or an anatomical correctness model.
"""
import numpy as np
from criteria import arrays,directed

def compare_hand(a,b,axis,side):
    aa,av=arrays(a,axis);bb,bv=arrays(b,axis)
    start=21*side
    ids=np.arange(start+(1 if axis=='local' else 0),start+21)
    ar=av[:,ids].all(1);br=bv[:,ids].all(1)
    out={'reference_frames':len(a),'query_frames':len(b),'reference_uncomparable':int((~ar).sum()),'query_uncomparable':int((~br).sum()),
         'side':['L','R'][side],'required_landmarks':len(ids),'semantic_verdict':None,'distance_scope':'same_side_complete_landmarks_only'}
    if not ar.any() or not br.any():return dict(out,status='INSUFFICIENT',bag=None,ordered=None,order_gap=None)
    aa=aa[ar][:,ids];bb=bb[br][:,ids]
    c=np.linalg.norm(aa[:,None,:,:]-bb[None,:,:,:],axis=-1).mean(-1)
    bag=(c.min(1).mean()+c.min(0).mean())/2
    ordered=(directed(c)+directed(c.T))/2
    assert ordered+1e-12>=bag
    return dict(out,status='OBSERVED_DISTANCE_ONLY',bag=float(bag),ordered=float(ordered),order_gap=float(max(0,ordered-bag)))
