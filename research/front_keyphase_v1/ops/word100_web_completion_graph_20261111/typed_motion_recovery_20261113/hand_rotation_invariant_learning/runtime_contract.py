"""Pure observation guards; missing input must never become positive evidence.

The model remains unchanged. These functions do not infer, repair, interpolate
or assign a target. The conservative whole-window guard is a numeric contract,
not a measured deployed operating policy or a newly selected threshold.
"""
import numpy as np
from rotation_core import canonicalize_hand63

REQUIRED_POINTS=np.array([0]+list(range(5,21)))

def prepare_shape_observation(x):
    canonical,basis_valid,audit=canonicalize_hand63(x)
    observed=basis_valid&(canonical[:,42+REQUIRED_POINTS]>.5).all(1)
    return canonical,observed,dict(audit,basis_valid=basis_valid)

def transition_evidence(x,logits):
    """A confident neural output cannot override an unobservable input window."""
    _,observed,audit=prepare_shape_observation(x)
    if not len(observed) or not observed.all():
        return dict(state='observation_unknown',class_index=None,probability=None,
                    observed_frames=int(observed.sum()),total_frames=len(observed),
                    invalid_basis_frames=int((~audit['basis_valid']).sum()),semantic_completion=None)
    v=np.asarray(logits,dtype=float)
    if v.shape!=(3,) or not np.isfinite(v).all():
        raise ValueError('Expected three finite saved transition logits')
    p=np.exp(v-v.max());p/=p.sum();j=int(p.argmax());conf=float(p[j])
    return dict(state='observed' if conf>=.65 else 'uncertain',class_index=j if conf>=.65 else None,
                probability=conf,observed_frames=len(observed),total_frames=len(observed),
                invalid_basis_frames=0,semantic_completion=None)

def attribute_evidence(x,frame_index,logits):
    _,observed,audit=prepare_shape_observation(x)
    if not 0<=frame_index<len(observed):raise IndexError('Frame is outside supplied context')
    if not observed[frame_index]:
        return dict(state='observation_unknown',attribute_predictions=None,semantic_completion=None)
    v=np.asarray(logits,dtype=float)
    if v.shape!=(4,) or not np.isfinite(v).all():raise ValueError('Expected four finite attribute logits')
    return dict(state='observed_attribute_audit',attribute_predictions=(v>=0).astype(int).tolist(),semantic_completion=None)
