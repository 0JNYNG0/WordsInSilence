"""Numerical observability, NOT true joint correctness.

Finite/in-frame coordinates can collapse to one location. These do not describe
finger bending. Nondegenerate points may still follow a wrong finger: passing
this check never certifies anatomical correctness.
"""
import numpy as np

def finger_observability(x,hand):
    p=np.asarray(x[:,:102]).reshape(-1,51,2);v=x[:,186:237]>.5
    b=30 if hand=='R'else 9;p=p[:,b:b+21];v=v[:,b:b+21]
    # Numerical zero in normalized coordinates, not a guessed angle/size gate.
    v=v&np.isfinite(p).all(-1)
    palm=v[:,0]&v[:,9];axis=np.linalg.norm(p[:,9]-p[:,0],axis=1)>1e-6
    usable=[];reasons=[]
    for k in [5,9,13,17]:
        present=palm&v[:,k:k+4].all(1)
        length=np.linalg.norm(np.diff(p[:,k:k+4],axis=1),axis=-1).sum(1)
        ok=present&axis&(length>1e-6)
        why=np.full(len(p),'geometry_present',dtype='<U40')
        why[~present]='points_unavailable'
        why[present&~axis]='palm_axis_degenerate'
        why[present&axis&~(length>1e-6)]='finger_chain_degenerate'
        usable.append(ok);reasons.append(why)
    return np.stack(usable,1),np.stack(reasons,1)

def classification_reasons(preds,states,usable,geometry_reason):
    families={k:np.where((v==v[0]).all(0),v[0],-1)for k,v in preds.items()}
    reasons=geometry_reason.copy();reasons[usable]='model_confidence_insufficient'
    values=np.stack(list(families.values()))
    disagree=(values==0).any(0)&(values==1).any(0)
    for v in preds.values():disagree|=(v==0).any(0)&(v==1).any(0)
    reasons[usable&disagree]='model_disagreement'
    reasons[usable&(states>=0)]='model_decided'
    return reasons
