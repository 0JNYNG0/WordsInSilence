"""Conditional relation of two learned candidate-ranking distributions.

Never an action-presence, omission, semantic correctness, or calibrated
confidence estimator. It preserves positive-duration overlap as its own mass.
No expected order, word, source-edit provenance, duration penalty or learned
threshold enters this module. Softmax temperature is fixed at one.
"""
import numpy as np

def candidate_distribution(spans, scores, presence, timestamp_us):
    spans=np.asarray(spans);scores=np.asarray(scores,dtype=np.float64)
    presence=np.asarray(presence,dtype=np.float64);t=np.asarray(timestamp_us)
    if spans.ndim!=2 or spans.shape[1]!=2 or len(scores)!=len(spans) or presence.shape!=scores.shape:
        raise ValueError('Candidate arrays must have matching lengths')
    if not np.isfinite(scores).all() or not np.isfinite(presence).all():raise ValueError('Finite scores required')
    if not np.issubdtype(spans.dtype,np.integer) or not np.issubdtype(t.dtype,np.integer):raise ValueError('Integer frame indices and microsecond PTS required')
    if np.any(np.diff(t)<=0):raise ValueError('Strictly increasing query timestamps required')
    if len(np.unique(spans,axis=0))!=len(spans):raise ValueError('Duplicate candidate intervals change prior mass')
    if len(spans) and (spans.min()<0 or spans.max()>=len(t) or np.any(spans[:,0]>=spans[:,1])):raise ValueError('Invalid nonzero interval')
    eligible=presence>=0
    if not eligible.any():return None
    s=scores[eligible];w=np.exp(s-s.max());w/=w.sum()
    return t[spans[eligible]],w

def relation_mass(a,b):
    if a is None or b is None:
        return dict(state='no_admitted_candidates',forward=None,reverse=None,overlap=None,action_presence_established=False)
    aa,wa=a;bb,wb=b
    aa=np.asarray(aa);bb=np.asarray(bb);wa=np.asarray(wa,float);wb=np.asarray(wb,float)
    for intervals,weights in [(aa,wa),(bb,wb)]:
        if intervals.ndim!=2 or intervals.shape!=(len(weights),2) or np.any(intervals[:,0]>=intervals[:,1]):raise ValueError('Positive-duration intervals required')
        if np.any(weights<0) or not np.isfinite(weights).all() or not np.isclose(weights.sum(),1):raise ValueError('Normalized nonnegative mass required')
    starts=np.argsort(bb[:,0]);ends=np.argsort(bb[:,1])
    cstart=np.r_[0,np.cumsum(wb[starts])];cend=np.r_[0,np.cumsum(wb[ends])]
    forward=float(wa@(1-cstart[np.searchsorted(bb[starts,0],aa[:,1],side='left')]))
    reverse=float(wa@cend[np.searchsorted(bb[ends,1],aa[:,0],side='right')])
    overlap=1-forward-reverse
    if min(forward,reverse,overlap)<-1e-10:raise AssertionError('Invalid relation partition')
    mass=np.maximum([forward,reverse,overlap],0);mass/=mass.sum()
    names=['forward','reverse','positive_duration_overlap'];i=int(mass.argmax())
    return dict(state=names[i]if mass[i]>=.65 else'mixed_candidate_relations',
                forward=float(mass[0]),reverse=float(mass[1]),overlap=float(mass[2]),
                action_presence_established=False,interpretation='conditional_rank_mass_only_not_calibrated_correctness')
