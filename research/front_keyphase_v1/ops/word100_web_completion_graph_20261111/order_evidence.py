"""Timing evidence aggregation; retains all occurrences, no expected chronology prior."""
import numpy as np
def temporal_order_evidence(prob,valid,ts,family_a,family_b,confidence=.65):
 if family_a==family_b:return dict(state='not_applicable')
 p=np.asarray(prob);v=np.asarray(valid,bool);t=np.asarray(ts,np.float64)
 if v.ndim==1:v=np.repeat(v[:,None],p.shape[1],axis=1)
 dt=np.diff(t);dt=np.r_[dt,np.median(dt) if len(dt) else 33333.];dt=np.maximum(dt,1)
 def evidence(f):
  w=np.where(v[:,f]&(p[:,f]>=confidence),p[:,f]*dt,0.);s=w.sum()
  return w/s if s else None
 a,b=evidence(family_a),evidence(family_b)
 if a is None or b is None:return dict(state='association_unknown',whole_sign_correct=False)
 # Every observed support contributes, including brief and multiple occurrences.
 before=np.cumsum(a)-a;score=float((b*(before+.5*a)).sum())
 state='supported' if score>=confidence else 'reversed' if score<=1-confidence else 'ambiguous'
 def span(w):
  c=np.cumsum(w);return [int(np.searchsorted(c,.1)),int(np.searchsorted(c,.9))]
 return dict(state=state,order_evidence=score,first_evidence_span=span(a),second_evidence_span=span(b),scope='dominant chronological evidence, not exact repetition count or semantic correctness',whole_sign_correct=False)
