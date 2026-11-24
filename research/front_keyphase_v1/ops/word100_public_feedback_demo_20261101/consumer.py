"""Word-independent feature comparison. No educational tolerance invented."""
import math
GROUPS=[['R_index_only','R_open_four','R_closed_four'],['L_open_four','L_closed_four'],['R_higher','L_higher']]
def compare(q,r,targets,qgate,rgate):
 def known(v):return v is not None and math.isfinite(v)
 def value(v):return int(v>=.5) if known(v) else None
 qv=dict(zip(targets,map(value,q)));rv=dict(zip(targets,map(value,r)))
 conflict=set()
 for group in GROUPS:
  if sum(qv.get(t)==1 for t in group)>1 or sum(rv.get(t)==1 for t in group)>1:conflict.update(group)
 out=[]
 for i,t in enumerate(targets):
  a,b=qv[t],rv[t]
  status=('MODEL_OUTPUT_UNAVAILABLE' if a is None or b is None else
          'CONFLICTING_MODEL_ESTIMATES' if t in conflict else
          'ESTIMATED_FEATURE_DIFFERENCE' if a!=b else
          'SAME_POSITIVE_ATTRIBUTE' if a==1 else 'SAME_NEGATIVE_ATTRIBUTE_ONLY')
  out.append(dict(target=t,query=a,reference=b,query_probability=q[i],reference_probability=r[i],
    status=status,query_geometry_supported=bool(qgate[i]),reference_geometry_supported=bool(rgate[i]),
    raw_estimated_difference=None if a is None or b is None else a!=b,
    same_handshape=None,important_difference=None,educational_correction=None,request_new_recording=None))
 return out
