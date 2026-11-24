import numpy as np
def decode_saved(sp,bp,valid,cfg):
 labels=sp.argmax(-1);events=[];partials=[]
 for k in range(10):
  lab=labels[:,k].copy();lab[(sp[:,k].max(-1)<cfg['state_threshold'])|~valid[:,k]]=-1
  for a,b in bc.groups((lab==0)|(lab==2)):
   starts=[a]+[i for i in range(a+1,b+1) if lab[i]!=lab[i-1]];ends=[i-1 for i in starts[1:]]+[b]
   for lo,hi in zip(starts,ends):
    # Boundary heads refine within a local range, not one maximum over an entire
    # high-activity run containing several repetitions.
    rad=cfg['boundary_radius'];la=lo;hb=hi
    aa=la+int(bp[la:min(lo+rad+1,hi+1),k,0].argmax());bb=max(lo,hi-rad)+int(bp[max(lo,hi-rad):hb+1,k,1].argmax())
    if bb-aa<cfg['min_span_frames'] or not valid[aa:bb+1,k].all():continue
    e=dict(a=int(aa),b=int(bb),axis=k//2,moving='L' if k%2 else 'R',source='learned_temporal_state',prediction=int(lab[lo]),confidence=float(sp[lo:hi+1,k,lab[lo]].mean()),coverage=1.,complete=False,semantic_complete=False,onset_confidence=float(bp[aa,k,0]),offset_confidence=float(bp[bb,k,1]))
    # Missing an endpoint is retained as an attempted action, not discarded.
    e['boundary_supported']=e['onset_confidence']>=.5 and e['offset_confidence']>=.5
    (events if e['boundary_supported'] else partials).append(e)
 return dict(events=events,partials=partials,state=sp,boundary=bp,valid=valid)
