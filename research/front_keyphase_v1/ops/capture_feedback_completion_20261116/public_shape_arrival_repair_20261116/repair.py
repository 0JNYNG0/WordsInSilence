"""Common evidence bridge. No word/person identity or requested repetition input."""
from audit_public import *
from observed_gap import preserve_sparse_arrivals

def linked_arrivals(static,motion):
    events=[];separate=closer=0;armed=None;last_arrival=None
    for i,(s,m) in enumerate(zip(static,motion)):
        if s==0:
            separate=closer=0;armed=None;continue
        if m==2:
            separate+=1;closer=0
            # The motion head already observes a seven-frame window. A short
            # lift independently confirmed by the posture head is evidence;
            # requiring a second overlapping window discarded real reversals.
            if separate>=2:armed=i-1
            elif s==1:armed=i
        elif m==1:
            closer+=1;separate=0
            if armed is None and last_arrival is None:armed=i
        else:closer=separate=0
        if closer>=2 and armed is not None:
            end=i+3
            if end<len(static) and static[end]==2 and (last_arrival is None or armed>last_arrival):
                events.append(dict(start=int(armed),end=int(end),source='independent_motion_plus_observed_near'));last_arrival=end;armed=None;closer=0
    return events

class RepairedVisible(SparseVisible):
    def movement(self,s,m,events,available,ts):
        arrivals=linked_arrivals(s,m)
        evidence=preserve_sparse_arrivals(s,m,arrivals,available,ts)
        evidence['bridge_version']='separation_posture_and_temporal_motion_v1'
        return evidence
