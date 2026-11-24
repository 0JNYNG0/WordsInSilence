"""Target-independent evidence logic, no model, word or signer constants."""
import numpy as np

def partial_match(fingers, available, expected):
    f=np.asarray(fingers);a=np.asarray(available,bool);e=np.asarray(expected)
    known=a & (f>=0)
    mismatch=(known & (f!=e)).any(-1)
    support=known.all(-1) & (f==e).all(-1)
    return np.where(mismatch,0,np.where(support,1,-1))

def overlap_bridges(states, facts, ts, a, b):
    """Only preserve bracketing evidence, never assign a state inside a blind gap."""
    s=np.asarray(states);ts=np.asarray(ts);ov=facts['projected_overlap'];missing=facts['own_missing']
    accepted=[];rejected=[];ids=np.flatnonzero(ov[a:b+1])+a
    for run in np.split(ids,np.flatnonzero(np.diff(ids)>1)+1) if len(ids) else []:
        lo,hi=int(run[0]),int(run[-1]);left,right=lo-1,hi+1
        ok=(left>=a and right<=b and ts[right]-ts[left]<=250000
            and not missing[left:right+1].any() and not ov[left] and not ov[right]
            and s[left]>=0 and s[right]==s[left]
            and np.all((s[lo:hi+1]<0)|(s[lo:hi+1]==s[left])))
        (accepted if ok else rejected).append([lo,hi])
    return accepted,rejected
