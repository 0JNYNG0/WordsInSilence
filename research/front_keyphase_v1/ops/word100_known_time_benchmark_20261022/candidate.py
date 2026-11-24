"""Local shape support diagnostic. No transform/original-time labels accepted."""
import numpy as np
EPS=1e-6
RUN=3

def runs(mask, minimum=RUN):
    out=np.zeros_like(mask,bool)
    for h in range(mask.shape[1]):
        for a in range(mask.shape[2]):
            ids=np.flatnonzero(mask[:,h,a])
            for seg in np.split(ids,np.flatnonzero(np.diff(ids)>1)+1):
                if len(seg)>=minimum:out[seg,h,a]=True
    return out

def diagnose(query,reference,path,required):
    """A changing shape need not match one frame: allow adjacent observed envelopes.

    Reference support domain never depends on query detection. Local convex angle
    envelope is a permissive approximation, not anatomical or semantic truth.
    Missing neighbors are not crossed. Different angles remain independent.
    """
    n=len(reference['core']);shape=reference['angle_valid'].shape
    expected=np.broadcast_to(reference['core'][:,None,None],shape).copy()
    for h,side in enumerate(('L','R')):
        expected[:,h]&=side in required
        expected[:,h]&=~reference['source_rest'][:,h,None]
    eligible=expected&reference['angle_valid']
    unknown=expected&~reference['angle_valid'];mismatch=np.zeros(shape,bool)
    distance=np.full(shape,np.nan);low=np.full(shape,np.nan);high=np.full(shape,np.nan)
    for j in range(n):
        matched=np.unique(path[path[:,1]==j,0])
        assert len(matched)
        for h in range(2):
            for a in range(15):
                if not eligible[j,h,a]:continue
                ids=set(int(i) for i in matched if query['angle_valid'][i,h,a])
                # Expand only from a currently supported endpoint; never jump a hole.
                for i in list(ids):
                    for k in (i-1,i+1):
                        if 0<=k<len(query['core']) and query['angle_valid'][k,h,a]:ids.add(k)
                if not ids:unknown[j,h,a]=True;continue
                ids=sorted(ids);lo=query['angle_low'][ids,h,a].min();hi=query['angle_high'][ids,h,a].max()
                low[j,h,a]=lo;high[j,h,a]=hi
                gap=max(reference['angle_low'][j,h,a]-hi,lo-reference['angle_high'][j,h,a],0.)
                distance[j,h,a]=gap;mismatch[j,h,a]=gap>EPS
    persistent=runs(mismatch)
    return dict(expected=expected,unknown=unknown,mismatch=mismatch,persistent=persistent,distance=distance,local_low=low,local_high=high)
