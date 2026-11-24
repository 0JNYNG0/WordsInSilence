"""Preserve supported endpoint ranges beside the original representative point.

These are model support ranges, not calibrated confidence intervals. No
endpoint is moved, no ground-truth band is read and no completion is declared.
"""
from copy import deepcopy
import numpy as np
from occurrence_binding import contiguous_runs


def attach_boundary_support(candidates, boundary_probability, observed, threshold=.5):
    p=np.asarray(boundary_probability);valid=np.asarray(observed,bool)
    if p.ndim!=3 or p.shape[-1]!=3 or valid.shape!=p.shape[:2]:
        raise ValueError('Expected boundary [T,K,3] and observed [T,K]')
    cache={};out=[]
    for candidate in candidates:
        c=deepcopy(candidate);k=c['axis']*2+(c['moving']=='L')
        if not 0<=k<p.shape[1]:raise ValueError('Unsupported axis/side')
        for name,channel,point in [('onset',1,c['a']),('offset',2,c['b'])]:
            if (k,channel) not in cache:
                cache[k,channel]=contiguous_runs((p[:,k,channel]>=threshold)&valid[:,k]&np.isfinite(p[:,k,channel]))
            bands=[span for span in cache[k,channel] if span[0]<=point<=span[1]]
            c[name+'_support_ranges']=bands
        c['endpoint_support_scope']='Model threshold support, not exact RGB truth or calibrated confidence interval'
        c['representative_points_unchanged']=True;c['physical_complete']=None;c['semantic_complete']=None
        out.append(c)
    return out
