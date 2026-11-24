"""Small continuous phase sidecar. No pose edits, labels, or true time map."""
import numpy as np

KERNEL=np.array([1.,4.,6.,4.,1.])/16
MAX_SHIFT=1.

def phase(index):
    x=np.asarray(index,float);assert x.ndim==1 and np.isfinite(x).all() and (np.diff(x)>=0).all()
    if len(x)<5:return x.copy()
    y=x.copy()
    y[2:-2]=np.convolve(x,KERNEL,'valid')
    y[2:-2]=np.clip(y[2:-2],x[1],x[-2])
    y=np.minimum(np.maximum(y,x-MAX_SHIFT),x+MAX_SHIFT)
    assert (np.diff(y)>=-1e-12).all() and np.max(np.abs(y-x))<=MAX_SHIFT+1e-12
    return y

def map_continuous(channels,base,rest,resample):
    index=phase(base['index']);out=resample(channels,index)
    # Preserve every original matched frame as evidence: a new sample cannot
    # silently bypass a missing point or a confirmed resting interval.
    out.update(index=index,first=np.minimum(base['first'],np.floor(index).astype(int)),
               last=np.maximum(base['last'],np.ceil(index).astype(int)),
               source_valid=channels['valid'],source_rest=rest)
    return out

def selftest():
    x=np.arange(19.);assert np.array_equal(phase(x),x)
    for x in (np.array([0,0,0,4,4,4,4,8,8,8.]),np.zeros(8),np.array([0,1,2.])):
        y=phase(x);assert (np.diff(y)>=0).all() and np.max(np.abs(y-x))<=1 and y[0]==x[0] and y[-1]==x[-1]
    return dict(linear_phase_unchanged=True,monotone=True,max_one_input_frame=True,endpoints_unchanged=True,no_true_timing_input=True)
