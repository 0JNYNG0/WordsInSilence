"""Coordinate-first time transform. Never interpolate precomputed velocities."""
import numpy as np
from wis_lab.core.observation import Observation

def mapping(n,core,kind):
    active=np.flatnonzero(core);assert len(active)>=5 and np.all(np.diff(active)==1)
    if kind=='identity':return np.arange(n,dtype=float)
    if kind in ('fast075','slow15'):return np.linspace(0,n-1,round((n-1)*(.75 if kind=='fast075' else 1.5))+1)
    if kind=='pace':return np.interp(np.linspace(0,1,n),[0,.3,.7,1],np.array([0,.2,.8,1])*(n-1))
    if kind=='pause':
        mid=active[len(active)//2];return np.r_[np.arange(mid+1),np.full(15,mid),np.arange(mid+1,n)].astype(float)
    size,portion=kind[4:].split('_');length=3 if size=='3' else max(3,round(.2*len(active)));centre=int(active[round((len(active)-1)*int(portion)/100)]);start=int(np.clip(centre-length//2,active[0],active[-1]-length+1));return np.setdiff1d(np.arange(n),np.arange(start,start+length)).astype(float)

def observation(source,index):
    index=np.asarray(index,float);n=len(index);assert np.isfinite(index).all() and (np.diff(index)>=0).all() and index.min()>=0 and index.max()<len(source['frame_id'])
    a=np.floor(index).astype(int);b=np.ceil(index).astype(int);w=(index-a)[:,None,None]
    observed=source['observed'][a]&source['observed'][b];usable=source['usable'][a]&source['usable'][b]
    xy=(source['xy_px'][a]*(1-w)+source['xy_px'][b]*w).astype(np.float32);xy[~observed]=np.nan
    conf=np.minimum(source['conf'][a],source['conf'][b]).astype(np.float32);score=np.minimum(source['hand_score'][a],source['hand_score'][b]).astype(np.float32)
    cadence=float(np.median(np.diff(source['timestamp_us'])));times=np.rint(source['timestamp_us'][0]+np.arange(n)*cadence).astype(np.int64)
    obs=Observation(xy,times,np.arange(n,dtype=np.int64),observed.copy(),usable.copy(),conf,score);obs.validate();return obs


def body_origin(source,index):
    """Resample the original per-frame shoulder midpoint; retain original scalar scale."""
    index=np.asarray(index,float)
    a=np.floor(index).astype(int);b=np.ceil(index).astype(int);w=(index-a)[:,None]
    return source['origin'][a]*(1-w)+source['origin'][b]*w
