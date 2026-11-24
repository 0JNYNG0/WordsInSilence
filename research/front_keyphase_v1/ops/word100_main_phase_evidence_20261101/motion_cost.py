"""One fixed magnitude-bounded motion-cost candidate, no source mutation.

U and bounded use the SAME reconstructed velocity validity. U is a new
sidecar-based control, not the old stored C path: an observed zero velocity
is valid vector zero in both. No zero velocity is called a semantic hold.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        return lambda f: f

_SOURCE = Path(__file__).resolve().parent.parent / "word100_relational_sequence_compare_20261031/relational.py"
_SPEC = importlib.util.spec_from_file_location("_wis_motion_cost_readonly_relational", _SOURCE)
_REL = importlib.util.module_from_spec(_SPEC)
_BYTECODE = sys.dont_write_bytecode
try:
    sys.dont_write_bytecode = True
    _SPEC.loader.exec_module(_REL)
finally:
    sys.dont_write_bytecode = _BYTECODE

GROUPS = tuple(_REL.GROUPS)
MODES = ("unit", "bounded")
MAX_GAP_SECONDS = .15
BAND = .25
VELOCITY_SCALE = 1.  # One stored shoulder-width per second, never data-fitted.


def _input(a):
    required = {"x", "mask", "frame_id", "timestamp_us"}
    if not required.issubset(a) or set(a)-required-{"names"}:
        raise ValueError("Only measured69 arrays, their time axes and optional names are accepted")
    x, mask = np.asarray(a["x"],dtype=np.float64), np.asarray(a["mask"])
    ids, pts = np.asarray(a["frame_id"]), np.asarray(a["timestamp_us"])
    if x.ndim != 2 or x.shape[1] != 69 or not len(x) or mask.shape != x.shape or mask.dtype != np.bool_:
        raise ValueError("Nonempty [N,69] values and boolean mask required")
    if ids.shape != (len(x),) or pts.shape != ids.shape or not np.issubdtype(ids.dtype,np.integer):
        raise ValueError("Frame IDs/PTS shape or ID dtype invalid")
    if not np.isfinite(pts).all() or not np.all(np.diff(ids)>0) or not np.all(np.diff(pts)>0):
        raise ValueError("Original frame IDs and timestamps must increase strictly")
    if "names" in a and (len(a["names"])!=69 or len(set(a["names"]))!=69):
        raise ValueError("Invalid optional feature names")
    return dict(x=x,mask=mask & np.isfinite(x),frame_id=ids,timestamp_us=pts)


def velocity(a):
    """Independent sidecar from consecutive observed body-normalized wrists.

    Raw columns0:2/28:30 already carry the source body-normalizer mask. No
    stored velocity/direction column is trusted or overwritten. First frames,
    missing predecessors, nonconsecutive IDs and >.15s gaps remain unknown.
    """
    a=_input(a);n=len(a["x"])
    dt=np.zeros(n,dtype=np.float64)
    if n>1:dt[1:]=np.diff(a["timestamp_us"].astype(np.float64))/1e6
    adjacent=np.zeros(n,dtype=bool)
    if n>1:adjacent[1:]=(np.diff(a["frame_id"])==1)&(dt[1:]>0)&(dt[1:]<=MAX_GAP_SECONDS)
    result={}
    for side,off in (("L",0),("R",28)):
        current=a["mask"][:,off:off+2].all(1)
        valid=adjacent & current & np.r_[False,current[:-1]]
        values=np.zeros((n,2),dtype=np.float64)
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            if n>1:
                np.divide(a["x"][1:,off:off+2]-a["x"][:-1,off:off+2],dt[1:,None],
                          out=values[1:],where=valid[1:,None])
            speed=np.hypot(values[:,0],values[:,1])
        numeric=valid & (~np.isfinite(values).all(1)|~np.isfinite(speed))
        valid &= ~numeric
        values[~valid]=0.
        speed[~valid]=0.
        result[side]=dict(values=values,valid=valid,speed=speed,
                          numeric_invalid=numeric,observed_zero=valid & (speed==0))
    return dict(hands=result,dt_seconds=dt,frame_id=a["frame_id"].copy(),
                timestamp_us=a["timestamp_us"].copy(),max_gap_seconds=MAX_GAP_SECONDS,
                units="stored body-normalized shoulder-width per second",
                semantic_hold_label=False,source_values_changed=False)


def transform(values, valid, mode):
    if mode not in MODES:raise ValueError("mode must be unit or bounded")
    v=np.asarray(values,dtype=np.float64);ok=np.asarray(valid)
    if v.ndim!=2 or v.shape[1]!=2 or ok.shape!=(len(v),) or ok.dtype!=np.bool_:
        raise ValueError("Velocity [N,2] and boolean validity [N] required")
    if np.any(ok & ~np.isfinite(v).all(1)):raise ValueError("Supported velocity must be finite")
    speed=np.hypot(v[:,0],v[:,1]);out=np.zeros_like(v)
    if np.any(ok & ~np.isfinite(speed)):raise ValueError("Supported speed must be finite")
    if mode=='unit':
        np.divide(v,speed[:,None],out=out,where=(ok&(speed>0))[:,None])
    else:
        np.divide(v,VELOCITY_SCALE+speed[:,None],out=out,where=ok[:,None])
    # All components remain in the original direction-group [-.5,.5] scale.
    return out/2.,np.repeat(ok[:,None],2,axis=1)


def features(a, mode='bounded'):
    a=_input(a);out=_REL.features(a);v=velocity(a)
    for side in ('L','R'):
        hand=v['hands'][side]
        out[side+'_direction']=transform(hand['values'],hand['valid'],mode)
    return out


def cost(query, reference, mode='bounded', return_groups=False):
    """The same13 group means, reference mask penalty1 and empty neutral.5."""
    q,r=features(query,mode),features(reference,mode)
    if not return_groups:return _REL.alignment_cost(q,r)
    matrices={}
    for group in GROUPS:
        x,qm=q[group];y,rm=r[group]
        shared=qm[:,None]&rm[None]
        expected=rm.sum(1)[None]
        residual=np.where(shared,np.minimum(np.abs(x[:,None]-y[None]),1.),0.).sum(2)
        missing=(rm[None]&~qm[:,None]).sum(2)
        matrices[group]=np.divide(residual+missing,expected,
                                  out=np.full((len(x),len(y)),.5),where=expected>0)
    return sum(matrices.values())/13.,matrices


def unit_cost(query,reference):return cost(query,reference,'unit')
def bounded_cost(query,reference):return cost(query,reference,'bounded')


@njit(cache=False)
def _dp(matrix):
    n,m=matrix.shape
    d=np.full((n+1,m+1),np.inf);d[0,0]=0.
    back=np.full((n,m),-1,np.int8)
    ways=np.zeros((n+1,m+1),np.int8);ways[0,0]=1
    for i in range(n):
        for j in range(m):
            if abs(i/max(n-1,1)-j/max(m-1,1))>BAND:continue
            vals=(d[i,j],d[i,j+1],d[i+1,j])
            chosen=0
            if vals[1]<vals[chosen]:chosen=1
            if vals[2]<vals[chosen]:chosen=2
            best=vals[chosen]
            if not np.isfinite(best):continue
            count=0
            if vals[0]==best:count+=ways[i,j]
            if vals[1]==best:count+=ways[i,j+1]
            if vals[2]==best:count+=ways[i+1,j]
            d[i+1,j+1]=best+matrix[i,j]
            back[i,j]=chosen;ways[i+1,j+1]=min(count,2)
    return d,back,ways


def dtw(matrix, return_details=False):
    """Original C three-move/full-endpoint/band objective and exact tie order."""
    matrix=np.asarray(matrix,dtype=np.float64)
    if matrix.ndim!=2 or not matrix.size or not np.isfinite(matrix).all() or (matrix<0).any():
        raise ValueError("Finite nonnegative nonempty cost matrix required")
    d,back,ways=_dp(matrix);n,m=matrix.shape
    if not np.isfinite(d[n,m]):raise ValueError("No feasible full-scope path within original .25 band")
    i,j=n-1,m-1;path=[]
    while i>=0 and j>=0:
        path.append([i,j]);move=int(back[i,j])
        if move==0:i-=1;j-=1
        elif move==1:i-=1
        elif move==2:j-=1
        else:raise RuntimeError("Broken C13 backtrace")
    path.reverse()
    if {i for i,j in path}!=set(range(n)) or {j for i,j in path}!=set(range(m)):
        raise RuntimeError("Source frame scope lost")
    if not return_details:return path
    return dict(path=path,objective_sum=float(d[n,m]),exact_optimal_path_count_capped=int(ways[n,m]),
                tie_break='diagonal,query-only,reference-only on exact primary equality',
                objective_is_accuracy=False,physical_timing_verified=False)


def compare(query,reference):
    """Convenience wrapper. Public execution remains owned by the root runner."""
    result={mode:dtw(cost(query,reference,mode),return_details=True) for mode in MODES}
    result['sidecar_support']={axis:{side:dict(possible=len(h['valid']),observed=int(h['valid'].sum()),
                                    missing=int((~h['valid']).sum()),observed_zero=int(h['observed_zero'].sum()))
                                  for side,h in velocity(a)['hands'].items()}
                               for axis,a in [('query',query),('reference',reference)]}
    result.update(unit_is_new_sidecar_control_not_old_saved_C=True,
                  velocity_scale=VELOCITY_SCALE,max_gap_seconds=MAX_GAP_SECONDS,
                  source_values_changed=False,new_inference=0,new_training=0,
                  whole_sign_verdict=None,semantic_hold_label=False,
                  normal_speed_difference_can_change_cost=True)
    return result
