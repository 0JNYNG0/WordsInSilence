"""Word-blind whole-sequence alignment of saved measured/learned features.

No labels, phases, reference selection, inference or file I/O. Fixed equal
families; reference-unknown groups have neutral cost .5, never agreement.
The five-state recurrence follows context_match.window_dp: at most TWO
successive moves on the SAME axis. Alternating H/V moves remain permitted.
Global monotonic order is an algorithmic constraint, not semantic validation.
"""
from __future__ import annotations
import time
import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        return lambda f: f

EPS = 1e-9
PROBABILITY_ORDER = ("R_index_only", "R_open_four", "L_open_four")
FAMILIES = {
    "shape": ("L_shape", "R_shape", "learned3", "L_axis", "R_axis"),
    "relations": ("L_location", "R_location", "L_arm", "R_arm", "L_to_R_forearm", "R_to_L_forearm", "hands_relation"),
    "movement": ("L_displacement", "R_displacement", "L_direction", "R_direction"),
}


def features(a):
    """Required: x/mask[T,D], names[D], frame_id/PTS[T], probs/support[T,3].

    Angles and body coordinates reuse their saved normalization. Finger/palm
    geometry is projected measurement, not 3D. Unknown/nonfinite values stay
    masked. Probability masks are explicit runtime support, never RGB labels.
    Extra metadata (including word names) is ignored and cannot affect output.
    """
    required = ("x", "mask", "names", "frame_id", "timestamp_us", "probs", "support")
    if any(k not in a for k in required):
        raise ValueError("Missing explicit observation/output input")
    x = np.asarray(a["x"], dtype=np.float64)
    mask = np.asarray(a["mask"])
    names = list(a["names"])
    p = np.asarray(a["probs"], dtype=np.float64)
    ps = np.asarray(a["support"])
    ids, pts = np.asarray(a["frame_id"]), np.asarray(a["timestamp_us"])
    if x.ndim != 2 or not len(x) or mask.shape != x.shape or mask.dtype != np.bool_:
        raise ValueError("x and boolean mask must have identical nonempty [T,D] shape")
    n = len(x)
    if len(names) != x.shape[1] or len(set(names)) != len(names):
        raise ValueError("Feature names mismatch/duplicate")
    if p.shape != (n,3) or ps.shape != (n,3) or ps.dtype != np.bool_:
        raise ValueError("Explicit probs and boolean support must be [T,3]")
    if ids.shape != (n,) or pts.shape != (n,) or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("Original frame IDs/PTS shape mismatch")
    if not np.all(np.isfinite(pts)) or not np.all(np.diff(pts)>0) or not np.all(np.diff(ids)>0):
        raise ValueError("Original frame IDs and PTS must be strictly ordered")
    valid_p = ps & np.isfinite(p)
    if np.any(valid_p & ((p<0)|(p>1))):
        raise ValueError("Supported probability outside [0,1]")
    m = mask & np.isfinite(x)
    groups = {}

    def columns(column_names, scale=1.):
        try:
            ix=[names.index(name) for name in column_names]
        except ValueError as exc:
            raise ValueError("Required named measured feature unavailable") from exc
        return np.where(m[:,ix], x[:,ix]/scale, 0.), m[:,ix].copy()

    for hand, other in (("L","R"),("R","L")):
        angles = [hand+"_thumb_"+j+"_angle" for j in ("mcp","ip")]
        angles += [hand+"_"+f+"_"+j+"_angle" for f in ("index","middle","ring","little") for j in ("mcp","pip","dip")]
        groups[hand+"_shape"] = columns(angles)
        groups[hand+"_axis"] = columns([hand+"_palm_axis_x",hand+"_palm_axis_y"],2.)
        groups[hand+"_location"] = columns([hand+"_hand_wrist_x",hand+"_hand_wrist_y",hand+"_palm_center_x",hand+"_palm_center_y"])
        pw, pm = columns([hand+"_pose_wrist_x",hand+"_pose_wrist_y"])
        elbow, em = columns([hand+"_elbow_x",hand+"_elbow_y"])
        v=pw-elbow; length=np.linalg.norm(v,axis=1)
        ok=pm.all(1)&em.all(1)&(length>1e-8)  # inherited geometric axis validity
        axis=np.divide(v,length[:,None],out=np.zeros_like(v),where=ok[:,None])/2
        base,bm=columns([hand+"_elbow_x",hand+"_elbow_y",hand+"_elbow_angle"])
        groups[hand+"_arm"]=(np.concatenate([base,axis,np.where(ok,length,0)[:,None]],axis=1),
                              np.concatenate([bm,np.tile(ok[:,None],(1,3))],axis=1))
        wrist,wm=columns([hand+"_hand_wrist_x",hand+"_hand_wrist_y"])
        opw,opm=columns([other+"_pose_wrist_x",other+"_pose_wrist_y"])
        oe,oem=columns([other+"_elbow_x",other+"_elbow_y"])
        direction=opw-oe;disp=wrist-oe;norm2=np.sum(direction*direction,axis=1)
        good=wm.all(1)&opm.all(1)&oem.all(1)&(norm2>1e-8)
        along=np.divide(np.sum(disp*direction,axis=1),norm2,out=np.zeros(n),where=good)
        cross=np.divide(direction[:,0]*disp[:,1]-direction[:,1]*disp[:,0],norm2,out=np.zeros(n),where=good)
        groups[hand+"_to_"+other+"_forearm"]=(np.stack([along,cross],axis=1),np.tile(good[:,None],(1,2)))
        # Symmetric +/- one saved native frame; one-sided recording edges.
        # All positions in the span must be observed. No gap crossing/imputation.
        delta=np.zeros((n,2)); movement_mask=np.zeros((n,2),dtype=bool)
        for i in range(n):
            lo,hi=max(0,i-1),min(n-1,i+1)
            if hi>lo and wm[lo:hi+1].all() and ids[hi]-ids[lo]==hi-lo:
                delta[i]=wrist[hi]-wrist[lo]
                movement_mask[i]=True
        magnitude=np.linalg.norm(delta,axis=1)
        unit=np.divide(delta,magnitude[:,None],out=np.zeros_like(delta),where=(magnitude>0)[:,None])/2
        groups[hand+"_displacement"]=(delta,movement_mask.copy())
        # Exact observed zero displacement is numeric zero, not a HOLD label.
        groups[hand+"_direction"]=(unit,movement_mask.copy())
    groups["hands_relation"] = columns(["hands_wrist_delta_x","hands_wrist_delta_y","hands_wrist_distance"])
    groups["learned3"] = (np.where(valid_p,p,0.),valid_p.copy())
    return {"groups":groups,"frame_id":ids.copy(),"timestamp_us":pts.copy(),
            "family_groups":FAMILIES,"probability_order":list(PROBABILITY_ORDER),
            "movement_span":"native index +/-1; one-sided edges, all span observed; no velocity/time division",
            "observed_zero_is_semantic_hold":False}


def cost(q, r, return_families=False):
    """Fixed mean of three family means; group dimensions average known refs.

    A supported reference dimension with absent query costs 1. A group with
    zero known reference dimensions costs neutral .5. Neither absent families
    nor groups are dropped or reweighted. Reference unknown is not zero truth.
    """
    n,m=len(q["frame_id"]),len(r["frame_id"])
    family_costs={}
    for family,groups in FAMILIES.items():
        acc=np.zeros((n,m),dtype=np.float64)
        for group in groups:
            x,qm=q["groups"][group];y,rm=r["groups"][group]
            shared=qm[:,None,:]&rm[None,:,:]
            den=rm.sum(1)
            delta=np.minimum(np.abs(x[:,None,:]-y[None,:,:]),1.)
            missing=rm[None,:,:]&~qm[:,None,:]
            numerator=np.where(shared,delta,0.).sum(2)+missing.sum(2)
            acc+=np.divide(numerator,den[None,:],out=np.full((n,m),.5),where=den[None,:]>0)
        family_costs[family]=acc/len(groups)
    total=sum(family_costs.values())/len(FAMILIES)
    return (total,family_costs) if return_families else total


@njit(cache=False)
def _dp(matrix):
    n,m=matrix.shape
    dp=np.full((n,m,5),np.inf)
    back=np.full((n,m,5),-1,np.int8)
    ways=np.zeros((n,m,5),np.int8)
    tie_predecessors=np.zeros((n,m,5),np.int8)
    dp[0,0,0]=2*matrix[0,0];ways[0,0,0]=1
    for i in range(n):
        for j in range(m):
            for state in range(5):
                if i==0 and j==0:continue
                pi,pj=i,j
                if state==0:pi-=1;pj-=1
                elif state==1 or state==2:pi-=1
                else:pj-=1
                if pi<0 or pj<0:continue
                best=np.inf;chosen=-1
                for k in range(5):
                    permitted=(state==0 or (state==1 and (k==0 or k==3 or k==4))
                        or (state==2 and k==1) or (state==3 and (k==0 or k==1 or k==2))
                        or (state==4 and k==3))
                    if permitted and dp[pi,pj,k]<best:
                        best=dp[pi,pj,k];chosen=k
                if chosen<0:continue
                count=0;pred_ties=0
                for k in range(5):
                    permitted=(state==0 or (state==1 and (k==0 or k==3 or k==4))
                        or (state==2 and k==1) or (state==3 and (k==0 or k==1 or k==2))
                        or (state==4 and k==3))
                    if permitted and np.isfinite(dp[pi,pj,k]) and abs(dp[pi,pj,k]-best)<=EPS:
                        count+=ways[pi,pj,k];pred_ties+=1
                dp[i,j,state]=best+(2 if state==0 else 1)*matrix[i,j]
                back[i,j,state]=chosen;ways[i,j,state]=min(count,2)
                tie_predecessors[i,j,state]=min(pred_ties,2)
    return dp,back,ways,tie_predecessors


def solve(matrix):
    """Full endpoint-to-endpoint path; no band/phase/window truncation."""
    matrix=np.asarray(matrix,dtype=np.float64)
    if matrix.ndim!=2 or not matrix.size or not np.isfinite(matrix).all() or (matrix<0).any():
        raise ValueError("Finite nonnegative nonempty cost matrix required")
    dp,back,ways,ties=_dp(matrix)
    n,m=matrix.shape;state=int(np.argmin(dp[-1,-1]));value=float(dp[-1,-1,state])
    if not np.isfinite(value):
        return {"path":[],"normalized_cost":None,"weighted_cost_sum":None,
            "path_weight_sum":0,"optimal_path_ambiguous":False,"optimal_path_count_capped":0,
            "status":"NO_FEASIBLE_FULL_SCOPE_PATH","ambiguity_status":"NO_PATH"}
    end_ties=[k for k in range(5) if np.isfinite(dp[-1,-1,k]) and abs(dp[-1,-1,k]-value)<=EPS]
    count=min(sum(int(ways[-1,-1,k]) for k in end_ties),2)
    i,j=n-1,m-1;path=[];tie_nodes=[];weights=[]
    while True:
        path.append([i,j]);weights.append(2 if state==0 else 1)
        if ties[i,j,state]>1:tie_nodes.append([i,j,state])
        if i==0 and j==0:break
        previous=int(back[i,j,state])
        if previous<0:raise RuntimeError("Broken DP backtrace")
        if state==0:i-=1;j-=1
        elif state in (1,2):i-=1
        else:j-=1
        state=previous
    path.reverse();weights.reverse()
    if sum(weights)!=n+m:raise RuntimeError("Weighted path denominator changed")
    if {p[0] for p in path}!=set(range(n)) or {p[1] for p in path}!=set(range(m)):
        raise RuntimeError("Full input scope lost")
    return {"path":path,"normalized_cost":value/(n+m),"weighted_cost_sum":value,
        "path_weights":weights,"path_weight_sum":n+m,"optimal_path_ambiguous":count>1,
        "optimal_path_count_capped":count,"status":"FULL_SCOPE_NUMERIC_ALIGNMENT_CANDIDATE",
        "ambiguity_status":"NUMERICALLY_TIED_PATHS" if count>1 else "SINGLE_NUMERIC_OPTIMUM_NOT_PHYSICAL_CERTAINTY",
        "tie_epsilon":EPS,"count_capped_at":2,"backtracked_tie_nodes":tie_nodes,
        "alternate_paths_fully_enumerated":False}


def _support(q,r,path):
    result={}
    for group in [g for family in FAMILIES.values() for g in family]:
        x,qm=q["groups"][group];y,rm=r["groups"][group]
        seen=np.zeros_like(rm);missing=0;shared_links=0
        for i,j in path:
            shared=qm[i]&rm[j];seen[j]|=shared
            missing+=int((rm[j]&~qm[i]).sum());shared_links+=int(shared.sum())
        expected=int(rm.sum());compared=int(seen.sum())
        result[group]={"reference_possible_cells":int(rm.size),"reference_known_cells":expected,
            "reference_unknown_cells":int(rm.size)-expected,"compared_distinct_reference_cells":compared,
            "reference_coverage":compared/expected if expected else None,
            "query_possible_cells":int(qm.size),"query_known_cells":int(qm.sum()),
            "shared_path_cells":shared_links,"known_reference_missing_query_path_cells":missing,
            "absence_is_zero_truth":False}
    return result


def align(query, reference):
    start=time.perf_counter()
    q,r=features(query),features(reference)
    matrix,families=cost(q,r,return_families=True)
    answer=solve(matrix);path=answer["path"]
    answer["family_costs"]={name:sum(float(values[i,j])*w for (i,j),w in zip(path,answer.get("path_weights",[])))/(len(q["frame_id"])+len(r["frame_id"]))
        if path else None for name,values in families.items()}
    answer.update(query_frame_ids=q["frame_id"].tolist(),reference_frame_ids=r["frame_id"].tolist(),
        query_timestamp_us=q["timestamp_us"].tolist(),reference_timestamp_us=r["timestamp_us"].tolist(),
        query_frames=len(q["frame_id"]),reference_frames=len(r["frame_id"]),
        group_support=_support(q,r,path),family_weights={name:1/3 for name in FAMILIES},
        family_groups={name:list(groups) for name,groups in FAMILIES.items()},
        known_reference_missing_query_cost=1.,reference_empty_group_neutral_cost=.5,
        same_axis_max_run=2,alternating_nondiagonal_moves_allowed=True,
        global_order_forced=True,semantic_order_verified=False,whole_sign_verdict=None,
        word_or_phase_used=False,new_inference=0,seconds=time.perf_counter()-start)
    return answer
