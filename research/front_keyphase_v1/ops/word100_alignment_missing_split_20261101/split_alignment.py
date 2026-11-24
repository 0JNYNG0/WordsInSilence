"""Shared-residual alignment objective, separate two-axis missingness readout.

Zero objective on unobserved cells is NOT agreement. Missing evidence can still
attract a path; the mandatory hidden-change counterexample is not solved by this
module. No words, phases, prior alignment indices, model inference or file writes.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
import time
import numpy as np

_SOURCE = Path(__file__).resolve().parent.parent / "word100_joint_stage_alignment_20261101/joint_alignment.py"
_SPEC = importlib.util.spec_from_file_location("_wis_joint_readonly_for_split", _SOURCE)
_OLD = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _OLD
_NO_BYTECODE = sys.dont_write_bytecode
try:
    sys.dont_write_bytecode = True
    _SPEC.loader.exec_module(_OLD)
finally:
    sys.dont_write_bytecode = _NO_BYTECODE

features = _OLD.features
FAMILIES = _OLD.FAMILIES
_support = _OLD._support
EPS = _OLD.EPS
njit = _OLD.njit


def _arrays(q, r, name):
    x, qm = q["groups"][name]
    y, rm = r["groups"][name]
    x, y, qm, rm = np.asarray(x), np.asarray(y), np.asarray(qm), np.asarray(rm)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or not x.shape[1]:
        raise ValueError("Fixed group dimension mismatch")
    if qm.shape != x.shape or rm.shape != y.shape or qm.dtype != np.bool_ or rm.dtype != np.bool_:
        raise ValueError("Explicit boolean group masks required")
    if len(x) != len(q["frame_id"]) or len(y) != len(r["frame_id"]):
        raise ValueError("Group frame axis mismatch")
    if np.any(qm & ~np.isfinite(x)) or np.any(rm & ~np.isfinite(y)):
        raise ValueError("Observed group value is nonfinite")
    return x, qm, y, rm


def cost(q, r, return_families=False):
    """Shared clipped residual / fixed group D; fixed subgroup/family means.

    Neither missingness penalties nor active/shared-count renormalization are
    present. Unknown values cannot become observed zero labels in readout.
    """
    n, m = len(q["frame_id"]), len(r["frame_id"])
    family_costs = {}
    for family, groups in FAMILIES.items():
        total = np.zeros((n,m), dtype=np.float64)
        for name in groups:
            x, qm, y, rm = _arrays(q,r,name)
            shared = qm[:,None,:] & rm[None,:,:]
            delta = np.minimum(np.abs(x[:,None,:]-y[None,:,:]),1.)
            total += np.where(shared,delta,0.).sum(2) / x.shape[1]
        family_costs[family] = total / len(groups)
    result = sum(family_costs.values()) / len(FAMILIES)
    return (result,family_costs) if return_families else result


@njit(cache=False)
def _dp(matrix):
    n,m=matrix.shape
    dp=np.full((n,m,5),np.inf)
    warps=np.full((n,m,5),2147483647,np.int32)
    back=np.full((n,m,5),-1,np.int8)
    ways=np.zeros((n,m,5),np.int8)
    numeric_ambiguity=np.zeros((n,m,5),np.bool_)
    dp[0,0,0]=2*matrix[0,0];warps[0,0,0]=0;ways[0,0,0]=1
    for i in range(n):
        for j in range(m):
            for state in range(5):
                if i==0 and j==0:continue
                pi,pj=i,j
                if state==0:pi-=1;pj-=1
                elif state==1 or state==2:pi-=1
                else:pj-=1
                if pi<0 or pj<0:continue
                best=np.inf
                for k in range(5):
                    allowed=(state==0 or (state==1 and (k==0 or k==3 or k==4))
                        or (state==2 and k==1) or (state==3 and (k==0 or k==1 or k==2))
                        or (state==4 and k==3))
                    if allowed and dp[pi,pj,k]<best:best=dp[pi,pj,k]
                if not np.isfinite(best):continue
                chosen=-1;least=2147483647;count=0;near=False
                for k in range(5):
                    allowed=(state==0 or (state==1 and (k==0 or k==3 or k==4))
                        or (state==2 and k==1) or (state==3 and (k==0 or k==1 or k==2))
                        or (state==4 and k==3))
                    if not allowed or not np.isfinite(dp[pi,pj,k]):continue
                    value=dp[pi,pj,k]
                    # Secondary objective ONLY on exact floating-point equality.
                    if value==best:
                        count+=ways[pi,pj,k]
                        near=near or numeric_ambiguity[pi,pj,k]
                        if warps[pi,pj,k]<least:
                            chosen=k;least=warps[pi,pj,k]
                    elif abs(value-best)<=EPS:
                        near=True
                dp[i,j,state]=best+(2 if state==0 else 1)*matrix[i,j]
                warps[i,j,state]=least+(0 if state==0 else 1)
                back[i,j,state]=chosen;ways[i,j,state]=min(count,2)
                numeric_ambiguity[i,j,state]=near
    return dp,warps,back,ways,numeric_ambiguity


def solve(matrix):
    matrix=np.asarray(matrix,dtype=np.float64)
    if matrix.ndim!=2 or not matrix.size or not np.isfinite(matrix).all() or (matrix<0).any():
        raise ValueError("Finite nonnegative nonempty cost matrix required")
    dp,warps,back,ways,near=_dp(matrix)
    n,m=matrix.shape;minimum=float(dp[-1,-1].min())
    if not np.isfinite(minimum):
        return {"path":[],"path_weights":[],"normalized_cost":None,"weighted_cost_sum":None,
            "path_weight_sum":0,"non_diagonal_moves":None,"optimal_path_count_capped":0,
            "optimal_path_ambiguous":False,"numeric_tie_ambiguous":False,
            "status":"NO_FEASIBLE_FULL_SCOPE_PATH"}
    exact=[k for k in range(5) if dp[-1,-1,k]==minimum]
    chosen=min(exact,key=lambda k:(int(warps[-1,-1,k]),k))
    count=min(sum(int(ways[-1,-1,k]) for k in exact),2)
    numeric=any(bool(near[-1,-1,k]) for k in exact) or any(
        np.isfinite(dp[-1,-1,k]) and dp[-1,-1,k]!=minimum and abs(dp[-1,-1,k]-minimum)<=EPS for k in range(5))
    min_warps=int(warps[-1,-1,chosen]);i,j=n-1,m-1;state=chosen;path=[];weights=[]
    while True:
        path.append([i,j]);weights.append(2 if state==0 else 1)
        if i==0 and j==0:break
        previous=int(back[i,j,state])
        if previous<0:raise RuntimeError("Broken path backtrace")
        if state==0:i-=1;j-=1
        elif state in (1,2):i-=1
        else:j-=1
        state=previous
    path.reverse();weights.reverse()
    actual=float(sum(matrix[i,j]*w for (i,j),w in zip(path,weights)))
    if not np.isclose(actual,minimum,atol=EPS,rtol=0):raise RuntimeError("Representative altered primary cost")
    if sum(weights)!=n+m or {i for i,j in path}!=set(range(n)) or {j for i,j in path}!=set(range(m)):
        raise RuntimeError("Full path scope/denominator lost")
    return {"path":path,"path_weights":weights,"path_weight_sum":n+m,
        "weighted_cost_sum":minimum,"representative_weighted_cost_sum":actual,
        "normalized_cost":minimum/(n+m),"non_diagonal_moves":min_warps,
        "optimal_path_count_capped":count,"optimal_path_ambiguous":bool(count>1 or numeric),
        "exact_primary_tie_ambiguous":count>1,"numeric_tie_ambiguous":bool(numeric),
        "numeric_ambiguity_epsilon":EPS,"tie_break":"MIN_NON_DIAGONAL_ON_EXACT_PRIMARY_EQUALITY_ONLY",
        "ambiguity_retained_before_secondary_tiebreak":True,
        "status":"NUMERIC_PATH_ONLY_NOT_AN_AGREEMENT_SCORE"}


def _labels(group,d):
    if group=="learned3":return list(_OLD.PROBABILITY_ORDER)
    if group.endswith("_shape"):
        return ["thumb_mcp","thumb_ip"]+[f+"_"+j for f in ("index","middle","ring","little") for j in ("mcp","pip","dip")]
    if group.endswith("_location"):return ["hand_wrist_x","hand_wrist_y","palm_center_x","palm_center_y"]
    if group.endswith("_axis"):return ["palm_axis_x_div2","palm_axis_y_div2"]
    if group.endswith("_arm"):return ["elbow_x","elbow_y","elbow_angle","forearm_axis_x_div2","forearm_axis_y_div2","forearm_length"]
    if group.endswith("_forearm"):return ["projection_along_forearm","signed_cross_forearm"]
    if group=="hands_relation":return ["hands_wrist_delta_x","hands_wrist_delta_y","hands_wrist_distance"]
    if group.endswith("_displacement"):return ["native_span_displacement_x","native_span_displacement_y"]
    if group.endswith("_direction"):return ["native_span_unit_direction_x_div2","native_span_unit_direction_y_div2"]
    return [str(j) for j in range(d)]


def _axis_readout(own,mask,other,other_mask,links,frame_id,pts,labels):
    n,d=own.shape
    sums=np.zeros((n,d));peaks=np.full((n,d),np.nan)
    matched_count=np.zeros((n,d),np.int64);missing_count=np.zeros((n,d),np.int64)
    aligned_count=np.zeros(n,np.int64)
    for i,j in links:
        aligned_count[i]+=1
        shared=mask[i]&other_mask[j]
        delta=np.minimum(np.abs(own[i]-other[j]),1.)
        sums[i,shared]+=delta[shared];matched_count[i,shared]+=1
        peaks[i,shared]=np.fmax(peaks[i,shared],delta[shared])
        missing_count[i]+=mask[i]&~other_mask[j]
    matched=matched_count>0
    means=np.divide(sums,matched_count,out=np.full_like(sums,np.nan),where=matched)
    missing=mask&~matched
    mixed=matched&(missing_count>0)
    status=np.full((n,d),"NOT_COMPARABLE",dtype=object)
    status[matched]="COMPARED";status[mixed]="PARTIAL_SHARED_SUPPORT"
    reasons=np.full((n,d),"OWN_UNOBSERVED",dtype=object)
    reasons[mask]="COUNTERPART_UNOBSERVED"
    reasons[mask & (aligned_count[:,None]==0)]="NO_ALIGNED_COUNTERPART"
    reasons[matched]="SHARED_OBSERVATION";reasons[mixed]="SHARED_AND_MISSING_COUNTERPART_LINKS"
    def nullable(values):return [[float(v) if np.isfinite(v) else None for v in row] for row in values]
    return {"frame_id":np.asarray(frame_id).tolist(),"timestamp_us":np.asarray(pts).tolist(),
        "dimension_labels":labels,"possible_shape":[n,d],"all_opportunities_retained":True,
        "own_observed":mask.tolist(),"matched_support":matched.tolist(),
        "mean_residual":nullable(means),"max_residual":nullable(peaks),
        "matched_link_count":matched_count.tolist(),"aligned_link_count":aligned_count.tolist(),
        "missing_counterpart":missing.tolist(),"missing_counterpart_link_count":missing_count.tolist(),
        "mixed_counterpart_missing":mixed.tolist(),"cell_status":status.tolist(),"cell_reason":reasons.tolist(),
        "counts":{"possible":n*d,"own_observed":int(mask.sum()),"own_unobserved":int((~mask).sum()),
            "matched_support":int(matched.sum()),"observed_but_not_comparable":int(missing.sum()),
            "counterpart_missing_link_cells":int(missing_count.sum()),"mixed_support_cells":int(mixed.sum())},
        "missing_indices_by_dimension":{
            "own_unobserved":[np.flatnonzero(~mask[:,j]).tolist() for j in range(d)],
            "counterpart_unobserved_any_link":[np.flatnonzero(missing_count[:,j]>0).tolist() for j in range(d)],
            "observed_without_shared_counterpart":[np.flatnonzero(missing[:,j]).tolist() for j in range(d)],
            "not_comparable":[np.flatnonzero(~matched[:,j]).tolist() for j in range(d)]},
        "missing_indices_address_original_axis":True,"raw_residual_is_automatic_error":False,
        "semantic_verdict":None}


def readout(q, r, path):
    """Full query AND reference axes; residuals only from shared observations."""
    n,m=len(q["frame_id"]),len(r["frame_id"])
    path=[list(p) for p in path]
    if len({tuple(p) for p in path})!=len(path):raise ValueError("Duplicate correspondence")
    if any(len(p)!=2 or any(not isinstance(v,(int,np.integer)) for v in p)
           or not (0<=p[0]<n and 0<=p[1]<m) for p in path):raise ValueError("Invalid path indices")
    if path and ({i for i,j in path}!=set(range(n)) or {j for i,j in path}!=set(range(m))):
        raise ValueError("Readout cannot silently drop either original frame axis")
    groups={};any_support=False
    for group in [g for names in FAMILIES.values() for g in names]:
        x,qm,y,rm=_arrays(q,r,group);labels=_labels(group,x.shape[1])
        query=_axis_readout(x,qm,y,rm,path,q["frame_id"],q["timestamp_us"],labels)
        reference=_axis_readout(y,rm,x,qm,[(j,i) for i,j in path],r["frame_id"],r["timestamp_us"],labels)
        supported=query["counts"]["matched_support"]>0
        any_support |= supported
        groups[group]={"query":query,"reference":reference,
            "status":"LOCAL_RESIDUALS_ONLY" if supported else "UNASSESSABLE", "semantic_verdict":None}
    return {"groups":groups,"status":"LOCAL_RESIDUALS_ONLY" if any_support else "UNASSESSABLE",
        "query_frame_ids":np.asarray(q["frame_id"]).tolist(),"reference_frame_ids":np.asarray(r["frame_id"]).tolist(),
        "all_query_and_reference_opportunities_retained":True,"missing_values_imputed":False,
        "scalar_agreement":None,"semantic_verdict":None,
        "known_risk":"Shared-only cost can align an observed change to unobserved counterpart; retain NOT_COMPARABLE, never whole-sign normal."}


def align(query, reference):
    started=time.perf_counter()
    q,r=features(query),features(reference)
    matrix,families=cost(q,r,return_families=True)
    result=solve(matrix);evidence=readout(q,r,result["path"])
    result["numeric_path_status"]=result["status"]
    result["status"]=evidence["status"] if result["path"] else "NO_FEASIBLE_FULL_SCOPE_PATH"
    result["evidence"]=evidence
    result["group_support"]=_support(q,r,result["path"])
    result["family_costs"]={name:sum(float(values[i,j])*w for (i,j),w in zip(result["path"],result["path_weights"]))/(len(q["frame_id"])+len(r["frame_id"]))
        if result["path"] else None for name,values in families.items()}
    result.update(query_frame_ids=q["frame_id"].tolist(),reference_frame_ids=r["frame_id"].tolist(),
        query_timestamp_us=q["timestamp_us"].tolist(),reference_timestamp_us=r["timestamp_us"].tolist(),
        same_axis_max_run=2,alternating_nondiagonal_moves_allowed=True,
        global_order_forced=True,semantic_order_verified=False,
        observed_zero_is_semantic_hold=False,alignment_objective_is_agreement=False,
        source_alignment_indices_used=False,word_or_phase_used=False,
        scalar_agreement=None,semantic_verdict=None,whole_sign_verdict=None,
        seconds=time.perf_counter()-started,new_inference=0)
    return result
