"""Reference-scheduled DTW. No word/signer/user visibility threshold."""
import numpy as np

HAND_GROUPS = ("shape_", "orient_", "handpos_", "facerel_", "motion_", "bend_")
BAND = .2
SUPPORT = .35


def weights_for(required, resting, groups):
    """Keep observed arm anchors. Drop only certified resting-hand comparison.

    Neither query presence nor reference detector availability defines the weight
    denominator. Unknown periods retain their original weights.
    """
    out = {name: float(w) if len(required)==2 or name.endswith("_"+required[0]) else 0.
           for name,(_,w) in groups.items()}
    for h,side in enumerate(("L","R")):
        if resting[h] and side in required:
            for stem in HAND_GROUPS: out[stem+side] = 0.
            out["bimanual"] = 0.
    return out


def matrices(q, r, required, rest, groups, legacy):
    rest=np.asarray(rest,bool)
    assert rest.shape == (len(r["features"]),2)
    for h,side in enumerate(("L","R")):
        if side not in required:
            rest=rest.copy();rest[:,h]=False
    code=rest[:,0].astype(np.uint8)+2*rest[:,1].astype(np.uint8)
    cost=np.empty((len(q["features"]),len(r["features"])))
    evidence=np.empty_like(cost);denominator=np.empty(len(r["features"]))
    for c in np.unique(code):
        weights=weights_for(required,(bool(c&1),bool(c&2)),groups)
        co,ev=legacy(q,r,group_weights=weights,hand_participation=False,one_sided_mode="flat")
        cols=code==c;cost[:,cols]=co[:,cols];evidence[:,cols]=ev[:,cols]
        denominator[cols]=sum(weights.values())
    return dict(cost=cost,evidence=evidence,expected_group_weight=denominator,rest_code=code)


def align(q,r,required,rest,groups,legacy,path_function):
    m=matrices(q,r,required,rest,groups,legacy)
    path=np.asarray(path_function(m["cost"],BAND),int).reshape(-1,2)
    assert len(path)
    return path,float(m["evidence"][tuple(path.T)].mean()),m


def correspondence_diagnostics(q,r,path,ref_rest,ref_expected,query_source,ref_source,cells):
    """Offline positive-source witnesses, not sign-error judgments.

    Query full-frame witnesses are diagnostic only; NEVER feed costs/schedule.
    """
    qi,ri=path.T;w=cells(q["timestamp_us"])
    active_to_rest=np.zeros((len(w),2),bool)
    rest_to_expected=np.zeros_like(active_to_rest)
    for i in np.unique(qi):
        ids=ri[qi==i]
        for h in (0,1):
            active_to_rest[i,h]=query_source["active"][i,h] and ref_rest[ids,h].any()
            rest_to_expected[i,h]=query_source["rest"][i,h] and (ref_expected[ids,h]&ref_source["active"][ids,h]).any()
    count=np.bincount(qi,minlength=len(q["timestamp_us"]))
    unique_ref_to_one_query=int(sum(len(set(qi[ri==j]))==1 for j in np.unique(ri)))
    return dict(active_to_rest_seconds=[float(w@active_to_rest[:,h]) for h in (0,1)],
                rest_to_active_seconds=[float(w@rest_to_expected[:,h]) for h in (0,1)],
                max_reference_rows_per_query=int(count.max()),
                reference_frames_only_one_query=unique_ref_to_one_query,
                active_to_rest=active_to_rest,rest_to_active=rest_to_expected)
