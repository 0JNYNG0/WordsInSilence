"""Per-finger geometric ambiguity, separate from retained hand position.

No labels, identities, timestamps, or word-specific constants are inputs.
The disagreement envelope describes TWO PREDICTIONS, not true measurement uncertainty.
"""
import numpy as np

PALM = np.array([0, 5, 9, 13, 17])
CHAINS = ((0,1,2,3,4), (0,5,6,7,8), (0,9,10,11,12),
          (0,13,14,15,16), (0,17,18,19,20))
BONES = [(ch[i], ch[i+1]) for ch in CHAINS for i in range(4)]
ANGLES = [(ch[i-1],ch[i],ch[i+1]) for ch in CHAINS for i in (1,2,3)]
EPS = 1e-8  # Arithmetic degeneracy only; not a calibrated pose-error cutoff.


def compare_shape(a, va, b=None, vb=None):
    """Return coordinate/angle evidence without changing input coordinates.

    Align B to A by a proper 2D similarity using wrist+four MCP anchors. Then
    each corresponding point pair defines midpoint and radius=half discrepancy.
    If adjacent endpoint discs overlap, their bone direction is unresolved.
    Missing comparisons remain unassessed. Passing this rule is NOT correctness.
    """
    a=np.asarray(a,dtype=np.float64)
    va=np.asarray(va,bool)&np.isfinite(a).all(1)
    out=dict(
        pair_available=False, alignment_scale=np.nan,
        aligned_b=np.full((21,2),np.nan),
        comparable=np.zeros(21,bool), point_uncertain=np.zeros(21,bool),
        bone_available=np.zeros(20,bool), bone_unresolved=np.zeros(20,bool),
        bone_discrepancy_ratio=np.full(20,np.nan),
        angle_available=np.zeros(15,bool), angle_unresolved=np.zeros(15,bool),
        angle_envelope_low_deg=np.full(15,np.nan), angle_envelope_high_deg=np.full(15,np.nan),
        finger_available_count=np.zeros(5,np.uint8), finger_uncertain=np.zeros(5,bool),
        finger_missing=np.ones(5,bool), residual_px=np.full(21,np.nan),
    )
    if b is None:
        return out
    b=np.asarray(b,dtype=np.float64)
    vb=np.asarray(vb,bool)&np.isfinite(b).all(1)
    both=va&vb
    if not both[PALM].all():
        return out
    ac=a[PALM].mean(0);bc=b[PALM].mean(0)
    ap=a[PALM]-ac;bp=b[PALM]-bc
    denominator=np.sum(bp*bp)
    if denominator<=EPS or np.sum(ap*ap)<=EPS:
        return out
    u, singular, vt=np.linalg.svd(bp.T@ap)
    sign=np.array([1.,1. if np.linalg.det(u@vt)>=0 else -1.])
    rotation=(u*sign)@vt
    scale=float(np.sum(singular*sign)/denominator)
    if not np.isfinite(scale) or scale<=EPS:
        return out
    aligned=(b-bc)@rotation*scale+ac
    residual=np.linalg.norm(a-aligned,axis=1)
    centers=(a+aligned)/2;radius=residual/2
    out.update(pair_available=True,alignment_scale=scale,aligned_b=aligned,
               comparable=both,residual_px=np.where(both,residual,np.nan))
    halfwidth=np.full(20,np.nan)
    for k,(i,j) in enumerate(BONES):
        ok=both[i] and both[j]
        if not ok:continue
        la=np.linalg.norm(a[j]-a[i]);lb=np.linalg.norm(aligned[j]-aligned[i])
        length=np.linalg.norm(centers[j]-centers[i]);r=radius[i]+radius[j]
        out["bone_available"][k]=True
        unresolved=min(la,lb,length)<=EPS or r>=length
        out["bone_unresolved"][k]=unresolved
        out["bone_discrepancy_ratio"][k]=r/max(length,EPS)
        halfwidth[k]=180. if unresolved else np.degrees(np.arcsin(np.clip(r/length,0,1)))
        if unresolved:
            out["point_uncertain"][[i,j]]=True
    for k,(i,j,l) in enumerate(ANGLES):
        finger=k//3;local=k%3
        bi=finger*4+local;bj=bi+1
        if not (out["bone_available"][bi] and out["bone_available"][bj]):continue
        out["angle_available"][k]=True
        v=centers[i]-centers[j];w=centers[l]-centers[j]
        center=np.degrees(np.arctan2(abs(v[0]*w[1]-v[1]*w[0]),np.dot(v,w)))
        width=halfwidth[bi]+halfwidth[bj]
        out["angle_envelope_low_deg"][k]=max(0.,center-width)
        out["angle_envelope_high_deg"][k]=min(180.,center+width)
        entire_range=(center-width<=0 and center+width>=180)
        out["angle_unresolved"][k]=out["bone_unresolved"][bi] or out["bone_unresolved"][bj] or entire_range
        if out["angle_unresolved"][k]:
            out["point_uncertain"][[i,j,l]]=True
    for finger,ch in enumerate(CHAINS):
        out["finger_available_count"][finger]=out["angle_available"][finger*3:finger*3+3].sum()
        out["finger_uncertain"][finger]=(out["bone_unresolved"][finger*4:finger*4+4].any()
                                          or out["angle_unresolved"][finger*3:finger*3+3].any())
        out["finger_missing"][finger]=not both[list(ch)].all()
    return out


def safe_new_shape_channels(x, point_uncertain, source_added):
    """Optional research adapter, fixed normalization; never modifies source x.

    Apply only to NEW observations. Preserve wrist and four palm anchors in the
    body-relative position block; hold all other uncertain finger points in BOTH
    coordinate paths. Hand-local shape coordinates are held even for an uncertain
    MCP. Existing baseline points are never dropped. This is not production use.
    """
    out=x.copy()
    hold=np.asarray(point_uncertain,bool)&np.asarray(source_added,bool)
    for s in range(2):
        for j in range(1,21):
            mask=hold[:,s,j]
            local=s*21+j
            out[mask,102+2*local:104+2*local]=0
            out[mask,237+local]=0
            if j not in PALM:
                common=9+s*21+j
                out[mask,2*common:2*common+2]=0
                out[mask,186+common]=0
    return out
