"""Observed position + direction aid for existing DTW. Research only."""
import numpy as np
POSITION_WEIGHT=.25
DIRECTION_WEIGHT=.25
MOTION_FLOOR=.01  # shoulder widths; frozen heuristic, not calibrated confidence

def primitives(e,required):
    xy=np.concatenate([e['centre'][:,:,None,:],e['arm_xy']],axis=2).copy()
    valid=np.concatenate([e['palm_valid'][:,:,None],e['arm_valid']],axis=2).copy()
    radius=np.zeros(valid.shape,float);radius[:,:,0]=e['radius']
    valid &= np.isfinite(xy).all(-1)&np.isfinite(radius)
    valid &= np.array([s in required for s in ('L','R')])[None,:,None]
    # Positive source resting is not inferred from absence. Keep arm anchors.
    valid[:,:,0] &= ~e['source_rest']
    n=len(xy);direction=np.full((n,2,3,2,2),np.nan);dv=np.zeros((n,2,3,2),bool)
    delta=xy[1:]-xy[:-1];norm=np.linalg.norm(delta,axis=-1)
    moving=valid[1:]&valid[:-1]&(norm>radius[1:]+radius[:-1]+MOTION_FLOOR)
    unit=np.divide(delta,norm[...,None],out=np.zeros_like(delta),where=norm[...,None]>0)
    # incoming and outgoing, both in chronological direction. Never bridge gaps.
    direction[1:,:,:,0]=np.where(moving[...,None],unit,np.nan);dv[1:,:,:,0]=moving
    direction[:-1,:,:,1]=np.where(moving[...,None],unit,np.nan);dv[:-1,:,:,1]=moving
    return dict(xy=xy,valid=valid,radius=radius,direction=direction,direction_valid=dv)

def costs(q,r,required):
    a=primitives(q,required);b=primitives(r,required)
    joint=a['valid'][:,None]&b['valid'][None,:]
    dist=np.linalg.norm(a['xy'][:,None]-b['xy'][None,:],axis=-1)
    dist=np.maximum(dist-a['radius'][:,None]-b['radius'][None,:],0)
    pn=joint.sum((2,3));pc=np.where(joint,dist/(1+dist),0).sum((2,3))/np.maximum(pn,1)
    dj=a['direction_valid'][:,None]&b['direction_valid'][None,:]
    dot=np.sum(a['direction'][:,None]*b['direction'][None,:],axis=-1)
    disagreement=(1-np.clip(dot,-1,1))/2
    dn=dj.sum((2,3,4));dc=np.where(dj,disagreement,0).sum((2,3,4))/np.maximum(dn,1)
    assert np.isfinite(pc).all() and np.isfinite(dc).all()
    return dict(position_cost=pc,direction_cost=dc,position_count=pn,direction_count=dn,
                addition=POSITION_WEIGHT*pc+DIRECTION_WEIGHT*dc)

def query_unknown(e,required):
    """Native query timeline for audit, never reduced by alignment or rest columns.

    The source annotation core is evaluation-only and never used in costs().
    """
    expected=e['core'][:,None]&np.array([s in required for s in ('L','R')])[None,:]&~e['source_rest']
    return dict(query_expected=expected,query_angle_unknown=expected[:,:,None]&~e['angle_valid'],
        query_palm_unknown=expected&~e['palm_valid'],query_arm_unknown=expected[:,:,None]&~e['arm_valid'])
