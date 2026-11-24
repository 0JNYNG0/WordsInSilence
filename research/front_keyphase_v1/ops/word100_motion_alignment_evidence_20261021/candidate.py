"""Descriptive evidence only; never changes coordinates, DTW, scores or scope."""
import numpy as np
PALM=np.array([0,5,9,13,17])
EPS=1e-6

def angle_values(x):
    out=[]
    for chain in ((0,1,2,3,4),(0,5,6,7,8),(0,9,10,11,12),(0,13,14,15,16),(0,17,18,19,20)):
        for k in range(3):
            a,b,c=chain[k:k+3];v=x[...,a,:]-x[...,b,:];w=x[...,c,:]-x[...,b,:]
            valid=(np.linalg.norm(v,axis=-1)>EPS)&(np.linalg.norm(w,axis=-1)>EPS)
            out.append(np.where(valid,np.degrees(np.arctan2(abs(v[...,0]*w[...,1]-v[...,1]*w[...,0]),(v*w).sum(-1))),np.nan))
    return np.stack(out,-1)

def primitives(q,source_rest,origin,scale):
    xy=q['xy_px'][:,33:].reshape(-1,2,21,2).astype(float)
    valid=q['usable'][:,33:].reshape(-1,2,21)&q['observed'][:,33:].reshape(-1,2,21)
    b=q['pair_b'].astype(float);vb=q['pair_valid']
    pa=xy[:,:,PALM].mean(2);pb=b[:,:,PALM].mean(2)
    pv=valid[:,:,PALM].all(2)&vb[:,:,PALM].all(2)&q['pair_available']
    pv&=np.isfinite(pa).all(2)&np.isfinite(pb).all(2)
    centre=((pa+pb)/2-origin[:,None,:])/scale
    radius=np.linalg.norm(pa-pb,axis=2)/(2*scale)+1/scale
    centre[~pv]=np.nan;radius[~pv]=np.nan
    ba=angle_values(b);av=(q['angle_state']==4)&np.isfinite(ba)
    lo=np.where(av,np.minimum(q['angle_deg'],ba),np.nan)
    hi=np.where(av,np.maximum(q['angle_deg'],ba),np.nan)
    return dict(centre=centre,radius=radius,palm_valid=pv,angle_low=lo,angle_high=hi,angle_valid=av,
        observed=valid.any(2),source_rest=np.asarray(source_rest,bool).copy(),
        missing=q['angle_state']==0,unconfirmed=q['angle_state']!=4,
        timestamp_us=q['timestamp_us'].copy(),frame_id=q['frame_id'].copy(),core=q['core'].copy())

def states(z):
    n=len(z['core']);pos=np.zeros((n,2),bool);shape=np.zeros((n,2,15),bool)
    linked=(np.diff(z['frame_id'])==1)&(np.diff(z['timestamp_us'])>0)
    pv=z['palm_valid'][1:]&z['palm_valid'][:-1]&linked[:,None]
    distance=np.linalg.norm(z['centre'][1:]-z['centre'][:-1],axis=2)
    pos[1:]=pv&(distance>z['radius'][1:]+z['radius'][:-1]+EPS)
    av=z['angle_valid'][1:]&z['angle_valid'][:-1]&linked[:,None,None]
    shape[1:]=av&((z['angle_low'][1:]>z['angle_high'][:-1]+EPS)|(z['angle_low'][:-1]>z['angle_high'][1:]+EPS))
    movement=pos|shape.any(2)
    # Position, change and lowering are separate axes, NOT exclusive semantic labels.
    return dict(location_seen=z['observed'].copy(),position_change=pos,angle_change=shape,
        measured_change=movement,lowered_witness=z['source_rest'].copy(),
        rest_change_conflict=z['source_rest']&movement,
        observed_without_change_evidence=z['observed']&~movement,
        reappearance=z['observed']&~np.vstack([np.zeros((1,2),bool),z['observed'][:-1]]))

def span_change(z,ids,h):
    """Disjoint two-estimate envelopes; not calibrated confidence or sign truth."""
    ids=np.asarray(ids,int);out=np.zeros(15,bool)
    for a in range(15):
        js=ids[z['angle_valid'][ids,h,a]]
        if len(js)>=2:out[a]=z['angle_low'][js,h,a].max()>z['angle_high'][js,h,a].min()+EPS
    js=ids[z['palm_valid'][ids,h]];pos=False
    for j in range(len(js)):
        if np.any(np.linalg.norm(z['centre'][js[j+1:],h]-z['centre'][js[j],h],axis=1)>
                  z['radius'][js[j+1:],h]+z['radius'][js[j],h]+EPS):pos=True;break
    return out,pos

def inspect(q,r,path,expected,required):
    """Preserve score inputs; attach one-to-many evidence warnings and local gaps."""
    path=np.asarray(path,int);qi,ri=path.T;n=len(q['core'])
    angle_flags=np.zeros((n,2,15),bool);position_flags=np.zeros((n,2),bool);events=[]
    for direction,key,value,source in [('reference_change_to_one_query',qi,ri,r),('query_change_to_one_reference',ri,qi,q)]:
        for fixed in np.unique(key):
            ids=np.unique(value[key==fixed])
            if len(ids)<2:continue
            # A DTW plateau includes adjacent indices; missing values never establish change.
            for h,side in enumerate(('L','R')):
                if side not in required:continue
                if direction.startswith('reference'):
                    ids_h=ids[expected[ids,h].any(1)];targets=np.array([fixed])
                else:
                    if not expected[fixed,h].any():continue
                    ids_h=ids[q['core'][ids]];targets=ids_h
                if len(ids_h)<2:continue
                aa,pp=span_change(source,ids_h,h)
                angle_flags[targets,h]|=aa;position_flags[targets,h]|=pp
                if aa.any() or pp:
                    events.append(dict(kind=direction,fixed=int(fixed),start=int(ids_h[0]),end=int(ids_h[-1]),
                        side=side,angles=np.flatnonzero(aa).tolist(),position=bool(pp),query_rows=targets.tolist()))
    # Public annotated core only for diagnostic timeline. It never changes reference expected.
    relevant=np.array([side in required for side in ('L','R')])[None,:,None]
    gaps=q['unconfirmed']&q['core'][:,None,None]&~q['source_rest'][:,:,None]&relevant
    missing=gaps&q['missing']
    return dict(angle_warning=angle_flags,position_warning=position_flags,
        insufficient_timeline=gaps,missing_timeline=missing,events=events,
        expected=expected.copy(),path=path.copy(),wrong_action=None,complete_performance=None)
