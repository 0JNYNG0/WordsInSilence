"""Research-only separate angle, palm and arm evidence. No verdict or path edit."""
import numpy as np

def local_ids(valid, matched):
    ids=set(int(i) for i in matched if valid[i])
    for i in list(ids):
        for j in (i-1,i+1):
            if 0<=j<len(valid) and valid[j]:ids.add(j)
    return np.array(sorted(ids),int)

def expected(reference,required):
    return reference['core'][:,None]&np.array([s in required for s in ('L','R')])[None,:]&~reference['source_rest']

def geometry(query,reference,path,required):
    """Palm, elbow, wrist kept separate; no interpolated invented position.

    Palm distance outside two-estimate discs is descriptive (not calibrated
    confidence). Arms have only one source and therefore raw observed distances.
    The reference time domain cannot shrink with query disappearance.
    """
    domain=np.broadcast_to(expected(reference,required)[:,:,None],(len(reference['core']),2,3)).copy()
    qxy=np.concatenate([query['centre'][:,:,None,:],query['arm_xy']],axis=2)
    rxy=np.concatenate([reference['centre'][:,:,None,:],reference['arm_xy']],axis=2)
    qvalid=np.concatenate([query['palm_valid'][:,:,None],query['arm_valid']],axis=2)
    rvalid=np.concatenate([reference['palm_valid'][:,:,None],reference['arm_valid']],axis=2)
    shape=domain.shape
    unknown=domain&~rvalid;strict=np.zeros(shape,bool)
    raw=np.full(shape,np.nan);gap=np.full(shape,np.nan);selected=np.full(shape,-1,int)
    fraction=np.full(shape,np.nan)
    for j in range(len(domain)):
        matched=np.unique(path[path[:,1]==j,0])
        for h in range(2):
            for k in range(3):
                if not domain[j,h,k] or not rvalid[j,h,k]:continue
                fraction[j,h,k]=qvalid[matched,h,k].mean()
                strict[j,h,k]=qvalid[matched,h,k].all()
                ids=local_ids(qvalid[:,h,k],matched)
                if not len(ids):unknown[j,h,k]=True;continue
                d=np.linalg.norm(qxy[ids,h,k]-rxy[j,h,k],axis=1)
                # Select an actual observation, never a convex hull centre.
                adjusted=np.maximum(d-query['radius'][ids,h]-reference['radius'][j,h],0) if k==0 else d
                a=int(np.argmin(adjusted));selected[j,h,k]=ids[a];raw[j,h,k]=d[a];gap[j,h,k]=adjusted[a]
    return dict(geom_expected=domain,geom_unknown=unknown,geom_strict=strict,geom_distance=raw,
                geom_gap=gap,geom_selected=selected,geom_fraction=fraction)

def angle_strict(query,reference,path,domain):
    strict=np.zeros_like(domain)
    fraction=np.full(domain.shape,np.nan)
    for j in range(len(domain)):
        ids=np.unique(path[path[:,1]==j,0])
        fraction[j]=query['angle_valid'][ids].mean(0)
        strict[j]=domain[j]&reference['angle_valid'][j]&query['angle_valid'][ids].all(0)
    return strict,fraction
