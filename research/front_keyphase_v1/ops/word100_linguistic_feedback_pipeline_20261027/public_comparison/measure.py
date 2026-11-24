"""Descriptive 2D measurements only. No label, model, identity assignment or DTW."""
import numpy as np

TRIPLETS=np.array([(0,1,2),(1,2,3),(2,3,4)]+[(0,b,b+1) if j==0 else (b+j-1,b+j,b+j+1) for b in (5,9,13,17) for j in range(3)])
ANGLE_NAMES=[f'{f}_{j}' for f in ('thumb','index','middle','ring','little') for j in ('base','middle','distal')]

def features(xy,usable,angle_support=None):
    p=np.asarray(xy,dtype=np.float64);v=np.asarray(usable,bool)&np.isfinite(p).all(-1)
    out=np.full((len(p),2,15),np.nan);valid=np.zeros(out.shape,bool)
    for side,base in enumerate((33,54)):
        ids=TRIPLETS+base;a=p[:,ids[:,0]]-p[:,ids[:,1]];b=p[:,ids[:,2]]-p[:,ids[:,1]]
        na=np.linalg.norm(a,axis=-1);nb=np.linalg.norm(b,axis=-1)
        ok=v[:,ids].all(-1)&(na>1e-9)&(nb>1e-9)
        if angle_support is not None:ok &= angle_support[:,side]
        with np.errstate(invalid='ignore',divide='ignore'):
            bend=180-np.degrees(np.arccos(np.clip(np.sum(a*b,axis=-1)/(na*nb),-1,1)))
        out[:,side][ok]=bend[ok];valid[:,side]=ok
    sw=np.linalg.norm(p[:,11]-p[:,12],axis=-1);origin=(p[:,11]+p[:,12])/2
    body=v[:,11]&v[:,12]&(sw>1e-9)
    position=np.full((len(p),2,3,2),np.nan);posvalid=np.zeros((len(p),2,3),bool)
    for side,base in enumerate((33,54)):
        for k,idx in enumerate((base,15+side,13+side)):
            good=body&v[:,idx];posvalid[:,side,k]=good
            position[good,side,k]=(p[good,idx]-origin[good])/sw[good,None]
    return dict(angle=out,angle_valid=valid,position=position,position_valid=posvalid)

def compare(q,qv,r,rv,mapping,planned,vector=False):
    m=np.asarray(mapping);ok=(m>=0)&(m<len(r));idx=np.clip(m,0,len(r)-1)
    qm=qv & planned.reshape((-1,)+(1,)*(qv.ndim-1))
    rm=rv[idx]&ok.reshape((-1,)+(1,)*(rv.ndim-1))&planned.reshape((-1,)+(1,)*(rv.ndim-1))
    both=qm&rm
    delta=np.linalg.norm(q-r[idx],axis=-1) if vector else np.abs(q-r[idx])
    delta=np.where(both,delta,np.nan)
    denom=int(planned.sum()*np.prod(qv.shape[1:]));n=int(both.sum())
    vals=delta[both]
    return dict(planned=denom,query_observed=int(qm.sum()),reference_observed=int(rm.sum()),jointly_observed=n,
       insufficient=denom-n,query_coverage=float(qm.sum()/denom) if denom else None,
       reference_coverage=float(rm.sum()/denom) if denom else None,joint_coverage=n/denom if denom else None,
       median_abs_difference=float(np.median(vals)) if n else None,p90_abs_difference=float(np.percentile(vals,90)) if n else None),delta,both
