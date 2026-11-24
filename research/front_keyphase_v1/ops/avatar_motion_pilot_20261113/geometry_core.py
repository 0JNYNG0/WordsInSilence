"""Explicit-camera geometry tools; no assumptions about real-source coordinate units."""
import numpy as np


def matrix(record, key, shape):
    field=record[key]
    value=field['data'] if isinstance(field,dict) else field
    a=np.fromstring(value,sep=' ') if isinstance(value,str) else np.asarray(value,dtype=float)
    if a.size!=np.prod(shape) or not np.isfinite(a).all():
        raise ValueError(f'{key}: expected finite {shape}')
    return a.reshape(shape)


def camera(record):
    K=matrix(record,'Intrinsics',(3,3))
    M=matrix(record,'CameraMatrix',(3,4))
    d=matrix(record,'Distortion',(5,))
    if not np.allclose(K[2],[0,0,1],atol=1e-8) or K[0,0]<=0 or K[1,1]<=0:
        raise ValueError('Unsupported intrinsic convention')
    R=M[:,:3]
    if not np.allclose(R@R.T,np.eye(3),atol=1e-3) or not np.isclose(np.linalg.det(R),1,atol=1e-3):
        raise ValueError('CameraMatrix is not an orthonormal world-to-camera rigid transform; inspect source convention')
    return K,M,d


def distort_xy(xy,d):
    xy=np.asarray(xy,dtype=np.float64)
    k1,k2,p1,p2,k3=np.asarray(d,dtype=float)
    x,y=xy[...,0],xy[...,1]
    r2=x*x+y*y
    radial=1+k1*r2+k2*r2*r2+k3*r2*r2*r2
    return np.stack([x*radial+2*p1*x*y+p2*(r2+2*x*x),
                     y*radial+p1*(r2+2*y*y)+2*p2*x*y],axis=-1)


def project(xyz,K,M,d,*,coordinate_mode):
    xyz=np.asarray(xyz,dtype=np.float64)
    if coordinate_mode=='world':
        p=xyz@M[:,:3].T+M[:,3]
    elif coordinate_mode=='camera':
        p=xyz
    else:
        raise ValueError('Explicit coordinate_mode must be world or camera; no automatic fitted convention')
    positive=np.isfinite(p).all(axis=-1)&(p[...,2]>1e-12)
    xy=np.full(p.shape[:-1]+(2,),np.nan)
    xy[positive]=p[positive,:2]/p[positive,2,None]
    distorted=distort_xy(xy,d)
    uvh=np.concatenate([distorted,np.ones(distorted.shape[:-1]+(1,))],axis=-1)@K.T
    return uvh[...,:2]/uvh[...,2,None],positive


def undistort_pixels(uv,K,d):
    """Inverse Brown5, including intrinsic skew; reject nonconvergence."""
    uv=np.asarray(uv,float)
    h=np.concatenate([uv,np.ones(uv.shape[:-1]+(1,))],-1)@np.linalg.inv(K).T
    target=h[...,:2]/h[...,2,None]
    guess=target.copy()
    eps=1e-6
    for _ in range(25):
        error=distort_xy(guess,d)-target
        if np.max(np.abs(error))<1e-11:
            return guess
        col0=(distort_xy(guess+np.array([eps,0]),d)-distort_xy(guess-np.array([eps,0]),d))/(2*eps)
        col1=(distort_xy(guess+np.array([0,eps]),d)-distort_xy(guess-np.array([0,eps]),d))/(2*eps)
        jac=np.stack([col0,col1],-1)
        step=np.linalg.solve(jac,error[...,None])[...,0]
        guess-=step
    raise ValueError('Distortion inversion did not converge; no substituted undistorted data')


def triangulate(uv_by_view,cameras):
    """Linear DLT on explicitly calibrated corresponding same-time observations."""
    if len(cameras)<2 or len(uv_by_view)!=len(cameras):
        raise ValueError('At least two matching cameras are required')
    normalized=[undistort_pixels(uv,K,d)for uv,(K,M,d)in zip(uv_by_view,cameras)]
    n=len(normalized[0]);points=[]
    if not all(v.shape==(n,2)for v in normalized):
        raise ValueError('Matching point identities/counts required')
    for i in range(n):
        rows=[]
        for xy,(_,M,_)in zip(normalized,cameras):
            x,y=xy[i]
            rows.extend([x*M[2]-M[0],y*M[2]-M[1]])
        _,_,V=np.linalg.svd(np.stack(rows))
        h=V[-1]
        if abs(h[3])<1e-12:
            raise ValueError('Degenerate triangulation')
        points.append(h[:3]/h[3])
    return np.asarray(points)


def hand_basis(xyz):
    """Right-handed palm frame; caller preserves anatomical labels explicitly."""
    xyz=np.asarray(xyz,float)
    y=xyz[9]-xyz[0]
    across=xyz[5]-xyz[17]
    if np.linalg.norm(y)<1e-10:
        raise ValueError('Degenerate wrist-to-middle axis')
    y/=np.linalg.norm(y)
    x=across-y*np.dot(across,y)
    if np.linalg.norm(x)<1e-10:
        raise ValueError('Degenerate palm width')
    x/=np.linalg.norm(x)
    z=np.cross(x,y)
    return np.stack([x,y,z],axis=1)
