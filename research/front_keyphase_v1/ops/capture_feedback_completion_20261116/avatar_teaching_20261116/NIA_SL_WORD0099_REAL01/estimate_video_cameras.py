"""Exploratory camera estimation with explicit unknown intrinsics; never official calibration."""
from pathlib import Path
import json,numpy as np,cv2
from scipy.optimize import minimize_scalar,least_squares
from scipy.sparse import lil_matrix
U=Path(__file__).resolve().parent;V=U/'video_only';cv2.setNumThreads(2);cv2.setRNGSeed(113)
views=['F','L','R','U','D']
audit=json.loads((V/'EPIPOLAR_AUDIT.json').read_text());protocol=json.loads((V/'EPIPOLAR_PROTOCOL.json').read_text())
rows=json.loads((V/'EXTRACTION_COMPLETE.json').read_text())['rows'];raw={r['view']:np.load(r['raw'],allow_pickle=False)for r in rows}
joints=protocol['joints'];fitframes=np.arange(0,len(raw['F']['frame_id']),3)
xy=np.stack([raw[v]['xy_px'][fitframes][:,joints]for v in views]).reshape(5,-1,2).astype(float)
sc=np.stack([raw[v]['score'][fitframes][:,joints]for v in views]).reshape(5,-1).astype(float)
valid=np.isfinite(xy).all(2)&(sc>=.5)&(xy[:,:,0]>=0)&(xy[:,:,0]<1920)&(xy[:,:,1]>=0)&(xy[:,:,1]<1080)
Flist=[np.asarray(x['F'])for x in audit['pairs']]
def K(f):return np.array([[f,0,960],[0,f,540],[0,0,1.]],float)
def ecost(f):
    costs=[]
    for F in Flist:
        s=np.linalg.svd(K(f).T@F@K(f),compute_uv=False);costs.append(((s[0]-s[1])/(s[0]+s[1]))**2)
    return np.mean(costs)
fopt=minimize_scalar(ecost,bounds=(800,6000),method='bounded')
f=float(fopt.x);intr=K(f);cams=[np.c_[np.eye(3),np.zeros(3)]];diagnostics=[]
for vi,F in enumerate(Flist,1):
    mask=valid[0]&valid[vi]
    E=intr.T@F@intr;u,s,vt=np.linalg.svd(E);E=u@np.diag([(s[0]+s[1])/2]*2+[0])@vt
    n,R,t,ch=cv2.recoverPose(E,xy[0,mask],xy[vi,mask],intr)
    cams.append(np.c_[R,t])
    diagnostics.append(dict(view=views[vi],positive_depth_count=int(n),pairs=int(mask.sum()),initial_R=R.tolist(),initial_t=t[:,0].tolist()))
# Front/left pair is only the scale gauge. It does not define actual metres.
both=valid[0]&valid[1]
h=cv2.triangulatePoints(intr@cams[0],intr@cams[1],xy[0,both].T,xy[1,both].T);p=(h[:3]/h[3]).T
full=np.full((xy.shape[1],3),np.nan);full[both]=p
good=both.copy();good[both]&=(p[:,2]>.05)&(p[:,2]<50)&((p@cams[1][:,:3].T+cams[1][:,3])[:,2]>.05)
for vi in range(2,5):
    m=good&valid[vi];ok,rvec,tvec,inliers=cv2.solvePnPRansac(full[m],xy[vi,m],intr,None,iterationsCount=2000,reprojectionError=8.,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
    if not ok:raise ValueError('Cannot initialize camera '+views[vi])
    cams[vi]=np.c_[cv2.Rodrigues(rvec)[0],tvec]
    diagnostics[vi-1]['pnp_inliers']=len(inliers);diagnostics[vi-1]['pnp_pairs']=int(m.sum())

# Use points observed in at least three views and triangulate initial estimates.
pointids=np.flatnonzero(valid.sum(0)>=3);points=[]
for j in pointids:
    A=[]
    for vi in np.flatnonzero(valid[:,j]):
        P=intr@cams[vi];x,y=xy[vi,j];A.extend([x*P[2]-P[0],y*P[2]-P[1]])
    _,_,vt=np.linalg.svd(A);q=vt[-1];points.append(q[:3]/q[3])
points=np.array(points);valid2=valid[:,pointids];obsxy=xy[:,pointids];obsscore=sc[:,pointids]
ci,pi=np.where(valid2);observed=obsxy[ci,pi];weights=np.sqrt(np.clip(obsscore[ci,pi],.5,1))
camparams=np.array([np.r_[cv2.Rodrigues(c[:,:3])[0][:,0],c[:,3]]for c in cams[1:]])
x0=np.r_[np.log(f),camparams.ravel(),points.ravel()]
def unpack(x):
    ff=np.exp(x[0]);pars=x[1:25].reshape(4,6);cc=[np.c_[np.eye(3),np.zeros(3)]]+[np.c_[cv2.Rodrigues(a[:3])[0],a[3:]]for a in pars]
    return ff,cc,x[25:].reshape(-1,3)
def residual(x):
    ff,cc,pp=unpack(x);proj=np.empty_like(observed)
    for v in range(5):
        m=ci==v;cam=pp[pi[m]]@cc[v][:,:3].T+cc[v][:,3];z=cam[:,2];safe=np.where(np.abs(z)<1e-7,1e-7,z);proj[m]=cam[:,:2]/safe[:,None]*ff+[960,540]
    out=((proj-observed)*weights[:,None]).ravel()
    return np.r_[out,1000*(np.linalg.norm(cc[1][:,3])-1)]
sparse=lil_matrix((len(observed)*2+1,len(x0)),dtype=int)
for k,(vi,pj)in enumerate(zip(ci,pi)):
    sparse[2*k:2*k+2,0]=1
    if vi:sparse[2*k:2*k+2,1+(vi-1)*6:1+vi*6]=1
    sparse[2*k:2*k+2,25+pj*3:28+pj*3]=1
sparse[-1,4:7]=1
lower=np.full(len(x0),-np.inf);upper=np.full(len(x0),np.inf);lower[0]=np.log(800);upper[0]=np.log(6000)
fit=least_squares(residual,x0,jac_sparsity=sparse.tocsr(),loss='soft_l1',f_scale=3,max_nfev=200,ftol=1e-5,xtol=1e-5,bounds=(lower,upper),verbose=0)
ff,cc,pp=unpack(fit.x)
np.savez_compressed(V/'estimated_cameras.npz',K=K(ff),cameras=np.array(cc),fit_xyz=pp,fit_point_ids=pointids,fit_frame_ids=fitframes,joints=np.array(joints))
out=dict(status='ESTIMATED_CAMERAS_REQUIRE_HELD_OUT_AND_RGB_VALIDATION',intrinsics_assumptions='Shared square focal length; principal point image center960,540; zero distortion; no real calibration supplied',
 scale='Arbitrary front-left translation norm1, not metres',focal_initial=f,focal_refined=ff,
 essential_focal_diagnostics=[dict(focal=g,cost=float(ecost(g)))for g in [1000,1700,2400,3400,5000]],
 initialization=diagnostics,fit_frames=fitframes.tolist(),fit_points=len(pp),fit_observations=len(observed),
 optimizer_status=int(fit.status),optimizer_message=fit.message,optimizer_nfev=fit.nfev,initial_cost=float(np.mean(np.abs(residual(x0)[:-1]))),final_cost=float(np.mean(np.abs(residual(fit.x)[:-1]))),
 forbidden_claim='Low fit residual does not prove true3D, metric calibration or correct handshape')
(V/'ESTIMATED_CAMERAS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
