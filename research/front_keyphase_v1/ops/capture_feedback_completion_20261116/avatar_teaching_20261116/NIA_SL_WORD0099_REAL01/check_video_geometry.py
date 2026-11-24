"""Held-out timestamp and left-out-view consistency; reconstructed XYZ remains approximate."""
from pathlib import Path
import json,numpy as np,cv2,itertools
U=Path(__file__).resolve().parent;V=U/'video_only'
rows=json.loads((V/'EXTRACTION_COMPLETE.json').read_text())['rows'];views=['F','L','R','U','D']
raw={r['view']:np.load(r['raw'],allow_pickle=False)for r in rows};z=np.load(V/'estimated_cameras.npz');K=z['K'];cams=z['cameras'];P=np.einsum('ij,vjk->vik',K,cams)
nframes=len(raw['F']['frame_id']);joints=list(range(13))+[71,74,77,80]+list(range(91,133));xy=np.stack([raw[v]['xy_px'][:,joints]for v in views]).astype(float);sc=np.stack([raw[v]['score'][:,joints]for v in views]).astype(float)
valid=np.isfinite(xy).all(3)&(sc>=.5)&(xy[:,:,:,0]>=0)&(xy[:,:,:,0]<1920)&(xy[:,:,:,1]>=0)&(xy[:,:,:,1]<1080)
def proj(p,c=P):
    h=np.r_[p,1]@c.transpose(0,2,1);return h[:,:2]/h[:,2,None]
def dlt(vs,obs,weights=None):
    weights=np.ones(len(vs))if weights is None else weights;A=[]
    for v,pt,w in zip(vs,obs,weights):
        x,y=pt;A.extend([(x*P[v,2]-P[v,0])*w,(y*P[v,2]-P[v,1])*w])
    _,_,vt=np.linalg.svd(A);h=vt[-1];return h[:3]/h[3]
def solve(vs,obs):
    options=[]
    for inds in itertools.combinations(range(len(vs)),2):
        i=list(inds);p=dlt(np.array(vs)[i],obs[i]);error=np.linalg.norm(proj(p)[vs]-obs,axis=1)
        options.append((np.median(error),p))
    p=min(options,key=lambda q:q[0])[1]
    for _ in range(4):
        err=np.linalg.norm(proj(p)[vs]-obs,axis=1);w=np.sqrt(np.minimum(1,8/np.maximum(err,1e-7)));p=dlt(vs,obs,w)
    return p
def stats(a):
    a=np.asarray(a);return dict(n=len(a),median=float(np.median(a)),p90=float(np.quantile(a,.9)),p95=float(np.quantile(a,.95)),max=float(a.max()))if len(a)else dict(n=0)
out_xyz=np.full((nframes,len(joints),3),np.nan);errors=np.full((5,nframes,len(joints)),np.nan);nviews=valid.sum(0)
loo={v:{'arms_shoulders_wrists':[],'head':[],'lower_body_unobserved_diagnostic':[],'hands':[]}for v in views};loo_rows=[]
for f in range(nframes):
    for ji,j in enumerate(joints):
        vs=np.flatnonzero(valid[:,f,ji])
        if len(vs)<3:continue
        obs=xy[vs,f,ji];p=solve(vs,obs);out_xyz[f,ji]=p;errors[vs,f,ji]=np.linalg.norm(proj(p)[vs]-obs,axis=1)
        if f%3==0:continue
        for leave in vs:
            remain=vs[vs!=leave]
            if len(remain)<3:continue
            p0=solve(remain,xy[remain,f,ji]);err=float(np.linalg.norm(proj(p0)[leave]-xy[leave,f,ji]));group='head'if j<5 else('arms_shoulders_wrists'if j<11 else('lower_body_unobserved_diagnostic'if j<17 else'hands'))
            loo[views[leave]][group].append(err);loo_rows.append((int(f),int(j),int(leave),err))
np.savez_compressed(V/'approximate_xyz.npz',xyz=out_xyz,joints=np.array(joints),n_views=nviews,reprojection_error_px=errors,source_frames=np.arange(nframes),timestamp_us=raw['F']['timestamp_us'],valid2d=valid,loo_rows=np.array(loo_rows))
body_edges=[(5,7),(7,9),(6,8),(8,10),(5,6)];lengths=[]
for a,b in body_edges:
    d=np.linalg.norm(out_xyz[:,joints.index(a)]-out_xyz[:,joints.index(b)],axis=1);d=d[np.isfinite(d)]
    lengths.append(dict(edge=[a,b],n=len(d),median=float(np.median(d)),p05=float(np.quantile(d,.05)),p95=float(np.quantile(d,.95)),relative_p95_p05=float((np.quantile(d,.95)-np.quantile(d,.05))/np.median(d))))
depth=np.einsum('vij,fpj->vfpi',cams[:,:,:3],out_xyz)+cams[:,None,None,:,3];positive=(depth[:,:,:,2]>0)
out=dict(status='APPROXIMATE_VIDEO_GEOMETRY_DIAGNOSTIC',description='Cameras fitted only frame%3==0; LOO below excludes each scored view and uses other>=3views at remaining timestamps. Still detection self-consistency, not groundtruth.',
         source_calibration_available=False,units='arbitrary baseline scale',reconstructed_cells=int(np.isfinite(out_xyz).all(2).sum()),total_cells=nframes*len(joints),
         left_out_view_errors_px={v:{k:stats(x)for k,x in g.items()}for v,g in loo.items()},
         allview_fit_residual_px=stats(errors[np.isfinite(errors)]),arm_length_stability=lengths,
         negative_depth_cells=int((~positive&np.isfinite(out_xyz).all(2)[None]).sum()),
         interpretation='Do not call these coordinates measured/validated motion. Bone-length variability and image comparison remain required before demonstration.')
(V/'GEOMETRY_DIAGNOSTIC.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
