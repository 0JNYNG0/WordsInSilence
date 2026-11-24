"""Same-clip2D multiview consistency before any metric camera or retargeting assumption."""
from pathlib import Path
import json,numpy as np,cv2,hashlib
U=Path(__file__).resolve().parent;V=U/'video_only'
cv2.setNumThreads(2);cv2.setRNGSeed(113)
rows=json.loads((V/'EXTRACTION_COMPLETE.json').read_text())['rows'];raw={r['view']:np.load(r['raw'],allow_pickle=False)for r in rows}
for r in rows:assert hashlib.sha256(Path(r['raw']).read_bytes()).hexdigest()==r['raw_sha256']
assert all(np.array_equal(raw[v]['timestamp_us'],raw['F']['timestamp_us'])for v in raw)
# Body joints plus palm anchors, before reading epipolar results. Scores are a response screen only.
joints=list(range(11))+[91+q for q in [5,9,13,17]]+[112+q for q in [5,9,13,17]]
protocol=dict(joints=joints,response_threshold=.5,fit='frame_id%3==0',check='frame_id%3!=0',
              threshold_semantics='screen for geometric fit, not correctness/visibility',
              fundamental_method='USAC_MAGSAC; residual3px,confidence.999,maxiters10000',
              score='symmetric point-to-epipolar-line distance, separate fit/check')
(V/'EPIPOLAR_PROTOCOL.json').write_text(json.dumps(protocol,indent=2)+'\n')
def residual(F,a,b):
    p=np.c_[a,np.ones(len(a))];q=np.c_[b,np.ones(len(b))]
    l=p@F.T;m=q@F;v=np.abs((q*l).sum(1));return .5*(v/np.maximum(np.linalg.norm(l[:,:2],axis=1),1e-9)+v/np.maximum(np.linalg.norm(m[:,:2],axis=1),1e-9))
def stats(x):return dict(n=len(x),median=float(np.median(x)),p90=float(np.quantile(x,.9)),p95=float(np.quantile(x,.95)),max=float(np.max(x))) if len(x)else None
results=[]
for view in ['L','R','U','D']:
    a=raw['F']['xy_px'][:,joints];b=raw[view]['xy_px'][:,joints]
    valid=np.isfinite(a).all(2)&np.isfinite(b).all(2)&(raw['F']['score'][:,joints]>=.5)&(raw[view]['score'][:,joints]>=.5)
    valid&=(a[:,:,0]>=0)&(a[:,:,0]<1920)&(a[:,:,1]>=0)&(a[:,:,1]<1080)&(b[:,:,0]>=0)&(b[:,:,0]<1920)&(b[:,:,1]>=0)&(b[:,:,1]<1080)
    fit=valid&(np.arange(len(a))[:,None]%3==0);check=valid&~(np.arange(len(a))[:,None]%3==0)
    F,mask=cv2.findFundamentalMat(a[fit].astype(float),b[fit].astype(float),cv2.USAC_MAGSAC,3.,.999,10000)
    assert F is not None and F.shape==(3,3)
    resfit=residual(F,a[fit],b[fit]);rescheck=residual(F,a[check],b[check])
    result=dict(view=view,F=F.tolist(),fit=stats(resfit),check=stats(rescheck),fit_inliers=int(mask.sum()),
                fit_count=int(fit.sum()),check_count=int(check.sum()),status='EPIPOLAR_DIAGNOSTIC_NOT_METRIC_CALIBRATION')
    np.savez_compressed(V/('epipolar_F_'+view+'.npz'),F=F,fit_mask=fit,check_mask=check,joints=np.array(joints),fit_residual_px=resfit,check_residual_px=rescheck)
    results.append(result)
(V/'EPIPOLAR_AUDIT.json').write_text(json.dumps(dict(status='DIAGNOSTIC_COMPLETE',pairs=results,conclusion='No metric3D accuracy follows from epipolar consistency; camera calibration remains estimated'),indent=2)+'\n')


print(json.dumps([dict(view=r['view'],median=r['check']['median'],p90=r['check']['p90'])for r in results]))
