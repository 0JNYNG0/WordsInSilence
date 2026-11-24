"""Pure measurement checks, not neural inference or benchmark rescoring."""
from audit import *
from palm_projection_core import geometry

def main():
    assert not(U/'PALM_CORE_VERIFICATION.json').exists()
    result=read(U/'POINT_MEASURE_AUDIT.json');ready={r['clip_id']:r for r in read(OPS/'word100_shared_motion_learning_20261109/READY.json')['rows']if r['role']=='fit'and r['person']in range(2,8)}
    cache={};checks=0;cases=0;maxerr=0.
    for r in result['rows']:
        if r['view']!='full' or r['reverse']:continue
        cid=r['clip_id']
        if cid not in cache:
            src=ready[cid];assert sha(src['channel'])==src['channel_sha256'];ch=np.load(src['channel']);cache[cid]=(ch['full'].copy(),ch['timestamp_us'].copy())
        x,ts=cache[cid];x=x.copy();ts=ts.copy()
        if r['label_kind']=='hold':a,b=r['source_span'];x=x[a:b+1].copy();ts=ts[a:b+1];x[0,279]=0
        a,b=r['span'];g,ok,meta=geometry(x,ts,a,b,2);assert ok==r['palm']['eligible'];checks+=1
        if ok:
            err=abs(float(g[40])-r['palm']['projected_delta']);maxerr=max(maxerr,err);assert err<1e-6;checks+=1
        # Distal fingers may take arbitrary shape; kept palm and body points
        # must yield EXACTLY the same measurement, including eligibility.
        xx=x.copy();point=xx[:,:102].reshape(-1,51,2)
        distal=[30+j for j in range(21)if j not in(0,5,9,13,17)]
        point[:,distal]=point[:,30,None];xx[:,np.asarray(distal)+186]=0
        gd,od,_=geometry(xx,ts,a,b,2);assert od==ok and np.array_equal(g,gd);checks+=1
        miss=x.copy();miss[a:b+1,186+35]=0;gm,om,im=geometry(miss,ts,a,b,2);assert not om and not gm.any() and im['state']=='unavailable_measurement';checks+=1
        xr,tr=reverse_interval(x,ts,a,b);ga,oa,_=geometry(xr,tr,a,b,2);gr,orr,_=geometry(x,ts,a,b,2,reverse=True);assert oa==orr==ok and np.allclose(ga,gr,atol=1e-6);checks+=1
        if ok:assert abs(float(g[40])+float(gr[40]))<1e-6;checks+=1
        for axis in[0,1,3,4]:
            gn,on,_=geometry(x,ts,a,b,axis);go,oo,_=old.geometry(x,ts,a,b,axis);assert on==oo and np.array_equal(gn,go);checks+=1
        cases+=1
    pins={str(p):sha(p)for p in[Path(__file__),U/'palm_projection_core.py',U/'POINT_MEASURE_AUDIT.json',U/'POINT_MEASURE_CONTRACT.json']}
    write(U/'PALM_CORE_VERIFICATION.json',dict(at=stamp(),status='PASS',checks=checks,cases=cases,public_fit_clips=len(cache),maximum_endpoint_difference=maxerr,
        checked='Same full-view FIT candidates; source annotation audit consistency, distal-finger/missing-tip invariance, missing-MCP unknown, actual interval reversal matches reverse parameter, other axes byte-equal.',
        pin=pins,no_model_inference=True,no_training=True,no_DEV=True,not_semantic_truth=True))
    print('Pure palm core PASS',checks,'checks',cases,'FIT candidates',flush=True)

if __name__=='__main__':main()
