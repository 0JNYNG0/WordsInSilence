"""Separate scalar/polygon implementation. Does not import candidate or OpenCV."""
import json,hashlib,time,traceback,math
from pathlib import Path
from collections import Counter
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent;B=O/'word100_uncertain_hand_preservation_20261018';F=O/'word100_forearm_contour_20261018'
def read(p):return json.loads(Path(p).read_text('utf8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cross(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
def hull(points):
    v=sorted(set(map(tuple,np.asarray(points,np.float32).astype(float).tolist())))
    if len(v)<3:return v
    lo=[];hi=[]
    for p in v:
        while len(lo)>=2 and cross(lo[-2],lo[-1],p)<=0:lo.pop()
        lo.append(p)
    for p in v[::-1]:
        while len(hi)>=2 and cross(hi[-2],hi[-1],p)<=0:hi.pop()
        hi.append(p)
    return lo[:-1]+hi[:-1]
def area(p):
    return abs(sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(p,p[1:]+p[:1])))*.5 if len(p)>2 else 0.
def intersect(subject,clip):
    out=subject
    for a,b in zip(clip,clip[1:]+clip[:1]):
        inp=out;out=[]
        if not inp:break
        last=inp[-1];lv=cross(a,b,last)
        for cur in inp:
            cv=cross(a,b,cur)
            if (cv>=0)!=(lv>=0):
                t=lv/(lv-cv);out.append((last[0]+t*(cur[0]-last[0]),last[1]+t*(cur[1]-last[1])))
            if cv>=0:out.append(cur)
            last=cur;lv=cv
    return out
def geom(q,m):
    ok=m & np.isfinite(q).all(axis=-1)
    if sum(ok)<16 or not all(ok[i] for i in [0,5,9,13,17]):return None
    ps=[q[i] for i in [5,9,13,17]]
    span=max(math.dist(a,b) for a in ps for b in ps)
    if span<=1e-6:return None
    center=[sum(p[k] for p in ps)/4 for k in [0,1]]
    poly=hull(q[ok]);return math.dist(center,q[0])/span,span,poly,area(poly)
def verify():
    assert not (U/'INDEPENDENT.json').exists() and not (U/'RUNNING.json').exists()
    ctx=read(U/'CONTEXT.json');start=time.monotonic();c=Counter();maxerror=0.;changed=[]
    # Explicit constants are checked against the frozen contract.
    assert ctx['parameters']==dict(min_points=16,palm=[0,5,9,13,17],mcp=[5,9,13,17],collapse_ratio=.25,reference_ratio=.5,overlap_smaller=.75,wrist_separation=.5,max_gap_us=100000,previous_distance=.4)
    selected={(x['clip_id'],x['view'],x['index']):x for x in read(F/'RESULTS.json')}
    allpins={**ctx['pins'],**read(U/'OUTPUT_PINS.json')}
    for p,h in allpins.items():assert sha(p)==h,p
    for z,r in enumerate(ctx['rows']):
        for view in ['masked','sensor']:
            with np.load(r['views'][view]['raw']) as n,np.load(B/'results'/f"{r['clip_id']}_{view}.npz") as b,np.load(U/'results'/f"{r['clip_id']}_{view}.npz") as out:
                q=n['hand_xy_px'];mask=b['point_mask'];cnt=n['raw_count'];old=b['sides'].copy();ref=old.copy();idn=b['identity_side'];ts=n['timestamp_us'];previous=[]
                for t in range(len(cnt)):
                    h=int(cnt[t]);k=(r['clip_id'],view,t)
                    if k in selected:old[t,:h]=selected[k]['sides'];ref[t,:h]=selected[k]['sides']
                    g=[geom(q[t,i],mask[t,i]) for i in range(h)];ratio=[a[0] if a else np.nan for a in g]
                    np.testing.assert_allclose(out['ratio'][t,:h],ratio,atol=1e-10,rtol=1e-10,equal_nan=True)
                    ov=0.;sep=0.
                    if h==2 and all(a and a[3]>1e-6 for a in g):
                        ov=area(intersect(g[0][2],g[1][2]))/min(a[3] for a in g)
                        sep=math.dist(q[t,0,0],q[t,1,0])/max(a[1] for a in g);c['polygon_pairs']+=1
                    error=abs(out['overlap'][t]-ov);maxerror=max(maxerror,error)
                    assert error<2e-4,('polygon',k,error,out['overlap'][t],ov)
                    assert abs(out['wrist_separation'][t]-sep)<1e-10
                    supports=[False]*h
                    if t and len(previous)==1 and previous[0] and previous[0][3]>1e-6 and previous[0][0]>=.5 and 0<ts[t]-ts[t-1]<=100000:
                        scale=math.dist(n['pose_xy_px'][t,11],n['pose_xy_px'][t,12])
                        if scale>0 and math.isfinite(scale):
                            for i,a in enumerate(g):
                                if a and a[3]>1e-6:
                                    vals=sorted(math.dist(q[t,i,j],q[t-1,0,j]) for j in [5,9,13,17]);supports[i]=((vals[1]+vals[2])/2)/scale<=.4
                    assert out['previous_support'][t,:h].tolist()==supports,('previous',k)
                    if h==2:
                        for i in range(2):
                            j=1-i
                            if (old[t,i]>=0 and idn[t,i]<0 and g[i] and g[j] and g[i][3]>1e-6 and g[j][3]>1e-6 and
                                g[i][0]<.25 and g[j][0]>=.5 and ov>=.75 and sep>=.5 and all(supports)):
                                ref[t,i]=-1
                    for i in range(h):
                        if ref[t,i]!=old[t,i]:changed.append([*k,i,int(old[t,i]),int(ref[t,i])])
                    c['frames']+=1;c['hand_candidates']+=h;c['observed_points_preserved']+=int(mask[t,:h].sum());previous=g
                assert np.array_equal(old,out['baseline_sides'])
                assert np.array_equal(ref,out['sides']),('sides',r['clip_id'],view)
                c['view_outputs']+=1
        if (z+1)%150==0:print('INDEPENDENT',z+1,'/899',flush=True)
    recorded=[[x['clip_id'],x['view'],x['index'],i,x['before'][i],x['after'][i]] for x in read(U/'CHANGES.json') for i in x['changed_raw']]
    assert recorded==changed
    ann=read(U/'ANNOTATED.json');normal=[a for a in ann if a['verdict']=='normal_identity']
    assert len(normal)==441 and all(a['before']==a['after']==a['actual_side'] for a in normal)
    crosses=read(U/'CROSSING.json');assert len(crosses)==315 and all(a['before']==a['after'] for a in crosses)
    s=read(U/'SUMMARY.json');assert s['prior13changes_preserved']==13
    assert s['counts']['frames']==c['frames'] and s['counts']['hand_candidates']==c['hand_candidates'] and s['counts']['observed_points']==c['observed_points_preserved']
    for p,h in allpins.items():assert sha(p)==h,p
    result={'status':'PASS','counts':dict(c),'changes':len(changed),'max_polygon_overlap_error':maxerror,'protected_pins':len(ctx['pins']),'output_pins':len(read(U/'OUTPUT_PINS.json')),'normal441':True,'crossing315':True,'prior13':True,'elapsed_seconds':time.monotonic()-start,'verification':'Separate scalar geometry and monotone-chain/Sutherland-Hodgman polygons; same Codex, not independent human. Exact side decisions and original hashes.'}
    write(U/'INDEPENDENT.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
    try:verify()
    except BaseException:
        if not (U/'FIRST_VERIFY_ERROR.json').exists():write(U/'FIRST_VERIFY_ERROR.json',{'error':traceback.format_exc()})
        raise
