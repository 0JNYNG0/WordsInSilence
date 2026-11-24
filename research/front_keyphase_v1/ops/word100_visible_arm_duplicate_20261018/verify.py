"""Separate scalar geometry / saved-output audit. Does not import candidate."""
import math,time
import numpy as np
from collections import Counter
from run import U,P,A,read,write,load,pinscheck
def distance(a,b):return math.hypot(float(a[0])-float(b[0]),float(a[1])-float(b[1]))
def turn(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
def hull(pts):
    pts=sorted(set(tuple(map(float,p)) for p in np.asarray(pts,np.float32)))
    low=[];high=[]
    for p in pts:
        while len(low)>1 and turn(low[-2],low[-1],p)<=0:low.pop()
        low.append(p)
    for p in reversed(pts):
        while len(high)>1 and turn(high[-2],high[-1],p)<=0:high.pop()
        high.append(p)
    return low[:-1]+high[:-1]
def segment(x,a,b):
    dx=float(b[0])-float(a[0]);dy=float(b[1])-float(a[1]);d=dx*dx+dy*dy
    t=max(0,min(1,((float(x[0])-a[0])*dx+(float(x[1])-a[1])*dy)/d)) if d>1e-12 else 0
    return distance(x,(a[0]+t*dx,a[1]+t*dy))
def region(x,poly):
    if len(poly)<3:return math.inf
    if abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(poly,poly[1:]+poly[:1])))/2<=1e-6:return math.inf
    if all(turn(a,b,x)>=-1e-7 for a,b in zip(poly,poly[1:]+poly[:1])):return 0.
    return min(segment(x,a,b) for a,b in zip(poly,poly[1:]+poly[:1]))
def seen(p,u,ids,w,h):return all(bool(u[k]) and all(math.isfinite(float(x)) for x in p[k]) and 0<=p[k,0]<w and 0<=p[k,1]<h for k in ids)
def scalar(h,p,u,w,ht):
    n=len(h);a=[[math.inf]*2 for _ in h];e=[[math.inf]*2 for _ in h];d=[[math.inf]*2 for _ in h];cov=[[0.]*n for _ in h]
    visible=[seen(p,u,[11+s,13+s,15+s],w,ht) for s in (0,1)];sw=distance(p[11],p[12]) if seen(p,u,[11,12],w,ht) else 0
    palm=[0,5,9,13,17];valid=[all(math.isfinite(float(x)) for pt in hh for x in pt) and all(0<=hh[k,0]<w and 0<=hh[k,1]<ht for k in palm) for hh in h]
    for i,hh in enumerate(h):
        if not valid[i] or sw<=1e-6:continue
        ph=hull(hh[palm])
        for s in (0,1):
            if seen(p,u,[11+s,13+s],w,ht):e[i][s]=distance(hh[0],p[13+s])/sw
            if visible[s]:a[i][s]=max(segment(hh[0],p[13+s],p[15+s]),region(p[15+s],ph))/sw
            if seen(p,u,[11+s,13+s,17+s,19+s,21+s],w,ht):d[i][s]=sum(distance(hh[k],p[q+s]) for k,q in [(20,17),(8,19),(4,21)])/3/sw
        for j,hj in enumerate(h):
            if i==j or not valid[j]:continue
            span=distance(hj[5],hj[17])
            if span>1e-6:cov[i][j]=sum(region(pt,hull(hj))<=.2*span for pt in hh[1:])/20
    pref=[next((s for s in (0,1) if math.isfinite(row[s]) and row[s]<=.35 and row[s]<.7*row[1-s]),-1) for row in a]
    return a,e,d,cov,pref,visible,valid
def expected(sides,states,armvis,pref,cov):
    sides=list(map(int,sides));out=sides.copy()
    if len(sides)==2 and all(armvis):
        for i,s in enumerate(sides):
            j=1-i
            if s>=0 and states[i]==1 and sides[j]==1-s and states[j]==2 and pref[i]==pref[j]==1-s and cov[i][j]>=.8:out[i]=-1
    return out
def main():
    c=read(U/'CONTEXT.json');pinscheck(c['pins']);pinscheck(read(U/'OUTPUT_PINS.json'));counts=Counter();start=time.monotonic();maxerr=0.
    crosskeys={(x['clip_id'],x['view'],x['index']) for x in read(P/'CROSSING_COMPARISON.json')}
    annkeys={(x['clip_id'],x['view'],x['index']) for x in read(A/'VISUAL_REVIEW.json')['records']}
    for rn,r in enumerate(c['rows'],1):
        for v in ('masked','sensor'):
            raw=load(r['views'][v]['raw']);old=load(P/'results'/f"{r['clip_id']}_{v}.npz");out=load(U/'results'/f"{r['clip_id']}_{v}.npz")
            assert np.array_equal(raw['frame_id'],out['frame_id']) and np.array_equal(raw['timestamp_us'],out['timestamp_us'])
            for f,nn in enumerate(raw['raw_count']):
                n=int(nn);ss=old['sides'][f,:n];states=old['state'][f,:n];cov=out['coverage'][f,:n,:n]
                ex=expected(ss,states,out['arms_visible'][f],out['preference'][f,:n],cov)
                assert ex==out['sides'][f,:n].tolist(),(r['clip_id'],v,f,'decision')
                assert np.array_equal(out['new_withheld'][f,:n],np.asarray(ex)!=ss)
                counts['decision_frames']+=1;counts['decision_slots']+=n
                key=(r['clip_id'],v,f)
                # Every potentially mutable provisional observation, every fixed
                # control and every change, plus uniform1/20 frames. No scoring.
                check=bool((states==1).any() or out['new_withheld'][f].any() or key in crosskeys or key in annkeys or f%20==0)
                if check:
                    a,e,d,co,pr,vis,valid=scalar(raw['hand_xy_px'][f,:n],raw['pose_xy_px'][f],raw['pose_usable'][f],int(raw['width'][f]),int(raw['height'][f]))
                    for calc,name in [(a,'arm_cost'),(e,'elbow_distance'),(d,'distal_cost')]:
                        z=np.array(calc).reshape(n,2);rec=out[name][f,:n];assert np.array_equal(np.isfinite(z),np.isfinite(rec)),(key,name,'availability')
                        mask=np.isfinite(z)
                        if mask.any():
                            err=float(np.max(np.abs(z[mask]-rec[mask])));maxerr=max(maxerr,err);assert err<2e-5,(key,name,err)
                    assert pr==out['preference'][f,:n].tolist() and vis==out['arms_visible'][f].tolist() and valid==out['strict_hand_inside'][f,:n].tolist(),key
                    assert np.allclose(np.array(co).reshape(n,n),cov,atol=0,rtol=0),(key,'coverage')
                    assert expected(ss,states,vis,pr,co)==ex
                    counts['independent_geometry_frames']+=1;counts['independent_geometry_slots']+=n
                    counts['all_provisional_slots_checked']+=int(sum(states==1))
        if rn%200==0:print('VERIFY',rn,'/899',flush=True)
    # Separate aggregation of all known truth and preservation controls.
    annotated=read(U/'ANNOTATED_COMPARISON.json');cross=read(U/'CROSSING_COMPARISON.json');changes=read(U/'CHANGES.json')
    for x in annotated:
        out=load(U/'results'/f"{x['clip_id']}_{x['view']}.npz");val=int(out['sides'][x['index'],x['raw_index']]);assert val==(-1 if x['new_side'] is None else x['new_side'])
    s=read(U/'SUMMARY.json');assert s['counts']['new_withheld']==len(changes)
    assert s['normal_preserved']==sum(x['verdict']=='normal_identity' and x['old_side']==x['new_side'] for x in annotated)
    assert s['crossing_preserved']==sum(x['preserved'] for x in cross)
    pinscheck(c['pins']);pinscheck(read(U/'OUTPUT_PINS.json'))
    write(U/'INDEPENDENT.json',{'status':'PASS','counts':dict(counts),'maximum_scalar_geometry_error':maxerr,'protected_pins':len(c['pins']),'saved_output_pins':len(read(U/'OUTPUT_PINS.json')),'seconds':time.monotonic()-start,'limitation':'Same Codex operator, separate scalar math; not independent human anatomy or unseen recognition accuracy.'})
    print(dict(counts),'PASS',flush=True)
if __name__=='__main__':main()
