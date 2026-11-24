from pathlib import Path
import json, numpy as np
import candidate as C
U=Path(__file__).resolve().parent
def main():
    tests=[];rng=np.random.default_rng(31);xy=rng.normal(size=(12,21,2));ok=np.ones((12,21),bool)
    a,av=C.angles(xy,ok);assert av.all();tests.append('all observed nondegenerate triplets computed')
    mask=ok.copy();mask[:,8]=False
    b,bv=C.angles(xy,mask);changed=[j for j,t in enumerate(C.TRIPLETS) if 8 in t]
    assert changed==[5] and (~bv[:,changed]).all() and bv.sum()==12*14
    assert np.allclose(a[:,[j for j in range(15) if j not in changed]],b[:,[j for j in range(15) if j not in changed]])
    tests.append('missing index tip withholds only its dependent angle, not other fingers')
    perturbed=xy.copy();perturbed[:,8]=1e8
    c,cv=C.angles(perturbed,mask);assert np.array_equal(bv,cv) and np.allclose(b,c,equal_nan=True);tests.append('unusable hidden joint values do not affect available angles')
    rot=np.array([[0.,-1.],[1.,0.]])
    c,cv=C.angles(xy@rot*3+50,ok);assert np.allclose(a,c,atol=1e-10);tests.append('local angles invariant to translation image rotation and uniform scale')
    c,cv=C.angles(xy,np.zeros_like(ok));assert not cv.any() and np.isnan(c).all();tests.append('missing hand produces no zeros or invented angles')
    deg=xy.copy();deg[:,1]=deg[:,2];c,cv=C.angles(deg,ok);assert not cv[:,0].any() and not cv[:,1].any();tests.append('zero bone withholds dependent angles')
    path=list(zip(range(12),range(12)))
    d=C.compared_angles(a,a,path);assert d['values']==[0.]*15
    agg=C.aggregate_angles([d]*5);assert agg['whole_hand']==0 and agg['angles_ready']==15
    d2=C.compared_angles(a,b,path);agg=C.aggregate_angles([d2]*5);assert agg['angles_ready']==14 and agg['whole_hand'] is None
    assert C.aggregate_angles([d]*4)['angles_ready']==0;tests.append('partial angles or too few references cannot emit full-hand result')
    assert all(x is None for x in C.compared_angles(a,a,path,False)['values']);tests.append('frozen insufficient alignment remains ineligible')
    q=np.stack([np.arange(12)*.1,np.zeros(12)],1);r=q.copy();pv=np.ones(12,bool);rv=pv.copy();rv[0]=False
    tr=C.trajectory(q,r,pv,rv,path);assert abs(tr['old']-10)<1e-8 and tr['paired_start']<1e-8;tests.append('mutual aligned origin removes unequal observed-prefix offset in controlled example')
    tr2=C.trajectory(q+100,r+500,pv,rv,path);assert abs(tr2['paired_start']-tr['paired_start'])<1e-8;tests.append('trajectory retains independent translation invariance')
    n=20;pts=np.zeros((n,75,2));valid=np.ones((n,75),bool);pts[:,11]=[80,50];pts[:,12]=[120,50]
    for sl in (slice(33,54),slice(54,75)):
        pts[:,sl,0]=np.arange(21)+80;pts[:,sl,1]=140+np.arange(21)*.1
    native={'xy_px':pts,'usable':valid};view={'xy_px':pts.copy(),'usable':valid.copy()};view['usable'][:,33:]=False;view['xy_px'][:,33:]=np.nan
    ts=np.arange(n)*33333;origin=np.tile([100,50],(n,1));states=C.source_states(native,view,ts,130,200,origin,40,[[.1,.5]])
    assert states['L']['offline_full_view_state'][-1]==3 and not states['L']['position_observed'].any();tests.append('full-view stable lowered witness is separate from empty cropped-view evidence')
    hidden={'xy_px':pts.copy(),'usable':valid.copy()};hidden['usable'][:,33:]=False;hidden['xy_px'][:,33:]=np.nan
    states=C.source_states(hidden,view,ts,130,200,origin,40,[[.1,.5]])
    assert not states['L']['offline_full_view_state'].any();tests.append('unobserved full source cannot be called lowered or resting')
    pts[:,33:54,1]=90;states=C.source_states(native,view,ts,130,200,origin,40,[[.1,.5]])
    assert states['L']['inside_source_but_view_missing'].all() and not (states['L']['offline_full_view_state']==3).any();tests.append('stationary in-frame support hand is not classified as lowered')
    out={'status':'PASS','tests':tests};(U/'SELFTEST.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),'utf-8');print(json.dumps({'status':'PASS','tests':len(tests)}))
if __name__=='__main__':main()
