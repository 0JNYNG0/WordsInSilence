"""Independent observed-motion formula/DP verification; no training or data writes.

Public interface:
  verify_cost(query_evidence, reference_evidence, required_sides, saved_arrays)
  verify_path(cost, path)
  fixtures() -> named checks; imports only the new candidate in fixtures().
The production candidate is not called by verify_cost or verify_path.
"""
from pathlib import Path
import importlib.util
import math
import numpy as np


def _points(e, required):
    """Read each anatomical point independently, not candidate concatenations."""
    n=len(e['centre'])
    out={}
    for h,side in enumerate(('L','R')):
        for k in range(3):
            xy=np.empty((n,2),float);valid=np.zeros(n,bool);radius=np.zeros(n,float)
            for i in range(n):
                if k==0:
                    xy[i]=e['centre'][i,h];radius[i]=e['radius'][i,h]
                    present=bool(e['palm_valid'][i,h]) and not bool(e['source_rest'][i,h])
                else:
                    xy[i]=e['arm_xy'][i,h,k-1]
                    present=bool(e['arm_valid'][i,h,k-1])
                valid[i]=side in required and present and np.isfinite(xy[i]).all() and np.isfinite(radius[i])
            din=np.full((n,2),np.nan);dout=np.full((n,2),np.nan)
            vin=np.zeros(n,bool);vout=np.zeros(n,bool)
            for i in range(1,n):
                if not (valid[i-1] and valid[i]):continue
                dx=float(xy[i,0]-xy[i-1,0]);dy=float(xy[i,1]-xy[i-1,1]);length=math.hypot(dx,dy)
                if length<=float(radius[i-1]+radius[i])+.01:continue
                d=np.array([dx/length,dy/length]);din[i]=d;dout[i-1]=d;vin[i]=vout[i-1]=True
            out[h,k]=(xy,valid,radius,((din,vin),(dout,vout)))
    return out


def reconstruct_cost(qe,re,required):
    q=_points(qe,required);r=_points(re,required)
    shape=(len(qe['centre']),len(re['centre']))
    ps=np.zeros(shape);ds=np.zeros(shape);pn=np.zeros(shape,int);dn=np.zeros(shape,int)
    for key in q:
        qxy,qv,qr,qd=q[key];rxy,rv,rr,rd=r[key]
        observed=np.outer(qv,rv)
        distance=np.hypot(qxy[:,0,None]-rxy[None,:,0],qxy[:,1,None]-rxy[None,:,1])
        distance=np.maximum(0.,distance-qr[:,None]-rr[None,:])
        ps[observed]+=(distance/(1+distance))[observed];pn+=observed
        for ((qu,qok),(ru,rok)) in zip(qd,rd):
            observed=np.outer(qok,rok)
            cosine=qu[:,0,None]*ru[None,:,0]+qu[:,1,None]*ru[None,:,1]
            penalty=.5-.5*np.clip(cosine,-1.,1.)
            ds[observed]+=penalty[observed];dn+=observed
    pc=np.divide(ps,pn,out=np.zeros(shape),where=pn>0)
    dc=np.divide(ds,dn,out=np.zeros(shape),where=dn>0)
    return dict(position_cost=pc,direction_cost=dc,position_count=pn,direction_count=dn,addition=(pc+dc)/4)


def verify_cost(qe,re,required,saved):
    expected=reconstruct_cost(qe,re,required)
    for key,value in expected.items():
        assert key in saved,key
        if key.endswith('count'):np.testing.assert_array_equal(value,saved[key])
        else:np.testing.assert_allclose(value,saved[key],atol=2e-12,rtol=2e-12,equal_nan=False)
    for key in ('position_cost','direction_cost'):
        assert np.isfinite(expected[key]).all() and np.all((expected[key]>=-1e-14)&(expected[key]<=1+1e-14))
    assert np.all(expected['addition']>=-1e-14) and np.all(expected['addition']<=.5+1e-14)
    return True


def verify_path(cost,path):
    """Independent accumulated optimum; no production DP/backtracking call."""
    cost=np.asarray(cost,float);path=np.asarray(path,int);n,m=cost.shape
    assert n>=2 and m>=2 and path.ndim==2 and path.shape[1]==2
    assert tuple(path[0])==(0,0) and tuple(path[-1])==(n-1,m-1)
    step=np.diff(path,axis=0)
    assert np.all((step[:,0]>=0)&(step[:,0]<=1)&(step[:,1]>=0)&(step[:,1]<=1)&(step.sum(1)>0))
    band=max(1,round(.2*max(n,m)));prior=np.full(m+1,np.inf);prior[0]=0.
    for i in range(n):
        row=np.full(m+1,np.inf);centre=i*(m-1)/(n-1)
        lo=max(0,math.floor(centre-band));hi=min(m-1,math.ceil(centre+band))
        for j in range(lo,hi+1):
            if np.isfinite(cost[i,j]):row[j+1]=cost[i,j]+min(prior[j],prior[j+1],row[j])
        prior=row
    for i,j in path:
        centre=i*(m-1)/(n-1)
        assert math.floor(centre-band)<=j<=math.ceil(centre+band)
    got=float(cost[tuple(path.T)].sum());best=float(prior[-1]);assert np.isfinite(got) and np.isfinite(best)
    np.testing.assert_allclose(got,best,rtol=2e-10,atol=2e-10)
    return True


def fixtures():
    spec=importlib.util.spec_from_file_location('motion_candidate_fixture',Path(__file__).with_name('candidate.py'))
    c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
    def make(x):
        x=np.asarray(x,float);n=len(x);xy=np.zeros((n,2,2));xy[:,:,0]=x[:,None]
        arm=np.repeat(xy[:,:,None,:],2,axis=2)
        return dict(centre=xy,radius=np.zeros((n,2)),palm_valid=np.ones((n,2),bool),
            arm_xy=arm,arm_valid=np.ones((n,2,2),bool),source_rest=np.zeros((n,2),bool),
            core=np.ones(n,bool),angle_valid=np.ones((n,2,15),bool))
    def check(a,b,req=('L','R')):
        out=c.costs(a,b,req);verify_cost(a,b,req,out);return out
    results={}
    a=make([0,.1,.2,.3,.4]);b=make([0,.2,.4]);z=check(a,b)
    assert np.max(z['direction_cost'])<1e-12
    results['same_direction_different_speed_no_magnitude_penalty']=True
    rev=make([.4,.3,.2,.1,0]);z=check(a,rev)
    assert np.allclose(z['direction_cost'][1:-1,1:-1],1.)
    results['opposite_direction_distinct']=True
    still=make([.2,.2,.2,.2,.2]);z=check(a,still)
    assert not z['direction_count'].any() and not z['direction_cost'].any()
    results['stationary_direction_undefined_not_motion_agreement']=True
    gap=make([0,.1,.2,.3,.4]);gap['palm_valid'][2,0]=False;gap['centre'][2,0]=np.nan
    check(gap,a);p=c.primitives(gap,('L','R'))
    assert not p['direction_valid'][2,0,0].any()
    assert not p['direction_valid'][1,0,0,1] and not p['direction_valid'][3,0,0,0]
    assert p['valid'][1,0,0] and p['valid'][3,0,0]
    results['gap_not_bridged_normal_points_preserved']=True
    rest=make([0,.1,.2,.3,.4]);rest['source_rest'][:,0]=True;check(rest,a)
    p=c.primitives(rest,('L','R'));assert not p['valid'][:,0,0].any() and p['valid'][:,0,1:].all()
    results['confirmed_hand_rest_excludes_palm_keeps_arm']=True
    right=check(a,a,('R',));assert np.all(right['position_count']==3)
    results['onehand_required_scope_preserved']=True
    # Native gap audit has no path argument: changing alignment cannot remove it.
    gap['angle_valid'][2,0]=False;gap['arm_valid'][2,0]=False
    u=c.query_unknown(gap,('L','R'))
    assert u['query_expected'].sum()==10 and u['query_palm_unknown'].sum()==1
    assert u['query_angle_unknown'].sum()==15 and u['query_arm_unknown'].sum()==2
    results['native_query_gap_retained']=True
    no=make([0,.1,.2]);no['palm_valid'][:]=False;no['arm_valid'][:]=False
    z=check(no,a);assert not z['position_count'].any() and not z['direction_count'].any()
    results['absence_is_zero_added_cost_with_zero_evidence_not_success']=True
    aa=make([0,.1,.2,.3,.4]);aa['core'][:]=False
    zz=check(aa,a)
    orig=c.costs(a,a,('L','R'))
    for key in zz:np.testing.assert_array_equal(zz[key],orig[key])
    results['core_labels_not_cost_inputs']=True
    small=make([0,.001,.002,.003]);z=check(small,a)
    assert not z['direction_count'].any()
    results['noise_floor_suppresses_tiny_direction_only']=True
    return results


if __name__=='__main__':
    print(fixtures())
