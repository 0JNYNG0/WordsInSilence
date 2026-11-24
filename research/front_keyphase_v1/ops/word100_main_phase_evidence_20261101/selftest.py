"""Synthetic-only numerical contract tests for the single bounded candidate."""
from pathlib import Path
import copy
import hashlib
import json
import sys
import numpy as np
sys.dont_write_bytecode=True
import motion_cost as M


def sample(n=6,step=(0.,0.),dt_us=100000):
    x=np.zeros((n,69),dtype=np.float64)
    # Nondegenerate measured arms for checking the unchanged geometry groups.
    x[:,4],x[:,5]=-.2,.2;x[:,32],x[:,33]=.2,.2
    x[:,2],x[:,3]=-.2,0.;x[:,30],x[:,31]=.2,0.
    x[:,0]=-0.2;x[:,28]=.2
    x[:,0:2]+=np.arange(n)[:,None]*np.asarray(step)
    x[:,28:30]+=np.arange(n)[:,None]*np.asarray(step)
    x[:,56:58]=x[:,28:30]-x[:,0:2]
    x[:,58]=np.linalg.norm(x[:,56:58],axis=1)
    return dict(x=x,mask=np.ones_like(x,dtype=bool),frame_id=np.arange(n,dtype=np.int64),
                timestamp_us=np.arange(n,dtype=np.int64)*dt_us)


def brute_paths(n,m):
    def walk(path):
        i,j=path[-1]
        if i==n-1 and j==m-1:
            yield path
            return
        for di,dj in [(1,1),(1,0),(0,1)]:
            a,b=i+di,j+dj
            if a<n and b<m and abs(a/max(n-1,1)-b/max(m-1,1))<=.25:
                yield from walk(path+[[a,b]])
    return list(walk([[0,0]]))


def run():
    checks=[]
    def ck(name,ok):
        if not bool(ok):raise AssertionError(name)
        checks.append(name)
    def close(name,a,b,tol=1e-12):ck(name,np.allclose(a,b,atol=tol,rtol=0))
    def raises(name,f):
        try:f()
        except ValueError:ck(name,True)
        else:raise AssertionError(name)

    q=sample();v=M.velocity(q)
    for side in ['L','R']:
        ck('first frame unknown '+side,not v['hands'][side]['valid'][0])
        ck('consecutive zero observed '+side,v['hands'][side]['valid'][1:].all())
        ck('zero not missing '+side,v['hands'][side]['observed_zero'].sum()==5)
        for mode in M.MODES:
            values,masks=M.transform(v['hands'][side]['values'],v['hands'][side]['valid'],mode)
            ck('zero transformed valid '+mode+side,masks[1:].all() and not masks[0].any())
            close('zero transformed values '+mode+side,values,0.)
    ck('no linguistic hold',not v['semantic_hold_label'])

    def transformed(vec,mode,known=True):
        return M.transform(np.asarray([vec],dtype=float),np.asarray([known],bool),mode)[0][0]
    tiny=1e-10
    uplus=transformed([0,tiny],'unit');uminus=transformed([0,-tiny],'unit')
    bplus=transformed([0,tiny],'bounded');bminus=transformed([0,-tiny],'bounded')
    close('tiny opposite unit remains .5 group difference',np.abs(uplus-uminus).mean(),.5)
    close('tiny opposite bounded tends to zero',np.abs(bplus-bminus).mean(),tiny/(1+tiny)/2)
    ck('tiny bounded change retained positive',np.abs(bplus-bminus).mean()>0)
    zero=transformed([0,0],'bounded');fast=transformed([0,1],'bounded')
    close('one zero one moving bounded formula',np.abs(zero-fast).mean(),.125)
    ck('one zero one moving not suppressed',np.abs(zero-fast).max()>0)
    bigplus=transformed([0,1e6],'bounded');bigminus=transformed([0,-1e6],'bounded')
    ck('large opposing directions retain near-unit disagreement',.49<np.abs(bigplus-bigminus).mean()<.5)
    close('same direction different speed unit equal',transformed([0,1],'unit'),transformed([0,2],'unit'))
    ck('same direction different speed bounded differs',not np.array_equal(transformed([0,1],'bounded'),transformed([0,2],'bounded')))
    close('bounded formula arbitrary vector',transformed([3,4],'bounded'),[.25,1/3])
    ck('bounded norm finite less than .5',np.linalg.norm(transformed([3,4],'bounded'))<.5)

    moving=sample(step=(.02,-.03));vel=M.velocity(moving)
    for side in ['L','R']:close('velocity from positions/PTS '+side,vel['hands'][side]['values'][1:],[.2,-.3])
    altered=copy.deepcopy(moving);altered['x'][:,61:69]=1234.;altered['mask'][:,61:69]=False
    for mode in M.MODES:
        for side in ['L','R']:
            close('stored old direction ignored '+mode+side,M.features(moving,mode)[side+'_direction'][0],M.features(altered,mode)[side+'_direction'][0])
    before=copy.deepcopy(moving)
    unit_cost,ug=M.cost(moving,q,'unit',return_groups=True)
    bounded_cost,bg=M.cost(moving,q,'bounded',return_groups=True)
    for name in M.GROUPS:
        if not name.endswith('_direction'):close('same other11 '+name,ug[name],bg[name],0.)
    close('sum group means unit',unit_cost,sum(ug.values())/13)
    close('sum group means bounded',bounded_cost,sum(bg.values())/13)
    close('cost wrapper equivalence',M.bounded_cost(moving,q),bounded_cost)
    for k in moving:ck('no input mutation '+k,np.array_equal(moving[k],before[k]))
    result=M.compare(moving,moving)
    ck('new U control explicitly distinguished',result['unit_is_new_sidecar_control_not_old_saved_C'])
    for mode in M.MODES:
        ck('identity synthetic known path '+mode,result[mode]['path']==[[i,i] for i in range(len(moving['x']))])
        ck('identity not correctness '+mode,not result[mode]['physical_timing_verified'])
    ck('no whole sign verdict',result['whole_sign_verdict'] is None)

    missing=copy.deepcopy(q);missing['mask'][:,0:2]=False
    for mode in M.MODES:
        _,g=M.cost(missing,q,mode,True)
        close('known ref/missing query penalty1 '+mode,g['L_direction'][2,2],1.)
        _,g=M.cost(q,missing,mode,True)
        close('unknown ref neutral .5 '+mode,g['L_direction'][2,2],.5)
        _,g=M.cost(q,q,mode,True)
        close('observed zero vs zero cost0 '+mode,g['L_direction'][2,2],0.)
    gap=sample();gap['frame_id'][3:]+=1
    ck('nonconsecutive ID breaks velocity',not M.velocity(gap)['hands']['L']['valid'][3])
    ck('after gap new consecutive pair valid',M.velocity(gap)['hands']['L']['valid'][4])
    gap=sample();gap['timestamp_us'][3:]+=100000
    ck('dt above .15 invalid',not M.velocity(gap)['hands']['L']['valid'][3])
    edge=sample(dt_us=150000)
    ck('dt at .15 valid',M.velocity(edge)['hands']['L']['valid'][1:].all())
    gap=sample();gap['mask'][2,0]=False
    ck('missing current and predecessor retained',M.velocity(gap)['hands']['L']['valid'].tolist()==[False,True,False,False,True,True])
    ck('other hand remains independent',M.velocity(gap)['hands']['R']['valid'][1:].all())
    gap=sample();gap['x'][2,0]=np.nan
    ck('nonfinite wrist unknown',not M.velocity(gap)['hands']['L']['valid'][2:4].any())
    gap=sample();gap['mask'][2,11]=False
    ck('finger-only missing does not remove velocity',M.velocity(gap)['hands']['L']['valid'][1:].all())
    smallshape=copy.deepcopy(q);smallshape['x'][1:6,39]+=.1
    before=copy.deepcopy(smallshape)
    M.compare(smallshape,q)
    close('5frame shape peak untouched',smallshape['x'][1:6,39]-q['x'][1:6,39],.1)
    ck('5frame shape masks untouched',np.array_equal(smallshape['mask'],before['mask']))
    for mode in M.MODES:
        ck('source 14shape kept independent '+mode,M.features(smallshape,mode)['R_shape'][0].shape==(6,14))

    # Independent enumeration checks exact objective, original tie convention
    # on zero costs, and preservation of both full axes.
    for n,m in [(3,3),(4,4),(4,5)]:
        paths=brute_paths(n,m)
        for k,cost in enumerate([np.zeros((n,m)),np.eye(n,m),np.arange(n*m).reshape(n,m)%4]):
            got=M.dtw(cost,True)
            vals=[sum(float(cost[i,j]) for i,j in p) for p in paths];best=min(vals)
            close(f'exhaustive optimum {n}/{m}/{k}',got['objective_sum'],best,0.)
            close(f'representative cost {n}/{m}/{k}',sum(float(cost[i,j]) for i,j in got['path']),best,0.)
            ck(f'exact ambiguity {n}/{m}/{k}',got['exact_optimal_path_count_capped']==min(2,sum(v==best for v in vals)))
            ck(f'full axes {n}/{m}/{k}',{i for i,j in got['path']}==set(range(n)) and {j for i,j in got['path']}==set(range(m)))
    ck('zero cost exact C tie diagonal',M.dtw(np.zeros((4,4)))==[[i,i] for i in range(4)])
    # One explicit hold/timewarp sequence with a zero-cost known mapping.
    qseq=np.array([0.,0.,1.,2.,2.]);rseq=np.array([0.,1.,2.,2.])
    p=M.dtw(np.abs(qseq[:,None]-rseq[None]))
    ck('known timewarp matches the observed sequence',all(qseq[i]==rseq[j] for i,j in p))

    invalid=sample();invalid['phase']='truth'
    raises('truth metadata rejected',lambda:M.velocity(invalid))
    invalid=sample();invalid['mask']=invalid['mask'].astype(int)
    raises('boolean mask required',lambda:M.velocity(invalid))
    invalid=sample();invalid['timestamp_us'][2]=invalid['timestamp_us'][1]
    raises('nonpositive time interval rejected',lambda:M.velocity(invalid))
    raises('mode name exact',lambda:M.features(q,'weighted'))
    raises('unknown cost matrix rejected',lambda:M.dtw(np.array([[np.nan]])))
    raises('impossible band does not shrink source',lambda:M.dtw(np.zeros((1,4))))
    return dict(status='PASS',checks=len(checks),check_names=checks,public_data_loaded=False,
                new_paths_on_public_data=0,new_inference=0,new_training=0,
                limits=['Same-direction different speed produces different bounded costs.',
                        'Magnitude damping does not identify jitter or semantic holds.',
                        'New U uses observed-zero validity and is not identical to the old saved C baseline.',
                        'Synthetic optimality and immutability are not natural or educational accuracy.'])


if __name__=='__main__':
    result=run()
    result['motion_cost_sha256']=hashlib.sha256(Path(M.__file__).read_bytes()).hexdigest()
    result['selftest_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    p=Path(__file__).with_name('SELFTEST.json')
    with p.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ['status','checks','motion_cost_sha256']}))
