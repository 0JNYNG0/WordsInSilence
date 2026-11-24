"""Prospectively fixed LP/rational checks; synthetic inputs only."""
from pathlib import Path
import json,sys,hashlib,traceback
sys.dont_write_bytecode=True
import numpy as np
from independent_math import independent_lp_distance,rational_decision,discrete_enclosure

U=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(U))


def fixtures():
    cases=[('one_point',[[0.,0.]],[[1.,2.]]),
           ('point_vs_interior_peak',[[0.,0.]],[[0.,0.],[1.,1.],[0.,0.]]),
           ('zero_segment',[[0,0],[0,0]],[[0,0],[0,0]]),
           ('collinear',[[0,0],[.5,0],[1,0]],[[0,0],[1,0]]),
           ('out_and_back_same_polyline',[[0,0],[1,0],[2,0],[0,0]],[[0,0],[2,0],[1,0],[0,0]]),
           ('turn_removed',[[0,0],[1,1],[2,0]],[[0,0],[2,0]]),
           ('noncollinear_order',[[0,0],[1,0],[0,1],[0,0]],[[0,0],[0,1],[1,0],[0,0]]),
           ('endpoint_difference',[[0,0],[1,1]],[[0,0],[1,2]]),
           ('tiny_positive',[[0,0],[1,0]],[[0,1e-8],[1,1e-8]]),
           ('duplicate_pause',[[0,0],[0,0],[1,1],[1,1]],[[0,0],[1,1]]),
           ('reverse_not_symmetric',[[0,0],[1,1],[2,0]],[[2,0],[1,1],[0,0]])]
    rng=np.random.default_rng(20261101)
    for i in range(64):
        n,m=2+i%4,2+(i//4)%4
        p=rng.integers(-8,9,(n,2))/8
        q=rng.integers(-8,9,(m,2))/8
        if i%8==0:p[1]=p[0]
        if i%8==1:q[1]=q[0]
        cases.append(('fixed_dyadic_'+str(i),p,q))
    return cases


def main():
    import continuous
    results=[];checks=[];failures=[]
    def check(name,value,case):
        checks.append({'name':name,'case':case,'passed':bool(value)})
        if not value:failures.append(checks[-1])
    try:
        for name,p,q in fixtures():
            p,q=np.asarray(p,float),np.asarray(q,float)
            lp,chains=independent_lp_distance(p,q)
            # LP solver has finite feasibility tolerances; tiny-positive case
            # is checked by exact rational decisions separately, not rounded LP.
            check('LP_upper_reachable',rational_decision(p,q,lp+1e-6),name)
            if lp>1e-6:check('LP_lower_unreachable',not rational_decision(p,q,lp-1e-6),name)
            dl,du=discrete_enclosure(p,q,4)
            check('LP_within_subdivision_bounds',dl-1e-8<=lp<=du+1e-8,name)
            c=continuous.compare(p,q,np.ones_like(p,dtype=bool),np.ones_like(q,dtype=bool),atol=1e-7,max_iterations=60)
            lo,hi=c['lower_bound'],c['upper_bound']
            check('producer_bracket_contains_independent_LP',lo-1e-8<=lp<=hi+1e-8,name)
            check('producer_upper_exact_feasible',rational_decision(p,q,hi),name)
            if c.get('zero_distance_certified'):
                check('producer_zero_exact',rational_decision(p,q,0),name)
            else:
                check('producer_zero_not_feasible',not rational_decision(p,q,0),name)
            if lo>0:
                endpoint=(np.max(np.abs(p[:,None,:]-q[None,:,:])) if len(p)==1 or len(q)==1 else max(np.max(abs(p[0]-q[0])),np.max(abs(p[-1]-q[-1]))))
                check('producer_lower_infeasible_or_endpoint',not rational_decision(p,q,lo) or abs(lo-endpoint)<=1e-14,name)
            check('producer_precision',0<=lo<=hi and hi-lo<=1e-7,name)
            results.append({'case':name,'lp':lp,'cell_chains_enumerated':chains,'lower':lo,'upper':hi,'subdivision_lower':dl,'subdivision_upper':du,'zero_test':bool(c.get('zero_distance_certified'))})
        p=np.array([[0.,0.],[1.,0.]])
        check('exact_zero_rejects_tiny_positive',not rational_decision(p,p+[0,1e-8],0),'tiny_positive')
    except Exception:
        failures.append({'exception':traceback.format_exc()})
    report={'schema':'wis_independent_continuous_math_selftest_v1','status':'PASS' if not failures else 'FAIL','checks':len(checks),'failure_count':len(failures),'failures':failures,'results':results,'check_details':checks,
      'method':'Independent cell-chain crossing LP plus exact rational boundary decisions and discrete subdivision enclosure; no producer decision or DP reused.',
      'rng_seed':20261101,'public_data_executed':False,'model_inference':0,'training':0,
      'pins':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [U/'continuous.py',Path(__file__),Path(__file__).with_name('independent_math.py')]}}
    out=U/'review/INDEPENDENT_MATH_SELFTEST.json'
    with out.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:report[k] for k in ['status','checks','failure_count','failures']}))
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
