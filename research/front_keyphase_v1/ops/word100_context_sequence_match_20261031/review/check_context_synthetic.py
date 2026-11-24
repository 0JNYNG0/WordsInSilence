"""Independent synthetic tests only. Exhaustive paths are the DP oracle."""
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
U=HERE.parent
os.environ['NUMBA_CACHE_DIR']=str(HERE/'synthetic_numba_cache')
path=U/'context_match.py'
before=hashlib.sha256(path.read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('synthetic_context_match',path)
cm=importlib.util.module_from_spec(spec);spec.loader.exec_module(cm)
from numba import njit
# Windows cache temporary filename exceeded the usable path length on first run.
# Compile the exact imported function body without caching; source stays untouched.
cm.window_dp=njit(cache=False)(cm.window_dp.py_func)
checks=[];failures=[]
def check(name,condition,**detail):
    row={'name':name,'pass':bool(condition),**detail};checks.append(row)
    if not condition:failures.append(row)

def exhaustive(cost):
    """Enumerate every monotone path; no DP recurrence reused."""
    n,m=cost.shape;best=float('inf');count=0
    def visit(i,j,last,run,value):
        nonlocal best,count
        if i==n-1 and j==m-1:
            count+=1;best=min(best,value);return
        for di,dj,name,weight in [(1,1,'D',2),(1,0,'Q',1),(0,1,'R',1)]:
            ii,jj=i+di,j+dj
            nr=run+1 if name==last else 1
            if ii>=n or jj>=m or (name!='D' and nr>2):continue
            visit(ii,jj,name,nr,value+weight*float(cost[ii,jj]))
    if n and m:visit(0,0,'D',1,2*float(cost[0,0]))
    return best,count

def inspect_path(cost,start,length,pp):
    n=length;m=cost.shape[1];valid=tuple(pp[0])==(start,0) and tuple(pp[-1])==(start+n-1,m-1)
    valid &= set(pp[:,0])==set(range(start,start+n)) and set(pp[:,1])==set(range(m))
    value=2*float(cost[start,0]);weight=2;last=None;run=0;max_run=0
    for prev,cur in zip(pp,pp[1:]):
        step=tuple(cur-prev);valid &= step in [(1,1),(1,0),(0,1)]
        w=2 if step==(1,1) else 1;value+=w*float(cost[tuple(cur)]);weight+=w
        if step==(1,1):last=None;run=0
        else:
            run=run+1 if step==last else 1;last=step;max_run=max(max_run,run)
    return bool(valid and max_run<=2 and weight==n+m),value,max_run,weight

rng=np.random.default_rng(20261031)
dp_cases=[]
for n,m in itertools.product(range(1,7),repeat=2):
    for typ in ['zero','constant','integer_ties','random']:
        mat=np.zeros((n,m)) if typ=='zero' else np.full((n,m),.375) if typ=='constant' else rng.integers(0,4,(n,m)).astype(float)/3 if typ=='integer_ties' else rng.random((n,m))
        for start in [0,1]:
            full=np.vstack([np.full((start,m),9.),mat,np.full((1,m),8.)])
            oracle,path_count=exhaustive(mat)
            dp,back=cm.window_dp(full,start,n)
            got=float(dp[-1,-1].min())
            same=(np.isinf(oracle) and np.isinf(got)) or np.isclose(got,oracle,atol=1e-12,rtol=0)
            check('weighted_dp_exhaustive',same,n=n,m=m,start=start,matrix_type=typ)
            entry={'n':n,'m':m,'start':start,'matrix_type':typ,'matrix':mat.tolist(),
              'legal_paths_enumerated':path_count,'oracle_weighted_min':None if np.isinf(oracle) else oracle,
              'dp_weighted_min':None if np.isinf(got) else got,'score_divisor':n+m,
              'normalized_cost':None if np.isinf(got) else got/(n+m)}
            if np.isfinite(got):
                pp=cm.recover(full,start,n);valid,recovered,max_run,total_weight=inspect_path(full,start,n,pp)
                check('recovered_path_all_frames_run_limit_weight',valid and np.isclose(recovered,oracle,atol=1e-12,rtol=0),n=n,m=m,start=start,matrix_type=typ)
                entry.update(recovered_path=pp.tolist(),max_axis_repeat_run=max_run,total_weight=total_weight)
                if typ=='constant':check('constant_cost_normalizes_to_constant',np.isclose(got/(n+m),.375,atol=1e-12,rtol=0),n=n,m=m)
            dp_cases.append(entry)

window_cases=[]
for label,mat in [('random_6x4',rng.random((6,4))),('zero_5x3',np.zeros((5,3))),('constant_2x1',np.full((2,1),.2)),('short_query_1x3',np.zeros((1,3))),('empty_query_0x4',np.empty((0,4)))]:
    n,m=mat.shape;lo=max(2,int(np.ceil(.5*m)));hi=min(n,int(np.floor(2*m)));expect=[]
    for st in range(n):
        for ln in range(lo,hi+1):
            if st+ln>n:continue
            best,_=exhaustive(mat[st:st+ln])
            if np.isfinite(best):expect.append((best/(ln+m),st,ln))
    expect.sort();got=cm.windows(mat)
    match=len(got)==len(expect) and all(x[1:]==y[1:] and np.isclose(x[0],y[0],atol=1e-12,rtol=0) for x,y in zip(got,expect))
    check('windows_all_eligible_lengths_offsets_normalization',match,case=label,count=len(got))
    window_cases.append({'case':label,'shape':list(mat.shape),'length_bounds':[lo,hi],'count':len(got),'matches_exhaustive':match})

def scalar_cost(q,r,groups):
    n=len(q[groups[0]][0]);m=len(r[groups[0]][0]);out=np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            group_values=[]
            for g in groups:
                x,xm=q[g];y,ym=r[g];expected=np.flatnonzero(ym[j])
                if not len(expected):continue
                vals=[min(abs(float(x[i,k])-float(y[j,k])),1.) if xm[i,k] else 1. for k in expected]
                group_values.append(sum(vals)/len(vals))
            out[i,j]=sum(group_values)/len(group_values) if group_values else 0.
    return out

groups=['a','b','c'];q={};r={}
for g,d in zip(groups,[1,2,4]):
    q[g]=(rng.uniform(-2,2,(4,d)),rng.random((4,d))>.4)
    r[g]=(rng.uniform(-2,2,(3,d)),rng.random((3,d))>.4)
base=cm.frame_cost(q,r,groups);oracle=scalar_cost(q,r,groups)
check('frame_cost_scalar_missing_penalty_and_reference_group_denominator',np.allclose(base,oracle,atol=1e-12,rtol=0),shape=list(base.shape))
cost_controls=[]
for kind in ['large_finite','nonfinite_masked']:
    qq={g:(x.copy(),m.copy()) for g,(x,m) in q.items()};rr={g:(x.copy(),m.copy()) for g,(x,m) in r.items()}
    for item in [qq,rr]:
        for x,m in item.values():
            x[~m]=1e12 if kind=='large_finite' else np.resize(np.array([np.nan,np.inf,-np.inf]),int((~m).sum()))
    with np.errstate(invalid='ignore'):z=cm.frame_cost(qq,rr,groups)
    ok=np.array_equal(z,base)
    check('invalid_storage_values_ignored',ok,kind=kind)
    cost_controls.append({'kind':kind,'unchanged':ok,'max_difference':float(np.max(np.abs(z-base)))})
for qseen,rseen,value in [(False,True,1.),(True,False,0.),(False,False,0.),(True,True,0.)]:
    x={'a':(np.zeros((3,2)),np.full((3,2),qseen,bool))};y={'a':(np.zeros((4,2)),np.full((4,2),rseen,bool))}
    got=cm.frame_cost(x,y,['a'])
    check('all_missing_or_zero_frame_cost',np.all(got==value) and np.isfinite(got).all(),query_observed=qseen,reference_observed=rseen,expected=value)

def summary(q,r,pp,ids):
    out={};ids=np.asarray(ids);local=pp[np.isin(pp[:,1],ids)]
    for g in q:
        x,qm=q[g];y,rm=r[g];expected=int(rm[ids].sum());comp=np.zeros_like(rm);missing=0;query=set()
        for i,j in local:
            shared=qm[i]&rm[j];comp[j]|=shared;missing+=int((rm[j]&~qm[i]).sum())
            if shared.any():query.add(int(i))
        count=int(comp[ids].sum());available_frames=int(rm[ids].any(1).sum())
        out[g]={'expected':expected,'compared':count,'sufficient':bool(expected and count/expected>=.75 and missing==0 and available_frames>=2 and len(query)>=2)}
    return out

pts=np.arange(11,dtype=np.int64)*100000
vals=np.arange(11,dtype=float)[:,None]/10
original={'g':(vals,np.ones_like(vals,bool))};middle=[{'role':'core_candidate','core':[4,6],'segment_id':'synthetic_mid'}]
trace_cases=[]
for label,qq,rr,segs in [
 ('identity_unique_context',original,original,middle),
 ('query_all_missing',{'g':(vals,np.zeros_like(vals,bool))},original,middle),
 ('reference_all_missing',original,{'g':(vals,np.zeros_like(vals,bool))},middle),
 ('both_all_missing',{'g':(vals,np.zeros_like(vals,bool))},{'g':(vals,np.zeros_like(vals,bool))},middle),
 ('clipped_video_entry',original,original,[{'role':'core_candidate','core':[0,2],'segment_id':'synthetic_edge'}]),
 ('repeated_disjoint_occurrences',{'g':(np.vstack([vals,vals]),np.ones((22,1),bool))},original,middle)]:
    result=cm.trace_match(qq,rr,segs,pts,['g'],summary);a=result['anchors'][0]
    expected='supported_candidate' if label=='identity_unique_context' else 'observation_insufficient' if 'all_missing' in label else 'context_incomplete' if label=='clipped_video_entry' else 'ambiguous_occurrence'
    check('synthetic_trace_support_and_context',a['state']==expected,case=label,state=a['state'],expected=expected)
    if a['best'] is not None:
        pp=np.asarray(a['best']['path']);qw=a['best']['query_window'];rc=a['reference_context']
        check('synthetic_trace_retains_full_selected_query_and_reference_context',set(pp[:,0])==set(range(qw[0],qw[1]+1)) and set(pp[:,1])==set(range(rc[0],rc[1]+1)),case=label)
    trace_cases.append({'case':label,'state':a['state'],'expected_state':expected,'query_core':a['best']['query_core'] if a['best'] else None,
      'reference_context':a['reference_context'],'context_support':a['best']['context_support'] if a['best'] else None,
      'core_supported':a['best']['core_supported'] if a['best'] else None,'disjoint_exact_tie':a.get('disjoint_exact_tie'),
      'requested_context_seconds':a.get('requested_context_seconds'),'clipped_at_video_edge':a.get('clipped_at_video_edge')})

check('source_code_unchanged',hashlib.sha256(path.read_bytes()).hexdigest()==before)
out={'status':'PASS_SYNTHETIC_ALGORITHM_CHECKS' if not failures else 'FAIL_SYNTHETIC_ALGORITHM_CHECKS',
 'author':'Codex agent A; independent exhaustive path/scalar arithmetic oracle; same model family',
 'scope':{'public_data_loaded':False,'model_calls':0,'actual_public_candidate_searches':0,'synthetic_DP_cases':len(dp_cases),
          'tests':len(checks),'failures':len(failures),'numba_compiled_runtime':True,
          'jit_cache_disabled_after_windows_cache_path_failure':True,'same_imported_DP_function_body':True},
 'weighted_path_contract':'origin and diagonal arrival cost weight2; horizontal/vertical arrival weight1. Every legal path weight sum=n+m. At most2 consecutive query-only or reference-only steps; diagonal or opposite axis resets the run.',
 'frame_cost_contract':'Only expected reference dimensions contribute. Shared valid dimensions use clipped absolute difference; missing query dimension costs1. Active reference groups averaged equally. Entirely unobserved reference columns yield numeric0 and must be rejected downstream by evidence support.',
 'DP_cases':dp_cases,'window_cases':window_cases,'frame_cost_perturbations':cost_controls,'trace_cases':trace_cases,
 'checks':checks,'failure_details':failures,'source_sha256':before,'code_unchanged':True,
 'limitations':['Synthetic nonnegative finite cost matrices only; no fit/DEV candidate search or performance estimate.',
   'Recovered path is checked for one optimum; alternative tied internal paths are not enumerated by production recover.',
   'Nonempty reference sequence is the current API precondition; zero-value matrices and all-missing masks are tested. Empty query windows were tested, empty-reference window_dp was not called.',
   'Synthetic summary function independently checks mask coverage and two distinct observed frames; this is not a visual or linguistic validator.',
   'DP normalization/run limits do not establish real speed invariance, temporal identity, bodily correctness or required sign-language order.']}
(HERE/'SYNTHETIC_CHECK.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':out['status'],'tests':len(checks),'failures':failures,'dp_cases':len(dp_cases),'trace_cases':trace_cases},ensure_ascii=False))
if failures:raise SystemExit(1)
