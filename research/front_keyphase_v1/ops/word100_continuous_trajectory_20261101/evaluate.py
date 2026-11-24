from run import *
from collections import defaultdict
def stat(a):
    if not a:return dict(n=0,min=None,max=None,mean=None,median=None)
    s=sorted(a);n=len(s)
    return dict(n=n,min=s[0],max=s[-1],mean=sum(s)/n,median=(s[(n-1)//2]+s[n//2])/2)
def main():
    checkpins();inp=read(U/'INPUTS.json');m=read(U/'RESULT_MANIFEST.json');eps=inp['atol'];rows=[];fail=[];groups=defaultdict(list)
    for rr in m['results']:
        assert sha(rr['path'])==rr['sha256'];d=read(rr['path']);s=read(d['source_record']);c=d['candidate'];old=d['old_discrete_distance']
        assert sha(d['source_record'])==d['source_sha256']
        complete=c['complete'];lo=c['lower_bound'];hi=c['upper_bound']
        numeric=bool(complete and c.get('certified') and c.get('converged') and lo is not None and hi is not None)
        if complete and not numeric:fail.append(dict(id=d['id'],error='uncertified_or_unconverged',status=c['status']))
        if numeric:
            if not (0<=lo<=hi and hi-lo<=eps*1.01):fail.append(dict(id=d['id'],error='interval_precision'))
            if hi>old+eps*1.01:fail.append(dict(id=d['id'],error='exceeds_saved_discrete_upper_bound'))
        elif not complete and (lo is not None or hi is not None):fail.append(dict(id=d['id'],error='missing_promoted'))
        row={k:d[k] for k in ['id','kind','role','query','reference','component','control','parent_id','source_record']}
        row.update(complete=complete,numeric_certified=numeric,old_distance=old,lower=lo,upper=hi,zero_test=bool(c.get('zero_distance_certified')),
            upper_within_numerical_tolerance=numeric and hi<=eps,
            strict_numeric_reduction=numeric and hi<old-eps,
            relative_reduction=(old-hi)/old if numeric and old>eps else None,
            edit_outside_parent=bool(s.get('edit',{}).get('relevant_changed_outside_scope',[])),
            semantic_verdict=None,rotation_or_wrist_bend_verdict=None)
        rows.append(row);groups[(row['kind'],row['role'],row['control'],row['component'])].append(row)
    summary=[]
    for key,rs in sorted(groups.items(),key=lambda x:str(x[0])):
        cs=[r for r in rs if r['numeric_certified']];oc=sum(r['complete'] for r in rs)
        summary.append(dict(kind=key[0],role=key[1],control=key[2],component=key[3],opportunities=len(rs),complete=oc,unknown=len(rs)-oc,numeric_certified=len(cs),
            old_distance=stat([r['old_distance'] for r in cs]),continuous_upper=stat([r['upper'] for r in cs]),
            reduced=sum(r['strict_numeric_reduction'] for r in cs),zero_test=sum(r['zero_test'] for r in cs),
            within_numerical_tolerance=sum(r['upper_within_numerical_tolerance'] for r in cs),
            edit_outside_parent=sum(r['edit_outside_parent'] for r in rs)))
    controls=[]
    for kind in sorted({r['control'] for r in rows if r['kind']=='control'}):
        rs=[r for r in rows if r['control']==kind];cs=[r for r in rs if r['numeric_certified']];missing=[x for x in m['missing'] if x['control']==kind];oc=sum(r['complete'] for r in rs)
        controls.append(dict(control=kind,opportunities=len(rs)+len(missing),computed=len(rs),missing=len(missing),complete=oc,unknown=len(rs)-oc,numeric_certified=len(cs),
            zero_test=sum(r['zero_test'] for r in cs),within_numerical_tolerance=sum(r['upper_within_numerical_tolerance'] for r in cs),
            positive_above_numerical_tolerance=sum(r['lower']>eps for r in cs),
            numerically_reduced=sum(r['strict_numeric_reduction'] for r in cs),edit_outside_parent=sum(r['edit_outside_parent'] for r in rs)))
    nat=[r for r in rows if r['kind']=='natural'];cn=[r for r in nat if r['complete']]
    result=dict(status='COMPLETE_FIXED_CONTINUOUS_GEOMETRY_DIAGNOSTIC',failures=fail,failure_count=len(fail),
        natural=dict(opportunities=len(nat),complete=len(cn),unknown=len(nat)-len(cn),
            numerically_reduced=sum(r['strict_numeric_reduction'] for r in cn),within_numerical_tolerance=sum(r['upper_within_numerical_tolerance'] for r in cn)),
        controls=controls,summary=summary,rows=rows,missing=m['missing'],atol=eps,
        decision=dict(technical_pass=not fail,production_adopted=False,educational_accuracy=None,recognition_accuracy=None,
            smaller_continuous_distance_not_accuracy_gain=True,rotation_or_wrist_bend_detection_validated=False),
        timing=dict(wall_seconds=m['elapsed_wall_seconds'],geometry_seconds=m['geometry_compute_seconds'],end_to_end_user_latency_measured=False))
    new(U/'EVALUATION.json',result)
    print('natural',result['natural'],'failures',len(fail));print(json.dumps(controls,ensure_ascii=False))
if __name__=='__main__':main()
