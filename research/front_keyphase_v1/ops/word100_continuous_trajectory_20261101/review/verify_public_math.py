"""Verify saved public candidate bounds only; no producer/model/path call."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import sys,json,gzip,hashlib,time,traceback
sys.dont_write_bytecode=True
import numpy as np
from independent_math import rational_decision,rational_curve

U=Path(__file__).resolve().parents[1]
def read(p):
    p=Path(p)
    return json.loads(gzip.decompress(p.read_bytes())) if p.suffix=='.gz' else json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    start=time.perf_counter();checks=Counter();fail=[];results=[];counts=Counter()
    def check(name,ok,identity):
        checks[name]+=1
        if not ok:fail.append({'check':name,'id':identity})
    try:
        inputs=read(U/'INPUTS.json');manifest=read(U/'RESULT_MANIFEST.json');frozen=read(U/'FROZEN.json')
        source_index={r['id']:r for r in inputs['records']}
        check('record_id_scope',len(manifest['results'])==2825 and {r['id'] for r in manifest['results']}==set(source_index),'manifest')
        check('missing_opportunities_unchanged',manifest['missing']==inputs['missing'] and len(manifest['missing'])==10,'manifest')
        for index,r in enumerate(manifest['results']):
            check('saved_result_hash',sha(r['path'])==r['sha256'],r['id'])
            d=read(r['path']);c=d['candidate'];source=read(d['source_record']);s=source_index[r['id']]
            check('source_identity_hash',d['source_record']==s['path'] and d['source_sha256']==s['sha256']==sha(d['source_record']),r['id'])
            check('frozen_execution_pointer',d['frozen_sha256']==sha(U/'FROZEN.json'),r['id'])
            p=np.asarray(source['query_plot']['values'],float);q=np.asarray(source['reference_plot']['values'],float)
            pm=np.asarray(source['query_plot']['mask'],bool);qm=np.asarray(source['reference_plot']['mask'],bool)
            complete=bool(p.size and q.size and pm.all() and qm.all())
            check('complete_exact_source_masks',c['complete']==complete==s['old_complete'],r['id'])
            check('old_discrete_value_preserved',d['old_discrete_distance']==s['old_distance'],r['id'])
            check('other_hand_features_not_decided',d['handshape_orientation_wrist_bend_verdict'] is None and d['educational_verdict'] is None and d['position_distance_is_not_whole_handshape_agreement'],r['id'])
            counts['complete' if complete else 'unknown']+=1
            if not complete:
                check('unknown_bracket_null',c['lower_bound'] is None and c['upper_bound'] is None,r['id'])
                continue
            lo,hi=c['lower_bound'],c['upper_bound'];cert_zero=bool(c.get('zero_distance_certified'))
            numeric=bool(c.get('certified') and c.get('converged') and lo is not None and hi is not None)
            check('complete_numerically_certified',numeric,r['id'])
            if not numeric:
                results.append({'id':r['id'],'result':r['path'],'status':'NUMERICAL_FAILURE_RETAINED','candidate_status':c['status']})
                continue
            check('numeric_interval_width',0<=lo<=hi and hi-lo<=inputs['atol'],r['id'])
            check('continuous_below_discrete',hi<=d['old_discrete_distance']+inputs['atol'],r['id'])
            rp,rq=rational_curve(p),rational_curve(q)
            endpoint=(max(max(abs(a-b) for a,b in zip(x,y)) for x in rp for y in rq) if len(rp)==1 or len(rq)==1 else max(max(abs(a-b) for a,b in zip(rp[0],rq[0])),max(abs(a-b) for a,b in zip(rp[-1],rq[-1]))))
            upper_ok=rational_decision(p,q,hi)
            endpoint_lower=F(float(lo))<=endpoint
            lower_reachable=None if endpoint_lower else rational_decision(p,q,lo)
            lower_ok=endpoint_lower or not lower_reachable
            check('upper_exact_rational_feasible',upper_ok,r['id'])
            check('lower_exact_rational_valid',lower_ok,r['id'])
            # A strictly positive proven lower bound already certifies nonzero.
            zero=rational_decision(p,q,0) if cert_zero or lo==0 else False
            check('zero_claim_exactly_matches',cert_zero==zero,r['id'])
            counts['zero_certified']+=cert_zero
            results.append({'id':r['id'],'result':r['path'],'source':d['source_record'],'lower':lo,'upper':hi,'upper_exact_feasible':upper_ok,'lower_valid':lower_ok,'lower_from_endpoint_bound':endpoint_lower,'zero_exact':zero})
            if (index+1)%100==0:print(index+1,'/',len(manifest['results']),flush=True)
        check('complete_unknown_totals',counts['complete']==1417 and counts['unknown']==1408,'manifest')
        for name in ['continuous.py','run.py','evaluate.py']:
            check('runtime_code_still_frozen',sha(U/name)==frozen['pins'][str(U/name)],name)
    except Exception:
        fail.append({'exception':traceback.format_exc()})
    report={'schema':'wis_continuous_trajectory_independent_math_result_v1','status':'PASS' if not fail else 'FAIL','checks':dict(checks),'check_count':sum(checks.values()),'failure_count':len(fail),'failures':fail,'counts':dict(counts),'results':results,'elapsed_seconds':time.perf_counter()-start,
      'method':'Independently precomputed exact rational free-edge intervals and reachability validate every saved complete upper bound and either infeasible lower bound or exact endpoint lower bound. The producer is never imported or executed. Separate cell-chain LP synthetic verification checks the method independently of the interval recurrence.',
      'claims':'Numeric geometry contract only; no semantic, rotation, wrist-flexion, educational correctness or whole-sign judgement.',
      'actions':{'producer_comparisons':0,'new_alignment_paths':0,'old_scores_recalculated':0,'model_inference':0,'training':0},
      'pins':{str(p):sha(p) for p in [U/'INPUTS.json',U/'FROZEN.json',U/'RESULT_MANIFEST.json',U/'review/INDEPENDENT_MATH_SELFTEST.json',Path(__file__),Path(__file__).with_name('independent_math.py')] if p.exists()}}
    target=U/'review/MATH_RESULT_VERIFICATION.json'
    with target.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:report[k] for k in ['status','check_count','failure_count','counts','elapsed_seconds','failures']}),flush=True)
    if fail:raise SystemExit(1)

if __name__=='__main__':main()
