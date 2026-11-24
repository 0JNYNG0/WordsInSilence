"""Saved arithmetic and geometric symmetry audit; never infer a model."""
from pathlib import Path
import json,hashlib,datetime
from collections import Counter
import numpy as np
S=Path(__file__).resolve().parent;UNIT=S.parent;GRAPH=UNIT.parent
def rd(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[]
def ck(name,ok,detail=None):checks.append(dict(name=name,ok=bool(ok),detail=detail))
def pins(p,label):
    for f,h in p.items():ck(label+str(f),Path(f).is_file() and sha(f)==h)
def key(r):return r['seed'],r['clip_id'],r['kind']
def reverse(g):
    r=g.copy();v=g[:,:36].reshape(-1,9,4)
    r[:,:36]=(v[:,::-1]-v[:,-1:]).reshape(-1,36)
    r[:,36:40]=g[:,36:40]+g[:,40:44];r[:,40:44]=-g[:,40:44]
    return r
def verdict(p,d,ok):
    if not ok:return 'observation_unknown'
    if d==1:return 'supported' if p[1]>=.65 and max(p[0],p[2])<.65 else 'uncertain'
    if p[d]>=.65 and p[2-d]<.65 and p[1]<.65:return 'supported'
    if p[2-d]>=.65 and p[d]<.35 and p[1]<.65:return 'opposite_direction'
    if p[1]>=.65:return 'movement_not_established'
    return 'uncertain'
def status(c):
    v=[e['state'] for e in c['evidence']]
    if c['span'] is None:return 'association_unknown'
    if 'difference' in v or 'opposite_direction' in v:return 'difference_evidence'
    return 'supported_features_only' if v and all(s=='supported' for s in v) else 'insufficient_evidence'
def main():
    assert not (S/'VERIFICATION.json').exists()
    cfg=rd(S/'FROZEN.json');tc=rd(S/'TRAINING_COMPLETE.json');ef=rd(S/'evaluation/FROZEN.json');result=rd(S/'evaluation/RESULTS.json')
    meta=rd(UNIT/'INPUTS_COMPLETE.json');rows=meta['rows'];z=np.load(meta['path']);g=z['projected_geometry'];y=z['y']
    for label,values in [('freeze',cfg['pins']),('eval',ef['pins']),('inputs',meta['pins'])]:pins(values,label)
    pins({meta['path']:meta['sha256']},'input_array')
    ck('finite_geometry',np.isfinite(g).all())
    ck('geometry_delta_matches_sequence',np.allclose(g[:,:36].reshape(-1,9,4)[:,-1],g[:,40:44],atol=1e-6))
    invol=float(np.max(np.abs(reverse(reverse(g))-g)));ck('raw_reversal_involution',invol<1e-5,invol)
    ck('reversal_preserves_aux',np.array_equal(reverse(g)[:,44:],g[:,44:]))
    ck('typed_axes',all(r['axis'] in [1,2,3,4] for r in rows))
    fit=[i for i,r in enumerate(rows) if r['split']=='train'];probe=[i for i,r in enumerate(rows) if r['split']=='probe']
    ck('people_split',all(rows[i]['person'] in range(1,8) for i in fit) and all(rows[i]['person'] in [8,9] for i in probe))
    ck('source_split',not {rows[i]['source_sha256'] for i in fit}&{rows[i]['source_sha256'] for i in probe})
    fields=['clip_id','view','a','b','axis','moving','anchor','origin']
    grouped={}
    for i,r in enumerate(rows):grouped.setdefault(tuple(r[k] for k in fields),{}).setdefault(r['reverse'],[]).append(i)
    pairs=[]
    for k,v in grouped.items():
        ck('paired_geometry_rows:'+str(k),set(v)=={False,True} and len(v[False])==len(v[True]))
        if set(v)=={False,True}:pairs+=list(zip(v[False],v[True]))
    aa=np.array([a for a,b in pairs]);bb=np.array([b for a,b in pairs])
    pairerr=float(np.max(np.abs(reverse(g[aa])-g[bb])));ck('stored_actual_geometry_matches_reversal',pairerr<1e-5,pairerr)
    ck('paired_labels_flip',np.array_equal(2-y[aa],y[bb]))
    ck('six_runs',len(tc['rows'])==6 and {(r['condition'],r['seed']) for r in tc['rows']}=={(c,s) for c in cfg['conditions'] for s in [42,43,44]})
    symmetry=[];training_agreement=[]
    for r in tc['rows']:
        pins({r[k]:r[k+'_sha256'] for k in ['model','predictions']},'run')
        lo=np.load(r['predictions'])['logits'];ck('prediction_shape:'+str((r['condition'],r['seed'])),lo.shape==(len(rows),3) and np.isfinite(lo).all())
        ck('completed_step:'+str((r['condition'],r['seed'])),r['log'][-1]['step']==cfg['steps']==800)
        e=float(np.max(np.abs(lo[aa]-lo[bb][:,[2,1,0]])))
        symmetry.append(dict(condition=r['condition'],seed=r['seed'],paired_rows=len(pairs),max_logit_error=e))
        if r['condition']=='body_only_symmetric':ck('stored_logit_equivariance:'+str(r['seed']),e<1e-4,e)
        ck('recorded_context_independence:'+str((r['condition'],r['seed'])),r['context_independence_exact'] is True)
        for split,ix in [('train',fit),('probe',probe)]:training_agreement.append(dict(condition=r['condition'],seed=r['seed'],split=split,n=len(ix),correct=int((lo[ix].argmax(-1)==y[ix]).sum())))
    old={key(r):r for r in rd(UNIT/'evaluation/RESULTS.json')['rows']};out={key(r):r for r in result['rows']}
    ck('all_246_rows',len(out)==len(result['rows'])==246 and set(out)==set(old))
    pk={(p['condition'],*key(p),p['card']):p for p in result['direction_predictions']};ck('unique_predictions',len(pk)==len(result['direction_predictions']))
    used=set();preserved=0;changes=[]
    for kk,r in out.items():
        ck('condition_set:'+str(kk),set(r['conditions'])==set(ef['conditions']))
        for c in ['baseline','local_old','reencoded_projected_geometry']:ck('saved_condition:'+str((c,kk)),r['conditions'][c]==old[kk]['conditions'][c])
        for c in cfg['conditions']:
            a=r['conditions'][c];b=old[kk]['conditions']['local_old'];ck('card_count:'+str((c,kk)),len(a)==len(b))
            for ca,cb in zip(a,b):
                ck('card_invariants:'+str((c,kk,ca['card'])),{k:v for k,v in ca.items() if k not in ['status','evidence']}=={k:v for k,v in cb.items() if k not in ['status','evidence']})
                ck('card_status:'+str((c,kk,ca['card'])),ca['status']==status(ca));ck('evidence_count:'+str((c,kk,ca['card'])),len(ca['evidence'])==len(cb['evidence']))
                for ea,eb in zip(ca['evidence'],cb['evidence']):
                    k=(c,*kk,ca['card'])
                    if ea['task']!='direction' or ea.get('axis')!=2:ck('nonaxis2_exact:'+str(k),ea==eb);preserved+=1;continue
                    ck('axis2_identity:'+str(k),{k:v for k,v in ea.items() if k not in ['state','probabilities','model_scope']}=={k:v for k,v in eb.items() if k not in ['state','probabilities','model_scope']})
                    ck('prediction_exists:'+str(k),k in pk)
                    if k not in pk:continue
                    p=pk[k];used.add(k);v=np.asarray(p['logits']);q=np.exp(v-v.max());q/=q.sum()
                    ck('softmax:'+str(k),np.allclose(q,p['probabilities'],atol=1e-6))
                    ck('state:'+str(k),verdict(q,p['desired'],p['eligible'])==p['state']==ea['state'])
                    ck('same_evidence_probability:'+str(k),p['probabilities']==ea['probabilities'])
                    ck('same_span_query:'+str(k),p['span']==ea['span'] and p['desired']==ea['desired'])
                    if ea['state']!=eb['state']:changes.append(dict(condition=c,seed=kk[0],clip_id=kk[1],kind=kk[2],card=ca['card'],before=eb['state'],after=ea['state']))
    ck('all_predictions_used',used==set(pk));pins({p['source']:p['source_sha256'] for p in pk.values()},'eval_source')
    summary=[]
    for seed in [42,43,44]:
        for c in ef['conditions']:
            for kind in sorted({r['kind'] for r in result['rows']}):
                es=[e for r in result['rows'] if r['seed']==seed and r['kind']==kind for card in r['conditions'][c] for e in card['evidence'] if e['task']=='direction' and e.get('axis')==2]
                summary.append(dict(seed=seed,condition=c,kind=kind,n=len(es),states=dict(Counter(e['state'] for e in es))))
    sk=lambda s:(s['seed'],s['condition'],s['kind'])
    ck('all_summary_denominators',sorted(summary,key=sk)==sorted(result['summary'],key=sk))
    protected=rd(GRAPH/'PROTECTIONS.json');pins(protected,'protected');ck('protected_477',len(protected)==477)
    ck('first_failure_preserved',(S/'FIRST_FAILURE.txt').is_file())
    report=dict(at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),status='PASS' if all(c['ok'] for c in checks) else 'FAIL',checks=len(checks),failed=[c for c in checks if not c['ok']],
        rows=len(out),direction_predictions=len(pk),nonaxis2_preserved=preserved,protected_files=len(protected),reversal_involution_max_error=invol,reversal_paired_geometry_max_error=pairerr,symmetry=symmetry,summary=summary,changes=changes,training_agreement=training_agreement,
        no_model_inference=True,adopted=False,full_G2_pass=False,scope='Saved arithmetic and code contract only, not performance acceptance or human ground truth',
        reviewer='Delegated Codex motion_contract_review_1113 independent script/process',script_sha256=sha(__file__),first_failure_sha256=sha(S/'FIRST_FAILURE.txt'))
    (S/'VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
    print(json.dumps({k:report[k] for k in ['status','checks','failed','rows','direction_predictions','nonaxis2_preserved','protected_files','symmetry']},ensure_ascii=False))
    if report['failed']:raise SystemExit(1)
if __name__=='__main__':main()
