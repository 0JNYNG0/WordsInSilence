"""Saved-result contracts, symmetric PTS order audit, and failure decomposition.

No forward pass, optimizer, encoder or old-result mutation. Provenance is used
only AFTER selection for diagnostics; it never changes a score or chosen span.
"""
import ast, collections, itertools, sys, time
import numpy as np
from prepare import U, H, G, O, W, L, read, write, sha, check_pins
sys.path.insert(0,str(W))
from local_rank_core import F, AX
import torch
E=U/'evaluation'
def key(r):return r['seed'],r['clip_id'],r['kind']

def main():
    assert not (U/'POSTTRAIN_VERIFICATION.json').exists()
    cfg=read(U/'FROZEN.json');tc=read(U/'TRAINING_FROZEN.json');ec=read(E/'FROZEN.json')
    check_pins(cfg['pins']);check_pins(tc['pins']);check_pins(ec['pins'])
    train=read(U/'TRAINING_COMPLETE.json');data=read(U/'INPUTS_COMPLETE.json')
    result=read(E/'COMPARISON.json');old=read(L/'evaluation/COMPARISON.json')
    checks=collections.Counter()
    def ck(condition,label):
        assert bool(condition),label
        checks[label]+=1
    models={};fit=[]
    oldmeta=read(L/'INPUTS_COMPLETE.json'); oldbagindex={r['path']:i for i,r in enumerate(oldmeta['bags'])}
    for run in train['rows']:
        ck(run['steps']==1200,'exact_training_steps');ck(sha(run['model'])==run['model_sha256'],'new_model_pin')
        state=torch.load(run['model'],map_location='cpu',weights_only=True)
        before=torch.load(run['initial_model'],map_location='cpu',weights_only=True)
        ck(all(torch.equal(state[k],before[k])for k in ['mean','std']),'normalization_unchanged')
        ck(any(not torch.equal(state[k],before[k])for k in state if k.startswith('head.')),'new_head_changed')
        models[run['seed']]=state
        ck(sha(run['fit_predictions'])==run['fit_predictions_sha256'],'fit_prediction_pin')
        pred=np.load(run['fit_predictions']);index=read(__import__('pathlib').Path(run['model']).parent/'FIT_PREDICTION_INDEX.json')
        priorrun=next(r for r in read(L/'TRAINING_COMPLETE.json')['rows']if r['condition']=='local_full'and r['seed']==run['seed'])
        oldpred=np.load(priorrun['predictions'])
        for r in index:
            z=np.load(r['path']);y=z['y'];j=F.index(r['family']);score=pred[r['key']][:,j]
            ck(np.isfinite(score).all()and len(score)==len(y),'fit_saved_prediction_shape')
            origin=r['key'].split('_')[0]
            row=dict(seed=run['seed'],origin=origin,family=r['family'],bag=r['path'],
                positive=int(y.sum()),negative=int((~y).sum()),top_is_full_positive=bool(y[int(score.argmax())]),
                full_minus_negative_margin=float(score[y].max()-score[~y].max()))
            if origin=='original':
                os=oldpred[f"logits_{oldbagindex[r['path']]}"][:,j]
                row['old_top_is_full_positive']=bool(y[int(os.argmax())])
                row['old_margin']=float(os[y].max()-os[~y].max())
            fit.append(row)
    ck(set(models)=={42,43,44},'all_seeds_preserved')
    rm={key(r):r for r in result['rows']};om={key(r):r for r in old['rows']}
    rp={key(r):r for r in result['predictions']};op={key(r):r for r in old['predictions']}
    ck(set(rm)==set(om)and len(rm)==246,'same246_rows')
    # Exact canonical PTS function, loaded without running its audit entrypoint.
    boundary=H/'null_candidate_learning/shared_boundary_audit.py'
    tree=ast.parse(boundary.read_text(encoding='utf-8'));fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='temporal_relation')
    namespace={};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(boundary),'exec'),namespace)
    temporal_relation=namespace['temporal_relation'];pts_records=[];diagnostics=[];sources={};features={}
    ext={r['clip_id']:r for r in read(G/'reviews/sequence_target/MOTION_EXTENTS.json')['rows']}
    for k,row in rm.items():
        prior=om[k];p=rp[k];po=op[k]
        ck(all(row['conditions'][c]==prior['conditions'][c]for c in prior['conditions']),'all_old_conditions_exact')
        ck(sha(p['path'])==p['sha256'],'new_eval_prediction_pin')
        a=np.load(p['path']);oa=np.load(po['path'])
        if p['source']not in sources:
            ck(sha(p['source'])==p['source_sha256'],'source_pin');sources[p['source']]=dict(np.load(p['source']))
        source=sources[p['source']];ts=source['ts'];qr=source['z'][:,292:392].reshape(-1,10,10)
        ck(np.all(np.diff(ts)>0),'source_pts_monotone')
        newcards=row['conditions']['splice_rank']['cards'];oldcards=prior['conditions']['local_full']['cards']
        for i,(c,oc)in enumerate(zip(newcards,oldcards)):
            sp=a[f'spans_{i}'];presence=a[f'presence_{i}'];score=a[f'splice_rank_logits_{i}']
            ck(np.array_equal(sp,oa[f'spans_{i}'])and np.array_equal(presence,oa[f'presence_{i}']),'spans_gate_exact')
            if c['family']not in [2,3]:
                ck(c==oc and np.array_equal(score,oa[f'local_full_logits_{i}']),'other_family_output_exact')
                continue
            masked=np.where(presence>=0,score,-np.inf)
            chosen=c['chosen']
            if len(masked)and np.isfinite(masked).any():
                at=int(masked.argmax());aa,bb=map(int,sp[at]);axis=AX[c['family']]*2
                observed=np.arange(aa,bb+1);observed=observed[qr[observed,axis,8]>.5]
                ck(chosen['span']==[aa,bb]and chosen['score']==float(masked[at]),'original_argmax_selector')
                ck(chosen['body_observed']==observed.tolist()and chosen['hand_qualified']==observed[qr[observed,axis,9]>.5].tolist(),'qualifiers_exact')
                if'truth'in c:
                    ck(c['truth']['all_witness']==(set(c['truth']['witness'])<=set(observed))and c['truth']['wrong_points']==len(set(c['truth']['wrong_witness'])&set(observed)),'witness_arithmetic')
            else:ck(chosen is None,'no_gate_admission_held')
            if row['kind']=='dynamic_complete_blocks'and'truth'in c:
                cuts=[s['query_right']for s in ext.get(row['clip_id'],{}).get('splice_joins',[])if s['synthetic_discontinuity']]
                valid=qr[:,AX[c['family']]*2,8]>.5;w=set(c['truth']['witness']);bad=set(c['truth']['wrong_witness'])
                good=np.array([w<={j for j in range(aa,bb+1)if valid[j]}and not bad&{j for j in range(aa,bb+1)if valid[j]}for aa,bb in sp])
                admitted=np.flatnonzero(good&(presence>=0));best=int(admitted[np.argmax(score[admitted])])if len(admitted)else None
                d=dict(seed=k[0],clip_id=k[1],kind=k[2],card=c['card'],selected=chosen['span']if chosen else None,
                    old_selected=oc['chosen']['span']if oc['chosen']else None,all_witness=c['truth']['all_witness'],
                    wrong_points=c['truth']['wrong_points'],selected_crosses_join=bool(chosen and any(chosen['span'][0]<cut<=chosen['span'][1]for cut in cuts)),
                    valid_witness_alternative=best is not None,best_valid_span=sp[best].tolist()if best is not None else None)
                if chosen and best is not None:
                    ci=int(np.flatnonzero((sp==np.array(chosen['span'])).all(1))[0])
                    d['best_valid_rank']=int((masked>masked[best]).sum()+1)
                    d['selected_minus_valid_score']=float(masked[ci]-masked[best])
                    if not c['truth']['all_witness']or c['truth']['wrong_points']:
                        fp=p['feature_path']
                        if fp not in features:
                            f=np.load(fp);features[fp]=(f['x'],{tuple(v):i for i,v in enumerate(f['spans'])})
                        ff,lu=features[fp];delta=(ff[lu[tuple(sp[ci])]]-ff[lu[tuple(sp[best])]])/models[k[0]]['std'].numpy()
                        d['standardized_feature_rms_to_valid']=float(np.sqrt(np.mean(delta**2)))
                        d['temporal_third_and_std_rms']=[float(np.sqrt(np.mean(z**2)))for z in delta.reshape(4,290)]
                        def wrist_fraction(span):
                            aa,bb=span;xy=source['x'][aa:bb+1,12:14];steps=np.linalg.norm(np.diff(xy,axis=0),axis=1)
                            return float(steps.max()/steps.sum())if steps.sum()>0 else 0.
                        d['selected_max_wrist_step_over_path']=wrist_fraction(sp[ci]);d['valid_max_wrist_step_over_path']=wrist_fraction(sp[best])
                diagnostics.append(d)
        for condition,content in row['conditions'].items():
            cards=content['cards']
            for i,j in itertools.combinations(range(len(cards)),2):
                if sorted([cards[i]['family'],cards[j]['family']])!=[2,3]:continue
                spans=[cards[q]['chosen']['span']if cards[q]['chosen']else None for q in [i,j]]
                times=[[int(ts[t])for t in s]if s is not None else None for s in spans]
                state,detail,overlap=temporal_relation(*times)
                oldorder=next(o['state']for o in content['order']if o['cards']==[cards[i]['card'],cards[j]['card']])
                pts_records.append(dict(seed=k[0],clip_id=k[1],kind=k[2],condition=condition,cards=[cards[i]['card'],cards[j]['card']],
                    spans=spans,time_extents_us=times,stored_discrete_state=oldorder,timestamp_extent_state=state,
                    detail=detail,overlap_duration_us=overlap))
    regressions=[]
    for row in result['summary']:
        if row['condition']!='splice_rank':continue
        baseline=next(v for v in old['summary']if(v['seed'],v['kind'],v['condition'])==(row['seed'],row['kind'],'local_full'))
        regressions.append(dict(seed=row['seed'],kind=row['kind'],old=baseline,new=row,
            all_witness_delta=row['all_witness']-baseline['all_witness'],wrong_points_delta=row['wrong_points']-baseline['wrong_points'],
            missing_delta=row['missing']-baseline['missing']))
    fit_summary=[]
    for seed,origin,family in itertools.product([42,43,44],['original','splice'],[2,3,4,5,7]):
        rr=[r for r in fit if(r['seed'],r['origin'],r['family'])==(seed,origin,family)]
        if rr:fit_summary.append(dict(seed=seed,origin=origin,family=family,bags=len(rr),top_full_positive=sum(r['top_is_full_positive']for r in rr),
            old_top_full_positive=sum(r.get('old_top_is_full_positive',False)for r in rr)if origin=='original'else None,
            minimum_margin=min(r['full_minus_negative_margin']for r in rr)))
    write(E/'PTS_ORDER_AUDIT.json',dict(status='SAVED_PTS_ARITHMETIC_ONLY',helper=str(boundary),helper_sha256=sha(boundary),
        rule='Same temporal_relation actual PTS first/last extents applied symmetrically to ALL old conditions and new condition. Positive duration overlap only. Stored comparison preserved.',
        relations=pts_records,changed=sum(r['stored_discrete_state']!=r['timestamp_extent_state']for r in pts_records),
        no_inference=True,semantic_completion=None))
    write(E/'FIT_AND_REGRESSION_DIAGNOSIS.json',dict(status='DIAGNOSTIC_NOT_MODEL_SELECTION',fit_summary=fit_summary,fit_bags=fit,
        regression=regressions,dynamic_cases=diagnostics,
        interpretation='Separate(a) whether supervised FIT splice bags favor inherited full extents,(b) feature proximity and concentrated jumps for remaining bad DEV selections,(c) normal/speed regressions. Feature distance does not prove temporal averaging or causal failure. No runtime seam flags and no automatic retune.',
        no_new_predictions=True,no_threshold_tuning=True))
    check_pins(cfg['pins']);check_pins(tc['pins']);check_pins(ec['pins'])
    write(U/'POSTTRAIN_VERIFICATION.json',dict(status='SAVED_CONTRACT_PASS_RESEARCH_RESULTS_REQUIRE_REVIEW',at=time.time(),
        checks=dict(checks),total_checks=sum(checks.values()),all_seeds=[42,43,44],old_artifacts_preserved=True,
        new_training_sha256=sha(U/'TRAINING_COMPLETE.json'),new_comparison_sha256=sha(E/'COMPARISON.json'),
        verifier='Codex subagent saved-array arithmetic; parent review remains distinct; no human expert ground truth',
        no_training_or_inference=True,automatic_adoption=False,script_sha256=sha(__file__)))
    print('Saved result contracts PASS',sum(checks.values()),'PTS relations',len(pts_records),flush=True)

if __name__=='__main__':main()
