"""Audit saved null-selection comparison; no model/encoder inference."""
from pathlib import Path
from collections import Counter
import json,hashlib,datetime,itertools
import numpy as np
N=Path(__file__).resolve().parent;UNIT=N.parent;GRAPH=UNIT.parent
PRIOR=GRAPH/'candidate_description_supervision/local_rank_learning'
F=[2,3,4,5,7];AX={2:0,3:1,4:2,5:4,7:3}
def rd(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[]
def ck(k,v,d=None):checks.append(dict(name=k,ok=bool(v),detail=d))
def pins(values,label):
    for p,h in values.items():ck(label+str(p),Path(p).is_file() and sha(p)==h)
def key(r):return r['seed'],r['clip_id'],r['kind']
def order(cards):
    out=[]
    for i,j in itertools.combinations(range(len(cards)),2):
        if sorted([cards[i]['family'],cards[j]['family']])!=[2,3]:continue
        a,b=cards[i]['chosen'],cards[j]['chosen']
        if a is None or b is None:state='unobserved'
        elif max(a['span'][0],b['span'][0])<=min(a['span'][1],b['span'][1]):state='ambiguous_overlap'
        elif a['span'][1]<b['span'][0]:state='forward'
        else:state='reverse'
        out.append(dict(cards=[cards[i]['card'],cards[j]['card']],state=state))
    return out
def main():
    assert not (N/'VERIFICATION.json').exists()
    cfg=rd(N/'FROZEN.json');tc=rd(N/'TRAINING_COMPLETE.json');index=rd(N/'TRAINING_INPUT_INDEX.json')
    ef=rd(N/'evaluation/FROZEN.json');result=rd(N/'evaluation/COMPARISON.json');prior=rd(PRIOR/'evaluation/COMPARISON.json')
    pins(cfg['pins'],'frozen');pins(ef['pins'],'evaluation')
    rows=index['rows'];fit=index['train_indices'];probe=[i for i in range(len(rows)) if i not in set(fit)]
    ck('fit_index_exact',fit==[i for i,r in enumerate(rows) if r['split']=='train'])
    ck('no_source_overlap',not {rows[i]['source_sha256'] for i in fit}&{rows[i]['source_sha256'] for i in probe})
    ck('people_split',all(rows[i]['person'] in [2,3,4,5,6,7] for i in fit) and all(rows[i]['person'] in [8,9] for i in probe))
    ck('typed_family_scope',all(r['family'] in [3,4,5,7] for r in rows))
    bag_arrays=[]
    for i,r in enumerate(rows):
        pins({r['path']:r['sha256']},'bag');z=np.load(r['path']);bag_arrays.append((z['local_full'],z['y']))
        ck('bag_finite:'+str(i),np.isfinite(z['local_full']).all() and len(z['local_full'])==len(z['y']))
        ck('bag_boolean_labels:'+str(i),z['y'].dtype.kind=='b')
        ck('positive_or_null_target:'+str(i),bool(z['y'].any())==(r['origin']=='positive_source_extent'))
    expected={(f,o) for f in [3,4,5,7] for o in ['positive_source_extent','reviewed_local_hold']}
    counts=Counter((rows[i]['family'],rows[i]['origin']) for i in fit)
    ck('balanced_training_pools',set(counts)==expected and min(counts.values())>=2)
    ck('three_seeds',len(tc['rows'])==3 and {r['seed'] for r in tc['rows']}=={42,43,44})
    null_by_seed={};bag_summary=[]
    for r in tc['rows']:
        pins({r[k]:r[k+'_sha256'] for k in ['model','predictions']},'training_run')
        pins({r['initial_model']:r['initial_sha256']},'initial_model')
        ck('final_training_step:'+str(r['seed']),r['log'][-1]['step']==cfg['steps']==1200)
        pred=np.load(r['predictions']);null=pred['null_logits'];null_by_seed[r['seed']]=null
        ck('finite_null:'+str(r['seed']),null.shape==(5,) and np.isfinite(null).all())
        for i,row in enumerate(rows):
            lo=pred[f'logits_{i}'];xx,y=bag_arrays[i]
            ck('saved_bag_scores:'+str((r['seed'],i)),lo.shape==(len(xx),5) and np.isfinite(lo).all())
            j=F.index(row['family']);values=lo[:,j];selected=int(values.argmax()) if len(values) and values.max()>null[j] else None
            bag_summary.append(dict(seed=r['seed'],input_row=i,split=row['split'],family=row['family'],origin=row['origin'],candidates=len(values),selected=selected,
                 positive_selected=bool(selected is not None and y[selected]),null_selected=selected is None))
    old={key(r):r for r in prior['rows']};new={key(r):r for r in result['rows']}
    oldpred={key(r):r for r in prior['predictions']};pred={key(r):r for r in result['predictions']}
    ck('all_comparison_rows',len(new)==len(result['rows'])==246 and set(new)==set(old))
    ck('all_prediction_rows',len(pred)==len(result['predictions'])==246 and set(pred)==set(oldpred))
    ck('82_scenarios_per_seed',all(sum(k[0]==s for k in new)==82 for s in [42,43,44]))
    fcache={};ocache={};sources={};families2=0;diagnostic=[];changed=[]
    for kk,row in new.items():
        before=old[kk];p=pred[kk];op=oldpred[kk]
        pins({p['path']:p['sha256']},'new_prediction')
        ck('metadata_original_source:'+str(kk),{k:v for k,v in p.items() if k not in ['path','sha256','feature_path','feature_sha256']}=={k:v for k,v in op.items() if k not in ['path','sha256']})
        if p['feature_path'] not in fcache:
            pins({p['feature_path']:p['feature_sha256']},'new_feature');fcache[p['feature_path']]=np.load(p['feature_path'])
        if p['source'] not in sources:
            pins({p['source']:p['source_sha256']},'stored_source');sources[p['source']]=np.load(p['source'])
        if op['path'] not in ocache:
            pins({op['path']:op['sha256']},'old_prediction');ocache[op['path']]=np.load(op['path'])
        arr=np.load(p['path']);oa=ocache[op['path']];feat=fcache[p['feature_path']];source=sources[p['source']];qr=source['z'][:,292:392].reshape(-1,10,10)
        ck('three_conditions:'+str(kk),set(row['conditions'])==set(ef['conditions']))
        for c in ['baseline','local_full']:ck('old_condition_exact:'+str((c,kk)),row['conditions'][c]==before['conditions'][c])
        cards=row['conditions']['local_null']['cards'];prev=before['conditions']['local_full']['cards']
        ck('card_count:'+str(kk),len(cards)==len(prev))
        union=sorted(set(tuple(s) for i in range(len(prev)) for s in oa[f'spans_{i}']))
        ck('feature_span_union:'+str(kk),np.array_equal(feat['spans'],np.asarray(union,np.int32).reshape(-1,2)))
        ck('feature_shape:'+str(kk),feat['x'].shape==(len(union),1160) and np.isfinite(feat['x']).all())
        for i,(card,pc) in enumerate(zip(cards,prev)):
            ck('card_identity:'+str((kk,i)),card['card']==pc['card'] and card['family']==pc['family'] and card['semantic_completion'] is None)
            if card['family']==2:
                ck('family2_exact:'+str((kk,i)),card==pc);families2+=1;continue
            spans=arr[f'spans_{i}'];presence=arr[f'presence_{i}'];logits=arr[f'logits_{i}'];null=float(arr[f'null_{i}']);j=F.index(card['family']);axis=AX[card['family']]*2
            ck('span_and_presence_exact:'+str((kk,i)),np.array_equal(spans,oa[f'spans_{i}']) and np.array_equal(presence,oa[f'presence_{i}']))
            ck('score_shapes_and_finite:'+str((kk,i)),len(spans)==len(presence)==len(logits) and np.isfinite(logits).all() and np.isfinite(presence).all() and np.isfinite(null))
            ck('null_matches_checkpoint_prediction:'+str((kk,i)),null==float(null_by_seed[kk[0]][j]))
            eligible=presence>=0;masked=np.where(eligible,logits,-np.inf);best=int(masked.argmax()) if len(masked) and np.isfinite(masked).any() else None
            if not len(masked):reason='no_generated_candidates'
            elif best is None:reason='no_candidate_passed_presence_gate'
            elif masked[best]>null:reason='selected_observed_candidate'
            elif masked[best]<null:reason='learned_null_exceeds_admitted_candidates'
            else:reason='tie_with_null'
            expected_choice=best if reason=='selected_observed_candidate' else None
            selected=card['chosen'];ck('null_competition:'+str((kk,i)),(selected is None)==(expected_choice is None))
            if expected_choice is not None and selected is not None:
                a,b=map(int,spans[expected_choice]);obs=np.arange(a,b+1);obs=obs[qr[obs,axis,8]>.5];hands=obs[qr[obs,axis,9]>.5]
                ck('chosen_span:'+str((kk,i)),selected['span']==[a,b])
                ck('chosen_score:'+str((kk,i)),selected['score']==float(masked[best]) and selected['null_logit']==null)
                ck('chosen_observed_points:'+str((kk,i)),selected['body_observed']==obs.tolist() and selected['hand_qualified']==hands.tolist())
                ck('chosen_completion_unknown:'+str((kk,i)),selected['semantic_completion'] is None)
            if 'truth' in card:
                t=card['truth'];oldtruth=pc['truth'];observed=set(selected['body_observed']) if selected else set()
                ck('truth_source_unchanged:'+str((kk,i)),{k:v for k,v in t.items() if k not in ['all_witness','wrong_points']}=={k:v for k,v in oldtruth.items() if k not in ['all_witness','wrong_points']})
                ck('witness_arithmetic:'+str((kk,i)),t['all_witness']==(set(t['witness'])<=observed) and t['wrong_points']==len(set(t['wrong_witness'])&observed))
            diagnostic.append(dict(seed=kk[0],clip_id=kk[1],kind=kk[2],card=card['card'],family=card['family'],reason=reason,candidates=len(spans),admitted=int(eligible.sum()),null_logit=null,
                                  best_score=float(masked[best]) if best is not None else None,raw_status_present='selection_state' in card or 'selection_reason' in card))
            if selected!=pc['chosen']:changed.append(dict(seed=kk[0],clip_id=kk[1],kind=kk[2],card=card['card'],old_span=pc['chosen']['span'] if pc['chosen'] else None,new_span=selected['span'] if selected else None,reason=reason))
        ck('order_arithmetic:'+str(kk),row['conditions']['local_null']['order']==order(cards))
    ck('82_unique_feature_inputs',len(fcache)==82)
    summary=[]
    for seed,c,kind in itertools.product([42,43,44],ef['conditions'],sorted({r['kind'] for r in result['rows']})):
        rr=[r for r in result['rows'] if r['seed']==seed and r['kind']==kind];cards=[ca for r in rr for ca in r['conditions'][c]['cards']];truth=[ca['truth'] for ca in cards if 'truth' in ca];orders=[o['state'] for r in rr for o in r['conditions'][c]['order']]
        summary.append(dict(seed=seed,condition=c,kind=kind,cases=len(rr),cards=len(cards),missing=sum(ca['chosen'] is None for ca in cards),known_groups=len(truth),all_witness=sum(t['all_witness'] for t in truth),wrong_points=sum(t['wrong_points'] for t in truth),order=dict(Counter(orders))))
    sk=lambda x:(x['seed'],x['condition'],x['kind'])
    ck('all_summary_denominators',sorted(summary,key=sk)==sorted(result['summary'],key=sk))
    protected=rd(GRAPH/'PROTECTIONS.json');pins(protected,'protected');ck('protected_477',len(protected)==477)
    for obj,name in [(cfg,'freeze'),(tc,'training'),(ef,'evaluation')]:
        for k in ['private_access','final8','web_changed']:ck(name+k,obj[k] is False)
    ck('no_completion_claim',all(c['semantic_completion'] is None for r in result['rows'] for c in r['conditions']['local_null']['cards']))
    report=dict(at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),status='PASS' if all(c['ok'] for c in checks) else 'FAIL',checks=len(checks),failed=[c for c in checks if not c['ok']],rows=len(new),family2_preserved=families2,
        feature_scenarios=len(fcache),protected_files=len(protected),summary=summary,selection_reasons=dict(Counter(r['reason'] for r in diagnostic)),selection_diagnostics=diagnostic,changed_choices=changed,
        saved_bag_outcomes=bag_summary,no_model_inference=True,adopted=False,full_G2_pass=False,
        warnings=['chosen=None currently conflates no candidates, failed presence gate and learned null. Saved arrays permit the recorded reason audit; add explicit reason before user feedback.','Local reviewed hold is non-progression, not certified omission or semantic completion.'],
        reviewer='Delegated Codex motion_contract_review_1113 saved arithmetic/code review, not human KSL truth',script_sha256=sha(__file__))
    (N/'VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
    print(json.dumps({k:report[k] for k in ['status','checks','failed','rows','family2_preserved','feature_scenarios','protected_files','selection_reasons']},ensure_ascii=False),flush=True)
    if report['failed']:raise SystemExit(1)
if __name__=='__main__':main()
