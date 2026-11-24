"""Same82 frozen regression: saved local features, new rank-head inference only.

The original selector is retained. No encoder/old model/shape/direction inference
is run here. This is the rank unit regression, not complete semantic G2 approval.
"""
import collections, copy, itertools, sys, time, traceback
import numpy as np
from prepare import U, H, W, L, read, write, sha, check_pins
sys.path.insert(0,str(W))
from local_rank_core import LocalRankHead, F, AX
import torch
E=U/'evaluation'
CACHE=H/'null_candidate_learning/evaluation/COMPARISON.json'

def rowkey(r):return r['seed'],r['clip_id'],r['kind']
def scenario(r):return r['clip_id'],r['kind']

def original_orders(cards):
    """Exact old82 discrete-span diagnostic; never input to candidate selection."""
    out=[]
    for i,j in itertools.combinations(range(len(cards)),2):
        if sorted([cards[i]['family'],cards[j]['family']])!=[2,3]:continue
        a,b=cards[i]['chosen'],cards[j]['chosen']
        state=('unobserved'if a is None or b is None else
               'ambiguous_overlap'if max(a['span'][0],b['span'][0])<=min(a['span'][1],b['span'][1])else
               'forward'if a['span'][1]<b['span'][0]else'reverse')
        out.append(dict(cards=[cards[i]['card'],cards[j]['card']],state=state))
    return out

def freeze():
    assert not (E/'FROZEN.json').exists()
    prior=read(L/'evaluation/COMPARISON.json'); cached=read(CACHE)
    pm={rowkey(r):r for r in prior['predictions']}
    cm={rowkey(r):r for r in cached['predictions']}
    assert set(pm)==set(cm) and len(pm)==246
    scenarios={}
    pins={str(p):sha(p)for p in [U/'evaluate.py',U/'TRAINING_FROZEN.json',
          L/'evaluation/COMPARISON.json',L/'evaluation/FROZEN.json',W/'evaluate_local_rank.py',
          CACHE,H/'null_candidate_learning/evaluate.py',H/'null_candidate_learning/evaluation/FROZEN.json',
          W/'transfer_local_timefix_core.py',W/'local_rank_core.py']}
    checked=0
    for rr in prior['rows']:
        key=rowkey(rr); old=pm[key]; cache=cm[key]
        assert old['source']==cache['source'] and old['source_sha256']==cache['source_sha256']
        assert sha(old['path'])==old['sha256'];pins[old['path']]=old['sha256']
        sid=scenario(rr)
        if sid not in scenarios:
            assert sha(cache['feature_path'])==cache['feature_sha256']
            assert sha(old['source'])==old['source_sha256']
            pins[cache['feature_path']]=cache['feature_sha256'];pins[old['source']]=old['source_sha256']
            ff=np.load(cache['feature_path']);sp=ff['spans']
            scenarios[sid]=dict(clip_id=rr['clip_id'],kind=rr['kind'],feature_path=cache['feature_path'],
                feature_sha256=cache['feature_sha256'],source=old['source'],source_sha256=old['source_sha256'],
                candidates=len(sp))
        else:
            assert scenarios[sid]['feature_path']==cache['feature_path'] and scenarios[sid]['feature_sha256']==cache['feature_sha256']
            sp=np.load(cache['feature_path'])['spans']
        arr=np.load(old['path']); cards=rr['conditions']['local_full']['cards']
        expected=np.array(sorted({tuple(s)for i in range(len(cards))for s in arr[f'spans_{i}']}),np.int32).reshape(-1,2)
        assert np.array_equal(expected,sp);checked+=1
        assert original_orders(cards)==rr['conditions']['local_full']['order']
    assert len(scenarios)==82
    write(E/'FROZEN.json',dict(status='EVALUATION_PREPARED_NOT_EXECUTED',at=time.time(),
        primary='splice_rank',reference='Saved immutable local_full; every old condition reused exactly.',
        seeds=[42,43,44],same82_scenarios=True,scenarios=list(scenarios.values()),cache_span_checks=checked,
        input='Existing full1160 candidate-local cache from null_candidate evaluation. Verified same original source hashes and exact union of all old spans in246 rows. No new encoder inference.',
        new_inference='Matching-seed newly trained head only, only scenarios with family2/3. No old network inference.',
        routing='Family2/3 use new raw rank; families4/5/7 scores, cards, truth and qualifiers copied exactly from original local_full.',
        selector='Exactly original saved presence>=0 and raw argmax; unchanged candidate list/body observation/hand qualification. No extra threshold, duration/jump penalty, shape gate, expected guide order, seam or provenance input.',
        shape_direction='No shape/direction model or requirement change. Preserve source observation qualifiers; this rank unit does not assert recomputed shape/direction semantic success. Full composed G2 remains root follow-up.',
        order='Exact original discrete-span82 diagnostic on new selections, not correspondence input. No change to shared-boundary/positive-duration G2 relation implementation in this experiment.',
        completion=None,prohibited=['DEV training/tuning/seed selection','old evaluation rerun','final8','private/user/mentor','web adoption'],
        pass_contract='Compare every seed and every fixed kind; require original normal/speed/partial/shape/direction/cross/order/omission behavior preserved and fewer dynamic join mistakes where a valid alternative exists. No aggregate-only acceptance. Report regressions and abstention; this file does not automatically declare G2 passed.',
        pins=pins))
    print('82 scenario cached feature contract PASS;',checked,'saved row matches; no inference',flush=True)

def select_card(original,spans,presence,logits,qr):
    axis=AX[original['family']]*2
    scores=np.where(presence>=0,logits,-np.inf);chosen=None
    if len(scores)and np.isfinite(scores.max()):
        at=int(scores.argmax());a,b=map(int,spans[at]);obs=np.arange(a,b+1)
        obs=obs[qr[obs,axis,8]>.5]
        chosen=dict(span=[a,b],source='splice_rank',body_observed=obs.tolist(),
            hand_qualified=obs[qr[obs,axis,9]>.5].tolist(),score=float(scores[at]),semantic_completion=None)
    card=dict(card=original['card'],family=original['family'],chosen=chosen,semantic_completion=None)
    if 'truth'in original:
        observed=set(chosen['body_observed'])if chosen else set()
        card['truth']=dict(original['truth'],all_witness=set(original['truth']['witness'])<=observed,
                           wrong_points=len(set(original['truth']['wrong_witness'])&observed))
    return card

def evaluate():
    assert not (E/'COMPARISON.json').exists() and not (E/'predictions').exists()
    cfg=read(E/'FROZEN.json');check_pins(cfg['pins'])
    train=read(U/'TRAINING_COMPLETE.json');prior=read(L/'evaluation/COMPARISON.json')
    assert train['status']=='TRAINED_NOT_SELECTED_OR_DEPLOYED'
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    models={}
    for r in train['rows']:
        assert sha(r['model'])==r['model_sha256']
        st=torch.load(r['model'],map_location='cpu',weights_only=True)
        model=LocalRankHead(st['mean'].numpy(),st['std'].numpy());model.load_state_dict(st);model.eval();models[r['seed']]=model
    assert set(models)=={42,43,44}
    pm={rowkey(r):r for r in prior['predictions']}; rm={rowkey(r):r for r in prior['rows']}
    rows=[];predictions=[];timing=[];protections=0
    for s in cfg['scenarios']:
        sid=scenario(s);base_row=rm[42,*sid];base_cards=base_row['conditions']['local_full']['cards']
        changing=any(c['family']in [2,3]for c in base_cards)
        f=np.load(s['feature_path']); sp=f['spans'];lu={tuple(v):i for i,v in enumerate(sp)}
        source=np.load(s['source']);qr=source['z'][:,292:392].reshape(-1,10,10)
        features=f['x'] if changing else None
        if changing:assert features.shape==(len(sp),1160)and np.isfinite(features).all()
        for seed in cfg['seeds']:
            oldrow=rm[seed,*sid];op=pm[seed,*sid];oa=np.load(op['path'])
            tick=time.perf_counter();scores=None
            if changing:
                with torch.inference_mode():scores=models[seed](torch.tensor(features)).numpy()
            timing.append(dict(seed=seed,clip_id=sid[0],kind=sid[1],seconds=time.perf_counter()-tick,
                inference='new rank head only'if changing else'none; original family scores copied'))
            cards=[];arrays={}
            for i,c in enumerate(oldrow['conditions']['local_full']['cards']):
                spans=oa[f'spans_{i}'];presence=oa[f'presence_{i}']
                arrays[f'spans_{i}']=spans;arrays[f'presence_{i}']=presence
                if c['family']in [2,3]:
                    idx=np.asarray([lu[tuple(v)]for v in spans],int);logits=scores[idx,F.index(c['family'])]
                    card=select_card(c,spans,presence,logits,qr)
                else:
                    logits=oa[f'local_full_logits_{i}'].copy();card=copy.deepcopy(c);protections+=1
                    assert card==c and np.array_equal(logits,oa[f'local_full_logits_{i}'])
                arrays[f'splice_rank_logits_{i}']=logits;cards.append(card)
            conditions=copy.deepcopy(oldrow['conditions'])
            conditions['splice_rank']=dict(cards=cards,order=original_orders(cards))
            assert all(conditions[k]==oldrow['conditions'][k]for k in oldrow['conditions'])
            rows.append(dict(seed=seed,clip_id=sid[0],kind=sid[1],conditions=conditions))
            path=E/'predictions'/str(seed)/f'{sid[0]}_{sid[1]}.npz';path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,**arrays)
            predictions.append(dict(op,path=str(path),sha256=sha(path),original_prediction=op['path'],
                original_prediction_sha256=op['sha256'],feature_path=s['feature_path'],feature_sha256=s['feature_sha256']))
        print('splice rank compare',len(rows),246,flush=True)
    # All original summary records are copied, not recomputed or replaced.
    summary=copy.deepcopy(prior['summary'])
    for seed,kind in itertools.product(cfg['seeds'],sorted({r['kind']for r in rows})):
        rr=[r for r in rows if r['seed']==seed and r['kind']==kind]
        cards=[c for r in rr for c in r['conditions']['splice_rank']['cards']]
        truth=[c['truth']for c in cards if'truth'in c]
        orders=[o['state']for r in rr for o in r['conditions']['splice_rank']['order']]
        summary.append(dict(seed=seed,condition='splice_rank',kind=kind,cases=len(rr),cards=len(cards),
            missing=sum(c['chosen']is None for c in cards),known_groups=len(truth),all_witness=sum(t['all_witness']for t in truth),
            wrong_points=sum(t['wrong_points']for t in truth),order=dict(collections.Counter(orders))))
    check_pins(cfg['pins'])
    write(E/'COMPARISON.json',dict(status='RANK_UNIT_REGRESSION_REQUIRES_REVIEW',at=time.time(),rows=rows,
        predictions=predictions,timing=timing,summary=summary,protected_family_cards_exact=protections,
        frozen_sha256=sha(E/'FROZEN.json'),new_training_sha256=sha(U/'TRAINING_COMPLETE.json'),
        old_predictions_reused=True,encoder_inference=False,old_model_inference=False,
        shape_direction_models_changed=False,semantic_completion=None,web_changed=False))

if __name__=='__main__':
    try:
        assert len(sys.argv)==2 and sys.argv[1]in ['freeze','run-reviewed']
        freeze() if sys.argv[1]=='freeze' else evaluate()
    except BaseException:
        E.mkdir(exist_ok=True)
        (E/f'FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
