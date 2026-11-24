"""Fixed rate-head and 1:1 old-head blend on existing 82 feature caches.

Only the new head is inferred. Previous predictions and semantic results are
immutable. A changed interval needs separate semantic verification before use.
"""
from pathlib import Path
import collections,copy,importlib.util,itertools,sys,time,traceback
import numpy as np
U=Path(__file__).resolve().parent;H=U.parent;W=H.parent/'candidate_description_supervision';J=H/'splice_rank_learning';S=H/'stability_rank_blend';E=U/'evaluation'
sys.path.insert(0,str(H));from common import read,write,sha,stamp
sys.path.insert(0,str(W));from local_rank_core import LocalRankHead,F
import torch
spec=importlib.util.spec_from_file_location('_rate_saved_blend',S/'evaluate_saved.py');saved=importlib.util.module_from_spec(spec);spec.loader.exec_module(saved)
MODES=['rate_head','old_local_full_plus_rate_half']
def key(r):return r['seed'],r['clip_id'],r['kind']
def stop():
    for p in[U,H,H.parent]:
        for n in['STOP','STOP.txt','STOP_REQUESTED','STOP_REQUESTED.json']:assert not(p/n).exists()
def check(pins):
    for p,h in pins.items():assert sha(p)==h,p

def freeze():
    assert not(E/'FROZEN.json').exists();stop()
    old=read(S/'RESULTS.json');scenarios=read(J/'evaluation/FROZEN.json')['scenarios']
    assert len(old['rows'])==246 and len(scenarios)==82
    expected={(seed,s['clip_id'],s['kind'])for seed in[42,43,44]for s in scenarios}
    assert len(expected)==246 and {key(r)for r in old['rows']}==expected and {key(r)for r in old['predictions']}==expected
    rm={key(r):r for r in old['rows']}
    for s in scenarios:
        signatures=[[(c['card'],c['family'])for c in rm[seed,s['clip_id'],s['kind']]['conditions']['local_full']['cards']]for seed in[42,43,44]]
        assert signatures[0]==signatures[1]==signatures[2]
    pins=read(H.parent/'PROTECTIONS.json').copy()
    for p in[Path(__file__),U/'TRAINING_FROZEN.json',S/'RESULTS.json',S/'evaluate_saved.py',saved.BOUNDARY,J/'evaluation/FROZEN.json',H/'G2_REFERENCE_LINEAGE_REVIEW.json',W/'local_rank_core.py']:
        pins[str(p)]=sha(p)
    for r in old['predictions']:
        for p,h in[(r['original_prediction'],r['original_prediction_sha256']),(r['path'],r['sha256']),(r['source'],r['source_sha256']),(r['feature_path'],r['feature_sha256'])]:pins[p]=h
    check(pins)
    write(E/'FROZEN.json',dict(at=stamp(),status='FIXED_NEW_HEAD_EVALUATION_BEFORE_SCORES',conditions=MODES,scenarios=scenarios,seeds=[42,43,44],
        input='Same immutable82 candidate-local full1160 caches. Match exact source hashes and sorted union of old candidate spans.',
        inference='Matching newly trained rate-head only, once per scenario/seed. Family2/3 output only; other family logits/cards exact original local_full.',
        blend='Float64 0.5*old_local_full+0.5*rate_head, matching seed/candidate. No coefficient/seed/epoch choice.',
        selector='Existing presence>=0 and first raw argmax. No expected guide order, no temporal penalty, no new eligibility.',
        relation='Old discrete results retained. Same original positive-duration PTS relation for all conditions; shared instant is not a duration overlap.',
        original_reference='G2_REFERENCE_LINEAGE_REVIEW: original midpoint and occurrence criteria remain separate from intermediate baseline/local_full. REAL11 fast failure must be reported; original gates have no single named checkpoint.',
        semantics='Rank-unit only. No shape/direction rerun, no correctness/completion claim. Every changed semantic span requires later checked request.',
        no_encoder_inference=True,no_old_head_inference=True,no_private=True,no_final8=True,no_web=True,pins=pins))

def run():
    assert not(E/'COMPARISON.json').exists()and not(E/'predictions').exists()and not(E/'EXECUTION_FROZEN.json').exists();stop()
    cfg=read(E/'FROZEN.json');check(cfg['pins']);tr=read(U/'TRAINING_COMPLETE.json')
    assert tr['status']=='FIXED_RATE_HEAD_TRAINING_COMPLETE_NOT_ADOPTED'
    assert len(tr['rows'])==3 and {r['seed']for r in tr['rows']}=={42,43,44}
    runpins={str(U/'TRAINING_COMPLETE.json'):sha(U/'TRAINING_COMPLETE.json')}
    runpins.update({r['model']:r['model_sha256']for r in tr['rows']});check(runpins)
    write(E/'EXECUTION_FROZEN.json',dict(at=stamp(),models_fixed_before_inference=True,pins=runpins))
    torch.set_num_threads(4);torch.set_num_interop_threads(1);models={}
    for r in tr['rows']:
        assert sha(r['model'])==r['model_sha256'];st=torch.load(r['model'],map_location='cpu',weights_only=True)
        m=LocalRankHead(st['mean'].numpy(),st['std'].numpy());m.load_state_dict(st);m.eval();models[r['seed']]=m
    assert set(models)=={42,43,44}
    previous=read(S/'RESULTS.json');rm={key(r):r for r in previous['rows']};pm={key(r):r for r in previous['predictions']}
    relation=saved.pts_function();rows=[];predictions=[];timing=[];protected=0
    for s in cfg['scenarios']:
        stop();cid,kind=s['clip_id'],s['kind'];ff=np.load(s['feature_path']);sp=ff['spans'];lu={tuple(v):i for i,v in enumerate(sp)}
        q=np.load(s['source']);qr=q['z'][:,292:392].reshape(-1,10,10);ts=q['ts']
        assert np.issubdtype(ts.dtype,np.integer)and(np.diff(ts)>0).all()
        changing=any(c['family']in[2,3]for c in rm[42,cid,kind]['conditions']['local_full']['cards'])
        if changing:assert ff['x'].shape==(len(sp),1160)and np.isfinite(ff['x']).all()
        for seed in cfg['seeds']:
            base=rm[seed,cid,kind];meta=pm[seed,cid,kind];old=np.load(meta['original_prediction']);cards=base['conditions']['local_full']['cards']
            assert meta['feature_sha256']==s['feature_sha256']and meta['source_sha256']==s['source_sha256']
            expected=np.array(sorted({tuple(v)for i in range(len(cards))for v in old[f'spans_{i}']}),np.int32).reshape(-1,2)
            assert np.array_equal(sp,expected)
            tick=time.perf_counter();scores=None
            if changing:
                with torch.inference_mode():scores=models[seed](torch.tensor(ff['x'])).numpy()
            timing.append(dict(seed=seed,clip_id=cid,kind=kind,seconds=time.perf_counter()-tick,scope='new small rank head only'if changing else'no inference'))
            output={mode:[]for mode in MODES};arrays={}
            for i,c in enumerate(cards):
                spans=old[f'spans_{i}'];presence=old[f'presence_{i}'];a=old[f'local_full_logits_{i}']
                arrays[f'spans_{i}']=spans;arrays[f'presence_{i}']=presence
                if c['family']in[2,3]:
                    ids=[lu[tuple(v)]for v in spans];b=scores[ids,F.index(c['family'])]
                    values=[b,.5*a.astype(np.float64)+.5*b.astype(np.float64)]
                else:values=[a,a]
                for mode,v in zip(MODES,values):
                    assert np.isfinite(v).all();arrays[f'{mode}_logits_{i}']=v
                    if c['family']in[2,3]:
                        after=saved.select(c,spans,presence,v,qr)
                        if after['chosen']:after['chosen']['source']=mode
                    else:after=copy.deepcopy(c);protected+=1;assert np.array_equal(v,a)
                    output[mode].append(after)
            conditions=copy.deepcopy(base['conditions']);pts=copy.deepcopy(base['pts_order'])
            for mode in MODES:
                raw,po=saved.orders(output[mode],ts,relation);conditions[mode]=dict(cards=output[mode],order=raw);pts[mode]=po
            assert all(conditions[m]==base['conditions'][m]for m in base['conditions'])
            rows.append(dict(seed=seed,clip_id=cid,kind=kind,conditions=conditions,pts_order=pts))
            path=E/'predictions'/str(seed)/f'{cid}_{kind}.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**arrays)
            predictions.append(dict(meta,path=str(path),sha256=sha(path),previous_blend_prediction=meta['path'],previous_blend_sha256=meta['sha256']))
        print('rate compare',len(rows),246,flush=True)
    summary=copy.deepcopy(previous['summary'])
    for seed,mode,kind in itertools.product(cfg['seeds'],MODES,sorted({r['kind']for r in rows})):
        rr=[r for r in rows if r['seed']==seed and r['kind']==kind];cards=[c for r in rr for c in r['conditions'][mode]['cards']];truth=[c['truth']for c in cards if'truth'in c]
        summary.append(dict(seed=seed,condition=mode,kind=kind,scenarios=len(rr),cards=len(cards),missing=sum(c['chosen']is None for c in cards),truth_cards=len(truth),
            all_witness=sum(t['all_witness']for t in truth),wrong_points=sum(t['wrong_points']for t in truth),
            PTS_order=dict(collections.Counter(o['state']for r in rr for o in r['pts_order'][mode])),
            raw_discrete_order=dict(collections.Counter(o['state']for r in rr for o in r['conditions'][mode]['order']))))
    check(cfg['pins']);check(runpins);stop()
    write(E/'COMPARISON.json',dict(at=stamp(),status='FIXED_RATE_RANK_COMPARISON_REQUIRES_SEMANTIC_AND_GATE_REVIEW',rows=rows,predictions=predictions,summary=summary,timing=timing,
        protected_other_family_cards=protected,previous_conditions_exact=True,previous_PTS_exact=True,training_sha256=sha(U/'TRAINING_COMPLETE.json'),frozen_sha256=sha(E/'FROZEN.json'),
        no_encoder_inference=True,no_old_model_inference=True,no_private=True,no_final8=True,no_web=True,semantic_completion=None,G2_pass=None))

if __name__=='__main__':
    try:
        assert len(sys.argv)==2 and sys.argv[1]in['freeze','run-reviewed'];freeze()if sys.argv[1]=='freeze'else run()
    except BaseException:
        E.mkdir(exist_ok=True);(E/f'FIRST_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
