"""Replay K's RNG consumption only and save its exact source-selection schedule.

No neural-network imports, parameter reads, model forward, feature rebuild or
optimizer. K's rate/negative RNG draws are discarded after count verification.
"""
from pathlib import Path
import collections,hashlib,json,time
import numpy as np
L=Path(__file__).resolve().parent;H=L.parent;K=H/'temporal_rate_rank_learning';J=H/'splice_rank_learning'
FAMILIES=[2,3,4,5,7]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def key(r):return r['clip_id'],r['view'],r['reverse'],r['family']

def main():
    assert not(L/'SOURCE_SCHEDULE.json').exists()and not(L/'SOURCE_SCHEDULE_VERIFICATION.json').exists()
    meta=read(K/'PREPARATION_INPUTS_COMPLETE.json');cfg=read(K/'TRAINING_FROZEN.json');trained=read(K/'TRAINING_COMPLETE.json')
    assert cfg['steps']==1200 and cfg['seeds']==[42,43,44]
    pins=dict(cfg['pins'])
    for p in [K/'PREPARATION_INPUTS_COMPLETE.json',K/'TRAINING_FROZEN.json',K/'TRAINING_COMPLETE.json',K/'train.py',J/'TRAINING_COMPLETE.json',Path(__file__)]:pins[str(p)]=sha(p)
    for p,h in pins.items():assert sha(p)==h,p
    allbags=[];original=[];variants=collections.defaultdict(lambda:collections.defaultdict(list))
    def item(r,kind,rate):
        assert sha(r['path'])==r['sha256'];y=np.load(r['path'])['y']
        assert y.dtype==np.bool_ and y.any()and(~y).any()
        d=dict(row=r,kind=kind,rate=rate,neg=np.flatnonzero(~y),identity=(r['path'],kind,rate));allbags.append(d);return d
    for r in meta['replay_original']:
        d=item(r,'normal',1.);original.append(d)
        if r['family']in[2,3]:variants[key(r)]['normal'].append(d)
    for r in meta['replay_splice']:variants[key(r)]['splice'].append(item(r,'splice',1.))
    for r in meta['bags']:
        if not r['skipped']:variants[key(r)][r['base_kind']].append(item(r,r['base_kind'],r['rate']))
    assert len(original)==178 and len(allbags)==460 and len(variants)==72
    groups={f:[i for i,d in enumerate(original)if d['row']['family']==f]for f in FAMILIES}
    assert {f:len(v)for f,v in groups.items()}=={2:36,3:36,4:34,5:36,7:36}
    schedules={};reports=[];observed_index={};checks=0
    for run in trained['rows']:
        seed=run['seed'];rng=np.random.default_rng(seed);counts=collections.Counter();steps=[]
        index_path=Path(run['model']).parent/'FIT_PREDICTION_INDEX.json';idx=read(index_path)
        assert len(idx)==460 and sha(run['fit_predictions'])==run['fit_predictions_sha256']
        pins[str(index_path)]=sha(index_path);pins[run['fit_predictions']]=run['fit_predictions_sha256'];pins[run['model']]=run['model_sha256']
        for step in range(1200):
            family_indices=[]
            for f in FAMILIES:
                chosen=rng.choice(groups[f],2,replace=False);family_indices.append([int(i)for i in chosen])
                for i in chosen:
                    d=original[int(i)]
                    if f in[2,3]:
                        v=variants[key(d['row'])]
                        # K chooses BOTH branch rates before drawing negatives.
                        selected=[v[b][int(rng.integers(len(v[b])))]for b in['normal','splice']]
                    else:selected=[d]
                    for d in selected:
                        rng.choice(d['neg'],min(len(d['neg']),56),replace=False)
                        counts[d['identity']]+=1
            steps.append(family_indices)
        for i,(r,d)in enumerate(zip(idx,allbags)):
            assert r['key']==f'bag_{i}'and r['path']==d['row']['path']and r['sha256']==d['row']['sha256']
            assert r['family']==d['row']['family']and r['base_kind']==d['kind']and r['rate']==d['rate']
            assert r['sampled_steps']==counts[d['identity']]
            checks+=1
        assert sum(counts.values())==16800
        schedules[str(seed)]=steps;observed_index[seed]=idx
        reports.append(dict(seed=seed,steps=1200,source_draws=12000,discarded_K_bag_draws=16800,
            sampled_steps_exact_matches=460,minimum_original_K_bag_draws=min(counts[d['identity']]for d in allbags),
            K_all_training_variants_sampled=run['all_training_variants_sampled'],
            source_schedule_basis='Exact replay of pinned K code/seed/default_rng and retained-label negative arrays; K did not separately store per-step source history. Saved per-bag counts independently match.'))
    assert set(schedules)=={'42','43','44'}
    schedule=dict(status='FROZEN_SOURCE_SCHEDULE_MATCHING_K_RNG',created_at=time.time(),family_order=FAMILIES,
        seeds=[42,43,44],steps_per_seed=1200,indexing='schedules[str(seed)][step_zero_based][family_index] -> two distinct indices into original_bags (same order as K replay_original178).',
        original_bags=[dict(index=i,clip_id=d['row']['clip_id'],person=d['row']['person'],view=d['row']['view'],reverse=d['row']['reverse'],
            family=d['row']['family'],path=d['row']['path'],sha256=d['row']['sha256'])for i,d in enumerate(original)],
        schedules=schedules,k_input_manifest=dict(path=str(K/'PREPARATION_INPUTS_COMPLETE.json'),sha256=sha(K/'PREPARATION_INPUTS_COMPLETE.json')),
        k_train_code=dict(path=str(K/'train.py'),sha256=sha(K/'train.py')),numpy_version=np.__version__,
        initialization='L matches K initial matching-seed J checkpoint, NOT K final checkpoint. Same normalization/fresh optimizer/1200steps/lr/weight_decay.',
        intended_changes_only='Use all available rates in each branch with .5/nrate weight and all stored negatives per selected bag. Source schedule fixed to K. Other family original replay retained.',
        caveat='All stored negatives means the existing capped candidate bag, not every negative in the original exhaustive grid. Averaging all rates reduces rate-sampling gradient variance; no new equality/distillation target.',
        no_training=True,no_model_inference=True,pins=pins)
    write(L/'SOURCE_SCHEDULE.json',schedule)
    # Independent validation from the SAVED source schedule and K per-bag counts.
    saved=read(L/'SOURCE_SCHEDULE.json');sourcechecks=0
    for seed in saved['seeds']:
        seq=saved['schedules'][str(seed)];assert len(seq)==1200;source_counts=collections.Counter()
        for step in seq:
            assert len(step)==5
            for f,chosen in zip(saved['family_order'],step):
                assert len(chosen)==2 and len(set(chosen))==2
                assert all(0<=i<178 and original[i]['row']['family']==f for i in chosen)
                source_counts.update(chosen);sourcechecks+=1
        idx=observed_index[seed];lookup={(r['path'],r['base_kind'],r['rate']):r['sampled_steps']for r in idx}
        for i,d in enumerate(original):
            if d['row']['family']in[2,3]:
                for branch in ['normal','splice']:
                    actual=sum(lookup[v['identity']]for v in variants[key(d['row'])][branch])
                    assert actual==source_counts[i];sourcechecks+=1
            else:assert lookup[d['identity']]==source_counts[i];sourcechecks+=1
        assert sum(source_counts.values())==12000
    for p,h in pins.items():assert sha(p)==h,p
    verification=dict(status='SOURCE_SCHEDULE_SAVED_RNG_COUNTS_VERIFIED_NO_TRAINING',family_order=FAMILIES,
        replayed_bag_count_checks=checks,saved_schedule_checks=sourcechecks,reports=reports,
        original_bags=178,training_bags=460,source_steps_per_seed=1200,all_pins_unchanged=True,pins_checked=len(pins),
        source_schedule_sha256=sha(L/'SOURCE_SCHEDULE.json'),no_training=True,no_model_inference=True,
        verifier='Codex subagent RNG/label arithmetic plus independent saved-schedule source counts; no parameter or feature inference',
        future_parent='Same matching J parent as K began from, excluding extra training-budget confounding.')
    write(L/'SOURCE_SCHEDULE_VERIFICATION.json',verification)
    print('Source schedule verified',checks,'K bag counts;',sourcechecks,'saved schedule checks;',len(pins),'pins unchanged')
if __name__=='__main__':main()
