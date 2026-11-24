"""Saved synthetic-static scores only; no new or old model inference.

The matched216 old decisions are training diagnostics after augmentation.
No old full-set static baseline exists and none is invented or re-inferred.
"""
from body_core import *
S=B
from collections import defaultdict, Counter
import time, traceback

ANCHORS=['start','middle','end']

def summarize(rows,keys):
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in keys)].append(r)
    return [dict(**dict(zip(keys,key)),generated=len(rr),
        observed_eligible=sum(r['observation_eligible'] for r in rr),
        excluded_unobserved=sum(not r['observation_eligible'] for r in rr),
        false_progression_observed=sum(r['observation_eligible'] and r['predicted_progression'] for r in rr),
        rejected_static_observed=sum(r['observation_eligible'] and not r['predicted_progression'] for r in rr),
        predicted_progression_on_unobserved=sum(not r['observation_eligible'] and r['predicted_progression'] for r in rr))
        for key,rr in sorted(groups.items())]

def main():
    E=S/'static_evaluation';E.mkdir(exist_ok=True)
    assert not(E/'COMPARISON.json').exists(),'Completed static evaluation cannot rerun'
    assert not(E/'RULES_FROZEN.json').exists(),'Use explicit audited recovery after failure'
    meta=read(S/'INPUTS_COMPLETE.json');train=read(S/'TRAINING_COMPLETE.json')
    index=read(N/'TRAINING_INPUT_INDEX.json')['rows'];audit=read(S/'FIT_LABEL_AUDIT.json')
    previous=read(P/'static_contract_audit/RESULTS.json')
    priorplan=read(P/'static_contract_audit/FROZEN.json')
    old_rows={(r['seed'],r['bag_index'],r['candidate_index'],r['anchor']):r for r in previous['rows']}
    assert len(old_rows)==len(previous['rows'])==216
    eligibility={r['bag_index']:r for r in audit['static_anchor_eligibility']}
    prepared={};metadata_rows=[];pins={}; matched_keys=set()
    for r in meta['static_bags']:
        i=r['bag_index'];source=index[i]
        assert all(r[k]==source[k] for k in ['clip_id','person','split','family','view','reverse'])
        assert source['origin']=='positive_source_extent'
        assert r['family'] in FAMILIES and r['split']==('probe' if r['person'] in [8,9] else 'train')
        assert sha(r['path'])==r['sha256'];pins[r['path']]=r['sha256']
        z=np.load(r['path']);mapping=z['mapping'];pos=z['positive_indices']
        assert np.array_equal(mapping[:,0],np.repeat(pos,3))
        assert np.array_equal(mapping[:,1],np.tile(np.arange(3),len(pos)))
        assert len(z['x'])==len(mapping)==r['static_candidates']
        mask_index=64+(old.AX[r['family']]*2)*10+8
        coverage=z['x'][:,mask_index];assert np.isin(coverage,[0,1]).all()
        seen=coverage>.5;eligible=eligibility[i]
        assert int(seen.sum())==eligible['eligible'] and int((~seen).sum())==eligible['excluded_unobserved']
        assert sha(source['path'])==source['sha256'];pins[source['path']]=source['sha256']
        original=np.load(source['path']);spans=original['spans'];full=original['y'].astype(bool)
        local=[]
        for static_id,(candidate,anchor,frame) in enumerate(mapping):
            candidate,anchor,frame=map(int,[candidate,anchor,frame])
            a,b=map(int,spans[candidate]);assert frame==[a,(a+b)//2,b][anchor]
            row=dict(bag_index=i,static_index=static_id,clip_id=r['clip_id'],person=r['person'],
                split=r['split'],family=r['family'],view=r['view'],reverse=r['reverse'],
                source_sha256=r['source_sha256'],candidate_index=candidate,
                source_candidate_kind='full' if full[candidate] else 'partial',
                transformed_span=[a,b],frames=b-a+1,duration_bucket=int(bucket(b-a+1)),
                anchor=ANCHORS[anchor],anchor_frame_transformed=frame,
                observation_eligible=bool(seen[static_id]),observation_mask=float(coverage[static_id]),
                expected_observed_progression=False if seen[static_id] else None,
                semantic_completion=None)
            local.append(row)
            for seed in [42,43,44]:
                key=(seed,i,candidate,ANCHORS[anchor])
                if key in old_rows:
                    pr=old_rows[key]
                    assert r['split']=='train' and r['view']=='full' and not r['reverse']
                    assert pr['span']==[a,b] and pr['anchor_frame']==frame
                    assert all(pr[k]==r[k] for k in ['clip_id','person','family'])
                    matched_keys.add(key)
        prepared[i]=local
        metadata_rows.append(dict(bag_index=i,clip_id=r['clip_id'],split=r['split'],
            family=r['family'],static_count=len(local),eligible=int(seen.sum()),excluded=int((~seen).sum())))
    assert matched_keys==set(old_rows),'All216 same-span/anchor records must map exactly'
    assert not({r['source_sha256'] for r in meta['static_bags'] if r['split']=='train'}&
               {r['source_sha256'] for r in meta['static_bags'] if r['split']=='probe'})
    old_hash=sha(P/'static_contract_audit/RESULTS.json');frozen_hash=sha(S/'FROZEN.json')
    write(E/'RULES_FROZEN.json',dict(at=stamp(),
        target='Zero observed keypoint change in synthetic repeated-pose inputs only; not wrong signs, semantic omissions, real physical immobility, or completed meaning.',
        threshold='Fixed saved logit>=0 from the frozen learning contract; no adjustment.',
        observation='Only relevant-body observation_mask==1 enters static false-progression denominator. Mask0 is unknown observation, recorded separately and never a negative outcome.',
        before_after='Match216 saved P audit rows on seed/bag/candidate/span/start-middle-end source pose. Representation changes to body-only, so not identical model input. These FIT examples are included in the new augmentation pool: training diagnostic, NOT new validation.',
        new_probe='People08/09,8public source clips, not trained by this head; upstream-exposed diagnostic. No old same-input probe baseline exists.',
        denominator='Generated candidates/anchors/view/reverse/seed are correlated and never reported as distinct people.',
        selection='All prepared static entries; no score-dependent selection or threshold tuning.',
        prepared_metadata=metadata_rows,matched_prior_rows=len(matched_keys),training_frozen_sha256=frozen_hash,
        pins={str(p):sha(p) for p in [Path(__file__),B/'body_core.py',S/'FROZEN.json',S/'INPUTS_COMPLETE.json',
            S/'FIT_LABEL_AUDIT.json',S/'TRAINING_COMPLETE.json',N/'TRAINING_INPUT_INDEX.json',
            P/'static_contract_audit/FROZEN.json',P/'static_contract_audit/RESULTS.json']},
        input_pins=pins,no_model_inference=True))
    # Open new static prediction arrays only after criteria and mapping are frozen.
    rows=[];matched=[];prediction_pins={}
    for run in train['rows']:
        seed=run['seed'];assert seed in [42,43,44]
        assert sha(run['static_predictions'])==run['static_predictions_sha256']
        prediction_pins[run['static_predictions']]=run['static_predictions_sha256']
        pred=np.load(run['static_predictions'])
        for i,local in prepared.items():
            family=local[0]['family'];scores=pred[f'logits_{i}'][:,old.F.index(family)]
            assert len(scores)==len(local) and np.isfinite(scores).all()
            for m,score in zip(local,scores):
                result=dict(**m,seed=seed,score=float(score),predicted_progression=bool(score>=0))
                rows.append(result)
                key=(seed,i,m['candidate_index'],m['anchor'])
                if key in old_rows:
                    pr=old_rows[key]
                    assert pr['static_false_progression']==(pr['static_logit']>=0)
                    matched.append(dict(**result,previous_score=pr['static_logit'],
                        previous_false_progression=pr['static_false_progression'],
                        new_false_progression=bool(score>=0),
                        in_new_training_augmentation_pool=True,
                        scope='matched FIT training diagnostic, not held-out validation'))
        print('saved static evaluation seed',seed,flush=True)
    assert len(matched)==216
    old_comparison=[]
    for seed in [42,43,44]:
        rr=[r for r in matched if r['seed']==seed];ok=[r for r in rr if r['observation_eligible']]
        old_comparison.append(dict(seed=seed,matched_rows=len(rr),observed_eligible=len(ok),
            excluded_unobserved=len(rr)-len(ok),
            previous_false_progression=sum(r['previous_false_progression'] for r in ok),
            new_false_progression=sum(r['new_false_progression'] for r in ok),
            corrected=sum(r['previous_false_progression'] and not r['new_false_progression'] for r in ok),
            regressed=sum(not r['previous_false_progression'] and r['new_false_progression'] for r in ok),
            scope='training diagnostic on same source pose/span; input representation now body-only'))
    report=dict(at=stamp(),rules_sha256=sha(E/'RULES_FROZEN.json'),rows=rows,
        summary=summarize(rows,['seed','split','family','duration_bucket','anchor']),
        by_family=summarize(rows,['seed','split','family']),
        by_duration=summarize(rows,['seed','split','duration_bucket']),
        by_anchor=summarize(rows,['seed','split','anchor']),
        overall=summarize(rows,['seed','split']),
        matched_prior_rows=matched,matched_prior_summary=old_comparison,
        prediction_pins=prediction_pins,
        unique_sources={split:len({r['clip_id'] for r in meta['static_bags'] if r['split']==split}) for split in ['train','probe']},
        unique_people={split:sorted({r['person'] for r in meta['static_bags'] if r['split']==split}) for split in ['train','probe']},
        no_old_complete_static_baseline=True,no_old_model_reinference=True,no_new_model_reinference=True,
        upstream_probe_exposed=True,private_access=False,final8=False,web_changed=False,
        actual_motion_regression_separate=True,
        limitation='Synthetic constant input rejection does not establish normal/short-motion preservation, detector-jitter robustness or real-world feedback correctness. Matched old216 cases are training diagnostics; new probe has no saved prior baseline.')
    assert sha(S/'FROZEN.json')==frozen_hash and sha(P/'static_contract_audit/RESULTS.json')==old_hash
    write(E/'COMPARISON.json',report)
    for r in report['overall']: print(r,flush=True)
    for r in old_comparison: print(r,flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        dest=S/'static_evaluation';dest.mkdir(exist_ok=True)
        (dest/f'FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
