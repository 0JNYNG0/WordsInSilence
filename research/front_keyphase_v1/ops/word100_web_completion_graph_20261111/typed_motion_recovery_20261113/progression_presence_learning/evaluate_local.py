"""Local saved-prediction evaluation; no model inference of any kind.

Old extent selections come verbatim from the preceding saved baseline records.
Full candidate body-gate arrays were not saved there: raw any-progression is
therefore explicitly NOT described as gated runtime performance.
"""
from presence_core import *
from collections import defaultdict, Counter
import time, traceback


def summarize(rows, keys, metrics):
    groups=defaultdict(list)
    for r in rows: groups[tuple(r[k] for k in keys)].append(r)
    return [dict(**dict(zip(keys,k)),rows=len(rr),
                 **{name:sum(int(fn(r)) for r in rr) for name,fn in metrics.items()})
            for k,rr in sorted(groups.items())]


def main():
    E=P/'local_evaluation'; E.mkdir(exist_ok=True)
    assert not (E/'COMPARISON.json').exists(), 'Completed evaluation cannot rerun'
    assert not (E/'RULES_FROZEN.json').exists(), 'Explicit audited recovery required'
    rows,data=labeled_data()
    training=read(P/'TRAINING_COMPLETE.json')
    source=read(N/'local_bag_evaluation/COMPARISON.json')
    labels=read(P/'FIT_LABEL_AUDIT.json')
    old_rows={(r['seed'],r['bag_index']):r for r in source['rows']}
    old_partial={(r['seed'],r['bag_index'],r['candidate_index']):r for r in source['partial_rows']}
    rules=read(N/'local_bag_evaluation/RULES_FROZEN.json')
    masks=rules['partial_masks']; assert len(rows)==len(masks)
    train_hash=sha(P/'FROZEN.json')
    source_hash=sha(N/'local_bag_evaluation/COMPARISON.json')
    pins={str(p):sha(p) for p in [Path(__file__),P/'presence_core.py',P/'FROZEN.json',
        P/'TRAINING_COMPLETE.json',P/'FIT_LABEL_AUDIT.json',N/'TRAINING_INPUT_INDEX.json',
        N/'local_bag_evaluation/RULES_FROZEN.json',N/'local_bag_evaluation/COMPARISON.json']}
    for seed in [42,43,44]:
        for i,(r,d) in enumerate(zip(rows,data)):
            prev=old_rows[seed,i]
            assert all(prev[k]==r[k] for k in ['clip_id','split','person','view','reverse','family','origin'])
            ix=prev['baseline']['index']
            if ix is not None:
                assert list(map(int,d['spans'][ix]))==prev['baseline']['span']
                assert bool(d['full'][ix])==prev['baseline']['known_positive']
            assert np.array_equal(np.flatnonzero(d['partial']),
                np.array([c['candidate_index'] for c in masks[i]['candidates']],dtype=int))
    write(E/'RULES_FROZEN.json',dict(at=stamp(),
        primary='Read saved separate progression-head logits; fixed logit>=0.',
        labels='Only existing full-compatible extent and same frozen partial mask are progression-positive; reviewed continuous local hold is progression-negative. Other candidates remain unknown, excluded from accuracy/recall/FP denominators.',
        old_extent='Exactly reuse saved previous baseline span/index. Do not rerank, shorten, replace or hide it. Measure support independently.',
        any_progression='Raw binary prediction across all existing visibility-qualified candidates. Full frozen body-gate logits are unavailable here; this is NOT body-gated runtime output. Count any known-positive support and unknown-only support separately.',
        old_chosen_support='Saved baseline chosen span already passed frozen body gate. Supporting binary>=0 is measured on exactly that index.',
        partial_comparison='Same candidate IDs and source masks as previous null diagnostic. Its saved body-gate status is reused only for those partial candidates; compare null suppression and separate progression-positive decisions.',
        durations='Fixed bins4..6,7..10,11..15,16..21,22+ frames; report matched-FIT-bucket scope separately, never adjust thresholds using outcomes.',
        denominators='Overlapping candidate windows, observation views, reverse transforms and3seeds are dependent; report distinct public source clips/people separately. Probe08/09 have upstream exposure, not final evaluation.',
        no_semantic_completion=True,no_educational_correctness=True,no_model_inference=True,
        training_frozen_sha256=train_hash,old_local_comparison_sha256=source_hash,
        pins=pins))
    # First read of the new saved prediction arrays is BELOW frozen rules.
    candidate_rows=[]; bag_rows=[]; partial_rows=[]; unknown_rows=[]; prediction_pins={}
    for run in training['rows']:
        seed=run['seed']; assert seed in [42,43,44]
        assert sha(run['predictions'])==run['predictions_sha256']
        prediction_pins[run['predictions']]=run['predictions_sha256']
        saved=np.load(run['predictions'])
        for i,(r,d) in enumerate(zip(rows,data)):
            j=old.F.index(r['family']); scores=saved[f'logits_{i}'][:,j]
            assert len(scores)==len(d['known']) and np.isfinite(scores).all()
            positive=scores>=0; previous=old_rows[seed,i]; chosen=previous['baseline']
            matched=set(labels['matched_buckets'][str(r['family'])])
            common=dict(seed=seed,bag_index=i,clip_id=r['clip_id'],person=r['person'],
                split=r['split'],family=r['family'],view=r['view'],reverse=r['reverse'])
            for k in np.flatnonzero(d['known']):
                k=int(k); target=bool(d['y'][k]); kind='full' if d['full'][k] else 'partial' if d['partial'][k] else 'hold'
                a,b=map(int,d['spans'][k]); duration=b-a+1; bucket_id=int(d['buckets'][k])
                candidate_rows.append(dict(**common,candidate_index=k,span=[a,b],
                    label_kind=kind,expected_progression=target,predicted_progression=bool(positive[k]),
                    score=float(scores[k]),frames=duration,duration_bucket=bucket_id,
                    duration_bucket_in_fit_matched=bool(bucket_id in matched),
                    correct=bool(target==bool(positive[k])),semantic_completion=None))
            unknown=~d['known']
            unknown_rows.append(dict(**common,unknown_candidates=int(unknown.sum()),
                predicted_progression_on_unknown=int((positive&unknown).sum()),
                excluded_from_performance_denominator=True))
            selected_index=chosen['index']; support=None if selected_index is None else bool(positive[selected_index])
            known_progress=d['known']&(d['y']>0)
            any_known=bool((positive&known_progress).any())
            any_raw=bool(positive.any())
            any_unknown=bool((positive&unknown).any())
            if selected_index is None:
                state='no_prior_extent_body_gate_admission'
            elif support:
                state='old_extent_with_progression_prediction'
            elif any_raw:
                state='progression_predicted_elsewhere_extent_uncertain'
            else:
                state='no_progression_prediction_no_semantic_omission_claim'
            bag_rows.append(dict(**common,origin=r['origin'],candidate_count=len(scores),
                known_candidates=int(d['known'].sum()),unknown_candidates=int(unknown.sum()),
                any_progression_raw=any_raw,any_known_positive_supported=any_known,
                any_unknown_progression_prediction=any_unknown,
                only_unknown_progression=bool(any_raw and not (positive&d['known']).any()),
                prior_body_gate_admitted_count=previous['body_gate_admitted'],
                old_chosen_index=selected_index,old_chosen_span=chosen['span'],
                old_chosen_known_full=chosen['known_positive'],
                old_chosen_supported=support,old_extent_unchanged=True,
                known_old_full_extent_supported=bool(chosen['known_positive'] and support),
                known_old_full_extent_unsupported=bool(chosen['known_positive'] and support is False),
                old_hold_false_selection=previous['baseline_false_hold_selection'],
                previous_null_hold_false_selection=previous['new_false_hold_selection'],
                hold_false_positive_any_raw=bool(r['origin']=='reviewed_local_hold' and any_raw),
                hold_false_positive_old_chosen=bool(r['origin']=='reviewed_local_hold' and support),
                state=state,semantic_completion=None))
            for part in masks[i]['candidates']:
                k=part['candidate_index']; prev=old_partial[seed,i,k]
                assert d['partial'][k] and d['known'][k] and d['y'][k]==1
                assert prev['original_span']==part['original_span']
                admitted=prev['old_body_gate_admitted']; new=bool(positive[k])
                partial_rows.append(dict(**common,**part,
                    previous_body_gate_admitted=admitted,
                    previous_null_state=prev['state'],previous_above_null=prev['above_null_ungated'],
                    predicted_progression=new,score=float(scores[k]),
                    old_null_suppressed=prev['state']=='learned_null_suppressed',
                    recovered_from_null_suppression=bool(prev['state']=='learned_null_suppressed' and new),
                    lost_previous_above_null=bool(prev['state']=='candidate_above_null' and not new),
                    after_same_partial_body_gate=bool(admitted and new),semantic_completion=None))
        print('presence saved local evaluation seed',seed,flush=True)
    cm=dict(positive_labels=lambda r:r['expected_progression'],
        negative_labels=lambda r:not r['expected_progression'],
        true_positive=lambda r:r['expected_progression'] and r['predicted_progression'],
        false_negative=lambda r:r['expected_progression'] and not r['predicted_progression'],
        false_positive=lambda r:not r['expected_progression'] and r['predicted_progression'],
        true_negative=lambda r:not r['expected_progression'] and not r['predicted_progression'])
    bm=dict(any_progression_raw=lambda r:r['any_progression_raw'],
        any_known_positive_supported=lambda r:r['any_known_positive_supported'],
        only_unknown_progression=lambda r:r['only_unknown_progression'],
        old_chosen_present=lambda r:r['old_chosen_index'] is not None,
        old_chosen_supported=lambda r:r['old_chosen_supported'] is True,
        old_chosen_known_full=lambda r:r['old_chosen_known_full'],
        known_old_full_extent_supported=lambda r:r['known_old_full_extent_supported'],
        known_old_full_extent_unsupported=lambda r:r['known_old_full_extent_unsupported'],
        old_hold_false_selection=lambda r:r['old_hold_false_selection'],
        previous_null_hold_false_selection=lambda r:r['previous_null_hold_false_selection'],
        hold_false_positive_any_raw=lambda r:r['hold_false_positive_any_raw'],
        hold_false_positive_old_chosen=lambda r:r['hold_false_positive_old_chosen'])
    pm=dict(body_gate_admitted=lambda r:r['previous_body_gate_admitted'],
        old_null_suppressed=lambda r:r['old_null_suppressed'],
        new_progression_positive=lambda r:r['predicted_progression'],
        new_positive_after_saved_body_gate=lambda r:r['after_same_partial_body_gate'],
        recovered_from_null_suppression=lambda r:r['recovered_from_null_suppression'],
        lost_previous_above_null=lambda r:r['lost_previous_above_null'])
    report=dict(at=stamp(),rules_sha256=sha(E/'RULES_FROZEN.json'),prediction_pins=prediction_pins,
        candidate_rows=candidate_rows,
        candidate_summary=summarize(candidate_rows,['seed','split','family','label_kind','duration_bucket'],cm),
        candidate_all_durations=summarize(candidate_rows,['seed','split','family','label_kind'],cm),
        candidate_all_families=summarize(candidate_rows,['seed','split','label_kind'],cm),
        matched_duration_scope=summarize(candidate_rows,['seed','split','family','duration_bucket_in_fit_matched','label_kind'],cm),
        bag_rows=bag_rows,bag_summary=summarize(bag_rows,['seed','split','origin','family'],bm),
        bag_all_families=summarize(bag_rows,['seed','split','origin'],bm),
        partial_rows=partial_rows,
        partial_comparison=summarize(partial_rows,['seed','split','family','duration_bucket'],pm),
        partial_all_families=summarize(partial_rows,['seed','split'],pm),
        unknown_rows=unknown_rows,
        unique_sources={s:len({r['clip_id'] for r in rows if r['split']==s}) for s in ['train','probe']},
        unique_people={s:sorted({r['person'] for r in rows if r['split']==s}) for s in ['train','probe']},
        repeated_views_reverse_seeds_are_not_independent_people=True,
        probe_upstream_exposed=True,old_extent_changed=False,new_model_inference=False,
        old_model_inference=False,all_candidate_body_gate_not_reconstructed=True,
        full_path_regression_separate=True,private_access=False,final8=False,web_changed=False,
        limitation='Candidate-level labels concern observed progression only. Unknown candidates are not false negatives/negatives. No semantic completion, educational error rate, new-user accuracy or gated runtime latency claim.')
    assert sha(P/'FROZEN.json')==train_hash and sha(N/'local_bag_evaluation/COMPARISON.json')==source_hash
    write(E/'COMPARISON.json',report)
    for r in report['candidate_all_families']: print(r,flush=True)
    for r in report['bag_all_families']: print(r,flush=True)


if __name__=='__main__':
    try: main()
    except BaseException:
        dest=P/'local_evaluation'; dest.mkdir(exist_ok=True)
        (dest/f'FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
