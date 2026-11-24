"""Read-only numerical diagnosis of saved point readouts; zero model forward."""
from pathlib import Path
import ast
import collections
import datetime
import hashlib
import json
import sys

import numpy as np

E = Path(__file__).resolve().parent
M = E.parent
H = M.parent
G = H.parent
W = G / 'candidate_description_supervision'
F = [2, 3, 4, 5, 7]
AX = {2: 0, 3: 1, 4: 2, 5: 4, 7: 3}
MODES = ['full_bag_rate_rank', 'old_local_full_plus_fullbag_half']
NORMAL = ['full_normal', 'speed075', 'speed125']
PINS = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path, expected=None):
    path = str(Path(path).resolve())
    actual = sha(path)
    assert expected is None or expected == actual, path
    assert path not in PINS or PINS[path] == actual
    PINS[path] = actual


def read(path):
    pin(path)
    return json.loads(Path(path).read_text('utf-8-sig'))


def key(row):
    return row['seed'], row['clip_id'], row['kind']


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return dict(count=0, min=None, q25=None, median=None, q75=None, max=None)
    assert np.isfinite(values).all()
    quantiles = np.quantile(values, [0, .25, .5, .75, 1]).tolist()
    return dict(count=int(values.size), **dict(zip(['min', 'q25', 'median', 'q75', 'max'], quantiles)))


def point_metrics(x, family):
    blocks = x.reshape(-1, 3, 10, 5)
    q = blocks[:, :, AX[family] * 2, :4]
    return dict(current_norm=np.linalg.norm(q[:, 1], axis=1),
                before_to_current=np.linalg.norm(q[:, 1] - q[:, 0], axis=1),
                current_to_after=np.linalg.norm(q[:, 2] - q[:, 1], axis=1),
                before_to_after=np.linalg.norm(q[:, 2] - q[:, 0], axis=1),
                whole_x_min=x.min(1), whole_x_max=x.max(1),
                whole_context_delta=np.linalg.norm((blocks[:, 2, :, :4] - blocks[:, 0, :, :4]).reshape(-1, 40), axis=1))


def coordinate_indices(family):
    return np.asarray([block * 50 + AX[family] * 10 + c for block in range(3) for c in range(4)])


def relation(first, second):
    if len(first) < 2 or len(second) < 2:
        return 'unobserved'
    if first[-1] <= second[0]:
        return 'forward'
    if second[-1] <= first[0]:
        return 'reverse'
    return 'ambiguous_overlap'


def array(path, expected, names):
    pin(path, expected)
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in names}


def fit_statistics():
    meta = read(M / 'INPUTS_COMPLETE.json')
    pools = collections.defaultdict(list)
    frames = []
    whole = []
    for row in meta['sequences']:
        assert row['split'] == 'train' and row['person'] in [2, 3, 4, 5, 6, 7]
        a = array(row['path'], row['sha256'], ['x', 'mask', 'observed', 'label_provenance', 'timestamp_us', 'source_frame_id'])
        x, code, ts = a['x'], a['label_provenance'], a['timestamp_us']
        assert x.shape == (len(ts), 150) and np.isfinite(x).all()
        assert np.issubdtype(ts.dtype, np.integer) and (np.diff(ts) > 0).all()
        whole.append(x)
        for j, family in enumerate(F):
            metrics = point_metrics(x, family)
            for value, name in [(1, 'direct_motion'), (3, 'own_terminal_hold'), (4, 'other_terminal_hold')]:
                take = (code[:, j] == value) & (a['mask'][:, j] > 0)
                assert np.all(a['observed'][take, j])
                if take.any():
                    pools[family, name].append(x[take])
                for frame in np.flatnonzero(take):
                    frames.append(dict(clip_id=row['clip_id'], person=row['person'], view=row['view'], reverse=row['reverse'],
                        family=family, label=name, source_frame_id=int(a['source_frame_id'][frame]),
                        query_frame_id=int(frame), timestamp_us=int(ts[frame]),
                        **{metric: float(values[frame]) for metric, values in metrics.items()}))
    pools = {k: np.concatenate(v) for k, v in pools.items()}
    assert sum(len(v) for k, v in pools.items() if k[1] == 'direct_motion') == 822
    assert sum(len(v) for k, v in pools.items() if k[1] == 'own_terminal_hold') == 522
    assert sum(len(v) for k, v in pools.items() if k[1] == 'other_terminal_hold') == 126
    summary = []
    for (family, name), x in sorted(pools.items()):
        summary.append(dict(family=family, label=name, points=len(x),
            metrics={k: stats(v) for k, v in point_metrics(x, family).items()},
            per_dimension_min_150=x.min(0).tolist(), per_dimension_max_150=x.max(0).tolist()))
    overlap = []
    for family in F:
        motion = pools[family, 'direct_motion']
        holds = np.concatenate([v for (f, label), v in pools.items() if f == family and label != 'direct_motion'])
        indices = coordinate_indices(family)
        common = np.minimum(motion.max(0), holds.max(0)) >= np.maximum(motion.min(0), holds.min(0))
        mm = point_metrics(motion, family)['before_to_after']
        hm = point_metrics(holds, family)['before_to_after']
        interval = [max(float(mm.min()), float(hm.min())), min(float(mm.max()), float(hm.max()))]
        overlap.append(dict(family=family, direct_motion_points=len(motion), hold_points=len(holds),
            minmax_overlap_dimensions_150=int(common.sum()),
            relevant_body_coordinate_dimensions=12, relevant_overlap_dimensions=int(common[indices].sum()),
            motion_context_delta=stats(mm), hold_context_delta=stats(hm),
            context_delta_range_overlap=interval if interval[0] <= interval[1] else None,
            exact_zero_context_delta_motion=int((mm == 0).sum()),
            note='Marginal range overlap is not proof that the joint features are inseparable or that either observation is semantically wrong.'))
    source_summary = []
    for cid, family, label in sorted({(r['clip_id'], r['family'], r['label']) for r in frames}):
        selected = [r for r in frames if (r['clip_id'], r['family'], r['label']) == (cid, family, label)]
        source_summary.append(dict(clip_id=cid, family=family, label=label, view_reverse_observations=len(selected),
            unique_source_frames=len({r['source_frame_id'] for r in selected}),
            context_delta=stats([r['before_to_after'] for r in selected]),
            current_norm=stats([r['current_norm'] for r in selected])))
    return pools, np.concatenate(whole), dict(summary=summary, motion_hold_overlap=overlap,
        source_summary=source_summary, source_frame_observations=frames,
        interpretation='x150 is already normalized body-relation coordinates plus availability, not pixels or physical speed. Euclidean changes mix normalized coordinate dimensions. Quantiles are descriptive only; they are not new thresholds or labels.')


def main():
    destination = E / 'CAUSE_AUDIT.json'
    assert not destination.exists()
    pin(__file__)
    current = read(E / 'COMPARISON.json')
    old = read(W / 'source_rank_learning/evaluation/COMPARISON.json')
    old_training = read(W / 'source_rank_learning/TRAINING_COMPLETE.json')
    read(W / 'source_rank_learning/evaluation/FROZEN.json')
    frozen = read(E / 'FROZEN.json')
    for path, digest in frozen['protected_pins'].items():
        assert sha(path) == digest
    for path in [W / 'evaluate_rank_learning.py', W / 'source_boundary_core.py', W / 'core.py',
                 G.parent / 'word100_event_phase_learning_20261110/phase_core.py',
                 M / 'model.py', M / 'train.py', E / 'INPUT_PARITY.json']:
        pin(path)
    evaluator_text = (W / 'evaluate_rank_learning.py').read_text('utf-8')
    assert "arrays['boundary_logits']=out['boundary_extended']" in evaluator_text
    training = read(M / 'TRAINING_COMPLETE.json')
    inputs_meta = read(M / 'INPUTS_COMPLETE.json')
    old_parents = {r['seed']: r for r in old_training['rows'] if r['condition'] == 'boundary_extended'}
    for parent in inputs_meta['parent_models']:
        assert parent['model_sha256'] == old_parents[parent['seed']]['model_sha256']
        pin(parent['model'], parent['model_sha256'])  # Hash bytes only; no model load.
    fit_review = read(M / 'SAVED_FIT_VERIFICATION.json')
    assert all(r['normalization_exactly_unchanged'] for r in fit_review['runs'])
    pools, fit_all_x, fit = fit_statistics()
    fit_lo, fit_hi = fit_all_x.min(0), fit_all_x.max(0)
    previous = {key(r): r for r in old['predictions']}
    predictions = {key(r): r for r in current['predictions']}
    assert set(previous) == set(predictions) and len(previous) == 246
    cached = {}
    input_checks = []
    for row in current['inputs']:
        a = array(row['path'], row['sha256'], ['x', 'timestamp_us', 'body_observed'])
        source = array(row['source'], row['source_sha256'], ['z', 'ts'])
        q, ts = source['z'][:, 292:392].reshape(-1, 10, 10), source['ts']
        current50 = np.concatenate([q[..., :4], q[..., 8:9]], -1).reshape(-1, 50)
        lo = np.searchsorted(ts, ts - 133333)
        hi = np.searchsorted(ts, ts + 133333, side='right')
        before = np.stack([current50[start:i+1].mean(0) for i, start in enumerate(lo)])
        after = np.stack([current50[i:end].mean(0) for i, end in enumerate(hi)])
        rebuilt = np.c_[before, current50, after].astype(np.float32)
        observed = np.stack([q[:, AX[f] * 2, 8] > .5 for f in F], 1)
        assert np.array_equal(rebuilt, a['x']) and np.array_equal(ts, a['timestamp_us'])
        assert np.array_equal(observed, a['body_observed'])
        assert np.issubdtype(ts.dtype, np.integer) and (np.diff(ts) > 0).all()
        cached[row['clip_id'], row['kind']] = a
        input_checks.append(dict(clip_id=row['clip_id'], kind=row['kind'], frames=len(ts),
            before_current_after_exact=True, signed_coordinates_exact=True, family_side_masks_exact=True,
            original_PTS_exact=True, PTS_step_us=stats(np.diff(ts)),
            relevant_body_mask_counts={str(f): int(observed[:, j].sum()) for j, f in enumerate(F)},
            whole_x_range=[float(a['x'].min()), float(a['x'].max())]))
    cards, pairs = [], []
    for row in current['rows']:
        k = key(row)
        old_meta, new_meta = previous[k], predictions[k]
        assert old_meta['source_sha256'] == new_meta['source_sha256']
        assert old_meta['source'] == new_meta['source']
        old_logits = array(old_meta['path'], old_meta['sha256'], ['boundary_logits'])['boundary_logits'][:, :, 2]
        new_data = array(new_meta['path'], new_meta['sha256'], ['progression_logits', 'timestamp_us', 'body_observed'])
        new_logits = new_data['progression_logits']
        a = cached[row['clip_id'], row['kind']]
        x, ts, mask = a['x'], a['timestamp_us'], a['body_observed']
        assert old_logits.shape == new_logits.shape == mask.shape == (len(ts), 5)
        assert np.array_equal(new_data['timestamp_us'], ts) and np.array_equal(new_data['body_observed'], mask)
        for mode in MODES:
            results = {}
            for original, saved in zip(row['conditions'][mode]['cards'], row['observed_progression'][mode]):
                assert original['card'] == saved['card']
                family = original['family']
                j = F.index(family)
                span = original['chosen']['span'] if original['chosen'] else None
                selected = np.arange(span[0], span[1]+1) if span else np.zeros(0, dtype=int)
                selected = selected[mask[selected, j]]
                new_ids = selected[new_logits[selected, j] >= 0]
                old_ids = selected[old_logits[selected, j] >= 0]
                assert new_ids.tolist() == saved['support_indices']
                all_new = np.flatnonzero(mask[:, j] & (new_logits[:, j] >= 0))
                all_old = np.flatnonzero(mask[:, j] & (old_logits[:, j] >= 0))
                outside = np.setdiff1d(all_new, selected)
                observed_body = np.flatnonzero(mask[:, j])
                witness = np.asarray(original.get('truth', {}).get('witness', []), dtype=int)
                metrics = point_metrics(x, family)
                fit_motion = pools[family, 'direct_motion']
                hold = np.concatenate([v for (f, label), v in pools.items() if f == family and label != 'direct_motion'])
                inside_fit = (x[selected] >= fit_lo) & (x[selected] <= fit_hi)
                relevant = coordinate_indices(family)
                motion_box = (x[selected][:, relevant] >= fit_motion[:, relevant].min(0)) & (x[selected][:, relevant] <= fit_motion[:, relevant].max(0))
                hold_box = (x[selected][:, relevant] >= hold[:, relevant].min(0)) & (x[selected][:, relevant] <= hold[:, relevant].max(0))
                cause = ('no_selected_candidate' if span is None else
                         'body_observations_below_two' if len(selected) < 2 else
                         'progression_supported' if len(new_ids) >= 2 else
                         'score_support_below_two_entire_sequence' if len(all_new) < 2 else
                         'score_support_elsewhere_outside_unchanged_extent')
                result = dict(seed=k[0], clip_id=k[1], kind=k[2], condition=mode, card=original['card'], family=family,
                    span=span, selected_frames=(span[1]-span[0]+1) if span else 0,
                    selected_body_observed=len(selected), selected_new_support=len(new_ids), selected_parent_support=len(old_ids),
                    whole_sequence_body_observed=len(observed_body), whole_sequence_new_support=len(all_new),
                    whole_sequence_parent_support=len(all_old), outside_extent_new_support=outside.tolist(),
                    selected_new_indices=new_ids.tolist(), selected_parent_indices=old_ids.tolist(),
                    selected_new_logits=stats(new_logits[selected, j]), selected_parent_logits=stats(old_logits[selected, j]),
                    whole_observed_new_logits=stats(new_logits[observed_body, j]), cause_category=cause,
                    reviewed_witness=list(map(int, witness)),
                    new_nonnegative_at_reviewed_witness=int((new_logits[witness, j] >= 0).sum()),
                    parent_nonnegative_at_reviewed_witness=int((old_logits[witness, j] >= 0).sum()),
                    selected_input_metrics={name: stats(values[selected]) for name, values in metrics.items()},
                    selected_coordinates_outside_full_FIT_minmax=int((~inside_fit).sum()),
                    selected_coordinates_total=int(inside_fit.size),
                    selected_frames_inside_family_motion_marginal_box=int(motion_box.all(1).sum()),
                    selected_frames_inside_family_hold_marginal_box=int(hold_box.all(1).sum()),
                    missing_support_is_not_incorrect_sign=True)
                cards.append(result)
                results[original['card']] = dict(new=ts[new_ids], old=ts[old_ids])
            for saved in row['observed_progression_order'][mode]:
                first, second = [results[c] for c in saved['cards']]
                new_order, old_order = relation(first['new'], second['new']), relation(first['old'], second['old'])
                assert new_order == saved['observed_body_progression_order']['state']
                pairs.append(dict(seed=k[0], clip_id=k[1], kind=k[2], condition=mode, cards=saved['cards'],
                    old_boundary_progress_on_same_L_span=old_order, new_progress_on_same_L_span=new_order,
                    original_L_extent_order=saved['original_candidate_extent_order']['state']))
    grouped_cards = []
    for seed, mode, kind, family in sorted({(r['seed'], r['condition'], r['kind'], r['family']) for r in cards}):
        subset = [r for r in cards if (r['seed'], r['condition'], r['kind'], r['family']) == (seed, mode, kind, family)]
        grouped_cards.append(dict(seed=seed, condition=mode, kind=kind, family=family, cards=len(subset),
            new_support_0=sum(r['selected_new_support'] == 0 for r in subset),
            new_support_1=sum(r['selected_new_support'] == 1 for r in subset),
            old_support_below2=sum(r['selected_parent_support'] < 2 for r in subset),
            new_support_below2=sum(r['selected_new_support'] < 2 for r in subset),
            causes=dict(collections.Counter(r['cause_category'] for r in subset))))
    pair_summary = []
    for seed, mode, kind in sorted({(r['seed'], r['condition'], r['kind']) for r in pairs}):
        subset = [r for r in pairs if (r['seed'], r['condition'], r['kind']) == (seed, mode, kind)]
        pair_summary.append(dict(seed=seed, condition=mode, kind=kind, pairs=len(subset),
            parent_point_order=dict(collections.Counter(r['old_boundary_progress_on_same_L_span'] for r in subset)),
            new_point_order=dict(collections.Counter(r['new_progress_on_same_L_span'] for r in subset)),
            old_extent_order=dict(collections.Counter(r['original_L_extent_order'] for r in subset))))
    failures = [r for r in cards if r['kind'] in NORMAL and r['family'] in [2, 3] and r['selected_new_support'] < 2]
    assert len(failures) == 26 and all(r['selected_new_support'] == 0 for r in failures)
    assert all(r['selected_body_observed'] >= 2 and r['whole_sequence_new_support'] < 2 for r in failures)
    assert all(r['selected_parent_support'] == 0 for r in failures)
    prior_B = read(H / 'body_progression_learning/EVALUATION_REVIEW.json')
    prior_B_contract = read(H / 'body_progression_learning/FROZEN.json')
    pin(H / 'body_progression_learning/EVALUATION_REVIEW.txt')
    pin(H / 'body_progression_learning/body_core.py')
    c5 = read(H / 'observed_action_anchor_design/RESULTS.json')
    correction = read(H / 'observed_action_anchor_design/CORRECTION_BOUNDARY_HOLD.json')
    probe = read(M / 'probe_review/RESULTS.json')  # Saved JSON only, no probe arrays.
    probe_metrics = [dict(seed=r['seed'], by_provenance=r['by_provenance']) for r in probe['runs']]
    report = dict(at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
        status='SAVED_NUMERICAL_CAUSE_AUDIT_NO_NEW_FORWARD_NO_TUNING', reviewer='Codex progress_runtime_eval',
        scope=dict(DEV_scenarios=82, seeds=[42,43,44], conditions=MODES, same_seed_source_comparisons=246,
                   FIT_sources=24, FIT_sequences=144, source_frames_are_repeated_views_and_reverse=True),
        findings=[
            'All13 normal/slow/fast failed card readouts per fixed condition have zero new positive points inside the extent despite6-15 observed body frames. Twelve whole-sequence scores have zero qualifying points and one has only1. The current extent is not hiding a sufficient support island for these cases.',
            'The matching saved boundary_extended progression channel also has zero support inside all13 failed normal/speed spans. This pre-existing point-readout generalization gap must not be labeled a regression from old extent order to new progression order.',
            'REAL11 family3 fails all three speeds/seeds. REAL13 family2 fails normal and slow for seeds43/44. Family/context range statistics are descriptive evidence about representation coverage, not sign correctness or a tuned decision boundary.',
            'Some edited cases have nonnegative points outside the unchanged candidate extent. Those points have not been shown to be the intended movement. Record them without reselection or calling them a correct association.',
            'The separate exposed-probe JSON uses the same point-readout target and does show loss of direct-motion support after training. This is distinct from comparing old extent order with a different new order metric.',
            'Source/current50, context ordering, family side/axis mask and integer PTS are exact across all82 arrays. No runtime sign, axis, mask or PTS transformation mismatch was found. Saved FIT parity144/144 and normalization preservation remain valid.'
        ],
        normal_speed_failures=failures, all_card_diagnostics=cards, all_pair_diagnostics=pairs,
        card_summary=grouped_cards, pair_summary=pair_summary, DEV_input_checks=input_checks, FIT_input_statistics=fit,
        saved_probe_json_reference=dict(path=str(M/'probe_review/RESULTS.json'), sha256=sha(M/'probe_review/RESULTS.json'),
            status=probe['status'], by_seed=probe_metrics, probe_arrays_opened_by_this_audit=0),
        prior_work=dict(
            body_progression_learning=dict(status=prior_B.get('status'), contract_architecture=prior_B_contract['architecture'],
                distinction='B is a1160-value candidate-window readout with temporal-third/std pooled body-masked encoder latent plus body relations; it preserves family2. M is150-value local body context only, no latent, pointwise five-family logits. B normal-window support is not point-order support and its short-progression misses were already retained.'),
            observed_action_anchor_design=dict(status=c5['status'], scope=c5['partial_scope'],
                corrected_hold_warning=correction['correction'],
                distinction='C5 audited earlier saved boundary outputs and proposed a pure point-order contract; it did not validate new DEV point generalization. M already supplies reviewed hold negatives, so repeating the same hold-supervision run is unsupported.')),
        next_single_experiment_proposal=dict(
            status='DESIGN_ONLY_REQUIRES_ROOT_REVIEW_NO_RUN',
            question='Does exposing current body-relative position and local context differences as separate input blocks improve head-level performer generalization under equal training conditions?',
            information_contract='The existing150 values ALREADY contain temporal information: before/current/after context. Current plus before-current plus after-current is an invertible coordinate reparameterization when masks/context availability are retained, not additional motion information. Its rationale is conditioning/inductive bias, not a claim that the old input lacked time or that all failures are pose memorization.',
            prerequisites='First use the reported FIT family/source/frame distributions to specify one representation change. Keep source and hand-attribute labels separate; do not use DEV failures to choose a threshold, epoch, family exception or coefficient.',
            design='One paired FIT02-07 six-fold leave-one-performer-out diagnostic: raw before/current/after150 versus current body-relative position plus before-current and after-current coordinate blocks, retaining all original availability and150 dimensions. Use the same small150->96->5 architecture, same scratch initialization per fold/seed, same known labels and balanced family/source/class objective, same fixed steps and final checkpoint. Fit each condition normalization only on the five training performers. All views, reversal and frames of the sixth person belong exclusively to that fold evaluation. Report every fold and seed; no epoch/threshold/model selection. No probe or82-DEV input is used to choose the representation.',
            parent_leakage='Do NOT copy boundary_extended or the M head into a leave-one-FIT-performer-out experiment: both parents have trained on all six FIT people. Scratch matched initialization is required for honest head-supervision exclusion. This diagnostic is not a replacement of the deployed fine-tuning lineage.',
            upstream_exposure='The stored x150 already uses fixed upstream body-relation normalization whose exposure history must remain documented. Fold-local normalizers prevent new fold leakage but do not erase upstream exposure. Before a strict performer-generalization claim, verify/reconstruct body relations and fold-fit their normalization without held-out statistics; otherwise name the result head-label-held-out diagnostic on previously exposed public features, not unseen end-to-end or fresh-final accuracy.',
            nonduplication='This tests a relative-context representation and performer generalization. It does not repeat M hold labels, C5 anchor extraction, B latent-window presence training, rank association, or an overlap tolerance adjustment.',
            limit='A new representation is a hypothesis, not an established fix. Marginal ranges and source coverage cannot prove temporal ambiguity or semantic error. No performance claim or deployment follows this audit.'),
        original_L_extent_G2_pass=False, semantic_completion=None, G2_pass=None, G3_pass=None,
        model_forward_calls=0, old_model_loads=0, training_runs=0, thresholds_changed=False,
        candidate_reselection=False, probe_arrays_opened=0, private_access=False, final8_access=False, web_changed=False,
        protected_hashes_verified=477, pins=PINS)
    for path, digest in PINS.items():
        assert sha(path) == digest, path
    for path, digest in frozen['protected_pins'].items():
        assert sha(path) == digest
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=report['status'], failed_normal_speed_cards=len(failures),
        old_point_support_also_zero=sum(r['selected_parent_support']==0 for r in failures),
        per_condition=True, card_diagnostics=len(cards), pair_diagnostics=len(pairs),
        fit_frame_observations=len(fit['source_frame_observations']), model_forward_calls=0,
        destination=str(destination)), ensure_ascii=False))


if __name__ == '__main__':
    main()
