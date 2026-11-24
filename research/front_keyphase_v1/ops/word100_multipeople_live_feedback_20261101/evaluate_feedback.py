"""Fixed public feedback evaluation; root executes after bank/calibration freeze.

prepare: freeze96 cases and derive three-reference cutoffs from24 SAVED FIT
pair comparisons (never re-align/re-infer them). run: compare each DEV/control
against nine references once; reuse those pair results for the ID01-03 subset.
No detector, RGB encoder, head, training, user video or old experiment import.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

sys.dont_write_bytecode = True
U = Path(__file__).resolve().parent
TARGETS = ['R_index_only', 'R_open_four', 'L_open_four', 'R_closed_four',
           'L_closed_four', 'R_higher', 'L_higher']
FAMILIES = ['R_shape', 'L_shape', 'R_angles', 'L_angles', 'hand_relation', 'arm_path', 'hand_axis']
STATUS = ['SIMILAR_REFERENCE_RANGE', 'DIFFERENCE_CANDIDATE', 'INSUFFICIENT_EVIDENCE', 'NOT_REQUIRED_REFERENCE_REST']
COUNTS = {'same_expression': 12, 'different_target_expression': 36,
          'speed_075': 12, 'speed_125': 12, 'main_R_hand_missing': 12, 'short_R_shape_axis': 12}


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def save(p, value, compact=False):
    p = Path(p).resolve()
    if not p.is_relative_to(U):
        raise ValueError('New unit outputs only')
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False,
                  indent=None if compact else 2, separators=(',', ':') if compact else None)


def verify(pins):
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError('Pinned source changed: ' + path)


def load_engine():
    spec = importlib.util.spec_from_file_location('_fixed_public_feedback_engine', U / 'engine.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.FAMILIES != FAMILIES:
        raise ValueError('Family order changed')
    return module


def load_arrays(row, names):
    if sha(row['input']) != row['input_sha256'] or sha(row['probability_file']) != row['probability_sha256']:
        raise ValueError('Stored input/posterior changed')
    with np.load(row['input'], allow_pickle=False) as z:
        a = {k: z[k].copy() for k in ('x', 'mask', 'rgb', 'frame_id', 'timestamp_us')}
    with np.load(row['probability_file'], allow_pickle=False) as z:
        if (not np.array_equal(z['frame_id'], a['frame_id']) or
                not np.array_equal(z['timestamp_us'], a['timestamp_us']) or z['targets'].tolist() != TARGETS):
            raise ValueError('Probability axes/target order differ')
        a['probabilities'] = z['probabilities'].copy()
    n = len(a['frame_id'])
    if (a['x'].shape != (n, 69) or a['mask'].shape != (n, 69) or a['mask'].dtype != bool or
            a['probabilities'].shape != (n, 7) or np.any(np.diff(a['timestamp_us']) <= 0)):
        raise ValueError('Native full-axis schema differs')
    a['feature_names'] = names
    return a


def common_inputs():
    frozen = read(U / 'FROZEN.json')
    bank = read(U / 'REFERENCE_BANK.json')
    predictions = read(U / 'BANK_PREDICTIONS.json')
    normal = read(U / 'NORMAL_RANGES.json')
    refs = bank['rows']
    dev = [r for r in predictions['rows'] if r['role'] == 'development']
    if (len(refs) != 36 or len(dev) != 12 or
        Counter((r['word_id'], r['person']) for r in refs) !=
            {(w, p): 1 for w in frozen['word_ids'] for p in range(1, 10)} or
        Counter((r['word_id'], r['person']) for r in dev) !=
            {(w, p): 1 for w in frozen['word_ids'] for p in [11, 12, 13]} or
        any(r['role'] != 'fit' for r in refs)):
        raise ValueError('Fixed36 FIT/12 DEV bank differs')
    if (normal['development_used'] or normal['complete_pair_opportunities'] != 288 or
        normal['engine_sha256'] != sha(U / 'engine.py') or
        normal['reference_bank_sha256'] != sha(U / 'REFERENCE_BANK.json') or
        bank['observer'] != read(U / 'TRAINING_REPORT.json')['adoption']['chosen_condition'] or
        frozen['multi_reference']['coverage_min'] != .65 or
        frozen['multi_reference']['normal_support_min'] != 2):
        raise ValueError('Observer/calibration/frozen contract mismatch')
    verify(bank['source_pins'])
    return frozen, bank, normal, sorted(dev, key=lambda r: r['clip_id'])


def subset_ranges(bank, normal):
    """Use only saved query01-03 to the other TWO ID01-03 FIT references."""
    rows = {r['clip_id']: r for r in bank['rows']}
    subset = [r for r in bank['rows'] if r['person'] <= 3]
    saved = {(r['query'], r['reference']): r for r in normal['pair_index']}
    if len(saved) != 288:
        raise ValueError('Calibration inventory is not288 unique pairs')
    values = {f: [] for f in FAMILIES}
    ledger = []
    pairs = []
    for query in sorted(subset, key=lambda r: r['clip_id']):
        candidates = [r for r in subset if r['word_id'] == query['word_id'] and r['person'] != query['person']]
        if len(candidates) != 2:
            raise ValueError('Three-reference leave-person-out must have exactly two candidates')
        loaded = []
        for reference in candidates:
            item = saved[(query['clip_id'], reference['clip_id'])]
            verify({item['path']: item['sha256']})
            comparison = read(item['path'])
            if (comparison['id'] != reference['clip_id'] or comparison['query_id'] != query['clip_id'] or
                    comparison['person'] != reference['person'] or comparison['query_person'] != query['person']):
                raise ValueError('Saved calibration identity mismatch')
            loaded.append((comparison, item))
            pairs.append(item)
        # Both remain selected. Sorting preserves the same whole-reference rule.
        for rank, (comparison, source) in enumerate(sorted(loaded, key=lambda r: r[0]['distance']), 1):
            for si, stage in enumerate(comparison['stages']):
                for family in FAMILIES:
                    v = stage['families'][family]
                    required = v.get('comparison_required', True)
                    usable = bool(required and v['coverage'] is not None and v['coverage'] >= .65 and
                                  v['distance'] is not None and v.get('peak_distance') is not None)
                    if usable:
                        values[family].append([v['distance'], v['peak_distance']])
                    ledger.append(dict(query=query['clip_id'], reference=comparison['id'], selected_rank=rank,
                        canonical_stage_id=stage['canonical_stage_id'], family=family,
                        comparison_required=required, usable_for_range=usable, distance=v['distance'],
                        peak_distance=v.get('peak_distance'), coverage=v['coverage'], expected=v['expected'],
                        observed=v['observed'], query_expected=v['query_expected'], query_observed=v['query_observed'],
                        path=source['path'], sha256=source['sha256'], pointer=f'/stages/{si}/families/{family}'))
    if len(pairs) != 24 or len(ledger) != 39 * 2 * 7:
        raise ValueError('Fixed24 saved subset pairs /546 stage-family denominator differs')
    ranges = {}
    counts = {}
    for f in FAMILIES:
        v = np.asarray(values[f], float).reshape(-1, 2)
        ranges[f] = dict(p90=float(np.quantile(v[:, 0], .95)) if len(v) else None,
                         peak=float(np.quantile(v[:, 1], .95)) if len(v) else None)
        ff = [r for r in ledger if r['family'] == f]
        counts[f] = dict(possible=len(ff), supported=len(v),
            not_required_reference_rest=sum(not r['comparison_required'] for r in ff),
            insufficient=sum(r['comparison_required'] and not r['usable_for_range'] for r in ff),
            expected_cells=sum(r['expected'] for r in ff), observed_cells=sum(r['observed'] for r in ff),
            query_expected_cells=sum(r['query_expected'] for r in ff), query_observed_cells=sum(r['query_observed'] for r in ff))
    return dict(status='FIT_ONLY_SUBSET_FROM_SAVED_COMPARISONS', query_people=[1, 2, 3], query_clips=12,
        candidate_references_per_query=2, pair_opportunities=24, stage_family_opportunities=len(ledger),
        ranges=ranges, counts=counts, pair_index=pairs, ledger=ledger, quantile=.95,
        source_normal_ranges_sha256=sha(U / 'NORMAL_RANGES.json'),
        query_development_used=False, new_inference_calls=0, new_alignment_calls=0,
        limitation='Same saved comparisons and .95 pooled family rule, but fewer FIT examples alter calibration; not a pure bank-size causal effect.')


def main_indices(a, intervals):
    times = a['timestamp_us'] / 1e6
    inside = np.zeros(len(times), bool)
    for lo, hi in intervals:
        inside |= (times >= lo) & (times <= hi)
    return np.flatnonzero(inside)


def short_indices(a, intervals):
    """Two consecutive native samples nearest midpoint of the main-interval hull."""
    main = main_indices(a, intervals)
    pairs = [int(i) for i in main[:-1] if i + 1 in set(main)]
    if not pairs:
        raise ValueError('No two adjacent main-interval frames; do not expand or choose a scored window')
    midpoint = (float(a['timestamp_us'][main[0]]) + float(a['timestamp_us'][main[-1]])) / 2
    first = min(pairs, key=lambda i: (abs((int(a['timestamp_us'][i]) + int(a['timestamp_us'][i+1])) / 2 - midpoint), i))
    return [first, first + 1]


def right_missing_columns(names):
    # Hand observations and their dependent measurements only. Pose arm survives.
    result = []
    for i, name in enumerate(names):
        right_hand = name.startswith('R_') and not any(name.startswith('R_' + p) for p in ('pose_wrist_', 'elbow_'))
        bilateral = name in ('hands_wrist_delta_x', 'hands_wrist_delta_y', 'hands_wrist_distance')
        if right_hand or bilateral:
            result.append(i)
    return result


def transform(a, task):
    """No feature re-extraction: known transformations of stored vectors only."""
    original = {k: v.copy() if isinstance(v, np.ndarray) else list(v) for k, v in a.items()}
    kind = task['kind']
    n = len(a['x'])
    mapping = np.arange(n, dtype=int)
    b = {k: v.copy() if isinstance(v, np.ndarray) else list(v) for k, v in a.items()}
    ledger = dict(kind=kind, original_frame_count=n, original_main_indices=task['main_indices'],
                  model_or_encoder_calls=0, new_RGB_values_computed=False,
                  RGB_source_vectors_only=True, semantic_error_truth=None)
    if kind.startswith('speed_'):
        factor = task['speed_factor']
        mapping = np.round(np.linspace(0, n - 1, round(n / factor))).astype(int)
        for k in ('x', 'mask', 'rgb', 'probabilities'):
            b[k] = a[k][mapping].copy()
        b['frame_id'] = np.arange(len(mapping), dtype=np.int64)
        delta = int(np.median(np.diff(a['timestamp_us'])))
        b['timestamp_us'] = int(a['timestamp_us'][0]) + np.arange(len(mapping), dtype=np.int64) * delta
        ledger.update(speed_factor=factor, output_dt_us=delta,
            meaning='Nearest stored vector time-resampling; original derivative features and posteriors are reused, not recomputed camera-speed features.')
    elif kind == 'main_R_hand_missing':
        ix = np.asarray(task['main_indices'], int)
        dims = task['masked_feature_columns']
        pi = task['masked_posterior_columns']
        known_removed = int(b['mask'][np.ix_(ix, dims)].sum())
        posterior_removed = int(np.isfinite(b['probabilities'][np.ix_(ix, pi)]).sum())
        b['mask'][np.ix_(ix, dims)] = False
        b['probabilities'][np.ix_(ix, pi)] = np.nan
        ledger.update(edited_indices=ix.tolist(), masked_feature_columns=dims,
            masked_feature_names=[a['feature_names'][j] for j in dims], masked_posterior_columns=pi,
            newly_removed_feature_cells=known_removed, newly_removed_posterior_cells=posterior_removed,
            original_x_values_preserved=True, pose_arm_and_L_shape_preserved=True,
            meaning='Artificial unavailable R hand and dependent relations, not an image occlusion/detector experiment.')
    elif kind == 'short_R_shape_axis':
        ix = np.asarray(task['short_indices'], int)
        pi = [0, 1, 3]
        shape = b['probabilities'][np.ix_(ix, pi)]
        valid = np.isfinite(shape)
        shape[valid] = 1 - shape[valid]
        b['probabilities'][np.ix_(ix, pi)] = shape
        axis = [a['feature_names'].index('R_palm_axis_' + d) for d in ('x', 'y')]
        known = b['mask'][np.ix_(ix, axis)].all(1) & np.isfinite(b['x'][np.ix_(ix, axis)]).all(1)
        for i in ix[known]:
            x, y = b['x'][i, axis]
            b['x'][i, axis] = [-y, x]
        changed_shape = valid & (shape != a['probabilities'][np.ix_(ix, pi)])
        changed_axis = (b['x'][np.ix_(ix, axis)] != a['x'][np.ix_(ix, axis)]) & known[:, None]
        ledger.update(edited_indices=ix.tolist(), finite_shape_cells_toggled=int(valid.sum()),
            finite_shape_cells_actually_changed=int(changed_shape.sum()),
            shape_posterior_columns=pi, axis_columns=axis, axis_rotation_degrees=90,
            supported_axis_frames_rotated=int(known.sum()),
            supported_axis_cells_actually_changed=int(changed_axis.sum()), masks_preserved=True,
            finite_shape_conflict_frames_before=int((np.sum(a['probabilities'][np.ix_(ix, pi)] >= .5, axis=1) > 1).sum()),
            finite_shape_conflict_frames_after=int((np.sum(shape >= .5, axis=1) > 1).sum()),
            no_effective_edit=bool(not changed_shape.any() and not changed_axis.any()),
            meaning='Post-observer short synthetic perturbation; may cause conflicting head attributes and therefore unknown rather than a difference warning. No educational error truth.')
    elif kind not in ('same_expression', 'different_target_expression'):
        raise ValueError('Unregistered task')
    for key in original:
        if isinstance(original[key], np.ndarray):
            if not np.array_equal(original[key], a[key], equal_nan=True):
                raise AssertionError('Original arrays mutated')
    ledger.update(query_frame_count=len(mapping), source_indices=mapping.tolist(),
        source_frame_ids=a['frame_id'][mapping].tolist(), source_timestamp_us=a['timestamp_us'][mapping].tolist(),
        output_frame_ids=b['frame_id'].tolist(), output_timestamp_us=b['timestamp_us'].tolist(),
        known69_before=int(a['mask'].sum()), known69_after=int(b['mask'].sum()),
        finite7_before=int(np.isfinite(a['probabilities']).sum()), finite7_after=int(np.isfinite(b['probabilities']).sum()),
        mapping_is_audit_only_not_engine_input=True)
    return b, ledger


def prepare():
    for name in ('EVALUATION_TASKS.json', 'EVALUATION_FROZEN.json', 'THREE_REFERENCE_RANGES.json'):
        if (U / name).exists():
            raise FileExistsError('Preserve prepared evaluation: ' + name)
    frozen, bank, normal, dev = common_inputs()
    subset = subset_ranges(bank, normal)
    tasks = []
    for query in dev:
        a = load_arrays(query, bank['feature_names'])
        main = main_indices(a, query['intervals']).tolist()
        if not main:
            raise ValueError('Public main interval absent')
        configs = [('same_expression', query['word_id'], {}),
                   *[('different_target_expression', w, {}) for w in frozen['word_ids'] if w != query['word_id']],
                   ('speed_075', query['word_id'], {'speed_factor': .75}),
                   ('speed_125', query['word_id'], {'speed_factor': 1.25}),
                   ('main_R_hand_missing', query['word_id'], {'masked_feature_columns': right_missing_columns(bank['feature_names']),
                                                            'masked_posterior_columns': [0, 1, 3, 5, 6]}),
                   ('short_R_shape_axis', query['word_id'], {'short_indices': short_indices(a, query['intervals'])})]
        for kind, target, extra in configs:
            refs = [r for r in bank['rows'] if r['word_id'] == target]
            tasks.append(dict(id=f'E{len(tasks)+1:03}', query=query['clip_id'], query_person=query['person'],
                query_source_word=query['word_id'], target_word=target, kind=kind, **extra,
                main_intervals=query['intervals'], main_indices=main,
                references=[r['clip_id'] for r in refs], subset_references=[r['clip_id'] for r in refs if r['person'] <= 3],
                source_input=query['input'], source_input_sha256=query['input_sha256'],
                source_probabilities=query['probability_file'], source_probabilities_sha256=query['probability_sha256'],
                query_phase_truth_passed_to_engine=False))
    if Counter(t['kind'] for t in tasks) != COUNTS or any(len(t['references']) != 9 or len(t['subset_references']) != 3 for t in tasks):
        raise ValueError('Frozen96 task /9-reference denominators differ')
    save(U / 'THREE_REFERENCE_RANGES.json', subset)
    save(U / 'EVALUATION_TASKS.json', dict(tasks=tasks, cases=96, new_comparisons=864,
        per_kind=COUNTS, all_references=9, subset_references=3, subset_new_comparisons=0,
        new_head_inference=0, new_RGB_decode=0, source_normal_ranges_sha256=sha(U / 'NORMAL_RANGES.json')))
    pins = {str(U / p): sha(U / p) for p in ('FROZEN.json', 'PUBLIC_INPUTS.json', 'TRAINING_REPORT.json',
        'REFERENCE_BANK.json', 'BANK_PREDICTIONS.json', 'NORMAL_RANGES.json', 'THREE_REFERENCE_RANGES.json',
        'EVALUATION_TASKS.json', 'engine.py', 'evaluate_feedback.py')}
    for row in bank['rows'] + dev:
        pins[row['input']] = row['input_sha256']
        pins[row['probability_file']] = row['probability_sha256']
    pins.update({p['path']: p['sha256'] for p in subset['pair_index']})
    verify(pins)
    save(U / 'EVALUATION_FROZEN.json', dict(status='FROZEN_BEFORE_DEV_COMPARISONS', pins=pins,
        expected_cases=96, expected_new_paths=864, expected_pair_stage_opportunities=2808,
        expected_aggregate_stages_per_mode=312, expected_aggregate_features_per_mode=2184,
        original_public_tests=frozen['public_tests'], new_inference_or_RGB_decode=0,
        control_window_selection='Native main-interval midpoint before comparison; no residual/score selection.',
        calibration='All9 normal ranges unchanged; subset01-03 uses saved24 FIT comparisons and same pooled .95 per-family rule.',
        interpretations=['Same-expression warning is only a false-warning proxy, not educational error truth.',
            'Different target expression is a semantic mismatch stress test, not a per-feature error annotation.',
            'Known-speed mapping audits saved-index correspondence stability relative to native cross-person paths, not independent phase accuracy.',
            'Missing and contradictory synthetic observations must remain unknown/support counts; no denominator reduction or zero-distance success.',
            'All9 versus subset3 changes both reference diversity and fixed FIT calibration sample size.',
            'Raw public upload end-to-end is handled separately by root; this array suite does not claim that test completed.']))
    print(json.dumps(dict(status='PREPARED_NOT_RUN', cases=96, comparisons=864, subset_saved_FIT_pairs=24)))


def path_check(comparison, nq, nr):
    p = np.asarray(comparison['path'], int)
    delta = np.diff(p, axis=0)
    if (p.ndim != 2 or p.shape[1] != 2 or not np.array_equal(p[0], [0, 0]) or
        not np.array_equal(p[-1], [nq-1, nr-1]) or
        any(tuple(v) not in ((1, 0), (0, 1), (1, 1)) for v in delta) or
        len(np.unique(p[:, 0])) != nq or len(np.unique(p[:, 1])) != nr):
        raise AssertionError('Incomplete/non-unit native axes')
    return p


def status_counts(stages):
    st = Counter(s['status'] for s in stages)
    ff = Counter(f['status'] for s in stages for f in s['features'])
    if any(k not in STATUS for k in set(st) | set(ff)):
        raise ValueError('Unknown engine status')
    return dict(stage_opportunities=len(stages), feature_opportunities=sum(len(s['features']) for s in stages),
                stages={s: st[s] for s in STATUS}, features={s: ff[s] for s in STATUS})


def mapping_audit(control, native, mapping, original_times):
    """Compare linked source times at every reference frame; no new alignment."""
    cp = np.asarray(control['path'], int)
    bp = np.asarray(native['path'], int)
    mapped = np.asarray(mapping)[cp[:, 0]]
    all_rows = []
    for ri in np.unique(bp[:, 1]):
        original = np.unique(bp[bp[:, 1] == ri, 0])
        altered = np.unique(mapped[cp[:, 1] == ri])
        distances = np.abs(original_times[original, None] - original_times[altered][None, :]) / 1e6
        all_rows.append(dict(reference_index=int(ri), native_query_indices=original.tolist(),
            control_mapped_source_indices=altered.tolist(), exact_same_source_set=bool(np.array_equal(original, altered)),
            native_to_control_mean_seconds=float(distances.min(1).mean()),
            control_to_native_mean_seconds=float(distances.min(0).mean()),
            hausdorff_seconds=float(max(distances.min(1).max(), distances.min(0).max()))))
    return dict(reference_frames=len(all_rows), exact_set_frames=sum(r['exact_same_source_set'] for r in all_rows),
        native_to_control_mean_seconds=float(np.mean([r['native_to_control_mean_seconds'] for r in all_rows])),
        control_to_native_mean_seconds=float(np.mean([r['control_to_native_mean_seconds'] for r in all_rows])),
        peak_hausdorff_seconds=max(r['hausdorff_seconds'] for r in all_rows), frames=all_rows,
        interpretation='Known source mapping relative to saved native cross-person alignment; that native alignment is not phase ground truth.')


def denominator_audit(comparison, native):
    rows = []
    if [s['canonical_stage_id'] for s in comparison['stages']] != [s['canonical_stage_id'] for s in native['stages']]:
        raise AssertionError('Control stage occurrence inventory changed')
    for stage, baseline in zip(comparison['stages'], native['stages']):
        for family in FAMILIES:
            v, b = stage['families'][family], baseline['families'][family]
            reference_invariants = ('expected', 'dimension_reference_expected', 'comparison_required',
                'original_expected', 'original_possible', 'scoped_possible', 'scope_excluded',
                'scope_excluded_possible', 'dimension_original_reference_expected', 'dimension_scope_excluded')
            if any(v[key] != b[key] for key in reference_invariants):
                raise AssertionError('Control shrank original reference evidence denominator')
            if (v['expected'] + v['scope_excluded'] != v['original_expected'] or
                    v['scoped_possible'] + v['scope_excluded_possible'] != v['original_possible'] or
                    (np.asarray(v['dimension_reference_expected']) + v['dimension_scope_excluded']).tolist()
                    != v['dimension_original_reference_expected']):
                raise AssertionError('Original/scoped/excluded reference ledger is inconsistent')
            if not (0 <= v['observed'] <= v['expected'] and 0 <= v['query_observed'] <= v['query_expected']):
                raise AssertionError('Invalid support count')
            rows.append(dict(stage=stage['canonical_stage_id'], family=family, expected_reference=v['expected'],
                original_expected=v['original_expected'], original_possible=v['original_possible'],
                scoped_possible=v['scoped_possible'], scope_excluded=v['scope_excluded'],
                scope_excluded_possible=v['scope_excluded_possible'],
                dimension_reference_expected=v['dimension_reference_expected'],
                dimension_original_reference_expected=v['dimension_original_reference_expected'],
                dimension_scope_excluded=v['dimension_scope_excluded'],
                reference_scope_unchanged_from_native=True,
                reference_observed_before=b['observed'], reference_observed_after=v['observed'],
                query_expected_before=b['query_expected'], query_expected_after=v['query_expected'],
                query_observed_before=b['query_observed'], query_observed_after=v['query_observed'],
                coverage_before=b['coverage'], coverage_after=v['coverage'],
                distance_before=b['distance'], distance_after=v['distance'],
                peak_before=b['peak_distance'], peak_after=v['peak_distance'],
                comparison_required=v['comparison_required']))
    return rows


def run():
    for name in ('EVALUATION_STARTED.json', 'FEEDBACK_EVALUATION.json', 'evaluation_pairs', 'evaluation_cases', 'evaluation_inputs'):
        if (U / name).exists():
            raise FileExistsError('Completed/partial evaluation is immutable: ' + name)
    contract = read(U / 'EVALUATION_FROZEN.json')
    verify(contract['pins'])
    _, bank, normal, dev = common_inputs()
    tasks = read(U / 'EVALUATION_TASKS.json')['tasks']
    subset = read(U / 'THREE_REFERENCE_RANGES.json')
    engine = load_engine()
    refrows = {r['clip_id']: r for r in bank['rows']}
    devrows = {r['clip_id']: r for r in dev}
    refs = {cid: load_arrays(row, bank['feature_names']) for cid, row in refrows.items()}
    queries = {cid: load_arrays(row, bank['feature_names']) for cid, row in devrows.items()}
    save(U / 'EVALUATION_STARTED.json', dict(contract_sha256=sha(U / 'EVALUATION_FROZEN.json'),
        tasks=96, pair_comparisons=864, old_pair_replay=0, observer_forward=0, RGB_decode=0))
    for name in ('evaluation_pairs', 'evaluation_cases', 'evaluation_inputs'):
        (U / name).mkdir()
    start = time.perf_counter()
    cases = []
    native = {}
    pair_count = stage_count = 0
    try:
        for task in tasks:
            source = queries[task['query']]
            a, edit = transform(source, task)
            control = task['kind'] not in ('same_expression', 'different_target_expression')
            modified_input = None
            if control:
                p = U / 'evaluation_inputs' / (task['id'] + '.npz')
                with p.open('xb') as f:
                    np.savez_compressed(f, **{k: a[k] for k in ('x', 'mask', 'rgb', 'frame_id', 'timestamp_us', 'probabilities')})
                modified_input = dict(path=str(p), sha256=sha(p))
            comparisons = []
            pair_index = []
            controls = []
            for rid in task['references']:
                result = engine.compare_one(a, refs[rid], refrows[rid])
                path_check(result, len(a['x']), len(refs[rid]['x']))
                result['query_id'] = task['query']
                result['task_id'] = task['id']
                result['kind'] = task['kind']
                if task['kind'] == 'same_expression':
                    native[(task['query'], rid)] = result
                if control:
                    baseline = native[(task['query'], rid)]
                    audit = dict(reference=rid, same_reference_denominators=denominator_audit(result, baseline))
                    if task['kind'].startswith('speed_'):
                        audit['mapped_time'] = mapping_audit(result, baseline, edit['source_indices'], source['timestamp_us'])
                    controls.append(audit)
                p = U / 'evaluation_pairs' / f"{task['id']}__{rid}.json"
                save(p, result, compact=True)
                pair_index.append(dict(reference=rid, person=refrows[rid]['person'], path=str(p), sha256=sha(p)))
                comparisons.append(result)
                pair_count += 1
                stage_count += len(result['stages'])
            modes = {}
            for mode, selected, limits in [('all9', comparisons, normal['ranges']),
                    ('subset3', [c for c in comparisons if c['id'] in task['subset_references']], subset['ranges'])]:
                stages, selected_ids = engine.aggregate(selected, limits)
                modes[mode] = dict(reference_candidates=len(selected), selected_whole_references=selected_ids,
                                   stages=stages, counts=status_counts(stages))
            value = dict(task=task, source_observation_unchanged=True, modified_input=modified_input,
                edit_ledger=edit, comparisons=pair_index, modes=modes, control_audits=controls,
                unknown_not_negative=True, educational_correctness_ground_truth=None)
            p = U / 'evaluation_cases' / (task['id'] + '.json')
            save(p, value, compact=True)
            cases.append(dict(id=task['id'], kind=task['kind'], query=task['query'], target_word=task['target_word'],
                path=str(p), sha256=sha(p), counts={k: v['counts'] for k, v in modes.items()},
                no_effective_edit=edit.get('no_effective_edit', False)))
            print(json.dumps(dict(stage='public_feedback', completed_cases=len(cases), total_cases=96,
                                  comparisons=pair_count, kind=task['kind'])), flush=True)
        if len(cases) != 96 or pair_count != 864 or stage_count != 2808:
            raise AssertionError('Full public opportunity count differs')
        summary = {}
        for kind, expected in COUNTS.items():
            selected = [c for c in cases if c['kind'] == kind]
            if len(selected) != expected:
                raise AssertionError('Missing frozen cases: ' + kind)
            summary[kind] = dict(cases=expected, errors=0, no_effective_edit=sum(c['no_effective_edit'] for c in selected), modes={})
            for mode in ('all9', 'subset3'):
                counts = [c['counts'][mode] for c in selected]
                summary[kind]['modes'][mode] = dict(stage_opportunities=sum(c['stage_opportunities'] for c in counts),
                    feature_opportunities=sum(c['feature_opportunities'] for c in counts),
                    stages={s: sum(c['stages'][s] for c in counts) for s in STATUS},
                    features={s: sum(c['features'][s] for c in counts) for s in STATUS})
        for mode in ('all9', 'subset3'):
            if (sum(v['modes'][mode]['stage_opportunities'] for v in summary.values()) != 312 or
                sum(v['modes'][mode]['feature_opportunities'] for v in summary.values()) != 2184):
                raise AssertionError('Aggregate opportunity denominators changed')
        verify(contract['pins'])
        save(U / 'FEEDBACK_EVALUATION.json', dict(status='COMPLETE_FIXED_PUBLIC_ARRAY_EVALUATION',
            contract_sha256=sha(U / 'EVALUATION_FROZEN.json'), source_pins=contract['pins'],
            cases=96, errors=0, new_pair_paths=pair_count, pair_stage_opportunities=stage_count,
            subset_new_paths=0, observer_forward=0, RGB_decode=0, training=0,
            original_input_values_and_hashes_preserved=True, per_kind=summary, case_index=cases,
            seconds=time.perf_counter()-start, adopted_new_engine_or_threshold=False,
            interpretations=contract['interpretations'],
            raw_public_upload_test='PENDING_SEPARATE_ROOT_END_TO_END_RESULT',
            warning='Same-expression DIFFERENCE_CANDIDATE is only a false-warning proxy; wrong target and synthetic controls do not establish per-feature educational accuracy.'))
    except BaseException:
        failure = U / 'EVALUATION_FIRST_FAILURE.json'
        if not failure.exists():
            save(failure, dict(status='FIRST_FAILURE_PRESERVED', traceback=traceback.format_exc(),
                expected_cases=96, completed_cases=len(cases), attempted_pair_results=pair_count,
                remaining_cases=96-len(cases), completed_case_index=cases,
                instruction='Do not rerun completed paths or overwrite partial results. Review the exact failure first.'))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'run'])
    args = parser.parse_args()
    {'prepare': prepare, 'run': run}[args.command]()
