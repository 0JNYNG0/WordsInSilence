"""Independent saved-data audit. No producer imports, model forward, video decode or DTW.

Writes only this unit's review/VERIFY_DEMO.json, exclusively. The numeric forward
pass on 1,844 new centers is deliberately NOT repeated. This verifies its saved
lineage, finite/schema/time scope, and all downstream arithmetic instead.
"""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import sys
import traceback

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
UNIT = HERE.parent
OPS = UNIT.parent
SEM = OPS / 'word100_semantic_feature_cycle_20261101'
OLD = OPS / 'word100_stage_feedback_cycle_20261031'
CHECKS = Counter()
ARRAY_CELLS = Counter()
FAILURES = []
PINS = {}
FACTS = {}
TARGETS = ['R_index_only', 'R_open_four', 'L_open_four', 'R_closed_four',
           'L_closed_four', 'R_higher', 'L_higher']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pin(path):
    path = str(Path(path).resolve())
    digest = sha(path)
    if path in PINS:
        check('preservation', PINS[path] == digest, 'changed during verification: ' + path)
    else:
        PINS[path] = digest
    return digest


def read(path):
    pin(path)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def arrays(path, keys=None):
    pin(path)
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in (keys or archive.files)}


def check(group, condition, message):
    CHECKS[group] += 1
    if not bool(condition):
        FAILURES.append({'group': group, 'message': message})


def eq(group, actual, expected, message):
    check(group, actual == expected, message)


def same_array(group, actual, expected, message):
    aa, bb = np.asarray(actual), np.asarray(expected)
    ARRAY_CELLS[group] += int(bb.size)
    check(group, aa.shape == bb.shape and np.array_equal(aa, bb, equal_nan=True), message)


def gates(mask):
    """Independent fixed69 positional implementation, not common.support()."""
    mask = np.asarray(mask, dtype=bool)
    left = mask[:, 0] & mask[:, 1] & (np.count_nonzero(mask[:, 11:25], axis=1) >= 10)
    right = mask[:, 28] & mask[:, 29] & (np.count_nonzero(mask[:, 39:53], axis=1) >= 10)
    both = mask[:, 8] & mask[:, 9] & mask[:, 36] & mask[:, 37]
    return np.column_stack([right, right, left, right, left, both, both])


def statistics(cells):
    counts = Counter((c['y'], c['pred']) for c in cells)
    tn, fp, fn, tp = (counts[0, 0], counts[0, 1], counts[1, 0], counts[1, 1])
    positive, negative = tp + fn, tn + fp
    return dict(known_cells=len(cells), positive=positive, negative=negative,
                tp=tp, tn=tn, fp=fp, fn=fn, correct=tp + tn,
                accuracy=(tp + tn) / len(cells) if cells else None,
                balanced_accuracy=(tp / positive + tn / negative) / 2
                if positive and negative else None,
                clips=len({c['clip_id'] for c in cells}),
                persons=sorted({c['person'] for c in cells}),
                words=len({c['word_id'] for c in cells}))


def check_stats(actual, cells, context):
    for key, value in statistics(cells).items():
        eq('audit_summary', actual[key], value, context + ':' + key)


def probability_json(a):
    return [[None if not np.isfinite(v) else float(v) for v in row] for row in a]


def audit(frozen, dataset, audit_result):
    names = dataset['feature_names']
    expected_columns = {0: 'L_hand_wrist_x', 1: 'L_hand_wrist_y',
                        8: 'L_palm_center_x', 9: 'L_palm_center_y',
                        28: 'R_hand_wrist_x', 29: 'R_hand_wrist_y',
                        36: 'R_palm_center_x', 37: 'R_palm_center_y'}
    for col, name in expected_columns.items():
        eq('schema', names[col], name, 'fixed69 position ' + str(col))
    for start, side in [(11, 'L'), (39, 'R')]:
        check('schema', all(n.startswith(side + '_') and n.endswith('_angle')
                           and 'elbow' not in n for n in names[start:start + 14]), '14 angle columns ' + side)
    eq('schema', dataset['selected_targets'], TARGETS, 'selected targets')
    eq('schema', dataset['targets'], TARGETS, 'source targets')
    eq('schema', dataset['selected_indices'], list(range(7)), 'source target ordering')
    eq('schema', frozen['audit']['threshold'], .5, 'fixed class threshold')
    eq('schema', frozen['audit']['seeds'], [42, 43, 44], 'all seeds')
    eq('schema', frozen['demo']['seed'], 42, 'preset demo seed')
    rows = dataset['rows']
    eq('audit_source', len(rows), 288, 'exact center denominator')
    eq('audit_source', len({r['clip_id'] for r in rows}), 96, 'clip denominator')
    eq('audit_source', len({(r['clip_id'], r['center_frame']) for r in rows}), 288, 'unique centers')
    by_role = defaultdict(set)
    cache, center_index = {}, {}
    for row in rows:
        by_role[row['role']].add(row['person'])
        path = row['input']
        if path not in cache:
            a = arrays(path, ['mask', 'frame_id', 'timestamp_us'])
            cache[path] = a
            eq('audit_source', pin(path), row['input_sha256'], 'input digest ' + path)
            eq('schema', a['mask'].shape[1], 69, '69 features')
            check('schema', a['mask'].dtype == np.bool_, 'mask boolean')
        a = cache[path]
        positions = np.flatnonzero(a['frame_id'] == row['center_frame'])
        eq('audit_source', len(positions), 1, 'unique frame ' + row['sample_id'])
        index = int(positions[0])
        center_index[row['sample_id']] = index
        check('audit_source', 6 <= index < len(a['frame_id']) - 6, '13 real context frames')
        same_array('audit_source', a['frame_id'][index - 6:index + 7], row['frame_ids'], 'context frame IDs')
        same_array('audit_source', a['timestamp_us'][index - 6:index + 7], row['timestamps_us'], 'context PTS')
        eq('audit_source', int(a['timestamp_us'][index]), row['center_timestamp_us'], 'center PTS')
    expected_roles = {'fit': [1, 2, 3], 'development': [4], 'new_feature_expression_evaluation': [7, 8]}
    eq('partition', {k: sorted(v) for k, v in by_role.items()}, expected_roles, 'recorded-person partition')
    roles = sorted(by_role)
    for i, role in enumerate(roles):
        for other in roles[i + 1:]:
            check('partition', by_role[role].isdisjoint(by_role[other]), role + '/' + other)
    predicted = {}
    records = []
    for seed in [42, 43, 44]:
        path = SEM / f'runs/protected_shape_split_features_{seed}/probabilities.npy'
        pin(path)
        predicted[seed] = p = np.load(path, allow_pickle=False)
        eq('audit_source', p.shape, (288, 7), 'probability schema ' + str(seed))
        check('audit_source', np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'probability range')
        for i, row in enumerate(rows):
            supported = gates(cache[row['input']]['mask'][[center_index[row['sample_id']]]])[0]
            for col, target in enumerate(TARGETS):
                if not row['target_mask'][col]:
                    continue
                check('audit_source', row['y'][col] in (0, 1), 'binary direct center label')
                records.append(dict(seed=seed, sample_id=row['sample_id'], clip_id=row['clip_id'],
                                    word_id=row['word_id'], person=row['person'], role=row['role'],
                                    target=target, frame_id=row['center_frame'],
                                    stratum='all_required_geometry_present' if supported[col] else 'geometry_gate_failed',
                                    y=int(row['y'][col]), pred=int(p[i, col] >= .5),
                                    probability=float(p[i, col]), dataset_index=i))
    eq('audit_record', len(audit_result['records']), len(records), 'all known cells retained')
    for actual, expected in zip(audit_result['records'], records):
        eq('audit_record', actual, expected, expected['sample_id'] + '/' + str(expected['seed']) + '/' + expected['target'])
    unknown = sum(7 - sum(row['target_mask']) for row in rows)
    eq('audit_summary', audit_result['excluded_unknown_per_seed'], unknown, 'unknown labels excluded explicitly')
    eq('audit_summary', unknown + len(records) // 3, 288 * 7, 'complete label opportunity denominator')
    eq('audit_summary', audit_result['dataset_rows'], 288, 'saved rows')
    eq('audit_summary', audit_result['dataset_clips'], 96, 'saved clips')
    eq('audit_summary', audit_result['source_dataset_sha256'], pin(SEM / 'DATASET.json'), 'dataset pin')
    for key in ['new_inference', 'new_training']:
        eq('audit_summary', audit_result[key], 0, 'saved probability reanalysis only ' + key)
    expected_group_keys = {(s, r, k) for s in [42, 43, 44] for r in roles
                           for k in ['all_required_geometry_present', 'geometry_gate_failed']}
    eq('audit_summary', {(g['seed'], g['role'], g['stratum']) for g in audit_result['groups']}, expected_group_keys, 'all groups')
    eq('audit_summary', len(audit_result['groups']), len(expected_group_keys), 'unique groups')
    for group in audit_result['groups']:
        cells = [c for c in records if all(c[k] == group[k] for k in ('seed', 'role', 'stratum'))]
        eq('audit_summary', [f['target'] for f in group['features']], TARGETS, 'all features including zero support')
        for feature in group['features']:
            check_stats(feature, [c for c in cells if c['target'] == feature['target']], 'feature group')
    eq('audit_summary', len(audit_result['pooled']), len(expected_group_keys), 'pooled groups')
    for group in audit_result['pooled']:
        check_stats(group, [c for c in records if all(c[k] == group[k] for k in ('seed', 'role', 'stratum'))], 'pooled')
        check('claim_boundary', 'not word or educational accuracy' in group['warning'], 'pooled caveat')
    FACTS['audit'] = {'clips': 96, 'centers': 288, 'label_opportunities_per_seed': 2016,
                      'known_label_cells_per_seed': len(records) // 3, 'unknown_label_cells_per_seed': unknown,
                      'recorded_person_roles': expected_roles,
                      'development_pooled': [g for g in audit_result['pooled'] if g['role'] == 'development']}
    return predicted, records


def dense_and_clips(case, dataset, dense, demo, predicted):
    old = read(OLD / 'LEARNING_INPUTS.json')
    old_index = {r['clip_id']: i for i, r in enumerate(old['rows'])}
    old_path = OLD / 'runs/pose_plus_signrep_42/probabilities.npy'
    pin(old_path)
    native = np.load(old_path, allow_pickle=False)
    centers = {(r['clip_id'], r['center_frame']): i for i, r in enumerate(dataset['rows'])}
    eq('dense', dense['seed'], 42, 'seed fixed')
    eq('dense', dense['targets'], TARGETS, 'column schema')
    eq('dense', {r['alias'] for r in dense['rows']}, set(case['rows']), '16 source aliases')
    eq('dense', len(dense['rows']), 16, '16 files')
    eq('dense', set(demo['clips']), set(case['rows']), 'demo clip coverage')
    totals = Counter()
    geometry_failed_but_estimated = 0
    clip_arrays = {}
    for manifest in dense['rows']:
        alias = manifest['alias']; source = case['rows'][alias]; clip = demo['clips'][alias]
        a = arrays(source['input'], ['x', 'mask', 'rgb', 'frame_id', 'timestamp_us'])
        n = len(a['frame_id'])
        eq('dense', pin(source['input']), source['input_sha256'], 'input ' + alias)
        eq('dense', source['feature_names'], dataset['feature_names'], 'feature name identity ' + alias)
        eq('dense', a['x'].shape, (n, 69), 'x schema ' + alias)
        eq('dense', a['rgb'].shape, (n, 768), 'cached RGB schema ' + alias)
        check('dense', np.isfinite(a['x']).all() and np.isfinite(a['rgb']).all(), 'finite input ' + alias)
        same_array('dense_time', np.diff(a['frame_id']), np.ones(n - 1), 'native consecutive frames ' + alias)
        check('dense_time', (np.diff(a['timestamp_us']) > 0).all(), 'increasing PTS ' + alias)
        saved = arrays(manifest['path'])
        eq('dense', pin(manifest['path']), manifest['sha256'], 'prediction hash ' + alias)
        same_array('dense_time', saved['frame_id'], a['frame_id'], 'exact source IDs ' + alias)
        same_array('dense_time', saved['timestamp_us'], a['timestamp_us'], 'exact source PTS ' + alias)
        p = saved['probabilities']; sources = saved['new_four_source']
        eq('dense', p.shape, (n, 7), 'seven output columns ' + alias)
        oi = old_index[source['clip_id']]
        eq('dense', old['rows'][oi]['input_sha256'], source['input_sha256'], 'old input identity ' + alias)
        old_columns = [old['targets'].index(t) for t in TARGETS[:3]]
        same_array('protected_native_three', p[:, :3], native[oi, :n][:, old_columns], 'all native old3 bitwise ' + alias)
        expected_source = np.zeros(n, np.int8)
        reused, new = [], []
        for i in range(6, n - 6):
            row_index = centers.get((source['clip_id'], int(a['frame_id'][i])))
            if row_index is None:
                expected_source[i] = 2; new.append(int(a['frame_id'][i]))
            else:
                expected_source[i] = 1; reused.append(int(a['frame_id'][i]))
                same_array('reused_centers', p[i], predicted[42][row_index], 'all seven cached exact ' + alias + '/' + str(i))
                row = dataset['rows'][row_index]
                same_array('reused_centers', row['frame_ids'], a['frame_id'][i - 6:i + 7], 'old and new context IDs identical')
                same_array('reused_centers', row['timestamps_us'], a['timestamp_us'][i - 6:i + 7], 'old and new context PTS identical')
            eq('dense_time', len(a['frame_id'][i - 6:i + 7]), 13, '13 real frames per center')
        same_array('dense', sources, expected_source, 'source classification ' + alias)
        same_array('dense', np.isfinite(p[:, 3:]), np.repeat((expected_source > 0)[:, None], 4, axis=1), 'new4 boundary unavailable ' + alias)
        check('dense', ((p[np.isfinite(p)] >= 0) & (p[np.isfinite(p)] <= 1)).all(), 'finite probabilities in [0,1]')
        eq('dense', manifest['new_frame_ids'], new, 'new native center list')
        eq('dense', manifest['reused_frame_ids'], reused, 'reused native center list')
        for key, value in [('frames', n), ('new_centers', len(new)), ('reused_centers', len(reused)), ('boundary_centers_without_new_four', 12)]:
            eq('dense', manifest[key], value, alias + ':' + key)
        check('latency', 0 <= manifest['inference_seconds'] <= manifest['cached_input_total_seconds'], 'reported cached timing ' + alias)
        totals.update(frames=n, new=len(new), reused=len(reused), boundary=12)
        gate = gates(a['mask']); clip_arrays[alias] = (p, gate, a)
        geometry_failed_but_estimated += int((~gate & np.isfinite(p)).sum())
        for key, value in [('alias', alias), ('clip_id', source['clip_id']), ('label', source['label']),
                           ('recorded_person_id', source['person']), ('role', source['role']), ('frames', n),
                           ('box', source['box']), ('annotation_intervals', source['intervals']), ('media', '/media/' + alias)]:
            eq('demo_clip', clip[key], value, alias + ':' + key)
        eq('demo_clip', clip['frame_ids'], a['frame_id'].tolist(), 'display IDs ' + alias)
        eq('demo_clip', clip['times'], (a['timestamp_us'] / 1e6).tolist(), 'display PTS ' + alias)
        eq('demo_clip', clip['probabilities'], probability_json(p), 'display probabilities/null ' + alias)
        eq('demo_clip', clip['geometry_support'], gate.tolist(), 'separate geometry support ' + alias)
        eq('demo_clip', clip['new_four_source'], sources.tolist(), 'source class display ' + alias)
        media = read(OPS / f'word100_core_stage_evidence_20261031/media/{alias}.json')
        check('media_time', len(media['source_pts_seconds']) == n and np.max(np.abs(np.asarray(media['source_pts_seconds']) * 1e6 - a['timestamp_us'])) < 1.1, 'source video PTS ' + alias)
        observation = arrays(source['observation'], ['xy_px', 'observed'])
        xy, observed = observation['xy_px'], observation['observed']
        eq('source_points', xy.shape, (n, 75, 2), 'native point dimensions ' + alias)
        for i in range(n):
            for j in range(75):
                expected = [round(float(v), 2) for v in xy[i, j]] if observed[i, j] and np.isfinite(xy[i, j]).all() else None
                eq('source_points', clip['points'][i][j], expected, alias + '/' + str(i) + '/' + str(j))
    eq('dense', totals, Counter(frames=2084, new=1844, reused=48, boundary=192), 'complete native denominator')
    eq('dense', dense['new_center_predictions'], totals['new'], 'saved new total')
    eq('dense', dense['reused_center_predictions'], totals['reused'], 'saved reuse total')
    for key in ['new_training', 'completed_center_reruns', 'new_rgb_extractions', 'new_detector_calls']:
        eq('dense', dense[key], 0, key)
    eq('dense', dense['old_three_unchanged'], True, 'old3 claim')
    FACTS['dense'] = dict(totals, old_three_values_verified=2084 * 3, new_four_boundary_null_cells=192 * 4,
                          per_clip_estimates_with_geometry_gate_false=geometry_failed_but_estimated,
                          new_forward_reexecuted=False, new_forward_numeric_value_independently_reproduced=False)
    return clip_arrays


def pairs_and_consumer(case, demo, clip_arrays):
    legacy = read(OPS / 'word100_description_evidence_feedback_20261101/REPORT_DATA.json')
    expected_pairs = sorted({(row['query'], row['reference']) for row in case['natural']})
    eq('routing', [(p['query'], p['reference']) for p in demo['pairs']], expected_pairs, 'all 12 individual references')
    expected_queries = ['P036', 'P048', 'P066', 'P072']
    eq('routing', demo['queries'], expected_queries, 'fixed four DEV queries')
    statuses = Counter(); source_statuses = Counter(); frame_count = 0; stage_count = 0; linked = 0
    raw_difference_in_conflict = 0; missing_gate_estimates = Counter()
    for q in expected_queries:
        refs = [r for qq, r in expected_pairs if qq == q]
        eq('routing', sorted(case['rows'][r]['person'] for r in refs), [1, 2, 3], q + ' all FIT recorded IDs')
        eq('routing', case['rows'][q]['person'], 4, q + ' DEV recorded ID')
    for pair in demo['pairs']:
        q, r = pair['query'], pair['reference']
        eq('routing', case['rows'][q]['word_id'], case['rows'][r]['word_id'], 'same supplied target bank')
        eq('routing', case['rows'][q]['role'], 'development', 'query DEV')
        eq('routing', case['rows'][r]['role'], 'fit', 'reference FIT')
        old_case = next(c for c in legacy['cases'] if c['query'] == q and c['reference'] == r)
        eq('path', pair['path'], old_case['path'], 'saved path unchanged ' + pair['id'])
        eq('path', pair['source_pair'], old_case['source_pair'], 'original path pointer')
        eq('path', pin(pair['source_pair']), pair['source_pair_sha256'], 'original path digest')
        source_pair = read(pair['source_pair'])
        eq('path', source_pair['path'], pair['path'], 'original pair path direct')
        qp, qgate, qa = clip_arrays[q]; rp, rgate, ra = clip_arrays[r]
        nq, nr = len(qp), len(rp)
        path = np.asarray(pair['path'])
        same_array('path', path[0], [0, 0], 'fullscope start')
        same_array('path', path[-1], [nq - 1, nr - 1], 'fullscope end')
        check('path', all(tuple(step) in {(1, 0), (0, 1), (1, 1)} for step in np.diff(path, axis=0)), 'monotone native moves')
        eq('path', set(path[:, 0].tolist()), set(range(nq)), 'all query source frames retained')
        eq('path', set(path[:, 1].tolist()), set(range(nr)), 'all reference source frames retained')
        mapped = [[] for _ in range(nq)]
        for qi, ri in pair['path']:
            mapped[qi].append(ri)
        selected = [sorted(values)[(len(values) - 1) // 2] for values in mapped]
        eq('path', pair['query_to_reference'], selected, 'representative is actual lower median member')
        eq('path', len(pair['frames']), nq, 'all display frames')
        eq('claim_boundary', pair['saved_path_preserved'], True, 'saved path flag')
        eq('claim_boundary', pair['correct_stage_accuracy'], None, 'no stage accuracy claim')
        keys = ['id', 'reference_core', 'query_frames', 'path', 'reference_rgb_description', 'source_cards', 'alignment']
        expected_stages = [{key: s[key] for key in keys} for s in old_case['stages']]
        eq('stage_context', pair['stages'], expected_stages, 'source context/cards/alignment copied unchanged')
        stage_count += len(expected_stages)
        linked += sum(bool(s['source_cards']) for s in expected_stages)
        for s in expected_stages:
            for card in s['source_cards']:
                if 'support_status' in card:
                    source_statuses[card['support_status']] += 1
        for qi, frame in enumerate(pair['frames']):
            ri = selected[qi]; frame_count += 1
            eq('path', frame['query_frame'], qi, 'query native index')
            eq('path', frame['reference_frame'], ri, 'reference actual member')
            eq('path', frame['reference_candidates'], mapped[qi], 'all many-to-one candidates preserved')
            check('path', [qi, ri] in pair['path'], 'representative membership')
            eq('stage_context', frame['stage_ids'], [s['id'] for s in expected_stages if s['reference_core'][0] <= ri <= s['reference_core'][1]], 'reference-only stage context')
            eq('claim_boundary', frame['educational_verdict'], None, 'no educational verdict')
            eq('claim_boundary', frame['alignment_verified'], False, 'no semantic alignment verification')
            classes = []
            for row in [qp[qi], rp[ri]]:
                classes.append([int(v >= .5) if np.isfinite(v) else None for v in row])
            conflicts = set()
            for group in [(0, 1, 3), (2, 4), (5, 6)]:
                if any(sum(c[j] == 1 for j in group) > 1 for c in classes):
                    conflicts.update(group)
            eq('consumer', len(frame['estimates']), 7, 'all seven opportunities')
            for j, estimate in enumerate(frame['estimates']):
                a, b = classes[0][j], classes[1][j]
                if a is None or b is None:
                    status = 'MODEL_OUTPUT_UNAVAILABLE'
                elif j in conflicts:
                    status = 'CONFLICTING_MODEL_ESTIMATES'
                elif a != b:
                    status = 'ESTIMATED_FEATURE_DIFFERENCE'
                else:
                    status = 'SAME_POSITIVE_ATTRIBUTE' if a else 'SAME_NEGATIVE_ATTRIBUTE_ONLY'
                expected = dict(target=TARGETS[j], query=a, reference=b,
                                query_probability=float(qp[qi, j]) if a is not None else None,
                                reference_probability=float(rp[ri, j]) if b is not None else None,
                                status=status, query_geometry_supported=bool(qgate[qi, j]),
                                reference_geometry_supported=bool(rgate[ri, j]),
                                raw_estimated_difference=None if a is None or b is None else a != b,
                                same_handshape=None, important_difference=None, educational_correction=None,
                                request_new_recording=None)
                eq('consumer', estimate, expected, pair['id'] + '/' + str(qi) + '/' + TARGETS[j])
                statuses[status] += 1
                if status == 'CONFLICTING_MODEL_ESTIMATES' and a != b:
                    raw_difference_in_conflict += 1
                if not qgate[qi, j] or not rgate[ri, j]:
                    missing_gate_estimates[status] += 1
    FACTS['demo'] = dict(pairs=len(expected_pairs), queries=len(expected_queries), stages=stage_count,
                         linked_stages=linked, reference_repeated_query_frames=frame_count,
                         repeated_reference_feature_cells=frame_count * 7,
                         statuses=dict(statuses), raw_differences_retained_under_conflict=raw_difference_in_conflict,
                         statuses_when_either_geometry_gate_failed=dict(missing_gate_estimates),
                         accuracy_denominator=False)
    return statuses, stage_count, linked, frame_count


def main():
    frozen = read(UNIT / 'FROZEN.json')
    for path, expected in frozen['pins'].items():
        eq('frozen_pins', pin(path), expected, path)
    ledger = read(UNIT / 'EXECUTION_LEDGER.json')
    eq('execution_ledger', ledger['source_contract_sha256'], pin(UNIT / 'FROZEN.json'), 'frozen contract digest')
    check('execution_ledger', 'after execution' in ledger['note'], 'postexecution code provenance disclosed')
    for path, expected in ledger['code_pins'].items():
        eq('execution_ledger', pin(path), expected, 'executed source pin ' + path)
    dataset = read(SEM / 'DATASET.json')
    case = read(OPS / 'word100_action_feedback_casebook_20261101/INPUTS.json')
    audit_result = read(UNIT / 'AUDIT.json')
    dense = read(UNIT / 'DENSE_MANIFEST.json')
    demo = read(UNIT / 'DEMO_DATA.json')
    build = read(UNIT / 'DEMO_BUILD.json')
    training = read(UNIT / 'TRAINING_DECISION.json')
    predicted, records = audit(frozen, dataset, audit_result)
    fit_new = [c for c in records if c['role'] == 'fit' and c['target'] in TARGETS[3:]]
    eq('training', training['fit_new_feature_cells'], len(fit_new), 'all seed repeated FIT cell count')
    eq('training', training['fit_new_feature_errors'], sum(c['y'] != c['pred'] for c in fit_new), 'FIT errors')
    eq('training', training['status'], 'REUSE_ALREADY_SUPERVISED_CHECKPOINT_FOR_BOUNDED_DEMO', 'reuse decision')
    for key in ['new_verified_fit_targets', 'new_training']:
        eq('training', training[key], 0, key)
    eq('training', training['new_input_or_target_defect_found'], False, 'no defect claimed')
    clip_arrays = dense_and_clips(case, dataset, dense, demo, predicted)
    statuses, stages, linked, frames = pairs_and_consumer(case, demo, clip_arrays)
    eq('demo_summary', demo['evaluation'], [g for g in audit_result['pooled'] if g['role'] == 'development'], 'same DEV results')
    eq('demo_summary', demo['training'], training, 'training decision reused')
    eq('demo_summary', demo['targets'], TARGETS, 'target ordering')
    for key, expected in [('clips', 16), ('expressions', 4), ('pairs', 12), ('stage_cards', stages),
                          ('linked_stage_cards', linked), ('comparisons', frames * 7), ('states', dict(statuses)),
                          ('repeated_frames_not_independent_trials', True), ('raw_video_latency_measured', False)]:
        eq('demo_summary', build[key], expected, key)
    eq('demo_summary', stages, 39, '39 preserved parent stages')
    eq('demo_summary', frames * 7, 13104, 'display cells not accuracy denominator')
    eq('demo_summary', build['sha256'], pin(UNIT / 'DEMO_DATA.json'), 'demo digest')
    eq('claim_boundary', demo['model_probabilities_not_calibrated_confidence'], True, 'probability not calibration')
    eq('latency', demo['timing']['end_to_end_raw_video_latency'], None, 'no raw end-to-end timing')
    eq('latency', demo['timing']['head_only_new_centers_seconds'], sum(r['inference_seconds'] for r in dense['rows']), 'reported head timings')
    for key, expected in [('new_classifier_centers', 1844), ('reused_classifier_centers', 48),
                          ('completed_center_reruns', 0), ('new_training', 0), ('new_rgb_extractions', 0),
                          ('new_detection', 0), ('new_dtw_paths', 0)]:
        eq('execution_ledger', ledger[key], expected, key)
    eq('demo_summary', demo['new_inference'], 1844, 'new head centers')
    eq('demo_summary', demo['new_training'], 0, 'no training')
    FACTS['training'] = {'fit_new_feature_cells_repeated_across_seeds': len(fit_new),
                         'fit_new_feature_errors': sum(c['y'] != c['pred'] for c in fit_new),
                         'decision': training['status'], 'new_training': 0}


if __name__ == '__main__':
    output = HERE / 'VERIFY_DEMO.json'
    if output.exists():
        raise SystemExit('Prior verification exists; preserve it. Do not rerun/overwrite automatically.')
    caught = None
    try:
        main()
    except BaseException:
        caught = traceback.format_exc()
        FAILURES.append({'group': 'execution_exception', 'message': caught})
    for path, expected in list(PINS.items()):
        try:
            eq('preservation_after', sha(path), expected, path)
        except BaseException as exc:
            check('preservation_after', False, path + ': ' + repr(exc))
    result = dict(
        status='PASS' if not FAILURES else 'FAIL',
        at=datetime.now(timezone.utc).isoformat(),
        verifier='Codex separate-context saved-data arithmetic review; not independent human/KSL labels',
        verifier_source={'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)},
        assertions=sum(CHECKS.values()), assertions_by_group=dict(CHECKS),
        array_cells_compared_by_group=dict(ARRAY_CELLS), failures=FAILURES,
        facts=FACTS, source_pins=PINS,
        code_provenance='Producer code hashes were recorded after execution. Input/condition frozen pins were recorded before execution; this verification does not retroactively pre-pin executed code.',
        method='Independent numpy/stdlib reconstruction from saved arrays, direct center labels and original saved DTW paths. No producer import or function call.',
        static_review=dict(files=['common.py', 'audit.py', 'dense.py', 'build_demo.py', 'consumer.py'],
                           runtime_target_labels_used=False, word_specific_threshold=False,
                           new_four_normalization_matches_stored_training_contract=True,
                           geometry_support_is_separate_from_model_availability=True,
                           saved_guides_are_reference_context_not_query_truth=True),
        limitations=[
            'The 1,844 newly saved four-feature probabilities were not reproduced by a second forward pass. Finite values, source/context identity, source codes and downstream arithmetic are verified.',
            'Declared zero inference reruns/training/new alignment is consistent with reviewed source and ledger, not a hardware execution trace.',
            '96 clips/288 labeled centers are historically reused; recorded-person ID separation is not evidence of unique biological signers or a fresh test.',
            'Known center labels do not label the whole 13-frame context, all native frames, whole stages, or learner errors.',
            'Geometry gate failure is not proof of invisible hands; model availability and saved geometric support are independent.',
            '13,104 cells repeat query frames across three references and are not an accuracy denominator.',
            'A representative existing path member is not proven correct physical recurrence; all path links and unresolved alignment metadata remain visible.',
            'Same-negative and conflicting estimates do not establish same handshape or educational correctness. No ground-truth feedback improvement or calibrated missing-input safety is established.',
            'Video frames were not decoded, RGB was not reviewed again, UI/browser behavior and server access controls are outside this saved-data review.',
            'Cached-input head timing excludes feature extraction, cold start, original alignment and raw-video end-to-end latency.'
        ],
        actions=dict(new_inference=0, training=0, producer_calls=0, alignment_runs=0,
                     video_decodes=0, old_artifact_writes=0, user_data_access=0, external_transmission=0))
    with output.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'status': result['status'], 'assertions': result['assertions'],
                      'failures': len(FAILURES), 'output': str(output)}, ensure_ascii=False))
    if FAILURES:
        print(json.dumps(FAILURES[:8], ensure_ascii=False, indent=2))
        raise SystemExit(1)
