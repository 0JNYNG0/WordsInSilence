"""Read-only independent arithmetic audit; never imports project/model code.

python -B review/verify_results.py --self-test
python -B review/verify_results.py --verify

Only the final audit JSON is written, under this review directory. The verifier
does not load a checkpoint, run a model, reconstruct observations, or rerun DTW.
"""
from pathlib import Path
import argparse
import collections
import datetime
import hashlib
import json
import math
import statistics
import sys

import numpy as np

sys.dont_write_bytecode = True
REVIEW = Path(__file__).resolve().parent
UNIT = REVIEW.parent
GROUPS = ['fit', 'development', 'held_person', 'held_expression_person']
REQUIRED = ['PROTOCOL.json', 'LEARNING_INPUTS.json', 'TRAIN_READY.json',
            'NORMALIZATION.json', 'TRAINING.json', 'SCORES.json',
            'CLIP_SCORES.json', 'DEVELOPMENT_COMPARISON.json',
            'FEEDBACK_FREEZE.json', 'PAIRED_FEEDBACK.json', 'CONTROLS.json',
            'RUNTIME_PROTOCOL.json', 'RUNTIME_RESULT.json',
            'TEMPORAL_EVAL_FREEZE.json', 'TEMPORAL_SCORES.json',
            'review/HELD_TEMPORAL_RGB.json']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def mean(values):
    values = list(values)
    return math.fsum(values) / len(values) if values else None


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def independent_metric(probability, target, mask):
    """Count each known cell explicitly, using no project metric implementation."""
    assert probability.shape == target.shape == mask.shape
    assert probability.ndim == 2
    assert mask.dtype == np.bool_
    assert np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()
    assert np.isin(target[mask], [0, 1]).all()
    details = []
    for j in range(target.shape[1]):
        counts = dict(tp=0, tn=0, fp=0, fn=0)
        for i in np.flatnonzero(mask[:, j]):
            actual = int(target[i, j])
            predicted = int(float(probability[i, j]) >= 0.5)
            cell = ('tp' if predicted else 'fn') if actual else ('fp' if predicted else 'tn')
            counts[cell] += 1
        positives = counts['tp'] + counts['fn']
        negatives = counts['tn'] + counts['fp']
        n = positives + negatives
        recall = []
        if positives:
            recall.append(counts['tp'] / positives)
        if negatives:
            recall.append(counts['tn'] / negatives)
        details.append(dict(known=n, **counts, balanced_accuracy=mean(recall),
                            accuracy=(counts['tp'] + counts['tn']) / n if n else None))
    return dict(known=sum(x['known'] for x in details), targets=details,
                macro_balanced_accuracy=mean(x['balanced_accuracy'] for x in details
                                            if x['balanced_accuracy'] is not None))


class Audit:
    def __init__(self):
        self.checks = 0
        self.failures = []
        self.sections = collections.Counter()

    def check(self, condition, label, detail=None):
        self.checks += 1
        self.sections[label.split('/')[0]] += 1
        if not bool(condition):
            self.failures.append(dict(check=label, detail=detail))

    def equal(self, actual, expected, label, tolerance=1e-11):
        if isinstance(expected, dict):
            self.check(isinstance(actual, dict), label + '/is_object')
            if not isinstance(actual, dict):
                return
            for key, value in expected.items():
                self.check(key in actual, label + '/' + key + '/present')
                if key in actual:
                    self.equal(actual[key], value, label + '/' + key, tolerance)
        elif isinstance(expected, (list, tuple)):
            self.check(isinstance(actual, (list, tuple)), label + '/is_array')
            if not isinstance(actual, (list, tuple)):
                return
            self.check(len(actual) == len(expected), label + '/length', [len(actual), len(expected)])
            for i, (a, e) in enumerate(zip(actual, expected)):
                self.equal(a, e, label + '/' + str(i), tolerance)
        elif isinstance(expected, float):
            self.check(finite_number(actual) and math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance), label, dict(actual=actual, expected=expected))
        else:
            self.check(type(actual) is type(expected) and actual == expected, label, dict(actual=actual, expected=expected))


def indexed(items, key_fields, audit, label):
    result = {}
    for row in items:
        key = tuple(row[k] for k in key_fields)
        audit.check(key not in result, label + '/unique', list(key))
        result[key] = row
    return result


def self_test():
    p = np.array([[.9, .8, .4], [.1, .3, .9], [.2, .7, .2], [.7, .1, .6]])
    y = np.array([[1., 1., np.nan], [0., 0., np.nan], [1., 1., np.nan], [0., 0., np.nan]])
    m = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]], dtype=bool)
    got = independent_metric(p, y, m)
    assert got['known'] == 7 and got['macro_balanced_accuracy'] == .75
    assert got['targets'][0] == dict(known=4, tp=1, tn=1, fp=1, fn=1, balanced_accuracy=.5, accuracy=.5)
    assert got['targets'][1] == dict(known=3, tp=1, tn=2, fp=0, fn=0, balanced_accuracy=1., accuracy=1.)
    assert got['targets'][2]['balanced_accuracy'] is None
    one = independent_metric(np.array([[.5], [.49]]), np.array([[1.], [1.]]), np.ones((2, 1), bool))
    assert one['macro_balanced_accuracy'] == .5
    assert mean([.1, .2, .3]) == math.fsum([.1, .2, .3]) / 3
    assert sorted([1, 2, 3, 4])[math.ceil(.95 * 3)] == 4
    audit = Audit()
    audit.equal(got, got, 'self/identity')
    assert not audit.failures
    audit.equal({'x': 1}, {'x': 2}, 'self/detect_wrong_value')
    assert len(audit.failures) == 1
    print(json.dumps(dict(status='SELF_TEST_PASS', model_calls=0, result_files_read=0)))


def verify(output_path):
    missing = [name for name in REQUIRED if not (UNIT / name).is_file()]
    if missing:
        print(json.dumps(dict(status='WAITING_FOR_COMPLETE_RESULTS', missing=missing)))
        return 2
    if output_path.exists():
        raise FileExistsError('Prior audit is preserved; choose a different --output filename: ' + str(output_path))
    a = Audit()
    report = dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  verifier=str(Path(__file__).resolve()), verifier_sha256=sha(__file__),
                  independent_arithmetic=True, project_metric_imported=False,
                  model_inference_calls=0, checkpoint_loads=0,
                  scope='Stored predictions/targets, logs, hashes, report arithmetic and DTW evidence contracts only')
    source_hashes = {name: sha(UNIT / name) for name in REQUIRED}
    try:
        proto, info = read(UNIT / 'PROTOCOL.json'), read(UNIT / 'LEARNING_INPUTS.json')
        rows, targets, selected = info['rows'], info['targets'], info['target_indices']
        aliases = [r['alias'] for r in rows]
        by_alias = {r['alias']: r for r in rows}
        alias_index = {r['alias']: i for i, r in enumerate(rows)}
        a.check(len(rows) == 80 and len(set(aliases)) == 80, 'inputs/80_unique_clips')
        a.equal(proto['targets'], targets, 'protocol/targets')
        a.equal(proto['conditions'], ['pose_capacity_matched', 'pose_plus_signrep'], 'protocol/conditions')
        a.equal(proto['seeds'], [42, 43, 44], 'protocol/seeds')
        a.equal(proto['epochs'], 160, 'protocol/epochs')
        a.equal(proto['selected_epoch'], 160, 'protocol/final_epoch')
        a.equal(proto['prediction_threshold'], .5, 'protocol/prediction_threshold')
        conditions = ['legacy_dtw'] + proto['conditions']
        fit = [i for i, r in enumerate(rows) if r['role'] == 'fit']
        fit_aliases, fit_ids = [rows[i]['alias'] for i in fit], [rows[i]['clip_id'] for i in fit]
        a.equal(fit_aliases, proto['fit'], 'protocol/fit_aliases')
        a.check(len(fit) == len(set(fit_ids)) == 36, 'inputs/36_unique_fit_ids')
        for group, expected_count, people in [('fit', 36, [1, 2, 3]), ('development', 12, [4]), ('held_person', 24, [5, 6]), ('held_expression_person', 8, [5, 6])]:
            group_rows = [r for r in rows if r['role'] == group]
            a.equal(len(group_rows), expected_count, 'inputs/' + group + '/count')
            a.equal(sorted({r['person'] for r in group_rows}), people, 'inputs/' + group + '/people')
            if group != 'fit':
                a.check(set(fit_ids).isdisjoint(r['clip_id'] for r in group_rows), 'inputs/' + group + '/disjoint_clip_ids')
        fit_words = {rows[i]['word_id'] for i in fit}
        a.check(fit_words.isdisjoint(r['word_id'] for r in rows if r['role'] == 'held_expression_person'), 'inputs/held_expression_disjoint_words')
        hash_groups = {'protected': proto['protected'], 'protocol_pins': proto['pins'], 'source_pins': info['source_pins']}
        for group, pins in hash_groups.items():
            for filename, expected in pins.items():
                a.equal(sha(filename), expected, 'hash/' + group + '/' + Path(filename).name)
        report['protected_files_verified'] = len(proto['protected'])
        report['protected_model_files_verified'] = sum(Path(p).suffix == '.pt' for p in proto['protected'])

        arrays, labels, frame_lookup, time_lookup = [], [], {}, {}
        for r in rows:
            label = 'inputs/' + r['alias']
            a.equal(sha(r['input']), r['input_sha256'], label + '/input_sha256')
            a.equal(sha(r['targets_file']), r['targets_sha256'], label + '/targets_sha256')
            with np.load(r['input'], allow_pickle=False) as source:
                data = {k: source[k] for k in ['x', 'mask', 'rgb', 'frame_id', 'timestamp_us']}
            with np.load(r['targets_file'], allow_pickle=False) as source:
                yy, mm = source['y'][:, selected], source['target_mask'][:, selected]
            n = len(data['x'])
            a.check(yy.shape == mm.shape == (n, len(targets)) and mm.dtype == np.bool_, label + '/target_shape_mask')
            a.check(np.isin(yy[mm], [0, 1]).all(), label + '/known_values_binary')
            a.check(np.isfinite(data['x'][data['mask']]).all() and np.isfinite(data['rgb']).all(), label + '/input_finite')
            a.check(len(set(data['frame_id'].tolist())) == n, label + '/unique_frames')
            arrays.append(data); labels.append((yy, mm))
            frame_lookup[r['alias']] = {int(f): i for i, f in enumerate(data['frame_id'])}
            time_lookup[r['alias']] = {int(f): int(t) / 1e6 for f, t in zip(data['frame_id'], data['timestamp_us'])}
        max_t, k = max(len(x['x']) for x in arrays), len(targets)
        y = np.zeros((len(rows), max_t, k), dtype=np.float32)
        mask = np.zeros_like(y, dtype=bool)
        for i, (yy, mm) in enumerate(labels):
            y[i, :len(yy)], mask[i, :len(mm)] = yy, mm
        counts = np.zeros((2, k), dtype=np.int64)
        for i in fit:
            yy, mm = labels[i]
            for j in range(k):
                for value in (0, 1):
                    counts[value, j] += sum(int(yy[t, j]) == value for t in np.flatnonzero(mm[:, j]))
        a.check((counts > 0).all(), 'training/both_classes_supported')
        weights = (counts.sum(0)[None, :] / (2 * counts)).tolist()
        stats = read(UNIT / 'NORMALIZATION.json')
        fit_x = np.concatenate([arrays[i]['x'] for i in fit])
        fit_m = np.concatenate([arrays[i]['mask'] for i in fit])
        fit_z = np.concatenate([arrays[i]['rgb'] for i in fit]).astype(np.float64)
        expected_pose_mean, expected_pose_std = [], []
        for j in range(69):
            observed = fit_x[fit_m[:, j], j].astype(np.float64)
            expected_pose_mean.append(float(observed.mean()) if len(observed) else 0.)
            expected_pose_std.append(max(float(observed.std()), .001) if len(observed) else 1.)
        # Reduction precision differs from stored float32 statistics; this tolerance is
        # only for normalization, not for reported classification scores.
        for field, expected in [('pose_mean', expected_pose_mean), ('pose_std', expected_pose_std), ('rgb_mean', fit_z.mean(0).tolist()), ('rgb_std', np.maximum(fit_z.std(0), .001).tolist())]:
            a.equal(stats[field], expected, 'normalization/fit_only/' + field, tolerance=2e-5)

        training = read(UNIT / 'TRAINING.json')
        a.equal(training['status'], 'COMPLETE', 'training/status')
        a.equal(training['backbone_updates'], 0, 'training/backbone_updates')
        a.equal(training['locked_test_scores_not_read'], True, 'training/no_test_selection_declaration')
        expected_runs = [(c, seed) for c in proto['conditions'] for seed in proto['seeds']]
        runs = indexed(training['runs'], ['condition', 'seed'], a, 'training/runs')
        a.check(set(runs) == set(expected_runs), 'training/exact_six_runs')
        scores, clip_scores = read(UNIT / 'SCORES.json'), read(UNIT / 'CLIP_SCORES.json')
        a.equal(scores['targets'], targets, 'scores/targets')
        saved_scores = indexed(scores['rows'], ['condition', 'seed', 'group'], a, 'scores/rows')
        saved_clips = indexed(clip_scores['rows'], ['condition', 'seed', 'alias'], a, 'scores/clips')
        a.equal(len(saved_scores), len(expected_runs) * len(GROUPS), 'scores/group_row_count')
        a.equal(len(saved_clips), len(expected_runs) * len(rows), 'scores/clip_row_count')
        recalculated, probabilities, parameters = {}, {}, []
        for condition, seed in expected_runs:
            name = f'{condition}_{seed}'
            directory = UNIT / 'runs' / name
            run = read(directory / 'RUN.json')
            prefix = 'training/' + name
            a.equal(run['condition'], condition, prefix + '/condition')
            a.equal(run['seed'], seed, prefix + '/seed')
            a.equal(run['epochs'], proto['epochs'], prefix + '/epochs')
            a.equal(run['selected_epoch'], proto['selected_epoch'], prefix + '/selected_epoch')
            a.equal(run['fit_aliases'], fit_aliases, prefix + '/fit_aliases')
            a.equal(run['backbone_updates'], 0, prefix + '/backbone_updates')
            a.equal(run['class_counts'], counts.astype(np.float32).tolist(), prefix + '/class_counts')
            a.equal(run['class_weights'], weights, prefix + '/class_weights', tolerance=1e-6)
            a.check(finite_number(run['seconds']) and run['seconds'] > 0, prefix + '/positive_runtime')
            parameters.append(run['parameters'])
            a.equal(runs[(condition, seed)], {key: run[key] for key in ['condition', 'seed', 'seconds', 'parameters']}, prefix + '/training_manifest')
            for filename, field in [('head.pt', 'checkpoint_sha256'), ('probabilities.npy', 'prediction_sha256'), ('EXPOSURES.jsonl', 'exposure_sha256')]:
                a.equal(sha(directory / filename), run[field], prefix + '/' + field)
            exposures = [json.loads(line) for line in (directory / 'EXPOSURES.jsonl').read_text(encoding='utf8').splitlines() if line.strip()]
            a.equal(len(exposures), 160, prefix + '/exposure_epoch_count')
            a.equal(len(run['loss_trace']), 160, prefix + '/loss_trace_length')
            for epoch, record in enumerate(exposures, 1):
                a.equal(record['epoch'], epoch, prefix + f'/exposure/{epoch}/epoch')
                a.equal(record['clip_ids'], fit_ids, prefix + f'/exposure/{epoch}/exact_fit_ids')
                a.check(len(record['clip_ids']) == len(set(record['clip_ids'])) == 36, prefix + f'/exposure/{epoch}/no_duplicate')
                a.check(finite_number(record['loss']) and record['loss'] >= 0, prefix + f'/exposure/{epoch}/finite_loss')
                if epoch <= len(run['loss_trace']):
                    a.equal(record['loss'], run['loss_trace'][epoch - 1], prefix + f'/exposure/{epoch}/trace_match')
            p = np.load(directory / 'probabilities.npy', allow_pickle=False)
            assert p.shape == y.shape, (name, p.shape, y.shape)
            assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), name
            probabilities[(condition, seed)] = p
            for i, r in enumerate(rows):
                independently = independent_metric(p[i], y[i], mask[i])
                a.equal(saved_clips[(condition, seed, r['alias'])], dict(condition=condition, seed=seed, alias=r['alias'], role=r['role'], **independently), 'scores/clip/' + name + '/' + r['alias'])
            for group in GROUPS:
                selected_clips = [i for i, r in enumerate(rows) if r['role'] == group]
                measured = independent_metric(p[selected_clips].reshape(-1, k), y[selected_clips].reshape(-1, k), mask[selected_clips].reshape(-1, k))
                clip_values = [independent_metric(p[i], y[i], mask[i])['macro_balanced_accuracy'] for i in selected_clips]
                measured.update(condition=condition, seed=seed, group=group, clips=len(selected_clips), clip_macro_mean=mean(v for v in clip_values if v is not None))
                recalculated[(condition, seed, group)] = measured
                a.equal(saved_scores[(condition, seed, group)], measured, 'scores/group/' + name + '/' + group)
        a.check(len(set(parameters)) == 1, 'training/matched_parameter_count')
        expected_parameter_count = 138*32+32 + 768*32+32 + 64*64*3+64 + 64*32*3+32 + 32*k+k
        a.equal(parameters[0], expected_parameter_count, 'training/architecture_parameter_count')
        saved_summary = indexed(scores['summary'], ['condition', 'group'], a, 'scores/summary')
        recalculated_summary = []
        for condition in proto['conditions']:
            for group in GROUPS:
                rr = [recalculated[(condition, seed, group)] for seed in proto['seeds']]
                values = [r['macro_balanced_accuracy'] for r in rr]
                summary = dict(condition=condition, group=group, mean=mean(values), range=[min(values), max(values)], clip_macro_mean=mean(r['clip_macro_mean'] for r in rr), per_seed=values)
                a.equal(saved_summary[(condition, group)], summary, 'scores/summary/' + condition + '/' + group)
                recalculated_summary.append(summary)
        report['recalculated_classification_summary'] = recalculated_summary
        weighting = []
        for group in GROUPS:
            summaries = {r['condition']: r for r in recalculated_summary if r['group']==group}
            base, rgb = (summaries[c] for c in proto['conditions'])
            counts_by_support = collections.Counter()
            known_by_clip = []
            for i,r in enumerate(rows):
                if r['role']!=group:
                    continue
                yy,mm=labels[i];known_by_clip.append(int(mm.sum()))
                for j in range(k):
                    classes={int(yy[t,j]) for t in np.flatnonzero(mm[:,j])}
                    counts_by_support[['no_known_class','one_known_class','both_known_classes'][len(classes)]]+=1
            pooled_delta=(rgb['mean']-base['mean'])*100
            clip_delta=(rgb['clip_macro_mean']-base['clip_macro_mean'])*100
            weighting.append(dict(group=group,pooled_target_balanced_delta_pp=pooled_delta,equal_clip_macro_delta_pp=clip_delta,directions_differ=pooled_delta*clip_delta<0,clip_target_support=dict(counts_by_support),known_cells_per_clip_range=[min(known_by_clip),max(known_by_clip)],explanation='Pooled target/class recall weights cells within each class; clip macro first weights each available target within each clip, then clips equally. Single-class clip-target entries average only available recall. These estimands can legitimately move in opposite directions.'))
        report['metric_weighting_diagnostics']=weighting
        report['training_exposures'] = dict(runs=6, epochs_each=160, unique_fit_clips_each=36, clip_epoch_exposures=6*160*36)

        def check_pair(pair, query, condition, label):
            reference = pair['reference']
            a.check(reference in fit_aliases and by_alias[reference]['word_id'] == by_alias[query]['word_id'], label + '/fit_same_word_reference')
            if pair['status'] == 'NO_PATH':
                a.equal(pair['distance'], None, label + '/no_path_distance')
                return
            path = pair['path']; nq, nr = pair['query_frames'], pair['reference_frames']
            a.equal(path[0], [0, 0], label + '/start')
            a.equal(path[-1], [nq-1, nr-1], label + '/end')
            a.check(all((v[0]-u[0], v[1]-u[1]) in {(0,1), (1,0), (1,1)} for u,v in zip(path,path[1:])), label + '/monotonic_steps')
            a.check({v[0] for v in path} == set(range(nq)) and {v[1] for v in path} == set(range(nr)), label + '/both_full_scopes')
            a.equal(pair['reference_scope_retained'], True, label + '/reference_scope_flag')
            qm = [int(statistics.median(v[1] for v in path if v[0] == i)) for i in range(nq)]
            rm = [int(statistics.median(v[0] for v in path if v[1] == i)) for i in range(nr)]
            a.equal(pair['query_to_reference'], qm, label + '/query_map')
            a.equal(pair['reference_to_query'], rm, label + '/reference_map')
            expected, compared, missing = pair['reference_expected_cells'], pair['compared_cells'], pair['missing_cells']
            a.check(0 <= compared <= expected <= nr*k and missing == expected-compared, label + '/reference_denominator')
            a.equal(pair['coverage'], compared/expected if expected else None, label + '/coverage')
            a.equal(pair['educational_verdict'], None, label + '/no_educational_verdict')
            a.equal(pair['mandatory_omission_verdict'], None, label + '/no_mandatory_verdict')
            a.check(finite_number(pair['distance']) and pair['distance'] >= 0, label + '/distance_finite')
            a.check(finite_number(pair['geometry_evidence']) and 0 <= pair['geometry_evidence'] <= 1+1e-12, label + '/geometry_evidence_range')
            status = 'NO_COMPARISON_EVIDENCE' if not compared else 'PARTIAL_EVIDENCE' if missing or pair['query_direct_unobserved_cells'] else 'ESTIMATED_COMPARISON'
            a.equal(pair['status'], status, label + '/evidence_status')
            if condition == 'legacy_dtw':
                a.equal(pair['events'], [], label + '/legacy_no_trained_events')
            for ei, event in enumerate(pair['events']):
                e_label = label + '/event/' + str(ei)
                j = targets.index(event['feature'])
                qp = float(probabilities[(condition, proto['feedback_seed'])][alias_index[query], frame_lookup[query][event['query_frame']], j])
                rp = float(probabilities[(condition, proto['feedback_seed'])][alias_index[reference], frame_lookup[reference][event['reference_frame']], j])
                a.equal(event['query_probability'], qp, e_label + '/saved_query_probability')
                a.equal(event['reference_probability'], rp, e_label + '/saved_reference_probability')
                a.equal(event['query_seconds'], time_lookup[query][event['query_frame']], e_label + '/query_time')
                a.equal(event['reference_seconds'], time_lookup[reference][event['reference_frame']], e_label + '/reference_time')
                a.check(max(qp,1-qp) >= proto['feature_feedback_confidence']-1e-7 and max(rp,1-rp) >= proto['feature_feedback_confidence']-1e-7 and abs(qp-rp) >= proto['feedback_probability_difference']-1e-7, e_label + '/fixed_thresholds')
                a.equal(event['kind'], 'ESTIMATED_FEATURE_DIFFERENCE_NOT_SIGN_ERROR', e_label + '/nonsemantic_kind')

        def check_aggregate(record, role, label):
            query, condition = record['query'], record['condition']
            a.equal(by_alias[query]['role'], role, label + '/query_role')
            a.equal(record['word_id'], by_alias[query]['word_id'], label + '/word_id')
            expected_refs = {r['alias'] for r in rows if r['role']=='fit' and r['word_id']==by_alias[query]['word_id']}
            a.check(len(record['pairs']) == 3 and {p['reference'] for p in record['pairs']} == expected_refs, label + '/exact_three_references')
            for pair in record['pairs']:
                check_pair(pair, query, condition, label + '/pair/' + pair['reference'])
            valid = [p for p in record['pairs'] if p['distance'] is not None and p.get('compared_cells',0)>0]
            excluded = len(record['pairs'])-len(valid)
            expected_distance = statistics.median(p['distance'] for p in valid) if valid else None
            a.equal(record['distance'], expected_distance, label + '/median_supported_distance')
            a.equal(record['compared_reference_count'], len(valid), label + '/compared_reference_count')
            a.equal(record['excluded_reference_count'], excluded, label + '/excluded_reference_count')
            coverage = [p['coverage'] for p in record['pairs'] if p.get('coverage') is not None]
            a.equal(record['minimum_reference_coverage'], min(coverage) if coverage else None, label + '/minimum_coverage')
            status = 'NO_COMPARISON_EVIDENCE' if not valid else 'PARTIAL_EVIDENCE' if excluded or any(p['status']!='ESTIMATED_COMPARISON' for p in valid) else 'ESTIMATED_COMPARISON'
            a.equal(record['status'], status, label + '/status')
            a.equal(record['comparison_is_not_correctness'], True, label + '/not_correctness')
            votes = collections.defaultdict(set)
            for pair in valid:
                for event in pair['events']:
                    votes[(event['query_frame'],event['feature'])].add(pair['reference'])
            expected_keys = {key for key,refs in votes.items() if len(refs)>=2}
            actual_keys = {(e['query_frame'],e['feature']) for e in record['events']}
            a.check(len(actual_keys)==len(record['events']) and actual_keys==expected_keys, label + '/exact_consensus_keys')
            for event in record['events']:
                key=(event['query_frame'],event['feature'])
                a.check(set(event['supporting_references'])==votes[key], label + '/consensus_references')
                a.equal(event['educational_verdict'], None, label + '/consensus_no_verdict')
                a.check(len(event['evidence'])==len(votes[key]), label + '/consensus_evidence_count')
                for evidence in event['evidence']:
                    pp=next(p for p in valid if p['reference']==evidence['reference'])
                    ee=next(e for e in pp['events'] if (e['query_frame'],e['feature'])==key)
                    a.equal(evidence,dict(reference=pp['reference'],**ee),label+'/consensus_evidence')

        dev = read(UNIT / 'DEVELOPMENT_COMPARISON.json')['rows']
        frozen = read(UNIT / 'FEEDBACK_FREEZE.json')
        paired = read(UNIT / 'PAIRED_FEEDBACK.json')
        a.equal(frozen['comparison_code_sha256'], sha(UNIT/'feedback.py'), 'feedback/frozen_code')
        a.equal(frozen['protocol_sha256'], sha(UNIT/'PROTOCOL.json'), 'feedback/frozen_protocol')
        a.equal(frozen['selected_seed'], proto['feedback_seed'], 'feedback/fixed_seed')
        a.equal(frozen['mandatory_requirements_from_unconfirmed_guide'], False, 'feedback/no_unconfirmed_requirements')
        for group, data in [('development',dev), ('held_person',paired['rows'])]:
            index=indexed(data,['condition','query'],a,'feedback/'+group)
            expected_keys={(c,r['alias']) for c in conditions for r in rows if r['role']==group}
            a.check(set(index)==expected_keys,'feedback/'+group+'/complete_manifest')
            for (condition,query),record in index.items():
                check_aggregate(record,group,'feedback/'+group+'/'+condition+'/'+query)
        for condition in conditions:
            distances=sorted(r['distance'] for r in dev if r['condition']==condition and r['distance'] is not None)
            threshold=distances[math.ceil(.95*(len(distances)-1))] if distances else None
            a.equal(frozen['thresholds'][condition],threshold,'feedback/calibration/'+condition+'/higher95')
            a.equal(frozen['calibration_valid_queries'][condition],len(distances),'feedback/calibration/'+condition+'/valid_queries')
        a.equal(frozen['calibration_queries'],12,'feedback/calibration/queries')
        for record in paired['rows']:
            threshold=frozen['thresholds'][record['condition']]
            expected_alert=record['distance']>threshold if record['distance'] is not None and threshold is not None else None
            a.equal(record['distance_threshold'],threshold,'feedback/locked/threshold')
            a.equal(record['descriptive_distance_alert'],expected_alert,'feedback/locked/alert')
        summary_index=indexed(paired['summary'],['condition'],a,'feedback/summary')
        for condition in conditions:
            rr=[r for r in paired['rows'] if r['condition']==condition]
            coverage=[r['minimum_reference_coverage'] for r in rr if r.get('minimum_reference_coverage') is not None]
            expected=dict(condition=condition,queries=len(rr),reference_pairs=sum(len(r['pairs']) for r in rr),descriptive_distance_alerts=sum(r['descriptive_distance_alert'] is True for r in rr),descriptive_distance_evaluable_queries=sum(r['descriptive_distance_alert'] is not None for r in rr),descriptive_distance_withheld_queries=sum(r['descriptive_distance_alert'] is None for r in rr),feature_difference_candidate_clips=sum(bool(r['events']) for r in rr),candidate_cells=sum(len(r['events']) for r in rr),all_reference_scopes_retained=all(p.get('reference_scope_retained',False) for r in rr for p in r['pairs']),minimum_reference_coverage=min(coverage) if coverage else None,educational_false_correction_rate=None)
            a.equal(summary_index[(condition,)],expected,'feedback/summary/'+condition)
        a.equal(paired['semantic_match_verified'],False,'feedback/no_semantic_accuracy')
        a.check({r['alias'] for r in paired['queries']}=={r['alias'] for r in rows if r['role']=='held_person'} and len(paired['queries'])==24,'feedback/export_manifest')

        controls=read(UNIT/'CONTROLS.json')
        control_index=indexed(controls['rows'],['alias','condition','control'],a,'controls/manifest')
        control_names=['known_time_map_0.75','known_time_map_1.25','R_sensor_missing','L_sensor_missing','short_middle_deleted']
        expected_keys={(r['alias'],c,name) for r in rows if r['role']=='held_person' for c in conditions for name in control_names}
        a.check(set(control_index)==expected_keys and len(control_index)==360,'controls/exact360_rows')
        a.equal(controls['semantic_error_accuracy'],None,'controls/no_semantic_accuracy')
        a.equal(controls['synthetic_not_real_errors'],True,'controls/synthetic_scope')
        for key,record in control_index.items():
            label='controls/'+'/'.join(key)
            if 'sensor_missing' in key[2]:
                a.equal(record['pass_contract'],True,label+'/pass_contract')
                a.equal(record['same_denominator'],True,label+'/same_denominator')
                a.check(0<=record['missing_cells']<=record['expected_cells'],label+'/missing_bound')
            else:
                a.equal(record['reference_scope_retained'],True,label+'/reference_scope')
                if key[2].startswith('known_time_map'):
                    for field in ['mean_source_frame_error','mean_time_error_seconds']:
                        a.check(finite_number(record[field]) and record[field]>=0,label+'/'+field)
                else:
                    a.check(record['deleted_frames']>0 and record['max_reference_reuse']>=1,label+'/deletion_present')
                    a.check(finite_number(record['distance']) and record['distance']>=0,label+'/distance')
                    a.equal(record['mandatory_omission_accuracy'],None,label+'/no_mandatory_accuracy')
        report['feedback_counts']=dict(development_rows=len(dev),locked_rows=len(paired['rows']),control_rows=len(controls['rows']))

        temporal_truth=read(UNIT/'review/HELD_TEMPORAL_RGB.json')
        temporal_freeze=read(UNIT/'TEMPORAL_EVAL_FREEZE.json')
        temporal_scores=read(UNIT/'TEMPORAL_SCORES.json')
        runtime_protocol=read(UNIT/'RUNTIME_PROTOCOL.json')
        a.equal(temporal_truth['aliases'],runtime_protocol['temporal_eval'],'temporal/prespecified_aliases')
        a.equal(temporal_freeze['labels_sha256'],sha(UNIT/'review/HELD_TEMPORAL_RGB.json'),'temporal/label_freeze')
        a.equal(temporal_freeze['code_sha256'],sha(UNIT/'temporal_metrics.py'),'temporal/code_freeze')
        a.equal(temporal_freeze['threshold'],.5,'temporal/threshold')
        for filename,expected in temporal_truth['pins'].items():
            a.equal(sha(filename),expected,'temporal/label_evidence_pin/'+Path(filename).name)
        temporal_index=indexed(temporal_scores['rows'],['condition','seed'],a,'temporal/runs')
        a.check(set(temporal_index)==set(expected_runs),'temporal/exact_six_runs')
        temporal_summary=[]
        for condition,seed in expected_runs:
            expected_events=[];expected_clips=[]
            for rr in temporal_truth['rows']:
                alias=rr['alias'];i=alias_index[alias];n=len(arrays[i]['frame_id'])
                a.equal(by_alias[alias]['role'],'held_person','temporal/'+alias+'/held_role')
                p=probabilities[(condition,seed)][i,:n]
                clip=dict(alias=alias,known=0,correct=0,sampled_stable_pairs=0,predicted_changes_between_equal_observed_samples=0)
                for feature in temporal_truth['features']:
                    j=targets.index(feature)
                    for sample in rr['frames']:
                        actual=sample['labels'][feature]
                        a.equal(sample['seconds'],time_lookup[alias][sample['frame_id']],'temporal/'+alias+'/sample_timestamp',tolerance=1e-6)
                        if actual is not None:
                            clip['known']+=1
                            clip['correct']+=int(int(p[frame_lookup[alias][sample['frame_id']],j]>=.5)==actual)
                    for before,after in zip(rr['frames'],rr['frames'][1:]):
                        v1,v2=before['labels'][feature],after['labels'][feature]
                        if v1 is not None and v1==v2:
                            clip['sampled_stable_pairs']+=1
                            b=int(p[frame_lookup[alias][before['frame_id']],j]>=.5)
                            e=int(p[frame_lookup[alias][after['frame_id']],j]>=.5)
                            clip['predicted_changes_between_equal_observed_samples']+=int(b!=e)
                for event in rr['transition_brackets']:
                    j=targets.index(event['feature'])
                    lo,hi=(frame_lookup[alias][f] for f in event['frame_bracket'])
                    a.check(0<=lo<hi<n,'temporal/'+alias+'/real_frame_bracket')
                    prediction=[int(float(v)>=.5) for v in p[:,j]]
                    changes=[t for t in range(1,n) if prediction[t-1]==event['from_value'] and prediction[t]==event['to_value']]
                    within=[t for t in changes if lo<t<=hi]
                    endpoints=prediction[lo]==event['from_value'] and prediction[hi]==event['to_value']
                    left,right=event['seconds_bracket']
                    nearest=0. if within else min((min(abs(float(arrays[i]['timestamp_us'][t])/1e6-left),abs(float(arrays[i]['timestamp_us'][t])/1e6-right)) for t in changes),default=None)
                    expected_events.append(dict(alias=alias,**event,endpoints_correct=endpoints,predicted_transition_in_bracket=bool(within),both=bool(endpoints and within),nearest_transition_distance_outside_bracket_seconds=nearest,predicted_transition_frames=[int(arrays[i]['frame_id'][t]) for t in within]))
                expected_clips.append(clip)
            expected=dict(condition=condition,seed=seed,events=expected_events,clips=expected_clips,transition_brackets=len(expected_events),both_endpoints_and_bracket=sum(e['both'] for e in expected_events),known_cells=sum(c['known'] for c in expected_clips),correct_cells=sum(c['correct'] for c in expected_clips),sampled_stable_pairs=sum(c['sampled_stable_pairs'] for c in expected_clips),stable_pair_predicted_changes=sum(c['predicted_changes_between_equal_observed_samples'] for c in expected_clips))
            a.equal(temporal_index[(condition,seed)],expected,'temporal/scores/'+condition+'/'+str(seed))
            temporal_summary.append({key:expected[key] for key in ['condition','seed','transition_brackets','both_endpoints_and_bracket','known_cells','correct_cells','sampled_stable_pairs','stable_pair_predicted_changes']})
        report['recalculated_temporal_summary']=temporal_summary
        report['temporal_limits']='Only sparse observed-state brackets and sampled endpoint changes are verified. Equal observed sample endpoints do not prove the intervening interval was stable; the stable-pair metric does not count every interior predicted reversal. This is not exact timing MAE or semantic-stage correctness.'

        runtime=read(UNIT/'RUNTIME_RESULT.json')
        runtime_index=indexed(runtime['rows'],['alias','repeat','condition'],a,'runtime/rows')
        runtime_keys={(alias,repeat,c) for alias in runtime_protocol['aliases'] for repeat in [1,2] for c in conditions}
        a.check(set(runtime_index)==runtime_keys and len(runtime_index)==18,'runtime/exact18_rows')
        a.check(finite_number(runtime['setup_seconds']) and runtime['setup_seconds']>=0,'runtime/setup_seconds')
        verified_capture_pins=set()
        for key,record in runtime_index.items():
            label='runtime/'+str(key)
            components=['mediapipe_decode_assignment_save_seconds','rgb_decode_gpu_seconds','parallel_frontend_seconds','feature_processing_seconds','head_seconds','dtw_three_references_seconds','response_serialization_seconds']
            for field in components:
                a.check(finite_number(record[field]) and record[field]>=0,label+'/'+field)
            front=record['parallel_frontend_seconds'] if record['condition']=='pose_plus_signrep' else record['mediapipe_decode_assignment_save_seconds']
            total=math.fsum([front]+[record[f] for f in ['feature_processing_seconds','head_seconds','dtw_three_references_seconds','response_serialization_seconds']])
            a.equal(record['total_component_seconds'],total,label+'/component_sum')
            a.check(record['parallel_frontend_seconds']+1e-8>=max(record['mediapipe_decode_assignment_save_seconds'],record['rgb_decode_gpu_seconds']),label+'/parallel_frontend_covers_workers')
            a.equal(record['accuracy_not_rescored'],True,label+'/no_accuracy_rescore')
            response=Path(record['response']).read_bytes()
            a.equal(len(response),record['response_bytes'],label+'/response_bytes')
            payload=json.loads(response)
            a.equal(payload['query'],record['alias'],label+'/response_query')
            a.equal(payload['condition'],record['condition'],label+'/response_condition')
            a.equal(len(payload['pairs']),3,label+'/three_references')
            for filename,expected in record['capture_pins'].items():
                if filename not in verified_capture_pins:
                    a.equal(sha(filename),expected,label+'/capture_pin/'+Path(filename).name)
                    verified_capture_pins.add(filename)
        runtime_summary=indexed(runtime['summary'],['condition'],a,'runtime/summary')
        recalculated_runtime=[]
        for condition in conditions:
            values=sorted(r['total_component_seconds'] for r in runtime['rows'] if r['condition']==condition)
            pos=.95*(len(values)-1);low,high=math.floor(pos),math.ceil(pos)
            percentile=values[low]+(values[high]-values[low])*(pos-low)
            expected=dict(condition=condition,n=len(values),median=statistics.median(values),p95=percentile,range=[values[0],values[-1]])
            a.equal(runtime_summary[(condition,)],expected,'runtime/summary/'+condition)
            recalculated_runtime.append(expected)
        report['recalculated_runtime_summary']=recalculated_runtime
        report['runtime_scope']='Component sums after recording-end simulation with fixed public ROI and precomputed references. Pose/legacy capture is shared with the GPU-concurrent trial and is not a separately executed CPU-only route. Six observations per condition provide a descriptive p95, not a production guarantee. Browser time and CPU-only SignRep are outside this file.'
        report['limits']=[
            'Model weights were hashed, not loaded or re-inferred. Logs confirm recorded exposures, not arbitrary unseen internal behavior.',
            'DTW path completeness, stored evidence arithmetic, probability/event provenance, thresholds and consensus are checked; DTW cost matrices and detector support are not recomputed.',
            'Controls do not save their underlying paths, so saved alignment-error magnitudes are checked for finite range and full-case coverage, not independently reconstructed.',
            'Label truth remains provisional same-Codex RGB observation, not independent human or educational correctness.'
        ]
    except Exception as exc:
        a.check(False,'verifier/exception',dict(type=type(exc).__name__,message=str(exc)))
        import traceback
        report['exception_traceback']=traceback.format_exc()
    for name, expected in source_hashes.items():
        a.equal(sha(UNIT/name),expected,'stability/'+name)
    report.update(status='PASS' if not a.failures else 'FAIL',checks=a.checks,checks_by_section=dict(a.sections),failures=a.failures,source_sha256=source_hashes,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    with output_path.open('x',encoding='utf8') as out:
        json.dump(report,out,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(status=report['status'],checks=a.checks,failures=len(a.failures),output=str(output_path)),ensure_ascii=False))
    return 0 if not a.failures else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-test',action='store_true')
    mode.add_argument('--verify',action='store_true')
    parser.add_argument('--output',default='RESULTS_VERIFICATION.json',help='New JSON filename within review/; prior audits cannot be overwritten')
    args=parser.parse_args()
    if args.self_test:
        self_test();return 0
    output=(REVIEW/args.output).resolve()
    if output.parent!=REVIEW.resolve() or output.suffix.lower()!='.json':
        parser.error('--output must be a JSON filename directly within review/')
    return verify(output)


if __name__=='__main__':
    raise SystemExit(main())
