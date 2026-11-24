"""Independent saved-training audit. No producer imports, torch, model load or forward.

Reads fixed JSON/JSONL/NPY/NPZ and streams checkpoint hashes only. The original
results are never changed. A failed first audit is retained, not overwritten.
"""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import traceback
import numpy as np

U = Path(__file__).resolve().parents[1]
SEM = U.parent / 'word100_semantic_feature_cycle_20261101'
OUT = U / 'review/SAVED_TRAINING_VERIFICATION.json'
TARGETS = ['R_index_only', 'R_open_four', 'L_open_four', 'R_closed_four',
           'L_closed_four', 'R_higher', 'L_higher']
CONDITIONS = ['matched_three_people', 'expanded_nine_people']
SEEDS = [42, 43, 44]
checks = 0
failures = []
pins = {}


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def check(ok, label, detail=None):
    global checks
    checks += 1
    if not bool(ok):
        failures.append({'check': label, 'detail': detail})


def equal(actual, expected, label, atol=1e-12):
    if isinstance(expected, dict):
        check(isinstance(actual, dict), label + ':dict')
        if not isinstance(actual, dict):
            return
        for k, v in expected.items():
            check(k in actual, label + ':' + str(k) + ':present')
            if k in actual:
                equal(actual[k], v, label + ':' + str(k), atol)
    elif isinstance(expected, list):
        check(isinstance(actual, list) and len(actual) == len(expected), label + ':length')
        if isinstance(actual, list) and len(actual) == len(expected):
            for i, (a, b) in enumerate(zip(actual, expected)):
                equal(a, b, label + ':' + str(i), atol)
    elif isinstance(expected, float):
        check(actual is not None and math.isfinite(float(actual)) and
              math.isclose(float(actual), expected, rel_tol=1e-10, abs_tol=atol), label,
              None if actual == expected else {'actual': actual, 'expected': expected})
    else:
        check(actual == expected, label, None if actual == expected else
              {'actual': actual, 'expected': expected})


def pin(p, expected=None):
    p = str(Path(p).resolve())
    got = sha(p)
    if p in pins:
        check(got == pins[p], 'source_stable_during_audit:' + p)
    pins[p] = got
    if expected is not None:
        check(got == expected, 'sha256:' + p)
    return got


def observed_metrics(p, rows):
    """Confusion counts first: unknowns never enter TP/FP/TN/FN."""
    result = []
    confusion = []
    for j, name in enumerate(TARGETS):
        tp = fp = tn = fn = unknown = 0
        for i, row in enumerate(rows):
            if not row['target_mask'][j]:
                unknown += 1
                continue
            predicted, truth = float(p[i, j]) >= .5, row['y'][j] == 1
            tp += int(predicted and truth)
            fp += int(predicted and not truth)
            tn += int(not predicted and not truth)
            fn += int(not predicted and truth)
        positive, negative = tp + fn, tn + fp
        known = positive + negative
        pa = tp / positive if positive else None
        na = tn / negative if negative else None
        result.append(dict(target=name, possible_cells=len(rows), known=known, unknown=unknown,
            positive=positive, negative=negative, correct=tp + tn,
            accuracy=(tp + tn) / known if known else None,
            positive_accuracy=pa, negative_accuracy=na,
            balanced_accuracy=(pa + na) / 2 if positive and negative else None))
        confusion.append(dict(target=name, tp=tp, fp=fp, tn=tn, fn=fn, unknown=unknown))
    accuracies = [r['accuracy'] for r in result if r['accuracy'] is not None]
    old = [r['accuracy'] for r in result[:3] if r['accuracy'] is not None]
    return dict(rows=len(rows), possible_cells=len(rows) * 7,
        known_cells=sum(r['known'] for r in result), unknown_cells=sum(r['unknown'] for r in result),
        per_target=result, macro_supported_target_accuracy=sum(accuracies) / len(accuracies),
        old_three_macro_accuracy=sum(old) / len(old)), confusion


def distribution(values):
    if any(v is None for v in values):
        return dict(values=values, mean=None, std=None, minimum=None, maximum=None)
    mean = sum(values) / len(values)
    return dict(values=values, mean=mean,
                std=math.sqrt(sum((x - mean)**2 for x in values) / len(values)),
                minimum=min(values), maximum=max(values))


def audit():
    report = read(U / 'TRAINING_REPORT.json')
    ready = read(U / 'TRAIN_READY.json')
    frozen = read(U / 'FROZEN.json')
    prep = read(U / 'TRAIN_PREPARED.json')
    data = read(U / 'TRAIN_DATASET.json')
    bank = read(U / 'PUBLIC_INPUTS.json')
    labels = read(U / 'data_audit/LABELS.json')
    original = read(SEM / 'DATASET.json')
    check(report['status'] == 'COMPLETE_DEVELOPMENT_COMPARISON', 'completed_report')
    check(ready['allow_training'] is True and ready['status'].startswith('PASS'), 'ready_approved')
    for key, rel in [('frozen_sha256', 'FROZEN.json'), ('dataset_sha256', 'TRAIN_DATASET.json'),
        ('prepared_sha256', 'TRAIN_PREPARED.json'), ('code_sha256', 'train_expansion.py'),
        ('labels_sha256', 'data_audit/LABELS.json'), ('public_inputs_sha256', 'PUBLIC_INPUTS.json'),
        ('validation_sha256', 'DATASET_VALIDATION.json'),
        ('separate_code_review_sha256', 'review/TRAIN_CODE_REVIEW_FINAL.json'),
        ('root_secondary_visual_review_sha256', 'LABEL_SECONDARY_REVIEW.json')]:
        pin(U / rel, ready[key])
    ready_sha = pin(U / 'TRAIN_READY.json', report['training_ready_sha256'])
    dataset_sha = pin(U / 'TRAIN_DATASET.json', prep['dataset_sha256'])
    code_sha = pin(U / 'train_expansion.py', report['code_sha256'])
    report_sha = pin(U / 'TRAINING_REPORT.json')
    pin(U / 'FROZEN.json', prep['frozen_sha256'])
    for kind in ['source_pins', 'protected_pins']:
        for path, digest in prep[kind].items():
            pin(path, digest)
    equal(data['source_pins'], prep['source_pins'], 'dataset_source_pins')
    equal(frozen['learning'], dict(targets=TARGETS, conditions=CONDITIONS, seeds=SEEDS,
          epochs=160, threshold=.5, lr=.001, weight_decay=.01), 'frozen_learning')
    equal(report['targets'], TARGETS, 'report_targets')
    equal(report['seeds'], SEEDS, 'report_seeds')
    equal(data['targets'], TARGETS, 'dataset_targets')
    equal(labels['targets'], TARGETS, 'labels_targets')
    equal(data['feature_names'], original['feature_names'], 'original_69_feature_order')
    check(len(data['feature_names']) == 69, '69_features')
    equal(frozen['fit_person_ids'], list(range(1, 10)), 'FIT_people_contract')
    equal(frozen['development_person_ids'], [11, 12, 13], 'DEV_people_contract')
    check(report['historical_exposure'] == frozen['historical_exposure'], 'historical_reuse_disclosed')

    # Reconstruct the row identity/order from the two immutable label sources.
    byclip = {r['clip_id']: r for r in bank['rows']}
    check(len(byclip) == 48 and len(labels['rows']) == 144, 'bank48_labels144')
    expected = [dict(r, sample_id='old:' + r['sample_id'], cohort='original_fit')
                for r in original['rows'] if r['role'] == 'fit']
    for r in labels['rows']:
        if r['person'] <= 3:
            continue
        source = byclip[r['clip_id']]
        check((r['word_id'], r['person'], r['role']) ==
              (source['word_id'], source['person'], source['role']), 'label_bank_identity:' + r['sample_id'])
        expected.append(dict(r, sample_id=f"new:{r['clip_id']}:{r['center_frame']}",
            input=source['input'], input_sha256=source['input_sha256'],
            cohort='development' if r['person'] in [11, 12, 13] else 'expansion'))
    rows = data['rows']
    check(rows == expected, 'exact_original180_and_extra108_rows_order_values_masks')
    check(Counter(r['cohort'] for r in rows) == {'original_fit': 180, 'expansion': 72, 'development': 36},
          'fixed180_72_36')
    check(len({r['sample_id'] for r in rows}) == 288, 'unique_sample_ids')
    dev = [r for r in rows if r['role'] == 'development']
    dev_ids = [r['sample_id'] for r in dev]
    equal(report['development_sample_ids'], dev_ids, 'report_DEV_order')
    check(sorted({r['person'] for r in dev}) == [11, 12, 13], 'DEV_people')
    check(len({r['clip_id'] for r in dev}) == 12, 'DEV12_clips')
    check(Counter((r['word_id'], r['person']) for r in dev) ==
          {(w, p): 3 for w in frozen['word_ids'] for p in [11, 12, 13]}, 'DEV4words_3people_3centers')
    check(sum(sum(r['target_mask']) for r in dev) == 194, 'DEV194_known58_unknown')

    # Read one input clip at a time; retain only short windows and small metadata.
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row['input']].append((i, row))
    windows = [None] * len(rows)
    metadata = {}
    for path, rr in groups.items():
        pin(path, rr[0][1]['input_sha256'])
        check(len({r['role'] for _, r in rr}) == 1, 'no_clip_role_leak:' + path)
        with np.load(path, allow_pickle=False) as z:
            a = {k: z[k] for k in ('x', 'mask', 'rgb', 'frame_id', 'timestamp_us')}
        n = len(a['frame_id'])
        check(a['x'].shape == (n, 69) and a['mask'].shape == (n, 69) and
              a['rgb'].shape == (n, 768), 'raw_dimensions:' + path)
        check(a['mask'].dtype == np.bool_ and np.isfinite(a['x'][a['mask']]).all() and
              np.isfinite(a['rgb']).all(), 'raw_known_values_finite:' + path)
        fingerprint = hashlib.sha256()
        for name in ('x', 'mask', 'rgb', 'frame_id', 'timestamp_us'):
            arr = np.ascontiguousarray(a[name])
            for part in (name.encode(), arr.dtype.str.encode(), str(arr.shape).encode(), arr.tobytes()):
                fingerprint.update(part)
        for i, row in rr:
            ix = [int(np.flatnonzero(a['frame_id'] == f)[0]) for f in row['frame_ids']]
            check(len(ix) == 13 and np.array_equal(np.diff(ix), np.ones(12)) and
                  row['frame_ids'][6] == row['center_frame'], 'window_exact13:' + row['sample_id'])
            check(a['timestamp_us'][ix].tolist() == row['timestamps_us'] and
                  int(a['timestamp_us'][ix[6]]) == row['center_timestamp_us'], 'window_PTS:' + row['sample_id'])
            windows[i] = {k: a[k][ix] for k in ('x', 'mask', 'rgb')}
        metadata[rr[0][1]['clip_id']] = dict(fingerprint=fingerprint.hexdigest(),
            frame_id=a['frame_id'].copy(), timestamp_us=a['timestamp_us'].copy())

    fit_sets = {}
    norm_statistics = {}
    for condition in CONDITIONS:
        ids = [i for i, r in enumerate(rows) if r['role'] == 'fit' and
               (condition == CONDITIONS[1] or r['cohort'] == 'original_fit')]
        fit = [rows[i] for i in ids]
        fit_sets[condition] = fit
        declared = prep['conditions'][condition]
        equal(declared['fit_sample_ids'], [r['sample_id'] for r in fit], condition + ':FIT_order')
        equal(declared['fit_people'], sorted({r['person'] for r in fit}), condition + ':FIT_people')
        check(not {r['person'] for r in fit} & {r['person'] for r in dev}, condition + ':person_disjoint')
        check(not {r['clip_id'] for r in fit} & {r['clip_id'] for r in dev}, condition + ':clip_disjoint')
        check(len(fit) == declared['fit_rows'] and len({r['clip_id'] for r in fit}) == declared['fit_clips'],
              condition + ':FIT_denominator')
        counts = [[sum(r['target_mask'][j] and r['y'][j] == sign for r in fit)
                   for j in range(7)] for sign in (0, 1)]
        equal(declared['class_counts_zero_one'], counts, condition + ':known_only_class_counts')
        weights = np.asarray([[(counts[0][j] + counts[1][j]) / (2 * counts[s][j])
                              for j in range(7)] for s in range(2)], np.float32).tolist()
        equal(declared['class_weights_zero_one'], weights, condition + ':FIT_class_weights')
        known_clip_targets = len({(r['clip_id'], j) for r in fit for j in range(7) if r['target_mask'][j]})
        equal(declared['equal_clip_target_event_sum'], float(known_clip_targets), condition + ':event_denominator')
        pin(declared['normalization'], declared['normalization_sha256'])
        norm = read(declared['normalization'])
        equal(norm['fit_sample_ids'], [r['sample_id'] for r in fit], condition + ':normalization_FIT_only')
        x = np.concatenate([windows[i]['x'] for i in ids])
        m = np.concatenate([windows[i]['mask'] for i in ids])
        rgb = np.concatenate([windows[i]['rgb'] for i in ids])
        calculated = dict(pose_mean=[float(x[:, j][m[:, j]].mean()) if m[:, j].any() else 0. for j in range(69)],
            pose_std=[max(.001, float(x[:, j][m[:, j]].std())) if m[:, j].any() else 1. for j in range(69)],
            rgb_mean=rgb.mean(0).tolist(), rgb_std=np.maximum(rgb.std(0), .001).tolist())
        for key in calculated:
            check(np.allclose(norm[key], calculated[key], rtol=0, atol=1e-7), condition + ':raw_FIT_' + key)
        norm_statistics[condition] = dict(fit_rows=len(fit), fit_clips=declared['fit_clips'],
            fit_people=declared['fit_people'], known_target_cells=sum(map(sum, counts)),
            class_counts_zero_one=counts, known_clip_target_events=known_clip_targets,
            frame_occurrences=len(ids) * 13, normalization_sha256=declared['normalization_sha256'])

    check(len(report['results']) == 6 and len(report['protected_baseline']) == 3, 'six_runs_three_baselines')
    check({(r['condition'], r['seed']) for r in report['results']} ==
          {(c, s) for c in CONDITIONS for s in SEEDS}, 'complete_fixed_run_grid')
    audit_metrics = {}
    compact_runs = []
    for record in report['results'] + report['protected_baseline']:
        condition = record.get('condition', 'protected_baseline')
        seed = record['seed']
        tag = f'{condition}_{seed}'
        folder = U / 'trainruns' / tag
        complete = read(folder / 'COMPLETE.json')
        check({k: v for k, v in complete.items() if k != 'status'} == record, tag + ':manifest_report_exact')
        pin(folder / 'COMPLETE.json')
        p = np.load(folder / 'probs.npy', allow_pickle=False)
        pin(folder / 'probs.npy', record['probabilities_sha256'])
        check(p.shape == (36, 7) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), tag + ':valid_probs36x7')
        equal(record['development_sample_ids'], dev_ids, tag + ':same_DEV_order')
        metric, confusion = observed_metrics(p, dev)
        equal(record['metrics'], metric, tag + ':independent_metrics')
        check(metric['known_cells'] == 194 and metric['unknown_cells'] == 58, tag + ':fixed_known_unknown')
        check(metric['per_target'][0]['balanced_accuracy'] is None and
              metric['per_target'][0]['positive'] == 0, tag + ':absent_index_positive_is_not_balanced_accuracy')
        for grouping in ('word_id', 'person'):
            for value in sorted({r[grouping] for r in dev}):
                indices = [i for i, r in enumerate(dev) if r[grouping] == value]
                mm, _ = observed_metrics(p[indices], [dev[i] for i in indices])
                equal(record['breakdown'][grouping][str(value)], mm, tag + ':' + grouping + ':' + str(value))
        audit_metrics[(condition, seed)] = metric
        compact_runs.append(dict(condition=condition, seed=seed, metrics=metric, confusion=confusion))
        if condition in CONDITIONS:
            fit = fit_sets[condition]
            equal(record['fit_sample_ids'], [r['sample_id'] for r in fit], tag + ':exact_FIT_order')
            equal(record['fit_people'], sorted({r['person'] for r in fit}), tag + ':people')
            check(record['epochs'] == record['selected_epoch'] == 160 and record['fit_rows'] == len(fit), tag + ':fixed_epoch160')
            check(complete['status'] == 'COMPLETE_FIXED_FINAL_EPOCH', tag + ':complete')
            pin(record['checkpoint'], record['checkpoint_sha256'])
            check(Path(record['checkpoint']).resolve() == (folder / 'model.pt').resolve() and
                  Path(record['checkpoint']).stat().st_size > 0, tag + ':checkpoint_identity')
            pin(folder / 'norm.json', record['normalization_sha256'])
            check(record['normalization_sha256'] == prep['conditions'][condition]['normalization_sha256'], tag + ':prepared_norm_unchanged')
            equal(record['class_counts_zero_one'], norm_statistics[condition]['class_counts_zero_one'], tag + ':class_counts')
            check(record['code_sha256'] == code_sha and record['ready_sha256'] == ready_sha and
                  record['dataset_sha256'] == dataset_sha, tag + ':execution_pins')
            pin(folder / 'EXPOSURES.jsonl', record['exposures_sha256'])
            trace = [json.loads(line) for line in (folder / 'EXPOSURES.jsonl').read_text(encoding='utf8').splitlines()]
            check(len(trace) == len(record['losses']) == 160, tag + ':160_updates_recorded')
            for epoch, line in enumerate(trace, 1):
                check(line['epoch'] == epoch and line['fit_rows'] == len(fit) and
                      line['fit_sample_ids'] == [r['sample_id'] for r in fit] and
                      line['known_target_cells'] == norm_statistics[condition]['known_target_cells'], tag + ':exposure:' + str(epoch))
                check(math.isfinite(line['loss']) and line['loss'] >= 0 and
                      line['loss'] == record['losses'][epoch - 1], tag + ':loss:' + str(epoch))
            equal(record['final_loss'], trace[-1]['loss'], tag + ':final_loss')
            check(len(record['development_center_cache']) == 36, tag + ':36_center_provenance')
            for i, (center, row) in enumerate(zip(record['development_center_cache'], dev)):
                equal(center, dict(clip_id=row['clip_id'], center_frame=row['center_frame'],
                      center_timestamp_us=row['center_timestamp_us'], probability_row=i,
                      array_fingerprint=metadata[row['clip_id']]['fingerprint']), tag + ':center_source:' + str(i))
        else:
            check(complete['status'] == 'COMPLETE_NEW_DEVELOPMENT_BASELINE', tag + ':complete')
            check(record['new_fullclip_old3_calls'] == 12 and record['scored_labeled_centers'] == 36 and
                  record['original_completed_predictions_recomputed'] == 0, tag + ':only_new_DEV_baseline')
            check(len(record['full_outputs']) == 12 and {v['clip_id'] for v in record['full_outputs']} ==
                  {r['clip_id'] for r in dev}, tag + ':full_clip_inventory')
            total_centers = 0
            for source in record['full_outputs']:
                pin(source['path'], source['sha256'])
                meta = metadata[source['clip_id']]
                with np.load(source['path'], allow_pickle=False) as z:
                    pp, ids, ts = z['probabilities'], z['frame_id'], z['timestamp_us']
                    check(z['targets'].tolist() == TARGETS and int(z['seed']) == seed and
                          str(z['array_fingerprint']) == source['array_fingerprint'] == meta['fingerprint'], tag + ':full_identity:' + source['clip_id'])
                    check(np.array_equal(ids, meta['frame_id']) and np.array_equal(ts, meta['timestamp_us']), tag + ':native_full_axis:' + source['clip_id'])
                    available = np.zeros((len(ids), 7), bool)
                    available[:, :3] = True
                    centers = [i for i in range(6, len(ids) - 6) if np.all(np.diff(ids[i-6:i+7]) == 1)]
                    available[centers, 3:] = True
                    check(pp.shape == available.shape and np.array_equal(z['posterior_available'], available) and
                          np.array_equal(np.isfinite(pp), available), tag + ':no_padded_or_fabricated_centers:' + source['clip_id'])
                    check(((pp[available] >= 0) & (pp[available] <= 1)).all(), tag + ':full_probability_range:' + source['clip_id'])
                    check(source['new4_centers'] == len(centers), tag + ':full_new4_centers:' + source['clip_id'])
                    total_centers += len(centers)
                    for i, row in enumerate(dev):
                        if row['clip_id'] == source['clip_id']:
                            index = int(np.flatnonzero(ids == row['center_frame'])[0])
                            check(int(ts[index]) == row['center_timestamp_us'] and np.array_equal(pp[index], p[i]), tag + ':saved_center_join:' + str(i))
            equal(record['new4_window_centers'], total_centers, tag + ':baseline_total_new4_centers')

    aggregates = {}
    for condition in CONDITIONS + ['protected_baseline']:
        mm = [audit_metrics[(condition, seed)] for seed in SEEDS]
        aggregates[condition] = dict(
            macro_supported_target_accuracy=distribution([r['macro_supported_target_accuracy'] for r in mm]),
            old_three_macro_accuracy=distribution([r['old_three_macro_accuracy'] for r in mm]),
            per_target=[dict(target=name, accuracy=distribution([r['per_target'][j]['accuracy'] for r in mm]),
                balanced_accuracy=distribution([r['per_target'][j]['balanced_accuracy'] for r in mm]))
                for j, name in enumerate(TARGETS)])
    equal(report['aggregates'], aggregates, 'independent_three_seed_aggregates')
    gain = aggregates[CONDITIONS[1]]['macro_supported_target_accuracy']['mean'] - aggregates[CONDITIONS[0]]['macro_supported_target_accuracy']['mean']
    old_delta = aggregates[CONDITIONS[1]]['old_three_macro_accuracy']['mean'] - aggregates['protected_baseline']['old_three_macro_accuracy']['mean']
    allowed = gain > 0 and old_delta >= 0
    equal(report['adoption'], dict(allowed=allowed, chosen_seed=42 if allowed else None,
        chosen_condition=CONDITIONS[1] if allowed else 'protected_baseline',
        expanded_minus_matched_macro_accuracy=gain, expanded_minus_baseline_old3_macro_accuracy=old_delta,
        model_path=str(U / 'trainruns/expanded_nine_people_42/model.pt') if allowed else None), 'frozen_adoption_rule')
    check(not allowed, 'recorded_baseline_retention_is_supported')
    # Source files can be written by other tasks; detect changes during this audit.
    for path, digest in list(pins.items()):
        check(sha(path) == digest, 'end_hash:' + path)
    return dict(report_sha256=report_sha, denominators=dict(distinct_DEV_centers=36, distinct_DEV_clips=12,
        distinct_DEV_people=3, known_cells_per_seed=194, unknown_cells_per_seed=58, possible_cells_per_seed=252,
        new_training_runs=6, saved_development_probability_arrays=9, epochs_per_training_run=160,
        baseline_full_outputs=36, repeated_seeds_are_not_independent_people=True),
        normalization_and_exposure=norm_statistics, run_metrics=compact_runs, aggregates=aggregates,
        adoption=dict(allowed=allowed, retained='protected_baseline', expanded_minus_matched=gain,
                      expanded_minus_baseline_old3=old_delta),
        protected_source_files=len(prep['protected_pins']), source_pins=pins,
        limits=[
            'No model load, forward, training, extraction or completed prediction replay. Hashes/exposures corroborate saved execution; optimizer internals are not reconstructed.',
            'Three seeds repeat the same36 development centers;194 known and58 unknown per seed are not independent people.',
            'ID11-13 are new to this head FIT but historical TCN development people; this is not a fresh final test.',
            '144 current-expression labels include48 exact prior labels and96 new same-Codex annotations; no independent human linguistic truth.',
            'Secondary review covered51 new centers (all36 DEV and15 FIT);45 new FIT centers remain single-annotator.',
            'Nine-person expansion also changes FIT180 to252, relevant-word clip share12/60 to36/84, FIT normalization and class weights; not isolated causal person-count evidence.',
            'R_index_only has no DEV positives, so balanced accuracy is null; L_open_four has only one DEV positive. Ordinary accuracy can conceal these support limits.',
            'Protected baseline is a different composite architecture/history; the matched and expanded seven-output heads share architecture and fixed160 updates.',
            'No educational correctness, full handshape correctness, live-demo performance or whole-sign generalization is established.'])


if __name__ == '__main__':
    if OUT.exists():
        raise FileExistsError('Preserve first saved verification: ' + str(OUT))
    result = {}
    try:
        result = audit()
    except BaseException:
        failure = U / 'review/SAVED_TRAINING_VERIFICATION_FIRST_FAILURE.txt'
        if not failure.exists():
            failure.write_text(traceback.format_exc(), encoding='utf8')
        failures.append({'check': 'verifier_exception', 'detail': traceback.format_exc()})
    result.update(status='PASS_INDEPENDENT_SAVED_TRAINING_AUDIT' if not failures else 'FAIL_PRESERVED',
        checked_at=datetime.now(timezone.utc).isoformat(), verifier=str(Path(__file__).resolve()),
        verifier_sha256=sha(__file__), checks=checks, failure_count=len(failures), failures=failures,
        forbidden_execution=dict(model_load=0, forward=0, training=0, DP=0, extraction=0))
    with OUT.open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: result.get(k) for k in ('status', 'checks', 'failure_count', 'adoption')}, ensure_ascii=False))
    raise SystemExit(0 if not failures else 1)
