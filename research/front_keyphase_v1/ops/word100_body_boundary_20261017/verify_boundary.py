"""Independent numeric review of the body fallback; no extraction or training."""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import numpy as np

sys.dont_write_bytecode = True
UNIT = Path(__file__).resolve().parent
ROOT = UNIT.parents[3]
sys.path.insert(0, str(UNIT.parents[1] / 'src'))
from wis_lab.core.observation import Observation, empty_frames


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def oracle_window(obs, rest=False):
    """Enumerate mathematical candidate windows, independent of production code."""
    if not obs.n_frames:
        return None
    xy = obs.xy_px.astype(np.float64)
    ts = obs.timestamp_us
    candidate_indices = np.flatnonzero(ts - ts[0] <= 1_000_000)
    candidates = [(a, b) for a in candidate_indices for b in candidate_indices
                  if b - a + 1 >= 8 and ts[b] - ts[a] >= 300_000]
    required = [11, 12, 13, 14, 33, 54] if rest else [11, 12]
    stationary = [11, 12, 33, 54] if rest else [11, 12]
    for a, b in sorted(candidates, key=lambda p: (p[1], p[0])):
        frames = np.arange(a, b + 1)
        if np.max(np.diff(ts[frames])) > 100_000:
            continue
        usable = obs.usable[np.ix_(frames, required)]
        observed = obs.observed[np.ix_(frames, required)]
        if not (usable & observed).all() or not np.isfinite(xy[np.ix_(frames, required)]).all():
            continue
        win = xy[frames]
        widths = np.sqrt(np.square(win[:, 11] - win[:, 12]).sum(axis=1))
        scale = float(np.median(widths))
        if not np.isfinite(scale) or scale <= 0:
            continue
        if rest:
            accepted = True
            for shoulder, elbow, wrist in [(11, 13, 33), (12, 14, 54)]:
                upper = win[:, elbow, 1] - win[:, shoulder, 1]
                lower = win[:, wrist, 1] - win[:, elbow, 1]
                horizontal = np.abs(win[:, wrist, 0] - win[:, elbow, 0])
                accepted &= bool(((upper >= .2 * scale) & (lower >= .2 * scale)
                                  & (horizontal <= .5 * lower)).all())
            if not accepted:
                continue
        positions = win[:, stationary]
        distances = np.sqrt(np.square(positions - np.median(positions, axis=0)).sum(axis=2))
        maximum = float(distances.max())
        if maximum > .05 * scale:
            continue
        center = np.median((win[:, 11] + win[:, 12]) / 2, axis=0)
        value = {'indices': frames.tolist(), 'scale': scale, 'center': center.tolist(),
                 'maximum': maximum}
        if rest:
            value['bottom'] = math.floor(min(np.median(win[:, 33, 1]),
                                            np.median(win[:, 54, 1])) - .02 * scale)
        return value
    return None


def assert_anchor(obs, actual):
    expected = oracle_window(obs)
    assert (actual['status'] == 'CALIBRATED') == (expected is not None), (actual, expected)
    if expected is not None:
        assert actual['selected_frame_indices'] == expected['indices']
        assert actual['selected_frame_ids'] == obs.frame_id[expected['indices']].tolist()
        assert actual['selected_timestamp_us'] == obs.timestamp_us[expected['indices']].tolist()
        np.testing.assert_allclose(actual['median_shoulder_center'], expected['center'], rtol=0, atol=1e-12)
        np.testing.assert_allclose(actual['shoulder_width'], expected['scale'], rtol=0, atol=1e-12)
        np.testing.assert_allclose(actual['maximum_stability_distance_pixels'], expected['maximum'], rtol=0, atol=1e-12)
    return expected


def assert_calibration(obs, width, height, coefficient, actual):
    rest = oracle_window(obs, rest=True)
    if rest is not None and 0 < rest['bottom'] <= height:
        assert actual['status'] == 'CALIBRATED'
        assert actual['method'] == 'preserved_stable_rest'
        assert actual['bottom'] == rest['bottom']
        assert actual['box_source_pixels'] == [0, 0, width, rest['bottom']]
        assert actual['selected_frame_indices'] == rest['indices']
        return 'preserved_stable_rest'
    anchor = oracle_window(obs)
    assert actual['method'] == 'train_median_body_fallback'
    if anchor is None:
        assert actual['status'] == 'NOT_CALIBRATED'
        assert 'bottom' not in actual
        return 'no_anchor'
    b = math.floor(anchor['center'][1] + float(coefficient['ratio']) * anchor['scale'])
    if not 0 < b < height:
        assert actual['status'] == 'NOT_CALIBRATED'
        assert actual['proposed_bottom'] == b
        assert 'bottom' not in actual
        return 'invalid_boundary'
    assert actual['status'] == 'CALIBRATED'
    assert actual['box_source_pixels'] == [0, 0, width, b]
    assert actual['bottom'] == b
    assert actual['selected_frame_indices'] == anchor['indices']
    return 'train_median_body_fallback'


def fixture(n=40, step=33334, hands=False):
    arrays = empty_frames(n)
    arrays['timestamp_us'] = np.arange(n, dtype=np.int64) * step
    arrays['frame_id'] = np.arange(n, dtype=np.int64) * 2 + 100
    coords = {11: (200, 100), 12: (300, 100)}
    if hands:
        coords.update({13: (200, 180), 14: (300, 180), 33: (200, 260), 54: (300, 260)})
    for idx, coord in coords.items():
        arrays['xy_px'][:, idx] = coord
        arrays['observed'][:, idx] = True
        arrays['usable'][:, idx] = True
    return Observation(**arrays)


def module():
    spec = importlib.util.spec_from_file_location('boundary_under_independent_review', UNIT / 'boundary.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def synthetic(mod):
    coefficient = {'schema': mod.SCHEMA, 'status': 'FROZEN_TRAIN_MEDIAN', 'ratio': 1.4}
    cases = []
    def check(name, obs, width=640, height=480, coef=None):
        obs.validate()
        saved = {k: v.copy() for k, v in vars(obs).items() if isinstance(v, np.ndarray)}
        anchor = assert_anchor(obs, mod.shoulder_anchor(obs))
        chosen = coefficient if coef is None else coef
        actual = mod.calibrate(obs, width, height, chosen)
        status = assert_calibration(obs, width, height, chosen, actual)
        for key, val in saved.items():
            np.testing.assert_array_equal(getattr(obs, key), val)
        cases.append({'name': name, 'branch': status,
                      'selected': None if anchor is None else anchor['indices'],
                      'bottom': actual.get('bottom')})
        return actual
    def raises(name, fn):
        try:
            fn()
        except (ValueError, AssertionError, KeyError, TypeError):
            cases.append({'name': name, 'branch': 'expected_rejection'})
            return
        raise AssertionError('Expected rejection: ' + name)

    check('empty', fixture(0))
    check('too_few_frames', fixture(7, 100000))
    check('minimum_8_frames_valid_duration', fixture(8, 50000))
    check('duration_below_300ms', fixture(10, 33333))
    check('duration_above_300ms', fixture(10, 33334))
    check('all_hands_NaN_shoulder_anchor_valid', fixture())
    check('original_stable_rest_preserved', fixture(hands=True))
    check('stable_rest_ignores_unused_coefficient', fixture(hands=True), coef={})
    a = fixture(); a.timestamp_us += 1_000_000_000_000
    check('relative_PTS_large_offset', a)
    a = fixture(); a.frame_id += 72391
    check('noncontiguous_frame_ID', a)
    a = fixture(); a.observed[:3, 11] = False; a.usable[:3, 11] = False; a.xy_px[:3, 11] = np.nan
    check('initial_shoulder_missing_earliest_later_window', a)
    a = fixture(); a.usable[3, 11] = False
    check('observed_but_unusable_shoulder', a)
    a = fixture(); a.xy_px[3, 11] = np.nan
    check('observed_NaN_shoulder_rejected_by_anchor', a)
    a = fixture(12, 100000)
    check('max_gap_exact_allowed', a)
    a = fixture(12, 100001)
    check('max_gap_above_limit', a)
    a = fixture(); a.xy_px[:, 12] = a.xy_px[:, 11]
    check('zero_width', a)
    a = fixture(); a.xy_px[0, [11, 12], 1] += 5
    check('exact_5percent_shoulder_motion_allowed', a)
    a = fixture(10); a.xy_px[0, [11, 12], 1] += 5.001
    check('above_5percent_shoulder_motion_rejected', a)
    a = fixture(40); a.usable[:31, [11, 12]] = False
    check('valid_shoulders_only_after_horizon', a)
    a = fixture(40, 50000); a.usable[:13, [11, 12]] = False
    check('window_ends_exactly_at_1second', a)
    a = fixture(40, 50001); a.usable[:13, [11, 12]] = False
    check('window_would_end_after_1second', a)
    a = fixture(hands=True); a.xy_px[:, 33, 1] = np.arange(40) * 12 + 90
    one = check('fast_hand_motion_same_shoulder_fallback', a)
    a.xy_px[:, 33, 0] = 700; a.xy_px[:, 33, 1] = -900
    two = check('arbitrary_hand_coordinates_same_fallback', a)
    assert one['bottom'] == two['bottom'] == 240
    a = fixture(); a.xy_px[:, [11, 12]] *= 2
    a.xy_px[:, [11, 12]] += np.array([20, 30])
    three = check('uniform_scale_translation', a, width=1500, height=1500)
    assert three['bottom'] == 2 * 240 + 30
    check('boundary_exact_source_height_explicit_failure', fixture(), height=240)
    check('boundary_beyond_source_explicit_failure', fixture(), height=239)
    a = fixture(); a.xy_px[:, [11, 12], 1] = -200
    check('boundary_above_source_explicit_failure', a)
    a = fixture(); a.sidecar['roi_mask'] = {'anything': True}
    raises('masked_input_forbidden', lambda: mod.calibrate(a, 640, 480, coefficient))
    a = fixture(); a.timestamp_us[3] = a.timestamp_us[2]
    raises('duplicate_timestamps_forbidden', lambda: mod.calibrate(a, 640, 480, coefficient))
    for name, k in [('nan', float('nan')), ('infinity', float('inf')), ('zero', 0), ('negative', -1)]:
        raises('invalid_ratio_' + name, lambda k=k: mod.calibrate(fixture(), 640, 480, {**coefficient, 'ratio': k}))
    raises('unfrozen_coefficient', lambda: mod.calibrate(fixture(), 640, 480, {**coefficient, 'status': 'DRAFT'}))
    raises('wrong_coefficient_schema', lambda: mod.calibrate(fixture(), 640, 480, {**coefficient, 'schema': 'other'}))
    raises('invalid_source_dimensions', lambda: mod.calibrate(fixture(), True, 480, coefficient))
    return {'status': 'PASS_SYNTHETIC_INDEPENDENT_NUMERIC_NOT_DATASET_READINESS',
            'cases': len(cases), 'details': cases,
            'scope_note': 'Hybrid branch selection still depends on old hand-based calibration; fallback numeric formula ignores hand positions. Offline source framing only.',
            'production_code_sha256': sha(UNIT / 'boundary.py'), 'verifier_sha256': sha(__file__),
            'new_detector_runs': 0, 'training_runs': 0, 'classifier_inference_runs': 0}


def verify_fit(mod, coefficient_path):
    coefficient_path = Path(coefficient_path)
    before = sha(coefficient_path)
    coef = read(coefficient_path)
    assert coef['schema'] == mod.SCHEMA and coef['status'] == 'FROZEN_TRAIN_MEDIAN'
    manifest_path = UNIT.parent / 'word100_training_20261013/MANIFEST.json'
    manifest = read(manifest_path)
    train = {r['clip_id']: r for r in manifest if r['role'] == 'train_candidate'}
    dev = {r['clip_id'] for r in manifest if r['role'] == 'development_only'}
    assert len(train) == 899 and len(dev) == 300 and not set(train) & dev
    assert {r['signer_id'] for r in train.values()} == set(range(1, 10))
    fit_rows = {r['clip_id']: r for r in coef['population']}
    excluded = {r['clip_id']: r for r in coef['excluded_fit_rows']}
    assert len(fit_rows) == len(coef['population']) == 889
    assert len(excluded) == len(coef['excluded_fit_rows']) == 10
    assert not set(fit_rows) & set(excluded)
    assert set(fit_rows) | set(excluded) == set(train)
    legacy = Path('W:/WordsInSilence/data/AIHub_103/word100_inputs_20261013')
    ratios = []
    fallback = []
    max_ratio_difference = 0.0
    for cid, row in train.items():
        record = read(legacy / 'records' / (cid + '.json'))
        obs = Observation.load(record['observation'])
        anchor = oracle_window(obs)
        assert anchor is not None, cid
        old = oracle_window(obs, rest=True)
        expected_row = fit_rows.get(cid, excluded.get(cid))
        assert_anchor(obs, expected_row['shoulder_anchor'])
        if old is None:
            assert cid in excluded, cid
            actual = mod.calibrate(obs, int(obs.sidecar['width']), int(obs.sidecar['height']), coef)
            assert_calibration(obs, int(obs.sidecar['width']), int(obs.sidecar['height']), coef, actual)
            fallback.append({'clip_id': cid, 'bottom': actual.get('bottom'), 'status': actual['status']})
        else:
            assert cid in fit_rows, cid
            r = fit_rows[cid]
            assert r['old_bottom'] == old['bottom']
            assert r['role'] == 'train_candidate' and r['signer_id'] == row['signer_id']
            assert Path(r['observation']) == Path(record['observation'])
            assert r['observation_sha256'] == sha(record['observation'])
            ratio = (old['bottom'] - anchor['center'][1]) / anchor['scale']
            max_ratio_difference = max(max_ratio_difference, abs(ratio - r['ratio']))
            np.testing.assert_allclose(ratio, r['ratio'], rtol=0, atol=1e-12)
            ratios.append(ratio)
    independent_median = float(np.median(ratios))
    np.testing.assert_allclose(independent_median, coef['ratio'], rtol=0, atol=1e-12)
    for path, expected in coef['source_pins'].items():
        assert sha(path) == expected, path
    assert sha(coefficient_path) == before
    return {'status': 'PASS_TRAIN_ONLY_COEFFICIENT_INDEPENDENT_NOT_SEMANTIC_READINESS',
            'source_train_rows': len(train), 'fit_rows': len(ratios), 'excluded_from_fit_only': len(excluded),
            'shoulder_anchor_available_train': len(train), 'independent_ratio': independent_median,
            'reported_ratio': coef['ratio'], 'maximum_per_row_ratio_difference': max_ratio_difference,
            'fallback_rows': fallback, 'source_pins_unchanged': len(coef['source_pins']),
            'coefficient_path': str(coefficient_path), 'coefficient_sha256': before,
            'production_code_sha256': sha(UNIT / 'boundary.py'), 'verifier_sha256': sha(__file__),
            'dev_arrays_read': 0, 'new_detector_runs': 0, 'training_runs': 0, 'classifier_inference_runs': 0}


def input_oracle_module():
    path = UNIT.parent / 'word100_roi_training_20261017/verify_inputs.py'
    spec = importlib.util.spec_from_file_location('independent_input_oracle', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def verify_preflight(coefficient_path):
    coefficient_path = Path(coefficient_path)
    coefficient_hash = sha(coefficient_path)
    coef = read(coefficient_path)
    saved_fit = read(UNIT / 'BOUNDARY_FIT_INDEPENDENT.json')
    assert saved_fit['coefficient_sha256'] == coefficient_hash
    context_path = UNIT / 'PREP_CONTEXT.json'
    context = read(context_path)
    saved = read(UNIT / 'LEGACY_PREFLIGHT.json')
    assert saved['context_sha256'] == sha(context_path)
    io = input_oracle_module()
    rows = context['ordered_rows']
    stats = io.manifest_audit(rows)
    manifest = {r['clip_id']: r for r in read(context['manifest_path'])}
    summaries = {r['clip_id']: r for r in saved['rows']}
    assert len(summaries) == len(saved['rows']) == len(rows) == 1199
    for path, expected in context['pins'].items():
        assert sha(path) == expected, path
    preserved = fallback = flagged = feature_rows = 0
    independently_constructed_values = 0
    details = []
    for row in rows:
        cid = row['clip_id']; summary = summaries[cid]
        assert (row['role'], row['signer_id'], row['label_index']) == (
            manifest[cid]['role'], manifest[cid]['signer_id'], manifest[cid]['label_index'])
        record = read(row['legacy_record_path'])
        obs = Observation.load(record['observation'])
        width, height = int(obs.sidecar['width']), int(obs.sidecar['height'])
        old = oracle_window(obs, rest=True)
        if old is not None and 0 < old['bottom'] <= height:
            method = 'preserved_stable_rest'; bottom = old['bottom']; preserved += 1
        else:
            anchor = oracle_window(obs); assert anchor is not None, cid
            method = 'train_median_body_fallback'
            bottom = math.floor(anchor['center'][1] + coef['ratio'] * anchor['scale'])
            assert 0 < bottom < height, cid
            fallback += 1
        assert summary['calibration'] == 'CALIBRATED' and summary['method'] == method
        assert summary['bottom'] == bottom
        native = io.load(record['observation'])
        masked = io.mask(native, [0, 0, width, bottom])
        arrays, info = io.feature_oracle(masked, row)
        assert info['status'] == 'OK' and summary['feature_info']['reason'] is None, (cid, info)
        for key in ('whole_span', 'context_span'):
            if key in info:
                assert summary['feature_info'][key] == info[key], (cid, key)
        feature_rows += 1
        independently_constructed_values += sum(v.size for k, v in arrays.items() if k.startswith('x_'))
        flags = io.loss_flags(native, masked, row, 'native_to_mask')
        actual_flags = summary['geometry_flags']
        assert {(f['comparison'], f['side']) for f in flags} == {(f['comparison'], f['side']) for f in actual_flags}, cid
        for f in flags:
            actual = next(a for a in actual_flags if a['side'] == f['side'])
            np.testing.assert_allclose(actual['palm5_drop_pp'], f['palm5_drop_pp'], rtol=0, atol=1e-12)
            start = 33 if f['side'] == 'left' else 54
            ts = native['timestamp_us']; a, b = row['intervals'][0]
            core = (ts >= round(a * 1e6)) & (ts <= round(b * 1e6))
            lost = core & native['usable'][:, start:start+21].any(1) & ~masked['usable'][:, start:start+21].any(1)
            ids = np.flatnonzero(lost)
            assert actual['newly_absent_frame_ids'] == native['frame_id'][ids].tolist()
            runs = np.split(ids, np.flatnonzero(np.diff(ids) != 1) + 1) if len(ids) else []
            step = float(np.median(np.diff(ts)))
            maxrun = max(((ts[g[-1]] - ts[g[0]] + step) / 1e6 for g in runs), default=0.0)
            np.testing.assert_allclose(actual['newly_absent_run_seconds'], maxrun, rtol=0, atol=1e-12)
        flagged += bool(flags)
        if method == 'train_median_body_fallback':
            details.append({'clip_id': cid, 'role': row['role'], 'bottom': bottom,
                            'whole_span': info['whole_span'], 'context_span': info.get('context_span'),
                            'geometry_flag_count': len(flags)})
    assert not saved['calibration_failures'] and not saved['train_feature_failures']
    assert saved['fallback_rows'] == fallback == 12
    assert preserved == 1187 and feature_rows == 1199
    assert sha(coefficient_path) == coefficient_hash
    return {'status': 'PASS_LEGACY_GEOMETRY_SPANS_AND_FLAGS_NOT_NEW_DETECTOR_READINESS',
            'manifest': stats, 'preserved_boxes': preserved, 'fallback_boxes': fallback,
            'valid_masked_feature_rows': feature_rows, 'geometry_flagged_train_clips': flagged,
            'independent_feature_values_constructed_for_span_gate': independently_constructed_values,
            'feature_value_limit': 'Preflight stores spans/gates, not channel arrays; this is not a stored-array value equality claim.',
            'fallback_rows': details, 'context_pins_unchanged': len(context['pins']),
            'context_sha256': sha(context_path), 'preflight_sha256': sha(UNIT / 'LEGACY_PREFLIGHT.json'),
            'coefficient_sha256': coefficient_hash, 'verifier_sha256': sha(__file__),
            'dev_evaluation_only_rows': 300, 'dev_used_in_fit': 0,
            'new_detector_runs': 0, 'training_runs': 0, 'classifier_inference_runs': 0}


def verify_records(mod, coefficient_path):
    coefficient_path = Path(coefficient_path)
    coefficient_hash = sha(coefficient_path)
    coef = read(coefficient_path)
    context = read(UNIT / 'PREP_CONTEXT.json')
    manifest = {r['clip_id']: r for r in context['ordered_rows']}
    row_results = []
    for record_path in sorted((UNIT / 'data/records').glob('*.json')):
        rec = read(record_path); cid = rec['clip_id']
        native = Observation.load(rec['native_capture']['candidate'])
        old_rec = read(manifest[cid]['legacy_record_path'])
        old_sidecar = read(Path(old_rec['observation']).with_suffix('.sidecar.json'))
        actual = rec['calibration']
        branch = assert_calibration(native, int(old_sidecar['width']), int(old_sidecar['height']), coef, actual)
        if branch == 'preserved_stable_rest':
            # Independent geometry is checked above; also verify wrapper preserves every old field.
            old = mod.stable.calibrate_rest(native, int(old_sidecar['width']), int(old_sidecar['height']))
            for key, value in old.items():
                if key != 'schema':
                    assert actual[key] == value, (cid, key)
            assert actual['original_schema'] == old['schema']
        else:
            assert_anchor(native, actual)
        assert rec['sensor_capture']['box'] == actual['box_source_pixels']
        tokens = [rec[name].get('detector_instance') for name in ('native_capture', 'sensor_capture')]
        if all(tokens):
            assert tokens[0] != tokens[1], 'Shared native/crop detector token: ' + cid
        row_results.append({'clip_id': cid, 'branch': branch, 'bottom': actual.get('bottom'),
                            'record_sha256': sha(record_path)})
    io = input_oracle_module()
    io.U = UNIT  # Only audit/output location changes; independent NumPy oracle remains frozen.
    channels_path = UNIT / 'BOUNDARY_RECORDS_CHANNELS_INDEPENDENT.json'
    io.records(channels_path)
    channels = read(channels_path)
    assert channels['prepared_records'] == len(row_results)
    assert sha(coefficient_path) == coefficient_hash
    return {'status': 'PASS_PREPARED_RECORDS_NUMERIC_PENDING_SEMANTIC_REVIEW',
            'prepared_records': len(row_results), 'requested_records': 1199,
            'boundary_methods': {name: sum(r['branch'] == name for r in row_results)
                                 for name in ('preserved_stable_rest', 'train_median_body_fallback')},
            'matched_feature_values': channels['matched_feature_values'],
            'hidden_points_perturbed': channels['hidden_points_perturbed'],
            'context_pins_unchanged': channels['context_pins_unchanged'],
            'channels_audit_path': str(channels_path), 'channels_audit_sha256': sha(channels_path),
            'rows': row_results, 'verifier_sha256': sha(__file__),
            'independent_oracle_path': str(Path(io.__file__)), 'independent_oracle_sha256': sha(io.__file__),
            'new_detector_runs': 0, 'training_runs': 0, 'classifier_inference_runs': 0,
            'training_allowed_by_this_audit': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['synthetic', 'fit', 'preflight', 'records'], default='synthetic')
    parser.add_argument('--coefficient', default=str(UNIT / 'design_data_COEFFICIENT.json'))
    parser.add_argument('--output')
    args = parser.parse_args()
    output = Path(args.output) if args.output else UNIT / ('BOUNDARY_' + args.mode.upper() + '_INDEPENDENT.json')
    if output.exists():
        raise FileExistsError(output)
    try:
        if args.mode == 'synthetic':
            result = synthetic(module())
        elif args.mode == 'fit':
            result = verify_fit(module(), args.coefficient)
        elif args.mode == 'preflight':
            result = verify_preflight(args.coefficient)
        else:
            result = verify_records(module(), args.coefficient)
    except Exception as exc:
        result = {'status': 'FAILED', 'error_type': type(exc).__name__, 'error': str(exc)}
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        raise
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('details', 'rows')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
