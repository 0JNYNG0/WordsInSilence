"""Independent read-only input audit. Does not run a detector or classifier.

The numeric oracle below never imports the preparation ROI, active-span or
normalisation implementations. Reports are new audit artifacts in this unit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
U = Path(__file__).resolve().parent
OPS = U.parent
BODY = [0, 11, 12, 13, 14, 15, 16, 23, 24]
COMMON = BODY + list(range(33, 75))


def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def same(a, b):
    np.testing.assert_array_equal(a, b)


def validate(o):
    xy, ob, us, ts = (o[k] for k in ('xy_px', 'observed', 'usable', 'timestamp_us'))
    n = len(ts)
    assert xy.shape == (n, 75, 2) and ob.shape == us.shape == (n, 75)
    assert np.all(np.diff(ts) > 0) and len(o['frame_id']) == n
    assert not (us & ~ob).any()
    assert np.isfinite(xy[ob]).all() and np.isnan(xy[~ob]).all()
    assert np.isnan(o['conf'][~ob]).all()


def mask(o, box):
    q = {k: v.copy() for k, v in o.items()}
    x0, y0, x1, y1 = box
    xy = o['xy_px']
    visible = (np.isfinite(xy).all(2) & (xy[:, :, 0] >= x0)
               & (xy[:, :, 0] < x1) & (xy[:, :, 1] >= y0) & (xy[:, :, 1] < y1))
    q['observed'] = o['observed'] & visible
    q['usable'] = o['usable'] & q['observed']
    q['xy_px'][~q['observed']] = np.nan
    q['conf'][~q['observed']] = np.nan
    for side, start in enumerate((33, 54)):
        q['hand_score'][~q['observed'][:, start:start + 21].any(1), side] = np.nan
    validate(q)
    return q


def channels(o, lo, hi):
    """Direct 51-point numeric oracle, including wrist-dependent local masks."""
    xy = o['xy_px'][lo:hi + 1, COMMON].astype(np.float32).copy()
    valid = o['usable'][lo:hi + 1, COMMON].astype(bool)
    xy[~valid] = 0
    ts = o['timestamp_us'][lo:hi + 1]
    shoulder_ok = valid[:, 1] & valid[:, 2]
    if not shoulder_ok.any():
        raise ValueError('NO_USABLE_BODY_SCALE')
    bs = float(np.median(np.linalg.norm(xy[shoulder_ok, 1] - xy[shoulder_ok, 2], axis=1)))
    if not np.isfinite(bs) or bs <= 1e-6:
        raise ValueError('DEGENERATE_BODY_SCALE')
    n = len(ts)
    center = (xy[:, 1] + xy[:, 2]) / 2.0
    bmask = valid & shoulder_ok[:, None]
    body = np.zeros((n, 51, 2), np.float32)
    body[bmask] = ((xy - center[:, None, :]) / bs)[bmask]
    local = np.zeros((n, 42, 2), np.float32)
    hmask = np.zeros((n, 42), bool)
    for si, start in enumerate((9, 30)):
        lengths = []
        for f in range(n):
            if valid[f, start]:
                for j in (5, 9, 13, 17):
                    if valid[f, start + j]:
                        lengths.append(float(np.linalg.norm(xy[f, start + j] - xy[f, start])))
        hs = float(np.median(lengths)) if lengths else float('nan')
        if not np.isfinite(hs) or hs <= 1e-6:
            continue
        good = valid[:, start:start + 21] & valid[:, start, None]
        rel = (xy[:, start:start + 21] - xy[:, start, None, :]) / hs
        local[:, 21 * si:21 * (si + 1)][good] = rel[good]
        hmask[:, 21 * si:21 * (si + 1)] = good
    dt = np.zeros((n, 1), np.float32)
    dt[1:, 0] = np.diff(ts).astype(np.float32) / 1e6
    x = np.concatenate((body.reshape(n, 102), local.reshape(n, 84), bmask, hmask, dt), 1).astype(np.float32)
    assert x.shape == (n, 280) and np.isfinite(x).all()
    assert (x[:, :102].reshape(n, 51, 2)[~bmask] == 0).all()
    assert (x[:, 102:186].reshape(n, 42, 2)[~hmask] == 0).all()
    return x


def feature_oracle(o, row):
    validate(o)
    seen = o['usable'][:, 33:75].any(1)
    ix = np.flatnonzero(seen)
    if not len(ix):
        return {}, {'status': 'NO_USABLE_HAND_FRAME'}
    lo, hi = int(ix[0]), int(ix[-1])
    if hi - lo + 1 < 4 or seen[lo:hi + 1].sum() < 4:
        return {}, {'status': 'TOO_FEW_OBSERVED_HAND_FRAMES'}
    spans = {'whole': (lo, hi)}
    if row['role'] == 'train_candidate':
        assert len(row['intervals']) == 1
        a, b = row['intervals'][0]
        ts = o['timestamp_us']
        ids = np.flatnonzero((ts >= round((a - .30) * 1e6)) & (ts <= round((b + .30) * 1e6)))
        if not len(ids):
            return {}, {'status': 'NO_CONTEXT_INTERVAL'}
        cl, ch = max(lo, int(ids[0])), min(hi, int(ids[-1]))
        if ch < cl or seen[cl:ch + 1].sum() < 4:
            return {}, {'status': 'INSUFFICIENT_CONTEXT_HAND_FRAMES', 'context_span': [cl, ch]}
        spans['context'] = (cl, ch)
    out, meta = {}, {'status': 'OK'}
    try:
        for name, (a, b) in spans.items():
            out['x_' + name] = channels(o, a, b)
            out[name + '_frame_id'] = o['frame_id'][a:b + 1]
            out[name + '_ts'] = o['timestamp_us'][a:b + 1]
            meta[name + '_span'] = [a, b]
            meta[name + '_observed_frames'] = int(seen[a:b + 1].sum())
            meta[name + '_internal_absent_frames'] = int((~seen[a:b + 1]).sum())
    except ValueError as exc:
        return {}, {'status': str(exc)}
    return out, meta


def loss_flags(before, after, row, label):
    """Separate geometry and sensor loss; flags are not semantic diagnoses."""
    same(before['timestamp_us'], after['timestamp_us'])
    same(before['frame_id'], after['frame_id'])
    if row['role'] != 'train_candidate':
        return []
    ts = before['timestamp_us']
    a, b = row['intervals'][0]
    core = (ts >= round(a * 1e6)) & (ts <= round(b * 1e6))
    assert core.any()
    tail = int(np.median(np.diff(ts)))
    flags = []
    for side, start in [('left', 33), ('right', 54)]:
        av = before['usable'][:, start:start + 21]
        bv = after['usable'][:, start:start + 21]
        lost = core & av.any(1) & ~bv.any(1)
        runs, begin = [], None
        for i, yes in enumerate(np.r_[lost, False]):
            if yes and begin is None:
                begin = i
            if not yes and begin is not None:
                duration = int(ts[i - 1] - ts[begin] + tail)
                runs.append({'start_frame_index': begin, 'end_frame_index': i - 1,
                             'duration_us': duration})
                begin = None
        drop = float((av[core][:, [0, 5, 9, 13, 17]].all(1).mean()
                      - bv[core][:, [0, 5, 9, 13, 17]].all(1).mean()) * 100)
        long = [r for r in runs if r['duration_us'] >= 300000]
        if long or drop >= 10:
            flags.append({'comparison': label, 'side': side, 'palm5_drop_pp': drop,
                          'new_absence_runs_ge_300ms': long, 'requires_visual_review': True})
    return flags


def manifest_audit(rows):
    assert len(rows) == 1199 and len({r['clip_id'] for r in rows}) == 1199
    assert Counter(r['role'] for r in rows) == {'train_candidate': 899, 'development_only': 300}
    assert len({r['label_index'] for r in rows}) == 100
    for r in rows:
        allowed = set(range(1, 10)) if r['role'] == 'train_candidate' else {11, 12, 13}
        assert r['signer_id'] in allowed
        assert len(r['video_entries']) == 1
        v = r['video_entries'][0]
        assert v['filename_signer_id'] == r['signer_id'] and v['word_code'] == r['word_code']
    return {'total': 1199, 'training': 899, 'development': 300, 'classes': 100}


def early_cache(output):
    prior = OPS / 'word100_stable_rest_roi_20261017'
    plan = read(prior / 'PLAN.json')
    manifest = read(OPS / 'word100_training_20261013/MANIFEST.json')
    stats = manifest_audit(manifest)
    byid = {r['clip_id']: r for r in manifest}
    rows, values, pts = [], 0, 0
    for it in plan['items']:
        cid = it['clip_id']; row = byid[cid]
        assert row['role'] == 'train_candidate'
        native = load(it['cached']['native']['observation'])
        masked = mask(native, it['stable_box'])
        saved = load(prior / 'outputs' / cid / 'stable_full_raw/model_input.npz')
        for key in masked:
            same(masked[key], saved[key])
        rec = read(prior / 'sensor_crop' / cid / 'SUCCESS.json')
        sensor_source = load(rec['observation'])
        sensor = mask(sensor_source, it['stable_box'])
        saved = load(prior / 'outputs' / cid / 'stable_sensor_raw/model_input.npz')
        for key in sensor:
            same(sensor[key], saved[key])
        views = {}
        for name, o in [('native', native), ('masked', masked), ('sensor', sensor)]:
            arrays, info = feature_oracle(o, row)
            values += sum(v.size for k, v in arrays.items() if k.startswith('x_'))
            views[name] = info
        # Hidden full-frame coordinate perturbation stays outside the already
        # fixed offline ROI. Recalibration is intentionally not part of this test.
        hidden = native['observed'] & ~masked['observed']
        changed = {k: v.copy() for k, v in native.items()}
        changed['xy_px'][hidden] = [it['stable_box'][2] + 1000, it['stable_box'][3] + 1000]
        changed['conf'][hidden] = .99
        changed_mask = mask(changed, it['stable_box'])
        expected, _ = feature_oracle(masked, row)
        got, _ = feature_oracle(changed_mask, row)
        for key in expected:
            same(expected[key], got[key])
        pts += int(hidden.sum())
        flags = loss_flags(native, masked, row, 'native_to_masked_geometry')
        flags += loss_flags(masked, sensor, row, 'masked_to_sensor_detector')
        rows.append({'clip_id': cid, 'label': row['canonical_label'], 'box': it['stable_box'],
                     'views': views, 'review_flags': flags})
    result = {'status': 'EARLY_CACHE_AUDIT_COMPLETE_NOT_TRAINING_APPROVAL',
              'manifest': stats, 'clips': len(rows), 'independently_constructed_channel_values': values,
              'hidden_points_perturbed': pts, 'gate_failures': [r['clip_id'] for r in rows if any(v['status'] != 'OK' for v in r['views'].values())],
              'flagged_clips': len({r['clip_id'] for r in rows if r['review_flags']}),
              'geometry_flagged_clips': sum(any(f['comparison'] == 'native_to_masked_geometry' for f in r['review_flags']) for r in rows),
              'sensor_flagged_clips': sum(any(f['comparison'] == 'masked_to_sensor_detector' for f in r['review_flags']) for r in rows),
              'run_duration_convention': 'lastPTS-firstPTS+clip median frame interval',
              'rows': rows, 'detector_runs': 0, 'training_runs': 0, 'rest_guard': 'OFF',
              'auditor': 'roi_missing_protocol independent NumPy oracle'}
    write_new(output, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, ensure_ascii=False))


def core_details(o, core):
    result = {}
    for side, start in [('left', 33), ('right', 54)]:
        v = o['usable'][core, start:start + 21]
        result[side] = {'core_frames': len(v), 'any_frames': int(v.any(1).sum()),
                        'palm5_frames': int(v[:, [0, 5, 9, 13, 17]].all(1).sum()),
                        'all21_frames': int(v.all(1).sum()), 'point_frames': int(v.sum())}
    return result


def prep_contract(output):
    """Call only preparation's pure feature/flag functions, never its run/freeze."""
    spec = importlib.util.spec_from_file_location('input_prep_under_test', U / 'prep.py')
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    prior = OPS / 'word100_stable_rest_roi_20261017'
    plan = read(prior / 'PLAN.json')
    byid = {r['clip_id']: r for r in read(OPS / 'word100_training_20261013/MANIFEST.json')}
    rows = []; values = 0
    for it in plan['items']:
        cid = it['clip_id']; row = byid[cid]
        paths = {'native': it['cached']['native']['observation'],
                 'masked': prior / 'outputs' / cid / 'stable_full_raw/model_input.npz',
                 'sensor': prior / 'outputs' / cid / 'stable_sensor_raw/model_input.npz'}
        arrays = {k: load(p) for k, p in paths.items()}
        objects = {k: prep.Observation.load(str(p)) for k, p in paths.items()}
        for name, obj in objects.items():
            expected, meta = feature_oracle(arrays[name], row)
            got, info = prep.features(obj, row)
            assert meta['status'] == 'OK' and info['reason'] is None
            assert set(got) == set(expected)
            for key in got:
                same(got[key], expected[key])
                if key.startswith('x_'):
                    values += got[key].size
            for key in ('whole_span', 'context_span'):
                assert meta[key] == info[key]
        expected_flags = loss_flags(arrays['native'], arrays['masked'], row, 'native_to_mask')
        expected_flags += loss_flags(arrays['masked'], arrays['sensor'], row, 'mask_to_sensor')
        got_flags = prep.visibility_flags(row, objects['native'], objects['masked'], objects['sensor'])
        assert {(f['comparison'], f['side']) for f in got_flags} == {(f['comparison'], f['side']) for f in expected_flags}
        for expected in expected_flags:
            actual = next(f for f in got_flags if (f['comparison'], f['side']) == (expected['comparison'], expected['side']))
            assert abs(actual['palm5_drop_pp'] - expected['palm5_drop_pp']) < 1e-12
        ts = arrays['native']['timestamp_us']; a, b = row['intervals'][0]
        core = (ts >= round(a * 1e6)) & (ts <= round(b * 1e6))
        evidence = []
        for flag in expected_flags:
            if flag['comparison'] != 'mask_to_sensor':
                continue
            s = 33 if flag['side'] == 'left' else 54
            before, after = arrays['masked'], arrays['sensor']
            lost = core & before['usable'][:, s:s + 21].any(1) & ~after['usable'][:, s:s + 21].any(1)
            for frame in np.flatnonzero(lost):
                valid = before['usable'][frame, s:s + 21]
                xy = before['xy_px'][frame, s:s + 21][valid]
                evidence.append({'frame_id': int(before['frame_id'][frame]), 'timestamp_us': int(ts[frame]),
                                 'side': flag['side'], 'native_assigned_points_inside_same_roi': int(valid.sum()),
                                 'native_source_xy_min': xy.min(0).tolist(), 'native_source_xy_max': xy.max(0).tolist(),
                                 'sensor_usable_hand_points': 0})
        rows.append({'clip_id': cid, 'label': row['canonical_label'],
                     'core_interval_seconds': [a, b], 'core_frames': int(core.sum()),
                     'metrics': {k: core_details(o, core) for k, o in arrays.items()},
                     'flags': got_flags, 'source_roi_inside_missing_evidence': evidence})
    # This audit does not decide linguistic necessity. Parent visually confirmed
    # existing detector losses separately; numeric source coordinates are evidence,
    # not hand-coordinate ground truth or a linguistic annotation.
    report = {'status': 'PASS_NUMERIC_CONTRACT_SEMANTIC_READINESS_PENDING', 'clips': len(rows),
              'matched_feature_values': values, 'pure_prep_functions_checked': ['features', 'visibility_flags'],
              'prep_sha256': sha(U / 'prep.py'), 'rows': rows,
              'known_required_manual_review_cases': ['NIA_SL_WORD1545_REAL01_F', 'NIA_SL_WORD2348_REAL02_F'],
              'detector_runs': 0, 'classifier_runs': 0, 'training_runs': 0,
              'semantics': 'Existing cached detector loss, not a newly introduced loss. Numerical flags require visual/semantic review before training.'}
    write_new(output, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, ensure_ascii=False))


def baseline(output):
    pred_path = U / 'baseline_PREDICTIONS.json'
    pred = read(pred_path); result = read(U / 'baseline_RESULT.json')
    provenance = read(U / 'baseline_PROVENANCE.json')
    assert sha(pred_path) == result['new_prediction_sha256']
    old = {r['clip_id']: r for r in read(OPS.parent / 'runs/word100_tcn_v1/whole_context/PREDICTIONS.json')}
    manifest = {r['clip_id']: r for r in read(OPS / 'word100_training_20261013/MANIFEST.json') if r['role'] == 'development_only'}
    assert len(pred) == len(manifest) == 300 and {r['clip_id'] for r in pred} == set(manifest)
    correct = top5 = 0; ces = []; signers = {}
    for r in pred:
        cid = r['clip_id']; source = manifest[cid]
        assert r['signer_id'] == source['signer_id'] and r['label_index'] == source['label_index']
        l = np.asarray(r['logits'], dtype=np.float64)
        same(l, np.asarray(old[cid]['logits'], dtype=np.float64))
        assert l.shape == (100,) and np.isfinite(l).all()
        order = np.argsort(-l, kind='stable')
        y = source['label_index']; yes = int(order[0]) == y
        five = y in order[:5]
        assert r['prediction'] == int(order[0]) and r['correct'] == yes and r['top5_correct'] == five
        ce = float(np.log(np.exp(l - l.max()).sum()) + l.max() - l[y])
        assert abs(ce - r['cross_entropy']) < 1e-10
        correct += yes; top5 += five; ces.append(ce)
        s = signers.setdefault(str(r['signer_id']), {'correct': 0, 'total': 0})
        s['correct'] += int(yes); s['total'] += 1
    assert correct == result['correct'] == 290 and top5 == result['top5_correct'] == 299
    assert signers == result['by_signer'] and abs(float(np.mean(ces)) - result['mean_ce']) < 1e-10
    for p, h in provenance['source_pins'].items():
        assert sha(p) == h, p
    report = {'status': 'PASS_SAVED_BASELINE_INDEPENDENT_RECOUNT', 'total': 300,
              'top1_correct': correct, 'top5_correct': top5, 'by_signer': signers,
              'mean_ce': float(np.mean(ces)), 'logits_equal_previous_saved_results': True,
              'protected_files_unchanged': len(provenance['source_pins']),
              'new_inference': 0, 'training': 0, 'prediction_sha256': sha(pred_path)}
    write_new(output, report)
    print(json.dumps(report, ensure_ascii=False))


def records(output):
    """Audit existing versioned records only; never produce missing records."""
    context = read(U / 'PREP_CONTEXT.json'); ch = sha(U / 'PREP_CONTEXT.json')
    source = read(context['manifest_path']); manifest_audit(source)
    byid = {r['clip_id']: r for r in source}
    assert {r['clip_id'] for r in context['ordered_rows']} == set(byid)
    for row in context['ordered_rows']:
        for key, value in byid[row['clip_id']].items():
            assert row[key] == value, (row['clip_id'], key)
    expected_refs = {min((r for r in source if r['role'] == 'train_candidate' and r['label_index'] == label),
                         key=lambda r: (r['signer_id'], r['clip_id']))['clip_id'] for label in range(100)}
    assert set(context['fixed_reference_ids']) == expected_refs
    protected = 0
    for p, h in context['pins'].items():
        assert sha(p) == h, p
        protected += 1
    rows = []; values = points = 0
    for record_path in sorted((U / 'data/records').glob('*.json')):
        rec = read(record_path); cid = rec['clip_id']; row = byid[cid]
        assert rec['context_sha256'] == ch and rec['training'] == 0 and rec['rest_wrist_filter'] == 'OFF'
        for key in ('role', 'signer_id', 'label_index', 'word_code', 'canonical_label'):
            assert rec[key] == row[key]
        for p, h in rec['outputs'].items():
            assert sha(p) == h, p
        for capture in ('native_capture', 'sensor_capture'):
            for p, h in rec[capture]['pins'].items():
                assert sha(p) == h, p
        cal = rec['calibration']; assert cal == read(rec['calibration_path'])
        assert cal['status'] == 'CALIBRATED'
        box = cal['box_source_pixels']; assert box[0:2] == [0, 0]
        original = load(rec['native_capture']['candidate'])
        cropped = load(rec['sensor_capture']['candidate'])
        same(original['timestamp_us'], cropped['timestamp_us'])
        same(original['frame_id'], cropped['frame_id'])
        assert box == rec['sensor_capture']['box']
        raw_sources = {'native': original, 'masked': mask(original, box), 'sensor': mask(cropped, box),
                       'm0_sensor': load(rec['sensor_capture']['baseline'])}
        outcomes = {}
        for name, view in rec['views'].items():
            actual_obs = load(view['observation']); validate(actual_obs)
            same(actual_obs['timestamp_us'], original['timestamp_us'])
            same(actual_obs['frame_id'], original['frame_id'])
            if name in raw_sources:
                for key in raw_sources[name]:
                    same(raw_sources[name][key], actual_obs[key])
            expected, info = feature_oracle(actual_obs, row)
            if info['status'] != 'OK':
                assert view['channel_file'] is None and view['feature_info']['reason']
                assert row['role'] == 'development_only' or name == 'm0_sensor'
            else:
                assert view['channel_file']
                actual = load(view['channel_file'])
                assert set(actual) == set(expected)
                for key in expected:
                    same(expected[key], actual[key])
                    if key.startswith('x_'):
                        values += actual[key].size
                for key in ('whole_span', 'context_span'):
                    if key in info:
                        assert view['feature_info'][key] == info[key]
            outcomes[name] = info
        hidden = original['observed'] & ~raw_sources['masked']['observed']
        changed = {k: v.copy() for k, v in original.items()}
        changed['xy_px'][hidden] = [box[2] + 500, box[3] + 500]
        changed['conf'][hidden] = .95
        exp, _ = feature_oracle(raw_sources['masked'], row)
        altered, _ = feature_oracle(mask(changed, box), row)
        for key in exp:
            same(exp[key], altered[key])
        points += int(hidden.sum())
        flags = loss_flags(original, raw_sources['masked'], row, 'native_to_mask')
        flags += loss_flags(raw_sources['masked'], raw_sources['sensor'], row, 'mask_to_sensor')
        assert {(f['comparison'], f['side']) for f in flags} == {(f['comparison'], f['side']) for f in rec['visual_flags']}
        for flag in flags:
            got = next(f for f in rec['visual_flags'] if (f['comparison'], f['side']) == (flag['comparison'], flag['side']))
            assert abs(got['palm5_drop_pp'] - flag['palm5_drop_pp']) < 1e-12
        rows.append({'clip_id': cid, 'role': row['role'], 'outcomes': outcomes,
                     'visual_flag_count': len(flags), 'record_sha256': sha(record_path)})
    report = {'status': 'PASS_EXISTING_RECORDS_PENDING_VISUAL_REVIEW', 'prepared_records': len(rows),
              'requested_records': 1199, 'all_requested_prepared': len(rows) == 1199,
              'matched_feature_values': values, 'hidden_points_perturbed': points,
              'context_pins_unchanged': protected, 'rest_guard': 'OFF',
              'requires_separate_semantic_review': True, 'training_authorized_by_audit': False,
              'rows': rows, 'detector_runs': 0, 'classifier_runs': 0, 'training_runs': 0}
    write_new(output, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, ensure_ascii=False))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['early-cache', 'prep-contract', 'baseline', 'records'])
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    defaults = {'early-cache': 'EARLY_CACHE_INDEPENDENT.json', 'prep-contract': 'EARLY_PREP_CONTRACT_INDEPENDENT.json',
                'baseline': 'BASELINE_SAVED_INDEPENDENT.json', 'records': 'INPUT_RECORDS_INDEPENDENT.json'}
    actions = {'early-cache': early_cache, 'prep-contract': prep_contract, 'baseline': baseline, 'records': records}
    actions[args.action](args.output or U / defaults[args.action])
