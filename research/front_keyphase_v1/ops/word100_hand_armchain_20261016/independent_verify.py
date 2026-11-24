"""Independent conservation and ordering audit. No camera/model/raw extraction."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import hashlib
import importlib.util
import io
import json
import sys
import unittest

sys.dont_write_bytecode = True
import numpy as np

U = Path(__file__).resolve().parent
OLD = U.with_name('word100_hand_assignment_joint_20261016')
ROOT = U.parents[3]
LAB = U.parents[1]
sys.path[:0] = [str(U), str(LAB/'src'), str(ROOT/'research/sign_model_pilot_v1/src')]


def read(p):
    return json.loads(Path(p).read_text('utf-8-sig'))


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def arrays(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def exact(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape, (a.dtype, b.dtype, a.shape, b.shape)
    assert np.array_equal(a, b, equal_nan=True) if a.dtype.kind == 'f' else np.array_equal(a, b)


def load_module(name, p):
    spec = importlib.util.spec_from_file_location(name, p)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def tests():
    import assignment
    prior = load_module('prior_independent_tests', OLD/'test_assignment_independent.py')
    assert prior.HandAssigner is assignment.HandAssigner
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(prior.HandIdentityTests)
    if (U/'test_armchain_independent.py').exists():
        new = load_module('armchain_independent_tests', U/'test_armchain_independent.py')
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(new))
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    print(output.getvalue())
    return {'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
            'status': 'PASS' if result.wasSuccessful() else 'FAIL',
            'candidate_sha256': digest(U/'assignment.py'),
            'legacy_test_source_sha256': digest(OLD/'test_assignment_independent.py'),
            'log': output.getvalue()}


def audit():
    from assignment import HandAssigner
    replay = read(U/'REPLAY_RESULT.json')
    old = read(OLD/'REPLAY_RESULT.json')
    by_pair = {(r['clip_id'], r['mode']): r for r in old['rows']}
    assert len(replay['rows']) == 40
    assert {(r['clip_id'], r['mode']) for r in replay['rows']} == set(by_pair)
    rule = read(U/'POINT_CONTRACT.json')
    margin, score_threshold = rule['bounds_margin_ratio'], rule['hand_score_threshold']
    protected = {}
    for p in [OLD/'REPLAY_CONTEXT.json', OLD/'CANDIDATE_FREEZE.json',
              U/'REPLAY_CONTEXT.json', U/'CANDIDATE_FREEZE.json']:
        if p.exists():
            protected.update(read(p)['pins'])
    for p in [OLD/'PROTECTED_FILES.json', U/'PROTECTED_FILES.json']:
        if p.exists():
            protected.update(read(p))
    for row in old['rows']:
        protected.update(row['outputs'])
    for row in replay['rows']:
        protected.update(row['outputs'])
    for p, sha in protected.items():
        assert digest(p) == sha, ('protected file changed', p)
    totals, totals_prev = Counter(), Counter()
    all_changes, regressions = [], []
    frames = detections = raw_zero = points = 0
    old_corrections = old_new_observations = 0
    target_records = []
    targets = {('NIA_SL_WORD1139_REAL01_F', 'native', 71, 1): 0,
               ('NIA_SL_WORD1139_REAL01_F', 'crop', 56, 1): 0,
               ('NIA_SL_WORD0106_REAL01_F', 'crop', 52, 0): 1}
    for row in replay['rows']:
        cid, mode = row['clip_id'], row['mode']
        prior_row = by_pair[cid, mode]
        assert Path(row['raw_dir']).resolve() == Path(prior_row['raw_dir']).resolve()
        raw = arrays(Path(row['raw_dir'])/'raw.npz')
        base = arrays(Path(row['raw_dir'])/'baseline.npz')
        dest = U/'replay'/cid/mode
        cand = arrays(dest/'candidate.npz')
        sides = arrays(dest/'assignment.npz')['sides']
        prev = arrays(dest/'previous_assignment.npz')['sides']
        exact(prev, arrays(OLD/'replay'/cid/mode/'assignment.npz')['sides'])
        trace = read(dest/'trace.json')
        for key in ['timestamp_us', 'frame_id']:
            exact(cand[key], base[key])
        for key in ['xy_px', 'observed', 'usable', 'conf']:
            exact(cand[key][:, :33], base[key][:, :33])
        assert not (cand['usable'] & ~cand['observed']).any()
        assert np.isnan(cand['xy_px'][~cand['observed']]).all()
        assert np.isnan(cand['conf'][:, 33:]).all()
        counts, previous_counts = Counter(), Counter()
        changes, previous_changes = [], []
        reversed_assigner = HandAssigner()
        for f, value in enumerate(raw['raw_count']):
            n = int(value)
            frames += 1
            detections += n
            raw_zero += n == 0
            assert (sides[f, n:] == -1).all()
            assigned = [int(s) for s in sides[f, :n] if s >= 0]
            assert len(assigned) == len(set(assigned))
            assert trace[f]['assigned_sides'] == [None if s < 0 else int(s) for s in sides[f, :n]]
            assert trace[f]['baseline_assigned_sides'] == [None if s < 0 else int(s) for s in prev[f, :n]], ('legacy replay state drift', cid, mode, f)
            assert trace[f]['arm_chain']['baseline_tracks_modified'] is False
            assert trace[f]['coordinates_generated'] is False
            assert trace[f]['detected_hands'] == n
            assert trace[f]['frame'] == f and trace[f]['timestamp_us'] == int(raw['timestamp_us'][f])
            order = list(reversed(range(n)))
            reverse_map, _ = reversed_assigner.assign(raw['hand_xy_px'][f, order], raw['raw_label'][f, order],
                raw['raw_score'][f, order], raw['pose_xy_px'][f], raw['pose_usable'][f],
                int(raw['timestamp_us'][f]), int(raw['width'][f]), int(raw['height'][f]))
            for j, i in enumerate(order):
                assert (-1 if reverse_map[j] is None else reverse_map[j]) == sides[f, i], ('raw order', cid, mode, f, i)
            for side in [0, 1]:
                sl = slice(33 + 21*side, 54 + 21*side)
                selected = [i for i in range(n) if sides[f, i] == side]
                if not selected:
                    assert not cand['observed'][f, sl].any() and not cand['usable'][f, sl].any()
                    assert np.isnan(cand['xy_px'][f, sl]).all() and np.isnan(cand['hand_score'][f, side])
                    continue
                assert len(selected) == 1
                i = selected[0]
                p = raw['hand_xy_px'][f, i].astype(np.float32)
                exact(cand['xy_px'][f, sl], p)
                points += 21
                assert cand['observed'][f, sl].all()
                assert cand['hand_score'][f, side] == np.float32(raw['raw_score'][f, i])
                extent = np.array([raw['width'][f], raw['height'][f]])
                expected_mask = np.isfinite(p).all(1) & (p >= -margin*extent).all(1) & (p <= (1+margin)*extent).all(1) & (raw['raw_score'][f, i] >= score_threshold)
                exact(cand['usable'][f, sl], expected_mask)
            for i in range(n):
                b, p, s = int(raw['baseline_side'][f, i]), int(prev[f, i]), int(sides[f, i])
                reason = trace[f]['reasons'][i]
                category = 'unchanged' if b == s else 'newly_assigned' if b < 0 else 'dropped' if s < 0 else 'side_changed'
                pc = 'unchanged' if p == s else 'newly_assigned' if p < 0 else 'dropped' if s < 0 else 'side_changed'
                counts[category] += 1
                counts['raw_detections'] += 1
                counts['reason_'+reason] += 1
                previous_counts[pc] += 1
                if category != 'unchanged':
                    changes.append(dict(frame=f, raw_index=i, old_side=b, new_side=s, category=category, reason=reason))
                if pc != 'unchanged':
                    detail = dict(frame=f, raw_index=i, previous_side=p, new_side=s, category=pc, reason=reason)
                    previous_changes.append(detail)
                    all_changes.append(dict(clip_id=cid, mode=mode, **detail))
                if p >= 0 and s != p:
                    regressions.append(dict(clip_id=cid, mode=mode, frame=f, raw_index=i, previous_side=p, new_side=s))
                if b >= 0 and p >= 0 and b != p:
                    old_corrections += 1
                    assert s == p, ('previous correction lost', cid, mode, f, i)
                if b < 0 and p >= 0:
                    old_new_observations += 1
                    assert s == p, ('previous observation lost', cid, mode, f, i)
                if (cid, mode, f, i) in targets:
                    expected = targets[cid, mode, f, i]
                    assert s == expected
                    target_records.append(dict(clip_id=cid, mode=mode, frame=f, raw_index=i, expected=expected, actual=s))
            counts['zero_detection_frames'] += n == 0
        assert dict(counts) == row['counts'], ('counts', cid, mode)
        assert dict(previous_counts) == row['counts_vs_previous'], ('previous counts', cid, mode)
        assert changes == read(dest/'changes.json')
        assert previous_changes == read(dest/'changes_vs_previous.json')
        totals.update(counts)
        totals_prev.update(previous_counts)
    assert dict(totals) == replay['counts']
    assert dict(totals_prev) == replay['counts_vs_previous']
    assert (frames, detections, raw_zero) == (5072, 6720, 1442)
    assert len(target_records) == 3
    return dict(status='PASS', conditions=40, frames=frames, raw_detections=detections,
        no_raw_hand_frames=raw_zero, protected_files=len(protected), raw_order_changes=0,
        point_coordinates_exact=points, compared_to_production=dict(totals), compared_to_joint=dict(totals_prev),
        prior_side_corrections_retained=old_corrections, prior_new_observations_retained=old_new_observations,
        three_prior_target_recoveries=target_records, all_changes_vs_joint=all_changes,
        previous_assigned_raw_changes=regressions,
        limits='Conservation and side consistency only: no 21-point ground truth, recognition accuracy, or training claim.')


if __name__ == '__main__':
    report = {'verifier': 'armchain_verify', 'synthetic': tests(), 'training': 0, 'classifier_inference': 0}
    if '--audit' in sys.argv:
        assert report['synthetic']['status'] == 'PASS'
        report['saved_output_audit'] = audit()
    if '--save' in sys.argv:
        assert report['synthetic']['status'] == 'PASS'
        target = U/('INDEPENDENT_VERIFICATION.json' if '--audit' in sys.argv else 'INDEPENDENT_SYNTHETIC.json')
        with target.open('x', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v if k != 'synthetic' else {kk: vv for kk, vv in v.items() if kk != 'log'} for k, v in report.items()}, indent=2, ensure_ascii=False))
    raise SystemExit(report['synthetic']['status'] != 'PASS')
