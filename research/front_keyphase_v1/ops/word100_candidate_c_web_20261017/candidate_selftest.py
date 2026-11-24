"""Camera-free C adapter tests. TRAIN cached observations only; no forward."""
from pathlib import Path
import inspect
import json
import sys
import time
import traceback
from unittest.mock import patch
import numpy as np

U = Path(__file__).resolve().parent
LAB = U.parents[1]
sys.path.insert(0, str(LAB / 'src'))
from wis_lab.service import word100_candidate as c
from wis_lab.core.observation import Observation, empty_frames, sha256_file


def read(p):
    return json.loads(Path(p).read_text('utf-8-sig'))


def synthetic():
    arrays = empty_frames(9)
    arrays['timestamp_us'][:] = [0, 20000, 50000, 80000, 160000, 200000, 240000, 290000, 340000]
    arrays['frame_id'][:] = np.arange(9) * 2
    for p, xy in [(11, [200, 100]), (12, [400, 100]), (13, [180, 170]), (14, [420, 170])]:
        arrays['xy_px'][:, p] = xy
        arrays['usable'][:, p] = arrays['observed'][:, p] = True
        arrays['conf'][:, p] = 1
    for f in (1, 2, 3, 5, 7):
        arrays['xy_px'][f, 33:54] = np.array([[260 + p * 2, 210 + p] for p in range(21)])
        arrays['observed'][f, 33:54] = arrays['usable'][f, 33:54] = True
        arrays['hand_score'][f, 0] = 0.9
    # Marginal extrapolations are originally allowed but must disappear in C.
    for p, xy in [(15, [640, 200]), (16, [-1, 200]), (25, [300, 480])]:
        arrays['xy_px'][:, p] = xy
        arrays['observed'][:, p] = arrays['usable'][:, p] = True
    return Observation(**arrays, sidecar=dict(width=640, height=480))


def run():
    start = time.perf_counter()
    checks = []
    evidence = []
    def ok(name, **data):
        checks.append(dict(check=name, status='PASS', **data))
    contract = read(LAB / 'config/contract.json')
    roi, joint = c._frozen()
    identity = c.identity()
    assert identity['model_status'] == 'READY', identity.get('reason')
    assert identity['fixed_seed'] == 20261013 and identity['selected_epoch'] == 77
    ok('frozen C checkpoint metadata, source pins, strict state load; no forward')
    obs = c.finalize_observation(synthetic())
    assert not obs.observed[:, [15, 16, 25]].any()
    assert np.isnan(obs.xy_px[:, [15, 16, 25]]).all()
    ok('strict all75 boundaries reject x=width, y=height and negative coordinates')
    clip, reason, stats = c.build_candidate_channels(obs)
    assert reason is None and stats['span_frames'] == 7 and stats['observed_hand_frames'] == 5
    np.testing.assert_array_equal(clip['frame_id'], obs.frame_id[1:8])
    np.testing.assert_array_equal(clip['timestamp_us'], obs.timestamp_us[1:8])
    assert not obs.usable[4, 33:].any() and not obs.usable[6, 33:].any()
    from wis_lab.service.neural import _channels
    old, why, _ = _channels(obs)
    assert why is None
    np.testing.assert_array_equal(clip['x'], old['x'])
    ok('same frozen active_span/normalisation, internal gaps and original PTS preserved', channels=int(clip['x'].size))
    original = synthetic()
    original.xy_px[:, [15, 16, 25]] *= 100
    changed, _, _ = c.build_candidate_channels(c.finalize_observation(original))
    np.testing.assert_array_equal(changed['x'], clip['x'])
    ok('hidden-coordinate perturbations have no feature influence')
    sparse = synthetic()
    sparse.observed[2:7, 33:] = sparse.usable[2:7, 33:] = False
    sparse.xy_px[2:7, 33:] = np.nan
    _, why, st = c.build_candidate_channels(c.finalize_observation(sparse))
    assert why == 'TOO_FEW_OBSERVED_HAND_FRAMES' and st['observed_hand_frames'] == 2
    ok('two endpoint observations cannot impersonate an observed take')
    assert list(inspect.signature(c.recognise).parameters) == ['obs']
    with patch.object(c, '_load', side_effect=RuntimeError('synthetic unavailable')):
        result = c.recognise(obs)
    assert result['model_status'] == 'MODEL_ERROR' and result['ranking'] == []
    assert result['model_key'] == c.KEY and result['no_fallback_to_default']
    ok('blind all100 API and C load failure never fall back to M0')

    context = read(c.UNIT / 'PREP_CONTEXT.json')
    ref = c._module('candidate_test_frozen_pair', LAB / 'ops/word100_fullframe_roi_pair_20261016/run_pair.py')
    for code in ('WORD0106', 'WORD1139', 'WORD1545'):
        cid = f'NIA_SL_{code}_REAL01_F'
        row = next(x for x in context['ordered_rows'] if x['clip_id'] == cid)
        assert row['role'] == 'train_candidate' and row['signer_id'] == 1
        source = row['reuse_sensor']
        for p, expected in source['pins'].items():
            assert sha256_file(p) == expected
        baseline = Observation.load(source['baseline'])
        with np.load(source['raw'], allow_pickle=False) as z:
            raw = {key: z[key] for key in z.files}
        w, h = int(raw['width'][0]), int(raw['height'][0])
        expected, _, _ = ref.associate(raw, baseline, joint.HandAssigner, contract)
        expected = roi.apply_roi(expected, (0, 0, w, h))
        class FakeCapture:
            def __init__(self, supplied):
                assert supplied == contract
                self.raw_rows = []
                self._raw_frame = None
                self.i = 0
                self.closed = False
            def extractor_id(self):
                return 'TEST_STUB_NO_DETECTOR'
            def process(self, rgb, ts, out, ri):
                i = self.i
                assert ts == baseline.timestamp_us[i]
                for key in ('xy_px', 'observed', 'usable', 'conf', 'hand_score'):
                    out[key][ri] = getattr(baseline, key)[i]
                self.raw_rows.append({key: value[i] for key, value in raw.items() if value.ndim > 0 and len(value) == baseline.n_frames})
                self.i += 1
            def close(self):
                self.closed = True
        with patch.object(c, '_capture_type', return_value=FakeCapture):
            capture = c.CandidateLandmarker(contract)
            second = c.CandidateLandmarker(contract)
            assert capture.assigner is not second.assigner and capture.raw is not second.raw
            second.close()
            out = empty_frames(baseline.n_frames)
            out['frame_id'][:] = baseline.frame_id
            rgb = np.zeros((h, w, 3), np.uint8)
            for i in range(baseline.n_frames):
                capture.process(rgb, int(baseline.timestamp_us[i]), out, i)
                assert capture.raw.raw_rows == []
            assert capture.assigner.frame == baseline.n_frames - 1
            observed = capture.finalize_observation(Observation(**out))
            capture.close()
            assert capture.raw.closed
        for key in c._ARRAYS:
            np.testing.assert_array_equal(getattr(observed, key), getattr(expected, key))
        new_clip, why, _ = c.build_candidate_channels(observed)
        record = read(c.UNIT / 'data/records' / (cid + '.json'))
        feature = record['views']['sensor'] if 'views' in record else record['modes']['sensor']
        # Prepared tensor is a saved TRAIN-only input, not a prediction or score.
        with np.load(feature['channel_file'], allow_pickle=False) as z:
            np.testing.assert_array_equal(new_clip['x'], z['x_whole'])
            np.testing.assert_array_equal(new_clip['frame_id'], z['whole_frame_id'])
            np.testing.assert_array_equal(new_clip['timestamp_us'], z['whole_ts'])
        ok('streaming matches frozen batch association+strictmask and saved TRAIN x_whole',
           clip_id=cid, frames=baseline.n_frames, zero_hand_frames=int((raw['raw_count'] == 0).sum()),
           values=int(new_clip['x'].size))
        evidence.append(dict(clip_id=cid, role=row['role'], raw_path=source['raw'], raw_sha256=sha256_file(source['raw']),
                             baseline_path=source['baseline'], baseline_sha256=sha256_file(source['baseline'])))

    class FailingCapture:
        def __init__(self, contract):
            self.raw_rows = []
            self._raw_frame = None
        def extractor_id(self): return 'FAILING_TEST_STUB'
        def process(self, rgb, ts, out, row):
            out['xy_px'][row] = 1
            out['observed'][row] = True
            raise RuntimeError('synthetic detector exception')
        def close(self): pass
    with patch.object(c, '_capture_type', return_value=FailingCapture):
        failed = c.CandidateLandmarker(contract)
        out = empty_frames(1)
        try:
            failed.process(np.zeros((20, 30, 3), np.uint8), 0, out, 0)
            raise AssertionError('exception must propagate')
        except RuntimeError as exc:
            assert str(exc) == 'synthetic detector exception'
        assert failed.failed and not out['observed'].any() and np.isnan(out['xy_px']).all()
        failed.close()
    ok('partial detector failure is terminal and cleared, never baseline-as-C')
    return dict(status='PASS', checks=checks, source_evidence=evidence, elapsed_seconds=time.perf_counter() - start,
                module_sha256=sha256_file(c.__file__), test_sha256=sha256_file(__file__),
                actual_detector_runs=0, camera_runs=0, actual_model_forward=0, training=0, DEV_reads=0,
                limitation='Adapter parity on fixed TRAIN cached observations and synthetic contracts; not live webcam accuracy.')


if __name__ == '__main__':
    try:
        result = run()
    except Exception:
        result = dict(status='FAIL', traceback=traceback.format_exc(), actual_detector_runs=0, training=0)
    n = 1
    while (U / f'candidate_SELFTEST_{n:03}.json').exists(): n += 1
    path = U / f'candidate_SELFTEST_{n:03}.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', 'utf-8')
    print(path)
    print(result['status'])
    if result['status'] != 'PASS':
        print(result['traceback'])
        raise SystemExit(1)
