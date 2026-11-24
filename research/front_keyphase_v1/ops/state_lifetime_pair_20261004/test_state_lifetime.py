"""Synthetic contract tests: real helper interfaces, fake VIDEO tasks, no media/models.

Run only after the runner repair is complete:
  python -B test_state_lifetime.py --config PREPARED_CONFIG.json
The receipt binds the exact runner, this test, and config bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
FRONT = HERE.parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


S = module('_slp_synthetic_runner', HERE / 'state_lifetime_pair.py')
CP = module('_slp_synthetic_cadence', HERE.parent / 'capture_cadence_pair_20261004' / 'cadence_pair.py')
HIP = module('_slp_synthetic_instrument', HERE.parent / 'hand_image_mode_pair_20261004' / 'hand_image_pair.py')
OB = module('_slp_synthetic_observation', FRONT / 'src' / 'wis_lab' / 'core' / 'observation.py')
FIELDS = CP.OBS_FIELDS


class FakeTask:
    def __init__(self, kind):
        self.kind, self.times, self.close_calls = kind, [], 0
        self.close_error = False

    def detect_for_video(self, image, ms):
        if self.close_calls:
            raise RuntimeError('detect after close')
        self.times.append(int(ms))
        if self.kind == 'hand':
            return NS(hand_landmarks=[], hand_world_landmarks=[], handedness=[])
        return NS(pose_landmarks=[])

    def close(self):
        self.close_calls += 1
        if self.close_error:
            raise RuntimeError('synthetic task close failure')


class FakeLM:
    """Same process(out,row)->None interface as the source Landmarker.

    A non-empty pose point exposes retained task history without any model call.
    The actual instrumentation/proxies own their real schemas and lifecycles.
    """
    def __init__(self, model_stub):
        self.hand_model_path = self.pose_model_path = str(model_stub)
        self.hand, self.pose = FakeTask('hand'), FakeTask('pose')
        self._last_ms = -1
        self.handedness_disagreements = self.handedness_comparisons = self.unassigned_hands = 0

    def extractor_id(self):
        return 'synthetic-interface-only'

    def _assign_sides(self, result, pose_xy, pose_usable, w, h):
        return {}

    def process(self, rgb, timestamp_us, out, row):
        ms = max(int(timestamp_us) // 1000, self._last_ms + 1)
        self._last_ms = ms
        self.pose.detect_for_video(rgb, ms)
        self.hand.detect_for_video(rgb, ms)
        out['xy_px'][row, 0] = (float(self.hand.calls), float(rgb[0, 0, 0]))
        out['observed'][row, 0] = out['usable'][row, 0] = True
        out['conf'][row, 0] = 1.0
        # Deliberately returns None, as the real API does.

    def close(self):
        # Same exception-swallowing behavior as production. Proxy evidence must catch it.
        for task in (self.hand, self.pose):
            try:
                task.close()
            except Exception:
                pass


FAKE_LM_MODULE = NS(
    POSE_L_WRIST=15, POSE_R_WRIST=16,
    mp_python=NS(BaseOptions=NS),
    mp_vision=NS(RunningMode=NS(VIDEO='VIDEO', IMAGE='IMAGE'), HandLandmarkerOptions=NS),
)


class FakeRuntime:
    obs_fields = FIELDS
    observation_cls = OB.Observation

    def __init__(self, folder):
        self.stub = Path(folder) / 'synthetic_task_bytes.bin'
        self.stub.write_bytes(b'not a model; synthetic hashing fixture')
        self.all_instr, self.all_lm = [], []
        self.inject = None
        self.calls = 0
        self.decode_metadata = {}
        self.rgb_calls = []
        self.time_base = '1/30000'

    def new_landmarker(self, tag):
        sink = []
        def base():
            lm = FakeLM(self.stub)
            self.all_lm.append(lm)
            return lm
        lm = HIP.make_factory(base, FAKE_LM_MODULE, tag, 'VIDEO', sink)()
        self.all_instr.extend(sink)
        return lm, sink[0]

    def iter_clip_frames(self, clip):
        self.decode_metadata[clip['clip_id']] = {'time_base': self.time_base}
        for fid in range(clip['expected_frames_from_prior']):
            raw_pts = 100 + fid * 1000
            rel = int(round(float((raw_pts - 100) / 30000) * 1_000_000))
            rgb = np.full((3, 4, 3), fid + 1, dtype=np.uint8)
            if self.inject == 'decode' and fid == 1:
                raise RuntimeError('synthetic decode failure')
            yield fid, raw_pts, rel, rgb

    def empty_frames(self, n):
        return OB.empty_frames(n)

    def validate_full(self, out):
        OB.Observation(**out).validate()

    def process(self, h, rgb, absolute_us, out, row):
        self.calls += 1
        self.rgb_calls.append((id(rgb), rgb.flags.writeable, int(absolute_us)))
        if self.inject == 'process' and self.calls == 4:
            raise RuntimeError('synthetic processing failure')
        h.lm.process(rgb, absolute_us, out, row)
        if self.inject == 'first_parity' and h.arm == 'B_session':
            out['xy_px'][row, 0, 0] += 1
        if self.inject == 'bad_observation':
            out['usable'][row, 1] = True

    def task_ms(self, row):
        return int(row['hand']['timestamp_ms_passed']), int(row['pose']['timestamp_ms_passed'])

    def tasks_of(self, h):
        return list(h.instr.tasks)

    def close(self, h):
        h.lm.close()
        records = [{'task_obj_id': id(t), 'ok': bool(t.close_rec and t.close_rec['ok']),
                    'close_rec': t.close_rec, 'task_record': t.record()} for t in h.instr.tasks]
        return None, records


def config(n=3):
    groups = [list(g) for g in S.FROZEN_GROUPS]
    clips = [{'clip_id': cid, 'path': f'synthetic://{cid}', 'expected_frames_from_prior': n,
              'score': None} for group in groups for cid in group]
    return {'groups': groups, 'clips': clips, 'gap_us': S.GAP_US}


class LifetimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='wis_state_lifetime_synthetic_')
        self.folder = Path(self.tmp.name)
        self.rt = FakeRuntime(self.folder)
        self.cfg = config()
        self.run = S.RunState()
        self.run.registry = S.Registry(self.rt, self.run)
        self.out = self.folder / 'result'
        (self.out / 'clips').mkdir(parents=True)
        (self.out / 'diag').mkdir()

    def tearDown(self):
        self.run.registry.close_all_open()
        self.tmp.cleanup()

    def extract(self):
        try:
            S.extract_all(self.cfg, self.rt, str(self.out), self.run)
        finally:
            self.run.registry.close_all_open()

    def assert_closed_once(self):
        for lm in self.rt.all_lm:
            for proxy in (lm.hand, lm.pose):
                self.assertEqual(proxy.task.close_calls, 1)

    def test_real_instrumentation_schemas_and_first_two_group_controls(self):
        self.extract()
        for group in self.cfg['groups']:
            pair = self.run.pairs[group[0]]
            self.assertEqual(S.obs_bitexact_diff(pair.obs['A_fresh'], pair.obs['B_session'], FIELDS), [])
        self.assertEqual(len(self.run.gates), 2)
        for ins in self.rt.all_instr:
            self.assertEqual(ins.errors, [])
            for row in ins.frames:
                self.assertIn('hand_raw', row)
                self.assertIn('assignment', row)
                self.assertEqual(row['hand']['api'], 'detect_for_video')
                self.assertEqual(row['rgb_sha256_before'], row['rgb_sha256_after'])

    def test_full_rows_no_even_selection_and_exact_budgets(self):
        self.extract()
        self.assertEqual(self.run.counters['decoded_frames'], 18)
        self.assertEqual(self.run.counters['process_calls'], 36)
        self.assertEqual(self.run.counters['landmarkers'], 8)
        self.assertEqual(self.run.counters['internal_tasks'], 16)
        self.assertEqual(self.run.counters['main_forwards'], 0)
        for pair in self.run.pairs.values():
            for arm in S.ARMS:
                self.assertEqual(len(pair.obs[arm]), 3)
                self.assertEqual([int(o['frame_id']) for o in pair.obs[arm]], [0, 1, 2])

    def test_each_pair_receives_same_readonly_rgb_and_absolute_time(self):
        self.extract()
        calls = self.rt.rgb_calls
        self.assertEqual(len(calls), 36)
        for a, b in zip(calls[::2], calls[1::2]):
            self.assertEqual(a, b)
            self.assertFalse(a[1])

    def test_fixed_gap_absolute_api_relative_saved_and_raw_pts(self):
        self.extract()
        for group in self.cfg['groups']:
            previous = None
            for cid in group:
                pair = self.run.pairs[cid]
                self.assertEqual(pair.first_absolute_us, 0 if previous is None else previous + 1_000_000)
                for arm in S.ARMS:
                    rows = pair.obs[arm]
                    self.assertEqual([int(o['timestamp_us']) for o in rows], [0, 33333, 66667])
                    for f in pair.frames:
                        self.assertEqual(f[arm]['hand_ms'], f['absolute_us'] // 1000)
                        self.assertEqual(f[arm]['pose_ms'], f['absolute_us'] // 1000)
                self.assertEqual([r['pts'] for r in pair.frames], [100, 1100, 2100])
                previous = pair.last_absolute_us

    def test_shared_state_survives_clip_boundary_but_not_signer_boundary(self):
        self.extract()
        for group in self.cfg['groups']:
            keys = [self.run.pairs[cid].lm_keys['B_session'] for cid in group]
            self.assertEqual(len(set(keys)), 1)
            second = self.run.pairs[group[1]]
            self.assertEqual(float(second.obs['A_fresh'][0]['xy_px'][0, 0]), 1.0)
            self.assertEqual(float(second.obs['B_session'][0]['xy_px'][0, 0]), 4.0)
        firsts = [self.run.pairs[g[0]].lm_keys['B_session'] for g in self.cfg['groups']]
        self.assertEqual(len(set(firsts)), 2)

    def test_all_eight_instances_sixteen_tasks_close_once(self):
        self.extract()
        handles = self.run.registry.handles
        self.assertEqual(len({id(h.lm) for h in handles}), 8)
        self.assertEqual(len({id(t.task) for h in handles for t in h.instr.tasks}), 16)
        self.assertEqual(sorted(len(h.clips) for h in handles if h.arm == 'B_session'), [2, 4])
        self.assertTrue(all(len(h.clips) == 1 for h in handles if h.arm == 'A_fresh'))
        self.run.registry.require_all_closed_ok()
        self.assert_closed_once()

    def test_saved_arrays_are_real_observation_shapes_and_all_rows(self):
        self.extract()
        for cid, pair in self.run.pairs.items():
            for arm in S.ARMS:
                with np.load(self.out / 'clips' / f'{cid}__{arm}.npz', allow_pickle=False) as z:
                    out = {k: z[f'obs_{k}'] for k in FIELDS}
                    OB.Observation(**out).validate()
                    self.assertEqual(out['xy_px'].shape, (3, 75, 2))
                    np.testing.assert_array_equal(out['frame_id'], [0, 1, 2])
                    np.testing.assert_array_equal(z['pts'], [100, 1100, 2100])
                    np.testing.assert_array_equal(z['absolute_us'], out['timestamp_us'] + pair.offset_us)

    def test_decode_error_closes_both_arms_without_scores(self):
        self.rt.inject = 'decode'
        with self.assertRaisesRegex(RuntimeError, 'decode failure'):
            self.extract()
        self.assert_closed_once()
        self.assertEqual(self.run.counters['main_forwards'], 0)

    def test_process_error_closes_both_arms_without_scores(self):
        self.rt.inject = 'process'
        with self.assertRaisesRegex(RuntimeError, 'processing failure'):
            self.extract()
        self.assert_closed_once()
        self.assertEqual(self.run.counters['main_forwards'], 0)

    def test_changed_task_time_is_rejected_before_scoring(self):
        original = self.rt.task_ms
        def wrong_ms(row):
            hand, pose = original(row)
            return hand + 1, pose
        self.rt.task_ms = wrong_ms
        with self.assertRaises(S.StageError):
            self.extract()
        self.assertEqual(self.run.counters['main_forwards'], 0)
        self.assert_closed_once()

    def test_first_group_parity_failure_stops_after_first_clip(self):
        self.rt.inject = 'first_parity'
        with self.assertRaises(S.TechnicalGateError):
            self.extract()
        self.assertEqual(self.run.counters['decoded_frames'], 3)
        self.assertEqual(self.run.counters['main_forwards'], 0)
        self.assert_closed_once()

    def test_wrong_frame_count_stops_without_scores(self):
        actual = self.rt.iter_clip_frames
        def too_short(clip):
            yield from actual(dict(clip, expected_frames_from_prior=2))
        self.rt.iter_clip_frames = too_short
        with self.assertRaises(S.StageError):
            self.extract()
        self.assert_closed_once()

    def test_invalid_observation_rejected_by_actual_validator(self):
        self.rt.inject = 'bad_observation'
        with self.assertRaises((AssertionError, S.StageError)):
            self.extract()
        self.assert_closed_once()

    def test_task_close_failure_is_not_silently_passed(self):
        original = self.rt.new_landmarker
        def bad_close(tag):
            lm, ins = original(tag)
            lm.hand.task.close_error = True
            return lm, ins
        self.rt.new_landmarker = bad_close
        with self.assertRaises(S.StageError):
            self.extract()
        self.assert_closed_once()


class HelperContractTests(unittest.TestCase):
    def test_real_decoder_contract_is_four_tuple_and_pts_rounding(self):
        image = np.zeros((2, 3, 3), np.uint8)
        frames = list(CP.strict_frames([(100, image), (1100, image), (2100, image)], '1/30000'))
        self.assertEqual([(fid, pts, us) for fid, pts, us, _ in frames],
                         [(0, 100, 0), (1, 1100, 33333), (2, 2100, 66667)])
        self.assertIs(frames[0][3], image)

    def test_missing_duplicate_and_reverse_pts_fail(self):
        image = np.zeros((2, 3, 3), np.uint8)
        for pts in ([None], [0, 0], [1, 0]):
            with self.subTest(pts=pts), self.assertRaises(CP.DecodeContractError):
                list(CP.strict_frames([(p, image) for p in pts], '1/30000'))

    def test_actual_observation_has_sidecar_but_only_seven_array_fields(self):
        self.assertEqual(len(FIELDS), 7)
        self.assertNotIn('sidecar', FIELDS)
        self.assertEqual(set(OB.empty_frames(2)), set(FIELDS))
        out = OB.empty_frames(2)
        out['timestamp_us'][:] = [0, 33333]
        OB.Observation(**out).validate()

    def test_half_open_annotation_uses_relative_us(self):
        self.assertEqual(S.half_open([0, 1, 2, 3], [0, 1000, 2000, 3000], 1000, 3000),
                         [(1, 1000), (2, 2000)])

    def test_rank_ties_follow_class_index(self):
        classes = ['zero', 'one', 'two']
        logits = np.array([4.0, 4.0, 1.0])
        rank = CP.rank_against_label(logits, classes, 'one')
        self.assertEqual(rank['top1'], 'zero')
        self.assertEqual(rank['true_rank'], 2)


class RuntimeAdapterTests(unittest.TestCase):
    """Exercise the repaired production adapter, not just a parallel fake adapter."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='wis_state_lifetime_adapter_')
        self.fixture = FakeRuntime(self.tmp.name)
        self.rt = S.PinnedRuntime.__new__(S.PinnedRuntime)
        self.rt.cp, self.rt.hip, self.rt.lm_mod = CP, HIP, FAKE_LM_MODULE
        self.rt.obs_fields, self.rt.observation_cls = FIELDS, OB.Observation
        self.rt.decode_metadata = {}
        self.rt.deps = NS(empty_frames=OB.empty_frames, observation_cls=OB.Observation,
                          landmarker_factory=lambda: FakeLM(self.fixture.stub))

    def tearDown(self):
        self.tmp.cleanup()

    def test_factory_process_instrumented_times_and_dictionary_close_record(self):
        lm, ins = self.rt.new_landmarker('synthetic_adapter')
        h = NS(lm=lm, instr=ins)
        out = self.rt.empty_frames(2)
        out['timestamp_us'][:] = [0, 33333]
        out['frame_id'][:] = [0, 1]
        image = np.zeros((3, 4, 3), np.uint8)
        try:
            self.assertIsNone(self.rt.process(h, image, 2_000_000, out, 0))
            self.assertIsNone(self.rt.process(h, image, 2_033_333, out, 1))
            self.rt.validate_full(out)
            self.assertEqual(self.rt.task_ms(ins.frames[1]), (2033, 2033))
            self.assertEqual(ins.frames[1]['frame_id'], 1)
            self.assertEqual(ins.frames[1]['timestamp_us'], 2_033_333)
            np.testing.assert_array_equal(out['timestamp_us'], [0, 33333])
        finally:
            error, records = self.rt.close(h)
        self.assertIsNone(error)
        self.assertEqual(len(records), 2)
        self.assertTrue(all(r['ok'] for r in records))

    def test_decoder_context_four_tuple_and_original_pts(self):
        image = np.zeros((3, 4, 3), np.uint8)
        class Decoder:
            time_base = '1/30000'
            meta = {'time_base': '1/30000', 'width': 4, 'height': 3}
            entered = False
            exited = False
            def __enter__(self):
                self.entered = True
                return self
            def __exit__(self, *_):
                self.exited = True
            def raw(self):
                yield 100, image
                yield 1100, image
        dec = Decoder()
        self.rt.deps.open_decoder = lambda path: dec
        rows = list(self.rt.iter_clip_frames({'path': 'synthetic://generated', 'clip_id': 'SYNTH'}))
        self.assertTrue(dec.entered and dec.exited)
        self.assertEqual([(f, p, us) for f, p, us, _ in rows], [(0, 100, 0), (1, 1100, 33333)])

    def test_window_adapter_passes_full_dict_indices_and_deps(self):
        seen = {}
        def to_native(xy, mask):
            seen['rows'] = len(xy)
            return xy, mask
        def channels(xy, mask, ts):
            seen['ts'] = ts.copy()
            return {'frames': len(ts), 'body_scale': 2.0, 'hand_scales': [3.0, 4.0]}
        def collate(batch):
            n = batch[0]['frames']
            self.assertEqual(batch[0]['label'], 0)
            return np.zeros((1, n, 280), np.float32), np.ones((1, n), np.float32), np.array([0])
        self.rt.deps.to_native_like = to_native
        self.rt.deps.build_channels = channels
        self.rt.deps.collate = collate
        self.rt.deps.n_channels = 280
        full = OB.empty_frames(4)
        full['timestamp_us'][:] = [0, 1000, 2000, 3000]
        full['frame_id'][:] = [0, 1, 2, 3]
        rows = [{k: full[k][i].copy() for k in FIELDS} for i in range(4)]
        out = self.rt.prepare_window(rows, 1000, 3000)
        self.assertEqual(seen['rows'], 2)
        np.testing.assert_array_equal(seen['ts'], [1000, 2000])
        self.assertEqual(out['x'].shape, (1, 2, 280))
        self.assertEqual(out['body_scale'], 2.0)
        self.assertEqual(out['hand_scales'], [3.0, 4.0])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--receipt', default=str(HERE / 'TEST_RECEIPT.json'))
    args = ap.parse_args()
    cfg = Path(args.config).resolve()
    tracked = {'state_lifetime_pair.py': HERE / 'state_lifetime_pair.py',
               'test_state_lifetime.py': Path(__file__).resolve()}
    before = {name: digest(path) for name, path in tracked.items()}
    config_sha = digest(cfg)
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    after = {name: digest(path) for name, path in tracked.items()}
    unchanged = before == after and config_sha == digest(cfg)
    passed = result.wasSuccessful() and unchanged
    receipt = {'schema': 'wis_state_lifetime_synthetic_receipt_v1',
               'status': 'PASS' if passed else 'FAIL', 'tests_run': result.testsRun,
               'failures': len(result.failures), 'errors': len(result.errors),
               'skipped': len(result.skipped), 'code_sha256': after,
               'config_sha256': config_sha, 'protected_unchanged': unchanged,
               'coverage': 'real helper contracts with fake VIDEO tasks; generated RGB only',
               'real_media_decodes': 0, 'mediapipe_runs': 0, 'model_loads': 0,
               'model_forwards': 0, 'camera_calls': 0}
    path = Path(args.receipt)
    with path.open('x', encoding='utf-8') as fh:
        json.dump(receipt, fh, ensure_ascii=False, indent=2, allow_nan=False)
        fh.write('\n')
    print(json.dumps(receipt, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
