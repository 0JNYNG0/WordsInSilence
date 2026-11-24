import math
import sys
import types
import unittest
from unittest import mock

import numpy as np
import torch

try:
    import reference_adapter  # noqa: F401
except ImportError:  # standalone: only LAYOUTS is needed, patched per test
    _stub = types.ModuleType('reference_adapter')
    _stub.LAYOUTS = {}
    sys.modules['reference_adapter'] = _stub

import full_span_adapter as fsa

MP = tuple(range(74, 47, -1))  # reversed so joint selection order is observable
TEST_LAYOUTS = {'mp75': (75, MP), 'native67': (67, tuple(range(27)))}
W, H = 640.0, 480.0
CASES = (1, 59, 60, 61, 123, 203)
PRESET_CALL = {'reference_points_preset': 'shoulder_mediapipe_holistic_minimal_27', 'frame_level': False}


class StubNormalize:
    """Mean-center on joints 3/4 over all frames, divide by mean shoulder distance."""
    calls, seen = [], None

    def __init__(self, **kwargs):
        StubNormalize.calls.append(kwargs)

    def __call__(self, data):
        x = data['frames']
        StubNormalize.seen = x.clone()
        ref = x[:, :, [3, 4]]
        center = ref.mean(dim=(1, 2), keepdim=True)
        scale = torch.linalg.vector_norm(ref[:, :, 0] - ref[:, :, 1], dim=0).mean()
        data['frames'] = (x - center) / scale
        return data


STUB = types.SimpleNamespace(CenterAndScaleNormalize=StubNormalize)


def returning(frames):
    return types.SimpleNamespace(CenterAndScaleNormalize=lambda **_: (lambda data: {'frames': frames}))


def make(t, seed=0, v=75):
    rng = np.random.default_rng(seed)
    return rng.uniform(20, 620, size=(t, v, 2)), np.ones((t, v), dtype=bool)


def reference(xy, present, indices=MP):
    a = np.where(present[..., None], xy, 0.0) / np.array([W, H])
    sel = a[:, list(indices)].transpose(2, 0, 1)
    ref = sel[:, :, [3, 4]]
    center = ref.mean(axis=(1, 2), keepdims=True)
    return (sel - center) / np.linalg.norm(ref[:, :, 0] - ref[:, :, 1], axis=0).mean()


class FullSpanAdapterTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(fsa.LAYOUTS, TEST_LAYOUTS, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        StubNormalize.calls, StubNormalize.seen = [], None

    def run_adapter(self, xy, present, layout='mp75', width=W, height=H, module=STUB):
        return fsa.prepare_full_span_windows(xy, present, layout, width, height, module)

    def test_sample_indices(self):
        self.assertEqual(fsa.sample_indices(61), list(range(59)) + [60])
        for t in CASES:
            idx = fsa.sample_indices(t)
            self.assertEqual(len(idx), 60)
            if t <= 60:
                self.assertEqual(idx, [-1] * (60 - t) + list(range(t)))
            else:
                self.assertEqual(idx, [j * (t - 1) // 59 for j in range(60)])
                self.assertEqual((idx[0], idx[-1]), (0, t - 1))
                self.assertTrue(all(a < b for a, b in zip(idx, idx[1:])))
        for bad in (0, -3, True, 2.0, None):
            with self.assertRaises((TypeError, ValueError)):
                fsa.sample_indices(bad)

    def test_cases_exact_selection_and_axis_order(self):
        for t in CASES:
            with self.subTest(t=t):
                StubNormalize.calls = []
                xy, present = make(t, seed=t)
                out, d = self.run_adapter(xy, present)
                self.assertEqual(tuple(out.shape), (1, 6, 2, 10, 27))
                self.assertEqual(out.dtype, torch.float32)
                self.assertTrue(out.is_contiguous())
                self.assertEqual(d['selected_source_indices'], fsa.sample_indices(t))
                self.assertEqual((d['original_frames'], d['padded_frames'], d['dropped_frames']),
                                 (t, max(0, 60 - t), max(0, t - 60)))
                self.assertEqual(d['policy'], 'full_span_uniform_frames_v1')
                self.assertEqual((d['width'], d['height']), (W, H))
                self.assertEqual(d['input_shape'], (t, 75, 2))
                self.assertEqual(d['output_shape'], (1, 6, 2, 10, 27))
                self.assertEqual(StubNormalize.calls, [PRESET_CALL])
                self.assertEqual(tuple(StubNormalize.seen.shape), (2, t, 27))
                ref = reference(xy, present)
                for slot, src in enumerate(d['selected_source_indices']):
                    w, k = divmod(slot, 10)
                    expected = np.zeros((2, 27)) if src < 0 else ref[:, src]
                    np.testing.assert_allclose(out[0, w, :, k, :].numpy(), expected, atol=1e-4)

    def test_input_not_mutated(self):
        xy, present = make(70, seed=1)
        present[5, 10] = False
        xy[5, 10] = np.nan
        xy_copy, present_copy = xy.copy(), present.copy()
        self.run_adapter(xy, present)
        np.testing.assert_array_equal(xy, xy_copy)
        np.testing.assert_array_equal(present, present_copy)

    def test_last_frame_retained(self):
        for t in (61, 123, 203):
            xy, present = make(t, seed=7)
            out, d = self.run_adapter(xy, present)
            self.assertEqual(d['selected_source_indices'][-1], t - 1)
            np.testing.assert_allclose(out[0, 5, :, 9, :].numpy(), reference(xy, present)[:, t - 1], atol=1e-4)

    def test_frame_61_affects_full_normalization(self):
        xy, present = make(61, seed=3)
        moved = xy.copy()
        moved[60, MP[3]] += 150.0
        first = self.run_adapter(xy, present)[0][0, 0, :, 0, :]
        second = self.run_adapter(moved, present)[0][0, 0, :, 0, :]
        self.assertFalse(torch.allclose(first, second))
        np.testing.assert_allclose(second.numpy(), reference(moved, present)[:, 0], atol=1e-4)

    def test_absence_zero_before_not_after_normalization(self):
        xy, present = make(12, seed=4)
        present[:, MP[7]] = False
        xy[:, MP[7]] = np.nan
        out, _ = self.run_adapter(xy, present)
        self.assertTrue(torch.equal(StubNormalize.seen[:, :, 7], torch.zeros(2, 12)))
        joint = out[0, 5, :, :, 7]  # last window holds observed frames 2..11
        np.testing.assert_allclose(joint.numpy(), reference(xy, present)[:, 2:, 7], atol=1e-4)
        self.assertTrue(bool((joint.abs() > 1e-3).any()))

    def test_present_and_xy_validation(self):
        xy, present = make(5)
        nan_xy = xy.copy()
        nan_xy[2, 0, 1] = np.nan
        inf_xy = xy.copy()
        inf_xy[0, 3, 0] = np.inf
        cases = [
            (xy, present.astype(np.int64), TypeError),
            (xy, present.astype(np.float32), TypeError),
            (xy, present[:, :74], ValueError),
            (nan_xy, present, ValueError),
            (inf_xy, present, ValueError),
            (xy > 100, present, TypeError),
            (xy[:, :67], present[:, :67], ValueError),
            (xy[:0], present[:0], ValueError),
            (np.concatenate([xy, xy[..., :1]], -1), present, ValueError),
        ]
        for i, (x, p, err) in enumerate(cases):
            with self.subTest(i=i), self.assertRaises(err):
                self.run_adapter(x, p)
        self.assertEqual(StubNormalize.calls, [])

    def test_shoulder_gate_is_explicit(self):
        xy, present = make(8, seed=5)
        same = xy.copy()
        same[:, MP[4]] = same[:, MP[3]]
        absent = present.copy()
        absent[:, [MP[3], MP[4]]] = False
        for x, p in ((same, present), (xy, absent)):
            with self.assertRaisesRegex(ValueError, 'shoulder'):
                self.run_adapter(x, p)
        self.assertEqual(StubNormalize.calls, [])

    def test_bad_metadata_has_no_fallback(self):
        xy, present = make(5)
        for bad in (None, True, np.bool_(True), 0, -480.0, math.nan, math.inf, '640'):
            for key in ('width', 'height'):
                with self.subTest(key=key, bad=bad), self.assertRaises((TypeError, ValueError)):
                    self.run_adapter(xy, present, **{key: bad})
        with self.assertRaises(ValueError):
            self.run_adapter(xy, present, layout='unknown')
        with self.assertRaises(ValueError):
            self.run_adapter(xy, present, layout='native67')
        self.assertEqual(StubNormalize.calls, [])

    def test_native_layout_and_bad_normalizer_output(self):
        xy, present = make(5, v=67)
        out, _ = self.run_adapter(xy, present, layout='native67')
        np.testing.assert_allclose(out[0, 5, :, 9, :].numpy(), reference(xy, present, range(27))[:, 4], atol=1e-4)
        for bad in (torch.zeros(2, 5, 27, dtype=torch.float64), torch.full((2, 5, 27), math.nan),
                    torch.zeros(2, 4, 27), np.zeros((2, 5, 27), dtype=np.float32)):
            with self.assertRaises(ValueError):
                self.run_adapter(xy, present, layout='native67', module=returning(bad))


if __name__ == '__main__':
    unittest.main()
