import hashlib
import os
import tempfile
import unittest

import numpy as np
import torch

from generic_pose_io import augment_windows, load_row, prepare_dataset, prepare_windows

BAD = (TypeError, ValueError)
LAYOUTS = {'layout_a': {'xy_key': 'pts', 'mask_key': 'vis', 'time_key': 'stamp', 'vertices': 5,
                        'hand_slices': {'hand_b': [3, 5]}}}


def identity(clip):
    return clip


def fake_normalizer(left, right):
    # Clip-mean shoulder midpoint as origin, mean shoulder distance as unit.
    def normalize(clip):
        center = ((clip[:, :, left] + clip[:, :, right]) / 2).mean(dim=1)
        scale = torch.linalg.vector_norm(clip[:, :, left] - clip[:, :, right], dim=0).mean()
        return (clip - center[:, None, None]) / scale
    return normalize


def fake_linear(matrix, offset):
    matrix = torch.tensor(matrix, dtype=torch.float32)
    offset = torch.tensor(offset, dtype=torch.float32)

    def transform(clip):
        return torch.einsum('ij,jtv->itv', matrix, clip) + offset[:, None, None]
    return transform


def make_clip(frames, vertices, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(frames, vertices, 2)), np.ones((frames, vertices), dtype=bool)


def unwindow(out):
    # [1, W, 2, Tw, V] -> [2, W * Tw, V]
    _, windows, _, frames, vertices = out.shape
    return out[0].permute(1, 0, 2, 3).reshape(2, windows * frames, vertices)


def as_clip(xy):
    return torch.from_numpy(xy.astype(np.float32)).permute(2, 0, 1)


class PrepareWindowsTests(unittest.TestCase):
    def run_prepare(self, xy, present, **overrides):
        kwargs = dict(joint_indices=[0, 1, 2], shoulder_indices=[0, 1], target_frames=4,
                      window_frames=2, normalize_fn=identity)
        kwargs.update(overrides)
        return prepare_windows(xy, present, **kwargs)

    def test_short_clip_is_front_padded_with_zero_frames(self):
        xy, present = make_clip(3, 4)
        out, diag = self.run_prepare(xy, present)
        self.assertEqual(tuple(out.shape), (1, 2, 2, 2, 3))
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(diag['T'], 3)
        self.assertEqual(diag['indices'], [-1, 0, 1, 2])
        self.assertEqual(diag['shape'], [1, 2, 2, 2, 3])
        frames = unwindow(out)
        self.assertTrue(torch.equal(frames[:, 0], torch.zeros(2, 3)))
        self.assertTrue(torch.equal(frames[:, 1:], as_clip(xy[:, :3])))

    def test_equal_length_keeps_all_frames(self):
        xy, present = make_clip(4, 4)
        out, diag = self.run_prepare(xy, present)
        self.assertEqual(diag['indices'], [0, 1, 2, 3])
        self.assertTrue(torch.equal(unwindow(out), as_clip(xy[:, :3])))

    def test_long_clip_uses_floor_indices(self):
        for frames, expected in ((5, [0, 1, 2, 4]), (7, [0, 2, 4, 6]), (10, [0, 3, 6, 9])):
            xy, present = make_clip(frames, 4)
            out, diag = self.run_prepare(xy, present)
            self.assertEqual(diag['indices'], expected)
            self.assertTrue(torch.equal(unwindow(out), as_clip(xy[expected][:, :3])))

    def test_missing_joints_zeroed_before_normalize_and_not_rezeroed(self):
        xy, present = make_clip(4, 4)
        xy[1, 2] = np.nan
        present[1, 2] = False
        seen = []

        def shift(clip):
            seen.append(clip.clone())
            return clip + 1.0

        out, _ = self.run_prepare(xy, present, normalize_fn=shift)
        self.assertTrue(torch.equal(seen[0][:, 1, 2], torch.zeros(2)))
        self.assertTrue(torch.equal(unwindow(out)[:, 1, 2], torch.ones(2)))

    def test_joint_order_and_shoulders_index_selection(self):
        xy, present = make_clip(4, 5)
        out, _ = self.run_prepare(xy, present, joint_indices=[4, 1, 3], shoulder_indices=[2, 0])
        self.assertTrue(torch.equal(unwindow(out), as_clip(xy[:, [4, 1, 3]])))

    def test_fake_normalizer_centers_and_scales(self):
        xy, present = make_clip(6, 4, seed=3)
        out, diag = self.run_prepare(xy, present, target_frames=6, window_frames=3,
                                     normalize_fn=fake_normalizer(0, 1))
        clip = unwindow(out)
        center = ((clip[:, :, 0] + clip[:, :, 1]) / 2).mean(dim=1)
        scale = torch.linalg.vector_norm(clip[:, :, 0] - clip[:, :, 1], dim=0).mean()
        torch.testing.assert_close(center, torch.zeros(2), atol=1e-5, rtol=0)
        torch.testing.assert_close(scale, torch.tensor(1.0), atol=1e-5, rtol=0)
        expected = np.linalg.norm(xy[:, 0] - xy[:, 1], axis=1).mean()
        self.assertAlmostEqual(diag['shoulder_distance'], expected, places=5)

    def test_inputs_not_mutated(self):
        xy, present = make_clip(4, 4)
        xy[0, 2] = np.nan
        present[0, 2] = False
        xy_before, present_before = xy.copy(), present.copy()
        self.run_prepare(xy, present, normalize_fn=lambda clip: clip.mul_(2.0))
        self.assertTrue(np.array_equal(xy, xy_before, equal_nan=True))
        self.assertTrue(np.array_equal(present, present_before))

    def test_degenerate_shoulders_rejected(self):
        xy, present = make_clip(4, 4)
        xy[:, 1] = xy[:, 0]
        with self.assertRaises(ValueError):
            self.run_prepare(xy, present)
        xy, present = make_clip(4, 4)
        present[:2, 0] = False
        present[2:, 1] = False
        _, diag = self.run_prepare(xy, present)
        filled = np.where(present[:, :, None], xy, 0.0)
        expected = np.linalg.norm(filled[:, 0] - filled[:, 1], axis=1).mean()
        self.assertAlmostEqual(diag['shoulder_distance'], expected, places=5)

    def test_invalid_inputs_rejected(self):
        xy, present = make_clip(4, 4)
        nan_observed = xy.copy()
        nan_observed[2, 0, 1] = np.nan
        cases = {
            'empty clip': (np.zeros((0, 4, 2)), np.zeros((0, 4), dtype=bool), {}),
            'int mask': (xy, present.astype(np.int64), {}),
            'mask shape': (xy, present[:, :3], {}),
            'xy channels': (np.zeros((4, 4, 3)), present, {}),
            'bool xy': (xy > 0, present, {}),
            'complex xy': (xy.astype(np.complex64), present, {}),
            'observed nan': (nan_observed, present, {}),
            'duplicate joint': (xy, present, {'joint_indices': [0, 1, 1]}),
            'joint out of range': (xy, present, {'joint_indices': [0, 1, 4]}),
            'empty joints': (xy, present, {'joint_indices': []}),
            'one shoulder': (xy, present, {'shoulder_indices': [0]}),
            'same shoulder': (xy, present, {'shoulder_indices': [1, 1]}),
            'shoulder outside selection': (xy, present, {'shoulder_indices': [0, 3]}),
            'bool index': (xy, present, {'shoulder_indices': [False, True]}),
            'target too small': (xy, present, {'target_frames': 1, 'window_frames': 1}),
            'indivisible target': (xy, present, {'target_frames': 5}),
            'float64 output': (xy, present, {'normalize_fn': lambda c: c.double()}),
            'shape output': (xy, present, {'normalize_fn': lambda c: c[:, :2]}),
            'nonfinite output': (xy, present, {'normalize_fn': lambda c: c / 0.0}),
        }
        for name, (case_xy, case_present, overrides) in cases.items():
            with self.subTest(name), self.assertRaises(BAD):
                self.run_prepare(case_xy, case_present, **overrides)


class AugmentWindowsTests(unittest.TestCase):
    def test_transform_shares_parameters_across_windows(self):
        torch.manual_seed(0)
        sample = torch.randn(3, 2, 4, 5)
        before = sample.clone()
        matrix, offset = [[0.0, -1.0], [1.0, 0.0]], [0.5, -0.25]
        out = augment_windows(sample, fake_linear(matrix, offset))
        expected = torch.einsum('ij,wjtv->witv', torch.tensor(matrix), sample) \
            + torch.tensor(offset)[None, :, None, None]
        self.assertEqual(out.shape, sample.shape)
        self.assertTrue(out.is_contiguous())
        torch.testing.assert_close(out, expected)
        self.assertTrue(torch.equal(sample, before))

    def test_transform_called_once_on_whole_clip(self):
        calls = []

        def record(clip):
            calls.append(tuple(clip.shape))
            return clip.mul_(3.0)

        sample = torch.ones(2, 2, 3, 4)
        out = augment_windows(sample, record)
        self.assertEqual(calls, [(2, 6, 4)])
        self.assertTrue(torch.equal(out, torch.full((2, 2, 3, 4), 3.0)))
        self.assertTrue(torch.equal(sample, torch.ones(2, 2, 3, 4)))

    def test_bad_inputs_and_outputs_rejected(self):
        sample = torch.zeros(2, 2, 3, 4)
        cases = {
            'sample rank': (torch.zeros(2, 3, 4), identity),
            'sample channels': (torch.zeros(2, 3, 3, 4), identity),
            'integer sample': (torch.zeros(2, 2, 3, 4, dtype=torch.int64), identity),
            'shape output': (sample, lambda c: c[:, :1]),
            'dtype output': (sample, lambda c: c.double()),
            'nonfinite output': (sample, lambda c: c / 0.0),
        }
        for name, (case, transform) in cases.items():
            with self.subTest(name), self.assertRaises(BAD):
                augment_windows(case, transform)


class NpzTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.calls = []

    def write(self, name, **arrays):
        path = os.path.join(self.tmp.name, name + '.npz')
        np.savez(path, **arrays)
        with open(path, 'rb') as handle:
            return path, hashlib.sha256(handle.read()).hexdigest()

    def arrays(self, frames=6):
        rng = np.random.default_rng(1)
        return {'pts': rng.uniform(-1, 1, size=(frames, 5, 2)).astype(np.float32),
                'vis': np.ones((frames, 5), dtype=np.uint8),
                'stamp': np.arange(frames, dtype=np.float64) * 0.5}

    def prepare(self, xy, mask, layout):
        self.calls.append((xy.copy(), mask.copy(), layout))
        return prepare_windows(xy, mask, [0, 1, 2, 3, 4], [0, 1], 4, 2, identity)

    def row(self, path, sha, **extra):
        row = {'path': path, 'sha256': sha, 'layout': 'layout_a', 'start': 1, 'end_exclusive': 5}
        row.update(extra)
        return row


class LoadRowTests(NpzTestCase):
    def test_loads_verified_slice_and_masks_hand_on_copy(self):
        arrays = self.arrays()
        path, sha = self.write('clip', **arrays)
        sample, diag = load_row(self.row(path, sha, mask_hand='hand_b'), LAYOUTS, self.prepare)
        xy, mask, layout = self.calls[0]
        self.assertEqual(layout, 'layout_a')
        self.assertTrue(np.array_equal(xy, arrays['pts'][1:5]))
        self.assertEqual(mask.dtype, np.bool_)
        self.assertFalse(mask[:, 3:].any())
        self.assertTrue(mask[:, :3].all())
        self.assertEqual(tuple(sample.shape), (2, 2, 2, 5))
        self.assertEqual(float(sample[..., 3:].abs().sum()), 0.0)
        self.assertEqual(diag['source_sha256'], sha)
        self.assertEqual(diag['slice'], [1, 5])
        self.assertEqual(diag['hand_slice'], [3, 5])
        self.assertEqual(len(diag['input_tensor_sha256']), 64)
        self.assertEqual(diag['prepare_diag']['T'], 4)
        with np.load(path) as stored:
            self.assertTrue(stored['vis'].all())

    def test_no_mask_hand_keeps_all_joints(self):
        path, sha = self.write('clip', **self.arrays())
        load_row(self.row(path, sha, mask_hand=None), LAYOUTS, self.prepare)
        self.assertTrue(self.calls[0][1].all())

    def test_hash_mismatch_rejected_before_prepare(self):
        path, sha = self.write('clip', **self.arrays())
        for bad in ('0' * 64, sha.upper(), sha[:-1]):
            with self.subTest(bad), self.assertRaises(ValueError):
                load_row(self.row(path, bad), LAYOUTS, self.prepare)
        self.assertEqual(self.calls, [])

    def test_time_checks(self):
        cases = {
            'repeated': np.array([0.0, 1.0, 1.0, 2.0, 3.0, 4.0]),
            'decreasing unsigned': np.array([5, 4, 3, 2, 1, 0], dtype=np.uint8),
            'nan': np.array([0.0, 1.0, np.nan, 3.0, 4.0, 5.0]),
            'length': np.arange(5.0),
            'rank': np.arange(6.0).reshape(6, 1),
            'bool': np.ones(6, dtype=bool),
        }
        for name, stamp in cases.items():
            arrays = self.arrays()
            arrays['stamp'] = stamp
            path, sha = self.write(name.replace(' ', '_'), **arrays)
            with self.subTest(name), self.assertRaises(BAD):
                load_row(self.row(path, sha), LAYOUTS, self.prepare)

    def test_array_checks(self):
        base = self.arrays()
        cases = {
            'mask_value': {'vis': np.full((6, 5), 2, dtype=np.int64)},
            'float_mask': {'vis': np.ones((6, 5))},
            'mask_shape': {'vis': np.ones((6, 4), dtype=bool)},
            'integer_xy': {'pts': np.zeros((6, 5, 2), dtype=np.int64)},
            'xy_vertices': {'pts': np.zeros((6, 4, 2), dtype=np.float32)},
        }
        for name, change in cases.items():
            path, sha = self.write(name, **{**base, **change})
            with self.subTest(name), self.assertRaises(BAD):
                load_row(self.row(path, sha), LAYOUTS, self.prepare)
        missing = {k: v for k, v in base.items() if k != 'stamp'}
        path, sha = self.write('missing_key', **missing)
        with self.assertRaises((KeyError, ValueError)):
            load_row(self.row(path, sha), LAYOUTS, self.prepare)

    def test_row_field_checks(self):
        path, sha = self.write('clip', **self.arrays())
        cases = {
            'empty slice': {'start': 3, 'end_exclusive': 3},
            'past end': {'end_exclusive': 7},
            'negative start': {'start': -1},
            'float start': {'start': 1.0},
            'unknown layout': {'layout': 'other'},
            'unknown hand': {'mask_hand': 'hand_x'},
            'nonstring hand': {'mask_hand': 1},
        }
        for name, change in cases.items():
            with self.subTest(name), self.assertRaises(BAD):
                load_row(self.row(path, sha, **change), LAYOUTS, self.prepare)

    def test_prepare_fn_contract(self):
        path, sha = self.write('clip', **self.arrays())
        cases = {
            'not a tuple': lambda xy, mask, layout: torch.zeros(1, 2, 2, 2, 5),
            'missing batch dim': lambda xy, mask, layout: (torch.zeros(2, 2, 2, 5), {}),
            'float64': lambda xy, mask, layout: (torch.zeros(1, 2, 2, 2, 5, dtype=torch.float64), {}),
            'nonfinite': lambda xy, mask, layout: (torch.full((1, 2, 2, 2, 5), float('nan')), {}),
        }
        for name, prepare_fn in cases.items():
            with self.subTest(name), self.assertRaises(BAD):
                load_row(self.row(path, sha), LAYOUTS, prepare_fn)


class PrepareDatasetTests(NpzTestCase):
    def rows(self):
        path_a, sha_a = self.write('a', **self.arrays())
        path_b, sha_b = self.write('b', **self.arrays(frames=8))
        return [
            self.row(path_a, sha_a, id='a', origin='o1', y=0, weight_count=0),
            self.row(path_b, sha_b, id='b', origin='o2', y=2, weight_count=2.5, start=0, end_exclusive=8),
        ]

    def test_zero_weight_rows_accepted(self):
        dataset, diagnostics = prepare_dataset(self.rows(), LAYOUTS, self.prepare)
        self.assertEqual(set(dataset), {'x', 'y', 'weight', 'ids', 'origins'})
        self.assertEqual(tuple(dataset['x'].shape), (2, 2, 2, 2, 5))
        self.assertEqual(dataset['x'].dtype, torch.float32)
        self.assertEqual(dataset['y'].dtype, torch.int64)
        self.assertEqual(dataset['y'].tolist(), [0, 2])
        self.assertEqual(dataset['weight'].dtype, torch.float64)
        self.assertEqual(dataset['weight'].tolist(), [0.0, 2.5])
        self.assertEqual(dataset['ids'], ['a', 'b'])
        self.assertEqual(dataset['origins'], ['o1', 'o2'])
        self.assertEqual([d['id'] for d in diagnostics], ['a', 'b'])

    def test_bad_rows_rejected(self):
        rows = self.rows()
        cases = {
            'duplicate id': lambda r: r[1].update(id='a'),
            'empty id': lambda r: r[0].update(id=''),
            'negative weight': lambda r: r[0].update(weight_count=-1.0),
            'nan weight': lambda r: r[0].update(weight_count=float('nan')),
            'inf weight': lambda r: r[0].update(weight_count=float('inf')),
            'bool weight': lambda r: r[0].update(weight_count=True),
            'bool label': lambda r: r[0].update(y=True),
            'negative label': lambda r: r[0].update(y=-1),
            'missing origin': lambda r: r[0].pop('origin'),
        }
        for name, mutate in cases.items():
            case = [dict(r) for r in rows]
            mutate(case)
            with self.subTest(name), self.assertRaises(BAD):
                prepare_dataset(case, LAYOUTS, self.prepare)
        with self.assertRaises(BAD):
            prepare_dataset([], LAYOUTS, self.prepare)


if __name__ == '__main__':
    unittest.main()
