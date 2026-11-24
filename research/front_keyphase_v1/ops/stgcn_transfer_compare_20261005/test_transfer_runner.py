'''Synthetic unit tests for transfer_runner: temporary NPZs only, no real data.'''
import math
import os
import tempfile
import unittest

import numpy as np

import transfer_runner as tr

CLASSES = ['c%d' % i for i in range(93)]


def semantic_points(T, seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(T, 27, 2)) * 40.0 + 300.0
    pts[:, 3] = (250.0, 200.0)
    pts[:, 4] = (350.0, 200.0)
    return pts


def layout_arrays(pts, layout):
    T, n = pts.shape[0], tr.LAYOUT[layout]['n']
    xy = np.full((T, n, 2), np.nan, np.float32)
    valid = np.zeros((T, n), bool)
    xy[:, tr.SELECT[layout]] = pts
    valid[:, tr.SELECT[layout]] = True
    return xy, valid


class TransferRunnerTests(unittest.TestCase):
    def test_manifest_identity_includes_condition(self):
        classes = ['c%02d' % k for k in range(93)]
        def row(i, y, cond):
            return {'group': 'base', 'id': 'clip%03d' % i, 'label': classes[y], 'y': y,
                    'role': 'train', 'signer_id': 1, 'condition': cond,
                    'path': os.path.abspath(os.path.join(os.sep, 'synthetic', 'clip%03d.npy' % i)),
                    'sha256': '0' * 64, 'layout': 'native67', 'start': 0, 'end_exclusive': 1,
                    'mask_hand': None, 'weight_count': 1, 'exposed': False}
        rows = [row(i, i, 'whole_clip') for i in range(93)]
        ok = rows + [row(0, 0, 'provided_annotation')]
        tr.validate_manifest({'classes': classes, 'rows': ok}, strict=False)
        bad = ok + [row(0, 0, 'provided_annotation')]
        with self.assertRaisesRegex(Exception, 'duplicate'):
            tr.validate_manifest({'classes': classes, 'rows': bad}, strict=False)

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def write_clip(self, name, layout, xy, valid, ts=None):
        spec = tr.LAYOUT[layout]
        if ts is None:
            ts = np.arange(xy.shape[0], dtype=np.int64) * 33333
        path = os.path.abspath(os.path.join(self.tmp, name + '.npz'))
        np.savez(path, **{spec['xy']: xy, spec['valid']: valid, spec['ts']: ts})
        return path, tr.sha256_file(path)

    def row(self, path, sha, layout, start, end, mask=None, group='base', y=0, weight=1):
        return {'group': group, 'id': os.path.basename(path), 'label': CLASSES[y], 'y': y, 'role': 'synthetic',
                'signer_id': None, 'condition': 'syn', 'path': path, 'sha256': sha, 'layout': layout,
                'start': start, 'end_exclusive': end, 'mask_hand': mask, 'weight_count': weight, 'exposed': False}

    def test_native_and_mp_same_semantic_input_normalize_identically(self):
        pts = semantic_points(6)
        outs = {}
        for layout in ('native67', 'mp75'):
            xy, valid = layout_arrays(pts, layout)
            valid[2, tr.SELECT[layout][10]] = False
            p, h = self.write_clip(layout, layout, xy, valid)
            outs[layout] = tr.preprocess_row(self.row(p, h, layout, 0, 6))
        (xn, dn), (xm, dm) = outs['native67'], outs['mp75']
        self.assertEqual(xn.shape, (1, 2, 6, 27))
        self.assertEqual(xn.dtype, np.float32)
        self.assertTrue(xn.flags['C_CONTIGUOUS'])
        np.testing.assert_array_equal(xn, xm)
        self.assertAlmostEqual(dn['scale'], 100.0)
        self.assertEqual(dn['input_array_sha256'], dm['input_array_sha256'])
        np.testing.assert_allclose(xn[0, :, 0, 3], [-0.5, 0.0])
        self.assertEqual(xn[0, :, 2, 10].tolist(), [0.0, 0.0])

    def test_missing_nan_ignored_present_nan_rejected(self):
        xy, valid = layout_arrays(semantic_points(3), 'native67')
        j = tr.SELECT['native67'][8]
        xy[1, j] = np.nan
        valid[1, j] = False
        out, _ = tr.normalize_clip(xy, valid, 'native67')
        self.assertTrue(np.isfinite(out).all())
        self.assertEqual(out[0, :, 1, 8].tolist(), [0.0, 0.0])
        valid[1, j] = True
        with self.assertRaises(ValueError):
            tr.normalize_clip(xy, valid, 'native67')

    def test_frames_without_both_shoulders_are_zeroed(self):
        xy, valid = layout_arrays(semantic_points(4), 'mp75')
        valid[2, 12] = False
        out, diag = tr.normalize_clip(xy, valid, 'mp75')
        self.assertTrue(np.all(out[0, :, 2, :] == 0))
        self.assertTrue(np.any(out[0, :, 1, :] != 0))
        self.assertEqual(diag['zero_frame_count'], 1)
        self.assertEqual(diag['frame_count'], 4)

    def test_no_valid_scale_raises(self):
        xy, valid = layout_arrays(semantic_points(3), 'native67')
        v2 = valid.copy()
        v2[:, 5] = False
        with self.assertRaises(ValueError):
            tr.normalize_clip(xy, v2, 'native67')
        xy2 = xy.copy()
        xy2[:, 2] = xy2[:, 5]
        with self.assertRaises(ValueError):
            tr.normalize_clip(xy2, valid, 'native67')

    def test_mask_hand_preserves_interval_and_source(self):
        pts = semantic_points(10, seed=1)
        xy, valid = layout_arrays(pts, 'native67')
        p, h = self.write_clip('m', 'native67', xy, valid)
        x_plain, _ = tr.preprocess_row(self.row(p, h, 'native67', 2, 7))
        x_mask, d_mask = tr.preprocess_row(self.row(p, h, 'native67', 2, 7, mask='left'))
        self.assertEqual(x_mask.shape, (1, 2, 5, 27))
        self.assertEqual((d_mask['start'], d_mask['end_exclusive'], d_mask['frame_count']), (2, 7, 5))
        left = list(range(7, 17))
        others = [j for j in range(27) if j not in left]
        self.assertTrue(np.all(x_mask[0][:, :, left] == 0))
        np.testing.assert_array_equal(x_mask[0][:, :, others], x_plain[0][:, :, others])
        ref, _ = tr.normalize_clip(xy[2:7], valid[2:7], 'native67')
        np.testing.assert_array_equal(x_plain, ref)
        self.assertEqual(tr.sha256_file(p), h)

    def test_bad_source_aborts_with_row_identity(self):
        xy, valid = layout_arrays(semantic_points(4), 'mp75')
        p, h = self.write_clip('bad', 'mp75', xy, valid)
        with self.assertRaises(RuntimeError) as ctx:
            tr.preprocess_row(self.row(p, 'f' * 64, 'mp75', 0, 4), 7)
        self.assertIn('bad.npz', str(ctx.exception))
        self.assertIn('row 7', str(ctx.exception))
        p2, h2 = self.write_clip('ts', 'mp75', xy, valid, ts=np.array([0, 10, 10, 20], np.int64))
        with self.assertRaises(RuntimeError):
            tr.preprocess_row(self.row(p2, h2, 'mp75', 0, 4))
        with self.assertRaises(RuntimeError):
            tr.preprocess_row(self.row(p, h, 'mp75', 0, 5))

    def test_weighted_stats_zero_weight_rows_excluded(self):
        rng = np.random.default_rng(3)
        f = rng.normal(size=(5, 4))
        f[:, 0] = 5.0
        c = np.array([3, 0, 1, 2, 0])
        f_bad = f.copy()
        f_bad[1] = 1e30
        f_bad[4] = np.nan
        m1, s1 = tr.weighted_stats(f_bad, c)
        m2, s2 = tr.weighted_stats(f[[0, 2, 3]], c[[0, 2, 3]])
        np.testing.assert_allclose(m1, m2)
        np.testing.assert_allclose(s1, s2)
        np.testing.assert_allclose(m1, (3 * f[0] + f[2] + 2 * f[3]) / 6.0)
        self.assertEqual(s1[0], tr.STD_FLOOR)
        with self.assertRaises(ValueError):
            tr.weighted_stats(f_bad, c, total=7)

    def test_weighted_head_trains_tiny_example(self):
        rng = np.random.default_rng(4)
        centers = np.eye(3) * 3.0
        y = np.repeat(np.arange(3), 6)
        Z = centers[y] + rng.normal(scale=0.3, size=(18, 3))
        counts = np.full(18, 2.0)
        counts[0] = 0.0
        Z[0] = np.nan
        w = counts / counts.sum()
        cfg = {'max_iter': 25, 'max_eval': 40}
        W, b, rep = tr.fit_head(Z, y, w, num_classes=3, optimizer_config=cfg)
        self.assertTrue(rep['final_grad_finite'])
        self.assertTrue(math.isfinite(rep['final_train_loss']))
        self.assertAlmostEqual(rep['initial_train_loss'], math.log(3.0))
        self.assertLess(rep['final_train_loss'], 0.5 * rep['initial_train_loss'])
        self.assertLessEqual(rep['lbfgs_n_iter'], 25)
        self.assertGreaterEqual(rep['closure_calls'], 1)
        self.assertEqual(rep['n_rows_in_loss'], 17)
        self.assertEqual(rep['optimizer_config']['max_iter'], 25)
        self.assertEqual(tr.FROZEN_LBFGS['max_iter'], 500)
        pred = np.argmax(Z[1:] @ W.T + b, axis=1)
        np.testing.assert_array_equal(pred, y[1:])
        W2, b2, _ = tr.fit_head(Z, y, w, num_classes=3, optimizer_config=cfg)
        np.testing.assert_array_equal(W, W2)
        np.testing.assert_array_equal(b, b2)

    def test_metrics_top1_uses_all_93_and_macro_only_gt_present(self):
        y = np.array([0, 0, 1])
        s = np.zeros((3, 93))
        s[0, 0] = 1.0
        s[1, 90] = 5.0
        s[2, 1] = 2.0
        pred = tr.top1(s)
        self.assertEqual(pred.tolist(), [0, 90, 1])
        summ = tr.summarize(pred, y, CLASSES)
        self.assertAlmostEqual(summ['accuracy_pct'], 200.0 / 3.0)
        self.assertAlmostEqual(summ['macro_recall_gt_present_pct'], 75.0)
        self.assertEqual(summ['n_gt_present_classes'], 2)
        self.assertEqual(sorted(summ['per_class']), ['c0', 'c1'])
        with self.assertRaises(ValueError):
            tr.top1(np.zeros((3, 92)))
        cmp = tr.compare(pred, np.array([0, 0, 0]), y, np.array([10, 11, 12]), np.array(['a', 'b', 'c']))
        self.assertEqual(cmp['a_correct_b_wrong'], [[12, 'c']])
        self.assertEqual(cmp['regressions_b_correct_a_wrong'], [[11, 'b']])
        self.assertEqual(cmp['pp_delta_a_minus_b'], 0.0)

    def test_manifest_schema_checks(self):
        p = os.path.abspath(os.path.join(self.tmp, 'none.npz'))
        rows = [dict(self.row(p, '0' * 64, 'native67', 0, 4, y=c, weight=2), id='r%d' % c) for c in range(93)]
        ev = dict(self.row(p, '0' * 64, 'mp75', 0, 4, group='mp', weight=0), id='e0', baseline_scores=[0.0] * 93)
        tr.validate_manifest({'classes': CLASSES, 'rows': rows + [ev]}, strict=False)
        with self.assertRaises(ValueError):
            tr.validate_manifest({'classes': CLASSES, 'rows': rows + [ev]}, strict=True)
        bad = [dict(rows[0], baseline_scores=[0.0] * 93)] + rows[1:] + [ev]
        with self.assertRaises(ValueError):
            tr.validate_manifest({'classes': CLASSES, 'rows': bad}, strict=False)
        with self.assertRaises(ValueError):
            tr.validate_manifest({'classes': CLASSES, 'rows': rows[1:] + [ev]}, strict=False)
        no_base = dict(ev)
        del no_base['baseline_scores']
        with self.assertRaises(ValueError):
            tr.validate_manifest({'classes': CLASSES, 'rows': rows + [no_base]}, strict=False)


if __name__ == '__main__':
    unittest.main()
