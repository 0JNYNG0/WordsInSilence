"""Synthetic offline tests; all temporary media stays in this unit's directory."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import av
import numpy as np

from prepare_rgb_pairs import make_crop, run, sha


class CropTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.zeros((100, 160, 3), np.uint8)
        self.rgb[:] = (220, 40, 10)
        self.xy = np.stack((np.linspace(50, 80, 21), np.linspace(35, 65, 21)), 1)
        self.mask = np.ones(21, bool)
        self.shoulder = np.array([[40, 20], [100, 20]], dtype=float)

    def test_normal_channel_order_and_scale(self):
        c = make_crop(self.rgb, self.xy, self.mask, self.shoulder)
        self.assertTrue(c['crop_valid'])
        np.testing.assert_array_equal(c['bbox_xyxy'], [42, 27, 87, 72])
        np.testing.assert_array_equal(c['rgb'][64, 64], [220, 40, 10])
        self.assertTrue(c['pixel_valid'].all())
        self.assertEqual(c['padding_fraction'], 0)
        self.assertEqual(c['shoulder_relative_size'], 45/60)
        self.assertEqual(c['frame_relative_size'], 45/160)

    def test_missing_and_partial(self):
        for n, reason in [(0, 'missing'), (20, 'partial')]:
            m = np.arange(21) < n
            c = make_crop(self.rgb, self.xy, m, self.shoulder)
            self.assertFalse(c['crop_valid'])
            self.assertEqual(c['crop_reason'], reason)
            self.assertFalse(c['rgb'].any())
            self.assertFalse(c['pixel_valid'].any())
            self.assertTrue(np.isnan(c['frame_relative_size']))

    def test_invalid_coordinates(self):
        for case, reason in [('nan', 'nonfinite'), ('outside', 'outside'), ('point', 'degenerate')]:
            x = self.xy.copy()
            if case == 'nan':
                x[0, 0] = np.nan
            elif case == 'outside':
                x[0, 0] = 160
            else:
                x[:] = (40, 40)
            c = make_crop(self.rgb, x, self.mask, self.shoulder)
            self.assertEqual(c['crop_reason'], reason)
            self.assertFalse(c['crop_valid'])

    def test_edge_padding_and_unknown_shoulders(self):
        x = self.xy - (49, 34)
        c = make_crop(self.rgb, x, self.mask, np.full((2, 2), np.nan))
        np.testing.assert_array_equal(c['bbox_xyxy'], [-7, -7, 38, 38])
        self.assertAlmostEqual(c['padding_fraction'], 1-38*38/(45*45))
        self.assertFalse(c['pixel_valid'][0, 0])
        self.assertTrue(c['pixel_valid'][-1, -1])
        self.assertTrue(np.isnan(c['shoulder_relative_size']))

    def test_inputs_unchanged_and_bad_shapes(self):
        originals = [a.copy() for a in (self.rgb, self.xy, self.mask, self.shoulder)]
        make_crop(self.rgb, self.xy, self.mask, self.shoulder)
        for a, b in zip(originals, (self.rgb, self.xy, self.mask, self.shoulder)):
            np.testing.assert_array_equal(a, b)
        with self.assertRaises(ValueError):
            make_crop(self.rgb, self.xy[:20], self.mask[:20], self.shoulder)
        with self.assertRaises(ValueError):
            make_crop(self.rgb.astype(float), self.xy, self.mask, self.shoulder)


class FilePairTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='synthetic_', dir=Path(__file__).parent)
        self.p = Path(self.tmp.name)
        video = self.p/'synthetic.mp4'
        with av.open(str(video), 'w') as con:
            st = con.add_stream('mpeg4', rate=10)
            st.width, st.height, st.pix_fmt = 160, 100, 'yuv420p'
            for i in range(4):
                rgb = np.zeros((100, 160, 3), np.uint8)
                rgb[:] = (40+i*45, 100, 200-i*30)
                frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
                for packet in st.encode(frame): con.mux(packet)
            for packet in st.encode(): con.mux(packet)
        with av.open(str(video)) as con:
            st = con.streams.video[0]
            frames = list(con.decode(st))
            times = np.array([round(f.pts*st.time_base*1e6) for f in frames], np.int64)
            self.truth = {i: hashlib.sha256(f.to_ndarray(format='rgb24').tobytes()).hexdigest()
                          for i, f in enumerate(frames)}
        xy = np.full((4, 75, 2), np.nan, np.float32)
        us = np.zeros((4, 75), bool)
        xy[:, [11, 12]] = [[40, 20], [100, 20]]
        us[:, [11, 12]] = True
        pts = np.stack((np.linspace(50, 80, 21), np.linspace(35, 65, 21)), 1)
        xy[:, 33:54] = pts
        us[:, 33:54] = True
        xy[:, 54:75] = pts + (40, 0)
        us[:, 54:75] = True
        xy[3, 54:75], us[3, 54:75] = np.nan, False
        obs, inp = self.p/'obs.npz', self.p/'input.npz'
        np.savez(obs, xy_px=xy, usable=us, timestamp_us=times, frame_id=np.arange(4, dtype=np.int64))
        np.savez(inp, x=np.zeros((2, 280), np.float32),
                 timestamp_us=times[[1, 3]], frame_id=np.array([1, 3], np.int64))
        self.row = dict(sample_id='sample', clip_id='clip', signer_id=1, label='synthetic')
        for key, path in [('raw', video), ('observation', obs), ('input', inp)]:
            self.row[key+'_path'], self.row[key+'_sha256'] = str(path), sha(path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_complete_file_pair(self):
        out = self.p/'success'
        result = run({'rows': [self.row]}, out)
        self.assertEqual(result['frames'], 2)
        self.assertEqual(result['per_hand_reasons'][1], {'valid': 1, 'missing': 1})
        self.assertEqual(result['timing_max_error_us'], 0)
        for f in result['rows_detail'][0]['frames_rgb']:
            self.assertEqual(f['rgb_sha256'], self.truth[f['frame_id']])
        with np.load(out/'sample.npz') as z:
            self.assertEqual(z['rgb'].shape, (2, 2, 128, 128, 3))
            np.testing.assert_array_equal(z['frame_id'], [1, 3])
            self.assertFalse(z['crop_valid'][1, 1])
            self.assertFalse(z['pixel_valid'][1, 1].any())
        for key in ('raw', 'observation', 'input'):
            self.assertEqual(sha(self.row[key+'_path']), self.row[key+'_sha256'])
        with self.assertRaises(FileExistsError): run({'rows': [self.row]}, out)

    def test_hash_rejection_preserved(self):
        row = dict(self.row, raw_sha256='0'*64)
        out = self.p/'bad_hash'
        with self.assertRaisesRegex(ValueError, 'Hash mismatch'): run({'rows': [row]}, out)
        self.assertTrue((out/'ERROR.json').exists())

    def test_prepared_time_mismatch_rejected(self):
        path = Path(self.row['input_path'])
        with np.load(path) as z: a = {k: z[k] for k in z.files}
        a['timestamp_us'] = a['timestamp_us'] + 100
        np.savez(path, **a)
        self.row['input_sha256'] = sha(path)
        with self.assertRaisesRegex(ValueError, 'frame-time mismatch'):
            run({'rows': [self.row]}, self.p/'bad_time')

    def test_decoder_time_mismatch_rejected(self):
        for key in ('input', 'observation'):
            path = Path(self.row[key+'_path'])
            with np.load(path) as z: a = {k: z[k] for k in z.files}
            a['timestamp_us'] = a['timestamp_us'] + 100
            np.savez(path, **a)
            self.row[key+'_sha256'] = sha(path)
        with self.assertRaisesRegex(ValueError, 'PTS mismatch'):
            run({'rows': [self.row]}, self.p/'bad_pts')

    def test_invalid_sample_id_rejected(self):
        row = dict(self.row, sample_id='../sample')
        with self.assertRaisesRegex(ValueError, 'sample IDs'):
            run({'rows': [row]}, self.p/'bad_id')


if __name__ == '__main__':
    unittest.main(verbosity=2)
