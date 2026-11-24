"""Synthetic preparation-contract checks, no video extraction or model inference."""
import unittest
import numpy as np
import prep
from wis_lab.core.observation import Observation, empty_frames


def make_observation():
    a = empty_frames(18)
    a['frame_id'][:] = np.arange(18)
    a['timestamp_us'][:] = np.arange(18)*33333
    points = {0: (300, 80), 11: (400, 200), 12: (200, 200),
              13: (420, 300), 14: (180, 300), 15: (420, 400), 16: (180, 400)}
    for j, xy in points.items():
        a['xy_px'][:, j] = xy; a['observed'][:, j] = True; a['usable'][:, j] = True
        a['conf'][:, j] = .99
    for f in (2, 3, 4, 5, 11, 12, 13, 14):
        for j in range(21):
            a['xy_px'][f, 33+j] = [350+j, 300+2*j]
            a['observed'][f, 33+j] = a['usable'][f, 33+j] = True
        a['hand_score'][f, 0] = .99
    return Observation(**a, sidecar={'width': 640, 'height': 480})


class PreparationContract(unittest.TestCase):
    def test_whole_span_preserves_internal_absence_and_pts(self):
        obs = make_observation()
        arrays, info = prep.features(obs, {'role': 'development_only'})
        self.assertEqual(info['whole_span'], [2, 14])
        self.assertEqual(arrays['x_whole'].shape, (13, 280))
        np.testing.assert_array_equal(arrays['whole_frame_id'], np.arange(2, 15))
        np.testing.assert_array_equal(arrays['whole_ts'], obs.timestamp_us[2:15])
        self.assertFalse(arrays['x_whole'][4:9, 195:237].any())
        self.assertTrue((arrays['x_whole'][1:, -1] > 0).all())

    def test_development_ignores_annotation_and_never_context(self):
        a, _ = prep.features(make_observation(), {'role': 'development_only', 'intervals': [[900, 901]]})
        self.assertNotIn('x_context', a)

    def test_train_context_intersect_and_independent_dt(self):
        obs = make_observation()
        a, _ = prep.features(obs, {'role': 'train_candidate', 'intervals': [[.30, .40]]})
        self.assertEqual(a['x_context'][0, -1], 0)
        self.assertTrue(set(a['context_frame_id']).issubset(set(a['whole_frame_id'])))

    def test_train_empty_context_is_unavailable(self):
        a, info = prep.features(make_observation(), {'role': 'train_candidate', 'intervals': [[3, 4]]})
        self.assertIsNone(a)
        self.assertEqual(info['reason'], 'EMPTY_CONTEXT_INTERVAL')

    def test_missing_hand_is_not_synthetic_hand(self):
        obs = make_observation()
        a, _ = prep.features(obs, {'role': 'development_only'})
        self.assertFalse(a['x_whole'][:, 216:237].any())
        self.assertFalse(a['x_whole'][:, 258:279].any())

    def test_observed_finger_without_hand_wrist_has_only_body_channels(self):
        obs = make_observation()
        obs.observed[:, 33] = obs.usable[:, 33] = False
        obs.xy_px[:, 33] = np.nan
        a, _ = prep.features(obs, {'role': 'development_only'})
        self.assertTrue(a['x_whole'][:, 196:216].any())
        self.assertFalse(a['x_whole'][:, 237:258].any())

    def test_no_hand_is_explicit_failure(self):
        obs = make_observation()
        obs.observed[:, 33:] = obs.usable[:, 33:] = False
        obs.xy_px[:, 33:] = np.nan
        a, info = prep.features(obs, {'role': 'development_only'})
        self.assertIsNone(a)
        self.assertEqual(info['reason'], 'NO_USABLE_HAND_FRAME')

    def test_unusable_shoulders_is_explicit_failure(self):
        obs = make_observation()
        obs.usable[:, 11:13] = False
        a, info = prep.features(obs, {'role': 'development_only'})
        self.assertIsNone(a)
        self.assertEqual(info['reason'], 'FEATURE_NORMALIZATION_FAILED')

    def test_fixed_roi_hides_points_and_keeps_timeline(self):
        obs = make_observation()
        masked = prep.roi.apply_roi(obs, [0, 0, 640, 330])
        self.assertFalse(masked.observed[:, 49:54].any())
        np.testing.assert_array_equal(obs.timestamp_us, masked.timestamp_us)
        a, _ = prep.features(masked, {'role': 'development_only'})
        self.assertTrue(np.isfinite(a['x_whole']).all())

    def test_duration_uses_interval_cell_width(self):
        ts = np.arange(10)*100000
        self.assertAlmostEqual(prep.missing_run_seconds(np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 0], bool), ts), .3)

    def test_flags_separate_geometry_and_sensor_and_do_not_use_dev_core(self):
        obs = make_observation()
        masked = prep.roi.apply_roi(obs, [0, 0, 640, 310])
        row = {'role': 'train_candidate', 'intervals': [[0, 1]]}
        flags = prep.visibility_flags(row, obs, masked, masked)
        self.assertTrue(flags)
        self.assertTrue(all(f['comparison'] == 'native_to_mask' for f in flags))
        self.assertEqual(prep.visibility_flags({'role': 'development_only'}, obs, masked, masked), [])


if __name__ == '__main__': unittest.main(verbosity=2)
