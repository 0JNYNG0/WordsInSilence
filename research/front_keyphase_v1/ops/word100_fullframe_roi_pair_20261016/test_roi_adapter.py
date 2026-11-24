"""Offline adapter checks; no files, detector, camera, training or model calls."""
from __future__ import annotations

import copy
import unittest

import numpy as np

from roi_adapter import apply_roi, build_roi_channels
from wis_lab.core.observation import Observation, empty_frames
from pilot.data import PilotDataError


def observation(frames=4):
    arrays = empty_frames(frames)
    arrays["timestamp_us"][:] = np.arange(frames, dtype=np.int64) * 41000 + 7000
    arrays["frame_id"][:] = np.arange(frames, dtype=np.int64) * 2 + 9
    arrays["xy_px"][:] = (50, 50)
    arrays["xy_px"][:, 11] = (35, 30)
    arrays["xy_px"][:, 12] = (65, 30)
    for hand_start in (33, 54):
        arrays["xy_px"][:, hand_start:hand_start + 21, 0] = np.arange(21) + 40
        arrays["xy_px"][:, hand_start:hand_start + 21, 1] = np.arange(21) + 45
    arrays["observed"][:] = True
    arrays["usable"][:] = True
    arrays["conf"][:] = .9
    arrays["hand_score"][:] = .95
    return Observation(**arrays, sidecar={"clip_id": "synthetic", "fps": 25.0})


class RoiAdapterTests(unittest.TestCase):
    def test_strict_bounds_for_all_75_points(self):
        obs = observation(1)
        values = [(0, 0), (99, 99), (100, 40), (40, 100), (-1, 40), (40, -1)]
        for point in range(75):
            obs.xy_px[0, point] = values[point % len(values)]
        got = apply_roi(obs, (0, 0, 100, 100))
        expected = np.array([i % len(values) < 2 for i in range(75)])
        np.testing.assert_array_equal(got.observed[0], expected)
        np.testing.assert_array_equal(got.usable[0], expected)
        self.assertTrue(np.isnan(got.xy_px[0, ~expected]).all())

    def test_nearest_float32_inside_upper_boundary(self):
        obs = observation(1)
        obs.xy_px[0, 0] = (np.nextafter(np.float32(100), np.float32(0)), 50)
        obs.xy_px[0, 1] = (100, 50)
        got = apply_roi(obs, (0, 0, 100, 100))
        self.assertTrue(got.observed[0, 0])
        self.assertFalse(got.observed[0, 1])

    def test_origin_translates_retained_coordinates_to_source_space(self):
        obs = observation()
        got = apply_roi(obs, (200, 300, 300, 400), source_origin=(200, 300))
        np.testing.assert_array_equal(got.xy_px, obs.xy_px + np.array([200, 300], np.float32))
        self.assertEqual(got.sidecar["roi_mask"]["coordinate_space"], "source_pixels")
        self.assertEqual(got.sidecar["roi_mask"]["source_origin"], [200., 300.])

    def test_source_origin_and_fullframe_produce_identical_observations(self):
        crop = observation()
        native = copy.deepcopy(crop)
        native.xy_px += np.array([200, 300], np.float32)
        a = apply_roi(crop, (200, 300, 300, 400), (200, 300))
        b = apply_roi(native, (200, 300, 300, 400))
        for key in ("xy_px", "usable", "observed", "conf", "hand_score"):
            np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
        np.testing.assert_array_equal(build_roi_channels(a)["x"], build_roi_channels(b)["x"])

    def test_nonfinite_coordinates_are_absent(self):
        obs = observation(1)
        obs.xy_px[0, 0] = (np.nan, 50)
        obs.xy_px[0, 1] = (50, np.inf)
        got = apply_roi(obs, (0, 0, 100, 100))
        self.assertFalse(got.observed[0, :2].any())
        self.assertTrue(np.isnan(got.xy_px[0, :2]).all())
        self.assertTrue(np.isnan(got.conf[0, :2]).all())

    def test_prior_absence_and_quality_are_never_promoted(self):
        obs = observation(1)
        obs.xy_px[0, 0] = np.nan
        obs.observed[0, 0] = obs.usable[0, 0] = False
        obs.usable[0, 1] = False
        got = apply_roi(obs, (0, 0, 100, 100))
        self.assertFalse(got.observed[0, 0])
        self.assertTrue(got.observed[0, 1])
        self.assertFalse(got.usable[0, 1])

    def test_partial_hand_and_completely_removed_hand_scores(self):
        obs = observation(1)
        obs.xy_px[0, 33:54] = (200, 200)
        obs.xy_px[0, 54] = (200, 200)
        got = apply_roi(obs, (0, 0, 100, 100))
        self.assertFalse(got.observed[0, 33:54].any())
        self.assertTrue(np.isnan(got.hand_score[0, 0]))
        self.assertEqual(got.observed[0, 54:75].sum(), 20)
        self.assertEqual(got.hand_score[0, 1], obs.hand_score[0, 1])

    def test_missing_wrist_invalidates_local_but_not_other_body_relative_points(self):
        obs = observation()
        obs.xy_px[:, 33] = (200, 200)
        got = build_roi_channels(apply_roi(obs, (0, 0, 100, 100)))
        self.assertFalse(got["hand_valid"][:, :21].any())
        self.assertTrue((got["body_valid"][:, 10:30] == 1).all())
        self.assertTrue((got["x"][:, 102:144] == 0).all())

    def test_missing_wrist_in_one_frame_invalidates_only_that_frame_local(self):
        obs = observation()
        obs.xy_px[2, 33] = (200, 200)
        got = build_roi_channels(apply_roi(obs, (0, 0, 100, 100)))
        self.assertFalse(got["hand_valid"][2, :21].any())
        self.assertTrue((got["hand_valid"][[0, 1, 3], :21] == 1).all())

    def test_no_trim_or_time_replacement_and_missing_coordinate_zero_fill(self):
        obs = observation(5)
        obs.timestamp_us[:] = [7000, 41000, 100000, 130000, 300000]
        obs.xy_px[[0, 4], 33:75] = (200, 200)
        masked = apply_roi(obs, (0, 0, 100, 100))
        got = build_roi_channels(masked)
        self.assertEqual(got["x"].shape, (5, 280))
        np.testing.assert_array_equal(got["frame_id"], obs.frame_id)
        np.testing.assert_array_equal(got["timestamp_us"], obs.timestamp_us)
        np.testing.assert_array_equal(masked.timestamp_us, obs.timestamp_us)
        np.testing.assert_allclose(got["x"][:, -1], [0, .034, .059, .03, .17])
        global_xy = got["x"][:, :102].reshape(-1, 51, 2)
        local_xy = got["x"][:, 102:186].reshape(-1, 42, 2)
        self.assertTrue((global_xy[~got["body_valid"].astype(bool)] == 0).all())
        self.assertTrue((local_xy[~got["hand_valid"].astype(bool)] == 0).all())

    def test_outside_values_do_not_change_features_or_scales(self):
        a = observation()
        a.xy_px[0, 11:13] = (200, 200)
        a.xy_px[1, 33:54] = (200, 200)
        a.xy_px[:, 23:25] = (200, 200)
        b = copy.deepcopy(a)
        hidden = (a.xy_px[..., 0] >= 100)
        b.xy_px[hidden] = (-5000, -9000)
        b.conf[hidden] = .001
        ca = build_roi_channels(apply_roi(a, (0, 0, 100, 100)))
        cb = build_roi_channels(apply_roi(b, (0, 0, 100, 100)))
        np.testing.assert_array_equal(ca["x"], cb["x"])
        self.assertEqual(ca["body_scale"], cb["body_scale"])
        self.assertEqual(ca["hand_scales"], cb["hand_scales"])

    def test_normalisation_uses_only_roi_retained_shoulders(self):
        obs = observation()
        obs.xy_px[0, 11] = (-10000, 30)
        obs.xy_px[0, 12] = (10000, 30)
        got = build_roi_channels(apply_roi(obs, (0, 0, 100, 100)))
        self.assertEqual(got["body_scale"], 30.0)
        self.assertFalse(got["body_valid"][0].any())

    def test_translation_preserves_channels(self):
        a = observation()
        a.xy_px[0, 33] = (200, 200)
        b = copy.deepcopy(a)
        b.xy_px += np.array([128, 256], np.float32)
        ca = build_roi_channels(apply_roi(a, (0, 0, 100, 100)))
        cb = build_roi_channels(apply_roi(b, (128, 256, 228, 356)))
        np.testing.assert_array_equal(ca["x"], cb["x"])

    def test_input_and_metadata_unchanged_and_output_not_aliased(self):
        obs = observation()
        obs.sidecar.update(raw_xy=[[123, 456]], quality={"body_scale": 1234}, observation_sha256="old")
        before = copy.deepcopy(obs)
        got = apply_roi(obs, (0, 0, 100, 100))
        self.assertEqual(obs.sidecar, before.sidecar)
        self.assertNotIn("raw_xy", got.sidecar)
        self.assertNotIn("quality", got.sidecar)
        self.assertNotIn("observation_sha256", got.sidecar)
        self.assertEqual(got.sidecar["clip_id"], "synthetic")
        for name in ("xy_px", "timestamp_us", "frame_id", "observed", "usable", "conf", "hand_score"):
            np.testing.assert_array_equal(getattr(obs, name), getattr(before, name))
            self.assertFalse(np.shares_memory(getattr(obs, name), getattr(got, name)))

    def test_no_shoulders_is_explicit_failure_not_filled_geometry(self):
        obs = observation()
        obs.xy_px[:, 11:13] = (200, 200)
        with self.assertRaises(PilotDataError):
            build_roi_channels(apply_roi(obs, (0, 0, 100, 100)))

    def test_empty_clip_preserved_and_channel_builder_fails_explicitly(self):
        obs = Observation(**empty_frames(0))
        got = apply_roi(obs, (0, 0, 100, 100))
        self.assertEqual(got.n_frames, 0)
        with self.assertRaises(PilotDataError):
            build_roi_channels(got)

    def test_unmasked_or_tampered_observation_rejected(self):
        obs = observation()
        with self.assertRaises(ValueError):
            build_roi_channels(obs)
        got = apply_roi(obs, (0, 0, 100, 100))
        got.xy_px[0, 0] = (200, 200)
        with self.assertRaises(ValueError):
            build_roi_channels(got)

    def test_invalid_geometry_rejected(self):
        obs = observation()
        for box in [(0, 0, 0, 100), (0, 100, 100, 0), (0, 0, np.nan, 100), (0, 1, 2)]:
            with self.subTest(box=box), self.assertRaises(ValueError):
                apply_roi(obs, box)
        for origin in [(1,), (0, np.inf)]:
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                apply_roi(obs, (0, 0, 100, 100), origin)


if __name__ == "__main__":
    unittest.main(verbosity=2)
