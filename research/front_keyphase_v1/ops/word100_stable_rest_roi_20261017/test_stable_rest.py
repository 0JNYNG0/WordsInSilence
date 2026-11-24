"""Synthetic offline tests; no source clips, detector, models, or camera calls."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

import numpy as np

from stable_rest import calibrate_rest, suppress_rest_fragments
from wis_lab.core.observation import Observation, empty_frames

_PREVIOUS = Path(__file__).resolve().parent.parent / "word100_fullframe_roi_pair_20261016"
if str(_PREVIOUS) not in sys.path:
    sys.path.insert(0, str(_PREVIOUS))
from roi_adapter import apply_roi


def full_source(frames=30):
    arrays = empty_frames(frames)
    arrays["timestamp_us"][:] = 7_000 + np.arange(frames, dtype=np.int64) * 50_000
    arrays["frame_id"][:] = 19 + np.arange(frames, dtype=np.int64) * 3
    arrays["observed"][:] = True
    arrays["usable"][:] = True
    arrays["conf"][:] = .9
    arrays["hand_score"][:] = .95
    arrays["xy_px"][:] = (300, 150)
    for shoulder, elbow, pose_wrist, hand_start, x in ((11, 13, 15, 33, 200), (12, 14, 16, 54, 400)):
        arrays["xy_px"][:, shoulder] = (x, 100)
        arrays["xy_px"][:, elbow] = (x, 230)
        arrays["xy_px"][:, pose_wrist] = (x, 395)
        arrays["xy_px"][:, hand_start:hand_start + 21] = (x, 420)
        arrays["xy_px"][:, hand_start] = (x, 400)
    obs = Observation(**arrays, sidecar={"clip_id": "synthetic_only", "frame_width": 600,
                                        "frame_height": 600})
    obs.validate()
    return obs


def calibration():
    return calibrate_rest(full_source(), 600, 600)


def wrist_fragments(frames=14):
    obs = full_source(frames)
    obs.xy_px[:, (33, 54), 1] = 395
    return obs


def run_guard(source, cal=None):
    cal = calibration() if cal is None else cal
    masked = apply_roi(source, cal["box_source_pixels"])
    out, trace = suppress_rest_fragments(masked, cal)
    return masked, out, trace


def make_missing(obs, frame, point):
    obs.observed[frame, point] = False
    obs.usable[frame, point] = False
    obs.xy_px[frame, point] = np.nan
    obs.conf[frame, point] = np.nan


class CalibrationTests(unittest.TestCase):
    def test_earliest_window_fixed_boundary_and_anchors(self):
        obs = full_source()
        before = copy.deepcopy(obs)
        result = calibrate_rest(obs, 600, 600)
        self.assertEqual(result["status"], "CALIBRATED")
        self.assertEqual(result["selected_frame_indices"], list(range(8)))
        self.assertEqual(result["duration_us"], 350_000)
        self.assertEqual(result["box_source_pixels"], [0, 0, 600, 396])
        self.assertEqual(result["shoulder_width"], 200)
        self.assertEqual(result["median_shoulder_center"], [300, 100])
        self.assertEqual(result["wrist_anchors_normalized"], {"left": [-.5, 1.5], "right": [.5, 1.5]})
        self.assertEqual(result["selected_frame_ids"], obs.frame_id[:8].tolist())
        self.assertEqual(result["selected_timestamp_us"], obs.timestamp_us[:8].tolist())
        np.testing.assert_array_equal(obs.xy_px, before.xy_px)
        self.assertEqual(obs.sidecar, before.sidecar)

    def test_unstable_start_is_skipped_without_using_later_clip(self):
        obs = full_source()
        obs.xy_px[:3, (33, 54), 1] += np.array([40, 20, 30], np.float32)[:, None]
        result = calibrate_rest(obs, 600, 600)
        self.assertEqual(result["selected_frame_indices"], list(range(3, 11)))
        obs.xy_px[21:, (33, 54), 1] = 590
        self.assertEqual(calibrate_rest(obs, 600, 600), result)

    def test_no_qualifying_window_has_no_fallback_or_late_calibration(self):
        obs = full_source(50)
        obs.xy_px[:21, (33, 54), 1] += (np.arange(21) % 2 * 60 - 30)[:, None]
        result = calibrate_rest(obs, 600, 600)
        self.assertEqual(result["status"], "NOT_CALIBRATED")
        self.assertNotIn("bottom", result)
        self.assertNotIn("box_source_pixels", result)

    def test_300ms_is_insufficient_without_eight_actual_frames(self):
        result = calibrate_rest(full_source(7), 600, 600)
        self.assertEqual(result["status"], "NOT_CALIBRATED")

    def test_missing_required_frame_cannot_be_skipped(self):
        obs = full_source(12)
        make_missing(obs, 3, 14)
        result = calibrate_rest(obs, 600, 600)
        self.assertEqual(result["selected_frame_indices"], list(range(4, 12)))

    def test_unusable_or_nonfinite_required_points_reject_window(self):
        for failure in ("unusable", "nonfinite"):
            with self.subTest(failure=failure):
                obs = full_source(8)
                if failure == "unusable":
                    obs.usable[3, 33] = False
                else:
                    obs.xy_px[3, 33, 0] = np.nan
                self.assertEqual(calibrate_rest(obs, 600, 600)["status"], "NOT_CALIBRATED")

    def test_large_timestamp_gap_rejects_contiguous_window(self):
        obs = full_source(8)
        obs.timestamp_us[4:] += 100_000
        self.assertEqual(calibrate_rest(obs, 600, 600)["status"], "NOT_CALIBRATED")

    def test_static_bent_arm_is_not_preparation_pose(self):
        obs = full_source()
        obs.xy_px[:, 13, 1] = 380
        self.assertEqual(calibrate_rest(obs, 600, 600)["status"], "NOT_CALIBRATED")

    def test_shoulders_must_be_stable_as_well_as_wrists(self):
        obs = full_source()
        obs.xy_px[:, (11, 12), 0] += (np.arange(obs.n_frames) % 2 * 60 - 30)[:, None]
        self.assertEqual(calibrate_rest(obs, 600, 600)["status"], "NOT_CALIBRATED")

    def test_masked_observation_cannot_supply_preparation(self):
        with self.assertRaisesRegex(ValueError, "full-source"):
            calibrate_rest(apply_roi(full_source(), (0, 0, 600, 600)), 600, 600)

    def test_duplicate_or_decreasing_timestamps_are_rejected(self):
        for delta in (0, -1):
            with self.subTest(delta=delta):
                obs = full_source()
                obs.timestamp_us[3] = obs.timestamp_us[2] + delta
                with self.assertRaisesRegex(AssertionError, "strictly increasing"):
                    calibrate_rest(obs, 600, 600)


class GuardTests(unittest.TestCase):
    def test_persistent_wrist_only_suppression_has_no_backdating(self):
        masked, out, trace = run_guard(wrist_fragments())
        self.assertTrue(out.observed[:6, (33, 54)].all())
        self.assertFalse(out.observed[6:, (33, 54)].any())
        self.assertEqual(trace["frames"][5]["left"]["dwell_us"], 250_000)
        self.assertEqual(trace["frames"][6]["left"]["dwell_us"], 300_000)
        self.assertEqual(trace["suppressed_frame_counts"], {"left": 8, "right": 8})
        self.assertTrue(np.isnan(out.xy_px[6:, (33, 54)]).all())
        self.assertTrue(np.isnan(out.hand_score[6:]).all())
        for attr in ("xy_px", "observed", "usable", "conf"):
            np.testing.assert_array_equal(getattr(out, attr)[:, :33], getattr(masked, attr)[:, :33])
        for attr in ("timestamp_us", "frame_id"):
            np.testing.assert_array_equal(getattr(out, attr), getattr(masked, attr))
        self.assertTrue(masked.observed[:, (33, 54)].all())
        self.assertNotIn("rest_wrist_guard", masked.sidecar)
        out.validate()

    def test_brief_lowered_wrist_is_retained(self):
        masked, out, trace = run_guard(wrist_fragments(6))
        np.testing.assert_array_equal(out.observed, masked.observed)
        self.assertEqual(trace["suppressed_frame_counts"], {"left": 0, "right": 0})

    def test_guard_is_prefix_causal(self):
        source = wrist_fragments(20)
        _, full_out, full_trace = run_guard(source)
        for length in (3, 6, 7, 12):
            prefix = copy.deepcopy(source)
            for attr in ("xy_px", "observed", "usable", "conf", "hand_score", "timestamp_us", "frame_id"):
                setattr(prefix, attr, getattr(prefix, attr)[:length].copy())
            _, out, trace = run_guard(prefix)
            for attr in ("xy_px", "observed", "usable", "conf", "hand_score"):
                np.testing.assert_array_equal(getattr(out, attr), getattr(full_out, attr)[:length])
            self.assertEqual(trace["frames"], full_trace["frames"][:length])

    def test_observed_unusable_finger_protects_and_restarts_dwell(self):
        source = wrist_fragments(14)
        source.xy_px[6, 34] = (200, 390)
        source.usable[6, 34] = False
        masked, out, trace = run_guard(source)
        self.assertTrue(out.observed[6, 34])
        self.assertTrue(out.observed[:13, 33].all())
        self.assertFalse(out.observed[13, 33])
        self.assertFalse(out.observed[6:, 54].any())
        self.assertEqual(trace["frames"][6]["left"]["reason"], "OBSERVED_FINGER_PROTECTS_HAND")
        np.testing.assert_array_equal(out.xy_px[:, 34:54], masked.xy_px[:, 34:54])

    def test_missing_wrist_resets_without_interpolation(self):
        source = wrist_fragments(14)
        make_missing(source, 6, 33)
        _, out, trace = run_guard(source)
        self.assertTrue(out.observed[:6, 33].all())
        self.assertFalse(out.observed[6, 33])
        self.assertTrue(out.observed[7:13, 33].all())
        self.assertFalse(out.observed[13, 33])
        self.assertEqual(trace["frames"][7]["left"]["dwell_us"], 0)

    def test_missing_body_evidence_resets_only_affected_hand(self):
        source = wrist_fragments(14)
        make_missing(source, 6, 13)
        _, out, trace = run_guard(source)
        self.assertTrue(out.observed[:13, 33].all())
        self.assertFalse(out.observed[13, 33])
        self.assertFalse(out.observed[6:, 54].any())
        self.assertEqual(trace["frames"][6]["left"]["reason"], "VISIBLE_BODY_EVIDENCE_MISSING")

    def test_large_time_gap_restarts_at_current_frame(self):
        source = wrist_fragments(10)
        source.timestamp_us[3:] += 100_000
        _, out, trace = run_guard(source)
        self.assertTrue(out.observed[:9, 33].all())
        self.assertFalse(out.observed[9, 33])
        self.assertEqual(trace["frames"][3]["left"]["reason"], "TIME_GAP_RESET")
        self.assertEqual(trace["frames"][3]["left"]["dwell_us"], 0)

    def test_wrist_or_elbow_motion_restarts_from_first_position(self):
        for moving_point in (33, 13):
            with self.subTest(moving_point=moving_point):
                source = wrist_fragments(12)
                source.xy_px[5:, moving_point, 0] += 5
                _, out, trace = run_guard(source)
                self.assertTrue(out.observed[:11, 33].all())
                self.assertFalse(out.observed[11, 33])
                self.assertEqual(trace["frames"][5]["left"]["reason"], "MOVEMENT_RESET")

    def test_static_bent_elbow_near_boundary_remains(self):
        source = wrist_fragments()
        source.xy_px[:, 13, 1] = 380
        _, out, trace = run_guard(source)
        self.assertTrue(out.observed[:, 33].all())
        self.assertEqual(trace["frames"][10]["left"]["reason"], "ARM_NOT_DOWN")

    def test_outside_band_or_anchor_wrist_remains(self):
        for position, reason in (((200, 389), "OUTSIDE_BOTTOM_BAND"),
                                 ((220, 395), "OUTSIDE_REST_ANCHOR")):
            with self.subTest(position=position):
                source = wrist_fragments()
                source.xy_px[:, 33] = position
                _, out, trace = run_guard(source)
                self.assertTrue(out.observed[:, 33].all())
                self.assertEqual(trace["frames"][10]["left"]["reason"], reason)

    def test_hidden_points_cannot_change_any_guard_decision(self):
        source = wrist_fragments()
        changed = copy.deepcopy(source)
        changed.xy_px[:, 34:54] = (5000, 5000)
        changed.xy_px[:, 55:75] = (-5000, -5000)
        _, a, ta = run_guard(source)
        _, b, tb = run_guard(changed)
        self.assertEqual(ta, tb)
        for attr in ("xy_px", "observed", "usable", "conf", "hand_score"):
            np.testing.assert_array_equal(getattr(a, attr), getattr(b, attr))

    def test_not_calibrated_preserves_all_input_observations(self):
        source = wrist_fragments()
        masked = apply_roi(source, (0, 0, 600, 396))
        failed = calibrate_rest(full_source(7), 600, 600)
        out, trace = suppress_rest_fragments(masked, failed)
        for attr in ("xy_px", "observed", "usable", "conf", "hand_score", "timestamp_us", "frame_id"):
            np.testing.assert_array_equal(getattr(out, attr), getattr(masked, attr))
        self.assertEqual(trace["frames"][0]["left"]["reason"], "NOT_CALIBRATED")

    def test_requires_roi_and_exact_fixed_boundary(self):
        with self.assertRaisesRegex(ValueError, "apply_roi"):
            suppress_rest_fragments(wrist_fragments(), calibration())
        with self.assertRaisesRegex(ValueError, "differs"):
            suppress_rest_fragments(apply_roi(wrist_fragments(), (0, 0, 600, 397)), calibration())

    def test_duplicate_or_decreasing_timestamps_are_rejected(self):
        for delta in (0, -1):
            with self.subTest(delta=delta):
                masked = apply_roi(wrist_fragments(), (0, 0, 600, 396))
                masked.timestamp_us[3] = masked.timestamp_us[2] + delta
                with self.assertRaisesRegex(AssertionError, "strictly increasing"):
                    suppress_rest_fragments(masked, calibration())


if __name__ == "__main__":
    unittest.main(verbosity=2)
