"""Pure synthetic checks; no media, service, MediaPipe, or model imports.

Run with Python -B. Only the two source modules and synthetic NumPy arrays are
used. The runner may capture stdout to retain a first-failure receipt.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "partial_hand_reconstruction_20261003" / "implementation"))
sys.path.insert(0, str(HERE))
import handgap as HG
import gate_candidate as G


SHAPE = np.column_stack((np.arange(21) % 5, np.arange(21) // 5)) * .1


def put_hand(xy, present, frame, side, centre):
    hand = HG.HANDS[side]
    xy[frame, hand] = SHAPE + np.asarray(centre)
    present[frame, hand] = 1
    xy[frame, G.POSE_WRISTS[side]] = xy[frame, hand.start]
    present[frame, G.POSE_WRISTS[side]] = 1


def remove_hand(xy, present, frame, side):
    xy[frame, HG.HANDS[side]] = np.nan
    present[frame, HG.HANDS[side]] = 0


def clip(n=3, dtype=np.float64, present_dtype=np.bool_):
    xy = np.full((n, 67, 2), np.nan, dtype=dtype)
    present = np.zeros((n, 67), dtype=present_dtype)
    for frame in range(n):
        put_hand(xy, present, frame, "left", (-4 + .1 * frame, .05 * frame))
        put_hand(xy, present, frame, "right", (4 + .05 * frame, .025 * frame))
    ts = np.arange(n, dtype=np.int64) * 20000
    if n == 3:
        ts[-1] = 60000
    remove_hand(xy, present, 1, "left")
    return xy, present, ts


def rejected(test, arrays, reason):
    result = G.restore(*arrays)
    test.assertTrue(result["identity"])
    test.assertIn(reason, result["decisions"][0]["reasons"])
    return result


class GateTests(unittest.TestCase):
    def test_accepted_uses_exact_old_values_and_api(self):
        arrays = clip(dtype=np.float32, present_dtype=np.uint8)
        old, got = HG.restore(*arrays), G.restore(*arrays)
        self.assertEqual(set(got), set(old) | {"decisions"})
        for key in ("xy", "present", "ts", "observed", "estimated"):
            self.assertEqual(got[key].dtype, old[key].dtype)
            self.assertEqual(got[key].tobytes(), old[key].tobytes())
        self.assertFalse(got["identity"])
        self.assertTrue(got["decisions"][0]["accepted"])
        self.assertEqual(got["decisions"][0]["reasons"], [])
        # Uneven timestamps: old interpolation remains 1/3, not a midpoint.
        np.testing.assert_allclose(got["xy"][1, 25],
                                   arrays[0][0, 25] * (2 / 3) + arrays[0][2, 25] / 3)

    def test_readonly_inputs_and_measured_absent_bytes_preserved(self):
        arrays = clip(present_dtype=np.float32)
        arrays[0][0, 0] = [-0.0, 123.25]  # Absent arbitrary values are preserved.
        arrays[0].view(np.uint64)[0, 1, 0] = 0x7ff8000000000042
        before = [a.tobytes() for a in arrays]
        for a in arrays:
            a.flags.writeable = False
        got = G.restore(*arrays)
        self.assertEqual(before, [a.tobytes() for a in arrays])
        keep = ~got["estimated"]
        self.assertEqual(got["xy"][keep].tobytes(), arrays[0][keep].tobytes())
        self.assertEqual(got["present"][keep].tobytes(), arrays[1][keep].tobytes())
        self.assertEqual(got["ts"].tobytes(), before[2])
        for key, original in zip(("xy", "present", "ts"), arrays):
            self.assertFalse(np.shares_memory(got[key], original))

    def test_rejected_event_preserves_even_absent_sentinels(self):
        xy, pr, ts = clip()
        xy[1, 25:46] = -4321.125
        xy[0, G.POSE_WRISTS["left"]] = [99, 0]
        got = rejected(self, (xy, pr, ts), "mismatch_pose_ownership_before")
        self.assertEqual(got["xy"].tobytes(), xy.tobytes())
        self.assertEqual(got["present"].tobytes(), pr.tobytes())
        self.assertFalse(got["estimated"].any())

    def test_missing_either_pose_wrist_at_either_endpoint(self):
        for row, name in ((0, "before"), (2, "after")):
            for side in G.SIDES:
                with self.subTest(row=row, side=side):
                    xy, pr, ts = clip()
                    pr[row, G.POSE_WRISTS[side]] = 0
                    xy[row, G.POSE_WRISTS[side]] = np.nan
                    rejected(self, (xy, pr, ts), "insufficient_pose_wrists_" + name)

    def test_pose_wrong_side_and_tie_reject(self):
        for tie in (False, True):
            with self.subTest(tie=tie):
                xy, pr, ts = clip()
                xy[2, 7] = xy[2, 4] if tie else [100, 100]
                rejected(self, (xy, pr, ts), "mismatch_pose_ownership_after")

    def test_temporal_forward_mismatch_despite_consistent_pose(self):
        xy, pr, ts = clip()
        put_hand(xy, pr, 2, "left", (8, 0))
        put_hand(xy, pr, 2, "right", (-4, 0))
        got = rejected(self, (xy, pr, ts), "mismatch_temporal_forward")
        self.assertFalse(any("pose_ownership" in r for r in got["decisions"][0]["reasons"]))

    def test_temporal_backward_is_checked_independently(self):
        xy, pr, ts = clip()
        for row, side, centre in ((0, "left", (0, 0)), (0, "right", (2, 0)),
                                  (2, "left", (2.1, 0)), (2, "right", (10, 0))):
            put_hand(xy, pr, row, side, centre)
        got = rejected(self, (xy, pr, ts), "mismatch_temporal_backward")
        self.assertNotIn("mismatch_temporal_forward", got["decisions"][0]["reasons"])

    def test_temporal_tie_rejects(self):
        xy, pr, ts = clip()
        put_hand(xy, pr, 0, "left", (0, 0))
        put_hand(xy, pr, 2, "left", (2, 0))
        put_hand(xy, pr, 2, "right", (-2, 0))
        # Identical palms make exact distance ties independent of decimal offsets.
        xy[0, 25:46] = [0, 0]
        xy[2, 25:46] = [2, 0]
        xy[2, 46:67] = [-2, 0]
        rejected(self, (xy, pr, ts), "mismatch_temporal_forward")

    def test_partial_endpoint_opposite_is_not_absence(self):
        for row, name in ((0, "before"), (2, "after")):
            with self.subTest(row=row):
                xy, pr, ts = clip()
                remove_hand(xy, pr, row, "right")
                pr[row, 47] = 1  # Only a non-palm joint exists.
                xy[row, 47] = [4, 0]
                rejected(self, (xy, pr, ts), "insufficient_opposite_palm_" + name)

    def test_absent_opposite_has_no_competitor_or_witness(self):
        xy, pr, ts = clip()
        for row in range(3):
            remove_hand(xy, pr, row, "right")
        got = G.restore(xy, pr, ts)
        self.assertTrue(got["decisions"][0]["accepted"])
        self.assertEqual(got["decisions"][0]["metrics"]["opposite_gap_state"], "absent")

    def test_complete_palm_allows_missing_nonpalm_opposite_joints(self):
        xy, pr, ts = clip()
        for row in range(3):
            pr[row, 47] = 0
            xy[row, 47] = np.nan
        self.assertTrue(G.restore(xy, pr, ts)["decisions"][0]["accepted"])

    def test_gap_opposite_incomplete_palm_rejects(self):
        xy, pr, ts = clip()
        pr[1, 51] = 0
        xy[1, 51] = np.nan
        rejected(self, (xy, pr, ts), "insufficient_gap_opposite_palm")

    def test_gap_witness_requires_both_endpoint_opposite_palms(self):
        for row, name in ((0, "before"), (2, "after")):
            with self.subTest(row=row):
                xy, pr, ts = clip()
                remove_hand(xy, pr, row, "right")
                rejected(self, (xy, pr, ts), "insufficient_gap_witness_endpoint_" + name)

    def test_gap_opposite_nearer_target_rejects_despite_good_endpoints(self):
        xy, pr, ts = clip()
        xy[1, 46:67] = xy[0, 25:46]
        got = rejected(self, (xy, pr, ts), "mismatch_gap_witness_before")
        reasons = got["decisions"][0]["reasons"]
        self.assertIn("mismatch_gap_witness_after", reasons)
        self.assertFalse(any("temporal" in r or "pose_ownership" in r for r in reasons))

    def test_gap_witness_tie_rejects(self):
        xy, pr, ts = clip()
        for row in (0, 2):
            put_hand(xy, pr, row, "left", (-4, 0))
            put_hand(xy, pr, row, "right", (4, 0))
            xy[row, 25:46] = [-4, 0]
            xy[row, 46:67] = [4, 0]
        xy[1, 46:67] = [0, 0]
        rejected(self, (xy, pr, ts), "mismatch_gap_witness_before")

    def test_all_rejection_reasons_are_recorded_and_json_safe(self):
        xy, pr, ts = clip()
        pr[0, 7] = 0
        pr[1, 51] = 0
        got = G.restore(xy, pr, ts)
        reasons = got["decisions"][0]["reasons"]
        self.assertIn("insufficient_pose_wrists_before", reasons)
        self.assertIn("insufficient_gap_opposite_palm", reasons)
        json.dumps(got["decisions"], allow_nan=False)

    def test_original_eligibility_limits_not_extended(self):
        for case in ("boundary", "consecutive", "partial", "endpoint", "time", "one_frame"):
            with self.subTest(case=case):
                xy, pr, ts = clip(n=4 if case == "consecutive" else 3)
                if case == "boundary":
                    put_hand(xy, pr, 1, "left", (-3.9, .05))
                    remove_hand(xy, pr, 0, "left")
                    remove_hand(xy, pr, 2, "left")
                elif case == "consecutive":
                    remove_hand(xy, pr, 2, "left")
                elif case == "partial":
                    pr[1, 25] = 1
                    xy[1, 25] = [-3.9, .05]
                elif case == "endpoint":
                    pr[0, 26] = 0
                elif case == "time":
                    ts[2] = 66669
                else:
                    xy, pr, ts = xy[:1], pr[:1], ts[:1]
                got = G.restore(xy, pr, ts)
                self.assertTrue(got["identity"])
                self.assertEqual(got["decisions"], [])
                self.assertEqual(got["xy"].tobytes(), xy.tobytes())
        xy, pr, ts = clip()
        ts[2] = 66668
        self.assertFalse(G.restore(xy, pr, ts)["identity"])

    def test_rotation_reflection_translation_and_scale_invariance(self):
        transforms = ((np.array([[0., -1.], [1., 0.]]), np.zeros(2), 1.),
                      (np.diag([-1., 1.]), np.zeros(2), 1.),
                      (np.eye(2), np.array([123., -47.]), 1.),
                      (np.eye(2), np.zeros(2), 7.5))
        for rejected_case in (False, True):
            arrays = clip()
            if rejected_case:
                arrays[0][1, 46:67] = arrays[0][0, 25:46]
            baseline = G.restore(*arrays)
            for matrix, offset, scale in transforms:
                with self.subTest(rejected=rejected_case, matrix=matrix.tolist(), scale=scale):
                    transformed = arrays[0] @ matrix.T * scale + offset
                    got = G.restore(transformed, arrays[1], arrays[2])
                    np.testing.assert_array_equal(got["estimated"], baseline["estimated"])
                    self.assertEqual([d["reasons"] for d in got["decisions"]],
                                     [d["reasons"] for d in baseline["decisions"]])
                    np.testing.assert_allclose(got["xy"], baseline["xy"] @ matrix.T * scale + offset,
                                               rtol=1e-13, atol=1e-13, equal_nan=True)

    def test_metric_is_mean_euclidean_not_centroid_or_rms(self):
        a = np.zeros((5, 2))
        b = np.array([[0, 0], [1, 0], [-2, 0], [0, 3], [0, -4]], dtype=np.float64)
        self.assertEqual(G._mean_distance(a, b), 2.)
        a32 = np.zeros((5, 2), dtype=np.float32)
        b32 = np.tile(np.array([3e20, 4e20], dtype=np.float32), (5, 1))
        expected = float(np.hypot(float(b32[0, 0]), float(b32[0, 1])))
        self.assertEqual(G._mean_distance(a32, b32), expected)

    def test_original_only_order_and_time_reversal(self):
        arrays = clip(n=5)
        remove_hand(arrays[0], arrays[1], 3, "right")
        got = G.restore(*arrays)
        self.assertEqual([(d["frame"], d["side"], d["accepted"]) for d in got["decisions"]],
                         [(1, "left", True), (3, "right", True)])
        with patch.object(HG, "SIDES", tuple(reversed(HG.SIDES))):
            reverse_order = G.restore(*arrays)
        self.assertEqual(got["xy"].tobytes(), reverse_order["xy"].tobytes())
        self.assertEqual(got["decisions"], reverse_order["decisions"])
        reversed_input = (arrays[0][::-1], arrays[1][::-1], arrays[2][-1] - arrays[2][::-1])
        rev = G.restore(*reversed_input)
        np.testing.assert_array_equal(rev["estimated"][::-1], got["estimated"])
        np.testing.assert_allclose(rev["xy"][::-1], got["xy"], equal_nan=True)
        # Adjacent opposite-side holes must remain original missing evidence.
        adjacent = clip(n=5)
        remove_hand(adjacent[0], adjacent[1], 2, "right")
        adj = G.restore(*adjacent)
        self.assertTrue(adj["identity"])
        self.assertEqual(len(adj["decisions"]), 2)
        self.assertTrue(all(any(r.startswith("insufficient_gap_witness_endpoint")
                                 for r in d["reasons"]) for d in adj["decisions"]))

    def test_base_restore_called_once_and_only_accepted_mask_written(self):
        arrays = clip(n=5)
        remove_hand(arrays[0], arrays[1], 3, "right")
        arrays[0][4, 4] = [100, 100]  # Reject the later right-hand fill only.
        old_restore = HG.restore
        with patch.object(HG, "restore", wraps=old_restore) as original:
            got = G.restore(*arrays)
            original.assert_called_once()
        self.assertEqual([d["accepted"] for d in got["decisions"]], [True, False])
        self.assertEqual(int(got["estimated"].sum()), 21)
        unchanged = ~got["estimated"]
        self.assertEqual(got["xy"][unchanged].tobytes(), arrays[0][unchanged].tobytes())

    def test_old_validation_failures_are_preserved(self):
        for case in ("present", "timestamps", "nonfinite_present", "shape"):
            with self.subTest(case=case):
                xy, pr, ts = clip(present_dtype=np.float64)
                if case == "present":
                    pr[0, 0] = .5
                elif case == "timestamps":
                    ts[2] = ts[1]
                elif case == "nonfinite_present":
                    xy[0, 25] = np.inf
                else:
                    xy = xy[:, :66]
                with self.assertRaises(ValueError):
                    G.restore(xy, pr, ts)

    def test_nonfinite_distance_fails_closed_without_nonfinite_json(self):
        xy, pr, ts = clip()
        for row in range(3):
            put_hand(xy, pr, row, "left", (-1e308, 0))
            put_hand(xy, pr, row, "right", (1e308, 0))
        remove_hand(xy, pr, 1, "left")
        got = G.restore(xy, pr, ts)
        self.assertTrue(got["identity"])
        self.assertTrue(any(r.startswith("insufficient_nonfinite_distance")
                            for r in got["decisions"][0]["reasons"]))
        json.dumps(got["decisions"], allow_nan=False)
        self.assertEqual(got["xy"].tobytes(), xy.tobytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
