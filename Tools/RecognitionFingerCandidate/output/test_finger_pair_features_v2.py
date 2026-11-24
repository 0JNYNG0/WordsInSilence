#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synthetic checks for finger_pair_features_v2.

AUTHORED BUT NOT RUN. The author had no execution tool in this session, so no pass/fail
result is claimed anywhere in this packet. The operator runs this once and keeps the
first failures.

Every input here is a synthetic array built in this file. No capture, no learner material,
no profile, no joint recording, no personal log and no model parameter is read. The
synthetic hand variants are feature test material; they are not 'correct' or 'wrong' sign
labels and they say nothing about recognition accuracy or teaching quality.

    python -m unittest test_finger_pair_features_v2 -v
"""

from __future__ import annotations

import copy
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import finger_pair_features_v2 as v2                  # noqa: E402
import full_hand_pair_features_v1 as baseline         # noqa: E402

FRAMES = 32
LEFT_BASE, RIGHT_BASE = 4, 25


# ---- synthetic material -----------------------------------------------------

def base_coordinates(frames: int = FRAMES) -> np.ndarray:
    """A deterministic masked-2D [T, 46, 3] capture-shaped array. Analytic, not sampled."""
    coords = np.zeros((frames, baseline.SOURCE_POINT_COUNT, 3), dtype=np.float64)
    for t in range(frames):
        dx, dy = 0.010 * t, 0.005 * t
        coords[t, 0] = (0.00 + dx, 0.00 + dy, 0.0)
        coords[t, 1] = (0.10 + dx, 0.00 + dy, 0.0)
        coords[t, 2] = (0.00 + dx, 0.10 + dy, 0.0)
        coords[t, 3] = (0.10 + dx, 0.10 + dy, 0.0)
        for base, origin in ((LEFT_BASE, -0.50), (RIGHT_BASE, 0.50)):
            coords[t, base] = (origin + dx, dy, 0.0)
            for joint in range(1, baseline.HAND_JOINT_COUNT):
                finger, link = (joint - 1) // 4, (joint - 1) % 4
                coords[t, base + joint] = (
                    origin + dx + 0.05 * (finger + 1) + 0.002 * t,
                    dy + 0.04 * (link + 1),
                    0.0)
    return coords


def full_mask(frames: int = FRAMES) -> np.ndarray:
    return np.ones((frames, baseline.SOURCE_POINT_COUNT), dtype=bool)


def shifted(coords: np.ndarray, delta: float) -> np.ndarray:
    """Translate xy only; z stays exactly 0, the masked-2D contract."""
    out = coords.copy()
    out[:, :, :2] += delta
    return out


def sequence(coords: np.ndarray, mask: np.ndarray):
    return baseline.build_feature_sequence(coords, mask)


def joint_indices(base: int, joints) -> list:
    return [base + joint for joint in joints]


def finger_slot(name: str, statistic: str) -> int:
    """Index of one finger value inside the 30-value tail."""
    return v2.FINGER_FEATURE_NAMES.index("finger_%s_%s" % (name, statistic))


def tail(features: np.ndarray, prefix: int) -> np.ndarray:
    return features[prefix:]


def finger_oracle(a, b, path) -> np.ndarray:
    """Independent, deliberately naive recomputation of the 30 values from ONE path.

    Written from the specification, not from the implementation: plain python loops over
    the path rows, the 2 local blocks and the 4 columns of each finger."""
    values = []
    for low in (baseline.GROUPS[3][1], baseline.GROUPS[4][1]):
        for finger_index in range(5):
            start = low + 4 * finger_index
            pool = []
            for i, j in path:
                for column in range(start, start + 4):
                    if a.mask[i, column] and b.mask[j, column]:
                        delta = a.points[i, column, :2] - b.points[j, column, :2]
                        pool.append(float(np.hypot(delta[0], delta[1])))
            if pool:
                values.extend([float(np.mean(pool)), float(np.percentile(pool, 90))])
            else:
                values.extend([float("nan"), float("nan")])
            values.append(len(pool) / float(len(path) * 4))
    return np.asarray(values, dtype=np.float64)


# ---- risk 1: width, name order, exact baseline prefix ------------------------

class TestWidthNamesAndPrefix(unittest.TestCase):

    def test_counts_and_schema(self):
        self.assertEqual(v2.FIXED_FEATURE_COUNT, 45)
        self.assertEqual(v2.ALIGNED_FEATURE_COUNT, 48)
        self.assertEqual(v2.FINGER_FEATURE_COUNT, 30)
        self.assertEqual(len(set(v2.FIXED_FEATURE_NAMES)), 45)
        self.assertEqual(len(set(v2.ALIGNED_FEATURE_NAMES)), 48)
        self.assertNotEqual(v2.SCHEMA_ID, baseline.SCHEMA_ID)
        self.assertEqual(v2.SEQUENCE_SCHEMA_ID, baseline.SCHEMA_ID)

    def test_name_prefix_and_finger_order(self):
        self.assertEqual(v2.FIXED_FEATURE_NAMES[:15], baseline.FIXED_FEATURE_NAMES)
        self.assertEqual(v2.ALIGNED_FEATURE_NAMES[:18], baseline.ALIGNED_FEATURE_NAMES)
        self.assertEqual(v2.FIXED_FEATURE_NAMES[15:], v2.FINGER_FEATURE_NAMES)
        self.assertEqual(v2.ALIGNED_FEATURE_NAMES[18:], v2.FINGER_FEATURE_NAMES)
        expected = tuple(
            "finger_%s_%s_%s" % (hand, finger, statistic)
            for hand in ("left", "right")
            for finger in ("thumb", "index", "middle", "ring", "pinky")
            for statistic in ("residual_mean", "residual_p90", "joint_valid_fraction"))
        self.assertEqual(v2.FINGER_FEATURE_NAMES, expected)

    def test_finger_point_ranges_cover_the_local_blocks_without_the_wrist(self):
        ranges = [(low, high) for _, low, high in v2.FINGER_POINT_RANGES]
        self.assertEqual(ranges[:5], [(46, 50), (50, 54), (54, 58), (58, 62), (62, 66)])
        self.assertEqual(ranges[5:], [(66, 70), (70, 74), (74, 78), (78, 82), (82, 86)])
        covered = sorted(index for low, high in ranges for index in range(low, high))
        self.assertEqual(covered, list(range(46, 86)))

    def test_prefix_is_bit_identical_to_the_baseline(self):
        coords, mask = base_coordinates(), full_mask()
        other = coords.copy()
        other[:, LEFT_BASE + 6:LEFT_BASE + 9, 0] += 0.12
        mask_b = mask.copy()
        mask_b[:, RIGHT_BASE:RIGHT_BASE + 21] = False     # forces NaN into the prefix
        a, b = sequence(coords, mask), sequence(other, mask_b)

        mine = v2.fixed_time_features(a, b)
        theirs = baseline.fixed_time_features(a, b)
        self.assertTrue(mine.ok and theirs.ok)
        self.assertEqual(mine.features.size, 45)
        self.assertEqual(mine.features.dtype, np.float64)
        self.assertTrue(np.array_equal(mine.features[:15], theirs.features, equal_nan=True))
        self.assertTrue(np.isnan(theirs.features).any(), "prefix NaN case not exercised")

        mine = v2.aligned_pair_features(a, b)
        theirs = baseline.aligned_pair_features(a, b)
        self.assertTrue(mine.ok and theirs.ok)
        self.assertEqual(mine.features.size, 48)
        self.assertTrue(np.array_equal(mine.features[:18], theirs.features, equal_nan=True))
        self.assertEqual(mine.path_length, theirs.path_length)
        self.assertEqual(mine.dtw_cost, theirs.dtw_cost)


# ---- risk 2: a thumb error and an index error are not interchangeable --------

class TestFingerLevelDiscrimination(unittest.TestCase):
    """Swapping which finger carries which error leaves the whole-group descriptors the
    same pooled multiset, so only the finger values can tell the two apart."""

    def setUp(self):
        coords, mask = base_coordinates(), full_mask()
        thumb = joint_indices(LEFT_BASE, (2, 3, 4))        # no MCP joint is touched, so
        index = joint_indices(LEFT_BASE, (6, 7, 8))        # the hand scale is identical
        small, large = 0.05, 0.20
        a = coords.copy()
        a[:, thumb, 0] += small
        a[:, index, 0] += large
        b = coords.copy()
        b[:, thumb, 0] += large
        b[:, index, 0] += small
        self.reference = sequence(coords, mask)
        self.a = sequence(a, mask)
        self.b = sequence(b, mask)

    def test_hand_scales_are_unchanged_by_the_swap(self):
        for hand in ("left", "right"):
            self.assertAlmostEqual(self.a.hand_scales[hand], self.b.hand_scales[hand], 12)
            self.assertAlmostEqual(self.a.hand_scales[hand],
                                   self.reference.hand_scales[hand], 12)

    def test_group_prefix_matches_but_finger_values_separate(self):
        left = v2.fixed_time_features(self.a, self.reference)
        right = v2.fixed_time_features(self.b, self.reference)
        self.assertTrue(left.ok and right.ok)
        np.testing.assert_allclose(left.features[:15], right.features[:15],
                                   rtol=1e-11, atol=1e-11, equal_nan=True,
                                   err_msg="the swap must be invisible to the groups")
        for name in ("left_thumb", "left_index"):
            slot = 15 + finger_slot(name, "residual_mean")
            self.assertGreater(abs(left.features[slot] - right.features[slot]), 1e-3,
                               "%s must separate the two errors" % name)
        thumb = 15 + finger_slot("left_thumb", "residual_mean")
        index = 15 + finger_slot("left_index", "residual_mean")
        self.assertLess(left.features[thumb], left.features[index])
        self.assertGreater(right.features[thumb], right.features[index])
        for name in ("left_middle", "left_ring", "left_pinky", "right_thumb"):
            slot = 15 + finger_slot(name, "residual_mean")
            self.assertAlmostEqual(left.features[slot], right.features[slot], 12)

    def test_alignment_path_also_separates_the_two(self):
        left = v2.aligned_pair_features(self.a, self.reference)
        right = v2.aligned_pair_features(self.b, self.reference)
        self.assertTrue(left.ok and right.ok)
        for name in ("left_thumb", "left_index"):
            slot = 18 + finger_slot(name, "residual_mean")
            self.assertGreater(abs(left.features[slot] - right.features[slot]), 1e-3)


# ---- risk 3: missingness is never agreement ---------------------------------

class TestMissingnessIsNotAgreement(unittest.TestCase):

    def _values(self, name, features, prefix):
        return tuple(features[prefix + finger_slot(name, statistic)]
                     for statistic in ("residual_mean", "residual_p90",
                                       "joint_valid_fraction"))

    def test_one_hand_only_is_not_rejected_and_reports_the_absence(self):
        coords, mask = base_coordinates(), full_mask()
        mask[:, RIGHT_BASE:RIGHT_BASE + 21] = False
        a = sequence(coords, mask)
        b = sequence(shifted(coords, 0.01), mask)
        for result, prefix in ((v2.fixed_time_features(a, b), 15),
                               (v2.aligned_pair_features(a, b), 18)):
            self.assertTrue(result.ok, result.failure)
            for finger in ("thumb", "index", "middle", "ring", "pinky"):
                mean, p90, coverage = self._values("right_%s" % finger,
                                                   result.features, prefix)
                self.assertTrue(np.isnan(mean) and np.isnan(p90))
                self.assertEqual(coverage, 0.0)
                mean, p90, coverage = self._values("left_%s" % finger,
                                                   result.features, prefix)
                self.assertTrue(np.isfinite(mean) and np.isfinite(p90))
                self.assertAlmostEqual(coverage, 1.0, 12)

    def test_a_single_missing_finger_is_absent_not_matching(self):
        coords = base_coordinates()
        mask_a = full_mask()
        mask_a[:, joint_indices(LEFT_BASE, (17, 18, 19, 20))] = False     # pinky gone
        a, b = sequence(coords, mask_a), sequence(coords, full_mask())
        result = v2.fixed_time_features(a, b)
        self.assertTrue(result.ok, result.failure)
        mean, p90, coverage = self._values("left_pinky", result.features, 15)
        self.assertTrue(np.isnan(mean) and np.isnan(p90))
        self.assertEqual(coverage, 0.0)
        mean, _, coverage = self._values("left_ring", result.features, 15)
        self.assertTrue(np.isfinite(mean))
        self.assertAlmostEqual(coverage, 1.0, 12)

    def test_partly_missing_time_gives_partial_coverage(self):
        coords = base_coordinates()
        mask_a = full_mask()
        mask_a[0:8, joint_indices(LEFT_BASE, (9, 10, 11, 12))] = False
        a, b = sequence(coords, mask_a), sequence(coords, full_mask())
        result = v2.fixed_time_features(a, b)
        self.assertTrue(result.ok, result.failure)
        mean, p90, coverage = self._values("left_middle", result.features, 15)
        self.assertTrue(np.isfinite(mean) and np.isfinite(p90))
        self.assertGreater(coverage, 0.0)
        self.assertLess(coverage, 1.0)
        for finger in ("thumb", "index", "ring", "pinky"):
            self.assertAlmostEqual(
                self._values("left_%s" % finger, result.features, 15)[2], 1.0, 12)

    def test_coverage_stays_within_zero_and_one_everywhere(self):
        coords = base_coordinates()
        mask_a = full_mask()
        mask_a[3:9, 30:40] = False
        mask_a[:, joint_indices(LEFT_BASE, (2, 3))] = False
        a, b = sequence(coords, mask_a), sequence(shifted(coords, 0.02), full_mask())
        for result, prefix in ((v2.fixed_time_features(a, b), 15),
                               (v2.aligned_pair_features(a, b), 18)):
            self.assertTrue(result.ok, result.failure)
            coverage = result.features[prefix + 2::3]
            self.assertEqual(coverage.size, 10)
            self.assertTrue(np.all(coverage >= 0.0) and np.all(coverage <= 1.0))

    def test_masked_slots_may_hold_nan_or_garbage_with_the_same_result(self):
        coords, mask = base_coordinates(), full_mask()
        mask[5:12, joint_indices(LEFT_BASE, (6, 7, 8))] = False
        mask[:, RIGHT_BASE + 20] = False
        with_nan, with_garbage = coords.copy(), coords.copy()
        with_nan[~mask] = np.nan
        with_garbage[~mask] = 1.0e6
        other = sequence(shifted(coords, 0.01), full_mask())
        for builder in (v2.fixed_time_features, v2.aligned_pair_features):
            one = builder(sequence(with_nan, mask), other)
            two = builder(sequence(with_garbage, mask), other)
            self.assertTrue(one.ok and two.ok)
            self.assertTrue(np.array_equal(one.features, two.features, equal_nan=True),
                            "unread masked slots changed the result")


# ---- risk 4: the existing failure meanings survive --------------------------

class TestExistingFailuresPreserved(unittest.TestCase):

    def _compare_with_baseline(self, a, b):
        for mine, theirs in ((v2.fixed_time_features(a, b),
                              baseline.fixed_time_features(a, b)),
                             (v2.aligned_pair_features(a, b),
                              baseline.aligned_pair_features(a, b))):
            self.assertEqual(mine.ok, theirs.ok)
            self.assertEqual(mine.failure, theirs.failure)
            if not mine.ok:
                self.assertIsNone(mine.features)
        return (v2.fixed_time_features(a, b), v2.aligned_pair_features(a, b))

    def test_everything_missing(self):
        coords = base_coordinates()
        empty = np.zeros((FRAMES, baseline.SOURCE_POINT_COUNT), dtype=bool)
        a = sequence(coords, empty)
        b = sequence(coords, full_mask())
        self.assertEqual(a.failure, "ALL_POINTS_MISSING")
        fixed, aligned = self._compare_with_baseline(a, b)
        self.assertFalse(fixed.ok)
        self.assertFalse(aligned.ok)
        self.assertEqual(fixed.failure, baseline.NO_SHARED_HAND_SHAPE)

    def test_no_shared_hand_shape_when_the_hands_are_disjoint(self):
        coords = base_coordinates()
        left_only, right_only = full_mask(), full_mask()
        left_only[:, RIGHT_BASE:RIGHT_BASE + 21] = False
        right_only[:, LEFT_BASE:LEFT_BASE + 21] = False
        a, b = sequence(coords, left_only), sequence(coords, right_only)
        fixed, aligned = self._compare_with_baseline(a, b)
        self.assertEqual(fixed.failure, baseline.NO_SHARED_HAND_SHAPE)
        self.assertEqual(aligned.failure, baseline.NO_SHARED_HAND_SHAPE)

    def test_no_valid_dtw_path_when_a_middle_frame_has_nothing(self):
        coords = base_coordinates()
        gap = full_mask()
        gap[14:17, :] = False                     # blanks resampled rows 14 and 15
        a, b = sequence(coords, gap), sequence(coords, full_mask())
        fixed, aligned = self._compare_with_baseline(a, b)
        self.assertEqual(aligned.failure, "NO_VALID_DTW_PATH")
        self.assertIsNone(aligned.features)


# ---- risk 5: DTW call budget and one shared path -----------------------------

class TestOneSharedPathAndDtwBudget(unittest.TestCase):

    def setUp(self):
        coords, mask = base_coordinates(), full_mask()
        other = coords.copy()
        other[:, LEFT_BASE + 3, 0] += 0.07
        mask_b = mask.copy()
        mask_b[10:14, joint_indices(RIGHT_BASE, (18, 19))] = False
        self.a, self.b = sequence(coords, mask), sequence(other, mask_b)

    def _instrument(self):
        """Count DTW calls and capture the path each descriptor block was handed."""
        state = {"dtw": 0, "group_path": None, "finger_path": None}
        real_dtw = baseline.dtw_path
        real_group = baseline.descriptors_on_path
        real_finger = v2.finger_descriptors

        def counting_dtw(cost):
            state["dtw"] += 1
            return real_dtw(cost)

        def watching_group(geometry, path):
            state["group_path"] = path
            return real_group(geometry, path)

        def watching_finger(geometry, path):
            state["finger_path"] = path
            return real_finger(geometry, path)

        baseline.dtw_path = counting_dtw
        baseline.descriptors_on_path = watching_group
        v2.finger_descriptors = watching_finger

        def restore():
            baseline.dtw_path = real_dtw
            baseline.descriptors_on_path = real_group
            v2.finger_descriptors = real_finger

        return state, restore

    def test_ml_path_calls_dtw_zero_times(self):
        state, restore = self._instrument()
        try:
            result = v2.fixed_time_features(self.a, self.b)
        finally:
            restore()
        self.assertTrue(result.ok, result.failure)
        self.assertEqual(state["dtw"], 0)
        self.assertEqual(result.path_length, baseline.RESAMPLE_T)
        self.assertTrue(np.isnan(result.dtw_cost))
        self.assertIs(state["finger_path"], state["group_path"])
        self.assertEqual(list(state["finger_path"]),
                         baseline.diagonal_path(baseline.RESAMPLE_T))

    def test_hybrid_calls_dtw_once_and_all_ten_fingers_reuse_that_path(self):
        state, restore = self._instrument()
        try:
            result = v2.aligned_pair_features(self.a, self.b)
        finally:
            restore()
        self.assertTrue(result.ok, result.failure)
        self.assertEqual(state["dtw"], 1, "the hybrid must align exactly once")
        self.assertIs(state["finger_path"], state["group_path"],
                      "the fingers must read the one shared path object")
        path = state["finger_path"]
        self.assertEqual(len(path), result.path_length)
        expected = finger_oracle(self.a, self.b, path)
        np.testing.assert_allclose(tail(result.features, 18), expected,
                                   rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_the_fingers_agree_with_the_oracle_on_the_diagonal_too(self):
        result = v2.fixed_time_features(self.a, self.b)
        self.assertTrue(result.ok, result.failure)
        expected = finger_oracle(self.a, self.b,
                                 baseline.diagonal_path(baseline.RESAMPLE_T))
        np.testing.assert_allclose(tail(result.features, 15), expected,
                                   rtol=1e-12, atol=1e-12, equal_nan=True)


# ---- risk 6: the inputs are untouched and the output repeats -----------------

class TestPurityAndDeterminism(unittest.TestCase):

    def setUp(self):
        coords, mask = base_coordinates(), full_mask()
        mask[6:11, joint_indices(LEFT_BASE, (13, 14))] = False
        self.a = sequence(coords, mask)
        self.b = sequence(shifted(coords, 0.03), full_mask())

    def test_inputs_are_not_modified(self):
        before = [(copy.deepcopy(s.points), copy.deepcopy(s.mask))
                  for s in (self.a, self.b)]
        v2.fixed_time_features(self.a, self.b)
        v2.aligned_pair_features(self.a, self.b)
        for (points, mask), s in zip(before, (self.a, self.b)):
            self.assertTrue(np.array_equal(points, s.points, equal_nan=True))
            self.assertTrue(np.array_equal(mask, s.mask))

    def test_repeated_calls_agree_exactly(self):
        for builder in (v2.fixed_time_features, v2.aligned_pair_features):
            one, two = builder(self.a, self.b), builder(self.a, self.b)
            self.assertTrue(one.ok and two.ok)
            self.assertTrue(np.array_equal(one.features, two.features, equal_nan=True))
            self.assertEqual(one.path_length, two.path_length)

    def test_the_result_does_not_alias_a_previous_result(self):
        one = v2.fixed_time_features(self.a, self.b)
        two = v2.fixed_time_features(self.a, self.b)
        self.assertIsNot(one.features, two.features)


# ---- input validation, kept apart from genuine pair failures ----------------

class TestInputValidation(unittest.TestCase):

    def setUp(self):
        self.a = sequence(base_coordinates(), full_mask())
        self.b = sequence(shifted(base_coordinates(), 0.01), full_mask())

    def _expect_refusal(self, damaged):
        for builder in (v2.fixed_time_features, v2.aligned_pair_features):
            with self.assertRaises(v2.InvalidFeatureSequence):
                builder(damaged, self.b)
            with self.assertRaises(v2.InvalidFeatureSequence):
                builder(self.a, damaged)

    def test_refusal_is_a_feature_error(self):
        self.assertTrue(issubclass(v2.InvalidFeatureSequence, baseline.FeatureError))

    def test_wrong_schema(self):
        damaged = copy.deepcopy(self.a)
        damaged.schema = "something_else_v9"
        self._expect_refusal(damaged)

    def test_wrong_shape(self):
        damaged = copy.deepcopy(self.a)
        damaged.points = damaged.points[:, :40, :]
        self._expect_refusal(damaged)
        damaged = copy.deepcopy(self.a)
        damaged.mask = damaged.mask[:10]
        self._expect_refusal(damaged)

    def test_non_finite_xy_at_a_valid_slot(self):
        damaged = copy.deepcopy(self.a)
        self.assertTrue(damaged.mask[3, 7])
        damaged.points[3, 7, 0] = np.nan
        self._expect_refusal(damaged)
        damaged = copy.deepcopy(self.a)
        damaged.points[3, 7, 1] = np.inf
        self._expect_refusal(damaged)

    def test_non_zero_z_at_a_valid_slot(self):
        damaged = copy.deepcopy(self.a)
        self.assertTrue(damaged.mask[2, 50])
        damaged.points[2, 50, 2] = 0.5
        self._expect_refusal(damaged)

    def test_non_boolean_mask(self):
        damaged = copy.deepcopy(self.a)
        damaged.mask = damaged.mask.astype(np.float64)
        self._expect_refusal(damaged)

    def test_not_a_feature_sequence(self):
        with self.assertRaises(v2.InvalidFeatureSequence):
            v2.fixed_time_features(np.zeros((32, 86, 3)), self.b)

    def test_a_pair_failure_is_returned_not_raised(self):
        empty = sequence(base_coordinates(),
                         np.zeros((FRAMES, baseline.SOURCE_POINT_COUNT), dtype=bool))
        result = v2.fixed_time_features(empty, self.b)
        self.assertFalse(result.ok)
        self.assertEqual(result.failure, baseline.NO_SHARED_HAND_SHAPE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
