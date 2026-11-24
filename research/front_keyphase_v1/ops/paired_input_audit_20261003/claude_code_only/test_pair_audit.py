"""Unit tests for pair_audit pure helpers (synthetic values / temp files only; no model import)."""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pair_audit as pa  # noqa: E402


class RankMetricsTests(unittest.TestCase):
    def test_basic_rank_margin_gap(self):
        m = pa.rank_metrics([0.5, 2.0, -1.0], 0)
        self.assertEqual(m["top1"], 1)
        self.assertFalse(m["correct"])
        self.assertEqual(m["true_rank"], 2)
        self.assertAlmostEqual(m["true_margin"], -1.5)
        self.assertAlmostEqual(m["top2_gap"], 1.5)

    def test_correct_positive_margin(self):
        m = pa.rank_metrics([3.0, 1.0, 2.5], 0)
        self.assertTrue(m["correct"])
        self.assertEqual(m["true_rank"], 1)
        self.assertAlmostEqual(m["true_margin"], 0.5)

    def test_ties_broken_by_class_index(self):
        logits = [1.0, 3.0, 3.0, 0.0]
        a, b = pa.rank_metrics(logits, 1), pa.rank_metrics(logits, 2)
        self.assertEqual((a["top1"], a["correct"], a["true_rank"]), (1, True, 1))
        self.assertEqual((b["top1"], b["correct"], b["true_rank"]), (1, False, 2))
        self.assertEqual(a["true_margin"], 0.0)
        self.assertEqual(b["true_margin"], 0.0)
        self.assertEqual(a["top2_gap"], 0.0)

    def test_rejects_nonfinite(self):
        with self.assertRaises(pa.AuditError):
            pa.rank_metrics([1.0, float("nan")], 0)


class PairCountTests(unittest.TestCase):
    def test_hand_checked_counts(self):
        # Synthetic logits for 5 rows x 3 classes, two arms; true class per row.
        true = [0, 1, 2, 0, 1]
        arm1 = np.array([[3, 1, 0], [0, 2, 1], [1, 2, 0], [0, 0, 5], [0, 4, 1]], float)
        arm2 = np.array([[3, 1, 0], [0, 1, 2], [0, 1, 3], [0, 5, 0], [1, 4, 0]], float)
        m1 = [pa.rank_metrics(l, t) for l, t in zip(arm1, true)]
        m2 = [pa.rank_metrics(l, t) for l, t in zip(arm2, true)]
        # arm1 preds 0,1,1,2,1 -> correct T,T,F,F,T ; arm2 preds 0,2,2,1,1 -> correct T,F,T,F,T
        c = pa.pair_counts([m["correct"] for m in m1], [m["correct"] for m in m2],
                           [m["top1"] for m in m1], [m["top1"] for m in m2])
        self.assertEqual(c, {"n": 5, "both_correct": 2, "first_only_correct": 1, "second_only_correct": 1,
                             "both_wrong": 1, "prediction_same": 2})

    def test_overlap_and_sum_counts(self):
        a = np.array([[1, 1, 0, 0], [1, 0, 1, 0]], bool)
        b = np.array([[1, 0, 1, 0], [1, 0, 0, 1]], bool)
        c = pa.overlap_counts(a, b, (("first", slice(0, 2)),))
        self.assertEqual(c["all"], {"slots": 8, "native": 4, "mediapipe": 4, "native_only": 2, "mp_only": 2, "both": 2})
        self.assertEqual(c["first"], {"slots": 4, "native": 3, "mediapipe": 2, "native_only": 1, "mp_only": 0, "both": 2})
        s = pa.sum_counts([c, c])
        self.assertEqual(s["all"]["both"], 4)
        self.assertEqual(s["first"]["slots"], 8)


class AlignmentTests(unittest.TestCase):
    def test_exact_selection(self):
        idx = pa.select_exact(np.array([10, 12, 11, 13]), np.array([11, 12]))
        self.assertEqual(idx.tolist(), [2, 1])

    def test_absent_id_fails(self):
        with self.assertRaises(pa.AuditError):
            pa.select_exact(np.array([10, 12, 13]), np.array([11]))  # no nearest fallback

    def test_duplicate_observation_id_fails(self):
        with self.assertRaises(pa.AuditError):
            pa.select_exact(np.array([10, 11, 11]), np.array([10, 11]))

    def test_non_integer_ids_fail(self):
        with self.assertRaises(pa.AuditError):
            pa.select_exact(np.array([10.0, 11.0]), np.array([10]))

    def test_order_match(self):
        self.assertEqual(pa.check_order(np.array([3, 5, 9]), np.array([3, 5, 9])).tolist(), [3, 5, 9])

    def test_mismatched_order_fails(self):
        with self.assertRaises(pa.AuditError):
            pa.check_order(np.array([3, 5, 9]), np.array([3, 9, 5]))
        with self.assertRaises(pa.AuditError):
            pa.check_order(np.array([3, 5]), np.array([3, 5, 9]))

    def test_duplicate_or_decreasing_native_ids_fail(self):
        with self.assertRaises(pa.AuditError):
            pa.check_order(np.array([3, 3, 4]), np.array([3, 3, 4]))
        with self.assertRaises(pa.AuditError):
            pa.check_order(np.array([5, 4]), np.array([5, 4]))

    def test_times_strictly_increasing(self):
        pa.strictly_increasing(np.array([0, 33333, 66666], np.int64), "t")
        with self.assertRaises(pa.AuditError):
            pa.strictly_increasing(np.array([0, 0, 1], np.int64), "t")


class GeometryTests(unittest.TestCase):
    def test_masked_l2_ignores_unavailable(self):
        a = np.zeros((2, 3, 2))
        b = np.zeros((2, 3, 2))
        b[0, 0] = [3.0, 4.0]       # valid in both -> 5
        b[0, 1] = [1e6, 1e6]       # invalid in a -> ignored
        b[1, 2] = [np.nan, 0.0]    # invalid in b -> ignored even though NaN
        ma = np.array([[1, 0, 1], [1, 1, 1]], bool)
        mb = np.array([[1, 1, 1], [1, 1, 0]], bool)
        v = pa.l2_values(a, b, ma, mb)
        self.assertEqual(sorted(v.tolist()), [0.0, 0.0, 0.0, 5.0])
        s = pa.l2_stats(v)
        self.assertEqual(s["count"], 4)
        self.assertAlmostEqual(s["meanL2"], 1.25)
        self.assertAlmostEqual(s["medianL2"], 0.0)

    def test_empty_stats_null(self):
        v = pa.l2_values(np.zeros((1, 2, 2)), np.ones((1, 2, 2)), np.zeros((1, 2), bool), np.ones((1, 2), bool))
        self.assertEqual(pa.l2_stats(v), {"count": 0, "meanL2": None, "medianL2": None, "p95L2": None})

    def test_verify_rebuild_nan_scales_and_mismatch(self):
        prep = {"x": np.zeros((2, 280), np.float32), "body_valid": np.ones((2, 51), bool),
                "hand_valid": np.zeros((2, 42), bool), "body_scale": np.array(1.0),
                "hand_scales": np.array([np.nan, 2.0])}
        clip = {"x": np.zeros((2, 280), np.float32), "frames": 2, "body_valid": np.ones((2, 51), bool),
                "hand_valid": np.zeros((2, 42), bool), "body_scale": 1.0,
                "hand_scales": {"left": float("nan"), "right": 2.0}}
        pa.verify_rebuild(clip, prep, 1e-6)
        clip["hand_valid"] = np.ones((2, 42), bool)
        with self.assertRaises(pa.AuditError):
            pa.verify_rebuild(clip, prep, 1e-6)


class MaskTests(unittest.TestCase):
    def test_float32_exact_binary_accepted(self):
        m = pa.as_bool(np.array([[0.0, 1.0], [1.0, 0.0]], np.float32))
        self.assertEqual(m.dtype, bool)
        self.assertEqual(m.tolist(), [[False, True], [True, False]])

    def test_bool_and_integer_accepted(self):
        self.assertEqual(pa.as_bool(np.array([True, False])).tolist(), [True, False])
        self.assertEqual(pa.as_bool(np.array([0, 1, 1], np.int8)).tolist(), [False, True, True])
        self.assertEqual(pa.as_bool(np.array([1, 0], np.float64)).tolist(), [True, False])

    def test_fractional_and_out_of_range_rejected(self):
        for bad in (np.array([0.0, 0.5], np.float32), np.array([0, 2], np.int64), np.array([-1.0, 1.0])):
            with self.assertRaises(pa.AuditError):
                pa.as_bool(bad)

    def test_nonfinite_rejected(self):
        for v in (np.nan, np.inf, -np.inf):
            with self.assertRaises(pa.AuditError):
                pa.as_bool(np.array([1.0, v], np.float32))

    def test_non_numeric_rejected(self):
        with self.assertRaises(pa.AuditError):
            pa.as_bool(np.array(["0", "1"]))

    def test_verify_rebuild_float32_masks(self):
        # build_channels / prepared NPZ give float32 0/1 masks; equal masks pass, differing masks fail.
        bv = np.ones((2, 51), np.float32)
        hv = np.zeros((2, 42), np.float32)
        hv[0, :21] = 1.0
        prep = {"x": np.zeros((2, 280), np.float32), "body_valid": bv, "hand_valid": hv,
                "body_scale": np.array(1.0), "hand_scales": np.array([1.0, 2.0])}
        clip = {"x": np.zeros((2, 280), np.float32), "frames": 2, "body_valid": bv.copy(),
                "hand_valid": hv.astype(bool), "body_scale": 1.0, "hand_scales": {"left": 1.0, "right": 2.0}}
        pa.verify_rebuild(clip, prep, 1e-6)
        clip["hand_valid"] = np.ones((2, 42), np.float32)
        with self.assertRaises(pa.AuditError):
            pa.verify_rebuild(clip, prep, 1e-6)
        clip["hand_valid"] = np.full((2, 42), 0.5, np.float32)
        with self.assertRaises(pa.AuditError):
            pa.verify_rebuild(clip, prep, 1e-6)


class ManifestIdentityTests(unittest.TestCase):
    ROWS = [{"clip_id": "c1", "label": "A", "signer_id": "1"},
            {"clip_id": "c2", "label": "B", "signer_id": "2"},
            {"clip_id": "c3", "label": "A", "signer_id": 2}]
    EXP = {"rows": 3, "clips": 3, "labels": 2, "signers": [1, 2]}

    def test_exact_match_passes_with_canonical_string_ids(self):
        got = pa.check_manifest_identity(self.ROWS, self.EXP)
        self.assertEqual(got, {"rows": 3, "clips": 3, "labels": 2, "signers": ["1", "2"]})
        pa.check_manifest_identity(self.ROWS, dict(self.EXP, signers=["2", "1"]))

    def test_right_count_wrong_ids_fails(self):
        with self.assertRaises(pa.AuditError):
            pa.check_manifest_identity(self.ROWS, dict(self.EXP, signers=[1, 3]))

    def test_subset_or_superset_fails(self):
        for s in ([1], [1, 2, 3]):
            with self.assertRaises(pa.AuditError):
                pa.check_manifest_identity(self.ROWS, dict(self.EXP, signers=s))

    def test_integer_count_config_rejected(self):
        # Old count-only form must fail closed rather than be accepted: identity matters.
        with self.assertRaises(pa.AuditError):
            pa.check_manifest_identity(self.ROWS, dict(self.EXP, signers=2))

    def test_empty_or_duplicate_expected_ids_rejected(self):
        for s in ([], [1, 2, 2], [1, "1", 2]):
            with self.assertRaises(pa.AuditError):
                pa.check_manifest_identity(self.ROWS, dict(self.EXP, signers=s))

    def test_counts_still_exact(self):
        for k, v in (("rows", 4), ("clips", 2), ("labels", 3)):
            with self.assertRaises(pa.AuditError):
                pa.check_manifest_identity(self.ROWS, dict(self.EXP, **{k: v}))

    def test_missing_expected_key_fails(self):
        exp = dict(self.EXP)
        del exp["signers"]
        with self.assertRaises(pa.AuditError):
            pa.check_manifest_identity(self.ROWS, exp)


class ImmutabilityTests(unittest.TestCase):
    def test_detects_modification(self):
        with tempfile.TemporaryDirectory() as d:
            p, q = os.path.join(d, "a.bin"), os.path.join(d, "b.bin")
            for path in (p, q):
                with open(path, "wb") as f:
                    f.write(b"original")
            before = pa.hash_files([p, q, p])  # duplicates collapse
            self.assertEqual(len(before), 2)
            self.assertEqual(pa.check_immutable(before, pa.hash_files([p, q])), [])
            with open(p, "ab") as f:
                f.write(b"!")
            self.assertEqual(pa.check_immutable(before, pa.hash_files([p, q])), [pa.norm(p)])


if __name__ == "__main__":
    unittest.main()
