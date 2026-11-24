"""Tests for the offline personal-research-reference replay utility.

Synthetic scalars and opaque objects only; no private data is created here.
"""

import unittest

from full_hand_personalization_replay_v1 import (
    METHODS,
    augment_outcome,
    select_research_references,
)

VOCAB = ("apple", "banana", "cherry")
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


class _Opaque:
    """Stand-in for a caller-owned feature sequence."""


def cand(sample_id, sha, label="apple", **over):
    base = {
        "sample_id": sample_id,
        "source_sha256": sha,
        "label": label,
        "partition": "development",
        "positive": True,
        "quality_basis": "user_requested_positive_only",
        "seq": _Opaque(),
    }
    base.update(over)
    return base


def baseline(scores, failed_pairs=0, status="RANKED", **over):
    out = {"status": status, "raw_class_scores": dict(scores), "failed_pairs": failed_pairs}
    out.update(over)
    return out


class SelectResearchReferencesTest(unittest.TestCase):
    def test_leave_one_out_by_sample_id(self):
        bank = [cand("s1", SHA_A), cand("s2", SHA_B), cand("s3", SHA_C, "banana")]
        selected, exclusion = select_research_references(bank, VOCAB, "s2", SHA_B)
        self.assertEqual([c["sample_id"] for c in selected], ["s1", "s3"])
        self.assertEqual(exclusion["excluded_sample_ids"], ["s2"])
        self.assertEqual(exclusion["reason"], "QUERY_ID_OR_SOURCE_HASH")
        self.assertEqual((exclusion["candidates_seen"], exclusion["selected_count"]), (3, 2))
        self.assertIs(selected[0], bank[0])  # original mapping, not a copy

    def test_alias_id_same_hash_is_excluded(self):
        bank = [cand("alias_of_query", SHA_A), cand("s2", SHA_B)]
        selected, exclusion = select_research_references(bank, VOCAB, "query_id", SHA_A)
        self.assertEqual([c["sample_id"] for c in selected], ["s2"])
        self.assertEqual(exclusion["excluded_sample_ids"], ["alias_of_query"])

    def test_hash_comparison_is_case_insensitive(self):
        bank = [cand("s1", "A" * 64)]
        selected, exclusion = select_research_references(bank, VOCAB, "other", "a" * 64)
        self.assertEqual(selected, ())
        self.assertEqual(exclusion["excluded_count"], 1)

    def test_empty_bank_is_allowed(self):
        selected, exclusion = select_research_references([], VOCAB, "q", SHA_A)
        self.assertEqual((selected, exclusion["excluded_sample_ids"]), ((), []))

    def test_duplicate_ids_and_hash_aliases_rejected_before_filtering(self):
        for bank in (
            [cand("s1", SHA_A), cand("s1", SHA_B)],
            [cand("s1", SHA_A), cand("s2", "A" * 64)],
        ):
            with self.assertRaises(ValueError):
                # rejected even though the query would have excluded a duplicate
                select_research_references(bank, VOCAB, "s1", SHA_A)

    def test_metadata_rejections(self):
        bad = [
            cand("s1", SHA_A, partition="final"),
            cand("s1", SHA_A, positive=False),
            cand("s1", SHA_A, quality_basis="expert_verified"),
            cand("s1", SHA_A, label="durian"),
            cand("s1", "zz" + "a" * 62),
            cand("", SHA_A),
        ]
        for entry in bad:
            with self.assertRaises(ValueError):
                select_research_references([entry], VOCAB, "q", SHA_B)

    def test_vocabulary_and_query_validation(self):
        with self.assertRaises(ValueError):
            select_research_references([], (), "q", SHA_A)
        with self.assertRaises(ValueError):
            select_research_references([], ("a", "a"), "q", SHA_A)
        with self.assertRaises(ValueError):
            select_research_references([], VOCAB, "", SHA_A)
        with self.assertRaises(ValueError):
            select_research_references([], VOCAB, "q", "short")

    def test_no_input_mutation(self):
        bank = [cand("s1", SHA_A), cand("s2", SHA_B)]
        snapshot = [dict(c) for c in bank]
        select_research_references(bank, VOCAB, "s1", SHA_A)
        self.assertEqual(bank, snapshot)


class AugmentOutcomeTest(unittest.TestCase):
    def test_dtw_takes_minimum_distance(self):
        base = baseline({"apple": 3.0, "banana": 2.0, "cherry": 5.0})
        out = augment_outcome(base, [{"label": "apple", "ok": True, "value": 0.5}], "dtw_only", VOCAB)
        self.assertEqual(out["raw_class_scores"], {"apple": 0.5, "banana": 2.0, "cherry": 5.0})
        self.assertEqual(out["top1"], "apple")
        self.assertEqual((out["score"], out["q"]), (0.5, -0.5))
        self.assertEqual(out["margin"], 1.5)
        self.assertEqual(out["status"], "RANKED")

    def test_probability_methods_take_maximum(self):
        base = baseline({"apple": 0.2, "banana": 0.4, "cherry": 0.1})
        pairs = [{"label": "cherry", "ok": True, "value": 0.9}]
        for method in ("ml_only", "hybrid"):
            out = augment_outcome(base, pairs, method, VOCAB)
            self.assertEqual(out["raw_class_scores"]["cherry"], 0.9)
            self.assertEqual(out["top1"], "cherry")
            self.assertEqual((out["score"], out["q"]), (0.9, 0.9))
            self.assertAlmostEqual(out["margin"], 0.5)

    def test_ties_break_alphabetically(self):
        dtw = augment_outcome(baseline({"apple": 1.0, "banana": 1.0, "cherry": 4.0}), [], "dtw_only", VOCAB)
        self.assertEqual((dtw["top1"], dtw["margin"]), ("apple", 0.0))
        ml = augment_outcome(baseline({"apple": 0.7, "banana": 0.7, "cherry": 0.1}), [], "ml_only", VOCAB)
        self.assertEqual((ml["top1"], ml["margin"]), ("apple", 0.0))

    def test_empty_references_reproduce_baseline_scores(self):
        base = baseline({"apple": 3.0, "banana": 2.0, "cherry": 5.0}, failed_pairs=2)
        out = augment_outcome(base, [], "dtw_only", VOCAB)
        self.assertEqual(out["raw_class_scores"], base["raw_class_scores"])
        self.assertEqual((out["top1"], out["failed_pairs"]), ("banana", 2))

    def test_empty_extension_reproduces_full_existing_shape_for_all_methods(self):
        cases = {
            "dtw_only": ({"apple": 3.0, "banana": 2.0, "cherry": 5.0}, "banana", 2.0, -2.0, 1.0),
            "ml_only": ({"apple": 0.25, "banana": 0.5, "cherry": 0.125}, "banana", 0.5, 0.5, 0.25),
            "hybrid": ({"apple": 0.75, "banana": 0.5, "cherry": 0.125}, "apple", 0.75, 0.75, 0.25),
        }
        for method, (scores, top1, score, q, margin) in cases.items():
            base = {
                "status": "RANKED",
                "top1": top1,
                "score": score,
                "q": q,
                "margin": margin,
                "raw_class_scores": dict(scores),
                "failed_pairs": 1,
            }
            out = augment_outcome(base, [], method, VOCAB)
            self.assertEqual(out, base, method)
            self.assertEqual(sorted(out), sorted(base), method)

    def test_failed_pairs_counted_and_values_ignored(self):
        base = baseline({"apple": 3.0, "banana": 2.0, "cherry": 5.0}, failed_pairs=1)
        pairs = [
            {"label": "apple", "ok": False},
            {"label": "apple", "ok": False, "value": 0.0},
        ]
        out = augment_outcome(base, pairs, "dtw_only", VOCAB)
        self.assertEqual(out["failed_pairs"], 3)
        self.assertEqual(out["raw_class_scores"]["apple"], 3.0)

    def test_one_class_vocabulary_has_no_margin(self):
        out = augment_outcome(baseline({"apple": 0.4}), [], "hybrid", ("apple",))
        self.assertIsNone(out["margin"])
        self.assertEqual(out["top1"], "apple")

    def test_technical_failure_is_preserved_unchanged(self):
        base = baseline({}, status="TECHNICAL_FAILURE", reason="NO_HAND_TRACK", detail=["x"])
        out = augment_outcome(base, [{"label": "apple", "ok": True, "value": 0.1}], "dtw_only", VOCAB)
        self.assertEqual(out, base)
        self.assertIsNot(out["detail"], base["detail"])  # deep copy
        with self.assertRaises(ValueError):  # method validated before the early return
            augment_outcome(base, [], "no_such_method", VOCAB)

    def test_rejects_bad_baseline_shape_and_scores(self):
        bad_calls = [
            (baseline({"apple": 1.0}), "dtw_only", VOCAB),
            (baseline({"apple": -1.0, "banana": 1.0, "cherry": 1.0}), "dtw_only", VOCAB),
            (baseline({"apple": 1.5, "banana": 0.1, "cherry": 0.1}), "ml_only", VOCAB),
            (baseline({"apple": float("nan"), "banana": 1.0, "cherry": 1.0}), "dtw_only", VOCAB),
            (baseline({"apple": "1.0", "banana": 1.0, "cherry": 1.0}), "dtw_only", VOCAB),
            (baseline({"apple": 1.0, "banana": 1.0, "cherry": 1.0}, failed_pairs=True), "dtw_only", VOCAB),
            (baseline({"apple": 1.0, "banana": 1.0, "cherry": 1.0}, failed_pairs=-1), "dtw_only", VOCAB),
            (baseline({}, status="REJECTED"), "dtw_only", VOCAB),
        ]
        for base, method, vocab in bad_calls:
            with self.assertRaises(ValueError):
                augment_outcome(base, [], method, vocab)

    def test_rejects_malformed_pair_results(self):
        base = baseline({"apple": 3.0, "banana": 2.0, "cherry": 5.0})
        bad_pairs = [
            {"label": "durian", "ok": True, "value": 1.0},
            {"label": "apple", "ok": 1, "value": 1.0},
            {"label": "apple", "ok": True, "value": float("inf")},
            {"label": "apple", "ok": True, "value": None},
            {"label": "apple", "ok": True},
        ]
        for pair in bad_pairs:
            with self.assertRaises(ValueError):
                augment_outcome(base, [pair], "dtw_only", VOCAB)

    def test_inputs_are_not_modified(self):
        base = baseline({"apple": 3.0, "banana": 2.0, "cherry": 5.0})
        snapshot = {"status": base["status"], "raw": dict(base["raw_class_scores"]), "fp": base["failed_pairs"]}
        pairs = [{"label": "apple", "ok": True, "value": 0.5}, {"label": "banana", "ok": False}]
        pairs_snapshot = [dict(p) for p in pairs]
        out = augment_outcome(base, pairs, "dtw_only", VOCAB)
        self.assertEqual(base["raw_class_scores"], snapshot["raw"])
        self.assertEqual(base["failed_pairs"], snapshot["fp"])
        self.assertEqual(pairs, pairs_snapshot)
        self.assertIsNot(out["raw_class_scores"], base["raw_class_scores"])

    def test_methods_constant(self):
        self.assertEqual(METHODS, ("dtw_only", "ml_only", "hybrid"))


if __name__ == "__main__":
    unittest.main()
