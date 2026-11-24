#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synthetic checks for finger_candidate_experiment_v2.

AUTHORED BUT NOT RUN. The author had no execution tool, so no pass/fail, count, skip or
exit code is claimed anywhere in this packet. The operator runs this once and keeps the
first failures.

Every array here is synthetic and built in this file. No capture, learner material,
profile, joint recording, personal log or model parameter is read. A FitSpy replaces the
estimator in every run, so nothing is ever fitted on synthetic arrays and no number
produced here can be mistaken for candidate performance. sklearn is therefore not needed.

    python -m unittest test_finger_candidate_experiment_v2 -v
"""

from __future__ import annotations

import copy
import inspect
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import finger_candidate_experiment_v2 as f10        # noqa: E402
import finger_pair_features_v2 as f9                # noqa: E402
import full_hand_pair_features_v1 as fh             # noqa: E402
import full_hand_comparison_v1 as cmp               # noqa: E402

FRAMES = 24
LEFT_BASE, RIGHT_BASE = 4, 25


# ---- synthetic material -----------------------------------------------------

def capture(label_offset: float, signer_offset: float, frames: int = FRAMES):
    """A deterministic masked-2D [T, 46, 3] capture. z stays exactly 0."""
    coords = np.zeros((frames, fh.SOURCE_POINT_COUNT, 3), dtype=np.float64)
    for t in range(frames):
        dx, dy = 0.010 * t + signer_offset, 0.005 * t
        coords[t, 0] = (dx, dy, 0.0)
        coords[t, 1] = (0.10 + dx, dy, 0.0)
        coords[t, 2] = (dx, 0.10 + dy, 0.0)
        coords[t, 3] = (0.10 + dx, 0.10 + dy, 0.0)
        for base, origin in ((LEFT_BASE, -0.50), (RIGHT_BASE, 0.50)):
            coords[t, base] = (origin + dx, dy, 0.0)
            for joint in range(1, fh.HAND_JOINT_COUNT):
                finger, link = (joint - 1) // 4, (joint - 1) % 4
                # the label moves ONE finger only: exactly what this candidate is about
                bend = label_offset if finger == 1 else 0.0
                coords[t, base + joint] = (
                    origin + dx + 0.05 * (finger + 1) + 0.002 * t + bend,
                    dy + 0.04 * (link + 1),
                    0.0)
    mask = np.ones((frames, fh.SOURCE_POINT_COUNT), dtype=bool)
    return coords, mask


def sequence_for(label_offset: float, signer_offset: float):
    return fh.build_feature_sequence(*capture(label_offset, signer_offset))


LABEL_OFFSET = {"apple": 0.00, "banana": 0.09, "cherry": 0.18}


def make_dataset(root: Path, spec, split: str = "train", write_arrays: bool = True,
                 point_schema=None, split_overrides=None) -> Path:
    """spec: [(key, label, signer)] -> a TRAIN index JSON plus its npz/ directory."""
    npz_dir = root / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (key, label, signer) in enumerate(spec):
        coords, mask = capture(LABEL_OFFSET[label], 0.01 * index)
        name = key + ".npz"
        path = npz_dir / name
        if write_arrays:
            np.savez(str(path), schema=np.array(fh.F1_SCHEMA_ID),
                     coordinates=coords, mask=mask)
        records.append({
            "key": key, "label": label, "signer": signer,
            "group_signer": "group_" + signer,
            "split": (split_overrides or {}).get(key, split),
            "length": int(coords.shape[0]), "output_file": name,
            "output_sha256": cmp.sha256_path(path) if write_arrays else "0" * 64,
            "coordinates_shape": list(coords.shape), "mask_shape": list(mask.shape)})
    document = {"point_schema": point_schema or fh.F1_SCHEMA_ID, "records": records}
    index_path = root / "train_index.json"
    index_path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return index_path


THREE_SIGNERS = [("s1_apple", "apple", "s1"), ("s1_banana", "banana", "s1"),
                 ("s2_apple", "apple", "s2"), ("s2_banana", "banana", "s2"),
                 ("s3_apple", "apple", "s3"), ("s3_banana", "banana", "s3")]


class FitSpy:
    """Records every fit request and returns FIXED, NON-FITTED parameters.

    Nothing is learned from synthetic arrays, so no test output is candidate performance."""

    def __init__(self):
        self.calls = []

    def __call__(self, rows, labels, names):
        values = np.asarray(rows, dtype=np.float64)
        width = int(values.shape[1])
        self.calls.append({"width": width, "rows": int(values.shape[0]),
                           "names": list(names), "labels": list(labels)})
        return spy_parameters(width, list(names))


def spy_parameters(width: int, names=None):
    coefficients = [0.0] * width
    coefficients[0] = -1.0
    return {"feature_names": list(names) if names is not None else None,
            "scaler_mean": [0.0] * width, "scaler_scale": [1.0] * width,
            "feature_available": [True] * width, "coefficients": coefficients,
            "intercept": 0.0, "rows": 0, "positive_rows": 0,
            "estimator": "SYNTHETIC_TEST_SPY_NOT_A_FIT",
            "config": dict(cmp.LOGISTIC_CONFIG), "sklearn_version": "unused",
            "warnings": []}


class TempCase(unittest.TestCase):

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="f10_test_"))
        self.addCleanup(shutil.rmtree, str(self.root), True)

    def out(self, name="out"):
        return self.root / name

    def private(self, out):
        return json.loads((out / "private" / "per_query.json").read_text("utf-8"))


# ---- the held-out person is out of the pairs AND the references -------------

class TestHeldOutExclusion(TempCase):

    def test_no_held_out_key_reaches_a_pair_endpoint_or_a_reference(self):
        index = make_dataset(self.root, THREE_SIGNERS)
        spy = FitSpy()
        summary = f10.run(index, self.out(), fit_function=spy)
        document = self.private(self.out())
        keys_by_signer = {}
        for key, _, signer in THREE_SIGNERS:
            keys_by_signer.setdefault(signer, set()).add(key)

        self.assertEqual(len(document["fold_models"]), 3)
        for fold in document["fold_models"]:
            held = set()
            for signer in fold["held_out_signers"]:
                held |= keys_by_signer[signer]
            self.assertTrue(held)
            self.assertNotIn(fold["held_out_group"], fold["fit_groups"])
            for signer in fold["held_out_signers"]:
                self.assertNotIn(signer, fold["fit_signers"])
            for key in fold["reference_keys"]:
                self.assertNotIn(key, held, "a held-out capture stayed a reference")
            for method in ("ml_only", "hybrid"):
                for row in fold["training_pair_ledger"][method]:
                    for endpoint in row["pair"]:
                        self.assertNotIn(endpoint, held,
                                         "a held-out capture reached a pair endpoint")
            self.assertEqual(sorted(fold["query_keys"]), sorted(held))

        scored = [row["key"] for row in document["queries"]]
        self.assertEqual(sorted(scored), sorted(key for key, _, _ in THREE_SIGNERS))
        self.assertEqual(len(scored), len(set(scored)), "a capture was scored twice")
        self.assertEqual(summary["counts"]["queries"], len(THREE_SIGNERS))

    def test_every_reference_key_of_a_query_is_the_fold_reference_set(self):
        index = make_dataset(self.root, THREE_SIGNERS)
        f10.run(index, self.out(), fit_function=FitSpy())
        document = self.private(self.out())
        folds = {fold["held_out_group"]: fold for fold in document["fold_models"]}
        for row in document["queries"]:
            fold = folds[row["fold_group"]]
            self.assertEqual(row["allowed_reference_keys"], fold["reference_keys"])
            self.assertNotIn(row["key"], row["allowed_reference_keys"])

    def test_fit_count_and_widths(self):
        index = make_dataset(self.root, THREE_SIGNERS)
        spy = FitSpy()
        summary = f10.run(index, self.out(), fit_function=spy)
        # 3 groups here; the real 11-person dataset gives 11 * 2 + 2 = 24
        self.assertEqual(len(spy.calls), 3 * 2 + 2)
        self.assertEqual(summary["counts"]["fits_actual"], len(spy.calls))
        self.assertEqual(summary["counts"]["fits_expected"], len(spy.calls))
        self.assertEqual([call["width"] for call in spy.calls], [45, 48] * 4)
        for call in spy.calls:
            expected = list(f9.FIXED_FEATURE_NAMES) if call["width"] == 45 \
                else list(f9.ALIGNED_FEATURE_NAMES)
            self.assertEqual(call["names"], expected)
        self.assertEqual(summary["feature_schema"]["pair_schema"], f9.SCHEMA_ID)
        self.assertNotEqual(f9.SCHEMA_ID, fh.SCHEMA_ID)


# ---- same-signer pairs never train ------------------------------------------

class TestPairConstruction(TempCase):

    def test_every_training_pair_is_cross_signer(self):
        index = make_dataset(self.root, THREE_SIGNERS)
        f10.run(index, self.out(), fit_function=FitSpy())
        document = self.private(self.out())
        signer_of = {key: signer for key, _, signer in THREE_SIGNERS}
        ledgers = [fold["training_pair_ledger"] for fold in document["fold_models"]]
        ledgers.append(document["final_models"]["training_pair_ledger"])
        seen = 0
        for ledger in ledgers:
            for method in ("ml_only", "hybrid"):
                for row in ledger[method]:
                    first, second = row["pair"]
                    self.assertNotEqual(signer_of[first], signer_of[second],
                                        "a same-signer pair reached training")
                    self.assertEqual(row["pair"], sorted(row["pair"]))
                    seen += 1
        self.assertGreater(seen, 0)

    def test_the_cache_keys_are_ordered_and_dtw_symmetry_is_not_assumed(self):
        a, b = sequence_for(0.0, 0.0), sequence_for(0.09, 0.02)
        cache = f10.FingerPairCache({"a": a, "b": b})
        cache.aligned_pair("a", "b")
        cache.aligned_pair("b", "a")
        cache.aligned_pair("a", "b")
        self.assertEqual(cache.calls["aligned"], 2)
        self.assertIn(("a", "b"), cache.aligned)
        self.assertIn(("b", "a"), cache.aligned)
        self.assertEqual(cache.fixed_pair("a", "b").features.size, 45)
        self.assertEqual(cache.aligned_pair("a", "b").features.size, 48)


# ---- a class with no usable reference keeps its query -----------------------

class TestClassWithoutReference(TempCase):

    def test_absent_class_reference_is_a_retained_technical_failure(self):
        spec = [("s1_apple", "apple", "s1"), ("s1_cherry", "cherry", "s1"),
                ("s2_apple", "apple", "s2"), ("s3_apple", "apple", "s3")]
        index = make_dataset(self.root, spec)
        summary = f10.run(index, self.out(), fit_function=FitSpy())
        document = self.private(self.out())
        rows = {row["key"]: row for row in document["queries"]}
        self.assertEqual(sorted(rows), sorted(key for key, _, _ in spec),
                         "a query was dropped instead of failing")
        for key in ("s1_apple", "s1_cherry"):
            for method in f10.METHODS:
                result = rows[key]["methods"][method]
                self.assertEqual(result["status"], "TECHNICAL_FAILURE")
                self.assertEqual(result["reason"], "NO_REFERENCE_FOR_CLASS")
                self.assertIn("cherry", result["classes_without_evidence"])
                self.assertIsNone(result["correct"])
        for method in f10.METHODS:
            counted = summary["per_method"][method]
            self.assertEqual(counted["top1_correct"] + counted["top1_incorrect"]
                             + counted["technical_failures"], len(spec))
            self.assertGreaterEqual(counted["technical_failures"], 2)
        self.assertEqual(summary["counts"]["classes"], 2)
        self.assertIn("cherry", summary["counts"]["class_query_counts"])


# ---- a wrong index split is refused before any array is touched -------------

class TestIndexRefusals(TempCase):

    def test_non_train_split_is_refused_before_array_access(self):
        # the arrays are deliberately NOT written: if the refusal still names the split,
        # no npz was opened
        index = make_dataset(self.root, THREE_SIGNERS, split="val", write_arrays=False)
        self.assertFalse(any((self.root / "npz").iterdir()))
        spy = FitSpy()
        with self.assertRaises(cmp.ComparisonError) as caught:
            f10.run(index, self.out(), fit_function=spy)
        self.assertIn("non-train", str(caught.exception))
        self.assertEqual(spy.calls, [])
        self.assertFalse((self.out() / "private").exists())

    def test_one_non_train_record_refuses_the_whole_index(self):
        index = make_dataset(self.root, THREE_SIGNERS,
                             split_overrides={"s2_banana": "test"})
        spy = FitSpy()
        with self.assertRaises(cmp.ComparisonError):
            f10.run(index, self.out(), fit_function=spy)
        self.assertEqual(spy.calls, [])

    def test_a_foreign_point_schema_is_refused(self):
        index = make_dataset(self.root, THREE_SIGNERS, point_schema="something_else_v9")
        spy = FitSpy()
        with self.assertRaises(cmp.ComparisonError):
            f10.run(index, self.out(), fit_function=spy)
        self.assertEqual(spy.calls, [])

    def test_a_single_outer_group_is_refused(self):
        spec = [("s1_apple", "apple", "s1"), ("s1_banana", "banana", "s1")]
        index = make_dataset(self.root, spec)
        spy = FitSpy()
        with self.assertRaises(cmp.ComparisonError):
            f10.run(index, self.out(), fit_function=spy)
        self.assertEqual(spy.calls, [], "nothing may be fitted before the fold check")


# ---- an existing output is refused before anything is fitted ----------------

class TestOutputProtection(TempCase):

    def test_existing_output_directory_refuses_the_cli_before_fitting(self):
        index = make_dataset(self.root, THREE_SIGNERS)
        out = self.out()
        out.mkdir()
        sentinel = out / "earlier_result.json"
        sentinel.write_text("{\"kept\": true}", encoding="utf-8")
        spy = FitSpy()
        code = f10.main(["--train-index", str(index), "--out", str(out)],
                        fit_function=spy)
        self.assertEqual(code, 1)
        self.assertEqual(spy.calls, [], "a fit started against an existing output")
        self.assertEqual(sentinel.read_text("utf-8"), "{\"kept\": true}")
        self.assertFalse((out / "run.json").exists())
        self.assertFalse((out / "private").exists())

    def test_run_refuses_an_output_that_already_holds_a_result(self):
        index = make_dataset(self.root, THREE_SIGNERS)
        out = self.out()
        (out / "private").mkdir(parents=True)
        spy = FitSpy()
        with self.assertRaises(cmp.ComparisonError):
            f10.run(index, out, fit_function=spy)
        self.assertEqual(spy.calls, [])

    def test_the_first_failure_is_kept_and_never_replaced(self):
        spec = [("s1_apple", "apple", "s1"), ("s1_banana", "banana", "s1")]
        index = make_dataset(self.root, spec)
        out = self.out()
        code = f10.main(["--train-index", str(index), "--out", str(out)],
                        fit_function=FitSpy())
        self.assertEqual(code, 1)
        failure = out / "first_failure.json"
        self.assertTrue(failure.exists())
        first = failure.read_text("utf-8")
        self.assertIn("traceback", first)
        self.assertEqual(json.loads((out / "run.json").read_text("utf-8"))["status"],
                         "FAILED")
        second = f10.main(["--train-index", str(index), "--out", str(out)],
                          fit_function=FitSpy())
        self.assertEqual(second, 1)
        self.assertEqual(failure.read_text("utf-8"), first,
                         "the first failure was replaced")


# ---- the completed run writes what it claims --------------------------------

class TestRunArtifacts(TempCase):

    def test_cli_completes_and_saves_hashes_models_and_references(self):
        index = make_dataset(self.root, THREE_SIGNERS)
        out = self.out()
        code = f10.main(["--train-index", str(index), "--out", str(out)],
                        fit_function=FitSpy())
        self.assertEqual(code, 0)
        run_document = json.loads((out / "run.json").read_text("utf-8"))
        self.assertEqual(run_document["status"], "COMPLETED")
        self.assertEqual(run_document["split"], cmp.TRAIN_SPLIT)
        self.assertEqual(run_document["pair_schema"], f9.SCHEMA_ID)
        self.assertEqual(run_document["config"], dict(cmp.LOGISTIC_CONFIG))
        self.assertEqual(set(run_document["code_sha256"]),
                         {"full_hand_pair_features_v1", "full_hand_comparison_v1",
                          "finger_pair_features_v2", "finger_candidate_experiment_v2"})
        self.assertEqual(run_document["train_index_sha256"], cmp.sha256_path(index))
        self.assertIn("started_utc", run_document)
        self.assertIn("finished_utc", run_document)
        self.assertFalse((out / "first_failure.json").exists())

        summary = json.loads((out / "safe_summary.json").read_text("utf-8"))
        self.assertEqual(summary["schema"], f10.OUTPUT_SCHEMA)
        self.assertEqual(summary["conclusion"], f10.CONCLUSION)
        self.assertIn("NOT INDEPENDENT FINAL PERFORMANCE", summary["material"])
        self.assertFalse(summary["threshold_policy"]["applied_in_this_run"])
        self.assertFalse(summary["final_models"]["used_for_outer_predictions"])
        self.assertTrue(summary["limits"])
        blob = json.dumps(summary, ensure_ascii=False)
        for key, _, signer in THREE_SIGNERS:            # no key or signer may leak out
            self.assertNotIn(key, blob)
            self.assertNotIn("group_" + signer, blob)

        with np.load(str(out / "private" / "final_references.npz"),
                     allow_pickle=False) as saved:      # arrays only, nothing pickled
            stored = list(saved.keys())
            self.assertEqual(str(saved["schema"]), f9.SCHEMA_ID)
            for key, _, _ in THREE_SIGNERS:
                self.assertIn("points_" + key, stored)
                self.assertEqual(saved["points_" + key].shape,
                                 (fh.RESAMPLE_T, fh.FEATURE_POINT_COUNT, 3))
                self.assertEqual(saved["mask_" + key].dtype, np.bool_)

        models = json.loads((out / "private" / "final_models.json").read_text("utf-8"))
        self.assertEqual(models["feature_schema"]["fixed_features"],
                         list(f9.FIXED_FEATURE_NAMES))
        self.assertEqual(models["feature_schema"]["aligned_features"],
                         list(f9.ALIGNED_FEATURE_NAMES))
        self.assertTrue(models["not_bundled_into_unity"])
        self.assertFalse(models["models"]["used_for_outer_predictions"])


# ---- score_sequence: no target, no mutation ---------------------------------

class TestScoreSequence(TempCase):

    def setUp(self):
        super().setUp()
        self.query = sequence_for(LABEL_OFFSET["apple"], 0.00)
        self.references = {
            "r_apple": {"label": "apple", "seq": sequence_for(LABEL_OFFSET["apple"], 0.03)},
            "r_banana": {"label": "banana",
                         "seq": sequence_for(LABEL_OFFSET["banana"], 0.05)}}
        self.models = {"ml_only": spy_parameters(45, f9.FIXED_FEATURE_NAMES),
                       "hybrid": spy_parameters(48, f9.ALIGNED_FEATURE_NAMES)}

    def test_the_signature_has_no_target_or_user_argument(self):
        names = list(inspect.signature(f10.score_sequence).parameters)
        self.assertEqual(names, ["seq", "references", "models"])
        for function in (f9.fixed_time_features, f9.aligned_pair_features):
            self.assertEqual(list(inspect.signature(function).parameters), ["a", "b"])

    def test_all_three_methods_report_raw_scores_top1_and_margin(self):
        result = f10.score_sequence(self.query, self.references, self.models)
        self.assertEqual(sorted(result["methods"]), sorted(f10.METHODS))
        self.assertFalse(result["is_accept_decision"])
        self.assertEqual(result["classes"], ["apple", "banana"])
        for method in f10.METHODS:
            outcome = result["methods"][method]
            self.assertEqual(outcome["status"], "RANKED")
            self.assertEqual(sorted(outcome["raw"]), ["apple", "banana"])
            self.assertIn(outcome["top1"], ("apple", "banana"))
            self.assertTrue(np.isfinite(outcome["margin"]))
            self.assertGreaterEqual(outcome["margin"], 0.0)

    def test_inputs_are_not_mutated(self):
        before_points = {key: entry["seq"].points.copy()
                         for key, entry in self.references.items()}
        before_mask = {key: entry["seq"].mask.copy()
                       for key, entry in self.references.items()}
        before_query = self.query.points.copy()
        before_models = copy.deepcopy(self.models)
        before_keys = sorted(self.references)
        f10.score_sequence(self.query, self.references, self.models)
        self.assertEqual(sorted(self.references), before_keys)
        self.assertEqual(self.models, before_models)
        self.assertTrue(np.array_equal(self.query.points, before_query, equal_nan=True))
        for key, entry in self.references.items():
            self.assertTrue(np.array_equal(entry["seq"].points, before_points[key],
                                           equal_nan=True))
            self.assertTrue(np.array_equal(entry["seq"].mask, before_mask[key]))

    def test_an_unrelated_extra_field_cannot_act_as_a_target(self):
        plain = f10.score_sequence(self.query, self.references, self.models)
        decorated = {key: dict(entry, target="banana", user_id="s9")
                     for key, entry in self.references.items()}
        hinted = f10.score_sequence(self.query, decorated, self.models)
        for method in f10.METHODS:
            self.assertEqual(plain["methods"][method]["raw"],
                             hinted["methods"][method]["raw"])
            self.assertEqual(plain["methods"][method]["top1"],
                             hinted["methods"][method]["top1"])

    def test_repeated_scoring_is_deterministic(self):
        one = f10.score_sequence(self.query, self.references, self.models)
        two = f10.score_sequence(self.query, self.references, self.models)
        self.assertEqual(one["methods"], two["methods"])

    def test_a_class_whose_pairs_all_fail_stays_a_technical_failure(self):
        blank_coords, blank_mask = capture(0.0, 0.0)
        blank_mask[:, :] = False
        blank = fh.build_feature_sequence(blank_coords, blank_mask)
        references = dict(self.references)
        references["r_cherry"] = {"label": "cherry", "seq": blank}
        result = f10.score_sequence(self.query, references, self.models)
        self.assertIn("cherry", result["classes"])
        for method in f10.METHODS:
            outcome = result["methods"][method]
            self.assertEqual(outcome["status"], "TECHNICAL_FAILURE")
            self.assertEqual(outcome["reason"], "NO_USABLE_REFERENCE_PAIR")
            self.assertIn("cherry", outcome["classes_without_evidence"])

    def test_baseline_width_models_are_refused(self):
        stale = {"ml_only": spy_parameters(15), "hybrid": spy_parameters(18)}
        with self.assertRaises(cmp.ComparisonError):
            f10.score_sequence(self.query, self.references, stale)

    def test_wrong_feature_names_are_refused(self):
        wrong = {"ml_only": spy_parameters(45, ["x"] * 45),
                 "hybrid": spy_parameters(48, f9.ALIGNED_FEATURE_NAMES)}
        with self.assertRaises(cmp.ComparisonError):
            f10.score_sequence(self.query, self.references, wrong)

    def test_empty_references_and_the_reserved_key_are_refused(self):
        with self.assertRaises(cmp.ComparisonError):
            f10.score_sequence(self.query, {}, self.models)
        with self.assertRaises(cmp.ComparisonError):
            f10.score_sequence(self.query,
                               {f10.QUERY_KEY: {"label": "apple", "seq": self.query}},
                               self.models)

    def test_a_malformed_reference_entry_is_refused(self):
        broken = dict(self.references)
        broken["r_broken"] = {"label": "apple"}
        with self.assertRaises(cmp.ComparisonError):
            f10.score_sequence(self.query, broken, self.models)


if __name__ == "__main__":
    unittest.main(verbosity=2)
