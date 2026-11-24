"""Focused tests for the R28 projected-arm candidate.

Scientific controls first (invariance, bone length, forearm rotation, the reflection
counterexample), then robustness: missingness, nan/inf, degenerate bones, invalid
bounds, equal-reference weighting, no mutation, NaN-free JSON. Synthetic geometry only;
no participant data is imported here.
"""
from __future__ import annotations

import copy
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import run_geometry_challenge as runner
from projected_arm_review import (
    MEASURES, SIDES, angle_deg, base_scene, build_report, normalize_rows, rebuild_arm,
    reflect_arms_across_shoulder_line, review_projected_arm, synthetic_controls,
    transform_scene,
)

TOL = 1e-7


def one_step_report(query_scene, reference_scene=None, drop=None):
    return build_report([query_scene], [reference_scene or base_scene()], [(0, 0)], drop=drop)


def summary(report, side="L", measure="shoulder_angle", bounds=(0, 0)):
    review = review_projected_arm(report, list(bounds))
    assert review["status"] == "AVAILABLE", review.get("reason")
    return review["sides"][side][measure]


class AngleMathTests(unittest.TestCase):
    def test_range_and_stability(self):
        self.assertAlmostEqual(angle_deg((1.0, 0.0), (0.0, 1.0)), 90.0, delta=TOL)
        self.assertAlmostEqual(angle_deg((1.0, 0.0), (0.0, -1.0)), 90.0, delta=TOL,
                               msg="the angle must be unsigned")
        self.assertAlmostEqual(angle_deg((1.0, 0.0), (-1.0, 1e-18)), 180.0, delta=1e-6,
                               msg="atan2 must stay stable near the 180 deg fold")
        self.assertAlmostEqual(angle_deg((3.0, 0.0), (1.0, 0.0)), 0.0, delta=TOL)

    def test_zero_length_and_nonfinite_are_unavailable(self):
        self.assertIsNone(angle_deg((0.0, 0.0), (1.0, 0.0)))
        self.assertIsNone(angle_deg((1.0, 0.0), (0.0, 0.0)))
        self.assertIsNone(angle_deg((float("nan"), 0.0), (1.0, 0.0)))
        self.assertIsNone(angle_deg((float("inf"), 0.0), (1.0, 0.0)))


class Control1SimilarityInvarianceTests(unittest.TestCase):
    """Translation / uniform scale / whole-scene rotation / reflection leave angles fixed."""

    def test_angles_invariant_and_baseline_renormalised_too(self):
        controls = synthetic_controls()
        experiment = controls["experiments"][0]
        self.assertEqual(experiment["name"], "global_similarity_invariance")
        for entry in experiment["per_transform"]:
            self.assertLess(entry["max_abs_angle_change_deg"], 1e-6, entry["transform"])
            # The existing shoulder normalisation already absorbs these transforms, so the
            # renormalised baseline location is invariant as well: no fake advantage.
            self.assertLess(entry["baseline_location_delta_change"], 1e-9, entry["transform"])
            self.assertGreater(entry["baseline_location_delta_norm"], 0.0)
        self.assertIn("not evidence", experiment["caveat"])

    def test_review_level_invariance_under_combined_transform(self):
        query = rebuild_arm(base_scene(), "L", forearm_rotation_deg=-30.0)
        plain = summary(one_step_report(query), measure="elbow_angle")
        kwargs = dict(scale=0.4, rotation_deg=-115.0, reflect=True, translate=(9.0, 3.0))
        moved = summary(one_step_report(transform_scene(query, **kwargs),
                                        transform_scene(base_scene(), **kwargs)),
                        measure="elbow_angle")
        for key in ("query_angle_deg_mean", "reference_angle_deg_mean",
                    "signed_difference_deg_mean"):
            self.assertAlmostEqual(plain[key], moved[key], delta=TOL, msg=key)

    def test_query_only_rotation_is_invisible_to_angles_but_not_to_location(self):
        experiment = synthetic_controls()["experiments"][0]["query_only_rotation_25deg"]
        self.assertLess(experiment["max_abs_angle_change_deg"], 1e-6)
        self.assertGreater(experiment["baseline_location_delta_change"], 1e-3)
        self.assertIn("ambiguous", experiment["reading"])


class Control2BoneLengthTests(unittest.TestCase):
    def test_changing_projected_bone_lengths_keeps_angles(self):
        experiment = synthetic_controls()["experiments"][1]
        self.assertEqual(experiment["name"], "projected_bone_length_sensitivity")
        self.assertLess(experiment["max_abs_angle_change_deg"], 1e-6)
        self.assertGreater(experiment["baseline_location_delta_change"], 1e-3,
                           "the hand must actually move, otherwise the control is vacuous")
        self.assertIn("NOT a certified acceptable", experiment["label"])

    def test_endpoints_recomputed_coherently(self):
        scene = rebuild_arm(base_scene(), "L", upper_len=96.0, fore_len=30.0)
        upper = math.dist(scene["shoulder_L"], scene["elbow_L"])
        fore = math.dist(scene["elbow_L"], scene["wrist_L"])
        self.assertAlmostEqual(upper, 96.0, delta=TOL)
        self.assertAlmostEqual(fore, 30.0, delta=TOL,
                               msg="the wrist must follow the elbow, not stay behind")
        review = summary(one_step_report(scene), measure="elbow_angle")
        base = summary(one_step_report(base_scene()), measure="elbow_angle")
        self.assertAlmostEqual(review["query_angle_deg_mean"], base["query_angle_deg_mean"],
                               delta=TOL)


class Control3ForearmRotationTests(unittest.TestCase):
    def test_elbow_angle_moves_by_the_applied_rotation_shoulder_does_not(self):
        experiment = synthetic_controls()["experiments"][2]
        self.assertEqual(experiment["name"], "forearm_rotation_about_elbow")
        self.assertAlmostEqual(abs(experiment["elbow_angle_change_deg"]), 25.0, delta=1e-6)
        self.assertAlmostEqual(experiment["shoulder_angle_change_deg"], 0.0, delta=1e-6)

    def test_nondegenerate_fixture_no_universal_monotonic_claim(self):
        base = one_step_report(base_scene())
        elbow0 = summary(base, measure="elbow_angle")["query_angle_deg_mean"]
        self.assertGreater(elbow0, 5.0)
        self.assertLess(elbow0, 175.0, "fixture must stay away from the 0/180 deg folds")
        turned = rebuild_arm(base_scene(), "L", forearm_rotation_deg=-25.0)
        elbow1 = summary(one_step_report(turned), measure="elbow_angle")["query_angle_deg_mean"]
        self.assertAlmostEqual(abs(elbow1 - elbow0), 25.0, delta=1e-6)
        shoulder = summary(one_step_report(turned))["signed_difference_deg_mean"]
        self.assertAlmostEqual(shoulder, 0.0, delta=1e-6)
        # Folding past 0 deg gives the same unsigned angle back: no monotonic relation.
        folded = rebuild_arm(base_scene(), "L", forearm_rotation_deg=-155.0)
        elbow2 = summary(one_step_report(folded), measure="elbow_angle")["query_angle_deg_mean"]
        self.assertAlmostEqual(elbow2, 65.0, delta=1e-6)


class Control4ReflectionCounterexampleTests(unittest.TestCase):
    """Equal unsigned angles, different hand location: angles cannot replace location."""

    def test_mirrored_arm_has_identical_angles_but_a_moved_hand(self):
        mirrored = reflect_arms_across_shoulder_line(base_scene())
        report = one_step_report(mirrored)
        for measure in MEASURES:
            stats = summary(report, measure=measure)
            self.assertAlmostEqual(stats["signed_difference_deg_mean"], 0.0, delta=1e-6, msg=measure)
            self.assertAlmostEqual(stats["absolute_difference_deg_mean"], 0.0, delta=1e-6, msg=measure)
            self.assertIsNotNone(stats["query_angle_deg_mean"])
        delta = report["steps"][0]["hands"]["L"]["shoulder_position"]["delta"]
        self.assertGreater(math.hypot(*delta), 0.5,
                           "the independent location difference must survive untouched")

    def test_controls_payload_states_the_rejection(self):
        controls = synthetic_controls()
        experiment = controls["experiments"][3]
        self.assertEqual(experiment["name"], "reflection_counterexample")
        self.assertAlmostEqual(experiment["shoulder_angle_difference_deg"], 0.0, delta=1e-6)
        self.assertAlmostEqual(experiment["elbow_angle_difference_deg"], 0.0, delta=1e-6)
        self.assertGreater(experiment["baseline_location_delta_norm"], 0.5)
        self.assertIn("rejected as a sole location rule", controls["conclusion"])
        self.assertIn("NOT recognition accuracy", controls["value_type"])


class SeparatePointRequirementTests(unittest.TestCase):
    def test_shoulder_angle_survives_a_missing_hand_wrist(self):
        report = one_step_report(base_scene(), drop={("query", 0, "wrist_L"): None})
        shoulder = summary(report, measure="shoulder_angle")
        elbow = summary(report, measure="elbow_angle")
        self.assertIsNotNone(shoulder["query_angle_deg_mean"])
        self.assertEqual(shoulder["counts"]["paired_observed_steps"], 1)
        self.assertIsNone(elbow["query_angle_deg_mean"])
        self.assertIsNone(elbow["signed_difference_deg_mean"])
        self.assertEqual(elbow["unavailable_reasons"]["query"], {"missing_hand_wrist": 1})
        self.assertIsNotNone(elbow["reference_angle_deg_mean"],
                             "the reference side keeps its own observed support")

    def test_elbow_angle_survives_a_missing_opposite_shoulder(self):
        report = one_step_report(base_scene(), drop={("query", 0, "shoulder_R"): None})
        shoulder = summary(report, measure="shoulder_angle")
        elbow = summary(report, measure="elbow_angle")
        self.assertIsNone(shoulder["query_angle_deg_mean"])
        self.assertEqual(shoulder["unavailable_reasons"]["query"],
                         {"missing_opposite_shoulder": 1})
        self.assertIsNotNone(elbow["query_angle_deg_mean"])
        self.assertEqual(elbow["counts"]["paired_observed_steps"], 1)

    def test_total_missingness_is_unavailable_not_zero(self):
        drop = {("query", 0, name): None for name in
                ("shoulder_L", "shoulder_R", "elbow_L", "elbow_R")}
        drop.update({("reference", 0, name): None for name in
                     ("shoulder_L", "shoulder_R", "elbow_L", "elbow_R")})
        drop.update({("query", 0, "wrist_L"): None, ("reference", 0, "wrist_L"): None,
                     ("query", 0, "wrist_R"): None, ("reference", 0, "wrist_R"): None})
        review = review_projected_arm(one_step_report(base_scene(), drop=drop), [0, 0])
        self.assertEqual(review["status"], "AVAILABLE")
        self.assertFalse(review["any_angle_observed"])
        for side in SIDES:
            for measure in MEASURES:
                stats = review["sides"][side][measure]
                for key in ("query_angle_deg_mean", "reference_angle_deg_mean",
                            "signed_difference_deg_mean", "absolute_difference_deg_mean"):
                    self.assertIsNone(stats[key], "%s/%s/%s" % (side, measure, key))
                self.assertEqual(stats["counts"]["paired_observed_steps"], 0)

    def test_nonfinite_points_are_reported_as_nonfinite(self):
        report = one_step_report(base_scene(), drop={
            ("query", 0, "elbow_L"): [float("nan"), 0.0],
            ("query", 0, "wrist_R"): [0.0, float("inf")]})
        shoulder_l = summary(report, side="L", measure="shoulder_angle")
        elbow_r = summary(report, side="R", measure="elbow_angle")
        self.assertEqual(shoulder_l["unavailable_reasons"]["query"], {"nonfinite_elbow": 1})
        self.assertEqual(elbow_r["unavailable_reasons"]["query"], {"nonfinite_hand_wrist": 1})
        self.assertIsNone(shoulder_l["query_angle_deg_mean"])

    def test_degenerate_bone_is_unavailable_not_zero_degrees(self):
        collapsed_forearm = dict(base_scene())
        collapsed_forearm["wrist_L"] = collapsed_forearm["elbow_L"]
        stats = summary(one_step_report(collapsed_forearm), measure="elbow_angle")
        self.assertIsNone(stats["query_angle_deg_mean"])
        self.assertEqual(stats["unavailable_reasons"]["query"], {"degenerate_vector": 1})
        self.assertIsNotNone(summary(one_step_report(collapsed_forearm))["query_angle_deg_mean"],
                             "a collapsed forearm must not disable the shoulder angle")

        collapsed_upper = dict(base_scene())
        collapsed_upper["elbow_L"] = collapsed_upper["shoulder_L"]
        for measure in MEASURES:
            stats = summary(one_step_report(collapsed_upper), measure=measure)
            self.assertIsNone(stats["query_angle_deg_mean"], measure)
            self.assertEqual(stats["unavailable_reasons"]["query"], {"degenerate_vector": 1})


class RangeAndPathTests(unittest.TestCase):
    def setUp(self):
        self.report = build_report([base_scene(), base_scene()],
                                   [base_scene(), base_scene()], [(0, 0), (1, 1)])

    def test_invalid_bounds(self):
        for bounds in ([1, 0], [-1, 0], [0], "0-1", None, [0.0, 1.0], [True, True], [0, 1, 2]):
            review = review_projected_arm(self.report, bounds)
            self.assertEqual(review["status"], "UNAVAILABLE", bounds)
            self.assertEqual(review["reason"], "invalid_reference_range", bounds)

    def test_range_outside_reference_rows(self):
        review = review_projected_arm(self.report, [0, 9])
        self.assertEqual(review["reason"], "range_outside_reference")

    def test_range_with_no_selected_steps(self):
        report = build_report([base_scene()], [base_scene(), base_scene()], [(0, 0)])
        review = review_projected_arm(report, [1, 1])
        self.assertEqual(review["reason"], "no_selected_steps_in_range")

    def test_rejects_foreign_or_unavailable_reports(self):
        self.assertEqual(review_projected_arm(None, [0, 0])["reason"], "invalid_report")
        other = dict(self.report, schema="something_else_v1")
        self.assertEqual(review_projected_arm(other, [0, 0])["reason"], "unsupported_schema")
        blocked = dict(self.report, status="UNAVAILABLE")
        self.assertEqual(review_projected_arm(blocked, [0, 0])["reason"], "report_not_available")
        broken = dict(self.report, steps={})
        self.assertEqual(review_projected_arm(broken, [0, 0])["reason"], "invalid_steps")
        headless = dict(self.report)
        headless.pop("inputs")
        self.assertEqual(review_projected_arm(headless, [0, 0])["reason"], "missing_inputs")

    def test_only_selected_path_pairs_are_used(self):
        report = build_report([base_scene(), rebuild_arm(base_scene(), "L",
                                                         forearm_rotation_deg=-30.0)],
                              [base_scene(), base_scene()], [(0, 0), (1, 1)])
        inside = summary(report, measure="elbow_angle", bounds=(1, 1))
        self.assertEqual(inside["counts"]["steps_in_range"], 1)
        self.assertEqual(inside["counts"]["reference_samples_in_range"], 1)
        self.assertAlmostEqual(abs(inside["signed_difference_deg_mean"]), 30.0, delta=1e-6)

    def test_malformed_steps_are_skipped_not_guessed(self):
        report = build_report([base_scene()], [base_scene()], [(0, 0)])
        report["steps"].append({"i": "x", "j": 0})
        report["steps"].append({"j": 0})
        review = review_projected_arm(report, [0, 0])
        self.assertEqual(review["malformed_steps_skipped"], 2)
        self.assertEqual(review["sides"]["L"]["shoulder_angle"]["counts"]["steps_in_range"], 1)


class EqualReferenceWeightingTests(unittest.TestCase):
    """Repeated DTW matches must not outvote a reference sample matched only once."""

    def setUp(self):
        self.query = [base_scene(), rebuild_arm(base_scene(), "L", forearm_rotation_deg=-30.0)]
        self.reference = [base_scene(), base_scene()]
        self.path = [(0, 0), (0, 0), (0, 0), (1, 1)]
        self.report = build_report(self.query, self.reference, self.path)

    def test_mean_within_j_then_equally_across_j(self):
        stats = summary(self.report, measure="elbow_angle", bounds=(0, 1))
        self.assertAlmostEqual(stats["query_angle_deg_mean"], 75.0, delta=1e-6,
                               msg="step-weighted would give 82.5 deg")
        self.assertAlmostEqual(abs(stats["signed_difference_deg_mean"]), 15.0, delta=1e-6)
        self.assertAlmostEqual(stats["absolute_difference_deg_mean"], 15.0, delta=1e-6)
        self.assertEqual(stats["counts"]["steps_in_range"], 4)
        self.assertEqual(stats["counts"]["reference_samples_in_range"], 2)
        self.assertEqual(stats["counts"]["paired_observed_steps"], 4)
        self.assertEqual(stats["counts"]["paired_observed_samples"], 2)

    def test_signed_and_absolute_are_reported_separately(self):
        query = [rebuild_arm(base_scene(), "L", forearm_rotation_deg=-30.0),
                 rebuild_arm(base_scene(), "L", forearm_rotation_deg=30.0)]
        report = build_report(query, self.reference, [(0, 0), (1, 1)])
        stats = summary(report, measure="elbow_angle", bounds=(0, 1))
        self.assertAlmostEqual(stats["signed_difference_deg_mean"], 0.0, delta=1e-6,
                               msg="opposite signed differences cancel")
        self.assertAlmostEqual(stats["absolute_difference_deg_mean"], 30.0, delta=1e-6,
                               msg="the absolute difference must not cancel")

    def test_per_reference_sample_rows_are_sorted_and_counted(self):
        stats = summary(self.report, measure="elbow_angle", bounds=(0, 1))
        rows = stats["per_reference_sample"]
        self.assertEqual([row["j"] for row in rows], [0, 1])
        self.assertEqual([row["query_steps"] for row in rows], [3, 1])
        self.assertAlmostEqual(rows[0]["query_angle_deg"], 90.0, delta=1e-6)
        self.assertAlmostEqual(rows[1]["query_angle_deg"], 60.0, delta=1e-6)


class ContractTests(unittest.TestCase):
    def test_report_is_not_mutated(self):
        report = build_report([base_scene()], [base_scene()], [(0, 0)])
        before = copy.deepcopy(report)
        review_projected_arm(report, [0, 0])
        self.assertEqual(report, before)

    def test_no_verdict_fields(self):
        review = review_projected_arm(one_step_report(base_scene()), [0, 0])
        self.assertIsNone(review["accepted"])
        self.assertIsNone(review["correction"])
        self.assertFalse(review["is_a_verdict"])
        self.assertIn("not anatomical 3D flexion", review["angle_unit"])
        self.assertIn("hand-detector wrist anchor", review["wrist_source"])
        unavailable = review_projected_arm(one_step_report(base_scene()), [5, 5])
        self.assertIsNone(unavailable["accepted"])
        self.assertIsNone(unavailable["correction"])

    def test_json_output_has_no_nan(self):
        report = one_step_report(base_scene(), drop={("query", 0, "elbow_L"): [float("nan"), 1.0]})
        text = json.dumps(review_projected_arm(report, [0, 0]), allow_nan=False)
        self.assertNotIn("NaN", text)
        self.assertNotIn("Infinity", text)
        json.dumps(synthetic_controls(), allow_nan=False)

    def test_angles_stay_within_zero_to_180(self):
        for rotation in range(0, 360, 15):
            scene = rebuild_arm(base_scene(), "L", forearm_rotation_deg=float(rotation))
            stats = summary(one_step_report(scene), measure="elbow_angle")
            self.assertGreaterEqual(stats["query_angle_deg_mean"], 0.0)
            self.assertLessEqual(stats["query_angle_deg_mean"], 180.0 + TOL)

    def test_normalisation_uses_clip_median_shoulder_width(self):
        rows, scale = normalize_rows([base_scene(), base_scene(half_span=50.0)])
        self.assertAlmostEqual(scale, 90.0, delta=TOL)
        self.assertAlmostEqual(rows[0]["shoulder_L"][0], 40.0 / 90.0, delta=TOL)


class RunnerTests(unittest.TestCase):
    def _run(self, argv):
        code = runner.main(argv)
        self.assertEqual(code, 0)

    def test_report_mode_output(self):
        report = build_report([base_scene(), base_scene()], [base_scene(), base_scene()],
                              [(0, 0), (1, 1)])
        records = [{"id": "a", "report": report, "range": [0, 1], "extra": "ignored"},
                   {"id": "b", "report": report, "range": [3, 4]}]
        with tempfile.TemporaryDirectory() as folder:
            source = os.path.join(folder, "in.json")
            target = os.path.join(folder, "out.json")
            with open(source, "w", encoding="utf-8") as handle:
                json.dump(records, handle)
            before = Path(source).read_text(encoding="utf-8")
            self._run(["--input", source, "--output", target])
            self.assertEqual(Path(source).read_text(encoding="utf-8"), before,
                             "the runner must not write back into the input report")
            text = Path(target).read_text(encoding="utf-8")
            payload = json.loads(text)

        self.assertEqual(payload["aggregate"]["records_total"], 2)
        self.assertEqual(payload["aggregate"]["records_available"], 1)
        self.assertEqual(payload["aggregate"]["records_unavailable_reasons"],
                         {"range_outside_reference": 1})
        bucket = payload["aggregate"]["by_side_measure"]["L.elbow_angle"]
        self.assertEqual(bucket["paired_observed_steps"], 2)
        self.assertEqual(bucket["paired_step_availability"], 1.0)
        self.assertNotIn("body_points", text, "no landmark arrays in the output")
        self.assertNotIn('"delta"', text, "no per-step coordinate deltas in the output")
        self.assertNotIn('"steps":', text, "no copied path payload in the output")
        self.assertNotIn("NaN", text)
        self.assertIn("hand-detector wrist anchor", text,
                      "the wrist provenance must stay documented")

    def test_synthetic_controls_mode(self):
        with tempfile.TemporaryDirectory() as folder:
            target = os.path.join(folder, "controls.json")
            self._run(["--synthetic-controls", "--output", target])
            payload = json.loads(Path(target).read_text(encoding="utf-8"))
        self.assertEqual(payload["mode"], "synthetic_controls")
        self.assertEqual(len(payload["synthetic_controls"]["experiments"]), 4)
        self.assertIn("NOT recognition accuracy", payload["synthetic_controls"]["value_type"])
        self.assertTrue(any("never accuracy" in note for note in payload["notes"]))

    def test_bad_records_are_reported_not_crashed(self):
        with tempfile.TemporaryDirectory() as folder:
            source = os.path.join(folder, "in.json")
            target = os.path.join(folder, "out.json")
            with open(source, "w", encoding="utf-8") as handle:
                json.dump(["not a record", {"id": "c"}, {"id": "d", "report": {}, "range": [0, 0]}],
                          handle)
            self._run(["--input", source, "--output", target])
            payload = json.loads(Path(target).read_text(encoding="utf-8"))
        self.assertEqual([r["status"] for r in payload["records"]], ["UNAVAILABLE"] * 3)
        self.assertEqual(payload["records"][2]["reason"], "unsupported_schema")

    def test_input_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            source = os.path.join(folder, "same.json")
            with open(source, "w", encoding="utf-8") as handle:
                json.dump([], handle)
            with self.assertRaises(SystemExit):
                runner.main(["--input", source, "--output", source])


if __name__ == "__main__":
    unittest.main(verbosity=2)
