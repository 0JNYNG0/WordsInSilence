"""Read-only independent arithmetic for four saved geometry sensitivity cells.

Does not import or rerun the consumer, detector, extractor, matcher or selftests.
The synthetic double-error cancellation is checked algebraically, not treated
as measured natural performance.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np

U = Path(__file__).resolve().parent.parent
O = U.parent
PINS, FAILURES = {}, []
CHECKS = 0


def check(ok, label):
    global CHECKS
    CHECKS += 1
    if not bool(ok):
        FAILURES.append(label)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path, expected=None):
    path = Path(path).resolve()
    h = sha(path)
    if expected is not None:
        check(h == expected, "pin " + str(path))
    PINS[str(path)] = h
    return path


def read(path):
    return json.loads(pin(path).read_text(encoding="utf-8-sig"))


def arrays(path, expected=None):
    with np.load(pin(path, expected), allow_pickle=False) as a:
        return {k: a[k].copy() for k in a.files}


def main():
    frozen = read(U / "FROZEN_COMBINED_CHECK.json")
    result = read(U / "COMBINED_CHECK.json")
    for path, h in frozen["pins"].items():
        pin(path, h)
    check(frozen["reference"] == ["P065", 48], "fixed reference")
    check(frozen["query_alternatives"] == [["P066", 74], ["P066", 98]], "fixed query hypotheses")
    check(frozen["new_head_predictions"] is False, "no new model evidence")
    case = read(O / "word100_action_feedback_casebook_20261101/INPUTS.json")
    original = {}
    for alias in ["P065", "P066"]:
        source = case["rows"][alias]
        original[alias] = arrays(source["input"], source["input_sha256"])
    recovery = read(U / "ONE_RECOVERY_RESULT.json")
    recovered = arrays(U / "P065_f48_recovered.npz", recovery["output_sha256"])
    recovery_review = read(U / "review/VERIFY_RECOVERY.json")
    check(recovery_review["status"] == "PASS", "independent69 source derivation already verified")
    check(recovery_review["maximum_independent_feature_absolute_error"] == 0, "independent69 exact numeric match")
    names = frozen["features"]
    check(names == case["rows"]["P065"]["feature_names"] == case["rows"]["P066"]["feature_names"], "same69 schema")
    check(len(names) == 69, "all69 source feature opportunities")
    baseline_path = read(O / "word100_relational_sequence_compare_20261031/pairs/C_relational_path_P066_P065.json")
    check([74, 48] in baseline_path["path"], "current endpoint is actually in saved C")
    check([98, 48] not in baseline_path["path"], "alternate not original current link")
    expected_keys = {(version, frame) for version in ["original", "reobserved_current_image"] for frame in [74, 98]}
    check({(r["input_version"], r["query_frame"]) for r in result["rows"]} == expected_keys, "four hypotheses retained")
    check(len(result["rows"]) == 4, "four and only four cells")
    brief, maximum_error, known_preserved = [], 0., 0
    by_key = {}
    for row in result["rows"]:
        version, frame = row["input_version"], row["query_frame"]
        q = original["P066"]
        r = original["P065"] if version == "original" else recovered
        ri = 48 if version == "original" else 0
        qi = int(np.flatnonzero(q["frame_id"] == frame)[0])
        check(qi == frame and int(r["frame_id"][ri]) == 48, "original frame IDs")
        check(row["query"] == "P066" and row["reference"] == "P065" and row["reference_frame"] == 48, "fixed pair identity")
        check(set(row["fields"]) == set(names), "no feature omitted")
        both = q["mask"][qi] & r["mask"][ri]
        check(row["shared"] == int(both.sum()), "shared count")
        check(row["not_jointly_observed"] == int((~both).sum()), "unknown count")
        check(row["opportunities"] == row["shared"] + row["not_jointly_observed"] == 69, "fixed denominator")
        for j, name in enumerate(names):
            field = row["fields"][name]
            qvalue = float(q["x"][qi, j]) if q["mask"][qi, j] else None
            rvalue = float(r["x"][ri, j]) if r["mask"][ri, j] else None
            check(field["query"] == qvalue, name + "original query")
            check(field["reference"] == rvalue, name + "original reference")
            # One final float32 rounding is the producer's source arithmetic contract.
            difference = float(np.float32(abs(qvalue - rvalue))) if both[j] else None
            check(field["absolute_difference"] == difference, name + "absolute difference or null")
            if difference is not None:
                maximum_error = max(maximum_error, abs(field["absolute_difference"] - difference))
        for axis, a, index in [("query", q, qi), ("reference", r, ri)]:
            valid = all(bool(a["mask"][index, j]) for j in [8, 9, 36, 37])
            delta = float(np.float32(float(a["x"][index, 9]) - float(a["x"][index, 37]))) if valid else None
            check(row[axis + "_palm_left_y_minus_right_y"] == delta, axis + "palm geometry proxy")
        check(row["baseline_path_endpoint"] == (frame == 74), "baseline endpoint flag")
        check(row["alternative_is_corrected_time"] is False and row["unique_correct_stage"] is None, "no oracle timing assertion")
        check(row["whole_sign_judgement"] is None and row["learned_outputs_unchanged"] is True, "no semantic/model promotion")
        by_key[(version, frame)] = row
        brief.append(dict(version=version, query_frame=frame, shared=row["shared"], unknown=row["not_jointly_observed"]))
    for frame in [74, 98]:
        before, after = by_key[("original", frame)], by_key[("reobserved_current_image", frame)]
        for name in names:
            if before["fields"][name]["absolute_difference"] is not None:
                check(before["fields"][name] == after["fields"][name], "prior known difference retained " + str(frame) + ":" + name)
                known_preserved += 1
        check(after["shared"] - before["shared"] == 11, "11 new supported comparisons for same hypothesis")
    check(result["confirmed_feature_by_confirmed_stage_trial"] == "CLOSED_NOT_IDENTIFIABLE", "true factorial unavailable")
    check(result["four_cells_executed"] is True, "sensitivity cells executed")
    check(result["landmark_or_educational_accuracy"] is None, "no accuracy claim")
    for key in ["new_training", "new_classifier_inference", "new_dtw_paths"]:
        check(result[key] == 0, key)
    tech = result["technical"]
    check(tech["natural_performance"] is False and tech["does_not_establish_educational_correctness"] is True, "technical separation")
    check(tech["exhaustive_cell_preservations"] == 7 * 3 ** 7, "combinatorial size, not rerun")
    check(len(tech["checks"]) == 46 and all(x["pass_"] for x in tech["checks"]), "stored selftest declaration")
    table = tech["factorial_corruption_table"]
    check(len(table) == 4, "all synthetic factor states")
    principle = []
    for row in table:
        feature = int(row["feature_values_deliberately_flipped"])
        endpoint = int(row["deliberately_wrong_endpoint"])
        # With a positive reference, crossing changes1→0 and feature inversion
        # changes0→1. Two inversions cancel: (-1)^2=1, not two corrections.
        predicted = (1 - endpoint) if feature == 0 else endpoint
        discrepancy = predicted != 1
        comp = row["comparison"]
        check(comp["query"] == predicted and comp["reference"] == 1, "constructed endpoints")
        check(comp["raw_attribute_difference"] == discrepancy, "XOR discrepancy")
        check(comp["status"] == ("ATTRIBUTE_DIFFERENCE_CANDIDATE" if discrepancy else "SAME_POSITIVE_ATTRIBUTE"), "synthetic status")
        check(row["hidden_errors"] == bool(feature and endpoint), "two errors conceal discrepancy")
        check(row["technical_control_only"] is True and comp["same_action_stage"] is None and comp["educational_correction"] is None, "no natural oracle or correction")
        principle.append(dict(feature_error=bool(feature), endpoint_error=bool(endpoint), predicted=predicted,
                              difference_visible=discrepancy, errors_cancel=bool(feature and endpoint)))
    point_review = read(U / "review/VERIFY_POINT.json")
    point = read(U / "POINT_DIAGNOSIS.json")
    for path, h in point_review["source_pins"].items():
        pin(path, h)
    check(all(x["opportunities"] == 30 and x["usable"] == 26 and x["unknown"] == 4 for x in point["summary"]), "learned original masks remain unchanged")
    for path, h in list(PINS.items()):
        check(sha(path) == h, "source unchanged " + path)
    PINS[str(Path(__file__).resolve())] = sha(__file__)
    report = dict(schema="causal_pilot_combined_independent_verification.v1", status="PASS" if not FAILURES else "FAIL",
        created_utc=datetime.now(timezone.utc).isoformat(), checks=CHECKS, failures=FAILURES,
        geometry_rows=brief, fixed_feature_opportunities_per_row=69, preserved_old_shared_fields=known_preserved,
        maximum_absolute_difference_rebuild_error=maximum_error,
        technical_double_error_principle=principle, verified_point_summary=point["summary"],
        source_pins=PINS, actions=dict(producer_import=0, extractor_or_detector_run=0, classifier_run=0,
            DP_or_path_run=0, old_selftest_rerun=0, original_edits=0),
        interpretation=["38→49 shared features per time hypothesis is observation availability, not a more accurate time choice.",
            "q74/current andq98/another recurrence remain competing hypotheses; no corrected-stage ground truth was introduced.",
            "Technical endpoint inversion and feature inversion can cancel the observable discrepancy; this proves a constructed logical failure mode, not its frequency in natural data.",
            "The old learned30 opportunities still have26usable/4unknown per seed. Recovered geometry does not upgrade saved model probability masks.",
            "No point-classification error or eligible corrected-stage contrast was established here; these results do not establish a need for retraining."],
        limitations=["46 stored pass flags and15309 declared invariant cells are provenance/combinatorial checks, not15309 independently rerun producer tests.",
            "Independent geometry derivation is pinned in VERIFY_RECOVERY; this check verifies the downstream saved69 arithmetic and evidence preservation.",
            "No visual, anatomical, linguistic, learner-error or generalization accuracy is established."])
    with (U / "review/VERIFY_COMBINED.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k:report[k] for k in ["status", "checks", "failures", "geometry_rows"]}, ensure_ascii=False))
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    raise SystemExit(main())
