"""Source/feature-preservation and saved-result arithmetic audit only.

No continuous/discrete solver, model, old path search, or producer import.
Mathematical bracket verification belongs to separate MATH_RESULT_VERIFICATION.
Execute only after root authorizes completed public-output verification.
"""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import ast
import gzip
import hashlib
import json
import math
import statistics
import traceback

HERE = Path(__file__).resolve().parent
U, OLD = HERE.parent, HERE.parent.parent / "word100_hold_invariant_trajectory_20261101"
OUTPUT = HERE / "RESULT_VERIFICATION.json"
checks, failures, pins = 0, [], {}
rows, partitions = [], Counter()
expected_precision_failures = []


def read(p):
    p = Path(p); raw = p.read_bytes()
    return json.loads(gzip.decompress(raw) if p.suffix == ".gz" else raw.decode("utf-8-sig"))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check(ok, label):
    global checks
    checks += 1
    if not ok:
        failures.append(label)


def require(ok, label):
    check(ok, label)
    if not ok:
        raise AssertionError(label)


def pin(p, expected=None):
    p = str(Path(p).resolve())
    actual = sha(p)
    check(expected is None or actual == expected, "SHA " + p)
    if p in pins:
        check(pins[p] == actual, "Source changed during read " + p)
    pins[p] = actual
    return actual


def same(actual, expected, label):
    if isinstance(expected, dict):
        check(set(actual) == set(expected), label + " keys")
        for key in set(actual) & set(expected):
            same(actual[key], expected[key], label + "/" + str(key))
    elif isinstance(expected, list):
        check(isinstance(actual, list) and len(actual) == len(expected), label + " length")
        if isinstance(actual, list):
            for a, b in zip(actual, expected):
                same(a, b, label)
    elif isinstance(expected, float):
        check(actual is not None and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-11), label)
    else:
        check(actual == expected, label)


def stat(values):
    return dict(n=len(values), min=min(values) if values else None, max=max(values) if values else None,
                mean=statistics.fmean(values) if values else None, median=statistics.median(values) if values else None)


def verify_sources(inputs):
    audit = read(HERE / "INPUT_AUDIT.json")
    preservation = read(HERE / "FEATURE_PRESERVATION.json")
    expected = {}
    for mapping in (audit["pins"], audit["protected"], preservation["code_pins"],
                    preservation["original_input_pins"], preservation["all_feature_support_pins"]):
        for p, h in mapping.items():
            require(p not in expected or expected[p] == h, "Consistent preservation SHA expectations")
            expected[p] = h
    for probability in preservation["preserved_native_probabilities"]:
        expected[probability["path"]] = probability["sha256"]
    for p, h in expected.items():
        pin(p, h)
    previous = read(OLD / "review/RESULT_VERIFICATION.json")
    require(previous["status"] == "PASS", "Previous2825 raw-column and path verification remains PASS")
    old_manifest = read(OLD / "RESULT_MANIFEST.json")
    old_records = {r["id"]: r for r in old_manifest["results"]}
    old_evaluation = {r["id"]: r for r in read(OLD / "EVALUATION.json")["rows"]}
    audited_records = {r["id"]: r for r in audit["source_records"]}
    require(len(inputs["records"]) == len(old_records) == len(audited_records) == 2825,
            "Complete prior record inventory")
    require({r["id"] for r in inputs["records"]} == set(old_records) == set(audited_records),
            "No newly selected/dropped/duplicated source record")
    for row in inputs["records"]:
        old, old_value = old_records[row["id"]], old_evaluation[row["id"]]
        check(all(row[k] == old[k] for k in old), "Original source descriptor " + row["id"])
        check(row["old_complete"] == old_value["complete"] and row["old_distance"] == old_value["candidate_distance"],
              "Old discrete distance/status copied, not recomputed " + row["id"])
        check(row["sha256"] == audited_records[row["id"]]["sha256"], "Preregistered old result pin " + row["id"])
    same(inputs["missing"], old_manifest["missing_component_opportunities"], "Original10 missing opportunities")
    check(inputs["possible_opportunities"] == 2835 and inputs["complete_count"] == 1417 and
          inputs["unknown_count"] == 1408 and inputs["natural_count"] == 585, "Immutable full-scope input denominators")
    check(inputs["source_metadata_evaluation_only"] is True, "Source generation metadata isolated from candidate")
    driver = (U / "run.py").read_text(encoding="utf-8-sig")
    check("q=d['query_plot'];ref=d['reference_plot']" in driver and
          "continuous.compare(q['values'],ref['values'],q['mask'],ref['mask'],atol=t['atol'],max_iterations=t['max_iterations'])" in driver,
          "Driver directly supplies saved plots/masks without source smoothing or feature replacement")
    check("handshape_orientation_wrist_bend_verdict=None" in driver and
          "position_distance_is_not_whole_handshape_agreement=True" in driver,
          "Position-only result cannot suppress or decide rotation/wrist bending")
    return old_records


def verify_record(source, item, frozen_hash, eps):
    pin(item["path"], item["sha256"]); output = read(item["path"]); old = read(source["path"])
    require(item["id"] == output["id"] == source["id"], "Output/source ID")
    check(output["source_record"] == source["path"] and output["source_sha256"] == source["sha256"] and
          output["frozen_sha256"] == frozen_hash, source["id"] + ": source/freeze links")
    for field in ("kind", "role", "query", "reference", "component", "control", "parent_id"):
        check(output[field] == source[field], source["id"] + ": identity " + field)
    check(output["old_discrete_distance"] == source["old_distance"], source["id"] + ": prior distance unaltered")
    # Source payload is linked by exact SHA; forbid replacement fields with the
    # same names in the additive record, which could hide the original payload.
    source_fields = ["query_plot", "reference_plot", "query_range", "reference_range",
                     "comparison.query", "comparison.reference", "parent", "baseline_full_path", "edit", "source_readouts"]
    same(output["unchanged_source_fields"], source_fields, source["id"] + ": preserved source namespace")
    check(not any(k in output for k in ("query_plot", "reference_plot", "query_range", "reference_range",
                                       "comparison", "parent", "baseline_full_path", "edit", "source_readouts")),
          source["id"] + ": no shadow replacement of raw source fields")
    for axis in ("query", "reference"):
        plot = old[axis + "_plot"]; scope = old["comparison"][axis]
        check(plot["mask"] == scope["observed_mask"] and len(plot["values"]) == scope["possible_frames"] and
              len(plot["mask"]) == len(scope["frame_id"]) == len(scope["timestamp_us"]) and
              all(len(v) == 2 for v in plot["values"]) and all(len(v) == 2 for v in plot["mask"]),
              source["id"] + ": saved native plots/mask/axis consistent")
    check(output["source_values_copied_or_smoothed"] is False and output["old_path_replaced"] is False and
          output["educational_verdict"] is None and output["handshape_orientation_wrist_bend_verdict"] is None and
          output["position_distance_is_not_whole_handshape_agreement"] is True,
          source["id"] + ": no smoothing, path replacement, shape/rotation/bend verdict")
    complete = all(all(all(row) for row in old[axis + "_plot"]["mask"]) for axis in ("query", "reference"))
    candidate = output["candidate"]; low, high = candidate["lower_bound"], candidate["upper_bound"]
    check(candidate["complete"] == source["old_complete"] == complete, source["id"] + ": complete-mask status")
    if complete:
        if not (0 <= low <= high and high - low <= eps * 1.01):
            expected_precision_failures.append(dict(id=source["id"], error="interval_precision"))
        if high > source["old_distance"] + eps * 1.01:
            expected_precision_failures.append(dict(id=source["id"], error="exceeds_saved_discrete_upper_bound"))
    elif low is not None or high is not None:
        expected_precision_failures.append(dict(id=source["id"], error="missing_promoted"))
    check(complete or (low is None and high is None), source["id"] + ": unknown whole parent remains null")
    row = {k: output[k] for k in ("id", "kind", "role", "query", "reference", "component", "control", "parent_id", "source_record")}
    old_distance = source["old_distance"]
    row.update(complete=complete, old_distance=old_distance, lower=low, upper=high,
        zero_test=bool(candidate.get("zero_distance_certified")),
        upper_within_numerical_tolerance=bool(complete and high <= eps),
        strict_numeric_reduction=bool(complete and high < old_distance - eps),
        relative_reduction=(old_distance - high) / old_distance if complete and old_distance > eps else None,
        edit_outside_parent=bool(old.get("edit", {}).get("relevant_changed_outside_scope", [])),
        semantic_verdict=None, rotation_or_wrist_bend_verdict=None)
    rows.append(row); partitions[source["kind"], complete] += 1


def verify_aggregates(evaluation, manifest, eps):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["kind"], r["role"], r["control"], r["component"]].append(r)
    summary = []
    for key, rr in sorted(grouped.items(), key=lambda x: str(x[0])):
        cc = [r for r in rr if r["complete"]]
        summary.append(dict(kind=key[0], role=key[1], control=key[2], component=key[3],
            opportunities=len(rr), complete=len(cc), unknown=len(rr) - len(cc),
            old_distance=stat([r["old_distance"] for r in cc]), continuous_upper=stat([r["upper"] for r in cc]),
            reduced=sum(r["strict_numeric_reduction"] for r in cc), zero_test=sum(r["zero_test"] for r in cc),
            within_numerical_tolerance=sum(r["upper_within_numerical_tolerance"] for r in cc),
            edit_outside_parent=sum(r["edit_outside_parent"] for r in rr)))
    same(evaluation["summary"], summary, "Every grouped source/result summary")
    controls = []
    for control in sorted({r["control"] for r in rows if r["kind"] == "control"}):
        rr = [r for r in rows if r["control"] == control]; cc = [r for r in rr if r["complete"]]
        missing = [r for r in manifest["missing"] if r["control"] == control]
        controls.append(dict(control=control, opportunities=len(rr) + len(missing), computed=len(rr), missing=len(missing),
            complete=len(cc), unknown=len(rr) - len(cc), zero_test=sum(r["zero_test"] for r in cc),
            within_numerical_tolerance=sum(r["upper_within_numerical_tolerance"] for r in cc),
            positive_above_numerical_tolerance=sum(r["lower"] > eps for r in cc),
            numerically_reduced=sum(r["strict_numeric_reduction"] for r in cc),
            edit_outside_parent=sum(r["edit_outside_parent"] for r in rr)))
    same(evaluation["controls"], controls, "Every technical-control denominator/summary")
    natural = [r for r in rows if r["kind"] == "natural"]; complete = [r for r in natural if r["complete"]]
    same(evaluation["natural"], dict(opportunities=len(natural), complete=len(complete), unknown=len(natural) - len(complete),
        numerically_reduced=sum(r["strict_numeric_reduction"] for r in complete),
        within_numerical_tolerance=sum(r["upper_within_numerical_tolerance"] for r in complete)), "Natural denominator/summary")
    same(evaluation["rows"], rows, "All2825 published saved-output rows")
    same(evaluation["failures"], expected_precision_failures, "Frozen arithmetic/coverage gate")
    check(evaluation["failure_count"] == len(expected_precision_failures), "Frozen gate failure count")
    check(sum(c["opportunities"] for c in controls) == 2250 and sum(c["computed"] for c in controls) == 2240 and
          sum(c["missing"] for c in controls) == 10, "Control2250 denominator maintained")
    same(evaluation["decision"], dict(technical_pass=not expected_precision_failures, production_adopted=False,
        educational_accuracy=None, recognition_accuracy=None, smaller_continuous_distance_not_accuracy_gain=True,
        rotation_or_wrist_bend_detection_validated=False), "Position-only interpretation and no adoption")
    return controls, evaluation["natural"]


def main():
    for name in ("FROZEN.json", "RESULT_MANIFEST.json", "EVALUATION.json"):
        require((U / name).exists(), "Completed public result required: " + name)
    frozen = read(U / "FROZEN.json"); frozen_hash = pin(U / "FROZEN.json")
    for section in ("pins", "protected"):
        for p, h in frozen[section].items():
            pin(p, h)
    inputs, manifest, evaluation = (read(U / p) for p in ("INPUTS.json", "RESULT_MANIFEST.json", "EVALUATION.json"))
    pin(U / "RESULT_MANIFEST.json"); pin(U / "EVALUATION.json")
    verify_sources(inputs)
    by_id = {r["id"]: r for r in inputs["records"]}
    require(len(manifest["results"]) == 2825 and {r["id"] for r in manifest["results"]} == set(by_id),
            "Exact new2825 record inventory")
    same(manifest["missing"], inputs["missing"], "All10 unavailable records remain")
    same(evaluation["missing"], inputs["missing"], "All10 unavailable records in report")
    check(frozen["orientation_or_wrist_bend_classifier_trained"] is False and
          frozen["shape_features_changed"] is False and frozen["whole_sign_verdict"] is False,
          "Frozen feature capability boundaries")
    for i, item in enumerate(manifest["results"]):
        verify_record(by_id[item["id"]], item, frozen_hash, inputs["atol"])
        if (i + 1) % 500 == 0:
            print(f"Preserved sources and result rows {i + 1}/2825", flush=True)
    check(partitions == {("natural", True): 293, ("natural", False): 292,
                          ("control", True): 1124, ("control", False): 1116},
          "All complete/unknown partitions unchanged")
    controls, natural = verify_aggregates(evaluation, manifest, inputs["atol"])
    check(manifest["new_training"] == 0 and manifest["new_model_inference"] == 0 and
          manifest["old_path_recalculation"] == 0 and manifest["source_originals_changed"] is False,
          "No old model/path/source execution")
    for p, h in list(pins.items()):
        check(sha(p) == h, "Source unchanged during independent verification " + p)
    return controls, natural


if __name__ == "__main__":
    if OUTPUT.exists():
        raise SystemExit("First verification result must be preserved; no overwrite.")
    error, controls, natural = None, None, None
    try:
        controls, natural = main()
    except Exception:
        error = traceback.format_exc(); failures.append("Verification exception")
    report = dict(schema="wis_continuous_source_scope_feature_preservation_verification_v1",
        status="PASS" if not failures else "FAIL", at_utc=datetime.now(timezone.utc).isoformat(),
        reviewer="Codex semantic_data_audit", checks=checks, failure_count=len(failures), failures=failures[:100],
        exception=error, verified_records=len(rows),
        source_partition_counts={str(k):v for k,v in partitions.items()},
        aggregate_natural=natural, aggregate_controls=controls, reproduced_gate_failures=expected_precision_failures,
        mathematical_scope="This verifier audits sources, masks, capabilities, copied distances and published arithmetic. Independent exact/LP continuous bracket verification is separate review/MATH_RESULT_VERIFICATION.json and is not rerun here.",
        feature_preservation="All original69-column input files,108 readouts and3 native output files are unchanged by source hashes. New records only append position-curve candidates and cannot assert shape/rotation/wrist-bend agreement.",
        limits=["Small position distance is not evidence that fist rotation or wrist bending is absent, noise, or correct.",
                "No actual rotation/wrist-bend classifier or educational benchmark was added.",
                "Polylinear interpolation represents assumed connectors between observed position samples, not new measured intermediate frames.",
                "Historical public development and synthetic controls are not fresh independent sign-language correctness data."],
        actions=dict(new_models=0, inference=0, old_path_recalculation=0, old_distance_recalculation=0,
                     continuous_solver_execution=0, new_labels=0, raw_video_access=0,
                     user_or_mentor_video_access=0, old_files_modified=0),
        pins=pins, script_sha256=sha(__file__))
    with OUTPUT.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(dict(status=report["status"], checks=checks, failure_count=len(failures),
                         records=len(rows), output=str(OUTPUT), sha256=sha(OUTPUT))))
    if failures:
        raise SystemExit(1)
