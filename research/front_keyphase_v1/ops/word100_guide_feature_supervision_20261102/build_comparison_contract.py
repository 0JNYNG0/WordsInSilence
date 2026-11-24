"""Compile audited source predicates. No model outputs or evaluation labels."""
from pathlib import Path
import hashlib
import json

U = Path(__file__).resolve().parent
O = U.parent


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    dst = U / "COMPARISON_CONTRACT.json"
    if dst.exists():
        raise FileExistsError("The comparison contract is immutable once compiled")
    source = U / "SOURCE_FEATURE_CONSTRAINT_AUDIT.json"
    physical = U / "PHYSICAL_GUIDE_BINDING_AUDIT.json"
    audit = read(source)
    assert audit["source_clauses"] == 11 and all(c["pass"] for c in audit["checks"])
    assert read(physical)["checks"] == 1590
    runtime = {
        "right_non_thumb_closed": "point",
        "right_hand_at_left_arm_projection": "point",
        "bilateral_upper_ends_converged": "point",
        "right_hand_traverses_left_arm_toward_wrist": "interval",
    }
    necessary = {"right_non_thumb_closed", "both_visible_thumbs_up_with_closed_finger_masses"}
    rows = []
    for row in audit["rows"]:
        for target in row["existing_typed_component_ids"] or [None]:
            rows.append(dict(
                clause_id=row["clause_id"], word_id=row["word_id_for_source_lookup_only"],
                text=row["text"], entry_id=row["entry_id"], url=row["url"], source=row["source"],
                component_id=target,
                condition_relation="NECESSARY_PARTIAL" if target in necessary else "RELATED_ONLY",
                expected_component_value=1 if target is not None else None,
                expectation_is_training_label=False,
                runtime_observer=runtime.get(target),
                conditional_on_correct_source_variant=True,
                temporal_binding_requirements=row["temporal_binding_requirements"],
                logical_implication=row["logical_implication"],
                unsupported=row["unsupported"],
                old_requiredness=row["requiredness"],
                old_whole_clause_verified=row["whole_clause_verified"],
                correction_enabled=False,
            ))
    pins = {str(p): digest(p) for p in [source, physical, Path(__file__),
        O / "word100_learned_component_extent_20261102/source_bridge.py"]}
    contract = dict(
        version="source-observation-comparison-v1", constraints=rows, pins=pins,
        scope="Additive comparison of existing learned observations with source components; not new classification or educational correctness",
        point_thresholds=dict(compatible_min=.8, counter_max=.2, joint_support_min=.5),
        threshold_provenance="Unchanged source_bridge.py .8/.2/.5; not selected from current DEV results",
        time_policy="DTW source windows remain PROPOSED_DTW. Neither model confidence nor word ID proves local applicability.",
        aggregation_policy="Retain every native point and every short counterpoint; mixed, absent, estimated and uncertain remain separate. No averaging into a whole-window pass.",
        low_geometry_policy="Both positive and negative high-confidence predictions remain IMAGE_TIME_ESTIMATE",
        motion_policy="Preserve the prior exact-scope movement evidence; no endpoint-only completion or negative inference from low probability",
        required_tests=["normal_and_crossing_output_preservation", "all_native_points_preserved", "missingness_denominator_preserved", "short_counterpoint_preserved", "transition_not_guide_violation", "variant_isolation", "source_provenance", "no_model_reinference_or_rescoring"],
        saved_public_scope="All 16 public saved cases in partial_order_live/SAVED_CASES_MANIFEST, four expressions by normal/deletion/reverse/masked. These are engineering controls, not learner-error gold.",
        public_comparison_method="Append the new comparison field to immutable saved outputs; inspect exact preservation and new behavior. Do not rerun models or recompute prior performance scores.",
        first_schema_inspection="Only one saved WORD0739 normal result/schema inspected before this contract; not used to select thresholds or candidates",
        new_training=0, new_model_inference=0, educational_error_truth=False,
        accuracy_gain_claim=False, whole_goal_complete=False,
    )
    dst.write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(path=str(dst), sha256=digest(dst), constraints=len(rows)), ensure_ascii=False))


if __name__ == "__main__":
    main()
