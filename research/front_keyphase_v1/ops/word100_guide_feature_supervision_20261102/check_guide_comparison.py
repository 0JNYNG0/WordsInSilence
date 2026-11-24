"""Behavioral checks independent of public model scores."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
from guide_comparison import GuideComparator, point_state, summarize

U = Path(__file__).resolve().parent


def main():
    c = GuideComparator()
    t = c.contract["point_thresholds"]
    checks = []

    def check(name, ok):
        checks.append(dict(name=name, passed=bool(ok)))
        assert ok, name

    def point(p=.9, g=1., av=True):
        return dict(probability=p, joint_support=g, image_available=av)

    for p, expected in [(.9, "MODEL_COMPATIBLE"), (.1, "MODEL_COUNTER_EVIDENCE"), (.5, "UNCERTAIN")]:
        check("observed_direction_" + str(p), point_state(point(p), t)[0] == expected)
        check("missing_cannot_be_confirmed_" + str(p), point_state(point(p, av=False), t)[0] == "UNOBSERVED")
    for p in [.9, .1]:
        check("both_directions_need_geometry_" + str(p), point_state(point(p, g=.1), t)[0] == "IMAGE_TIME_ESTIMATE")
    check("no_probability_is_unknown", point_state(point(None), t)[0] == "UNOBSERVED")
    for p in [float("nan"), -1, 2]:
        caught = False
        try:
            point_state(point(p), t)
        except ValueError:
            caught = True
        check("bad_probability_rejected_" + str(p), caught)
    points = [dict(frame=i, observation=s) for i, s in enumerate([
        "MODEL_COMPATIBLE", "MODEL_COMPATIBLE", "MODEL_COUNTER_EVIDENCE", "MODEL_COMPATIBLE", "UNOBSERVED"])]
    s = summarize(points)
    check("one_short_counterpoint_not_averaged_away", s["status"] == "MIXED_OBSERVATIONS" and s["counts"]["MODEL_COUNTER_EVIDENCE"] == 1)
    check("missing_does_not_shrink_denominator", s["total_points"] == 5 and s["coverage"] == .8)
    check("partial_support_is_not_whole_window", not s["whole_window_confirmed"] and s["has_unresolved_points"])
    check("all_missing_is_not_success", summarize([dict(observation="UNOBSERVED")])["status"] == "INSUFFICIENT_OBSERVATION")
    check("estimated_negative_is_not_counterpoint", summarize([dict(observation="IMAGE_TIME_ESTIMATE")])["status"] == "UNCERTAIN_OR_ESTIMATED_ONLY")

    rule = next(r for r in c.constraints if r["component_id"] == "right_non_thumb_closed")
    clause = dict(id=rule["clause_id"], text=rule["text"], source=rule["source"], url=rule["url"])
    pp = [dict(frame=i, seconds=i / 30., status="UNCHANGED_BASE", **point(p)) for i, p in enumerate([.95, .05, .95])]
    base = dict(word_id=rule["word_id"], private_dummy_for_preservation=[1, 2], source_components=[dict(
        component_id=rule["component_id"], source_clauses=[clause], frame_ids=[0, 1, 2],
        point_evidence=pp, status="POINT_EVIDENCE_AVAILABLE")])
    # Native integer microsecond timestamps, not an invented rounded seconds axis.
    us = [0, 33333, 66667]
    for p, tt in zip(pp, us):
        p["seconds"] = tt / 1e6
    original = deepcopy(base)
    out = c.attach(base, [0, 1, 2], us)
    row = out["guide_feature_comparison"]["items"][0]
    check("source_and_observation_independent", row["condition_relation"] == "NECESSARY_PARTIAL" and row["summary"]["status"] == "MIXED_OBSERVATIONS")
    check("DTW_cannot_verify_applicability", row["time_binding"] == "PROPOSED_DTW" and not row["correction_enabled"])
    check("counterpoint_not_semantic_violation", row["native_points"][1]["guide_violation"] is None)
    check("all_old_fields_and_input_untouched", base == original and all(out[k] == v for k, v in base.items()))
    check("every_native_point_retained", [p["frame"] for p in row["native_points"]] == [0, 1, 2])
    check("unavailable_component_is_explicit", any(i["summary"]["status"] == "NO_BOUND_RUNTIME_OBSERVER" for i in out["guide_feature_comparison"]["items"]))
    for name, mutate in [
        ("changed_variant", lambda b: b["source_components"][0]["source_clauses"][0]["source"].update(json_pointer="wrong")),
        ("lost_middle_frame", lambda b: b["source_components"][0].update(frame_ids=[0, 2])),
        ("changed_native_time", lambda b: b["source_components"][0]["point_evidence"][1].update(seconds=999)),
        ("duplicate_component", lambda b: b["source_components"].append(deepcopy(b["source_components"][0]))),
    ]:
        invalid = deepcopy(base)
        mutate(invalid)
        caught = False
        try:
            c.attach(invalid, [0, 1, 2], us)
        except ValueError:
            caught = True
        check(name + "_rejected", caught)
    neutral = c.attach(dict(word_id="UNKNOWN", source_components=[]), [0], [0])
    check("unsupported_word_has_no_borrowed_rules", neutral["guide_feature_comparison"]["items"] == [])
    result = dict(status="PASS", checks=checks, check_count=len(checks),
        code_sha256=hashlib.sha256((U / "guide_comparison.py").read_bytes()).hexdigest(),
        contract_sha256=c.contract_sha, synthetic_behavior_checks_not_performance=True,
        new_model_inference=0, new_training=0, whole_goal_complete=False)
    (U / "COMPARATOR_CHECKS.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(status=result["status"], checks=len(checks))))


if __name__ == "__main__":
    main()
