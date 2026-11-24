"""Read-only categorical feature feedback diagnostic; never a KSL grader.

The two comparisons are deliberately small and inspectable.  The pointwise
baseline compares normalized time.  The ordered safeguard preserves all
observed state changes, abstains per feature on missing evidence, and supplies
a monotone DTW path as a diagnostic (not the production alignment algorithm).
Saved RGB labels, synthetic manipulations and learned outputs have different
provenance.  No learned-model or educational accuracy is inferred from labels.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

UNIT = Path(__file__).resolve().parent
OPS = UNIT.parent
SCHEMA = "wis_goal_feature_comparison_v1"
PROBABILITY_THRESHOLD = 0.5
MATCH = "OBSERVED_PATTERN_COMPATIBLE"
DIFFERENT = "OBSERVED_PATTERN_DIFFERENCE_CANDIDATE"
ABSTAIN = "INSUFFICIENT_SEQUENCE_COVERAGE"
UNSUPPORTED = "UNSUPPORTED_FEATURE"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    """Evaluation outputs are append-only artifacts, not overwrite targets."""
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def make_track(track_id, features, *, sample_ids=None, times=None,
               scope="recorded_samples_only", evidence="unspecified", metadata=None):
    if not features:
        raise ValueError("At least one recorded feature is required")
    lengths = {len(v) for v in features.values()}
    if len(lengths) != 1 or not next(iter(lengths)):
        raise ValueError("Features require the same positive original sample denominator")
    n = next(iter(lengths))
    for values in features.values():
        if any(v is not None and not (type(v) is int and v in (0, 1))
               and not (type(v) is str and v) for v in values):
            raise ValueError("Only explicit categorical strings, binary integers or null")
    sample_ids = list(range(n)) if sample_ids is None else list(sample_ids)
    times = list(range(n)) if times is None else list(times)
    if len(sample_ids) != n or len(times) != n:
        raise ValueError("Sample IDs and times must preserve the full denominator")
    if not all(type(v) in (int, float) and math.isfinite(v) for v in times):
        raise ValueError("Nonfinite or nonnumeric sample time")
    if not all(a < b for a, b in zip(times, times[1:])):
        raise ValueError("Strict sample time order required")
    return {"track_id": str(track_id), "features": copy.deepcopy(features),
            "sample_ids": sample_ids, "times": times, "scope": scope,
            "evidence": evidence, "metadata": copy.deepcopy(metadata or {}),
            "whole_sign_truth": False, "mandatory_ksl_truth": False}


def track_from_probabilities(track_id, probabilities, masks, target_names,
                             supported_features, **kwargs):
    """Optional learned-head adapter; .5 is classification, never confidence.

    A mask must reflect actual input/target observability, not merely a finite
    prediction. Unsupported targets are omitted, hence requested comparisons
    explicitly report UNSUPPORTED_FEATURE. No inference is performed here.
    The caller must describe time scope accurately (e.g. three centre samples).
    """
    if len(probabilities) != len(masks) or not probabilities:
        raise ValueError("Probabilities and masks need identical nonempty sample axes")
    if len(set(target_names)) != len(target_names):
        raise ValueError("Duplicate target names")
    supported = set(supported_features)
    if not supported <= set(target_names):
        raise ValueError("Supported feature not present in model target names")
    columns = {f: [] for f in target_names if f in supported}
    for row, mask in zip(probabilities, masks):
        if len(row) != len(target_names) or len(mask) != len(target_names):
            raise ValueError("Probability/mask target axis mismatch")
        for i, feature in enumerate(target_names):
            if feature not in columns:
                continue
            p, valid = row[i], mask[i]
            if type(valid) is not bool:
                raise ValueError("Mask values must be explicit booleans")
            if p is not None and (not isinstance(p, (int, float)) or isinstance(p, bool)):
                raise ValueError("Probability must be numeric or null")
            if p is not None and math.isfinite(p) and not 0 <= p <= 1:
                raise ValueError("Probability outside [0,1]")
            value = None if not valid or p is None or not math.isfinite(p) else int(p >= 0.5)
            columns[feature].append(value)
    kwargs.setdefault("evidence", "learned_prediction_not_direct_RGB_truth")
    return make_track(track_id, columns, **kwargs)


def runs(values):
    out = []
    for i, value in enumerate(values):
        if out and out[-1]["value"] == value:
            out[-1]["end"] = i
        else:
            out.append({"value": value, "start": i, "end": i})
    return out


def bounded_changes(values):
    known = [i for i, value in enumerate(values) if value is not None]
    return [{"from_index": a, "to_index": b, "from_value": values[a],
             "to_value": values[b], "unknown_indices": list(range(a + 1, b)),
             "exact_transition_time": None, "linguistic_importance": None}
            for a, b in zip(known, known[1:]) if values[a] != values[b]]


def sequence_coverage(values):
    known = sum(v is not None for v in values)
    return {"known": known, "expected": len(values), "unknown": len(values) - known,
            "fraction": known / len(values)}


def pointwise(query, reference):
    n = max(len(query), len(reference))
    # Conventional round-half-up, explicitly independent of Python round ties.
    path = [(int(i * (len(query) - 1) / max(n - 1, 1) + .5),
             int(i * (len(reference) - 1) / max(n - 1, 1) + .5)) for i in range(n)]
    known = [(i, j) for i, j in path if query[i] is not None and reference[j] is not None]
    mismatches = [(i, j) for i, j in known if query[i] != reference[j]]
    return {"path": path, "expected_grid_cells": n, "jointly_known_grid_cells": len(known),
            "common_known_fraction": len(known) / n, "mismatches": mismatches,
            "raw_difference_candidate": bool(mismatches),
            "unknown_is_match": False, "educational_warning": None}


def categorical_dtw(query, reference):
    n, m = len(query), len(reference)
    cost = [[math.inf] * (m + 1) for _ in range(n + 1)]
    previous = {}
    cost[0][0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            options = [(cost[i-1][j-1], i-1, j-1),
                       (cost[i-1][j], i-1, j), (cost[i][j-1], i, j-1)]
            parent = min(options, key=lambda v: v[0])
            local = 0 if query[i-1] is not None and reference[j-1] is not None \
                and query[i-1] == reference[j-1] else 1
            cost[i][j] = parent[0] + local
            previous[(i, j)] = parent[1:]
    path, i, j = [], n, m
    while i or j:
        path.append((i-1, j-1))
        i, j = previous[(i, j)]
    path.reverse()
    known = [(i, j) for i, j in path if query[i] is not None and reference[j] is not None]
    return {"path": path, "categorical_cost": cost[n][m],
            "known_mismatch_cells": sum(query[i] != reference[j] for i, j in known),
            "unknown_path_cells": len(path) - len(known),
            "distinct_jointly_known_query_samples": len({i for i, _ in known}),
            "distinct_jointly_known_reference_samples": len({j for _, j in known}),
            "cost_is_linguistic_similarity": False,
            "same_state_path_not_unique_physical_correspondence": True}


def compare_feature(query, reference):
    qr, rr = runs(query), runs(reference)
    complete = all(v is not None for v in query + reference)
    signature_q, signature_r = [r["value"] for r in qr], [r["value"] for r in rr]
    verdict = (MATCH if signature_q == signature_r else DIFFERENT) if complete else ABSTAIN
    # Unknown runs remain barriers. Equal states across a gap are not merged.
    anchors_q = [r["value"] for r in qr if r["value"] is not None]
    anchors_r = [r["value"] for r in rr if r["value"] is not None]
    return {"status": verdict, "complete_recorded_pattern_comparable": complete,
            "query_coverage": sequence_coverage(query), "reference_coverage": sequence_coverage(reference),
            "query_runs": qr, "reference_runs": rr,
            "query_bounded_changes": bounded_changes(query),
            "reference_bounded_changes": bounded_changes(reference),
            "observed_anchor_order_equal": (anchors_q == anchors_r) if anchors_q and anchors_r else None,
            "partial_anchor_order_is_complete": False,
            "pointwise": pointwise(query, reference), "dtw": categorical_dtw(query, reference),
            "educational_warning": None, "linguistic_tolerance": None,
            "exact_repetition_requirement": None}


def compare_tracks(query, reference, goal_features):
    started = time.perf_counter()
    features = {}
    for feature in goal_features:
        if feature not in query["features"] or feature not in reference["features"]:
            features[feature] = {"status": UNSUPPORTED, "educational_warning": None}
        else:
            features[feature] = compare_feature(query["features"][feature], reference["features"][feature])
    return {"schema": SCHEMA, "query_id": query["track_id"], "reference_id": reference["track_id"],
            "query_scope": query["scope"], "reference_scope": reference["scope"],
            "query_evidence": query["evidence"], "reference_evidence": reference["evidence"],
            "query_sample_ids": query["sample_ids"], "reference_sample_ids": reference["sample_ids"],
            "query_times": query["times"], "reference_times": reference["times"],
            "query_metadata": query["metadata"], "reference_metadata": reference["metadata"],
            "features": features, "elapsed_seconds": time.perf_counter() - started,
            "whole_sign_verdict": None, "educational_accuracy": None}


def compare_references(query, references, goal_features):
    ids = [r["track_id"] for r in references]
    if len(set(ids)) != len(ids):
        raise ValueError("Reference IDs must be unique; averaging/pooling is forbidden")
    return {"query_id": query["track_id"], "references": [compare_tracks(query, r, goal_features)
            for r in references], "reference_pooling": None, "whole_sign_verdict": None}


def selftest():
    checks = []

    def check(name, value):
        checks.append({"name": name, "pass": bool(value)})
        if not value:
            raise AssertionError(name)

    for name, q, r, wanted in [
        ("speed", [0, 0, 0, 1, 1], [0, 1], MATCH),
        ("single_sample_change_preserved", [0, 0, 0], [0, 1, 0], DIFFERENT),
        ("opposite_order", [1, 0], [0, 1], DIFFERENT),
        ("missing_middle", [0, None, 0], [0, 1, 0], ABSTAIN),
        ("null_never_zero", [None, None], [0, 0], ABSTAIN),
        ("null_never_match", [None], [None], ABSTAIN),
        ("repeat_observed_only", [0, 1, 0, 1], [0, 1], DIFFERENT),
        ("categorical_relation", ["above", "below"], ["below", "above"], DIFFERENT)]:
        result = compare_feature(q, r)
        check(name, result["status"] == wanted)
        path = result["dtw"]["path"]
        check(name + "_path_endpoints", path[0] == (0, 0) and path[-1] == (len(q)-1, len(r)-1))
        check(name + "_path_monotone", all((c-a, d-b) in ((1,1), (1,0), (0,1))
              for (a,b),(c,d) in zip(path,path[1:])))
    check("gap_not_bridged", len(runs([0, None, 0])) == 3)
    check("bounded_change_preserved", bounded_changes([0, None, 1])[0]["unknown_indices"] == [1])
    q = make_track("q", {"shape": [0, None, 1], "relation": [0, 1, 1]},
                   metadata={"body_x": [0., .002, -.002]})
    r = make_track("r", {"shape": [0, 0, 1], "relation": [0, 1, 1]},
                   metadata={"body_x": [0., 0., 0.]})
    before = copy.deepcopy((q, r))
    comparison = compare_tracks(q, r, ["shape", "relation", "contact"])
    check("feature_local_unknown", comparison["features"]["shape"]["status"] == ABSTAIN
          and comparison["features"]["relation"]["status"] == MATCH)
    check("unsupported_contact", comparison["features"]["contact"]["status"] == UNSUPPORTED)
    check("immutable_input", (q, r) == before)
    check("body_drift_architectural_separation", comparison["features"]["relation"]["status"] == MATCH)
    r2 = copy.deepcopy(r); r2["track_id"] = "other_reference"; r2["features"]["relation"] = [1, 0, 1]
    separate = compare_references(q, [r, r2], ["relation"])
    check("individual_refs_preserved", [x["features"]["relation"]["status"] for x in separate["references"]]
          == [MATCH, DIFFERENT] and separate["reference_pooling"] is None)
    adapted = track_from_probabilities("p", [[.499, .5, float("nan"), .9], [.5, .3, .1, .8]],
        [[True, True, True, False], [True, False, True, True]],
        ["a", "b", "c", "unsupported"], ["a", "b", "c"], scope="three_centres_not_dense")
    check("fixed_threshold", adapted["features"]["a"] == [0, 1])
    check("mask_nonfinite_abstain", adapted["features"]["b"] == [1, None]
          and adapted["features"]["c"] == [None, 0])
    check("unsupported_not_inferred", "unsupported" not in adapted["features"])
    for name, call in [
        ("reject_float_state", lambda: make_track("x", {"f": [.2]})),
        ("reject_boolean_state", lambda: make_track("x", {"f": [True]})),
        ("reject_unsorted_time", lambda: make_track("x", {"f": [0,1]}, times=[2,1])),
        ("reject_bad_probability", lambda: track_from_probabilities("x", [[1.1]], [[True]], ["f"], ["f"])),
        ("reject_missing_mask", lambda: track_from_probabilities("x", [[.2]], [[None]], ["f"], ["f"])),
        ("reject_duplicate_reference", lambda: compare_references(q,[r,r],["shape"]))]:
        try:
            call()
        except ValueError:
            check(name, True)
        else:
            check(name, False)
    return {"schema": SCHEMA, "status": "PASS", "checks": checks, "count": len(checks),
            "author": "Codex semantic_feedback implementer selftest; independent review pending",
            "body_drift_is_architecture_test_only": True,
            "new_model_end_to_end_feedback_validated": False,
            "real_educational_accuracy": None}


def load_sources():
    frozen = read_json(UNIT / "FROZEN_EVALUATION.json")
    for path, digest in frozen["source_pins"].items():
        if sha256(path) != digest:
            raise ValueError("Frozen input hash changed: " + path)
    bundle = read_json(OPS / "word100_shape_sequence_bundle_20261027/BUNDLE.json")
    physical = read_json(OPS / "word100_crossperson_stage_audit_20261101/PHYSICAL_CORRESPONDENCE.json")
    metadata = read_json(OPS / "word100_core_stage_evidence_20261031/PHASE_DATA.json")
    return bundle, physical, {c["alias"]: c for c in metadata["clips"]}


def evaluate_saved():
    approval = read_json(UNIT / "EVALUATION_APPROVAL.json")
    if not approval.get("approved") or approval["proposal_sha256"] != sha256(UNIT / "FROZEN_EVALUATION.json"):
        raise ValueError("Root approval for this exact frozen proposal required")
    bundle, physical, metadata = load_sources()
    cases = []

    def record(family, case_id, query, reference, *, provenance, expected=None):
        start = time.perf_counter()
        result = compare_feature(query, reference)
        cases.append({"family": family, "case_id": case_id,
                      "query_values": query, "reference_values": reference,
                      "provenance": provenance, "expected_status": expected,
                      "technical_expectation_met": result["status"] == expected if expected else None,
                      "result": result, "elapsed_seconds": time.perf_counter() - start})

    for sequence in bundle["sequences"]:
        for feature, saved in sequence["features"].items():
            values = saved["values"]
            direct = [f["direct"][feature] for f in sequence["frames"]]
            if values != direct:
                raise ValueError("Saved sequence/direct RGB label disagreement")
            cid = sequence["case_id"] + "/" + feature
            provenance = {"source": "BUNDLE.json", "case_id": sequence["case_id"],
                "feature": feature, "role": sequence["role"], "word_id": sequence["word_id"],
                "recorded_person_id": sequence["recorded_person_id"], "side": sequence["side"],
                "frame_ids": [f["frame_id"] for f in sequence["frames"]],
                "pts_us": [f["pts_us"] for f in sequence["frames"]],
                "scope": "13 recorded frames, one hand, two predicates; not a whole word",
                "source_truth": sequence["source_label_status"]}
            expected = MATCH if all(v is not None for v in values) else ABSTAIN
            record("shape_identity", cid, values[:], values, provenance=provenance, expected=expected)
            mapping = [i for i in range(len(values)) for _ in range(1 + i % 3)]
            record("shape_timewarp", cid, [values[i] for i in mapping], values,
                   provenance={**provenance, "synthetic_source_indices": mapping}, expected=expected)
            complete = all(v is not None for v in values)
            if complete and len(runs(values)) >= 2:
                reversed_values = values[::-1]
                if [r["value"] for r in runs(reversed_values)] != [r["value"] for r in runs(values)]:
                    record("shape_order_reversal", cid, reversed_values, values,
                           provenance={**provenance, "synthetic": True}, expected=DIFFERENT)
                for idx, run in enumerate(runs(values)):
                    altered = values[:run["start"]] + values[run["end"] + 1:]
                    if altered:
                        record("shape_run_omission", cid + "/run" + str(idx), altered, values,
                               provenance={**provenance, "omitted_run": run, "synthetic": True}, expected=DIFFERENT)
            for idx, link in enumerate(bounded_changes(values)):
                a, b = link["from_index"], link["to_index"]
                interval = values[a:b+1]
                expected_change = ABSTAIN if any(v is None for v in interval) else DIFFERENT
                record("shape_bounded_change", cid + "/change" + str(idx), [values[a]] * len(interval), interval,
                    provenance={**provenance, "source_interval_indices": [a,b], "synthetic": True},
                    expected=expected_change)
            for index, value in enumerate(values):
                if value is not None:
                    missing = values[:]; missing[index] = None
                    record("shape_single_missing", cid + "/missing" + str(index), missing, values,
                           provenance={**provenance, "masked_index": index, "synthetic": True}, expected=ABSTAIN)
            record("shape_all_missing", cid, [None] * len(values), values,
                   provenance={**provenance, "synthetic": True}, expected=ABSTAIN)

    native = {}
    physical_inventory = []
    for clip in physical["clips"]:
        alias = clip["alias"]
        values = [None] * clip["native_frame_count"]
        cores = [r for r in clip["regions"] if r["region_type"] == "observed_physical_core"]
        for core in cores:
            a, b = core["core"]
            if not 0 <= a <= b < len(values):
                raise ValueError("Invalid physical core interval")
            for index in range(a, b + 1):
                if values[index] is not None:
                    raise ValueError("Overlapping physical cores")
                values[index] = core["kind"]
        if sum(v is not None for v in values) != clip["reviewed_known_frames"]:
            raise ValueError("Reviewed native known-frame count mismatch")
        native[alias] = values
        tokens = [r["kind"] for r in cores]
        physical_inventory.append({"alias": alias, "word_id": metadata[alias]["word_id"],
            "person": metadata[alias]["person"], "role": clip["role"],
            "coverage": sequence_coverage(values), "core_tokens": tokens,
            "uncertain_transition_frames": clip["uncertain_transition_frames"],
            "outside_reviewed_scope_frames": clip["outside_reviewed_scope_frames"]})
        provenance = {"source": "PHYSICAL_CORRESPONDENCE.json", "alias": alias,
            "scope": "sparse observed physical core anchors; intervening frames not relabeled",
            "linguistic_truth": False, "synthetic": True,
            "core_source_regions": [{"id": r["id"], "core": r["core"], "kind": r["kind"]} for r in cores]}
        expanded = [v for i,v in enumerate(tokens) for _ in range(1+i%3)]
        record("physical_anchor_timewarp", alias, expanded, tokens, provenance=provenance, expected=MATCH)
        if [r["value"] for r in runs(tokens[::-1])] != [r["value"] for r in runs(tokens)]:
            record("physical_anchor_reversal", alias, tokens[::-1], tokens, provenance=provenance, expected=DIFFERENT)
        for index in range(len(tokens)):
            changed = tokens[:index] + tokens[index+1:]
            if changed:
                expected = MATCH if [r["value"] for r in runs(changed)] == [r["value"] for r in runs(tokens)] else DIFFERENT
                record("physical_anchor_omission", alias + "/anchor" + str(index), changed, tokens,
                    provenance={**provenance, "omitted_anchor_index": index,
                                "same_adjacent_state_duration_change_only": expected == MATCH}, expected=expected)

    crossperson = []
    for query in physical_inventory:
        refs = [r for r in physical_inventory if r["word_id"] == query["word_id"]
                and r["role"] == "fit" and r["person"] != query["person"]]
        qid = query["alias"]
        for ref in refs:
            rid = ref["alias"]
            result = compare_feature(native[qid], native[rid])
            sparse = compare_feature(query["core_tokens"], ref["core_tokens"])
            crossperson.append({"query": qid, "reference": rid, "word_id": query["word_id"],
                "query_person": query["person"], "reference_person": ref["person"], "role": query["role"],
                "native_frame_diagnostic": result,
                "sparse_anchor_diagnostic": sparse,
                "sparse_anchor_scope": "Observed cores only. Exact equality is descriptive, not complete-video compatibility.",
                "real_learner_error": None, "educational_false_alert": None,
                "reference_individual_preserved": True})
    if len(crossperson) != 36:
        raise ValueError("Frozen cross-person pair denominator changed")
    families = {}
    for family in sorted({c["family"] for c in cases}):
        selected = [c for c in cases if c["family"] == family]
        families[family] = {"cases": len(selected),
            "status_counts": dict(Counter(c["result"]["status"] for c in selected)),
            "pointwise_raw_difference_candidates": sum(c["result"]["pointwise"]["raw_difference_candidate"] for c in selected),
            "technical_expectation_met": sum(c["technical_expectation_met"] is True for c in selected),
            "common_known_grid_cells": sum(c["result"]["pointwise"]["jointly_known_grid_cells"] for c in selected),
            "expected_grid_cells": sum(c["result"]["pointwise"]["expected_grid_cells"] for c in selected)}
    role_summaries = {}
    for role in ("fit", "development"):
        selected = [c for c in crossperson if c["role"] == role]
        role_summaries[role] = {"pairs": len(selected),
            "native_status_counts": dict(Counter(c["native_frame_diagnostic"]["status"] for c in selected)),
            "sparse_anchor_status_counts": dict(Counter(c["sparse_anchor_diagnostic"]["status"] for c in selected)),
            "pointwise_raw_difference_candidate_pairs": sum(c["native_frame_diagnostic"]["pointwise"]["raw_difference_candidate"] for c in selected),
            "common_known_grid_cells": sum(c["native_frame_diagnostic"]["pointwise"]["jointly_known_grid_cells"] for c in selected),
            "expected_grid_cells": sum(c["native_frame_diagnostic"]["pointwise"]["expected_grid_cells"] for c in selected)}
    all_met = all(c["technical_expectation_met"] for c in cases)
    load_sources()  # source hashes still unchanged after all comparisons
    return {"schema": SCHEMA, "status": "PASS_TECHNICAL_ONLY" if all_met else "FAIL_TECHNICAL",
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "code_sha256": sha256(__file__), "frozen_sha256": sha256(UNIT / "FROZEN_EVALUATION.json"),
            "synthetic_families": families, "synthetic_cases": cases,
            "real_saved_physical_inventory": physical_inventory,
            "real_saved_crossperson_summary": role_summaries, "real_saved_crossperson_cases": crossperson,
            "technical_comparison_timing_seconds": {"median": statistics.median(c["elapsed_seconds"] for c in cases),
                "max": max(c["elapsed_seconds"] for c in cases), "scope": "categorical comparator only; no video/model latency"},
            "unsupported_feedback_features": ["exact handshape beyond saved predicates", "fine palm/finger 3D direction",
                "actual contact vs near contact", "whole-word role identity", "linguistically mandatory order/count",
                "natural human drift tolerance", "complete dense learned-head sequence accuracy"],
            "new_model_end_to_end_feedback_validated": False,
            "real_educational_accuracy": None, "learned_model_gain": None,
            "whole_word_verdicts": 0, "new_training": 0, "new_model_inference": 0,
            "source_hashes_preserved": True,
            "limitations": ["Saved observations are same-Codex AI RGB drafts, not independent human ground truth.",
                "Synthetic tests change labels/timing, not real videos or actual learner performances.",
                "Strict unknown abstention removes complete verdict coverage; reduced candidate count alone is not improvement.",
                "Sparse anchor agreement does not certify unobserved boundaries or omitted events.",
                "Exact repeated-state order compares one recorded example, not a mandatory repetition count.",
                "Same-word different-person clips are descriptive normal-public-video comparisons; no adjudicated errors.",
                "Simple pointwise baseline is not production DTW and this does not replace production comparison.",
                "New seven-feature learned head has only an input adapter here; end-to-end feedback is unvalidated."]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("selftest", "evaluate"))
    args = parser.parse_args()
    if args.mode == "selftest":
        result = selftest()
        write_new(UNIT / "FEEDBACK_SELFTEST.json", result)
        print(json.dumps({"status": result["status"], "checks": result["count"]}))
    else:
        result = evaluate_saved()
        write_new(UNIT / "FEEDBACK_EVALUATION.json", result)
        print(json.dumps({"status": result["status"], "synthetic_families": result["synthetic_families"],
                          "real_crossperson": result["real_saved_crossperson_summary"]}))


if __name__ == "__main__":
    main()
