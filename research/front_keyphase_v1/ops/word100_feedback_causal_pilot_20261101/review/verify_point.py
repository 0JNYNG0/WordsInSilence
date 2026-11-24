"""Independent saved-point checker; no producer imports or new inference.

Reads the frozen30 opportunities, native probabilities, exact-centre arrays,
input masks and original labels. Writes only VERIFY_POINT.json.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import ast
import hashlib
import json
import numpy as np

UNIT = Path(__file__).resolve().parent.parent
OPS = UNIT.parent
CHECKS = 0
FAILURES = []
PINS = {}


def check(ok, reason):
    global CHECKS
    CHECKS += 1
    if not bool(ok):
        FAILURES.append(str(reason))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path, expected=None):
    path = Path(path).resolve()
    h = digest(path)
    if expected is not None:
        check(h == expected, f"source hash {path}")
    PINS[str(path)] = h
    return path


def read(path):
    return json.loads(pin(path).read_text(encoding="utf-8-sig"))


def equal(actual, expected, where):
    """Exact JSON scalar values, including missing/null and boolean distinctions."""
    if isinstance(expected, dict):
        check(isinstance(actual, dict), where + " mapping")
        if not isinstance(actual, dict):
            return
        check(set(actual) == set(expected), where + " keys")
        for key in expected.keys() & actual.keys():
            equal(actual[key], expected[key], where + "/" + str(key))
    elif isinstance(expected, list):
        check(isinstance(actual, list), where + " list")
        if not isinstance(actual, list):
            return
        check(len(actual) == len(expected), where + " length")
        for i, (a, b) in enumerate(zip(actual, expected)):
            equal(a, b, where + "/" + str(i))
    else:
        check(actual == expected and (not isinstance(expected, bool) or isinstance(actual, bool)),
              f"{where}: {actual!r} != {expected!r}")


# Explicit incompatible binary pairs; no implementation imported from consumer.
EXCLUSIONS = {
    "R_shape": [("R_index_only", "R_open_four"),
                ("R_index_only", "R_closed_four"),
                ("R_open_four", "R_closed_four")],
    "L_shape": [("L_open_four", "L_closed_four")],
    "height": [("R_higher", "L_higher")],
}


def contradictions(values):
    found = {}
    positive = {name for name, value in values.items() if value == 1}
    for group, pairs in EXCLUSIONS.items():
        hits = [list(pair) for pair in pairs if set(pair) <= positive]
        if hits:
            found[group] = hits
    return found


def compare_values(q, r, target):
    a, b = q[target], r[target]
    qc, rc = contradictions(q), contradictions(r)
    group = "height" if target.endswith("higher") else target[0] + "_shape"
    clash = group in qc or group in rc
    status_table = {
        (0, 0): "SAME_NEGATIVE_ATTRIBUTE_ONLY",
        (1, 1): "SAME_POSITIVE_ATTRIBUTE",
        (0, 1): "ATTRIBUTE_DIFFERENCE_CANDIDATE",
        (1, 0): "ATTRIBUTE_DIFFERENCE_CANDIDATE",
    }
    known = a is not None and b is not None
    return {
        "target": target, "query": a, "reference": b,
        "status": status_table[(a, b)] if known else "INSUFFICIENT_OBSERVATION",
        "raw_attribute_difference": (a != b) if known else None,
        "query_conflicts": qc, "reference_conflicts": rc,
        "inconsistent_relevant_group": clash,
        "supports_single_attribute_claim": known and not clash,
        "whole_handshape_equal": None, "same_action_stage": None,
        "educational_correction": None,
    }


def main():
    frozen = read(UNIT / "FROZEN_POINT_DIAGNOSIS.json")
    result = read(UNIT / "POINT_DIAGNOSIS.json")
    for path, h in frozen["source_pins"].items():
        pin(path, h)
    for path, h in result["source_pins"].items():
        pin(path, h)
    for path, h in result["code_pins"].items():
        pin(path, h)
        ast.parse(Path(path).read_text(encoding="utf-8-sig"))
        check(True, "producer parses: " + path)
    check(frozen["denominator"] == 30, "frozen denominator30")
    check(frozen["seeds"] == [42, 43, 44], "all seeds")
    check(frozen["threshold"] == 0.5, "fixed class threshold")
    opportunities = frozen["opportunities"]
    check(len(opportunities) == 30, "all30 source opportunities")
    for field in ["new_model_inferences", "new_training", "new_dtw_paths"]:
        check(result[field] == 0, field)
    check(result["elapsed_seconds"] >= 0, "nonnegative replay time")

    case = read(OPS / "word100_action_feedback_casebook_20261101/INPUTS.json")
    data = read(OPS / "word100_semantic_feature_cycle_20261101/DATASET.json")
    old = read(OPS / "word100_stage_feedback_cycle_20261031/LEARNING_INPUTS.json")
    names = data["targets"]
    check(names == ["R_index_only", "R_open_four", "L_open_four", "R_closed_four",
                    "L_closed_four", "R_higher", "L_higher"], "target order")
    native_row = {r["clip_id"]: i for i, r in enumerate(old["rows"])}
    centers = {}
    for i, row in enumerate(data["rows"]):
        key = (row["clip_id"], row["center_frame"])
        check(key not in centers, "unique exact centre " + str(key))
        centers[key] = i
    arrays, label_arrays, old_rows = {}, {}, {}
    aliases = sorted({op[axis] for op in opportunities for axis in ["query", "reference"]})
    for alias in aliases:
        row = case["rows"][alias]
        with np.load(pin(row["input"], row["input_sha256"]), allow_pickle=False) as z:
            arrays[alias] = {key: z[key].copy() for key in ["x", "mask", "frame_id", "timestamp_us"]}
        x, mask = arrays[alias]["x"], arrays[alias]["mask"]
        check(x.shape == mask.shape and x.shape[1] == 69, alias + "69 shape")
        check(mask.dtype == np.bool_, alias + " boolean masks")
        check(np.isfinite(x[mask]).all(), alias + "finite observed values")
        original = old["rows"][native_row[row["clip_id"]]]
        old_rows[alias] = original
        check(original["alias"] == alias, alias + "alias/clip join")
        check(original["input_sha256"] == row["input_sha256"], alias + "same native input")
        with np.load(pin(original["targets_file"], original["targets_sha256"]), allow_pickle=False) as z:
            label_arrays[alias] = {key: z[key].copy() for key in z.files}

    def direct_label(alias, frame, target):
        """Verify labels independently from both actual saved label sources."""
        row = case["rows"][alias]
        source_values = []
        if target in old["all_targets"]:
            j = old["all_targets"].index(target)
            labels = label_arrays[alias]
            if labels["target_mask"][frame, j]:
                source_values.append(int(labels["y"][frame, j]))
        ci = centers.get((row["clip_id"], frame))
        if ci is not None:
            j = names.index(target)
            if data["rows"][ci]["target_mask"][j]:
                source_values.append(int(data["rows"][ci]["y"][j]))
        check(bool(source_values), f"direct label available {alias}:{frame}:{target}")
        check(len(set(source_values)) <= 1, f"label agreement {alias}:{frame}:{target}")
        return source_values[0] if source_values else None

    node_keys = set()
    for op in opportunities:
        check(direct_label(op["reference"], op["reference_frame"], op["target"]) == op["reference_label"], "reference label")
        equal([direct_label(op["query"], f, op["target"]) for f in op["current_query_indices"]],
              op["current_query_labels"], "query labels")
        check(op["unique_correct_query_time"] is None and not op["eligible_as_confirmed_stage_oracle"], "no stage truth")
        for a in op["same_family_known_query_anchors"]:
            check(direct_label(op["query"], a["query_frame"], op["target"]) == a["label"], "anchor label")
        for seed in frozen["seeds"]:
            node_keys.add((seed, op["reference"], op["reference_frame"]))
            for f in op["current_query_indices"] + [a["query_frame"] for a in op["same_family_known_query_anchors"]]:
                node_keys.add((seed, op["query"], f))
    expected_node_ids = {f"{s}:{a}:{f}" for s, a, f in node_keys}
    equal(sorted(result["nodes"]), sorted(expected_node_ids), "exact node set")
    expected_nodes, prediction_arrays = {}, {}
    exact_centers = 0
    for seed in frozen["seeds"]:
        native_dir = OPS / f"word100_stage_feedback_cycle_20261031/runs/pose_plus_signrep_{seed}"
        center_dir = OPS / f"word100_semantic_feature_cycle_20261101/runs/protected_shape_split_features_{seed}"
        native_meta = read(native_dir / "RUN.json")
        center_meta = read(center_dir / "COMPLETE.json")
        native = np.load(pin(native_dir / "probabilities.npy", native_meta["prediction_sha256"]), allow_pickle=False)
        center = np.load(pin(center_dir / "probabilities.npy", center_meta["probabilities_sha256"]), allow_pickle=False)
        prediction_arrays[seed] = (native, center)
        check(center.shape == (len(data["rows"]), 7), "centre probability shape")
        for _, alias, frame in sorted(x for x in node_keys if x[0] == seed):
            row, a = case["rows"][alias], arrays[alias]
            frame_rows = np.flatnonzero(a["frame_id"] == frame)
            check(len(frame_rows) == 1, "unique native frame")
            i = int(frame_rows[0])
            check(i == frame, "native ordinal equals frameID")
            oi, ci = native_row[row["clip_id"]], centers.get((row["clip_id"], frame))
            if ci is not None:
                check(a["timestamp_us"][i] == data["rows"][ci]["center_timestamp_us"], "exact centre PTS")
            probabilities, available, values, routes = {}, {}, {}, {}
            for j, target in enumerate(names):
                if target in names[:3]:
                    raw = float(native[oi, i, old["targets"].index(target)])
                    route = "protected_original_native"
                    if ci is not None:
                        check(raw == float(center[ci, j]), "protected raw probability exact")
                        exact_centers += 1
                elif ci is None:
                    raw, route = None, "no_center_prediction"
                else:
                    raw, route = float(center[ci, j]), "protected_split_exact_center"
                offset = {"L": 0, "R": 28}[target[0]]
                if target.endswith("higher"):
                    geometric = all(bool(a["mask"][i, c]) for c in [8, 9, 36, 37])
                else:
                    geometric = all(bool(a["mask"][i, c]) for c in [offset, offset + 1]) and sum(bool(a["mask"][i, c]) for c in range(offset + 11, offset + 25)) >= 10
                known = bool(geometric and raw is not None and np.isfinite(raw))
                if raw is not None:
                    check(np.isfinite(raw) and 0 <= raw <= 1, "probability bounds")
                probabilities[target], routes[target] = raw, route
                available[target] = known
                values[target] = int(raw >= frozen["threshold"]) if known else None
            palm_known = all(bool(a["mask"][i, c]) for c in [8, 9, 36, 37])
            node = dict(alias=alias, clip_id=row["clip_id"], frame_id=frame,
                        timestamp_us=int(a["timestamp_us"][i]), seed=seed,
                        probabilities=probabilities, support=available, values=values,
                        prediction_source=routes, contradictions=contradictions(values),
                        center_dataset_row=ci,
                        palm_left_y_minus_right_y=float(a["x"][i, 9] - a["x"][i, 37]) if palm_known else None,
                        palm_delta_definition="positive=right palm image-above left; not clear separation or equal-height judgement")
            key = f"{seed}:{alias}:{frame}"
            expected_nodes[key] = node
            equal(result["nodes"][key], node, "node/" + key)
    check(exact_centers == result["protected_center_probability_exact_checks"], "protected check denominator")

    expected_records, quality_rows = [], []
    for seed in frozen["seeds"]:
        for k, op in enumerate(opportunities):
            target = op["target"]
            rid = f"{seed}:{op['reference']}:{op['reference_frame']}"
            qids = [f"{seed}:{op['query']}:{f}" for f in op["current_query_indices"]]
            ref, qs = expected_nodes[rid], [expected_nodes[key] for key in qids]
            comps = [compare_values(q["values"], ref["values"], target) for q in qs]
            usable = all(c["raw_attribute_difference"] is not None for c in comps)
            difference = any(c["raw_attribute_difference"] is True for c in comps)
            truth = any(v != op["reference_label"] for v in op["current_query_labels"])
            raw_diff = any((q["probabilities"][target] >= .5) != (ref["probabilities"][target] >= .5)
                           for q in qs if q["probabilities"][target] is not None and ref["probabilities"][target] is not None)
            anchors = [dict(frame_id=a["query_frame"], comparison=compare_values(
                expected_nodes[f"{seed}:{op['query']}:{a['query_frame']}"]["values"], ref["values"], target))
                       for a in op["same_family_known_query_anchors"]]
            item = dict(opportunity=k, seed=seed, source=op, reference_node=rid, query_nodes=qids,
                        comparisons=comps, truth_attribute_difference=truth,
                        raw_ungated_prediction_difference=raw_diff, usable=usable,
                        observed_attribute_difference=difference if usable else None,
                        false_attribute_difference=bool(usable and difference and not truth),
                        known_feature_diagnostic_only=dict(values=op["current_query_labels"], reference=op["reference_label"], feature_difference=truth, automatic_performance=False),
                        candidate_anchors=anchors, stage_oracle_available=False,
                        whole_handshape_equal=None, educational_correctness=None)
            expected_records.append(item)
            for qi, (qid, q, label) in enumerate(zip(qids, qs, op["current_query_labels"])):
                qp, rp = q["probabilities"][target], ref["probabilities"][target]
                qpred = None if qp is None else int(qp >= .5)
                rpred = None if rp is None else int(rp >= .5)
                known_predictions = qpred is not None and rpred is not None
                quality_rows.append(dict(seed=seed, opportunity=k, target=target,
                    query_node=qid, reference_node=rid, query_direct_label=label,
                    reference_direct_label=op["reference_label"], query_raw_prediction=qpred,
                    reference_raw_prediction=rpred, query_supported=q["support"][target],
                    reference_supported=ref["support"][target], usable=item["usable"],
                    both_wrong=bool(known_predictions and qpred != label and rpred != op["reference_label"]),
                    either_wrong=bool(known_predictions and (qpred != label or rpred != op["reference_label"])),
                    source_record_pointer=f"/records/{len(expected_records)-1}/comparisons/{qi}"))
    equal(result["records"], expected_records, "records")
    summaries = []
    for seed in frozen["seeds"]:
        rows = [x for x in expected_records if x["seed"] == seed]
        cc = Counter(c["status"] for x in rows for c in x["comparisons"])
        summaries.append(dict(seed=seed, opportunities=len(rows), usable=sum(x["usable"] for x in rows),
            unknown=sum(not x["usable"] for x in rows), false_attribute_difference=sum(x["false_attribute_difference"] for x in rows),
            statuses=dict(cc), conflicting_opportunities=sum(any(c["inconsistent_relevant_group"] for c in x["comparisons"]) for x in rows),
            same_negative_is_not_full_handshape=cc["SAME_NEGATIVE_ATTRIBUTE_ONLY"]))
    equal(result["summary"], summaries, "summary")
    quality_summary = []
    for seed in frozen["seeds"]:
        rows = [x for x in quality_rows if x["seed"] == seed]
        quality_summary.append(dict(seed=seed, pair_feature_links=len(rows),
            raw_either_wrong=sum(x["either_wrong"] for x in rows), raw_both_wrong=sum(x["both_wrong"] for x in rows),
            supported_either_wrong=sum(x["either_wrong"] and x["usable"] for x in rows),
            supported_both_wrong=sum(x["both_wrong"] and x["usable"] for x in rows)))
    for path, h in list(PINS.items()):
        check(digest(path) == h, "read-only source preserved " + path)
    PINS[str(Path(__file__).resolve())] = digest(__file__)
    output = dict(schema="causal_pilot_point_independent_verification.v1",
        status="PASS" if not FAILURES else "FAIL", created_utc=datetime.now(timezone.utc).isoformat(),
        checks=CHECKS, failures=FAILURES, frozen_opportunities=30,
        seed_opportunities=len(expected_records), unique_saved_nodes=len(expected_nodes),
        exact_protected_center_checks=exact_centers, summary=summaries,
        raw_predictions_vs_direct_labels=quality_summary,
        both_wrong_or_other_error_details=[x for x in quality_rows if x["either_wrong"]],
        unknown_opportunities=[dict(seed=x["seed"], opportunity=x["opportunity"], target=x["source"]["target"], query_nodes=x["query_nodes"], reference_node=x["reference_node"]) for x in expected_records if not x["usable"]],
        source_pins=PINS, actions=dict(producer_import=0, inference=0, training=0, path_search=0, original_edits=0, RGB_access=0),
        limits=["Raw predictions are compared with existing sparse physical-feature labels, not independent educational truth.",
                "All30 source opportunities have matching direct feature labels; no stage-oracle or actual learner-error detection test is available.",
                "No contradiction occurs in the saved30; conflict annotations have no measured benefit here.",
                "Same-negative clarification is a new consumer contract, not evidence that a past consumer made a whole-handshape bug.",
                "This verifier does not rerun or evaluate source models, alignment or their objective optimality."])
    out = UNIT / "review/VERIFY_POINT.json"
    with out.open("x", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k:output[k] for k in ["status", "checks", "failures", "raw_predictions_vs_direct_labels"]}, ensure_ascii=False))
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    raise SystemExit(main())
