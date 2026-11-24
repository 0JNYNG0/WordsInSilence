"""Read stored annotation contracts only. No model import, target mutation or inference."""
from pathlib import Path
import collections
import datetime
import hashlib
import json

A = Path(__file__).resolve().parent
SIDES = ("setup_relation", "terminal_relation")


def load(name):
    return json.loads((A / name).read_text(encoding="utf-8"))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def base(r):
    return {k: r[k] for k in ("view_id", "query_id", "clip_id", "role", "physical_query", "source_sha256", "observed_process_evidence", "review_source")}


def run():
    files = ["PROCESS_VIEW_TARGETS.json", "DEV_PROCESS_VIEW_TARGETS.json",
             "PROCESS_LABEL_PROTOCOL.json", "PHASE_LABEL_ADDENDUM.json",
             "PARTIAL_CONTROL_CONTRACT.json", "DEV_PARTIAL_CONTRACT.json",
             "compile_reviewed_views.py", "compile_dev_reviewed_views.py",
             "train_process_head.py", "evaluate_process_head.py"]
    pins = {str(A / f): sha(A / f) for f in files}
    missing_typed, explicit_relations, motion, conflicts = [], [], [], []
    summaries = {}
    for filename, split, expected in [(files[0], "fit", 217), (files[1], "development", 80)]:
        rows = load(filename)["rows"]
        assert len(rows) == expected
        c = collections.Counter()
        new_points = {}
        typed_new_keys = set()
        numeric_new_keys = set()
        for r in rows:
            assert r["role"] == split
            fm = r["ordered_frame_map"]
            sparse = {p["output_index"]: p for p in r["sparse_relation_targets"]}
            assert len(sparse) == len(r["sparse_relation_targets"])
            idx_by_frame = {f["source_native_frame"]: f["output_index"] for f in fm}
            for field, side in [("bound_setup_witnesses", SIDES[0]), ("bound_terminal_witnesses", SIDES[1])]:
                for p in r[field]:
                    idx = p["output_index"]
                    assert fm[idx]["source_native_frame"] == p["source_frame"]
                    assert fm[idx]["image_available"]
                    existing = sparse.get(idx, {}).get(side, "ABSENT")
                    c["typed_point_witness_records"] += 1
                    c[f"typed_{side}_existing_{existing}"] += 1
                    if existing == 1:
                        continue
                    entry = {**base(r), "relation": side, "source_frame": p["source_frame"],
                             "output_index": idx, "native_timestamp_us": fm[idx]["source_native_timestamp_us"],
                             "view_timestamp_us": fm[idx]["view_timestamp_us"],
                             "image_available": True, "original_witness": p["original_witness"],
                             "existing_sparse_target": existing,
                             "predicate_definition": r["predicate_definitions"][side],
                             "direct_visual_reason": r["evidence_reason"],
                             "evidence_kind": "typed_direct_positive_witness_not_numeric_label",
                             "interpretation": "This named witness slot asserts visible queried relation under the direct-review rubric; it does not assert temporal arrival, full process or the opposite relation being false."}
                    if existing == "ABSENT":
                        missing_typed.append(entry)
                        key = (r["view_id"], idx, side)
                        new_points[key] = entry
                        typed_new_keys.add((r["source_sha256"], r["physical_query"], p["source_frame"], side))
                    else:
                        conflicts.append(entry)
            for p in r["direct_review"].get("relations", []):
                idx = p.get("output_index", idx_by_frame[p["source_frame"]])
                assert fm[idx]["source_native_frame"] == p["source_frame"]
                for side in SIDES:
                    if side not in p:
                        continue
                    new = p[side]
                    existing = sparse.get(idx, {}).get(side, "ABSENT")
                    c["explicit_numeric_relation_records"] += 1
                    c[f"explicit_relation_{new}_existing_{existing}"] += 1
                    if new == existing:
                        continue
                    entry = {**base(r), "relation": side, "source_frame": p["source_frame"],
                             "output_index": idx, "explicit_numeric_label": new,
                             "existing_sparse_target": existing, "source_annotation": p,
                             "evidence_kind": "direct_explicit_numeric_relation"}
                    explicit_relations.append(entry)
                    if existing == "ABSENT" and new is not None:
                        assert fm[idx]["image_available"]
                        key = (r["view_id"], idx, side)
                        if key in new_points:
                            assert new == 1  # Typed positive must agree with the direct numeric annotation.
                        new_points[key] = entry
                        numeric_new_keys.add((r["source_sha256"], r["physical_query"], p["source_frame"], side))
                    elif existing != "ABSENT" and new is not None and existing != new:
                        conflicts.append(entry)
            consumed_intervals = {tuple(p["output_indices"]): p for p in r["fully_retained_motion_interval_targets"]}
            for p in r["bound_direct_motion_witnesses"]:
                ix = p["output_indices"]
                orig = p["original_witness"]
                explicit = "label" in orig
                assert explicit == p["label_was_explicit"]
                known = explicit and p["label"] in (0, 1)
                old = consumed_intervals.get(tuple(ix))
                old_label = old["label"] if old else "ABSENT"
                status = ("unlabelled_witness_not_numeric_gold" if not explicit else
                          "explicit_unknown_not_training_gold" if p["label"] is None else
                          "known_already_consumed_exact_interval" if old_label == p["label"] else
                          "explicit_known_unused_exact_interval" if old_label == "ABSENT" else
                          "conflict")
                c[f"motion_{status}"] += 1
                c[f"motion_direct_label_{p['label']}_explicit_{explicit}"] += 1
                entry = {**base(r), "source_frames": p["source_frames"], "output_indices": ix,
                         "label": p["label"], "label_was_explicit": explicit,
                         "existing_exact_interval_label": old_label, "status": status,
                         "source_annotation": orig,
                         "contiguous_available": (ix == list(range(ix[0], ix[-1] + 1))
                           and p["source_frames"] == list(range(p["source_frames"][0], p["source_frames"][-1] + 1))
                           and all(fm[i]["image_available"] for i in ix)),
                         "not_dense_frame_targets": True}
                motion.append(entry)
                if status == "conflict":
                    conflicts.append(entry)
                if known:
                    assert entry["contiguous_available"]
        missing_here = [v for v in missing_typed if v["role"] == split]
        unconsumed_motion = [v for v in motion if v["role"] == split and v["status"] == "explicit_known_unused_exact_interval"]
        c["typed_missing_view_records"] = len(missing_here)
        c["typed_missing_unique_source_query_frame_predicate"] = len(typed_new_keys)
        c["numeric_missing_unique_source_query_frame_predicate"] = len(numeric_new_keys)
        c["union_missing_point_view_dimension_records"] = len(new_points)
        c["union_missing_unique_source_query_frame_predicate"] = len(typed_new_keys | numeric_new_keys)
        c["missing_direct_known_motion_unique_source_query_interval"] = len({(v["source_sha256"], v["physical_query"], tuple(v["source_frames"]), v["label"]) for v in unconsumed_motion})
        c["typed_missing_with_positive_process_label"] = sum(v["observed_process_evidence"] == 1 for v in missing_here)
        summaries[split] = {"views": len(rows), "source_clips": len({r["clip_id"] for r in rows}),
                            "source_queries": len({r["query_id"] for r in rows}), "counts": dict(c)}
    # No target, review or model is edited by this script.
    assert all(sha(p) == digest for p, digest in pins.items())
    report = {
        "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "SUPERVISION_CONSUMPTION_OMISSION_CONFIRMED_NO_TARGET_MUTATION",
        "reviewer": "Codex predicate_contract_review; annotation/consumer audit, not independent human expert gold",
        "pins": pins, "summaries": summaries,
        "critical_case": {
            "query_id": "NIA_SL_WORD1567_REAL06_F_directed_50_61",
            "views": ["suffix", "middle_splice"], "frame": 73,
            "direct_fact": "Separate surviving fall reaches visibly right-below-left at native f73, explicitly stated in the reviewed reason with an enlarged native image witness.",
            "consumption": "No sparse relation row at f73; therefore terminal point has no loss and no sparse point evaluation target. This is absent supervision, not an existing labelled-null overwritten by zero.",
            "scope": "Two dependent views of one source/query/frame; genuine physical annotation omission, not two independent people and not proof of all performance failure.",
            "motion_67_73": "The associated trajectory witness has no explicit numeric interval label; do not manufacture one from the positive process or endpoint relation."},
        "contract_interpretation": [
            "Named direct setup/terminal witness lists have a positive visual-relation meaning only under the explicit phase/partial-review rubric and row predicate definition. Generic time roles, source prose, reference correctness, start/end positions and motion endpoints are not point gold.",
            "A relation visible before the chosen occurrence or after an opposite movement can remain present. It does not imply arrival or successful queried movement. Do not set the opposite relation to zero by complement.",
            "Numeric relations 0/1/null take precedence as explicit evidence; conflicts with a typed witness must fail and be adjudicated rather than guessed.",
            "All missing typed points found here are visibly available and have no conflicting existing sparse known target. Their positive candidate status is from reviewed physical assertion, never from view class.",
            "Shoulder/neck variants remain governed by the original broad predicate and uncertainty notes. Do not broaden the predicate or force an explicitly ambiguous native relation into one.",
            "Direct numeric motion intervals can be compared only at the same reviewed frame sequence. Subinterval containment is not exact equivalence. Unlabelled or null witnesses remain nonnumeric provenance."
        ],
        "causal_code_findings": [
            "Both target compilers preserve new direct witnesses as bound_* provenance but build sparse_relation_targets only from original native query relations.",
            "Both target compilers similarly keep direct motion witnesses but train/evaluate only fully_retained_motion_interval_targets inherited from native annotations.",
            "train_process_head.py and evaluate_process_head.py consume sparse_relation_targets and fully_retained_motion_interval_targets; they never consume bound_* merely because it was saved.",
            "Earlier independent verification established preservation of declared native labels and saved metric arithmetic; it did not establish consumption completeness for every new direct-reviewed fact. This audit identifies that missing contract check."
        ],
        "missing_typed_positive_point_witnesses": missing_typed,
        "different_or_unused_direct_numeric_relations": explicit_relations,
        "direct_motion_consumption": motion,
        "conflicts": conflicts,
        "recommended_correction_plan": [
            "Create versioned v2 target artifacts and a new frozen correction contract; retain v1 targets, one completed training run, predictions and gates unchanged.",
            "Compile direct numeric relations first. Add only named direct positive witnesses whose row-specific physical predicate and direct visual assertion agree. Reject conflict or ambiguity; preserve null and do not infer the other dimension, missing frames or dense labels.",
            "Deduplicate by view/output/predicate and retain all source annotations. Across views retain source/query/native-frame identity for loss multiplicity; repeated edits are not independent examples.",
            "Merge explicitly numeric direct interval annotations at their exact contiguous, available scope. Do not add the six FIT/three DEV unlabelled trajectory witnesses as motion label1. Preserve seam/blank unknowns and separate view-process label0 from residual motion1.",
            "Independently compare the compiler's consumed-label inventory to every eligible direct assertion, with per-label provenance and an explicit exclusion reason for anything not consumed.",
            "Keep FIT and DEV target compilation strictly separated. DEV corrections fix annotation-consumption fidelity only; do not select thresholds, windows, epochs or new labels from failed predictions.",
            "Any future training is one separately authorized fixed correction comparison with identical candidate architecture and thresholds unless an independent execution-contract repair is separately specified. Completion probability improvement is not guaranteed; automatic interval meaning and transfer remain separate causes."
        ],
        "new_images_reviewed": 0, "new_model_inference": 0, "new_training": 0,
        "targets_modified": False, "existing_results_modified": False,
        "performance_gain_established": False, "whole_goal_complete": False,
        "independent_human_expert": False,
    }
    out = A / "PROCESS_WITNESS_CONSUMPTION_AUDIT.json"
    assert not out.exists(), "Preserve prior audit rather than overwrite"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(out), "sha256": sha(out), "summaries": summaries, "conflicts": len(conflicts)}, ensure_ascii=False))


if __name__ == "__main__":
    run()
