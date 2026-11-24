"""Read saved public metadata/arrays only. No model, feature extraction or path search."""
import collections
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
OPS = ROOT / "research/front_keyphase_v1/ops"
OUT = Path(__file__).resolve().parent
CASE = OPS / "word100_action_feedback_casebook_20261101"
REC = OPS / "word100_observation_lineage_repair_20261101"
REL = OPS / "word100_relational_sequence_compare_20261031"
KNOWN = OPS / "word100_known_time_benchmark_20261022"
checks = []
def check(condition, name):
    checks.append({"name": name, "pass": bool(condition)})
def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))
def pin(p):
    p = Path(p).resolve()
    return {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
def schema(z):
    return {k: {"shape": list(z[k].shape), "dtype": str(z[k].dtype)} for k in z.files}
def write(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

inputs = load(CASE / "INPUTS.json")
visual = load(CASE / "review/VISUAL_REVIEW.json")
recovery = load(REC / "RECOVERY_RESULT_MANIFEST.json")
prepared = load(REC / "RECOVERY_PREPARED.json")
natural = inputs["natural"]
rows = inputs["rows"]
pairs = sorted({(r["query"], r["reference"]) for r in natural})
check(len(natural) == 39 and len(pairs) == 12 and len(rows) == 16, "fixed_natural_scope")
check(set(r["query"] for r in natural) == {"P036", "P048", "P066", "P072"}, "four_fixed_development_queries")
input_inventory = []
for alias, row in sorted(rows.items()):
    item = {k: row[k] for k in ["clip_id", "word_id", "label", "person", "role", "box"]}
    item.update(alias=alias, input=pin(row["input"]), native_observation=pin(row["observation"]),
                record=pin(row["record"]), public_video_provenance={"path": row["source"], "sha256": row["source_sha256"], "read_this_audit": False})
    check(item["input"]["sha256"] == row["input_sha256"], f"{alias}:original_input_pin")
    check(item["record"]["sha256"] == row["record_sha256"], f"{alias}:record_pin")
    with np.load(row["input"]) as z:
        item["schema"] = schema(z)
        item["frame_count"] = len(z["x"])
        check(z["x"].shape == z["mask"].shape == (row["frames"], 69), f"{alias}:69_schema")
        check(len(np.unique(z["frame_id"])) == row["frames"], f"{alias}:unique_native_frame_ids")
    item["runtime_excluded_fields"] = ["targets_file", "targets_sha256", "known_cells", "intervals"]
    input_inventory.append(item)

overlay_inventory = []
prep_by_alias = {r["alias"]: r for r in prepared["clips"]}
for record in recovery["records"]:
    alias = record["alias"]
    item = {"alias": alias, "output": pin(record["output"]), "ledger": pin(record["ledger"]),
            "baseline_input": pin(rows[alias]["input"]), "selected_frame_count": record["frames"]}
    check(item["output"]["sha256"] == record["output_sha256"], f"{alias}:overlay_pin")
    check(item["ledger"]["sha256"] == record["ledger_sha256"], f"{alias}:overlay_ledger_pin")
    with np.load(record["output"]) as z, np.load(rows[alias]["input"]) as old:
        idx = z["source_index"]
        item.update(schema=schema(z), source_index=idx.tolist(), frame_id=z["frame_id"].tolist(),
                    timestamp_us=z["timestamp_us"].tolist(), newly_supported_cells=int(z["newly_supported"].sum()))
        check(idx.tolist() == prep_by_alias[alias]["indices"], f"{alias}:frozen_selected_indices")
        check(np.array_equal(old["frame_id"][idx], z["frame_id"]) and np.array_equal(old["timestamp_us"][idx], z["timestamp_us"]), f"{alias}:overlay_native_id_pts_join")
        check(np.array_equal(old["mask"][idx], z["baseline_mask"]), f"{alias}:baseline_mask_exact")
        check(np.array_equal(old["x"][idx], z["baseline_x"], equal_nan=True), f"{alias}:baseline_values_exact")
        check(np.all(z["mask"][z["baseline_mask"]]), f"{alias}:old_support_retained")
        check(np.array_equal(z["x"][z["baseline_mask"]], z["baseline_x"][z["baseline_mask"]], equal_nan=True), f"{alias}:old_valid_values_exact")
        check(np.array_equal(z["newly_supported"], z["mask"] & ~z["baseline_mask"]), f"{alias}:new_support_ledger")
    overlay_inventory.append(item)
check(sum(x["selected_frame_count"] for x in overlay_inventory) == 303 and len(overlay_inventory) == 8, "partial_overlay_303_in_eight_clips")
check(sum(x["newly_supported_cells"] for x in overlay_inventory) == 604, "604_new_supported_feature_cells")

pair_inventory = []
for q, r in pairs:
    path = REL / "pairs" / f"B_same_path_{q}_{r}.json"
    data = load(path)
    links = np.asarray(data["path"], dtype=int)
    nq, nr = rows[q]["frames"], rows[r]["frames"]
    check(links[0].tolist() == [0, 0] and links[-1].tolist() == [nq-1, nr-1], f"{q}_{r}:old_full_endpoints")
    check(len(np.unique(links[:, 0])) == nq and len(np.unique(links[:, 1])) == nr, f"{q}_{r}:old_both_axes_full_coverage")
    cpath = REL / "pairs" / f"C_relational_path_{q}_{r}.json"
    cdata = load(cpath)
    clinks = np.asarray(cdata["path"], dtype=int)
    check(clinks[0].tolist() == [0, 0] and clinks[-1].tolist() == [nq-1, nr-1], f"{q}_{r}:C_full_endpoints")
    check(len(np.unique(clinks[:, 0])) == nq and len(np.unique(clinks[:, 1])) == nr, f"{q}_{r}:C_both_axes_full_coverage")
    pair_inventory.append({"query": q, "reference": r, "baseline": pin(cpath), "baseline_method": "C_relational_path",
                           "old_global_path_context_only": pin(path), "old_global_path_json_pointer": "/path",
                           "path_json_pointer": "/path",
                           "path_axis_order": ["query_sequence_index", "reference_sequence_index"],
                           "query_frames": nq, "reference_frames": nr, "stored_path_length": len(clinks),
                           "old_global_path_length": len(links),
                           "parent_ids": [n["parent_id"] for n in natural if n["query"] == q and n["reference"] == r],
                           "new_overlay_not_used_to_generate_this_old_path": True})

ctrl_summary = load(REL / "CONTROL_SUMMARY.json")
controls = []
for row in ctrl_summary["rows"]:
    path = Path(row["file"])
    zjson = load(path)
    array_path = path.with_suffix(".npz")
    item = {"alias": row["alias"], "control": row["control"], "json": pin(path), "input_npz": pin(array_path),
            "baseline_path": {"path": str(path), "sha256": pin(path)["sha256"], "json_pointer": "/path"},
            "source_original_input": pin(rows[row["alias"]]["input"]),
            "query_frames": zjson["query_frames"], "reference_frames": zjson["reference_frames"],
            "metadata": zjson["metadata"], "metadata_json_pointer": "/metadata",
            "existing_baseline_path_is_prediction_not_ground_truth": True,
            "uses_recovered_overlay": False}
    check(item["input_npz"]["sha256"] == zjson["array_sha256"], f"{path.stem}:array_pin")
    check(item["source_original_input"]["sha256"] == zjson["source_input_sha256"], f"{path.stem}:source_pin")
    with np.load(array_path) as arr:
        item["schema"] = schema(arr)
        check(arr["x"].shape == arr["mask"].shape == (zjson["query_frames"], 69), f"{path.stem}:69_schema")
    original = zjson["metadata"].get("original_indices")
    if original is None:
        original = list(range(zjson["query_frames"]))
        mapping_provenance = {"kind": "unchanged_sequence_indices_by_saved_transform_code", "path": str(REL / "run_compare.py"), "function": "controls", "source_json_has_original_indices": False}
    else:
        mapping_provenance = {"kind": "explicit_saved_indices", "json_pointer": "/metadata/original_indices"}
    item["time_ground_truth"] = {"query_index_to_original_reference_index": original, "provenance": mapping_provenance,
                                "algorithm_input_allowed": False, "natural_crossperson_ground_truth": False}
    check(len(original) == zjson["query_frames"], f"{path.stem}:mapping_length")
    item["monotone_mapping"] = bool(np.all(np.diff(original) >= 0))
    item["limits"] = {
        "speed_125": "Nearest-sample acceleration deletes original samples; absence of a short sampled event is not algorithm deletion or educational error.",
        "swap_adjacent_core_blocks": "Nonmonotone source order cannot be reproduced by a single monotone alignment; use order-conflict/ambiguity reporting, not a zero source-time-error gate.",
        "right_hand_unobserved": "Artificial feature-mask loss, not a filmed occlusion or proof that the hand is unimportant.",
        "short_shape_change_5f": "Post-extraction angle edit; only already observed changed cells are evidence. Not a filmed sign error.",
        "left_short_shape_change_5f": "Post-extraction angle edit; retain zero-effective-edit cases explicitly.",
        "omit_short_core_5f": "Known source deletion, not established linguistically required action omission.",
    }.get(row["control"], "Known synthetic temporal mapping only; no new performer or educational label.")
    controls.append(item)
control_counts = dict(collections.Counter(c["control"] for c in controls))
check(len(controls) == 126, "126_saved_controls")
check(set(c["alias"] for c in controls) == set(rows), "saved_controls_exact_current16_sources")

known = load(KNOWN / "FROZEN.json")
known_inventory = []
for i, case in enumerate(known["cases"]):
    npz_path = KNOWN / "cases" / f'{case["kind"]}__{case["clip"]}.npz'
    with np.load(npz_path) as z:
        check(np.array_equal(z["source_index"], np.asarray(case["source_index"])), f"known{i}:mapping_matches_frozen")
        check(np.array_equal(z["deleted"], np.asarray(case["deleted"])), f"known{i}:deletion_matches_frozen")
        nlinks = len(z["path"])
    known_inventory.append({"clip": case["clip"], "condition": case["kind"], "saved_array": pin(npz_path),
                            "baseline_path_npz_key": "path", "source_index_npz_key": "source_index", "deleted_npz_key": "deleted",
                            "frozen_case_pointer": f"/cases/{i}", "query_frames": len(case["source_index"]),
                            "deleted_count": len(case["deleted"]), "path_length": nlinks})
known_clips = {x["clip_id"] for x in known["rows"]}
check(len(known_inventory) == 297 and len(known_clips) == 27, "known_time_297_and_27")
check(not (known_clips & {x["clip_id"] for x in rows.values()}), "known_time_disjoint_from_current16")

visual_parents = {(w["query"], w["reference"], w["parent_id"]): w for w in visual["windows"]}
parents = []
for i, n in enumerate(natural):
    w = visual_parents.get((n["query"], n["reference"], n["parent_id"]))
    parents.append({"id": n["id"], "query": n["query"], "reference": n["reference"], "parent_id": n["parent_id"],
                    "query_range_inclusive": n["query_range"], "reference_range_inclusive": n["reference_range"],
                    "physical_description": n["parent_physical_description"],
                    "source": {"path": str(CASE / "INPUTS.json"), "pointer": f"/natural/{i}"},
                    "existing_hand_record": n["hand_record"], "existing_legacy_link": n["legacy"],
                    "source_card_count": n["source_card_count"],
                    "existing_visual_window": w["window_id"] if w else None,
                    "unique_correct_frame_correspondence": None, "same_occurrence_ground_truth": None,
                    "educational_correctness": None, "algorithm_may_use_query_physical_description": False})
check(sum(p["existing_visual_window"] is not None for p in parents) == 10, "only10_reviewed_pairparents_not39")

visual_ledger = []
for i, w in enumerate(visual["windows"]):
    axes = {}
    for axis in ["query_axis", "reference_axis"]:
        a = w[axis]
        axes[axis] = {k: a[k] for k in ["alias", "clip_id", "reviewed_frame_ids", "reviewed_timestamp_us", "observations"]}
        axes[axis]["source_pointer"] = f"/windows/{i}/{axis}"
    visual_ledger.append({"window_id": w["window_id"], "query": w["query"], "reference": w["reference"], "parent_id": w["parent_id"],
                         "source_pointer": f"/windows/{i}", "axes": axes,
                         "saved_core_correspondence": w["source_evidence"]["saved_core_correspondence"],
                         "visible_difference_review": w["visible_difference_review"], "uncertainty_ko": w["uncertainty_ko"],
                         "unique_correct_correspondence": None, "educational_error_label": None, "runtime_input_allowed": False})
w06 = visual["windows"][5]
w07 = visual["windows"][6]
counterexample = next(x for x in w06["source_evidence"]["representative_existing_pairs"] if x["query_frame"]["frame_id"] == 68)
evidence = {"source": pin(CASE / "review/VISUAL_REVIEW.json"),
            "W06": {"kind": "saved_projected_hand_height_relation_conflict", "query": "P066", "reference": "P063",
                    "query_frame": 68, "reference_frame": 58, "query_relation": "R_above_L", "reference_relation": "L_above_R",
                    "source_pointer": "/windows/5", "frame_provenance": counterexample,
                    "other_contact_page_observed_conflict_pairs": [[69, 59], [69, 60]],
                    "ambiguous_contact_pair": [67, 57],
                    "unique_correct_reference_frame": None, "unique_correct_occurrence": None,
                    "label_scope": "projected visible relation only; neither learner error nor exact temporal GT",
                    "algorithm_input_allowed": False},
            "W07": {"kind": "saved_many_to_one_and_visible_transition_hold_mix",
                    "compressed_query_range": [71, 79], "reference_frame": 63,
                    "source_pointer": "/windows/6/source_evidence/saved_core_correspondence",
                    "saved_pairs": w07["source_evidence"]["saved_core_correspondence"],
                    "all_pairs_in_interval_wrong": None, "unique_correct_correspondence": None,
                    "repetition_linguistically_required": None, "algorithm_input_allowed": False}}
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
sources = [CASE / "INPUTS.json", CASE / "review/VISUAL_REVIEW.json", CASE / "review/WINDOW_SELECTION.json",
           REC / "NEXT_PLAN_KO.txt", REC / "review/RESIDUAL_PLAN_REVIEW.json", REC / "RECOVERY_PREPARED.json",
           REC / "RECOVERY_RESULT_MANIFEST.json", REC / "BRIDGE_MANIFEST.json", REC / "review/RECOVERY_SOURCE_VERIFICATION.json",
           REC / "review/RECOVERY_VISUAL_REVIEW.json", REL / "CONTROL_SUMMARY.json", REL / "run_compare.py", KNOWN / "FROZEN.json", KNOWN / "study.py",
           Path(__file__)]
ledger = {"schema": "phase_correspondence_evaluation_ledger.v1", "status": "READY_WITH_EXPLICIT_GROUND_TRUTH_LIMITS", "created_utc": now,
          "algorithm_input_separation": "This entire ledger is evaluator-only. Build runtime arrays solely from explicitly listed input NPZ x/mask and native IDs; never pass source_index/original_indices/copy_id, W06/W07 labels, physical descriptions, or query annotation ranges into candidate alignment.",
          "original_inputs": input_inventory, "partial_recovery_overlays": overlay_inventory,
          "overlay_join_contract": {"copy": "Copy full original x/mask; replace only selected source_index rows with overlay x/mask after asserting frame_id and timestamp_us match. No zero filling or extrapolation outside 303 rows.",
                                    "native_coordinates": "native_xy_px/native_observed/native_usable are pre-ROI; xy_px/observed/usable are fixed-ROI evidence. Preserve their distinct semantics.",
                                    "frame_ids": "Control query indices can be newly numbered. Use source_index only in evaluation, not as an alignment cue.",
                                    "learned_probabilities": "Existing learned handshape posteriors remain their original saved predictions. New recovered 69-feature support does not imply recomputed classifier output.",
                                    "fair_comparison": "Primary12 natural original69 uses archived C_relational versus one new constrained candidate on identical original input. Separate recovery12 natural uses merged69 for both newly computed unconstrained E0 and constrained E1. Old Bglobal is context only; do not pick the better panel after results."},
          "natural_baseline_paths": pair_inventory, "natural_parents": parents,
          "saved_current_controls": controls, "saved_control_counts": control_counts,
          "saved_controls_missing": {"swap_adjacent_core_blocks": sorted(set(rows) - {x["alias"] for x in controls if x["control"] == "swap_adjacent_core_blocks"})},
          "known_time_legacy": {"frozen": pin(KNOWN / "FROZEN.json"), "source_clip_count": 27, "case_count": 297,
                                "condition_counts": dict(collections.Counter(c["condition"] for c in known_inventory)),
                                "current16_overlap": [], "inventory": known_inventory,
                                "schema_limit": "Saved angle-bound/diagnostic arrays and paths, not current x[T,69]/mask inputs. Reuse mappings and temporal-control protocol; do not silently substitute these cases into current 69-feature evaluation.",
                                "pause_is_repeated_motion_ground_truth": False},
          "reused_visual_windows": visual_ledger, "observed_relation_counterevidence": evidence,
          "unavailable_labels": {"natural_crossperson_unique_correct_time": None, "natural_repeat_occurrence_match": None,
                                 "natural_crossing_exact_time": None, "linguistic_required_repetition": None,
                                 "learner_error_correctness": None},
          "source_pins": [pin(p) for p in sources]}
write("EVALUATION_LEDGER.json", ledger)

proposal = {
 "minimal_new_control_design": "Root confirmed: for each fixed16 clips, add hold15 at global middle, repeat2same and repeat3same. Both repetition axes have the same copy count; query only receives old nearest-sample speed_075 round-linspace temporal resampling. 48 new technical controls, alongside saved126. Synthetic source-frame, occurrence_id and absolute_unrolled_index are evaluator-only.",
 "denominators": {"fixed_source_clips": 16, "existing_identity_reused": 16, "proposed_new_2copy": 16, "proposed_new_3copy": 16, "proposed_new_hold15": 16, "proposed_total_with_identity": 64},
 "repetition_pairing_must_be_frozen": "Confirmed equal2/equal3 counts plus query time deformation tests occurrence preservation under known chronological order. No unequal-copy case is included. Whole unrolled index rather than modulo source frame distinguishes intended copies; exact-observation aliases remain a limitation.",
 "occurrence_leakage": "Keep copy_id and source_index in evaluator ledger only. Strip/reset array frame_id/time so copy-index resets cannot leak construction identity through runtime metadata.",
 "copy_join_boundary": "Whole-video concatenation creates an artificial final-to-first jump. Mark every seam and separately report interior versus seam links; do not treat this as a filmed fluent repetition.",
 "hold_position": "Confirmed globalmiddle index n//2, 15 additional duplicates, word-independent. Record no-op or missing-support holds instead of dropping cases. No W06/W07 annotation or residual controls selection.",
 "equivalent_repeats": "Identical copied feature traces alone may not identify a unique copy match. Report source-time error together with occurrence ambiguity/extra repetitions and full endpoint coverage, rather than awarding confidence from arbitrary tie-breaking.",
 "training": "No new independently validated supervised action/occurrence labels are created by this construction."
}
review = {"schema": "phase_correspondence_data_scope_review.v1", "created_utc": now,
          "status": "PASS_WITH_LIMITS" if all(c["pass"] for c in checks) else "FAIL",
          "check_count": len(checks), "failures": [c for c in checks if not c["pass"]],
          "scope": {"natural_pairs": 12, "natural_parents": 39, "public_clips": 16, "words": 4, "queries": ["P036", "P048", "P066", "P072"],
                    "existing_RGB_reviewed_pairparents": 10, "existing_RGB_unique_frames": 303, "overlay_clips": 8, "overlay_new_69_cells": 604,
                    "source_guide_linked_parents": sum(p["source_card_count"] > 0 for p in parents), "known_time_current_overlap": 0},
          "case_counts": dict(collections.Counter(n["query"] for n in natural)),
          "baseline_path_inventory": "EVALUATION_LEDGER.json#/natural_baseline_paths",
          "control_inventory": "EVALUATION_LEDGER.json#/saved_current_controls", "control_counts": control_counts,
          "repetition_hold_proposal_review": proposal,
          "preregistered_metric_recommendations": [
             "Report each fixed condition and source clip; do not pool unequal lengths/repetitions into one apparently independent sample count.",
             "For monotone known mappings report absolute source-time error per query index, mean and upper tail/max, and all matched alternatives; retain every query/reference index and endpoints.",
             "Separate missing observation, deleted source samples, unsupported dimensions and ambiguous occurrence. More missing/abstention is not lower mismatch success.",
             "For short shape changes retain known changed query-frame counts and all-reference projection mean/max and affected-cell support. No disappearing short change by denominator shrinkage.",
             "For nonmonotone swap/unequal-copy controls report unavoidable order conflict and ambiguity instead of imposing impossible zero-error monotone correspondence.",
             "W06 projected relation conflict and W07 compression are narrow diagnostic checks only. Unique cross-person correct frame and educational accuracy remain null.",
             "Natural4 cases are repeatedly exposed development examples; existing16 sources and old controls are not fresh held-out generalization evidence.",
             "Identical input route and feature/mask treatment must be frozen for both methods; keep shape, hand rotation/bend limits and raw differences outside alignment cost intact."
          ],
          "training_need": {"new_supervised_training_authorized_by_this_audit": False, "validated_new_action_or_occurrence_labels": 0,
                            "reusable_labels": "Existing limited known shape attributes remain their old contracts; physical descriptions and dictionary/source clauses do not become time-resolved required-action targets."},
          "actions": {"new_alignment": 0, "model_inference": 0, "new_RGB_decode_or_review": 0, "private_data_access": 0, "previous_results_modified": 0},
          "ledger": pin(OUT / "EVALUATION_LEDGER.json"), "checks": checks}
write("DATA_SCOPE_REVIEW.json", review)
print(json.dumps({k: review[k] for k in ["status", "check_count", "failures", "scope", "control_counts"]}, ensure_ascii=False))
print(json.dumps({"ledger": pin(OUT / "EVALUATION_LEDGER.json"), "review": pin(OUT / "DATA_SCOPE_REVIEW.json")}, ensure_ascii=False))
