"""Saved-file/feature provenance only. No extraction, models, or trajectory solve."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
U, OPS = HERE.parent, HERE.parent.parent
OLD = OPS / "word100_hold_invariant_trajectory_20261101"
STAGE = OPS / "word100_stage_feedback_cycle_20261031"
CONTEXT = OPS / "word100_alignment_context_review_20261101"
TARGET = OPS / "word100_target_known_feedback_20261101"
MISSING = OPS / "word100_alignment_missing_split_20261101"
OBSERVER = OPS / "word100_guide_video_feedback_20261028/motion_observer.py"
pins = {}


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pin(p, expected=None):
    p = str(Path(p).resolve()); h = sha(p)
    assert expected is None or expected == h, p
    pins[p] = h
    return h


def save(name, value):
    p = HERE / name
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(dict(file=str(p), sha256=sha(p))))


previous = read(OLD / "review/RESULT_VERIFICATION.json")
assert previous["status"] == "PASS" and previous["verified_geometry_records"] == 2825
manifest, evaluation, tasks = (read(OLD / p) for p in ("RESULT_MANIFEST.json", "EVALUATION.json", "TASKS.json"))
source_inventory = read(OLD / "review/SOURCE_INVENTORY.json")
for p in (OLD / "FROZEN.json", OLD / "TASKS.json", OLD / "RESULT_MANIFEST.json", OLD / "EVALUATION.json",
          OLD / "NEXT_PLAN_KO.txt", OLD / "review/RESULT_VERIFICATION.json", OLD / "review/SOURCE_INVENTORY.json",
          OLD / "review/POSTEXECUTION_REVIEW.json", OLD / "common.py", OLD / "run.py", OLD / "trajectory.py",
          CONTEXT / "review/RESULT_VERIFICATION.json", CONTEXT / "review/SAFETY_RESULT_VERIFICATION.json"):
    pin(p, previous["pins"].get(str(p)))
rows_by_id = {r["id"]: r for r in evaluation["rows"]}
assert len(rows_by_id) == len(manifest["results"]) == 2825
records = []
for r in manifest["results"]:
    assert previous["pins"][r["path"]] == r["sha256"]
    pin(r["path"], r["sha256"])
    e = rows_by_id[r["id"]]
    assert all(r[k] == e[k] for k in ("kind", "query", "reference", "role", "component", "parent_id"))
    records.append(dict(**r, complete=e["complete"], input_scope_status="COMPLETE" if e["complete"] else "KEEP_UNKNOWN_NULL",
                        previous_ambiguity=e["optimal_path_ambiguous"],
                        relevant_edit_outside_parent=e.get("edited_query_outside_parent", 0)))
assert Counter((r["kind"], r["complete"]) for r in records) == {
    ("natural", True): 293, ("natural", False): 292, ("control", True): 1124, ("control", False): 1116}
assert len(manifest["missing_component_opportunities"]) == 10
for p, h in tasks["protected"].items():
    pin(p, h)
for row in tasks["rows"].values():
    pin(row["input"], row["input_sha256"])
for c in tasks["controls"]:
    pin(c["array"], tasks["pins"][c["array"]])
for r in manifest["new_control_baselines"]:
    pin(r["path"], r["sha256"])
for event in tasks["natural"]:
    pin(event["baseline"], tasks["pins"][event["baseline"]])
for c in tasks["controls"]:
    if c["baseline"]:
        pin(c["baseline"], tasks["pins"][c["baseline"]])
readouts = read(MISSING / "READOUT_ONLY_RESULT.json")
pin(MISSING / "READOUT_ONLY_RESULT.json")
pin(MISSING / "READOUT_ONLY_CONTRACT.json")
for r in readouts["records"]:
    pin(r["path"], r["sha256"])

input_report = dict(schema="wis_continuous_trajectory_input_audit_v1",
    status="PASS_SAVED_SCOPE_HASH_AUDIT_NO_NEW_TRAJECTORY_COMPUTATION",
    at_utc=datetime.now(timezone.utc).isoformat(), reviewer="Codex semantic_data_audit",
    inherited_verified_result=str(OLD / "review/RESULT_VERIFICATION.json"),
    source_scope=dict(public_words=4, public_clips=16, global_pairs=36, natural_parent_instances=117,
        natural_component_opportunities=585, natural_complete=293, natural_unknown=292,
        control_case_denominator=160, actual_control_cases=158, missing_control_cases=2,
        control_component_opportunities=2250, control_complete=1124, control_unknown=1116, missing_component_opportunities=10,
        actual_geometry_records=2825, complete_geometry_records=1417, unknown_geometry_records=1408,
        original_seed_alias_records=108, geometry_computed_once=True),
    original_two_axis_cells=previous["axis_denominators"],
    source_records=records, missing_component_records=manifest["missing_component_opportunities"],
    required_input_adapter=[
        "Use each old result query_plot/reference_plot values and masks exactly, with frame_id/timestamp_us from its comparison.query/reference. The prior independent verifier already checked those against named raw69 columns.",
        "Retain each old parent/query/reference range, baseline_full_path, old baseline statistics, prior discrete candidate, source-query/reference pointers and control edit ledger. Append the continuous result in a separate namespace.",
        "All2825 original IDs remain, including1408 unknown whole-parent records. The10 NO_SAVED_CONTROL records remain separate. Do not evaluate only1417 complete records and present their denominator as the whole task.",
        "Do not rerun old discrete candidates, original36 global paths,94 saved control baselines or64 already-completed new control baselines. Link their fixed files/hashes.",
        "Any masked coordinate keeps the whole-parent candidate null. Do not bridge observed islands, fill gaps, choose a favorable block, move parent boundaries or drop edited points outside a crop.",
        "Every anatomical feature identity, source row order, frame ID, PTS and original coordinate/mask is immutable. Continuous line-segment parameters are a separate comparison representation, not replacement source samples.",
        "Control original_indices is a generation/evaluation ledger only. Source subsampling can remove a bend; absence of an interior original sample does not prove the real motion followed its straight connector.",
        "Same-expression different-reference results stay separate. No reference/word/seed selection or post-result tolerance tuning."
    ],
    reporting_limits=[
        "These are the same historical public development records, not a new independent semantic cohort.",
        "A change from sampled-point to polyline distance is a geometric representation comparison, not a new recognition score.",
        "Natural complete293 versus unknown292 and control complete1124 versus unknown1116/missing10 must be reported separately.",
        "A small position-curve residual cannot declare handshape, palm rotation, wrist bending, order, or the whole sign correct."
    ],
    pins=pins.copy(), protected=tasks["protected"],
    reuse_method="Reused prior independent numerical audit and current hashes; did not reload/re-extract observations, decode videos, infer models, or rerun alignments.",
    actions=dict(model_inference=0, extraction=0, path_search=0, old_score_recalculation=0, raw_video_access=0,
                 private_or_mentor_access=0, old_files_modified=0),
    script_sha256=sha(__file__))
save("INPUT_AUDIT.json", input_report)

# Inspect the small existing target masks, not raw video, to distinguish sparse
# palm-front labels from actually trained/exposed orientation outputs.
learning = read(STAGE / "LEARNING_INPUTS.json")
pin(STAGE / "LEARNING_INPUTS.json")
rubric = read(STAGE / "RGB_RUBRIC.json")
pin(STAGE / "RGB_RUBRIC.json")
selected = {r["alias"] for r in source_inventory["clips"]}
label_rows, label_totals = [], Counter()
for row in learning["rows"]:
    if row["alias"] not in selected:
        continue
    pin(row["targets_file"], row["targets_sha256"])
    with np.load(row["targets_file"], allow_pickle=False) as z:
        y, mask = z["y"], z["target_mask"]
    assert y.shape == mask.shape == (row["frames"], len(learning["all_targets"]))
    for feature in ("R_palm_front", "L_palm_front"):
        j = learning["all_targets"].index(feature)
        known = mask[:, j]; values = y[known, j]
        counts = dict(known=int(known.sum()), front=int((values == 1).sum()),
                      back=int((values == 0).sum()), unknown=int((~known).sum()))
        label_rows.append(dict(alias=row["alias"], role=row["role"], target=feature,
                               target_file=row["targets_file"], column=j, **counts))
        for k, value in counts.items():
            label_totals[feature + "/" + k] += value
joint_pins = read(OPS / "word100_joint_stage_alignment_20261101/TASKS.json")["pins"]
probabilities = []
for seed in (42, 43, 44):
    p = STAGE / "runs" / f"pose_plus_signrep_{seed}" / "probabilities.npy"
    h = pin(p, joint_pins[str(p)])
    probabilities.append(dict(seed=seed, path=str(p), sha256=h,
        output_targets=learning["targets"], preserved_readout_targets=["R_index_only", "R_open_four", "L_open_four"],
        palm_front_output_present=False, wrist_bend_or_rotation_output_present=False))
code_paths = [OBSERVER, STAGE / "prepare_bundle.py",
              OPS / "word100_signrep_temporal_head_20261029/prepare_inputs.py",
              OPS / "word100_relational_sequence_compare_20261031/relational.py",
              OPS / "word100_joint_stage_alignment_20261101/joint_alignment.py",
              TARGET / "comparator.py", TARGET / "REFERENCE_PHASES.json",
              OPS / "word100_semantic_feature_cycle_20261101/FEATURE_CONTRACT.json"]
for p in code_paths:
    pin(p)
feature_names = source_inventory["feature_names_69"]
assert not any("wrist_angle" in name or "wrist_bend" in name or "normal" in name or "rotation" in name for name in feature_names)
assert all(name not in learning["targets"] for name in ("R_palm_front", "L_palm_front"))
assert {r.get("fine_finger_3d_or_exact_contact_validated") for c in read(TARGET / "REFERENCE_PHASES.json")["clips"].values()
        for r in c["regions"] if r["region_type"] == "observed_physical_core"} == {False}
feature_report = dict(schema="wis_rotation_and_wrist_evidence_preservation_v1",
    status="READ_ONLY_FEATURE_CAPABILITY_AND_PRESERVATION_AUDIT",
    at_utc=datetime.now(timezone.utc).isoformat(), reviewer="Codex semantic_data_audit",
    extraction_lineage=[str(STAGE / "prepare_bundle.py"),
                       str(OPS / "word100_signrep_temporal_head_20261029/prepare_inputs.py"), str(OBSERVER)],
    source_layout="Saved69 features derive from75 projected XY landmark points (Pose33+anatomical Hand21+Hand21), explicit observed&usable masks. No depth is used by this extractor.",
    present_information=[
        dict(feature="palm_axis_xy", columns={"L":[25,26],"R":[53,54]},
             definition="Unit screen-plane vector from hand landmark0(wrist) to landmark9(middle MCP). It is a longitudinal palm-axis proxy, not a palm-surface normal.",
             availability="Requires both source points observed/usable/finite and axis length>1e-8; own original masks retained.",
             limit="A fist can rotate around its longitudinal/forearm axis while this projected direction remains unchanged. Out-of-plane orientation is not uniquely identified."),
        dict(feature="palm_width", columns={"L":[27],"R":[55]},
             definition="Projected distance between index MCP5 and little MCP17 in contemporaneous shoulder-width units.",
             limit="May vary under projection/rotation, camera or detector noise. It is not a calibrated3D rotation angle or wrist-bend measure."),
        dict(feature="palm_center_and_index_tip_xy", columns={"L":[6,7,8,9],"R":[34,35,36,37]},
             definition="Index tip8; palm center mean of hand0,5,9,13,17, body-normalized XY.",
             limit="Preserves some hand-axis/shape motion around a stationary wrist, but does not certify palm-facing direction or3D flexion."),
        dict(feature="fourteen_finger_internal_angles_per_hand", columns={"L":list(range(11,25)),"R":list(range(39,53))},
             definition="Projected2D triplet internal angle divided bypi; thumb MCP/IP plus MCP/PIP/DIP of four nonthumb fingers.",
             limit="Rigid in-plane hand rotation can leave these angles unchanged. Out-of-plane changes mix real articulation and projection. These are not wrist joint angles."),
        dict(feature="pose_wrist_elbow_and_elbow_angle", columns={"L":[2,3,4,5,10],"R":[30,31,32,33,38]},
             definition="Separate body-detector wrist/elbow positions and shoulder-elbow-wrist internal angle.",
             limit="elbow_angle is an elbow angle. Combining forearm direction with palm_axis could form a new2D wrist-angle proxy, but that proxy is not an existing stored or validated wrist-bend classifier."),
        dict(feature="wrist_estimator_distance", columns={"L":[59],"R":[60]},
             definition="Distance between hand-detector and pose-detector estimates of the same wrist.",
             limit="Detector disagreement diagnostic, not an anatomical wrist-bending angle."),
        dict(feature="translation_velocity_and_direction", columns={"L":list(range(61,65)),"R":list(range(65,69))},
             definition="Consecutive native-frame wrist displacement/time, and normalized translation direction where speed is nonzero.",
             limit="No angular velocity or pronation/supination signal; zero wrist translation does not mean no meaningful hand movement.")
    ],
    absent_or_unvalidated=[
        "No saved palm-normal3D vector, wrist flexion/extension angle, radial/ulnar deviation angle, forearm pronation/supination angle, or explicit fist-rotation time series in these69 columns.",
        "No explicit trained rotation/wrist-bend output among the native10-target head or its preserved3-output readout.",
        "No independent educational rotation/bending correctness benchmark was run in this unit or the prior positional-trajectory unit.",
        "Reference physical cores explicitly retain fine_finger_3d_or_exact_contact_validated=false.",
        "A geometric2D proxy or existing sparse palm-front label must not be promoted to full3D rotation ground truth."
    ],
    sparse_existing_palm_surface_labels=dict(
        source=str(STAGE / "RGB_RUBRIC.json"), definition=rubric["definitions"]["palm_front"],
        provenance=rubric["truth_status"], total_by_target=dict(label_totals), rows=label_rows,
        selected_for_native_head_training=False, used_in_preserved_three_output_readout=False,
        motion_rotation_sequence_truth=False, wrist_bend_truth=False,
        note="Existing front/back camera-facing labels are sparse physical RGB drafts; edge-on/occluded/ambiguous is unknown. Their existence does not establish a rotation sequence classifier."),
    preserved_native_probabilities=probabilities,
    position_curve_boundary=dict(
        components=tasks["component_names"],
        uses_palm_axis=False, uses_palm_width=False, uses_finger_angles=False, uses_palm_center_or_index_tip=False,
        inference="Stationary wrists and unchanged wrist/forearm relations can coexist with finger changes, fist rotation or wrist bending that this5-component comparator does not assess.",
        required_status="NOT_ASSESSED_BY_POSITION_TRAJECTORY, never noise, normal, irrelevant, or absent",
        small_translation_is_permission_to_remove_other_features=False),
    preservation_contract=[
        "Freeze exact input NPZ bytes for all16 clips and technical input arrays; preserve every69-column x value, boolean mask, frame ID and PTS. No denoising, coordinate smoothing, low-motion gating or removal.",
        "Keep all28 finger-angle columns, four palm-axis coordinates, two palm widths, palm centers, fingertip positions, pose-arm measurements and estimator diagnostics even though only10 coordinate components enter this trajectory candidate.",
        "Preserve all108 prior readout files, including L/R_shape, L/R_axis, L/R_location/arm and learned3 groups with their original mean/max/null/partial-support fields. The new curve namespace cannot overwrite their values or statuses.",
        "Preserve all three native probability files and their target-name mapping; no inference, output averaging, target relabeling, or conversion of open_four0 into a fist/rotation label.",
        "A small continuous position distance must not suppress a measured palm-axis/finger-angle change, a short event, an unknown orientation interval, or an existing disagreement in another feature.",
        "Continuous interpolation only describes line segments between existing observed position samples. It does not invent observations through missing intervals or imply anything about unmeasured rotation between frames.",
        "Keep orientation and wrist-bend capability explicitly limited in user-facing interpretation: unchanged source evidence is verified preservation; successful real rotation/bending recognition remains unverified."
    ],
    later_independent_verification=[
        "Verify old2825 IDs, parent/query/ref ranges, masks, source pointers, edit ledgers and missing10 are unchanged.",
        "Hash-check native69 inputs, three probability files and108 prior readouts before/after the new run.",
        "For a synthetic stationary-wrist case with changed palm-axis or finger-angle values, the position comparator may remain equal; every changed auxiliary value must survive and no whole-sign/no-change verdict may appear.",
        "Test a masked orientation interval without filling it from neighboring position samples.",
        "Do not label rotation/bending recovery PASS without an explicitly scoped actual feature test and appropriate verified labels."
    ],
    user_summary_ko=[
        "손목이 거의 움직이지 않아도 주먹 회전이나 손목 꺾임은 중요한 변화일 수 있으므로, 기존 손가락 각도·손바닥 축·모양 출력과 미관측 표시를 그대로 보존합니다.",
        "지금 위치 궤적 비교만으로 그 변화를 모두 판별할 수는 없습니다. 저장된 손바닥 축은 화면2D 방향이며,3D 주먹 회전·손목 굽힘의 실제 판별 성능은 아직 검증되지 않았습니다."
    ],
    code_pins={str(p):pins[str(p)] for p in code_paths}, original_input_pins={row["input"]:pins[row["input"]] for row in tasks["rows"].values()},
    protected=tasks["protected"], all_feature_support_pins={p:h for p,h in pins.items() if p not in input_report["pins"]},
    actions=dict(new_features_computed=0, new_labels=0, new_model_inference=0, raw_video_access=0,
                 user_video_access=0, training=0, trajectory_execution=0, old_files_modified=0),
    script_sha256=sha(__file__))
save("FEATURE_PRESERVATION.json", feature_report)
print(json.dumps(dict(source_records=len(records), complete=1417, unknown=1408, missing=10,
                      sparse_palm_surface_label_totals=dict(label_totals), input_pins=len(input_report["pins"]),
                      total_pins=len(pins)),ensure_ascii=False))
