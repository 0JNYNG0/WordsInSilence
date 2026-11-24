"""Prepare deterministic proxy-dropout descriptors; no inference or training.

Only materialize_model_input's returned allowlisted arrays may reach a future
model. It never exposes native/raw/redetected values or evaluator-only targets.
No altered arrays are saved. RGB and real timestamps remain unchanged.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

UNIT = Path(__file__).resolve().parent
HOLES = (1, 3, 7)
MIN_RUN = 11
BASE_KEYS = (
    "baseline_roi_xy_px", "baseline_roi_observed", "baseline_roi_usable",
    "baseline_roi_conf", "baseline_roi_hand_score",
)
SHARED_KEYS = ("frame_id", "timestamp_us", "roi_box", "width", "height")
ALLOWLIST = BASE_KEYS + SHARED_KEYS
SIDE_NAMES = ("left", "right")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def runs(flags: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[False, flags, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(),
                    np.flatnonzero(edges == -1).tolist()))


def load_baseline(row: dict) -> dict[str, np.ndarray]:
    path = Path(row["arrays"])
    assert sha(path) == row["arrays_sha256"], (row["clip_id"], "array hash")
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in ALLOWLIST}


def materialize_model_input(row: dict, descriptor: dict) -> dict[str, np.ndarray]:
    """Return safe tensors only; never return the NPZ loader or source object.

    row resolves an opaque bundle_input_ref outside the model. No clip/gloss,
    label, target, source filename, assignment index, native/raw/redetected
    coordinate, point-origin map or core-time interval is a returned tensor.
    """
    result = load_baseline(row)
    transform = descriptor["model_input"]["coordinate_transform"]
    if transform["kind"] == "none":
        return result
    assert transform["kind"] == "single_hand_missing"
    side = transform["side_index"]
    assert side in (0, 1)
    start, end = transform["frame_range_half_open"]
    joints = slice(33 + 21 * side, 54 + 21 * side)
    result["baseline_roi_xy_px"][start:end, joints] = 0
    for key in ("baseline_roi_observed", "baseline_roi_usable"):
        result[key][start:end, joints] = False
    result["baseline_roi_conf"][start:end, joints] = 0
    result["baseline_roi_hand_score"][start:end, side] = 0
    return result


def check_materialization(original: dict, changed: dict, transform: dict) -> None:
    assert set(changed) == set(ALLOWLIST)
    assert not any(key.startswith(("native_", "raw_", "redetected_")) for key in changed)
    for key in SHARED_KEYS:
        assert np.array_equal(original[key], changed[key], equal_nan=True), key
    if transform["kind"] == "none":
        for key in BASE_KEYS:
            assert np.array_equal(original[key], changed[key], equal_nan=True), key
        return
    side = transform["side_index"]
    start, end = transform["frame_range_half_open"]
    target = np.zeros(original["baseline_roi_usable"].shape, dtype=bool)
    target[start:end, 33 + 21 * side:54 + 21 * side] = True
    for key in BASE_KEYS[:-1]:
        assert np.array_equal(original[key][~target], changed[key][~target], equal_nan=True), key
        assert not np.any(changed[key][target]), key
    hand_target = np.zeros(original["baseline_roi_hand_score"].shape, dtype=bool)
    hand_target[start:end, side] = True
    assert np.array_equal(original["baseline_roi_hand_score"][~hand_target],
                          changed["baseline_roi_hand_score"][~hand_target], equal_nan=True)
    assert not np.any(changed["baseline_roi_hand_score"][hand_target])
    assert not target[:, :33].any()
    assert not target[:, 33 + 21 * (1-side):54 + 21 * (1-side)].any()


def make_descriptor(row: dict, bundle_index: int, transform: dict, case_index: int) -> dict:
    """Metadata for replay; evaluator_only is never a model input."""
    return {
        "case_id": f"case_{case_index:04d}",
        "source_group": row["source_group"],
        "model_input": {
            "bundle_input_ref": f"input_{bundle_index:04d}",
            "array_allowlist": list(ALLOWLIST),
            "coordinate_transform": transform,
            "rgb_policy": "same original RGB frames; no pixel occlusion or synthetic frame duplication",
            "time_policy": "original frame_id/timestamp_us unchanged",
            "reader": "prepare_scenarios.materialize_model_input",
            "do_not_pass_descriptor_text_to_model": True,
        },
        "evaluator_only": {
            "clip_id": row["clip_id"],
            "target_source_path": row["arrays"],
            "target_source_sha256": row["arrays_sha256"],
            "target_keys": list(BASE_KEYS),
            "reference_type": "saved full-frame MediaPipe-derived baseline ROI coordinates; proxy, not independent anatomy truth",
            "rgb_source_path": row["rgb_path"],
            "rgb_source_sha256": row["rgb_sha256"],
            "signer_id_for_grouping_only": row["signer_id"],
            "word_code_for_reporting_only": row["word_code"],
            "cannot_claim": ["actual physical occlusion recovery accuracy", "3D finger accuracy", "real sign mistake detection"],
        },
    }


def main() -> None:
    manifest_path = UNIT / "MANIFEST.json"
    preparation_path = UNIT / "PREPARE_RESULT.json"
    assert manifest_path.exists() and preparation_path.exists(), "Parent input preparation must finish first"
    manifest = read(manifest_path)
    assert isinstance(manifest, list) and len(manifest) == 61
    manifest = sorted(manifest, key=lambda x: x["clip_id"])
    outputs = (UNIT / "SCENARIOS.json", UNIT / "SCENARIO_CHECK.json")
    assert not any(p.exists() for p in outputs), "Preserve completed scenario artifacts; do not overwrite"
    scenarios, hand_decisions = [], []
    validation = Counter()
    source_pins = {str(manifest_path): sha(manifest_path),
                   str(preparation_path): sha(preparation_path),
                   str(Path(__file__).resolve()): sha(Path(__file__).resolve())}
    for bundle_index, row in enumerate(manifest):
        original = load_baseline(row)
        source_pins[row["arrays"]] = row["arrays_sha256"]
        n = len(original["frame_id"])
        assert n == row["frames"]
        assert original["baseline_roi_xy_px"].shape == (n, 75, 2)
        assert original["baseline_roi_hand_score"].shape == (n, 2)
        intact = make_descriptor(row, bundle_index, {"kind": "none"}, len(scenarios))
        check_materialization(original, materialize_model_input(row, intact), {"kind": "none"})
        scenarios.append(intact)
        validation["intact_controls"] += 1
        for side, name in enumerate(SIDE_NAMES):
            sl = slice(33 + 21 * side, 54 + 21 * side)
            complete = original["baseline_roi_usable"][:, sl].all(axis=1)
            complete &= np.isfinite(original["baseline_roi_xy_px"][:, sl]).all(axis=(1, 2))
            candidates = runs(complete)
            longest = max(candidates, key=lambda x: (x[1]-x[0], -x[0])) if candidates else None
            length = longest[1]-longest[0] if longest else 0
            decision = {
                "clip_id": row["clip_id"], "source_group": row["source_group"],
                "side_index": side, "anatomical_side": name,
                "complete_baseline_roi_frames": int(complete.sum()),
                "longest_run_half_open": list(longest) if longest else None,
                "longest_run_length": length,
                "eligible": length >= MIN_RUN,
                "reason": "eligible" if length >= MIN_RUN else "no_complete_run_of_at_least_11_frames",
            }
            hand_decisions.append(decision)
            if length < MIN_RUN:
                validation["unsupported_hands"] += 1
                continue
            validation["eligible_hands"] += 1
            center = (longest[0] + longest[1] - 1) // 2
            decision["shared_hole_center_index"] = center
            for hole in HOLES:
                start, end = center-hole//2, center+hole//2+1
                assert longest[0] <= start-2 and end+2 <= longest[1]
                assert end-start == hole and complete[start-2:end+2].all()
                transform = {
                    "kind": "single_hand_missing", "side_index": side,
                    "anatomical_side": name, "frame_range_half_open": [start, end],
                    "joint_range_half_open": [33+21*side, 54+21*side],
                    "hole_frames": hole,
                    "shared_center_index": center,
                    "hole_timestamp_first_us": int(original["timestamp_us"][start]),
                    "hole_timestamp_last_us": int(original["timestamp_us"][end-1]),
                    "xy_fill": 0, "observed_fill": False, "usable_fill": False,
                    "conf_fill": 0, "hand_score_fill": 0,
                    "all_other_points_and_body_unchanged": True,
                    "missing_mask_retained": True,
                }
                descriptor = make_descriptor(row, bundle_index, transform, len(scenarios))
                changed = materialize_model_input(row, descriptor)
                check_materialization(original, changed, transform)
                scenarios.append(descriptor)
                validation[f"hole_{hole}_cases"] += 1
                validation["masked_joint_frames"] += hole*21
                validation["materialized_dropout_checks"] += 1
        validation["clips_checked"] += 1
        validation["source_frame_records"] += n
    assert validation["eligible_hands"] + validation["unsupported_hands"] == 122
    assert len(scenarios) == 61 + validation["eligible_hands"]*3
    assert len(hand_decisions) == 122
    for p, h in source_pins.items():
        assert sha(Path(p)) == h, (p, "changed during scenario preparation")
    created = datetime.now(ZoneInfo("Asia/Seoul")).isoformat()
    document = {
        "schema": "wis_temporal_proxy_scenarios_v1",
        "created_at": created,
        "status": "DESCRIPTORS_AND_SAFE_INPUT_TRANSFORMS_CHECKED_NO_INFERENCE_OR_TRAINING",
        "selection": "Per anatomical hand choose longest complete baseline ROI 21-point usable run, tie earliest; require11 frames; use shared central frame for1/3/7-frame holes with>=2 intact frames before/after. No word, annotation core, model score or class result in eligibility/selection.",
        "side_contract": {"0": "anatomical left, indices33:54", "1": "anatomical right, indices54:75", "source": "research/front_keyphase_v1/src/wis_lab/core/observation.py"},
        "input_resolution": "input_NNNN resolves to the NNNN-th clip in MANIFEST sorted by clip_id, outside the model; only returned allowlisted tensors and original RGB reach the model.",
        "safe_reader_contract": "Do not load the entire source NPZ into the candidate. materialize_model_input copies only allowlisted baseline fields and zeros masked coordinates/confidence/hand_score while false-masking observed/usable. Native/raw/redetected arrays, point-origin maps and evaluator_only values are forbidden inputs. The source pack may contain evaluator assets but they are not returned.",
        "ground_truth_limit": "Saved MediaPipe-coordinate proxy reconstruction with unchanged visible RGB. Independent visible2D annotation and real-occlusion/3D truth remain pending. Not a human-action omission test.",
        "grouping": "Every intact/dropout variant remains in its original capture/signer group; do not randomly split variants across learning/evaluation.",
        "case_count": len(scenarios), "hand_decisions": hand_decisions,
        "cases": scenarios,
    }
    check = {
        "schema": "wis_temporal_proxy_scenarios_check_v1", "created_at": created,
        "status": "PASS_DESCRIPTOR_ELIGIBILITY_AND_IN_MEMORY_TRANSFORM_CHECKS",
        "counts": dict(validation), "cases": len(scenarios),
        "source_pins": source_pins,
        "checks": ["all122 hands reported, unsupported not silently excluded", "label/core/score independent window selection", "same center for1/3/7frame holes", "two intact neighbors each side", "complete usable finite21point targets", "body and opposite hand byte values preserved", "only target hand conf/score/coordinates/masks changed", "original timestamps and RGB references unchanged", "strict input allowlist excludes native/raw/redetected/evaluator values", "all variants remain original source group", "source files unchanged"],
        "new_inference": 0, "new_training": 0, "altered_arrays_saved": 0,
        "manual_anatomical_ground_truth": "PENDING",
        "verification_kind": "same subagent recomputed eligibility and in-memory tensor mutations; independent parent verification still required",
    }
    for path, obj in zip(outputs, (document, check)):
        with path.open("x", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
    print(json.dumps({"status":check["status"],"counts":dict(validation),"cases":len(scenarios)},ensure_ascii=False))


if __name__ == "__main__":
    main()
