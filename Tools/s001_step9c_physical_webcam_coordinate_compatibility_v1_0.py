#!/usr/bin/env python3
"""Step 9C readiness recorder and strict physical evidence parser/verifier.

This tool deliberately does not synthesize physical captures and does not create a
delivery package while the fixed webcam protocol is waiting for the user.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/sessions/S001_P001/step9c_physical_webcam_coordinate_compatibility_v1_0"
RUN_ID = "8d17ef49-89ff-4d40-ae59-0386cc406e2e"
SCHEMA_VERSION = "words_in_silence.step9c.physical_capture.v1"
FIXED_TAKES = ((1, "GREETING"), (2, "LOVE"), (3, "THANK_YOU"))
FROZEN_THRESHOLD = 0.2908679008489196
APPROVED_DELIVERIES = {
    "WordsInSilence_step8b_recognition_core_v1_0_delivery.zip": (661_831, "3401ea74f73902c5ea7b4f34abb06ebfe933ef0a076820f15b6471937a5c7c6d"),
    "WordsInSilence_step8c_offline_recognition_evaluation_v1_0_delivery.zip": (445_505, "e1cf9d7f197feba99cc0499795f3f04702cd110bbe6d3200341a7a0aa2d55900"),
    "WordsInSilence_step9a_runtime_recognition_service_v1_0_delivery.zip": (415_867, "497fb4bedc20b8439e8e00e8bf56cb5f4f199cdc2143da5f366e3a90fd069148"),
    "WordsInSilence_step9b_live_mediapipe_input_v1_0_delivery.zip": (337_244, "ab7b55ea4fbb0f1033d944a019ebf389000753bc6ce51635fcce81959d38d48d"),
}
STEP9B_INPUT = ROOT / "data/processed/sessions/S001_P001/step9b_live_mediapipe_input_v1_0/input_manifest.json"
STEP9B_OUTPUT = ROOT / "data/processed/sessions/S001_P001/step9b_live_mediapipe_input_v1_0/output_file_manifest.json"
CAPTURE_PLAN = ROOT / "Assets/StreamingAssets/RecognitionCoreV1/PhysicalCompatibility/step9c_capture_plan.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(ROOT).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def verify_manifest(path: Path, field: str) -> tuple[int, list[dict[str, Any]]]:
    manifest = read_json(path)
    rows = manifest[field]
    mismatches: list[dict[str, Any]] = []
    for row in rows:
        target = ROOT / row["relative_path"]
        if not target.is_file():
            mismatches.append({"path": row["relative_path"], "reason": "missing"})
            continue
        actual_size = target.stat().st_size
        actual_sha = sha256_file(target)
        if actual_size != row["size_bytes"] or actual_sha != row["sha256"]:
            mismatches.append({
                "path": row["relative_path"],
                "reason": "size_or_sha256",
                "expected_size": row["size_bytes"],
                "actual_size": actual_size,
                "expected_sha256": row["sha256"],
                "actual_sha256": actual_sha,
            })
    return len(rows), mismatches


def unity_summary(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    return {
        "result": root.attrib["result"],
        "total": int(root.attrib["total"]),
        "passed": int(root.attrib["passed"]),
        "failed": int(root.attrib["failed"]),
        "skipped": int(root.attrib["skipped"]),
        "inconclusive": int(root.attrib["inconclusive"]),
        "duration_seconds": float(root.attrib["duration"]),
    }


def add_assertion(rows: list[dict[str, Any]], name: str, passed: bool, detail: Any) -> None:
    rows.append({"name": name, "status": "PASS" if passed else "FAIL", "detail": detail})


def prepare_waiting(edit_results: Path, play_results: Path) -> int:
    assertions: list[dict[str, Any]] = []
    deliveries: list[dict[str, Any]] = []
    for name, (expected_size, expected_sha) in APPROVED_DELIVERIES.items():
        path = ROOT / name
        actual = file_record(path) if path.is_file() else {"relative_path": name, "size_bytes": None, "sha256": None}
        passed = actual["size_bytes"] == expected_size and actual["sha256"] == expected_sha
        actual.update({"expected_size_bytes": expected_size, "expected_sha256": expected_sha, "verified": passed})
        deliveries.append(actual)
        add_assertion(assertions, f"approved_delivery::{name}", passed, actual)

    input_count, input_mismatches = verify_manifest(STEP9B_INPUT, "inputs")
    output_count, output_mismatches = verify_manifest(STEP9B_OUTPUT, "outputs")
    add_assertion(assertions, "step9b_input_manifest_all_records", input_count == 103 and not input_mismatches,
                  {"records": input_count, "mismatches": input_mismatches})
    add_assertion(assertions, "step9b_output_manifest_all_records", output_count == 44 and not output_mismatches,
                  {"records": output_count, "mismatches": output_mismatches})

    project_version = (ROOT / "ProjectSettings/ProjectVersion.txt").read_text(encoding="utf-8")
    package = read_json(ROOT / "Packages/com.github.homuler.mediapipe/package.json")
    add_assertion(assertions, "unity_version_6000_3_14f1", "m_EditorVersion: 6000.3.14f1" in project_version, project_version.splitlines()[0])
    add_assertion(assertions, "mediapipe_package_0_16_3", package.get("version") == "0.16.3", package.get("version"))
    add_assertion(assertions, "provider_scene_present", (ROOT / "Assets/_Project/Scenes/Technical/02_PoseIntegration.unity").is_file(), "02_PoseIntegration.unity")
    add_assertion(assertions, "approved_step9b_prefab_present", (ROOT / "Assets/_Project/Resources/Diagnostics/LiveMediaPipeInputDiagnostic.prefab").is_file(), "LiveMediaPipeInputDiagnostic.prefab")
    add_assertion(assertions, "step9c_prefab_present", (ROOT / "Assets/_Project/Resources/Diagnostics/PhysicalWebcamCoordinateCompatibility.prefab").is_file(), "PhysicalWebcamCoordinateCompatibility.prefab")

    plan = read_json(CAPTURE_PLAN)
    plan_order = [(item["take_number"], item["expected_word"]) for item in plan["fixed_take_order"]]
    add_assertion(assertions, "capture_plan_run_id", plan.get("run_id") == RUN_ID, plan.get("run_id"))
    add_assertion(assertions, "capture_plan_fixed_order", tuple(plan_order) == FIXED_TAKES, plan_order)
    add_assertion(assertions, "capture_plan_frozen_threshold", plan.get("frozen_threshold") == FROZEN_THRESHOLD, plan.get("frozen_threshold"))
    add_assertion(assertions, "capture_plan_no_outcome_retake", plan.get("retake_on_wrong_or_rejected_prediction") is False, plan.get("retake_on_wrong_or_rejected_prediction"))
    add_assertion(assertions, "capture_plan_no_pixel_media", plan.get("pixel_media_recording") is False, plan.get("pixel_media_recording"))

    edit = unity_summary(edit_results)
    play = unity_summary(play_results)
    add_assertion(assertions, "unity_editmode_step9c_only", edit == {
        **edit, "result": "Passed", "total": 9, "passed": 9, "failed": 0, "skipped": 0, "inconclusive": 0
    }, edit)
    add_assertion(assertions, "unity_playmode_step9c_only", play == {
        **play, "result": "Passed", "total": 1, "passed": 1, "failed": 0, "skipped": 0, "inconclusive": 0
    }, play)

    physical_dir = OUT / "physical_captures"
    physical_files = sorted(path.name for path in physical_dir.glob("*") if path.is_file()) if physical_dir.is_dir() else []
    add_assertion(assertions, "no_physical_capture_claimed", not physical_files, physical_files)
    package_path = ROOT / "WordsInSilence_step9c_physical_webcam_coordinate_compatibility_v1_0_delivery.zip"
    add_assertion(assertions, "no_premature_delivery_package", not package_path.exists(), package_path.name)
    names = [row["name"] for row in assertions]
    add_assertion(assertions, "assertion_names_unique", len(names) == len(set(names)), {"count": len(names), "unique": len(set(names))})

    input_manifest = {
        "schema_version": "words_in_silence.step9c.input_manifest.v1",
        "run_id": RUN_ID,
        "approved_state": "LIVE_MEDIAPIPE_INPUT_V1_0_EXTERNAL_REVIEW_PASS",
        "approved_deliveries": deliveries,
        "approved_manifest_bindings": [file_record(STEP9B_INPUT), file_record(STEP9B_OUTPUT)],
        "step9b_input_records_verified": input_count,
        "step9b_output_records_verified": output_count,
        "step9b_manifest_mismatches": len(input_mismatches) + len(output_mismatches),
        "unity_version": "6000.3.14f1",
        "mediapipe_package_version": "0.16.3",
        "capture_plan": file_record(CAPTURE_PLAN),
    }
    atomic_json(OUT / "input_manifest.json", input_manifest)
    atomic_json(OUT / "unity_editmode_test_summary.json", {"run_id": RUN_ID, **edit})
    atomic_json(OUT / "unity_playmode_test_summary.json", {"run_id": RUN_ID, **play})
    edit_command = "& 'C:\\Program Files\\Unity\\Hub\\Editor\\6000.3.14f1\\Editor\\Unity.exe' -batchmode -projectPath 'C:\\MyUnityProjects\\WordsInSilence' -runTests -testPlatform EditMode -testFilter 'WordsInSilence.Tests.EditMode.PhysicalWebcamCoordinateCompatibilityV1Tests' -testResults 'C:\\MyUnityProjects\\WordsInSilence\\data\\processed\\sessions\\S001_P001\\step9c_physical_webcam_coordinate_compatibility_v1_0\\unity_editmode_test_results.xml' -logFile 'C:\\MyUnityProjects\\WordsInSilence\\data\\processed\\sessions\\S001_P001\\step9c_physical_webcam_coordinate_compatibility_v1_0\\unity_editmode_test.log'\n"
    play_command = "& 'C:\\Program Files\\Unity\\Hub\\Editor\\6000.3.14f1\\Editor\\Unity.exe' -batchmode -projectPath 'C:\\MyUnityProjects\\WordsInSilence' -runTests -testPlatform PlayMode -testFilter 'WordsInSilence.Tests.PlayMode.PhysicalWebcamCoordinateCompatibilityHarnessV1Tests' -testResults 'C:\\MyUnityProjects\\WordsInSilence\\data\\processed\\sessions\\S001_P001\\step9c_physical_webcam_coordinate_compatibility_v1_0\\unity_playmode_test_results.xml' -logFile 'C:\\MyUnityProjects\\WordsInSilence\\data\\processed\\sessions\\S001_P001\\step9c_physical_webcam_coordinate_compatibility_v1_0\\unity_playmode_test.log'\n"
    atomic_text(OUT / "unity_editmode_test_command.txt", edit_command)
    atomic_text(OUT / "unity_playmode_test_command.txt", play_command)
    passed = sum(row["status"] == "PASS" for row in assertions)
    failed = len(assertions) - passed
    atomic_json(OUT / "readiness_assertion_results.json", {
        "schema_version": "words_in_silence.step9c.readiness_assertions.v1",
        "run_id": RUN_ID,
        "summary": {"total": len(assertions), "passed": passed, "failed": failed,
                    "duplicate_names": len(assertions) - len({row["name"] for row in assertions})},
        "assertions": assertions,
    })
    atomic_json(OUT / "run_state.json", {
        "schema_version": "words_in_silence.step9c.run_state.v1",
        "run_id": RUN_ID,
        "status": "PHYSICAL_WEBCAM_COMPATIBILITY_WAITING_FOR_USER",
        "physical_attempt_occurred": False,
        "physical_capture_count": 0,
        "fixed_take_order": [{"take_number": number, "expected_word": word} for number, word in FIXED_TAKES],
        "unity_tests": {"editmode": edit, "playmode": play},
        "readiness_assertions": {"passed": passed, "failed": failed},
        "delivery_package_created": False,
    })
    print(json.dumps({"status": "PHYSICAL_WEBCAM_COMPATIBILITY_WAITING_FOR_USER", "assertions_passed": passed,
                      "assertions_failed": failed, "editmode": edit, "playmode": play}, ensure_ascii=False))
    return 0 if failed == 0 else 1


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def verify_evidence(path: Path) -> dict[str, Any]:
    path = path.resolve()
    checksum_path = Path(str(path) + ".sha256")
    raw = path.read_bytes()
    actual_hash = hashlib.sha256(raw).hexdigest()
    checksum = checksum_path.read_text(encoding="utf-8")
    expected_line = f"{actual_hash}  {path.name}\n"
    errors: list[str] = []
    if checksum != expected_line:
        errors.append("detached_sha256")
    payload = json.loads(raw.decode("utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version")
    if payload.get("run_id") != RUN_ID:
        errors.append("run_id")
    take = payload.get("take_number")
    expected_word = payload.get("expected_word")
    if not isinstance(take, int) or take < 1 or take > 3 or expected_word != FIXED_TAKES[take - 1][1]:
        errors.append("fixed_take_order")
    try:
        uuid.UUID(payload.get("run_id", ""))
    except (ValueError, AttributeError):
        errors.append("run_id_uuid")
    frames = payload.get("frames")
    if not isinstance(frames, list):
        errors.append("frames")
        frames = []
    timestamps: list[float] = []
    normalized_frames = 0
    normalized_valid = 0
    for index, frame in enumerate(frames):
        timestamp = frame.get("timestamp_seconds")
        if not finite_number(timestamp):
            errors.append(f"timestamp[{index}]")
        else:
            timestamps.append(float(timestamp))
        mask = frame.get("normalized_mask")
        coordinates = frame.get("normalized_coordinates")
        if not isinstance(mask, list) or len(mask) != 16 or not all(isinstance(item, bool) for item in mask):
            errors.append(f"mask[{index}]")
            continue
        if not isinstance(coordinates, list) or len(coordinates) != 16:
            errors.append(f"coordinates[{index}]")
            continue
        if frame.get("has_normalized_frame") is True:
            normalized_frames += 1
        for landmark, valid in enumerate(mask):
            point = coordinates[landmark]
            if not isinstance(point, list) or len(point) != 3:
                errors.append(f"coordinate_shape[{index},{landmark}]")
            elif valid:
                normalized_valid += 1
                if not all(finite_number(value) for value in point):
                    errors.append(f"valid_coordinate[{index},{landmark}]")
            elif point != [None, None, None]:
                errors.append(f"invalid_coordinate_not_null[{index},{landmark}]")
    strict = all(timestamps[index] > timestamps[index - 1] for index in range(1, len(timestamps))) and len(timestamps) == len(frames)
    summary = payload.get("timestamp_summary", {})
    counts = payload.get("counts", {})
    if summary.get("strictly_increasing") is not strict:
        errors.append("timestamp_monotonicity_summary")
    if summary.get("timestamps") != timestamps:
        errors.append("lossless_timestamps")
    if counts.get("raw_frames") != len(frames):
        errors.append("raw_frame_count")
    if counts.get("normalized_frames") != normalized_frames:
        errors.append("normalized_frame_count")
    if counts.get("normalized_valid_landmarks") != normalized_valid:
        errors.append("normalized_valid_count")
    recognition = payload.get("recognition")
    failure = payload.get("failure")
    if recognition is not None:
        if recognition.get("frozen_threshold") != FROZEN_THRESHOLD:
            errors.append("frozen_threshold")
        if recognition.get("threshold_rule") != "inclusive_less_than_or_equal":
            errors.append("threshold_rule")
        if len(recognition.get("template_distances", [])) != 9:
            errors.append("nine_template_distances")
        best = recognition.get("best_distance")
        accepted = recognition.get("accepted")
        if not finite_number(best) or accepted is not (best <= FROZEN_THRESHOLD):
            errors.append("inclusive_gate_decision")
    elif not isinstance(failure, dict):
        errors.append("recognition_or_typed_failure")
    forbidden_keys = {"pixels", "image", "images", "video", "audio", "screenshot", "screenshots"}
    if forbidden_keys.intersection(payload):
        errors.append("forbidden_media_payload")
    return {
        "status": "PASS" if not errors else "FAIL",
        "path": str(path),
        "size_bytes": len(raw),
        "sha256": actual_hash,
        "capture_id": payload.get("capture_id"),
        "take_number": take,
        "expected_word": expected_word,
        "raw_frames": len(frames),
        "normalized_frames": normalized_frames,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    waiting = subparsers.add_parser("prepare-waiting")
    waiting.add_argument("--edit-results", type=Path, required=True)
    waiting.add_argument("--play-results", type=Path, required=True)
    verify = subparsers.add_parser("verify-evidence")
    verify.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command == "prepare-waiting":
        return prepare_waiting((ROOT / args.edit_results).resolve() if not args.edit_results.is_absolute() else args.edit_results,
                               (ROOT / args.play_results).resolve() if not args.play_results.is_absolute() else args.play_results)
    result = verify_evidence(args.path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
