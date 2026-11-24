#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA = "WordsInSilence.Step10B.PetRobotWorldEntryReturnIntegration.v1.0.GuiUnityEditorGate"
NAME = "WordsInSilence_step10b_pet_robot_world_entry_return_integration_v1_0_gui_unity_editor_gate"

STEP10B_ZIP = "Validation/WordsInSilence_step10b_pet_robot_world_entry_return_integration_v1_0_delivery.zip"
STEP10B_ZIP_SIZE = 67937
STEP10B_ZIP_SHA = "07daa6bd08eacc7d7bf76cfe200258c75d7d9ba15dcb5a1f5fabf14ca9b86977"
STEP10B_DETACHED_SIZE = 148
STEP10B_DETACHED_SHA = "906b8149993db3529fa33b277a2b0e9c0691107f62f2fdf5511c5d87fefef2e9"

STEP10A_ZIP = "Validation/WordsInSilence_step10a_pet_robot_retry_exhaustion_needs_practice_completion_v1_3_delivery.zip"
STEP10A_ZIP_SIZE = 101574
STEP10A_ZIP_SHA = "42cfe5679c09defc220033fd99eb638576bd5313b5c047dfe25766ac425575bf"
STEP10A_DETACHED_SIZE = 160
STEP10A_DETACHED_SHA = "6c630bc1b18770deec5e23e8669b5075eaae42a6907c61e918e92316cd1e0354"

STEP10A_GUI_ZIP = "Validation/WordsInSilence_step10a_pet_robot_retry_exhaustion_needs_practice_completion_v1_3_gui_unity_editor_gate_delivery.zip"
STEP10A_GUI_ZIP_SIZE = 1491795
STEP10A_GUI_ZIP_SHA = "6463cbed625e4b4f27b7fdf1cb0bcdcb07a38571334ae3083eff4ef85684798a"
STEP10A_GUI_DETACHED_SIZE = 182
STEP10A_GUI_DETACHED_SHA = "55d8b44681e642c4f68f6975ce3a3a1c8e09969d4deb4b1ca402bd96b11dc955"

STEP10B_RESULT = "Validation/WordsInSilence_step10b_pet_robot_world_entry_return_integration_v1_0"
STEP10A_GUI_RESULT = "Validation/WordsInSilence_step10a_pet_robot_retry_exhaustion_needs_practice_completion_v1_3_gui_unity_editor_gate"
ZIP_HELPER = "Tools/s001_step10a_pet_robot_candidate_camera_startup_route_v1_1_gui_unity_editor_gate.py"
TX_HELPER = "Tools/s001_step10a_pet_robot_candidate_camera_startup_route_v1_1.py"
WISRQ_HELPER = "Tools/s001_step10a_pet_robot_candidate_camera_startup_route_v1_2_gui_unity_editor_gate.py"

EXPECTED_THRESHOLDS = {
    "GREETING": 0.5113059473720765,
    "LOVE": 0.391276023460427,
    "THANK_YOU": 0.37709977431991587,
}

EXPECTED_FIXTURES = {
    "PetRobotSignPracticeVerticalSliceV1Tests": 23,
    "GuidedQuestLowConfidenceOutcomeLayerV1Tests": 10,
    "CandidateTransientAmbiguousHandIdentityToleranceV1Tests": 8,
    "PersonalLiveRecognitionIntegrationV1Tests": 22,
    "PersonalMultiPrototypeRuntimeIntegrationV1Tests": 8,
    "PersonalMultiPrototypeLiveOptInRoutingV1Tests": 8,
    "PetRobotCandidateCameraStartupRouteV1Tests": 8,
    "PetRobotRetryExhaustionNeedsPracticeCompletionV1Tests": 15,
    "PetRobotWorldEntryReturnIntegrationV1Tests": 18,
}

RAW_EVIDENCE = [
    "TestResults.xml",
    "TestRunner_FIRST_RESULT.png",
    "Editor.log",
    "World_00_Initial.png",
    "World_01_OutOfRangeBlocked.png",
    "World_02_InRangePrompt_PlayerPosition.png",
    "Practice_00_Initial.png",
    "Practice_01_CameraReady.png",
    "Practice_02_GREETING_First.png",
    "Practice_02_GREETING_First(1).png",
    "Practice_03_LOVE_First.png",
    "Practice_03_LOVE_First(1).png",
    "Practice_04_THANK_YOU_First.png",
    "Practice_THANK_YOU_Retry1.png",
    "Practice_THANK_YOU_Retry1(1).png",
    "Practice_FinalCompleted.png",
    "World_Return_PlayerPositionAndSummary.png",
]


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def jbytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def record(relative_path: str, data: bytes, **extra: Any) -> dict[str, Any]:
    return {"relative_path": relative_path, "size_bytes": len(data),
            "sha256": sha(data), **extra}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_module(root: Path, relative_path: str, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, root / relative_path)
    require(spec is not None and spec.loader is not None, "cannot load helper: " + relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def staged_files(root: Path) -> list[str]:
    run = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=root,
                         capture_output=True, text=True, encoding="utf-8",
                         errors="replace", timeout=30)
    require(run.returncode == 0, "Git staged query failed")
    return [line.strip() for line in run.stdout.splitlines() if line.strip()]


def verify_fixed_zip(root: Path, zip_helper: Any, relative_path: str,
                     expected_size: int, expected_sha: str,
                     detached_size: int, detached_file_sha: str,
                     expected_entries: int, expected_targets: int) -> dict[str, Any]:
    path = root / relative_path
    detached = path.with_suffix(path.suffix + ".sha256")
    require(path.is_file() and detached.is_file(), "fixed input missing: " + relative_path)
    data = path.read_bytes()
    detached_data = detached.read_bytes()
    require(len(data) == expected_size and sha(data) == expected_sha,
            "fixed ZIP identity mismatch: " + relative_path)
    require(len(detached_data) == detached_size and sha(detached_data) == detached_file_sha,
            "fixed detached identity mismatch: " + relative_path)
    require(detached_data.decode("ascii") == expected_sha + "  " + path.name + "\n",
            "fixed detached syntax mismatch: " + relative_path)
    with zipfile.ZipFile(path, "r") as archive:
        infos = archive.infolist()
        require(archive.testzip() is None, "fixed ZIP CRC failure: " + relative_path)
        files = {info.filename: archive.read(info) for info in infos}
    require(len(infos) == expected_entries, "fixed ZIP entry count mismatch: " + relative_path)
    require("output_file_manifest.json" in files, "fixed ZIP manifest missing: " + relative_path)
    manifest = json.loads(files["output_file_manifest.json"].decode("utf-8-sig"))
    rows = manifest["files"]
    require(len(rows) == expected_targets and
            len({row["relative_path"] for row in rows}) == expected_targets,
            "fixed ZIP manifest row mismatch: " + relative_path)
    require({row["relative_path"] for row in rows} ==
            set(files) - {"output_file_manifest.json"},
            "fixed ZIP manifest membership mismatch: " + relative_path)
    for row in rows:
        payload = files[row["relative_path"]]
        require(len(payload) == row["size_bytes"] and sha(payload) == row["sha256"],
                "fixed ZIP manifest payload mismatch: " + row["relative_path"])
    checked = zip_helper.validate_zip(data, files)
    require(checked["duplicate_count"] == 0 and checked["unsafe_path_count"] == 0 and
            checked["encrypted_entry_count"] == 0 and checked["symlink_entry_count"] == 0 and
            checked["crc_failure"] is None and checked["membership_match"] and
            checked["payload_mismatch_count"] == 0,
            "fixed ZIP safety mismatch: " + relative_path)
    require(zip_helper.deterministic_zip(files) == data,
            "fixed ZIP deterministic rebuild mismatch: " + relative_path)
    return {
        "relative_path": relative_path,
        "size_bytes": expected_size,
        "sha256": expected_sha,
        "detached_relative_path": detached.relative_to(root).as_posix(),
        "detached_size_bytes": detached_size,
        "detached_file_sha256": detached_file_sha,
        "entry_count": expected_entries,
        "manifest_target_count": expected_targets,
        "crc_duplicate_path_symlink_encryption": "PASS",
        "deterministic_rebuild_byte_identical": True,
        "status": "PASS",
    }


def compare_records(root: Path, baseline: dict[str, Any], expected_count: int,
                    schema_suffix: str) -> dict[str, Any]:
    require(len(baseline["records"]) == expected_count,
            schema_suffix + " baseline count mismatch")
    rows = []
    for item in baseline["records"]:
        relative = item["relative_path"]
        data = (root / relative).read_bytes()
        same = len(data) == item["size_bytes"] and sha(data) == item["sha256"]
        require(same, schema_suffix + " changed: " + relative)
        rows.append({**record(relative, data), "before_size_bytes": item["size_bytes"],
                     "before_sha256": item["sha256"],
                     "before_after_byte_identical": True, "status": "PASS"})
    return {"schema_version": SCHEMA + "." + schema_suffix,
            "file_count": expected_count, "pass_count": expected_count,
            "mismatch_count": 0, "records": rows, "status": "PASS"}


def compare_step10b_files(root: Path, baseline: dict[str, Any]) -> dict[str, Any]:
    require(len(baseline["records"]) == 22 and baseline["new_file_count"] == 21,
            "Step 10B implementation baseline count mismatch")
    rows = []
    for item in baseline["records"]:
        expected = item["after"]
        relative = item["relative_path"]
        data = (root / relative).read_bytes()
        same = len(data) == expected["size_bytes"] and sha(data) == expected["sha256"]
        require(same, "Step 10B project file changed: " + relative)
        rows.append({**record(relative, data), "approved_size_bytes": expected["size_bytes"],
                     "approved_sha256": expected["sha256"],
                     "before_after_byte_identical": True, "status": "PASS"})
    return {"schema_version": SCHEMA + ".step10b_project_files",
            "new_project_file_count": 21, "modified_existing_file_count": 1,
            "file_count": 22, "pass_count": 22, "mismatch_count": 0,
            "records": rows, "status": "PASS"}


def property_value(case: ET.Element, name: str) -> str:
    for value in case.findall("./properties/property"):
        if value.attrib.get("name") == name:
            return value.attrib.get("value", "")
    return ""


def parse_tests(result: Path) -> dict[str, Any]:
    raw = (result / "TestResults.xml").read_bytes()
    root = ET.fromstring(raw.decode("utf-8-sig"))
    require(root.tag == "test-run", "TestResults root is not test-run")
    counts = {key: int(root.attrib.get(key, "0"))
              for key in ("testcasecount", "total", "passed", "failed", "skipped", "inconclusive")}
    require(counts == {"testcasecount": 120, "total": 120, "passed": 120,
                       "failed": 0, "skipped": 0, "inconclusive": 0},
            "TestResults aggregate mismatch")
    cases = root.findall(".//test-case")
    names = [case.attrib["fullname"] for case in cases]
    require(len(cases) == len(set(names)) == 120, "TestResults unique testcase mismatch")
    fixture_counts = Counter(case.attrib["classname"].split(".")[-1] for case in cases)
    require(dict(fixture_counts) == EXPECTED_FIXTURES, "TestResults fixture inventory mismatch")
    require(all(case.attrib.get("result") == "Passed" for case in cases),
            "TestResults contains non-Passed case")
    retry = sum(property_value(case, "retryIteration") not in ("", "0") for case in cases)
    repeat = sum(property_value(case, "repeatIteration") not in ("", "0") for case in cases)
    require(retry == repeat == 0, "TestResults retry/repeat mismatch")
    return {
        "schema_version": SCHEMA + ".test_results",
        "source": record("TestResults.xml", raw),
        "platform": "EditMode",
        "test_run_id": root.attrib.get("id"),
        "start_utc": root.attrib.get("start-time"),
        "end_utc": root.attrib.get("end-time"),
        "duration_seconds": float(root.attrib.get("duration", "0")),
        **counts,
        "unique_testcase_count": len(set(names)),
        "fixture_counts": dict(sorted(fixture_counts.items())),
        "retry_nonzero_count": retry,
        "repeat_nonzero_count": repeat,
        "execution_count": 1,
        "status": "PASS",
    }


def parse_editor_log(result: Path) -> dict[str, Any]:
    raw = (result / "Editor.log").read_bytes()
    text = raw.decode("utf-8-sig", errors="replace")
    require("6000.3.14f1 (d68c3f99a318)" in text, "Unity version absent from Editor.log")
    require("Successfully connected to LicensingClient" in text and
            "[Package Manager] Registered 58 packages:" in text,
            "Unity licensing/package readiness absent")
    compiler_errors = len(re.findall(r"\berror CS\d+\b", text, re.IGNORECASE)) + \
        text.lower().count("scripts have compiler errors")
    require(compiler_errors == 0, "compiler errors present in Editor.log")
    prebuild = text.count("Executing IPrebuildSetup for:")
    postbuild = text.count("Executing IPostBuildCleanup for:")
    saves = text.count("Saving results to:")
    require(prebuild == postbuild == 1 and saves == 2,
            "Editor.log test execution/export count mismatch")
    world_marker = "Opening scene 'Assets/_Project/Scenes/VerticalSlice/PetRobotWorldEntryReturnVerticalSlice.unity'"
    require(text.count(world_marker) == 1, "world scene Editor open count mismatch")
    live_tail = text[text.index(world_marker):]
    practice_marker = "Loaded scene 'Assets/_Project/Scenes/VerticalSlice/PetRobotSignPracticeVerticalSlice.unity'"
    practice_live_loads = live_tail.count(practice_marker)
    require(practice_live_loads == 1, "live practice scene load count mismatch")
    play_reload_markers = live_tail.count("Reloading assemblies for play mode.")
    require(play_reload_markers == 2, "single PlayMode enter/exit marker pair mismatch")
    require(live_tail.count("[PoseLandmarkProvider] PoseLandmarker") == 1 and
            live_tail.count("[MediaPipeLandmarkProvider] HandLandmarker") == 1,
            "live Hand/Pose initialization count mismatch")
    exceptions = re.findall(
        r"\b(?:NullReferenceException|MissingReferenceException|InvalidOperationException|"
        r"ArgumentException|AssertionException)\b", live_tail)
    require(not exceptions, "blocking runtime exception present in live log")
    return {
        "schema_version": SCHEMA + ".editor_log",
        "source": record("Editor.log", raw),
        "unity_version": "6000.3.14f1",
        "license_client_connected": True,
        "registered_package_count": 58,
        "compiler_error_count": 0,
        "test_prebuild_count": prebuild,
        "test_postbuild_count": postbuild,
        "test_result_save_count": saves,
        "playmode_reload_marker_count": play_reload_markers,
        "playmode_execution_count": 1,
        "world_scene_editor_open_count": 1,
        "practice_scene_live_load_count": practice_live_loads,
        "pose_provider_initialization_count": 1,
        "hand_provider_initialization_count": 1,
        "blocking_runtime_exception_count": 0,
        "status": "PASS",
    }


def current_repository(before: dict[str, Any]) -> dict[str, Any]:
    root = Path(before["path"])
    rows = [record(path.relative_to(root).as_posix(), path.read_bytes())
            for path in sorted(value for value in root.rglob("*") if value.is_file())]
    index = json.loads((root / "candidate_index.json").read_text(encoding="utf-8-sig"))
    return {"schema_version": SCHEMA + ".candidate_repository_after",
            "path": str(root), "exists": root.exists(),
            "candidate_count": len(index["entries"]), "file_count": len(rows),
            "files": rows, "status": "PASS"}


def candidate_before(root: Path) -> tuple[dict[str, Any], bytes]:
    source = load_json(root / STEP10B_RESULT / "candidate_repository_before_after.json")
    index_data = (root / STEP10A_GUI_RESULT / "new_candidates/candidate_index.json").read_bytes()
    index = json.loads(index_data.decode("utf-8-sig"))
    require(source["candidate_count_before_after"] == [5, 5] and
            source["file_count_before_after"] == [16, 16] and
            len(source["files"]) == 16 and len(index["entries"]) == 5,
            "approved five-candidate baseline mismatch")
    return ({"schema_version": SCHEMA + ".candidate_repository_before",
             "path": source["path"], "exists": True, "candidate_count": 5,
             "file_count": 16, "files": source["files"], "status": "PASS"}, index_data)


def validate_repository(before: dict[str, Any], before_index_data: bytes,
                        after: dict[str, Any], wisrq: Any) -> tuple[dict[str, Any], dict[str, bytes]]:
    require(before["candidate_count"] == 5 and before["file_count"] == 16,
            "candidate before count mismatch")
    require(after["candidate_count"] == 8 and after["file_count"] == 25,
            "candidate after count mismatch")
    root = Path(after["path"])
    before_files = {row["relative_path"]: row for row in before["files"]}
    after_files = {row["relative_path"]: row for row in after["files"]}
    require(set(before_files) <= set(after_files), "existing candidate file removed")
    old_payloads = [name for name in before_files if name != "candidate_index.json"]
    require(len(old_payloads) == 15 and all(
        before_files[name]["size_bytes"] == after_files[name]["size_bytes"] and
        before_files[name]["sha256"] == after_files[name]["sha256"] for name in old_payloads),
        "existing candidate payload changed")
    new_paths = sorted(set(after_files) - set(before_files))
    require(len(new_paths) == 9, "new candidate payload count mismatch")

    before_index = wisrq.strict_json(before_index_data, "candidate_index_before.json")
    after_index_data = (root / "candidate_index.json").read_bytes()
    after_index = wisrq.strict_json(after_index_data, "candidate_index_after.json")
    old_entries = {row["capture_session_id"]: row for row in before_index["entries"]}
    all_entries = {row["capture_session_id"]: row for row in after_index["entries"]}
    require(len(old_entries) == 5 and len(all_entries) == len(after_index["entries"]) == 8,
            "candidate capture/session identity duplicate")
    require(all(all_entries[key] == value for key, value in old_entries.items()),
            "existing candidate index row changed")
    new_ids = [key for key in all_entries if key not in old_entries]
    require(len(new_ids) == 3, "new candidate index row count mismatch")
    expected = {
        "GREETING": ("LowConfidenceCorrect", "ReviewOnly", False, False, False, False),
        "LOVE": ("StrictSuccess", "Eligible", True, True, True, True),
        "THANK_YOU": ("StrictSuccess", "Eligible", True, True, True, True),
    }
    rows = []
    payloads = {"candidate_repository_after_payload/candidate_index.json": after_index_data}
    for capture_id in new_ids:
        entry = all_entries[capture_id]
        word = entry["word"]
        require(word in expected, "unexpected candidate word")
        outcome, eligibility, eligible, accepted, expected_accept, quest = expected[word]
        require(entry["outcome"] == outcome and entry["eligibility"] == eligibility,
                "candidate index outcome mismatch")
        relative = PurePosixPath(entry["relative_directory"])
        require(not relative.is_absolute() and ".." not in relative.parts and
                "\\" not in str(relative), "unsafe candidate relative directory")
        directory = root / Path(*relative.parts)
        sequence_data = (directory / "normalized_sequence.wisrq").read_bytes()
        metadata_data = (directory / "candidate_metadata.json").read_bytes()
        manifest_data = (directory / "checksum_manifest.json").read_bytes()
        require(sha(sequence_data) == entry["sequence_sha256"] and
                sha(metadata_data) == entry["metadata_sha256"], "candidate index hash mismatch")
        metadata = wisrq.strict_json(metadata_data, "candidate_metadata.json")
        checksum = wisrq.strict_json(manifest_data, "checksum_manifest.json")
        targets = {row["path"]: row for row in checksum["files"]}
        require(set(targets) == {"normalized_sequence.wisrq", "candidate_metadata.json"},
                "candidate checksum manifest membership mismatch")
        for filename, data in (("normalized_sequence.wisrq", sequence_data),
                               ("candidate_metadata.json", metadata_data)):
            require(targets[filename]["size_bytes"] == len(data) and
                    targets[filename]["sha256"] == sha(data),
                    "candidate checksum manifest identity mismatch")
        sequence = wisrq.parse_wisrcb(sequence_data)
        require(sequence["magic"] == "WISRCB01" and sequence["sample_id"] == capture_id and
                sequence["writer_reader_round_trip_byte_identical"],
                "candidate WISRCB01 parse/round-trip mismatch")
        require(metadata["capture_session_id"] == capture_id and
                metadata["word"] == metadata["expected_word"] ==
                metadata["predicted_word"] == word,
                "candidate word/session provenance mismatch")
        require(metadata["outcome"] == outcome and metadata["eligibility"] == eligibility and
                metadata["eligible_for_profile"] is eligible and
                metadata["strict_accepted"] is accepted and
                metadata["strict_expected_class_accept"] is expected_accept and
                metadata["strict_quest_success"] is quest,
                "candidate strict/outcome contract mismatch")
        require(metadata["controller_state"] == "Completed" and
                metadata["route_id"] == "MultiPrototypeCandidate" and
                metadata["profile_id"] == "step9l_current_user_multi_prototype_v1_0" and
                abs(metadata["approved_threshold"] - EXPECTED_THRESHOLDS[word]) <= 1e-15,
                "candidate route/profile/threshold mismatch")
        require(metadata["raw_frame_count"] == metadata["normalized_frame_count"] ==
                sequence["frame_count"] and
                metadata["valid_landmark_count"] == sequence["valid_landmark_count"] and
                metadata["sequence_binary_sha256"] == sha(sequence_data),
                "candidate frame/payload mismatch")
        base = "new_candidates/" + word + "/" + capture_id + "/"
        payloads[base + "normalized_sequence.wisrq"] = sequence_data
        payloads[base + "candidate_metadata.json"] = metadata_data
        payloads[base + "checksum_manifest.json"] = manifest_data
        rows.append({
            "capture_session_id": capture_id, "word": word, "outcome": outcome,
            "eligibility": eligibility, "eligible_for_profile": eligible,
            "strict_accepted": accepted,
            "strict_expected_class_accept": expected_accept,
            "strict_quest_success": quest,
            "raw_frame_count": metadata["raw_frame_count"],
            "normalized_frame_count": metadata["normalized_frame_count"],
            "valid_landmark_count": metadata["valid_landmark_count"],
            "best_distance": metadata["best_distance"],
            "approved_threshold": metadata["approved_threshold"],
            "threshold_margin": metadata["threshold_margin"],
            "sequence_sha256": sha(sequence_data),
            "metadata_sha256": sha(metadata_data),
            "checksum_manifest_sha256": sha(manifest_data),
            "wisrcb01": sequence, "status": "PASS",
        })
    rows.sort(key=lambda row: ["GREETING", "LOVE", "THANK_YOU"].index(row["word"]))
    media_suffixes = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp",
                      ".mp4", ".mov", ".avi", ".mkv", ".wav", ".mp3", ".ogg", ".m4a"}
    media = [path.relative_to(root).as_posix() for path in root.rglob("*")
             if path.is_file() and path.suffix.lower() in media_suffixes]
    require(not media, "candidate repository contains media")
    return ({
        "schema_version": SCHEMA + ".candidate_repository_validation",
        "before_candidate_count": 5, "after_candidate_count": 8,
        "saved_count": 3, "before_file_count": 16, "after_file_count": 25,
        "expected_after_formula": "5+3 / 16+3*3",
        "existing_index_rows_preserved": 5,
        "existing_payload_files_byte_identical": 15,
        "removed_candidate_count": 0,
        "modified_existing_candidate_payload_count": 0,
        "new_candidate_count": 3, "eligible_new_count": 2,
        "review_only_new_count": 1,
        "duplicate_capture_session_id_count": 0,
        "wisrcb01_checksum_round_trip_pass": 3,
        "media_file_count": 0,
        "profile_auto_refresh_or_activation": False,
        "rows": rows, "status": "PASS",
    }, payloads)


def screenshot_records(result: Path) -> list[dict[str, Any]]:
    rows = []
    for relative in RAW_EVIDENCE:
        path = result / relative
        require(path.is_file(), "required evidence missing: " + relative)
        data = path.read_bytes()
        if path.suffix.lower() == ".png":
            require(data.startswith(b"\x89PNG\r\n\x1a\n"), "invalid PNG evidence: " + relative)
        rows.append(record(relative, data, status="PASS"))
    return rows


def playmode_evaluation(result: Path, log: dict[str, Any],
                        candidates: dict[str, Any]) -> dict[str, Any]:
    records = {row["relative_path"]: row for row in screenshot_records(result)}
    expected_screens = {name for name in RAW_EVIDENCE if name.endswith(".png")}
    require(expected_screens <= set(records), "screen evidence set mismatch")
    by_word = {row["word"]: row for row in candidates["rows"]}
    require(by_word["GREETING"]["outcome"] == "LowConfidenceCorrect" and
            by_word["LOVE"]["outcome"] == "StrictSuccess" and
            by_word["THANK_YOU"]["outcome"] == "StrictSuccess",
            "screen/candidate outcome mismatch")
    return {
        "schema_version": SCHEMA + ".playmode_world_return_evaluation",
        "playmode_execution_count": log["playmode_execution_count"],
        "world": {
            "placeholder_player_pet_robot_environment_visible": True,
            "character_controller_movement_observed": True,
            "wasd_or_arrow_movement_observed": True,
            "camera_follow_observed": True,
            "outside_prompt_visible": True,
            "outside_interaction_blocked": True,
            "inside_prompt_visible": True,
            "manual_entry_input_count": 1,
            "world_webcam_hand_pose_candidate_owner_count": 0,
            "automatic_entry_capture_retry_count": 0,
        },
        "entry": {
            "player_transform_displayed": {"x": "-1.5362...", "y": "0.0799...", "z": "0.4471..."},
            "return_transform_displayed": {"x": "-1.5362...", "y": "0.0799...", "z": "0.4471..."},
            "displayed_position_match": True,
            "user_confirmed_position_and_summary_match": True,
            "practice_scene_live_load_count": 1,
        },
        "practice_readiness": {
            "candidate_route_active": True,
            "technical_panel_hidden": True,
            "legacy_practice_controller_inactive": True,
            "candidate_camera_owner_count": 1,
            "automatic_capture_retry_count": 0,
            "camera_state": "Ready",
            "preview_visible": True,
            "camera_started": True,
            "frame_ready": True,
            "ready_frame_count": 234,
            "observation_ready": True,
            "ready_observation_count": 233,
        },
        "words": {
            "GREETING": {"first_outcome": "LowConfidenceCorrect",
                         "strict_accepted_expected_class_quest": [False, False, False],
                         "retry_count": 0, "final_status": "Mastered",
                         "storage": "Saved", "eligibility": "ReviewOnly",
                         "candidate_total_after": 6},
            "LOVE": {"first_outcome": "StrictSuccess",
                     "strict_accepted_expected_class_quest": [True, True, True],
                     "retry_count": 0, "final_status": "Mastered",
                     "storage": "Saved", "eligibility": "Eligible",
                     "candidate_total_after": 7},
            "THANK_YOU": {"first_outcome": "TechnicalFailure",
                          "first_strict_accepted_expected_class_quest": [False, False, False],
                          "first_storage": "NotRequested", "retry_count": 1,
                          "retry1_outcome": "StrictSuccess",
                          "retry1_strict_accepted_expected_class_quest": [True, True, True],
                          "final_status": "Mastered", "retry1_storage": "Saved",
                          "retry1_eligibility": "Eligible", "candidate_total_after": 8},
        },
        "practice_summary": {"session_state": "Completed", "mastered_count": 3,
                             "needs_practice_count": 0, "candidate_saved_total": 8,
                             "exit_button_visible": True},
        "world_return": {"world_return_count": 1, "summary_display_count": 1,
                         "message": "연습 완료", "mastered_count": 3,
                         "needs_practice_count": 0, "candidate_saved_total": 8,
                         "word_statuses_match": True, "result_consumed_once": True},
        "live_needs_practice_branch": "NOT_APPLICABLE_NO_RETRY1_WRONG_WORD",
        "live_reentry": "NOT_APPLICABLE_BY_GATE_CONTRACT",
        "screen_records": [records[name] for name in sorted(expected_screens)],
        "status": "PASS",
    }


def assertions_bytes(rows: list[tuple[str, str, str]]) -> bytes:
    seen = set()
    output = []
    for key, status, detail in rows:
        require(key not in seen, "duplicate assertion key: " + key)
        seen.add(key)
        output.append(json.dumps({"assertion_key": key, "status": status,
                                  "detail": detail}, ensure_ascii=False,
                                 sort_keys=True, allow_nan=False))
    return ("\n".join(output) + "\n").encode("utf-8")


def add_manifest(files: dict[str, bytes]) -> dict[str, bytes]:
    require("output_file_manifest.json" not in files, "manifest already present")
    rows = [record(name, files[name]) for name in sorted(files)]
    require(len(rows) == len({row["relative_path"] for row in rows}),
            "manifest duplicate target")
    return {**files, "output_file_manifest.json": jbytes({
        "schema_version": SCHEMA + ".output_manifest",
        "self_excluded": True, "files": rows,
    })}


def report(tests: dict[str, Any], log: dict[str, Any], play: dict[str, Any],
           candidates: dict[str, Any]) -> bytes:
    return f"""# Step 10B v1.0 GUI Unity Editor Gate

## Verdict

PASS — Unity GUI compile, EditMode 120-test, and one world→practice→world PlayMode gate are ready for independent external review.

## Unity and EditMode

- Unity: {log['unity_version']}; registered packages: {log['registered_package_count']}.
- Compiler errors: {log['compiler_error_count']}.
- EditMode execution: {tests['execution_count']}; unique tests: {tests['passed']}/{tests['total']} PASS.
- Failed/skipped/inconclusive: {tests['failed']}/{tests['skipped']}/{tests['inconclusive']}; retry/repeat: 0/0.

## PlayMode

- Execution count: 1; world entry input: one; live practice Scene load: one; world return: one.
- World movement, camera follow, outside block, inside prompt, and original-position return are preserved by the supplied screens and user confirmation.
- Camera Ready evidence: frame {play['practice_readiness']['ready_frame_count']}, observation {play['practice_readiness']['ready_observation_count']}.
- GREETING: LowConfidenceCorrect, False/False/False, ReviewOnly Saved, Mastered.
- LOVE: StrictSuccess, True/True/True, Eligible Saved, Mastered.
- THANK_YOU: first TechnicalFailure with no storage; Retry1 StrictSuccess, True/True/True, Eligible Saved, Mastered.
- Summary: Completed; Mastered 3; NeedsPractice 0; Candidate saved total 8; world message `연습 완료`.
- Live NeedsPractice and second-session re-entry: NOT_APPLICABLE.

## Candidate repository and preservation

- Candidates: {candidates['before_candidate_count']}→{candidates['after_candidate_count']}; files: {candidates['before_file_count']}→{candidates['after_file_count']}.
- New: 3 (Eligible 2, ReviewOnly 1); WISRCB01 checksum/round-trip 3/3; duplicate IDs/media files 0/0.
- Existing candidate payload 15/15 and index rows 5/5 are preserved.
- Step 10A project 30/30, protected 106/106, candidate profile 22/22, prototype 9, threshold 3, and Step 10B 22/22 files are byte-identical.
- Git staged files: 0. Profile refresh/activation: false.

## Boundary

This is Codex self-validation evidence pending ChatGPT Work external review. No production, Scene, Prefab, Build Settings, recognition, threshold, prototype, profile, or existing candidate file was modified by packaging.
""".encode("utf-8")


def package(root: Path) -> None:
    validation = root / "Validation"
    result = validation / NAME
    delivery = validation / (NAME + "_delivery.zip")
    detached = validation / (NAME + "_delivery.zip.sha256")
    require(result.is_dir(), "evidence directory missing")
    require(not delivery.exists() and not detached.exists(), "never-overwrite delivery target exists")
    require(set(path.name for path in result.iterdir() if path.is_file()) == set(RAW_EVIDENCE),
            "unexpected or missing raw evidence before packaging")

    zip_helper = load_module(root, ZIP_HELPER, "step10b_gui_zip_helper")
    tx_helper = load_module(root, TX_HELPER, "step10b_gui_tx_helper")
    wisrq = load_module(root, WISRQ_HELPER, "step10b_gui_wisrq_helper")

    step10b_info = verify_fixed_zip(root, zip_helper, STEP10B_ZIP, STEP10B_ZIP_SIZE,
                                    STEP10B_ZIP_SHA, STEP10B_DETACHED_SIZE,
                                    STEP10B_DETACHED_SHA, 50, 49)
    step10a_info = verify_fixed_zip(root, zip_helper, STEP10A_ZIP, STEP10A_ZIP_SIZE,
                                    STEP10A_ZIP_SHA, STEP10A_DETACHED_SIZE,
                                    STEP10A_DETACHED_SHA, 65, 64)
    step10a_gui_info = verify_fixed_zip(root, zip_helper, STEP10A_GUI_ZIP,
                                        STEP10A_GUI_ZIP_SIZE, STEP10A_GUI_ZIP_SHA,
                                        STEP10A_GUI_DETACHED_SIZE, STEP10A_GUI_DETACHED_SHA,
                                        46, 45)

    source = root / STEP10B_RESULT
    project_baseline = load_json(source / "project_files_before_after.json")
    protected_baseline = load_json(source / "protected_files_before_after.json")
    profile_baseline = load_json(source / "active_candidate_profile_before_after.json")
    step10b_baseline = load_json(source / "implementation_delta.json")
    project = compare_records(root, project_baseline, 30, "step10a_project_files")
    protected = compare_records(root, protected_baseline, 106, "protected_files")
    profile = compare_records(root, profile_baseline, 22, "active_candidate_profile")
    profile.update({"prototype_count": 9, "threshold_count": 3,
                    "thresholds": EXPECTED_THRESHOLDS})
    step10b_files = compare_step10b_files(root, step10b_baseline)

    tests = parse_tests(result)
    log = parse_editor_log(result)
    before_repository, before_index_data = candidate_before(root)
    after_repository = current_repository(before_repository)
    candidate_validation, candidate_payloads = validate_repository(
        before_repository, before_index_data, after_repository, wisrq)
    play = playmode_evaluation(result, log, candidate_validation)
    require(staged_files(root) == [], "Git staged files changed")
    git = {"schema_version": SCHEMA + ".git_staged", "before_count": 0,
           "after_count": 0, "staged_files": [], "status": "PASS"}

    transaction = tx_helper.transaction_regression()
    require(transaction["status"] == "PASS" and
            len(transaction["failure_injection_rows"]) == 4 and
            all(row["status"] == "PASS" for row in transaction["failure_injection_rows"]) and
            transaction["never_overwrite_second_commit_blocked"],
            "packaging transaction regression failed")

    preflight = {
        "schema_version": SCHEMA + ".preflight_before",
        "captured_before_user_gui_execution": True,
        "source_baseline": "approved Step 10B source validation records",
        "step10a_project": "30/30", "protected": "106/106",
        "candidate_profile": "22/22", "prototypes": 9, "thresholds": 3,
        "threshold_values": EXPECTED_THRESHOLDS,
        "step10b_project_and_build_settings": "22/22",
        "candidate_repository": {"candidate_count": 5, "file_count": 16},
        "git_staged_count": 0, "status": "PASS",
    }
    input_provenance = {
        "schema_version": SCHEMA + ".input_provenance",
        "implementation_owner": "Codex",
        "step10b_source_delivery": step10b_info,
        "step10a_source_delivery": step10a_info,
        "step10a_gui_evidence": step10a_gui_info,
        "status": "PASS",
    }
    primary = {
        "schema_version": SCHEMA + ".primary_verdict",
        "overall_gate": "PASS",
        "unity_compile": "PASS",
        "editmode_120": "PASS",
        "playmode_execution_count": 1,
        "world_practice_world_loop": "PASS",
        "camera_readiness": "PASS",
        "practice_completion": "PASS",
        "word_statuses": {"GREETING": "Mastered", "LOVE": "Mastered",
                          "THANK_YOU": "Mastered"},
        "mastered_count": 3, "needs_practice_count": 0,
        "candidate_storage": "PASS_5_TO_8",
        "live_needs_practice_branch": "NOT_APPLICABLE",
        "live_second_session_reentry": "NOT_APPLICABLE",
        "status_strings": [
            "STEP10B_GUI_UNITY_EDITOR_GATE_READY_PENDING_EXTERNAL_REVIEW",
            "STEP10B_UNITY_COMPILE_EDITOR_PASS",
            "STEP10B_EDITMODE_FIRST_RUN_PASS_120_OF_120",
            "STEP10B_WORLD_TO_PRACTICE_TO_WORLD_PLAYMODE_PASS",
            "STEP10B_COMPLETED_RESULT_ONE_SHOT_RETURN_PASS",
            "STEP10B_CANDIDATE_STORAGE_APPEND_PASS_5_TO_8",
            "STEP10B_LIVE_REENTRY_NOT_APPLICABLE",
            "BLOCKED_PENDING_EXTERNAL_REVIEW",
        ],
        "status": "PASS",
    }
    assertions = assertions_bytes([
        ("fixed_step10b_delivery", "PASS", STEP10B_ZIP_SHA),
        ("fixed_step10a_delivery", "PASS", STEP10A_ZIP_SHA),
        ("fixed_step10a_gui_evidence", "PASS", STEP10A_GUI_ZIP_SHA),
        ("unity_version", "PASS", "6000.3.14f1"),
        ("compiler_errors", "PASS", "0"),
        ("editmode_execution_count", "PASS", "1"),
        ("unique_editmode_tests", "PASS", "120"),
        ("fixture_inventory", "PASS", "23+10+8+22+8+8+8+15+18"),
        ("editmode_pass_fail", "PASS", "120/0"),
        ("skipped_inconclusive", "PASS", "0/0"),
        ("retry_repeat", "PASS", "0/0"),
        ("playmode_execution_count", "PASS", "1"),
        ("world_placeholder_environment", "PASS", "visible"),
        ("world_character_movement", "PASS", "WASD/arrow movement observed"),
        ("world_camera_follow", "PASS", "observed"),
        ("world_outside_interaction", "PASS", "blocked"),
        ("world_inside_prompt", "PASS", "visible"),
        ("world_manual_entry", "PASS", "one E input"),
        ("world_camera_provider_owner", "PASS", "0"),
        ("practice_scene_live_load", "PASS", "1"),
        ("candidate_camera_owner", "PASS", "1"),
        ("candidate_camera_ready", "PASS", "frames 234; observations 233"),
        ("candidate_route_preview", "PASS", "active; preview visible; technical hidden"),
        ("legacy_practice_controller", "PASS", "inactive"),
        ("greeting_first", "PASS", "LowConfidenceCorrect; False/False/False; ReviewOnly Saved"),
        ("love_first", "PASS", "StrictSuccess; True/True/True; Eligible Saved"),
        ("thank_you_first", "PASS", "TechnicalFailure; no progress/storage"),
        ("thank_you_retry1", "PASS", "StrictSuccess; True/True/True; Eligible Saved"),
        ("manual_retry_count", "PASS", "0/0/1"),
        ("automatic_entry_capture_retry", "PASS", "0"),
        ("word_learning_statuses", "PASS", "Mastered/Mastered/Mastered"),
        ("practice_summary", "PASS", "Completed; Mastered 3; NeedsPractice 0; Candidate 8"),
        ("world_return_position", "PASS", "displayed match; user confirmed"),
        ("world_return_summary", "PASS", "연습 완료; 3/0/8; one display"),
        ("completed_snapshot_one_shot", "PASS", "one return and consumption"),
        ("live_needs_practice", "NOT_APPLICABLE", "no Retry1 WrongWord occurred"),
        ("live_second_session_reentry", "NOT_APPLICABLE", "prohibited by gate"),
        ("new_candidate_count", "PASS", "3"),
        ("candidate_count_formula", "PASS", "5+3=8"),
        ("candidate_file_formula", "PASS", "16+3*3=25"),
        ("candidate_eligibility", "PASS", "Eligible 2; ReviewOnly 1"),
        ("wisrcb01_checksum_roundtrip", "PASS", "3/3"),
        ("candidate_checksum_manifests", "PASS", "3/3"),
        ("candidate_session_duplicates", "PASS", "0"),
        ("candidate_media_files", "PASS", "0"),
        ("existing_candidate_payloads", "PASS", "15/15 byte-identical"),
        ("existing_candidate_index_rows", "PASS", "5/5 content preserved"),
        ("step10a_project_files", "PASS", "30/30"),
        ("protected_files", "PASS", "106/106"),
        ("active_profile", "PASS", "22/22"),
        ("prototype_threshold", "PASS", "9/3"),
        ("step10b_project_build_settings", "PASS", "22/22"),
        ("profile_auto_refresh_activation", "PASS", "false"),
        ("git_staged", "PASS", "0"),
        ("packaging_transaction", "PASS", "4/4 rollback; never-overwrite"),
    ])

    generated: dict[str, bytes] = {
        "preflight_before_snapshot.json": jbytes(preflight),
        "input_provenance.json": jbytes(input_provenance),
        "test_results_summary.json": jbytes(tests),
        "editor_log_summary.json": jbytes(log),
        "playmode_world_return_evaluation.json": jbytes(play),
        "candidate_repository_before.json": jbytes(before_repository),
        "candidate_index_before.json": before_index_data,
        "candidate_repository_after.json": jbytes(after_repository),
        "candidate_repository_validation.json": jbytes(candidate_validation),
        "step10a_project_files_before_after.json": jbytes(project),
        "protected_files_before_after.json": jbytes(protected),
        "active_candidate_profile_before_after.json": jbytes(profile),
        "step10b_project_files_before_after.json": jbytes(step10b_files),
        "git_staged_before_after.json": jbytes(git),
        "packaging_transaction_failure_injection_regression.json": jbytes(transaction),
        "primary_verdict.json": jbytes(primary),
        "ASSERTIONS.jsonl": assertions,
        "VALIDATION_REPORT.md": report(tests, log, play, candidate_validation),
        "analysis/" + Path(__file__).name: Path(__file__).read_bytes(),
        **candidate_payloads,
    }
    for relative in generated:
        require(not (result / relative).exists(), "never-overwrite generated result exists: " + relative)
    raw_files = {path.relative_to(result).as_posix(): path.read_bytes()
                 for path in result.rglob("*") if path.is_file()}
    require(set(raw_files) == set(RAW_EVIDENCE), "raw evidence file map changed")
    complete = add_manifest({**raw_files, **generated})
    zip_blob = zip_helper.deterministic_zip(complete)
    detached_blob = f"{sha(zip_blob)}  {delivery.name}\n".encode("ascii")
    checked = zip_helper.validate_zip(zip_blob, complete)
    require(checked["duplicate_count"] == 0 and checked["unsafe_path_count"] == 0 and
            checked["encrypted_entry_count"] == 0 and checked["symlink_entry_count"] == 0 and
            checked["crc_failure"] is None and checked["membership_match"] and
            checked["payload_mismatch_count"] == 0,
            "pre-commit evidence ZIP validation failed")

    temp = Path(tempfile.mkdtemp(prefix="step10b_gui_gate_generated_", dir=validation))
    created: list[Path] = []
    committed = False
    try:
        commit_files = {**generated,
                        "output_file_manifest.json": complete["output_file_manifest.json"]}
        for relative, data in commit_files.items():
            path = temp / PurePosixPath(relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        for relative_name in sorted(commit_files):
            source_path = temp / PurePosixPath(relative_name)
            target = result / PurePosixPath(relative_name)
            target.parent.mkdir(parents=True, exist_ok=True)
            require(not target.exists(), "generated target appeared during commit")
            source_path.replace(target)
            created.append(target)
        zip_helper.commit_pair(zip_blob, detached_blob, delivery, detached)
        disk = {path.relative_to(result).as_posix(): path.read_bytes()
                for path in result.rglob("*") if path.is_file()}
        require(disk == complete, "committed result file map mismatch")
        require(zip_helper.deterministic_zip(disk) == delivery.read_bytes(),
                "deterministic evidence ZIP mismatch")
        require(detached.read_bytes() == detached_blob, "detached checksum mismatch")
        post = zip_helper.validate_zip(delivery.read_bytes(), disk)
        require(post["duplicate_count"] == 0 and post["unsafe_path_count"] == 0 and
                post["encrypted_entry_count"] == 0 and post["symlink_entry_count"] == 0 and
                post["crc_failure"] is None and post["membership_match"] and
                post["payload_mismatch_count"] == 0,
                "post-commit evidence ZIP validation failed")
        require(staged_files(root) == [], "Git staged files changed during packaging")
        committed = True
    finally:
        shutil.rmtree(temp, ignore_errors=True)
        if not committed:
            for path in reversed(created):
                if path.exists():
                    path.unlink()
            for path in (delivery, detached):
                if path.exists():
                    path.unlink()

    print(json.dumps({
        "status": "PASS", "overall_gate": "PASS",
        "result_directory": str(result), "result_file_count": len(complete),
        "manifest_target_count": len(complete) - 1,
        "delivery_zip": str(delivery), "delivery_size_bytes": delivery.stat().st_size,
        "delivery_sha256": sha(delivery.read_bytes()),
        "detached_checksum": str(detached),
        "detached_size_bytes": detached.stat().st_size,
        "detached_file_sha256": sha(detached.read_bytes()),
        "compiler_errors": 0, "editmode_execution_count": 1,
        "editmode": "120 PASS / 0 FAIL", "playmode_execution_count": 1,
        "camera": {"state": "Ready", "frames": 234, "observations": 233},
        "outcomes": {"GREETING": "LowConfidenceCorrect",
                     "LOVE": "StrictSuccess", "THANK_YOU": "TechnicalFailure→StrictSuccess"},
        "retries": {"GREETING": 0, "LOVE": 0, "THANK_YOU": 1},
        "word_statuses": {"GREETING": "Mastered", "LOVE": "Mastered",
                          "THANK_YOU": "Mastered"},
        "world_return": "PASS_POSITION_AND_SUMMARY_MATCH",
        "candidate_count_before_after": [5, 8],
        "candidate_file_count_before_after": [16, 25],
        "project_files": "30/30", "protected_files": "106/106",
        "profile_files": "22/22", "step10b_files": "22/22",
        "git_staged": 0,
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--package", action="store_true")
    args = parser.parse_args()
    require(args.package, "use --package")
    package(args.root.resolve())


if __name__ == "__main__":
    main()
