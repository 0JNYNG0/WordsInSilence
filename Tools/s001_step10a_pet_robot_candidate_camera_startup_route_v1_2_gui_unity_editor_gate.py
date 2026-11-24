#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import shutil
import stat
import struct
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA = "WordsInSilence.Step10A.CandidateCameraStartupRoute.v1.2.GuiGate"
NAME = "WordsInSilence_step10a_pet_robot_candidate_camera_startup_route_v1_2_gui_unity_editor_gate"
V12_ZIP = "Validation/WordsInSilence_step10a_pet_robot_candidate_camera_startup_route_v1_2_delivery.zip"
V12_SIZE = 73741
V12_SHA = "edea92e19bac40ce88b8eb0c54c62c1f6e66a2699fb1d9c48e016eb4421a5799"
V12_DETACHED_FILE_SHA = "1f64f5b54dd8346f1a39ef306c87ca310c6d82411ddcfe671f84fcdaddd9074b"
V11_ZIP = "Validation/WordsInSilence_step10a_pet_robot_candidate_camera_startup_route_v1_1_delivery.zip"
V11_SIZE = 108034
V11_SHA = "f5ccb3438728d8d96c4bcff68c2fcccccc23d1afc92140d0f12d8294af773518"
V11_DETACHED_FILE_SHA = "30a89d3af38be88ef5052dcc86a7933f85c6b690f148d8c30b4d7e56822a1d02"
FIRST_GATE_ZIP = "Validation/WordsInSilence_step10a_pet_robot_candidate_camera_startup_route_v1_1_gui_unity_editor_gate_delivery.zip"
FIRST_GATE_SIZE = 393403
FIRST_GATE_SHA = "c140d366b40cbd822bb22d7691050d84e08f806094c4eb2a52a2d793a421708d"
FIRST_GATE_DETACHED_FILE_SHA = "02702e9c386b4fa46217a6cf772b741c0ff2182171bf5f1f6a433965f4e4260a"
V11_TOOL = "Tools/s001_step10a_pet_robot_candidate_camera_startup_route_v1_1.py"
V11_GATE_TOOL = "Tools/s001_step10a_pet_robot_candidate_camera_startup_route_v1_1_gui_unity_editor_gate.py"
CANDIDATE_ROOT = Path(
    r"C:\Users\local-user\AppData\LocalLow\DefaultCompany\WordsInSilence\WordsInSilence"
    r"\PersonalizationCandidates\v1\local_learner"
)
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
}


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


def load_v11(root: Path):
    path = root / V11_TOOL
    spec = importlib.util.spec_from_file_location("step10a_v11_approved", path)
    require(spec is not None and spec.loader is not None, "approved v1.1 tool cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_v11_gate(root: Path):
    path = root / V11_GATE_TOOL
    spec = importlib.util.spec_from_file_location("step10a_v11_gate_approved", path)
    require(spec is not None and spec.loader is not None, "approved v1.1 gate tool cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_fixed(v11: Any, root: Path, relative_path: str, size: int,
                 digest: str, detached_file_digest: str) -> tuple[dict[str, Any], dict[str, bytes]]:
    return v11.verify_fixed_zip(root, relative_path, size, digest, detached_file_digest)


def verify_current_set(root: Path, baseline: dict[str, Any], expected_count: int,
                       label: str) -> dict[str, Any]:
    rows = []
    for expected in baseline["records"]:
        relative = expected["relative_path"]
        path = root / relative
        require(path.is_file(), f"{label} missing: {relative}")
        data = path.read_bytes()
        expected_size = expected["size_bytes"]
        expected_sha = expected["sha256"]
        ok = len(data) == expected_size and sha(data) == expected_sha
        rows.append(record(relative, data, expected_size_bytes=expected_size,
                           expected_sha256=expected_sha, status="PASS" if ok else "FAIL"))
    require(len(rows) == expected_count, f"{label} count mismatch")
    require(all(row["status"] == "PASS" for row in rows), f"{label} identity mismatch")
    return {
        "schema_version": SCHEMA + "." + label,
        "file_count": len(rows),
        "pass_count": len(rows),
        "mismatch_count": 0,
        "records": rows,
        "status": "PASS",
    }


def candidate_count(index: Any) -> int:
    if isinstance(index, list):
        return len(index)
    if not isinstance(index, dict):
        return 0
    for key in ("candidates", "entries", "records", "items"):
        if isinstance(index.get(key), list):
            return len(index[key])
    for key in ("candidate_count", "count", "total_count"):
        if isinstance(index.get(key), int):
            return index[key]
    per_word = index.get("counts_by_word") or index.get("per_word")
    if isinstance(per_word, dict) and all(isinstance(value, int) for value in per_word.values()):
        return sum(per_word.values())
    return 0


def snapshot_repository() -> dict[str, Any]:
    rows = []
    if CANDIDATE_ROOT.is_dir():
        for path in sorted(value for value in CANDIDATE_ROOT.rglob("*") if value.is_file()):
            rows.append(record(path.relative_to(CANDIDATE_ROOT).as_posix(), path.read_bytes()))
    index_path = CANDIDATE_ROOT / "candidate_index.json"
    count = 0
    if index_path.is_file():
        count = candidate_count(json.loads(index_path.read_text(encoding="utf-8-sig")))
    return {
        "schema_version": SCHEMA + ".candidate_repository",
        "path": str(CANDIDATE_ROOT),
        "exists": CANDIDATE_ROOT.exists(),
        "file_count": len(rows),
        "candidate_count": count,
        "files": rows,
        "status": "PASS",
    }


def staged_files(root: Path) -> list[str]:
    run = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=root,
                         capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    require(run.returncode == 0, "Git staged query failed")
    return [line.strip() for line in run.stdout.splitlines() if line.strip()]


def prepare(root: Path) -> None:
    validation = root / "Validation"
    final = validation / NAME
    require(not final.exists(), "never-overwrite gate directory already exists")
    v11 = load_v11(root)
    v12_info, v12_files = verify_fixed(
        v11, root, V12_ZIP, V12_SIZE, V12_SHA, V12_DETACHED_FILE_SHA)
    v11_info, _ = verify_fixed(
        v11, root, V11_ZIP, V11_SIZE, V11_SHA, V11_DETACHED_FILE_SHA)
    first_info, first_files = verify_fixed(
        v11, root, FIRST_GATE_ZIP, FIRST_GATE_SIZE, FIRST_GATE_SHA,
        FIRST_GATE_DETACHED_FILE_SHA)
    first_summary = json.loads(first_files["test_results_summary.json"])
    require(first_summary["unique_testcase_count"] == 87 and
            first_summary["passed"] == 86 and first_summary["failed"] == 1,
            "fixed first GUI failure identity mismatch")

    project_baseline = json.loads(v12_files["project_files_after.json"])
    protected_baseline = json.loads(v12_files["protected_files_after.json"])
    profile_baseline = json.loads(v12_files["active_candidate_profile_after.json"])
    project = verify_current_set(root, project_baseline, 30, "project_files_before")
    protected = verify_current_set(root, protected_baseline, 106, "protected_files_before")
    profile = verify_current_set(root, profile_baseline, 22, "active_candidate_profile_before")
    require(profile_baseline.get("prototype_count", 9) == 9, "prototype count mismatch")
    thresholds = profile_baseline.get("thresholds", EXPECTED_THRESHOLDS)
    require(thresholds == EXPECTED_THRESHOLDS, "threshold map mismatch")
    repository = snapshot_repository()
    staged = staged_files(root)
    require(staged == [], "Git staged files must be zero before GUI gate")

    context = (root / "WordsInSilence_PROJECT_CONTEXT.md").read_bytes()
    files = {
        "fixed_input_preflight.json": jbytes({
            "schema_version": SCHEMA + ".fixed_inputs",
            "v1_2_delivery": v12_info,
            "v1_1_delivery": v11_info,
            "first_gui_failure_delivery": first_info,
            "first_gui_failure_counts": {
                "execution_count": first_summary["execution_count"],
                "unique_tests": first_summary["unique_testcase_count"],
                "passed": first_summary["passed"],
                "failed": first_summary["failed"],
                "retry_nonzero": first_summary["retry_nonzero"],
                "repeat_nonzero": first_summary["repeat_nonzero"],
            },
            "status": "PASS",
        }),
        "project_files_before.json": jbytes(project),
        "protected_files_before.json": jbytes(protected),
        "active_candidate_profile_before.json": jbytes({
            **profile,
            "profile_id": "step9l_current_user_multi_prototype_v1_0",
            "prototype_count": 9,
            "threshold_count": 3,
            "thresholds": EXPECTED_THRESHOLDS,
        }),
        "candidate_repository_before.json": jbytes(repository),
        "git_staged_before.json": jbytes({
            "schema_version": SCHEMA + ".git_before",
            "staged_count": 0,
            "staged_files": staged,
            "status": "PASS",
        }),
        "preflight_before_snapshot.json": jbytes({
            "schema_version": SCHEMA + ".preflight",
            "implementation_owner": "Codex",
            "unity_version_required": "6000.3.14f1",
            "project_files": "30/30 PASS",
            "protected_files": "106/106 PASS",
            "candidate_profile": "22/22 PASS",
            "prototype_count": 9,
            "threshold_count": 3,
            "candidate_repository_file_count": repository["file_count"],
            "candidate_count": repository["candidate_count"],
            "git_staged_count": 0,
            "context": record("WordsInSilence_PROJECT_CONTEXT.md", context),
            "batchmode_used": False,
            "status": "PASS",
        }),
        "EVIDENCE_INSTRUCTIONS.txt": (
            "Do not rename or modify the preflight JSON files.\n"
            "Add exactly one same-run TestResults.xml, one full Test Runner PNG, and one full Editor.log.\n"
            "After 87/87 PASS only, add Camera Ready/Failed and conditional per-word/Completed screenshots.\n"
        ).encode("utf-8"),
    }

    token = os.urandom(8).hex()
    temp = validation / ("." + NAME + "-prepare-" + token)
    committed = False
    try:
        temp.mkdir(parents=True)
        for relative, data in files.items():
            path = temp / PurePosixPath(relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        temp.replace(final)
        committed = True
    finally:
        if not committed and temp.exists():
            shutil.rmtree(temp)

    print(json.dumps({
        "status": "PASS",
        "gate_directory": str(final),
        "preflight_file_count": len(files),
        "project_files": project["pass_count"],
        "protected_files": protected["pass_count"],
        "candidate_profile_files": profile["pass_count"],
        "prototype_count": 9,
        "threshold_count": 3,
        "candidate_repository_file_count": repository["file_count"],
        "candidate_count": repository["candidate_count"],
        "git_staged_count": 0,
        "next_action": "USER_GUI_SINGLE_RUN",
    }, ensure_ascii=False, indent=2))


def property_value(case: ET.Element, name: str) -> str:
    for item in case.findall("./properties/property"):
        if item.attrib.get("name") == name:
            return item.attrib.get("value", "")
    return ""


def parse_test_results(result: Path) -> dict[str, Any]:
    path = result / "TestResults.xml"
    require(path.is_file(), "TestResults.xml missing")
    root = ET.parse(path).getroot()
    cases = root.findall(".//test-case")
    selected = [
        case for case in cases
        if case.attrib.get("classname", "").split(".")[-1] in EXPECTED_FIXTURES
    ]
    names = [case.attrib.get("fullname", "") for case in selected]
    counts = Counter(case.attrib.get("result", "") for case in selected)
    fixtures = []
    for fixture, expected in EXPECTED_FIXTURES.items():
        rows = [case for case in selected if case.attrib.get("classname", "").endswith("." + fixture)]
        fixtures.append({
            "fixture": fixture,
            "expected": expected,
            "actual": len(rows),
            "passed": sum(case.attrib.get("result") == "Passed" for case in rows),
            "failed": sum(case.attrib.get("result") == "Failed" for case in rows),
            "skipped": sum(case.attrib.get("result") == "Skipped" for case in rows),
            "inconclusive": sum(case.attrib.get("result") == "Inconclusive" for case in rows),
        })
    retries = sum(int(property_value(case, "retryIteration") or "0") != 0 for case in selected)
    repeats = sum(int(property_value(case, "repeatIteration") or "0") != 0 for case in selected)
    ok = (
        len(selected) == len(set(names)) == 87
        and counts["Passed"] == 87
        and counts["Failed"] == 0
        and counts["Skipped"] == 0
        and counts["Inconclusive"] == 0
        and retries == 0
        and repeats == 0
        and all(row["actual"] == row["expected"] == row["passed"] for row in fixtures)
    )
    require(ok, "Unity 87-test result contract failed")
    return {
        "schema_version": SCHEMA + ".test_results",
        "status": "PASS",
        "platform": "EditMode",
        "execution_count": 1,
        "testcase_count": len(selected),
        "unique_testcase_count": len(set(names)),
        "passed": counts["Passed"],
        "failed": counts["Failed"],
        "skipped": counts["Skipped"],
        "inconclusive": counts["Inconclusive"],
        "retry_nonzero": retries,
        "repeat_nonzero": repeats,
        "start_time_utc": root.attrib.get("start-time"),
        "end_time_utc": root.attrib.get("end-time"),
        "duration_seconds": root.attrib.get("duration"),
        "fixtures": fixtures,
        "xml": record("TestResults.xml", path.read_bytes()),
        "screen": record("TestRunner_FIRST_RESULT.png",
                         (result / "TestRunner_FIRST_RESULT.png").read_bytes()),
    }


def parse_editor_log(result: Path) -> dict[str, Any]:
    path = result / "Editor.log"
    require(path.is_file(), "Editor.log missing")
    data = path.read_bytes()
    text = data.decode("utf-8", errors="replace")
    counts = {
        "unity_version": text.count("6000.3.14f1"),
        "batchmode_zero": text.count("BatchMode: 0"),
        "human_controlled": text.count("IsHumanControllingUs: 1"),
        "tundra_build_success": text.count("Tundra build success"),
        "compiler_error_cs": len(re.findall(r"\berror CS\d+", text)),
        "scripts_have_compiler_errors": text.count("Scripts have compiler errors"),
        "compilation_failed": text.count("Compilation failed"),
        "prebuild": text.count("Executing IPrebuildSetup"),
        "postbuild": text.count("Executing IPostBuildCleanup"),
        "default_result_save": text.count(
            "Saving results to: C:/Users/local-user/AppData/LocalLow/DefaultCompany/WordsInSilence\\TestResults.xml"),
        "gate_result_export": text.count("Saving results to: C:/MyUnityProjects/WordsInSilence/Validation/" + NAME + "/TestResults.xml"),
        "playmode_assembly_reload": text.count("Reloading assemblies for play mode."),
        "pose_provider_initialized": text.count("[PoseLandmarkProvider]"),
        "hand_provider_initialized": text.count("[MediaPipeLandmarkProvider]"),
        "null_reference_exception": text.count("NullReferenceException"),
        "invalid_operation_exception": text.count("InvalidOperationException"),
        "unhandled_exception": text.count("Unhandled Exception"),
    }
    compiler_errors = (counts["compiler_error_cs"] + counts["scripts_have_compiler_errors"] +
                       counts["compilation_failed"])
    require(
        counts["unity_version"] > 0
        and counts["batchmode_zero"] == 1
        and counts["human_controlled"] == 1
        and counts["tundra_build_success"] >= 1
        and compiler_errors == 0
        and counts["prebuild"] == 1
        and counts["postbuild"] == 1
        and counts["default_result_save"] == 1
        and counts["gate_result_export"] == 1
        and counts["playmode_assembly_reload"] == 1
        and counts["pose_provider_initialized"] >= 1
        and counts["hand_provider_initialized"] >= 1
        and counts["null_reference_exception"] == 0
        and counts["invalid_operation_exception"] == 0
        and counts["unhandled_exception"] == 0,
        "Editor.log same-run/compile/PlayMode contract failed")
    return {
        "schema_version": SCHEMA + ".editor_log",
        "status": "PASS",
        "unity_version": "6000.3.14f1",
        "compiler_error_count": compiler_errors,
        "editmode_execution_count": 1,
        "playmode_execution_count": 1,
        "counts": counts,
        "same_editor_session": True,
        "log": record("Editor.log", data),
    }


def snapshot_after(root: Path, before: dict[str, Any], schema_suffix: str) -> dict[str, Any]:
    rows = []
    for expected in before["records"]:
        relative = expected["relative_path"]
        path = root / relative
        require(path.is_file(), "after snapshot file missing: " + relative)
        data = path.read_bytes()
        same = len(data) == expected["size_bytes"] and sha(data) == expected["sha256"]
        rows.append(record(relative, data,
                           before_size_bytes=expected["size_bytes"],
                           before_sha256=expected["sha256"],
                           before_after_byte_identical=same,
                           status="PASS" if same else "FAIL"))
    result = {
        "schema_version": SCHEMA + "." + schema_suffix,
        "file_count": len(rows),
        "pass_count": sum(row["status"] == "PASS" for row in rows),
        "mismatch_count": sum(row["status"] != "PASS" for row in rows),
        "records": rows,
    }
    result["status"] = "PASS" if result["mismatch_count"] == 0 else "FAIL"
    return result


def strict_json(data: bytes, label: str) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"non-standard JSON number in {label}: {value}")
    return json.loads(data.decode("utf-8-sig"), parse_constant=reject)


def read_u32(stream: io.BytesIO) -> int:
    data = stream.read(4)
    require(len(data) == 4, "truncated uint32")
    return struct.unpack("<I", data)[0]


def read_u64(stream: io.BytesIO) -> int:
    data = stream.read(8)
    require(len(data) == 8, "truncated uint64")
    return struct.unpack("<Q", data)[0]


def read_string(stream: io.BytesIO) -> str:
    length = read_u32(stream)
    data = stream.read(length)
    require(len(data) == length, "truncated UTF-8 string")
    return data.decode("utf-8")


def write_string(stream: io.BytesIO, value: str) -> None:
    data = value.encode("utf-8")
    stream.write(struct.pack("<I", len(data)))
    stream.write(data)


def parse_wisrcb(data: bytes) -> dict[str, Any]:
    require(len(data) > 32, "WISRCB01 too short")
    payload, embedded = data[:-32], data[-32:]
    require(hashlib.sha256(payload).digest() == embedded, "WISRCB01 embedded checksum mismatch")
    stream = io.BytesIO(payload)
    magic = stream.read(8)
    version = read_u32(stream)
    endian = read_u32(stream)
    landmarks = read_u32(stream)
    frames = read_u32(stream)
    strings = [read_string(stream) for _ in range(6)]
    coordinate_count = read_u64(stream)
    coordinate_bits = [read_u64(stream) for _ in range(coordinate_count)]
    coordinates = [struct.unpack("<d", struct.pack("<Q", bits))[0] for bits in coordinate_bits]
    mask_count = read_u64(stream)
    packed_length = read_u32(stream)
    packed = stream.read(packed_length)
    require(len(packed) == packed_length and stream.tell() == len(payload), "WISRCB01 payload length mismatch")
    mask = [bool(packed[index >> 3] & (1 << (index & 7))) for index in range(mask_count)]
    require(magic == b"WISRCB01" and version == 1 and endian == 0x01020304,
            "WISRCB01 header mismatch")
    require(landmarks == 16 and coordinate_count == frames * landmarks * 3,
            "WISRCB01 coordinate shape mismatch")
    require(mask_count == frames * landmarks and packed_length == (mask_count + 7) // 8,
            "WISRCB01 mask shape mismatch")
    valid = 0
    invalid = 0
    for index, is_valid in enumerate(mask):
        xyz = coordinates[index * 3:index * 3 + 3]
        if is_valid:
            require(all(math.isfinite(value) for value in xyz), "valid coordinate is not finite")
            valid += 1
        else:
            require(all(math.isnan(value) for value in xyz), "invalid coordinate is not NaN")
            invalid += 1
    rebuilt = io.BytesIO()
    rebuilt.write(magic)
    rebuilt.write(struct.pack("<IIII", version, endian, landmarks, frames))
    for value in strings:
        write_string(rebuilt, value)
    rebuilt.write(struct.pack("<Q", coordinate_count))
    for bits in coordinate_bits:
        rebuilt.write(struct.pack("<Q", bits))
    rebuilt.write(struct.pack("<QI", mask_count, packed_length))
    rebuilt.write(packed)
    rebuilt_payload = rebuilt.getvalue()
    rebuilt_data = rebuilt_payload + hashlib.sha256(rebuilt_payload).digest()
    require(rebuilt_data == data, "WISRCB01 writer/reader round-trip mismatch")
    return {
        "magic": magic.decode("ascii"),
        "format_version": version,
        "endian_marker": f"0x{endian:08x}",
        "landmark_count": landmarks,
        "frame_count": frames,
        "sample_id": strings[0],
        "word": strings[1],
        "speed": strings[2],
        "split": strings[3],
        "source_payload_sha256": strings[4],
        "normalized_payload_sha256": strings[5],
        "coordinate_count": coordinate_count,
        "mask_count": mask_count,
        "valid_landmark_count": valid,
        "invalid_landmark_count": invalid,
        "embedded_checksum_sha256": embedded.hex(),
        "writer_reader_round_trip_byte_identical": True,
    }


def validate_candidate_repository(before: dict[str, Any], after: dict[str, Any]) -> tuple[dict[str, Any], dict[str, bytes]]:
    require(before["candidate_count"] == 0 and before["file_count"] == 0,
            "this fixed GUI gate expected an empty before repository")
    require(after["candidate_count"] == 2 and after["file_count"] == 7,
            "candidate repository after count mismatch")
    index_path = CANDIDATE_ROOT / "candidate_index.json"
    index_data = index_path.read_bytes()
    index = strict_json(index_data, "candidate_index.json")
    entries = index.get("entries")
    require(index.get("learner_id") == "local_learner" and isinstance(entries, list) and len(entries) == 2,
            "candidate index schema/count mismatch")
    capture_ids = [entry["capture_session_id"] for entry in entries]
    require(len(capture_ids) == len(set(capture_ids)) == 2, "duplicate capture/session ID")
    require([entry["word"] for entry in entries] == ["GREETING", "LOVE"],
            "candidate index word/order mismatch")
    require(all(entry["outcome"] == "StrictSuccess" and entry["eligibility"] == "Eligible"
                for entry in entries), "candidate eligibility mismatch")

    payloads = {"candidate_repository_after_payload/candidate_index.json": index_data}
    rows = []
    media_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".mp4",
                        ".mov", ".avi", ".mkv", ".wav", ".mp3", ".ogg", ".m4a"}
    media_files = []
    for path in CANDIDATE_ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in media_extensions:
            media_files.append(path.relative_to(CANDIDATE_ROOT).as_posix())
    require(not media_files, "candidate repository contains media files")

    for entry in entries:
        relative = PurePosixPath(entry["relative_directory"])
        require(not relative.is_absolute() and ".." not in relative.parts and "\\" not in str(relative),
                "unsafe candidate relative directory")
        directory = CANDIDATE_ROOT / Path(*relative.parts)
        sequence_path = directory / "normalized_sequence.wisrq"
        metadata_path = directory / "candidate_metadata.json"
        manifest_path = directory / "checksum_manifest.json"
        require(sequence_path.is_file() and metadata_path.is_file() and manifest_path.is_file(),
                "candidate payload file missing")
        sequence_data = sequence_path.read_bytes()
        metadata_data = metadata_path.read_bytes()
        manifest_data = manifest_path.read_bytes()
        require(sha(sequence_data) == entry["sequence_sha256"] and
                sha(metadata_data) == entry["metadata_sha256"], "candidate index hash mismatch")
        metadata = strict_json(metadata_data, "candidate_metadata.json")
        manifest = strict_json(manifest_data, "checksum_manifest.json")
        targets = {row["path"]: row for row in manifest.get("files", [])}
        require(set(targets) == {"normalized_sequence.wisrq", "candidate_metadata.json"},
                "candidate checksum manifest membership mismatch")
        for name, data in (("normalized_sequence.wisrq", sequence_data),
                           ("candidate_metadata.json", metadata_data)):
            target = targets[name]
            require(target["size_bytes"] == len(data) and target["sha256"] == sha(data),
                    "candidate checksum manifest identity mismatch")
        sequence = parse_wisrcb(sequence_data)
        require(sequence["sample_id"] == entry["capture_session_id"] == metadata["capture_session_id"],
                "candidate session provenance mismatch")
        require(sequence["word"] == "UNLABELED" and sequence["speed"] == "RUNTIME" and
                sequence["split"] == "runtime", "candidate runtime sequence labeling mismatch")
        require(entry["word"] == metadata["word"] == metadata["expected_word"],
                "candidate metadata word provenance mismatch")
        require(metadata["predicted_word"] == entry["word"] and
                metadata["outcome"] == "StrictSuccess" and metadata["eligibility"] == "Eligible" and
                metadata["eligible_for_profile"] is True, "candidate outcome metadata mismatch")
        require(metadata["controller_state"] == "Completed" and
                metadata["strict_accepted"] is True and
                metadata["strict_expected_class_accept"] is True and
                metadata["strict_quest_success"] is True, "candidate strict gate metadata mismatch")
        require(metadata["profile_id"] == "step9l_current_user_multi_prototype_v1_0" and
                metadata["route_id"] == "MultiPrototypeCandidate", "candidate route/profile mismatch")
        require(metadata["normalized_frame_count"] == sequence["frame_count"] and
                metadata["raw_frame_count"] == sequence["frame_count"] and
                metadata["valid_landmark_count"] == sequence["valid_landmark_count"],
                "candidate frame/valid count mismatch")
        require(metadata["sequence_binary_sha256"] == sha(sequence_data) and
                metadata["sequence_source_payload_sha256"] == sequence["source_payload_sha256"] and
                metadata["sequence_normalized_payload_sha256"] == sequence["normalized_payload_sha256"],
                "candidate payload provenance mismatch")
        require(metadata["approved_threshold"] == EXPECTED_THRESHOLDS[entry["word"]],
                "candidate threshold mismatch")
        base = "candidate_repository_after_payload/" + relative.as_posix() + "/"
        payloads[base + "normalized_sequence.wisrq"] = sequence_data
        payloads[base + "candidate_metadata.json"] = metadata_data
        payloads[base + "checksum_manifest.json"] = manifest_data
        rows.append({
            "capture_session_id": entry["capture_session_id"],
            "word": entry["word"],
            "outcome": entry["outcome"],
            "eligibility": entry["eligibility"],
            "eligible_for_profile": metadata["eligible_for_profile"],
            "relative_directory": entry["relative_directory"],
            "raw_frame_count": metadata["raw_frame_count"],
            "normalized_frame_count": metadata["normalized_frame_count"],
            "valid_landmark_count": metadata["valid_landmark_count"],
            "sequence_size_bytes": len(sequence_data),
            "sequence_sha256": sha(sequence_data),
            "metadata_size_bytes": len(metadata_data),
            "metadata_sha256": sha(metadata_data),
            "manifest_size_bytes": len(manifest_data),
            "manifest_sha256": sha(manifest_data),
            "wisrcb01": sequence,
            "status": "PASS",
        })
    return ({
        "schema_version": SCHEMA + ".candidate_validation",
        "status": "PASS",
        "before_candidate_count": 0,
        "after_candidate_count": 2,
        "new_candidate_count": 2,
        "eligible_count": 2,
        "review_only_count": 0,
        "capture_session_id_duplicate_count": 0,
        "media_file_count": 0,
        "existing_candidate_overwrite_count": 0,
        "index_candidate_count_matches_ui": True,
        "rows": rows,
    }, payloads)


def screen_record(result: Path, name: str) -> dict[str, Any]:
    path = result / name
    require(path.is_file(), "required screen missing: " + name)
    return record(name, path.read_bytes())


def playmode_screen_review(result: Path) -> dict[str, Any]:
    screens = {
        "test_runner": screen_record(result, "TestRunner_FIRST_RESULT.png"),
        "camera_ready": screen_record(result, "PlayMode_00_CameraReady.png"),
        "greeting_first": screen_record(result, "PlayMode_01_GREETING_First.png"),
        "greeting_retry1": screen_record(result, "PlayMode_01_GREETING_Retry1.png"),
        "greeting_retry1_detail": screen_record(result, "PlayMode_01_GREETING_Retry1(1).png"),
        "love_first": screen_record(result, "PlayMode_02_LOVE_First.png"),
        "love_first_detail": screen_record(result, "PlayMode_02_LOVE_First(1).png"),
        "thank_you_first": screen_record(result, "PlayMode_03_THANK_YOU_First.png"),
        "thank_you_first_detail": screen_record(result, "PlayMode_03_THANK_YOU_First(1).png"),
        "thank_you_retry1": screen_record(result, "PlayMode_03_THANK_YOU_Retry1.png"),
        "thank_you_retry1_detail": screen_record(result, "PlayMode_03_THANK_YOU_Retry1(1).png"),
    }
    return {
        "schema_version": SCHEMA + ".screen_review",
        "review_method": "manual full-resolution evidence review by Codex; XML/repository/log cross-checked",
        "test_runner": {
            "selected_fixture_count": 7,
            "visible_passed": 87,
            "visible_failed": 0,
            "console_error_count": 0,
            "status": "PASS",
        },
        "camera": {
            "state": "Ready",
            "preview_visible": True,
            "candidate_route_active": True,
            "camera_started": True,
            "frame_ready": True,
            "frame_count": 333,
            "provider_observation_ready": True,
            "observation_count": 332,
            "technical_panel_hidden": True,
            "legacy_practice_controller_active": False,
            "status": "PASS",
        },
        "word_attempts": [
            {"word": "GREETING", "attempt": "First", "outcome": "TechnicalFailure",
             "strict_accepted": False, "expected_class_accept": False, "quest_success": False,
             "storage_status": "NotRequested", "progressed": False, "retry_count_after": 0},
            {"word": "GREETING", "attempt": "Retry1", "outcome": "StrictSuccess",
             "strict_accepted": True, "expected_class_accept": True, "quest_success": True,
             "storage_status": "Saved", "eligibility": "Eligible", "progressed": True,
             "capture_session_id": "121b5160-dbbd-4811-a156-1b5bc6b00fee", "retry_count_after": 1},
            {"word": "LOVE", "attempt": "First", "outcome": "StrictSuccess",
             "strict_accepted": True, "expected_class_accept": True, "quest_success": True,
             "storage_status": "Saved", "eligibility": "Eligible", "progressed": True,
             "capture_session_id": "4c87d105-fcf7-43b0-9658-555ca97d77e5", "retry_count_after": 0},
            {"word": "THANK_YOU", "attempt": "First", "outcome": "WrongWord",
             "strict_accepted": False, "expected_class_accept": False, "quest_success": False,
             "storage_status": "NotRequested", "progressed": False, "retry_count_after": 0},
            {"word": "THANK_YOU", "attempt": "Retry1", "outcome": "WrongWord",
             "strict_accepted": False, "expected_class_accept": False, "quest_success": False,
             "storage_status": "NotRequested", "progressed": False, "retry_count_after": 1},
        ],
        "retry_counts": {"GREETING": 1, "LOVE": 0, "THANK_YOU": 1},
        "ui_candidate_counts": {
            "GREETING": {"eligible": 1, "review_only": 0, "minimum": 3},
            "LOVE": {"eligible": 1, "review_only": 0, "minimum": 3},
            "THANK_YOU": {"eligible": 0, "review_only": 0, "minimum": 3},
        },
        "practice_completed": False,
        "completion_screen": "NOT_APPLICABLE",
        "stop_reason": "THANK_YOU remained WrongWord after the one allowed manual retry",
        "automatic_capture_or_retry_observed": False,
        "screens": screens,
        "status": "PASS",
    }


def assertion_bytes(rows: list[tuple[str, str, str]]) -> bytes:
    keys = set()
    output = []
    for key, status, evidence in rows:
        require(key not in keys, "duplicate assertion key: " + key)
        keys.add(key)
        output.append(json.dumps({"assertion_key": key, "status": status, "evidence": evidence},
                                 ensure_ascii=False, sort_keys=True, allow_nan=False))
    return ("\n".join(output) + "\n").encode("utf-8")


def validation_report(tests: dict[str, Any], log: dict[str, Any], play: dict[str, Any],
                      candidates: dict[str, Any]) -> bytes:
    return f"""# Step 10A v1.2 GUI Unity Editor Gate

## Result

- Unity: `6000.3.14f1`; compiler errors: `{log['compiler_error_count']}`.
- EditMode: one selected run, `{tests['passed']}/{tests['unique_testcase_count']} PASS`, failed/skipped/inconclusive `0/0/0`, retry/repeat `0/0`.
- PlayMode: one entry in the same Editor session.
- Camera: `{play['camera']['state']}`, actual preview visible, frame `{play['camera']['frame_count']}`, observation `{play['camera']['observation_count']}`.

## Live outcomes

- GREETING First: `TechnicalFailure`, no progress, no storage. Retry1: `StrictSuccess`, `Eligible` saved.
- LOVE First: `StrictSuccess`, `Eligible` saved; no retry.
- THANK_YOU First: `WrongWord`, no progress/storage. Retry1: `WrongWord`, no progress/storage.
- The run stopped after the one allowed THANK_YOU retry. Practice `Completed` and exit screen are `NOT_APPLICABLE`.
- `LowConfidenceCorrect` was not observed live and is `NOT_APPLICABLE`; its regression remains covered by the passed EditMode suite.

## Candidate repository

- Candidate count: `{candidates['before_candidate_count']} → {candidates['after_candidate_count']}`.
- Eligible/ReviewOnly: `{candidates['eligible_count']}/{candidates['review_only_count']}`.
- Both WISRCB01 payloads pass embedded checksum, shape/mask validity, provenance, manifest, and writer/reader byte-identical round-trip.
- UI and index counts match; duplicate capture/session IDs and stored media files are zero.
- Active profile, 9 prototypes, 3 thresholds, project files, and protected files remain byte-identical.

## Verdict

The Editor compile/EditMode retest, candidate camera readiness, observed outcome routing, and live candidate storage are `PASS`. The overall playable completion gate is `FAIL` because THANK_YOU remained `WrongWord` after Retry1, so the three-word practice did not reach `Completed`.

```text
STEP10A_V1_2_GUI_UNITY_EDITOR_RETEST_PASS_87_OF_87
STEP10A_V1_2_CANDIDATE_CAMERA_READY_AND_STORAGE_ROUTE_PASS
STEP10A_V1_2_GREETING_AND_LOVE_ELIGIBLE_CANDIDATES_SAVED
STEP10A_V1_2_PLAYABLE_LIVE_GATE_FAIL_THANK_YOU_WRONG_WORD_AFTER_RETRY1
STEP10A_V1_2_PRACTICE_COMPLETION_NOT_REACHED
BLOCKED_PENDING_EXTERNAL_REVIEW
```
""".encode("utf-8")


def add_output_manifest(files: dict[str, bytes]) -> dict[str, bytes]:
    require("output_file_manifest.json" not in files, "output manifest already present")
    manifest = {
        "schema_version": SCHEMA + ".manifest",
        "self_excluded": True,
        "file_count": len(files),
        "files": [record(name, files[name]) for name in sorted(files)],
    }
    result = dict(files)
    result["output_file_manifest.json"] = jbytes(manifest)
    return result


def post_verify_package(gate: Any, result: Path, delivery: Path, detached: Path,
                        expected: dict[str, bytes]) -> dict[str, Any]:
    require(delivery.is_file() and detached.is_file(), "delivery output missing")
    blob = delivery.read_bytes()
    detached_text = detached.read_text(encoding="ascii")
    require(detached_text == f"{sha(blob)}  {delivery.name}\n", "detached checksum mismatch")
    verification = gate.validate_zip(blob, expected)
    require(
        verification["duplicate_count"] == 0
        and verification["unsafe_path_count"] == 0
        and verification["encrypted_entry_count"] == 0
        and verification["symlink_entry_count"] == 0
        and verification["crc_failure"] is None
        and verification["membership_match"]
        and verification["payload_mismatch_count"] == 0
        and verification["fixed_timestamp_count"] == len(expected),
        "post-ZIP safety/payload validation failed")
    require(gate.deterministic_zip(expected) == blob, "deterministic ZIP regeneration mismatch")
    disk_files = {path.relative_to(result).as_posix(): path.read_bytes()
                  for path in result.rglob("*") if path.is_file()}
    require(disk_files == expected, "result directory/file map mismatch")
    manifest = strict_json(expected["output_file_manifest.json"], "output_file_manifest.json")
    require({row["relative_path"] for row in manifest["files"]} ==
            set(expected) - {"output_file_manifest.json"}, "manifest membership mismatch")
    for row in manifest["files"]:
        data = expected[row["relative_path"]]
        require(len(data) == row["size_bytes"] and sha(data) == row["sha256"],
                "manifest target identity mismatch")
    return {
        **verification,
        "deterministic_regeneration_byte_identical": True,
        "detached_checksum_match": True,
        "manifest_target_count": len(manifest["files"]),
        "status": "PASS",
    }


def package(root: Path) -> None:
    validation = root / "Validation"
    result = validation / NAME
    delivery = validation / (NAME + "_delivery.zip")
    detached = validation / (NAME + "_delivery.zip.sha256")
    require(result.is_dir(), "prepared gate directory missing")
    require(not delivery.exists() and not detached.exists(), "never-overwrite delivery target exists")
    required_raw = [
        "fixed_input_preflight.json", "project_files_before.json", "protected_files_before.json",
        "active_candidate_profile_before.json", "candidate_repository_before.json",
        "git_staged_before.json", "preflight_before_snapshot.json", "EVIDENCE_INSTRUCTIONS.txt",
        "TestResults.xml", "TestRunner_FIRST_RESULT.png", "Editor.log",
        "PlayMode_00_CameraReady.png", "PlayMode_01_GREETING_First.png",
        "PlayMode_01_GREETING_Retry1.png", "PlayMode_01_GREETING_Retry1(1).png",
        "PlayMode_02_LOVE_First.png", "PlayMode_02_LOVE_First(1).png",
        "PlayMode_03_THANK_YOU_First.png", "PlayMode_03_THANK_YOU_First(1).png",
        "PlayMode_03_THANK_YOU_Retry1.png", "PlayMode_03_THANK_YOU_Retry1(1).png",
    ]
    for name in required_raw:
        require((result / name).is_file(), "required raw evidence missing: " + name)

    v11 = load_v11(root)
    verify_fixed(v11, root, V12_ZIP, V12_SIZE, V12_SHA, V12_DETACHED_FILE_SHA)
    verify_fixed(v11, root, V11_ZIP, V11_SIZE, V11_SHA, V11_DETACHED_FILE_SHA)
    verify_fixed(v11, root, FIRST_GATE_ZIP, FIRST_GATE_SIZE, FIRST_GATE_SHA,
                 FIRST_GATE_DETACHED_FILE_SHA)
    tests = parse_test_results(result)
    log = parse_editor_log(result)
    play = playmode_screen_review(result)
    project_before = strict_json((result / "project_files_before.json").read_bytes(), "project before")
    protected_before = strict_json((result / "protected_files_before.json").read_bytes(), "protected before")
    profile_before = strict_json((result / "active_candidate_profile_before.json").read_bytes(), "profile before")
    repository_before = strict_json((result / "candidate_repository_before.json").read_bytes(), "repository before")
    project_after = snapshot_after(root, project_before, "project_files_after")
    protected_after = snapshot_after(root, protected_before, "protected_files_after")
    profile_after = snapshot_after(root, profile_before, "active_candidate_profile_after")
    profile_after.update({
        "profile_id": profile_before["profile_id"],
        "prototype_count": 9,
        "threshold_count": 3,
        "thresholds": EXPECTED_THRESHOLDS,
    })
    require(project_after["pass_count"] == 30 and protected_after["pass_count"] == 106 and
            profile_after["pass_count"] == 22, "protected after identity mismatch")
    repository_after = snapshot_repository()
    candidate_validation, candidate_payloads = validate_candidate_repository(
        repository_before, repository_after)
    staged = staged_files(root)
    require(staged == [], "Git staged files changed")
    git_after = {
        "schema_version": SCHEMA + ".git_after",
        "staged_count": 0,
        "staged_files": [],
        "status": "PASS",
    }

    gate = load_v11_gate(root)
    transaction = gate.transaction_regression()
    require(transaction["status"] == "PASS", "packaging transaction regression failed")
    primary = {
        "schema_version": SCHEMA + ".primary_verdict",
        "overall_gate": "FAIL",
        "unity_compile": "PASS",
        "editmode_87": "PASS",
        "camera_readiness": "PASS",
        "observed_live_candidate_storage": "PASS",
        "practice_completion": "FAIL",
        "failure_reason": "THANK_YOU remained WrongWord after the one allowed manual retry",
        "low_confidence_correct_live": "NOT_APPLICABLE",
        "completion_screen": "NOT_APPLICABLE",
        "status_strings": [
            "STEP10A_V1_2_GUI_UNITY_EDITOR_RETEST_PASS_87_OF_87",
            "STEP10A_V1_2_CANDIDATE_CAMERA_READY_AND_STORAGE_ROUTE_PASS",
            "STEP10A_V1_2_GREETING_AND_LOVE_ELIGIBLE_CANDIDATES_SAVED",
            "STEP10A_V1_2_PLAYABLE_LIVE_GATE_FAIL_THANK_YOU_WRONG_WORD_AFTER_RETRY1",
            "STEP10A_V1_2_PRACTICE_COMPLETION_NOT_REACHED",
            "BLOCKED_PENDING_EXTERNAL_REVIEW",
        ],
    }
    assertions = assertion_bytes([
        ("fixed_v1_2_delivery", "PASS", V12_SHA),
        ("fixed_v1_1_delivery", "PASS", V11_SHA),
        ("fixed_first_gui_failure", "PASS", FIRST_GATE_SHA),
        ("unity_version", "PASS", "6000.3.14f1"),
        ("compiler_errors", "PASS", "0"),
        ("editmode_execution_count", "PASS", "1"),
        ("unique_editmode_tests", "PASS", "87"),
        ("editmode_pass_fail", "PASS", "87/0"),
        ("skipped_inconclusive", "PASS", "0/0"),
        ("retry_repeat", "PASS", "0/0"),
        ("playmode_execution_count", "PASS", "1"),
        ("candidate_camera_ready", "PASS", "frames 333; observations 332"),
        ("candidate_route_and_preview", "PASS", "active; preview visible"),
        ("greeting_first_technical_failure", "PASS", "no progress; no storage"),
        ("greeting_retry1", "PASS", "StrictSuccess; Eligible saved"),
        ("love_first", "PASS", "StrictSuccess; Eligible saved"),
        ("thank_you_first", "PASS", "WrongWord; no progress/storage"),
        ("thank_you_retry1", "PASS", "WrongWord; no progress/storage"),
        ("manual_retry_limits", "PASS", "GREETING 1; LOVE 0; THANK_YOU 1"),
        ("candidate_repository_count", "PASS", "0 to 2"),
        ("candidate_eligibility", "PASS", "Eligible 2; ReviewOnly 0"),
        ("wisrcb01_checksum_roundtrip", "PASS", "2/2"),
        ("candidate_index_manifest", "PASS", "2/2"),
        ("candidate_session_duplicates", "PASS", "0"),
        ("candidate_media_files", "PASS", "0"),
        ("ui_index_candidate_counts", "PASS", "GREETING 1; LOVE 1; THANK_YOU 0"),
        ("project_files", "PASS", "30/30"),
        ("protected_files", "PASS", "106/106"),
        ("active_profile", "PASS", "22/22; 9 prototypes; 3 thresholds"),
        ("git_staged", "PASS", "0"),
        ("packaging_transaction", "PASS", "3/3 injected rollback; never-overwrite"),
        ("practice_completed", "FAIL", "THANK_YOU WrongWord after Retry1"),
        ("low_confidence_correct_live", "NOT_APPLICABLE", "not observed in fixed one-session gate"),
        ("completion_screen", "NOT_APPLICABLE", "practice did not complete"),
    ])

    generated: dict[str, bytes] = {
        "test_results_summary.json": jbytes(tests),
        "editor_log_summary.json": jbytes(log),
        "playmode_live_gate_evaluation.json": jbytes(play),
        "candidate_repository_after.json": jbytes(repository_after),
        "candidate_repository_validation.json": jbytes(candidate_validation),
        "project_files_after.json": jbytes(project_after),
        "protected_files_after.json": jbytes(protected_after),
        "active_candidate_profile_after.json": jbytes(profile_after),
        "git_staged_after.json": jbytes(git_after),
        "packaging_transaction_failure_injection_regression.json": jbytes(transaction),
        "primary_verdict.json": jbytes(primary),
        "ASSERTIONS.jsonl": assertions,
        "VALIDATION_REPORT.md": validation_report(tests, log, play, candidate_validation),
        "analysis/" + Path(__file__).name: Path(__file__).read_bytes(),
        **candidate_payloads,
    }
    for relative in generated:
        require(not (result / relative).exists(), "never-overwrite generated result exists: " + relative)
    raw_files = {path.relative_to(result).as_posix(): path.read_bytes()
                 for path in result.rglob("*") if path.is_file()}
    require("output_file_manifest.json" not in raw_files, "result already packaged")
    complete = add_output_manifest({**raw_files, **generated})
    zip_blob = gate.deterministic_zip(complete)
    detached_blob = f"{sha(zip_blob)}  {delivery.name}\n".encode("ascii")
    zip_check = gate.validate_zip(zip_blob, complete)
    require(zip_check["entry_count"] == len(complete) and zip_check["duplicate_count"] == 0 and
            zip_check["unsafe_path_count"] == 0 and zip_check["encrypted_entry_count"] == 0 and
            zip_check["symlink_entry_count"] == 0 and zip_check["crc_failure"] is None and
            zip_check["membership_match"] and zip_check["payload_mismatch_count"] == 0,
            "pre-commit ZIP validation failed")

    temp = Path(tempfile.mkdtemp(prefix="step10a_v12_gate_generated_", dir=validation))
    created: list[Path] = []
    committed = False
    try:
        for relative, data in {**generated, "output_file_manifest.json": complete["output_file_manifest.json"]}.items():
            path = temp / PurePosixPath(relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        for source in sorted((path for path in temp.rglob("*") if path.is_file()),
                             key=lambda path: len(path.parts)):
            relative = source.relative_to(temp)
            target = result / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            require(not target.exists(), "generated target appeared during commit")
            source.replace(target)
            created.append(target)
        gate.commit_pair(zip_blob, detached_blob, delivery, detached)
        verification = post_verify_package(gate, result, delivery, detached, complete)
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
        "status": "PASS",
        "overall_gate": "FAIL",
        "result_directory": str(result),
        "result_file_count": len(complete),
        "delivery_zip": str(delivery),
        "delivery_size_bytes": delivery.stat().st_size,
        "delivery_sha256": sha(delivery.read_bytes()),
        "detached_checksum": str(detached),
        "detached_size_bytes": detached.stat().st_size,
        "detached_sha256": sha(detached.read_bytes()),
        "manifest_target_count": len(complete) - 1,
        "zip_verification": verification,
        "compiler_errors": 0,
        "editmode_execution_count": 1,
        "editmode": "87 PASS / 0 FAIL",
        "playmode_execution_count": 1,
        "camera": {"state": "Ready", "frames": 333, "observations": 332},
        "candidate_count_before_after": [0, 2],
        "git_staged_count": 0,
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--package", action="store_true")
    args = parser.parse_args()
    require(args.prepare != args.package, "select exactly one of --prepare or --package")
    if args.prepare:
        prepare(args.root.resolve())
    else:
        package(args.root.resolve())


if __name__ == "__main__":
    main()
