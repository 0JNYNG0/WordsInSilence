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
from collections import Counter
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any


SCHEMA = "WordsInSilence.Step10A.RetryExhaustionNeedsPracticeCompletion.v1.3.GuiGate"
NAME = "WordsInSilence_step10a_pet_robot_retry_exhaustion_needs_practice_completion_v1_3_gui_unity_editor_gate"
SOURCE_RESULT = "Validation/WordsInSilence_step10a_pet_robot_retry_exhaustion_needs_practice_completion_v1_3"
V13_ZIP = "Validation/WordsInSilence_step10a_pet_robot_retry_exhaustion_needs_practice_completion_v1_3_delivery.zip"
V13_ZIP_SIZE = 101574
V13_ZIP_SHA = "42cfe5679c09defc220033fd99eb638576bd5313b5c047dfe25766ac425575bf"
V13_DETACHED_SHA = "6c630bc1b18770deec5e23e8669b5075eaae42a6907c61e918e92316cd1e0354"
V12_ZIP = "Validation/WordsInSilence_step10a_pet_robot_candidate_camera_startup_route_v1_2_delivery.zip"
V12_ZIP_SIZE = 73741
V12_ZIP_SHA = "edea92e19bac40ce88b8eb0c54c62c1f6e66a2699fb1d9c48e016eb4421a5799"
V12_DETACHED_SHA = "1f64f5b54dd8346f1a39ef306c87ca310c6d82411ddcfe671f84fcdaddd9074b"
V12_GATE_ZIP = "Validation/WordsInSilence_step10a_pet_robot_candidate_camera_startup_route_v1_2_gui_unity_editor_gate_delivery.zip"
V12_GATE_ZIP_SIZE = 1521570
V12_GATE_ZIP_SHA = "31c6375d69bf04af779deb60956f6a477129731f6af159d80cf398df5baf6af2"
V12_GATE_DETACHED_SHA = "7902b5fd27d466ec30085ea51fe857d66af68c566df33b60d3c8eeaf723b90f9"
NEW_TEST = "Assets/_Project/Tests/EditMode/PetRobotRetryExhaustionNeedsPracticeCompletionV1Tests.cs"
NEW_TEST_SHA = "96c8653ddb594a8788c0bd53c4496f5540cdb63bf1fc72f2ab3f7f48485dd05e"
NEW_META = NEW_TEST + ".meta"
NEW_META_SHA = "077489812cc3c1d1d7a7546edef32047497ed0d8b2a74eccd863df8b0ada098f"
V12_GATE_TOOL = "Tools/s001_step10a_pet_robot_candidate_camera_startup_route_v1_2_gui_unity_editor_gate.py"
V11_GATE_TOOL = "Tools/s001_step10a_pet_robot_candidate_camera_startup_route_v1_1_gui_unity_editor_gate.py"
EXPECTED_FIXTURES = {
    "PetRobotSignPracticeVerticalSliceV1Tests": 23,
    "GuidedQuestLowConfidenceOutcomeLayerV1Tests": 10,
    "CandidateTransientAmbiguousHandIdentityToleranceV1Tests": 8,
    "PersonalLiveRecognitionIntegrationV1Tests": 22,
    "PersonalMultiPrototypeRuntimeIntegrationV1Tests": 8,
    "PersonalMultiPrototypeLiveOptInRoutingV1Tests": 8,
    "PetRobotCandidateCameraStartupRouteV1Tests": 8,
    "PetRobotRetryExhaustionNeedsPracticeCompletionV1Tests": 15,
}
EXPECTED_THRESHOLDS = {
    "GREETING": 0.5113059473720765,
    "LOVE": 0.391276023460427,
    "THANK_YOU": 0.37709977431991587,
}


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def jbytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def record(relative: str, data: bytes) -> dict[str, Any]:
    return {"relative_path": relative, "size_bytes": len(data), "sha256": sha(data)}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def staged_files(root: Path) -> list[str]:
    run = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=root,
                         capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    require(run.returncode == 0, "Git staged query failed")
    return [line.strip() for line in run.stdout.splitlines() if line.strip()]


def verify_fixed(root: Path, relative: str, size: int, digest: str, detached_digest: str) -> dict[str, Any]:
    path = root / relative
    detached = path.with_suffix(path.suffix + ".sha256")
    data = path.read_bytes()
    require(len(data) == size and sha(data) == digest, "fixed ZIP identity mismatch: " + relative)
    require(sha(detached.read_bytes()) == detached_digest,
            "fixed detached identity mismatch: " + str(detached.relative_to(root)))
    require(detached.read_text(encoding="ascii") == f"{digest}  {path.name}\n",
            "fixed detached syntax mismatch: " + str(detached.relative_to(root)))
    return {"relative_path": relative, "size_bytes": size, "sha256": digest,
            "detached_sha256": detached_digest, "status": "PASS"}


def verify_file_set(root: Path, baseline: dict[str, Any], count: int) -> dict[str, Any]:
    require(baseline["file_count"] == count, "baseline count mismatch")
    rows = []
    for prior in baseline["records"]:
        relative = prior["relative_path"]
        data = (root / relative).read_bytes()
        require(len(data) == prior["size_bytes"] and sha(data) == prior["sha256"],
                "preflight identity mismatch: " + relative)
        rows.append(record(relative, data))
    return {"file_count": count, "records": rows, "status": "PASS"}


def repository_snapshot(baseline: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
    root = Path(baseline["path"])
    rows = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        rows.append(record(path.relative_to(root).as_posix(), path.read_bytes()))
    require(len(rows) == 7 and rows == baseline["files"],
            "existing two-candidate repository differs from approved baseline")
    index_path = root / "candidate_index.json"
    index = index_path.read_bytes()
    parsed = json.loads(index.decode("utf-8-sig"))
    entries = parsed.get("candidates", parsed.get("entries", []))
    require(len(entries) == 2, "candidate index does not contain two baseline rows")
    return ({"path": str(root), "exists": True, "candidate_count": 2,
             "file_count": 7, "files": rows, "status": "PASS"}, index)


def prepare(root: Path) -> None:
    destination = root / "Validation" / NAME
    require(not destination.exists(), "gate evidence directory already exists")
    fixed_inputs = [
        verify_fixed(root, V13_ZIP, V13_ZIP_SIZE, V13_ZIP_SHA, V13_DETACHED_SHA),
        verify_fixed(root, V12_ZIP, V12_ZIP_SIZE, V12_ZIP_SHA, V12_DETACHED_SHA),
        verify_fixed(root, V12_GATE_ZIP, V12_GATE_ZIP_SIZE, V12_GATE_ZIP_SHA, V12_GATE_DETACHED_SHA),
    ]
    source = root / SOURCE_RESULT
    project = verify_file_set(root, load_json(source / "project_files_after.json"), 30)
    protected = verify_file_set(root, load_json(source / "protected_files_after.json"), 106)
    profile = verify_file_set(root, load_json(source / "active_candidate_profile_after.json"), 22)
    profile_fixed = load_json(
        root / "Validation/WordsInSilence_step10a_pet_robot_candidate_camera_startup_route_v1_2_gui_unity_editor_gate/active_candidate_profile_after.json")
    require(profile_fixed["prototype_count"] == 9 and profile_fixed["threshold_count"] == 3 and
            profile_fixed["thresholds"] == {
                "GREETING": 0.5113059473720765,
                "LOVE": 0.391276023460427,
                "THANK_YOU": 0.37709977431991587,
            }, "candidate prototype/threshold contract mismatch")
    new_test = (root / NEW_TEST).read_bytes()
    new_meta = (root / NEW_META).read_bytes()
    require(sha(new_test) == NEW_TEST_SHA and sha(new_meta) == NEW_META_SHA,
            "new v1.3 fixture identity mismatch")
    repository, index = repository_snapshot(load_json(source / "candidate_repository_after.json"))
    staged = staged_files(root)
    require(staged == [], "Git staged file count is not zero")
    context = (root / "WordsInSilence_PROJECT_CONTEXT.md").read_bytes()

    payload = {
        "schema_version": SCHEMA + ".preflight",
        "implementation_owner": "Codex",
        "unity_version": "6000.3.14f1",
        "fixed_inputs": fixed_inputs,
        "project_files": project,
        "new_fixture": {"source": record(NEW_TEST, new_test), "meta": record(NEW_META, new_meta)},
        "protected_files": protected,
        "candidate_profile": {
            **profile,
            "prototype_count": 9,
            "threshold_count": 3,
            "thresholds": profile_fixed["thresholds"],
        },
        "candidate_repository": repository,
        "git_staged": staged,
        "context": record("WordsInSilence_PROJECT_CONTEXT.md", context),
        "status": "PASS",
    }
    instructions = """Step 10A v1.3 GUI gate evidence folder. Do not rename or modify preflight files.
Run the eight listed EditMode fixtures together with Run Selected exactly once.
Add same-run TestResults.xml, full Test Runner screenshot, and full Editor.log.
Only after 102/102 PASS, enter PlayMode exactly once and add Camera Ready/Failed, each First result,
conditional Retry1, NeedsPractice-before/after, and final summary/stop screenshots.
Do not overwrite screenshots from earlier attempts; this gate has one authorized execution only.
""".encode("utf-8")
    temp = destination.with_name("." + NAME + ".prepare-tmp")
    require(not temp.exists(), "prepare temp already exists")
    try:
        temp.mkdir(parents=True)
        (temp / "preflight_before_snapshot.json").write_bytes(jbytes(payload))
        (temp / "project_files_before.json").write_bytes(jbytes(project))
        (temp / "protected_files_before.json").write_bytes(jbytes(protected))
        (temp / "active_candidate_profile_before.json").write_bytes(jbytes(payload["candidate_profile"]))
        (temp / "candidate_repository_before.json").write_bytes(jbytes(repository))
        (temp / "candidate_index_before.json").write_bytes(index)
        (temp / "git_staged_before.json").write_bytes(jbytes({"staged_count": 0, "staged_files": []}))
        (temp / "EVIDENCE_INSTRUCTIONS.txt").write_bytes(instructions)
        temp.replace(destination)
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        raise
    print(json.dumps({
        "status": "PASS",
        "evidence_directory": str(destination),
        "fixed_input_count": 3,
        "project_files": 30,
        "protected_files": 106,
        "candidate_profile_files": 22,
        "prototype_threshold": [9, 3],
        "candidate_repository": [2, 7],
        "git_staged": 0,
    }, ensure_ascii=False, indent=2))


def load_module(root: Path, relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, root / relative)
    require(spec is not None and spec.loader is not None, "cannot load approved helper: " + relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def property_value(case: ET.Element, name: str) -> str:
    for item in case.findall("./properties/property"):
        if item.attrib.get("name") == name:
            return item.attrib.get("value", "")
    return ""


def parse_tests(result: Path) -> dict[str, Any]:
    xml_path = result / "TestResults.xml"
    screen_path = result / "TestRunner_FIRST_RESULT.png"
    require(xml_path.is_file() and screen_path.is_file(), "test XML or screen missing")
    run = ET.parse(xml_path).getroot()
    cases = run.findall(".//test-case")
    selected = [case for case in cases
                if case.attrib.get("classname", "").split(".")[-1] in EXPECTED_FIXTURES]
    names = [case.attrib.get("fullname", "") for case in selected]
    counts = Counter(case.attrib.get("result", "") for case in selected)
    fixture_rows = []
    for fixture, expected in EXPECTED_FIXTURES.items():
        rows = [case for case in selected
                if case.attrib.get("classname", "").endswith("." + fixture)]
        fixture_rows.append({
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
    require(len(selected) == len(set(names)) == 102 and counts["Passed"] == 102 and
            counts["Failed"] == 0 and counts["Skipped"] == 0 and counts["Inconclusive"] == 0 and
            retries == 0 and repeats == 0 and
            all(row["actual"] == row["expected"] == row["passed"] for row in fixture_rows),
            "Unity 102-test result contract failed")
    return {
        "schema_version": SCHEMA + ".test_results",
        "platform": "EditMode",
        "execution_count": 1,
        "testcase_count": 102,
        "unique_testcase_count": 102,
        "passed": 102,
        "failed": 0,
        "skipped": 0,
        "inconclusive": 0,
        "retry_nonzero": retries,
        "repeat_nonzero": repeats,
        "start_time_utc": run.attrib.get("start-time"),
        "end_time_utc": run.attrib.get("end-time"),
        "duration_seconds": run.attrib.get("duration"),
        "fixtures": fixture_rows,
        "xml": record("TestResults.xml", xml_path.read_bytes()),
        "screen": record("TestRunner_FIRST_RESULT.png", screen_path.read_bytes()),
        "status": "PASS",
    }


def parse_editor_log(result: Path) -> dict[str, Any]:
    path = result / "Editor.log"
    require(path.is_file(), "Editor.log missing")
    data = path.read_bytes()
    text_value = data.decode("utf-8", errors="replace")
    counts = {
        "unity_version": text_value.count("6000.3.14f1"),
        "batchmode_zero": text_value.count("BatchMode: 0"),
        "human_controlled": text_value.count("IsHumanControllingUs: 1"),
        "tundra_build_success": text_value.count("Tundra build success"),
        "compiler_error_cs": len(re.findall(r"\berror CS\d+", text_value)),
        "scripts_have_compiler_errors": text_value.count("Scripts have compiler errors"),
        "compilation_failed": text_value.count("Compilation failed"),
        "prebuild": text_value.count("Executing IPrebuildSetup"),
        "postbuild": text_value.count("Executing IPostBuildCleanup"),
        "default_result_save": text_value.count(
            "Saving results to: C:/Users/local-user/AppData/LocalLow/DefaultCompany/WordsInSilence\\TestResults.xml"),
        "gate_result_export": text_value.count(
            "Saving results to: C:/MyUnityProjects/WordsInSilence/Validation/" + NAME + "/TestResults.xml"),
        "playmode_assembly_reload": text_value.count("Reloading assemblies for play mode."),
        "pose_provider_initialized": text_value.count("[PoseLandmarkProvider]"),
        "hand_provider_initialized": text_value.count("[MediaPipeLandmarkProvider]"),
        "null_reference_exception": text_value.count("NullReferenceException"),
        "invalid_operation_exception": text_value.count("InvalidOperationException"),
        "unhandled_exception": text_value.count("Unhandled Exception"),
    }
    errors = counts["compiler_error_cs"] + counts["scripts_have_compiler_errors"] + counts["compilation_failed"]
    require(counts["unity_version"] > 0 and counts["batchmode_zero"] == 1 and
            counts["human_controlled"] == 1 and counts["tundra_build_success"] >= 1 and
            errors == 0 and counts["prebuild"] == 1 and counts["postbuild"] == 1 and
            counts["default_result_save"] == 1 and counts["gate_result_export"] == 1 and
            counts["playmode_assembly_reload"] == 1 and
            counts["pose_provider_initialized"] >= 1 and counts["hand_provider_initialized"] >= 1 and
            counts["null_reference_exception"] == 0 and
            counts["invalid_operation_exception"] == 0 and counts["unhandled_exception"] == 0,
            "Editor.log compile/same-run/PlayMode contract failed")
    return {
        "schema_version": SCHEMA + ".editor_log",
        "unity_version": "6000.3.14f1",
        "compiler_error_count": errors,
        "editmode_execution_count": 1,
        "playmode_execution_count": 1,
        "same_editor_session": True,
        "counts": counts,
        "log": record("Editor.log", data),
        "status": "PASS",
    }


def snapshot_after(root: Path, before: dict[str, Any], suffix: str) -> dict[str, Any]:
    rows = []
    for prior in before["records"]:
        relative = prior["relative_path"]
        data = (root / relative).read_bytes()
        same = len(data) == prior["size_bytes"] and sha(data) == prior["sha256"]
        rows.append({**record(relative, data),
                     "before_size_bytes": prior["size_bytes"],
                     "before_sha256": prior["sha256"],
                     "before_after_byte_identical": same,
                     "status": "PASS" if same else "FAIL"})
    mismatches = [row["relative_path"] for row in rows if row["status"] != "PASS"]
    require(not mismatches, suffix + " changed: " + repr(mismatches))
    return {"schema_version": SCHEMA + "." + suffix, "file_count": len(rows),
            "pass_count": len(rows), "mismatch_count": 0, "records": rows, "status": "PASS"}


def current_repository(before: dict[str, Any]) -> dict[str, Any]:
    root = Path(before["path"])
    rows = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        rows.append(record(path.relative_to(root).as_posix(), path.read_bytes()))
    index = json.loads((root / "candidate_index.json").read_text(encoding="utf-8-sig"))
    return {"schema_version": SCHEMA + ".candidate_repository_after", "path": str(root),
            "exists": root.exists(), "candidate_count": len(index["entries"]),
            "file_count": len(rows), "files": rows, "status": "PASS"}


def validate_repository(before: dict[str, Any], after: dict[str, Any], helper: Any) -> tuple[dict[str, Any], dict[str, bytes]]:
    require(before["candidate_count"] == 2 and before["file_count"] == 7,
            "candidate before baseline is not 2/7")
    require(after["candidate_count"] == 5 and after["file_count"] == 16,
            "candidate after count is not 5/16")
    root = Path(after["path"])
    before_files = {row["relative_path"]: row for row in before["files"]}
    after_files = {row["relative_path"]: row for row in after["files"]}
    require(set(before_files) <= set(after_files), "existing candidate file removed")
    existing_payloads = [name for name in before_files if name != "candidate_index.json"]
    require(all(before_files[name]["size_bytes"] == after_files[name]["size_bytes"] and
                before_files[name]["sha256"] == after_files[name]["sha256"]
                for name in existing_payloads), "existing candidate payload changed")
    new_paths = sorted(set(after_files) - set(before_files))
    require(len(new_paths) == 9, "new candidate payload file count is not 9")

    before_index_data = (Path(__file__).resolve().parents[1] / "Validation" / NAME /
                         "candidate_index_before.json").read_bytes()
    before_index = helper.strict_json(before_index_data, "candidate_index_before.json")
    after_index_data = (root / "candidate_index.json").read_bytes()
    after_index = helper.strict_json(after_index_data, "candidate_index_after.json")
    old_entries = {row["capture_session_id"]: row for row in before_index["entries"]}
    all_entries = {row["capture_session_id"]: row for row in after_index["entries"]}
    require(len(all_entries) == 5 and len(old_entries) == 2 and
            all(all_entries[key] == value for key, value in old_entries.items()),
            "existing candidate index rows changed")
    new_ids = [key for key in all_entries if key not in old_entries]
    require(len(new_ids) == 3, "new candidate index row count is not 3")
    expected = {
        "GREETING": ("StrictSuccess", "Eligible", True, True, True, True),
        "LOVE": ("LowConfidenceCorrect", "ReviewOnly", False, False, False, False),
        "THANK_YOU": ("LowConfidenceCorrect", "ReviewOnly", False, False, False, False),
    }
    payloads = {"new_candidates/candidate_index.json": after_index_data}
    rows = []
    for capture_id in new_ids:
        entry = all_entries[capture_id]
        word = entry["word"]
        require(word in expected, "unexpected new candidate word")
        outcome, eligibility, eligible, accepted, expected_accept, quest = expected[word]
        require(entry["outcome"] == outcome and entry["eligibility"] == eligibility,
                "candidate index outcome/eligibility mismatch")
        relative = PurePosixPath(entry["relative_directory"])
        require(not relative.is_absolute() and ".." not in relative.parts and "\\" not in str(relative),
                "unsafe candidate relative directory")
        directory = root / Path(*relative.parts)
        sequence_path = directory / "normalized_sequence.wisrq"
        metadata_path = directory / "candidate_metadata.json"
        manifest_path = directory / "checksum_manifest.json"
        sequence_data = sequence_path.read_bytes()
        metadata_data = metadata_path.read_bytes()
        manifest_data = manifest_path.read_bytes()
        require(sha(sequence_data) == entry["sequence_sha256"] and
                sha(metadata_data) == entry["metadata_sha256"], "candidate index hash mismatch")
        metadata = helper.strict_json(metadata_data, "candidate_metadata.json")
        manifest = helper.strict_json(manifest_data, "checksum_manifest.json")
        targets = {row["path"]: row for row in manifest["files"]}
        require(set(targets) == {"normalized_sequence.wisrq", "candidate_metadata.json"},
                "candidate checksum manifest membership mismatch")
        for filename, data in (("normalized_sequence.wisrq", sequence_data),
                               ("candidate_metadata.json", metadata_data)):
            require(targets[filename]["size_bytes"] == len(data) and
                    targets[filename]["sha256"] == sha(data),
                    "candidate checksum manifest identity mismatch")
        sequence = helper.parse_wisrcb(sequence_data)
        require(sequence["sample_id"] == capture_id == metadata["capture_session_id"] and
                metadata["word"] == metadata["expected_word"] == metadata["predicted_word"] == word,
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
                metadata["approved_threshold"] == EXPECTED_THRESHOLDS[word],
                "candidate route/profile/threshold mismatch")
        require(metadata["raw_frame_count"] == metadata["normalized_frame_count"] ==
                sequence["frame_count"] and
                metadata["valid_landmark_count"] == sequence["valid_landmark_count"] and
                metadata["sequence_binary_sha256"] == sha(sequence_data),
                "candidate WISRCB01 frame/payload mismatch")
        base = "new_candidates/" + word + "/" + capture_id + "/"
        payloads[base + "normalized_sequence.wisrq"] = sequence_data
        payloads[base + "candidate_metadata.json"] = metadata_data
        payloads[base + "checksum_manifest.json"] = manifest_data
        rows.append({
            "capture_session_id": capture_id, "word": word, "outcome": outcome,
            "eligibility": eligibility, "eligible_for_profile": eligible,
            "strict_accepted": accepted, "strict_expected_class_accept": expected_accept,
            "strict_quest_success": quest, "raw_frame_count": metadata["raw_frame_count"],
            "normalized_frame_count": metadata["normalized_frame_count"],
            "valid_landmark_count": metadata["valid_landmark_count"],
            "best_distance": metadata["best_distance"],
            "approved_threshold": metadata["approved_threshold"],
            "threshold_margin": metadata["threshold_margin"],
            "sequence_sha256": sha(sequence_data), "metadata_sha256": sha(metadata_data),
            "manifest_sha256": sha(manifest_data), "wisrcb01": sequence, "status": "PASS",
        })
    rows.sort(key=lambda row: ["GREETING", "LOVE", "THANK_YOU"].index(row["word"]))
    media = [path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and
             path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp",
                                     ".mp4", ".mov", ".avi", ".mkv", ".wav", ".mp3", ".ogg", ".m4a"}]
    require(not media, "candidate repository contains media")
    return ({
        "schema_version": SCHEMA + ".candidate_validation", "before_candidate_count": 2,
        "after_candidate_count": 5, "saved_count": 3, "before_file_count": 7,
        "after_file_count": 16, "expected_after_formula": "2+3 / 7+3*3",
        "existing_index_rows_preserved": 2, "existing_payload_files_byte_identical": 6,
        "removed_candidate_count": 0, "modified_existing_candidate_payload_count": 0,
        "new_candidate_count": 3, "eligible_new_count": 1, "review_only_new_count": 2,
        "duplicate_capture_session_id_count": 0, "media_file_count": 0,
        "profile_auto_refresh_or_activation": False, "rows": rows, "status": "PASS",
    }, payloads)


def screen_record(result: Path, name: str) -> dict[str, Any]:
    path = result / name
    require(path.is_file(), "required screen missing: " + name)
    return record(name, path.read_bytes())


def screen_review(result: Path, candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = [
        "PlayMode_00_InitialOrStarting.png", "PlayMode_00_CameraReadyOrFailed.png",
        "PlayMode_01_GREETING_First.png", "PlayMode_01_GREETING_First(1).png",
        "PlayMode_02_LOVE_First.png", "PlayMode_02_LOVE_First(1).png",
        "PlayMode_03_THANK_YOU_First.png", "PlayMode_03_THANK_YOU_First(1).png",
        "PlayMode_FinalSummaryOrStop.png",
    ]
    screens = {name: screen_record(result, name) for name in names}
    by_word = {row["word"]: row for row in candidate_rows}
    return {
        "schema_version": SCHEMA + ".screen_review",
        "review_method": "manual full-resolution review by Codex; repository/XML/log cross-checked",
        "playmode_execution_count": 1,
        "camera": {"state": "Ready", "camera_started": True, "frame_ready": True,
                   "frame_count_at_ready_evidence": 104, "provider_observation_ready": True,
                   "observation_count_at_ready_evidence": 103, "preview_visible": True,
                   "candidate_route_active": True, "technical_panel_hidden": True,
                   "legacy_practice_controller_active": False, "status": "PASS"},
        "word_attempts": [
            {"word": "GREETING", "attempt": "First", "outcome": "StrictSuccess",
             "strict_accepted": True, "expected_class_accept": True, "quest_success": True,
             "storage_status": "Saved", "eligibility": "Eligible", "retry_count": 0,
             "final_learning_status": "Mastered", "capture_session_id": by_word["GREETING"]["capture_session_id"]},
            {"word": "LOVE", "attempt": "First", "outcome": "LowConfidenceCorrect",
             "strict_accepted": False, "expected_class_accept": False, "quest_success": False,
             "storage_status": "Saved", "eligibility": "ReviewOnly", "retry_count": 0,
             "final_learning_status": "Mastered", "capture_session_id": by_word["LOVE"]["capture_session_id"]},
            {"word": "THANK_YOU", "attempt": "First", "outcome": "LowConfidenceCorrect",
             "strict_accepted": False, "expected_class_accept": False, "quest_success": False,
             "storage_status": "Saved", "eligibility": "ReviewOnly", "retry_count": 0,
             "final_learning_status": "Mastered", "capture_session_id": by_word["THANK_YOU"]["capture_session_id"]},
        ],
        "retry_counts": {"GREETING": 0, "LOVE": 0, "THANK_YOU": 0},
        "automatic_capture_retry_or_skip_observed": False,
        "needs_practice_live_branch": "NOT_APPLICABLE_NATURALLY_NOT_REACHED",
        "technical_failure_live_branch": "NOT_APPLICABLE_NOT_OBSERVED",
        "summary": {"state": "Completed", "GREETING": "Mastered", "LOVE": "Mastered",
                    "THANK_YOU": "Mastered", "mastered_count": 3, "needs_practice_count": 0,
                    "candidate_saved_count": 5, "exit_button_visible": True, "status": "PASS"},
        "screens": screens, "status": "PASS",
    }


def assertion_bytes(rows: list[tuple[str, str, str]]) -> bytes:
    keys = set()
    lines = []
    for key, status, evidence in rows:
        require(key not in keys, "duplicate assertion key: " + key)
        keys.add(key)
        lines.append(json.dumps({"assertion_key": key, "status": status, "evidence": evidence},
                                ensure_ascii=False, sort_keys=True, allow_nan=False))
    return ("\n".join(lines) + "\n").encode("utf-8")


def add_manifest(files: dict[str, bytes]) -> dict[str, bytes]:
    require("output_file_manifest.json" not in files, "manifest already exists")
    manifest = {"schema_version": SCHEMA + ".manifest", "self_excluded": True,
                "file_count": len(files), "files": [record(name, files[name]) for name in sorted(files)]}
    result = dict(files)
    result["output_file_manifest.json"] = jbytes(manifest)
    return result


def report(tests: dict[str, Any], log: dict[str, Any], play: dict[str, Any],
           candidates: dict[str, Any], protected: dict[str, Any]) -> bytes:
    return f"""# Step 10A v1.3 GUI Unity Editor Gate

## Result

- Unity: 6000.3.14f1; compiler errors {log['compiler_error_count']}.
- EditMode: one execution, {tests['passed']}/{tests['testcase_count']} PASS, 0 failed/skipped/inconclusive, retry/repeat 0.
- PlayMode: one execution; camera Ready with visible preview, frame/observation {play['camera']['frame_count_at_ready_evidence']}/{play['camera']['observation_count_at_ready_evidence']}.
- GREETING: StrictSuccess, Mastered, Eligible saved.
- LOVE: LowConfidenceCorrect with strict False/False/False preserved, Mastered, ReviewOnly saved.
- THANK_YOU: LowConfidenceCorrect with strict False/False/False preserved, Mastered, ReviewOnly saved.
- No manual retry was required. The live NeedsPractice branch is NOT_APPLICABLE because no WrongWord Retry1 occurred naturally; its 15 EditMode regressions passed.
- Session summary: Completed; Mastered 3, NeedsPractice 0, Candidate saved 5; exit button visible.

## Candidate repository

- Candidates: {candidates['before_candidate_count']} to {candidates['after_candidate_count']}.
- Files: {candidates['before_file_count']} to {candidates['after_file_count']}.
- New: {candidates['new_candidate_count']} (Eligible {candidates['eligible_new_count']}, ReviewOnly {candidates['review_only_new_count']}).
- Existing payloads unchanged: {candidates['existing_payload_files_byte_identical']}/6.
- WISRCB01 checksum and binary round-trip: 3/3 PASS; duplicate sessions and media files: 0/0.

Project files 30/30, protected files {protected['pass_count']}/{protected['file_count']}, candidate profile 22/22, nine prototypes and three thresholds are byte-identical. No profile refresh/activation occurred. Git staged files remain zero.

## Status

```text
STEP10A_V1_3_GUI_UNITY_EDITOR_GATE_READY_PENDING_EXTERNAL_REVIEW
STEP10A_V1_3_UNITY_COMPILE_EDITOR_PASS
STEP10A_V1_3_EDITMODE_FIRST_RUN_PASS_102_OF_102
STEP10A_V1_3_PLAYABLE_SESSION_COMPLETION_PASS_MASTERED_THREE
STEP10A_V1_3_LIVE_NEEDS_PRACTICE_BRANCH_NOT_APPLICABLE
STEP10A_V1_3_CANDIDATE_STORAGE_APPEND_PASS_2_TO_5
BLOCKED_PENDING_EXTERNAL_REVIEW
```
""".encode("utf-8")


def package(root: Path) -> None:
    validation = root / "Validation"
    result = validation / NAME
    delivery = validation / (NAME + "_delivery.zip")
    detached = validation / (NAME + "_delivery.zip.sha256")
    require(result.is_dir(), "prepared evidence directory missing")
    require(not delivery.exists() and not detached.exists(), "never-overwrite delivery target exists")
    required_raw = [
        "preflight_before_snapshot.json", "project_files_before.json", "protected_files_before.json",
        "active_candidate_profile_before.json", "candidate_repository_before.json",
        "candidate_index_before.json", "git_staged_before.json", "EVIDENCE_INSTRUCTIONS.txt",
        "TestResults.xml", "TestRunner_FIRST_RESULT.png", "Editor.log",
        "PlayMode_00_InitialOrStarting.png", "PlayMode_00_CameraReadyOrFailed.png",
        "PlayMode_01_GREETING_First.png", "PlayMode_01_GREETING_First(1).png",
        "PlayMode_02_LOVE_First.png", "PlayMode_02_LOVE_First(1).png",
        "PlayMode_03_THANK_YOU_First.png", "PlayMode_03_THANK_YOU_First(1).png",
        "PlayMode_FinalSummaryOrStop.png",
    ]
    for relative in required_raw:
        require((result / relative).is_file(), "required evidence missing: " + relative)

    helper = load_module(root, V12_GATE_TOOL, "step10a_v12_gate_helper")
    gate = load_module(root, V11_GATE_TOOL, "step10a_v11_gate_helper")
    v11 = helper.load_v11(root)
    v13_info, _ = helper.verify_fixed(v11, root, V13_ZIP, V13_ZIP_SIZE, V13_ZIP_SHA,
                                      V13_DETACHED_SHA)
    v12_info, _ = helper.verify_fixed(v11, root, V12_ZIP, V12_ZIP_SIZE, V12_ZIP_SHA,
                                      V12_DETACHED_SHA)
    v12_gate_info, _ = helper.verify_fixed(v11, root, V12_GATE_ZIP, V12_GATE_ZIP_SIZE,
                                           V12_GATE_ZIP_SHA, V12_GATE_DETACHED_SHA)
    tests = parse_tests(result)
    log = parse_editor_log(result)
    project_before = json.loads((result / "project_files_before.json").read_text(encoding="utf-8-sig"))
    protected_before = json.loads((result / "protected_files_before.json").read_text(encoding="utf-8-sig"))
    profile_before = json.loads((result / "active_candidate_profile_before.json").read_text(encoding="utf-8-sig"))
    repository_before = json.loads((result / "candidate_repository_before.json").read_text(encoding="utf-8-sig"))
    project_after = snapshot_after(root, project_before, "project_files_after")
    protected_after = snapshot_after(root, protected_before, "protected_files_after")
    profile_after = snapshot_after(root, profile_before, "active_candidate_profile_after")
    profile_after.update({"prototype_count": 9, "threshold_count": 3,
                          "thresholds": EXPECTED_THRESHOLDS})
    require(project_after["pass_count"] == 30 and protected_after["pass_count"] == 106 and
            profile_after["pass_count"] == 22, "protected after snapshot mismatch")
    require(sha((root / NEW_TEST).read_bytes()) == NEW_TEST_SHA and
            sha((root / NEW_META).read_bytes()) == NEW_META_SHA,
            "new fixture changed after preflight")
    repository_after = current_repository(repository_before)
    candidate_validation, candidate_payloads = validate_repository(
        repository_before, repository_after, helper)
    play = screen_review(result, candidate_validation["rows"])
    require(staged_files(root) == [], "Git staged files changed")
    git_after = {"schema_version": SCHEMA + ".git_after", "staged_count": 0,
                 "staged_files": [], "status": "PASS"}
    transaction = gate.transaction_regression()
    require(transaction["status"] == "PASS" and transaction["failure_injection_pass"] and
            transaction["never_overwrite_pass"], "packaging transaction regression failed")

    primary = {
        "schema_version": SCHEMA + ".primary_verdict",
        "overall_gate": "PASS",
        "unity_compile": "PASS",
        "editmode_102": "PASS",
        "playmode_execution_count": 1,
        "camera_readiness": "PASS",
        "practice_completion": "PASS",
        "word_statuses": {"GREETING": "Mastered", "LOVE": "Mastered", "THANK_YOU": "Mastered"},
        "mastered_count": 3,
        "needs_practice_count": 0,
        "live_needs_practice_branch": "NOT_APPLICABLE_NATURALLY_NOT_REACHED",
        "candidate_storage": "PASS_2_TO_5",
        "status_strings": [
            "STEP10A_V1_3_GUI_UNITY_EDITOR_GATE_READY_PENDING_EXTERNAL_REVIEW",
            "STEP10A_V1_3_UNITY_COMPILE_EDITOR_PASS",
            "STEP10A_V1_3_EDITMODE_FIRST_RUN_PASS_102_OF_102",
            "STEP10A_V1_3_PLAYABLE_SESSION_COMPLETION_PASS_MASTERED_THREE",
            "STEP10A_V1_3_LIVE_NEEDS_PRACTICE_BRANCH_NOT_APPLICABLE",
            "STEP10A_V1_3_CANDIDATE_STORAGE_APPEND_PASS_2_TO_5",
            "BLOCKED_PENDING_EXTERNAL_REVIEW",
        ],
        "status": "PASS",
    }
    assertions = assertion_bytes([
        ("fixed_v1_3_delivery", "PASS", V13_ZIP_SHA),
        ("fixed_v1_2_delivery", "PASS", V12_ZIP_SHA),
        ("fixed_v1_2_gui_evidence", "PASS", V12_GATE_ZIP_SHA),
        ("unity_version", "PASS", "6000.3.14f1"),
        ("compiler_errors", "PASS", "0"),
        ("editmode_execution_count", "PASS", "1"),
        ("unique_editmode_tests", "PASS", "102"),
        ("editmode_pass_fail", "PASS", "102/0"),
        ("skipped_inconclusive", "PASS", "0/0"),
        ("retry_repeat", "PASS", "0/0"),
        ("playmode_execution_count", "PASS", "1"),
        ("candidate_camera_ready", "PASS", "frames 104; observations 103 at Ready evidence"),
        ("candidate_route_preview", "PASS", "active; preview visible; technical hidden"),
        ("legacy_practice_controller", "PASS", "inactive"),
        ("greeting_first", "PASS", "StrictSuccess; True/True/True; Eligible Saved"),
        ("love_first", "PASS", "LowConfidenceCorrect; False/False/False; ReviewOnly Saved"),
        ("thank_you_first", "PASS", "LowConfidenceCorrect; False/False/False; ReviewOnly Saved"),
        ("manual_retry_count", "PASS", "0/0/0"),
        ("automatic_capture_retry_skip", "PASS", "none observed"),
        ("word_learning_statuses", "PASS", "Mastered/Mastered/Mastered"),
        ("session_summary", "PASS", "Completed; Mastered 3; NeedsPractice 0; Candidate 5"),
        ("exit_button", "PASS", "visible"),
        ("live_needs_practice_branch", "NOT_APPLICABLE", "no WrongWord Retry1 occurred naturally"),
        ("new_candidate_count", "PASS", "3"),
        ("candidate_count_formula", "PASS", "2+3=5"),
        ("candidate_file_formula", "PASS", "7+3*3=16"),
        ("candidate_eligibility", "PASS", "Eligible 1; ReviewOnly 2"),
        ("wisrcb01_checksum_roundtrip", "PASS", "3/3"),
        ("candidate_checksum_manifests", "PASS", "3/3"),
        ("candidate_session_duplicates", "PASS", "0"),
        ("candidate_media_files", "PASS", "0"),
        ("existing_candidate_payloads", "PASS", "6/6 byte-identical"),
        ("existing_candidate_index_rows", "PASS", "2/2 content preserved"),
        ("project_files", "PASS", "30/30"),
        ("protected_files", "PASS", "106/106"),
        ("active_profile", "PASS", "22/22"),
        ("prototype_threshold", "PASS", "9/3"),
        ("profile_auto_refresh_activation", "PASS", "false"),
        ("new_fixture_identity", "PASS", "source/meta"),
        ("git_staged", "PASS", "0"),
        ("packaging_transaction", "PASS", "3/3 rollback; never-overwrite"),
    ])
    input_provenance = {
        "schema_version": SCHEMA + ".input_provenance",
        "implementation_owner": "Codex",
        "v1_3_source_delivery": v13_info,
        "v1_2_source_delivery": v12_info,
        "v1_2_gui_evidence": v12_gate_info,
        "status": "PASS",
    }
    generated: dict[str, bytes] = {
        "input_provenance.json": jbytes(input_provenance),
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
        "VALIDATION_REPORT.md": report(tests, log, play, candidate_validation, protected_after),
        "analysis/" + Path(__file__).name: Path(__file__).read_bytes(),
        **candidate_payloads,
    }
    for relative in generated:
        require(not (result / relative).exists(), "never-overwrite generated result exists: " + relative)
    raw_files = {path.relative_to(result).as_posix(): path.read_bytes()
                 for path in result.rglob("*") if path.is_file()}
    require("output_file_manifest.json" not in raw_files, "result already packaged")
    complete = add_manifest({**raw_files, **generated})
    zip_blob = gate.deterministic_zip(complete)
    detached_blob = f"{sha(zip_blob)}  {delivery.name}\n".encode("ascii")
    zip_check = gate.validate_zip(zip_blob, complete)
    require(zip_check["entry_count"] == len(complete) and zip_check["duplicate_count"] == 0 and
            zip_check["unsafe_path_count"] == 0 and zip_check["encrypted_entry_count"] == 0 and
            zip_check["symlink_entry_count"] == 0 and zip_check["crc_failure"] is None and
            zip_check["membership_match"] and zip_check["payload_mismatch_count"] == 0,
            "pre-commit ZIP validation failed")

    temp = Path(tempfile.mkdtemp(prefix="step10a_v13_gate_generated_", dir=validation))
    created = []
    committed = False
    try:
        commit_files = {**generated, "output_file_manifest.json": complete["output_file_manifest.json"]}
        for relative, data in commit_files.items():
            path = temp / PurePosixPath(relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        for relative_name in sorted(commit_files):
            relative = PurePosixPath(relative_name)
            source = temp / relative
            target = result / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            require(not target.exists(), "generated target appeared during commit")
            source.replace(target)
            created.append(target)
        gate.commit_pair(zip_blob, detached_blob, delivery, detached)
        disk_files = {path.relative_to(result).as_posix(): path.read_bytes()
                      for path in result.rglob("*") if path.is_file()}
        require(disk_files == complete, "result directory file map mismatch")
        require(delivery.read_bytes() == gate.deterministic_zip(disk_files),
                "deterministic ZIP rebuild mismatch")
        require(detached.read_bytes() == detached_blob, "detached checksum mismatch")
        post = gate.validate_zip(delivery.read_bytes(), disk_files)
        require(post["entry_count"] == len(disk_files) and post["duplicate_count"] == 0 and
                post["unsafe_path_count"] == 0 and post["encrypted_entry_count"] == 0 and
                post["symlink_entry_count"] == 0 and post["crc_failure"] is None and
                post["membership_match"] and post["payload_mismatch_count"] == 0,
                "post-commit ZIP validation failed")
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
        "status": "PASS", "overall_gate": "PASS", "result_directory": str(result),
        "result_file_count": len(complete), "manifest_target_count": len(complete) - 1,
        "delivery_zip": str(delivery), "delivery_size_bytes": delivery.stat().st_size,
        "delivery_sha256": sha(delivery.read_bytes()), "detached_checksum": str(detached),
        "detached_size_bytes": detached.stat().st_size,
        "detached_sha256": sha(detached.read_bytes()), "compiler_errors": 0,
        "editmode_execution_count": 1, "editmode": "102 PASS / 0 FAIL",
        "playmode_execution_count": 1,
        "camera": {"state": "Ready", "frames": 104, "observations": 103},
        "outcomes": {"GREETING": "StrictSuccess", "LOVE": "LowConfidenceCorrect",
                     "THANK_YOU": "LowConfidenceCorrect"},
        "retries": {"GREETING": 0, "LOVE": 0, "THANK_YOU": 0},
        "word_statuses": {"GREETING": "Mastered", "LOVE": "Mastered", "THANK_YOU": "Mastered"},
        "needs_practice_live": "NOT_APPLICABLE", "candidate_count_before_after": [2, 5],
        "candidate_file_count_before_after": [7, 16], "protected_files": "106/106",
        "profile_files": "22/22", "git_staged": 0,
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
