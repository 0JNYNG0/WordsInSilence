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
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA = "WordsInSilence.Step10D.ExistingFbxAnimatorCompanion.v1.0.GuiUnityEditorGate"
NAME = (
    "WordsInSilence_step10d_existing_fbx_animator_floating_pet_robot_"
    "companion_presentation_layer_v1_0_gui_unity_editor_gate"
)

V1_ZIP = (
    "Validation/WordsInSilence_step10d_existing_fbx_animator_floating_pet_robot_"
    "companion_presentation_layer_v1_0_delivery.zip"
)
V1_ZIP_SIZE = 19_270_682
V1_ZIP_SHA256 = "2c51fb71d6bf39057f82ab66a68a6e5ee9863b553d81a6535740d1864f53ae52"
V1_DETACHED_SIZE = 177
V1_DETACHED_SHA256 = "4e25ebc23a0f6bd1253a8bd03fb4c2f9bee95f6dce607a68ac4dbe27b0f3b6a3"

CHECKPOINT = (
    "Validation/WordsInSilence_step10d_existing_fbx_animator_floating_pet_robot_"
    "companion_presentation_layer_v1_0/asset_identity_and_guid_graph.json"
)
ZIP_HELPER = (
    "Tools/s001_step10a_pet_robot_candidate_camera_startup_route_"
    "v1_1_gui_unity_editor_gate.py"
)
TX_HELPER = "Tools/s001_step10a_pet_robot_candidate_camera_startup_route_v1_1.py"

FIXED_PROJECT_FILES = {
    "Assets/_Project/Resources/VerticalSlice/PetRobotPortfolioQuestPresentation.prefab":
        "4fd26a3ed6b1b3aacdfc1262eb73e1b136b1bcbd63849c59dcccd9e58dfde602",
    "Assets/_Project/Scenes/VerticalSlice/PetRobotPortfolioQuestVerticalSlice.unity":
        "a4696cadc0e688e24807f8b00d5508bec029a4da81a08883e4c3bfb7e3cea7fe",
    "Assets/_Project/Scripts/Practice/PetRobotPortfolioQuest/PetRobotPortfolioQuestContracts.cs":
        "b04e9b338c60db7a40bad96be5b80e99e52b687218406b63a05f1e7b33846ae2",
    "Assets/_Project/Tests/EditMode/PetRobotPortfolioQuestPresentationV1Tests.cs":
        "ea1e367494ab226244ea30268f6f66c3610cb2ead425fb19485e97b5f1c27f92",
    "Assets/_Project/Scripts/Practice/PetRobotPortfolioQuest/PetRobotFloatingCompanionController.cs":
        "b0278e8ca26a666c48ca73c59529200c1fb61ffab4af7c1e933cb65c860889e9",
    "Assets/_Project/Scripts/Practice/PetRobotPortfolioQuest/PetRobotFloatingCompanionController.cs.meta":
        "f88ad4429e8e0ee4c8c4672b6f614fec629550774c94c3210cbe3661deca3809",
    "Assets/_Project/Resources/VerticalSlice/PetRobotAnimatedVisualWrapper.prefab":
        "ce33933e637890a2332628f3c43675b4d803bb69f276bda57e624f0b359ec984",
    "Assets/_Project/Resources/VerticalSlice/PetRobotAnimatedVisualWrapper.prefab.meta":
        "e81319aa0a032e04442c0b20694bcdf9bbacbc3ea822718cf0f2a93dcf484e3e",
    "Assets/_Project/Tests/EditMode/PetRobotFloatingCompanionPresentationV1Tests.cs":
        "4cfeccec7c4f5f86a3fa8f1377b495075a01d8dfd30bf70eaaad61b93dd872dd",
    "Assets/_Project/Tests/EditMode/PetRobotFloatingCompanionPresentationV1Tests.cs.meta":
        "25aab8ff47e163defcdeba3deea693f88cbca73f841bef171846c6e645c10743",
}

EXPECTED_FIXTURES = {
    "CandidateTransientAmbiguousHandIdentityToleranceV1Tests": 8,
    "GuidedQuestLowConfidenceOutcomeLayerV1Tests": 10,
    "PersonalLiveRecognitionIntegrationV1Tests": 22,
    "PersonalMultiPrototypeLiveOptInRoutingV1Tests": 8,
    "PersonalMultiPrototypeRuntimeIntegrationV1Tests": 8,
    "PetRobotCandidateCameraStartupRouteV1Tests": 8,
    "PetRobotFloatingCompanionPresentationV1Tests": 14,
    "PetRobotPortfolioQuestPresentationV1Tests": 18,
    "PetRobotRetryExhaustionNeedsPracticeCompletionV1Tests": 15,
    "PetRobotSignPracticeVerticalSliceV1Tests": 23,
    "PetRobotWorldEntryReturnIntegrationV1Tests": 18,
}

RAW_EVIDENCE = {
    "01_test_runner_152.png",
    "scene_error.png",
    "Step10D_TestResults_152.xml",
    "Editor.log",
}

SCENE_ERROR = (
    "PPtr cast failed when dereferencing! Casting from PrefabInstance to Prefab "
    "at FileID 100100000!"
)
SCENE_PATH = "Assets/_Project/Scenes/VerticalSlice/PetRobotPortfolioQuestVerticalSlice.unity"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def jbytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def record(relative_path: str, data: bytes, **extra: Any) -> dict[str, Any]:
    return {
        "relative_path": relative_path,
        "size_bytes": len(data),
        "sha256": sha(data),
        **extra,
    }


def load_module(root: Path, relative_path: str, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, root / relative_path)
    require(spec is not None and spec.loader is not None,
            "cannot load helper: " + relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def staged_files(root: Path) -> list[str]:
    run = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    require(run.returncode == 0, "Git staged query failed")
    return [line.strip() for line in run.stdout.splitlines() if line.strip()]


def verify_v1_identity(root: Path) -> dict[str, Any]:
    zip_path = root / V1_ZIP
    detached = Path(str(zip_path) + ".sha256")
    require(zip_path.is_file() and detached.is_file(), "fixed v1.0 delivery missing")
    zip_blob = zip_path.read_bytes()
    detached_blob = detached.read_bytes()
    require(len(zip_blob) == V1_ZIP_SIZE and sha(zip_blob) == V1_ZIP_SHA256,
            "fixed v1.0 ZIP identity mismatch")
    require(len(detached_blob) == V1_DETACHED_SIZE and
            sha(detached_blob) == V1_DETACHED_SHA256,
            "fixed v1.0 detached identity mismatch")
    line = detached_blob.decode("ascii").strip().split()
    require(len(line) == 2 and line[0] == V1_ZIP_SHA256 and
            line[1] == zip_path.name, "fixed v1.0 detached content mismatch")
    return {
        "zip": record(V1_ZIP, zip_blob),
        "detached_checksum": record(V1_ZIP + ".sha256", detached_blob),
        "status": "PASS",
    }


def fixed_project_current(root: Path) -> dict[str, Any]:
    rows = []
    for relative, expected in sorted(FIXED_PROJECT_FILES.items()):
        path = root / relative
        require(path.is_file(), "fixed project file missing: " + relative)
        data = path.read_bytes()
        actual = sha(data)
        rows.append({
            "relative_path": relative,
            "expected_sha256": expected,
            "current_size_bytes": len(data),
            "current_sha256": actual,
            "byte_identical_to_approved_v1_0": actual == expected,
            "status": "PASS" if actual == expected else "FAIL",
        })
    require(all(row["status"] == "PASS" for row in rows),
            "fixed project file mismatch")
    return {
        "schema_version": SCHEMA + ".fixed_project_current",
        "target_count": len(rows),
        "match_count": sum(row["status"] == "PASS" for row in rows),
        "mismatch_count": sum(row["status"] != "PASS" for row in rows),
        "rows": rows,
        "status": "PASS",
    }


def checkpoint_current(root: Path) -> dict[str, Any]:
    checkpoint_path = root / CHECKPOINT
    require(checkpoint_path.is_file(), "post-Step10D checkpoint missing")
    checkpoint_blob = checkpoint_path.read_bytes()
    checkpoint = json.loads(checkpoint_blob.decode("utf-8-sig"))
    source_rows = checkpoint["gameassets_dependencies"]
    require(len(source_rows) == 242, "checkpoint dependency count mismatch")
    paths = [row["relative_path"] for row in source_rows]
    require(len(paths) == len(set(paths)) == 242, "checkpoint path uniqueness mismatch")
    rows = []
    for source in source_rows:
        relative = source["relative_path"]
        path = root / relative
        exists = path.is_file()
        data = path.read_bytes() if exists else b""
        current_size = len(data) if exists else None
        current_sha = sha(data) if exists else None
        match = (exists and current_size == int(source["size_bytes"]) and
                 current_sha == source["sha256"])
        rows.append({
            "relative_path": relative,
            "checkpoint_size_bytes": int(source["size_bytes"]),
            "checkpoint_sha256": source["sha256"],
            "current_size_bytes": current_size,
            "current_sha256": current_sha,
            "unchanged_since_post_step10d_checkpoint": match,
            "verification_status": "PASS" if match else "FAIL",
        })
    mismatch = [row for row in rows if row["verification_status"] != "PASS"]
    require(not mismatch, "GameAssets current/checkpoint mismatch")
    return {
        "schema_version": SCHEMA + ".gameassets_checkpoint_current",
        "checkpoint": record(CHECKPOINT, checkpoint_blob),
        "checkpoint_semantics": "post-Step10D current checkpoint only",
        "target_count": 242,
        "match_count": 242,
        "mismatch_count": 0,
        "rows": rows,
        "retroactive_preimplementation_preservation": "UNVERIFIED_NONBLOCKING",
        "status": "PASS_POST_STEP10D_CHECKPOINT_UNCHANGED",
    }


def parse_tests(result: Path) -> dict[str, Any]:
    relative = "Step10D_TestResults_152.xml"
    raw = (result / relative).read_bytes()
    root = ET.fromstring(raw)
    require(root.tag == "test-run", "TestResults root mismatch")
    cases = root.findall(".//test-case")
    names = [case.attrib["fullname"] for case in cases]
    fixtures = Counter(case.attrib["classname"].split(".")[-1] for case in cases)
    retries = []
    repeats = []
    for case in cases:
        properties = {
            node.attrib.get("name"): node.attrib.get("value")
            for node in case.findall("./properties/property")
        }
        retries.append(int(properties.get("retryIteration", "0")))
        repeats.append(int(properties.get("repeatIteration", "0")))
    summary = {
        "schema_version": SCHEMA + ".test_results",
        "source": record(relative, raw),
        "execution_count": 1,
        "total": int(root.attrib.get("total", "-1")),
        "passed": int(root.attrib.get("passed", "-1")),
        "failed": int(root.attrib.get("failed", "-1")),
        "skipped": int(root.attrib.get("skipped", "-1")),
        "inconclusive": int(root.attrib.get("inconclusive", "-1")),
        "testcase_count": len(cases),
        "unique_testcase_count": len(set(names)),
        "duplicate_testcase_count": len(names) - len(set(names)),
        "retry_nonzero_count": sum(value != 0 for value in retries),
        "repeat_nonzero_count": sum(value != 0 for value in repeats),
        "fixture_counts": dict(sorted(fixtures.items())),
        "expected_fixture_counts": EXPECTED_FIXTURES,
        "start_utc": root.attrib.get("start-time"),
        "end_utc": root.attrib.get("end-time"),
    }
    require(summary["total"] == summary["passed"] == 152 and
            summary["failed"] == summary["skipped"] == summary["inconclusive"] == 0,
            "TestResults aggregate mismatch")
    require(summary["testcase_count"] == summary["unique_testcase_count"] == 152 and
            summary["duplicate_testcase_count"] == 0,
            "TestResults testcase uniqueness mismatch")
    require(summary["retry_nonzero_count"] == summary["repeat_nonzero_count"] == 0,
            "TestResults retry/repeat mismatch")
    require(dict(fixtures) == EXPECTED_FIXTURES, "TestResults fixture inventory mismatch")
    require(all(case.attrib.get("result") == "Passed" for case in cases),
            "TestResults contains non-passed testcase")
    summary["status"] = "PASS"
    return summary


def png_record(result: Path, relative: str) -> dict[str, Any]:
    raw = (result / relative).read_bytes()
    require(raw.startswith(b"\x89PNG\r\n\x1a\n") and len(raw) >= 24,
            "invalid PNG: " + relative)
    width = int.from_bytes(raw[16:20], "big")
    height = int.from_bytes(raw[20:24], "big")
    return record(relative, raw, width=width, height=height, status="PASS")


def parse_editor_log(result: Path) -> dict[str, Any]:
    relative = "Editor.log"
    raw = (result / relative).read_bytes()
    text = raw.decode("utf-8", errors="replace")
    compiler_errors = len(re.findall(r"(?mi)^.*error CS\d{4}.*$", text))
    error_count = text.count(SCENE_ERROR)
    scene_open_count = text.count("Opening scene '" + SCENE_PATH + "'")
    playmode_count = len(re.findall(r"Entering Playmode|Entering PlayMode", text))
    exported_path = str(result.resolve() /
                        "Step10D_TestResults_152.xml").replace("\\", "/")
    exported = text.count("Saving results to: " + exported_path)
    require("6000.3.14f1 (d68c3f99a318)" in text, "Unity version missing from log")
    require("BatchMode: 0, IsHumanControllingUs: 1" in text,
            "GUI Editor identity missing from log")
    require(compiler_errors == 0, "compiler errors present")
    require(error_count > 0 and scene_open_count > 0, "Scene error missing from log")
    require(playmode_count == 0, "PlayMode unexpectedly entered")
    require(exported == 1, "selected TestResults export count mismatch")
    return {
        "schema_version": SCHEMA + ".editor_log",
        "source": record(relative, raw),
        "unity_version": "6000.3.14f1",
        "unity_revision": "d68c3f99a318",
        "batchmode": False,
        "human_controlling": True,
        "compiler_error_count": compiler_errors,
        "selected_results_export_count": exported,
        "portfolio_scene_open_log_count": scene_open_count,
        "scene_reference_error": SCENE_ERROR,
        "scene_reference_error_occurrence_count": error_count,
        "playmode_execution_count": playmode_count,
        "status": "PASS_LOG_EVIDENCE_OF_SCENE_OPEN_FAILURE",
    }


def assertions_bytes(rows: list[tuple[str, str, str]]) -> bytes:
    keys = [row[0] for row in rows]
    require(len(keys) == len(set(keys)), "duplicate assertion key")
    return b"".join(jbytes({"assertion_key": key, "status": status, "detail": detail})
                    for key, status, detail in rows)


def add_manifest(files: dict[str, bytes]) -> dict[str, bytes]:
    require("output_file_manifest.json" not in files, "manifest already present")
    rows = [record(relative, files[relative]) for relative in sorted(files)]
    manifest = {
        "schema_version": SCHEMA + ".output_file_manifest",
        "manifest_excludes_itself": True,
        "target_count": len(rows),
        "files": rows,
        "status": "PASS",
    }
    return {**files, "output_file_manifest.json": jbytes(manifest)}


def report(tests: dict[str, Any], log: dict[str, Any], fixed: dict[str, Any],
           checkpoint: dict[str, Any]) -> bytes:
    text = f"""# WordsInSilence Step 10D GUI Unity Editor gate validation report

## Primary verdict

`FAIL_BLOCKED_BEFORE_PLAYMODE_SCENE_REFERENCE_ERROR`

The first GUI EditMode run passed all 152 unique tests. Opening `{SCENE_PATH}` then emitted the preserved Unity error `{SCENE_ERROR}`. The user correctly did not enter PlayMode and did not rerun the tests.

## Results

- Unity GUI Editor: `6000.3.14f1`
- Compiler errors: `{log['compiler_error_count']}` (`PASS`)
- EditMode executions: `{tests['execution_count']}`
- EditMode: `{tests['passed']}/{tests['total']} PASS`, fail/skip/inconclusive `0/0/0`
- Retry/repeat nonzero: `{tests['retry_nonzero_count']}/{tests['repeat_nonzero_count']}`
- Scene reference error occurrences in full Editor log: `{log['scene_reference_error_occurrence_count']}`
- PlayMode executions: `{log['playmode_execution_count']}` (`NOT_APPLICABLE`, blocked before entry)
- Fixed Step 10D project files: `{fixed['match_count']}/{fixed['target_count']}` unchanged
- GameAssets compared to post-Step10D checkpoint: `{checkpoint['match_count']}/{checkpoint['target_count']}` unchanged
- Retroactive whole-GameAssets preservation: `UNVERIFIED_NONBLOCKING`

## Presentation items

- Companion follow: `UNVERIFIED`
- Robot animation playback: `UNVERIFIED`
- Procedural-hover absence in PlayMode: `UNVERIFIED` (source/EditMode contract remains approved)
- Prompt follows companion: `UNVERIFIED`
- Practice entry: `NOT_APPLICABLE`
- Webcam/live capture: `NOT_APPLICABLE`

No C# source, test source, Scene, Prefab, `.meta`, GameAssets, recognition asset, candidate repository, Build Settings, or fixed Step 10D v1.0 Delivery was modified by this validation-only packaging run.

This is Codex self-validation evidence pending ChatGPT Work external review.
"""
    return text.encode("utf-8")


def package(root: Path) -> None:
    result = root / "Validation" / NAME
    delivery = root / "Validation" / (NAME + "_delivery.zip")
    detached = Path(str(delivery) + ".sha256")
    require(result.is_dir(), "gate evidence directory missing")
    require(not delivery.exists() and not detached.exists(), "never-overwrite delivery exists")
    current_raw = {path.name for path in result.iterdir() if path.is_file()}
    require(current_raw == RAW_EVIDENCE, "unexpected or missing raw evidence")

    zip_helper = load_module(root, ZIP_HELPER, "step10d_gui_zip_helper")
    tx_helper = load_module(root, TX_HELPER, "step10d_gui_tx_helper")
    v1 = verify_v1_identity(root)
    fixed = fixed_project_current(root)
    checkpoint = checkpoint_current(root)
    tests = parse_tests(result)
    log = parse_editor_log(result)
    test_screen = png_record(result, "01_test_runner_152.png")
    error_screen = png_record(result, "scene_error.png")
    staged = staged_files(root)
    require(staged == [], "Git staged files changed")
    git = {
        "schema_version": SCHEMA + ".git_staged",
        "before_count": 0,
        "after_count": 0,
        "staged_paths": [],
        "status": "PASS",
    }
    transaction = tx_helper.transaction_regression()
    require(transaction["status"] == "PASS" and
            len(transaction["failure_injection_rows"]) == 4 and
            all(row["status"] == "PASS" for row in transaction["failure_injection_rows"]) and
            transaction["never_overwrite_second_commit_blocked"],
            "packaging transaction regression failed")

    scene_evaluation = {
        "schema_version": SCHEMA + ".scene_open_evaluation",
        "scene": SCENE_PATH,
        "scene_opened_in_editor": True,
        "error": SCENE_ERROR,
        "error_occurrence_count_in_full_editor_log":
            log["scene_reference_error_occurrence_count"],
        "screen_evidence": error_screen,
        "playmode_started": False,
        "tests_rerun": False,
        "failure_cause": "UNVERIFIED_SERIALIZED_PREFAB_REFERENCE_CAST",
        "gate_action": "STOPPED_WITH_FIRST_EVIDENCE",
        "status": "FAIL",
    }
    primary = {
        "schema_version": SCHEMA + ".primary_verdict",
        "overall_gate": "FAIL_BLOCKED_BEFORE_PLAYMODE_SCENE_REFERENCE_ERROR",
        "unity_compile": "PASS",
        "editmode": "PASS_152_OF_152",
        "editmode_execution_count": 1,
        "scene_open": "FAIL_PREFABINSTANCE_TO_PREFAB_PPTR_CAST",
        "playmode_execution_count": 0,
        "companion_follow": "UNVERIFIED",
        "robot_animation_playback": "UNVERIFIED",
        "procedural_hover_absence_playmode": "UNVERIFIED",
        "prompt_companion_follow": "UNVERIFIED",
        "practice_entry": "NOT_APPLICABLE",
        "webcam_live_capture": "NOT_APPLICABLE",
        "gameassets_post_step10d_checkpoint": "PASS_242_OF_242",
        "whole_gameassets_retroactive_preservation": "UNVERIFIED_NONBLOCKING",
        "status_strings": [
            "STEP10D_GUI_UNITY_EDITOR_GATE_EVIDENCE_READY",
            "STEP10D_EDITMODE_PASS_152_OF_152",
            "STEP10D_COMPANION_PRESENTATION_PLAYMODE_BLOCKED_BY_SCENE_REFERENCE_ERROR",
            "STEP10D_NONCORE_GAMEASSETS_RETROACTIVE_PRESERVATION_UNVERIFIED_NONBLOCKING",
            "BLOCKED_PENDING_EXTERNAL_REVIEW",
        ],
        "status": "FAIL",
    }
    assertion_rows = [
        ("fixed_step10d_v1_0_delivery", "PASS", V1_ZIP_SHA256),
        ("fixed_project_files", "PASS", "10/10"),
        ("gameassets_post_step10d_checkpoint", "PASS", "242/242 current match"),
        ("gameassets_retroactive_preservation", "UNVERIFIED",
         "UNVERIFIED_NONBLOCKING; checkpoint is post-Step10D only"),
        ("unity_gui_editor", "PASS", "6000.3.14f1; batchmode false"),
        ("compiler_errors", "PASS", "0"),
        ("editmode_execution_count", "PASS", "1"),
        ("editmode_unique_tests", "PASS", "152"),
        ("editmode_pass_fail", "PASS", "152/0"),
        ("editmode_skipped_inconclusive", "PASS", "0/0"),
        ("editmode_retry_repeat", "PASS", "0/0"),
        ("fixture_inventory", "PASS", "11 fixtures; 152 tests"),
        ("scene_reference_integrity", "FAIL", SCENE_ERROR),
        ("playmode_execution_count", "NOT_APPLICABLE", "0; blocked before PlayMode"),
        ("companion_follow", "UNVERIFIED", "PlayMode not entered"),
        ("robot_animation_playback", "UNVERIFIED", "PlayMode not entered"),
        ("procedural_hover_absence_playmode", "UNVERIFIED", "PlayMode not entered"),
        ("prompt_companion_follow", "UNVERIFIED", "PlayMode not entered"),
        ("practice_entry", "NOT_APPLICABLE", "PlayMode not entered"),
        ("webcam_live_capture", "NOT_APPLICABLE", "prohibited by gate"),
        ("tests_not_rerun_after_failure", "PASS", "first result preserved"),
        ("git_staged", "PASS", "0"),
        ("packaging_transaction", "PASS", "4/4 failure injection and never-overwrite"),
    ]
    assertions = assertions_bytes(assertion_rows)
    assertion_counts = dict(Counter(row[1] for row in assertion_rows))
    preflight = {
        "schema_version": SCHEMA + ".preflight",
        "fixed_project_files": "10/10",
        "gameassets_post_step10d_checkpoint": "242/242",
        "whole_gameassets_retroactive_preservation": "UNVERIFIED_NONBLOCKING",
        "git_staged_count": 0,
        "status": "PASS",
    }
    provenance = {
        "schema_version": SCHEMA + ".input_provenance",
        "implementation_owner": "Codex",
        "fixed_step10d_v1_0_delivery": v1,
        "checkpoint_semantics": "post-Step10D current checkpoint only",
        "status": "PASS",
    }
    generated = {
        "preflight_before_snapshot.json": jbytes(preflight),
        "input_provenance.json": jbytes(provenance),
        "test_results_summary.json": jbytes(tests),
        "editor_log_summary.json": jbytes(log),
        "scene_open_failure_evaluation.json": jbytes(scene_evaluation),
        "fixed_project_files_after_gui.json": jbytes(fixed),
        "gameassets_post_step10d_checkpoint_after_gui.json": jbytes(checkpoint),
        "git_staged_before_after.json": jbytes(git),
        "packaging_transaction_failure_injection_regression.json": jbytes(transaction),
        "primary_verdict.json": jbytes(primary),
        "assertion_summary.json": jbytes({
            "schema_version": SCHEMA + ".assertion_summary",
            "total": len(assertion_rows),
            "counts": assertion_counts,
            "status": "EXPECTED_GATE_FAILURE_RECORDED",
        }),
        "ASSERTIONS.jsonl": assertions,
        "VALIDATION_REPORT.md": report(tests, log, fixed, checkpoint),
        "analysis/" + Path(__file__).name: Path(__file__).read_bytes(),
    }
    for relative in generated:
        require(not (result / relative).exists(),
                "never-overwrite generated result exists: " + relative)

    raw_files = {
        path.relative_to(result).as_posix(): path.read_bytes()
        for path in result.rglob("*") if path.is_file()
    }
    require(set(raw_files) == RAW_EVIDENCE, "raw evidence changed during validation")
    complete = add_manifest({**raw_files, **generated})
    zip_blob = zip_helper.deterministic_zip(complete)
    detached_blob = f"{sha(zip_blob)}  {delivery.name}\n".encode("ascii")
    validation = zip_helper.validate_zip(zip_blob, complete)
    require(validation["duplicate_count"] == 0 and
            validation["unsafe_path_count"] == 0 and
            validation["encrypted_entry_count"] == 0 and
            validation["symlink_entry_count"] == 0 and
            validation["crc_failure"] is None and validation["membership_match"] and
            validation["payload_mismatch_count"] == 0,
            "pre-commit ZIP validation failed")

    temp = Path(tempfile.mkdtemp(prefix="step10d_gui_gate_", dir=root / "Validation"))
    created: list[Path] = []
    committed = False
    try:
        commit_files = {
            **generated,
            "output_file_manifest.json": complete["output_file_manifest.json"],
        }
        for relative, data in commit_files.items():
            path = temp / PurePosixPath(relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        for relative in sorted(commit_files):
            source = temp / PurePosixPath(relative)
            target = result / PurePosixPath(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            require(not target.exists(), "generated target appeared during commit")
            source.replace(target)
            created.append(target)
        zip_helper.commit_pair(zip_blob, detached_blob, delivery, detached)
        disk = {
            path.relative_to(result).as_posix(): path.read_bytes()
            for path in result.rglob("*") if path.is_file()
        }
        require(disk == complete, "committed result file map mismatch")
        require(zip_helper.deterministic_zip(disk) == delivery.read_bytes(),
                "deterministic ZIP mismatch")
        require(detached.read_bytes() == detached_blob, "detached checksum mismatch")
        post = zip_helper.validate_zip(delivery.read_bytes(), disk)
        require(post["duplicate_count"] == 0 and post["unsafe_path_count"] == 0 and
                post["encrypted_entry_count"] == 0 and post["symlink_entry_count"] == 0 and
                post["crc_failure"] is None and post["membership_match"] and
                post["payload_mismatch_count"] == 0,
                "post-commit ZIP validation failed")
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
        "packaging_status": "PASS",
        "gate_status": primary["overall_gate"],
        "result_directory": str(result),
        "result_file_count": len(complete),
        "manifest_target_count": len(complete) - 1,
        "delivery_zip": str(delivery),
        "delivery_size_bytes": delivery.stat().st_size,
        "delivery_sha256": sha(delivery.read_bytes()),
        "detached_checksum": str(detached),
        "detached_size_bytes": detached.stat().st_size,
        "detached_file_sha256": sha(detached.read_bytes()),
        "compiler_errors": 0,
        "editmode": "152 PASS / 0 FAIL",
        "playmode_execution_count": 0,
        "scene_error_occurrence_count": log["scene_reference_error_occurrence_count"],
        "fixed_project_files": "10/10",
        "gameassets_checkpoint": "242/242",
        "gameassets_retroactive_preservation": "UNVERIFIED_NONBLOCKING",
        "git_staged": 0,
        "assertion_counts": assertion_counts,
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
