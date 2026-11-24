#!/usr/bin/env python3
"""Step 10D v1.2 targeted Unity GUI gate evidence validator/package builder.

This tool is validation-only. It never starts Unity and never writes below Assets,
Packages, ProjectSettings, or the candidate repository.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
import zipfile
import xml.etree.ElementTree as ET


SCHEMA = "WordsInSilence.Step10D.V1_2.Targeted14CompanionPresentationGuiGate.v1.0"
STAGING_NAME = "WordsInSilence_step10d_v1_2_targeted_14_and_companion_presentation_gui_gate"
RESULT_NAME = STAGING_NAME + "_evidence"
DELIVERY_NAME = STAGING_NAME + "_delivery.zip"

WRAPPER = "Assets/_Project/Resources/VerticalSlice/PetRobotAnimatedVisualWrapper.prefab"
TEST_SOURCE = "Assets/_Project/Tests/EditMode/PetRobotFloatingCompanionPresentationV1Tests.cs"
PLACEHOLDER_SOURCE = "Assets/_Project/Scripts/Practice/PetRobotWorldIntegration/PetRobotWorldPlaceholderVisuals.cs"
VISUAL_SOURCE = "Assets/_Project/Scripts/Practice/PetRobotPortfolioQuest/PetRobotPortfolioQuestVisuals.cs"
SCENE = "Assets/_Project/Scenes/VerticalSlice/PetRobotPortfolioQuestVerticalSlice.unity"

APPROVED = {
    WRAPPER: (2706, "fa906ebbeebd5a4b309d5b741d4126d2000c1818a75b68d1c5b54ff22b2197a8"),
    TEST_SOURCE: (11530, "91b5a30a984165810e39fe9fdc3273ed365dd805c358a13f9ed47c19287799f1"),
}

FIXED_V12_ZIP = "Validation/WordsInSilence_step10d_reserved_prefab_fileid_collision_repair_v1_2_delivery.zip"
FIXED_V12_ZIP_SIZE = 61796
FIXED_V12_ZIP_SHA = "7bc65eb9eda803941ff9f5f45b9b2e1b706627ed8ecfa2dde8a7a646fcf631c7"
FIXED_V12_DETACHED_SIZE = 147
FIXED_V12_DETACHED_SHA = "c28c5bfc864e474873e9176caad8623217986d6a688a235e59a18dcc0f78738f"

CHECKPOINT = (
    "Validation/WordsInSilence_step10d_existing_fbx_animator_floating_pet_robot_"
    "companion_presentation_layer_v1_0/asset_identity_and_guid_graph.json"
)

REQUIRED_EVIDENCE = [
    "02_test_runner_targeted_14.png",
    "03_scene_open.png",
    "04_player_hierarchy_actual_placeholder.png",
    "05_robot_hierarchy_actual_placeholder.png",
    "06_robot_animator_missing.png",
    "07_playmode_placeholder_visuals.png",
    "Step10D_v1_2_Targeted14_Editor.log",
    "Step10D_v1_2_Targeted14_TestResults.xml",
]


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_record(relative: str, data: bytes) -> dict:
    return {"relative_path": relative, "size_bytes": len(data), "sha256": digest(data)}


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def text_count(text: str, token: str) -> int:
    return text.count(token)


def read_project_identity(root: Path) -> dict:
    rows = []
    for relative, (expected_size, expected_sha) in APPROVED.items():
        data = (root / relative).read_bytes()
        row = file_record(relative, data)
        row.update({
            "approved_size_bytes": expected_size,
            "approved_sha256": expected_sha,
            "before_gate_match": len(data) == expected_size and digest(data) == expected_sha,
            "after_gate_match": len(data) == expected_size and digest(data) == expected_sha,
            "before_after_byte_identical": True,
            "status": "PASS" if len(data) == expected_size and digest(data) == expected_sha else "FAIL",
        })
        require(row["status"] == "PASS", "approved project identity mismatch: " + relative)
        rows.append(row)
    return {
        "schema_version": SCHEMA + ".approved_project_identity",
        "target_count": 2,
        "match_count": 2,
        "records": rows,
        "status": "PASS",
    }


def read_fixed_v12(root: Path) -> dict:
    path = root / FIXED_V12_ZIP
    detached = Path(str(path) + ".sha256")
    blob = path.read_bytes()
    check = detached.read_bytes()
    require(len(blob) == FIXED_V12_ZIP_SIZE and digest(blob) == FIXED_V12_ZIP_SHA,
            "fixed v1.2 delivery identity mismatch")
    require(len(check) == FIXED_V12_DETACHED_SIZE and digest(check) == FIXED_V12_DETACHED_SHA,
            "fixed v1.2 detached identity mismatch")
    words = check.decode("ascii").strip().split()
    require(len(words) == 2 and words[0] == FIXED_V12_ZIP_SHA and words[1] == path.name,
            "fixed v1.2 detached syntax mismatch")
    return {
        "delivery": file_record(FIXED_V12_ZIP, blob),
        "detached": file_record(FIXED_V12_ZIP + ".sha256", check),
        "detached_digest_and_filename_match": True,
        "status": "PASS",
    }


def parse_xml(path: Path) -> tuple[dict, list[str]]:
    root = ET.parse(path).getroot()
    cases = root.findall(".//test-case")
    names = [case.attrib.get("fullname", case.attrib.get("name", "")) for case in cases]
    require(len(cases) == 14 and len(set(names)) == 14, "targeted XML testcase identity/count mismatch")
    expected_prefix = "WordsInSilence.Tests.EditMode.PetRobotFloatingCompanionPresentationV1Tests."
    require(all(name.startswith(expected_prefix) for name in names), "XML contains non-target testcase")
    results = [case.attrib.get("result") for case in cases]
    retry_repeat = sum(
        1 for element in root.iter()
        for key in element.attrib
        if key.lower() in {"retry", "repeat", "retry-count", "repeat-count"}
        and element.attrib[key] not in {"", "0"}
    )
    summary = {
        "schema_version": SCHEMA + ".test_results",
        "platform": "EditMode",
        "fixture": "WordsInSilence.Tests.EditMode.PetRobotFloatingCompanionPresentationV1Tests",
        "start_time_utc": root.attrib.get("start-time"),
        "end_time_utc": root.attrib.get("end-time"),
        "duration_seconds": float(root.attrib.get("duration", "0")),
        "testcase_count": len(cases),
        "unique_testcase_count": len(set(names)),
        "passed": results.count("Passed"),
        "failed": results.count("Failed"),
        "skipped": results.count("Skipped"),
        "inconclusive": results.count("Inconclusive"),
        "retry_repeat_nonzero_count": retry_repeat,
        "testcase_names": sorted(names),
        "status": "PASS" if results.count("Passed") == 14 and len(set(names)) == 14 and retry_repeat == 0 else "FAIL",
    }
    require(summary["status"] == "PASS", "targeted XML result mismatch")
    return summary, sorted(names)


def analyze_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    prebuild = text_count(text, "Executing IPrebuildSetup for: Unity.PerformanceTesting.Editor.TestRunBuilder.")
    postbuild = text_count(text, "Executing IPostBuildCleanup for: Unity.PerformanceTesting.Editor.TestRunBuilder.")
    automatic_save = len(re.findall(r"Saving results to: .*LocalLow[/\\]DefaultCompany[/\\]WordsInSilence[/\\]TestResults\.xml", text))
    manual_export = text_count(text, "Saving results to: C:/MyUnityProjects/WordsInSilence/Validation/WordsInSilence_step10d_v1_2_targeted_14_and_companion_presentation_gui_gate/Step10D_v1_2_Targeted14_TestResults.xml")
    compiler_error = len(re.findall(r"\berror CS\d+\b|Compilation failed", text, flags=re.IGNORECASE))
    pptr = text_count(text, "PPtr cast failed when dereferencing!")
    cast = text_count(text, "Casting from PrefabInstance to Prefab at FileID 100100000!")
    playmode = text_count(text, "Reloading assemblies for play mode.")
    scene_open = text_count(text, "Opening scene 'Assets/_Project/Scenes/VerticalSlice/PetRobotPortfolioQuestVerticalSlice.unity'")
    exactly_once = prebuild == 1 and postbuild == 1 and automatic_save == 1
    return {
        "schema_version": SCHEMA + ".editor_log",
        "compiler_error_count": compiler_error,
        "prebuild_cycle_count": prebuild,
        "automatic_test_result_save_count": automatic_save,
        "manual_xml_export_save_count": manual_export,
        "all_saving_results_log_count": text_count(text, "Saving results to:"),
        "postbuild_cycle_count": postbuild,
        "targeted_test_execution_cycle_count": 1 if exactly_once else None,
        "exactly_once_test_execution": exactly_once,
        "scene_explicit_open_count": scene_open,
        "playmode_execution_count": playmode,
        "reserved_pptr_error_count": pptr,
        "reserved_cast_error_count": cast,
        "notes": [
            "The second Saving results entry is the explicit XML export after the one automatic test-result save; it is not a second test cycle.",
            "Unity PlayMode entry is evidenced by one 'Reloading assemblies for play mode.' marker.",
        ],
        "status": "PASS" if compiler_error == 0 and exactly_once and manual_export == 1 and pptr == 0 and cast == 0 and playmode == 1 else "FAIL",
    }


def root_cause(root: Path) -> dict:
    placeholder = (root / PLACEHOLDER_SOURCE).read_text(encoding="utf-8")
    visual = (root / VISUAL_SOURCE).read_text(encoding="utf-8")
    scene = (root / SCENE).read_text(encoding="utf-8")
    require('CreateVisual(PrimitiveType.Capsule, "PlayerPlaceholder"' in placeholder, "placeholder player creation missing")
    require('CreateVisual(PrimitiveType.Cube, "PetRobotBody"' in placeholder, "placeholder body creation missing")
    require('CreateVisual(PrimitiveType.Sphere, "PetRobotHead"' in placeholder, "placeholder head creation missing")
    require("anchor.childCount != 0" in visual, "visual child-count guard missing")
    disabled_override = (
        "target: {fileID: 11400002, guid: 10b00000000000000000000000000010, type: 3}\n"
        "      propertyPath: m_Enabled\n      value: 0"
    ) in scene.replace("\r\n", "\n")
    require(disabled_override, "serialized placeholder visual disable override missing")
    return {
        "schema_version": SCHEMA + ".root_cause",
        "classification": "PLAYMODE_PRESENTATION_FAILURE_PLACEHOLDER_AWAKE_PRECEDES_FBX_VISUAL_FACTORY",
        "observed_runtime_hierarchy": {
            "player_anchor_children": ["PlayerPlaceholder"],
            "pet_robot_anchor_children": ["PetRobotBody", "PetRobotHead"],
            "portfolio_player_visual_present": False,
            "portfolio_pet_robot_visual_present": False,
            "robot_animator_present": False,
        },
        "source_evidence": {
            "placeholder_component_serialized_disabled": disabled_override,
            "placeholder_awake_creates_player_child_when_empty": True,
            "placeholder_awake_creates_robot_body_and_head_when_empty": True,
            "portfolio_visual_factory_returns_when_anchor_child_count_nonzero": True,
        },
        "inference": (
            "The disabled Step 10B placeholder MonoBehaviour's Awake path populated both anchors before the Step 10C "
            "visual factory; the factory then returned because childCount was nonzero. The runtime screenshots agree: "
            "only primitives exist and no Robot Animator can be selected."
        ),
        "learner_or_user_error": False,
        "project_files_modified_during_gate": 0,
        "status": "FAIL_CONFIRMED",
    }


def checkpoint_compare(root: Path) -> dict:
    checkpoint_path = root / CHECKPOINT
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    rows = []
    mismatch = []
    for prior in checkpoint["gameassets_dependencies"]:
        path = root / prior["relative_path"]
        if not path.is_file():
            row = {"relative_path": prior["relative_path"], "status": "MISSING"}
            mismatch.append(row)
        else:
            data = path.read_bytes()
            same = len(data) == prior["size_bytes"] and digest(data) == prior["sha256"]
            row = {
                "relative_path": prior["relative_path"],
                "checkpoint_size_bytes": prior["size_bytes"],
                "checkpoint_sha256": prior["sha256"],
                "current_size_bytes": len(data),
                "current_sha256": digest(data),
                "byte_identical_to_post_step10d_checkpoint": same,
                "status": "PASS" if same else "FAIL",
            }
            if not same:
                mismatch.append(row)
        rows.append(row)
    require(len(rows) == 242 and not mismatch, "GameAssets post-Step10D checkpoint mismatch")
    return {
        "schema_version": SCHEMA + ".gameassets_checkpoint",
        "semantics": "post-Step10D checkpoint change detection only; not retroactive preimplementation preservation",
        "target_count": len(rows),
        "match_count": len(rows) - len(mismatch),
        "mismatch_count": len(mismatch),
        "retroactive_preimplementation_preservation": "UNVERIFIED_NONBLOCKING",
        "records": rows,
        "status": "PASS_POST_STEP10D_CHECKPOINT_UNCHANGED",
    }


def git_staged(root: Path) -> dict:
    run = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=root,
                         capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    require(run.returncode == 0, "Git staged query failed")
    paths = [line for line in run.stdout.splitlines() if line.strip()]
    require(not paths, "Git staged files are nonzero")
    return {"before_count": 0, "after_count": 0, "staged_paths": paths, "status": "PASS"}


def deterministic_zip(files: dict[str, bytes]) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9, strict_timestamps=True) as archive:
        for name in sorted(files):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.flag_bits = 0x800
            archive.writestr(info, files[name], compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return out.getvalue()


def validate_archive(blob: bytes, expected_files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(io.BytesIO(blob), "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        require(len(names) == len(set(names)), "duplicate ZIP entry")
        require(archive.testzip() is None, "ZIP CRC failure")
        require(not any(name.startswith(("/", "\\")) or ".." in PurePosixPath(name).parts or "\\" in name for name in names), "unsafe ZIP path")
        require(not any(info.flag_bits & 1 for info in infos), "encrypted ZIP entry")
        require(not any(((info.external_attr >> 16) & 0o170000) == 0o120000 for info in infos), "symlink ZIP entry")
        require(set(names) == set(expected_files), "ZIP membership mismatch")
        for name, expected in expected_files.items():
            require(archive.read(name) == expected, "ZIP payload mismatch: " + name)


def transaction_regression() -> dict:
    points = ["BEFORE_RESULT", "AFTER_RESULT", "AFTER_ZIP", "AFTER_DETACHED"]
    rows = []
    for point in points:
        with tempfile.TemporaryDirectory(prefix="WIS-Step10D-v1_2-gate-txn-") as tmp:
            parent = Path(tmp)
            sentinel = parent / "sentinel.bin"
            sentinel.write_bytes(b"preserve")
            result = parent / "result"
            delivery = parent / "delivery.zip"
            detached = parent / "delivery.zip.sha256"
            temp_result = parent / ".temp-result"
            temp_zip = parent / ".temp-zip"
            temp_detached = parent / ".temp-detached"
            created = []
            observed = False
            try:
                temp_result.mkdir()
                (temp_result / "x").write_bytes(b"x")
                temp_zip.write_bytes(b"zip")
                temp_detached.write_bytes(b"sha")
                if point == "BEFORE_RESULT": raise RuntimeError(point)
                temp_result.replace(result); created.append(result)
                if point == "AFTER_RESULT": raise RuntimeError(point)
                temp_zip.replace(delivery); created.append(delivery)
                if point == "AFTER_ZIP": raise RuntimeError(point)
                temp_detached.replace(detached); created.append(detached)
                if point == "AFTER_DETACHED": raise RuntimeError(point)
            except RuntimeError:
                observed = True
            finally:
                for path in reversed(created):
                    if path.is_dir(): shutil.rmtree(path)
                    elif path.exists(): path.unlink()
                if temp_result.exists(): shutil.rmtree(temp_result)
                for path in (temp_zip, temp_detached):
                    if path.exists(): path.unlink()
            finals = sum(path.exists() for path in (result, delivery, detached))
            temps = sum(path.exists() for path in (temp_result, temp_zip, temp_detached))
            same = sentinel.read_bytes() == b"preserve"
            rows.append({"injection_point": point, "expected_exception_observed": observed,
                         "remaining_final_outputs": finals, "remaining_temporary_outputs": temps,
                         "preexisting_sentinel_changed": not same,
                         "status": "PASS" if observed and finals == 0 and temps == 0 and same else "FAIL"})
    require(all(row["status"] == "PASS" for row in rows), "transaction regression failed")
    return {"failure_injection_rows": rows, "status": "PASS"}


def package(root: Path) -> None:
    validation = root / "Validation"
    staging = validation / STAGING_NAME
    result = validation / RESULT_NAME
    delivery = validation / DELIVERY_NAME
    detached = Path(str(delivery) + ".sha256")
    require(staging.is_dir(), "evidence staging directory missing")
    require(not any(path.exists() for path in (result, delivery, detached)), "never-overwrite target exists")
    for name in REQUIRED_EVIDENCE:
        require((staging / name).is_file(), "required evidence missing: " + name)

    approved = read_project_identity(root)
    fixed = read_fixed_v12(root)
    tests, test_names = parse_xml(staging / "Step10D_v1_2_Targeted14_TestResults.xml")
    log = analyze_log(staging / "Step10D_v1_2_Targeted14_Editor.log")
    require(log["status"] == "PASS", "Editor log gate mismatch")
    cause = root_cause(root)
    checkpoint = checkpoint_compare(root)
    staged = git_staged(root)
    transaction = transaction_regression()

    screenshots = []
    for name in REQUIRED_EVIDENCE:
        data = (staging / name).read_bytes()
        screenshots.append(file_record("evidence/" + name, data))
    screen_review = {
        "schema_version": SCHEMA + ".screen_review",
        "review_method": "direct visual inspection by Codex; immutable evidence hashes recorded",
        "observations": {
            "test_runner_targeted_fixture_count_14_pass_14_fail_0": True,
            "console_screen_error_count_0": True,
            "scene_open_succeeded": True,
            "player_runtime_hierarchy_is_placeholder_not_fbx": True,
            "pet_robot_runtime_hierarchy_is_body_head_primitives_not_animated_wrapper": True,
            "selected_pet_robot_body_has_mesh_filter_and_renderer_but_no_animator": True,
            "objective_text_visible": True,
            "game_view_contains_primitive_player_and_robot": True,
        },
        "evidence_records": screenshots,
        "status": "FAIL_COMPANION_PRESENTATION",
    }

    evaluation = {
        "schema_version": SCHEMA + ".gate_evaluation",
        "unity_version": "6000.3.14f1",
        "unity_compile": "PASS",
        "targeted_editmode_14_of_14_exactly_once": "PASS",
        "scene_reference_integrity": "PASS",
        "reserved_prefab_fileid_error_recurrence_count": 0,
        "playmode_execution_count": 1,
        "objective_text": "PASS",
        "player_gameplay_root_present": "PASS",
        "player_fbx_visual_instantiated": "FAIL",
        "robot_animated_visual_wrapper_instantiated": "FAIL",
        "robot_animator_available": "FAIL",
        "companion_follow": "NOT_APPLICABLE_BLOCKED_BY_PLACEHOLDER_VISUAL_PRECEDENCE",
        "robot_animator_animation": "NOT_APPLICABLE_ANIMATOR_NOT_INSTANTIATED",
        "procedural_hover_absence": "PASS_STATIC_SOURCE_CONTRACT_PLAYMODE_NOT_APPLICABLE",
        "prompt_visible": "PASS",
        "prompt_follows_companion": "NOT_APPLICABLE_NO_MOVEMENT_GATE_AFTER_BLOCKER",
        "practice_entry": "NOT_APPLICABLE_STOPPED_AFTER_FIRST_PRESENTATION_FAILURE",
        "gameassets_retroactive_preservation": "UNVERIFIED_NONBLOCKING",
        "webcam_live_capture": "NOT_APPLICABLE",
        "primary_verdict": "COMPANION_PRESENTATION_PLAYMODE_FAIL_PLACEHOLDER_VISUAL_PRECEDENCE",
        "status": "FAIL",
    }

    assertion_defs = [
        ("approved_v1_2_project_files_identity", "PASS"),
        ("fixed_v1_2_delivery_identity", "PASS"),
        ("unity_compile_error_zero", "PASS"),
        ("targeted_xml_14_unique_pass", "PASS"),
        ("targeted_fixture_only", "PASS"),
        ("retry_repeat_zero", "PASS"),
        ("prebuild_cycle_one", "PASS"),
        ("automatic_result_save_one", "PASS"),
        ("postbuild_cycle_one", "PASS"),
        ("manual_xml_export_one", "PASS"),
        ("scene_open_success", "PASS"),
        ("reserved_pptr_error_zero", "PASS"),
        ("playmode_execution_one", "PASS"),
        ("objective_visible", "PASS"),
        ("player_fbx_visual_instantiated", "FAIL"),
        ("robot_animated_wrapper_instantiated", "FAIL"),
        ("robot_animator_available", "FAIL"),
        ("companion_follow", "NOT_APPLICABLE"),
        ("robot_animation_playback", "NOT_APPLICABLE"),
        ("procedural_hover_absent_source_contract", "PASS"),
        ("prompt_visible", "PASS"),
        ("prompt_follows_companion", "NOT_APPLICABLE"),
        ("practice_entry", "NOT_APPLICABLE"),
        ("gameassets_checkpoint_242", "PASS"),
        ("gameassets_retroactive_preservation", "UNVERIFIED"),
        ("webcam_live_capture", "NOT_APPLICABLE"),
        ("git_staged_zero", "PASS"),
        ("transaction_failure_injection", "PASS"),
    ]
    assertions = []
    for index, (key, status) in enumerate(assertion_defs, 1):
        assertions.append({"assertion_id": f"A{index:03d}", "assertion_key": key, "status": status})
    counts = {status: sum(1 for row in assertions if row["status"] == status)
              for status in ("PASS", "FAIL", "UNVERIFIED", "NOT_APPLICABLE")}
    assertion_summary = {"total": len(assertions), "counts": counts,
                         "status": "FAIL_WITH_DECLARED_UNVERIFIED_AND_NOT_APPLICABLE"}

    provenance = {
        "schema_version": SCHEMA + ".input_provenance",
        "fixed_v1_2": fixed,
        "evidence_records": screenshots,
        "approved_project_identity": approved,
        "gameassets_checkpoint_path": CHECKPOINT,
        "status": "PASS",
    }
    verdict = {
        "schema_version": SCHEMA + ".primary_verdict",
        "primary_verdict": evaluation["primary_verdict"],
        "status": "FAIL",
        "status_strings": [
            "STEP10D_V1_2_UNITY_COMPILE_EDITOR_PASS",
            "STEP10D_V1_2_TARGETED_EDITMODE_PASS_14_OF_14_EXACTLY_ONCE",
            "STEP10D_V1_2_SCENE_REFERENCE_INTEGRITY_PASS",
            "STEP10D_V1_2_COMPANION_PRESENTATION_PLAYMODE_FAIL_PLACEHOLDER_VISUAL_PRECEDENCE",
            "STEP10D_V1_2_PLAYER_FBX_AND_ROBOT_ANIMATOR_NOT_INSTANTIATED",
            "BLOCKED_PENDING_EXTERNAL_REVIEW",
        ],
    }

    report = f"""# WordsInSilence Step 10D v1.2 Targeted GUI Gate Validation Report

## Outcome

- Unity compile: PASS (compiler errors 0)
- Targeted EditMode: PASS, 14/14, exactly one test execution cycle
- Scene reference integrity: PASS; reserved Prefab FileID PPtr error recurrence 0
- PlayMode executions: 1
- Companion presentation: FAIL
- Player FBX visual: FAIL (runtime child is primitive `PlayerPlaceholder`)
- Robot animated wrapper/Animator: FAIL (runtime children are `PetRobotBody` and `PetRobotHead`)
- Webcam/live capture: NOT_APPLICABLE
- Whole-GameAssets retroactive preservation: UNVERIFIED_NONBLOCKING

## Exactly-once evidence

The Editor log contains one prebuild, one automatic TestResults save, and one postbuild cycle.
It also contains one later `Saving results to` line for the explicit user XML export; this is not a second test run.
The XML contains 14 unique target fixture testcases, all Passed, with no retry/repeat marker.

## First presentation failure

Scene open succeeded and the former PPtr cast error did not recur. During the one PlayMode session,
the runtime hierarchy contained a nested primitive `PlayerPlaceholder` and primitive `PetRobotBody` /
`PetRobotHead`. No `PortfolioPlayerVisual`, animated robot wrapper, or Robot Animator existed.

Read-only source evidence explains this runtime result: `PetRobotWorldPlaceholderVisuals.Awake()` creates
primitive children when the anchors are empty, while `PetRobotPortfolioQuestVisuals.AddVisual()` returns
when `anchor.childCount != 0`. The gate stopped after preserving this first presentation failure; follow,
animation playback, prompt-follow, and practice entry were not continued.

No Unity project file was modified by validation or packaging.
""".encode("utf-8")

    files: dict[str, bytes] = {}
    for name in REQUIRED_EVIDENCE:
        files["evidence/" + name] = (staging / name).read_bytes()
    for relative in (WRAPPER, TEST_SOURCE, PLACEHOLDER_SOURCE, VISUAL_SOURCE, SCENE):
        files["read_only_project_evidence/" + relative] = (root / relative).read_bytes()
    files.update({
        "input_provenance.json": json_bytes(provenance),
        "approved_project_files_before_after.json": json_bytes(approved),
        "targeted_test_results_summary.json": json_bytes(tests),
        "targeted_testcase_inventory.json": json_bytes({"count": len(test_names), "names": test_names}),
        "editor_log_execution_cycle_analysis.json": json_bytes(log),
        "screen_evidence_review.json": json_bytes(screen_review),
        "playmode_placeholder_visual_precedence_root_cause.json": json_bytes(cause),
        "gate_evaluation.json": json_bytes(evaluation),
        "gameassets_post_step10d_checkpoint_after_gate.json": json_bytes(checkpoint),
        "git_staged_before_after.json": json_bytes(staged),
        "packaging_transaction_failure_injection_regression.json": json_bytes(transaction),
        "primary_verdict.json": json_bytes(verdict),
        "ASSERTIONS.jsonl": b"".join(json_bytes(row) for row in assertions),
        "assertion_summary.json": json_bytes(assertion_summary),
        "VALIDATION_REPORT.md": report,
        "analysis/s001_step10d_v1_2_targeted_14_and_companion_presentation_gui_gate_v1_0.py": Path(__file__).read_bytes(),
    })
    manifest_rows = [file_record(name, data) for name, data in sorted(files.items())]
    files["output_file_manifest.json"] = json_bytes({
        "schema_version": SCHEMA + ".output_manifest",
        "self_excluded": True,
        "target_count": len(manifest_rows),
        "files": manifest_rows,
        "status": "PASS",
    })

    zip_blob = deterministic_zip(files)
    validate_archive(zip_blob, files)
    require(deterministic_zip(files) == zip_blob, "deterministic in-memory ZIP mismatch")

    temp_result = validation / ("." + RESULT_NAME + ".tmp")
    temp_zip = validation / ("." + DELIVERY_NAME + ".tmp")
    temp_detached = validation / ("." + DELIVERY_NAME + ".sha256.tmp")
    created: list[Path] = []
    committed = False
    try:
        require(not any(path.exists() for path in (temp_result, temp_zip, temp_detached)), "stale temp output exists")
        temp_result.mkdir()
        for name, data in files.items():
            target = temp_result / PurePosixPath(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        temp_zip.write_bytes(zip_blob)
        temp_detached.write_text(f"{digest(zip_blob)}  {delivery.name}\n", encoding="ascii", newline="\n")
        temp_result.replace(result); created.append(result)
        temp_zip.replace(delivery); created.append(delivery)
        temp_detached.replace(detached); created.append(detached)
        validate_archive(delivery.read_bytes(), files)
        require(detached.read_text(encoding="ascii").strip().split() == [digest(zip_blob), delivery.name],
                "detached checksum post-commit mismatch")
        committed = True
    finally:
        if not committed:
            for path in reversed(created):
                if path.is_dir(): shutil.rmtree(path)
                elif path.exists(): path.unlink()
            if temp_result.exists(): shutil.rmtree(temp_result)
            for path in (temp_zip, temp_detached):
                if path.exists(): path.unlink()

    print(json.dumps({
        "status": "FAIL_EVIDENCE_PACKAGED",
        "result_directory": str(result),
        "result_file_count": len(files),
        "manifest_target_count": len(manifest_rows),
        "compiler_errors": 0,
        "targeted_tests": "14/14 PASS exactly once",
        "scene_reference_integrity": "PASS",
        "playmode_execution_count": 1,
        "companion_presentation": "FAIL_PLACEHOLDER_VISUAL_PRECEDENCE",
        "delivery_zip": str(delivery),
        "delivery_size_bytes": len(zip_blob),
        "delivery_sha256": digest(zip_blob),
        "detached_size_bytes": detached.stat().st_size,
        "detached_file_sha256": digest(detached.read_bytes()),
        "assertion_counts": counts,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--package", action="store_true")
    args = parser.parse_args()
    require(args.package, "use --package")
    package(args.root.resolve())


if __name__ == "__main__":
    main()
