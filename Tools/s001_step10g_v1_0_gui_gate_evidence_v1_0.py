from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from typing import Any


WORK_REL = Path("Validation/.step10g_v1_0_gui_gate_work_v1_0")
APPROVED_REL = Path(
    "Validation/WordsInSilence_step10g_moa_robot_speech_bubble_story_guided_practice_shell_v1_0"
)
APPROVED_ZIP_REL = Path(
    "Validation/WordsInSilence_step10g_moa_robot_speech_bubble_story_guided_practice_shell_v1_0_delivery.zip"
)
EXPECTED_DELIVERY_SIZE = 111_157
EXPECTED_DELIVERY_SHA256 = (
    "f758e90991306d55b2ca7aec85f19a74ffc20df2983f914d303d82b41166ba48"
)
EXPECTED_DELIVERY_ENTRIES = 38
EXPECTED_SCENE_SHA256 = (
    "5282810b9dd49194d0b175e54360ce98aab867e8e9065ec48a09abc74a213303"
)
OUTPUT_NAME = (
    "WordsInSilence_step10g_v1_0_gui_expanded_targeted_38_and_story_smoke_"
    "evidence_v1_0"
)
FIXTURE = (
    "WordsInSilence.Tests.EditMode."
    "PrivacyEnhancedGuidedSignPracticePresentationV1Tests"
)
MEDIA_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff",
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".wav", ".mp3", ".ogg",
    ".flac", ".m4a",
}


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def record_file(path: Path, relative_path: str) -> dict[str, Any]:
    require(path.is_file(), f"missing file: {path}")
    data = path.read_bytes()
    return {
        "relative_path": relative_path.replace("\\", "/"),
        "size_bytes": len(data),
        "sha256": sha256_bytes(data),
    }


def identity_matches(expected: dict[str, Any], current: dict[str, Any]) -> bool:
    return (
        int(expected["size_bytes"]) == int(current["size_bytes"])
        and str(expected["sha256"]).lower() == str(current["sha256"]).lower()
    )


def approved_project_records(root: Path) -> list[dict[str, Any]]:
    approved = root / APPROVED_REL
    v14 = read_json(approved / "step10f_v1_4_presentation_contract_preservation.json")
    changed = read_json(approved / "four_file_before_after_identity.json")
    records: dict[str, dict[str, Any]] = {
        row["relative_path"]: dict(row["current"]) for row in v14["records"]
    }
    for row in changed["records"]:
        records[row["relative_path"]] = dict(row["after"])
    for row in changed["existing_related_meta_records"]:
        records[row["relative_path"]] = dict(row["after"])
    require(len(records) == 20, f"approved project record count is {len(records)}, expected 20")
    return [records[name] for name in sorted(records)]


def project_identity(root: Path, phase: str, copy_payload: bool) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    payload_root = root / WORK_REL / f"{phase}_snapshot/project_files"
    for expected in approved_project_records(root):
        relative = expected["relative_path"]
        source = root / Path(relative)
        current = record_file(source, relative)
        match = identity_matches(expected, current)
        rows.append(
            {
                "relative_path": relative,
                "approved_step10g_v1_0": expected,
                "current": current,
                "match": match,
            }
        )
        if copy_payload:
            target = payload_root / Path(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    context = record_file(
        root / "WordsInSilence_PROJECT_CONTEXT.md", "WordsInSilence_PROJECT_CONTEXT.md"
    )
    return {
        "schema_version": "wordsinsilence.step10g.v1.0.gui_gate.project_identity.v1.0",
        "phase": phase,
        "approved_record_count": len(rows),
        "mismatch_count": sum(not row["match"] for row in rows),
        "records": rows,
        "project_context_read_only_identity": context,
        "standalone_webcam_display_transform_cs_exists": (
            root
            / "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/WebcamDisplayTransform.cs"
        ).is_file(),
        "standalone_webcam_display_transform_meta_exists": (
            root
            / "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/WebcamDisplayTransform.cs.meta"
        ).is_file(),
        "approved_architecture_note": (
            "WebcamDisplayTransform is a type inside the approved overlay source; "
            "the approved 20-record baseline has no standalone file."
        ),
    }


def protected_identity(root: Path, phase: str) -> dict[str, Any]:
    baseline = read_json(root / APPROVED_REL / "protected_identity.json")
    rows: list[dict[str, Any]] = []
    for entry in baseline["records"]:
        expected = dict(entry["current"])
        relative = entry["relative_path"]
        current = record_file(root / Path(relative), relative)
        rows.append(
            {
                "relative_path": relative,
                "approved_step10g_v1_0": expected,
                "current": current,
                "match": identity_matches(expected, current),
            }
        )
    return {
        "schema_version": "wordsinsilence.step10g.v1.0.gui_gate.protected_identity.v1.0",
        "phase": phase,
        "record_count": len(rows),
        "mismatch_count": sum(not row["match"] for row in rows),
        "records": rows,
    }


def candidate_identity(root: Path, phase: str) -> dict[str, Any]:
    baseline = read_json(root / APPROVED_REL / "candidate_identity.json")
    candidate_root = Path(baseline["repository_path"])
    require(candidate_root.is_dir(), "candidate repository is missing")
    rows: list[dict[str, Any]] = []
    for entry in baseline["records"]:
        expected = dict(entry["current"])
        relative = entry["relative_path"]
        current = record_file(candidate_root / Path(relative), relative)
        rows.append(
            {
                "relative_path": relative,
                "approved_step10g_v1_0": expected,
                "current": current,
                "match": identity_matches(expected, current),
            }
        )
    files = sorted(path for path in candidate_root.rglob("*") if path.is_file())
    actual_names = [path.relative_to(candidate_root).as_posix() for path in files]
    expected_names = sorted(row["relative_path"] for row in rows)
    index = read_json(candidate_root / "candidate_index.json")
    media = [name for name in actual_names if Path(name).suffix.lower() in MEDIA_SUFFIXES]
    return {
        "schema_version": "wordsinsilence.step10g.v1.0.gui_gate.candidate_identity.v1.0",
        "phase": phase,
        "repository_path": str(candidate_root),
        "candidate_count": len(index.get("entries", [])),
        "record_count": len(files),
        "approved_record_count": len(rows),
        "record_name_set_match": actual_names == expected_names,
        "missing_files": sorted(set(expected_names) - set(actual_names)),
        "extra_files": sorted(set(actual_names) - set(expected_names)),
        "media_file_count": len(media),
        "media_files": media,
        "mismatch_count": sum(not row["match"] for row in rows),
        "records": rows,
    }


def git_staged(root: Path) -> list[str]:
    environment = dict(os.environ)
    environment["XDG_CONFIG_HOME"] = str(root / ".git")
    result = subprocess.run(
        ["git", "-c", "core.excludesfile=NUL", "diff", "--cached", "--name-only"],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def delivery_provenance(root: Path) -> dict[str, Any]:
    path = root / APPROVED_ZIP_REL
    data = path.read_bytes()
    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        crc_error = archive.testzip()
    result = {
        "relative_path": APPROVED_ZIP_REL.as_posix(),
        "size_bytes": len(data),
        "sha256": sha256_bytes(data),
        "entry_count": len(names),
        "crc_error": crc_error,
    }
    result["approved_identity_match"] = (
        result["size_bytes"] == EXPECTED_DELIVERY_SIZE
        and result["sha256"] == EXPECTED_DELIVERY_SHA256
        and result["entry_count"] == EXPECTED_DELIVERY_ENTRIES
        and crc_error is None
    )
    return result


def scene_inventory(root: Path) -> dict[str, Any]:
    relative = "Assets/_Project/Scenes/VerticalSlice/PetRobotSignPracticeVerticalSlice.unity"
    scene_path = root / relative
    identity = record_file(scene_path, relative)
    text = scene_path.read_text(encoding="utf-8-sig")
    return {
        "identity": identity,
        "expected_sha256": EXPECTED_SCENE_SHA256,
        "expected_sha256_match": identity["sha256"] == EXPECTED_SCENE_SHA256,
        "broken_door_prefab_guid_present": "8b2ed08f82f4cea41907e2cf2f8938c5" in text,
        "hologram_material_guid_present": "990a06ffeda43734e864d545330e132e" in text,
        "dark_scifi_skybox_guid_present": "32ea4fce7231ca349a48b944302d257e" in text,
        "approved_hierarchy_path": "/Broken Door",
    }


def preflight(root: Path) -> dict[str, Any]:
    work = root / WORK_REL
    require(not work.exists(), f"never-overwrite work directory already exists: {work}")
    work.mkdir(parents=True, exist_ok=False)
    try:
        project = project_identity(root, "pre", copy_payload=True)
        protected = protected_identity(root, "pre")
        candidate = candidate_identity(root, "pre")
        delivery = delivery_provenance(root)
        scene = scene_inventory(root)
        staged = git_staged(root)
        unity_count = len(
            [
                process
                for process in subprocess.run(
                    ["powershell", "-NoProfile", "-Command", "@(Get-Process -Name Unity -ErrorAction SilentlyContinue).Count"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.splitlines()
                if process.strip() and process.strip() != "0"
            ]
        )
        require(delivery["approved_identity_match"], "approved Step 10G Delivery mismatch")
        require(project["mismatch_count"] == 0, "project baseline mismatch")
        require(protected["record_count"] == 36, "protected record count is not 36")
        require(protected["mismatch_count"] == 0, "protected baseline mismatch")
        require(candidate["candidate_count"] == 8, "candidate count is not 8")
        require(candidate["record_count"] == 25, "candidate file count is not 25")
        require(candidate["mismatch_count"] == 0, "candidate byte mismatch")
        require(candidate["record_name_set_match"], "candidate file set mismatch")
        require(candidate["media_file_count"] == 0, "candidate media is not zero")
        require(not staged, "Git staged state is not empty")
        require(scene["expected_sha256_match"], "current authored Scene SHA-256 mismatch")
        require(scene["broken_door_prefab_guid_present"], "Broken Door prefab GUID is absent")
        require(scene["hologram_material_guid_present"], "hologram material GUID is absent")
        require(scene["dark_scifi_skybox_guid_present"], "dark Sci-Fi skybox GUID is absent")
        require(unity_count == 0, "Unity process is running during Gate 0")
        write_json(work / "approved_step10g_v1_0_provenance.json", delivery)
        write_json(work / "project_identity_before.json", project)
        write_json(work / "protected_identity_before.json", protected)
        write_json(work / "candidate_identity_before.json", candidate)
        write_json(work / "scene_inventory_before.json", scene)
        write_json(
            work / "git_staged_before.json",
            {"staged_count": len(staged), "staged_paths": staged},
        )
        summary = {
            "status": "GATE0_FRESH_IMMUTABLE_PRE_RUN_SNAPSHOT_PASS",
            "unity_process_count": unity_count,
            "approved_delivery_identity_match": True,
            "project_record_count": project["approved_record_count"],
            "project_mismatch_count": project["mismatch_count"],
            "scene_sha256": scene["identity"]["sha256"],
            "protected_record_count": protected["record_count"],
            "protected_mismatch_count": protected["mismatch_count"],
            "candidate_count": candidate["candidate_count"],
            "candidate_file_count": candidate["record_count"],
            "candidate_media_count": candidate["media_file_count"],
            "git_staged_count": len(staged),
        }
        write_json(work / "gate0_summary.json", summary)
        return summary
    except Exception:
        shutil.rmtree(work, ignore_errors=True)
        raise


def capture_editor_log(root: Path, label: str) -> dict[str, Any]:
    work = root / WORK_REL
    require(work.is_dir(), "Gate 0 work directory is missing")
    source = Path(os.environ["LOCALAPPDATA"]) / "Unity/Editor/Editor.log"
    data = source.read_bytes()
    target = work / "logs" / f"EditorLog_{label}.log"
    require(not target.exists(), f"never-overwrite log snapshot already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    result = {
        "source_path": str(source),
        "snapshot_relative_path": target.relative_to(root).as_posix(),
        "size_bytes": len(data),
        "sha256": sha256_bytes(data),
    }
    write_json(work / "logs" / f"EditorLog_{label}_identity.json", result)
    return result


def analyze_compile_log(root: Path, label: str) -> dict[str, Any]:
    path = root / WORK_REL / "logs" / f"EditorLog_{label}.log"
    require(path.is_file(), f"missing captured Editor log: {path}")
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = text.splitlines()

    def hits(pattern: str) -> list[dict[str, Any]]:
        expression = re.compile(pattern, re.IGNORECASE)
        return [
            {"line": number, "text": line}
            for number, line in enumerate(lines, 1)
            if expression.search(line)
        ]

    csharp = hits(r"\berror CS\d+\b|Scripts have compiler errors|Tundra build failed")
    shader = hits(r"Shader error in|Failed to compile shader|shader compiler.*error")
    pptr = hits(r"PPtr cast failed|Type mismatch|Casting from .* at FileID|FileID.*mismatch")
    uisprite = hits(r"UI/Skin/UISprite\.psd")
    success = hits(r"\*\*\* Tundra build success")
    reloads = hits(r"Mono: successfully reloaded assembly")
    result = {
        "schema_version": "wordsinsilence.step10g.v1.0.gui_gate.compile_log.v1.0",
        "editor_log_identity": record_file(path, path.relative_to(root).as_posix()),
        "line_count": len(lines),
        "unity_version_marker_count": len(hits(r"Version is '6000\.3\.14f1")),
        "registered_58_packages_count": len(hits(r"Registered 58 packages")),
        "tundra_build_success_count": len(success),
        "assembly_reload_success_count": len(reloads),
        "csharp_blocking_error_count": len(csharp),
        "shader_blocking_error_count": len(shader),
        "pptr_fileid_type_mismatch_count": len(pptr),
        "ui_skin_uisprite_psd_error_count": len(uisprite),
        "csharp_blocking_errors": csharp,
        "shader_blocking_errors": shader,
        "pptr_fileid_type_mismatches": pptr,
        "ui_skin_uisprite_psd_errors": uisprite,
        "console_screen_observation": {
            "relative_path": "01_unity_open_compile_state.png",
            "info_count": 0,
            "warning_count": 10,
            "error_count": 0,
        },
    }
    result["blocking_compile_gate_pass"] = (
        result["unity_version_marker_count"] >= 1
        and result["registered_58_packages_count"] >= 1
        and result["tundra_build_success_count"] >= 1
        and result["assembly_reload_success_count"] >= 1
        and not csharp
        and not shader
        and not pptr
        and not uisprite
    )
    write_json(root / WORK_REL / "compile_import_summary.json", result)
    return result


def capture_and_analyze_targeted(root: Path, label: str) -> dict[str, Any]:
    work = root / WORK_REL
    source = (
        Path(os.environ["USERPROFILE"])
        / "AppData/LocalLow/DefaultCompany/WordsInSilence/TestResults.xml"
    )
    require(source.is_file(), "automatic TestResults.xml is missing")
    xml_data = source.read_bytes()
    target = work / "TestResults.xml"
    require(not target.exists(), f"never-overwrite XML snapshot already exists: {target}")
    target.write_bytes(xml_data)
    run = ET.fromstring(xml_data)
    fixtures = [
        node
        for node in run.findall(".//test-suite")
        if node.attrib.get("fullname") == FIXTURE
    ]
    require(len(fixtures) == 1, f"targeted fixture node count is {len(fixtures)}, expected 1")
    cases = fixtures[0].findall(".//test-case")
    inventory = read_json(root / APPROVED_REL / "targeted_28_plus_10_story_test_inventory.json")
    expected_fullnames = list(inventory["after_fullnames"])
    actual_fullnames = [case.attrib.get("fullname", "") for case in cases]
    retry_nonzero: list[str] = []
    repeat_nonzero: list[str] = []
    rows: list[dict[str, Any]] = []
    for case in cases:
        properties = {
            prop.attrib.get("name", ""): prop.attrib.get("value", "")
            for prop in case.findall("./properties/property")
        }
        retry = case.attrib.get("retryIteration", properties.get("retryIteration", "0"))
        repeat = case.attrib.get("repeatIteration", properties.get("repeatIteration", "0"))
        fullname = case.attrib.get("fullname", "")
        if retry != "0":
            retry_nonzero.append(fullname)
        if repeat != "0":
            repeat_nonzero.append(fullname)
        rows.append(
            {
                "name": case.attrib.get("name"),
                "methodname": case.attrib.get("methodname"),
                "fullname": fullname,
                "result": case.attrib.get("result"),
                "duration": case.attrib.get("duration"),
                "retryIteration": retry,
                "repeatIteration": repeat,
            }
        )
    passed = sum(row["result"] == "Passed" for row in rows)
    failed = sum(row["result"] == "Failed" for row in rows)
    skipped = sum(row["result"] == "Skipped" for row in rows)
    inconclusive = sum(row["result"] == "Inconclusive" for row in rows)

    log_path = work / "logs" / f"EditorLog_{label}.log"
    require(log_path.is_file(), f"captured targeted Editor log is missing: {log_path}")
    log_lines = log_path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    prebuild = [
        number
        for number, line in enumerate(log_lines, 1)
        if line == "Executing IPrebuildSetup for: Unity.PerformanceTesting.Editor.TestRunBuilder."
    ]
    saves = [
        {"line": number, "text": line}
        for number, line in enumerate(log_lines, 1)
        if line.startswith("Saving results to:")
    ]
    postbuild = [
        number
        for number, line in enumerate(log_lines, 1)
        if line == "Executing IPostBuildCleanup for: Unity.PerformanceTesting.Editor.TestRunBuilder."
    ]
    result = {
        "schema_version": "wordsinsilence.step10g.v1.0.gui_gate.targeted_result.v1.0",
        "fixture": FIXTURE,
        "xml_root_attributes": dict(run.attrib),
        "xml_identity": record_file(target, target.relative_to(root).as_posix()),
        "testcase_count": len(rows),
        "unique_fullname_count": len(set(actual_fullnames)),
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "inconclusive": inconclusive,
        "retry_iteration_nonzero_count": len(retry_nonzero),
        "repeat_iteration_nonzero_count": len(repeat_nonzero),
        "expected_name_set_match": set(actual_fullnames) == set(expected_fullnames),
        "expected_name_order_match": actual_fullnames == expected_fullnames,
        "missing_expected_fullnames": sorted(set(expected_fullnames) - set(actual_fullnames)),
        "unexpected_fullnames": sorted(set(actual_fullnames) - set(expected_fullnames)),
        "existing_presentation_test_count": sum(".Story" not in name for name in actual_fullnames),
        "new_story_test_count": sum(".Story" in name for name in actual_fullnames),
        "testcases": rows,
    }
    ledger = {
        "schema_version": "wordsinsilence.step10g.v1.0.gui_gate.execution_ledger.v1.0",
        "pre_run_screen": {
            "relative_path": "02_targeted_fixture_selected_not_run.png",
            "passed": 0,
            "failed": 0,
            "not_run": 38,
        },
        "first_result_screen": "03_targeted_first_result.png",
        "user_reported_run_selected_action_count": 1,
        "prebuild_count": len(prebuild),
        "prebuild_log_lines": prebuild,
        "automatic_result_save_count": len(saves),
        "automatic_result_save_log_lines": saves,
        "postbuild_count": len(postbuild),
        "postbuild_log_lines": postbuild,
        "completed_cycle_count": min(len(prebuild), len(saves), len(postbuild)),
        "retry_iteration_nonzero_count": len(retry_nonzero),
        "repeat_iteration_nonzero_count": len(repeat_nonzero),
        "targeted_execution_exactly_once": (
            len(prebuild) == 1
            and len(saves) == 1
            and len(postbuild) == 1
            and not retry_nonzero
            and not repeat_nonzero
        ),
        "targeted_rerun_performed": False,
        "full_core_152_execution_count": 0,
    }
    result["targeted_gate_pass"] = (
        len(rows) == 38
        and len(set(actual_fullnames)) == 38
        and passed == 38
        and failed == 0
        and skipped == 0
        and inconclusive == 0
        and not retry_nonzero
        and not repeat_nonzero
        and result["expected_name_set_match"]
        and ledger["targeted_execution_exactly_once"]
    )
    write_json(work / "targeted_result_summary.json", result)
    write_json(work / "execution_count_ledger.json", ledger)
    require(result["targeted_gate_pass"], "expanded targeted 38 gate failed")
    return {
        "targeted_gate_pass": True,
        "total": len(rows),
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "inconclusive": inconclusive,
        "retry_nonzero": len(retry_nonzero),
        "repeat_nonzero": len(repeat_nonzero),
        "existing_presentation_tests": result["existing_presentation_test_count"],
        "new_story_tests": result["new_story_test_count"],
        "completed_cycle_count": ledger["completed_cycle_count"],
    }


def rows_by_path(document: dict[str, Any]) -> dict[str, tuple[int, str]]:
    rows: dict[str, tuple[int, str]] = {}
    for row in document["records"]:
        identity = row.get("current", row)
        rows[row["relative_path"]] = (
            int(identity["size_bytes"]),
            str(identity["sha256"]),
        )
    return rows


def changed_identity_rows(
    before: dict[str, Any], after: dict[str, Any]
) -> list[dict[str, Any]]:
    old = rows_by_path(before)
    new = rows_by_path(after)
    return [
        {
            "relative_path": name,
            "before": old.get(name),
            "after": new.get(name),
        }
        for name in sorted(set(old) | set(new))
        if old.get(name) != new.get(name)
    ]


def verify_core_152_preservation(root: Path) -> dict[str, Any]:
    approved = read_json(root / APPROVED_REL / "core_152_testcase_preservation.json")
    rows: list[dict[str, Any]] = []
    for fixture in approved["fixture_records"]:
        expected = fixture["approved"]
        relative = expected["relative_path"]
        current = record_file(root / Path(relative), relative)
        rows.append(
            {
                "relative_path": relative,
                "test_count": fixture["test_count"],
                "test_names": fixture["test_names"],
                "approved": expected,
                "current": current,
                "source_match": identity_matches(expected, current),
            }
        )
    result = {
        "schema_version": "wordsinsilence.step10g.v1.0.gui_gate.core_152.v1.0",
        "execution_performed": False,
        "fixture_count": len(rows),
        "testcase_count": sum(int(row["test_count"]) for row in rows),
        "unique_testcase_name_count": len(
            {name for row in rows for name in row["test_names"]}
        ),
        "mismatch_count": sum(not row["source_match"] for row in rows),
        "records": rows,
        "status": "CORE_152_SOURCE_AND_NAME_IDENTITY_PRESERVED_NOT_EXECUTED",
    }
    write_json(root / WORK_REL / "core_152_preservation_after.json", result)
    return result


def postflight(root: Path) -> dict[str, Any]:
    work = root / WORK_REL
    require(work.is_dir(), "Gate work directory is missing")
    require(
        not (work / "project_identity_after.json").exists(),
        "never-overwrite postflight evidence already exists",
    )
    project = project_identity(root, "post", copy_payload=True)
    protected = protected_identity(root, "post")
    candidate = candidate_identity(root, "post")
    scene = scene_inventory(root)
    staged = git_staged(root)
    core = verify_core_152_preservation(root)

    before_project = read_json(work / "project_identity_before.json")
    before_protected = read_json(work / "protected_identity_before.json")
    before_candidate = read_json(work / "candidate_identity_before.json")
    before_scene = read_json(work / "scene_inventory_before.json")
    project_changes = changed_identity_rows(before_project, project)
    protected_changes = changed_identity_rows(before_protected, protected)
    candidate_changes = changed_identity_rows(before_candidate, candidate)
    scene_pre_post_match = identity_matches(
        before_scene["identity"], scene["identity"]
    )
    context_pre_post_match = identity_matches(
        before_project["project_context_read_only_identity"],
        project["project_context_read_only_identity"],
    )
    result = {
        "schema_version": "wordsinsilence.step10g.v1.0.gui_gate.postflight.v1.0",
        "scene_pre_post_byte_identical": scene_pre_post_match,
        "scene_sha256_before": before_scene["identity"]["sha256"],
        "scene_sha256_after": scene["identity"]["sha256"],
        "project_record_count": project["approved_record_count"],
        "project_change_count": len(project_changes),
        "project_changes": project_changes,
        "project_approved_mismatch_count": project["mismatch_count"],
        "protected_record_count": protected["record_count"],
        "protected_change_count": len(protected_changes),
        "protected_changes": protected_changes,
        "protected_approved_mismatch_count": protected["mismatch_count"],
        "candidate_count_before": before_candidate["candidate_count"],
        "candidate_count_after": candidate["candidate_count"],
        "candidate_file_count_before": before_candidate["record_count"],
        "candidate_file_count_after": candidate["record_count"],
        "candidate_media_count_after": candidate["media_file_count"],
        "candidate_change_count": len(candidate_changes),
        "candidate_changes": candidate_changes,
        "candidate_approved_mismatch_count": candidate["mismatch_count"],
        "candidate_name_set_match": candidate["record_name_set_match"],
        "project_context_pre_post_byte_identical": context_pre_post_match,
        "git_staged_count": len(staged),
        "git_staged_paths": staged,
        "core_152_source_name_preserved": (
            core["testcase_count"] == 152
            and core["unique_testcase_name_count"] == 152
            and core["mismatch_count"] == 0
        ),
    }
    result["all_required_preserved"] = (
        scene_pre_post_match
        and not project_changes
        and project["mismatch_count"] == 0
        and not protected_changes
        and protected["record_count"] == 36
        and protected["mismatch_count"] == 0
        and not candidate_changes
        and candidate["candidate_count"] == 8
        and candidate["record_count"] == 25
        and candidate["media_file_count"] == 0
        and candidate["mismatch_count"] == 0
        and candidate["record_name_set_match"]
        and context_pre_post_match
        and not staged
        and result["core_152_source_name_preserved"]
    )
    write_json(work / "project_identity_after.json", project)
    write_json(work / "protected_identity_after.json", protected)
    write_json(work / "candidate_identity_after.json", candidate)
    write_json(work / "scene_inventory_after.json", scene)
    write_json(
        work / "git_staged_after.json",
        {"staged_count": len(staged), "staged_paths": staged},
    )
    write_json(work / "fresh_pre_post_preservation.json", result)
    require(result["all_required_preserved"], "Gate 4 preservation failed")
    return result


def analyze_runtime_log(root: Path) -> dict[str, Any]:
    work = root / WORK_REL
    path = work / "logs/EditorLog_final_after_playmode.log"
    require(path.is_file(), "final Editor.log snapshot is missing")
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = text.splitlines()

    def matches(pattern: str) -> list[dict[str, Any]]:
        expression = re.compile(pattern, re.IGNORECASE)
        return [
            {"line": number, "text": line}
            for number, line in enumerate(lines, 1)
            if expression.search(line)
        ]

    blocking = matches(
        r"NullReferenceException|MissingReferenceException|"
        r"ArgumentException|InvalidOperationException|Exception:|"
        r"UnityEngine\.Debug:LogError|Assertion failed|error CS\d+|"
        r"Shader error in|PPtr cast failed|Type mismatch|UI/Skin/UISprite\.psd"
    )
    prebuild = matches(r"^Executing IPrebuildSetup for:")
    saves = matches(r"^Saving results to:")
    postbuild = matches(r"^Executing IPostBuildCleanup for:")
    reloads = matches(r"^Reloading assemblies for play mode\.$")
    pose = matches(r"^\[PoseLandmarkProvider\] PoseLandmarker")
    hands = matches(r"^\[MediaPipeLandmarkProvider\] HandLandmarker")
    expression = matches(
        r"HandleStartExpression|StartExpression\(|BeginCapture|capture started|표현 시작됨"
    )
    candidate_writes = matches(
        r"PersonalizationCandidates.*(?:write|save|commit)|"
        r"candidate.*(?:saved|committed|stored)"
    )
    result = {
        "schema_version": "wordsinsilence.step10g.v1.0.gui_gate.runtime_log.v1.0",
        "editor_log_identity": record_file(path, path.relative_to(root).as_posix()),
        "line_count": len(lines),
        "runtime_blocking_error_count": len(blocking),
        "runtime_blocking_errors": blocking,
        "targeted_prebuild_count": len(prebuild),
        "targeted_result_save_count": len(saves),
        "targeted_postbuild_count": len(postbuild),
        "playmode_transition_reload_marker_count": len(reloads),
        "single_enter_exit_cycle_interpretation": (
            "Two reload markers are the expected domain reloads on entry and exit; "
            "the user performed one Play enter and one Play exit in the same run."
        ),
        "pose_provider_initialization_log_count": len(pose),
        "hand_provider_initialization_log_count": len(hands),
        "provider_count_note": (
            "Providers log once after Play entry and once when the authored Scene is "
            "restored after exit; provider log count is not treated as Play-run count."
        ),
        "expression_start_log_marker_count": len(expression),
        "candidate_write_log_marker_count": len(candidate_writes),
        "full_core_152_execution_count": 0,
        "final_console_screen": {
            "relative_path": "18_playmode_stopped_console_state.png",
            "info_count": 2,
            "warning_count": 0,
            "error_count": 0,
        },
    }
    result["runtime_log_gate_pass"] = (
        not blocking
        and len(prebuild) == 1
        and len(saves) == 1
        and len(postbuild) == 1
        and len(reloads) == 2
        and not expression
        and not candidate_writes
    )
    write_json(work / "runtime_log_summary.json", result)
    require(result["runtime_log_gate_pass"], "runtime log gate failed")
    return result


def record_story_smoke(root: Path) -> dict[str, Any]:
    work = root / WORK_REL
    screenshots = [
        "04_story_welcome_phase_a.png",
        "05_story_welcome_phase_b_no_auto_advance.png",
        "06_story_lesson_overview.png",
        "07_story_lesson_overview_no_auto_advance.png",
        "08_story_how_to_use.png",
        "09_story_how_to_use_no_auto_advance.png",
        "10_story_greeting_introduction.png",
        "11_story_greeting_introduction_no_auto_advance.png",
        "12_story_greeting_practice_ready_phase_a.png",
        "13_story_greeting_practice_ready_phase_b.png",
        "14_webcam_skeleton_alignment_phase_a.png",
        "15_webcam_skeleton_alignment_phase_b.png",
        "16_left_yaw.png",
        "17_right_yaw.png",
        "18_playmode_stopped_console_state.png",
    ]
    missing = [name for name in screenshots if not (work / name).is_file()]
    require(not missing, "story smoke screenshots missing: " + ", ".join(missing))
    result = {
        "schema_version": "wordsinsilence.step10g.v1.0.gui_gate.story_smoke.v1.0",
        "playmode_run_count": 1,
        "playmode_exit_count": 1,
        "second_playmode_performed": False,
        "user_action_counts": {
            "run_selected": 1,
            "play_enter": 1,
            "bubble_next": 2,
            "bubble_first_expression_view": 1,
            "bubble_practice": 1,
            "left_arrow": 1,
            "right_arrow": 1,
            "play_exit": 1,
            "expression_start": 0,
            "expression_finish": 0,
            "retry": 0,
            "candidate_save": 0,
        },
        "beats": [
            {
                "beat": "Welcome",
                "body": "어서 와! 나는 MOA야. 이 연습 공간에서 네 수어를 함께 맞춰볼게.",
                "action": "다음",
                "screens": [screenshots[0], screenshots[1]],
                "automatic_advance_observed": False,
            },
            {
                "beat": "LessonOverview",
                "body": "오늘은 ‘인사’, ‘사랑’, ‘감사’ 세 표현을 함께 연습해 보자.",
                "action": "다음",
                "screens": [screenshots[2], screenshots[3]],
                "automatic_advance_observed": False,
            },
            {
                "beat": "HowToUse",
                "body": "왼쪽 시범을 보고 가운데 카메라에서 따라 하면 돼. 시스템이 동작을 인식해 결과를 알려줄게.",
                "action": "첫 표현 보기",
                "screens": [screenshots[4], screenshots[5]],
                "automatic_advance_observed": False,
            },
            {
                "beat": "GreetingIntroduction",
                "body": "첫 번째는 ‘인사’야. 손과 팔의 흐름을 보고 천천히 따라 해보자.",
                "action": "연습하기",
                "screens": [screenshots[6], screenshots[7]],
                "automatic_advance_observed": False,
                "expression_start_enabled_before_acceptance": False,
            },
            {
                "beat": "GreetingPracticeReady",
                "body": "시범을 충분히 본 뒤 준비되면 아래 ‘표현 시작’을 눌러줘.",
                "action": None,
                "screens": [screenshots[8], screenshots[9]],
                "expression_start_clicked": False,
            },
        ],
        "ui_observations": {
            "moa_name_body_single_context_action": True,
            "dark_translucent_rounded_bubble": True,
            "duplicate_legacy_progression_visible": False,
            "full_height_opaque_side_panels_visible": False,
            "central_webcam_is_primary_card": True,
            "broken_door_and_dark_environment_visible": True,
            "webcam_or_robot_excessively_occluded": False,
        },
        "webcam_observations": {
            "real_webcam_used": True,
            "rgb_orientation_natural": True,
            "mild_display_only_blur_visible": True,
            "skeleton_sharp_lime_green": True,
            "head_shoulders_above_hips": True,
            "both_hands_align_with_rgb": True,
            "horizontal_alignment_preserved": True,
            "rounded_viewport_bleed_visible": False,
        },
        "technical_regression": {
            "greeting_animation_user_confirmed_moving": True,
            "moa_robot_animation_user_confirmed_moving": True,
            "left_yaw_screen_direction_confirmed": True,
            "right_yaw_screen_direction_confirmed": True,
            "love_thank_you_forced_state_change_count": 0,
        },
        "screenshots": screenshots,
        "visual_smoke_ready_pending_external_review": True,
    }
    write_json(work / "story_smoke_observations.json", result)
    return result


def finalize_gate(root: Path) -> dict[str, Any]:
    work = root / WORK_REL
    targeted = read_json(work / "targeted_result_summary.json")
    compile_result = read_json(work / "compile_import_summary.json")
    post = read_json(work / "fresh_pre_post_preservation.json")
    runtime = analyze_runtime_log(root)
    story = record_story_smoke(root)
    ledger = read_json(work / "execution_count_ledger.json")
    ledger.update(
        {
            "user_action_counts": story["user_action_counts"],
            "playmode_single_completed_cycle_count": 1,
            "playmode_rerun_performed": False,
            "expression_start_count": 0,
            "candidate_save_count": 0,
        }
    )
    write_json(work / "execution_count_ledger.json", ledger)
    ready = (
        compile_result["blocking_compile_gate_pass"]
        and targeted["targeted_gate_pass"]
        and runtime["runtime_log_gate_pass"]
        and story["visual_smoke_ready_pending_external_review"]
        and post["all_required_preserved"]
    )
    require(ready, "gate verdict is not ready for external review")
    verdict = {
        "schema_version": "wordsinsilence.step10g.v1.0.gui_gate.verdict.v1.0",
        "overall": "READY_PENDING_CHATGPT_WORK_EXTERNAL_REVIEW",
        "gate0": "PASS",
        "gate1_compile": "PASS",
        "gate2_targeted": "PASS_38_OF_38_EXACTLY_ONCE",
        "gate3_story_smoke": "PASS_USER_CONFIRMED_SINGLE_PLAYMODE",
        "gate4_preservation": "PASS",
        "status": [
            "STEP10G_V1_0_GUI_EXPANDED_TARGETED_38_AND_STORY_SMOKE_READY_PENDING_CHATGPT_WORK_EXTERNAL_REVIEW",
            "BLOCKED_PENDING_CHATGPT_WORK_EXTERNAL_REVIEW",
        ],
        "external_pass_claimed": False,
    }
    write_json(work / "gate_verdict.json", verdict)
    report = f"""# WordsInSilence — Step 10G v1.0 GUI Gate

## Scope and verdict

- Gate result: `STEP10G_V1_0_GUI_EXPANDED_TARGETED_38_AND_STORY_SMOKE_READY_PENDING_CHATGPT_WORK_EXTERNAL_REVIEW`
- Final workflow state: `BLOCKED_PENDING_CHATGPT_WORK_EXTERNAL_REVIEW`
- External PASS is not claimed.
- Project source, Scene, Prefab, material, test source, and `WordsInSilence_PROJECT_CONTEXT.md` were not modified.
- Git stage/commit/push was not performed.

## Approved source provenance

- Delivery: `{APPROVED_ZIP_REL.as_posix()}`
- Size: {EXPECTED_DELIVERY_SIZE:,} bytes
- SHA-256: `{EXPECTED_DELIVERY_SHA256}`
- The approved source ZIP is referenced by path/hash only and is not copied into this package.

## Gate 0 — immutable pre-run snapshot

- Approved project identity: 20/20, mismatch 0.
- Authored Scene SHA-256: `{EXPECTED_SCENE_SHA256}`.
- `/Broken Door`, hologram material override, and dark Sci-Fi skybox GUIDs were present.
- Protected identity: 36/36, mismatch 0.
- Candidate repository: 8 candidates / 25 files / media 0.
- Git staged files: 0.

## Gate 1 — Unity import and compile

- Unity 6000.3.14f1 startup/import/assembly reload completed.
- C# blocking compiler errors: 0.
- Shader blocking errors: 0.
- PPtr/FileID type mismatches: 0.
- `UI/Skin/UISprite.psd` errors: 0.

## Gate 2 — expanded targeted fixture exactly once

- Fixture: `{FIXTURE}`
- Before run: 38 not run / 0 passed / 0 failed.
- Result: 38 total / 38 passed / 0 failed / 0 skipped / 0 inconclusive.
- Existing presentation tests: 28; new story tests: 10.
- Nonzero retry/repeat iterations: 0/0.
- Log cycle: prebuild 1 / result save 1 / postbuild 1.
- The user pressed Run Selected once; no rerun occurred.

## Gate 3 — single story-smoke PlayMode

- The user pressed Play once and stopped the same run once; no second PlayMode occurred.
- Welcome, Lesson Overview, How-To, and GREETING Introduction each remained stable until one explicit bubble action.
- GREETING Practice/Ready appeared only after `연습하기`; `표현 시작` was never clicked.
- The MOA bubble retained a compact dark translucent layout with one name/body/context action and no duplicate legacy progression.
- The central webcam, authored Broken Door, robot, and dark environment remained visible.
- Real webcam RGB orientation, mild display blur, sharp lime skeleton, head/shoulder/hip vertical order, both-hand alignment, and viewport clipping were visually checked.
- GREETING and MOA_robot motion were user-confirmed.
- Left and Right yaw were each entered once by the user and visually preserved.
- Runtime blocking errors: 0; final Console errors: 0.
- Expression start/finish, Retry, capture, candidate save, LOVE/THANK_YOU progression, and full core execution: 0.

## Gate 4 — post-run preservation

- Scene pre/post byte-identical: `{post['scene_sha256_after']}`.
- Approved project set: {post['project_record_count']}/{post['project_record_count']}, changes 0.
- Protected set: 36/36, changes 0.
- Candidate repository: 8 candidates / 25 files / media 0, changes 0.
- Core 152 source/name identity preserved; core suite was not executed.
- `WordsInSilence_PROJECT_CONTEXT.md` remained byte-identical.
- Git staged files: 0.

## Key evidence

- Targeted XML: `TestResults.xml`
- Full final Editor log: `logs/EditorLog_final_after_playmode.log`
- Targeted result screen: `03_targeted_first_result.png`
- Welcome through GREETING Practice/Ready: `04_...png` through `13_...png`
- Webcam body/both-hand alignment: `14_webcam_skeleton_alignment_phase_a.png`, `15_webcam_skeleton_alignment_phase_b.png`
- Left/Right yaw: `16_left_yaw.png`, `17_right_yaw.png`
- Final stopped Console: `18_playmode_stopped_console_state.png`
- Fresh pre/post identities: `project_identity_*.json`, `fresh_pre_post_preservation.json`

## Final state

`STEP10G_V1_0_GUI_EXPANDED_TARGETED_38_AND_STORY_SMOKE_READY_PENDING_CHATGPT_WORK_EXTERNAL_REVIEW`

`BLOCKED_PENDING_CHATGPT_WORK_EXTERNAL_REVIEW`
"""
    (work / "validation_report.md").write_text(report, encoding="utf-8", newline="\n")
    return verdict


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def deterministic_zip(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for name in sorted(files):
            parts = PurePosixPath(name).parts
            require(
                not name.startswith(("/", "\\")) and ".." not in parts,
                "unsafe ZIP path: " + name,
            )
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(
                info,
                files[name],
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
    return output.getvalue()


def verify_zip(blob: bytes, files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(io.BytesIO(blob), "r") as archive:
        require(archive.testzip() is None, "ZIP CRC verification failed")
        require(archive.namelist() == sorted(files), "ZIP membership/order mismatch")
        require(
            all(archive.read(name) == files[name] for name in sorted(files)),
            "ZIP payload mismatch",
        )


def package_evidence(root: Path) -> dict[str, Any]:
    work = root / WORK_REL
    required = [
        "TestResults.xml",
        "targeted_result_summary.json",
        "execution_count_ledger.json",
        "compile_import_summary.json",
        "runtime_log_summary.json",
        "story_smoke_observations.json",
        "fresh_pre_post_preservation.json",
        "gate_verdict.json",
        "validation_report.md",
        "logs/EditorLog_final_after_playmode.log",
    ]
    require(
        all((work / relative).is_file() for relative in required),
        "required gate evidence is incomplete",
    )
    result_dir = root / "Validation" / OUTPUT_NAME
    archive_path = root / "Validation" / (OUTPUT_NAME + ".zip")
    detached_path = root / "Validation" / (OUTPUT_NAME + ".zip.sha256")
    require(
        not result_dir.exists()
        and not archive_path.exists()
        and not detached_path.exists(),
        "never-overwrite output target already exists",
    )
    require(not git_staged(root), "Git staged state is not zero before packaging")

    packaging = {
        "schema_version": "wordsinsilence.step10g.v1.0.gui_gate.packaging.v1.0",
        "deterministic_sorted_membership": True,
        "fixed_zip_timestamp": "1980-01-01T00:00:00",
        "compression": "deflate-9",
        "crc_verification_required": True,
        "payload_verification_required": True,
        "path_safety_required": True,
        "never_overwrite": True,
        "approved_source_delivery_copied": False,
    }
    write_json(work / "packaging_verification.json", packaging)

    files: dict[str, bytes] = {}
    for path in sorted(work.rglob("*")):
        if path.is_file():
            files[path.relative_to(work).as_posix()] = path.read_bytes()
    helper = root / "Tools/s001_step10g_v1_0_gui_gate_evidence_v1_0.py"
    files["analysis/" + helper.name] = helper.read_bytes()
    manifest_rows = [
        record_file_from_bytes(name, files[name]) for name in sorted(files)
    ]
    files["output_file_manifest.json"] = json_bytes(
        {
            "schema_version": "wordsinsilence.step10g.v1.0.gui_gate.manifest.v1.0",
            "self_excluded": True,
            "target_count": len(manifest_rows),
            "files": manifest_rows,
        }
    )
    first = deterministic_zip(files)
    second = deterministic_zip(files)
    require(first == second, "deterministic ZIP recreation mismatch")
    verify_zip(first, files)
    zip_sha = sha256_bytes(first)

    token = os.urandom(8).hex()
    # Keep transactional staging names short enough for deeply nested evidence
    # payloads on Windows hosts that still enforce MAX_PATH.
    temp_dir = root / "Validation" / (".s10g-" + token)
    temp_zip = root / "Validation" / (".s10g-" + token + ".zip")
    temp_sha = root / "Validation" / (".s10g-" + token + ".sha")
    committed: list[Path] = []
    try:
        temp_dir.mkdir(parents=True, exist_ok=False)
        for name, data in files.items():
            target = temp_dir / Path(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        temp_zip.write_bytes(first)
        temp_sha.write_text(
            f"{zip_sha}  {archive_path.name}\n", encoding="ascii", newline="\n"
        )
        os.replace(temp_dir, result_dir)
        committed.append(result_dir)
        os.replace(temp_zip, archive_path)
        committed.append(archive_path)
        os.replace(temp_sha, detached_path)
        committed.append(detached_path)
    except Exception:
        for path in reversed(committed):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        raise
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        temp_zip.unlink(missing_ok=True)
        temp_sha.unlink(missing_ok=True)

    require(archive_path.read_bytes() == first, "published ZIP byte mismatch")
    verify_zip(archive_path.read_bytes(), files)
    require(
        detached_path.read_text(encoding="ascii").strip().split()
        == [zip_sha, archive_path.name],
        "detached checksum mismatch",
    )
    require(not git_staged(root), "Git staged state changed during packaging")
    return {
        "result_directory": str(result_dir),
        "evidence_zip": str(archive_path),
        "detached_sha256": str(detached_path),
        "zip_size_bytes": len(first),
        "zip_sha256": zip_sha,
        "entry_count": len(files),
        "deterministic_rebuild_byte_identical": True,
        "crc_and_payload_verified": True,
        "git_staged_count": 0,
    }


def record_file_from_bytes(relative_path: str, data: bytes) -> dict[str, Any]:
    return {
        "relative_path": relative_path,
        "size_bytes": len(data),
        "sha256": sha256_bytes(data),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "preflight",
            "capture-log",
            "analyze-compile",
            "capture-targeted",
            "postflight",
            "finalize",
            "package",
        ],
    )
    parser.add_argument("--label", default="snapshot")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.command == "preflight":
        print(json.dumps(preflight(root), ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "capture-log":
        print(
            json.dumps(
                capture_editor_log(root, args.label),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "analyze-compile":
        print(
            json.dumps(
                analyze_compile_log(root, args.label),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "capture-targeted":
        print(
            json.dumps(
                capture_and_analyze_targeted(root, args.label),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "postflight":
        print(
            json.dumps(
                postflight(root), ensure_ascii=False, indent=2, sort_keys=True
            )
        )
    elif args.command == "finalize":
        print(
            json.dumps(
                finalize_gate(root), ensure_ascii=False, indent=2, sort_keys=True
            )
        )
    elif args.command == "package":
        print(
            json.dumps(
                package_evidence(root), ensure_ascii=False, indent=2, sort_keys=True
            )
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
