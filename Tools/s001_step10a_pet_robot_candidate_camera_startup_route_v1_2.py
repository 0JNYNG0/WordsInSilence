#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any


SCHEMA = "WordsInSilence.Step10A.CandidateCameraStartupRoute.TestFix.v1.2"
NAME = "WordsInSilence_step10a_pet_robot_candidate_camera_startup_route_v1_2"
TARGET = "Assets/_Project/Tests/EditMode/PetRobotCandidateCameraStartupRouteV1Tests.cs"
TARGET_META = TARGET + ".meta"
SCENE = "Assets/_Project/Scenes/VerticalSlice/PetRobotSignPracticeVerticalSlice.unity"
BEFORE_SHA = "c730ee9382c93cdd3224fe333f06ab76f342514515c1673d0cc0d40017debc84"
AFTER_SHA = "f1bc9d351a9802cca8fe8d951370f07d4e4fc4ec01df1e19b42904ca94be598c"
META_SHA = "a2711656f0b873a9c462170d6ac8e125a607a613a33c142883b54834ea7caed2"

V11_ZIP = "Validation/WordsInSilence_step10a_pet_robot_candidate_camera_startup_route_v1_1_delivery.zip"
V11_ZIP_SIZE = 108034
V11_ZIP_SHA = "f5ccb3438728d8d96c4bcff68c2fcccccc23d1afc92140d0f12d8294af773518"
V11_DETACHED_SHA = "30a89d3af38be88ef5052dcc86a7933f85c6b690f148d8c30b4d7e56822a1d02"
GATE_ZIP = "Validation/WordsInSilence_step10a_pet_robot_candidate_camera_startup_route_v1_1_gui_unity_editor_gate_delivery.zip"
GATE_ZIP_SIZE = 393403
GATE_ZIP_SHA = "c140d366b40cbd822bb22d7691050d84e08f806094c4eb2a52a2d793a421708d"
GATE_DETACHED_SHA = "02702e9c386b4fa46217a6cf772b741c0ff2182171bf5f1f6a433965f4e4260a"
GATE_XML_SHA = "557b068fd7ba758f39df92fb5025de781069b7b2667914e74de9e1aaf94ec497"
V11_TOOL = "Tools/s001_step10a_pet_robot_candidate_camera_startup_route_v1_1.py"
PREFLIGHT = "Validation/.step10a-v1_2-preflight"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def jbytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def record(path: str, data: bytes, **extra: Any) -> dict[str, Any]:
    return {"relative_path": path, "size_bytes": len(data), "sha256": sha(data), **extra}


def load_v11(root: Path):
    path = root / V11_TOOL
    spec = importlib.util.spec_from_file_location("step10a_v11_approved", path)
    require(spec is not None and spec.loader is not None, "cannot load approved v1.1 packaging tool")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def current_records(root: Path, baseline: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for before in baseline["records"]:
        relative = before["relative_path"]
        path = root / relative
        require(path.is_file(), "project/protected file missing: " + relative)
        data = path.read_bytes()
        rows.append(record(relative, data,
                           before_size_bytes=before["size_bytes"],
                           before_sha256=before["sha256"],
                           changed=sha(data) != before["sha256"]))
    changed = [row["relative_path"] for row in rows if row["changed"]]
    return {
        "schema_version": SCHEMA + ".file_set",
        "file_count": len(rows),
        "unchanged_count": len(rows) - len(changed),
        "changed_count": len(changed),
        "changed_paths": changed,
        "records": rows,
        "status": "PASS",
    }


def snapshot_candidate_repository(before: dict[str, Any]) -> dict[str, Any]:
    raw_path = before.get("path") or before.get("root_path")
    require(bool(raw_path), "candidate repository before path missing")
    root = Path(raw_path)
    rows = []
    if root.is_dir():
        for path in sorted(value for value in root.rglob("*") if value.is_file()):
            rows.append(record(path.relative_to(root).as_posix(), path.read_bytes()))
    return {
        "path": str(root),
        "exists": root.exists(),
        "file_count": len(rows),
        "files": rows,
    }


def staged_files(root: Path) -> list[str]:
    run = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=root,
                         capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    require(run.returncode == 0, "Git staged query failed")
    return [line.strip() for line in run.stdout.splitlines() if line.strip()]


def extract_count_helper(text: str) -> str:
    match = re.search(
        r"        static int Count\(string text, string value\)\s*\{.*?^        \}",
        text, flags=re.MULTILINE | re.DOTALL)
    require(match is not None, "generic Count helper missing")
    return match.group(0)


def exact_field_regression(root: Path, before_source: bytes, after_source: bytes) -> dict[str, Any]:
    scene = (root / SCENE).read_text(encoding="utf-8")
    candidate = "  cameraDisplay: {fileID: 447812208}"
    legacy = "  _cameraDisplay: {fileID: 447812208}"
    raw = "  _rawImage: {fileID: 447812208}"
    lines = [line.rstrip("\r") for line in scene.split("\n")]
    values = {
        "naive_substring_count": scene.count("cameraDisplay: {fileID: 447812208}"),
        "exact_camera_display_field_line_count": lines.count(candidate),
        "exact_legacy_camera_display_field_line_count": lines.count(legacy),
        "exact_raw_image_field_line_count": lines.count(raw),
    }
    require(values == {
        "naive_substring_count": 2,
        "exact_camera_display_field_line_count": 1,
        "exact_legacy_camera_display_field_line_count": 1,
        "exact_raw_image_field_line_count": 2,
    }, "fixed Scene exact-field regression mismatch")
    before = before_source.decode("utf-8")
    after = after_source.decode("utf-8")
    require(extract_count_helper(before) == extract_count_helper(after), "generic Count helper changed")
    require("scene.Split('\\n').Count(line =>" in after, "exact-line LINQ assertion missing")
    require('"  cameraDisplay: {fileID: 447812208}"' in after, "exact candidate field token missing")
    require('Assert.AreEqual(2, Count(scene, "_rawImage: {fileID: 447812208}"));' in after,
            "raw image assertion changed")
    require('StringAssert.Contains("captureConfig: {fileID: 11400000, guid: c7817e503402bd8498b8358b680830a9, type: 2}", scene);' in after,
            "captureConfig assertion changed")
    require('StringAssert.Contains("cameraRoute.PreviewTexture", ui);' in after,
            "PreviewTexture assertion changed")
    return {
        "schema_version": SCHEMA + ".exact_field_regression",
        **values,
        "underscore_camera_display_excluded_from_candidate_count": True,
        "generic_count_helper_unchanged": True,
        "raw_image_capture_config_preview_assertions_preserved": True,
        "status": "PASS",
    }


def method_inventory(root: Path, approved: dict[str, Any]) -> dict[str, Any]:
    expected = {(row["fixture"], row["method"], row["source_path"]) for row in approved["rows"]}
    sources = sorted({row["source_path"] for row in approved["rows"]})
    actual = set()
    rows = []
    for relative in sources:
        text = (root / relative).read_text(encoding="utf-8-sig")
        class_match = re.search(r"public\s+sealed\s+class\s+(\w+)", text)
        require(class_match is not None, "fixture class missing: " + relative)
        fixture = class_match.group(1)
        methods = re.findall(r"\[Test\]\s*public\s+void\s+(\w+)\s*\(", text)
        for method in methods:
            row = {"fixture": fixture, "method": method, "source_path": relative}
            rows.append(row)
            actual.add((fixture, method, relative))
    require(actual == expected, "87-test inventory changed")
    require(len(rows) == len(actual) == 87, "test inventory is not 87 unique methods")
    new_rows = [row for row in rows if row["fixture"] == "PetRobotCandidateCameraStartupRouteV1Tests"]
    require(len(new_rows) == 8, "candidate camera fixture count changed")
    return {
        "schema_version": SCHEMA + ".test_inventory",
        "preserved_existing_test_count": 79,
        "candidate_camera_test_count": 8,
        "combined_unique_test_count": 87,
        "fixture_count": len({row["fixture"] for row in rows}),
        "rows": sorted(rows, key=lambda row: (row["fixture"], row["method"])),
        "status": "PASS",
    }


def failure_identity(gate_files: dict[str, bytes]) -> dict[str, Any]:
    xml = gate_files["TestResults.xml"]
    require(len(xml) == 75210 and sha(xml) == GATE_XML_SHA, "first GUI failure XML identity mismatch")
    summary = json.loads(gate_files["test_results_summary.json"])
    require(summary["unique_testcase_count"] == 87 and summary["passed"] == 86 and
            summary["failed"] == 1 and summary["skipped"] == 0 and summary["inconclusive"] == 0,
            "first GUI failure counts mismatch")
    require(summary["retry_nonzero"] == 0 and summary["repeat_nonzero"] == 0,
            "first GUI failure retry/repeat mismatch")
    failure = summary["failures"][0]
    require(failure["fullname"].endswith(
        "PetRobotCandidateCameraStartupRouteV1Tests.PreviewAndHandPoseProviders_UseTheSameSerializedFrameSource"),
        "unexpected fixed failing test")
    require("Expected: 1" in failure["message"] and "But was:  2" in failure["message"],
            "unexpected fixed failure values")
    return {
        "schema_version": SCHEMA + ".first_failure",
        "unity_version": "6000.3.14f1",
        "compiler_error_count": 0,
        "execution_count": 1,
        "unique_test_count": 87,
        "passed": 86,
        "failed": 1,
        "skipped": 0,
        "inconclusive": 0,
        "retry_nonzero": 0,
        "repeat_nonzero": 0,
        "failing_test": failure["fullname"],
        "expected": 1,
        "actual": 2,
        "xml": record("TestResults.xml", xml),
        "status": "PASS",
    }


def unified(before: bytes, after: bytes) -> bytes:
    return "".join(difflib.unified_diff(
        before.decode("utf-8").splitlines(True),
        after.decode("utf-8").splitlines(True),
        fromfile="v1.1/" + TARGET,
        tofile="v1.2/" + TARGET,
    )).encode("utf-8")


def assertion_lines(rows: list[tuple[str, str, str]]) -> bytes:
    keys = set()
    result = []
    for key, status, evidence in rows:
        require(key not in keys, "duplicate assertion key: " + key)
        keys.add(key)
        result.append(json.dumps({"assertion_key": key, "status": status, "evidence": evidence},
                                 ensure_ascii=False, sort_keys=True, allow_nan=False))
    return ("\n".join(result) + "\n").encode("utf-8")


def add_manifest(files: dict[str, bytes]) -> dict[str, bytes]:
    require("output_file_manifest.json" not in files, "manifest already present")
    manifest = {
        "schema_version": SCHEMA + ".manifest",
        "self_excluded": True,
        "file_count": len(files),
        "files": [record(name, files[name]) for name in sorted(files)],
    }
    result = dict(files)
    result["output_file_manifest.json"] = jbytes(manifest)
    return result


def report(exact: dict[str, Any], inventory: dict[str, Any], compile_evidence: dict[str, Any],
           project: dict[str, Any], protected: dict[str, Any]) -> bytes:
    return f"""# Step 10A v1.2 Test-only Shared Camera Source Exact-field Assertion Fix

## Outcome

- Only `{TARGET}` changed.
- The candidate `cameraDisplay` assertion now counts one exact Unity YAML field line, so `_cameraDisplay` is not included.
- `_rawImage` count 2, `captureConfig`, `PreviewTexture`, the generic substring helper, all fixture names, and all production/Scene/Prefab/assets remain unchanged.

## Fixed regression

- Naive substring count: {exact['naive_substring_count']}
- Exact `cameraDisplay` field-line count: {exact['exact_camera_display_field_line_count']}
- Exact `_cameraDisplay` field-line count: {exact['exact_legacy_camera_display_field_line_count']}
- Exact `_rawImage` field-line count: {exact['exact_raw_image_field_line_count']}
- Test inventory: 79 preserved + 8 candidate-camera = {inventory['combined_unique_test_count']} unique.
- Offline Unity Mono Roslyn production/test compile errors: {compile_evidence['production_error_count']}/{compile_evidence['test_error_count']}.
- Step 10A project files: changed {project['changed_count']} of {project['file_count']} (the allowed test source only).
- Protected files: {protected['unchanged_count']}/{protected['file_count']} byte-identical.

Unity GUI, batchmode, Test Runner, PlayMode, webcam, and candidate storage were not executed. The fixed first GUI result remains 86 PASS / 1 FAIL evidence; a fresh 87-test Editor gate is deferred until external review of this source/Delivery.

## Status

```text
STEP10A_V1_2_TEST_ONLY_SHARED_CAMERA_SOURCE_EXACT_FIELD_ASSERTION_FIX_SOURCE_AND_PACKAGE_READY
STEP10A_V1_1_PRODUCTION_SCENE_PREFAB_ASSETS_PRESERVED
STEP10A_V1_2_OFFLINE_EXACT_FIELD_REGRESSION_PASS
STEP10A_V1_2_UNITY_87_TEST_RETEST_UNVERIFIED
BLOCKED_PENDING_EXTERNAL_REVIEW
```
""".encode("utf-8")


def build(root: Path, v11: Any) -> tuple[dict[str, bytes], dict[str, Any]]:
    preflight = root / PREFLIGHT
    require(preflight.is_dir(), "v1.2 preflight snapshot missing")
    fixed_before = load_json(preflight / "fixed_input_preflight.json")
    project_before = load_json(preflight / "project_files_before.json")
    protected_before = load_json(preflight / "protected_files_before.json")
    profile_before = load_json(preflight / "active_candidate_profile_before.json")
    repository_before_doc = load_json(preflight / "candidate_repository_before.json")
    git_before_doc = load_json(preflight / "git_staged_before.json")
    source_before = (preflight / "source_before/PetRobotCandidateCameraStartupRouteV1Tests.cs").read_bytes()
    meta_before = (preflight / "source_before/PetRobotCandidateCameraStartupRouteV1Tests.cs.meta").read_bytes()
    require(sha(source_before) == BEFORE_SHA, "test source before hash mismatch")
    require(sha(meta_before) == META_SHA, "test meta before hash mismatch")

    v11_info, v11_files = v11.verify_fixed_zip(root, V11_ZIP, V11_ZIP_SIZE, V11_ZIP_SHA, V11_DETACHED_SHA)
    gate_info, gate_files = v11.verify_fixed_zip(root, GATE_ZIP, GATE_ZIP_SIZE, GATE_ZIP_SHA, GATE_DETACHED_SHA)
    require(fixed_before["records"][0]["sha256"] == V11_ZIP_SHA and
            fixed_before["records"][1]["sha256"] == GATE_ZIP_SHA,
            "preflight fixed input records mismatch")

    source_after = (root / TARGET).read_bytes()
    meta_after = (root / TARGET_META).read_bytes()
    require(sha(source_after) == AFTER_SHA, "test source after hash mismatch")
    require(sha(meta_after) == META_SHA and meta_after == meta_before, "test meta changed")
    diff = unified(source_before, source_after)
    diff_text = diff.decode("utf-8")
    require(diff_text.count("@@") == 2 and "scene.Split('\\n').Count" in diff_text,
            "unexpected unified diff")

    project_after = current_records(root, project_before)
    require(project_after["file_count"] == 30 and project_after["changed_paths"] == [TARGET],
            "project change scope is not exactly the allowed test source")
    protected_after = current_records(root, protected_before)
    require(protected_after["file_count"] == 106 and protected_after["changed_count"] == 0,
            "protected files changed")
    profile_after = current_records(root, profile_before)
    require(profile_after["file_count"] == 22 and profile_after["changed_count"] == 0,
            "candidate profile files changed")
    require(profile_before["prototype_count"] == 9 and profile_before["threshold_count"] == 3 and
            profile_before["thresholds"] == {
                "GREETING": 0.5113059473720765,
                "LOVE": 0.391276023460427,
                "THANK_YOU": 0.37709977431991587,
            }, "candidate prototype/threshold baseline mismatch")

    repository_before_raw = repository_before_doc.get("snapshot", repository_before_doc)
    repository_before = {
        "path": repository_before_raw["path"],
        "exists": repository_before_raw["exists"],
        "file_count": repository_before_raw["file_count"],
        "files": repository_before_raw["files"],
    }
    repository_after = snapshot_candidate_repository(repository_before)
    require(repository_after == repository_before, "candidate repository changed")
    git_before = git_before_doc.get("staged_files", [])
    git_after = staged_files(root)
    require(git_before == [] and git_after == [], "Git staged state changed")

    exact = exact_field_regression(root, source_before, source_after)
    approved_inventory = json.loads(v11_files["test_inventory.json"])
    inventory = method_inventory(root, approved_inventory)
    failure = failure_identity(gate_files)

    compile_evidence, production_output, test_output, temp, _ = v11.offline_compile(root)
    try:
        require(compile_evidence["production_error_count"] == 0 and
                compile_evidence["test_error_count"] == 0,
                "offline compile errors are non-zero")
    finally:
        shutil.rmtree(temp, ignore_errors=True)
    compile_evidence = dict(compile_evidence)
    compile_evidence.update({
        "schema_version": SCHEMA + ".offline_compile",
        "approved_v1_1_compile_contract_reused": True,
        "unity_gui_compile_claimed": False,
        "unity_editmode_pass_claimed": False,
    })
    transaction = v11.transaction_regression()

    input_provenance = {
        "schema_version": SCHEMA + ".provenance",
        "implementation_owner": "Codex",
        "v1_1_source_delivery": v11_info,
        "first_gui_failure_delivery": gate_info,
        "first_gui_failure": failure,
        "context": record("WordsInSilence_PROJECT_CONTEXT.md",
                          (root / "WordsInSilence_PROJECT_CONTEXT.md").read_bytes()),
        "status": "PASS",
    }
    assertions = assertion_lines([
        ("implementation_owner", "PASS", "Codex"),
        ("v1_1_delivery_fixed", "PASS", V11_ZIP_SHA),
        ("first_gui_failure_delivery_fixed", "PASS", GATE_ZIP_SHA),
        ("first_gui_failure_xml_fixed", "PASS", GATE_XML_SHA),
        ("first_gui_failure_identity", "PASS", "86 PASS / 1 FAIL; Expected 1 Actual 2"),
        ("single_allowed_source_changed", "PASS", TARGET),
        ("test_meta_identity", "PASS", META_SHA),
        ("naive_substring_regression", "PASS", "2"),
        ("exact_camera_display_field", "PASS", "1"),
        ("exact_legacy_camera_display_field", "PASS", "1"),
        ("exact_raw_image_field", "PASS", "2"),
        ("generic_count_helper_preserved", "PASS", "byte-identical method body"),
        ("related_assertions_preserved", "PASS", "_rawImage/captureConfig/PreviewTexture"),
        ("candidate_fixture_inventory", "PASS", "8"),
        ("preserved_test_inventory", "PASS", "79"),
        ("combined_unique_test_inventory", "PASS", "87"),
        ("offline_production_compile", "PASS", "errors 0"),
        ("offline_test_compile", "PASS", "errors 0"),
        ("project_file_scope", "PASS", "1 changed / 30"),
        ("protected_file_identity", "PASS", "106/106"),
        ("candidate_profile_identity", "PASS", "22/22"),
        ("prototype_threshold_identity", "PASS", "9 prototypes / 3 thresholds"),
        ("candidate_repository_unchanged", "PASS", str(repository_after["file_count"])),
        ("packaging_transaction_rollback", "PASS", "4/4 injected failures"),
        ("never_overwrite", "PASS", "second commit blocked"),
        ("git_staged_files", "PASS", "0"),
        ("unity_87_test_retest", "UNVERIFIED", "deferred until external review"),
        ("playmode_webcam_candidate_storage", "NOT_APPLICABLE", "prohibited in test-fix source step"),
    ])

    files: dict[str, bytes] = {
        "input_provenance.json": jbytes(input_provenance),
        "test_fix_contract.json": jbytes({
            "schema_version": SCHEMA + ".contract",
            "allowed_modified_project_files": [TARGET],
            "production_scene_prefab_asset_changes_allowed": False,
            "generic_count_helper_change_allowed": False,
            "unity_execution_allowed": False,
            "status": "PASS",
        }),
        "source_identity_before_after.json": jbytes({
            "schema_version": SCHEMA + ".source_identity",
            "before": record(TARGET, source_before),
            "after": record(TARGET, source_after),
            "meta_before": record(TARGET_META, meta_before),
            "meta_after": record(TARGET_META, meta_after),
            "only_test_source_changed": True,
            "status": "PASS",
        }),
        "exact_yaml_field_line_regression.json": jbytes(exact),
        "first_gui_failure_identity.json": jbytes(failure),
        "test_inventory.json": jbytes(inventory),
        "offline_compile_evidence.json": jbytes(compile_evidence),
        "offline_production_compile_output.txt": production_output.encode("utf-8"),
        "offline_test_compile_output.txt": test_output.encode("utf-8"),
        "project_files_before.json": jbytes(project_before),
        "project_files_after.json": jbytes(project_after),
        "protected_files_before.json": jbytes(protected_before),
        "protected_files_after.json": jbytes(protected_after),
        "active_candidate_profile_before.json": jbytes(profile_before),
        "active_candidate_profile_after.json": jbytes(profile_after),
        "candidate_repository_before.json": jbytes(repository_before),
        "candidate_repository_after.json": jbytes(repository_after),
        "git_staged_before.json": jbytes({"staged_count": 0, "staged_files": git_before}),
        "git_staged_after.json": jbytes({"staged_count": 0, "staged_files": git_after}),
        "packaging_transaction_failure_injection_regression.json": jbytes(transaction),
        "source_before/Assets___Project__Tests__EditMode__PetRobotCandidateCameraStartupRouteV1Tests.cs": source_before,
        "source_after/Assets___Project__Tests__EditMode__PetRobotCandidateCameraStartupRouteV1Tests.cs": source_after,
        "source_before/Assets___Project__Tests__EditMode__PetRobotCandidateCameraStartupRouteV1Tests.cs.meta": meta_before,
        "source_after/Assets___Project__Tests__EditMode__PetRobotCandidateCameraStartupRouteV1Tests.cs.meta": meta_after,
        "diff/PetRobotCandidateCameraStartupRouteV1Tests.cs.diff": diff,
        "project_files/" + TARGET: source_after,
        "project_files/" + TARGET_META: meta_after,
        "input_subset/first_gui_failure/TestResults.xml": gate_files["TestResults.xml"],
        "input_subset/first_gui_failure/test_results_summary.json": gate_files["test_results_summary.json"],
        "input_subset/first_gui_failure/editmode_failure_diagnosis.json": gate_files["editmode_failure_diagnosis.json"],
        "input_subset/first_gui_failure/primary_verdict.json": gate_files["primary_verdict.json"],
        "input_subset/first_gui_failure/detached_checksum.txt": (root / (GATE_ZIP + ".sha256")).read_bytes(),
        "input_subset/v1_1/test_inventory.json": v11_files["test_inventory.json"],
        "input_subset/v1_1/implementation_delta.json": v11_files["implementation_delta.json"],
        "input_subset/v1_1/detached_checksum.txt": (root / (V11_ZIP + ".sha256")).read_bytes(),
        "ASSERTIONS.jsonl": assertions,
        "VALIDATION_REPORT.md": report(exact, inventory, compile_evidence, project_after, protected_after),
        "analysis/" + Path(__file__).name: Path(__file__).read_bytes(),
    }
    return files, {
        "project": project_after,
        "protected": protected_after,
        "profile": profile_after,
        "repository_before": repository_before,
        "repository_after": repository_after,
        "inventory": inventory,
        "compile": compile_evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    v11 = load_v11(root)
    files, summary = build(root, v11)
    files = add_manifest(files)
    validation = root / "Validation"
    names = (NAME, NAME + "_delivery.zip", NAME + "_delivery.zip.sha256")
    result_dir, delivery, detached = v11.commit_outputs(
        validation, files, None, names, v11.post_verify)
    preflight = (root / PREFLIGHT).resolve()
    require(preflight.parent == validation.resolve() and preflight.name == ".step10a-v1_2-preflight",
            "unsafe preflight cleanup target")
    shutil.rmtree(preflight)
    print(json.dumps({
        "status": "PASS",
        "result_directory": str(result_dir),
        "result_file_count": len(files),
        "delivery_zip": str(delivery),
        "delivery_size_bytes": delivery.stat().st_size,
        "delivery_sha256": sha(delivery.read_bytes()),
        "detached_checksum": str(detached),
        "detached_size_bytes": detached.stat().st_size,
        "detached_sha256": sha(detached.read_bytes()),
        "changed_project_files": summary["project"]["changed_paths"],
        "protected_files_unchanged": summary["protected"]["unchanged_count"],
        "candidate_profile_files_unchanged": summary["profile"]["unchanged_count"],
        "combined_test_inventory": summary["inventory"]["combined_unique_test_count"],
        "offline_compile_errors": [summary["compile"]["production_error_count"],
                                   summary["compile"]["test_error_count"]],
        "unity_87_test_retest": "UNVERIFIED",
        "candidate_repository_file_count_before_after": [summary["repository_before"]["file_count"],
                                                           summary["repository_after"]["file_count"]],
        "git_staged_files": 0,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
