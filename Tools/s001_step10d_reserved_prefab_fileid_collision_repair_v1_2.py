#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.util
import json
import re
import subprocess
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA = "WordsInSilence.Step10D.ReservedPrefabFileIdCollisionRepair.v1.2"
NAME = "WordsInSilence_step10d_reserved_prefab_fileid_collision_repair_v1_2"

V1_ZIP = (
    "Validation/WordsInSilence_step10d_existing_fbx_animator_floating_pet_robot_"
    "companion_presentation_layer_v1_0_delivery.zip"
)
V1_ZIP_SIZE = 19_270_682
V1_ZIP_SHA = "2c51fb71d6bf39057f82ab66a68a6e5ee9863b553d81a6535740d1864f53ae52"
V1_DETACHED_SIZE = 177
V1_DETACHED_SHA = "4e25ebc23a0f6bd1253a8bd03fb4c2f9bee95f6dce607a68ac4dbe27b0f3b6a3"

GUI_ZIP = (
    "Validation/WordsInSilence_step10d_existing_fbx_animator_floating_pet_robot_"
    "companion_presentation_layer_v1_0_gui_unity_editor_gate_delivery.zip"
)
GUI_ZIP_SIZE = 677_006
GUI_ZIP_SHA = "b9e50390db15c02c451d2b97dc5cf55bc77156260e91617f2b2fd991801762b3"
GUI_DETACHED_SIZE = 199
GUI_DETACHED_SHA = "63a677fdab335c8b88bf247650aef03a307d5e24d1378399b2e99e49cb21c7f1"

V1_RESULT = (
    "Validation/WordsInSilence_step10d_existing_fbx_animator_floating_pet_robot_"
    "companion_presentation_layer_v1_0"
)
TX_HELPER = "Tools/s001_step10a_pet_robot_candidate_camera_startup_route_v1_1.py"

WRAPPER = "Assets/_Project/Resources/VerticalSlice/PetRobotAnimatedVisualWrapper.prefab"
WRAPPER_META = WRAPPER + ".meta"
TEST_SOURCE = "Assets/_Project/Tests/EditMode/PetRobotFloatingCompanionPresentationV1Tests.cs"
TEST_META = TEST_SOURCE + ".meta"
PRESENTATION = "Assets/_Project/Resources/VerticalSlice/PetRobotPortfolioQuestPresentation.prefab"
BUILD_SETTINGS = "ProjectSettings/EditorBuildSettings.asset"
CHECKPOINT = V1_RESULT + "/asset_identity_and_guid_graph.json"

WRAPPER_BEFORE_SIZE = 2_706
WRAPPER_BEFORE_SHA = "ce33933e637890a2332628f3c43675b4d803bb69f276bda57e624f0b359ec984"
TEST_BEFORE_SIZE = 11_139
TEST_BEFORE_SHA = "4cfeccec7c4f5f86a3fa8f1377b495075a01d8dfd30bf70eaaad61b93dd872dd"
WRAPPER_META_SHA = "e81319aa0a032e04442c0b20694bcdf9bbacbc3ea822718cf0f2a93dcf484e3e"
TEST_META_SHA = "25aab8ff47e163defcdeba3deea693f88cbca73f841bef171846c6e645c10743"
BUILD_SETTINGS_SHA = "584bc074ee61f0225ddfb52b8f9ef3a64825ea7fffd5815c8c8d5e96966c2820"

ROBOT_FBX_GUID = "50573fbff0bc00c418eb7319a40093d3"
WRAPPER_GUID = "10d00000000000000000000000000010"
CONTROLLER_GUID = "a414e66cfd19823419a0f9a1954db8d3"
ANIMATION_GUID = "8a56dbc85c0c9b74381607d7d13f5e52"

RESERVED_DOCUMENT = "--- !u!1001 &100100000"
NEW_DOCUMENT = "--- !u!1001 &100200000"
OLD_LOCAL_POINTER = "  m_PrefabInstance: {fileID: 100100000}"
NEW_LOCAL_POINTER = "  m_PrefabInstance: {fileID: 100200000}"
EXTERNAL_SOURCE = (
    "  m_SourcePrefab: {fileID: 100100000, guid: " + ROBOT_FBX_GUID + ", type: 3}"
)

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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_module(root: Path, relative_path: str, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, root / relative_path)
    require(spec is not None and spec.loader is not None,
            "cannot load helper: " + relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def staged_files(root: Path) -> list[str]:
    run = subprocess.run(
        ["git", "diff", "--cached", "--name-only"], cwd=root,
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
    )
    require(run.returncode == 0, "Git staged query failed")
    return [line.strip() for line in run.stdout.splitlines() if line.strip()]


def unsafe_zip_path(name: str) -> bool:
    path = PurePosixPath(name)
    return path.is_absolute() or ".." in path.parts or "\\" in name or not name


def verify_fixed_zip(root: Path, relative: str, size: int, digest: str,
                     detached_size: int, detached_digest: str) -> dict[str, Any]:
    path = root / relative
    detached = Path(str(path) + ".sha256")
    require(path.is_file() and detached.is_file(), "fixed archive missing: " + relative)
    blob = path.read_bytes()
    check = detached.read_bytes()
    require(len(blob) == size and sha(blob) == digest, "fixed archive identity mismatch")
    require(len(check) == detached_size and sha(check) == detached_digest,
            "fixed detached identity mismatch")
    fields = check.decode("ascii").strip().split()
    require(fields == [digest, path.name], "fixed detached content mismatch")
    with zipfile.ZipFile(path, "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        require(len(names) == len(set(names)), "fixed archive duplicate entry")
        require(not any(unsafe_zip_path(name) for name in names),
                "fixed archive unsafe path")
        require(not any(info.flag_bits & 1 for info in infos),
                "fixed archive encrypted entry")
        require(not any(((info.external_attr >> 16) & 0o170000) == 0o120000
                        for info in infos), "fixed archive symlink entry")
        require(archive.testzip() is None, "fixed archive CRC failure")
        manifest = json.loads(archive.read("output_file_manifest.json"))
        members = {row["relative_path"] for row in manifest["files"]}
        require(members == set(names) - {"output_file_manifest.json"},
                "fixed archive manifest membership mismatch")
        for row in manifest["files"]:
            payload = archive.read(row["relative_path"])
            require(len(payload) == row["size_bytes"] and sha(payload) == row["sha256"],
                    "fixed archive manifest payload mismatch")
    return {
        "zip": record(relative, blob),
        "detached_checksum": record(relative + ".sha256", check),
        "entry_count": len(names),
        "manifest_target_count": len(members),
        "status": "PASS",
    }


def v1_project_files(root: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    path = root / V1_ZIP
    with zipfile.ZipFile(path, "r") as archive:
        entries = {
            name[len("project_files/"):]: archive.read(name)
            for name in archive.namelist() if name.startswith("project_files/")
        }
    require(len(entries) == 22, "v1.0 project file count mismatch")
    require(WRAPPER in entries and TEST_SOURCE in entries, "v1.0 changed source missing")
    require(len(entries[WRAPPER]) == WRAPPER_BEFORE_SIZE and
            sha(entries[WRAPPER]) == WRAPPER_BEFORE_SHA,
            "v1.0 wrapper before identity mismatch")
    require(len(entries[TEST_SOURCE]) == TEST_BEFORE_SIZE and
            sha(entries[TEST_SOURCE]) == TEST_BEFORE_SHA,
            "v1.0 test before identity mismatch")
    rows = []
    changed = []
    for relative, before in sorted(entries.items()):
        path = root / relative
        require(path.is_file(), "project file missing: " + relative)
        current = path.read_bytes()
        same = current == before
        if not same:
            changed.append(relative)
        rows.append({
            "relative_path": relative,
            "v1_0_size_bytes": len(before),
            "v1_0_sha256": sha(before),
            "current_size_bytes": len(current),
            "current_sha256": sha(current),
            "byte_identical_to_v1_0": same,
            "change_authorized_v1_2": relative in {WRAPPER, TEST_SOURCE},
            "status": "PASS" if same or relative in {WRAPPER, TEST_SOURCE} else "FAIL",
        })
    require(changed == sorted([WRAPPER, TEST_SOURCE]),
            "project change scope is not exactly two authorized files")
    require(all(row["status"] == "PASS" for row in rows),
            "unauthorized project file change")
    return entries, {
        "schema_version": SCHEMA + ".project_change_scope",
        "v1_0_project_file_count": len(entries),
        "changed_project_file_count": len(changed),
        "changed_project_files": changed,
        "unchanged_project_file_count": len(entries) - len(changed),
        "rows": rows,
        "status": "PASS",
    }


def expected_repairs(before_wrapper: bytes, before_test: bytes) -> tuple[bytes, bytes]:
    wrapper_text = before_wrapper.decode("utf-8")
    require(wrapper_text.count(RESERVED_DOCUMENT) == 1 and
            wrapper_text.count(OLD_LOCAL_POINTER) == 1,
            "v1.0 wrapper collision shape mismatch")
    expected_wrapper = wrapper_text.replace(RESERVED_DOCUMENT, NEW_DOCUMENT, 1)
    expected_wrapper = expected_wrapper.replace(OLD_LOCAL_POINTER, NEW_LOCAL_POINTER, 1)

    test_text = before_test.decode("utf-8")
    old = (
        '            Assert.AreEqual(1, Count(wrapper, "guid: '
        + CONTROLLER_GUID + '"));\n'
        '            StringAssert.Contains("propertyPath: m_Controller", wrapper);'
    )
    new = (
        '            Assert.AreEqual(1, Count(wrapper, "guid: '
        + CONTROLLER_GUID + '"));\n'
        '            Assert.AreEqual(0, Count(wrapper, "--- !u!1001 &100100000"));\n'
        '            Assert.AreEqual(1, Count(wrapper, "--- !u!1001 &100200000"));\n'
        '            Assert.AreEqual(1, Count(wrapper, "  m_PrefabInstance: {fileID: 100200000}"));\n'
        '            Assert.AreEqual(1, Count(wrapper,\n'
        '                "m_SourcePrefab: {fileID: 100100000, guid: '
        + ROBOT_FBX_GUID + ', type: 3}"));\n'
        '            StringAssert.Contains("propertyPath: m_Controller", wrapper);'
    )
    require(test_text.count(old) == 1, "v1.0 fixture insertion point mismatch")
    expected_test = test_text.replace(old, new, 1)
    return expected_wrapper.encode("utf-8"), expected_test.encode("utf-8")


def diff_blob(relative: str, before: bytes, after: bytes) -> tuple[bytes, int, int]:
    before_lines = before.decode("utf-8").splitlines()
    after_lines = after.decode("utf-8").splitlines()
    diff = "\n".join(difflib.unified_diff(
        before_lines, after_lines,
        fromfile="before/" + relative,
        tofile="after/" + relative,
        lineterm="",
    )) + "\n"
    additions = deletions = 0
    matcher = difflib.SequenceMatcher(a=before_lines, b=after_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in {"replace", "delete"}:
            deletions += i2 - i1
        if tag in {"replace", "insert"}:
            additions += j2 - j1
    return diff.encode("utf-8"), additions, deletions


def validate_wrapper(root: Path, before: bytes, after: bytes) -> dict[str, Any]:
    text = after.decode("utf-8")
    before_text = before.decode("utf-8")
    line = lambda token: len(re.findall(r"(?m)^" + re.escape(token) + r"\r?$", text))
    document_matches = re.findall(r"(?m)^--- !u!(\d+) &(\d+)(?: stripped)?\r?$", text)
    document_ids = [value for _, value in document_matches]
    local_references = []
    for source_line in text.splitlines():
        if "guid:" in source_line:
            continue
        for match in re.finditer(r"\{fileID: (\d+)\}", source_line):
            value = match.group(1)
            if value != "0":
                local_references.append({"file_id": value, "line": source_line})
    dangling = [row for row in local_references if row["file_id"] not in set(document_ids)]
    result = {
        "schema_version": SCHEMA + ".yaml_document_and_pointer_validation",
        "reserved_prefabinstance_document_id_count": line(RESERVED_DOCUMENT),
        "new_prefabinstance_document_id_count": line(NEW_DOCUMENT),
        "old_local_prefabinstance_pointer_count": line(OLD_LOCAL_POINTER),
        "new_local_prefabinstance_pointer_count": line(NEW_LOCAL_POINTER),
        "external_robot_fbx_source_reference_count": line(EXTERNAL_SOURCE),
        "external_robot_fbx_source_reference_before_count":
            before_text.count(EXTERNAL_SOURCE),
        "yaml_document_count": len(document_ids),
        "unique_yaml_document_id_count": len(set(document_ids)),
        "yaml_document_ids": document_ids,
        "duplicate_yaml_document_ids": sorted(
            value for value, count in Counter(document_ids).items() if count > 1),
        "local_pointer_count": len(local_references),
        "dangling_local_pointer_count": len(dangling),
        "dangling_local_pointers": dangling,
        "stripped_transform_document_id_count": line("--- !u!4 &100100001 stripped"),
        "root_child_pointer_count": line("  - {fileID: 100100001}"),
        "robot_fbx_guid_count": text.count("guid: " + ROBOT_FBX_GUID),
        "animator_controller_guid_count": text.count("guid: " + CONTROLLER_GUID),
        "controller_property_count": text.count("propertyPath: m_Controller"),
        "apply_root_motion_property_count": text.count("propertyPath: m_ApplyRootMotion"),
        "apply_root_motion_false_value_preserved":
            "propertyPath: m_ApplyRootMotion\n      value: 0" in text,
    }
    require(result["reserved_prefabinstance_document_id_count"] == 0 and
            result["new_prefabinstance_document_id_count"] == 1 and
            result["old_local_prefabinstance_pointer_count"] == 0 and
            result["new_local_prefabinstance_pointer_count"] == 1 and
            result["external_robot_fbx_source_reference_count"] == 1 and
            result["external_robot_fbx_source_reference_before_count"] == 1,
            "reserved FileID repair count mismatch")
    require(result["yaml_document_count"] == result["unique_yaml_document_id_count"] and
            result["dangling_local_pointer_count"] == 0,
            "YAML document or local pointer integrity mismatch")
    require(result["stripped_transform_document_id_count"] == 1 and
            result["root_child_pointer_count"] == 1 and
            result["animator_controller_guid_count"] == 1 and
            result["controller_property_count"] == 1 and
            result["apply_root_motion_property_count"] == 1 and
            result["apply_root_motion_false_value_preserved"],
            "wrapper non-colliding contract changed")
    presentation = (root / PRESENTATION).read_text(encoding="utf-8")
    result["presentation_wrapper_guid_reference_count"] = presentation.count(
        "guid: " + WRAPPER_GUID)
    require(result["presentation_wrapper_guid_reference_count"] == 1,
            "presentation wrapper GUID reference mismatch")
    controller = (root / "Assets/GameAssets/petrobot_model_animation/Terrain.controller").read_text(
        encoding="utf-8")
    animation = (root / (
        "Assets/GameAssets/petrobot_model_animation/"
        "Meshy_Cosmic Companion_Mesh_0_Anim_Blink.anim"
    )).read_text(encoding="utf-8")
    result["controller_animation_guid_reference_count"] = controller.count(
        "guid: " + ANIMATION_GUID)
    result["animation_local_y_curve_count"] = animation.count("attribute: m_LocalPosition.y")
    require(result["controller_animation_guid_reference_count"] >= 1 and
            result["animation_local_y_curve_count"] >= 1,
            "Animator controller/animation contract mismatch")
    result["status"] = "PASS"
    return result


def validate_test_inventory(root: Path, before: bytes, after: bytes) -> dict[str, Any]:
    before_text = before.decode("utf-8")
    after_text = after.decode("utf-8")
    method = "FbxAnimatorControllerClipAndVisualChildHierarchy_AreIntact"
    require(before_text.count("public void " + method + "(") == 1 and
            after_text.count("public void " + method + "(") == 1,
            "target testcase identity mismatch")
    required_assertions = [
        'Assert.AreEqual(0, Count(wrapper, "--- !u!1001 &100100000"))',
        'Assert.AreEqual(1, Count(wrapper, "--- !u!1001 &100200000"))',
        'Assert.AreEqual(1, Count(wrapper, "  m_PrefabInstance: {fileID: 100200000}"))',
        '"m_SourcePrefab: {fileID: 100100000, guid: ' + ROBOT_FBX_GUID + ', type: 3}"',
    ]
    require(all(after_text.count(value) == 1 for value in required_assertions),
            "target testcase repair assertions mismatch")

    inventory = load_json(root / V1_RESULT / "test_inventory.json")
    rows = list(inventory["approved_rows"]) + list(inventory["new_rows"])
    require(len(rows) == 152, "baseline static test inventory mismatch")
    by_source: dict[Path, set[str]] = defaultdict(set)
    fixture_counts = Counter()
    for row in rows:
        source_value = Path(row["source_path"])
        source = source_value if source_value.is_absolute() else root / source_value
        by_source[source.resolve()].add(row["method"])
        fixture_counts[row["fixture"]] += 1
    current_names = []
    missing = []
    unexpected = []
    for source, expected_methods in sorted(by_source.items(), key=lambda item: str(item[0])):
        require(source.is_file(), "test source missing: " + str(source))
        text = source.read_text(encoding="utf-8")
        current = set(re.findall(r"\[Test\]\s*public void\s+(\w+)\s*\(", text))
        for value in sorted(expected_methods - current):
            missing.append(str(source) + "::" + value)
        for value in sorted(current - expected_methods):
            unexpected.append(str(source) + "::" + value)
        current_names.extend(source.stem + "." + value for value in current)
    require(not missing and not unexpected, "testcase name set changed")
    require(dict(fixture_counts) == EXPECTED_FIXTURES,
            "fixture test count mismatch")
    require(len(current_names) == len(set(current_names)) == 152,
            "current testcase uniqueness mismatch")
    target_methods = set(re.findall(r"\[Test\]\s*public void\s+(\w+)\s*\(", after_text))
    require(len(target_methods) == 14, "Step 10D fixture count changed")
    return {
        "schema_version": SCHEMA + ".test_inventory",
        "baseline_inventory": record(
            V1_RESULT + "/test_inventory.json",
            (root / V1_RESULT / "test_inventory.json").read_bytes(),
        ),
        "target_testcase": method,
        "target_testcase_name_preserved": True,
        "step10d_fixture_test_count": len(target_methods),
        "step10d_fixture_test_names": sorted(target_methods),
        "fixture_counts": dict(sorted(fixture_counts.items())),
        "total_static_test_count": len(current_names),
        "unique_static_test_count": len(set(current_names)),
        "missing_testcases": missing,
        "unexpected_testcases": unexpected,
        "unity_execution_count": 0,
        "unity_test_status": "UNVERIFIED",
        "status": "PASS",
    }


def compare_records(root: Path, relative_json: str, expected_count: int,
                    label: str) -> dict[str, Any]:
    source = root / V1_RESULT / relative_json
    baseline = load_json(source)
    records = baseline["records"]
    require(len(records) == expected_count, label + " baseline count mismatch")
    base_dir = root
    if label == "candidate_repository":
        base_dir = Path(baseline["path"])
    rows = []
    for row in records:
        relative = row["relative_path"]
        path = base_dir / relative
        require(path.is_file(), label + " file missing: " + relative)
        data = path.read_bytes()
        expected_sha = row.get("sha256", row.get("after_sha256"))
        expected_size = row.get("size_bytes", row.get("after_size_bytes"))
        match = len(data) == int(expected_size) and sha(data) == expected_sha
        rows.append({
            "relative_path": relative,
            "expected_size_bytes": int(expected_size),
            "expected_sha256": expected_sha,
            "current_size_bytes": len(data),
            "current_sha256": sha(data),
            "byte_identical": match,
            "status": "PASS" if match else "FAIL",
        })
    require(all(row["status"] == "PASS" for row in rows), label + " preservation mismatch")
    result = {
        "schema_version": SCHEMA + ".preservation." + label,
        "source_baseline": record(relative_json, source.read_bytes()),
        "target_count": expected_count,
        "match_count": expected_count,
        "mismatch_count": 0,
        "records": rows,
        "status": "PASS",
    }
    if label == "candidate_repository":
        result.update({"candidate_count": 8, "file_count": 25})
    return result


def compare_checkpoint(root: Path) -> dict[str, Any]:
    source = root / CHECKPOINT
    baseline = load_json(source)
    records = baseline["gameassets_dependencies"]
    require(len(records) == 242, "GameAssets checkpoint count mismatch")
    rows = []
    for row in records:
        path = root / row["relative_path"]
        require(path.is_file(), "GameAssets checkpoint file missing")
        data = path.read_bytes()
        match = len(data) == int(row["size_bytes"]) and sha(data) == row["sha256"]
        rows.append({
            "relative_path": row["relative_path"],
            "checkpoint_size_bytes": int(row["size_bytes"]),
            "checkpoint_sha256": row["sha256"],
            "current_size_bytes": len(data),
            "current_sha256": sha(data),
            "unchanged_since_post_step10d_checkpoint": match,
            "status": "PASS" if match else "FAIL",
        })
    require(all(row["status"] == "PASS" for row in rows),
            "GameAssets current checkpoint mismatch")
    return {
        "schema_version": SCHEMA + ".gameassets_current_checkpoint",
        "checkpoint": record(CHECKPOINT, source.read_bytes()),
        "checkpoint_semantics": "post-Step10D current checkpoint only",
        "target_count": 242,
        "match_count": 242,
        "mismatch_count": 0,
        "retroactive_preimplementation_preservation": "UNVERIFIED_NONBLOCKING",
        "records": rows,
        "status": "PASS_POST_STEP10D_CHECKPOINT_UNCHANGED",
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


def report(wrapper: dict[str, Any], tests: dict[str, Any], scope: dict[str, Any],
           checkpoint: dict[str, Any], after_records: dict[str, dict[str, Any]],
           diff_counts: dict[str, dict[str, int]]) -> bytes:
    text = f"""# WordsInSilence Step 10D v1.2 validation report

## Outcome

The reserved Unity Prefab asset FileID collision was repaired by changing only the wrapper-local `PrefabInstance` document ID and its stripped local pointer from `100100000` to `100200000`. The external Robot FBX `m_SourcePrefab` reference remains `fileID: 100100000`.

## Changed project files

- `{WRAPPER}`: {after_records[WRAPPER]['size_bytes']} bytes, SHA-256 `{after_records[WRAPPER]['sha256']}`, diff `+{diff_counts[WRAPPER]['additions']}/-{diff_counts[WRAPPER]['deletions']}`
- `{TEST_SOURCE}`: {after_records[TEST_SOURCE]['size_bytes']} bytes, SHA-256 `{after_records[TEST_SOURCE]['sha256']}`, diff `+{diff_counts[TEST_SOURCE]['additions']}/-{diff_counts[TEST_SOURCE]['deletions']}`

Exactly `{scope['changed_project_file_count']}` of the 22 fixed Step 10D v1.0 project files changed; all other 20 are byte-identical.

## Static validation

- Reserved wrapper-local document ID count: `{wrapper['reserved_prefabinstance_document_id_count']}`
- New wrapper-local document ID count: `{wrapper['new_prefabinstance_document_id_count']}`
- New stripped local pointer count: `{wrapper['new_local_prefabinstance_pointer_count']}`
- External Robot FBX source reference count: `{wrapper['external_robot_fbx_source_reference_count']}`
- YAML document IDs: `{wrapper['unique_yaml_document_id_count']}/{wrapper['yaml_document_count']}` unique
- Dangling local pointers: `{wrapper['dangling_local_pointer_count']}`
- Step 10D fixture: `{tests['step10d_fixture_test_count']}/14` testcase names preserved
- Full static inventory: `{tests['total_static_test_count']}/152`
- GameAssets post-Step10D checkpoint: `{checkpoint['match_count']}/242` unchanged
- Whole-GameAssets retroactive preservation: `UNVERIFIED_NONBLOCKING`
- Unity executions: `0`

No Unity GUI, Test Runner, EditMode, PlayMode, batchmode, webcam, or live capture was run. No production C# source, Scene, `.meta`, GameAssets, recognition asset, candidate repository, Build Settings, fixed v1.0 Delivery, GUI evidence, or project context was modified.

This is Codex self-validation pending ChatGPT Work external review.
"""
    return text.encode("utf-8")


def package(root: Path) -> None:
    validation = root / "Validation"
    final_result = validation / NAME
    final_zip = validation / (NAME + "_delivery.zip")
    final_detached = Path(str(final_zip) + ".sha256")
    require(not any(path.exists() for path in (final_result, final_zip, final_detached)),
            "never-overwrite output target exists")

    gui_info = verify_fixed_zip(root, GUI_ZIP, GUI_ZIP_SIZE, GUI_ZIP_SHA,
                                GUI_DETACHED_SIZE, GUI_DETACHED_SHA)
    v1_info = verify_fixed_zip(root, V1_ZIP, V1_ZIP_SIZE, V1_ZIP_SHA,
                               V1_DETACHED_SIZE, V1_DETACHED_SHA)
    before_files, scope = v1_project_files(root)
    before_wrapper = before_files[WRAPPER]
    before_test = before_files[TEST_SOURCE]
    expected_wrapper, expected_test = expected_repairs(before_wrapper, before_test)
    after_wrapper = (root / WRAPPER).read_bytes()
    after_test = (root / TEST_SOURCE).read_bytes()
    require(after_wrapper == expected_wrapper, "wrapper contains changes beyond authorized repair")
    require(after_test == expected_test, "test source contains changes beyond authorized repair")

    after_records = {
        WRAPPER: record(WRAPPER, after_wrapper),
        TEST_SOURCE: record(TEST_SOURCE, after_test),
    }
    wrapper_diff, wrapper_add, wrapper_del = diff_blob(WRAPPER, before_wrapper, after_wrapper)
    test_diff, test_add, test_del = diff_blob(TEST_SOURCE, before_test, after_test)
    require((wrapper_add, wrapper_del) == (2, 2), "wrapper diff line count mismatch")
    require((test_add, test_del) == (5, 0), "test diff line count mismatch")
    diff_counts = {
        WRAPPER: {"additions": wrapper_add, "deletions": wrapper_del},
        TEST_SOURCE: {"additions": test_add, "deletions": test_del},
        "total": {"additions": wrapper_add + test_add,
                  "deletions": wrapper_del + test_del},
    }
    wrapper_validation = validate_wrapper(root, before_wrapper, after_wrapper)
    tests = validate_test_inventory(root, before_test, after_test)

    require(sha((root / WRAPPER_META).read_bytes()) == WRAPPER_META_SHA,
            "wrapper meta changed")
    require(sha((root / TEST_META).read_bytes()) == TEST_META_SHA,
            "test meta changed")
    require(sha((root / BUILD_SETTINGS).read_bytes()) == BUILD_SETTINGS_SHA,
            "Build Settings changed")

    step10a = compare_records(root, "step10a_project_files_before_after.json", 30,
                              "step10a_project")
    protected = compare_records(root, "protected_files_before_after.json", 106,
                                "protected")
    step10b = compare_records(root, "step10b_files_before_after.json", 22, "step10b")
    profile = compare_records(root, "active_candidate_profile_before_after.json", 22,
                              "candidate_profile")
    candidates = compare_records(root, "candidate_repository_before_after.json", 25,
                                 "candidate_repository")
    checkpoint = compare_checkpoint(root)

    staged = staged_files(root)
    require(staged == [], "Git staged files changed")
    git = {
        "schema_version": SCHEMA + ".git_staged",
        "before_count": 0,
        "after_count": 0,
        "staged_paths": [],
        "status": "PASS",
    }
    tx_helper = load_module(root, TX_HELPER, "step10d_v12_transaction_helper")
    transaction = tx_helper.transaction_regression()
    require(transaction["status"] == "PASS" and
            len(transaction["failure_injection_rows"]) == 4 and
            all(row["status"] == "PASS" for row in transaction["failure_injection_rows"]) and
            transaction["never_overwrite_second_commit_blocked"],
            "packaging transaction regression failed")

    external_context = {
        "schema_version": SCHEMA + ".external_review_context",
        "step10d_v1_0_source_and_package_external_review": "FAIL_EVIDENCE_GAP_ONLY",
        "step10d_v1_0_core_implementation": "PASS_EXCEPT_RESERVED_FILEID_IMPORT_DEFECT",
        "gui_compile": "PASS_ZERO_ERRORS",
        "gui_editmode_aggregate": "PASS_152_OF_152",
        "gui_exactly_once": "FAIL_TWO_COMPLETION_CYCLES",
        "gui_scene_open": "FAIL_PREFABINSTANCE_TO_PREFAB_FILEID_100100000",
        "v1_2_unity_execution_count": 0,
        "status": "PASS_CONTEXT_RECORDED",
    }
    provenance = {
        "schema_version": SCHEMA + ".input_provenance",
        "implementation_owner": "Codex",
        "fixed_step10d_v1_0_delivery": v1_info,
        "fixed_step10d_gui_gate_evidence": gui_info,
        "status": "PASS",
    }
    repair_contract = {
        "schema_version": SCHEMA + ".repair_contract",
        "cause": "wrapper-local PrefabInstance reused Unity reserved Prefab asset FileID 100100000",
        "authorized_changes": [WRAPPER, TEST_SOURCE],
        "old_local_document_id": 100100000,
        "new_local_document_id": 100200000,
        "external_robot_fbx_source_file_id": 100100000,
        "external_robot_fbx_guid": ROBOT_FBX_GUID,
        "external_reference_changed": False,
        "production_csharp_changed": False,
        "scene_changed": False,
        "meta_changed": False,
        "gameassets_changed": False,
        "status": "PASS",
    }
    change_summary = {
        "schema_version": SCHEMA + ".change_summary",
        "changed_project_file_count": 2,
        "changed_project_files": [WRAPPER, TEST_SOURCE],
        "diff_counts": diff_counts,
        "before": {
            WRAPPER: record(WRAPPER, before_wrapper),
            TEST_SOURCE: record(TEST_SOURCE, before_test),
        },
        "after": after_records,
        "status": "PASS",
    }
    primary = {
        "schema_version": SCHEMA + ".primary_verdict",
        "primary_verdict": "RESERVED_PREFAB_FILEID_COLLISION_REPAIR_SOURCE_EVIDENCE_READY",
        "authorized_project_change_count": 2,
        "static_validation": "PASS",
        "unity_execution_count": 0,
        "unity_gui_targeted_14_and_presentation_gate": "UNVERIFIED",
        "gameassets_retroactive_preservation": "UNVERIFIED_NONBLOCKING",
        "status_strings": [
            "STEP10D_V1_2_RESERVED_PREFAB_FILEID_COLLISION_REPAIR_SOURCE_EVIDENCE_READY",
            "STEP10D_V1_0_APPROVED_NONCOLLIDING_SCOPE_PRESERVED",
            "STEP10D_UNITY_GUI_TARGETED_14_AND_PRESENTATION_GATE_UNVERIFIED",
            "BLOCKED_PENDING_EXTERNAL_REVIEW",
        ],
        "status": "PASS",
    }
    assertion_rows = [
        ("fixed_step10d_v1_0_delivery", "PASS", V1_ZIP_SHA),
        ("fixed_step10d_gui_gate_evidence", "PASS", GUI_ZIP_SHA),
        ("authorized_project_change_count", "PASS", "2"),
        ("authorized_project_change_paths", "PASS", WRAPPER + " | " + TEST_SOURCE),
        ("wrapper_before_identity", "PASS", WRAPPER_BEFORE_SHA),
        ("test_before_identity", "PASS", TEST_BEFORE_SHA),
        ("reserved_local_document_id", "PASS", "0"),
        ("new_local_document_id", "PASS", "1"),
        ("new_local_pointer", "PASS", "1"),
        ("external_robot_fbx_source_reference", "PASS", "1 unchanged"),
        ("yaml_document_ids_unique", "PASS", "4/4"),
        ("dangling_local_pointers", "PASS", "0"),
        ("presentation_wrapper_guid_reference", "PASS", "1"),
        ("animator_controller_and_animation", "PASS", "controller/clip/local-Y preserved"),
        ("apply_root_motion_false", "PASS", "preserved"),
        ("wrapper_meta", "PASS", WRAPPER_META_SHA),
        ("test_meta", "PASS", TEST_META_SHA),
        ("step10d_fixture_inventory", "PASS", "14/14"),
        ("full_static_test_inventory", "PASS", "152/152"),
        ("step10a_project", "PASS", "30/30"),
        ("protected_files", "PASS", "106/106"),
        ("step10b_files", "PASS", "22/22"),
        ("candidate_profile", "PASS", "22/22; prototypes 9; thresholds 3"),
        ("candidate_repository", "PASS", "8 candidates / 25 files"),
        ("build_settings", "PASS", BUILD_SETTINGS_SHA),
        ("gameassets_post_step10d_checkpoint", "PASS", "242/242 current match"),
        ("gameassets_retroactive_preservation", "UNVERIFIED",
         "UNVERIFIED_NONBLOCKING; current checkpoint is not preimplementation baseline"),
        ("unity_execution_count", "NOT_APPLICABLE", "0 by source-stage prohibition"),
        ("git_staged", "PASS", "0"),
        ("packaging_transaction", "PASS", "4/4 rollback and never-overwrite"),
    ]
    assertions = assertions_bytes(assertion_rows)
    assertion_counts = dict(Counter(row[1] for row in assertion_rows))

    generated: dict[str, bytes] = {
        "input_provenance.json": jbytes(provenance),
        "external_review_context.json": jbytes(external_context),
        "reserved_prefab_fileid_collision_repair_contract.json": jbytes(repair_contract),
        "change_summary.json": jbytes(change_summary),
        "yaml_document_id_and_local_pointer_validation.json": jbytes(wrapper_validation),
        "test_inventory.json": jbytes(tests),
        "project_change_scope.json": jbytes(scope),
        "step10a_project_files_preservation.json": jbytes(step10a),
        "protected_files_preservation.json": jbytes(protected),
        "step10b_files_preservation.json": jbytes(step10b),
        "candidate_profile_preservation.json": jbytes(profile),
        "candidate_repository_preservation.json": jbytes(candidates),
        "gameassets_post_step10d_checkpoint_comparison.json": jbytes(checkpoint),
        "git_staged_before_after.json": jbytes(git),
        "packaging_transaction_failure_injection_regression.json": jbytes(transaction),
        "primary_verdict.json": jbytes(primary),
        "assertion_summary.json": jbytes({
            "schema_version": SCHEMA + ".assertion_summary",
            "total": len(assertion_rows),
            "counts": assertion_counts,
            "status": "PASS_WITH_DECLARED_UNVERIFIED_AND_NOT_APPLICABLE",
        }),
        "ASSERTIONS.jsonl": assertions,
        "VALIDATION_REPORT.md": report(
            wrapper_validation, tests, scope, checkpoint, after_records, diff_counts),
        "before/" + WRAPPER: before_wrapper,
        "after/" + WRAPPER: after_wrapper,
        "before/" + TEST_SOURCE: before_test,
        "after/" + TEST_SOURCE: after_test,
        "diff/reserved_prefab_fileid_collision_repair.patch": wrapper_diff + test_diff,
        "analysis/" + Path(__file__).name: Path(__file__).read_bytes(),
    }
    complete = add_manifest(generated)

    def validate_commit(result: Path, delivery: Path, detached: Path) -> None:
        disk = {
            path.relative_to(result).as_posix(): path.read_bytes()
            for path in result.rglob("*") if path.is_file()
        }
        require(disk == complete, "committed result file map mismatch")
        blob = delivery.read_bytes()
        require(blob == tx_helper.deterministic_zip(complete),
                "deterministic ZIP recreation mismatch")
        require(detached.read_text(encoding="ascii").strip() ==
                sha(blob) + "  " + delivery.name,
                "detached checksum content mismatch")
        with zipfile.ZipFile(delivery, "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            require(len(names) == len(set(names)), "delivery duplicate entry")
            require(not any(unsafe_zip_path(name) for name in names),
                    "delivery unsafe path")
            require(not any(info.flag_bits & 1 for info in infos),
                    "delivery encrypted entry")
            require(not any(((info.external_attr >> 16) & 0o170000) == 0o120000
                            for info in infos), "delivery symlink entry")
            require(archive.testzip() is None, "delivery CRC failure")
            archive_files = {name: archive.read(name) for name in names}
            require(archive_files == complete, "delivery payload mismatch")
            manifest = json.loads(archive_files["output_file_manifest.json"])
            require({row["relative_path"] for row in manifest["files"]} ==
                    set(names) - {"output_file_manifest.json"},
                    "delivery manifest membership mismatch")
            for row in manifest["files"]:
                payload = archive_files[row["relative_path"]]
                require(len(payload) == row["size_bytes"] and
                        sha(payload) == row["sha256"],
                        "delivery manifest payload mismatch")
        require(staged_files(root) == [], "Git staged files changed during commit")

    result, delivery, detached = tx_helper.commit_outputs(
        validation,
        complete,
        names=(NAME, NAME + "_delivery.zip", NAME + "_delivery.zip.sha256"),
        validator=validate_commit,
    )
    print(json.dumps({
        "status": "PASS",
        "result_directory": str(result),
        "result_file_count": len(complete),
        "manifest_target_count": len(complete) - 1,
        "changed_project_files": [WRAPPER, TEST_SOURCE],
        "before_after": {
            WRAPPER: {
                "before": record(WRAPPER, before_wrapper),
                "after": after_records[WRAPPER],
                "diff": diff_counts[WRAPPER],
            },
            TEST_SOURCE: {
                "before": record(TEST_SOURCE, before_test),
                "after": after_records[TEST_SOURCE],
                "diff": diff_counts[TEST_SOURCE],
            },
        },
        "reserved_document_id_count":
            wrapper_validation["reserved_prefabinstance_document_id_count"],
        "new_document_id_count":
            wrapper_validation["new_prefabinstance_document_id_count"],
        "new_local_pointer_count":
            wrapper_validation["new_local_prefabinstance_pointer_count"],
        "external_robot_fbx_source_reference_count":
            wrapper_validation["external_robot_fbx_source_reference_count"],
        "static_test_inventory": tests["total_static_test_count"],
        "unity_execution_count": 0,
        "git_staged": 0,
        "delivery_zip": str(delivery),
        "delivery_size_bytes": delivery.stat().st_size,
        "delivery_sha256": sha(delivery.read_bytes()),
        "detached_checksum": str(detached),
        "detached_size_bytes": detached.stat().st_size,
        "detached_file_sha256": sha(detached.read_bytes()),
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
