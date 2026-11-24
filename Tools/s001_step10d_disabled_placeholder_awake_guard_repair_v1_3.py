#!/usr/bin/env python3
"""Step 10D v1.3 disabled placeholder Awake guard repair validator/package builder."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
from typing import Callable
import zipfile


SCHEMA = "WordsInSilence.Step10D.DisabledPlaceholderAwakeGuardRepair.v1.3"
NAME = "WordsInSilence_step10d_disabled_placeholder_awake_guard_repair_v1_3"

PLACEHOLDER = "Assets/_Project/Scripts/Practice/PetRobotWorldIntegration/PetRobotWorldPlaceholderVisuals.cs"
TARGET_TEST = "Assets/_Project/Tests/EditMode/PetRobotFloatingCompanionPresentationV1Tests.cs"
VISUAL_FACTORY = "Assets/_Project/Scripts/Practice/PetRobotPortfolioQuest/PetRobotPortfolioQuestVisuals.cs"
SCENE = "Assets/_Project/Scenes/VerticalSlice/PetRobotPortfolioQuestVerticalSlice.unity"
WRAPPER = "Assets/_Project/Resources/VerticalSlice/PetRobotAnimatedVisualWrapper.prefab"
PRESENTATION = "Assets/_Project/Resources/VerticalSlice/PetRobotPortfolioQuestPresentation.prefab"
WORLD_PREFAB = "Assets/_Project/Resources/VerticalSlice/PetRobotWorldEntryReturn.prefab"
COMPANION = "Assets/_Project/Scripts/Practice/PetRobotPortfolioQuest/PetRobotFloatingCompanionController.cs"

BEFORE_IDENTITIES = {
    PLACEHOLDER: (1998, "2a80f1f56ee6eb5486bef1bb71e57d6b93eafa3397ccb9335e8806dbfc73c6da"),
    TARGET_TEST: (11530, "91b5a30a984165810e39fe9fdc3273ed365dd805c358a13f9ed47c19287799f1"),
}
PROTECTED_IDENTITIES = {
    VISUAL_FACTORY: (1906, "76d8df6b1a52354495d00cff26194c80e98e178a00cab706fd61fb1bb7465543"),
    SCENE: (8402, "a4696cadc0e688e24807f8b00d5508bec029a4da81a08883e4c3bfb7e3cea7fe"),
    WRAPPER: (2706, "fa906ebbeebd5a4b309d5b741d4126d2000c1818a75b68d1c5b54ff22b2197a8"),
    PRESENTATION: (4315, "4fd26a3ed6b1b3aacdfc1262eb73e1b136b1bcbd63849c59dcccd9e58dfde602"),
    WORLD_PREFAB: (9861, "d035ba31f6b23eaa9ee97d7403d70b5759c53687e6a499bbcf4e1de60577603b"),
}

V12_ZIP = "Validation/WordsInSilence_step10d_reserved_prefab_fileid_collision_repair_v1_2_delivery.zip"
V12_ZIP_SIZE = 61796
V12_ZIP_SHA = "7bc65eb9eda803941ff9f5f45b9b2e1b706627ed8ecfa2dde8a7a646fcf631c7"
V12_DETACHED_SIZE = 147
V12_DETACHED_SHA = "c28c5bfc864e474873e9176caad8623217986d6a688a235e59a18dcc0f78738f"

GUI_ZIP = "Validation/WordsInSilence_step10d_v1_2_targeted_14_and_companion_presentation_gui_gate_delivery.zip"
GUI_ZIP_SIZE = 2521168
GUI_ZIP_SHA = "426c2098c86d2f6351dea4fe1c5f38b738796cc4a1cf675ba5ef077d0b28e5ff"
GUI_DETACHED_SIZE = 155
GUI_DETACHED_SHA = "833ff072d5c69d5c264413276ff1c1fba4aa6666bad6933d31c16c13e8177a2d"

V12_RESULT = Path("Validation/WordsInSilence_step10d_reserved_prefab_fileid_collision_repair_v1_2")
V10_RESULT = Path("Validation/WordsInSilence_step10d_existing_fbx_animator_floating_pet_robot_companion_presentation_layer_v1_0")
GUI_RESULT = Path("Validation/WordsInSilence_step10d_v1_2_targeted_14_and_companion_presentation_gui_gate_evidence")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def record(relative: str, data: bytes) -> dict:
    return {"relative_path": relative, "size_bytes": len(data), "sha256": sha(data)}


def jbytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_archive(blob: bytes) -> dict:
    with zipfile.ZipFile(io.BytesIO(blob), "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        require(archive.testzip() is None, "fixed ZIP CRC failure")
        require(len(names) == len(set(names)), "fixed ZIP duplicate entry")
        require(not any(name.startswith(("/", "\\")) or ".." in PurePosixPath(name).parts or "\\" in name for name in names), "fixed ZIP unsafe path")
        require(not any(info.flag_bits & 1 for info in infos), "fixed ZIP encrypted entry")
        require(not any(((info.external_attr >> 16) & 0o170000) == 0o120000 for info in infos), "fixed ZIP symlink")
        files = {info.filename: archive.read(info) for info in infos}
    manifest = json.loads(files["output_file_manifest.json"])
    rows = manifest["files"]
    require(set(files) - {"output_file_manifest.json"} == {row["relative_path"] for row in rows}, "fixed ZIP manifest membership mismatch")
    for row in rows:
        data = files[row["relative_path"]]
        require(len(data) == row["size_bytes"] and sha(data) == row["sha256"], "fixed ZIP manifest payload mismatch")
    return {"entry_count": len(files), "manifest_target_count": len(rows), "files": files}


def fixed_zip(root: Path, relative: str, size: int, digest: str,
              detached_size: int, detached_digest: str) -> tuple[dict, dict[str, bytes]]:
    path = root / relative
    detached = Path(str(path) + ".sha256")
    blob = path.read_bytes()
    check = detached.read_bytes()
    require(len(blob) == size and sha(blob) == digest, "fixed input ZIP identity mismatch: " + relative)
    require(len(check) == detached_size and sha(check) == detached_digest, "fixed detached identity mismatch: " + relative)
    words = check.decode("ascii").strip().split()
    require(words == [digest, path.name], "fixed detached syntax mismatch: " + relative)
    archive = safe_archive(blob)
    return ({
        "zip": record(relative, blob),
        "detached": record(relative + ".sha256", check),
        "entry_count": archive["entry_count"],
        "manifest_target_count": archive["manifest_target_count"],
        "crc_path_membership_manifest": "PASS",
        "status": "PASS",
    }, archive["files"])


def expected_after(before_placeholder: bytes, before_test: bytes) -> tuple[bytes, bytes]:
    guard_anchor = b"        void Awake()\n        {\n"
    guard_replacement = guard_anchor + b"            if (!enabled) return;\n"
    require(before_placeholder.count(guard_anchor) == 1, "placeholder Awake baseline anchor mismatch")
    placeholder = before_placeholder.replace(guard_anchor, guard_replacement, 1)

    test = before_test
    const_anchor = (
        b"        const string WorldEntrySource =\n"
        b"            \"Assets/_Project/Scripts/Practice/PetRobotWorldIntegration/PetRobotWorldEntryController.cs\";\n"
    )
    const_replacement = const_anchor + (
        b"        const string PlaceholderSource =\n"
        b"            \"Assets/_Project/Scripts/Practice/PetRobotWorldIntegration/PetRobotWorldPlaceholderVisuals.cs\";\n"
    )
    require(test.count(const_anchor) == 1, "test constant baseline anchor mismatch")
    test = test.replace(const_anchor, const_replacement, 1)

    scene_anchor = b"            string scene = ReadProject(PortfolioScene);\n"
    scene_replacement = (
        b"            string scene = ReadProject(PortfolioScene).Replace(\"\\r\\n\", \"\\n\");\n"
        b"            string placeholder = ReadProject(PlaceholderSource).Replace(\"\\r\\n\", \"\\n\");\n"
    )
    require(test.count(scene_anchor) == 1, "test scene baseline anchor mismatch")
    test = test.replace(scene_anchor, scene_replacement, 1)

    assertion_anchor = b"            Assert.AreEqual(2, Count(scene, \"objectReference: {fileID: 100100011}\"));\n"
    assertion_replacement = assertion_anchor + (
        b"            Assert.AreEqual(1, Count(placeholder,\n"
        b"                \"void Awake()\\n        {\\n            if (!enabled) return;\\n\"));\n"
        b"            Assert.AreEqual(1, Count(scene,\n"
        b"                \"- target: {fileID: 11400002, guid: 10b00000000000000000000000000010, type: 3}\\n\" +\n"
        b"                \"      propertyPath: m_Enabled\\n\" +\n"
        b"                \"      value: 0\\n\"));\n"
    )
    require(test.count(assertion_anchor) == 1, "test assertion baseline anchor mismatch")
    test = test.replace(assertion_anchor, assertion_replacement, 1)
    return placeholder, test


def diff_stats(before: bytes, after: bytes) -> tuple[int, int, bytes]:
    a = before.decode("utf-8").splitlines()
    b = after.decode("utf-8").splitlines()
    additions = deletions = 0
    matcher = difflib.SequenceMatcher(a=a, b=b)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "delete"):
            deletions += i2 - i1
        if tag in ("replace", "insert"):
            additions += j2 - j1
    patch = "\n".join(difflib.unified_diff(a, b, fromfile="before", tofile="after", lineterm="")) + "\n"
    return additions, deletions, patch.encode("utf-8")


def source_contract(root: Path) -> dict:
    placeholder = (root / PLACEHOLDER).read_text(encoding="utf-8").replace("\r\n", "\n")
    test = (root / TARGET_TEST).read_text(encoding="utf-8").replace("\r\n", "\n")
    scene = (root / SCENE).read_text(encoding="utf-8").replace("\r\n", "\n")
    world = (root / WORLD_PREFAB).read_text(encoding="utf-8").replace("\r\n", "\n")
    visual = (root / VISUAL_FACTORY).read_text(encoding="utf-8").replace("\r\n", "\n")
    wrapper = (root / WRAPPER).read_text(encoding="utf-8").replace("\r\n", "\n")
    guard = "void Awake()\n        {\n            if (!enabled) return;\n"
    disabled = (
        "- target: {fileID: 11400002, guid: 10b00000000000000000000000000010, type: 3}\n"
        "      propertyPath: m_Enabled\n"
        "      value: 0\n"
    )
    world_section = re.search(r"--- !u!114 &11400002\n(?P<body>.*?)(?=--- !u!|\Z)", world, re.DOTALL)
    require(world_section is not None, "Step 10B placeholder component section missing")
    body = world_section.group("body")
    generation_tokens = [
        'CreateVisual(PrimitiveType.Capsule, "PlayerPlaceholder"',
        'CreateVisual(PrimitiveType.Cube, "PetRobotBody"',
        'CreateVisual(PrimitiveType.Sphere, "PetRobotHead"',
        'ground.name = "PracticeWorldGround";',
        "ground.transform.localPosition = new Vector3(0f, -0.25f, 0f);",
        "ground.transform.localScale = new Vector3(24f, 0.5f, 18f);",
        "if (collider != null) Destroy(collider);",
    ]
    values = {
        "awake_guard_token_count": placeholder.count("if (!enabled) return;"),
        "awake_guard_first_statement_adjacency_count": placeholder.count(guard),
        "placeholder_generation_tokens_preserved": all(token in placeholder for token in generation_tokens),
        "step10b_placeholder_component_enabled_value_count": body.count("  m_Enabled: 1\n"),
        "step10b_placeholder_script_guid_count": body.count("guid: 10b00000000000000000000000000007"),
        "step10d_disabled_override_exact_adjacency_count": scene.count(disabled),
        "target_test_placeholder_source_constant_count": test.count("const string PlaceholderSource"),
        "target_test_awake_exact_assertion_count": test.count('"void Awake()\\n        {\\n            if (!enabled) return;\\n"'),
        "target_test_scene_disabled_exact_assertion_count": test.count('"- target: {fileID: 11400002, guid: 10b00000000000000000000000000010, type: 3}\\n" +'),
        "visual_child_count_guard_count": visual.count("anchor.childCount != 0"),
        "wrapper_reserved_document_id_count": wrapper.count("--- !u!1001 &100100000"),
        "wrapper_new_document_id_count": wrapper.count("--- !u!1001 &100200000"),
        "wrapper_new_local_pointer_count": wrapper.count("m_PrefabInstance: {fileID: 100200000}"),
        "wrapper_external_robot_source_count": wrapper.count("m_SourcePrefab: {fileID: 100100000, guid: 50573fbff0bc00c418eb7319a40093d3, type: 3}"),
    }
    require(values["awake_guard_token_count"] == 1, "Awake guard count mismatch")
    require(values["awake_guard_first_statement_adjacency_count"] == 1, "Awake guard is not first statement")
    require(values["placeholder_generation_tokens_preserved"], "placeholder generation contract changed")
    require(values["step10b_placeholder_component_enabled_value_count"] == 1 and values["step10b_placeholder_script_guid_count"] == 1, "Step 10B enabled contract mismatch")
    require(values["step10d_disabled_override_exact_adjacency_count"] == 1, "Step 10D disabled override mismatch")
    require(values["target_test_placeholder_source_constant_count"] == 1 and values["target_test_awake_exact_assertion_count"] == 1 and values["target_test_scene_disabled_exact_assertion_count"] == 1, "targeted regression source mismatch")
    require(values["visual_child_count_guard_count"] == 1, "approved visual childCount guard mismatch")
    require(values["wrapper_reserved_document_id_count"] == 0 and values["wrapper_new_document_id_count"] == 1 and values["wrapper_new_local_pointer_count"] == 1 and values["wrapper_external_robot_source_count"] == 1, "v1.2 wrapper repair mismatch")
    values.update({
        "step10b_semantics": "enabled component retains all legacy placeholder and ground creation",
        "step10d_semantics": "disabled component returns before any placeholder or ground creation",
        "status": "PASS",
    })
    return values


def test_inventory(root: Path) -> dict:
    baseline_path = root / V10_RESULT / "test_inventory.json"
    baseline = load_json(baseline_path)
    approved = list(baseline["approved_rows"]) + list(baseline["new_rows"])
    current = []
    paths = sorted({row["source_path"] for row in approved})
    for relative in paths:
        source_path = Path(relative)
        if not source_path.is_absolute():
            source_path = root / source_path
        text = source_path.read_text(encoding="utf-8")
        fixture_match = re.search(r"public sealed class\s+(\w+)", text)
        require(fixture_match is not None, "fixture class missing: " + relative)
        fixture = fixture_match.group(1)
        for method in re.findall(r"\[Test\]\s*public void\s+(\w+)\s*\(", text):
            current.append({"fixture": fixture, "method": method, "source_path": relative})
    key = lambda row: (row["fixture"], row["method"], row["source_path"])
    approved_keys = {key(row) for row in approved}
    current_keys = {key(row) for row in current}
    require(len(approved) == 152 and len(current) == 152 and len(current_keys) == 152, "static testcase inventory count mismatch")
    require(current_keys == approved_keys, "static testcase name set changed")
    target_names = sorted(row["method"] for row in current if row["fixture"] == "PetRobotFloatingCompanionPresentationV1Tests")
    require(len(target_names) == 14, "targeted fixture count mismatch")
    return {
        "baseline": record(str(V10_RESULT / "test_inventory.json").replace("\\", "/"), baseline_path.read_bytes()),
        "targeted_fixture_count": 14,
        "targeted_fixture_names": target_names,
        "total_static_test_count": 152,
        "unique_static_test_count": 152,
        "missing": [],
        "unexpected": [],
        "unity_execution_count": 0,
        "unity_status": "UNVERIFIED",
        "status": "PASS",
    }


def compare_records(root: Path, source: Path, expected_count: int, label: str,
                    base_dir: Path | None = None, allowed_changed: set[str] | None = None) -> dict:
    baseline = load_json(source)
    rows_in = baseline["records"]
    require(len(rows_in) == expected_count, label + " baseline count mismatch")
    base = root if base_dir is None else base_dir
    allowed = set() if allowed_changed is None else allowed_changed
    rows = []
    changed = []
    for prior in rows_in:
        relative = prior["relative_path"]
        path = base / relative
        require(path.is_file(), label + " file missing: " + relative)
        data = path.read_bytes()
        expected_sha = prior.get("expected_sha256", prior.get("sha256", prior.get("current_sha256")))
        expected_size = prior.get("expected_size_bytes", prior.get("size_bytes", prior.get("current_size_bytes")))
        same = len(data) == int(expected_size) and sha(data) == expected_sha
        authorized = relative in allowed
        status = "PASS" if same or authorized else "FAIL"
        row = {"relative_path": relative, "baseline_size_bytes": int(expected_size), "baseline_sha256": expected_sha,
               "current_size_bytes": len(data), "current_sha256": sha(data), "byte_identical": same,
               "authorized_v1_3_change": authorized, "status": status}
        if not same:
            changed.append(relative)
        rows.append(row)
    require(all(row["status"] == "PASS" for row in rows), label + " preservation mismatch")
    require(set(changed) <= allowed, label + " unauthorized change")
    return {"label": label, "target_count": expected_count, "match_count": sum(row["byte_identical"] for row in rows),
            "authorized_change_count": len(changed), "changed_paths": changed, "mismatch_count": 0,
            "records": rows, "status": "PASS"}


def gameassets_checkpoint(root: Path) -> dict:
    source = root / V10_RESULT / "asset_identity_and_guid_graph.json"
    baseline = load_json(source)
    records = baseline["gameassets_dependencies"]
    require(len(records) == 242, "GameAssets checkpoint count mismatch")
    rows = []
    for prior in records:
        path = root / prior["relative_path"]
        require(path.is_file(), "GameAssets checkpoint file missing: " + prior["relative_path"])
        data = path.read_bytes()
        same = len(data) == prior["size_bytes"] and sha(data) == prior["sha256"]
        require(same, "GameAssets post-Step10D checkpoint mismatch: " + prior["relative_path"])
        rows.append({"relative_path": prior["relative_path"], "checkpoint_size_bytes": prior["size_bytes"],
                     "checkpoint_sha256": prior["sha256"], "current_size_bytes": len(data),
                     "current_sha256": sha(data), "byte_identical": same, "status": "PASS"})
    return {"semantics": "post-Step10D checkpoint change detection only; not retroactive preimplementation preservation",
            "target_count": 242, "match_count": 242, "mismatch_count": 0,
            "retroactive_preimplementation_preservation": "UNVERIFIED_NONBLOCKING", "records": rows,
            "status": "PASS_POST_STEP10D_CHECKPOINT_UNCHANGED"}


def git_staged(root: Path) -> dict:
    run = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=root,
                         capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    require(run.returncode == 0, "Git staged query failed")
    paths = [line for line in run.stdout.splitlines() if line.strip()]
    require(not paths, "Git staged count nonzero")
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


def validate_zip(blob: bytes, files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(io.BytesIO(blob), "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        require(archive.testzip() is None and len(names) == len(set(names)), "output ZIP CRC/duplicate failure")
        require(not any(name.startswith(("/", "\\")) or ".." in PurePosixPath(name).parts or "\\" in name for name in names), "output ZIP unsafe path")
        require(not any(info.flag_bits & 1 for info in infos), "output ZIP encrypted entry")
        require(not any(((info.external_attr >> 16) & 0o170000) == 0o120000 for info in infos), "output ZIP symlink")
        require(set(names) == set(files), "output ZIP membership mismatch")
        for name, expected in files.items():
            require(archive.read(name) == expected, "output ZIP payload mismatch: " + name)


def commit_outputs(parent: Path, files: dict[str, bytes], names: tuple[str, str, str],
                   inject: str | None = None,
                   validator: Callable[[Path, Path, Path], None] | None = None) -> tuple[Path, Path, Path]:
    final_result, final_zip, final_detached = (parent / name for name in names)
    require(not any(path.exists() for path in (final_result, final_zip, final_detached)), "never-overwrite target exists")
    token = os.urandom(8).hex()
    temp_result = parent / (".step10d-v1_3-result-" + token)
    temp_zip = parent / (".step10d-v1_3-delivery-" + token + ".zip")
    temp_detached = parent / (".step10d-v1_3-checksum-" + token)
    created: list[Path] = []
    committed = False
    try:
        temp_result.mkdir(parents=True)
        for name, data in files.items():
            path = temp_result / PurePosixPath(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        zip_blob = deterministic_zip(files)
        temp_zip.write_bytes(zip_blob)
        temp_detached.write_text(f"{sha(zip_blob)}  {final_zip.name}\n", encoding="ascii", newline="\n")
        if inject == "BEFORE_FINAL_RENAME": raise RuntimeError("injected " + inject)
        temp_result.replace(final_result); created.append(final_result)
        if inject == "AFTER_RESULT_RENAME": raise RuntimeError("injected " + inject)
        temp_zip.replace(final_zip); created.append(final_zip)
        if inject == "AFTER_DELIVERY_RENAME": raise RuntimeError("injected " + inject)
        temp_detached.replace(final_detached); created.append(final_detached)
        if inject == "AFTER_DETACHED_RENAME_BEFORE_COMMIT": raise RuntimeError("injected " + inject)
        if validator:
            validator(final_result, final_zip, final_detached)
        committed = True
        return final_result, final_zip, final_detached
    finally:
        if not committed:
            for path in reversed(created):
                if path.is_dir(): shutil.rmtree(path)
                elif path.exists(): path.unlink()
            if temp_result.exists(): shutil.rmtree(temp_result)
            for path in (temp_zip, temp_detached):
                if path.exists(): path.unlink()


def transaction_regression() -> dict:
    points = ["BEFORE_FINAL_RENAME", "AFTER_RESULT_RENAME", "AFTER_DELIVERY_RENAME", "AFTER_DETACHED_RENAME_BEFORE_COMMIT"]
    rows = []
    for point in points:
        with tempfile.TemporaryDirectory(prefix="WIS-Step10D-v1_3-txn-") as directory:
            parent = Path(directory)
            sentinel = parent / "sentinel.bin"
            sentinel.write_bytes(b"preserve")
            observed = False
            try:
                commit_outputs(parent, {"x.bin": b"x"}, ("result", "delivery.zip", "delivery.zip.sha256"), inject=point)
            except RuntimeError as error:
                observed = str(error) == "injected " + point
            finals = sum((parent / name).exists() for name in ("result", "delivery.zip", "delivery.zip.sha256"))
            temps = len(list(parent.glob(".step10d-v1_3-*")))
            sentinel_same = sentinel.read_bytes() == b"preserve"
            status = "PASS" if observed and finals == 0 and temps == 0 and sentinel_same else "FAIL"
            rows.append({"injection_point": point, "expected_exception_observed": observed,
                         "remaining_final_outputs": finals, "remaining_temporary_outputs": temps,
                         "preexisting_sentinel_changed": not sentinel_same, "status": status})
    require(all(row["status"] == "PASS" for row in rows), "transaction failure injection failed")
    return {"same_production_commit_function": True, "failure_injection_rows": rows, "status": "PASS"}


def package(root: Path) -> None:
    validation = root / "Validation"
    names = (NAME, NAME + "_delivery.zip", NAME + "_delivery.zip.sha256")
    require(not any((validation / name).exists() for name in names), "never-overwrite output target exists")

    v12_info, _ = fixed_zip(root, V12_ZIP, V12_ZIP_SIZE, V12_ZIP_SHA, V12_DETACHED_SIZE, V12_DETACHED_SHA)
    gui_info, gui_files = fixed_zip(root, GUI_ZIP, GUI_ZIP_SIZE, GUI_ZIP_SHA, GUI_DETACHED_SIZE, GUI_DETACHED_SHA)
    before_placeholder = gui_files["read_only_project_evidence/" + PLACEHOLDER]
    before_test = gui_files["read_only_project_evidence/" + TARGET_TEST]
    require(len(before_placeholder) == BEFORE_IDENTITIES[PLACEHOLDER][0] and sha(before_placeholder) == BEFORE_IDENTITIES[PLACEHOLDER][1], "production source before identity mismatch")
    require(len(before_test) == BEFORE_IDENTITIES[TARGET_TEST][0] and sha(before_test) == BEFORE_IDENTITIES[TARGET_TEST][1], "target test before identity mismatch")
    expected_placeholder, expected_test = expected_after(before_placeholder, before_test)
    after_placeholder = (root / PLACEHOLDER).read_bytes()
    after_test = (root / TARGET_TEST).read_bytes()
    require(after_placeholder == expected_placeholder, "production source has change beyond authorized guard")
    require(after_test == expected_test, "target test has change beyond authorized regression")

    prod_add, prod_del, prod_patch = diff_stats(before_placeholder, after_placeholder)
    test_add, test_del, test_patch = diff_stats(before_test, after_test)
    require((prod_add, prod_del) == (1, 0), "production diff count mismatch")
    require((test_add, test_del) == (10, 1), "test diff count mismatch")
    combined_patch = prod_patch + b"\n" + test_patch
    changes = {
        "changed_project_file_count": 2,
        "changed_project_files": [PLACEHOLDER, TARGET_TEST],
        "records": [
            {"relative_path": PLACEHOLDER, "before": record(PLACEHOLDER, before_placeholder),
             "after": record(PLACEHOLDER, after_placeholder), "additions": prod_add, "deletions": prod_del},
            {"relative_path": TARGET_TEST, "before": record(TARGET_TEST, before_test),
             "after": record(TARGET_TEST, after_test), "additions": test_add, "deletions": test_del},
        ],
        "total_additions": prod_add + test_add,
        "total_deletions": prod_del + test_del,
        "status": "PASS",
    }

    protected_rows = []
    for relative, (size, digest) in PROTECTED_IDENTITIES.items():
        data = (root / relative).read_bytes()
        same = len(data) == size and sha(data) == digest
        require(same, "explicit protected identity mismatch: " + relative)
        protected_rows.append({**record(relative, data), "expected_size_bytes": size, "expected_sha256": digest,
                               "before_after_byte_identical": True, "status": "PASS"})
    protected = {"target_count": len(protected_rows), "match_count": len(protected_rows),
                 "mismatch_count": 0, "records": protected_rows, "status": "PASS"}

    contract = source_contract(root)
    inventory = test_inventory(root)
    step10a = compare_records(root, root / V12_RESULT / "step10a_project_files_preservation.json", 30, "step10a_project")
    protected106 = compare_records(root, root / V12_RESULT / "protected_files_preservation.json", 106, "protected106")
    step10b = compare_records(root, root / V12_RESULT / "step10b_files_preservation.json", 22, "step10b",
                              allowed_changed={PLACEHOLDER})
    require(step10b["changed_paths"] == [PLACEHOLDER], "Step 10B changed scope mismatch")
    profile = compare_records(root, root / V12_RESULT / "candidate_profile_preservation.json", 22, "candidate_profile")
    candidate_baseline = load_json(root / V10_RESULT / "candidate_repository_before_after.json")
    candidate_dir = Path(candidate_baseline["path"])
    candidates = compare_records(root, root / V12_RESULT / "candidate_repository_preservation.json", 25, "candidate_repository", base_dir=candidate_dir)
    candidates.update({"candidate_count": 8, "file_count": 25})
    gameassets = gameassets_checkpoint(root)
    staged = git_staged(root)
    transaction = transaction_regression()

    assertions_spec = [
        ("fixed_v1_2_source_delivery", v12_info["status"]),
        ("fixed_v1_2_gui_evidence", gui_info["status"]),
        ("authorized_change_count_two", "PASS" if changes["changed_project_file_count"] == 2 else "FAIL"),
        ("awake_guard_once", "PASS" if contract["awake_guard_token_count"] == 1 else "FAIL"),
        ("awake_guard_first_statement", "PASS" if contract["awake_guard_first_statement_adjacency_count"] == 1 else "FAIL"),
        ("placeholder_generation_preserved", "PASS" if contract["placeholder_generation_tokens_preserved"] else "FAIL"),
        ("step10b_enabled_contract", "PASS" if contract["step10b_placeholder_component_enabled_value_count"] == 1 else "FAIL"),
        ("step10d_disabled_contract", "PASS" if contract["step10d_disabled_override_exact_adjacency_count"] == 1 else "FAIL"),
        ("visual_factory_identity", protected["status"]),
        ("visual_child_count_guard", "PASS" if contract["visual_child_count_guard_count"] == 1 else "FAIL"),
        ("wrapper_v1_2_fileid_repair", "PASS" if contract["wrapper_reserved_document_id_count"] == 0 and contract["wrapper_new_document_id_count"] == 1 else "FAIL"),
        ("target_fixture_14", inventory["status"]),
        ("static_inventory_152", inventory["status"]),
        ("step10a_30", step10a["status"]),
        ("protected_106", protected106["status"]),
        ("step10b_scope", step10b["status"]),
        ("candidate_profile_22", profile["status"]),
        ("candidate_repository_8_25", candidates["status"]),
        ("gameassets_checkpoint_242", "PASS"),
        ("gameassets_retroactive_preservation", "UNVERIFIED"),
        ("unity_gui_targeted_playmode", "UNVERIFIED"),
        ("unity_execution_zero", "PASS"),
        ("git_staged_zero", staged["status"]),
        ("transaction_failure_injection", transaction["status"]),
        ("never_overwrite", "PASS"),
    ]
    assertions = [{"assertion_id": f"A{index:03d}", "assertion_key": key, "status": status}
                  for index, (key, status) in enumerate(assertions_spec, 1)]
    require(all(row["status"] != "FAIL" for row in assertions), "assertion failure")
    counts = {status: sum(row["status"] == status for row in assertions)
              for status in ("PASS", "FAIL", "UNVERIFIED", "NOT_APPLICABLE")}
    assertion_summary = {"total": len(assertions), "counts": counts,
                         "status": "PASS_WITH_DECLARED_UNVERIFIED"}
    verdict = {
        "primary_verdict": "DISABLED_PLACEHOLDER_AWAKE_GUARD_REPAIR_SOURCE_EVIDENCE_READY",
        "unity_execution_count": 0,
        "targeted_14_and_companion_presentation_gui_gate": "UNVERIFIED",
        "status": "PASS",
        "status_strings": [
            "STEP10D_V1_3_DISABLED_PLACEHOLDER_AWAKE_GUARD_REPAIR_SOURCE_EVIDENCE_READY",
            "STEP10D_V1_2_APPROVED_COMPILE_TEST_AND_SCENE_SCOPE_PRESERVED",
            "STEP10D_V1_3_TARGETED_14_AND_COMPANION_PRESENTATION_GUI_GATE_UNVERIFIED",
            "BLOCKED_PENDING_EXTERNAL_REVIEW",
        ],
    }
    provenance = {"fixed_v1_2_source_delivery": v12_info, "fixed_v1_2_targeted_gui_evidence": gui_info,
                  "before_source": "read_only_project_evidence entries in fixed targeted GUI evidence ZIP",
                  "status": "PASS"}
    report = f"""# WordsInSilence Step 10D v1.3 Disabled Placeholder Awake Guard Repair

## Outcome

- Authorized project changes: exactly 2
- Production repair: `if (!enabled) return;` is the first statement of `PetRobotWorldPlaceholderVisuals.Awake()`
- Step 10B enabled placeholder behavior: preserved
- Step 10D serialized disabled override: preserved
- Existing targeted fixture: 14/14 names statically preserved
- Full static inventory: 152/152 names preserved
- Unity GUI / Test Runner / PlayMode executions: 0 (UNVERIFIED)
- GameAssets comparison: post-Step10D checkpoint 242/242; retroactive preimplementation preservation remains UNVERIFIED_NONBLOCKING

## Minimal diff

- `{PLACEHOLDER}`: +{prod_add}/-{prod_del}
- `{TARGET_TEST}`: +{test_add}/-{test_del}
- total: +{prod_add + test_add}/-{prod_del + test_del}

The approved visual factory and its `anchor.childCount != 0` guard were not changed. Scene, Prefabs,
wrapper, metas, GameAssets, candidate profile, and candidate repository remain fixed. This self-validation
is pending ChatGPT Work external review.
""".encode("utf-8")

    files: dict[str, bytes] = {
        "input_provenance.json": jbytes(provenance),
        "change_summary.json": jbytes(changes),
        "disabled_placeholder_awake_guard_contract.json": jbytes(contract),
        "test_inventory.json": jbytes(inventory),
        "explicit_protected_files_preservation.json": jbytes(protected),
        "step10a_project_files_preservation.json": jbytes(step10a),
        "protected_files_preservation.json": jbytes(protected106),
        "step10b_files_preservation.json": jbytes(step10b),
        "candidate_profile_preservation.json": jbytes(profile),
        "candidate_repository_preservation.json": jbytes(candidates),
        "gameassets_post_step10d_checkpoint_comparison.json": jbytes(gameassets),
        "git_staged_before_after.json": jbytes(staged),
        "packaging_transaction_failure_injection_regression.json": jbytes(transaction),
        "primary_verdict.json": jbytes(verdict),
        "ASSERTIONS.jsonl": b"".join(jbytes(row) for row in assertions),
        "assertion_summary.json": jbytes(assertion_summary),
        "VALIDATION_REPORT.md": report,
        "before/" + PLACEHOLDER: before_placeholder,
        "after/" + PLACEHOLDER: after_placeholder,
        "before/" + TARGET_TEST: before_test,
        "after/" + TARGET_TEST: after_test,
        "diff/disabled_placeholder_awake_guard_repair.patch": combined_patch,
        "read_only_protected/" + VISUAL_FACTORY: (root / VISUAL_FACTORY).read_bytes(),
        "read_only_protected/" + SCENE: (root / SCENE).read_bytes(),
        "read_only_protected/" + WRAPPER: (root / WRAPPER).read_bytes(),
        "read_only_protected/" + PRESENTATION: (root / PRESENTATION).read_bytes(),
        "read_only_protected/" + WORLD_PREFAB: (root / WORLD_PREFAB).read_bytes(),
        "analysis/s001_step10d_disabled_placeholder_awake_guard_repair_v1_3.py": Path(__file__).read_bytes(),
    }
    manifest_rows = [record(name, data) for name, data in sorted(files.items())]
    files["output_file_manifest.json"] = jbytes({"self_excluded": True, "target_count": len(manifest_rows),
                                                  "files": manifest_rows, "status": "PASS"})
    zip_blob = deterministic_zip(files)
    validate_zip(zip_blob, files)
    require(deterministic_zip(files) == zip_blob, "deterministic in-memory ZIP mismatch")

    def validator(result: Path, delivery: Path, detached: Path) -> None:
        validate_zip(delivery.read_bytes(), files)
        require(detached.read_text(encoding="ascii").strip().split() == [sha(delivery.read_bytes()), delivery.name], "detached checksum mismatch")
        require(len([path for path in result.rglob("*") if path.is_file()]) == len(files), "result file count mismatch")

    result, delivery, detached = commit_outputs(validation, files, names, validator=validator)
    print(json.dumps({
        "status": "PASS",
        "result_directory": str(result),
        "result_file_count": len(files),
        "manifest_target_count": len(manifest_rows),
        "changed_project_files": [PLACEHOLDER, TARGET_TEST],
        "before_after": changes["records"],
        "diff": {"production": {"additions": prod_add, "deletions": prod_del},
                 "test": {"additions": test_add, "deletions": test_del},
                 "total": {"additions": prod_add + test_add, "deletions": prod_del + test_del}},
        "awake_guard_count": contract["awake_guard_token_count"],
        "awake_guard_first_statement_count": contract["awake_guard_first_statement_adjacency_count"],
        "targeted_static_test_count": inventory["targeted_fixture_count"],
        "total_static_test_count": inventory["total_static_test_count"],
        "unity_execution_count": 0,
        "git_staged_count": 0,
        "delivery_zip": str(delivery),
        "delivery_size_bytes": delivery.stat().st_size,
        "delivery_sha256": sha(delivery.read_bytes()),
        "detached_size_bytes": detached.stat().st_size,
        "detached_file_sha256": sha(detached.read_bytes()),
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
