#!/usr/bin/env python3
"""Validate and package Step 9K personal live recognition technical prefab.

The tool parses the prefab YAML bytes directly, verifies the explicit local
Step 9C bridge -> Step 9J harness reference, checks GUID uniqueness and fixed
project identities, and packages through the approved Step 9I v1.1 rollback
transaction. It does not launch Unity, PlayMode, a provider, or a webcam.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any


SCHEMA = "words_in_silence.step9k.personal_live_recognition_technical_prefab.v1_0"
PREFAB_REL = Path("Assets/_Project/Resources/Diagnostics/PersonalLiveRecognitionTechnical.prefab")
PREFAB_META_REL = Path(str(PREFAB_REL) + ".meta")
RESULT_REL = Path("Validation/WordsInSilence_step9k_personal_live_recognition_technical_prefab_v1_0")
DELIVERY_REL = Path("Validation/WordsInSilence_step9k_personal_live_recognition_technical_prefab_v1_0_delivery.zip")
TOOL_REL = Path("Tools/s001_step9k_personal_live_recognition_technical_prefab_v1_0.py")

BRIDGE_SCRIPT_GUID = "c6773d5ba8b0ae3498e54fbe28cf8e97"
HARNESS_SCRIPT_GUID = "b683587873954482b05ddb9c95f690c4"
PREFAB_GUID = "5d2c771219eb48fc88e3b3d62aa37c15"
ROOT_FILE_ID = 100000
TRANSFORM_FILE_ID = 400000
BRIDGE_FILE_ID = 11400000
HARNESS_FILE_ID = 11400001

TRANSACTION_TOOL_REL = Path("Tools/s001_step9i_personal_runtime_opt_in_integration_v1_1.py")
TRANSACTION_TOOL_SIZE = 37_722
TRANSACTION_TOOL_SHA256 = "30471dc6725bdbdd0cc80c345b238485d1cb0d8d7a5933aac7477d536d714caf"
STEP9J_V10_TOOL_REL = Path("Tools/s001_step9j_personal_live_recognition_integration_v1_0.py")
STEP9J_V10_TOOL_SIZE = 47_009
STEP9J_V10_TOOL_SHA256 = "72c716128275f638a179de195daa90668097c83852249349e068ce45d12a6d69"

STEP9J_FIXED = (
    ("Assets/_Project/Scripts/Recognition/LiveInput/PersonalLiveRecognitionController.cs", 30_013, "69fc64354c0f1466b9b161cddaacb13087a185c77cc98304bbdb86daaf006ba7"),
    ("Assets/_Project/Scripts/Recognition/LiveInput/PersonalLiveRecognitionController.cs.meta", 243, "5262ea0d5605c95fc570033432bb000b7cb2c22fe51b376c4039227cfdc9280d"),
    ("Assets/_Project/Scripts/Recognition/LiveInput/PersonalLiveRecognitionHarness.cs", 7_209, "b5cee1181f5b0927dc431486d23ccea4ee4778466de729686e0a451342a8b81f"),
    ("Assets/_Project/Scripts/Recognition/LiveInput/PersonalLiveRecognitionHarness.cs.meta", 243, "f54efe92219b323e99c3a1a9466ae6ac5a2cc28569c171a1534683dda98d722e"),
    ("Assets/_Project/Tests/EditMode/PersonalLiveRecognitionIntegrationV1Tests.cs", 30_226, "7b53e98f93ef165e3c283ddcc6434e0254caacd22370c5e6dc66888bdf93775d"),
    ("Assets/_Project/Tests/EditMode/PersonalLiveRecognitionIntegrationV1Tests.cs.meta", 243, "48271faad1512b429086a906ed9d454f5dfaed07f5b1a6d5d2f4c1267e01769c"),
)

SCENE_AND_PREFAB_FIXED = (
    ("Assets/_Project/Resources/Diagnostics/PhysicalWebcamCoordinateCompatibility.prefab", 1_792, "f5910cf8381bd7769e518a70b57b1680c9d90709ac13f96689729094eb38f74b", "existing_step9c_prefab"),
    ("Assets/_Project/Resources/Diagnostics/PhysicalWebcamCoordinateCompatibility.prefab.meta", 154, "678840d956656c26f743d49e5b17da79bb4adf8f6277df72a2c999fd539a88dc", "existing_step9c_prefab_meta"),
    ("Assets/_Project/Scenes/Technical/02_PoseIntegration.unity", 240_625, "1fb83d0f304277317a4571319e432c8d85394c7f84abce9036e23a6b3de16a0b", "provider_scene"),
    ("Assets/_Project/Scenes/Technical/02_PoseIntegration.unity.meta", 155, "d5a0540425a760d73fb1b3d7c5b65b79a1e9f1f2528466af4e1f735dec7f0b8a", "provider_scene_meta"),
    ("Assets/_Project/Scenes/Game/03_SignPractice.unity", 40_083, "dac9d8d51a561825314cd8fb4021e4e803590d9275d3d65b55af96bfa47338e8", "main_game_scene"),
    ("Assets/_Project/Scenes/Game/03_SignPractice.unity.meta", 155, "3e0de2ecfd5cfaf5cccde62e1c1cf7175bf3c59ae652322f1041557599413aa7", "main_game_scene_meta"),
)

CONTEXT_FIXED = (
    ("AGENTS.md", 1_666, "fd9126d3f965283461bf5ed88a7a3b58bb8e3b4ca41bebe1c8ebda3cd7652085"),
    ("WordsInSilence_PROJECT_CONTEXT.md", 80_875, "efe9a44ad90bd1a9f59dbb761f2f206fe6f6ba05ddca2ba0e28fe76ddd0a8072"),
)

FORBIDDEN_TOKENS = (
    "MediaPipeLandmarkProvider", "HandPoseIntegrator", "WebCamTexture",
    "MediaPipeHandPoseObservationBridge", "PhysicalWebcamCoordinateCompatibilityHarness",
    "PhysicalCaptureEvidenceRecorder", "PhysicalCaptureEvidence", "camera provider",
    "RuntimeRecognitionService", "MaskAwareDtw", "threshold", "template",
)


class Step9KError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise Step9KError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def import_fixed(root: Path, relative: Path, size: int, digest: str, name: str):
    path = root / relative
    require(path.is_file() and path.stat().st_size == size and sha256_file(path) == digest,
            "fixed tool identity mismatch: " + relative.as_posix())
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "cannot import fixed tool")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_fixed_records(root: Path, definitions) -> tuple[list[dict[str, Any]], dict[str, str]]:
    records = []
    snapshot = {}
    for definition in definitions:
        relative, size, digest = definition[:3]
        path = root / PurePosixPath(relative)
        require(path.is_file(), "fixed file missing: " + relative)
        actual_size = path.stat().st_size
        actual_digest = sha256_file(path)
        require(actual_size == size and actual_digest == digest, "fixed file identity mismatch: " + relative)
        record = {"path": relative, "size_bytes": size, "sha256": digest, "byte_identical": True}
        if len(definition) > 3:
            record["role"] = definition[3]
        records.append(record)
        snapshot[relative] = digest
    return records, snapshot


def parse_prefab(prefab_bytes: bytes) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    require(b"\x00" not in prefab_bytes, "prefab contains NUL")
    text = prefab_bytes.decode("utf-8")
    require(text.startswith("%YAML 1.1\n%TAG !u! tag:unity3d.com,2011:\n"), "prefab YAML header mismatch")
    documents = re.findall(r"^--- !u!([0-9]+) &([0-9]+)\n([\s\S]*?)(?=^--- !u!|\Z)", text, re.M)
    require(len(documents) == 4, "prefab document count must be four")
    by_id = {int(file_id): (int(class_id), body) for class_id, file_id, body in documents}
    require(len(by_id) == 4, "prefab duplicate local fileID")
    require(set(by_id) == {ROOT_FILE_ID, TRANSFORM_FILE_ID, BRIDGE_FILE_ID, HARNESS_FILE_ID},
            "prefab local fileID set mismatch")
    require(by_id[ROOT_FILE_ID][0] == 1 and by_id[TRANSFORM_FILE_ID][0] == 4 and
            by_id[BRIDGE_FILE_ID][0] == by_id[HARNESS_FILE_ID][0] == 114,
            "prefab class ID mapping mismatch")
    root = by_id[ROOT_FILE_ID][1]
    transform = by_id[TRANSFORM_FILE_ID][1]
    bridge = by_id[BRIDGE_FILE_ID][1]
    harness = by_id[HARNESS_FILE_ID][1]
    require(re.search(r"^  m_Name: PersonalLiveRecognitionTechnical$", root, re.M) is not None,
            "prefab root name mismatch")
    require(re.search(r"^  m_IsActive: 1$", root, re.M) is not None, "prefab root is inactive")
    component_ids = [int(value) for value in re.findall(r"^  - component: \{fileID: ([0-9]+)\}$", root, re.M)]
    require(component_ids == [TRANSFORM_FILE_ID, BRIDGE_FILE_ID, HARNESS_FILE_ID],
            "prefab component list/order mismatch")
    identity_lines = (
        "m_LocalRotation: {x: 0, y: 0, z: 0, w: 1}",
        "m_LocalPosition: {x: 0, y: 0, z: 0}",
        "m_LocalScale: {x: 1, y: 1, z: 1}",
        "m_Father: {fileID: 0}",
    )
    require(all(token in transform for token in identity_lines), "prefab Transform is not identity/root")
    bridge_script = re.findall(r"m_Script: \{fileID: 11500000, guid: ([0-9a-f]{32}), type: 3\}", bridge)
    harness_script = re.findall(r"m_Script: \{fileID: 11500000, guid: ([0-9a-f]{32}), type: 3\}", harness)
    require(bridge_script == [BRIDGE_SCRIPT_GUID], "Step 9C bridge script GUID mismatch")
    require(harness_script == [HARNESS_SCRIPT_GUID], "Step 9J harness script GUID mismatch")
    require(re.search(r"^  _integrator: \{fileID: 0\}$", bridge, re.M) is not None,
            "prefab Step 9C integrator must be null")
    local_reference = re.search(r"^  observationBridge: \{fileID: ([0-9]+)\}$", harness, re.M)
    require(local_reference is not None and int(local_reference.group(1)) == BRIDGE_FILE_ID,
            "harness observationBridge local reference mismatch")
    panel = re.search(
        r"^  panelRect:\n    serializedVersion: 2\n    x: ([^\n]+)\n    y: ([^\n]+)\n    width: ([^\n]+)\n    height: ([^\n]+)$",
        harness, re.M)
    require(panel is not None and tuple(float(value) for value in panel.groups()) == (16.0, 16.0, 520.0, 580.0),
            "Step 9J panel rect mismatch")
    guid_references = re.findall(r"\{fileID: [^,}]+, guid: ([0-9a-f]{32}), type: [0-9]+\}", text)
    require(sorted(guid_references) == sorted([BRIDGE_SCRIPT_GUID, HARNESS_SCRIPT_GUID]),
            "prefab contains unexpected external asset reference")
    local_references = [int(value) for value in re.findall(r"\{fileID: (-?[0-9]+)\}", text)]
    dangling = sorted(set(value for value in local_references if value != 0 and value not in by_id))
    require(not dangling, "prefab contains dangling local reference: " + str(dangling))
    forbidden_hits = [token for token in FORBIDDEN_TOKENS if token in text]
    require(not forbidden_hits, "prefab contains prohibited component/ownership token: " + str(forbidden_hits))
    inventory = {
        "schema_version": SCHEMA + ".prefab_component_inventory",
        "root_game_object": "PersonalLiveRecognitionTechnical", "root_count": 1,
        "active": True, "transform_identity": True, "component_count": 3,
        "components": [
            {"type": "Transform", "file_id": TRANSFORM_FILE_ID, "count": 1},
            {"type": "Step9CHandIdentityObservationBridge", "file_id": BRIDGE_FILE_ID,
             "script_guid": BRIDGE_SCRIPT_GUID, "count": 1},
            {"type": "PersonalLiveRecognitionHarness", "file_id": HARNESS_FILE_ID,
             "script_guid": HARNESS_SCRIPT_GUID, "count": 1},
        ],
        "panel_rect": {"x": 16, "y": 16, "width": 520, "height": 580}, "status": "PASS",
    }
    references = {
        "schema_version": SCHEMA + ".serialized_local_reference_mapping",
        "harness_file_id": HARNESS_FILE_ID, "field": "observationBridge",
        "referenced_file_id": BRIDGE_FILE_ID, "referenced_component": "Step9CHandIdentityObservationBridge",
        "same_prefab_local_reference": True, "integrator_file_id": 0,
        "dangling_local_reference_count": 0, "external_scene_reference_count": 0,
        "external_asset_references": [
            {"role": "Step9C bridge script", "guid": BRIDGE_SCRIPT_GUID, "file_id": 11500000},
            {"role": "Step9J harness script", "guid": HARNESS_SCRIPT_GUID, "file_id": 11500000},
        ], "status": "PASS",
    }
    prohibited = {
        "schema_version": SCHEMA + ".forbidden_component_provider_ownership_check",
        "forbidden_tokens_checked": list(FORBIDDEN_TOKENS), "forbidden_hit_count": 0,
        "script_component_guid_count": 2, "only_approved_script_guids": True,
        "camera_or_provider_component_count": 0, "preview_component_count": 0,
        "runtime_loader_component_count": 0, "dtw_threshold_template_component_count": 0,
        "status": "PASS",
    }
    return inventory, references, prohibited


def validate_meta_and_guid(root: Path, meta_bytes: bytes) -> dict[str, Any]:
    text = meta_bytes.decode("utf-8")
    require(re.search(r"^fileFormatVersion: 2$", text, re.M) is not None, "prefab meta version mismatch")
    match = re.search(r"^guid: ([0-9a-f]{32})$", text, re.M)
    require(match is not None and match.group(1) == PREFAB_GUID, "prefab meta GUID mismatch")
    require("PrefabImporter:" in text, "prefab meta importer mismatch")
    records = []
    occurrences = []
    meta_count = 0
    malformed_count = 0
    for path in sorted((root / "Assets").rglob("*.meta")):
        meta_count += 1
        try:
            candidate = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            malformed_count += 1
            continue
        found = re.search(r"^guid: ([0-9a-f]{32})$", candidate, re.M)
        if found is None:
            malformed_count += 1
            continue
        if found.group(1) == PREFAB_GUID:
            occurrences.append(path.relative_to(root).as_posix())
    require(occurrences == [PREFAB_META_REL.as_posix()], "new prefab GUID collides: " + str(occurrences))
    records.append({"guid": PREFAB_GUID, "occurrence_count": len(occurrences), "paths": occurrences})
    return {"schema_version": SCHEMA + ".guid_uniqueness", "assets_meta_scanned": meta_count,
            "meta_without_standard_guid_count": malformed_count, "new_guid": PREFAB_GUID,
            "new_guid_occurrence_count": 1, "collision_count": 0, "records": records, "status": "PASS"}


def staged_count(root: Path) -> int:
    result = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=root,
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    require(result.returncode == 0, "git staged-file query failed")
    return len([line for line in result.stdout.splitlines() if line.strip()])


def assertion(key: str, status: str, subject: str, evidence: Any) -> dict[str, Any]:
    return {"assertion_key": key, "status": status, "subject": subject, "evidence": evidence}


def build(root: Path) -> dict[str, Any]:
    result = root / RESULT_REL
    delivery = root / DELIVERY_REL
    detached = Path(str(delivery) + ".sha256")
    validation_root = root / "Validation"
    for target in (result, delivery, detached):
        require(not target.exists() and not target.is_symlink(), "never-overwrite target exists: " + str(target))
    transaction = import_fixed(root, TRANSACTION_TOOL_REL, TRANSACTION_TOOL_SIZE,
                               TRANSACTION_TOOL_SHA256, "step9i_v11_transaction")
    step9j_v10 = import_fixed(root, STEP9J_V10_TOOL_REL, STEP9J_V10_TOOL_SIZE,
                             STEP9J_V10_TOOL_SHA256, "step9j_v10_fixed")
    require((root / PREFAB_REL).is_file() and (root / PREFAB_META_REL).is_file(), "new prefab or meta missing")
    prefab_bytes = (root / PREFAB_REL).read_bytes()
    meta_bytes = (root / PREFAB_META_REL).read_bytes()
    inventory, references, prohibited = parse_prefab(prefab_bytes)
    guid = validate_meta_and_guid(root, meta_bytes)
    step9j_records, step9j_before = verify_fixed_records(root, STEP9J_FIXED)
    scene_records, scene_before = verify_fixed_records(root, SCENE_AND_PREFAB_FIXED)
    context_records, context_before = verify_fixed_records(root, CONTEXT_FIXED)
    approved, approved_before = step9j_v10.verify_fixed_files(root)
    new_before = {PREFAB_REL.as_posix(): sha256_bytes(prefab_bytes),
                  PREFAB_META_REL.as_posix(): sha256_bytes(meta_bytes)}
    require(staged_count(root) == 0, "Git staged file count is nonzero")

    transaction_identity = {"records": [{"project_relative_path": relative} for relative in sorted(
        set(step9j_before) | set(scene_before) | set(context_before) | set(approved_before) | set(new_before))]}
    injection = transaction.run_failure_injection_regression(root, validation_root, transaction_identity)
    never_overwrite = transaction.run_never_overwrite_regression(validation_root)
    require(injection["verdict"] == "PASS" and never_overwrite["verdict"] == "PASS",
            "packaging transaction regression failed")
    step9j_after = {relative: sha256_file(root / PurePosixPath(relative)) for relative in step9j_before}
    scene_after = {relative: sha256_file(root / PurePosixPath(relative)) for relative in scene_before}
    context_after = {relative: sha256_file(root / PurePosixPath(relative)) for relative in context_before}
    approved_after = {relative: sha256_file(root / PurePosixPath(relative)) for relative in approved_before}
    new_after = {relative: sha256_file(root / PurePosixPath(relative)) for relative in new_before}
    require(step9j_before == step9j_after and scene_before == scene_after and context_before == context_after and
            approved_before == approved_after and new_before == new_after,
            "input or new prefab changed during validation")

    step9j_preservation = {"schema_version": SCHEMA + ".step9j_fixed_file_preservation",
                           "file_count": len(step9j_records), "mismatch_count": 0,
                           "before_after_identical": True, "records": step9j_records, "status": "PASS"}
    scene_preservation = {"schema_version": SCHEMA + ".scene_existing_prefab_preservation",
                          "file_count": len(scene_records), "mismatch_count": 0,
                          "before_after_identical": True, "records": scene_records, "status": "PASS"}
    approved_preservation = {"schema_version": SCHEMA + ".step9c_step9i_runtime_asset_preservation",
                             "file_count": approved["file_count"], "mismatch_count": 0,
                             "before_after_identical": True, "records": approved["records"], "status": "PASS"}
    primary = {"schema_version": SCHEMA + ".primary_verdict",
               "primary_verdict": "STEP9K_PERSONAL_LIVE_RECOGNITION_TECHNICAL_PREFAB_V1_0_SOURCE_AND_PACKAGE_READY",
               "explicit_step9c_bridge_to_step9j_harness_prefab_wiring": "IMPLEMENTED",
               "step9j_source_and_assets": "PRESERVED", "step9c_provider_and_scenes": "PRESERVED",
               "unity_import_playmode_webcam": "UNVERIFIED", "external_review": "PENDING",
               "status_strings": [
                   "STEP9K_PERSONAL_LIVE_RECOGNITION_TECHNICAL_PREFAB_V1_0_SOURCE_AND_PACKAGE_READY",
                   "STEP9K_EXPLICIT_STEP9C_BRIDGE_TO_STEP9J_HARNESS_PREFAB_WIRING_IMPLEMENTED",
                   "STEP9J_APPROVED_SOURCE_AND_ASSETS_PRESERVED",
                   "STEP9C_PROVIDER_AND_SCENES_PRESERVED",
                   "STEP9K_UNITY_IMPORT_PLAYMODE_WEBCAM_UNVERIFIED",
                   "BLOCKED_PENDING_EXTERNAL_REVIEW"]}
    assertions = [
        assertion("new_prefab_and_meta_exist", "PASS", PREFAB_REL.as_posix(), {"prefab_size": len(prefab_bytes), "meta_size": len(meta_bytes)}),
        assertion("prefab_root_name_exact", "PASS", "PersonalLiveRecognitionTechnical", True),
        assertion("prefab_component_inventory_exact_three", "PASS", PREFAB_REL.as_posix(), inventory),
        assertion("prefab_transform_identity_active", "PASS", PREFAB_REL.as_posix(), {"active": True, "identity": True}),
        assertion("harness_bridge_explicit_local_reference", "PASS", "observationBridge", references),
        assertion("step9c_integrator_serialized_null", "PASS", "_integrator", 0),
        assertion("step9j_panel_rect_default_preserved", "PASS", "panelRect", inventory["panel_rect"]),
        assertion("no_dangling_or_external_scene_reference", "PASS", PREFAB_REL.as_posix(), {"dangling": 0, "external_scene": 0}),
        assertion("forbidden_component_provider_ownership_zero", "PASS", PREFAB_REL.as_posix(), prohibited),
        assertion("prefab_guid_unique", "PASS", PREFAB_META_REL.as_posix(), guid),
        assertion("step9j_fixed_six_preserved", "PASS", "Step 9J", {"count": len(step9j_records)}),
        assertion("existing_step9c_prefab_preserved", "PASS", SCENE_AND_PREFAB_FIXED[0][0], True),
        assertion("provider_scene_preserved", "PASS", SCENE_AND_PREFAB_FIXED[2][0], True),
        assertion("main_game_scene_preserved", "PASS", SCENE_AND_PREFAB_FIXED[4][0], True),
        assertion("approved_step9c_step9i_runtime_assets_preserved", "PASS", "approved files", {"count": approved["file_count"]}),
        assertion("context_not_modified", "PASS", "AGENTS+PROJECT_CONTEXT", {"count": len(context_records)}),
        assertion("packaging_failure_injection_four", "PASS", "approved transaction", injection),
        assertion("never_overwrite", "PASS", "approved transaction", never_overwrite),
        assertion("unity_import_playmode_webcam_not_claimed", "PASS", "execution scope", "UNVERIFIED"),
        assertion("git_staged_zero", "PASS", "Git index", 0),
    ]
    keys = [item["assertion_key"] for item in assertions]
    require(len(keys) == len(set(keys)), "duplicate assertion key")

    file_map: dict[str, bytes] = {
        "input_provenance.json": json_bytes({"schema_version": SCHEMA + ".input_provenance",
                                               "approved_transaction_tool": {"path": TRANSACTION_TOOL_REL.as_posix(),
                                               "size_bytes": TRANSACTION_TOOL_SIZE, "sha256": TRANSACTION_TOOL_SHA256},
                                               "step9j_fixed_file_count": len(step9j_records),
                                               "approved_step9c_step9i_file_count": approved["file_count"]}),
        "project_files/PersonalLiveRecognitionTechnical.prefab": prefab_bytes,
        "project_files/PersonalLiveRecognitionTechnical.prefab.meta": meta_bytes,
        "prefab_component_inventory.json": json_bytes(inventory),
        "serialized_local_reference_mapping.json": json_bytes(references),
        "forbidden_component_provider_ownership_check.json": json_bytes(prohibited),
        "guid_uniqueness.json": json_bytes(guid),
        "step9j_fixed_file_preservation.json": json_bytes(step9j_preservation),
        "step9c_step9i_runtime_asset_preservation.json": json_bytes(approved_preservation),
        "scene_existing_prefab_preservation.json": json_bytes(scene_preservation),
        "packaging_transaction_failure_injection_regression.json": json_bytes(injection),
        "never_overwrite_regression.json": json_bytes(never_overwrite),
        "primary_verdict.json": json_bytes(primary),
        "ASSERTIONS.jsonl": b"".join(json.dumps(item, ensure_ascii=False, allow_nan=False,
                                                  separators=(",", ":")).encode("utf-8") + b"\n" for item in assertions),
        "analysis/s001_step9k_personal_live_recognition_technical_prefab_v1_0.py": (root / TOOL_REL).read_bytes(),
    }
    report = "\n".join([
        "# WordsInSilence Step 9K — Personal Live Recognition Technical Prefab v1.0", "",
        "## Outcome", "",
        "A new active diagnostic prefab contains exactly Transform, the approved Step 9C bridge, and the approved Step 9J harness. The harness observationBridge field is a local reference to the bridge component; the bridge integrator remains null in the asset.",
        "", "## Offline serialization verification", "",
        "- Root/component counts: `1 / 3`", "- Local harness → bridge reference: `11400001 → 11400000`",
        "- Step 9C integrator fileID: `0`", "- Transform: identity; panel rect: `16,16,520,580`",
        f"- GUID collision count: `{guid['collision_count']}`; forbidden component/ownership hits: `{prohibited['forbidden_hit_count']}`",
        f"- Step 9J fixed files: `{len(step9j_records)}/6`; scene/existing prefab files: `{len(scene_records)}/6`; approved runtime files: `{approved['file_count']}`",
        f"- Packaging failure injection: `{injection['pass_count']}/{injection['injection_point_count']}`; never-overwrite: `{never_overwrite['verdict']}`",
        "", "## Limits", "",
        "Unity import, Unity compilation, PlayMode, and live webcam were not executed or claimed. The prefab must remain external-review pending before temporary placement in 02_PoseIntegration.",
        "", "## Status", "", "```text", *primary["status_strings"], "```", "",
    ])
    file_map["VALIDATION_REPORT.md"] = report.encode("utf-8")
    records = [{"path": name, "size_bytes": len(data), "sha256": sha256_bytes(data)}
               for name, data in sorted(file_map.items())]
    file_map["output_file_manifest.json"] = json_bytes({"schema_version": SCHEMA + ".output_file_manifest",
                                                          "self_excluded": True, "file_count": len(records),
                                                          "files": records})
    zip_one = transaction.deterministic_zip_bytes(file_map)
    require(zip_one == transaction.deterministic_zip_bytes(file_map), "deterministic ZIP regeneration mismatch")
    detached_bytes = transaction.strict_detached_bytes(zip_one, delivery.name)

    def post_validate(final_result: Path, final_delivery: Path, final_detached: Path) -> None:
        require(final_delivery.read_bytes() == zip_one, "committed delivery differs")
        transaction.verify_strict_detached(final_detached.read_bytes(), zip_one, final_delivery.name)
        require(all((final_result / PurePosixPath(name)).read_bytes() == data for name, data in file_map.items()),
                "committed result differs")
        safety = transaction.inspect_zip(zip_one, set(file_map))
        require(safety["crc"] == "PASS" and safety["duplicates"] == safety["unsafe"] ==
                safety["symlinks"] == safety["encrypted"] == 0 and safety["membership"] == "PASS",
                "committed ZIP safety failure")

    transaction.commit_packaging_transaction(
        result_files=file_map, delivery_bytes=zip_one, detached_bytes=detached_bytes,
        final_result=result, final_delivery=delivery, final_detached=detached,
        allowed_root=validation_root, post_rename_validator=post_validate,
        transaction_token="step9k_v1_0_production")
    _, step9j_final = verify_fixed_records(root, STEP9J_FIXED)
    _, scene_final = verify_fixed_records(root, SCENE_AND_PREFAB_FIXED)
    _, context_final = verify_fixed_records(root, CONTEXT_FIXED)
    _, approved_final = step9j_v10.verify_fixed_files(root)
    require(step9j_final == step9j_before and scene_final == scene_before and context_final == context_before and
            approved_final == approved_before and sha256_file(root / PREFAB_REL) == new_before[PREFAB_REL.as_posix()] and
            sha256_file(root / PREFAB_META_REL) == new_before[PREFAB_META_REL.as_posix()],
            "project identity changed during package commit")
    require(staged_count(root) == 0, "Git staged files changed during package commit")
    return {"prefab": str(root / PREFAB_REL), "prefab_size_bytes": len(prefab_bytes),
            "prefab_sha256": sha256_bytes(prefab_bytes), "prefab_meta_size_bytes": len(meta_bytes),
            "prefab_meta_sha256": sha256_bytes(meta_bytes), "result_directory": str(result),
            "result_file_count": len(file_map), "delivery_zip": str(delivery),
            "delivery_size_bytes": len(zip_one), "delivery_sha256": sha256_bytes(zip_one),
            "detached_checksum": str(detached), "detached_size_bytes": len(detached_bytes),
            "detached_file_sha256": sha256_bytes(detached_bytes), "assertion_count": len(assertions),
            "assertion_pass_count": len(assertions), "git_staged_file_count": 0,
            "status_strings": primary["status_strings"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(build(args.project_root.resolve()), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
