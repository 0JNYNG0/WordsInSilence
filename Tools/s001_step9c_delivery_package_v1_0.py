#!/usr/bin/env python3
"""Packaging-only Step 9C Delivery builder.

This module has no import-time writes. It never launches Unity, accesses a
camera, creates physical captures, or modifies production/test/evidence files.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import uuid
import xml.etree.ElementTree as ET
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/processed/sessions/S001_P001/step9c_physical_webcam_coordinate_compatibility_v1_0"
PHYSICAL = OUTPUT / "physical_captures"
RUN_ID = "8d17ef49-89ff-4d40-ae59-0386cc406e2e"

VERIFIER = ROOT / "Tools/s001_step9c_physical_webcam_coordinate_compatibility_v1_0.py"
PACKAGER = ROOT / "Tools/s001_step9c_delivery_package_v1_0.py"
STEP9B_INPUT = ROOT / "data/processed/sessions/S001_P001/step9b_live_mediapipe_input_v1_0/input_manifest.json"
STEP9B_OUTPUT = ROOT / "data/processed/sessions/S001_P001/step9b_live_mediapipe_input_v1_0/output_file_manifest.json"

DELIVERY_NAME = "WordsInSilence_step9c_physical_webcam_coordinate_compatibility_v1_0_delivery.zip"
DELIVERY_ZIP = ROOT / DELIVERY_NAME
DELIVERY_HASH = ROOT / f"{DELIVERY_NAME}.sha256"
DELIVERY_SIDECAR = ROOT / "WordsInSilence_step9c_physical_webcam_coordinate_compatibility_v1_0_delivery_post_zip_verification.json"

HANDOFF_SHA256 = "32bb65ec049bccea2e1bb3e991ea50bdf038868988e0fce732a4ec62c7b01fcd"
VERIFIER_SHA256 = "692e10f99a8588d6d6abb0b53cd6fa6dbc79ae906151fb64c21bcc4e1fdcd0ef"
EDIT_TEST_SHA256 = "9e9be52b4e4eb4915e7aedbdac3652c1f44dff816e9985dbb0b380d625a9b975"
PLAY_TEST_SHA256 = "ab3aaa064eb3bda429f27e94b7ff40a4ff41d02649ea437a676e81e389253cb8"
NEW_THANK_YOU_SHA256 = "c30e68736fd9762864c6d5da3cb5e2c995060c0136a99e48391e8f5b5572f5c2"

EDIT_TEST_REL = "Assets/_Project/Tests/EditMode/PhysicalWebcamCoordinateCompatibilityV1Tests.cs"
PLAY_TEST_REL = "Assets/_Project/Tests/PlayMode/PhysicalWebcamCoordinateCompatibilityHarnessV1Tests.cs"
PREFAB_REL = "Assets/_Project/Resources/Diagnostics/PhysicalWebcamCoordinateCompatibility.prefab"
SCENE_REL = "Assets/_Project/Scenes/Technical/02_PoseIntegration.unity"
CAPTURE_PLAN_REL = "Assets/StreamingAssets/RecognitionCoreV1/PhysicalCompatibility/step9c_capture_plan.json"
EDITOR_EVIDENCE_REL = "data/processed/sessions/S001_P001/step9c_physical_webcam_coordinate_compatibility_v1_0/editor_test_evidence"

EDITOR_FILES = {
    "Step9C_EditMode_21PASS.xml",
    "Step9C_EditMode_21PASS_visual.png",
    "Step9C_PlayMode_3PASS.xml",
    "Step9C_PlayMode_3PASS_visual.png",
    "editor_test_evidence_manifest.json",
}

PRODUCTION_REL = [
    "Assets/_Project/Scripts/Recognition/PhysicalCompatibility/PhysicalCaptureEvidenceJson.cs",
    "Assets/_Project/Scripts/Recognition/PhysicalCompatibility/PhysicalCaptureEvidenceRecorder.cs",
    "Assets/_Project/Scripts/Recognition/PhysicalCompatibility/PhysicalCaptureEvidenceResume.cs",
    "Assets/_Project/Scripts/Recognition/PhysicalCompatibility/PhysicalWebcamCoordinateCompatibilityHarness.cs",
    "Assets/_Project/Scripts/Recognition/PhysicalCompatibility/Step9CCaptureLifecycle.cs",
    "Assets/_Project/Scripts/Recognition/PhysicalCompatibility/Step9CHandIdentityObservationBridge.cs",
    "Assets/_Project/Scripts/Recognition/PhysicalCompatibility/Step9CHandIdentityResolver.cs",
    "Assets/_Project/Scripts/Recognition/PhysicalCompatibility/Step9CPhysicalCaptureEvidence.cs",
]

RELATED_APPROVED_REL = [
    "Assets/_Project/Scripts/Recognition/LiveInput/MediaPipeHandPoseObservationBridge.cs",
    "Assets/_Project/Scripts/Landmarks/MediaPipeLandmarkProvider.cs",
    "Assets/_Project/Scripts/Pose/HandPoseIntegrator.cs",
]

FIXTURE_REL = [
    "Assets/_Project/Tests/EditMode/Fixtures/Step9CPhysicalDuplicateHandednessV1.meta",
    "Assets/_Project/Tests/EditMode/Fixtures/Step9CPhysicalDuplicateHandednessV1/physical_evidence_manifest.csv",
    "Assets/_Project/Tests/EditMode/Fixtures/Step9CPhysicalDuplicateHandednessV1/physical_evidence_manifest.csv.meta",
    "Assets/_Project/Tests/EditMode/Fixtures/Step9CPhysicalDuplicateHandednessV1/thank_you_98116ec1_duplicate_events.csv",
    "Assets/_Project/Tests/EditMode/Fixtures/Step9CPhysicalDuplicateHandednessV1/thank_you_98116ec1_duplicate_events.csv.meta",
]

APPROVED_FIXED_HASHES = {
    "Assets/_Project/Scripts/Recognition/PhysicalCompatibility/PhysicalCaptureEvidenceJson.cs": "b1a45d65f50ea5b6c626866f4876235d514ef7da6f32403d97d00199d5eb602f",
    "Assets/_Project/Scripts/Recognition/PhysicalCompatibility/PhysicalCaptureEvidenceRecorder.cs": "2d2eff465f4087df1ba4e29240f2032a22a4b1e251ac0ec012c822c6885b67d6",
    "Assets/_Project/Scripts/Recognition/PhysicalCompatibility/PhysicalCaptureEvidenceResume.cs": "aa711f9867b2844d9a0476179ef25dd2cc42e92475ef372f5c0d7805cf506c84",
    "Assets/_Project/Scripts/Recognition/PhysicalCompatibility/PhysicalWebcamCoordinateCompatibilityHarness.cs": "f1c1a43b6437830665d82447d06aedb2308a33f706255a2993dde131e6d7b1e8",
    "Assets/_Project/Scripts/Recognition/PhysicalCompatibility/Step9CCaptureLifecycle.cs": "0919e386fd9754159be95a2b1d6e3734c93dc1dc8fb4a05d66ea258e9dbe4e89",
    "Assets/_Project/Scripts/Recognition/PhysicalCompatibility/Step9CHandIdentityObservationBridge.cs": "f028f1485dc396fd9d7c38672f2a4097cb769ecfd321e45f3b63bb2f02c7adc2",
    "Assets/_Project/Scripts/Recognition/PhysicalCompatibility/Step9CHandIdentityResolver.cs": "4ea8b4a59253963e460b9ac05f6d472f8663f6f57467fae9500abadc93bdead3",
    "Assets/_Project/Scripts/Recognition/PhysicalCompatibility/Step9CPhysicalCaptureEvidence.cs": "8ec6f37a945fbbb3a12294f9a8e36b8e6f13b612cb408c47ea061133637cf7a5",
    "Assets/_Project/Scripts/Recognition/LiveInput/MediaPipeHandPoseObservationBridge.cs": "4490435a956d9ab5e607349c400503a3913231783a7e11c9d77a72734b263b13",
    "Assets/_Project/Scripts/Landmarks/MediaPipeLandmarkProvider.cs": "5be2c673cfb9470235224ca26bb792a63231b82019529c1b1045f9f01569c148",
    "Assets/_Project/Scripts/Pose/HandPoseIntegrator.cs": "620921e0bfab647c5af3d293bdb524d2b9a92052fcdb830a015ae964ac5f3703",
    PREFAB_REL: "f5910cf8381bd7769e518a70b57b1680c9d90709ac13f96689729094eb38f74b",
    SCENE_REL: "1fb83d0f304277317a4571319e432c8d85394c7f84abce9036e23a6b3de16a0b",
    CAPTURE_PLAN_REL: "21238a7b4da96c3e6f91dce2a4c0ed5fd18b827c975536a8275ada97797d9b11",
    EDIT_TEST_REL: EDIT_TEST_SHA256,
    PLAY_TEST_REL: PLAY_TEST_SHA256,
    "Tools/s001_step9c_physical_webcam_coordinate_compatibility_v1_0.py": VERIFIER_SHA256,
}

GENERATED_REL = [
    "data/processed/sessions/S001_P001/step9c_physical_webcam_coordinate_compatibility_v1_0/delivery_unity_editmode_test_summary.json",
    "data/processed/sessions/S001_P001/step9c_physical_webcam_coordinate_compatibility_v1_0/delivery_unity_playmode_test_summary.json",
    "data/processed/sessions/S001_P001/step9c_physical_webcam_coordinate_compatibility_v1_0/delivery_input_manifest.json",
    "data/processed/sessions/S001_P001/step9c_physical_webcam_coordinate_compatibility_v1_0/delivery_assertion_results.json",
    "data/processed/sessions/S001_P001/step9c_physical_webcam_coordinate_compatibility_v1_0/delivery_run_state.json",
    "data/processed/sessions/S001_P001/step9c_physical_webcam_coordinate_compatibility_v1_0/STEP9C_PHYSICAL_WEBCAM_COORDINATE_COMPATIBILITY_V1_0_DELIVERY_REPORT.md",
]
OUTPUT_MANIFEST_REL = "data/processed/sessions/S001_P001/step9c_physical_webcam_coordinate_compatibility_v1_0/output_file_manifest.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def record(path: Path, relative_path: str | None = None, category: str | None = None) -> dict:
    row = {
        "relative_path": relative_path or path.relative_to(ROOT).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if category is not None:
        row["category"] = category
    return row


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def safe_zip_name(name: str) -> bool:
    pure = PurePosixPath(name)
    return bool(name) and not pure.is_absolute() and ".." not in pure.parts and "\\" not in name


def load_existing_verifier():
    require(VERIFIER.is_file(), f"missing verifier: {VERIFIER}")
    require(sha256_file(VERIFIER) == VERIFIER_SHA256, "existing verifier SHA-256 mismatch")
    spec = importlib.util.spec_from_file_location("step9c_existing_verifier", VERIFIER)
    require(spec is not None and spec.loader is not None, "cannot load existing verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_xml(data: bytes, platform: str, assembly_name: str, fixture_name: str, expected_total: int) -> dict:
    root = ET.fromstring(data)
    require(root.tag == "test-run", f"{platform} XML root is not test-run")
    require(root.attrib.get("result") == "Passed", f"{platform} XML result is not Passed")
    require(int(root.attrib.get("total", "-1")) == expected_total, f"{platform} XML total mismatch")
    require(int(root.attrib.get("passed", "-1")) == expected_total, f"{platform} XML passed mismatch")
    require(int(root.attrib.get("failed", "-1")) == 0, f"{platform} XML failed mismatch")
    require(int(root.attrib.get("skipped", "0")) == 0, f"{platform} XML skipped mismatch")
    require(int(root.attrib.get("inconclusive", "0")) == 0, f"{platform} XML inconclusive mismatch")
    properties = {node.attrib.get("name"): node.attrib.get("value") for node in root.findall(".//property")}
    require(properties.get("platform") == platform or root.attrib.get("platform") == platform, f"{platform} XML platform mismatch")
    assemblies = [node for node in root.iter("test-suite") if node.attrib.get("type") == "Assembly" and node.attrib.get("name") == assembly_name]
    require(len(assemblies) == 1, f"{platform} XML assembly mismatch")
    fixtures = [
        node for node in root.iter("test-suite")
        if node.attrib.get("type") == "TestFixture"
        and (node.attrib.get("name") == fixture_name or node.attrib.get("fullname", "").endswith("." + fixture_name))
    ]
    require(len(fixtures) == 1, f"{platform} XML fixture mismatch")
    cases = list(fixtures[0].iter("test-case"))
    require(len(cases) == expected_total, f"{platform} XML fixture test-case count mismatch")
    require(all(case.attrib.get("result") == "Passed" for case in cases), f"{platform} XML contains non-Passed test case")
    return {
        "platform": platform,
        "assembly": assembly_name,
        "fixture": fixture_name,
        "result": "Passed",
        "total": expected_total,
        "passed": expected_total,
        "failed": 0,
        "skipped": 0,
        "inconclusive": 0,
        "start_time_utc": root.attrib.get("start-time"),
        "end_time_utc": root.attrib.get("end-time"),
        "duration_seconds": float(root.attrib.get("duration", "0")),
        "test_names": [case.attrib.get("fullname") or case.attrib.get("name") for case in cases],
    }


def validate_handoff(path: Path) -> dict:
    require(path.is_file(), f"missing editor evidence handoff: {path}")
    require(sha256_file(path) == HANDOFF_SHA256, "editor evidence handoff SHA-256 mismatch")
    with zipfile.ZipFile(path, "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        require(len(infos) == 5 and set(names) == EDITOR_FILES, "editor evidence handoff entry set mismatch")
        require(len(names) == len(set(names)), "editor evidence handoff duplicate entry")
        require(all(safe_zip_name(name) for name in names), "editor evidence handoff unsafe path")
        require(not [info for info in infos if stat.S_ISLNK((info.external_attr >> 16) & 0xFFFF)], "editor evidence handoff symlink")
        require(not [info for info in infos if info.flag_bits & 0x1], "editor evidence handoff encrypted entry")
        require(archive.testzip() is None, "editor evidence handoff CRC failure")
        payloads = {name: archive.read(name) for name in names}

    manifest = json.loads(payloads["editor_test_evidence_manifest.json"].decode("utf-8"))
    require(manifest.get("schema_version") == "step9c_editor_test_evidence_handoff_v1", "editor evidence manifest schema mismatch")
    rows = manifest.get("files")
    require(isinstance(rows, list) and len(rows) == 4, "editor evidence manifest record count mismatch")
    require(len({row.get("file_name") for row in rows}) == 4, "editor evidence manifest duplicate record")
    require({row.get("file_name") for row in rows} == EDITOR_FILES - {"editor_test_evidence_manifest.json"}, "editor evidence manifest target set mismatch")
    for row in rows:
        data = payloads[row["file_name"]]
        require(len(data) == row.get("size_bytes"), f"editor evidence size mismatch: {row['file_name']}")
        require(sha256_bytes(data) == row.get("sha256"), f"editor evidence SHA-256 mismatch: {row['file_name']}")

    edit_summary = parse_xml(
        payloads["Step9C_EditMode_21PASS.xml"], "EditMode", "WordsInSilence.Tests.EditMode.dll",
        "PhysicalWebcamCoordinateCompatibilityV1Tests", 21,
    )
    play_summary = parse_xml(
        payloads["Step9C_PlayMode_3PASS.xml"], "PlayMode", "WordsInSilence.Tests.PlayMode.dll",
        "PhysicalWebcamCoordinateCompatibilityHarnessV1Tests", 3,
    )
    row_map = {row["file_name"]: row for row in rows}
    require(row_map["Step9C_EditMode_21PASS.xml"].get("role") == "canonical_machine_readable_test_result", "EditMode XML role mismatch")
    require(row_map["Step9C_PlayMode_3PASS.xml"].get("role") == "canonical_machine_readable_test_result", "PlayMode XML role mismatch")
    require(row_map["Step9C_EditMode_21PASS_visual.png"].get("role") == "independent_visual_pass_confirmation", "EditMode visual role mismatch")
    require(row_map["Step9C_PlayMode_3PASS_visual.png"].get("role") == "matching_visual_pass_confirmation", "PlayMode visual role mismatch")
    require("do not claim it is the exact same execution" in row_map["Step9C_EditMode_21PASS_visual.png"].get("provenance_note", ""), "EditMode visual provenance note missing")
    return {
        "path": path,
        "record": record(path),
        "payloads": payloads,
        "manifest": manifest,
        "manifest_rows": row_map,
        "edit_summary": edit_summary,
        "play_summary": play_summary,
    }


def validate_fixed_hashes() -> list[dict]:
    rows = []
    for relative_path, expected_hash in APPROVED_FIXED_HASHES.items():
        path = ROOT / relative_path
        require(path.is_file(), f"missing approved fixed file: {relative_path}")
        actual_hash = sha256_file(path)
        require(actual_hash == expected_hash, f"approved fixed SHA-256 mismatch: {relative_path}; actual={actual_hash}")
        rows.append(record(path, category="approved_fixed"))
    return rows


def validate_evidence(verifier) -> dict:
    json_files = sorted(PHYSICAL.glob("*.json"), key=lambda path: path.name)
    checksum_files = sorted(PHYSICAL.glob("*.json.sha256"), key=lambda path: path.name)
    all_files = sorted(path for path in PHYSICAL.iterdir() if path.is_file())
    require(len(json_files) == 20 and len(checksum_files) == 20 and len(all_files) == 40, "physical evidence count mismatch")
    results = []
    valid_captures = []
    for path in json_files:
        verification = verifier.verify_evidence(path)
        require(verification.get("status") == "PASS", f"physical evidence verifier failure: {path.name}: {verification.get('errors')}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        require(payload.get("run_id") == RUN_ID, f"physical evidence run ID mismatch: {path.name}")
        verification["pre_export_objectively_valid"] = payload.get("pre_export_objectively_valid")
        verification["finalization_state"] = payload.get("finalization_state")
        verification["prediction"] = payload.get("recognition")
        results.append(verification)
        if payload.get("pre_export_objectively_valid") is True:
            valid_captures.append({
                "take_number": payload.get("take_number"),
                "expected_word": payload.get("expected_word"),
                "capture_id": payload.get("capture_id"),
                "json_file": path.name,
            })
    valid_captures.sort(key=lambda row: row["take_number"])
    require([row["expected_word"] for row in valid_captures] == ["GREETING", "LOVE", "THANK_YOU"], "verified fixed capture order mismatch")
    require(len({row["capture_id"] for row in valid_captures}) == 3, "verified fixed capture IDs are not unique")
    new_path = PHYSICAL / "take_03_THANK_YOU_step9c-take-03-thank-you-53870472-c644-489c-a94d-95e9667ded42.json"
    require(sha256_file(new_path) == NEW_THANK_YOU_SHA256, "approved THANK_YOU evidence hash mismatch")
    new_payload = json.loads(new_path.read_text(encoding="utf-8"))
    require(new_payload.get("pre_export_objectively_valid") is True, "approved THANK_YOU objective validity mismatch")
    require(new_payload.get("recognition", {}).get("accepted") is False, "approved THANK_YOU accepted=false was not preserved")
    return {
        "json_files": json_files,
        "checksum_files": checksum_files,
        "all_files": all_files,
        "results": results,
        "valid_captures": valid_captures,
        "new_thank_you": {
            "relative_path": new_path.relative_to(ROOT).as_posix(),
            "size_bytes": new_path.stat().st_size,
            "sha256": sha256_file(new_path),
            "accepted": False,
        },
    }


def validate_step9b_parity() -> dict:
    input_manifest = json.loads(STEP9B_INPUT.read_text(encoding="utf-8"))
    output_manifest = json.loads(STEP9B_OUTPUT.read_text(encoding="utf-8"))
    rows = [
        row for row in output_manifest["outputs"]
        if row["relative_path"].startswith("Assets/_Project/Scripts/Recognition/LiveInput/")
        or row["relative_path"].startswith("Assets/_Project/Tests/EditMode/Fixtures/LiveMediaPipeInputV1/")
        or row["relative_path"].startswith("Assets/StreamingAssets/RecognitionCoreV1/LiveInput/")
    ]
    rows.extend(
        row for row in input_manifest["inputs"]
        if row["relative_path"] in {
            "Assets/_Project/Scripts/Landmarks/MediaPipeLandmarkProvider.cs",
            "Assets/_Project/Scripts/Pose/HandPoseIntegrator.cs",
        }
    )
    require(len(rows) == 20, f"Step 9B parity target count mismatch: {len(rows)}")
    mismatches = []
    for row in rows:
        path = ROOT / row["relative_path"]
        if not path.is_file() or path.stat().st_size != row["size_bytes"] or sha256_file(path) != row["sha256"]:
            mismatches.append(row["relative_path"])
    require(not mismatches, f"Step 9B parity mismatch: {mismatches}")
    return {
        "checked": len(rows),
        "passed": len(rows),
        "failed": 0,
        "targets": rows,
        "input_manifest": record(STEP9B_INPUT),
        "output_manifest": record(STEP9B_OUTPUT),
    }


def fixed_allowlist_paths() -> list[tuple[Path, str]]:
    paths: list[tuple[Path, str]] = []

    def add_with_meta(relative_path: str, category: str) -> None:
        path = ROOT / relative_path
        meta = Path(str(path) + ".meta")
        require(path.is_file(), f"allowlist file missing: {relative_path}")
        require(meta.is_file(), f"allowlist meta missing: {meta.relative_to(ROOT).as_posix()}")
        paths.extend([(path, category), (meta, category + "_meta")])

    for relative_path in PRODUCTION_REL:
        add_with_meta(relative_path, "step9c_production")
    for relative_path in RELATED_APPROVED_REL:
        add_with_meta(relative_path, "approved_related_source")
    add_with_meta(PREFAB_REL, "step9c_diagnostic_prefab")
    add_with_meta(EDIT_TEST_REL, "step9c_editmode_test")
    add_with_meta(PLAY_TEST_REL, "step9c_playmode_test")
    add_with_meta(CAPTURE_PLAN_REL, "step9c_capture_plan")
    for relative_path in FIXTURE_REL:
        path = ROOT / relative_path
        require(path.is_file(), f"allowlist fixture missing: {relative_path}")
        paths.append((path, "step9c_fixture"))
    paths.extend([(VERIFIER, "step9c_existing_verifier"), (PACKAGER, "step9c_packager")])
    return paths


def protected_paths(allowlist: list[tuple[Path, str]], parity: dict, evidence: dict) -> list[Path]:
    paths = {path.resolve() for path, _ in allowlist}
    paths.add((ROOT / SCENE_REL).resolve())
    paths.add(STEP9B_INPUT.resolve())
    paths.add(STEP9B_OUTPUT.resolve())
    paths.update((ROOT / row["relative_path"]).resolve() for row in parity["targets"])
    paths.update(path.resolve() for path in evidence["all_files"])
    return sorted(paths, key=lambda path: path.as_posix())


def snapshot(paths: list[Path]) -> dict[str, dict]:
    result = {}
    for path in paths:
        require(path.is_file(), f"protected file missing: {path}")
        relative_path = path.relative_to(ROOT).as_posix()
        result[relative_path] = record(path)
    return result


def compare_snapshot(before: dict[str, dict]) -> list[dict]:
    mismatches = []
    for relative_path, expected in before.items():
        path = ROOT / relative_path
        actual = record(path) if path.is_file() else {"relative_path": relative_path, "size_bytes": None, "sha256": None}
        if actual["size_bytes"] != expected["size_bytes"] or actual["sha256"] != expected["sha256"]:
            mismatches.append({"relative_path": relative_path, "before": expected, "after": actual})
    return mismatches


def stage_write(staging: Path, relative_path: str, data: bytes) -> Path:
    path = staging / PurePosixPath(relative_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def assert_final_outputs_absent() -> None:
    final_paths = [DELIVERY_ZIP, DELIVERY_HASH, DELIVERY_SIDECAR, OUTPUT / "output_file_manifest.json", OUTPUT / "editor_test_evidence"]
    final_paths.extend(ROOT / relative_path for relative_path in GENERATED_REL)
    existing = [path.as_posix() for path in final_paths if path.exists()]
    require(not existing, f"final output already exists; refusing overwrite: {existing}")


def add_check(checks: list[dict], name: str, passed: bool, evidence_value=None) -> None:
    checks.append({"name": name, "status": "PASS" if passed else "FAIL", "evidence": evidence_value})


def build_package(handoff_path: Path) -> dict:
    require((ROOT / "Packages/manifest.json").is_file(), "not a Unity project: Packages/manifest.json missing")
    require((ROOT / "Packages/packages-lock.json").is_file(), "not a Unity project: Packages/packages-lock.json missing")
    require((ROOT / "ProjectSettings/ProjectVersion.txt").is_file(), "not a Unity project: ProjectVersion.txt missing")
    require("m_EditorVersion: 6000.3.14f1" in (ROOT / "ProjectSettings/ProjectVersion.txt").read_text(encoding="utf-8"), "Unity project version mismatch")
    assert_final_outputs_absent()
    approved_fixed = validate_fixed_hashes()
    handoff = validate_handoff(handoff_path)
    verifier = load_existing_verifier()
    evidence = validate_evidence(verifier)
    parity = validate_step9b_parity()
    allowlist = fixed_allowlist_paths()
    protected_before = snapshot(protected_paths(allowlist, parity, evidence))

    operation_id = uuid.uuid4().hex
    staging = ROOT / f".s9c_{operation_id[:8]}.tmp"
    temp_zip = ROOT / f".{DELIVERY_NAME}.{operation_id}.tmp"
    temp_hash = ROOT / f".{DELIVERY_NAME}.sha256.{operation_id}.tmp"
    temp_sidecar = ROOT / f".{DELIVERY_SIDECAR.name}.{operation_id}.tmp"
    committed: list[Path] = []
    success = False

    try:
        staging.mkdir(parents=False, exist_ok=False)
        staged_editor: dict[str, Path] = {}
        for name, data in handoff["payloads"].items():
            relative_path = f"{EDITOR_EVIDENCE_REL}/{name}"
            staged_editor[name] = stage_write(staging, relative_path, data)

        edit_manifest_row = handoff["manifest_rows"]["Step9C_EditMode_21PASS.xml"]
        edit_visual_row = handoff["manifest_rows"]["Step9C_EditMode_21PASS_visual.png"]
        play_manifest_row = handoff["manifest_rows"]["Step9C_PlayMode_3PASS.xml"]
        play_visual_row = handoff["manifest_rows"]["Step9C_PlayMode_3PASS_visual.png"]

        edit_summary = {
            "schema_version": "words_in_silence.step9c.editor_test_summary.v1",
            "run_id": RUN_ID,
            **handoff["edit_summary"],
            "xml": edit_manifest_row,
            "visual": edit_visual_row,
            "visual_same_execution_as_xml": False,
            "provenance_note": edit_visual_row["provenance_note"],
            "unity_reexecuted_by_packager": False,
        }
        play_summary = {
            "schema_version": "words_in_silence.step9c.editor_test_summary.v1",
            "run_id": RUN_ID,
            **handoff["play_summary"],
            "xml": play_manifest_row,
            "visual": play_visual_row,
            "visual_same_execution_as_xml": True,
            "provenance_note": play_visual_row["provenance_note"],
            "unity_reexecuted_by_packager": False,
        }

        input_manifest = {
            "schema_version": "words_in_silence.step9c.delivery_input_manifest.v1",
            "run_id": RUN_ID,
            "status": "STEP9C_DELIVERY_PACKAGE_READY",
            "packaging_only": True,
            "unity_reexecuted": False,
            "physical_capture_executed": False,
            "editor_evidence_handoff": handoff["record"],
            "editor_evidence_manifest": handoff["manifest"],
            "approved_fixed_files": approved_fixed,
            "step9b_parity": parity,
            "physical_evidence": {
                "json_count": len(evidence["json_files"]),
                "checksum_count": len(evidence["checksum_files"]),
                "strict_verified": len(evidence["results"]),
                "valid_fixed_captures": evidence["valid_captures"],
                "new_thank_you": evidence["new_thank_you"],
            },
            "protected_files_before": list(protected_before.values()),
        }

        pre_checks: list[dict] = []
        add_check(pre_checks, "unity_project_root", True, ROOT.as_posix())
        add_check(pre_checks, "existing_verifier_sha256", True, VERIFIER_SHA256)
        add_check(pre_checks, "approved_test_source_sha256", True, {"editmode": EDIT_TEST_SHA256, "playmode": PLAY_TEST_SHA256})
        add_check(pre_checks, "approved_fixed_hashes", True, len(approved_fixed))
        add_check(pre_checks, "editor_handoff_strict", True, handoff["record"])
        add_check(pre_checks, "editmode_xml_21_of_21", True, handoff["edit_summary"])
        add_check(pre_checks, "playmode_xml_3_of_3", True, handoff["play_summary"])
        add_check(pre_checks, "editmode_visual_separate_execution_documented", True, edit_visual_row["provenance_note"])
        add_check(pre_checks, "playmode_visual_matching_execution_documented", True, play_visual_row["provenance_note"])
        add_check(pre_checks, "physical_evidence_20_pairs", True, {"json": 20, "sha256": 20})
        add_check(pre_checks, "physical_evidence_strict_verifier", True, len(evidence["results"]))
        add_check(pre_checks, "approved_thank_you_accepted_false_preserved", True, evidence["new_thank_you"])
        add_check(pre_checks, "step9b_parity_20_of_20", True, {"passed": 20, "failed": 0})
        add_check(pre_checks, "protected_snapshot_created", True, len(protected_before))
        add_check(pre_checks, "packaging_allowlist_only", True, len(allowlist) + len(evidence["all_files"]))
        names = [row["name"] for row in pre_checks]
        require(len(names) == len(set(names)), "pre-package assertion name duplicate")

        protected_pre_zip_mismatches = compare_snapshot(protected_before)
        add_check(pre_checks, "protected_files_unchanged_before_zip", not protected_pre_zip_mismatches, protected_pre_zip_mismatches)
        require(not protected_pre_zip_mismatches, "protected file changed before ZIP creation")
        names = [row["name"] for row in pre_checks]
        require(len(names) == len(set(names)), "pre-package assertion name duplicate")
        assertion_payload = {
            "schema_version": "words_in_silence.step9c.delivery_assertions.v1",
            "run_id": RUN_ID,
            "summary": {
                "total": len(pre_checks),
                "passed": sum(row["status"] == "PASS" for row in pre_checks),
                "failed": sum(row["status"] != "PASS" for row in pre_checks),
                "duplicate_names": len(names) - len(set(names)),
            },
            "assertions": pre_checks,
        }
        require(assertion_payload["summary"]["failed"] == 0, "pre-package assertion failure")

        run_state = {
            "schema_version": "words_in_silence.step9c.delivery_run_state.v1",
            "run_id": RUN_ID,
            "status": "STEP9C_DELIVERY_PACKAGE_CREATED_PENDING_EXTERNAL_REVIEW",
            "delivery_package_created": True,
            "unity_tests_reexecuted": False,
            "physical_capture_executed": False,
            "physical_evidence_json_count": 20,
            "physical_evidence_checksum_count": 20,
            "verified_fixed_capture_order": ["GREETING", "LOVE", "THANK_YOU"],
            "thank_you_prediction_accepted": False,
        }

        report_text = f"""# Step 9C Physical Webcam Coordinate Compatibility v1.0 Delivery\n\n**Run ID**: {RUN_ID}\n**Status**: STEP9C_DELIVERY_PACKAGE_CREATED_PENDING_EXTERNAL_REVIEW\n\n## Approved Editor tests\n\n- EditMode XML: 21 PASS / 0 FAIL.\n- EditMode visual: independent pass corroboration approximately two minutes earlier; it is not claimed as the exact XML execution.\n- PlayMode XML and visual: 3 PASS / 0 FAIL from the matching execution interval.\n- The packager did not execute Unity.\n\n## Physical evidence\n\n- 20 JSON + 20 detached SHA-256 files passed the existing Step 9C verifier.\n- Verified fixed capture recovery order: GREETING -> LOVE -> THANK_YOU.\n- The approved THANK_YOU capture remains objectively valid with prediction THANK_YOU and accepted=false.\n- No physical capture was executed by the packager.\n\n## Approved dependencies\n\n- Step 9B parity: 20/20.\n- Approved production, tests, prefab, scene, evidence and dependency files were snapshotted for before/after verification.\n- The archive is built from an explicit allowlist and excludes project caches, unrelated assets and prior handoff/delivery archives.\n\n`STEP9C_DELIVERY_PACKAGE_CREATED_PENDING_EXTERNAL_REVIEW`\n"""

        generated_content = {
            GENERATED_REL[0]: json_bytes(edit_summary),
            GENERATED_REL[1]: json_bytes(play_summary),
            GENERATED_REL[2]: json_bytes(input_manifest),
            GENERATED_REL[3]: json_bytes(assertion_payload),
            GENERATED_REL[4]: json_bytes(run_state),
            GENERATED_REL[5]: report_text.encode("utf-8"),
        }
        staged_generated = {relative_path: stage_write(staging, relative_path, data) for relative_path, data in generated_content.items()}

        entries: dict[str, dict] = {}

        def add_entry(source: Path, archive_path: str, category: str) -> None:
            require(safe_zip_name(archive_path), f"unsafe allowlist archive path: {archive_path}")
            require(source.is_file() and not source.is_symlink(), f"invalid allowlist source: {source}")
            require(archive_path not in entries, f"duplicate allowlist archive path: {archive_path}")
            entries[archive_path] = {"source": source, "category": category}

        for path, category in allowlist:
            add_entry(path, path.relative_to(ROOT).as_posix(), category)
        for path in evidence["json_files"]:
            add_entry(path, path.relative_to(ROOT).as_posix(), "physical_evidence_json")
        for path in evidence["checksum_files"]:
            add_entry(path, path.relative_to(ROOT).as_posix(), "physical_evidence_sha256")
        for name, path in staged_editor.items():
            add_entry(path, f"{EDITOR_EVIDENCE_REL}/{name}", "editor_test_evidence")
        for relative_path, path in staged_generated.items():
            add_entry(path, relative_path, "delivery_generated_output")

        output_records = []
        for archive_path in sorted(entries):
            item = entries[archive_path]
            output_records.append(record(item["source"], archive_path, item["category"]))
        output_manifest = {
            "version": "1.0",
            "schema_version": "self_excluding_v1",
            "run_id": RUN_ID,
            "self_reference_excluded": True,
            "outputs": output_records,
            "total_files": len(output_records),
        }
        staged_manifest = stage_write(staging, OUTPUT_MANIFEST_REL, json_bytes(output_manifest))

        with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for archive_path in sorted(entries):
                archive.write(entries[archive_path]["source"], archive_path)
            archive.write(staged_manifest, OUTPUT_MANIFEST_REL)

        zip_hash = sha256_file(temp_zip)
        detached_line = f"{zip_hash}  {DELIVERY_NAME}\n"
        temp_hash.write_text(detached_line, encoding="utf-8", newline="\n")

        post_checks: list[dict] = []

        def post(name: str, passed: bool, evidence_value=None) -> None:
            post_checks.append({"check_name": name, "status": "PASS" if passed else "FAIL", "evidence": evidence_value})

        manifest_missing: list[str] = []
        manifest_extra: list[str] = []
        size_mismatches: list[str] = []
        sha_mismatches: list[str] = []
        unsafe_paths: list[str] = []
        symlinks: list[str] = []
        encrypted: list[str] = []
        editor_mismatches: list[str] = []
        physical_mismatches: list[str] = []

        with zipfile.ZipFile(temp_zip, "r") as archive:
            infos = archive.infolist()
            names_in_zip = [info.filename for info in infos]
            bad_crc = archive.testzip()
            duplicates = sorted(name for name, count in Counter(names_in_zip).items() if count > 1)
            for info in infos:
                if not safe_zip_name(info.filename):
                    unsafe_paths.append(info.filename)
                if stat.S_ISLNK((info.external_attr >> 16) & 0xFFFF):
                    symlinks.append(info.filename)
                if info.flag_bits & 0x1:
                    encrypted.append(info.filename)
                data = archive.read(info)
                post(f"entry_read::{info.filename}", len(data) == info.file_size, info.filename)

            manifest_inside = json.loads(archive.read(OUTPUT_MANIFEST_REL).decode("utf-8"))
            manifest_rows = manifest_inside.get("outputs", [])
            record_names = [row.get("relative_path") for row in manifest_rows]
            record_duplicates = sorted(name for name, count in Counter(record_names).items() if count > 1)
            expected = {row["relative_path"]: row for row in manifest_rows}
            actual_targets = set(names_in_zip) - {OUTPUT_MANIFEST_REL}
            manifest_missing = sorted(set(expected) - actual_targets)
            manifest_extra = sorted(actual_targets - set(expected))
            for archive_path, row in expected.items():
                data = archive.read(archive_path)
                if len(data) != row["size_bytes"]:
                    size_mismatches.append(archive_path)
                if sha256_bytes(data) != row["sha256"]:
                    sha_mismatches.append(archive_path)

            for name, expected_bytes in handoff["payloads"].items():
                archive_path = f"{EDITOR_EVIDENCE_REL}/{name}"
                if archive.read(archive_path) != expected_bytes:
                    editor_mismatches.append(archive_path)
            parse_xml(
                archive.read(f"{EDITOR_EVIDENCE_REL}/Step9C_EditMode_21PASS.xml"), "EditMode",
                "WordsInSilence.Tests.EditMode.dll", "PhysicalWebcamCoordinateCompatibilityV1Tests", 21,
            )
            parse_xml(
                archive.read(f"{EDITOR_EVIDENCE_REL}/Step9C_PlayMode_3PASS.xml"), "PlayMode",
                "WordsInSilence.Tests.PlayMode.dll", "PhysicalWebcamCoordinateCompatibilityHarnessV1Tests", 3,
            )
            for path in evidence["all_files"]:
                archive_path = path.relative_to(ROOT).as_posix()
                data = archive.read(archive_path)
                if len(data) != path.stat().st_size or sha256_bytes(data) != sha256_file(path):
                    physical_mismatches.append(archive_path)

            image_entries = sorted(name for name in names_in_zip if PurePosixPath(name).suffix.lower() in {".png", ".jpg", ".jpeg"})
            approved_images = sorted([
                f"{EDITOR_EVIDENCE_REL}/Step9C_EditMode_21PASS_visual.png",
                f"{EDITOR_EVIDENCE_REL}/Step9C_PlayMode_3PASS_visual.png",
            ])
            forbidden_prefixes = ("Library/", "Temp/", "Logs/", "Obj/", "obj/", "UserSettings/", ".git/", "Packages/")
            forbidden_entries = [name for name in names_in_zip if name.startswith(forbidden_prefixes) or "/PackageCache/" in name]
            old_archives = [name for name in names_in_zip if name.endswith(".zip") or "handoff" in name.lower()]
            new_evidence_path = evidence["new_thank_you"]["relative_path"]
            new_payload = json.loads(archive.read(new_evidence_path).decode("utf-8"))

            post("archive_crc", bad_crc is None, bad_crc)
            post("archive_duplicate_entries", not duplicates, duplicates)
            post("archive_safe_paths", not unsafe_paths, unsafe_paths)
            post("archive_no_symlinks", not symlinks, symlinks)
            post("archive_not_encrypted", not encrypted, encrypted)
            post("manifest_present", manifest_inside.get("schema_version") == "self_excluding_v1", None)
            post("manifest_self_excluding", manifest_inside.get("self_reference_excluded") is True and OUTPUT_MANIFEST_REL not in expected, None)
            post("manifest_target_count", manifest_inside.get("total_files") == len(output_records) == len(manifest_rows), len(manifest_rows))
            post("manifest_record_duplicates", not record_duplicates, record_duplicates)
            post("manifest_missing", not manifest_missing, manifest_missing)
            post("manifest_extra", not manifest_extra, manifest_extra)
            post("manifest_size", not size_mismatches, size_mismatches)
            post("manifest_sha256", not sha_mismatches, sha_mismatches)
            post("editor_evidence_exact", not editor_mismatches, editor_mismatches)
            post("editor_xml_strict", True, {"editmode": "21/21/0", "playmode": "3/3/0"})
            post("physical_evidence_20_pairs", len(evidence["json_files"]) == 20 and len(evidence["checksum_files"]) == 20, None)
            post("physical_evidence_archive_exact", not physical_mismatches, physical_mismatches)
            post("approved_images_only", image_entries == approved_images, image_entries)
            post("no_project_cache_or_packages", not forbidden_entries, forbidden_entries)
            post("no_prior_archive_or_handoff", not old_archives, old_archives)
            post("thank_you_accepted_false_preserved", new_payload.get("recognition", {}).get("accepted") is False, None)

        protected_after_mismatches = compare_snapshot(protected_before)
        post("protected_files_unchanged", not protected_after_mismatches, protected_after_mismatches)
        post("step9b_parity_20_of_20", validate_step9b_parity()["failed"] == 0, None)
        post("detached_hash_strict", temp_hash.read_text(encoding="utf-8") == detached_line, None)
        post("packager_source_in_archive", sha256_file(PACKAGER) == next(row["sha256"] for row in output_records if row["relative_path"] == PACKAGER.relative_to(ROOT).as_posix()), None)
        post("approved_test_sources_in_archive", all(
            next(row["sha256"] for row in output_records if row["relative_path"] == relative_path) == expected
            for relative_path, expected in [(EDIT_TEST_REL, EDIT_TEST_SHA256), (PLAY_TEST_REL, PLAY_TEST_SHA256)]
        ), None)

        check_names = [row["check_name"] for row in post_checks]
        duplicate_check_names = sorted(name for name, count in Counter(check_names).items() if count > 1)
        failures = [row for row in post_checks if row["status"] != "PASS"]
        sidecar = {
            "version": "1.0",
            "schema_version": "words_in_silence.step9c.post_zip_verification.v1",
            "run_id": RUN_ID,
            "zip_file": DELIVERY_NAME,
            "zip_size_bytes": temp_zip.stat().st_size,
            "zip_sha256": zip_hash,
            "zip_entry_count": len(output_records) + 1,
            "zip_unique_entry_count": len(output_records) + 1,
            "output_manifest_target_count": len(output_records),
            "output_manifest_self_reference_excluded": True,
            "archive_crc_error_count": 0 if not any(row["check_name"] == "archive_crc" and row["status"] != "PASS" for row in post_checks) else 1,
            "duplicate_entry_count": len(duplicates),
            "unsafe_path_count": len(unsafe_paths),
            "symlink_count": len(symlinks),
            "encrypted_entry_count": len(encrypted),
            "manifest_missing_count": len(manifest_missing),
            "manifest_extra_count": len(manifest_extra),
            "manifest_record_duplicate_count": len(record_duplicates),
            "size_mismatch_count": len(size_mismatches),
            "sha256_mismatch_count": len(sha_mismatches),
            "editor_evidence_mismatch_count": len(editor_mismatches),
            "physical_evidence_json_count": len(evidence["json_files"]),
            "physical_evidence_checksum_count": len(evidence["checksum_files"]),
            "physical_evidence_mismatch_count": len(physical_mismatches),
            "step9b_parity_passed": parity["passed"],
            "step9b_parity_failed": parity["failed"],
            "protected_file_count": len(protected_before),
            "protected_file_mismatch_count": len(protected_after_mismatches),
            "detached_hash_strict_parse": temp_hash.read_text(encoding="utf-8") == detached_line,
            "post_zip_check_total": len(post_checks),
            "post_zip_check_unique": len(set(check_names)),
            "post_zip_check_name_duplicates": len(duplicate_check_names),
            "post_zip_failure_count": len(failures),
            "duplicate_check_names": duplicate_check_names,
            "verification_status": "PASS" if not failures and not duplicate_check_names else "FAIL",
            "checks": post_checks,
        }
        temp_sidecar.write_bytes(json_bytes(sidecar))
        require(sidecar["verification_status"] == "PASS", f"post-ZIP verification failed: {[row['check_name'] for row in failures]}")

        assert_final_outputs_absent()
        final_editor_dir = ROOT / EDITOR_EVIDENCE_REL
        staged_editor_dir = staging / PurePosixPath(EDITOR_EVIDENCE_REL)
        final_editor_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged_editor_dir, final_editor_dir)
        committed.append(final_editor_dir)
        for relative_path in GENERATED_REL:
            final_path = ROOT / relative_path
            final_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging / PurePosixPath(relative_path), final_path)
            committed.append(final_path)
        final_manifest_path = ROOT / OUTPUT_MANIFEST_REL
        os.replace(staged_manifest, final_manifest_path)
        committed.append(final_manifest_path)
        os.replace(temp_hash, DELIVERY_HASH)
        committed.append(DELIVERY_HASH)
        os.replace(temp_sidecar, DELIVERY_SIDECAR)
        committed.append(DELIVERY_SIDECAR)
        os.replace(temp_zip, DELIVERY_ZIP)
        committed.append(DELIVERY_ZIP)
        success = True
        return {
            "status": "STEP9C_DELIVERY_PACKAGE_CREATED_PENDING_EXTERNAL_REVIEW",
            "packager": record(PACKAGER),
            "zip": record(DELIVERY_ZIP),
            "detached_sha256": record(DELIVERY_HASH),
            "post_zip_verification": record(DELIVERY_SIDECAR),
            "output_manifest": record(final_manifest_path),
            "delivery_report": record(ROOT / GENERATED_REL[5]),
            "manifest_target_count": len(output_records),
            "zip_entry_count": len(output_records) + 1,
            "physical_evidence": {"json": 20, "sha256": 20},
            "step9b_parity": "20/20",
            "protected_files": {"checked": len(protected_before), "mismatches": 0},
            "editor_evidence": [
                {"file_name": name, "size_bytes": len(data), "sha256": sha256_bytes(data)}
                for name, data in sorted(handoff["payloads"].items())
            ],
            "unity_executed": False,
            "physical_capture_executed": False,
        }
    except Exception:
        if not success:
            for path in reversed(committed):
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                elif path.exists():
                    path.unlink()
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        for path in (temp_zip, temp_hash, temp_sidecar):
            if path.exists():
                path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the approved Step 9C Delivery package without running Unity or a camera.")
    parser.add_argument("--evidence-handoff", type=Path, required=True, help="Path to Step9C_Editor_Test_Evidence_Handoff.zip")
    args = parser.parse_args()
    handoff_path = args.evidence_handoff
    if not handoff_path.is_absolute():
        handoff_path = (ROOT / handoff_path).resolve()
    result = build_package(handoff_path.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
