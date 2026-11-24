#!/usr/bin/env python3
"""Prepare, verify, and package WordsInSilence Step 9B v1.0 evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import stat
import struct
import uuid
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "data/processed/sessions/S001_P001/step9b_live_mediapipe_input_v1_0"
SOURCE_ROOT = ROOT / "Assets/_Project/Scripts/Recognition/LiveInput"
FIXTURE_ROOT = ROOT / "Assets/_Project/Tests/EditMode/Fixtures/LiveMediaPipeInputV1"
RAW_FIXTURE = FIXTURE_ROOT / "approved_GREETING_NORMAL_R01.raw9b"
QUERY_FIXTURE = FIXTURE_ROOT / "query_GREETING_NORMAL_R01.wisrq"
FIXTURE_MANIFEST = FIXTURE_ROOT / "parity_fixture_manifest.json"
CONFIG_ROOT = ROOT / "Assets/StreamingAssets/RecognitionCoreV1/LiveInput"
CONFIG_PATH = CONFIG_ROOT / "live_mediapipe_input_config.json"
PREFAB = ROOT / "Assets/_Project/Resources/Diagnostics/LiveMediaPipeInputDiagnostic.prefab"
EDIT_TEST = ROOT / "Assets/_Project/Tests/EditMode/LiveMediaPipeInputV1Tests.cs"
PLAY_TEST = ROOT / "Assets/_Project/Tests/PlayMode/LiveMediaPipeDiagnosticHarnessV1Tests.cs"
STATE_PATH = OUTPUT / "run_state.json"

STEP8B_ZIP = ROOT / "WordsInSilence_step8b_recognition_core_v1_0_delivery.zip"
STEP8C_ZIP = ROOT / "WordsInSilence_step8c_offline_recognition_evaluation_v1_0_delivery.zip"
STEP9A_ZIP = ROOT / "WordsInSilence_step9a_runtime_recognition_service_v1_0_delivery.zip"
APPROVED_DELIVERIES = {
    STEP8B_ZIP: (661_831, "3401ea74f73902c5ea7b4f34abb06ebfe933ef0a076820f15b6471937a5c7c6d"),
    STEP8C_ZIP: (445_505, "e1cf9d7f197feba99cc0499795f3f04702cd110bbe6d3200341a7a0aa2d55900"),
    STEP9A_ZIP: (415_867, "497fb4bedc20b8439e8e00e8bf56cb5f4f199cdc2143da5f366e3a90fd069148"),
}

STEP9A = ROOT / "data/processed/sessions/S001_P001/step9a_runtime_recognition_service_v1_0"
STEP7B = ROOT / "data/processed/sessions/S001_P001/step7b_full_v1_0"
STEP8A = ROOT / "data/processed/sessions/S001_P001/step8a_dtw_templates_v1_0"
PARITY_SEQUENCE = STEP7B / "sequences/full_GREETING_NORMAL_R01.json"
PARITY_G1 = STEP7B / "npz/interp_GREETING_NORMAL_R01_g1.npz"
PARITY_FEATURE = STEP8A / "features/feature_GREETING_NORMAL_R01.npz"
SOURCE_QUERY = ROOT / "Assets/_Project/Tests/EditMode/Fixtures/RuntimeRecognitionServiceV1/TestQueries/query_GREETING_NORMAL_R01.wisrq"

DELIVERY_NAME = "WordsInSilence_step9b_live_mediapipe_input_v1_0_delivery.zip"
DELIVERY_ZIP = ROOT / DELIVERY_NAME
DELIVERY_HASH = ROOT / f"{DELIVERY_NAME}.sha256"
DELIVERY_SIDECAR = ROOT / "WordsInSilence_step9b_live_mediapipe_input_v1_0_delivery_post_zip_verification.json"
VERSION = "1.0"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def record(path: Path) -> dict:
    info = path.stat()
    return {
        "relative_path": path.relative_to(ROOT).as_posix(),
        "size_bytes": info.st_size,
        "sha256": sha256_file(path),
        "mtime_ns": info.st_mtime_ns,
    }


def add(checks: list[dict], name: str, passed: bool, evidence=None) -> None:
    checks.append({"name": name, "status": "PASS" if passed else "FAIL", "evidence": evidence})


def assertion_payload(run_id: str, checks: list[dict]) -> dict:
    names = [item["name"] for item in checks]
    duplicate_names = sorted(name for name, count in Counter(names).items() if count > 1)
    failures = [item for item in checks if item["status"] != "PASS"]
    return {
        "version": VERSION,
        "run_id": run_id,
        "total": len(checks),
        "pass": len(checks) - len(failures),
        "fail": len(failures),
        "duplicate_count": len(duplicate_names),
        "duplicate_names": duplicate_names,
        "results": checks,
    }


def ensure_pass(payload: dict, label: str) -> None:
    if payload["fail"] or payload["duplicate_count"]:
        failed = [item["name"] for item in payload["results"] if item["status"] != "PASS"]
        raise RuntimeError(f"{label} failed: {failed}")


def deterministic_guid(path: Path) -> str:
    return hashlib.sha256(("WordsInSilence-Step9B:" + path.relative_to(ROOT).as_posix()).encode("utf-8")).hexdigest()[:32]


def ensure_meta(path: Path, folder: bool = False) -> Path:
    meta = Path(str(path) + ".meta")
    if meta.exists():
        return meta
    guid = deterministic_guid(path)
    if folder:
        text = f"fileFormatVersion: 2\nguid: {guid}\nfolderAsset: yes\nDefaultImporter:\n  externalObjects: {{}}\n  userData: \n  assetBundleName: \n  assetBundleVariant: \n"
    elif path.suffix == ".cs":
        text = f"fileFormatVersion: 2\nguid: {guid}\nMonoImporter:\n  externalObjects: {{}}\n  serializedVersion: 2\n  defaultReferences: []\n  executionOrder: 0\n  icon: {{instanceID: 0}}\n  userData: \n  assetBundleName: \n  assetBundleVariant: \n"
    elif path.suffix == ".prefab":
        text = f"fileFormatVersion: 2\nguid: {guid}\nPrefabImporter:\n  externalObjects: {{}}\n  userData: \n  assetBundleName: \n  assetBundleVariant: \n"
    else:
        text = f"fileFormatVersion: 2\nguid: {guid}\nDefaultImporter:\n  externalObjects: {{}}\n  userData: \n  assetBundleName: \n  assetBundleVariant: \n"
    meta.write_text(text, encoding="utf-8")
    return meta


def delivery_companions(delivery: Path) -> tuple[Path, Path]:
    detached = ROOT / f"{delivery.name}.sha256"
    stem = delivery.name.removesuffix("_delivery.zip")
    sidecar = ROOT / f"{stem}_delivery_post_zip_verification.json"
    return detached, sidecar


def verify_delivery(delivery: Path, expected_size: int, expected_hash: str, checks: list[dict]) -> None:
    detached, sidecar = delivery_companions(delivery)
    prefix = delivery.name.split("_")[1]
    add(checks, f"{prefix}_delivery_exists", delivery.is_file(), delivery.as_posix())
    if not delivery.is_file():
        return
    actual_hash = sha256_file(delivery)
    add(checks, f"{prefix}_delivery_size", delivery.stat().st_size == expected_size, delivery.stat().st_size)
    add(checks, f"{prefix}_delivery_sha256", actual_hash == expected_hash, actual_hash)
    expected_line = f"{expected_hash}  {delivery.name}"
    add(checks, f"{prefix}_detached_strict", detached.is_file() and detached.read_text(encoding="utf-8").strip() == expected_line, None)
    side = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else {}
    add(checks, f"{prefix}_sidecar_pass", side.get("verification_status") == "PASS" and side.get("zip_sha256") == expected_hash, side.get("verification_status"))


def load_records(path: Path, field: str) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload[field]


def preflight(checks: list[dict]) -> list[dict]:
    for path, expected in APPROVED_DELIVERIES.items():
        verify_delivery(path, expected[0], expected[1], checks)

    approved: dict[str, dict] = {}
    for item in load_records(STEP9A / "input_manifest.json", "inputs"):
        approved[item["relative_path"]] = item
    for item in load_records(STEP9A / "output_file_manifest.json", "outputs"):
        approved[item["relative_path"]] = item
    for path in [STEP9A / "output_file_manifest.json", STEP9A_ZIP, *delivery_companions(STEP9A_ZIP)]:
        approved[path.relative_to(ROOT).as_posix()] = record(path)

    mismatches = []
    for item in approved.values():
        path = ROOT / item["relative_path"]
        if not path.is_file() or path.stat().st_size != item["size_bytes"] or sha256_file(path) != item["sha256"]:
            mismatches.append(item["relative_path"])
    add(checks, "approved_step8b_8c_9a_dependencies_unchanged", not mismatches, mismatches)

    version = json.loads((ROOT / "Packages/com.github.homuler.mediapipe/package.json").read_text(encoding="utf-8"))["version"]
    add(checks, "mediapipe_package_version_0_16_3", version == "0.16.3", version)
    project_version = (ROOT / "ProjectSettings/ProjectVersion.txt").read_text(encoding="utf-8")
    add(checks, "unity_version_6000_3_14f1", "m_EditorVersion: 6000.3.14f1" in project_version, project_version.splitlines()[0])

    required_local = [
        ROOT / "Assets/_Project/Scripts/Camera/UnityWebCamFrameSource.cs",
        ROOT / "Assets/_Project/Scripts/Landmarks/MediaPipeLandmarkProvider.cs",
        ROOT / "Assets/_Project/Scripts/Pose/PoseLandmarkProvider.cs",
        ROOT / "Assets/_Project/Scripts/Pose/HandPoseIntegrator.cs",
        ROOT / "Tools/s001_step7b_preprocess_full_v1_0.py",
        PARITY_SEQUENCE, PARITY_G1, PARITY_FEATURE, SOURCE_QUERY,
    ]
    add(checks, "required_local_sources_present", all(path.is_file() for path in required_local), [path.as_posix() for path in required_local if not path.is_file()])

    with np.load(PARITY_FEATURE, allow_pickle=False) as feature:
        metadata = json.loads(str(feature["metadata_json"].item()))
        add(checks, "parity_feature_shape", feature["coordinates"].shape == (131, 16, 3), list(feature["coordinates"].shape))
        add(checks, "parity_mask_shape", feature["valid_mask"].shape == (131, 16), list(feature["valid_mask"].shape))
        add(checks, "parity_feature_float64", feature["coordinates"].dtype == np.float64, str(feature["coordinates"].dtype))
        add(checks, "parity_feature_source_sequence_hash", metadata["source_files"]["sequences/full_GREETING_NORMAL_R01.json"]["sha256"] == sha256_file(PARITY_SEQUENCE), sha256_file(PARITY_SEQUENCE))
        add(checks, "parity_feature_source_g1_hash", metadata["source_files"]["npz/interp_GREETING_NORMAL_R01_g1.npz"]["sha256"] == sha256_file(PARITY_G1), sha256_file(PARITY_G1))
        add(checks, "parity_feature_policy", metadata["normalization"] == "raw_framewise" and metadata["smoothing"] == "no_smoothing" and metadata["selected_max_gap"] == 1, metadata)
    return sorted(approved.values(), key=lambda item: item["relative_path"])


def write_raw_fixture() -> dict:
    sequence = json.loads(PARITY_SEQUENCE.read_text(encoding="utf-8"))
    frames = sequence["frames"]
    component_specs = [("pose", 33), ("left_hand", 21), ("right_hand", 21)]
    component_key = {"pose": "pose", "left_hand": "lh", "right_hand": "rh"}
    payload = bytearray(struct.pack("<8sII", b"WISRAW9B", 1, len(frames)))
    with np.load(PARITY_G1, allow_pickle=False) as g1:
        for frame_index, frame in enumerate(frames):
            payload += struct.pack("<d", float(frame["timestamp_s"]))
            for json_key, count in component_specs:
                short = component_key[json_key]
                strong = g1[f"{short}_strong_forbidden"]
                for landmark_index, landmark in enumerate(frame[json_key]):
                    valid = bool(landmark["te"])
                    raw = landmark["raw_xyz"] if valid and landmark["raw_xyz"] is not None else [float("nan")] * 3
                    flags = (1 if valid else 0) | (2 if landmark["is_reappearance"] else 0) | (4 if strong[frame_index, landmark_index] else 0)
                    payload += struct.pack("<dddB", float(raw[0]), float(raw[1]), float(raw[2]), flags)
    RAW_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    RAW_FIXTURE.write_bytes(payload)
    shutil.copyfile(SOURCE_QUERY, QUERY_FIXTURE)
    return {
        "raw_fixture_bytes": len(payload),
        "raw_fixture_sha256": sha256_bytes(payload),
        "frame_count": len(frames),
    }


def prepare() -> None:
    conflicts = [path for path in (OUTPUT, FIXTURE_ROOT, CONFIG_ROOT, DELIVERY_ZIP, DELIVERY_HASH, DELIVERY_SIDECAR) if path.exists()]
    if conflicts:
        raise RuntimeError("Step 9B output conflict: " + ", ".join(path.relative_to(ROOT).as_posix() for path in conflicts))
    run_id = str(uuid.uuid4())
    checks: list[dict] = []
    approved_records = preflight(checks)
    fixture_info = write_raw_fixture()

    with np.load(PARITY_FEATURE, allow_pickle=False) as feature:
        expected_info = {
            "coordinates_sha256": sha256_bytes(np.ascontiguousarray(feature["coordinates"]).tobytes()),
            "valid_mask_sha256": sha256_bytes(np.ascontiguousarray(feature["valid_mask"]).tobytes()),
            "valid_landmark_count": int(feature["valid_mask"].sum()),
            "interpolated_landmark_count": int(feature["interpolation_mask"].sum()),
            "logical_payload_sha256": str(feature["logical_payload_sha256"].item()),
        }

    fixture_manifest = {
        "version": VERSION,
        "run_id": run_id,
        "sample_id": "S001_P001_GREETING_NORMAL_R01",
        "split": "test",
        "allowed_usage": "Step 9B deterministic regression only; never tuning",
        "frame_count": fixture_info["frame_count"],
        "landmark_order": {"pose": [13, 14, 15, 16], "left_hand": [0, 4, 8, 12, 16, 20], "right_hand": [0, 4, 8, 12, 16, 20]},
        "source_convention": "approved_step7b_front_camera_metric_3d",
        "raw_source": record(PARITY_SEQUENCE),
        "g1_source": record(PARITY_G1),
        "normalized_source": record(PARITY_FEATURE),
        "approved_query_source": record(SOURCE_QUERY),
        "raw_fixture": {"relative_path": RAW_FIXTURE.relative_to(ROOT).as_posix(), "size_bytes": fixture_info["raw_fixture_bytes"], "sha256": fixture_info["raw_fixture_sha256"]},
        "query_fixture": {"relative_path": QUERY_FIXTURE.relative_to(ROOT).as_posix(), "size_bytes": QUERY_FIXTURE.stat().st_size, "sha256": sha256_file(QUERY_FIXTURE)},
        "expected": expected_info,
    }
    write_json(FIXTURE_MANIFEST, fixture_manifest)

    provider_files = [
        ROOT / "Packages/com.github.homuler.mediapipe/package.json",
        ROOT / "Assets/_Project/Scripts/Camera/UnityWebCamFrameSource.cs",
        ROOT / "Assets/_Project/Scripts/Landmarks/MediaPipeLandmarkProvider.cs",
        ROOT / "Assets/_Project/Scripts/Pose/PoseLandmarkProvider.cs",
        ROOT / "Assets/_Project/Scripts/Pose/HandPoseIntegrator.cs",
    ]
    config = {
        "version": VERSION,
        "run_id": run_id,
        "unity_version": "6000.3.14f1",
        "mediapipe_package": "com.github.homuler.mediapipe",
        "mediapipe_version": "0.16.3",
        "selected_path": "UnityWebCamFrameSource -> HandLandmarker/PoseLandmarker VIDEO -> HandPoseIntegrator -> MediaPipeHandPoseObservationBridge",
        "timestamp": "existing provider Time.realtimeSinceStartup result boundary",
        "handedness": "MediaPipe handedness categoryName direct: Left -> left, otherwise right",
        "mirror": "RawImage.uvRect display-only; detector reads the raw WebCamTexture",
        "axes_depth": "NormalizedLandmark x-right/y-down/z-source-relative, passed unchanged into approved body-frame port",
        "normalization": "raw_framewise",
        "selected_max_gap": 1,
        "smoothing": "no_smoothing",
        "automatic_segmentation": False,
        "provider_source_files": [record(path) for path in provider_files],
    }
    write_json(CONFIG_PATH, config)

    smoke = {
        "version": VERSION,
        "run_id": run_id,
        "check": "PHYSICAL_WEBCAM_SMOKE",
        "status": "UNVERIFIED",
        "state": "PHYSICAL_WEBCAM_SMOKE_UNVERIFIED",
        "device_open_attempted": False,
        "reason": "Implementation environment is non-interactive Unity batchmode; physical camera availability, OS permission, preview, and a human manual gesture cannot be verified without fabricating evidence.",
        "deterministic_mock_and_unity_tests_are_separate": True,
    }
    OUTPUT.mkdir(parents=True)
    write_json(OUTPUT / "physical_webcam_smoke.json", smoke)

    for folder in [FIXTURE_ROOT, CONFIG_ROOT]:
        ensure_meta(folder, folder=True)
    for path in [RAW_FIXTURE, QUERY_FIXTURE, FIXTURE_MANIFEST, CONFIG_PATH]:
        ensure_meta(path)

    extra_inputs = [
        ROOT / "Packages/com.github.homuler.mediapipe/package.json",
        ROOT / "ProjectSettings/ProjectVersion.txt",
        ROOT / "Assets/_Project/Scripts/Camera/UnityWebCamFrameSource.cs",
        ROOT / "Assets/_Project/Scripts/Landmarks/MediaPipeLandmarkProvider.cs",
        ROOT / "Assets/_Project/Scripts/Pose/PoseLandmarkProvider.cs",
        ROOT / "Assets/_Project/Scripts/Pose/HandPoseIntegrator.cs",
        ROOT / "Tools/s001_step7b_preprocess_full_v1_0.py",
        PARITY_SEQUENCE, PARITY_G1, PARITY_FEATURE, SOURCE_QUERY,
    ]
    input_map = {item["relative_path"]: item for item in approved_records}
    for path in extra_inputs:
        input_map[path.relative_to(ROOT).as_posix()] = record(path)
    input_manifest = {
        "version": VERSION,
        "run_id": run_id,
        "self_reference_excluded": True,
        "approved_step8b_delivery_sha256": APPROVED_DELIVERIES[STEP8B_ZIP][1],
        "approved_step8c_delivery_sha256": APPROVED_DELIVERIES[STEP8C_ZIP][1],
        "approved_step9a_delivery_sha256": APPROVED_DELIVERIES[STEP9A_ZIP][1],
        "inputs": sorted(input_map.values(), key=lambda item: item["relative_path"]),
        "total_files": len(input_map),
    }
    write_json(OUTPUT / "input_manifest.json", input_manifest)

    add(checks, "raw_fixture_frame_count_131", fixture_info["frame_count"] == 131, fixture_info["frame_count"])
    add(checks, "raw_fixture_nonempty", fixture_info["raw_fixture_bytes"] > 0, fixture_info["raw_fixture_bytes"])
    add(checks, "query_fixture_byte_exact", QUERY_FIXTURE.read_bytes() == SOURCE_QUERY.read_bytes(), sha256_file(QUERY_FIXTURE))
    add(checks, "configuration_manifest_written", CONFIG_PATH.is_file(), CONFIG_PATH.relative_to(ROOT).as_posix())
    add(checks, "physical_webcam_explicitly_unverified", smoke["state"] == "PHYSICAL_WEBCAM_SMOKE_UNVERIFIED" and not smoke["device_open_attempted"], smoke)
    prepare_assertions = assertion_payload(run_id, checks)
    ensure_pass(prepare_assertions, "preflight/prepare")
    write_json(OUTPUT / "prepare_assertion_results.json", prepare_assertions)

    state = {
        "version": VERSION,
        "run_id": run_id,
        "phase": "prepared",
        "status": "LIVE_MEDIAPIPE_INPUT_IMPLEMENTATION_IN_PROGRESS",
        "input_snapshot": [record(ROOT / item["relative_path"]) for item in input_manifest["inputs"]],
    }
    write_json(STATE_PATH, state)
    print(f"Step 9B prepare PASS: {prepare_assertions['pass']}/{prepare_assertions['total']}")
    print(f"Run ID: {run_id}")
    print(f"Raw parity fixture: {fixture_info['raw_fixture_bytes']} bytes, sha256={fixture_info['raw_fixture_sha256']}")


def load_state() -> dict:
    if not STATE_PATH.is_file():
        raise RuntimeError("Step 9B prepare state is missing")
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def verify_input_snapshot(state: dict) -> list[str]:
    mismatches = []
    for item in state["input_snapshot"]:
        path = ROOT / item["relative_path"]
        if not path.is_file() or path.stat().st_size != item["size_bytes"] or sha256_file(path) != item["sha256"]:
            mismatches.append(item["relative_path"])
    return mismatches


def parse_unity(path: Path, fixture: str, command: str) -> dict:
    root = ET.parse(path).getroot()
    cases = [case for case in root.iter("test-case") if case.attrib.get("fullname", "").startswith(fixture)]
    return {
        "result": root.attrib.get("result"),
        "testcasecount": len(cases),
        "passed": sum(case.attrib.get("result") == "Passed" for case in cases),
        "failed": sum(case.attrib.get("result") == "Failed" for case in cases),
        "skipped": sum(case.attrib.get("result") == "Skipped" for case in cases),
        "inconclusive": sum(case.attrib.get("result") == "Inconclusive" for case in cases),
        "duration_seconds": float(root.attrib.get("duration", "0")),
        "fixture": fixture,
        "test_names": sorted(case.attrib.get("fullname") for case in cases),
        "command": command,
        "results_relative_path": path.relative_to(ROOT).as_posix(),
        "results_size_bytes": path.stat().st_size,
        "results_sha256": sha256_file(path),
    }


def output_paths() -> list[Path]:
    paths = []
    paths.extend(sorted(SOURCE_ROOT.glob("*.cs")))
    paths.extend(Path(str(path) + ".meta") for path in list(paths))
    paths.append(Path(str(SOURCE_ROOT) + ".meta"))
    paths.extend([PREFAB, Path(str(PREFAB) + ".meta"), ROOT / "Assets/_Project/Resources.meta", ROOT / "Assets/_Project/Resources/Diagnostics.meta"])
    paths.extend([CONFIG_PATH, Path(str(CONFIG_PATH) + ".meta"), Path(str(CONFIG_ROOT) + ".meta")])
    paths.extend([RAW_FIXTURE, QUERY_FIXTURE, FIXTURE_MANIFEST])
    paths.extend(Path(str(path) + ".meta") for path in [RAW_FIXTURE, QUERY_FIXTURE, FIXTURE_MANIFEST])
    paths.append(Path(str(FIXTURE_ROOT) + ".meta"))
    paths.extend([EDIT_TEST, Path(str(EDIT_TEST) + ".meta"), PLAY_TEST, Path(str(PLAY_TEST) + ".meta")])
    paths.append(ROOT / "Tools/s001_step9b_live_mediapipe_input_v1_0.py")
    for name in [
        "physical_webcam_smoke.json", "input_manifest.json", "prepare_assertion_results.json", "run_state.json",
        "unity_editmode_test_results.xml", "unity_editmode_test.log", "unity_editmode_test_summary.json", "unity_editmode_test_command.txt",
        "unity_playmode_test_results.xml", "unity_playmode_test.log", "unity_playmode_test_summary.json", "unity_playmode_test_command.txt",
        "assertion_results.json", "STEP9B_LIVE_MEDIAPIPE_INPUT_V1_0_REPORT.md",
    ]:
        paths.append(OUTPUT / name)
    return sorted(set(paths), key=lambda path: path.relative_to(ROOT).as_posix())


def package(run_id: str, paths: list[Path]) -> dict:
    manifest_path = OUTPUT / "output_file_manifest.json"
    records = [{**record(path), "artifact_type": "step9b_output"} for path in paths]
    manifest = {
        "version": VERSION,
        "schema_version": "self_excluding_v1",
        "run_id": run_id,
        "self_reference_excluded": True,
        "outputs": records,
        "total_files": len(records),
    }
    write_json(manifest_path, manifest)
    packaged = paths + [manifest_path]

    temp_zip = ROOT / f".{DELIVERY_NAME}.{run_id}.tmp"
    if temp_zip.exists():
        temp_zip.unlink()
    with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in packaged:
            archive.write(path, path.relative_to(ROOT).as_posix())
    zip_hash = sha256_file(temp_zip)
    hash_line = f"{zip_hash}  {DELIVERY_NAME}\n"
    DELIVERY_HASH.write_text(hash_line, encoding="utf-8", newline="\n")

    post_checks: list[dict] = []
    def post(name: str, passed: bool, evidence=None):
        post_checks.append({"check_name": name, "status": "PASS" if passed else "FAIL", "evidence": evidence})

    expected = {item["relative_path"]: item for item in records}
    with zipfile.ZipFile(temp_zip, "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        bad_crc = archive.testzip()
        post("archive_crc", bad_crc is None, bad_crc)
        post("archive_unique_entries", len(names) == len(set(names)), len(names))
        unsafe = []
        symlinks = []
        encrypted = []
        manifest_inside = None
        bad_json_run_ids = []
        report_run_id_ok = False
        for index, info in enumerate(infos):
            pure = PurePosixPath(info.filename)
            safe = not pure.is_absolute() and ".." not in pure.parts and "\\" not in info.filename
            if not safe: unsafe.append(info.filename)
            if stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK: symlinks.append(info.filename)
            if info.flag_bits & 0x1: encrypted.append(info.filename)
            data = archive.read(info)
            post(f"entry_{index:04d}_crc_read", len(data) == info.file_size, info.filename)
            if info.filename.endswith(".json"):
                payload = json.loads(data.decode("utf-8"))
                if isinstance(payload, dict) and payload.get("run_id") != run_id:
                    bad_json_run_ids.append(info.filename)
            if info.filename.endswith("STEP9B_LIVE_MEDIAPIPE_INPUT_V1_0_REPORT.md"):
                report_text = data.decode("utf-8")
                report_run_id_ok = run_id in report_text and "LIVE_MEDIAPIPE_INPUT_COMPLETE_BLOCKED_PENDING_EXTERNAL_REVIEW" in report_text
            if info.filename.endswith("output_file_manifest.json"):
                manifest_inside = json.loads(data.decode("utf-8"))
                continue
            item = expected.get(info.filename)
            post(f"entry_{index:04d}_manifest_record", item is not None, info.filename)
            post(f"entry_{index:04d}_size", item is not None and len(data) == item["size_bytes"], info.filename)
            post(f"entry_{index:04d}_sha256", item is not None and sha256_bytes(data) == item["sha256"], info.filename)
        post("archive_safe_paths", not unsafe, unsafe)
        post("archive_no_symlinks", not symlinks, symlinks)
        post("archive_not_encrypted", not encrypted, encrypted)
        post("manifest_present", manifest_inside is not None, None)
        post("manifest_target_count", manifest_inside is not None and manifest_inside["total_files"] == len(records), None)
        post("manifest_exact_entry_set", set(names) == set(expected) | {manifest_path.relative_to(ROOT).as_posix()}, None)
        post("single_step9b_run_id_all_json", not bad_json_run_ids, bad_json_run_ids)
        post("single_step9b_run_id_report", report_run_id_ok, None)
        post("no_package_cache_or_third_party_files", not [name for name in names if name.startswith("Packages/") or "/PackageCache/" in name], None)
        post("output_manifest_record_duplicates", len(expected) == len(records), len(records) - len(expected))
        config_bytes = archive.read(CONFIG_PATH.relative_to(ROOT).as_posix())
        post("single_step9b_run_id_config", json.loads(config_bytes)["run_id"] == run_id, None)
        smoke_bytes = archive.read((OUTPUT / "physical_webcam_smoke.json").relative_to(ROOT).as_posix())
        post("physical_webcam_unverified_not_faked", json.loads(smoke_bytes)["state"] == "PHYSICAL_WEBCAM_SMOKE_UNVERIFIED", None)
        fixture_payload = json.loads(archive.read(FIXTURE_MANIFEST.relative_to(ROOT).as_posix()).decode("utf-8"))
        post("parity_fixture_raw_hash", sha256_bytes(archive.read(RAW_FIXTURE.relative_to(ROOT).as_posix())) == fixture_payload["raw_fixture"]["sha256"], None)
        post("parity_fixture_query_hash", sha256_bytes(archive.read(QUERY_FIXTURE.relative_to(ROOT).as_posix())) == fixture_payload["query_fixture"]["sha256"], None)

    post("detached_hash_strict", DELIVERY_HASH.read_text(encoding="utf-8") == hash_line, None)
    for delivery, (_, approved_hash) in APPROVED_DELIVERIES.items():
        post(f"approved_{delivery.name.split('_')[1]}_unchanged", sha256_file(delivery) == approved_hash, sha256_file(delivery))
    input_mismatch = verify_input_snapshot(load_state())
    post("approved_input_snapshot_unchanged", not input_mismatch, input_mismatch)

    check_names = [item["check_name"] for item in post_checks]
    duplicate_names = sorted(name for name, count in Counter(check_names).items() if count > 1)
    failures = [item for item in post_checks if item["status"] != "PASS"]
    sidecar = {
        "version": VERSION,
        "run_id": run_id,
        "zip_file": DELIVERY_NAME,
        "zip_size_bytes": temp_zip.stat().st_size,
        "zip_sha256": zip_hash,
        "zip_entry_count": len(packaged),
        "zip_unique_entry_count": len(packaged),
        "output_manifest_target_count": len(records),
        "output_manifest_self_reference_excluded": True,
        "detached_hash_strict_parse": True,
        "post_zip_check_total": len(post_checks),
        "post_zip_check_unique": len(set(check_names)),
        "post_zip_check_name_duplicates": len(duplicate_names),
        "post_zip_record_duplicates": 0,
        "post_zip_failure_count": len(failures),
        "duplicate_check_names": duplicate_names,
        "physical_webcam_smoke_state": "PHYSICAL_WEBCAM_SMOKE_UNVERIFIED",
        "verification_status": "PASS" if not failures and not duplicate_names else "FAIL",
        "checks": post_checks,
    }
    write_json(DELIVERY_SIDECAR, sidecar)
    if sidecar["verification_status"] != "PASS":
        raise RuntimeError(f"post-ZIP verification failed: {[item['check_name'] for item in failures]}")
    os.replace(temp_zip, DELIVERY_ZIP)
    return sidecar


def finalize(edit_results: Path, play_results: Path) -> None:
    state = load_state()
    run_id = state["run_id"]
    if state["phase"] != "prepared":
        raise RuntimeError(f"invalid state phase: {state['phase']}")
    mismatches = verify_input_snapshot(state)
    if mismatches:
        raise RuntimeError(f"approved input changed after prepare: {mismatches}")

    edit_command = (OUTPUT / "unity_editmode_test_command.txt").read_text(encoding="utf-8").strip()
    play_command = (OUTPUT / "unity_playmode_test_command.txt").read_text(encoding="utf-8").strip()
    edit_summary = parse_unity(edit_results, "WordsInSilence.Tests.EditMode.LiveMediaPipeInputV1Tests", edit_command)
    play_summary = parse_unity(play_results, "WordsInSilence.Tests.PlayMode.LiveMediaPipeDiagnosticHarnessV1Tests", play_command)
    parity_match = re.search(r"STEP9B_PARITY_MAX_DELTA=([0-9.Ee+\-]+)", edit_results.read_text(encoding="utf-8"))
    parity_max_delta = float(parity_match.group(1)) if parity_match else None
    edit_summary["approved_raw_normalized_parity_max_delta"] = parity_max_delta
    edit_summary["approved_raw_normalized_parity_tolerance"] = 1e-12
    edit_summary["approved_parity_frames"] = 131
    edit_summary["approved_parity_valid_landmarks"] = 1915
    edit_summary["approved_parity_interpolated_landmarks"] = 29
    edit_log = OUTPUT / "unity_editmode_test.log"
    play_log = OUTPUT / "unity_playmode_test.log"
    edit_summary.update({"run_id": run_id, "log_relative_path": edit_log.relative_to(ROOT).as_posix(), "log_size_bytes": edit_log.stat().st_size, "log_sha256": sha256_file(edit_log)})
    play_summary.update({"run_id": run_id, "log_relative_path": play_log.relative_to(ROOT).as_posix(), "log_size_bytes": play_log.stat().st_size, "log_sha256": sha256_file(play_log)})
    write_json(OUTPUT / "unity_editmode_test_summary.json", edit_summary)
    write_json(OUTPUT / "unity_playmode_test_summary.json", play_summary)

    prepare_assertions = json.loads((OUTPUT / "prepare_assertion_results.json").read_text(encoding="utf-8"))
    checks = [{"name": f"prepare::{item['name']}", "status": item["status"], "evidence": item["evidence"]} for item in prepare_assertions["results"]]
    add(checks, "approved_inputs_unchanged_after_unity", not mismatches, mismatches)
    add(checks, "unity_editmode_passed", edit_summary["result"] == "Passed" and edit_summary["failed"] == 0 and edit_summary["passed"] == edit_summary["testcasecount"] and edit_summary["testcasecount"] == 10, edit_summary)
    add(checks, "unity_playmode_passed", play_summary["result"] == "Passed" and play_summary["failed"] == 0 and play_summary["passed"] == play_summary["testcasecount"] and play_summary["testcasecount"] == 1, play_summary)
    add(checks, "approved_raw_normalized_max_delta", parity_max_delta is not None and parity_max_delta <= 1e-12, parity_max_delta)
    add(checks, "unity_logs_no_compile_error", "error CS" not in edit_log.read_text(encoding="utf-8", errors="replace") and "error CS" not in play_log.read_text(encoding="utf-8", errors="replace"), None)
    smoke = json.loads((OUTPUT / "physical_webcam_smoke.json").read_text(encoding="utf-8"))
    add(checks, "physical_webcam_state_separate", smoke["state"] == "PHYSICAL_WEBCAM_SMOKE_UNVERIFIED" and smoke["status"] == "UNVERIFIED", smoke)
    assertions = assertion_payload(run_id, checks)
    ensure_pass(assertions, "final assertions")
    write_json(OUTPUT / "assertion_results.json", assertions)

    report = f"""# Step 9B Live MediaPipe Frame Adapter and Manual Capture Diagnostic v1.0\n\n**Run ID**: {run_id}\n**Status**: LIVE_MEDIAPIPE_INPUT_COMPLETE_BLOCKED_PENDING_EXTERNAL_REVIEW\n**Physical webcam**: PHYSICAL_WEBCAM_SMOKE_UNVERIFIED\n\n## Installed path\n\n- Unity 6000.3.14f1; MediaPipe Unity Plugin 0.16.3.\n- UnityWebCamFrameSource -> HandLandmarker/PoseLandmarker VIDEO -> HandPoseIntegrator -> Step 9B bridge.\n- Raw WebCamTexture pixels are detector input; RawImage UV mirror is display-only. Handedness categoryName is mapped directly.\n\n## Deterministic adapter\n\n- Exact order: pose [13,14,15,16], left/right hand [0,4,8,12,16,20].\n- raw_framewise, max_gap=1, no_smoothing, NaN/mask preservation, timestamp/reappearance/strong-depth boundaries.\n- Approved GREETING_NORMAL_R01 raw-to-Step8A parity fixture: 131 frames.\n\n## Unity\n\n- EditMode: {edit_summary['passed']} PASS / {edit_summary['failed']} FAIL / {edit_summary['skipped']} skipped / {edit_summary['inconclusive']} inconclusive.\n- PlayMode: {play_summary['passed']} PASS / {play_summary['failed']} FAIL / {play_summary['skipped']} skipped / {play_summary['inconclusive']} inconclusive.\n- Assertions: {assertions['pass']} PASS / {assertions['fail']} FAIL / {assertions['duplicate_count']} duplicate names.\n\n## Physical webcam\n\nNo physical webcam PASS is claimed. This non-interactive batchmode environment cannot verify a camera device, OS permission, preview, or a human manual gesture.\n\n`LIVE_MEDIAPIPE_INPUT_COMPLETE_BLOCKED_PENDING_EXTERNAL_REVIEW`\n`PHYSICAL_WEBCAM_SMOKE_UNVERIFIED`\n"""
    report = report.replace(
        "- Approved GREETING_NORMAL_R01 raw-to-Step8A parity fixture: 131 frames.",
        f"- Approved GREETING_NORMAL_R01 raw-to-Step8A parity fixture: 131 frames, exact mask, maximum coordinate delta {parity_max_delta:.17g} (tolerance 1e-12).\n"
        "- Mock-provider recognition and all nine approved Step 9A replay decisions are unchanged.")
    (OUTPUT / "STEP9B_LIVE_MEDIAPIPE_INPUT_V1_0_REPORT.md").write_text(report, encoding="utf-8", newline="\n")

    state.update({"phase": "finalized", "status": "LIVE_MEDIAPIPE_INPUT_COMPLETE_BLOCKED_PENDING_EXTERNAL_REVIEW", "physical_webcam_smoke": "PHYSICAL_WEBCAM_SMOKE_UNVERIFIED"})
    write_json(STATE_PATH, state)
    paths = output_paths()
    missing = [path.relative_to(ROOT).as_posix() for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"required outputs missing: {missing}")
    sidecar = package(run_id, paths)
    print(f"Step 9B finalize PASS: assertions={assertions['pass']}/{assertions['total']}")
    print(f"Package PASS: entries={sidecar['zip_entry_count']}, bytes={DELIVERY_ZIP.stat().st_size}, sha256={sidecar['zip_sha256']}")
    print(f"Post-ZIP PASS: {sidecar['post_zip_check_total']} checks / 0 failures / 0 duplicates")
    print("Physical webcam: PHYSICAL_WEBCAM_SMOKE_UNVERIFIED")
    print("Status: LIVE_MEDIAPIPE_INPUT_COMPLETE_BLOCKED_PENDING_EXTERNAL_REVIEW")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--edit-results", type=Path)
    parser.add_argument("--play-results", type=Path)
    args = parser.parse_args()
    if args.prepare == args.finalize:
        parser.error("select exactly one of --prepare or --finalize")
    if args.prepare:
        prepare()
    else:
        if args.edit_results is None or args.play_results is None:
            parser.error("--finalize requires --edit-results and --play-results")
        finalize((ROOT / args.edit_results).resolve() if not args.edit_results.is_absolute() else args.edit_results,
                 (ROOT / args.play_results).resolve() if not args.play_results.is_absolute() else args.play_results)


if __name__ == "__main__":
    main()
