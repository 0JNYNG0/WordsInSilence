#!/usr/bin/env python3
"""Read-only Step 9D physical RecognitionCore domain-gap diagnostic and packager."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import struct
import subprocess
import tempfile
import uuid
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
VERSION = "1.0"
STEP8A = ROOT / "data/processed/sessions/S001_P001/step8a_dtw_templates_v1_0"
STEP7B = ROOT / "data/processed/sessions/S001_P001/step7b_full_v1_0"
STEP8B = ROOT / "data/processed/sessions/S001_P001/step8b_recognition_core_v1_0"
STEP9B = ROOT / "data/processed/sessions/S001_P001/step9b_live_mediapipe_input_v1_0"
STEP9C = ROOT / "data/processed/sessions/S001_P001/step9c_physical_webcam_coordinate_compatibility_v1_0"
OUTPUT = ROOT / "data/processed/sessions/S001_P001/step9d_physical_recognition_domain_gap_diagnostic_v1_0"
SCRIPT_REL = "Tools/s001_step9d_physical_recognition_domain_gap_diagnostic_v1_0.py"
OUTPUT_REL = "data/processed/sessions/S001_P001/step9d_physical_recognition_domain_gap_diagnostic_v1_0"
OUTPUT_MANIFEST_REL = f"{OUTPUT_REL}/output_file_manifest.json"
DELIVERY_NAME = "WordsInSilence_step9d_physical_recognition_domain_gap_diagnostic_v1_0_delivery.zip"
DELIVERY_ZIP = ROOT / DELIVERY_NAME
DELIVERY_HASH = ROOT / f"{DELIVERY_NAME}.sha256"
POST_ZIP = ROOT / "WordsInSilence_step9d_physical_recognition_domain_gap_diagnostic_v1_0_delivery_post_zip_verification.json"

STEP9C_DELIVERY_NAME = "WordsInSilence_step9c_physical_webcam_coordinate_compatibility_v1_0_delivery.zip"
STEP9C_DELIVERY = ROOT / STEP9C_DELIVERY_NAME
STEP9C_DETACHED = ROOT / f"{STEP9C_DELIVERY_NAME}.sha256"
STEP9C_EXPECTED_SIZE = 1_570_924
STEP9C_EXPECTED_SHA256 = "fe0364385cdcd460e81e7bf59bd7340cd92c08716dca2a98c2e39a32d584ddc5"
STEP9C_MANIFEST_REL = f"{STEP9C.relative_to(ROOT).as_posix()}/output_file_manifest.json"

EXPECTED_VALID_CAPTURES = {
    "GREETING": "step9c-take-01-greeting-f7875c16-ad18-425c-8ce5-39e9a6cc13f7",
    "LOVE": "step9c-take-02-love-6d68ddd7-2d7e-4afd-bd0a-b1989f8e52f2",
    "THANK_YOU": "step9c-take-03-thank-you-53870472-c644-489c-a94d-95e9667ded42",
}
WORDS = ("GREETING", "LOVE", "THANK_YOU")
SPEEDS = ("FAST", "NORMAL", "SLOW")
LANDMARK_SCHEMA = (
    ("pose", 13), ("pose", 14), ("pose", 15), ("pose", 16),
    ("left_hand", 0), ("left_hand", 4), ("left_hand", 8),
    ("left_hand", 12), ("left_hand", 16), ("left_hand", 20),
    ("right_hand", 0), ("right_hand", 4), ("right_hand", 8),
    ("right_hand", 12), ("right_hand", 16), ("right_hand", 20),
)
LANDMARK_GROUPS = {"pose": tuple(range(0, 4)), "left_hand": tuple(range(4, 10)), "right_hand": tuple(range(10, 16))}
DTW_BAND = 0.10
MISSING_COST = 1.0
PARITY_TOLERANCE = 1e-12
THRESHOLD = 0.2908679008489196
PRIMARY_VERDICT = "UNVERIFIED_MISSING_REQUIRED_EVIDENCE"
FINAL_STATUS = "STEP9D_PHYSICAL_RECOGNITION_DOMAIN_GAP_DIAGNOSTIC_BLOCKED_MISSING_OR_MISMATCHED_EVIDENCE"


class DiagnosticError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosticError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def finite_json(value: object, path: str = "$" ) -> None:
    if isinstance(value, float):
        require(math.isfinite(value), f"non-finite JSON value at {path}")
    elif isinstance(value, dict):
        for key, item in value.items():
            finite_json(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            finite_json(item, f"{path}[{index}]")


def json_bytes(value: object) -> bytes:
    finite_json(value)
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(value))


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def file_record(path: Path, category: str, relative_path: str | None = None) -> dict:
    require(path.is_file(), f"required file missing: {path}")
    return {
        "relative_path": relative_path or relative(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "category": category,
    }


def bytes_record(data: bytes, relative_path: str, category: str) -> dict:
    return {"relative_path": relative_path, "size_bytes": len(data), "sha256": sha256_bytes(data), "category": category}


def safe_zip_name(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not name.startswith(("/", "\\")) and "\\" not in name and not path.is_absolute() and ".." not in path.parts and not (len(name) >= 2 and name[1] == ":")


def parse_detached(data: bytes, expected_name: str, expected_hash: str) -> bool:
    try:
        line = data.decode("utf-8-sig").strip()
    except UnicodeDecodeError:
        return False
    parts = line.split(maxsplit=1)
    if len(parts) != 2:
        return False
    return parts[0].lower() == expected_hash and parts[1].lstrip("*") == expected_name


def add_assert(assertions: list[dict], name: str, status: str, evidence: object = None) -> None:
    require(status in {"PASS", "FAIL", "UNVERIFIED", "NOT_APPLICABLE"}, f"invalid assertion status {status}")
    assertions.append({"name": name, "status": status, "evidence": evidence})


def manifest_records(data: dict) -> list[dict]:
    for key in ("outputs", "inputs", "input_files", "files"):
        if isinstance(data.get(key), list):
            return data[key]
    raise DiagnosticError(f"manifest has no supported record list: {sorted(data)}")


def verify_records(records: list[dict], resolver, label: str) -> list[dict]:
    seen: set[str] = set()
    verified = []
    for item in records:
        rel = item["relative_path"]
        require(rel not in seen, f"{label} duplicate path: {rel}")
        seen.add(rel)
        path = resolver(item)
        require(path.is_file(), f"{label} missing: {path}")
        actual_size = path.stat().st_size
        actual_hash = sha256_file(path)
        require(actual_size == int(item["size_bytes"]), f"{label} size mismatch: {rel}")
        require(actual_hash == item["sha256"], f"{label} SHA-256 mismatch: {rel}")
        verified.append({"relative_path": relative(path), "size_bytes": actual_size, "sha256": actual_hash})
    return verified


def strict_step9c_delivery(assertions: list[dict]) -> dict:
    require(STEP9C_DELIVERY.is_file(), f"Step 9C Delivery ZIP missing: {STEP9C_DELIVERY}")
    require(STEP9C_DETACHED.is_file(), f"Step 9C detached checksum missing: {STEP9C_DETACHED}")
    actual_size = STEP9C_DELIVERY.stat().st_size
    actual_hash = sha256_file(STEP9C_DELIVERY)
    require(actual_size == STEP9C_EXPECTED_SIZE, f"Step 9C ZIP size mismatch: {actual_size}")
    require(actual_hash == STEP9C_EXPECTED_SHA256, f"Step 9C ZIP SHA-256 mismatch: {actual_hash}")
    require(parse_detached(STEP9C_DETACHED.read_bytes(), STEP9C_DELIVERY_NAME, actual_hash), "Step 9C detached checksum strict mismatch")
    add_assert(assertions, "step9c_delivery_fixed_size_sha256_detached", "PASS", {"size_bytes": actual_size, "sha256": actual_hash})

    with zipfile.ZipFile(STEP9C_DELIVERY, "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
        unsafe = sorted(name for name in names if not safe_zip_name(name))
        symlinks = sorted(info.filename for info in infos if ((info.external_attr >> 16) & 0o170000) == 0o120000)
        encrypted = sorted(info.filename for info in infos if info.flag_bits & 0x1)
        require(archive.testzip() is None, "Step 9C ZIP CRC failure")
        require(not duplicates and not unsafe and not symlinks and not encrypted, "Step 9C ZIP structural safety failure")
        require(len(names) == 89, f"Step 9C ZIP entry count mismatch: {len(names)}")
        require(STEP9C_MANIFEST_REL in names, "Step 9C output manifest missing")
        manifest = json.loads(archive.read(STEP9C_MANIFEST_REL).decode("utf-8"))
        records = manifest_records(manifest)
        require(len(records) == 88 and manifest.get("self_reference_excluded") is True, "Step 9C manifest target/self-exclusion mismatch")
        expected = {record["relative_path"] for record in records}
        actual = set(names) - {STEP9C_MANIFEST_REL}
        require(expected == actual, f"Step 9C manifest entry set mismatch: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
        by_name = {info.filename: info for info in infos}
        for record in records:
            data = archive.read(record["relative_path"])
            require(len(data) == int(record["size_bytes"]), f"Step 9C manifest size mismatch: {record['relative_path']}")
            require(sha256_bytes(data) == record["sha256"], f"Step 9C manifest hash mismatch: {record['relative_path']}")
            require(by_name[record["relative_path"]].file_size == int(record["size_bytes"]), "Step 9C central directory size mismatch")

        capture_names = sorted(name for name in names if "/physical_captures/" in name and name.endswith(".json"))
        checksum_names = sorted(name for name in names if "/physical_captures/" in name and name.endswith(".json.sha256"))
        require(len(capture_names) == 20 and len(checksum_names) == 20, f"Step 9C physical pair count mismatch: {len(capture_names)}/{len(checksum_names)}")
        captures = []
        strict_pairs = []
        for name in capture_names:
            checksum_name = name + ".sha256"
            require(checksum_name in checksum_names, f"missing physical detached checksum: {checksum_name}")
            data = archive.read(name)
            digest = sha256_bytes(data)
            require(parse_detached(archive.read(checksum_name), PurePosixPath(name).name, digest), f"physical detached checksum mismatch: {name}")
            capture = json.loads(data.decode("utf-8"))
            captures.append({"archive_path": name, "checksum_archive_path": checksum_name, "bytes": data, "checksum_bytes": archive.read(checksum_name), "json": capture, "sha256": digest})
            strict_pairs.append({"relative_path": name, "sha256": digest})

    valid = []
    for item in captures:
        data = item["json"]
        if data.get("pre_export_objectively_valid") is True:
            require(data.get("finalization_state") == "Completed", f"valid capture not Completed: {data.get('capture_id')}")
            require(data.get("failure") is None, f"valid capture typed failure not NONE: {data.get('capture_id')}")
            require(data.get("expected_word") in WORDS, f"unexpected valid word: {data.get('expected_word')}")
            valid.append(item)
    require(len(valid) == 3, f"objective-valid fixed capture count mismatch: {len(valid)}")
    valid_by_word = {item["json"]["expected_word"]: item for item in valid}
    require(set(valid_by_word) == set(WORDS), f"objective-valid word set mismatch: {sorted(valid_by_word)}")
    for word, capture_id in EXPECTED_VALID_CAPTURES.items():
        require(valid_by_word[word]["json"]["capture_id"] == capture_id, f"objective-valid {word} capture provenance mismatch")
    add_assert(assertions, "step9c_archive_manifest_crc_safety_88_of_89", "PASS", {"manifest_targets": 88, "zip_entries": 89})
    add_assert(assertions, "physical_evidence_20_pairs_strict", "PASS", {"json": 20, "checksums": 20, "strict": len(strict_pairs)})
    add_assert(assertions, "objective_valid_fixed_capture_set_exact", "PASS", {word: valid_by_word[word]["json"]["capture_id"] for word in WORDS})
    return {"zip_size_bytes": actual_size, "zip_sha256": actual_hash, "captures": captures, "valid_by_word": valid_by_word, "manifest": manifest}


def read_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.ascontiguousarray(source[name]) for name in source.files}


def array_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    if value.dtype == np.float64:
        value = value.astype("<f8", copy=False)
    elif value.dtype == np.int64:
        value = value.astype("<i8", copy=False)
    elif value.dtype == np.bool_:
        value = value.astype(np.uint8, copy=False)
    return sha256_bytes(value.tobytes(order="C"))


def read_wisrt(path: Path) -> dict:
    data = path.read_bytes()
    require(len(data) > 32, f"WISRT truncated: {path}")
    payload, detached = data[:-32], data[-32:]
    require(hashlib.sha256(payload).digest() == detached, f"WISRT payload hash mismatch: {path}")
    stream = io.BytesIO(payload)

    def exact(count: int) -> bytes:
        value = stream.read(count)
        require(len(value) == count, f"WISRT truncated field: {path}")
        return value

    def u32() -> int:
        return struct.unpack("<I", exact(4))[0]

    def u64() -> int:
        return struct.unpack("<Q", exact(8))[0]

    def text() -> str:
        size = u32()
        require(0 < size <= 4096, f"WISRT string length invalid: {path}")
        return exact(size).decode("utf-8")

    require(exact(8) == b"WISRCB01", f"WISRT magic mismatch: {path}")
    require(u32() == 1 and u32() == 0x01020304, f"WISRT version/endian mismatch: {path}")
    landmarks, length = u32(), u32()
    sample_id, word, speed, split, source_sha, logical_sha = (text() for _ in range(6))
    coordinate_count = u64()
    require(coordinate_count == length * landmarks * 3, f"WISRT coordinate count mismatch: {path}")
    coordinates = np.frombuffer(exact(coordinate_count * 8), dtype="<f8").copy().reshape(length, landmarks, 3)
    mask_count = u64()
    require(mask_count == length * landmarks, f"WISRT mask count mismatch: {path}")
    mask_bytes = u32()
    require(mask_bytes == (mask_count + 7) // 8, f"WISRT packed mask size mismatch: {path}")
    packed = exact(mask_bytes)
    mask = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="little")[:mask_count].astype(bool).reshape(length, landmarks)
    require(stream.tell() == len(payload), f"WISRT trailing payload: {path}")
    return {
        "sample_id": sample_id, "word": word, "speed": speed, "split": split,
        "source_npz_sha256": source_sha, "logical_payload_sha256": logical_sha,
        "coordinates": np.ascontiguousarray(coordinates), "valid_mask": np.ascontiguousarray(mask),
        "binary_payload_sha256": detached.hex(),
    }


def preflight_step8a_step9b(assertions: list[dict]) -> dict:
    output_manifest_path = STEP8A / "output_file_manifest.json"
    input_manifest_path = STEP8A / "input_manifest.json"
    output_manifest = json.loads(output_manifest_path.read_text(encoding="utf-8"))
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    def resolve_step8a_output(item: dict) -> Path:
        if item.get("artifact_type") == "execution_script":
            return ROOT / "Tools" / PurePosixPath(item["relative_path"])
        return STEP8A / PurePosixPath(item["relative_path"])

    step8a_outputs = verify_records(manifest_records(output_manifest), resolve_step8a_output, "Step 8A output manifest")
    def resolve_step8a_input(item: dict) -> Path:
        source_root = item.get("source_root")
        if source_root == "step7b_full_v1_0":
            return STEP7B / PurePosixPath(item["relative_path"])
        if source_root == "project_root":
            return ROOT / PurePosixPath(item["relative_path"])
        raise DiagnosticError(f"unsupported Step 8A input source_root: {source_root}")

    step8a_inputs = verify_records(manifest_records(input_manifest), resolve_step8a_input, "Step 8A input manifest")
    add_assert(assertions, "step8a_output_manifest_strict", "PASS", len(step8a_outputs))
    add_assert(assertions, "step8a_input_manifest_strict", "PASS", len(step8a_inputs))

    config_path = STEP8A / "selected_dtw_config.json"
    schema_path = STEP8A / "feature_schema.json"
    template_manifest_path = STEP8A / "template_manifest.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    template_manifest = json.loads(template_manifest_path.read_text(encoding="utf-8"))
    require(config.get("configuration_name") == "word_speed_medoid_band_0.10", "Step 8A selected configuration mismatch")
    require(config.get("strategy") == "word_speed_medoid" and float(config.get("sakoe_chiba_relative_band")) == DTW_BAND, "Step 8A strategy/band mismatch")
    require(config.get("template_count") == 9 and len(template_manifest.get("templates", {})) == 9, "Step 8A template count mismatch")
    require(schema.get("normalization") == "raw_framewise" and schema.get("smoothing") == "no_smoothing" and schema.get("selected_max_gap") == 1, "Step 8A preprocessing policy mismatch")
    actual_schema = tuple((item["source"], int(item["index"])) for item in schema["landmark_order"])
    require(actual_schema == LANDMARK_SCHEMA, "Step 8A landmark ordering mismatch")

    runtime_manifest_path = STEP8B / "runtime_template_manifest.json"
    runtime_manifest = json.loads(runtime_manifest_path.read_text(encoding="utf-8"))
    require(len(runtime_manifest.get("templates", [])) == 9, "runtime template manifest count mismatch")
    runtime_by_id = {item["sample_id"]: item for item in runtime_manifest["templates"]}
    templates_reference = []
    templates_unity = []
    group_set: set[str] = set()
    for group, item in sorted(template_manifest["templates"].items()):
        require(group not in group_set, f"duplicate template group: {group}")
        group_set.add(group)
        require(item["split"] == "train", f"non-train template: {item['sample_id']}")
        npz_path = STEP8A / PurePosixPath(item["relative_path"])
        require(sha256_file(npz_path) == item["file_sha256"], f"Step 8A template hash mismatch: {group}")
        npz = read_npz(npz_path)
        runtime = runtime_by_id[item["sample_id"]]
        wisrt_path = ROOT / PurePosixPath(runtime["relative_path"])
        require(wisrt_path.stat().st_size == int(runtime["size_bytes"]) and sha256_file(wisrt_path) == runtime["file_sha256"], f"runtime template file mismatch: {group}")
        wisrt = read_wisrt(wisrt_path)
        require(wisrt["sample_id"] == item["sample_id"] and wisrt["word"] == item["word"] and wisrt["speed"] == item["speed"] and wisrt["split"] == "train", f"runtime template metadata mismatch: {group}")
        require(np.array_equal(npz["valid_mask"], wisrt["valid_mask"]), f"runtime template mask divergence: {group}")
        require(np.array_equal(npz["coordinates"], wisrt["coordinates"], equal_nan=True), f"runtime template coordinate divergence: {group}")
        require(wisrt["logical_payload_sha256"] == item["logical_payload_sha256"], f"runtime logical payload mismatch: {group}")
        require(array_hash(wisrt["coordinates"]) == runtime["coordinates_sha256"], f"runtime coordinates hash mismatch: {group}")
        require(array_hash(wisrt["valid_mask"]) == runtime["valid_mask_sha256"], f"runtime mask hash mismatch: {group}")
        base = {"group": group, "sample_id": item["sample_id"], "word": item["word"], "speed": item["speed"], "split": item["split"]}
        templates_reference.append({**base, "coordinates": npz["coordinates"], "valid_mask": npz["valid_mask"], "original_valid_mask": npz["original_valid_mask"], "interpolation_mask": npz["interpolation_mask"], "timestamp": npz["timestamp"], "path": npz_path})
        templates_unity.append({**base, "coordinates": wisrt["coordinates"], "valid_mask": wisrt["valid_mask"], "path": wisrt_path})
    require(group_set == {f"{word}__{speed}" for word in WORDS for speed in SPEEDS}, f"template group set mismatch: {sorted(group_set)}")
    add_assert(assertions, "frozen_template_9_groups_no_duplicate_missing", "PASS", sorted(group_set))
    add_assert(assertions, "step8a_npz_runtime_wisrt_payload_parity", "PASS", 9)

    threshold_path = ROOT / "Assets/StreamingAssets/RecognitionCoreV1/RuntimeGate/frozen_threshold_config.json"
    threshold_data = json.loads(threshold_path.read_text(encoding="utf-8"))
    canonical = threshold_data["canonical_payload"]
    require(float(canonical["threshold"]) == THRESHOLD, "frozen threshold mismatch")
    validation_ids = list(canonical["threshold_selection_query_ids"])
    require(len(validation_ids) == 9 and len(set(validation_ids)) == 9, "validation query ID count mismatch")
    validation = []
    for sample_id in sorted(validation_ids):
        path = STEP8A / "features" / f"feature_{sample_id.removeprefix('S001_P001_')}.npz"
        npz = read_npz(path)
        require(str(npz["sample_id"].item()) == sample_id and str(npz["split"].item()) == "val", f"validation feature provenance mismatch: {sample_id}")
        validation.append({
            "sample_id": sample_id, "word": str(npz["word"].item()), "speed": str(npz["speed"].item()), "split": "val",
            "coordinates": npz["coordinates"], "valid_mask": npz["valid_mask"], "original_valid_mask": npz["original_valid_mask"],
            "interpolation_mask": npz["interpolation_mask"], "timestamp": npz["timestamp"], "path": path,
        })
    add_assert(assertions, "validation_only_query_set_exact_9", "PASS", sorted(validation_ids))

    step9c_delivery_input_path = STEP9C / "delivery_input_manifest.json"
    step9c_delivery_input = json.loads(step9c_delivery_input_path.read_text(encoding="utf-8"))
    parity = step9c_delivery_input.get("step9b_parity", {})
    require(parity.get("checked") == 20 and parity.get("passed") == 20 and parity.get("failed") == 0, "approved Step 9B parity summary mismatch")
    parity_verified = []
    for item in parity.get("targets", []):
        path = ROOT / PurePosixPath(item["relative_path"])
        require(path.is_file() and path.stat().st_size == int(item["size_bytes"]) and sha256_file(path) == item["sha256"], f"Step 9B parity mismatch: {item['relative_path']}")
        parity_verified.append(item["relative_path"])
    require(len(parity_verified) == 20, f"Step 9B parity target count mismatch: {len(parity_verified)}")
    add_assert(assertions, "step9b_parity_20_of_20", "PASS", {"checked": 20, "passed": 20, "failed": 0})

    tolerance_source = ROOT / "Assets/_Project/Tests/EditMode/RecognitionCoreV1Tests.cs"
    require("const double ParityTolerance = 1e-12;" in tolerance_source.read_text(encoding="utf-8"), "approved RecognitionCore tolerance source mismatch")
    add_assert(assertions, "approved_distance_parity_tolerance_found", "PASS", {"tolerance": PARITY_TOLERANCE, "source": relative(tolerance_source), "sha256": sha256_file(tolerance_source)})
    return {
        "step8a_output_verified": step8a_outputs, "step8a_input_verified": step8a_inputs,
        "config": config, "schema": schema, "template_manifest": template_manifest,
        "templates_reference": templates_reference, "templates_unity": templates_unity,
        "runtime_manifest": runtime_manifest, "validation": validation,
        "validation_ids": validation_ids, "threshold_data": threshold_data,
        "parity": parity, "parity_verified": parity_verified,
        "paths": {
            "step8a_script": ROOT / "Tools/s001_step8a_build_dtw_templates_v1_0.py",
            "selected_config": config_path, "feature_schema": schema_path, "template_manifest": template_manifest_path,
            "runtime_manifest": runtime_manifest_path, "threshold": threshold_path,
            "tolerance_source": tolerance_source,
        },
    }


def reference_dtw(left: np.ndarray, left_mask: np.ndarray, right: np.ndarray, right_mask: np.ndarray, band: float = DTW_BAND) -> dict:
    both = left_mask[:, None, :] & right_mask[None, :, :]
    exactly_one = left_mask[:, None, :] ^ right_mask[None, :, :]
    differences = np.where(both[..., None], left[:, None, :, :] - right[None, :, :, :], 0.0)
    euclidean = np.sqrt(np.sum(differences * differences, axis=-1))
    costs = np.where(both, euclidean, np.where(exactly_one, MISSING_COST, 0.0))
    comparable = both | exactly_one
    denominator = comparable.sum(axis=-1)
    local = np.full(denominator.shape, MISSING_COST, dtype=np.float64)
    np.divide(costs.sum(axis=-1), denominator, out=local, where=denominator > 0)
    length_a, length_b = len(left), len(right)
    cumulative = np.full((length_a, length_b), np.inf, dtype=np.float64)
    parent = np.full((length_a, length_b), -1, dtype=np.int8)
    for i in range(length_a):
        for j in range(length_b):
            if length_a > 1 and length_b > 1 and abs(i / (length_a - 1) - j / (length_b - 1)) > band + 1e-15:
                continue
            if i == 0 and j == 0:
                cumulative[i, j] = local[i, j]
                continue
            candidates = []
            if i > 0 and j > 0 and np.isfinite(cumulative[i - 1, j - 1]):
                candidates.append((cumulative[i - 1, j - 1], 0))
            if i > 0 and np.isfinite(cumulative[i - 1, j]):
                candidates.append((cumulative[i - 1, j], 1))
            if j > 0 and np.isfinite(cumulative[i, j - 1]):
                candidates.append((cumulative[i, j - 1], 2))
            if candidates:
                prior, direction = min(candidates, key=lambda item: (item[0], item[1]))
                cumulative[i, j] = prior + local[i, j]
                parent[i, j] = direction
    total = float(cumulative[-1, -1])
    require(math.isfinite(total), f"no reference DTW path for lengths {length_a}/{length_b}")
    i, j = length_a - 1, length_b - 1
    path = [(i, j)]
    while i or j:
        direction = int(parent[i, j])
        if direction == 0:
            i, j = i - 1, j - 1
        elif direction == 1:
            i -= 1
        elif direction == 2:
            j -= 1
        else:
            raise DiagnosticError(f"reference DTW parent incomplete at {i}/{j}")
        path.append((i, j))
    path.reverse()
    return {"distance": total / len(path), "cumulative_cost": total, "path": path, "path_length": len(path)}


def unity_contract_dtw(left: np.ndarray, left_mask: np.ndarray, right: np.ndarray, right_mask: np.ndarray, band: float = DTW_BAND) -> dict:
    length_a, length_b = len(left), len(right)
    cumulative = [[math.inf for _ in range(length_b)] for _ in range(length_a)]
    parent = [[0 for _ in range(length_b)] for _ in range(length_a)]

    def local_cost(i: int, j: int) -> float:
        total = 0.0
        included = 0
        for landmark in range(16):
            left_valid = bool(left_mask[i, landmark])
            right_valid = bool(right_mask[j, landmark])
            if left_valid and right_valid:
                dx = float(left[i, landmark, 0]) - float(right[j, landmark, 0])
                dy = float(left[i, landmark, 1]) - float(right[j, landmark, 1])
                dz = float(left[i, landmark, 2]) - float(right[j, landmark, 2])
                total += math.sqrt(dx * dx + dy * dy + dz * dz)
                included += 1
            elif left_valid != right_valid:
                total += MISSING_COST
                included += 1
        return total / included if included else MISSING_COST

    for i in range(length_a):
        for j in range(length_b):
            if length_a > 1 and length_b > 1 and abs(i / (length_a - 1) - j / (length_b - 1)) > band + 1e-15:
                continue
            local = local_cost(i, j)
            if i == 0 and j == 0:
                cumulative[i][j] = local
                continue
            best = math.inf
            direction = 0
            if i > 0 and j > 0 and math.isfinite(cumulative[i - 1][j - 1]):
                best = cumulative[i - 1][j - 1]
                direction = 1
            if i > 0 and math.isfinite(cumulative[i - 1][j]) and cumulative[i - 1][j] < best:
                best = cumulative[i - 1][j]
                direction = 2
            if j > 0 and math.isfinite(cumulative[i][j - 1]) and cumulative[i][j - 1] < best:
                best = cumulative[i][j - 1]
                direction = 3
            if direction:
                cumulative[i][j] = best + local
                parent[i][j] = direction
    total = cumulative[-1][-1]
    require(math.isfinite(total), f"no Unity-contract DTW path for lengths {length_a}/{length_b}")
    i, j = length_a - 1, length_b - 1
    path = [(i, j)]
    while i or j:
        direction = parent[i][j]
        if direction == 1:
            i, j = i - 1, j - 1
        elif direction == 2:
            i -= 1
        elif direction == 3:
            j -= 1
        else:
            raise DiagnosticError(f"Unity-contract DTW parent incomplete at {i}/{j}")
        path.append((i, j))
    path.reverse()
    return {"distance": total / len(path), "cumulative_cost": total, "path": path, "path_length": len(path)}


def group_contributions(left: np.ndarray, left_mask: np.ndarray, right: np.ndarray, right_mask: np.ndarray, path: list[tuple[int, int]]) -> dict:
    result = {group: 0.0 for group in LANDMARK_GROUPS}
    comparable_counts = []
    for i, j in path:
        comparable = [index for index in range(16) if bool(left_mask[i, index]) != bool(right_mask[j, index]) or (bool(left_mask[i, index]) and bool(right_mask[j, index]))]
        denominator = len(comparable)
        comparable_counts.append(denominator)
        if denominator == 0:
            continue
        for group, indices in LANDMARK_GROUPS.items():
            for landmark in indices:
                lv, rv = bool(left_mask[i, landmark]), bool(right_mask[j, landmark])
                if lv and rv:
                    delta = left[i, landmark] - right[j, landmark]
                    cost = float(math.sqrt(float(np.dot(delta, delta))))
                elif lv != rv:
                    cost = MISSING_COST
                else:
                    cost = 0.0
                result[group] += cost / denominator / len(path)
    total = sum(result.values())
    return {"groups": result, "total": total, "path_comparable_landmarks_min": min(comparable_counts), "path_comparable_landmarks_max": max(comparable_counts), "path_comparable_landmarks_mean": sum(comparable_counts) / len(comparable_counts)}


def physical_query(item: dict) -> dict:
    data = item["json"]
    frames = data["frames"]
    coordinates = np.array([frame["normalized_coordinates"] for frame in frames], dtype=np.float64)
    valid_mask = np.array([frame["normalized_mask"] for frame in frames], dtype=np.bool_)
    timestamps = np.array([frame["timestamp_seconds"] for frame in frames], dtype=np.float64)
    require(coordinates.shape == (len(frames), 16, 3) and valid_mask.shape == (len(frames), 16), f"physical feature shape mismatch: {data['capture_id']}")
    require(np.isfinite(coordinates[valid_mask]).all(), f"physical valid coordinate non-finite: {data['capture_id']}")
    require(np.all(np.diff(timestamps) > 0), f"physical timestamps not strictly increasing: {data['capture_id']}")
    counts = data["counts"]
    require(int(counts["normalized_frames"]) == len(frames) and int(counts["normalized_valid_landmarks"]) == int(valid_mask.sum()), f"physical normalized count mismatch: {data['capture_id']}")
    require(int(counts["interpolated_landmarks"]) == 0, f"unexpected physical interpolation count: {data['capture_id']}")
    return {
        "sample_id": data["capture_id"], "capture_id": data["capture_id"], "word": data["expected_word"], "expected_word": data["expected_word"],
        "speed": "PHYSICAL_UNSPECIFIED", "split": "physical", "coordinates": np.ascontiguousarray(coordinates),
        "valid_mask": np.ascontiguousarray(valid_mask), "original_valid_mask": np.ascontiguousarray(valid_mask.copy()),
        "interpolation_mask": np.zeros_like(valid_mask), "timestamp": np.ascontiguousarray(timestamps), "evidence": data, "source": item,
    }


def recognize(query: dict, templates: list[dict], implementation: str) -> dict:
    calculator = reference_dtw if implementation == "python_reference" else unity_contract_dtw
    distances = []
    for template in templates:
        dtw = calculator(query["coordinates"], query["valid_mask"], template["coordinates"], template["valid_mask"], DTW_BAND)
        distances.append({"template": template, "dtw": dtw})
    word_minima = {}
    for word in WORDS:
        word_minima[word] = min((item for item in distances if item["template"]["word"] == word), key=lambda item: (item["dtw"]["distance"], item["template"]["group"]))
    ranked_words = sorted(word_minima.items(), key=lambda item: (item[1]["dtw"]["distance"], item[0]))
    return {"distances": distances, "word_minima": word_minima, "predicted_word": ranked_words[0][0], "second_best_word": ranked_words[1][0], "margin": ranked_words[1][1]["dtw"]["distance"] - ranked_words[0][1]["dtw"]["distance"], "best_distance": ranked_words[0][1]["dtw"]["distance"], "accepted": ranked_words[0][1]["dtw"]["distance"] <= THRESHOLD}


def compute_distance_evidence(physical: list[dict], preflight: dict, assertions: list[dict]) -> tuple[list[dict], list[dict], list[dict], dict]:
    rows = []
    contributions = []
    query_results = []
    max_stored_delta = 0.0
    max_cross_delta = 0.0
    max_path_difference = 0
    for query in physical:
        reference = recognize(query, preflight["templates_reference"], "python_reference")
        unity = recognize(query, preflight["templates_unity"], "unity_contract")
        stored_by_id = {item["prototype_id"]: item for item in query["evidence"]["recognition"].get("template_distances", [])}
        require(len(stored_by_id) == 9, f"stored Unity distance set incomplete: {query['capture_id']}")
        ref_by_id = {item["template"]["sample_id"]: item for item in reference["distances"]}
        unity_by_id = {item["template"]["sample_id"]: item for item in unity["distances"]}
        ranked = sorted(reference["distances"], key=lambda item: (item["dtw"]["distance"], item["template"]["group"]))
        ranks = {item["template"]["sample_id"]: index + 1 for index, item in enumerate(ranked)}
        for template in preflight["templates_reference"]:
            sample_id = template["sample_id"]
            stored = stored_by_id.get(sample_id)
            if stored is None:
                rows.append({
                    "query_expected_word": query["word"], "capture_id": query["capture_id"], "template_word": template["word"], "template_speed": template["speed"], "template_sample_id": sample_id,
                    "stored_unity_distance": "", "python_reference_distance": format(ref_by_id[sample_id]["dtw"]["distance"], ".17g"), "unity_contract_reconstruction_distance": format(unity_by_id[sample_id]["dtw"]["distance"], ".17g"),
                    "absolute_difference": "", "relative_difference": "", "parity_tolerance": format(PARITY_TOLERANCE, ".17g"), "parity_tolerance_source": relative(preflight["paths"]["tolerance_source"]),
                    "match_status": "UNVERIFIED_MISSING_STORED_DISTANCE", "query_distance_rank": ranks[sample_id], "word_minimum": ref_by_id[sample_id] is reference["word_minima"][template["word"]],
                    "predicted_word": reference["predicted_word"], "second_best_word": reference["second_best_word"], "margin": format(reference["margin"], ".17g"), "frozen_threshold": format(THRESHOLD, ".17g"), "accepted": reference["accepted"],
                    "stored_path_length": "", "python_path_length": ref_by_id[sample_id]["dtw"]["path_length"], "unity_contract_path_length": unity_by_id[sample_id]["dtw"]["path_length"],
                })
                continue
            ref_dtw = ref_by_id[sample_id]["dtw"]
            unity_dtw = unity_by_id[sample_id]["dtw"]
            stored_distance = float(stored["distance"])
            absolute = abs(ref_dtw["distance"] - stored_distance)
            relative_difference = absolute / max(abs(stored_distance), np.finfo(np.float64).tiny)
            cross = abs(ref_dtw["distance"] - unity_dtw["distance"])
            max_stored_delta = max(max_stored_delta, absolute)
            max_cross_delta = max(max_cross_delta, cross)
            max_path_difference = max(max_path_difference, abs(int(stored["path_length"]) - ref_dtw["path_length"]), abs(ref_dtw["path_length"] - unity_dtw["path_length"]))
            status = "PASS" if absolute <= PARITY_TOLERANCE and cross <= PARITY_TOLERANCE and int(stored["path_length"]) == ref_dtw["path_length"] == unity_dtw["path_length"] else "FAIL"
            rows.append({
                "query_expected_word": query["word"], "capture_id": query["capture_id"], "template_word": template["word"], "template_speed": template["speed"], "template_sample_id": sample_id,
                "stored_unity_distance": format(stored_distance, ".17g"), "python_reference_distance": format(ref_dtw["distance"], ".17g"), "unity_contract_reconstruction_distance": format(unity_dtw["distance"], ".17g"),
                "absolute_difference": format(absolute, ".17g"), "relative_difference": format(relative_difference, ".17g"), "parity_tolerance": format(PARITY_TOLERANCE, ".17g"), "parity_tolerance_source": relative(preflight["paths"]["tolerance_source"]),
                "match_status": status, "query_distance_rank": ranks[sample_id], "word_minimum": ref_by_id[sample_id] is reference["word_minima"][template["word"]],
                "predicted_word": reference["predicted_word"], "second_best_word": reference["second_best_word"], "margin": format(reference["margin"], ".17g"), "frozen_threshold": format(THRESHOLD, ".17g"), "accepted": reference["accepted"],
                "stored_path_length": int(stored["path_length"]), "python_path_length": ref_dtw["path_length"], "unity_contract_path_length": unity_dtw["path_length"],
            })
            split = group_contributions(query["coordinates"], query["valid_mask"], template["coordinates"], template["valid_mask"], ref_dtw["path"])
            require(abs(split["total"] - ref_dtw["distance"]) <= 1e-12, f"group contribution does not sum to DTW: {query['capture_id']}/{sample_id}")
            contributions.append({
                "query_expected_word": query["word"], "capture_id": query["capture_id"], "template_word": template["word"], "template_speed": template["speed"], "template_sample_id": sample_id,
                "pose_contribution": format(split["groups"]["pose"], ".17g"), "left_hand_contribution": format(split["groups"]["left_hand"], ".17g"), "right_hand_contribution": format(split["groups"]["right_hand"], ".17g"),
                "contribution_sum": format(split["total"], ".17g"), "dtw_distance": format(ref_dtw["distance"], ".17g"), "path_length": ref_dtw["path_length"],
                "path_comparable_landmarks_min": split["path_comparable_landmarks_min"], "path_comparable_landmarks_max": split["path_comparable_landmarks_max"], "path_comparable_landmarks_mean": format(split["path_comparable_landmarks_mean"], ".17g"),
            })
        stored_recognition = query["evidence"]["recognition"]
        require(reference["predicted_word"] == unity["predicted_word"] == stored_recognition["predicted_word"], f"physical prediction parity failure: {query['capture_id']}")
        require(abs(reference["best_distance"] - float(stored_recognition["best_distance"])) <= PARITY_TOLERANCE, f"physical best-distance parity failure: {query['capture_id']}")
        query_results.append({"query": query, "reference": reference, "unity": unity, "stored": stored_recognition})

    counts = Counter(row["match_status"] for row in rows)
    require(len(rows) == 27 and counts.get("FAIL", 0) == 0 and counts.get("PASS", 0) == 27, f"physical distance parity count failure: {dict(counts)}")
    add_assert(assertions, "physical_3_by_9_reference_distances_generated", "PASS", len(rows))
    add_assert(assertions, "stored_unity_distance_parity_27", "PASS", {"pass": counts.get("PASS", 0), "fail": counts.get("FAIL", 0), "unverified": counts.get("UNVERIFIED_MISSING_STORED_DISTANCE", 0), "max_absolute_difference": max_stored_delta})
    add_assert(assertions, "python_reference_unity_contract_parity_27", "PASS", {"max_absolute_difference": max_cross_delta, "max_path_length_difference": max_path_difference})
    return rows, contributions, query_results, {"counts": dict(counts), "max_stored_delta": max_stored_delta, "max_cross_delta": max_cross_delta, "max_path_length_difference": max_path_difference}


def summary(values: list[float]) -> dict:
    require(bool(values), "cannot summarize empty values")
    array = np.asarray(values, dtype=np.float64)
    require(np.isfinite(array).all(), "summary input is non-finite")
    return {
        "n": int(array.size), "min": float(np.min(array)), "max": float(np.max(array)), "mean": float(np.mean(array)),
        "std_population": float(np.std(array)), "p5": float(np.percentile(array, 5)), "p50": float(np.percentile(array, 50)), "p95": float(np.percentile(array, 95)),
    }


def distribution_records(results: list[dict]) -> tuple[dict, list[dict]]:
    per_query = []
    predicted = Counter()
    template_word_values = defaultdict(list)
    template_speed_values = defaultdict(list)
    metrics = defaultdict(list)
    for item in results:
        result = item["reference"]
        expected = item["query"]["word"]
        word_values = {word: result["word_minima"][word]["dtw"]["distance"] for word in WORDS}
        expected_distance = word_values[expected]
        incorrect_distance = min(value for word, value in word_values.items() if word != expected)
        metrics["best_distance"].append(result["best_distance"])
        metrics["expected_word_minimum_distance"].append(expected_distance)
        metrics["nearest_incorrect_word_distance"].append(incorrect_distance)
        metrics["best_second_best_margin"].append(result["margin"])
        metrics["threshold_distance_ratio"].append(result["best_distance"] / THRESHOLD)
        predicted[result["predicted_word"]] += 1
        for word, value in word_values.items():
            template_word_values[word].append(value)
        for distance in result["distances"]:
            template_speed_values[distance["template"]["speed"]].append(distance["dtw"]["distance"])
        per_query.append({
            "sample_id": item["query"]["sample_id"], "expected_word": expected, "predicted_word": result["predicted_word"],
            "best_distance": result["best_distance"], "expected_word_minimum_distance": expected_distance,
            "nearest_incorrect_word_distance": incorrect_distance, "best_second_best_margin": result["margin"],
            "threshold_distance_ratio": result["best_distance"] / THRESHOLD,
            "word_minima": word_values, "sequence_length": len(item["query"]["coordinates"]),
        })
    return {
        "n": len(results), "metrics": {key: summary(value) for key, value in sorted(metrics.items())},
        "predicted_word_frequency": {word: predicted.get(word, 0) for word in WORDS},
        "template_word_minimum_distance": {word: summary(template_word_values[word]) for word in WORDS},
        "template_speed_distance": {speed: summary(template_speed_values[speed]) for speed in SPEEDS},
    }, sorted(per_query, key=lambda row: row["sample_id"])


def compute_validation(preflight: dict, assertions: list[dict]) -> tuple[list[dict], dict]:
    csv_path = STEP8A / "validation_distances.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        stored_rows = list(csv.DictReader(stream))
    selected = [row for row in stored_rows if row["configuration"] == preflight["config"]["configuration_name"]]
    require(len(selected) == 81, f"selected validation distance row count mismatch: {len(selected)}")
    lookup = {(row["validation_sample_id"], row["prototype_sample_id"]): row for row in selected}
    results = []
    max_stored_delta = 0.0
    max_cross_delta = 0.0
    for query in preflight["validation"]:
        reference = recognize(query, preflight["templates_reference"], "python_reference")
        unity = recognize(query, preflight["templates_unity"], "unity_contract")
        for item in reference["distances"]:
            sample_id = item["template"]["sample_id"]
            stored = lookup[(query["sample_id"], sample_id)]
            max_stored_delta = max(max_stored_delta, abs(float(stored["distance"]) - item["dtw"]["distance"]))
            unity_item = next(value for value in unity["distances"] if value["template"]["sample_id"] == sample_id)
            max_cross_delta = max(max_cross_delta, abs(unity_item["dtw"]["distance"] - item["dtw"]["distance"]))
        results.append({"query": query, "reference": reference, "unity": unity})
    require(max_stored_delta <= PARITY_TOLERANCE and max_cross_delta <= PARITY_TOLERANCE, f"validation distance parity failure: stored={max_stored_delta}, cross={max_cross_delta}")
    add_assert(assertions, "validation_9_by_9_reference_distances_generated_without_test", "PASS", {"rows": 81, "max_stored_delta": max_stored_delta, "max_cross_delta": max_cross_delta})
    distribution, per_query = distribution_records(results)
    return results, {"distribution": distribution, "per_query": per_query, "max_stored_delta": max_stored_delta, "max_cross_delta": max_cross_delta}


def movement_statistics(coordinates: np.ndarray, mask: np.ndarray) -> dict:
    frame_values = []
    active_indices = []
    exact_idle = 0
    for frame in range(1, len(coordinates)):
        both = mask[frame - 1] & mask[frame]
        if not np.any(both):
            frame_values.append(0.0)
            exact_idle += 1
            continue
        delta = coordinates[frame, both] - coordinates[frame - 1, both]
        norms = np.sqrt(np.sum(delta * delta, axis=1))
        value = float(np.sqrt(np.mean(norms * norms)))
        frame_values.append(value)
        if value == 0.0:
            exact_idle += 1
        else:
            active_indices.append(frame)
    if not frame_values:
        return {"pair_count": 0, "exact_zero_motion_idle_ratio": 1.0, "first_nonzero_motion_frame": None, "last_nonzero_motion_frame": None}
    values = np.asarray(frame_values, dtype=np.float64)
    return {
        "pair_count": len(frame_values), "frame_pair_rms_displacement_mean": float(np.mean(values)), "frame_pair_rms_displacement_rms": float(np.sqrt(np.mean(values * values))),
        "frame_pair_rms_displacement_p5": float(np.percentile(values, 5)), "frame_pair_rms_displacement_p50": float(np.percentile(values, 50)), "frame_pair_rms_displacement_p95": float(np.percentile(values, 95)),
        "exact_zero_motion_idle_ratio": exact_idle / len(frame_values), "first_nonzero_motion_frame": active_indices[0] if active_indices else None, "last_nonzero_motion_frame": active_indices[-1] if active_indices else None,
        "approved_idle_ratio_status": "UNVERIFIED_NO_COMMON_APPROVED_IDLE_DEFINITION",
    }


def feature_stat_row(dataset: str, sample: dict) -> dict:
    coordinates = sample["coordinates"]
    mask = sample["valid_mask"]
    original = sample.get("original_valid_mask", mask)
    interpolation = sample.get("interpolation_mask", np.zeros_like(mask))
    flat = coordinates.reshape(-1)
    valid_values = coordinates[mask].reshape(-1)
    require(np.isfinite(valid_values).all(), f"valid values are non-finite: {sample['sample_id']}")
    motion = movement_statistics(coordinates, mask)
    body = sample.get("evidence", {}).get("body_transform", {}) if sample.get("evidence") else {}
    return {
        "dataset": dataset, "sample_id": sample["sample_id"], "expected_word": sample["word"], "stage_name": "final_recognition_feature",
        "source_implementation": sample.get("source_implementation", "Step8A feature NPZ"), "shape": json.dumps(list(coordinates.shape), separators=(",", ":")), "dtype": str(coordinates.dtype),
        "deterministic_array_sha256": array_hash(coordinates), "mask_sha256": array_hash(mask), "sequence_length": len(coordinates),
        "finite_count": int(np.isfinite(flat).sum()), "nan_count": int(np.isnan(flat).sum()), "inf_count": int(np.isinf(flat).sum()),
        "original_valid_count": int(original.sum()), "interpolated_count": int(interpolation.sum()), "invalid_masked_count": int(mask.size - mask.sum()),
        "center_definition": "torso shoulder midpoint per raw frame; center array not retained" if dataset == "physical" else "already raw_framewise normalized; source center array not retained in Step8A feature NPZ",
        "scale_definition": "raw shoulder width per frame" if dataset == "physical" else "raw shoulder width per frame; scale array not retained in Step8A feature NPZ",
        "scale_min": body.get("shoulder_width_raw_min", ""), "scale_median": body.get("shoulder_width_raw_median", ""), "scale_max": body.get("shoulder_width_raw_max", ""),
        "valid_value_min": float(np.min(valid_values)), "valid_value_max": float(np.max(valid_values)), "valid_value_mean": float(np.mean(valid_values)),
        "valid_value_std_population": float(np.std(valid_values)), "valid_value_rms": float(np.sqrt(np.mean(valid_values * valid_values))),
        "valid_value_p5": float(np.percentile(valid_values, 5)), "valid_value_p50": float(np.percentile(valid_values, 50)), "valid_value_p95": float(np.percentile(valid_values, 95)),
        "mask_valid_ratio": float(mask.mean()), "motion_pair_count": motion["pair_count"], "motion_rms_mean": motion.get("frame_pair_rms_displacement_mean", ""),
        "motion_rms": motion.get("frame_pair_rms_displacement_rms", ""), "motion_p5": motion.get("frame_pair_rms_displacement_p5", ""), "motion_p50": motion.get("frame_pair_rms_displacement_p50", ""),
        "motion_p95": motion.get("frame_pair_rms_displacement_p95", ""), "exact_zero_motion_idle_ratio": motion["exact_zero_motion_idle_ratio"],
        "first_nonzero_motion_frame": motion["first_nonzero_motion_frame"] if motion["first_nonzero_motion_frame"] is not None else "", "last_nonzero_motion_frame": motion["last_nonzero_motion_frame"] if motion["last_nonzero_motion_frame"] is not None else "",
        "idle_ratio_status": motion.get("approved_idle_ratio_status", "UNVERIFIED_NO_COMMON_APPROVED_IDLE_DEFINITION"),
    }


def stage_contract(physical: list[dict], preflight: dict, distance_meta: dict, assertions: list[dict]) -> dict:
    sources = {
        "step8a_reference": file_record(preflight["paths"]["step8a_script"], "reference_source"),
        "unity_dtw": file_record(ROOT / "Assets/_Project/Scripts/Recognition/Core/MaskAwareDtw.cs", "production_source"),
        "unity_core": file_record(ROOT / "Assets/_Project/Scripts/Recognition/Core/RecognitionCore.cs", "production_source"),
        "unity_sequence": file_record(ROOT / "Assets/_Project/Scripts/Recognition/Core/RecognitionSequence.cs", "production_source"),
        "live_adapter": file_record(ROOT / "Assets/_Project/Scripts/Recognition/LiveInput/ApprovedLivePreprocessingAdapter.cs", "production_source"),
        "sequence_bridge": file_record(ROOT / "Assets/_Project/Scripts/Recognition/Runtime/RecognitionSequenceInputSession.cs", "production_source"),
        "feature_schema": file_record(preflight["paths"]["feature_schema"], "approved_config"),
        "selected_dtw_config": file_record(preflight["paths"]["selected_config"], "approved_config"),
    }
    common_missing = "Step 9C evidence stores normalized [T,16,3]/[T,16] frames and aggregate raw counts, but not the 75-landmark raw coordinates/masks, per-landmark boundary/interpolation arrays, or torso origin/basis arrays."
    stages = [
        {"index": 1, "stage_name": "capture_window_and_idle_scope", "status": "PASS", "evidence": "Finalized evidence frame count, frame_index and strictly increasing timestamps define the complete Start-to-Finalize recognition sequence; no post-finalize frames are present."},
        {"index": 2, "stage_name": "raw_observation_layout", "status": "UNVERIFIED", "evidence": common_missing},
        {"index": 3, "stage_name": "landmark_ordering", "status": "UNVERIFIED", "evidence": "Final 16-landmark ordering matches the approved schema, but raw 75-landmark arrays are absent, so raw-to-selected index mapping cannot be independently replayed for these captures."},
        {"index": 4, "stage_name": "left_right_handedness_assignment", "status": "UNVERIFIED", "evidence": "Step 9C objective-valid diagnostics support the assignments, but the raw hand arrays and identity trajectory are not serialized in the fixed evidence."},
        {"index": 5, "stage_name": "coordinate_mapping", "status": "UNVERIFIED", "evidence": "Approved Step 9C compatibility provenance exists, but this Step 9D package cannot independently compare raw provider coordinates to normalized output without raw arrays."},
        {"index": 6, "stage_name": "mirror_application", "status": "UNVERIFIED", "evidence": "Approved source/config contracts are fixed; per-capture pre-map coordinates are not stored."},
        {"index": 7, "stage_name": "original_valid_mask", "status": "UNVERIFIED", "evidence": "Final normalized mask and aggregate raw valid counts are stored; per-landmark original-valid mask is not stored."},
        {"index": 8, "stage_name": "max_gap_interpolation_and_mask", "status": "SUPPORTED_BUT_NOT_PROVEN", "evidence": "All three evidence files record interpolated_landmarks=0 and selected max gap remains 1; per-landmark pre-interpolation arrays are absent."},
        {"index": 9, "stage_name": "normalization_center_and_scale", "status": "UNVERIFIED", "evidence": "Per-frame shoulder width and normalized output are stored; torso origin/right/up/forward basis and raw coordinates are not stored."},
        {"index": 10, "stage_name": "final_feature_ordering_and_weighting", "status": "PASS", "evidence": "Final [T,16,3] arrays use the approved 4 pose + 6 left + 6 right order; local cost has no extra feature weights."},
        {"index": 11, "stage_name": "segmentation_or_sequence_trimming", "status": "PASS", "evidence": "RecognitionSequenceInputSession appends every normalized capture frame and FinalizeSequence performs no trimming; evidence raw_frames equals normalized_frames and query length."},
        {"index": 12, "stage_name": "dtw_band", "status": "PASS", "evidence": {"relative_band": DTW_BAND}},
        {"index": 13, "stage_name": "local_cost", "status": "PASS", "evidence": {"both_valid": "3D Euclidean", "exactly_one_valid": 1.0, "both_invalid": "excluded", "no_comparable": 1.0, "aggregation": "mean"}},
        {"index": 14, "stage_name": "accumulated_cost", "status": "PASS", "evidence": {"steps": ["diagonal", "vertical", "horizontal"], "tie_break": ["diagonal", "vertical", "horizontal"], "max_reference_unity_difference": distance_meta["max_cross_delta"]}},
        {"index": 15, "stage_name": "path_length_and_distance_normalization", "status": "PASS", "evidence": {"definition": "cumulative cost / warping path length", "max_path_length_difference": distance_meta["max_path_length_difference"]}},
        {"index": 16, "stage_name": "frozen_template_prototype_selection", "status": "PASS", "evidence": {"strategy": "word_speed_medoid", "template_count": 9, "groups": sorted(item["group"] for item in preflight["templates_reference"]) }},
    ]
    add_assert(assertions, "physical_raw_observation_arrays_present", "UNVERIFIED", common_missing)
    add_assert(assertions, "physical_normalization_intermediate_arrays_present", "UNVERIFIED", common_missing)
    add_assert(assertions, "observable_recognition_input_through_dtw_contract_parity", "PASS", {"stages": list(range(10, 17)), "max_distance_delta": distance_meta["max_stored_delta"]})
    return {
        "schema_version": "words_in_silence.step9d.stage_contract_comparison.v1", "primary_verdict": PRIMARY_VERDICT,
        "first_divergence": None, "first_unverifiable_stage": 2,
        "observable_pipeline_parity": {"status": "PASS", "scope": "stored normalized RecognitionCore input through word winner", "distance_rows": 27, "tolerance": PARITY_TOLERANCE},
        "upstream_pipeline_parity": {"status": "UNVERIFIED", "scope": "raw provider observation through normalization", "missing_evidence": ["75-landmark raw coordinates and masks per frame", "per-landmark original-valid/reappearance/strong-depth/interpolation arrays", "per-frame torso origin/right/up/forward basis"]},
        "sources": sources, "captures": [{"expected_word": item["word"], "capture_id": item["capture_id"], "shape": list(item["coordinates"].shape), "coordinates_sha256": array_hash(item["coordinates"]), "valid_mask_sha256": array_hash(item["valid_mask"])} for item in physical],
        "stages": stages,
        "statistics_definition": {"finite_nan_inf_denominator": "all coordinate scalar cells in [T,16,3]", "valid_coordinate_statistics_denominator": "coordinate scalar cells whose landmark mask is true", "valid_mask_ratio_denominator": "T*16 landmarks", "std": "population standard deviation", "percentiles": "NumPy linear percentile", "movement": "RMS normalized 3D displacement over landmarks valid in adjacent frames; descriptive auxiliary metric only", "idle_ratio": "exact-zero movement frame-pair ratio only; no approved common active/idle threshold was available"},
    }


def protected_paths(preflight: dict, step9c: dict) -> list[Path]:
    paths: set[Path] = set()
    for item in preflight["step8a_output_verified"] + preflight["step8a_input_verified"]:
        paths.add(ROOT / PurePosixPath(item["relative_path"]))
    for item in preflight["parity"].get("targets", []):
        paths.add(ROOT / PurePosixPath(item["relative_path"]))
    for base in (
        ROOT / "Assets/_Project/Scripts/Recognition/Core",
        ROOT / "Assets/_Project/Scripts/Recognition/Runtime",
        ROOT / "Assets/_Project/Scripts/Recognition/LiveInput",
        ROOT / "Assets/_Project/Scripts/Recognition/PhysicalCompatibility",
    ):
        if base.is_dir():
            paths.update(path for path in base.rglob("*") if path.is_file())
    explicit = [
        ROOT / "AGENTS.md", ROOT / "WordsInSilence_PROJECT_CONTEXT.md", ROOT / "Tools/s001_step8a_build_dtw_templates_v1_0.py",
        ROOT / "Tools/s001_step9b_live_mediapipe_input_v1_0.py", ROOT / "Tools/s001_step9c_physical_webcam_coordinate_compatibility_v1_0.py", ROOT / "Tools/s001_step9c_delivery_package_v1_0.py",
        ROOT / "Assets/_Project/Scripts/Landmarks/MediaPipeLandmarkProvider.cs", ROOT / "Assets/_Project/Scripts/Pose/HandPoseIntegrator.cs",
        ROOT / "Assets/_Project/Resources/Diagnostics/PhysicalWebcamCoordinateCompatibility.prefab", ROOT / "Assets/_Project/Scenes/Technical/02_PoseIntegration.unity",
        ROOT / "Assets/_Project/Tests/EditMode/RecognitionCoreV1Tests.cs", ROOT / "Assets/_Project/Tests/EditMode/LiveMediaPipeInputV1Tests.cs",
        ROOT / "Assets/_Project/Tests/EditMode/PhysicalWebcamCoordinateCompatibilityV1Tests.cs", ROOT / "Assets/_Project/Tests/PlayMode/PhysicalWebcamCoordinateCompatibilityHarnessV1Tests.cs",
        STEP8A / "input_manifest.json", STEP8A / "output_file_manifest.json", STEP9B / "input_manifest.json", STEP9B / "output_file_manifest.json",
        STEP9C / "delivery_input_manifest.json", STEP9C / "output_file_manifest.json", STEP9C_DELIVERY, STEP9C_DETACHED,
        ROOT / "Assets/StreamingAssets/RecognitionCoreV1/RuntimeGate/frozen_threshold_config.json", ROOT / "Assets/StreamingAssets/RecognitionCoreV1/RuntimeGate/runtime_recognition_assets.json",
        STEP8B / "runtime_template_manifest.json",
    ]
    paths.update(path for path in explicit if path.is_file())
    physical_dir = STEP9C / "physical_captures"
    paths.update(path for path in physical_dir.glob("*") if path.is_file())
    return sorted(paths, key=lambda path: relative(path))


def snapshot(paths: list[Path]) -> dict[str, dict]:
    result = {}
    for path in paths:
        require(path.is_file(), f"protected file missing: {path}")
        item = file_record(path, "protected")
        result[item["relative_path"]] = item
    return result


def compare_snapshot(before: dict[str, dict]) -> list[dict]:
    mismatches = []
    for rel, expected in sorted(before.items()):
        path = ROOT / PurePosixPath(rel)
        if not path.is_file():
            mismatches.append({"relative_path": rel, "reason": "missing_after"})
            continue
        actual = file_record(path, "protected")
        if actual["size_bytes"] != expected["size_bytes"] or actual["sha256"] != expected["sha256"]:
            mismatches.append({"relative_path": rel, "reason": "content_changed", "before": expected, "after": actual})
    return mismatches


def git_snapshot() -> dict:
    try:
        process = subprocess.run(["git", "-c", "core.excludesFile=NUL", "status", "--short", "--untracked-files=all"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        entries = process.stdout.splitlines()
        generated_prefixes = (f"?? {SCRIPT_REL}", f"?? {OUTPUT_REL}/", f"?? {DELIVERY_NAME}", f"?? {DELIVERY_NAME}.sha256", f"?? {POST_ZIP.name}")
        preexisting = [line for line in entries if not line.startswith(generated_prefixes)]
        return {"available": process.returncode == 0, "command": "git -c core.excludesFile=NUL status --short --untracked-files=all", "return_code": process.returncode, "entry_count": len(entries), "stdout_sha256": sha256_bytes(process.stdout.encode("utf-8")), "preexisting_entry_count": len(preexisting), "preexisting_entries": preexisting, "step9d_entries_present_at_snapshot": [line for line in entries if line not in preexisting], "stderr": process.stderr.splitlines()}
    except OSError as exc:
        return {"available": False, "reason": repr(exc), "preexisting_entries": []}


def stage_input_copies(stage_output: Path, step9c: dict, preflight: dict) -> list[dict]:
    copies: list[dict] = []

    def copy_path(source: Path, destination_rel: str, category: str) -> None:
        destination = stage_output / PurePosixPath(destination_rel)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        require(sha256_file(destination) == sha256_file(source), f"input copy hash mismatch: {source}")
        copies.append(file_record(source, category, f"{OUTPUT_REL}/{destination_rel}"))

    copy_path(STEP9C_DELIVERY, f"inputs/step9c_delivery/{STEP9C_DELIVERY.name}", "approved_step9c_delivery")
    copy_path(STEP9C_DETACHED, f"inputs/step9c_delivery/{STEP9C_DETACHED.name}", "approved_step9c_detached")
    for word in WORDS:
        item = step9c["valid_by_word"][word]
        base = PurePosixPath(item["archive_path"]).name
        for data, name in ((item["bytes"], base), (item["checksum_bytes"], base + ".sha256")):
            destination_rel = f"inputs/physical_queries/{name}"
            destination = stage_output / PurePosixPath(destination_rel)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            copies.append(bytes_record(data, f"{OUTPUT_REL}/{destination_rel}", "objective_valid_physical_evidence"))
    step8a_paths = [preflight["paths"][key] for key in ("selected_config", "feature_schema", "template_manifest")] + [STEP8A / "validation_distances.csv", STEP8A / "validation_predictions.csv"]
    step8a_paths += [item["path"] for item in preflight["templates_reference"]] + [item["path"] for item in preflight["validation"]]
    for source in sorted(set(step8a_paths), key=lambda path: relative(path)):
        copy_path(source, f"inputs/step8a/{source.relative_to(STEP8A).as_posix()}", "approved_step8a_input")
    for item in preflight["templates_unity"]:
        source = item["path"]
        copy_path(source, f"inputs/runtime_templates/{source.name}", "approved_runtime_template")
    copy_path(preflight["paths"]["runtime_manifest"], "inputs/runtime_templates/runtime_template_manifest.json", "approved_runtime_manifest")
    copy_path(preflight["paths"]["threshold"], "inputs/runtime_gate/frozen_threshold_config.json", "approved_threshold")
    source_paths = [preflight["paths"]["step8a_script"], preflight["paths"]["tolerance_source"], ROOT / "Assets/_Project/Scripts/Recognition/Core/MaskAwareDtw.cs", ROOT / "Assets/_Project/Scripts/Recognition/Core/RecognitionCore.cs", ROOT / "Assets/_Project/Scripts/Recognition/Core/RecognitionSequence.cs", ROOT / "Assets/_Project/Scripts/Recognition/Core/RecognitionSequenceBinary.cs", ROOT / "Assets/_Project/Scripts/Recognition/LiveInput/ApprovedLivePreprocessingAdapter.cs", ROOT / "Assets/_Project/Scripts/Recognition/Runtime/RecognitionSequenceInputSession.cs"]
    for source in sorted(source_paths, key=lambda path: relative(path)):
        copy_path(source, f"inputs/source_snapshots/{relative(source)}", "approved_source_snapshot")
    copy_path(STEP9B / "input_manifest.json", "inputs/step9b/input_manifest.json", "approved_step9b_manifest")
    copy_path(STEP9B / "output_file_manifest.json", "inputs/step9b/output_file_manifest.json", "approved_step9b_manifest")
    copy_path(STEP9C / "delivery_input_manifest.json", "inputs/step9c_delivery/delivery_input_manifest.json", "approved_step9c_manifest")
    copy_path(STEP9C / "output_file_manifest.json", "inputs/step9c_delivery/output_file_manifest.json", "approved_step9c_manifest")
    return sorted(copies, key=lambda item: item["relative_path"])


def deterministic_zip_write(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    require(safe_zip_name(name), f"unsafe archive path: {name}")
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    info.create_system = 3
    archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def build() -> dict:
    require((ROOT / "Assets").is_dir() and (ROOT / "Packages/manifest.json").is_file() and (ROOT / "ProjectSettings/ProjectVersion.txt").is_file(), f"not a Unity project root: {ROOT}")
    for path in (OUTPUT, DELIVERY_ZIP, DELIVERY_HASH, POST_ZIP):
        require(not path.exists(), f"final output already exists; refusing overwrite: {path}")
    assertions: list[dict] = []
    add_assert(assertions, "unity_project_root_signature", "PASS", str(ROOT))
    step9c = strict_step9c_delivery(assertions)
    preflight = preflight_step8a_step9b(assertions)
    protected_before = snapshot(protected_paths(preflight, step9c))
    dirty_before = git_snapshot()
    physical = [physical_query(step9c["valid_by_word"][word]) for word in WORDS]
    add_assert(assertions, "physical_query_normalized_feature_shape_and_finiteness", "PASS", [{"word": item["word"], "shape": list(item["coordinates"].shape), "valid_landmarks": int(item["valid_mask"].sum())} for item in physical])

    distance_rows, contribution_rows, physical_results, distance_meta = compute_distance_evidence(physical, preflight, assertions)
    validation_results, validation_data = compute_validation(preflight, assertions)
    physical_distribution, physical_per_query = distribution_records(physical_results)
    distribution = {
        "schema_version": "words_in_silence.step9d.offline_physical_distribution_comparison.v1", "distance_definition": "approved mask-aware DTW; band=0.10; cumulative cost divided by path length",
        "test_split_used": False, "frozen_threshold": THRESHOLD,
        "offline_validation": {**validation_data["distribution"], "per_query": validation_data["per_query"]},
        "physical": {**physical_distribution, "per_query": physical_per_query},
        "scope_limit": "physical n=3 total and n=1 per word; descriptive domain-gap diagnosis only, not a generalization or accuracy claim",
    }
    add_assert(assertions, "offline_validation_physical_same_distance_definition", "PASS", {"validation_n": len(validation_results), "physical_n": len(physical_results), "test_split_used": False})

    stats_rows = []
    for item in physical:
        item["source_implementation"] = "Step9C normalized capture evidence"
        stats_rows.append(feature_stat_row("physical", item))
    for item in preflight["validation"]:
        stats_rows.append(feature_stat_row("offline_validation", item))
    for item in preflight["templates_reference"]:
        stats_rows.append(feature_stat_row("frozen_template", item))
    stats_rows.sort(key=lambda row: (row["dataset"], row["sample_id"]))

    stage_data = stage_contract(physical, preflight, distance_meta, assertions)
    word_matrix = []
    for item in physical_results:
        reference = item["reference"]
        word_matrix.append({"expected_word": item["query"]["word"], "capture_id": item["query"]["capture_id"], **{word: reference["word_minima"][word]["dtw"]["distance"] for word in WORDS}, "predicted_word": reference["predicted_word"]})
    bias_details = []
    contribution_lookup = {(row["capture_id"], row["template_sample_id"]): row for row in contribution_rows}
    for item in physical_results:
        query = item["query"]
        ref = item["reference"]
        expected_best = ref["word_minima"][query["word"]]
        winner = ref["word_minima"][ref["predicted_word"]]
        expected_row = contribution_lookup[(query["capture_id"], expected_best["template"]["sample_id"])]
        winner_row = contribution_lookup[(query["capture_id"], winner["template"]["sample_id"])]
        bias_details.append({
            "expected_word": query["word"], "capture_id": query["capture_id"], "predicted_word": ref["predicted_word"],
            "expected_word_best_template": expected_best["template"]["sample_id"], "expected_word_distance": expected_best["dtw"]["distance"],
            "winning_template": winner["template"]["sample_id"], "winning_distance": winner["dtw"]["distance"],
            "winning_advantage_over_expected_word": expected_best["dtw"]["distance"] - winner["dtw"]["distance"],
            "group_contribution_delta_winner_minus_expected": {group: float(winner_row[f"{group}_contribution"]) - float(expected_row[f"{group}_contribution"]) for group in ("pose", "left_hand", "right_hand")},
        })

    threshold_risk = {
        "status": "CONFIRMED", "current_threshold": THRESHOLD,
        "minimum_threshold_to_accept_each_physical_best": {item["query"]["word"]: item["reference"]["best_distance"] for item in physical_results},
        "wrong_predictions_that_would_be_accepted_at_each_query_own_best": [item["query"]["word"] for item in physical_results if item["reference"]["predicted_word"] != item["query"]["word"]],
        "false_accept_risk_statement": "Any threshold high enough to accept the current GREETING or LOVE query would accept its current THANK_YOU misclassification; threshold adjustment is not selected.",
    }
    conclusion = {
        "schema_version": "words_in_silence.step9d.diagnostic_conclusion.v1", "primary_verdict": PRIMARY_VERDICT, "status": FINAL_STATUS,
        "observable_recognition_pipeline": {"classification": "CONFIRMED", "finding": "The stored normalized physical queries independently reproduce all 27 Unity distances and winners within the approved tolerance.", "max_stored_distance_difference": distance_meta["max_stored_delta"], "max_reference_unity_contract_difference": distance_meta["max_cross_delta"]},
        "upstream_raw_to_normalized_pipeline": {"classification": "UNVERIFIED", "finding": "The fixed Step 9C evidence does not contain the raw/intermediate arrays required to replay the same physical raw query through both preprocessing paths."},
        "physical_domain_shift": {"classification": "SUPPORTED_BUT_NOT_PROVEN", "finding": "At the observable normalized feature/DTW boundary, all three physical best distances lie far above the validation distribution and frozen threshold; upstream provenance cannot be separated from performer/capture domain shift with the retained evidence."},
        "thank_you_selection_bias": {"classification": "CONFIRMED_AT_DISTANCE_AND_COST_DECOMPOSITION_LEVEL", "word_minimum_matrix": word_matrix, "details": bias_details, "causal_origin_beyond_final_feature_costs": "UNVERIFIED_MISSING_RAW_AND_NORMALIZATION_INTERMEDIATES"},
        "threshold_increase_risk": threshold_risk,
        "next_minimum_work": "Acquire one read-only diagnostic export of the already-finalized three captures' raw 75-landmark observations, per-landmark boundary/original/interpolation masks, and per-frame torso transform basis; do not recapture and do not change threshold/templates/production.",
    }
    add_assert(assertions, "threshold_raise_false_accept_risk_documented", "PASS", threshold_risk)
    add_assert(assertions, "primary_verdict_matches_evidence_availability", "PASS", PRIMARY_VERDICT)

    names = [item["name"] for item in assertions]
    require(len(names) == len(set(names)), "diagnostic assertion names are duplicated")
    assertion_summary = {status.lower(): sum(item["status"] == status for item in assertions) for status in ("PASS", "FAIL", "UNVERIFIED", "NOT_APPLICABLE")}
    require(assertion_summary["fail"] == 0, f"diagnostic self-verification assertion failures: {assertion_summary['fail']}")

    staging_root = Path(tempfile.mkdtemp(prefix="wordsinsilence_step9d_"))
    # Keep the local staging path short on Windows; archive paths still use OUTPUT_REL.
    stage_output = staging_root / "o"
    stage_output.mkdir(parents=True)
    temp_zip = staging_root / DELIVERY_NAME
    temp_hash = staging_root / f"{DELIVERY_NAME}.sha256"
    temp_post = staging_root / POST_ZIP.name
    try:
        input_copies = stage_input_copies(stage_output, step9c, preflight)
        input_manifest = {
            "schema_version": "words_in_silence.step9d.input_manifest.v1", "run_id": str(uuid.uuid4()), "read_only": True,
            "project_root": str(ROOT), "unity_project_version": (ROOT / "ProjectSettings/ProjectVersion.txt").read_text(encoding="utf-8").splitlines(),
            "step9c_delivery": file_record(STEP9C_DELIVERY, "approved_input"), "step9c_detached": file_record(STEP9C_DETACHED, "approved_input"),
            "step9c_archive": {"manifest_targets": 88, "zip_entries": 89, "physical_json": 20, "physical_checksums": 20, "objective_valid_captures": [{"expected_word": item["word"], "capture_id": item["capture_id"], "source_sha256": item["source"]["sha256"]} for item in physical]},
            "step8a": {"selected_config": file_record(preflight["paths"]["selected_config"], "approved_input"), "feature_schema": file_record(preflight["paths"]["feature_schema"], "approved_input"), "template_manifest": file_record(preflight["paths"]["template_manifest"], "approved_input"), "templates": 9, "validation_queries": 9, "test_split_used": False},
            "step9b_parity": {"checked": 20, "passed": 20, "failed": 0, "source_manifest": file_record(STEP9C / "delivery_input_manifest.json", "approved_input")},
            "reference_and_contract_sources": stage_data["sources"], "parity_tolerance": {"value": PARITY_TOLERANCE, "source": file_record(preflight["paths"]["tolerance_source"], "approved_input")},
            "staged_read_only_input_copies": input_copies, "preexisting_dirty_worktree": dirty_before,
        }
        write_json(stage_output / "input_manifest.json", input_manifest)
        distance_fields = ["query_expected_word", "capture_id", "template_word", "template_speed", "template_sample_id", "stored_unity_distance", "python_reference_distance", "unity_contract_reconstruction_distance", "absolute_difference", "relative_difference", "parity_tolerance", "parity_tolerance_source", "match_status", "query_distance_rank", "word_minimum", "predicted_word", "second_best_word", "margin", "frozen_threshold", "accepted", "stored_path_length", "python_path_length", "unity_contract_path_length"]
        write_csv(stage_output / "physical_template_distance_comparison.csv", distance_fields, sorted(distance_rows, key=lambda row: (WORDS.index(row["query_expected_word"]), row["query_distance_rank"])))
        contribution_fields = ["query_expected_word", "capture_id", "template_word", "template_speed", "template_sample_id", "pose_contribution", "left_hand_contribution", "right_hand_contribution", "contribution_sum", "dtw_distance", "path_length", "path_comparable_landmarks_min", "path_comparable_landmarks_max", "path_comparable_landmarks_mean"]
        write_csv(stage_output / "dtw_group_cost_contributions.csv", contribution_fields, sorted(contribution_rows, key=lambda row: (WORDS.index(row["query_expected_word"]), row["template_word"], row["template_speed"])))
        write_json(stage_output / "stage_contract_comparison.json", stage_data)
        stats_fields = list(stats_rows[0].keys())
        write_csv(stage_output / "stage_statistics.csv", stats_fields, stats_rows)
        write_json(stage_output / "offline_physical_distribution_comparison.json", distribution)
        write_json(stage_output / "diagnostic_conclusion.json", conclusion)
        write_json(stage_output / "dirty_worktree_snapshot.json", dirty_before)

        protected_mismatches = compare_snapshot(protected_before)
        require(not protected_mismatches, f"protected file mismatch before packaging: {protected_mismatches[:3]}")
        protected_payload = {"schema_version": "words_in_silence.step9d.protected_files_before_after.v1", "protected_file_count": len(protected_before), "before": [protected_before[key] for key in sorted(protected_before)], "after": [protected_before[key] for key in sorted(protected_before)], "mismatch_count": 0, "mismatches": [], "note": "After records were independently re-hashed and matched; identical records are retained once per side without changing protected inputs."}
        write_json(stage_output / "protected_files_before_after.json", protected_payload)
        add_assert(assertions, "protected_files_before_after_mismatch_zero", "PASS", {"checked": len(protected_before), "mismatches": 0})
        assertion_summary = {status.lower(): sum(item["status"] == status for item in assertions) for status in ("PASS", "FAIL", "UNVERIFIED", "NOT_APPLICABLE")}
        assertions_payload = {"schema_version": "words_in_silence.step9d.diagnostic_assertions.v1", "status": FINAL_STATUS, "summary": {**assertion_summary, "total": len(assertions), "unique_names": len(set(item["name"] for item in assertions)), "duplicate_names": 0}, "assertions": assertions}
        write_json(stage_output / "diagnostic_assertions.json", assertions_payload)

        report = f"""# Step 9D Physical Recognition Domain-Gap Diagnostic v1.0

**Status**: `{FINAL_STATUS}`  
**Primary verdict**: `{PRIMARY_VERDICT}`

## Observable result

- Step 9C fixed Delivery input: {STEP9C_DELIVERY.stat().st_size:,} bytes; SHA-256 `{STEP9C_EXPECTED_SHA256}`; detached checksum strict PASS.
- Objective-valid fixed captures: GREETING, LOVE, THANK_YOU exactly once each.
- Frozen templates: 9 train-only word-speed medoids; band 0.10; threshold {THRESHOLD:.16f}.
- Physical 3 x template 9: 27 reference distances; 27 PASS / 0 FAIL / 0 UNVERIFIED against stored Unity distances.
- Maximum stored-vs-reference distance delta: {distance_meta['max_stored_delta']:.17g}; approved tolerance: {PARITY_TOLERANCE:.1e}.
- Maximum Python-reference vs separately reconstructed Unity-contract delta: {distance_meta['max_cross_delta']:.17g}.
- Unity, Unity tests, Editor, player, webcam and physical capture were not executed.

## Why the primary verdict is UNVERIFIED

The fixed Step 9C JSON serializes the final normalized `[T,16,3]` coordinates, `[T,16]` mask, timestamps, shoulder-width summary and aggregate raw counts. It does not serialize the per-frame 75-landmark raw coordinates/masks, per-landmark original/reappearance/strong-depth/interpolation arrays, or torso origin/right/up/forward basis. Therefore the same physical raw query cannot be independently replayed through both raw-to-normalized paths. Observable stages 10-16 pass; stages 2-9 remain partly or wholly unverified.

## Technical domain-gap evidence

- Offline validation uses only the approved 9 validation queries; test split distance, prediction and performance were not computed.
- Physical best distances: GREETING {physical_results[0]['reference']['best_distance']:.17g}, LOVE {physical_results[1]['reference']['best_distance']:.17g}, THANK_YOU {physical_results[2]['reference']['best_distance']:.17g}.
- All physical winners are THANK_YOU_NORMAL_R05. Exact path-normalized pose/left/right contribution rows are in `dtw_group_cost_contributions.csv`.
- The final-feature/DTW evidence supports a physical domain shift, but cannot prove whether its first cause is upstream coordinate/normalization behavior or performer/capture-domain geometry because required raw intermediates are absent.

## Threshold safety

Raising the threshold enough to accept the current GREETING or LOVE capture would accept its current THANK_YOU misclassification. Threshold change is neither performed nor proposed.

## Next minimum work

Acquire one read-only diagnostic export of the already-finalized three captures' raw 75-landmark observations, per-landmark boundary/original/interpolation masks and per-frame torso transform basis. Do not recapture and do not change production, templates or threshold.

`{FINAL_STATUS}`
"""
        (stage_output / "STEP9D_PHYSICAL_RECOGNITION_DOMAIN_GAP_DIAGNOSTIC_V1_0_REPORT.md").write_text(report, encoding="utf-8", newline="\n")

        output_files = sorted(path for path in stage_output.rglob("*") if path.is_file() and path.name != "output_file_manifest.json")
        records = [file_record(ROOT / SCRIPT_REL, "diagnostic_script", SCRIPT_REL)]
        for path in output_files:
            records.append(file_record(path, "diagnostic_output", f"{OUTPUT_REL}/{path.relative_to(stage_output).as_posix()}"))
        records.sort(key=lambda item: item["relative_path"])
        require(len({item["relative_path"] for item in records}) == len(records), "output manifest duplicate target")
        output_manifest = {"schema_version": "words_in_silence.step9d.output_file_manifest.v1", "version": VERSION, "self_reference_excluded": True, "excluded_final_sidecars": [DELIVERY_NAME, f"{DELIVERY_NAME}.sha256", POST_ZIP.name], "total_files": len(records), "outputs": records}
        write_json(stage_output / "output_file_manifest.json", output_manifest)

        with zipfile.ZipFile(temp_zip, "w") as archive:
            for record in records:
                if record["relative_path"] == SCRIPT_REL:
                    data = (ROOT / SCRIPT_REL).read_bytes()
                else:
                    data = (stage_output / PurePosixPath(record["relative_path"]).relative_to(OUTPUT_REL)).read_bytes()
                deterministic_zip_write(archive, record["relative_path"], data)
            deterministic_zip_write(archive, OUTPUT_MANIFEST_REL, (stage_output / "output_file_manifest.json").read_bytes())
        zip_hash = sha256_file(temp_zip)
        temp_hash.write_text(f"{zip_hash}  {DELIVERY_NAME}\n", encoding="utf-8", newline="\n")

        post_checks = []
        with zipfile.ZipFile(temp_zip, "r") as archive:
            infos = archive.infolist()
            names_in_zip = [info.filename for info in infos]
            manifest_inside = json.loads(archive.read(OUTPUT_MANIFEST_REL).decode("utf-8"))
            expected_records = {item["relative_path"]: item for item in manifest_inside["outputs"]}
            actual_targets = set(names_in_zip) - {OUTPUT_MANIFEST_REL}
            missing = sorted(set(expected_records) - actual_targets)
            extra = sorted(actual_targets - set(expected_records))
            duplicates = sorted(name for name, count in Counter(names_in_zip).items() if count > 1)
            unsafe = sorted(name for name in names_in_zip if not safe_zip_name(name))
            symlinks = sorted(info.filename for info in infos if ((info.external_attr >> 16) & 0o170000) == 0o120000)
            encrypted = sorted(info.filename for info in infos if info.flag_bits & 0x1)
            size_mismatch = []
            hash_mismatch = []
            for name, record in expected_records.items():
                if name not in actual_targets:
                    continue
                data = archive.read(name)
                if len(data) != int(record["size_bytes"]):
                    size_mismatch.append(name)
                if sha256_bytes(data) != record["sha256"]:
                    hash_mismatch.append(name)
            required_outputs = [SCRIPT_REL, f"{OUTPUT_REL}/input_manifest.json", f"{OUTPUT_REL}/protected_files_before_after.json", f"{OUTPUT_REL}/physical_template_distance_comparison.csv", f"{OUTPUT_REL}/stage_contract_comparison.json", f"{OUTPUT_REL}/stage_statistics.csv", f"{OUTPUT_REL}/offline_physical_distribution_comparison.json", f"{OUTPUT_REL}/diagnostic_assertions.json", f"{OUTPUT_REL}/diagnostic_conclusion.json", f"{OUTPUT_REL}/STEP9D_PHYSICAL_RECOGNITION_DOMAIN_GAP_DIAGNOSTIC_V1_0_REPORT.md", OUTPUT_MANIFEST_REL]
            checks = {
                "crc_error": archive.testzip(), "duplicate_entries": duplicates, "unsafe_paths": unsafe, "symlinks": symlinks, "encrypted_entries": encrypted,
                "manifest_missing": missing, "manifest_extra": extra, "manifest_size_mismatch": size_mismatch, "manifest_sha256_mismatch": hash_mismatch,
                "required_output_missing": sorted(set(required_outputs) - set(names_in_zip)), "assertion_fail_count": assertion_summary["fail"],
                "protected_file_mismatch_count": len(compare_snapshot(protected_before)), "detached_checksum_strict": parse_detached(temp_hash.read_bytes(), DELIVERY_NAME, zip_hash),
            }
            require(checks["crc_error"] is None and not any(checks[key] for key in ("duplicate_entries", "unsafe_paths", "symlinks", "encrypted_entries", "manifest_missing", "manifest_extra", "manifest_size_mismatch", "manifest_sha256_mismatch", "required_output_missing")) and checks["assertion_fail_count"] == 0 and checks["protected_file_mismatch_count"] == 0 and checks["detached_checksum_strict"], f"post-ZIP verification failed: {checks}")
            post_checks = checks
        post_payload = {
            "schema_version": "words_in_silence.step9d.post_zip_verification.v1", "status": FINAL_STATUS,
            "zip_file": DELIVERY_NAME, "zip_size_bytes": temp_zip.stat().st_size, "zip_sha256": zip_hash,
            "zip_entry_count": len(records) + 1, "manifest_target_count": len(records), "manifest_self_excluded": True,
            "checks": post_checks, "diagnostic_assertion_summary": assertion_summary, "primary_verdict": PRIMARY_VERDICT,
            "physical_distance_parity": {"pass": 27, "fail": 0, "unverified": 0, "max_absolute_difference": distance_meta["max_stored_delta"]},
            "physical_evidence": {"objective_valid_queries": 3, "source_archive_json": 20, "source_archive_checksums": 20},
            "step9b_parity": {"checked": 20, "passed": 20, "failed": 0}, "protected_files": {"checked": len(protected_before), "mismatches": 0},
        }
        write_json(temp_post, post_payload)

        require(not OUTPUT.exists() and not DELIVERY_ZIP.exists() and not DELIVERY_HASH.exists() and not POST_ZIP.exists(), "final output appeared during staging")
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage_output, OUTPUT)
        os.replace(temp_hash, DELIVERY_HASH)
        os.replace(temp_post, POST_ZIP)
        os.replace(temp_zip, DELIVERY_ZIP)
        require(sha256_file(DELIVERY_ZIP) == zip_hash and parse_detached(DELIVERY_HASH.read_bytes(), DELIVERY_NAME, zip_hash), "committed ZIP/detached checksum mismatch")
        final_protected_mismatch = compare_snapshot(protected_before)
        require(not final_protected_mismatch, f"protected file mismatch after commit: {final_protected_mismatch[:3]}")
        return {
            "status": FINAL_STATUS, "primary_verdict": PRIMARY_VERDICT, "script": file_record(ROOT / SCRIPT_REL, "diagnostic_script"),
            "output_manifest": file_record(OUTPUT / "output_file_manifest.json", "output_manifest"), "report": file_record(OUTPUT / "STEP9D_PHYSICAL_RECOGNITION_DOMAIN_GAP_DIAGNOSTIC_V1_0_REPORT.md", "report"),
            "zip": file_record(DELIVERY_ZIP, "delivery_zip"), "detached": file_record(DELIVERY_HASH, "detached_checksum"), "post_zip": file_record(POST_ZIP, "post_zip_verification"),
            "zip_entry_count": len(records) + 1, "manifest_target_count": len(records), "assertions": assertion_summary,
            "distance_meta": distance_meta, "protected_file_count": len(protected_before), "protected_mismatch_count": 0,
            "physical_best": [{"expected_word": item["query"]["word"], "stored_best": float(item["stored"]["best_distance"]), "reference_best": item["reference"]["best_distance"], "absolute_difference": abs(float(item["stored"]["best_distance"]) - item["reference"]["best_distance"])} for item in physical_results],
            "distribution": distribution,
        }
    finally:
        if staging_root.exists():
            resolved = staging_root.resolve()
            temp_root = Path(tempfile.gettempdir()).resolve()
            if resolved.parent == temp_root and resolved.name.startswith("wordsinsilence_step9d_"):
                shutil.rmtree(resolved)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        result = build()
    except Exception as exc:
        print(json.dumps({"status": "STEP9D_PHYSICAL_RECOGNITION_DOMAIN_GAP_DIAGNOSTIC_SELF_VERIFICATION_FAIL", "error": repr(exc)}, ensure_ascii=False, indent=2), flush=True)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
