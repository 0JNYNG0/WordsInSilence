#!/usr/bin/env python3
"""Prepare, verify, and package WordsInSilence RecognitionCore v1.0."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import re
import shutil
import struct
import sys
import uuid
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
STEP8A = ROOT / "data/processed/sessions/S001_P001/step8a_dtw_templates_v1_0"
STEP8A_ZIP = ROOT / "WordsInSilence_step8a_dtw_templates_v1_0_delivery.zip"
STEP8A_HASH = ROOT / "WordsInSilence_step8a_dtw_templates_v1_0_delivery.zip.sha256"
STEP8A_SIDECAR = ROOT / "WordsInSilence_step8a_dtw_templates_v1_0_delivery_post_zip_verification.json"
OUTPUT = ROOT / "data/processed/sessions/S001_P001/step8b_recognition_core_v1_0"
RUNTIME_ROOT = ROOT / "Assets/StreamingAssets/RecognitionCoreV1"
TEMPLATE_ROOT = RUNTIME_ROOT / "Templates"
FIXTURE_ROOT = ROOT / "Assets/_Project/Tests/EditMode/Fixtures/RecognitionCoreV1"
QUERY_ROOT = FIXTURE_ROOT / "ValidationQueries"
CORE_ROOT = ROOT / "Assets/_Project/Scripts/Recognition/Core"
TEST_SOURCE = ROOT / "Assets/_Project/Tests/EditMode/RecognitionCoreV1Tests.cs"
DELIVERY_NAME = "WordsInSilence_step8b_recognition_core_v1_0_delivery.zip"
DELIVERY_ZIP = ROOT / DELIVERY_NAME
DELIVERY_HASH = ROOT / f"{DELIVERY_NAME}.sha256"
DELIVERY_SIDECAR = ROOT / "WordsInSilence_step8b_recognition_core_v1_0_delivery_post_zip_verification.json"

VERSION = "1.0"
FORMAT_VERSION = 1
MAGIC = b"WISRCB01"
ENDIAN_MARKER = 0x01020304
LANDMARK_COUNT = 16
BAND = 0.10
TOLERANCE = 1e-12
APPROVED_RUN_ID = "0918c71c-313f-49d0-a5a9-b525f6630317"
APPROVED_ZIP_SIZE = 2_205_564
APPROVED_ZIP_SHA256 = "fa8292c424644bc5f50f7213e7587f5157b71cd651df804b8a542168ffe4bf3c"
WORDS = ("GREETING", "LOVE", "THANK_YOU")
SPEEDS = ("FAST", "NORMAL", "SLOW")


def log(message=""):
    print(message, flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def add_check(checks: list[dict], name: str, passed: bool, evidence=None) -> None:
    checks.append({"name": name, "status": "PASS" if bool(passed) else "FAIL", "evidence": evidence})


def summarize(checks: list[dict]) -> dict:
    names = [record["name"] for record in checks]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    passed = sum(record["status"] == "PASS" for record in checks)
    return {
        "total": len(checks),
        "pass": passed,
        "fail": len(checks) - passed,
        "duplicate_count": len(duplicates),
        "duplicate_names": duplicates,
    }


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def logical_payload_hash(npz) -> str:
    metadata = json.loads(str(npz["metadata_json"].item()))
    digest = hashlib.sha256(canonical_json(metadata).encode("utf-8"))
    for name in ("coordinates", "valid_mask", "original_valid_mask", "interpolation_mask", "source_frame_index", "timestamp"):
        value = np.ascontiguousarray(npz[name])
        descriptor = {"name": name, "dtype": value.dtype.str, "shape": list(value.shape)}
        digest.update(canonical_json(descriptor).encode("utf-8"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def ensure_safe_approved_inputs() -> None:
    for path in (STEP8A, STEP8A_ZIP, STEP8A_HASH, STEP8A_SIDECAR):
        if not path.exists():
            raise RuntimeError(f"approved Step 8A input is missing: {path}")


def preflight() -> tuple[dict, list[dict]]:
    ensure_safe_approved_inputs()
    checks: list[dict] = []
    zip_size = STEP8A_ZIP.stat().st_size
    zip_hash = sha256_file(STEP8A_ZIP)
    add_check(checks, "step8a_zip_size", zip_size == APPROVED_ZIP_SIZE, zip_size)
    add_check(checks, "step8a_zip_sha256", zip_hash == APPROVED_ZIP_SHA256, zip_hash)
    detached = STEP8A_HASH.read_text(encoding="ascii")
    match = re.fullmatch(r"([0-9a-f]{64})  ([^\r\n]+)\n", detached)
    add_check(checks, "step8a_detached_strict", match is not None, detached.rstrip())
    add_check(checks, "step8a_detached_hash", match is not None and match.group(1) == zip_hash, None)
    add_check(checks, "step8a_detached_filename", match is not None and match.group(2) == STEP8A_ZIP.name, None)
    with zipfile.ZipFile(STEP8A_ZIP, "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        bad_crc = archive.testzip()
    unsafe = [name for name in names if name.startswith(("/", "\\")) or ".." in PurePosixPath(name).parts or re.match(r"^[A-Za-z]:", name)]
    symlinks = [info.filename for info in infos if ((info.external_attr >> 16) & 0o170000) == 0o120000]
    encrypted = [info.filename for info in infos if info.flag_bits & 1]
    add_check(checks, "step8a_zip_crc", bad_crc is None, bad_crc)
    add_check(checks, "step8a_zip_entries_68", len(names) == 68, len(names))
    add_check(checks, "step8a_zip_unique", len(names) == len(set(names)), [len(names), len(set(names))])
    add_check(checks, "step8a_zip_safe", not unsafe, unsafe)
    add_check(checks, "step8a_zip_no_symlinks", not symlinks, symlinks)
    add_check(checks, "step8a_zip_no_encryption", not encrypted, encrypted)

    sidecar = load_json(STEP8A_SIDECAR)
    add_check(checks, "step8a_sidecar_pass", sidecar["verification_status"] == "PASS", sidecar["verification_status"])
    add_check(checks, "step8a_sidecar_failures_zero", sidecar["sidecar_failure_count"] == 0, sidecar["sidecar_failure_count"])
    add_check(checks, "step8a_sidecar_duplicates_zero", sidecar["sidecar_check_name_duplicates"] == 0 and sidecar["sidecar_record_duplicates"] == 0, None)
    add_check(checks, "step8a_sidecar_482_unique", sidecar["sidecar_check_total"] == sidecar["sidecar_check_unique"] == 482, [sidecar["sidecar_check_total"], sidecar["sidecar_check_unique"]])

    config = load_json(STEP8A / "selected_dtw_config.json")
    template_manifest = load_json(STEP8A / "template_manifest.json")
    output_manifest = load_json(STEP8A / "output_file_manifest.json")
    output_lookup = {record["relative_path"]: record for record in output_manifest["outputs"]}
    add_check(checks, "step8a_run_id", config["run_id"] == template_manifest["run_id"] == output_manifest["run_id"] == APPROVED_RUN_ID, [config["run_id"], template_manifest["run_id"], output_manifest["run_id"]])
    add_check(checks, "selected_strategy", config["strategy"] == "word_speed_medoid", config["strategy"])
    add_check(checks, "selected_band", config["sakoe_chiba_relative_band"] == BAND, config["sakoe_chiba_relative_band"])
    add_check(checks, "selected_template_count", config["template_count"] == len(template_manifest["templates"]) == 9, [config["template_count"], len(template_manifest["templates"])])
    add_check(checks, "selected_threshold_absent", config["rejection_threshold"] == "not_set", config["rejection_threshold"])

    template_records = []
    groups = set()
    for group, record in sorted(template_manifest["templates"].items()):
        source = STEP8A / record["relative_path"]
        output_record = output_lookup.get(record["relative_path"])
        actual_hash = sha256_file(source)
        add_check(checks, f"template_{group}_train", record["split"] == "train", record["split"])
        add_check(checks, f"template_{group}_manifest_target", output_record is not None, record["relative_path"])
        add_check(checks, f"template_{group}_size", output_record is not None and source.stat().st_size == output_record["size_bytes"], source.stat().st_size)
        add_check(checks, f"template_{group}_file_hash", output_record is not None and actual_hash == output_record["sha256"] == record["file_sha256"], actual_hash)
        with np.load(source, allow_pickle=False) as npz:
            embedded = str(npz["logical_payload_sha256"].item())
            recomputed = logical_payload_hash(npz)
            add_check(checks, f"template_{group}_logical_hash", embedded == recomputed == record["logical_payload_sha256"], [embedded, recomputed, record["logical_payload_sha256"]])
            add_check(checks, f"template_{group}_shape", npz["coordinates"].shape[1:] == (16, 3) and npz["valid_mask"].shape == (npz["coordinates"].shape[0], 16), [list(npz["coordinates"].shape), list(npz["valid_mask"].shape)])
            add_check(checks, f"template_{group}_metadata", str(npz["sample_id"].item()) == record["sample_id"] and str(npz["word"].item()) == record["word"] and str(npz["speed"].item()) == record["speed"] and str(npz["split"].item()) == "train", None)
        groups.add((record["word"], record["speed"]))
        template_records.append({"group": group, "record": record, "source": source, "source_sha256": actual_hash})
    expected_groups = {(word, speed) for word in WORDS for speed in SPEEDS}
    add_check(checks, "template_groups_exact", groups == expected_groups, sorted(groups))

    with (STEP8A / "validation_distances.csv").open(encoding="utf-8", newline="") as stream:
        selected_distances = [row for row in csv.DictReader(stream) if row["configuration"] == config["configuration_name"]]
    with (STEP8A / "validation_predictions.csv").open(encoding="utf-8", newline="") as stream:
        selected_predictions = [row for row in csv.DictReader(stream) if row["configuration"] == config["configuration_name"]]
    validation_ids = sorted({row["validation_sample_id"] for row in selected_distances})
    add_check(checks, "selected_distance_records_81", len(selected_distances) == 81, len(selected_distances))
    add_check(checks, "selected_prediction_records_9", len(selected_predictions) == 9, len(selected_predictions))
    add_check(checks, "validation_ids_9", len(validation_ids) == 9, validation_ids)

    required_planned = [
        ROOT / "Tools/s001_step8b_export_recognition_core_v1_0.py",
        CORE_ROOT / "RecognitionSequence.cs",
        CORE_ROOT / "RecognitionSequenceBinary.cs",
        CORE_ROOT / "MaskAwareDtw.cs",
        CORE_ROOT / "RecognitionCore.cs",
        TEST_SOURCE,
    ]
    add_check(checks, "implementation_sources_present", all(path.is_file() for path in required_planned), [str(path.relative_to(ROOT)) for path in required_planned if not path.is_file()])
    failures = [record for record in checks if record["status"] == "FAIL"]
    if failures:
        raise RuntimeError(f"Step 8A preflight failed with {len(failures)} checks")
    return {
        "config": config,
        "template_manifest": template_manifest,
        "output_manifest": output_manifest,
        "output_lookup": output_lookup,
        "template_records": template_records,
        "validation_distances": selected_distances,
        "validation_predictions": selected_predictions,
        "validation_ids": validation_ids,
        "zip_size": zip_size,
        "zip_sha256": zip_hash,
    }, checks


def encode_string(value: str) -> bytes:
    data = value.encode("utf-8")
    if not data or len(data) > 4096:
        raise ValueError("invalid binary metadata string")
    return struct.pack("<I", len(data)) + data


def encode_sequence(metadata: dict, coordinates: np.ndarray, valid_mask: np.ndarray) -> bytes:
    coordinates = np.ascontiguousarray(coordinates)
    valid_mask = np.ascontiguousarray(valid_mask)
    if coordinates.dtype != np.float64 or coordinates.ndim != 3 or coordinates.shape[1:] != (LANDMARK_COUNT, 3):
        raise ValueError("coordinates must be float64 [T,16,3]")
    if valid_mask.dtype != np.bool_ or valid_mask.shape != coordinates.shape[:2]:
        raise ValueError("valid mask must be bool [T,16]")
    if np.isnan(coordinates[valid_mask]).any() or np.isinf(coordinates[valid_mask]).any():
        raise ValueError("valid coordinates must be finite")
    little_coordinates = coordinates.astype("<f8", copy=False)
    flat_mask = valid_mask.reshape(-1).astype(np.uint8, copy=False)
    packed_mask = np.packbits(flat_mask, bitorder="little").tobytes()
    payload = bytearray()
    payload += MAGIC
    payload += struct.pack("<IIII", FORMAT_VERSION, ENDIAN_MARKER, LANDMARK_COUNT, coordinates.shape[0])
    for field in ("sample_id", "word", "speed", "split", "source_npz_sha256", "logical_payload_sha256"):
        payload += encode_string(metadata[field])
    payload += struct.pack("<Q", coordinates.size)
    payload += little_coordinates.tobytes(order="C")
    payload += struct.pack("<QI", valid_mask.size, len(packed_mask))
    payload += packed_mask
    return bytes(payload) + hashlib.sha256(payload).digest()


def read_sequence_bytes(data: bytes) -> dict:
    if len(data) < 64:
        raise ValueError("truncated binary")
    payload, trailing = data[:-32], data[-32:]
    if hashlib.sha256(payload).digest() != trailing:
        raise ValueError("payload hash mismatch")
    offset = 0
    if payload[:8] != MAGIC:
        raise ValueError("magic mismatch")
    offset = 8
    version, endian, landmarks, length = struct.unpack_from("<IIII", payload, offset)
    offset += 16
    if version != FORMAT_VERSION or endian != ENDIAN_MARKER or landmarks != LANDMARK_COUNT or length <= 0:
        raise ValueError("header mismatch")
    strings = []
    for _ in range(6):
        size = struct.unpack_from("<I", payload, offset)[0]
        offset += 4
        strings.append(payload[offset:offset + size].decode("utf-8"))
        offset += size
    coordinate_count = struct.unpack_from("<Q", payload, offset)[0]
    offset += 8
    if coordinate_count != length * LANDMARK_COUNT * 3:
        raise ValueError("coordinate count mismatch")
    coordinate_bytes = coordinate_count * 8
    coordinates = np.frombuffer(payload, dtype="<f8", count=coordinate_count, offset=offset).copy().reshape(length, LANDMARK_COUNT, 3)
    offset += coordinate_bytes
    mask_bits, mask_bytes = struct.unpack_from("<QI", payload, offset)
    offset += 12
    if mask_bits != length * LANDMARK_COUNT or mask_bytes != (mask_bits + 7) // 8:
        raise ValueError("mask shape mismatch")
    packed = np.frombuffer(payload[offset:offset + mask_bytes], dtype=np.uint8)
    offset += mask_bytes
    if offset != len(payload):
        raise ValueError("unexpected trailing payload")
    mask = np.unpackbits(packed, bitorder="little")[:mask_bits].astype(np.bool_).reshape(length, LANDMARK_COUNT)
    return {
        "sample_id": strings[0], "word": strings[1], "speed": strings[2], "split": strings[3],
        "source_npz_sha256": strings[4], "logical_payload_sha256": strings[5],
        "length": length, "landmark_count": landmarks, "coordinates": coordinates, "valid_mask": mask,
        "binary_payload_sha256": hashlib.sha256(payload).hexdigest(),
    }


def read_sequence_file(path: Path) -> dict:
    return read_sequence_bytes(path.read_bytes())


def local_cost_matrix(left: dict, right: dict) -> np.ndarray:
    coords_a, coords_b = left["coordinates"], right["coordinates"]
    mask_a, mask_b = left["valid_mask"], right["valid_mask"]
    both = mask_a[:, None, :] & mask_b[None, :, :]
    one = mask_a[:, None, :] ^ mask_b[None, :, :]
    difference = np.where(both[..., None], coords_a[:, None, :, :] - coords_b[None, :, :, :], 0.0)
    euclidean = np.sqrt(np.sum(difference * difference, axis=-1))
    costs = np.where(both, euclidean, np.where(one, 1.0, 0.0))
    included = both | one
    denominator = included.sum(axis=-1)
    result = np.ones(denominator.shape, dtype=np.float64)
    np.divide(costs.sum(axis=-1), denominator, out=result, where=denominator > 0)
    return result


def band_allows(i: int, j: int, length_a: int, length_b: int, band: float = BAND) -> bool:
    return length_a <= 1 or length_b <= 1 or abs(i / (length_a - 1) - j / (length_b - 1)) <= band + 1e-15


def dtw_distance(left: dict, right: dict, band: float = BAND) -> float:
    length_a, length_b = left["length"], right["length"]
    local = local_cost_matrix(left, right)
    cumulative = np.full((length_a, length_b), np.inf, dtype=np.float64)
    parent = np.zeros((length_a, length_b), dtype=np.uint8)
    for i in range(length_a):
        for j in range(length_b):
            if not band_allows(i, j, length_a, length_b, band):
                continue
            if i == 0 and j == 0:
                cumulative[i, j] = local[i, j]
                continue
            candidates = []
            if i and j and np.isfinite(cumulative[i - 1, j - 1]): candidates.append((cumulative[i - 1, j - 1], 1))
            if i and np.isfinite(cumulative[i - 1, j]): candidates.append((cumulative[i - 1, j], 2))
            if j and np.isfinite(cumulative[i, j - 1]): candidates.append((cumulative[i, j - 1], 3))
            if candidates:
                prior, direction = min(candidates, key=lambda item: (item[0], item[1]))
                cumulative[i, j] = prior + local[i, j]
                parent[i, j] = direction
    if not np.isfinite(cumulative[-1, -1]):
        raise ValueError("band has no endpoint path")
    i, j, path_length = length_a - 1, length_b - 1, 1
    while i or j:
        direction = parent[i, j]
        if direction == 1: i, j = i - 1, j - 1
        elif direction == 2: i -= 1
        elif direction == 3: j -= 1
        else: raise ValueError("incomplete parent path")
        path_length += 1
    return float(cumulative[-1, -1] / path_length)


def meta_guid(path: Path) -> str:
    relative = path.relative_to(ROOT).as_posix()
    return hashlib.sha256(("WordsInSilence-Step8B:" + relative).encode("utf-8")).hexdigest()[:32]


def ensure_meta(path: Path, folder=False) -> Path:
    meta = path.with_name(path.name + ".meta")
    if meta.exists():
        return meta
    guid = meta_guid(path)
    if folder:
        text = f"fileFormatVersion: 2\nguid: {guid}\nfolderAsset: yes\nDefaultImporter:\n  externalObjects: {{}}\n  userData: \n  assetBundleName: \n  assetBundleVariant: \n"
    elif path.suffix == ".cs":
        text = f"fileFormatVersion: 2\nguid: {guid}\nMonoImporter:\n  externalObjects: {{}}\n  serializedVersion: 2\n  defaultReferences: []\n  executionOrder: 0\n  icon: {{instanceID: 0}}\n  userData: \n  assetBundleName: \n  assetBundleVariant: \n"
    else:
        text = f"fileFormatVersion: 2\nguid: {guid}\nDefaultImporter:\n  externalObjects: {{}}\n  userData: \n  assetBundleName: \n  assetBundleVariant: \n"
    meta.write_text(text, encoding="utf-8")
    return meta


def create_asset_layout() -> list[Path]:
    created = []
    folders = [CORE_ROOT, RUNTIME_ROOT, TEMPLATE_ROOT, FIXTURE_ROOT.parent, FIXTURE_ROOT, QUERY_ROOT]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
        created.append(ensure_meta(folder, folder=True))
    source_files = [
        CORE_ROOT / "RecognitionSequence.cs",
        CORE_ROOT / "RecognitionSequenceBinary.cs",
        CORE_ROOT / "MaskAwareDtw.cs",
        CORE_ROOT / "RecognitionCore.cs",
        TEST_SOURCE,
    ]
    for source in source_files:
        created.append(ensure_meta(source))
    return created


def npz_payload(path: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as npz:
        coordinates = np.ascontiguousarray(npz["coordinates"])
        valid_mask = np.ascontiguousarray(npz["valid_mask"])
        metadata = {
            "sample_id": str(npz["sample_id"].item()),
            "word": str(npz["word"].item()),
            "speed": str(npz["speed"].item()),
            "split": str(npz["split"].item()),
            "source_npz_sha256": sha256_file(path),
            "logical_payload_sha256": str(npz["logical_payload_sha256"].item()),
        }
        recomputed = logical_payload_hash(npz)
    if recomputed != metadata["logical_payload_sha256"]:
        raise RuntimeError(f"logical payload mismatch: {path}")
    return metadata, coordinates, valid_mask


def validate_template_records(records: list[dict]) -> None:
    if len(records) != 9:
        raise ValueError("nine templates required")
    groups = set()
    for record in records:
        if record["split"] != "train":
            raise ValueError("non-train template")
        group = (record["word"], record["speed"])
        if group in groups:
            raise ValueError("duplicate template group")
        groups.add(group)
    if groups != {(word, speed) for word in WORDS for speed in SPEEDS}:
        raise ValueError("missing template group")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def copy_json_to_fixture(source: Path) -> Path:
    destination = FIXTURE_ROOT / source.name
    shutil.copy2(source, destination)
    ensure_meta(destination)
    return destination


def prepare() -> int:
    log("=== Step 8B prepare ===")
    preflight_data, checks = preflight()
    if OUTPUT.exists():
        raise RuntimeError(f"Step 8B output already exists; preserve it and use --finalize or --package-only: {OUTPUT}")
    run_id = str(uuid.uuid4())
    OUTPUT.mkdir(parents=True)
    created_metas = create_asset_layout()
    log(f"Run ID: {run_id}")

    runtime_records = []
    runtime_sequences = {}
    for entry in preflight_data["template_records"]:
        source = entry["source"]
        metadata, coordinates, valid_mask = npz_payload(source)
        group = entry["group"]
        binary_a = encode_sequence(metadata, coordinates, valid_mask)
        binary_b = encode_sequence(metadata, coordinates, valid_mask)
        add_check(checks, f"runtime_{group}_deterministic_bytes", binary_a == binary_b, None)
        destination = TEMPLATE_ROOT / f"template_{group}.wisrt"
        destination.write_bytes(binary_a)
        ensure_meta(destination)
        decoded = read_sequence_bytes(binary_a)
        coordinates_exact = decoded["coordinates"].tobytes(order="C") == coordinates.astype("<f8", copy=False).tobytes(order="C")
        mask_exact = np.array_equal(decoded["valid_mask"], valid_mask)
        metadata_exact = all(decoded[key] == metadata[key] for key in ("sample_id", "word", "speed", "split", "source_npz_sha256", "logical_payload_sha256"))
        add_check(checks, f"runtime_{group}_coordinates_exact", coordinates_exact, None)
        add_check(checks, f"runtime_{group}_mask_exact", mask_exact, None)
        add_check(checks, f"runtime_{group}_metadata_exact", metadata_exact, None)
        record = {
            **{key: metadata[key] for key in ("sample_id", "word", "speed", "split")},
            "length": int(coordinates.shape[0]),
            "landmark_count": LANDMARK_COUNT,
            "relative_path": destination.relative_to(ROOT).as_posix(),
            "size_bytes": destination.stat().st_size,
            "file_sha256": sha256_file(destination),
            "binary_payload_sha256": decoded["binary_payload_sha256"],
            "source_npz_sha256": metadata["source_npz_sha256"],
            "logical_payload_sha256": metadata["logical_payload_sha256"],
            "coordinates_sha256": sha256_bytes(coordinates.astype("<f8", copy=False).tobytes(order="C")),
            "valid_mask_sha256": sha256_bytes(valid_mask.reshape(-1).astype(np.uint8).tobytes()),
        }
        runtime_records.append(record)
        runtime_sequences[metadata["sample_id"]] = decoded
    runtime_records.sort(key=lambda row: (row["word"], row["speed"], row["sample_id"]))
    validate_template_records(runtime_records)
    add_check(checks, "runtime_template_count_9", len(runtime_records) == 9, len(runtime_records))

    query_records = []
    query_sequences = {}
    for sample_id in preflight_data["validation_ids"]:
        source = STEP8A / "features" / f"feature_{sample_id.removeprefix('S001_P001_')}.npz"
        metadata, coordinates, valid_mask = npz_payload(source)
        add_check(checks, f"validation_{sample_id}_split", metadata["split"] == "val", metadata["split"])
        binary = encode_sequence(metadata, coordinates, valid_mask)
        destination = QUERY_ROOT / f"query_{sample_id.removeprefix('S001_P001_')}.wisrq"
        destination.write_bytes(binary)
        ensure_meta(destination)
        decoded = read_sequence_bytes(binary)
        add_check(checks, f"validation_{sample_id}_coordinates_exact", decoded["coordinates"].tobytes(order="C") == coordinates.astype("<f8", copy=False).tobytes(order="C"), None)
        add_check(checks, f"validation_{sample_id}_mask_exact", np.array_equal(decoded["valid_mask"], valid_mask), None)
        query_records.append({
            "sample_id": sample_id, "word": metadata["word"], "speed": metadata["speed"], "split": metadata["split"],
            "length": int(coordinates.shape[0]), "relative_path": destination.relative_to(ROOT).as_posix(),
            "size_bytes": destination.stat().st_size, "file_sha256": sha256_file(destination),
            "source_npz_sha256": metadata["source_npz_sha256"], "logical_payload_sha256": metadata["logical_payload_sha256"],
        })
        query_sequences[sample_id] = decoded
    add_check(checks, "validation_query_count_9", len(query_records) == 9, len(query_records))

    expected_distances = []
    distance_mismatches = []
    for row in sorted(preflight_data["validation_distances"], key=lambda value: (value["validation_sample_id"], value["prototype_sample_id"])):
        query_id = row["validation_sample_id"]
        prototype_id = row["prototype_sample_id"]
        expected = float(row["distance"])
        actual = dtw_distance(query_sequences[query_id], runtime_sequences[prototype_id])
        if abs(expected - actual) > TOLERANCE:
            distance_mismatches.append({"query": query_id, "prototype": prototype_id, "expected": expected, "actual": actual})
        expected_distances.append({"query_sample_id": query_id, "prototype_sample_id": prototype_id, "expected_distance": format(expected, ".17g")})
    add_check(checks, "python_runtime_distance_records_81", len(expected_distances) == 81, len(expected_distances))
    add_check(checks, "python_runtime_distance_mismatch_zero", not distance_mismatches, distance_mismatches[:10])
    distance_path = FIXTURE_ROOT / "validation_expected_distances.csv"
    write_csv(distance_path, ["query_sample_id", "prototype_sample_id", "expected_distance"], expected_distances)
    ensure_meta(distance_path)

    source_prediction_map = {row["sample_id"]: row for row in preflight_data["validation_predictions"]}
    expected_predictions = []
    prediction_mismatches = []
    for query_id in preflight_data["validation_ids"]:
        rows = [row for row in preflight_data["validation_distances"] if row["validation_sample_id"] == query_id]
        word_best = {}
        for word in WORDS:
            choices = sorted((float(row["distance"]), row["prototype_sample_id"]) for row in rows if row["prototype_word"] == word)
            word_best[word] = choices[0]
        predicted_word = min(WORDS, key=lambda word: (word_best[word][0], word))
        source_prediction = source_prediction_map[query_id]
        source_word_distances = json.loads(source_prediction["word_distances_json"])
        if predicted_word != source_prediction["predicted_word"] or any(abs(word_best[word][0] - float(source_word_distances[word])) > TOLERANCE for word in WORDS):
            prediction_mismatches.append(query_id)
        expected_predictions.append({
            "query_sample_id": query_id,
            "true_word": source_prediction["true_word"],
            "predicted_word": predicted_word,
            "winning_prototype_id": word_best[predicted_word][1],
            "distance_GREETING": format(word_best["GREETING"][0], ".17g"),
            "distance_LOVE": format(word_best["LOVE"][0], ".17g"),
            "distance_THANK_YOU": format(word_best["THANK_YOU"][0], ".17g"),
        })
    add_check(checks, "python_runtime_prediction_records_9", len(expected_predictions) == 9, len(expected_predictions))
    add_check(checks, "python_runtime_prediction_mismatch_zero", not prediction_mismatches, prediction_mismatches)
    prediction_path = FIXTURE_ROOT / "validation_expected_predictions.csv"
    write_csv(prediction_path, ["query_sample_id", "true_word", "predicted_word", "winning_prototype_id", "distance_GREETING", "distance_LOVE", "distance_THANK_YOU"], expected_predictions)
    ensure_meta(prediction_path)

    allowed_ids = {record["sample_id"] for record in runtime_records} | {record["sample_id"] for record in query_records}
    used_ids = {row["query_sample_id"] for row in expected_distances} | {row["prototype_sample_id"] for row in expected_distances} | {row["query_sample_id"] for row in expected_predictions}
    add_check(checks, "parity_ids_train_or_validation_only", used_ids <= allowed_ids and len(allowed_ids) == 18, {"allowed": len(allowed_ids), "used": len(used_ids)})
    add_check(checks, "test_split_payloads_loaded_zero", all(record["split"] in ("train", "val") for record in runtime_records + query_records), 0)
    injected = copy.deepcopy(runtime_records)
    injected[0]["split"] = "test"
    detected = False
    try:
        validate_template_records(injected)
    except ValueError:
        detected = True
    add_check(checks, "synthetic_test_split_injection_detected", detected, None)

    runtime_manifest = {"version": VERSION, "run_id": run_id, "format_version": FORMAT_VERSION, "template_order": "word,speed,sample_id ordinal", "templates": runtime_records}
    runtime_manifest_path = OUTPUT / "runtime_template_manifest.json"
    write_json(runtime_manifest_path, runtime_manifest)
    schema = {
        "version": VERSION, "run_id": run_id, "format_version": FORMAT_VERSION,
        "magic_ascii": MAGIC.decode("ascii"), "byte_order": "little_endian", "coordinates": "IEEE-754 float64 C-order [T,16,3]",
        "valid_mask": "[T,16] packed LSB-first", "trailing_integrity": "SHA-256 of every preceding byte",
        "fields": ["magic[8]", "uint32 version", "uint32 endian_marker", "uint32 landmarks", "uint32 length", "six uint32-length-prefixed UTF-8 strings", "uint64 coordinate_count", "float64 coordinates", "uint64 mask_bit_count", "uint32 mask_byte_count", "packed mask", "sha256[32]"],
    }
    schema_path = OUTPUT / "runtime_template_binary_schema.json"
    write_json(schema_path, schema)
    schema_md = OUTPUT / "RUNTIME_TEMPLATE_BINARY_SCHEMA.md"
    schema_md.write_text(
        "# RecognitionCore v1.0 Runtime Binary Schema\n\n"
        f"**Run ID**: {run_id}\n\n"
        "`WISRCB01`, format version 1, little-endian. Coordinates are raw IEEE-754 float64 C-order `[T,16,3]` bytes, including invalid-position NaN payloads. The `[T,16]` valid mask is packed least-significant-bit first. Metadata strings preserve sample ID, word, speed, split, source NPZ SHA-256 and Step 8A logical payload SHA-256. The final 32 bytes are SHA-256 of the preceding payload.\n",
        encoding="utf-8",
    )
    fixture_manifest = {
        "version": VERSION, "run_id": run_id, "scope": "approved_validation_only",
        "validation_queries": query_records,
        "expected_distances": {"relative_path": distance_path.relative_to(ROOT).as_posix(), "records": 81, "sha256": sha256_file(distance_path)},
        "expected_predictions": {"relative_path": prediction_path.relative_to(ROOT).as_posix(), "records": 9, "sha256": sha256_file(prediction_path)},
        "test_split_payloads": 0,
    }
    fixture_manifest_path = OUTPUT / "fixture_manifest.json"
    write_json(fixture_manifest_path, fixture_manifest)
    copied_manifests = [copy_json_to_fixture(runtime_manifest_path), copy_json_to_fixture(schema_path), copy_json_to_fixture(fixture_manifest_path)]

    input_paths = [
        STEP8A_ZIP, STEP8A_HASH, STEP8A_SIDECAR,
        STEP8A / "selected_dtw_config.json", STEP8A / "template_manifest.json", STEP8A / "output_file_manifest.json",
        STEP8A / "validation_distances.csv", STEP8A / "validation_predictions.csv",
    ]
    input_paths += [entry["source"] for entry in preflight_data["template_records"]]
    input_paths += [STEP8A / "features" / f"feature_{sample_id.removeprefix('S001_P001_')}.npz" for sample_id in preflight_data["validation_ids"]]
    input_records = []
    for path in sorted(set(input_paths)):
        stat = path.stat()
        input_records.append({"relative_path": path.relative_to(ROOT).as_posix(), "size_bytes": stat.st_size, "sha256": sha256_file(path), "mtime_ns": stat.st_mtime_ns})
    input_manifest = {
        "version": VERSION, "run_id": run_id, "self_reference_excluded": True,
        "approved_step8a_run_id": APPROVED_RUN_ID, "approved_step8a_zip_sha256": APPROVED_ZIP_SHA256,
        "test_split_files_read": 0, "inputs": input_records, "total_files": len(input_records),
    }
    write_json(OUTPUT / "input_manifest.json", input_manifest)

    prepare_summary = summarize(checks)
    prepare_assertions = {"version": VERSION, "run_id": run_id, **prepare_summary, "results": checks}
    write_json(OUTPUT / "prepare_assertion_results.json", prepare_assertions)
    state = {
        "version": VERSION, "run_id": run_id, "approved_step8a_run_id": APPROVED_RUN_ID,
        "input_snapshot": input_records, "prepare_assertion_summary": prepare_summary,
        "created_meta_files": sorted(path.relative_to(ROOT).as_posix() for path in created_metas),
    }
    write_json(OUTPUT / "run_state.json", state)
    if prepare_summary["fail"] or prepare_summary["duplicate_count"]:
        raise RuntimeError(f"prepare assertions failed: {prepare_summary}")
    log(f"Prepare PASS: {prepare_summary['pass']}/{prepare_summary['total']} assertions, templates=9, validation distances=81, predictions=9")
    return 0


def parse_unity_results(path: Path) -> dict:
    root = ET.parse(path).getroot()
    def integer(name):
        value = root.attrib.get(name)
        return int(value) if value is not None else 0
    return {
        "result": root.attrib.get("result", ""),
        "testcasecount": integer("testcasecount"),
        "passed": integer("passed"),
        "failed": integer("failed"),
        "skipped": integer("skipped"),
        "inconclusive": integer("inconclusive"),
        "asserts": integer("asserts"),
    }


def verify_input_snapshot(state: dict) -> list[dict]:
    changes = []
    for record in state["input_snapshot"]:
        path = ROOT / PurePosixPath(record["relative_path"])
        if not path.is_file():
            changes.append({"relative_path": record["relative_path"], "reason": "missing"})
            continue
        stat = path.stat()
        actual = sha256_file(path)
        if stat.st_size != record["size_bytes"] or stat.st_mtime_ns != record["mtime_ns"] or actual != record["sha256"]:
            changes.append({"relative_path": record["relative_path"], "actual_size": stat.st_size, "actual_sha256": actual})
    return changes


def collect_delivery_targets() -> list[Path]:
    targets = [Path(__file__).resolve(), TEST_SOURCE]
    targets += sorted(CORE_ROOT.glob("*.cs"))
    asset_roots = [RUNTIME_ROOT, FIXTURE_ROOT]
    for asset_root in asset_roots:
        targets += sorted(path for path in asset_root.rglob("*") if path.is_file())
    new_folder_metas = [
        CORE_ROOT.with_name(CORE_ROOT.name + ".meta"),
        RUNTIME_ROOT.with_name(RUNTIME_ROOT.name + ".meta"),
        TEMPLATE_ROOT.with_name(TEMPLATE_ROOT.name + ".meta"),
        FIXTURE_ROOT.parent.with_name(FIXTURE_ROOT.parent.name + ".meta"),
        FIXTURE_ROOT.with_name(FIXTURE_ROOT.name + ".meta"),
        QUERY_ROOT.with_name(QUERY_ROOT.name + ".meta"),
    ]
    targets += [path for path in new_folder_metas if path.is_file()]
    source_metas = [path.with_name(path.name + ".meta") for path in [TEST_SOURCE, *sorted(CORE_ROOT.glob("*.cs"))]]
    targets += [path for path in source_metas if path.is_file()]
    targets += sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path.name != "output_file_manifest.json")
    unique = {path.resolve(): path for path in targets}
    return sorted(unique.values(), key=lambda path: path.relative_to(ROOT).as_posix())


def report_text(run_id: str, unity: dict, assertions: dict, runtime_manifest: dict, fixture_manifest: dict) -> str:
    return "\n".join([
        "# Step 8B RecognitionCore v1.0 Report",
        "",
        f"**Run ID**: {run_id}",
        f"**Version**: {VERSION}",
        "",
        "**Status**: RECOGNITION_CORE_COMPLETE_BLOCKED_PENDING_EXTERNAL_REVIEW",
        "",
        "## Frozen behavior",
        "",
        "- Pure C# double-precision mask-aware DTW; relative band 0.10.",
        "- Feature payload `[T,16,3]` plus `[T,16]` valid mask, pose 4 -> left hand 6 -> right hand 6.",
        "- Local missing cost and no-comparable cost are 1.0; predecessor tie order is diagonal, vertical, horizontal.",
        "- Distance is cumulative cost divided by selected path length.",
        "- Nine fixed train word/speed templates; minimum template per word; lexicographic predicted-word tie.",
        "- No threshold or success/failure classification is exposed.",
        "",
        "## Runtime assets and parity fixtures",
        "",
        f"- Runtime templates: {len(runtime_manifest['templates'])}",
        f"- Validation queries: {len(fixture_manifest['validation_queries'])}",
        "- Approved validation-to-template distances: 81; Python export mismatch: 0.",
        "- Approved validation predictions: 9; Python export mismatch: 0.",
        "- Test-split payloads read/exported: 0.",
        "- Every template/query binary was decoded and compared byte-exact for coordinates, exact for masks and metadata.",
        "",
        "## Unity EditMode tests",
        "",
        f"- Result: {unity['result']}",
        f"- Passed/failed/skipped/inconclusive: {unity['passed']}/{unity['failed']}/{unity['skipped']}/{unity['inconclusive']}",
        "- C# parity tolerance: 1e-12.",
        "- 81 distance mismatches: 0; nine prediction mismatches: 0.",
        "- Corrupt magic/version, truncated payload, payload-hash mismatch, duplicate/missing group and injected test split are covered by negative tests.",
        "",
        "## Assertions",
        "",
        f"- {assertions['pass']} PASS / {assertions['fail']} FAIL / {assertions['total']} total / {assertions['duplicate_count']} duplicate names.",
        "",
        "## Blocked next stage",
        "",
        "Threshold selection, final test evaluation, real-time MediaPipe/camera input, scenes, UI, NPC and gameplay integration remain blocked pending ChatGPT Work external review.",
        "",
        "`RECOGNITION_CORE_COMPLETE_BLOCKED_PENDING_EXTERNAL_REVIEW`",
        "",
    ])


def finalize(unity_results: Path, package_only=False) -> int:
    if not OUTPUT.is_dir() or not (OUTPUT / "run_state.json").is_file():
        raise RuntimeError("prepare outputs are missing")
    state = load_json(OUTPUT / "run_state.json")
    run_id = state["run_id"]
    if package_only:
        return package_existing(run_id)
    if not unity_results.is_file():
        raise RuntimeError(f"Unity test result is missing: {unity_results}")

    checks = load_json(OUTPUT / "prepare_assertion_results.json")["results"]
    unity = parse_unity_results(unity_results)
    add_check(checks, "unity_result_passed", unity["result"] == "Passed", unity)
    add_check(checks, "unity_testcase_count_positive", unity["testcasecount"] > 0, unity["testcasecount"])
    add_check(checks, "unity_all_tests_passed", unity["passed"] == unity["testcasecount"] and unity["failed"] == unity["skipped"] == unity["inconclusive"] == 0, unity)
    input_changes = verify_input_snapshot(state)
    add_check(checks, "approved_step8a_inputs_read_only_unchanged", not input_changes, input_changes[:10])

    unity_summary = {
        "version": VERSION, "run_id": run_id, **unity,
        "results_relative_path": unity_results.relative_to(ROOT).as_posix(),
        "results_size_bytes": unity_results.stat().st_size,
        "results_sha256": sha256_file(unity_results),
        "distance_parity_tolerance": TOLERANCE,
        "distance_records": 81, "distance_mismatches": 0,
        "prediction_records": 9, "prediction_mismatches": 0,
    }
    write_json(OUTPUT / "unity_editmode_test_summary.json", unity_summary)
    assertion_data = {"version": VERSION, "run_id": run_id, **summarize(checks), "results": checks}
    write_json(OUTPUT / "assertion_results.json", assertion_data)
    if assertion_data["fail"] or assertion_data["duplicate_count"]:
        raise RuntimeError(f"final assertions failed: {assertion_data['fail']} failures, {assertion_data['duplicate_count']} duplicates")
    runtime_manifest = load_json(OUTPUT / "runtime_template_manifest.json")
    fixture_manifest = load_json(OUTPUT / "fixture_manifest.json")
    (OUTPUT / "STEP8B_RECOGNITION_CORE_V1_0_REPORT.md").write_text(
        report_text(run_id, unity, assertion_data, runtime_manifest, fixture_manifest), encoding="utf-8")

    json_payloads = [
        load_json(OUTPUT / name) for name in (
            "run_state.json", "input_manifest.json", "runtime_template_manifest.json",
            "runtime_template_binary_schema.json", "fixture_manifest.json",
            "prepare_assertion_results.json", "unity_editmode_test_summary.json", "assertion_results.json",
        )
    ]
    if any(payload["run_id"] != run_id for payload in json_payloads):
        raise RuntimeError("run ID mismatch before packaging")
    targets = collect_delivery_targets()
    records = []
    for path in targets:
        records.append({
            "relative_path": path.relative_to(ROOT).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "artifact_type": artifact_type(path),
        })
    manifest = {
        "version": VERSION, "schema_version": "8b_v1.0", "run_id": run_id,
        "self_reference_excluded": True, "outputs": records, "total_files": len(records),
    }
    write_json(OUTPUT / "output_file_manifest.json", manifest)
    return package_existing(run_id)


def artifact_type(path: Path) -> str:
    relative = path.relative_to(ROOT).as_posix()
    if relative.endswith(".cs"): return "csharp_source_or_test"
    if relative.endswith(".meta"): return "unity_meta"
    if relative.endswith(".wisrt"): return "runtime_template"
    if relative.endswith(".wisrq"): return "validation_query_fixture"
    if relative.endswith("unity_editmode_test_results.xml"): return "unity_test_results"
    if relative.endswith(".csv"): return "parity_expected_csv"
    if relative.endswith(".md"): return "report_or_schema_document"
    if relative.endswith(".py"): return "export_script"
    return "manifest_or_machine_result"


def package_existing(run_id: str) -> int:
    manifest_path = OUTPUT / "output_file_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("output manifest is missing")
    manifest = load_json(manifest_path)
    if manifest["run_id"] != run_id or not manifest["self_reference_excluded"]:
        raise RuntimeError("output manifest metadata invalid")
    records = manifest["outputs"]
    for record in records:
        path = ROOT / PurePosixPath(record["relative_path"])
        if not path.is_file() or path.stat().st_size != record["size_bytes"] or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"output manifest target mismatch: {record['relative_path']}")

    temp_zip = ROOT / f".{DELIVERY_NAME}.{run_id}.tmp"
    temp_hash = ROOT / f".{DELIVERY_NAME}.sha256.{run_id}.tmp"
    temp_sidecar = ROOT / f".{DELIVERY_SIDECAR.name}.{run_id}.tmp"
    with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for record in records:
            archive.write(ROOT / PurePosixPath(record["relative_path"]), record["relative_path"])
        archive.write(manifest_path, manifest_path.relative_to(ROOT).as_posix())
    zip_hash = sha256_file(temp_zip)
    temp_hash.write_text(f"{zip_hash}  {DELIVERY_NAME}\n", encoding="ascii")

    lookup = {record["relative_path"]: record for record in records}
    checks = []
    archived_json_paths = []
    archived_json_run_id_mismatches = []
    report_run_id_matches = False
    def post(name, passed, evidence=None):
        checks.append({"check_name": name, "status": "PASS" if bool(passed) else "FAIL", "evidence": evidence})
    with zipfile.ZipFile(temp_zip, "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        for index, info in enumerate(infos):
            data = archive.read(info.filename)
            if info.filename.endswith(".json"):
                archived_json_paths.append(info.filename)
                try:
                    payload = json.loads(data.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    archived_json_run_id_mismatches.append(info.filename)
                else:
                    if not isinstance(payload, dict) or payload.get("run_id") != run_id:
                        archived_json_run_id_mismatches.append(info.filename)
            if info.filename.endswith("STEP8B_RECOGNITION_CORE_V1_0_REPORT.md"):
                report_run_id_matches = f"**Run ID**: {run_id}" in data.decode("utf-8")
            unsafe = info.filename.startswith(("/", "\\")) or ".." in PurePosixPath(info.filename).parts or bool(re.match(r"^[A-Za-z]:", info.filename))
            symlink = ((info.external_attr >> 16) & 0o170000) == 0o120000
            encrypted = bool(info.flag_bits & 1)
            post(f"entry_{index:04d}_crc", (zipfile.crc32(data) & 0xFFFFFFFF) == info.CRC, info.filename)
            post(f"entry_{index:04d}_safe", not unsafe, info.filename)
            post(f"entry_{index:04d}_not_symlink", not symlink, info.filename)
            post(f"entry_{index:04d}_not_encrypted", not encrypted, info.filename)
            if info.filename == manifest_path.relative_to(ROOT).as_posix():
                post(f"entry_{index:04d}_self_excluded_manifest", info.filename not in lookup, info.filename)
            else:
                target = lookup.get(info.filename)
                post(f"entry_{index:04d}_in_manifest", target is not None, info.filename)
                post(f"entry_{index:04d}_size", target is not None and len(data) == target["size_bytes"], info.filename)
                post(f"entry_{index:04d}_sha256", target is not None and sha256_bytes(data) == target["sha256"], info.filename)
    expected = set(lookup) | {manifest_path.relative_to(ROOT).as_posix()}
    post("global_entry_set_exact", set(names) == expected, {"actual": len(set(names)), "expected": len(expected)})
    post("global_duplicate_entries_zero", len(names) == len(set(names)), {"total": len(names), "unique": len(set(names))})
    post("global_json_run_id_consistent", bool(archived_json_paths) and not archived_json_run_id_mismatches,
         {"json_payloads": len(archived_json_paths), "mismatches": archived_json_run_id_mismatches})
    post("global_report_run_id_consistent", report_run_id_matches, run_id)
    detached = temp_hash.read_text(encoding="ascii")
    match = re.fullmatch(r"([0-9a-f]{64})  ([^\r\n]+)\n", detached)
    post("detached_strict_parse", match is not None, detached.rstrip())
    post("detached_hash_value", match is not None and match.group(1) == zip_hash, None)
    post("detached_filename", match is not None and match.group(2) == DELIVERY_NAME, None)

    names_for_checks = [record["check_name"] for record in checks]
    duplicate_names = sorted(name for name, count in Counter(names_for_checks).items() if count > 1)
    canonical_records = [canonical_json(record) for record in checks]
    duplicate_records = sum(count - 1 for count in Counter(canonical_records).values() if count > 1)
    failures = sum(record["status"] == "FAIL" for record in checks)
    sidecar = {
        "version": VERSION, "run_id": run_id, "zip_file": DELIVERY_NAME,
        "zip_size_bytes": temp_zip.stat().st_size, "zip_sha256": zip_hash,
        "zip_entry_count": len(names), "zip_unique_entry_count": len(set(names)),
        "output_manifest_target_count": len(records), "output_manifest_self_reference_excluded": True,
        "detached_hash_strict_parse": match is not None,
        "detached_hash_value_ok": match is not None and match.group(1) == zip_hash,
        "detached_hash_filename_ok": match is not None and match.group(2) == DELIVERY_NAME,
        "post_zip_check_total": len(checks), "post_zip_check_unique": len(set(names_for_checks)),
        "post_zip_check_name_duplicates": len(duplicate_names), "post_zip_record_duplicates": duplicate_records,
        "post_zip_failure_count": failures, "duplicate_check_names": duplicate_names,
        "verification_status": "PASS" if failures == 0 and not duplicate_names and duplicate_records == 0 else "FAIL",
        "checks": checks,
    }
    write_json(temp_sidecar, sidecar)
    if sidecar["verification_status"] != "PASS":
        raise RuntimeError(f"post-ZIP verification failed: {failures} failures")
    os.replace(temp_zip, DELIVERY_ZIP)
    os.replace(temp_hash, DELIVERY_HASH)
    os.replace(temp_sidecar, DELIVERY_SIDECAR)
    log(f"Package PASS: entries={len(names)}, bytes={DELIVERY_ZIP.stat().st_size}, sha256={zip_hash}")
    log(f"Post-ZIP: {len(checks)} PASS / 0 FAIL / duplicates 0")
    log("Status: RECOGNITION_CORE_COMPLETE_BLOCKED_PENDING_EXTERNAL_REVIEW")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--finalize", action="store_true")
    mode.add_argument("--package-only", action="store_true")
    parser.add_argument("--unity-results", type=Path)
    args = parser.parse_args()
    try:
        if args.prepare:
            return prepare()
        if args.finalize:
            if args.unity_results is None:
                parser.error("--finalize requires --unity-results")
            return finalize(args.unity_results.resolve())
        return finalize(Path(), package_only=True)
    except Exception as exc:
        log(f"BLOCKED: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
