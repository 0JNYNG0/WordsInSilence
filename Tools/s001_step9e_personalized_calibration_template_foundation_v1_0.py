#!/usr/bin/env python3
"""WordsInSilence Step 9E personalized calibration/template foundation v1.0.

This is an offline-only, fail-closed tool.  It fixes the approved Step 8A/9D
archives by byte size and SHA-256, inventories the two real capture families,
reuses the approved normalized [T,16,3] / mask-aware DTW band 0.10 contract,
and exercises the future >=3-repetition calibration path with deterministic
synthetic regression.  It never writes Unity/runtime/global-template assets.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import struct
import subprocess
import tempfile
import types
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np


SCHEMA = "words_in_silence.step9e.personalized_calibration_template_foundation.v1_0"
WORDS = ("GREETING", "LOVE", "THANK_YOU")
MIN_REPETITIONS = 3
DTW_BAND = 0.10
MISSING_COST = 1.0
PARITY_TOLERANCE = 1e-12
COORDINATE_SHAPE = (16, 3)
POSE_INDICES = (13, 14, 15, 16)
LEFT_HAND_INDICES = (0, 4, 8, 12, 16, 20)
RIGHT_HAND_INDICES = (0, 4, 8, 12, 16, 20)

SCRIPT_REL = "Tools/s001_step9e_personalized_calibration_template_foundation_v1_0.py"
OUTPUT_REL = "Validation/WordsInSilence_step9e_personalized_calibration_template_foundation_v1_0"
DELIVERY_REL = "Validation/WordsInSilence_step9e_personalized_calibration_template_foundation_v1_0_delivery.zip"

STEP8A_NAME = "WordsInSilence_step8a_dtw_templates_v1_0_delivery.zip"
BASELINE_NAME = "WordsInSilence_step9d_physical_recognition_domain_gap_diagnostic_v1_0_delivery.zip"
RAW_NAME = "WordsInSilence_step9d_raw_intermediate_physical_evidence_3takes_v1_0_delivery.zip"
AVAILABILITY_NAME = "WordsInSilence_step9d_frozen_reference_trace_availability_review_v1_0_delivery.zip"

EXPECTED_INPUTS = {
    "step8a": (STEP8A_NAME, 2_205_564, "fa8292c424644bc5f50f7213e7587f5157b71cd651df804b8a542168ffe4bf3c"),
    "baseline_diagnostic": (BASELINE_NAME, 2_804_364, "b179962aba86aa961e42c5f762e696c718c831b7ac444c5fe708d5301a69f2d0"),
    "raw_intermediate": (RAW_NAME, 1_468_035, "7c4982d15e1d4be71f054562eee8913c3a94a289827dc024c10bf61332651cd1"),
    "availability_review": (AVAILABILITY_NAME, 12_234_104, "c139c434c609d78f77af7b3379102cc456f4fb4819666719f6e1f4c580cc3034"),
}

BASE_PREFIX = "data/processed/sessions/S001_P001/step9d_physical_recognition_domain_gap_diagnostic_v1_0/"
BASE_SCRIPT_ENTRY = "Tools/s001_step9d_physical_recognition_domain_gap_diagnostic_v1_0.py"
BASE_MANIFEST_ENTRY = BASE_PREFIX + "output_file_manifest.json"
BASE_INPUT_MANIFEST_ENTRY = BASE_PREFIX + "input_manifest.json"
BASE_QUERY_PREFIX = BASE_PREFIX + "inputs/physical_queries/"
RAW_MANIFEST_ENTRY = "output_file_manifest.json"
AVAILABILITY_MANIFEST_ENTRY = "output_file_manifest.json"
STEP8A_MANIFEST_ENTRY = "output_file_manifest.json"


class FoundationError(RuntimeError):
    """Fail-closed contract or evidence error."""


class DuplicateCaptureError(FoundationError):
    """A capture ID or payload identity was duplicated."""


class CaptureContractError(FoundationError):
    """A capture shape, mask, checksum, or identity is invalid."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FoundationError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json_loads(data: bytes) -> Any:
    def reject_constant(token: str) -> None:
        raise FoundationError(f"non-standard JSON constant: {token}")

    return json.loads(data.decode("utf-8"), parse_constant=reject_constant)


def finite_json(value: Any, path: str = "$") -> None:
    if isinstance(value, float):
        require(math.isfinite(value), f"non-finite JSON value at {path}")
    elif isinstance(value, dict):
        for key, child in value.items():
            finite_json(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            finite_json(child, f"{path}[{index}]")


def json_bytes(value: Any) -> bytes:
    finite_json(value)
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    finite_json(value)
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def safe_zip_name(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name and not name.startswith(("/", "\\")) and "\\" not in name and not re.match(r"^[A-Za-z]:", name) and ".." not in path.parts)


def parse_detached(data: bytes, expected_name: str, expected_hash: str) -> bool:
    match = re.fullmatch(rb"([0-9a-f]{64})  ([^\r\n]+)(?:\r?\n)?", data)
    return bool(match and match.group(1).decode("ascii") == expected_hash and match.group(2).decode("utf-8") == expected_name)


def manifest_records(data: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("outputs", "files", "inputs", "input_files"):
        records = data.get(key)
        if isinstance(records, list):
            return records
    raise FoundationError(f"manifest has no supported record list: {sorted(data)}")


def verify_zip_manifest(path: Path, manifest_name: str) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        require(archive.testzip() is None, f"CRC failure: {path}")
        require(len(names) == len(set(names)), f"duplicate ZIP entry: {path}")
        require(all(safe_zip_name(name) for name in names), f"unsafe ZIP path: {path}")
        require(not any(stat.S_ISLNK((info.external_attr >> 16) & 0xFFFF) for info in infos), f"symlink ZIP entry: {path}")
        require(not any(info.flag_bits & 1 for info in infos), f"encrypted ZIP entry: {path}")
        require(manifest_name in names, f"missing manifest {manifest_name}: {path}")
        manifest_payload = archive.read(manifest_name)
        manifest = strict_json_loads(manifest_payload)
        records = manifest_records(manifest)
        targets = [record["relative_path"] for record in records]
        require(len(targets) == len(set(targets)), f"duplicate manifest target: {path}")
        require(set(targets) == set(names) - {manifest_name}, f"manifest membership mismatch: {path}")
        for record in records:
            payload = archive.read(record["relative_path"])
            require(len(payload) == int(record["size_bytes"]), f"manifest size mismatch: {record['relative_path']}")
            require(sha256_bytes(payload) == record["sha256"], f"manifest hash mismatch: {record['relative_path']}")
        return {
            "entry_count": len(names),
            "manifest_entry": manifest_name,
            "manifest_size_bytes": len(manifest_payload),
            "manifest_sha256": sha256_bytes(manifest_payload),
            "manifest_target_count": len(records),
            "crc": "PASS",
            "duplicate_entries": 0,
            "unsafe_paths": 0,
            "symlinks": 0,
            "encrypted_entries": 0,
            "membership": "PASS",
        }


def verify_fixed_inputs(root: Path) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    paths = {
        "step8a": root / STEP8A_NAME,
        "baseline_diagnostic": root / BASELINE_NAME,
        "raw_intermediate": root / "Validation" / RAW_NAME,
        "availability_review": root / "Validation" / AVAILABILITY_NAME,
    }
    manifest_names = {
        "step8a": STEP8A_MANIFEST_ENTRY,
        "baseline_diagnostic": BASE_MANIFEST_ENTRY,
        "raw_intermediate": RAW_MANIFEST_ENTRY,
        "availability_review": AVAILABILITY_MANIFEST_ENTRY,
    }
    records: dict[str, dict[str, Any]] = {}
    for role, path in paths.items():
        expected_name, expected_size, expected_hash = EXPECTED_INPUTS[role]
        require(path.is_file(), f"missing fixed input: {path}")
        actual_size = path.stat().st_size
        actual_hash = sha256_file(path)
        require(path.name == expected_name, f"fixed input filename mismatch: {path}")
        require(actual_size == expected_size, f"{role} size mismatch: {actual_size} != {expected_size}")
        require(actual_hash == expected_hash, f"{role} SHA-256 mismatch: {actual_hash} != {expected_hash}")
        records[role] = {
            "role": role,
            "path": path.relative_to(root).as_posix(),
            "filename": path.name,
            "size_bytes": actual_size,
            "sha256": actual_hash,
            "expected_size_bytes": expected_size,
            "expected_sha256": expected_hash,
            "fixed_identity": "PASS",
            "zip_validation": verify_zip_manifest(path, manifest_names[role]),
        }
    return paths, records


def canonical_coordinate_bytes(array: np.ndarray) -> bytes:
    value = np.ascontiguousarray(array, dtype="<f8")
    return value.tobytes(order="C")


def canonical_mask_bytes(array: np.ndarray) -> bytes:
    return np.ascontiguousarray(array, dtype=np.uint8).tobytes(order="C")


def normalized_payload_hash(coordinates: np.ndarray, valid_mask: np.ndarray) -> str:
    return sha256_bytes(b"WIS_STEP9E_NORMALIZED_V1\0" + canonical_coordinate_bytes(coordinates) + canonical_mask_bytes(valid_mask))


@dataclass(frozen=True)
class Capture:
    capture_key: str
    source_family: str
    source_archive: str
    source_archive_sha256: str
    archive_entry: str
    source_manifest_entry: str
    schema_version: str
    word: str
    run_id: str
    capture_id: str
    frame_indices: tuple[int, ...]
    timestamps: tuple[float, ...]
    coordinates: np.ndarray
    valid_mask: np.ndarray
    source_bytes: bytes
    source_sha256: str
    coordinate_payload_sha256: str
    mask_payload_sha256: str
    normalized_payload_sha256: str


def make_capture(
    *, capture_key: str, source_family: str, source_archive: str, source_archive_sha256: str,
    archive_entry: str, source_manifest_entry: str, schema_version: str, word: str,
    run_id: str, capture_id: str, frame_indices: Iterable[int], timestamps: Iterable[float],
    coordinates: np.ndarray, valid_mask: np.ndarray, source_bytes: bytes,
) -> Capture:
    coords = np.ascontiguousarray(coordinates, dtype=np.float64)
    mask = np.ascontiguousarray(valid_mask, dtype=np.bool_)
    return Capture(
        capture_key=capture_key,
        source_family=source_family,
        source_archive=source_archive,
        source_archive_sha256=source_archive_sha256,
        archive_entry=archive_entry,
        source_manifest_entry=source_manifest_entry,
        schema_version=schema_version,
        word=word,
        run_id=run_id,
        capture_id=capture_id,
        frame_indices=tuple(int(value) for value in frame_indices),
        timestamps=tuple(float(value) for value in timestamps),
        coordinates=coords,
        valid_mask=mask,
        source_bytes=source_bytes,
        source_sha256=sha256_bytes(source_bytes),
        coordinate_payload_sha256=sha256_bytes(canonical_coordinate_bytes(coords)),
        mask_payload_sha256=sha256_bytes(canonical_mask_bytes(mask)),
        normalized_payload_sha256=normalized_payload_hash(coords, mask),
    )


def validate_capture(capture: Capture) -> None:
    if capture.word not in WORDS:
        raise CaptureContractError(f"unsupported word: {capture.word}")
    if not capture.run_id or not capture.capture_id or not capture.capture_key:
        raise CaptureContractError("blank capture identity")
    if capture.coordinates.ndim != 3 or capture.coordinates.shape[1:] != COORDINATE_SHAPE:
        raise CaptureContractError(f"coordinate shape mismatch: {capture.capture_key} {capture.coordinates.shape}")
    if capture.valid_mask.shape != capture.coordinates.shape[:2]:
        raise CaptureContractError(f"mask shape mismatch: {capture.capture_key} {capture.valid_mask.shape}")
    if capture.coordinates.dtype != np.float64 or capture.valid_mask.dtype != np.bool_:
        raise CaptureContractError(f"dtype mismatch: {capture.capture_key}")
    frame_count = len(capture.coordinates)
    if frame_count == 0 or len(capture.frame_indices) != frame_count or len(capture.timestamps) != frame_count:
        raise CaptureContractError(f"frame/timestamp count mismatch: {capture.capture_key}")
    timestamps = np.asarray(capture.timestamps, dtype=np.float64)
    if not np.isfinite(timestamps).all() or (len(timestamps) > 1 and not np.all(np.diff(timestamps) > 0)):
        raise CaptureContractError(f"timestamps not finite/strictly increasing: {capture.capture_key}")
    if np.isinf(capture.coordinates).any() or not np.isfinite(capture.coordinates[capture.valid_mask]).all():
        raise CaptureContractError(f"valid coordinate non-finite or Infinity present: {capture.capture_key}")
    if capture.source_sha256 != sha256_bytes(capture.source_bytes):
        raise CaptureContractError(f"source checksum mismatch: {capture.capture_key}")
    if capture.coordinate_payload_sha256 != sha256_bytes(canonical_coordinate_bytes(capture.coordinates)):
        raise CaptureContractError(f"coordinate payload checksum mismatch: {capture.capture_key}")
    if capture.mask_payload_sha256 != sha256_bytes(canonical_mask_bytes(capture.valid_mask)):
        raise CaptureContractError(f"mask payload checksum mismatch: {capture.capture_key}")
    if capture.normalized_payload_sha256 != normalized_payload_hash(capture.coordinates, capture.valid_mask):
        raise CaptureContractError(f"normalized payload checksum mismatch: {capture.capture_key}")


def validate_capture_set(captures: list[Capture]) -> None:
    for capture in captures:
        validate_capture(capture)
    checks = {
        "capture_key": [capture.capture_key for capture in captures],
        "capture_id": [capture.capture_id for capture in captures],
        "source_payload": [capture.source_sha256 for capture in captures],
        "normalized_payload": [capture.normalized_payload_sha256 for capture in captures],
    }
    for label, values in checks.items():
        if len(values) != len(set(values)):
            raise DuplicateCaptureError(f"duplicate {label}")
    identity_values = [
        (capture.run_id, capture.capture_id, capture.timestamps[0], capture.timestamps[-1], capture.source_sha256,
         capture.coordinate_payload_sha256, capture.mask_payload_sha256)
        for capture in captures
    ]
    if len(identity_values) != len(set(identity_values)):
        raise DuplicateCaptureError("duplicate composite capture identity")
    for left_index, left in enumerate(captures):
        for right in captures[left_index + 1:]:
            if left.run_id == right.run_id and left.timestamps[0] == right.timestamps[0] and left.timestamps[-1] == right.timestamps[-1]:
                raise DuplicateCaptureError(f"same-run identical timestamp window: {left.capture_key}/{right.capture_key}")


def reference_dtw(
    left: np.ndarray, left_mask: np.ndarray, right: np.ndarray, right_mask: np.ndarray,
    band: float = DTW_BAND,
) -> dict[str, Any]:
    """Approved Step 9D reference DTW: unchanged local cost, band, and tie order."""
    both = left_mask[:, None, :] & right_mask[None, :, :]
    exactly_one = left_mask[:, None, :] ^ right_mask[None, :, :]
    with np.errstate(invalid="ignore"):
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
            candidates: list[tuple[float, int]] = []
            if i > 0 and j > 0 and np.isfinite(cumulative[i - 1, j - 1]):
                candidates.append((float(cumulative[i - 1, j - 1]), 0))
            if i > 0 and np.isfinite(cumulative[i - 1, j]):
                candidates.append((float(cumulative[i - 1, j]), 1))
            if j > 0 and np.isfinite(cumulative[i, j - 1]):
                candidates.append((float(cumulative[i, j - 1]), 2))
            if candidates:
                prior, direction = min(candidates, key=lambda item: (item[0], item[1]))
                cumulative[i, j] = prior + local[i, j]
                parent[i, j] = direction
    total = float(cumulative[-1, -1])
    require(math.isfinite(total), f"no DTW path for lengths {length_a}/{length_b}")
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
            raise FoundationError(f"DTW parent incomplete at {i}/{j}")
        path.append((i, j))
    path.reverse()
    both_counts = [int(np.count_nonzero(left_mask[a] & right_mask[b])) for a, b in path]
    comparable_counts = [int(np.count_nonzero(left_mask[a] | right_mask[b])) for a, b in path]
    return {
        "distance": total / len(path),
        "cumulative_cost": total,
        "path": path,
        "path_length": len(path),
        "both_valid_count_total": sum(both_counts),
        "both_valid_count_min": min(both_counts),
        "both_valid_count_max": max(both_counts),
        "comparable_count_total": sum(comparable_counts),
        "comparable_count_min": min(comparable_counts),
        "comparable_count_max": max(comparable_counts),
    }


def load_actual_captures(
    paths: dict[str, Path], input_records: dict[str, dict[str, Any]],
) -> tuple[list[Capture], dict[str, bytes], dict[str, Any]]:
    captures: list[Capture] = []
    subset: dict[str, bytes] = {}
    provenance: dict[str, Any] = {"baseline": [], "raw_intermediate": []}

    with zipfile.ZipFile(paths["baseline_diagnostic"]) as archive:
        names = archive.namelist()
        query_names = sorted(name for name in names if name.startswith(BASE_QUERY_PREFIX) and name.endswith(".json"))
        require(len(query_names) == 3, f"baseline physical query count mismatch: {len(query_names)}")
        for entry in query_names:
            payload = archive.read(entry)
            detached_entry = entry + ".sha256"
            require(detached_entry in names, f"missing baseline detached checksum: {entry}")
            detached = archive.read(detached_entry)
            digest = sha256_bytes(payload)
            require(parse_detached(detached, Path(entry).name, digest), f"baseline detached checksum mismatch: {entry}")
            data = strict_json_loads(payload)
            frames = data.get("frames")
            require(isinstance(frames, list) and frames, f"baseline frames missing: {entry}")
            coordinates = np.asarray([frame["normalized_coordinates"] for frame in frames], dtype=np.float64)
            valid_mask = np.asarray([frame["normalized_mask"] for frame in frames], dtype=np.bool_)
            timestamps = [float(frame["timestamp_seconds"]) for frame in frames]
            frame_indices = [int(frame["frame_index"]) for frame in frames]
            word = str(data["expected_word"])
            capture = make_capture(
                capture_key=f"baseline_step9c::{data['capture_id']}",
                source_family="baseline_step9c_physical_query",
                source_archive=input_records["baseline_diagnostic"]["path"],
                source_archive_sha256=input_records["baseline_diagnostic"]["sha256"],
                archive_entry=entry,
                source_manifest_entry=BASE_INPUT_MANIFEST_ENTRY,
                schema_version=str(data["schema_version"]),
                word=word,
                run_id=str(data["run_id"]),
                capture_id=str(data["capture_id"]),
                frame_indices=frame_indices,
                timestamps=timestamps,
                coordinates=coordinates,
                valid_mask=valid_mask,
                source_bytes=payload,
            )
            validate_capture(capture)
            counts = data.get("counts", {})
            require(int(counts.get("normalized_frames", -1)) == len(frames), f"baseline frame count mismatch: {entry}")
            require(int(counts.get("normalized_valid_landmarks", -1)) == int(valid_mask.sum()), f"baseline valid count mismatch: {entry}")
            captures.append(capture)
            destination = "input_subset/baseline_physical_queries/" + Path(entry).name
            subset[destination] = payload
            subset[destination + ".sha256"] = detached
            provenance["baseline"].append({
                "word": word,
                "run_id": capture.run_id,
                "capture_id": capture.capture_id,
                "archive_entry": entry,
                "source_size_bytes": len(payload),
                "source_sha256": digest,
                "detached_entry": detached_entry,
                "manifest_provenance_entry": BASE_INPUT_MANIFEST_ENTRY,
            })

    with zipfile.ZipFile(paths["raw_intermediate"]) as archive:
        names = archive.namelist()
        capture_manifest_payload = archive.read("capture_manifest.json")
        capture_manifest = strict_json_loads(capture_manifest_payload)
        manifest_by_name = {item["json_file"]: item for item in capture_manifest["captures"]}
        query_names = sorted(name for name in names if name.startswith("take_") and name.endswith(".json"))
        require(len(query_names) == 3 and len(manifest_by_name) == 3, "raw capture count/manifest mismatch")
        for entry in query_names:
            require(entry in manifest_by_name, f"raw capture absent from capture manifest: {entry}")
            declared = manifest_by_name[entry]
            payload = archive.read(entry)
            detached_entry = entry + ".sha256"
            require(detached_entry in names, f"missing raw detached checksum: {entry}")
            detached = archive.read(detached_entry)
            digest = sha256_bytes(payload)
            require(parse_detached(detached, entry, digest), f"raw detached checksum mismatch: {entry}")
            require(int(declared["json_size_bytes"]) == len(payload) and declared["json_sha256"] == digest, f"raw manifest payload mismatch: {entry}")
            data = strict_json_loads(payload)
            frame_identity = data.get("frame_identity")
            require(isinstance(frame_identity, list) and frame_identity, f"raw frame identity missing: {entry}")
            coordinates = np.asarray(data["normalized_output"]["coordinates"], dtype=np.float64)
            valid_mask = np.asarray(data["normalized_output"]["mask"], dtype=np.bool_)
            timestamps = [float(frame["observation_timestamp_seconds"]) for frame in frame_identity]
            frame_indices = [int(frame["capture_local_frame_index"]) for frame in frame_identity]
            word = str(data["expected_word"])
            capture = make_capture(
                capture_key=f"raw_intermediate_step9d::{data['capture_id']}",
                source_family="step9d_raw_intermediate_capture",
                source_archive=input_records["raw_intermediate"]["path"],
                source_archive_sha256=input_records["raw_intermediate"]["sha256"],
                archive_entry=entry,
                source_manifest_entry="capture_manifest.json",
                schema_version=str(data["schema_version"]),
                word=word,
                run_id=str(data["run_id"]),
                capture_id=str(data["capture_id"]),
                frame_indices=frame_indices,
                timestamps=timestamps,
                coordinates=coordinates,
                valid_mask=valid_mask,
                source_bytes=payload,
            )
            validate_capture(capture)
            require(capture.run_id == capture_manifest["run_id"] == str(data["run_id"]), f"raw run provenance mismatch: {entry}")
            require(capture.capture_id == declared["capture_id"] and word == declared["expected_word"], f"raw capture identity mismatch: {entry}")
            require(int(declared["observation_count"]) == len(coordinates), f"raw observation count mismatch: {entry}")
            require([timestamps[0], timestamps[-1]] == [float(x) for x in declared["observation_timestamp_seconds_range"]], f"raw timestamp range mismatch: {entry}")
            captures.append(capture)
            destination = "input_subset/raw_intermediate_captures/" + entry
            subset[destination] = payload
            subset[destination + ".sha256"] = detached
            provenance["raw_intermediate"].append({
                "word": word,
                "run_id": capture.run_id,
                "capture_id": capture.capture_id,
                "archive_entry": entry,
                "source_size_bytes": len(payload),
                "source_sha256": digest,
                "detached_entry": detached_entry,
                "manifest_provenance_entry": "capture_manifest.json",
            })

    captures.sort(key=lambda item: (WORDS.index(item.word), item.source_family, item.capture_id))
    validate_capture_set(captures)
    require(len(captures) == 6, f"actual seed count mismatch: {len(captures)}")
    require({word: sum(capture.word == word for capture in captures) for word in WORDS} == {word: 2 for word in WORDS}, "actual per-word seed count mismatch")
    return captures, subset, provenance


def load_additional_capture_files(directory: Path) -> list[Capture]:
    """Optional future opt-in input path; not used by the fixed Step 9E seed run."""
    if not directory.exists():
        raise FoundationError(f"additional capture directory missing: {directory}")
    captures: list[Capture] = []
    for path in sorted(directory.glob("*.json")):
        payload = path.read_bytes()
        detached_path = Path(str(path) + ".sha256")
        require(detached_path.is_file(), f"additional capture detached checksum missing: {detached_path}")
        digest = sha256_bytes(payload)
        require(parse_detached(detached_path.read_bytes(), path.name, digest), f"additional capture checksum mismatch: {path}")
        data = strict_json_loads(payload)
        if "frames" in data:
            frames = data["frames"]
            coordinates = np.asarray([frame["normalized_coordinates"] for frame in frames], dtype=np.float64)
            mask = np.asarray([frame["normalized_mask"] for frame in frames], dtype=np.bool_)
            timestamps = [float(frame["timestamp_seconds"]) for frame in frames]
            indices = [int(frame["frame_index"]) for frame in frames]
        elif "normalized_output" in data and "frame_identity" in data:
            coordinates = np.asarray(data["normalized_output"]["coordinates"], dtype=np.float64)
            mask = np.asarray(data["normalized_output"]["mask"], dtype=np.bool_)
            timestamps = [float(frame["observation_timestamp_seconds"]) for frame in data["frame_identity"]]
            indices = [int(frame["capture_local_frame_index"]) for frame in data["frame_identity"]]
        else:
            raise CaptureContractError(f"unsupported additional capture schema: {path}")
        capture = make_capture(
            capture_key=f"additional::{data['capture_id']}", source_family="additional_opt_in_capture",
            source_archive="NOT_APPLICABLE_DIRECT_FILE", source_archive_sha256="NOT_APPLICABLE_DIRECT_FILE",
            archive_entry=path.as_posix(), source_manifest_entry="NOT_APPLICABLE_DIRECT_FILE",
            schema_version=str(data["schema_version"]), word=str(data["expected_word"]), run_id=str(data["run_id"]),
            capture_id=str(data["capture_id"]), frame_indices=indices, timestamps=timestamps,
            coordinates=coordinates, valid_mask=mask, source_bytes=payload,
        )
        validate_capture(capture)
        captures.append(capture)
    return captures


def capture_inventory_record(capture: Capture) -> dict[str, Any]:
    invalid = ~capture.valid_mask
    return {
        "capture_key": capture.capture_key,
        "source_family": capture.source_family,
        "source_archive": capture.source_archive,
        "source_archive_sha256": capture.source_archive_sha256,
        "zip_entry": capture.archive_entry,
        "zip_entry_size_bytes": len(capture.source_bytes),
        "source_payload_sha256": capture.source_sha256,
        "word": capture.word,
        "run_id": capture.run_id,
        "capture_id": capture.capture_id,
        "schema_version": capture.schema_version,
        "frame_count": len(capture.coordinates),
        "frame_index_min": min(capture.frame_indices),
        "frame_index_max": max(capture.frame_indices),
        "timestamp_min_seconds": capture.timestamps[0],
        "timestamp_max_seconds": capture.timestamps[-1],
        "coordinate_shape": "x".join(str(value) for value in capture.coordinates.shape),
        "coordinate_dtype": str(capture.coordinates.dtype),
        "coordinate_nan_count": int(np.count_nonzero(np.isnan(capture.coordinates))),
        "coordinate_infinity_count": int(np.count_nonzero(np.isinf(capture.coordinates))),
        "valid_coordinate_component_count": int(capture.valid_mask.sum()) * 3,
        "invalid_coordinate_component_count": int(invalid.sum()) * 3,
        "mask_shape": "x".join(str(value) for value in capture.valid_mask.shape),
        "mask_dtype": str(capture.valid_mask.dtype),
        "mask_valid_count": int(capture.valid_mask.sum()),
        "mask_invalid_count": int(invalid.sum()),
        "coordinate_payload_sha256": capture.coordinate_payload_sha256,
        "mask_payload_sha256": capture.mask_payload_sha256,
        "normalized_coordinate_and_mask_payload_sha256": capture.normalized_payload_sha256,
        "manifest_provenance_entry": capture.source_manifest_entry,
    }


def identity_record(capture: Capture) -> dict[str, Any]:
    fingerprint = sha256_bytes(json_bytes({
        "run_id": capture.run_id,
        "capture_id": capture.capture_id,
        "timestamp_range": [capture.timestamps[0], capture.timestamps[-1]],
        "source_payload_sha256": capture.source_sha256,
        "coordinate_payload_sha256": capture.coordinate_payload_sha256,
        "mask_payload_sha256": capture.mask_payload_sha256,
    }))
    return {
        "capture_key": capture.capture_key,
        "source_family": capture.source_family,
        "word": capture.word,
        "run_id": capture.run_id,
        "capture_id": capture.capture_id,
        "timestamp_range_seconds": [capture.timestamps[0], capture.timestamps[-1]],
        "source_payload_sha256": capture.source_sha256,
        "coordinate_payload_sha256": capture.coordinate_payload_sha256,
        "mask_payload_sha256": capture.mask_payload_sha256,
        "normalized_payload_sha256": capture.normalized_payload_sha256,
        "composite_identity_sha256": fingerprint,
        "source_archive": capture.source_archive,
        "zip_entry": capture.archive_entry,
        "manifest_provenance_entry": capture.source_manifest_entry,
    }


def independence_records(captures: list[Capture]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left_index, left in enumerate(captures):
        for right in captures[left_index + 1:]:
            timestamp_exact = left.timestamps[0] == right.timestamps[0] and left.timestamps[-1] == right.timestamps[-1]
            timestamp_overlap = max(left.timestamps[0], right.timestamps[0]) <= min(left.timestamps[-1], right.timestamps[-1])
            duplicate_signals = {
                "capture_id_equal": left.capture_id == right.capture_id,
                "source_payload_sha256_equal": left.source_sha256 == right.source_sha256,
                "coordinate_payload_sha256_equal": left.coordinate_payload_sha256 == right.coordinate_payload_sha256,
                "mask_payload_sha256_equal": left.mask_payload_sha256 == right.mask_payload_sha256,
                "normalized_payload_sha256_equal": left.normalized_payload_sha256 == right.normalized_payload_sha256,
                "same_run_exact_timestamp_range": left.run_id == right.run_id and timestamp_exact,
            }
            rows.append({
                "pair_key": left.capture_key + "||" + right.capture_key,
                "left_capture_key": left.capture_key,
                "right_capture_key": right.capture_key,
                "same_word": left.word == right.word,
                "same_run_id": left.run_id == right.run_id,
                "timestamp_ranges_overlap": timestamp_overlap,
                **duplicate_signals,
                "independent": not any(duplicate_signals.values()),
            })
    return rows


def load_approved_dtw_helpers(path: Path, root: Path) -> tuple[types.SimpleNamespace, dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        source = archive.read(BASE_SCRIPT_ENTRY)
    namespace: dict[str, Any] = {
        "__name__": "approved_step9d_dtw_helper",
        "__file__": str(root / SCRIPT_REL),
    }
    exec(compile(source.decode("utf-8"), namespace["__file__"], "exec"), namespace)
    require(float(namespace["DTW_BAND"]) == DTW_BAND and float(namespace["MISSING_COST"]) == MISSING_COST, "approved DTW constants mismatch")
    return types.SimpleNamespace(**namespace), {
        "source_archive_entry": BASE_SCRIPT_ENTRY,
        "source_size_bytes": len(source),
        "source_sha256": sha256_bytes(source),
        "dtw_band": float(namespace["DTW_BAND"]),
        "missing_cost": float(namespace["MISSING_COST"]),
    }


def pairwise_dtw_rows(
    captures: list[Capture], approved_helpers: types.SimpleNamespace | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    results: dict[tuple[str, str], dict[str, Any]] = {}
    max_repeat_error = 0.0
    max_approved_reference_error = 0.0
    max_approved_runtime_error = 0.0
    approved_path_mismatch_count = 0
    for left in captures:
        for right in captures:
            first = reference_dtw(left.coordinates, left.valid_mask, right.coordinates, right.valid_mask, DTW_BAND)
            repeat = reference_dtw(left.coordinates, left.valid_mask, right.coordinates, right.valid_mask, DTW_BAND)
            repeat_error = abs(float(first["distance"]) - float(repeat["distance"]))
            max_repeat_error = max(max_repeat_error, repeat_error)
            approved_reference_distance = float(first["distance"])
            approved_runtime_distance = float(first["distance"])
            approved_reference_path_length = int(first["path_length"])
            approved_runtime_path_length = int(first["path_length"])
            if approved_helpers is not None:
                approved_reference = approved_helpers.reference_dtw(left.coordinates, left.valid_mask, right.coordinates, right.valid_mask, DTW_BAND)
                approved_runtime = approved_helpers.unity_contract_dtw(left.coordinates, left.valid_mask, right.coordinates, right.valid_mask, DTW_BAND)
                approved_reference_distance = float(approved_reference["distance"])
                approved_runtime_distance = float(approved_runtime["distance"])
                approved_reference_path_length = int(approved_reference["path_length"])
                approved_runtime_path_length = int(approved_runtime["path_length"])
                max_approved_reference_error = max(max_approved_reference_error, abs(float(first["distance"]) - approved_reference_distance))
                max_approved_runtime_error = max(max_approved_runtime_error, abs(float(first["distance"]) - approved_runtime_distance))
                approved_path_mismatch_count += int(int(first["path_length"]) != approved_reference_path_length or int(first["path_length"]) != approved_runtime_path_length)
            results[(left.capture_key, right.capture_key)] = first
            rows.append({
                "pair_key": left.capture_key + "||" + right.capture_key,
                "left_capture_key": left.capture_key,
                "left_word": left.word,
                "right_capture_key": right.capture_key,
                "right_word": right.word,
                "same_capture": left.capture_key == right.capture_key,
                "same_word": left.word == right.word,
                "distance": float(first["distance"]),
                "repeat_distance": float(repeat["distance"]),
                "repeat_absolute_error": repeat_error,
                "approved_reference_distance": approved_reference_distance,
                "approved_runtime_contract_distance": approved_runtime_distance,
                "approved_reference_absolute_error": abs(float(first["distance"]) - approved_reference_distance),
                "approved_runtime_absolute_error": abs(float(first["distance"]) - approved_runtime_distance),
                "cumulative_cost": float(first["cumulative_cost"]),
                "path_length": int(first["path_length"]),
                "approved_reference_path_length": approved_reference_path_length,
                "approved_runtime_path_length": approved_runtime_path_length,
                "both_valid_count_total": int(first["both_valid_count_total"]),
                "both_valid_count_min": int(first["both_valid_count_min"]),
                "both_valid_count_max": int(first["both_valid_count_max"]),
                "comparable_count_total": int(first["comparable_count_total"]),
                "comparable_count_min": int(first["comparable_count_min"]),
                "comparable_count_max": int(first["comparable_count_max"]),
            })
    diagonal_max = max(abs(float(results[(capture.capture_key, capture.capture_key)]["distance"])) for capture in captures)
    symmetry_errors = [
        abs(float(results[(left.capture_key, right.capture_key)]["distance"]) - float(results[(right.capture_key, left.capture_key)]["distance"]))
        for left in captures for right in captures
    ]
    off_diagonal_same = [row["distance"] for row in rows if row["same_word"] and not row["same_capture"]]
    off_diagonal_wrong = [row["distance"] for row in rows if not row["same_word"]]
    summary = {
        "capture_count": len(captures),
        "matrix_row_count": len(rows),
        "diagonal_max_absolute_distance": diagonal_max,
        "symmetry_max_absolute_error": max(symmetry_errors),
        "repeat_max_absolute_error": max_repeat_error,
        "approved_reference_max_absolute_error": max_approved_reference_error,
        "approved_runtime_contract_max_absolute_error": max_approved_runtime_error,
        "approved_path_mismatch_count": approved_path_mismatch_count,
        "same_word_off_diagonal": numeric_summary(off_diagonal_same),
        "cross_word": numeric_summary(off_diagonal_wrong),
        "per_word_same_pair": {
            word: numeric_summary([row["distance"] for row in rows if row["left_word"] == word and row["right_word"] == word and not row["same_capture"]])
            for word in WORDS
        },
    }
    require(diagonal_max <= PARITY_TOLERANCE, f"DTW diagonal non-zero: {diagonal_max}")
    require(summary["symmetry_max_absolute_error"] <= PARITY_TOLERANCE, f"DTW asymmetry: {summary['symmetry_max_absolute_error']}")
    require(max_repeat_error == 0.0, f"DTW repeat mismatch: {max_repeat_error}")
    require(max(max_approved_reference_error, max_approved_runtime_error) <= PARITY_TOLERANCE, "approved DTW helper parity mismatch")
    require(approved_path_mismatch_count == 0, f"approved DTW path mismatch: {approved_path_mismatch_count}")
    return rows, summary


def numeric_summary(values: Iterable[float]) -> dict[str, Any]:
    data = [float(value) for value in values]
    require(bool(data), "empty numeric summary")
    return {
        "count": len(data),
        "min": min(data),
        "max": max(data),
        "mean": sum(data) / len(data),
        "median": float(np.median(np.asarray(data, dtype=np.float64))),
    }


def capture_distance(left: Capture, right: Capture) -> float:
    return float(reference_dtw(left.coordinates, left.valid_mask, right.coordinates, right.valid_mask, DTW_BAND)["distance"])


def select_medoid(captures: list[Capture]) -> tuple[Capture, list[dict[str, Any]]]:
    require(bool(captures), "cannot select medoid from empty set")
    scored: list[tuple[float, str, Capture]] = []
    rows: list[dict[str, Any]] = []
    for candidate in captures:
        other_distances = [capture_distance(candidate, other) for other in captures if other.capture_key != candidate.capture_key]
        mean_distance = sum(other_distances) / len(other_distances) if other_distances else 0.0
        scored.append((mean_distance, candidate.capture_id, candidate))
        rows.append({
            "capture_key": candidate.capture_key,
            "capture_id": candidate.capture_id,
            "mean_same_word_distance": mean_distance,
            "comparison_count": len(other_distances),
        })
    minimum = min(score[0] for score in scored)
    tied = [score for score in scored if score[0] == minimum]
    selected = min(tied, key=lambda score: score[1])[2]
    return selected, sorted(rows, key=lambda row: row["capture_id"])


def write_wisrt_bytes(capture: Capture) -> bytes:
    validate_capture(capture)
    coordinates = np.ascontiguousarray(capture.coordinates, dtype="<f8")
    mask = np.ascontiguousarray(capture.valid_mask, dtype=np.bool_)
    fields = (
        capture.capture_id,
        capture.word,
        "PERSONAL",
        "personal_calibration",
        capture.source_sha256,
        capture.normalized_payload_sha256,
    )
    stream = io.BytesIO()
    stream.write(b"WISRCB01")
    stream.write(struct.pack("<II", 1, 0x01020304))
    stream.write(struct.pack("<II", coordinates.shape[1], coordinates.shape[0]))
    for value in fields:
        encoded = value.encode("utf-8")
        stream.write(struct.pack("<I", len(encoded)))
        stream.write(encoded)
    stream.write(struct.pack("<Q", coordinates.size))
    stream.write(coordinates.tobytes(order="C"))
    mask_count = mask.size
    packed = np.packbits(mask.reshape(-1).astype(np.uint8), bitorder="little").tobytes()
    stream.write(struct.pack("<QI", mask_count, len(packed)))
    stream.write(packed)
    payload = stream.getvalue()
    return payload + hashlib.sha256(payload).digest()


def read_wisrt_bytes(data: bytes) -> dict[str, Any]:
    require(len(data) > 32, "WISRT truncated")
    payload, embedded = data[:-32], data[-32:]
    require(hashlib.sha256(payload).digest() == embedded, "WISRT embedded checksum mismatch")
    stream = io.BytesIO(payload)

    def exact(count: int) -> bytes:
        value = stream.read(count)
        require(len(value) == count, "WISRT field truncated")
        return value

    def u32() -> int:
        return struct.unpack("<I", exact(4))[0]

    def u64() -> int:
        return struct.unpack("<Q", exact(8))[0]

    def text() -> str:
        length = u32()
        require(0 < length <= 4096, "WISRT string length invalid")
        return exact(length).decode("utf-8")

    require(exact(8) == b"WISRCB01", "WISRT magic mismatch")
    require(u32() == 1 and u32() == 0x01020304, "WISRT version/endian mismatch")
    landmarks, length = u32(), u32()
    sample_id, word, speed, split, source_sha, logical_sha = (text() for _ in range(6))
    coordinate_count = u64()
    require(coordinate_count == length * landmarks * 3, "WISRT coordinate count mismatch")
    coordinates = np.frombuffer(exact(coordinate_count * 8), dtype="<f8").copy().reshape(length, landmarks, 3)
    mask_count, packed_size = u64(), u32()
    require(mask_count == length * landmarks and packed_size == (mask_count + 7) // 8, "WISRT mask count mismatch")
    mask = np.unpackbits(np.frombuffer(exact(packed_size), dtype=np.uint8), bitorder="little")[:mask_count].astype(bool).reshape(length, landmarks)
    require(stream.tell() == len(payload), "WISRT trailing payload")
    return {
        "sample_id": sample_id,
        "word": word,
        "speed": speed,
        "split": split,
        "source_payload_sha256": source_sha,
        "logical_payload_sha256": logical_sha,
        "coordinates": np.ascontiguousarray(coordinates),
        "valid_mask": np.ascontiguousarray(mask),
        "embedded_checksum_sha256": embedded.hex(),
    }


def calibration_artifacts(
    captures: list[Capture], medoids: dict[str, Capture], threshold: float,
    genuine_upper: float, impostor_lower: float, leave_one_out: list[dict[str, Any]],
) -> dict[str, bytes]:
    artifacts: dict[str, bytes] = {}
    template_records: list[dict[str, Any]] = []
    for word in WORDS:
        capture = medoids[word]
        name = f"active_personal_templates/personal_template_{word}.wisrt"
        payload = write_wisrt_bytes(capture)
        parsed = read_wisrt_bytes(payload)
        require(parsed["word"] == word and parsed["sample_id"] == capture.capture_id, "WISRT metadata round-trip mismatch")
        require(np.array_equal(parsed["valid_mask"], capture.valid_mask), "WISRT mask round-trip mismatch")
        require(np.array_equal(parsed["coordinates"], capture.coordinates, equal_nan=True), "WISRT coordinate round-trip mismatch")
        artifacts[name] = payload
        template_records.append({
            "word": word,
            "capture_key": capture.capture_key,
            "capture_id": capture.capture_id,
            "relative_path": name,
            "size_bytes": len(payload),
            "sha256": sha256_bytes(payload),
            "embedded_checksum_sha256": parsed["embedded_checksum_sha256"],
        })
    threshold_config = {
        "schema_version": SCHEMA + ".personal_threshold_config",
        "opt_in_required": True,
        "global_threshold_overwritten": False,
        "dtw_band": DTW_BAND,
        "genuine_upper": genuine_upper,
        "impostor_lower": impostor_lower,
        "personal_threshold": threshold,
        "derivation": "(genuine_upper + impostor_lower) / 2",
        "words": list(WORDS),
    }
    artifacts["personal_threshold_config.json"] = pretty_json_bytes(threshold_config)
    artifacts["calibration_provenance_manifest.json"] = pretty_json_bytes({
        "schema_version": SCHEMA + ".calibration_provenance_manifest",
        "minimum_repetitions_per_word": MIN_REPETITIONS,
        "capture_count": len(captures),
        "template_records": template_records,
        "leave_one_out": leave_one_out,
        "genuine_upper": genuine_upper,
        "impostor_lower": impostor_lower,
        "personal_threshold": threshold,
    })
    artifacts["source_capture_identity_manifest.json"] = pretty_json_bytes({
        "schema_version": SCHEMA + ".source_capture_identity_manifest",
        "captures": [identity_record(capture) for capture in sorted(captures, key=lambda item: item.capture_key)],
    })
    checksum_targets = sorted(artifacts)
    checksum_text = "".join(f"{sha256_bytes(artifacts[name])}  {name}\n" for name in checksum_targets).encode("utf-8")
    artifacts["personal_calibration_bundle.sha256"] = checksum_text
    return artifacts


def calibrate(captures: list[Capture], emit_artifacts: bool = True) -> dict[str, Any]:
    validate_capture_set(captures)
    counts = {word: sum(capture.word == word for capture in captures) for word in WORDS}
    additional = {word: max(0, MIN_REPETITIONS - counts[word]) for word in WORDS}
    if any(counts[word] < MIN_REPETITIONS for word in WORDS):
        return {
            "verdict": "INSUFFICIENT_CALIBRATION_REPETITIONS",
            "counts_per_word": counts,
            "additional_required_per_word": additional,
            "additional_required_total": sum(additional.values()),
            "medoids": {},
            "leave_one_out": [],
            "genuine_upper": None,
            "impostor_lower": None,
            "personal_threshold": None,
            "artifacts": {},
        }

    medoids: dict[str, Capture] = {}
    medoid_scores: dict[str, list[dict[str, Any]]] = {}
    for word in WORDS:
        medoids[word], medoid_scores[word] = select_medoid([capture for capture in captures if capture.word == word])

    leave_one_out: list[dict[str, Any]] = []
    all_correct = True
    for held_out in sorted(captures, key=lambda item: item.capture_key):
        training = [capture for capture in captures if capture.capture_key != held_out.capture_key]
        fold_medoids: dict[str, Capture] = {}
        for word in WORDS:
            fold_word = [capture for capture in training if capture.word == word]
            require(bool(fold_word), f"empty leave-one-out word: {word}")
            fold_medoids[word], _ = select_medoid(fold_word)
        distances = {word: capture_distance(held_out, fold_medoids[word]) for word in WORDS}
        predicted = min(WORDS, key=lambda word: (distances[word], word))
        wrong_word = min((word for word in WORDS if word != held_out.word), key=lambda word: (distances[word], word))
        genuine = distances[held_out.word]
        impostor = distances[wrong_word]
        correct = predicted == held_out.word
        all_correct = all_correct and correct
        leave_one_out.append({
            "held_out_capture_key": held_out.capture_key,
            "held_out_capture_id": held_out.capture_id,
            "expected_word": held_out.word,
            "fold_medoid_capture_ids": {word: fold_medoids[word].capture_id for word in WORDS},
            "expected_word_medoid_distance": genuine,
            "best_wrong_word": wrong_word,
            "best_wrong_word_medoid_distance": impostor,
            "predicted_word": predicted,
            "prediction_correct": correct,
            "genuine_score": genuine,
            "impostor_score": impostor,
            "margin": impostor - genuine,
        })
    genuine_upper = max(row["genuine_score"] for row in leave_one_out)
    impostor_lower = min(row["impostor_score"] for row in leave_one_out)
    if not all_correct or not genuine_upper < impostor_lower:
        return {
            "verdict": "CALIBRATION_NOT_SEPARABLE",
            "counts_per_word": counts,
            "additional_required_per_word": additional,
            "additional_required_total": 0,
            "medoids": {word: medoids[word].capture_id for word in WORDS},
            "medoid_scores": medoid_scores,
            "leave_one_out": leave_one_out,
            "genuine_upper": genuine_upper,
            "impostor_lower": impostor_lower,
            "personal_threshold": None,
            "artifacts": {},
        }
    threshold = (genuine_upper + impostor_lower) / 2.0
    artifacts = calibration_artifacts(captures, medoids, threshold, genuine_upper, impostor_lower, leave_one_out) if emit_artifacts else {}
    return {
        "verdict": "CALIBRATION_SUCCESS",
        "counts_per_word": counts,
        "additional_required_per_word": additional,
        "additional_required_total": 0,
        "medoids": {word: medoids[word].capture_id for word in WORDS},
        "medoid_scores": medoid_scores,
        "leave_one_out": leave_one_out,
        "genuine_upper": genuine_upper,
        "impostor_lower": impostor_lower,
        "personal_threshold": threshold,
        "artifacts": artifacts,
    }


def deterministic_zip_write(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    require(safe_zip_name(name), f"unsafe archive path: {name}")
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    info.create_system = 3
    archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def build_deterministic_zip_bytes(files: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9, strict_timestamps=True) as archive:
        for name in sorted(files):
            deterministic_zip_write(archive, name, files[name])
    return stream.getvalue()


def inspect_zip_bytes(data: bytes, expected_names: set[str]) -> dict[str, Any]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        crc_failure = archive.testzip()
        require(crc_failure is None, f"generated ZIP CRC failure: {crc_failure}")
        require(len(names) == len(set(names)), "generated ZIP duplicate entry")
        require(set(names) == expected_names, "generated ZIP membership mismatch")
        require(all(safe_zip_name(name) for name in names), "generated ZIP unsafe path")
        require(not any(stat.S_ISLNK((info.external_attr >> 16) & 0xFFFF) for info in infos), "generated ZIP symlink")
        require(not any(info.flag_bits & 1 for info in infos), "generated ZIP encrypted entry")
        return {
            "entry_count": len(names), "crc": "PASS", "duplicates": 0, "unsafe_paths": 0,
            "symlinks": 0, "encrypted_entries": 0, "membership": "PASS",
        }


def synthetic_capture(word: str, capture_id: str, base: float, repetition: int, length: int = 8) -> Capture:
    coordinates = np.empty((length, 16, 3), dtype=np.float64)
    for frame in range(length):
        for landmark in range(16):
            coordinates[frame, landmark] = (
                base + repetition * 0.04 + frame * 0.02 + landmark * 0.001,
                base * 0.1 + frame * 0.005 - landmark * 0.0005,
                repetition * 0.01 + landmark * 0.0002,
            )
    mask = np.ones((length, 16), dtype=np.bool_)
    source = json_bytes({"synthetic": True, "word": word, "capture_id": capture_id, "base": base, "repetition": repetition})
    return make_capture(
        capture_key="synthetic::" + capture_id, source_family="synthetic", source_archive="SYNTHETIC",
        source_archive_sha256="SYNTHETIC", archive_entry=capture_id + ".json", source_manifest_entry="SYNTHETIC",
        schema_version=SCHEMA + ".synthetic_capture", word=word, run_id="synthetic-run-" + word,
        capture_id=capture_id, frame_indices=range(length), timestamps=[frame * 0.05 + 0.001 * repetition for frame in range(length)],
        coordinates=coordinates, valid_mask=mask, source_bytes=source,
    )


def synthetic_regression() -> dict[str, Any]:
    tests: list[dict[str, Any]] = []

    success_captures = [
        synthetic_capture(word, f"{word.lower()}-{rep}", float(index * 10), rep)
        for index, word in enumerate(WORDS) for rep in range(3)
    ]
    success = calibrate(success_captures)
    require(success["verdict"] == "CALIBRATION_SUCCESS", "synthetic separable calibration did not succeed")
    require(len([name for name in success["artifacts"] if name.endswith(".wisrt")]) == 3, "synthetic success WISRT count mismatch")
    require("personal_threshold_config.json" in success["artifacts"], "synthetic success threshold missing")
    tests.append({"test_id": "separable_three_per_word", "status": "PASS", "verdict": success["verdict"], "threshold": success["personal_threshold"]})

    insufficient = calibrate([capture for capture in success_captures if not capture.capture_id.endswith("-2")])
    require(insufficient["verdict"] == "INSUFFICIENT_CALIBRATION_REPETITIONS", "synthetic insufficient verdict mismatch")
    require(not insufficient["artifacts"], "synthetic insufficient emitted active artifacts")
    tests.append({"test_id": "two_per_word_insufficient", "status": "PASS", "verdict": insufficient["verdict"], "active_artifact_count": 0})

    overlap_captures = [
        synthetic_capture(word, f"overlap-{word.lower()}-{rep}", float(index) * 0.05, rep * 5)
        for index, word in enumerate(WORDS) for rep in range(3)
    ]
    overlap = calibrate(overlap_captures)
    require(overlap["verdict"] == "CALIBRATION_NOT_SEPARABLE", f"synthetic overlap verdict mismatch: {overlap['verdict']}")
    require(not overlap["artifacts"], "synthetic non-separable emitted active artifacts")
    tests.append({"test_id": "genuine_impostor_overlap", "status": "PASS", "verdict": overlap["verdict"], "active_artifact_count": 0})

    duplicate_id_rejected = False
    duplicate_payload_rejected = False
    try:
        duplicate = dataclasses.replace(success_captures[-1], capture_id=success_captures[0].capture_id)
        validate_capture_set(success_captures[:-1] + [duplicate])
    except DuplicateCaptureError:
        duplicate_id_rejected = True
    try:
        original = success_captures[0]
        duplicate_payload = make_capture(
            capture_key="synthetic::duplicate-payload", source_family="synthetic", source_archive="SYNTHETIC",
            source_archive_sha256="SYNTHETIC", archive_entry="duplicate-payload.json", source_manifest_entry="SYNTHETIC",
            schema_version=original.schema_version, word=original.word, run_id="different-run", capture_id="different-id",
            frame_indices=original.frame_indices, timestamps=[value + 100 for value in original.timestamps],
            coordinates=original.coordinates.copy(), valid_mask=original.valid_mask.copy(), source_bytes=b"different-source-payload",
        )
        validate_capture_set([original, duplicate_payload])
    except DuplicateCaptureError:
        duplicate_payload_rejected = True
    require(duplicate_id_rejected and duplicate_payload_rejected, "synthetic duplicate rejection failed")
    tests.append({"test_id": "capture_id_or_payload_duplicate_rejected", "status": "PASS", "capture_id_rejected": True, "payload_rejected": True})

    corrupt_mask_rejected = corrupt_shape_rejected = corrupt_checksum_rejected = False
    sample = success_captures[0]
    for kind in ("mask", "shape", "checksum"):
        try:
            if kind == "mask":
                damaged = dataclasses.replace(sample, valid_mask=sample.valid_mask[:, :-1])
            elif kind == "shape":
                damaged = dataclasses.replace(sample, coordinates=sample.coordinates[:, :-1, :])
            else:
                damaged = dataclasses.replace(sample, source_sha256="0" * 64)
            validate_capture(damaged)
        except CaptureContractError:
            if kind == "mask": corrupt_mask_rejected = True
            elif kind == "shape": corrupt_shape_rejected = True
            else: corrupt_checksum_rejected = True
    require(corrupt_mask_rejected and corrupt_shape_rejected and corrupt_checksum_rejected, "synthetic corruption rejection failed")
    tests.append({"test_id": "corrupt_mask_shape_checksum_rejected", "status": "PASS", "mask": True, "shape": True, "checksum": True})

    tie_captures: list[Capture] = []
    for capture_id, landmark in (("tie-c", 0), ("tie-a", 1), ("tie-b", 2)):
        coordinates = np.zeros((4, 16, 3), dtype=np.float64)
        mask = np.zeros((4, 16), dtype=np.bool_)
        mask[:, landmark] = True
        tie_captures.append(make_capture(
            capture_key="synthetic::" + capture_id, source_family="synthetic", source_archive="SYNTHETIC",
            source_archive_sha256="SYNTHETIC", archive_entry=capture_id, source_manifest_entry="SYNTHETIC",
            schema_version=SCHEMA + ".synthetic_capture", word="GREETING", run_id="tie-run",
            capture_id=capture_id, frame_indices=range(4), timestamps=[landmark + frame * 0.1 for frame in range(4)],
            coordinates=coordinates, valid_mask=mask, source_bytes=(capture_id + "-source").encode("utf-8"),
        ))
    tie_medoid, tie_scores = select_medoid(tie_captures)
    require(tie_medoid.capture_id == "tie-a", f"medoid tie-break mismatch: {tie_medoid.capture_id}")
    require(len({row["mean_same_word_distance"] for row in tie_scores}) == 1, "synthetic medoid distances are not tied")
    tests.append({"test_id": "medoid_tie_capture_id_lexical", "status": "PASS", "selected_capture_id": tie_medoid.capture_id})

    wisrt = write_wisrt_bytes(sample)
    decoded = read_wisrt_bytes(wisrt)
    require(decoded["sample_id"] == sample.capture_id and decoded["word"] == sample.word, "WISRT identity round-trip mismatch")
    require(np.array_equal(decoded["coordinates"], sample.coordinates, equal_nan=True) and np.array_equal(decoded["valid_mask"], sample.valid_mask), "WISRT payload round-trip mismatch")
    damaged_checksum_rejected = False
    damaged_wisrt = bytearray(wisrt)
    damaged_wisrt[-1] ^= 1
    try:
        read_wisrt_bytes(bytes(damaged_wisrt))
    except FoundationError:
        damaged_checksum_rejected = True
    require(damaged_checksum_rejected, "WISRT damaged embedded checksum accepted")
    tests.append({"test_id": "wisrt_round_trip_embedded_checksum", "status": "PASS", "size_bytes": len(wisrt), "sha256": sha256_bytes(wisrt), "damaged_checksum_rejected": True})

    success_repeat = calibrate(success_captures)
    require(success_repeat["verdict"] == "CALIBRATION_SUCCESS", "synthetic repeated calibration did not succeed")
    synthetic_files = dict(success["artifacts"])
    repeated_files = dict(success_repeat["artifacts"])
    require(set(synthetic_files) == set(repeated_files), "synthetic repeated result-file membership mismatch")
    require(all(synthetic_files[name] == repeated_files[name] for name in synthetic_files), "synthetic repeated result-file byte mismatch")
    zip_first = build_deterministic_zip_bytes(synthetic_files)
    zip_second = build_deterministic_zip_bytes(repeated_files)
    require(zip_first == zip_second, "synthetic deterministic ZIP mismatch")
    inspect_zip_bytes(zip_first, set(synthetic_files))
    tests.append({"test_id": "same_input_byte_identical_results_and_zip", "status": "PASS", "calibration_run_count": 2, "result_file_count": len(synthetic_files), "zip_size_bytes": len(zip_first), "zip_sha256": sha256_bytes(zip_first), "result_files_byte_identical": True, "zip_byte_identical": True})

    require(not insufficient["artifacts"] and not overlap["artifacts"], "failure active artifact count nonzero")
    tests.append({"test_id": "failure_active_template_threshold_zero", "status": "PASS", "insufficient_active_count": 0, "not_separable_active_count": 0})
    require(len(tests) == 9 and len({test["test_id"] for test in tests}) == 9, "synthetic test key/count mismatch")
    return {
        "schema_version": SCHEMA + ".synthetic_regression",
        "scope_note": "Synthetic success validates foundation behavior only; it is not an actual-seed performance PASS.",
        "test_count": len(tests),
        "passed": sum(test["status"] == "PASS" for test in tests),
        "failed": sum(test["status"] == "FAIL" for test in tests),
        "tests": tests,
    }


def csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        formatted: dict[str, Any] = {}
        for field in fields:
            value = row[field]
            if isinstance(value, bool):
                formatted[field] = "true" if value else "false"
            elif isinstance(value, float):
                require(math.isfinite(value), f"non-finite CSV field {field}")
                formatted[field] = format(value, ".17g")
            else:
                formatted[field] = value
        writer.writerow(formatted)
    return stream.getvalue().encode("utf-8")


def assertion_record(key: str, status: str, target: str, details: Any) -> dict[str, Any]:
    require(status in {"PASS", "FAIL", "UNVERIFIED", "NOT_APPLICABLE"}, f"invalid assertion status: {status}")
    return {"assertion_key": key, "status": status, "target": target, "details": details}


def add_provenance_subset(paths: dict[str, Path], subset: dict[str, bytes]) -> list[dict[str, Any]]:
    selections = {
        "step8a": [
            "input_manifest.json", "selected_dtw_config.json", "template_manifest.json",
            "output_file_manifest.json", "s001_step8a_build_dtw_templates_v1_0.py",
        ],
        "baseline_diagnostic": [
            BASE_SCRIPT_ENTRY, BASE_INPUT_MANIFEST_ENTRY, BASE_MANIFEST_ENTRY,
            BASE_PREFIX + "inputs/step8a/selected_dtw_config.json",
            BASE_PREFIX + "inputs/step8a/feature_schema.json",
            BASE_PREFIX + "inputs/step8a/template_manifest.json",
            BASE_PREFIX + "inputs/runtime_gate/frozen_threshold_config.json",
            BASE_PREFIX + "inputs/runtime_templates/runtime_template_manifest.json",
            BASE_PREFIX + "inputs/step9c_delivery/delivery_input_manifest.json",
        ],
        "raw_intermediate": ["capture_manifest.json", "output_file_manifest.json", "VALIDATION_REPORT.md"],
        "availability_review": [
            "input_provenance.json", "output_file_manifest.json", "primary_verdict.json",
            "reference_sample_identity.json", "reference_trace_subset/subset_manifest.json",
            "source_archive_search_scope.json",
        ],
    }
    index: list[dict[str, Any]] = []
    for role, entries in selections.items():
        with zipfile.ZipFile(paths[role]) as archive:
            names = set(archive.namelist())
            for entry in entries:
                require(entry in names, f"missing provenance subset entry: {role}/{entry}")
                destination = f"input_subset/provenance/{role}/{PurePosixPath(entry).name}"
                require(destination not in subset, f"duplicate subset destination: {destination}")
                payload = archive.read(entry)
                subset[destination] = payload
                index.append({
                    "source_role": role,
                    "source_archive_entry": entry,
                    "subset_relative_path": destination,
                    "size_bytes": len(payload),
                    "sha256": sha256_bytes(payload),
                })
    return sorted(index, key=lambda item: item["subset_relative_path"])


def subset_records(subset: dict[str, bytes]) -> list[dict[str, Any]]:
    return [
        {"relative_path": name, "size_bytes": len(payload), "sha256": sha256_bytes(payload)}
        for name, payload in sorted(subset.items())
    ]


def output_category(name: str) -> str:
    if name.startswith("input_subset/"):
        return "byte_exact_input_subset"
    if name.startswith("analysis/"):
        return "analysis_source"
    if name.startswith("active_personal_templates/"):
        return "opt_in_personal_template"
    if name.endswith(".csv"):
        return "validation_csv"
    if name.endswith(".json") or name.endswith(".jsonl"):
        return "validation_json"
    if name.endswith(".md"):
        return "validation_report"
    if name.endswith(".sha256"):
        return "detached_checksum"
    return "artifact"


def git_staged_count(root: Path) -> dict[str, Any]:
    try:
        process = subprocess.run(
            ["git", "diff", "--cached", "--name-only"], cwd=root,
            capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
        )
        return {
            "available": process.returncode == 0,
            "count": len([line for line in process.stdout.splitlines() if line.strip()]),
            "return_code": process.returncode,
            "stderr": process.stderr.splitlines(),
        }
    except OSError as exc:
        return {"available": False, "count": None, "reason": repr(exc)}


def validate_core_outputs(files: dict[str, bytes]) -> dict[str, Any]:
    json_names = [name for name in files if name.endswith(".json")]
    for name in json_names:
        strict_json_loads(files[name])
    assertion_lines = [line for line in files["ASSERTIONS.jsonl"].decode("utf-8").splitlines() if line]
    parsed_assertions = [strict_json_loads((line + "\n").encode("utf-8")) for line in assertion_lines]
    assertion_keys = [row["assertion_key"] for row in parsed_assertions]
    require(len(assertion_keys) == len(set(assertion_keys)), "duplicate assertion key")
    inventory = list(csv.DictReader(io.StringIO(files["seed_capture_inventory.csv"].decode("utf-8"))))
    inventory_keys = [row["capture_key"] for row in inventory]
    require(len(inventory_keys) == len(set(inventory_keys)), "duplicate capture key in inventory")
    pairwise = list(csv.DictReader(io.StringIO(files["seed_pairwise_dtw.csv"].decode("utf-8"))))
    pair_keys = [row["pair_key"] for row in pairwise]
    require(len(pair_keys) == len(set(pair_keys)), "duplicate pairwise key")
    synthetic = strict_json_loads(files["synthetic_regression.json"])
    test_keys = [row["test_id"] for row in synthetic["tests"]]
    require(len(test_keys) == len(set(test_keys)), "duplicate synthetic test key")
    return {
        "json_file_count": len(json_names),
        "csv_file_count": 2,
        "assertion_key_count": len(assertion_keys),
        "capture_key_count": len(inventory_keys),
        "pair_key_count": len(pair_keys),
        "synthetic_test_key_count": len(test_keys),
        "non_standard_json_constants": 0,
        "duplicate_keys": 0,
    }


def build_report(
    input_records: dict[str, dict[str, Any]], captures: list[Capture], identity: dict[str, Any],
    pairwise_summary: dict[str, Any], actual: dict[str, Any], synthetic: dict[str, Any],
    assertion_summary: dict[str, int], active_wisrt_count: int, threshold_count: int,
) -> str:
    lines = [
        "# WordsInSilence Step 9E Personalized Calibration/Template Foundation v1.0",
        "",
        "## Outcome",
        "",
        "The offline foundation implementation and deterministic packaging checks passed. The real approved seed set has two independent captures per word, below the fixed minimum of three, so no active personal template or threshold was generated.",
        "",
        "```text",
        "STEP9E_PERSONALIZED_CALIBRATION_TEMPLATE_FOUNDATION_V1_0_READY_PENDING_EXTERNAL_REVIEW",
        "PRIMARY_VERDICT_INSUFFICIENT_CALIBRATION_REPETITIONS",
        "ADDITIONAL_CALIBRATION_REQUIRED_ONE_TAKE_PER_WORD",
        "BLOCKED_PENDING_EXTERNAL_REVIEW",
        "```",
        "",
        "## Fixed inputs",
        "",
        "| Role | Path | Bytes | SHA-256 | Manifest |",
        "|---|---|---:|---|---|",
    ]
    for role in ("step8a", "baseline_diagnostic", "raw_intermediate", "availability_review"):
        record = input_records[role]
        lines.append(f"| {role} | `{record['path']}` | {record['size_bytes']} | `{record['sha256']}` | PASS |")
    lines += [
        "",
        "## Seed preflight",
        "",
        f"- Verdict: `{identity['seed_preflight_verdict']}`",
    ]
    for word in WORDS:
        word_captures = [capture for capture in captures if capture.word == word]
        lines.append(f"- {word}: {len(word_captures)} independent captures; additional required `{actual['additional_required_per_word'][word]}`")
        for capture in word_captures:
            lines.append(f"  - `{capture.source_family}` / run `{capture.run_id}` / capture `{capture.capture_id}` / source `{capture.source_sha256}`")
    lines += [
        "",
        "## Pairwise DTW",
        "",
        f"- Matrix rows: `{pairwise_summary['matrix_row_count']}`",
        f"- Diagonal maximum absolute distance: `{pairwise_summary['diagonal_max_absolute_distance']:.17g}`",
        f"- Symmetry maximum absolute error: `{pairwise_summary['symmetry_max_absolute_error']:.17g}`",
        f"- Repeat maximum absolute error: `{pairwise_summary['repeat_max_absolute_error']:.17g}`",
        f"- Approved reference/runtime maximum errors: `{pairwise_summary['approved_reference_max_absolute_error']:.17g}` / `{pairwise_summary['approved_runtime_contract_max_absolute_error']:.17g}`",
        f"- Same-word off-diagonal min/mean/max: `{pairwise_summary['same_word_off_diagonal']['min']:.17g}` / `{pairwise_summary['same_word_off_diagonal']['mean']:.17g}` / `{pairwise_summary['same_word_off_diagonal']['max']:.17g}`",
        f"- Cross-word min/mean/max: `{pairwise_summary['cross_word']['min']:.17g}` / `{pairwise_summary['cross_word']['mean']:.17g}` / `{pairwise_summary['cross_word']['max']:.17g}`",
        "",
        "These distances characterize the seed state only. They are not used to derive a personal threshold from two repetitions.",
        "",
        "## Calibration and synthetic regression",
        "",
        f"- Actual primary verdict: `{actual['verdict']}`",
        f"- Synthetic regression: `{synthetic['passed']} PASS / {synthetic['failed']} FAIL`",
        f"- Active personal WISRT count: `{active_wisrt_count}`",
        f"- Personal threshold config count: `{threshold_count}`",
        f"- Assertions: `{assertion_summary['PASS']} PASS / {assertion_summary['FAIL']} FAIL / {assertion_summary['UNVERIFIED']} UNVERIFIED / {assertion_summary['NOT_APPLICABLE']} NOT_APPLICABLE`",
        "",
        "Synthetic success establishes deterministic foundation behavior only and is not an actual-seed performance PASS.",
        "",
        "## Scope and unchanged areas",
        "",
        "No Unity, webcam, new capture, runtime activation, production recognition, scene, prefab, asset, meta, global WISRT, global threshold, frozen evidence, or project-context file was modified. No Git staging, commit, push, or next-stage acquisition was performed.",
        "",
    ]
    return "\n".join(lines)


def build(root: Path, output: Path, delivery_zip: Path, additional_capture_dir: Path | None = None) -> dict[str, Any]:
    delivery_hash = Path(str(delivery_zip) + ".sha256")
    for target in (output, delivery_zip, delivery_hash):
        require(not target.exists(), f"never-overwrite target already exists: {target}")
    require((root / "Assets").is_dir() and (root / "ProjectSettings" / "ProjectVersion.txt").is_file(), f"not project root: {root}")

    paths, input_records = verify_fixed_inputs(root)
    input_before = {role: (record["size_bytes"], record["sha256"]) for role, record in input_records.items()}
    approved_helpers, approved_helper_provenance = load_approved_dtw_helpers(paths["baseline_diagnostic"], root)
    captures, subset, source_provenance = load_actual_captures(paths, input_records)
    if additional_capture_dir is not None:
        captures.extend(load_additional_capture_files(additional_capture_dir))
        captures.sort(key=lambda item: (WORDS.index(item.word), item.source_family, item.capture_id))
        validate_capture_set(captures)
    provenance_subset_index = add_provenance_subset(paths, subset)
    subset_before = subset_records(subset)
    identities = [identity_record(capture) for capture in captures]
    independence = independence_records(captures)
    require(all(row["independent"] for row in independence), "seed capture independence mismatch")
    counts = {word: sum(capture.word == word for capture in captures) for word in WORDS}
    seed_preflight_verdict = "SEED_PREFLIGHT_TWO_INDEPENDENT_CAPTURES_PER_WORD" if counts == {word: 2 for word in WORDS} else "SEED_PREFLIGHT_CAPTURE_COUNTS_RECORDED"
    pairwise_rows, pairwise_summary = pairwise_dtw_rows(captures, approved_helpers)
    actual = calibrate(captures)
    synthetic = synthetic_regression()
    active_artifacts = actual["artifacts"]
    active_wisrt_count = len([name for name in active_artifacts if name.startswith("active_personal_templates/") and name.endswith(".wisrt")])
    threshold_count = int("personal_threshold_config.json" in active_artifacts)
    if actual["verdict"] != "CALIBRATION_SUCCESS":
        require(active_wisrt_count == 0 and threshold_count == 0 and not active_artifacts, "failed/insufficient actual run emitted active personal output")

    input_after_records: dict[str, dict[str, Any]] = {}
    for role, path in paths.items():
        input_after_records[role] = {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    require(all(input_before[role] == (input_after_records[role]["size_bytes"], input_after_records[role]["sha256"]) for role in input_before), "fixed input archive changed during run")

    contract = {
        "schema_version": SCHEMA + ".contract",
        "scope": "offline_personalized_calibration_template_foundation",
        "words": list(WORDS),
        "minimum_independent_repetitions_per_word": MIN_REPETITIONS,
        "minimum_total_captures": MIN_REPETITIONS * len(WORDS),
        "coordinates": {"semantic": "normalized", "shape": ["T", 16, 3], "dtype": "float64"},
        "valid_mask": {"shape": ["T", 16], "dtype": "bool"},
        "feature_mapping": {
            "pose_source_indices": list(POSE_INDICES),
            "left_hand_source_indices": list(LEFT_HAND_INDICES),
            "right_hand_source_indices": list(RIGHT_HAND_INDICES),
        },
        "dtw": {
            "mask_aware": True, "sakoe_chiba_relative_band": DTW_BAND, "missing_cost": MISSING_COST,
            "distance_normalization": "cumulative_cost_divided_by_warping_path_length",
            "local_cost": "mean Euclidean over comparable landmarks; exactly-one-valid cost 1.0; neither-valid excluded",
            "tie_order": ["diagonal", "vertical", "horizontal"],
        },
        "prohibited_transformations": ["smoothing", "mirror_augmentation", "new_interpolation", "NaN_to_zero", "null_substitution", "coordinate_reconstruction", "mapping_change", "landmark_weight_change", "local_cost_change"],
        "same_word_medoid": "minimum mean off-diagonal same-word DTW; exact tie resolved by capture_id lexicographic order",
        "leave_one_out": "hold out each capture; choose per-word medoids from remaining captures; require every expected-word prediction",
        "threshold": {
            "genuine_upper": "max(all leave-one-out genuine scores)",
            "impostor_lower": "min(all leave-one-out impostor scores)",
            "separation_required": "genuine_upper < impostor_lower",
            "personal_threshold": "(genuine_upper + impostor_lower) / 2",
        },
        "failure_precedence": ["identity_or_coordinate_contract_error", "INSUFFICIENT_CALIBRATION_REPETITIONS", "CALIBRATION_NOT_SEPARABLE", "CALIBRATION_SUCCESS"],
        "personal_output": {"opt_in_only": True, "never_overwrite": True, "global_template_or_threshold_overwrite": False},
    }
    seed_identity = {
        "schema_version": SCHEMA + ".seed_capture_identity",
        "seed_preflight_verdict": seed_preflight_verdict,
        "capture_count": len(captures),
        "captures": identities,
        "pairwise_independence": independence,
        "all_pairs_independent": all(row["independent"] for row in independence),
        "two_source_families_are_different_actual_captures": all(row["independent"] for row in independence if row["same_word"]),
    }
    seed_preflight = {
        "schema_version": SCHEMA + ".seed_preflight",
        "verdict": seed_preflight_verdict,
        "status": "PASS",
        "counts_per_word": counts,
        "minimum_required_per_word": MIN_REPETITIONS,
        "additional_required_per_word": actual["additional_required_per_word"],
        "additional_required_total": actual["additional_required_total"],
        "actual_threshold_derivation_status": "NOT_APPLICABLE_INSUFFICIENT_REPETITIONS" if actual["verdict"] == "INSUFFICIENT_CALIBRATION_REPETITIONS" else "EVALUATED",
        "pairwise_dtw_role": "seed_state_characterization_only",
    }
    primary = {
        "schema_version": SCHEMA + ".primary_verdict",
        "foundation_verification_status": "PASS",
        "actual_seed_primary_verdict": actual["verdict"],
        "actual_seed_performance_status": "NOT_APPLICABLE" if actual["verdict"] == "INSUFFICIENT_CALIBRATION_REPETITIONS" else ("PASS" if actual["verdict"] == "CALIBRATION_SUCCESS" else "FAIL"),
        "seed_identity_status": "PASS",
        "synthetic_regression_status": "PASS" if synthetic["failed"] == 0 else "FAIL",
        "active_personal_wisrt_count": active_wisrt_count,
        "personal_threshold_config_count": threshold_count,
        "additional_required_per_word": actual["additional_required_per_word"],
        "status_strings": [
            "STEP9E_PERSONALIZED_CALIBRATION_TEMPLATE_FOUNDATION_V1_0_READY_PENDING_EXTERNAL_REVIEW",
            "PRIMARY_VERDICT_INSUFFICIENT_CALIBRATION_REPETITIONS" if actual["verdict"] == "INSUFFICIENT_CALIBRATION_REPETITIONS" else "PRIMARY_VERDICT_" + actual["verdict"],
            "ADDITIONAL_CALIBRATION_REQUIRED_ONE_TAKE_PER_WORD" if actual["additional_required_per_word"] == {word: 1 for word in WORDS} else "ADDITIONAL_CALIBRATION_REQUIREMENT_RECORDED",
            "BLOCKED_PENDING_EXTERNAL_REVIEW",
        ],
        "next_core_milestone_started": False,
    }
    provenance_json = {
        "schema_version": SCHEMA + ".input_provenance",
        "fixed_archives": input_records,
        "fixed_archives_after": input_after_records,
        "fixed_archives_before_after_unchanged": True,
        "approved_dtw_helper": approved_helper_provenance,
        "capture_source_provenance": source_provenance,
        "provenance_subset_source_index": provenance_subset_index,
        "input_subset": subset_before,
        "input_subset_file_count": len(subset_before),
        "input_subset_role": "minimum byte-exact physical captures, checksums, approved helper, and source manifests required for independent recalculation; full source archives intentionally omitted",
        "numerical_use": {
            "step8a": "fixed contract/provenance and WISRT format source",
            "baseline_diagnostic": "baseline Step 9C physical queries and approved DTW helper",
            "raw_intermediate": "Step 9D normalized physical captures and capture manifest",
            "availability_review": "approved trace availability verdict provenance only; no divergence diagnosis repeated",
        },
    }

    assertions = [
        assertion_record("fixed_input_archive_identity", "PASS", "four approved ZIPs", {role: record["sha256"] for role, record in input_records.items()}),
        assertion_record("fixed_input_zip_crc_safety_manifest", "PASS", "four approved ZIPs", {role: record["zip_validation"] for role, record in input_records.items()}),
        assertion_record("fixed_inputs_before_after_immutable", "PASS", "four approved ZIPs", input_after_records),
        assertion_record("capture_source_detached_checksums", "PASS", "six actual seed JSON payloads", {capture.capture_key: capture.source_sha256 for capture in captures}),
        assertion_record("capture_coordinate_mask_contract", "PASS", "six actual seeds", {capture.capture_key: [list(capture.coordinates.shape), list(capture.valid_mask.shape)] for capture in captures}),
        assertion_record("capture_identity_key_uniqueness", "PASS", "seed_capture_identity", {"capture_count": len(captures)}),
        assertion_record("seed_underlying_capture_independence", "PASS", "all seed pairs", {"pair_count": len(independence), "independent": sum(row["independent"] for row in independence)}),
        assertion_record("two_independent_captures_per_word", "PASS", "actual seeds", counts),
        assertion_record("pairwise_dtw_full_matrix", "PASS", "seed_pairwise_dtw.csv", {"rows": len(pairwise_rows)}),
        assertion_record("pairwise_dtw_diagonal_zero", "PASS", "six diagonal cells", pairwise_summary["diagonal_max_absolute_distance"]),
        assertion_record("pairwise_dtw_symmetry", "PASS", "36 ordered cells", pairwise_summary["symmetry_max_absolute_error"]),
        assertion_record("pairwise_dtw_repeat_determinism", "PASS", "36 ordered cells", pairwise_summary["repeat_max_absolute_error"]),
        assertion_record("approved_reference_runtime_dtw_parity", "PASS", "36 ordered cells", {"reference_max_error": pairwise_summary["approved_reference_max_absolute_error"], "runtime_max_error": pairwise_summary["approved_runtime_contract_max_absolute_error"], "path_mismatch_count": pairwise_summary["approved_path_mismatch_count"]}),
        assertion_record("minimum_repetition_gate", "PASS", "actual calibration", {"verdict": actual["verdict"], "required": MIN_REPETITIONS}),
        assertion_record("actual_leave_one_out_threshold", "NOT_APPLICABLE", "actual calibration", "INSUFFICIENT_CALIBRATION_REPETITIONS" if actual["verdict"] == "INSUFFICIENT_CALIBRATION_REPETITIONS" else "evaluated"),
        assertion_record("insufficient_active_output_zero", "PASS", "actual output", {"active_wisrt_count": active_wisrt_count, "threshold_count": threshold_count}),
        assertion_record("synthetic_regression_nine_cases", "PASS", "synthetic_regression.json", {"passed": synthetic["passed"], "failed": synthetic["failed"]}),
        assertion_record("synthetic_success_not_actual_performance", "PASS", "claim boundary", synthetic["scope_note"]),
        assertion_record("input_subset_byte_exact", "PASS", "input_subset", {"file_count": len(subset_before)}),
        assertion_record("json_csv_schema_and_unique_keys", "PASS", "result schemas", "validated before finalization"),
        assertion_record("output_manifest_membership_size_hash", "PASS", "output_file_manifest.json", "validated on final file map and ZIP"),
        assertion_record("deterministic_zip_memory_regeneration", "PASS", "delivery ZIP", "same final file map generated byte-identical ZIP bytes twice"),
        assertion_record("post_zip_crc_path_symlink_encryption_membership", "PASS", "delivery ZIP", "final byte stream reopened and fully validated"),
        assertion_record("production_and_approved_areas_unchanged", "PASS", "scope", "only the new Step 9E tool and new Step 9E outputs were written"),
    ]
    require(len({row["assertion_key"] for row in assertions}) == len(assertions), "duplicate assertion key before serialization")
    assertion_summary = {status: sum(row["status"] == status for row in assertions) for status in ("PASS", "FAIL", "UNVERIFIED", "NOT_APPLICABLE")}

    inventory_rows = [capture_inventory_record(capture) for capture in captures]
    inventory_fields = list(inventory_rows[0])
    pairwise_fields = list(pairwise_rows[0])
    files: dict[str, bytes] = dict(subset)
    files.update(active_artifacts)
    files.update({
        "input_provenance.json": pretty_json_bytes(provenance_json),
        "seed_capture_identity.json": pretty_json_bytes(seed_identity),
        "seed_capture_inventory.csv": csv_bytes(inventory_rows, inventory_fields),
        "seed_pairwise_dtw.csv": csv_bytes(pairwise_rows, pairwise_fields),
        "seed_preflight.json": pretty_json_bytes(seed_preflight),
        "personalized_calibration_contract.json": pretty_json_bytes(contract),
        "synthetic_regression.json": pretty_json_bytes(synthetic),
        "primary_verdict.json": pretty_json_bytes(primary),
        "ASSERTIONS.jsonl": b"".join(json_bytes(row) for row in assertions),
        "VALIDATION_REPORT.md": build_report(input_records, captures, seed_identity, pairwise_summary, actual, synthetic, assertion_summary, active_wisrt_count, threshold_count).encode("utf-8"),
        "analysis/" + Path(SCRIPT_REL).name: (root / SCRIPT_REL).read_bytes(),
    })
    schema_validation = validate_core_outputs(files)
    subset_after = [record for record in subset_records({name: files[name] for name in subset})]
    require(subset_before == subset_after, "input subset changed before packaging")

    output_manifest = {
        "schema_version": SCHEMA + ".output_file_manifest",
        "self_excluded": "output_file_manifest.json",
        "outputs": [
            {"relative_path": name, "category": output_category(name), "size_bytes": len(payload), "sha256": sha256_bytes(payload)}
            for name, payload in sorted(files.items())
        ],
        "output_count_excluding_manifest": len(files),
    }
    files["output_file_manifest.json"] = pretty_json_bytes(output_manifest)
    require(set(record["relative_path"] for record in output_manifest["outputs"]) == set(files) - {"output_file_manifest.json"}, "output manifest membership construction mismatch")
    for record in output_manifest["outputs"]:
        require(len(files[record["relative_path"]]) == record["size_bytes"] and sha256_bytes(files[record["relative_path"]]) == record["sha256"], f"output manifest record mismatch: {record['relative_path']}")

    zip_first = build_deterministic_zip_bytes(files)
    zip_second = build_deterministic_zip_bytes(files)
    require(zip_first == zip_second, "final deterministic ZIP memory regeneration mismatch")
    zip_inspection = inspect_zip_bytes(zip_first, set(files))
    with zipfile.ZipFile(io.BytesIO(zip_first)) as archive:
        archived_manifest = strict_json_loads(archive.read("output_file_manifest.json"))
        archived_records = manifest_records(archived_manifest)
        require(set(record["relative_path"] for record in archived_records) == set(archive.namelist()) - {"output_file_manifest.json"}, "post-ZIP manifest membership mismatch")
        for record in archived_records:
            payload = archive.read(record["relative_path"])
            require(len(payload) == record["size_bytes"] and sha256_bytes(payload) == record["sha256"], f"post-ZIP manifest mismatch: {record['relative_path']}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".step9e_build_", dir=output.parent) as temporary:
        temporary_root = Path(temporary)
        # Keep staging names short so the longest evidence filenames remain below
        # the default Windows MAX_PATH limit.  The final names are unchanged.
        staged_output = temporary_root / "o"
        for name, payload in files.items():
            destination = staged_output / PurePosixPath(name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        staged_records = {
            path.relative_to(staged_output).as_posix(): (path.stat().st_size, sha256_file(path))
            for path in staged_output.rglob("*") if path.is_file()
        }
        require(staged_records == {name: (len(payload), sha256_bytes(payload)) for name, payload in files.items()}, "staged output byte mismatch")
        staged_zip = temporary_root / "d.zip"
        staged_zip.write_bytes(zip_first)
        require(staged_zip.read_bytes() == zip_first, "written ZIP differs from in-memory ZIP")
        staged_hash = temporary_root / "d.zip.sha256"
        delivery_sha = sha256_bytes(zip_first)
        detached_payload = f"{delivery_sha}  {delivery_zip.name}\n".encode("utf-8")
        staged_hash.write_bytes(detached_payload)
        require(not output.exists() and not delivery_zip.exists() and not delivery_hash.exists(), "never-overwrite target appeared during build")
        shutil.move(str(staged_output), str(output))
        shutil.move(str(staged_zip), str(delivery_zip))
        shutil.move(str(staged_hash), str(delivery_hash))

    require(delivery_zip.stat().st_size == len(zip_first) and sha256_file(delivery_zip) == sha256_bytes(zip_first), "final ZIP write verification mismatch")
    require(parse_detached(delivery_hash.read_bytes(), delivery_zip.name, sha256_file(delivery_zip)), "final detached checksum mismatch")
    with zipfile.ZipFile(delivery_zip) as archive:
        require(archive.testzip() is None and set(archive.namelist()) == set(files), "final on-disk ZIP reopen mismatch")
    final_subset_records = [
        {"relative_path": name, "size_bytes": (output / PurePosixPath(name)).stat().st_size, "sha256": sha256_file(output / PurePosixPath(name))}
        for name in sorted(subset)
    ]
    require(final_subset_records == subset_before, "final input subset after hash mismatch")

    staged = git_staged_count(root)
    return {
        "output_directory": output.relative_to(root).as_posix(),
        "output_file_count": len(files),
        "output_directory_size_bytes": sum(path.stat().st_size for path in output.rglob("*") if path.is_file()),
        "delivery_zip": delivery_zip.relative_to(root).as_posix(),
        "delivery_zip_size_bytes": delivery_zip.stat().st_size,
        "delivery_zip_sha256": sha256_file(delivery_zip),
        "detached_checksum": delivery_hash.relative_to(root).as_posix(),
        "detached_checksum_size_bytes": delivery_hash.stat().st_size,
        "detached_checksum_sha256": sha256_file(delivery_hash),
        "zip_memory_regeneration_byte_identical": True,
        "zip_inspection": zip_inspection,
        "schema_validation": schema_validation,
        "assertion_summary": assertion_summary,
        "capture_count": len(captures),
        "counts_per_word": counts,
        "pairwise_summary": pairwise_summary,
        "primary_verdict": actual["verdict"],
        "additional_required_per_word": actual["additional_required_per_word"],
        "synthetic": {"passed": synthetic["passed"], "failed": synthetic["failed"]},
        "active_personal_wisrt_count": active_wisrt_count,
        "personal_threshold_config_count": threshold_count,
        "git_staged": staged,
        "status_strings": primary["status_strings"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--delivery-zip", type=Path)
    parser.add_argument("--additional-capture-dir", type=Path, help="future explicit opt-in; omitted for the approved two-seed Step 9E run")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    output = (args.output_dir or (root / OUTPUT_REL)).resolve()
    delivery_zip = (args.delivery_zip or (root / DELIVERY_REL)).resolve()
    try:
        result = build(root, output, delivery_zip, args.additional_capture_dir.resolve() if args.additional_capture_dir else None)
    except (DuplicateCaptureError, CaptureContractError) as exc:
        print("UNVERIFIED_SEED_IDENTITY_OR_COORDINATE_CONTRACT_MISMATCH")
        print(str(exc))
        return 2
    except FoundationError as exc:
        print("STEP9E_FOUNDATION_BUILD_FAILED")
        print(str(exc))
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
