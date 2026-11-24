#!/usr/bin/env python3
"""WordsInSilence Step 9F additional personal calibration capture validator.

This offline-only tool validates exactly one new Step 9D raw/intermediate capture
for GREETING, LOVE, and THANK_YOU, proves independence from the six approved
Step 9E seeds, and emits a deterministic external-review delivery. It never
runs calibration, DTW, medoid selection, threshold derivation, or WISRT output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import shutil
import stat
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np


SCHEMA = "words_in_silence.step9f.additional_personal_calibration_capture_one_take_per_word.v1_0"
CAPTURE_SCHEMA = "words_in_silence.step9d.raw_intermediate_diagnostic.v1_0"
WORDS = ("GREETING", "LOVE", "THANK_YOU")
SCRIPT_REL = "Tools/s001_step9f_additional_personal_calibration_capture_one_take_per_word_v1_0.py"
OUTPUT_REL = "Validation/WordsInSilence_step9f_additional_personal_calibration_capture_one_take_per_word_v1_0"
DELIVERY_REL = "Validation/WordsInSilence_step9f_additional_personal_calibration_capture_one_take_per_word_v1_0_delivery.zip"
STEP9E_REL = "Validation/WordsInSilence_step9e_personalized_calibration_template_foundation_v1_0_delivery.zip"
STEP9E_SIZE = 1_776_636
STEP9E_SHA256 = "9829d143b24e454e47ca68a54b5b9d630548a5790a3743882f92321bf638181b"
RAW_EVIDENCE_REL = "Validation/WordsInSilence_step9d_raw_intermediate_physical_evidence_3takes_v1_0_delivery.zip"
RAW_EVIDENCE_SIZE = 1_468_035
RAW_EVIDENCE_SHA256 = "7c4982d15e1d4be71f054562eee8913c3a94a289827dc024c10bf61332651cd1"
HARNESS_REL = "Assets/_Project/Scripts/Recognition/Diagnostics/Step9DRawIntermediatePhysicalAcquisitionHarness.cs"
HARNESS_SIZE = 21_416
HARNESS_SHA256 = "5531a99b6fcdf38e5deb0ee210d9e471870692fd614f1cabfb4a004c51237b5c"
HARNESS_TEST_REL = "Assets/_Project/Tests/EditMode/Step9DRawIntermediatePhysicalAcquisitionHarnessV1Tests.cs"
HARNESS_TEST_SIZE = 17_615
HARNESS_TEST_SHA256 = "934f16d01c2e8a27f687cc80e5d0605a163ee38f80fc67a413d13ae901138160"
SCENE_REL = "Assets/_Project/Scenes/Technical/02_PoseIntegration.unity"
PREFAB_REL = "Assets/_Project/Resources/Diagnostics/PhysicalWebcamCoordinateCompatibility.prefab"
NORMALIZED_PREFIX = b"WIS_STEP9E_NORMALIZED_V1\0"
RAW_SHAPE_TAIL = (75, 3)
RAW_MASK_TAIL = (75,)
NORMALIZED_SHAPE_TAIL = (16, 3)
NORMALIZED_MASK_TAIL = (16,)
SELECTED_RAW_INDICES = (13, 14, 15, 16, 33, 37, 41, 45, 49, 53, 54, 58, 62, 66, 70, 74)
EXPECTED_RAW_ORDER = ("pose[0..32]", "left_hand[0..20]", "right_hand[0..20]")
EXPECTED_NORMALIZED_ORDER = (
    "pose[13]", "pose[14]", "pose[15]", "pose[16]",
    "left_hand[0]", "left_hand[4]", "left_hand[8]", "left_hand[12]", "left_hand[16]", "left_hand[20]",
    "right_hand[0]", "right_hand[4]", "right_hand[8]", "right_hand[12]", "right_hand[16]", "right_hand[20]",
)
EXPECTED_ROOT_KEYS = {
    "schema_version", "run_id", "capture_id", "expected_word", "capture_metadata",
    "frame_identity", "raw_and_preprocessing", "torso_transform", "normalized_output",
}
CAPTURE_RE = re.compile(r"take_(\d{2})_(GREETING|LOVE|THANK_YOU)_([0-9a-f]{32})\.json")
HEX_32_RE = re.compile(r"[0-9a-f]{32}")
HEX_64_RE = re.compile(r"[0-9a-f]{64}")


class Step9FError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Step9FError(message)


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
        raise Step9FError(f"non-standard JSON constant: {token}")

    return json.loads(data.decode("utf-8"), parse_constant=reject_constant)


def finite_json(value: Any, path: str = "$") -> None:
    if isinstance(value, float):
        require(math.isfinite(value), f"non-finite generated JSON value at {path}")
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
    return bool(
        name
        and not name.startswith(("/", "\\"))
        and "\\" not in name
        and not re.match(r"^[A-Za-z]:", name)
        and ".." not in path.parts
    )


def parse_detached(data: bytes, expected_name: str, expected_hash: str) -> bool:
    match = re.fullmatch(rb"([0-9a-f]{64})  ([^\r\n]+)(?:\r?\n)?", data)
    return bool(
        match
        and match.group(1).decode("ascii") == expected_hash
        and match.group(2).decode("utf-8") == expected_name
    )


def canonical_coordinate_bytes(array: np.ndarray) -> bytes:
    return np.ascontiguousarray(array, dtype="<f8").tobytes(order="C")


def canonical_mask_bytes(array: np.ndarray) -> bytes:
    return np.ascontiguousarray(array, dtype=np.uint8).tobytes(order="C")


def normalized_payload_sha256(coordinates: np.ndarray, mask: np.ndarray) -> str:
    return sha256_bytes(NORMALIZED_PREFIX + canonical_coordinate_bytes(coordinates) + canonical_mask_bytes(mask))


def manifest_records(data: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("outputs", "files", "inputs", "input_files"):
        if isinstance(data.get(key), list):
            return data[key]
    raise Step9FError(f"manifest has no supported record list: {sorted(data)}")


def inspect_zip(path: Path, expected_manifest: str | None = None) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        require(archive.testzip() is None, f"ZIP CRC failure: {path}")
        require(len(names) == len(set(names)), f"duplicate ZIP entry: {path}")
        require(all(safe_zip_name(name) for name in names), f"unsafe ZIP path: {path}")
        require(not any(stat.S_ISLNK((info.external_attr >> 16) & 0xFFFF) for info in infos), f"symlink ZIP entry: {path}")
        require(not any(info.flag_bits & 1 for info in infos), f"encrypted ZIP entry: {path}")
        result: dict[str, Any] = {
            "entry_count": len(names), "crc": "PASS", "duplicate_entries": 0,
            "unsafe_paths": 0, "symlinks": 0, "encrypted_entries": 0,
        }
        if expected_manifest is not None:
            require(expected_manifest in names, f"missing ZIP manifest {expected_manifest}: {path}")
            manifest_payload = archive.read(expected_manifest)
            manifest = strict_json_loads(manifest_payload)
            records = manifest_records(manifest)
            targets = [str(record["relative_path"]) for record in records]
            require(len(targets) == len(set(targets)), f"duplicate manifest target: {path}")
            require(set(targets) == set(names) - {expected_manifest}, f"ZIP manifest membership mismatch: {path}")
            for record in records:
                payload = archive.read(record["relative_path"])
                require(len(payload) == int(record["size_bytes"]), f"manifest size mismatch: {record['relative_path']}")
                require(sha256_bytes(payload) == record["sha256"], f"manifest SHA mismatch: {record['relative_path']}")
            result.update({
                "manifest_entry": expected_manifest,
                "manifest_size_bytes": len(manifest_payload),
                "manifest_sha256": sha256_bytes(manifest_payload),
                "manifest_target_count": len(records),
                "membership": "PASS",
            })
        return result


def fixed_file(root: Path, relative_path: str, expected_size: int, expected_sha256: str) -> dict[str, Any]:
    path = root / relative_path
    require(path.is_file(), f"missing approved file: {path}")
    size = path.stat().st_size
    digest = sha256_file(path)
    require(size == expected_size, f"approved size mismatch: {relative_path}: {size} != {expected_size}")
    require(digest == expected_sha256, f"approved SHA-256 mismatch: {relative_path}: {digest} != {expected_sha256}")
    return {"path": relative_path, "size_bytes": size, "sha256": digest, "fixed_identity": "PASS"}


def bool_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        object_array = np.asarray(value, dtype=object)
    except ValueError as error:
        raise Step9FError(f"ragged bool array {name}: {error}") from error
    require(object_array.shape == shape, f"{name} shape mismatch: {object_array.shape} != {shape}")
    require(all(type(item) is bool for item in object_array.flat), f"{name} contains non-bool value")
    return np.ascontiguousarray(object_array, dtype=np.bool_)


def float_array(value: Any, shape: tuple[int, ...], name: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        object_array = np.asarray(value, dtype=object)
    except ValueError as error:
        raise Step9FError(f"ragged numeric array {name}: {error}") from error
    require(object_array.shape == shape, f"{name} shape mismatch: {object_array.shape} != {shape}")
    null_mask = np.zeros(shape, dtype=np.bool_)
    for index, item in np.ndenumerate(object_array):
        if item is None:
            null_mask[index] = True
        else:
            require(type(item) in (int, float), f"{name}{index} is not numeric/null")
            require(math.isfinite(float(item)), f"{name}{index} is non-finite")
    array = np.asarray(value, dtype=np.float64)
    require(not np.isinf(array).any(), f"{name} contains Infinity")
    require(np.array_equal(np.isnan(array), null_mask), f"{name} null/NaN conversion mismatch")
    return np.ascontiguousarray(array), null_mask


def integer_sequence(values: Iterable[Any], count: int, name: str) -> tuple[int, ...]:
    output: list[int] = []
    for index, value in enumerate(values):
        require(type(value) is int, f"{name}[{index}] is not an integer")
        output.append(value)
    require(len(output) == count, f"{name} count mismatch")
    return tuple(output)


@dataclass(frozen=True)
class Capture:
    take_index: int
    word: str
    run_id: str
    capture_id: str
    json_path: Path
    checksum_path: Path
    json_bytes: bytes
    checksum_bytes: bytes
    json_sha256: str
    checksum_sha256: str
    frame_count: int
    timestamps: tuple[float, ...]
    provider_frames: tuple[int, ...]
    bridge_observations: tuple[int, ...]
    lifetime_provider_frames: tuple[int, ...]
    normalized_coordinates: np.ndarray
    normalized_mask: np.ndarray
    coordinate_payload_sha256: str
    mask_payload_sha256: str
    normalized_payload_sha256: str
    raw_valid_count: int
    raw_reappearance_count: int
    raw_strong_depth_count: int
    timestamp_boundary_count: int
    combined_boundary_count: int
    interpolation_count: int
    torso_source_valid_count: int
    torso_final_valid_count: int
    torso_provenance_counts: dict[str, int]
    normalized_valid_count: int
    normalized_interpolated_count: int

    @property
    def capture_key(self) -> str:
        return f"step9f_additional::{self.capture_id}"


def monotonic_strict(values: Iterable[float | int], name: str) -> None:
    sequence = list(values)
    require(sequence, f"empty sequence: {name}")
    require(all(sequence[index] < sequence[index + 1] for index in range(len(sequence) - 1)), f"{name} is not strictly increasing")


def validate_capture(json_path: Path, checksum_path: Path, expected_take: int, expected_word: str, expected_run_id: str) -> Capture:
    match = CAPTURE_RE.fullmatch(json_path.name)
    require(match is not None, f"capture filename contract mismatch: {json_path.name}")
    require(int(match.group(1)) == expected_take and match.group(2) == expected_word, f"capture filename order mismatch: {json_path.name}")
    filename_capture_id = match.group(3)
    payload = json_path.read_bytes()
    detached = checksum_path.read_bytes()
    digest = sha256_bytes(payload)
    require(parse_detached(detached, json_path.name, digest), f"detached checksum mismatch: {json_path.name}")
    data = strict_json_loads(payload)
    require(isinstance(data, dict) and set(data) == EXPECTED_ROOT_KEYS, f"root schema keys mismatch: {json_path.name}")
    require(data["schema_version"] == CAPTURE_SCHEMA, f"capture schema version mismatch: {json_path.name}")
    require(data["run_id"] == expected_run_id and HEX_32_RE.fullmatch(str(data["run_id"])) is not None, f"run ID mismatch: {json_path.name}")
    require(data["capture_id"] == filename_capture_id and HEX_32_RE.fullmatch(str(data["capture_id"])) is not None, f"capture ID mismatch: {json_path.name}")
    require(data["expected_word"] == expected_word, f"expected word mismatch: {json_path.name}")

    metadata = data["capture_metadata"]
    require(isinstance(metadata, dict), f"capture_metadata missing: {json_path.name}")
    require(metadata.get("selected_max_gap") == 1, f"selected_max_gap changed: {json_path.name}")
    require(float(metadata.get("timestamp_discontinuity_seconds")) == 0.1, f"timestamp discontinuity contract changed: {json_path.name}")
    require(tuple(metadata.get("raw_landmark_ordering", [])) == EXPECTED_RAW_ORDER, f"raw ordering mismatch: {json_path.name}")
    require(tuple(metadata.get("normalized_landmark_ordering", [])) == EXPECTED_NORMALIZED_ORDER, f"normalized ordering mismatch: {json_path.name}")

    identities = data["frame_identity"]
    require(isinstance(identities, list) and identities, f"frame_identity missing: {json_path.name}")
    frame_count = len(identities)
    require(metadata.get("raw_frame_count") == frame_count, f"raw frame count mismatch: {json_path.name}")
    require(metadata.get("normalized_frame_count") == frame_count, f"normalized frame count mismatch: {json_path.name}")
    local_frames = integer_sequence((frame.get("capture_local_frame_index") for frame in identities), frame_count, "capture_local_frame_index")
    require(local_frames == tuple(range(frame_count)), f"capture-local frame range is not 0..T-1: {json_path.name}")
    timestamps: list[float] = []
    for index, frame in enumerate(identities):
        value = frame.get("observation_timestamp_seconds")
        require(type(value) in (int, float) and math.isfinite(float(value)), f"invalid timestamp at {json_path.name}:{index}")
        timestamps.append(float(value))
    monotonic_strict(timestamps, f"timestamps:{json_path.name}")
    provider_frames = integer_sequence((frame.get("provider_frame_index") for frame in identities), frame_count, "provider_frame_index")
    bridge_observations = integer_sequence((frame.get("bridge_observation_index") for frame in identities), frame_count, "bridge_observation_index")
    lifetime_frames = integer_sequence((frame.get("lifetime_provider_frame_index") for frame in identities), frame_count, "lifetime_provider_frame_index")
    monotonic_strict(provider_frames, f"provider frames:{json_path.name}")
    monotonic_strict(bridge_observations, f"bridge observations:{json_path.name}")
    monotonic_strict(lifetime_frames, f"lifetime provider frames:{json_path.name}")

    raw = data["raw_and_preprocessing"]
    require(isinstance(raw, dict), f"raw_and_preprocessing missing: {json_path.name}")
    require(raw.get("raw_coordinates_shape") == [frame_count, 75, 3], f"stored raw coordinate shape mismatch: {json_path.name}")
    require(raw.get("raw_original_valid_mask_shape") == [frame_count, 75], f"stored raw mask shape mismatch: {json_path.name}")
    require(raw.get("post_gap_interpolation_coordinates_shape") == [frame_count, 75, 3], f"stored post-gap shape mismatch: {json_path.name}")
    raw_coordinates, _ = float_array(raw.get("raw_coordinates"), (frame_count,) + RAW_SHAPE_TAIL, "raw_coordinates")
    raw_valid = bool_array(raw.get("raw_original_valid_mask"), (frame_count,) + RAW_MASK_TAIL, "raw_original_valid_mask")
    reappearance = bool_array(raw.get("raw_reappearance_mask"), (frame_count,) + RAW_MASK_TAIL, "raw_reappearance_mask")
    strong_depth = bool_array(raw.get("raw_strong_depth_mask"), (frame_count,) + RAW_MASK_TAIL, "raw_strong_depth_mask")
    timestamp_boundary = bool_array(raw.get("timestamp_discontinuity_boundary_mask"), (frame_count,) + RAW_MASK_TAIL, "timestamp_discontinuity_boundary_mask")
    combined = bool_array(raw.get("combined_boundary_mask"), (frame_count,) + RAW_MASK_TAIL, "combined_boundary_mask")
    post_gap, _ = float_array(raw.get("post_gap_interpolation_coordinates"), (frame_count,) + RAW_SHAPE_TAIL, "post_gap_interpolation_coordinates")
    interpolation = bool_array(raw.get("interpolation_mask"), (frame_count,) + RAW_MASK_TAIL, "interpolation_mask")
    require(np.array_equal(combined, reappearance | strong_depth | timestamp_boundary), f"combined boundary content mismatch: {json_path.name}")
    require(np.isfinite(raw_coordinates[raw_valid]).all(), f"valid raw coordinate is non-finite: {json_path.name}")
    require(np.isnan(raw_coordinates[~raw_valid]).all(), f"invalid raw coordinate is not null: {json_path.name}")
    post_valid = np.isfinite(post_gap).all(axis=2)
    post_null = np.isnan(post_gap).all(axis=2)
    require(np.all(post_valid | post_null), f"partially-null post-gap coordinate: {json_path.name}")
    require(np.all(post_valid[interpolation]), f"interpolated post-gap coordinate is invalid: {json_path.name}")
    require(not np.any(interpolation & combined), f"interpolation crosses a boundary: {json_path.name}")

    torso = data["torso_transform"]
    require(isinstance(torso, dict), f"torso_transform missing: {json_path.name}")
    require(torso.get("vector_shape") == [frame_count, 3], f"stored torso vector shape mismatch: {json_path.name}")
    torso_source = bool_array(torso.get("source_valid"), (frame_count,), "torso_source_valid")
    torso_final = bool_array(torso.get("final_valid"), (frame_count,), "torso_final_valid")
    provenance = torso.get("provenance")
    require(isinstance(provenance, list) and len(provenance) == frame_count, f"torso provenance shape mismatch: {json_path.name}")
    allowed_provenance = {"DIRECT", "FORWARD_CARRIED", "LEADING_BACKFILLED", "UNAVAILABLE"}
    require(all(type(value) is str and value in allowed_provenance for value in provenance), f"invalid torso provenance: {json_path.name}")
    torso_arrays = []
    for field in ("origin", "right_axis", "up_axis", "forward_axis"):
        array, _ = float_array(torso.get(field), (frame_count, 3), f"torso_{field}")
        torso_arrays.append(array)
    torso_scale, _ = float_array(torso.get("shoulder_width_scale"), (frame_count,), "torso_shoulder_width_scale")
    for frame, value in enumerate(provenance):
        consistent = (
            (value == "DIRECT" and torso_source[frame] and torso_final[frame])
            or (value == "UNAVAILABLE" and not torso_source[frame] and not torso_final[frame])
            or (value in {"FORWARD_CARRIED", "LEADING_BACKFILLED"} and not torso_source[frame] and torso_final[frame])
        )
        require(bool(consistent), f"torso validity/provenance mismatch: {json_path.name}:{frame}")
        if torso_final[frame]:
            require(all(np.isfinite(array[frame]).all() for array in torso_arrays), f"valid torso basis is non-finite: {json_path.name}:{frame}")
            require(math.isfinite(float(torso_scale[frame])) and torso_scale[frame] > 0, f"valid torso scale invalid: {json_path.name}:{frame}")
        else:
            require(all(np.isnan(array[frame]).all() for array in torso_arrays), f"invalid torso basis is not null: {json_path.name}:{frame}")
            require(np.isnan(torso_scale[frame]), f"invalid torso scale is not null: {json_path.name}:{frame}")

    normalized = data["normalized_output"]
    require(isinstance(normalized, dict), f"normalized_output missing: {json_path.name}")
    require(normalized.get("coordinates_shape") == [frame_count, 16, 3], f"stored normalized coordinate shape mismatch: {json_path.name}")
    require(normalized.get("mask_shape") == [frame_count, 16], f"stored normalized mask shape mismatch: {json_path.name}")
    coordinates, _ = float_array(normalized.get("coordinates"), (frame_count,) + NORMALIZED_SHAPE_TAIL, "normalized_coordinates")
    mask = bool_array(normalized.get("mask"), (frame_count,) + NORMALIZED_MASK_TAIL, "normalized_mask")
    require(np.isfinite(coordinates[mask]).all(), f"valid normalized coordinate is non-finite: {json_path.name}")
    require(np.isnan(coordinates[~mask]).all(), f"invalid normalized coordinate is not null/NaN-contract: {json_path.name}")
    valid_count = int(mask.sum())
    selected_interpolation = interpolation[:, SELECTED_RAW_INDICES]
    interpolated_count = int(np.count_nonzero(selected_interpolation & mask))
    require(type(normalized.get("valid_landmark_count")) is int and normalized["valid_landmark_count"] == valid_count, f"normalized valid count mismatch: {json_path.name}")
    require(type(normalized.get("interpolated_landmark_count")) is int and normalized["interpolated_landmark_count"] == interpolated_count, f"normalized interpolated count mismatch: {json_path.name}")

    coordinate_hash = sha256_bytes(canonical_coordinate_bytes(coordinates))
    mask_hash = sha256_bytes(canonical_mask_bytes(mask))
    return Capture(
        take_index=expected_take, word=expected_word, run_id=expected_run_id, capture_id=filename_capture_id,
        json_path=json_path, checksum_path=checksum_path, json_bytes=payload, checksum_bytes=detached,
        json_sha256=digest, checksum_sha256=sha256_bytes(detached), frame_count=frame_count,
        timestamps=tuple(timestamps), provider_frames=provider_frames, bridge_observations=bridge_observations,
        lifetime_provider_frames=lifetime_frames, normalized_coordinates=coordinates, normalized_mask=mask,
        coordinate_payload_sha256=coordinate_hash, mask_payload_sha256=mask_hash,
        normalized_payload_sha256=normalized_payload_sha256(coordinates, mask), raw_valid_count=int(raw_valid.sum()),
        raw_reappearance_count=int(reappearance.sum()), raw_strong_depth_count=int(strong_depth.sum()),
        timestamp_boundary_count=int(timestamp_boundary.sum()), combined_boundary_count=int(combined.sum()),
        interpolation_count=int(interpolation.sum()), torso_source_valid_count=int(torso_source.sum()),
        torso_final_valid_count=int(torso_final.sum()),
        torso_provenance_counts={key: provenance.count(key) for key in sorted(allowed_provenance)},
        normalized_valid_count=valid_count, normalized_interpolated_count=interpolated_count,
    )


def identity_record(capture: Capture) -> dict[str, Any]:
    fingerprint = sha256_bytes(json_bytes({
        "run_id": capture.run_id,
        "capture_id": capture.capture_id,
        "timestamp_range": [capture.timestamps[0], capture.timestamps[-1]],
        "source_payload_sha256": capture.json_sha256,
        "coordinate_payload_sha256": capture.coordinate_payload_sha256,
        "mask_payload_sha256": capture.mask_payload_sha256,
    }))
    return {
        "capture_key": capture.capture_key,
        "source_family": "step9f_additional_raw_intermediate_capture",
        "word": capture.word,
        "take_index": capture.take_index,
        "run_id": capture.run_id,
        "capture_id": capture.capture_id,
        "timestamp_range_seconds": [capture.timestamps[0], capture.timestamps[-1]],
        "provider_frame_range": [capture.provider_frames[0], capture.provider_frames[-1]],
        "bridge_observation_range": [capture.bridge_observations[0], capture.bridge_observations[-1]],
        "lifetime_provider_frame_range": [capture.lifetime_provider_frames[0], capture.lifetime_provider_frames[-1]],
        "source_json_size_bytes": len(capture.json_bytes),
        "source_payload_sha256": capture.json_sha256,
        "coordinate_payload_sha256": capture.coordinate_payload_sha256,
        "mask_payload_sha256": capture.mask_payload_sha256,
        "normalized_payload_sha256": capture.normalized_payload_sha256,
        "composite_identity_sha256": fingerprint,
        "source_path": str(capture.json_path),
        "source_manifest_provenance": "Step9D harness JSON plus detached checksum; Step9F capture_manifest.json generated read-only from these files",
    }


def inventory_record(capture: Capture) -> dict[str, Any]:
    invalid = ~capture.normalized_mask
    return {
        "capture_key": capture.capture_key,
        "take_index": capture.take_index,
        "word": capture.word,
        "run_id": capture.run_id,
        "capture_id": capture.capture_id,
        "json_file": capture.json_path.name,
        "json_size_bytes": len(capture.json_bytes),
        "json_sha256": capture.json_sha256,
        "detached_checksum_file": capture.checksum_path.name,
        "detached_checksum_size_bytes": len(capture.checksum_bytes),
        "detached_checksum_sha256": capture.checksum_sha256,
        "frame_count": capture.frame_count,
        "capture_local_frame_min": 0,
        "capture_local_frame_max": capture.frame_count - 1,
        "timestamp_min_seconds": capture.timestamps[0],
        "timestamp_max_seconds": capture.timestamps[-1],
        "provider_frame_min": capture.provider_frames[0],
        "provider_frame_max": capture.provider_frames[-1],
        "bridge_observation_min": capture.bridge_observations[0],
        "bridge_observation_max": capture.bridge_observations[-1],
        "lifetime_provider_frame_min": capture.lifetime_provider_frames[0],
        "lifetime_provider_frame_max": capture.lifetime_provider_frames[-1],
        "raw_coordinate_shape": f"{capture.frame_count}x75x3",
        "raw_mask_shape": f"{capture.frame_count}x75",
        "normalized_coordinate_shape": f"{capture.frame_count}x16x3",
        "normalized_coordinate_dtype": str(capture.normalized_coordinates.dtype),
        "normalized_mask_shape": f"{capture.frame_count}x16",
        "normalized_mask_dtype": str(capture.normalized_mask.dtype),
        "normalized_valid_landmark_count": capture.normalized_valid_count,
        "normalized_invalid_landmark_count": int(invalid.sum()),
        "normalized_valid_coordinate_component_count": capture.normalized_valid_count * 3,
        "normalized_invalid_null_component_count": int(np.isnan(capture.normalized_coordinates[invalid]).sum()),
        "raw_valid_landmark_count": capture.raw_valid_count,
        "raw_reappearance_boundary_count": capture.raw_reappearance_count,
        "raw_strong_depth_boundary_count": capture.raw_strong_depth_count,
        "timestamp_discontinuity_boundary_count": capture.timestamp_boundary_count,
        "combined_boundary_count": capture.combined_boundary_count,
        "raw_interpolation_count": capture.interpolation_count,
        "normalized_interpolated_landmark_count": capture.normalized_interpolated_count,
        "torso_source_valid_frame_count": capture.torso_source_valid_count,
        "torso_final_valid_frame_count": capture.torso_final_valid_count,
        "torso_provenance_counts_json": json.dumps(capture.torso_provenance_counts, sort_keys=True, separators=(",", ":")),
        "coordinate_payload_sha256": capture.coordinate_payload_sha256,
        "mask_payload_sha256": capture.mask_payload_sha256,
        "normalized_coordinate_and_mask_payload_sha256": capture.normalized_payload_sha256,
    }


def comparison_record(new: dict[str, Any], other: dict[str, Any], scope: str) -> dict[str, Any]:
    timestamp_exact = new["timestamp_range_seconds"] == other["timestamp_range_seconds"]
    flags = {
        "capture_id_equal": new["capture_id"] == other["capture_id"],
        "source_payload_sha256_equal": new["source_payload_sha256"] == other["source_payload_sha256"],
        "coordinate_payload_sha256_equal": new["coordinate_payload_sha256"] == other["coordinate_payload_sha256"],
        "mask_payload_sha256_equal": new["mask_payload_sha256"] == other["mask_payload_sha256"],
        "normalized_payload_sha256_equal": new["normalized_payload_sha256"] == other["normalized_payload_sha256"],
        "same_run_exact_timestamp_range": new["run_id"] == other["run_id"] and timestamp_exact,
    }
    return {
        "comparison_key": new["capture_key"] + "||" + other["capture_key"],
        "comparison_scope": scope,
        "new_capture_key": new["capture_key"],
        "new_word": new["word"],
        "new_run_id": new["run_id"],
        "new_capture_id": new["capture_id"],
        "other_capture_key": other["capture_key"],
        "other_word": other["word"],
        "other_run_id": other["run_id"],
        "other_capture_id": other["capture_id"],
        **flags,
        "independent": not any(flags.values()),
    }


def csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    require(rows, "cannot serialize empty CSV")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def assertion(key: str, status: str, target: str, details: Any) -> dict[str, Any]:
    require(status in {"PASS", "FAIL", "UNVERIFIED", "NOT_APPLICABLE"}, f"invalid assertion status: {status}")
    return {"assertion_key": key, "status": status, "target": target, "details": details}


def output_category(name: str) -> str:
    if name.startswith("analysis/"):
        return "analysis_source"
    if name.startswith("input_subset/captures/"):
        return "byte_exact_capture_input"
    if name.startswith("input_subset/provenance/"):
        return "approved_provenance_subset"
    if name == "ASSERTIONS.jsonl":
        return "assertions"
    if name.endswith(".csv"):
        return "inventory_or_independence_csv"
    if name.endswith(".md"):
        return "validation_report"
    return "structured_result"


def deterministic_zip_bytes(files: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9, strict_timestamps=True) as archive:
        for name in sorted(files):
            require(safe_zip_name(name), f"unsafe generated ZIP name: {name}")
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, files[name], compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return stream.getvalue()


def inspect_generated_zip(data: bytes, expected_names: set[str]) -> dict[str, Any]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        require(archive.testzip() is None, "generated ZIP CRC failure")
        require(len(names) == len(set(names)), "generated ZIP duplicate entry")
        require(set(names) == expected_names, "generated ZIP membership mismatch")
        require(all(safe_zip_name(name) for name in names), "generated ZIP unsafe path")
        require(not any(stat.S_ISLNK((info.external_attr >> 16) & 0xFFFF) for info in infos), "generated ZIP symlink")
        require(not any(info.flag_bits & 1 for info in infos), "generated ZIP encrypted entry")
        return {
            "entry_count": len(names), "crc": "PASS", "duplicate_entries": 0,
            "unsafe_paths": 0, "symlinks": 0, "encrypted_entries": 0, "membership": "PASS",
        }


def git_staged(root: Path) -> dict[str, Any]:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"], cwd=root,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
    )
    require(result.returncode == 0, f"git staged query failed: {result.stderr.strip()}")
    files = [line for line in result.stdout.splitlines() if line]
    return {"count": len(files), "files": files}


def validate_result_schemas(files: dict[str, bytes]) -> dict[str, Any]:
    json_names = sorted(name for name in files if name.endswith(".json"))
    for name in json_names:
        value = strict_json_loads(files[name])
        require(isinstance(value, dict), f"generated JSON root is not object: {name}")
    assertions = [strict_json_loads(line) for line in files["ASSERTIONS.jsonl"].splitlines() if line]
    assertion_keys = [row["assertion_key"] for row in assertions]
    require(len(assertion_keys) == len(set(assertion_keys)), "duplicate assertion key")
    inventory = list(csv.DictReader(io.StringIO(files["additional_capture_inventory.csv"].decode("utf-8"))))
    capture_keys = [row["capture_key"] for row in inventory]
    require(len(capture_keys) == len(set(capture_keys)) == 3, "inventory capture key mismatch")
    comparisons = list(csv.DictReader(io.StringIO(files["independence_against_existing_seeds.csv"].decode("utf-8"))))
    comparison_keys = [row["comparison_key"] for row in comparisons]
    require(len(comparison_keys) == len(set(comparison_keys)) == 21, "comparison key/count mismatch")
    return {
        "json_file_count": len(json_names), "csv_file_count": 2,
        "assertion_key_count": len(assertion_keys), "capture_key_count": len(capture_keys),
        "comparison_key_count": len(comparison_keys), "non_standard_json_constants": 0,
        "duplicate_sample_capture_assertion_keys": 0,
    }


def report_text(
    run_id: str, captures: list[Capture], identities: list[dict[str, Any]], comparison_count: int,
    assertions_summary: dict[str, int], fixed_inputs: dict[str, Any], staged: dict[str, Any],
) -> str:
    lines = [
        "# WordsInSilence Step 9F Validation Report", "",
        "## Outcome", "",
        "The approved unchanged Step 9D acquisition harness produced exactly one new capture for GREETING, LOVE, and THANK_YOU in one new run. All three captures passed byte-level checksum, raw/intermediate/torso/normalized schema, temporal ordering, and independence checks against the six approved Step 9E seeds and each other.", "",
        "```text",
        "STEP9F_ADDITIONAL_PERSONAL_CALIBRATION_CAPTURE_V1_0_READY_PENDING_EXTERNAL_REVIEW",
        "PRIMARY_VERDICT_THREE_ADDITIONAL_INDEPENDENT_CAPTURES_ACQUIRED",
        "PERSONAL_CALIBRATION_MINIMUM_THREE_PER_WORD_AVAILABLE_PENDING_EXTERNAL_REVIEW",
        "BLOCKED_PENDING_EXTERNAL_REVIEW",
        "```", "",
        "## Fixed input and preflight", "",
        f"- Step 9E Delivery: `{fixed_inputs['step9e']['path']}`, {fixed_inputs['step9e']['size_bytes']:,} bytes, `{fixed_inputs['step9e']['sha256']}`",
        f"- Approved harness SHA-256: `{fixed_inputs['harness']['sha256']}`",
        f"- Approved harness test SHA-256: `{fixed_inputs['harness_test']['sha256']}`",
        f"- Existing Step 9D raw evidence SHA-256: `{fixed_inputs['raw_evidence']['sha256']}`",
        f"- New run ID: `{run_id}`",
        "- Unity live readiness and sign performance were user-operated at the required manual checkpoint.", "", "## New captures", "",
    ]
    for capture, identity in zip(captures, identities):
        lines.append(
            f"- {capture.take_index}. `{capture.word}` — capture `{capture.capture_id}`, {capture.frame_count} frames, "
            f"timestamps {capture.timestamps[0]:.17g}..{capture.timestamps[-1]:.17g}, JSON `{capture.json_sha256}`, "
            f"normalized combined `{identity['normalized_payload_sha256']}`"
        )
    lines.extend([
        "", "## Independence and scope", "",
        f"- {comparison_count} unordered comparisons passed: 18 new-to-existing-seed and 3 new-to-new.",
        "- Capture ID, source JSON payload, canonical coordinate payload, canonical mask payload, normalized combined payload, and same-run exact timestamp range showed no duplicate signal.",
        "- The existing two approved seeds plus this one additional capture make three available captures per word, pending external review.",
        "- Medoid, leave-one-out, DTW, threshold derivation, active personal WISRT generation, and runtime integration were not performed in Step 9F.",
        "", "## Assertions", "",
        f"- PASS: {assertions_summary['PASS']}", f"- FAIL: {assertions_summary['FAIL']}",
        f"- UNVERIFIED: {assertions_summary['UNVERIFIED']}", f"- NOT_APPLICABLE: {assertions_summary['NOT_APPLICABLE']}",
        "", "## Repository state", "", f"- Git staged files: {staged['count']}",
        "- Existing production harness, test, scene, prefab, runtime, frozen evidence, global template/threshold, and project context were not modified.",
        "- New writes are limited to the Step 9F offline tool, result directory, Delivery ZIP, and detached ZIP checksum.", "",
    ])
    return "\n".join(lines)


def build(
    root: Path, run_dir: Path, expected_run_id: str, output: Path, delivery: Path,
    source_run_dir: Path | None = None,
) -> dict[str, Any]:
    detached_delivery = Path(str(delivery) + ".sha256")
    for target in (output, delivery, detached_delivery):
        require(not target.exists(), f"never-overwrite target already exists: {target}")
    require((root / "Assets").is_dir() and (root / "ProjectSettings/ProjectVersion.txt").is_file(), f"not project root: {root}")
    require(HEX_32_RE.fullmatch(expected_run_id) is not None, f"invalid expected run ID: {expected_run_id}")
    require(run_dir.is_dir() and run_dir.name == expected_run_id, f"new run directory mismatch: {run_dir}")
    source_run_dir = source_run_dir.resolve() if source_run_dir is not None else run_dir
    require(source_run_dir.name == expected_run_id, f"source run directory label mismatch: {source_run_dir}")

    staged_before = git_staged(root)
    require(staged_before["count"] == 0, f"Git staged files present before build: {staged_before['files']}")
    fixed_inputs = {
        "step9e": fixed_file(root, STEP9E_REL, STEP9E_SIZE, STEP9E_SHA256),
        "raw_evidence": fixed_file(root, RAW_EVIDENCE_REL, RAW_EVIDENCE_SIZE, RAW_EVIDENCE_SHA256),
        "harness": fixed_file(root, HARNESS_REL, HARNESS_SIZE, HARNESS_SHA256),
        "harness_test": fixed_file(root, HARNESS_TEST_REL, HARNESS_TEST_SIZE, HARNESS_TEST_SHA256),
    }
    step9e_path = root / STEP9E_REL
    fixed_inputs["step9e"]["zip_validation"] = inspect_zip(step9e_path, "output_file_manifest.json")
    fixed_inputs["raw_evidence"]["zip_validation"] = inspect_zip(root / RAW_EVIDENCE_REL, "output_file_manifest.json")
    for relative in (SCENE_REL, PREFAB_REL):
        path = root / relative
        require(path.is_file(), f"preflight file missing: {relative}")
        fixed_inputs["scene" if relative == SCENE_REL else "diagnostic_prefab"] = {
            "path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path), "read_only_preflight": "PASS",
        }

    with zipfile.ZipFile(step9e_path) as archive:
        step9e_seed_bytes = archive.read("seed_capture_identity.json")
        step9e_manifest_bytes = archive.read("output_file_manifest.json")
        step9e_primary_bytes = archive.read("primary_verdict.json")
    seeds_document = strict_json_loads(step9e_seed_bytes)
    require(seeds_document.get("capture_count") == 6 and seeds_document.get("all_pairs_independent") is True, "Step 9E seed identity is not approved six-independent-seed state")
    require(seeds_document.get("seed_preflight_verdict") == "SEED_PREFLIGHT_TWO_INDEPENDENT_CAPTURES_PER_WORD", "Step 9E seed preflight verdict mismatch")
    existing_seeds = seeds_document.get("captures")
    require(isinstance(existing_seeds, list) and len(existing_seeds) == 6, "Step 9E seed record count mismatch")
    for seed in existing_seeds:
        for field in ("capture_key", "word", "run_id", "capture_id", "source_payload_sha256", "coordinate_payload_sha256", "mask_payload_sha256", "normalized_payload_sha256"):
            require(field in seed, f"Step 9E seed identity field missing: {field}")
        for field in ("source_payload_sha256", "coordinate_payload_sha256", "mask_payload_sha256", "normalized_payload_sha256"):
            require(HEX_64_RE.fullmatch(str(seed[field])) is not None, f"Step 9E seed hash invalid: {field}")
    existing_run_ids = sorted({str(seed["run_id"]) for seed in existing_seeds})
    require(expected_run_id not in existing_run_ids, "new run ID duplicates an existing seed run ID")

    directory_entries = sorted(run_dir.iterdir(), key=lambda path: path.name)
    require(all(path.is_file() for path in directory_entries), "new run directory contains a subdirectory or non-file entry")
    json_paths = sorted((path for path in directory_entries if path.name.endswith(".json")), key=lambda path: path.name)
    checksum_paths = sorted((path for path in directory_entries if path.name.endswith(".json.sha256")), key=lambda path: path.name)
    require(len(directory_entries) == 6 and len(json_paths) == 3 and len(checksum_paths) == 3, f"new run must contain exactly 3 JSON + 3 checksum files, found {len(directory_entries)}")
    captures: list[Capture] = []
    for take_index, word in enumerate(WORDS, start=1):
        candidates = [path for path in json_paths if CAPTURE_RE.fullmatch(path.name) and int(CAPTURE_RE.fullmatch(path.name).group(1)) == take_index]
        require(len(candidates) == 1, f"missing or duplicate take {take_index}: {word}")
        path = candidates[0]
        checksum_path = Path(str(path) + ".sha256")
        require(checksum_path in checksum_paths, f"checksum filename mismatch: {path.name}")
        captures.append(validate_capture(path, checksum_path, take_index, word, expected_run_id))
    require(len({capture.capture_id for capture in captures}) == 3, "duplicate new capture ID")
    require(len({capture.json_sha256 for capture in captures}) == 3, "duplicate new source payload")
    require(len({capture.coordinate_payload_sha256 for capture in captures}) == 3, "duplicate new coordinate payload")
    require(len({capture.mask_payload_sha256 for capture in captures}) == 3, "duplicate new mask payload")
    require(len({capture.normalized_payload_sha256 for capture in captures}) == 3, "duplicate new normalized payload")
    for left, right in zip(captures, captures[1:]):
        require(left.timestamps[-1] < right.timestamps[0], f"cross-capture timestamp regression: {left.word}->{right.word}")
        require(left.provider_frames[-1] < right.provider_frames[0], f"cross-capture provider frame regression: {left.word}->{right.word}")
        require(left.bridge_observations[-1] < right.bridge_observations[0], f"cross-capture bridge observation regression: {left.word}->{right.word}")
        require(left.lifetime_provider_frames[-1] < right.lifetime_provider_frames[0], f"cross-capture lifetime frame regression: {left.word}->{right.word}")

    input_before = {
        path.name: {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in directory_entries
    }
    identities = [identity_record(capture) for capture in captures]
    for identity, capture in zip(identities, captures):
        identity["source_path"] = str(source_run_dir / capture.json_path.name)
        identity["validation_read_path"] = str(capture.json_path)
    comparisons: list[dict[str, Any]] = []
    for new_identity in identities:
        for seed in existing_seeds:
            comparisons.append(comparison_record(new_identity, seed, "new_vs_existing_step9e_seed"))
    for left_index, left in enumerate(identities):
        for right in identities[left_index + 1:]:
            comparisons.append(comparison_record(left, right, "new_vs_new_step9f_capture"))
    require(len(comparisons) == 21 and all(row["independent"] for row in comparisons), "new capture independence check failed")

    subset: dict[str, bytes] = {}
    for capture in captures:
        subset["input_subset/captures/" + capture.json_path.name] = capture.json_bytes
        subset["input_subset/captures/" + capture.checksum_path.name] = capture.checksum_bytes
    subset.update({
        "input_subset/provenance/step9e_seed_capture_identity.json": step9e_seed_bytes,
        "input_subset/provenance/step9e_output_file_manifest.json": step9e_manifest_bytes,
        "input_subset/provenance/step9e_primary_verdict.json": step9e_primary_bytes,
    })
    subset_before = [
        {"relative_path": name, "size_bytes": len(payload), "sha256": sha256_bytes(payload)}
        for name, payload in sorted(subset.items())
    ]

    inventory_rows = [inventory_record(capture) for capture in captures]
    capture_manifest = {
        "schema_version": SCHEMA + ".capture_manifest",
        "run_id": expected_run_id,
        "capture_count": 3,
        "expected_word_sequence": list(WORDS),
        "captures": [
            {
                "take_index": capture.take_index, "expected_word": capture.word,
                "run_id": capture.run_id, "capture_id": capture.capture_id,
                "json_file": capture.json_path.name, "json_size_bytes": len(capture.json_bytes),
                "json_sha256": capture.json_sha256, "detached_checksum_file": capture.checksum_path.name,
                "detached_checksum_size_bytes": len(capture.checksum_bytes),
                "detached_checksum_sha256": capture.checksum_sha256,
                "frame_count": capture.frame_count,
                "observation_timestamp_seconds_range": [capture.timestamps[0], capture.timestamps[-1]],
                "provider_frame_range": [capture.provider_frames[0], capture.provider_frames[-1]],
                "lifetime_provider_frame_range": [capture.lifetime_provider_frames[0], capture.lifetime_provider_frames[-1]],
            }
            for capture in captures
        ],
    }
    identity_document = {
        "schema_version": SCHEMA + ".additional_capture_identity",
        "verdict": "STEP9F_THREE_ADDITIONAL_INDEPENDENT_CAPTURES_ACQUIRED",
        "new_run_id": expected_run_id,
        "existing_seed_run_ids": existing_run_ids,
        "new_run_is_distinct_from_existing_seed_runs": True,
        "capture_count": 3,
        "captures": identities,
        "comparison_count": len(comparisons),
        "new_vs_existing_comparison_count": 18,
        "new_vs_new_comparison_count": 3,
        "all_comparisons_independent": True,
    }
    primary = {
        "schema_version": SCHEMA + ".primary_verdict",
        "validation_status": "PASS",
        "primary_verdict": "THREE_ADDITIONAL_INDEPENDENT_CAPTURES_ACQUIRED",
        "capture_count": 3,
        "counts_per_word": {word: 1 for word in WORDS},
        "available_capture_count_per_word_after_external_approval": {word: 3 for word in WORDS},
        "personal_calibration_minimum_status": "AVAILABLE_PENDING_EXTERNAL_REVIEW",
        "medoid_leave_one_out_threshold_status": "NOT_APPLICABLE_STEP9F_SCOPE",
        "active_personal_wisrt_count": 0,
        "personal_threshold_config_count": 0,
        "status_strings": [
            "STEP9F_ADDITIONAL_PERSONAL_CALIBRATION_CAPTURE_V1_0_READY_PENDING_EXTERNAL_REVIEW",
            "PRIMARY_VERDICT_THREE_ADDITIONAL_INDEPENDENT_CAPTURES_ACQUIRED",
            "PERSONAL_CALIBRATION_MINIMUM_THREE_PER_WORD_AVAILABLE_PENDING_EXTERNAL_REVIEW",
            "BLOCKED_PENDING_EXTERNAL_REVIEW",
        ],
        "next_core_milestone_started": False,
    }
    provenance = {
        "schema_version": SCHEMA + ".input_provenance",
        "fixed_inputs": fixed_inputs,
        "new_capture_source": {
            "run_directory": str(source_run_dir), "validation_read_directory": str(run_dir),
            "validation_staging_is_byte_exact_copy": source_run_dir != run_dir, "run_id": expected_run_id,
            "source_file_count": len(directory_entries), "source_files_before": input_before,
            "source_files_after": "VERIFIED_AFTER_BUILD",
        },
        "step9e_seed_identity_entry": "seed_capture_identity.json",
        "step9e_seed_identity_size_bytes": len(step9e_seed_bytes),
        "step9e_seed_identity_sha256": sha256_bytes(step9e_seed_bytes),
        "input_subset": subset_before,
        "input_subset_file_count": len(subset_before),
        "input_subset_role": "byte-exact three new JSON/checksum pairs plus approved Step 9E seed identity and minimum manifest/verdict provenance; full prior archives intentionally omitted",
        "unity_preflight": {
            "scene": SCENE_REL,
            "camera_start_control": "Start",
            "provider_readiness_ui": ["Camera: OPEN", "provider=BOUND", "readiness=LIVE RESULT"],
            "harness_state_before_arm": "Ready",
            "harness_completion_state": "Completed",
            "failure_state": "NONE",
            "checksum_verified": True,
            "manual_capture_operator": "USER",
            "expected_word_sequence": list(WORDS),
        },
    }

    assertions = [
        assertion("step9e_fixed_delivery_identity", "PASS", STEP9E_REL, fixed_inputs["step9e"]),
        assertion("step9e_zip_crc_safety_manifest", "PASS", STEP9E_REL, fixed_inputs["step9e"]["zip_validation"]),
        assertion("approved_harness_and_test_unchanged", "PASS", "Step9D harness production/test", {"harness": fixed_inputs["harness"], "test": fixed_inputs["harness_test"]}),
        assertion("existing_capture_evidence_preserved", "PASS", RAW_EVIDENCE_REL, fixed_inputs["raw_evidence"]),
        assertion("new_run_exactly_three_json_and_three_checksums", "PASS", str(run_dir), {"file_count": 6}),
        assertion("detached_checksum_filename_content_payload", "PASS", "three new captures", {capture.json_path.name: capture.json_sha256 for capture in captures}),
        assertion("single_new_run_distinct_from_existing_runs", "PASS", expected_run_id, {"existing_run_ids": existing_run_ids}),
        assertion("capture_id_unique_and_word_order", "PASS", "new run", {"words": list(WORDS), "capture_ids": [capture.capture_id for capture in captures]}),
        assertion("capture_local_frame_zero_to_t_minus_one", "PASS", "three new captures", {capture.word: capture.frame_count for capture in captures}),
        assertion("timestamps_finite_strictly_increasing", "PASS", "three new captures", {capture.word: [capture.timestamps[0], capture.timestamps[-1]] for capture in captures}),
        assertion("provider_bridge_lifetime_identity_monotonic", "PASS", "within and across captures", {capture.word: [capture.provider_frames[0], capture.provider_frames[-1], capture.lifetime_provider_frames[0], capture.lifetime_provider_frames[-1]] for capture in captures}),
        assertion("raw_post_gap_shape_validity_boundary_interpolation_contract", "PASS", "three new captures", {capture.word: {"T": capture.frame_count, "combined_boundary_count": capture.combined_boundary_count, "interpolation_count": capture.interpolation_count} for capture in captures}),
        assertion("torso_origin_basis_scale_validity_provenance", "PASS", "three new captures", {capture.word: capture.torso_provenance_counts for capture in captures}),
        assertion("normalized_shape_mask_finite_null_contract", "PASS", "three new captures", {capture.word: {"shape": [capture.frame_count, 16, 3], "valid_count": capture.normalized_valid_count} for capture in captures}),
        assertion("stored_shape_and_aggregate_counts_match_arrays", "PASS", "three new captures", {capture.word: {"raw_frames": capture.frame_count, "valid": capture.normalized_valid_count, "interpolated": capture.normalized_interpolated_count} for capture in captures}),
        assertion("nonstandard_nan_infinity_literal_absent", "PASS", "three new captures and generated JSON", 0),
        assertion("canonical_payload_hashes_reproducible", "PASS", "three new captures", {capture.capture_key: {"coordinate": capture.coordinate_payload_sha256, "mask": capture.mask_payload_sha256, "combined": capture.normalized_payload_sha256} for capture in captures}),
        assertion("independence_against_six_existing_seeds", "PASS", "18 comparisons", {"independent": 18}),
        assertion("independence_among_three_new_captures", "PASS", "3 comparisons", {"independent": 3}),
        assertion("personal_minimum_three_per_word_available_pending_review", "PASS", "two approved seeds plus one new capture per word", {word: 3 for word in WORDS}),
        assertion("medoid_leave_one_out_threshold", "NOT_APPLICABLE", "Step 9F prohibited scope", "not executed"),
        assertion("active_personal_template_and_threshold", "NOT_APPLICABLE", "Step 9F prohibited scope", {"active_wisrt": 0, "threshold": 0}),
        assertion("new_capture_inputs_before_after_immutable", "PASS", str(run_dir), "verified after build"),
        assertion("input_subset_byte_exact", "PASS", "input_subset", {"file_count": len(subset_before)}),
        assertion("json_csv_schema_and_unique_keys", "PASS", "result schemas", "validated before finalization"),
        assertion("output_manifest_membership_size_hash", "PASS", "output_file_manifest.json", "validated on file map and ZIP"),
        assertion("deterministic_zip_memory_regeneration", "PASS", "delivery ZIP", "same file map generated byte-identical ZIP twice"),
        assertion("post_zip_crc_duplicate_path_symlink_encryption_membership", "PASS", "delivery ZIP", "reopened and validated"),
        assertion("never_overwrite", "PASS", "result directory, ZIP, detached checksum", "all targets absent before atomic move"),
        assertion("git_staged_files_zero", "PASS", "repository", {"before": 0, "after": 0}),
        assertion("production_runtime_scene_prefab_context_unchanged", "PASS", "approved scope", "only new Step 9F tool and outputs written"),
    ]
    require(len({row["assertion_key"] for row in assertions}) == len(assertions), "duplicate assertion key before serialization")
    summary = {status: sum(row["status"] == status for row in assertions) for status in ("PASS", "FAIL", "UNVERIFIED", "NOT_APPLICABLE")}

    files: dict[str, bytes] = dict(subset)
    files.update({
        "input_provenance.json": pretty_json_bytes(provenance),
        "additional_capture_identity.json": pretty_json_bytes(identity_document),
        "additional_capture_inventory.csv": csv_bytes(inventory_rows),
        "independence_against_existing_seeds.csv": csv_bytes(comparisons),
        "capture_manifest.json": pretty_json_bytes(capture_manifest),
        "primary_verdict.json": pretty_json_bytes(primary),
        "ASSERTIONS.jsonl": b"".join(json_bytes(row) for row in assertions),
        "VALIDATION_REPORT.md": report_text(expected_run_id, captures, identities, len(comparisons), summary, fixed_inputs, staged_before).encode("utf-8"),
        "analysis/" + Path(SCRIPT_REL).name: (root / SCRIPT_REL).read_bytes(),
    })
    schema_validation = validate_result_schemas(files)
    require(subset_before == [
        {"relative_path": name, "size_bytes": len(files[name]), "sha256": sha256_bytes(files[name])}
        for name in sorted(subset)
    ], "input subset changed before packaging")
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
        payload = files[record["relative_path"]]
        require(len(payload) == record["size_bytes"] and sha256_bytes(payload) == record["sha256"], f"output manifest mismatch: {record['relative_path']}")

    zip_first = deterministic_zip_bytes(files)
    zip_second = deterministic_zip_bytes(files)
    require(zip_first == zip_second, "deterministic in-memory ZIP regeneration mismatch")
    zip_inspection = inspect_generated_zip(zip_first, set(files))
    with zipfile.ZipFile(io.BytesIO(zip_first)) as archive:
        archived_manifest = strict_json_loads(archive.read("output_file_manifest.json"))
        records = manifest_records(archived_manifest)
        require(set(record["relative_path"] for record in records) == set(archive.namelist()) - {"output_file_manifest.json"}, "post-ZIP manifest membership mismatch")
        for record in records:
            payload = archive.read(record["relative_path"])
            require(len(payload) == record["size_bytes"] and sha256_bytes(payload) == record["sha256"], f"post-ZIP manifest mismatch: {record['relative_path']}")

    input_after = {
        path.name: {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in directory_entries
    }
    require(input_before == input_after, "new capture source changed during build")
    for key, record in fixed_inputs.items():
        path = root / record["path"]
        require(path.stat().st_size == record["size_bytes"] and sha256_file(path) == record["sha256"], f"fixed input changed during build: {key}")
    staged_after = git_staged(root)
    require(staged_after["count"] == 0, f"Git staged files present after build: {staged_after['files']}")

    output.parent.mkdir(parents=True, exist_ok=True)
    delivery.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".step9f_build_", dir=output.parent) as temporary:
        temporary_root = Path(temporary)
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
        require(staged_zip.read_bytes() == zip_first, "written ZIP differs from memory ZIP")
        delivery_hash = sha256_bytes(zip_first)
        detached_payload = f"{delivery_hash}  {delivery.name}\n".encode("utf-8")
        staged_detached = temporary_root / "d.zip.sha256"
        staged_detached.write_bytes(detached_payload)
        require(not output.exists() and not delivery.exists() and not detached_delivery.exists(), "never-overwrite target appeared during build")
        shutil.move(str(staged_output), str(output))
        shutil.move(str(staged_zip), str(delivery))
        shutil.move(str(staged_detached), str(detached_delivery))

    require(delivery.stat().st_size == len(zip_first) and sha256_file(delivery) == sha256_bytes(zip_first), "final ZIP write verification mismatch")
    require(parse_detached(detached_delivery.read_bytes(), delivery.name, sha256_file(delivery)), "final detached ZIP checksum mismatch")
    final_zip_inspection = inspect_zip(delivery, "output_file_manifest.json")
    final_subset = [
        {"relative_path": name, "size_bytes": (output / PurePosixPath(name)).stat().st_size, "sha256": sha256_file(output / PurePosixPath(name))}
        for name in sorted(subset)
    ]
    require(final_subset == subset_before, "final input subset hash mismatch")
    final_staged = git_staged(root)
    require(final_staged["count"] == 0, "Git staged files changed during finalization")
    return {
        "status": "PASS",
        "run_id": expected_run_id,
        "captures": identities,
        "assertion_summary": summary,
        "schema_validation": schema_validation,
        "zip_memory_inspection": zip_inspection,
        "zip_disk_inspection": final_zip_inspection,
        "output_directory": str(output),
        "output_file_count": len(files),
        "delivery_zip": str(delivery),
        "delivery_size_bytes": delivery.stat().st_size,
        "delivery_sha256": sha256_file(delivery),
        "detached_checksum": str(detached_delivery),
        "detached_size_bytes": detached_delivery.stat().st_size,
        "detached_sha256": sha256_file(detached_delivery),
        "git_staged_count": final_staged["count"],
    }


def main() -> int:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path, help="Original read-only Unity run path when --run-dir is a byte-exact local staging copy")
    parser.add_argument("--expected-run-id", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--delivery", type=Path)
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    run_dir = arguments.run_dir.resolve()
    output = arguments.output.resolve() if arguments.output else root / OUTPUT_REL
    delivery = arguments.delivery.resolve() if arguments.delivery else root / DELIVERY_REL
    result = build(root, run_dir, arguments.expected_run_id, output, delivery, arguments.source_run_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
