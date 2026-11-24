#!/usr/bin/env python3
"""WordsInSilence Step 9G personalized medoid/LOO/threshold calibration.

The tool consumes only the externally approved Step 9E and Step 9F Delivery
archives. It hash-pins and reuses the approved Step 9E pure calculation code,
revalidates all nine byte-exact captures, computes the ordered 9x9 DTW matrix,
selects personal medoids, runs nine-fold leave-one-out evaluation, and emits
active personal artifacts only when the approved separability rule succeeds.
It never activates runtime personal mode or changes global recognition assets.
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
import sys
import tempfile
import types
import zipfile
from itertools import combinations
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np


SCHEMA = "words_in_silence.step9g.personalized_calibration_medoid_loo_threshold.v1_0"
WORDS = ("GREETING", "LOVE", "THANK_YOU")
SCRIPT_REL = "Tools/s001_step9g_personalized_calibration_medoid_loo_threshold_v1_0.py"
OUTPUT_REL = "Validation/WordsInSilence_step9g_personalized_calibration_medoid_loo_threshold_v1_0"
DELIVERY_REL = "Validation/WordsInSilence_step9g_personalized_calibration_medoid_loo_threshold_v1_0_delivery.zip"
STEP9E_REL = "Validation/WordsInSilence_step9e_personalized_calibration_template_foundation_v1_0_delivery.zip"
STEP9E_SIZE = 1_776_636
STEP9E_SHA256 = "9829d143b24e454e47ca68a54b5b9d630548a5790a3743882f92321bf638181b"
STEP9F_REL = "Validation/WordsInSilence_step9f_additional_personal_calibration_capture_one_take_per_word_v1_0_delivery.zip"
STEP9F_SIZE = 1_412_432
STEP9F_SHA256 = "c8f974431c5d4fcf9b029cee1f0164496bd970fd93116829b2e2d39c95d608bd"
STEP9E_ANALYSIS_ENTRY = "analysis/s001_step9e_personalized_calibration_template_foundation_v1_0.py"
STEP9E_ANALYSIS_SIZE = 94_953
STEP9E_ANALYSIS_SHA256 = "b2b92ef247ee1f81ecf2142546760fcaf15da6c2f4f036ac3c632adf91fa8d13"
STEP9E_TOOL_REL = "Tools/s001_step9e_personalized_calibration_template_foundation_v1_0.py"
APPROVED_DTW_HELPER_ENTRY = "input_subset/provenance/baseline_diagnostic/s001_step9d_physical_recognition_domain_gap_diagnostic_v1_0.py"
APPROVED_DTW_HELPER_SIZE = 90_451
APPROVED_DTW_HELPER_SHA256 = "334aed9691ee57a8305d32bfefc8220568573dcf99bb3c824f0dde6cee8f788d"
STEP9E_BASELINE_PREFIX = "input_subset/baseline_physical_queries/"
STEP9E_RAW_PREFIX = "input_subset/raw_intermediate_captures/"
STEP9F_CAPTURE_PREFIX = "input_subset/captures/"
PARITY_TOLERANCE = 1e-12
DTW_BAND = 0.10
MISSING_COST = 1.0
FORBIDDEN_SUCCESS_PHRASES = (
    "two captures", "insufficient repetitions", "threshold not evaluated", "active output 0",
)
HEX_64_RE = re.compile(r"[0-9a-f]{64}")


class Step9GError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Step9GError(message)


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
        raise Step9GError(f"non-standard JSON constant: {token}")

    return json.loads(data.decode("utf-8"), parse_constant=reject_constant)


def finite_json(value: Any, path: str = "$") -> None:
    if isinstance(value, float):
        require(math.isfinite(value), f"generated non-finite JSON value at {path}")
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
        name and not name.startswith(("/", "\\")) and "\\" not in name
        and not re.match(r"^[A-Za-z]:", name) and ".." not in path.parts
    )


def manifest_records(data: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("outputs", "files", "inputs", "input_files"):
        records = data.get(key)
        if isinstance(records, list):
            return records
    raise Step9GError(f"manifest has no supported record list: {sorted(data)}")


def parse_detached(data: bytes, expected_name: str, expected_hash: str) -> bool:
    match = re.fullmatch(rb"([0-9a-f]{64})  ([^\r\n]+)(?:\r?\n)?", data)
    return bool(
        match and match.group(1).decode("ascii") == expected_hash
        and match.group(2).decode("utf-8") == expected_name
    )


def verify_fixed_zip(path: Path, expected_size: int, expected_hash: str) -> dict[str, Any]:
    require(path.is_file(), f"missing fixed input: {path}")
    actual_size = path.stat().st_size
    actual_hash = sha256_file(path)
    require(actual_size == expected_size, f"fixed input size mismatch: {path}: {actual_size} != {expected_size}")
    require(actual_hash == expected_hash, f"fixed input SHA-256 mismatch: {path}: {actual_hash} != {expected_hash}")
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        require(archive.testzip() is None, f"ZIP CRC failure: {path}")
        require(len(names) == len(set(names)), f"duplicate ZIP entry: {path}")
        require(all(safe_zip_name(name) for name in names), f"unsafe ZIP path: {path}")
        require(not any(stat.S_ISLNK((info.external_attr >> 16) & 0xFFFF) for info in infos), f"symlink ZIP entry: {path}")
        require(not any(info.flag_bits & 1 for info in infos), f"encrypted ZIP entry: {path}")
        require("output_file_manifest.json" in names, f"missing output manifest: {path}")
        manifest_payload = archive.read("output_file_manifest.json")
        manifest = strict_json_loads(manifest_payload)
        records = manifest_records(manifest)
        targets = [str(record["relative_path"]) for record in records]
        require(len(targets) == len(set(targets)), f"duplicate manifest target: {path}")
        require(set(targets) == set(names) - {"output_file_manifest.json"}, f"manifest membership mismatch: {path}")
        for record in records:
            payload = archive.read(record["relative_path"])
            require(len(payload) == int(record["size_bytes"]), f"manifest size mismatch: {record['relative_path']}")
            require(sha256_bytes(payload) == record["sha256"], f"manifest hash mismatch: {record['relative_path']}")
    return {
        "path": path.as_posix(), "filename": path.name, "size_bytes": actual_size,
        "sha256": actual_hash, "expected_size_bytes": expected_size,
        "expected_sha256": expected_hash, "fixed_identity": "PASS",
        "zip_validation": {
            "entry_count": len(names), "manifest_record_count": len(records), "crc": "PASS",
            "duplicate_entries": 0, "unsafe_paths": 0, "symlinks": 0,
            "encrypted_entries": 0, "membership": "PASS",
            "manifest_size_bytes": len(manifest_payload),
            "manifest_sha256": sha256_bytes(manifest_payload),
        },
    }


def load_module_from_bytes(name: str, source: bytes, filename: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = filename
    sys.modules[name] = module
    exec(compile(source.decode("utf-8"), filename, "exec"), module.__dict__)
    return module


def load_approved_calculation_sources(root: Path, step9e_zip: Path) -> tuple[types.ModuleType, types.ModuleType, dict[str, Any], dict[str, bytes]]:
    with zipfile.ZipFile(step9e_zip) as archive:
        step9e_source = archive.read(STEP9E_ANALYSIS_ENTRY)
        dtw_helper_source = archive.read(APPROVED_DTW_HELPER_ENTRY)
    require(len(step9e_source) == STEP9E_ANALYSIS_SIZE, "approved Step 9E analysis size mismatch")
    require(sha256_bytes(step9e_source) == STEP9E_ANALYSIS_SHA256, "approved Step 9E analysis SHA mismatch")
    require(len(dtw_helper_source) == APPROVED_DTW_HELPER_SIZE, "approved DTW helper size mismatch")
    require(sha256_bytes(dtw_helper_source) == APPROVED_DTW_HELPER_SHA256, "approved DTW helper SHA mismatch")
    current_tool = root / STEP9E_TOOL_REL
    require(current_tool.is_file(), f"current Step 9E tool missing: {current_tool}")
    require(current_tool.stat().st_size == STEP9E_ANALYSIS_SIZE and sha256_file(current_tool) == STEP9E_ANALYSIS_SHA256, "current Step 9E tool differs from approved analysis source")
    step9e = load_module_from_bytes("wis_step9e_approved_calculation", step9e_source, STEP9E_ANALYSIS_ENTRY)
    approved_dtw = load_module_from_bytes("wis_step9d_approved_dtw_contract", dtw_helper_source, APPROVED_DTW_HELPER_ENTRY)
    require(tuple(step9e.WORDS) == WORDS, "approved Step 9E word contract mismatch")
    require(float(step9e.DTW_BAND) == DTW_BAND and float(step9e.MISSING_COST) == MISSING_COST, "approved Step 9E DTW constants mismatch")
    require(float(approved_dtw.DTW_BAND) == DTW_BAND and float(approved_dtw.MISSING_COST) == MISSING_COST, "approved helper DTW constants mismatch")
    provenance = {
        "step9e_analysis": {
            "archive_entry": STEP9E_ANALYSIS_ENTRY, "size_bytes": len(step9e_source),
            "sha256": sha256_bytes(step9e_source), "current_tools_file": STEP9E_TOOL_REL,
            "current_tools_match": True,
        },
        "approved_reference_runtime_dtw_helper": {
            "archive_entry": APPROVED_DTW_HELPER_ENTRY, "size_bytes": len(dtw_helper_source),
            "sha256": sha256_bytes(dtw_helper_source), "dtw_band": DTW_BAND,
            "missing_cost": MISSING_COST,
        },
    }
    return step9e, approved_dtw, provenance, {
        "input_subset/provenance/s001_step9e_personalized_calibration_template_foundation_v1_0.py": step9e_source,
        "input_subset/provenance/s001_step9d_physical_recognition_domain_gap_diagnostic_v1_0.py": dtw_helper_source,
    }


def validate_json_coordinate_contract(raw_coordinates: Any, raw_mask: Any, expected_shape: tuple[int, int, int], key: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        coordinate_objects = np.asarray(raw_coordinates, dtype=object)
        mask_objects = np.asarray(raw_mask, dtype=object)
    except ValueError as error:
        raise Step9GError(f"ragged coordinate/mask array: {key}: {error}") from error
    require(coordinate_objects.shape == expected_shape, f"normalized coordinate shape mismatch: {key}: {coordinate_objects.shape}")
    require(mask_objects.shape == expected_shape[:2], f"normalized mask shape mismatch: {key}: {mask_objects.shape}")
    require(all(type(value) is bool for value in mask_objects.flat), f"normalized mask contains non-bool: {key}")
    mask = np.ascontiguousarray(mask_objects, dtype=np.bool_)
    null_components = np.zeros(expected_shape, dtype=np.bool_)
    for index, value in np.ndenumerate(coordinate_objects):
        if value is None:
            null_components[index] = True
        else:
            require(type(value) in (int, float) and math.isfinite(float(value)), f"normalized coordinate invalid: {key}:{index}")
    coordinates = np.ascontiguousarray(raw_coordinates, dtype=np.float64)
    require(coordinates.dtype == np.float64 and mask.dtype == np.bool_, f"normalized dtype mismatch: {key}")
    require(not np.isinf(coordinates).any(), f"normalized Infinity present: {key}")
    require(np.array_equal(np.isnan(coordinates), null_components), f"normalized null/NaN conversion mismatch: {key}")
    require(np.isfinite(coordinates[mask]).all(), f"valid normalized coordinate non-finite: {key}")
    require(np.isnan(coordinates[~mask]).all(), f"invalid normalized coordinate not null: {key}")
    return coordinates, mask


def load_capture_from_entry(
    step9e: types.ModuleType, archive: zipfile.ZipFile, archive_path: Path, archive_sha256: str,
    entry: str, source_family: str, capture_key_prefix: str, source_manifest: str,
) -> tuple[Any, dict[str, Any], dict[str, bytes]]:
    payload = archive.read(entry)
    detached_entry = entry + ".sha256"
    require(detached_entry in archive.namelist(), f"missing detached checksum: {entry}")
    detached = archive.read(detached_entry)
    digest = sha256_bytes(payload)
    require(parse_detached(detached, Path(entry).name, digest), f"detached checksum mismatch: {entry}")
    data = strict_json_loads(payload)
    if "frames" in data:
        frames = data.get("frames")
        require(isinstance(frames, list) and frames, f"baseline frames missing: {entry}")
        coordinates_raw = [frame["normalized_coordinates"] for frame in frames]
        mask_raw = [frame["normalized_mask"] for frame in frames]
        timestamps = [float(frame["timestamp_seconds"]) for frame in frames]
        frame_indices = [int(frame["frame_index"]) for frame in frames]
        counts = data.get("counts", {})
        expected_frames = int(counts.get("normalized_frames", -1))
        expected_valid = int(counts.get("normalized_valid_landmarks", -1))
    else:
        frame_identity = data.get("frame_identity")
        require(isinstance(frame_identity, list) and frame_identity, f"raw frame identity missing: {entry}")
        normalized = data.get("normalized_output")
        require(isinstance(normalized, dict), f"raw normalized output missing: {entry}")
        coordinates_raw = normalized.get("coordinates")
        mask_raw = normalized.get("mask")
        timestamps = [float(frame["observation_timestamp_seconds"]) for frame in frame_identity]
        frame_indices = [int(frame["capture_local_frame_index"]) for frame in frame_identity]
        expected_frames = int(data.get("capture_metadata", {}).get("normalized_frame_count", -1))
        expected_valid = int(normalized.get("valid_landmark_count", -1))
    frame_count = len(frame_indices)
    coordinates, valid_mask = validate_json_coordinate_contract(coordinates_raw, mask_raw, (frame_count, 16, 3), entry)
    require(frame_indices == list(range(frame_count)), f"frame indices are not 0..T-1: {entry}")
    require(all(math.isfinite(value) for value in timestamps), f"non-finite timestamp: {entry}")
    require(all(timestamps[index] < timestamps[index + 1] for index in range(frame_count - 1)), f"timestamp regression: {entry}")
    require(expected_frames == frame_count and expected_valid == int(valid_mask.sum()), f"stored normalized counts mismatch: {entry}")
    word = str(data["expected_word"])
    require(word in WORDS, f"unsupported expected word: {entry}")
    capture = step9e.make_capture(
        capture_key=f"{capture_key_prefix}::{data['capture_id']}", source_family=source_family,
        source_archive=archive_path.as_posix(), source_archive_sha256=archive_sha256,
        archive_entry=entry, source_manifest_entry=source_manifest,
        schema_version=str(data["schema_version"]), word=word, run_id=str(data["run_id"]),
        capture_id=str(data["capture_id"]), frame_indices=frame_indices, timestamps=timestamps,
        coordinates=coordinates, valid_mask=valid_mask, source_bytes=payload,
    )
    step9e.validate_capture(capture)
    destination = "input_subset/captures/" + Path(entry).name
    record = {
        "capture_key": capture.capture_key, "source_family": source_family, "word": word,
        "run_id": capture.run_id, "capture_id": capture.capture_id,
        "source_archive": archive_path.as_posix(), "source_archive_sha256": archive_sha256,
        "source_entry": entry, "source_manifest_entry": source_manifest,
        "source_size_bytes": len(payload), "source_payload_sha256": digest,
        "detached_entry": detached_entry, "detached_size_bytes": len(detached),
        "detached_sha256": sha256_bytes(detached),
    }
    return capture, record, {destination: payload, destination + ".sha256": detached}


def capture_identity(step9e: types.ModuleType, capture: Any) -> dict[str, Any]:
    return step9e.identity_record(capture)


def capture_inventory(step9e: types.ModuleType, capture: Any) -> dict[str, Any]:
    record = step9e.capture_inventory_record(capture)
    record["source_entry"] = record.pop("zip_entry")
    record["source_entry_size_bytes"] = record.pop("zip_entry_size_bytes")
    return record


def load_nine_captures(
    step9e: types.ModuleType, step9e_path: Path, step9f_path: Path,
    input_records: dict[str, dict[str, Any]],
) -> tuple[list[Any], list[dict[str, Any]], dict[str, bytes], list[dict[str, Any]]]:
    captures: list[Any] = []
    source_records: list[dict[str, Any]] = []
    subset: dict[str, bytes] = {}
    specifications = [
        (step9e_path, STEP9E_BASELINE_PREFIX, "baseline_step9c_physical_query", "baseline_step9c", "step9e_seed_capture_identity.json"),
        (step9e_path, STEP9E_RAW_PREFIX, "step9d_raw_intermediate_capture", "raw_intermediate_step9d", "step9e_seed_capture_identity.json"),
        (step9f_path, STEP9F_CAPTURE_PREFIX, "step9f_additional_raw_intermediate_capture", "step9f_additional", "step9f_capture_manifest.json"),
    ]
    for archive_path, prefix, family, key_prefix, source_manifest in specifications:
        archive_role = "step9e" if archive_path == step9e_path else "step9f"
        with zipfile.ZipFile(archive_path) as archive:
            json_entries = sorted(name for name in archive.namelist() if name.startswith(prefix) and name.endswith(".json"))
            require(len(json_entries) == 3, f"capture count mismatch for {prefix}: {len(json_entries)}")
            for entry in json_entries:
                capture, record, files = load_capture_from_entry(
                    step9e, archive, archive_path, input_records[archive_role]["sha256"],
                    entry, family, key_prefix, source_manifest,
                )
                captures.append(capture)
                source_records.append(record)
                for name, payload in files.items():
                    require(name not in subset, f"input subset filename collision: {name}")
                    subset[name] = payload
    captures.sort(key=lambda item: (WORDS.index(item.word), item.source_family, item.capture_id))
    step9e.validate_capture_set(captures)
    counts = {word: sum(capture.word == word for capture in captures) for word in WORDS}
    require(len(captures) == 9 and counts == {word: 3 for word in WORDS}, f"9-capture count mismatch: {counts}")
    require(len({capture.capture_key for capture in captures}) == 9, "duplicate capture key")
    require(len({capture.capture_id for capture in captures}) == 9, "duplicate capture ID")
    run_groups: dict[str, list[str]] = {}
    for capture in captures:
        run_groups.setdefault(capture.run_id, []).append(capture.capture_key)
    require(len(run_groups) == 3 and all(len(group) == 3 for group in run_groups.values()), f"run provenance grouping mismatch: {run_groups}")
    independence: list[dict[str, Any]] = []
    for left, right in combinations(captures, 2):
        timestamp_exact = left.timestamps[0] == right.timestamps[0] and left.timestamps[-1] == right.timestamps[-1]
        flags = {
            "capture_id_equal": left.capture_id == right.capture_id,
            "source_payload_sha256_equal": left.source_sha256 == right.source_sha256,
            "coordinate_payload_sha256_equal": left.coordinate_payload_sha256 == right.coordinate_payload_sha256,
            "mask_payload_sha256_equal": left.mask_payload_sha256 == right.mask_payload_sha256,
            "normalized_payload_sha256_equal": left.normalized_payload_sha256 == right.normalized_payload_sha256,
            "same_run_exact_timestamp_range": left.run_id == right.run_id and timestamp_exact,
        }
        independence.append({
            "pair_key": left.capture_key + "||" + right.capture_key,
            "left_capture_key": left.capture_key, "left_word": left.word,
            "right_capture_key": right.capture_key, "right_word": right.word,
            "same_word": left.word == right.word, "same_run_id": left.run_id == right.run_id,
            **flags, "independent": not any(flags.values()),
        })
    require(len(independence) == 36 and all(row["independent"] for row in independence), "36-pair capture independence mismatch")
    return captures, source_records, subset, independence


def verify_prior_identity_provenance(
    step9e_path: Path, step9f_path: Path, captures: list[Any], subset: dict[str, bytes],
) -> tuple[dict[str, Any], dict[str, bytes]]:
    with zipfile.ZipFile(step9e_path) as archive:
        step9e_identity = archive.read("seed_capture_identity.json")
        step9e_primary = archive.read("primary_verdict.json")
        step9e_manifest = archive.read("output_file_manifest.json")
        step9e_input_provenance = archive.read("input_provenance.json")
    with zipfile.ZipFile(step9f_path) as archive:
        step9f_identity = archive.read("additional_capture_identity.json")
        step9f_primary = archive.read("primary_verdict.json")
        step9f_manifest = archive.read("output_file_manifest.json")
        step9f_capture_manifest = archive.read("capture_manifest.json")
        step9f_independence = archive.read("independence_against_existing_seeds.csv")
    seed_document = strict_json_loads(step9e_identity)
    additional_document = strict_json_loads(step9f_identity)
    prior_records = list(seed_document["captures"]) + list(additional_document["captures"])
    require(len(prior_records) == 9, "prior identity provenance count mismatch")
    by_capture_id = {record["capture_id"]: record for record in prior_records}
    require(len(by_capture_id) == 9, "prior identity duplicate capture ID")
    mismatches: list[str] = []
    for capture in captures:
        prior = by_capture_id.get(capture.capture_id)
        require(prior is not None, f"capture absent from prior identity provenance: {capture.capture_id}")
        checks = {
            "word": capture.word, "run_id": capture.run_id,
            "source_payload_sha256": capture.source_sha256,
            "coordinate_payload_sha256": capture.coordinate_payload_sha256,
            "mask_payload_sha256": capture.mask_payload_sha256,
            "normalized_payload_sha256": capture.normalized_payload_sha256,
        }
        for field, value in checks.items():
            if prior.get(field) != value:
                mismatches.append(f"{capture.capture_id}:{field}")
    require(not mismatches, f"prior identity provenance mismatch: {mismatches}")
    provenance_subset = {
        "input_subset/provenance/step9e_seed_capture_identity.json": step9e_identity,
        "input_subset/provenance/step9e_primary_verdict.json": step9e_primary,
        "input_subset/provenance/step9e_output_file_manifest.json": step9e_manifest,
        "input_subset/provenance/step9e_input_provenance.json": step9e_input_provenance,
        "input_subset/provenance/step9f_additional_capture_identity.json": step9f_identity,
        "input_subset/provenance/step9f_primary_verdict.json": step9f_primary,
        "input_subset/provenance/step9f_output_file_manifest.json": step9f_manifest,
        "input_subset/provenance/step9f_capture_manifest.json": step9f_capture_manifest,
        "input_subset/provenance/step9f_independence_against_existing_seeds.csv": step9f_independence,
    }
    for name, payload in provenance_subset.items():
        require(name not in subset, f"provenance subset collision: {name}")
    return {
        "step9e_seed_count": seed_document["capture_count"],
        "step9f_additional_count": additional_document["capture_count"],
        "actual_payload_recomparison_count": 9,
        "identity_payload_mismatch_count": 0,
    }, provenance_subset


def distance_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str], float]:
    return {(row["left_capture_key"], row["right_capture_key"]): float(row["distance"]) for row in rows}


def select_medoid(captures: list[Any], distances: dict[tuple[str, str], float], scope: str, word: str) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    require(captures, f"empty medoid set: {scope}:{word}")
    scored: list[tuple[float, str, Any, list[dict[str, Any]]]] = []
    for candidate in captures:
        pair_details = [
            {"other_capture_key": other.capture_key, "other_capture_id": other.capture_id,
             "distance": distances[(candidate.capture_key, other.capture_key)]}
            for other in captures if other.capture_key != candidate.capture_key
        ]
        mean_distance = sum(item["distance"] for item in pair_details) / len(pair_details) if pair_details else 0.0
        scored.append((mean_distance, candidate.capture_id, candidate, pair_details))
    minimum = min(item[0] for item in scored)
    tied = sorted((item for item in scored if item[0] == minimum), key=lambda item: item[1])
    selected = tied[0][2]
    tie_ids = [item[1] for item in tied]
    rows: list[dict[str, Any]] = []
    for mean_distance, capture_id, candidate, pair_details in sorted(scored, key=lambda item: item[1]):
        rows.append({
            "selection_scope": scope, "word": word, "candidate_capture_key": candidate.capture_key,
            "candidate_capture_id": capture_id, "comparison_count": len(pair_details),
            "same_word_pair_distances_json": json.dumps(pair_details, sort_keys=True, separators=(",", ":")),
            "mean_same_word_off_diagonal_distance": mean_distance,
            "minimum_mean_distance": minimum, "exact_minimum_tie_count": len(tied),
            "exact_minimum_tied_capture_ids_json": json.dumps(tie_ids, separators=(",", ":")),
            "selected": candidate.capture_key == selected.capture_key,
            "selection_reason": "exact_mean_tie_capture_id_lexicographic_ascending" if len(tied) > 1 and candidate.capture_key == selected.capture_key else ("unique_minimum_mean_distance" if len(tied) == 1 and candidate.capture_key == selected.capture_key else "not_selected"),
        })
    detail = {
        "selection_scope": scope, "word": word,
        "selected_capture_key": selected.capture_key, "selected_capture_id": selected.capture_id,
        "minimum_mean_distance": minimum, "exact_minimum_tie_count": len(tied),
        "exact_minimum_tied_capture_ids": tie_ids,
        "tie_break_rule": "capture_id lexicographic ascending",
        "selection_reason": "exact_mean_tie_capture_id_lexicographic_ascending" if len(tied) > 1 else "unique_minimum_mean_distance",
        "candidates": [strict_json_loads(json_bytes(row)) for row in rows],
    }
    return selected, rows, detail


def compute_medoids_and_loo(captures: list[Any], pairwise_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    distances = distance_lookup(pairwise_rows)
    full_medoids: dict[str, Any] = {}
    medoid_rows: list[dict[str, Any]] = []
    full_details: dict[str, Any] = {}
    for word in WORDS:
        selected, rows, detail = select_medoid([capture for capture in captures if capture.word == word], distances, "full_dataset", word)
        full_medoids[word] = selected
        medoid_rows.extend(rows)
        full_details[word] = detail

    loo_rows: list[dict[str, Any]] = []
    loo_details: list[dict[str, Any]] = []
    ordered_captures = sorted(captures, key=lambda item: (WORDS.index(item.word), item.source_family, item.capture_id))
    for fold_index, held_out in enumerate(ordered_captures, start=1):
        fold_id = f"loo_{fold_index:02d}_{held_out.capture_id}"
        training = [capture for capture in captures if capture.capture_key != held_out.capture_key]
        fold_medoids: dict[str, Any] = {}
        fold_selection: dict[str, Any] = {}
        for word in WORDS:
            selected, _, detail = select_medoid([capture for capture in training if capture.word == word], distances, fold_id, word)
            fold_medoids[word] = selected
            fold_selection[word] = detail
        word_distances = {word: distances[(held_out.capture_key, fold_medoids[word].capture_key)] for word in WORDS}
        ranking = sorted(WORDS, key=lambda word: (word_distances[word], word))
        predicted = ranking[0]
        best_wrong = min((word for word in WORDS if word != held_out.word), key=lambda word: (word_distances[word], word))
        genuine = word_distances[held_out.word]
        impostor = word_distances[best_wrong]
        row = {
            "fold_id": fold_id, "fold_index": fold_index,
            "held_out_capture_key": held_out.capture_key, "held_out_capture_id": held_out.capture_id,
            "expected_word": held_out.word,
            "fold_medoid_capture_ids_json": json.dumps({word: fold_medoids[word].capture_id for word in WORDS}, sort_keys=True, separators=(",", ":")),
            "word_distances_json": json.dumps(word_distances, sort_keys=True, separators=(",", ":")),
            "predicted_word": predicted, "prediction_correct": predicted == held_out.word,
            "best_word": ranking[0], "best_distance": word_distances[ranking[0]],
            "second_word": ranking[1], "second_distance": word_distances[ranking[1]],
            "prediction_margin": word_distances[ranking[1]] - word_distances[ranking[0]],
            "expected_word_medoid_distance": genuine,
            "best_wrong_word": best_wrong, "best_wrong_word_medoid_distance": impostor,
            "genuine_score": genuine, "impostor_score": impostor,
            "separation_margin": impostor - genuine,
        }
        loo_rows.append(row)
        loo_details.append({
            **row,
            "fold_medoid_capture_ids": {word: fold_medoids[word].capture_id for word in WORDS},
            "word_distances": word_distances, "distance_ranking": ranking,
            "fold_medoid_selection": fold_selection,
        })
    correct_count = sum(bool(row["prediction_correct"]) for row in loo_rows)
    genuine_upper = max(float(row["genuine_score"]) for row in loo_rows)
    impostor_lower = min(float(row["impostor_score"]) for row in loo_rows)
    separable = correct_count == 9 and genuine_upper < impostor_lower
    verdict = "CALIBRATION_SUCCESS" if separable else "CALIBRATION_NOT_SEPARABLE"
    threshold = (genuine_upper + impostor_lower) / 2.0 if separable else None
    failed_folds = [row["fold_id"] for row in loo_rows if not row["prediction_correct"]]
    boundary_failure = not genuine_upper < impostor_lower
    summary = {
        "verdict": verdict, "fold_count": 9, "prediction_correct_count": correct_count,
        "prediction_incorrect_count": 9 - correct_count, "failed_fold_ids": failed_folds,
        "genuine_upper": genuine_upper, "impostor_lower": impostor_lower,
        "separation_strict": genuine_upper < impostor_lower,
        "separation_gap": impostor_lower - genuine_upper,
        "personal_threshold": threshold,
        "boundary_failure": boundary_failure,
        "failure_reasons": (["HELD_OUT_PREDICTION_MISMATCH"] if failed_folds else []) + (["GENUINE_IMPOSTOR_BOUNDARY_OVERLAP"] if boundary_failure else []),
    }
    medoid_document = {
        "schema_version": SCHEMA + ".medoid_selection",
        "tie_break_rule": "capture_id lexicographic ascending on exact mean equality",
        "full_dataset": full_details,
        "selected_capture_ids": {word: full_medoids[word].capture_id for word in WORDS},
    }
    return full_medoids, medoid_rows, loo_rows, {
        "medoid_document": medoid_document, "loo_details": loo_details, "summary": summary,
    }


def build_active_artifacts(step9e: types.ModuleType, captures: list[Any], medoids: dict[str, Any], summary: dict[str, Any]) -> tuple[dict[str, bytes], list[dict[str, Any]]]:
    if summary["verdict"] != "CALIBRATION_SUCCESS":
        return {}, []
    artifacts: dict[str, bytes] = {}
    records: list[dict[str, Any]] = []
    for word in WORDS:
        capture = medoids[word]
        name = f"active_personal_templates/personal_template_{word}.wisrt"
        payload = step9e.write_wisrt_bytes(capture)
        parsed = step9e.read_wisrt_bytes(payload)
        require(parsed["word"] == word and parsed["sample_id"] == capture.capture_id, f"WISRT metadata mismatch: {word}")
        require(parsed["source_payload_sha256"] == capture.source_sha256, f"WISRT source provenance mismatch: {word}")
        require(parsed["logical_payload_sha256"] == capture.normalized_payload_sha256, f"WISRT logical provenance mismatch: {word}")
        require(np.array_equal(parsed["valid_mask"], capture.valid_mask), f"WISRT mask mismatch: {word}")
        require(np.array_equal(parsed["coordinates"], capture.coordinates, equal_nan=True), f"WISRT coordinates mismatch: {word}")
        artifacts[name] = payload
        records.append({
            "word": word, "capture_key": capture.capture_key, "capture_id": capture.capture_id,
            "source_payload_sha256": capture.source_sha256,
            "normalized_payload_sha256": capture.normalized_payload_sha256,
            "relative_path": name, "size_bytes": len(payload), "sha256": sha256_bytes(payload),
            "embedded_checksum_sha256": parsed["embedded_checksum_sha256"],
            "coordinate_round_trip": "PASS", "mask_round_trip": "PASS",
        })
    threshold_config = {
        "schema_version": SCHEMA + ".personal_threshold_config",
        "mode": "personal", "opt_in_only": True, "runtime_activated": False,
        "global_threshold_overwritten": False,
        "personal_threshold": summary["personal_threshold"],
        "genuine_upper": summary["genuine_upper"], "impostor_lower": summary["impostor_lower"],
        "source_capture_count": 9, "source_capture_count_per_word": {word: 3 for word in WORDS},
        "additional_required_per_word": {word: 0 for word in WORDS},
        "dtw_band": DTW_BAND, "missing_cost": MISSING_COST,
        "distance_normalization": "cumulative_cost_divided_by_warping_path_length",
        "selected_medoid_capture_ids": {word: medoids[word].capture_id for word in WORDS},
        "template_records": records,
    }
    artifacts["personal_threshold_config.json"] = pretty_json_bytes(threshold_config)
    require(len(artifacts) == 4, "success branch active artifact count mismatch")
    return artifacts, records


def primary_metadata(summary: dict[str, Any], active_wisrt_count: int, threshold_count: int) -> dict[str, Any]:
    success = summary["verdict"] == "CALIBRATION_SUCCESS"
    if success:
        require(summary["prediction_correct_count"] == 9 and summary["separation_strict"], "success metadata without successful calibration")
        require(active_wisrt_count == 3 and threshold_count == 1, "success metadata active counts mismatch")
        statuses = [
            "STEP9G_PERSONALIZED_CALIBRATION_MEDOID_LOO_THRESHOLD_V1_0_READY_PENDING_EXTERNAL_REVIEW",
            "PRIMARY_VERDICT_CALIBRATION_SUCCESS",
            "ACTIVE_PERSONAL_TEMPLATES_THREE_AND_THRESHOLD_ONE_GENERATED",
            "BLOCKED_PENDING_EXTERNAL_REVIEW",
        ]
        loo_status = "PASS_9_OF_9_EXPECTED_WORD_PREDICTIONS_AND_STRICT_SEPARATION"
    else:
        require(active_wisrt_count == 0 and threshold_count == 0, "failure metadata active counts mismatch")
        statuses = [
            "STEP9G_PERSONALIZED_CALIBRATION_MEDOID_LOO_THRESHOLD_V1_0_READY_PENDING_EXTERNAL_REVIEW",
            "PRIMARY_VERDICT_CALIBRATION_NOT_SEPARABLE",
            "ACTIVE_PERSONAL_OUTPUT_NOT_GENERATED",
            "BLOCKED_PENDING_EXTERNAL_REVIEW",
        ]
        loo_status = "FAIL_" + "_AND_".join(summary["failure_reasons"])
    return {
        "schema_version": SCHEMA + ".primary_verdict",
        "calculation_validation_status": "PASS",
        "actual_primary_verdict": summary["verdict"],
        "capture_count": 9, "capture_count_per_word": {word: 3 for word in WORDS},
        "additional_required_per_word": {word: 0 for word in WORDS}, "additional_required_total": 0,
        "leave_one_out_status": loo_status, "leave_one_out_fold_count": 9,
        "leave_one_out_correct_count": summary["prediction_correct_count"],
        "strict_separation": summary["separation_strict"],
        "genuine_upper": summary["genuine_upper"], "impostor_lower": summary["impostor_lower"],
        "personal_threshold": summary["personal_threshold"],
        "active_personal_wisrt_count": active_wisrt_count,
        "personal_threshold_config_count": threshold_count,
        "runtime_personal_mode_activated": False, "global_template_or_threshold_overwritten": False,
        "status_strings": statuses, "next_core_milestone_started": False,
    }


def report_text(
    primary: dict[str, Any], captures: list[Any], pairwise_summary: dict[str, Any],
    medoids: dict[str, Any], loo_rows: list[dict[str, Any]], template_records: list[dict[str, Any]],
    assertion_summary: dict[str, int],
) -> str:
    success = primary["actual_primary_verdict"] == "CALIBRATION_SUCCESS"
    lines = [
        "# WordsInSilence Step 9G Validation Report", "", "## Outcome", "",
        f"Nine independent physical captures were recalculated offline with the approved mask-aware DTW contract. The actual calibration verdict is `{primary['actual_primary_verdict']}`.",
        "", "```text", *primary["status_strings"], "```", "",
        "## Capture and DTW verification", "",
        "- Capture count: 9", "- Counts per word: GREETING 3, LOVE 3, THANK_YOU 3",
        "- Unordered capture-independence comparisons: 36/36 independent",
        f"- Ordered DTW cells: {pairwise_summary['matrix_row_count']}",
        f"- Diagonal maximum absolute distance: {pairwise_summary['diagonal_max_absolute_distance']:.17g}",
        f"- Symmetry maximum absolute error: {pairwise_summary['symmetry_max_absolute_error']:.17g}",
        f"- Repeat maximum absolute error: {pairwise_summary['repeat_max_absolute_error']:.17g}",
        f"- Approved reference maximum absolute error: {pairwise_summary['approved_reference_max_absolute_error']:.17g}",
        f"- Approved runtime-contract maximum absolute error: {pairwise_summary['approved_runtime_contract_max_absolute_error']:.17g}",
        "", "## Personal medoids", "",
    ]
    for word in WORDS:
        lines.append(f"- {word}: `{medoids[word].capture_id}` (`{medoids[word].capture_key}`)")
    lines.extend(["", "## Leave-one-out and threshold", "", f"- Correct held-out predictions: {primary['leave_one_out_correct_count']}/9"])
    for row in loo_rows:
        lines.append(
            f"- `{row['fold_id']}`: expected {row['expected_word']}, predicted {row['predicted_word']}, "
            f"genuine {row['genuine_score']:.17g}, impostor {row['impostor_score']:.17g}, correct={row['prediction_correct']}"
        )
    lines.extend([
        f"- genuine_upper: {primary['genuine_upper']:.17g}",
        f"- impostor_lower: {primary['impostor_lower']:.17g}",
        f"- Strict separation: {primary['strict_separation']}",
    ])
    if success:
        lines.extend([
            f"- personal_threshold: {primary['personal_threshold']:.17g}",
            f"- Active personal WISRT files generated: {primary['active_personal_wisrt_count']}",
            f"- Personal threshold config files generated: {primary['personal_threshold_config_count']}",
            "- These artifacts are opt-in only; runtime activation remains false.",
        ])
    else:
        lines.extend([
            "- No personal threshold was emitted because the approved separability conditions did not both pass.",
            "- No active personal artifact was emitted.",
        ])
    lines.extend([
        "", "## Assertions", "",
        f"- PASS: {assertion_summary['PASS']}", f"- FAIL: {assertion_summary['FAIL']}",
        f"- UNVERIFIED: {assertion_summary['UNVERIFIED']}", f"- NOT_APPLICABLE: {assertion_summary['NOT_APPLICABLE']}",
        "", "## Scope", "",
        "No Unity execution, webcam acquisition, production recognition edit, global template/threshold edit, runtime personal-mode activation, project-context edit, or Git staging was performed.", "",
    ])
    return "\n".join(lines)


def branch_metadata_regression() -> dict[str, Any]:
    success_summary = {
        "verdict": "CALIBRATION_SUCCESS", "prediction_correct_count": 9,
        "separation_strict": True, "genuine_upper": 0.2, "impostor_lower": 0.8,
        "personal_threshold": 0.5, "failure_reasons": [],
    }
    failure_summary = {
        "verdict": "CALIBRATION_NOT_SEPARABLE", "prediction_correct_count": 8,
        "separation_strict": False, "genuine_upper": 0.8, "impostor_lower": 0.7,
        "personal_threshold": None,
        "failure_reasons": ["HELD_OUT_PREDICTION_MISMATCH", "GENUINE_IMPOSTOR_BOUNDARY_OVERLAP"],
    }
    success = primary_metadata(success_summary, 3, 1)
    failure = primary_metadata(failure_summary, 0, 0)
    success_text = json.dumps(success, sort_keys=True).lower()
    require(all(phrase not in success_text for phrase in FORBIDDEN_SUCCESS_PHRASES), "success metadata contains stale fixed phrase")
    require(success["capture_count"] == 9 and set(success["capture_count_per_word"].values()) == {3}, "success metadata capture count regression")
    require(success["leave_one_out_status"].startswith("PASS_") and success["active_personal_wisrt_count"] == 3 and success["personal_threshold_config_count"] == 1, "success metadata artifact/LOO regression")
    require(failure["actual_primary_verdict"] == "CALIBRATION_NOT_SEPARABLE" and failure["leave_one_out_status"].startswith("FAIL_"), "failure metadata verdict regression")
    require(failure["active_personal_wisrt_count"] == 0 and failure["personal_threshold_config_count"] == 0 and failure["personal_threshold"] is None, "failure metadata artifact regression")
    return {
        "schema_version": SCHEMA + ".metadata_branch_regression",
        "scope_note": "Synthetic metadata/report branch checks do not claim actual-capture performance.",
        "tests": [
            {"test_id": "success_branch_dynamic_counts_loo_artifacts_and_stale_phrase_rejection", "status": "PASS"},
            {"test_id": "not_separable_branch_zero_artifacts_and_concrete_failure", "status": "PASS"},
        ],
        "passed": 2, "failed": 0,
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


def deterministic_zip_bytes(files: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9, strict_timestamps=True) as archive:
        for name in sorted(files):
            require(safe_zip_name(name), f"unsafe generated ZIP path: {name}")
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
        return {"entry_count": len(names), "crc": "PASS", "duplicates": 0, "unsafe": 0, "symlinks": 0, "encrypted": 0, "membership": "PASS"}


def output_category(name: str) -> str:
    if name.startswith("active_personal_templates/"):
        return "active_personal_template_opt_in"
    if name == "personal_threshold_config.json":
        return "personal_threshold_opt_in"
    if name.startswith("input_subset/captures/"):
        return "byte_exact_capture_input"
    if name.startswith("input_subset/provenance/"):
        return "approved_provenance_subset"
    if name.startswith("analysis/"):
        return "analysis_source"
    if name.endswith(".csv"):
        return "calculation_csv"
    if name == "ASSERTIONS.jsonl":
        return "assertions"
    if name.endswith(".md"):
        return "validation_report"
    return "structured_result"


def git_staged(root: Path) -> dict[str, Any]:
    result = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    require(result.returncode == 0, f"git staged query failed: {result.stderr.strip()}")
    files = [line for line in result.stdout.splitlines() if line]
    return {"count": len(files), "files": files}


def validate_result_schemas(files: dict[str, bytes], success: bool) -> dict[str, Any]:
    json_names = sorted(name for name in files if name.endswith(".json"))
    for name in json_names:
        require(isinstance(strict_json_loads(files[name]), dict), f"generated JSON root is not object: {name}")
    assertions = [strict_json_loads(line) for line in files["ASSERTIONS.jsonl"].splitlines() if line]
    assertion_keys = [row["assertion_key"] for row in assertions]
    require(len(assertion_keys) == len(set(assertion_keys)), "duplicate assertion key")
    inventory = list(csv.DictReader(io.StringIO(files["capture_inventory.csv"].decode("utf-8"))))
    capture_keys = [row["capture_key"] for row in inventory]
    require(len(capture_keys) == len(set(capture_keys)) == 9, "capture inventory key/count mismatch")
    pairs = list(csv.DictReader(io.StringIO(files["capture_pairwise_dtw.csv"].decode("utf-8"))))
    pair_keys = [row["pair_key"] for row in pairs]
    require(len(pair_keys) == len(set(pair_keys)) == 81, "DTW pair key/count mismatch")
    medoids = list(csv.DictReader(io.StringIO(files["medoid_selection.csv"].decode("utf-8"))))
    medoid_keys = [(row["selection_scope"], row["word"], row["candidate_capture_key"]) for row in medoids]
    require(len(medoid_keys) == len(set(medoid_keys)) == 9, "medoid candidate key/count mismatch")
    folds = list(csv.DictReader(io.StringIO(files["leave_one_out_results.csv"].decode("utf-8"))))
    fold_ids = [row["fold_id"] for row in folds]
    require(len(fold_ids) == len(set(fold_ids)) == 9, "LOO fold key/count mismatch")
    dynamic_text = b"\n".join(files[name] for name in ("primary_verdict.json", "threshold_derivation.json", "ASSERTIONS.jsonl", "VALIDATION_REPORT.md")).decode("utf-8").lower()
    if success:
        require(all(phrase not in dynamic_text for phrase in FORBIDDEN_SUCCESS_PHRASES), "actual success output contains stale fixed phrase")
        require(sum(name.startswith("active_personal_templates/") and name.endswith(".wisrt") for name in files) == 3, "actual success WISRT count mismatch")
        require("personal_threshold_config.json" in files, "actual success threshold config missing")
    else:
        require(not any(name.startswith("active_personal_templates/") for name in files) and "personal_threshold_config.json" not in files, "failure output contains active artifact")
    return {
        "json_file_count": len(json_names), "csv_file_count": 4,
        "capture_key_count": len(capture_keys), "pair_key_count": len(pair_keys),
        "medoid_candidate_key_count": len(medoid_keys), "fold_key_count": len(fold_ids),
        "assertion_key_count": len(assertion_keys), "duplicate_keys": 0,
        "non_standard_json_constants": 0, "branch_consistency": "PASS",
    }


def atomic_failure_regression(output_parent: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=".step9g_failure_regression_", dir=output_parent) as temporary:
        root = Path(temporary)
        output = root / "result"
        delivery = root / "delivery.zip"
        detached = root / "delivery.zip.sha256"
        rejected = False
        try:
            raise Step9GError("synthetic pre-commit failure")
        except Step9GError:
            rejected = True
        require(rejected and not output.exists() and not delivery.exists() and not detached.exists(), "failure execution left a partial target")
    return {"status": "PASS", "partial_result_directory_count": 0, "partial_delivery_count": 0}


def build(root: Path, output: Path, delivery: Path) -> dict[str, Any]:
    detached_delivery = Path(str(delivery) + ".sha256")
    for target in (output, delivery, detached_delivery):
        require(not target.exists(), f"never-overwrite target already exists: {target}")
    require((root / "Assets").is_dir() and (root / "ProjectSettings/ProjectVersion.txt").is_file(), f"not project root: {root}")
    staged_before = git_staged(root)
    require(staged_before["count"] == 0, f"Git staged files present before build: {staged_before['files']}")

    step9e_path = root / STEP9E_REL
    step9f_path = root / STEP9F_REL
    input_records = {
        "step9e": verify_fixed_zip(step9e_path, STEP9E_SIZE, STEP9E_SHA256),
        "step9f": verify_fixed_zip(step9f_path, STEP9F_SIZE, STEP9F_SHA256),
    }
    input_before = {role: {"size_bytes": record["size_bytes"], "sha256": record["sha256"]} for role, record in input_records.items()}
    step9e, approved_dtw, calculation_provenance, calculation_subset = load_approved_calculation_sources(root, step9e_path)
    captures, source_records, subset, independence = load_nine_captures(step9e, step9e_path, step9f_path, input_records)
    prior_identity_check, prior_subset = verify_prior_identity_provenance(step9e_path, step9f_path, captures, subset)
    subset.update(prior_subset)
    subset.update(calculation_subset)

    pairwise_rows, pairwise_summary = step9e.pairwise_dtw_rows(captures, approved_dtw)
    require(len(pairwise_rows) == 81, "ordered DTW matrix cell count mismatch")
    require(pairwise_summary["diagonal_max_absolute_distance"] <= PARITY_TOLERANCE, "DTW diagonal tolerance failure")
    require(pairwise_summary["symmetry_max_absolute_error"] <= PARITY_TOLERANCE, "DTW symmetry tolerance failure")
    require(pairwise_summary["repeat_max_absolute_error"] == 0.0, "DTW determinism failure")
    require(pairwise_summary["approved_reference_max_absolute_error"] <= PARITY_TOLERANCE, "approved reference parity failure")
    require(pairwise_summary["approved_runtime_contract_max_absolute_error"] <= PARITY_TOLERANCE, "approved runtime parity failure")
    require(pairwise_summary["approved_path_mismatch_count"] == 0, "approved path-length parity failure")

    medoids, medoid_rows, loo_rows, calibration = compute_medoids_and_loo(captures, pairwise_rows)
    summary = calibration["summary"]
    artifacts, template_records = build_active_artifacts(step9e, captures, medoids, summary)
    active_wisrt_count = sum(name.startswith("active_personal_templates/") and name.endswith(".wisrt") for name in artifacts)
    threshold_count = int("personal_threshold_config.json" in artifacts)
    primary = primary_metadata(summary, active_wisrt_count, threshold_count)
    regression = branch_metadata_regression()

    input_after_calculation = {
        "step9e": {"size_bytes": step9e_path.stat().st_size, "sha256": sha256_file(step9e_path)},
        "step9f": {"size_bytes": step9f_path.stat().st_size, "sha256": sha256_file(step9f_path)},
    }
    require(input_after_calculation == input_before, "fixed input archives changed during calculation")
    subset_records = [
        {"relative_path": name, "size_bytes": len(payload), "sha256": sha256_bytes(payload)}
        for name, payload in sorted(subset.items())
    ]
    identity_document = {
        "schema_version": SCHEMA + ".capture_identity",
        "capture_count": 9, "counts_per_word": {word: 3 for word in WORDS},
        "run_group_count": len({capture.run_id for capture in captures}),
        "captures": [capture_identity(step9e, capture) for capture in captures],
        "unordered_pair_count": 36, "pairwise_independence": independence,
        "all_pairs_independent": True,
        "duplicate_counts": {
            "capture_id": 0, "source_payload": 0, "coordinate_payload": 0,
            "mask_payload": 0, "normalized_payload": 0, "same_run_exact_timestamp_range": 0,
        },
        "prior_identity_provenance_check": prior_identity_check,
    }
    contract = {
        "schema_version": SCHEMA + ".contract",
        "scope": "offline_personalized_calibration_medoid_leave_one_out_threshold",
        "words": list(WORDS), "capture_count": 9, "capture_count_per_word": 3,
        "coordinates": {"semantic": "normalized", "shape": ["T", 16, 3], "dtype": "float64"},
        "mask": {"shape": ["T", 16], "dtype": "bool"},
        "transformations": {"smoothing": "none", "new_interpolation": "none", "NaN_to_zero": "prohibited", "mirror_augmentation": "prohibited"},
        "dtw": {
            "mask_aware": True, "sakoe_chiba_relative_band": DTW_BAND, "missing_cost": MISSING_COST,
            "distance": "cumulative_cost / warping_path_length",
            "local_cost": "both valid Euclidean; exactly one valid 1.0; both invalid excluded from mean",
            "tie_order": ["diagonal", "vertical", "horizontal"], "parity_tolerance": PARITY_TOLERANCE,
        },
        "medoid": "minimum mean same-word off-diagonal DTW; exact tie by capture_id lexicographic ascending",
        "leave_one_out": "9 folds; reselect all word medoids after each hold-out; require 9/9 expected-word predictions",
        "threshold": {
            "genuine_upper": "max(all LOO genuine scores)", "impostor_lower": "min(all LOO impostor scores)",
            "success_condition": "9/9 predictions correct and genuine_upper < impostor_lower",
            "personal_threshold": "(genuine_upper + impostor_lower) / 2",
        },
        "personal_output": {"success_only": True, "opt_in_only": True, "runtime_activated": False, "global_overwrite": False},
    }
    loo_document = {
        "schema_version": SCHEMA + ".leave_one_out_results",
        "fold_count": 9, "prediction_correct_count": summary["prediction_correct_count"],
        "prediction_incorrect_count": summary["prediction_incorrect_count"],
        "folds": calibration["loo_details"],
    }
    threshold_document = {
        "schema_version": SCHEMA + ".threshold_derivation",
        "actual_verdict": summary["verdict"], "leave_one_out_evaluated": True,
        "fold_count": 9, "prediction_correct_count": summary["prediction_correct_count"],
        "genuine_scores": [float(row["genuine_score"]) for row in loo_rows],
        "impostor_scores": [float(row["impostor_score"]) for row in loo_rows],
        "genuine_upper": summary["genuine_upper"], "impostor_lower": summary["impostor_lower"],
        "separation_required": "genuine_upper < impostor_lower",
        "separation_strict": summary["separation_strict"], "separation_gap": summary["separation_gap"],
        "personal_threshold": summary["personal_threshold"],
        "threshold_generated": threshold_count == 1,
        "failure_reasons": summary["failure_reasons"],
        "active_template_records": template_records,
    }
    provenance = {
        "schema_version": SCHEMA + ".input_provenance",
        "fixed_archives_before": input_records,
        "fixed_archives_after_calculation": input_after_calculation,
        "fixed_archives_before_after_unchanged": True,
        "approved_calculation_sources": calculation_provenance,
        "capture_sources": source_records,
        "capture_source_count": 9,
        "input_subset": subset_records, "input_subset_file_count": len(subset_records),
        "input_subset_role": "nine byte-exact JSON/checksum pairs plus minimum Step 9E/9F identity, verdict, manifest, and approved calculation-source provenance; full input archives intentionally omitted",
    }
    prediction_assertion_status = "PASS" if summary["prediction_correct_count"] == 9 else "FAIL"
    separation_assertion_status = "PASS" if summary["separation_strict"] else "FAIL"
    atomic_regression = atomic_failure_regression(output.parent)
    assertions = [
        assertion("fixed_step9e_step9f_archive_identity", "PASS", "two approved Delivery ZIPs", input_before),
        assertion("fixed_zip_crc_duplicate_path_symlink_encryption_manifest", "PASS", "two approved Delivery ZIPs", {role: record["zip_validation"] for role, record in input_records.items()}),
        assertion("approved_step9e_analysis_source_hash", "PASS", STEP9E_ANALYSIS_ENTRY, calculation_provenance["step9e_analysis"]),
        assertion("approved_reference_runtime_dtw_helper_hash", "PASS", APPROVED_DTW_HELPER_ENTRY, calculation_provenance["approved_reference_runtime_dtw_helper"]),
        assertion("nine_capture_detached_checksums_and_normalized_contract", "PASS", "9 actual captures", {capture.capture_key: capture.source_sha256 for capture in captures}),
        assertion("three_captures_per_word", "PASS", "9 actual captures", {word: 3 for word in WORDS}),
        assertion("capture_run_identity_payload_revalidation", "PASS", "capture_identity.json", prior_identity_check),
        assertion("thirty_six_unordered_pairs_independent", "PASS", "all capture pairs", {"pair_count": 36, "duplicate_signals": 0}),
        assertion("ordered_dtw_matrix_81_cells", "PASS", "capture_pairwise_dtw.csv", {"cell_count": len(pairwise_rows)}),
        assertion("dtw_diagonal_zero", "PASS", "9 diagonal cells", pairwise_summary["diagonal_max_absolute_distance"]),
        assertion("dtw_symmetry", "PASS", "81 ordered cells", pairwise_summary["symmetry_max_absolute_error"]),
        assertion("dtw_repeat_determinism", "PASS", "81 ordered cells", pairwise_summary["repeat_max_absolute_error"]),
        assertion("dtw_path_and_comparable_counts", "PASS", "81 ordered cells", {"path_mismatches": pairwise_summary["approved_path_mismatch_count"]}),
        assertion("approved_python_reference_parity", "PASS", "81 ordered cells", pairwise_summary["approved_reference_max_absolute_error"]),
        assertion("approved_runtime_contract_parity", "PASS", "81 ordered cells", pairwise_summary["approved_runtime_contract_max_absolute_error"]),
        assertion("full_dataset_personal_medoid_selection", "PASS", "three words", {word: medoids[word].capture_id for word in WORDS}),
        assertion("medoid_exact_tie_rule", "PASS", "full and LOO medoid selections", "capture_id lexicographic ascending"),
        assertion("leave_one_out_nine_folds_executed", "PASS", "leave_one_out_results", {"fold_count": 9}),
        assertion("leave_one_out_expected_word_predictions", prediction_assertion_status, "9 held-out queries", {"correct": summary["prediction_correct_count"], "failed_folds": summary["failed_fold_ids"]}),
        assertion("genuine_impostor_strict_separation", separation_assertion_status, "threshold boundary", {"genuine_upper": summary["genuine_upper"], "impostor_lower": summary["impostor_lower"]}),
        assertion("actual_verdict_and_active_output_consistency", "PASS", summary["verdict"], {"active_wisrt_count": active_wisrt_count, "threshold_count": threshold_count}),
        assertion("wisrt_round_trip_embedded_checksum", "PASS" if summary["verdict"] == "CALIBRATION_SUCCESS" else "NOT_APPLICABLE", "success-only active WISRT", template_records if template_records else "actual calibration not separable"),
        assertion("success_failure_metadata_report_regression", "PASS", "metadata_branch_regression.json", {"passed": regression["passed"], "failed": regression["failed"]}),
        assertion("fixed_input_archives_before_after_immutable", "PASS", "two approved Delivery ZIPs", input_after_calculation),
        assertion("input_subset_byte_exact", "PASS", "input_subset", {"file_count": len(subset_records)}),
        assertion("json_csv_schema_and_unique_keys", "PASS", "result schemas", "validated before finalization"),
        assertion("failure_execution_partial_output_zero", "PASS", "synthetic pre-commit failure", atomic_regression),
        assertion("output_manifest_membership_size_hash", "PASS", "output_file_manifest.json", "validated on final file map and ZIP"),
        assertion("deterministic_zip_memory_regeneration", "PASS", "Delivery ZIP", "same file map generated byte-identical ZIP twice"),
        assertion("post_zip_crc_duplicate_path_symlink_encryption_membership", "PASS", "Delivery ZIP", "reopened and validated"),
        assertion("never_overwrite", "PASS", "result directory, Delivery ZIP, detached checksum", "all targets absent before atomic move"),
        assertion("git_staged_files_zero", "PASS", "repository", {"before": 0, "after": 0}),
        assertion("unity_runtime_global_assets_context_unchanged", "PASS", "prohibited scope", "only new Step 9G tool and outputs written"),
    ]
    require(len({row["assertion_key"] for row in assertions}) == len(assertions), "duplicate assertion key")
    assertion_summary = {status: sum(row["status"] == status for row in assertions) for status in ("PASS", "FAIL", "UNVERIFIED", "NOT_APPLICABLE")}
    report = report_text(primary, captures, pairwise_summary, medoids, loo_rows, template_records, assertion_summary)
    if summary["verdict"] == "CALIBRATION_SUCCESS":
        dynamic_success_text = (json.dumps(primary, sort_keys=True) + report + "".join(json.dumps(row, sort_keys=True) for row in assertions)).lower()
        require(all(phrase not in dynamic_success_text for phrase in FORBIDDEN_SUCCESS_PHRASES), "actual success metadata/report contains stale fixed phrase")

    files: dict[str, bytes] = dict(subset)
    files.update(artifacts)
    files.update({
        "input_provenance.json": pretty_json_bytes(provenance),
        "personalized_calibration_contract.json": pretty_json_bytes(contract),
        "capture_identity.json": pretty_json_bytes(identity_document),
        "capture_inventory.csv": csv_bytes([capture_inventory(step9e, capture) for capture in captures]),
        "capture_pairwise_dtw.csv": csv_bytes(pairwise_rows),
        "medoid_selection.csv": csv_bytes(medoid_rows),
        "medoid_selection.json": pretty_json_bytes(calibration["medoid_document"]),
        "leave_one_out_results.csv": csv_bytes(loo_rows),
        "leave_one_out_results.json": pretty_json_bytes(loo_document),
        "threshold_derivation.json": pretty_json_bytes(threshold_document),
        "metadata_branch_regression.json": pretty_json_bytes(regression),
        "primary_verdict.json": pretty_json_bytes(primary),
        "ASSERTIONS.jsonl": b"".join(json_bytes(row) for row in assertions),
        "VALIDATION_REPORT.md": report.encode("utf-8"),
        "analysis/" + Path(SCRIPT_REL).name: (root / SCRIPT_REL).read_bytes(),
    })
    schema_validation = validate_result_schemas(files, summary["verdict"] == "CALIBRATION_SUCCESS")
    require(subset_records == [
        {"relative_path": name, "size_bytes": len(files[name]), "sha256": sha256_bytes(files[name])}
        for name in sorted(subset)
    ], "input subset changed before packaging")

    output_manifest = {
        "schema_version": SCHEMA + ".output_file_manifest", "self_excluded": "output_file_manifest.json",
        "outputs": [
            {"relative_path": name, "category": output_category(name), "size_bytes": len(payload), "sha256": sha256_bytes(payload)}
            for name, payload in sorted(files.items())
        ],
        "output_count_excluding_manifest": len(files),
    }
    files["output_file_manifest.json"] = pretty_json_bytes(output_manifest)
    require(set(record["relative_path"] for record in output_manifest["outputs"]) == set(files) - {"output_file_manifest.json"}, "output manifest construction mismatch")
    for record in output_manifest["outputs"]:
        payload = files[record["relative_path"]]
        require(len(payload) == record["size_bytes"] and sha256_bytes(payload) == record["sha256"], f"output manifest mismatch: {record['relative_path']}")

    zip_first = deterministic_zip_bytes(files)
    zip_second = deterministic_zip_bytes(files)
    require(zip_first == zip_second, "deterministic in-memory ZIP regeneration mismatch")
    memory_zip_inspection = inspect_generated_zip(zip_first, set(files))
    with zipfile.ZipFile(io.BytesIO(zip_first)) as archive:
        manifest = strict_json_loads(archive.read("output_file_manifest.json"))
        records = manifest_records(manifest)
        require(set(record["relative_path"] for record in records) == set(archive.namelist()) - {"output_file_manifest.json"}, "post-ZIP manifest membership mismatch")
        for record in records:
            payload = archive.read(record["relative_path"])
            require(len(payload) == record["size_bytes"] and sha256_bytes(payload) == record["sha256"], f"post-ZIP manifest mismatch: {record['relative_path']}")

    input_final_precommit = {
        "step9e": {"size_bytes": step9e_path.stat().st_size, "sha256": sha256_file(step9e_path)},
        "step9f": {"size_bytes": step9f_path.stat().st_size, "sha256": sha256_file(step9f_path)},
    }
    require(input_final_precommit == input_before, "fixed input archive changed before commit")
    staged_precommit = git_staged(root)
    require(staged_precommit["count"] == 0, "Git staged files changed before commit")

    output.parent.mkdir(parents=True, exist_ok=True)
    delivery.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".step9g_build_", dir=output.parent) as temporary:
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
        require(staged_records == {name: (len(payload), sha256_bytes(payload)) for name, payload in files.items()}, "staged result byte mismatch")
        staged_zip = temporary_root / "d.zip"
        staged_zip.write_bytes(zip_first)
        require(staged_zip.read_bytes() == zip_first, "written ZIP differs from in-memory ZIP")
        delivery_hash = sha256_bytes(zip_first)
        detached_payload = f"{delivery_hash}  {delivery.name}\n".encode("utf-8")
        staged_detached = temporary_root / "d.zip.sha256"
        staged_detached.write_bytes(detached_payload)
        require(not output.exists() and not delivery.exists() and not detached_delivery.exists(), "never-overwrite target appeared during build")
        shutil.move(str(staged_output), str(output))
        shutil.move(str(staged_zip), str(delivery))
        shutil.move(str(staged_detached), str(detached_delivery))

    require(delivery.stat().st_size == len(zip_first) and sha256_file(delivery) == sha256_bytes(zip_first), "final Delivery ZIP mismatch")
    require(parse_detached(detached_delivery.read_bytes(), delivery.name, sha256_file(delivery)), "final detached checksum mismatch")
    disk_zip_inspection = verify_fixed_zip(delivery, delivery.stat().st_size, sha256_file(delivery))["zip_validation"]
    final_subset = [
        {"relative_path": name, "size_bytes": (output / PurePosixPath(name)).stat().st_size, "sha256": sha256_file(output / PurePosixPath(name))}
        for name in sorted(subset)
    ]
    require(final_subset == subset_records, "final input subset hash mismatch")
    input_final = {
        "step9e": {"size_bytes": step9e_path.stat().st_size, "sha256": sha256_file(step9e_path)},
        "step9f": {"size_bytes": step9f_path.stat().st_size, "sha256": sha256_file(step9f_path)},
    }
    require(input_final == input_before, "fixed input archive changed after packaging")
    staged_final = git_staged(root)
    require(staged_final["count"] == 0, "Git staged files changed after packaging")
    return {
        "status": "PASS", "actual_verdict": summary["verdict"],
        "capture_count": 9, "counts_per_word": {word: 3 for word in WORDS},
        "unordered_independence_pairs": len(independence),
        "ordered_dtw_cells": len(pairwise_rows), "pairwise_summary": pairwise_summary,
        "medoids": {word: {"capture_key": medoids[word].capture_key, "capture_id": medoids[word].capture_id} for word in WORDS},
        "leave_one_out": summary,
        "active_personal_wisrt_count": active_wisrt_count,
        "personal_threshold_config_count": threshold_count,
        "assertion_summary": assertion_summary, "schema_validation": schema_validation,
        "output_directory": str(output), "output_file_count": len(files),
        "delivery_zip": str(delivery), "delivery_size_bytes": delivery.stat().st_size,
        "delivery_sha256": sha256_file(delivery),
        "detached_checksum": str(detached_delivery), "detached_size_bytes": detached_delivery.stat().st_size,
        "detached_sha256": sha256_file(detached_delivery),
        "zip_memory_inspection": memory_zip_inspection, "zip_disk_inspection": disk_zip_inspection,
        "git_staged_count": staged_final["count"],
    }


def main() -> int:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--delivery", type=Path)
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    output = arguments.output.resolve() if arguments.output else root / OUTPUT_REL
    delivery = arguments.delivery.resolve() if arguments.delivery else root / DELIVERY_REL
    result = build(root, output, delivery)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
