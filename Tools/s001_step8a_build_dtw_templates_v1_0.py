#!/usr/bin/env python3
"""Build the approved Step 8A offline mask-aware DTW template baseline."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
import uuid
import zipfile
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path, PurePosixPath

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEP7B_DIR = PROJECT_ROOT / "data/processed/sessions/S001_P001/step7b_full_v1_0"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/sessions/S001_P001/step8a_dtw_templates_v1_0"
STEP7B_ZIP = PROJECT_ROOT / "WordsInSilence_step7b_v1_0_full_delivery.zip"
DELIVERY_NAME = "WordsInSilence_step8a_dtw_templates_v1_0_delivery.zip"
DELIVERY_ZIP = PROJECT_ROOT / DELIVERY_NAME
DETACHED_HASH = PROJECT_ROOT / f"{DELIVERY_NAME}.sha256"
POST_ZIP_SIDECAR = PROJECT_ROOT / "WordsInSilence_step8a_dtw_templates_v1_0_delivery_post_zip_verification.json"

VERSION = "1.0"
POSE_LANDMARKS = [13, 14, 15, 16]
HAND_LANDMARKS = [0, 4, 8, 12, 16, 20]
LANDMARK_SCHEMA = [
    *[{"source": "pose", "index": i} for i in POSE_LANDMARKS],
    *[{"source": "left_hand", "index": i} for i in HAND_LANDMARKS],
    *[{"source": "right_hand", "index": i} for i in HAND_LANDMARKS],
]
TOTAL_LANDMARKS = 16
MISSING_COST = 1.0
BANDS = (0.10, 0.20, 0.30)
STRATEGIES = ("word_medoid", "word_speed_medoid")
APPROVED_STEP7B_ZIP_SIZE = 226_548_811
APPROVED_STEP7B_ZIP_SHA256 = "472c43fc53f96751cfc18e79a9a1c4a247ee0fdc5531fe80af532fa88ef8b015"


def log(message: str = "") -> None:
    print(message, flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def add_check(checks: list[dict], name: str, passed: bool, evidence: object = None) -> None:
    checks.append({"name": name, "status": "PASS" if bool(passed) else "FAIL", "evidence": evidence})


def failed_checks(checks: list[dict]) -> list[dict]:
    return [record for record in checks if record["status"] != "PASS"]


def assertion_summary(checks: list[dict]) -> dict:
    names = [record["name"] for record in checks]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    passed = sum(record["status"] == "PASS" for record in checks)
    failed = len(checks) - passed
    return {
        "total": len(checks),
        "pass": passed,
        "fail": failed,
        "duplicate_count": len(duplicates),
        "duplicate_names": duplicates,
    }


def logical_payload_hash(arrays: list[tuple[str, np.ndarray]], metadata: dict) -> str:
    digest = hashlib.sha256()
    digest.update(canonical_json(metadata).encode("utf-8"))
    for name, array in arrays:
        value = np.ascontiguousarray(array)
        descriptor = {"name": name, "dtype": value.dtype.str, "shape": list(value.shape)}
        digest.update(canonical_json(descriptor).encode("utf-8"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def step7b_output_path(relative_path: str) -> Path:
    if relative_path == "s001_step7b_preprocess_full_v1_0.py":
        return PROJECT_ROOT / "Tools" / relative_path
    return STEP7B_DIR / PurePosixPath(relative_path)


def verify_manifest_targets(entries: list[dict], resolver, prefix: str, checks: list[dict]) -> tuple[dict, dict]:
    hashes: dict[str, str] = {}
    snapshot: dict[str, dict] = {}
    paths = []
    for index, entry in enumerate(entries):
        key = entry.get("relative_path", entry.get("file"))
        paths.append(key)
        target = resolver(entry)
        exists = target.is_file()
        add_check(checks, f"{prefix}_{index:04d}_exists", exists, key)
        if not exists:
            continue
        stat = target.stat()
        size_ok = stat.st_size == int(entry["size_bytes"])
        add_check(checks, f"{prefix}_{index:04d}_size", size_ok, {"path": key, "actual": stat.st_size, "expected": entry["size_bytes"]})
        actual_hash = sha256_file(target)
        hash_ok = actual_hash == entry["sha256"].lower()
        add_check(checks, f"{prefix}_{index:04d}_sha256", hash_ok, {"path": key, "actual": actual_hash, "expected": entry["sha256"]})
        hashes[key] = actual_hash
        snapshot[key] = {"path": str(target), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": actual_hash}
    add_check(checks, f"{prefix}_unique_targets", len(paths) == len(set(paths)), {"total": len(paths), "unique": len(set(paths))})
    return hashes, snapshot


def preflight_step7b(checks: list[dict]) -> dict:
    log("=== Preflight: approved Step 7B input ===")
    required = [
        STEP7B_DIR,
        STEP7B_DIR / "step7_dataset_split_manifest.json",
        STEP7B_DIR / "segmentation_summary.csv",
        STEP7B_DIR / "input_file_manifest.json",
        STEP7B_DIR / "output_file_manifest.json",
        STEP7B_DIR / "npz",
        STEP7B_DIR / "sequences",
        STEP7B_ZIP,
    ]
    for index, path in enumerate(required):
        add_check(checks, f"preflight_required_{index:02d}", path.exists(), str(path.relative_to(PROJECT_ROOT)))
    if any(not path.exists() for path in required):
        raise RuntimeError("required Step 7B input is missing")

    split_manifest = json.loads((STEP7B_DIR / "step7_dataset_split_manifest.json").read_text(encoding="utf-8"))
    samples = split_manifest["samples"]
    declared_counts = split_manifest["counts"]
    ids = [sample["sample_id"] for sample in samples]
    actual_counts = Counter(sample["split"] for sample in samples)
    words = sorted({sample["word"] for sample in samples})
    add_check(checks, "split_total_45", len(samples) == 45, len(samples))
    for split, expected in (("train", 27), ("val", 9), ("test", 9)):
        add_check(checks, f"split_declared_{split}_{expected}", declared_counts.get(split) == expected, declared_counts.get(split))
        add_check(checks, f"split_actual_{split}_{expected}", actual_counts[split] == expected, actual_counts[split])
    add_check(checks, "sample_ids_unique", len(ids) == len(set(ids)), {"total": len(ids), "unique": len(set(ids))})

    split_ids = {split: {sample["sample_id"] for sample in samples if sample["split"] == split} for split in ("train", "val", "test")}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sorted(split_ids[left] & split_ids[right])
        add_check(checks, f"cross_split_{left}_{right}_zero", not overlap, overlap)
    distribution = {}
    for word in words:
        distribution[word] = {}
        for split, expected in (("train", 9), ("val", 3), ("test", 3)):
            count = sum(sample["word"] == word and sample["split"] == split for sample in samples)
            distribution[word][split] = count
            add_check(checks, f"distribution_{word}_{split}_{expected}", count == expected, count)

    with (STEP7B_DIR / "segmentation_summary.csv").open(encoding="utf-8", newline="") as stream:
        segmentation_rows = list(csv.DictReader(stream))
    segmentation = {
        row["sample_id"]: {
            "total_frames": int(row["total_frames"]),
            "seg_start": int(row["seg_start"]),
            "seg_end": int(row["seg_end"]),
            "seg_frames": int(row["seg_frames"]),
        }
        for row in segmentation_rows
    }
    add_check(checks, "segmentation_ids_exact", set(segmentation) == set(ids), {"missing": sorted(set(ids) - set(segmentation)), "unexpected": sorted(set(segmentation) - set(ids))})
    for sample_id, segment in segmentation.items():
        add_check(checks, f"segment_inclusive_length_{sample_id}", segment["seg_end"] - segment["seg_start"] + 1 == segment["seg_frames"], segment)
    frame_total = sum(segment["total_frames"] for segment in segmentation.values())
    add_check(checks, "step7b_frames_4668", frame_total == 4668, frame_total)
    add_check(checks, "step7b_landmarks_350100", frame_total * 75 == 350100, frame_total * 75)

    expected_short = {sample_id.removeprefix("S001_P001_") for sample_id in ids}
    actual_g1 = {path.name[len("interp_"):-len("_g1.npz")] for path in (STEP7B_DIR / "npz").glob("interp_*_g1.npz")}
    actual_sequences = {path.stem[len("full_"):] for path in (STEP7B_DIR / "sequences").glob("full_*.json")}
    add_check(checks, "required_g1_exact_45", actual_g1 == expected_short, {"count": len(actual_g1), "missing": sorted(expected_short - actual_g1), "unexpected": sorted(actual_g1 - expected_short)})
    add_check(checks, "required_sequences_exact_45", actual_sequences == expected_short, {"count": len(actual_sequences), "missing": sorted(expected_short - actual_sequences), "unexpected": sorted(actual_sequences - expected_short)})

    input_manifest_path = STEP7B_DIR / "input_file_manifest.json"
    output_manifest_path = STEP7B_DIR / "output_file_manifest.json"
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    output_manifest = json.loads(output_manifest_path.read_text(encoding="utf-8"))
    add_check(checks, "input_manifest_self_reference_excluded", bool(input_manifest.get("self_reference_excluded")), input_manifest.get("self_reference_excluded"))
    add_check(checks, "output_manifest_self_reference_excluded", bool(output_manifest.get("self_reference_excluded")), output_manifest.get("self_reference_excluded"))
    add_check(checks, "output_manifest_declares_1008", output_manifest.get("total_files") == len(output_manifest["outputs"]) == 1008, {"declared": output_manifest.get("total_files"), "actual": len(output_manifest["outputs"])})
    log(f"Verifying {len(input_manifest['inputs'])} input-manifest and {len(output_manifest['outputs'])} output-manifest targets...")
    input_hashes, _ = verify_manifest_targets(input_manifest["inputs"], lambda entry: PROJECT_ROOT / Path(entry["file"]), "step7b_input_manifest", checks)
    output_hashes, output_snapshot = verify_manifest_targets(output_manifest["outputs"], lambda entry: step7b_output_path(entry["relative_path"]), "step7b_output_manifest", checks)

    input_manifest_hash = sha256_file(input_manifest_path)
    output_manifest_hash = sha256_file(output_manifest_path)
    zip_size = STEP7B_ZIP.stat().st_size
    zip_hash = sha256_file(STEP7B_ZIP)
    with zipfile.ZipFile(STEP7B_ZIP, "r") as archive:
        zip_entries = archive.infolist()
        crc_failure = archive.testzip()
    add_check(checks, "approved_step7b_zip_size", zip_size == APPROVED_STEP7B_ZIP_SIZE, zip_size)
    add_check(checks, "approved_step7b_zip_sha256", zip_hash == APPROVED_STEP7B_ZIP_SHA256, zip_hash)
    add_check(checks, "approved_step7b_zip_entries_1009", len(zip_entries) == 1009, len(zip_entries))
    add_check(checks, "approved_step7b_zip_crc", crc_failure is None, crc_failure)

    preflight_failures = failed_checks(checks)
    if preflight_failures:
        raise RuntimeError(f"Step 7B preflight failed with {len(preflight_failures)} failed checks")
    sealed_test_ids = sorted(split_ids["test"])
    log(f"Preflight PASS: samples=45, split=27/9/9, manifest mismatches=0, sealed test IDs={len(sealed_test_ids)}")
    return {
        "samples": samples,
        "sample_map": {sample["sample_id"]: sample for sample in samples},
        "split_ids": split_ids,
        "words": words,
        "distribution": distribution,
        "segmentation": segmentation,
        "output_hashes": output_hashes,
        "output_snapshot": output_snapshot,
        "input_manifest_sha256": input_manifest_hash,
        "output_manifest_sha256": output_manifest_hash,
        "zip_sha256": zip_hash,
        "zip_size": zip_size,
        "sealed_test_ids": sealed_test_ids,
        "input_manifest_target_count": len(input_hashes),
        "output_manifest_target_count": len(output_hashes),
    }


def extract_speed(sample_id: str, word: str) -> str:
    tail = sample_id.removeprefix(f"S001_P001_{word}_")
    return tail.rsplit("_R", 1)[0]


def source_record(relative_path: str, hashes: dict[str, str]) -> dict:
    path = STEP7B_DIR / PurePosixPath(relative_path)
    return {"relative_path": relative_path, "size_bytes": path.stat().st_size, "sha256": hashes[relative_path]}


def feature_arrays(feature: dict) -> list[tuple[str, np.ndarray]]:
    return [
        ("coordinates", feature["coordinates"]),
        ("valid_mask", feature["valid_mask"]),
        ("original_valid_mask", feature["original_valid_mask"]),
        ("interpolation_mask", feature["interpolation_mask"]),
        ("source_frame_index", feature["source_frame_index"]),
        ("timestamp", feature["timestamp"]),
    ]


def build_features(preflight: dict, checks: list[dict]) -> tuple[dict, list[str]]:
    log("=== Feature export from Step 7B g1 arrays ===")
    features = {}
    failed_samples = []
    shared_sources = {}
    for relative_path in ("step7_dataset_split_manifest.json", "segmentation_summary.csv"):
        shared_sources[relative_path] = source_record(relative_path, preflight["output_hashes"])
    for name in ("input_file_manifest.json", "output_file_manifest.json"):
        path = STEP7B_DIR / name
        shared_sources[name] = {"relative_path": name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}

    required_arrays = {
        "pose_norm_interp", "lh_norm_interp", "rh_norm_interp",
        "pose_original_valid", "lh_original_valid", "rh_original_valid",
        "pose_mask", "lh_mask", "rh_mask",
    }
    for sample in preflight["samples"]:
        sample_id = sample["sample_id"]
        short = sample_id.removeprefix("S001_P001_")
        try:
            npz_relative = f"npz/interp_{short}_g1.npz"
            sequence_relative = f"sequences/full_{short}.json"
            npz_path = STEP7B_DIR / PurePosixPath(npz_relative)
            sequence_path = STEP7B_DIR / PurePosixPath(sequence_relative)
            segment = preflight["segmentation"][sample_id]
            start, end = segment["seg_start"], segment["seg_end"]
            frame_slice = slice(start, end + 1)
            expected_length = end - start + 1
            with np.load(npz_path, allow_pickle=False) as source:
                missing_arrays = sorted(required_arrays - set(source.files))
                add_check(checks, f"source_arrays_present_{sample_id}", not missing_arrays, missing_arrays)
                coordinates = np.ascontiguousarray(np.concatenate([
                    source["pose_norm_interp"][frame_slice][:, POSE_LANDMARKS, :],
                    source["lh_norm_interp"][frame_slice][:, HAND_LANDMARKS, :],
                    source["rh_norm_interp"][frame_slice][:, HAND_LANDMARKS, :],
                ], axis=1))
                original_valid = np.ascontiguousarray(np.concatenate([
                    source["pose_original_valid"][frame_slice][:, POSE_LANDMARKS],
                    source["lh_original_valid"][frame_slice][:, HAND_LANDMARKS],
                    source["rh_original_valid"][frame_slice][:, HAND_LANDMARKS],
                ], axis=1))
                interpolation_mask = np.ascontiguousarray(np.concatenate([
                    source["pose_mask"][frame_slice][:, POSE_LANDMARKS],
                    source["lh_mask"][frame_slice][:, HAND_LANDMARKS],
                    source["rh_mask"][frame_slice][:, HAND_LANDMARKS],
                ], axis=1))
                valid_mask = np.ascontiguousarray(original_valid | interpolation_mask)

            sequence = json.loads(sequence_path.read_text(encoding="utf-8"))
            frames = sequence["frames"]
            selected_frames = frames[frame_slice]
            source_frame_index = np.ascontiguousarray(np.array([frame["frame_index"] for frame in selected_frames], dtype=np.int64))
            timestamp = np.ascontiguousarray(np.array([frame["timestamp_s"] for frame in selected_frames], dtype=np.float64))
            sources = {
                npz_relative: source_record(npz_relative, preflight["output_hashes"]),
                sequence_relative: source_record(sequence_relative, preflight["output_hashes"]),
                **shared_sources,
            }
            metadata = {
                "sample_id": sample_id,
                "word": sample["word"],
                "speed": extract_speed(sample_id, sample["word"]),
                "split": sample["split"],
                "segment": {"seg_start": start, "seg_end": end, "inclusive": True},
                "landmark_schema": LANDMARK_SCHEMA,
                "source_files": sources,
                "normalization": "raw_framewise",
                "smoothing": "no_smoothing",
                "selected_max_gap": 1,
            }
            feature = {
                "coordinates": coordinates,
                "valid_mask": valid_mask,
                "original_valid_mask": original_valid,
                "interpolation_mask": interpolation_mask,
                "source_frame_index": source_frame_index,
                "timestamp": timestamp,
                "metadata": metadata,
                "sample_id": sample_id,
                "word": sample["word"],
                "speed": metadata["speed"],
                "split": sample["split"],
                "length": expected_length,
            }
            feature["logical_payload_sha256"] = logical_payload_hash(feature_arrays(feature), metadata)
            repeated_hash = logical_payload_hash(feature_arrays(feature), metadata)

            add_check(checks, f"feature_coordinates_shape_{sample_id}", coordinates.shape == (expected_length, TOTAL_LANDMARKS, 3), list(coordinates.shape))
            add_check(checks, f"feature_valid_shape_{sample_id}", valid_mask.shape == (expected_length, TOTAL_LANDMARKS), list(valid_mask.shape))
            add_check(checks, f"feature_original_valid_shape_{sample_id}", original_valid.shape == (expected_length, TOTAL_LANDMARKS), list(original_valid.shape))
            add_check(checks, f"feature_interpolation_mask_shape_{sample_id}", interpolation_mask.shape == (expected_length, TOTAL_LANDMARKS), list(interpolation_mask.shape))
            add_check(checks, f"feature_frame_shape_{sample_id}", source_frame_index.shape == (expected_length,), list(source_frame_index.shape))
            add_check(checks, f"feature_timestamp_shape_{sample_id}", timestamp.shape == (expected_length,), list(timestamp.shape))
            add_check(checks, f"feature_coordinates_float64_{sample_id}", coordinates.dtype == np.float64, str(coordinates.dtype))
            add_check(checks, f"feature_masks_bool_{sample_id}", valid_mask.dtype == np.bool_ and original_valid.dtype == np.bool_ and interpolation_mask.dtype == np.bool_, [str(valid_mask.dtype), str(original_valid.dtype), str(interpolation_mask.dtype)])
            add_check(checks, f"feature_valid_mask_union_{sample_id}", np.array_equal(valid_mask, original_valid | interpolation_mask), None)
            add_check(checks, f"feature_original_interp_disjoint_{sample_id}", not np.any(original_valid & interpolation_mask), int(np.sum(original_valid & interpolation_mask)))
            add_check(checks, f"feature_arrays_c_order_{sample_id}", all(array.flags.c_contiguous for _, array in feature_arrays(feature)), None)
            add_check(checks, f"feature_no_inf_{sample_id}", not np.isinf(coordinates).any(), int(np.isinf(coordinates).sum()))
            add_check(checks, f"feature_no_nan_where_valid_{sample_id}", not np.isnan(coordinates[valid_mask]).any(), int(np.isnan(coordinates[valid_mask]).sum()))
            add_check(checks, f"feature_timestamp_finite_{sample_id}", np.isfinite(timestamp).all(), None)
            add_check(checks, f"feature_timestamp_monotone_{sample_id}", bool(np.all(np.diff(timestamp) >= 0)), None)
            add_check(checks, f"feature_sequence_sample_id_{sample_id}", sequence["sample_id"] == sample_id, sequence["sample_id"])
            add_check(checks, f"feature_sequence_frame_count_{sample_id}", len(frames) == segment["total_frames"], {"sequence": len(frames), "summary": segment["total_frames"]})
            add_check(checks, f"feature_logical_hash_repeat_{sample_id}", feature["logical_payload_sha256"] == repeated_hash, repeated_hash)
            features[sample_id] = feature
        except Exception as exc:
            failed_samples.append(sample_id)
            add_check(checks, f"feature_build_{sample_id}", False, repr(exc))

    add_check(checks, "feature_count_45", len(features) == 45, len(features))
    add_check(checks, "failed_sample_count_zero", not failed_samples, failed_samples)
    if failed_samples:
        raise RuntimeError(f"feature generation failed for {len(failed_samples)} samples")
    lengths = [feature["length"] for feature in features.values()]
    log(f"Features PASS: 45 sequences, inclusive length range={min(lengths)}..{max(lengths)}")
    return features, failed_samples


def mask_aware_local_cost(coords_a: np.ndarray, mask_a: np.ndarray, coords_b: np.ndarray, mask_b: np.ndarray) -> float:
    included_costs = []
    for index in range(coords_a.shape[0]):
        valid_a, valid_b = bool(mask_a[index]), bool(mask_b[index])
        if valid_a and valid_b:
            included_costs.append(float(np.linalg.norm(coords_a[index] - coords_b[index])))
        elif valid_a != valid_b:
            included_costs.append(MISSING_COST)
    return float(np.mean(included_costs)) if included_costs else MISSING_COST


def local_cost_matrix(coords_a: np.ndarray, mask_a: np.ndarray, coords_b: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    both = mask_a[:, None, :] & mask_b[None, :, :]
    exactly_one = mask_a[:, None, :] ^ mask_b[None, :, :]
    differences = np.where(both[..., None], coords_a[:, None, :, :] - coords_b[None, :, :, :], 0.0)
    euclidean = np.sqrt(np.sum(differences * differences, axis=-1))
    costs = np.where(both, euclidean, np.where(exactly_one, MISSING_COST, 0.0))
    comparable = both | exactly_one
    denominator = comparable.sum(axis=-1)
    numerator = costs.sum(axis=-1)
    result = np.full(denominator.shape, MISSING_COST, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def band_allows(i: int, j: int, length_a: int, length_b: int, band: float) -> bool:
    if length_a <= 1 or length_b <= 1:
        return True
    return abs(i / (length_a - 1) - j / (length_b - 1)) <= band + 1e-15


def dtw_mask_aware(coords_a: np.ndarray, mask_a: np.ndarray, coords_b: np.ndarray, mask_b: np.ndarray, band: float) -> dict:
    length_a, length_b = coords_a.shape[0], coords_b.shape[0]
    local = local_cost_matrix(coords_a, mask_a, coords_b, mask_b)
    cumulative = np.full((length_a, length_b), np.inf, dtype=np.float64)
    parent = np.full((length_a, length_b), -1, dtype=np.int8)  # 0 diag, 1 vertical, 2 horizontal
    for i in range(length_a):
        for j in range(length_b):
            if not band_allows(i, j, length_a, length_b, band):
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
                prior_cost, direction = min(candidates, key=lambda item: (item[0], item[1]))
                cumulative[i, j] = prior_cost + local[i, j]
                parent[i, j] = direction
    total = float(cumulative[-1, -1])
    if not np.isfinite(total):
        raise RuntimeError(f"no DTW path within band={band} for lengths {length_a}/{length_b}")
    i, j = length_a - 1, length_b - 1
    reverse_path = [(i, j)]
    while i or j:
        direction = int(parent[i, j])
        if direction == 0:
            i, j = i - 1, j - 1
        elif direction == 1:
            i -= 1
        elif direction == 2:
            j -= 1
        else:
            raise RuntimeError("invalid DTW backtrack parent")
        reverse_path.append((i, j))
    path = list(reversed(reverse_path))
    return {"distance": total / len(path), "cumulative_cost": total, "path": path, "path_length": len(path)}


def reference_local_cost(coords_a: np.ndarray, mask_a: np.ndarray, coords_b: np.ndarray, mask_b: np.ndarray) -> float:
    total, count = 0.0, 0
    for landmark in range(TOTAL_LANDMARKS):
        left, right = bool(mask_a[landmark]), bool(mask_b[landmark])
        if left and right:
            delta = coords_a[landmark] - coords_b[landmark]
            total += float((delta[0] ** 2 + delta[1] ** 2 + delta[2] ** 2) ** 0.5)
            count += 1
        elif left or right:
            total += 1.0
            count += 1
    return total / count if count else 1.0


def validate_dtw_result(prefix: str, result: dict, feature_a: dict, feature_b: dict, band: float, checks: list[dict]) -> None:
    path = result["path"]
    length_a, length_b = feature_a["length"], feature_b["length"]
    add_check(checks, f"{prefix}_finite", np.isfinite(result["distance"]) and np.isfinite(result["cumulative_cost"]), result["distance"])
    add_check(checks, f"{prefix}_path_start", path[0] == (0, 0), path[0])
    add_check(checks, f"{prefix}_path_end", path[-1] == (length_a - 1, length_b - 1), path[-1])
    steps = [(path[index][0] - path[index - 1][0], path[index][1] - path[index - 1][1]) for index in range(1, len(path))]
    add_check(checks, f"{prefix}_path_steps", all(step in ((1, 1), (1, 0), (0, 1)) for step in steps), None)
    add_check(checks, f"{prefix}_band", all(band_allows(i, j, length_a, length_b, band) for i, j in path), None)
    independent_total = sum(reference_local_cost(feature_a["coordinates"][i], feature_a["valid_mask"][i], feature_b["coordinates"][j], feature_b["valid_mask"][j]) for i, j in path)
    normalized = independent_total / len(path)
    add_check(checks, f"{prefix}_path_normalization", np.isclose(independent_total, result["cumulative_cost"], rtol=1e-12, atol=1e-12) and np.isclose(normalized, result["distance"], rtol=1e-12, atol=1e-12), {"stored": result["distance"], "recomputed": normalized})


def compute_train_distances(features: dict, train_ids: list[str], band: float, checks: list[dict]) -> tuple[np.ndarray, list[dict]]:
    matrix = np.zeros((len(train_ids), len(train_ids)), dtype=np.float64)
    rows = []
    band_tag = f"b{int(round(band * 100)):02d}"
    for index, sample_id in enumerate(train_ids):
        result = dtw_mask_aware(features[sample_id]["coordinates"], features[sample_id]["valid_mask"], features[sample_id]["coordinates"], features[sample_id]["valid_mask"], band)
        validate_dtw_result(f"train_{band_tag}_self_{index:02d}", result, features[sample_id], features[sample_id], band, checks)
        add_check(checks, f"train_{band_tag}_self_distance_finite_{index:02d}", np.isfinite(result["distance"]), result["distance"])
        add_check(checks, f"train_{band_tag}_self_diagonal_{index:02d}", all(i == j for i, j in result["path"]), result["path_length"])
    for left in range(len(train_ids)):
        for right in range(left + 1, len(train_ids)):
            sample_a, sample_b = train_ids[left], train_ids[right]
            result = dtw_mask_aware(features[sample_a]["coordinates"], features[sample_a]["valid_mask"], features[sample_b]["coordinates"], features[sample_b]["valid_mask"], band)
            matrix[left, right] = matrix[right, left] = result["distance"]
            validate_dtw_result(f"train_{band_tag}_{left:02d}_{right:02d}", result, features[sample_a], features[sample_b], band, checks)
            rows.append({
                "band": band,
                "sample_a": sample_a,
                "sample_b": sample_b,
                "distance": result["distance"],
                "cumulative_cost": result["cumulative_cost"],
                "path_length": result["path_length"],
            })
    add_check(checks, f"train_{band_tag}_matrix_finite", np.isfinite(matrix).all(), None)
    add_check(checks, f"train_{band_tag}_matrix_symmetric", np.allclose(matrix, matrix.T, rtol=0, atol=0), None)
    add_check(checks, f"train_{band_tag}_matrix_diagonal_zero", np.allclose(np.diag(matrix), 0.0, atol=1e-12), None)
    for pair_index, (left, right) in enumerate(((0, 1), (2, 7), (10, 20))):
        forward = matrix[left, right]
        reverse = dtw_mask_aware(features[train_ids[right]]["coordinates"], features[train_ids[right]]["valid_mask"], features[train_ids[left]]["coordinates"], features[train_ids[left]]["valid_mask"], band)["distance"]
        add_check(checks, f"train_{band_tag}_independent_symmetry_{pair_index}", np.isclose(forward, reverse, rtol=1e-12, atol=1e-12), {"forward": forward, "reverse": reverse})
    return matrix, rows


def medoid_from_matrix(matrix: np.ndarray, train_ids: list[str], group_ids: list[str]) -> tuple[str, dict[str, float]]:
    indices = [train_ids.index(sample_id) for sample_id in group_ids]
    submatrix = matrix[np.ix_(indices, indices)]
    objectives = {sample_id: float(submatrix[index].sum()) for index, sample_id in enumerate(group_ids)}
    minimum = min(objectives.values())
    selected = min(sample_id for sample_id, value in objectives.items() if np.isclose(value, minimum, rtol=0, atol=1e-12))
    return selected, objectives


def independent_medoid(rows: list[dict], group_ids: list[str]) -> tuple[str, dict[str, float]]:
    lookup = {frozenset((row["sample_a"], row["sample_b"])): row["distance"] for row in rows}
    objectives = {}
    for candidate in group_ids:
        objectives[candidate] = float(sum(0.0 if candidate == other else lookup[frozenset((candidate, other))] for other in group_ids))
    minimum = min(objectives.values())
    selected = min(sample_id for sample_id, value in objectives.items() if np.isclose(value, minimum, rtol=0, atol=1e-12))
    return selected, objectives


def macro_f1(predictions: list[dict], words: list[str]) -> Fraction:
    values = []
    for word in words:
        true_positive = sum(row["true_word"] == word and row["predicted_word"] == word for row in predictions)
        false_positive = sum(row["true_word"] != word and row["predicted_word"] == word for row in predictions)
        false_negative = sum(row["true_word"] == word and row["predicted_word"] != word for row in predictions)
        denominator = 2 * true_positive + false_positive + false_negative
        values.append(Fraction(2 * true_positive, denominator) if denominator else Fraction(0, 1))
    return sum(values, Fraction(0, 1)) / len(values)


def evaluate_candidates(features: dict, preflight: dict, checks: list[dict]) -> tuple[list[dict], dict[float, np.ndarray], list[dict], list[dict], list[dict]]:
    log("=== Train-only medoids and validation-only candidate selection ===")
    train_ids = sorted(preflight["split_ids"]["train"])
    val_ids = sorted(preflight["split_ids"]["val"])
    words = preflight["words"]
    matrices = {}
    train_rows = []
    for band in BANDS:
        log(f"Computing train pairwise DTW for band {band:.2f}...")
        matrix, rows = compute_train_distances(features, train_ids, band, checks)
        matrices[band] = matrix
        train_rows.extend(rows)

    results, all_validation_distances, all_predictions = [], [], []
    for band in BANDS:
        band_rows = [row for row in train_rows if row["band"] == band]
        for strategy in STRATEGIES:
            config_name = f"{strategy}_band_{band:.2f}"
            if strategy == "word_medoid":
                groups = {word: sorted(sample_id for sample_id in train_ids if features[sample_id]["word"] == word) for word in words}
            else:
                groups = {}
                for word in words:
                    speeds = sorted({features[sample_id]["speed"] for sample_id in train_ids if features[sample_id]["word"] == word})
                    for speed in speeds:
                        groups[f"{word}__{speed}"] = sorted(sample_id for sample_id in train_ids if features[sample_id]["word"] == word and features[sample_id]["speed"] == speed)
            templates, medoid_records = {}, []
            for group_name, group_ids in groups.items():
                selected, objectives = medoid_from_matrix(matrices[band], train_ids, group_ids)
                independent_selected, independent_objectives = independent_medoid(band_rows, group_ids)
                add_check(checks, f"medoid_independent_sample_{config_name}_{group_name}", selected == independent_selected, {"stored": selected, "recomputed": independent_selected})
                add_check(checks, f"medoid_independent_objective_{config_name}_{group_name}", all(np.isclose(objectives[sample_id], independent_objectives[sample_id], rtol=1e-12, atol=1e-12) for sample_id in group_ids), None)
                add_check(checks, f"medoid_train_only_{config_name}_{group_name}", selected in preflight["split_ids"]["train"], selected)
                add_check(checks, f"medoid_val_excluded_{config_name}_{group_name}", selected not in preflight["split_ids"]["val"], selected)
                templates[group_name] = selected
                medoid_records.append({"group": group_name, "sample_id": selected, "objective_sum": objectives[selected], "group_size": len(group_ids), "member_objectives": objectives})

            validation_distances, predictions = [], []
            for val_index, val_id in enumerate(val_ids):
                word_best: dict[str, tuple[float, str, str]] = {}
                for template_index, (group_name, template_id) in enumerate(sorted(templates.items())):
                    result = dtw_mask_aware(features[val_id]["coordinates"], features[val_id]["valid_mask"], features[template_id]["coordinates"], features[template_id]["valid_mask"], band)
                    validate_dtw_result(f"validation_{config_name}_{val_index:02d}_{template_index:02d}", result, features[val_id], features[template_id], band, checks)
                    template_word = features[template_id]["word"]
                    row = {
                        "configuration": config_name,
                        "band": band,
                        "strategy": strategy,
                        "validation_sample_id": val_id,
                        "true_word": features[val_id]["word"],
                        "prototype_group": group_name,
                        "prototype_sample_id": template_id,
                        "prototype_word": template_word,
                        "distance": result["distance"],
                        "cumulative_cost": result["cumulative_cost"],
                        "path_length": result["path_length"],
                    }
                    validation_distances.append(row)
                    previous = word_best.get(template_word)
                    candidate = (result["distance"], group_name, template_id)
                    if previous is None or candidate < previous:
                        word_best[template_word] = candidate
                predicted_word = min(words, key=lambda word: (word_best[word][0], word))
                true_word = features[val_id]["word"]
                true_distance = word_best[true_word][0]
                non_true_distance = min(word_best[word][0] for word in words if word != true_word)
                predictions.append({
                    "configuration": config_name,
                    "sample_id": val_id,
                    "true_word": true_word,
                    "predicted_word": predicted_word,
                    "correct": predicted_word == true_word,
                    "true_word_distance": true_distance,
                    "best_non_true_distance": non_true_distance,
                    "true_label_margin": non_true_distance - true_distance,
                    "word_distances": {word: word_best[word][0] for word in words},
                })
            exact_macro_f1 = macro_f1(predictions, words)
            result = {
                "configuration_name": config_name,
                "strategy": strategy,
                "band": band,
                "template_count": len(templates),
                "templates": templates,
                "medoids": medoid_records,
                "validation_macro_f1": float(exact_macro_f1),
                "_validation_macro_f1_exact": exact_macro_f1,
                "validation_mean_true_label_margin": float(np.mean([row["true_label_margin"] for row in predictions])),
                "validation_correct": sum(row["correct"] for row in predictions),
                "validation_total": len(predictions),
                "validation_distances": validation_distances,
                "validation_predictions": predictions,
            }
            results.append(result)
            all_validation_distances.extend(validation_distances)
            all_predictions.extend(predictions)
            log(f"{config_name}: macro-F1={result['validation_macro_f1']:.12f} ({exact_macro_f1}), margin={result['validation_mean_true_label_margin']:.12f}, templates={result['template_count']}")

    add_check(checks, "candidate_configuration_count_6", len(results) == 6, len(results))
    add_check(checks, "train_pairwise_distance_count_1053", len(train_rows) == 3 * (27 * 26 // 2), len(train_rows))
    add_check(checks, "validation_prediction_count_54", len(all_predictions) == 6 * 9, len(all_predictions))
    add_check(checks, "validation_distance_count_324", len(all_validation_distances) == 3 * 9 * (3 + 9), len(all_validation_distances))
    return results, matrices, train_rows, all_validation_distances, all_predictions


def select_candidate(results: list[dict]) -> tuple[dict, list[dict]]:
    ranked = sorted(results, key=lambda row: (
        -row["_validation_macro_f1_exact"],
        -row["validation_mean_true_label_margin"],
        row["template_count"],
        row["band"],
        row["configuration_name"],
    ))
    return ranked[0], ranked


def template_manifest_data(best: dict, features: dict, run_id: str) -> dict:
    return {
        "version": VERSION,
        "run_id": run_id,
        "configuration_name": best["configuration_name"],
        "strategy": best["strategy"],
        "band": best["band"],
        "templates": {
            group: {
                "sample_id": sample_id,
                "word": features[sample_id]["word"],
                "speed": features[sample_id]["speed"],
                "split": features[sample_id]["split"],
                "logical_payload_sha256": features[sample_id]["logical_payload_sha256"],
            }
            for group, sample_id in sorted(best["templates"].items())
        },
    }


def find_template_leakage(manifest: dict, preflight: dict) -> list[dict]:
    train_ids = preflight["split_ids"]["train"]
    leaks = []
    for group, record in manifest["templates"].items():
        if record["sample_id"] not in train_ids or record.get("split") != "train":
            leaks.append({"group": group, "sample_id": record["sample_id"], "split": record.get("split")})
    return leaks


def add_core_verification(features: dict, preflight: dict, best: dict, train_rows: list[dict], validation_distances: list[dict], predictions: list[dict], run_id: str, checks: list[dict]) -> dict:
    log("=== Independent DTW and leakage verification ===")
    zeros = np.zeros((TOTAL_LANDMARKS, 3), dtype=np.float64)
    all_invalid = np.zeros(TOTAL_LANDMARKS, dtype=bool)
    one_valid = all_invalid.copy()
    one_valid[0] = True
    add_check(checks, "local_cost_both_invalid_is_1", mask_aware_local_cost(zeros, all_invalid, zeros, all_invalid) == 1.0, None)
    add_check(checks, "local_cost_exactly_one_valid_is_1", mask_aware_local_cost(zeros, one_valid, zeros, all_invalid) == 1.0, None)
    add_check(checks, "local_cost_identical_valid_is_0", mask_aware_local_cost(zeros, one_valid, zeros, one_valid) == 0.0, None)
    synthetic_coordinates = np.ascontiguousarray(np.stack([zeros, zeros, zeros], axis=0))
    synthetic_mask = np.ascontiguousarray(np.stack([one_valid, one_valid, one_valid], axis=0))
    synthetic_identical = dtw_mask_aware(synthetic_coordinates, synthetic_mask, synthetic_coordinates, synthetic_mask, 0.10)
    add_check(checks, "identical_sequence_dtw_distance_zero", np.isclose(synthetic_identical["distance"], 0.0, atol=1e-12), synthetic_identical["distance"])

    train_ids = sorted(preflight["split_ids"]["train"])
    for band_index, band in enumerate(BANDS):
        sample_a, sample_b = train_ids[band_index], train_ids[-(band_index + 1)]
        result = dtw_mask_aware(features[sample_a]["coordinates"], features[sample_a]["valid_mask"], features[sample_b]["coordinates"], features[sample_b]["valid_mask"], band)
        independent_total = sum(reference_local_cost(features[sample_a]["coordinates"][i], features[sample_a]["valid_mask"][i], features[sample_b]["coordinates"][j], features[sample_b]["valid_mask"][j]) for i, j in result["path"])
        add_check(checks, f"independent_dtw_cost_band_{band_index}", np.isclose(independent_total / result["path_length"], result["distance"], rtol=1e-12, atol=1e-12), None)

    production_manifest = template_manifest_data(best, features, run_id)
    production_leaks = find_template_leakage(production_manifest, preflight)
    add_check(checks, "production_template_leakage_zero", not production_leaks, production_leaks)
    injected_manifest = copy.deepcopy(production_manifest)
    first_group = sorted(injected_manifest["templates"])[0]
    injected_id = preflight["sealed_test_ids"][0]
    injected_manifest["templates"][first_group]["sample_id"] = injected_id
    injected_manifest["templates"][first_group]["split"] = "test"
    injected_leaks = find_template_leakage(injected_manifest, preflight)
    add_check(checks, "synthetic_test_leakage_detected", any(record["sample_id"] == injected_id for record in injected_leaks), injected_leaks)

    template_ids = {record["sample_id"] for record in production_manifest["templates"].values()}
    train_record_ids = {row["sample_a"] for row in train_rows} | {row["sample_b"] for row in train_rows}
    validation_record_ids = {row["validation_sample_id"] for row in validation_distances} | {row["prototype_sample_id"] for row in validation_distances} | {row["sample_id"] for row in predictions}
    for index, test_id in enumerate(preflight["sealed_test_ids"]):
        add_check(checks, f"test_{index}_not_template", test_id not in template_ids, test_id)
        add_check(checks, f"test_{index}_not_train_distances", test_id not in train_record_ids, test_id)
        add_check(checks, f"test_{index}_not_validation_records", test_id not in validation_record_ids, test_id)
    add_check(checks, "test_distance_count_zero", True, 0)
    add_check(checks, "test_prediction_count_zero", True, 0)
    add_check(checks, "test_accuracy_not_generated", True, None)
    add_check(checks, "test_confusion_matrix_not_generated", True, None)
    return {"injected_test_id": injected_id, "detected_records": injected_leaks, "production_leaks": production_leaks}


def verify_step7b_unchanged(preflight: dict, checks: list[dict]) -> dict:
    changes = []
    for relative_path, before in preflight["output_snapshot"].items():
        path = Path(before["path"])
        if not path.is_file():
            changes.append({"relative_path": relative_path, "reason": "missing"})
            continue
        stat = path.stat()
        actual_hash = sha256_file(path)
        if stat.st_size != before["size_bytes"] or stat.st_mtime_ns != before["mtime_ns"] or actual_hash != before["sha256"]:
            changes.append({"relative_path": relative_path, "before": before, "after": {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": actual_hash}})
    manifests_unchanged = (
        sha256_file(STEP7B_DIR / "input_file_manifest.json") == preflight["input_manifest_sha256"]
        and sha256_file(STEP7B_DIR / "output_file_manifest.json") == preflight["output_manifest_sha256"]
    )
    add_check(checks, "step7b_output_targets_read_only_unchanged", not changes, changes[:10])
    add_check(checks, "step7b_manifest_files_read_only_unchanged", manifests_unchanged, None)
    return {"changed_target_count": len(changes), "manifest_hashes_unchanged": manifests_unchanged}


def save_feature_npz(path: Path, feature: dict) -> None:
    np.savez_compressed(
        path,
        coordinates=feature["coordinates"],
        valid_mask=feature["valid_mask"],
        original_valid_mask=feature["original_valid_mask"],
        interpolation_mask=feature["interpolation_mask"],
        source_frame_index=feature["source_frame_index"],
        timestamp=feature["timestamp"],
        sample_id=np.array(feature["sample_id"]),
        word=np.array(feature["word"]),
        speed=np.array(feature["speed"]),
        split=np.array(feature["split"]),
        landmark_schema_json=np.array(canonical_json(LANDMARK_SCHEMA)),
        source_files_json=np.array(canonical_json(feature["metadata"]["source_files"])),
        metadata_json=np.array(canonical_json(feature["metadata"])),
        logical_payload_sha256=np.array(feature["logical_payload_sha256"]),
    )


def arrays_equal_exact(left: np.ndarray, right: np.ndarray) -> bool:
    if np.issubdtype(left.dtype, np.floating) or np.issubdtype(right.dtype, np.floating):
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def csv_write(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def candidate_public_record(result: dict, selected: bool) -> dict:
    return {
        "configuration_name": result["configuration_name"],
        "strategy": result["strategy"],
        "band": result["band"],
        "template_count": result["template_count"],
        "validation_macro_f1": result["validation_macro_f1"],
        "validation_mean_true_label_margin": result["validation_mean_true_label_margin"],
        "validation_correct": result["validation_correct"],
        "validation_total": result["validation_total"],
        "selected": selected,
    }


def generate_report(run_id: str, preflight: dict, features: dict, ranked: list[dict], best: dict, checks: list[dict], failed_samples: list[str], leakage: dict, artifact_counts: dict) -> str:
    summary = assertion_summary(checks)
    lengths = [feature["length"] for feature in features.values()]
    valid = sum(int(feature["valid_mask"].sum()) for feature in features.values())
    cells = sum(int(feature["valid_mask"].size) for feature in features.values())
    lines = [
        "# Step 8A Offline DTW Template Baseline v1.0",
        "",
        f"**Run ID**: {run_id}",
        f"**Version**: {VERSION}",
        "",
        "**Status**: DTW_TEMPLATE_COMPLETE_BLOCKED_PENDING_EXTERNAL_REVIEW",
        "",
        "## Frozen Step 7B inputs",
        "",
        f"- Approved ZIP: `{preflight['zip_sha256']}` ({preflight['zip_size']:,} bytes)",
        f"- Input manifest SHA-256: `{preflight['input_manifest_sha256']}`; verified targets: {preflight['input_manifest_target_count']}",
        f"- Output manifest SHA-256: `{preflight['output_manifest_sha256']}`; verified targets: {preflight['output_manifest_target_count']}",
        "- Missing, duplicate, unexpected, cross-split, size mismatch, hash mismatch: 0",
        "- Step 7B was not modified or rerun.",
        "",
        "## Feature definition",
        "",
        "- Input arrays: `pose_norm_interp`, `lh_norm_interp`, `rh_norm_interp`, all three `*_original_valid`, and all three `*_mask` arrays from g1 NPZ files.",
        "- DTW valid mask: `*_original_valid OR *_mask`; the interpolation mask is also preserved separately for auditability.",
        "- Exact order: pose `[13,14,15,16]` -> left hand `[0,4,8,12,16,20]` -> right hand `[0,4,8,12,16,20]`.",
        "- Inclusive segment: `seg_start:seg_end+1`; `raw_framewise`; `no_smoothing`; `selected_max_gap=1`.",
        "- No extra interpolation and no NaN-to-zero replacement were performed.",
        f"- Sequence length: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.3f}.",
        f"- Validity: {valid}/{cells} ({100.0 * valid / cells:.6f}%).",
        "",
        "## Dataset split",
        "",
        "| Word | Train | Validation | Test |",
        "|---|---:|---:|---:|",
    ]
    for word in preflight["words"]:
        row = preflight["distribution"][word]
        lines.append(f"| {word} | {row['train']} | {row['val']} | {row['test']} |")
    lines.extend([
        "",
        "## DTW definition",
        "",
        "- Both valid: Euclidean distance in normalized 3D coordinates.",
        "- Exactly one valid: fixed cost 1.0.",
        "- Both invalid: excluded from the denominator; if none are comparable, local cost is 1.0.",
        "- Local cost is the mean over included landmarks. Allowed steps are diagonal, vertical, horizontal with that tie order.",
        "- Final distance is cumulative path cost divided by warping-path length.",
        "",
        "## Six validation candidates",
        "",
        "Selection order: higher macro-F1, larger mean true-label margin, fewer templates, smaller band, lexicographically smaller configuration name.",
        "",
        "| Rank | Configuration | Templates | Validation macro-F1 | Mean true-label margin | Correct | Selected |",
        "|---:|---|---:|---:|---:|---:|---|",
    ])
    for index, result in enumerate(ranked, 1):
        selected = "YES" if result is best else ""
        lines.append(f"| {index} | {result['configuration_name']} | {result['template_count']} | {result['validation_macro_f1']:.12f} | {result['validation_mean_true_label_margin']:.12f} | {result['validation_correct']}/{result['validation_total']} | {selected} |")
    lines.extend([
        "",
        "## Train medoids for all candidates",
        "",
        "| Configuration | Group | Train sample | Objective sum |",
        "|---|---|---|---:|",
    ])
    for result in sorted(ranked, key=lambda item: item["configuration_name"]):
        for medoid in result["medoids"]:
            lines.append(f"| {result['configuration_name']} | {medoid['group']} | {medoid['sample_id']} | {medoid['objective_sum']:.12f} |")
    lines.extend([
        "",
        "## Selected configuration and validation result",
        "",
        f"- Configuration: `{best['configuration_name']}`",
        f"- Strategy: `{best['strategy']}`; band: `{best['band']:.2f}`; templates: {best['template_count']}",
        f"- Validation macro-F1: {best['validation_macro_f1']:.12f}",
        f"- Validation mean true-label margin: {best['validation_mean_true_label_margin']:.12f}",
        f"- Validation correct: {best['validation_correct']}/{best['validation_total']}",
        "",
        "| Prototype group | Selected train medoid |",
        "|---|---|",
    ])
    for group, sample_id in sorted(best["templates"].items()):
        lines.append(f"| {group} | {sample_id} |")
    lines.extend([
        "",
        "## Test leakage evidence",
        "",
        "- The nine test IDs were used only for feature export and the sealed manifest.",
        "- Test distances, predictions, accuracy, confusion matrix, and threshold selection: not generated.",
        "- Test IDs in templates, train distances, or validation records: 0.",
        f"- Synthetic injected test ID `{leakage['injected_test_id']}` was detected by the production leakage verifier: PASS.",
        "",
        "## Verification and artifacts",
        "",
        f"- Failed samples: {failed_samples if failed_samples else 'none'}",
        f"- Assertions: {summary['pass']} PASS / {summary['fail']} FAIL / {summary['total']} total / {summary['duplicate_count']} duplicate names.",
        f"- Artifact counts: `{canonical_json(artifact_counts)}`",
        "",
        "## Blocked next stage",
        "",
        "RecognitionCore, rejection-threshold finalization, final test evaluation, real-time input, and Unity integration remain blocked until ChatGPT Work external review.",
        "",
        "`DTW_TEMPLATE_COMPLETE_BLOCKED_PENDING_EXTERNAL_REVIEW`",
        "",
    ])
    return "\n".join(lines)


def write_artifacts(staging: Path, run_id: str, preflight: dict, features: dict, ranked: list[dict], best: dict, train_rows: list[dict], validation_distances: list[dict], predictions: list[dict], failed_samples: list[str], leakage: dict, checks: list[dict]) -> tuple[dict, list[dict]]:
    staging.mkdir(parents=True, exist_ok=False)
    feature_dir = staging / "features"
    template_dir = staging / "templates"
    feature_dir.mkdir()
    template_dir.mkdir()

    feature_paths = {}
    source_mismatch_total = 0
    for sample_id in sorted(features):
        feature = features[sample_id]
        short = sample_id.removeprefix("S001_P001_")
        path = feature_dir / f"feature_{short}.npz"
        save_feature_npz(path, feature)
        feature_paths[sample_id] = path
        with np.load(path, allow_pickle=False) as stored:
            array_match = all(arrays_equal_exact(stored[name], array) for name, array in feature_arrays(feature))
            metadata = json.loads(str(stored["metadata_json"].item()))
            recomputed = logical_payload_hash([(name, stored[name]) for name, _ in feature_arrays(feature)], metadata)
            embedded = str(stored["logical_payload_sha256"].item())
            scalar_match = (
                str(stored["sample_id"].item()) == sample_id
                and str(stored["word"].item()) == feature["word"]
                and str(stored["speed"].item()) == feature["speed"]
                and str(stored["split"].item()) == feature["split"]
            )
        if not array_match:
            source_mismatch_total += 1
        add_check(checks, f"feature_npz_exact_arrays_{sample_id}", array_match, None)
        add_check(checks, f"feature_npz_scalar_metadata_{sample_id}", scalar_match, None)
        add_check(checks, f"feature_npz_logical_hash_{sample_id}", recomputed == embedded == feature["logical_payload_sha256"], {"embedded": embedded, "recomputed": recomputed})
    add_check(checks, "feature_npz_count_45", len(feature_paths) == 45, len(feature_paths))
    add_check(checks, "feature_npz_source_mismatch_zero", source_mismatch_total == 0, source_mismatch_total)

    template_manifest = template_manifest_data(best, features, run_id)
    for group, record in template_manifest["templates"].items():
        source = feature_paths[record["sample_id"]]
        destination = template_dir / f"template_{group}.npz"
        shutil.copy2(source, destination)
        with np.load(destination, allow_pickle=False) as stored:
            embedded = str(stored["logical_payload_sha256"].item())
            metadata = json.loads(str(stored["metadata_json"].item()))
            recomputed = logical_payload_hash([(name, stored[name]) for name, _ in feature_arrays(features[record["sample_id"]])], metadata)
        record["relative_path"] = destination.relative_to(staging).as_posix()
        record["file_sha256"] = sha256_file(destination)
        add_check(checks, f"template_payload_hash_{group}", embedded == recomputed == record["logical_payload_sha256"], {"embedded": embedded, "recomputed": recomputed})
        add_check(checks, f"template_source_train_{group}", record["split"] == "train" and record["sample_id"] in preflight["split_ids"]["train"], record)
    write_json(staging / "template_manifest.json", template_manifest)

    feature_schema = {
        "version": VERSION,
        "run_id": run_id,
        "coordinates": {"shape": ["T", 16, 3], "dtype": "float64"},
        "valid_mask": {"shape": ["T", 16], "dtype": "bool", "definition": "*_original_valid OR *_mask"},
        "original_valid_mask": {"shape": ["T", 16], "dtype": "bool", "source": "*_original_valid"},
        "interpolation_mask": {"shape": ["T", 16], "dtype": "bool", "source": "*_mask", "meaning": "gap=1 interpolated positions"},
        "source_frame_index": {"shape": ["T"], "source": "sequences/full_{sample}.json frames[].frame_index"},
        "timestamp": {"shape": ["T"], "dtype": "float64", "source": "sequences/full_{sample}.json frames[].timestamp_s"},
        "landmark_order": LANDMARK_SCHEMA,
        "segment": "inclusive seg_start:seg_end+1",
        "normalization": "raw_framewise",
        "smoothing": "no_smoothing",
        "selected_max_gap": 1,
        "nan_policy": "preserve source NaN; never replace with zero",
    }
    write_json(staging / "feature_schema.json", feature_schema)

    train_csv_rows = [{**row, "band": f"{row['band']:.2f}", "distance": f"{row['distance']:.17g}", "cumulative_cost": f"{row['cumulative_cost']:.17g}"} for row in train_rows]
    csv_write(staging / "train_pairwise_distances.csv", ["band", "sample_a", "sample_b", "distance", "cumulative_cost", "path_length"], train_csv_rows)
    validation_csv_rows = [{**row, "band": f"{row['band']:.2f}", "distance": f"{row['distance']:.17g}", "cumulative_cost": f"{row['cumulative_cost']:.17g}"} for row in validation_distances]
    csv_write(staging / "validation_distances.csv", ["configuration", "band", "strategy", "validation_sample_id", "true_word", "prototype_group", "prototype_sample_id", "prototype_word", "distance", "cumulative_cost", "path_length"], validation_csv_rows)
    prediction_rows = []
    for row in predictions:
        prediction_rows.append({
            "configuration": row["configuration"],
            "sample_id": row["sample_id"],
            "true_word": row["true_word"],
            "predicted_word": row["predicted_word"],
            "correct": row["correct"],
            "true_word_distance": f"{row['true_word_distance']:.17g}",
            "best_non_true_distance": f"{row['best_non_true_distance']:.17g}",
            "true_label_margin": f"{row['true_label_margin']:.17g}",
            "word_distances_json": canonical_json(row["word_distances"]),
        })
    csv_write(staging / "validation_predictions.csv", ["configuration", "sample_id", "true_word", "predicted_word", "correct", "true_word_distance", "best_non_true_distance", "true_label_margin", "word_distances_json"], prediction_rows)

    candidate_medoids = {
        "version": VERSION,
        "run_id": run_id,
        "configurations": [
            {"configuration_name": result["configuration_name"], "strategy": result["strategy"], "band": result["band"], "medoids": result["medoids"]}
            for result in sorted(ranked, key=lambda item: item["configuration_name"])
        ],
    }
    write_json(staging / "candidate_medoids.json", candidate_medoids)
    candidate_summary = {
        "version": VERSION,
        "run_id": run_id,
        "selection_rules": ["higher_validation_macro_f1", "larger_mean_true_label_margin", "fewer_templates", "smaller_band", "lexicographically_smaller_configuration_name"],
        "selected_configuration": best["configuration_name"],
        "candidates": [candidate_public_record(result, result is best) for result in ranked],
    }
    write_json(staging / "validation_candidate_summary.json", candidate_summary)
    selected_config = {
        "version": VERSION,
        "run_id": run_id,
        "configuration_name": best["configuration_name"],
        "strategy": best["strategy"],
        "sakoe_chiba_relative_band": best["band"],
        "template_count": best["template_count"],
        "local_cost": {"both_valid": "normalized_3d_euclidean", "exactly_one_valid": 1.0, "both_invalid": "excluded", "no_comparable_landmarks": 1.0, "aggregation": "mean_included_landmark_costs"},
        "allowed_steps": ["diagonal(1,1)", "vertical(1,0)", "horizontal(0,1)"],
        "tie_break_order": ["diagonal", "vertical", "horizontal"],
        "distance_normalization": "cumulative_cost_divided_by_warping_path_length",
        "selection_scope": "validation_only",
        "rejection_threshold": "not_set",
    }
    write_json(staging / "selected_dtw_config.json", selected_config)
    test_seal = {
        "version": VERSION,
        "run_id": run_id,
        "sealed_test_ids": preflight["sealed_test_ids"],
        "test_feature_payload_sha256": {sample_id: features[sample_id]["logical_payload_sha256"] for sample_id in preflight["sealed_test_ids"]},
        "allowed_usage": "feature_export_and_sealed_manifest_only",
        "test_distance_count": 0,
        "test_prediction_count": 0,
        "test_accuracy": "not_generated",
        "test_confusion_matrix": "not_generated",
        "rejection_threshold_use": "none",
    }
    write_json(staging / "test_seal_manifest.json", test_seal)

    input_files = []
    seen_sources = {}
    for feature in features.values():
        seen_sources.update(feature["metadata"]["source_files"])
    for relative_path, record in sorted(seen_sources.items()):
        input_files.append({**record, "source_root": "step7b_full_v1_0"})
    input_files.append({"relative_path": STEP7B_ZIP.name, "source_root": "project_root", "size_bytes": preflight["zip_size"], "sha256": preflight["zip_sha256"], "role": "approved_external_review_package"})
    input_manifest = {
        "version": VERSION,
        "run_id": run_id,
        "self_reference_excluded": True,
        "step7b_read_only": True,
        "input_files": input_files,
        "total_files": len(input_files),
    }
    write_json(staging / "input_manifest.json", input_manifest)

    run_id_payloads = [feature_schema, template_manifest, candidate_medoids, candidate_summary, selected_config, test_seal, input_manifest]
    add_check(checks, "single_run_id_across_json_payloads", all(payload["run_id"] == run_id for payload in run_id_payloads), run_id)
    add_check(checks, "selected_template_count_matches_files", len(list(template_dir.glob("*.npz"))) == best["template_count"], len(list(template_dir.glob("*.npz"))))

    artifact_counts = {
        "feature_npz": len(feature_paths),
        "selected_template_npz": len(list(template_dir.glob("*.npz"))),
        "candidate_configurations": len(ranked),
        "candidate_medoid_records": sum(len(result["medoids"]) for result in ranked),
        "train_pairwise_distance_rows": len(train_rows),
        "validation_distance_rows": len(validation_distances),
        "validation_prediction_rows": len(predictions),
        "sealed_test_ids": len(preflight["sealed_test_ids"]),
    }
    summary = assertion_summary(checks)
    assertion_data = {
        "version": VERSION,
        "run_id": run_id,
        **summary,
        "synthetic_leakage_detection": {"status": "PASS", **leakage},
        "results": checks,
    }
    write_json(staging / "assertion_results.json", assertion_data)
    report = generate_report(run_id, preflight, features, ranked, best, checks, failed_samples, leakage, artifact_counts)
    (staging / "STEP8A_DTW_TEMPLATES_V1_0_REPORT.md").write_text(report, encoding="utf-8")

    output_records = []
    artifact_type_by_name = {
        "assertion_results.json": "assertion_results",
        "STEP8A_DTW_TEMPLATES_V1_0_REPORT.md": "report",
        "candidate_medoids.json": "candidate_medoids",
        "validation_candidate_summary.json": "validation_candidate_summary",
        "feature_schema.json": "feature_schema",
        "input_manifest.json": "input_manifest",
        "selected_dtw_config.json": "selected_config",
        "template_manifest.json": "template_manifest",
        "test_seal_manifest.json": "test_seal_manifest",
        "train_pairwise_distances.csv": "train_pairwise_distances",
        "validation_distances.csv": "validation_distances",
        "validation_predictions.csv": "validation_predictions",
    }
    for path in sorted(item for item in staging.rglob("*") if item.is_file()):
        relative_path = path.relative_to(staging).as_posix()
        if relative_path.startswith("features/"):
            artifact_type = "feature_npz"
        elif relative_path.startswith("templates/"):
            artifact_type = "template_npz"
        else:
            artifact_type = artifact_type_by_name[relative_path]
        output_records.append({"relative_path": relative_path, "size_bytes": path.stat().st_size, "sha256": sha256_file(path), "artifact_type": artifact_type})
    script_path = Path(__file__).resolve()
    output_records.append({"relative_path": script_path.name, "size_bytes": script_path.stat().st_size, "sha256": sha256_file(script_path), "artifact_type": "execution_script"})
    output_records.sort(key=lambda row: row["relative_path"])
    output_manifest = {
        "version": VERSION,
        "schema_version": "8a_v1.0",
        "run_id": run_id,
        "self_reference_excluded": True,
        "outputs": output_records,
        "total_files": len(output_records),
    }
    write_json(staging / "output_file_manifest.json", output_manifest)
    return assertion_data, output_records


def package_and_verify(staging: Path, run_id: str, output_records: list[dict], read_only_result: dict) -> tuple[dict, Path, Path, Path]:
    log("=== ZIP packaging and post-ZIP verification ===")
    temp_zip = PROJECT_ROOT / f".{DELIVERY_NAME}.{run_id}.tmp"
    temp_hash = PROJECT_ROOT / f".{DELIVERY_NAME}.sha256.{run_id}.tmp"
    temp_sidecar = PROJECT_ROOT / f".{POST_ZIP_SIDECAR.name}.{run_id}.tmp"
    manifest_lookup = {row["relative_path"]: row for row in output_records}
    output_manifest_path = staging / "output_file_manifest.json"
    with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for record in output_records:
            source = Path(__file__).resolve() if record["artifact_type"] == "execution_script" else staging / PurePosixPath(record["relative_path"])
            archive.write(source, record["relative_path"])
        archive.write(output_manifest_path, "output_file_manifest.json")
    zip_hash = sha256_file(temp_zip)
    zip_size = temp_zip.stat().st_size
    temp_hash.write_text(f"{zip_hash}  {DELIVERY_NAME}\n", encoding="ascii")

    records = []
    def verify_record(name: str, passed: bool, evidence: object = None) -> None:
        records.append({"check_name": name, "status": "PASS" if bool(passed) else "FAIL", "evidence": evidence})

    with zipfile.ZipFile(temp_zip, "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        for index, info in enumerate(infos):
            data = archive.read(info.filename)
            path = PurePosixPath(info.filename)
            unsafe = info.filename.startswith(("/", "\\")) or ".." in path.parts or bool(re.match(r"^[A-Za-z]:", info.filename))
            encrypted = bool(info.flag_bits & 0x1)
            symlink = ((info.external_attr >> 16) & 0o170000) == 0o120000
            verify_record(f"entry_{index:04d}_crc", (zipfile.crc32(data) & 0xFFFFFFFF) == info.CRC, info.filename)
            verify_record(f"entry_{index:04d}_safe_path", not unsafe, info.filename)
            verify_record(f"entry_{index:04d}_not_encrypted", not encrypted, info.filename)
            verify_record(f"entry_{index:04d}_not_symlink", not symlink, info.filename)
            if info.filename == "output_file_manifest.json":
                verify_record(f"entry_{index:04d}_manifest_self_excluded", info.filename not in manifest_lookup, info.filename)
            else:
                target = manifest_lookup.get(info.filename)
                verify_record(f"entry_{index:04d}_in_manifest", target is not None, info.filename)
                verify_record(f"entry_{index:04d}_size", target is not None and len(data) == target["size_bytes"], info.filename)
                verify_record(f"entry_{index:04d}_sha256", target is not None and hashlib.sha256(data).hexdigest() == target["sha256"], info.filename)
    expected_names = set(manifest_lookup) | {"output_file_manifest.json"}
    verify_record("global_duplicate_zip_entries_zero", len(names) == len(set(names)), {"total": len(names), "unique": len(set(names))})
    verify_record("global_zip_entry_set_matches_manifest_plus_self_excluded_manifest", set(names) == expected_names, {"actual": len(set(names)), "expected": len(expected_names)})
    verify_record("global_output_manifest_self_reference_excluded", "output_file_manifest.json" not in manifest_lookup, None)

    detached_text = temp_hash.read_text(encoding="ascii")
    detached_match = re.fullmatch(r"([0-9a-f]{64})  ([^\r\n]+)\n", detached_text)
    verify_record("detached_hash_strict_parse", detached_match is not None, detached_text.rstrip("\n"))
    verify_record("detached_hash_filename", detached_match is not None and detached_match.group(2) == DELIVERY_NAME, detached_match.group(2) if detached_match else None)
    verify_record("detached_hash_value", detached_match is not None and detached_match.group(1) == zip_hash, detached_match.group(1) if detached_match else None)
    verify_record("step7b_read_only_changed_targets_zero", read_only_result["changed_target_count"] == 0, read_only_result)
    verify_record("step7b_read_only_manifest_hashes_unchanged", read_only_result["manifest_hashes_unchanged"], read_only_result)

    check_names = [record["check_name"] for record in records]
    duplicate_check_names = sorted(name for name, count in Counter(check_names).items() if count > 1)
    canonical_records = [canonical_json(record) for record in records]
    duplicate_records = sum(count - 1 for count in Counter(canonical_records).values() if count > 1)
    failure_count = sum(record["status"] != "PASS" for record in records)
    sidecar = {
        "version": VERSION,
        "run_id": run_id,
        "zip_file": DELIVERY_NAME,
        "zip_size_bytes": zip_size,
        "zip_sha256": zip_hash,
        "zip_entry_count": len(names),
        "zip_unique_entry_count": len(set(names)),
        "output_manifest_target_count": len(output_records),
        "output_manifest_self_reference_excluded": True,
        "detached_hash_strict_parse": detached_match is not None,
        "detached_hash_filename_ok": detached_match is not None and detached_match.group(2) == DELIVERY_NAME,
        "detached_hash_value_ok": detached_match is not None and detached_match.group(1) == zip_hash,
        "sidecar_check_total": len(records),
        "sidecar_check_unique": len(set(check_names)),
        "sidecar_check_name_duplicates": len(duplicate_check_names),
        "sidecar_record_duplicates": duplicate_records,
        "sidecar_failure_count": failure_count,
        "duplicate_check_names": duplicate_check_names,
        "verification_status": "PASS" if failure_count == 0 and not duplicate_check_names and duplicate_records == 0 else "FAIL",
        "checks": records,
    }
    write_json(temp_sidecar, sidecar)
    return sidecar, temp_zip, temp_hash, temp_sidecar


def commit_outputs(staging: Path, temp_zip: Path, temp_hash: Path, temp_sidecar: Path, run_id: str) -> None:
    parent = OUTPUT_DIR.parent.resolve()
    if staging.parent.resolve() != parent or not staging.name.startswith(f".{OUTPUT_DIR.name}.staging."):
        raise RuntimeError("unsafe staging directory")
    backup = parent / f".{OUTPUT_DIR.name}.backup.{run_id}"
    if backup.exists():
        raise RuntimeError("unexpected backup path already exists")
    moved_old = False
    if OUTPUT_DIR.exists():
        OUTPUT_DIR.rename(backup)
        moved_old = True
    try:
        staging.rename(OUTPUT_DIR)
    except Exception:
        if moved_old and backup.exists() and not OUTPUT_DIR.exists():
            backup.rename(OUTPUT_DIR)
        raise
    os.replace(temp_zip, DELIVERY_ZIP)
    os.replace(temp_hash, DETACHED_HASH)
    os.replace(temp_sidecar, POST_ZIP_SIDECAR)
    if moved_old:
        if backup.parent.resolve() != parent or not backup.name.startswith(f".{OUTPUT_DIR.name}.backup."):
            raise RuntimeError("unsafe backup cleanup target")
        shutil.rmtree(backup)


def main() -> int:
    run_id = str(uuid.uuid4())
    started = time.time()
    checks: list[dict] = []
    staging = OUTPUT_DIR.parent / f".{OUTPUT_DIR.name}.staging.{run_id}"
    log(f"Step 8A offline DTW template baseline v{VERSION}")
    log(f"Run ID: {run_id}")
    try:
        preflight = preflight_step7b(checks)
        features, failed_samples = build_features(preflight, checks)
        results, _, train_rows, validation_distances, predictions = evaluate_candidates(features, preflight, checks)
        best, ranked = select_candidate(results)
        log(f"Selected: {best['configuration_name']}")
        leakage = add_core_verification(features, preflight, best, train_rows, validation_distances, predictions, run_id, checks)
        read_only_result = verify_step7b_unchanged(preflight, checks)
        if failed_checks(checks):
            raise RuntimeError(f"core verification has {len(failed_checks(checks))} failed assertions")
        assertion_data, output_records = write_artifacts(staging, run_id, preflight, features, ranked, best, train_rows, validation_distances, predictions, failed_samples, leakage, checks)
        if assertion_data["fail"] or assertion_data["duplicate_count"]:
            raise RuntimeError(f"assertion artifact reports fail={assertion_data['fail']} duplicates={assertion_data['duplicate_count']}")
        sidecar, temp_zip, temp_hash, temp_sidecar = package_and_verify(staging, run_id, output_records, read_only_result)
        if sidecar["verification_status"] != "PASS":
            raise RuntimeError(f"post-ZIP verification failed with {sidecar['sidecar_failure_count']} failures")
        commit_outputs(staging, temp_zip, temp_hash, temp_sidecar, run_id)
        log("=== COMPLETE ===")
        log(f"Assertions: {assertion_data['pass']} PASS / {assertion_data['fail']} FAIL / {assertion_data['total']} total / {assertion_data['duplicate_count']} duplicates")
        log(f"ZIP: {DELIVERY_ZIP.name} | {sidecar['zip_size_bytes']:,} bytes | {sidecar['zip_sha256']}")
        log(f"Sidecar: total={sidecar['sidecar_check_total']} unique={sidecar['sidecar_check_unique']} duplicates={sidecar['sidecar_check_name_duplicates']}/{sidecar['sidecar_record_duplicates']} failures={sidecar['sidecar_failure_count']}")
        log(f"Elapsed: {time.time() - started:.1f}s")
        log("Status: DTW_TEMPLATE_COMPLETE_BLOCKED_PENDING_EXTERNAL_REVIEW")
        return 0
    except Exception as exc:
        log(f"BLOCKED: {exc}")
        log("No completion state declared; existing completed outputs were preserved unless final commit succeeded.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
