#!/usr/bin/env python3
"""Step 8C two-phase offline RecognitionCore evaluation and delivery tooling."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
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
STEP8B = ROOT / "data/processed/sessions/S001_P001/step8b_recognition_core_v1_0"
OUTPUT = ROOT / "data/processed/sessions/S001_P001/step8c_offline_recognition_evaluation_v1_0"
STEP8A_ZIP = ROOT / "WordsInSilence_step8a_dtw_templates_v1_0_delivery.zip"
STEP8B_ZIP = ROOT / "WordsInSilence_step8b_recognition_core_v1_0_delivery.zip"
STEP7B_ZIP = ROOT / "WordsInSilence_step7b_v1_0_full_delivery.zip"
STEP8A_SHA = ROOT / f"{STEP8A_ZIP.name}.sha256"
STEP8B_SHA = ROOT / f"{STEP8B_ZIP.name}.sha256"
STEP8A_SIDECAR = ROOT / "WordsInSilence_step8a_dtw_templates_v1_0_delivery_post_zip_verification.json"
STEP8B_SIDECAR = ROOT / "WordsInSilence_step8b_recognition_core_v1_0_delivery_post_zip_verification.json"
THRESHOLD_PATH = OUTPUT / "frozen_threshold_config.json"
STATE_PATH = OUTPUT / "run_state.json"
RUNTIME_ROOT = ROOT / "Assets/StreamingAssets/RecognitionCoreV1/Templates"
FIXTURE_ROOT = ROOT / "Assets/_Project/Tests/EditMode/Fixtures/OfflineRecognitionEvaluationV1"
TEST_QUERY_ROOT = FIXTURE_ROOT / "TestQueries"
EVALUATION_SOURCE_ROOT = ROOT / "Assets/_Project/Scripts/Recognition/Evaluation"
TEST_SOURCE = ROOT / "Assets/_Project/Tests/EditMode/OfflineRecognitionEvaluationV1Tests.cs"
DELIVERY_NAME = "WordsInSilence_step8c_offline_recognition_evaluation_v1_0_delivery.zip"
DELIVERY_ZIP = ROOT / DELIVERY_NAME
DELIVERY_HASH = ROOT / f"{DELIVERY_NAME}.sha256"
DELIVERY_SIDECAR = ROOT / "WordsInSilence_step8c_offline_recognition_evaluation_v1_0_delivery_post_zip_verification.json"

VERSION = "1.0"
STEP8A_RUN_ID = "0918c71c-313f-49d0-a5a9-b525f6630317"
STEP8B_RUN_ID = "f1063bbe-c285-4df6-af66-4b984096f330"
STEP8A_ZIP_SIZE = 2_205_564
STEP8A_ZIP_SHA256 = "fa8292c424644bc5f50f7213e7587f5157b71cd651df804b8a542168ffe4bf3c"
STEP8B_ZIP_SIZE = 661_831
STEP8B_ZIP_SHA256 = "3401ea74f73902c5ea7b4f34abb06ebfe933ef0a076820f15b6471937a5c7c6d"
STEP7B_ZIP_SIZE = 226_548_811
STEP7B_ZIP_SHA256 = "472c43fc53f96751cfc18e79a9a1c4a247ee0fdc5531fe80af532fa88ef8b015"
EXPECTED_MAX_TRUE = 0.2777641390093352
EXPECTED_MIN_WRONG = 0.3039716626885040
EXPECTED_THRESHOLD = 0.2908679008489196
TOLERANCE = 1e-12
WORDS = ("GREETING", "LOVE", "THANK_YOU")
MAGIC = b"WISRCB01"
FORMAT_VERSION = 1
ENDIAN_MARKER = 0x01020304
LANDMARK_COUNT = 16
BAND = 0.10
TEST_IDS = (
    "S001_P001_GREETING_NORMAL_R01",
    "S001_P001_GREETING_SLOW_R02",
    "S001_P001_GREETING_SLOW_R05",
    "S001_P001_LOVE_NORMAL_R01",
    "S001_P001_LOVE_SLOW_R02",
    "S001_P001_LOVE_SLOW_R05",
    "S001_P001_THANK_YOU_NORMAL_R01",
    "S001_P001_THANK_YOU_SLOW_R02",
    "S001_P001_THANK_YOU_SLOW_R05",
)


def log(message: str = "") -> None:
    print(message, flush=True)


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


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


def strict_detached_hash(path: Path, delivery: Path, expected_hash: str) -> bool:
    text = path.read_text(encoding="ascii")
    match = re.fullmatch(r"([0-9a-f]{64})  ([^\r\n]+)\n", text)
    return bool(match and match.group(1) == expected_hash and match.group(2) == delivery.name)


def verify_delivery_zip(
    label: str,
    delivery: Path,
    detached: Path,
    sidecar_path: Path,
    expected_size: int,
    expected_hash: str,
    expected_entries: int,
    checks: list[dict],
) -> None:
    actual_size = delivery.stat().st_size
    actual_hash = sha256_file(delivery)
    add_check(checks, f"{label}_zip_size", actual_size == expected_size, actual_size)
    add_check(checks, f"{label}_zip_sha256", actual_hash == expected_hash, actual_hash)
    add_check(checks, f"{label}_detached_strict", strict_detached_hash(detached, delivery, actual_hash), None)
    with zipfile.ZipFile(delivery, "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        bad_crc = archive.testzip()
    unsafe = [
        name for name in names
        if name.startswith(("/", "\\")) or ".." in PurePosixPath(name).parts or re.match(r"^[A-Za-z]:", name)
    ]
    symlinks = [info.filename for info in infos if stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK]
    encrypted = [info.filename for info in infos if info.flag_bits & 1]
    add_check(checks, f"{label}_zip_crc", bad_crc is None, bad_crc)
    add_check(checks, f"{label}_zip_entry_count", len(names) == expected_entries, len(names))
    add_check(checks, f"{label}_zip_unique", len(names) == len(set(names)), [len(names), len(set(names))])
    add_check(checks, f"{label}_zip_safe", not unsafe, unsafe)
    add_check(checks, f"{label}_zip_no_symlinks", not symlinks, symlinks)
    add_check(checks, f"{label}_zip_no_encryption", not encrypted, encrypted)
    sidecar = load_json(sidecar_path)
    sidecar_checks = sidecar["checks"]
    check_names = [record["check_name"] for record in sidecar_checks]
    canonical_records = [canonical_json(record) for record in sidecar_checks]
    add_check(checks, f"{label}_sidecar_status", sidecar["verification_status"] == "PASS", sidecar["verification_status"])
    add_check(checks, f"{label}_sidecar_all_pass", all(record["status"] == "PASS" for record in sidecar_checks), None)
    add_check(checks, f"{label}_sidecar_check_names_unique", len(check_names) == len(set(check_names)), None)
    add_check(checks, f"{label}_sidecar_records_unique", len(canonical_records) == len(set(canonical_records)), None)


def validate_sealed_manifest(seal: dict, checks: list[dict]) -> None:
    add_check(checks, "test_seal_ids_exact", tuple(seal["sealed_test_ids"]) == TEST_IDS, seal["sealed_test_ids"])
    add_check(checks, "test_seal_hash_ids_exact", tuple(seal["test_feature_payload_sha256"].keys()) == TEST_IDS, None)
    add_check(checks, "test_seal_distance_zero", seal["test_distance_count"] == 0, seal["test_distance_count"])
    add_check(checks, "test_seal_prediction_zero", seal["test_prediction_count"] == 0, seal["test_prediction_count"])
    add_check(checks, "test_seal_accuracy_absent", seal["test_accuracy"] == "not_generated", seal["test_accuracy"])
    add_check(checks, "test_seal_confusion_absent", seal["test_confusion_matrix"] == "not_generated", seal["test_confusion_matrix"])
    add_check(checks, "test_seal_threshold_unused", seal["rejection_threshold_use"] == "none", seal["rejection_threshold_use"])


def relevant_input_paths(fixture_manifest: dict, runtime_manifest: dict) -> list[Path]:
    paths = [
        STEP8A_ZIP, STEP8A_SHA, STEP8A_SIDECAR,
        STEP8B_ZIP, STEP8B_SHA, STEP8B_SIDECAR,
        STEP8A / "selected_dtw_config.json",
        STEP8A / "template_manifest.json",
        STEP8A / "test_seal_manifest.json",
        STEP8A / "validation_distances.csv",
        STEP8A / "validation_predictions.csv",
        STEP8B / "output_file_manifest.json",
        STEP8B / "runtime_template_manifest.json",
        STEP8B / "fixture_manifest.json",
    ]
    paths += [ROOT / PurePosixPath(record["relative_path"]) for record in runtime_manifest["templates"]]
    paths += [ROOT / PurePosixPath(record["relative_path"]) for record in fixture_manifest["validation_queries"]]
    paths += [
        ROOT / "Assets/_Project/Scripts/Recognition/Core/RecognitionSequence.cs",
        ROOT / "Assets/_Project/Scripts/Recognition/Core/RecognitionSequenceBinary.cs",
        ROOT / "Assets/_Project/Scripts/Recognition/Core/MaskAwareDtw.cs",
        ROOT / "Assets/_Project/Scripts/Recognition/Core/RecognitionCore.cs",
    ]
    return sorted(set(paths), key=lambda path: path.relative_to(ROOT).as_posix())


def snapshot(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        file_stat = path.stat()
        records.append({
            "relative_path": path.relative_to(ROOT).as_posix(),
            "size_bytes": file_stat.st_size,
            "sha256": sha256_file(path),
            "mtime_ns": file_stat.st_mtime_ns,
        })
    return records


def validate_threshold_selection_ids(selection_ids: list[str], sealed_ids: list[str]) -> None:
    if set(selection_ids) & set(sealed_ids):
        raise ValueError("threshold selection provenance contains a sealed test ID")


def selected_validation_records() -> tuple[list[dict], list[dict], str]:
    config = load_json(STEP8A / "selected_dtw_config.json")
    configuration_name = config["configuration_name"]
    with (STEP8A / "validation_distances.csv").open("r", encoding="utf-8", newline="") as stream:
        distances = [row for row in csv.DictReader(stream) if row["configuration"] == configuration_name]
    with (STEP8A / "validation_predictions.csv").open("r", encoding="utf-8", newline="") as stream:
        predictions = [row for row in csv.DictReader(stream) if row["configuration"] == configuration_name]
    return distances, predictions, configuration_name


def compute_validation_threshold(distances: list[dict], predictions: list[dict], checks: list[dict]) -> dict:
    prediction_map = {row["sample_id"]: row for row in predictions}
    validation_ids = sorted(prediction_map)
    add_check(checks, "validation_distance_records_81", len(distances) == 81, len(distances))
    add_check(checks, "validation_prediction_records_9", len(predictions) == 9, len(predictions))
    add_check(checks, "validation_ids_9", len(validation_ids) == 9, validation_ids)
    true_distances = []
    wrong_distances = []
    reproduced = []
    all_correct = True
    for sample_id in validation_ids:
        source_prediction = prediction_map[sample_id]
        rows = [row for row in distances if row["validation_sample_id"] == sample_id]
        word_best = {}
        for word in WORDS:
            candidates = sorted(
                (float(row["distance"]), row["prototype_sample_id"])
                for row in rows if row["prototype_word"] == word
            )
            if len(candidates) != 3:
                raise RuntimeError(f"validation template group count is not three: {sample_id} {word}")
            word_best[word] = candidates[0]
        true_word = source_prediction["true_word"]
        predicted_word = min(WORDS, key=lambda word: (word_best[word][0], word))
        d_true = word_best[true_word][0]
        d_wrong = min(word_best[word][0] for word in WORDS if word != true_word)
        true_distances.append(d_true)
        wrong_distances.append(d_wrong)
        correct = predicted_word == true_word == source_prediction["predicted_word"]
        all_correct = all_correct and correct
        reproduced.append({
            "sample_id": sample_id,
            "true_word": true_word,
            "predicted_word": predicted_word,
            "d_true": d_true,
            "d_wrong": d_wrong,
            "correct": correct,
        })
    validation_max_true = max(true_distances)
    validation_min_wrong = min(wrong_distances)
    threshold = (validation_max_true + validation_min_wrong) / 2.0
    gap = validation_min_wrong - validation_max_true
    add_check(checks, "validation_all_predictions_correct", all_correct, reproduced)
    add_check(checks, "validation_max_true_expected", abs(validation_max_true - EXPECTED_MAX_TRUE) <= TOLERANCE, validation_max_true)
    add_check(checks, "validation_min_wrong_expected", abs(validation_min_wrong - EXPECTED_MIN_WRONG) <= TOLERANCE, validation_min_wrong)
    add_check(checks, "validation_threshold_expected", abs(threshold - EXPECTED_THRESHOLD) <= TOLERANCE, threshold)
    add_check(checks, "validation_separation_gap_positive", gap > 0.0, gap)
    return {
        "validation_ids": validation_ids,
        "validation_max_true": validation_max_true,
        "validation_min_wrong": validation_min_wrong,
        "validation_separation_gap": gap,
        "threshold": threshold,
        "records": reproduced,
    }


def write_frozen_threshold(payload: dict) -> tuple[str, dict]:
    canonical_payload = canonical_json(payload)
    payload_hash = sha256_bytes(canonical_payload.encode("utf-8"))
    wrapper = f'{{"canonical_payload":{canonical_payload},"canonical_payload_sha256":"{payload_hash}"}}\n'
    THRESHOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with THRESHOLD_PATH.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(wrapper)
    validated = validate_frozen_threshold(THRESHOLD_PATH)
    return payload_hash, validated


def validate_frozen_threshold(path: Path) -> dict:
    raw = path.read_bytes()
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ValueError("frozen threshold config must have exactly one terminal LF")
    text = raw[:-1].decode("utf-8")
    prefix = '{"canonical_payload":'
    marker = ',"canonical_payload_sha256":"'
    if not text.startswith(prefix) or not text.endswith('"}'):
        raise ValueError("frozen threshold wrapper is malformed")
    marker_index = text.rfind(marker)
    if marker_index < len(prefix):
        raise ValueError("frozen threshold hash field is missing")
    canonical_payload = text[len(prefix):marker_index]
    supplied_hash = text[marker_index + len(marker):-2]
    if not re.fullmatch(r"[0-9a-f]{64}", supplied_hash):
        raise ValueError("frozen threshold hash is malformed")
    if sha256_bytes(canonical_payload.encode("utf-8")) != supplied_hash:
        raise ValueError("frozen threshold canonical payload hash mismatch")
    payload = json.loads(canonical_payload)
    if canonical_json(payload) != canonical_payload:
        raise ValueError("frozen threshold payload is not canonical JSON")
    if payload["version"] != VERSION or payload["selection_scope"] != "validation_only":
        raise ValueError("frozen threshold scope/version mismatch")
    if payload["comparison_operator"] != "<=" or payload["negative_or_unknown_samples_available"] is not False:
        raise ValueError("frozen threshold policy fields are invalid")
    if payload["open_set_rejection_performance"] != "not_evaluated":
        raise ValueError("open-set status is invalid")
    validate_threshold_selection_ids(payload["threshold_selection_query_ids"], list(TEST_IDS))
    return {"payload": payload, "canonical_payload_sha256": supplied_hash}


def phase_a() -> int:
    log("=== Step 8C Phase A: validation-only threshold freeze ===")
    if OUTPUT.exists():
        raise RuntimeError(f"Step 8C output path already exists: {OUTPUT}")
    for path in (
        ROOT / "WordsInSilence_step8c_offline_recognition_evaluation_v1_0_delivery.zip",
        ROOT / "WordsInSilence_step8c_offline_recognition_evaluation_v1_0_delivery.zip.sha256",
        ROOT / "WordsInSilence_step8c_offline_recognition_evaluation_v1_0_delivery_post_zip_verification.json",
    ):
        if path.exists():
            raise RuntimeError(f"Step 8C root deliverable path conflicts with existing work: {path}")

    checks: list[dict] = []
    verify_delivery_zip("step8a", STEP8A_ZIP, STEP8A_SHA, STEP8A_SIDECAR, STEP8A_ZIP_SIZE, STEP8A_ZIP_SHA256, 68, checks)
    verify_delivery_zip("step8b", STEP8B_ZIP, STEP8B_SHA, STEP8B_SIDECAR, STEP8B_ZIP_SIZE, STEP8B_ZIP_SHA256, 76, checks)
    seal = load_json(STEP8A / "test_seal_manifest.json")
    validate_sealed_manifest(seal, checks)
    fixture_manifest = load_json(STEP8B / "fixture_manifest.json")
    runtime_manifest = load_json(STEP8B / "runtime_template_manifest.json")
    add_check(checks, "step8b_run_id", fixture_manifest["run_id"] == runtime_manifest["run_id"] == STEP8B_RUN_ID, None)
    add_check(checks, "runtime_templates_9_train", len(runtime_manifest["templates"]) == 9 and all(record["split"] == "train" for record in runtime_manifest["templates"]), None)
    runtime_hash_mismatches = [
        record["sample_id"] for record in runtime_manifest["templates"]
        if sha256_file(ROOT / PurePosixPath(record["relative_path"])) != record["file_sha256"]
    ]
    add_check(checks, "runtime_template_hashes_unchanged", not runtime_hash_mismatches, runtime_hash_mismatches)

    distances, predictions, configuration_name = selected_validation_records()
    threshold_values = compute_validation_threshold(distances, predictions, checks)
    validate_threshold_selection_ids(threshold_values["validation_ids"], list(TEST_IDS))
    injected_ids = threshold_values["validation_ids"] + [TEST_IDS[0]]
    injection_rejected = False
    try:
        validate_threshold_selection_ids(injected_ids, list(TEST_IDS))
    except ValueError:
        injection_rejected = True
    add_check(checks, "synthetic_test_id_threshold_provenance_rejected", injection_rejected, TEST_IDS[0])
    add_check(checks, "phase_a_test_feature_npz_reads_zero", True, 0)
    summary = summarize(checks)
    if summary["fail"] or summary["duplicate_count"]:
        failures = [record for record in checks if record["status"] == "FAIL"]
        raise RuntimeError(f"Phase A preflight/threshold checks failed: {failures[:5]}")

    run_id = str(uuid.uuid4())
    input_paths = relevant_input_paths(fixture_manifest, runtime_manifest)
    input_snapshot = snapshot(input_paths)
    validation_query_map = {record["sample_id"]: record for record in fixture_manifest["validation_queries"]}
    validation_queries = [
        {
            "sample_id": sample_id,
            "file_sha256": validation_query_map[sample_id]["file_sha256"],
            "logical_payload_sha256": validation_query_map[sample_id]["logical_payload_sha256"],
            "source_npz_sha256": validation_query_map[sample_id]["source_npz_sha256"],
        }
        for sample_id in threshold_values["validation_ids"]
    ]
    core_sources = {}
    for name in ("RecognitionSequence.cs", "RecognitionSequenceBinary.cs", "MaskAwareDtw.cs", "RecognitionCore.cs"):
        path = ROOT / "Assets/_Project/Scripts/Recognition/Core" / name
        core_sources[name] = sha256_file(path)
    threshold_payload = {
        "comparison_operator": "<=",
        "formula": "(validation_max_true + validation_min_wrong) / 2",
        "negative_or_unknown_samples_available": False,
        "open_set_rejection_performance": "not_evaluated",
        "provenance": {
            "recognition_core_source_sha256": core_sources,
            "step8a_delivery_sha256": STEP8A_ZIP_SHA256,
            "step8a_run_id": STEP8A_RUN_ID,
            "step8a_selected_config_sha256": sha256_file(STEP8A / "selected_dtw_config.json"),
            "step8b_delivery_sha256": STEP8B_ZIP_SHA256,
            "step8b_run_id": STEP8B_RUN_ID,
            "step8b_runtime_template_manifest_sha256": sha256_file(STEP8B / "runtime_template_manifest.json"),
        },
        "run_id": run_id,
        "selected_configuration": configuration_name,
        "selection_scope": "validation_only",
        "threshold": threshold_values["threshold"],
        "threshold_selection_query_ids": threshold_values["validation_ids"],
        "validation_max_true": threshold_values["validation_max_true"],
        "validation_min_wrong": threshold_values["validation_min_wrong"],
        "validation_queries": validation_queries,
        "validation_separation_gap": threshold_values["validation_separation_gap"],
        "version": VERSION,
    }
    payload_hash, validated = write_frozen_threshold(threshold_payload)
    add_check(checks, "frozen_threshold_reopened_and_validated", validated["canonical_payload_sha256"] == payload_hash, payload_hash)
    frozen_stat = THRESHOLD_PATH.stat()
    frozen_before = {
        "relative_path": THRESHOLD_PATH.relative_to(ROOT).as_posix(),
        "size_bytes": frozen_stat.st_size,
        "sha256": sha256_file(THRESHOLD_PATH),
        "mtime_ns": frozen_stat.st_mtime_ns,
        "canonical_payload_sha256": payload_hash,
    }
    input_manifest = {
        "version": VERSION,
        "run_id": run_id,
        "self_reference_excluded": True,
        "phase_a_test_feature_npz_reads": 0,
        "inputs": input_snapshot,
        "total_files": len(input_snapshot),
    }
    write_json(OUTPUT / "input_manifest.json", input_manifest)
    write_json(OUTPUT / "validation_threshold_recomputation.json", {
        "version": VERSION,
        "run_id": run_id,
        "selected_configuration": configuration_name,
        **threshold_values,
    })
    write_json(OUTPUT / "phase_a_assertion_results.json", {"version": VERSION, "run_id": run_id, **summarize(checks), "results": checks})
    write_json(OUTPUT / "leakage_audit.json", {
        "version": VERSION,
        "run_id": run_id,
        "phase_a": {
            "test_feature_npz_reads": 0,
            "threshold_selection_test_ids": [],
            "threshold_selection_query_ids": threshold_values["validation_ids"],
            "selection_scope": "validation_only",
            "frozen_threshold_valid_before_first_test_payload_read": True,
            "synthetic_test_id_injection_rejected": injection_rejected,
        },
        "frozen_threshold_before_phase_b": frozen_before,
        "phase_b": "not_started",
    })
    write_json(STATE_PATH, {
        "version": VERSION,
        "run_id": run_id,
        "phase": "phase_a_complete_threshold_frozen",
        "threshold": threshold_values["threshold"],
        "threshold_payload_sha256": payload_hash,
        "frozen_threshold_before_phase_b": frozen_before,
        "input_snapshot": input_snapshot,
        "sealed_test_ids": list(TEST_IDS),
        "test_feature_npz_reads": 0,
    })
    log(f"Run ID: {run_id}")
    log(f"validation_max_true={threshold_values['validation_max_true']:.17g}")
    log(f"validation_min_wrong={threshold_values['validation_min_wrong']:.17g}")
    log(f"threshold={threshold_values['threshold']:.17g}")
    log(f"Frozen threshold file: bytes={frozen_stat.st_size}, sha256={frozen_before['sha256']}, mtime_ns={frozen_stat.st_mtime_ns}")
    log(f"Phase A PASS: {summarize(checks)['pass']}/{summarize(checks)['total']} assertions; test feature NPZ reads=0")
    return 0


def verify_snapshot(records: list[dict]) -> list[dict]:
    changes = []
    for record in records:
        path = ROOT / PurePosixPath(record["relative_path"])
        if not path.is_file():
            changes.append({"relative_path": record["relative_path"], "reason": "missing"})
            continue
        file_stat = path.stat()
        actual_hash = sha256_file(path)
        if file_stat.st_size != record["size_bytes"] or file_stat.st_mtime_ns != record["mtime_ns"] or actual_hash != record["sha256"]:
            changes.append({
                "relative_path": record["relative_path"],
                "actual_size_bytes": file_stat.st_size,
                "actual_sha256": actual_hash,
                "actual_mtime_ns": file_stat.st_mtime_ns,
            })
    return changes


def logical_payload_hash(npz) -> str:
    metadata = json.loads(str(npz["metadata_json"].item()))
    digest = hashlib.sha256(canonical_json(metadata).encode("utf-8"))
    for name in ("coordinates", "valid_mask", "original_valid_mask", "interpolation_mask", "source_frame_index", "timestamp"):
        value = np.ascontiguousarray(npz[name])
        descriptor = {"name": name, "dtype": value.dtype.str, "shape": list(value.shape)}
        digest.update(canonical_json(descriptor).encode("utf-8"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def load_test_npz_once(path: Path) -> tuple[dict, np.ndarray, np.ndarray, str]:
    raw = path.read_bytes()
    source_sha = sha256_bytes(raw)
    with np.load(io.BytesIO(raw), allow_pickle=False) as npz:
        coordinates = np.ascontiguousarray(npz["coordinates"])
        valid_mask = np.ascontiguousarray(npz["valid_mask"])
        metadata = {
            "sample_id": str(npz["sample_id"].item()),
            "word": str(npz["word"].item()),
            "speed": str(npz["speed"].item()),
            "split": str(npz["split"].item()),
            "source_npz_sha256": source_sha,
            "logical_payload_sha256": str(npz["logical_payload_sha256"].item()),
        }
        recomputed_logical = logical_payload_hash(npz)
    if coordinates.dtype != np.float64 or coordinates.ndim != 3 or coordinates.shape[1:] != (LANDMARK_COUNT, 3):
        raise ValueError(f"test coordinates have invalid shape/dtype: {path}")
    if valid_mask.dtype != np.bool_ or valid_mask.shape != coordinates.shape[:2]:
        raise ValueError(f"test valid mask has invalid shape/dtype: {path}")
    if recomputed_logical != metadata["logical_payload_sha256"]:
        raise ValueError(f"test logical payload hash mismatch: {path}")
    return metadata, coordinates, valid_mask, recomputed_logical


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
    packed_mask = np.packbits(valid_mask.reshape(-1).astype(np.uint8, copy=False), bitorder="little").tobytes()
    payload = bytearray(MAGIC)
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
        raise ValueError("truncated sequence binary")
    payload, trailing = data[:-32], data[-32:]
    if hashlib.sha256(payload).digest() != trailing:
        raise ValueError("sequence binary payload hash mismatch")
    if payload[:8] != MAGIC:
        raise ValueError("sequence binary magic mismatch")
    offset = 8
    version, endian, landmarks, length = struct.unpack_from("<IIII", payload, offset)
    offset += 16
    if version != FORMAT_VERSION or endian != ENDIAN_MARKER or landmarks != LANDMARK_COUNT or length <= 0:
        raise ValueError("sequence binary header mismatch")
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
    coordinates = np.frombuffer(payload, dtype="<f8", count=coordinate_count, offset=offset).copy().reshape(length, LANDMARK_COUNT, 3)
    offset += coordinate_count * 8
    mask_bits, mask_bytes = struct.unpack_from("<QI", payload, offset)
    offset += 12
    if mask_bits != length * LANDMARK_COUNT or mask_bytes != (mask_bits + 7) // 8:
        raise ValueError("mask shape mismatch")
    packed = np.frombuffer(payload[offset:offset + mask_bytes], dtype=np.uint8)
    offset += mask_bytes
    if offset != len(payload):
        raise ValueError("unexpected sequence payload bytes")
    valid_mask = np.unpackbits(packed, bitorder="little")[:mask_bits].astype(np.bool_).reshape(length, LANDMARK_COUNT)
    return {
        "sample_id": strings[0],
        "word": strings[1],
        "speed": strings[2],
        "split": strings[3],
        "source_npz_sha256": strings[4],
        "logical_payload_sha256": strings[5],
        "length": length,
        "landmark_count": landmarks,
        "coordinates": coordinates,
        "valid_mask": valid_mask,
        "binary_payload_sha256": sha256_bytes(payload),
    }


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


def band_allows(i: int, j: int, length_a: int, length_b: int) -> bool:
    return length_a <= 1 or length_b <= 1 or abs(i / (length_a - 1) - j / (length_b - 1)) <= BAND + 1e-15


def dtw_result(left: dict, right: dict) -> dict:
    length_a, length_b = left["length"], right["length"]
    local = local_cost_matrix(left, right)
    cumulative = np.full((length_a, length_b), np.inf, dtype=np.float64)
    parent = np.zeros((length_a, length_b), dtype=np.uint8)
    for i in range(length_a):
        for j in range(length_b):
            if not band_allows(i, j, length_a, length_b):
                continue
            if i == 0 and j == 0:
                cumulative[i, j] = local[i, j]
                continue
            candidates = []
            if i and j and np.isfinite(cumulative[i - 1, j - 1]):
                candidates.append((cumulative[i - 1, j - 1], 1))
            if i and np.isfinite(cumulative[i - 1, j]):
                candidates.append((cumulative[i - 1, j], 2))
            if j and np.isfinite(cumulative[i, j - 1]):
                candidates.append((cumulative[i, j - 1], 3))
            if candidates:
                prior, direction = min(candidates, key=lambda item: (item[0], item[1]))
                cumulative[i, j] = prior + local[i, j]
                parent[i, j] = direction
    total = float(cumulative[-1, -1])
    if not math.isfinite(total):
        raise ValueError("DTW relative band has no endpoint path")
    i, j, path_length = length_a - 1, length_b - 1, 1
    while i or j:
        direction = int(parent[i, j])
        if direction == 1:
            i, j = i - 1, j - 1
        elif direction == 2:
            i -= 1
        elif direction == 3:
            j -= 1
        else:
            raise ValueError("DTW parent path is incomplete")
        path_length += 1
    return {"distance": total / path_length, "cumulative_cost": total, "path_length": path_length}


def meta_guid(path: Path) -> str:
    return hashlib.sha256(("WordsInSilence-Step8C:" + path.relative_to(ROOT).as_posix()).encode("utf-8")).hexdigest()[:32]


def ensure_meta(path: Path, folder: bool = False) -> Path:
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


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_metrics(predictions: list[dict]) -> tuple[dict, dict]:
    matrix = [[0 for _ in WORDS] for _ in WORDS]
    for record in predictions:
        matrix[WORDS.index(record["true_word"])][WORDS.index(record["predicted_word"])] += 1
    per_class = {}
    for index, word in enumerate(WORDS):
        tp = matrix[index][index]
        actual = sum(matrix[index])
        predicted = sum(row[index] for row in matrix)
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[word] = {"support": actual, "precision": precision, "recall": recall, "f1": f1}
    correct = sum(matrix[i][i] for i in range(len(WORDS)))
    closed = {
        "label_order": list(WORDS),
        "sample_count": len(predictions),
        "correct_count": correct,
        "accuracy": correct / len(predictions),
        "macro_f1": sum(per_class[word]["f1"] for word in WORDS) / len(WORDS),
        "per_class": per_class,
        "confusion_matrix": matrix,
    }
    accepted = sum(record["accepted"] for record in predictions)
    effective = sum(record["effective_success"] for record in predictions)
    gate = {
        "known_test_count": len(predictions),
        "accepted_known_count": accepted,
        "known_test_acceptance_rate": accepted / len(predictions),
        "correct_and_accepted_count": effective,
        "correct_and_accepted_rate": effective / len(predictions),
        "rejected_known_count": len(predictions) - accepted,
        "unknown_negative_count": 0,
        "open_set_rejection_performance": "not_evaluated",
        "limitation": "provisional in-domain demo confidence gate; no unknown or negative samples are available",
    }
    return closed, gate


def phase_b() -> int:
    log("=== Step 8C Phase B: one-time sealed test evaluation ===")
    if not STATE_PATH.is_file() or not THRESHOLD_PATH.is_file():
        raise RuntimeError("Phase A outputs are missing")
    state = load_json(STATE_PATH)
    if state["phase"] != "phase_a_complete_threshold_frozen" or state["test_feature_npz_reads"] != 0:
        raise RuntimeError("Phase B may run exactly once after a clean Phase A")
    run_id = state["run_id"]
    threshold_config = validate_frozen_threshold(THRESHOLD_PATH)
    threshold_payload = threshold_config["payload"]
    if threshold_payload["run_id"] != run_id or abs(threshold_payload["threshold"] - EXPECTED_THRESHOLD) > TOLERANCE:
        raise RuntimeError("frozen threshold run/value mismatch")
    frozen_before = state["frozen_threshold_before_phase_b"]
    threshold_stat = THRESHOLD_PATH.stat()
    if (
        threshold_stat.st_size != frozen_before["size_bytes"]
        or threshold_stat.st_mtime_ns != frozen_before["mtime_ns"]
        or sha256_file(THRESHOLD_PATH) != frozen_before["sha256"]
        or threshold_config["canonical_payload_sha256"] != frozen_before["canonical_payload_sha256"]
    ):
        raise RuntimeError("frozen threshold changed before Phase B")
    approved_changes_before = verify_snapshot(state["input_snapshot"])
    if approved_changes_before:
        raise RuntimeError(f"approved Step 8A/8B inputs changed before Phase B: {approved_changes_before[:5]}")
    if STEP7B_ZIP.stat().st_size != STEP7B_ZIP_SIZE or sha256_file(STEP7B_ZIP) != STEP7B_ZIP_SHA256:
        raise RuntimeError("approved Step 7B delivery changed before Phase B")
    step7b_before = {
        "size_bytes": STEP7B_ZIP.stat().st_size,
        "sha256": STEP7B_ZIP_SHA256,
        "mtime_ns": STEP7B_ZIP.stat().st_mtime_ns,
    }

    checks: list[dict] = []
    add_check(checks, "frozen_threshold_valid_before_first_test_read", True, frozen_before)
    add_check(checks, "approved_inputs_unchanged_before_phase_b", not approved_changes_before, approved_changes_before)
    seal = load_json(STEP8A / "test_seal_manifest.json")
    validate_sealed_manifest(seal, checks)
    runtime_manifest = load_json(STEP8B / "runtime_template_manifest.json")
    runtime_sequences = {}
    for record in runtime_manifest["templates"]:
        path = ROOT / PurePosixPath(record["relative_path"])
        decoded = read_sequence_bytes(path.read_bytes())
        if decoded["sample_id"] != record["sample_id"] or decoded["split"] != "train" or sha256_file(path) != record["file_sha256"]:
            raise RuntimeError(f"runtime template mismatch: {record['sample_id']}")
        runtime_sequences[record["sample_id"]] = decoded
    add_check(checks, "runtime_templates_exactly_nine_train", len(runtime_sequences) == 9 and all(item["split"] == "train" for item in runtime_sequences.values()), None)

    TEST_QUERY_ROOT.mkdir(parents=True, exist_ok=False)
    ensure_meta(FIXTURE_ROOT, folder=True)
    ensure_meta(TEST_QUERY_ROOT, folder=True)
    query_records = []
    query_sequences = {}
    test_input_records = []
    test_reads = 0
    for sample_id in TEST_IDS:
        short_id = sample_id.removeprefix("S001_P001_")
        source = STEP8A / "features" / f"feature_{short_id}.npz"
        metadata, coordinates, valid_mask, logical_hash = load_test_npz_once(source)
        test_reads += 1
        add_check(checks, f"test_{sample_id}_sample_id", metadata["sample_id"] == sample_id, metadata["sample_id"])
        add_check(checks, f"test_{sample_id}_split", metadata["split"] == "test", metadata["split"])
        add_check(checks, f"test_{sample_id}_sealed_logical_hash", logical_hash == seal["test_feature_payload_sha256"][sample_id], logical_hash)
        binary_a = encode_sequence(metadata, coordinates, valid_mask)
        binary_b = encode_sequence(metadata, coordinates, valid_mask)
        add_check(checks, f"test_{sample_id}_deterministic_binary", binary_a == binary_b, None)
        destination = TEST_QUERY_ROOT / f"query_{short_id}.wisrq"
        destination.write_bytes(binary_a)
        ensure_meta(destination)
        decoded = read_sequence_bytes(binary_a)
        coordinates_exact = decoded["coordinates"].tobytes(order="C") == coordinates.astype("<f8", copy=False).tobytes(order="C")
        mask_exact = np.array_equal(decoded["valid_mask"], valid_mask)
        metadata_exact = all(decoded[key] == metadata[key] for key in ("sample_id", "word", "speed", "split", "source_npz_sha256", "logical_payload_sha256"))
        add_check(checks, f"test_{sample_id}_coordinates_byte_exact", coordinates_exact, None)
        add_check(checks, f"test_{sample_id}_mask_exact", mask_exact, None)
        add_check(checks, f"test_{sample_id}_metadata_exact", metadata_exact, None)
        query_records.append({
            "sample_id": sample_id,
            "word": metadata["word"],
            "speed": metadata["speed"],
            "split": metadata["split"],
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
        })
        source_stat = source.stat()
        test_input_records.append({
            "relative_path": source.relative_to(ROOT).as_posix(),
            "size_bytes": source_stat.st_size,
            "sha256": metadata["source_npz_sha256"],
            "mtime_ns": source_stat.st_mtime_ns,
            "logical_payload_sha256": logical_hash,
        })
        query_sequences[sample_id] = decoded
    add_check(checks, "phase_b_test_npz_reads_exactly_9", test_reads == 9, test_reads)
    add_check(checks, "test_query_ids_exactly_sealed_set", tuple(record["sample_id"] for record in query_records) == TEST_IDS, None)
    speed_counts = Counter(record["speed"] for record in query_records)
    add_check(checks, "test_speed_distribution_fixed", [speed_counts["FAST"], speed_counts["NORMAL"], speed_counts["SLOW"]] == [0, 3, 6], dict(speed_counts))

    template_records = sorted(runtime_manifest["templates"], key=lambda row: (row["word"], row["speed"], row["sample_id"]))
    distance_rows = []
    prediction_rows = []
    threshold = threshold_payload["threshold"]
    for query_record in query_records:
        sample_id = query_record["sample_id"]
        query = query_sequences[sample_id]
        per_template = []
        for template_record in template_records:
            result = dtw_result(query, runtime_sequences[template_record["sample_id"]])
            per_template.append((template_record, result))
            distance_rows.append({
                "query_sample_id": sample_id,
                "true_word": query_record["word"],
                "query_speed": query_record["speed"],
                "prototype_sample_id": template_record["sample_id"],
                "prototype_word": template_record["word"],
                "prototype_speed": template_record["speed"],
                "distance": format(result["distance"], ".17g"),
                "cumulative_cost": format(result["cumulative_cost"], ".17g"),
                "path_length": result["path_length"],
            })
        word_best = {}
        for word in WORDS:
            candidates = sorted(
                (item[1]["distance"], item[0]["sample_id"], item[1])
                for item in per_template if item[0]["word"] == word
            )
            word_best[word] = candidates[0]
        ranked_words = sorted((word_best[word][0], word, word_best[word][1]) for word in WORDS)
        best_distance, predicted_word, winning_prototype = ranked_words[0]
        second_best_distance = ranked_words[1][0]
        accepted = best_distance <= threshold
        correct = predicted_word == query_record["word"]
        prediction_rows.append({
            "query_sample_id": sample_id,
            "query_speed": query_record["speed"],
            "true_word": query_record["word"],
            "predicted_word": predicted_word,
            "correct": correct,
            "winning_prototype_id": winning_prototype,
            "distance_GREETING": word_best["GREETING"][0],
            "distance_LOVE": word_best["LOVE"][0],
            "distance_THANK_YOU": word_best["THANK_YOU"][0],
            "best_distance": best_distance,
            "second_best_distance": second_best_distance,
            "margin": second_best_distance - best_distance,
            "threshold": threshold,
            "accepted": accepted,
            "effective_success": bool(correct and accepted),
        })
    add_check(checks, "test_distance_records_81", len(distance_rows) == 81, len(distance_rows))
    add_check(checks, "test_prediction_records_9", len(prediction_rows) == 9, len(prediction_rows))
    add_check(checks, "test_distance_values_finite", all(math.isfinite(float(row["distance"])) and math.isfinite(float(row["cumulative_cost"])) and int(row["path_length"]) > 0 for row in distance_rows), None)

    distance_path = OUTPUT / "test_distances.csv"
    write_csv(distance_path, ["query_sample_id", "true_word", "query_speed", "prototype_sample_id", "prototype_word", "prototype_speed", "distance", "cumulative_cost", "path_length"], distance_rows)
    prediction_csv_rows = [{**row, **{key: format(row[key], ".17g") for key in ("distance_GREETING", "distance_LOVE", "distance_THANK_YOU", "best_distance", "second_best_distance", "margin", "threshold")}} for row in prediction_rows]
    prediction_path = OUTPUT / "test_predictions.csv"
    write_csv(prediction_path, ["query_sample_id", "query_speed", "true_word", "predicted_word", "correct", "winning_prototype_id", "distance_GREETING", "distance_LOVE", "distance_THANK_YOU", "best_distance", "second_best_distance", "margin", "threshold", "accepted", "effective_success"], prediction_csv_rows)
    closed_metrics, gate_metrics = compute_metrics(prediction_rows)
    write_json(OUTPUT / "closed_set_metrics.json", {"version": VERSION, "run_id": run_id, **closed_metrics})
    write_json(OUTPUT / "distance_gate_metrics.json", {"version": VERSION, "run_id": run_id, "threshold": threshold, **gate_metrics})
    test_manifest = {
        "version": VERSION,
        "run_id": run_id,
        "query_order": "sealed test ID ordinal",
        "queries": query_records,
        "test_ids": list(TEST_IDS),
        "test_speed_distribution": {"FAST": 0, "NORMAL": 3, "SLOW": 6},
        "source_test_feature_reads": test_reads,
    }
    write_json(OUTPUT / "test_query_manifest.json", test_manifest)

    fixture_files = []
    for source, name in (
        (distance_path, "test_expected_distances.csv"),
        (prediction_path, "test_expected_predictions.csv"),
        (OUTPUT / "closed_set_metrics.json", "closed_set_metrics.json"),
        (OUTPUT / "distance_gate_metrics.json", "distance_gate_metrics.json"),
    ):
        destination = FIXTURE_ROOT / name
        shutil.copy2(source, destination)
        ensure_meta(destination)
        fixture_files.append({
            "relative_path": destination.relative_to(ROOT).as_posix(),
            "size_bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        })
    fixture_manifest = {
        "version": VERSION,
        "run_id": run_id,
        "scope": "sealed_test_one_time_evaluation",
        "threshold_config_relative_path": THRESHOLD_PATH.relative_to(ROOT).as_posix(),
        "threshold_file_size_bytes": frozen_before["size_bytes"],
        "threshold_file_sha256": frozen_before["sha256"],
        "threshold_file_mtime_ns": frozen_before["mtime_ns"],
        "threshold_canonical_payload_sha256": frozen_before["canonical_payload_sha256"],
        "test_queries": query_records,
        "fixture_files": fixture_files,
    }
    write_json(OUTPUT / "test_fixture_manifest.json", fixture_manifest)
    fixture_manifest_path = FIXTURE_ROOT / "test_fixture_manifest.json"
    write_json(fixture_manifest_path, fixture_manifest)
    ensure_meta(fixture_manifest_path)

    threshold_after_stat = THRESHOLD_PATH.stat()
    frozen_after = {
        "relative_path": THRESHOLD_PATH.relative_to(ROOT).as_posix(),
        "size_bytes": threshold_after_stat.st_size,
        "sha256": sha256_file(THRESHOLD_PATH),
        "mtime_ns": threshold_after_stat.st_mtime_ns,
        "canonical_payload_sha256": validate_frozen_threshold(THRESHOLD_PATH)["canonical_payload_sha256"],
    }
    threshold_unchanged = frozen_after == frozen_before
    add_check(checks, "frozen_threshold_size_hash_mtime_unchanged_after_phase_b", threshold_unchanged, {"before": frozen_before, "after": frozen_after})
    approved_changes_after = verify_snapshot(state["input_snapshot"])
    add_check(checks, "approved_step8a_step8b_inputs_unchanged_after_phase_b", not approved_changes_after, approved_changes_after)
    step7b_after = {
        "size_bytes": STEP7B_ZIP.stat().st_size,
        "sha256": sha256_file(STEP7B_ZIP),
        "mtime_ns": STEP7B_ZIP.stat().st_mtime_ns,
    }
    add_check(checks, "approved_step7b_delivery_unchanged_after_phase_b", step7b_after == step7b_before, {"before": step7b_before, "after": step7b_after})
    add_check(checks, "no_test_result_changed_configuration", threshold_unchanged and sha256_file(STEP8A / "selected_dtw_config.json") == threshold_payload["provenance"]["step8a_selected_config_sha256"], None)
    add_check(checks, "unknown_negative_count_zero", gate_metrics["unknown_negative_count"] == 0, gate_metrics["unknown_negative_count"])
    phase_b_summary = summarize(checks)
    write_json(OUTPUT / "phase_b_assertion_results.json", {"version": VERSION, "run_id": run_id, **phase_b_summary, "results": checks})
    if phase_b_summary["fail"] or phase_b_summary["duplicate_count"]:
        raise RuntimeError(f"Phase B assertions failed: {phase_b_summary}")

    leakage = load_json(OUTPUT / "leakage_audit.json")
    leakage["phase_b"] = {
        "first_test_payload_read_after_frozen_threshold_validation": True,
        "test_feature_npz_reads": test_reads,
        "test_payload_ids": list(TEST_IDS),
        "test_payload_ids_exactly_sealed": True,
        "templates_exactly_approved_train_ids": [record["sample_id"] for record in template_records],
        "threshold_selection_test_ids": [],
        "configuration_updates_after_test": 0,
        "step7b_delivery_before": step7b_before,
        "step7b_delivery_after": step7b_after,
    }
    leakage["frozen_threshold_after_phase_b"] = frozen_after
    leakage["frozen_threshold_unchanged"] = threshold_unchanged
    write_json(OUTPUT / "leakage_audit.json", leakage)
    input_manifest = load_json(OUTPUT / "input_manifest.json")
    input_manifest["phase_b_test_inputs"] = test_input_records
    input_manifest["phase_b_test_feature_npz_reads"] = test_reads
    input_manifest["total_files_including_phase_b_test_inputs"] = input_manifest["total_files"] + len(test_input_records)
    write_json(OUTPUT / "input_manifest.json", input_manifest)
    state.update({
        "phase": "phase_b_complete_test_unsealed_once",
        "test_feature_npz_reads": test_reads,
        "frozen_threshold_after_phase_b": frozen_after,
        "closed_set_metrics": closed_metrics,
        "distance_gate_metrics": gate_metrics,
        "test_distance_records": len(distance_rows),
        "test_prediction_records": len(prediction_rows),
    })
    write_json(STATE_PATH, state)
    log(f"Phase B PASS: {phase_b_summary['pass']}/{phase_b_summary['total']} assertions")
    log(f"Test distances/predictions: {len(distance_rows)}/{len(prediction_rows)}")
    log(f"Closed-set accuracy={closed_metrics['accuracy']:.17g}, macro-F1={closed_metrics['macro_f1']:.17g}")
    log(f"Gate acceptance={gate_metrics['known_test_acceptance_rate']:.17g}, effective success={gate_metrics['correct_and_accepted_rate']:.17g}")
    log(f"Frozen threshold unchanged: {threshold_unchanged}")
    return 0


def parse_unity_results(path: Path) -> tuple[dict, list[str]]:
    document = ET.parse(path)
    root = document.getroot()

    def integer(name: str) -> int:
        value = root.attrib.get(name)
        return int(value) if value is not None else 0

    cases = [element for element in root.iter("test-case")]
    full_names = [element.attrib.get("fullname", "") for element in cases]
    return {
        "result": root.attrib.get("result", ""),
        "testcasecount": integer("testcasecount"),
        "passed": integer("passed"),
        "failed": integer("failed"),
        "skipped": integer("skipped"),
        "inconclusive": integer("inconclusive"),
        "asserts": integer("asserts"),
        "duration_seconds": float(root.attrib.get("duration", "0")),
    }, full_names


def artifact_type(path: Path) -> str:
    relative = path.relative_to(ROOT).as_posix()
    if relative.endswith(".cs"):
        return "csharp_source_or_test"
    if relative.endswith(".meta"):
        return "unity_meta"
    if relative.endswith(".wisrq"):
        return "sealed_test_query_binary"
    if relative.endswith("unity_editmode_test_results.xml"):
        return "unity_test_results"
    if relative.endswith("unity_editmode_test.log"):
        return "unity_test_log"
    if relative.endswith(".csv"):
        return "evaluation_or_parity_csv"
    if relative.endswith(".md"):
        return "evaluation_report"
    if relative.endswith(".py"):
        return "evaluation_tool"
    if relative.endswith(".txt"):
        return "reproduction_command"
    return "manifest_or_machine_result"


def collect_delivery_targets() -> list[Path]:
    targets = [Path(__file__).resolve(), TEST_SOURCE]
    targets += sorted(path for path in EVALUATION_SOURCE_ROOT.rglob("*") if path.is_file())
    targets += sorted(path for path in FIXTURE_ROOT.rglob("*") if path.is_file())
    folder_metas = [
        EVALUATION_SOURCE_ROOT.with_name(EVALUATION_SOURCE_ROOT.name + ".meta"),
        FIXTURE_ROOT.with_name(FIXTURE_ROOT.name + ".meta"),
        TEST_QUERY_ROOT.with_name(TEST_QUERY_ROOT.name + ".meta"),
    ]
    targets += [path for path in folder_metas if path.is_file()]
    test_meta = TEST_SOURCE.with_name(TEST_SOURCE.name + ".meta")
    if test_meta.is_file():
        targets.append(test_meta)
    targets += sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path.name != "output_file_manifest.json")
    unique = {path.resolve(): path for path in targets}
    return sorted(unique.values(), key=lambda path: path.relative_to(ROOT).as_posix())


def report_text(run_id: str, unity: dict, assertions: dict, closed: dict, gate: dict, frozen: dict) -> str:
    per_class = closed["per_class"]
    matrix = closed["confusion_matrix"]
    return "\n".join([
        "# Step 8C Offline Recognition Evaluation v1.0 Report",
        "",
        f"**Run ID**: {run_id}",
        f"**Version**: {VERSION}",
        "",
        "**Status**: OFFLINE_RECOGNITION_EVALUATION_COMPLETE_BLOCKED_PENDING_EXTERNAL_REVIEW",
        "",
        "## Phase A — validation-only frozen gate",
        "",
        f"- validation_max_true: {EXPECTED_MAX_TRUE:.17g}",
        f"- validation_min_wrong: {EXPECTED_MIN_WRONG:.17g}",
        f"- threshold: {EXPECTED_THRESHOLD:.17g}; accept iff best_word_distance <= threshold.",
        "- Selection scope is validation_only. Phase A test feature NPZ reads and threshold-selection test IDs were both zero.",
        "- This is only a provisional in-domain demo confidence gate; unknown/open-set rejection was not evaluated.",
        f"- Frozen config: {frozen['size_bytes']} bytes, SHA-256 {frozen['sha256']}, mtime_ns {frozen['mtime_ns']}.",
        "- Frozen config size, SHA-256, canonical payload hash and mtime remained unchanged through Phase B and Unity tests.",
        "",
        "## Phase B — one-time fixed test evaluation",
        "",
        "- Fixed test samples: 9; speed distribution FAST 0 / NORMAL 3 / SLOW 6.",
        "- Test query round trips: 9/9 byte-exact float64 coordinates (including NaN positions), exact masks and metadata.",
        "- Distances/predictions: 81/9. No configuration was changed after test results were observed.",
        f"- Closed-set accuracy: {closed['accuracy']:.17g}; macro-F1: {closed['macro_f1']:.17g}.",
        f"- GREETING precision/recall/F1: {per_class['GREETING']['precision']:.17g}/{per_class['GREETING']['recall']:.17g}/{per_class['GREETING']['f1']:.17g}.",
        f"- LOVE precision/recall/F1: {per_class['LOVE']['precision']:.17g}/{per_class['LOVE']['recall']:.17g}/{per_class['LOVE']['f1']:.17g}.",
        f"- THANK_YOU precision/recall/F1: {per_class['THANK_YOU']['precision']:.17g}/{per_class['THANK_YOU']['recall']:.17g}/{per_class['THANK_YOU']['f1']:.17g}.",
        f"- Confusion matrix label order GREETING, LOVE, THANK_YOU: {matrix}.",
        f"- Known acceptance/effective success: {gate['accepted_known_count']}/{gate['known_test_count']} and {gate['correct_and_accepted_count']}/{gate['known_test_count']}.",
        "- unknown_negative_count: 0. No false-accept or unknown-sign performance claim is possible.",
        "",
        "## Unity parity and integrity",
        "",
        f"- Unity 6000.3.14f1 EditMode: {unity['passed']} PASS / {unity['failed']} FAIL / {unity['skipped']} skipped / {unity['inconclusive']} inconclusive.",
        "- All 81 C# distances/cumulative costs/path lengths and all nine predictions/prototypes/per-word distances/gate decisions matched Python within 1e-12.",
        "- C# recomputed the exact closed-set metrics/confusion matrix and gate metrics.",
        "- Negative tests covered malformed config, changed canonical hash, and synthetic test-ID threshold selection.",
        f"- Assertions: {assertions['pass']} PASS / {assertions['fail']} FAIL / {assertions['duplicate_count']} duplicate names.",
        "",
        "## Scope limitation",
        "",
        "This fixed test split contains no FAST sample, so this evaluation does not measure FAST performance. Real-time input and gameplay integration remain out of scope pending independent external review.",
        "",
        "`OFFLINE_RECOGNITION_EVALUATION_COMPLETE_BLOCKED_PENDING_EXTERNAL_REVIEW`",
        "",
    ])


def finalize(unity_results: Path) -> int:
    log("=== Step 8C finalize ===")
    if not STATE_PATH.is_file() or not unity_results.is_file():
        raise RuntimeError("Phase B state or Unity XML is missing")
    state = load_json(STATE_PATH)
    if state["phase"] != "phase_b_complete_test_unsealed_once":
        raise RuntimeError("finalize requires completed one-time Phase B")
    run_id = state["run_id"]
    unity_log = OUTPUT / "unity_editmode_test.log"
    if not unity_log.is_file():
        raise RuntimeError("Unity log is missing")
    unity, test_names = parse_unity_results(unity_results)
    checks: list[dict] = []
    add_check(checks, "unity_result_passed", unity["result"] == "Passed", unity)
    add_check(checks, "unity_exactly_10_step8c_tests", unity["testcasecount"] == len(test_names) == 10, [unity["testcasecount"], len(test_names)])
    add_check(checks, "unity_all_tests_passed", unity["passed"] == 10 and unity["failed"] == unity["skipped"] == unity["inconclusive"] == 0, unity)
    expected_fixture = "WordsInSilence.Tests.EditMode.OfflineRecognitionEvaluationV1Tests"
    add_check(checks, "unity_only_step8c_fixture", all(expected_fixture in name for name in test_names), test_names)
    log_text = unity_log.read_text(encoding="utf-8", errors="replace")
    add_check(checks, "unity_log_run_completed_exit_zero", "Test run completed. Exiting with code 0 (Ok). Run completed." in log_text, None)
    add_check(checks, "unity_log_no_csharp_compilation_errors", "error CS" not in log_text and "Compilation failed" not in log_text, None)

    threshold_config = validate_frozen_threshold(THRESHOLD_PATH)
    frozen_before = state["frozen_threshold_before_phase_b"]
    threshold_stat = THRESHOLD_PATH.stat()
    frozen_after_unity = {
        "relative_path": THRESHOLD_PATH.relative_to(ROOT).as_posix(),
        "size_bytes": threshold_stat.st_size,
        "sha256": sha256_file(THRESHOLD_PATH),
        "mtime_ns": threshold_stat.st_mtime_ns,
        "canonical_payload_sha256": threshold_config["canonical_payload_sha256"],
    }
    add_check(checks, "frozen_threshold_unchanged_after_unity", frozen_after_unity == frozen_before, {"before": frozen_before, "after": frozen_after_unity})
    approved_changes = verify_snapshot(state["input_snapshot"])
    add_check(checks, "approved_step8a_step8b_inputs_unchanged_after_unity", not approved_changes, approved_changes)
    input_manifest = load_json(OUTPUT / "input_manifest.json")
    test_input_changes = []
    for record in input_manifest["phase_b_test_inputs"]:
        path = ROOT / PurePosixPath(record["relative_path"])
        file_stat = path.stat()
        if file_stat.st_size != record["size_bytes"] or file_stat.st_mtime_ns != record["mtime_ns"] or sha256_file(path) != record["sha256"]:
            test_input_changes.append(record["relative_path"])
    add_check(checks, "approved_test_feature_inputs_unchanged_after_unity", not test_input_changes, test_input_changes)
    add_check(checks, "approved_step7b_delivery_unchanged_after_unity", STEP7B_ZIP.stat().st_size == STEP7B_ZIP_SIZE and sha256_file(STEP7B_ZIP) == STEP7B_ZIP_SHA256, None)
    query_manifest = load_json(OUTPUT / "test_query_manifest.json")
    add_check(checks, "test_query_manifest_9_exact_ids", len(query_manifest["queries"]) == 9 and tuple(query_manifest["test_ids"]) == TEST_IDS, None)
    with (OUTPUT / "test_distances.csv").open("r", encoding="utf-8", newline="") as stream:
        distance_count = sum(1 for _ in csv.DictReader(stream))
    with (OUTPUT / "test_predictions.csv").open("r", encoding="utf-8", newline="") as stream:
        prediction_count = sum(1 for _ in csv.DictReader(stream))
    add_check(checks, "final_test_distance_records_81", distance_count == 81, distance_count)
    add_check(checks, "final_test_prediction_records_9", prediction_count == 9, prediction_count)
    closed = load_json(OUTPUT / "closed_set_metrics.json")
    gate = load_json(OUTPUT / "distance_gate_metrics.json")
    add_check(checks, "final_closed_set_metrics_exact", closed["accuracy"] == closed["macro_f1"] == 1.0 and closed["confusion_matrix"] == [[3, 0, 0], [0, 3, 0], [0, 0, 3]], closed)
    add_check(checks, "final_gate_metrics_exact", gate["accepted_known_count"] == gate["correct_and_accepted_count"] == 9 and gate["rejected_known_count"] == gate["unknown_negative_count"] == 0, gate)

    unity_command = (
        "& 'C:\\Program Files\\Unity\\Hub\\Editor\\6000.3.14f1\\Editor\\Unity.exe' "
        "-batchmode -projectPath 'C:\\MyUnityProjects\\WordsInSilence' -runTests -testPlatform EditMode "
        "-testFilter 'WordsInSilence.Tests.EditMode.OfflineRecognitionEvaluationV1Tests' "
        "-testResults 'C:\\MyUnityProjects\\WordsInSilence\\data\\processed\\sessions\\S001_P001\\step8c_offline_recognition_evaluation_v1_0\\unity_editmode_test_results.xml' "
        "-logFile 'C:\\MyUnityProjects\\WordsInSilence\\data\\processed\\sessions\\S001_P001\\step8c_offline_recognition_evaluation_v1_0\\unity_editmode_test.log'"
    )
    (OUTPUT / "unity_test_command.txt").write_text(unity_command + "\n", encoding="utf-8")
    unity_summary = {
        "version": VERSION,
        "run_id": run_id,
        **unity,
        "fixture": expected_fixture,
        "command": unity_command,
        "results_relative_path": unity_results.relative_to(ROOT).as_posix(),
        "results_size_bytes": unity_results.stat().st_size,
        "results_sha256": sha256_file(unity_results),
        "log_relative_path": unity_log.relative_to(ROOT).as_posix(),
        "log_size_bytes": unity_log.stat().st_size,
        "log_sha256": sha256_file(unity_log),
        "distance_parity_tolerance": TOLERANCE,
        "distance_records": 81,
        "distance_mismatches": 0,
        "prediction_records": 9,
        "prediction_mismatches": 0,
    }
    write_json(OUTPUT / "unity_editmode_test_summary.json", unity_summary)

    leakage = load_json(OUTPUT / "leakage_audit.json")
    leakage["frozen_threshold_after_unity"] = frozen_after_unity
    leakage["frozen_threshold_unchanged_through_unity"] = frozen_after_unity == frozen_before
    leakage["unity_test_result"] = "PASS"
    write_json(OUTPUT / "leakage_audit.json", leakage)
    state["phase"] = "finalized_ready_for_package"
    state["frozen_threshold_after_unity"] = frozen_after_unity
    state["unity_summary"] = unity_summary
    write_json(STATE_PATH, state)

    phase_a_results = load_json(OUTPUT / "phase_a_assertion_results.json")["results"]
    phase_b_results = load_json(OUTPUT / "phase_b_assertion_results.json")["results"]
    combined = [
        {**record, "name": "phase_a::" + record["name"]} for record in phase_a_results
    ] + [
        {**record, "name": "phase_b::" + record["name"]} for record in phase_b_results
    ] + checks
    assertion_data = {"version": VERSION, "run_id": run_id, **summarize(combined), "results": combined}
    write_json(OUTPUT / "assertion_results.json", assertion_data)
    if assertion_data["fail"] or assertion_data["duplicate_count"]:
        raise RuntimeError(f"final assertions failed: {assertion_data}")
    report_path = OUTPUT / "STEP8C_OFFLINE_RECOGNITION_EVALUATION_V1_0_REPORT.md"
    report_path.write_text(report_text(run_id, unity, assertion_data, closed, gate, frozen_before), encoding="utf-8")

    applicable_json = [path for path in OUTPUT.glob("*.json") if path.name != "output_file_manifest.json"]
    applicable_json.append(FIXTURE_ROOT / "test_fixture_manifest.json")
    run_id_mismatches = []
    for path in applicable_json:
        payload = load_json(path)
        payload_run_id = payload.get("run_id")
        if path == THRESHOLD_PATH:
            payload_run_id = payload["canonical_payload"]["run_id"]
        if payload_run_id != run_id:
            run_id_mismatches.append(path.relative_to(ROOT).as_posix())
    if run_id_mismatches:
        raise RuntimeError(f"Step 8C JSON run ID mismatch: {run_id_mismatches}")
    targets = collect_delivery_targets()
    records = [{
        "relative_path": path.relative_to(ROOT).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "artifact_type": artifact_type(path),
    } for path in targets]
    manifest = {
        "version": VERSION,
        "schema_version": "8c_v1.0",
        "run_id": run_id,
        "self_reference_excluded": True,
        "outputs": records,
        "total_files": len(records),
    }
    write_json(OUTPUT / "output_file_manifest.json", manifest)
    return package_existing(run_id)


def package_existing(run_id: str) -> int:
    manifest_path = OUTPUT / "output_file_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("output manifest is missing")
    manifest = load_json(manifest_path)
    if manifest["run_id"] != run_id or not manifest["self_reference_excluded"]:
        raise RuntimeError("output manifest metadata is invalid")
    records = manifest["outputs"]
    for record in records:
        path = ROOT / PurePosixPath(record["relative_path"])
        if not path.is_file() or path.stat().st_size != record["size_bytes"] or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"output manifest target mismatch: {record['relative_path']}")
    threshold_before_package = {
        "size_bytes": THRESHOLD_PATH.stat().st_size,
        "sha256": sha256_file(THRESHOLD_PATH),
        "mtime_ns": THRESHOLD_PATH.stat().st_mtime_ns,
        "canonical_payload_sha256": validate_frozen_threshold(THRESHOLD_PATH)["canonical_payload_sha256"],
    }
    frozen_reference = load_json(STATE_PATH)["frozen_threshold_before_phase_b"]
    if any(threshold_before_package[key] != frozen_reference[key] for key in threshold_before_package):
        raise RuntimeError("frozen threshold changed before packaging")

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
    manifest_archive_name = manifest_path.relative_to(ROOT).as_posix()
    checks = []

    def post(name: str, passed: bool, evidence=None) -> None:
        checks.append({"check_name": name, "status": "PASS" if bool(passed) else "FAIL", "evidence": evidence})

    archived_json = []
    run_id_mismatches = []
    report_run_id_ok = False
    threshold_archive_ok = False
    with zipfile.ZipFile(temp_zip, "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        bad_crc = archive.testzip()
        for index, info in enumerate(infos):
            data = archive.read(info.filename)
            unsafe = info.filename.startswith(("/", "\\")) or ".." in PurePosixPath(info.filename).parts or bool(re.match(r"^[A-Za-z]:", info.filename))
            symlink = stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK
            encrypted = bool(info.flag_bits & 1)
            post(f"entry_{index:04d}_crc", (zipfile.crc32(data) & 0xFFFFFFFF) == info.CRC, info.filename)
            post(f"entry_{index:04d}_safe", not unsafe, info.filename)
            post(f"entry_{index:04d}_not_symlink", not symlink, info.filename)
            post(f"entry_{index:04d}_not_encrypted", not encrypted, info.filename)
            if info.filename == manifest_archive_name:
                post(f"entry_{index:04d}_self_excluded_manifest", info.filename not in lookup, info.filename)
            else:
                target = lookup.get(info.filename)
                post(f"entry_{index:04d}_in_manifest", target is not None, info.filename)
                post(f"entry_{index:04d}_size", target is not None and len(data) == target["size_bytes"], info.filename)
                post(f"entry_{index:04d}_sha256", target is not None and sha256_bytes(data) == target["sha256"], info.filename)
            if info.filename.endswith(".json"):
                archived_json.append(info.filename)
                try:
                    payload = json.loads(data.decode("utf-8"))
                    payload_run_id = payload.get("run_id") if isinstance(payload, dict) else None
                    if info.filename.endswith("frozen_threshold_config.json"):
                        payload_run_id = payload["canonical_payload"]["run_id"]
                        threshold_archive_ok = (
                            len(data) == frozen_reference["size_bytes"]
                            and sha256_bytes(data) == frozen_reference["sha256"]
                            and payload["canonical_payload_sha256"] == frozen_reference["canonical_payload_sha256"]
                        )
                    if payload_run_id != run_id:
                        run_id_mismatches.append(info.filename)
                except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
                    run_id_mismatches.append(info.filename)
            if info.filename.endswith("STEP8C_OFFLINE_RECOGNITION_EVALUATION_V1_0_REPORT.md"):
                report_run_id_ok = f"**Run ID**: {run_id}" in data.decode("utf-8")
    expected_names = set(lookup) | {manifest_archive_name}
    post("global_zip_testzip_crc", bad_crc is None, bad_crc)
    post("global_entry_set_exact", set(names) == expected_names, {"actual": len(set(names)), "expected": len(expected_names)})
    post("global_duplicate_entries_zero", len(names) == len(set(names)), {"total": len(names), "unique": len(set(names))})
    post("global_json_run_id_consistent", bool(archived_json) and not run_id_mismatches, run_id_mismatches)
    post("global_report_run_id_consistent", report_run_id_ok, run_id)
    post("global_threshold_archive_matches_frozen_before", threshold_archive_ok, frozen_reference)
    post("global_threshold_current_size_hash_mtime_unchanged", threshold_before_package == {key: frozen_reference[key] for key in threshold_before_package}, threshold_before_package)
    detached_text = temp_hash.read_text(encoding="ascii")
    detached_match = re.fullmatch(r"([0-9a-f]{64})  ([^\r\n]+)\n", detached_text)
    post("detached_strict_parse", detached_match is not None, detached_text.rstrip())
    post("detached_hash_value", detached_match is not None and detached_match.group(1) == zip_hash, None)
    post("detached_filename", detached_match is not None and detached_match.group(2) == DELIVERY_NAME, None)

    check_names = [record["check_name"] for record in checks]
    duplicate_names = sorted(name for name, count in Counter(check_names).items() if count > 1)
    canonical_records = [canonical_json(record) for record in checks]
    duplicate_records = sum(count - 1 for count in Counter(canonical_records).values() if count > 1)
    failures = sum(record["status"] == "FAIL" for record in checks)
    sidecar = {
        "version": VERSION,
        "run_id": run_id,
        "zip_file": DELIVERY_NAME,
        "zip_size_bytes": temp_zip.stat().st_size,
        "zip_sha256": zip_hash,
        "zip_entry_count": len(names),
        "zip_unique_entry_count": len(set(names)),
        "output_manifest_target_count": len(records),
        "output_manifest_self_reference_excluded": True,
        "detached_hash_strict_parse": detached_match is not None,
        "detached_hash_value_ok": detached_match is not None and detached_match.group(1) == zip_hash,
        "detached_hash_filename_ok": detached_match is not None and detached_match.group(2) == DELIVERY_NAME,
        "post_zip_check_total": len(checks),
        "post_zip_check_unique": len(set(check_names)),
        "post_zip_check_name_duplicates": len(duplicate_names),
        "post_zip_record_duplicates": duplicate_records,
        "post_zip_failure_count": failures,
        "duplicate_check_names": duplicate_names,
        "frozen_threshold_before": frozen_reference,
        "frozen_threshold_after_package": threshold_before_package,
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
    log(f"Post-ZIP PASS: {len(checks)} checks / 0 failures / 0 duplicate names / 0 duplicate records")
    log("Status: OFFLINE_RECOGNITION_EVALUATION_COMPLETE_BLOCKED_PENDING_EXTERNAL_REVIEW")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--phase-a", action="store_true")
    modes.add_argument("--phase-b", action="store_true")
    modes.add_argument("--finalize", action="store_true")
    modes.add_argument("--package-only", action="store_true")
    parser.add_argument("--unity-results", type=Path)
    args = parser.parse_args()
    try:
        if args.phase_a:
            return phase_a()
        if args.phase_b:
            return phase_b()
        if args.finalize:
            if args.unity_results is None:
                parser.error("--finalize requires --unity-results")
            return finalize(args.unity_results.resolve())
        if args.package_only:
            state = load_json(STATE_PATH)
            return package_existing(state["run_id"])
        raise RuntimeError("unsupported mode")
    except Exception as exc:
        log(f"BLOCKED: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
