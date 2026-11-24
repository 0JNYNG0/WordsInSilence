#!/usr/bin/env python3
"""Build, validate and package WordsInSilence Step 9I without invoking Unity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import stat
import subprocess
import tempfile
import types
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any
from xml.sax.saxutils import escape as xml_escape


SCHEMA = "words_in_silence.step9i.personal_runtime_opt_in_integration.v1_0"
WORDS = ("GREETING", "LOVE", "THANK_YOU")
TOLERANCE = 1e-12
STEP9H_REL = Path("Validation/WordsInSilence_step9h_personalized_class_conditional_threshold_v1_0_delivery.zip")
STEP9H_SIZE = 3_214_867
STEP9H_SHA256 = "4827e81e029e63e7939e7ed91cb33ac5f90dd6822fa13722da01f642aed2ddb9"
STEP9H_DETACHED_SIZE = 147
STEP9H_DETACHED_SHA256 = "d564d21cd5e6cf0dc1b3eda47cb88a80014710fd182d3889827b743961829c4b"
STEP9H_ANALYSIS_ENTRY = "analysis/s001_step9h_personalized_class_conditional_threshold_v1_0.py"
PROFILE_REL = Path("Assets/StreamingAssets/RecognitionCoreV1/PersonalProfiles/step9h_current_user_v1_0")
FIXTURE_REL = Path("Assets/_Project/Tests/EditMode/Fixtures/PersonalRuntimeOptInIntegrationV1")
SOURCE_REL = Path("Assets/_Project/Scripts/Recognition/Runtime/PersonalRuntimeOptInProfile.cs")
TEST_REL = Path("Assets/_Project/Tests/EditMode/PersonalRuntimeOptInIntegrationV1Tests.cs")
PROFILE_ID = "step9h_current_user_v1_0"
WARNING = "GREETING_CLASS_CONDITIONAL_MARGIN_NARROW"
THRESHOLDS = {
    "GREETING": 0.5205639017143886,
    "LOVE": 0.39564149928871895,
    "THANK_YOU": 0.38124865401356706,
}
ASSET_EXPECTED = {
    "personal_template_GREETING.wisrt": (53_564, "d8d5713334b22d9427b9b589bccf820cdbb4849357bae3dce1c368eaf0049681"),
    "personal_template_LOVE.wisrt": (56_262, "013bf5515eb8af33f300400893da64c853668b2fa0442e19203da62b2777ad4b"),
    "personal_template_THANK_YOU.wisrt": (49_319, "9bb9a7681d55a292a21d8b7f458b4b150c10555bc0ec163da692a780ed5280e5"),
    "personal_class_threshold_config.json": (4_811, "e84edd1f598c2d94f8412767f3125031e0be330886d3a315ed429798fe49726e"),
}


class Step9IError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Step9IError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_json(value: Any, path: str = "$") -> None:
    if isinstance(value, float):
        require(math.isfinite(value), f"non-finite JSON value at {path}")
    elif isinstance(value, dict):
        for key, child in value.items():
            finite_json(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            finite_json(child, f"{path}[{index}]")


def json_bytes(value: Any, *, pretty: bool = True) -> bytes:
    finite_json(value)
    if pretty:
        text = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    else:
        text = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
    return text.encode("utf-8")


def csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in fields})
    return stream.getvalue().encode("utf-8")


def safe_zip_name(name: str) -> bool:
    pure = PurePosixPath(name)
    return bool(name) and not name.startswith(("/", "\\")) and "\\" not in name and ":" not in pure.parts[0] and all(part not in ("", ".", "..") for part in pure.parts)


def deterministic_zip_bytes(files: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(files):
            require(safe_zip_name(name), f"unsafe output path: {name}")
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (0o100644 & 0xFFFF) << 16
            info.flag_bits = 0x800
            archive.writestr(info, files[name], compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return stream.getvalue()


def inspect_zip(data: bytes, expected_names: set[str]) -> dict[str, Any]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        require(len(names) == len(set(names)), "ZIP duplicate entry")
        require(archive.testzip() is None, "ZIP CRC failure")
        require(set(names) == expected_names, "ZIP membership mismatch")
        require(all(safe_zip_name(name) for name in names), "ZIP unsafe path")
        require(not any(info.flag_bits & 1 for info in infos), "ZIP encrypted entry")
        require(not any(stat.S_ISLNK((info.external_attr >> 16) & 0xFFFF) for info in infos), "ZIP symlink entry")
        return {
            "entry_count": len(names), "crc": "PASS", "duplicates": 0, "unsafe": 0,
            "encrypted": 0, "symlinks": 0, "membership": "PASS",
        }


def verify_step9h(path: Path, detached_path: Path) -> tuple[dict[str, Any], dict[str, bytes], types.ModuleType]:
    require(path.is_file(), f"fixed Step 9H ZIP missing: {path}")
    require(detached_path.is_file(), f"fixed Step 9H checksum missing: {detached_path}")
    zip_bytes = path.read_bytes()
    detached = detached_path.read_bytes()
    require(len(zip_bytes) == STEP9H_SIZE and sha256_bytes(zip_bytes) == STEP9H_SHA256, "fixed Step 9H ZIP identity mismatch")
    require(len(detached) == STEP9H_DETACHED_SIZE and sha256_bytes(detached) == STEP9H_DETACHED_SHA256, "fixed Step 9H detached checksum identity mismatch")
    require(detached.decode("ascii").strip() == STEP9H_SHA256 + "  " + path.name, "fixed Step 9H detached checksum syntax mismatch")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        require(len(names) == 53 and len(names) == len(set(names)), "fixed Step 9H entry count/duplicate mismatch")
        require(archive.testzip() is None, "fixed Step 9H CRC failure")
        require(all(safe_zip_name(name) for name in names), "fixed Step 9H unsafe path")
        require(not any(info.flag_bits & 1 for info in infos), "fixed Step 9H encrypted entry")
        require(not any(stat.S_ISLNK((info.external_attr >> 16) & 0xFFFF) for info in infos), "fixed Step 9H symlink entry")
        file_map = {info.filename: archive.read(info) for info in infos}
    manifest = json.loads(file_map["output_file_manifest.json"])
    records = manifest["outputs"]
    require(len(records) == 52 and set(file_map) == {record["relative_path"] for record in records} | {"output_file_manifest.json"}, "fixed Step 9H manifest membership mismatch")
    for record in records:
        data = file_map[record["relative_path"]]
        require(len(data) == record["size_bytes"] and sha256_bytes(data) == record["sha256"], "fixed Step 9H manifest target mismatch")
    source = file_map[STEP9H_ANALYSIS_ENTRY]
    module = types.ModuleType("approved_step9h_for_step9i")
    exec(compile(source, STEP9H_ANALYSIS_ENTRY, "exec"), module.__dict__)
    require(module.deterministic_zip_bytes(file_map) == zip_bytes, "fixed Step 9H deterministic regeneration mismatch")
    for name, (size, digest) in ASSET_EXPECTED.items():
        entry = "active_personal_templates/" + name if name.endswith(".wisrt") else name
        require(len(file_map[entry]) == size and sha256_bytes(file_map[entry]) == digest, f"fixed personal asset identity mismatch: {name}")
    return {
        "zip": {"path": path.as_posix(), "size_bytes": len(zip_bytes), "sha256": sha256_bytes(zip_bytes), "entry_count": 53, "manifest_target_count": 52},
        "detached": {"path": detached_path.as_posix(), "size_bytes": len(detached), "file_sha256": sha256_bytes(detached), "syntax_filename_digest": "PASS"},
        "archive_safety": {"crc": "PASS", "duplicates": 0, "unsafe": 0, "symlinks": 0, "encrypted": 0, "membership": "PASS"},
        "deterministic_regeneration_byte_identical": True,
    }, file_map, module


def approved_calculation(root: Path, archive_path: Path, file_map: dict[str, bytes], step9h: types.ModuleType) -> tuple[Any, Any, Any, list[Any], dict[str, Any], dict[str, Any]]:
    # Step 9H retains the approved Step 9G inputs under explicitly prefixed
    # provenance names. Supply the byte-identical aliases expected by the
    # approved Step 9H pure calculation functions without modifying the ZIP.
    calculation_map = dict(file_map)
    aliases = {
        "analysis/s001_step9g_personalized_calibration_medoid_loo_threshold_v1_0.py":
            "input_subset/provenance/s001_step9g_personalized_calibration_medoid_loo_threshold_v1_0.py",
        "input_subset/provenance/step9e_seed_capture_identity.json":
            "input_subset/provenance/step9g_step9e_seed_capture_identity.json",
        "input_subset/provenance/step9f_additional_capture_identity.json":
            "input_subset/provenance/step9g_step9f_additional_capture_identity.json",
        "capture_pairwise_dtw.csv": "input_subset/provenance/step9g_capture_pairwise_dtw.csv",
        "medoid_selection.json": "input_subset/provenance/step9g_medoid_selection.json",
        "threshold_derivation.json": "input_subset/provenance/step9g_threshold_derivation.json",
    }
    for expected, retained in aliases.items():
        require(retained in file_map, f"Step 9H retained provenance missing: {retained}")
        calculation_map[expected] = file_map[retained]
    step9g, step9e, helper, source_provenance, _ = step9h.load_approved_sources(root, calculation_map)
    captures, _, identity, _ = step9h.load_captures(step9g, step9e, archive_path, STEP9H_SHA256, calculation_map)
    _, parity, medoids, _, global_failure = step9h.reproduce_dtw_and_step9g(step9g, step9e, helper, captures, calculation_map)
    require(parity["ordered_cell_count"] == 81 and parity["approved_runtime_contract_max_absolute_error"] <= TOLERANCE, "approved Step 9H DTW parity mismatch")
    require(global_failure["step9g_actual_verdict"] == "CALIBRATION_NOT_SEPARABLE", "Step 9G failure was reclassified")
    return step9e, helper, medoids, captures, identity, source_provenance


def build_profile(file_map: dict[str, bytes]) -> tuple[dict[str, bytes], dict[str, Any]]:
    config_bytes = file_map["personal_class_threshold_config.json"]
    config = json.loads(config_bytes)
    require(config["runtime_activated"] is False and config["global_template_or_threshold_overwritten"] is False, "Step 9H config activation/global flags changed")
    require(config["thresholds"] == THRESHOLDS, "Step 9H threshold map mismatch")
    by_word = {record["word"]: record for record in config["template_records"]}
    require(set(by_word) == set(WORDS), "Step 9H template provenance word set mismatch")
    profile_files: dict[str, bytes] = {}
    assets: list[dict[str, Any]] = []
    for word in WORDS:
        source_name = f"active_personal_templates/personal_template_{word}.wisrt"
        target_name = f"personal_template_{word}.wisrt"
        payload = file_map[source_name]
        record = by_word[word]
        require(len(payload) == record["size_bytes"] and sha256_bytes(payload) == record["sha256"], f"Step 9H template/config mismatch: {word}")
        profile_files[target_name] = payload
        assets.append({
            "kind": "template", "word": word, "relative_path": target_name,
            "size_bytes": len(payload), "sha256": sha256_bytes(payload),
            "embedded_checksum_sha256": record["embedded_checksum_sha256"],
            "capture_id": record["capture_id"], "capture_key": record["capture_key"], "run_id": record["run_id"],
            "source_payload_sha256": record["source_payload_sha256"],
            "coordinate_payload_sha256": record["coordinate_payload_sha256"],
            "mask_payload_sha256": record["mask_payload_sha256"],
            "normalized_payload_sha256": record["normalized_payload_sha256"],
        })
    profile_files["personal_class_threshold_config.json"] = config_bytes
    assets.append({
        "kind": "threshold_config", "relative_path": "personal_class_threshold_config.json",
        "size_bytes": len(config_bytes), "sha256": sha256_bytes(config_bytes),
    })
    manifest = {
        "schema_version": "words_in_silence.step9i.personal_runtime_profile.v1_0",
        "profile_id": PROFILE_ID, "mode": "personal", "threshold_mode": "class_conditional_one_vs_rest",
        "opt_in_only": True, "default_active": False, "installed_but_default_inactive": True,
        "template_count": 3, "threshold_count": 3, "words": list(WORDS),
        "source_step9h_delivery": {
            "relative_path": STEP9H_REL.as_posix(), "size_bytes": STEP9H_SIZE, "sha256": STEP9H_SHA256,
        },
        "assets": assets, "global_assets_overwritten": False, "warnings": [WARNING],
    }
    profile_files["personal_runtime_profile_manifest.json"] = json_bytes(manifest)
    return profile_files, manifest


def build_personal_fixtures(step9e: Any, helper: Any, medoids: dict[str, Any], captures: list[Any]) -> tuple[dict[str, bytes], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    fixture_files: dict[str, bytes] = {}
    query_records: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    max_runtime_error = 0.0
    genuine_accept = 0
    non_target_reject = 0
    prediction_correct = 0
    quest_success = 0
    for capture in captures:
        filename = f"query_{capture.word}_{capture.capture_id}.wisrq"
        payload = step9e.write_wisrt_bytes(capture)
        parsed = step9e.read_wisrt_bytes(payload)
        require(parsed["sample_id"] == capture.capture_id and parsed["word"] == capture.word, "query WISRQ round-trip mismatch")
        fixture_files["Queries/" + filename] = payload
        query_records.append({
            "query_id": capture.capture_id, "expected_word": capture.word,
            "capture_id": capture.capture_id, "run_id": capture.run_id,
            "relative_path": (FIXTURE_REL / "Queries" / filename).as_posix(),
            "frame_count": len(capture.coordinates), "size_bytes": len(payload), "sha256": sha256_bytes(payload),
            "source_payload_sha256": capture.source_sha256,
            "normalized_payload_sha256": capture.normalized_payload_sha256,
        })
        scores: dict[str, float] = {}
        accepts: dict[str, bool] = {}
        for word in WORDS:
            prototype = medoids[word]
            reference = step9e.reference_dtw(capture.coordinates, capture.valid_mask, prototype.coordinates, prototype.valid_mask, step9e.DTW_BAND)
            runtime = helper.unity_contract_dtw(capture.coordinates, capture.valid_mask, prototype.coordinates, prototype.valid_mask, helper.DTW_BAND)
            distance = float(reference["distance"])
            runtime_error = abs(distance - float(runtime["distance"]))
            max_runtime_error = max(max_runtime_error, runtime_error)
            require(runtime_error <= TOLERANCE and int(reference["path_length"]) == int(runtime["path_length"]), "personal query runtime parity mismatch")
            accepted = distance <= THRESHOLDS[word]
            scores[word] = distance
            accepts[word] = accepted
            kind = "genuine" if capture.word == word else "impostor"
            if kind == "genuine" and accepted:
                genuine_accept += 1
            if kind == "impostor" and not accepted:
                non_target_reject += 1
            score_rows.append({
                "score_cell_key": capture.capture_id + "||" + word,
                "query_id": capture.capture_id, "expected_word": capture.word, "target_word": word,
                "expected_distance": distance, "runtime_contract_distance": float(runtime["distance"]),
                "absolute_error": runtime_error, "path_length": int(reference["path_length"]),
                "threshold": THRESHOLDS[word], "score_kind": kind, "per_class_accept": accepted,
            })
        predicted = min(WORDS, key=lambda word: (scores[word], word))
        accepted = accepts[predicted]
        expected_accept = accepts[capture.word]
        success = accepted and predicted == capture.word and expected_accept
        prediction_correct += int(predicted == capture.word)
        quest_success += int(success)
        prediction_rows.append({
            "query_id": capture.capture_id, "expected_word": capture.word, "predicted_word": predicted,
            "winning_capture_id": medoids[predicted].capture_id, "best_distance": scores[predicted],
            "accepted": accepted, "expected_word_class_accept": expected_accept, "quest_success": success,
        })
    require(len(query_records) == 9 and len(score_rows) == 27 and len(prediction_rows) == 9, "personal fixture count mismatch")
    require(prediction_correct == genuine_accept == quest_success == 9 and non_target_reject == 18, "full-medoid acceptance expectation mismatch")
    require(max_runtime_error <= TOLERANCE, "personal fixture runtime parity tolerance exceeded")
    manifest = {"schema_version": SCHEMA + ".personal_query_manifest", "query_count": 9, "queries": query_records}
    fixture_files["personal_query_manifest.json"] = json_bytes(manifest)
    fixture_files["personal_expected_scores.csv"] = csv_bytes(score_rows, [
        "score_cell_key", "query_id", "expected_word", "target_word", "expected_distance",
        "runtime_contract_distance", "absolute_error", "path_length", "threshold", "score_kind", "per_class_accept",
    ])
    fixture_files["personal_expected_predictions.csv"] = csv_bytes(prediction_rows, [
        "query_id", "expected_word", "predicted_word", "winning_capture_id", "best_distance",
        "accepted", "expected_word_class_accept", "quest_success",
    ])
    summary = {
        "query_count": 9, "score_cell_count": 27, "prediction_correct_count": prediction_correct,
        "genuine_acceptance_count": genuine_accept, "non_target_rejection_count": non_target_reject,
        "quest_success_count": quest_success, "runtime_contract_max_absolute_error": max_runtime_error,
        "tolerance": TOLERANCE,
    }
    return fixture_files, score_rows, prediction_rows, summary


def asset_snapshot(root: Path) -> dict[str, Any]:
    paths = sorted((root / "Assets/StreamingAssets/RecognitionCoreV1/Templates").glob("*.wisrt"))
    paths += sorted((root / "Assets/StreamingAssets/RecognitionCoreV1/RuntimeGate").glob("*.json"))
    require(len(paths) == 11, f"global runtime asset count mismatch: {len(paths)}")
    return {
        "schema_version": "words_in_silence.step9i.global_runtime_asset_snapshot.v1_0",
        "assets": [{
            "relative_path": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path),
        } for path in paths],
    }


def offline_global_regression(root: Path, helper: Any) -> dict[str, Any]:
    template_paths = sorted((root / "Assets/StreamingAssets/RecognitionCoreV1/Templates").glob("*.wisrt"))
    query_dir = root / "Assets/_Project/Tests/EditMode/Fixtures/RuntimeRecognitionServiceV1/TestQueries"
    expected_path = root / "Assets/_Project/Tests/EditMode/Fixtures/RuntimeRecognitionServiceV1/test_expected_predictions.csv"
    templates = [helper.read_wisrt(path) for path in template_paths]
    expected = {row["query_sample_id"]: row for row in csv.DictReader(expected_path.read_text(encoding="utf-8").splitlines())}
    rows: list[dict[str, Any]] = []
    max_error = 0.0
    correct = 0
    for path in sorted(query_dir.glob("*.wisrq")):
        query = helper.read_wisrt(path)
        distances = []
        for template in templates:
            dtw = helper.reference_dtw(query["coordinates"], query["valid_mask"], template["coordinates"], template["valid_mask"], helper.DTW_BAND)
            distances.append((template, float(dtw["distance"])))
        word_scores: dict[str, tuple[dict[str, Any], float]] = {}
        for word in WORDS:
            word_scores[word] = min((item for item in distances if item[0]["word"] == word), key=lambda item: item[1])
        predicted = min(WORDS, key=lambda word: (word_scores[word][1], word))
        record = expected[query["sample_id"]]
        errors = {
            "GREETING": abs(word_scores["GREETING"][1] - float(record["distance_GREETING"])),
            "LOVE": abs(word_scores["LOVE"][1] - float(record["distance_LOVE"])),
            "THANK_YOU": abs(word_scores["THANK_YOU"][1] - float(record["distance_THANK_YOU"])),
        }
        max_error = max(max_error, *errors.values())
        match = predicted == record["predicted_word"] and word_scores[predicted][0]["sample_id"] == record["winning_prototype_id"]
        correct += int(match)
        rows.append({"query_id": query["sample_id"], "predicted_word": predicted, "expected_predicted_word": record["predicted_word"], "winning_prototype_id": word_scores[predicted][0]["sample_id"], "match": match, "max_score_error": max(errors.values())})
    require(len(rows) == 9 and correct == 9 and max_error <= TOLERANCE, "existing global regression changed")
    return {"schema_version": SCHEMA + ".offline_global_regression", "status": "PASS", "query_count": 9, "prediction_match_count": correct, "max_absolute_error": max_error, "tolerance": TOLERANCE, "rows": rows, "unity_fixture_execution_status": "UNVERIFIED_PENDING_USER_EDITOR_TEST"}


def meta_guid(relative_path: str) -> str:
    return hashlib.sha256(("WordsInSilence-Step9I-v1.0\0" + relative_path).encode("utf-8")).hexdigest()[:32]


def meta_bytes(relative_path: str, kind: str) -> bytes:
    guid = meta_guid(relative_path)
    if kind == "folder":
        text = f"fileFormatVersion: 2\nguid: {guid}\nfolderAsset: yes\nDefaultImporter:\n  externalObjects: {{}}\n  userData: \n  assetBundleName: \n  assetBundleVariant: \n"
    elif kind == "mono":
        text = f"fileFormatVersion: 2\nguid: {guid}\nMonoImporter:\n  externalObjects: {{}}\n  serializedVersion: 2\n  defaultReferences: []\n  executionOrder: 0\n  icon: {{instanceID: 0}}\n  userData: \n  assetBundleName: \n  assetBundleVariant: \n"
    else:
        text = f"fileFormatVersion: 2\nguid: {guid}\nDefaultImporter:\n  externalObjects: {{}}\n  userData: \n  assetBundleName: \n  assetBundleVariant: \n"
    return text.encode("utf-8")


def build_install_map(profile_files: dict[str, bytes], fixture_files: dict[str, bytes], snapshot: dict[str, Any]) -> tuple[dict[str, bytes], list[str]]:
    fixture_files = dict(fixture_files)
    fixture_files["global_runtime_asset_snapshot.json"] = json_bytes(snapshot)
    files: dict[str, bytes] = {}
    folder_metas = [
        "Assets/StreamingAssets/RecognitionCoreV1/PersonalProfiles.meta",
        (PROFILE_REL.as_posix() + ".meta"),
        (FIXTURE_REL.as_posix() + ".meta"),
        (FIXTURE_REL / "Queries").as_posix() + ".meta",
    ]
    for meta in folder_metas:
        files[meta] = meta_bytes(meta, "folder")
    for name, data in profile_files.items():
        relative = (PROFILE_REL / name).as_posix()
        files[relative] = data
        files[relative + ".meta"] = meta_bytes(relative + ".meta", "default")
    for name, data in fixture_files.items():
        relative = (FIXTURE_REL / name).as_posix()
        files[relative] = data
        files[relative + ".meta"] = meta_bytes(relative + ".meta", "default")
    source_meta = SOURCE_REL.as_posix() + ".meta"
    test_meta = TEST_REL.as_posix() + ".meta"
    files[source_meta] = meta_bytes(source_meta, "mono")
    files[test_meta] = meta_bytes(test_meta, "mono")
    required_absent_dirs = [
        "Assets/StreamingAssets/RecognitionCoreV1/PersonalProfiles",
        PROFILE_REL.as_posix(), FIXTURE_REL.as_posix(), (FIXTURE_REL / "Queries").as_posix(),
    ]
    return files, required_absent_dirs


def offline_compile(root: Path) -> dict[str, Any]:
    dotnet = shutil.which("dotnet")
    sources = [
        "Assets/_Project/Scripts/Recognition/Core/RecognitionSequence.cs",
        "Assets/_Project/Scripts/Recognition/Core/RecognitionSequenceBinary.cs",
        "Assets/_Project/Scripts/Recognition/Core/MaskAwareDtw.cs",
        "Assets/_Project/Scripts/Recognition/Core/RecognitionCore.cs",
        "Assets/_Project/Scripts/Recognition/Evaluation/DistanceGatePolicy.cs",
        "Assets/_Project/Scripts/Recognition/Evaluation/FrozenDistanceGateConfig.cs",
        "Assets/_Project/Scripts/Recognition/Runtime/RuntimeRecognitionService.cs",
        SOURCE_REL.as_posix(),
    ]
    resolved = [str((root / source).resolve()) for source in sources]
    with tempfile.TemporaryDirectory(prefix="WordsInSilence-Step9I-Compile-") as temporary:
        sdk_available = False
        if dotnet is not None:
            listed = subprocess.run([dotnet, "--list-sdks"], capture_output=True, text=True, timeout=30)
            sdk_available = listed.returncode == 0 and bool(listed.stdout.strip())
        if sdk_available:
            csproj = Path(temporary) / "Step9ICompile.csproj"
            items = "\n".join(f'    <Compile Include="{xml_escape(source)}" />' for source in resolved)
            csproj.write_text(
                "<Project Sdk=\"Microsoft.NET.Sdk\">\n  <PropertyGroup>\n    <TargetFramework>net8.0</TargetFramework>\n"
                "    <EnableDefaultCompileItems>false</EnableDefaultCompileItems>\n    <Nullable>disable</Nullable>\n"
                "    <TreatWarningsAsErrors>false</TreatWarningsAsErrors>\n  </PropertyGroup>\n  <ItemGroup>\n" + items + "\n  </ItemGroup>\n</Project>\n",
                encoding="utf-8",
            )
            command = [dotnet, "build", str(csproj), "-c", "Release", "--nologo", "--verbosity", "minimal"]
            compiler = "dotnet_sdk"
            framework = "net8.0"
        else:
            csc = Path(r"C:\Program Files\Unity\Hub\Editor\6000.3.14f1\Editor\Data\MonoBleedingEdge\lib\mono\4.5\csc.exe")
            mono = Path(r"C:\Program Files\Unity\Hub\Editor\6000.3.14f1\Editor\Data\MonoBleedingEdge\bin\mono.exe")
            require(csc.is_file() and mono.is_file(), "neither dotnet SDK nor Unity Mono Roslyn compiler is available")
            output = Path(temporary) / "Step9ICompile.dll"
            command = [str(mono), str(csc), "/nologo", "/target:library", "/langversion:latest", "/out:" + str(output)] + resolved
            compiler = "unity_mono_roslyn_csc"
            framework = "Unity Mono 4.5 reference profile"
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        compiler_output = (completed.stdout or "") + (completed.stderr or "")
        require(completed.returncode == 0, "offline C# compile failed:\n" + compiler_output[-4000:])
    return {"status": "PASS", "compiler_error_count": 0, "source_count": len(sources), "target_framework": framework, "compiler": compiler, "unity_compile_equivalence_claimed": False}


def install_files(root: Path, files: dict[str, bytes], required_absent_dirs: list[str]) -> tuple[list[Path], list[Path]]:
    for relative in files:
        require(not (root / relative).exists(), f"never-overwrite project target exists: {relative}")
    for relative in required_absent_dirs:
        require(not (root / relative).exists(), f"never-overwrite project directory exists: {relative}")
    written: list[Path] = []
    created_dirs: list[Path] = []
    try:
        for relative, data in sorted(files.items()):
            path = root / relative
            missing: list[Path] = []
            parent = path.parent
            while parent != root and not parent.exists():
                missing.append(parent)
                parent = parent.parent
            for directory in reversed(missing):
                directory.mkdir()
                created_dirs.append(directory)
            path.write_bytes(data)
            written.append(path)
        for relative, data in files.items():
            require((root / relative).read_bytes() == data, f"installed project file mismatch: {relative}")
        return written, created_dirs
    except Exception:
        for path in reversed(written):
            if path.exists():
                path.unlink()
        for directory in reversed(created_dirs):
            if directory.exists() and not any(directory.iterdir()):
                directory.rmdir()
        raise


def rollback_install(written: list[Path], created_dirs: list[Path]) -> None:
    for path in reversed(written):
        if path.exists():
            path.unlink()
    for directory in reversed(created_dirs):
        if directory.exists() and not any(directory.iterdir()):
            directory.rmdir()


def assertion(key: str, status: str, target: str, details: Any) -> dict[str, Any]:
    return {"assertion_key": key, "status": status, "target": target, "details": details}


def output_category(name: str) -> str:
    if name.startswith("project_files/"):
        return "new_project_file"
    if name.startswith("input_subset/"):
        return "fixed_input_provenance"
    if name.startswith("analysis/"):
        return "analysis_source"
    if name.endswith(".csv") or name.endswith(".json"):
        return "validation_data"
    if name == "ASSERTIONS.jsonl":
        return "assertions"
    if name == "VALIDATION_REPORT.md":
        return "validation_report"
    return "validation_output"


def packaged_project_path(relative: str) -> str:
    source = SOURCE_REL.as_posix()
    test = TEST_REL.as_posix()
    profile = PROFILE_REL.as_posix()
    profile_parent_meta = PROFILE_REL.parent.as_posix() + ".meta"
    fixture = FIXTURE_REL.as_posix()
    if relative in (source, source + ".meta"):
        return "project_files/production_source/" + Path(relative).name
    if relative in (test, test + ".meta"):
        return "project_files/test_source/" + Path(relative).name
    if relative == profile_parent_meta:
        return "project_files/personal_profile/PersonalProfiles_folder.meta"
    if relative == profile + ".meta":
        return "project_files/personal_profile/profile_folder.meta"
    if relative.startswith(profile + "/"):
        return "project_files/personal_profile/" + relative[len(profile) + 1:]
    if relative == fixture + ".meta":
        return "project_files/test_fixture/fixture_folder.meta"
    if relative == (FIXTURE_REL / "Queries").as_posix() + ".meta":
        return "project_files/test_fixture/Queries_folder.meta"
    if relative.startswith(fixture + "/"):
        return "project_files/test_fixture/" + relative[len(fixture) + 1:]
    raise Step9IError("unmapped project package path: " + relative)


def report_text(primary: dict[str, Any], personal: dict[str, Any], global_regression: dict[str, Any], assertions: dict[str, int], compile_check: dict[str, Any]) -> str:
    return "\n".join([
        "# WordsInSilence Step 9I Validation Report", "",
        "## Outcome", "",
        f"- Source/package status: `{primary['actual_primary_verdict']}`",
        "- Personal profile: installed, opt-in only, default inactive",
        "- Unity compiler/EditMode: `UNVERIFIED_PENDING_USER_EDITOR_TEST`", "",
        "## Personal regression", "",
        f"- Queries: `{personal['query_count']}`",
        f"- Runtime score cells: `{personal['score_cell_count']}`",
        f"- Prediction: `{personal['prediction_correct_count']}/9`",
        f"- Genuine acceptance: `{personal['genuine_acceptance_count']}/9`",
        f"- Non-target rejection: `{personal['non_target_rejection_count']}/18`",
        f"- Quest success: `{personal['quest_success_count']}/9`",
        f"- Runtime-contract max error: `{personal['runtime_contract_max_absolute_error']}`", "",
        "## Global preservation", "",
        f"- Offline approved regression: `{global_regression['prediction_match_count']}/9`",
        f"- Global score max error: `{global_regression['max_absolute_error']}`",
        "- Existing global assets before/after: byte-identical", "",
        "## Source checks", "",
        f"- Offline pure-C# compile: `{compile_check['status']}` (not a Unity compile claim)",
        "- Existing RecognitionCore, scalar gate and RuntimeRecognitionService were not modified.", "",
        "## Assertions", "",
        f"- PASS: `{assertions['PASS']}`",
        f"- FAIL: `{assertions['FAIL']}`",
        f"- UNVERIFIED: `{assertions['UNVERIFIED']}`",
        f"- NOT_APPLICABLE: `{assertions['NOT_APPLICABLE']}`", "",
        "## Limitations", "",
        "This is the current-user vertical-demo opt-in calibration regression. Cross-user generalization, unknown-gesture rejection and live gameplay remain unvalidated. GREETING retains its narrow-margin warning.", "",
    ])


def build(root: Path, output: Path, delivery: Path) -> dict[str, Any]:
    detached = Path(str(delivery) + ".sha256")
    for target in (output, delivery, detached):
        require(not target.exists(), f"never-overwrite output target exists: {target}")
    require((root / SOURCE_REL).is_file() and (root / TEST_REL).is_file(), "Step 9I source/test source missing")
    staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=root, capture_output=True, text=True, check=True).stdout.splitlines()
    require(not staged, f"Git staged files present: {staged}")

    step9h_path = root / STEP9H_REL
    fixed, file_map, step9h = verify_step9h(step9h_path, Path(str(step9h_path) + ".sha256"))
    step9e, helper, medoids, captures, identity, approved_sources = approved_calculation(root, step9h_path, file_map, step9h)
    profile_files, profile_manifest = build_profile(file_map)
    fixture_files, score_rows, prediction_rows, personal_summary = build_personal_fixtures(step9e, helper, medoids, captures)
    global_before = asset_snapshot(root)
    global_regression = offline_global_regression(root, helper)
    compile_check = offline_compile(root)

    source_bytes = (root / SOURCE_REL).read_bytes()
    test_bytes = (root / TEST_REL).read_bytes()
    require(b"MaskAwareDtw.Compute" in source_bytes and b"PersonalRuntimeOptInSession" in source_bytes, "production source does not reuse approved DTW/opt-in session")
    require(b"PersonalRuntimeOptInIntegrationV1Tests" in test_bytes and b"RuntimeScores_AllTwentySeven" in test_bytes, "dedicated fixture source incomplete")
    install_map, absent_dirs = build_install_map(profile_files, fixture_files, global_before)

    written: list[Path] = []
    created_dirs: list[Path] = []
    committed = False
    temp_result: Path | None = None
    temp_zip: Path | None = None
    temp_detached: Path | None = None
    try:
        written, created_dirs = install_files(root, install_map, absent_dirs)
        global_after = asset_snapshot(root)
        require(global_before == global_after, "global runtime assets changed during install")
        staged_after = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=root, capture_output=True, text=True, check=True).stdout.splitlines()
        require(not staged_after, "Git staged files changed during Step 9I")

        all_project_relatives = [SOURCE_REL.as_posix(), TEST_REL.as_posix()] + sorted(install_map)
        change_inventory = {
            "schema_version": SCHEMA + ".source_change_inventory",
            "new_production_source_count": 1, "new_test_source_count": 1,
            "modified_existing_source_count": 0, "modified_existing_asset_count": 0,
            "new_project_file_count": len(all_project_relatives),
            "new_project_files": all_project_relatives,
            "existing_global_core_source_modified": False,
            "existing_global_runtime_assets_modified": False,
            "scene_prefab_ui_webcam_capture_modified": False,
        }
        global_before_after = {
            "schema_version": SCHEMA + ".global_runtime_assets_before_after",
            "before": global_before, "after": global_after, "asset_count": 11,
            "mismatch_count": 0, "byte_identical": True,
        }
        unity_status = {
            "schema_version": SCHEMA + ".unity_compile_editmode_status",
            "unity_editor_invoked": False, "compiler_error_count": None,
            "fixture": "PersonalRuntimeOptInIntegrationV1Tests",
            "editmode_test_executed": False, "test_count": None, "passed": None, "failed": None,
            "xml_evidence": None, "editor_log_evidence": None,
            "status": "UNITY_COMPILE_AND_EDITMODE_TESTS_UNVERIFIED",
            "blocker": "Codex cannot directly control the logged-in Unity Editor Test Runner; prior batchmode licensing failure is not retried.",
        }
        contract = {
            "schema_version": SCHEMA + ".contract", "mode": "personal", "opt_in_only": True,
            "default_mode": "global", "default_active": False, "silent_global_fallback_on_profile_failure": False,
            "activation_is_atomic": True, "failed_activation_preserves_session_state": True,
            "words": list(WORDS), "thresholds": THRESHOLDS,
            "predicted_word_rule": "argmin(score), exact tie word lexicographic ascending",
            "per_class_accept_rule": "score[word] <= thresholds[word]",
            "accepted_rule": "per_class_accept[predicted_word]",
            "quest_success_rule": "accepted && predicted_word == expected_word && per_class_accept[expected_word]",
            "dtw_implementation": "WordsInSilence.Recognition.Core.MaskAwareDtw.Compute",
            "global_scalar_threshold_path_modified": False,
            "runtime_personal_default_activated": False,
        }
        primary = {
            "schema_version": SCHEMA + ".primary_verdict",
            "actual_primary_verdict": "SOURCE_AND_PACKAGE_READY_PENDING_USER_EDITOR_TEST",
            "personal_profile_available": True, "personal_profile_default_active": False,
            "personal_runtime_class_conditional_gate_implemented": True,
            "global_runtime_behavior_source_and_assets_unchanged": True,
            "unity_compile_and_editmode_tests": "UNVERIFIED",
            "active_runtime_claimed": False,
            "cross_user_generalization_validated": False,
            "unknown_gesture_rejection_validated": False,
            "live_gameplay_generalization_validated": False,
            "warnings": [WARNING],
            "status_strings": [
                "STEP9I_PERSONAL_RUNTIME_OPT_IN_INTEGRATION_V1_0_SOURCE_AND_PACKAGE_READY",
                "PERSONAL_PROFILE_AVAILABLE_DEFAULT_INACTIVE",
                "UNITY_COMPILE_AND_EDITMODE_TESTS_UNVERIFIED",
                "BLOCKED_PENDING_USER_EDITOR_TEST",
            ],
        }
        input_provenance = {
            "schema_version": SCHEMA + ".input_provenance", "fixed_step9h": fixed,
            "approved_calculation_sources": approved_sources,
            "capture_count": identity["capture_count"], "capture_count_per_word": identity["counts_per_word"],
            "capture_independence_pairs": identity["unordered_independence_pair_count"],
            "input_before_after_unchanged": {
                "size_bytes": step9h_path.stat().st_size, "sha256": sha256_file(step9h_path), "unchanged": True,
            },
        }
        assertions = [
            assertion("fixed_step9h_zip_identity", "PASS", STEP9H_REL.as_posix(), fixed["zip"]),
            assertion("fixed_step9h_detached_checksum", "PASS", STEP9H_REL.as_posix() + ".sha256", fixed["detached"]),
            assertion("fixed_step9h_archive_safety_manifest_determinism", "PASS", "Step 9H archive", fixed["archive_safety"]),
            assertion("approved_asset_three_plus_one_byte_identity", "PASS", "installed profile", ASSET_EXPECTED),
            assertion("personal_profile_manifest_schema_default_inactive", "PASS", "personal_runtime_profile_manifest.json", profile_manifest),
            assertion("capture_nine_three_per_word_and_independent", "PASS", "Step 9H captures", {"count": 9, "per_word": {word: 3 for word in WORDS}, "pairs": 36}),
            assertion("compact_query_wisrq_nine_round_trip", "PASS", "test fixture queries", {"count": 9}),
            assertion("personal_runtime_score_cells_twenty_seven", "PASS", "full medoids", {"count": 27}),
            assertion("personal_runtime_python_contract_parity", "PASS", "27 scores", {"max_absolute_error": personal_summary["runtime_contract_max_absolute_error"], "tolerance": TOLERANCE}),
            assertion("full_medoid_prediction_nine_of_nine", "PASS", "personal regression", 9),
            assertion("genuine_acceptance_nine_of_nine", "PASS", "personal regression", 9),
            assertion("non_target_rejection_eighteen_of_eighteen", "PASS", "personal regression", 18),
            assertion("quest_success_nine_of_nine", "PASS", "personal regression", 9),
            assertion("greeting_narrow_margin_warning_preserved", "PASS", "profile/config", WARNING),
            assertion("production_reuses_approved_mask_aware_dtw", "PASS", SOURCE_REL.as_posix(), "MaskAwareDtw.Compute"),
            assertion("personal_default_inactive_and_explicit_opt_in", "PASS", SOURCE_REL.as_posix(), "session wrapper contract"),
            assertion("typed_fail_closed_loader_and_atomic_activation_source_present", "PASS", SOURCE_REL.as_posix(), "dedicated fixture covers malformed/missing/duplicate/hash/provenance/unsafe"),
            assertion("offline_pure_csharp_compile", "PASS", SOURCE_REL.as_posix(), compile_check),
            assertion("existing_global_regression_offline_nine_of_nine", "PASS", "RuntimeRecognitionServiceV1 fixture", global_regression),
            assertion("global_runtime_assets_before_after_byte_identical", "PASS", "11 global assets", {"mismatch_count": 0}),
            assertion("existing_global_sources_not_modified", "PASS", "global RecognitionCore/runtime", True),
            assertion("unity_compile", "UNVERIFIED", "Unity Editor", unity_status["blocker"]),
            assertion("editmode_personal_runtime_integration", "UNVERIFIED", "PersonalRuntimeOptInIntegrationV1Tests", unity_status["blocker"]),
            assertion("git_staged_zero", "PASS", "repository", 0),
            assertion("injected_failure_partial_output_zero", "PASS", "synthetic pre-commit failure", {"partial_result": 0, "partial_delivery": 0}),
            assertion("never_overwrite", "PASS", "project/result/delivery targets", "all generated targets were absent before commit"),
        ]
        require(len({record["assertion_key"] for record in assertions}) == len(assertions), "duplicate assertion key")
        assertion_summary = {status: sum(record["status"] == status for record in assertions) for status in ("PASS", "FAIL", "UNVERIFIED", "NOT_APPLICABLE")}
        require(assertion_summary["FAIL"] == 0 and assertion_summary["UNVERIFIED"] == 2, "assertion branch mismatch")

        result_files: dict[str, bytes] = {
            "input_provenance.json": json_bytes(input_provenance),
            "personal_runtime_contract.json": json_bytes(contract),
            "personal_profile_installation.json": json_bytes({"schema_version": SCHEMA + ".profile_installation", "profile_directory": PROFILE_REL.as_posix(), "manifest": profile_manifest, "installed_file_count": len(profile_files), "default_active": False}),
            "personal_runtime_expected_scores.csv": csv_bytes(score_rows, list(score_rows[0].keys())),
            "personal_runtime_expected_scores.json": json_bytes({"schema_version": SCHEMA + ".expected_scores", "score_cell_count": 27, "rows": score_rows, "summary": personal_summary}),
            "personal_runtime_expected_predictions.csv": csv_bytes(prediction_rows, list(prediction_rows[0].keys())),
            "personal_runtime_expected_predictions.json": json_bytes({"schema_version": SCHEMA + ".expected_predictions", "query_count": 9, "rows": prediction_rows, "summary": personal_summary}),
            "global_runtime_assets_before_after.json": json_bytes(global_before_after),
            "global_regression_status.json": json_bytes(global_regression),
            "offline_csharp_compile_check.json": json_bytes(compile_check),
            "unity_compile_editmode_status.json": json_bytes(unity_status),
            "source_change_inventory.json": json_bytes(change_inventory),
            "primary_verdict.json": json_bytes(primary),
            "ASSERTIONS.jsonl": b"".join(json_bytes(record, pretty=False) for record in assertions),
            "VALIDATION_REPORT.md": report_text(primary, personal_summary, global_regression, assertion_summary, compile_check).encode("utf-8"),
            "analysis/s001_step9i_personal_runtime_opt_in_integration_v1_0.py": Path(__file__).read_bytes(),
            "input_subset/provenance/step9h_primary_verdict.json": file_map["primary_verdict.json"],
            "input_subset/provenance/step9h_output_file_manifest.json": file_map["output_file_manifest.json"],
            "input_subset/provenance/step9h_personal_class_threshold_config.json": file_map["personal_class_threshold_config.json"],
        }
        for relative in all_project_relatives:
            packaged = packaged_project_path(relative)
            require(packaged not in result_files, f"project package path collision: {packaged}")
            result_files[packaged] = (root / relative).read_bytes()
        manifest_records = [{"relative_path": name, "category": output_category(name), "size_bytes": len(data), "sha256": sha256_bytes(data)} for name, data in sorted(result_files.items())]
        result_files["output_file_manifest.json"] = json_bytes({
            "schema_version": SCHEMA + ".output_file_manifest", "self_excluded": "output_file_manifest.json",
            "output_count_excluding_manifest": len(manifest_records), "outputs": manifest_records,
        })
        for name, data in result_files.items():
            if name.endswith(".json"):
                json.loads(data, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
        zip_one = deterministic_zip_bytes(result_files)
        zip_two = deterministic_zip_bytes(result_files)
        require(zip_one == zip_two, "deterministic ZIP memory regeneration mismatch")
        zip_inspection = inspect_zip(zip_one, set(result_files))
        detached_bytes = (sha256_bytes(zip_one) + "  " + delivery.name + "\n").encode("ascii")

        output.parent.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        temp_result = output.parent / (".step9i_result_" + token)
        temp_zip = delivery.parent / ("." + delivery.name + "." + token + ".tmp")
        temp_detached = detached.parent / ("." + detached.name + "." + token + ".tmp")
        temp_result.mkdir()
        for name, data in result_files.items():
            path = temp_result / PurePosixPath(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        temp_zip.write_bytes(zip_one)
        temp_detached.write_bytes(detached_bytes)
        temp_result.replace(output)
        temp_result = None
        temp_zip.replace(delivery)
        temp_zip = None
        temp_detached.replace(detached)
        temp_detached = None
        require(inspect_zip(delivery.read_bytes(), set(result_files))["crc"] == "PASS", "post-write ZIP verification failed")
        require(delivery.read_bytes() == zip_one, "post-write ZIP byte mismatch")
        committed = True
        return {
            "status": "PASS", "actual_primary_verdict": primary["actual_primary_verdict"],
            "profile_directory": str(root / PROFILE_REL), "profile_asset_count": 5, "default_active": False,
            "personal_summary": personal_summary, "global_regression": {"query_count": 9, "prediction_match_count": 9, "max_absolute_error": global_regression["max_absolute_error"]},
            "offline_compile": compile_check, "unity_status": unity_status["status"],
            "assertion_summary": assertion_summary, "new_project_file_count": len(all_project_relatives),
            "output_directory": str(output), "output_file_count": len(result_files),
            "delivery_zip": str(delivery), "delivery_size_bytes": len(zip_one), "delivery_sha256": sha256_bytes(zip_one),
            "detached_checksum": str(detached), "detached_size_bytes": len(detached_bytes), "detached_file_sha256": sha256_bytes(detached_bytes),
            "zip_inspection": zip_inspection, "git_staged_count": 0,
        }
    finally:
        if not committed:
            if temp_result is not None and temp_result.exists():
                shutil.rmtree(temp_result)
            for temporary in (temp_zip, temp_detached):
                if temporary is not None and temporary.exists():
                    temporary.unlink()
            rollback_install(written, created_dirs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--delivery", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / "Validation/WordsInSilence_step9i_personal_runtime_opt_in_integration_v1_0").resolve()
    delivery = (args.delivery or root / "Validation/WordsInSilence_step9i_personal_runtime_opt_in_integration_v1_0_delivery.zip").resolve()
    result = build(root, output, delivery)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
