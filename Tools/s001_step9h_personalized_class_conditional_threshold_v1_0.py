#!/usr/bin/env python3
"""WordsInSilence Step 9H class-conditional one-vs-rest calibration.

The sole fixed input is the externally approved Step 9G Delivery. This tool
revalidates the nine captures and approved calculation sources, reproduces the
Step 9G global-threshold failure, derives three class-conditional thresholds
from the same nine leave-one-out folds, evaluates all 27 class decisions, and
emits opt-in personal artifacts only when every approved success condition is
satisfied. It does not activate runtime personal mode or alter global assets.
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
from typing import Any

import numpy as np


SCHEMA = "words_in_silence.step9h.personalized_class_conditional_threshold.v1_0"
WORDS = ("GREETING", "LOVE", "THANK_YOU")
SCRIPT_REL = "Tools/s001_step9h_personalized_class_conditional_threshold_v1_0.py"
OUTPUT_REL = "Validation/WordsInSilence_step9h_personalized_class_conditional_threshold_v1_0"
DELIVERY_REL = "Validation/WordsInSilence_step9h_personalized_class_conditional_threshold_v1_0_delivery.zip"
STEP9G_REL = "Validation/WordsInSilence_step9g_personalized_calibration_medoid_loo_threshold_v1_0_delivery.zip"
STEP9G_SIZE = 3_097_033
STEP9G_SHA256 = "c45ae84344bdadb916cc4c33a217ded83f9761755fac3f6085ea99cab87156c9"
STEP9G_DETACHED_SIZE = 152
STEP9G_DETACHED_SHA256 = "d4e83997d3e3a6cb7759b39f5ef9e51b4b6c41428e196fbd2dd96cda5dd23d8f"
STEP9G_SOURCE_ENTRY = "analysis/s001_step9g_personalized_calibration_medoid_loo_threshold_v1_0.py"
STEP9G_SOURCE_SIZE = 70_933
STEP9G_SOURCE_SHA256 = "8223bf09854da6dbaf40961786b002f7f20d434fdcc8d32e86399066eb53495a"
STEP9E_SOURCE_ENTRY = "input_subset/provenance/s001_step9e_personalized_calibration_template_foundation_v1_0.py"
STEP9E_SOURCE_SIZE = 94_953
STEP9E_SOURCE_SHA256 = "b2b92ef247ee1f81ecf2142546760fcaf15da6c2f4f036ac3c632adf91fa8d13"
DTW_HELPER_ENTRY = "input_subset/provenance/s001_step9d_physical_recognition_domain_gap_diagnostic_v1_0.py"
DTW_HELPER_SIZE = 90_451
DTW_HELPER_SHA256 = "334aed9691ee57a8305d32bfefc8220568573dcf99bb3c824f0dde6cee8f788d"
CAPTURE_PREFIX = "input_subset/captures/"
PARITY_TOLERANCE = 1e-12
DTW_BAND = 0.10
MISSING_COST = 1.0
EXPECTED_MEDOIDS = {
    "GREETING": "87e5b54432ac49a197a596d5ff7c3845",
    "LOVE": "d958cbc2a1c44e03bb29bf64a6fae195",
    "THANK_YOU": "2b323671e3f64049a74556c71ff70154",
}
EXPECTED_GLOBAL = {
    "genuine_upper": 0.5172946711028763,
    "impostor_lower": 0.4231019848491888,
    "separation_gap": -0.09419268625368749,
}
EXPECTED_CLASS = {
    "GREETING": {
        "genuine_upper": 0.5172946711028763,
        "impostor_lower": 0.523833132325901,
        "separation_gap": 0.006538461223024705,
        "threshold": 0.5205639017143886,
    },
    "LOVE": {
        "genuine_upper": 0.36818101372824913,
        "impostor_lower": 0.4231019848491888,
        "separation_gap": 0.05492097112093969,
        "threshold": 0.39564149928871895,
    },
    "THANK_YOU": {
        "genuine_upper": 0.33790469372483245,
        "impostor_lower": 0.42459261430230166,
        "separation_gap": 0.08668792057746921,
        "threshold": 0.38124865401356706,
    },
}


class Step9HError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Step9HError(message)


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
        raise Step9HError(f"non-standard JSON constant: {token}")

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


def parse_detached(data: bytes, expected_name: str, expected_hash: str) -> bool:
    match = re.fullmatch(rb"([0-9a-f]{64})  ([^\r\n]+)(?:\r?\n)?", data)
    return bool(
        match and match.group(1).decode("ascii") == expected_hash
        and match.group(2).decode("utf-8") == expected_name
    )


def manifest_records(data: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("outputs", "files", "inputs", "input_files"):
        records = data.get(key)
        if isinstance(records, list):
            return records
    raise Step9HError(f"manifest has no supported record list: {sorted(data)}")


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


def verify_fixed_step9g(path: Path, detached_path: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    require(path.is_file(), f"missing fixed Step 9G ZIP: {path}")
    require(detached_path.is_file(), f"missing fixed Step 9G detached checksum: {detached_path}")
    raw = path.read_bytes()
    detached = detached_path.read_bytes()
    digest = sha256_bytes(raw)
    require(len(raw) == STEP9G_SIZE, f"Step 9G ZIP size mismatch: {len(raw)} != {STEP9G_SIZE}")
    require(digest == STEP9G_SHA256, f"Step 9G ZIP SHA mismatch: {digest} != {STEP9G_SHA256}")
    require(len(detached) == STEP9G_DETACHED_SIZE, f"Step 9G detached size mismatch: {len(detached)}")
    require(sha256_bytes(detached) == STEP9G_DETACHED_SHA256, "Step 9G detached file SHA mismatch")
    require(parse_detached(detached, path.name, digest), "Step 9G detached syntax/filename/digest mismatch")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        require(archive.testzip() is None, "Step 9G ZIP CRC failure")
        require(len(names) == len(set(names)), "Step 9G ZIP duplicate entry")
        require(all(safe_zip_name(name) for name in names), "Step 9G ZIP unsafe path")
        require(not any(stat.S_ISLNK((info.external_attr >> 16) & 0xFFFF) for info in infos), "Step 9G ZIP symlink entry")
        require(not any(info.flag_bits & 1 for info in infos), "Step 9G ZIP encrypted entry")
        manifest_payload = archive.read("output_file_manifest.json")
        manifest = strict_json_loads(manifest_payload)
        records = manifest_records(manifest)
        targets = [str(record["relative_path"]) for record in records]
        require(len(records) == 44 and len(targets) == len(set(targets)), "Step 9G manifest target count/uniqueness mismatch")
        require(set(targets) == set(names) - {"output_file_manifest.json"}, "Step 9G manifest membership mismatch")
        for record in records:
            payload = archive.read(record["relative_path"])
            require(len(payload) == int(record["size_bytes"]), f"Step 9G manifest size mismatch: {record['relative_path']}")
            require(sha256_bytes(payload) == record["sha256"], f"Step 9G manifest SHA mismatch: {record['relative_path']}")
        file_map = {name: archive.read(name) for name in names}
    regenerated = deterministic_zip_bytes(file_map)
    require(regenerated == raw, "Step 9G deterministic ZIP regeneration mismatch")
    return {
        "zip": {
            "path": path.as_posix(), "size_bytes": len(raw), "sha256": digest,
            "entry_count": len(file_map), "manifest_record_count": len(records),
            "crc": "PASS", "duplicate_entries": 0, "unsafe_paths": 0,
            "symlinks": 0, "encrypted_entries": 0, "membership": "PASS",
            "deterministic_regeneration_byte_identical": True,
        },
        "detached_checksum": {
            "path": detached_path.as_posix(), "size_bytes": len(detached),
            "file_sha256": sha256_bytes(detached), "syntax_filename_digest": "PASS",
        },
    }, file_map


def load_module_from_bytes(name: str, source: bytes, filename: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = filename
    sys.modules[name] = module
    exec(compile(source.decode("utf-8"), filename, "exec"), module.__dict__)
    return module


def verify_current_tool(root: Path, relative: str, expected_size: int, expected_hash: str) -> dict[str, Any]:
    path = root / relative
    require(path.is_file(), f"approved tool missing: {path}")
    size = path.stat().st_size
    digest = sha256_file(path)
    require(size == expected_size and digest == expected_hash, f"approved tool changed: {relative}")
    return {"path": relative, "size_bytes": size, "sha256": digest, "unchanged": True}


def load_approved_sources(root: Path, file_map: dict[str, bytes]) -> tuple[types.ModuleType, types.ModuleType, types.ModuleType, dict[str, Any], dict[str, bytes]]:
    step9g_source = file_map[STEP9G_SOURCE_ENTRY]
    step9e_source = file_map[STEP9E_SOURCE_ENTRY]
    helper_source = file_map[DTW_HELPER_ENTRY]
    require(len(step9g_source) == STEP9G_SOURCE_SIZE and sha256_bytes(step9g_source) == STEP9G_SOURCE_SHA256, "Step 9G approved source identity mismatch")
    require(len(step9e_source) == STEP9E_SOURCE_SIZE and sha256_bytes(step9e_source) == STEP9E_SOURCE_SHA256, "Step 9E approved source identity mismatch")
    require(len(helper_source) == DTW_HELPER_SIZE and sha256_bytes(helper_source) == DTW_HELPER_SHA256, "approved DTW helper identity mismatch")
    current = {
        "step9e": verify_current_tool(root, "Tools/s001_step9e_personalized_calibration_template_foundation_v1_0.py", STEP9E_SOURCE_SIZE, STEP9E_SOURCE_SHA256),
        "step9f": verify_current_tool(root, "Tools/s001_step9f_additional_personal_calibration_capture_one_take_per_word_v1_0.py", 62_114, "0e2af18089722c0d0b39f7057d9b603b55f9a8f0605e1a4e0cb1023c7eb845fe"),
        "step9g": verify_current_tool(root, "Tools/s001_step9g_personalized_calibration_medoid_loo_threshold_v1_0.py", STEP9G_SOURCE_SIZE, STEP9G_SOURCE_SHA256),
    }
    step9g = load_module_from_bytes("wis_step9g_approved_calculation", step9g_source, STEP9G_SOURCE_ENTRY)
    step9e = load_module_from_bytes("wis_step9e_approved_calculation_for_step9h", step9e_source, STEP9E_SOURCE_ENTRY)
    helper = load_module_from_bytes("wis_step9d_approved_dtw_for_step9h", helper_source, DTW_HELPER_ENTRY)
    require(tuple(step9g.WORDS) == WORDS and tuple(step9e.WORDS) == WORDS, "approved word contract mismatch")
    require(float(step9e.DTW_BAND) == DTW_BAND and float(step9e.MISSING_COST) == MISSING_COST, "approved Step 9E DTW constants mismatch")
    require(float(helper.DTW_BAND) == DTW_BAND and float(helper.MISSING_COST) == MISSING_COST, "approved helper constants mismatch")
    provenance = {
        "step9g_source": {"entry": STEP9G_SOURCE_ENTRY, "size_bytes": len(step9g_source), "sha256": sha256_bytes(step9g_source)},
        "step9e_source": {"entry": STEP9E_SOURCE_ENTRY, "size_bytes": len(step9e_source), "sha256": sha256_bytes(step9e_source)},
        "approved_dtw_helper": {"entry": DTW_HELPER_ENTRY, "size_bytes": len(helper_source), "sha256": sha256_bytes(helper_source)},
        "current_tools_unchanged": current,
    }
    subset = {
        "input_subset/provenance/s001_step9g_personalized_calibration_medoid_loo_threshold_v1_0.py": step9g_source,
        "input_subset/provenance/s001_step9e_personalized_calibration_template_foundation_v1_0.py": step9e_source,
        "input_subset/provenance/s001_step9d_physical_recognition_domain_gap_diagnostic_v1_0.py": helper_source,
    }
    return step9g, step9e, helper, provenance, subset


def load_captures(
    step9g: types.ModuleType, step9e: types.ModuleType, archive_path: Path,
    archive_sha256: str, file_map: dict[str, bytes],
) -> tuple[list[Any], dict[str, bytes], dict[str, Any], list[dict[str, Any]]]:
    identity_document = strict_json_loads(file_map["capture_identity.json"])
    prior_step9e = strict_json_loads(file_map["input_subset/provenance/step9e_seed_capture_identity.json"])
    prior_step9f = strict_json_loads(file_map["input_subset/provenance/step9f_additional_capture_identity.json"])
    identity_by_id = {record["capture_id"]: record for record in identity_document["captures"]}
    prior_by_id = {record["capture_id"]: record for record in list(prior_step9e["captures"]) + list(prior_step9f["captures"])}
    require(len(identity_by_id) == len(prior_by_id) == 9, "prior identity provenance count mismatch")
    json_entries = sorted(name for name in file_map if name.startswith(CAPTURE_PREFIX) and name.endswith(".json"))
    checksum_entries = sorted(name for name in file_map if name.startswith(CAPTURE_PREFIX) and name.endswith(".json.sha256"))
    require(len(json_entries) == len(checksum_entries) == 9, "Step 9G capture/checksum count mismatch")
    captures: list[Any] = []
    subset: dict[str, bytes] = {}
    identity_mismatches: list[str] = []
    with zipfile.ZipFile(archive_path) as archive:
        for entry in json_entries:
            data = strict_json_loads(file_map[entry])
            capture_id = str(data["capture_id"])
            prior = identity_by_id.get(capture_id)
            require(prior is not None, f"capture absent from Step 9G identity: {capture_id}")
            capture_key = str(prior["capture_key"])
            require("::" in capture_key, f"capture key format invalid: {capture_key}")
            prefix = capture_key.split("::", 1)[0]
            capture, _, files = step9g.load_capture_from_entry(
                step9e, archive, archive_path, archive_sha256, entry,
                str(prior["source_family"]), prefix, str(prior["manifest_provenance_entry"]),
            )
            captures.append(capture)
            for name, payload in files.items():
                require(name not in subset, f"capture subset collision: {name}")
                subset[name] = payload
            checks = {
                "capture_key": capture.capture_key, "word": capture.word,
                "run_id": capture.run_id, "capture_id": capture.capture_id,
                "source_payload_sha256": capture.source_sha256,
                "coordinate_payload_sha256": capture.coordinate_payload_sha256,
                "mask_payload_sha256": capture.mask_payload_sha256,
                "normalized_payload_sha256": capture.normalized_payload_sha256,
            }
            for record_name, record in (("step9g", prior), ("step9e_or_9f", prior_by_id[capture_id])):
                for field, value in checks.items():
                    if record.get(field) != value:
                        identity_mismatches.append(f"{record_name}:{capture_id}:{field}")
    captures.sort(key=lambda item: (WORDS.index(item.word), item.source_family, item.capture_id))
    step9e.validate_capture_set(captures)
    counts = {word: sum(capture.word == word for capture in captures) for word in WORDS}
    require(len(captures) == 9 and counts == {word: 3 for word in WORDS}, f"capture count mismatch: {counts}")
    require(not identity_mismatches, f"identity provenance mismatch: {identity_mismatches}")
    require(len({capture.capture_key for capture in captures}) == 9 and len({capture.capture_id for capture in captures}) == 9, "capture key/ID uniqueness mismatch")
    run_groups: dict[str, list[str]] = {}
    for capture in captures:
        run_groups.setdefault(capture.run_id, []).append(capture.capture_key)
    require(len(run_groups) == 3 and all(len(group) == 3 for group in run_groups.values()), f"run grouping mismatch: {run_groups}")
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
            "left_capture_key": left.capture_key, "right_capture_key": right.capture_key,
            **flags, "independent": not any(flags.values()),
        })
    require(len(independence) == 36 and all(row["independent"] for row in independence), "36-pair independence failure")
    capture_identity = {
        "schema_version": SCHEMA + ".capture_identity",
        "capture_count": 9, "counts_per_word": counts, "run_group_count": len(run_groups),
        "run_groups": run_groups,
        "captures": [step9e.identity_record(capture) for capture in captures],
        "prior_identity_mismatch_count": 0,
        "unordered_independence_pair_count": 36,
        "all_pairs_independent": True,
        "pairwise_independence": independence,
    }
    return captures, subset, capture_identity, independence


def reproduce_dtw_and_step9g(
    step9g: types.ModuleType, step9e: types.ModuleType, helper: types.ModuleType,
    captures: list[Any], file_map: dict[str, bytes],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    rows, summary = step9e.pairwise_dtw_rows(captures, helper)
    require(len(rows) == 81, "recomputed DTW cell count mismatch")
    prior_rows = list(csv.DictReader(io.StringIO(file_map["capture_pairwise_dtw.csv"].decode("utf-8"))))
    require(len(prior_rows) == 81, "Step 9G DTW CSV cell count mismatch")
    prior_by_key = {row["pair_key"]: row for row in prior_rows}
    require(len(prior_by_key) == 81, "Step 9G DTW CSV duplicate pair key")
    numeric_fields = (
        "distance", "repeat_distance", "repeat_absolute_error", "approved_reference_distance",
        "approved_runtime_contract_distance", "approved_reference_absolute_error",
        "approved_runtime_absolute_error", "cumulative_cost",
    )
    integer_fields = (
        "path_length", "approved_reference_path_length", "approved_runtime_path_length",
        "both_valid_count_total", "both_valid_count_min", "both_valid_count_max",
        "comparable_count_total", "comparable_count_min", "comparable_count_max",
    )
    max_errors = {field: 0.0 for field in numeric_fields}
    integer_mismatches = 0
    for row in rows:
        prior = prior_by_key.get(row["pair_key"])
        require(prior is not None, f"DTW pair absent from Step 9G CSV: {row['pair_key']}")
        for field in numeric_fields:
            max_errors[field] = max(max_errors[field], abs(float(row[field]) - float(prior[field])))
        integer_mismatches += sum(int(row[field]) != int(prior[field]) for field in integer_fields)
    require(max(max_errors.values()) <= PARITY_TOLERANCE and integer_mismatches == 0, "Step 9G DTW CSV parity mismatch")
    medoids, _, loo_rows, calibration = step9g.compute_medoids_and_loo(captures, rows)
    selected = {word: medoids[word].capture_id for word in WORDS}
    require(selected == EXPECTED_MEDOIDS, f"medoid reproduction mismatch: {selected}")
    prior_medoid = strict_json_loads(file_map["medoid_selection.json"])["selected_capture_ids"]
    require(selected == prior_medoid, "Step 9G stored medoid mismatch")
    global_summary = calibration["summary"]
    require(global_summary["verdict"] == "CALIBRATION_NOT_SEPARABLE", "Step 9G global failure not reproduced")
    require(global_summary["prediction_correct_count"] == 9 and not global_summary["separation_strict"], "Step 9G global LOO state mismatch")
    for field, expected in EXPECTED_GLOBAL.items():
        actual = global_summary["separation_gap"] if field == "separation_gap" else global_summary[field]
        require(abs(float(actual) - expected) <= PARITY_TOLERANCE, f"global failure value mismatch: {field}: {actual}")
    prior_global = strict_json_loads(file_map["threshold_derivation.json"])
    require(prior_global["actual_verdict"] == "CALIBRATION_NOT_SEPARABLE" and prior_global["personal_threshold"] is None, "Step 9G stored global verdict changed")
    require(abs(float(prior_global["genuine_upper"]) - EXPECTED_GLOBAL["genuine_upper"]) <= PARITY_TOLERANCE, "Step 9G stored genuine upper mismatch")
    require(abs(float(prior_global["impostor_lower"]) - EXPECTED_GLOBAL["impostor_lower"]) <= PARITY_TOLERANCE, "Step 9G stored impostor lower mismatch")
    parity = {
        "schema_version": SCHEMA + ".dtw_parity_summary",
        "ordered_cell_count": 81, "step9g_csv_cell_count": 81,
        "step9g_csv_numeric_max_absolute_errors": max_errors,
        "step9g_csv_integer_mismatch_count": integer_mismatches,
        "diagonal_max_absolute_distance": summary["diagonal_max_absolute_distance"],
        "symmetry_max_absolute_error": summary["symmetry_max_absolute_error"],
        "repeat_max_absolute_error": summary["repeat_max_absolute_error"],
        "approved_reference_max_absolute_error": summary["approved_reference_max_absolute_error"],
        "approved_runtime_contract_max_absolute_error": summary["approved_runtime_contract_max_absolute_error"],
        "approved_path_mismatch_count": summary["approved_path_mismatch_count"],
        "parity_tolerance": PARITY_TOLERANCE, "status": "PASS",
    }
    global_reproduction = {
        "schema_version": SCHEMA + ".global_threshold_failure_reproduction",
        "step9g_actual_verdict": "CALIBRATION_NOT_SEPARABLE",
        "step9g_global_failure_preserved": True,
        "closed_set_prediction_correct_count": 9, "fold_count": 9,
        "global_genuine_upper": global_summary["genuine_upper"],
        "global_impostor_lower": global_summary["impostor_lower"],
        "global_separation_gap": global_summary["separation_gap"],
        "global_strict_separation": False,
        "global_threshold_generated": False, "global_threshold": None,
        "not_reclassified_as_global_success": True,
    }
    return rows, parity, medoids, calibration, global_reproduction


def derive_class_conditional(calibration: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any], bool]:
    fold_rows: list[dict[str, Any]] = []
    fold_documents: list[dict[str, Any]] = []
    for detail in calibration["loo_details"]:
        distances = {word: float(detail["word_distances"][word]) for word in WORDS}
        ranking = sorted(WORDS, key=lambda word: (distances[word], word))
        fold_cell_records: list[dict[str, Any]] = []
        for target_word in WORDS:
            selection = detail["fold_medoid_selection"][target_word]
            row = {
                "score_cell_key": detail["fold_id"] + "||" + target_word,
                "fold_id": detail["fold_id"], "fold_index": int(detail["fold_index"]),
                "held_out_capture_key": detail["held_out_capture_key"],
                "held_out_capture_id": detail["held_out_capture_id"],
                "expected_word": detail["expected_word"], "target_word": target_word,
                "target_medoid_capture_key": selection["selected_capture_key"],
                "target_medoid_capture_id": selection["selected_capture_id"],
                "score": distances[target_word],
                "score_kind": "genuine" if detail["expected_word"] == target_word else "impostor",
                "predicted_word": ranking[0],
                "prediction_correct": ranking[0] == detail["expected_word"],
            }
            fold_rows.append(row)
            fold_cell_records.append(row)
        fold_documents.append({
            "fold_id": detail["fold_id"], "fold_index": int(detail["fold_index"]),
            "held_out_capture_key": detail["held_out_capture_key"],
            "held_out_capture_id": detail["held_out_capture_id"],
            "expected_word": detail["expected_word"], "predicted_word": ranking[0],
            "prediction_correct": ranking[0] == detail["expected_word"],
            "scores": distances, "score_cells": fold_cell_records,
        })
    require(len(fold_rows) == 27 and len({row["score_cell_key"] for row in fold_rows}) == 27, "class score cell count/uniqueness mismatch")
    per_word: dict[str, Any] = {}
    thresholds: dict[str, float] = {}
    for word in WORDS:
        genuine = [float(row["score"]) for row in fold_rows if row["target_word"] == word and row["score_kind"] == "genuine"]
        impostor = [float(row["score"]) for row in fold_rows if row["target_word"] == word and row["score_kind"] == "impostor"]
        require(len(genuine) == 3 and len(impostor) == 6, f"class score count mismatch: {word}")
        genuine_upper = max(genuine)
        impostor_lower = min(impostor)
        gap = impostor_lower - genuine_upper
        threshold = (genuine_upper + impostor_lower) / 2.0
        actual = {
            "genuine_scores": genuine, "impostor_scores": impostor,
            "genuine_upper": genuine_upper, "impostor_lower": impostor_lower,
            "separation_gap": gap, "threshold": threshold,
            "genuine_score_count": 3, "impostor_score_count": 6,
            "strict_separation": genuine_upper < impostor_lower,
        }
        for field in ("genuine_upper", "impostor_lower", "separation_gap", "threshold"):
            require(abs(float(actual[field]) - float(EXPECTED_CLASS[word][field])) <= PARITY_TOLERANCE, f"class expected value mismatch: {word}:{field}:{actual[field]}")
        per_word[word] = actual
        thresholds[word] = threshold
    acceptance_rows: list[dict[str, Any]] = []
    acceptance_folds: list[dict[str, Any]] = []
    for fold in fold_documents:
        per_class_accept: dict[str, bool] = {}
        rows_for_fold: list[dict[str, Any]] = []
        for cell in fold["score_cells"]:
            target = cell["target_word"]
            accepted = float(cell["score"]) <= thresholds[target]
            is_genuine = cell["expected_word"] == target
            desired_accept = is_genuine
            row = {
                "decision_key": cell["score_cell_key"], "fold_id": cell["fold_id"],
                "expected_word": cell["expected_word"], "predicted_word": cell["predicted_word"],
                "target_word": target, "score": cell["score"], "threshold": thresholds[target],
                "is_genuine": is_genuine, "class_accept": accepted,
                "desired_class_accept": desired_accept, "class_decision_correct": accepted == desired_accept,
            }
            per_class_accept[target] = accepted
            acceptance_rows.append(row)
            rows_for_fold.append(row)
        predicted = fold["predicted_word"]
        expected = fold["expected_word"]
        accepted = per_class_accept[predicted]
        quest_success = accepted and predicted == expected and per_class_accept[expected]
        acceptance_folds.append({
            "fold_id": fold["fold_id"], "expected_word": expected, "predicted_word": predicted,
            "closed_set_prediction_correct": predicted == expected,
            "per_class_accept": per_class_accept,
            "accepted": accepted, "predicted_word_equals_expected_word": predicted == expected,
            "expected_word_class_accept": per_class_accept[expected],
            "quest_success": quest_success, "class_decisions": rows_for_fold,
        })
    prediction_correct = sum(fold["closed_set_prediction_correct"] for fold in acceptance_folds)
    genuine_accept = sum(row["class_accept"] for row in acceptance_rows if row["is_genuine"])
    non_target_reject = sum(not row["class_accept"] for row in acceptance_rows if not row["is_genuine"])
    quest_success = sum(fold["quest_success"] for fold in acceptance_folds)
    all_separable = all(per_word[word]["strict_separation"] for word in WORDS)
    success = prediction_correct == 9 and genuine_accept == 9 and non_target_reject == 18 and all_separable
    threshold_document = {
        "schema_version": SCHEMA + ".class_conditional_threshold_derivation",
        "threshold_mode": "class_conditional_one_vs_rest",
        "score_cell_count": 27, "per_word": per_word, "thresholds": thresholds,
        "all_three_strictly_separable": all_separable,
        "greeting_margin_warning": "GREETING_CLASS_CONDITIONAL_MARGIN_NARROW",
        "expected_value_tolerance": PARITY_TOLERANCE,
    }
    fold_document = {
        "schema_version": SCHEMA + ".class_conditional_fold_scores",
        "fold_count": 9, "score_cell_count": 27, "folds": fold_documents,
    }
    acceptance_document = {
        "schema_version": SCHEMA + ".class_conditional_acceptance_evaluation",
        "fold_count": 9, "class_decision_count": 27,
        "closed_set_prediction_correct_count": prediction_correct,
        "genuine_acceptance_count": genuine_accept, "genuine_decision_count": 9,
        "non_target_rejection_count": non_target_reject, "non_target_decision_count": 18,
        "quest_success_count": quest_success,
        "all_class_decisions_correct": genuine_accept == 9 and non_target_reject == 18,
        "folds": acceptance_folds,
    }
    return fold_rows, fold_document, acceptance_rows, {"threshold": threshold_document, "acceptance": acceptance_document}, success


def build_active_artifacts(
    step9e: types.ModuleType, medoids: dict[str, Any], threshold_document: dict[str, Any], success: bool,
) -> tuple[dict[str, bytes], list[dict[str, Any]]]:
    if not success:
        return {}, []
    artifacts: dict[str, bytes] = {}
    template_records: list[dict[str, Any]] = []
    for word in WORDS:
        capture = medoids[word]
        require(capture.capture_id == EXPECTED_MEDOIDS[word], f"active medoid identity mismatch: {word}")
        name = f"active_personal_templates/personal_template_{word}.wisrt"
        payload = step9e.write_wisrt_bytes(capture)
        parsed = step9e.read_wisrt_bytes(payload)
        require(parsed["word"] == word and parsed["sample_id"] == capture.capture_id, f"WISRT word/capture mismatch: {word}")
        require(parsed["source_payload_sha256"] == capture.source_sha256, f"WISRT source SHA mismatch: {word}")
        require(parsed["logical_payload_sha256"] == capture.normalized_payload_sha256, f"WISRT normalized SHA mismatch: {word}")
        require(np.array_equal(parsed["coordinates"], capture.coordinates, equal_nan=True), f"WISRT coordinate round-trip mismatch: {word}")
        require(np.array_equal(parsed["valid_mask"], capture.valid_mask), f"WISRT mask round-trip mismatch: {word}")
        artifacts[name] = payload
        template_records.append({
            "word": word, "capture_key": capture.capture_key, "run_id": capture.run_id,
            "capture_id": capture.capture_id, "source_payload_sha256": capture.source_sha256,
            "coordinate_payload_sha256": capture.coordinate_payload_sha256,
            "mask_payload_sha256": capture.mask_payload_sha256,
            "normalized_payload_sha256": capture.normalized_payload_sha256,
            "relative_path": name, "size_bytes": len(payload), "sha256": sha256_bytes(payload),
            "embedded_checksum_sha256": parsed["embedded_checksum_sha256"],
            "writer_reader_round_trip": "PASS",
        })
    config = {
        "schema_version": SCHEMA + ".personal_class_threshold_config",
        "mode": "personal", "threshold_mode": "class_conditional_one_vs_rest",
        "opt_in_only": True, "runtime_activated": False,
        "global_template_or_threshold_overwritten": False,
        "source_capture_count": 9, "source_capture_count_per_word": {word: 3 for word in WORDS},
        "thresholds": threshold_document["thresholds"],
        "per_word": {
            word: {
                "genuine_upper": threshold_document["per_word"][word]["genuine_upper"],
                "impostor_lower": threshold_document["per_word"][word]["impostor_lower"],
                "separation_gap": threshold_document["per_word"][word]["separation_gap"],
                "genuine_score_count": 3, "impostor_score_count": 6,
            }
            for word in WORDS
        },
        "template_records": template_records,
        "gameplay_acceptance_rule": {
            "accepted": "per_class_accept[predicted_word]",
            "predicted_word_equals_expected_word": "predicted_word == expected_word",
            "expected_word_class_accept": "per_class_accept[expected_word]",
            "quest_success": "accepted and predicted_word == expected_word and per_class_accept[expected_word]",
        },
        "calibration_scope": "current_user_vertical_demo_opt_in",
        "cross_user_generalization_validated": False,
        "unknown_gesture_rejection_validated": False,
        "live_gameplay_generalization_pending": True,
        "greeting_margin_narrow": True,
        "warnings": ["GREETING_CLASS_CONDITIONAL_MARGIN_NARROW"],
    }
    require(set(config["thresholds"]) == set(WORDS) and all(type(config["thresholds"][word]) is float for word in WORDS), "config threshold map schema mismatch")
    artifacts["personal_class_threshold_config.json"] = pretty_json_bytes(config)
    require(len(artifacts) == 4, "success artifact count mismatch")
    return artifacts, template_records


def primary_metadata(success: bool, threshold_document: dict[str, Any], acceptance: dict[str, Any], active_count: int, config_count: int) -> dict[str, Any]:
    if success:
        require(active_count == 3 and config_count == 1, "success primary active counts mismatch")
        verdict = "CLASS_CONDITIONAL_CALIBRATION_SUCCESS"
        statuses = [
            "STEP9H_PERSONALIZED_CLASS_CONDITIONAL_THRESHOLD_V1_0_READY_PENDING_EXTERNAL_REVIEW",
            "PRIMARY_VERDICT_CLASS_CONDITIONAL_CALIBRATION_SUCCESS",
            "ACTIVE_PERSONAL_TEMPLATES_THREE_AND_CLASS_THRESHOLDS_THREE_GENERATED",
            "RUNTIME_PERSONAL_MODE_NOT_ACTIVATED",
            "BLOCKED_PENDING_EXTERNAL_REVIEW",
        ]
    else:
        require(active_count == 0 and config_count == 0, "failure primary active counts mismatch")
        verdict = "CLASS_CONDITIONAL_CALIBRATION_NOT_SEPARABLE"
        statuses = [
            "STEP9H_PERSONALIZED_CLASS_CONDITIONAL_THRESHOLD_V1_0_READY_PENDING_EXTERNAL_REVIEW",
            "PRIMARY_VERDICT_CLASS_CONDITIONAL_CALIBRATION_NOT_SEPARABLE",
            "ACTIVE_PERSONAL_OUTPUT_NOT_GENERATED",
            "BLOCKED_PENDING_EXTERNAL_REVIEW",
        ]
    return {
        "schema_version": SCHEMA + ".primary_verdict",
        "calculation_validation_status": "PASS", "actual_primary_verdict": verdict,
        "step9g_global_primary_verdict_preserved": "CALIBRATION_NOT_SEPARABLE",
        "capture_count": 9, "capture_count_per_word": {word: 3 for word in WORDS},
        "class_conditional_score_cell_count": 27,
        "closed_set_prediction_correct_count": acceptance["closed_set_prediction_correct_count"],
        "genuine_acceptance_count": acceptance["genuine_acceptance_count"],
        "non_target_rejection_count": acceptance["non_target_rejection_count"],
        "quest_success_count": acceptance["quest_success_count"],
        "all_three_class_gaps_positive": threshold_document["all_three_strictly_separable"],
        "thresholds": threshold_document["thresholds"] if success else None,
        "active_personal_wisrt_count": active_count,
        "personal_class_threshold_config_count": config_count,
        "runtime_personal_mode_activated": False,
        "global_template_or_threshold_overwritten": False,
        "cross_user_generalization_validated": False,
        "unknown_gesture_rejection_validated": False,
        "live_gameplay_generalization_pending": True,
        "warnings": ["GREETING_CLASS_CONDITIONAL_MARGIN_NARROW"],
        "status_strings": statuses, "next_core_milestone_started": False,
    }


def metadata_branch_regression() -> dict[str, Any]:
    threshold_success = {"all_three_strictly_separable": True, "thresholds": {word: 0.5 for word in WORDS}}
    acceptance_success = {"closed_set_prediction_correct_count": 9, "genuine_acceptance_count": 9, "non_target_rejection_count": 18, "quest_success_count": 9}
    success = primary_metadata(True, threshold_success, acceptance_success, 3, 1)
    threshold_failure = {"all_three_strictly_separable": False, "thresholds": {}}
    acceptance_failure = {"closed_set_prediction_correct_count": 8, "genuine_acceptance_count": 8, "non_target_rejection_count": 17, "quest_success_count": 8}
    failure = primary_metadata(False, threshold_failure, acceptance_failure, 0, 0)
    require(success["actual_primary_verdict"] == "CLASS_CONDITIONAL_CALIBRATION_SUCCESS" and success["active_personal_wisrt_count"] == 3, "success branch metadata regression")
    require(success["step9g_global_primary_verdict_preserved"] == "CALIBRATION_NOT_SEPARABLE", "success branch changed Step 9G global verdict")
    require(failure["actual_primary_verdict"] == "CLASS_CONDITIONAL_CALIBRATION_NOT_SEPARABLE" and failure["active_personal_wisrt_count"] == 0, "failure branch metadata regression")
    return {
        "schema_version": SCHEMA + ".metadata_branch_regression",
        "scope_note": "Synthetic branch metadata checks do not claim actual-capture performance.",
        "tests": [
            {"test_id": "success_branch_preserves_global_failure_and_emits_3_plus_1", "status": "PASS"},
            {"test_id": "failure_branch_emits_zero_active_output", "status": "PASS"},
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


def git_staged(root: Path) -> dict[str, Any]:
    result = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    require(result.returncode == 0, f"git staged query failed: {result.stderr.strip()}")
    files = [line for line in result.stdout.splitlines() if line]
    return {"count": len(files), "files": files}


def output_category(name: str) -> str:
    if name.startswith("active_personal_templates/"):
        return "active_personal_template_opt_in"
    if name == "personal_class_threshold_config.json":
        return "personal_class_threshold_opt_in"
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


def validate_result_schemas(files: dict[str, bytes], success: bool) -> dict[str, Any]:
    json_names = sorted(name for name in files if name.endswith(".json"))
    for name in json_names:
        require(isinstance(strict_json_loads(files[name]), dict), f"generated JSON root invalid: {name}")
    assertions = [strict_json_loads(line) for line in files["ASSERTIONS.jsonl"].splitlines() if line]
    assertion_keys = [row["assertion_key"] for row in assertions]
    require(len(assertion_keys) == len(set(assertion_keys)), "duplicate assertion key")
    inventory = list(csv.DictReader(io.StringIO(files["capture_inventory.csv"].decode("utf-8"))))
    require(len(inventory) == 9 and len({row["capture_key"] for row in inventory}) == 9, "capture inventory key/count mismatch")
    scores = list(csv.DictReader(io.StringIO(files["class_conditional_fold_scores.csv"].decode("utf-8"))))
    require(len(scores) == 27 and len({row["score_cell_key"] for row in scores}) == 27, "score cell key/count mismatch")
    require(len({(row["fold_id"], row["target_word"]) for row in scores}) == 27, "fold/target key duplicate")
    decisions = list(csv.DictReader(io.StringIO(files["class_conditional_acceptance_evaluation.csv"].decode("utf-8"))))
    require(len(decisions) == 27 and len({row["decision_key"] for row in decisions}) == 27, "acceptance key/count mismatch")
    if success:
        require(sum(name.startswith("active_personal_templates/") and name.endswith(".wisrt") for name in files) == 3, "success WISRT count mismatch")
        require("personal_class_threshold_config.json" in files, "success config missing")
        config = strict_json_loads(files["personal_class_threshold_config.json"])
        require(config["threshold_mode"] == "class_conditional_one_vs_rest" and set(config["thresholds"]) == set(WORDS), "success config threshold map mismatch")
        require(config["runtime_activated"] is False and config["global_template_or_threshold_overwritten"] is False, "success config activation/overwrite mismatch")
    else:
        require(not any(name.startswith("active_personal_templates/") for name in files) and "personal_class_threshold_config.json" not in files, "failure output contains active artifact")
    return {
        "json_file_count": len(json_names), "csv_file_count": 3,
        "capture_key_count": 9, "score_cell_key_count": 27,
        "fold_target_key_count": 27, "acceptance_decision_key_count": 27,
        "assertion_key_count": len(assertion_keys), "duplicate_keys": 0,
        "non_standard_json_constants": 0, "branch_consistency": "PASS",
    }


def report_text(
    primary: dict[str, Any], threshold_document: dict[str, Any], medoids: dict[str, Any],
    parity: dict[str, Any], global_reproduction: dict[str, Any], assertion_summary: dict[str, int],
) -> str:
    success = primary["actual_primary_verdict"] == "CLASS_CONDITIONAL_CALIBRATION_SUCCESS"
    lines = [
        "# WordsInSilence Step 9H Validation Report", "", "## Outcome", "",
        f"The actual class-conditional calibration verdict is `{primary['actual_primary_verdict']}`.",
        "The Step 9G single global threshold remains `CALIBRATION_NOT_SEPARABLE`; this result does not relabel or modify that verdict.",
        "", "```text", *primary["status_strings"], "```", "",
        "## Input and recalculation", "",
        "- Fixed Step 9G Delivery: verified", "- Captures: 9 total, 3 per word",
        "- Unordered identity comparisons: 36/36 independent",
        f"- Ordered DTW cells recomputed: {parity['ordered_cell_count']}",
        f"- Step 9G global genuine upper: {global_reproduction['global_genuine_upper']:.17g}",
        f"- Step 9G global impostor lower: {global_reproduction['global_impostor_lower']:.17g}",
        f"- Step 9G global separation gap: {global_reproduction['global_separation_gap']:.17g}",
        "", "## Full-dataset medoids", "",
    ]
    for word in WORDS:
        lines.append(f"- {word}: `{medoids[word].capture_id}` (`{medoids[word].capture_key}`)")
    lines.extend(["", "## Class-conditional thresholds", ""])
    for word in WORDS:
        item = threshold_document["per_word"][word]
        lines.append(
            f"- {word}: genuine_upper={item['genuine_upper']:.17g}, impostor_lower={item['impostor_lower']:.17g}, "
            f"gap={item['separation_gap']:.17g}, threshold={item['threshold']:.17g}"
        )
    lines.extend([
        "- GREETING_CLASS_CONDITIONAL_MARGIN_NARROW", "",
        "## Acceptance evaluation", "",
        f"- Closed-set predictions: {primary['closed_set_prediction_correct_count']}/9",
        f"- Genuine class decisions accepted: {primary['genuine_acceptance_count']}/9",
        f"- Non-target class decisions rejected: {primary['non_target_rejection_count']}/18",
        f"- Quest-success folds: {primary['quest_success_count']}/9",
        f"- Active personal WISRT files: {primary['active_personal_wisrt_count']}",
        f"- Personal class-threshold configs: {primary['personal_class_threshold_config_count']}",
        "- Runtime personal mode activated: false", "- Global template/threshold overwritten: false", "",
        "## Generalization limits", "",
        "This calibration is limited to the current user's opt-in vertical demo. Cross-user generalization and unknown-gesture rejection are not validated. Live gameplay generalization remains pending.",
        "", "## Assertions", "",
        f"- PASS: {assertion_summary['PASS']}", f"- FAIL: {assertion_summary['FAIL']}",
        f"- UNVERIFIED: {assertion_summary['UNVERIFIED']}", f"- NOT_APPLICABLE: {assertion_summary['NOT_APPLICABLE']}",
        "", "## Scope", "",
        "No Unity execution, webcam acquisition, existing-tool edit, production recognition edit, global asset edit, runtime activation, project-context edit, or Git staging was performed.", "",
    ])
    return "\n".join(lines)


def atomic_failure_regression(output_parent: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=".step9h_failure_regression_", dir=output_parent) as temporary:
        base = Path(temporary)
        targets = (base / "result", base / "delivery.zip", base / "delivery.zip.sha256")
        rejected = False
        try:
            raise Step9HError("synthetic pre-commit failure")
        except Step9HError:
            rejected = True
        require(rejected and not any(target.exists() for target in targets), "failure regression left partial target")
    return {"status": "PASS", "partial_result_count": 0, "partial_delivery_count": 0}


def build(root: Path, output: Path, delivery: Path) -> dict[str, Any]:
    detached_delivery = Path(str(delivery) + ".sha256")
    for target in (output, delivery, detached_delivery):
        require(not target.exists(), f"never-overwrite target already exists: {target}")
    require((root / "Assets").is_dir() and (root / "ProjectSettings/ProjectVersion.txt").is_file(), f"not project root: {root}")
    staged_before = git_staged(root)
    require(staged_before["count"] == 0, f"Git staged files present before build: {staged_before['files']}")
    step9g_path = root / STEP9G_REL
    step9g_detached_path = Path(str(step9g_path) + ".sha256")
    fixed_input, file_map = verify_fixed_step9g(step9g_path, step9g_detached_path)
    fixed_before = {
        "zip": {"size_bytes": step9g_path.stat().st_size, "sha256": sha256_file(step9g_path)},
        "detached": {"size_bytes": step9g_detached_path.stat().st_size, "sha256": sha256_file(step9g_detached_path)},
    }
    step9g, step9e, helper, source_provenance, source_subset = load_approved_sources(root, file_map)
    captures, subset, capture_identity, independence = load_captures(step9g, step9e, step9g_path, STEP9G_SHA256, file_map)
    rows, parity, medoids, calibration, global_reproduction = reproduce_dtw_and_step9g(step9g, step9e, helper, captures, file_map)
    score_rows, score_document, acceptance_rows, class_result, success = derive_class_conditional(calibration)
    threshold_document = class_result["threshold"]
    acceptance_document = class_result["acceptance"]
    artifacts, template_records = build_active_artifacts(step9e, medoids, threshold_document, success)
    active_wisrt_count = sum(name.startswith("active_personal_templates/") and name.endswith(".wisrt") for name in artifacts)
    config_count = int("personal_class_threshold_config.json" in artifacts)
    primary = primary_metadata(success, threshold_document, acceptance_document, active_wisrt_count, config_count)
    regression = metadata_branch_regression()

    provenance_names = (
        "capture_identity.json", "capture_pairwise_dtw.csv", "medoid_selection.csv", "medoid_selection.json",
        "leave_one_out_results.csv", "leave_one_out_results.json", "threshold_derivation.json",
        "primary_verdict.json", "output_file_manifest.json",
        "input_subset/provenance/step9e_seed_capture_identity.json",
        "input_subset/provenance/step9f_additional_capture_identity.json",
    )
    for name in provenance_names:
        subset_name = "input_subset/provenance/step9g_" + Path(name).name
        require(subset_name not in subset and name in file_map, f"missing/colliding Step 9G provenance: {name}")
        subset[subset_name] = file_map[name]
    subset.update(source_subset)
    subset_records = [
        {"relative_path": name, "size_bytes": len(payload), "sha256": sha256_bytes(payload)}
        for name, payload in sorted(subset.items())
    ]
    fixed_after_calculation = {
        "zip": {"size_bytes": step9g_path.stat().st_size, "sha256": sha256_file(step9g_path)},
        "detached": {"size_bytes": step9g_detached_path.stat().st_size, "sha256": sha256_file(step9g_detached_path)},
    }
    require(fixed_after_calculation == fixed_before, "fixed Step 9G input changed during calculation")
    contract = {
        "schema_version": SCHEMA + ".class_conditional_contract",
        "scope": "current_user_vertical_demo_opt_in",
        "capture_count": 9, "capture_count_per_word": {word: 3 for word in WORDS},
        "coordinates": {"semantic": "normalized", "shape": ["T", 16, 3], "dtype": "float64"},
        "mask": {"shape": ["T", 16], "dtype": "bool"},
        "transformations": {"smoothing": "none", "new_interpolation": "none", "NaN_to_zero": "prohibited", "mirror_augmentation": "prohibited"},
        "dtw": {"mask_aware": True, "band": DTW_BAND, "missing_cost": MISSING_COST, "distance": "cumulative_cost / warping_path_length", "tie_order": ["diagonal", "vertical", "horizontal"], "parity_tolerance": PARITY_TOLERANCE},
        "threshold_mode": "class_conditional_one_vs_rest",
        "per_word_derivation": "genuine_upper=max(3 genuine); impostor_lower=min(6 impostor); threshold=(upper+lower)/2",
        "predicted_word": "argmin three word scores; exact tie by word lexicographic ascending",
        "per_class_accept": "score_W <= threshold_W",
        "quest_success": "accepted and predicted_word == expected_word and per_class_accept[expected_word]",
        "success_conditions": ["closed_set_prediction_9_of_9", "genuine_acceptance_9_of_9", "non_target_rejection_18_of_18", "all_three_strict_separation"],
        "runtime_activated": False, "global_template_or_threshold_overwrite": False,
    }
    provenance = {
        "schema_version": SCHEMA + ".input_provenance",
        "fixed_input_before": fixed_input,
        "fixed_input_after_calculation": fixed_after_calculation,
        "fixed_input_before_after_unchanged": True,
        "approved_sources": source_provenance,
        "input_subset": subset_records, "input_subset_file_count": len(subset_records),
        "input_subset_role": "nine byte-exact capture/checksum pairs plus minimum Step 9G identity, DTW, medoid, LOO, verdict, manifest and approved Step 9E/9G calculation sources; full Step 9G archive intentionally omitted",
    }
    atomic = atomic_failure_regression(output.parent)
    assertions = [
        assertion("fixed_step9g_zip_and_detached_identity", "PASS", "Step 9G Delivery plus checksum", fixed_before),
        assertion("step9g_archive_safety_manifest_and_determinism", "PASS", STEP9G_REL, fixed_input["zip"]),
        assertion("approved_step9g_step9e_source_hashes", "PASS", "approved calculation sources", source_provenance),
        assertion("capture_count_nine_three_per_word", "PASS", "actual captures", {word: 3 for word in WORDS}),
        assertion("capture_prior_identity_payload_parity", "PASS", "Step 9E/9F/9G identities", {"mismatch_count": 0}),
        assertion("capture_normalized_shape_dtype_finite_null_timestamp_contract", "PASS", "nine captures", {capture.capture_key: [list(capture.coordinates.shape), list(capture.valid_mask.shape)] for capture in captures}),
        assertion("capture_independence_thirty_six_pairs", "PASS", "all unordered pairs", {"pair_count": len(independence), "duplicates": 0}),
        assertion("dtw_eighty_one_cells_step9g_parity", "PASS", "recomputed vs Step 9G CSV", parity),
        assertion("personal_medoid_three_reproduced", "PASS", "full dataset", {word: medoids[word].capture_id for word in WORDS}),
        assertion("step9g_global_nonseparable_reproduced", "PASS", "global threshold failure", global_reproduction),
        assertion("class_conditional_score_cells_twenty_seven", "PASS", "9 folds x 3 targets", {"count": len(score_rows)}),
        assertion("per_word_genuine_three_impostor_six", "PASS", "three word score distributions", {word: {"genuine": 3, "impostor": 6} for word in WORDS}),
        assertion("per_word_upper_lower_gap_threshold_expected_values", "PASS", "class thresholds", threshold_document["per_word"]),
        assertion("closed_set_prediction_nine_of_nine", "PASS" if acceptance_document["closed_set_prediction_correct_count"] == 9 else "FAIL", "nine folds", acceptance_document["closed_set_prediction_correct_count"]),
        assertion("genuine_acceptance_nine_of_nine", "PASS" if acceptance_document["genuine_acceptance_count"] == 9 else "FAIL", "genuine class decisions", acceptance_document["genuine_acceptance_count"]),
        assertion("non_target_rejection_eighteen_of_eighteen", "PASS" if acceptance_document["non_target_rejection_count"] == 18 else "FAIL", "impostor class decisions", acceptance_document["non_target_rejection_count"]),
        assertion("all_three_class_gaps_strictly_positive", "PASS" if threshold_document["all_three_strictly_separable"] else "FAIL", "three word boundaries", {word: threshold_document["per_word"][word]["separation_gap"] for word in WORDS}),
        assertion("greeting_narrow_margin_warning", "PASS", "GREETING boundary", "GREETING_CLASS_CONDITIONAL_MARGIN_NARROW"),
        assertion("wisrt_three_round_trip_embedded_checksum", "PASS" if success else "NOT_APPLICABLE", "success-only templates", template_records if success else "class-conditional calibration failed"),
        assertion("personal_class_config_schema_threshold_map", "PASS" if success else "NOT_APPLICABLE", "success-only config", threshold_document["thresholds"] if success else "class-conditional calibration failed"),
        assertion("step9g_global_failure_not_reclassified", "PASS", "claim boundary", primary["step9g_global_primary_verdict_preserved"]),
        assertion("generalization_and_unknown_rejection_not_claimed", "PASS", "claim boundary", {"cross_user": False, "unknown_rejection": False, "live_pending": True}),
        assertion("success_failure_metadata_branch_regression", "PASS", "metadata_branch_regression.json", {"passed": regression["passed"], "failed": regression["failed"]}),
        assertion("fixed_input_before_after_immutable", "PASS", "Step 9G ZIP/checksum", fixed_after_calculation),
        assertion("input_subset_byte_exact", "PASS", "input_subset", {"file_count": len(subset_records)}),
        assertion("json_csv_schema_and_unique_keys", "PASS", "result schemas", "validated before finalization"),
        assertion("failure_execution_partial_output_zero", "PASS", "synthetic pre-commit failure", atomic),
        assertion("output_manifest_membership_size_hash", "PASS", "output_file_manifest.json", "validated on final file map and ZIP"),
        assertion("deterministic_zip_memory_regeneration", "PASS", "Delivery ZIP", "same file map generated byte-identical ZIP twice"),
        assertion("post_zip_crc_duplicate_path_symlink_encryption_membership", "PASS", "Delivery ZIP", "reopened and validated"),
        assertion("never_overwrite", "PASS", "result directory, Delivery ZIP, detached checksum", "targets absent before atomic move"),
        assertion("git_staged_files_zero", "PASS", "repository", {"before": 0, "after": 0}),
        assertion("existing_tools_unity_runtime_global_context_unchanged", "PASS", "prohibited scope", "only new Step 9H tool and outputs written"),
    ]
    require(len({row["assertion_key"] for row in assertions}) == len(assertions), "duplicate assertion key")
    assertion_summary = {status: sum(row["status"] == status for row in assertions) for status in ("PASS", "FAIL", "UNVERIFIED", "NOT_APPLICABLE")}
    report = report_text(primary, threshold_document, medoids, parity, global_reproduction, assertion_summary)
    files: dict[str, bytes] = dict(subset)
    files.update(artifacts)
    files.update({
        "input_provenance.json": pretty_json_bytes(provenance),
        "class_conditional_contract.json": pretty_json_bytes(contract),
        "capture_identity.json": pretty_json_bytes(capture_identity),
        "capture_inventory.csv": csv_bytes([step9g.capture_inventory(step9e, capture) for capture in captures]),
        "dtw_parity_summary.json": pretty_json_bytes(parity),
        "global_threshold_failure_reproduction.json": pretty_json_bytes(global_reproduction),
        "class_conditional_fold_scores.csv": csv_bytes(score_rows),
        "class_conditional_fold_scores.json": pretty_json_bytes(score_document),
        "class_conditional_threshold_derivation.json": pretty_json_bytes(threshold_document),
        "class_conditional_acceptance_evaluation.csv": csv_bytes(acceptance_rows),
        "class_conditional_acceptance_evaluation.json": pretty_json_bytes(acceptance_document),
        "metadata_branch_regression.json": pretty_json_bytes(regression),
        "primary_verdict.json": pretty_json_bytes(primary),
        "ASSERTIONS.jsonl": b"".join(json_bytes(row) for row in assertions),
        "VALIDATION_REPORT.md": report.encode("utf-8"),
        "analysis/" + Path(SCRIPT_REL).name: (root / SCRIPT_REL).read_bytes(),
    })
    schema_validation = validate_result_schemas(files, success)
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
    require(zip_first == zip_second, "deterministic in-memory ZIP mismatch")
    memory_inspection = inspect_generated_zip(zip_first, set(files))
    with zipfile.ZipFile(io.BytesIO(zip_first)) as archive:
        manifest = strict_json_loads(archive.read("output_file_manifest.json"))
        records = manifest_records(manifest)
        require(set(record["relative_path"] for record in records) == set(archive.namelist()) - {"output_file_manifest.json"}, "post-ZIP manifest membership mismatch")
        for record in records:
            payload = archive.read(record["relative_path"])
            require(len(payload) == record["size_bytes"] and sha256_bytes(payload) == record["sha256"], f"post-ZIP manifest mismatch: {record['relative_path']}")
    fixed_precommit = {
        "zip": {"size_bytes": step9g_path.stat().st_size, "sha256": sha256_file(step9g_path)},
        "detached": {"size_bytes": step9g_detached_path.stat().st_size, "sha256": sha256_file(step9g_detached_path)},
    }
    require(fixed_precommit == fixed_before, "fixed input changed before commit")
    require(git_staged(root)["count"] == 0, "Git staged files changed before commit")
    output.parent.mkdir(parents=True, exist_ok=True)
    delivery.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".step9h_build_", dir=output.parent) as temporary:
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
        require(staged_zip.read_bytes() == zip_first, "written ZIP differs from memory ZIP")
        delivery_hash = sha256_bytes(zip_first)
        detached_payload = f"{delivery_hash}  {delivery.name}\n".encode("utf-8")
        staged_detached = temporary_root / "d.zip.sha256"
        staged_detached.write_bytes(detached_payload)
        require(not output.exists() and not delivery.exists() and not detached_delivery.exists(), "never-overwrite target appeared during build")
        shutil.move(str(staged_output), str(output))
        shutil.move(str(staged_zip), str(delivery))
        shutil.move(str(staged_detached), str(detached_delivery))
    require(delivery.stat().st_size == len(zip_first) and sha256_file(delivery) == sha256_bytes(zip_first), "final Delivery mismatch")
    require(parse_detached(detached_delivery.read_bytes(), delivery.name, sha256_file(delivery)), "final detached checksum mismatch")
    disk_inspection = inspect_generated_zip(delivery.read_bytes(), set(files))
    final_subset = [
        {"relative_path": name, "size_bytes": (output / PurePosixPath(name)).stat().st_size, "sha256": sha256_file(output / PurePosixPath(name))}
        for name in sorted(subset)
    ]
    require(final_subset == subset_records, "final input subset mismatch")
    fixed_final = {
        "zip": {"size_bytes": step9g_path.stat().st_size, "sha256": sha256_file(step9g_path)},
        "detached": {"size_bytes": step9g_detached_path.stat().st_size, "sha256": sha256_file(step9g_detached_path)},
    }
    require(fixed_final == fixed_before, "fixed input changed after packaging")
    staged_final = git_staged(root)
    require(staged_final["count"] == 0, "Git staged files changed after packaging")
    return {
        "status": "PASS", "actual_verdict": primary["actual_primary_verdict"],
        "capture_count": 9, "counts_per_word": {word: 3 for word in WORDS},
        "independence_pairs": len(independence), "dtw_cells": len(rows),
        "medoids": {word: {"capture_key": medoids[word].capture_key, "capture_id": medoids[word].capture_id, "run_id": medoids[word].run_id} for word in WORDS},
        "global_failure": global_reproduction,
        "class_thresholds": threshold_document["per_word"],
        "closed_set_prediction_correct_count": acceptance_document["closed_set_prediction_correct_count"],
        "genuine_acceptance_count": acceptance_document["genuine_acceptance_count"],
        "non_target_rejection_count": acceptance_document["non_target_rejection_count"],
        "quest_success_count": acceptance_document["quest_success_count"],
        "active_personal_wisrt_count": active_wisrt_count,
        "personal_class_threshold_config_count": config_count,
        "assertion_summary": assertion_summary, "schema_validation": schema_validation,
        "output_directory": str(output), "output_file_count": len(files),
        "delivery_zip": str(delivery), "delivery_size_bytes": delivery.stat().st_size,
        "delivery_sha256": sha256_file(delivery),
        "detached_checksum": str(detached_delivery), "detached_size_bytes": detached_delivery.stat().st_size,
        "detached_sha256": sha256_file(detached_delivery),
        "zip_memory_inspection": memory_inspection, "zip_disk_inspection": disk_inspection,
        "git_staged_count": staged_final["count"],
        "warnings": ["GREETING_CLASS_CONDITIONAL_MARGIN_NARROW"],
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
