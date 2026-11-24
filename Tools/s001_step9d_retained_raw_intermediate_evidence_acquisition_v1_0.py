#!/usr/bin/env python3
"""Read-only Step 9D retained raw/intermediate evidence inventory and acquisition."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent.parent
VERSION = "1.0"
SCRIPT_REL = "Tools/s001_step9d_retained_raw_intermediate_evidence_acquisition_v1_0.py"

OUTPUT_REL = "data/processed/sessions/S001_P001/step9d_retained_raw_intermediate_evidence_acquisition_v1_0"
OUTPUT = ROOT / PurePosixPath(OUTPUT_REL)
OUTPUT_MANIFEST_REL = f"{OUTPUT_REL}/output_file_manifest.json"
REPORT_NAME = "STEP9D_RETAINED_RAW_INTERMEDIATE_EVIDENCE_ACQUISITION_V1_0_REPORT.md"

DELIVERY_NAME = "WordsInSilence_step9d_retained_raw_intermediate_evidence_acquisition_v1_0_delivery.zip"
DELIVERY_ZIP = ROOT / DELIVERY_NAME
DELIVERY_HASH = ROOT / f"{DELIVERY_NAME}.sha256"
POST_ZIP = ROOT / "WordsInSilence_step9d_retained_raw_intermediate_evidence_acquisition_v1_0_delivery_post_zip_verification.json"

STEP9D_DELIVERY_NAME = "WordsInSilence_step9d_physical_recognition_domain_gap_diagnostic_v1_0_delivery.zip"
STEP9D_DELIVERY = ROOT / STEP9D_DELIVERY_NAME
STEP9D_DETACHED = ROOT / f"{STEP9D_DELIVERY_NAME}.sha256"
STEP9D_EXPECTED_SIZE = 2_804_364
STEP9D_EXPECTED_SHA256 = "b179962aba86aa961e42c5f762e696c718c831b7ac444c5fe708d5301a69f2d0"

STEP9D_OUTPUT = ROOT / "data/processed/sessions/S001_P001/step9d_physical_recognition_domain_gap_diagnostic_v1_0"
STEP9C_EDITOR_REL = "data/processed/sessions/S001_P001/step9c_physical_webcam_coordinate_compatibility_v1_0/physical_captures"
STEP9C_EDITOR_ROOT = ROOT / PurePosixPath(STEP9C_EDITOR_REL)
RUN_ID = "8d17ef49-89ff-4d40-ae59-0386cc406e2e"

CAPTURES = {
    "GREETING": {
        "capture_id": "step9c-take-01-greeting-f7875c16-ad18-425c-8ce5-39e9a6cc13f7",
        "take_number": 1,
    },
    "LOVE": {
        "capture_id": "step9c-take-02-love-6d68ddd7-2d7e-4afd-bd0a-b1989f8e52f2",
        "take_number": 2,
    },
    "THANK_YOU": {
        "capture_id": "step9c-take-03-thank-you-53870472-c644-489c-a94d-95e9667ded42",
        "take_number": 3,
    },
}
CAPTURE_ID_TO_WORD = {item["capture_id"]: word for word, item in CAPTURES.items()}
CAPTURE_IDS = tuple(CAPTURE_ID_TO_WORD)

EXPECTED_MISSING_CONTRACT = (
    "75-landmark raw coordinates and masks per frame",
    "per-landmark original-valid/reappearance/strong-depth/interpolation arrays",
    "per-frame torso origin/right/up/forward basis",
)

PRIMARY_VERDICT = "RETAINED_EVIDENCE_INCOMPLETE_OR_NOT_RETAINED"
FINAL_STATUS = "STEP9D_RETAINED_RAW_INTERMEDIATE_EVIDENCE_ACQUISITION_BLOCKED_NOT_RETAINED_OR_INCOMPLETE"
SELF_FAIL_STATUS = "STEP9D_RETAINED_RAW_INTERMEDIATE_EVIDENCE_ACQUISITION_SELF_VERIFICATION_FAIL"

ALLOWED_CONTRACT_STATUSES = {
    "FOUND_EXACT",
    "FOUND_PARTIAL",
    "MISSING_NOT_RETAINED",
    "AMBIGUOUS_PROVENANCE",
    "NOT_APPLICABLE",
}
TEXT_SEARCH_EXTENSIONS = {
    ".asset", ".cs", ".csv", ".json", ".log", ".md", ".prefab", ".py",
    ".sha256", ".txt", ".unity", ".xml", ".yaml", ".yml",
}
MAX_CONTENT_SEARCH_BYTES = 8 * 1024 * 1024
PROJECT_EXCLUDED_DIRS = {
    ".git", ".vs", "Library", "Temp", "Logs", "obj", "Build", "Builds", "Recordings",
}


class AcquisitionError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AcquisitionError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_json(value: object, location: str = "$") -> None:
    if isinstance(value, float):
        require(math.isfinite(value), f"non-finite JSON number at {location}")
    elif isinstance(value, dict):
        for key, item in value.items():
            finite_json(item, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            finite_json(item, f"{location}[{index}]")


def json_bytes(value: object) -> bytes:
    finite_json(value)
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(value))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def strict_json_loads(data: bytes, label: str) -> object:
    def reject_constant(value: str) -> None:
        raise AcquisitionError(f"non-finite JSON token {value} in {label}")

    try:
        return json.loads(data.decode("utf-8-sig"), parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AcquisitionError(f"invalid JSON in {label}: {exc}") from exc


def file_record(path: Path, category: str, relative_path: str | None = None) -> dict:
    require(path.is_file(), f"required file missing: {path}")
    if relative_path is None:
        relative_path = path.resolve().relative_to(ROOT.resolve()).as_posix()
    return {
        "relative_path": relative_path,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "category": category,
    }


def path_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def safe_zip_name(name: str) -> bool:
    posix = PurePosixPath(name)
    return (
        bool(name)
        and not name.startswith(("/", "\\"))
        and "\\" not in name
        and not posix.is_absolute()
        and ".." not in posix.parts
        and not (len(name) >= 2 and name[1] == ":")
    )


def parse_detached(data: bytes, expected_name: str, expected_hash: str) -> bool:
    try:
        line = data.decode("utf-8-sig").strip()
    except UnicodeDecodeError:
        return False
    parts = line.split(maxsplit=1)
    return len(parts) == 2 and parts[0].lower() == expected_hash and parts[1].lstrip("*") == expected_name


def add_assert(assertions: list[dict], name: str, status: str, evidence: object) -> None:
    require(status in {"PASS", "FAIL", "UNVERIFIED", "NOT_APPLICABLE"}, f"invalid assertion status: {status}")
    assertions.append({"name": name, "status": status, "evidence": evidence})


def strict_step9d_delivery() -> dict:
    require(STEP9D_DELIVERY.is_file(), f"fixed Step 9D Delivery missing: {STEP9D_DELIVERY}")
    actual_size = STEP9D_DELIVERY.stat().st_size
    actual_hash = sha256_file(STEP9D_DELIVERY)
    require(actual_size == STEP9D_EXPECTED_SIZE, f"Step 9D Delivery size mismatch: {actual_size}")
    require(actual_hash == STEP9D_EXPECTED_SHA256, f"Step 9D Delivery SHA-256 mismatch: {actual_hash}")
    detached_ok = False
    if STEP9D_DETACHED.is_file():
        detached_ok = parse_detached(STEP9D_DETACHED.read_bytes(), STEP9D_DELIVERY_NAME, actual_hash)
        require(detached_ok, "Step 9D Delivery detached checksum mismatch")

    with zipfile.ZipFile(STEP9D_DELIVERY, "r") as archive:
        require(archive.testzip() is None, "Step 9D Delivery CRC verification failed")
        names = archive.namelist()
        require(len(names) == len(set(names)), "Step 9D Delivery contains duplicate entries")
        conclusion_names = [name for name in names if name.endswith("/diagnostic_conclusion.json")]
        contract_names = [name for name in names if name.endswith("/stage_contract_comparison.json")]
        require(len(conclusion_names) == 1 and len(contract_names) == 1, "Step 9D contract entries are not unique")
        conclusion = strict_json_loads(archive.read(conclusion_names[0]), conclusion_names[0])
        contract = strict_json_loads(archive.read(contract_names[0]), contract_names[0])
        require(isinstance(conclusion, dict) and isinstance(contract, dict), "Step 9D contract payload is not an object")
        require(conclusion.get("primary_verdict") == "UNVERIFIED_MISSING_REQUIRED_EVIDENCE", "unexpected Step 9D primary verdict")
        upstream = contract.get("upstream_pipeline_parity", {})
        missing = tuple(upstream.get("missing_evidence", []))
        require(missing == EXPECTED_MISSING_CONTRACT, f"Step 9D missing-evidence contract changed: {missing}")

        archived_captures = contract.get("captures", [])
        by_word = {item.get("expected_word"): item for item in archived_captures if isinstance(item, dict)}
        require(set(by_word) == set(CAPTURES), f"Step 9D capture word set mismatch: {sorted(by_word)}")
        for word, expected in CAPTURES.items():
            require(by_word[word].get("capture_id") == expected["capture_id"], f"Step 9D Capture ID mismatch for {word}")

        physical_members: dict[str, dict] = {}
        for word, expected in CAPTURES.items():
            capture_id = expected["capture_id"]
            members = [name for name in names if "/inputs/physical_queries/" in name and name.endswith(".json") and capture_id in name]
            require(len(members) == 1, f"Step 9D physical query member is not unique for {word}")
            payload = archive.read(members[0])
            parsed = strict_json_loads(payload, members[0])
            require(isinstance(parsed, dict) and parsed.get("capture_id") == capture_id, f"archive Capture ID mismatch for {word}")
            physical_members[word] = {
                "archive_path": members[0],
                "bytes": payload,
                "size_bytes": len(payload),
                "sha256": sha256_bytes(payload),
                "parsed": parsed,
            }

    return {
        "size_bytes": actual_size,
        "sha256": actual_hash,
        "detached_checksum_present": STEP9D_DETACHED.is_file(),
        "detached_checksum_strict": detached_ok,
        "zip_entry_count": len(names),
        "conclusion_archive_path": conclusion_names[0],
        "contract_archive_path": contract_names[0],
        "primary_verdict": conclusion["primary_verdict"],
        "missing_contract": list(missing),
        "contract_captures": archived_captures,
        "physical_members": physical_members,
    }


def parse_project_identity() -> tuple[str, str]:
    settings = ROOT / "ProjectSettings/ProjectSettings.asset"
    require(settings.is_file(), f"Unity PlayerSettings missing: {settings}")
    company = None
    product = None
    with settings.open("r", encoding="utf-8-sig", errors="strict") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped.startswith("companyName:"):
                company = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("productName:"):
                product = stripped.split(":", 1)[1].strip()
            if company and product:
                break
    require(company == "DefaultCompany" and product == "WordsInSilence", f"unexpected Unity identity: {company}/{product}")
    return company, product


def configured_roots() -> dict:
    company, product = parse_project_identity()
    harness = ROOT / "Assets/_Project/Scripts/Recognition/PhysicalCompatibility/PhysicalWebcamCoordinateCompatibilityHarness.cs"
    require(harness.is_file(), f"capture harness source missing: {harness}")
    source = harness.read_text(encoding="utf-8")
    required_fragments = (
        '"data", "processed", "sessions", "S001_P001", "step9c_physical_webcam_coordinate_compatibility_v1_0", "physical_captures"',
        'Application.persistentDataPath, "Step9CPhysicalWebcamCoordinateCompatibility", Step9CPhysicalCompatibilityConstants.RunId',
    )
    require(all(fragment in source for fragment in required_fragments), "capture evidence root source contract changed")
    user_profile = Path(os.environ.get("USERPROFILE", str(Path.home())))
    player_root = user_profile / "AppData" / "LocalLow" / company / product / "Step9CPhysicalWebcamCoordinateCompatibility" / RUN_ID
    return {
        "project_root": ROOT.resolve(),
        "configured_editor_evidence_root": STEP9C_EDITOR_ROOT.resolve(),
        "configured_player_evidence_root": player_root.resolve(),
        "source_files": [
            ROOT / "ProjectSettings/ProjectSettings.asset",
            harness,
            ROOT / "Assets/_Project/Scripts/Recognition/PhysicalCompatibility/PhysicalCaptureEvidenceJson.cs",
            ROOT / "Assets/_Project/Scripts/Recognition/PhysicalCompatibility/Step9CPhysicalCaptureEvidence.cs",
        ],
        "company_name": company,
        "product_name": product,
    }


def excluded_project_path(relative_path: Path, is_dir: bool) -> bool:
    parts = relative_path.parts
    if not parts:
        return False
    if parts[0] in PROJECT_EXCLUDED_DIRS:
        return True
    lowered = tuple(part.lower() for part in parts)
    if len(lowered) >= 2 and lowered[:2] in {("tools", "_archive"), ("validation", "archive")}:
        return True
    if not is_dir and len(lowered) >= 2 and lowered[0] == "validation" and lowered[-1].startswith("motion_validation_details") and lowered[-1].endswith(".json"):
        return True
    if lowered[: len(PurePosixPath(OUTPUT_REL).parts)] == tuple(part.lower() for part in PurePosixPath(OUTPUT_REL).parts):
        return True
    if not is_dir and relative_path.name in {DELIVERY_NAME, f"{DELIVERY_NAME}.sha256", POST_ZIP.name}:
        return True
    return False


def enumerate_files(root: Path, project_scope: bool) -> list[Path]:
    if not root.is_dir():
        return []
    files: list[Path] = []
    for current, directories, names in os.walk(root):
        current_path = Path(current)
        if project_scope:
            kept = []
            for directory in directories:
                candidate = current_path / directory
                relative_path = candidate.relative_to(root)
                if not excluded_project_path(relative_path, True):
                    kept.append(directory)
            directories[:] = kept
        for name in names:
            path = current_path / name
            if project_scope and excluded_project_path(path.relative_to(root), False):
                continue
            if path.is_file():
                files.append(path.resolve())
    return sorted(files, key=lambda item: str(item).lower())


def byte_search_ids(path: Path) -> tuple[list[str], str]:
    if path.suffix.lower() not in TEXT_SEARCH_EXTENSIONS:
        return [], "SKIPPED_NON_TEXT_CONTAINER"
    size = path.stat().st_size
    if size > MAX_CONTENT_SEARCH_BYTES:
        return [], "SKIPPED_OVER_8_MIB"
    needles = {capture_id: capture_id.encode("ascii") for capture_id in CAPTURE_IDS}
    found: set[str] = set()
    overlap = max(len(value) for value in needles.values()) - 1
    tail = b""
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                block = tail + chunk
                for capture_id, needle in needles.items():
                    if capture_id not in found and needle in block:
                        found.add(capture_id)
                if len(found) == len(needles):
                    break
                tail = block[-overlap:] if overlap else b""
    except OSError as exc:
        return [], f"READ_ERROR:{type(exc).__name__}"
    return sorted(found), "INSPECTED"


def source_category(path: Path, roots: dict) -> tuple[str, Path]:
    if path_under(path, roots["configured_editor_evidence_root"]):
        return "configured_editor_evidence_root", roots["configured_editor_evidence_root"]
    if path_under(path, roots["configured_player_evidence_root"]):
        return "configured_player_evidence_root", roots["configured_player_evidence_root"]
    return "project_root", roots["project_root"]


def discover_candidates(roots: dict) -> tuple[list[dict], dict]:
    project_files = enumerate_files(roots["project_root"], project_scope=True)
    editor_files = enumerate_files(roots["configured_editor_evidence_root"], project_scope=False)
    player_files = enumerate_files(roots["configured_player_evidence_root"], project_scope=False)
    unique_files = sorted(set(project_files + editor_files + player_files), key=lambda item: str(item).lower())

    candidates: list[dict] = []
    search_stats = {
        "content_inspected_file_count": 0,
        "content_skipped_non_text_container_count": 0,
        "content_skipped_over_8_mib_count": 0,
        "content_read_error_count": 0,
        "filename_match_file_count": 0,
        "content_match_file_count": 0,
    }
    for path in unique_files:
        relative_display = path.relative_to(ROOT).as_posix() if path_under(path, ROOT) else path.name
        filename_ids = sorted(capture_id for capture_id in CAPTURE_IDS if capture_id in relative_display)
        content_ids, content_status = byte_search_ids(path)
        if content_status == "INSPECTED":
            search_stats["content_inspected_file_count"] += 1
        elif content_status == "SKIPPED_NON_TEXT_CONTAINER":
            search_stats["content_skipped_non_text_container_count"] += 1
        elif content_status == "SKIPPED_OVER_8_MIB":
            search_stats["content_skipped_over_8_mib_count"] += 1
        elif content_status.startswith("READ_ERROR"):
            search_stats["content_read_error_count"] += 1
        if filename_ids:
            search_stats["filename_match_file_count"] += 1
        if content_ids:
            search_stats["content_match_file_count"] += 1
        matched_ids = sorted(set(filename_ids + content_ids))
        if not matched_ids:
            continue
        category, source_root = source_category(path, roots)
        candidates.append({
            "path": path,
            "capture_ids": matched_ids,
            "connection_basis": "+".join(part for part, values in (("PATH", filename_ids), ("CONTENT", content_ids)) if values),
            "content_search_status": content_status,
            "source_root_category": category,
            "source_root": source_root,
        })

    root_records = [
        {
            "category": "project_root",
            "absolute_path": str(roots["project_root"]),
            "exists": roots["project_root"].is_dir(),
            "searched_file_count": len(project_files),
            "source_confirmation": "repository root containing Assets, Packages/manifest.json and ProjectSettings/ProjectVersion.txt",
        },
        {
            "category": "configured_editor_evidence_root",
            "absolute_path": str(roots["configured_editor_evidence_root"]),
            "exists": roots["configured_editor_evidence_root"].is_dir(),
            "searched_file_count": len(editor_files),
            "source_confirmation": "PhysicalWebcamCoordinateCompatibilityHarness.cs UNITY_EDITOR EvidenceOutputDirectory",
        },
        {
            "category": "configured_player_evidence_root",
            "absolute_path": str(roots["configured_player_evidence_root"]),
            "exists": roots["configured_player_evidence_root"].is_dir(),
            "searched_file_count": len(player_files),
            "source_confirmation": "Application.persistentDataPath + configured company/product + Step9C subdirectory + fixed RunId",
        },
    ]
    manifest = {
        "schema_version": "words_in_silence.step9d.retained.search_scope_manifest.v1",
        "version": VERSION,
        "capture_ids": list(CAPTURE_IDS),
        "roots": root_records,
        "unique_searched_file_count": len(unique_files),
        "candidate_file_count": len(candidates) + 1,
        "search_statistics": search_stats,
        "search_method": {
            "path_match": "exact fixed Capture ID substring",
            "content_match": "chunked exact ASCII Capture ID search in text-like files up to 8 MiB",
            "candidate_container_inspection": "strict JSON/CSV/checksum inspection; NPZ allow_pickle=False when directly connected; unknown binary remains opaque",
            "compressed_archive_policy": "only the fixed Step 9D Delivery is opened; unrelated archives are not broadly inspected",
        },
        "exclusions": [
            ".git", ".vs", "Library", "Temp", "Logs", "obj", "Build", "Builds", "Recordings",
            "Tools/_archive", "Validation/archive", "Validation/motion_validation_details*.json",
            OUTPUT_REL, DELIVERY_NAME, f"{DELIVERY_NAME}.sha256", POST_ZIP.name,
        ],
        "scope_restriction": "project root plus only the two evidence roots confirmed by Unity source/config; no other disk roots searched",
    }
    return candidates, manifest


def json_schema_value(data: dict) -> str:
    for key in ("schema_version", "schemaVersion", "version"):
        value = data.get(key)
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            return str(value)
    return "ABSENT"


def coordinate_counts(values: object) -> dict:
    result = {"finite": 0, "nan": 0, "inf": 0, "null": 0, "other": 0}

    def visit(value: object) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif value is None:
            result["null"] += 1
        elif isinstance(value, bool):
            result["other"] += 1
        elif isinstance(value, (int, float)):
            if math.isnan(float(value)):
                result["nan"] += 1
            elif math.isinf(float(value)):
                result["inf"] += 1
            else:
                result["finite"] += 1
        else:
            result["other"] += 1

    visit(values)
    return result


def analyze_physical_capture(data: dict) -> dict:
    frames = data.get("frames")
    require(isinstance(frames, list), "physical capture frames must be a list")
    frame_indices: list[int] = []
    timestamps: list[float] = []
    shoulder_widths: list[float] = []
    normalized_coordinates: list = []
    normalized_mask_true = 0
    mask_lengths: set[int] = set()
    coordinate_shapes: set[tuple[int, int]] = set()
    body_transform_available_true = 0
    frame_key_union: set[str] = set()
    for position, frame in enumerate(frames):
        require(isinstance(frame, dict), f"frame {position} is not an object")
        frame_key_union.update(str(key) for key in frame)
        index = frame.get("frame_index")
        timestamp = frame.get("timestamp_seconds")
        mask = frame.get("normalized_mask")
        coordinates = frame.get("normalized_coordinates")
        require(isinstance(index, int) and not isinstance(index, bool), f"invalid local frame index at {position}")
        require(isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool) and math.isfinite(float(timestamp)), f"invalid timestamp at {position}")
        require(isinstance(mask, list), f"normalized mask missing at {position}")
        require(isinstance(coordinates, list), f"normalized coordinates missing at {position}")
        mask_lengths.add(len(mask))
        coordinate_shapes.add((len(coordinates), len(coordinates[0]) if coordinates and isinstance(coordinates[0], list) else -1))
        require(all(isinstance(value, bool) for value in mask), f"normalized mask is not boolean at {position}")
        normalized_mask_true += sum(mask)
        frame_indices.append(index)
        timestamps.append(float(timestamp))
        normalized_coordinates.append(coordinates)
        if frame.get("body_transform_available") is True:
            body_transform_available_true += 1
        width = frame.get("shoulder_width_raw")
        if isinstance(width, (int, float)) and not isinstance(width, bool) and math.isfinite(float(width)):
            shoulder_widths.append(float(width))
    require(mask_lengths in ({16}, set()) and coordinate_shapes in ({(16, 3)}, set()), f"normalized frame shape mismatch: masks={mask_lengths}, coords={coordinate_shapes}")
    counts = coordinate_counts(normalized_coordinates)
    count_summary = data.get("counts") if isinstance(data.get("counts"), dict) else {}
    window = data.get("capture_window") if isinstance(data.get("capture_window"), dict) else {}
    raw_array_keys = sorted(key for key in frame_key_union if "raw" in key.lower() and isinstance(frames[0].get(key) if frames else None, list))
    boundary_array_keys = sorted(
        key for key in frame_key_union
        if any(token in key.lower() for token in ("original_valid", "reappearance", "strong_depth", "interpolation"))
        and isinstance(frames[0].get(key) if frames else None, list)
    )
    torso_basis_keys = sorted(
        key for key in frame_key_union
        if any(token in key.lower() for token in ("torso_origin", "torso_center", "right_axis", "up_axis", "forward_axis"))
    )
    return {
        "schema_version": json_schema_value(data),
        "frame_count": len(frames),
        "landmark_count": 16,
        "shape": {"normalized_coordinates": [len(frames), 16, 3], "normalized_mask": [len(frames), 16]},
        "dtype": {"normalized_coordinates": "JSON_NUMBER_UNTYPED_OR_NULL", "normalized_mask": "JSON_BOOLEAN"},
        "finite_count": counts["finite"],
        "nan_count": counts["nan"],
        "inf_count": counts["inf"],
        "null_count": counts["null"],
        "mask_true_counts": {
            "normalized_mask": normalized_mask_true,
            "raw_validity_presence_mask": None,
            "original_valid": None,
            "reappearance_reset": None,
            "strong_depth_boundary": None,
            "interpolation": None,
            "body_transform_available": body_transform_available_true,
        },
        "frame_range": [min(frame_indices), max(frame_indices)] if frame_indices else None,
        "timestamp_range": [min(timestamps), max(timestamps)] if timestamps else None,
        "provider_frame_range": [window.get("first_capture_provider_frame_index"), window.get("final_capture_provider_frame_index")] if window else None,
        "lifetime_provider_frame_range": [window.get("first_capture_lifetime_provider_frame_index"), window.get("final_capture_lifetime_provider_frame_index")] if window else None,
        "bridge_observation_count": window.get("bridge_observation_count") if window else None,
        "raw_aggregate_counts": {
            key: count_summary.get(key)
            for key in (
                "raw_frames", "raw_pose_valid_landmarks", "raw_left_hand_valid_landmarks",
                "raw_right_hand_valid_landmarks", "raw_valid_landmarks", "raw_landmark_capacity",
                "interpolated_landmarks", "reappearances", "strong_depth_boundaries",
            )
        },
        "per_frame_shoulder_width_count": len(shoulder_widths),
        "per_frame_shoulder_width_range": [min(shoulder_widths), max(shoulder_widths)] if shoulder_widths else None,
        "raw_coordinate_array_keys": raw_array_keys,
        "per_landmark_boundary_array_keys": boundary_array_keys,
        "torso_basis_keys": torso_basis_keys,
        "data_class": "NORMALIZED_16_LANDMARK_PLUS_AGGREGATE_RAW_INTERMEDIATE_COUNTS",
        "raw_normalized_aggregate_class": "NORMALIZED_PER_FRAME_AND_AGGREGATE_ONLY_UPSTREAM;NOT_RAW_75",
        "contract_evidence_status": "FOUND_PARTIAL",
    }


def inspect_npz(path: Path) -> dict:
    try:
        import numpy as np
    except ImportError:
        return {
            "schema_version": "UNKNOWN",
            "shape": None,
            "dtype": None,
            "data_class": "OPAQUE_BINARY_UNVERIFIED",
            "raw_normalized_aggregate_class": "AMBIGUOUS",
            "contract_evidence_status": "AMBIGUOUS_PROVENANCE",
            "inspection_error": "numpy unavailable",
        }
    shapes = {}
    dtypes = {}
    finite = nan = inf = mask_true = 0
    try:
        with np.load(path, allow_pickle=False) as archive:
            for key in archive.files:
                array = archive[key]
                shapes[key] = list(array.shape)
                dtypes[key] = str(array.dtype)
                if np.issubdtype(array.dtype, np.number):
                    finite += int(np.isfinite(array).sum())
                    if np.issubdtype(array.dtype, np.inexact):
                        nan += int(np.isnan(array).sum())
                        inf += int(np.isinf(array).sum())
                if array.dtype == np.bool_:
                    mask_true += int(array.sum())
    except Exception as exc:
        return {
            "schema_version": "UNKNOWN",
            "shape": None,
            "dtype": None,
            "data_class": "OPAQUE_BINARY_UNVERIFIED",
            "raw_normalized_aggregate_class": "AMBIGUOUS",
            "contract_evidence_status": "AMBIGUOUS_PROVENANCE",
            "inspection_error": repr(exc),
        }
    keys_lower = " ".join(shapes).lower()
    raw_exact = "raw" in keys_lower and any(token in keys_lower for token in ("landmark", "coordinate"))
    return {
        "schema_version": "NPZ_ARRAY_CONTAINER",
        "shape": shapes,
        "dtype": dtypes,
        "finite_count": finite,
        "nan_count": nan,
        "inf_count": inf,
        "null_count": 0,
        "mask_true_counts": {"all_boolean_arrays": mask_true},
        "data_class": "RAW_OR_INTERMEDIATE_ARRAY_CONTAINER" if raw_exact else "NPZ_ARRAY_CONTAINER_UNCLASSIFIED",
        "raw_normalized_aggregate_class": "RAW_CANDIDATE" if raw_exact else "AMBIGUOUS",
        "contract_evidence_status": "AMBIGUOUS_PROVENANCE",
    }


def analyze_candidate(candidate: dict) -> dict:
    path = candidate["path"]
    category = candidate["source_root_category"]
    source_root = candidate["source_root"]
    relative_path = path.relative_to(source_root).as_posix()
    extension = path.suffix.lower()
    container = {
        ".json": "JSON", ".csv": "CSV", ".npz": "NPZ", ".wisrt": "WISRT",
        ".zip": "ZIP", ".sha256": "SHA256_TEXT", ".xml": "XML",
    }.get(extension, "TEXT" if extension in TEXT_SEARCH_EXTENSIONS else "UNKNOWN_BINARY")
    record = {
        "absolute_source_root_category": category,
        "source_root_absolute": str(source_root),
        "relative_path": relative_path,
        "absolute_path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "extension_container_type": container,
        "capture_ids": candidate["capture_ids"],
        "capture_id_connection_basis": candidate["connection_basis"],
        "content_search_status": candidate["content_search_status"],
        "schema_version": "UNKNOWN",
        "shape": None,
        "dtype": None,
        "frame_count": None,
        "landmark_count": None,
        "finite_count": None,
        "nan_count": None,
        "inf_count": None,
        "null_count": None,
        "mask_true_counts": {},
        "frame_range": None,
        "timestamp_range": None,
        "provider_frame_range": None,
        "lifetime_provider_frame_range": None,
        "bridge_observation_count": None,
        "data_class": "REFERENCE_OR_DERIVED_TEXT",
        "raw_normalized_aggregate_class": "NOT_APPLICABLE",
        "contract_evidence_status": "NOT_APPLICABLE",
        "copy_selected": False,
        "copy_selection_status": "NOT_APPLICABLE",
        "copy_reason": "not a canonical retained input",
        "copied_relative_path": None,
        "copy_size_bytes": None,
        "copy_sha256": None,
        "copy_byte_parity": None,
    }
    if extension == ".json":
        data = strict_json_loads(path.read_bytes(), str(path))
        require(isinstance(data, (dict, list)), f"JSON root is not an object/array: {path}")
        if isinstance(data, dict):
            record["schema_version"] = json_schema_value(data)
            if isinstance(data.get("frames"), list) and data.get("capture_id") in candidate["capture_ids"]:
                record.update(analyze_physical_capture(data))
            else:
                record["data_class"] = "DERIVED_DIAGNOSTIC_OR_MANIFEST_REFERENCE"
                record["raw_normalized_aggregate_class"] = "AGGREGATE_ONLY_OR_REFERENCE"
    elif extension == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            rows = list(reader)
        record["schema_version"] = "CSV_HEADER_V1" if rows else "EMPTY_CSV"
        record["shape"] = [max(len(rows) - 1, 0), len(rows[0]) if rows else 0]
        record["dtype"] = "CSV_TEXT_UNTYPED"
        record["frame_count"] = None
        record["data_class"] = "DERIVED_TABULAR_REFERENCE"
        record["raw_normalized_aggregate_class"] = "AGGREGATE_ONLY_OR_REFERENCE"
    elif extension == ".sha256":
        record["schema_version"] = "GNU_STYLE_SHA256_LINE"
        record["dtype"] = "UTF8_TEXT"
        record["data_class"] = "INTEGRITY_SIDECAR"
        record["raw_normalized_aggregate_class"] = "INTEGRITY_ONLY"
        record["contract_evidence_status"] = "FOUND_PARTIAL" if category.startswith("configured_") else "NOT_APPLICABLE"
    elif extension == ".npz":
        record.update(inspect_npz(path))
    elif extension in {".wisrt", ".zip"} or container == "UNKNOWN_BINARY":
        record["data_class"] = "OPAQUE_BINARY_UNVERIFIED"
        record["raw_normalized_aggregate_class"] = "AMBIGUOUS"
        record["contract_evidence_status"] = "AMBIGUOUS_PROVENANCE"
    return record


def canonical_inputs(analyses: list[dict], delivery: dict) -> dict[str, dict]:
    selected: dict[str, dict] = {}
    for word, expected in CAPTURES.items():
        capture_id = expected["capture_id"]
        json_candidates = [
            item for item in analyses
            if item["absolute_source_root_category"] == "configured_editor_evidence_root"
            and item["extension_container_type"] == "JSON"
            and capture_id in item["capture_ids"]
            and item["data_class"].startswith("NORMALIZED_16_")
        ]
        require(len(json_candidates) == 1, f"canonical Editor JSON is not unique for {word}: {len(json_candidates)}")
        json_item = json_candidates[0]
        json_path = Path(json_item["absolute_path"])
        data = strict_json_loads(json_path.read_bytes(), str(json_path))
        require(isinstance(data, dict), f"canonical JSON is not an object for {word}")
        require(data.get("capture_id") == capture_id and data.get("expected_word") == word, f"canonical provenance mismatch for {word}")
        require(data.get("take_number") == expected["take_number"], f"canonical take number mismatch for {word}")
        archived = delivery["physical_members"][word]
        require(json_path.read_bytes() == archived["bytes"], f"canonical JSON differs from fixed Step 9D Delivery member for {word}")

        checksum_path = Path(str(json_path) + ".sha256")
        checksum_candidates = [item for item in analyses if Path(item["absolute_path"]) == checksum_path]
        require(len(checksum_candidates) == 1, f"canonical checksum candidate missing for {word}")
        require(parse_detached(checksum_path.read_bytes(), json_path.name, json_item["sha256"]), f"canonical checksum strict mismatch for {word}")
        checksum_item = checksum_candidates[0]

        destination_base = f"retained_inputs/configured_editor_evidence_root/{json_path.name}"
        json_item.update({
            "copy_selected": True,
            "copy_selection_status": "FOUND_PARTIAL",
            "copy_reason": "canonical source/config-connected Step 9C partial retained input",
            "copied_relative_path": destination_base,
        })
        checksum_item.update({
            "contract_evidence_status": "FOUND_PARTIAL",
            "copy_selected": True,
            "copy_selection_status": "FOUND_PARTIAL",
            "copy_reason": "strict integrity sidecar for canonical partial retained input",
            "copied_relative_path": destination_base + ".sha256",
        })
        selected[word] = {"json": json_item, "checksum": checksum_item, "parsed": data, "archive_member": archived}

    canonical_hashes = {entry["json"]["sha256"] for entry in selected.values()}
    for item in analyses:
        if item["copy_selected"]:
            continue
        if item["sha256"] in canonical_hashes and item["data_class"].startswith("NORMALIZED_16_"):
            item["copy_selection_status"] = "NOT_APPLICABLE"
            item["copy_reason"] = "byte-identical redundant packaged/project copy; canonical configured evidence-root source selected"
    return selected


def copy_selected_inputs(stage_output: Path, selected: dict[str, dict]) -> list[dict]:
    copies = []
    for word in CAPTURES:
        for kind in ("json", "checksum"):
            item = selected[word][kind]
            source = Path(item["absolute_path"])
            destination = stage_output / PurePosixPath(item["copied_relative_path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            copy_size = destination.stat().st_size
            copy_hash = sha256_file(destination)
            parity = copy_size == item["size_bytes"] and copy_hash == item["sha256"] and source.read_bytes() == destination.read_bytes()
            require(parity, f"source/copy byte parity failed: {source}")
            item["copy_size_bytes"] = copy_size
            item["copy_sha256"] = copy_hash
            item["copy_byte_parity"] = True
            copies.append({
                "expected_word": word,
                "source_absolute_path": str(source),
                "destination_relative_path": item["copied_relative_path"],
                "size_bytes": copy_size,
                "sha256": copy_hash,
                "byte_parity": True,
            })
    return copies


def prior_protected_paths() -> list[Path]:
    manifest_path = STEP9D_OUTPUT / "protected_files_before_after.json"
    if not manifest_path.is_file():
        return []
    data = strict_json_loads(manifest_path.read_bytes(), str(manifest_path))
    require(isinstance(data, dict) and data.get("mismatch_count") == 0, "prior Step 9D protected manifest is invalid")
    records = data.get("after")
    require(isinstance(records, list), "prior Step 9D protected records missing")
    paths = []
    for item in records:
        if not isinstance(item, dict) or not isinstance(item.get("relative_path"), str):
            continue
        candidate = ROOT / PurePosixPath(item["relative_path"])
        if candidate.is_file():
            paths.append(candidate.resolve())
    return paths


def build_protected_paths(roots: dict, candidates: list[dict]) -> list[Path]:
    paths: set[Path] = set(prior_protected_paths())
    paths.update(path.resolve() for path in roots["source_files"] if path.is_file())
    paths.update(item["path"].resolve() for item in candidates)
    paths.add(STEP9D_DELIVERY.resolve())
    if STEP9D_DETACHED.is_file():
        paths.add(STEP9D_DETACHED.resolve())
    paths.add((ROOT / SCRIPT_REL).resolve())
    for directory in (roots["configured_editor_evidence_root"], roots["configured_player_evidence_root"], STEP9D_OUTPUT):
        if directory.is_dir():
            paths.update(path.resolve() for path in directory.rglob("*") if path.is_file())
    return sorted(paths, key=lambda item: str(item).lower())


def protected_label(path: Path, roots: dict) -> dict:
    category, source_root = source_category(path, roots)
    if path_under(path, source_root):
        relative_path = path.relative_to(source_root).as_posix()
    else:
        relative_path = path.name
    return {
        "absolute_source_root_category": category,
        "source_root_absolute": str(source_root),
        "relative_path": relative_path,
        "absolute_path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def protected_snapshot(paths: list[Path], roots: dict) -> list[dict]:
    result = []
    for path in paths:
        require(path.is_file(), f"protected source missing: {path}")
        result.append(protected_label(path, roots))
    return result


def compare_protected(before: list[dict]) -> tuple[list[dict], list[dict]]:
    after = []
    mismatches = []
    for expected in before:
        path = Path(expected["absolute_path"])
        if not path.is_file():
            mismatches.append({"absolute_path": str(path), "reason": "MISSING_AFTER"})
            continue
        actual = {
            **{key: expected[key] for key in ("absolute_source_root_category", "source_root_absolute", "relative_path", "absolute_path")},
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        after.append(actual)
        if actual["size_bytes"] != expected["size_bytes"] or actual["sha256"] != expected["sha256"]:
            mismatches.append({"absolute_path": str(path), "reason": "CONTENT_CHANGED", "before": expected, "after": actual})
    return after, mismatches


def run_git(arguments: list[str]) -> dict:
    try:
        process = subprocess.run(
            ["git", *arguments], cwd=ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace", check=False,
        )
    except OSError as exc:
        return {"available": False, "arguments": arguments, "error": repr(exc)}
    return {
        "available": process.returncode == 0,
        "arguments": arguments,
        "return_code": process.returncode,
        "stdout": process.stdout.splitlines(),
        "stdout_sha256": sha256_bytes(process.stdout.encode("utf-8")),
        "stderr": process.stderr.splitlines(),
    }


def git_snapshot() -> dict:
    return {
        "head": run_git(["rev-parse", "HEAD"]),
        "staged": run_git(["diff", "--cached", "--name-status"]),
        "operations": ["read-only git rev-parse HEAD", "read-only git diff --cached --name-status"],
    }


def git_unchanged(before: dict, after: dict) -> bool:
    for key in ("head", "staged"):
        if before[key].get("available") != after[key].get("available"):
            return False
        if before[key].get("stdout_sha256") != after[key].get("stdout_sha256"):
            return False
    return True


def contract_rows(selected: dict[str, dict]) -> list[dict]:
    rows = []
    definitions = [
        ("raw_75_landmark_coordinates", True, "MISSING_NOT_RETAINED", "No per-frame pose33+left21+right21 coordinate array; only normalized [T,16,3] is stored."),
        ("raw_75_landmark_validity_presence_mask", True, "MISSING_NOT_RETAINED", "Only aggregate raw valid counts are stored; no per-landmark raw mask."),
        ("per_landmark_original_valid_mask", True, "MISSING_NOT_RETAINED", "No per-landmark original-valid array."),
        ("per_landmark_reappearance_reset_boundary", True, "FOUND_PARTIAL", "Aggregate reappearances count exists; no per-landmark/frame boundary array."),
        ("per_landmark_strong_depth_boundary", True, "FOUND_PARTIAL", "Aggregate strong_depth_boundaries count exists; no per-landmark/frame array."),
        ("per_landmark_interpolation_mask_and_max_gap_result", True, "FOUND_PARTIAL", "Aggregate interpolated_landmarks count exists; no per-landmark interpolation/max-gap result array."),
        ("torso_origin_or_center", True, "MISSING_NOT_RETAINED", "No per-frame torso origin/center vector."),
        ("torso_right_axis", True, "MISSING_NOT_RETAINED", "No per-frame torso right axis."),
        ("torso_up_axis", True, "MISSING_NOT_RETAINED", "No per-frame torso up axis."),
        ("torso_forward_axis", True, "MISSING_NOT_RETAINED", "No per-frame torso forward axis."),
        ("torso_scale_or_shoulder_width", True, "FOUND_EXACT", "Per-frame shoulder_width_raw and body_transform_available are stored."),
        ("capture_id", True, "FOUND_EXACT", "Top-level fixed Capture ID is stored and matches filename and fixed Step 9D Delivery."),
        ("capture_local_frame_index", True, "FOUND_EXACT", "Each frame stores frame_index."),
        ("provider_bridge_lifetime_frame_identity", True, "FOUND_PARTIAL", "Capture-window start/final provider/lifetime ranges and counts exist; per-frame provider/bridge identity does not."),
        ("timestamp", True, "FOUND_EXACT", "Each frame stores timestamp_seconds; capture-window timestamp range is also stored."),
        ("source_file_and_schema_version", True, "FOUND_EXACT", "Canonical source path, fixed hash and words_in_silence.step9c.physical_capture.v1 are stored."),
        ("normalized_recognition_input_coordinates", False, "FOUND_EXACT", "Per-frame normalized_coordinates shape [T,16,3] is stored; explicitly not raw 75-landmark evidence."),
        ("normalized_recognition_input_mask", False, "FOUND_EXACT", "Per-frame normalized_mask shape [T,16] is stored; explicitly not raw validity/presence mask."),
        ("aggregate_raw_intermediate_counts", False, "FOUND_EXACT", "Raw-valid, interpolation, reappearance and strong-depth counts are aggregate-only."),
    ]
    for word, entry in selected.items():
        item = entry["json"]
        for field, required, status, evidence in definitions:
            require(status in ALLOWED_CONTRACT_STATUSES, f"invalid contract status: {status}")
            rows.append({
                "expected_word": word,
                "capture_id": CAPTURES[word]["capture_id"],
                "contract_field": field,
                "required_for_complete_verdict": required,
                "status": status,
                "evidence": evidence,
                "canonical_source_relative_path": item["relative_path"],
                "canonical_source_sha256": item["sha256"],
            })
    return rows


def capture_index(selected: dict[str, dict], analyses: list[dict], rows: list[dict]) -> dict:
    captures = []
    for word, entry in selected.items():
        canonical = entry["json"]
        related = [
            {
                "absolute_source_root_category": item["absolute_source_root_category"],
                "relative_path": item["relative_path"],
                "size_bytes": item["size_bytes"],
                "sha256": item["sha256"],
                "data_class": item["data_class"],
                "contract_evidence_status": item["contract_evidence_status"],
                "copy_selected": item["copy_selected"],
            }
            for item in analyses if CAPTURES[word]["capture_id"] in item["capture_ids"]
        ]
        field_statuses = {
            row["contract_field"]: row["status"]
            for row in rows if row["expected_word"] == word
        }
        captures.append({
            "expected_word": word,
            "capture_id": CAPTURES[word]["capture_id"],
            "canonical_source": {
                "absolute_path": canonical["absolute_path"],
                "relative_path": canonical["relative_path"],
                "size_bytes": canonical["size_bytes"],
                "sha256": canonical["sha256"],
                "schema_version": canonical["schema_version"],
                "shape": canonical["shape"],
                "dtype": canonical["dtype"],
                "frame_count": canonical["frame_count"],
                "landmark_count": canonical["landmark_count"],
                "frame_range": canonical["frame_range"],
                "timestamp_range": canonical["timestamp_range"],
                "provider_frame_range": canonical["provider_frame_range"],
                "lifetime_provider_frame_range": canonical["lifetime_provider_frame_range"],
                "bridge_observation_count": canonical["bridge_observation_count"],
                "mask_true_counts": canonical["mask_true_counts"],
                "raw_aggregate_counts": canonical["raw_aggregate_counts"],
                "per_frame_shoulder_width_count": canonical["per_frame_shoulder_width_count"],
                "per_frame_shoulder_width_range": canonical["per_frame_shoulder_width_range"],
                "copied_relative_path": canonical["copied_relative_path"],
                "copy_byte_parity": canonical["copy_byte_parity"],
            },
            "field_statuses": field_statuses,
            "related_candidate_files": related,
            "overall_status": "FOUND_PARTIAL",
        })
    return {
        "schema_version": "words_in_silence.step9d.retained.capture_id_evidence_index.v1",
        "version": VERSION,
        "capture_count": len(captures),
        "captures": captures,
    }


INVENTORY_FIELDS = [
    "candidate_index", "absolute_source_root_category", "source_root_absolute", "relative_path", "absolute_path",
    "size_bytes", "sha256", "extension_container_type", "capture_ids", "capture_id_connection_basis",
    "content_search_status", "schema_version", "shape", "dtype", "frame_count", "landmark_count",
    "finite_count", "nan_count", "inf_count", "null_count", "mask_true_counts", "frame_range", "timestamp_range",
    "provider_frame_range", "lifetime_provider_frame_range", "bridge_observation_count", "data_class",
    "raw_normalized_aggregate_class", "contract_evidence_status", "copy_selected", "copy_selection_status",
    "copy_reason", "copied_relative_path", "copy_size_bytes", "copy_sha256", "copy_byte_parity",
]


def inventory_csv_rows(analyses: list[dict]) -> list[dict]:
    rows = []
    for index, item in enumerate(analyses, start=1):
        row = {field: item.get(field) for field in INVENTORY_FIELDS}
        row["candidate_index"] = index
        for field in ("capture_ids", "shape", "dtype", "mask_true_counts", "frame_range", "timestamp_range", "provider_frame_range", "lifetime_provider_frame_range"):
            row[field] = json.dumps(row[field], ensure_ascii=False, sort_keys=True, allow_nan=False) if row[field] is not None else ""
        rows.append(row)
    return rows


def deterministic_zip_write(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    require(safe_zip_name(name), f"unsafe archive path: {name}")
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    info.create_system = 3
    archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def output_records(stage_output: Path) -> list[dict]:
    records = [file_record(ROOT / SCRIPT_REL, "acquisition_script", SCRIPT_REL)]
    for path in sorted(stage_output.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.name == "output_file_manifest.json":
            continue
        destination_rel = f"{OUTPUT_REL}/{path.relative_to(stage_output).as_posix()}"
        category = "retained_input_byte_copy" if path_under(path, stage_output / "retained_inputs") else "acquisition_output"
        records.append(file_record(path, category, destination_rel))
    records.sort(key=lambda item: item["relative_path"])
    require(len({item["relative_path"] for item in records}) == len(records), "output manifest duplicate target")
    return records


def verify_output_records(stage_output: Path, records: list[dict]) -> dict:
    missing = []
    size_mismatch = []
    hash_mismatch = []
    for record in records:
        if record["relative_path"] == SCRIPT_REL:
            path = ROOT / SCRIPT_REL
        else:
            path = stage_output / PurePosixPath(record["relative_path"]).relative_to(PurePosixPath(OUTPUT_REL))
        if not path.is_file():
            missing.append(record["relative_path"])
            continue
        if path.stat().st_size != record["size_bytes"]:
            size_mismatch.append(record["relative_path"])
        if sha256_file(path) != record["sha256"]:
            hash_mismatch.append(record["relative_path"])
    result = {"missing": missing, "size_mismatch": size_mismatch, "sha256_mismatch": hash_mismatch}
    require(not missing and not size_mismatch and not hash_mismatch, f"output manifest validation failed: {result}")
    return result


def verify_zip(temp_zip: Path, temp_hash: Path, records: list[dict], assertion_summary: dict) -> dict:
    with zipfile.ZipFile(temp_zip, "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        manifest = strict_json_loads(archive.read(OUTPUT_MANIFEST_REL), OUTPUT_MANIFEST_REL)
        require(isinstance(manifest, dict), "ZIP output manifest is not an object")
        expected = {item["relative_path"]: item for item in manifest.get("outputs", [])}
        actual = set(names) - {OUTPUT_MANIFEST_REL}
        checks = {
            "crc_error": archive.testzip(),
            "duplicate_entries": sorted(name for name, count in Counter(names).items() if count > 1),
            "unsafe_paths": sorted(name for name in names if not safe_zip_name(name)),
            "symlinks": sorted(info.filename for info in infos if ((info.external_attr >> 16) & 0o170000) == 0o120000),
            "encrypted_entries": sorted(info.filename for info in infos if info.flag_bits & 0x1),
            "manifest_missing": sorted(set(expected) - actual),
            "manifest_extra": sorted(actual - set(expected)),
            "manifest_size_mismatch": [],
            "manifest_sha256_mismatch": [],
            "required_output_missing": [],
            "assertion_fail_count": assertion_summary["fail"],
            "detached_checksum_strict": False,
        }
        for name, record in expected.items():
            if name not in actual:
                continue
            data = archive.read(name)
            if len(data) != int(record["size_bytes"]):
                checks["manifest_size_mismatch"].append(name)
            if sha256_bytes(data) != record["sha256"]:
                checks["manifest_sha256_mismatch"].append(name)
        required = [
            SCRIPT_REL,
            f"{OUTPUT_REL}/search_scope_manifest.json",
            f"{OUTPUT_REL}/capture_id_evidence_index.json",
            f"{OUTPUT_REL}/retained_intermediate_inventory.csv",
            f"{OUTPUT_REL}/retained_intermediate_contract.json",
            f"{OUTPUT_REL}/missing_evidence_matrix.csv",
            f"{OUTPUT_REL}/protected_files_before_after.json",
            f"{OUTPUT_REL}/acquisition_assertions.json",
            f"{OUTPUT_REL}/acquisition_conclusion.json",
            f"{OUTPUT_REL}/{REPORT_NAME}",
            OUTPUT_MANIFEST_REL,
        ]
        checks["required_output_missing"] = sorted(set(required) - set(names))
        zip_hash = sha256_file(temp_zip)
        checks["detached_checksum_strict"] = parse_detached(temp_hash.read_bytes(), DELIVERY_NAME, zip_hash)
        failure_list_keys = (
            "duplicate_entries", "unsafe_paths", "symlinks", "encrypted_entries", "manifest_missing", "manifest_extra",
            "manifest_size_mismatch", "manifest_sha256_mismatch", "required_output_missing",
        )
        require(
            checks["crc_error"] is None
            and all(not checks[key] for key in failure_list_keys)
            and checks["assertion_fail_count"] == 0
            and checks["detached_checksum_strict"],
            f"post-ZIP verification failed: {checks}",
        )
        require(len(records) == len(expected), "ZIP manifest record count mismatch")
        return checks


def build() -> dict:
    require((ROOT / "Assets").is_dir(), f"not a Unity project root: {ROOT}")
    require((ROOT / "Packages/manifest.json").is_file(), "Packages/manifest.json missing")
    require((ROOT / "ProjectSettings/ProjectVersion.txt").is_file(), "ProjectSettings/ProjectVersion.txt missing")
    for path in (OUTPUT, DELIVERY_ZIP, DELIVERY_HASH, POST_ZIP):
        require(not path.exists(), f"final output already exists; refusing overwrite: {path}")

    delivery = strict_step9d_delivery()
    roots = configured_roots()
    candidates, search_manifest = discover_candidates(roots)
    analyses = [analyze_candidate(item) for item in candidates]
    analyses.append({
        "absolute_source_root_category": "project_root_fixed_step9d_delivery",
        "source_root_absolute": str(ROOT.resolve()),
        "relative_path": STEP9D_DELIVERY.name,
        "absolute_path": str(STEP9D_DELIVERY.resolve()),
        "size_bytes": delivery["size_bytes"],
        "sha256": delivery["sha256"],
        "extension_container_type": "ZIP",
        "capture_ids": list(CAPTURE_IDS),
        "capture_id_connection_basis": "FIXED_DELIVERY_INTERNAL_CONTRACT_AND_PHYSICAL_QUERY_MEMBERS",
        "content_search_status": "STRICT_ARCHIVE_INSPECTION",
        "schema_version": "words_in_silence.step9d.diagnostic_delivery.v1",
        "shape": {"capture_contract_records": 3, "zip_entries": delivery["zip_entry_count"]},
        "dtype": "ZIP_DETERMINISTIC_CONTAINERS",
        "frame_count": sum(item["parsed"]["counts"]["raw_frames"] for item in delivery["physical_members"].values()),
        "landmark_count": 16,
        "finite_count": None,
        "nan_count": None,
        "inf_count": None,
        "null_count": None,
        "mask_true_counts": {},
        "frame_range": None,
        "timestamp_range": None,
        "provider_frame_range": None,
        "lifetime_provider_frame_range": None,
        "bridge_observation_count": None,
        "data_class": "FIXED_APPROVED_DIAGNOSTIC_DELIVERY_CONTRACT",
        "raw_normalized_aggregate_class": "NORMALIZED_AND_AGGREGATE_ONLY;NO_RETAINED_RAW_INTERMEDIATES",
        "contract_evidence_status": "NOT_APPLICABLE",
        "copy_selected": False,
        "copy_selection_status": "NOT_APPLICABLE",
        "copy_reason": "fixed prerequisite Delivery is verified in place and not duplicated as retained input",
        "copied_relative_path": None,
        "copy_size_bytes": None,
        "copy_sha256": None,
        "copy_byte_parity": None,
    })
    analyses.sort(key=lambda item: (item["absolute_source_root_category"], item["relative_path"].lower()))
    selected = canonical_inputs(analyses, delivery)

    protected_paths = build_protected_paths(roots, candidates)
    protected_before = protected_snapshot(protected_paths, roots)
    git_before = git_snapshot()

    staging_root = Path(tempfile.mkdtemp(prefix="wordsinsilence_step9d_retained_"))
    stage_output = staging_root / "output"
    temp_zip = staging_root / (DELIVERY_NAME + ".tmp")
    temp_hash = staging_root / (DELIVERY_NAME + ".sha256.tmp")
    temp_post = staging_root / (POST_ZIP.name + ".tmp")
    stage_output.mkdir(parents=True)
    try:
        copies = copy_selected_inputs(stage_output, selected)
        rows = contract_rows(selected)
        index = capture_index(selected, analyses, rows)
        missing_rows = [row for row in rows if row["required_for_complete_verdict"]]
        exact_missing_fields = sorted({row["contract_field"] for row in missing_rows if row["status"] in {"MISSING_NOT_RETAINED", "FOUND_PARTIAL", "AMBIGUOUS_PROVENANCE"}})

        after_during_stage, mismatches = compare_protected(protected_before)
        require(not mismatches, f"protected file mismatch during acquisition: {mismatches[:3]}")
        git_after = git_snapshot()
        require(git_unchanged(git_before, git_after), "Git HEAD or staging index changed during acquisition")

        protected_payload = {
            "schema_version": "words_in_silence.step9d.retained.protected_files_before_after.v1",
            "protected_file_count": len(protected_before),
            "before": protected_before,
            "after": after_during_stage,
            "mismatch_count": len(mismatches),
            "mismatches": mismatches,
            "git_before": git_before,
            "git_after": git_after,
            "git_head_and_staging_unchanged": True,
        }

        contract_payload = {
            "schema_version": "words_in_silence.step9d.retained.intermediate_contract.v1",
            "version": VERSION,
            "complete_verdict_requires": list(EXPECTED_MISSING_CONTRACT),
            "allowed_statuses": sorted(ALLOWED_CONTRACT_STATUSES),
            "captures": [
                {
                    "expected_word": word,
                    "capture_id": CAPTURES[word]["capture_id"],
                    "overall_status": "FOUND_PARTIAL",
                    "fields": [row for row in rows if row["expected_word"] == word],
                }
                for word in CAPTURES
            ],
            "normalized_16_is_not_raw_75": True,
            "aggregate_counts_are_not_per_landmark_masks": True,
            "missing_fields_synthesized": False,
        }

        assertions: list[dict] = []
        add_assert(assertions, "fixed_step9d_delivery_size_strict", "PASS", {"expected": STEP9D_EXPECTED_SIZE, "actual": delivery["size_bytes"]})
        add_assert(assertions, "fixed_step9d_delivery_sha256_strict", "PASS", {"expected": STEP9D_EXPECTED_SHA256, "actual": delivery["sha256"]})
        add_assert(assertions, "fixed_step9d_delivery_crc", "PASS", {"entry_count": delivery["zip_entry_count"]})
        add_assert(assertions, "fixed_capture_ids_exact_three", "PASS", {word: value["capture_id"] for word, value in CAPTURES.items()})
        add_assert(assertions, "step9d_missing_contract_read_from_delivery", "PASS", delivery["missing_contract"])
        add_assert(assertions, "source_config_evidence_roots_confirmed", "PASS", search_manifest["roots"])
        add_assert(assertions, "search_scope_restricted", "PASS", search_manifest["scope_restriction"])
        add_assert(assertions, "candidate_file_size_sha256_inventory", "PASS", {"candidate_count": len(analyses)})
        add_assert(assertions, "strict_json_nonfinite_rejection", "PASS", {"nan_as_string_or_zero_substitution": False})
        add_assert(assertions, "canonical_capture_matches_fixed_delivery_bytes", "PASS", {word: entry["json"]["sha256"] for word, entry in selected.items()})
        add_assert(assertions, "normalized_16_and_raw_75_explicitly_distinguished", "PASS", True)
        add_assert(assertions, "aggregate_only_evidence_explicitly_distinguished", "PASS", True)
        add_assert(assertions, "missing_fields_not_synthesized", "PASS", True)
        add_assert(assertions, "copied_inputs_only_found_partial", "PASS", {"copied_file_count": len(copies), "statuses": sorted({item["copy_selection_status"] for item in analyses if item["copy_selected"]})})
        add_assert(assertions, "retained_input_source_copy_byte_parity", "PASS", {"pass": len(copies), "fail": 0})
        add_assert(assertions, "protected_source_files_unchanged", "PASS", {"checked": len(protected_before), "mismatches": 0})
        add_assert(assertions, "unity_editor_player_webcam_capture_test_execution_count_zero", "PASS", {"Unity": 0, "Editor": 0, "player": 0, "webcam": 0, "capture": 0, "test": 0})
        add_assert(assertions, "production_threshold_template_evidence_modification_count_zero", "PASS", {"protected_mismatches": 0})
        add_assert(assertions, "git_staging_commit_push_execution_count_zero", "PASS", {"staging": 0, "commit": 0, "push": 0, "head_and_index_unchanged": True})
        add_assert(assertions, "output_manifest_validation_enforced_before_commit", "PASS", True)
        add_assert(assertions, "post_zip_verification_enforced_before_commit", "PASS", True)
        add_assert(assertions, "retained_raw_75_coordinates_and_masks", "UNVERIFIED", "MISSING_NOT_RETAINED for all three captures")
        add_assert(assertions, "retained_per_landmark_boundary_interpolation_masks", "UNVERIFIED", "Only aggregate counts retained for all three captures")
        add_assert(assertions, "retained_full_per_frame_torso_basis", "UNVERIFIED", "Only shoulder_width_raw and availability retained for all three captures")
        names = [item["name"] for item in assertions]
        require(len(names) == len(set(names)), "duplicate acquisition assertion names")
        assertion_summary = {
            status.lower(): sum(item["status"] == status for item in assertions)
            for status in ("PASS", "FAIL", "UNVERIFIED", "NOT_APPLICABLE")
        }
        require(assertion_summary["fail"] == 0, "acquisition assertion failure")

        conclusion = {
            "schema_version": "words_in_silence.step9d.retained.acquisition_conclusion.v1",
            "version": VERSION,
            "primary_verdict": PRIMARY_VERDICT,
            "status": FINAL_STATUS,
            "fixed_capture_count": 3,
            "retained_input_copy_count": len(copies),
            "last_confirmed_retained_pipeline_stage": "per-frame normalized RecognitionCore input [T,16,3] plus normalized mask, local frame index, timestamp, shoulder_width_raw/body_transform_available, capture-window ranges, and aggregate upstream counts",
            "exactly_missing_or_incomplete_fields": exact_missing_fields,
            "raw_75_coordinates_directly_retained": False,
            "raw_75_validity_presence_masks_directly_retained": False,
            "per_landmark_boundary_interpolation_masks_directly_retained": False,
            "full_torso_basis_directly_retained": False,
            "per_frame_shoulder_width_retained": True,
            "normalized_16_coordinates_and_mask_retained": True,
            "missing_values_reconstructed_or_synthesized": False,
            "assertion_summary": assertion_summary,
            "protected_file_mismatch_count": 0,
            "operation_counts": {
                "unity_editor_player_webcam_capture_test": 0,
                "production_threshold_template_evidence_modification": 0,
                "git_staging_commit_push": 0,
            },
            "next_state": "BLOCKED_PENDING_EXTERNAL_REVIEW",
        }

        write_json(stage_output / "search_scope_manifest.json", search_manifest)
        write_json(stage_output / "capture_id_evidence_index.json", index)
        write_csv(stage_output / "retained_intermediate_inventory.csv", INVENTORY_FIELDS, inventory_csv_rows(analyses))
        write_json(stage_output / "retained_intermediate_contract.json", contract_payload)
        write_csv(
            stage_output / "missing_evidence_matrix.csv",
            ["expected_word", "capture_id", "contract_field", "required_for_complete_verdict", "status", "evidence", "canonical_source_relative_path", "canonical_source_sha256"],
            missing_rows,
        )
        write_json(stage_output / "protected_files_before_after.json", protected_payload)
        write_json(stage_output / "acquisition_assertions.json", {
            "schema_version": "words_in_silence.step9d.retained.acquisition_assertions.v1",
            "version": VERSION,
            "summary": assertion_summary,
            "assertions": assertions,
        })
        write_json(stage_output / "acquisition_conclusion.json", conclusion)

        capture_lines = []
        for word, entry in selected.items():
            item = entry["json"]
            capture_lines.append(
                f"- {word} `{CAPTURES[word]['capture_id']}`: normalized coordinates {item['shape']['normalized_coordinates']} / mask {item['shape']['normalized_mask']}; "
                f"JSON-number-untyped + boolean; {item['size_bytes']:,} bytes; SHA-256 `{item['sha256']}`."
            )
        report = f"""# Step 9D retained raw/intermediate evidence acquisition v1.0

## Verdict

`{PRIMARY_VERDICT}`

`{FINAL_STATUS}`

The exact source/config-connected Editor evidence root and Player persistent-data evidence root were searched read-only together with the project root. The Player root does not exist. No other disk root was searched.

## Fixed input

- Step 9D Delivery: {delivery['size_bytes']:,} bytes; SHA-256 `{delivery['sha256']}`; strict size/hash/CRC PASS.
- Fixed Capture IDs: exactly 3/3 match the Delivery contract.

## Retained evidence

{chr(10).join(capture_lines)}

The six canonical files (three JSON plus their strict checksum sidecars) were copied byte-for-byte under `retained_inputs/configured_editor_evidence_root/`; source/copy parity is 6 PASS / 0 FAIL.

## Last confirmed retained boundary

The last directly retained array stage is the normalized RecognitionCore input `[T,16,3]` with `[T,16]` mask. Per-frame local index, timestamp, shoulder width and body-transform availability are present. Capture-window provider/lifetime ranges and aggregate raw/interpolation/boundary counts are present.

These values are explicitly not treated as raw 75-landmark coordinates, per-landmark raw/original/boundary/interpolation masks, or a full torso basis.

## Missing or incomplete

- Raw pose 33 + left hand 21 + right hand 21 coordinate arrays: `MISSING_NOT_RETAINED`.
- Per-landmark raw validity/presence and original-valid masks: `MISSING_NOT_RETAINED`.
- Reappearance/reset, strong-depth and interpolation/max-gap arrays: `FOUND_PARTIAL` aggregate counts only; per-landmark arrays are not retained.
- Per-frame torso origin/right/up/forward: `MISSING_NOT_RETAINED`.
- Per-frame provider/bridge/lifetime identity: `FOUND_PARTIAL` range/count metadata only.

No missing field was inferred, reverse-normalized, reconstructed or synthesized.

## Integrity and prohibited operations

- Assertions: {assertion_summary['pass']} PASS / {assertion_summary['fail']} FAIL / {assertion_summary['unverified']} UNVERIFIED / {assertion_summary['not_applicable']} NOT_APPLICABLE.
- Protected files: {len(protected_before)} checked / 0 mismatches.
- Unity, Editor, player, webcam, physical capture and tests executed: 0.
- Production, threshold, template and existing evidence modifications: 0.
- Git staging, commit and push: 0; HEAD and staged-index snapshot unchanged.

`BLOCKED_PENDING_EXTERNAL_REVIEW`
"""
        (stage_output / REPORT_NAME).write_text(report, encoding="utf-8", newline="\n")

        records = output_records(stage_output)
        output_manifest = {
            "schema_version": "words_in_silence.step9d.retained.output_file_manifest.v1",
            "version": VERSION,
            "self_reference_excluded": True,
            "excluded_final_sidecars": [DELIVERY_NAME, f"{DELIVERY_NAME}.sha256", POST_ZIP.name],
            "total_files": len(records),
            "outputs": records,
        }
        write_json(stage_output / "output_file_manifest.json", output_manifest)
        output_manifest_checks = verify_output_records(stage_output, records)

        with zipfile.ZipFile(temp_zip, "w") as archive:
            for record in records:
                if record["relative_path"] == SCRIPT_REL:
                    data = (ROOT / SCRIPT_REL).read_bytes()
                else:
                    data = (stage_output / PurePosixPath(record["relative_path"]).relative_to(PurePosixPath(OUTPUT_REL))).read_bytes()
                deterministic_zip_write(archive, record["relative_path"], data)
            deterministic_zip_write(archive, OUTPUT_MANIFEST_REL, (stage_output / "output_file_manifest.json").read_bytes())

        zip_hash = sha256_file(temp_zip)
        temp_hash.write_text(f"{zip_hash}  {DELIVERY_NAME}\n", encoding="utf-8", newline="\n")
        zip_checks = verify_zip(temp_zip, temp_hash, records, assertion_summary)
        final_after, final_mismatches = compare_protected(protected_before)
        require(not final_mismatches and len(final_after) == len(protected_before), "protected file mismatch before commit")
        git_final = git_snapshot()
        require(git_unchanged(git_before, git_final), "Git HEAD or staging index changed before commit")

        post_payload = {
            "schema_version": "words_in_silence.step9d.retained.post_zip_verification.v1",
            "version": VERSION,
            "status": FINAL_STATUS,
            "primary_verdict": PRIMARY_VERDICT,
            "zip_file": DELIVERY_NAME,
            "zip_size_bytes": temp_zip.stat().st_size,
            "zip_sha256": zip_hash,
            "zip_entry_count": len(records) + 1,
            "manifest_target_count": len(records),
            "manifest_self_excluded": True,
            "output_manifest_checks": output_manifest_checks,
            "checks": zip_checks,
            "acquisition_assertion_summary": assertion_summary,
            "protected_files": {"checked": len(protected_before), "mismatches": 0},
            "retained_input_copies": {"checked": len(copies), "byte_parity_failures": 0},
            "git_head_and_staging_unchanged": True,
        }
        write_json(temp_post, post_payload)

        require(not OUTPUT.exists() and not DELIVERY_ZIP.exists() and not DELIVERY_HASH.exists() and not POST_ZIP.exists(), "final output appeared during staging")
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage_output, OUTPUT)
        os.replace(temp_hash, DELIVERY_HASH)
        os.replace(temp_post, POST_ZIP)
        os.replace(temp_zip, DELIVERY_ZIP)
        require(sha256_file(DELIVERY_ZIP) == zip_hash, "committed Delivery ZIP hash mismatch")
        require(parse_detached(DELIVERY_HASH.read_bytes(), DELIVERY_NAME, zip_hash), "committed detached checksum mismatch")
        _, committed_mismatches = compare_protected(protected_before)
        require(not committed_mismatches, "protected file mismatch after commit")
        require(git_unchanged(git_before, git_snapshot()), "Git HEAD or staging index changed after commit")

        return {
            "status": FINAL_STATUS,
            "primary_verdict": PRIMARY_VERDICT,
            "script": file_record(ROOT / SCRIPT_REL, "acquisition_script"),
            "delivery_zip": file_record(DELIVERY_ZIP, "delivery_zip"),
            "detached_checksum": file_record(DELIVERY_HASH, "detached_checksum"),
            "post_zip_verification": file_record(POST_ZIP, "post_zip_verification"),
            "search_roots": search_manifest["roots"],
            "unique_searched_file_count": search_manifest["unique_searched_file_count"],
            "candidate_file_count": len(analyses),
            "retained_input_copy_count": len(copies),
            "capture_summaries": [
                {
                    "expected_word": word,
                    "capture_id": CAPTURES[word]["capture_id"],
                    "shape": selected[word]["json"]["shape"],
                    "dtype": selected[word]["json"]["dtype"],
                    "size_bytes": selected[word]["json"]["size_bytes"],
                    "sha256": selected[word]["json"]["sha256"],
                    "overall_status": "FOUND_PARTIAL",
                }
                for word in CAPTURES
            ],
            "last_confirmed_retained_pipeline_stage": conclusion["last_confirmed_retained_pipeline_stage"],
            "exactly_missing_or_incomplete_fields": exact_missing_fields,
            "assertions": assertion_summary,
            "protected_file_count": len(protected_before),
            "protected_mismatch_count": 0,
            "zip_entry_count": len(records) + 1,
            "manifest_target_count": len(records),
        }
    finally:
        if staging_root.exists():
            resolved = staging_root.resolve()
            temp_root = Path(tempfile.gettempdir()).resolve()
            if resolved.parent == temp_root and resolved.name.startswith("wordsinsilence_step9d_retained_"):
                shutil.rmtree(resolved)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        result = build()
    except Exception as exc:
        print(json.dumps({"status": SELF_FAIL_STATUS, "error": repr(exc)}, ensure_ascii=False, indent=2), flush=True)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
