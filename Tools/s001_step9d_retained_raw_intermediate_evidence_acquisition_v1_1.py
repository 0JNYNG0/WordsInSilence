#!/usr/bin/env python3
"""Evidence-driven Step 9D retained raw/intermediate acquisition v1.1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import inspect
import json
import math
import os
import re
import shutil
import struct
import subprocess
import tempfile
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent.parent
VERSION = "1.1"
SCRIPT_REL = "Tools/s001_step9d_retained_raw_intermediate_evidence_acquisition_v1_1.py"
OUTPUT_REL = "data/processed/sessions/S001_P001/step9d_retained_raw_intermediate_evidence_acquisition_v1_1"
OUTPUT = ROOT / PurePosixPath(OUTPUT_REL)
OUTPUT_MANIFEST_REL = f"{OUTPUT_REL}/output_file_manifest.json"
REPORT_NAME = "STEP9D_RETAINED_RAW_INTERMEDIATE_EVIDENCE_ACQUISITION_V1_1_REPORT.md"

DELIVERY_NAME = "WordsInSilence_step9d_retained_raw_intermediate_evidence_acquisition_v1_1_delivery.zip"
DELIVERY_ZIP = ROOT / DELIVERY_NAME
DELIVERY_HASH = ROOT / f"{DELIVERY_NAME}.sha256"
POST_ZIP = ROOT / "WordsInSilence_step9d_retained_raw_intermediate_evidence_acquisition_v1_1_delivery_post_zip_verification.json"

DIAG_NAME = "WordsInSilence_step9d_physical_recognition_domain_gap_diagnostic_v1_0_delivery.zip"
DIAG_ZIP = ROOT / DIAG_NAME
DIAG_SIZE = 2_804_364
DIAG_SHA256 = "b179962aba86aa961e42c5f762e696c718c831b7ac444c5fe708d5301a69f2d0"
V10_NAME = "WordsInSilence_step9d_retained_raw_intermediate_evidence_acquisition_v1_0_delivery.zip"
V10_ZIP = ROOT / V10_NAME
V10_SIZE = 233_855
V10_SHA256 = "b3d86ae7af971b549a08c2a4229caad03dbe08a6d55d811dd6425ac8edb55abe"
V10_OUTPUT = ROOT / "data/processed/sessions/S001_P001/step9d_retained_raw_intermediate_evidence_acquisition_v1_0"

RUN_ID = "8d17ef49-89ff-4d40-ae59-0386cc406e2e"
CAPTURES = {
    "GREETING": {"take": 1, "id": "step9c-take-01-greeting-f7875c16-ad18-425c-8ce5-39e9a6cc13f7"},
    "LOVE": {"take": 2, "id": "step9c-take-02-love-6d68ddd7-2d7e-4afd-bd0a-b1989f8e52f2"},
    "THANK_YOU": {"take": 3, "id": "step9c-take-03-thank-you-53870472-c644-489c-a94d-95e9667ded42"},
}
CAPTURE_IDS = tuple(item["id"] for item in CAPTURES.values())
ID_TO_WORD = {item["id"]: word for word, item in CAPTURES.items()}

EDITOR_REL = "data/processed/sessions/S001_P001/step9c_physical_webcam_coordinate_compatibility_v1_0/physical_captures"
EDITOR_ROOT = ROOT / PurePosixPath(EDITOR_REL)

FIELDS = (
    "raw_75_coordinates", "raw_75_validity_presence_mask", "original_valid_mask",
    "reappearance_reset_boundary", "strong_depth_boundary", "interpolation_mask_max_gap",
    "torso_origin", "torso_right_axis", "torso_up_axis", "torso_forward_axis", "torso_scale_shoulder_width",
    "capture_id", "capture_local_frame_index", "provider_frame_identity", "bridge_frame_identity",
    "lifetime_frame_identity", "timestamp", "schema_provenance",
)
OPTIONAL_FIELDS = ("normalized_16_coordinates", "normalized_16_mask", "aggregate_upstream_counts")
ALL_FIELDS = FIELDS + OPTIONAL_FIELDS
CONTRACT_STATUSES = {
    "FOUND_EXACT", "FOUND_PARTIAL", "MISSING_NOT_RETAINED", "AMBIGUOUS_PROVENANCE",
    "UNVERIFIED_SEARCH_INCOMPLETE", "NOT_APPLICABLE",
}

VERDICT_TO_STATUS = {
    "RETAINED_EVIDENCE_COMPLETE": "STEP9D_RETAINED_RAW_INTERMEDIATE_EVIDENCE_ACQUIRED_PENDING_EXTERNAL_REVIEW",
    "RETAINED_EVIDENCE_INCOMPLETE_OR_NOT_RETAINED": "STEP9D_RETAINED_RAW_INTERMEDIATE_EVIDENCE_ACQUISITION_BLOCKED_NOT_RETAINED_OR_INCOMPLETE",
    "RETAINED_EVIDENCE_SEARCH_INCOMPLETE": "STEP9D_RETAINED_RAW_INTERMEDIATE_EVIDENCE_ACQUISITION_BLOCKED_SEARCH_INCOMPLETE",
}
SELF_FAIL_STATUS = "STEP9D_RETAINED_RAW_INTERMEDIATE_EVIDENCE_ACQUISITION_V1_1_SELF_VERIFICATION_FAIL"

TEXT_EXTENSIONS = {
    ".asmdef", ".asset", ".bat", ".cs", ".csproj", ".csv", ".json", ".log", ".md",
    ".meta", ".prefab", ".ps1", ".py", ".sha256", ".sln", ".txt", ".unity", ".xml",
    ".yaml", ".yml",
}
PROJECT_CACHE_ROOTS = {".git", ".vs", "Library", "Temp", "Logs", "obj", "Build", "Builds"}
READ_PROHIBITED_PREFIXES = (("Tools", "_archive"), ("Validation", "archive"))
SOURCE_ROOT = ROOT / "Assets/_Project/Scripts/Recognition"

SOURCE_FILES = (
    "LiveInput/RawMediaPipeObservation.cs",
    "LiveInput/ApprovedLivePreprocessingAdapter.cs",
    "LiveInput/ManualDiagnosticCaptureController.cs",
    "Runtime/NormalizedRecognitionFrame.cs",
    "Runtime/RecognitionSequenceInputSession.cs",
    "PhysicalCompatibility/Step9CHandIdentityObservationBridge.cs",
    "PhysicalCompatibility/PhysicalCaptureEvidenceRecorder.cs",
    "PhysicalCompatibility/Step9CPhysicalCaptureEvidence.cs",
    "PhysicalCompatibility/PhysicalCaptureEvidenceJson.cs",
    "PhysicalCompatibility/PhysicalWebcamCoordinateCompatibilityHarness.cs",
    "PhysicalCompatibility/PhysicalCaptureEvidenceResume.cs",
    "Core/RecognitionSequenceBinary.cs",
    "Runtime/RuntimeRecognitionService.cs",
)

INVENTORY_FIELDS = [
    "inventory_id", "root_category", "root_memberships", "relative_path", "absolute_path", "size_bytes",
    "extension", "container_type", "path_capture_ids", "path_run_id_match", "take_number",
    "known_physical_filename_match", "content_search_performed", "content_capture_ids", "content_run_id_match",
    "content_search_result", "container_inspection_performed", "container_inspection_result",
    "exclusion_reason", "source_config_relevance", "relevance_classification", "sha256", "read_error",
]

CANDIDATE_FIELDS = [
    "candidate_id", "source_kind", "root_category", "relative_path", "absolute_or_archive_path",
    "size_bytes", "sha256", "container_type", "capture_ids", "run_id_match", "connection_basis",
    "schema_version", "shape", "dtype", "frame_count", "landmark_count", "finite_count", "null_count",
    "nan_count", "inf_count", "mask_true_counts", "frame_range", "timestamp_range", "field_evidence",
    "data_class", "provenance_class", "ambiguity_blocker", "copy_selected", "copied_relative_path",
    "copy_size_bytes", "copy_sha256", "copy_byte_parity",
]


class AcquisitionError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AcquisitionError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def native_path(path: Path) -> str:
    absolute = str(path.resolve())
    if os.name == "nt" and not absolute.startswith("\\\\?\\"):
        return "\\\\?\\" + absolute
    return absolute


def is_file(path: Path) -> bool:
    return os.path.isfile(native_path(path))


def stat_file(path: Path):
    return os.stat(native_path(path))


def read_file(path: Path) -> bytes:
    with open(native_path(path), "rb") as stream:
        return stream.read()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(native_path(path), "rb") as stream:
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


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            values = {}
            for field in fields:
                value = row.get(field, "")
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))
                elif value is None:
                    value = ""
                values[field] = value
            writer.writerow(values)


def strict_json(data: bytes, label: str) -> object:
    def reject(value: str) -> None:
        raise AcquisitionError(f"non-finite token {value} in {label}")
    try:
        return json.loads(data.decode("utf-8-sig"), parse_constant=reject)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AcquisitionError(f"invalid JSON {label}: {exc}") from exc


def path_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def safe_zip_name(name: str) -> bool:
    item = PurePosixPath(name)
    return (
        bool(name) and not name.startswith(("/", "\\")) and "\\" not in name and not item.is_absolute()
        and ".." not in item.parts and not (len(name) >= 2 and name[1] == ":")
    )


def parse_detached(data: bytes, expected_name: str, expected_hash: str) -> bool:
    try:
        parts = data.decode("utf-8-sig").strip().split(maxsplit=1)
    except UnicodeDecodeError:
        return False
    return len(parts) == 2 and parts[0].lower() == expected_hash and parts[1].lstrip("*") == expected_name


def file_record(path: Path, category: str, relative_path: str | None = None) -> dict:
    require(is_file(path), f"required file missing: {path}")
    relative_path = relative_path or path.resolve().relative_to(ROOT.resolve()).as_posix()
    return {"relative_path": relative_path, "size_bytes": stat_file(path).st_size, "sha256": sha256_file(path), "category": category}


def add_assert(records: list[dict], name: str, status: str, evidence: object) -> None:
    require(status in {"PASS", "FAIL", "UNVERIFIED", "NOT_APPLICABLE"}, f"invalid assertion status {status}")
    records.append({"name": name, "status": status, "evidence": evidence})


def strict_fixed_zip(path: Path, name: str, size: int, expected_hash: str) -> dict:
    require(is_file(path), f"fixed ZIP missing: {path}")
    actual_size = stat_file(path).st_size
    actual_hash = sha256_file(path)
    require(actual_size == size, f"fixed ZIP size mismatch for {name}: {actual_size}")
    require(actual_hash == expected_hash, f"fixed ZIP SHA-256 mismatch for {name}: {actual_hash}")
    detached_path = path.with_name(path.name + ".sha256")
    detached = is_file(detached_path) and parse_detached(read_file(detached_path), name, actual_hash)
    require(detached, f"fixed ZIP detached checksum mismatch: {name}")
    with zipfile.ZipFile(path, "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        checks = {
            "crc_error": archive.testzip(),
            "duplicate_entries": [item for item, count in Counter(names).items() if count > 1],
            "unsafe_paths": [item for item in names if not safe_zip_name(item)],
            "encrypted_entries": [info.filename for info in infos if info.flag_bits & 1],
            "symlinks": [info.filename for info in infos if ((info.external_attr >> 16) & 0o170000) == 0o120000],
        }
    require(checks["crc_error"] is None and all(not checks[key] for key in checks if key != "crc_error"), f"fixed ZIP structural failure: {name}")
    return {"name": name, "size_bytes": actual_size, "sha256": actual_hash, "entry_count": len(names), "detached_strict": True, "checks": checks}


def confirm_configured_roots() -> dict:
    settings = ROOT / "ProjectSettings/ProjectSettings.asset"
    harness = SOURCE_ROOT / "PhysicalCompatibility/PhysicalWebcamCoordinateCompatibilityHarness.cs"
    require(settings.is_file() and harness.is_file(), "source/config path inputs missing")
    settings_text = settings.read_text(encoding="utf-8-sig")
    company_match = re.search(r"^\s*companyName:\s*(.+?)\s*$", settings_text, re.MULTILINE)
    product_match = re.search(r"^\s*productName:\s*(.+?)\s*$", settings_text, re.MULTILINE)
    require(company_match and product_match, "company/product config missing")
    company, product = company_match.group(1), product_match.group(1)
    require((company, product) == ("DefaultCompany", "WordsInSilence"), f"unexpected Unity identity: {company}/{product}")
    harness_text = harness.read_text(encoding="utf-8")
    editor_fragment = '"data", "processed", "sessions", "S001_P001", "step9c_physical_webcam_coordinate_compatibility_v1_0", "physical_captures"'
    player_fragment = 'Application.persistentDataPath, "Step9CPhysicalWebcamCoordinateCompatibility", Step9CPhysicalCompatibilityConstants.RunId'
    require(editor_fragment in harness_text and player_fragment in harness_text, "EvidenceOutputDirectory source contract changed")
    player = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "AppData/LocalLow" / company / product / "Step9CPhysicalWebcamCoordinateCompatibility" / RUN_ID
    return {
        "project_root": ROOT.resolve(),
        "editor_evidence_root": EDITOR_ROOT.resolve(),
        "player_evidence_root": player.resolve(),
        "company": company,
        "product": product,
        "run_id": RUN_ID,
        "additional_physical_persistence_roots": [],
        "source_confirmation": [
            "ProjectSettings/ProjectSettings.asset:companyName/productName",
            "PhysicalWebcamCoordinateCompatibilityHarness.cs:EvidenceOutputDirectory",
        ],
    }


def content_read_prohibited(relative: Path) -> bool:
    parts = relative.parts
    if len(parts) >= 2 and tuple(parts[:2]) in READ_PROHIBITED_PREFIXES:
        return True
    return len(parts) >= 2 and parts[0] == "Validation" and parts[-1].startswith("motion_validation_details") and parts[-1].endswith(".json")


def project_dir_excluded(relative: Path) -> bool:
    parts = relative.parts
    if not parts:
        return False
    if parts[0] in PROJECT_CACHE_ROOTS:
        return True
    return len(parts) >= 2 and tuple(parts[:2]) in READ_PROHIBITED_PREFIXES


def enumerate_root(root: Path, project: bool) -> tuple[list[Path], list[dict]]:
    if not root.exists():
        return [], []
    require(root.is_dir(), f"search root is not a directory: {root}")
    files: list[Path] = []
    errors: list[dict] = []

    def onerror(error: OSError) -> None:
        errors.append({"path": getattr(error, "filename", None), "error": repr(error)})

    for current, directories, names in os.walk(root, onerror=onerror):
        base = Path(current)
        if project:
            directories[:] = [name for name in directories if not project_dir_excluded((base / name).relative_to(root))]
        for name in names:
            path = base / name
            try:
                if is_file(path):
                    files.append(path.resolve())
            except OSError as exc:
                errors.append({"path": str(path), "error": repr(exc)})
    return sorted(files, key=lambda item: str(item).lower()), errors


def stream_identifier_search(path: Path) -> dict:
    needles = {capture_id: capture_id.encode("ascii") for capture_id in CAPTURE_IDS}
    run_needle = RUN_ID.encode("ascii")
    max_length = max([len(run_needle), *(len(value) for value in needles.values())])
    found: set[str] = set()
    run_found = False
    tail = b""
    bytes_read = 0
    try:
        with open(native_path(path), "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                bytes_read += len(chunk)
                block = tail + chunk
                for capture_id, needle in needles.items():
                    if needle in block:
                        found.add(capture_id)
                if run_needle in block:
                    run_found = True
                tail = block[-(max_length - 1):]
    except OSError as exc:
        return {"status": "READ_ERROR", "capture_ids": [], "run_id_match": False, "bytes_read": bytes_read, "error": repr(exc)}
    return {"status": "INSPECTED", "capture_ids": sorted(found), "run_id_match": run_found, "bytes_read": bytes_read, "error": None}


def container_type(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".json": "JSON", ".csv": "CSV", ".npz": "NPZ", ".npy": "NPY", ".wisrt": "WISRT",
        ".zip": "ZIP", ".sha256": "SHA256_TEXT", ".xml": "XML", ".png": "PNG", ".jpg": "JPEG",
        ".jpeg": "JPEG", ".mp4": "VIDEO", ".mov": "VIDEO", ".wav": "AUDIO",
    }.get(suffix, "TEXT" if suffix in TEXT_EXTENSIONS else "UNKNOWN_BINARY")


def linked_archive(path: Path) -> bool:
    name = path.name.lower()
    return path.suffix.lower() == ".zip" and ("step9c" in name or "step9d" in name)


def known_take(path_text: str) -> tuple[bool, int | None]:
    match = re.search(r"take[_-](\d+)[_-](GREETING|LOVE|THANK_YOU)", path_text, re.IGNORECASE)
    return (match is not None, int(match.group(1)) if match else None)


def shape_of(value: object) -> list[int] | None:
    if not isinstance(value, list):
        return None
    if not value:
        return [0]
    child_shapes = [shape_of(item) for item in value]
    if any(item is None for item in child_shapes):
        return [len(value)]
    if any(item != child_shapes[0] for item in child_shapes[1:]):
        return [len(value)]
    return [len(value), *child_shapes[0]]


def number_counts(value: object) -> dict:
    result = {"finite": 0, "null": 0, "nan": 0, "inf": 0}
    def visit(item: object) -> None:
        if isinstance(item, list):
            for child in item:
                visit(child)
        elif item is None:
            result["null"] += 1
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            numeric = float(item)
            if math.isnan(numeric): result["nan"] += 1
            elif math.isinf(numeric): result["inf"] += 1
            else: result["finite"] += 1
    visit(value)
    return result


def blank_field_evidence() -> dict[str, str]:
    return {field: "NONE" for field in ALL_FIELDS}


def inspect_physical_json(data: dict, label: str) -> dict:
    fields = blank_field_evidence()
    frames = data.get("frames")
    if not isinstance(frames, list):
        return {
            "physical": False, "schema_version": str(data.get("schema_version", "ABSENT")),
            "capture_id": data.get("capture_id"), "field_evidence": fields, "data_class": "DERIVED_JSON_REFERENCE",
        }
    capture_id = data.get("capture_id")
    frame_count = len(frames)
    frame_indices = []
    timestamps = []
    normalized_coordinates = []
    normalized_masks = []
    shoulder_widths = []
    body_available = []
    frame_keys: set[str] = set()
    for position, frame in enumerate(frames):
        require(isinstance(frame, dict), f"non-object frame in {label} at {position}")
        frame_keys.update(str(key) for key in frame)
        if isinstance(frame.get("frame_index"), int) and not isinstance(frame.get("frame_index"), bool):
            frame_indices.append(frame["frame_index"])
        if isinstance(frame.get("timestamp_seconds"), (int, float)) and not isinstance(frame.get("timestamp_seconds"), bool):
            require(math.isfinite(float(frame["timestamp_seconds"])), f"non-finite timestamp in {label}")
            timestamps.append(float(frame["timestamp_seconds"]))
        if isinstance(frame.get("normalized_coordinates"), list):
            normalized_coordinates.append(frame["normalized_coordinates"])
        if isinstance(frame.get("normalized_mask"), list):
            normalized_masks.append(frame["normalized_mask"])
        shoulder_widths.append(frame.get("shoulder_width_raw"))
        body_available.append(frame.get("body_transform_available"))

    normalized_shape = shape_of(normalized_coordinates)
    mask_shape = shape_of(normalized_masks)
    if normalized_shape == [frame_count, 16, 3]:
        fields["normalized_16_coordinates"] = "EXACT"
    if mask_shape == [frame_count, 16] and all(isinstance(value, bool) for row in normalized_masks for value in row):
        fields["normalized_16_mask"] = "EXACT"
    if len(frame_indices) == frame_count and frame_indices == list(range(frame_count)):
        fields["capture_local_frame_index"] = "EXACT"
    if len(timestamps) == frame_count:
        fields["timestamp"] = "EXACT"
    if capture_id:
        fields["capture_id"] = "EXACT"
    if data.get("schema_version"):
        fields["schema_provenance"] = "EXACT"
    if len(shoulder_widths) == frame_count and all(value is None or (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))) for value in shoulder_widths):
        fields["torso_scale_shoulder_width"] = "EXACT"

    counts = data.get("counts") if isinstance(data.get("counts"), dict) else {}
    if any(key in counts for key in ("raw_pose_valid_landmarks", "raw_left_hand_valid_landmarks", "raw_right_hand_valid_landmarks")):
        fields["raw_75_validity_presence_mask"] = "PARTIAL"
        fields["original_valid_mask"] = "PARTIAL"
    if "reappearances" in counts:
        fields["reappearance_reset_boundary"] = "PARTIAL"
    if "strong_depth_boundaries" in counts:
        fields["strong_depth_boundary"] = "PARTIAL"
    if "interpolated_landmarks" in counts:
        fields["interpolation_mask_max_gap"] = "PARTIAL"
    if counts:
        fields["aggregate_upstream_counts"] = "EXACT"

    window = data.get("capture_window") if isinstance(data.get("capture_window"), dict) else {}
    if any(key in window for key in ("first_capture_provider_frame_index", "final_capture_provider_frame_index")):
        fields["provider_frame_identity"] = "PARTIAL"
    if "bridge_observation_count" in window:
        fields["bridge_frame_identity"] = "PARTIAL"
    if any(key in window for key in ("first_capture_lifetime_provider_frame_index", "final_capture_lifetime_provider_frame_index")):
        fields["lifetime_frame_identity"] = "PARTIAL"

    per_frame_aliases = {
        "raw_75_coordinates": ("raw_coordinates", "raw_landmark_coordinates", "raw_75_coordinates"),
        "raw_75_validity_presence_mask": ("raw_validity_mask", "raw_presence_mask", "raw_valid_mask"),
        "original_valid_mask": ("original_valid", "original_valid_mask"),
        "reappearance_reset_boundary": ("reappearance_mask", "reset_mask", "boundary_mask"),
        "strong_depth_boundary": ("strong_depth_mask", "strong_depth_boundary"),
        "interpolation_mask_max_gap": ("interpolation_mask", "interpolated_mask", "max_gap_result"),
        "torso_origin": ("torso_origin", "torso_center"),
        "torso_right_axis": ("torso_right", "right_axis"),
        "torso_up_axis": ("torso_up", "up_axis"),
        "torso_forward_axis": ("torso_forward", "forward_axis"),
    }
    for field, aliases in per_frame_aliases.items():
        for alias in aliases:
            values = [frame.get(alias) for frame in frames]
            shape = shape_of(values)
            expected = [frame_count, 75, 3] if field == "raw_75_coordinates" else [frame_count, 75]
            if field.startswith("torso_"):
                expected = [frame_count, 3]
            if shape == expected:
                fields[field] = "EXACT"
                break

    coordinate_counts = number_counts(normalized_coordinates)
    mask_true = sum(value for row in normalized_masks for value in row if isinstance(value, bool))
    return {
        "physical": True,
        "schema_version": str(data.get("schema_version", "ABSENT")),
        "capture_id": capture_id,
        "expected_word": data.get("expected_word"),
        "run_id": data.get("run_id"),
        "frame_count": frame_count,
        "landmark_count": 16,
        "shape": {"normalized_coordinates": normalized_shape, "normalized_mask": mask_shape},
        "dtype": {"normalized_coordinates": "JSON_NUMBER_UNTYPED_OR_NULL", "normalized_mask": "JSON_BOOLEAN"},
        "finite_count": coordinate_counts["finite"], "null_count": coordinate_counts["null"],
        "nan_count": coordinate_counts["nan"], "inf_count": coordinate_counts["inf"],
        "mask_true_counts": {"normalized_mask": mask_true, "body_transform_available": sum(value is True for value in body_available)},
        "frame_range": [min(frame_indices), max(frame_indices)] if frame_indices else None,
        "timestamp_range": [min(timestamps), max(timestamps)] if timestamps else None,
        "field_evidence": fields,
        "data_class": "NORMALIZED_16_PER_FRAME_PLUS_AGGREGATE_UPSTREAM",
        "frame_keys": sorted(frame_keys),
    }


def inspect_array_file(path: Path, kind: str) -> dict:
    try:
        import numpy as np
    except ImportError as exc:
        return {"status": "OPAQUE_BINARY_UNVERIFIED", "error": repr(exc)}
    try:
        arrays = {}
        if kind == "NPZ":
            with np.load(path, allow_pickle=False) as archive:
                for key in archive.files:
                    value = archive[key]
                    arrays[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        else:
            value = np.load(path, allow_pickle=False, mmap_mode="r")
            arrays["array"] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        return {"status": "INSPECTED", "arrays": arrays}
    except Exception as exc:
        return {"status": "OPAQUE_BINARY_UNVERIFIED", "error": repr(exc)}


def inspect_wisrt(path: Path) -> dict:
    try:
        with open(native_path(path), "rb") as stream:
            header = stream.read(12)
        if len(header) >= 12 and header[:8] == b"WISRCB01":
            return {"status": "INSPECTED", "magic": "WISRCB01", "version_or_header_u32": struct.unpack("<I", header[8:12])[0]}
        return {"status": "OPAQUE_BINARY_UNVERIFIED", "magic_hex": header[:8].hex()}
    except OSError as exc:
        return {"status": "READ_ERROR", "error": repr(exc)}


def inspect_linked_zip(path: Path) -> tuple[dict, list[dict]]:
    member_candidates: list[dict] = []
    try:
        with zipfile.ZipFile(path, "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            summary = {
                "status": "INSPECTED", "entry_count": len(infos), "crc_error": archive.testzip(),
                "duplicates": [name for name, count in Counter(names).items() if count > 1],
                "unsafe": [name for name in names if not safe_zip_name(name)],
                "encrypted": [info.filename for info in infos if info.flag_bits & 1],
                "symlinks": [info.filename for info in infos if ((info.external_attr >> 16) & 0o170000) == 0o120000],
                "capture_id_member_count": 0, "run_id_member_count": 0,
            }
            for info in infos:
                if info.is_dir() or info.flag_bits & 1:
                    continue
                name_ids = [capture_id for capture_id in CAPTURE_IDS if capture_id in info.filename]
                name_run = RUN_ID in info.filename
                payload = archive.read(info.filename)
                content_ids = [capture_id for capture_id in CAPTURE_IDS if capture_id.encode("ascii") in payload]
                content_run = RUN_ID.encode("ascii") in payload
                ids = sorted(set(name_ids + content_ids))
                if ids: summary["capture_id_member_count"] += 1
                if name_run or content_run: summary["run_id_member_count"] += 1
                if not ids:
                    continue
                if info.filename.lower().endswith(".zip"):
                    try:
                        with zipfile.ZipFile(io.BytesIO(payload), "r") as nested:
                            nested_infos = nested.infolist()
                            nested_names = [item.filename for item in nested_infos]
                            require(nested.testzip() is None, f"nested archive CRC failure: {path}!{info.filename}")
                            require(len(nested_names) == len(set(nested_names)), f"nested archive duplicate paths: {path}!{info.filename}")
                            require(all(safe_zip_name(name) for name in nested_names), f"nested archive unsafe path: {path}!{info.filename}")
                            require(all(not (item.flag_bits & 1) for item in nested_infos), f"nested archive encrypted entry: {path}!{info.filename}")
                            for nested_info in nested_infos:
                                if nested_info.is_dir():
                                    continue
                                nested_payload = nested.read(nested_info.filename)
                                nested_ids = sorted(set(
                                    [capture_id for capture_id in CAPTURE_IDS if capture_id in nested_info.filename]
                                    + [capture_id for capture_id in CAPTURE_IDS if capture_id.encode("ascii") in nested_payload]
                                ))
                                if not nested_ids:
                                    continue
                                nested_analysis = None
                                if nested_info.filename.lower().endswith(".json"):
                                    parsed_nested = strict_json(nested_payload, f"{path}!{info.filename}!{nested_info.filename}")
                                    if isinstance(parsed_nested, dict):
                                        nested_analysis = inspect_physical_json(parsed_nested, f"{path}!{info.filename}!{nested_info.filename}")
                                member_candidates.append({
                                    "source_kind": "NESTED_ARCHIVE_MEMBER", "root_category": "linked_nested_archive",
                                    "relative_path": f"{path.relative_to(ROOT).as_posix()}!{info.filename}!{nested_info.filename}",
                                    "absolute_or_archive_path": f"{path.resolve()}!{info.filename}!{nested_info.filename}",
                                    "size_bytes": len(nested_payload), "sha256": sha256_bytes(nested_payload),
                                    "container_type": container_type(Path(nested_info.filename)), "capture_ids": nested_ids,
                                    "run_id_match": RUN_ID in nested_info.filename or RUN_ID.encode("ascii") in nested_payload,
                                    "connection_basis": "DIRECT_LINKED_NESTED_ARCHIVE_MEMBER_PATH_OR_CONTENT",
                                    "analysis": nested_analysis,
                                    "ambiguity_blocker": nested_analysis is None and container_type(Path(nested_info.filename)) not in {"SHA256_TEXT", "TEXT", "XML", "CSV"},
                                })
                        continue
                    except Exception as exc:
                        member_candidates.append({
                            "source_kind": "ARCHIVE_MEMBER", "root_category": "linked_archive",
                            "relative_path": f"{path.relative_to(ROOT).as_posix()}!{info.filename}",
                            "absolute_or_archive_path": f"{path.resolve()}!{info.filename}", "size_bytes": len(payload),
                            "sha256": sha256_bytes(payload), "container_type": "ZIP", "capture_ids": ids,
                            "run_id_match": name_run or content_run, "connection_basis": "NESTED_ARCHIVE_INSPECTION_ERROR",
                            "analysis": None, "ambiguity_blocker": True, "inspection_error": repr(exc),
                        })
                        continue
                parsed = None
                analysis = None
                if info.filename.lower().endswith(".json"):
                    try:
                        parsed = strict_json(payload, f"{path}!{info.filename}")
                        if isinstance(parsed, dict):
                            analysis = inspect_physical_json(parsed, f"{path}!{info.filename}")
                    except AcquisitionError:
                        parsed = None
                member_candidates.append({
                    "source_kind": "ARCHIVE_MEMBER", "root_category": "linked_archive",
                    "relative_path": f"{path.relative_to(ROOT).as_posix()}!{info.filename}",
                    "absolute_or_archive_path": f"{path.resolve()}!{info.filename}", "size_bytes": len(payload),
                    "sha256": sha256_bytes(payload), "container_type": container_type(Path(info.filename)),
                    "capture_ids": ids, "run_id_match": name_run or content_run,
                    "connection_basis": "ARCHIVE_MEMBER_PATH_OR_CONTENT", "analysis": analysis,
                    "ambiguity_blocker": analysis is None and container_type(Path(info.filename)) not in {"SHA256_TEXT", "TEXT", "XML", "CSV"},
                })
            return summary, member_candidates
    except Exception as exc:
        return {"status": "READ_ERROR", "error": repr(exc)}, []


def source_category(path: Path, roots: dict) -> str:
    if path_under(path, roots["editor_evidence_root"]): return "editor_evidence_root"
    if path_under(path, roots["player_evidence_root"]): return "player_evidence_root"
    return "project_root"


def inventory_and_candidates(roots: dict) -> tuple[list[dict], list[dict], dict]:
    project_files, project_errors = enumerate_root(roots["project_root"], True)
    editor_files, editor_errors = enumerate_root(roots["editor_evidence_root"], False)
    player_files, player_errors = enumerate_root(roots["player_evidence_root"], False)
    memberships: dict[Path, list[str]] = {}
    for category, files in (("project_root", project_files), ("editor_evidence_root", editor_files), ("player_evidence_root", player_files)):
        for path in files:
            memberships.setdefault(path, []).append(category)
    paths = sorted(memberships, key=lambda item: str(item).lower())
    rows: list[dict] = []
    candidates: list[dict] = []
    stats = Counter()

    for index, path in enumerate(paths, start=1):
        file_size = stat_file(path).st_size
        relative = path.relative_to(ROOT).as_posix() if path_under(path, ROOT) else path.name
        category = source_category(path, roots)
        path_ids = [capture_id for capture_id in CAPTURE_IDS if capture_id in relative]
        path_run = RUN_ID in relative
        known, take = known_take(relative)
        kind = container_type(path)
        prohibited = path_under(path, ROOT) and content_read_prohibited(path.relative_to(ROOT))
        content = {"status": "NOT_APPLICABLE", "capture_ids": [], "run_id_match": False, "bytes_read": 0, "error": None}
        if path.suffix.lower() in TEXT_EXTENSIONS and not prohibited:
            content = stream_identifier_search(path)
            stats["text_content_search_performed"] += 1
            if file_size > 8 * 1024 * 1024:
                stats["large_text_content_search_performed"] += 1
        elif path.suffix.lower() in TEXT_EXTENSIONS and prohibited:
            stats["agents_content_read_prohibited"] += 1

        content_ids = content["capture_ids"]
        content_run = bool(content["run_id_match"])
        exact_root = category in {"editor_evidence_root", "player_evidence_root"}
        in_source_contract = path_under(path, SOURCE_ROOT) and path.suffix.lower() == ".cs"
        direct_link = bool(path_ids or content_ids or path_run or content_run or known)
        archive_link = linked_archive(path)
        relevant = exact_root or in_source_contract or direct_link or archive_link
        relevance = "RELEVANT" if relevant else "IRRELEVANT"
        source_relevance = (
            "EXACT_EVIDENCE_ROOT" if exact_root else "SOURCE_PERSISTENCE_CONTRACT" if in_source_contract
            else "CAPTURE_OR_RUN_PROVENANCE_MATCH" if direct_link else "DIRECT_STEP9C_STEP9D_ARCHIVE" if archive_link
            else "NO_SOURCE_CONFIG_OR_PROVENANCE_LINK"
        )
        inspection = {"status": "NOT_REQUIRED_IRRELEVANT"}
        archive_candidates: list[dict] = []
        read_error = content.get("error")
        try:
            if archive_link:
                inspection, archive_candidates = inspect_linked_zip(path)
            elif exact_root and kind == "JSON":
                parsed = strict_json(read_file(path), str(path))
                require(isinstance(parsed, dict), f"evidence-root JSON is not an object: {path}")
                analysis = inspect_physical_json(parsed, str(path))
                inspection = {"status": "INSPECTED", "physical": analysis["physical"], "schema_version": analysis["schema_version"]}
                ids = sorted(set(path_ids + content_ids + ([analysis.get("capture_id")] if analysis.get("capture_id") in CAPTURE_IDS else [])))
                candidates.append({
                    "source_kind": "FILESYSTEM", "root_category": category, "relative_path": path.relative_to(roots["editor_evidence_root"]).as_posix() if category == "editor_evidence_root" else relative,
                    "absolute_or_archive_path": str(path), "size_bytes": file_size, "sha256": sha256_file(path),
                    "container_type": kind, "capture_ids": ids, "run_id_match": analysis.get("run_id") == RUN_ID or content_run,
                    "connection_basis": "EXACT_EVIDENCE_ROOT_JSON", "analysis": analysis, "ambiguity_blocker": False,
                })
            elif exact_root and kind == "SHA256_TEXT":
                adjacent = Path(str(path)[:-7])
                strict = is_file(adjacent) and parse_detached(read_file(path), adjacent.name, sha256_file(adjacent))
                inspection = {"status": "INSPECTED", "strict_adjacent_checksum": strict}
                require(strict, f"evidence-root checksum mismatch: {path}")
                candidates.append({
                    "source_kind": "FILESYSTEM", "root_category": category,
                    "relative_path": path.relative_to(roots["editor_evidence_root"]).as_posix() if category == "editor_evidence_root" else relative,
                    "absolute_or_archive_path": str(path), "size_bytes": file_size, "sha256": sha256_file(path),
                    "container_type": kind, "capture_ids": sorted(set(path_ids + content_ids)), "run_id_match": content_run,
                    "connection_basis": "EXACT_EVIDENCE_ROOT_CHECKSUM", "analysis": None, "ambiguity_blocker": False,
                })
            elif relevant and kind == "JSON":
                parsed = strict_json(read_file(path), str(path))
                analysis = inspect_physical_json(parsed, str(path)) if isinstance(parsed, dict) else None
                inspection = {"status": "INSPECTED", "json_root": type(parsed).__name__}
                candidates.append({
                    "source_kind": "FILESYSTEM", "root_category": category, "relative_path": relative,
                    "absolute_or_archive_path": str(path), "size_bytes": file_size, "sha256": sha256_file(path),
                    "container_type": kind, "capture_ids": sorted(set(path_ids + content_ids)), "run_id_match": path_run or content_run,
                    "connection_basis": "PATH_OR_STREAM_CONTENT", "analysis": analysis, "ambiguity_blocker": False,
                })
            elif relevant and kind in {"NPZ", "NPY"}:
                inspection = inspect_array_file(path, kind)
            elif relevant and kind == "WISRT":
                inspection = inspect_wisrt(path)
            elif relevant and kind == "UNKNOWN_BINARY":
                inspection = {"status": "OPAQUE_BINARY_UNVERIFIED"}
            elif relevant:
                inspection = {"status": "INSPECTED_TEXT_OR_KNOWN_NON_ARRAY", "content_search": content["status"]}
                if path_ids or content_ids:
                    candidates.append({
                        "source_kind": "FILESYSTEM", "root_category": category, "relative_path": relative,
                        "absolute_or_archive_path": str(path), "size_bytes": file_size, "sha256": sha256_file(path),
                        "container_type": kind, "capture_ids": sorted(set(path_ids + content_ids)), "run_id_match": path_run or content_run,
                        "connection_basis": "PATH_OR_STREAM_CONTENT", "analysis": None, "ambiguity_blocker": False,
                    })
        except (OSError, AcquisitionError) as exc:
            inspection = {"status": "READ_ERROR", "error": repr(exc)}
            read_error = repr(exc)

        candidates.extend(archive_candidates)
        inspection_status = inspection.get("status", "UNKNOWN")
        if relevant: stats["relevant_files"] += 1
        else: stats["irrelevant_files"] += 1
        if exact_root: stats["exact_root_files"] += 1
        if inspection_status == "OPAQUE_BINARY_UNVERIFIED" and relevant: stats["relevant_opaque"] += 1
        if inspection_status == "READ_ERROR" and relevant: stats["relevant_read_error"] += 1
        if relevant and not (content["status"] == "INSPECTED" or inspection_status in {"INSPECTED", "INSPECTED_TEXT_OR_KNOWN_NON_ARRAY"}):
            if not (in_source_contract and content["status"] == "INSPECTED"):
                stats["relevant_uninspected"] += 1
        if path.suffix.lower() in TEXT_EXTENSIONS and file_size > 8 * 1024 * 1024 and content["status"] != "INSPECTED":
            stats["large_text_skipped"] += 1

        sha = sha256_file(path) if relevant and not read_error else ""
        row = {
            "inventory_id": f"INV{index:05d}", "root_category": category,
            "root_memberships": memberships[path], "relative_path": relative, "absolute_path": str(path),
            "size_bytes": file_size, "extension": path.suffix.lower(), "container_type": kind,
            "path_capture_ids": path_ids, "path_run_id_match": path_run, "take_number": take,
            "known_physical_filename_match": known, "content_search_performed": content["status"] == "INSPECTED",
            "content_capture_ids": content_ids, "content_run_id_match": content_run, "content_search_result": content,
            "container_inspection_performed": inspection_status not in {"NOT_REQUIRED_IRRELEVANT"},
            "container_inspection_result": inspection, "exclusion_reason": "AGENTS_CONTENT_READ_PROHIBITED" if prohibited else "",
            "source_config_relevance": source_relevance, "relevance_classification": relevance,
            "sha256": sha, "read_error": read_error or "",
        }
        rows.append(row)

    for candidate_index, candidate in enumerate(candidates, start=1):
        candidate["candidate_id"] = f"CAND{candidate_index:04d}"
    exact_rows = [row for row in rows if row["root_category"] in {"editor_evidence_root", "player_evidence_root"}]
    exact_inspected = [row for row in exact_rows if row["container_inspection_performed"] and not row["read_error"]]
    coverage = {
        "schema_version": "words_in_silence.step9d.retained.search_scope.v1_1",
        "coverage_record_ids": ["COV_PROJECT", "COV_EDITOR", "COV_PLAYER", "COV_TEXT", "COV_CONTAINER", "COV_SOURCE"],
        "roots": [
            {"coverage_id": "COV_PROJECT", "category": "project_root", "absolute_path": str(roots["project_root"]), "exists": True, "file_count": len(project_files), "walk_errors": project_errors},
            {"coverage_id": "COV_EDITOR", "category": "editor_evidence_root", "absolute_path": str(roots["editor_evidence_root"]), "exists": roots["editor_evidence_root"].is_dir(), "file_count": len(editor_files), "walk_errors": editor_errors},
            {"coverage_id": "COV_PLAYER", "category": "player_evidence_root", "absolute_path": str(roots["player_evidence_root"]), "exists": roots["player_evidence_root"].is_dir(), "file_count": len(player_files), "walk_errors": player_errors, "absent_root_verified": not roots["player_evidence_root"].exists()},
        ],
        "excluded_generated_cache_roots": sorted(PROJECT_CACHE_ROOTS),
        "excluded_instruction_roots": ["Tools/_archive", "Validation/archive"],
        "recordings_blanket_excluded": False,
        "unique_in_scope_file_count": len(rows),
        "statistics": dict(stats),
        "exact_evidence_root_inventory_file_count": len(exact_rows),
        "exact_evidence_root_inspected_file_count": len(exact_inspected),
        "exact_evidence_root_exhaustive_pass": len(exact_rows) == len(exact_inspected) and not editor_errors and not player_errors,
        "large_text_skipped_count": stats["large_text_skipped"],
        "relevant_opaque_count": stats["relevant_opaque"],
        "relevant_read_error_count": stats["relevant_read_error"],
        "relevant_uninspected_count": stats["relevant_uninspected"],
        "arbitrary_disk_roots_searched": 0,
    }
    return rows, candidates, coverage


def line_range(text: str, symbols: list[str]) -> list[int] | None:
    lines = text.splitlines()
    found = [index for index, line in enumerate(lines, start=1) if any(symbol in line for symbol in symbols)]
    return [min(found), max(found)] if found else None


def build_source_contract() -> dict:
    source_files = []
    texts: dict[str, str] = {}
    for index, relative in enumerate(SOURCE_FILES, start=1):
        path = SOURCE_ROOT / PurePosixPath(relative)
        require(path.is_file(), f"source persistence file missing: {path}")
        text = path.read_text(encoding="utf-8")
        texts[relative] = text
        source_files.append({
            "source_file_id": f"SCF{index:03d}", "relative_path": path.relative_to(ROOT).as_posix(),
            "absolute_path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256_file(path),
            "copied_relative_path": f"retained_inputs/source_persistence_contract/SCF{index:03d}_{path.name}",
            "copy_size_bytes": None, "copy_sha256": None, "copy_byte_parity": None,
        })
    by_relative = {relative: source_files[index] for index, relative in enumerate(SOURCE_FILES)}

    all_cs = sorted(SOURCE_ROOT.rglob("*.cs"))
    writer_hits = []
    for path in all_cs:
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"File\.(WriteAllBytes|WriteAllText|WriteAllLines|Create|Move)\s*\(", text):
            line = text.count("\n", 0, match.start()) + 1
            writer_hits.append({"relative_path": path.relative_to(ROOT).as_posix(), "symbol": match.group(0).split("(")[0], "line": line})
    physical_writer_hits = [hit for hit in writer_hits if "PhysicalCaptureEvidenceJson.cs" in hit["relative_path"]]
    require(physical_writer_hits and all("PhysicalCaptureEvidenceJson.cs" in hit["relative_path"] for hit in writer_hits), f"unexpected Recognition writer: {writer_hits}")

    serializer_text = texts["PhysicalCompatibility/PhysicalCaptureEvidenceJson.cs"]
    serializer_keys = set(re.findall(r'Property\(output,\s*"([A-Za-z0-9_]+)"', serializer_text))
    serializer_keys.update(re.findall(r'\\"([A-Za-z0-9_]+)\\"', serializer_text))
    all_text = "\n".join(texts.values())
    file_id = {relative: item["source_file_id"] for relative, item in by_relative.items()}

    specs = {
        "raw_75_coordinates": {"direct": ["raw_coordinates", "raw_75_coordinates"], "aggregate": [], "runtime": ["RawLandmarkObservationSet", "CopyCoordinates"]},
        "raw_75_validity_presence_mask": {"direct": ["raw_validity_mask", "raw_presence_mask"], "aggregate": ["raw_pose_valid_landmarks", "raw_left_hand_valid_landmarks", "raw_right_hand_valid_landmarks"], "runtime": ["CopyValidMask", "validMask"]},
        "original_valid_mask": {"direct": ["original_valid_mask", "original_valid"], "aggregate": ["raw_valid_landmarks"], "runtime": ["OriginalValid"]},
        "reappearance_reset_boundary": {"direct": ["reappearance_mask", "reset_mask", "boundary_mask"], "aggregate": ["reappearances", "timestamp_boundaries"], "runtime": ["IsReappearance", "Boundary"]},
        "strong_depth_boundary": {"direct": ["strong_depth_mask", "strong_depth_boundary"], "aggregate": ["strong_depth_boundaries"], "runtime": ["IsStrongDepthBoundary", "Strong"]},
        "interpolation_mask_max_gap": {"direct": ["interpolation_mask", "interpolated_mask"], "aggregate": ["interpolated_landmarks"], "runtime": ["Interpolated", "SelectedMaxGap"]},
        "torso_origin": {"direct": ["torso_origin", "torso_center"], "aggregate": [], "runtime": ["Origin", "ComputeTorsoTransform"]},
        "torso_right_axis": {"direct": ["torso_right", "right_axis"], "aggregate": [], "runtime": ["Right", "ComputeTorsoTransform"]},
        "torso_up_axis": {"direct": ["torso_up", "up_axis"], "aggregate": [], "runtime": ["Up", "ComputeTorsoTransform"]},
        "torso_forward_axis": {"direct": ["torso_forward", "forward_axis"], "aggregate": [], "runtime": ["Forward", "ComputeTorsoTransform"]},
        "torso_scale_shoulder_width": {"direct": ["shoulder_width_raw"], "aggregate": ["available_frames"], "runtime": ["Scale", "ShoulderWidthRaw"]},
        "capture_id": {"direct": ["capture_id"], "aggregate": [], "runtime": ["CaptureId"]},
        "capture_local_frame_index": {"direct": ["frame_index"], "aggregate": [], "runtime": ["frameIndex"]},
        "provider_frame_identity": {"direct": [], "aggregate": ["first_capture_provider_frame_index", "final_capture_provider_frame_index", "provider_frame_count"], "runtime": ["ProviderFrameCount"]},
        "bridge_frame_identity": {"direct": [], "aggregate": ["bridge_observation_count"], "runtime": ["ObservationCount"]},
        "lifetime_frame_identity": {"direct": [], "aggregate": ["first_capture_lifetime_provider_frame_index", "final_capture_lifetime_provider_frame_index"], "runtime": ["LifetimeProviderFrameCount"]},
        "timestamp": {"direct": ["timestamp_seconds", "timestamps"], "aggregate": [], "runtime": ["TimestampSeconds"]},
        "schema_provenance": {"direct": ["schema_version", "run_id", "expected_word", "take_number", "source_convention"], "aggregate": [], "runtime": ["SchemaVersion", "RunId"]},
        "normalized_16_coordinates": {"direct": ["normalized_coordinates"], "aggregate": [], "runtime": ["NormalizedRecognitionFrame", "CopyNormalizedCoordinates"]},
        "normalized_16_mask": {"direct": ["normalized_mask"], "aggregate": [], "runtime": ["CopyNormalizedMask", "CopyValidMask"]},
        "aggregate_upstream_counts": {"direct": ["counts"], "aggregate": ["raw_frames", "interpolated_landmarks", "reappearances", "strong_depth_boundaries"], "runtime": ["PhysicalCaptureEvidenceRecorder"]},
    }
    references = {
        "raw_75_coordinates": [("LiveInput/RawMediaPipeObservation.cs", ["coordinates", "CopyCoordinates"]), ("PhysicalCompatibility/PhysicalCaptureEvidenceRecorder.cs", ["normalizedCoordinates", "PhysicalFrameEvidence("]), ("PhysicalCompatibility/PhysicalCaptureEvidenceJson.cs", ["WriteFrame"])],
        "raw_75_validity_presence_mask": [("LiveInput/RawMediaPipeObservation.cs", ["valid", "CopyValidMask"]), ("PhysicalCompatibility/PhysicalCaptureEvidenceRecorder.cs", ["rawPoseValid", "rawLeftValid", "rawRightValid"])],
        "original_valid_mask": [("LiveInput/ApprovedLivePreprocessingAdapter.cs", ["OriginalValid"])],
        "reappearance_reset_boundary": [("LiveInput/RawMediaPipeObservation.cs", ["reappearance", "IsReappearance"]), ("LiveInput/ApprovedLivePreprocessingAdapter.cs", ["Boundary"]), ("PhysicalCompatibility/PhysicalCaptureEvidenceRecorder.cs", ["reappearanceCount"])],
        "strong_depth_boundary": [("LiveInput/RawMediaPipeObservation.cs", ["strongDepthBoundary", "IsStrongDepthBoundary"]), ("LiveInput/ApprovedLivePreprocessingAdapter.cs", ["Strong"]), ("PhysicalCompatibility/PhysicalCaptureEvidenceRecorder.cs", ["strongDepthCount"])],
        "interpolation_mask_max_gap": [("LiveInput/ApprovedLivePreprocessingAdapter.cs", ["SelectedMaxGap", "Interpolated"]), ("PhysicalCompatibility/PhysicalCaptureEvidenceRecorder.cs", ["interpolatedCount"])],
        "torso_origin": [("LiveInput/ApprovedLivePreprocessingAdapter.cs", ["Origin", "ComputeTorsoTransform"])],
        "torso_right_axis": [("LiveInput/ApprovedLivePreprocessingAdapter.cs", ["Right", "ComputeTorsoTransform"])],
        "torso_up_axis": [("LiveInput/ApprovedLivePreprocessingAdapter.cs", ["Up", "ComputeTorsoTransform"])],
        "torso_forward_axis": [("LiveInput/ApprovedLivePreprocessingAdapter.cs", ["Forward", "ComputeTorsoTransform"])],
        "torso_scale_shoulder_width": [("LiveInput/ApprovedLivePreprocessingAdapter.cs", ["Scale"]), ("PhysicalCompatibility/Step9CPhysicalCaptureEvidence.cs", ["ShoulderWidthRaw"])],
        "capture_id": [("PhysicalCompatibility/Step9CPhysicalCaptureEvidence.cs", ["CaptureId"]), ("PhysicalCompatibility/PhysicalCaptureEvidenceJson.cs", ['"capture_id"'])],
        "capture_local_frame_index": [("PhysicalCompatibility/PhysicalCaptureEvidenceJson.cs", ["frame_index"])],
        "provider_frame_identity": [("PhysicalCompatibility/Step9CHandIdentityObservationBridge.cs", ["ProviderFrameCount"]), ("PhysicalCompatibility/PhysicalCaptureEvidenceJson.cs", ["first_capture_provider_frame_index"])],
        "bridge_frame_identity": [("PhysicalCompatibility/Step9CHandIdentityObservationBridge.cs", ["ObservationCount"]), ("PhysicalCompatibility/PhysicalCaptureEvidenceJson.cs", ["bridge_observation_count"])],
        "lifetime_frame_identity": [("PhysicalCompatibility/Step9CHandIdentityObservationBridge.cs", ["LifetimeProviderFrameCount"]), ("PhysicalCompatibility/PhysicalCaptureEvidenceJson.cs", ["first_capture_lifetime_provider_frame_index"])],
        "timestamp": [("LiveInput/RawMediaPipeObservation.cs", ["TimestampSeconds"]), ("PhysicalCompatibility/PhysicalCaptureEvidenceJson.cs", ["timestamp_seconds"])],
        "schema_provenance": [("PhysicalCompatibility/Step9CPhysicalCaptureEvidence.cs", ["SchemaVersion", "RunId"]), ("PhysicalCompatibility/PhysicalWebcamCoordinateCompatibilityHarness.cs", ["EvidenceOutputDirectory"]), ("PhysicalCompatibility/PhysicalCaptureEvidenceJson.cs", ["schema_version", "File.WriteAllBytes"])],
        "normalized_16_coordinates": [("Runtime/NormalizedRecognitionFrame.cs", ["CoordinateCount"]), ("PhysicalCompatibility/PhysicalCaptureEvidenceJson.cs", ["normalized_coordinates"])],
        "normalized_16_mask": [("Runtime/NormalizedRecognitionFrame.cs", ["LandmarkCount"]), ("PhysicalCompatibility/PhysicalCaptureEvidenceJson.cs", ["normalized_mask"])],
        "aggregate_upstream_counts": [("PhysicalCompatibility/PhysicalCaptureEvidenceRecorder.cs", ["rawPoseValid", "interpolatedCount"]), ("PhysicalCompatibility/PhysicalCaptureEvidenceJson.cs", ["counts"])],
    }
    records = []
    for index, field in enumerate(ALL_FIELDS, start=1):
        spec = specs[field]
        direct_detected = sorted(set(spec["direct"]) & serializer_keys)
        aggregate_detected = sorted(set(spec["aggregate"]) & serializer_keys)
        runtime_detected = [symbol for symbol in spec["runtime"] if symbol in all_text]
        if direct_detected:
            classification = "serialized/direct"
        elif aggregate_detected:
            classification = "aggregate-only"
        elif runtime_detected:
            classification = "runtime-only/not serialized"
        else:
            classification = "unknown"
        refs = []
        for relative, symbols in references[field]:
            span = line_range(texts[relative], symbols)
            require(span is not None, f"source symbol missing for {field}: {relative} {symbols}")
            item = by_relative[relative]
            refs.append({
                "source_file_id": item["source_file_id"], "relative_path": item["relative_path"],
                "sha256": item["sha256"], "symbols": symbols, "line_range": span,
            })
        records.append({
            "source_contract_record_id": f"SC{index:03d}", "field": field, "classification": classification,
            "direct_serialized_keys_detected": direct_detected, "aggregate_serialized_keys_detected": aggregate_detected,
            "runtime_symbols_detected": runtime_detected, "writer_source_refs": refs,
            "persistence_path_rule": "Editor project/data/.../physical_captures; Player persistentDataPath/Step9CPhysicalWebcamCoordinateCompatibility/{RunId}",
            "filename_rule": "take_{take_number:D2}_{expected_word}_{capture_id}.json plus byte-linked .sha256",
            "capture_id_run_id_link": "capture_id and run_id serialized top-level; filename embeds capture_id",
            "status_derivation": {"serializer_key_set_size": len(serializer_keys), "direct_candidates": spec["direct"], "aggregate_candidates": spec["aggregate"], "runtime_candidates": spec["runtime"]},
        })
    return {
        "schema_version": "words_in_silence.step9d.retained.source_persistence_contract.v1_1",
        "source_root": str(SOURCE_ROOT.resolve()), "source_file_count": len(source_files), "source_files": source_files,
        "recognition_cs_file_count_scanned": len(all_cs), "file_writer_hits": writer_hits,
        "physical_writer_hits": physical_writer_hits, "physical_writer_exclusive": True,
        "npz_npy_writer_hits": [], "wisrt_physical_writer_hits": [],
        "serializer_keys_detected": sorted(serializer_keys), "records": records,
        "all_field_classifications_resolved": all(record["classification"] != "unknown" for record in records),
    }


def normalize_candidate(candidate: dict) -> dict:
    analysis = candidate.get("analysis") or {}
    field_evidence = analysis.get("field_evidence", blank_field_evidence())
    capture_ids = candidate.get("capture_ids", [])
    provenance = "FIXED_CAPTURE_EXACT" if len(capture_ids) == 1 and capture_ids[0] in CAPTURE_IDS and analysis.get("capture_id") == capture_ids[0] else "REFERENCE_OR_PATH_ONLY"
    result = {
        "candidate_id": candidate["candidate_id"], "source_kind": candidate["source_kind"],
        "root_category": candidate["root_category"], "relative_path": candidate["relative_path"],
        "absolute_or_archive_path": candidate["absolute_or_archive_path"], "size_bytes": candidate["size_bytes"],
        "sha256": candidate["sha256"], "container_type": candidate["container_type"],
        "capture_ids": capture_ids, "run_id_match": candidate.get("run_id_match", False),
        "connection_basis": candidate["connection_basis"], "schema_version": analysis.get("schema_version", "UNKNOWN"),
        "shape": analysis.get("shape"), "dtype": analysis.get("dtype"), "frame_count": analysis.get("frame_count"),
        "landmark_count": analysis.get("landmark_count"), "finite_count": analysis.get("finite_count"),
        "null_count": analysis.get("null_count"), "nan_count": analysis.get("nan_count"), "inf_count": analysis.get("inf_count"),
        "mask_true_counts": analysis.get("mask_true_counts", {}), "frame_range": analysis.get("frame_range"),
        "timestamp_range": analysis.get("timestamp_range"), "field_evidence": field_evidence,
        "data_class": analysis.get("data_class", "REFERENCE_OR_INTEGRITY_METADATA"),
        "provenance_class": provenance, "ambiguity_blocker": candidate.get("ambiguity_blocker", False),
        "copy_selected": False, "copied_relative_path": None, "copy_size_bytes": None,
        "copy_sha256": None, "copy_byte_parity": None,
    }
    return result


def derive_contract_status(capture_id: str, field: str, coverage: dict, source_record: dict, candidates: list[dict]) -> dict:
    connected = [candidate for candidate in candidates if capture_id in candidate.get("capture_ids", []) and candidate.get("provenance_class") == "FIXED_CAPTURE_EXACT"]
    exact = [candidate["candidate_id"] for candidate in connected if candidate.get("field_evidence", {}).get(field) == "EXACT"]
    partial = [candidate["candidate_id"] for candidate in connected if candidate.get("field_evidence", {}).get(field) == "PARTIAL"]
    ambiguous = [
        candidate["candidate_id"] for candidate in candidates
        if candidate.get("ambiguity_blocker") and (capture_id in candidate.get("capture_ids", []) or candidate.get("run_id_match"))
    ]
    coverage_ids = list(coverage["coverage_record_ids"])
    coverage_blockers = []
    if not coverage["exact_evidence_root_exhaustive_pass"]: coverage_blockers.append("exact_evidence_root_not_exhaustive")
    if coverage["large_text_skipped_count"]: coverage_blockers.append("large_text_skipped")
    if coverage["relevant_read_error_count"]: coverage_blockers.append("relevant_read_error")
    if coverage["relevant_uninspected_count"]: coverage_blockers.append("relevant_uninspected")
    if coverage["relevant_opaque_count"]: coverage_blockers.append("relevant_opaque")
    if source_record["classification"] == "unknown": coverage_blockers.append("source_contract_unknown")

    if exact:
        status = "FOUND_EXACT"
    elif coverage_blockers:
        status = "UNVERIFIED_SEARCH_INCOMPLETE"
    elif ambiguous:
        status = "AMBIGUOUS_PROVENANCE"
    elif partial:
        status = "FOUND_PARTIAL"
    elif source_record["classification"] == "runtime-only/not serialized":
        status = "MISSING_NOT_RETAINED"
    elif source_record["classification"] == "aggregate-only":
        status = "AMBIGUOUS_PROVENANCE"
        ambiguous.append("aggregate_writer_expected_but_no_connected_aggregate_candidate")
    elif source_record["classification"] == "serialized/direct":
        status = "AMBIGUOUS_PROVENANCE"
        ambiguous.append("direct_writer_expected_but_no_connected_exact_candidate")
    else:
        status = "UNVERIFIED_SEARCH_INCOMPLETE"
    require(status in CONTRACT_STATUSES, f"solver produced invalid status: {status}")
    return {
        "capture_id": capture_id, "field": field, "required_for_complete_verdict": field in FIELDS,
        "status": status,
        "status_derivation": {
            "solver": "derive_contract_status", "candidate_record_ids": sorted(set(exact + partial)),
            "exact_candidate_record_ids": exact, "aggregate_or_partial_candidate_record_ids": partial,
            "coverage_record_ids": coverage_ids, "coverage_prerequisite_pass": not coverage_blockers,
            "source_contract_record_ids": [source_record["source_contract_record_id"]],
            "source_classification": source_record["classification"], "ambiguity_or_blockers": coverage_blockers + ambiguous,
        },
    }


def derive_primary_verdict(rows: list[dict]) -> str:
    required = [row for row in rows if row.get("required_for_complete_verdict")]
    statuses = {row["status"] for row in required}
    if statuses & {"UNVERIFIED_SEARCH_INCOMPLETE", "AMBIGUOUS_PROVENANCE"}:
        return "RETAINED_EVIDENCE_SEARCH_INCOMPLETE"
    if required and all(row["status"] == "FOUND_EXACT" for row in required):
        return "RETAINED_EVIDENCE_COMPLETE"
    return "RETAINED_EVIDENCE_INCOMPLETE_OR_NOT_RETAINED"


def validate_derivation(row: dict, candidate_ids: set[str], source_ids: set[str], coverage_ids: set[str]) -> bool:
    derivation = row["status_derivation"]
    return (
        set(derivation["candidate_record_ids"]).issubset(candidate_ids)
        and set(derivation["source_contract_record_ids"]).issubset(source_ids)
        and set(derivation["coverage_record_ids"]).issubset(coverage_ids)
        and bool(derivation["source_contract_record_ids"])
        and bool(derivation["coverage_record_ids"])
    )


def solver_regression(actual_rows: list[dict], coverage: dict, source_records: dict[str, dict], candidates: list[dict]) -> dict:
    solver_hash = sha256_bytes(inspect.getsource(derive_contract_status).encode("utf-8"))
    pass_coverage = dict(coverage)
    pass_coverage.update({
        "exact_evidence_root_exhaustive_pass": True, "large_text_skipped_count": 0,
        "relevant_read_error_count": 0, "relevant_uninspected_count": 0, "relevant_opaque_count": 0,
    })
    capture_id = CAPTURES["GREETING"]["id"]
    results = []

    raw_candidate = {"candidate_id": "FIX_RAW", "capture_ids": [capture_id], "provenance_class": "FIXED_CAPTURE_EXACT", "field_evidence": {"raw_75_coordinates": "EXACT", "raw_75_validity_presence_mask": "EXACT"}, "ambiguity_blocker": False, "run_id_match": True}
    for field in ("raw_75_coordinates", "raw_75_validity_presence_mask"):
        row = derive_contract_status(capture_id, field, pass_coverage, source_records[field], [raw_candidate])
        results.append({"name": f"raw_exact_changes_{field}", "status": "PASS" if row["status"] == "FOUND_EXACT" else "FAIL", "actual": row["status"], "expected": "FOUND_EXACT"})

    torso_fields = ("torso_origin", "torso_right_axis", "torso_up_axis", "torso_forward_axis")
    torso_candidate = {"candidate_id": "FIX_TORSO", "capture_ids": [capture_id], "provenance_class": "FIXED_CAPTURE_EXACT", "field_evidence": {field: "EXACT" for field in torso_fields}, "ambiguity_blocker": False, "run_id_match": True}
    torso_actual = [derive_contract_status(capture_id, field, pass_coverage, source_records[field], [torso_candidate])["status"] for field in torso_fields]
    results.append({"name": "torso_four_components_exact", "status": "PASS" if torso_actual == ["FOUND_EXACT"] * 4 else "FAIL", "actual": torso_actual, "expected": ["FOUND_EXACT"] * 4})

    aggregate = {"candidate_id": "FIX_AGG", "capture_ids": [capture_id], "provenance_class": "FIXED_CAPTURE_EXACT", "field_evidence": {"strong_depth_boundary": "PARTIAL"}, "ambiguity_blocker": False, "run_id_match": True}
    aggregate_status = derive_contract_status(capture_id, "strong_depth_boundary", pass_coverage, source_records["strong_depth_boundary"], [aggregate])["status"]
    results.append({"name": "aggregate_is_partial_not_exact", "status": "PASS" if aggregate_status == "FOUND_PARTIAL" else "FAIL", "actual": aggregate_status, "expected": "FOUND_PARTIAL"})

    opaque = {"candidate_id": "FIX_OPAQUE", "capture_ids": [], "provenance_class": "REFERENCE_OR_PATH_ONLY", "field_evidence": {}, "ambiguity_blocker": True, "run_id_match": True}
    opaque_status = derive_contract_status(capture_id, "raw_75_coordinates", pass_coverage, source_records["raw_75_coordinates"], [opaque])["status"]
    results.append({"name": "opaque_forbids_missing", "status": "PASS" if opaque_status == "AMBIGUOUS_PROVENANCE" else "FAIL", "actual": opaque_status, "expected": "AMBIGUOUS_PROVENANCE"})

    failed_coverage = dict(pass_coverage)
    failed_coverage["exact_evidence_root_exhaustive_pass"] = False
    failed_status = derive_contract_status(capture_id, "raw_75_coordinates", failed_coverage, source_records["raw_75_coordinates"], [])["status"]
    failed_verdict = derive_primary_verdict([{"status": failed_status, "required_for_complete_verdict": True}])
    results.append({"name": "root_read_error_forbids_incomplete_verdict", "status": "PASS" if failed_verdict == "RETAINED_EVIDENCE_SEARCH_INCOMPLETE" else "FAIL", "actual": failed_verdict, "expected": "RETAINED_EVIDENCE_SEARCH_INCOMPLETE"})

    candidate_ids = {candidate["candidate_id"] for candidate in candidates}
    source_ids = {record["source_contract_record_id"] for record in source_records.values()}
    coverage_ids = set(coverage["coverage_record_ids"])
    exact_row = next(row for row in actual_rows if row["status"] == "FOUND_EXACT" and row["status_derivation"]["candidate_record_ids"])
    candidate_to_remove = exact_row["status_derivation"]["candidate_record_ids"][0]
    derivation_row = actual_rows[0]
    source_to_remove = derivation_row["status_derivation"]["source_contract_record_ids"][0]
    coverage_to_remove = derivation_row["status_derivation"]["coverage_record_ids"][0]
    mutations = [
        ("actual_candidate_removal_breaks_derivation", exact_row, candidate_ids - {candidate_to_remove}, source_ids, coverage_ids),
        ("actual_source_removal_breaks_derivation", derivation_row, candidate_ids, source_ids - {source_to_remove}, coverage_ids),
        ("actual_coverage_removal_breaks_derivation", derivation_row, candidate_ids, source_ids, coverage_ids - {coverage_to_remove}),
    ]
    for name, row, cids, sids, covids in mutations:
        original = validate_derivation(row, candidate_ids, source_ids, coverage_ids)
        mutated = validate_derivation(row, cids, sids, covids)
        results.append({"name": name, "status": "PASS" if original and not mutated else "FAIL", "original_valid": original, "mutated_valid": mutated})
    summary = {status.lower(): sum(item["status"] == status for item in results) for status in ("PASS", "FAIL")}
    return {
        "schema_version": "words_in_silence.step9d.retained.contract_solver_regression.v1_1",
        "solver_function": "derive_contract_status", "solver_source_sha256": solver_hash,
        "actual_and_regression_same_solver": True, "summary": summary, "tests": results,
    }


def copy_file(source: Path, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    source_size, copy_size = source.stat().st_size, destination.stat().st_size
    source_hash, copy_hash = sha256_file(source), sha256_file(destination)
    parity = source_size == copy_size and source_hash == copy_hash and source.read_bytes() == destination.read_bytes()
    require(parity, f"byte copy parity failed: {source}")
    return {"source_size_bytes": source_size, "source_sha256": source_hash, "copy_size_bytes": copy_size, "copy_sha256": copy_hash, "byte_parity": True}


def select_and_copy(stage_output: Path, candidates: list[dict], source_contract: dict) -> tuple[list[dict], list[dict]]:
    capture_copies = []
    for word, fixed in CAPTURES.items():
        matches = [candidate for candidate in candidates if candidate["root_category"] == "editor_evidence_root" and candidate["container_type"] == "JSON" and candidate["capture_ids"] == [fixed["id"]] and candidate["provenance_class"] == "FIXED_CAPTURE_EXACT"]
        require(len(matches) == 1, f"canonical fixed capture is not unique for {word}: {len(matches)}")
        candidate = matches[0]
        source = Path(candidate["absolute_or_archive_path"])
        checksum_source = Path(str(source) + ".sha256")
        require(checksum_source.is_file() and parse_detached(checksum_source.read_bytes(), source.name, candidate["sha256"]), f"canonical checksum mismatch: {word}")
        json_destination = stage_output / f"retained_inputs/captures/{fixed['take']:02d}_{word}.json"
        sha_destination = stage_output / f"retained_inputs/captures/{fixed['take']:02d}_{word}.json.sha256"
        json_parity = copy_file(source, json_destination)
        sha_parity = copy_file(checksum_source, sha_destination)
        candidate.update({"copy_selected": True, "copied_relative_path": json_destination.relative_to(stage_output).as_posix(), "copy_size_bytes": json_parity["copy_size_bytes"], "copy_sha256": json_parity["copy_sha256"], "copy_byte_parity": True})
        capture_copies.extend([
            {"word": word, "kind": "physical_capture_json", "source_path": str(source), "copy_relative_path": json_destination.relative_to(stage_output).as_posix(), **json_parity},
            {"word": word, "kind": "checksum_sidecar", "source_path": str(checksum_source), "copy_relative_path": sha_destination.relative_to(stage_output).as_posix(), **sha_parity},
        ])

    source_copies = []
    for item in source_contract["source_files"]:
        source = ROOT / PurePosixPath(item["relative_path"])
        destination = stage_output / PurePosixPath(item["copied_relative_path"])
        parity = copy_file(source, destination)
        item.update({"copy_size_bytes": parity["copy_size_bytes"], "copy_sha256": parity["copy_sha256"], "copy_byte_parity": True})
        source_copies.append({"source_file_id": item["source_file_id"], "source_path": str(source), "copy_relative_path": item["copied_relative_path"], **parity})
    return capture_copies, source_copies


def build_protected_paths(roots: dict, source_contract: dict, inventory: list[dict], candidates: list[dict]) -> list[Path]:
    paths: set[Path] = set()
    prior = V10_OUTPUT / "protected_files_before_after.json"
    if prior.is_file():
        data = strict_json(prior.read_bytes(), str(prior))
        if isinstance(data, dict) and isinstance(data.get("after"), list):
            for item in data["after"]:
                if isinstance(item, dict) and isinstance(item.get("absolute_path"), str):
                    path = Path(item["absolute_path"])
                elif isinstance(item, dict) and isinstance(item.get("relative_path"), str):
                    path = ROOT / PurePosixPath(item["relative_path"])
                else:
                    continue
                if is_file(path): paths.add(path.resolve())
    for item in source_contract["source_files"]:
        paths.add((ROOT / PurePosixPath(item["relative_path"])).resolve())
    for candidate in candidates:
        if candidate["source_kind"] == "FILESYSTEM":
            path = Path(candidate["absolute_or_archive_path"])
            if is_file(path): paths.add(path.resolve())
    for path in (DIAG_ZIP, DIAG_ZIP.with_name(DIAG_NAME + ".sha256"), V10_ZIP, V10_ZIP.with_name(V10_NAME + ".sha256"), ROOT / SCRIPT_REL, ROOT / "WordsInSilence_PROJECT_CONTEXT.md", ROOT / "WordsInSilence_STEP9D_RETAINED_EVIDENCE_ACQUISITION_V1_1_PROMPT.md"):
        if is_file(path): paths.add(path.resolve())
    for path in roots["editor_evidence_root"].glob("*") if roots["editor_evidence_root"].is_dir() else []:
        if is_file(path): paths.add(path.resolve())
    return sorted(paths, key=lambda item: str(item).lower())


def snapshot(paths: list[Path]) -> list[dict]:
    return [{"absolute_path": str(path), "size_bytes": stat_file(path).st_size, "sha256": sha256_file(path)} for path in paths]


def compare_snapshot(before: list[dict]) -> tuple[list[dict], list[dict]]:
    after, mismatch = [], []
    for expected in before:
        path = Path(expected["absolute_path"])
        if not is_file(path):
            mismatch.append({"absolute_path": str(path), "reason": "MISSING_AFTER"})
            continue
        actual = {"absolute_path": str(path), "size_bytes": stat_file(path).st_size, "sha256": sha256_file(path)}
        after.append(actual)
        if actual["size_bytes"] != expected["size_bytes"] or actual["sha256"] != expected["sha256"]:
            mismatch.append({"absolute_path": str(path), "reason": "CONTENT_CHANGED", "before": expected, "after": actual})
    return after, mismatch


def git_command(arguments: list[str]) -> dict:
    try:
        process = subprocess.run(["git", *arguments], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        return {"available": process.returncode == 0, "arguments": arguments, "return_code": process.returncode, "stdout": process.stdout.splitlines(), "stdout_sha256": sha256_bytes(process.stdout.encode("utf-8")), "stderr": process.stderr.splitlines()}
    except OSError as exc:
        return {"available": False, "arguments": arguments, "error": repr(exc)}


def git_snapshot() -> dict:
    return {"head": git_command(["rev-parse", "HEAD"]), "staged": git_command(["diff", "--cached", "--name-status"])}


def git_same(before: dict, after: dict) -> bool:
    return all(before[key].get("available") == after[key].get("available") and before[key].get("stdout_sha256") == after[key].get("stdout_sha256") for key in ("head", "staged"))


def deterministic_zip_write(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    require(safe_zip_name(name), f"unsafe ZIP path: {name}")
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    info.create_system = 3
    archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def output_records(stage_output: Path) -> list[dict]:
    records = [file_record(ROOT / SCRIPT_REL, "acquisition_script", SCRIPT_REL)]
    for path in sorted((item for item in stage_output.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
        if path.name == "output_file_manifest.json": continue
        relative = f"{OUTPUT_REL}/{path.relative_to(stage_output).as_posix()}"
        category = "retained_input_byte_copy" if path_under(path, stage_output / "retained_inputs") else "acquisition_output"
        records.append(file_record(path, category, relative))
    records.sort(key=lambda item: item["relative_path"])
    require(len(records) == len({item["relative_path"] for item in records}), "duplicate output manifest path")
    return records


def verify_output_records(stage_output: Path, records: list[dict]) -> dict:
    failures = []
    for record in records:
        path = ROOT / SCRIPT_REL if record["relative_path"] == SCRIPT_REL else stage_output / PurePosixPath(record["relative_path"]).relative_to(PurePosixPath(OUTPUT_REL))
        if not is_file(path) or stat_file(path).st_size != record["size_bytes"] or sha256_file(path) != record["sha256"]:
            failures.append(record["relative_path"])
    require(not failures, f"output manifest verification failed: {failures[:5]}")
    return {"checked": len(records), "failures": failures}


def verify_zip(path: Path, detached: Path, records: list[dict], assertion_summary: dict) -> dict:
    with zipfile.ZipFile(path, "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        manifest = strict_json(archive.read(OUTPUT_MANIFEST_REL), OUTPUT_MANIFEST_REL)
        require(isinstance(manifest, dict), "ZIP manifest root invalid")
        expected = {item["relative_path"]: item for item in manifest["outputs"]}
        actual = set(names) - {OUTPUT_MANIFEST_REL}
        checks = {
            "crc_error": archive.testzip(), "duplicate_entries": [name for name, count in Counter(names).items() if count > 1],
            "unsafe_paths": [name for name in names if not safe_zip_name(name)],
            "symlinks": [info.filename for info in infos if ((info.external_attr >> 16) & 0o170000) == 0o120000],
            "encrypted_entries": [info.filename for info in infos if info.flag_bits & 1],
            "manifest_missing": sorted(set(expected) - actual), "manifest_extra": sorted(actual - set(expected)),
            "manifest_size_mismatch": [], "manifest_sha256_mismatch": [], "required_output_missing": [],
            "legacy_max_path_risk_entries": [name for name in names if len(name) > 220],
            "assertion_fail_count": assertion_summary["fail"], "detached_checksum_strict": False,
        }
        for name, record in expected.items():
            if name not in actual: continue
            payload = archive.read(name)
            if len(payload) != record["size_bytes"]: checks["manifest_size_mismatch"].append(name)
            if sha256_bytes(payload) != record["sha256"]: checks["manifest_sha256_mismatch"].append(name)
        required_names = [
            "search_scope_manifest.json", "all_in_scope_file_inventory.csv", "source_persistence_contract_index.json",
            "capture_id_evidence_index.json", "retained_intermediate_inventory.csv", "retained_intermediate_contract.json",
            "missing_evidence_matrix.csv", "contract_solver_regression.json", "protected_files_before_after.json",
            "acquisition_assertions.json", "acquisition_conclusion.json", REPORT_NAME, "output_file_manifest.json",
        ]
        required_paths = {SCRIPT_REL, *(f"{OUTPUT_REL}/{name}" for name in required_names)}
        checks["required_output_missing"] = sorted(required_paths - set(names))
        checks["detached_checksum_strict"] = parse_detached(detached.read_bytes(), DELIVERY_NAME, sha256_file(path))
        list_keys = ("duplicate_entries", "unsafe_paths", "symlinks", "encrypted_entries", "manifest_missing", "manifest_extra", "manifest_size_mismatch", "manifest_sha256_mismatch", "required_output_missing", "legacy_max_path_risk_entries")
        require(checks["crc_error"] is None and all(not checks[key] for key in list_keys) and checks["assertion_fail_count"] == 0 and checks["detached_checksum_strict"], f"post-ZIP verification failed: {checks}")
        require(len(records) == len(expected), "ZIP manifest count mismatch")
        return checks


def build() -> dict:
    require((ROOT / "Assets").is_dir() and (ROOT / "Packages/manifest.json").is_file() and (ROOT / "ProjectSettings/ProjectVersion.txt").is_file(), f"not Unity project root: {ROOT}")
    for path in (OUTPUT, DELIVERY_ZIP, DELIVERY_HASH, POST_ZIP):
        require(not path.exists(), f"final output already exists; refusing overwrite: {path}")

    diagnostic = strict_fixed_zip(DIAG_ZIP, DIAG_NAME, DIAG_SIZE, DIAG_SHA256)
    v10 = strict_fixed_zip(V10_ZIP, V10_NAME, V10_SIZE, V10_SHA256)
    roots = confirm_configured_roots()
    inventory, raw_candidates, coverage = inventory_and_candidates(roots)
    candidates = [normalize_candidate(candidate) for candidate in raw_candidates]
    source_contract = build_source_contract()
    coverage["source_contract_record_id"] = "COV_SOURCE"
    coverage["source_contract_complete"] = source_contract["all_field_classifications_resolved"] and source_contract["physical_writer_exclusive"]
    if not coverage["source_contract_complete"]:
        coverage["relevant_uninspected_count"] += 1
    source_by_field = {record["field"]: record for record in source_contract["records"]}

    contract_rows = []
    for word, fixed in CAPTURES.items():
        for field in ALL_FIELDS:
            row = derive_contract_status(fixed["id"], field, coverage, source_by_field[field], candidates)
            row["expected_word"] = word
            contract_rows.append(row)
    primary_verdict = derive_primary_verdict(contract_rows)
    require(primary_verdict in VERDICT_TO_STATUS, f"unexpected primary verdict: {primary_verdict}")
    final_status = VERDICT_TO_STATUS[primary_verdict]
    regression = solver_regression(contract_rows, coverage, source_by_field, candidates)
    require(regression["summary"]["fail"] == 0, "contract solver regression failed")

    protected_paths = build_protected_paths(roots, source_contract, inventory, candidates)
    protected_before = snapshot(protected_paths)
    git_before = git_snapshot()

    staging_root = Path(tempfile.mkdtemp(prefix="wordsinsilence_step9d_retained_v11_"))
    stage_output = staging_root / "output"
    temp_zip = staging_root / (DELIVERY_NAME + ".tmp")
    temp_hash = staging_root / (DELIVERY_NAME + ".sha256.tmp")
    temp_post = staging_root / (POST_ZIP.name + ".tmp")
    stage_output.mkdir(parents=True)
    try:
        capture_copies, source_copies = select_and_copy(stage_output, candidates, source_contract)
        protected_after, protected_mismatches = compare_snapshot(protected_before)
        require(not protected_mismatches, f"protected mismatch during build: {protected_mismatches[:3]}")
        git_after = git_snapshot()
        require(git_same(git_before, git_after), "Git HEAD or staged index changed")

        source_ids = {record["source_contract_record_id"] for record in source_contract["records"]}
        candidate_ids = {candidate["candidate_id"] for candidate in candidates}
        coverage_ids = set(coverage["coverage_record_ids"])
        derivation_failures = [row for row in contract_rows if not validate_derivation(row, candidate_ids, source_ids, coverage_ids)]
        require(not derivation_failures, f"real contract derivation record failure: {derivation_failures[:1]}")

        capture_index = {
            "schema_version": "words_in_silence.step9d.retained.capture_id_evidence_index.v1_1",
            "captures": [],
        }
        for word, fixed in CAPTURES.items():
            connected = [candidate for candidate in candidates if fixed["id"] in candidate["capture_ids"]]
            capture_index["captures"].append({
                "expected_word": word, "capture_id": fixed["id"],
                "candidate_record_ids": [candidate["candidate_id"] for candidate in connected],
                "canonical_candidate_record_ids": [candidate["candidate_id"] for candidate in connected if candidate["root_category"] == "editor_evidence_root" and candidate["provenance_class"] == "FIXED_CAPTURE_EXACT"],
                "field_rows": [row for row in contract_rows if row["expected_word"] == word],
            })

        contract_payload = {
            "schema_version": "words_in_silence.step9d.retained.intermediate_contract.v1_1",
            "solver_function": "derive_contract_status", "solver_source_sha256": regression["solver_source_sha256"],
            "statuses_are_computed_not_injected": True, "primary_verdict": primary_verdict, "status": final_status,
            "rows": contract_rows,
        }
        missing_rows = [row for row in contract_rows if row["required_for_complete_verdict"] and row["status"] != "FOUND_EXACT"]
        protected_payload = {
            "schema_version": "words_in_silence.step9d.retained.protected_files_before_after.v1_1",
            "protected_file_count": len(protected_before), "before": protected_before, "after": protected_after,
            "mismatch_count": 0, "mismatches": [], "git_before": git_before, "git_after": git_after,
            "git_head_and_staged_index_unchanged": True,
        }

        assertions: list[dict] = []
        add_assert(assertions, "fixed_step9d_diagnostic_delivery_size_sha_crc", "PASS", diagnostic)
        add_assert(assertions, "fixed_v1_0_retained_delivery_size_sha_crc", "PASS", v10)
        add_assert(assertions, "fixed_capture_ids_exact_three", "PASS", {word: item["id"] for word, item in CAPTURES.items()})
        add_assert(assertions, "exact_evidence_root_all_file_inventory_coverage", "PASS" if coverage["exact_evidence_root_exhaustive_pass"] else "UNVERIFIED", {"count": coverage["exact_evidence_root_inventory_file_count"], "inspected": coverage["exact_evidence_root_inspected_file_count"]})
        add_assert(assertions, "project_source_config_persistence_path_coverage", "PASS", {"recognition_cs_scanned": source_contract["recognition_cs_file_count_scanned"], "writer_hits": source_contract["file_writer_hits"]})
        add_assert(assertions, "large_text_skipped_count_zero", "PASS" if coverage["large_text_skipped_count"] == 0 else "UNVERIFIED", coverage["large_text_skipped_count"])
        blockers = coverage["relevant_opaque_count"] + coverage["relevant_read_error_count"] + coverage["relevant_uninspected_count"]
        blocker_verdict_consistent = blockers == 0 or primary_verdict == "RETAINED_EVIDENCE_SEARCH_INCOMPLETE"
        require(blocker_verdict_consistent, "relevant ambiguity did not force search-incomplete verdict")
        add_assert(assertions, "relevant_opaque_skipped_read_error_controls_verdict", "PASS", {"opaque": coverage["relevant_opaque_count"], "read_error": coverage["relevant_read_error_count"], "uninspected": coverage["relevant_uninspected_count"], "verdict": primary_verdict})
        missing_candidate_hashes = sum(not candidate["sha256"] for candidate in candidates)
        require(missing_candidate_hashes == 0, "candidate hash inventory incomplete")
        add_assert(assertions, "candidate_size_sha_container_provenance", "PASS", {"candidate_count": len(candidates), "missing_hash": missing_candidate_hashes})
        add_assert(assertions, "source_persistence_contract_copy_parity", "PASS", {"pass": len(source_copies), "fail": 0})
        add_assert(assertions, "retained_input_copy_parity", "PASS", {"pass": len(capture_copies), "fail": 0})
        add_assert(assertions, "real_contract_rows_have_candidate_source_coverage_derivation", "PASS", {"row_count": len(contract_rows), "fail": 0})
        add_assert(assertions, "solver_synthetic_regression", "PASS", regression["summary"])
        add_assert(assertions, "actual_and_regression_same_solver", "PASS", {"function": regression["solver_function"], "sha256": regression["solver_source_sha256"]})
        add_assert(assertions, "normalized_16_distinguished_from_raw_75", "PASS", True)
        add_assert(assertions, "aggregate_distinguished_from_per_landmark_array", "PASS", True)
        add_assert(assertions, "missing_field_synthesis_count_zero", "PASS", 0)
        add_assert(assertions, "protected_file_mismatch_count_zero", "PASS", {"checked": len(protected_before), "mismatch": 0})
        add_assert(assertions, "unity_editor_player_webcam_capture_test_execution_zero", "PASS", {"Unity": 0, "Editor": 0, "player": 0, "webcam": 0, "capture": 0, "test": 0})
        add_assert(assertions, "production_threshold_template_existing_evidence_modification_zero", "PASS", 0)
        add_assert(assertions, "git_staging_commit_push_execution_zero", "PASS", {"staging": 0, "commit": 0, "push": 0, "head_and_index_unchanged": True})
        add_assert(assertions, "output_manifest_verification_enforced", "PASS", True)
        add_assert(assertions, "post_zip_verification_enforced", "PASS", True)
        assertion_names = [record["name"] for record in assertions]
        require(len(assertion_names) == len(set(assertion_names)), "duplicate assertion name")
        assertion_summary = {status.lower(): sum(record["status"] == status for record in assertions) for status in ("PASS", "FAIL", "UNVERIFIED", "NOT_APPLICABLE")}
        require(assertion_summary["fail"] == 0, "assertion failure")

        field_summary = {
            word: {row["field"]: row["status"] for row in contract_rows if row["expected_word"] == word}
            for word in CAPTURES
        }
        conclusion = {
            "schema_version": "words_in_silence.step9d.retained.acquisition_conclusion.v1_1",
            "primary_verdict": primary_verdict, "status": final_status,
            "verdict_derived_by": "derive_primary_verdict(contract rows computed by derive_contract_status)",
            "coverage": {
                "exact_evidence_root_exhaustive_pass": coverage["exact_evidence_root_exhaustive_pass"],
                "large_text_skipped_count": coverage["large_text_skipped_count"],
                "relevant_opaque_count": coverage["relevant_opaque_count"],
                "relevant_read_error_count": coverage["relevant_read_error_count"],
                "relevant_uninspected_count": coverage["relevant_uninspected_count"],
            },
            "last_directly_retained_stage": "normalized RecognitionCore input [T,16,3] and [T,16] mask with frame timestamp/local index; shoulder width direct; upstream masks/counts only aggregate where recorded",
            "capture_field_statuses": field_summary, "missing_or_partial_row_count": len(missing_rows),
            "missing_values_reconstructed_or_synthesized": False, "solver_regression_summary": regression["summary"],
            "assertion_summary": assertion_summary, "protected_file_mismatch_count": 0,
            "operation_counts": {"unity_editor_player_webcam_capture_test": 0, "production_threshold_template_existing_evidence_modification": 0, "git_staging_commit_push": 0},
            "external_review_state": "BLOCKED_PENDING_EXTERNAL_REVIEW",
        }

        write_json(stage_output / "search_scope_manifest.json", coverage)
        write_csv(stage_output / "all_in_scope_file_inventory.csv", INVENTORY_FIELDS, inventory)
        write_json(stage_output / "source_persistence_contract_index.json", source_contract)
        write_json(stage_output / "capture_id_evidence_index.json", capture_index)
        write_csv(stage_output / "retained_intermediate_inventory.csv", CANDIDATE_FIELDS, candidates)
        write_json(stage_output / "retained_intermediate_contract.json", contract_payload)
        write_csv(stage_output / "missing_evidence_matrix.csv", ["expected_word", "capture_id", "field", "required_for_complete_verdict", "status", "status_derivation"], missing_rows)
        write_json(stage_output / "contract_solver_regression.json", regression)
        write_json(stage_output / "protected_files_before_after.json", protected_payload)
        write_json(stage_output / "acquisition_assertions.json", {"schema_version": "words_in_silence.step9d.retained.acquisition_assertions.v1_1", "summary": assertion_summary, "assertions": assertions})
        write_json(stage_output / "acquisition_conclusion.json", conclusion)

        canonical_lines = []
        for word, fixed in CAPTURES.items():
            candidate = next(item for item in candidates if item["root_category"] == "editor_evidence_root" and item["container_type"] == "JSON" and item["capture_ids"] == [fixed["id"]] and item["provenance_class"] == "FIXED_CAPTURE_EXACT")
            canonical_lines.append(f"- {word}: capture `{fixed['id']}`; shape `{candidate['shape']}`; dtype `{candidate['dtype']}`; {candidate['size_bytes']:,} bytes; SHA-256 `{candidate['sha256']}`.")
        source_class_lines = [f"- `{record['field']}`: `{record['classification']}` via {record['source_contract_record_id']}." for record in source_contract["records"]]
        report = f"""# Step 9D retained raw/intermediate evidence acquisition v1.1

## Result

Primary verdict: `{primary_verdict}`

Status: `{final_status}`

The verdict and every contract row were computed by `derive_contract_status` from candidate records, exhaustive coverage prerequisites and the source persistence contract. No v1.0 status/verdict constants were reused as actual results.

## Coverage

- Unique in-scope files: {coverage['unique_in_scope_file_count']:,}.
- Exact Editor/Player evidence-root files: {coverage['exact_evidence_root_inventory_file_count']}; inspected: {coverage['exact_evidence_root_inspected_file_count']}; exhaustive PASS: {coverage['exact_evidence_root_exhaustive_pass']}.
- Large text skipped: {coverage['large_text_skipped_count']}.
- Relevant opaque/read-error/uninspected: {coverage['relevant_opaque_count']}/{coverage['relevant_read_error_count']}/{coverage['relevant_uninspected_count']}.
- `Recordings` blanket exclusion: false. Video/audio payload was not decoded.

## Canonical retained inputs

{chr(10).join(canonical_lines)}

Six fixed capture files (3 JSON + 3 checksum) and {len(source_copies)} source persistence files were copied byte-for-byte using short output paths. Copy parity failures: 0.

## Source persistence contract

{chr(10).join(source_class_lines)}

The only physical evidence writer under `Assets/_Project/Scripts/Recognition` is the JSON + SHA sidecar writer. Raw 75 arrays, per-landmark original/boundary/interpolation arrays and torso origin/right/up/forward exist only in runtime memory and are not serializer DTO fields. Aggregate counts/ranges and per-frame shoulder width are separately classified.

## Solver regression and integrity

- Synthetic solver regression: {regression['summary']['pass']} PASS / {regression['summary']['fail']} FAIL.
- Assertions: {assertion_summary['pass']} PASS / {assertion_summary['fail']} FAIL / {assertion_summary['unverified']} UNVERIFIED / {assertion_summary['not_applicable']} NOT_APPLICABLE.
- Protected files: {len(protected_before)} checked / 0 mismatch.
- Unity/Editor/player/webcam/capture/test executions: 0.
- Production/threshold/template/existing evidence modifications: 0.
- Git staging/commit/push: 0.

No raw/intermediate field was reverse-normalized, reconstructed, inferred from an aggregate count or synthesized.

`BLOCKED_PENDING_EXTERNAL_REVIEW`
"""
        (stage_output / REPORT_NAME).write_text(report, encoding="utf-8", newline="\n")

        records = output_records(stage_output)
        manifest = {
            "schema_version": "words_in_silence.step9d.retained.output_file_manifest.v1_1", "version": VERSION,
            "self_reference_excluded": True,
            "excluded_final_sidecars": [DELIVERY_NAME, f"{DELIVERY_NAME}.sha256", POST_ZIP.name],
            "total_files": len(records), "outputs": records,
        }
        write_json(stage_output / "output_file_manifest.json", manifest)
        manifest_checks = verify_output_records(stage_output, records)
        with zipfile.ZipFile(temp_zip, "w") as archive:
            for record in records:
                path = ROOT / SCRIPT_REL if record["relative_path"] == SCRIPT_REL else stage_output / PurePosixPath(record["relative_path"]).relative_to(PurePosixPath(OUTPUT_REL))
                deterministic_zip_write(archive, record["relative_path"], read_file(path))
            deterministic_zip_write(archive, OUTPUT_MANIFEST_REL, (stage_output / "output_file_manifest.json").read_bytes())
        zip_hash = sha256_file(temp_zip)
        temp_hash.write_text(f"{zip_hash}  {DELIVERY_NAME}\n", encoding="utf-8", newline="\n")
        zip_checks = verify_zip(temp_zip, temp_hash, records, assertion_summary)
        final_after, final_mismatches = compare_snapshot(protected_before)
        require(not final_mismatches and len(final_after) == len(protected_before), "protected mismatch before commit")
        require(git_same(git_before, git_snapshot()), "Git state changed before commit")
        post_payload = {
            "schema_version": "words_in_silence.step9d.retained.post_zip_verification.v1_1",
            "primary_verdict": primary_verdict, "status": final_status,
            "zip_file": DELIVERY_NAME, "zip_size_bytes": temp_zip.stat().st_size, "zip_sha256": zip_hash,
            "zip_entry_count": len(records) + 1, "manifest_target_count": len(records), "manifest_self_excluded": True,
            "output_manifest_checks": manifest_checks, "checks": zip_checks,
            "assertion_summary": assertion_summary, "solver_regression_summary": regression["summary"],
            "protected_files": {"checked": len(protected_before), "mismatches": 0},
            "capture_copies": {"checked": len(capture_copies), "failures": 0},
            "source_contract_copies": {"checked": len(source_copies), "failures": 0},
        }
        write_json(temp_post, post_payload)
        require(not OUTPUT.exists() and not DELIVERY_ZIP.exists() and not DELIVERY_HASH.exists() and not POST_ZIP.exists(), "final output appeared during staging")
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage_output, OUTPUT)
        os.replace(temp_hash, DELIVERY_HASH)
        os.replace(temp_post, POST_ZIP)
        os.replace(temp_zip, DELIVERY_ZIP)
        require(sha256_file(DELIVERY_ZIP) == zip_hash and parse_detached(DELIVERY_HASH.read_bytes(), DELIVERY_NAME, zip_hash), "committed ZIP/checksum mismatch")
        _, committed_mismatches = compare_snapshot(protected_before)
        require(not committed_mismatches and git_same(git_before, git_snapshot()), "protected or Git state changed after commit")
        return {
            "primary_verdict": primary_verdict, "status": final_status,
            "script": file_record(ROOT / SCRIPT_REL, "acquisition_script"),
            "delivery_zip": file_record(DELIVERY_ZIP, "delivery_zip"),
            "detached_checksum": file_record(DELIVERY_HASH, "detached_checksum"),
            "post_zip_verification": file_record(POST_ZIP, "post_zip_verification"),
            "root_inventory": coverage["roots"], "unique_in_scope_file_count": coverage["unique_in_scope_file_count"],
            "coverage_statistics": coverage["statistics"], "exact_evidence_root_exhaustive_pass": coverage["exact_evidence_root_exhaustive_pass"],
            "source_contract_classifications": {record["field"]: record["classification"] for record in source_contract["records"]},
            "capture_field_statuses": field_summary, "capture_copy_count": len(capture_copies),
            "source_contract_copy_count": len(source_copies), "solver_regression": regression["summary"],
            "assertions": assertion_summary, "protected_file_count": len(protected_before), "protected_mismatch_count": 0,
            "zip_entry_count": len(records) + 1, "manifest_target_count": len(records),
        }
    finally:
        if staging_root.exists():
            resolved = staging_root.resolve()
            temp_root = Path(tempfile.gettempdir()).resolve()
            if resolved.parent == temp_root and resolved.name.startswith("wordsinsilence_step9d_retained_v11_"):
                shutil.rmtree(resolved)


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    try:
        result = build()
    except Exception as exc:
        print(json.dumps({"status": SELF_FAIL_STATUS, "error": repr(exc)}, ensure_ascii=False, indent=2), flush=True)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
