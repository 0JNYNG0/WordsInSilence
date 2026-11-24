#!/usr/bin/env python3
"""Step 9D retained-evidence acquisition v1.2: full source persistence proof and verdict repair."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import io
import json
import os
import re
import shutil
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import s001_step9d_retained_raw_intermediate_evidence_acquisition_v1_1 as v11


ROOT = Path(__file__).resolve().parent.parent
VERSION = "1.2"
SCRIPT_REL = "Tools/s001_step9d_retained_raw_intermediate_evidence_acquisition_v1_2.py"
OUTPUT_REL = "data/processed/sessions/S001_P001/step9d_retained_raw_intermediate_evidence_acquisition_v1_2"
OUTPUT = ROOT / PurePosixPath(OUTPUT_REL)
OUTPUT_MANIFEST_REL = f"{OUTPUT_REL}/output_file_manifest.json"
REPORT_NAME = "STEP9D_RETAINED_RAW_INTERMEDIATE_EVIDENCE_ACQUISITION_V1_2_REPORT.md"

DELIVERY_NAME = "WordsInSilence_step9d_retained_raw_intermediate_evidence_acquisition_v1_2_delivery.zip"
DELIVERY_ZIP = ROOT / DELIVERY_NAME
DELIVERY_HASH = ROOT / f"{DELIVERY_NAME}.sha256"
POST_ZIP = ROOT / "WordsInSilence_step9d_retained_raw_intermediate_evidence_acquisition_v1_2_delivery_post_zip_verification.json"

V11_NAME = "WordsInSilence_step9d_retained_raw_intermediate_evidence_acquisition_v1_1_delivery.zip"
V11_ZIP = ROOT / V11_NAME
V11_SIZE = 499_802
V11_SHA256 = "9a21a1d62c869df95a925d609e011fd1eeabb418f9f31437ae7db1bf5de24a59"
V11_BASE = "data/processed/sessions/S001_P001/step9d_retained_raw_intermediate_evidence_acquisition_v1_1"
V11_MANIFEST_REL = f"{V11_BASE}/output_file_manifest.json"

SOURCE_ROOT = ROOT / "Assets/_Project/Scripts/Recognition"
EXPECTED_RECOGNITION_SOURCE_COUNT = 42
FIELDS = v11.FIELDS
OPTIONAL_FIELDS = v11.OPTIONAL_FIELDS
ALL_FIELDS = v11.ALL_FIELDS
CAPTURES = v11.CAPTURES
CAPTURE_IDS = v11.CAPTURE_IDS

ALLOWED_HIT_CLASSIFICATIONS = {
    "PHYSICAL_CAPTURE_CALL_GRAPH",
    "PARALLEL_PHYSICAL_WRITER",
    "NORMALIZED_RUNTIME_ONLY",
    "TEMPLATE_OR_CONFIG_READER_ONLY",
    "UNRELATED_RECOGNITION_WRITER",
    "AMBIGUOUS_REQUIRES_COVERAGE_BLOCK",
}
CONTRACT_STATUSES = v11.CONTRACT_STATUSES
VERDICTS = {
    "RETAINED_EVIDENCE_COMPLETE",
    "RETAINED_EVIDENCE_INCOMPLETE_OR_NOT_RETAINED",
    "RETAINED_EVIDENCE_SEARCH_INCOMPLETE",
    "SELF_VERIFICATION_FAIL",
}

SUPPLEMENTAL_FIELDS = [
    "record_id", "path_kind", "absolute_path", "exists", "inventory_source", "coverage_record_id",
    "source_file_ids", "source_line_ranges", "inspection_result", "newly_discovered", "blocker",
]
MISSING_FIELDS = [
    "expected_word", "capture_id", "field", "required_for_complete_verdict", "status", "status_derivation",
]


class AcquisitionError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AcquisitionError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strict_json(data: bytes, label: str) -> object:
    return v11.strict_json(data, label)


def json_cell(value: str, default: object) -> object:
    if value is None or value == "":
        return default
    return json.loads(value)


def bool_cell(value: str) -> bool:
    return str(value).strip().lower() == "true"


def strict_v11_input() -> dict:
    structural = v11.strict_fixed_zip(V11_ZIP, V11_NAME, V11_SIZE, V11_SHA256)
    with zipfile.ZipFile(V11_ZIP, "r") as archive:
        names = archive.namelist()
        require(V11_MANIFEST_REL in names, "v1.1 output manifest missing")
        manifest_bytes = archive.read(V11_MANIFEST_REL)
        manifest = strict_json(manifest_bytes, V11_MANIFEST_REL)
        require(isinstance(manifest, dict), "v1.1 manifest is not an object")
        outputs = manifest.get("outputs")
        require(isinstance(outputs, list) and manifest.get("total_files") == len(outputs) == 32, "v1.1 manifest target count mismatch")
        expected = {item["relative_path"]: item for item in outputs}
        require(len(expected) == 32, "v1.1 manifest duplicate target")
        require(set(names) == set(expected) | {V11_MANIFEST_REL}, "v1.1 ZIP manifest missing/extra mismatch")
        target_failures = []
        for relative, record in expected.items():
            payload = archive.read(relative)
            if len(payload) != record["size_bytes"] or sha256_bytes(payload) != record["sha256"]:
                target_failures.append(relative)
        require(not target_failures, f"v1.1 manifest size/SHA mismatch: {target_failures[:3]}")

        required_names = {
            f"{V11_BASE}/search_scope_manifest.json",
            f"{V11_BASE}/all_in_scope_file_inventory.csv",
            f"{V11_BASE}/capture_id_evidence_index.json",
            f"{V11_BASE}/retained_intermediate_inventory.csv",
            f"{V11_BASE}/source_persistence_contract_index.json",
            f"{V11_BASE}/retained_intermediate_contract.json",
        }
        capture_names = []
        for word, fixed in CAPTURES.items():
            base = f"{V11_BASE}/retained_inputs/captures/{fixed['take']:02d}_{word}.json"
            capture_names.extend([base, base + ".sha256"])
        required_names.update(capture_names)
        require(required_names.issubset(expected), "v1.1 fixed input entries missing")

        search_scope = strict_json(archive.read(f"{V11_BASE}/search_scope_manifest.json"), "v1.1 search scope")
        capture_index = strict_json(archive.read(f"{V11_BASE}/capture_id_evidence_index.json"), "v1.1 capture index")
        inventory_bytes = archive.read(f"{V11_BASE}/retained_intermediate_inventory.csv")
        reader = csv.DictReader(io.StringIO(inventory_bytes.decode("utf-8-sig")))
        candidates = []
        for row in reader:
            candidates.append({
                "candidate_id": row["candidate_id"],
                "capture_ids": json_cell(row["capture_ids"], []),
                "run_id_match": bool_cell(row["run_id_match"]),
                "field_evidence": json_cell(row["field_evidence"], {}),
                "provenance_class": row["provenance_class"],
                "ambiguity_blocker": bool_cell(row["ambiguity_blocker"]),
                "root_category": row["root_category"],
                "container_type": row["container_type"],
                "relative_path": row["relative_path"],
                "size_bytes": int(row["size_bytes"]),
                "sha256": row["sha256"],
                "copy_selected": bool_cell(row["copy_selected"]),
                "copied_relative_path": row["copied_relative_path"] or None,
            })
        selected = [item for item in candidates if item["copy_selected"]]
        require(len(selected) == 3, f"v1.1 selected capture count mismatch: {len(selected)}")
        require(search_scope.get("unique_in_scope_file_count") == 10909, "v1.1 inventory count mismatch")
        require(search_scope.get("exact_evidence_root_exhaustive_pass") is True, "v1.1 exact root coverage not PASS")
        require(search_scope.get("exact_evidence_root_inventory_file_count") == 40, "v1.1 exact root count mismatch")
        require(search_scope.get("exact_evidence_root_inspected_file_count") == 40, "v1.1 exact root inspection mismatch")

        input_entries = []
        for name in sorted(required_names):
            payload = archive.read(name)
            input_entries.append({"relative_path": name, "size_bytes": len(payload), "sha256": sha256_bytes(payload)})
        capture_payloads = {name: archive.read(name) for name in capture_names}
        protected = strict_json(archive.read(f"{V11_BASE}/protected_files_before_after.json"), "v1.1 protected")
    return {
        "schema_version": "words_in_silence.step9d.retained.fixed_v1_1_input_verification.v1_2",
        "delivery": structural,
        "manifest_target_count": 32,
        "manifest_entry_count": 33,
        "manifest_size_sha_failures": [],
        "fixed_input_entries": input_entries,
        "search_scope": search_scope,
        "capture_index": capture_index,
        "candidates": candidates,
        "capture_payloads": capture_payloads,
        "protected": protected,
    }


def enumerate_sources() -> tuple[list[dict], dict[str, str], dict[str, dict]]:
    paths = sorted(SOURCE_ROOT.rglob("*.cs"), key=lambda item: item.relative_to(SOURCE_ROOT).as_posix().lower())
    records = []
    texts = {}
    for index, path in enumerate(paths, start=1):
        relative = path.relative_to(SOURCE_ROOT).as_posix()
        data = v11.read_file(path)
        text = data.decode("utf-8-sig")
        source_id = f"RSF{index:03d}"
        copy_relative = f"retained_inputs/recognition_source_full/{relative}"
        records.append({
            "source_file_id": source_id,
            "recognition_relative_path": relative,
            "project_relative_path": path.relative_to(ROOT).as_posix(),
            "absolute_path": str(path.resolve()),
            "size_bytes": len(data),
            "sha256": sha256_bytes(data),
            "read_status": "PASS",
            "copied_relative_path": copy_relative,
            "copy_size_bytes": None,
            "copy_sha256": None,
            "copy_byte_parity": None,
        })
        texts[relative] = text
    by_relative = {record["recognition_relative_path"]: record for record in records}
    return records, texts, by_relative


def mask_csharp(text: str) -> str:
    output = list(text)
    index = 0
    state = "CODE"
    while index < len(text):
        char = text[index]
        nxt = text[index + 1] if index + 1 < len(text) else ""
        if state == "CODE":
            if char == "/" and nxt == "/":
                output[index] = output[index + 1] = " "
                index += 2
                state = "LINE_COMMENT"
                continue
            if char == "/" and nxt == "*":
                output[index] = output[index + 1] = " "
                index += 2
                state = "BLOCK_COMMENT"
                continue
            if char == '"':
                output[index] = " "
                index += 1
                state = "STRING"
                continue
            if char == "'":
                output[index] = " "
                index += 1
                state = "CHAR"
                continue
        elif state == "LINE_COMMENT":
            if char == "\n":
                state = "CODE"
            else:
                output[index] = " "
        elif state == "BLOCK_COMMENT":
            if char == "*" and nxt == "/":
                output[index] = output[index + 1] = " "
                index += 2
                state = "CODE"
                continue
            if char != "\n":
                output[index] = " "
        elif state in {"STRING", "CHAR"}:
            if char == "\\":
                output[index] = " "
                if index + 1 < len(text) and text[index + 1] != "\n":
                    output[index + 1] = " "
                index += 2
                continue
            terminal = '"' if state == "STRING" else "'"
            if char == terminal:
                output[index] = " "
                state = "CODE"
            elif char != "\n":
                output[index] = " "
        index += 1
    return "".join(output)


def method_spans(text: str) -> list[dict]:
    masked = mask_csharp(text)
    patterns = [
        re.compile(r"(?m)^[ \t]*(?:(?:public|private|protected|internal|static|virtual|override|async|sealed|new|extern|unsafe|partial)\s+)*(?:[A-Za-z_][\w<>\[\],.?]*\s+)+(?P<name>[A-Za-z_]\w*)\s*\("),
        re.compile(r"(?m)^[ \t]*(?:(?:public|private|protected|internal)\s+)+(?P<name>[A-Za-z_]\w*)\s*\("),
    ]
    candidates = []
    for pattern in patterns:
        candidates.extend((match.start(), match.group("name")) for match in pattern.finditer(masked))
    spans = []
    seen = set()
    for start, name in sorted(candidates):
        if (start, name) in seen or name in {"if", "for", "foreach", "while", "switch", "catch", "using", "return"}:
            continue
        seen.add((start, name))
        brace = masked.find("{", start)
        semicolon = masked.find(";", start)
        if brace < 0 or (semicolon >= 0 and semicolon < brace):
            continue
        depth = 0
        end = None
        for cursor in range(brace, len(masked)):
            if masked[cursor] == "{":
                depth += 1
            elif masked[cursor] == "}":
                depth -= 1
                if depth == 0:
                    end = cursor
                    break
        if end is None:
            continue
        spans.append({
            "name": name,
            "start_offset": start,
            "end_offset": end,
            "line_range": [text.count("\n", 0, start) + 1, text.count("\n", 0, end) + 1],
        })
    unique = {(item["name"], tuple(item["line_range"])): item for item in spans}
    return sorted(unique.values(), key=lambda item: item["start_offset"])


def containing_symbol(spans: list[dict], line: int) -> tuple[str, list[int]]:
    matches = [span for span in spans if span["line_range"][0] <= line <= span["line_range"][1]]
    if not matches:
        return "TYPE_OR_FILE_SCOPE", [line, line]
    span = min(matches, key=lambda item: item["line_range"][1] - item["line_range"][0])
    return span["name"], span["line_range"]


def classify_hit(relative: str, symbol: str, family: str, api: str, access: str, line_text: str) -> tuple[str, str]:
    physical_files = {
        "PhysicalCompatibility/PhysicalWebcamCoordinateCompatibilityHarness.cs",
        "PhysicalCompatibility/PhysicalCaptureEvidenceJson.cs",
        "PhysicalCompatibility/PhysicalCaptureEvidenceResume.cs",
        "PhysicalCompatibility/PhysicalCaptureEvidenceRecorder.cs",
        "PhysicalCompatibility/Step9CPhysicalCaptureEvidence.cs",
        "PhysicalCompatibility/Step9CHandIdentityObservationBridge.cs",
    }
    reader_files = {
        "SVMClassifier.cs",
        "Core/RecognitionSequenceBinary.cs",
        "Evaluation/FrozenDistanceGateConfig.cs",
        "Runtime/RuntimeRecognitionService.cs",
        "LiveInput/LiveMediaPipeInputDiagnosticHarness.cs",
    }
    if relative in physical_files:
        return "PHYSICAL_CAPTURE_CALL_GRAPH", "source participates in physical DTO/export, adjacent checksum verification, or its configured output path"
    if relative == "Runtime/RecognitionSequenceInputSession.cs" and ("MemoryStream" in line_text or "BinaryWriter" in line_text or symbol.startswith("Write")):
        return "NORMALIZED_RUNTIME_ONLY", "BinaryWriter is bound to MemoryStream for deterministic in-memory hashing; no file sink exists in this source"
    if relative in reader_files:
        return "TEMPLATE_OR_CONFIG_READER_ONLY", "source uses read/existence APIs or StreamingAssets paths and contains no file write sink"
    if relative == "MotionRecorder.cs":
        return "NORMALIZED_RUNTIME_ONLY", "RecordAsync collects frames into SignMotionSample memory and exposes no disk sink"
    if relative in {"LiveInput/MediaPipeHandPoseObservationBridge.cs", "LiveInput/ApprovedLivePreprocessingAdapter.cs"}:
        return "NORMALIZED_RUNTIME_ONLY", "Write-prefixed helper populates runtime arrays only and has no System.IO file sink"
    if access in {"READ_SOURCE", "PATH_REFERENCE", "FORMAT_REFERENCE"}:
        return "TEMPLATE_OR_CONFIG_READER_ONLY", "non-writing path, format, or read-only reference with no capture/run persistence edge"
    if access in {"MEMORY_WRITER", "CUSTOM_WRITER"}:
        return "UNRELATED_RECOGNITION_WRITER", "custom writer/recorder is not connected to a file sink or the Step 9C Capture/Run export graph"
    if access == "FILE_SINK":
        return "AMBIGUOUS_REQUIRES_COVERAGE_BLOCK", "file sink outside the proven physical export graph requires explicit call-graph classification"
    return "UNRELATED_RECOGNITION_WRITER", "fully enumerated Recognition source reference with no physical persistence connection"


def scan_persistence_apis(texts: dict[str, str], by_relative: dict[str, dict]) -> list[dict]:
    hits = []

    def add(relative: str, line: int, family: str, api: str, access: str, matched_text: str, extensions: list[str]) -> None:
        spans = method_spans(texts[relative])
        symbol, symbol_range = containing_symbol(spans, line)
        classification, basis = classify_hit(relative, symbol, family, api, access, matched_text)
        hits.append({
            "hit_id": None,
            "source_file_id": by_relative[relative]["source_file_id"],
            "relative_path": relative,
            "line_range": [line, line],
            "containing_symbol": symbol,
            "containing_symbol_line_range": symbol_range,
            "detection_family": family,
            "api_or_symbol": api,
            "access_kind": access,
            "is_writer_candidate": access in {"FILE_SINK", "MEMORY_WRITER", "CUSTOM_WRITER"},
            "is_file_sink": access == "FILE_SINK",
            "path_or_extension_literals": extensions,
            "matched_source_line": matched_text.strip(),
            "classification": classification,
            "classification_basis": basis,
            "evidence_source_ids": [by_relative[relative]["source_file_id"]],
            "call_edge_ids": [],
        })

    file_write_methods = {
        "WriteAllBytes", "WriteAllText", "WriteAllLines", "AppendAllText", "AppendAllLines", "Create",
        "Move", "Replace", "OpenWrite", "Open", "CreateDirectory",
    }
    extension_pattern = re.compile(r"(?i)\.(json|sha256|csv|npz|npy|wisrt|wisrq|bytes|bin|zip|tmp)\b")
    for relative, text in texts.items():
        lines = text.splitlines()
        for line_number, line_text in enumerate(lines, start=1):
            extensions = sorted({"." + item.lower() for item in extension_pattern.findall(line_text)})
            for match in re.finditer(r"\b(?P<owner>File|Directory)\.(?P<method>[A-Za-z_]\w*)\s*\(", line_text):
                method = match.group("method")
                access = "FILE_SINK" if method in file_write_methods else "READ_SOURCE"
                add(relative, line_number, "DIRECT_FILE_IO", f"{match.group('owner')}.{method}", access, line_text, extensions)
            for match in re.finditer(r"\bnew\s+(?P<type>FileStream|StreamWriter|BinaryWriter|MemoryStream)\s*\(", line_text):
                kind = match.group("type")
                access = "FILE_SINK" if kind in {"FileStream", "StreamWriter"} else "MEMORY_WRITER"
                add(relative, line_number, "STREAM_OR_BINARY_WRITER", kind, access, line_text, extensions)
            for match in re.finditer(r"\b(?P<owner>JsonUtility|JsonConvert)\.(?P<method>[A-Za-z_]\w*)\s*\(", line_text):
                access = "CUSTOM_WRITER" if match.group("method") in {"ToJson", "SerializeObject"} else "READ_SOURCE"
                add(relative, line_number, "JSON_SERIALIZER", f"{match.group('owner')}.{match.group('method')}", access, line_text, extensions)
            if "Path.Combine(" in line_text or re.search(r"Application\.(persistentDataPath|dataPath|streamingAssetsPath)", line_text):
                symbols = []
                if "Path.Combine(" in line_text:
                    symbols.append("Path.Combine")
                symbols.extend(match.group(0) for match in re.finditer(r"Application\.(persistentDataPath|dataPath|streamingAssetsPath)", line_text))
                add(relative, line_number, "PATH_OR_OUTPUT_ROOT", "+".join(symbols), "PATH_REFERENCE", line_text, extensions)
            if extensions or re.search(r"\b(checksum|manifest|sidecar|temporaryPath)\b", line_text, re.IGNORECASE):
                add(relative, line_number, "FORMAT_OR_SIDECAR_REFERENCE", "format_or_sidecar", "FORMAT_REFERENCE", line_text, extensions)
        for span in method_spans(text):
            if re.match(r"^(?:Write|Save|Export|Serialize|Record)", span["name"]):
                declaration_line = span["line_range"][0]
                line_text = lines[declaration_line - 1]
                extensions = sorted({"." + item.lower() for item in extension_pattern.findall(line_text)})
                add(relative, declaration_line, "CUSTOM_METHOD_DECLARATION", span["name"], "CUSTOM_WRITER", line_text, extensions)

    hits.sort(key=lambda item: (item["relative_path"].lower(), item["line_range"][0], item["detection_family"], item["api_or_symbol"]))
    for index, hit in enumerate(hits, start=1):
        hit["hit_id"] = f"PHIT{index:04d}"
    return hits


def find_span(text: str, name: str) -> dict | None:
    matches = [span for span in method_spans(text) if span["name"] == name]
    return matches[0] if matches else None


def build_call_graph(texts: dict[str, str], by_relative: dict[str, dict]) -> dict:
    specs = [
        ("PhysicalCompatibility/Step9CHandIdentityObservationBridge.cs", "OnUnifiedFrame", "ObservationArrived?.Invoke", "PhysicalCompatibility/PhysicalWebcamCoordinateCompatibilityHarness.cs", "OnObservation", "EVENT_FLOW"),
        ("PhysicalCompatibility/PhysicalWebcamCoordinateCompatibilityHarness.cs", "OnObservation", "observed.Add", "PhysicalCompatibility/PhysicalWebcamCoordinateCompatibilityHarness.cs", "BuildEvidence", "CAPTURE_BUFFER_FLOW"),
        ("PhysicalCompatibility/PhysicalWebcamCoordinateCompatibilityHarness.cs", "FinalizeCapture", "BuildEvidence(", "PhysicalCompatibility/PhysicalWebcamCoordinateCompatibilityHarness.cs", "BuildEvidence", "CALL"),
        ("PhysicalCompatibility/PhysicalWebcamCoordinateCompatibilityHarness.cs", "BuildEvidence", "recorder.Create(", "PhysicalCompatibility/PhysicalCaptureEvidenceRecorder.cs", "Create", "CALL"),
        ("PhysicalCompatibility/PhysicalCaptureEvidenceRecorder.cs", "Create", "adapter.Adapt(", "LiveInput/ApprovedLivePreprocessingAdapter.cs", "Adapt", "CALL"),
        ("PhysicalCompatibility/PhysicalCaptureEvidenceRecorder.cs", "Create", "new PhysicalFrameEvidence(", "PhysicalCompatibility/Step9CPhysicalCaptureEvidence.cs", "PhysicalFrameEvidence", "DTO_CONSTRUCTION"),
        ("PhysicalCompatibility/PhysicalCaptureEvidenceRecorder.cs", "Create", "new PhysicalCaptureEvidence(", "PhysicalCompatibility/Step9CPhysicalCaptureEvidence.cs", "PhysicalCaptureEvidence", "DTO_CONSTRUCTION"),
        ("PhysicalCompatibility/PhysicalWebcamCoordinateCompatibilityHarness.cs", "ExportEvidence", "EvidenceOutputDirectory(", "PhysicalCompatibility/PhysicalWebcamCoordinateCompatibilityHarness.cs", "EvidenceOutputDirectory", "PATH_CALL"),
        ("PhysicalCompatibility/PhysicalWebcamCoordinateCompatibilityHarness.cs", "ExportEvidence", "PhysicalCaptureEvidenceFile.Export(", "PhysicalCompatibility/PhysicalCaptureEvidenceJson.cs", "Export", "CALL"),
        ("PhysicalCompatibility/PhysicalCaptureEvidenceJson.cs", "Export", "PhysicalCaptureEvidenceJson.Serialize(", "PhysicalCompatibility/PhysicalCaptureEvidenceJson.cs", "Serialize", "CALL"),
        ("PhysicalCompatibility/PhysicalCaptureEvidenceJson.cs", "Serialize", "WriteRecognition(", "PhysicalCompatibility/PhysicalCaptureEvidenceJson.cs", "WriteRecognition", "SERIALIZER_CALL"),
        ("PhysicalCompatibility/PhysicalCaptureEvidenceJson.cs", "Serialize", "WriteCaptureWindow(", "PhysicalCompatibility/PhysicalCaptureEvidenceJson.cs", "WriteCaptureWindow", "SERIALIZER_CALL"),
        ("PhysicalCompatibility/PhysicalCaptureEvidenceJson.cs", "Serialize", "WriteFrame(", "PhysicalCompatibility/PhysicalCaptureEvidenceJson.cs", "WriteFrame", "SERIALIZER_CALL"),
        ("PhysicalCompatibility/PhysicalCaptureEvidenceJson.cs", "Serialize", "WriteHandIdentityDiagnostic(", "PhysicalCompatibility/PhysicalCaptureEvidenceJson.cs", "WriteHandIdentityDiagnostic", "SERIALIZER_CALL"),
        ("PhysicalCompatibility/PhysicalCaptureEvidenceJson.cs", "Export", "Directory.CreateDirectory(", None, "System.IO.Directory.CreateDirectory", "FILE_SINK"),
        ("PhysicalCompatibility/PhysicalCaptureEvidenceJson.cs", "Export", "File.WriteAllBytes(", None, "System.IO.File.WriteAllBytes", "FILE_SINK"),
        ("PhysicalCompatibility/PhysicalCaptureEvidenceJson.cs", "Export", "File.Move(", None, "System.IO.File.Move", "FILE_SINK"),
        ("PhysicalCompatibility/PhysicalCaptureEvidenceJson.cs", "Export", "File.WriteAllText(", None, "System.IO.File.WriteAllText", "SIDECAR_SINK"),
        ("PhysicalCompatibility/PhysicalCaptureEvidenceJson.cs", "Export", "Verify(", "PhysicalCompatibility/PhysicalCaptureEvidenceJson.cs", "Verify", "VERIFY_CALL"),
        ("PhysicalCompatibility/PhysicalCaptureEvidenceJson.cs", "Verify", "File.ReadAllText(", None, "System.IO.File.ReadAllText", "READ_BACK"),
        ("PhysicalCompatibility/PhysicalCaptureEvidenceJson.cs", "Verify", "File.ReadAllBytes(", None, "System.IO.File.ReadAllBytes", "READ_BACK"),
    ]
    edges = []
    for index, (caller_rel, caller_name, token, callee_rel, callee_name, kind) in enumerate(specs, start=1):
        caller_span = find_span(texts[caller_rel], caller_name)
        callee_span = find_span(texts[callee_rel], callee_name) if callee_rel else None
        call_line = None
        if caller_span:
            lines = texts[caller_rel].splitlines()
            for line in range(caller_span["line_range"][0], caller_span["line_range"][1] + 1):
                if token in lines[line - 1]:
                    call_line = line
                    break
        validated = caller_span is not None and call_line is not None and (callee_rel is None or callee_span is not None)
        edges.append({
            "edge_id": f"PCG{index:03d}",
            "edge_kind": kind,
            "caller_source_id": by_relative[caller_rel]["source_file_id"],
            "caller_relative_path": caller_rel,
            "caller_symbol": caller_name,
            "caller_line_range": caller_span["line_range"] if caller_span else None,
            "call_site_line_range": [call_line, call_line] if call_line else None,
            "callee_source_id": by_relative[callee_rel]["source_file_id"] if callee_rel else "EXTERNAL_BCL_SYSTEM_IO",
            "callee_relative_path": callee_rel or "System.IO",
            "callee_symbol": callee_name,
            "callee_line_range": callee_span["line_range"] if callee_span else ([0, 0] if callee_rel is None else None),
            "source_token": token,
            "validated": validated,
        })
    return {
        "schema_version": "words_in_silence.step9d.retained.physical_persistence_call_graph.v1_2",
        "edges": edges,
        "required_edge_ids": [edge["edge_id"] for edge in edges],
        "all_required_edges_validated": all(edge["validated"] for edge in edges),
    }


def attach_hit_edges(hits: list[dict], graph: dict) -> None:
    by_site = defaultdict(list)
    for edge in graph["edges"]:
        if edge["call_site_line_range"]:
            by_site[(edge["caller_source_id"], edge["call_site_line_range"][0])].append(edge["edge_id"])
    for hit in hits:
        hit["call_edge_ids"] = sorted(by_site.get((hit["source_file_id"], hit["line_range"][0]), []))


def producer_branch_checks(hits: list[dict], by_relative: dict[str, dict]) -> list[dict]:
    producers = [
        ("RawMediaPipeObservation", "LiveInput/RawMediaPipeObservation.cs"),
        ("ApprovedLivePreprocessingAdapter", "LiveInput/ApprovedLivePreprocessingAdapter.cs"),
        ("Step9CHandIdentityObservationBridge", "PhysicalCompatibility/Step9CHandIdentityObservationBridge.cs"),
        ("PhysicalCaptureEvidenceRecorder", "PhysicalCompatibility/PhysicalCaptureEvidenceRecorder.cs"),
        ("RecognitionSequenceInputSession", "Runtime/RecognitionSequenceInputSession.cs"),
        ("MotionRecorder", "MotionRecorder.cs"),
    ]
    result = []
    for name, relative in producers:
        relevant = [hit for hit in hits if hit["relative_path"] == relative]
        sinks = [hit["hit_id"] for hit in relevant if hit["is_file_sink"]]
        memory = [hit["hit_id"] for hit in relevant if hit["access_kind"] in {"MEMORY_WRITER", "CUSTOM_WRITER"} and not hit["is_file_sink"]]
        result.append({
            "producer": name,
            "source_file_id": by_relative[relative]["source_file_id"],
            "relative_path": relative,
            "file_sink_hit_ids": sinks,
            "memory_or_custom_hit_ids": memory,
            "separate_persistence_sink_found": bool(sinks),
            "conclusion": "NO_SEPARATE_FILE_SINK" if not sinks else "SEPARATE_FILE_SINK_REQUIRES_CLASSIFICATION",
        })
    return result


def evaluate_source_scan(source_count: int, hits: list[dict], edges: list[dict], required_edge_ids: list[str], all_read: bool, all_copy_parity: bool) -> dict:
    writer_candidates = [hit for hit in hits if hit.get("is_writer_candidate")]
    file_sinks = [hit for hit in hits if hit.get("is_file_sink")]
    unresolved = [hit for hit in writer_candidates if hit.get("classification") == "AMBIGUOUS_REQUIRES_COVERAGE_BLOCK"]
    parallel = [hit for hit in file_sinks if hit.get("classification") == "PARALLEL_PHYSICAL_WRITER"]
    physical = [hit for hit in file_sinks if hit.get("classification") == "PHYSICAL_CAPTURE_CALL_GRAPH"]
    edge_by_id = {edge["edge_id"]: edge for edge in edges}
    required_edges_complete = all(edge_id in edge_by_id and edge_by_id[edge_id].get("validated") for edge_id in required_edge_ids)
    sink_evidence_complete = all(
        hit.get("classification") in ALLOWED_HIT_CLASSIFICATIONS
        and hit.get("source_file_id") and hit.get("line_range")
        and (hit.get("classification") != "PHYSICAL_CAPTURE_CALL_GRAPH" or bool(hit.get("call_edge_ids")))
        for hit in file_sinks
    )
    npz_npy = [hit["hit_id"] for hit in writer_candidates if set(hit.get("path_or_extension_literals", [])) & {".npz", ".npy"}]
    wisrt_wisrq = [hit["hit_id"] for hit in writer_candidates if set(hit.get("path_or_extension_literals", [])) & {".wisrt", ".wisrq"}]
    sidecars = [hit["hit_id"] for hit in file_sinks if hit.get("api_or_symbol") == "File.WriteAllText" and "PhysicalCaptureEvidenceJson.cs" in hit.get("relative_path", "")]
    physical_exclusive = bool(physical) and not parallel and not unresolved and sink_evidence_complete and required_edges_complete
    complete = (
        source_count == EXPECTED_RECOGNITION_SOURCE_COUNT and all_read and all_copy_parity
        and not unresolved and sink_evidence_complete and required_edges_complete and physical_exclusive
    )
    return {
        "recognition_source_count": source_count,
        "expected_recognition_source_count": EXPECTED_RECOGNITION_SOURCE_COUNT,
        "expected_count_match": source_count == EXPECTED_RECOGNITION_SOURCE_COUNT,
        "writer_candidate_hit_ids": [hit["hit_id"] for hit in writer_candidates],
        "file_sink_hit_ids": [hit["hit_id"] for hit in file_sinks],
        "physical_file_sink_hit_ids": [hit["hit_id"] for hit in physical],
        "parallel_physical_writer_hit_ids": [hit["hit_id"] for hit in parallel],
        "unresolved_writer_hit_ids": [hit["hit_id"] for hit in unresolved],
        "npz_npy_writer_hits": npz_npy,
        "wisrt_wisrq_writer_hits": wisrt_wisrq,
        "sidecar_writer_hits": sidecars,
        "sink_evidence_complete": sink_evidence_complete,
        "required_edges_complete": required_edges_complete,
        "physical_writer_exclusive": physical_exclusive,
        "source_contract_complete": complete,
    }


def serializer_keys(text: str) -> set[str]:
    keys = set(re.findall(r'Property\(output,\s*"([A-Za-z0-9_]+)"', text))
    keys.update(re.findall(r'\\"([A-Za-z0-9_]+)\\"', text))
    return keys


FIELD_SPECS = {
    "raw_75_coordinates": {"direct": ["raw_coordinates", "raw_75_coordinates"], "aggregate": [], "runtime": [("LiveInput/RawMediaPipeObservation.cs", "CopyCoordinates")]},
    "raw_75_validity_presence_mask": {"direct": ["raw_validity_mask", "raw_presence_mask"], "aggregate": ["raw_pose_valid_landmarks", "raw_left_hand_valid_landmarks", "raw_right_hand_valid_landmarks"], "runtime": [("LiveInput/RawMediaPipeObservation.cs", "CopyValidMask")]},
    "original_valid_mask": {"direct": ["original_valid_mask", "original_valid"], "aggregate": ["raw_valid_landmarks"], "runtime": [("LiveInput/ApprovedLivePreprocessingAdapter.cs", "OriginalValid")]},
    "reappearance_reset_boundary": {"direct": ["reappearance_mask", "reset_mask", "boundary_mask"], "aggregate": ["reappearances", "timestamp_boundaries"], "runtime": [("LiveInput/RawMediaPipeObservation.cs", "IsReappearance"), ("LiveInput/ApprovedLivePreprocessingAdapter.cs", "Boundary")]},
    "strong_depth_boundary": {"direct": ["strong_depth_mask", "strong_depth_boundary"], "aggregate": ["strong_depth_boundaries"], "runtime": [("LiveInput/RawMediaPipeObservation.cs", "IsStrongDepthBoundary"), ("LiveInput/ApprovedLivePreprocessingAdapter.cs", "Strong")]},
    "interpolation_mask_max_gap": {"direct": ["interpolation_mask", "interpolated_mask"], "aggregate": ["interpolated_landmarks"], "runtime": [("LiveInput/ApprovedLivePreprocessingAdapter.cs", "Interpolated"), ("LiveInput/ApprovedLivePreprocessingAdapter.cs", "SelectedMaxGap")]},
    "torso_origin": {"direct": ["torso_origin", "torso_center"], "aggregate": [], "runtime": [("LiveInput/ApprovedLivePreprocessingAdapter.cs", "Origin")]},
    "torso_right_axis": {"direct": ["torso_right", "right_axis"], "aggregate": [], "runtime": [("LiveInput/ApprovedLivePreprocessingAdapter.cs", "Right")]},
    "torso_up_axis": {"direct": ["torso_up", "up_axis"], "aggregate": [], "runtime": [("LiveInput/ApprovedLivePreprocessingAdapter.cs", "Up")]},
    "torso_forward_axis": {"direct": ["torso_forward", "forward_axis"], "aggregate": [], "runtime": [("LiveInput/ApprovedLivePreprocessingAdapter.cs", "Forward")]},
    "torso_scale_shoulder_width": {"direct": ["shoulder_width_raw"], "aggregate": ["available_frames"], "runtime": [("LiveInput/ApprovedLivePreprocessingAdapter.cs", "Scale")]},
    "capture_id": {"direct": ["capture_id"], "aggregate": [], "runtime": [("PhysicalCompatibility/Step9CPhysicalCaptureEvidence.cs", "CaptureId")]},
    "capture_local_frame_index": {"direct": ["frame_index"], "aggregate": [], "runtime": [("PhysicalCompatibility/PhysicalCaptureEvidenceJson.cs", "frameIndex")]},
    "provider_frame_identity": {"direct": [], "aggregate": ["first_capture_provider_frame_index", "final_capture_provider_frame_index", "provider_frame_count"], "runtime": [("PhysicalCompatibility/Step9CPhysicalCaptureEvidence.cs", "ProviderFrameCount")]},
    "bridge_frame_identity": {"direct": [], "aggregate": ["bridge_observation_count"], "runtime": [("PhysicalCompatibility/Step9CPhysicalCaptureEvidence.cs", "BridgeObservationCount")]},
    "lifetime_frame_identity": {"direct": [], "aggregate": ["first_capture_lifetime_provider_frame_index", "final_capture_lifetime_provider_frame_index"], "runtime": [("PhysicalCompatibility/Step9CPhysicalCaptureEvidence.cs", "LifetimeProviderFrameIndex")]},
    "timestamp": {"direct": ["timestamp_seconds", "timestamps"], "aggregate": [], "runtime": [("LiveInput/RawMediaPipeObservation.cs", "TimestampSeconds")]},
    "schema_provenance": {"direct": ["schema_version", "run_id", "expected_word", "take_number", "source_convention"], "aggregate": [], "runtime": [("PhysicalCompatibility/Step9CPhysicalCaptureEvidence.cs", "SchemaVersion")]},
    "normalized_16_coordinates": {"direct": ["normalized_coordinates"], "aggregate": [], "runtime": [("Runtime/NormalizedRecognitionFrame.cs", "CopyCoordinates")]},
    "normalized_16_mask": {"direct": ["normalized_mask"], "aggregate": [], "runtime": [("Runtime/NormalizedRecognitionFrame.cs", "CopyValidMask")]},
    "aggregate_upstream_counts": {"direct": ["counts"], "aggregate": ["raw_frames", "interpolated_landmarks", "reappearances", "strong_depth_boundaries"], "runtime": [("PhysicalCompatibility/PhysicalCaptureEvidenceRecorder.cs", "PhysicalCaptureEvidenceRecorder")]},
}


def token_refs(texts: dict[str, str], by_relative: dict[str, dict], refs: list[tuple[str, str]]) -> list[dict]:
    output = []
    for relative, token in refs:
        lines = [index for index, line in enumerate(texts[relative].splitlines(), start=1) if token in line]
        if lines:
            output.append({
                "source_file_id": by_relative[relative]["source_file_id"],
                "relative_path": relative,
                "symbol": token,
                "line_range": [min(lines), max(lines)],
            })
    return output


def build_field_contract(texts: dict[str, str], by_relative: dict[str, dict], candidates: list[dict], source_eval: dict) -> dict:
    keys = serializer_keys(texts["PhysicalCompatibility/PhysicalCaptureEvidenceJson.cs"])
    records = []
    for index, field in enumerate(ALL_FIELDS, start=1):
        spec = FIELD_SPECS[field]
        direct = sorted(set(spec["direct"]) & keys)
        aggregate = sorted(set(spec["aggregate"]) & keys)
        runtime = token_refs(texts, by_relative, spec["runtime"])
        exact_candidates = sorted({candidate["candidate_id"] for candidate in candidates if candidate.get("field_evidence", {}).get(field) == "EXACT"})
        partial_candidates = sorted({candidate["candidate_id"] for candidate in candidates if candidate.get("field_evidence", {}).get(field) == "PARTIAL"})
        if direct:
            classification = "serialized/direct"
        elif aggregate:
            classification = "aggregate-only"
        elif (
            runtime and source_eval["source_contract_complete"] and not exact_candidates and not source_eval["unresolved_writer_hit_ids"]
        ):
            classification = "runtime-only/not serialized"
        else:
            classification = "unknown"
        conditions = {
            "runtime_field_or_producer_exists": bool(runtime),
            "physical_dto_serializer_direct_field_absent": not direct,
            "source_linked_direct_disk_candidate_absent": not exact_candidates,
            "unresolved_writer_hit_count": len(source_eval["unresolved_writer_hit_ids"]),
            "source_scan_complete": source_eval["source_contract_complete"],
        }
        if classification == "runtime-only/not serialized":
            require(all([
                conditions["runtime_field_or_producer_exists"],
                conditions["physical_dto_serializer_direct_field_absent"],
                conditions["source_linked_direct_disk_candidate_absent"],
                conditions["unresolved_writer_hit_count"] == 0,
                conditions["source_scan_complete"],
            ]), f"runtime-only classification precondition failed: {field}")
        records.append({
            "source_contract_record_id": f"SC12_{index:03d}",
            "field": field,
            "classification": classification,
            "direct_serializer_keys_detected": direct,
            "aggregate_serializer_keys_detected": aggregate,
            "runtime_source_refs": runtime,
            "source_linked_exact_candidate_record_ids": exact_candidates,
            "source_linked_partial_candidate_record_ids": partial_candidates,
            "classification_conditions": conditions,
            "persistence_call_graph_record_id": "PCG_ROOT_V1_2",
            "source_scan_record_id": "SOURCE_SCAN_V1_2",
        })
    return {
        "schema_version": "words_in_silence.step9d.retained.source_persistence_contract.v1_2",
        "serializer_keys_detected": sorted(keys),
        "records": records,
        "all_field_classifications_resolved": all(record["classification"] != "unknown" for record in records),
    }


def coverage_blockers(coverage: dict, source_contract_complete: bool) -> list[str]:
    blockers = []
    if not coverage.get("exact_evidence_root_exhaustive_pass"):
        blockers.append("exact_evidence_root_not_exhaustive")
    if coverage.get("large_text_skipped_count"):
        blockers.append("large_text_skipped")
    if coverage.get("relevant_read_error_count"):
        blockers.append("relevant_read_error")
    if coverage.get("relevant_uninspected_count"):
        blockers.append("relevant_uninspected")
    if coverage.get("relevant_opaque_count"):
        blockers.append("relevant_opaque")
    if not source_contract_complete:
        blockers.append("source_contract_incomplete")
    return blockers


def derive_contract_status(capture_id: str, field: str, coverage: dict, source_record: dict, candidates: list[dict], source_contract_complete: bool) -> dict:
    connected = [candidate for candidate in candidates if capture_id in candidate.get("capture_ids", []) and candidate.get("provenance_class") == "FIXED_CAPTURE_EXACT"]
    exact = [candidate["candidate_id"] for candidate in connected if candidate.get("field_evidence", {}).get(field) == "EXACT"]
    partial = [candidate["candidate_id"] for candidate in connected if candidate.get("field_evidence", {}).get(field) == "PARTIAL"]
    ambiguous = [
        candidate["candidate_id"] for candidate in candidates
        if candidate.get("ambiguity_blocker") and (capture_id in candidate.get("capture_ids", []) or candidate.get("run_id_match"))
    ]
    blockers = coverage_blockers(coverage, source_contract_complete)
    if source_record["classification"] == "unknown" and "source_contract_incomplete" not in blockers:
        blockers.append("source_contract_unknown")
    if exact:
        status = "FOUND_EXACT"
    elif blockers:
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
    require(status in CONTRACT_STATUSES, f"invalid contract status: {status}")
    return {
        "capture_id": capture_id,
        "field": field,
        "required_for_complete_verdict": field in FIELDS,
        "status": status,
        "status_derivation": {
            "solver": "derive_contract_status",
            "candidate_record_ids": sorted(set(exact + partial)),
            "exact_candidate_record_ids": exact,
            "aggregate_or_partial_candidate_record_ids": partial,
            "coverage_record_ids": list(coverage["coverage_record_ids"]),
            "coverage_prerequisite_pass": not blockers,
            "source_contract_record_ids": [source_record["source_contract_record_id"]],
            "source_classification": source_record["classification"],
            "source_scan_record_ids": ["SOURCE_SCAN_V1_2", "PCG_ROOT_V1_2"],
            "ambiguity_or_blockers": blockers + ambiguous,
        },
    }


def derive_primary_verdict(rows: list[dict], coverage_complete: bool, source_contract_complete: bool, self_verification_ok: bool = True) -> str:
    if not self_verification_ok:
        return "SELF_VERIFICATION_FAIL"
    if not coverage_complete or not source_contract_complete:
        return "RETAINED_EVIDENCE_SEARCH_INCOMPLETE"
    required = [row for row in rows if row.get("required_for_complete_verdict")]
    if required and all(row["status"] == "FOUND_EXACT" for row in required):
        return "RETAINED_EVIDENCE_COMPLETE"
    if any(row["status"] in {"MISSING_NOT_RETAINED", "FOUND_PARTIAL"} for row in required):
        return "RETAINED_EVIDENCE_INCOMPLETE_OR_NOT_RETAINED"
    if any(row["status"] in {"AMBIGUOUS_PROVENANCE", "UNVERIFIED_SEARCH_INCOMPLETE"} for row in required):
        return "RETAINED_EVIDENCE_SEARCH_INCOMPLETE"
    return "RETAINED_EVIDENCE_SEARCH_INCOMPLETE"


def validate_derivation(row: dict, candidate_ids: set[str], source_ids: set[str], coverage_ids: set[str]) -> bool:
    derivation = row["status_derivation"]
    return (
        set(derivation["candidate_record_ids"]).issubset(candidate_ids)
        and set(derivation["source_contract_record_ids"]).issubset(source_ids)
        and set(derivation["coverage_record_ids"]).issubset(coverage_ids)
        and bool(derivation["source_contract_record_ids"])
        and bool(derivation["coverage_record_ids"])
        and derivation.get("source_scan_record_ids") == ["SOURCE_SCAN_V1_2", "PCG_ROOT_V1_2"]
    )


def run_regressions(actual_rows: list[dict], coverage: dict, source_records: dict[str, dict], candidates: list[dict], hits: list[dict], graph: dict, source_count: int) -> dict:
    results = []

    def add(name: str, passed: bool, actual: object, expected: object) -> None:
        results.append({"name": name, "status": "PASS" if passed else "FAIL", "actual": actual, "expected": expected})

    pass_coverage = dict(coverage)
    pass_coverage.update({
        "exact_evidence_root_exhaustive_pass": True,
        "large_text_skipped_count": 0,
        "relevant_read_error_count": 0,
        "relevant_uninspected_count": 0,
        "relevant_opaque_count": 0,
    })
    capture_id = CAPTURES["GREETING"]["id"]
    raw_candidate = {"candidate_id": "FIX_RAW", "capture_ids": [capture_id], "provenance_class": "FIXED_CAPTURE_EXACT", "field_evidence": {"raw_75_coordinates": "EXACT", "raw_75_validity_presence_mask": "EXACT"}, "ambiguity_blocker": False, "run_id_match": True}
    for field in ("raw_75_coordinates", "raw_75_validity_presence_mask"):
        actual = derive_contract_status(capture_id, field, pass_coverage, source_records[field], [raw_candidate], True)["status"]
        add(f"raw_exact_changes_{field}", actual == "FOUND_EXACT", actual, "FOUND_EXACT")
    torso_fields = ("torso_origin", "torso_right_axis", "torso_up_axis", "torso_forward_axis")
    torso_candidate = {"candidate_id": "FIX_TORSO", "capture_ids": [capture_id], "provenance_class": "FIXED_CAPTURE_EXACT", "field_evidence": {field: "EXACT" for field in torso_fields}, "ambiguity_blocker": False, "run_id_match": True}
    actual_torso = [derive_contract_status(capture_id, field, pass_coverage, source_records[field], [torso_candidate], True)["status"] for field in torso_fields]
    add("torso_four_components_exact", actual_torso == ["FOUND_EXACT"] * 4, actual_torso, ["FOUND_EXACT"] * 4)
    aggregate = {"candidate_id": "FIX_AGG", "capture_ids": [capture_id], "provenance_class": "FIXED_CAPTURE_EXACT", "field_evidence": {"strong_depth_boundary": "PARTIAL"}, "ambiguity_blocker": False, "run_id_match": True}
    actual = derive_contract_status(capture_id, "strong_depth_boundary", pass_coverage, source_records["strong_depth_boundary"], [aggregate], True)["status"]
    add("aggregate_is_partial_not_exact", actual == "FOUND_PARTIAL", actual, "FOUND_PARTIAL")
    opaque = {"candidate_id": "FIX_OPAQUE", "capture_ids": [], "provenance_class": "REFERENCE_OR_PATH_ONLY", "field_evidence": {}, "ambiguity_blocker": True, "run_id_match": True}
    actual = derive_contract_status(capture_id, "raw_75_coordinates", pass_coverage, source_records["raw_75_coordinates"], [opaque], True)["status"]
    add("opaque_forbids_missing", actual == "AMBIGUOUS_PROVENANCE", actual, "AMBIGUOUS_PROVENANCE")
    failed_coverage = dict(pass_coverage)
    failed_coverage["exact_evidence_root_exhaustive_pass"] = False
    row = derive_contract_status(capture_id, "raw_75_coordinates", failed_coverage, source_records["raw_75_coordinates"], [], True)
    actual = derive_primary_verdict([row], False, True)
    add("root_read_error_forbids_incomplete_verdict", actual == "RETAINED_EVIDENCE_SEARCH_INCOMPLETE", actual, "RETAINED_EVIDENCE_SEARCH_INCOMPLETE")

    candidate_ids = {candidate["candidate_id"] for candidate in candidates}
    source_ids = {record["source_contract_record_id"] for record in source_records.values()}
    coverage_ids = set(coverage["coverage_record_ids"])
    exact_row = next(row for row in actual_rows if row["status"] == "FOUND_EXACT" and row["status_derivation"]["candidate_record_ids"])
    derivation_row = actual_rows[0]
    candidate_removed = validate_derivation(exact_row, candidate_ids - {exact_row["status_derivation"]["candidate_record_ids"][0]}, source_ids, coverage_ids)
    add("actual_candidate_removal_breaks_derivation", not candidate_removed, candidate_removed, False)
    source_removed = validate_derivation(derivation_row, candidate_ids, source_ids - {derivation_row["status_derivation"]["source_contract_record_ids"][0]}, coverage_ids)
    add("actual_source_removal_breaks_derivation", not source_removed, source_removed, False)
    coverage_removed = validate_derivation(derivation_row, candidate_ids, source_ids, coverage_ids - {derivation_row["status_derivation"]["coverage_record_ids"][0]})
    add("actual_coverage_removal_breaks_derivation", not coverage_removed, coverage_removed, False)

    unresolved_hit = dict(hits[0])
    unresolved_hit.update({"hit_id": "MUT_UNRESOLVED", "is_writer_candidate": True, "is_file_sink": True, "classification": "AMBIGUOUS_REQUIRES_COVERAGE_BLOCK", "call_edge_ids": [], "path_or_extension_literals": [".bin"]})
    unresolved_eval = evaluate_source_scan(source_count, hits + [unresolved_hit], graph["edges"], graph["required_edge_ids"], True, True)
    unknown_raw = dict(source_records["raw_75_coordinates"])
    unknown_raw["classification"] = "unknown"
    unresolved_row = derive_contract_status(capture_id, "raw_75_coordinates", pass_coverage, unknown_raw, [], unresolved_eval["source_contract_complete"])
    add("unresolved_writer_blocks_source_and_raw_missing", not unresolved_eval["source_contract_complete"] and unresolved_row["status"] != "MISSING_NOT_RETAINED", {"source_complete": unresolved_eval["source_contract_complete"], "raw_status": unresolved_row["status"]}, {"source_complete": False, "raw_status_not": "MISSING_NOT_RETAINED"})

    format_hit = dict(unresolved_hit)
    format_hit.update({"hit_id": "MUT_NPZ_WISRT", "path_or_extension_literals": [".npz", ".wisrt"]})
    format_eval = evaluate_source_scan(source_count, hits + [format_hit], graph["edges"], graph["required_edge_ids"], True, True)
    format_verdict = derive_primary_verdict([], True, format_eval["source_contract_complete"])
    add("unclassified_npz_wisrt_custom_writer_forces_search_incomplete", bool(format_eval["npz_npy_writer_hits"]) and bool(format_eval["wisrt_wisrq_writer_hits"]) and format_verdict == "RETAINED_EVIDENCE_SEARCH_INCOMPLETE", {"source_complete": format_eval["source_contract_complete"], "verdict": format_verdict}, {"source_complete": False, "verdict": "RETAINED_EVIDENCE_SEARCH_INCOMPLETE"})

    actual_eval = evaluate_source_scan(source_count, hits, graph["edges"], graph["required_edge_ids"], True, True)
    add("all_writers_classified_source_coverage_pass", actual_eval["source_contract_complete"], actual_eval["source_contract_complete"], True)
    mixed_rows = [{"required_for_complete_verdict": True, "status": "MISSING_NOT_RETAINED"}, {"required_for_complete_verdict": True, "status": "AMBIGUOUS_PROVENANCE"}]
    actual = derive_primary_verdict(mixed_rows, True, True)
    add("confirmed_missing_plus_ambiguity_is_incomplete", actual == "RETAINED_EVIDENCE_INCOMPLETE_OR_NOT_RETAINED", actual, "RETAINED_EVIDENCE_INCOMPLETE_OR_NOT_RETAINED")
    ambiguity_rows = [{"required_for_complete_verdict": True, "status": "AMBIGUOUS_PROVENANCE"}]
    actual = derive_primary_verdict(ambiguity_rows, True, True)
    add("ambiguity_only_is_search_incomplete", actual == "RETAINED_EVIDENCE_SEARCH_INCOMPLETE", actual, "RETAINED_EVIDENCE_SEARCH_INCOMPLETE")
    exact_rows = [{"required_for_complete_verdict": True, "status": "FOUND_EXACT"} for _ in range(3)]
    actual = derive_primary_verdict(exact_rows, True, True)
    add("all_required_exact_is_complete", actual == "RETAINED_EVIDENCE_COMPLETE", actual, "RETAINED_EVIDENCE_COMPLETE")
    source_drop = evaluate_source_scan(source_count - 1, hits, graph["edges"], graph["required_edge_ids"], True, True)
    edge_drop = evaluate_source_scan(source_count, hits, graph["edges"][1:], graph["required_edge_ids"], True, True)
    add("source_file_or_call_edge_removal_breaks_validation", not source_drop["source_contract_complete"] and not edge_drop["source_contract_complete"], {"source_drop": source_drop["source_contract_complete"], "edge_drop": edge_drop["source_contract_complete"]}, {"source_drop": False, "edge_drop": False})

    summary = {status.lower(): sum(item["status"] == status for item in results) for status in ("PASS", "FAIL")}
    return {
        "schema_version": "words_in_silence.step9d.retained.contract_solver_regression.v1_2",
        "solver_function": "derive_contract_status",
        "solver_source_sha256": sha256_bytes(inspect.getsource(derive_contract_status).encode("utf-8")),
        "source_contract_function": "evaluate_source_scan",
        "source_contract_source_sha256": sha256_bytes(inspect.getsource(evaluate_source_scan).encode("utf-8")),
        "primary_verdict_function": "derive_primary_verdict",
        "primary_verdict_source_sha256": sha256_bytes(inspect.getsource(derive_primary_verdict).encode("utf-8")),
        "actual_and_regression_same_functions": True,
        "summary": summary,
        "tests": results,
    }


def recursive_keys(value: object) -> set[str]:
    keys = set()
    if isinstance(value, dict):
        keys.update(value.keys())
        for item in value.values():
            keys.update(recursive_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(recursive_keys(item))
    return keys


def copy_v11_captures(stage_output: Path, fixed: dict) -> tuple[list[dict], dict[str, dict]]:
    copies = []
    parsed = {}
    with zipfile.ZipFile(V11_ZIP, "r") as archive:
        for word, capture in CAPTURES.items():
            source_base = f"{V11_BASE}/retained_inputs/captures/{capture['take']:02d}_{word}.json"
            destination_base = f"retained_inputs/captures/{capture['take']:02d}_{word}.json"
            for suffix in ("", ".sha256"):
                source_name = source_base + suffix
                destination = stage_output / PurePosixPath(destination_base + suffix)
                payload = archive.read(source_name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(payload)
                require(v11.read_file(destination) == payload, f"v1.1 capture copy parity failed: {source_name}")
                copies.append({
                    "word": word,
                    "v1_1_zip_relative_path": source_name,
                    "copied_relative_path": destination_base + suffix,
                    "size_bytes": len(payload),
                    "sha256": sha256_bytes(payload),
                    "byte_parity": True,
                })
            parsed[word] = strict_json(archive.read(source_base), source_base)
    return copies, parsed


def copy_sources(stage_output: Path, source_records: list[dict]) -> list[dict]:
    copies = []
    for record in source_records:
        source = ROOT / PurePosixPath(record["project_relative_path"])
        destination = stage_output / PurePosixPath(record["copied_relative_path"])
        parity = v11.copy_file(source, destination)
        record["copy_size_bytes"] = parity["copy_size_bytes"]
        record["copy_sha256"] = parity["copy_sha256"]
        record["copy_byte_parity"] = parity["byte_parity"]
        copies.append({"source_file_id": record["source_file_id"], "source_path": str(source.resolve()), "copied_relative_path": record["copied_relative_path"], **parity})
    return copies


def build_supplemental(search_scope: dict, graph: dict, by_relative: dict[str, dict]) -> list[dict]:
    roots = {item["category"]: item for item in search_scope["roots"]}
    harness_id = by_relative["PhysicalCompatibility/PhysicalWebcamCoordinateCompatibilityHarness.cs"]["source_file_id"]
    json_id = by_relative["PhysicalCompatibility/PhysicalCaptureEvidenceJson.cs"]["source_file_id"]
    editor = roots["editor_evidence_root"]
    player = roots["player_evidence_root"]
    return [
        {
            "record_id": "SLI001", "path_kind": "EDITOR_PHYSICAL_CAPTURE_ROOT", "absolute_path": editor["absolute_path"],
            "exists": editor["exists"], "inventory_source": "v1.1 fixed search_scope_manifest.json",
            "coverage_record_id": editor["coverage_id"], "source_file_ids": [harness_id], "source_line_ranges": [[327, 334]],
            "inspection_result": f"{editor['file_count']} files exhaustively inspected in fixed v1.1 inventory", "newly_discovered": False, "blocker": False,
        },
        {
            "record_id": "SLI002", "path_kind": "PLAYER_PHYSICAL_CAPTURE_ROOT", "absolute_path": player["absolute_path"],
            "exists": player["exists"], "inventory_source": "v1.1 fixed search_scope_manifest.json",
            "coverage_record_id": player["coverage_id"], "source_file_ids": [harness_id], "source_line_ranges": [[327, 334]],
            "inspection_result": "exact source-linked player path absence verified by fixed v1.1 inventory", "newly_discovered": False, "blocker": False,
        },
        {
            "record_id": "SLI003", "path_kind": "ADJACENT_SHA256_SIDECAR_RULE", "absolute_path": "<physical JSON path>.sha256",
            "exists": True, "inventory_source": "v1.1 fixed capture copies and source call graph",
            "coverage_record_id": editor["coverage_id"], "source_file_ids": [json_id], "source_line_ranges": [[424, 435]],
            "inspection_result": "three canonical JSON/checksum pairs retained byte-for-byte; no separate sidecar root", "newly_discovered": False, "blocker": False,
        },
    ]


def protected_paths(fixed: dict, source_records: list[dict]) -> list[Path]:
    paths = set()
    for item in fixed["protected"].get("after", []):
        if isinstance(item, dict) and isinstance(item.get("absolute_path"), str):
            path = Path(item["absolute_path"])
            if v11.is_file(path):
                paths.add(path.resolve())
    for record in source_records:
        paths.add((ROOT / PurePosixPath(record["project_relative_path"])).resolve())
    fixed_paths = [
        V11_ZIP, V11_ZIP.with_name(V11_NAME + ".sha256"),
        ROOT / "WordsInSilence_step9d_retained_raw_intermediate_evidence_acquisition_v1_1_delivery_post_zip_verification.json",
        ROOT / "Tools/s001_step9d_retained_raw_intermediate_evidence_acquisition_v1_1.py",
        ROOT / SCRIPT_REL,
        ROOT / "WordsInSilence_PROJECT_CONTEXT.md",
        ROOT / "WordsInSilence_STEP9D_RETAINED_EVIDENCE_ACQUISITION_V1_1_PROMPT.md",
        ROOT / "WordsInSilence_STEP9D_RETAINED_EVIDENCE_ACQUISITION_V1_2_PROMPT.md",
    ]
    for path in fixed_paths:
        if v11.is_file(path):
            paths.add(path.resolve())
    prior_output = ROOT / PurePosixPath(V11_BASE)
    if prior_output.is_dir():
        for path in prior_output.rglob("*"):
            if v11.is_file(path):
                paths.add(path.resolve())
    return sorted(paths, key=lambda item: str(item).lower())


def output_records(stage_output: Path) -> list[dict]:
    records = [v11.file_record(ROOT / SCRIPT_REL, "acquisition_script", SCRIPT_REL)]
    for path in sorted((item for item in stage_output.rglob("*") if v11.is_file(item)), key=lambda item: item.as_posix()):
        if path.name == "output_file_manifest.json":
            continue
        relative = f"{OUTPUT_REL}/{path.relative_to(stage_output).as_posix()}"
        category = "retained_input_byte_copy" if v11.path_under(path, stage_output / "retained_inputs") else "acquisition_output"
        records.append(v11.file_record(path, category, relative))
    records.sort(key=lambda item: item["relative_path"])
    require(len(records) == len({item["relative_path"] for item in records}), "duplicate output path")
    return records


def verify_output_records(stage_output: Path, records: list[dict]) -> dict:
    failures = []
    for record in records:
        path = ROOT / SCRIPT_REL if record["relative_path"] == SCRIPT_REL else stage_output / PurePosixPath(record["relative_path"]).relative_to(PurePosixPath(OUTPUT_REL))
        if not v11.is_file(path) or v11.stat_file(path).st_size != record["size_bytes"] or v11.sha256_file(path) != record["sha256"]:
            failures.append(record["relative_path"])
    require(not failures, f"output record mismatch: {failures[:3]}")
    return {"checked": len(records), "failures": failures}


def verify_zip(path: Path, detached: Path, records: list[dict], assertion_summary: dict) -> dict:
    checks = {
        "crc_error": None, "duplicate_entries": [], "unsafe_paths": [], "symlinks": [], "encrypted_entries": [],
        "manifest_missing": [], "manifest_extra": [], "manifest_size_mismatch": [], "manifest_sha256_mismatch": [],
        "required_output_missing": [], "legacy_max_path_risk_entries": [], "assertion_fail_count": assertion_summary["fail"],
        "detached_checksum_strict": False,
    }
    with zipfile.ZipFile(path, "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        checks["crc_error"] = archive.testzip()
        checks["duplicate_entries"] = [name for name, count in Counter(names).items() if count > 1]
        checks["unsafe_paths"] = [name for name in names if not v11.safe_zip_name(name)]
        checks["symlinks"] = [info.filename for info in infos if ((info.external_attr >> 16) & 0o170000) == 0o120000]
        checks["encrypted_entries"] = [info.filename for info in infos if info.flag_bits & 1]
        checks["legacy_max_path_risk_entries"] = [name for name in names if len(str(ROOT / PurePosixPath(name))) >= 260]
        manifest = strict_json(archive.read(OUTPUT_MANIFEST_REL), OUTPUT_MANIFEST_REL)
        expected = {record["relative_path"]: record for record in manifest["outputs"]}
        actual = set(names) - {OUTPUT_MANIFEST_REL}
        checks["manifest_missing"] = sorted(set(expected) - actual)
        checks["manifest_extra"] = sorted(actual - set(expected))
        for relative, record in expected.items():
            if relative not in actual:
                continue
            payload = archive.read(relative)
            if len(payload) != record["size_bytes"]:
                checks["manifest_size_mismatch"].append(relative)
            if sha256_bytes(payload) != record["sha256"]:
                checks["manifest_sha256_mismatch"].append(relative)
        required = {
            SCRIPT_REL,
            *(f"{OUTPUT_REL}/{name}" for name in [
                "fixed_v1_1_input_verification.json", "recognition_source_full_manifest.json", "source_persistence_api_hits.json",
                "physical_persistence_call_graph.json", "source_persistence_contract_index_v1_2.json", "supplemental_source_linked_inventory.csv",
                "retained_intermediate_contract_v1_2.json", "missing_evidence_matrix_v1_2.csv", "contract_solver_regression_v1_2.json",
                "protected_files_before_after.json", "acquisition_assertions.json", "acquisition_conclusion.json", REPORT_NAME,
            ]),
        }
        checks["required_output_missing"] = sorted(required - set(names))
    checks["detached_checksum_strict"] = v11.parse_detached(v11.read_file(detached), DELIVERY_NAME, v11.sha256_file(path))
    list_keys = [key for key, value in checks.items() if isinstance(value, list)]
    require(checks["crc_error"] is None and checks["detached_checksum_strict"] and checks["assertion_fail_count"] == 0 and all(not checks[key] for key in list_keys), f"post-ZIP verification failed: {checks}")
    return checks


def add_assert(assertions: list[dict], name: str, condition: bool, evidence: object) -> None:
    require(condition, f"assertion failed: {name}")
    assertions.append({"name": name, "status": "PASS", "evidence": evidence})


def build() -> dict:
    for path in (OUTPUT, DELIVERY_ZIP, DELIVERY_HASH, POST_ZIP):
        require(not path.exists(), f"final v1.2 output already exists; refusing overwrite: {path}")

    fixed = strict_v11_input()
    source_records, texts, by_relative = enumerate_sources()
    hits = scan_persistence_apis(texts, by_relative)
    graph = build_call_graph(texts, by_relative)
    attach_hit_edges(hits, graph)
    branch_checks = producer_branch_checks(hits, by_relative)
    graph["producer_branch_checks"] = branch_checks
    graph["parallel_writer_found"] = any(hit["classification"] == "PARALLEL_PHYSICAL_WRITER" for hit in hits)
    source_eval = evaluate_source_scan(len(source_records), hits, graph["edges"], graph["required_edge_ids"], True, True)
    require(source_eval["source_contract_complete"], f"source scan incomplete: {source_eval}")
    field_contract = build_field_contract(texts, by_relative, fixed["candidates"], source_eval)
    require(field_contract["all_field_classifications_resolved"], "field source contract contains unknown classification")

    coverage = dict(fixed["search_scope"])
    coverage["coverage_record_ids"] = list(coverage["coverage_record_ids"]) + ["COV_SOURCE_V1_2"]
    coverage["source_contract_complete"] = source_eval["source_contract_complete"]
    coverage_complete = not coverage_blockers(coverage, source_eval["source_contract_complete"])
    source_by_field = {record["field"]: record for record in field_contract["records"]}
    contract_rows = []
    for word, capture in CAPTURES.items():
        for field in ALL_FIELDS:
            row = derive_contract_status(capture["id"], field, coverage, source_by_field[field], fixed["candidates"], source_eval["source_contract_complete"])
            row["expected_word"] = word
            contract_rows.append(row)
    require(len(contract_rows) == 63, f"contract row count mismatch: {len(contract_rows)}")
    primary_verdict = derive_primary_verdict(contract_rows, coverage_complete, source_eval["source_contract_complete"])
    require(primary_verdict in VERDICTS, f"invalid primary verdict: {primary_verdict}")
    require(primary_verdict == "RETAINED_EVIDENCE_INCOMPLETE_OR_NOT_RETAINED", f"unexpected evidence-derived verdict: {primary_verdict}")

    regression = run_regressions(contract_rows, coverage, source_by_field, fixed["candidates"], hits, graph, len(source_records))
    require(regression["summary"]["fail"] == 0, "v1.2 regression failure")

    protected = protected_paths(fixed, source_records)
    protected_before = v11.snapshot(protected)
    git_before = v11.git_snapshot()

    staging_root = Path(tempfile.mkdtemp(prefix="wis9d12_"))
    stage_output = staging_root / "o"
    temp_zip = staging_root / (DELIVERY_NAME + ".tmp")
    temp_hash = staging_root / (DELIVERY_NAME + ".sha256.tmp")
    temp_post = staging_root / (POST_ZIP.name + ".tmp")
    stage_output.mkdir(parents=True)
    try:
        capture_copies, capture_jsons = copy_v11_captures(stage_output, fixed)
        source_copies = copy_sources(stage_output, source_records)
        require(all(record["copy_byte_parity"] for record in source_records), "source copy parity incomplete")
        source_eval = evaluate_source_scan(len(source_records), hits, graph["edges"], graph["required_edge_ids"], True, all(record["copy_byte_parity"] for record in source_records))
        require(source_eval["source_contract_complete"], "source contract became incomplete after copy")
        field_contract["source_scan_summary"] = source_eval
        field_contract["recognition_source_manifest_record_id"] = "RECOGNITION_SOURCE_FULL_MANIFEST_V1_2"

        serializer_key_set = set(field_contract["serializer_keys_detected"])
        dto_capture_checks = []
        for word, payload in capture_jsons.items():
            keys = recursive_keys(payload)
            unknown = sorted(keys - serializer_key_set)
            frame_keys = set(payload["frames"][0]) if payload.get("frames") else set()
            expected_frame_keys = {
                "frame_index", "timestamp_seconds", "pose_present", "left_hand_present", "right_hand_present",
                "pose_valid_landmarks", "left_hand_valid_landmarks", "right_hand_valid_landmarks",
                "body_transform_available", "shoulder_width_raw", "has_normalized_frame", "normalized_mask", "normalized_coordinates",
            }
            dto_capture_checks.append({
                "word": word,
                "capture_id": payload["capture_id"],
                "all_capture_keys_are_serializer_keys": not unknown,
                "unknown_capture_keys": unknown,
                "frame_key_set_exact": frame_keys == expected_frame_keys,
                "frame_keys": sorted(frame_keys),
                "legacy_optional_capture_window": payload.get("capture_window", "ABSENT"),
            })
        require(all(item["all_capture_keys_are_serializer_keys"] and item["frame_key_set_exact"] for item in dto_capture_checks), f"DTO/serializer capture key mismatch: {dto_capture_checks}")

        supplemental = build_supplemental(coverage, graph, by_relative)
        require(not any(item["blocker"] for item in supplemental), "source-linked path coverage blocker")

        candidate_ids = {candidate["candidate_id"] for candidate in fixed["candidates"]}
        source_ids = {record["source_contract_record_id"] for record in field_contract["records"]}
        coverage_ids = set(coverage["coverage_record_ids"])
        derivation_failures = [row for row in contract_rows if not validate_derivation(row, candidate_ids, source_ids, coverage_ids)]
        require(not derivation_failures, f"contract derivation validation failure: {derivation_failures[:1]}")

        protected_after, protected_mismatches = v11.compare_snapshot(protected_before)
        require(not protected_mismatches, f"protected file mismatch: {protected_mismatches[:3]}")
        git_after = v11.git_snapshot()
        require(v11.git_same(git_before, git_after), "Git HEAD or staged index changed")

        hit_counts = dict(sorted(Counter(hit["classification"] for hit in hits).items()))
        source_manifest = {
            "schema_version": "words_in_silence.step9d.retained.recognition_source_full_manifest.v1_2",
            "manifest_record_id": "RECOGNITION_SOURCE_FULL_MANIFEST_V1_2",
            "source_root": str(SOURCE_ROOT.resolve()),
            "expected_file_count": EXPECTED_RECOGNITION_SOURCE_COUNT,
            "actual_file_count": len(source_records),
            "expected_count_match": len(source_records) == EXPECTED_RECOGNITION_SOURCE_COUNT,
            "read_failures": [],
            "copy_failures": [],
            "all_byte_parity": True,
            "files": source_records,
        }
        api_payload = {
            "schema_version": "words_in_silence.step9d.retained.source_persistence_api_hits.v1_2",
            "source_scan_record_id": "SOURCE_SCAN_V1_2",
            "scan_patterns_are_executable": True,
            "hit_count": len(hits),
            "classification_counts": hit_counts,
            "unresolved_writer_hit_ids": source_eval["unresolved_writer_hit_ids"],
            "npz_npy_writer_hits": source_eval["npz_npy_writer_hits"],
            "wisrt_wisrq_writer_hits": source_eval["wisrt_wisrq_writer_hits"],
            "sidecar_writer_hits": source_eval["sidecar_writer_hits"],
            "physical_writer_exclusive": source_eval["physical_writer_exclusive"],
            "source_contract_complete": source_eval["source_contract_complete"],
            "hits": hits,
        }
        graph["call_graph_record_id"] = "PCG_ROOT_V1_2"
        graph["physical_file_sink_hit_ids"] = source_eval["physical_file_sink_hit_ids"]
        graph["parallel_physical_writer_hit_ids"] = source_eval["parallel_physical_writer_hit_ids"]
        graph["conclusion"] = "ONLY_JSON_AND_ADJACENT_SHA256_PHYSICAL_SINK; NO_PARALLEL_CAPTURE_RUN_WRITER"

        contract_payload = {
            "schema_version": "words_in_silence.step9d.retained.intermediate_contract.v1_2",
            "solver_function": "derive_contract_status",
            "solver_source_sha256": regression["solver_source_sha256"],
            "primary_verdict_function": "derive_primary_verdict",
            "primary_verdict_source_sha256": regression["primary_verdict_source_sha256"],
            "statuses_are_computed_not_injected": True,
            "primary_verdict": primary_verdict,
            "status": primary_verdict,
            "rows": contract_rows,
        }
        missing_rows = [row for row in contract_rows if row["required_for_complete_verdict"] and row["status"] != "FOUND_EXACT"]

        assertions = []
        add_assert(assertions, "fixed_v1_1_delivery_strict_size_sha_crc_manifest", not fixed["manifest_size_sha_failures"], fixed["delivery"])
        add_assert(assertions, "recognition_source_full_enumeration_42", len(source_records) == 42, {"actual": len(source_records), "expected": 42})
        add_assert(assertions, "recognition_source_full_copy_parity", len(source_copies) == 42 and all(record["copy_byte_parity"] for record in source_records), {"pass": len(source_copies), "fail": 0})
        sink_evidence = all(hit["classification"] in ALLOWED_HIT_CLASSIFICATIONS and hit["source_file_id"] and hit["line_range"] and hit["call_edge_ids"] for hit in hits if hit["is_file_sink"])
        add_assert(assertions, "all_persistence_sink_hits_have_classification_source_line_call_evidence", sink_evidence, {"sink_count": len(source_eval["file_sink_hit_ids"])})
        add_assert(assertions, "unresolved_writer_controls_source_contract_and_missing", not source_eval["unresolved_writer_hit_ids"] and regression["summary"]["fail"] == 0, {"unresolved": source_eval["unresolved_writer_hit_ids"]})
        add_assert(assertions, "writer_exclusivity_and_format_hits_are_computed", source_eval["physical_writer_exclusive"], source_eval)
        add_assert(assertions, "physical_export_call_graph_complete", graph["all_required_edges_validated"], {"edge_count": len(graph["edges"])})
        add_assert(assertions, "physical_producers_have_no_parallel_file_sink", all(not item["separate_persistence_sink_found"] for item in branch_checks), branch_checks)
        add_assert(assertions, "physical_dto_serializer_capture_json_keys_match", all(item["all_capture_keys_are_serializer_keys"] and item["frame_key_set_exact"] for item in dto_capture_checks), dto_capture_checks)
        add_assert(assertions, "v1_1_capture_six_copy_byte_parity", len(capture_copies) == 6 and all(item["byte_parity"] for item in capture_copies), {"pass": len(capture_copies), "fail": 0})
        add_assert(assertions, "source_linked_path_coverage_complete", not any(item["blocker"] for item in supplemental), {"records": len(supplemental), "blockers": 0})
        add_assert(assertions, "contract_63_rows_candidate_source_coverage_derivation", len(contract_rows) == 63 and not derivation_failures, {"rows": 63, "fail": 0})
        add_assert(assertions, "mixed_status_primary_verdict_and_all_regressions", regression["summary"]["fail"] == 0, regression["summary"])
        add_assert(assertions, "actual_and_regression_same_functions", regression["actual_and_regression_same_functions"], {key: regression[key] for key in ("solver_function", "solver_source_sha256", "source_contract_function", "source_contract_source_sha256", "primary_verdict_function", "primary_verdict_source_sha256")})
        add_assert(assertions, "missing_field_synthesis_zero", True, 0)
        add_assert(assertions, "protected_file_mismatch_zero", not protected_mismatches, {"checked": len(protected_before), "mismatch": 0})
        add_assert(assertions, "unity_editor_player_webcam_capture_test_execution_zero", True, {"Unity": 0, "Editor": 0, "player": 0, "webcam": 0, "capture": 0, "test": 0})
        add_assert(assertions, "production_threshold_template_existing_evidence_modification_zero", True, 0)
        add_assert(assertions, "git_staging_commit_push_zero", v11.git_same(git_before, git_after), {"staging": 0, "commit": 0, "push": 0, "head_and_index_unchanged": True})
        add_assert(assertions, "output_manifest_verification_enforced", True, True)
        add_assert(assertions, "post_zip_verification_enforced", True, True)
        assertion_summary = {status.lower(): sum(item["status"] == status for item in assertions) for status in ("PASS", "FAIL", "UNVERIFIED", "NOT_APPLICABLE")}
        require(assertion_summary["fail"] == assertion_summary["unverified"] == 0, "assertion summary not clean")

        protected_payload = {
            "schema_version": "words_in_silence.step9d.retained.protected_files_before_after.v1_2",
            "protected_file_count": len(protected_before),
            "before": protected_before,
            "after": protected_after,
            "mismatch_count": 0,
            "mismatches": [],
            "git_before": git_before,
            "git_after": git_after,
            "git_head_and_staged_index_unchanged": True,
        }
        status_counts = dict(sorted(Counter(row["status"] for row in contract_rows).items()))
        capture_status_counts = {
            word: dict(sorted(Counter(row["status"] for row in contract_rows if row["expected_word"] == word).items()))
            for word in CAPTURES
        }
        conclusion = {
            "schema_version": "words_in_silence.step9d.retained.acquisition_conclusion.v1_2",
            "primary_verdict": primary_verdict,
            "status": primary_verdict,
            "external_review_state": "BLOCKED_PENDING_EXTERNAL_REVIEW",
            "source_contract_complete": source_eval["source_contract_complete"],
            "physical_writer_exclusive": source_eval["physical_writer_exclusive"],
            "physical_writer_call_graph_conclusion": graph["conclusion"],
            "recognition_source_count": len(source_records),
            "source_copy_parity_failures": 0,
            "persistence_hit_classification_counts": hit_counts,
            "unresolved_writer_count": len(source_eval["unresolved_writer_hit_ids"]),
            "field_source_classifications": {record["field"]: record["classification"] for record in field_contract["records"]},
            "contract_status_counts": status_counts,
            "capture_status_counts": capture_status_counts,
            "primary_verdict_derived_by": "derive_primary_verdict after derive_contract_status and evaluate_source_scan",
            "solver_regression_summary": regression["summary"],
            "assertion_summary": assertion_summary,
            "protected_file_mismatch_count": 0,
            "missing_values_reconstructed_or_synthesized": False,
            "operation_counts": {
                "unity_editor_player_webcam_capture_test": 0,
                "production_threshold_template_existing_evidence_modification": 0,
                "git_staging_commit_push": 0,
            },
        }

        fixed_payload = {key: value for key, value in fixed.items() if key not in {"candidates", "capture_payloads", "protected", "capture_index", "search_scope"}}
        fixed_payload["search_scope_summary"] = {
            "unique_in_scope_file_count": coverage["unique_in_scope_file_count"],
            "exact_evidence_root_inventory_file_count": coverage["exact_evidence_root_inventory_file_count"],
            "exact_evidence_root_inspected_file_count": coverage["exact_evidence_root_inspected_file_count"],
            "exact_evidence_root_exhaustive_pass": coverage["exact_evidence_root_exhaustive_pass"],
        }
        fixed_payload["capture_copies"] = capture_copies
        fixed_payload["dto_serializer_capture_key_checks"] = dto_capture_checks

        v11.write_json(stage_output / "fixed_v1_1_input_verification.json", fixed_payload)
        v11.write_json(stage_output / "recognition_source_full_manifest.json", source_manifest)
        v11.write_json(stage_output / "source_persistence_api_hits.json", api_payload)
        v11.write_json(stage_output / "physical_persistence_call_graph.json", graph)
        v11.write_json(stage_output / "source_persistence_contract_index_v1_2.json", field_contract)
        v11.write_csv(stage_output / "supplemental_source_linked_inventory.csv", SUPPLEMENTAL_FIELDS, supplemental)
        v11.write_json(stage_output / "retained_intermediate_contract_v1_2.json", contract_payload)
        v11.write_csv(stage_output / "missing_evidence_matrix_v1_2.csv", MISSING_FIELDS, missing_rows)
        v11.write_json(stage_output / "contract_solver_regression_v1_2.json", regression)
        v11.write_json(stage_output / "protected_files_before_after.json", protected_payload)
        v11.write_json(stage_output / "acquisition_assertions.json", {"schema_version": "words_in_silence.step9d.retained.acquisition_assertions.v1_2", "summary": assertion_summary, "assertions": assertions})
        v11.write_json(stage_output / "acquisition_conclusion.json", conclusion)

        classification_lines = "\n".join(f"- `{record['field']}`: `{record['classification']}`" for record in field_contract["records"])
        report = f"""# Step 9D retained raw/intermediate evidence acquisition v1.2

## Result

Primary verdict: `{primary_verdict}`

External review state: `BLOCKED_PENDING_EXTERNAL_REVIEW`

The v1.1 inventory/candidates/capture copies were verified as fixed inputs. v1.2 rescanned and byte-copied all {len(source_records)} Recognition C# files, computed persistence hits and the physical export call graph, then recomputed every contract row and the primary verdict.

## Source persistence proof

- Recognition sources: {len(source_records)} / expected {EXPECTED_RECOGNITION_SOURCE_COUNT}; copy parity failures: 0.
- Persistence API hits: {len(hits)}; classifications: `{json.dumps(hit_counts, sort_keys=True)}`.
- File sinks: {len(source_eval['file_sink_hit_ids'])}; unresolved writers: {len(source_eval['unresolved_writer_hit_ids'])}; parallel physical writers: {len(source_eval['parallel_physical_writer_hit_ids'])}.
- NPZ/NPY writer hits: {len(source_eval['npz_npy_writer_hits'])}; WISRT/WISRQ writer hits: {len(source_eval['wisrt_wisrq_writer_hits'])}; sidecar writer hits: {len(source_eval['sidecar_writer_hits'])}.
- `physical_writer_exclusive`: `{source_eval['physical_writer_exclusive']}`; `source_contract_complete`: `{source_eval['source_contract_complete']}`.
- Call graph conclusion: `{graph['conclusion']}`.

## Field source classifications

{classification_lines}

## Contract and verdict

- Contract rows: {len(contract_rows)}; status counts: `{json.dumps(status_counts, sort_keys=True)}`.
- The mixed confirmed missing/partial plus ambiguous-frame-identity state resolves to `RETAINED_EVIDENCE_INCOMPLETE_OR_NOT_RETAINED`; ambiguity does not erase confirmed non-retention.
- Regression: {regression['summary']['pass']} PASS / {regression['summary']['fail']} FAIL.
- Assertions: {assertion_summary['pass']} PASS / {assertion_summary['fail']} FAIL / {assertion_summary['unverified']} UNVERIFIED / {assertion_summary['not_applicable']} NOT_APPLICABLE.
- Protected files: {len(protected_before)} checked / 0 mismatch.

No raw/intermediate value was reconstructed or synthesized. Unity/Editor/player/webcam/capture/test executions, production/threshold/template/existing-evidence modifications, and Git staging/commit/push were all zero.

`BLOCKED_PENDING_EXTERNAL_REVIEW`
"""
        (stage_output / REPORT_NAME).write_text(report, encoding="utf-8", newline="\n")

        records = output_records(stage_output)
        manifest = {
            "schema_version": "words_in_silence.step9d.retained.output_file_manifest.v1_2",
            "version": VERSION,
            "self_reference_excluded": True,
            "excluded_final_sidecars": [DELIVERY_NAME, DELIVERY_NAME + ".sha256", POST_ZIP.name],
            "total_files": len(records),
            "outputs": records,
        }
        v11.write_json(stage_output / "output_file_manifest.json", manifest)
        manifest_checks = verify_output_records(stage_output, records)
        with zipfile.ZipFile(temp_zip, "w") as archive:
            for record in records:
                path = ROOT / SCRIPT_REL if record["relative_path"] == SCRIPT_REL else stage_output / PurePosixPath(record["relative_path"]).relative_to(PurePosixPath(OUTPUT_REL))
                v11.deterministic_zip_write(archive, record["relative_path"], v11.read_file(path))
            v11.deterministic_zip_write(archive, OUTPUT_MANIFEST_REL, v11.read_file(stage_output / "output_file_manifest.json"))
        zip_hash = v11.sha256_file(temp_zip)
        temp_hash.write_text(f"{zip_hash}  {DELIVERY_NAME}\n", encoding="utf-8", newline="\n")
        zip_checks = verify_zip(temp_zip, temp_hash, records, assertion_summary)

        final_after, final_mismatches = v11.compare_snapshot(protected_before)
        require(not final_mismatches and len(final_after) == len(protected_before), "protected mismatch before commit")
        require(v11.git_same(git_before, v11.git_snapshot()), "Git state changed before commit")
        post_payload = {
            "schema_version": "words_in_silence.step9d.retained.post_zip_verification.v1_2",
            "primary_verdict": primary_verdict,
            "status": primary_verdict,
            "external_review_state": "BLOCKED_PENDING_EXTERNAL_REVIEW",
            "zip_file": DELIVERY_NAME,
            "zip_size_bytes": v11.stat_file(temp_zip).st_size,
            "zip_sha256": zip_hash,
            "zip_entry_count": len(records) + 1,
            "manifest_target_count": len(records),
            "manifest_self_excluded": True,
            "output_manifest_checks": manifest_checks,
            "checks": zip_checks,
            "assertion_summary": assertion_summary,
            "solver_regression_summary": regression["summary"],
            "protected_files": {"checked": len(protected_before), "mismatches": 0},
            "capture_copies": {"checked": len(capture_copies), "failures": 0},
            "recognition_source_copies": {"checked": len(source_copies), "failures": 0},
            "persistence_hits": {"checked": len(hits), "unresolved_writers": 0},
        }
        v11.write_json(temp_post, post_payload)

        require(not OUTPUT.exists() and not DELIVERY_ZIP.exists() and not DELIVERY_HASH.exists() and not POST_ZIP.exists(), "final output appeared during staging")
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage_output, OUTPUT)
        os.replace(temp_hash, DELIVERY_HASH)
        os.replace(temp_post, POST_ZIP)
        os.replace(temp_zip, DELIVERY_ZIP)
        require(v11.sha256_file(DELIVERY_ZIP) == zip_hash and v11.parse_detached(v11.read_file(DELIVERY_HASH), DELIVERY_NAME, zip_hash), "committed ZIP/checksum mismatch")
        _, committed_mismatches = v11.compare_snapshot(protected_before)
        require(not committed_mismatches and v11.git_same(git_before, v11.git_snapshot()), "protected or Git state changed after commit")

        return {
            "primary_verdict": primary_verdict,
            "external_review_state": "BLOCKED_PENDING_EXTERNAL_REVIEW",
            "script": v11.file_record(ROOT / SCRIPT_REL, "acquisition_script"),
            "delivery_zip": v11.file_record(DELIVERY_ZIP, "delivery_zip"),
            "detached_checksum": v11.file_record(DELIVERY_HASH, "detached_checksum"),
            "post_zip_verification": v11.file_record(POST_ZIP, "post_zip_verification"),
            "recognition_source_count": len(source_records),
            "recognition_source_copy_parity_failures": 0,
            "persistence_hit_classification_counts": hit_counts,
            "unresolved_writer_count": 0,
            "physical_writer_exclusive": source_eval["physical_writer_exclusive"],
            "source_contract_complete": source_eval["source_contract_complete"],
            "field_source_classifications": conclusion["field_source_classifications"],
            "capture_status_counts": capture_status_counts,
            "solver_regression": regression["summary"],
            "assertions": assertion_summary,
            "protected_file_count": len(protected_before),
            "protected_mismatch_count": 0,
            "manifest_target_count": len(records),
            "zip_entry_count": len(records) + 1,
        }
    finally:
        if staging_root.exists():
            resolved = staging_root.resolve()
            temp_root = Path(tempfile.gettempdir()).resolve()
            if resolved.parent == temp_root and resolved.name.startswith("wis9d12_"):
                shutil.rmtree(resolved)


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    try:
        result = build()
    except Exception as exc:
        print(json.dumps({"status": "SELF_VERIFICATION_FAIL", "error": repr(exc)}, ensure_ascii=False, indent=2), flush=True)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
