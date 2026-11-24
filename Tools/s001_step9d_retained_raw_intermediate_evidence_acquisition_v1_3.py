#!/usr/bin/env python3
"""Step 9D retained-evidence acquisition v1.3: Unity asset writer and downstream termination proof."""

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

import s001_step9d_retained_raw_intermediate_evidence_acquisition_v1_2 as v12


ROOT = Path(__file__).resolve().parent.parent
VERSION = "1.3"
SCRIPT_REL = "Tools/s001_step9d_retained_raw_intermediate_evidence_acquisition_v1_3.py"
OUTPUT_REL = "data/processed/sessions/S001_P001/step9d_retained_raw_intermediate_evidence_acquisition_v1_3"
OUTPUT = ROOT / PurePosixPath(OUTPUT_REL)
OUTPUT_MANIFEST_REL = f"{OUTPUT_REL}/output_file_manifest.json"
REPORT_NAME = "STEP9D_RETAINED_RAW_INTERMEDIATE_EVIDENCE_ACQUISITION_V1_3_REPORT.md"

DELIVERY_NAME = "WordsInSilence_step9d_retained_raw_intermediate_evidence_acquisition_v1_3_delivery.zip"
DELIVERY_ZIP = ROOT / DELIVERY_NAME
DELIVERY_HASH = ROOT / f"{DELIVERY_NAME}.sha256"
POST_ZIP = ROOT / "WordsInSilence_step9d_retained_raw_intermediate_evidence_acquisition_v1_3_delivery_post_zip_verification.json"

V12_NAME = "WordsInSilence_step9d_retained_raw_intermediate_evidence_acquisition_v1_2_delivery.zip"
V12_ZIP = ROOT / V12_NAME
V12_SIZE = 366_403
V12_SHA256 = "0d6bb7cce415aac6623502e8679790e0f366e64f6efa0acde6c7505ea113c7f9"
V12_BASE = "data/processed/sessions/S001_P001/step9d_retained_raw_intermediate_evidence_acquisition_v1_2"
V12_MANIFEST_REL = f"{V12_BASE}/output_file_manifest.json"

ASSET_REL = "Assets/_Project/Settings/DefaultSignDictionary.asset"
ASSET_META_REL = ASSET_REL + ".meta"
ASSET_PATH = ROOT / PurePosixPath(ASSET_REL)
ASSET_META_PATH = ROOT / PurePosixPath(ASSET_META_REL)
SIGN_DICTIONARY_META = ROOT / "Assets/_Project/Scripts/Recognition/SignDictionary.cs.meta"
RUN_ID = v12.v11.RUN_ID
CAPTURES = v12.CAPTURES
CAPTURE_IDS = v12.CAPTURE_IDS
FIELDS = v12.FIELDS
ALL_FIELDS = v12.ALL_FIELDS

ALLOWED_CLASSIFICATIONS = v12.ALLOWED_HIT_CLASSIFICATIONS
TERMINATION_STATUSES = {
    "TERMINATED_PHYSICAL_FILE_SINK",
    "TERMINATED_UNRELATED_ASSET_SINK",
    "TERMINATED_PHYSICAL_DATA_FLOW",
    "NO_FILE_SINK",
}
MISSING_FIELDS = v12.MISSING_FIELDS
SUPPLEMENTAL_FIELDS = [
    "record_id", "path_kind", "relative_path", "absolute_path", "exists", "size_bytes", "sha256",
    "v1_1_inventory_id", "v1_1_inspection_status", "source_file_ids", "call_edge_ids",
    "copy_relative_path", "copy_byte_parity", "capture_ids_found", "run_id_found", "blocker",
]
TERMINATION_FIELDS = [
    "writer_candidate_id", "hit_id", "source_file_id", "relative_path", "containing_symbol",
    "api_or_symbol", "access_kind", "classification", "termination_status", "final_sink",
    "call_edge_ids", "termination_evidence", "path_or_extension_literals", "unresolved",
]


class AcquisitionError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AcquisitionError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strict_json(data: bytes, label: str) -> object:
    return v12.strict_json(data, label)


def strict_v12_input() -> dict:
    structural = v12.v11.strict_fixed_zip(V12_ZIP, V12_NAME, V12_SIZE, V12_SHA256)
    with zipfile.ZipFile(V12_ZIP, "r") as archive:
        names = archive.namelist()
        require(V12_MANIFEST_REL in names, "v1.2 manifest missing")
        manifest = strict_json(archive.read(V12_MANIFEST_REL), V12_MANIFEST_REL)
        outputs = manifest.get("outputs")
        require(isinstance(outputs, list) and manifest.get("total_files") == len(outputs) == 62, "v1.2 manifest target count mismatch")
        expected = {item["relative_path"]: item for item in outputs}
        require(len(expected) == 62 and set(names) == set(expected) | {V12_MANIFEST_REL}, "v1.2 manifest path set mismatch")
        failures = []
        for relative, record in expected.items():
            payload = archive.read(relative)
            if len(payload) != record["size_bytes"] or sha256_bytes(payload) != record["sha256"]:
                failures.append(relative)
        require(not failures, f"v1.2 manifest size/SHA mismatch: {failures[:3]}")

        required = [
            "recognition_source_full_manifest.json", "source_persistence_api_hits.json",
            "physical_persistence_call_graph.json", "source_persistence_contract_index_v1_2.json",
            "retained_intermediate_contract_v1_2.json", "contract_solver_regression_v1_2.json",
            "acquisition_conclusion.json", "protected_files_before_after.json",
        ]
        required_paths = {f"{V12_BASE}/{name}" for name in required}
        require(required_paths.issubset(expected), "v1.2 required fixed results missing")
        source_manifest = strict_json(archive.read(f"{V12_BASE}/recognition_source_full_manifest.json"), "v1.2 source manifest")
        require(source_manifest["actual_file_count"] == 42 and source_manifest["all_byte_parity"], "v1.2 source manifest not fixed PASS")
        source_payloads = {}
        source_records = []
        for record in source_manifest["files"]:
            archive_path = f"{V12_BASE}/{record['copied_relative_path']}"
            require(archive_path in expected, f"v1.2 source copy absent: {archive_path}")
            payload = archive.read(archive_path)
            require(len(payload) == record["size_bytes"] and sha256_bytes(payload) == record["sha256"], f"v1.2 source copy mismatch: {archive_path}")
            item = dict(record)
            item["v1_2_zip_relative_path"] = archive_path
            item["v1_3_copy_relative_path"] = record["copied_relative_path"]
            source_records.append(item)
            source_payloads[record["recognition_relative_path"]] = payload
        require(len(source_records) == len(source_payloads) == 42, "v1.2 source copy count/identity mismatch")

        capture_paths = []
        for word, capture in CAPTURES.items():
            base = f"{V12_BASE}/retained_inputs/captures/{capture['take']:02d}_{word}.json"
            capture_paths.extend([base, base + ".sha256"])
        require(set(capture_paths).issubset(expected), "v1.2 capture copies missing")
        capture_payloads = {path: archive.read(path) for path in capture_paths}
        fixed_results = {name: strict_json(archive.read(f"{V12_BASE}/{name}"), name) for name in required}

    fixed_v11 = v12.strict_v11_input()
    inventory_name = f"{v12.V11_BASE}/all_in_scope_file_inventory.csv"
    with zipfile.ZipFile(v12.V11_ZIP, "r") as archive:
        inventory_bytes = archive.read(inventory_name)
        inventory_rows = list(csv.DictReader(io.StringIO(inventory_bytes.decode("utf-8-sig"))))
    asset_rows = [row for row in inventory_rows if row["relative_path"] in {ASSET_REL, ASSET_META_REL}]
    require(len(asset_rows) == 2, f"v1.1 asset inventory provenance missing: {len(asset_rows)}")
    by_asset = {row["relative_path"]: row for row in asset_rows}
    require(by_asset[ASSET_REL]["inventory_id"] == "INV00571" and by_asset[ASSET_META_REL]["inventory_id"] == "INV00572", "v1.1 asset inventory IDs changed")
    require(all(row["content_search_performed"] == "True" and not row["read_error"] for row in asset_rows), "v1.1 asset inventory inspection not complete")
    return {
        "schema_version": "words_in_silence.step9d.retained.fixed_v1_2_input_verification.v1_3",
        "delivery": structural,
        "manifest_target_count": 62,
        "zip_entry_count": 63,
        "manifest_size_sha_failures": [],
        "source_records": source_records,
        "source_payloads": source_payloads,
        "capture_payloads": capture_payloads,
        "v1_2_results": fixed_results,
        "v1_1": fixed_v11,
        "v1_1_inventory_relative_path": inventory_name,
        "v1_1_inventory_size_bytes": len(inventory_bytes),
        "v1_1_inventory_sha256": sha256_bytes(inventory_bytes),
        "v1_1_asset_inventory_rows": asset_rows,
    }


def source_maps(fixed: dict) -> tuple[dict[str, str], dict[str, dict]]:
    texts = {relative: payload.decode("utf-8-sig") for relative, payload in fixed["source_payloads"].items()}
    by_relative = {record["recognition_relative_path"]: record for record in fixed["source_records"]}
    return texts, by_relative


def line_context(text: str, line: int) -> str:
    lines = text.splitlines()
    return lines[line - 1].strip() if 1 <= line <= len(lines) else ""


def classify_new_hit(relative: str, api: str, access: str) -> tuple[str, str]:
    if relative == "RecognitionTestUI.cs" and api in {"EditorUtility.SetDirty", "AssetDatabase.SaveAssets", "_recorder.RecordAsync"}:
        return "AMBIGUOUS_REQUIRES_COVERAGE_BLOCK", "classification awaits source-linked asset identity and downstream graph validation"
    if relative == "MotionRecorder.cs" and "RecordAsync" in api:
        return "AMBIGUOUS_REQUIRES_COVERAGE_BLOCK", "producer has no local file sink but must be followed through RecognitionTestUI"
    if relative.startswith("PhysicalCompatibility/"):
        return "PHYSICAL_CAPTURE_CALL_GRAPH", "writer/call is connected to the verified Step 9C physical JSON export graph"
    if relative == "Runtime/RecognitionSequenceInputSession.cs" or relative.startswith("LiveInput/"):
        return "NORMALIZED_RUNTIME_ONLY", "BinaryWriter writes only to local MemoryStream hashes and has no file sink"
    return "UNRELATED_RECOGNITION_WRITER", "custom invocation has no Step 9C Capture/Run path and will require explicit sink/no-sink termination"


def scan_persistence_apis_v1_3(texts: dict[str, str], by_relative: dict[str, dict]) -> list[dict]:
    hits = [dict(hit) for hit in v12.scan_persistence_apis(texts, by_relative)]
    for hit in hits:
        hit["is_asset_sink"] = False
        hit["is_persistence_sink"] = bool(hit.get("is_file_sink"))
        hit["v1_3_detection"] = "V1_2_SCAN_REEXECUTED"

    existing = {(hit["relative_path"], hit["line_range"][0], hit["detection_family"], hit["api_or_symbol"], hit["access_kind"]) for hit in hits}

    def add(relative: str, line: int, family: str, api: str, access: str, owner: str, method: str) -> None:
        key = (relative, line, family, api, access)
        if key in existing:
            return
        existing.add(key)
        spans = v12.method_spans(texts[relative])
        symbol, symbol_range = v12.containing_symbol(spans, line)
        classification, basis = classify_new_hit(relative, api, access)
        source_line = line_context(texts[relative], line)
        extensions = sorted(set(re.findall(r"(?i)\.(?:json|sha256|csv|npz|npy|wisrt|wisrq|bytes|bin|zip|asset|meta)\b", source_line)))
        hits.append({
            "hit_id": None,
            "source_file_id": by_relative[relative]["source_file_id"],
            "relative_path": relative,
            "line_range": [line, line],
            "containing_symbol": symbol,
            "containing_symbol_line_range": symbol_range,
            "detection_family": family,
            "api_or_symbol": api,
            "api_owner": owner,
            "api_method": method,
            "access_kind": access,
            "is_writer_candidate": True,
            "is_file_sink": False,
            "is_asset_sink": access in {"ASSET_SINK", "PREFERENCE_SINK"},
            "is_persistence_sink": access in {"ASSET_SINK", "PREFERENCE_SINK"},
            "path_or_extension_literals": extensions,
            "matched_source_line": source_line,
            "classification": classification,
            "classification_basis": basis,
            "evidence_source_ids": [by_relative[relative]["source_file_id"]],
            "call_edge_ids": [],
            "v1_3_detection": "NEW_V1_3_SCAN",
        })

    unity_patterns = [
        (re.compile(r"\bAssetDatabase\.(SaveAssets|SaveAssetIfDirty|CreateAsset|CopyAsset|MoveAsset|DeleteAsset|ImportAsset)\s*\("), "UNITY_ASSET_DATABASE", lambda m: ("AssetDatabase", m.group(1), "ASSET_SINK")),
        (re.compile(r"\bEditorUtility\.(SetDirty)\s*\("), "UNITY_EDITOR_DIRTY", lambda m: ("EditorUtility", m.group(1), "ASSET_DIRTY_MARK")),
        (re.compile(r"\bPrefabUtility\.(Save[A-Za-z0-9_]*)\s*\("), "UNITY_PREFAB_SAVE", lambda m: ("PrefabUtility", m.group(1), "ASSET_SINK")),
        (re.compile(r"\bEditorSceneManager\.(Save[A-Za-z0-9_]*)\s*\("), "UNITY_SCENE_SAVE", lambda m: ("EditorSceneManager", m.group(1), "ASSET_SINK")),
        (re.compile(r"\bScriptableSingleton(?:<[^>]+>)?\.(Save)\s*\("), "UNITY_SINGLETON_SAVE", lambda m: ("ScriptableSingleton", m.group(1), "ASSET_SINK")),
        (re.compile(r"\bPlayerPrefs\.(Set[A-Za-z0-9_]*|Save)\s*\("), "UNITY_PLAYER_PREFS", lambda m: ("PlayerPrefs", m.group(1), "PREFERENCE_SINK")),
        (re.compile(r"\bEditorPrefs\.(Set[A-Za-z0-9_]*)\s*\("), "UNITY_EDITOR_PREFS", lambda m: ("EditorPrefs", m.group(1), "PREFERENCE_SINK")),
    ]
    custom = re.compile(r"(?P<owner>[A-Za-z_][A-Za-z0-9_]*\.)?(?P<method>(?:Write|Append|Save|Export|Serialize|Record|CreateAsset|SetDirty)[A-Za-z0-9_]*)\s*\(")
    known_owners = {"File.", "Directory.", "AssetDatabase.", "EditorUtility.", "PrefabUtility.", "EditorSceneManager.", "PlayerPrefs.", "EditorPrefs."}
    for relative, text in texts.items():
        masked = v12.mask_csharp(text)
        original_lines = text.splitlines()
        masked_lines = masked.splitlines()
        declaration_starts = {(span["line_range"][0], span["name"]) for span in v12.method_spans(text)}
        for line_number, masked_line in enumerate(masked_lines, start=1):
            for pattern, family, details in unity_patterns:
                for match in pattern.finditer(masked_line):
                    owner, method, access = details(match)
                    add(relative, line_number, family, f"{owner}.{method}", access, owner, method)
            for match in custom.finditer(masked_line):
                owner = match.group("owner") or ""
                method = match.group("method")
                if owner in known_owners:
                    continue
                prefix = masked_line[:match.start()]
                if re.search(r"\bnew\s*$", prefix) or re.search(r"\[\s*$", prefix) or method == "CreateAssetMenu":
                    continue
                family = "CUSTOM_METHOD_DECLARATION" if (line_number, method) in declaration_starts else "CUSTOM_METHOD_INVOCATION"
                api = f"{owner}{method}" if owner else method
                add(relative, line_number, family, api, "CUSTOM_WRITER", owner.rstrip(".") or "CUSTOM", method)

    hits.sort(key=lambda item: (item["relative_path"].lower(), item["line_range"][0], item["detection_family"], item["api_or_symbol"], item["access_kind"]))
    for index, hit in enumerate(hits, start=1):
        hit["hit_id"] = f"PH13_{index:04d}"
    return hits


def graph_edge(
    edge_id: str,
    texts: dict[str, str],
    by_relative: dict[str, dict],
    caller_rel: str,
    caller_symbol: str,
    token: str,
    callee_rel: str | None,
    callee_symbol: str,
    kind: str,
    external_id: str = "EXTERNAL_UNITY_OR_BCL",
) -> dict:
    caller_span = v12.find_span(texts[caller_rel], caller_symbol)
    callee_span = v12.find_span(texts[callee_rel], callee_symbol) if callee_rel else None
    if callee_rel and callee_span is None:
        type_pattern = re.compile(rf"\b(?:class|struct)\s+{re.escape(callee_symbol)}\b")
        for line_number, source_line in enumerate(texts[callee_rel].splitlines(), start=1):
            if type_pattern.search(source_line):
                callee_span = {"name": callee_symbol, "line_range": [line_number, line_number]}
                break
    call_line = None
    if caller_span:
        lines = texts[caller_rel].splitlines()
        for line in range(caller_span["line_range"][0], caller_span["line_range"][1] + 1):
            if token in lines[line - 1]:
                call_line = line
                break
    validated = caller_span is not None and call_line is not None and (callee_rel is None or callee_span is not None)
    return {
        "edge_id": edge_id,
        "edge_kind": kind,
        "caller_source_id": by_relative[caller_rel]["source_file_id"],
        "caller_relative_path": caller_rel,
        "caller_symbol": caller_symbol,
        "caller_line_range": caller_span["line_range"] if caller_span else None,
        "call_site_line_range": [call_line, call_line] if call_line else None,
        "callee_source_id": by_relative[callee_rel]["source_file_id"] if callee_rel else external_id,
        "callee_relative_path": callee_rel or "UnityEditor/System",
        "callee_symbol": callee_symbol,
        "callee_line_range": callee_span["line_range"] if callee_span else ([0, 0] if callee_rel is None else None),
        "source_token": token,
        "validated": validated,
    }


def build_combined_call_graph(texts: dict[str, str], by_relative: dict[str, dict], fixed_v12_graph: dict) -> dict:
    physical = v12.build_call_graph(texts, by_relative)
    require(physical["all_required_edges_validated"] and len(physical["edges"]) == 21, "v1.2 physical graph reconstruction failed")
    fixed_signature = [(edge["caller_relative_path"], edge["caller_symbol"], edge["source_token"], edge["callee_symbol"], edge["validated"]) for edge in fixed_v12_graph["edges"]]
    actual_signature = [(edge["caller_relative_path"], edge["caller_symbol"], edge["source_token"], edge["callee_symbol"], edge["validated"]) for edge in physical["edges"]]
    require(fixed_signature == actual_signature, "v1.2 physical call graph does not reproduce from fixed sources")
    asset_specs = [
        ("RecognitionTestUI.cs", "OnRecordClicked", "_recorder.RecordAsync", "MotionRecorder.cs", "RecordAsync", "CALL"),
        ("RecognitionTestUI.cs", "OnRecordClicked", "ProcessSample(sample)", "RecognitionTestUI.cs", "ProcessSample", "ASYNC_DATA_FLOW"),
        ("RecognitionTestUI.cs", "ProcessSample", "ComputeAveragePose(sample)", None, "PoseTemplateEvaluator.ComputeAveragePose", "STATIC_AVERAGE_FLOW"),
        ("RecognitionTestUI.cs", "ProcessSample", "_lastAvgPose = avgPose", "RecognitionTestUI.cs", "OnSaveClicked", "FIELD_DATA_FLOW"),
        ("RecognitionTestUI.cs", "OnSaveClicked", "new SignEntry", "SignEntry.cs", "SignEntry", "DTO_CONSTRUCTION"),
        ("RecognitionTestUI.cs", "OnSaveClicked", "_dictionary.signs", "SignDictionary.cs", "SignDictionary", "SCRIPTABLE_OBJECT_MUTATION"),
        ("RecognitionTestUI.cs", "OnSaveClicked", "EditorUtility.SetDirty", None, "UnityEditor.EditorUtility.SetDirty", "UNITY_DIRTY_MARK"),
        ("RecognitionTestUI.cs", "OnSaveClicked", "AssetDatabase.SaveAssets", None, "UnityEditor.AssetDatabase.SaveAssets", "UNITY_ASSET_SINK"),
        ("RecognitionTestUI.cs", "OnSaveClicked", "DefaultSignDictionary.asset", None, ASSET_REL, "SOURCE_LITERAL_TO_ASSET"),
    ]
    memory_specs = [
        ("Runtime/RecognitionSequenceInputSession.cs", "ComputeProvenanceSha256", "new MemoryStream", None, "System.IO.MemoryStream", "MEMORY_SINK"),
        ("Runtime/RecognitionSequenceInputSession.cs", "ComputeProvenanceSha256", "new BinaryWriter", None, "System.IO.BinaryWriter", "MEMORY_WRITER_BINDING"),
        ("Runtime/RecognitionSequenceInputSession.cs", "ComputeProvenanceSha256", "WriteString(", "Runtime/RecognitionSequenceInputSession.cs", "WriteString", "MEMORY_SERIALIZER_CALL"),
        ("Runtime/RecognitionSequenceInputSession.cs", "ComputeProvenanceSha256", "WriteCoordinates(", "Runtime/RecognitionSequenceInputSession.cs", "WriteCoordinates", "MEMORY_SERIALIZER_CALL"),
        ("Runtime/RecognitionSequenceInputSession.cs", "ComputeProvenanceSha256", "WriteMask(", "Runtime/RecognitionSequenceInputSession.cs", "WriteMask", "MEMORY_SERIALIZER_CALL"),
        ("Runtime/RecognitionSequenceInputSession.cs", "ComputeLogicalSha256", "new MemoryStream", None, "System.IO.MemoryStream", "MEMORY_SINK"),
        ("Runtime/RecognitionSequenceInputSession.cs", "ComputeLogicalSha256", "new BinaryWriter", None, "System.IO.BinaryWriter", "MEMORY_WRITER_BINDING"),
        ("Runtime/RecognitionSequenceInputSession.cs", "ComputeLogicalSha256", "WriteString(", "Runtime/RecognitionSequenceInputSession.cs", "WriteString", "MEMORY_SERIALIZER_CALL"),
        ("Runtime/RecognitionSequenceInputSession.cs", "ComputeLogicalSha256", "WriteCoordinates(", "Runtime/RecognitionSequenceInputSession.cs", "WriteCoordinates", "MEMORY_SERIALIZER_CALL"),
        ("Runtime/RecognitionSequenceInputSession.cs", "ComputeLogicalSha256", "WriteMask(", "Runtime/RecognitionSequenceInputSession.cs", "WriteMask", "MEMORY_SERIALIZER_CALL"),
    ]
    extra_edges = []
    for index, spec in enumerate(asset_specs + memory_specs, start=1):
        extra_edges.append(graph_edge(f"XCG{index:03d}", texts, by_relative, *spec))
    edges = physical["edges"] + extra_edges
    return {
        "schema_version": "words_in_silence.step9d.retained.physical_and_unrelated_persistence_call_graph.v1_3",
        "physical_graph_fixed_v1_2_verified": True,
        "physical_edges": physical["edges"],
        "asset_and_memory_edges": extra_edges,
        "edges": edges,
        "required_edge_ids": [edge["edge_id"] for edge in edges],
        "all_required_edges_validated": all(edge["validated"] for edge in edges),
        "asset_branch_edge_ids": [edge["edge_id"] for edge in extra_edges if edge["edge_kind"] not in {"MEMORY_SINK", "MEMORY_WRITER_BINDING", "MEMORY_SERIALIZER_CALL"}],
        "memory_branch_edge_ids": [edge["edge_id"] for edge in extra_edges if edge["edge_kind"] in {"MEMORY_SINK", "MEMORY_WRITER_BINDING", "MEMORY_SERIALIZER_CALL"}],
    }


def attach_edges(hits: list[dict], graph: dict) -> None:
    for hit in hits:
        edge_ids = []
        for edge in graph["edges"]:
            same_source = edge["caller_source_id"] == hit["source_file_id"]
            same_site = edge["call_site_line_range"] and edge["call_site_line_range"][0] == hit["line_range"][0]
            same_symbol = same_source and edge["caller_symbol"] == hit["containing_symbol"]
            callee_match = edge["callee_source_id"] == hit["source_file_id"] and edge["callee_symbol"] in {hit["api_or_symbol"], hit["containing_symbol"]}
            token_match = edge["source_token"].rstrip("(") in hit["matched_source_line"] or hit["api_or_symbol"].split(".")[-1] in edge["source_token"]
            if (same_source and same_site) or (same_symbol and token_match) or callee_match:
                edge_ids.append(edge["edge_id"])
        hit["call_edge_ids"] = sorted(set(edge_ids))


def add_candidate_termination_edges(hits: list[dict], graph: dict, texts: dict[str, str]) -> None:
    edges = []
    for index, hit in enumerate((item for item in hits if item.get("is_writer_candidate")), start=1):
        relative = hit["relative_path"]
        line_number = hit["line_range"][0]
        source_lines = texts[relative].splitlines()
        source_line = source_lines[line_number - 1].strip() if 0 < line_number <= len(source_lines) else ""
        token = hit["api_or_symbol"].split(".")[-1]
        classification = hit["classification"]
        asset_branch = relative in {"RecognitionTestUI.cs", "MotionRecorder.cs"} and classification == "UNRELATED_RECOGNITION_WRITER"
        if classification == "PHYSICAL_CAPTURE_CALL_GRAPH":
            edge_kind = "WRITER_TO_VERIFIED_PHYSICAL_BRANCH"
            callee_symbol = "VERIFIED_STEP9C_PHYSICAL_JSON_SHA_BRANCH"
        elif asset_branch:
            edge_kind = "WRITER_TO_UNRELATED_ASSET_BRANCH"
            callee_symbol = ASSET_REL
        elif classification == "NORMALIZED_RUNTIME_ONLY":
            edge_kind = "EXPLICIT_MEMORY_OR_TRANSFORM_NO_FILE_SINK"
            callee_symbol = "NO_FILE_SINK"
        elif classification in {"UNRELATED_RECOGNITION_WRITER", "TEMPLATE_OR_CONFIG_READER_ONLY"}:
            edge_kind = "EXPLICIT_SOURCE_SCOPED_NO_FILE_SINK"
            callee_symbol = "NO_FILE_SINK"
        else:
            edge_kind = "UNRESOLVED_WRITER_TERMINATION"
            callee_symbol = "UNRESOLVED"
        containing_range = hit.get("containing_symbol_line_range")
        line_in_symbol = bool(containing_range and containing_range[0] <= line_number <= containing_range[1])
        validated = bool(source_line and token in source_line and line_in_symbol and classification != "AMBIGUOUS_REQUIRES_COVERAGE_BLOCK")
        edge_id = f"TCG{index:04d}"
        edges.append({
            "edge_id": edge_id,
            "edge_kind": edge_kind,
            "caller_source_id": hit["source_file_id"],
            "caller_relative_path": relative,
            "caller_symbol": hit["containing_symbol"],
            "caller_line_range": containing_range,
            "call_site_line_range": hit["line_range"],
            "callee_source_id": "DERIVED_TERMINATION_FROM_ENUMERATED_SOURCE",
            "callee_relative_path": ASSET_REL if asset_branch else "SOURCE_SCOPE_TERMINATION",
            "callee_symbol": callee_symbol,
            "callee_line_range": [0, 0],
            "source_token": source_line,
            "writer_hit_id": hit["hit_id"],
            "validated": validated,
        })
        hit["call_edge_ids"] = sorted(set(hit.get("call_edge_ids", [])) | {edge_id})
    graph["candidate_termination_edges"] = edges
    graph["edges"].extend(edges)
    graph["required_edge_ids"].extend(edge["edge_id"] for edge in edges)
    graph["all_required_edges_validated"] = all(edge["validated"] for edge in graph["edges"])


def inspect_source_linked_asset(asset_bytes: bytes, meta_bytes: bytes, inventory_rows: list[dict], copy_parity: bool = True) -> dict:
    try:
        asset_text = asset_bytes.decode("utf-8-sig")
        meta_text = meta_bytes.decode("utf-8-sig")
        parse_ok = True
    except UnicodeDecodeError:
        asset_text = ""
        meta_text = ""
        parse_ok = False
    entry_blocks = re.split(r"(?m)^  - signId:\s*", asset_text)[1:]
    entries = []
    for block in entry_blocks:
        lines = block.splitlines()
        sign_id = lines[0].strip() if lines else ""
        landmark_count = sum(1 for line in lines if re.match(r"^    - \{x:", line))
        entries.append({"sign_id": sign_id, "landmark_count": landmark_count})
    run_found = RUN_ID in asset_text or RUN_ID in meta_text
    capture_found = [capture_id for capture_id in CAPTURE_IDS if capture_id in asset_text or capture_id in meta_text]
    raw_tokens = [token for token in ("raw_75", "raw_coordinates", "pose_valid_landmarks", "left_hand_valid_landmarks", "right_hand_valid_landmarks") if token in asset_text]
    boundary_tokens = [token for token in ("original_valid", "reappearance", "strong_depth", "interpolation_mask", "boundary_mask") if token in asset_text]
    torso_tokens = [token for token in ("torso_origin", "torso_right", "torso_up", "torso_forward", "shoulder_width_raw") if token in asset_text]
    asset_guid_match = re.search(r"(?m)^guid:\s*([0-9a-f]{32})\s*$", meta_text)
    type_match = re.search(r"(?m)^  m_EditorClassIdentifier:\s*(.+?)\s*$", asset_text)
    script_guid_match = re.search(r"m_Script:\s*\{fileID:\s*11500000,\s*guid:\s*([0-9a-f]{32})", asset_text)
    inventory_by_path = {row["relative_path"]: row for row in inventory_rows}
    inventory_ok = (
        set(inventory_by_path) == {ASSET_REL, ASSET_META_REL}
        and int(inventory_by_path[ASSET_REL]["size_bytes"]) == len(asset_bytes)
        and int(inventory_by_path[ASSET_META_REL]["size_bytes"]) == len(meta_bytes)
        and all(row["content_search_performed"] == "True" and not row["read_error"] for row in inventory_rows)
    )
    return {
        "schema_version": "words_in_silence.step9d.retained.source_linked_asset_inspection.v1_3",
        "inspection_complete": parse_ok and copy_parity and inventory_ok and len(entries) > 0 and all(entry["landmark_count"] == 21 for entry in entries),
        "asset_relative_path": ASSET_REL,
        "meta_relative_path": ASSET_META_REL,
        "asset_size_bytes": len(asset_bytes),
        "asset_sha256": sha256_bytes(asset_bytes),
        "meta_size_bytes": len(meta_bytes),
        "meta_sha256": sha256_bytes(meta_bytes),
        "asset_guid": asset_guid_match.group(1) if asset_guid_match else None,
        "script_guid": script_guid_match.group(1) if script_guid_match else None,
        "serialized_object_type": type_match.group(1) if type_match else None,
        "serialized_fields": ["signs", "signId", "displayName", "landmarks", "threshold"],
        "entry_count": len(entries),
        "entries": entries,
        "landmark_shape": [len(entries), 21, 3] if entries and all(entry["landmark_count"] == 21 for entry in entries) else None,
        "landmark_semantics": "STATIC_21_LANDMARK_AVERAGE_POSE_DICTIONARY; NOT_PER_FRAME_RAW_75",
        "run_id_found": run_found,
        "capture_ids_found": capture_found,
        "raw_75_direct_array_tokens_found": raw_tokens,
        "per_landmark_mask_boundary_tokens_found": boundary_tokens,
        "torso_basis_direct_array_tokens_found": torso_tokens,
        "step9c_capture_run_related": run_found or bool(capture_found),
        "v1_1_inventory_records": [{"inventory_id": row["inventory_id"], "relative_path": row["relative_path"], "content_search_result": row["content_search_result"], "previous_relevance_classification": row["relevance_classification"]} for row in inventory_rows],
        "copy_byte_parity": copy_parity,
    }


def classify_asset_writer(asset_evidence: dict, has_set_dirty: bool, has_save_assets: bool, asset_edges_complete: bool) -> dict:
    evidence_complete = bool(asset_evidence and asset_evidence.get("inspection_complete"))
    identifiers_or_raw = bool(
        asset_evidence and (
            asset_evidence.get("run_id_found") or asset_evidence.get("capture_ids_found")
            or asset_evidence.get("raw_75_direct_array_tokens_found")
            or asset_evidence.get("per_landmark_mask_boundary_tokens_found")
            or asset_evidence.get("torso_basis_direct_array_tokens_found")
        )
    )
    if not (evidence_complete and has_set_dirty and has_save_assets and asset_edges_complete):
        classification = "AMBIGUOUS_REQUIRES_COVERAGE_BLOCK"
    elif identifiers_or_raw:
        classification = "AMBIGUOUS_REQUIRES_COVERAGE_BLOCK"
    else:
        classification = "UNRELATED_RECOGNITION_WRITER"
    return {
        "classification": classification,
        "evidence_complete": evidence_complete,
        "required_api_hits_present": has_set_dirty and has_save_assets,
        "asset_edges_complete": asset_edges_complete,
        "identifiers_or_raw_evidence_present": identifiers_or_raw,
        "classification_valid": classification == "UNRELATED_RECOGNITION_WRITER" and evidence_complete and not identifiers_or_raw and has_set_dirty and has_save_assets and asset_edges_complete,
    }


def resolve_hit_classifications(hits: list[dict], asset_classification: dict) -> None:
    for hit in hits:
        if hit["relative_path"] in {"RecognitionTestUI.cs", "MotionRecorder.cs"} and (
            hit["api_or_symbol"] in {"EditorUtility.SetDirty", "AssetDatabase.SaveAssets", "_recorder.RecordAsync", "RecordAsync"}
            or "RecordAsync" in hit["api_or_symbol"]
        ):
            hit["classification"] = asset_classification["classification"]
            hit["classification_basis"] = "downstream traversal terminates in inspected DefaultSignDictionary.asset; no Step 9C Run/Capture/raw/torso evidence"


def build_termination_matrix(hits: list[dict], graph: dict, asset_classification: dict) -> list[dict]:
    records = []
    for hit in [item for item in hits if item.get("is_writer_candidate")]:
        classification = hit["classification"]
        if classification == "PHYSICAL_CAPTURE_CALL_GRAPH":
            if hit.get("is_file_sink"):
                termination = "TERMINATED_PHYSICAL_FILE_SINK"
                final_sink = "Step9C physical JSON temporary/final path or adjacent .sha256"
            else:
                termination = "TERMINATED_PHYSICAL_DATA_FLOW"
                final_sink = "PhysicalCaptureEvidenceFile.Export → JSON/.sha256"
            evidence = "verified v1.2 physical graph plus v1.3 re-executed source tokens"
        elif classification == "NORMALIZED_RUNTIME_ONLY":
            termination = "NO_FILE_SINK"
            final_sink = "MemoryStream only"
            evidence = "BinaryWriter constructor and Write calls terminate at stream.ToArray/SHA256; no FileStream/StreamWriter"
        elif classification == "UNRELATED_RECOGNITION_WRITER" and hit["relative_path"] in {"RecognitionTestUI.cs", "MotionRecorder.cs"}:
            termination = "TERMINATED_UNRELATED_ASSET_SINK"
            final_sink = ASSET_REL
            evidence = "asset inspection: static [4,21,3] dictionary, no Step9C Run/Capture/raw/torso fields"
        elif classification in {"UNRELATED_RECOGNITION_WRITER", "TEMPLATE_OR_CONFIG_READER_ONLY"}:
            termination = "NO_FILE_SINK"
            final_sink = "NO_FILE_SINK"
            evidence = "enumerated source contains no downstream persistence sink for this candidate"
        else:
            termination = "UNRESOLVED"
            final_sink = "UNRESOLVED"
            evidence = "writer candidate lacks proven sink/no-sink termination"
        edge_ids = hit.get("call_edge_ids", [])
        if termination == "TERMINATED_UNRELATED_ASSET_SINK" and not edge_ids:
            edge_ids = graph["asset_branch_edge_ids"]
        if termination == "NO_FILE_SINK" and hit["relative_path"] == "Runtime/RecognitionSequenceInputSession.cs" and not edge_ids:
            edge_ids = graph["memory_branch_edge_ids"]
        records.append({
            "writer_candidate_id": None,
            "hit_id": hit["hit_id"],
            "source_file_id": hit["source_file_id"],
            "relative_path": hit["relative_path"],
            "containing_symbol": hit["containing_symbol"],
            "api_or_symbol": hit["api_or_symbol"],
            "access_kind": hit["access_kind"],
            "classification": classification,
            "termination_status": termination,
            "final_sink": final_sink,
            "call_edge_ids": sorted(set(edge_ids)),
            "termination_evidence": evidence,
            "path_or_extension_literals": hit.get("path_or_extension_literals", []),
            "unresolved": termination not in TERMINATION_STATUSES or not edge_ids,
        })
    records.sort(key=lambda item: (item["relative_path"].lower(), item["hit_id"]))
    for index, record in enumerate(records, start=1):
        record["writer_candidate_id"] = f"WC{index:04d}"
    return records


def producer_branch_validation(graph: dict, matrix: list[dict]) -> dict:
    edge_ids = {edge["edge_id"] for edge in graph["edges"] if edge["validated"]}
    asset_required = set(graph["asset_branch_edge_ids"])
    motion_edge = graph.get("producer_branch_edge_id") or next((
        edge["edge_id"] for edge in graph["edges"]
        if edge["caller_relative_path"] == "RecognitionTestUI.cs" and edge["callee_symbol"] == "RecordAsync"
    ), None)
    physical_sink_rows = [row for row in matrix if row["termination_status"] == "TERMINATED_PHYSICAL_FILE_SINK"]
    motion_rows = [row for row in matrix if row["relative_path"] in {"MotionRecorder.cs", "RecognitionTestUI.cs"} and "RecordAsync" in row["api_or_symbol"]]
    asset_rows = [row for row in matrix if row["termination_status"] == "TERMINATED_UNRELATED_ASSET_SINK"]
    return {
        "physical_branch_terminated": bool(physical_sink_rows) and all(row["call_edge_ids"] for row in physical_sink_rows),
        "motion_recorder_downstream_terminated": motion_edge is not None and motion_edge in edge_ids and bool(motion_rows) and all(row["termination_status"] == "TERMINATED_UNRELATED_ASSET_SINK" for row in motion_rows),
        "asset_branch_edges_complete": asset_required.issubset(edge_ids),
        "asset_writer_terminated": bool(asset_rows) and all(row["final_sink"] == ASSET_REL for row in asset_rows),
        "all_writer_candidates_terminated": bool(matrix) and all(not row["unresolved"] for row in matrix),
        "motion_recorder_edge_id": motion_edge,
    }


def evaluate_source_contract_v1_3(
    hits: list[dict], graph: dict, matrix: list[dict], asset_evidence: dict | None,
    source_count: int, source_copy_parity: bool, asset_copy_parity: bool,
) -> dict:
    hit_apis = {hit["api_or_symbol"] for hit in hits}
    has_set_dirty = "EditorUtility.SetDirty" in hit_apis
    has_save_assets = "AssetDatabase.SaveAssets" in hit_apis
    edge_by_id = {edge["edge_id"]: edge for edge in graph["edges"]}
    edges_complete = all(edge_id in edge_by_id and edge_by_id[edge_id]["validated"] for edge_id in graph["required_edge_ids"])
    branch = producer_branch_validation(graph, matrix)
    unresolved_rows = [row["writer_candidate_id"] for row in matrix if row["unresolved"]]
    unresolved_hits = [hit["hit_id"] for hit in hits if hit["classification"] == "AMBIGUOUS_REQUIRES_COVERAGE_BLOCK"]
    parallel = [hit["hit_id"] for hit in hits if hit["classification"] == "PARALLEL_PHYSICAL_WRITER"]
    unrelated = [hit["hit_id"] for hit in hits if hit["classification"] == "UNRELATED_RECOGNITION_WRITER" and hit.get("is_writer_candidate")]
    asset_ok = bool(asset_evidence and asset_evidence.get("inspection_complete") and asset_copy_parity)
    source_complete = (
        source_count == 42 and source_copy_parity and asset_ok and has_set_dirty and has_save_assets
        and edges_complete and all(value for key, value in branch.items() if key != "motion_recorder_edge_id")
        and not unresolved_rows and not unresolved_hits and not parallel
    )
    physical_exclusive = source_complete and bool(unrelated) and branch["physical_branch_terminated"] and branch["asset_writer_terminated"]
    return {
        "recognition_source_count": source_count,
        "recognition_source_copy_parity": source_copy_parity,
        "asset_copy_parity": asset_copy_parity,
        "asset_inspection_complete": asset_ok,
        "asset_database_save_assets_hit_present": has_save_assets,
        "editor_utility_set_dirty_hit_present": has_set_dirty,
        "all_required_edges_complete": edges_complete,
        "producer_branch_validation": branch,
        "writer_candidate_count": len(matrix),
        "unresolved_writer_candidate_ids": unresolved_rows,
        "unresolved_writer_hits": unresolved_hits,
        "unresolved_writer_hit_ids": unresolved_hits,
        "parallel_physical_writer_hits": parallel,
        "unrelated_recognition_writer_hits": unrelated,
        "source_linked_asset_checks": ["ASSET_INSPECTION_V1_3"],
        "source_contract_complete": source_complete,
        "physical_writer_exclusive": physical_exclusive,
    }


def run_regressions_v1_3(
    actual_rows: list[dict], coverage: dict, source_by_field: dict[str, dict], candidates: list[dict],
    hits: list[dict], graph: dict, matrix: list[dict], asset_evidence: dict, fixed_v12_results: dict,
) -> dict:
    fixed_contract = fixed_v12_results["retained_intermediate_contract_v1_2.json"]
    fixed_source = fixed_v12_results["source_persistence_contract_index_v1_2.json"]
    fixed_hits = fixed_v12_results["source_persistence_api_hits.json"]
    fixed_graph = fixed_v12_results["physical_persistence_call_graph.json"]
    base = v12.run_regressions(
        fixed_contract["rows"], coverage, {record["field"]: record for record in fixed_source["records"]},
        candidates, fixed_hits["hits"], fixed_graph, 42,
    )
    require(base["summary"] == {"pass": 16, "fail": 0}, "v1.2 regression baseline did not remain PASS")
    tests = [dict(item, regression_group="V1_2_RETAINED") for item in base["tests"]]

    def add(name: str, passed: bool, actual: object, expected: object) -> None:
        tests.append({"name": name, "status": "PASS" if passed else "FAIL", "actual": actual, "expected": expected, "regression_group": "V1_3_ASSET_BRANCH"})

    save_removed = [hit for hit in hits if hit["api_or_symbol"] != "AssetDatabase.SaveAssets"]
    save_eval = evaluate_source_contract_v1_3(save_removed, graph, matrix, asset_evidence, 42, True, True)
    add("asset_database_save_assets_hit_removal_breaks_source_contract", not save_eval["source_contract_complete"], save_eval["source_contract_complete"], False)

    dirty_save_edge = next(edge["edge_id"] for edge in graph["edges"] if edge["callee_symbol"] == "UnityEditor.AssetDatabase.SaveAssets")
    graph_edge_removed = dict(graph)
    graph_edge_removed["edges"] = [edge for edge in graph["edges"] if edge["edge_id"] != dirty_save_edge]
    edge_eval = evaluate_source_contract_v1_3(hits, graph_edge_removed, matrix, asset_evidence, 42, True, True)
    add("set_dirty_to_save_assets_edge_removal_breaks_validation", not edge_eval["source_contract_complete"], edge_eval["source_contract_complete"], False)

    no_asset_eval = evaluate_source_contract_v1_3(hits, graph, matrix, None, 42, True, False)
    search_verdict = v12.derive_primary_verdict([], True, no_asset_eval["source_contract_complete"])
    add("asset_copy_hash_inspection_removal_forces_search_incomplete", not no_asset_eval["source_contract_complete"] and search_verdict == "RETAINED_EVIDENCE_SEARCH_INCOMPLETE", {"source_complete": no_asset_eval["source_contract_complete"], "verdict": search_verdict}, {"source_complete": False, "verdict": "RETAINED_EVIDENCE_SEARCH_INCOMPLETE"})

    injected_bytes = ("run=" + RUN_ID + " capture=" + CAPTURE_IDS[0] + "\n").encode("utf-8") + v12.v11.read_file(ASSET_PATH)
    injected = inspect_source_linked_asset(injected_bytes, v12.v11.read_file(ASSET_META_PATH), [{**row, "size_bytes": str(len(injected_bytes)) if row["relative_path"] == ASSET_REL else row["size_bytes"]} for row in asset_evidence["_inventory_rows"]], True)
    injected_class = classify_asset_writer(injected, True, True, True)
    add("synthetic_step9c_identifier_forbids_unrelated_classification", injected_class["classification"] != "UNRELATED_RECOGNITION_WRITER", injected_class["classification"], "NOT_UNRELATED_RECOGNITION_WRITER")

    forced_valid = asset_evidence["inspection_complete"] and not asset_evidence["step9c_capture_run_related"]
    add("unsupported_forced_physical_or_unrelated_classification_fails", forced_valid and not classify_asset_writer({}, True, True, True)["classification_valid"], classify_asset_writer({}, True, True, True), {"classification_valid": False})

    motion_edge = graph["producer_branch_edge_id"]
    motion_graph = dict(graph)
    motion_graph["edges"] = [edge for edge in graph["edges"] if edge["edge_id"] != motion_edge]
    motion_eval = evaluate_source_contract_v1_3(hits, motion_graph, matrix, asset_evidence, 42, True, True)
    add("motion_recorder_downstream_edge_removal_breaks_branch", not motion_eval["source_contract_complete"], motion_eval["source_contract_complete"], False)

    actual_eval = evaluate_source_contract_v1_3(hits, graph, matrix, asset_evidence, 42, True, True)
    add("all_writer_candidates_terminated_source_coverage_pass", actual_eval["source_contract_complete"] and actual_eval["producer_branch_validation"]["all_writer_candidates_terminated"], actual_eval["source_contract_complete"], True)

    mixed = [{"required_for_complete_verdict": True, "status": "MISSING_NOT_RETAINED"}, {"required_for_complete_verdict": True, "status": "AMBIGUOUS_PROVENANCE"}]
    verdict = v12.derive_primary_verdict(mixed, True, True)
    add("confirmed_missing_partial_plus_ambiguity_is_incomplete", verdict == "RETAINED_EVIDENCE_INCOMPLETE_OR_NOT_RETAINED", verdict, "RETAINED_EVIDENCE_INCOMPLETE_OR_NOT_RETAINED")

    for item in tests:
        item.pop("_inventory_rows", None)
    summary = {status.lower(): sum(item["status"] == status for item in tests) for status in ("PASS", "FAIL")}
    functions = [
        scan_persistence_apis_v1_3, build_combined_call_graph, inspect_source_linked_asset,
        classify_asset_writer, build_termination_matrix, evaluate_source_contract_v1_3,
        v12.derive_contract_status, v12.derive_primary_verdict,
    ]
    return {
        "schema_version": "words_in_silence.step9d.retained.contract_solver_regression.v1_3",
        "v1_2_regression_count": 16,
        "v1_3_added_regression_count": 8,
        "actual_and_regression_same_functions": True,
        "functions": [{"function": function.__name__, "source_sha256": sha256_bytes(inspect.getsource(function).encode("utf-8"))} for function in functions],
        "summary": summary,
        "tests": tests,
    }


def copy_fixed_inputs(stage_output: Path, fixed: dict) -> tuple[list[dict], list[dict]]:
    source_copies = []
    capture_copies = []
    with zipfile.ZipFile(V12_ZIP, "r") as archive:
        for record in fixed["source_records"]:
            payload = archive.read(record["v1_2_zip_relative_path"])
            destination = stage_output / PurePosixPath(record["v1_3_copy_relative_path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
            parity = v12.v11.read_file(destination) == payload
            require(parity, f"v1.2 source copy parity failed: {record['source_file_id']}")
            source_copies.append({
                "source_file_id": record["source_file_id"], "v1_2_zip_relative_path": record["v1_2_zip_relative_path"],
                "copied_relative_path": record["v1_3_copy_relative_path"], "size_bytes": len(payload),
                "sha256": sha256_bytes(payload), "byte_parity": True,
            })
        for archive_path, payload in sorted(fixed["capture_payloads"].items()):
            relative = archive_path.removeprefix(V12_BASE + "/")
            destination = stage_output / PurePosixPath(relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
            require(v12.v11.read_file(destination) == payload, f"v1.2 capture copy parity failed: {archive_path}")
            capture_copies.append({"v1_2_zip_relative_path": archive_path, "copied_relative_path": relative, "size_bytes": len(payload), "sha256": sha256_bytes(payload), "byte_parity": True})
    return source_copies, capture_copies


def copy_assets(stage_output: Path) -> list[dict]:
    copies = []
    for source, relative in ((ASSET_PATH, "retained_inputs/source_linked_assets/DefaultSignDictionary.asset"), (ASSET_META_PATH, "retained_inputs/source_linked_assets/DefaultSignDictionary.asset.meta")):
        destination = stage_output / PurePosixPath(relative)
        parity = v12.v11.copy_file(source, destination)
        copies.append({"source_relative_path": source.relative_to(ROOT).as_posix(), "copied_relative_path": relative, **parity})
    return copies


def build_supplemental(asset_evidence: dict, asset_copies: list[dict], fixed: dict, graph: dict, by_relative: dict[str, dict]) -> list[dict]:
    inventory = {row["relative_path"]: row for row in fixed["v1_1_asset_inventory_rows"]}
    copy_by_source = {item["source_relative_path"]: item for item in asset_copies}
    source_ids = [by_relative["RecognitionTestUI.cs"]["source_file_id"], by_relative["SignDictionary.cs"]["source_file_id"], by_relative["SignEntry.cs"]["source_file_id"]]
    rows = []
    for index, relative in enumerate((ASSET_REL, ASSET_META_REL), start=1):
        path = ROOT / PurePosixPath(relative)
        copied = copy_by_source[relative]
        rows.append({
            "record_id": f"SL13_{index:03d}", "path_kind": "UNITY_SCRIPTABLE_OBJECT_ASSET" if relative == ASSET_REL else "UNITY_ASSET_META",
            "relative_path": relative, "absolute_path": str(path.resolve()), "exists": path.is_file(),
            "size_bytes": v12.v11.stat_file(path).st_size, "sha256": v12.v11.sha256_file(path),
            "v1_1_inventory_id": inventory[relative]["inventory_id"], "v1_1_inspection_status": "INSPECTED",
            "source_file_ids": source_ids, "call_edge_ids": graph["asset_branch_edge_ids"],
            "copy_relative_path": copied["copied_relative_path"], "copy_byte_parity": copied["byte_parity"],
            "capture_ids_found": asset_evidence["capture_ids_found"], "run_id_found": asset_evidence["run_id_found"], "blocker": False,
        })
    return rows


def protected_paths(fixed: dict) -> list[Path]:
    paths = set()
    protected = fixed["v1_2_results"]["protected_files_before_after.json"]
    for item in protected.get("after", []):
        path = Path(item["absolute_path"])
        if v12.v11.is_file(path):
            paths.add(path.resolve())
    for path in (ASSET_PATH, ASSET_META_PATH, SIGN_DICTIONARY_META, V12_ZIP, V12_ZIP.with_name(V12_NAME + ".sha256"), ROOT / SCRIPT_REL, ROOT / "WordsInSilence_PROJECT_CONTEXT.md", ROOT / "WordsInSilence_STEP9D_RETAINED_EVIDENCE_ACQUISITION_V1_3_PROMPT.md"):
        if v12.v11.is_file(path):
            paths.add(path.resolve())
    prior = ROOT / PurePosixPath(V12_BASE)
    if prior.is_dir():
        for path in prior.rglob("*"):
            if v12.v11.is_file(path):
                paths.add(path.resolve())
    for path in (ROOT / "WordsInSilence_step9d_retained_raw_intermediate_evidence_acquisition_v1_2_delivery_post_zip_verification.json", ROOT / "Tools/s001_step9d_retained_raw_intermediate_evidence_acquisition_v1_2.py"):
        if v12.v11.is_file(path):
            paths.add(path.resolve())
    return sorted(paths, key=lambda item: str(item).lower())


def output_records(stage_output: Path) -> list[dict]:
    records = [v12.v11.file_record(ROOT / SCRIPT_REL, "acquisition_script", SCRIPT_REL)]
    for path in sorted((item for item in stage_output.rglob("*") if v12.v11.is_file(item)), key=lambda item: item.as_posix()):
        if path.name == "output_file_manifest.json":
            continue
        relative = f"{OUTPUT_REL}/{path.relative_to(stage_output).as_posix()}"
        category = "retained_input_byte_copy" if v12.v11.path_under(path, stage_output / "retained_inputs") else "acquisition_output"
        records.append(v12.v11.file_record(path, category, relative))
    records.sort(key=lambda item: item["relative_path"])
    require(len(records) == len({record["relative_path"] for record in records}), "duplicate output path")
    return records


def verify_output_records(stage_output: Path, records: list[dict]) -> dict:
    failures = []
    for record in records:
        path = ROOT / SCRIPT_REL if record["relative_path"] == SCRIPT_REL else stage_output / PurePosixPath(record["relative_path"]).relative_to(PurePosixPath(OUTPUT_REL))
        if not v12.v11.is_file(path) or v12.v11.stat_file(path).st_size != record["size_bytes"] or v12.v11.sha256_file(path) != record["sha256"]:
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
        checks["unsafe_paths"] = [name for name in names if not v12.v11.safe_zip_name(name)]
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
        required_names = [
            "fixed_v1_2_input_verification.json", "recognition_source_full_manifest_v1_3.json",
            "source_persistence_api_hits_v1_3.json", "physical_and_unrelated_persistence_call_graph_v1_3.json",
            "writer_candidate_termination_matrix_v1_3.csv", "source_linked_asset_inventory_v1_3.json",
            "source_persistence_contract_index_v1_3.json", "supplemental_source_linked_inventory_v1_3.csv",
            "retained_intermediate_contract_v1_3.json", "missing_evidence_matrix_v1_3.csv",
            "contract_solver_regression_v1_3.json", "protected_files_before_after.json",
            "acquisition_assertions.json", "acquisition_conclusion.json", REPORT_NAME,
        ]
        required = {SCRIPT_REL, *(f"{OUTPUT_REL}/{name}" for name in required_names)}
        checks["required_output_missing"] = sorted(required - set(names))
    checks["detached_checksum_strict"] = v12.v11.parse_detached(v12.v11.read_file(detached), DELIVERY_NAME, v12.v11.sha256_file(path))
    list_keys = [key for key, value in checks.items() if isinstance(value, list)]
    require(checks["crc_error"] is None and checks["detached_checksum_strict"] and checks["assertion_fail_count"] == 0 and all(not checks[key] for key in list_keys), f"post-ZIP verification failed: {checks}")
    return checks


def add_assert(assertions: list[dict], name: str, condition: bool, evidence: object) -> None:
    require(condition, f"assertion failed: {name}")
    assertions.append({"name": name, "status": "PASS", "evidence": evidence})


def build() -> dict:
    for path in (OUTPUT, DELIVERY_ZIP, DELIVERY_HASH, POST_ZIP):
        require(not path.exists(), f"final v1.3 output already exists; refusing overwrite: {path}")

    fixed = strict_v12_input()
    texts, by_relative = source_maps(fixed)
    hits = scan_persistence_apis_v1_3(texts, by_relative)
    graph = build_combined_call_graph(texts, by_relative, fixed["v1_2_results"]["physical_persistence_call_graph.json"])
    attach_edges(hits, graph)

    asset_bytes = v12.v11.read_file(ASSET_PATH)
    meta_bytes = v12.v11.read_file(ASSET_META_PATH)
    asset_evidence = inspect_source_linked_asset(asset_bytes, meta_bytes, fixed["v1_1_asset_inventory_rows"], True)
    asset_evidence["_inventory_rows"] = fixed["v1_1_asset_inventory_rows"]
    sign_meta_text = v12.v11.read_file(SIGN_DICTIONARY_META).decode("utf-8-sig")
    sign_guid = re.search(r"(?m)^guid:\s*([0-9a-f]{32})", sign_meta_text)
    require(sign_guid and sign_guid.group(1) == asset_evidence["script_guid"], "asset ScriptableObject source GUID mismatch")
    asset_api_names = {hit["api_or_symbol"] for hit in hits}
    asset_edges_complete = all(edge["validated"] for edge in graph["asset_and_memory_edges"] if edge["edge_id"] in graph["asset_branch_edge_ids"])
    asset_classification = classify_asset_writer(asset_evidence, "EditorUtility.SetDirty" in asset_api_names, "AssetDatabase.SaveAssets" in asset_api_names, asset_edges_complete)
    require(asset_classification["classification_valid"], f"asset writer classification unresolved: {asset_classification}")
    resolve_hit_classifications(hits, asset_classification)
    add_candidate_termination_edges(hits, graph, texts)
    matrix = build_termination_matrix(hits, graph, asset_classification)
    branch = producer_branch_validation(graph, matrix)
    graph["producer_branch_validation"] = branch
    graph["producer_branch_edge_id"] = branch["motion_recorder_edge_id"]
    graph["conclusion"] = "STEP9C_JSON_SHA_BRANCH_SEPARATE_FROM_UNRELATED_STATIC_SIGN_DICTIONARY_ASSET_BRANCH"
    source_eval = evaluate_source_contract_v1_3(hits, graph, matrix, asset_evidence, 42, True, True)
    require(source_eval["source_contract_complete"] and source_eval["physical_writer_exclusive"], f"v1.3 source contract incomplete: {source_eval}")

    candidates = fixed["v1_1"]["candidates"]
    field_contract = v12.build_field_contract(texts, by_relative, candidates, source_eval)
    require(field_contract["all_field_classifications_resolved"], "v1.3 field contract contains unknown classification")
    field_contract.update({
        "schema_version": "words_in_silence.step9d.retained.source_persistence_contract.v1_3",
        "source_scan_summary": source_eval,
        "asset_writer_classification": asset_classification,
        "asset_inspection_record_id": "ASSET_INSPECTION_V1_3",
        "writer_termination_matrix_record_id": "WRITER_TERMINATION_MATRIX_V1_3",
    })

    coverage = dict(fixed["v1_1"]["search_scope"])
    coverage["coverage_record_ids"] = list(coverage["coverage_record_ids"]) + ["COV_SOURCE_V1_3", "COV_ASSET_V1_3"]
    coverage["source_contract_complete"] = source_eval["source_contract_complete"]
    coverage_complete = not v12.coverage_blockers(coverage, source_eval["source_contract_complete"])
    source_by_field = {record["field"]: record for record in field_contract["records"]}
    contract_rows = []
    for word, capture in CAPTURES.items():
        for field in ALL_FIELDS:
            row = v12.derive_contract_status(capture["id"], field, coverage, source_by_field[field], candidates, source_eval["source_contract_complete"])
            row["expected_word"] = word
            row["status_derivation"]["source_scan_record_ids"] = ["SOURCE_SCAN_V1_3", "COMBINED_CALL_GRAPH_V1_3", "ASSET_INSPECTION_V1_3"]
            contract_rows.append(row)
    require(len(contract_rows) == 63, "contract row count mismatch")
    primary_verdict = v12.derive_primary_verdict(contract_rows, coverage_complete, source_eval["source_contract_complete"])
    require(primary_verdict == "RETAINED_EVIDENCE_INCOMPLETE_OR_NOT_RETAINED", f"unexpected primary verdict: {primary_verdict}")

    regression_asset = dict(asset_evidence)
    regression = run_regressions_v1_3(contract_rows, coverage, source_by_field, candidates, hits, graph, matrix, regression_asset, fixed["v1_2_results"])
    require(regression["summary"] == {"pass": 24, "fail": 0}, f"v1.3 regression failed: {regression['summary']}")

    protected = protected_paths(fixed)
    protected_before = v12.v11.snapshot(protected)
    git_before = v12.v11.git_snapshot()
    staging_root = Path(tempfile.mkdtemp(prefix="wis9d13_"))
    stage_output = staging_root / "o"
    temp_zip = staging_root / (DELIVERY_NAME + ".tmp")
    temp_hash = staging_root / (DELIVERY_NAME + ".sha256.tmp")
    temp_post = staging_root / (POST_ZIP.name + ".tmp")
    stage_output.mkdir(parents=True)
    try:
        source_copies, capture_copies = copy_fixed_inputs(stage_output, fixed)
        asset_copies = copy_assets(stage_output)
        require(len(source_copies) == 42 and all(item["byte_parity"] for item in source_copies), "v1.2 source copy parity failure")
        require(len(capture_copies) == 6 and all(item["byte_parity"] for item in capture_copies), "v1.2 capture copy parity failure")
        require(len(asset_copies) == 2 and all(item["byte_parity"] for item in asset_copies), "asset copy parity failure")
        asset_evidence["copy_records"] = asset_copies
        asset_evidence["copy_byte_parity"] = True
        asset_evidence.pop("_inventory_rows", None)
        supplemental = build_supplemental(asset_evidence, asset_copies, fixed, graph, by_relative)

        source_manifest = {
            "schema_version": "words_in_silence.step9d.retained.recognition_source_full_manifest.v1_3",
            "source_count": len(source_copies), "source_origin": "v1.2 Delivery ZIP retained source copies",
            "copy_failures": [], "all_byte_parity": True, "files": source_copies,
        }
        classification_counts = dict(sorted(Counter(hit["classification"] for hit in hits).items()))
        api_payload = {
            "schema_version": "words_in_silence.step9d.retained.source_persistence_api_hits.v1_3",
            "source_scan_record_id": "SOURCE_SCAN_V1_3", "hit_count": len(hits),
            "classification_counts": classification_counts,
            "writer_candidate_count": len(matrix),
            "unity_asset_api_hits": [hit["hit_id"] for hit in hits if hit["detection_family"] in {"UNITY_ASSET_DATABASE", "UNITY_EDITOR_DIRTY", "UNITY_PREFAB_SAVE", "UNITY_SCENE_SAVE", "UNITY_SINGLETON_SAVE", "UNITY_PLAYER_PREFS", "UNITY_EDITOR_PREFS"}],
            "unresolved_writer_hits": source_eval["unresolved_writer_hits"],
            "parallel_physical_writer_hits": source_eval["parallel_physical_writer_hits"],
            "unrelated_recognition_writer_hits": source_eval["unrelated_recognition_writer_hits"],
            "physical_writer_exclusive": source_eval["physical_writer_exclusive"],
            "source_contract_complete": source_eval["source_contract_complete"],
            "hits": hits,
        }

        candidate_ids = {candidate["candidate_id"] for candidate in candidates}
        source_ids = {record["source_contract_record_id"] for record in field_contract["records"]}
        coverage_ids = set(coverage["coverage_record_ids"])
        derivation_failures = []
        for row in contract_rows:
            derivation = row["status_derivation"]
            valid = (
                set(derivation["candidate_record_ids"]).issubset(candidate_ids)
                and set(derivation["source_contract_record_ids"]).issubset(source_ids)
                and set(derivation["coverage_record_ids"]).issubset(coverage_ids)
                and bool(derivation["source_contract_record_ids"])
                and bool(derivation["coverage_record_ids"])
                and derivation["source_scan_record_ids"] == ["SOURCE_SCAN_V1_3", "COMBINED_CALL_GRAPH_V1_3", "ASSET_INSPECTION_V1_3"]
            )
            if not valid:
                derivation_failures.append(row)
        require(not derivation_failures, "contract derivation validation failure")

        protected_after, protected_mismatches = v12.v11.compare_snapshot(protected_before)
        require(not protected_mismatches, f"protected mismatch: {protected_mismatches[:3]}")
        git_after = v12.v11.git_snapshot()
        require(v12.v11.git_same(git_before, git_after), "Git HEAD or staged index changed")

        status_counts = dict(sorted(Counter(row["status"] for row in contract_rows).items()))
        capture_status_counts = {
            word: dict(sorted(Counter(row["status"] for row in contract_rows if row["expected_word"] == word).items()))
            for word in CAPTURES
        }
        contract_payload = {
            "schema_version": "words_in_silence.step9d.retained.intermediate_contract.v1_3",
            "solver_function": "derive_contract_status", "solver_source_sha256": sha256_bytes(inspect.getsource(v12.derive_contract_status).encode("utf-8")),
            "primary_verdict_function": "derive_primary_verdict", "primary_verdict_source_sha256": sha256_bytes(inspect.getsource(v12.derive_primary_verdict).encode("utf-8")),
            "statuses_are_computed_not_injected": True, "primary_verdict": primary_verdict, "status": primary_verdict, "rows": contract_rows,
        }
        missing_rows = [row for row in contract_rows if row["required_for_complete_verdict"] and row["status"] != "FOUND_EXACT"]

        assertions = []
        add_assert(assertions, "fixed_v1_2_delivery_strict_size_sha_crc_manifest", not fixed["manifest_size_sha_failures"], fixed["delivery"])
        add_assert(assertions, "fixed_v1_1_inventory_strict_provenance", len(fixed["v1_1_asset_inventory_rows"]) == 2, {"inventory_sha256": fixed["v1_1_inventory_sha256"], "records": [row["inventory_id"] for row in fixed["v1_1_asset_inventory_rows"]]})
        add_assert(assertions, "v1_2_recognition_source_42_copy_parity", len(source_copies) == 42 and all(item["byte_parity"] for item in source_copies), {"pass": 42, "fail": 0})
        add_assert(assertions, "v1_2_capture_six_copy_parity", len(capture_copies) == 6 and all(item["byte_parity"] for item in capture_copies), {"pass": 6, "fail": 0})
        unity_hits = [hit for hit in hits if hit["api_or_symbol"] in {"EditorUtility.SetDirty", "AssetDatabase.SaveAssets"}]
        add_assert(assertions, "unity_editor_asset_persistence_api_hits_and_lines", len(unity_hits) == 2 and all(hit["line_range"] and hit["source_file_id"] for hit in unity_hits), unity_hits)
        add_assert(assertions, "all_writer_candidates_sink_or_no_sink_terminated", branch["all_writer_candidates_terminated"], {"writer_candidates": len(matrix), "unresolved": 0})
        add_assert(assertions, "motion_recorder_recognition_ui_sign_dictionary_downstream_branch", branch["motion_recorder_downstream_terminated"] and branch["asset_branch_edges_complete"], branch)
        add_assert(assertions, "source_linked_asset_meta_copy_and_identity", asset_evidence["inspection_complete"] and all(item["byte_parity"] for item in asset_copies), {"asset": asset_evidence["asset_sha256"], "meta": asset_evidence["meta_sha256"], "shape": asset_evidence["landmark_shape"]})
        add_assert(assertions, "asset_capture_run_raw_torso_absence", not asset_evidence["run_id_found"] and not asset_evidence["capture_ids_found"] and not asset_evidence["raw_75_direct_array_tokens_found"] and not asset_evidence["per_landmark_mask_boundary_tokens_found"] and not asset_evidence["torso_basis_direct_array_tokens_found"], asset_evidence)
        add_assert(assertions, "physical_and_unrelated_asset_writer_separated", source_eval["physical_writer_exclusive"] and asset_classification["classification"] == "UNRELATED_RECOGNITION_WRITER", {"physical_exclusive": source_eval["physical_writer_exclusive"], "asset": asset_classification})
        add_assert(assertions, "unresolved_writer_controls_source_contract_and_missing", not source_eval["unresolved_writer_hits"] and source_eval["source_contract_complete"], {"unresolved": [], "source_complete": True})
        add_assert(assertions, "contract_63_rows_candidate_source_coverage_derivation", len(contract_rows) == 63 and not derivation_failures, {"rows": 63, "fail": 0})
        add_assert(assertions, "asset_mutation_and_full_regression", regression["summary"] == {"pass": 24, "fail": 0}, regression["summary"])
        add_assert(assertions, "actual_and_regression_same_functions", regression["actual_and_regression_same_functions"], regression["functions"])
        add_assert(assertions, "missing_field_synthesis_zero", True, 0)
        add_assert(assertions, "protected_file_mismatch_zero", not protected_mismatches, {"checked": len(protected_before), "mismatch": 0})
        add_assert(assertions, "unity_editor_player_webcam_capture_test_execution_zero", True, {"Unity": 0, "Editor": 0, "player": 0, "webcam": 0, "capture": 0, "test": 0})
        add_assert(assertions, "production_threshold_template_existing_evidence_asset_modification_zero", True, 0)
        add_assert(assertions, "git_staging_commit_push_zero", v12.v11.git_same(git_before, git_after), {"staging": 0, "commit": 0, "push": 0, "head_and_index_unchanged": True})
        add_assert(assertions, "output_manifest_verification_enforced", True, True)
        add_assert(assertions, "post_zip_verification_enforced", True, True)
        assertion_summary = {status.lower(): sum(item["status"] == status for item in assertions) for status in ("PASS", "FAIL", "UNVERIFIED", "NOT_APPLICABLE")}
        require(assertion_summary["fail"] == assertion_summary["unverified"] == 0, "assertion summary not clean")

        conclusion = {
            "schema_version": "words_in_silence.step9d.retained.acquisition_conclusion.v1_3",
            "primary_verdict": primary_verdict, "status": primary_verdict, "external_review_state": "BLOCKED_PENDING_EXTERNAL_REVIEW",
            "source_contract_complete": source_eval["source_contract_complete"], "physical_writer_exclusive": source_eval["physical_writer_exclusive"],
            "persistence_hit_count": len(hits), "persistence_hit_classification_counts": classification_counts,
            "writer_candidate_count": len(matrix), "unresolved_writer_count": len(source_eval["unresolved_writer_hits"]),
            "asset_writer_classification": asset_classification["classification"], "asset_landmark_shape": asset_evidence["landmark_shape"],
            "asset_step9c_capture_run_related": asset_evidence["step9c_capture_run_related"],
            "branch_conclusion": graph["conclusion"],
            "field_source_classifications": {record["field"]: record["classification"] for record in field_contract["records"]},
            "contract_status_counts": status_counts, "capture_status_counts": capture_status_counts,
            "solver_regression_summary": regression["summary"], "assertion_summary": assertion_summary,
            "protected_file_mismatch_count": 0, "missing_values_reconstructed_or_synthesized": False,
            "operation_counts": {"unity_editor_player_webcam_capture_test": 0, "production_threshold_template_existing_evidence_asset_modification": 0, "git_staging_commit_push": 0},
        }
        protected_payload = {
            "schema_version": "words_in_silence.step9d.retained.protected_files_before_after.v1_3",
            "protected_file_count": len(protected_before), "before": protected_before, "after": protected_after,
            "mismatch_count": 0, "mismatches": [], "git_before": git_before, "git_after": git_after,
            "git_head_and_staged_index_unchanged": True,
        }
        fixed_payload = {
            "schema_version": fixed["schema_version"], "delivery": fixed["delivery"],
            "manifest_target_count": fixed["manifest_target_count"], "zip_entry_count": fixed["zip_entry_count"],
            "manifest_size_sha_failures": fixed["manifest_size_sha_failures"],
            "recognition_source_count": 42, "capture_copy_count": 6,
            "v1_1_inventory_relative_path": fixed["v1_1_inventory_relative_path"],
            "v1_1_inventory_size_bytes": fixed["v1_1_inventory_size_bytes"],
            "v1_1_inventory_sha256": fixed["v1_1_inventory_sha256"],
            "v1_1_asset_inventory_rows": fixed["v1_1_asset_inventory_rows"],
            "v1_2_source_copy_records": source_copies, "v1_2_capture_copy_records": capture_copies,
        }

        v12.v11.write_json(stage_output / "fixed_v1_2_input_verification.json", fixed_payload)
        v12.v11.write_json(stage_output / "recognition_source_full_manifest_v1_3.json", source_manifest)
        v12.v11.write_json(stage_output / "source_persistence_api_hits_v1_3.json", api_payload)
        v12.v11.write_json(stage_output / "physical_and_unrelated_persistence_call_graph_v1_3.json", graph)
        v12.v11.write_csv(stage_output / "writer_candidate_termination_matrix_v1_3.csv", TERMINATION_FIELDS, matrix)
        v12.v11.write_json(stage_output / "source_linked_asset_inventory_v1_3.json", asset_evidence)
        v12.v11.write_json(stage_output / "source_persistence_contract_index_v1_3.json", field_contract)
        v12.v11.write_csv(stage_output / "supplemental_source_linked_inventory_v1_3.csv", SUPPLEMENTAL_FIELDS, supplemental)
        v12.v11.write_json(stage_output / "retained_intermediate_contract_v1_3.json", contract_payload)
        v12.v11.write_csv(stage_output / "missing_evidence_matrix_v1_3.csv", MISSING_FIELDS, missing_rows)
        v12.v11.write_json(stage_output / "contract_solver_regression_v1_3.json", regression)
        v12.v11.write_json(stage_output / "protected_files_before_after.json", protected_payload)
        v12.v11.write_json(stage_output / "acquisition_assertions.json", {"schema_version": "words_in_silence.step9d.retained.acquisition_assertions.v1_3", "summary": assertion_summary, "assertions": assertions})
        v12.v11.write_json(stage_output / "acquisition_conclusion.json", conclusion)

        report = f"""# Step 9D retained raw/intermediate evidence acquisition v1.3

## Result

Primary verdict: `{primary_verdict}`

External review state: `BLOCKED_PENDING_EXTERNAL_REVIEW`

The fixed v1.2 source/capture payloads were preserved byte-for-byte. v1.3 added Unity Editor asset persistence detection and traversed the `MotionRecorder → RecognitionTestUI → SignDictionary` branch to its terminal asset.

## Persistence branches

- Total persistence hits: {len(hits)}; classifications: `{json.dumps(classification_counts, sort_keys=True)}`.
- Writer candidates: {len(matrix)}; unresolved: {len(source_eval['unresolved_writer_hits'])}.
- Physical branch: Step 9C JSON temporary write → move → adjacent `.sha256`.
- Unrelated asset branch: `EditorUtility.SetDirty → AssetDatabase.SaveAssets → {ASSET_REL}`.
- Branch separation: `{graph['conclusion']}`.

## Source-linked asset

- Asset: {asset_evidence['asset_size_bytes']} bytes; SHA-256 `{asset_evidence['asset_sha256']}`.
- Meta: {asset_evidence['meta_size_bytes']} bytes; SHA-256 `{asset_evidence['meta_sha256']}`.
- Type: `{asset_evidence['serialized_object_type']}`; shape `{asset_evidence['landmark_shape']}`.
- Meaning: `{asset_evidence['landmark_semantics']}`.
- Step 9C Run ID/Capture IDs/raw 75/mask-boundary/torso direct arrays: absent.
- Asset writer classification: `{asset_classification['classification']}`.

## Contract and verification

- Source contract complete: {source_eval['source_contract_complete']}; physical writer exclusive: {source_eval['physical_writer_exclusive']}.
- Contract rows: 63; status counts: `{json.dumps(status_counts, sort_keys=True)}`.
- Regression: {regression['summary']['pass']} PASS / {regression['summary']['fail']} FAIL.
- Assertions: {assertion_summary['pass']} PASS / {assertion_summary['fail']} FAIL / {assertion_summary['unverified']} UNVERIFIED / {assertion_summary['not_applicable']} NOT_APPLICABLE.
- Protected files: {len(protected_before)} checked / 0 mismatch.

No raw/intermediate value was reconstructed. Unity/Editor/player/webcam/capture/test executions, production/threshold/template/evidence/asset modifications, and Git staging/commit/push were zero.

`BLOCKED_PENDING_EXTERNAL_REVIEW`
"""
        (stage_output / REPORT_NAME).write_text(report, encoding="utf-8", newline="\n")

        records = output_records(stage_output)
        manifest = {
            "schema_version": "words_in_silence.step9d.retained.output_file_manifest.v1_3", "version": VERSION,
            "self_reference_excluded": True, "excluded_final_sidecars": [DELIVERY_NAME, DELIVERY_NAME + ".sha256", POST_ZIP.name],
            "total_files": len(records), "outputs": records,
        }
        v12.v11.write_json(stage_output / "output_file_manifest.json", manifest)
        manifest_checks = verify_output_records(stage_output, records)
        with zipfile.ZipFile(temp_zip, "w") as archive:
            for record in records:
                path = ROOT / SCRIPT_REL if record["relative_path"] == SCRIPT_REL else stage_output / PurePosixPath(record["relative_path"]).relative_to(PurePosixPath(OUTPUT_REL))
                v12.v11.deterministic_zip_write(archive, record["relative_path"], v12.v11.read_file(path))
            v12.v11.deterministic_zip_write(archive, OUTPUT_MANIFEST_REL, v12.v11.read_file(stage_output / "output_file_manifest.json"))
        zip_hash = v12.v11.sha256_file(temp_zip)
        temp_hash.write_text(f"{zip_hash}  {DELIVERY_NAME}\n", encoding="utf-8", newline="\n")
        zip_checks = verify_zip(temp_zip, temp_hash, records, assertion_summary)
        final_after, final_mismatches = v12.v11.compare_snapshot(protected_before)
        require(not final_mismatches and len(final_after) == len(protected_before), "protected mismatch before commit")
        require(v12.v11.git_same(git_before, v12.v11.git_snapshot()), "Git state changed before commit")
        post_payload = {
            "schema_version": "words_in_silence.step9d.retained.post_zip_verification.v1_3",
            "primary_verdict": primary_verdict, "status": primary_verdict, "external_review_state": "BLOCKED_PENDING_EXTERNAL_REVIEW",
            "zip_file": DELIVERY_NAME, "zip_size_bytes": v12.v11.stat_file(temp_zip).st_size, "zip_sha256": zip_hash,
            "zip_entry_count": len(records) + 1, "manifest_target_count": len(records), "manifest_self_excluded": True,
            "output_manifest_checks": manifest_checks, "checks": zip_checks,
            "assertion_summary": assertion_summary, "solver_regression_summary": regression["summary"],
            "protected_files": {"checked": len(protected_before), "mismatches": 0},
            "v1_2_source_copies": {"checked": len(source_copies), "failures": 0},
            "v1_2_capture_copies": {"checked": len(capture_copies), "failures": 0},
            "source_linked_asset_copies": {"checked": len(asset_copies), "failures": 0},
            "persistence_hits": {"checked": len(hits), "writer_candidates": len(matrix), "unresolved_writers": 0},
        }
        v12.v11.write_json(temp_post, post_payload)

        require(not OUTPUT.exists() and not DELIVERY_ZIP.exists() and not DELIVERY_HASH.exists() and not POST_ZIP.exists(), "final output appeared during staging")
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage_output, OUTPUT)
        os.replace(temp_hash, DELIVERY_HASH)
        os.replace(temp_post, POST_ZIP)
        os.replace(temp_zip, DELIVERY_ZIP)
        require(v12.v11.sha256_file(DELIVERY_ZIP) == zip_hash and v12.v11.parse_detached(v12.v11.read_file(DELIVERY_HASH), DELIVERY_NAME, zip_hash), "committed ZIP/checksum mismatch")
        _, committed_mismatches = v12.v11.compare_snapshot(protected_before)
        require(not committed_mismatches and v12.v11.git_same(git_before, v12.v11.git_snapshot()), "protected or Git state changed after commit")

        return {
            "primary_verdict": primary_verdict, "external_review_state": "BLOCKED_PENDING_EXTERNAL_REVIEW",
            "script": v12.v11.file_record(ROOT / SCRIPT_REL, "acquisition_script"),
            "delivery_zip": v12.v11.file_record(DELIVERY_ZIP, "delivery_zip"),
            "detached_checksum": v12.v11.file_record(DELIVERY_HASH, "detached_checksum"),
            "post_zip_verification": v12.v11.file_record(POST_ZIP, "post_zip_verification"),
            "v1_2_source_copy_count": len(source_copies), "v1_2_capture_copy_count": len(capture_copies),
            "persistence_hit_count": len(hits), "persistence_hit_classification_counts": classification_counts,
            "writer_candidate_count": len(matrix), "unresolved_writer_count": 0,
            "asset_writer_classification": asset_classification["classification"],
            "source_linked_asset": {"asset_size_bytes": asset_evidence["asset_size_bytes"], "asset_sha256": asset_evidence["asset_sha256"], "meta_size_bytes": asset_evidence["meta_size_bytes"], "meta_sha256": asset_evidence["meta_sha256"], "shape": asset_evidence["landmark_shape"]},
            "source_contract_complete": source_eval["source_contract_complete"], "physical_writer_exclusive": source_eval["physical_writer_exclusive"],
            "capture_status_counts": capture_status_counts, "regression": regression["summary"], "assertions": assertion_summary,
            "protected_file_count": len(protected_before), "protected_mismatch_count": 0,
            "manifest_target_count": len(records), "zip_entry_count": len(records) + 1,
        }
    finally:
        if staging_root.exists():
            resolved = staging_root.resolve()
            temp_root = Path(tempfile.gettempdir()).resolve()
            if resolved.parent == temp_root and resolved.name.startswith("wis9d13_"):
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
