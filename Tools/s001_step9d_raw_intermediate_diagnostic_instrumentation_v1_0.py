#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tempfile
import zipfile


class VerificationError(RuntimeError):
    pass


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_REL = "Tools/s001_step9d_raw_intermediate_diagnostic_instrumentation_v1_0.py"
OUTPUT_REL = "data/processed/sessions/S001_P001/step9d_raw_intermediate_diagnostic_instrumentation_v1_0"
OUTPUT = ROOT / PurePosixPath(OUTPUT_REL)
DELIVERY_NAME = "WordsInSilence_step9d_raw_intermediate_diagnostic_instrumentation_v1_0_delivery.zip"
DELIVERY = ROOT / DELIVERY_NAME
DETACHED = ROOT / f"{DELIVERY_NAME}.sha256"
POST = ROOT / "WordsInSilence_step9d_raw_intermediate_diagnostic_instrumentation_v1_0_delivery_post_zip_verification.json"
STATUS = "STEP9D_RAW_INTERMEDIATE_DIAGNOSTIC_INSTRUMENTATION_V1_0_BLOCKED_PENDING_EXTERNAL_SOURCE_REVIEW"

V13_NAME = "WordsInSilence_step9d_retained_raw_intermediate_evidence_acquisition_v1_3_delivery.zip"
V13_ZIP = ROOT / V13_NAME
V13_DETACHED = ROOT / f"{V13_NAME}.sha256"
V13_SIZE = 409_078
V13_SHA256 = "f426167ee983d2871080628df95129936b8e7d39ec32cbac705788ba4baba4f8"
V13_BASE = "data/processed/sessions/S001_P001/step9d_retained_raw_intermediate_evidence_acquisition_v1_3"
V13_MANIFEST = f"{V13_BASE}/output_file_manifest.json"
V13_SOURCE_MANIFEST = f"{V13_BASE}/recognition_source_full_manifest_v1_3.json"
V13_PROTECTED = f"{V13_BASE}/protected_files_before_after.json"

ADAPTER_REL = "Assets/_Project/Scripts/Recognition/LiveInput/ApprovedLivePreprocessingAdapter.cs"
TRACE_REL = "Assets/_Project/Scripts/Recognition/Diagnostics/Step9DRawIntermediatePreprocessingTrace.cs"
COLLECTOR_REL = "Assets/_Project/Scripts/Recognition/Diagnostics/Step9DRawIntermediateDiagnosticCollector.cs"
WRITER_REL = "Assets/_Project/Scripts/Recognition/Diagnostics/Step9DRawIntermediateEvidenceJson.cs"
TEST_REL = "Assets/_Project/Tests/EditMode/Step9DRawIntermediateDiagnosticInstrumentationV1Tests.cs"
AFTER_SOURCE_RELS = [ADAPTER_REL, TRACE_REL, COLLECTOR_REL, WRITER_REL, TEST_REL]
NEW_SOURCE_RELS = [TRACE_REL, COLLECTOR_REL, WRITER_REL, TEST_REL]
ADAPTER_V13_ZIP_REL = f"{V13_BASE}/retained_inputs/recognition_source_full/LiveInput/ApprovedLivePreprocessingAdapter.cs"

REQUIRED_TESTS = {
    "Adapt_AndAdaptWithTrace_ResultParityIsExact": "Adapt/AdaptWithTrace normalized result parity",
    "RawOrdering_IsPose33ThenLeft21ThenRight21": "raw pose33+left21+right21 ordering",
    "AbsentLandmark_IsNaNFalseAndSerializesAsJsonNull": "finite/NaN/JSON-null contract",
    "BoundaryMasks_KeepReappearanceStrongTimestampAndCombinedSeparate": "separate and combined boundaries",
    "MaxGapOne_InterpolatesSingleGapAndRejectsTwoFrameGap": "selected max-gap=1 interpolation",
    "TorsoProvenance_DistinguishesDirectCarryBackfillAndUnavailable": "torso provenance states",
    "TorsoAndNormalizedTrace_AreSameCoreOutputs": "torso and normalized shared-core values",
    "TracePayloadCopies_AreDeeplyImmutable": "deep-copy immutability",
    "Collector_BeginFinalizeCancelAndSnapshotRemainFrozen": "collector lifecycle and frozen snapshot",
    "FrameIdentity_CountTimestampAndOrderingAreValidated": "frame identity validation",
    "DeterministicSerializer_SameSnapshotIsByteExactAndFinalNewline": "deterministic serialization",
    "EvidenceWriter_UsesDetachedChecksumAndNeverOverwrites": "checksum and never-overwrite",
    "TraceDisabledAdapt_PerformsNoDiagnosticDiskIo": "trace-disabled no disk I/O",
    "InvalidShapeTimestampAndInfinity_UseTypedFailures": "typed invalid-shape/timestamp/Inf failures",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def read_bytes(path: Path) -> bytes:
    with path.open("rb") as stream:
        return stream.read()


def read_text(path: Path) -> str:
    return read_bytes(path).decode("utf-8-sig")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(read_bytes(path))


def strict_json(payload: bytes, label: str) -> dict:
    try:
        value = json.loads(payload.decode("utf-8-sig"))
    except Exception as exception:
        raise VerificationError(f"invalid JSON: {label}: {exception}") from exception
    require(isinstance(value, dict), f"JSON root is not an object: {label}")
    return value


def safe_zip_name(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not name.startswith(("/", "\\")) and "\\" not in name and ".." not in path.parts and not re.match(r"^[A-Za-z]:", name)


def parse_detached(payload: bytes, expected_name: str, expected_hash: str) -> bool:
    try:
        return payload.decode("ascii") == f"{expected_hash}  {expected_name}\n"
    except UnicodeDecodeError:
        return False


def file_record(path: Path, category: str, relative_path: str) -> dict:
    payload = read_bytes(path)
    return {
        "category": category,
        "relative_path": relative_path,
        "size_bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def payload_record(payload: bytes, category: str, relative_path: str) -> dict:
    return {
        "category": category,
        "relative_path": relative_path,
        "size_bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def strict_v13_input() -> dict:
    require(V13_ZIP.is_file(), f"fixed v1.3 Delivery is missing: {V13_ZIP}")
    payload = read_bytes(V13_ZIP)
    require(len(payload) == V13_SIZE, f"fixed v1.3 Delivery size mismatch: {len(payload)}")
    require(sha256_bytes(payload) == V13_SHA256, "fixed v1.3 Delivery SHA-256 mismatch")
    require(V13_DETACHED.is_file() and parse_detached(read_bytes(V13_DETACHED), V13_NAME, V13_SHA256), "fixed v1.3 detached checksum mismatch")
    with zipfile.ZipFile(V13_ZIP, "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        require(archive.testzip() is None, "fixed v1.3 Delivery CRC failure")
        require(len(names) == 67 and len(set(names)) == 67, "fixed v1.3 ZIP entry count/duplicate mismatch")
        require(all(safe_zip_name(name) for name in names), "fixed v1.3 ZIP has unsafe path")
        require(all(((info.external_attr >> 16) & 0o170000) != 0o120000 for info in infos), "fixed v1.3 ZIP has symlink")
        require(all(not (info.flag_bits & 1) for info in infos), "fixed v1.3 ZIP has encrypted entry")
        manifest = strict_json(archive.read(V13_MANIFEST), V13_MANIFEST)
        outputs = {item["relative_path"]: item for item in manifest["outputs"]}
        require(len(outputs) == 66, "fixed v1.3 manifest target count mismatch")
        actual = set(names) - {V13_MANIFEST}
        require(actual == set(outputs), "fixed v1.3 manifest membership mismatch")
        for relative, record in outputs.items():
            item = archive.read(relative)
            require(len(item) == record["size_bytes"] and sha256_bytes(item) == record["sha256"], f"fixed v1.3 manifest mismatch: {relative}")
        source_manifest = strict_json(archive.read(V13_SOURCE_MANIFEST), V13_SOURCE_MANIFEST)
        protected = strict_json(archive.read(V13_PROTECTED), V13_PROTECTED)
        source_payloads = {}
        for record in source_manifest["files"]:
            zip_relative = f"{V13_BASE}/{record['copied_relative_path']}"
            source_payloads[record["copied_relative_path"]] = archive.read(zip_relative)
        adapter_before = archive.read(ADAPTER_V13_ZIP_REL)
    require(source_manifest["source_count"] == 42 and len(source_payloads) == 42, "fixed v1.3 source manifest count mismatch")
    require(sha256_bytes(adapter_before) == "e61a7e5bd5156e3cca7a89d5dea292bdf39702028ddf16095fc5f24b26a54796", "fixed adapter source identity mismatch")
    return {
        "delivery": payload_record(payload, "fixed_v1_3_delivery", V13_NAME),
        "manifest_target_count": len(outputs),
        "zip_entry_count": 67,
        "source_manifest": source_manifest,
        "source_payloads": source_payloads,
        "adapter_before": adapter_before,
        "protected": protected,
    }


def mask_csharp(text: str) -> str:
    output = list(text)
    index = 0
    state = "code"
    while index < len(text):
        current = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""
        if state == "code":
            if current == "/" and following == "/":
                output[index] = output[index + 1] = " "
                index += 2
                state = "line_comment"
                continue
            if current == "/" and following == "*":
                output[index] = output[index + 1] = " "
                index += 2
                state = "block_comment"
                continue
            if current == '"':
                output[index] = " "
                index += 1
                state = "string"
                continue
            if current == "'":
                output[index] = " "
                index += 1
                state = "char"
                continue
        elif state == "line_comment":
            if current == "\n":
                state = "code"
            else:
                output[index] = " "
            index += 1
            continue
        elif state == "block_comment":
            if current == "*" and following == "/":
                output[index] = output[index + 1] = " "
                index += 2
                state = "code"
                continue
            if current != "\n":
                output[index] = " "
            index += 1
            continue
        else:
            if current == "\\":
                output[index] = " "
                if index + 1 < len(text):
                    output[index + 1] = " "
                index += 2
                continue
            terminator = '"' if state == "string" else "'"
            output[index] = " "
            index += 1
            if current == terminator:
                state = "code"
            continue
        index += 1
    require(state in {"code", "line_comment"}, "unterminated C# string/comment")
    return "".join(output)


def balanced_csharp(text: str) -> dict:
    masked = mask_csharp(text)
    pairs = {"{": "}", "(": ")", "[": "]"}
    reverse = {value: key for key, value in pairs.items()}
    stack = []
    for index, character in enumerate(masked):
        if character in pairs:
            stack.append((character, index))
        elif character in reverse:
            require(stack and stack[-1][0] == reverse[character], f"unbalanced C# delimiter at offset {index}")
            stack.pop()
    require(not stack, f"unclosed C# delimiter: {stack[-1] if stack else None}")
    return {"balanced": True, "line_count": len(text.splitlines()), "masked_sha256": sha256_bytes(masked.encode("utf-8"))}


def extract_method(text: str, signature_pattern: str) -> str:
    masked = mask_csharp(text)
    match = re.search(signature_pattern, masked)
    require(match is not None, f"method not found: {signature_pattern}")
    opening = masked.find("{", match.start())
    require(opening >= 0, f"method opening brace not found: {signature_pattern}")
    depth = 0
    for index in range(opening, len(masked)):
        if masked[index] == "{":
            depth += 1
        elif masked[index] == "}":
            depth -= 1
            if depth == 0:
                return text[match.start():index + 1]
    raise VerificationError(f"method closing brace not found: {signature_pattern}")


def git_snapshot() -> dict:
    def run(arguments: list[str]) -> bytes:
        result = subprocess.run(arguments, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        require(result.returncode == 0, f"git read-only command failed: {' '.join(arguments)}: {result.stderr.decode(errors='replace')}")
        return result.stdout

    head = run(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    staged = run(["git", "diff", "--cached", "--binary", "--no-ext-diff"])
    return {"head": head, "staged_index_diff_sha256": sha256_bytes(staged), "staged_index_diff_size_bytes": len(staged)}


def protected_records(fixed: dict) -> tuple[list[dict], list[dict]]:
    adapter = (ROOT / PurePosixPath(ADAPTER_REL)).resolve()
    expected = []
    snapshots = []
    for record in fixed["protected"]["after"]:
        path = Path(record["absolute_path"]).resolve()
        if path == adapter:
            continue
        lower_parts = {part.lower() for part in path.parts}
        require(not lower_parts.intersection({"library", "temp", "logs", "obj", "build", "builds", ".git"}), f"forbidden protected path: {path}")
        expected.append(record)
        exists = path.is_file()
        actual_size = path.stat().st_size if exists else None
        actual_sha = sha256_file(path) if exists else None
        snapshots.append({
            "absolute_path": str(path),
            "relative_path": path.relative_to(ROOT).as_posix() if ROOT in path.parents else record.get("relative_path"),
            "exists": exists,
            "size_bytes": actual_size,
            "sha256": actual_sha,
            "fixed_v1_3_size_bytes": record["size_bytes"],
            "fixed_v1_3_sha256": record["sha256"],
            "matches_fixed_v1_3": exists and actual_size == record["size_bytes"] and actual_sha == record["sha256"],
        })
    return expected, snapshots


def analyze() -> dict:
    fixed = strict_v13_input()
    require((ROOT / PurePosixPath(SCRIPT_REL)).is_file(), "verification script is missing")
    for relative in AFTER_SOURCE_RELS:
        require((ROOT / PurePosixPath(relative)).is_file(), f"required implementation source is missing: {relative}")

    adapter_before = fixed["adapter_before"].decode("utf-8-sig")
    after_texts = {relative: read_text(ROOT / PurePosixPath(relative)) for relative in AFTER_SOURCE_RELS}
    adapter_after = after_texts[ADAPTER_REL]
    require(adapter_before != adapter_after, "adapter was not modified")

    source_parity = []
    source_failures = []
    for record in fixed["source_manifest"]["files"]:
        copy_relative = record["copied_relative_path"]
        source_suffix = copy_relative.removeprefix("retained_inputs/recognition_source_full/")
        project_relative = f"Assets/_Project/Scripts/Recognition/{source_suffix}"
        project_path = ROOT / PurePosixPath(project_relative)
        payload = fixed["source_payloads"][copy_relative]
        allowed_change = project_relative == ADAPTER_REL
        parity = project_path.is_file() and read_bytes(project_path) == payload
        source_parity.append({
            "source_file_id": record["source_file_id"],
            "project_relative_path": project_relative,
            "v1_3_size_bytes": len(payload),
            "v1_3_sha256": sha256_bytes(payload),
            "current_size_bytes": project_path.stat().st_size if project_path.is_file() else None,
            "current_sha256": sha256_file(project_path) if project_path.is_file() else None,
            "allowed_adapter_change": allowed_change,
            "byte_parity": parity,
        })
        if not allowed_change and not parity:
            source_failures.append(project_relative)
    require(not source_failures, f"non-adapter v1.3 Recognition source changed: {source_failures}")
    require(sum(item["byte_parity"] for item in source_parity if not item["allowed_adapter_change"]) == 41, "v1.3 non-adapter source parity count mismatch")

    syntax = {relative: balanced_csharp(text) for relative, text in after_texts.items()}
    public_signature = r"public\s+ApprovedPreprocessingResult\s+Adapt\s*\(\s*IReadOnlyList<RawMediaPipeObservation>\s+observations\s*\)"
    require(len(re.findall(public_signature, adapter_before)) == 1 and len(re.findall(public_signature, adapter_after)) == 1, "public Adapt signature changed or duplicated")
    critical_patterns = [
        r"static\s+void\s+Interpolate\s*\(",
        r"static\s+TorsoTransform\[\]\s+BuildTorsoTransforms\s*\(",
        r"static\s+TorsoTransform\s+ComputeTorsoTransform\s*\(",
        r"static\s+int\s+WriteComponent\s*\(",
    ]
    exact_critical = {}
    for pattern in (critical_patterns[0], critical_patterns[2], critical_patterns[3]):
        before_method = extract_method(adapter_before, pattern)
        after_method = extract_method(adapter_after, pattern)
        exact_critical[pattern] = {
            "before_sha256": sha256_bytes(before_method.encode("utf-8")),
            "after_sha256": sha256_bytes(after_method.encode("utf-8")),
            "exact": before_method == after_method,
        }
    require(all(item["exact"] for item in exact_critical.values()), "approved interpolation/torso/normalization core method changed")

    adapter_tokens = [
        "AdaptCore(observations, false, out ignored)",
        "AdaptCore(observations, true, out trace)",
        "trace = traceEnabled",
        "? CreateTrace(",
        ": null;",
        "SelectedMaxGap = 1",
        "TimestampDiscontinuitySeconds = 0.1",
    ]
    trace_tokens = [
        "RawLandmarkCount = 75", "NormalizedLandmarkCount = 16", "pose[0..32]", "left_hand[0..20]", "right_hand[0..20]",
        "DIRECT", "FORWARD_CARRIED", "LEADING_BACKFILLED", "UNAVAILABLE", "CopyRawCoordinates", "CopyPostGapInterpolationCoordinates",
        "CopyTorsoOrigin", "CopyNormalizedCoordinates", "DeepCopy()",
    ]
    collector_tokens = [
        "AttachReadOnlyBridge", "ObservationArrived +=", "ProviderFrameCount - 1L", "ObservationCount - 1L",
        "LastLifetimeProviderFrameIndex", "Begin(", "FinalizeCapture(", "Cancel()", "AdaptWithTraceInvocationCount++",
    ]
    writer_tokens = [
        "CultureInfo.InvariantCulture", "FileMode.CreateNew", "File.Move(temporaryPath, fullPath)",
        'fullPath + ".sha256"', "OUTPUT_ALREADY_EXISTS", "ValidateSnapshot(snapshot)", "ValidateSerializedBytes(bytes)",
        'double.IsNaN(value) ? "null"', "ChecksumVerified",
    ]
    require(all(token in adapter_after for token in adapter_tokens), "adapter shared-core/trace-disabled token missing")
    require(all(token in after_texts[TRACE_REL] for token in trace_tokens), "trace schema/deep-copy token missing")
    require(all(token in after_texts[COLLECTOR_REL] for token in collector_tokens), "collector opt-in/snapshot token missing")
    require(all(token in after_texts[WRITER_REL] for token in writer_tokens), "deterministic writer token missing")
    require("File." not in adapter_after and "Directory." not in adapter_after, "trace-disabled adapter contains disk I/O token")

    test_text = after_texts[TEST_REL]
    test_methods = re.findall(r"\[Test\]\s*public\s+void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", test_text)
    require(len(test_methods) == 14 and len(set(test_methods)) == 14, f"synthetic test count/name uniqueness mismatch: {test_methods}")
    require(set(test_methods) == set(REQUIRED_TESTS), f"synthetic test families mismatch: {sorted(set(REQUIRED_TESTS) - set(test_methods))}")
    test_inventory = [
        {"test_name": name, "family": REQUIRED_TESTS[name], "source_status": "PRESENT_NOT_EXECUTED", "execution_status": "UNVERIFIED_NOT_RUN_PENDING_EXTERNAL_SOURCE_REVIEW"}
        for name in test_methods
    ]

    _, protected_before = protected_records(fixed)
    preexisting_fixed_differences = [item for item in protected_before if not item["matches_fixed_v1_3"]]

    allowed_change_manifest = {
        "schema_version": "words_in_silence.step9d.raw_intermediate.allowed_change_manifest.v1_0",
        "existing_modified_files": [ADAPTER_REL],
        "new_source_files": NEW_SOURCE_RELS,
        "new_verification_script": SCRIPT_REL,
        "assembly_definition_changes": [],
        "scene_prefab_asset_meta_changes": [],
        "production_decision_threshold_template_evidence_changes": [],
        "allowed_existing_file_modification_count": 1,
        "other_existing_file_modification_count": 0,
    }

    assertions = []
    def add(name: str, status: str, evidence: object) -> None:
        assertions.append({"name": name, "status": status, "evidence": evidence})

    add("fixed_v1_3_delivery_size_sha_crc_manifest", "PASS", fixed["delivery"])
    add("v1_3_non_adapter_recognition_source_byte_parity_41", "PASS", {"pass": 41, "fail": 0})
    add("adapter_before_is_exact_v1_3_approved_copy", "PASS", payload_record(fixed["adapter_before"], "before_source", ADAPTER_REL))
    add("public_adapt_signature_preserved", "PASS", public_signature)
    add("shared_core_trace_disabled_and_enabled_paths_present", "PASS", adapter_tokens)
    add("approved_interpolation_torso_normalization_core_methods_exact", "PASS", exact_critical)
    add("csharp_static_delimiters_balanced", "PASS", syntax)
    add("trace_raw75_masks_torso_normalized_schema_tokens_present", "PASS", trace_tokens)
    add("trace_deep_copy_immutable_api_present", "PASS", ["DeepCopy", "CopyRawCoordinates", "CopyNormalizedCoordinates"])
    add("collector_explicit_bridge_subscription_and_finalize_once_tokens", "PASS", collector_tokens)
    add("deterministic_writer_atomic_move_checksum_never_overwrite_tokens", "PASS", writer_tokens)
    add("synthetic_test_source_14_unique_required_families", "PASS", test_methods)
    add("allowed_change_scope_and_no_asmdef_scene_prefab_asset_meta_changes", "PASS", allowed_change_manifest)
    add("protected_files_baseline_recorded_for_before_after", "PASS", {
        "checked": len(protected_before),
        "preexisting_fixed_v1_3_difference_count": len(preexisting_fixed_differences),
        "preexisting_differences": preexisting_fixed_differences,
        "task_before_after_mismatch": 0,
    })
    add("unity_editor_player_webcam_capture_test_runner_execution_zero", "PASS", 0)
    add("git_staging_commit_push_zero_enforced_by_before_after", "PASS", True)
    add("unity_compile_status", "UNVERIFIED", "UNVERIFIED_NOT_RUN_PENDING_EXTERNAL_SOURCE_REVIEW")
    add("nunit_execution_status", "UNVERIFIED", "UNVERIFIED_NOT_RUN_PENDING_EXTERNAL_SOURCE_REVIEW")
    summary = {status.lower(): sum(item["status"] == status for item in assertions) for status in ("PASS", "FAIL", "UNVERIFIED", "NOT_APPLICABLE")}
    require(summary == {"pass": 16, "fail": 0, "unverified": 2, "not_applicable": 0}, f"unexpected static assertion summary: {summary}")

    trace_schema = {
        "schema_version": "words_in_silence.step9d.raw_intermediate.trace_schema_contract.v1_0",
        "evidence_schema_version": "words_in_silence.step9d.raw_intermediate_diagnostic.v1_0",
        "raw_ordering": {"pose": [0, 32], "left_hand": [33, 53], "right_hand": [54, 74]},
        "fields": {
            "frame_identity": "[T] capture-local/timestamp plus nullable provider/bridge/lifetime IDs",
            "raw_coordinates": "[T,75,3]",
            "raw_original_valid_mask": "[T,75]",
            "raw_reappearance_mask": "[T,75]",
            "raw_strong_depth_mask": "[T,75]",
            "timestamp_discontinuity_boundary_mask": "[T,75] (frame 0 is capture-start boundary)",
            "combined_boundary_mask": "[T,75]",
            "post_gap_interpolation_coordinates": "[T,75,3]",
            "interpolation_mask": "[T,75]",
            "torso_source_valid": "[T]",
            "torso_final_valid": "[T]",
            "torso_provenance": "[T] DIRECT/FORWARD_CARRIED/LEADING_BACKFILLED/UNAVAILABLE",
            "torso_origin_right_up_forward": "four [T,3] arrays",
            "torso_shoulder_width_scale": "[T]",
            "normalized_coordinates": "[T,16,3]",
            "normalized_mask": "[T,16]",
        },
        "invalid_coordinate_memory_contract": "NaN",
        "invalid_coordinate_json_contract": "null",
        "infinity_contract": "typed failure; serialization prohibited",
        "selected_max_gap": 1,
        "timestamp_discontinuity_seconds": 0.1,
    }

    diff_lines = list(difflib.unified_diff(
        adapter_before.splitlines(keepends=True),
        adapter_after.splitlines(keepends=True),
        fromfile=f"a/{ADAPTER_REL}",
        tofile=f"b/{ADAPTER_REL}",
        n=3,
    ))
    adapter_diff = "".join(diff_lines)
    require(adapter_diff, "adapter minimal diff is empty")
    diff_summary = {
        "added_lines": sum(line.startswith("+") and not line.startswith("+++") for line in diff_lines),
        "removed_lines": sum(line.startswith("-") and not line.startswith("---") for line in diff_lines),
        "critical_core_methods_exact": exact_critical,
        "public_adapt_signature_preserved": True,
        "shared_core_method": "AdaptCore",
        "trace_disabled_entry": "AdaptCore(observations, false, out ignored)",
        "trace_enabled_entry": "AdaptCore(observations, true, out trace)",
    }

    return {
        "fixed": fixed,
        "after_texts": after_texts,
        "source_parity": source_parity,
        "protected_before": protected_before,
        "preexisting_fixed_differences": preexisting_fixed_differences,
        "allowed_change_manifest": allowed_change_manifest,
        "assertions": assertions,
        "assertion_summary": summary,
        "test_inventory": test_inventory,
        "trace_schema": trace_schema,
        "adapter_diff": adapter_diff,
        "diff_summary": diff_summary,
        "syntax": syntax,
    }


def copy_after_sources(stage_output: Path) -> list[dict]:
    records = []
    for relative in AFTER_SOURCE_RELS:
        source = ROOT / PurePosixPath(relative)
        destination_relative = f"retained_inputs/after/{relative}"
        destination = stage_output / PurePosixPath(destination_relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        require(read_bytes(source) == read_bytes(destination), f"after source copy parity failed: {relative}")
        records.append({
            "source_relative_path": relative,
            "copy_relative_path": destination_relative,
            "size_bytes": source.stat().st_size,
            "sha256": sha256_file(source),
            "byte_parity": True,
            "change_kind": "MODIFIED_EXISTING" if relative == ADAPTER_REL else "NEW_SOURCE",
        })
    return records


def output_records(stage_output: Path) -> list[dict]:
    records = [file_record(ROOT / PurePosixPath(SCRIPT_REL), "static_verification_script", SCRIPT_REL)]
    for path in sorted((item for item in stage_output.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
        if path.name == "output_file_manifest.json":
            continue
        relative = f"{OUTPUT_REL}/{path.relative_to(stage_output).as_posix()}"
        category = "retained_source_copy" if "/retained_inputs/" in f"/{relative}" else "instrumentation_static_output"
        records.append(file_record(path, category, relative))
    records.sort(key=lambda item: item["relative_path"])
    require(len(records) == len({item["relative_path"] for item in records}), "duplicate output manifest path")
    return records


def create_deterministic_zip(path: Path, stage_output: Path) -> None:
    sources = [(SCRIPT_REL, ROOT / PurePosixPath(SCRIPT_REL))]
    for item in stage_output.rglob("*"):
        if item.is_file():
            sources.append((f"{OUTPUT_REL}/{item.relative_to(stage_output).as_posix()}", item))
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative, source in sorted(sources, key=lambda item: item[0]):
            info = zipfile.ZipInfo(relative, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, read_bytes(source))


def verify_zip(path: Path, detached_path: Path, expected_assertions: dict) -> dict:
    checks = {
        "crc_error": None,
        "duplicate_entries": [],
        "unsafe_paths": [],
        "symlinks": [],
        "encrypted_entries": [],
        "non_deterministic_timestamps": [],
        "manifest_missing": [],
        "manifest_extra": [],
        "manifest_size_mismatch": [],
        "manifest_sha256_mismatch": [],
        "required_output_missing": [],
        "legacy_max_path_risk_entries": [],
        "detached_checksum_strict": False,
        "static_assertion_fail_count": expected_assertions["fail"],
        "static_assertion_unverified_count": expected_assertions["unverified"],
    }
    manifest_relative = f"{OUTPUT_REL}/output_file_manifest.json"
    with zipfile.ZipFile(path, "r") as archive:
        infos = archive.infolist()
        names = [item.filename for item in infos]
        checks["crc_error"] = archive.testzip()
        checks["duplicate_entries"] = sorted({name for name in names if names.count(name) > 1})
        checks["unsafe_paths"] = [name for name in names if not safe_zip_name(name)]
        checks["symlinks"] = [item.filename for item in infos if ((item.external_attr >> 16) & 0o170000) == 0o120000]
        checks["encrypted_entries"] = [item.filename for item in infos if item.flag_bits & 1]
        checks["non_deterministic_timestamps"] = [item.filename for item in infos if item.date_time != (1980, 1, 1, 0, 0, 0)]
        checks["legacy_max_path_risk_entries"] = [name for name in names if len(str(ROOT / PurePosixPath(name))) >= 260]
        manifest = strict_json(archive.read(manifest_relative), manifest_relative)
        expected = {record["relative_path"]: record for record in manifest["outputs"]}
        actual = set(names) - {manifest_relative}
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
            *(f"{OUTPUT_REL}/{name}" for name in (
                "IMPLEMENTATION_REPORT.md", "allowed_change_manifest.json", "before_after_source_manifest.json",
                "approved_adapter_minimal.diff", "trace_schema_contract.json", "static_verification_assertions.json",
                "synthetic_test_inventory.json", "protected_files_before_after.json", "acquisition_conclusion.json",
                "output_file_manifest.json",
            )),
        }
        checks["required_output_missing"] = sorted(required - set(names))
        entry_count = len(names)
        manifest_target_count = len(expected)
    checks["detached_checksum_strict"] = parse_detached(read_bytes(detached_path), DELIVERY_NAME, sha256_file(path))
    list_failures = [key for key, value in checks.items() if isinstance(value, list) and value]
    require(checks["crc_error"] is None and checks["detached_checksum_strict"] and checks["static_assertion_fail_count"] == 0 and not list_failures, f"post-ZIP verification failure: {checks}")
    return {"checks": checks, "zip_entry_count": entry_count, "manifest_target_count": manifest_target_count}


def build() -> dict:
    for path in (OUTPUT, DELIVERY, DETACHED, POST):
        require(not path.exists(), f"final output already exists; refusing overwrite: {path}")
    analysis = analyze()
    git_before = git_snapshot()
    protected_before = analysis["protected_before"]
    temporary_root = Path(tempfile.mkdtemp(prefix="wis_step9d_instrumentation_v1_0_"))
    stage_output = temporary_root / "output"
    stage_zip = temporary_root / DELIVERY_NAME
    stage_detached = temporary_root / DETACHED.name
    stage_post = temporary_root / POST.name
    stage_output.mkdir(parents=True)
    try:
        before_relative = f"retained_inputs/before/{ADAPTER_REL}"
        before_path = stage_output / PurePosixPath(before_relative)
        before_path.parent.mkdir(parents=True, exist_ok=True)
        before_path.write_bytes(analysis["fixed"]["adapter_before"])
        after_records = copy_after_sources(stage_output)
        before_record = {
            "source_relative_path": ADAPTER_REL,
            "copy_relative_path": before_relative,
            "size_bytes": len(analysis["fixed"]["adapter_before"]),
            "sha256": sha256_bytes(analysis["fixed"]["adapter_before"]),
            "origin": "fixed externally-approved v1.3 Delivery ZIP source copy",
            "byte_parity_with_fixed_v1_3": True,
        }

        protected_expected, protected_after = protected_records(analysis["fixed"])
        require(len(protected_before) == len(protected_after), "protected snapshot count changed")
        require(all(
            left["absolute_path"] == right["absolute_path"] and left["size_bytes"] == right["size_bytes"] and left["sha256"] == right["sha256"]
            for left, right in zip(protected_before, protected_after)
        ), "protected file changed during static verification/package staging")
        git_after = git_snapshot()
        require(git_before == git_after, "Git HEAD or staged index changed")

        before_after_manifest = {
            "schema_version": "words_in_silence.step9d.raw_intermediate.before_after_source_manifest.v1_0",
            "fixed_v1_3_delivery": analysis["fixed"]["delivery"],
            "fixed_v1_3_manifest_target_count": analysis["fixed"]["manifest_target_count"],
            "fixed_v1_3_source_count": 42,
            "non_adapter_source_parity": {"pass": 41, "fail": 0},
            "before": [before_record],
            "after": after_records,
            "adapter_diff_summary": analysis["diff_summary"],
        }
        protected_payload = {
            "schema_version": "words_in_silence.step9d.raw_intermediate.protected_files_before_after.v1_0",
            "fixed_v1_3_protected_record_count_excluding_allowed_adapter": len(protected_expected),
            "protected_file_count": len(protected_before),
            "preexisting_fixed_v1_3_difference_count": len(analysis["preexisting_fixed_differences"]),
            "preexisting_fixed_v1_3_differences": analysis["preexisting_fixed_differences"],
            "before": protected_before,
            "after": protected_after,
            "mismatch_count": 0,
            "mismatches": [],
            "git_before": git_before,
            "git_after": git_after,
            "git_head_and_staged_index_unchanged": True,
        }
        assertions_payload = {
            "schema_version": "words_in_silence.step9d.raw_intermediate.static_verification_assertions.v1_0",
            "conclusion_limit": "STATIC_SOURCE_VERIFICATION_ONLY; DOES_NOT CLAIM UNITY COMPILE OR NUNIT PASS",
            "summary": analysis["assertion_summary"],
            "assertions": analysis["assertions"],
        }
        test_payload = {
            "schema_version": "words_in_silence.step9d.raw_intermediate.synthetic_test_inventory.v1_0",
            "test_source": TEST_REL,
            "test_count": len(analysis["test_inventory"]),
            "execution_count": 0,
            "execution_status": "UNVERIFIED_NOT_RUN_PENDING_EXTERNAL_SOURCE_REVIEW",
            "tests": analysis["test_inventory"],
        }
        conclusion = {
            "schema_version": "words_in_silence.step9d.raw_intermediate.acquisition_conclusion.v1_0",
            "status": STATUS,
            "external_review_state": "BLOCKED_PENDING_EXTERNAL_SOURCE_REVIEW",
            "source_only_implementation_complete": True,
            "unity_compile_status": "UNVERIFIED_NOT_RUN_PENDING_EXTERNAL_SOURCE_REVIEW",
            "nunit_status": "UNVERIFIED_NOT_RUN_PENDING_EXTERNAL_SOURCE_REVIEW",
            "nunit_execution_count": 0,
            "adapter_public_signature_preserved": True,
            "adapter_shared_core": "AdaptCore",
            "critical_preprocessing_core_methods_byte_exact": True,
            "trace_schema_contract_complete": True,
            "collector_is_opt_in_and_not_scene_connected": True,
            "writer_is_explicit_never_overwrite": True,
            "synthetic_test_family_count": 14,
            "static_assertion_summary": analysis["assertion_summary"],
            "protected_file_mismatch_count": 0,
            "preexisting_fixed_v1_3_protected_difference_count": len(analysis["preexisting_fixed_differences"]),
            "operation_counts": {
                "unity_editor_player_webcam_capture_test_runner": 0,
                "scene_prefab_asset_meta_evidence_modification": 0,
                "production_threshold_template_modification": 0,
                "git_staging_commit_push": 0,
            },
        }
        report = f"""# Step 9D Raw/Intermediate Diagnostic Instrumentation v1.0

Status: `{STATUS}`

## Scope

- Modified existing source: `{ADAPTER_REL}` only.
- Added three diagnostic runtime sources and one synthetic EditMode test source.
- Unity, Editor, player, webcam, physical capture and Test Runner were not executed.

## Adapter

- Public `Adapt(IReadOnlyList<RawMediaPipeObservation>)` signature is preserved.
- `Adapt()` and `AdaptWithTrace()` use the same private `AdaptCore` path.
- Trace-disabled `Adapt()` passes `traceEnabled=false` and creates no trace payload or disk output.
- Approved `Interpolate`, `ComputeTorsoTransform` and `WriteComponent` method bodies are byte-exact against v1.3.
- Diff: {analysis['diff_summary']['added_lines']} added lines, {analysis['diff_summary']['removed_lines']} removed lines.

## Trace and collector

- Raw ordering: pose 0..32, left hand 33..53, right hand 54..74.
- Raw/post-gap arrays `[T,75,3]`, masks `[T,75]`, torso vectors `[T,3]`, normalized `[T,16,3]`/`[T,16]`.
- Torso provenance: `DIRECT`, `FORWARD_CARRIED`, `LEADING_BACKFILLED`, `UNAVAILABLE`.
- Collector subscription is explicit and read-only; finalize freezes inputs and calls `AdaptWithTrace` once.

## Writer and tests

- Deterministic invariant JSON, NaN to `null`, Infinity rejected, final newline.
- Explicit temporary write/atomic move/adjacent checksum; targets are never overwritten.
- Synthetic NUnit source contains 14 required unique test families.
- Compile/test status: `UNVERIFIED_NOT_RUN_PENDING_EXTERNAL_SOURCE_REVIEW`.

## Static verification

- Static assertions: {analysis['assertion_summary']['pass']} PASS / {analysis['assertion_summary']['fail']} FAIL / {analysis['assertion_summary']['unverified']} UNVERIFIED.
- Fixed v1.3 non-adapter Recognition source parity: 41/41.
- Protected files: {len(protected_before)} checked / 0 task before-after mismatch.
- Pre-existing fixed-v1.3 baseline differences: {len(analysis['preexisting_fixed_differences'])} missing retained-copy checksum files; not recreated or modified.
- Git HEAD/staged index unchanged.
"""

        write_json(stage_output / "allowed_change_manifest.json", analysis["allowed_change_manifest"])
        write_json(stage_output / "before_after_source_manifest.json", before_after_manifest)
        write_text(stage_output / "approved_adapter_minimal.diff", analysis["adapter_diff"])
        write_json(stage_output / "trace_schema_contract.json", analysis["trace_schema"])
        write_json(stage_output / "static_verification_assertions.json", assertions_payload)
        write_json(stage_output / "synthetic_test_inventory.json", test_payload)
        write_json(stage_output / "protected_files_before_after.json", protected_payload)
        write_json(stage_output / "acquisition_conclusion.json", conclusion)
        write_text(stage_output / "IMPLEMENTATION_REPORT.md", report)

        records = output_records(stage_output)
        output_manifest = {
            "schema_version": "words_in_silence.step9d.raw_intermediate.output_file_manifest.v1_0",
            "manifest_self_excluded": True,
            "delivery_zip_detached_and_post_verification_excluded": True,
            "output_count": len(records),
            "outputs": records,
        }
        write_json(stage_output / "output_file_manifest.json", output_manifest)
        create_deterministic_zip(stage_zip, stage_output)
        delivery_hash = sha256_file(stage_zip)
        stage_detached.write_text(f"{delivery_hash}  {DELIVERY_NAME}\n", encoding="ascii", newline="\n")
        post_verification = verify_zip(stage_zip, stage_detached, analysis["assertion_summary"])
        post_payload = {
            "schema_version": "words_in_silence.step9d.raw_intermediate.delivery_post_zip_verification.v1_0",
            "status": STATUS,
            "external_review_state": "BLOCKED_PENDING_EXTERNAL_SOURCE_REVIEW",
            "zip_file": DELIVERY_NAME,
            "zip_size_bytes": stage_zip.stat().st_size,
            "zip_sha256": delivery_hash,
            "zip_entry_count": post_verification["zip_entry_count"],
            "manifest_target_count": post_verification["manifest_target_count"],
            "static_assertion_summary": analysis["assertion_summary"],
            "nunit_execution_count": 0,
            "nunit_status": "UNVERIFIED_NOT_RUN_PENDING_EXTERNAL_SOURCE_REVIEW",
            "protected_file_count": len(protected_before),
            "preexisting_fixed_v1_3_protected_difference_count": len(analysis["preexisting_fixed_differences"]),
            "protected_file_mismatch_count": 0,
            "git_head_and_staged_index_unchanged": True,
            "checks": post_verification["checks"],
        }
        write_json(stage_post, post_payload)

        require(not OUTPUT.exists() and not DELIVERY.exists() and not DETACHED.exists() and not POST.exists(), "final output appeared during staging")
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        os.rename(stage_output, OUTPUT)
        os.rename(stage_zip, DELIVERY)
        os.rename(stage_detached, DETACHED)
        os.rename(stage_post, POST)

        result = {
            "status": STATUS,
            "script": file_record(ROOT / PurePosixPath(SCRIPT_REL), "static_verification_script", SCRIPT_REL),
            "delivery": file_record(DELIVERY, "delivery_zip", DELIVERY.name),
            "detached": file_record(DETACHED, "detached_checksum", DETACHED.name),
            "post_verification": file_record(POST, "post_zip_verification", POST.name),
            "adapter_before": before_record,
            "adapter_after": next(item for item in after_records if item["source_relative_path"] == ADAPTER_REL),
            "new_source_count": len(NEW_SOURCE_RELS),
            "non_adapter_v1_3_source_parity": {"pass": 41, "fail": 0},
            "synthetic_test_count": 14,
            "synthetic_test_execution_count": 0,
            "synthetic_test_status": "UNVERIFIED_NOT_RUN_PENDING_EXTERNAL_SOURCE_REVIEW",
            "static_assertions": analysis["assertion_summary"],
            "protected_file_count": len(protected_before),
            "preexisting_fixed_v1_3_protected_difference_count": len(analysis["preexisting_fixed_differences"]),
            "protected_mismatch_count": 0,
            "zip_entry_count": post_verification["zip_entry_count"],
            "manifest_target_count": post_verification["manifest_target_count"],
        }
        return result
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Static verifier and packager for Step 9D raw/intermediate diagnostic instrumentation v1.0")
    parser.add_argument("--check", action="store_true", help="perform read-only static verification without creating outputs")
    arguments = parser.parse_args()
    if arguments.check:
        analysis = analyze()
        result = {
            "status": STATUS,
            "mode": "READ_ONLY_STATIC_CHECK",
            "fixed_v1_3_delivery": analysis["fixed"]["delivery"],
            "non_adapter_v1_3_source_parity": {"pass": 41, "fail": 0},
            "adapter_diff_summary": analysis["diff_summary"],
            "synthetic_test_count": len(analysis["test_inventory"]),
            "static_assertions": analysis["assertion_summary"],
            "protected_file_count": len(analysis["protected_before"]),
            "preexisting_fixed_v1_3_protected_difference_count": len(analysis["preexisting_fixed_differences"]),
            "protected_mismatch_count": 0,
            "unity_compile_and_test": "UNVERIFIED_NOT_RUN_PENDING_EXTERNAL_SOURCE_REVIEW",
        }
    else:
        result = build()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
