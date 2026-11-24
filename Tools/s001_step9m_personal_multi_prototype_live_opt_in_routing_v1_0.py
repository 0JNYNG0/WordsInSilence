#!/usr/bin/env python3
"""Validate and package Step 9M personal multi-prototype live opt-in routing v1.0."""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import importlib.util
import io
import json
import re
import struct
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA = "words_in_silence.step9m.personal_multi_prototype_live_opt_in_routing.v1_0"
RESULT_NAME = "WordsInSilence_step9m_personal_multi_prototype_live_opt_in_routing_v1_0"
DELIVERY_NAME = RESULT_NAME + "_delivery.zip"
HARNESS = Path("Assets/_Project/Scripts/Recognition/LiveInput/PersonalLiveRecognitionHarness.cs")
FACTORY = Path("Assets/_Project/Scripts/Recognition/LiveInput/ApprovedPersonalMultiPrototypeLiveRuntimeFactory.cs")
TEST = Path("Assets/_Project/Tests/EditMode/PersonalMultiPrototypeLiveOptInRoutingV1Tests.cs")
EXISTING_LIVE_TEST = Path("Assets/_Project/Tests/EditMode/PersonalLiveRecognitionIntegrationV1Tests.cs")
PREFAB = Path("Assets/_Project/Resources/Diagnostics/PersonalMultiPrototypeLiveRecognitionTechnical.prefab")
BEFORE_HARNESS = Path("Validation/WordsInSilence_step9k_primary_failure_preservation_diagnostic_v1_0/source_after/PersonalLiveRecognitionHarness.cs")
BEFORE_EXISTING_LIVE_TEST = Path("Validation/WordsInSilence_step9k_primary_failure_preservation_diagnostic_v1_0/source_after/PersonalLiveRecognitionIntegrationV1Tests.cs")
INSTALL_EVIDENCE = Path("Validation/WordsInSilence_step9l_personal_multi_prototype_runtime_candidate_v1_1_unity_install")
GATE_EVIDENCE = Path("Validation/WordsInSilence_step9l_personal_multi_prototype_runtime_candidate_v1_1_unity_editor_gate_verification_v1_0")
STEP9K_RESULT = Path("Validation/WordsInSilence_step9k_personal_live_recognition_technical_prefab_v1_0")
EVIDENCE_DIR = Path("Validation/WordsInSilence_step9m_personal_multi_prototype_live_opt_in_routing_v1_0_unity_editor_gate")
XML_NAME = "Step9M_TestResults.xml"
SCREEN_NAME = "Step9M_TestRunner_full.png"
LOG_NAME = "Step9M_Editor_full.log"
FIRST_FAIL_XML_NAME = "TestResults_20260810_132122.xml"
FIRST_FAIL_SCREEN_NAME = "Step9M_FirstRun_37PASS_1FAIL.png"
FIXTURES = {
    "WordsInSilence.Tests.EditMode.PersonalMultiPrototypeLiveOptInRoutingV1Tests": 8,
    "WordsInSilence.Tests.EditMode.PersonalLiveRecognitionIntegrationV1Tests": 22,
    "WordsInSilence.Tests.EditMode.PersonalMultiPrototypeRuntimeIntegrationV1Tests": 8,
}
FIXED_INPUTS = (
    ("Validation/WordsInSilence_step9l_personal_multi_prototype_runtime_candidate_v1_1_delivery.zip", 765_962,
     "d3beede4797c4c70581b4a1f3f88af2dac25df57018f5da87cd4300b0e69bca6"),
    ("Validation/WordsInSilence_step9l_personal_multi_prototype_runtime_candidate_v1_1_unity_editor_gate_verification_v1_0_delivery.zip", 353_952,
     "3d6a7565bc5cd90fcf665c6b162f701e89ccd44b1791f8b459a8942c1fe193d7"),
)


class Step9MError(RuntimeError):
    pass


class InjectedPostValidationFailure(Step9MError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Step9MError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_record(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    data = path.read_bytes()
    return {"relative_path": relative.replace("\\", "/"), "size_bytes": len(data), "sha256": sha256_bytes(data)}


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixed_inputs(root: Path) -> list[dict[str, Any]]:
    rows = []
    for relative, size, digest in FIXED_INPUTS:
        row = file_record(root, relative)
        require(row["size_bytes"] == size and row["sha256"] == digest, f"fixed input mismatch: {relative}")
        rows.append(row)
    return rows


def build_project_mapping(root: Path) -> tuple[dict[str, bytes], list[dict[str, Any]], bytes]:
    before = (root / BEFORE_HARNESS).read_bytes()
    require(len(before) == 10_105 and sha256_bytes(before) == "4cb82dddbb682f465ef0905cae2a34418b7a64f3c66a3f19707d6dcd6a317097",
            "approved harness before source mismatch")
    before_existing_test = (root / BEFORE_EXISTING_LIVE_TEST).read_bytes()
    require(len(before_existing_test) == 36_212 and sha256_bytes(before_existing_test) == "6b3589de3d778d66117cdbcfae29e14a416d49332a8bfc3a429d6412a6821be3",
            "approved existing live test before source mismatch")
    paths = (
        HARNESS,
        EXISTING_LIVE_TEST,
        FACTORY,
        Path(str(FACTORY) + ".meta"),
        TEST,
        Path(str(TEST) + ".meta"),
        PREFAB,
        Path(str(PREFAB) + ".meta"),
    )
    payload: dict[str, bytes] = {}
    rows: list[dict[str, Any]] = []
    for path in paths:
        relative = path.as_posix()
        body = (root / path).read_bytes()
        payload[relative] = body
        row = {
            "project_relative_path": relative,
            "operation": "replace_approved_source" if path in (HARNESS, EXISTING_LIVE_TEST) else "create_never_overwrite",
            "size_bytes": len(body),
            "sha256": sha256_bytes(body),
        }
        if path in (HARNESS, EXISTING_LIVE_TEST):
            approved_before = before if path == HARNESS else before_existing_test
            row.update({"before_size_bytes": len(approved_before), "before_sha256": sha256_bytes(approved_before), "before_hex": approved_before.hex()})
        rows.append(row)
    require(sum(row["operation"] == "replace_approved_source" for row in rows) == 2, "replacement count mismatch")
    require(sum(row["operation"] == "create_never_overwrite" for row in rows) == 6, "new-file count mismatch")
    return payload, rows, before


def source_contract(root: Path, before: bytes, payload: dict[str, bytes]) -> tuple[dict[str, Any], bytes, bytes, bytes]:
    after = payload[HARNESS.as_posix()]
    factory = payload[FACTORY.as_posix()].decode("utf-8")
    harness = after.decode("utf-8")
    before_existing_test = (root / BEFORE_EXISTING_LIVE_TEST).read_bytes()
    after_existing_test = payload[EXISTING_LIVE_TEST.as_posix()]
    test_text = after_existing_test.decode("utf-8")
    checks = {
        "legacy_enum_zero": "LegacySingleMedoid = 0" in factory,
        "candidate_enum_one": "MultiPrototypeCandidate = 1" in factory,
        "candidate_activation_api": "ActivatePersonalMultiPrototypeProfile" in factory,
        "approved_global_loader_reused": "approvedLegacyFactory.LoadGlobal(paths)" in factory,
        "legacy_default_serialized": "profileMode = PersonalLiveProfileMode.LegacySingleMedoid" in harness,
        "candidate_factory_explicit_only": "new ApprovedPersonalMultiPrototypeLiveRuntimeFactory()" in harness,
        "candidate_directory_fixed": "ApprovedPersonalMultiPrototypeLiveRuntimeFactory.ProfileDirectoryName" in harness,
        "legacy_directory_preserved": '"step9h_current_user_v1_0"' in harness,
        "no_dtw_implementation": "MaskAwareDtw" not in factory and "MaskAwareDtw" not in harness,
        "approved_step9l_runtime_hash_baseline":
            'const string Step9IProfileHash = "468613729c0299d11a4c72343c17df0c4fc7ed5b36367c8ce86ea9d81b6b62cf";' in test_text,
    }
    require(all(checks.values()), "source routing contract mismatch")
    diff = "".join(difflib.unified_diff(
        before.decode("utf-8").splitlines(keepends=True),
        after.decode("utf-8").splitlines(keepends=True),
        fromfile="PersonalLiveRecognitionHarness.cs.before",
        tofile="PersonalLiveRecognitionHarness.cs.after",
    )).encode("utf-8")
    test_diff = "".join(difflib.unified_diff(
        before_existing_test.decode("utf-8").splitlines(keepends=True),
        after_existing_test.decode("utf-8").splitlines(keepends=True),
        fromfile="PersonalLiveRecognitionIntegrationV1Tests.cs.before",
        tofile="PersonalLiveRecognitionIntegrationV1Tests.cs.after",
    )).encode("utf-8")
    return {
        "schema_version": SCHEMA + ".source_contract",
        "checks": checks,
        "before": {"size_bytes": len(before), "sha256": sha256_bytes(before)},
        "after": {"size_bytes": len(after), "sha256": sha256_bytes(after)},
        "existing_live_test_before": {"size_bytes": len(before_existing_test), "sha256": sha256_bytes(before_existing_test)},
        "existing_live_test_after": {"size_bytes": len(after_existing_test), "sha256": sha256_bytes(after_existing_test)},
        "status": "PASS",
    }, diff, test_diff, before_existing_test


def prefab_contract(root: Path) -> dict[str, Any]:
    text = (root / PREFAB).read_text(encoding="utf-8")
    old_prefab = root / "Assets/_Project/Resources/Diagnostics/PersonalLiveRecognitionTechnical.prefab"
    checks = {
        "root_name": "m_Name: PersonalMultiPrototypeLiveRecognitionTechnical" in text,
        "component_count_three": text.count("- component: {fileID:") == 3,
        "bridge_guid_once": text.count("guid: c6773d5ba8b0ae3498e54fbe28cf8e97") == 1,
        "harness_guid_once": text.count("guid: b683587873954482b05ddb9c95f690c4") == 1,
        "local_bridge_reference": "observationBridge: {fileID: 11400000}" in text,
        "integrator_null": "_integrator: {fileID: 0}" in text,
        "candidate_mode_fixed": "profileMode: 1" in text,
        "identity_transform": all(value in text for value in (
            "m_LocalRotation: {x: 0, y: 0, z: 0, w: 1}",
            "m_LocalPosition: {x: 0, y: 0, z: 0}",
            "m_LocalScale: {x: 1, y: 1, z: 1}",
        )),
        "panel_rect": all(value in text for value in ("x: 16", "y: 16", "width: 520", "height: 580")),
        "old_prefab_hash_preserved": sha256_bytes(old_prefab.read_bytes()) == "2b2d1835d358bab3a59ee5e2fcd23570a6a40c86f6aed83f968364692fed0658",
    }
    forbidden = ("MediaPipeLandmarkProvider", "HandPoseIntegrator", "WebCamTexture", "PhysicalWebcamCoordinateCompatibilityHarness")
    checks["forbidden_components_absent"] = not any(item in text for item in forbidden)
    require(all(checks.values()), "prefab serialization contract mismatch")
    return {"schema_version": SCHEMA + ".prefab_contract", "checks": checks, "status": "PASS"}


def protected_files(root: Path) -> dict[str, Any]:
    baseline = json.loads((root / INSTALL_EVIDENCE / "protected_before.json").read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for group_name, group in baseline["groups"].items():
        for expected in group["records"]:
            relative = expected["relative_path"]
            if relative in (HARNESS.as_posix(), EXISTING_LIVE_TEST.as_posix()):
                continue
            actual = file_record(root, relative)
            same = actual["size_bytes"] == expected["size_bytes"] and actual["sha256"] == expected["sha256"]
            records.append({**actual, "group": group_name, "byte_identical": same})

    installed = json.loads((root / INSTALL_EVIDENCE / "project_mapping_manifest.json").read_text(encoding="utf-8"))
    for expected in installed["mappings"]:
        relative = expected["project_relative_path"]
        actual = file_record(root, relative)
        same = actual["size_bytes"] == expected["size_bytes"] and actual["sha256"] == expected["sha256"]
        records.append({**actual, "group": "step9l_candidate_installed_mapping", "byte_identical": same})

    step9k = json.loads((root / STEP9K_RESULT / "scene_existing_prefab_preservation.json").read_text(encoding="utf-8"))
    for expected in step9k["records"]:
        actual = file_record(root, expected["path"])
        same = actual["size_bytes"] == expected["size_bytes"] and actual["sha256"] == expected["sha256"]
        records.append({**actual, "group": "step9k_scene_existing_prefab", "byte_identical": same})
    for relative, size, digest in (
        ("Assets/_Project/Resources/Diagnostics/PersonalLiveRecognitionTechnical.prefab", 1807,
         "2b2d1835d358bab3a59ee5e2fcd23570a6a40c86f6aed83f968364692fed0658"),
        ("Assets/_Project/Resources/Diagnostics/PersonalLiveRecognitionTechnical.prefab.meta", 154,
         "d14d8f3117790d4d3ff2362fca2326f6a102af6b81ffc5268aa63bf24b7e2046"),
    ):
        actual = file_record(root, relative)
        same = actual["size_bytes"] == size and actual["sha256"] == digest
        records.append({**actual, "group": "step9k_technical_prefab", "byte_identical": same})
    require(all(row["byte_identical"] for row in records), "protected file mismatch")
    return {
        "schema_version": SCHEMA + ".protected_files_before_after",
        "record_count": len(records),
        "mismatch_count": 0,
        "records": records,
        "status": "PASS",
    }


def test_source_inventory(root: Path) -> tuple[dict[str, Any], bytes]:
    rows: list[dict[str, Any]] = []
    sources = {
        next(iter(FIXTURES)): root / TEST,
        "WordsInSilence.Tests.EditMode.PersonalLiveRecognitionIntegrationV1Tests":
            root / "Assets/_Project/Tests/EditMode/PersonalLiveRecognitionIntegrationV1Tests.cs",
        "WordsInSilence.Tests.EditMode.PersonalMultiPrototypeRuntimeIntegrationV1Tests":
            root / "Assets/_Project/Tests/EditMode/PersonalMultiPrototypeRuntimeIntegrationV1Tests.cs",
    }
    for fixture, path in sources.items():
        methods = re.findall(r"\[Test\]\s*\r?\n\s*public void ([A-Za-z0-9_]+)\s*\(", path.read_text(encoding="utf-8-sig"))
        require(len(methods) == len(set(methods)) == FIXTURES[fixture], f"test source inventory mismatch: {fixture}")
        for method in methods:
            rows.append({"fixture": fixture, "method_name": method})
    rows.sort(key=lambda item: (item["fixture"], item["method_name"]))
    return {
        "schema_version": SCHEMA + ".test_method_inventory",
        "fixture_count": len(FIXTURES),
        "test_count": len(rows),
        "expected_counts": FIXTURES,
        "status": "PASS",
    }, csv_bytes(rows, ["fixture", "method_name"])


def xml_verification(data: bytes, inventory_csv: bytes) -> tuple[dict[str, Any], bytes]:
    root = ET.fromstring(data)
    require(root.tag == "test-run", "XML root mismatch")
    expected_total = sum(FIXTURES.values())
    for key, value in {
        "testcasecount": expected_total, "total": expected_total, "passed": expected_total,
        "failed": 0, "inconclusive": 0, "skipped": 0,
    }.items():
        require(int(root.attrib.get(key, "-1")) == value, f"XML aggregate mismatch: {key}")
    require(root.attrib.get("result") == "Passed", "XML result is not Passed")
    expected_methods: dict[str, set[str]] = {fixture: set() for fixture in FIXTURES}
    for row in csv.DictReader(io.StringIO(inventory_csv.decode("utf-8"))):
        expected_methods[row["fixture"]].add(row["method_name"])
    rows = []
    full_names: set[str] = set()
    for case in root.findall(".//test-case"):
        classname = case.attrib.get("classname", "")
        if classname not in FIXTURES:
            continue
        properties = {item.attrib.get("name"): item.attrib.get("value") for item in case.findall("./properties/property")}
        row = {
            "fixture": classname,
            "testcase_id": case.attrib["id"],
            "method_name": case.attrib["methodname"],
            "full_name": case.attrib["fullname"],
            "result": case.attrib["result"],
            "retry_iteration": int(properties.get("retryIteration", "-1")),
            "repeat_iteration": int(properties.get("repeatIteration", "-1")),
            "duration_seconds": float(case.attrib["duration"]),
        }
        require(row["full_name"] not in full_names, "duplicate XML testcase")
        full_names.add(row["full_name"])
        rows.append(row)
    require(len(rows) == expected_total, "XML selected testcase count mismatch")
    for fixture, count in FIXTURES.items():
        selected = [row for row in rows if row["fixture"] == fixture]
        require(len(selected) == count, f"XML fixture count mismatch: {fixture}")
        require({row["method_name"] for row in selected} == expected_methods[fixture], f"XML/source set mismatch: {fixture}")
    require(all(row["result"] == "Passed" for row in rows), "XML contains non-passed testcase")
    require(all(row["retry_iteration"] == 0 and row["repeat_iteration"] == 0 for row in rows),
            "XML retry/repeat mismatch")
    rows.sort(key=lambda item: (item["fixture"], item["method_name"]))
    summary = {
        "schema_version": SCHEMA + ".unity_test_results_verification",
        "platform": "EditMode",
        "fixture_counts": FIXTURES,
        "testcase_count": len(rows),
        "unique_testcase_count": len(full_names),
        "passed": expected_total,
        "failed": 0,
        "skipped": 0,
        "inconclusive": 0,
        "retry_zero_count": expected_total,
        "repeat_zero_count": expected_total,
        "start_utc": root.attrib["start-time"],
        "end_utc": root.attrib["end-time"],
        "duration_seconds": float(root.attrib["duration"]),
        "status": "PASS",
    }
    fields = ["fixture", "testcase_id", "method_name", "full_name", "result", "retry_iteration", "repeat_iteration", "duration_seconds"]
    return summary, csv_bytes(rows, fields)


def initial_failure_verification(data: bytes) -> dict[str, Any]:
    root = ET.fromstring(data)
    require(root.tag == "test-run", "initial failure XML root mismatch")
    require(root.attrib.get("total") == "38" and root.attrib.get("passed") == "37"
            and root.attrib.get("failed") == "1" and root.attrib.get("skipped") == "0"
            and root.attrib.get("inconclusive") == "0", "initial failure aggregate mismatch")
    failed = [case for case in root.findall(".//test-case") if case.attrib.get("result") != "Passed"]
    require(len(failed) == 1, "initial failure testcase count mismatch")
    case = failed[0]
    message_node = case.find("./failure/message")
    message = "" if message_node is None else "".join(message_node.itertext())
    require(case.attrib.get("fullname") ==
            "WordsInSilence.Tests.EditMode.PersonalLiveRecognitionIntegrationV1Tests.Step9IApprovedFixtureAndTestSourceRemainUnchanged",
            "initial failure testcase mismatch")
    require("3e4579c153ad23ee85c00b01967956f0777a1092384e5e8ade8b57b9fa5798bc" in message
            and "468613729c0299d11a4c72343c17df0c4fc7ed5b36367c8ce86ea9d81b6b62cf" in message,
            "initial failure hash evidence mismatch")
    return {
        "schema_version": SCHEMA + ".initial_failure_reproduction",
        "total": 38, "passed": 37, "failed": 1, "skipped": 0, "inconclusive": 0,
        "failed_testcase": case.attrib["fullname"],
        "cause": "approved Step 9L runtime source hash replaced Step 9I baseline; existing test constant was stale",
        "minimal_fix": "one expected SHA-256 constant updated; production unchanged",
        "status": "PASS",
    }


def log_verification(data: bytes, xml_path: Path) -> dict[str, Any]:
    text = data.decode("utf-8", errors="replace")
    normalized = xml_path.resolve().as_posix()
    compiler_patterns = {
        "error_cs": len(re.findall(r"\berror CS\d+\b", text)),
        "compilation_failed": text.lower().count("compilation failed"),
        "scripts_have_compiler_errors": text.lower().count("scripts have compiler errors"),
    }
    result = {
        "schema_version": SCHEMA + ".unity_editor_log_verification",
        "unity_version_6000_3_14f1": "Version is '6000.3.14f1" in text,
        "normal_gui_editor": "BatchMode: 0, IsHumanControllingUs: 1" in text,
        "tundra_build_success_count": text.count("*** Tundra build success"),
        "compiler_error_count": sum(compiler_patterns.values()),
        "compiler_error_pattern_counts": compiler_patterns,
        "new_factory_import_count": text.count("Start importing Assets/_Project/Scripts/Recognition/LiveInput/ApprovedPersonalMultiPrototypeLiveRuntimeFactory.cs"),
        "new_test_import_count": text.count("Start importing Assets/_Project/Tests/EditMode/PersonalMultiPrototypeLiveOptInRoutingV1Tests.cs"),
        "new_prefab_import_count": text.count("Start importing Assets/_Project/Resources/Diagnostics/PersonalMultiPrototypeLiveRecognitionTechnical.prefab"),
        "existing_live_test_reimport_count": text.count("Start importing Assets/_Project/Tests/EditMode/PersonalLiveRecognitionIntegrationV1Tests.cs"),
        "prebuild_count": text.count("Executing IPrebuildSetup for: Unity.PerformanceTesting.Editor.TestRunBuilder"),
        "postbuild_count": text.count("Executing IPostBuildCleanup for: Unity.PerformanceTesting.Editor.TestRunBuilder"),
        "validation_xml_save_count": text.count("Saving results to: " + normalized),
        "validation_xml_path": normalized,
        "line_count": len(text.splitlines()),
    }
    result["initial_compile_and_minimal_fix_recompile_confirmed"] = result["tundra_build_success_count"] == 2
    passed = (
        result["unity_version_6000_3_14f1"] and result["normal_gui_editor"]
        and result["tundra_build_success_count"] >= 2 and result["compiler_error_count"] == 0
        and result["new_factory_import_count"] == 1 and result["new_test_import_count"] == 1
        and result["new_prefab_import_count"] >= 1
        and result["initial_compile_and_minimal_fix_recompile_confirmed"]
        and result["prebuild_count"] == 2 and result["postbuild_count"] == 2
        and result["validation_xml_save_count"] == 1
    )
    require(passed, "Editor.log compile or same-run evidence mismatch")
    result["same_run_xml_log_crosscheck"] = True
    result["status"] = "PASS"
    return result


def screen_verification(data: bytes, reviewed: bool) -> dict[str, Any]:
    require(data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR", "screen is not PNG")
    width, height = struct.unpack(">II", data[16:24])
    require(width >= 1200 and height >= 600, "screen dimensions too small")
    require(reviewed, "screen requires explicit operator visual review")
    return {
        "schema_version": SCHEMA + ".test_runner_screen_verification",
        "width": width,
        "height": height,
        "manual_visual_review_confirmed": True,
        "expected_visible": {"platform": "EditMode", "passed": 38, "failed": 0, "inconclusive": 0},
        "status": "PASS",
    }


def git_staged(root: Path) -> dict[str, Any]:
    result = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=root, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    require(result.returncode == 0, "git staged query failed")
    paths = [line for line in result.stdout.splitlines() if line.strip()]
    require(not paths, "Git staged files are not zero")
    return {"staged_file_count": 0, "paths": [], "status": "PASS"}


def packaging_regression(validation: Path, tx: Any) -> dict[str, Any]:
    points = list(tx.INJECTION_POINTS) + ["POST_RENAME_VALIDATOR"]
    rows = []
    with tempfile.TemporaryDirectory(prefix="s9m_pkg_", dir=validation) as temporary:
        parent = Path(temporary)
        for index, point in enumerate(points):
            result = parent / f"result_{index}"
            delivery = parent / f"delivery_{index}.zip"
            detached = parent / f"delivery_{index}.zip.sha256"
            observed = False
            try:
                tx.commit_packaging_transaction(
                    result_files={"probe.txt": b"step9m"}, delivery_bytes=b"zip", detached_bytes=b"sha",
                    final_result=result, final_delivery=delivery, final_detached=detached,
                    allowed_root=parent,
                    injection_point=None if point == "POST_RENAME_VALIDATOR" else point,
                    post_rename_validator=(lambda *_: (_ for _ in ()).throw(InjectedPostValidationFailure(point)))
                    if point == "POST_RENAME_VALIDATOR" else None,
                    transaction_token=f"case{index}",
                )
            except (tx.InjectedPackagingFailure, InjectedPostValidationFailure):
                observed = True
            residual = sum(path.exists() for path in (result, delivery, detached))
            rows.append({"injection_point": point, "expected_exception_observed": observed,
                         "remaining_final_outputs": residual, "verdict": "PASS" if observed and residual == 0 else "FAIL"})
    require(all(row["verdict"] == "PASS" for row in rows), "packaging rollback regression failed")
    return {"schema_version": SCHEMA + ".packaging_transaction_regression",
            "production_transaction_function": tx.commit_packaging_transaction.__name__,
            "injection_count": len(rows), "pass_count": len(rows), "fail_count": 0,
            "rows": rows, "status": "PASS"}


def assertion(key: str, status: str, details: Any) -> dict[str, Any]:
    return {"assertion_key": key, "status": status, "details": details}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--screen-reviewed", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    validation = root / "Validation"
    output = validation / RESULT_NAME
    delivery = validation / DELIVERY_NAME
    detached = validation / (DELIVERY_NAME + ".sha256")
    require(not output.exists() and not delivery.exists() and not detached.exists(), "never-overwrite final target exists")

    fixed = fixed_inputs(root)
    payload, mappings, before = build_project_mapping(root)
    before_inputs = {item["project_relative_path"]: sha256_bytes(payload[item["project_relative_path"]]) for item in mappings}
    source, diff, existing_test_diff, before_existing_test = source_contract(root, before, payload)
    prefab = prefab_contract(root)
    protection = protected_files(root)
    test_inventory, inventory_csv = test_source_inventory(root)

    evidence = root / EVIDENCE_DIR
    xml_path = evidence / XML_NAME
    screen_path = evidence / SCREEN_NAME
    log_path = evidence / LOG_NAME
    first_fail_xml_path = evidence / FIRST_FAIL_XML_NAME
    first_fail_screen_path = evidence / FIRST_FAIL_SCREEN_NAME
    require(xml_path.is_file() and screen_path.is_file() and log_path.is_file()
            and first_fail_xml_path.is_file() and first_fail_screen_path.is_file(),
            "Step 9M Unity evidence is incomplete")
    xml_data, screen_data, log_data = xml_path.read_bytes(), screen_path.read_bytes(), log_path.read_bytes()
    xml_summary, cases_csv = xml_verification(xml_data, inventory_csv)
    log_summary = log_verification(log_data, xml_path)
    screen_summary = screen_verification(screen_data, args.screen_reviewed)
    first_fail_xml_data = first_fail_xml_path.read_bytes()
    first_fail_screen_data = first_fail_screen_path.read_bytes()
    first_failure = initial_failure_verification(first_fail_xml_data)
    require(first_fail_screen_data[:8] == b"\x89PNG\r\n\x1a\n", "initial failure screen is not PNG")
    staged = git_staged(root)

    installer_path = root / "Tools/s001_step9m_install_personal_multi_prototype_live_opt_in_routing_v1_0.py"
    installer = load_module(installer_path, "step9m_installer")
    install_regression = installer.run_failure_injection_regression(payload, mappings, validation)
    require(install_regression["status"] == "PASS", "install rollback regression failed")
    never_overwrite = installer.run_never_overwrite_regression(payload, mappings, validation)
    require(never_overwrite["status"] == "PASS", "install never-overwrite regression failed")

    tx_path = root / "Tools/s001_step9i_personal_runtime_opt_in_integration_v1_1.py"
    tx = load_module(tx_path, "approved_step9i_transaction")
    package_regression = packaging_regression(validation, tx)

    after_inputs = {item["project_relative_path"]: sha256_bytes((root / item["project_relative_path"]).read_bytes()) for item in mappings}
    require(before_inputs == after_inputs, "project payload changed during validation")
    assertions = [
        assertion("fixed_step9l_inputs", "PASS", fixed),
        assertion("project_mapping_two_replace_six_new", "PASS", {"count": 8}),
        assertion("legacy_default_and_explicit_candidate_source_contract", "PASS", source["checks"]),
        assertion("candidate_prefab_serialization", "PASS", prefab["checks"]),
        assertion("protected_approved_files", "PASS", {"count": protection["record_count"], "mismatch": 0}),
        assertion("test_source_inventory_38", "PASS", test_inventory["expected_counts"]),
        assertion("initial_unity_regression_failure_confirmed", "PASS", first_failure),
        assertion("minimal_existing_test_hash_baseline_repair", "PASS", source["checks"]["approved_step9l_runtime_hash_baseline"]),
        assertion("unity_compile_editor", "PASS", {"compiler_errors": 0}),
        assertion("unity_editmode_new_fixture_8_of_8", "PASS", 8),
        assertion("unity_editmode_existing_fixture_22_of_22", "PASS", 22),
        assertion("unity_editmode_existing_step9l_fixture_8_of_8", "PASS", 8),
        assertion("retry_repeat_skipped_inconclusive_zero", "PASS", {"count": 38}),
        assertion("install_transaction_failure_injection", "PASS", install_regression),
        assertion("install_never_overwrite", "PASS", never_overwrite),
        assertion("packaging_transaction_failure_injection", "PASS", package_regression),
        assertion("git_staged_zero", "PASS", staged),
        assertion("live_recognition_execution", "NOT_APPLICABLE", "forbidden in Step 9M"),
    ]
    require(len({item["assertion_key"] for item in assertions}) == len(assertions), "duplicate assertion key")

    mapping_json = {
        "schema_version": SCHEMA + ".project_mapping_manifest",
        "mapping_count": len(mappings), "replacement_count": 2, "new_file_count": 6,
        "mappings": mappings,
    }
    verdict = {
        "schema_version": SCHEMA + ".primary_verdict",
        "status_strings": [
            "STEP9M_PERSONAL_MULTI_PROTOTYPE_LIVE_OPT_IN_ROUTING_V1_0_SOURCE_AND_PACKAGE_READY",
            "STEP9M_LEGACY_SINGLE_MEDOID_DEFAULT_BEHAVIOR_PRESERVED",
            "STEP9M_MULTI_PROTOTYPE_LIVE_ROUTE_EXPLICIT_OPT_IN_IMPLEMENTED",
            "STEP9M_UNITY_COMPILE_EDITOR_PASS",
            "STEP9M_EDITMODE_NEW_8_EXISTING_22_AND_8_PASS",
            "STEP9M_LIVE_RECOGNITION_NOT_EXECUTED",
            "BLOCKED_PENDING_EXTERNAL_REVIEW",
        ],
        "candidate_profile_id": "step9l_current_user_multi_prototype_v1_0",
        "legacy_profile_id": "step9h_current_user_v1_0",
        "candidate_default_active": False,
        "live_execution": "NOT_APPLICABLE",
        "status": "PASS",
    }
    report = "\n".join((
        "# WordsInSilence Step 9M v1.0 Validation Report", "",
        "- Legacy Step 9H single-medoid remains the serialized default.",
        "- Step 9L 9-prototype live routing is reachable only through the explicit candidate mode.",
        "- Unity 6000.3.14f1 compile: PASS (compiler error 0).",
        "- EditMode: new 8/8, existing live 22/22, existing Step 9L runtime 8/8.",
        "- Candidate prefab is separate; the Step 9K prefab and scenes are byte-identical.",
        "- Live recognition/webcam execution: NOT_APPLICABLE.",
        "- External review remains pending.", "",
    )).encode("utf-8")

    files: dict[str, bytes] = {
        "input_provenance.json": json_bytes({"schema_version": SCHEMA + ".input_provenance", "fixed_inputs": fixed}),
        "source_change_inventory.json": json_bytes(source),
        "unified_diff/PersonalLiveRecognitionHarness.cs.diff": diff,
        "unified_diff/PersonalLiveRecognitionIntegrationV1Tests.cs.diff": existing_test_diff,
        "project_mapping_manifest.json": json_bytes(mapping_json),
        "live_profile_routing_contract.json": json_bytes(source),
        "prefab_serialization_verification.json": json_bytes(prefab),
        "protected_files_before_after.json": json_bytes(protection),
        "test_method_inventory.json": json_bytes(test_inventory),
        "test_method_inventory.csv": inventory_csv,
        "unity_test_results_verification.json": json_bytes(xml_summary),
        "unity_test_case_inventory.csv": cases_csv,
        "unity_editor_log_verification.json": json_bytes(log_summary),
        "test_runner_screen_verification.json": json_bytes(screen_summary),
        "initial_failure_reproduction.json": json_bytes(first_failure),
        "install_transaction_failure_injection_regression.json": json_bytes(install_regression),
        "install_never_overwrite_regression.json": json_bytes(never_overwrite),
        "packaging_transaction_failure_injection_regression.json": json_bytes(package_regression),
        "git_staged_status.json": json_bytes(staged),
        "primary_verdict.json": json_bytes(verdict),
        "ASSERTIONS.jsonl": b"".join(json.dumps(item, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n" for item in assertions),
        "VALIDATION_REPORT.md": report,
        "source_before/PersonalLiveRecognitionHarness.cs": before,
        "source_before/PersonalLiveRecognitionIntegrationV1Tests.cs": before_existing_test,
        "input_subset/provenance/step9l_install_project_mapping_manifest.json": (root / INSTALL_EVIDENCE / "project_mapping_manifest.json").read_bytes(),
        "input_subset/provenance/step9l_gate_primary_verdict.json": (root / GATE_EVIDENCE / "primary_verdict.json").read_bytes(),
        "input_subset/unity_evidence/" + XML_NAME: xml_data,
        "input_subset/unity_evidence/" + SCREEN_NAME: screen_data,
        "input_subset/unity_evidence/" + LOG_NAME: log_data,
        "input_subset/unity_evidence/" + FIRST_FAIL_XML_NAME: first_fail_xml_data,
        "input_subset/unity_evidence/" + FIRST_FAIL_SCREEN_NAME: first_fail_screen_data,
        "analysis/s001_step9m_install_personal_multi_prototype_live_opt_in_routing_v1_0.py": installer_path.read_bytes(),
        "analysis/s001_step9m_personal_multi_prototype_live_opt_in_routing_v1_0.py": Path(__file__).read_bytes(),
        "analysis/approved_transaction/s001_step9i_personal_runtime_opt_in_integration_v1_1.py": tx_path.read_bytes(),
    }
    for relative, body in payload.items():
        files["project_files/" + relative] = body

    manifest_rows = [
        {"path": name, "size_bytes": len(body), "sha256": sha256_bytes(body)}
        for name, body in sorted(files.items())
    ]
    manifest = {"schema_version": SCHEMA + ".output_file_manifest", "self_excluded": True,
                "file_count": len(manifest_rows), "files": manifest_rows}
    files["output_file_manifest.json"] = json_bytes(manifest)
    delivery_bytes = tx.deterministic_zip_bytes(files)
    detached_bytes = tx.strict_detached_bytes(delivery_bytes, DELIVERY_NAME)
    require(tx.deterministic_zip_bytes(files) == delivery_bytes, "deterministic ZIP regeneration mismatch")
    zip_check = tx.inspect_zip(delivery_bytes, set(files))

    def post_validate(result_path: Path, delivery_path: Path, detached_path: Path) -> None:
        for name, body in files.items():
            require((result_path / PurePosixPath(name)).read_bytes() == body, f"result tree mismatch: {name}")
        require(delivery_path.read_bytes() == delivery_bytes, "delivery after commit mismatch")
        require(detached_path.read_bytes() == detached_bytes, "detached after commit mismatch")
        tx.verify_strict_detached(detached_path.read_bytes(), delivery_path.read_bytes(), DELIVERY_NAME)

    tx.commit_packaging_transaction(
        result_files=files, delivery_bytes=delivery_bytes, detached_bytes=detached_bytes,
        final_result=output, final_delivery=delivery, final_detached=detached,
        allowed_root=validation, post_rename_validator=post_validate,
        transaction_token="step9m_v1_0",
    )
    print(json.dumps({
        "result_directory": str(output), "delivery": str(delivery),
        "delivery_size_bytes": len(delivery_bytes), "delivery_sha256": sha256_bytes(delivery_bytes),
        "detached": str(detached), "detached_size_bytes": len(detached_bytes),
        "detached_sha256": sha256_bytes(detached_bytes), "zip_verification": zip_check,
        "assertions": {"PASS": 17, "FAIL": 0, "NOT_APPLICABLE": 1},
    }, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
