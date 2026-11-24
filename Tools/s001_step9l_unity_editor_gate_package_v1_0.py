#!/usr/bin/env python3
"""Verify and package the single approved Step 9L Unity EditMode gate run."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import subprocess
import tempfile
from typing import Any
import zipfile
import xml.etree.ElementTree as ET


SCHEMA = "words_in_silence.step9l.personal_multi_prototype_runtime_candidate.unity_editor_gate.v1_0"
FIXTURE = "WordsInSilence.Tests.EditMode.PersonalMultiPrototypeRuntimeIntegrationV1Tests"
EVIDENCE_REL = Path("Validation/WordsInSilence_step9l_personal_multi_prototype_runtime_candidate_v1_1_unity_editor_gate")
INSTALL_REL = Path("Validation/WordsInSilence_step9l_personal_multi_prototype_runtime_candidate_v1_1_unity_install")
XML_NAME = "TestResults_20260810_120522.xml"
XML_SIZE = 9_620
XML_SHA256 = "accf6f4b850ff1b1ec98c948447edcc74e2344b371c6fe268ec52aa733672a25"
SCREEN_NAME = "testRunner_result_8pass.png"
SCREEN_SIZE = 322_777
SCREEN_SHA256 = "c96564c1ec88c7a4067e8bafe95c35ebb1d646de51e5e7f93a6fe2893d3664a2"
LOG_NAME = "Step9L_Editor_full.log"
LOG_SIZE = 61_243
LOG_SHA256 = "55ecf682ff60785427f8d822c5234c0739df3e44e7f0f92fbe7bcba6e8ef9ce6"
TEST_SOURCE_REL = Path("Assets/_Project/Tests/EditMode/PersonalMultiPrototypeRuntimeIntegrationV1Tests.cs")
TEST_SOURCE_SIZE = 12_349
TEST_SOURCE_SHA256 = "d40397629fd65374c0c2b79ceccc561ba9474e3dc1135eff1ed4ea2371ba6733"
SOURCE_REL = Path("Assets/_Project/Scripts/Recognition/Runtime/PersonalRuntimeOptInProfile.cs")
SOURCE_AFTER_SIZE = 55_677
SOURCE_AFTER_SHA256 = "468613729c0299d11a4c72343c17df0c4fc7ed5b36367c8ce86ea9d81b6b62cf"
TRANSACTION_REL = Path("Tools/s001_step9i_personal_runtime_opt_in_integration_v1_1.py")
TRANSACTION_SIZE = 37_722
TRANSACTION_SHA256 = "30471dc6725bdbdd0cc80c345b238485d1cb0d8d7a5933aac7477d536d714caf"
INJECTION_POINTS = (
    "BEFORE_FINAL_RENAME", "AFTER_RESULT_RENAME", "AFTER_DELIVERY_RENAME",
    "AFTER_DETACHED_RENAME_BEFORE_COMMIT",
)


class GateError(RuntimeError):
    pass


class PostValidationFailure(GateError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def record(path: Path, relative: str) -> dict[str, Any]:
    require(path.is_file(), "required file missing: " + str(path))
    data = path.read_bytes()
    return {"relative_path": relative, "size_bytes": len(data), "sha256": sha256_bytes(data)}


def fixed(path: Path, size: int, digest: str, label: str) -> bytes:
    require(path.is_file(), label + " missing")
    data = path.read_bytes()
    require(len(data) == size and sha256_bytes(data) == digest, label + " identity mismatch")
    return data


def load_transaction(root: Path) -> Any:
    path = root / TRANSACTION_REL
    fixed(path, TRANSACTION_SIZE, TRANSACTION_SHA256, "approved packaging transaction source")
    spec = importlib.util.spec_from_file_location("step9i_transaction_v1_1_gate", path)
    require(spec is not None and spec.loader is not None, "cannot load approved transaction")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def xml_verification(data: bytes, test_source: bytes) -> tuple[dict[str, Any], bytes]:
    root = ET.fromstring(data)
    require(root.tag == "test-run", "XML root mismatch")
    expected_root = {"testcasecount": "8", "result": "Passed", "total": "8", "passed": "8", "failed": "0", "inconclusive": "0", "skipped": "0"}
    require(all(root.attrib.get(key) == value for key, value in expected_root.items()), "XML aggregate mismatch")
    cases = [case for case in root.findall(".//test-case") if case.attrib.get("classname") == FIXTURE]
    require(len(cases) == 8 and len({case.attrib["fullname"] for case in cases}) == 8, "XML testcase uniqueness mismatch")
    methods = re.findall(r"\[Test\]\s*\r?\n\s*public void ([A-Za-z0-9_]+)\s*\(", test_source.decode("utf-8-sig"))
    require(len(methods) == len(set(methods)) == 8, "test source method inventory mismatch")
    rows = []
    for case in cases:
        properties = {item.attrib.get("name"): item.attrib.get("value") for item in case.findall("./properties/property")}
        retry = int(properties.get("retryIteration", "-1"))
        repeat = int(properties.get("repeatIteration", "-1"))
        row = {
            "testcase_id": case.attrib["id"], "method_name": case.attrib["methodname"],
            "full_name": case.attrib["fullname"], "result": case.attrib["result"],
            "retry_iteration": retry, "repeat_iteration": repeat,
            "duration_seconds": float(case.attrib["duration"]),
        }
        rows.append(row)
    require({row["method_name"] for row in rows} == set(methods), "XML/source testcase set mismatch")
    require(all(row["result"] == "Passed" and row["retry_iteration"] == 0 and row["repeat_iteration"] == 0 for row in rows), "XML testcase result/retry/repeat mismatch")
    rows.sort(key=lambda row: row["method_name"])
    summary = {
        "schema_version": SCHEMA + ".test_results_verification", "platform": "EditMode",
        "fixture": FIXTURE, "testcase_count": len(rows), "unique_testcase_count": len({row["full_name"] for row in rows}),
        "passed": 8, "failed": 0, "skipped": 0, "inconclusive": 0,
        "retry_zero_count": sum(row["retry_iteration"] == 0 for row in rows),
        "repeat_zero_count": sum(row["repeat_iteration"] == 0 for row in rows),
        "start_utc": root.attrib["start-time"], "end_utc": root.attrib["end-time"],
        "duration_seconds": float(root.attrib["duration"]), "source_method_set_match": True, "status": "PASS",
    }
    return summary, csv_bytes(rows, ["testcase_id", "method_name", "full_name", "result", "retry_iteration", "repeat_iteration", "duration_seconds"])


def log_verification(data: bytes, xml_path: Path) -> dict[str, Any]:
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    normalized_path = xml_path.resolve().as_posix()
    compiler_patterns = {
        "error_cs": len(re.findall(r"\berror CS\d+\b", text)),
        "compilation_failed": text.lower().count("compilation failed"),
        "scripts_have_compiler_errors": text.lower().count("scripts have compiler errors"),
    }
    summary = {
        "schema_version": SCHEMA + ".compile_editor_log_verification",
        "unity_version_6000_3_14f1": "Version is '6000.3.14f1" in text,
        "normal_gui_editor": "BatchMode: 0, IsHumanControllingUs: 1" in text,
        "licensing_connection_success_count": text.count("[Licensing::IpcConnector] Successfully connected"),
        "tundra_build_success_count": text.count("*** Tundra build success"),
        "compiler_error_count": sum(compiler_patterns.values()), "compiler_error_pattern_counts": compiler_patterns,
        "test_source_import_count": text.count("Start importing Assets/_Project/Tests/EditMode/PersonalMultiPrototypeRuntimeIntegrationV1Tests.cs"),
        "prebuild_count": text.count("Executing IPrebuildSetup for: Unity.PerformanceTesting.Editor.TestRunBuilder"),
        "postbuild_count": text.count("Executing IPostBuildCleanup for: Unity.PerformanceTesting.Editor.TestRunBuilder"),
        "validation_xml_save_count": text.count("Saving results to: " + normalized_path),
        "validation_xml_path": normalized_path,
        "line_count": len(lines),
    }
    passed = (
        summary["unity_version_6000_3_14f1"] and summary["normal_gui_editor"]
        and summary["licensing_connection_success_count"] >= 1
        and summary["tundra_build_success_count"] == 1
        and summary["compiler_error_count"] == 0
        and summary["test_source_import_count"] == 1
        and summary["prebuild_count"] == 1 and summary["postbuild_count"] == 1
        and summary["validation_xml_save_count"] == 1
    )
    require(passed, "Editor.log compile or same-run evidence mismatch")
    summary["same_run_xml_log_crosscheck"] = True
    summary["status"] = "PASS"
    return summary


def screen_verification(data: bytes) -> dict[str, Any]:
    require(data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR", "screen evidence is not a valid PNG")
    width, height = struct.unpack(">II", data[16:24])
    require(width >= 1000 and height >= 700, "screen evidence dimensions too small")
    return {
        "schema_version": SCHEMA + ".screen_evidence_verification", "width": width, "height": height,
        "manual_visual_review": {
            "unity_test_runner_visible": True, "editmode_selected": True,
            "fixture_visible": "PersonalMultiPrototypeRuntimeIntegrationV1Tests (8 tests)",
            "passed_visible": 8, "failed_visible": 0, "inconclusive_visible": 0,
            "console_error_visible": 0,
        },
        "invalid_earlier_desktop_capture_excluded": "Step9L_TestRunner_8PASS_fullscreen.png",
        "status": "PASS",
    }


def verify_install(root: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    directory = root / INSTALL_REL
    require(directory.is_dir(), "install evidence directory missing")
    source_files = {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}
    require("output_file_manifest.json" in source_files, "install output manifest missing")
    manifest = json.loads(source_files["output_file_manifest.json"])
    records = manifest["outputs"]
    require(len(records) == 9 and set(source_files) == {row["relative_path"] for row in records} | {"output_file_manifest.json"}, "install evidence manifest membership mismatch")
    for row in records:
        payload = source_files[row["relative_path"]]
        require(len(payload) == row["size_bytes"] and sha256_bytes(payload) == row["sha256"], "install evidence manifest payload mismatch")
    mapping = json.loads(source_files["project_mapping_manifest.json"])
    require(mapping["mapping_count"] == 54 and mapping["replacement_count"] == 1 and mapping["new_file_count"] == 53, "install mapping count mismatch")
    mismatches = []
    for row in mapping["mappings"]:
        path = root / row["project_relative_path"]
        actual = record(path, row["project_relative_path"])
        if actual["size_bytes"] != row["size_bytes"] or actual["sha256"] != row["sha256"]:
            mismatches.append(row["project_relative_path"])
    require(not mismatches, "installed project mapping changed")
    before = json.loads(source_files["protected_before.json"])
    after_install = json.loads(source_files["protected_after_install.json"])
    require(before["groups"] == after_install["groups"], "install protected before/after mismatch")
    current_mismatches = []
    for group in before["groups"].values():
        for expected in group["records"]:
            actual = record(root / expected["relative_path"], expected["relative_path"])
            if actual != expected:
                current_mismatches.append(expected["relative_path"])
    require(not current_mismatches, "protected file changed after Unity run")
    source = fixed(root / SOURCE_REL, SOURCE_AFTER_SIZE, SOURCE_AFTER_SHA256, "installed runtime source")
    test_source = fixed(root / TEST_SOURCE_REL, TEST_SOURCE_SIZE, TEST_SOURCE_SHA256, "installed test source")
    return {
        "schema_version": SCHEMA + ".install_manifest_verification", "install_evidence_file_count": len(source_files),
        "mapping_count": 54, "installed_mapping_byte_identical_count": 54, "installed_mapping_mismatch_count": 0,
        "protected_file_count": before["total_file_count"], "protected_mismatch_count_after_unity": 0,
        "source_after_size_bytes": len(source), "source_after_sha256": sha256_bytes(source),
        "test_source_size_bytes": len(test_source), "test_source_sha256": sha256_bytes(test_source), "status": "PASS",
    }, source_files


def process_state() -> dict[str, Any]:
    command = "$p=Get-Process Unity -ErrorAction SilentlyContinue | Sort-Object Id | ForEach-Object {[pscustomobject]@{pid=$_.Id;responding=$_.Responding;start_utc=$_.StartTime.ToUniversalTime().ToString('o')}}; @($p) | ConvertTo-Json -Compress"
    result = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True, check=True)
    raw = result.stdout.strip()
    items = json.loads(raw) if raw else []
    if isinstance(items, dict): items = [items]
    return {"captured_read_only": True, "unity_process_count": len(items), "processes": items, "editor_still_running": len(items) > 0}


def git_staged(root: Path) -> dict[str, Any]:
    result = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=root, capture_output=True, text=True, check=True)
    paths = [line for line in result.stdout.splitlines() if line]
    require(not paths, "Git staged files present")
    return {"staged_count": 0, "staged_paths": [], "status": "PASS"}


def packaging_regression(validation: Path, tx: Any) -> dict[str, Any]:
    rows = []
    points = list(INJECTION_POINTS) + ["POST_VALIDATION_FAILURE"]
    for index, point in enumerate(points):
        with tempfile.TemporaryDirectory(prefix="step9l_gate_tx_", dir=validation) as temporary:
            root = Path(temporary)
            finals = (root / "result", root / "delivery.zip", root / "delivery.zip.sha256")
            token = "gate_tx_" + str(index)
            validation_complete = False
            expected = False
            def validator(result: Path, delivery: Path, detached: Path) -> None:
                nonlocal validation_complete
                require(result.is_dir() and delivery.is_file() and detached.is_file(), "synthetic output missing")
                validation_complete = True
                if point == "POST_VALIDATION_FAILURE": raise PostValidationFailure(point)
            try:
                tx.commit_packaging_transaction(
                    result_files={"marker.json": b"{}\n"}, delivery_bytes=b"zip", detached_bytes=b"checksum",
                    final_result=finals[0], final_delivery=finals[1], final_detached=finals[2], allowed_root=root,
                    injection_point=None if point == "POST_VALIDATION_FAILURE" else point,
                    post_rename_validator=validator, transaction_token=token,
                )
            except (tx.InjectedPackagingFailure, PostValidationFailure) as error:
                expected = str(error) == point
            residues = sum(path.exists() or path.is_symlink() for path in finals) + sum(1 for _ in root.glob(".step9i_v1_1_*"))
            passed = expected and residues == 0
            rows.append({"injection_point": point, "expected_exception_observed": expected, "post_validation_completed": validation_complete, "residual_output_count": residues, "verdict": "PASS" if passed else "FAIL"})
    require(all(row["verdict"] == "PASS" for row in rows), "packaging transaction regression failed")
    return {"same_approved_production_function_used": True, "injection_count": 5, "pass_count": 5, "fail_count": 0, "rows": rows, "status": "PASS"}


def output_category(name: str) -> str:
    if name.startswith("input_subset/"): return "fixed_evidence"
    if name.startswith("analysis/"): return "analysis_source"
    if name == "ASSERTIONS.jsonl": return "assertions"
    if name == "VALIDATION_REPORT.md": return "validation_report"
    return "validation_output"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.project_root.resolve()
    validation = root / "Validation"
    name = "WordsInSilence_step9l_personal_multi_prototype_runtime_candidate_v1_1_unity_editor_gate_verification_v1_0"
    output = validation / name
    delivery = validation / (name + "_delivery.zip")
    detached = Path(str(delivery) + ".sha256")
    require(not output.exists() and not delivery.exists() and not detached.exists(), "never-overwrite gate output exists")
    tx = load_transaction(root)
    evidence = root / EVIDENCE_REL
    xml_path = evidence / XML_NAME
    xml_data = fixed(xml_path, XML_SIZE, XML_SHA256, "test result XML")
    screen_data = fixed(evidence / SCREEN_NAME, SCREEN_SIZE, SCREEN_SHA256, "Test Runner screen")
    log_data = fixed(evidence / LOG_NAME, LOG_SIZE, LOG_SHA256, "full Editor.log")
    test_source = fixed(root / TEST_SOURCE_REL, TEST_SOURCE_SIZE, TEST_SOURCE_SHA256, "installed test source")
    tests, inventory_csv = xml_verification(xml_data, test_source)
    compile_log = log_verification(log_data, xml_path)
    screen = screen_verification(screen_data)
    install, install_files = verify_install(root)
    processes = process_state()
    staged = git_staged(root)
    transaction = packaging_regression(validation, tx)
    conditions = {
        "compiler_error_zero": compile_log["compiler_error_count"] == 0,
        "test_eight_pass": tests["passed"] == 8 and tests["failed"] == 0,
        "skipped_inconclusive_zero": tests["skipped"] == 0 and tests["inconclusive"] == 0,
        "retry_repeat_zero": tests["retry_zero_count"] == 8 and tests["repeat_zero_count"] == 8,
        "same_run_xml_log": compile_log["same_run_xml_log_crosscheck"],
        "test_runner_screen_pass": screen["status"] == "PASS",
        "installed_mapping_54_unchanged": install["installed_mapping_byte_identical_count"] == 54,
        "protected_mismatch_zero": install["protected_mismatch_count_after_unity"] == 0,
        "git_staged_zero": staged["staged_count"] == 0,
    }
    require(all(conditions.values()), "Unity gate condition failed")
    assertions = [
        {"assertion_key": key, "status": "PASS" if value else "FAIL", "details": value}
        for key, value in conditions.items()
    ] + [
        {"assertion_key": "packaging_transaction_failure_injection_five_of_five", "status": "PASS", "details": transaction},
        {"assertion_key": "live_recognition_not_executed", "status": "NOT_APPLICABLE", "details": "prohibited by Step 9L gate"},
    ]
    require(len({row["assertion_key"] for row in assertions}) == len(assertions), "assertion key duplicate")
    counts = Counter(row["status"] for row in assertions)
    primary = {
        "schema_version": SCHEMA + ".primary_verdict", "actual_primary_verdict": "UNITY_COMPILE_AND_EDITMODE_GATE_PASS",
        "fixture": FIXTURE, "compiler_errors": 0, "total": 8, "passed": 8, "failed": 0, "skipped": 0, "inconclusive": 0,
        "retry_zero": 8, "repeat_zero": 8, "candidate_profile_default_inactive": True, "live_recognition_executed": False,
        "status_strings": [
            "STEP9L_PERSONAL_MULTI_PROTOTYPE_RUNTIME_CANDIDATE_INSTALLED",
            "STEP9L_UNITY_COMPILE_EDITOR_PASS",
            "STEP9L_EDITMODE_FIRST_RUN_PASS_8_OF_8",
            "STEP9L_SINGLE_MEDOID_DEFAULT_AND_MULTI_PROTOTYPE_OPT_IN_CONTRACT_PASS",
            "STEP9L_LIVE_RECOGNITION_NOT_EXECUTED",
            "BLOCKED_PENDING_EXTERNAL_REVIEW",
        ],
    }
    files: dict[str, bytes] = {
        "analysis/s001_step9l_unity_editor_gate_package_v1_0.py": Path(__file__).read_bytes(),
        "input_subset/unity_evidence/" + XML_NAME: xml_data,
        "input_subset/unity_evidence/" + SCREEN_NAME: screen_data,
        "input_subset/unity_evidence/" + LOG_NAME: log_data,
        "unity_test_results_verification.json": json_bytes(tests),
        "test_case_inventory.csv": inventory_csv,
        "unity_compile_editor_log_verification.json": json_bytes(compile_log),
        "test_runner_screen_verification.json": json_bytes(screen),
        "install_manifest_verification.json": json_bytes(install),
        "unity_process_state.json": json_bytes(processes),
        "packaging_transaction_failure_injection_regression.json": json_bytes(transaction),
        "primary_verdict.json": json_bytes(primary),
    }
    for filename, payload in sorted(install_files.items()):
        files["input_subset/install_evidence/" + filename] = payload
    files["ASSERTIONS.jsonl"] = b"".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n" for row in assertions)
    files["VALIDATION_REPORT.md"] = ("\n".join([
        "# WordsInSilence Step 9L Unity Editor Gate", "", "## Outcome", "",
        "- Unity: `6000.3.14f1`, normal GUI Editor", "- Compiler errors: `0`",
        "- PersonalMultiPrototypeRuntimeIntegrationV1Tests: `8 PASS / 0 FAIL / 0 skipped / 0 inconclusive`",
        "- Retry / repeat: `0 / 0` for all eight unique testcases",
        "- Installed mapping: `54/54` byte-identical; protected mismatch: `0`",
        "- Candidate remains opt-in and default inactive; live recognition was not executed", "",
        "## Evidence", "", f"- XML: `{XML_NAME}`", f"- Screen: `{SCREEN_NAME}`", f"- Full log: `{LOG_NAME}`",
        f"- Assertions: PASS `{counts['PASS']}`, FAIL `{counts['FAIL']}`, NOT_APPLICABLE `{counts['NOT_APPLICABLE']}`", "",
        "External review is required before live recognition work.", "",
    ])).encode("utf-8")
    manifest_records = [{"relative_path": path, "category": output_category(path), "size_bytes": len(payload), "sha256": sha256_bytes(payload)} for path, payload in sorted(files.items())]
    files["output_file_manifest.json"] = json_bytes({"schema_version": SCHEMA + ".output_file_manifest", "self_excluded": "output_file_manifest.json", "output_count_excluding_manifest": len(manifest_records), "outputs": manifest_records})
    zip_data = tx.deterministic_zip_bytes(files)
    require(tx.deterministic_zip_bytes(files) == zip_data, "deterministic ZIP regeneration mismatch")
    zip_check = tx.inspect_zip(zip_data, set(files))
    detached_data = tx.strict_detached_bytes(zip_data, delivery.name)
    def validate(result: Path, zip_path: Path, checksum_path: Path) -> None:
        require(zip_path.read_bytes() == zip_data and checksum_path.read_bytes() == detached_data, "committed package bytes mismatch")
        actual = {path.relative_to(result).as_posix(): path.read_bytes() for path in result.rglob("*") if path.is_file()}
        require(actual == files, "committed result tree mismatch")
        tx.inspect_zip(zip_path.read_bytes(), set(files))
    tx.commit_packaging_transaction(
        result_files=files, delivery_bytes=zip_data, detached_bytes=detached_data,
        final_result=output, final_delivery=delivery, final_detached=detached,
        allowed_root=validation, post_rename_validator=validate, transaction_token="step9l_unity_gate_v1_0",
    )
    require(git_staged(root)["staged_count"] == 0, "Git staged files changed")
    print(json.dumps({
        "status": primary["actual_primary_verdict"], "result_file_count": len(files),
        "delivery_zip": delivery.as_posix(), "delivery_size_bytes": len(zip_data), "delivery_sha256": sha256_bytes(zip_data),
        "detached_checksum": detached.as_posix(), "detached_size_bytes": len(detached_data), "detached_file_sha256": sha256_bytes(detached_data),
        "zip_verification": zip_check, "assertions": dict(counts), "compiler_errors": 0,
        "tests": "8 PASS / 0 FAIL / 0 skipped / 0 inconclusive", "retry_repeat": "0/0 all 8", "git_staged_count": 0,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateError as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, ensure_ascii=False, indent=2))
        raise SystemExit(1)
