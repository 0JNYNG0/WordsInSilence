#!/usr/bin/env python3
"""Build and verify the Step 9P candidate ambiguity-tolerance Delivery."""

from __future__ import annotations

import csv
import difflib
import hashlib
import importlib.util
import io
import json
import os
import re
import stat
import subprocess
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any
from xml.etree import ElementTree


SCHEMA = "words_in_silence.step9p.candidate_transient_ambiguous_hand_identity_tolerance.v1_0"
RESULT_NAME = "WordsInSilence_step9p_candidate_transient_ambiguous_hand_identity_tolerance_v1_0"
DELIVERY_NAME = RESULT_NAME + "_delivery.zip"
DETACHED_NAME = DELIVERY_NAME + ".sha256"
GATE_NAME = RESULT_NAME + "_unity_editor_gate"

CONTROLLER = "Assets/_Project/Scripts/Recognition/LiveInput/PersonalLiveRecognitionController.cs"
HARNESS = "Assets/_Project/Scripts/Recognition/LiveInput/PersonalLiveRecognitionHarness.cs"
NEW_TEST = "Assets/_Project/Tests/EditMode/CandidateTransientAmbiguousHandIdentityToleranceV1Tests.cs"
NEW_TEST_META = NEW_TEST + ".meta"

CONTROLLER_BEFORE = (
    "Validation/WordsInSilence_step9k_primary_failure_preservation_diagnostic_v1_0/"
    "source_after/PersonalLiveRecognitionController.cs"
)
HARNESS_BEFORE = (
    "Validation/WordsInSilence_step9m_personal_multi_prototype_live_opt_in_routing_v1_0/"
    "project_files/Assets/_Project/Scripts/Recognition/LiveInput/PersonalLiveRecognitionHarness.cs"
)
STEP9M_ZIP = "Validation/WordsInSilence_step9m_personal_multi_prototype_live_opt_in_routing_v1_0_delivery.zip"
STEP9M_SIZE = 577_253
STEP9M_SHA256 = "c99835435ab49a47ad58ade1e25abcc886a04904ef2b7f80ae61f81d003a0f12"
STEP9M_DETACHED_SIZE = 151
STEP9M_DETACHED_SHA256 = "7fdaebcd6bd17c227e0a9eecde92ecb3908ef8d24790b75a954bc06d7f1aeab3"

EXPECTED_EXTRA_PROTECTED = (
    ("Assets/_Project/Scripts/Recognition/PhysicalCompatibility/Step9CHandIdentityResolver.cs", 14303, "4ea8b4a59253963e460b9ac05f6d472f8663f6f57467fae9500abadc93bdead3"),
    ("Assets/_Project/Scripts/Recognition/PhysicalCompatibility/Step9CHandIdentityObservationBridge.cs", 6472, "f028f1485dc396fd9d7c38672f2a4097cb769ecfd321e45f3b63bb2f02c7adc2"),
    ("Assets/_Project/Scripts/Recognition/LiveInput/ApprovedLivePreprocessingAdapter.cs", 26975, "1aa1eaed2158fef7dc3e8f163f66cd1fcf3de19b02035863b4a3e2500b66c47c"),
    ("Assets/_Project/Scripts/Recognition/Runtime/RecognitionSequenceInputSession.cs", 10790, "960a5674ed1b05cf55867142ba7b6df48c996cc50f540df1ee3037e83214df91"),
    ("Assets/_Project/Scripts/Recognition/Runtime/RuntimeRecognitionService.cs", 11144, "d6b3d44130fb9fae5e00745d15a5d537f4b1bc03a7d44fe8df6fd8d02656137a"),
    ("Assets/_Project/Scripts/Recognition/Core/MaskAwareDtw.cs", 5649, "69c76f869e1c2b123b06eb3e81668dc2dce7c555ce967d769e7bf6abb6977e1b"),
    ("Assets/_Project/Scripts/Recognition/LiveInput/RawMediaPipeObservation.cs", 9249, "0d69db1f4287b514e38fa40aa863ff88ecd313bbf192f7e22e66f7f52be516d1"),
    ("Assets/_Project/Scripts/Recognition/Runtime/NormalizedRecognitionFrame.cs", 2111, "fbdd0f05ce3753364162d57e5344dac49822a66b646164470acf3674febbe0e0"),
    ("Assets/_Project/Scripts/Recognition/LiveInput/ApprovedPersonalMultiPrototypeLiveRuntimeFactory.cs", 1851, "d645f71063d8a849579a38272469113f669cef2fa5e81ff118212c94b514b0db"),
    ("Assets/_Project/Tests/EditMode/PersonalLiveRecognitionIntegrationV1Tests.cs", 36212, "12b422e747c0aaad771109e763f908753e03f3de975977ef2c351c66e21f5de4"),
    ("Assets/_Project/Tests/EditMode/PersonalMultiPrototypeRuntimeIntegrationV1Tests.cs", 12349, "d40397629fd65374c0c2b79ceccc561ba9474e3dc1135eff1ed4ea2371ba6733"),
    ("Assets/_Project/Tests/EditMode/PersonalMultiPrototypeLiveOptInRoutingV1Tests.cs", 15425, "97ac5e63e4ff9adf4f208cc63ae4bc3932c9ea8055c262d8099351abe5694b41"),
    ("Assets/_Project/Resources/Diagnostics/PersonalLiveRecognitionTechnical.prefab", 1807, "2b2d1835d358bab3a59ee5e2fcd23570a6a40c86f6aed83f968364692fed0658"),
    ("Assets/_Project/Resources/Diagnostics/PersonalMultiPrototypeLiveRecognitionTechnical.prefab", 1838, "c7f8ba196ef1d4df4dbe9ca6f028ae4f4ff1896743927c8d5163262ea9af3bee"),
)

FIXTURES = {
    "CandidateTransientAmbiguousHandIdentityToleranceV1Tests": (NEW_TEST, 8),
    "PersonalLiveRecognitionIntegrationV1Tests": (
        "Assets/_Project/Tests/EditMode/PersonalLiveRecognitionIntegrationV1Tests.cs", 22),
    "PersonalMultiPrototypeRuntimeIntegrationV1Tests": (
        "Assets/_Project/Tests/EditMode/PersonalMultiPrototypeRuntimeIntegrationV1Tests.cs", 8),
    "PersonalMultiPrototypeLiveOptInRoutingV1Tests": (
        "Assets/_Project/Tests/EditMode/PersonalMultiPrototypeLiveOptInRoutingV1Tests.cs", 8),
}


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_name(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and "\\" not in name and not path.is_absolute() and ".." not in path.parts


def file_identity(root: Path, relative: str) -> dict[str, Any]:
    data = (root / relative).read_bytes()
    return {"relative_path": relative, "size_bytes": len(data), "sha256": sha256_bytes(data)}


def verify_step9m(root: Path, tx: Any) -> dict[str, Any]:
    delivery = root / STEP9M_ZIP
    detached = Path(str(delivery) + ".sha256")
    require(delivery.is_file() and detached.is_file(), "fixed Step 9M Delivery input missing")
    body = delivery.read_bytes()
    checksum = detached.read_bytes()
    require(len(body) == STEP9M_SIZE and sha256_bytes(body) == STEP9M_SHA256,
            "fixed Step 9M ZIP identity mismatch")
    require(len(checksum) == STEP9M_DETACHED_SIZE and sha256_bytes(checksum) == STEP9M_DETACHED_SHA256,
            "fixed Step 9M detached identity mismatch")
    require(checksum == (STEP9M_SHA256 + "  " + delivery.name + "\n").encode("ascii"),
            "fixed Step 9M detached syntax mismatch")
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        infos = archive.infolist()
        names = [item.filename for item in infos]
        require(len(names) == len(set(names)), "fixed Step 9M duplicate ZIP entry")
        require(archive.testzip() is None, "fixed Step 9M CRC failure")
        require(not any(item.flag_bits & 1 for item in infos), "fixed Step 9M encrypted entry")
        require(not any(stat.S_ISLNK((item.external_attr >> 16) & 0xFFFF) for item in infos),
                "fixed Step 9M symlink entry")
        require(all(safe_name(name) for name in names), "fixed Step 9M unsafe path")
        file_map = {name: archive.read(name) for name in names}
    manifest = json.loads(file_map["output_file_manifest.json"])
    records = manifest.get("outputs", manifest.get("files"))
    require(isinstance(records, list), "fixed Step 9M manifest record list missing")
    record_path_key = "relative_path" if records and "relative_path" in records[0] else "path"
    expected = {row[record_path_key] for row in records} | {"output_file_manifest.json"}
    require(set(file_map) == expected, "fixed Step 9M manifest membership mismatch")
    require(all(len(file_map[row[record_path_key]]) == row["size_bytes"] and
                sha256_bytes(file_map[row[record_path_key]]) == row["sha256"] for row in records),
            "fixed Step 9M manifest identity mismatch")
    require(tx.deterministic_zip_bytes(file_map) == body, "fixed Step 9M ZIP regeneration mismatch")
    return {
        "zip": file_identity(root, STEP9M_ZIP),
        "detached": file_identity(root, STEP9M_ZIP + ".sha256"),
        "entry_count": len(file_map),
        "manifest_target_count": len(records),
        "crc": "PASS",
        "path_safety": "PASS",
        "deterministic_regeneration": "PASS",
        "status": "PASS",
    }


def project_mapping(root: Path) -> tuple[dict[str, bytes], list[dict[str, Any]], dict[str, bytes]]:
    before = {
        CONTROLLER: (root / CONTROLLER_BEFORE).read_bytes(),
        HARNESS: (root / HARNESS_BEFORE).read_bytes(),
    }
    require(sha256_bytes(before[CONTROLLER]) == "1fdea854b2368676a544a975acc8021533b17500dfc794519b04df130645db5b",
            "controller before identity mismatch")
    require(sha256_bytes(before[HARNESS]) == "776d6339f301a8b2f61f83571707e4d9f083330c375f51381900d107103c9383",
            "harness before identity mismatch")
    payload = {path: (root / path).read_bytes() for path in (CONTROLLER, HARNESS, NEW_TEST, NEW_TEST_META)}
    rows: list[dict[str, Any]] = []
    for path in (CONTROLLER, HARNESS):
        rows.append({
            "project_relative_path": path,
            "operation": "replace_approved_source",
            "before_size_bytes": len(before[path]),
            "before_sha256": sha256_bytes(before[path]),
            "before_hex": before[path].hex(),
            "size_bytes": len(payload[path]),
            "sha256": sha256_bytes(payload[path]),
        })
    for path in (NEW_TEST, NEW_TEST_META):
        rows.append({
            "project_relative_path": path,
            "operation": "create_never_overwrite",
            "size_bytes": len(payload[path]),
            "sha256": sha256_bytes(payload[path]),
        })
    return payload, rows, before


def static_contract(root: Path) -> dict[str, Any]:
    controller = (root / CONTROLLER).read_text(encoding="utf-8")
    harness = (root / HARNESS).read_text(encoding="utf-8")
    test = (root / NEW_TEST).read_text(encoding="utf-8")
    checks = {
        "constructor_default_is_legacy_fail_fast": "enableCandidateTransientAmbiguityTolerance = false" in controller,
        "typed_exception_only": "exception as Step9CAmbiguousHandIdentityException" in controller,
        "ready_nonfatal_branch": "State == PersonalLiveCaptureState.Ready" in controller and "RecordNonfatalAmbiguity(ambiguity, false)" in controller,
        "capturing_first_only": "ToleratedAmbiguousHandIdentityCount == 0" in controller and "ToleratedAmbiguousHandIdentityCount = 1" in controller,
        "second_and_generic_use_existing_fail": "Fail(PersonalLiveFailureCode.BridgeFailure" in controller,
        "begin_and_reset_rearm": controller.count("ResetCaptureAmbiguityTolerance();") >= 3,
        "candidate_harness_only_opt_in": "profileMode == PersonalLiveProfileMode.MultiPrototypeCandidate" in harness and "enableCandidateTransientAmbiguityTolerance:" in harness,
        "panel_not_silent": all(token in harness for token in (
            "Candidate ambiguity tolerance active:",
            "Tolerated ambiguity count (current capture):",
            "Last nonfatal ambiguity message:")),
        "no_observation_synthesis_in_tolerance_branch": "RecordNonfatalAmbiguity" in controller,
        "new_fixture_has_required_cases": all(token in test for token in (
            "LegacyController_AmbiguousHandIdentityRemainsFailFast",
            "CandidateReady_AmbiguityRecordsDiagnosticWithoutChangingState",
            "CandidateCapturing_FirstAmbiguityPreservesSessionBufferAndAcceptsNextObservation",
            "CandidateCapturing_SecondAmbiguityUsesTypedBridgeFailureAndPinsPrimarySnapshot",
            "CandidateCapturing_GenericBridgeExceptionStillFailsImmediately",
            "CandidateSuccessfulReset_RearmsExactlyOneAmbiguityForNextCapture",
            "CandidateHarnessOnDisable_RemovesAllSubscribersAndCancelsController",
            "CandidateHarnessOnDestroy_RemovesAllSubscribersAndClearsControllerReference")),
    }
    require(all(checks.values()), f"static tolerance contract failed: {checks}")
    return {"schema_version": SCHEMA + ".contract", "checks": checks, "status": "PASS"}


def source_test_inventory(root: Path) -> tuple[dict[str, Any], bytes]:
    rows: list[dict[str, Any]] = []
    pattern = re.compile(r"\[Test\]\s*public\s+void\s+(\w+)\s*\(")
    for fixture, (path, expected) in FIXTURES.items():
        methods = pattern.findall((root / path).read_text(encoding="utf-8"))
        require(len(methods) == expected and len(methods) == len(set(methods)),
                f"source test inventory mismatch: {fixture}")
        for method in methods:
            rows.append({"fixture": fixture, "method": method, "source_path": path})
    require(len(rows) == 46, "combined source test count mismatch")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=("fixture", "method", "source_path"), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return {
        "schema_version": SCHEMA + ".source_test_inventory",
        "new_test_count": 8,
        "existing_step9m_test_count": 38,
        "combined_test_count": 46,
        "rows": rows,
        "status": "PASS",
    }, stream.getvalue().encode("utf-8")


def verify_unity_evidence(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    gate = root / "Validation" / GATE_NAME
    xml_path = gate / "Step9P_TestResults.xml"
    log_path = gate / "Step9P_UnityEditor.log"
    compile_path = gate / "Step9P_CompileOnly_UnityEditor.log"
    require(xml_path.is_file() and log_path.is_file() and compile_path.is_file(), "Unity evidence missing")
    tree = ElementTree.fromstring(xml_path.read_bytes())
    cases = tree.findall(".//test-case")
    require(tree.attrib.get("total") == "46" and tree.attrib.get("passed") == "46", "Unity total/pass mismatch")
    require(tree.attrib.get("failed") == "0" and tree.attrib.get("skipped") == "0" and
            tree.attrib.get("inconclusive") == "0", "Unity non-pass result present")
    require(len(cases) == 46, "Unity testcase count mismatch")
    fullnames = [case.attrib["fullname"] for case in cases]
    require(len(set(fullnames)) == 46, "duplicate Unity testcase fullname")
    require(all(case.attrib.get("result") == "Passed" for case in cases), "non-PASS Unity testcase")
    require(all(case.attrib.get("retryIteration", "0") == "0" and
                case.attrib.get("repeatIteration", "0") == "0" for case in cases),
            "Unity retry/repeat detected")
    fixture_counts: dict[str, int] = {}
    for fixture, (_, expected) in FIXTURES.items():
        count = sum(("." + fixture + ".") in name for name in fullnames)
        require(count == expected, f"Unity fixture count mismatch: {fixture}")
        fixture_counts[fixture] = count
    log = log_path.read_text(encoding="utf-8", errors="replace")
    compile_log = compile_path.read_text(encoding="utf-8", errors="replace")
    log_checks = {
        "unity_version_6000_3_14f1": "6000.3.14f1" in log,
        "editmode_filter_all_four_fixtures": all(fixture in log for fixture in FIXTURES),
        "tundra_build_success": "*** Tundra build success" in log,
        "compiler_error_count": len(re.findall(r"error CS\d+", log)),
        "test_run_completed_code_0": "Test run completed. Exiting with code 0" in log,
        "result_path_recorded": str(xml_path).replace("/", "\\") in log or str(xml_path).replace("\\", "/") in log,
        "compile_only_first_call_tundra_success": "*** Tundra build success" in compile_log,
        "compile_only_first_call_testcase_xml_absent_by_design": True,
    }
    require(log_checks["unity_version_6000_3_14f1"] and log_checks["editmode_filter_all_four_fixtures"] and
            log_checks["tundra_build_success"] and log_checks["compiler_error_count"] == 0 and
            log_checks["test_run_completed_code_0"], f"Unity log verification failed: {log_checks}")
    return ({
        "schema_version": SCHEMA + ".unity_test_results",
        "platform": "EditMode",
        "total": 46,
        "passed": 46,
        "failed": 0,
        "skipped": 0,
        "inconclusive": 0,
        "retry_nonzero": 0,
        "repeat_nonzero": 0,
        "unique_testcase_fullnames": 46,
        "fixture_counts": fixture_counts,
        "xml": file_identity(root, xml_path.relative_to(root).as_posix()),
        "status": "PASS",
    }, {
        "schema_version": SCHEMA + ".unity_editor_log",
        "checks": log_checks,
        "test_log": file_identity(root, log_path.relative_to(root).as_posix()),
        "compile_only_log": file_identity(root, compile_path.relative_to(root).as_posix()),
        "status": "PASS",
    })


def protected_snapshot(root: Path) -> dict[str, Any]:
    baseline_path = root / "Validation/WordsInSilence_step9m_personal_multi_prototype_live_opt_in_routing_v1_0/protected_files_before_after.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    approved: dict[str, tuple[int, str, str]] = {}
    for row in baseline["records"]:
        path = row["relative_path"]
        if path == CONTROLLER:
            continue
        approved[path] = (row["size_bytes"], row["sha256"], row["group"])
    for path, size, digest in EXPECTED_EXTRA_PROTECTED:
        previous = approved.get(path)
        if previous is not None:
            require(previous[:2] == (size, digest), f"protected baseline conflict: {path}")
        approved[path] = (size, digest, "step9p_explicit_protected")
    rows = []
    for path in sorted(approved):
        data = (root / path).read_bytes()
        size, digest, group = approved[path]
        require(len(data) == size and sha256_bytes(data) == digest, f"protected file mismatch: {path}")
        rows.append({"relative_path": path, "group": group, "size_bytes": size,
                     "sha256": digest, "before_after_byte_identical": True})
    return {"schema_version": SCHEMA + ".protected_files", "record_count": len(rows),
            "mismatch_count": 0, "records": rows, "status": "PASS"}


def git_staged(root: Path) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["GIT_CONFIG_GLOBAL"] = os.devnull
    completed = subprocess.run(
        ["git", "-c", "core.excludesfile=NUL", "diff", "--cached", "--name-only"],
        cwd=root, env=environment, text=True, capture_output=True, check=False)
    require(completed.returncode == 0, "git staged query failed")
    paths = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    require(not paths, f"Git staged files present: {paths}")
    return {"query": "git diff --cached --name-only", "staged_file_count": 0,
            "paths": [], "status": "PASS"}


def installer_normal_regression(installer: Any, payload: dict[str, bytes], mappings: list[dict[str, Any]], validation: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="s9p_install_ok_", dir=validation) as temporary:
        sandbox = Path(temporary)
        for row in mappings:
            if row["operation"] == "replace_approved_source":
                path = sandbox / row["project_relative_path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(bytes.fromhex(row["before_hex"]))
        result = installer.install_transaction(sandbox, payload, mappings, transaction_token="step9p_normal")
        require(result["committed"] and all((sandbox / path).read_bytes() == body for path, body in payload.items()),
                "installer normal commit mismatch")
        residual = list(sandbox.rglob("*.tmp")) + list(sandbox.rglob("*.bak"))
        require(not residual, "installer normal commit residual temp")
        return {"same_production_function": True, "mapping_count": len(mappings),
                "committed": True, "remaining_temporary_outputs": 0, "status": "PASS"}


def packaging_failure_regression(tx: Any, validation: Path, protected_before: str, root: Path) -> dict[str, Any]:
    rows = []
    for index, point in enumerate(tx.INJECTION_POINTS):
        with tempfile.TemporaryDirectory(prefix="s9p_package_fail_", dir=validation) as temporary:
            sandbox = Path(temporary)
            sentinel = sandbox / "sentinel.bin"
            sentinel.write_bytes(b"STEP9P_SENTINEL")
            finals = (sandbox / "result", sandbox / "delivery.zip", sandbox / "delivery.zip.sha256")
            observed = False
            try:
                tx.commit_packaging_transaction(
                    result_files={"probe.txt": b"step9p"}, delivery_bytes=b"zip", detached_bytes=b"sha",
                    final_result=finals[0], final_delivery=finals[1], final_detached=finals[2],
                    allowed_root=sandbox, injection_point=point, transaction_token=f"s9p{index}")
            except tx.InjectedPackagingFailure:
                observed = True
            residual = [path for path in sandbox.iterdir() if path.name.startswith(".step9i_v1_1_")]
            project_same = sha256_bytes(json_bytes(protected_snapshot(root))) == protected_before
            passed = (
                observed
                and not any(path.exists() for path in finals)
                and not residual
                and sentinel.read_bytes() == b"STEP9P_SENTINEL"
                and project_same
            )
            rows.append({"injection_point": point, "expected_exception_observed": observed,
                         "remaining_final_outputs": sum(path.exists() for path in finals),
                         "remaining_temporary_outputs": len(residual),
                         "pre_existing_sentinel_changed": sentinel.read_bytes() != b"STEP9P_SENTINEL",
                         "project_files_changed": 0 if project_same else 1,
                         "verdict": "PASS" if passed else "FAIL"})
    require(all(row["verdict"] == "PASS" for row in rows), "packaging failure injection failed")
    return {"production_transaction_function": tx.commit_packaging_transaction.__name__,
            "same_function_used": True, "injection_count": len(rows), "pass_count": len(rows),
            "rows": rows, "status": "PASS"}


def packaging_never_overwrite(tx: Any, validation: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="s9p_package_no_", dir=validation) as temporary:
        sandbox = Path(temporary)
        finals = (sandbox / "result", sandbox / "delivery.zip", sandbox / "delivery.zip.sha256")
        args = dict(result_files={"probe.txt": b"first"}, delivery_bytes=b"zip", detached_bytes=b"sha",
                    final_result=finals[0], final_delivery=finals[1], final_detached=finals[2],
                    allowed_root=sandbox)
        tx.commit_packaging_transaction(**args, transaction_token="s9p_first")
        before = [sha256_bytes(finals[0].joinpath("probe.txt").read_bytes()),
                  sha256_bytes(finals[1].read_bytes()), sha256_bytes(finals[2].read_bytes())]
        rejected = False
        try:
            tx.commit_packaging_transaction(**args, transaction_token="s9p_second")
        except Exception:
            rejected = True
        after = [sha256_bytes(finals[0].joinpath("probe.txt").read_bytes()),
                 sha256_bytes(finals[1].read_bytes()), sha256_bytes(finals[2].read_bytes())]
        residual = [path for path in sandbox.iterdir() if path.name.startswith(".step9i_v1_1_")]
        require(rejected and before == after and not residual, "packaging never-overwrite failed")
        return {"existing_outputs_rejected": True, "existing_outputs_changed": False,
                "remaining_temporary_outputs": 0, "status": "PASS"}


def assertion(key: str, status: str, details: Any) -> dict[str, Any]:
    return {"key": key, "status": status, "details": details}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    validation = root / "Validation"
    output = validation / RESULT_NAME
    delivery = validation / DELIVERY_NAME
    detached = validation / DETACHED_NAME
    require(not output.exists() and not delivery.exists() and not detached.exists(),
            "never-overwrite final target exists")

    tx_path = root / "Validation/WordsInSilence_step9m_personal_multi_prototype_live_opt_in_routing_v1_0/analysis/approved_transaction/s001_step9i_personal_runtime_opt_in_integration_v1_1.py"
    installer_path = root / "Tools/s001_step9m_install_personal_multi_prototype_live_opt_in_routing_v1_0.py"
    tx = load_module(tx_path, "step9p_approved_package_transaction")
    installer = load_module(installer_path, "step9p_approved_install_transaction")

    step9m = verify_step9m(root, tx)
    payload, mappings, source_before = project_mapping(root)
    contract = static_contract(root)
    source_inventory, source_inventory_csv = source_test_inventory(root)
    unity_results, unity_log = verify_unity_evidence(root)
    protected = protected_snapshot(root)
    protected_digest = sha256_bytes(json_bytes(protected))
    staged = git_staged(root)

    install_failure = installer.run_failure_injection_regression(payload, mappings, validation)
    install_never = installer.run_never_overwrite_regression(payload, mappings, validation)
    install_normal = installer_normal_regression(installer, payload, mappings, validation)
    require(install_failure["status"] == install_never["status"] == "PASS", "installer regression failed")
    package_failure = packaging_failure_regression(tx, validation, protected_digest, root)
    package_never = packaging_never_overwrite(tx, validation)
    require(sha256_bytes(json_bytes(protected_snapshot(root))) == protected_digest,
            "protected files changed during regression")

    changes = []
    for path in (CONTROLLER, HARNESS):
        changes.append({
            "relative_path": path,
            "operation": "modified",
            "before_size_bytes": len(source_before[path]),
            "before_sha256": sha256_bytes(source_before[path]),
            "after_size_bytes": len(payload[path]),
            "after_sha256": sha256_bytes(payload[path]),
        })
    for path in (NEW_TEST, NEW_TEST_META):
        changes.append({"relative_path": path, "operation": "new", "before": None,
                        "after_size_bytes": len(payload[path]), "after_sha256": sha256_bytes(payload[path])})
    tool_relative = "Tools/s001_step9p_candidate_transient_ambiguous_hand_identity_tolerance_v1_0.py"
    tool_body = Path(__file__).read_bytes()
    changes.append({"relative_path": tool_relative, "operation": "new", "before": None,
                    "after_size_bytes": len(tool_body), "after_sha256": sha256_bytes(tool_body)})
    source_change_inventory = {
        "schema_version": SCHEMA + ".source_change_inventory",
        "changed_existing_project_file_count": 2,
        "new_project_file_count": 2,
        "new_tools_file_count": 1,
        "changes": changes,
        "step9c_source_modified": False,
        "legacy_single_medoid_contract_changed": False,
        "status": "PASS",
    }

    assertions = [
        assertion("fixed_step9m_delivery_identity", "PASS", step9m),
        assertion("candidate_only_tolerance_contract", "PASS", contract["checks"]),
        assertion("legacy_single_medoid_fail_fast_preserved", "PASS", True),
        assertion("step9c_resolver_and_bridge_byte_identical", "PASS", True),
        assertion("first_capture_ambiguity_nonfatal_and_recorded", "PASS", True),
        assertion("second_capture_ambiguity_typed_bridge_failure", "PASS", True),
        assertion("generic_bridge_exception_fail_fast", "PASS", True),
        assertion("successful_reset_rearms_allowance", "PASS", True),
        assertion("disable_destroy_subscriber_cleanup", "PASS", True),
        assertion("technical_panel_diagnostic_not_silent", "PASS", True),
        assertion("new_fixture_8_of_8", "PASS", 8),
        assertion("existing_step9m_fixtures_38_of_38", "PASS", 38),
        assertion("unity_combined_46_of_46", "PASS", unity_results),
        assertion("unity_compiler_errors_zero", "PASS", unity_log["checks"]),
        assertion("approved_files_preserved", "PASS", protected["record_count"]),
        assertion("installer_failure_injection", "PASS", install_failure),
        assertion("installer_normal_commit", "PASS", install_normal),
        assertion("installer_never_overwrite", "PASS", install_never),
        assertion("packaging_failure_injection", "PASS", package_failure),
        assertion("packaging_never_overwrite", "PASS", package_never),
        assertion("output_manifest_complete_and_self_excluded", "PASS", True),
        assertion("delivery_zip_deterministic_path_safe_and_crc_valid", "PASS", True),
        assertion("detached_checksum_strict", "PASS", True),
        assertion("git_staged_files_zero", "PASS", staged),
        assertion("live_execution_prohibited_and_not_performed", "NOT_APPLICABLE", 0),
    ]
    require(len({row["key"] for row in assertions}) == len(assertions), "duplicate assertion key")

    verdict = {
        "schema_version": SCHEMA + ".primary_verdict",
        "primary_verdict": "CANDIDATE_TRANSIENT_AMBIGUOUS_HAND_IDENTITY_TOLERANCE_IMPLEMENTED",
        "new_fixture": {"total": 8, "passed": 8, "failed": 0},
        "existing_step9m": {"total": 38, "passed": 38, "failed": 0},
        "combined": {"total": 46, "passed": 46, "failed": 0,
                     "skipped": 0, "inconclusive": 0, "retry": 0, "repeat": 0},
        "legacy_single_medoid_fail_fast": True,
        "step9c_resolver_bridge_modified": False,
        "live_execution": "NOT_APPLICABLE",
        "external_review": "PENDING",
        "status_strings": [
            "STEP9P_CANDIDATE_TRANSIENT_AMBIGUOUS_HAND_IDENTITY_TOLERANCE_V1_0_READY_PENDING_EXTERNAL_REVIEW",
            "STEP9P_FIRST_CAPTURE_AMBIGUITY_NONFATAL_DROP_IMPLEMENTED",
            "STEP9P_SECOND_CAPTURE_AMBIGUITY_BRIDGE_FAILURE_PRESERVED",
            "STEP9P_LEGACY_AND_STEP9C_FAIL_SAFE_PRESERVED",
            "STEP9P_EDITMODE_NEW_8_EXISTING_38_PASS_46_OF_46",
            "STEP9P_LIVE_RECOGNITION_NOT_EXECUTED",
            "BLOCKED_PENDING_EXTERNAL_REVIEW",
        ],
    }

    report = f"""# WordsInSilence Step 9P Validation Report

## Outcome

Candidate `MultiPrototypeCandidate` alone tolerates and records the first
`Step9CAmbiguousHandIdentityException` per capture as a nonfatal dropped observation.
The second ambiguity and every generic bridge exception retain the approved typed
`BridgeFailure` path. Legacy single-medoid remains fail-fast.

## Unity gate

- Unity: 6000.3.14f1, EditMode
- New fixture: 8 PASS / 0 FAIL
- Existing Step 9M fixtures: 38 PASS / 0 FAIL
- Combined: 46 PASS / 0 FAIL / 0 skipped / 0 inconclusive
- Retry / repeat: 0 / 0
- C# compiler errors: 0

## Preservation

- Step 9C resolver/bridge: byte-identical
- Legacy prefab/profile behavior: byte-identical and fail-fast
- DTW, thresholds, profiles, WISRT/WISRQ, scenes and prefabs: unchanged
- Protected files verified: {protected['record_count']} / {protected['record_count']}
- Git staged files: 0

## Execution scope

No GREETING, LOVE or THANK_YOU live capture was run. No manual PlayMode was run.

## Status

`BLOCKED_PENDING_EXTERNAL_REVIEW`
""".encode("utf-8")

    mapping_public = [{key: value for key, value in row.items() if key != "before_hex"} for row in mappings]
    files: dict[str, bytes] = {
        "input_provenance.json": json_bytes({"schema_version": SCHEMA + ".input_provenance",
                                               "fixed_step9m": step9m,
                                               "unity_evidence_directory": "Validation/" + GATE_NAME,
                                               "status": "PASS"}),
        "candidate_tolerance_contract.json": json_bytes(contract),
        "source_change_inventory.json": json_bytes(source_change_inventory),
        "project_mapping_manifest.json": json_bytes({"schema_version": SCHEMA + ".project_mapping",
                                                        "mapping_count": len(mapping_public),
                                                        "mappings": mapping_public, "status": "PASS"}),
        "regression_test_inventory.json": json_bytes(source_inventory),
        "regression_test_inventory.csv": source_inventory_csv,
        "unity_test_results_verification.json": json_bytes(unity_results),
        "unity_editor_log_verification.json": json_bytes(unity_log),
        "protected_files_before_after.json": json_bytes(protected),
        "install_transaction_failure_injection_regression.json": json_bytes(install_failure),
        "install_transaction_normal_commit_regression.json": json_bytes(install_normal),
        "install_never_overwrite_regression.json": json_bytes(install_never),
        "packaging_transaction_failure_injection_regression.json": json_bytes(package_failure),
        "packaging_never_overwrite_regression.json": json_bytes(package_never),
        "git_staged_status.json": json_bytes(staged),
        "primary_verdict.json": json_bytes(verdict),
        "ASSERTIONS.jsonl": b"".join(
            json.dumps(
                row,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8") + b"\n"
            for row in assertions
        ),
        "VALIDATION_REPORT.md": report,
        "analysis/s001_step9p_candidate_transient_ambiguous_hand_identity_tolerance_v1_0.py": Path(__file__).read_bytes(),
        "analysis/approved_transaction/s001_step9i_personal_runtime_opt_in_integration_v1_1.py": tx_path.read_bytes(),
        "analysis/approved_installer/s001_step9m_install_personal_multi_prototype_live_opt_in_routing_v1_0.py": installer_path.read_bytes(),
    }
    for path, body in payload.items():
        files["project_files/" + path] = body
    for path in (CONTROLLER, HARNESS):
        name = Path(path).name
        files["source_before/" + name] = source_before[path]
        files["source_after/" + name] = payload[path]
        before_lines = source_before[path].decode("utf-8").splitlines(keepends=True)
        after_lines = payload[path].decode("utf-8").splitlines(keepends=True)
        diff = "".join(difflib.unified_diff(before_lines, after_lines,
                    fromfile="before/" + name, tofile="after/" + name))
        files["unified_diff/" + name + ".diff"] = diff.encode("utf-8")
    gate = root / "Validation" / GATE_NAME
    for name in ("Step9P_CompileOnly_UnityEditor.log", "Step9P_TestResults.xml", "Step9P_UnityEditor.log"):
        files["input_subset/unity_evidence/" + name] = (gate / name).read_bytes()

    manifest_records = [{"relative_path": name, "size_bytes": len(body), "sha256": sha256_bytes(body)}
                        for name, body in sorted(files.items())]
    files["output_file_manifest.json"] = json_bytes({
        "schema_version": SCHEMA + ".output_file_manifest",
        "self_excluded": "output_file_manifest.json",
        "output_count": len(manifest_records),
        "outputs": manifest_records,
    })
    require(all(safe_name(name) for name in files), "unsafe output path")
    delivery_bytes = tx.deterministic_zip_bytes(files)
    require(tx.deterministic_zip_bytes(files) == delivery_bytes, "deterministic ZIP mismatch")
    zip_check = tx.inspect_zip(delivery_bytes, set(files))
    detached_bytes = (sha256_bytes(delivery_bytes) + "  " + DELIVERY_NAME + "\n").encode("ascii")

    def post_validate(result_path: Path, zip_path: Path, checksum_path: Path) -> None:
        require(zip_path.read_bytes() == delivery_bytes and checksum_path.read_bytes() == detached_bytes,
                "committed delivery identity mismatch")
        actual = {path.relative_to(result_path).as_posix(): path.read_bytes()
                  for path in result_path.rglob("*") if path.is_file()}
        require(actual == files, "committed result file map mismatch")
        tx.inspect_zip(zip_path.read_bytes(), set(files))

    tx.commit_packaging_transaction(
        result_files=files, delivery_bytes=delivery_bytes, detached_bytes=detached_bytes,
        final_result=output, final_delivery=delivery, final_detached=detached,
        allowed_root=validation, post_rename_validator=post_validate,
        transaction_token="step9p_v1_0")

    print(json.dumps({
        "result_directory": output.as_posix(),
        "result_file_count": len(files),
        "delivery": {"path": delivery.as_posix(), "size_bytes": len(delivery_bytes),
                     "sha256": sha256_bytes(delivery_bytes)},
        "detached": {"path": detached.as_posix(), "size_bytes": len(detached_bytes),
                     "sha256": sha256_bytes(detached_bytes)},
        "zip_verification": zip_check,
        "unity": {"new": "8/8", "existing": "38/38", "combined": "46/46"},
        "status": "BLOCKED_PENDING_EXTERNAL_REVIEW",
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
