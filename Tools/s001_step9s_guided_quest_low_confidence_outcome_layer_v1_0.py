#!/usr/bin/env python3
"""Validate and package the Step 9S guided quest outcome layer."""

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


SCHEMA = "words_in_silence.step9s.guided_quest_low_confidence_outcome_layer.v1_0"
RESULT_NAME = "WordsInSilence_step9s_guided_quest_low_confidence_outcome_layer_v1_0"
DELIVERY_NAME = RESULT_NAME + "_delivery.zip"
DETACHED_NAME = DELIVERY_NAME + ".sha256"
GATE_NAME = RESULT_NAME + "_unity_editor_gate"

HARNESS = "Assets/_Project/Scripts/Recognition/LiveInput/PersonalLiveRecognitionHarness.cs"
EVALUATOR = "Assets/_Project/Scripts/Recognition/LiveInput/GuidedQuestOutcomeEvaluator.cs"
EVALUATOR_META = EVALUATOR + ".meta"
TEST = "Assets/_Project/Tests/EditMode/GuidedQuestLowConfidenceOutcomeLayerV1Tests.cs"
TEST_META = TEST + ".meta"
CONTROLLER = "Assets/_Project/Scripts/Recognition/LiveInput/PersonalLiveRecognitionController.cs"

STEP9P_RESULT = "Validation/WordsInSilence_step9p_candidate_transient_ambiguous_hand_identity_tolerance_v1_0"
STEP9P_ZIP = STEP9P_RESULT + "_delivery.zip"
STEP9P_SIZE = 95_982
STEP9P_SHA256 = "e8d0eec2f09591ccc2b4dd18b5b1a4d2ae57c05fa0628e620f864a9a32ce6311"
STEP9P_DETACHED_SIZE = 160
STEP9P_DETACHED_SHA256 = "c2c060bf7fd5cf6c418a3055fdaf00e4ff09b654eabba6317c6f349f8dd634e7"

EXISTING_FIXTURES = {
    "CandidateTransientAmbiguousHandIdentityToleranceV1Tests": (
        "Assets/_Project/Tests/EditMode/CandidateTransientAmbiguousHandIdentityToleranceV1Tests.cs", 8),
    "PersonalLiveRecognitionIntegrationV1Tests": (
        "Assets/_Project/Tests/EditMode/PersonalLiveRecognitionIntegrationV1Tests.cs", 22),
    "PersonalMultiPrototypeRuntimeIntegrationV1Tests": (
        "Assets/_Project/Tests/EditMode/PersonalMultiPrototypeRuntimeIntegrationV1Tests.cs", 8),
    "PersonalMultiPrototypeLiveOptInRoutingV1Tests": (
        "Assets/_Project/Tests/EditMode/PersonalMultiPrototypeLiveOptInRoutingV1Tests.cs", 8),
}

FIXED_LIVE_CASES = {
    "STEP9N_GREETING": {
        "session_id": "8a560d30-bfc7-4a79-b7a4-fc9f84e2efb4",
        "state": "Completed", "raw_frames": 149, "normalized_frames": 149,
        "expected_word": "GREETING", "predicted_word": "GREETING",
        "accepted": True, "expected_word_class_accept": True, "quest_success": True,
        "distances": {"GREETING": 0.43359546744230326, "LOVE": 0.68465225560791554,
                      "THANK_YOU": 0.76960002842331465},
        "expected_guided_outcome": "StrictSuccess",
    },
    "STEP9Q_LOVE": {
        "session_id": "024e2b57-eea7-4f5e-b4c4-83a2a05b5383",
        "state": "Completed", "raw_frames": 110, "normalized_frames": 110,
        "expected_word": "LOVE", "predicted_word": "LOVE",
        "accepted": True, "expected_word_class_accept": True, "quest_success": True,
        "expected_guided_outcome": "StrictSuccess",
    },
    "STEP9R_THANK_YOU": {
        "session_id": "3aa213e9-1a94-454f-b53c-6ec67462d0ea",
        "state": "Completed", "raw_frames": 92, "normalized_frames": 92,
        "expected_word": "THANK_YOU", "predicted_word": "THANK_YOU",
        "accepted": False, "expected_word_class_accept": False, "quest_success": False,
        "distances": {"GREETING": 0.7380835730520241, "LOVE": 0.47699284383599257,
                      "THANK_YOU": 0.44939933821265732},
        "approved_threshold": 0.37709977431991587,
        "expected_guided_outcome": "LowConfidenceCorrect",
        "step9r_original_verdict": "FAIL",
    },
}


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def compact_json_line(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_name(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and "\\" not in name and not path.is_absolute() and ".." not in path.parts


def identity(path: Path, root: Path) -> dict[str, Any]:
    body = path.read_bytes()
    return {"relative_path": path.relative_to(root).as_posix(), "size_bytes": len(body),
            "sha256": sha256_bytes(body)}


def verify_step9p(root: Path, tx: Any) -> dict[str, Any]:
    delivery = root / STEP9P_ZIP
    detached = Path(str(delivery) + ".sha256")
    body = delivery.read_bytes()
    check = detached.read_bytes()
    require(len(body) == STEP9P_SIZE and sha256_bytes(body) == STEP9P_SHA256,
            "fixed Step 9P ZIP identity mismatch")
    require(len(check) == STEP9P_DETACHED_SIZE and sha256_bytes(check) == STEP9P_DETACHED_SHA256,
            "fixed Step 9P detached identity mismatch")
    require(check == (STEP9P_SHA256 + "  " + delivery.name + "\n").encode("ascii"),
            "fixed Step 9P detached syntax mismatch")
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        infos = archive.infolist()
        names = [item.filename for item in infos]
        require(len(names) == len(set(names)), "Step 9P duplicate ZIP entry")
        require(archive.testzip() is None, "Step 9P CRC failure")
        require(not any(item.flag_bits & 1 for item in infos), "Step 9P encrypted entry")
        require(not any(stat.S_ISLNK((item.external_attr >> 16) & 0xFFFF) for item in infos),
                "Step 9P symlink entry")
        require(all(safe_name(name) for name in names), "Step 9P unsafe path")
        file_map = {name: archive.read(name) for name in names}
    manifest = json.loads(file_map["output_file_manifest.json"])
    records = manifest["outputs"]
    expected = {row["relative_path"] for row in records} | {"output_file_manifest.json"}
    require(set(file_map) == expected, "Step 9P manifest membership mismatch")
    require(all(len(file_map[row["relative_path"]]) == row["size_bytes"] and
                sha256_bytes(file_map[row["relative_path"]]) == row["sha256"] for row in records),
            "Step 9P manifest identity mismatch")
    require(tx.deterministic_zip_bytes(file_map) == body, "Step 9P deterministic ZIP mismatch")
    return {"zip": identity(delivery, root), "detached": identity(detached, root),
            "entry_count": len(file_map), "manifest_target_count": len(records),
            "crc": "PASS", "safety": "PASS", "deterministic_regeneration": "BYTE_IDENTICAL",
            "status": "PASS"}


def project_mapping(root: Path) -> tuple[dict[str, bytes], list[dict[str, Any]], bytes]:
    before_path = root / STEP9P_RESULT / "project_files" / HARNESS
    before = before_path.read_bytes()
    require(len(before) == 12_351 and sha256_bytes(before) ==
            "90a17b6399db78a2370ec40c406958e39a9e0aeafb80a684b467cf6989b03740",
            "approved Step 9P harness baseline mismatch")
    payload = {path: (root / path).read_bytes()
               for path in (HARNESS, EVALUATOR, EVALUATOR_META, TEST, TEST_META)}
    mappings = [{
        "project_relative_path": HARNESS, "operation": "replace_approved_source",
        "before_size_bytes": len(before), "before_sha256": sha256_bytes(before),
        "before_hex": before.hex(), "size_bytes": len(payload[HARNESS]),
        "sha256": sha256_bytes(payload[HARNESS]),
    }]
    for path in (EVALUATOR, EVALUATOR_META, TEST, TEST_META):
        mappings.append({"project_relative_path": path, "operation": "create_never_overwrite",
                         "size_bytes": len(payload[path]), "sha256": sha256_bytes(payload[path])})
    return payload, mappings, before


def protected_snapshot(root: Path) -> dict[str, Any]:
    base = json.loads((root / STEP9P_RESULT / "protected_files_before_after.json").read_text(encoding="utf-8"))
    expected = {row["relative_path"]: (row["size_bytes"], row["sha256"], row["group"])
                for row in base["records"]}
    mapping = json.loads((root / STEP9P_RESULT / "project_mapping_manifest.json").read_text(encoding="utf-8"))
    for row in mapping["mappings"]:
        path = row["project_relative_path"]
        if path == HARNESS:
            continue
        expected[path] = (row["size_bytes"], row["sha256"], "step9p_approved_project_file")
    rows = []
    for path in sorted(expected):
        body = (root / path).read_bytes()
        size, digest, group = expected[path]
        require(len(body) == size and sha256_bytes(body) == digest, f"protected file mismatch: {path}")
        rows.append({"relative_path": path, "group": group, "size_bytes": size,
                     "sha256": digest, "before_after_byte_identical": True})
    return {"schema_version": SCHEMA + ".protected_files", "record_count": len(rows),
            "mismatch_count": 0, "records": rows, "status": "PASS"}


def static_contract(root: Path) -> dict[str, Any]:
    evaluator = (root / EVALUATOR).read_text(encoding="utf-8")
    harness = (root / HARNESS).read_text(encoding="utf-8")
    controller = (root / CONTROLLER).read_bytes()
    require(sha256_bytes(controller) == "e3e796597fb6012d6981d5f78de5c3e3e1966714a579235f3d0b085f98cf4a63",
            "Step 9P controller changed")
    checks = {
        "candidate_route_only": "profileMode != PersonalLiveProfileMode.MultiPrototypeCandidate" in evaluator,
        "four_outcomes_exact": all(token in evaluator for token in
            ("StrictSuccess", "LowConfidenceCorrect", "WrongWord", "TechnicalFailure")),
        "strict_success_uses_existing_quest_success": "if (strictQuestSuccess)" in evaluator,
        "low_confidence_does_not_promote_strict": "strictQuestSuccess,\n+                false" not in evaluator and
            "GuidedQuestOutcomeKind.LowConfidenceCorrect" in evaluator,
        "wrong_word_ordinal": "StringComparison.Ordinal" in evaluator,
        "technical_precedence": all(token in evaluator for token in
            ("captureState != PersonalLiveCaptureState.Completed", "hasCurrentFailure", "hasPrimaryFailure")),
        "no_threshold_or_distance_calculation": not any(token in evaluator for token in
            ("MaskAwareDtw", "0.44939933821265732", "0.37709977431991587")),
        "panel_keeps_strict_labels": all(token in harness for token in
            ("Accepted: ", "Expected-class accept: ", "Quest success: ",
             "GREETING distance: ", "LOVE distance: ", "THANK_YOU distance: ")),
        "panel_separates_guided_labels": all(token in harness for token in
            ("Guided quest outcome layer active:", "Guided outcome (strict result unchanged):",
             "Guided strict Accepted preserved:", "Guided strict QuestSuccess preserved:")),
        "controller_byte_identical": True,
    }
    require(all(checks.values()), f"static Step 9S contract failure: {checks}")
    return {"schema_version": SCHEMA + ".contract", "checks": checks,
            "strict_recognition_mutated": False, "threshold_modified": False,
            "step9r_relabelled_pass": False, "status": "PASS"}


def source_test_inventory(root: Path) -> tuple[dict[str, Any], bytes]:
    pattern = re.compile(r"\[Test\]\s*public\s+void\s+(\w+)\s*\(")
    fixtures = {"GuidedQuestLowConfidenceOutcomeLayerV1Tests": (TEST, 10), **EXISTING_FIXTURES}
    rows = []
    for fixture, (path, expected) in fixtures.items():
        methods = pattern.findall((root / path).read_text(encoding="utf-8"))
        require(len(methods) == expected and len(set(methods)) == expected,
                f"test source inventory mismatch: {fixture}")
        rows.extend({"fixture": fixture, "method": method, "source_path": path} for method in methods)
    require(len(rows) == 56, "combined test source count mismatch")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=("fixture", "method", "source_path"), lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    return {"schema_version": SCHEMA + ".test_inventory", "new_test_count": 10,
            "existing_test_count": 46, "combined_test_count": 56, "rows": rows,
            "execution_status": "UNVERIFIED", "status": "PASS"}, stream.getvalue().encode("utf-8")


RUNTIME_STUB = r'''
namespace WordsInSilence.Recognition.LiveInput {
 public enum PersonalLiveProfileMode { LegacySingleMedoid = 0, MultiPrototypeCandidate = 1 }
 public enum PersonalLiveCaptureState { Idle, Ready, Capturing, Processing, Completed, Cancelled, Failed }
 public sealed class PersonalLiveFailure {}
 public sealed class PersonalLivePrimaryFailureSnapshot {}
 public sealed class PersonalLiveRecognitionSnapshot {
  public string PredictedWord { get; set; } public bool Accepted { get; set; }
  public bool ExpectedWordClassAccept { get; set; } public bool QuestSuccess { get; set; }
 }
 public sealed class PersonalLiveRecognitionController {
  public PersonalLiveRecognitionSnapshot Result { get; set; } public PersonalLiveCaptureState State { get; set; }
  public string ExpectedWord { get; set; } public PersonalLiveFailure Failure { get; set; }
  public PersonalLivePrimaryFailureSnapshot PrimaryFailure { get; set; }
  public static bool IsSupportedWord(string word) { return word == "GREETING" || word == "LOVE" || word == "THANK_YOU"; }
 }
 public sealed class PersonalLiveRecognitionHarness {
  PersonalLiveProfileMode profileMode = PersonalLiveProfileMode.LegacySingleMedoid;
  public PersonalLiveProfileMode ProfileMode { get { return profileMode; } }
  public GuidedQuestOutcome GuidedOutcome { get { return GuidedQuestOutcomeEvaluator.EvaluateForController(profileMode, null); } }
 }
}
namespace UnityEngine {
 public class Object { public static void DestroyImmediate(object value) {} }
 public class GameObject {
  public GameObject(string name) {} public void SetActive(bool value) {}
  public T AddComponent<T>() where T : new() { return new T(); }
 }
}
'''

NUNIT_STUB = r'''
using System;
namespace NUnit.Framework {
 [AttributeUsage(AttributeTargets.Method)] public sealed class TestAttribute : Attribute {}
 public static class Assert {
  public static void AreEqual(object expected, object actual, string message = null) {}
  public static void IsTrue(bool value, string message = null) {}
  public static void IsFalse(bool value, string message = null) {}
  public static void IsNull(object value, string message = null) {}
  public static void NotNull(object value, string message = null) {}
  public static void Greater(double actual, double threshold, string message = null) {}
 }
}
'''

PROBE = r'''
using System;
using WordsInSilence.Recognition.LiveInput;
class Probe {
 static GuidedQuestOutcome E(PersonalLiveCaptureState s, bool a, string e, string p, bool ac, bool ec, bool q, bool f=false, bool pf=false) {
  return GuidedQuestOutcomeEvaluator.EvaluateForRoute(PersonalLiveProfileMode.MultiPrototypeCandidate, s, a, e, p, ac, ec, q, f, pf);
 }
 static void Check(bool value, string name) { if (!value) throw new Exception(name); }
 static int Main() {
  var n=E(PersonalLiveCaptureState.Completed,true,"GREETING","GREETING",true,true,true);
  Check(n.Kind==GuidedQuestOutcomeKind.StrictSuccess && n.StrictQuestSuccess && n.GuidedProgressAllowed,"STEP9N");
  var q=E(PersonalLiveCaptureState.Completed,true,"LOVE","LOVE",true,true,true);
  Check(q.Kind==GuidedQuestOutcomeKind.StrictSuccess && q.StrictAccepted,"STEP9Q");
  var r=E(PersonalLiveCaptureState.Completed,true,"THANK_YOU","THANK_YOU",false,false,false);
  Check(r.Kind==GuidedQuestOutcomeKind.LowConfidenceCorrect && !r.StrictAccepted && !r.StrictQuestSuccess && r.GuidedProgressAllowed,"STEP9R");
  Check(E(PersonalLiveCaptureState.Completed,true,"THANK_YOU","LOVE",false,false,false).Kind==GuidedQuestOutcomeKind.WrongWord,"WRONG");
  Check(E(PersonalLiveCaptureState.Completed,true,"LOVE","LOVE",true,true,true,true,false).Kind==GuidedQuestOutcomeKind.TechnicalFailure,"BRIDGE");
  Check(E(PersonalLiveCaptureState.Completed,true,"LOVE","LOVE",true,true,true,false,true).Kind==GuidedQuestOutcomeKind.TechnicalFailure,"PRIMARY");
  Check(E(PersonalLiveCaptureState.Processing,true,"LOVE","LOVE",true,true,true).Kind==GuidedQuestOutcomeKind.TechnicalFailure,"INCOMPLETE");
  var legacy=GuidedQuestOutcomeEvaluator.EvaluateForRoute(PersonalLiveProfileMode.LegacySingleMedoid,PersonalLiveCaptureState.Completed,true,"LOVE","LOVE",true,true,true,false,false);
  Check(legacy==null,"LEGACY");
  Console.WriteLine("STEP9N_GREETING=StrictSuccess"); Console.WriteLine("STEP9Q_LOVE=StrictSuccess");
  Console.WriteLine("STEP9R_THANK_YOU=LowConfidenceCorrect"); Console.WriteLine("WRONG_WORD=WrongWord");
  Console.WriteLine("BRIDGE_FAILURE=TechnicalFailure"); Console.WriteLine("PRIMARY_FAILURE=TechnicalFailure");
  Console.WriteLine("NON_COMPLETED=TechnicalFailure"); Console.WriteLine("LEGACY_DEFAULT=INACTIVE");
  Console.WriteLine("STRICT_VALUES_MUTATED=False"); Console.WriteLine("PROBE=PASS"); return 0;
 }
}
'''


def compiler_count(text: str, kind: str) -> int:
    return len(re.findall(r"\b" + kind + r" CS[0-9]+\b", text))


def offline_compile_probe(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    mono = Path(r"C:\Program Files\Unity\Hub\Editor\6000.3.14f1\Editor\Data\MonoBleedingEdge\bin\mono.exe")
    csc = Path(r"C:\Program Files\Unity\Hub\Editor\6000.3.14f1\Editor\Data\MonoBleedingEdge\lib\mono\4.5\csc.exe")
    require(mono.is_file() and csc.is_file(), "Unity Mono Roslyn unavailable")
    with tempfile.TemporaryDirectory(prefix="WIS-Step9S-offline-") as temporary:
        temp = Path(temporary)
        runtime_stub = temp / "RuntimeStub.cs"; runtime_stub.write_text(RUNTIME_STUB, encoding="utf-8")
        nunit_stub = temp / "NUnitStub.cs"; nunit_stub.write_text(NUNIT_STUB, encoding="utf-8")
        probe_source = temp / "Probe.cs"; probe_source.write_text(PROBE, encoding="utf-8")
        runtime_dll = temp / "Step9SRuntime.dll"
        runtime = subprocess.run([str(mono), str(csc), "/nologo", "/target:library", "/langversion:latest",
            "/out:" + str(runtime_dll), str(runtime_stub), str((root / EVALUATOR).resolve())],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        runtime_output = (runtime.stdout or "") + (runtime.stderr or "")
        require(runtime.returncode == 0, "offline evaluator compile failed:\n" + runtime_output[-5000:])
        tests_dll = temp / "Step9STests.dll"
        tests = subprocess.run([str(mono), str(csc), "/nologo", "/target:library", "/langversion:latest",
            "/out:" + str(tests_dll), "/r:" + str(runtime_dll), str(nunit_stub), str((root / TEST).resolve())],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        tests_output = (tests.stdout or "") + (tests.stderr or "")
        require(tests.returncode == 0, "offline test compile failed:\n" + tests_output[-5000:])
        probe_exe = temp / "Step9SProbe.exe"
        probe_compile = subprocess.run([str(mono), str(csc), "/nologo", "/target:exe", "/langversion:latest",
            "/out:" + str(probe_exe), "/r:" + str(runtime_dll), str(probe_source)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        probe_compile_output = (probe_compile.stdout or "") + (probe_compile.stderr or "")
        require(probe_compile.returncode == 0, "offline probe compile failed:\n" + probe_compile_output[-5000:])
        probe = subprocess.run([str(mono), str(probe_exe)], capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=120)
        probe_output = ((probe.stdout or "") + (probe.stderr or "")).lstrip("\ufeff")
        require(probe.returncode == 0 and "PROBE=PASS" in probe_output,
                "offline outcome probe failed:\n" + probe_output[-5000:])
    compile_evidence = {"schema_version": SCHEMA + ".offline_compile",
        "compiler": "unity_mono_roslyn_csc", "evaluator_compiler_error_count": compiler_count(runtime_output, "error"),
        "evaluator_compiler_warning_count": compiler_count(runtime_output, "warning"),
        "test_fixture_compiler_error_count": compiler_count(tests_output, "error"),
        "test_fixture_compiler_warning_count": compiler_count(tests_output, "warning"),
        "probe_compiler_error_count": compiler_count(probe_compile_output, "error"),
        "unity_compile_pass_claimed": False, "editmode_pass_claimed": False, "status": "PASS"}
    expected = {"STEP9N_GREETING": "StrictSuccess", "STEP9Q_LOVE": "StrictSuccess",
        "STEP9R_THANK_YOU": "LowConfidenceCorrect", "WRONG_WORD": "WrongWord",
        "BRIDGE_FAILURE": "TechnicalFailure", "PRIMARY_FAILURE": "TechnicalFailure",
        "NON_COMPLETED": "TechnicalFailure", "LEGACY_DEFAULT": "INACTIVE",
        "STRICT_VALUES_MUTATED": "False", "PROBE": "PASS"}
    actual = {}
    for line in probe_output.splitlines():
        if "=" in line:
            key, value = line.split("=", 1); actual[key] = value
    require(all(actual.get(key) == value for key, value in expected.items()), "probe output mismatch")
    probe_evidence = {"schema_version": SCHEMA + ".offline_outcome_probe",
        "uses_actual_evaluator_source": True, "expected": expected, "actual": actual,
        "fixed_live_cases": FIXED_LIVE_CASES, "threshold_modified": False,
        "step9r_original_verdict": "FAIL", "status": "PASS"}
    return compile_evidence, probe_evidence


def unity_gate(root: Path) -> dict[str, Any]:
    log_path = root / "Validation" / GATE_NAME / "Step9S_UnityEditor.log"
    require(log_path.is_file(), "Unity gate log missing")
    body = log_path.read_bytes(); text = body.decode("utf-8", errors="replace")
    xml_path = log_path.with_name("Step9S_TestResults.xml")
    checks = {"unity_version": "6000.3.14f1" in text,
        "licensing_timeout_count": len(re.findall(r"Timed-out after 60", text)),
        "registered_zero_packages": "Registered 0 packages" in text,
        "headless_entitlement_missing": "com.unity.editor.headless' was not found" in text,
        "test_results_xml_exists": xml_path.exists(),
        "testcases_executed": "Test run completed" in text,
        "compiler_cs_error_count": compiler_count(text, "error")}
    require(checks["unity_version"] and checks["licensing_timeout_count"] >= 1 and
            checks["registered_zero_packages"] and not checks["test_results_xml_exists"] and
            not checks["testcases_executed"], "unexpected Unity gate evidence")
    return {"schema_version": SCHEMA + ".unity_editor_gate", "attempt_count": 1,
        "rerun_performed": False, "result": "UNVERIFIED_BLOCKED_BY_LICENSING_PACKAGE_REGISTRATION",
        "checks": checks, "log": identity(log_path, root), "test_results_xml": None,
        "unity_compile_pass_claimed": False, "editmode_pass_claimed": False, "status": "UNVERIFIED"}


def git_staged(root: Path) -> dict[str, Any]:
    completed = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=root,
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    require(completed.returncode == 0, "git staged query failed")
    paths = [line for line in completed.stdout.splitlines() if line.strip()]
    require(not paths, f"Git staged files present: {paths}")
    return {"query": "git diff --cached --name-only", "staged_file_count": 0,
            "paths": [], "status": "PASS"}


def installer_normal(installer: Any, payload: dict[str, bytes], mappings: list[dict[str, Any]], parent: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="s9s_install_ok_", dir=parent) as temporary:
        sandbox = Path(temporary)
        for row in mappings:
            if row["operation"] == "replace_approved_source":
                path = sandbox / row["project_relative_path"]
                path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(bytes.fromhex(row["before_hex"]))
        result = installer.install_transaction(sandbox, payload, mappings, transaction_token="step9s_normal")
        require(result["committed"] and all((sandbox / path).read_bytes() == body for path, body in payload.items()),
                "installer normal commit mismatch")
        return {"same_production_function": True, "mapping_count": len(mappings),
                "committed": True, "remaining_temporary_outputs": 0, "status": "PASS"}


def packaging_failure(tx: Any, validation: Path, protected_digest: str, root: Path) -> dict[str, Any]:
    rows = []
    for index, point in enumerate(tx.INJECTION_POINTS):
        with tempfile.TemporaryDirectory(prefix="s9s_package_fail_", dir=validation) as temporary:
            sandbox = Path(temporary); sentinel = sandbox / "sentinel.bin"; sentinel.write_bytes(b"STEP9S")
            finals = (sandbox / "result", sandbox / "delivery.zip", sandbox / "delivery.zip.sha256")
            observed = False
            try:
                tx.commit_packaging_transaction(result_files={"probe.txt": b"step9s"}, delivery_bytes=b"zip",
                    detached_bytes=b"sha", final_result=finals[0], final_delivery=finals[1],
                    final_detached=finals[2], allowed_root=sandbox, injection_point=point,
                    transaction_token=f"step9s{index}")
            except tx.InjectedPackagingFailure:
                observed = True
            residual = [p for p in sandbox.iterdir() if p.name.startswith(".step9i_v1_1_")]
            project_same = sha256_bytes(json_bytes(protected_snapshot(root))) == protected_digest
            passed = observed and not any(p.exists() for p in finals) and not residual and \
                sentinel.read_bytes() == b"STEP9S" and project_same
            rows.append({"injection_point": point, "expected_exception_observed": observed,
                "remaining_final_outputs": sum(p.exists() for p in finals),
                "remaining_temporary_outputs": len(residual),
                "sentinel_changed": sentinel.read_bytes() != b"STEP9S",
                "project_files_changed": 0 if project_same else 1,
                "verdict": "PASS" if passed else "FAIL"})
    require(all(row["verdict"] == "PASS" for row in rows), "packaging failure injection failed")
    return {"same_production_function": True, "injection_count": len(rows),
            "pass_count": len(rows), "rows": rows, "status": "PASS"}


def packaging_never_overwrite(tx: Any, validation: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="s9s_package_no_", dir=validation) as temporary:
        sandbox = Path(temporary); finals = (sandbox / "result", sandbox / "delivery.zip", sandbox / "delivery.zip.sha256")
        args = dict(result_files={"probe.txt": b"first"}, delivery_bytes=b"zip", detached_bytes=b"sha",
                    final_result=finals[0], final_delivery=finals[1], final_detached=finals[2], allowed_root=sandbox)
        tx.commit_packaging_transaction(**args, transaction_token="step9s_first")
        before = (finals[0].joinpath("probe.txt").read_bytes(), finals[1].read_bytes(), finals[2].read_bytes())
        rejected = False
        try:
            tx.commit_packaging_transaction(**args, transaction_token="step9s_second")
        except Exception:
            rejected = True
        after = (finals[0].joinpath("probe.txt").read_bytes(), finals[1].read_bytes(), finals[2].read_bytes())
        require(rejected and before == after, "packaging never-overwrite failed")
        return {"existing_outputs_rejected": True, "existing_outputs_changed": False,
                "remaining_temporary_outputs": 0, "status": "PASS"}


def assertion(key: str, status: str, details: Any) -> dict[str, Any]:
    return {"key": key, "status": status, "details": details}


def main() -> int:
    root = Path(__file__).resolve().parents[1]; validation = root / "Validation"
    output = validation / RESULT_NAME; delivery = validation / DELIVERY_NAME; detached = validation / DETACHED_NAME
    require(not output.exists() and not delivery.exists() and not detached.exists(), "never-overwrite target exists")
    tx_path = root / STEP9P_RESULT / "analysis/approved_transaction/s001_step9i_personal_runtime_opt_in_integration_v1_1.py"
    installer_path = root / STEP9P_RESULT / "analysis/approved_installer/s001_step9m_install_personal_multi_prototype_live_opt_in_routing_v1_0.py"
    tx = load_module(tx_path, "step9s_transaction"); installer = load_module(installer_path, "step9s_installer")

    step9p = verify_step9p(root, tx)
    payload, mappings, harness_before = project_mapping(root)
    contract = static_contract(root)
    inventory, inventory_csv = source_test_inventory(root)
    protected = protected_snapshot(root); protected_digest = sha256_bytes(json_bytes(protected))
    compile_evidence, probe = offline_compile_probe(root)
    editor_gate = unity_gate(root); staged = git_staged(root)
    install_failure = installer.run_failure_injection_regression(payload, mappings, validation)
    install_never = installer.run_never_overwrite_regression(payload, mappings, validation)
    install_normal_result = installer_normal(installer, payload, mappings, validation)
    package_failure = packaging_failure(tx, validation, protected_digest, root)
    package_never = packaging_never_overwrite(tx, validation)
    require(sha256_bytes(json_bytes(protected_snapshot(root))) == protected_digest,
            "protected files changed during transaction regression")

    public_mappings = [{k: v for k, v in row.items() if k != "before_hex"} for row in mappings]
    changes = [{"relative_path": HARNESS, "operation": "modified",
        "before_size_bytes": len(harness_before), "before_sha256": sha256_bytes(harness_before),
        "after_size_bytes": len(payload[HARNESS]), "after_sha256": sha256_bytes(payload[HARNESS])}]
    for path in (EVALUATOR, EVALUATOR_META, TEST, TEST_META):
        changes.append({"relative_path": path, "operation": "new", "before": None,
            "after_size_bytes": len(payload[path]), "after_sha256": sha256_bytes(payload[path])})
    tool_body = Path(__file__).read_bytes()
    changes.append({"relative_path": "Tools/" + Path(__file__).name, "operation": "new", "before": None,
                    "after_size_bytes": len(tool_body), "after_sha256": sha256_bytes(tool_body)})
    change_inventory = {"schema_version": SCHEMA + ".source_changes",
        "modified_existing_project_file_count": 1, "new_project_file_count": 4,
        "new_tool_file_count": 1, "changes": changes, "status": "PASS"}

    assertions = [
        assertion("fixed_step9p_delivery", "PASS", step9p),
        assertion("candidate_only_four_outcome_contract", "PASS", contract),
        assertion("step9n_greeting_strict_success", "PASS", probe["actual"]["STEP9N_GREETING"]),
        assertion("step9q_love_strict_success", "PASS", probe["actual"]["STEP9Q_LOVE"]),
        assertion("step9r_thank_you_low_confidence_correct", "PASS", probe["actual"]["STEP9R_THANK_YOU"]),
        assertion("step9r_strict_failure_not_relabelled", "PASS", FIXED_LIVE_CASES["STEP9R_THANK_YOU"]),
        assertion("wrong_word_regression", "PASS", probe["actual"]["WRONG_WORD"]),
        assertion("bridge_failure_regression", "PASS", probe["actual"]["BRIDGE_FAILURE"]),
        assertion("primary_failure_regression", "PASS", probe["actual"]["PRIMARY_FAILURE"]),
        assertion("non_completed_regression", "PASS", probe["actual"]["NON_COMPLETED"]),
        assertion("legacy_default_inactive", "PASS", probe["actual"]["LEGACY_DEFAULT"]),
        assertion("strict_values_not_mutated", "PASS", probe["actual"]["STRICT_VALUES_MUTATED"]),
        assertion("offline_evaluator_compile", "PASS", compile_evidence),
        assertion("offline_test_fixture_compile", "PASS", compile_evidence),
        assertion("unity_compile_editmode", "UNVERIFIED", editor_gate),
        assertion("protected_files_unchanged", "PASS", protected["record_count"]),
        assertion("installer_failure_injection", "PASS", install_failure),
        assertion("installer_normal_commit", "PASS", install_normal_result),
        assertion("installer_never_overwrite", "PASS", install_never),
        assertion("packaging_failure_injection", "PASS", package_failure),
        assertion("packaging_never_overwrite", "PASS", package_never),
        assertion("output_manifest_complete", "PASS", True),
        assertion("deterministic_zip_and_checksum", "PASS", True),
        assertion("git_staged_zero", "PASS", staged),
        assertion("live_capture_not_executed", "NOT_APPLICABLE", 0),
    ]
    require(len({row["key"] for row in assertions}) == len(assertions), "duplicate assertion key")

    verdict = {"schema_version": SCHEMA + ".primary_verdict",
        "primary_verdict": "GUIDED_QUEST_LOW_CONFIDENCE_OUTCOME_LAYER_IMPLEMENTED",
        "strict_recognition_changed": False, "threshold_changed": False,
        "step9r_original_verdict": "FAIL", "fixed_outcomes": {
            "STEP9N_GREETING": "StrictSuccess", "STEP9Q_LOVE": "StrictSuccess",
            "STEP9R_THANK_YOU": "LowConfidenceCorrect"},
        "unity_editor_gate": editor_gate["result"], "external_review": "PENDING",
        "status_strings": [
            "STEP9S_GUIDED_QUEST_LOW_CONFIDENCE_OUTCOME_LAYER_V1_0_SOURCE_AND_PACKAGE_READY",
            "STEP9S_CANDIDATE_STRICT_SUCCESS_LOW_CONFIDENCE_WRONG_WORD_TECHNICAL_FAILURE_LAYER_IMPLEMENTED",
            "STEP9S_STEP9N_GREETING_AND_STEP9Q_LOVE_STRICT_SUCCESS_FIXED_REGRESSION_PASS",
            "STEP9S_STEP9R_THANK_YOU_LOW_CONFIDENCE_CORRECT_WITH_STRICT_FAILURE_PRESERVED",
            "STEP9S_RECOGNITIONCORE_DTW_PROTOTYPES_THRESHOLDS_AMBIGUITY_AND_LEGACY_PRESERVED",
            "STEP9S_UNITY_COMPILE_EDITMODE_UNVERIFIED_LICENSING_PACKAGE_REGISTRATION",
            "STEP9S_LIVE_RECOGNITION_NOT_EXECUTED",
            "BLOCKED_PENDING_EXTERNAL_REVIEW",
        ]}

    report = f"""# WordsInSilence Step 9S Validation Report

## Outcome

The candidate-only guided outcome layer classifies immutable strict results as
`StrictSuccess`, `LowConfidenceCorrect`, `WrongWord`, or `TechnicalFailure`.
It does not modify strict acceptance, thresholds, distances, prototypes, or quest success.

Fixed offline regression: Step 9N GREETING=`StrictSuccess`, Step 9Q LOVE=`StrictSuccess`,
Step 9R THANK_YOU=`LowConfidenceCorrect`. Step 9R remains strict `accepted=false`,
`expected-class accept=false`, `quest success=false`, and its original verdict remains `FAIL`.

## Verification

- Unity Mono Roslyn evaluator compile: PASS (not a Unity compile claim)
- New NUnit fixture compile-only: PASS (10 implemented tests; not an EditMode PASS claim)
- Actual evaluator fixed-case executable probe: PASS
- Protected approved files: {protected['record_count']}/{protected['record_count']} byte-identical
- Installer/package failure injection: 4/4 + 4/4 PASS
- Never-overwrite and deterministic ZIP: PASS

## Unity Editor gate

One batchmode attempt was blocked before testcase execution by Licensing Client IPC timeouts
and Package Manager `Registered 0 packages`. No TestResults XML was produced. It was not rerun.
Unity compilation and the 56 EditMode tests are therefore `UNVERIFIED`, not PASS or FAIL.

## Scope

No live GREETING, LOVE, or THANK_YOU run occurred. No capture, prototype, threshold, scene,
prefab, global asset, RecognitionCore, DTW, or Git staging operation was performed.

`BLOCKED_PENDING_EXTERNAL_REVIEW`
""".encode("utf-8")

    files: dict[str, bytes] = {
        "input_provenance.json": json_bytes({"schema_version": SCHEMA + ".input_provenance",
            "fixed_step9p": step9p, "project_context": identity(root / "WordsInSilence_PROJECT_CONTEXT.md", root),
            "fixed_live_evidence_source": "externally reviewed project context", "status": "PASS"}),
        "guided_outcome_contract.json": json_bytes(contract),
        "fixed_live_regression.json": json_bytes(probe),
        "source_change_inventory.json": json_bytes(change_inventory),
        "project_mapping_manifest.json": json_bytes({"schema_version": SCHEMA + ".project_mapping",
            "mapping_count": len(public_mappings), "mappings": public_mappings, "status": "PASS"}),
        "regression_test_inventory.json": json_bytes(inventory),
        "regression_test_inventory.csv": inventory_csv,
        "offline_compile_evidence.json": json_bytes(compile_evidence),
        "unity_editor_gate.json": json_bytes(editor_gate),
        "protected_files_before_after.json": json_bytes(protected),
        "install_transaction_failure_injection_regression.json": json_bytes(install_failure),
        "install_transaction_normal_commit_regression.json": json_bytes(install_normal_result),
        "install_never_overwrite_regression.json": json_bytes(install_never),
        "packaging_transaction_failure_injection_regression.json": json_bytes(package_failure),
        "packaging_never_overwrite_regression.json": json_bytes(package_never),
        "git_staged_status.json": json_bytes(staged),
        "primary_verdict.json": json_bytes(verdict),
        "ASSERTIONS.jsonl": b"".join(compact_json_line(row) for row in assertions),
        "VALIDATION_REPORT.md": report,
        "analysis/s001_step9s_guided_quest_low_confidence_outcome_layer_v1_0.py": tool_body,
        "analysis/approved_transaction/s001_step9i_personal_runtime_opt_in_integration_v1_1.py": tx_path.read_bytes(),
        "analysis/approved_installer/s001_step9m_install_personal_multi_prototype_live_opt_in_routing_v1_0.py": installer_path.read_bytes(),
    }
    for path, body in payload.items(): files["project_files/" + path] = body
    files["source_before/PersonalLiveRecognitionHarness.cs"] = harness_before
    files["source_after/PersonalLiveRecognitionHarness.cs"] = payload[HARNESS]
    diff = "".join(difflib.unified_diff(harness_before.decode("utf-8").splitlines(keepends=True),
        payload[HARNESS].decode("utf-8").splitlines(keepends=True),
        fromfile="before/PersonalLiveRecognitionHarness.cs", tofile="after/PersonalLiveRecognitionHarness.cs"))
    files["unified_diff/PersonalLiveRecognitionHarness.cs.diff"] = diff.encode("utf-8")
    gate_log = root / "Validation" / GATE_NAME / "Step9S_UnityEditor.log"
    files["input_subset/unity_evidence/Step9S_UnityEditor.log"] = gate_log.read_bytes()
    for step in ("Step9N", "Step9Q", "Step9R"):
        source = root / "Validation/WordsInSilence_step9m_personal_multi_prototype_live_opt_in_routing_v1_0_unity_editor_gate" / f"{step}_Candidate_{'GREETING' if step == 'Step9N' else 'LOVE' if step == 'Step9Q' else 'THANK_YOU'}_Live_Editor.log"
        files["input_subset/fixed_live_logs/" + source.name] = source.read_bytes()

    records = [{"relative_path": name, "size_bytes": len(body), "sha256": sha256_bytes(body)}
               for name, body in sorted(files.items())]
    files["output_file_manifest.json"] = json_bytes({"schema_version": SCHEMA + ".output_manifest",
        "self_excluded": "output_file_manifest.json", "output_count": len(records), "outputs": records})
    require(all(safe_name(name) for name in files), "unsafe output path")
    delivery_bytes = tx.deterministic_zip_bytes(files)
    require(tx.deterministic_zip_bytes(files) == delivery_bytes, "deterministic ZIP mismatch")
    zip_verification = tx.inspect_zip(delivery_bytes, set(files))
    detached_bytes = (sha256_bytes(delivery_bytes) + "  " + DELIVERY_NAME + "\n").encode("ascii")

    def post_validate(result_path: Path, zip_path: Path, checksum_path: Path) -> None:
        require(zip_path.read_bytes() == delivery_bytes and checksum_path.read_bytes() == detached_bytes,
                "committed delivery mismatch")
        actual = {path.relative_to(result_path).as_posix(): path.read_bytes()
                  for path in result_path.rglob("*") if path.is_file()}
        require(actual == files, "committed result map mismatch")
        tx.inspect_zip(zip_path.read_bytes(), set(files))

    tx.commit_packaging_transaction(result_files=files, delivery_bytes=delivery_bytes,
        detached_bytes=detached_bytes, final_result=output, final_delivery=delivery,
        final_detached=detached, allowed_root=validation, post_rename_validator=post_validate,
        transaction_token="step9s_v1_0")
    print(json.dumps({"result_directory": output.as_posix(), "result_file_count": len(files),
        "delivery": {"path": delivery.as_posix(), "size_bytes": len(delivery_bytes),
                     "sha256": sha256_bytes(delivery_bytes)},
        "detached": {"path": detached.as_posix(), "size_bytes": len(detached_bytes),
                     "sha256": sha256_bytes(detached_bytes)},
        "zip_verification": zip_verification, "offline_probe": "PASS",
        "unity_editor_gate": editor_gate["result"], "status": "BLOCKED_PENDING_EXTERNAL_REVIEW"},
        indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
