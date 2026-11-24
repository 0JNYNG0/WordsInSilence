#!/usr/bin/env python3
"""Step 9J v1.1 test-only harness component lifecycle coverage repair.

The production controller, production harness, approved runtime/assets, and
Step 9J v1.0 outputs are fixed inputs. This tool validates that only the
EditMode test source changed, compile-checks the amended fixture, runs an
offline probe that invokes the actual harness OnDisable/OnDestroy methods by
reflection, and packages the evidence with the approved v1.1 transaction.
It never launches Unity or a webcam and never claims EditMode/PlayMode PASS.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tempfile
from typing import Any
import zipfile


SCHEMA = "words_in_silence.step9j.personal_live_recognition_integration.v1_1"
V10_DELIVERY_REL = Path("Validation/WordsInSilence_step9j_personal_live_recognition_integration_v1_0_delivery.zip")
V10_DELIVERY_SIZE = 44_445
V10_DELIVERY_SHA256 = "1775ac8e310fbd99bcd66b0b8055bc12a63554374df4f70c6cbcf73686985159"
V10_DETACHED_SIZE = 144
V10_DETACHED_SHA256 = "a7b9bd72d502d78ee3ef31b938cc4cdc8c4e7c99d294464198f4733eb7a68b84"
V10_TOOL_REL = Path("Tools/s001_step9j_personal_live_recognition_integration_v1_0.py")
V10_TOOL_SIZE = 47_009
V10_TOOL_SHA256 = "72c716128275f638a179de195daa90668097c83852249349e068ce45d12a6d69"
V11_TRANSACTION_REL = Path("Tools/s001_step9i_personal_runtime_opt_in_integration_v1_1.py")
V11_TRANSACTION_SIZE = 37_722
V11_TRANSACTION_SHA256 = "30471dc6725bdbdd0cc80c345b238485d1cb0d8d7a5933aac7477d536d714caf"

RESULT_REL = Path("Validation/WordsInSilence_step9j_personal_live_recognition_integration_v1_1")
DELIVERY_REL = Path("Validation/WordsInSilence_step9j_personal_live_recognition_integration_v1_1_delivery.zip")
TEST_REL = Path("Assets/_Project/Tests/EditMode/PersonalLiveRecognitionIntegrationV1Tests.cs")
TEST_META_REL = Path(str(TEST_REL) + ".meta")
CONTROLLER_REL = Path("Assets/_Project/Scripts/Recognition/LiveInput/PersonalLiveRecognitionController.cs")
CONTROLLER_META_REL = Path(str(CONTROLLER_REL) + ".meta")
HARNESS_REL = Path("Assets/_Project/Scripts/Recognition/LiveInput/PersonalLiveRecognitionHarness.cs")
HARNESS_META_REL = Path(str(HARNESS_REL) + ".meta")

V10_TEST_SIZE = 26_823
V10_TEST_SHA256 = "f8e095d1f4600346947d0bb168c8b3b1b2b47f715d221563066ca5c512d5a07c"
FIXED_PROJECT_FILES = (
    (CONTROLLER_REL, 30_013, "69fc64354c0f1466b9b161cddaacb13087a185c77cc98304bbdb86daaf006ba7"),
    (CONTROLLER_META_REL, 243, "5262ea0d5605c95fc570033432bb000b7cb2c22fe51b376c4039227cfdc9280d"),
    (HARNESS_REL, 7_209, "b5cee1181f5b0927dc431486d23ccea4ee4778466de729686e0a451342a8b81f"),
    (HARNESS_META_REL, 243, "f54efe92219b323e99c3a1a9466ae6ac5a2cc28569c171a1534683dda98d722e"),
    (TEST_META_REL, 243, "48271faad1512b429086a906ed9d454f5dfaed07f5b1a6d5d2f4c1267e01769c"),
)

PACKAGED_CONTROLLER = "project_files/production/PersonalLiveRecognitionController.cs"
PACKAGED_CONTROLLER_META = PACKAGED_CONTROLLER + ".meta"
PACKAGED_HARNESS = "project_files/production/PersonalLiveRecognitionHarness.cs"
PACKAGED_HARNESS_META = PACKAGED_HARNESS + ".meta"
PACKAGED_TEST = "project_files/tests/PersonalLiveRecognitionIntegrationV1Tests.cs"
PACKAGED_TEST_META = PACKAGED_TEST + ".meta"
NEW_TESTS = (
    "HarnessOnDisable_RemovesAllObservationSourceSubscribers",
    "HarnessOnDestroy_RemovesAllObservationSourceSubscribers",
)

UNITY_COMPONENT_STUB_EXTENSION = r"""
namespace UnityEngine {
  public class Object {
    public static void DestroyImmediate(Object value) {}
  }
  public sealed class GameObject : Object {
    public GameObject(string name) {}
    public void SetActive(bool value) {}
    public T AddComponent<T>() where T : new() { return new T(); }
  }
}
"""

HARNESS_LIFECYCLE_PROBE = r"""using System;
using System.Collections.Generic;
using System.Reflection;
using UnityEngine;
using WordsInSilence.Recognition.Core;
using WordsInSilence.Recognition.LiveInput;
using WordsInSilence.Recognition.Runtime;
class Step9JHarnessLifecycleProbe {
  sealed class Source : IPersonalLiveObservationSource {
    Action<RawMediaPipeObservation> observations; Action<Exception> failures;
    public bool IsBound => true;
    public int Subscribers => (observations == null ? 0 : observations.GetInvocationList().Length) +
      (failures == null ? 0 : failures.GetInvocationList().Length);
    public event Action<RawMediaPipeObservation> ObservationArrived { add { observations += value; } remove { observations -= value; } }
    public event Action<Exception> SourceFailed { add { failures += value; } remove { failures -= value; } }
  }
  sealed class Global : IPersonalLiveGlobalRuntime {}
  sealed class Runtime : IPersonalLiveRecognitionRuntime {
    public bool IsPersonalActive => true;
    public string ActiveProfileId => PersonalRuntimeProfileLoader.ProfileId;
    public PersonalLiveRoutingResult Recognize(string id, RecognitionSequence sequence, string expected) { throw new NotSupportedException(); }
  }
  sealed class Factory : IPersonalLiveRuntimeFactory {
    public IPersonalLiveGlobalRuntime LoadGlobal(PersonalLiveRuntimePaths paths) { return new Global(); }
    public IPersonalLiveRecognitionRuntime CreateAndActivate(IPersonalLiveGlobalRuntime global, PersonalLiveRuntimePaths paths) { return new Runtime(); }
  }
  static void Need(bool value,string message){if(!value)throw new Exception(message);}
  static PersonalLiveRecognitionController Controller(Source source) {
    var result=new PersonalLiveRecognitionController(source,new Factory());
    Need(result.Initialize(new PersonalLiveRuntimePaths("global","threshold","personal")),"controller initialization");
    return result;
  }
  static FieldInfo ControllerField() {
    var field=typeof(PersonalLiveRecognitionHarness).GetField("controller",BindingFlags.Instance|BindingFlags.NonPublic);
    Need(field!=null,"harness controller field");return field;
  }
  static void Invoke(PersonalLiveRecognitionHarness harness,string name) {
    var method=typeof(PersonalLiveRecognitionHarness).GetMethod(name,BindingFlags.Instance|BindingFlags.NonPublic);
    Need(method!=null,"harness lifecycle method "+name);method.Invoke(harness,null);
  }
  static int RunDisable() {
    var owner=new GameObject("Step9J-Probe-OnDisable");owner.SetActive(false);
    try {
      var harness=owner.AddComponent<PersonalLiveRecognitionHarness>();var source=new Source();var controller=Controller(source);
      ControllerField().SetValue(harness,controller);int before=source.Subscribers;Invoke(harness,"OnDisable");int after=source.Subscribers;
      Need(before==2&&after==0,"OnDisable subscribers");Need(controller.State==PersonalLiveCaptureState.Cancelled,"OnDisable terminal state");
      Console.WriteLine("HARNESS_ON_DISABLE_SUBSCRIBERS_BEFORE="+before);Console.WriteLine("HARNESS_ON_DISABLE_SUBSCRIBERS_AFTER="+after);return 0;
    } finally { UnityEngine.Object.DestroyImmediate(owner); }
  }
  static int RunDestroy() {
    var owner=new GameObject("Step9J-Probe-OnDestroy");owner.SetActive(false);
    try {
      var harness=owner.AddComponent<PersonalLiveRecognitionHarness>();var source=new Source();var controller=Controller(source);
      ControllerField().SetValue(harness,controller);int before=source.Subscribers;Invoke(harness,"OnDestroy");int after=source.Subscribers;
      Need(before==2&&after==0,"OnDestroy subscribers");Need(ControllerField().GetValue(harness)==null,"OnDestroy controller reference");
      Console.WriteLine("HARNESS_ON_DESTROY_SUBSCRIBERS_BEFORE="+before);Console.WriteLine("HARNESS_ON_DESTROY_SUBSCRIBERS_AFTER="+after);return 0;
    } finally { UnityEngine.Object.DestroyImmediate(owner); }
  }
  static int Main() { try { RunDisable();RunDestroy();Console.WriteLine("HARNESS_COMPONENT_LIFECYCLE=PASS");return 0; }
    catch(Exception error){Console.Error.WriteLine(error);return 1;} }
}
"""


class Step9JV11Error(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise Step9JV11Error(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def import_fixed_module(root: Path, relative: Path, size: int, digest: str, name: str):
    path = root / relative
    require(path.is_file() and path.stat().st_size == size and sha256_file(path) == digest,
            "fixed module identity mismatch: " + relative.as_posix())
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "cannot import fixed module: " + relative.as_posix())
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_v10_delivery(root: Path, transaction) -> tuple[dict[str, Any], dict[str, bytes]]:
    delivery = root / V10_DELIVERY_REL
    detached = Path(str(delivery) + ".sha256")
    require(delivery.is_file() and detached.is_file(), "Step 9J v1.0 Delivery input missing")
    delivery_bytes = delivery.read_bytes()
    detached_bytes = detached.read_bytes()
    require(len(delivery_bytes) == V10_DELIVERY_SIZE and sha256_bytes(delivery_bytes) == V10_DELIVERY_SHA256,
            "Step 9J v1.0 Delivery identity mismatch")
    require(len(detached_bytes) == V10_DETACHED_SIZE and sha256_bytes(detached_bytes) == V10_DETACHED_SHA256,
            "Step 9J v1.0 detached identity mismatch")
    transaction.verify_strict_detached(detached_bytes, delivery_bytes, delivery.name)
    with zipfile.ZipFile(io.BytesIO(delivery_bytes), "r") as archive:
        infos = archive.infolist()
        names = [item.filename for item in infos]
        require(len(names) == len(set(names)), "v1.0 duplicate ZIP entry")
        require(archive.testzip() is None, "v1.0 ZIP CRC failure")
        for item in infos:
            require(transaction.safe_zip_name(item.filename), "v1.0 unsafe ZIP path")
            require(not item.flag_bits & 1, "v1.0 encrypted ZIP entry")
            require(not stat.S_ISLNK((item.external_attr >> 16) & 0xFFFF), "v1.0 symlink ZIP entry")
        file_map = {item.filename: archive.read(item) for item in infos}
    manifest = json.loads(file_map["output_file_manifest.json"])
    records = manifest["files"]
    require({item["path"] for item in records} == set(file_map) - {"output_file_manifest.json"},
            "v1.0 manifest membership mismatch")
    for item in records:
        payload = file_map[item["path"]]
        require(len(payload) == item["size_bytes"] and sha256_bytes(payload) == item["sha256"],
                "v1.0 manifest identity mismatch: " + item["path"])
    require(transaction.deterministic_zip_bytes(file_map) == delivery_bytes, "v1.0 deterministic ZIP mismatch")
    result = root / "Validation/WordsInSilence_step9j_personal_live_recognition_integration_v1_0"
    require(result.is_dir() and all((result / PurePosixPath(name)).read_bytes() == data for name, data in file_map.items()),
            "v1.0 result directory differs from fixed Delivery")
    require(len(file_map[PACKAGED_TEST]) == V10_TEST_SIZE and sha256_bytes(file_map[PACKAGED_TEST]) == V10_TEST_SHA256,
            "v1.0 packaged test source identity mismatch")
    return ({"path": V10_DELIVERY_REL.as_posix(), "size_bytes": len(delivery_bytes),
             "sha256": sha256_bytes(delivery_bytes), "detached_size_bytes": len(detached_bytes),
             "detached_file_sha256": sha256_bytes(detached_bytes), "entry_count": len(file_map),
             "manifest_target_count": len(records), "archive_safety": "PASS",
             "deterministic_reproduction_byte_identical": True, "result_directory_byte_identical": True,
             "status": "PASS"}, file_map)


def verify_production_preservation(root: Path, v10_files: dict[str, bytes]) -> tuple[dict[str, Any], dict[str, str]]:
    packaged = {
        CONTROLLER_REL: PACKAGED_CONTROLLER, CONTROLLER_META_REL: PACKAGED_CONTROLLER_META,
        HARNESS_REL: PACKAGED_HARNESS, HARNESS_META_REL: PACKAGED_HARNESS_META,
        TEST_META_REL: PACKAGED_TEST_META,
    }
    records = []
    snapshot = {}
    for relative, size, digest in FIXED_PROJECT_FILES:
        path = root / relative
        require(path.is_file() and path.stat().st_size == size and sha256_file(path) == digest,
                "fixed v1.0 project file identity mismatch: " + relative.as_posix())
        require(path.read_bytes() == v10_files[packaged[relative]], "fixed project file differs from v1.0 package")
        snapshot[relative.as_posix()] = digest
        records.append({"path": relative.as_posix(), "size_bytes": size, "sha256": digest,
                        "v1_0_package_byte_identical": True})
    return ({"schema_version": SCHEMA + ".v1_0_production_preservation", "fixed_file_count": len(records),
             "mismatch_count": 0, "records": records, "status": "PASS"}, snapshot)


def method_blocks(source: str) -> dict[str, str]:
    result = {}
    pattern = re.compile(r"\[Test\]\s*public void\s+([A-Za-z0-9_]+)\s*\(\s*\)\s*\{")
    for match in pattern.finditer(source):
        depth = 1
        index = match.end()
        while index < len(source) and depth:
            if source[index] == "{":
                depth += 1
            elif source[index] == "}":
                depth -= 1
            index += 1
        require(depth == 0, "unbalanced test method: " + match.group(1))
        result[match.group(1)] = source[match.start():index]
    return result


def validate_test_change(before: bytes, after: bytes) -> dict[str, Any]:
    require(len(before) == V10_TEST_SIZE and sha256_bytes(before) == V10_TEST_SHA256,
            "fixed before test source mismatch")
    before_text = before.decode("utf-8")
    after_text = after.decode("utf-8")
    before_methods = method_blocks(before_text)
    after_methods = method_blocks(after_text)
    require(len(before_methods) == 15, "v1.0 test method count mismatch")
    require(len(after_methods) == 17, "v1.1 test method count must be 17")
    changed_existing = [name for name, body in before_methods.items() if after_methods.get(name) != body]
    require(not changed_existing, "existing v1.0 test method changed: " + str(changed_existing))
    require(set(after_methods) == set(before_methods) | set(NEW_TESTS), "v1.1 test method set mismatch")
    checks = {}
    for name, callback in zip(NEW_TESTS, ("OnDisable", "OnDestroy")):
        body = after_methods[name]
        needed = ("PersonalLiveRecognitionHarness", "new GameObject", "SetActive(false)",
                  "AddComponent<PersonalLiveRecognitionHarness>", "FakeObservationSource",
                  "SetPrivateHarnessField", "Assert.AreEqual(2, source.SubscriberCount)",
                  f'InvokePrivateHarnessLifecycle(harness, "{callback}")',
                  "Assert.AreEqual(0, source.SubscriberCount)", "UnityEngine.Object.DestroyImmediate(owner)")
        require(all(token in body for token in needed), "lifecycle test contract missing token: " + name)
        checks[name] = {"callback": callback, "uses_actual_harness_component": True,
                        "reflection_invocation": True, "subscriber_before": 2, "subscriber_after": 0,
                        "destroy_immediate_in_finally": True}
    require("Assert.AreEqual(PersonalLiveCaptureState.Cancelled, controller.State)" in after_methods[NEW_TESTS[0]],
            "OnDisable terminal state assertion missing")
    require('GetPrivateHarnessField(harness, "controller")' in after_methods[NEW_TESTS[1]],
            "OnDestroy controller cleanup assertion missing")
    return {"schema_version": SCHEMA + ".test_method_inventory", "before_count": len(before_methods),
            "after_count": len(after_methods), "existing_preserved_count": len(before_methods),
            "existing_changed_count": 0, "before_methods": sorted(before_methods),
            "new_methods": list(NEW_TESTS), "lifecycle_checks": checks, "status": "PASS"}


def compiler_count(text: str, kind: str) -> int:
    return len(re.findall(r"\b" + kind + r" CS[0-9]+\b", text))


def offline_compile_and_probe(root: Path, v10) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    mono = Path(r"C:\Program Files\Unity\Hub\Editor\6000.3.14f1\Editor\Data\MonoBleedingEdge\bin\mono.exe")
    csc = Path(r"C:\Program Files\Unity\Hub\Editor\6000.3.14f1\Editor\Data\MonoBleedingEdge\lib\mono\4.5\csc.exe")
    require(mono.is_file() and csc.is_file(), "Unity Mono Roslyn compiler unavailable")
    unity_stub = v10.UNITY_AND_BRIDGE_STUB + UNITY_COMPONENT_STUB_EXTENSION
    nunit_stub = v10.NUNIT_STUB.replace(
        "public static void IsNull(object value, string message = null) {}",
        "public static void IsNull(object value, string message = null) {}\n    public static void IsNotNull(object value, string message = null) {}")
    with tempfile.TemporaryDirectory(prefix="WIS-Step9J-v11-offline-") as temporary:
        temp = Path(temporary)
        unity_path = temp / "UnityAndBridgeComponentCompileStubs.cs"
        nunit_path = temp / "NUnitCompileStubs.cs"
        probe_path = temp / "HarnessComponentLifecycleProbe.cs"
        unity_path.write_text(unity_stub, encoding="utf-8")
        nunit_path.write_text(nunit_stub, encoding="utf-8")
        probe_path.write_text(HARNESS_LIFECYCLE_PROBE, encoding="utf-8")
        runtime_dll = temp / "Step9JRuntime.dll"
        production = subprocess.run(
            [str(mono), str(csc), "/nologo", "/target:library", "/langversion:latest", "/out:" + str(runtime_dll),
             str(unity_path), *[str((root / relative).resolve()) for relative in v10.RUNTIME_SOURCES]],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        production_output = (production.stdout or "") + (production.stderr or "")
        require(production.returncode == 0, "offline production compile failed:\n" + production_output[-6000:])
        test_dll = temp / "Step9JTests.dll"
        test = subprocess.run(
            [str(mono), str(csc), "/nologo", "/target:library", "/langversion:latest", "/out:" + str(test_dll),
             "/r:" + str(runtime_dll), str(nunit_path), str((root / TEST_REL).resolve())],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        test_output = (test.stdout or "") + (test.stderr or "")
        require(test.returncode == 0, "offline test fixture compile failed:\n" + test_output[-6000:])
        probe_exe = temp / "HarnessComponentLifecycleProbe.exe"
        probe_compile = subprocess.run(
            [str(mono), str(csc), "/nologo", "/target:exe", "/langversion:latest", "/out:" + str(probe_exe),
             "/r:" + str(runtime_dll), str(probe_path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        probe_compile_output = (probe_compile.stdout or "") + (probe_compile.stderr or "")
        require(probe_compile.returncode == 0, "offline lifecycle probe compile failed:\n" + probe_compile_output[-6000:])
        probe = subprocess.run([str(mono), str(probe_exe)], capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=120)
        output = ((probe.stdout or "") + (probe.stderr or "")).lstrip("\ufeff")
        require(probe.returncode == 0, "offline harness lifecycle probe failed:\n" + output[-6000:])
    compile_evidence = {
        "schema_version": SCHEMA + ".offline_compile", "compiler": "unity_mono_roslyn_csc",
        "production_source_count": len(v10.RUNTIME_SOURCES),
        "production_compiler_error_count": compiler_count(production_output, "error"),
        "production_compiler_warning_count": compiler_count(production_output, "warning"),
        "test_fixture_compiler_error_count": compiler_count(test_output, "error"),
        "test_fixture_compiler_warning_count": compiler_count(test_output, "warning"),
        "probe_compiler_error_count": compiler_count(probe_compile_output, "error"),
        "unity_editor_invoked": False, "unity_compile_pass_claimed": False,
        "editmode_or_playmode_pass_claimed": False, "status": "PASS",
    }
    expected = {
        "HARNESS_ON_DISABLE_SUBSCRIBERS_BEFORE": 2, "HARNESS_ON_DISABLE_SUBSCRIBERS_AFTER": 0,
        "HARNESS_ON_DESTROY_SUBSCRIBERS_BEFORE": 2, "HARNESS_ON_DESTROY_SUBSCRIBERS_AFTER": 0,
    }
    actual = {}
    for key in expected:
        match = re.search(r"^" + key + r"=([0-9]+)$", output, re.M)
        actual[key] = int(match.group(1)) if match else None
    passed = actual == expected and "HARNESS_COMPONENT_LIFECYCLE=PASS" in output and probe.returncode == 0
    lifecycle = {"schema_version": SCHEMA + ".harness_component_lifecycle_probe",
                 "probe_uses_actual_personal_live_recognition_harness": True,
                 "controller_direct_disable_dispose_used_as_substitute": False,
                 "private_callbacks_invoked_by_reflection": ["OnDisable", "OnDestroy"],
                 "expected": expected, "actual": actual, "probe_exit_code": probe.returncode,
                 "harness_component_lifecycle": "PASS" if passed else "FAIL",
                 "unity_editor_invoked": False, "status": "PASS" if passed else "FAIL"}
    require(passed, "offline harness lifecycle probe output mismatch")
    require(compile_evidence["production_compiler_error_count"] == 0 and
            compile_evidence["test_fixture_compiler_error_count"] == 0 and
            compile_evidence["probe_compiler_error_count"] == 0, "offline compiler errors are nonzero")
    return compile_evidence, lifecycle, unity_stub, nunit_stub


def staged_count(root: Path) -> int:
    result = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=root,
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    require(result.returncode == 0, "git staged-file query failed")
    return len([line for line in result.stdout.splitlines() if line.strip()])


def assertion(key: str, status: str, subject: str, evidence: Any) -> dict[str, Any]:
    return {"assertion_key": key, "status": status, "subject": subject, "evidence": evidence}


def build(root: Path) -> dict[str, Any]:
    result = root / RESULT_REL
    delivery = root / DELIVERY_REL
    detached = Path(str(delivery) + ".sha256")
    validation_root = root / "Validation"
    for target in (result, delivery, detached):
        require(not target.exists() and not target.is_symlink(), "never-overwrite target exists: " + str(target))
    v10 = import_fixed_module(root, V10_TOOL_REL, V10_TOOL_SIZE, V10_TOOL_SHA256, "step9j_v10_fixed")
    transaction = import_fixed_module(root, V11_TRANSACTION_REL, V11_TRANSACTION_SIZE,
                                      V11_TRANSACTION_SHA256, "step9i_v11_transaction")
    v10_input, v10_files = verify_v10_delivery(root, transaction)
    production, production_before = verify_production_preservation(root, v10_files)
    approved, approved_before = v10.verify_fixed_files(root)
    before_test = v10_files[PACKAGED_TEST]
    after_test = (root / TEST_REL).read_bytes()
    test_change = validate_test_change(before_test, after_test)
    after_test_digest = sha256_bytes(after_test)
    compile_evidence, lifecycle, unity_stub, nunit_stub = offline_compile_and_probe(root, v10)
    approved_after = {relative: sha256_file(root / PurePosixPath(relative)) for relative in approved_before}
    production_after = {relative: sha256_file(root / PurePosixPath(relative)) for relative in production_before}
    require(approved_before == approved_after and production_before == production_after,
            "fixed approved/production file changed during validation")
    require(sha256_file(root / TEST_REL) == after_test_digest, "allowed test source changed during validation")
    require(staged_count(root) == 0, "Git staged file count is nonzero")
    transaction_identity = {"records": [{"project_relative_path": relative}
                                         for relative in sorted(set(approved_before) | set(production_before) | {TEST_REL.as_posix()})]}
    injection = transaction.run_failure_injection_regression(root, validation_root, transaction_identity)
    never_overwrite = transaction.run_never_overwrite_regression(validation_root)
    require(injection["verdict"] == "PASS" and never_overwrite["verdict"] == "PASS",
            "packaging transaction regression failed")

    diff = "".join(difflib.unified_diff(
        before_test.decode("utf-8").splitlines(keepends=True),
        after_test.decode("utf-8").splitlines(keepends=True),
        fromfile="v1.0/PersonalLiveRecognitionIntegrationV1Tests.cs",
        tofile="v1.1/PersonalLiveRecognitionIntegrationV1Tests.cs"))
    source_change = {"schema_version": SCHEMA + ".source_change_inventory",
                     "allowed_changed_project_file_count": 1, "actual_changed_project_file_count": 1,
                     "changed_project_file": TEST_REL.as_posix(),
                     "before_size_bytes": len(before_test), "before_sha256": sha256_bytes(before_test),
                     "after_size_bytes": len(after_test), "after_sha256": after_test_digest,
                     "test_meta_changed": False, "production_changed": False, "status": "PASS"}
    primary = {"schema_version": SCHEMA + ".primary_verdict",
               "primary_verdict": "STEP9J_TEST_ONLY_LIFECYCLE_COVERAGE_V1_1_SOURCE_AND_PACKAGE_READY",
               "harness_on_disable_subscriber_regression": "IMPLEMENTED",
               "harness_on_destroy_subscriber_regression": "IMPLEMENTED",
               "v1_0_production_route_and_assets": "PRESERVED",
               "unity_compile_editmode_playmode": "UNVERIFIED", "external_review": "PENDING",
               "status_strings": [
                   "STEP9J_TEST_ONLY_LIFECYCLE_COVERAGE_V1_1_SOURCE_AND_PACKAGE_READY",
                   "STEP9J_HARNESS_ON_DISABLE_SUBSCRIBER_REGRESSION_IMPLEMENTED",
                   "STEP9J_HARNESS_ON_DESTROY_SUBSCRIBER_REGRESSION_IMPLEMENTED",
                   "STEP9J_V1_0_PRODUCTION_ROUTE_AND_ASSETS_PRESERVED",
                   "STEP9J_V1_1_UNITY_COMPILE_EDITMODE_PLAYMODE_UNVERIFIED",
                   "BLOCKED_PENDING_EXTERNAL_REVIEW"]}
    assertions = [
        assertion("fixed_v1_0_delivery_identity", "PASS", V10_DELIVERY_REL.as_posix(), v10_input),
        assertion("v1_0_production_two_sources_preserved", "PASS", "controller+harness", production),
        assertion("v1_0_production_meta_preserved", "PASS", "controller+harness meta", {"count": 2}),
        assertion("v1_0_test_meta_preserved", "PASS", TEST_META_REL.as_posix(), FIXED_PROJECT_FILES[-1][2]),
        assertion("approved_step9i_step9c_files_preserved", "PASS", "approved files", {"count": approved["file_count"]}),
        assertion("existing_test_methods_fifteen_preserved", "PASS", TEST_REL.as_posix(), test_change["existing_preserved_count"]),
        assertion("harness_on_disable_test_present", "PASS", NEW_TESTS[0], test_change["lifecycle_checks"][NEW_TESTS[0]]),
        assertion("harness_on_destroy_test_present", "PASS", NEW_TESTS[1], test_change["lifecycle_checks"][NEW_TESTS[1]]),
        assertion("lifecycle_tests_use_actual_harness_component", "PASS", "new tests", True),
        assertion("lifecycle_tests_reflection_callbacks", "PASS", "OnDisable+OnDestroy", True),
        assertion("lifecycle_tests_subscriber_before_two_after_zero", "PASS", "new tests", {"before": 2, "after": 0}),
        assertion("offline_production_compile_error_zero", "PASS", "Unity Mono Roslyn", 0),
        assertion("offline_test_fixture_compile_error_zero", "PASS", "Unity Mono Roslyn", 0),
        assertion("offline_harness_component_lifecycle_probe", "PASS", "actual harness callbacks", lifecycle),
        assertion("controller_direct_cleanup_not_substituted", "PASS", "offline probe", False),
        assertion("approved_files_before_after_unchanged", "PASS", "fixed inputs", {"changed": 0}),
        assertion("allowed_test_source_only_changed", "PASS", TEST_REL.as_posix(), source_change),
        assertion("packaging_failure_injection_four", "PASS", "approved transaction", injection),
        assertion("never_overwrite", "PASS", "approved transaction", never_overwrite),
        assertion("unity_tests_not_claimed", "PASS", "scope", {"Unity": "UNVERIFIED"}),
        assertion("git_staged_zero", "PASS", "Git index", 0),
    ]
    keys = [item["assertion_key"] for item in assertions]
    require(len(keys) == len(set(keys)), "duplicate assertion key")
    tool_rel = Path("Tools/s001_step9j_personal_live_recognition_integration_v1_1.py")
    file_map: dict[str, bytes] = {
        "input_provenance.json": json_bytes({"schema_version": SCHEMA + ".input_provenance",
                                               "step9j_v1_0": v10_input,
                                               "approved_transaction_tool": {"path": V11_TRANSACTION_REL.as_posix(),
                                               "size_bytes": V11_TRANSACTION_SIZE, "sha256": V11_TRANSACTION_SHA256}}),
        "source_before/PersonalLiveRecognitionIntegrationV1Tests.cs": before_test,
        "source_after/PersonalLiveRecognitionIntegrationV1Tests.cs": after_test,
        "unified_diff/PersonalLiveRecognitionIntegrationV1Tests.diff": diff.encode("utf-8"),
        "source_change_inventory.json": json_bytes(source_change),
        "test_method_inventory.json": json_bytes(test_change),
        "v1_0_production_file_preservation.json": json_bytes(production),
        "approved_file_preservation.json": json_bytes(approved),
        "harness_component_lifecycle_offline_probe.json": json_bytes(lifecycle),
        "offline_csharp_compile_check.json": json_bytes(compile_evidence),
        "packaging_transaction_failure_injection_regression.json": json_bytes(injection),
        "never_overwrite_regression.json": json_bytes(never_overwrite),
        "primary_verdict.json": json_bytes(primary),
        "ASSERTIONS.jsonl": b"".join(json.dumps(item, ensure_ascii=False, allow_nan=False,
                                                  separators=(",", ":")).encode("utf-8") + b"\n" for item in assertions),
        "analysis/s001_step9j_personal_live_recognition_integration_v1_1.py": (root / tool_rel).read_bytes(),
        "analysis/compile_fixtures/UnityAndBridgeComponentCompileStubs.cs": unity_stub.encode("utf-8"),
        "analysis/compile_fixtures/NUnitCompileStubs.cs": nunit_stub.encode("utf-8"),
        "analysis/compile_fixtures/HarnessComponentLifecycleProbe.cs": HARNESS_LIFECYCLE_PROBE.encode("utf-8"),
    }
    report = "\n".join([
        "# WordsInSilence Step 9J v1.1 — Test-only Component Lifecycle Coverage Repair", "",
        "## Outcome", "",
        "Only PersonalLiveRecognitionIntegrationV1Tests.cs changed. Two independent tests now instantiate the actual PersonalLiveRecognitionHarness component and explicitly invoke its private OnDisable and OnDestroy callbacks by reflection.",
        "", "## Verification", "",
        f"- Existing v1.0 test methods preserved byte-for-byte: `{test_change['existing_preserved_count']}/15`",
        "- New lifecycle tests: `2`; total fixture tests: `17`",
        f"- OnDisable subscribers: `{lifecycle['actual']['HARNESS_ON_DISABLE_SUBSCRIBERS_BEFORE']} → {lifecycle['actual']['HARNESS_ON_DISABLE_SUBSCRIBERS_AFTER']}`",
        f"- OnDestroy subscribers: `{lifecycle['actual']['HARNESS_ON_DESTROY_SUBSCRIBERS_BEFORE']} → {lifecycle['actual']['HARNESS_ON_DESTROY_SUBSCRIBERS_AFTER']}`",
        f"- Offline production/test/probe compiler errors: `{compile_evidence['production_compiler_error_count']}/{compile_evidence['test_fixture_compiler_error_count']}/{compile_evidence['probe_compiler_error_count']}`",
        f"- Packaging failure injection: `{injection['pass_count']}/{injection['injection_point_count']}`; never-overwrite: `{never_overwrite['verdict']}`",
        "", "## Limits", "",
        "Unity Editor, Unity compilation, EditMode, PlayMode, and webcam were not executed. The two new Unity tests are implemented and compile-only verified, not reported as PASS.",
        "", "## Status", "", "```text", *primary["status_strings"], "```", "",
    ])
    file_map["VALIDATION_REPORT.md"] = report.encode("utf-8")
    records = [{"path": name, "size_bytes": len(data), "sha256": sha256_bytes(data)}
               for name, data in sorted(file_map.items())]
    file_map["output_file_manifest.json"] = json_bytes({"schema_version": SCHEMA + ".output_file_manifest",
                                                          "self_excluded": True, "file_count": len(records),
                                                          "files": records})
    zip_one = transaction.deterministic_zip_bytes(file_map)
    require(zip_one == transaction.deterministic_zip_bytes(file_map), "deterministic ZIP regeneration mismatch")
    detached_bytes = transaction.strict_detached_bytes(zip_one, delivery.name)

    def post_validate(final_result: Path, final_delivery: Path, final_detached: Path) -> None:
        require(final_delivery.read_bytes() == zip_one, "committed delivery differs")
        transaction.verify_strict_detached(final_detached.read_bytes(), zip_one, final_delivery.name)
        require(all((final_result / PurePosixPath(name)).read_bytes() == data for name, data in file_map.items()),
                "committed result file mismatch")
        safety = transaction.inspect_zip(zip_one, set(file_map))
        require(safety["crc"] == "PASS" and safety["duplicates"] == safety["unsafe"] ==
                safety["symlinks"] == safety["encrypted"] == 0 and safety["membership"] == "PASS",
                "committed ZIP safety failure")

    transaction.commit_packaging_transaction(
        result_files=file_map, delivery_bytes=zip_one, detached_bytes=detached_bytes,
        final_result=result, final_delivery=delivery, final_detached=detached,
        allowed_root=validation_root, post_rename_validator=post_validate,
        transaction_token="step9j_v1_1_production")
    _, production_final = verify_production_preservation(root, v10_files)
    _, approved_final = v10.verify_fixed_files(root)
    require(production_final == production_before and approved_final == approved_before,
            "fixed files changed during package commit")
    require(sha256_file(root / TEST_REL) == after_test_digest, "test source changed during package commit")
    require(staged_count(root) == 0, "Git staged files changed during package commit")
    return {"result_directory": str(result), "result_file_count": len(file_map),
            "delivery_zip": str(delivery), "delivery_size_bytes": len(zip_one),
            "delivery_sha256": sha256_bytes(zip_one), "detached_checksum": str(detached),
            "detached_size_bytes": len(detached_bytes), "detached_file_sha256": sha256_bytes(detached_bytes),
            "test_method_count": 17, "new_test_method_count": 2,
            "assertion_count": len(assertions), "assertion_pass_count": len(assertions),
            "git_staged_file_count": 0, "status_strings": primary["status_strings"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(build(args.project_root.resolve()), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
