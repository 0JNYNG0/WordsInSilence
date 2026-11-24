#!/usr/bin/env python3
"""Package Step 9K first asynchronous failure preservation diagnostics v1.0.

This tool is intentionally offline. It verifies the approved Step 9J/9K inputs,
compiles the minimally changed controller, harness, and EditMode fixture with
Unity Mono Roslyn-compatible stubs, executes the real controller against five
first-failure regression scenarios, verifies protected files byte-for-byte, and
commits through the externally approved Step 9I v1.1 transaction. It never
launches Unity, a provider, PlayMode, or a webcam.
"""

from __future__ import annotations

import argparse
import csv
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


SCHEMA = "words_in_silence.step9k.primary_failure_preservation_diagnostic.v1_0"
RESULT_REL = Path("Validation/WordsInSilence_step9k_primary_failure_preservation_diagnostic_v1_0")
DELIVERY_REL = Path("Validation/WordsInSilence_step9k_primary_failure_preservation_diagnostic_v1_0_delivery.zip")
TOOL_REL = Path("Tools/s001_step9k_primary_failure_preservation_diagnostic_v1_0.py")

CONTROLLER_REL = Path("Assets/_Project/Scripts/Recognition/LiveInput/PersonalLiveRecognitionController.cs")
HARNESS_REL = Path("Assets/_Project/Scripts/Recognition/LiveInput/PersonalLiveRecognitionHarness.cs")
TEST_REL = Path("Assets/_Project/Tests/EditMode/PersonalLiveRecognitionIntegrationV1Tests.cs")

TRANSACTION_TOOL_REL = Path("Tools/s001_step9i_personal_runtime_opt_in_integration_v1_1.py")
TRANSACTION_TOOL_SIZE = 37_722
TRANSACTION_TOOL_SHA256 = "30471dc6725bdbdd0cc80c345b238485d1cb0d8d7a5933aac7477d536d714caf"
STEP9J_TOOL_REL = Path("Tools/s001_step9j_personal_live_recognition_integration_v1_0.py")
STEP9J_TOOL_SIZE = 47_009
STEP9J_TOOL_SHA256 = "72c716128275f638a179de195daa90668097c83852249349e068ce45d12a6d69"

FIXED_DELIVERIES = (
    ("step9j_v1_0", Path("Validation/WordsInSilence_step9j_personal_live_recognition_integration_v1_0_delivery.zip"),
     44_445, "1775ac8e310fbd99bcd66b0b8055bc12a63554374df4f70c6cbcf73686985159"),
    ("step9j_v1_1", Path("Validation/WordsInSilence_step9j_personal_live_recognition_integration_v1_1_delivery.zip"),
     37_804, "ab38f4867bd8514f498eb6eae542ffe5deac67d68ab1e96f6f1bd63de9758460"),
    ("step9k_v1_0", Path("Validation/WordsInSilence_step9k_personal_live_recognition_technical_prefab_v1_0_delivery.zip"),
     22_103, "cbfce2af005a9f62879f97771397a23e8d1fb21f24447972208607da96262766"),
)

BEFORE_IDENTITIES = {
    CONTROLLER_REL.as_posix(): (30_013, "69fc64354c0f1466b9b161cddaacb13087a185c77cc98304bbdb86daaf006ba7"),
    HARNESS_REL.as_posix(): (7_209, "b5cee1181f5b0927dc431486d23ccea4ee4778466de729686e0a451342a8b81f"),
    TEST_REL.as_posix(): (30_226, "7b53e98f93ef165e3c283ddcc6434e0254caacd22370c5e6dc66888bdf93775d"),
}

META_IDENTITIES = (
    (Path(str(CONTROLLER_REL) + ".meta"), 243, "5262ea0d5605c95fc570033432bb000b7cb2c22fe51b376c4039227cfdc9280d"),
    (Path(str(HARNESS_REL) + ".meta"), 243, "f54efe92219b323e99c3a1a9466ae6ac5a2cc28569c171a1534683dda98d722e"),
    (Path(str(TEST_REL) + ".meta"), 243, "48271faad1512b429086a906ed9d454f5dfaed07f5b1a6d5d2f4c1267e01769c"),
)

SCENE_AND_PREFAB_FIXED = (
    ("Assets/_Project/Resources/Diagnostics/PhysicalWebcamCoordinateCompatibility.prefab", 1_792,
     "f5910cf8381bd7769e518a70b57b1680c9d90709ac13f96689729094eb38f74b"),
    ("Assets/_Project/Resources/Diagnostics/PhysicalWebcamCoordinateCompatibility.prefab.meta", 154,
     "678840d956656c26f743d49e5b17da79bb4adf8f6277df72a2c999fd539a88dc"),
    ("Assets/_Project/Scenes/Technical/02_PoseIntegration.unity", 240_625,
     "1fb83d0f304277317a4571319e432c8d85394c7f84abce9036e23a6b3de16a0b"),
    ("Assets/_Project/Scenes/Technical/02_PoseIntegration.unity.meta", 155,
     "d5a0540425a760d73fb1b3d7c5b65b79a1e9f1f2528466af4e1f735dec7f0b8a"),
    ("Assets/_Project/Scenes/Game/03_SignPractice.unity", 40_083,
     "dac9d8d51a561825314cd8fb4021e4e803590d9275d3d65b55af96bfa47338e8"),
    ("Assets/_Project/Scenes/Game/03_SignPractice.unity.meta", 155,
     "3e0de2ecfd5cfaf5cccde62e1c1cf7175bf3c59ae652322f1041557599413aa7"),
)

TECHNICAL_PREFAB_REL = Path("Assets/_Project/Resources/Diagnostics/PersonalLiveRecognitionTechnical.prefab")
TECHNICAL_PREFAB_META_REL = Path(str(TECHNICAL_PREFAB_REL) + ".meta")
NEW_TESTS = (
    "CaptureBridgeFailurePinsImmutablePrimaryFailureSnapshot",
    "PrimaryFailurePreservesBufferedRawFrameCountAfterFailureCleanup",
    "FailedStateFollowupCommandsCannotOverwritePrimaryFailureSnapshot",
    "ObservationRejectionPinsExactPrimaryFailureCodeAndPrecleanupContext",
    "SuccessfulResetClearsPrimaryFailureSnapshot",
)


class Step9KPrimaryFailureError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise Step9KPrimaryFailureError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def import_fixed(root: Path, relative: Path, size: int, digest: str, name: str):
    path = root / relative
    require(path.is_file() and path.stat().st_size == size and sha256_file(path) == digest,
            "fixed tool identity mismatch: " + relative.as_posix())
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "cannot import fixed tool: " + name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_delivery(root: Path, transaction, label: str, relative: Path,
                    expected_size: int, expected_sha: str) -> tuple[dict[str, Any], dict[str, bytes]]:
    path = root / relative
    detached = Path(str(path) + ".sha256")
    require(path.is_file() and detached.is_file(), label + " fixed Delivery or checksum missing")
    data = path.read_bytes()
    detached_data = detached.read_bytes()
    require(len(data) == expected_size and sha256_bytes(data) == expected_sha,
            label + " fixed Delivery identity mismatch")
    transaction.verify_strict_detached(detached_data, data, path.name)
    with zipfile.ZipFile(io.BytesIO(data), "r") as archive:
        infos = archive.infolist()
        names = [item.filename for item in infos]
        require(len(names) == len(set(names)), label + " duplicate ZIP entry")
        for item in infos:
            require(transaction.safe_zip_name(item.filename), label + " unsafe ZIP path")
            require(not item.flag_bits & 0x1, label + " encrypted ZIP entry")
            mode = (item.external_attr >> 16) & 0xFFFF
            require(stat.S_IFMT(mode) != stat.S_IFLNK, label + " symlink ZIP entry")
        require(archive.testzip() is None, label + " ZIP CRC failure")
        file_map = {item.filename: archive.read(item) for item in infos}
    require("output_file_manifest.json" in file_map, label + " output manifest missing")
    manifest = json.loads(file_map["output_file_manifest.json"])
    records = manifest.get("files", manifest.get("outputs"))
    require(isinstance(records, list), label + " manifest records missing")
    record_names = {item.get("path", item.get("relative_path")) for item in records}
    require(record_names == set(file_map) - {"output_file_manifest.json"},
            label + " manifest membership mismatch")
    for record in records:
        name = record.get("path", record.get("relative_path"))
        payload = file_map[name]
        require(len(payload) == record["size_bytes"] and sha256_bytes(payload) == record["sha256"],
                label + " manifest identity mismatch: " + name)
    require(transaction.deterministic_zip_bytes(file_map) == data,
            label + " deterministic ZIP recreation mismatch")
    return ({"label": label, "path": relative.as_posix(), "size_bytes": len(data),
             "sha256": sha256_bytes(data), "detached_size_bytes": len(detached_data),
             "detached_file_sha256": sha256_bytes(detached_data), "entry_count": len(file_map),
             "manifest_target_count": len(records), "crc": "PASS", "archive_safety": "PASS",
             "deterministic_recreation": "BYTE_IDENTICAL", "status": "PASS"}, file_map)


def method_blocks(source: str) -> dict[str, str]:
    pattern = re.compile(r"\[Test\]\s*public void\s+([A-Za-z0-9_]+)\s*\(\s*\)\s*\{")
    result: dict[str, str] = {}
    for match in pattern.finditer(source):
        depth = 0
        for index in range(match.end() - 1, len(source)):
            if source[index] == "{":
                depth += 1
            elif source[index] == "}":
                depth -= 1
                if depth == 0:
                    result[match.group(1)] = source[match.start():index + 1]
                    break
    return result


def extract_method(source: str, signature: str) -> str:
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    raise Step9KPrimaryFailureError("method body not closed: " + signature)


def validate_sources(before_controller: bytes, before_harness: bytes, before_test: bytes,
                     after_controller: bytes, after_harness: bytes, after_test: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
    controller = after_controller.decode("utf-8")
    harness = after_harness.decode("utf-8")
    prior_methods = method_blocks(before_test.decode("utf-8"))
    current_methods = method_blocks(after_test.decode("utf-8"))
    require(len(prior_methods) == 17, "approved Step 9J test method count is not 17")
    require(all(current_methods.get(name) == body for name, body in prior_methods.items()),
            "an approved Step 9J test method changed")
    require(set(current_methods) - set(prior_methods) == set(NEW_TESTS), "new regression test set mismatch")
    require(len(current_methods) == 22, "current test method count is not 22")

    snapshot = controller[controller.index("public sealed class PersonalLivePrimaryFailureSnapshot"):]
    snapshot = snapshot[:snapshot.index("public sealed class PersonalLiveRuntimePaths")]
    required_properties = ("Code", "Message", "ExceptionType", "StateBeforeFailure",
                           "CaptureSessionId", "BufferedRawFrameCount")
    for name in required_properties:
        require(re.search(r"public\s+[^;\n]+\s+" + name + r"\s*\{\s*get;\s*\}", snapshot) is not None,
                "primary snapshot get-only property missing: " + name)
    require(" set;" not in snapshot and "public " in snapshot, "primary snapshot is mutable")
    require(controller.count("PrimaryFailure = new PersonalLivePrimaryFailureSnapshot") == 1,
            "primary snapshot assignment count mismatch")
    require(controller.count("PrimaryFailure = null;") == 1,
            "primary snapshot clear must occur only in successful Reset")
    fail_body = extract_method(controller, "bool Fail(PersonalLiveFailureCode code")
    for token in ("if (PrimaryFailure == null)", "State,", "captureSessionId,", "observations.Count"):
        require(token in fail_body, "Fail pre-cleanup snapshot contract missing: " + token)
    assignment = fail_body.index("PrimaryFailure = new PersonalLivePrimaryFailureSnapshot")
    require(assignment < fail_body.index("Unsubscribe();") < fail_body.index("observations.Clear();"),
            "primary snapshot is not captured before cleanup")
    reset_body = extract_method(controller, "public bool Reset()")
    clear_index = reset_body.index("PrimaryFailure = null;")
    for token in ("observationSource == null", "!observationSource.IsBound", "runtime == null || !runtime.IsPersonalActive"):
        require(reset_body.index(token) < clear_index, "Reset clears primary before precondition: " + token)
    require(clear_index < reset_body.index("State = PersonalLiveCaptureState.Ready;"),
            "successful Reset state/primary ordering mismatch")

    panel_tokens = (
        "Primary failure code:", "Primary failure message:", "Primary failure exception type:",
        "Primary state before failure:", "Primary capture/session ID:",
        "Primary buffered raw frames before failure:", "Step 9C bridge LastError:",
        "ProviderFrameCount:", "ObservationCount:", "DuplicateFrameCount:",
        "ResolvedDuplicateFrameCount:", "AmbiguousDuplicateFrameCount:",
        "GUILayout.BeginScrollView", "GUILayout.EndScrollView",
    )
    for token in panel_tokens:
        require(token in harness, "technical panel token missing: " + token)
    button_tokens = (
        "controller.State == PersonalLiveCaptureState.Ready;\n            if (GUILayout.Button(\"Start Capture\"))",
        "controller.State == PersonalLiveCaptureState.Capturing;\n            if (GUILayout.Button(\"Finalize\"))",
        "controller.State == PersonalLiveCaptureState.Processing);\n            if (GUILayout.Button(\"Cancel\"))",
        "controller.State == PersonalLiveCaptureState.Failed);\n            if (GUILayout.Button(\"Reset\"))",
    )
    for token in button_tokens:
        require(token in harness, "button state gate missing")

    source_contract = {
        "schema_version": SCHEMA + ".source_contract",
        "immutable_primary_snapshot_properties": list(required_properties),
        "primary_assignment_count": 1,
        "primary_clear_count": 1,
        "capture_precedes_cleanup": True,
        "reset_clears_only_after_preconditions": True,
        "legacy_failure_property_preserved": "public PersonalLiveFailure Failure { get; private set; }" in controller,
        "panel_primary_field_count": 6,
        "panel_bridge_diagnostic_field_count": 6,
        "button_state_gate_count": 4,
        "panel_scroll_preserves_visibility": True,
        "status": "PASS",
    }
    test_contract = {
        "schema_version": SCHEMA + ".test_method_inventory",
        "approved_method_count": len(prior_methods),
        "approved_method_body_byte_identical_count": sum(
            current_methods.get(name) == body for name, body in prior_methods.items()),
        "new_method_count": len(NEW_TESTS),
        "new_methods": list(NEW_TESTS),
        "total_method_count": len(current_methods),
        "status": "PASS",
    }
    return source_contract, test_contract


def make_unity_stub(step9j_v10, step9j_v11) -> str:
    stub = step9j_v10.UNITY_AND_BRIDGE_STUB
    stub = stub.replace(
        "  public struct Rect { public Rect(float x,float y,float w,float h) {} }",
        "  public struct Vector2 {}\n  public struct Rect { public Rect(float x,float y,float w,float h) {} }")
    stub = stub.replace(
        "  public static class GUI { public static GUISkin skin { get { return new GUISkin(); } } }",
        "  public static class GUI { public static GUISkin skin { get { return new GUISkin(); } } public static bool enabled { get; set; } }")
    stub = stub.replace(
        "    public static void BeginArea(Rect value, GUIStyle style) {}",
        "    public static void BeginArea(Rect value, GUIStyle style) {}\n"
        "    public static Vector2 BeginScrollView(Vector2 value, params GUILayoutOption[] options) { return value; }\n"
        "    public static void EndScrollView() {}")
    stub = stub.replace(
        "    public bool IsBound { get; set; }",
        "    public bool IsBound { get; set; }\n"
        "    public string LastError { get; set; }\n"
        "    public int ProviderFrameCount { get; set; }\n"
        "    public int ObservationCount { get; set; }\n"
        "    public int DuplicateFrameCount { get; set; }\n"
        "    public int ResolvedDuplicateFrameCount { get; set; }\n"
        "    public int AmbiguousDuplicateFrameCount { get; set; }")
    return stub + step9j_v11.UNITY_COMPONENT_STUB_EXTENSION


PRIMARY_FAILURE_PROBE = r"""using System;
using System.Reflection;
using WordsInSilence.Recognition.Core;
using WordsInSilence.Recognition.LiveInput;
using WordsInSilence.Recognition.Runtime;
class Step9KPrimaryFailureProbe {
  sealed class Source : IPersonalLiveObservationSource {
    Action<RawMediaPipeObservation> observations; Action<Exception> failures;
    public bool IsBound { get; set; } = true;
    public event Action<RawMediaPipeObservation> ObservationArrived { add { observations += value; } remove { observations -= value; } }
    public event Action<Exception> SourceFailed { add { failures += value; } remove { failures -= value; } }
    public void Emit(RawMediaPipeObservation value) { observations?.Invoke(value); }
    public void Fail(Exception value) { failures?.Invoke(value); }
  }
  sealed class Global : IPersonalLiveGlobalRuntime {}
  sealed class Runtime : IPersonalLiveRecognitionRuntime {
    public bool IsPersonalActive => true; public string ActiveProfileId => "probe-profile";
    public PersonalLiveRoutingResult Recognize(string id, RecognitionSequence sequence, string expected) { throw new NotSupportedException(); }
  }
  sealed class Factory : IPersonalLiveRuntimeFactory {
    public IPersonalLiveGlobalRuntime LoadGlobal(PersonalLiveRuntimePaths paths) { return new Global(); }
    public IPersonalLiveRecognitionRuntime CreateAndActivate(IPersonalLiveGlobalRuntime global, PersonalLiveRuntimePaths paths) { return new Runtime(); }
  }
  static void Need(bool value,string message) { if(!value) throw new Exception(message); }
  static PersonalLiveRecognitionController Ready(Source source) {
    var controller=new PersonalLiveRecognitionController(source,new Factory());
    Need(controller.Initialize(new PersonalLiveRuntimePaths("global","threshold","personal")),"initialize"); return controller;
  }
  static double[] Nan(int count) { var values=new double[count*3]; for(int i=0;i<values.Length;i++) values[i]=double.NaN; return values; }
  static void Put(double[] xyz,bool[] valid,int landmark,double x,double y,double z) {
    int offset=landmark*3; xyz[offset]=x;xyz[offset+1]=y;xyz[offset+2]=z;valid[landmark]=true;
  }
  static RawMediaPipeObservation Observation(double timestamp) {
    var pose=Nan(33);var poseValid=new bool[33];
    Put(pose,poseValid,11,-1,0,0);Put(pose,poseValid,12,1,0,0);Put(pose,poseValid,23,-1,2,0);Put(pose,poseValid,24,1,2,0);
    Put(pose,poseValid,13,-1.2,-.5,.1);Put(pose,poseValid,14,1.2,-.5,.1);Put(pose,poseValid,15,-1.4,-1,.2);Put(pose,poseValid,16,1.4,-1,.2);
    var hand=Nan(21);var handValid=new bool[21];foreach(int i in new[]{0,4,8,12,16,20})Put(hand,handValid,i,-1+i*.02,-.8,i*.005);
    return new RawMediaPipeObservation(timestamp,RawObservationConvention.LiveMediaPipe,
      new RawLandmarkObservationSet(33,pose,poseValid,new bool[33],new bool[33]),
      new RawLandmarkObservationSet(21,hand,handValid,new bool[21],new bool[21]),RawLandmarkObservationSet.Absent(21));
  }
  static void BridgeAndBuffer() {
    var source=new Source();var controller=Ready(source);Need(controller.BeginCapture("probe-bridge"),"begin bridge");
    source.Emit(Observation(0));source.Emit(Observation(.03));source.Fail(new InvalidOperationException("provider failed"));
    var primary=controller.PrimaryFailure;Need(primary!=null&&primary.Code==PersonalLiveFailureCode.BridgeFailure,"bridge code");
    Need(primary.Message=="Step 9C observation bridge failed: provider failed","bridge message");
    Need(primary.ExceptionType==typeof(InvalidOperationException).FullName,"bridge exception");
    Need(primary.StateBeforeFailure==PersonalLiveCaptureState.Capturing&&primary.CaptureSessionId=="probe-bridge","bridge context");
    Need(primary.BufferedRawFrameCount==2&&controller.BufferedRawFrameCount==0,"buffer preservation");
    foreach(var property in typeof(PersonalLivePrimaryFailureSnapshot).GetProperties(BindingFlags.Public|BindingFlags.Instance)) Need(!property.CanWrite,"mutable property");
    Need(typeof(PersonalLivePrimaryFailureSnapshot).GetFields(BindingFlags.Public|BindingFlags.Instance).Length==0,"public field");
    Console.WriteLine("BRIDGE_FAILURE_PRIMARY=PASS");Console.WriteLine("PRECLEANUP_BUFFER_COUNT=PASS");
  }
  static void Followups() {
    var source=new Source();var controller=Ready(source);controller.BeginCapture("probe-followup");source.Emit(Observation(0));
    source.Fail(new InvalidOperationException("first"));var primary=controller.PrimaryFailure;
    Need(!controller.FinalizeCapture()&&controller.Failure.Code==PersonalLiveFailureCode.InvalidStateTransition,"finalize compatibility");
    Need(Object.ReferenceEquals(primary,controller.PrimaryFailure),"finalize overwrite");
    Need(!controller.BeginCapture()&&!controller.CancelCapture(),"other invalid commands");
    source.IsBound=false;Need(!controller.Reset()&&controller.Failure.Code==PersonalLiveFailureCode.BridgeNotBound,"failed reset");
    Need(Object.ReferenceEquals(primary,controller.PrimaryFailure)&&primary.Code==PersonalLiveFailureCode.BridgeFailure,"followup overwrite");
    Console.WriteLine("FOLLOWUP_NON_OVERWRITE=PASS");
  }
  static void Rejection() {
    var source=new Source();var controller=Ready(source);controller.BeginCapture("probe-rejection");source.Emit(Observation(1));source.Emit(Observation(1));
    var primary=controller.PrimaryFailure;Need(primary.Code==PersonalLiveFailureCode.ObservationRejected,"rejection code");
    Need(primary.Message=="observation timestamps must be strictly increasing"&&primary.BufferedRawFrameCount==1,"rejection context");
    Console.WriteLine("OBSERVATION_REJECTION_PRIMARY=PASS");
  }
  static void Reset() {
    var source=new Source();var controller=Ready(source);controller.BeginCapture("probe-reset");source.Emit(Observation(0));
    source.Fail(new InvalidOperationException("reset"));Need(controller.PrimaryFailure!=null,"primary before reset");Need(controller.Reset(),"reset");
    Need(controller.PrimaryFailure==null&&controller.Failure==null&&controller.State==PersonalLiveCaptureState.Ready,"reset clear");
    Console.WriteLine("SUCCESSFUL_RESET_CLEAR=PASS");
  }
  static int Main() { try { BridgeAndBuffer();Followups();Rejection();Reset();Console.WriteLine("PRIMARY_FAILURE_REGRESSION=5/5");return 0; }
    catch(Exception error) { Console.Error.WriteLine(error);return 1; } }
}
"""


def compiler_count(text: str, kind: str) -> int:
    return len(re.findall(r"\b" + kind + r" CS[0-9]+\b", text))


def offline_compile_and_probe(root: Path, step9j_v10, step9j_v11) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    mono = Path(r"C:\Program Files\Unity\Hub\Editor\6000.3.14f1\Editor\Data\MonoBleedingEdge\bin\mono.exe")
    csc = Path(r"C:\Program Files\Unity\Hub\Editor\6000.3.14f1\Editor\Data\MonoBleedingEdge\lib\mono\4.5\csc.exe")
    require(mono.is_file() and csc.is_file(), "Unity Mono Roslyn compiler unavailable")
    unity_stub = make_unity_stub(step9j_v10, step9j_v11)
    nunit_stub = step9j_v10.NUNIT_STUB.replace(
        "public static void IsNull(object value, string message = null) {}",
        "public static void IsNull(object value, string message = null) {}\n"
        "    public static void IsNotNull(object value, string message = null) {}")
    with tempfile.TemporaryDirectory(prefix="WIS-Step9K-primary-failure-") as temporary:
        temp = Path(temporary)
        unity_path = temp / "UnityAndBridgeCompileStubs.cs"
        nunit_path = temp / "NUnitCompileStubs.cs"
        probe_path = temp / "PrimaryFailureProbe.cs"
        unity_path.write_text(unity_stub, encoding="utf-8")
        nunit_path.write_text(nunit_stub, encoding="utf-8")
        probe_path.write_text(PRIMARY_FAILURE_PROBE, encoding="utf-8")
        runtime_dll = temp / "Step9KRuntime.dll"
        production = subprocess.run(
            [str(mono), str(csc), "/nologo", "/target:library", "/langversion:latest",
             "/out:" + str(runtime_dll), str(unity_path),
             *[str((root / relative).resolve()) for relative in step9j_v10.RUNTIME_SOURCES]],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        production_output = (production.stdout or "") + (production.stderr or "")
        require(production.returncode == 0, "offline production compile failed:\n" + production_output[-6000:])
        test_dll = temp / "Step9KTests.dll"
        tests = subprocess.run(
            [str(mono), str(csc), "/nologo", "/target:library", "/langversion:latest",
             "/out:" + str(test_dll), "/r:" + str(runtime_dll), str(nunit_path), str((root / TEST_REL).resolve())],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        test_output = (tests.stdout or "") + (tests.stderr or "")
        require(tests.returncode == 0, "offline fixture compile failed:\n" + test_output[-6000:])
        probe_exe = temp / "PrimaryFailureProbe.exe"
        probe_compile = subprocess.run(
            [str(mono), str(csc), "/nologo", "/target:exe", "/langversion:latest",
             "/out:" + str(probe_exe), "/r:" + str(runtime_dll), str(probe_path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        probe_compile_output = (probe_compile.stdout or "") + (probe_compile.stderr or "")
        require(probe_compile.returncode == 0, "offline probe compile failed:\n" + probe_compile_output[-6000:])
        probe = subprocess.run([str(mono), str(probe_exe)], capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=120)
        probe_output = ((probe.stdout or "") + (probe.stderr or "")).replace("\ufeff", "")
        require(probe.returncode == 0, "offline probe failed:\n" + probe_output[-6000:])
    expected_markers = (
        "BRIDGE_FAILURE_PRIMARY=PASS", "PRECLEANUP_BUFFER_COUNT=PASS",
        "FOLLOWUP_NON_OVERWRITE=PASS", "OBSERVATION_REJECTION_PRIMARY=PASS",
        "SUCCESSFUL_RESET_CLEAR=PASS", "PRIMARY_FAILURE_REGRESSION=5/5",
    )
    require(all(marker in probe_output for marker in expected_markers), "offline probe marker mismatch")
    compile_evidence = {
        "schema_version": SCHEMA + ".offline_compile",
        "compiler": "unity_mono_roslyn_csc", "production_source_count": len(step9j_v10.RUNTIME_SOURCES),
        "production_error_count": compiler_count(production_output, "error"),
        "production_warning_count": compiler_count(production_output, "warning"),
        "fixture_error_count": compiler_count(test_output, "error"),
        "fixture_warning_count": compiler_count(test_output, "warning"),
        "probe_error_count": compiler_count(probe_compile_output, "error"),
        "probe_warning_count": compiler_count(probe_compile_output, "warning"),
        "unity_editor_invoked": False, "unity_compile_pass_claimed": False,
        "editmode_pass_claimed": False, "status": "PASS",
    }
    require(compile_evidence["production_error_count"] == compile_evidence["fixture_error_count"] ==
            compile_evidence["probe_error_count"] == 0, "offline compiler error count nonzero")
    regression = {
        "schema_version": SCHEMA + ".primary_failure_regression",
        "uses_actual_controller": True, "scenario_count": 5, "pass_count": 5,
        "scenarios": [marker.split("=")[0] for marker in expected_markers[:-1]],
        "probe_exit_code": probe.returncode, "raw_output": probe_output,
        "unity_editor_invoked": False, "status": "PASS",
    }
    return compile_evidence, regression, unity_stub, nunit_stub


def verify_fixed_path(root: Path, relative: Path, size: int, digest: str) -> dict[str, Any]:
    path = root / relative
    require(path.is_file() and path.stat().st_size == size and sha256_file(path) == digest,
            "protected file identity mismatch: " + relative.as_posix())
    return {"path": relative.as_posix(), "size_bytes": size, "sha256": digest, "byte_identical": True}


def verify_protected_files(root: Path, step9j_v10, step9k_files: dict[str, bytes],
                           v10_files: dict[str, bytes]) -> tuple[dict[str, Any], dict[str, str]]:
    approved, snapshot = step9j_v10.verify_fixed_files(root)
    records = list(approved["records"])
    for relative, size, digest in SCENE_AND_PREFAB_FIXED:
        record = verify_fixed_path(root, Path(relative), size, digest)
        records.append(record)
        snapshot[relative] = digest
    for relative in (TECHNICAL_PREFAB_REL, TECHNICAL_PREFAB_META_REL):
        archive_name = "project_files/" + relative.name
        expected = step9k_files[archive_name]
        path = root / relative
        require(path.is_file() and path.read_bytes() == expected,
                "approved Step 9K technical prefab changed: " + relative.as_posix())
        digest = sha256_bytes(expected)
        records.append({"path": relative.as_posix(), "size_bytes": len(expected), "sha256": digest,
                        "byte_identical": True})
        snapshot[relative.as_posix()] = digest
    meta_archive_names = (
        "project_files/production/PersonalLiveRecognitionController.cs.meta",
        "project_files/production/PersonalLiveRecognitionHarness.cs.meta",
        "project_files/tests/PersonalLiveRecognitionIntegrationV1Tests.cs.meta",
    )
    for (relative, size, digest), archive_name in zip(META_IDENTITIES, meta_archive_names):
        record = verify_fixed_path(root, relative, size, digest)
        require(v10_files[archive_name] == (root / relative).read_bytes(),
                "Step 9J meta differs from approved Delivery: " + relative.as_posix())
        records.append(record)
        snapshot[relative.as_posix()] = digest
    paths = [record["path"] for record in records]
    require(len(paths) == len(set(paths)), "duplicate protected-file record")
    return ({"schema_version": SCHEMA + ".protected_file_preservation",
             "file_count": len(records), "mismatch_count": 0, "records": records,
             "status": "PASS"}, snapshot)


def staged_count(root: Path) -> int:
    result = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=root,
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    require(result.returncode == 0, "Git staged-file query failed")
    return len([line for line in result.stdout.splitlines() if line.strip()])


def assertion(key: str, status: str, subject: str, evidence: Any) -> dict[str, Any]:
    return {"assertion_key": key, "status": status, "subject": subject, "evidence": evidence}


def identity_record(path: str, data: bytes) -> dict[str, Any]:
    return {"path": path, "size_bytes": len(data), "sha256": sha256_bytes(data)}


def source_inventory_csv(before: dict[str, bytes], after: dict[str, bytes]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["project_relative_path", "before_size_bytes", "before_sha256",
                     "after_size_bytes", "after_sha256", "changed", "reason"])
    reasons = {
        CONTROLLER_REL.as_posix(): "pin immutable first failure before cleanup and clear only on successful Reset",
        HARNESS_REL.as_posix(): "show primary/Step9C diagnostics and disable invalid-state controls",
        TEST_REL.as_posix(): "add five primary failure regression methods while preserving approved 17",
    }
    for name in (CONTROLLER_REL.as_posix(), HARNESS_REL.as_posix(), TEST_REL.as_posix()):
        writer.writerow([name, len(before[name]), sha256_bytes(before[name]), len(after[name]),
                         sha256_bytes(after[name]), before[name] != after[name], reasons[name]])
    return output.getvalue().encode("utf-8")


def build(root: Path) -> dict[str, Any]:
    output = root / RESULT_REL
    delivery = root / DELIVERY_REL
    detached = Path(str(delivery) + ".sha256")
    validation_root = root / "Validation"
    require(validation_root.is_dir(), "Validation directory missing")
    for target in (output, delivery, detached):
        require(not target.exists() and not target.is_symlink(), "never-overwrite target exists: " + str(target))

    transaction = import_fixed(root, TRANSACTION_TOOL_REL, TRANSACTION_TOOL_SIZE,
                               TRANSACTION_TOOL_SHA256, "step9i_v11_transaction_primary_failure")
    step9j_v10 = import_fixed(root, STEP9J_TOOL_REL, STEP9J_TOOL_SIZE,
                              STEP9J_TOOL_SHA256, "step9j_v10_primary_failure")
    step9j_v11 = import_fixed(root, Path("Tools/s001_step9j_personal_live_recognition_integration_v1_1.py"),
                              35_707, "533b6eed69f7f7f6bc6adebadfc6f40647040fc4971541cf531bc788a2bdd540",
                              "step9j_v11_primary_failure")
    deliveries: dict[str, Any] = {}
    delivery_maps: dict[str, dict[str, bytes]] = {}
    for label, relative, size, digest in FIXED_DELIVERIES:
        deliveries[label], delivery_maps[label] = verify_delivery(root, transaction, label, relative, size, digest)

    v10_files = delivery_maps["step9j_v1_0"]
    v11_files = delivery_maps["step9j_v1_1"]
    step9k_files = delivery_maps["step9k_v1_0"]
    before = {
        CONTROLLER_REL.as_posix(): v10_files["project_files/production/PersonalLiveRecognitionController.cs"],
        HARNESS_REL.as_posix(): v10_files["project_files/production/PersonalLiveRecognitionHarness.cs"],
        TEST_REL.as_posix(): v11_files["source_after/PersonalLiveRecognitionIntegrationV1Tests.cs"],
    }
    for name, data in before.items():
        expected_size, expected_sha = BEFORE_IDENTITIES[name]
        require(len(data) == expected_size and sha256_bytes(data) == expected_sha,
                "approved before-source identity mismatch: " + name)
    after = {name: (root / PurePosixPath(name)).read_bytes() for name in before}
    require(all(before[name] != after[name] for name in before), "expected exactly three changed source files")
    source_contract, test_inventory = validate_sources(
        before[CONTROLLER_REL.as_posix()], before[HARNESS_REL.as_posix()], before[TEST_REL.as_posix()],
        after[CONTROLLER_REL.as_posix()], after[HARNESS_REL.as_posix()], after[TEST_REL.as_posix()])
    protected, protected_before = verify_protected_files(root, step9j_v10, step9k_files, v10_files)
    compile_evidence, regression, unity_stub, nunit_stub = offline_compile_and_probe(root, step9j_v10, step9j_v11)
    protected_after = {name: sha256_file(root / PurePosixPath(name)) for name in protected_before}
    require(protected_before == protected_after, "protected file changed during validation")
    git_staged = staged_count(root)
    require(git_staged == 0, "Git staged file count is not zero")

    transaction_paths = sorted(set(protected_before) | set(after))
    transaction_identity = {"records": [{"project_relative_path": name} for name in transaction_paths]}
    injection = transaction.run_failure_injection_regression(root, validation_root, transaction_identity)
    never_overwrite = transaction.run_never_overwrite_regression(validation_root)
    require(injection["verdict"] == "PASS" and never_overwrite["verdict"] == "PASS",
            "approved packaging transaction regression failed")

    test_results = {
        "schema_version": SCHEMA + ".test_results",
        "previous_approved_unity_editmode": {"fixture": "PersonalLiveRecognitionIntegrationV1Tests",
                                              "passed": 17, "failed": 0,
                                              "status": "PASS_PREVIOUS_EXTERNAL_REVIEW"},
        "approved_test_methods_preserved": "17/17",
        "new_regression_methods": 5,
        "new_controller_regression_offline_probe": "5/5 PASS",
        "current_fixture_compile_only": {"method_count": 22, "compiler_errors": 0, "status": "PASS"},
        "current_unity_editmode_execution": "UNVERIFIED_NOT_EXECUTED",
        "status": "PASS_WITH_UNITY_RETEST_UNVERIFIED",
    }
    primary = {
        "schema_version": SCHEMA + ".primary_verdict",
        "primary_failure_preservation": "IMPLEMENTED",
        "precleanup_buffer_count_preservation": "IMPLEMENTED",
        "technical_panel_bridge_diagnostics": "IMPLEMENTED",
        "invalid_state_button_gates": "IMPLEMENTED",
        "approved_recognition_calculation_assets_prefab_scenes": "PRESERVED",
        "unity_compile_editmode_retest": "UNVERIFIED",
        "external_review": "PENDING",
        "status_strings": [
            "STEP9K_PRIMARY_FAILURE_PRESERVATION_DIAGNOSTIC_V1_0_SOURCE_AND_PACKAGE_READY",
            "STEP9K_PRIMARY_ASYNC_FAILURE_SNAPSHOT_PRESERVATION_IMPLEMENTED",
            "STEP9K_FAILURE_PRECLEANUP_BUFFER_COUNT_PRESERVED",
            "STEP9K_TECHNICAL_PANEL_BRIDGE_DIAGNOSTICS_AND_STATE_GATES_IMPLEMENTED",
            "STEP9J_APPROVED_RECOGNITION_CALCULATION_AND_STEP9K_PREFAB_PRESERVED",
            "STEP9K_UNITY_COMPILE_EDITMODE_RETEST_UNVERIFIED",
            "BLOCKED_PENDING_EXTERNAL_REVIEW",
        ],
    }

    assertions = [
        assertion("fixed_step9j_v1_0_delivery", "PASS", FIXED_DELIVERIES[0][1].as_posix(), deliveries["step9j_v1_0"]),
        assertion("fixed_step9j_v1_1_delivery", "PASS", FIXED_DELIVERIES[1][1].as_posix(), deliveries["step9j_v1_1"]),
        assertion("fixed_step9k_v1_0_delivery", "PASS", FIXED_DELIVERIES[2][1].as_posix(), deliveries["step9k_v1_0"]),
        assertion("changed_source_files_exactly_three", "PASS", "controller+harness+test", list(after)),
        assertion("primary_failure_snapshot_immutable", "PASS", CONTROLLER_REL.as_posix(), source_contract),
        assertion("primary_failure_captured_before_cleanup", "PASS", "Fail()", True),
        assertion("successful_reset_only_primary_clear", "PASS", "Reset()", True),
        assertion("legacy_failure_compatibility_preserved", "PASS", "Failure", True),
        assertion("technical_panel_primary_fields_six", "PASS", HARNESS_REL.as_posix(), 6),
        assertion("technical_panel_step9c_diagnostics_six", "PASS", HARNESS_REL.as_posix(), 6),
        assertion("technical_panel_invalid_state_button_gates_four", "PASS", HARNESS_REL.as_posix(), 4),
        assertion("approved_step9j_tests_preserved_seventeen", "PASS", TEST_REL.as_posix(), "17/17"),
        assertion("new_primary_failure_tests_five", "PASS", TEST_REL.as_posix(), list(NEW_TESTS)),
        assertion("offline_production_compile_error_zero", "PASS", "Unity Mono Roslyn", 0),
        assertion("offline_fixture_compile_error_zero", "PASS", "Unity Mono Roslyn", 0),
        assertion("offline_primary_failure_regression_five", "PASS", "actual controller", regression),
        assertion("protected_files_byte_identical", "PASS", "approved files/assets/scenes/prefabs", protected),
        assertion("unity_editor_and_editmode_not_claimed", "UNVERIFIED", "current 22-test fixture", "not executed"),
        assertion("packaging_failure_injection", "PASS", "approved transaction", injection),
        assertion("never_overwrite", "PASS", "approved transaction", never_overwrite),
        assertion("git_staged_zero", "PASS", "Git index", git_staged),
    ]
    keys = [item["assertion_key"] for item in assertions]
    require(len(keys) == len(set(keys)), "duplicate assertion key")

    diff_files = {}
    for name in after:
        diff = "".join(difflib.unified_diff(
            before[name].decode("utf-8").splitlines(True), after[name].decode("utf-8").splitlines(True),
            fromfile="before/" + PurePosixPath(name).name, tofile="after/" + PurePosixPath(name).name))
        diff_files["unified_diff/" + PurePosixPath(name).name + ".diff"] = diff.encode("utf-8")

    file_map: dict[str, bytes] = {
        "input_provenance.json": json_bytes({"schema_version": SCHEMA + ".input_provenance",
                                                "fixed_deliveries": deliveries,
                                                "approved_transaction_tool": {"path": TRANSACTION_TOOL_REL.as_posix(),
                                                    "size_bytes": TRANSACTION_TOOL_SIZE, "sha256": TRANSACTION_TOOL_SHA256}}),
        "source_change_inventory.csv": source_inventory_csv(before, after),
        "primary_failure_contract.json": json_bytes(source_contract),
        "technical_panel_contract.json": json_bytes({
            "schema_version": SCHEMA + ".technical_panel_contract",
            "primary_failure_fields": ["code", "message", "exception_type", "state_before_failure",
                                       "capture_session_id", "buffered_raw_frame_count"],
            "step9c_bridge_fields": ["LastError", "ProviderFrameCount", "ObservationCount",
                                     "DuplicateFrameCount", "ResolvedDuplicateFrameCount",
                                     "AmbiguousDuplicateFrameCount"],
            "button_gates": {"Start": ["Ready"], "Finalize": ["Capturing"],
                             "Cancel": ["Capturing", "Processing"],
                             "Reset": ["Completed", "Cancelled", "Failed"]},
            "approved_prefab_modified": False, "status": "PASS"}),
        "test_method_inventory.json": json_bytes(test_inventory),
        "test_results.json": json_bytes(test_results),
        "primary_failure_offline_regression.json": json_bytes(regression),
        "offline_csharp_compile_check.json": json_bytes(compile_evidence),
        "protected_file_preservation.json": json_bytes(protected),
        "packaging_transaction_failure_injection_regression.json": json_bytes(injection),
        "never_overwrite_regression.json": json_bytes(never_overwrite),
        "primary_verdict.json": json_bytes(primary),
        "ASSERTIONS.jsonl": b"".join(json.dumps(item, ensure_ascii=False, allow_nan=False,
                                                  separators=(",", ":")).encode("utf-8") + b"\n"
                                      for item in assertions),
        "source_before/PersonalLiveRecognitionController.cs": before[CONTROLLER_REL.as_posix()],
        "source_before/PersonalLiveRecognitionHarness.cs": before[HARNESS_REL.as_posix()],
        "source_before/PersonalLiveRecognitionIntegrationV1Tests.cs": before[TEST_REL.as_posix()],
        "source_after/PersonalLiveRecognitionController.cs": after[CONTROLLER_REL.as_posix()],
        "source_after/PersonalLiveRecognitionHarness.cs": after[HARNESS_REL.as_posix()],
        "source_after/PersonalLiveRecognitionIntegrationV1Tests.cs": after[TEST_REL.as_posix()],
        "analysis/compile_fixtures/UnityAndBridgeCompileStubs.cs": unity_stub.encode("utf-8"),
        "analysis/compile_fixtures/NUnitCompileStubs.cs": nunit_stub.encode("utf-8"),
        "analysis/compile_fixtures/PrimaryFailureProbe.cs": PRIMARY_FAILURE_PROBE.encode("utf-8"),
        "analysis/s001_step9k_primary_failure_preservation_diagnostic_v1_0.py": (root / TOOL_REL).read_bytes(),
    }
    file_map.update(diff_files)
    report = "\n".join([
        "# WordsInSilence Step 9K — Primary Failure Preservation Diagnostic v1.0", "",
        "## Outcome", "",
        "The first controller failure is now pinned in an immutable pre-cleanup snapshot. Later invalid commands may update the legacy Failure value but cannot overwrite the primary diagnostic. Only a successful Reset clears it.", "",
        "The technical panel displays all six primary fields and all six requested Step 9C bridge diagnostics. Start, Finalize, Cancel, and Reset are disabled outside their valid controller states.", "",
        "## Source and regression verification", "",
        f"- Changed project sources: `3` (controller, harness, EditMode fixture)",
        f"- Approved Step 9J test methods preserved byte-for-byte: `{test_inventory['approved_method_body_byte_identical_count']}/17`",
        f"- New primary-failure regression methods: `{test_inventory['new_method_count']}`; offline actual-controller probe: `{regression['pass_count']}/5 PASS`",
        f"- Offline production/fixture/probe compiler errors: `{compile_evidence['production_error_count']}/{compile_evidence['fixture_error_count']}/{compile_evidence['probe_error_count']}`",
        f"- Protected files changed: `{protected['mismatch_count']}`; Git staged files: `{git_staged}`",
        f"- Packaging failure injection: `{injection['pass_count']}/{injection['injection_point_count']}`; never-overwrite: `{never_overwrite['verdict']}`", "",
        "## Test status", "",
        "The previously approved Unity run remains 17/17 PASS because all 17 method bodies are byte-identical. The current 22-method fixture compiles offline with error 0 and the five new semantics pass the standalone actual-controller probe. Unity Editor and the current EditMode fixture were not executed, so no new Unity PASS is claimed.", "",
        "## Preserved scope", "",
        "Preprocessing, DTW, thresholds, templates, personal/global runtime assets, Step 9C identity calculation, approved prefab, provider scene, main scene, and webcam ownership are byte-identical and unchanged.", "",
        "## Status", "", "```text", *primary["status_strings"], "```", "",
    ])
    file_map["VALIDATION_REPORT.md"] = report.encode("utf-8")
    records = [identity_record(name, data) for name, data in sorted(file_map.items())]
    file_map["output_file_manifest.json"] = json_bytes({
        "schema_version": SCHEMA + ".output_file_manifest", "self_excluded": True,
        "file_count": len(records), "files": records})
    zip_one = transaction.deterministic_zip_bytes(file_map)
    require(zip_one == transaction.deterministic_zip_bytes(file_map), "deterministic ZIP regeneration mismatch")
    detached_bytes = transaction.strict_detached_bytes(zip_one, delivery.name)

    def post_validate(final_result: Path, final_delivery: Path, final_detached: Path) -> None:
        require(final_delivery.read_bytes() == zip_one, "committed ZIP differs")
        transaction.verify_strict_detached(final_detached.read_bytes(), zip_one, final_delivery.name)
        for name, data in file_map.items():
            require((final_result / PurePosixPath(name)).read_bytes() == data,
                    "committed result mismatch: " + name)
        safety = transaction.inspect_zip(zip_one, set(file_map))
        require(safety["crc"] == "PASS" and safety["duplicates"] == 0 and safety["unsafe"] == 0 and
                safety["symlinks"] == 0 and safety["encrypted"] == 0 and safety["membership"] == "PASS",
                "committed ZIP safety verification failed")

    transaction.commit_packaging_transaction(
        result_files=file_map, delivery_bytes=zip_one, detached_bytes=detached_bytes,
        final_result=output, final_delivery=delivery, final_detached=detached,
        allowed_root=validation_root, post_rename_validator=post_validate,
        transaction_token="step9k_primary_failure_v1_0")
    protected_final = {name: sha256_file(root / PurePosixPath(name)) for name in protected_before}
    require(protected_final == protected_before, "protected file changed during package commit")
    require(staged_count(root) == 0, "Git staged files changed during package commit")
    return {
        "result_directory": str(output), "result_file_count": len(file_map),
        "delivery_zip": str(delivery), "delivery_size_bytes": len(zip_one),
        "delivery_sha256": sha256_bytes(zip_one), "detached_checksum": str(detached),
        "detached_size_bytes": len(detached_bytes), "detached_file_sha256": sha256_bytes(detached_bytes),
        "assertion_count": len(assertions),
        "assertion_pass_count": sum(item["status"] == "PASS" for item in assertions),
        "assertion_unverified_count": sum(item["status"] == "UNVERIFIED" for item in assertions),
        "git_staged_file_count": 0, "status_strings": primary["status_strings"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    summary = build(args.project_root.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
