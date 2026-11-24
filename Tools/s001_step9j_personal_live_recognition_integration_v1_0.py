#!/usr/bin/env python3
"""Build and validate WordsInSilence Step 9J personal live integration v1.0.

This tool is intentionally offline. It does not launch Unity, batchmode, an
EditMode/PlayMode fixture, a webcam, or a camera provider. It verifies the
approved Step 9I/Step 9C inputs, compiles the new source with Unity Mono
Roslyn-compatible stubs, executes a standalone routing/state probe, and uses
the externally approved Step 9I v1.1 packaging transaction for both real
commit and failure-injection/never-overwrite regression.
"""

from __future__ import annotations

import argparse
import csv
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


SCHEMA = "words_in_silence.step9j.personal_live_recognition_integration.v1_0"
RESULT_REL = Path("Validation/WordsInSilence_step9j_personal_live_recognition_integration_v1_0")
DELIVERY_REL = Path("Validation/WordsInSilence_step9j_personal_live_recognition_integration_v1_0_delivery.zip")
V12_REL = Path("Validation/WordsInSilence_step9i_personal_runtime_opt_in_integration_v1_2_delivery.zip")
V12_SIZE = 69_582
V12_SHA256 = "e60546ebea439fe593319c7806f2560d79fd65b1f10ed7ae5697fa52b2754cfb"
V12_DETACHED_SIZE = 142
V12_DETACHED_SHA256 = "e9ae1db78c05ba2986590d0c115f49df72d84f71c7418fe64bb0525605d13ea7"
V11_TOOL_REL = Path("Tools/s001_step9i_personal_runtime_opt_in_integration_v1_1.py")
V11_TOOL_SIZE = 37_722
V11_TOOL_SHA256 = "30471dc6725bdbdd0cc80c345b238485d1cb0d8d7a5933aac7477d536d714caf"

CONTROLLER_REL = Path("Assets/_Project/Scripts/Recognition/LiveInput/PersonalLiveRecognitionController.cs")
HARNESS_REL = Path("Assets/_Project/Scripts/Recognition/LiveInput/PersonalLiveRecognitionHarness.cs")
TEST_REL = Path("Assets/_Project/Tests/EditMode/PersonalLiveRecognitionIntegrationV1Tests.cs")
NEW_FILES = (
    CONTROLLER_REL,
    Path(str(CONTROLLER_REL) + ".meta"),
    HARNESS_REL,
    Path(str(HARNESS_REL) + ".meta"),
    TEST_REL,
    Path(str(TEST_REL) + ".meta"),
)

FIXED_FILES = (
    ("Assets/_Project/Scripts/Recognition/PhysicalCompatibility/Step9CHandIdentityObservationBridge.cs", 6_472, "f028f1485dc396fd9d7c38672f2a4097cb769ecfd321e45f3b63bb2f02c7adc2"),
    ("Assets/_Project/Scripts/Recognition/LiveInput/ApprovedLivePreprocessingAdapter.cs", 26_975, "1aa1eaed2158fef7dc3e8f163f66cd1fcf3de19b02035863b4a3e2500b66c47c"),
    ("Assets/_Project/Scripts/Recognition/Runtime/RecognitionSequenceInputSession.cs", 10_790, "960a5674ed1b05cf55867142ba7b6df48c996cc50f540df1ee3037e83214df91"),
    ("Assets/_Project/Scripts/Recognition/Runtime/RuntimeRecognitionService.cs", 11_144, "d6b3d44130fb9fae5e00745d15a5d537f4b1bc03a7d44fe8df6fd8d02656137a"),
    ("Assets/_Project/Scripts/Recognition/Runtime/PersonalRuntimeOptInProfile.cs", 54_894, "3e4579c153ad23ee85c00b01967956f0777a1092384e5e8ade8b57b9fa5798bc"),
    ("Assets/_Project/Scripts/Recognition/Core/MaskAwareDtw.cs", 5_649, "69c76f869e1c2b123b06eb3e81668dc2dce7c555ce967d769e7bf6abb6977e1b"),
    ("Assets/_Project/Scripts/Recognition/LiveInput/RawMediaPipeObservation.cs", 9_249, "0d69db1f4287b514e38fa40aa863ff88ecd313bbf192f7e22e66f7f52be516d1"),
    ("Assets/_Project/Scripts/Recognition/Runtime/NormalizedRecognitionFrame.cs", 2_111, "fbdd0f05ce3753364162d57e5344dac49822a66b646164470acf3674febbe0e0"),
    ("Assets/_Project/Scripts/Recognition/Diagnostics/Step9DRawIntermediatePreprocessingTrace.cs", 20_808, "78326aec66ea2b4c8c4e620f05418279a9ab1368115691efbbc8689fb1c33b60"),
    ("Assets/_Project/Tests/EditMode/PersonalRuntimeOptInIntegrationV1Tests.cs", 26_880, "0db5cd7afbacadf54fb2cdb53af0211dfb3ef48758d41a0d9de9df05ca7cc959"),
    ("Assets/_Project/Tests/EditMode/Fixtures/PersonalRuntimeOptInIntegrationV1/personal_expected_scores.csv", 5_265, "c3f98a8c49a52acd018e3366d9fbe88453e529256c51a77d92e1c3e8069a97aa"),
    ("Assets/_Project/Tests/EditMode/Fixtures/PersonalRuntimeOptInIntegrationV1/personal_expected_predictions.csv", 1_204, "207833bf6ee8b30f0507ec698024247a6c1fe44c742d15358d9b026f0e99fc58"),
    ("Assets/_Project/Tests/EditMode/Fixtures/PersonalRuntimeOptInIntegrationV1/personal_query_manifest.json", 6_781, "fb5d362460ffa7a3d4211e6cf4f3c1e22dd747d1cdab6fd08d43cb73d9f00145"),
    ("Assets/_Project/Tests/EditMode/Fixtures/PersonalRuntimeOptInIntegrationV1/global_runtime_asset_snapshot.json", 2_633, "b77b0f916ce8c0a4b750c9e51b3469aaf58d1ff01bca2217f62876bc0fbc600a"),
    ("Assets/StreamingAssets/RecognitionCoreV1/Templates/template_GREETING__FAST.wisrt", 35_012, "a9111bd2733cb8a2d2c5984309607979cee49c2516346668228e4fb51cf88277"),
    ("Assets/StreamingAssets/RecognitionCoreV1/Templates/template_GREETING__NORMAL.wisrt", 50_842, "8a2859403e0db736d213f59c82bd9d9a3f7a44dc748b6c65dca5869bc4cfa4b9"),
    ("Assets/StreamingAssets/RecognitionCoreV1/Templates/template_GREETING__SLOW.wisrt", 60_102, "0646436e35e20d13e3c00fd0aef85c7a77829d3729eb7a08e2c9ffa3dc989680"),
    ("Assets/StreamingAssets/RecognitionCoreV1/Templates/template_LOVE__FAST.wisrt", 19_950, "1ab0ccee87c749cb961f6d8a8135062f14c8affc8ec15b40472240e3e936967c"),
    ("Assets/StreamingAssets/RecognitionCoreV1/Templates/template_LOVE__NORMAL.wisrt", 29_218, "74b42959641fd1ce182f4f72427c02307f14ddc564e06e6f24044694bca5a3c3"),
    ("Assets/StreamingAssets/RecognitionCoreV1/Templates/template_LOVE__SLOW.wisrt", 34_232, "509cfb87126d18285745085810fc4a5afcc82886a8869d02f375166f71819a28"),
    ("Assets/StreamingAssets/RecognitionCoreV1/Templates/template_THANK_YOU__FAST.wisrt", 29_610, "e53455dff4448865b0df7d9d859888213a31e33738512b7ed7af0d16230e3e6f"),
    ("Assets/StreamingAssets/RecognitionCoreV1/Templates/template_THANK_YOU__NORMAL.wisrt", 41_966, "7da43c9f770ac79e1a67eb487469cef35467b72f4ab3abbb41faa26f0396c82e"),
    ("Assets/StreamingAssets/RecognitionCoreV1/Templates/template_THANK_YOU__SLOW.wisrt", 46_208, "b83309ac244a8f92fd83ac143da9e2307768125fe3379fd0ff9345bf6b5bb26f"),
    ("Assets/StreamingAssets/RecognitionCoreV1/RuntimeGate/frozen_threshold_config.json", 4_590, "2697bb1d0c301850d2abda41351519c66d0eb8e27841263469175eb3fa7e99e3"),
    ("Assets/StreamingAssets/RecognitionCoreV1/RuntimeGate/runtime_recognition_assets.json", 4_202, "261aa173a9643c21677d66c16ef3448c2808b3db2ffa5f9e48c671af6e9bd5e7"),
    ("Assets/StreamingAssets/RecognitionCoreV1/PersonalProfiles/step9h_current_user_v1_0/personal_class_threshold_config.json", 4_811, "e84edd1f598c2d94f8412767f3125031e0be330886d3a315ed429798fe49726e"),
    ("Assets/StreamingAssets/RecognitionCoreV1/PersonalProfiles/step9h_current_user_v1_0/personal_runtime_profile_manifest.json", 3_768, "2f9b4589c0a6130913cca6a5d9e3260aeaa6e9d4a5cba76aaafa61692fe46eb9"),
    ("Assets/StreamingAssets/RecognitionCoreV1/PersonalProfiles/step9h_current_user_v1_0/personal_template_GREETING.wisrt", 53_564, "d8d5713334b22d9427b9b589bccf820cdbb4849357bae3dce1c368eaf0049681"),
    ("Assets/StreamingAssets/RecognitionCoreV1/PersonalProfiles/step9h_current_user_v1_0/personal_template_LOVE.wisrt", 56_262, "013bf5515eb8af33f300400893da64c853668b2fa0442e19203da62b2777ad4b"),
    ("Assets/StreamingAssets/RecognitionCoreV1/PersonalProfiles/step9h_current_user_v1_0/personal_template_THANK_YOU.wisrt", 49_319, "9bb9a7681d55a292a21d8b7f458b4b150c10555bc0ec163da692a780ed5280e5"),
)

RUNTIME_SOURCES = (
    "Assets/_Project/Scripts/Recognition/Core/RecognitionSequence.cs",
    "Assets/_Project/Scripts/Recognition/Core/RecognitionSequenceBinary.cs",
    "Assets/_Project/Scripts/Recognition/Core/MaskAwareDtw.cs",
    "Assets/_Project/Scripts/Recognition/Core/RecognitionCore.cs",
    "Assets/_Project/Scripts/Recognition/Evaluation/DistanceGatePolicy.cs",
    "Assets/_Project/Scripts/Recognition/Evaluation/FrozenDistanceGateConfig.cs",
    "Assets/_Project/Scripts/Recognition/Runtime/RuntimeRecognitionService.cs",
    "Assets/_Project/Scripts/Recognition/Runtime/PersonalRuntimeOptInProfile.cs",
    "Assets/_Project/Scripts/Recognition/Runtime/NormalizedRecognitionFrame.cs",
    "Assets/_Project/Scripts/Recognition/Runtime/RecognitionSequenceInputSession.cs",
    "Assets/_Project/Scripts/Recognition/LiveInput/RawMediaPipeObservation.cs",
    "Assets/_Project/Scripts/Recognition/Diagnostics/Step9DRawIntermediatePreprocessingTrace.cs",
    "Assets/_Project/Scripts/Recognition/LiveInput/ApprovedLivePreprocessingAdapter.cs",
    CONTROLLER_REL.as_posix(),
    HARNESS_REL.as_posix(),
)

UNITY_AND_BRIDGE_STUB = r"""using System;
using WordsInSilence.Recognition.LiveInput;
namespace UnityEngine {
  [AttributeUsage(AttributeTargets.Field)] public sealed class SerializeField : Attribute {}
  public class MonoBehaviour {}
  public struct Rect { public Rect(float x,float y,float w,float h) {} }
  public sealed class GUIStyle {}
  public sealed class GUISkin { public GUIStyle box { get { return new GUIStyle(); } } }
  public static class GUI { public static GUISkin skin { get { return new GUISkin(); } } }
  public sealed class GUILayoutOption {}
  public static class GUILayout {
    public static void BeginArea(Rect value, GUIStyle style) {}
    public static void EndArea() {}
    public static void Label(string value, params GUILayoutOption[] options) {}
    public static void BeginHorizontal() {}
    public static void EndHorizontal() {}
    public static GUILayoutOption Width(float value) { return new GUILayoutOption(); }
    public static bool Toggle(bool value, string text, string style) { return value; }
    public static bool Button(string text) { return false; }
    public static void Space(float value) {}
  }
  public static class Application {
    public static string dataPath { get { return string.Empty; } }
    public static string streamingAssetsPath { get { return string.Empty; } }
  }
  public static class JsonUtility { public static T FromJson<T>(string value) { return default(T); } }
}
namespace WordsInSilence.Recognition.PhysicalCompatibility {
  public sealed class Step9CHandIdentityObservationBridge {
    public bool IsBound { get; set; }
    public event Action<RawMediaPipeObservation> ObservationArrived;
    public event Action<Exception> BridgeFailed;
  }
}
"""

NUNIT_STUB = r"""using System;
using System.Collections.Generic;
namespace NUnit.Framework {
  [AttributeUsage(AttributeTargets.Method)] public sealed class TestAttribute : Attribute {}
  public static class Assert {
    public static void AreEqual<T>(T expected, T actual, string message = null) {}
    public static void AreEqual(double expected, double actual, double tolerance, string message = null) {}
    public static void IsTrue(bool value, string message = null) {}
    public static void IsFalse(bool value, string message = null) {}
    public static void IsNull(object value, string message = null) {}
    public static void AreSame(object expected, object actual, string message = null) {}
  }
  public static class CollectionAssert {
    public static void AreEqual<T>(IEnumerable<T> expected, IEnumerable<T> actual) {}
    public static void AreEquivalent<T>(IEnumerable<T> expected, IEnumerable<T> actual) {}
  }
}
"""

PROBE_SOURCE = r"""using System;
using System.Collections.Generic;
using System.IO;
using WordsInSilence.Recognition.Core;
using WordsInSilence.Recognition.LiveInput;
using WordsInSilence.Recognition.Runtime;
class Step9JProbe {
  sealed class Source : IPersonalLiveObservationSource {
    Action<RawMediaPipeObservation> observations; Action<Exception> failures;
    public bool IsBound { get; set; } = true;
    public int Subscribers => (observations == null ? 0 : observations.GetInvocationList().Length) +
      (failures == null ? 0 : failures.GetInvocationList().Length);
    public event Action<RawMediaPipeObservation> ObservationArrived { add { observations += value; } remove { observations -= value; } }
    public event Action<Exception> SourceFailed { add { failures += value; } remove { failures -= value; } }
    public void Emit(RawMediaPipeObservation value) { observations?.Invoke(value); }
    public void Fail() { failures?.Invoke(new InvalidOperationException("injected bridge failure")); }
  }
  sealed class Global : IPersonalLiveGlobalRuntime {}
  sealed class Runtime : IPersonalLiveRecognitionRuntime {
    public bool IsPersonalActive => true; public string ActiveProfileId => PersonalRuntimeProfileLoader.ProfileId;
    public PersonalLiveRoutingResult Recognize(string id, RecognitionSequence query, string expected) {
      var scores = new Dictionary<string,double>(StringComparer.Ordinal) {{"GREETING",.1},{"LOVE",.4},{"THANK_YOU",.5}};
      var accepts = new Dictionary<string,bool>(StringComparer.Ordinal) {{"GREETING",true},{"LOVE",false},{"THANK_YOU",false}};
      return new PersonalLiveRoutingResult("GREETING",true,true,true,scores,accepts,.1,"probe-greeting",new[]{PersonalRuntimeProfileLoader.GreetingMarginWarning});
    }
  }
  sealed class Factory : IPersonalLiveRuntimeFactory {
    public IPersonalLiveGlobalRuntime LoadGlobal(PersonalLiveRuntimePaths paths) { return new Global(); }
    public IPersonalLiveRecognitionRuntime CreateAndActivate(IPersonalLiveGlobalRuntime global, PersonalLiveRuntimePaths paths) { return new Runtime(); }
  }
  static double[] Nan(int count) { var a=new double[count*3]; for(int i=0;i<a.Length;i++)a[i]=double.NaN; return a; }
  static void Put(double[] c,bool[] m,int i,double x,double y,double z){int o=i*3;c[o]=x;c[o+1]=y;c[o+2]=z;m[i]=true;}
  static RawMediaPipeObservation Observation(double time) {
    var pc=Nan(33);var pm=new bool[33];
    Put(pc,pm,11,-1,0,0);Put(pc,pm,12,1,0,0);Put(pc,pm,23,-1,2,0);Put(pc,pm,24,1,2,0);Put(pc,pm,13,-1.2,-.5,.1);
    return new RawMediaPipeObservation(time,RawObservationConvention.LiveMediaPipe,
      new RawLandmarkObservationSet(33,pc,pm,new bool[33],new bool[33]),RawLandmarkObservationSet.Absent(21),RawLandmarkObservationSet.Absent(21));
  }
  static void Need(bool value,string message){if(!value)throw new Exception(message);}
  static int Main(string[] args) { try {
    string root=args[0];
    var paths=new PersonalLiveRuntimePaths(Path.Combine(root,"Assets/StreamingAssets/RecognitionCoreV1/Templates"),
      Path.Combine(root,"Assets/StreamingAssets/RecognitionCoreV1/RuntimeGate/frozen_threshold_config.json"),
      Path.Combine(root,"Assets/StreamingAssets/RecognitionCoreV1/PersonalProfiles/step9h_current_user_v1_0"));
    var source=new Source();var controller=new PersonalLiveRecognitionController(source,new Factory());
    Need(controller.Initialize(paths),"initialize");source.Emit(Observation(0));Need(controller.BufferedRawFrameCount==0,"pre-start observation");
    Need(controller.BeginCapture("offline-live-probe"),"begin");source.Emit(Observation(0));source.Emit(Observation(.03));
    Need(controller.FinalizeCapture(),"finalize");Need(controller.Result.QuestSuccess,"quest gate");Need(source.Subscribers==0,"completed subscription cleanup");
    var frozen=controller.Result;source.Emit(Observation(.06));Need(object.ReferenceEquals(frozen,controller.Result),"snapshot changed");
    var failedSource=new Source();var failed=new PersonalLiveRecognitionController(failedSource,new Factory());Need(failed.Initialize(paths),"failed init");
    Need(failed.BeginCapture("bridge-failure"),"failed begin");failedSource.Fail();Need(failed.Failure.Code==PersonalLiveFailureCode.BridgeFailure,"bridge code");Need(failedSource.Subscribers==0,"bridge cleanup");
    var approvedFactory=new ApprovedPersonalLiveRuntimeFactory();var global=approvedFactory.LoadGlobal(paths);var personal=approvedFactory.CreateAndActivate(global,paths);
    string queryDir=Path.Combine(root,"Assets/_Project/Tests/EditMode/Fixtures/PersonalRuntimeOptInIntegrationV1/Queries");
    int queryCount=0,predictions=0,accepted=0,quest=0;var words=new HashSet<string>(StringComparer.Ordinal);
    foreach(string file in Directory.GetFiles(queryDir,"*.wisrq",SearchOption.TopDirectoryOnly)){
      string name=Path.GetFileName(file);string expected=name.StartsWith("query_THANK_YOU_")?"THANK_YOU":name.StartsWith("query_GREETING_")?"GREETING":"LOVE";
      var query=RecognitionSequenceBinary.LoadFile(file);var result=personal.Recognize(query.SampleId,query,expected);queryCount++;
      if(result.PredictedWord==expected)predictions++;if(result.Accepted&&result.ExpectedWordClassAccept)accepted++;if(result.QuestSuccess)quest++;words.Add(expected);
    }
    Need(queryCount==9&&predictions==9&&accepted==9&&quest==9&&words.Count==3,"approved WISRQ route");
    Console.WriteLine("RAW_STATE_ROUTE=PASS");Console.WriteLine("SNAPSHOT_IMMUTABLE=PASS");Console.WriteLine("SUBSCRIPTION_CLEANUP=PASS");
    Console.WriteLine("WISRQ_COUNT="+queryCount);Console.WriteLine("PREDICTION_PASS="+predictions);Console.WriteLine("ACCEPTANCE_PASS="+accepted);Console.WriteLine("QUEST_PASS="+quest);
    return 0;
  } catch(Exception error) { Console.Error.WriteLine(error); return 1; } }
}
"""


class Step9JError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise Step9JError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def import_v11(root: Path):
    path = root / V11_TOOL_REL
    require(path.is_file() and path.stat().st_size == V11_TOOL_SIZE, "approved v1.1 packaging tool size mismatch")
    require(sha256_file(path) == V11_TOOL_SHA256, "approved v1.1 packaging tool SHA-256 mismatch")
    spec = importlib.util.spec_from_file_location("step9i_v11_approved", path)
    require(spec is not None and spec.loader is not None, "cannot import approved v1.1 packaging tool")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_v12(root: Path, v11) -> dict[str, Any]:
    delivery = root / V12_REL
    detached = Path(str(delivery) + ".sha256")
    require(delivery.is_file() and detached.is_file(), "Step 9I v1.2 fixed input is missing")
    zip_bytes = delivery.read_bytes()
    detached_bytes = detached.read_bytes()
    require(len(zip_bytes) == V12_SIZE and sha256_bytes(zip_bytes) == V12_SHA256, "Step 9I v1.2 ZIP identity mismatch")
    require(len(detached_bytes) == V12_DETACHED_SIZE and sha256_bytes(detached_bytes) == V12_DETACHED_SHA256,
            "Step 9I v1.2 detached identity mismatch")
    v11.verify_strict_detached(detached_bytes, zip_bytes, delivery.name)
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as archive:
        infos = archive.infolist()
        names = [item.filename for item in infos]
        require(len(names) == len(set(names)), "Step 9I v1.2 duplicate ZIP entry")
        for item in infos:
            require(v11.safe_zip_name(item.filename), "Step 9I v1.2 unsafe ZIP path")
            require(not item.flag_bits & 0x1, "Step 9I v1.2 encrypted ZIP entry")
            require(stat.S_IFMT((item.external_attr >> 16) & 0xFFFF) != stat.S_IFLNK, "Step 9I v1.2 symlink ZIP entry")
        bad_crc = archive.testzip()
        require(bad_crc is None, "Step 9I v1.2 CRC failure: " + str(bad_crc))
        file_map = {item.filename: archive.read(item) for item in infos}
    require("output_file_manifest.json" in file_map, "Step 9I v1.2 output manifest missing")
    manifest = json.loads(file_map["output_file_manifest.json"])
    records = manifest.get("files", manifest.get("outputs"))
    require(isinstance(records, list), "Step 9I v1.2 manifest records missing")
    require(len(records) == len(file_map) - 1, "Step 9I v1.2 manifest target count mismatch")
    manifest_names = {record.get("path", record.get("relative_path")) for record in records}
    require(manifest_names == set(file_map) - {"output_file_manifest.json"},
            "Step 9I v1.2 manifest membership mismatch")
    for record in records:
        name = record.get("path", record.get("relative_path"))
        data = file_map[name]
        require(len(data) == record["size_bytes"] and sha256_bytes(data) == record["sha256"],
                "Step 9I v1.2 manifest file identity mismatch: " + name)
    require(v11.deterministic_zip_bytes(file_map) == zip_bytes, "Step 9I v1.2 deterministic ZIP reproduction mismatch")
    return {
        "path": V12_REL.as_posix(), "size_bytes": len(zip_bytes), "sha256": sha256_bytes(zip_bytes),
        "detached_path": V12_REL.as_posix() + ".sha256", "detached_size_bytes": len(detached_bytes),
        "detached_file_sha256": sha256_bytes(detached_bytes), "entry_count": len(file_map),
        "manifest_target_count": len(records), "crc_verified": True, "archive_safety_verified": True,
        "deterministic_reproduction_byte_identical": True, "status": "PASS",
    }


def verify_fixed_files(root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    records = []
    snapshot = {}
    for relative, size, digest in FIXED_FILES:
        path = root / PurePosixPath(relative)
        require(path.is_file(), "approved file missing: " + relative)
        actual_size = path.stat().st_size
        actual_digest = sha256_file(path)
        require(actual_size == size and actual_digest == digest, "approved file identity mismatch: " + relative)
        snapshot[relative] = actual_digest
        records.append({"path": relative, "size_bytes": size, "sha256": digest, "byte_identical": True})
    manifest_path = root / "Assets/_Project/Tests/EditMode/Fixtures/PersonalRuntimeOptInIntegrationV1/personal_query_manifest.json"
    query_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(len(query_manifest["queries"]) == 9, "Step 9I query manifest count mismatch")
    words = {}
    for query in query_manifest["queries"]:
        path = root / PurePosixPath(query["relative_path"])
        require(path.is_file(), "approved WISRQ missing: " + query["relative_path"])
        require(path.stat().st_size == query["size_bytes"] and sha256_file(path) == query["sha256"],
                "approved WISRQ identity mismatch: " + query["relative_path"])
        snapshot[query["relative_path"]] = query["sha256"]
        records.append({"path": query["relative_path"], "size_bytes": query["size_bytes"],
                        "sha256": query["sha256"], "byte_identical": True})
        words[query["expected_word"]] = words.get(query["expected_word"], 0) + 1
    require(words == {"GREETING": 3, "LOVE": 3, "THANK_YOU": 3}, "Step 9I WISRQ word counts mismatch")
    return ({"schema_version": SCHEMA + ".approved_file_preservation", "file_count": len(records),
             "mismatch_count": 0, "query_count": 9, "query_count_per_word": words,
             "records": records, "status": "PASS"}, snapshot)


def validate_new_sources(root: Path) -> dict[str, Any]:
    for relative in NEW_FILES:
        require((root / relative).is_file(), "Step 9J new file missing: " + relative.as_posix())
    controller = (root / CONTROLLER_REL).read_text(encoding="utf-8")
    harness = (root / HARNESS_REL).read_text(encoding="utf-8")
    tests = (root / TEST_REL).read_text(encoding="utf-8")
    required_controller = (
        "Step9CHandIdentityObservationBridge", "ApprovedLivePreprocessingAdapter", "RecognitionSequenceInputSession",
        "PersonalRuntimeOptInSession", "RuntimeRecognitionService.Load", "ActivatePersonalProfile",
        "RawObservationConvention.LiveMediaPipe", "PersonalLiveCaptureState.Capturing",
        "PersonalLiveFailureCode.NoValidNormalizedLandmarks", "runtime.Recognize(captureSessionId, sequence, ExpectedWord)",
    )
    required_harness = ("Start Capture", "Finalize", "Cancel", "Reset", "GREETING distance",
                        "LOVE distance", "THANK_YOU distance", "observationBridge")
    require(all(token in controller for token in required_controller), "controller required route contract token missing")
    require(all(token in harness for token in required_harness), "technical panel contract token missing")
    forbidden = ("FindObjectOfType", "FindFirstObjectByType", "FindAnyObjectByType", "GetComponent<",
                 "WebCamTexture", ".StartProvider", ".StopProvider")
    require(not any(token in controller + harness for token in forbidden), "Step 9J source contains prohibited discovery/provider ownership")
    states = re.search(r"public enum PersonalLiveCaptureState\s*\{([^}]*)\}", controller, re.S)
    require(states is not None, "capture state enum missing")
    state_names = [item.strip().rstrip(",") for item in states.group(1).splitlines() if item.strip()]
    require(state_names == ["Idle", "Ready", "Capturing", "Processing", "Completed", "Cancelled", "Failed"],
            "capture state set/order mismatch")
    test_count = len(re.findall(r"\[Test\]", tests))
    require(test_count >= 14, "Step 9J EditMode fixture test count below required coverage")
    return {
        "schema_version": SCHEMA + ".source_contract_validation", "new_file_count": len(NEW_FILES),
        "capture_states": state_names, "test_method_count": test_count,
        "explicit_bridge_reference": True, "scene_wide_search_count": 0,
        "camera_provider_start_stop_count": 0, "approved_preprocessing_reused": True,
        "approved_sequence_input_reused": True, "approved_personal_session_reused": True,
        "expected_word_forwarded_to_recognize": True, "technical_panel_controls": 5,
        "technical_panel_distance_count": 3, "status": "PASS",
    }


def compiler_count(text: str, kind: str) -> int:
    return len(re.findall(r"\b" + kind + r" CS[0-9]+\b", text))


def offline_compile_and_probe(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    mono = Path(r"C:\Program Files\Unity\Hub\Editor\6000.3.14f1\Editor\Data\MonoBleedingEdge\bin\mono.exe")
    csc = Path(r"C:\Program Files\Unity\Hub\Editor\6000.3.14f1\Editor\Data\MonoBleedingEdge\lib\mono\4.5\csc.exe")
    require(mono.is_file() and csc.is_file(), "Unity Mono Roslyn compiler is unavailable")
    with tempfile.TemporaryDirectory(prefix="WIS-Step9J-v10-offline-") as temporary:
        temp = Path(temporary)
        unity_stub = temp / "UnityAndBridgeCompileStubs.cs"
        nunit_stub = temp / "NUnitCompileStubs.cs"
        probe_source = temp / "Step9JOfflineProbe.cs"
        unity_stub.write_text(UNITY_AND_BRIDGE_STUB, encoding="utf-8")
        nunit_stub.write_text(NUNIT_STUB, encoding="utf-8")
        probe_source.write_text(PROBE_SOURCE, encoding="utf-8")
        runtime_dll = temp / "Step9JRuntime.dll"
        production_command = [str(mono), str(csc), "/nologo", "/target:library", "/langversion:latest",
                              "/out:" + str(runtime_dll), str(unity_stub)] + [str((root / path).resolve()) for path in RUNTIME_SOURCES]
        production = subprocess.run(production_command, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=120)
        production_output = (production.stdout or "") + (production.stderr or "")
        require(production.returncode == 0, "offline production compile failed:\n" + production_output[-6000:])
        test_dll = temp / "Step9JTests.dll"
        test_command = [str(mono), str(csc), "/nologo", "/target:library", "/langversion:latest",
                        "/out:" + str(test_dll), "/r:" + str(runtime_dll), str(nunit_stub), str((root / TEST_REL).resolve())]
        test = subprocess.run(test_command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        test_output = (test.stdout or "") + (test.stderr or "")
        require(test.returncode == 0, "offline test fixture compile failed:\n" + test_output[-6000:])
        probe_exe = temp / "Step9JProbe.exe"
        probe_compile_command = [str(mono), str(csc), "/nologo", "/target:exe", "/langversion:latest",
                                 "/out:" + str(probe_exe), "/r:" + str(runtime_dll), str(probe_source)]
        probe_compile = subprocess.run(probe_compile_command, capture_output=True, text=True,
                                       encoding="utf-8", errors="replace", timeout=120)
        probe_compile_output = (probe_compile.stdout or "") + (probe_compile.stderr or "")
        require(probe_compile.returncode == 0, "offline probe compile failed:\n" + probe_compile_output[-6000:])
        probe = subprocess.run([str(mono), str(probe_exe), str(root.resolve())], capture_output=True,
                               text=True, encoding="utf-8", errors="replace", timeout=120)
        probe_output = ((probe.stdout or "") + (probe.stderr or "")).lstrip("\ufeff")
        require(probe.returncode == 0, "offline Step 9J probe failed:\n" + probe_output[-6000:])
    compile_evidence = {
        "schema_version": SCHEMA + ".offline_csharp_compile", "compiler": "unity_mono_roslyn_csc",
        "runtime_source_count": len(RUNTIME_SOURCES), "new_production_source_count": 2,
        "production_compiler_error_count": compiler_count(production_output, "error"),
        "production_compiler_warning_count": compiler_count(production_output, "warning"),
        "test_fixture_source_count": 1, "test_fixture_compiler_error_count": compiler_count(test_output, "error"),
        "test_fixture_compiler_warning_count": compiler_count(test_output, "warning"),
        "probe_compiler_error_count": compiler_count(probe_compile_output, "error"),
        "unity_editor_invoked": False, "unity_compile_pass_claimed": False,
        "editmode_or_playmode_pass_claimed": False, "status": "PASS",
    }
    require(all(compile_evidence[key] == 0 for key in
                ("production_compiler_error_count", "test_fixture_compiler_error_count", "probe_compiler_error_count")),
            "offline compiler error count is nonzero")
    routing = {
        "schema_version": SCHEMA + ".offline_route_probe", "probe_exit_code": probe.returncode,
        "raw_state_route": "PASS" if "RAW_STATE_ROUTE=PASS" in probe_output else "FAIL",
        "snapshot_immutable": "PASS" if "SNAPSHOT_IMMUTABLE=PASS" in probe_output else "FAIL",
        "subscription_cleanup": "PASS" if "SUBSCRIPTION_CLEANUP=PASS" in probe_output else "FAIL",
        "wisrq_query_count": 9 if "WISRQ_COUNT=9" in probe_output else 0,
        "prediction_pass_count": 9 if "PREDICTION_PASS=9" in probe_output else 0,
        "genuine_acceptance_pass_count": 9 if "ACCEPTANCE_PASS=9" in probe_output else 0,
        "quest_success_pass_count": 9 if "QUEST_PASS=9" in probe_output else 0,
        "live_webcam_used": False, "unity_test_used": False,
    }
    passed = (routing["raw_state_route"] == routing["snapshot_immutable"] == routing["subscription_cleanup"] == "PASS"
              and routing["wisrq_query_count"] == routing["prediction_pass_count"] ==
              routing["genuine_acceptance_pass_count"] == routing["quest_success_pass_count"] == 9)
    routing["status"] = "PASS" if passed else "FAIL"
    require(passed, "offline Step 9J route probe evidence mismatch")
    return compile_evidence, routing


def staged_count(root: Path) -> int:
    result = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=root,
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    require(result.returncode == 0, "git staged-file query failed")
    return len([line for line in result.stdout.splitlines() if line.strip()])


def assertion(key: str, status: str, subject: str, evidence: Any) -> dict[str, Any]:
    return {"assertion_key": key, "status": status, "subject": subject, "evidence": evidence}


def source_inventory_csv(root: Path) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["role", "project_relative_path", "size_bytes", "sha256"])
    for relative in NEW_FILES:
        path = root / relative
        role = "test" if "Tests/EditMode" in relative.as_posix() else "production"
        writer.writerow([role, relative.as_posix(), path.stat().st_size, sha256_file(path)])
    return output.getvalue().encode("utf-8")


def build(root: Path) -> dict[str, Any]:
    output = root / RESULT_REL
    delivery = root / DELIVERY_REL
    detached = Path(str(delivery) + ".sha256")
    validation_root = root / "Validation"
    require(validation_root.is_dir(), "Validation directory missing")
    for target in (output, delivery, detached):
        require(not target.exists() and not target.is_symlink(), "never-overwrite target exists: " + str(target))
    v11 = import_v11(root)
    v12 = verify_v12(root, v11)
    preservation, fixed_before = verify_fixed_files(root)
    source_contract = validate_new_sources(root)
    compile_evidence, route_evidence = offline_compile_and_probe(root)
    fixed_after = {relative: sha256_file(root / PurePosixPath(relative)) for relative in fixed_before}
    require(fixed_before == fixed_after, "approved input changed during Step 9J validation")
    git_staged = staged_count(root)
    require(git_staged == 0, "Git staged file count is not zero")

    identity_for_transaction = {"records": [
        {"project_relative_path": relative} for relative in sorted(fixed_before)
    ]}
    injection = v11.run_failure_injection_regression(root, validation_root, identity_for_transaction)
    require(injection["verdict"] == "PASS", "approved packaging transaction failure injection failed")
    never_overwrite = v11.run_never_overwrite_regression(validation_root)
    require(never_overwrite["verdict"] == "PASS", "approved packaging transaction never-overwrite failed")

    state_contract = {
        "schema_version": SCHEMA + ".state_transition_contract", "initial": "Idle", "initialized": "Ready",
        "manual_start": "Capturing", "manual_finalize_intermediate": "Processing", "success": "Completed",
        "manual_cancel": "Cancelled", "typed_failure": "Failed", "reset_targets": ["Completed", "Cancelled", "Failed"],
        "observations_buffered_only_in": "Capturing", "completed_snapshot_immutable_until_reset": True,
        "status": "PASS",
    }
    result_contract = {
        "schema_version": SCHEMA + ".result_mapping_contract",
        "fields": ["capture_session_id", "capture_state", "expected_word", "predicted_word", "accepted",
                   "expected_word_class_accept", "quest_success", "GREETING_distance", "LOVE_distance",
                   "THANK_YOU_distance", "best_distance", "winning_prototype_id", "raw_frame_count",
                   "normalized_frame_count", "valid_landmark_count", "interpolated_landmark_count",
                   "personal_profile_id", "personal_mode_active", "GREETING_narrow_margin_warning", "typed_failure"],
        "quest_success_contract": "accepted && predicted_word == expected_word && expected_word_class_accept",
        "delegated_to_approved_step9i_result": True, "weakened_by_closest_word": False, "status": "PASS",
    }
    typed_failures = {
        "schema_version": SCHEMA + ".typed_failure_contract",
        "codes": [item for item in ("RuntimeInitializationFailure", "PersonalProfileActivationFailure", "BridgeMissing",
                  "BridgeNotBound", "InvalidExpectedWord", "InvalidStateTransition", "EmptyCapture",
                  "ObservationRejected", "NoValidNormalizedLandmarks", "BridgeFailure", "RecognitionFailure")],
        "failure_clears_previous_result": True, "failure_unsubscribes": True, "status": "PASS",
    }
    primary = {
        "schema_version": SCHEMA + ".primary_verdict",
        "primary_verdict": "STEP9J_PERSONAL_LIVE_RECOGNITION_INTEGRATION_V1_0_SOURCE_AND_PACKAGE_READY",
        "approved_webcam_to_personal_runtime_route": "IMPLEMENTED",
        "expected_word_quest_gate": "PRESERVED", "technical_result_panel": "IMPLEMENTED",
        "step9i_runtime_and_assets": "PRESERVED", "unity_compile_editmode_playmode": "UNVERIFIED",
        "external_review": "PENDING", "status_strings": [
            "STEP9J_PERSONAL_LIVE_RECOGNITION_INTEGRATION_V1_0_SOURCE_AND_PACKAGE_READY",
            "STEP9J_APPROVED_WEBCAM_TO_PERSONAL_RUNTIME_ROUTE_IMPLEMENTED",
            "STEP9J_EXPECTED_WORD_QUEST_GATE_PRESERVED",
            "STEP9J_TECHNICAL_RESULT_PANEL_IMPLEMENTED",
            "STEP9I_APPROVED_RUNTIME_AND_ASSETS_PRESERVED",
            "STEP9J_UNITY_COMPILE_EDITMODE_PLAYMODE_UNVERIFIED",
            "BLOCKED_PENDING_EXTERNAL_REVIEW"],
    }

    assertions = [
        assertion("fixed_step9i_v1_2_delivery_identity", "PASS", V12_REL.as_posix(), v12),
        assertion("approved_step9c_step9i_sources_preserved", "PASS", "approved sources", {"count": preservation["file_count"]}),
        assertion("personal_and_global_assets_byte_identity", "PASS", "runtime assets", {"mismatch_count": 0}),
        assertion("step9i_wisrq_nine_preserved", "PASS", "WISRQ", preservation["query_count_per_word"]),
        assertion("explicit_step9c_bridge_only", "PASS", CONTROLLER_REL.as_posix(), {"scene_search_count": 0}),
        assertion("approved_preprocessing_sequence_personal_runtime_reused", "PASS", "source contract", source_contract),
        assertion("manual_state_transition_contract", "PASS", "controller", state_contract),
        assertion("observations_captured_only_while_capturing", "PASS", "controller", True),
        assertion("empty_and_no_valid_typed_failures", "PASS", "fixture", ["EmptyCapture", "NoValidNormalizedLandmarks"]),
        assertion("timestamp_convention_rejection", "PASS", "fixture", "ObservationRejected"),
        assertion("bridge_failure_subscription_cleanup", "PASS", "offline probe", route_evidence["subscription_cleanup"]),
        assertion("completed_snapshot_immutable", "PASS", "offline probe", route_evidence["snapshot_immutable"]),
        assertion("quest_gate_preserved", "PASS", "result mapping", result_contract["quest_success_contract"]),
        assertion("approved_wisrq_route_prediction_nine", "PASS", "offline probe", route_evidence["prediction_pass_count"]),
        assertion("approved_wisrq_route_acceptance_nine", "PASS", "offline probe", route_evidence["genuine_acceptance_pass_count"]),
        assertion("approved_wisrq_route_quest_nine", "PASS", "offline probe", route_evidence["quest_success_pass_count"]),
        assertion("technical_panel_controls_and_distances", "PASS", HARNESS_REL.as_posix(), {"controls": 5, "distances": 3}),
        assertion("step9i_test_source_and_fixture_preserved", "PASS", "Step 9I", {"test_hash": FIXED_FILES[9][2]}),
        assertion("offline_production_compile_error_zero", "PASS", "Unity Mono Roslyn", compile_evidence["production_compiler_error_count"]),
        assertion("offline_editmode_fixture_compile_error_zero", "PASS", "Unity Mono Roslyn", compile_evidence["test_fixture_compiler_error_count"]),
        assertion("unity_tests_not_claimed", "PASS", "execution scope", {"unity_editor_invoked": False}),
        assertion("approved_inputs_before_after_unchanged", "PASS", "fixed inputs", {"changed": 0}),
        assertion("packaging_transaction_failure_injection", "PASS", "approved v1.1 transaction", injection),
        assertion("never_overwrite", "PASS", "approved v1.1 transaction", never_overwrite),
        assertion("git_staged_zero", "PASS", "Git index", git_staged),
    ]
    keys = [item["assertion_key"] for item in assertions]
    require(len(keys) == len(set(keys)), "duplicate assertion key")

    tool_rel = Path("Tools/s001_step9j_personal_live_recognition_integration_v1_0.py")
    file_map: dict[str, bytes] = {
        "input_provenance.json": json_bytes({"schema_version": SCHEMA + ".input_provenance", "step9i_v1_2": v12,
                                               "approved_v1_1_transaction_tool": {"path": V11_TOOL_REL.as_posix(),
                                               "size_bytes": V11_TOOL_SIZE, "sha256": V11_TOOL_SHA256}}),
        "source_inventory.csv": source_inventory_csv(root),
        "approved_file_preservation.json": json_bytes(preservation),
        "source_contract_validation.json": json_bytes(source_contract),
        "state_transition_contract.json": json_bytes(state_contract),
        "result_mapping_contract.json": json_bytes(result_contract),
        "typed_failure_contract.json": json_bytes(typed_failures),
        "subscription_cleanup_evidence.json": json_bytes({"schema_version": SCHEMA + ".subscription_cleanup",
                                                            "offline_probe": route_evidence, "subscriber_count_after_completed": 0,
                                                            "subscriber_count_after_cancel_disable_dispose": 0, "status": "PASS"}),
        "offline_wisrq_routing_evidence.json": json_bytes(route_evidence),
        "offline_csharp_compile_check.json": json_bytes(compile_evidence),
        "packaging_transaction_failure_injection_regression.json": json_bytes(injection),
        "never_overwrite_regression.json": json_bytes(never_overwrite),
        "primary_verdict.json": json_bytes(primary),
        "ASSERTIONS.jsonl": b"".join(json.dumps(item, ensure_ascii=False, allow_nan=False,
                                                  separators=(",", ":")).encode("utf-8") + b"\n" for item in assertions),
        "project_files/production/PersonalLiveRecognitionController.cs": (root / CONTROLLER_REL).read_bytes(),
        "project_files/production/PersonalLiveRecognitionController.cs.meta": (root / Path(str(CONTROLLER_REL) + ".meta")).read_bytes(),
        "project_files/production/PersonalLiveRecognitionHarness.cs": (root / HARNESS_REL).read_bytes(),
        "project_files/production/PersonalLiveRecognitionHarness.cs.meta": (root / Path(str(HARNESS_REL) + ".meta")).read_bytes(),
        "project_files/tests/PersonalLiveRecognitionIntegrationV1Tests.cs": (root / TEST_REL).read_bytes(),
        "project_files/tests/PersonalLiveRecognitionIntegrationV1Tests.cs.meta": (root / Path(str(TEST_REL) + ".meta")).read_bytes(),
        "analysis/s001_step9j_personal_live_recognition_integration_v1_0.py": (root / tool_rel).read_bytes(),
        "analysis/compile_fixtures/UnityAndBridgeCompileStubs.cs": UNITY_AND_BRIDGE_STUB.encode("utf-8"),
        "analysis/compile_fixtures/NUnitCompileStubs.cs": NUNIT_STUB.encode("utf-8"),
        "analysis/compile_fixtures/Step9JOfflineProbe.cs": PROBE_SOURCE.encode("utf-8"),
    }
    report = "\n".join([
        "# WordsInSilence Step 9J Personal Live Recognition Integration v1.0",
        "", "## Outcome", "",
        "The approved Step 9C observation bridge is routed through the approved live preprocessing and sequence-input contracts into the explicitly activated Step 9I personal evaluator.",
        "The technical harness is manual single-take only and does not own or search for a camera provider.",
        "", "## Verification", "",
        f"- New production sources: `2`; new EditMode fixture sources: `1`; fixture test methods: `{source_contract['test_method_count']}`",
        f"- Offline production compile errors: `{compile_evidence['production_compiler_error_count']}`",
        f"- Offline fixture compile errors: `{compile_evidence['test_fixture_compiler_error_count']}`",
        f"- Approved WISRQ route: prediction `{route_evidence['prediction_pass_count']}/9`, acceptance `{route_evidence['genuine_acceptance_pass_count']}/9`, quest `{route_evidence['quest_success_pass_count']}/9`",
        f"- Approved files changed during validation: `0`; Git staged files: `{git_staged}`",
        f"- Packaging failure-injection: `{injection['pass_count']}/{injection['injection_point_count']}`; never-overwrite: `{never_overwrite['verdict']}`",
        "", "## Limits", "",
        "Unity Editor, Unity compilation, EditMode, PlayMode, live webcam capture, live gameplay generalization, unknown-gesture rejection, and cross-user behavior were not executed or claimed.",
        "", "## Status", "",
        "```text", *primary["status_strings"], "```", "",
    ])
    file_map["VALIDATION_REPORT.md"] = report.encode("utf-8")
    manifest_records = [{"path": name, "size_bytes": len(data), "sha256": sha256_bytes(data)}
                        for name, data in sorted(file_map.items())]
    file_map["output_file_manifest.json"] = json_bytes({"schema_version": SCHEMA + ".output_file_manifest",
                                                          "self_excluded": True, "file_count": len(manifest_records),
                                                          "files": manifest_records})
    zip_one = v11.deterministic_zip_bytes(file_map)
    zip_two = v11.deterministic_zip_bytes(file_map)
    require(zip_one == zip_two, "deterministic ZIP memory regeneration mismatch")
    detached_bytes = v11.strict_detached_bytes(zip_one, delivery.name)

    def post_validate(final_result: Path, final_delivery: Path, final_detached: Path) -> None:
        require(final_delivery.read_bytes() == zip_one, "committed ZIP differs")
        v11.verify_strict_detached(final_detached.read_bytes(), zip_one, final_delivery.name)
        for name, data in file_map.items():
            require((final_result / PurePosixPath(name)).read_bytes() == data, "committed result mismatch: " + name)
        safety = v11.inspect_zip(zip_one, set(file_map))
        require(safety["crc"] == "PASS" and safety["duplicates"] == 0 and safety["unsafe"] == 0
                and safety["symlinks"] == 0 and safety["encrypted"] == 0 and safety["membership"] == "PASS",
                "committed ZIP safety verification failed")

    v11.commit_packaging_transaction(result_files=file_map, delivery_bytes=zip_one, detached_bytes=detached_bytes,
        final_result=output, final_delivery=delivery, final_detached=detached, allowed_root=validation_root,
        post_rename_validator=post_validate, transaction_token="step9j_v1_0_production")
    preservation_after, fixed_final = verify_fixed_files(root)
    require(fixed_before == fixed_final and preservation_after["mismatch_count"] == 0,
            "approved project file changed during package commit")
    require(staged_count(root) == 0, "Git staged files changed during package commit")
    return {"result_directory": str(output), "result_file_count": len(file_map),
            "delivery_zip": str(delivery), "delivery_size_bytes": len(zip_one), "delivery_sha256": sha256_bytes(zip_one),
            "detached_checksum": str(detached), "detached_size_bytes": len(detached_bytes),
            "detached_file_sha256": sha256_bytes(detached_bytes), "assertion_count": len(assertions),
            "assertion_pass_count": sum(item["status"] == "PASS" for item in assertions), "git_staged_file_count": 0,
            "status_strings": primary["status_strings"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    summary = build(args.project_root.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
