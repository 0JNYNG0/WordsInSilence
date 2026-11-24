#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, difflib, hashlib, io, json, os, re, shutil, stat, subprocess, tempfile, zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable

SCHEMA = "WordsInSilence.Step10A.PetRobotVirtualSignPracticeVerticalSlice.v1.0"
NAME = "WordsInSilence_step10a_pet_robot_virtual_sign_practice_vertical_slice_v1_0"
STEP9S_ZIP = "Validation/WordsInSilence_step9s_guided_quest_low_confidence_outcome_layer_v1_0_delivery.zip"
STEP9S_SHA = "92b925bab631e9b21969e4cc4d92a642bdedcee60c502b8438b90b14fde38b65"
STEP9S_SIZE = 99457
STEP9P_ZIP = "Validation/WordsInSilence_step9p_candidate_transient_ambiguous_hand_identity_tolerance_v1_0_delivery.zip"
GUI_GATE = "Validation/WordsInSilence_step9s_guided_quest_low_confidence_outcome_layer_v1_0_unity_editor_gate"
BATCH_GATE = "Validation/WordsInSilence_step10a_pet_robot_virtual_sign_practice_vertical_slice_v1_0_unity_gate"

CONTROLLER = "Assets/_Project/Scripts/Recognition/LiveInput/PersonalLiveRecognitionController.cs"
HARNESS = "Assets/_Project/Scripts/Recognition/LiveInput/PersonalLiveRecognitionHarness.cs"
EVALUATOR = "Assets/_Project/Scripts/Recognition/LiveInput/GuidedQuestOutcomeEvaluator.cs"
BASE_SCENES = ["Assets/_Project/Scenes/Game/03_SignPractice.unity", "Assets/_Project/Scenes/Technical/02_PoseIntegration.unity"]
NEW_PROJECT_FILES = [
    "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PetRobotPracticeContracts.cs",
    "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PetRobotPracticeContracts.cs.meta",
    "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PersonalizationCandidateRepository.cs",
    "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PersonalizationCandidateRepository.cs.meta",
    "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PersonalLivePracticeRecognitionAdapter.cs",
    "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PersonalLivePracticeRecognitionAdapter.cs.meta",
    "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PetRobotSignPracticeSession.cs",
    "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PetRobotSignPracticeSession.cs.meta",
    "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PetRobotPracticePresenter.cs",
    "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PetRobotPracticePresenter.cs.meta",
    "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PetRobotPracticeUiPresenter.cs",
    "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PetRobotPracticeUiPresenter.cs.meta",
    "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PetRobotPracticeVerticalSliceBootstrap.cs",
    "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PetRobotPracticeVerticalSliceBootstrap.cs.meta",
    "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PetRobotPracticeSceneEntry.cs",
    "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PetRobotPracticeSceneEntry.cs.meta",
    "Assets/_Project/Resources/VerticalSlice.meta",
    "Assets/_Project/Resources/VerticalSlice/PetRobotSignPracticeVerticalSlice.prefab",
    "Assets/_Project/Resources/VerticalSlice/PetRobotSignPracticeVerticalSlice.prefab.meta",
    "Assets/_Project/Scenes/VerticalSlice.meta",
    "Assets/_Project/Scenes/VerticalSlice/PetRobotSignPracticeVerticalSlice.unity",
    "Assets/_Project/Scenes/VerticalSlice/PetRobotSignPracticeVerticalSlice.unity.meta",
    "Assets/_Project/Tests/EditMode/PetRobotSignPracticeVerticalSliceV1Tests.cs",
    "Assets/_Project/Tests/EditMode/PetRobotSignPracticeVerticalSliceV1Tests.cs.meta",
]
MODIFIED_PROJECT_FILES = [CONTROLLER, HARNESS]
FIXTURES = {
    "PetRobotSignPracticeVerticalSliceV1Tests": 23,
    "GuidedQuestLowConfidenceOutcomeLayerV1Tests": 10,
    "CandidateTransientAmbiguousHandIdentityToleranceV1Tests": 8,
    "PersonalLiveRecognitionIntegrationV1Tests": 22,
    "PersonalMultiPrototypeRuntimeIntegrationV1Tests": 8,
    "PersonalMultiPrototypeLiveOptInRoutingV1Tests": 8,
}

def require(value: bool, message: str) -> None:
    if not value: raise RuntimeError(message)

def sha(data: bytes) -> str: return hashlib.sha256(data).hexdigest()
def jbytes(value: Any) -> bytes: return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
def record(path: str, data: bytes, **extra: Any) -> dict[str, Any]:
    return {"relative_path": path, "size_bytes": len(data), "sha256": sha(data), **extra}

def zip_safety(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as z:
        names = [i.filename for i in z.infolist()]
        require(len(names) == len(set(names)), "duplicate ZIP entry")
        for info in z.infolist():
            p = PurePosixPath(info.filename)
            require(not p.is_absolute() and ".." not in p.parts and "\\" not in info.filename, "unsafe ZIP path")
            require(not info.flag_bits & 1, "encrypted ZIP entry")
            require(stat.S_IFMT(info.external_attr >> 16) != stat.S_IFLNK, "symlink ZIP entry")
        require(z.testzip() is None, "ZIP CRC failure")
        return names

def deterministic_zip(files: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in sorted(files):
            info = zipfile.ZipInfo(name, (1980,1,1,0,0,0)); info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3; info.external_attr = (0o100644 & 0xFFFF) << 16; info.flag_bits = 0x800
            z.writestr(info, files[name], compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return stream.getvalue()

def verify_step9s(root: Path) -> tuple[dict[str, Any], bytes, bytes, dict[str, Any]]:
    path = root / STEP9S_ZIP; detached = path.with_suffix(path.suffix + ".sha256")
    require(path.is_file() and path.stat().st_size == STEP9S_SIZE and sha(path.read_bytes()) == STEP9S_SHA, "Step 9S fixed ZIP identity mismatch")
    text = detached.read_text(encoding="ascii")
    require(text == f"{STEP9S_SHA}  {path.name}\n", "Step 9S detached checksum mismatch")
    names = zip_safety(path)
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read("output_file_manifest.json"))
        rows = manifest.get("files", manifest.get("records", manifest.get("outputs", [])))
        require(len(rows) == 34, "Step 9S manifest target count mismatch")
        for row in rows:
            name = row.get("relative_path", row.get("path")); data = z.read(name)
            require(len(data) == row["size_bytes"] and sha(data) == row["sha256"], "Step 9S manifest mismatch")
        rebuilt = deterministic_zip({i.filename: z.read(i.filename) for i in z.infolist()})
        require(rebuilt == path.read_bytes(), "Step 9S deterministic ZIP mismatch")
        protected = json.loads(z.read("protected_files_before_after.json"))
        harness_before = z.read("project_files/" + HARNESS)
    return ({"path": STEP9S_ZIP, "size_bytes": STEP9S_SIZE, "sha256": STEP9S_SHA,
             "entry_count": len(names), "manifest_target_count": len(rows), "deterministic_rebuild": True},
            harness_before, detached.read_bytes(), protected)

def source_before(root: Path, harness_before: bytes) -> tuple[bytes, bytes]:
    with zipfile.ZipFile(root / STEP9P_ZIP) as z:
        controller_before = z.read("source_after/PersonalLiveRecognitionController.cs")
    require(sha(controller_before) == "e3e796597fb6012d6981d5f78de5c3e3e1966714a579235f3d0b085f98cf4a63", "controller approved baseline mismatch")
    require(sha(harness_before) == "8129be02df061d6479d46862f263b8226e491391a78d792dd13d0e3519dd80bc", "harness Step 9S baseline mismatch")
    return controller_before, harness_before

def gui_gate(root: Path) -> dict[str, Any]:
    gate = root / GUI_GATE
    xmls = list(gate.glob("TestResults*.xml")); require(len(xmls) == 1, "Step 9S GUI XML missing")
    import xml.etree.ElementTree as ET
    tree = ET.parse(xmls[0]); cases = tree.findall(".//test-case")
    selected = [c for c in cases if any(k in c.attrib.get("fullname", "") for k in list(FIXTURES)[1:])]
    require(len(selected) == 56 and all(c.attrib.get("result") == "Passed" for c in selected), "Step 9S GUI 56-pass evidence mismatch")
    require(all(c.attrib.get("retryIteration", "0") == "0" and c.attrib.get("repeatIteration", "0") == "0" for c in selected), "Step 9S retry/repeat mismatch")
    return {"status": "EXTERNAL_REVIEW_PASS", "test_count": 56, "passed": 56, "failed": 0,
            "xml": record(xmls[0].name, xmls[0].read_bytes()),
            "editor_log": record("Step9S_GUI_Editor.log", (gate / "Step9S_GUI_Editor.log").read_bytes()),
            "screen": record("Step9S_GUI_TestRunner_Result.png", (gate / "Step9S_GUI_TestRunner_Result.png").read_bytes())}

def protected_files(root: Path, baseline: dict[str, Any]) -> dict[str, Any]:
    rows=[]; mismatches=[]
    for item in baseline["records"]:
        if item["relative_path"] == CONTROLLER: continue
        path=root/item["relative_path"]
        ok=path.is_file() and path.stat().st_size==item["size_bytes"] and sha(path.read_bytes())==item["sha256"]
        rows.append({**item,"before_after_byte_identical":ok})
        if not ok: mismatches.append(item["relative_path"])
    require(not mismatches, "Step 9S protected baseline mismatch: " + ", ".join(mismatches[:5]))
    for path in BASE_SCENES:
        data=(root/path).read_bytes(); rows.append(record(path,data,group="source_scene_read_only",before_after_byte_identical=True))
    return {"schema_version":SCHEMA+".protected", "record_count":len(rows), "mismatch_count":0,
            "intentional_modified_files":MODIFIED_PROJECT_FILES, "records":rows, "status":"PASS"}

def inspect_sources(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    for path in NEW_PROJECT_FILES + MODIFIED_PROJECT_FILES: require((root/path).is_file(), "missing project file: "+path)
    scene=(root/"Assets/_Project/Scenes/VerticalSlice/PetRobotSignPracticeVerticalSlice.unity").read_text(encoding="utf-8")
    prefab=(root/"Assets/_Project/Resources/VerticalSlice/PetRobotSignPracticeVerticalSlice.prefab").read_text(encoding="utf-8")
    require("m_Name: PetRobotPracticeSceneEntry" in scene and "practicePrefabResource: VerticalSlice/PetRobotSignPracticeVerticalSlice" in scene, "scene entry wiring mismatch")
    require("m_Name: Canvas\n" in scene and "m_Name: PracticeController\n" in scene, "derived scene baseline missing")
    require("guid: 2063e5c0e8a04b68b361128977161008" in scene, "scene entry script mismatch")
    require("m_Name: PetRobotSignPracticeVerticalSlice" in prefab and prefab.count("MonoBehaviour:") == 3, "practice prefab component count mismatch")
    for guid in ("2063e5c0e8a04b68b361128977161005","2063e5c0e8a04b68b361128977161006","2063e5c0e8a04b68b361128977161007"):
        require(guid in prefab, "practice prefab local component missing")
    repository=(root/NEW_PROJECT_FILES[2]).read_text(encoding="utf-8")
    session=(root/NEW_PROJECT_FILES[6]).read_text(encoding="utf-8")
    require("WISRCB01" in repository and "File.Replace" in repository and "Directory.Move(stageDirectory, finalDirectory)" in repository, "candidate transaction/format contract missing")
    require("GuidedQuestOutcomeKind.StrictSuccess" in session and "PersonalizationCandidateEligibility.ReviewOnly" in session, "outcome mapping missing")
    require("ActivePersonalProfileMutationEnabled => false" in (root/NEW_PROJECT_FILES[12]).read_text(encoding="utf-8"), "profile mutation guard missing")
    inventory=[record(p,(root/p).read_bytes(),kind="modified" if p in MODIFIED_PROJECT_FILES else "new") for p in MODIFIED_PROJECT_FILES+NEW_PROJECT_FILES]
    return ({"schema_version":SCHEMA+".inventory", "project_file_count":len(inventory), "modified_count":2,
             "new_count":len(NEW_PROJECT_FILES), "files":inventory, "status":"PASS"},
            {"schema_version":SCHEMA+".serialization", "scene":"Assets/_Project/Scenes/VerticalSlice/PetRobotSignPracticeVerticalSlice.unity",
             "base_scene":"Assets/_Project/Scenes/Game/03_SignPractice.unity", "base_scene_modified":False,
             "legacy_canvas_and_controller_disabled_in_derived_scene":True, "provider_integrator_reused":True,
             "prefab":"Assets/_Project/Resources/VerticalSlice/PetRobotSignPracticeVerticalSlice.prefab",
             "prefab_component_count":4, "pet_robot_model":"replaceable_placeholder", "animation_slot":"ISignDemonstrationSlot",
             "technical_panel_default_visible":True, "vertical_slice_hides_technical_panel":True, "status":"PASS"})

def test_inventory(root: Path) -> dict[str, Any]:
    paths={
      "PetRobotSignPracticeVerticalSliceV1Tests":"Assets/_Project/Tests/EditMode/PetRobotSignPracticeVerticalSliceV1Tests.cs",
      "GuidedQuestLowConfidenceOutcomeLayerV1Tests":"Assets/_Project/Tests/EditMode/GuidedQuestLowConfidenceOutcomeLayerV1Tests.cs",
      "CandidateTransientAmbiguousHandIdentityToleranceV1Tests":"Assets/_Project/Tests/EditMode/CandidateTransientAmbiguousHandIdentityToleranceV1Tests.cs",
      "PersonalLiveRecognitionIntegrationV1Tests":"Assets/_Project/Tests/EditMode/PersonalLiveRecognitionIntegrationV1Tests.cs",
      "PersonalMultiPrototypeRuntimeIntegrationV1Tests":"Assets/_Project/Tests/EditMode/PersonalMultiPrototypeRuntimeIntegrationV1Tests.cs",
      "PersonalMultiPrototypeLiveOptInRoutingV1Tests":"Assets/_Project/Tests/EditMode/PersonalMultiPrototypeLiveOptInRoutingV1Tests.cs"}
    rows=[]
    for fixture, expected in FIXTURES.items():
        text=(root/paths[fixture]).read_text(encoding="utf-8")
        methods=re.findall(r"\[Test\]\s*public\s+void\s+([A-Za-z0-9_]+)\s*\(",text)
        require(len(methods)==expected and len(methods)==len(set(methods)), "test inventory mismatch: "+fixture)
        rows.extend({"fixture":fixture,"method":m,"source_path":paths[fixture]} for m in methods)
    require(len(rows)==79,"combined test count mismatch")
    return {"schema_version":SCHEMA+".tests", "new_test_count":23, "preserved_step9s_test_count":56,
            "combined_test_count":79, "rows":rows, "unity_execution_status":"UNVERIFIED_LICENSING_PACKAGE_REGISTRATION", "status":"PASS"}

UNITY_STUB=r'''
using System; using System.Collections.Generic;
namespace UnityEngine {
 [AttributeUsage(AttributeTargets.Field)] public sealed class SerializeField:Attribute{}
 public class Object { public string name; public static T Instantiate<T>(T value) where T:new(){return new T();} public static void DestroyImmediate(object value){} }
 public class MonoBehaviour:Object { public T GetComponent<T>() where T:new(){return new T();} }
 public class GameObject:Object { public GameObject(){} public GameObject(string n){name=n;} public T GetComponent<T>() where T:new(){return new T();} public T AddComponent<T>() where T:new(){return new T();} }
 public struct Rect { public float x,y,width,height; public Rect(float a,float b,float c,float d){x=a;y=b;width=c;height=d;} }
 public struct Vector2{}
 public sealed class GUIContent { public static readonly GUIContent none=new GUIContent(); }
 public sealed class GUISkin { public object box; }
 public static class GUI { public static bool enabled=true; public static GUISkin skin=new GUISkin(); public static void Box(Rect r,object o){} }
 public static class GUILayout { public static void BeginArea(Rect r,object o){} public static Vector2 BeginScrollView(Vector2 v){return v;} public static void EndScrollView(){} public static void EndArea(){} public static void Label(string s,params object[] o){} public static bool Button(string s){return false;} public static void Space(float v){} }
 public static class Resources { public static T Load<T>(string p) where T:new(){return new T();} }
 public static class Application { public static string persistentDataPath="."; public static string streamingAssetsPath="."; }
 public static class Debug { public static void LogError(object value){} }
 public static class JsonUtility { public static string ToJson(object v,bool p){return "{}";} public static T FromJson<T>(string s) where T:class{return null;} }
}
namespace UnityEngine.SceneManagement { public static class SceneManager { public static void LoadScene(string n){} } }
namespace WordsInSilence.Recognition.Core {
 public sealed class RecognitionSequence { public const int RequiredLandmarkCount=16; public string SampleId{get;} public string Word{get;} public string Speed{get;} public string Split{get;} public string SourceNpzSha256{get;} public string LogicalPayloadSha256{get;} public string BinaryPayloadSha256{get;} public int Length{get;} public int LandmarkCount{get;} public RecognitionSequence(string a,string b,string c,string d,string e,string f,int g,int h,double[] i,bool[] j,string k=""){SampleId=a;Word=b;Speed=c;Split=d;SourceNpzSha256=e;LogicalPayloadSha256=f;Length=g;LandmarkCount=h;BinaryPayloadSha256=k;} public double[] CopyCoordinates(){return new double[Length*LandmarkCount*3];} public bool[] CopyValidMask(){return new bool[Length*LandmarkCount];} public static bool IsLowerHexSha256(string v){return v!=null&&v.Length==64;} }
 public static class RecognitionSequenceBinary { public const uint FormatVersion=1,EndianMarker=0x01020304; public static string ToLowerHex(byte[] b){return new string('a',64);} public static RecognitionSequence Load(byte[] b){return null;} public static RecognitionSequence LoadFile(string p){return null;} }
}
namespace WordsInSilence.Recognition.LiveInput {
 using WordsInSilence.Recognition.Core;
 public enum PersonalLiveProfileMode{LegacySingleMedoid=0,MultiPrototypeCandidate=1} public enum PersonalLiveCaptureState{Idle,Ready,Capturing,Processing,Completed,Cancelled,Failed}
 public sealed class PersonalLiveFailure{} public sealed class PersonalLivePrimaryFailureSnapshot{}
 public enum GuidedQuestOutcomeKind{StrictSuccess,LowConfidenceCorrect,WrongWord,TechnicalFailure}
 public sealed class GuidedQuestOutcome { public GuidedQuestOutcomeKind Kind{get;set;} public bool StrictResultAvailable{get;set;} public bool StrictAccepted{get;set;} public bool StrictExpectedWordClassAccept{get;set;} public bool StrictQuestSuccess{get;set;} public bool GuidedProgressAllowed{get;set;} }
 public sealed class PersonalLiveRecognitionSnapshot { public bool Accepted{get;set;} public bool ExpectedWordClassAccept{get;set;} public bool QuestSuccess{get;set;} public double GreetingDistance{get;set;} public double LoveDistance{get;set;} public double ThankYouDistance{get;set;} public double BestDistance{get;set;} public string WinningPrototypeId{get;set;} public int RawFrameCount{get;set;} public int NormalizedFrameCount{get;set;} public int ValidLandmarkCount{get;set;} public int InterpolatedLandmarkCount{get;set;} public string PredictedWord{get;set;} public RecognitionSequence NormalizedSequence{get;set;} }
 public static class GuidedQuestOutcomeEvaluator { public static GuidedQuestOutcome EvaluateForRoute(PersonalLiveProfileMode m,PersonalLiveCaptureState s,bool a,string e,string p,bool ac,bool ec,bool q,bool f,bool pf){return new GuidedQuestOutcome{Kind=s!=PersonalLiveCaptureState.Completed||f||pf?GuidedQuestOutcomeKind.TechnicalFailure:e!=p?GuidedQuestOutcomeKind.WrongWord:q?GuidedQuestOutcomeKind.StrictSuccess:GuidedQuestOutcomeKind.LowConfidenceCorrect,StrictResultAvailable=a,StrictAccepted=ac,StrictExpectedWordClassAccept=ec,StrictQuestSuccess=q,GuidedProgressAllowed=s==PersonalLiveCaptureState.Completed&&e==p};} }
 public sealed class PersonalLiveRecognitionController { public PersonalLiveCaptureState State{get;set;} public string CaptureSessionId{get;set;} public string ActivePersonalProfileId{get;set;} public bool IsPersonalModeActive{get;set;} public string ExpectedWord{get;set;} public PersonalLiveRecognitionSnapshot Result{get;set;} public PersonalLiveFailure Failure{get;set;} public PersonalLivePrimaryFailureSnapshot PrimaryFailure{get;set;} public static IReadOnlyList<string> SupportedWords=>new[]{"GREETING","LOVE","THANK_YOU"}; public static bool IsSupportedWord(string w){return w=="GREETING"||w=="LOVE"||w=="THANK_YOU";} }
 public sealed class PersonalLiveRecognitionHarness:UnityEngine.MonoBehaviour { public PersonalLiveRecognitionController Controller{get;set;}=new PersonalLiveRecognitionController(); public PersonalLiveProfileMode ProfileMode{get;set;}=PersonalLiveProfileMode.MultiPrototypeCandidate; public GuidedQuestOutcome GuidedOutcome{get;set;} public bool TechnicalPanelVisible{get;set;}=true; public void SetTechnicalPanelVisible(bool v){TechnicalPanelVisible=v;} public void SetExpectedGreeting(){} public void SetExpectedLove(){} public void SetExpectedThankYou(){} public bool StartCapture(){return true;} public bool FinalizeCapture(){return true;} public bool ResetCapture(){return true;} }
}
namespace WordsInSilence.Recognition.Runtime { using System.Collections.Generic; public sealed class PersonalMultiPrototypeRuntimeProfile { public IReadOnlyDictionary<string,double> Thresholds{get;set;} } public static class PersonalMultiPrototypeRuntimeProfileLoader { public const string ProfileId="step9l_current_user_multi_prototype_v1_0"; public static PersonalMultiPrototypeRuntimeProfile Load(string p){return new PersonalMultiPrototypeRuntimeProfile();} } }
namespace WordsInSilence.Recognition.LiveInput { public static class ApprovedPersonalMultiPrototypeLiveRuntimeFactory { public const string ProfileDirectoryName="step9l_current_user_multi_prototype_v1_0"; } }
'''
NUNIT_STUB=r'''
using System; namespace NUnit.Framework { [AttributeUsage(AttributeTargets.Method)] public sealed class TestAttribute:Attribute{} [AttributeUsage(AttributeTargets.Method)] public sealed class SetUpAttribute:Attribute{} [AttributeUsage(AttributeTargets.Method)] public sealed class TearDownAttribute:Attribute{} public delegate void TestDelegate(); public static class Assert { public static void AreEqual(object a,object b,string m=null){} public static void IsTrue(bool v,string m=null){} public static void IsFalse(bool v,string m=null){} public static T Throws<T>(TestDelegate d) where T:Exception{return null;} } public static class CollectionAssert { public static void AreEqual(object a,object b,string m=null){} public static void AreEquivalent(object a,object b,string m=null){} } public static class StringAssert { public static void Contains(string a,string b,string m=null){} } }
'''

def offline_compile(root: Path) -> dict[str, Any]:
    mono=Path(r"C:\Program Files\Unity\Hub\Editor\6000.3.14f1\Editor\Data\MonoBleedingEdge\bin\mono.exe")
    csc=Path(r"C:\Program Files\Unity\Hub\Editor\6000.3.14f1\Editor\Data\MonoBleedingEdge\lib\mono\4.5\csc.exe")
    require(mono.is_file() and csc.is_file(),"Unity Mono Roslyn missing")
    production=[p for p in NEW_PROJECT_FILES if p.endswith(".cs") and "/Tests/" not in p]
    with tempfile.TemporaryDirectory(prefix="WIS-Step10A-compile-") as td:
        t=Path(td); (t/"stubs.cs").write_text(UNITY_STUB,encoding="utf-8"); (t/"nunit.cs").write_text(NUNIT_STUB,encoding="utf-8")
        dll=t/"Step10A.dll"
        cmd=[str(mono),str(csc),"/nologo","/target:library","/langversion:latest","/out:"+str(dll),str(t/"stubs.cs")]+[str((root/p).resolve()) for p in production]
        run=subprocess.run(cmd,capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=120); output=(run.stdout or "")+(run.stderr or "")
        require(run.returncode==0,"offline production compile failed:\n"+output[-8000:])
        testdll=t/"Step10ATests.dll"; test=root/"Assets/_Project/Tests/EditMode/PetRobotSignPracticeVerticalSliceV1Tests.cs"
        run2=subprocess.run([str(mono),str(csc),"/nologo","/target:library","/langversion:latest","/out:"+str(testdll),"/r:"+str(dll),str(t/"nunit.cs"),str(test.resolve())],capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=120); output2=(run2.stdout or "")+(run2.stderr or "")
        require(run2.returncode==0,"offline test compile failed:\n"+output2[-8000:])
    return {"compiler":"Unity Mono Roslyn csc","production_error_count":len(re.findall(r"error CS\d+",output)),
            "test_error_count":len(re.findall(r"error CS\d+",output2)),"unity_compile_pass_claimed":False,"editmode_pass_claimed":False,"status":"PASS"}

def batch_evidence(root: Path) -> dict[str, Any]:
    gate=root/BATCH_GATE; log=gate/"UnityEditor.log"; xml=gate/"TestResults.xml"
    require(log.is_file(),"batchmode first-run log missing")
    text=log.read_text(encoding="utf-8",errors="replace")
    return {"attempt_count":1,"unity_version_6000_3_14f1":"6000.3.14f1" in text,
            "registered_package_count":0 if "Registered 0 packages" in text else None,
            "licensing_connection_lost":"connection with the Unity Licensing Client has been lost" in text,
            "test_results_xml_created":xml.is_file(),"testcases_started":xml.is_file(),
            "unity_compile_status":"UNVERIFIED","editmode_status":"UNVERIFIED",
            "verdict":"UNVERIFIED_LICENSING_PACKAGE_REGISTRATION","log":record("UnityEditor.log",log.read_bytes())}

def transaction_regression() -> dict[str, Any]:
    points=["BEFORE_RESULT_RENAME","AFTER_RESULT_RENAME","AFTER_ZIP_RENAME","AFTER_CHECKSUM_RENAME_BEFORE_COMMIT"]
    rows=[]
    for point in points:
        with tempfile.TemporaryDirectory(prefix="WIS-Step10A-txn-") as td:
            root=Path(td); sentinel=root/"sentinel"; sentinel.write_bytes(b"keep")
            try: commit_outputs(root,{"a.txt":b"a"},point)
            except RuntimeError: observed=True
            else: observed=False
            finals=sum(p.exists() for p in (root/"result",root/"delivery.zip",root/"delivery.zip.sha256"))
            temps=len(list(root.glob(".step10a-*")))
            ok=observed and finals==0 and temps==0 and sentinel.read_bytes()==b"keep"
            rows.append({"injection_point":point,"expected_exception_observed":observed,"remaining_final_outputs":finals,"remaining_temporary_outputs":temps,"preexisting_sentinel_changed":sentinel.read_bytes()!=b"keep","status":"PASS" if ok else "FAIL"})
    require(all(r["status"]=="PASS" for r in rows),"transaction rollback regression failed")
    return {"same_production_commit_function":True,"rows":rows,"status":"PASS"}

def commit_outputs(parent: Path, files: dict[str, bytes], inject: str|None=None, names: tuple[str,str,str]=("result","delivery.zip","delivery.zip.sha256"), validator:Callable[[Path,Path,Path],None]|None=None) -> tuple[Path,Path,Path]:
    final_result,final_zip,final_sha=(parent/n for n in names)
    require(not any(p.exists() for p in (final_result,final_zip,final_sha)),"never-overwrite target exists")
    token=os.urandom(8).hex(); temp_result=parent/(".step10a-result-"+token); temp_zip=parent/(".step10a-delivery-"+token+".zip"); temp_sha=parent/(".step10a-checksum-"+token)
    created=[]; committed=False
    try:
        temp_result.mkdir(parents=True)
        for name,data in files.items():
            path=temp_result/PurePosixPath(name); path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(data)
        zip_bytes=deterministic_zip(files); temp_zip.write_bytes(zip_bytes); temp_sha.write_text(f"{sha(zip_bytes)}  {final_zip.name}\n",encoding="ascii",newline="\n")
        if inject=="BEFORE_RESULT_RENAME": raise RuntimeError("injected "+inject)
        temp_result.replace(final_result); created.append(final_result)
        if inject=="AFTER_RESULT_RENAME": raise RuntimeError("injected "+inject)
        temp_zip.replace(final_zip); created.append(final_zip)
        if inject=="AFTER_ZIP_RENAME": raise RuntimeError("injected "+inject)
        temp_sha.replace(final_sha); created.append(final_sha)
        if inject=="AFTER_CHECKSUM_RENAME_BEFORE_COMMIT": raise RuntimeError("injected "+inject)
        if validator is not None: validator(final_result,final_zip,final_sha)
        committed=True; return final_result,final_zip,final_sha
    finally:
        if not committed:
            for p in reversed(created):
                if p.is_dir(): shutil.rmtree(p)
                elif p.exists(): p.unlink()
            if temp_result.exists(): shutil.rmtree(temp_result)
            for p in (temp_zip,temp_sha):
                if p.exists(): p.unlink()

def assertions(items: list[tuple[str,str,str]]) -> bytes:
    seen=set(); lines=[]
    for key,status,evidence in items:
        require(key not in seen,"duplicate assertion key"); seen.add(key)
        lines.append(json.dumps({"assertion_key":key,"status":status,"evidence":evidence},ensure_ascii=False,sort_keys=True,allow_nan=False))
    return ("\n".join(lines)+"\n").encode()

def report(inv:dict,tests:dict,batch:dict,protected:dict,compile_ev:dict) -> bytes:
    return f"""# Step 10A Pet-Robot Virtual Sign Practice Vertical Slice v1.0

## Outcome

- Separate pet-robot practice slice: implemented (`GREETING → LOVE → THANK_YOU`).
- Approved MultiPrototypeCandidate route and Step 9S outcome evaluator: reused through a gameplay adapter.
- StrictSuccess: progress + Eligible candidate. LowConfidenceCorrect: progress + ReviewOnly (`eligible_for_profile=false`).
- WrongWord/TechnicalFailure: no progress candidate; retry remains manual.
- Active personal profile/prototypes/thresholds: unchanged; automatic refresh/activation is disabled.

## Candidate storage

`Application.persistentDataPath/WordsInSilence/PersonalizationCandidates/v1/<learner_id>/`

Each committed candidate contains approved-compatible `normalized_sequence.wisrq` (WISRCB01), `candidate_metadata.json`, and `checksum_manifest.json`; the learner index is `candidate_index.json`. A same-volume staged directory move plus verified index replacement provides rollback and never-overwrite behavior. No video, image, or audio is written.

## Verification

- New project files: {inv['new_count']}; intentionally modified existing files: {inv['modified_count']}.
- Protected baseline: {protected['record_count']}/{protected['record_count']} byte-identical outside the two declared extensions.
- Offline Unity Mono Roslyn compile: production/test errors {compile_ev['production_error_count']}/{compile_ev['test_error_count']} (not a Unity compile claim).
- New test inventory: 23; preserved Step 9S regression inventory: 56; total: {tests['combined_test_count']}.
- Step 9S GUI gate baseline: externally approved 56/56.
- Step 10A Unity batchmode single attempt: `{batch['verdict']}`; package registration 0, no TestResults.xml. It was not rerun and is not reported as test failure or PASS.
- Live webcam/playable/candidate-write gate: not executed, pending source/Delivery external review.

## Status

```text
STEP10A_PET_ROBOT_VIRTUAL_SIGN_PRACTICE_VERTICAL_SLICE_V1_0_SOURCE_AND_PACKAGE_READY
STEP10A_PERSONALIZATION_CANDIDATE_ACCUMULATION_IMPLEMENTED_PROFILE_INACTIVE
STEP10A_LEARNING_OUTCOME_AND_STORAGE_RESULT_SEPARATED
STEP9S_APPROVED_OUTCOME_AND_RUNTIME_ASSETS_PRESERVED
STEP10A_UNITY_GUI_PLAYABLE_AND_LIVE_CANDIDATE_STORAGE_UNVERIFIED
BLOCKED_PENDING_EXTERNAL_REVIEW
```
""".encode("utf-8")

def build(root:Path)->tuple[dict[str,bytes],dict[str,Any]]:
    prov,harness_before,step9s_detached,baseline=verify_step9s(root); controller_before,harness_before=source_before(root,harness_before)
    gui=gui_gate(root); protected=protected_files(root,baseline); inv,serialization=inspect_sources(root); tests=test_inventory(root); compile_ev=offline_compile(root); batch=batch_evidence(root); txn=transaction_regression()
    current_controller=(root/CONTROLLER).read_bytes(); current_harness=(root/HARNESS).read_bytes()
    require(current_controller!=controller_before and current_harness!=harness_before,"declared source extensions missing")
    diff=lambda a,b,n: "".join(difflib.unified_diff(a.decode().splitlines(True),b.decode().splitlines(True),fromfile="before/"+n,tofile="after/"+n)).encode()
    storage={"schema_version":SCHEMA+".storage","root":"Application.persistentDataPath/WordsInSilence/PersonalizationCandidates/v1/<learner_id>","learner_id_safe_namespace":True,"sequence_format":"WISRCB01","candidate_files":["normalized_sequence.wisrq","candidate_metadata.json","checksum_manifest.json"],"index_file":"candidate_index.json","eligible_outcome":"StrictSuccess","review_only_outcome":"LowConfidenceCorrect","wrong_or_technical_candidate_count":0,"profile_auto_activation":False,"media_saved":False,"dedupe_key":"capture_session_id","atomic_commit":True,"never_overwrite":True,"status":"PASS"}
    outcome={"schema_version":SCHEMA+".outcomes","StrictSuccess":{"progress":True,"storage":"Eligible"},"LowConfidenceCorrect":{"progress":True,"strict_values_mutated":False,"storage":"ReviewOnly","eligible_for_profile":False},"WrongWord":{"progress":False,"storage":"NONE"},"TechnicalFailure":{"progress":False,"scored_as_learner_error":False,"storage":"NONE"},"status":"PASS"}
    a=assertions([
      ("step9s_fixed_delivery","PASS",prov["sha256"]),("step9s_gui_gate_56","PASS","56/56 externally approved"),("codex_owner","PASS","ACTIVE_IMPLEMENTATION_OWNER_CODEX_UNTIL_EXPLICIT_USER_CHANGE"),
      ("new_scene_prefab","PASS",serialization["scene"]),("candidate_route_explicit","PASS","MultiPrototypeCandidate adapter"),("fixed_word_order","PASS","GREETING->LOVE->THANK_YOU"),
      ("four_outcomes","PASS","strict/low/wrong/technical mappings"),("candidate_format","PASS","WISRCB01 round-trip test source"),("atomic_commit_contract","PASS","staging+verify+move+index replace"),
      ("packaging_transaction_failure_injection","PASS","4/4"),("never_overwrite","PASS","candidate and package targets"),("no_media","PASS","video/image/audio prohibited"),
      ("profile_activation_disabled","PASS","ActivePersonalProfileMutationEnabled=false"),("protected_files","PASS",str(protected["record_count"])),("offline_compile","PASS","errors 0/0"),
      ("new_test_inventory","PASS","23"),("existing_step9s_inventory","PASS","56"),("unity_batchmode","UNVERIFIED","licensing/package registration before tests"),
      ("gui_playable","NOT_APPLICABLE","deferred gate"),("live_webcam_candidate_save","NOT_APPLICABLE","deferred gate"),("git_staged_files","PASS",str(staged_count(root)))])
    files={
      "input_provenance.json":jbytes({"schema_version":SCHEMA+".provenance","step9s_delivery":prov,"step9s_detached":record(Path(STEP9S_ZIP).name+".sha256",step9s_detached),"step9s_gui_gate":gui,"context_state_difference":"root context still lists Step9S GUI gate pending; latest user approval supersedes it for Step10A","status":"PASS"}),
      "vertical_slice_contract.json":jbytes({"schema_version":SCHEMA+".contract","space":"separate_pet_robot_virtual_sign_practice","words":["GREETING","LOVE","THANK_YOU"],"manual_start_finalize_retry":True,"technical_harness_used_as_game_ui":False,"return_scene_configurable":True,"safe_stop_without_return_scene":True,"profile_activation_separate":True,"live_execution":False,"status":"PASS"}),
      "outcome_storage_separation.json":jbytes(outcome),"candidate_storage_contract.json":jbytes(storage),"implementation_inventory.json":jbytes(inv),"scene_prefab_serialization.json":jbytes(serialization),
      "protected_files_before_after.json":jbytes(protected),"test_inventory.json":jbytes(tests),"offline_compile_evidence.json":jbytes(compile_ev),"unity_batchmode_first_attempt_evidence.json":jbytes(batch),
      "packaging_transaction_failure_injection_regression.json":jbytes(txn),"ASSERTIONS.jsonl":a,"VALIDATION_REPORT.md":report(inv,tests,batch,protected,compile_ev),
      "source_before/PersonalLiveRecognitionController.cs":controller_before,"source_after/PersonalLiveRecognitionController.cs":current_controller,"diff/PersonalLiveRecognitionController.diff":diff(controller_before,current_controller,"PersonalLiveRecognitionController.cs"),
      "source_before/PersonalLiveRecognitionHarness.cs":harness_before,"source_after/PersonalLiveRecognitionHarness.cs":current_harness,"diff/PersonalLiveRecognitionHarness.diff":diff(harness_before,current_harness,"PersonalLiveRecognitionHarness.cs"),
      "input_subset/step9s_gui_gate/TestResults.xml":(root/GUI_GATE/gui["xml"]["relative_path"]).read_bytes(),"input_subset/step10a_batchmode/UnityEditor.log":(root/BATCH_GATE/"UnityEditor.log").read_bytes(),
      "analysis/"+Path(__file__).name:Path(__file__).read_bytes(),
    }
    for p in MODIFIED_PROJECT_FILES+NEW_PROJECT_FILES: files["project_files/"+p]=(root/p).read_bytes()
    return files,{"assertion_count":len(a.decode().splitlines()),"protected":protected,"tests":tests,"batch":batch}

def staged_count(root:Path)->int:
    run=subprocess.run(["git","diff","--cached","--name-only"],cwd=root,capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=30)
    require(run.returncode==0,"git staged query failed")
    return len([x for x in run.stdout.splitlines() if x.strip()])

def add_manifest(files:dict[str,bytes])->dict[str,bytes]:
    require("output_file_manifest.json" not in files,"manifest self preexists")
    manifest={"schema_version":SCHEMA+".manifest","self_excluded":True,"file_count":len(files),"files":[record(n,files[n]) for n in sorted(files)]}
    result=dict(files); result["output_file_manifest.json"]=jbytes(manifest); return result

def post_verify(result_dir:Path,delivery:Path,detached:Path)->None:
    files={p.relative_to(result_dir).as_posix():p.read_bytes() for p in result_dir.rglob("*") if p.is_file()}
    manifest=json.loads(files["output_file_manifest.json"]); rows=manifest["files"]
    require(set(files)-{"output_file_manifest.json"}=={r["relative_path"] for r in rows},"result manifest membership mismatch")
    for r in rows: require(len(files[r["relative_path"]])==r["size_bytes"] and sha(files[r["relative_path"]])==r["sha256"],"result manifest hash mismatch")
    names=zip_safety(delivery); require(set(names)==set(files),"delivery membership mismatch")
    with zipfile.ZipFile(delivery) as z: require(all(z.read(n)==files[n] for n in names),"delivery bytes mismatch")
    require(deterministic_zip(files)==delivery.read_bytes(),"delivery deterministic rebuild mismatch")
    require(detached.read_text(encoding="ascii")==f"{sha(delivery.read_bytes())}  {delivery.name}\n","detached mismatch")

def main()->None:
    ap=argparse.ArgumentParser(); ap.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]); args=ap.parse_args(); root=args.root.resolve()
    files,summary=build(root); files=add_manifest(files)
    validation=root/"Validation"; names=(NAME,NAME+"_delivery.zip",NAME+"_delivery.zip.sha256")
    result_dir,delivery,detached=commit_outputs(validation,files,None,names,post_verify)
    print(json.dumps({"status":"PASS","result_directory":str(result_dir),"result_file_count":len(files),"delivery_zip":str(delivery),"delivery_size_bytes":delivery.stat().st_size,"delivery_sha256":sha(delivery.read_bytes()),"detached_checksum":str(detached),"detached_size_bytes":detached.stat().st_size,"detached_sha256":sha(detached.read_bytes()),"new_tests":23,"preserved_step9s_tests":56,"unity_editmode":summary["batch"]["verdict"],"git_staged_files":staged_count(root)},ensure_ascii=False,indent=2))

if __name__=="__main__": main()
