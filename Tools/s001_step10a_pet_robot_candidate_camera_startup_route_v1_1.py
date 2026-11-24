#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable

SCHEMA = "WordsInSilence.Step10A.PetRobotCandidateCameraStartupRoute.v1.1"
NAME = "WordsInSilence_step10a_pet_robot_candidate_camera_startup_route_v1_1"
V10_ZIP = "Validation/WordsInSilence_step10a_pet_robot_virtual_sign_practice_vertical_slice_v1_0_delivery.zip"
V10_ZIP_SIZE = 103898
V10_ZIP_SHA = "0d02b8bc97569c96b0a69d1e321c690984ab5b923b2f5ced6f62b5cb50ee29c4"
V10_DETACHED_SHA = "219b053564e73ebfd66b637362fe96d6509cdd6d774b00a40aecf953ea1412ec"
GUI_ZIP = "Validation/WordsInSilence_step10a_pet_robot_virtual_sign_practice_vertical_slice_v1_0_gui_unity_editor_gate_delivery.zip"
GUI_ZIP_SIZE = 364015
GUI_ZIP_SHA = "29a0b4a198abbf78d5793dcf24afdfc4b265286284324c1db139d337e741e31c"
GUI_DETACHED_SHA = "6e5abf9634a3f48bf09ef99518c57804c2aef7f81e12e340683a79855a9cd13d"
V10_TOOL = "Tools/s001_step10a_pet_robot_virtual_sign_practice_vertical_slice_with_personalization_candidate_accumulation_v1_0.py"

CAMERA_ROUTE = "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PetRobotCandidateCameraStartupRoute.cs"
CAMERA_ROUTE_META = CAMERA_ROUTE + ".meta"
SESSION = "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PetRobotSignPracticeSession.cs"
BOOTSTRAP = "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PetRobotPracticeVerticalSliceBootstrap.cs"
UI = "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PetRobotPracticeUiPresenter.cs"
SCENE_ENTRY = "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PetRobotPracticeSceneEntry.cs"
SCENE = "Assets/_Project/Scenes/VerticalSlice/PetRobotSignPracticeVerticalSlice.unity"
NEW_TEST = "Assets/_Project/Tests/EditMode/PetRobotCandidateCameraStartupRouteV1Tests.cs"
NEW_TEST_META = NEW_TEST + ".meta"

MODIFIED_V10 = [SESSION, BOOTSTRAP, UI, SCENE_ENTRY, SCENE]
NEW_PROJECT_FILES = [CAMERA_ROUTE, CAMERA_ROUTE_META, NEW_TEST, NEW_TEST_META]
DELIVERY_PROJECT_FILES = MODIFIED_V10 + NEW_PROJECT_FILES

FIXTURES = {
    "PetRobotSignPracticeVerticalSliceV1Tests": ("Assets/_Project/Tests/EditMode/PetRobotSignPracticeVerticalSliceV1Tests.cs", 23),
    "GuidedQuestLowConfidenceOutcomeLayerV1Tests": ("Assets/_Project/Tests/EditMode/GuidedQuestLowConfidenceOutcomeLayerV1Tests.cs", 10),
    "CandidateTransientAmbiguousHandIdentityToleranceV1Tests": ("Assets/_Project/Tests/EditMode/CandidateTransientAmbiguousHandIdentityToleranceV1Tests.cs", 8),
    "PersonalLiveRecognitionIntegrationV1Tests": ("Assets/_Project/Tests/EditMode/PersonalLiveRecognitionIntegrationV1Tests.cs", 22),
    "PersonalMultiPrototypeRuntimeIntegrationV1Tests": ("Assets/_Project/Tests/EditMode/PersonalMultiPrototypeRuntimeIntegrationV1Tests.cs", 8),
    "PersonalMultiPrototypeLiveOptInRoutingV1Tests": ("Assets/_Project/Tests/EditMode/PersonalMultiPrototypeLiveOptInRoutingV1Tests.cs", 8),
    "PetRobotCandidateCameraStartupRouteV1Tests": (NEW_TEST, 8),
}

CANDIDATE_ROOT = Path(
    r"C:\Users\local-user\AppData\LocalLow\DefaultCompany\WordsInSilence\WordsInSilence"
    r"\PersonalizationCandidates\v1\local_learner"
)


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def jbytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def record(path: str, data: bytes, **extra: Any) -> dict[str, Any]:
    return {"relative_path": path, "size_bytes": len(data), "sha256": sha(data), **extra}


def deterministic_zip(files: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(files):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (0o100644 & 0xFFFF) << 16
            info.flag_bits = 0x800
            archive.writestr(info, files[name], compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return stream.getvalue()


def zip_safety(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [item.filename for item in infos]
        require(len(names) == len(set(names)), "duplicate ZIP entry: " + str(path))
        for info in infos:
            pure = PurePosixPath(info.filename)
            require(not pure.is_absolute() and ".." not in pure.parts and "\\" not in info.filename,
                    "unsafe ZIP entry: " + info.filename)
            require(not (info.flag_bits & 1), "encrypted ZIP entry: " + info.filename)
            require(stat.S_IFMT(info.external_attr >> 16) != stat.S_IFLNK,
                    "symlink ZIP entry: " + info.filename)
        require(archive.testzip() is None, "ZIP CRC failure: " + str(path))
        return names


def verify_fixed_zip(
    root: Path,
    relative_path: str,
    expected_size: int,
    expected_sha: str,
    expected_detached_file_sha: str,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    path = root / relative_path
    detached = path.with_suffix(path.suffix + ".sha256")
    require(path.is_file(), "fixed ZIP missing: " + relative_path)
    require(path.stat().st_size == expected_size and sha(path.read_bytes()) == expected_sha,
            "fixed ZIP identity mismatch: " + relative_path)
    require(detached.is_file() and sha(detached.read_bytes()) == expected_detached_file_sha,
            "fixed detached checksum file identity mismatch: " + str(detached.relative_to(root)))
    require(detached.read_text(encoding="ascii") == f"{expected_sha}  {path.name}\n",
            "fixed detached checksum syntax mismatch: " + str(detached.relative_to(root)))
    names = zip_safety(path)
    with zipfile.ZipFile(path) as archive:
        file_map = {info.filename: archive.read(info.filename) for info in archive.infolist()}
    require("output_file_manifest.json" in file_map, "fixed ZIP manifest missing")
    manifest = json.loads(file_map["output_file_manifest.json"])
    rows = manifest["files"]
    targets = {row["relative_path"] for row in rows}
    require(targets == set(file_map) - {"output_file_manifest.json"}, "fixed ZIP manifest membership mismatch")
    for row in rows:
        data = file_map[row["relative_path"]]
        require(len(data) == row["size_bytes"] and sha(data) == row["sha256"],
                "fixed ZIP manifest identity mismatch: " + row["relative_path"])
    require(deterministic_zip(file_map) == path.read_bytes(), "fixed ZIP deterministic rebuild mismatch")
    return ({
        "relative_path": relative_path,
        "size_bytes": expected_size,
        "sha256": expected_sha,
        "detached_relative_path": str(detached.relative_to(root)).replace("\\", "/"),
        "detached_size_bytes": detached.stat().st_size,
        "detached_sha256": expected_detached_file_sha,
        "entry_count": len(names),
        "manifest_target_count": len(rows),
        "crc_safe_path_duplicate_symlink_encryption": "PASS",
        "deterministic_rebuild": True,
        "status": "PASS",
    }, file_map)


def snapshot_tree(path: Path) -> dict[str, Any]:
    rows = []
    if path.is_dir():
        for item in sorted(value for value in path.rglob("*") if value.is_file()):
            data = item.read_bytes()
            rows.append(record(item.relative_to(path).as_posix(), data))
    return {
        "path": str(path),
        "exists": path.exists(),
        "file_count": len(rows),
        "files": rows,
    }


def v10_project_inventory(file_map: dict[str, bytes]) -> dict[str, Any]:
    return json.loads(file_map["implementation_inventory.json"])


def v10_project_bytes(file_map: dict[str, bytes], relative_path: str) -> bytes:
    key = "project_files/" + relative_path
    require(key in file_map, "v1.0 project baseline missing: " + relative_path)
    return file_map[key]


def verify_project_delta(root: Path, v10_files: dict[str, bytes]) -> tuple[dict[str, Any], dict[str, bytes]]:
    inventory = v10_project_inventory(v10_files)
    require(inventory["project_file_count"] == 26 and len(inventory["files"]) == 26,
            "v1.0 project inventory count mismatch")
    rows = []
    before = {}
    changed = []
    for entry in inventory["files"]:
        relative = entry["relative_path"]
        current = (root / relative).read_bytes()
        baseline = v10_project_bytes(v10_files, relative)
        before[relative] = baseline
        is_changed = current != baseline
        expected_change = relative in MODIFIED_V10
        require(is_changed == expected_change,
                f"v1.1 project delta mismatch: {relative}; changed={is_changed}, expected={expected_change}")
        if is_changed:
            changed.append(relative)
        rows.append({
            "relative_path": relative,
            "v1_0_size_bytes": len(baseline),
            "v1_0_sha256": sha(baseline),
            "v1_1_size_bytes": len(current),
            "v1_1_sha256": sha(current),
            "changed_in_v1_1": is_changed,
            "status": "PASS",
        })
    for relative in NEW_PROJECT_FILES:
        require((root / relative).is_file(), "new v1.1 project file missing: " + relative)
    require(set(changed) == set(MODIFIED_V10) and len(changed) == len(MODIFIED_V10),
            "modified v1.0 project file set mismatch")
    return ({
        "schema_version": SCHEMA + ".project_delta",
        "v1_0_project_file_count": 26,
        "modified_v1_0_file_count": len(changed),
        "modified_v1_0_files": changed,
        "unchanged_v1_0_file_count": 26 - len(changed),
        "unchanged_v1_0_files_pass": 26 - len(changed),
        "new_project_file_count": len(NEW_PROJECT_FILES),
        "new_project_files": NEW_PROJECT_FILES,
        "records": rows,
        "status": "PASS",
    }, before)


def inspect_wiring(root: Path) -> dict[str, Any]:
    scene = (root / SCENE).read_text(encoding="utf-8").replace("\r\n", "\n")
    route = (root / CAMERA_ROUTE).read_text(encoding="utf-8")
    session = (root / SESSION).read_text(encoding="utf-8")
    bootstrap = (root / BOOTSTRAP).read_text(encoding="utf-8")
    ui = (root / UI).read_text(encoding="utf-8")
    entry = (root / SCENE_ENTRY).read_text(encoding="utf-8")
    require(scene.count("guid: e33729c5ad6f4e44ad02aa19b2710d40") == 1,
            "candidate camera owner component count mismatch")
    require("candidateCameraRoute: {fileID: 1669740324}" in scene,
            "scene entry candidate owner reference missing")
    require("cameraDisplay: {fileID: 447812208}" in scene and
            scene.count("_rawImage: {fileID: 447812208}") == 2,
            "preview/provider shared RawImage wiring mismatch")
    require("m_Name: PracticeController\n  m_TagString: Untagged\n  m_Icon: {fileID: 0}\n"
            "  m_NavMeshLayer: 0\n  m_StaticEditorFlags: 0\n  m_IsActive: 0" in scene,
            "legacy PracticeController is not serialized inactive")
    require("SignPracticeController" not in entry and "SetActive(true)" not in entry,
            "legacy controller activation path introduced")
    require("new UnityWebCamFrameSource()" in route and "startAttempted" in route and
            "StopAndDisposeOnceAsync" in route, "camera lifecycle contract missing")
    require("cameraStarted && hasReceivedFrame && providerObservationReady" in route,
            "camera/frame/observation readiness gate missing")
    require("candidateCameraRoute.BindObservationBridge(harness.ObservationBridge)" in bootstrap,
            "candidate bridge readiness wiring missing")
    require("!cameraReadiness.IsReady" in session and "RecordCameraStartupFailure" in session and
            "CandidateStorageStatus.NotRequested" in session,
            "session readiness/failure no-storage gate missing")
    require("Camera: " in ui and "PreviewTexture" in ui and "session.CameraReady" in ui,
            "camera UI/preview/start-button gate missing")
    require("SetTechnicalPanelVisible(!hideTechnicalPanel)" in bootstrap,
            "technical panel hide behavior changed")
    return {
        "schema_version": SCHEMA + ".wiring",
        "scene": SCENE,
        "active_candidate_camera_owner_count": 1,
        "candidate_camera_owner_component_file_id": 1669740324,
        "scene_entry_explicit_owner_reference": True,
        "legacy_practice_controller_serialized_active": False,
        "legacy_runtime_activation_path_present": False,
        "shared_camera_display_file_id": 447812208,
        "camera_owner_display_reference_count": 1,
        "hand_pose_provider_display_reference_count": 2,
        "camera_start_once_guard": True,
        "readiness_requires_camera_start_frame_and_observation": True,
        "technical_panel_hidden_by_vertical_slice": True,
        "status": "PASS",
    }


def verify_guid_uniqueness(root: Path) -> dict[str, Any]:
    expected = {
        "e33729c5ad6f4e44ad02aa19b2710d40": CAMERA_ROUTE_META,
        "e33729c5ad6f4e44ad02aa19b2710d41": NEW_TEST_META,
    }
    rows = []
    for guid, expected_path in expected.items():
        matches = []
        for path in (root / "Assets").rglob("*.meta"):
            if f"guid: {guid}" in path.read_text(encoding="utf-8", errors="replace"):
                matches.append(path.relative_to(root).as_posix())
        require(matches == [expected_path], "new GUID collision: " + guid + " -> " + repr(matches))
        rows.append({"guid": guid, "relative_path": expected_path, "match_count": 1, "status": "PASS"})
    return {"schema_version": SCHEMA + ".guids", "records": rows, "status": "PASS"}


def test_inventory(root: Path, v10_files: dict[str, bytes]) -> dict[str, Any]:
    rows = []
    for fixture, (relative, expected_count) in FIXTURES.items():
        text = (root / relative).read_text(encoding="utf-8")
        methods = re.findall(r"\[Test\]\s*public\s+void\s+([A-Za-z0-9_]+)\s*\(", text)
        require(len(methods) == expected_count and len(methods) == len(set(methods)),
                "test inventory mismatch: " + fixture)
        if fixture != "PetRobotCandidateCameraStartupRouteV1Tests":
            archive_key = "project_files/" + relative
            if archive_key in v10_files:
                require(v10_files[archive_key] == (root / relative).read_bytes(),
                        "v1.0 test source changed: " + fixture)
        rows.extend({"fixture": fixture, "method": method, "source_path": relative} for method in methods)
    require(len(rows) == 87 and len({(row["fixture"], row["method"]) for row in rows}) == 87,
            "combined unique test inventory mismatch")
    return {
        "schema_version": SCHEMA + ".tests",
        "preserved_v1_0_test_count": 79,
        "new_test_count": 8,
        "combined_unique_test_count": 87,
        "rows": rows,
        "unity_execution_status": "UNVERIFIED_DEFERRED_GUI_GATE",
        "status": "PASS",
    }


def load_v10_stubs(root: Path) -> tuple[str, str]:
    spec = importlib.util.spec_from_file_location("step10a_v10_tool", root / V10_TOOL)
    require(spec is not None and spec.loader is not None, "could not load v1.0 tool module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    unity = module.UNITY_STUB
    nunit = module.NUNIT_STUB
    old_object = "public class Object { public string name; public static T Instantiate<T>(T value) where T:new(){return new T();} public static void DestroyImmediate(object value){} }"
    new_object = "public class Object { public string name; public static T Instantiate<T>(T value) where T:new(){return new T();} public static void DestroyImmediate(object value){} public static void Destroy(object value){} }"
    require(old_object in unity, "v1.0 Unity Object stub anchor missing")
    unity = unity.replace(old_object, new_object)
    old_mono = "public class MonoBehaviour:Object { public T GetComponent<T>() where T:new(){return new T();} }"
    new_mono = "public class MonoBehaviour:Object { public bool isActiveAndEnabled=true; public T GetComponent<T>() where T:new(){return new T();} }"
    require(old_mono in unity, "v1.0 MonoBehaviour stub anchor missing")
    unity = unity.replace(old_mono, new_mono)
    old_gui = "public static class GUI { public static bool enabled=true; public static GUISkin skin=new GUISkin(); public static void Box(Rect r,object o){} }"
    new_gui = "public enum ScaleMode{StretchToFill,ScaleAndCrop,ScaleToFit} public static class GUI { public static bool enabled=true; public static GUISkin skin=new GUISkin(); public static void Box(Rect r,object o){} public static void DrawTexture(Rect r,Texture t,ScaleMode s,bool a){} } public static class GUILayoutUtility { public static Rect GetRect(float w,float h){return new Rect(0,0,w,h);} }"
    require(old_gui in unity, "v1.0 GUI stub anchor missing")
    unity = unity.replace(old_gui, new_gui)
    old_application = "public static class Application { public static string persistentDataPath=\".\"; public static string streamingAssetsPath=\".\"; }"
    new_application = "public static class Application { public static string persistentDataPath=\".\"; public static string streamingAssetsPath=\".\"; public static string dataPath=\".\"; }"
    require(old_application in unity, "v1.0 Application stub anchor missing")
    unity = unity.replace(old_application, new_application)
    insert = """
 public class Texture:Object{}
 public class WebCamTexture:Texture { public bool didUpdateThisFrame; public bool videoVerticallyMirrored; public int videoRotationAngle; public bool isPlaying; public int width=640,height=480; public float requestedFPS=30; public WebCamTexture(){} public WebCamTexture(string n,int w,int h,int f){width=w;height=h;requestedFPS=f;} public void Play(){isPlaying=true;} public void Stop(){isPlaying=false;} public static WebCamDevice[] devices=>new WebCamDevice[0]; }
 public struct WebCamDevice { public string name; public bool isFrontFacing; }
 public struct Quaternion { public static Quaternion Euler(float x,float y,float z){return new Quaternion();} }
 public class RectTransform { public Quaternion localRotation; }
"""
    unity = unity.replace("namespace UnityEngine {\n", "namespace UnityEngine {\n" + insert)
    old_harness = "public sealed class PersonalLiveRecognitionHarness:UnityEngine.MonoBehaviour { public PersonalLiveRecognitionController Controller"
    new_harness = "public sealed class PersonalLiveRecognitionHarness:UnityEngine.MonoBehaviour { public WordsInSilence.Recognition.PhysicalCompatibility.Step9CHandIdentityObservationBridge ObservationBridge{get;set;}=new WordsInSilence.Recognition.PhysicalCompatibility.Step9CHandIdentityObservationBridge(); public PersonalLiveRecognitionController Controller"
    require(old_harness in unity, "v1.0 harness stub anchor missing")
    unity = unity.replace(old_harness, new_harness)
    unity += r'''
namespace UnityEngine.UI { public sealed class RawImage:UnityEngine.Object { public UnityEngine.Texture texture; public UnityEngine.Rect uvRect; public UnityEngine.RectTransform rectTransform=new UnityEngine.RectTransform(); } }
namespace WordsInSilence.Debugging { public sealed class CaptureDiagnostics{} }
namespace WordsInSilence.Camera {
 using System; using System.Threading; using System.Threading.Tasks; using UnityEngine; using WordsInSilence.Debugging;
 public sealed class CameraDeviceInfo { public string Name; public int Index; public bool IsFrontFacing; }
 public sealed class CameraStartRequest { public string DeviceName; public int RequestedWidth=640,RequestedHeight=480,RequestedFps=30; public bool MirrorDisplay=true; }
 public sealed class CameraStartResult { public bool IsSuccess; public string ErrorCode,ErrorDetail,DeviceName; public int ActualWidth,ActualHeight,ActualFps; public static CameraStartResult Failure(string c,string d){return new CameraStartResult{ErrorCode=c,ErrorDetail=d};} public static CameraStartResult Success(string n,int w,int h,int f){return new CameraStartResult{IsSuccess=true,DeviceName=n,ActualWidth=w,ActualHeight=h,ActualFps=f};} }
 public interface ICameraFrameSource { bool IsRunning{get;} string ActiveDeviceName{get;} WebCamTexture ActiveTexture{get;} event Action<CaptureDiagnostics> DiagnosticsUpdated; Task<CameraStartResult> StartAsync(CameraStartRequest r,CancellationToken c); Task StopAsync(CancellationToken c); CameraDeviceInfo[] GetAvailableDevices(); }
 public sealed class UnityWebCamFrameSource:ICameraFrameSource,IDisposable { public bool IsRunning{get;private set;} public string ActiveDeviceName=>"stub"; public WebCamTexture ActiveTexture{get;private set;} public event Action<CaptureDiagnostics> DiagnosticsUpdated{add{}remove{}} public CameraDeviceInfo[] GetAvailableDevices(){return new CameraDeviceInfo[0];} public Task<CameraStartResult> StartAsync(CameraStartRequest r,CancellationToken c){IsRunning=true;ActiveTexture=new WebCamTexture();return Task.FromResult(CameraStartResult.Success("stub",640,480,30));} public Task StopAsync(CancellationToken c){IsRunning=false;return Task.CompletedTask;} public void Dispose(){IsRunning=false;} }
}
namespace WordsInSilence.Settings { public sealed class CaptureConfig:UnityEngine.Object { public int requestedWidth=640,requestedHeight=480,requestedFps=30; public bool mirrorDisplay=true; } }
namespace WordsInSilence.Recognition.LiveInput { public sealed class RawMediaPipeObservation{} }
namespace WordsInSilence.Recognition.PhysicalCompatibility {
 using System; using WordsInSilence.Recognition.LiveInput;
 public sealed class Step9CHandIdentityObservationBridge:UnityEngine.MonoBehaviour { public bool IsBound=true; public bool HasReceivedObservation; public int ObservationCount; public event Action<RawMediaPipeObservation> ObservationArrived; public void Emit(){HasReceivedObservation=true;ObservationCount++;ObservationArrived?.Invoke(new RawMediaPipeObservation());} }
}
'''
    return unity, nunit


def offline_compile(root: Path) -> tuple[dict[str, Any], str, str, Path, Path]:
    mono = Path(r"C:\Program Files\Unity\Hub\Editor\6000.3.14f1\Editor\Data\MonoBleedingEdge\bin\mono.exe")
    csc = Path(r"C:\Program Files\Unity\Hub\Editor\6000.3.14f1\Editor\Data\MonoBleedingEdge\lib\mono\4.5\csc.exe")
    require(mono.is_file() and csc.is_file(), "Unity Mono Roslyn toolchain missing")
    unity_stub, nunit_stub = load_v10_stubs(root)
    production = [
        "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PetRobotPracticeContracts.cs",
        "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PersonalizationCandidateRepository.cs",
        "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PersonalLivePracticeRecognitionAdapter.cs",
        SESSION,
        "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PetRobotPracticePresenter.cs",
        UI,
        BOOTSTRAP,
        SCENE_ENTRY,
        CAMERA_ROUTE,
    ]
    temp = Path(tempfile.mkdtemp(prefix="WIS-Step10A-v1_1-compile-"))
    (temp / "stubs.cs").write_text(unity_stub, encoding="utf-8")
    (temp / "nunit.cs").write_text(nunit_stub, encoding="utf-8")
    dll = temp / "Step10Av11.dll"
    cmd = [str(mono), str(csc), "/nologo", "/target:library", "/langversion:latest",
           "/out:" + str(dll), str(temp / "stubs.cs")] + [str((root / item).resolve()) for item in production]
    run = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    production_output = (run.stdout or "") + (run.stderr or "")
    require(run.returncode == 0, "offline production compile failed:\n" + production_output[-12000:])
    test_dll = temp / "Step10Av11Tests.dll"
    test_sources = [
        "Assets/_Project/Tests/EditMode/PetRobotSignPracticeVerticalSliceV1Tests.cs",
        NEW_TEST,
    ]
    test_cmd = [str(mono), str(csc), "/nologo", "/target:library", "/langversion:latest",
                "/out:" + str(test_dll), "/r:" + str(dll), str(temp / "nunit.cs")] + [
                    str((root / item).resolve()) for item in test_sources]
    test_run = subprocess.run(test_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    test_output = (test_run.stdout or "") + (test_run.stderr or "")
    require(test_run.returncode == 0, "offline test compile failed:\n" + test_output[-12000:])
    return ({
        "compiler": "Unity 6000.3.14f1 Mono Roslyn csc",
        "production_source_count": len(production),
        "test_source_count": len(test_sources),
        "production_error_count": len(re.findall(r"error CS\d+", production_output)),
        "test_error_count": len(re.findall(r"error CS\d+", test_output)),
        "unity_compile_pass_claimed": False,
        "editmode_pass_claimed": False,
        "status": "PASS",
    }, production_output, test_output, temp, dll)


PROBE_SOURCE = r'''
using System; using System.Threading; using System.Threading.Tasks; using System.Reflection;
using WordsInSilence.Camera; using WordsInSilence.Debugging; using WordsInSilence.Practice.PetRobotVerticalSlice;
using WordsInSilence.Recognition.LiveInput;
sealed class Source:ICameraFrameSource,IDisposable { readonly bool succeeds; public int Starts,Stops,Disposes; public bool IsRunning{get;private set;} public string ActiveDeviceName=>"fake"; public UnityEngine.WebCamTexture ActiveTexture=>null; public event Action<CaptureDiagnostics> DiagnosticsUpdated{add{}remove{}} public Source(bool succeeds=true){this.succeeds=succeeds;} public CameraDeviceInfo[] GetAvailableDevices(){return new CameraDeviceInfo[0];} public Task<CameraStartResult> StartAsync(CameraStartRequest r,CancellationToken c){Starts++;IsRunning=succeeds;return Task.FromResult(succeeds?CameraStartResult.Success("fake",640,480,30):CameraStartResult.Failure("CAMERA_TIMEOUT","fake timeout"));} public Task StopAsync(CancellationToken c){Stops++;IsRunning=false;return Task.CompletedTask;} public void Dispose(){Disposes++;} }
sealed class Ready:IPetRobotCandidateCameraReadiness { public event Action Changed; public CandidateCameraStartupState CameraState{get;private set;}=CandidateCameraStartupState.Starting; public bool IsReady=>CameraState==CandidateCameraStartupState.Ready; public bool CameraStarted{get;private set;}=true; public bool HasReceivedFrame{get;private set;} public bool ProviderObservationReady{get;private set;} public int CameraFrameCount{get;private set;} public int ProviderObservationCount{get;private set;} public string LastErrorCode{get;private set;}=""; public string LastErrorMessage{get;private set;}=""; public void Frame(){HasReceivedFrame=true;CameraFrameCount=1;Changed?.Invoke();} public void Observation(){ProviderObservationReady=true;ProviderObservationCount=1;CameraState=CandidateCameraStartupState.Ready;Changed?.Invoke();} public void Fail(){CameraStarted=false;CameraState=CandidateCameraStartupState.Failed;LastErrorCode="CAMERA_TIMEOUT";LastErrorMessage="fake";Changed?.Invoke();} }
sealed class Adapter:IPersonalLivePracticeRecognitionAdapter { public int Begins; PersonalLiveCaptureState state=PersonalLiveCaptureState.Ready; public PersonalLiveCaptureState CaptureState=>state; public string CaptureSessionId=>""; public string ActiveProfileId=>"step9l_current_user_multi_prototype_v1_0"; public bool CandidateRouteActive=>true; public bool SetExpectedWord(string w){return state==PersonalLiveCaptureState.Ready;} public bool BeginCapture(){if(state!=PersonalLiveCaptureState.Ready)return false;Begins++;state=PersonalLiveCaptureState.Capturing;return true;} public PracticeRecognitionAttempt FinalizeCapture(){return null;} public bool ResetCapture(){return false;} }
sealed class Repo:IPersonalizationCandidateRepository { public int Saves; public string RootDirectory=>"fake"; public string LearnerId=>"fake"; public CandidateCountSnapshot Counts=>CandidateCountSnapshot.Empty; public CandidateCountSnapshot ReloadAndVerify(){return Counts;} public CandidateSaveResult Save(PersonalizationCandidateSaveRequest r){Saves++;return null;} }
static class Probe { static object Call(object o,string n){return o.GetType().GetMethod(n,BindingFlags.Instance|BindingFlags.NonPublic).Invoke(o,null);} static void SetSource(object route,Source source){route.GetType().GetField("cameraSource",BindingFlags.Instance|BindingFlags.NonPublic).SetValue(route,source);} static void Main(){var route=new PetRobotCandidateCameraStartupRoute();var source=new Source();SetSource(route,source);((Task)Call(route,"StartCameraOnceAsync")).GetAwaiter().GetResult();((Task)Call(route,"StartCameraOnceAsync")).GetAwaiter().GetResult();Call(route,"OnDisable");Call(route,"OnDestroy");var ready=new Ready();var adapter=new Adapter();var repo=new Repo();using(var session=new PetRobotSignPracticeSession(adapter,repo,null,ready)){bool blocked=!session.StartExpression();ready.Frame();blocked=blocked&&!session.StartExpression();ready.Observation();bool allowed=session.StartExpression();Console.WriteLine("START_BLOCKED_BEFORE_READY="+blocked);Console.WriteLine("START_ALLOWED_AFTER_READY="+allowed);}var failedRoute=new PetRobotCandidateCameraStartupRoute();var failedSource=new Source(false);SetSource(failedRoute,failedSource);var failedAdapter=new Adapter();var failedRepo=new Repo();using(var failed=new PetRobotSignPracticeSession(failedAdapter,failedRepo,null,failedRoute)){((Task)Call(failedRoute,"StartCameraOnceAsync")).GetAwaiter().GetResult();Console.WriteLine("STARTUP_FAILURE_TECHNICAL="+(failed.LastLearningOutcome!=null&&failed.LastLearningOutcome.Kind==GuidedQuestOutcomeKind.TechnicalFailure));Console.WriteLine("STARTUP_FAILURE_NO_PROGRESS_STORAGE="+(!failed.IsCurrentWordComplete&&failedAdapter.Begins==0&&failedRepo.Saves==0));}Call(failedRoute,"OnDisable");Call(failedRoute,"OnDestroy");Console.WriteLine("CAMERA_STARTS="+source.Starts);Console.WriteLine("CAMERA_STOPS="+source.Stops);Console.WriteLine("CAMERA_DISPOSES="+source.Disposes);bool pass=source.Starts==1&&source.Stops==1&&source.Disposes==1&&failedSource.Starts==1;Console.WriteLine("PROBE_STATUS="+(pass?"PASS":"FAIL"));if(!pass)Environment.Exit(2);} }
'''


def run_probe(mono: Path, csc: Path, temp: Path, dll: Path) -> tuple[dict[str, Any], str]:
    source = temp / "probe.cs"
    source.write_text(PROBE_SOURCE, encoding="utf-8")
    exe = temp / "probe.exe"
    compile_run = subprocess.run(
        [str(mono), str(csc), "/nologo", "/target:exe", "/langversion:latest",
         "/out:" + str(exe), "/r:" + str(dll), str(source)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    compile_output = (compile_run.stdout or "") + (compile_run.stderr or "")
    require(compile_run.returncode == 0, "offline probe compile failed:\n" + compile_output[-8000:])
    run = subprocess.run([str(mono), str(exe)], capture_output=True, text=True,
                         encoding="utf-8", errors="replace", timeout=120, cwd=temp)
    output = (run.stdout or "") + (run.stderr or "")
    require(run.returncode == 0 and "PROBE_STATUS=PASS" in output,
            "offline camera route probe failed:\n" + output[-8000:])
    values = {}
    for line in output.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.lstrip("\ufeff")] = value
    expected = {
        "START_BLOCKED_BEFORE_READY": "True",
        "START_ALLOWED_AFTER_READY": "True",
        "STARTUP_FAILURE_TECHNICAL": "True",
        "STARTUP_FAILURE_NO_PROGRESS_STORAGE": "True",
        "CAMERA_STARTS": "1",
        "CAMERA_STOPS": "1",
        "CAMERA_DISPOSES": "1",
        "PROBE_STATUS": "PASS",
    }
    require(all(values.get(key) == value for key, value in expected.items()),
            "offline probe evidence mismatch: " + repr(values))
    return ({"schema_version": SCHEMA + ".probe", "values": values,
             "fake_camera_source": True, "fake_provider_readiness": True,
             "unity_execution_claimed": False, "status": "PASS"}, output)


def protected_files(root: Path, v10_files: dict[str, bytes]) -> dict[str, Any]:
    baseline = json.loads(v10_files["protected_files_before_after.json"])
    unique = {}
    for row in baseline["records"]:
        relative = row["relative_path"]
        identity = (row["size_bytes"], row["sha256"])
        require(relative not in unique or unique[relative] == identity,
                "conflicting v1.0 protected identity: " + relative)
        unique[relative] = identity
    extra = {
        "Assets/_Project/Scripts/Recognition/Core/RecognitionCore.cs":
            (2342, "34ac176049a0340ef6983df29f433d0d5f4bb83720b7b8f7da9747fdfabef498"),
        "Assets/_Project/Scripts/Recognition/LiveInput/GuidedQuestOutcomeEvaluator.cs":
            (7112, "05a827d097c06012f0cec30c0c812e2b3aab8faa2e92865a163f07c08418fd1e"),
    }
    unique.update(extra)
    rows = []
    for relative in sorted(unique):
        size, digest = unique[relative]
        path = root / relative
        require(path.is_file(), "protected file missing: " + relative)
        data = path.read_bytes()
        require(len(data) == size and sha(data) == digest, "protected file changed: " + relative)
        rows.append({"relative_path": relative, "size_bytes": size, "sha256": digest,
                     "before_after_byte_identical": True, "status": "PASS"})
    return {"schema_version": SCHEMA + ".protected", "record_count": len(rows),
            "mismatch_count": 0, "records": rows, "status": "PASS"}


def profile_identity(root: Path) -> dict[str, Any]:
    profile = root / "Assets/StreamingAssets/RecognitionCoreV1/PersonalProfiles/step9l_current_user_multi_prototype_v1_0"
    rows = []
    for path in sorted(value for value in profile.iterdir() if value.is_file()):
        rows.append(record(path.name, path.read_bytes()))
    require(len(rows) == 22, "candidate profile file count mismatch")
    prototypes = [row for row in rows if row["relative_path"].endswith(".wisrt")]
    require(len(prototypes) == 9, "candidate prototype count mismatch")
    config_path = profile / "personal_multi_prototype_threshold_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    thresholds = config["thresholds"]
    expected = {
        "GREETING": 0.5113059473720765,
        "LOVE": 0.391276023460427,
        "THANK_YOU": 0.37709977431991587,
    }
    require(thresholds == expected, "candidate threshold map mismatch")
    return {"schema_version": SCHEMA + ".profile", "profile_id": profile.name,
            "file_count": len(rows), "prototype_count": 9, "threshold_count": 3,
            "thresholds": thresholds, "records": rows, "status": "PASS"}


def staged_count(root: Path) -> int:
    run = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=root,
                         capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    require(run.returncode == 0, "Git staged query failed")
    return len([line for line in run.stdout.splitlines() if line.strip()])


def commit_outputs(
    parent: Path,
    files: dict[str, bytes],
    inject: str | None = None,
    names: tuple[str, str, str] = ("result", "delivery.zip", "delivery.zip.sha256"),
    validator: Callable[[Path, Path, Path], None] | None = None,
) -> tuple[Path, Path, Path]:
    final_result, final_zip, final_detached = (parent / value for value in names)
    require(not any(path.exists() for path in (final_result, final_zip, final_detached)),
            "never-overwrite target exists")
    token = os.urandom(8).hex()
    temp_result = parent / (".step10a-v1_1-result-" + token)
    temp_zip = parent / (".step10a-v1_1-delivery-" + token + ".zip")
    temp_detached = parent / (".step10a-v1_1-checksum-" + token)
    created = []
    committed = False
    try:
        temp_result.mkdir(parents=True)
        for name, data in files.items():
            path = temp_result / PurePosixPath(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        zip_bytes = deterministic_zip(files)
        temp_zip.write_bytes(zip_bytes)
        temp_detached.write_text(f"{sha(zip_bytes)}  {final_zip.name}\n", encoding="ascii", newline="\n")
        if inject == "BEFORE_RESULT_RENAME":
            raise RuntimeError("injected " + inject)
        temp_result.replace(final_result)
        created.append(final_result)
        if inject == "AFTER_RESULT_RENAME":
            raise RuntimeError("injected " + inject)
        temp_zip.replace(final_zip)
        created.append(final_zip)
        if inject == "AFTER_DELIVERY_RENAME":
            raise RuntimeError("injected " + inject)
        temp_detached.replace(final_detached)
        created.append(final_detached)
        if inject == "AFTER_DETACHED_RENAME_BEFORE_COMMIT":
            raise RuntimeError("injected " + inject)
        if validator is not None:
            validator(final_result, final_zip, final_detached)
        committed = True
        return final_result, final_zip, final_detached
    finally:
        if not committed:
            for path in reversed(created):
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()
            if temp_result.exists():
                shutil.rmtree(temp_result)
            for path in (temp_zip, temp_detached):
                if path.exists():
                    path.unlink()


def transaction_regression() -> dict[str, Any]:
    points = ["BEFORE_RESULT_RENAME", "AFTER_RESULT_RENAME", "AFTER_DELIVERY_RENAME",
              "AFTER_DETACHED_RENAME_BEFORE_COMMIT"]
    rows = []
    for point in points:
        with tempfile.TemporaryDirectory(prefix="WIS-Step10A-v1_1-txn-") as directory:
            parent = Path(directory)
            sentinel = parent / "sentinel.bin"
            sentinel.write_bytes(b"preserve")
            try:
                commit_outputs(parent, {"proof.txt": b"proof"}, point)
            except RuntimeError:
                observed = True
            else:
                observed = False
            finals = sum(path.exists() for path in
                         (parent / "result", parent / "delivery.zip", parent / "delivery.zip.sha256"))
            temps = len(list(parent.glob(".step10a-v1_1-*")))
            ok = observed and finals == 0 and temps == 0 and sentinel.read_bytes() == b"preserve"
            rows.append({"injection_point": point, "expected_exception_observed": observed,
                         "remaining_final_outputs": finals, "remaining_temporary_outputs": temps,
                         "preexisting_sentinel_changed": sentinel.read_bytes() != b"preserve",
                         "status": "PASS" if ok else "FAIL"})
    require(all(row["status"] == "PASS" for row in rows), "transaction rollback regression failed")
    with tempfile.TemporaryDirectory(prefix="WIS-Step10A-v1_1-never-") as directory:
        parent = Path(directory)
        commit_outputs(parent, {"proof.txt": b"proof"})
        before = {path.name: sha(path.read_bytes()) if path.is_file() else "directory"
                  for path in parent.iterdir()}
        try:
            commit_outputs(parent, {"proof.txt": b"replacement"})
        except RuntimeError:
            blocked = True
        else:
            blocked = False
        after = {path.name: sha(path.read_bytes()) if path.is_file() else "directory"
                 for path in parent.iterdir()}
        require(blocked and before == after, "never-overwrite regression failed")
    return {"same_production_commit_function": True, "failure_injection_rows": rows,
            "never_overwrite_second_commit_blocked": True,
            "preexisting_outputs_unchanged": True, "status": "PASS"}


def assertions(items: list[tuple[str, str, str]]) -> bytes:
    keys = set()
    lines = []
    for key, status, evidence in items:
        require(key not in keys, "duplicate assertion key: " + key)
        keys.add(key)
        lines.append(json.dumps({"assertion_key": key, "status": status, "evidence": evidence},
                                ensure_ascii=False, sort_keys=True, allow_nan=False))
    return ("\n".join(lines) + "\n").encode("utf-8")


def add_manifest(files: dict[str, bytes]) -> dict[str, bytes]:
    require("output_file_manifest.json" not in files, "manifest self already present")
    manifest = {"schema_version": SCHEMA + ".manifest", "self_excluded": True,
                "file_count": len(files), "files": [record(name, files[name]) for name in sorted(files)]}
    result = dict(files)
    result["output_file_manifest.json"] = jbytes(manifest)
    return result


def post_verify(result_dir: Path, delivery: Path, detached: Path) -> None:
    files = {path.relative_to(result_dir).as_posix(): path.read_bytes()
             for path in result_dir.rglob("*") if path.is_file()}
    manifest = json.loads(files["output_file_manifest.json"])
    rows = manifest["files"]
    require(set(files) - {"output_file_manifest.json"} == {row["relative_path"] for row in rows},
            "result manifest membership mismatch")
    for row in rows:
        data = files[row["relative_path"]]
        require(len(data) == row["size_bytes"] and sha(data) == row["sha256"],
                "result manifest hash mismatch: " + row["relative_path"])
    names = zip_safety(delivery)
    require(set(names) == set(files), "delivery ZIP membership mismatch")
    with zipfile.ZipFile(delivery) as archive:
        require(all(archive.read(name) == files[name] for name in names), "delivery ZIP payload mismatch")
    require(deterministic_zip(files) == delivery.read_bytes(), "delivery ZIP deterministic rebuild mismatch")
    require(detached.read_text(encoding="ascii") == f"{sha(delivery.read_bytes())}  {delivery.name}\n",
            "delivery detached checksum mismatch")


def unified(before: bytes, after: bytes, name: str) -> bytes:
    return "".join(difflib.unified_diff(
        before.decode("utf-8").splitlines(True),
        after.decode("utf-8").splitlines(True),
        fromfile="v1.0/" + name,
        tofile="v1.1/" + name,
    )).encode("utf-8")


def report(
    project_delta: dict[str, Any],
    tests: dict[str, Any],
    compile_evidence: dict[str, Any],
    protected: dict[str, Any],
    wiring: dict[str, Any],
) -> bytes:
    return f"""# Step 10A v1.1 Candidate-Only Webcam Startup Route

## Outcome

- One active candidate-only camera lifecycle owner is explicitly wired in the pet-robot vertical-slice Scene.
- The owner reuses `ICameraFrameSource` / `UnityWebCamFrameSource`; legacy `SignPracticeController` remains serialized inactive and is never activated.
- The same `CameraFeed` texture is consumed by the gameplay preview and the existing Hand/Pose providers.
- Readiness requires camera start, at least one real webcam frame, and at least one bound Step 9C bridge observation.
- Start Expression is disabled/rejected before readiness. Startup failure is a non-scoring `TechnicalFailure` with no progress and no candidate storage.
- Disable/Destroy share one guarded stop/dispose/unsubscribe transaction.

## Verification

- v1.0 project baseline: {project_delta['v1_0_project_file_count']} files; modified {project_delta['modified_v1_0_file_count']}; unchanged {project_delta['unchanged_v1_0_file_count']}/{project_delta['unchanged_v1_0_file_count']}.
- New v1.1 project files: {project_delta['new_project_file_count']}.
- Scene owner count: {wiring['active_candidate_camera_owner_count']}; shared display fileID: {wiring['shared_camera_display_file_id']}.
- Offline Unity Mono Roslyn: production/test errors {compile_evidence['production_error_count']}/{compile_evidence['test_error_count']}.
- Test inventory: preserved 79 + new 8 = {tests['combined_unique_test_count']} unique tests.
- Protected approved files: {protected['record_count']}/{protected['record_count']} byte-identical.
- Unity GUI/webcam/candidate storage and 87-test execution were not run in this source step.

## Status

```text
STEP10A_V1_1_CANDIDATE_ONLY_CAMERA_STARTUP_ROUTE_SOURCE_AND_PACKAGE_READY
STEP10A_V1_1_SINGLE_CANDIDATE_CAMERA_OWNER_IMPLEMENTED
STEP10A_V1_1_CAMERA_FRAME_AND_OBSERVATION_READINESS_GATE_IMPLEMENTED
STEP10A_V1_0_APPROVED_RECOGNITION_STORAGE_AND_ASSETS_PRESERVED
STEP10A_V1_1_UNITY_GUI_WEBCAM_CANDIDATE_STORAGE_AND_87_TESTS_UNVERIFIED
BLOCKED_PENDING_EXTERNAL_REVIEW
```
""".encode("utf-8")


def build(root: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    candidate_before = snapshot_tree(CANDIDATE_ROOT)
    git_before = staged_count(root)
    require(git_before == 0, "Git staged files must remain zero")
    v10_info, v10_files = verify_fixed_zip(
        root, V10_ZIP, V10_ZIP_SIZE, V10_ZIP_SHA, V10_DETACHED_SHA)
    gui_info, _ = verify_fixed_zip(
        root, GUI_ZIP, GUI_ZIP_SIZE, GUI_ZIP_SHA, GUI_DETACHED_SHA)
    project_delta, v10_before = verify_project_delta(root, v10_files)
    wiring = inspect_wiring(root)
    guid_evidence = verify_guid_uniqueness(root)
    tests = test_inventory(root, v10_files)
    protected = protected_files(root, v10_files)
    profile = profile_identity(root)
    compile_evidence, production_output, test_output, temp, dll = offline_compile(root)
    try:
        mono = Path(r"C:\Program Files\Unity\Hub\Editor\6000.3.14f1\Editor\Data\MonoBleedingEdge\bin\mono.exe")
        csc = Path(r"C:\Program Files\Unity\Hub\Editor\6000.3.14f1\Editor\Data\MonoBleedingEdge\lib\mono\4.5\csc.exe")
        probe, probe_output = run_probe(mono, csc, temp, dll)
    finally:
        shutil.rmtree(temp, ignore_errors=True)
    transaction = transaction_regression()
    candidate_after = snapshot_tree(CANDIDATE_ROOT)
    require(candidate_before == candidate_after, "candidate repository changed during source/package validation")
    git_after = staged_count(root)
    require(git_after == 0, "Git staged files changed during validation")

    assertion_bytes = assertions([
        ("implementation_owner", "PASS", "Codex"),
        ("v1_0_delivery_fixed", "PASS", v10_info["sha256"]),
        ("v1_0_gui_gate_fixed", "PASS", gui_info["sha256"]),
        ("v1_0_project_unchanged_files", "PASS", str(project_delta["unchanged_v1_0_file_count"])),
        ("candidate_camera_owner_count", "PASS", "1"),
        ("legacy_controller_inactive", "PASS", "serialized false; no runtime activation path"),
        ("shared_preview_provider_source", "PASS", "RawImage fileID 447812208"),
        ("camera_start_once", "PASS", "fake source calls 1"),
        ("camera_frame_observation_readiness", "PASS", "camera+frame+Step9C observation"),
        ("start_expression_readiness_gate", "PASS", "blocked before; allowed after"),
        ("startup_failure_technical", "PASS", "non-scoring/no-progress/no-storage"),
        ("cleanup_single_stop_dispose", "PASS", "1/1 and unsubscribed"),
        ("new_tests", "PASS", "8"),
        ("combined_test_inventory", "PASS", "87"),
        ("offline_production_compile", "PASS", "errors 0"),
        ("offline_test_compile", "PASS", "errors 0"),
        ("protected_files", "PASS", str(protected["record_count"])),
        ("candidate_profile_prototypes_thresholds", "PASS", "9 prototypes; 3 fixed thresholds"),
        ("candidate_repository_unchanged", "PASS", str(candidate_after["file_count"])),
        ("packaging_transaction_rollback", "PASS", "4/4"),
        ("never_overwrite", "PASS", "second commit blocked; originals unchanged"),
        ("git_staged_files", "PASS", "0"),
        ("unity_gui_execution", "UNVERIFIED", "deferred external GUI gate"),
        ("live_webcam_candidate_storage", "NOT_APPLICABLE", "prohibited in source step"),
    ])

    files: dict[str, bytes] = {
        "input_provenance.json": jbytes({"schema_version": SCHEMA + ".provenance",
                                          "v1_0_delivery": v10_info, "v1_0_gui_gate": gui_info,
                                          "context_sha256": sha((root / "WordsInSilence_PROJECT_CONTEXT.md").read_bytes()),
                                          "status": "PASS"}),
        "implementation_delta.json": jbytes(project_delta),
        "scene_camera_wiring_evidence.json": jbytes(wiring),
        "camera_ownership_readiness_cleanup_contract.json": jbytes({
            "schema_version": SCHEMA + ".contract",
            "owner_count": 1,
            "owner_layer": "vertical_slice_gameplay",
            "frame_source_api": "ICameraFrameSource/UnityWebCamFrameSource",
            "legacy_sign_practice_controller_used": False,
            "personal_live_recognition_harness_owns_camera": False,
            "start_per_playmode_maximum": 1,
            "ready_requires": ["camera_started", "webcam_frame_received", "step9c_observation_received"],
            "startup_failure_outcome": "TechnicalFailure",
            "startup_failure_scored_as_learner_error": False,
            "startup_failure_progress": False,
            "startup_failure_storage": False,
            "automatic_capture": False,
            "automatic_retry": False,
            "cleanup": "single guarded stop/dispose/unsubscribe",
            "status": "PASS",
        }),
        "guid_uniqueness.json": jbytes(guid_evidence),
        "test_inventory.json": jbytes(tests),
        "offline_compile_evidence.json": jbytes(compile_evidence),
        "offline_production_compile_output.txt": production_output.encode("utf-8"),
        "offline_test_compile_output.txt": test_output.encode("utf-8"),
        "offline_camera_route_probe.json": jbytes(probe),
        "offline_camera_route_probe_output.txt": probe_output.encode("utf-8"),
        "protected_files_before_after.json": jbytes(protected),
        "active_candidate_profile_9_prototypes_3_thresholds.json": jbytes(profile),
        "candidate_repository_before_after.json": jbytes({"schema_version": SCHEMA + ".candidate_repository",
                                                           "before": candidate_before, "after": candidate_after,
                                                           "unchanged": True, "status": "PASS"}),
        "packaging_transaction_failure_injection_regression.json": jbytes(transaction),
        "ASSERTIONS.jsonl": assertion_bytes,
        "VALIDATION_REPORT.md": report(project_delta, tests, compile_evidence, protected, wiring),
        "analysis/" + Path(__file__).name: Path(__file__).read_bytes(),
    }
    for relative in MODIFIED_V10:
        current = (root / relative).read_bytes()
        baseline = v10_before[relative]
        leaf = relative.replace("/", "__")
        files["source_before/" + leaf] = baseline
        files["source_after/" + leaf] = current
        files["diff/" + leaf + ".diff"] = unified(baseline, current, relative)
    for relative in NEW_PROJECT_FILES:
        current = (root / relative).read_bytes()
        leaf = relative.replace("/", "__")
        files["source_after/" + leaf] = current
        files["diff/" + leaf + ".diff"] = unified(b"", current, relative)
    for relative in DELIVERY_PROJECT_FILES:
        files["project_files/" + relative] = (root / relative).read_bytes()
    return files, {"candidate_before": candidate_before, "candidate_after": candidate_after,
                   "git_before": git_before, "git_after": git_after, "tests": tests,
                   "protected": protected, "compile": compile_evidence}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    files, summary = build(root)
    files = add_manifest(files)
    validation = root / "Validation"
    names = (NAME, NAME + "_delivery.zip", NAME + "_delivery.zip.sha256")
    result_dir, delivery, detached = commit_outputs(validation, files, None, names, post_verify)
    print(json.dumps({
        "status": "PASS",
        "result_directory": str(result_dir),
        "result_file_count": len(files),
        "delivery_zip": str(delivery),
        "delivery_size_bytes": delivery.stat().st_size,
        "delivery_sha256": sha(delivery.read_bytes()),
        "detached_checksum": str(detached),
        "detached_size_bytes": detached.stat().st_size,
        "detached_sha256": sha(detached.read_bytes()),
        "preserved_tests": 79,
        "new_tests": 8,
        "combined_test_inventory": summary["tests"]["combined_unique_test_count"],
        "unity_gui_webcam_tests": "UNVERIFIED",
        "candidate_repository_file_count_before_after": [summary["candidate_before"]["file_count"],
                                                            summary["candidate_after"]["file_count"]],
        "git_staged_files": summary["git_after"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
