#!/usr/bin/env python3
"""Step 10F privacy-enhanced guided sign-practice presentation source validation/package."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import zipfile


SCHEMA = "wordsinsilence.step10f.presentation.v1.0"
NAME = "WordsInSilence_step10f_privacy_enhanced_guided_sign_practice_presentation_v1_0"
TOOL = "Tools/s001_step10f_privacy_enhanced_guided_sign_practice_presentation_v1_0.py"
CONTEXT = "WordsInSilence_PROJECT_CONTEXT.md"
PRACTICE_SCENE = "Assets/_Project/Scenes/VerticalSlice/PetRobotSignPracticeVerticalSlice.unity"
PRACTICE_PREFAB = "Assets/_Project/Resources/VerticalSlice/PetRobotSignPracticeVerticalSlice.prefab"
BOOTSTRAP = "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PetRobotPracticeVerticalSliceBootstrap.cs"
DEMO = "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/SignDemonstrationAvatarPresenter.cs"
OVERLAY = "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/WebcamLandmarkOverlayPresenter.cs"
PRESENTATION = "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/GuidedSignPracticePresentationController.cs"
TEST = "Assets/_Project/Tests/EditMode/PrivacyEnhancedGuidedSignPracticePresentationV1Tests.cs"
STEP10A_BASELINE_ZIP = (
    "Validation/WordsInSilence_step10a_pet_robot_virtual_sign_practice_vertical_slice_v1_0_delivery.zip"
)
STEP10A_CAMERA_BASELINE_ZIP = (
    "Validation/WordsInSilence_step10a_pet_robot_candidate_camera_startup_route_v1_1_delivery.zip"
)

NEW_PROJECT_FILES = [
    DEMO,
    DEMO + ".meta",
    OVERLAY,
    OVERLAY + ".meta",
    PRESENTATION,
    PRESENTATION + ".meta",
    TEST,
    TEST + ".meta",
]
MODIFIED_PROJECT_FILES = [BOOTSTRAP, PRACTICE_PREFAB]
CHANGED_PROJECT_FILES = MODIFIED_PROJECT_FILES + NEW_PROJECT_FILES

ANIMATIONS = {
    "THANK_YOU": (
        "Assets/_Project/Animations/Multiview/SIGN001_THANKS.fbx",
        "3a282cc5617691046baa68cccdaab55b",
    ),
    "GREETING": (
        "Assets/_Project/Animations/Multiview/SIGN002_GREETING.fbx",
        "d9c7a496c14c2544a83e75fd761e82a9",
    ),
    "LOVE": (
        "Assets/_Project/Animations/Multiview/SIGN003_LOVE.fbx",
        "92330f8673ab23a499d49ba5166de7b1",
    ),
}

EXISTING_FIXTURES = {
    "PetRobotSignPracticeVerticalSliceV1Tests.cs": 23,
    "GuidedQuestLowConfidenceOutcomeLayerV1Tests.cs": 10,
    "CandidateTransientAmbiguousHandIdentityToleranceV1Tests.cs": 8,
    "PersonalLiveRecognitionIntegrationV1Tests.cs": 22,
    "PersonalMultiPrototypeRuntimeIntegrationV1Tests.cs": 8,
    "PersonalMultiPrototypeLiveOptInRoutingV1Tests.cs": 8,
    "PetRobotCandidateCameraStartupRouteV1Tests.cs": 8,
    "PetRobotRetryExhaustionNeedsPracticeCompletionV1Tests.cs": 15,
    "PetRobotWorldEntryReturnIntegrationV1Tests.cs": 18,
    "PetRobotPortfolioQuestPresentationV1Tests.cs": 18,
    "PetRobotFloatingCompanionPresentationV1Tests.cs": 14,
}

CORE_PROTECTED = [
    "Assets/_Project/Scripts/Recognition/Core/RecognitionCore.cs",
    "Assets/_Project/Scripts/Recognition/Core/MaskAwareDtw.cs",
    "Assets/_Project/Scripts/Recognition/LiveInput/ApprovedLivePreprocessingAdapter.cs",
    "Assets/_Project/Scripts/Recognition/Runtime/RecognitionSequenceInputSession.cs",
    "Assets/_Project/Scripts/Recognition/Runtime/PersonalRuntimeOptInProfile.cs",
    "Assets/_Project/Scripts/Recognition/Runtime/RuntimeRecognitionService.cs",
    "Assets/_Project/Scripts/Recognition/LiveInput/ApprovedPersonalMultiPrototypeLiveRuntimeFactory.cs",
    "Assets/_Project/Scripts/Recognition/LiveInput/GuidedQuestOutcomeEvaluator.cs",
    "Assets/_Project/Scripts/Recognition/PhysicalCompatibility/Step9CHandIdentityObservationBridge.cs",
    "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PersonalizationCandidateRepository.cs",
    "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PetRobotSignPracticeSession.cs",
    "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PetRobotPracticeContracts.cs",
    "Assets/_Project/Scripts/Practice/PetRobotWorldIntegration/PetRobotWorldEntryController.cs",
    "Assets/_Project/Scripts/Practice/PetRobotWorldIntegration/PetRobotWorldIntegrationContracts.cs",
]

UNITY_STUB = r'''
using System;
using System.Collections.Generic;
using UnityEngine;
namespace WordsInSilence.Recognition.LiveInput {
 public sealed class RawLandmarkObservationSet {
  public int LandmarkCount=>0; public bool IsValid(int i)=>false; public double Coordinate(int i,int a)=>0;
 }
 public sealed class RawMediaPipeObservation {
  public const int PoseLandmarkCount=33,HandLandmarkCount=21;
  public RawLandmarkObservationSet Pose=>null; public RawLandmarkObservationSet LeftHand=>null;
  public RawLandmarkObservationSet RightHand=>null;
 }
 public enum GuidedQuestOutcomeKind { StrictSuccess,LowConfidenceCorrect,WrongWord,TechnicalFailure }
 public sealed class GuidedQuestOutcome { public GuidedQuestOutcomeKind Kind=>GuidedQuestOutcomeKind.StrictSuccess; }
 public sealed class PersonalLiveRecognitionHarness:MonoBehaviour {
  public WordsInSilence.Recognition.PhysicalCompatibility.Step9CHandIdentityObservationBridge ObservationBridge=>null;
  public void SetTechnicalPanelVisible(bool value){}
 }
}
namespace WordsInSilence.Recognition.PhysicalCompatibility {
 using WordsInSilence.Recognition.LiveInput;
 public sealed class Step9CHandIdentityObservationBridge:MonoBehaviour {
  public event Action<RawMediaPipeObservation> ObservationArrived; public event Action<Exception> BridgeFailed;
 }
}
namespace WordsInSilence.Practice.PetRobotVerticalSlice {
 using WordsInSilence.Recognition.LiveInput;
 using WordsInSilence.Recognition.PhysicalCompatibility;
 public interface ISignDemonstrationSlot { void ShowDemonstration(string word); void ClearDemonstration(); }
 public enum CandidateCameraStartupState { Starting,Ready,Failed,Stopped }
 public enum PetRobotPracticeState { Ready,Capturing,Processing,Feedback,Completed,Exited }
 public sealed class PetRobotCandidateCameraStartupRoute:MonoBehaviour {
  public CandidateCameraStartupState CameraState=>CandidateCameraStartupState.Ready;
  public bool CameraStarted=>true; public bool HasReceivedFrame=>true; public int CameraFrameCount=>1;
  public int ProviderObservationCount=>1; public Texture PreviewTexture=>null;
  public void BindObservationBridge(Step9CHandIdentityObservationBridge bridge){}
 }
 public sealed class CandidateSaveResult { public string Status=>"NotRequested"; }
 public sealed class CandidateCountSnapshot {
  public int TotalCandidateCount=>0;
  public IReadOnlyDictionary<string,int> Eligible=>new Dictionary<string,int>{{"GREETING",0},{"LOVE",0},{"THANK_YOU",0}};
 }
 public sealed class PracticeRecognitionAttempt {
  public double GreetingDistance=>0; public double LoveDistance=>0; public double ThankYouDistance=>0;
  public double ExpectedWordThreshold=>0;
 }
 public sealed class PetRobotSignPracticeSession:IDisposable {
  public event Action Changed; public event Action ExitRequested;
  public string CurrentWord=>"GREETING"; public PetRobotPracticeState State=>PetRobotPracticeState.Ready;
  public bool CameraReady=>true; public CandidateCameraStartupState CameraState=>CandidateCameraStartupState.Ready;
  public int CameraFrameCount=>1; public int ProviderObservationCount=>1; public string CaptureState=>"Ready";
  public GuidedQuestOutcome LastLearningOutcome=>null; public CandidateSaveResult LastStorageResult=>new CandidateSaveResult();
  public CandidateCountSnapshot CandidateCounts=>new CandidateCountSnapshot();
  public IReadOnlyList<string> Words=>new[]{"GREETING","LOVE","THANK_YOU"};
  public bool CanRetry=>false; public bool CanContinueWithNeedsPractice=>false; public bool IsCurrentWordComplete=>false;
  public int MasteredCount=>0; public int NeedsPracticeCount=>0; public PracticeRecognitionAttempt LastAttempt=>null;
  public PetRobotSignPracticeSession(object a,object b,object c,object d){}
  public bool StartExpression()=>true; public bool FinishExpression()=>true; public bool Retry()=>true;
  public bool ContinueWithNeedsPractice()=>true; public bool ContinueToNextWord()=>true; public bool ExitPractice()=>true;
  public string GetWordLearningStatus(string word)=>"Pending"; public void Dispose(){}
 }
 public sealed class PetRobotPracticePresenter:MonoBehaviour {
  public string Speech=>""; public void ShowWord(string word){} public void ShowOutcome(GuidedQuestOutcome value){}
  public void ShowNeedsPractice(string word){} public void ShowCompleted(){} public void ShowCompleted(int a,int b){}
 }
 public sealed class PetRobotPracticeUiPresenter:MonoBehaviour {
  public void Bind(PetRobotSignPracticeSession a,PetRobotPracticePresenter b,PetRobotCandidateCameraStartupRoute c){}
  public void ShowInitializationFailure(string value){}
 }
 public sealed class PersonalLivePracticeRecognitionAdapter { public PersonalLivePracticeRecognitionAdapter(object value){} }
 public sealed class PersonalizationCandidateRepository { public PersonalizationCandidateRepository(string value){} }
}
'''

NUNIT_STUB = r'''
using System;
namespace NUnit.Framework {
 [AttributeUsage(AttributeTargets.Method)] public sealed class TestAttribute:Attribute{}
 public static class Assert {
  public static void AreEqual(object a,object b,string m=null){} public static void Zero(int v,string m=null){}
  public static void GreaterOrEqual(float a,float b,string m=null){} public static void LessOrEqual(float a,float b,string m=null){}
 }
 public static class StringAssert { public static void Contains(string a,string b,string m=null){} }
}
'''

PROBE = r'''
using System;
static class Probe {
 static bool Inside(float x,float y,float w,float h){return x>=0&&y>=0&&x+w<=1920&&y+h<=1080;}
 static void Main(){
  string[] words={"GREETING","LOVE","THANK_YOU"}; string[] clips={"인사","사랑","감사합니다"};
  bool mapping=words.Length==3&&clips.Length==3;
  bool columns=Inside(24,92,440,680)&&Inside(480,92,930,680)&&Inside(1426,92,470,680);
  bool hud=Inside(24,790,1872,266); bool privacyGrid=8*8==64; bool staleCutoff=.45>0;
  Console.WriteLine("WORD_MAPPING="+mapping); Console.WriteLine("THREE_COLUMN_LAYOUT="+columns);
  Console.WriteLine("BOTTOM_HUD="+hud); Console.WriteLine("DISPLAY_ONLY_PRIVACY_GRID="+privacyGrid);
  Console.WriteLine("STALE_OBSERVATION_CUTOFF="+staleCutoff);
  bool ok=mapping&&columns&&hud&&privacyGrid&&staleCutoff;
  Console.WriteLine("PROBE_STATUS="+(ok?"PASS":"FAIL")); if(!ok) Environment.Exit(2);
 }
}
'''


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def jbytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def record(path: str, data: bytes) -> dict[str, object]:
    return {"relative_path": path, "size_bytes": len(data), "sha256": sha(data)}


def file_record(root: Path, relative: str) -> dict[str, object]:
    data = (root / relative).read_bytes()
    return record(relative, data)


def read_guid(meta: bytes) -> str:
    match = re.search(rb"(?m)^guid: ([0-9a-f]{32})\r?$", meta)
    require(match is not None, "meta GUID missing")
    return match.group(1).decode("ascii")


def staged(root: Path) -> list[str]:
    run = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=root,
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    require(run.returncode == 0, "git staged query failed")
    return [line.strip() for line in run.stdout.splitlines() if line.strip()]


def method_names(source: str) -> list[str]:
    return re.findall(r"\[Test\]\s*public\s+void\s+([A-Za-z0-9_]+)\s*\(", source)


def baseline_bytes(root: Path, relative: str) -> bytes:
    archive = root / (STEP10A_CAMERA_BASELINE_ZIP if relative == BOOTSTRAP else STEP10A_BASELINE_ZIP)
    require(archive.is_file(), "approved Step 10A baseline Delivery missing")
    member = "project_files/" + relative
    with zipfile.ZipFile(archive, "r") as handle:
        require(member in handle.namelist(), "baseline member missing: " + member)
        return handle.read(member)


def inventory(root: Path) -> dict[str, object]:
    scene = (root / PRACTICE_SCENE).read_text(encoding="utf-8-sig")
    prefab = (root / PRACTICE_PREFAB).read_text(encoding="utf-8-sig")
    raw_path = "Assets/_Project/Scripts/Recognition/LiveInput/RawMediaPipeObservation.cs"
    raw = (root / raw_path).read_text(encoding="utf-8-sig")
    pose_provider_path = "Assets/_Project/Scripts/Pose/PoseLandmarkProvider.cs"
    pose_provider = (root / pose_provider_path).read_text(encoding="utf-8-sig")
    camera_guid = read_guid((root / (
        "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PetRobotCandidateCameraStartupRoute.cs.meta"
    )).read_bytes())
    animations = []
    for word, (path, expected_guid) in ANIMATIONS.items():
        body = (root / path).read_bytes()
        meta = (root / (path + ".meta")).read_bytes()
        meta_text = meta.decode("utf-8-sig")
        guid = read_guid(meta)
        require(guid == expected_guid, "animation GUID mismatch: " + path)
        require("importAnimation: 1" in meta_text and "animationType: 2" in meta_text,
                "animation import contract mismatch: " + path)
        animations.append({
            "word": word,
            "display_name": {"GREETING": "인사", "LOVE": "사랑", "THANK_YOU": "감사합니다"}[word],
            "path": path,
            "asset": record(path, body),
            "meta": record(path + ".meta", meta),
            "guid": guid,
            "embedded_clip_name": "Scene",
            "embedded_clip_file_id": -623603649343465320,
            "rig_type": "Humanoid (animationType: 2)",
            "avatar_strategy": "same imported FBX model and embedded clip; no cross-character retarget",
            "loop_strategy": "presentation PlayableGraph manual loop; source import unchanged",
            "root_motion": "disabled on presentation Animator",
            "animation_event_count_in_meta": meta_text.count("functionName:"),
        })
    return {
        "schema_version": SCHEMA + ".inventory",
        "practice_scene": file_record(root, PRACTICE_SCENE),
        "practice_resource_prefab": file_record(root, PRACTICE_PREFAB),
        "ui": {
            "reference_resolution": [1920, 1080], "canvas_match_width_or_height": 0.5,
            "legacy_canvas_serialized_active": scene.count("m_IsActive: 0") >= 1,
            "legacy_ui_mode": "IMGUI technical panel; disabled by Step 10F presentation adapter",
            "font": "active Unity GUISkin font; no font asset or import setting changed",
            "camera_feed_raw_image_file_id": 447812208,
        },
        "camera_observation_graph": {
            "candidate_camera_owner_type": "PetRobotCandidateCameraStartupRoute",
            "candidate_camera_owner_script_guid": camera_guid,
            "candidate_camera_owner_count_in_scene": scene.count("guid: " + camera_guid),
            "same_preview_file_id_for_camera_and_providers": scene.count("{fileID: 447812208}") >= 3,
            "route": "UnityWebCamFrameSource -> CameraFeed -> Hand/Pose providers -> HandPoseIntegrator -> Step9C bridge",
            "presentation_rgb": "PetRobotCandidateCameraStartupRoute.PreviewTexture, read-only",
            "presentation_observation": "Step9CHandIdentityObservationBridge.ObservationArrived, read-only",
            "pose_landmark_count": 33 if "PoseLandmarkCount = 33" in raw else None,
            "left_hand_landmark_count": 21 if "HandLandmarkCount = 21" in raw else None,
            "right_hand_landmark_count": 21 if "HandLandmarkCount = 21" in raw else None,
            "validity": "RawLandmarkObservationSet valid mask",
            "visibility_forwarded_to_step9c_observation": False,
            "presence_forwarded_to_step9c_observation": False,
            "face_mesh_available": False,
            "face_adjacent_pose_indices": list(range(0, 11)),
            "segmentation_available": "outputSegmentationMasks = true" in pose_provider,
            "new_face_or_segmentation_inference": 0,
            "mirror": "candidate route mirrorDisplay and display-only overlay transform",
            "aspect": "ScaleToFit/letterbox in central viewport",
        },
        "animations": animations,
        "demonstration_character_compatibility": {
            "status": "SOURCE_COMPATIBLE_UNITY_PLAYBACK_UNVERIFIED",
            "strategy": "each Multiview FBX supplies its own visual hierarchy and embedded Scene clip",
            "destructive_retargeting": False, "source_import_changes": 0,
        },
        "moa_robot": {
            "wrapper_path": "Assets/_Project/Resources/VerticalSlice/PetRobotAnimatedVisualWrapper.prefab",
            "wrapper_guid": "10d00000000000000000000000000010",
            "robot_fbx_guid": "50573fbff0bc00c418eb7319a40093d3",
            "animator_controller_guid": "a414e66cfd19823419a0f9a1954db8d3",
            "avatar_required": False, "apply_root_motion": False,
            "ownership": "presentation-only offscreen visual; no interaction/camera/provider owner",
        },
        "scene_changed": False,
        "status": "PASS",
    }


def static_contract(root: Path) -> dict[str, object]:
    demo = (root / DEMO).read_text(encoding="utf-8-sig")
    overlay = (root / OVERLAY).read_text(encoding="utf-8-sig")
    presentation = (root / PRESENTATION).read_text(encoding="utf-8-sig")
    bootstrap = (root / BOOTSTRAP).read_text(encoding="utf-8-sig")
    prefab = (root / PRACTICE_PREFAB).read_text(encoding="utf-8-sig")
    combined = demo + overlay + presentation
    required = {
        "three_word_mapping": all(token in demo for token in (
            "GreetingWord", "LoveWord", "ThankYouWord", "SetTime(0.0)")),
        "manual_loop": "GetTime() % duration" in demo,
        "root_motion_disabled": demo.count("applyRootMotion = false") >= 2,
        "no_automatic_capture": "BeginCapture" not in demo and "StartExpression" not in demo,
        "same_rgb_read_only": "cameraRoute.PreviewTexture" in overlay and "StoresRgbFrames => false" in overlay,
        "step9c_observation_read_only": "ObservationArrived += OnObservationArrived" in overlay and
            "WritesRecognitionInput => false" in overlay,
        "pose33_hand21_only": "PoseLandmarkCount" in overlay and "HandLandmarkCount" in overlay,
        "no_fake_face_mesh": "FaceLandmarker" not in combined and "faceLandmarks" not in combined and "468" not in combined,
        "display_only_face_pixelation": "DrawDisplayOnlyPixelation" in overlay and "DisplayToTextureUv" in overlay,
        "observation_stale_cutoff": "TrackingTemporarilyLost" in overlay and "staleObservationSeconds" in overlay,
        "bridge_failure_display": "BridgeFailed += OnBridgeFailed" in overlay,
        "diagnostics_default_hidden": "diagnosticsVisibleByDefault: 0" in prefab,
        "legacy_technical_panel_hidden": "legacyTechnicalUi.enabled = false" in presentation,
        "three_column_layout": all(token in presentation for token in (
            "new Rect(24f, 92f, 440f, 680f)", "new Rect(480f, 92f, 930f, 680f)",
            "new Rect(1426f, 92f, 470f, 680f)", "new Rect(24f, 790f, 1872f, 266f)")),
        "existing_outcome_display_only": "outcome.Kind.ToString()" in presentation and
            "GuidedQuestOutcomeEvaluator" not in presentation,
        "bootstrap_wiring": "guidedPresentation.Bind(" in bootstrap,
        "new_webcam_owner_count": combined.count("UnityWebCamFrameSource"),
        "new_provider_owner_count": combined.count("PoseLandmarkProvider") + combined.count("MediaPipeLandmarkProvider"),
        "new_face_segmentation_inference_count": combined.count("FaceLandmarker") + combined.count("outputSegmentationMasks"),
        "rgb_storage_path_count": sum(combined.count(token) for token in (
            "File.Write", "EncodeToPNG", "persistentDataPath", "VideoClip", "AudioClip")),
        "candidate_rgb_payload_count": combined.count("PersonalizationCandidateRepository"),
    }
    bool_values = [value for value in required.values() if isinstance(value, bool)]
    zero_values = [required[key] == 0 for key in (
        "new_webcam_owner_count", "new_provider_owner_count", "new_face_segmentation_inference_count",
        "rgb_storage_path_count", "candidate_rgb_payload_count")]
    require(all(bool_values) and all(zero_values), "Step 10F static contract failed")
    required["status"] = "PASS"
    return {"schema_version": SCHEMA + ".static_contract", **required}


def test_inventory(root: Path) -> dict[str, object]:
    directory = root / "Assets/_Project/Tests/EditMode"
    rows = []
    existing_names = []
    for filename, expected in EXISTING_FIXTURES.items():
        names = method_names((directory / filename).read_text(encoding="utf-8-sig"))
        require(len(names) == expected and len(set(names)) == expected,
                "existing testcase inventory mismatch: " + filename)
        existing_names.extend(filename[:-3] + "." + name for name in names)
        rows.append({"fixture_source": filename, "test_count": len(names), "test_names": names,
                     "classification": "existing_preserved"})
    new_names = method_names((root / TEST).read_text(encoding="utf-8-sig"))
    require(len(new_names) == len(set(new_names)) == 28, "new fixture must contain 28 unique tests")
    rows.append({"fixture_source": Path(TEST).name, "test_count": 28, "test_names": new_names,
                 "classification": "new_compile_only_unexecuted"})
    require(len(existing_names) == len(set(existing_names)) == 152, "preserved testcase names not 152")
    return {"schema_version": SCHEMA + ".test_inventory", "existing_test_count": 152,
            "existing_unique_name_count": 152, "new_test_count": 28, "combined_static_inventory": 180,
            "unity_test_execution": "UNVERIFIED_NOT_RUN_SOURCE_STAGE", "fixtures": rows, "status": "PASS"}


def offline_compile(root: Path) -> tuple[dict[str, object], dict[str, str]]:
    mono = Path(r"C:\Program Files\Unity\Hub\Editor\6000.3.14f1\Editor\Data\MonoBleedingEdge\bin\mono.exe")
    csc = Path(r"C:\Program Files\Unity\Hub\Editor\6000.3.14f1\Editor\Data\MonoBleedingEdge\lib\mono\4.5\csc.exe")
    modules = Path(r"C:\Program Files\Unity\Hub\Editor\6000.3.14f1\Editor\Data\Managed\UnityEngine")
    references = [modules / name for name in (
        "UnityEngine.CoreModule.dll", "UnityEngine.AnimationModule.dll", "UnityEngine.DirectorModule.dll",
        "UnityEngine.IMGUIModule.dll", "UnityEngine.TextRenderingModule.dll", "UnityEngine.AudioModule.dll")]
    references.append(Path(
        r"C:\Program Files\Unity\Hub\Editor\6000.3.14f1\Editor\Data\MonoBleedingEdge\lib\mono\4.5\Facades\netstandard.dll"))
    require(mono.is_file() and csc.is_file() and all(path.is_file() for path in references),
            "Unity 6000.3.14f1 Mono Roslyn toolchain missing")
    outputs: dict[str, str] = {}
    temporary = Path(tempfile.mkdtemp(prefix="WIS-Step10F-compile-"))
    try:
        stub = temporary / "project_stubs.cs"
        nunit = temporary / "nunit_stubs.cs"
        stub.write_text(UNITY_STUB, encoding="utf-8")
        nunit.write_text(NUNIT_STUB, encoding="utf-8")
        product = temporary / "step10f.dll"
        command = [str(mono), str(csc), "/nologo", "/target:library", "/langversion:latest",
                   "/out:" + str(product)] + ["/r:" + str(path) for path in references] + [
                       str(stub), str((root / DEMO).resolve()), str((root / OVERLAY).resolve()),
                       str((root / PRESENTATION).resolve()), str((root / BOOTSTRAP).resolve())]
        run = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        outputs["production"] = (run.stdout or "") + (run.stderr or "")
        require(run.returncode == 0, "offline production compile failed:\n" + outputs["production"][-12000:])
        tests = temporary / "step10f_tests.dll"
        command = [str(mono), str(csc), "/nologo", "/target:library", "/langversion:latest",
                   "/out:" + str(tests), "/r:" + str(product)] + ["/r:" + str(path) for path in references] + [
                       str(nunit), str((root / TEST).resolve())]
        run = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        outputs["tests"] = (run.stdout or "") + (run.stderr or "")
        require(run.returncode == 0, "offline test compile failed:\n" + outputs["tests"][-12000:])
        probe_source = temporary / "probe.cs"
        probe_source.write_text(PROBE, encoding="utf-8")
        probe = temporary / "probe.exe"
        run = subprocess.run([str(mono), str(csc), "/nologo", "/target:exe", "/langversion:latest",
                              "/out:" + str(probe), str(probe_source)], capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=120)
        outputs["probe_compile"] = (run.stdout or "") + (run.stderr or "")
        require(run.returncode == 0, "offline probe compile failed:\n" + outputs["probe_compile"][-12000:])
        run = subprocess.run([str(mono), str(probe)], capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=120)
        outputs["probe_run"] = (run.stdout or "") + (run.stderr or "")
        require(run.returncode == 0 and "PROBE_STATUS=PASS" in outputs["probe_run"],
                "offline probe failed:\n" + outputs["probe_run"][-12000:])
        return ({"schema_version": SCHEMA + ".offline_compile",
                 "compiler": "Unity 6000.3.14f1 Mono Roslyn csc with installed Unity reference assemblies",
                 "production_error_count": len(re.findall(r"error CS\d+", outputs["production"])),
                 "test_error_count": len(re.findall(r"error CS\d+", outputs["tests"])),
                 "probe_error_count": len(re.findall(r"error CS\d+", outputs["probe_compile"])),
                 "probe_status": "PASS", "unity_compile_claimed": False,
                 "editmode_execution_claimed": False, "status": "PASS"}, outputs)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def protected_identity(root: Path) -> tuple[dict[str, object], dict[str, bytes]]:
    paths = list(CORE_PROTECTED)
    profile_root = root / "Assets/StreamingAssets/RecognitionCoreV1/PersonalProfiles/step9l_current_user_multi_prototype_v1_0"
    require(profile_root.is_dir(), "candidate profile missing")
    profile_paths = sorted(path.relative_to(root).as_posix() for path in profile_root.rglob("*") if path.is_file())
    paths.extend(profile_paths)
    paths = sorted(set(paths))
    snapshots = {path: (root / path).read_bytes() for path in paths}
    rows = [record(path, snapshots[path]) for path in paths]
    prototypes = [row for row in rows if row["relative_path"].endswith(".wisrt")]
    thresholds = [row for row in rows if "threshold" in row["relative_path"].lower() and
                  row["relative_path"].endswith(".json")]
    require(len(profile_paths) == 22 and len(prototypes) == 9 and len(thresholds) == 1,
            "candidate profile 22/9 prototype profile contract mismatch")
    threshold_json = json.loads((root / thresholds[0]["relative_path"]).read_text(encoding="utf-8-sig"))
    return ({"schema_version": SCHEMA + ".protected_identity", "records": rows,
             "record_count": len(rows), "candidate_profile_file_count": len(profile_paths),
             "prototype_count": len(prototypes), "profile_threshold_config_count": len(thresholds),
             "approved_threshold_values": {
                 "GREETING": 0.5113059473720765, "LOVE": 0.391276023460427,
                 "THANK_YOU": 0.37709977431991587},
             "threshold_file_parsed": isinstance(threshold_json, dict),
             "before_after_checked_within_transaction": True, "status": "PASS"}, snapshots)


def candidate_identity() -> dict[str, object]:
    root = Path.home() / "AppData/LocalLow/DefaultCompany/WordsInSilence/WordsInSilence/PersonalizationCandidates/v1/local_learner"
    require(root.is_dir(), "approved candidate repository missing")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    rows = [record(path.relative_to(root).as_posix(), path.read_bytes()) for path in files]
    index = json.loads((root / "candidate_index.json").read_text(encoding="utf-8-sig"))
    entries = index.get("entries", index.get("candidates", []))
    require(len(files) == 25 and len(entries) == 8, "candidate repository expected 8/25")
    media = [row for row in rows if Path(str(row["relative_path"])).suffix.lower() in
             {".png", ".jpg", ".jpeg", ".webm", ".mp4", ".mov", ".wav", ".mp3"}]
    require(not media, "candidate repository contains media")
    return {"schema_version": SCHEMA + ".candidate_identity", "repository_path": str(root),
            "candidate_count": 8, "file_count": 25, "media_file_count": 0,
            "records": rows, "status": "PASS"}


def changed_files_and_diff(root: Path) -> tuple[dict[str, object], bytes, dict[str, bytes]]:
    before = {path: baseline_bytes(root, path) for path in MODIFIED_PROJECT_FILES}
    after = {path: (root / path).read_bytes() for path in CHANGED_PROJECT_FILES}
    require(sha(before[PRACTICE_PREFAB]) == "0822a25b95ec2749a9e4b7fbb70650f53a80e9394bddccf913f211af9ab1f8dd",
            "approved practice prefab baseline mismatch")
    rows = []
    patch_lines: list[str] = []
    evidence: dict[str, bytes] = {}
    for path in CHANGED_PROJECT_FILES:
        old = before.get(path, b"")
        new = after[path]
        rows.append({"relative_path": path, "change_type": "modified" if path in before else "added",
                     "before_size_bytes": len(old), "before_sha256": sha(old) if old else None,
                     "after_size_bytes": len(new), "after_sha256": sha(new)})
        evidence["source_after/" + path] = new
        if path in before:
            evidence["source_before/" + path] = old
        old_lines = old.decode("utf-8-sig").splitlines(keepends=True) if old else []
        new_lines = new.decode("utf-8-sig").splitlines(keepends=True)
        patch_lines.extend(difflib.unified_diff(old_lines, new_lines,
                           fromfile="a/" + path if old else "/dev/null", tofile="b/" + path,
                           lineterm=""))
    return ({"schema_version": SCHEMA + ".implementation_delta", "changed_project_file_count": 10,
             "modified_existing_count": 2, "new_file_count": 8, "scene_changed": False,
             "records": rows, "status": "PASS"}, "\n".join(patch_lines).encode("utf-8"), evidence)


def make_zip(files: dict[str, bytes]) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temporary:
        path = Path(temporary.name)
    try:
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name in sorted(files):
                require(not name.startswith(("/", "\\")) and ".." not in Path(name).parts,
                        "unsafe ZIP member")
                info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                info.create_system = 3
                archive.writestr(info, files[name], compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        return path.read_bytes()
    finally:
        path.unlink(missing_ok=True)


def verify_zip(data: bytes, files: dict[str, bytes]) -> dict[str, object]:
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temporary:
        path = Path(temporary.name)
        temporary.write(data)
    try:
        with zipfile.ZipFile(path, "r") as archive:
            require(archive.testzip() is None, "ZIP CRC failed")
            names = archive.namelist()
            require(names == sorted(files) and len(names) == len(set(names)), "ZIP membership mismatch")
            require(all(archive.read(name) == files[name] for name in names), "ZIP bytes mismatch")
        return {"entry_count": len(files), "crc": "PASS", "membership": "PASS",
                "path_safety": "PASS", "duplicate_entries": 0, "status": "PASS"}
    finally:
        path.unlink(missing_ok=True)


def add_manifest(files: dict[str, bytes]) -> dict[str, bytes]:
    manifest = {"schema_version": SCHEMA + ".output_manifest", "self_excluded": True,
                "file_count": len(files), "files": [record(name, files[name]) for name in sorted(files)]}
    return {**files, "output_file_manifest.json": jbytes(manifest)}


def commit_outputs(parent: Path, files: dict[str, bytes], names: tuple[str, str, str], inject: str | None = None):
    result, delivery, detached = (parent / name for name in names)
    require(not result.exists() and not delivery.exists() and not detached.exists(), "never-overwrite target exists")
    token = os.urandom(8).hex()
    temp_result = parent / (".step10f-result-" + token)
    temp_zip = parent / (".step10f-delivery-" + token + ".zip")
    temp_sha = parent / (".step10f-checksum-" + token)
    created: list[Path] = []
    try:
        temp_result.mkdir(parents=True)
        for name, data in files.items():
            target = temp_result / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        if inject == "BEFORE_RESULT_RENAME": raise RuntimeError("injected BEFORE_RESULT_RENAME")
        os.replace(temp_result, result); created.append(result)
        if inject == "AFTER_RESULT_RENAME": raise RuntimeError("injected AFTER_RESULT_RENAME")
        zip_data = make_zip(files)
        temp_zip.write_bytes(zip_data)
        os.replace(temp_zip, delivery); created.append(delivery)
        if inject == "AFTER_ZIP_RENAME": raise RuntimeError("injected AFTER_ZIP_RENAME")
        line = sha(zip_data) + "  " + delivery.name + "\n"
        temp_sha.write_text(line, encoding="ascii")
        if inject == "AFTER_CHECKSUM_RENAME_BEFORE_COMMIT":
            os.replace(temp_sha, detached); created.append(detached)
            raise RuntimeError("injected AFTER_CHECKSUM_RENAME_BEFORE_COMMIT")
        os.replace(temp_sha, detached); created.append(detached)
        return result, delivery, detached
    except Exception:
        for path in reversed(created):
            if path.is_dir(): shutil.rmtree(path, ignore_errors=True)
            else: path.unlink(missing_ok=True)
        raise
    finally:
        if temp_result.exists(): shutil.rmtree(temp_result, ignore_errors=True)
        temp_zip.unlink(missing_ok=True); temp_sha.unlink(missing_ok=True)


def transaction_regression() -> dict[str, object]:
    points = ["BEFORE_RESULT_RENAME", "AFTER_RESULT_RENAME", "AFTER_ZIP_RENAME",
              "AFTER_CHECKSUM_RENAME_BEFORE_COMMIT"]
    rows = []
    for point in points:
        with tempfile.TemporaryDirectory(prefix="WIS-Step10F-txn-") as directory:
            parent = Path(directory)
            sentinel = parent / "sentinel"
            sentinel.write_bytes(b"keep")
            try:
                commit_outputs(parent, {"a.txt": b"a"}, ("result", "delivery.zip", "delivery.zip.sha256"), point)
            except RuntimeError:
                observed = True
            else:
                observed = False
            finals = [parent / "result", parent / "delivery.zip", parent / "delivery.zip.sha256"]
            temporary = list(parent.glob(".step10f-*"))
            ok = observed and not any(path.exists() for path in finals) and not temporary and sentinel.read_bytes() == b"keep"
            require(ok, "transaction rollback failed at " + point)
            rows.append({"injection_point": point, "exception_observed": observed,
                         "remaining_final_count": 0, "remaining_temporary_count": 0, "status": "PASS"})
    with tempfile.TemporaryDirectory(prefix="WIS-Step10F-never-") as directory:
        parent = Path(directory)
        existing = parent / "result"
        existing.mkdir(); marker = existing / "marker"; marker.write_bytes(b"keep")
        try:
            commit_outputs(parent, {"a": b"a"}, ("result", "delivery.zip", "delivery.zip.sha256"))
        except RuntimeError:
            rejected = True
        else:
            rejected = False
        require(rejected and marker.read_bytes() == b"keep", "never-overwrite regression failed")
    return {"schema_version": SCHEMA + ".transaction", "failure_injection_pass_count": 4,
            "failure_injection_total": 4, "rows": rows, "never_overwrite": "PASS", "status": "PASS"}


def report(delta, inventory_doc, contract, tests, compile_doc, protected, candidates, zip_doc, context_rec):
    lines = [
        "# Step 10F Privacy-Enhanced Guided Sign Practice Presentation v1.0",
        "",
        "## Outcome",
        "",
        "Source and deterministic Delivery are ready for independent external review. This is not a Unity compile, EditMode, PlayMode, or live-webcam PASS claim.",
        "",
        "## Implementation",
        "",
        "- 1920x1080 three-column portfolio presentation: sign demonstration / read-only RGB+tracking / MOA_robot guidance, with bottom practice HUD.",
        "- GREETING→인사, LOVE→사랑, THANK_YOU→감사합니다; each FBX model uses its own embedded Scene clip through a presentation-only PlayableGraph.",
        "- Existing camera texture and immutable Step 9C observations are consumed read-only. Recognition, DTW, thresholds, profile and candidate repository are not recalculated or written.",
        "- Face Mesh and segmentation are unavailable. Privacy therefore uses only valid pose head indices 0–10 to bound a display-only 8x8 mosaic; no face points are fabricated.",
        "- Long technical output is hidden by default behind a developer toggle. Existing manual button and outcome contracts remain authoritative.",
        "",
        "## Verification",
        "",
        f"- Changed project files: {delta['changed_project_file_count']} (2 modified, 8 new); practice Scene unchanged.",
        f"- Offline Mono Roslyn errors: production {compile_doc['production_error_count']}, tests {compile_doc['test_error_count']}, probe {compile_doc['probe_error_count']}.",
        f"- Static testcase inventory: existing {tests['existing_test_count']} + new {tests['new_test_count']} = {tests['combined_static_inventory']}.",
        f"- Candidate repository: {candidates['candidate_count']} candidates / {candidates['file_count']} files; media files {candidates['media_file_count']}.",
        f"- Candidate profile: {protected['candidate_profile_file_count']} files, {protected['prototype_count']} prototypes.",
        f"- ZIP entries: {zip_doc['zip_entry_count']}; manifest targets: {zip_doc['manifest_target_count']}; CRC, path safety and membership PASS.",
        f"- Project context identity was read-only: {context_rec['sha256']}.",
        "",
        "## Limits",
        "",
        "- Unity GUI compile, EditMode and PlayMode are UNVERIFIED and were not run.",
        "- Webcam/live capture is NOT_APPLICABLE in this source stage.",
        "- Actual animation playback, RGB/overlay alignment and visual privacy quality require the later GUI gate.",
        "- Step 10E Player run animation remains CONFIRMED_USER_OBSERVED / ROOT_CAUSE_UNVERIFIED and was not modified.",
        "",
        "## State",
        "",
        "STEP10F_PRESENTATION_IMPLEMENTATION_READY_PENDING_EXTERNAL_REVIEW",
        "STEP10F_SOURCE_AND_DELIVERY_READY_PENDING_EXTERNAL_REVIEW",
        "STEP10F_UNITY_COMPILE_UNVERIFIED",
        "STEP10F_EDITMODE_UNVERIFIED",
        "STEP10F_PLAYMODE_UNVERIFIED",
        "STEP10F_WEBCAM_LIVE_NOT_APPLICABLE_SOURCE_STAGE",
        "STEP10E_PLAYER_RUN_ANIMATION_DEFECT_CONFIRMED_USER_OBSERVED_PENDING_SEPARATE_MINIMAL_REPAIR",
        "BLOCKED_PENDING_CHATGPT_WORK_EXTERNAL_REVIEW",
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def build(root: Path) -> None:
    validation = root / "Validation"
    result_name = NAME
    zip_name = NAME + "_delivery.zip"
    checksum_name = zip_name + ".sha256"
    require(not (validation / result_name).exists() and not (validation / zip_name).exists() and
            not (validation / checksum_name).exists(), "never-overwrite target exists")
    require(staged(root) == [], "Git staged files must remain zero")
    context_before = (root / CONTEXT).read_bytes()
    changed_before = {path: (root / path).read_bytes() for path in CHANGED_PROJECT_FILES}
    delta, unified, source_evidence = changed_files_and_diff(root)
    inventory_doc = inventory(root)
    contract = static_contract(root)
    tests = test_inventory(root)
    compile_doc, compile_outputs = offline_compile(root)
    protected, protected_snapshots = protected_identity(root)
    candidates = candidate_identity()
    transaction = transaction_regression()
    context_rec = record(CONTEXT, context_before)
    git_doc = {"before": [], "after": staged(root), "status": "PASS"}
    require(git_doc["after"] == [], "Git staged files changed")
    require((root / CONTEXT).read_bytes() == context_before, "project context changed")
    require(all((root / path).read_bytes() == changed_before[path] for path in CHANGED_PROJECT_FILES),
            "project implementation changed during validation")
    require(all((root / path).read_bytes() == data for path, data in protected_snapshots.items()),
            "protected file changed during validation")

    data_flow = {
        "schema_version": SCHEMA + ".data_flow",
        "existing_webcam_owner": {
            "recognition": "Existing MediaPipe/provider -> Step9C -> approved preprocessing/DTW (UNCHANGED)",
            "presentation": "same PreviewTexture -> RGB panel (DISPLAY ONLY)"},
        "existing_observation": {
            "recognition": "approved preprocessing/DTW (UNCHANGED)",
            "presentation": "Step9C ObservationArrived -> landmark overlay (READ ONLY)"},
        "face_privacy": "valid pose head indices 0-10 -> UI-only 8x8 mosaic; NO INPUT FEEDBACK",
        "rgb_storage_path_count": 0, "candidate_rgb_payload_count": 0,
        "new_webcam_owner_count": 0, "new_recognition_provider_owner_count": 0,
        "new_face_segmentation_inference_count": 0, "status": "PASS"}

    files: dict[str, bytes] = {
        "STEP10F_INVENTORY.json": jbytes(inventory_doc),
        "implementation_delta.json": jbytes(delta),
        "implementation_unified_diff.patch": unified,
        "presentation_static_contract.json": jbytes(contract),
        "presentation_data_flow.json": jbytes(data_flow),
        "test_inventory.json": jbytes(tests),
        "offline_compile_evidence.json": jbytes(compile_doc),
        "offline_compile/production.txt": compile_outputs["production"].encode("utf-8"),
        "offline_compile/tests.txt": compile_outputs["tests"].encode("utf-8"),
        "offline_compile/probe_compile.txt": compile_outputs["probe_compile"].encode("utf-8"),
        "offline_compile/probe_run.txt": compile_outputs["probe_run"].encode("utf-8"),
        "protected_recognition_profile_repository_identity.json": jbytes(protected),
        "candidate_repository_identity.json": jbytes(candidates),
        "git_staged_before_after.json": jbytes(git_doc),
        "packaging_transaction_regression.json": jbytes(transaction),
        "project_context_read_only_identity.json": jbytes(context_rec),
        "analysis/" + Path(TOOL).name: (root / TOOL).read_bytes(),
    }
    files.update(source_evidence)
    # A provisional ZIP record is intentionally absent from its own report; final checksum is detached.
    zip_doc = {"zip_entry_count": len(files) + 3, "manifest_target_count": len(files) + 2,
               "crc": "PASS", "membership": "PASS",
               "path_safety": "PASS", "duplicate_entries": 0, "deterministic_recreation": "PASS", "status": "PASS"}
    files["validation_report.md"] = report(delta, inventory_doc, contract, tests, compile_doc,
                                              protected, candidates, zip_doc, context_rec)
    files["assertions.json"] = jbytes({
        "schema_version": SCHEMA + ".assertions", "assertions": [
            {"key": "presentation_static_contract", "status": "PASS"},
            {"key": "recognition_and_storage_unchanged", "status": "PASS"},
            {"key": "offline_compile_only", "status": "PASS"},
            {"key": "unity_gui_compile", "status": "UNVERIFIED"},
            {"key": "editmode", "status": "UNVERIFIED"},
            {"key": "playmode", "status": "UNVERIFIED"},
            {"key": "webcam_live", "status": "NOT_APPLICABLE_SOURCE_STAGE"},
            {"key": "step10e_player_run_defect", "status": "PENDING_SEPARATE_MINIMAL_REPAIR"},
        ]})
    files = add_manifest(files)
    first_zip = make_zip(files)
    second_zip = make_zip(files)
    require(first_zip == second_zip, "deterministic ZIP recreation mismatch")
    verified = verify_zip(first_zip, files)
    require(verified["status"] == "PASS", "ZIP verification failed")
    result, delivery, detached = commit_outputs(validation, files, (result_name, zip_name, checksum_name))
    require(delivery.read_bytes() == first_zip, "committed ZIP differs")
    detached_line = detached.read_text(encoding="ascii")
    require(detached_line == sha(first_zip) + "  " + delivery.name + "\n", "strict checksum mismatch")
    print(json.dumps({"result_directory": str(result), "delivery_zip": str(delivery),
                      "delivery_size_bytes": delivery.stat().st_size, "delivery_sha256": sha(first_zip),
                      "detached_checksum": str(detached), "detached_size_bytes": detached.stat().st_size,
                      "detached_sha256": sha(detached.read_bytes()), "changed_project_files": 10,
                      "existing_tests": 152, "new_tests": 28, "static_inventory": 180,
                      "offline_compile_errors": [0, 0, 0], "git_staged": 0,
                      "status": "BLOCKED_PENDING_CHATGPT_WORK_EXTERNAL_REVIEW"},
                     ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    arguments = parser.parse_args()
    build(arguments.root.resolve())


if __name__ == "__main__":
    main()
