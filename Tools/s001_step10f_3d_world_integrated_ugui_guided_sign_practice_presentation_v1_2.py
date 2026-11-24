#!/usr/bin/env python3
"""Build the deterministic Step 10F v1.2 source-stage Delivery.

This tool performs source/YAML inventory, offline Roslyn compilation, immutable
baseline comparisons, and transactional packaging. It never invokes Unity.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import zipfile


SCHEMA = "wordsinsilence.step10f.presentation.v1.2"
NAME = "WordsInSilence_step10f_3d_world_integrated_ugui_guided_sign_practice_presentation_v1_2"
TOOL = "Tools/s001_step10f_3d_world_integrated_ugui_guided_sign_practice_presentation_v1_2.py"
HELPER = "Tools/s001_step10f_privacy_enhanced_guided_sign_practice_presentation_v1_0.py"
CONTEXT = "WordsInSilence_PROJECT_CONTEXT.md"
PRACTICE_SCENE = "Assets/_Project/Scenes/VerticalSlice/PetRobotSignPracticeVerticalSlice.unity"
PRACTICE_PREFAB = "Assets/_Project/Resources/VerticalSlice/PetRobotSignPracticeVerticalSlice.prefab"
ROBOT_WRAPPER = "Assets/_Project/Resources/VerticalSlice/PetRobotAnimatedVisualWrapper.prefab"
BOOTSTRAP = "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PetRobotPracticeVerticalSliceBootstrap.cs"
DEMO = "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/SignDemonstrationAvatarPresenter.cs"
OVERLAY = "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/WebcamLandmarkOverlayPresenter.cs"
TRACKING = "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/WebcamTrackingGraphic.cs"
PRESENTATION = "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/GuidedSignPracticePresentationController.cs"
TEST = "Assets/_Project/Tests/EditMode/PrivacyEnhancedGuidedSignPracticePresentationV1Tests.cs"
SHADER = "Assets/_Project/Resources/VerticalSlice/DisplayOnlyMildBlur.shader"
MATERIAL = "Assets/_Project/Resources/VerticalSlice/DisplayOnlyMildBlur.mat"
V11_ZIP = "Validation/WordsInSilence_step10f_privacy_enhanced_guided_sign_practice_presentation_v1_1_delivery.zip"
V11_CAMERA_EVIDENCE = (
    "Validation/WordsInSilence_step10f_privacy_enhanced_guided_sign_practice_presentation_"
    "v1_1_gui_unity_gate_evidence_v1_0"
)
STEP10D_ZIP = "Validation/WordsInSilence_step10d_scene_placed_player_moa_robot_wiring_and_gui_gate_v1_4_delivery.zip"

WRAPPERS = {
    "GREETING": (
        "Assets/_Project/Resources/VerticalSlice/SignDemonstrationGreetingWrapper.prefab",
        "10f12000000000000000000000000001",
        "Assets/_Project/Animations/Multiview/SIGN002_GREETING.fbx",
        "d9c7a496c14c2544a83e75fd761e82a9",
        "115319acc4d1e0aadfe6f317fedc795ef5ee2d2776e49f1ab8f4898ed1c2c0f4",
    ),
    "LOVE": (
        "Assets/_Project/Resources/VerticalSlice/SignDemonstrationLoveWrapper.prefab",
        "10f12000000000000000000000000002",
        "Assets/_Project/Animations/Multiview/SIGN003_LOVE.fbx",
        "92330f8673ab23a499d49ba5166de7b1",
        "7ab9e62a01918b16bad5a86c505c72ba05822ab54c1a09de6519020322404188",
    ),
    "THANK_YOU": (
        "Assets/_Project/Resources/VerticalSlice/SignDemonstrationThankYouWrapper.prefab",
        "10f12000000000000000000000000003",
        "Assets/_Project/Animations/Multiview/SIGN001_THANKS.fbx",
        "3a282cc5617691046baa68cccdaab55b",
        "5e1b5b8b80fd9a7a845275624189714e557c1d2e51e71109889336bc57378655",
    ),
}

MODIFIED = [BOOTSTRAP, DEMO, OVERLAY, PRESENTATION, PRACTICE_PREFAB, TEST]
ADDED = [
    TRACKING, TRACKING + ".meta",
    SHADER, SHADER + ".meta",
    MATERIAL, MATERIAL + ".meta",
]
for _wrapper in WRAPPERS.values():
    ADDED.extend([_wrapper[0], _wrapper[0] + ".meta"])
CHANGED = MODIFIED + ADDED
SUPPORTING_METAS = [path + ".meta" for path in [BOOTSTRAP, DEMO, OVERLAY, PRESENTATION, PRACTICE_PREFAB, TEST]]


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
    return record(relative, (root / relative).read_bytes())


def read_guid(meta: bytes) -> str:
    match = re.search(rb"(?m)^guid: ([0-9a-f]{32})\r?$", meta)
    require(match is not None, "meta GUID missing")
    return match.group(1).decode("ascii")


def staged(root: Path) -> list[str]:
    run = subprocess.run(
        ["git", "diff", "--cached", "--name-only"], cwd=root,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    require(run.returncode == 0, "git staged query failed")
    return [line.strip() for line in run.stdout.splitlines() if line.strip()]


def method_names(source: str) -> list[str]:
    return re.findall(r"\[Test\]\s*public\s+void\s+([A-Za-z0-9_]+)\s*\(", source)


def zip_json(path: Path, member: str) -> object:
    with zipfile.ZipFile(path, "r") as archive:
        require(member in archive.namelist(), "ZIP member missing: " + member)
        return json.loads(archive.read(member).decode("utf-8-sig"))


def zip_bytes(path: Path, member: str) -> bytes:
    with zipfile.ZipFile(path, "r") as archive:
        require(member in archive.namelist(), "ZIP member missing: " + member)
        return archive.read(member)


def load_helper(root: Path):
    path = root / HELPER
    require(path.is_file(), "deterministic packaging helper missing")
    spec = importlib.util.spec_from_file_location("step10f_v10_helper", path)
    require(spec is not None and spec.loader is not None, "cannot load packaging helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def baseline_source(root: Path, relative: str) -> bytes:
    if relative == DEMO:
        repaired = root / V11_CAMERA_EVIDENCE / "source_after" / DEMO
        data = repaired.read_bytes()
        require(sha(data) == "85e734d563f11bed2d6f405a82ec66832d5191b59f1cf4dbf983ef7351f9e8c8",
                "post-camera-repair presenter baseline mismatch")
        return data
    data = zip_bytes(root / V11_ZIP, "project_files/" + relative)
    expected = {
        BOOTSTRAP: "5e0e7578d7f27cecd2ed0a7ca5b7f24a537c551575cef3407aefd5d41ca8f22a",
        OVERLAY: "9840bb9a888f1e16966fa91e0fcc4ae74e566bae74f5eb2ff7fec9f319c0d914",
        PRESENTATION: "5cfdcb11c222a2c8da740cde9c9389d89b2c3dc8f1d2fc65e1aa94ea0db47127",
        PRACTICE_PREFAB: "7b61b8d00c6b7595209218c1d0c90da72d2549feef2506a0aee8e98ec049a037",
        TEST: "a8df258f95f8fdd6ed7802d1f0a543d696713bf82a9fb6c6f7d0eb3435aaefdb",
    }[relative]
    require(sha(data) == expected, "v1.1 baseline mismatch: " + relative)
    return data


def implementation_delta(root: Path) -> tuple[dict[str, object], bytes, dict[str, bytes]]:
    rows = []
    patch_lines: list[str] = []
    evidence: dict[str, bytes] = {}
    for relative in CHANGED:
        new = (root / relative).read_bytes()
        old = baseline_source(root, relative) if relative in MODIFIED else b""
        rows.append({
            "relative_path": relative,
            "change_type": "modified" if old else "added",
            "before_size_bytes": len(old),
            "before_sha256": sha(old) if old else None,
            "after_size_bytes": len(new),
            "after_sha256": sha(new),
        })
        evidence["project_files/current/" + relative] = new
        if old:
            evidence["project_files/v1_1_post_repair_baseline/" + relative] = old
        old_lines = old.decode("utf-8-sig").splitlines(keepends=True) if old else []
        new_lines = new.decode("utf-8-sig").splitlines(keepends=True)
        patch_lines.extend(difflib.unified_diff(
            old_lines, new_lines,
            fromfile="a/" + relative if old else "/dev/null",
            tofile="b/" + relative, lineterm="",
        ))
    for relative in SUPPORTING_METAS:
        evidence["project_files/current/" + relative] = (root / relative).read_bytes()
    evidence["project_files/current/" + PRACTICE_SCENE] = (root / PRACTICE_SCENE).read_bytes()
    evidence["project_files/current/" + PRACTICE_SCENE + ".meta"] = (root / (PRACTICE_SCENE + ".meta")).read_bytes()
    return ({
        "schema_version": SCHEMA + ".implementation_delta",
        "changed_project_file_count": len(CHANGED),
        "modified_existing_count": len(MODIFIED),
        "added_count": len(ADDED),
        "scene_changed": False,
        "scene_change_reason": "runtime-loaded practice prefab owns the world stage and project-owned uGUI Canvas",
        "records": rows,
        "status": "SOURCE_DELTA_RECORDED",
    }, "\n".join(patch_lines).encode("utf-8"), evidence)


def presentation_inventory(root: Path) -> dict[str, object]:
    scene = (root / PRACTICE_SCENE).read_text(encoding="utf-8-sig")
    prefab = (root / PRACTICE_PREFAB).read_text(encoding="utf-8-sig")
    robot = (root / ROBOT_WRAPPER).read_text(encoding="utf-8-sig")
    require("m_Name: Main Camera" in scene and "field of view: 60" in scene, "Main Camera inventory mismatch")
    require("m_LocalPosition: {x: 0, y: 1, z: -10}" in scene, "Main Camera transform mismatch")
    require("m_Name: EventSystem" in scene and "m_ReferenceResolution: {x: 1920, y: 1080}" in scene,
            "Scene UI inventory mismatch")
    require("m_SourcePrefab: {fileID: 100100000, guid: 50573fbff0bc00c418eb7319a40093d3" in robot,
            "robot wrapper nested FBX mismatch")
    animations = []
    for word, (_, _, fbx, guid, expected_sha) in WRAPPERS.items():
        body = (root / fbx).read_bytes()
        meta = (root / (fbx + ".meta")).read_bytes()
        meta_text = meta.decode("utf-8-sig")
        require(sha(body) == expected_sha and read_guid(meta) == guid, "animation identity mismatch: " + word)
        require("importAnimation: 1" in meta_text and "animationType: 2" in meta_text and
                "avatarSetup: 0" in meta_text and "clipAnimations: []" in meta_text,
                "Humanoid importer contract mismatch: " + word)
        require("globalScale: 1" in meta_text and "useFileScale: 1" in meta_text,
                "model scale importer contract mismatch: " + word)
        animations.append({
            "word": word,
            "display_name": {"GREETING": "인사", "LOVE": "사랑", "THANK_YOU": "감사합니다"}[word],
            "path": fbx,
            "guid": guid,
            "asset": record(fbx, body),
            "meta": record(fbx + ".meta", meta),
            "serialized_clip_file_id": -623603649343465320,
            "serialized_clip_name_observed_by_unity": "Scene",
            "import": "Humanoid; importAnimation=1; avatarSetup=0; original importer unchanged",
        })
    return {
        "schema_version": SCHEMA + ".inventory",
        "practice_scene": file_record(root, PRACTICE_SCENE),
        "main_camera": {
            "position": [0, 1, -10], "rotation": [0, 0, 0, 1], "projection": "Perspective",
            "field_of_view_degrees": 60, "near": 0.3, "far": 1000,
            "culling_mask": 4294967295, "clear_flags": "Skybox", "target_texture": None,
        },
        "environment": {
            "skybox": "Unity built-in default skybox material fileID 10304",
            "background_rgba": [0.19215687, 0.3019608, 0.4745098, 0],
            "directional_light": {"position": [0, 3, 0], "euler": [50, -30, 0]},
        },
        "existing_scene_ui": {
            "legacy_canvas_present": True, "legacy_canvas_active": False,
            "canvas_scaler_reference": [1920, 1080], "match": 0.5,
            "event_system_present_and_active": True,
            "camera_feed_raw_image_and_tmp_present": True,
        },
        "practice_prefab": file_record(root, PRACTICE_PREFAB),
        "practice_prefab_components": [
            "PetRobotPracticeVerticalSliceBootstrap", "PetRobotPracticePresenter",
            "PetRobotPracticeUiPresenter (disabled by v1.2 presentation Awake)",
            "SignDemonstrationAvatarPresenter", "WebcamLandmarkOverlayPresenter",
            "GuidedSignPracticePresentationController",
        ],
        "v1_1_imgui_source_component_active_before_reconstruction": True,
        "v1_2_default_fullscreen_imgui": False,
        "font": {
            "path": "Assets/_Project/Fonts/NotoSansKR-VariableFont_wght.ttf",
            "guid": "aa36a16ad1c266948b24a0d14af31351",
            "serialized_file_id": 12800000,
        },
        "rounded_ui": "Unity built-in UI/Skin/UISprite.psd as 9-sliced Image plus uGUI Outline",
        "animations": animations,
        "v1_1_gui_gate_failure_provenance": {
            "all_three_character_type_mismatch": "USER_CONFIRMED_INPUT",
            "all_three_clip_fields_resolved_as_scene": "USER_CONFIRMED_INPUT",
            "greeting_blank_slot": "v1.1 gate_verdict.json and validation_report.md",
            "direct_external_fbx_gameobject_pptr_strategy": "retired; no alternate external fileID guessed",
        },
        "robot": {
            "wrapper": file_record(root, ROBOT_WRAPPER),
            "wrapper_guid": "10d00000000000000000000000000010",
            "source_fbx_guid": "50573fbff0bc00c418eb7319a40093d3",
            "animator_controller_guid": "a414e66cfd19823419a0f9a1954db8d3",
            "avatar": None, "avatar_absence_is_expected": True,
            "gameplay_follow_interaction_provider_camera_components": 0,
        },
        "status": "INVENTORY_RECORDED",
    }


def wrapper_evidence(root: Path) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    prefab = (root / PRACTICE_PREFAB).read_text(encoding="utf-8-sig")
    graph_rows = []
    audit_rows = []
    mapping_rows = []
    clip_id = "-623603649343465320"
    direct_external_character_count = 0
    for word, (wrapper, wrapper_guid, fbx, fbx_guid, _) in WRAPPERS.items():
        body = (root / wrapper).read_text(encoding="utf-8-sig")
        meta = (root / (wrapper + ".meta")).read_bytes()
        require(read_guid(meta) == wrapper_guid, "wrapper GUID mismatch: " + word)
        require("--- !u!1 &100000" in body, "wrapper root GameObject missing: " + word)
        require("--- !u!1001 &100200000" in body, "wrapper PrefabInstance missing: " + word)
        require("--- !u!4 &100100001 stripped" in body, "wrapper stripped transform missing: " + word)
        require("&100100000" not in body, "reserved local 100100000 reused: " + word)
        source_line = "m_SourcePrefab: {fileID: 100100000, guid: " + fbx_guid
        require(source_line in body, "wrapper external model edge missing: " + word)
        character_token = "{fileID: 100000, guid: " + wrapper_guid + ", type: 3}"
        require(character_token in prefab, "practice prefab wrapper reference missing: " + word)
        if "Character: {fileID: 100100000, guid: " + fbx_guid in prefab:
            direct_external_character_count += 1
        require(clip_id + ", guid: " + fbx_guid in prefab, "observed embedded clip mapping missing: " + word)
        graph_rows.append({
            "word": word, "runtime_character_pptr": {"file_id": 100000, "guid": wrapper_guid},
            "wrapper_path": wrapper,
            "nested_external_model_edge": {"file_id": 100100000, "guid": fbx_guid, "path": fbx},
            "relationship": "project-owned GameObject -> PrefabInstance -> Unity Model Prefab",
        })
        local_ids = re.findall(r"(?m)^--- !u!\d+ &(\d+)", body)
        audit_rows.append({
            "word": word, "local_document_ids": [int(value) for value in local_ids],
            "reserved_local_100100000_reuse_count": local_ids.count("100100000"),
            "external_source_prefab_file_id_100100000_count": body.count("m_SourcePrefab: {fileID: 100100000"),
        })
        mapping_rows.append({
            "word": word, "wrapper_guid": wrapper_guid, "fbx_guid": fbx_guid,
            "clip_file_id": -623603649343465320, "clip_observed_name": "Scene",
            "new_word_time_reset": 0.0, "loop": True, "root_motion": False,
        })
    require(direct_external_character_count == 0, "direct external character PPtr remains")
    return (
        {"schema_version": SCHEMA + ".wrapper_graph", "direct_external_fbx_character_pptr_count": 0,
         "rows": graph_rows, "status": "PROJECT_OWNED_WRAPPER_GRAPH_CONFIRMED"},
        {"schema_version": SCHEMA + ".wrapper_fileid_audit", "rows": audit_rows,
         "local_reserved_100100000_reuse_count": 0, "status": "LOCAL_FILE_IDS_AUDITED"},
        {"schema_version": SCHEMA + ".word_clip_mapping", "rows": mapping_rows,
         "mapping_count": 3, "alternate_clip_file_id_guesses": 0, "status": "MAPPING_RECORDED"},
    )


def static_contract(root: Path) -> dict[str, object]:
    demo = (root / DEMO).read_text(encoding="utf-8-sig")
    overlay = (root / OVERLAY).read_text(encoding="utf-8-sig")
    tracking = (root / TRACKING).read_text(encoding="utf-8-sig")
    presentation = (root / PRESENTATION).read_text(encoding="utf-8-sig")
    bootstrap = (root / BOOTSTRAP).read_text(encoding="utf-8-sig")
    prefab = (root / PRACTICE_PREFAB).read_text(encoding="utf-8-sig")
    shader = (root / SHADER).read_text(encoding="utf-8-sig")
    combined = demo + overlay + tracking + presentation + bootstrap
    checks = {
        "word_mapping_three": all(token in demo for token in ("GreetingWord", "LoveWord", "ThankYouWord")),
        "word_change_time_zero": "demonstrationPlayable.SetTime(0.0)" in demo,
        "manual_loop_and_done_reset": "GetTime() % duration" in demo and "SetDone(false)" in demo,
        "root_motion_off": demo.count("applyRootMotion = false") >= 2,
        "arrow_visual_yaw_only": all(token in demo for token in
            ("KeyCode.LeftArrow", "KeyCode.RightArrow", "currentDemonstrationYaw", "Time.unscaledDeltaTime")),
        "arrow_capture_session_call_count_zero": all(token not in demo for token in
            ("StartExpression", "FinishExpression", "BeginCapture", "PetRobotSignPracticeSession")),
        "secondary_camera_count_zero": all(token not in demo for token in
            ("AddComponent<Camera>", "AddComponent<UnityEngine.Camera>", "GetComponent<Camera>",
             "GetComponent<UnityEngine.Camera>", "new GameObject(\"Step10F_DemonstrationCamera",
             "new GameObject(\"Step10F_RobotCamera")) and
            re.search(r"\b(?:UnityEngine\.)?Camera\s+[a-zA-Z_][a-zA-Z0-9_]*\s*[;=,)]", demo) is None,
        "render_texture_count_zero": "RenderTexture" not in demo + presentation,
        "robot_wrapper_and_transform": all(token in prefab for token in
            ("10d00000000000000000000000000010", "robotPosition: {x: 6.05, y: 2.4, z: 0.75}",
             "robotScale: {x: 0.55, y: 0.55, z: 0.55}", "robotYaw: 205")),
        "no_imgui_default": "OnGUI(" not in presentation and "GUI." not in presentation,
        "ugui_1920_1080": all(token in presentation for token in
            ("ReferenceWidth = 1920f", "ReferenceHeight = 1080f", "ScaleWithScreenSize")),
        "rounded_sliced_outline": all(token in presentation for token in
            ("UI/Skin/UISprite.psd", "Image.Type.Sliced", "AddComponent<Outline>")),
        "existing_actions_only": all(token in presentation for token in
            ("session.StartExpression()", "session.FinishExpression()", "session.Retry()",
             "session.ContinueWithNeedsPractice()", "session.ContinueToNextWord()", "session.ExitPractice()")),
        "display_blur_material_only": all(token in overlay + shader for token in
            ("cameraRoute.PreviewTexture", "new Material(displayBlurMaterial)", "_BlurRadius", "tex2D(_MainTex")),
        "source_texture_mutation_count_zero": all(token not in overlay + shader for token in
            ("Graphics.Blit", "SetPixels", "Apply()", "ReadPixels", "RenderTexture.GetTemporary")),
        "pose_0_10_render_count_zero": "FirstBodyPoseIndex = 11" in tracking and
            all(re.search(r"\b" + str(i) + r"\b", re.search(r"BodyPosePoints\s*=\s*\{([^}]*)\}", tracking,
                          re.S).group(1)) is None for i in range(0, 11)),
        "face_mosaic_pixelation_count_zero": all(token not in overlay + tracking for token in
            ("FaceSupportIndices", "mosaic", "Mosaic", "pixelation", "Pixelation", "faceBounds", "FaceBounds")),
        "unified_tracking_color": overlay.count("[SerializeField] Color trackingColor") == 1 and
            all(token not in overlay for token in ("poseColor", "leftHandColor", "rightHandColor")),
        "body_and_two_hands_only": all(token in tracking for token in
            ("observation.Pose", "observation.LeftHand", "observation.RightHand", "HandLandmarkCount")),
        "stale_geometry_clear": "TrackingTemporarilyLost" in overlay and
            "trackingGraphic?.ClearTrackingData()" in overlay,
        "observation_mutation_reprocess_count_zero": all(token not in combined for token in
            ("ApprovedLivePreprocessingAdapter", "RecognitionSequenceInputSession", "RecognitionCore", "MaskAwareDtw")),
        "new_owner_count_zero": all(token not in combined for token in
            ("UnityWebCamFrameSource", "PoseLandmarkProvider", "MediaPipeLandmarkProvider")),
        "media_storage_count_zero": all(token not in combined for token in
            ("File.Write", "EncodeToPNG", "EncodeToJPG", "AudioClip", "VideoClip", "persistentDataPath")),
        "diagnostics_hidden_default": "diagnosticsVisibleByDefault: 0" in prefab,
        "camera_namespace_repair_not_reintroduced": "UnityEngine.Camera" not in demo and
            re.search(r"\bCamera\s+[a-zA-Z_][a-zA-Z0-9_]*\s*[;=,)]", demo) is None,
    }
    require(all(checks.values()), "v1.2 static source contract failed: " +
            ", ".join(key for key, value in checks.items() if not value))
    return {"schema_version": SCHEMA + ".static_contract", "checks": checks,
            "satisfied_count": len(checks), "total_count": len(checks),
            "status": "STATIC_SOURCE_CONTRACT_SATISFIED"}


def test_inventory(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    baseline = zip_json(root / V11_ZIP, "strengthened_28_test_inventory.json")
    baseline_fixtures = [row for row in baseline["fixtures"] if row["classification"] == "existing_preserved"]
    current_rows = []
    baseline_names = []
    current_names = []
    for row in baseline_fixtures:
        relative = "Assets/_Project/Tests/EditMode/" + row["fixture_source"]
        names = method_names((root / relative).read_text(encoding="utf-8-sig"))
        require(names == row["test_names"], "approved 152-test name/source inventory mismatch: " + relative)
        baseline_names.extend(row["test_names"])
        current_names.extend(names)
        current_rows.append({
            "relative_path": relative, "source": file_record(root, relative),
            "test_count": len(names), "test_names": names, "matches_v1_1_approved_name_order": True,
        })
    require(len(current_names) == len(set(current_names)) == 152, "approved core inventory is not 152 unique tests")
    new_names = method_names((root / TEST).read_text(encoding="utf-8-sig"))
    require(len(new_names) == len(set(new_names)) == 28, "v1.2 targeted fixture must contain exactly 28 unique tests")
    step10d = zip_json(root / STEP10D_ZIP, "test_inventory.json")
    require(step10d["approved_testcase_count"] == 152 and step10d["name_set_preserved"] is True,
            "Step 10D approved 152-test provenance mismatch")
    core = {
        "schema_version": SCHEMA + ".core_152_test_identity",
        "step10d_v1_4_approved_testcase_count": 152,
        "current_testcase_count": len(current_names),
        "current_unique_name_count": len(set(current_names)),
        "exact_v1_1_name_order_match": current_names == baseline_names,
        "fixture_records": current_rows,
        "source_changes_in_core_fixture_set": 0,
        "status": "CORE_152_TEST_SOURCE_NAME_IDENTITY_PRESERVED",
    }
    targeted = {
        "schema_version": SCHEMA + ".targeted_test_inventory",
        "fixture": file_record(root, TEST), "test_count": 28, "unique_test_count": 28,
        "test_names": new_names,
        "unity_test_execution": "UNVERIFIED_NOT_RUN_SOURCE_STAGE",
        "editmode_execution": "UNVERIFIED_NOT_RUN_SOURCE_STAGE",
        "status": "V1_2_TARGETED_SOURCE_INVENTORIED_NOT_EXECUTED",
    }
    return core, targeted


def compare_v11_identity(root: Path, member: str, schema_suffix: str) -> dict[str, object]:
    baseline = zip_json(root / V11_ZIP, member)
    rows = []
    mismatch = []
    for approved in baseline["records"]:
        relative = approved["relative_path"]
        data = (root / relative).read_bytes()
        current = record(relative, data)
        match = current["sha256"] == approved["sha256"] and current["size_bytes"] == approved["size_bytes"]
        rows.append({**current, "v1_1_sha256": approved["sha256"],
                     "v1_1_size_bytes": approved["size_bytes"], "match": match})
        if not match:
            mismatch.append(relative)
    require(not mismatch, "v1.1 protected identity mismatch: " + ", ".join(mismatch))
    return {"schema_version": SCHEMA + "." + schema_suffix, "record_count": len(rows),
            "mismatch_count": 0, "records": rows,
            "status": "V1_1_BYTE_IDENTITY_PRESERVED"}


def candidate_identity(root: Path) -> dict[str, object]:
    baseline = zip_json(root / V11_ZIP, "candidate_repository_identity.json")
    candidate_root = (Path.home() /
        "AppData/LocalLow/DefaultCompany/WordsInSilence/WordsInSilence/PersonalizationCandidates/v1/local_learner")
    require(candidate_root.is_dir(), "approved candidate repository missing")
    rows = []
    mismatch = []
    for approved in baseline["records"]:
        relative = approved["relative_path"]
        data = (candidate_root / relative).read_bytes()
        current = record(relative, data)
        match = current["sha256"] == approved["sha256"] and current["size_bytes"] == approved["size_bytes"]
        rows.append({**current, "v1_1_sha256": approved["sha256"], "match": match})
        if not match:
            mismatch.append(relative)
    all_files = sorted(path for path in candidate_root.rglob("*") if path.is_file())
    require(len(rows) == len(all_files) == 25 and not mismatch, "candidate repository identity mismatch")
    media = [path for path in all_files if path.suffix.lower() in
             {".png", ".jpg", ".jpeg", ".webm", ".mp4", ".mov", ".wav", ".mp3"}]
    require(not media, "candidate repository media payload found")
    return {"schema_version": SCHEMA + ".candidate_identity", "repository_path": str(candidate_root),
            "candidate_count": 8, "record_count": 25, "mismatch_count": 0,
            "media_file_count": 0, "capture_or_write_performed": False,
            "records": rows, "status": "V1_1_BYTE_IDENTITY_PRESERVED"}


def offline_compile(root: Path) -> tuple[dict[str, object], dict[str, bytes]]:
    editor = Path(r"C:\Program Files\Unity\Hub\Editor\6000.3.14f1\Editor")
    mono = editor / "Data/MonoBleedingEdge/bin/mono.exe"
    csc = editor / "Data/MonoBleedingEdge/lib/mono/4.5/csc.exe"
    modules = editor / "Data/Managed/UnityEngine"
    references = [modules / name for name in (
        "UnityEngine.CoreModule.dll", "UnityEngine.AnimationModule.dll", "UnityEngine.DirectorModule.dll",
        "UnityEngine.UIModule.dll", "UnityEngine.TextRenderingModule.dll", "UnityEngine.AudioModule.dll",
        "UnityEngine.InputLegacyModule.dll",
    )]
    references.extend([
        root / "Library/ScriptAssemblies/UnityEngine.UI.dll",
        root / "Library/ScriptAssemblies/WordsInSilence.Runtime.dll",
        editor / "Data/MonoBleedingEdge/lib/mono/4.5/Facades/netstandard.dll",
    ])
    nunit_candidates = sorted((root / "Library/PackageCache").glob("**/nunit.framework.dll"))
    require(mono.is_file() and csc.is_file() and all(path.is_file() for path in references) and nunit_candidates,
            "Unity 6000.3.14f1 offline Roslyn/reference inventory missing")
    temporary = Path(tempfile.mkdtemp(prefix="WIS-Step10F-v1_2-compile-"))
    logs: dict[str, bytes] = {}
    try:
        product = temporary / "Step10F_v1_2.dll"
        command = [str(mono), str(csc), "/nologo", "/target:library", "/langversion:latest",
                   "/out:" + str(product)] + ["/r:" + str(path) for path in references]
        command += [str((root / path).resolve()) for path in [DEMO, OVERLAY, TRACKING, PRESENTATION, BOOTSTRAP]]
        run = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        production_log = (run.stdout or "") + (run.stderr or "")
        logs["offline_compile/production.txt"] = production_log.encode("utf-8")
        require(run.returncode == 0, "offline production compile failed:\n" + production_log[-12000:])
        test_dll = temporary / "Step10F_v1_2_Tests.dll"
        test_refs = references[:-1] + [Path(nunit_candidates[0]), product, references[-1]]
        command = [str(mono), str(csc), "/nologo", "/target:library", "/langversion:latest",
                   "/out:" + str(test_dll)] + ["/r:" + str(path) for path in test_refs]
        command += [str((root / TEST).resolve())]
        run = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        test_log = (run.stdout or "") + (run.stderr or "")
        logs["offline_compile/targeted_fixture.txt"] = test_log.encode("utf-8")
        require(run.returncode == 0, "offline targeted fixture compile failed:\n" + test_log[-12000:])
        production_errors = len(re.findall(r"error CS\d+", production_log))
        test_errors = len(re.findall(r"error CS\d+", test_log))
        require(production_errors == 0 and test_errors == 0, "offline compile emitted C# errors")
        return ({
            "schema_version": SCHEMA + ".offline_compile",
            "compiler": "Unity 6000.3.14f1 Mono Roslyn csc",
            "production_source_count": 5, "targeted_fixture_source_count": 1,
            "production_error_count": 0, "targeted_fixture_error_count": 0,
            "shader_source_static_only": True,
            "unity_editor_compile_claimed": False, "editmode_execution_claimed": False,
            "status": "OFFLINE_CSHARP_COMPILE_SUCCEEDED_UNITY_COMPILE_UNVERIFIED",
        }, logs)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def data_flow_documents() -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    stage = {
        "schema_version": SCHEMA + ".world_stage",
        "hierarchy": {
            "practice_prefab": {
                "runtime_child": "Step10F_v1_2_WorldPresentationStage",
                "children": ["SignDemonstration_<CURRENT_WORD>", "Step10F_v1_2_PresentationOnly_MOA_robot"],
            },
            "render_path": "practice Scene Main Camera -> both real 3D children",
            "secondary_camera_count": 0, "render_texture_count": 0, "offscreen_stage_count": 0,
        },
        "recorded_transforms": {
            "demonstration": {"position": [-6.15, -1.3, 0], "scale": [2.15, 2.15, 2.15], "yaw": 180},
            "robot": {"position": [6.05, 2.4, 0.75], "scale": [0.55, 0.55, 0.55], "yaw": 205},
        },
        "framing_basis": "Main Camera (0,1,-10), perspective FOV 60; objects placed outside central x=520..1400 webcam HUD footprint at 1920x1080",
        "runtime_visual_confirmation": "UNVERIFIED_SOURCE_STAGE",
    }
    robot = {
        "schema_version": SCHEMA + ".robot_ownership",
        "source": "PetRobotAnimatedVisualWrapper.prefab -> MOA_robot.fbx",
        "animator": "approved controller a414e66cfd19823419a0f9a1954db8d3; root motion off",
        "presentation_owner": "SignDemonstrationAvatarPresenter",
        "gameplay_follow_owner": 0, "interaction_owner": 0, "provider_owner": 0,
        "camera_owner": 0, "duplicate_gameplay_robot": 0, "procedural_hover": 0,
        "purpose": "presentation-only visual",
    }
    webcam = {
        "schema_version": SCHEMA + ".webcam_data_flow",
        "rgb": "existing PetRobotCandidateCameraStartupRoute.PreviewTexture -> uGUI RawImage -> runtime clone of DisplayOnlyMildBlur material -> screen",
        "rgb_source_write_or_blit": 0,
        "recognition": "existing webcam owner/providers -> Step9C bridge -> approved recognition path (unchanged)",
        "overlay": "Step9C ObservationArrived immutable RawMediaPipeObservation -> shared display transform -> MaskableGraphic",
        "display_transform": ["horizontal mirror", "WebCamTexture vertical mirror", "0/90/180/270 rotation", "aspect-fit", "rounded viewport stencil clipping"],
        "rendered": {"body_pose_indices": [11, 12, 13, 14, 15, 16, 23, 24],
                     "left_hand_points": 21, "right_hand_points": 21,
                     "color_rgba": [0.55, 1.0, 0.22, 0.98]},
        "not_rendered": {"pose_indices_0_through_10": True, "face_connections": True,
                         "face_mosaic": True, "face_pixelation": True},
        "stale_policy_seconds": 0.45,
        "privacy_claim": "mild display treatment only; not an anonymization guarantee",
        "observation_mutation_or_feedback": 0, "rgb_storage": 0, "candidate_rgb_payload": 0,
    }
    buttons = {
        "schema_version": SCHEMA + ".ugui_button_mapping",
        "reference_resolution": [1920, 1080],
        "rows": [
            {"label": "표현 시작", "method": "PetRobotSignPracticeSession.StartExpression", "gate": "Ready && CameraReady"},
            {"label": "표현 완료", "method": "PetRobotSignPracticeSession.FinishExpression", "gate": "Capturing"},
            {"label": "다시 시도", "method": "PetRobotSignPracticeSession.Retry", "gate": "CanRetry"},
            {"label": "연습 필요로 남기고 계속", "method": "PetRobotSignPracticeSession.ContinueWithNeedsPractice", "gate": "CanContinueWithNeedsPractice"},
            {"label": "다음 단어 / 학습 완료", "method": "PetRobotSignPracticeSession.ContinueToNextWord", "gate": "Feedback && IsCurrentWordComplete"},
            {"label": "연습 공간 나가기", "method": "PetRobotSignPracticeSession.ExitPractice", "gate": "Completed"},
        ],
        "outcome_calculation_in_ui": 0, "candidate_calculation_in_ui": 0, "storage_logic_in_ui": 0,
    }
    return stage, robot, webcam, buttons


def baseline_provenance(root: Path) -> dict[str, object]:
    delivery = (root / V11_ZIP).read_bytes()
    identity_path = root / V11_CAMERA_EVIDENCE / "before_after_identity.json"
    gate_path = root / V11_CAMERA_EVIDENCE / "gate_verdict.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8-sig"))
    gate = json.loads(gate_path.read_text(encoding="utf-8-sig"))
    require(sha(delivery) == "535dc6f8c5d213397a0a12dfa1a9229f31de0f2dc6d19a9d917dc3ff051c1aaf",
            "v1.1 Delivery identity mismatch")
    require(identity["minimal_compile_repair_count"] == 1 and gate["post_repair_compiler_error_count"] == 0,
            "post-camera-repair evidence mismatch")
    return {
        "schema_version": SCHEMA + ".v1_1_post_camera_repair_provenance",
        "v1_1_source_delivery": record(V11_ZIP, delivery),
        "camera_repair_identity": record(str(identity_path.relative_to(root)).replace("\\", "/"), identity_path.read_bytes()),
        "gate_verdict": record(str(gate_path.relative_to(root)).replace("\\", "/"), gate_path.read_bytes()),
        "repaired_presenter_sha256": "85e734d563f11bed2d6f405a82ec66832d5191b59f1cf4dbf983ef7351f9e8c8",
        "repair": "all Camera declarations and camera component calls were UnityEngine.Camera-qualified",
        "v1_2_preservation": "secondary presentation cameras were removed; no ambiguous Camera symbol remains",
        "v1_1_targeted_editmode": "28/28 PASS exactly once (provenance only; not rerun)",
        "v1_1_presentation_smoke": "FAIL; character Type mismatch/blank slot and rejected flat RenderTexture composition",
        "misleading_screenshot_note": "01_compiler_error_0.png is pre-repair CS0118 evidence, not the post-repair compile authority",
    }


def player_identity(root: Path) -> dict[str, object]:
    paths = [
        "Assets/_Project/Scripts/Practice/PetRobotWorldIntegration/PetRobotWorldPlayerController.cs",
        "Assets/GameAssets/Player_model_animation_meshy/Player.controller",
    ]
    require(all(path not in CHANGED for path in paths), "Player IsRun file entered v1.2 changed scope")
    return {"schema_version": SCHEMA + ".player_isrun_identity", "records": [file_record(root, p) for p in paths],
            "changed_scope_overlap_count": 0, "repair_performed": False,
            "status": "PENDING_SEPARATE_MINIMAL_REPAIR_UNTOUCHED"}


def report(delta: dict, static: dict, compile_doc: dict, protected: dict, candidates: dict,
           core_tests: dict, targeted_tests: dict, staged_doc: dict, context_doc: dict) -> bytes:
    lines = [
        "# Step 10F v1.2 source-stage validation report", "",
        "This Delivery records the 3D world-integrated uGUI reconstruction at source stage. It is not a Unity GUI, PlayMode, recognition, or external-review PASS claim.", "",
        "## Implementation", "",
        f"- Changed project files: {delta['changed_project_file_count']} ({delta['modified_existing_count']} modified, {delta['added_count']} added)",
        "- Practice Scene YAML: unchanged; runtime-loaded practice prefab owns the real 3D stage and project-owned uGUI Canvas.",
        f"- Static source contracts satisfied: {static['satisfied_count']}/{static['total_count']}",
        "- Demonstration: three project-owned nested-FBX wrappers, existing embedded clips, time-zero reset, manual loop, root motion off.",
        "- Main Camera renders both demonstration and approved MOA_robot visual; no presentation Camera or RenderTexture remains.",
        "- Webcam: existing PreviewTexture read-only, mild display-material blur, body/arms plus two 21-point hands in one tracking color; pose 0-10 and mosaic display removed.",
        "- HUD: uGUI CanvasScaler 1920x1080, built-in sliced UI sprite, Outline borders, existing Step 10A session actions and gates.", "",
        "## Validation boundaries", "",
        f"- Offline production C# errors: {compile_doc['production_error_count']}",
        f"- Offline targeted fixture C# errors: {compile_doc['targeted_fixture_error_count']}",
        "- Unity compile: UNVERIFIED_SOURCE_STAGE", "- EditMode: UNVERIFIED_SOURCE_STAGE",
        "- PlayMode: UNVERIFIED_SOURCE_STAGE", "- Live recognition/candidate storage: NOT_APPLICABLE_SOURCE_STAGE", "",
        "## Preservation", "",
        f"- Protected recognition/profile/repository records: {protected['record_count']}, mismatches {protected['mismatch_count']}",
        f"- Candidate repository records: {candidates['record_count']}, mismatches {candidates['mismatch_count']}, media {candidates['media_file_count']}",
        f"- Approved core testcase inventory: {core_tests['current_testcase_count']}/152 exact v1.1 name-order match",
        f"- v1.2 targeted source inventory: {targeted_tests['test_count']} tests; not executed",
        f"- Git staged files before/after: {staged_doc['before_count']}/{staged_doc['after_count']}",
        f"- Project context before/after SHA-256 identical: {context_doc['unchanged']}",
        "- Player IsRun source/controller: outside changed scope and untouched.", "",
        "## Source-stage status", "",
        "STEP10F_V1_2_3D_WORLD_INTEGRATED_UGUI_PRESENTATION_READY_PENDING_EXTERNAL_REVIEW",  
        "STEP10F_V1_2_SOURCE_AND_DELIVERY_READY_PENDING_EXTERNAL_REVIEW",
        "STEP10F_V1_1_CAMERA_NAMESPACE_REPAIR_PRESERVED",
        "STEP10F_V1_2_UNITY_COMPILE_UNVERIFIED_SOURCE_STAGE",
        "STEP10F_V1_2_EDITMODE_UNVERIFIED_SOURCE_STAGE",
        "STEP10F_V1_2_PLAYMODE_UNVERIFIED_SOURCE_STAGE",
        "STEP10F_V1_2_LIVE_RECOGNITION_NOT_APPLICABLE_SOURCE_STAGE",
        "STEP10E_PLAYER_RUN_ANIMATION_DEFECT_PENDING_SEPARATE_MINIMAL_REPAIR",
        "BLOCKED_PENDING_CHATGPT_WORK_EXTERNAL_REVIEW", "",
    ]
    return "\n".join(lines).encode("utf-8")


def build(root: Path, helper) -> tuple[dict[str, bytes], dict[str, object]]:
    staged_before = staged(root)
    require(not staged_before, "Git staged files must remain zero")
    context_before = (root / CONTEXT).read_bytes()
    delta, patch, project_evidence = implementation_delta(root)
    inventory = presentation_inventory(root)
    wrapper_graph, fileid_audit, word_mapping = wrapper_evidence(root)
    static = static_contract(root)
    core_tests, targeted_tests = test_inventory(root)
    protected = compare_v11_identity(root, "protected_recognition_profile_repository_identity.json",
                                     "protected_identity")
    candidates = candidate_identity(root)
    compile_doc, compile_logs = offline_compile(root)
    stage, robot, webcam, buttons = data_flow_documents()
    provenance = baseline_provenance(root)
    player = player_identity(root)
    transaction = helper.transaction_regression()
    scene_data = (root / PRACTICE_SCENE).read_bytes()
    scene_identity = {
        "schema_version": SCHEMA + ".scene_identity", "before": record(PRACTICE_SCENE, scene_data),
        "after": record(PRACTICE_SCENE, scene_data), "unchanged": True,
        "change_type": "unchanged; included as inventory evidence",
    }
    context_after = (root / CONTEXT).read_bytes()
    require(context_before == context_after, "project context was modified")
    context_doc = {"relative_path": CONTEXT, "before_sha256": sha(context_before),
                   "after_sha256": sha(context_after), "size_bytes": len(context_before), "unchanged": True}
    staged_after = staged(root)
    require(not staged_after, "Git staged files changed during source validation")
    staged_doc = {"before": staged_before, "after": staged_after,
                  "before_count": 0, "after_count": 0, "unchanged": True}
    assertions = {
        "schema_version": SCHEMA + ".assertions",
        "source_contract": static["status"], "offline_compile": compile_doc["status"],
        "protected_mismatch_count": protected["mismatch_count"],
        "candidate_mismatch_count": candidates["mismatch_count"],
        "core_test_count": core_tests["current_testcase_count"],
        "targeted_test_source_count": targeted_tests["test_count"],
        "unity_execution_count": 0, "git_staged_count": 0,
        "project_context_modified": False, "player_isrun_modified": False,
        "overall_external_review_claimed": False,
    }
    files: dict[str, bytes] = {
        "analysis/" + Path(TOOL).name: (root / TOOL).read_bytes(),
        "analysis/" + Path(HELPER).name: (root / HELPER).read_bytes(),
        "implementation_delta.json": jbytes(delta),
        "source_scene_prefab_unified_diff.patch": patch,
        "presentation_inventory.json": jbytes(inventory),
        "practice_scene_before_after_identity.json": jbytes(scene_identity),
        "v1_1_post_camera_repair_baseline_provenance.json": jbytes(provenance),
        "demonstration_wrapper_fbx_graph.json": jbytes(wrapper_graph),
        "wrapper_local_fileid_audit.json": jbytes(fileid_audit),
        "word_to_existing_clip_mapping.json": jbytes(word_mapping),
        "world_stage_hierarchy_and_wiring.json": jbytes(stage),
        "moa_robot_ownership_graph.json": jbytes(robot),
        "webcam_read_only_blur_overlay_data_flow.json": jbytes(webcam),
        "ugui_button_existing_session_mapping.json": jbytes(buttons),
        "static_source_contract.json": jbytes(static),
        "offline_compile_evidence.json": jbytes(compile_doc),
        "v1_2_targeted_test_inventory.json": jbytes(targeted_tests),
        "core_152_test_source_identity_preservation.json": jbytes(core_tests),
        "protected_recognition_profile_repository_identity.json": jbytes(protected),
        "candidate_repository_identity.json": jbytes(candidates),
        "player_isrun_untouched_identity.json": jbytes(player),
        "project_context_read_only_identity.json": jbytes(context_doc),
        "git_staged_before_after.json": jbytes(staged_doc),
        "packaging_transaction_rollback_and_never_overwrite.json": jbytes(transaction),
        "assertions.json": jbytes(assertions),
    }
    files.update(project_evidence)
    files.update(compile_logs)
    files["validation_report.md"] = report(
        delta, static, compile_doc, protected, candidates, core_tests, targeted_tests, staged_doc, context_doc)
    return files, assertions


def add_manifest(files: dict[str, bytes]) -> dict[str, bytes]:
    manifest = {
        "schema_version": SCHEMA + ".output_manifest", "self_excluded": True,
        "file_count": len(files), "files": [record(name, files[name]) for name in sorted(files)],
    }
    return {**files, "output_file_manifest.json": jbytes(manifest)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    helper = load_helper(root)
    output_dir = root / "Validation" / NAME
    output_zip = root / "Validation" / (NAME + "_delivery.zip")
    output_sha = root / "Validation" / (NAME + "_delivery.zip.sha256")
    require(not output_dir.exists() and not output_zip.exists() and not output_sha.exists(),
            "never-overwrite target already exists")
    files, assertions = build(root, helper)
    files = add_manifest(files)
    zip_one = helper.make_zip(files)
    zip_two = helper.make_zip(files)
    require(zip_one == zip_two, "deterministic ZIP regression failed")
    probe_verification = helper.verify_zip(zip_one, files)
    verification = {
        "schema_version": SCHEMA + ".packaging_verification",
        "deterministic_payload_without_self_match": True,
        "payload_without_self_sha256": sha(zip_one),
        "payload_without_self_entry_count": probe_verification["entry_count"],
        "crc": probe_verification["crc"],
        "membership": probe_verification["membership"],
        "path_safety": probe_verification["path_safety"],
        "duplicate_entries": probe_verification["duplicate_entries"],
        "final_expected_entry_count": len(files) + 1,
        "final_archive_rebuilt_twice_and_verified_before_commit": True,
        "status": "PACKAGING_VERIFIED",
    }
    files["packaging_verification.json"] = jbytes(verification)
    files = add_manifest({key: value for key, value in files.items() if key != "output_file_manifest.json"})
    final_zip_one = helper.make_zip(files)
    final_zip_two = helper.make_zip(files)
    require(final_zip_one == final_zip_two, "final deterministic ZIP regression failed")
    final_verification = helper.verify_zip(final_zip_one, files)
    require(final_verification["entry_count"] == len(files), "final ZIP verification mismatch")
    result, delivery, detached = helper.commit_outputs(
        root / "Validation", files,
        (NAME, NAME + "_delivery.zip", NAME + "_delivery.zip.sha256"),
    )
    require(delivery.read_bytes() == final_zip_one, "committed ZIP differs from deterministic build")
    require(detached.read_text(encoding="ascii") == sha(final_zip_one) + "  " + delivery.name + "\n",
            "detached checksum mismatch")
    require(staged(root) == [], "Git staged state changed during Delivery commit")
    require((root / CONTEXT).is_file(), "project context missing after Delivery commit")
    print(json.dumps({
        "result_directory": str(result), "delivery_zip": str(delivery),
        "detached_checksum": str(detached), "delivery_sha256": sha(final_zip_one),
        "delivery_size_bytes": len(final_zip_one), "entry_count": len(files),
        "unity_invoked": False, "test_runner_invoked": False,
        "external_review_claimed": assertions["overall_external_review_claimed"],
    }, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
