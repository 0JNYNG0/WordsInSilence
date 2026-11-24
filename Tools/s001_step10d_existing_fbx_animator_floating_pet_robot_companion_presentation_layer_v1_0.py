#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.util
import io
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any


SCHEMA = "WordsInSilence.Step10D.ExistingFbxAnimatorFloatingCompanion.v1.0"
NAME = "WordsInSilence_step10d_existing_fbx_animator_floating_pet_robot_companion_presentation_layer_v1_0"
V11_ZIP = "Validation/WordsInSilence_step10c_portfolio_vertical_slice_quest_presentation_shell_v1_1_delivery.zip"
V11_SIZE = 76603
V11_SHA = "58d205b38217eb18297960b5f8645183ffce1674ead7fc5303da529e689d30c0"
V11_DETACHED_SHA = "8bb6ea6874f75bbd72debb8fe0e07be8a065f3565392d8a5bb6464bc2473d846"
V11_TOOL = "Tools/s001_step10c_portfolio_vertical_slice_quest_presentation_shell_v1_1.py"
V10_TOOL = "Tools/s001_step10c_portfolio_vertical_slice_quest_presentation_shell_v1_0.py"
BASE_TOOL = "Tools/s001_step10b_pet_robot_world_entry_return_integration_v1_0.py"
TX_TOOL = "Tools/s001_step10a_pet_robot_candidate_camera_startup_route_v1_1.py"

CONTRACTS = "Assets/_Project/Scripts/Practice/PetRobotPortfolioQuest/PetRobotPortfolioQuestContracts.cs"
CONTROLLER = "Assets/_Project/Scripts/Practice/PetRobotPortfolioQuest/PetRobotPortfolioQuestController.cs"
UI = "Assets/_Project/Scripts/Practice/PetRobotPortfolioQuest/PetRobotPortfolioQuestUiPresenter.cs"
VISUALS = "Assets/_Project/Scripts/Practice/PetRobotPortfolioQuest/PetRobotPortfolioQuestVisuals.cs"
COMPANION = "Assets/_Project/Scripts/Practice/PetRobotPortfolioQuest/PetRobotFloatingCompanionController.cs"
OLD_TEST = "Assets/_Project/Tests/EditMode/PetRobotPortfolioQuestPresentationV1Tests.cs"
NEW_TEST = "Assets/_Project/Tests/EditMode/PetRobotFloatingCompanionPresentationV1Tests.cs"
PREFAB = "Assets/_Project/Resources/VerticalSlice/PetRobotPortfolioQuestPresentation.prefab"
WRAPPER = "Assets/_Project/Resources/VerticalSlice/PetRobotAnimatedVisualWrapper.prefab"
SCENE = "Assets/_Project/Scenes/VerticalSlice/PetRobotPortfolioQuestVerticalSlice.unity"
BUILD = "ProjectSettings/EditorBuildSettings.asset"

PLAYER = "Assets/GameAssets/Player_model_animation_meshy/Meshy_AI_OriginalVer_Player_biped_Animation_Walking_withSkin.fbx"
ROBOT = "Assets/GameAssets/petrobot_model_animation/MOA_robot.fbx"
CLIP = "Assets/GameAssets/petrobot_model_animation/Meshy_Cosmic Companion_Mesh_0_Anim_Blink.anim"
ANIMATOR_CONTROLLER = "Assets/GameAssets/petrobot_model_animation/Terrain.controller"
ORIGINALS = {
    PLAYER: (18531644, "5b195d677e3ce9816f2701f6bcf858c9af4e58a1f69b005f0ef25ef001c22bd2",
             "72d9e61f9f2f7ff45880ba5850543e6b"),
    ROBOT: (784492, "2a88b32ce5f56240b4b591aa560cf1f8c9a434e1eb91eedfa92744ad7c3cbe7a",
            "50573fbff0bc00c418eb7319a40093d3"),
    CLIP: (74777, "db374c450654844dcf7f08df81ba8c06bf28c57060ce457fff44b8297ce8c396",
           "8a56dbc85c0c9b74381607d7d13f5e52"),
    ANIMATOR_CONTROLLER: (2055, "1c8421289d7612bcd59b3a9c022605ea78dc5284a23f287662a2b05e025f41ca",
                          "a414e66cfd19823419a0f9a1954db8d3"),
}
FIXED_HASHES = {
    CONTROLLER: "d53ace63858b3268e85abe9feed694b0e4223a533fff4f2490651d260ae10b12",
    UI: "b27d1fd9b380b4fa1294e315e291e3a033037d01d602338c33911c237ae3f93b",
    VISUALS: "76d8df6b1a52354495d00cff26194c80e98e178a00cab706fd61fb1bb7465543",
    BUILD: "584bc074ee61f0225ddfb52b8f9ef3a64825ea7fffd5815c8c8d5e96966c2820",
}
OLD_TEST_BEFORE_SHA = "17838f5ed3dd0465fa0341294f419623594b58a6eb0328a68012ba61dbe60d0a"
OLD_OBJECTIVE = "애완 로봇에게 가서 수어 연습을 완료하세요"
NEW_OBJECTIVE = "동행 애완 로봇과 대화해 수어 연습을 시작하세요 (E / Enter / Space)"
ALLOWED_CHANGED = sorted([CONTRACTS, OLD_TEST, PREFAB, SCENE])
ADDED = sorted([COMPANION, COMPANION + ".meta", WRAPPER, WRAPPER + ".meta",
                NEW_TEST, NEW_TEST + ".meta"])
STEP10B_GUI = "Validation/WordsInSilence_step10b_pet_robot_world_entry_return_integration_v1_0_gui_unity_editor_gate"
STEP10B_RESULT = "Validation/WordsInSilence_step10b_pet_robot_world_entry_return_integration_v1_0"
EXPECTED_THRESHOLDS = {"GREETING": 0.5113059473720765,
                       "LOVE": 0.391276023460427,
                       "THANK_YOU": 0.37709977431991587}


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def jbytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False,
                       sort_keys=True, indent=2) + "\n").encode("utf-8")


def rec(path: str, data: bytes) -> dict[str, Any]:
    return {"relative_path": path, "size_bytes": len(data), "sha256": sha(data)}


def load_module(root: Path, relative: str, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, root / relative)
    require(spec is not None and spec.loader is not None, "cannot load " + relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def staged(root: Path) -> list[str]:
    run = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=root,
                         capture_output=True, text=True, encoding="utf-8",
                         errors="replace", timeout=30)
    require(run.returncode == 0, "Git staged query failed")
    return [line.strip() for line in run.stdout.splitlines() if line.strip()]


def archive_map(data: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(data), "r") as archive:
        return {info.filename: archive.read(info) for info in archive.infolist()}


def verify_fixed_v11(root: Path, base: Any, tx: Any) -> tuple[dict[str, Any], dict[str, bytes]]:
    path = root / V11_ZIP
    detached = Path(str(path) + ".sha256")
    data, detached_data = path.read_bytes(), detached.read_bytes()
    require(len(data) == V11_SIZE and sha(data) == V11_SHA, "fixed v1.1 ZIP mismatch")
    require(sha(detached_data) == V11_DETACHED_SHA, "fixed v1.1 detached mismatch")
    require(detached_data == (V11_SHA + "  " + path.name + "\n").encode("ascii"),
            "fixed v1.1 detached syntax mismatch")
    files = archive_map(data)
    require(len(files) == 57, "fixed v1.1 entry count mismatch")
    manifest = base.verify_archive_manifest(files)
    require(manifest["target_count"] == 56, "fixed v1.1 manifest count mismatch")
    safety = base.validate_zip_bytes(data, files)
    require(safety["duplicate_count"] == safety["unsafe_path_count"] == 0 and
            safety["encrypted_entry_count"] == safety["symlink_entry_count"] == 0 and
            safety["crc_failure"] is None and safety["membership_match"] and
            safety["payload_mismatch_count"] == 0, "fixed v1.1 archive safety failure")
    require(tx.deterministic_zip(files) == data, "fixed v1.1 deterministic mismatch")
    return ({"path": V11_ZIP, "size_bytes": len(data), "sha256": sha(data),
             "detached_size_bytes": len(detached_data),
             "detached_file_sha256": sha(detached_data), "entry_count": 57,
             "manifest_target_count": 56, "deterministic_byte_identical": True,
             "status": "PASS"}, files)


def project_delta(root: Path, fixed_files: dict[str, bytes]) -> tuple[dict[str, Any], dict[str, bytes]]:
    prefix = "project_files/"
    entries = sorted(name for name in fixed_files if name.startswith(prefix))
    require(len(entries) == 16, "fixed v1.1 project file count mismatch")
    before = {name[len(prefix):]: fixed_files[name] for name in entries}
    require(sha(before[OLD_TEST]) == OLD_TEST_BEFORE_SHA, "old fixture baseline mismatch")
    changed, rows = [], []
    for relative, old in sorted(before.items()):
        current = (root / relative).read_bytes()
        if current != old:
            changed.append(relative)
        rows.append({"relative_path": relative, "before_size_bytes": len(old),
                     "before_sha256": sha(old), "after_size_bytes": len(current),
                     "after_sha256": sha(current),
                     "before_after_byte_identical": current == old,
                     "status": "PASS"})
    require(changed == ALLOWED_CHANGED, "unexpected Step10C changes: " + repr(changed))
    for relative, digest in FIXED_HASHES.items():
        require(sha((root / relative).read_bytes()) == digest, "fixed file changed: " + relative)
    old_test = before[OLD_TEST].decode("utf-8-sig")
    new_test = (root / OLD_TEST).read_text(encoding="utf-8-sig")
    require(old_test.count(OLD_OBJECTIVE) == 1 and OLD_OBJECTIVE not in new_test and
            new_test == old_test.replace(OLD_OBJECTIVE, NEW_OBJECTIVE),
            "existing fixture change is not the authorized one-line objective update")
    old_contract = before[CONTRACTS].decode("utf-8-sig")
    new_contract = (root / CONTRACTS).read_text(encoding="utf-8-sig")
    require(old_contract.count(OLD_OBJECTIVE) == 1 and
            new_contract == old_contract.replace(OLD_OBJECTIVE, NEW_OBJECTIVE),
            "contracts change exceeds one objective line")
    for relative in ADDED:
        require((root / relative).is_file(), "missing new project file: " + relative)
    return ({"schema_version": SCHEMA + ".project_delta", "baseline_file_count": 16,
             "modified_existing_count": 4, "modified_existing": changed,
             "preserved_existing_count": 12, "added_count": 6, "added": ADDED,
             "authorized_existing_fixture_change": "one expected objective string line",
             "records": rows, "status": "PASS"}, before)


def parse_fixture(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8-sig")
    fixture = re.search(r"public\s+sealed\s+class\s+(\w+)", text)
    require(fixture is not None, "fixture missing: " + str(path))
    methods = re.findall(r"\[Test\]\s*public\s+void\s+(\w+)\s*\(", text)
    require(len(methods) == len(set(methods)), "duplicate tests: " + str(path))
    return [{"fixture": fixture.group(1), "method": method,
             "source_path": path.as_posix()} for method in methods]


def test_inventory(root: Path, fixed_files: dict[str, bytes]) -> dict[str, Any]:
    approved = json.loads(fixed_files["test_inventory.json"].decode("utf-8"))
    approved_rows = approved["approved_rows"] + approved["new_rows"]
    require(len(approved_rows) == 138, "fixed inventory is not 138")
    actual = []
    for relative in sorted({row["source_path"] for row in approved_rows}):
        actual.extend(parse_fixture(root / relative))
    approved_keys = {(row["fixture"], row["method"]) for row in approved_rows}
    actual_keys = {(row["fixture"], row["method"]) for row in actual}
    require(len(actual) == len(actual_keys) == 138 and actual_keys == approved_keys,
            "existing 138 test names changed")
    new_rows = parse_fixture(root / NEW_TEST)
    new_keys = {(row["fixture"], row["method"]) for row in new_rows}
    require(len(new_rows) == len(new_keys) == 14 and not (new_keys & actual_keys),
            "new test inventory mismatch")
    return {"schema_version": SCHEMA + ".test_inventory",
            "approved_testcase_names_preserved": True, "approved_test_count": 138,
            "approved_rows": approved_rows, "new_fixture": new_rows[0]["fixture"],
            "new_test_count": 14, "new_rows": new_rows,
            "combined_unique_test_count": 152, "unity_execution_status": "UNVERIFIED",
            "status": "PASS"}


def guid_from_meta(data: bytes) -> str:
    match = re.search(rb"(?m)^guid:\s*([0-9a-f]{32})\s*$", data)
    require(match is not None, "meta GUID missing")
    return match.group(1).decode("ascii")


def asset_identity_and_graph(root: Path) -> tuple[dict[str, Any], dict[str, bytes], dict[str, bytes]]:
    meta_map: dict[str, list[str]] = {}
    meta_bytes: dict[str, bytes] = {}
    for path in (root / "Assets").rglob("*.meta"):
        data = path.read_bytes()
        match = re.search(rb"(?m)^guid:\s*([0-9a-f]{32})\s*$", data)
        if not match:
            continue
        relative = path.relative_to(root).as_posix()
        guid = match.group(1).decode("ascii")
        meta_map.setdefault(guid, []).append(relative)
        meta_bytes[relative] = data
    duplicates = {guid: values for guid, values in meta_map.items() if len(values) != 1}
    require(not duplicates, "duplicate GUIDs: " + repr(duplicates))

    selected, copies, snapshots = [], {}, {}
    for path, (size, digest, guid) in ORIGINALS.items():
        raw, meta = (root / path).read_bytes(), (root / (path + ".meta")).read_bytes()
        require(len(raw) == size and sha(raw) == digest and guid_from_meta(meta) == guid,
                "fixed original mismatch: " + path)
        selected.append({"asset": rec(path, raw), "meta": rec(path + ".meta", meta),
                         "guid": guid, "before_after_byte_identical": True, "status": "PASS"})
        copies["input_subset/selected_original_assets/" + Path(path).name] = raw
        copies["input_subset/selected_original_assets/" + Path(path).name + ".meta"] = meta
        snapshots[path], snapshots[path + ".meta"] = raw, meta

    controller = (root / ANIMATOR_CONTROLLER).read_text(encoding="utf-8-sig")
    clip = (root / CLIP).read_text(encoding="utf-8-sig")
    require(controller.count("guid: " + ORIGINALS[CLIP][2]) == 1 and
            "m_DefaultState:" in controller and "m_Motion:" in controller,
            "controller to default clip binding mismatch")
    y_match = re.search(r"attribute:\s*m_LocalPosition\.y", clip)
    require(y_match is not None and len(re.findall(r"(?m)^\s*time:\s*", clip)) > 2,
            "animation local Y curve missing")

    package_guids = {"933532a4fcc9baf4fa0491de14d08ed7":
                     "Packages/com.unity.render-pipelines.universal/Shaders/Lit.shader",
                     "d0353a89b1f911e48b9e16bdc9f2e058":
                     "Packages/com.unity.render-pipelines.universal/Editor/AssetVersion.cs"}
    roots = [PREFAB, PREFAB + ".meta", WRAPPER, WRAPPER + ".meta", SCENE, SCENE + ".meta"]
    roots += [value for path in ORIGINALS for value in (path, path + ".meta")]
    queue = list(roots)
    visited, graph, dependency_paths = set(), [], set()
    text_extensions = {".meta", ".prefab", ".unity", ".controller", ".anim", ".mat",
                       ".asset", ".shadergraph"}
    while queue:
        relative = queue.pop(0)
        if relative in visited:
            continue
        visited.add(relative)
        path = root / relative
        require(path.is_file(), "graph file missing: " + relative)
        if path.suffix.lower() not in text_extensions:
            continue
        text = path.read_text(encoding="utf-8-sig", errors="strict")
        own = guid_from_meta(path.read_bytes()) if relative.endswith(".meta") else None
        for guid in sorted(set(re.findall(r"guid:\s*([0-9a-f]{32})", text))):
            if guid == own or guid.startswith("0000000000000000"):
                continue
            if guid in package_guids:
                graph.append({"source_path": relative, "guid": guid,
                              "resolved_path": package_guids[guid],
                              "category": "UNITY_PACKAGE_ASSET", "status": "PASS"})
                continue
            paths = meta_map.get(guid, [])
            require(len(paths) == 1, "unresolved non-builtin GUID: " + guid + " from " + relative)
            meta_relative = paths[0]
            asset_relative = meta_relative[:-5]
            graph.append({"source_path": relative, "guid": guid,
                          "resolved_path": asset_relative, "resolved_meta_path": meta_relative,
                          "category": "GAMEASSETS_DEPENDENCY" if asset_relative.startswith("Assets/GameAssets/")
                          else "PROJECT_ASSET", "status": "PASS"})
            queue.extend([asset_relative, meta_relative])
            if asset_relative.startswith("Assets/GameAssets/"):
                dependency_paths.update([asset_relative, meta_relative])

    dependency_rows = []
    for relative in sorted(dependency_paths):
        data = (root / relative).read_bytes()
        snapshots.setdefault(relative, data)
        dependency_rows.append({**rec(relative, data), "before_after_byte_identical": True,
                                "status": "PASS"})
        if relative.endswith(".meta"):
            copies["input_subset/dependency_meta/" + guid_from_meta(data) + ".meta"] = data

    wrapper = (root / WRAPPER).read_text(encoding="utf-8-sig")
    presentation = (root / PREFAB).read_text(encoding="utf-8-sig")
    scene = (root / SCENE).read_text(encoding="utf-8-sig")
    require("m_SourcePrefab: {fileID: 100100000, guid: " + ORIGINALS[ROBOT][2] in wrapper,
            "robot FBX nested visual missing")
    require("guid: " + ORIGINALS[ANIMATOR_CONTROLLER][2] in wrapper and
            "propertyPath: m_Controller" in wrapper and
            "propertyPath: m_ApplyRootMotion\n      value: 0" in wrapper,
            "Animator controller/root-motion wiring missing")
    require(presentation.count("guid: " + ORIGINALS[PLAYER][2]) == 1 and
            presentation.count("guid: 10d00000000000000000000000000010") == 1,
            "player/robot presentation reference mismatch")
    require(scene.count("propertyPath: companionAnchor") == 1 and
            scene.count("objectReference: {fileID: 100100011}") == 2,
            "interaction anchor wiring mismatch")
    document = {"schema_version": SCHEMA + ".asset_identity_guid_graph",
                "selected_original_count": 4, "selected_originals": selected,
                "controller_clip_reference_count": 1, "default_state_present": True,
                "local_position_y_curve_present": True,
                "project_meta_guid_count": len(meta_map), "duplicate_guid_count": 0,
                "reference_edge_count": len(graph), "unresolved_non_builtin_guid_count": 0,
                "reference_graph": graph, "gameassets_dependency_count": len(dependency_rows),
                "gameassets_dependencies": dependency_rows,
                "unity_package_reference_count": sum(row["category"] == "UNITY_PACKAGE_ASSET" for row in graph),
                "source_assets_modified": 0, "status": "PASS"}
    return document, copies, snapshots


def static_contract(root: Path) -> dict[str, Any]:
    source = (root / COMPANION).read_text(encoding="utf-8-sig")
    prefab = (root / PREFAB).read_text(encoding="utf-8-sig")
    wrapper = (root / WRAPPER).read_text(encoding="utf-8-sig")
    scene = (root / SCENE).read_text(encoding="utf-8-sig")
    forbidden = ["Input", "SceneManager", "WebCam", "MediaPipe", "RecognitionCore",
                 "MaskAwareDtw", "CandidateRepository", "NavMesh", "Mathf.Sin",
                 "hoverAmplitude", "hoverFrequency", "bobAmplitude", "bobFrequency",
                 "localPosition"]
    hits = [token for token in forbidden if token in source]
    require(not hits, "forbidden companion references: " + repr(hits))
    for token in ["LateUpdate()", "Mathf.Exp", "sideOffset = DefaultSideOffset",
                  "heightOffset = DefaultHeightOffset", "forwardOffset = DefaultForwardOffset",
                  "followSharpness = DefaultFollowSharpness", "rotationSharpness = DefaultRotationSharpness",
                  "snapDistance = DefaultSnapDistance", "distanceSquared > snapSquared"]:
        require(token in source, "companion contract missing: " + token)
    require(prefab.count("guid: 10d00000000000000000000000000001") == 1,
            "companion owner count mismatch")
    require(scene.count("propertyPath: playerTransform") == 2 and
            scene.count("propertyPath: companionAnchor") == 1,
            "scene companion wiring mismatch")
    require(wrapper.count("propertyPath: m_Controller") == 1 and
            wrapper.count("propertyPath: m_ApplyRootMotion") == 1 and
            wrapper.count("propertyPath: m_Enabled") == 1,
            "robot Animator override count mismatch")
    require(NEW_OBJECTIVE in (root / CONTRACTS).read_text(encoding="utf-8-sig"),
            "objective mismatch")
    return {"schema_version": SCHEMA + ".static_contract",
            "companion_owner_count": 1, "scene_player_wiring_count": 1,
            "scene_anchor_wiring_count": 1, "robot_active_animator_count": 1,
            "animator_on_interaction_anchor": False, "apply_root_motion": False,
            "defaults": {"side_offset": 1.15, "height_offset": 1.25,
                         "forward_offset": -0.15, "follow_sharpness": 8.0,
                         "rotation_sharpness": 8.0, "snap_distance": 6.0},
            "procedural_hover_field_count": 0, "procedural_y_oscillation_count": 0,
            "forbidden_reference_count": 0,
            "hierarchy": ["Existing Player Gameplay Root -> Player FBX visual",
                          "PetRobot interaction anchor -> animated wrapper -> Robot FBX"],
            "status": "PASS"}


UNITY_STUB = r'''
using System;
namespace UnityEngine {
 [AttributeUsage(AttributeTargets.Field)] public sealed class SerializeField:Attribute{}
 public class Object { public string name; public static void DestroyImmediate(object value){} }
 public class GameObject:Object { public Transform transform=new Transform(); public GameObject(){} public GameObject(string n){name=n;} public T AddComponent<T>() where T:new(){return new T();} }
 public class MonoBehaviour:Object { public Transform transform=new Transform(); }
 public static class Application { public static string dataPath="."; }
 public static class Time { public static float deltaTime=0.016f; }
 public static class Mathf { public static float Min(float a,float b)=>a<b?a:b; public static float Exp(float v)=>(float)Math.Exp(v); public static float Clamp01(float v)=>v<0?0:v>1?1:v; public static float Sqrt(float v)=>(float)Math.Sqrt(v); }
 public struct Vector3 {
  public float x,y,z; public Vector3(float x,float y,float z){this.x=x;this.y=y;this.z=z;}
  public static Vector3 zero=>new Vector3(); public static Vector3 one=>new Vector3(1,1,1);
  public static Vector3 right=>new Vector3(1,0,0); public static Vector3 left=>new Vector3(-1,0,0); public static Vector3 up=>new Vector3(0,1,0); public static Vector3 forward=>new Vector3(0,0,1);
  public float sqrMagnitude=>x*x+y*y+z*z; public float magnitude=>Mathf.Sqrt(sqrMagnitude);
  public static Vector3 Lerp(Vector3 a,Vector3 b,float t)=>a+(b-a)*t;
  public static Vector3 operator +(Vector3 a,Vector3 b)=>new Vector3(a.x+b.x,a.y+b.y,a.z+b.z);
  public static Vector3 operator -(Vector3 a,Vector3 b)=>new Vector3(a.x-b.x,a.y-b.y,a.z-b.z);
  public static Vector3 operator *(Vector3 a,float b)=>new Vector3(a.x*b,a.y*b,a.z*b);
 }
 public struct Quaternion { public float x,y,z,w; public static Quaternion identity=>new Quaternion{w=1}; public static Quaternion Slerp(Quaternion a,Quaternion b,float t)=>b; }
 public class Transform:Object { public Vector3 position; public Quaternion rotation=Quaternion.identity; public Vector3 right=Vector3.right,up=Vector3.up,forward=Vector3.forward; }
}
'''

NUNIT_STUB = r'''
using System; namespace NUnit.Framework {
 [AttributeUsage(AttributeTargets.Method)] public sealed class TestAttribute:Attribute{}
 public static class Assert {
  public static void AreEqual(object a,object b,string m=null){} public static void AreEqual(float a,float b,double d){}
  public static void IsTrue(bool v,string m=null){} public static void IsFalse(bool v,string m=null){}
  public static void Less(float a,float b,string m=null){} public static void Greater(float a,float b,string m=null){}
  public static void GreaterOrEqual(int a,int b,string m=null){}
 }
 public static class StringAssert { public static void Contains(string a,string b,string m=null){} }
}
'''

PROBE = r'''
using System; using UnityEngine; using WordsInSilence.Practice.PetRobotPortfolioQuest;
static class Probe {
 static bool Near(float a,float b){return Math.Abs(a-b)<0.0001f;}
 static PetRobotFloatingCompanionController C(Transform p,Transform a,float side=1.15f){var c=new PetRobotFloatingCompanionController();c.ConfigureForTests(p,a,side,1.25f,-.15f,8f,8f,6f);return c;}
 static void Main(){
  var p=new Transform();var a=new Transform();var c=C(p,a);var t=c.TargetPositionForTests();bool right=Near(t.x,1.15f),height=Near(t.y,1.25f),range=t.magnitude<2.5f;
  bool left=C(p,a,-1.15f).TargetPositionForTests().x<0;
  a.position=t+Vector3.left;float bx=a.position.x;c.StepForTests(.1f);bool smooth=a.position.x>bx&&a.position.x<t.x;
  a.position=new Vector3(20,20,20);c.StepForTests(.1f);bool snap=Near(a.position.x,t.x)&&Near(a.position.y,t.y);
  p.position=Vector3.right;float sx=a.position.x,nt=c.TargetPositionForTests().x;c.StepForTests(.1f);bool post=a.position.x>sx&&a.position.x<nt;
  float yTarget=c.TargetPositionForTests().y;float[] times={.01f,.03f,.1f,.2f};bool fixedY=true;foreach(float dt in times){if(!Near(c.TargetPositionForTests().y,yTarget))fixedY=false;c.StepForTests(dt);}
  var owner=c.transform;owner.position=new Vector3(9,9,9);a.position=new Vector3(20,20,20);c.StepForTests(.1f);bool separate=Near(owner.position.x,9)&&Near(a.position.x,c.TargetPositionForTests().x);
  Console.WriteLine("RIGHT_OFFSET="+right);Console.WriteLine("LEFT_OFFSET="+left);Console.WriteLine("FIXED_HEIGHT="+height);Console.WriteLine("SMOOTH_FOLLOW="+smooth);Console.WriteLine("SNAP="+snap);Console.WriteLine("POST_SNAP_SMOOTH="+post);Console.WriteLine("INTERACTION_RANGE="+range);Console.WriteLine("NO_Y_OSCILLATION="+fixedY);Console.WriteLine("ANCHOR_VISUAL_SEPARATE="+separate);
  bool ok=right&&left&&height&&smooth&&snap&&post&&range&&fixedY&&separate;Console.WriteLine("PROBE_STATUS="+(ok?"PASS":"FAIL"));if(!ok)Environment.Exit(2);
 }
}
'''


def offline_compile(root: Path, v11: Any, v10: Any, base: Any, tx: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    legacy_compile, legacy_probe, legacy_outputs = v11.offline_compile(root, v10, base, tx)
    mono = Path(r"C:\Program Files\Unity\Hub\Editor\6000.3.14f1\Editor\Data\MonoBleedingEdge\bin\mono.exe")
    csc = Path(r"C:\Program Files\Unity\Hub\Editor\6000.3.14f1\Editor\Data\MonoBleedingEdge\lib\mono\4.5\csc.exe")
    require(mono.is_file() and csc.is_file(), "Unity Mono Roslyn missing")
    temporary = Path(tempfile.mkdtemp(prefix="WIS-Step10D-compile-"))
    outputs = dict(legacy_outputs)
    try:
        (temporary / "unity.cs").write_text(UNITY_STUB, encoding="utf-8")
        (temporary / "nunit.cs").write_text(NUNIT_STUB, encoding="utf-8")
        dll = temporary / "companion.dll"
        run = subprocess.run([str(mono), str(csc), "/nologo", "/target:library", "/langversion:latest",
                              "/out:" + str(dll), str(temporary / "unity.cs"),
                              str((root / COMPANION).resolve())], capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=120)
        outputs["companion_production"] = (run.stdout or "") + (run.stderr or "")
        require(run.returncode == 0, "companion production compile failed:\n" + outputs["companion_production"][-12000:])
        tests = temporary / "tests.dll"
        run = subprocess.run([str(mono), str(csc), "/nologo", "/target:library", "/langversion:latest",
                              "/out:" + str(tests), "/r:" + str(dll), str(temporary / "nunit.cs"),
                              str((root / NEW_TEST).resolve())], capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=120)
        outputs["companion_test"] = (run.stdout or "") + (run.stderr or "")
        require(run.returncode == 0, "companion test compile failed:\n" + outputs["companion_test"][-12000:])
        (temporary / "probe.cs").write_text(PROBE, encoding="utf-8")
        exe = temporary / "probe.exe"
        run = subprocess.run([str(mono), str(csc), "/nologo", "/target:exe", "/langversion:latest",
                              "/out:" + str(exe), "/r:" + str(dll), str(temporary / "probe.cs")],
                             capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        outputs["companion_probe_compile"] = (run.stdout or "") + (run.stderr or "")
        require(run.returncode == 0, "companion probe compile failed:\n" + outputs["companion_probe_compile"][-12000:])
        run = subprocess.run([str(mono), str(exe)], cwd=temporary, capture_output=True,
                             text=True, encoding="utf-8", errors="replace", timeout=120)
        outputs["companion_probe"] = (run.stdout or "") + (run.stderr or "")
        require(run.returncode == 0 and "PROBE_STATUS=PASS" in outputs["companion_probe"],
                "companion behavior probe failed:\n" + outputs["companion_probe"][-12000:])
        values = dict(line.lstrip("\ufeff").split("=", 1) for line in
                      outputs["companion_probe"].splitlines() if "=" in line)
        required = ["RIGHT_OFFSET", "LEFT_OFFSET", "FIXED_HEIGHT", "SMOOTH_FOLLOW", "SNAP",
                    "POST_SNAP_SMOOTH", "INTERACTION_RANGE", "NO_Y_OSCILLATION",
                    "ANCHOR_VISUAL_SEPARATE"]
        require(all(values.get(key) == "True" for key in required), "probe values mismatch")
        compile_doc = {"schema_version": SCHEMA + ".offline_compile",
                       "compiler": "Unity 6000.3.14f1 Mono Roslyn csc",
                       "legacy_step10c_production_error_count": legacy_compile["production_error_count"],
                       "legacy_step10c_test_error_count": legacy_compile["test_error_count"],
                       "legacy_step10c_probe_error_count": legacy_compile["probe_error_count"],
                       "companion_production_error_count": len(re.findall(r"error CS\d+", outputs["companion_production"])),
                       "companion_test_error_count": len(re.findall(r"error CS\d+", outputs["companion_test"])),
                       "companion_probe_error_count": len(re.findall(r"error CS\d+", outputs["companion_probe_compile"])),
                       "production_error_count": 0, "test_error_count": 0, "probe_error_count": 0,
                       "unity_gui_compile_claimed": False, "status": "PASS"}
        probe_doc = {"schema_version": SCHEMA + ".behavior_probe", "values": values,
                     "legacy_quest_probe": legacy_probe, "path_count": 9,
                     "procedural_y_oscillation": False, "unity_execution_claimed": False,
                     "status": "PASS"}
        return compile_doc, probe_doc, outputs
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def verify_repository_actual(document: dict[str, Any]) -> dict[str, Any]:
    root = Path(document["path"])
    require(root.is_dir(), "candidate repository path missing")
    actual = {path.relative_to(root).as_posix(): path.read_bytes()
              for path in root.rglob("*") if path.is_file()}
    expected = {row["relative_path"]: row for row in document["records"]}
    require(len(actual) == len(expected) == 25 and set(actual) == set(expected),
            "candidate repository file set mismatch")
    mismatches = [name for name, data in actual.items()
                  if len(data) != expected[name]["size_bytes"] or sha(data) != expected[name]["sha256"]]
    require(not mismatches, "candidate repository payload changed: " + repr(mismatches))
    index = json.loads(actual["candidate_index.json"].decode("utf-8-sig"))
    entries = index.get("entries", index.get("candidates", []))
    require(len(entries) == 8, "candidate index count mismatch")
    return {"path": str(root), "candidate_count_before_after": [8, 8],
            "file_count_before_after": [25, 25], "payload_file_count": 24,
            "index_entry_count": 8, "mismatch_count": 0, "records": document["records"],
            "status": "PASS"}


def assertions(rows: list[tuple[str, str, str]]) -> bytes:
    seen, out = set(), []
    for key, status, detail in rows:
        require(key not in seen, "duplicate assertion key")
        seen.add(key)
        out.append(json.dumps({"assertion_key": key, "status": status, "detail": detail},
                              ensure_ascii=False, allow_nan=False, sort_keys=True))
    return ("\n".join(out) + "\n").encode("utf-8")


def with_manifest(files: dict[str, bytes]) -> dict[str, bytes]:
    rows = [rec(name, files[name]) for name in sorted(files)]
    return {**files, "output_file_manifest.json": jbytes({
        "schema_version": SCHEMA + ".output_manifest", "self_excluded": True, "files": rows})}


def build(root: Path) -> None:
    validation = root / "Validation"
    result = validation / NAME
    delivery = validation / (NAME + "_delivery.zip")
    detached = Path(str(delivery) + ".sha256")
    require(not result.exists() and not delivery.exists() and not detached.exists(),
            "never-overwrite target exists")
    require(staged(root) == [], "Git staged files are not zero")
    v11 = load_module(root, V11_TOOL, "step10d_v11")
    v10 = load_module(root, V10_TOOL, "step10d_v10")
    base = load_module(root, BASE_TOOL, "step10d_base")
    tx = load_module(root, TX_TOOL, "step10d_tx")
    fixed, fixed_files = verify_fixed_v11(root, base, tx)
    delta, before = project_delta(root, fixed_files)
    inventory = test_inventory(root, fixed_files)
    asset_graph, asset_copies, original_snapshots = asset_identity_and_graph(root)
    contract = static_contract(root)

    gui = root / STEP10B_GUI
    project = v10.verify_records(root, load_json(gui / "step10a_project_files_before_after.json"),
                                 30, "step10a_project_files")
    protected = v10.verify_records(root, load_json(gui / "protected_files_before_after.json"),
                                   106, "protected_files")
    profile_source = load_json(gui / "active_candidate_profile_before_after.json")
    profile = v10.verify_records(root, profile_source, 22, "active_candidate_profile")
    require(profile_source["prototype_count"] == 9 and profile_source["threshold_count"] == 3 and
            profile_source["thresholds"] == EXPECTED_THRESHOLDS, "profile mismatch")
    profile.update({"prototype_count": 9, "threshold_count": 3,
                    "thresholds": EXPECTED_THRESHOLDS})
    repository = verify_repository_actual(
        json.loads(fixed_files["candidate_repository_before_after.json"].decode("utf-8")))
    before_build = (root / STEP10B_RESULT /
                    "changes/after/ProjectSettings/EditorBuildSettings.asset").read_bytes()
    step10b = v10.verify_step10b_files(
        root, load_json(gui / "step10b_project_files_before_after.json"), before_build)
    compile_doc, probe_doc, compile_outputs = offline_compile(root, v11, v10, base, tx)
    transaction = tx.transaction_regression()

    require(staged(root) == [], "Git staged files changed during validation")
    require(all((root / path).read_bytes() == data for path, data in original_snapshots.items()),
            "GameAssets original/dependency changed")

    diffs = []
    for relative in ALLOWED_CHANGED:
        diffs.extend(difflib.unified_diff(
            before[relative].decode("utf-8-sig").splitlines(True),
            (root / relative).read_text(encoding="utf-8-sig").splitlines(True),
            fromfile="a/" + relative, tofile="b/" + relative))
    diff_blob = "".join(diffs).encode("utf-8")
    modified = [{"relative_path": path, "before": rec(path, before[path]),
                 "after": rec(path, (root / path).read_bytes())} for path in ALLOWED_CHANGED]
    added = [rec(path, (root / path).read_bytes()) for path in ADDED]
    implementation = {"schema_version": SCHEMA + ".implementation_delta",
                      "modified_existing_count": 4, "modified_existing": modified,
                      "added_project_file_count": 6, "added_project_files": added,
                      "diff_addition_count": sum(1 for line in diff_blob.decode("utf-8").splitlines()
                                                 if line.startswith("+") and not line.startswith("+++")),
                      "diff_deletion_count": sum(1 for line in diff_blob.decode("utf-8").splitlines()
                                                 if line.startswith("-") and not line.startswith("---")),
                      "status": "PASS"}

    assertion_blob = assertions([
        ("implementation_owner_codex", "PASS", "project context section 3"),
        ("fixed_step10c_v11_delivery", "PASS", V11_SHA),
        ("selected_original_identity", "PASS", "4/4"),
        ("controller_clip_binding", "PASS", "default state -> designated clip"),
        ("clip_local_y_curve", "PASS", "m_LocalPosition.y"),
        ("gameassets_originals_preserved", "PASS", "modified 0"),
        ("step10c_fixed_controller", "PASS", FIXED_HASHES[CONTROLLER]),
        ("existing_fixture_before_identity", "PASS", OLD_TEST_BEFORE_SHA),
        ("existing_fixture_authorized_line_only", "PASS", "objective expected string"),
        ("existing_test_names_preserved", "PASS", "138/138"),
        ("new_test_inventory", "PASS", "14/14"),
        ("combined_test_inventory", "PASS", "152"),
        ("companion_owner_one", "PASS", "1"),
        ("player_anchor_wiring", "PASS", "explicit"),
        ("interaction_anchor_moves", "PASS", "external anchor probe"),
        ("player_fbx_visual", "PASS", ORIGINALS[PLAYER][2]),
        ("robot_fbx_visual", "PASS", ORIGINALS[ROBOT][2]),
        ("robot_animator_controller", "PASS", ORIGINALS[ANIMATOR_CONTROLLER][2]),
        ("animator_apply_root_motion_false", "PASS", "0"),
        ("animator_not_on_interaction_anchor", "PASS", "visual wrapper child"),
        ("procedural_hover_absent", "PASS", "fields/calculation 0"),
        ("default_interaction_range", "PASS", "offset magnitude < 2.5"),
        ("smooth_snap_post_snap_probe", "PASS", "3/3"),
        ("right_left_fixed_height_probe", "PASS", "3/3"),
        ("guid_duplicates", "PASS", "0"),
        ("guid_unresolved_non_builtin", "PASS", "0"),
        ("offline_production_compile", "PASS", "error 0"),
        ("offline_test_compile", "PASS", "error 0"),
        ("offline_probe_compile", "PASS", "error 0"),
        ("step10a_project_preserved", "PASS", "30/30"),
        ("protected_files_preserved", "PASS", "106/106"),
        ("step10b_baseline_preserved", "PASS", "approved files unchanged"),
        ("candidate_profile_preserved", "PASS", "22/22"),
        ("prototype_threshold_preserved", "PASS", "9/9 + 3/3"),
        ("candidate_repository_preserved", "PASS", "8/25"),
        ("build_settings_preserved", "PASS", FIXED_HASHES[BUILD]),
        ("git_staged_zero", "PASS", "0"),
        ("transaction_failure_injection", "PASS", "4/4"),
        ("never_overwrite", "PASS", "final targets absent and sentinel protected"),
        ("unity_gui", "UNVERIFIED", "source stage prohibited"),
        ("unity_editmode", "UNVERIFIED", "152 tests not executed"),
        ("unity_playmode", "UNVERIFIED", "animation playback not executed"),
        ("robot_animation_playback", "UNVERIFIED", "requires PlayMode"),
        ("webcam_live_capture", "NOT_APPLICABLE", "source stage prohibited"),
    ])
    primary = {"schema_version": SCHEMA + ".primary_verdict", "status": "PASS",
               "verdict": "EXISTING_FBX_ANIMATOR_COMPANION_SOURCE_AND_PACKAGE_READY",
               "unity_gui": "UNVERIFIED", "unity_editmode": "UNVERIFIED",
               "unity_playmode": "UNVERIFIED", "robot_animation_playback": "UNVERIFIED",
               "webcam_live_capture": "NOT_APPLICABLE",
               "status_strings": [
                   "STEP10D_EXISTING_FBX_ANIMATOR_COMPANION_SOURCE_AND_PACKAGE_READY",
                   "STEP10D_GAMEASSETS_ORIGINALS_PRESERVED",
                   "STEP10D_PROCEDURAL_HOVER_NOT_IMPLEMENTED",
                   "STEP10D_UNITY_GUI_EDITMODE_PLAYMODE_UNVERIFIED",
                   "BLOCKED_PENDING_EXTERNAL_REVIEW"]}
    inventory_md = """# Step 10D Inventory\n\n- Player visual: imported Player walking FBX under the existing gameplay root.\n- Robot visual: project-owned wrapper containing the imported MOA robot FBX.\n- Robot motion: existing Terrain Animator Controller and existing Blink/local-Y clip.\n- Gameplay roots: Step 10B player root and pet-robot interaction anchor are retained.\n- No saved user Player/Robot prefab is required.\n- GameAssets originals remain read-only.\n"""
    selection_md = f"""# Existing Asset Selection\n\n| Role | Path | GUID |\n|---|---|---|\n| Player FBX | `{PLAYER}` | `{ORIGINALS[PLAYER][2]}` |\n| Robot FBX | `{ROBOT}` | `{ORIGINALS[ROBOT][2]}` |\n| Robot clip | `{CLIP}` | `{ORIGINALS[CLIP][2]}` |\n| Robot controller | `{ANIMATOR_CONTROLLER}` | `{ORIGINALS[ANIMATOR_CONTROLLER][2]}` |\n\nThe previous saved-prefab blocker is superseded: Unity-imported FBX instances are used, with a project-owned robot visual wrapper only because Animator binding must be explicit.\n"""
    report = f"""# Step 10D Existing-FBX/Animator Companion Validation\n\n## Verdict\n\nPASS — source and deterministic Delivery are ready for independent external review.\n\n## Implementation\n\n- The existing interaction anchor follows the player at offsets `1.15 / 1.25 / -0.15`.\n- Smooth follow and rotation use frame-rate-independent exponential interpolation; snap distance is `6`.\n- No procedural hover, sine, bob or visual-local position write was added.\n- The player remains an imported FBX visual child of the gameplay root.\n- The robot wrapper preserves the imported Robot FBX instance and configures its Animator with the approved controller, `applyRootMotion=false`.\n- The existing animation clip contains the local-Y curve. Actual playback remains UNVERIFIED until PlayMode.\n\n## Verification\n\n- Existing test names `138/138`; new fixture `14/14`; static total `152`.\n- Offline production/test/probe compiler errors: `0/0/0`.\n- GameAssets originals and resolved dependencies changed: `0`.\n- Step 10A `30/30`, protected `106/106`, profile `22/22`, prototypes `9/9`, thresholds `3/3`, repository `8/25`.\n\n## Execution boundary\n\n- Unity GUI/EditMode/PlayMode: UNVERIFIED\n- Robot animation playback: UNVERIFIED\n- Webcam/live capture: NOT_APPLICABLE\n"""

    files: dict[str, bytes] = {
        "STEP10D_INVENTORY.md": inventory_md.encode("utf-8"),
        "prefab_selection_report.md": selection_md.encode("utf-8"),
        "input_provenance.json": jbytes({"schema_version": SCHEMA + ".input_provenance",
                                           "fixed_step10c_v11": fixed,
                                           "implementation_owner": "Codex", "status": "PASS"}),
        "implementation_delta.json": jbytes(implementation),
        "companion_static_contract.json": jbytes(contract),
        "asset_identity_and_guid_graph.json": jbytes(asset_graph),
        "test_inventory.json": jbytes(inventory),
        "offline_compile_evidence.json": jbytes(compile_doc),
        "offline_behavior_probe.json": jbytes(probe_doc),
        "offline_production_compile_output.txt": (compile_outputs["production"] + compile_outputs["companion_production"]).encode("utf-8"),
        "offline_test_compile_output.txt": (compile_outputs["test"] + compile_outputs["companion_test"]).encode("utf-8"),
        "offline_probe_compile_output.txt": (compile_outputs["probe_compile"] + compile_outputs["companion_probe_compile"]).encode("utf-8"),
        "offline_behavior_probe_output.txt": (compile_outputs["probe"] + compile_outputs["companion_probe"]).encode("utf-8"),
        "step10c_v11_project_before_after.json": jbytes(delta),
        "step10a_project_files_before_after.json": jbytes(project),
        "protected_files_before_after.json": jbytes(protected),
        "step10b_files_before_after.json": jbytes(step10b),
        "active_candidate_profile_before_after.json": jbytes(profile),
        "candidate_repository_before_after.json": jbytes(repository),
        "gameassets_originals_before_after.json": jbytes(asset_graph),
        "git_staged_before_after.json": jbytes({"before_count": 0, "after_count": 0,
                                                  "before_after_unchanged": True, "status": "PASS"}),
        "packaging_transaction_failure_injection_regression.json": jbytes(transaction),
        "primary_verdict.json": jbytes(primary), "ASSERTIONS.jsonl": assertion_blob,
        "VALIDATION_REPORT.md": report.encode("utf-8"),
        "changes/unified_diff.patch": diff_blob,
        "analysis/s001_step10d_existing_fbx_animator_floating_pet_robot_companion_presentation_layer_v1_0.py": Path(__file__).read_bytes(),
    }
    for relative in ALLOWED_CHANGED:
        files["changes/before/" + relative] = before[relative]
        files["changes/after/" + relative] = (root / relative).read_bytes()
    for relative in ADDED:
        files["project_files/" + relative] = (root / relative).read_bytes()
    for archive_name, data in fixed_files.items():
        if archive_name.startswith("project_files/"):
            relative = archive_name[len("project_files/"):]
            files[archive_name] = (root / relative).read_bytes()
    files.update(asset_copies)
    complete = with_manifest(files)
    zip_blob = tx.deterministic_zip(complete)
    detached_blob = (sha(zip_blob) + "  " + delivery.name + "\n").encode("ascii")
    pre = base.validate_zip_bytes(zip_blob, complete)
    require(pre["duplicate_count"] == pre["unsafe_path_count"] == 0 and
            pre["encrypted_entry_count"] == pre["symlink_entry_count"] == 0 and
            pre["crc_failure"] is None and pre["membership_match"] and
            pre["payload_mismatch_count"] == 0, "pre-commit ZIP validation failure")

    def validator(final_result: Path, final_zip: Path, final_detached: Path) -> None:
        disk = {path.relative_to(final_result).as_posix(): path.read_bytes()
                for path in final_result.rglob("*") if path.is_file()}
        require(disk == complete, "committed result map mismatch")
        require(tx.deterministic_zip(disk) == final_zip.read_bytes(), "deterministic ZIP mismatch")
        require(final_detached.read_bytes() == detached_blob, "detached mismatch")
        post = base.validate_zip_bytes(final_zip.read_bytes(), disk)
        require(post["duplicate_count"] == post["unsafe_path_count"] == 0 and
                post["encrypted_entry_count"] == post["symlink_entry_count"] == 0 and
                post["crc_failure"] is None and post["membership_match"] and
                post["payload_mismatch_count"] == 0, "post-commit ZIP validation failure")

    paths = tx.commit_outputs(validation, complete,
                              names=(NAME, NAME + "_delivery.zip", NAME + "_delivery.zip.sha256"),
                              validator=validator)
    require(paths == (result, delivery, detached), "unexpected transaction output")
    require(staged(root) == [], "Git staged files changed")
    require(all((root / path).read_bytes() == data for path, data in original_snapshots.items()),
            "GameAssets changed after packaging")
    print(json.dumps({"status": "PASS", "result_directory": str(result),
                      "result_file_count": len(complete), "manifest_target_count": len(complete) - 1,
                      "delivery_zip": str(delivery), "delivery_size_bytes": delivery.stat().st_size,
                      "delivery_sha256": sha(delivery.read_bytes()),
                      "detached_checksum": str(detached), "detached_size_bytes": detached.stat().st_size,
                      "detached_file_sha256": sha(detached.read_bytes()),
                      "modified_existing": modified, "added": added,
                      "tests": "138 + 14 = 152", "offline_compile_errors": "0/0/0",
                      "gameassets_modified": 0, "candidate_repository": "8/25 unchanged",
                      "git_staged": 0, "unity_gui_editmode_playmode": "UNVERIFIED",
                      "robot_animation_playback": "UNVERIFIED", "webcam_live": "NOT_APPLICABLE"},
                     ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    require(args.build, "use --build")
    build(args.root.resolve())


if __name__ == "__main__":
    main()
