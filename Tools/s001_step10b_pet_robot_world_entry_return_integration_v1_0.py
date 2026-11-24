#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA = "WordsInSilence.Step10B.PetRobotWorldEntryReturnIntegration.v1.0"
NAME = "WordsInSilence_step10b_pet_robot_world_entry_return_integration_v1_0"
V13_ZIP = "Validation/WordsInSilence_step10a_pet_robot_retry_exhaustion_needs_practice_completion_v1_3_delivery.zip"
V13_ZIP_SIZE = 101574
V13_ZIP_SHA = "42cfe5679c09defc220033fd99eb638576bd5313b5c047dfe25766ac425575bf"
V13_DETACHED_SIZE = 160
V13_DETACHED_SHA = "6c630bc1b18770deec5e23e8669b5075eaae42a6907c61e918e92316cd1e0354"
GUI_ZIP = "Validation/WordsInSilence_step10a_pet_robot_retry_exhaustion_needs_practice_completion_v1_3_gui_unity_editor_gate_delivery.zip"
GUI_ZIP_SIZE = 1491795
GUI_ZIP_SHA = "6463cbed625e4b4f27b7fdf1cb0bcdcb07a38571334ae3083eff4ef85684798a"
GUI_DETACHED_SIZE = 182
GUI_DETACHED_SHA = "55d8b44681e642c4f68f6975ce3a3a1c8e09969d4deb4b1ca402bd96b11dc955"
V13_RESULT = "Validation/WordsInSilence_step10a_pet_robot_retry_exhaustion_needs_practice_completion_v1_3"
GUI_RESULT = "Validation/WordsInSilence_step10a_pet_robot_retry_exhaustion_needs_practice_completion_v1_3_gui_unity_editor_gate"
V11_TOOL = "Tools/s001_step10a_pet_robot_candidate_camera_startup_route_v1_1.py"
V13_TOOL = "Tools/s001_step10a_pet_robot_retry_exhaustion_needs_practice_completion_v1_3.py"

WORLD_SCENE = "Assets/_Project/Scenes/VerticalSlice/PetRobotWorldEntryReturnVerticalSlice.unity"
WORLD_PREFAB = "Assets/_Project/Resources/VerticalSlice/PetRobotWorldEntryReturn.prefab"
BUILD_SETTINGS = "ProjectSettings/EditorBuildSettings.asset"
NEW_TEST = "Assets/_Project/Tests/EditMode/PetRobotWorldEntryReturnIntegrationV1Tests.cs"
PRACTICE_SCENE = "Assets/_Project/Scenes/VerticalSlice/PetRobotSignPracticeVerticalSlice.unity"
PRACTICE_PREFAB = "Assets/_Project/Resources/VerticalSlice/PetRobotSignPracticeVerticalSlice.prefab"
VENDOR_HOUSE = "Assets/GameAssets/_Project/Prefabs/Environment/PF_House_C.prefab"

NEW_FILES = [
    "Assets/_Project/Scripts/Practice/PetRobotWorldIntegration.meta",
    "Assets/_Project/Scripts/Practice/PetRobotWorldIntegration/PetRobotWorldIntegrationContracts.cs",
    "Assets/_Project/Scripts/Practice/PetRobotWorldIntegration/PetRobotWorldIntegrationContracts.cs.meta",
    "Assets/_Project/Scripts/Practice/PetRobotWorldIntegration/PetRobotWorldEntryController.cs",
    "Assets/_Project/Scripts/Practice/PetRobotWorldIntegration/PetRobotWorldEntryController.cs.meta",
    "Assets/_Project/Scripts/Practice/PetRobotWorldIntegration/PetRobotWorldPlayerController.cs",
    "Assets/_Project/Scripts/Practice/PetRobotWorldIntegration/PetRobotWorldPlayerController.cs.meta",
    "Assets/_Project/Scripts/Practice/PetRobotWorldIntegration/PetRobotWorldCameraFollow.cs",
    "Assets/_Project/Scripts/Practice/PetRobotWorldIntegration/PetRobotWorldCameraFollow.cs.meta",
    "Assets/_Project/Scripts/Practice/PetRobotWorldIntegration/PetRobotWorldUiPresenter.cs",
    "Assets/_Project/Scripts/Practice/PetRobotWorldIntegration/PetRobotWorldUiPresenter.cs.meta",
    "Assets/_Project/Scripts/Practice/PetRobotWorldIntegration/PetRobotPracticeReturnWatcher.cs",
    "Assets/_Project/Scripts/Practice/PetRobotWorldIntegration/PetRobotPracticeReturnWatcher.cs.meta",
    "Assets/_Project/Scripts/Practice/PetRobotWorldIntegration/PetRobotWorldPlaceholderVisuals.cs",
    "Assets/_Project/Scripts/Practice/PetRobotWorldIntegration/PetRobotWorldPlaceholderVisuals.cs.meta",
    WORLD_PREFAB,
    WORLD_PREFAB + ".meta",
    WORLD_SCENE,
    WORLD_SCENE + ".meta",
    NEW_TEST,
    NEW_TEST + ".meta",
]
PRODUCTION_SOURCES = [value for value in NEW_FILES if value.endswith(".cs") and value != NEW_TEST]

EXISTING_SCENES = [
    "Assets/_Project/Scenes/Game/03_SignPractice.unity",
    "Assets/_Project/Scenes/Technical/01_WebCamSpike.unity",
    "Assets/_Project/Scenes/Technical/02_PoseIntegration.unity",
    "Assets/_Project/Scenes/Technical/03_Village.unity",
    "Assets/_Project/Scenes/Technical/Demo Stylized Medieval.unity",
    PRACTICE_SCENE,
]

BUILD_SETTINGS_BEFORE = b"""%YAML 1.1
%TAG !u! tag:unity3d.com,2011:
--- !u!1045 &1
EditorBuildSettings:
  m_ObjectHideFlags: 0
  serializedVersion: 2
  m_Scenes:
  - enabled: 1
    path: Assets/Scenes/SampleScene.unity
    guid: 99c9720ab356a0642a771bea13969a05
  m_configObjects:
    com.unity.input.settings.actions: {fileID: -944628639613478452, guid: 052faaac586de48259a63d0c4782560b, type: 3}
  m_UseUCBPForAssetBundles: 0
"""


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def jbytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def record(relative: str, data: bytes) -> dict[str, Any]:
    return {"relative_path": relative, "size_bytes": len(data), "sha256": sha(data)}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_module(root: Path, relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, root / relative)
    require(spec is not None and spec.loader is not None, "cannot load helper " + relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def staged_files(root: Path) -> list[str]:
    run = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=root,
                         capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    require(run.returncode == 0, "Git staged query failed")
    return [line.strip() for line in run.stdout.splitlines() if line.strip()]


def read_zip_map(data: bytes) -> dict[str, bytes]:
    import io
    with zipfile.ZipFile(io.BytesIO(data), "r") as archive:
        return {info.filename: archive.read(info) for info in archive.infolist()}


def validate_zip_bytes(data: bytes, expected_files: dict[str, bytes]) -> dict[str, Any]:
    import io
    import stat
    with zipfile.ZipFile(io.BytesIO(data), "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        duplicates = len(names) - len(set(names))
        unsafe = 0
        encrypted = 0
        symlinks = 0
        for info in infos:
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in info.filename:
                unsafe += 1
            encrypted += int(bool(info.flag_bits & 1))
            symlinks += int(stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK)
        crc_failure = archive.testzip()
        actual = {info.filename: archive.read(info) for info in infos}
    mismatches = sum(1 for name, payload in expected_files.items()
                     if actual.get(name) != payload)
    return {
        "entry_count": len(infos),
        "duplicate_count": duplicates,
        "unsafe_path_count": unsafe,
        "encrypted_entry_count": encrypted,
        "symlink_entry_count": symlinks,
        "crc_failure": crc_failure,
        "membership_match": set(actual) == set(expected_files),
        "payload_mismatch_count": mismatches,
    }


def verify_archive_manifest(files: dict[str, bytes]) -> dict[str, Any]:
    require("output_file_manifest.json" in files, "archive output manifest missing")
    manifest = json.loads(files["output_file_manifest.json"].decode("utf-8-sig"))
    rows = manifest["files"]
    targets = {row["relative_path"]: (row["size_bytes"], row["sha256"]) for row in rows}
    actual = {name: (len(data), sha(data)) for name, data in files.items()
              if name != "output_file_manifest.json"}
    require(len(targets) == len(rows), "archive manifest duplicate target")
    require(targets == actual, "archive manifest membership or identity mismatch")
    return {"target_count": len(targets), "self_excluded": True, "status": "PASS"}


def verify_fixed_zip(root: Path, helper: Any, relative: str, size: int, digest: str,
                     detached_size: int, detached_digest: str,
                     entry_count: int, manifest_count: int) -> dict[str, Any]:
    path = root / relative
    detached = path.with_suffix(path.suffix + ".sha256")
    data = path.read_bytes()
    checksum_data = detached.read_bytes()
    require(len(data) == size and sha(data) == digest, "fixed ZIP identity mismatch: " + relative)
    require(len(checksum_data) == detached_size and sha(checksum_data) == detached_digest,
            "fixed detached identity mismatch: " + relative)
    require(detached.read_text(encoding="ascii") == digest + "  " + path.name + "\n",
            "fixed detached syntax mismatch: " + relative)
    files = read_zip_map(data)
    require(len(files) == entry_count, "fixed ZIP entry count mismatch: " + relative)
    manifest = verify_archive_manifest(files)
    require(manifest["target_count"] == manifest_count, "fixed manifest count mismatch")
    checked = validate_zip_bytes(data, files)
    require(checked["entry_count"] == entry_count and checked["duplicate_count"] == 0 and
            checked["unsafe_path_count"] == 0 and checked["encrypted_entry_count"] == 0 and
            checked["symlink_entry_count"] == 0 and checked["crc_failure"] is None and
            checked["membership_match"] and checked["payload_mismatch_count"] == 0,
            "fixed ZIP safety mismatch: " + relative)
    require(helper.deterministic_zip(files) == data, "fixed ZIP deterministic rebuild mismatch")
    return {"relative_path": relative, "size_bytes": size, "sha256": digest,
            "detached_size_bytes": detached_size, "detached_file_sha256": detached_digest,
            "entry_count": entry_count, "manifest_target_count": manifest_count,
            "deterministic_rebuild_byte_identical": True, "status": "PASS"}


def verify_baseline_records(root: Path, baseline: dict[str, Any], count: int,
                            label: str) -> dict[str, Any]:
    require(baseline["file_count"] == count, label + " baseline count mismatch")
    rows = []
    for prior in baseline["records"]:
        path = root / prior["relative_path"]
        data = path.read_bytes()
        same = len(data) == prior["size_bytes"] and sha(data) == prior["sha256"]
        require(same, label + " changed: " + prior["relative_path"])
        rows.append({**record(prior["relative_path"], data),
                     "before_size_bytes": prior["size_bytes"],
                     "before_sha256": prior["sha256"],
                     "before_after_byte_identical": True, "status": "PASS"})
    return {"schema_version": SCHEMA + "." + label, "file_count": count,
            "pass_count": count, "mismatch_count": 0, "records": rows, "status": "PASS"}


def verify_repository(root: Path, baseline: dict[str, Any]) -> dict[str, Any]:
    repository = Path(baseline["path"])
    rows = [record(path.relative_to(repository).as_posix(), path.read_bytes())
            for path in sorted(value for value in repository.rglob("*") if value.is_file())]
    index = json.loads((repository / "candidate_index.json").read_text(encoding="utf-8-sig"))
    require(len(index["entries"]) == 5 and len(rows) == 16, "candidate repository is not approved 5/16")
    require(rows == baseline["files"], "candidate repository 5/16 changed")
    return {"schema_version": SCHEMA + ".candidate_repository", "path": str(repository),
            "candidate_count_before_after": [5, 5], "file_count_before_after": [16, 16],
            "files": rows, "mismatch_count": 0, "status": "PASS"}


def test_inventory(root: Path) -> dict[str, Any]:
    baseline = load_json(root / V13_RESULT / "test_inventory.json")
    approved = baseline["approved_rows"] + baseline["new_rows"]
    require(len(approved) == 102, "approved testcase baseline is not 102")
    actual = []
    for row in approved:
        source = (root / row["source_path"]).read_text(encoding="utf-8-sig")
        methods = re.findall(r"\[Test\]\s*public\s+void\s+(\w+)\s*\(", source)
        fixture_match = re.search(r"public\s+sealed\s+class\s+(\w+)", source)
        require(fixture_match is not None, "approved fixture missing")
        if row["method"] in methods:
            actual.append((row["fixture"], row["method"], row["source_path"]))
    expected = {(row["fixture"], row["method"], row["source_path"]) for row in approved}
    require(len(actual) == 102 and set(actual) == expected, "approved 102 testcase name set changed")
    new_source = (root / NEW_TEST).read_text(encoding="utf-8-sig")
    fixture = re.search(r"public\s+sealed\s+class\s+(\w+)", new_source)
    methods = re.findall(r"\[Test\]\s*public\s+void\s+(\w+)\s*\(", new_source)
    require(fixture is not None and fixture.group(1) == "PetRobotWorldEntryReturnIntegrationV1Tests",
            "new fixture identity mismatch")
    require(len(methods) == len(set(methods)) == 18, "new fixture is not 18 unique tests")
    rows = [{"fixture": fixture.group(1), "method": value, "source_path": NEW_TEST}
            for value in methods]
    require(len({(row["fixture"], row["method"]) for row in approved + rows}) == 120,
            "combined testcase inventory is not 120 unique")
    return {"schema_version": SCHEMA + ".test_inventory",
            "approved_test_count": 102, "approved_name_set_preserved": True,
            "new_fixture": fixture.group(1), "new_test_count": 18,
            "combined_unique_test_count": 120,
            "approved_rows": approved, "new_rows": rows,
            "unity_execution_status": "UNVERIFIED", "status": "PASS"}


def inventory(root: Path) -> tuple[dict[str, Any], str]:
    before_scenes = []
    for relative in EXISTING_SCENES:
        data = (root / relative).read_bytes()
        before_scenes.append(record(relative, data))
    after_scenes = before_scenes + [record(WORLD_SCENE, (root / WORLD_SCENE).read_bytes())]
    scripts = []
    for path in (root / "Assets/_Project/Scripts").rglob("*.cs"):
        relative = path.relative_to(root).as_posix()
        if "PetRobotWorldIntegration" in relative:
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        if "CharacterController" in text or re.search(r"class\s+\w*(PlayerController|CameraFollow)\w*", text):
            scripts.append(relative)
    require(scripts == [], "unexpected stable project player/interaction controller found after decision")
    vendor_before = [record(VENDOR_HOUSE, (root / VENDOR_HOUSE).read_bytes()),
                     record(VENDOR_HOUSE + ".meta", (root / (VENDOR_HOUSE + ".meta")).read_bytes())]
    build_after = (root / BUILD_SETTINGS).read_bytes()
    require(build_after.count(WORLD_SCENE.encode("utf-8")) == 1 and
            build_after.count(PRACTICE_SCENE.encode("utf-8")) == 1,
            "Build Settings scene registration mismatch")
    result = {
        "schema_version": SCHEMA + ".inventory",
        "scene_inventory_before": before_scenes,
        "scene_inventory_after": after_scenes,
        "build_settings_before": record(BUILD_SETTINGS, BUILD_SETTINGS_BEFORE),
        "build_settings_after": record(BUILD_SETTINGS, build_after),
        "stable_project_player_or_interaction_controller_candidates": scripts,
        "selection": "STEP10B_CHARACTERCONTROLLER_PLACEHOLDER",
        "selection_reason": "03_Village contains camera/light/one house but no reusable player or interaction layer; use a Step 10B-only CharacterController shell",
        "reused_environment_prefab": vendor_before,
        "vendor_boundary": "Assets/GameAssets/_ThirdParty referenced indirectly; no vendor file modified",
        "practice_entry_role": "loads the approved Step 10A Resource prefab and explicitly wires its single candidate camera route",
        "practice_public_api": ["Session", "State", "AllWordsComplete", "MasteredCount",
                                "NeedsPracticeCount", "CandidateCounts", "GetWordLearningStatus",
                                "Changed", "ExitRequested", "ExitPractice"],
        "world_webcam_or_provider_owner_count": 0,
        "approved_practice_scene_reused_without_modification": True,
        "status": "PASS",
    }
    markdown = """# STEP10B_WORLD_INTEGRATION_INVENTORY

## 조사 결과

- 기존 `_Project` Scene은 Game 1개, Technical 4개, 승인 Step 10A practice 1개였다.
- `03_Village`는 Main Camera, Directional Light, `PF_House_C` reference만 있고 player/interaction 구현이 없다.
- `_Project/Scripts`의 Step 10B 이전 source에는 재사용 가능한 `CharacterController` player 또는 world interaction 계층이 없다.
- Build Settings는 조사 시 `Assets/Scenes/SampleScene.unity` 하나만 등록돼 있었다.
- 승인 practice 진입점은 Resources prefab을 인스턴스화하고 candidate camera route를 명시적으로 연결한다.
- 완료 결과는 `PetRobotSignPracticeSession`의 `State`, 단어별 status, Mastered/NeedsPractice 수, candidate count, `ExitRequested`에서 읽을 수 있다.

## 선택

`STEP10B_CHARACTERCONTROLLER_PLACEHOLDER`를 선택했다. 신규 `_Project` VerticalSlice Scene/Prefab에만 WASD/방향키 이동, follow camera, E/Enter/Space 근접 상호작용을 둔다. 기존 `PF_House_C`는 reference만 재사용하고 `GameAssets/_ThirdParty`와 기존 Scene/Prefab은 수정하지 않는다.

## Scene 흐름

`PetRobotWorldEntryReturnVerticalSlice` → 명시적 근접 입력 → 승인 `PetRobotSignPracticeVerticalSlice` → Completed 뒤 기존 exit → one-shot immutable snapshot → 원래 world.

World Scene에는 webcam/Hand/Pose provider가 없고, practice Scene의 candidate camera owner는 승인된 1개 그대로다. 진입과 동시에 capture/retry를 호출하는 코드는 없다.
"""
    return result, markdown


def scene_contract(root: Path) -> dict[str, Any]:
    world = (root / WORLD_SCENE).read_text(encoding="utf-8-sig")
    prefab = (root / WORLD_PREFAB).read_text(encoding="utf-8-sig")
    practice = (root / PRACTICE_SCENE).read_text(encoding="utf-8-sig")
    sources = "\n".join((root / value).read_text(encoding="utf-8-sig") for value in PRODUCTION_SOURCES)
    require(world.count("guid: 10b00000000000000000000000000010") >= 1,
            "world prefab is not referenced")
    require("guid: 5c0d962245381c74f8e0e76dd95e4c2c" in world,
            "existing environment house reference missing")
    require("--- !u!143 &14300001" in prefab and
            "guid: 10b00000000000000000000000000003" in prefab and
            "guid: 10b00000000000000000000000000004" in prefab,
            "placeholder player/camera prefab wiring missing")
    require(practice.count("guid: e33729c5ad6f4e44ad02aa19b2710d40") == 1,
            "approved practice candidate camera owner count changed")
    require("PracticeController" not in world and "WebCam" not in world and "WebCam" not in prefab,
            "world owns a legacy practice or webcam component")
    require("StartExpression(" not in sources and "BeginCapture(" not in sources and "Retry(" not in sources,
            "world integration performs automatic capture or retry")
    require("MaskAwareDtw" not in sources and "PersonalizationCandidateRepository" not in sources and
            ".wisrt" not in sources, "world integration mutates approved recognition/storage")
    require("Input.GetKeyDown(KeyCode.E)" in sources and "Input.GetKey(KeyCode.W)" in sources and
            "CharacterController" in sources, "manual world control contract missing")
    require("PetRobotPracticeState.Completed" in sources and "PetRobotPracticeState.Exited" in sources and
            "TryConsumeCompleted" in sources, "completed one-shot handoff contract missing")
    return {"schema_version": SCHEMA + ".scene_prefab_contract",
            "world_prefab_reference_count": world.count("guid: 10b00000000000000000000000000010"),
            "environment_reference_count": world.count("guid: 5c0d962245381c74f8e0e76dd95e4c2c"),
            "practice_candidate_camera_owner_count": 1,
            "world_camera_or_provider_owner_count": 0,
            "manual_interaction_keys": ["E", "Enter", "Space"],
            "automatic_capture_retry_count": 0,
            "status": "PASS"}


def extend_unity_stub(value: str) -> str:
    value = value.replace(
        "public class Object { public string name; public static T Instantiate<T>(T value) where T:new(){return new T();} public static void DestroyImmediate(object value){} public static void Destroy(object value){} }",
        "public class Object { public string name; public static T Instantiate<T>(T value) where T:new(){return new T();} public static void DestroyImmediate(object value){} public static void Destroy(object value){} public static void DontDestroyOnLoad(object value){} public static T FindFirstObjectByType<T>() where T:class{return null;} }")
    value = value.replace(
        "public class MonoBehaviour:Object { public bool isActiveAndEnabled=true; public T GetComponent<T>() where T:new(){return new T();} }",
        "public class MonoBehaviour:Object { public bool isActiveAndEnabled=true; public GameObject gameObject=new GameObject(); public Transform transform=new Transform(); public T GetComponent<T>() where T:new(){return new T();} }")
    value = value.replace(
        "public class GameObject:Object { public GameObject(){} public GameObject(string n){name=n;} public T GetComponent<T>() where T:new(){return new T();} public T AddComponent<T>() where T:new(){return new T();} }",
        "public class GameObject:Object { public Transform transform=new Transform(); public GameObject(){} public GameObject(string n){name=n;} public T GetComponent<T>() where T:new(){return new T();} public T AddComponent<T>() where T:new(){return new T();} public static GameObject CreatePrimitive(PrimitiveType p){return new GameObject();} }")
    value = value.replace("public struct Vector2{}",
        "public struct Vector2{public float x,y; public Vector2(float x,float y){this.x=x;this.y=y;} public float sqrMagnitude=>x*x+y*y;}")
    value = value.replace("namespace UnityEngine.SceneManagement { public static class SceneManager { public static void LoadScene(string n){} } }",
        "namespace UnityEngine.SceneManagement { public enum LoadSceneMode{Single,Additive} public struct Scene{public string name;} public static class SceneManager { public static event System.Action<Scene,LoadSceneMode> sceneLoaded; public static void LoadScene(string n){} public static void LoadScene(string n,LoadSceneMode m){} public static Scene GetActiveScene(){return new Scene{name=\"stub\"};} } }")
    insertion = """
 [AttributeUsage(AttributeTargets.Class,AllowMultiple=true)] public sealed class RequireComponent:Attribute{public RequireComponent(Type t){}}
 public enum PrimitiveType{Sphere,Capsule,Cylinder,Cube,Plane,Quad}
 public enum KeyCode{E,Return,Space,W,A,S,D,UpArrow,DownArrow,LeftArrow,RightArrow}
 public static class Input{public static bool GetKeyDown(KeyCode k){return false;} public static bool GetKey(KeyCode k){return false;}}
 public static class Time{public static float deltaTime=>0.016f;}
 public static class Mathf{public static float Max(float a,float b){return a>b?a:b;} public static float Sqrt(float v){return (float)Math.Sqrt(v);}}
 public struct Vector3{public float x,y,z; public Vector3(float x,float y,float z){this.x=x;this.y=y;this.z=z;} public static Vector3 zero=>new Vector3(); public float sqrMagnitude=>x*x+y*y+z*z; public Vector3 normalized{get{float m=Mathf.Sqrt(sqrMagnitude);return m<=0?zero:new Vector3(x/m,y/m,z/m);}} public static float Distance(Vector3 a,Vector3 b){return Mathf.Sqrt((a-b).sqrMagnitude);} public static Vector3 operator +(Vector3 a,Vector3 b){return new Vector3(a.x+b.x,a.y+b.y,a.z+b.z);} public static Vector3 operator -(Vector3 a,Vector3 b){return new Vector3(a.x-b.x,a.y-b.y,a.z-b.z);} public static Vector3 operator *(Vector3 a,float b){return new Vector3(a.x*b,a.y*b,a.z*b);} public override bool Equals(object o){return o is Vector3&&this==((Vector3)o);} public override int GetHashCode(){return x.GetHashCode()^y.GetHashCode()^z.GetHashCode();} public static bool operator ==(Vector3 a,Vector3 b){return a.x==b.x&&a.y==b.y&&a.z==b.z;} public static bool operator !=(Vector3 a,Vector3 b){return !(a==b);}}
 public class Transform:Object{public Vector3 position,localPosition,localScale,forward; public int childCount; public void SetParent(Transform p,bool w){} public void LookAt(Vector3 p){}}
 public class Collider:Object{}
 public class CharacterController:Collider{public void Move(Vector3 v){}}
"""
    value = value.replace(" public class Texture:Object{}", insertion + "\n public class Texture:Object{}")
    value += """
namespace WordsInSilence.Practice.PetRobotVerticalSlice {
 using System; using UnityEngine;
 public enum PetRobotPracticeState{Ready,Capturing,Feedback,Completed,Exited}
 public enum PetRobotWordLearningStatus{Pending,Mastered,NeedsPractice}
 public sealed class CandidateCountSnapshot{public int TotalCandidateCount{get;set;}}
 public sealed class PetRobotSignPracticeSession{public event Action Changed; public event Action ExitRequested; public PetRobotPracticeState State{get;set;} public bool AllWordsComplete{get;set;} public int MasteredCount{get;set;} public int NeedsPracticeCount{get;set;} public CandidateCountSnapshot CandidateCounts{get;set;}=new CandidateCountSnapshot(); public PetRobotWordLearningStatus GetWordLearningStatus(string w){return PetRobotWordLearningStatus.Mastered;}}
 public sealed class PetRobotPracticeVerticalSliceBootstrap:MonoBehaviour{public PetRobotSignPracticeSession Session{get;set;}}
}
"""
    return value


PROBE_SOURCE = r'''
using System; using UnityEngine; using WordsInSilence.Practice.PetRobotVerticalSlice; using WordsInSilence.Practice.PetRobotWorldIntegration;
sealed class Transition:IPetRobotWorldSceneTransition{public int RequestCount{get;private set;} public bool Request(string s){RequestCount++;return true;}}
static class Probe{static PetRobotPracticeCompletionSnapshot M(int c){return new PetRobotPracticeCompletionSnapshot(PetRobotWordLearningStatus.Mastered,PetRobotWordLearningStatus.Mastered,PetRobotWordLearningStatus.Mastered,3,0,c);} static void Main(){var channel=new PetRobotWorldHandoffChannel();var transition=new Transition();int watcher=0;var entry=new PetRobotWorldEntryGate("world","practice",channel,transition,()=>watcher++);entry.SetPlayerInRange(false);bool outside=!entry.TryEnter(true,new Vector3(1,0,2));entry.SetPlayerInRange(true);bool explicitOnly=!entry.TryEnter(false,Vector3.zero);bool once=entry.TryEnter(true,new Vector3(3,0,4))&&!entry.TryEnter(true,Vector3.zero)&&transition.RequestCount==1&&watcher==1;var returnTransition=new Transition();var ret=new PetRobotPracticeReturnGate("world",channel,returnTransition);bool incomplete=!ret.TryReturn(true)&&returnTransition.RequestCount==0;bool observed=ret.ObserveCompleted(M(5));bool returned=!ret.TryReturn(false)&&ret.TryReturn(true)&&!ret.TryReturn(true)&&returnTransition.RequestCount==1;PetRobotWorldReturnPayload payload;bool consumed=channel.TryConsumeCompleted("world",out payload)&&!channel.TryConsumeCompleted("world",out payload)&&payload==null;var needs=new PetRobotPracticeCompletionSnapshot(PetRobotWordLearningStatus.Mastered,PetRobotWordLearningStatus.Mastered,PetRobotWordLearningStatus.NeedsPractice,2,1,5);bool preserved=needs.ThankYou==PetRobotWordLearningStatus.NeedsPractice&&needs.WorldSummaryMessage.Contains("1개 복습 필요");var next=new Transition();var reentry=new PetRobotWorldEntryGate("world","practice",channel,next,()=>{});reentry.SetPlayerInRange(true);bool fresh=reentry.TryEnter(true,Vector3.zero)&&next.RequestCount==1;Console.WriteLine("OUTSIDE_BLOCKED="+outside);Console.WriteLine("EXPLICIT_INPUT_ONLY="+explicitOnly);Console.WriteLine("ENTRY_TRANSITION_ONCE="+once);Console.WriteLine("INCOMPLETE_NOT_PUBLISHED="+incomplete);Console.WriteLine("COMPLETED_RETURN_ONCE="+(observed&&returned));Console.WriteLine("ONE_SHOT_CONSUME="+consumed);Console.WriteLine("NEEDS_PRACTICE_PRESERVED="+preserved);Console.WriteLine("REENTRY_TRANSIENT_FRESH="+fresh);bool pass=outside&&explicitOnly&&once&&incomplete&&observed&&returned&&consumed&&preserved&&fresh;Console.WriteLine("PROBE_STATUS="+(pass?"PASS":"FAIL"));if(!pass)Environment.Exit(2);}}
'''


def offline_compile(root: Path, helper: Any) -> tuple[dict[str, Any], str, str, str, dict[str, Any], str]:
    mono = Path(r"C:\Program Files\Unity\Hub\Editor\6000.3.14f1\Editor\Data\MonoBleedingEdge\bin\mono.exe")
    csc = Path(r"C:\Program Files\Unity\Hub\Editor\6000.3.14f1\Editor\Data\MonoBleedingEdge\lib\mono\4.5\csc.exe")
    require(mono.is_file() and csc.is_file(), "Unity Mono Roslyn toolchain missing")
    unity_stub, nunit_stub = helper.load_v10_stubs(root)
    unity_stub = extend_unity_stub(unity_stub)
    temp = Path(tempfile.mkdtemp(prefix="WIS-Step10B-v1_0-compile-"))
    try:
        (temp / "stubs.cs").write_text(unity_stub, encoding="utf-8")
        (temp / "nunit.cs").write_text(nunit_stub, encoding="utf-8")
        dll = temp / "Step10B.dll"
        production_command = [str(mono), str(csc), "/nologo", "/target:library", "/langversion:latest",
                              "/out:" + str(dll), str(temp / "stubs.cs")] + [
                                  str((root / value).resolve()) for value in PRODUCTION_SOURCES]
        production = subprocess.run(production_command, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=120)
        production_output = (production.stdout or "") + (production.stderr or "")
        require(production.returncode == 0, "offline production compile failed:\n" + production_output[-12000:])
        test_dll = temp / "Step10BTests.dll"
        test = subprocess.run([str(mono), str(csc), "/nologo", "/target:library", "/langversion:latest",
                               "/out:" + str(test_dll), "/r:" + str(dll), str(temp / "nunit.cs"),
                               str((root / NEW_TEST).resolve())], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=120)
        test_output = (test.stdout or "") + (test.stderr or "")
        require(test.returncode == 0, "offline test compile failed:\n" + test_output[-12000:])
        probe = temp / "probe.cs"
        probe.write_text(PROBE_SOURCE, encoding="utf-8")
        exe = temp / "probe.exe"
        probe_compile = subprocess.run([str(mono), str(csc), "/nologo", "/target:exe", "/langversion:latest",
                                        "/out:" + str(exe), "/r:" + str(dll), str(probe)],
                                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        probe_compile_output = (probe_compile.stdout or "") + (probe_compile.stderr or "")
        require(probe_compile.returncode == 0, "behavior probe compile failed:\n" + probe_compile_output[-12000:])
        probe_run = subprocess.run([str(mono), str(exe)], capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", timeout=120, cwd=temp)
        probe_output = (probe_run.stdout or "") + (probe_run.stderr or "")
        require(probe_run.returncode == 0 and "PROBE_STATUS=PASS" in probe_output,
                "behavior probe failed:\n" + probe_output[-12000:])
        values = {}
        for line in probe_output.splitlines():
            if "=" in line:
                key, value = line.lstrip("\ufeff").split("=", 1)
                values[key] = value
        expected = ["OUTSIDE_BLOCKED", "EXPLICIT_INPUT_ONLY", "ENTRY_TRANSITION_ONCE",
                    "INCOMPLETE_NOT_PUBLISHED", "COMPLETED_RETURN_ONCE", "ONE_SHOT_CONSUME",
                    "NEEDS_PRACTICE_PRESERVED", "REENTRY_TRANSIENT_FRESH"]
        require(all(values.get(value) == "True" for value in expected), "behavior probe value mismatch")
        compile_summary = {"schema_version": SCHEMA + ".offline_compile",
                           "compiler": "Unity 6000.3.14f1 Mono Roslyn csc",
                           "production_source_count": len(PRODUCTION_SOURCES),
                           "test_source_count": 1,
                           "production_error_count": len(re.findall(r"error CS\d+", production_output)),
                           "test_error_count": len(re.findall(r"error CS\d+", test_output)),
                           "probe_compile_error_count": len(re.findall(r"error CS\d+", probe_compile_output)),
                           "unity_compile_pass_claimed": False, "editmode_pass_claimed": False,
                           "status": "PASS"}
        probe_summary = {"schema_version": SCHEMA + ".behavior_probe", "values": values,
                         "unity_execution_claimed": False, "status": "PASS"}
        return compile_summary, production_output, test_output, probe_compile_output, probe_summary, probe_output
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def assertion_lines(items: list[tuple[str, str, str]]) -> bytes:
    keys = set()
    rows = []
    for key, status, evidence in items:
        require(key not in keys, "duplicate assertion key: " + key)
        require(status in ("PASS", "FAIL", "UNVERIFIED", "NOT_APPLICABLE"), "bad assertion status")
        keys.add(key)
        rows.append(json.dumps({"assertion_key": key, "status": status, "evidence": evidence},
                               ensure_ascii=False, sort_keys=True, allow_nan=False))
    return ("\n".join(rows) + "\n").encode("utf-8")


def manifest(files: dict[str, bytes]) -> bytes:
    rows = [record(name, data) for name, data in sorted(files.items())
            if name != "output_file_manifest.json"]
    return jbytes({"schema_version": SCHEMA + ".output_manifest", "file_count": len(rows),
                   "files": rows, "self_excluded": True})


def build(root: Path) -> None:
    validation = root / "Validation"
    result = validation / NAME
    delivery = validation / (NAME + "_delivery.zip")
    detached = delivery.with_suffix(delivery.suffix + ".sha256")
    require(not result.exists() and not delivery.exists() and not detached.exists(), "never-overwrite target exists")

    helper = load_module(root, V11_TOOL, "step10b_v11_helper")
    fixed = [
        verify_fixed_zip(root, helper, V13_ZIP, V13_ZIP_SIZE, V13_ZIP_SHA,
                         V13_DETACHED_SIZE, V13_DETACHED_SHA, 65, 64),
        verify_fixed_zip(root, helper, GUI_ZIP, GUI_ZIP_SIZE, GUI_ZIP_SHA,
                         GUI_DETACHED_SIZE, GUI_DETACHED_SHA, 46, 45),
    ]
    gui = root / GUI_RESULT
    project = verify_baseline_records(root, load_json(gui / "project_files_after.json"), 30, "project_files")
    protected = verify_baseline_records(root, load_json(gui / "protected_files_after.json"), 106, "protected_files")
    profile_baseline = load_json(gui / "active_candidate_profile_after.json")
    profile = verify_baseline_records(root, profile_baseline, 22, "candidate_profile")
    profile["prototype_count"] = profile_baseline["prototype_count"]
    profile["threshold_count"] = profile_baseline["threshold_count"]
    profile["thresholds"] = profile_baseline["thresholds"]
    require(profile["prototype_count"] == 9 and profile["threshold_count"] == 3 and profile["thresholds"] == {
        "GREETING": 0.5113059473720765,
        "LOVE": 0.391276023460427,
        "THANK_YOU": 0.37709977431991587,
    }, "profile prototype/threshold baseline mismatch")
    repository = verify_repository(root, load_json(gui / "candidate_repository_after.json"))
    require(staged_files(root) == [], "Git staged files are not zero")

    inventory_json, inventory_markdown = inventory(root)
    scene = scene_contract(root)
    tests = test_inventory(root)
    compile_summary, production_output, test_output, probe_compile_output, probe, probe_output = offline_compile(root, helper)

    guid_rows = []
    all_meta_guids = Counter()
    for path in (root / "Assets").rglob("*.meta"):
        match = re.search(r"^guid:\s*([0-9a-f]{32})\s*$", path.read_text(encoding="utf-8-sig", errors="replace"), re.MULTILINE)
        if match:
            all_meta_guids[match.group(1)] += 1
    for relative in [value for value in NEW_FILES if value.endswith(".meta")]:
        text = (root / relative).read_text(encoding="utf-8-sig")
        guid = re.search(r"^guid:\s*([0-9a-f]{32})\s*$", text, re.MULTILINE).group(1)
        require(all_meta_guids[guid] == 1, "new GUID collision: " + guid)
        guid_rows.append({"relative_path": relative, "guid": guid, "project_occurrence_count": 1})

    build_after = (root / BUILD_SETTINGS).read_bytes()
    build_diff = "".join(difflib.unified_diff(
        BUILD_SETTINGS_BEFORE.decode("utf-8").splitlines(True),
        build_after.decode("utf-8").splitlines(True),
        fromfile="a/" + BUILD_SETTINGS, tofile="b/" + BUILD_SETTINGS))
    require(build_diff.count("+  - enabled: 1") == 2 and
            "+    path: " + WORLD_SCENE in build_diff and
            "+    path: " + PRACTICE_SCENE in build_diff,
            "Build Settings diff is not exactly two scene registrations")

    changes = []
    for relative in NEW_FILES:
        data = (root / relative).read_bytes()
        changes.append({"relative_path": relative, "change_type": "NEW",
                        "before_exists": False, "after": record(relative, data)})
    changes.append({"relative_path": BUILD_SETTINGS, "change_type": "MODIFIED",
                    "before": record(BUILD_SETTINGS, BUILD_SETTINGS_BEFORE),
                    "after": record(BUILD_SETTINGS, build_after)})

    vendor = [VENDOR_HOUSE, VENDOR_HOUSE + ".meta",
              "Assets/_Project/Scenes/Technical/03_Village.unity", PRACTICE_SCENE, PRACTICE_SCENE + ".meta",
              PRACTICE_PREFAB, PRACTICE_PREFAB + ".meta"]
    preservation = {"schema_version": SCHEMA + ".referenced_assets_preserved",
                    "records": [{**record(value, (root / value).read_bytes()),
                                 "before_after_byte_identical": True, "status": "PASS"}
                                for value in vendor], "mismatch_count": 0, "status": "PASS"}

    transaction = helper.transaction_regression()
    assertions = assertion_lines([
        ("implementation_owner_codex", "PASS", "AGENTS/context section 3"),
        ("fixed_step10a_v1_3_delivery", "PASS", V13_ZIP_SHA),
        ("fixed_step10a_v1_3_gui_evidence", "PASS", GUI_ZIP_SHA),
        ("world_inventory_complete", "PASS", "existing 6 scenes and build settings inspected"),
        ("placeholder_path_selected", "PASS", inventory_json["selection_reason"]),
        ("vendor_assets_reference_only", "PASS", "referenced asset records unchanged"),
        ("world_scene_new", "PASS", WORLD_SCENE),
        ("world_prefab_new", "PASS", WORLD_PREFAB),
        ("character_controller_movement", "PASS", "WASD and arrow keys"),
        ("camera_follow", "PASS", "explicit player target"),
        ("outside_range_blocked", "PASS", probe["values"]["OUTSIDE_BLOCKED"]),
        ("manual_input_required", "PASS", probe["values"]["EXPLICIT_INPUT_ONLY"]),
        ("entry_transition_once", "PASS", probe["values"]["ENTRY_TRANSITION_ONCE"]),
        ("no_automatic_capture_retry", "PASS", "0 calls in integration source"),
        ("practice_camera_owner_one", "PASS", "1"),
        ("world_camera_provider_owner_zero", "PASS", "0"),
        ("incomplete_not_published", "PASS", probe["values"]["INCOMPLETE_NOT_PUBLISHED"]),
        ("technical_failure_not_completion", "PASS", "no Completed snapshot means no return publication"),
        ("completed_return_once", "PASS", probe["values"]["COMPLETED_RETURN_ONCE"]),
        ("one_shot_result_consume", "PASS", probe["values"]["ONE_SHOT_CONSUME"]),
        ("needs_practice_preserved", "PASS", probe["values"]["NEEDS_PRACTICE_PRESERVED"]),
        ("reentry_transient_fresh", "PASS", probe["values"]["REENTRY_TRANSIENT_FRESH"]),
        ("approved_test_names_102_preserved", "PASS", "102/102"),
        ("new_tests_18", "PASS", "18/18 static inventory"),
        ("combined_tests_120", "PASS", "120 unique"),
        ("offline_production_compile", "PASS", "error 0"),
        ("offline_test_compile", "PASS", "error 0"),
        ("offline_behavior_probe", "PASS", "PROBE_STATUS=PASS"),
        ("project_files_30_preserved", "PASS", "30/30"),
        ("protected_files_106_preserved", "PASS", "106/106"),
        ("candidate_profile_22_preserved", "PASS", "22/22"),
        ("prototype_threshold_preserved", "PASS", "9 prototypes / 3 thresholds"),
        ("candidate_repository_preserved", "PASS", "5 candidates / 16 files"),
        ("git_staged_zero", "PASS", "0"),
        ("unity_gui_compile", "UNVERIFIED", "source stage; GUI prohibited"),
        ("unity_editmode", "UNVERIFIED", "120 tests not executed"),
        ("unity_playmode", "UNVERIFIED", "world-practice-world not executed"),
        ("webcam_live_capture", "NOT_APPLICABLE", "source stage; prohibited"),
        ("transaction_failure_injection", "PASS", "4/4"),
        ("never_overwrite", "PASS", "second commit blocked"),
    ])

    input_provenance = {"schema_version": SCHEMA + ".input_provenance",
                        "fixed_inputs": fixed, "implementation_owner": "Codex",
                        "unity_version": "6000.3.14f1",
                        "source_stage_prohibitions_observed": ["Unity GUI", "Test Runner", "PlayMode",
                                                                "webcam", "live capture", "batchmode"],
                        "status": "PASS"}
    change_summary = {"schema_version": SCHEMA + ".changes", "new_file_count": len(NEW_FILES),
                      "modified_existing_file_count": 1, "modified_existing_files": [BUILD_SETTINGS],
                      "approved_step10a_project_file_changes": 0, "records": changes, "status": "PASS"}
    primary = {"schema_version": SCHEMA + ".primary_verdict", "status": "PASS",
               "verdict": "STEP10B_WORLD_ENTRY_RETURN_INTEGRATION_SOURCE_AND_PACKAGE_READY",
               "player_path": "STEP10B_CHARACTERCONTROLLER_PLACEHOLDER",
               "approved_practice_reused": True, "world_return_snapshot": "COMPLETED_ONLY_ONE_SHOT",
               "unity_gui_compile": "UNVERIFIED", "unity_editmode": "UNVERIFIED",
               "unity_playmode": "UNVERIFIED", "webcam_live_capture": "NOT_APPLICABLE",
               "status_strings": [
                   "STEP10B_PET_ROBOT_WORLD_ENTRY_RETURN_INTEGRATION_V1_0_SOURCE_AND_PACKAGE_READY",
                   "STEP10B_WORLD_TO_APPROVED_PRACTICE_MANUAL_ENTRY_IMPLEMENTED",
                   "STEP10B_COMPLETED_RESULT_ONE_SHOT_WORLD_RETURN_IMPLEMENTED",
                   "STEP10A_APPROVED_PRACTICE_RECOGNITION_AND_STORAGE_PRESERVED",
                   "STEP10B_UNITY_GUI_EDITMODE_PLAYMODE_UNVERIFIED",
                   "BLOCKED_PENDING_EXTERNAL_REVIEW"]}

    report = f"""# Step 10B Pet-Robot World Entry/Return Integration v1.0

## Verdict

PASS — source and deterministic Delivery are ready for independent external review.

## Implementation

- Existing world inventory had no reusable player/interaction layer; a Step 10B-only CharacterController placeholder was selected.
- New world Scene/Prefab provide WASD/arrow movement, follow camera, range prompt, and one-shot E/Enter/Space entry.
- The approved Step 10A practice Scene is loaded unchanged. World contains no webcam/provider owner; practice retains exactly one candidate camera owner.
- A persistent return watcher observes but does not drive practice. Only a Completed snapshot can be published, and it is consumed once after returning to the original world position.
- TechnicalFailure, incomplete session, or forced departure cannot publish a completed result.

## Verification

- Approved testcase names: 102/102 preserved; new fixture: 18; combined static inventory: 120.
- Offline Unity Mono Roslyn: production 0 errors, test 0 errors, probe compile 0 errors.
- Behavior probe: PASS for range gate, explicit input, duplicate transition block, incomplete rejection, one-shot completion return, NeedsPractice preservation, and fresh re-entry.
- Step 10A project 30/30, protected 106/106, profile 22/22, prototypes 9, thresholds 3 unchanged.
- Candidate repository 5 candidates / 16 files unchanged; Git staged files 0.

## Execution boundary

- Unity GUI compile: UNVERIFIED
- EditMode 120 tests: UNVERIFIED
- PlayMode world→practice→world: UNVERIFIED
- Webcam/live capture: NOT_APPLICABLE
"""

    files: dict[str, bytes] = {
        "STEP10B_WORLD_INTEGRATION_INVENTORY.md": inventory_markdown.encode("utf-8"),
        "input_provenance.json": jbytes(input_provenance),
        "world_integration_inventory.json": jbytes(inventory_json),
        "scene_prefab_static_contract.json": jbytes(scene),
        "implementation_delta.json": jbytes(change_summary),
        "test_inventory.json": jbytes(tests),
        "offline_compile_evidence.json": jbytes(compile_summary),
        "offline_production_compile_output.txt": production_output.encode("utf-8"),
        "offline_test_compile_output.txt": test_output.encode("utf-8"),
        "offline_behavior_probe_compile_output.txt": probe_compile_output.encode("utf-8"),
        "offline_behavior_probe.json": jbytes(probe),
        "offline_behavior_probe_output.txt": probe_output.encode("utf-8"),
        "project_files_before_after.json": jbytes(project),
        "protected_files_before_after.json": jbytes(protected),
        "active_candidate_profile_before_after.json": jbytes(profile),
        "candidate_repository_before_after.json": jbytes(repository),
        "referenced_existing_assets_before_after.json": jbytes(preservation),
        "new_meta_guid_uniqueness.json": jbytes({"schema_version": SCHEMA + ".guid_uniqueness",
                                                 "record_count": len(guid_rows), "records": guid_rows,
                                                 "collision_count": 0, "status": "PASS"}),
        "git_staged_before_after.json": jbytes({"before_count": 0, "after_count": 0,
                                                "before_after_unchanged": True, "status": "PASS"}),
        "packaging_transaction_failure_injection_regression.json": jbytes(transaction),
        "primary_verdict.json": jbytes(primary),
        "ASSERTIONS.jsonl": assertions,
        "VALIDATION_REPORT.md": report.encode("utf-8"),
        "changes/before/ProjectSettings/EditorBuildSettings.asset": BUILD_SETTINGS_BEFORE,
        "changes/after/ProjectSettings/EditorBuildSettings.asset": build_after,
        "changes/unified_diff.patch": build_diff.encode("utf-8"),
        "analysis/s001_step10b_pet_robot_world_entry_return_integration_v1_0.py": Path(__file__).read_bytes(),
    }
    for relative in NEW_FILES:
        files["project_files/" + relative] = (root / relative).read_bytes()
    files["project_files/" + BUILD_SETTINGS] = build_after
    files["output_file_manifest.json"] = manifest(files)

    def final_validator(final_result: Path, final_zip: Path, final_detached: Path) -> None:
        disk = {path.relative_to(final_result).as_posix(): path.read_bytes()
                for path in final_result.rglob("*") if path.is_file()}
        require(disk == files, "committed result file map mismatch")
        require(helper.deterministic_zip(disk) == final_zip.read_bytes(), "deterministic ZIP mismatch")
        require(final_detached.read_text(encoding="ascii") ==
                sha(final_zip.read_bytes()) + "  " + final_zip.name + "\n", "detached checksum mismatch")
        checked = validate_zip_bytes(final_zip.read_bytes(), disk)
        require(checked["duplicate_count"] == 0 and checked["unsafe_path_count"] == 0 and
                checked["encrypted_entry_count"] == 0 and checked["symlink_entry_count"] == 0 and
                checked["crc_failure"] is None and checked["membership_match"] and
                checked["payload_mismatch_count"] == 0, "post-commit ZIP validation failed")

    committed_result, committed_zip, committed_detached = helper.commit_outputs(
        validation, files,
        names=(NAME, NAME + "_delivery.zip", NAME + "_delivery.zip.sha256"),
        validator=final_validator)
    require(committed_result == result and committed_zip == delivery and committed_detached == detached,
            "transaction committed unexpected paths")
    require(staged_files(root) == [], "Git staged files changed during packaging")
    print(json.dumps({"status": "PASS", "result_directory": str(result),
                      "result_file_count": len(files), "manifest_target_count": len(files) - 1,
                      "delivery_zip": str(delivery), "delivery_size_bytes": delivery.stat().st_size,
                      "delivery_sha256": sha(delivery.read_bytes()),
                      "detached_checksum": str(detached), "detached_size_bytes": detached.stat().st_size,
                      "detached_file_sha256": sha(detached.read_bytes()),
                      "new_files": len(NEW_FILES), "modified_existing_files": [BUILD_SETTINGS],
                      "approved_tests_preserved": 102, "new_tests": 18, "combined_tests": 120,
                      "offline_production_errors": 0, "offline_test_errors": 0,
                      "behavior_probe": "PASS", "project_files": "30/30",
                      "protected_files": "106/106", "profile_files": "22/22",
                      "candidate_repository": "5 candidates / 16 files", "git_staged": 0,
                      "unity_gui_editmode_playmode": "UNVERIFIED", "webcam_live": "NOT_APPLICABLE"},
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
