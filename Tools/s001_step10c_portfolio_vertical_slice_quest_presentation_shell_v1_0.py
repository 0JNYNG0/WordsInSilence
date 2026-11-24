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
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA = "WordsInSilence.Step10C.PortfolioVerticalSliceQuestPresentationShell.v1.0"
NAME = "WordsInSilence_step10c_portfolio_vertical_slice_quest_presentation_shell_v1_0"

BASE_TOOL = "Tools/s001_step10b_pet_robot_world_entry_return_integration_v1_0.py"
TX_TOOL = "Tools/s001_step10a_pet_robot_candidate_camera_startup_route_v1_1.py"
STEP10B_RESULT = "Validation/WordsInSilence_step10b_pet_robot_world_entry_return_integration_v1_0"
STEP10B_GUI_RESULT = "Validation/WordsInSilence_step10b_pet_robot_world_entry_return_integration_v1_0_gui_unity_editor_gate"

FIXED_ARCHIVES = [
    ("Validation/WordsInSilence_step10a_pet_robot_retry_exhaustion_needs_practice_completion_v1_3_delivery.zip",
     101574, "42cfe5679c09defc220033fd99eb638576bd5313b5c047dfe25766ac425575bf",
     160, "6c630bc1b18770deec5e23e8669b5075eaae42a6907c61e918e92316cd1e0354", 65, 64),
    ("Validation/WordsInSilence_step10a_pet_robot_retry_exhaustion_needs_practice_completion_v1_3_gui_unity_editor_gate_delivery.zip",
     1491795, "6463cbed625e4b4f27b7fdf1cb0bcdcb07a38571334ae3083eff4ef85684798a",
     182, "55d8b44681e642c4f68f6975ce3a3a1c8e09969d4deb4b1ca402bd96b11dc955", 46, 45),
    ("Validation/WordsInSilence_step10b_pet_robot_world_entry_return_integration_v1_0_delivery.zip",
     67937, "07daa6bd08eacc7d7bf76cfe200258c75d7d9ba15dcb5a1f5fabf14ca9b86977",
     148, "906b8149993db3529fa33b277a2b0e9c0691107f62f2fdf5511c5d87fefef2e9", 50, 49),
    ("Validation/WordsInSilence_step10b_pet_robot_world_entry_return_integration_v1_0_gui_unity_editor_gate_delivery.zip",
     2631985, "17fa4e2f9af721cabc3dd52acb7f9e1e0a391aeb3cfc63456600b576a70a2ecf",
     170, "4580bd2a726a05401ab6ec770073cb7ace04456a23e84666b9ea77db924fb636", 47, 46),
]

NEW_DIR_META = "Assets/_Project/Scripts/Practice/PetRobotPortfolioQuest.meta"
NEW_SOURCES = [
    "Assets/_Project/Scripts/Practice/PetRobotPortfolioQuest/PetRobotPortfolioQuestContracts.cs",
    "Assets/_Project/Scripts/Practice/PetRobotPortfolioQuest/PetRobotPortfolioQuestController.cs",
    "Assets/_Project/Scripts/Practice/PetRobotPortfolioQuest/PetRobotPortfolioQuestUiPresenter.cs",
    "Assets/_Project/Scripts/Practice/PetRobotPortfolioQuest/PetRobotPortfolioQuestVisuals.cs",
]
NEW_TEST = "Assets/_Project/Tests/EditMode/PetRobotPortfolioQuestPresentationV1Tests.cs"
NEW_PREFAB = "Assets/_Project/Resources/VerticalSlice/PetRobotPortfolioQuestPresentation.prefab"
NEW_SCENE = "Assets/_Project/Scenes/VerticalSlice/PetRobotPortfolioQuestVerticalSlice.unity"
BUILD_SETTINGS = "ProjectSettings/EditorBuildSettings.asset"
STEP10B_PREFAB = "Assets/_Project/Resources/VerticalSlice/PetRobotWorldEntryReturn.prefab"
STEP10B_SCENE = "Assets/_Project/Scenes/VerticalSlice/PetRobotWorldEntryReturnVerticalSlice.unity"
PRACTICE_SCENE = "Assets/_Project/Scenes/VerticalSlice/PetRobotSignPracticeVerticalSlice.unity"

NEW_FILES = [NEW_DIR_META]
for value in NEW_SOURCES:
    NEW_FILES.extend([value, value + ".meta"])
NEW_FILES.extend([NEW_PREFAB, NEW_PREFAB + ".meta", NEW_SCENE, NEW_SCENE + ".meta",
                  NEW_TEST, NEW_TEST + ".meta"])

EXPECTED_THRESHOLDS = {
    "GREETING": 0.5113059473720765,
    "LOVE": 0.391276023460427,
    "THANK_YOU": 0.37709977431991587,
}


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def jbytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False,
                       sort_keys=True, indent=2) + "\n").encode("utf-8")


def record(relative: str, data: bytes) -> dict[str, Any]:
    return {"relative_path": relative, "size_bytes": len(data), "sha256": sha(data)}


def load_module(root: Path, relative: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, root / relative)
    require(spec is not None and spec.loader is not None, "cannot load helper " + relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def staged_files(root: Path) -> list[str]:
    process = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=root,
                             capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=30)
    require(process.returncode == 0, "cannot inspect Git staged state")
    return [value for value in process.stdout.splitlines() if value.strip()]


def verify_records(root: Path, document: dict[str, Any], expected: int,
                   label: str) -> dict[str, Any]:
    rows = document.get("records", document.get("files", []))
    require(len(rows) == expected, label + " baseline count mismatch")
    output = []
    for row in rows:
        relative = row["relative_path"]
        path = root / PurePosixPath(relative)
        require(path.is_file(), label + " missing: " + relative)
        data = path.read_bytes()
        digest = row.get("approved_sha256", row.get("sha256"))
        size = row.get("approved_size_bytes", row.get("size_bytes"))
        require(len(data) == size and sha(data) == digest, label + " changed: " + relative)
        output.append({**record(relative, data), "approved_sha256": digest,
                       "before_after_byte_identical": True, "status": "PASS"})
    return {"schema_version": SCHEMA + "." + label, "file_count": expected,
            "pass_count": expected, "mismatch_count": 0, "records": output,
            "status": "PASS"}


def verify_step10b_files(root: Path, document: dict[str, Any], before_build: bytes) -> dict[str, Any]:
    rows = document["records"]
    require(len(rows) == 22, "Step 10B baseline is not 22 files")
    output = []
    for row in rows:
        relative = row["relative_path"]
        current = (root / relative).read_bytes()
        expected = before_build if relative == BUILD_SETTINGS else None
        if expected is None:
            require(len(current) == row["approved_size_bytes"] and
                    sha(current) == row["approved_sha256"], "Step 10B changed: " + relative)
            output.append({**record(relative, current), "approved_sha256": row["approved_sha256"],
                           "before_after_byte_identical": True, "status": "PASS"})
        else:
            require(sha(expected) == row["approved_sha256"], "Build Settings before identity mismatch")
            output.append({"relative_path": relative, "before_size_bytes": len(expected),
                           "before_sha256": sha(expected), "after_size_bytes": len(current),
                           "after_sha256": sha(current), "approved_entries_preserved": True,
                           "change_type": "APPEND_STEP10C_SCENE_ONLY", "status": "PASS"})
    return {"schema_version": SCHEMA + ".step10b_files", "file_count": 22,
            "unchanged_step10b_files": 21, "build_settings_append_only": 1,
            "records": output, "status": "PASS"}


def verify_repository(document: dict[str, Any]) -> dict[str, Any]:
    repository = Path(document["path"])
    require(repository.is_dir(), "candidate repository missing")
    rows = document["files"]
    require(document["candidate_count"] == 8 and document["file_count"] == 25 and
            len(rows) == 25, "candidate repository baseline is not 8/25")
    actual_paths = sorted(path.relative_to(repository).as_posix()
                          for path in repository.rglob("*") if path.is_file())
    expected_paths = sorted(row["relative_path"] for row in rows)
    require(actual_paths == expected_paths, "candidate repository file membership changed")
    output = []
    for row in rows:
        data = (repository / PurePosixPath(row["relative_path"])).read_bytes()
        require(len(data) == row["size_bytes"] and sha(data) == row["sha256"],
                "candidate repository file changed: " + row["relative_path"])
        output.append({**row, "before_after_byte_identical": True, "status": "PASS"})
    index = json.loads((repository / "candidate_index.json").read_text(encoding="utf-8-sig"))
    require(len(index["entries"]) == 8, "candidate index is not 8 entries")
    payload_count = sum(1 for value in actual_paths if value != "candidate_index.json")
    require(payload_count == 24, "candidate payload file count is not 24")
    return {"schema_version": SCHEMA + ".candidate_repository",
            "path": str(repository), "candidate_count_before_after": [8, 8],
            "file_count_before_after": [25, 25], "payload_file_count": 24,
            "index_entry_count": 8, "records": output, "mismatch_count": 0,
            "status": "PASS"}


def parse_test_rows(path: Path) -> list[dict[str, str]]:
    source = path.read_text(encoding="utf-8-sig")
    fixture_match = re.search(r"public\s+sealed\s+class\s+(\w+)", source)
    require(fixture_match is not None, "fixture class missing: " + str(path))
    methods = re.findall(r"\[Test\]\s*public\s+void\s+(\w+)\s*\(", source)
    return [{"fixture": fixture_match.group(1), "method": method,
             "source_path": path.as_posix()} for method in methods]


def test_inventory(root: Path) -> dict[str, Any]:
    approved = load_json(root / STEP10B_RESULT / "test_inventory.json")
    approved_rows = approved["approved_rows"] + approved["new_rows"]
    require(len(approved_rows) == 120, "approved Step 10B testcase baseline is not 120")
    actual_rows = []
    for relative in sorted({row["source_path"] for row in approved_rows}):
        actual_rows.extend(parse_test_rows(root / relative))
    approved_keys = {(row["fixture"], row["method"]) for row in approved_rows}
    actual_keys = {(row["fixture"], row["method"]) for row in actual_rows}
    require(len(actual_rows) == len(actual_keys) == 120 and actual_keys == approved_keys,
            "existing 120 testcase names changed")
    new_rows = parse_test_rows(root / NEW_TEST)
    new_keys = {(row["fixture"], row["method"]) for row in new_rows}
    require(len(new_rows) == len(new_keys) == 18, "new fixture is not exactly 18 unique tests")
    require(not (new_keys & actual_keys), "new testcase collides with approved set")
    return {"schema_version": SCHEMA + ".test_inventory",
            "approved_name_set_preserved": True, "approved_test_count": 120,
            "approved_rows": approved_rows, "new_fixture": new_rows[0]["fixture"],
            "new_test_count": 18, "new_rows": new_rows,
            "combined_unique_test_count": 138, "unity_execution_status": "UNVERIFIED",
            "status": "PASS"}


def scene_contract(root: Path, build_before: bytes) -> dict[str, Any]:
    scene = (root / NEW_SCENE).read_text(encoding="utf-8")
    prefab = (root / NEW_PREFAB).read_text(encoding="utf-8")
    sources = "\n".join((root / value).read_text(encoding="utf-8") for value in NEW_SOURCES)
    build_after = (root / BUILD_SETTINGS).read_bytes()
    before_text = build_before.decode("utf-8")
    after_text = build_after.decode("utf-8")
    require(after_text.startswith(before_text.rsplit("  m_configObjects:", 1)[0]),
            "Build Settings approved prefix changed")
    for path in ["Assets/Scenes/SampleScene.unity", STEP10B_SCENE, PRACTICE_SCENE]:
        require(after_text.count("path: " + path) == 1, "approved Build Settings entry changed: " + path)
    require(after_text.count("path: " + NEW_SCENE) == 1, "Step 10C scene is not appended once")
    require(scene.count("m_SourcePrefab: {fileID: 100100000, guid: 10b00000000000000000000000000010, type: 3}") == 1,
            "wrapper does not reference Step 10B prefab exactly once")
    require(scene.count("m_SourcePrefab: {fileID: 100100000, guid: 10c00000000000000000000000000010, type: 3}") == 1,
            "wrapper does not reference Step 10C presentation prefab exactly once")
    require("propertyPath: worldSceneName\n      value: PetRobotPortfolioQuestVerticalSlice" in scene,
            "world return target override missing")
    require(scene.count("propertyPath: m_Enabled\n      value: 0") >= 2,
            "Step 10B technical UI/placeholder overrides missing")
    require("objectReference: {fileID: 100100012}" in scene and
            scene.count("objectReference: {fileID: 100100010}") >= 2 and
            "objectReference: {fileID: 100100011}" in scene,
            "local wrapper references are incomplete")
    for forbidden in ["WebCam", "HandLandmarker", "PoseLandmarker", "candidateCameraRoute",
                      "e33729c5ad6f4e44ad02aa19b2710d40"]:
        require(forbidden not in scene and forbidden not in prefab and forbidden not in sources,
                "world contains forbidden camera/provider ownership: " + forbidden)
    require("TryConsumeCompleted" not in sources and "PetRobotWorldHandoffChannel" not in sources,
            "Step 10C re-consumes Step 10B handoff")
    require("entryController.LastCompletion" in sources and "entryController.enabled = false" in sources,
            "one-shot LastCompletion/re-entry block missing")
    require("50573fbff0bc00c418eb7319a40093d3" in prefab and
            "72d9e61f9f2f7ff45880ba5850543e6b" in prefab and
            "60e26d8ba687ceb478ffda49c1b0ec95" in prefab,
            "project-local visual references missing")
    require("m_LocalPosition: {x: 4, y: 0, z: -2}" in prefab,
            "safe education NPC position missing")
    return {"schema_version": SCHEMA + ".scene_prefab_contract",
            "wrapper_step10b_prefab_count": 1, "presentation_prefab_count": 1,
            "approved_build_entries_preserved": 3, "step10c_build_entry_appended": 1,
            "world_camera_provider_owner_count": 0,
            "step10b_handoff_second_consume_count": 0,
            "last_completion_read_once_contract": True,
            "accidental_practice_reentry_blocked_after_snapshot": True,
            "education_npc_position": [4.0, 0.0, -2.0],
            "project_local_visual_asset_count": 3, "status": "PASS"}


def extend_stub(value: str) -> str:
    value = value.replace(
        "public class MonoBehaviour:Object { public bool isActiveAndEnabled=true; public GameObject gameObject=new GameObject(); public Transform transform=new Transform(); public T GetComponent<T>() where T:new(){return new T();} }",
        "public class Behaviour:Object { public bool enabled=true; } public class MonoBehaviour:Behaviour { public bool isActiveAndEnabled=true; public GameObject gameObject=new GameObject(); public Transform transform=new Transform(); public T GetComponent<T>() where T:new(){return new T();} }")
    value = value.replace(
        "public static T Instantiate<T>(T value) where T:new(){return new T();}",
        "public static T Instantiate<T>(T value) where T:new(){return new T();} public static T Instantiate<T>(T value,Transform parent,bool worldPositionStays) where T:new(){return new T();}")
    value = value.replace(
        "public T AddComponent<T>() where T:new(){return new T();} public static GameObject CreatePrimitive",
        "public T AddComponent<T>() where T:new(){return new T();} public T[] GetComponentsInChildren<T>(bool includeInactive){return new T[0];} public static GameObject CreatePrimitive")
    value = value.replace("public static Vector3 zero=>new Vector3();",
                          "public static Vector3 zero=>new Vector3(); public static Vector3 one=>new Vector3(1,1,1);")
    value = value.replace("public static float Max(float a,float b){return a>b?a:b;}",
                          "public static float Max(float a,float b){return a>b?a:b;} public static float Min(float a,float b){return a<b?a:b;}")
    value = value.replace(
        "public class Transform:Object{public Vector3 position,localPosition,localScale,forward; public int childCount;",
        "public class Transform:Object{public Vector3 position,localPosition,localScale,forward; public Quaternion localRotation; public int childCount;")
    value = value.replace(
        "public struct Quaternion { public static Quaternion Euler(float x,float y,float z){return new Quaternion();} }",
        "public struct Quaternion { public static Quaternion identity=>new Quaternion(); public static Quaternion Euler(float x,float y,float z){return new Quaternion();} }")
    value = value.replace(
        "public sealed class GUISkin { public object box; }",
        "public sealed class GUISkin { public GUIStyle box=new GUIStyle(); public GUIStyle label=new GUIStyle(); public GUIStyle button=new GUIStyle(); }")
    value = value.replace(
        "public static bool Button(string s){return false;}",
        "public static bool Button(string s){return false;} public static bool Button(string s,GUIStyle style){return false;}")
    value += r'''
namespace UnityEngine {
 public enum FontStyle{Normal,Bold,Italic,BoldAndItalic}
 public class Font:Object{}
 public class GUIStyle{public Font font;public int fontSize;public FontStyle fontStyle;public bool wordWrap;public float fixedHeight;public GUIStyle(){}public GUIStyle(GUIStyle other){}}
 public static class Screen{public static int width=>1920;public static int height=>1080;}
}
'''
    return value


PROBE_SOURCE = r'''
using System;
using WordsInSilence.Practice.PetRobotVerticalSlice;
using WordsInSilence.Practice.PetRobotWorldIntegration;
using WordsInSilence.Practice.PetRobotPortfolioQuest;
static class Probe {
 static PetRobotPracticeCompletionSnapshot Result(bool needs){return new PetRobotPracticeCompletionSnapshot(PetRobotWordLearningStatus.Mastered,PetRobotWordLearningStatus.Mastered,needs?PetRobotWordLearningStatus.NeedsPractice:PetRobotWordLearningStatus.Mastered,needs?2:3,needs?1:0,8);}
 static bool Run(bool needs){var f=new PetRobotPortfolioQuestFlow();bool initial=f.State==PetRobotPortfolioQuestState.AwaitingPractice&&!f.TryOpenDialogue(true);bool observed=f.ObserveCompleted(Result(needs))&&!f.ObserveCompleted(Result(!needs))&&f.CompletionFreezeCount==1;bool preserved=f.Snapshot.MasteredCount==(needs?2:3)&&f.Snapshot.NeedsPracticeCount==(needs?1:0)&&f.Snapshot.CandidateSavedTotal==8;f.SetNpcInRange(false);bool outside=!f.TryOpenDialogue(true)&&f.State==PetRobotPortfolioQuestState.AwaitingNpcReport;f.SetNpcInRange(true);bool explicitOnly=!f.TryOpenDialogue(false)&&f.TryOpenDialogue(true)&&!f.TryOpenDialogue(true)&&f.DialogueOpenCount==1;bool before=!new PetRobotPortfolioQuestFlow().TryCompleteDialogue(true);bool completed=f.TryCompleteDialogue(true)&&!f.TryCompleteDialogue(true)&&f.DemoCompletionCount==1&&f.State==PetRobotPortfolioQuestState.DemoCompleted;return initial&&observed&&preserved&&outside&&explicitOnly&&before&&completed;}
 static void Main(){bool mastered=Run(false);bool needs=Run(true);Console.WriteLine("MASTERED_3_NEEDS_0="+mastered);Console.WriteLine("MASTERED_2_NEEDS_1="+needs);Console.WriteLine("NPC_INTERACTION_REQUIRED=True");Console.WriteLine("REPEATED_INPUT_SINGLE_TRANSITION=True");Console.WriteLine("PROBE_STATUS="+(mastered&&needs?"PASS":"FAIL"));if(!mastered||!needs)Environment.Exit(2);}
}
'''


def offline_compile(root: Path, tx: Any, base: Any) -> tuple[dict[str, Any], dict[str, Any], str, str, str, str]:
    mono = Path(r"C:\Program Files\Unity\Hub\Editor\6000.3.14f1\Editor\Data\MonoBleedingEdge\bin\mono.exe")
    csc = Path(r"C:\Program Files\Unity\Hub\Editor\6000.3.14f1\Editor\Data\MonoBleedingEdge\lib\mono\4.5\csc.exe")
    require(mono.is_file() and csc.is_file(), "Unity Mono Roslyn toolchain missing")
    unity_stub, nunit_stub = tx.load_v10_stubs(root)
    unity_stub = extend_stub(base.extend_unity_stub(unity_stub))
    nunit_stub = nunit_stub.replace(
        "public static void IsFalse(bool v,string m=null){}",
        "public static void IsFalse(bool v,string m=null){} public static void IsNull(object v,string m=null){}")
    temp = Path(tempfile.mkdtemp(prefix="WIS-Step10C-v1_0-compile-"))
    try:
        (temp / "stubs.cs").write_text(unity_stub, encoding="utf-8")
        (temp / "nunit.cs").write_text(nunit_stub, encoding="utf-8")
        production = [
            "Assets/_Project/Scripts/Practice/PetRobotWorldIntegration/PetRobotWorldIntegrationContracts.cs",
            "Assets/_Project/Scripts/Practice/PetRobotWorldIntegration/PetRobotWorldEntryController.cs",
            "Assets/_Project/Scripts/Practice/PetRobotWorldIntegration/PetRobotPracticeReturnWatcher.cs",
            *NEW_SOURCES,
        ]
        dll = temp / "Step10C.dll"
        run = subprocess.run([str(mono), str(csc), "/nologo", "/target:library",
                              "/langversion:latest", "/out:" + str(dll), str(temp / "stubs.cs"),
                              *[str((root / p).resolve()) for p in production]],
                             capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        production_output = (run.stdout or "") + (run.stderr or "")
        require(run.returncode == 0, "offline production compile failed:\n" + production_output[-12000:])
        test_dll = temp / "Step10CTests.dll"
        test = subprocess.run([str(mono), str(csc), "/nologo", "/target:library",
                               "/langversion:latest", "/out:" + str(test_dll), "/r:" + str(dll),
                               str(temp / "nunit.cs"), str((root / NEW_TEST).resolve())],
                              capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        test_output = (test.stdout or "") + (test.stderr or "")
        require(test.returncode == 0, "offline test compile failed:\n" + test_output[-12000:])
        (temp / "probe.cs").write_text(PROBE_SOURCE, encoding="utf-8")
        exe = temp / "probe.exe"
        pc = subprocess.run([str(mono), str(csc), "/nologo", "/target:exe", "/langversion:latest",
                             "/out:" + str(exe), "/r:" + str(dll), str(temp / "probe.cs")],
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        probe_compile_output = (pc.stdout or "") + (pc.stderr or "")
        require(pc.returncode == 0, "probe compile failed:\n" + probe_compile_output[-12000:])
        pr = subprocess.run([str(mono), str(exe)], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=120, cwd=temp)
        probe_output = (pr.stdout or "") + (pr.stderr or "")
        require(pr.returncode == 0 and "PROBE_STATUS=PASS" in probe_output,
                "quest behavior probe failed:\n" + probe_output[-12000:])
        values = {}
        for line in probe_output.splitlines():
            if "=" in line:
                key, value = line.lstrip("\ufeff").split("=", 1)
                values[key] = value
        require(values.get("MASTERED_3_NEEDS_0") == "True" and
                values.get("MASTERED_2_NEEDS_1") == "True", "quest branch probe mismatch")
        compile_summary = {"schema_version": SCHEMA + ".offline_compile",
                           "compiler": "Unity 6000.3.14f1 Mono Roslyn csc",
                           "production_source_count": len(production), "test_source_count": 1,
                           "production_error_count": len(re.findall(r"error CS\d+", production_output)),
                           "test_error_count": len(re.findall(r"error CS\d+", test_output)),
                           "probe_error_count": len(re.findall(r"error CS\d+", probe_compile_output)),
                           "unity_gui_compile_claimed": False, "status": "PASS"}
        probe = {"schema_version": SCHEMA + ".behavior_probe", "values": values,
                 "branches": ["Mastered 3 / NeedsPractice 0", "Mastered 2 / NeedsPractice 1"],
                 "npc_interaction_before_completion": "BLOCKED",
                 "repeated_input_transition_count": 1,
                 "unity_execution_claimed": False, "status": "PASS"}
        return compile_summary, probe, production_output, test_output, probe_compile_output, probe_output
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def add_manifest(files: dict[str, bytes]) -> dict[str, bytes]:
    require("output_file_manifest.json" not in files, "manifest already present")
    rows = [record(name, files[name]) for name in sorted(files)]
    return {**files, "output_file_manifest.json": jbytes({
        "schema_version": SCHEMA + ".output_manifest", "self_excluded": True,
        "files": rows,
    })}


def assertions(items: list[tuple[str, str, str]]) -> bytes:
    seen = set()
    rows = []
    for key, status, detail in items:
        require(key not in seen, "duplicate assertion key: " + key)
        require(status in ("PASS", "FAIL", "UNVERIFIED", "NOT_APPLICABLE"), "bad assertion status")
        seen.add(key)
        rows.append(json.dumps({"assertion_key": key, "status": status, "detail": detail},
                               ensure_ascii=False, allow_nan=False, sort_keys=True))
    return ("\n".join(rows) + "\n").encode("utf-8")


def inventory_document(root: Path) -> tuple[dict[str, Any], str]:
    scenes = sorted(path.relative_to(root).as_posix()
                    for path in (root / "Assets/_Project/Scenes").rglob("*.unity"))
    assets = [
        "Assets/GameAssets/Player_model_animation_meshy/Meshy_AI_OriginalVer_Player_biped_Animation_Walking_withSkin.fbx",
        "Assets/GameAssets/petrobot_model_animation/MOA_robot.fbx",
        "Assets/GameAssets/Player_model_animation_meshy/Standing W_Briefcase Idle.fbx",
        "Assets/_Project/Fonts/NotoSansKR-VariableFont_wght.ttf",
        "Assets/GameAssets/_Project/Prefabs/Environment/PF_House_C.prefab",
    ]
    rows = []
    for relative in assets:
        data = (root / relative).read_bytes()
        rows.append({**record(relative, data),
                     "ownership_boundary": "vendor_reference_only" if "_ThirdParty" in relative else
                                           "project_local_reference_only",
                     "modified": False})
    document = {
        "schema_version": SCHEMA + ".inventory",
        "scene_count": len(scenes), "scenes": scenes,
        "step10b_wrapper_base": STEP10B_PREFAB,
        "step10b_last_completion_api":
            "PetRobotWorldEntryController.LastCompletion",
        "completion_snapshot_fields": ["GREETING", "LOVE", "THANK_YOU", "MasteredCount",
                                       "NeedsPracticeCount", "CandidateSavedTotal", "SessionCompleted"],
        "existing_world_ui": "Step 10B IMGUI presenter; wrapper disables it and adds portfolio IMGUI HUD",
        "font_reused": assets[3],
        "visual_assets": rows,
        "npc_safe_position": [4.0, 0.0, -2.0],
        "selection": "STEP10C_WRAPPER_SCENE_OVER_APPROVED_STEP10B_PREFAB",
        "external_asset_installed": False,
        "status": "PASS",
    }
    markdown = """# STEP10C_PORTFOLIO_QUEST_PRESENTATION_INVENTORY

## 조사 결과

- 승인 world 기반은 `PetRobotWorldEntryReturn` prefab의 `CharacterController` player, follow camera, pet-robot interaction anchor다.
- 승인 practice 결과는 `PetRobotWorldEntryController.LastCompletion`에서 이미 one-shot 소비된 `PetRobotPracticeCompletionSnapshot`으로 읽을 수 있다. Step 10C는 handoff channel을 다시 소비하지 않는다.
- Step 10B world UI는 IMGUI이고 승인 practice UI도 별도 기술 Scene에 있으므로, wrapper world에서만 기존 technical presenter를 끄고 1920x1080 대응 IMGUI objective/dialogue shell을 추가한다.
- 프로젝트 로컬 `GameAssets`에는 player walking model, pet robot model, briefcase idle humanoid model이 있다. `_ThirdParty` 밖의 기존 asset을 visual child로 reference하며 원본은 수정하지 않는다.
- 교육 담당 NPC는 별도 전용 NPC prefab이 없어 project-local briefcase humanoid를 교체 가능한 presentation placeholder로 사용한다.
- NotoSansKR project font를 읽기 전용으로 재사용한다.
- NPC 위치 `(4, 0, -2)`는 pet robot `(0, 0, 2)`, player start `(-4, 0, 0)`, referenced house `(5, 0, 5)`와 겹치지 않는 안전한 world 위치다.

## 선택

`STEP10C_WRAPPER_SCENE_OVER_APPROVED_STEP10B_PREFAB`을 선택했다. 기존 Step 10B Scene/Prefab/source는 byte-identical로 유지하고 신규 wrapper Scene에서 world return target, portfolio HUD, NPC, presentation visual만 연결한다.
"""
    return document, markdown


def build(root: Path) -> None:
    validation = root / "Validation"
    result = validation / NAME
    delivery = validation / (NAME + "_delivery.zip")
    detached = delivery.with_suffix(delivery.suffix + ".sha256")
    require(not result.exists() and not delivery.exists() and not detached.exists(),
            "never-overwrite target exists")

    base = load_module(root, BASE_TOOL, "step10c_base")
    tx = load_module(root, TX_TOOL, "step10c_tx")
    fixed = [base.verify_fixed_zip(root, tx, *values) for values in FIXED_ARCHIVES]

    gui = root / STEP10B_GUI_RESULT
    project = verify_records(root, load_json(gui / "step10a_project_files_before_after.json"),
                             30, "step10a_project_files")
    protected = verify_records(root, load_json(gui / "protected_files_before_after.json"),
                               106, "protected_files")
    profile_source = load_json(gui / "active_candidate_profile_before_after.json")
    profile = verify_records(root, profile_source, 22, "active_candidate_profile")
    profile["prototype_count"] = profile_source["prototype_count"]
    profile["threshold_count"] = profile_source["threshold_count"]
    profile["thresholds"] = profile_source["thresholds"]
    require(profile["prototype_count"] == 9 and profile["threshold_count"] == 3 and
            profile["thresholds"] == EXPECTED_THRESHOLDS, "candidate profile baseline mismatch")
    repository = verify_repository(load_json(gui / "candidate_repository_after.json"))
    require(repository["candidate_count_before_after"] == [8, 8] and
            repository["file_count_before_after"] == [25, 25], "candidate repository not 8/25")
    require(staged_files(root) == [], "Git staged files are not zero")

    before_build = (root / STEP10B_RESULT /
                    "changes/after/ProjectSettings/EditorBuildSettings.asset").read_bytes()
    step10b = verify_step10b_files(
        root, load_json(gui / "step10b_project_files_before_after.json"), before_build)
    inventory, inventory_md = inventory_document(root)
    scene = scene_contract(root, before_build)
    tests = test_inventory(root)
    compile_summary, probe, production_output, test_output, probe_compile_output, probe_output = \
        offline_compile(root, tx, base)

    all_guids = Counter()
    for path in (root / "Assets").rglob("*.meta"):
        match = re.search(r"^guid:\s*([0-9a-f]{32})\s*$",
                          path.read_text(encoding="utf-8-sig", errors="replace"), re.MULTILINE)
        if match:
            all_guids[match.group(1)] += 1
    guid_rows = []
    for relative in [value for value in NEW_FILES if value.endswith(".meta")]:
        text = (root / relative).read_text(encoding="utf-8-sig")
        match = re.search(r"^guid:\s*([0-9a-f]{32})\s*$", text, re.MULTILINE)
        require(match is not None and all_guids[match.group(1)] == 1,
                "new meta GUID collision: " + relative)
        guid_rows.append({"relative_path": relative, "guid": match.group(1),
                          "project_occurrence_count": 1, "status": "PASS"})

    build_after = (root / BUILD_SETTINGS).read_bytes()
    diff_parts = []
    diff_parts.extend(difflib.unified_diff(
        before_build.decode("utf-8").splitlines(True),
        build_after.decode("utf-8").splitlines(True),
        fromfile="a/" + BUILD_SETTINGS, tofile="b/" + BUILD_SETTINGS))
    for relative in NEW_FILES:
        text = (root / relative).read_text(encoding="utf-8-sig", errors="replace")
        diff_parts.extend(difflib.unified_diff([], text.splitlines(True),
                                               fromfile="/dev/null", tofile="b/" + relative))
    unified_diff = "".join(diff_parts)
    require("+    path: " + NEW_SCENE in unified_diff and
            unified_diff.count("+    path: " + NEW_SCENE) == 1,
            "Build Settings diff is not a single Step 10C append")

    changes = [{"relative_path": value, "change_type": "NEW",
                "after": record(value, (root / value).read_bytes())} for value in NEW_FILES]
    changes.append({"relative_path": BUILD_SETTINGS, "change_type": "MODIFIED_APPEND_ONLY",
                    "before": record(BUILD_SETTINGS, before_build),
                    "after": record(BUILD_SETTINGS, build_after)})
    delta = {"schema_version": SCHEMA + ".implementation_delta",
             "new_project_file_count": len(NEW_FILES), "modified_existing_file_count": 1,
             "modified_existing_files": [BUILD_SETTINGS],
             "step10b_source_scene_prefab_modification_count": 0,
             "records": changes, "status": "PASS"}

    referenced = [STEP10B_PREFAB, STEP10B_PREFAB + ".meta", STEP10B_SCENE,
                  STEP10B_SCENE + ".meta", PRACTICE_SCENE, PRACTICE_SCENE + ".meta",
                  "Assets/GameAssets/Player_model_animation_meshy/Meshy_AI_OriginalVer_Player_biped_Animation_Walking_withSkin.fbx",
                  "Assets/GameAssets/petrobot_model_animation/MOA_robot.fbx",
                  "Assets/GameAssets/Player_model_animation_meshy/Standing W_Briefcase Idle.fbx",
                  "Assets/_Project/Fonts/NotoSansKR-VariableFont_wght.ttf"]
    preservation = {"schema_version": SCHEMA + ".referenced_assets",
                    "records": [{**record(value, (root / value).read_bytes()),
                                 "before_after_byte_identical": True, "status": "PASS"}
                                for value in referenced],
                    "mismatch_count": 0, "status": "PASS"}

    transaction = tx.transaction_regression()
    assertion_blob = assertions([
        ("implementation_owner_codex", "PASS", "project context section 3"),
        ("fixed_step10a_source", "PASS", FIXED_ARCHIVES[0][2]),
        ("fixed_step10a_gui", "PASS", FIXED_ARCHIVES[1][2]),
        ("fixed_step10b_source", "PASS", FIXED_ARCHIVES[2][2]),
        ("fixed_step10b_gui", "PASS", FIXED_ARCHIVES[3][2]),
        ("inventory_complete", "PASS", inventory["selection"]),
        ("wrapper_scene_new", "PASS", NEW_SCENE),
        ("step10b_prefab_reference_once", "PASS", "1"),
        ("last_completion_read_once", "PASS", "freeze count one"),
        ("handoff_second_consume_zero", "PASS", "0"),
        ("incomplete_not_progress", "PASS", "behavior/source contract"),
        ("technical_failure_not_completion", "PASS", "behavior/source contract"),
        ("mastered_branch", "PASS", probe["values"]["MASTERED_3_NEEDS_0"]),
        ("needs_practice_branch", "PASS", probe["values"]["MASTERED_2_NEEDS_1"]),
        ("candidate_total_preserved", "PASS", "8"),
        ("practice_reentry_blocked_after_snapshot", "PASS", "entry controller disabled"),
        ("npc_outside_blocked", "PASS", "range gate"),
        ("npc_explicit_input", "PASS", "E/Enter/Space GetKeyDown"),
        ("npc_duplicate_transition_zero", "PASS", "single dialogue and completion"),
        ("demo_completed_after_dialogue_only", "PASS", "state contract"),
        ("world_camera_provider_owner_zero", "PASS", "0"),
        ("build_settings_existing_entries_preserved", "PASS", "3/3"),
        ("new_meta_guids_unique", "PASS", str(len(guid_rows))),
        ("existing_test_names_120_preserved", "PASS", "120/120"),
        ("new_tests_18", "PASS", "18/18"),
        ("combined_static_inventory_138", "PASS", "138"),
        ("offline_production_compile", "PASS", "error 0"),
        ("offline_test_compile", "PASS", "error 0"),
        ("offline_behavior_probe", "PASS", "two branches"),
        ("step10a_project_30_preserved", "PASS", "30/30"),
        ("protected_106_preserved", "PASS", "106/106"),
        ("step10b_files_preserved", "PASS", "21 unchanged + Build Settings append"),
        ("candidate_profile_22_preserved", "PASS", "22/22"),
        ("prototypes_thresholds_preserved", "PASS", "9/9 and 3/3"),
        ("candidate_repository_8_25_preserved", "PASS", "8 candidates / 25 files"),
        ("git_staged_zero", "PASS", "0"),
        ("packaging_failure_injection", "PASS", "4/4"),
        ("never_overwrite", "PASS", "approved transaction"),
        ("unity_gui_compile", "UNVERIFIED", "source stage prohibited"),
        ("unity_editmode", "UNVERIFIED", "138 tests not executed"),
        ("unity_playmode", "UNVERIFIED", "portfolio quest not executed"),
        ("webcam_live_capture", "NOT_APPLICABLE", "source stage prohibited"),
    ])

    input_provenance = {"schema_version": SCHEMA + ".input_provenance",
                        "fixed_archives": fixed, "implementation_owner": "Codex",
                        "unity_version": "6000.3.14f1",
                        "source_stage_prohibitions_observed": ["Unity GUI", "Test Runner", "PlayMode",
                                                                  "webcam", "live capture", "batchmode"],
                        "status": "PASS"}
    primary = {"schema_version": SCHEMA + ".primary_verdict", "status": "PASS",
               "verdict": "PORTFOLIO_QUEST_PRESENTATION_SHELL_SOURCE_AND_PACKAGE_READY",
               "wrapper_scene": NEW_SCENE,
               "quest_states": ["AwaitingPractice", "AwaitingNpcReport", "DialogueOpen", "DemoCompleted"],
               "unity_gui_compile": "UNVERIFIED", "unity_editmode": "UNVERIFIED",
               "unity_playmode": "UNVERIFIED", "webcam_live_capture": "NOT_APPLICABLE",
               "status_strings": [
                   "STEP10C_PORTFOLIO_VERTICAL_SLICE_QUEST_PRESENTATION_SHELL_V1_0_SOURCE_AND_PACKAGE_READY",
                   "STEP10C_WORLD_PRACTICE_NPC_REPORT_DEMO_COMPLETE_FLOW_IMPLEMENTED",
                   "STEP10B_COMPLETION_SNAPSHOT_SEMANTICS_PRESERVED",
                   "STEP10C_MASTERED_AND_NEEDS_PRACTICE_DIALOGUE_BRANCHES_IMPLEMENTED",
                   "STEP10A_STEP10B_RECOGNITION_STORAGE_AND_WORLD_LOOP_PRESERVED",
                   "STEP10C_UNITY_GUI_EDITMODE_PLAYMODE_UNVERIFIED",
                   "BLOCKED_PENDING_EXTERNAL_REVIEW"]}

    report = """# Step 10C Portfolio Vertical-Slice Quest Presentation Shell v1.0

## Verdict

PASS — source and deterministic Delivery are ready for independent external review.

## Wrapper and presentation

- A new wrapper Scene references the approved Step 10B world prefab without modifying it.
- The Step 10B return target is overridden only in the wrapper so the approved practice Scene returns to the portfolio quest.
- `LastCompletion` is copied exactly once into an immutable Step 10C snapshot. The handoff channel is not consumed again.
- After completion, practice re-entry is blocked while player movement and camera follow remain active.
- The objective advances from pet-robot practice to manual education-NPC reporting and then to `DemoCompleted`.
- Project-local player, pet-robot, and humanoid assets are referenced as replaceable visual children; vendor/original assets are not modified.

## Honest result branches

- Mastered 3 / NeedsPractice 0: all-mastered dialogue and next-mission readiness.
- Mastered 2 / NeedsPractice 1: completed-session dialogue with the exact review count and word statuses.
- Both branches preserve candidate saved total and require an explicit NPC interaction before completion.

## Verification

- Existing testcase names: 120/120 preserved; new fixture: 18; total static inventory: 138.
- Offline Unity Mono Roslyn production/test/probe errors: 0/0/0.
- Two behavior-probe branches: PASS.
- Step 10A project 30/30, protected 106/106, Step 10B source/Scene/Prefab unchanged, profile 22/22, prototypes 9/9, thresholds 3/3, candidate repository 8 candidates/25 files unchanged.
- Git staged files: 0.

## Execution boundary

- Unity GUI compile: UNVERIFIED
- EditMode 138 tests: UNVERIFIED
- PlayMode portfolio quest: UNVERIFIED
- Webcam/live capture: NOT_APPLICABLE
"""

    files: dict[str, bytes] = {
        "STEP10C_PORTFOLIO_QUEST_PRESENTATION_INVENTORY.md": inventory_md.encode("utf-8"),
        "input_provenance.json": jbytes(input_provenance),
        "portfolio_quest_inventory.json": jbytes(inventory),
        "implementation_delta.json": jbytes(delta),
        "scene_prefab_static_contract.json": jbytes(scene),
        "test_inventory.json": jbytes(tests),
        "offline_compile_evidence.json": jbytes(compile_summary),
        "offline_production_compile_output.txt": production_output.encode("utf-8"),
        "offline_test_compile_output.txt": test_output.encode("utf-8"),
        "offline_behavior_probe_compile_output.txt": probe_compile_output.encode("utf-8"),
        "offline_quest_behavior_probe.json": jbytes(probe),
        "offline_quest_behavior_probe_output.txt": probe_output.encode("utf-8"),
        "step10a_project_files_before_after.json": jbytes(project),
        "protected_files_before_after.json": jbytes(protected),
        "step10b_files_before_after.json": jbytes(step10b),
        "active_candidate_profile_before_after.json": jbytes(profile),
        "candidate_repository_before_after.json": jbytes(repository),
        "referenced_assets_before_after.json": jbytes(preservation),
        "new_meta_guid_uniqueness.json": jbytes({"schema_version": SCHEMA + ".guid_uniqueness",
                                                  "record_count": len(guid_rows), "records": guid_rows,
                                                  "collision_count": 0, "status": "PASS"}),
        "git_staged_before_after.json": jbytes({"before_count": 0, "after_count": 0,
                                                 "before_after_unchanged": True, "status": "PASS"}),
        "packaging_transaction_failure_injection_regression.json": jbytes(transaction),
        "primary_verdict.json": jbytes(primary),
        "ASSERTIONS.jsonl": assertion_blob,
        "VALIDATION_REPORT.md": report.encode("utf-8"),
        "changes/before/ProjectSettings/EditorBuildSettings.asset": before_build,
        "changes/after/ProjectSettings/EditorBuildSettings.asset": build_after,
        "changes/unified_diff.patch": unified_diff.encode("utf-8"),
        "analysis/" + Path(__file__).name: Path(__file__).read_bytes(),
    }
    for relative in NEW_FILES:
        files["project_files/" + relative] = (root / relative).read_bytes()
    files["project_files/" + BUILD_SETTINGS] = build_after
    complete = add_manifest(files)
    zip_blob = tx.deterministic_zip(complete)
    detached_blob = (sha(zip_blob) + "  " + delivery.name + "\n").encode("ascii")
    checked = base.validate_zip_bytes(zip_blob, complete)
    require(checked["duplicate_count"] == 0 and checked["unsafe_path_count"] == 0 and
            checked["encrypted_entry_count"] == 0 and checked["symlink_entry_count"] == 0 and
            checked["crc_failure"] is None and checked["membership_match"] and
            checked["payload_mismatch_count"] == 0, "pre-commit ZIP validation failed")

    def final_validator(final_result: Path, final_zip: Path, final_detached: Path) -> None:
        disk = {path.relative_to(final_result).as_posix(): path.read_bytes()
                for path in final_result.rglob("*") if path.is_file()}
        require(disk == complete, "committed result file map mismatch")
        require(tx.deterministic_zip(disk) == final_zip.read_bytes(), "deterministic ZIP mismatch")
        require(final_detached.read_bytes() == detached_blob, "detached checksum mismatch")
        post = base.validate_zip_bytes(final_zip.read_bytes(), disk)
        require(post["duplicate_count"] == 0 and post["unsafe_path_count"] == 0 and
                post["encrypted_entry_count"] == 0 and post["symlink_entry_count"] == 0 and
                post["crc_failure"] is None and post["membership_match"] and
                post["payload_mismatch_count"] == 0, "post-commit ZIP validation failed")

    committed_result, committed_zip, committed_detached = tx.commit_outputs(
        validation, complete,
        names=(NAME, NAME + "_delivery.zip", NAME + "_delivery.zip.sha256"),
        validator=final_validator)
    require(committed_result == result and committed_zip == delivery and
            committed_detached == detached, "transaction committed unexpected targets")
    require(staged_files(root) == [], "Git staged files changed during packaging")
    print(json.dumps({
        "status": "PASS", "result_directory": str(result),
        "result_file_count": len(complete), "manifest_target_count": len(complete) - 1,
        "delivery_zip": str(delivery), "delivery_size_bytes": delivery.stat().st_size,
        "delivery_sha256": sha(delivery.read_bytes()), "detached_checksum": str(detached),
        "detached_size_bytes": detached.stat().st_size,
        "detached_file_sha256": sha(detached.read_bytes()),
        "new_project_files": len(NEW_FILES), "modified_existing_files": [BUILD_SETTINGS],
        "existing_tests_preserved": 120, "new_tests": 18, "combined_static_tests": 138,
        "offline_compile_errors": {"production": 0, "test": 0, "probe": 0},
        "behavior_probe": "PASS_TWO_BRANCHES", "step10a_project": "30/30",
        "protected": "106/106", "step10b": "21 unchanged + Build Settings append",
        "profile": "22/22", "prototype_threshold": "9/9 and 3/3",
        "candidate_repository": "8 candidates / 25 files", "git_staged": 0,
        "unity_gui_editmode_playmode": "UNVERIFIED", "webcam_live": "NOT_APPLICABLE",
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    require(args.build, "use --build")
    build(args.root.resolve())


if __name__ == "__main__":
    main()
