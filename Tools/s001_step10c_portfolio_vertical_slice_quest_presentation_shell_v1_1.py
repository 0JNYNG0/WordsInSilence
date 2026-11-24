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


SCHEMA = "WordsInSilence.Step10C.PortfolioVerticalSliceQuestPresentationShell.v1.1"
NAME = "WordsInSilence_step10c_portfolio_vertical_slice_quest_presentation_shell_v1_1"
V10_TOOL = "Tools/s001_step10c_portfolio_vertical_slice_quest_presentation_shell_v1_0.py"
BASE_TOOL = "Tools/s001_step10b_pet_robot_world_entry_return_integration_v1_0.py"
TX_TOOL = "Tools/s001_step10a_pet_robot_candidate_camera_startup_route_v1_1.py"
V10_ZIP = "Validation/WordsInSilence_step10c_portfolio_vertical_slice_quest_presentation_shell_v1_0_delivery.zip"
V10_ZIP_SIZE = 69845
V10_ZIP_SHA = "6792554f9bd8fc9b95c17c342db9b4003802e839b379238481404f63e7133697"
V10_DETACHED_SIZE = 157
V10_DETACHED_SHA = "6e91927f44d3c95b9ed3bb2a3cace9a917b0f9a7a75cd57481f11164e433731a"

CONTROLLER = "Assets/_Project/Scripts/Practice/PetRobotPortfolioQuest/PetRobotPortfolioQuestController.cs"
TEST = "Assets/_Project/Tests/EditMode/PetRobotPortfolioQuestPresentationV1Tests.cs"
PREFAB = "Assets/_Project/Resources/VerticalSlice/PetRobotPortfolioQuestPresentation.prefab"
SCENE = "Assets/_Project/Scenes/VerticalSlice/PetRobotPortfolioQuestVerticalSlice.unity"
BUILD = "ProjectSettings/EditorBuildSettings.asset"
V10_RESULT = "Validation/WordsInSilence_step10c_portfolio_vertical_slice_quest_presentation_shell_v1_0"
STEP10B_RESULT = "Validation/WordsInSilence_step10b_pet_robot_world_entry_return_integration_v1_0"
STEP10B_GUI = "Validation/WordsInSilence_step10b_pet_robot_world_entry_return_integration_v1_0_gui_unity_editor_gate"

CONTROLLER_BEFORE_SHA = "c3e1809c5b05fa7d5d27b7f656f0912a8e2319e8c6c8e46b987252d7fe73f541"
TEST_BEFORE_SHA = "85c8aeafcfa6af04e528b11191143dd8b81d371052a943520cf288affea3d614"
UNCHANGED_HASHES = {
    "Assets/_Project/Scripts/Practice/PetRobotPortfolioQuest/PetRobotPortfolioQuestContracts.cs":
        "f0006aaf88fe9f0cb3fd90845c38df624d4ec56027799ae57cfc08ccb0465167",
    "Assets/_Project/Scripts/Practice/PetRobotPortfolioQuest/PetRobotPortfolioQuestUiPresenter.cs":
        "b27d1fd9b380b4fa1294e315e291e3a033037d01d602338c33911c237ae3f93b",
    "Assets/_Project/Scripts/Practice/PetRobotPortfolioQuest/PetRobotPortfolioQuestVisuals.cs":
        "76d8df6b1a52354495d00cff26194c80e98e178a00cab706fd61fb1bb7465543",
    PREFAB: "7b62dbc69bc74fac3e785cad47f277c1afa6bca70d94be13c7a38c8eb3131101",
    SCENE: "7fb0fb9005f5648f394be5c8cbf54cfdc1f294229b42724f1838ce5e8a58d01f",
    BUILD: "584bc074ee61f0225ddfb52b8f9ef3a64825ea7fffd5815c8c8d5e96966c2820",
}

REFERENCE_ASSETS = [
    ("Assets/GameAssets/Player_model_animation_meshy/Meshy_AI_OriginalVer_Player_biped_Animation_Walking_withSkin.fbx",
     "72d9e61f9f2f7ff45880ba5850543e6b"),
    ("Assets/GameAssets/petrobot_model_animation/MOA_robot.fbx",
     "50573fbff0bc00c418eb7319a40093d3"),
    ("Assets/GameAssets/Player_model_animation_meshy/Standing W_Briefcase Idle.fbx",
     "60e26d8ba687ceb478ffda49c1b0ec95"),
    ("Assets/_Project/Fonts/NotoSansKR-VariableFont_wght.ttf",
     "aa36a16ad1c266948b24a0d14af31351"),
]

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


def load_module(root: Path, relative: str, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, root / relative)
    require(spec is not None and spec.loader is not None, "cannot load helper: " + relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def staged_files(root: Path) -> list[str]:
    run = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=root,
                         capture_output=True, text=True, encoding="utf-8",
                         errors="replace", timeout=30)
    require(run.returncode == 0, "Git staged query failed")
    return [line.strip() for line in run.stdout.splitlines() if line.strip()]


def archive_map(data: bytes) -> dict[str, bytes]:
    import io
    with zipfile.ZipFile(io.BytesIO(data), "r") as archive:
        return {info.filename: archive.read(info) for info in archive.infolist()}


def verify_v10(root: Path, base: Any, tx: Any) -> tuple[dict[str, Any], dict[str, bytes]]:
    path = root / V10_ZIP
    detached = Path(str(path) + ".sha256")
    data = path.read_bytes()
    detached_data = detached.read_bytes()
    require(len(data) == V10_ZIP_SIZE and sha(data) == V10_ZIP_SHA,
            "fixed v1.0 ZIP identity mismatch")
    require(len(detached_data) == V10_DETACHED_SIZE and sha(detached_data) == V10_DETACHED_SHA,
            "fixed v1.0 detached identity mismatch")
    require(detached_data == (V10_ZIP_SHA + "  " + path.name + "\n").encode("ascii"),
            "fixed v1.0 detached syntax mismatch")
    files = archive_map(data)
    require(len(files) == 45, "fixed v1.0 ZIP entry count mismatch")
    manifest = base.verify_archive_manifest(files)
    require(manifest["target_count"] == 44, "fixed v1.0 manifest target mismatch")
    safety = base.validate_zip_bytes(data, files)
    require(safety["duplicate_count"] == 0 and safety["unsafe_path_count"] == 0 and
            safety["encrypted_entry_count"] == 0 and safety["symlink_entry_count"] == 0 and
            safety["crc_failure"] is None and safety["membership_match"] and
            safety["payload_mismatch_count"] == 0, "fixed v1.0 archive safety failure")
    require(tx.deterministic_zip(files) == data, "fixed v1.0 deterministic recreation mismatch")
    return ({"relative_path": V10_ZIP, "size_bytes": len(data), "sha256": sha(data),
             "detached_size_bytes": len(detached_data),
             "detached_file_sha256": sha(detached_data), "entry_count": 45,
             "manifest_target_count": 44, "deterministic_byte_identical": True,
             "status": "PASS"}, files)


def preserve_v10_project(root: Path, v10_files: dict[str, bytes]) -> tuple[dict[str, Any], bytes, bytes]:
    prefix = "project_files/"
    rows = []
    before_controller = v10_files[prefix + CONTROLLER]
    before_test = v10_files[prefix + TEST]
    require(sha(before_controller) == CONTROLLER_BEFORE_SHA and
            sha(before_test) == TEST_BEFORE_SHA, "v1.0 modified-file baseline mismatch")
    project_entries = sorted(name for name in v10_files if name.startswith(prefix))
    require(len(project_entries) == 16, "v1.0 project file count mismatch")
    changed = []
    for archive_name in project_entries:
        relative = archive_name[len(prefix):]
        before = v10_files[archive_name]
        after = (root / relative).read_bytes()
        is_changed = before != after
        if is_changed:
            changed.append(relative)
        rows.append({"relative_path": relative,
                     "before_size_bytes": len(before), "before_sha256": sha(before),
                     "after_size_bytes": len(after), "after_sha256": sha(after),
                     "before_after_byte_identical": not is_changed,
                     "change_status": "MODIFIED_ALLOWED" if is_changed else "PRESERVED",
                     "status": "PASS"})
    require(changed == sorted([CONTROLLER, TEST]),
            "project modifications exceed controller and test: " + repr(changed))
    for relative, digest in UNCHANGED_HASHES.items():
        require(sha((root / relative).read_bytes()) == digest, "v1.0 protected Step10C file changed: " + relative)
    return ({"schema_version": SCHEMA + ".v10_project_preservation",
             "project_file_count": 16, "modified_allowed_count": 2,
             "preserved_count": 14, "modified_files": changed,
             "records": rows, "status": "PASS"}, before_controller, before_test)


def source_contract(root: Path) -> dict[str, Any]:
    source = (root / CONTROLLER).read_text(encoding="utf-8-sig")
    occurrence = source.count("entryController.LastCompletion")
    state_guard = source.index("flow.State != PetRobotPortfolioQuestState.AwaitingPractice")
    snapshot_guard = source.index("flow.Snapshot != null")
    local_read = source.index(
        "PetRobotPracticeCompletionSnapshot completion = entryController.LastCompletion;")
    null_guard = source.index("if (completion == null) return false;")
    freeze = source.index("flow.ObserveCompleted(completion)")
    disable = source.index("entryController.enabled = false")
    require(occurrence == 1, "LastCompletion source occurrence is not one")
    require(state_guard < snapshot_guard < local_read < null_guard < freeze < disable,
            "one-shot guard/read/freeze/disable order mismatch")
    require("TryConsumeCompleted" not in source and "PetRobotWorldHandoffChannel" not in source,
            "Step10B handoff is re-consumed")
    return {"schema_version": SCHEMA + ".one_shot_source_contract",
            "last_completion_source_occurrence_count": occurrence,
            "guard_precedes_property_read": True, "state_guard_offset": state_guard,
            "snapshot_guard_offset": snapshot_guard, "property_read_offset": local_read,
            "local_null_guard_offset": null_guard, "freeze_offset": freeze,
            "entry_disable_offset": disable, "subsequent_access_blocked_by_state_or_snapshot": True,
            "readiness_polling_accesses_per_call_maximum": 1,
            "handoff_second_consume_count": 0, "status": "PASS"}


def parse_tests(source: bytes, source_path: str) -> list[str]:
    text = source.decode("utf-8-sig")
    methods = re.findall(r"\[Test\]\s*public\s+void\s+(\w+)\s*\(", text)
    require(len(methods) == len(set(methods)), "duplicate test method: " + source_path)
    return methods


def test_inventory(root: Path, v10: Any, before_test: bytes) -> dict[str, Any]:
    inventory = v10.test_inventory(root)
    before_methods = parse_tests(before_test, "v1.0 test")
    after_methods = parse_tests((root / TEST).read_bytes(), TEST)
    require(before_methods == after_methods and len(after_methods) == 18,
            "Step10C fixture method names changed")
    return {**inventory, "v10_fixture_method_names_preserved": True,
            "v10_fixture_before_count": 18, "v11_fixture_after_count": 18,
            "source_assertion_checks_last_completion_occurrence": True,
            "source_assertion_checks_guard_order": True, "status": "PASS"}


def guid_evidence(root: Path) -> tuple[dict[str, Any], dict[str, bytes], dict[str, bytes]]:
    meta_by_guid: dict[str, list[str]] = {}
    meta_bytes: dict[str, bytes] = {}
    for path in (root / "Assets").rglob("*.meta"):
        data = path.read_bytes()
        match = re.search(rb"(?m)^guid:\s*([0-9a-f]{32})\s*$", data)
        if not match:
            continue
        relative = path.relative_to(root).as_posix()
        guid = match.group(1).decode("ascii")
        meta_by_guid.setdefault(guid, []).append(relative)
        meta_bytes[relative] = data
    duplicates = {guid: paths for guid, paths in meta_by_guid.items() if len(paths) != 1}
    require(not duplicates, "duplicate project meta GUIDs: " + repr(duplicates))

    prefab_text = (root / PREFAB).read_text(encoding="utf-8-sig")
    requested = []
    snapshots = {}
    for asset, expected_guid in REFERENCE_ASSETS:
        meta = asset + ".meta"
        raw_data = (root / asset).read_bytes()
        meta_data = (root / meta).read_bytes()
        parsed = re.search(rb"(?m)^guid:\s*([0-9a-f]{32})\s*$", meta_data)
        require(parsed is not None and parsed.group(1).decode("ascii") == expected_guid,
                "reference meta GUID mismatch: " + asset)
        require(meta_by_guid.get(expected_guid) == [meta], "reference GUID path mismatch: " + asset)
        require(prefab_text.count("guid: " + expected_guid) == 1,
                "Prefab reference GUID count mismatch: " + asset)
        requested.append({"asset": record(asset, raw_data), "meta": record(meta, meta_data),
                          "parsed_meta_guid": expected_guid, "prefab_serialized_guid": expected_guid,
                          "prefab_occurrence_count": 1, "guid_path_pair_match": True,
                          "before_after_byte_identical": True, "status": "PASS"})
        snapshots[asset] = raw_data
        snapshots[meta] = meta_data

    texts = {PREFAB: prefab_text, SCENE: (root / SCENE).read_text(encoding="utf-8-sig")}
    graph_rows = []
    referenced_guids = set()
    for source, text in texts.items():
        for guid in sorted(set(re.findall(r"guid:\s*([0-9a-f]{32})", text))):
            if guid.startswith("0000000000000000"):
                continue
            referenced_guids.add(guid)
            paths = meta_by_guid.get(guid, [])
            require(len(paths) == 1, "unresolved Scene/Prefab GUID: " + guid)
            path = paths[0]
            if path.startswith("Assets/_Project/Scripts/Practice/PetRobotPortfolioQuest/") or \
                    path in (PREFAB + ".meta", SCENE + ".meta"):
                category = "STEP10C_NEW_META"
            elif "PetRobotWorldIntegration" in path or path.endswith("PetRobotWorldEntryReturn.prefab.meta"):
                category = "APPROVED_STEP10B_ASSET"
            elif path.endswith("PF_House_C.prefab.meta"):
                category = "APPROVED_HOUSE_PREFAB"
            elif any(guid == expected for _, expected in REFERENCE_ASSETS):
                category = "EXISTING_VISUAL_OR_FONT_META"
            else:
                category = "OTHER_RESOLVED_PROJECT_META"
            graph_rows.append({"source_path": source, "guid": guid,
                               "resolved_meta_path": path, "category": category,
                               "project_guid_occurrence_count": 1, "status": "PASS"})
    require(len(referenced_guids) == 12 and len(graph_rows) == 12,
            "unexpected non-builtin Scene/Prefab GUID graph size")

    subset: dict[str, bytes] = {}
    for row in graph_rows:
        relative = row["resolved_meta_path"]
        subset["input_subset/reference_meta/" + row["guid"] + ".meta"] = meta_bytes[relative]
    output = {"schema_version": SCHEMA + ".reference_guid_graph",
              "project_meta_guid_count": len(meta_by_guid), "duplicate_guid_count": 0,
              "source_files": [PREFAB, SCENE], "non_builtin_reference_count": len(graph_rows),
              "resolved_reference_count": len(graph_rows), "unresolved_guid_count": 0,
              "requested_asset_pair_count": 4, "requested_asset_pairs": requested,
              "reference_graph": graph_rows, "unity_import_claimed": False, "status": "PASS"}
    return output, subset, snapshots


PROBE_SOURCE = r'''
using System;
using WordsInSilence.Practice.PetRobotVerticalSlice;
using WordsInSilence.Practice.PetRobotWorldIntegration;
using WordsInSilence.Practice.PetRobotPortfolioQuest;
static class Probe {
 static PetRobotPracticeCompletionSnapshot Result(bool needs,int saved){return new PetRobotPracticeCompletionSnapshot(PetRobotWordLearningStatus.Mastered,PetRobotWordLearningStatus.Mastered,needs?PetRobotWordLearningStatus.NeedsPractice:PetRobotWordLearningStatus.Mastered,needs?2:3,needs?1:0,saved);}
 static void Set(PetRobotWorldEntryController w,PetRobotPracticeCompletionSnapshot v){typeof(PetRobotWorldEntryController).GetProperty("LastCompletion").GetSetMethod(true).Invoke(w,new object[]{v});}
 static bool Branch(bool needs){var f=new PetRobotPortfolioQuestFlow();bool a=f.ObserveCompleted(Result(needs,8));f.SetNpcInRange(true);bool b=f.TryOpenDialogue(true)&&!f.TryOpenDialogue(true);bool c=f.TryCompleteDialogue(true)&&!f.TryCompleteDialogue(true);return a&&b&&c&&f.Snapshot.MasteredCount==(needs?2:3)&&f.Snapshot.NeedsPracticeCount==(needs?1:0)&&f.Snapshot.CandidateSavedTotal==8;}
 static bool OneShot(){var w=new PetRobotWorldEntryController();var q=new PetRobotPortfolioQuestController();q.ConfigureForTests(w,null,null,null,2.5f);Set(w,Result(false,8));bool first=q.CaptureCompletionFromWorldOnce();Set(w,Result(true,99));bool second=q.CaptureCompletionFromWorldOnce();return first&&!second&&!w.enabled&&q.Flow.CompletionFreezeCount==1&&q.Completion.CandidateSavedTotal==8&&q.State==PetRobotPortfolioQuestState.AwaitingNpcReport;}
 static void Main(){bool a=Branch(false),b=Branch(true),c=OneShot();Console.WriteLine("MASTERED_3_NEEDS_0="+a);Console.WriteLine("MASTERED_2_NEEDS_1="+b);Console.WriteLine("ONE_SHOT_CONTROLLER="+c);Console.WriteLine("PROBE_STATUS="+(a&&b&&c?"PASS":"FAIL"));if(!a||!b||!c)Environment.Exit(2);}
}
'''


def offline_compile(root: Path, v10: Any, base: Any, tx: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    mono = Path(r"C:\Program Files\Unity\Hub\Editor\6000.3.14f1\Editor\Data\MonoBleedingEdge\bin\mono.exe")
    csc = Path(r"C:\Program Files\Unity\Hub\Editor\6000.3.14f1\Editor\Data\MonoBleedingEdge\lib\mono\4.5\csc.exe")
    require(mono.is_file() and csc.is_file(), "Unity Mono Roslyn missing")
    unity_stub, nunit_stub = tx.load_v10_stubs(root)
    unity_stub = v10.extend_stub(base.extend_unity_stub(unity_stub))
    nunit_stub = nunit_stub.replace(
        "public static void IsFalse(bool v,string m=null){}",
        "public static void IsFalse(bool v,string m=null){} public static void IsNull(object v,string m=null){}")
    production = [
        "Assets/_Project/Scripts/Practice/PetRobotWorldIntegration/PetRobotWorldIntegrationContracts.cs",
        "Assets/_Project/Scripts/Practice/PetRobotWorldIntegration/PetRobotWorldEntryController.cs",
        "Assets/_Project/Scripts/Practice/PetRobotWorldIntegration/PetRobotPracticeReturnWatcher.cs",
        "Assets/_Project/Scripts/Practice/PetRobotPortfolioQuest/PetRobotPortfolioQuestContracts.cs",
        CONTROLLER,
        "Assets/_Project/Scripts/Practice/PetRobotPortfolioQuest/PetRobotPortfolioQuestUiPresenter.cs",
        "Assets/_Project/Scripts/Practice/PetRobotPortfolioQuest/PetRobotPortfolioQuestVisuals.cs",
    ]
    temporary = Path(tempfile.mkdtemp(prefix="WIS-Step10C-v1_1-compile-"))
    outputs: dict[str, str] = {}
    try:
        (temporary / "unity.cs").write_text(unity_stub, encoding="utf-8")
        (temporary / "nunit.cs").write_text(nunit_stub, encoding="utf-8")
        dll = temporary / "production.dll"
        run = subprocess.run([str(mono), str(csc), "/nologo", "/target:library", "/langversion:latest",
                              "/out:" + str(dll), str(temporary / "unity.cs"),
                              *[str((root / path).resolve()) for path in production]],
                             capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        outputs["production"] = (run.stdout or "") + (run.stderr or "")
        require(run.returncode == 0, "offline production compile failed:\n" + outputs["production"][-12000:])
        tests = temporary / "tests.dll"
        run = subprocess.run([str(mono), str(csc), "/nologo", "/target:library", "/langversion:latest",
                              "/out:" + str(tests), "/r:" + str(dll), str(temporary / "nunit.cs"),
                              str((root / TEST).resolve())], capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=120)
        outputs["test"] = (run.stdout or "") + (run.stderr or "")
        require(run.returncode == 0, "offline test compile failed:\n" + outputs["test"][-12000:])
        (temporary / "probe.cs").write_text(PROBE_SOURCE, encoding="utf-8")
        exe = temporary / "probe.exe"
        run = subprocess.run([str(mono), str(csc), "/nologo", "/target:exe", "/langversion:latest",
                              "/out:" + str(exe), "/r:" + str(dll), str(temporary / "probe.cs")],
                             capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        outputs["probe_compile"] = (run.stdout or "") + (run.stderr or "")
        require(run.returncode == 0, "offline probe compile failed:\n" + outputs["probe_compile"][-12000:])
        run = subprocess.run([str(mono), str(exe)], cwd=temporary, capture_output=True,
                             text=True, encoding="utf-8", errors="replace", timeout=120)
        outputs["probe"] = (run.stdout or "") + (run.stderr or "")
        require(run.returncode == 0 and "PROBE_STATUS=PASS" in outputs["probe"],
                "offline behavior probe failed:\n" + outputs["probe"][-12000:])
        values = dict(line.lstrip("\ufeff").split("=", 1) for line in outputs["probe"].splitlines() if "=" in line)
        require(values.get("MASTERED_3_NEEDS_0") == "True" and
                values.get("MASTERED_2_NEEDS_1") == "True" and
                values.get("ONE_SHOT_CONTROLLER") == "True", "probe value mismatch")
        compile_doc = {"schema_version": SCHEMA + ".offline_compile",
                       "compiler": "Unity 6000.3.14f1 Mono Roslyn csc",
                       "production_source_count": len(production), "test_source_count": 1,
                       "production_error_count": len(re.findall(r"error CS\d+", outputs["production"])),
                       "test_error_count": len(re.findall(r"error CS\d+", outputs["test"])),
                       "probe_error_count": len(re.findall(r"error CS\d+", outputs["probe_compile"])),
                       "unity_gui_compile_claimed": False, "status": "PASS"}
        probe_doc = {"schema_version": SCHEMA + ".behavior_probe", "values": values,
                     "quest_branches_passed": 2, "one_shot_controller_passed": True,
                     "second_capture_call_changed_snapshot": False,
                     "unity_execution_claimed": False, "status": "PASS"}
        return compile_doc, probe_doc, outputs
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def manifest(files: dict[str, bytes]) -> dict[str, bytes]:
    require("output_file_manifest.json" not in files, "manifest already exists")
    rows = [record(name, files[name]) for name in sorted(files)]
    return {**files, "output_file_manifest.json": jbytes({
        "schema_version": SCHEMA + ".output_manifest", "self_excluded": True, "files": rows})}


def assertions(rows: list[tuple[str, str, str]]) -> bytes:
    keys = set()
    output = []
    for key, status, detail in rows:
        require(key not in keys, "duplicate assertion key: " + key)
        require(status in ("PASS", "FAIL", "UNVERIFIED", "NOT_APPLICABLE"), "invalid status")
        keys.add(key)
        output.append(json.dumps({"assertion_key": key, "status": status, "detail": detail},
                                 ensure_ascii=False, allow_nan=False, sort_keys=True))
    return ("\n".join(output) + "\n").encode("utf-8")


def build(root: Path) -> None:
    validation = root / "Validation"
    result = validation / NAME
    delivery = validation / (NAME + "_delivery.zip")
    detached = Path(str(delivery) + ".sha256")
    require(not result.exists() and not delivery.exists() and not detached.exists(),
            "never-overwrite target exists")
    require(staged_files(root) == [], "Git staged files are not zero before build")

    v10 = load_module(root, V10_TOOL, "step10c_v10")
    base = load_module(root, BASE_TOOL, "step10c_v11_base")
    tx = load_module(root, TX_TOOL, "step10c_v11_tx")
    fixed, v10_files = verify_v10(root, base, tx)
    preservation, before_controller, before_test = preserve_v10_project(root, v10_files)
    contract = source_contract(root)
    tests = test_inventory(root, v10, before_test)
    guid, meta_subset, reference_snapshots = guid_evidence(root)

    gui = root / STEP10B_GUI
    project = v10.verify_records(root, load_json(gui / "step10a_project_files_before_after.json"),
                                 30, "step10a_project_files")
    protected = v10.verify_records(root, load_json(gui / "protected_files_before_after.json"),
                                   106, "protected_files")
    profile_source = load_json(gui / "active_candidate_profile_before_after.json")
    profile = v10.verify_records(root, profile_source, 22, "active_candidate_profile")
    require(profile_source["prototype_count"] == 9 and profile_source["threshold_count"] == 3 and
            profile_source["thresholds"] == EXPECTED_THRESHOLDS, "profile/prototype/threshold mismatch")
    profile.update({"prototype_count": 9, "threshold_count": 3,
                    "thresholds": EXPECTED_THRESHOLDS})
    repository = v10.verify_repository(load_json(gui / "candidate_repository_after.json"))
    before_build = (root / STEP10B_RESULT /
                    "changes/after/ProjectSettings/EditorBuildSettings.asset").read_bytes()
    step10b = v10.verify_step10b_files(
        root, load_json(gui / "step10b_project_files_before_after.json"), before_build)

    compile_doc, probe_doc, compile_outputs = offline_compile(root, v10, base, tx)
    transaction = tx.transaction_regression()
    require(all((root / relative).read_bytes() == data
                for relative, data in reference_snapshots.items()), "reference asset/meta changed")

    after_controller = (root / CONTROLLER).read_bytes()
    after_test = (root / TEST).read_bytes()
    diff = "".join(difflib.unified_diff(
        before_controller.decode("utf-8-sig").splitlines(True),
        after_controller.decode("utf-8-sig").splitlines(True),
        fromfile="a/" + CONTROLLER, tofile="b/" + CONTROLLER))
    diff += "".join(difflib.unified_diff(
        before_test.decode("utf-8-sig").splitlines(True),
        after_test.decode("utf-8-sig").splitlines(True),
        fromfile="a/" + TEST, tofile="b/" + TEST))
    delta = {"schema_version": SCHEMA + ".implementation_delta",
             "modified_project_file_count": 2,
             "modified_files": [
                 {"relative_path": CONTROLLER, "before": record(CONTROLLER, before_controller),
                  "after": record(CONTROLLER, after_controller)},
                 {"relative_path": TEST, "before": record(TEST, before_test),
                  "after": record(TEST, after_test)}],
             "added_project_file_count": 0, "other_v10_project_files_preserved": 14,
             "status": "PASS"}

    assertion_blob = assertions([
        ("implementation_owner_codex", "PASS", "project context section 3"),
        ("fixed_v10_delivery", "PASS", V10_ZIP_SHA),
        ("fixed_v10_deterministic", "PASS", "45 entries / 44 manifest targets"),
        ("controller_before_identity", "PASS", CONTROLLER_BEFORE_SHA),
        ("test_before_identity", "PASS", TEST_BEFORE_SHA),
        ("modified_project_files_exactly_two", "PASS", "controller + existing test"),
        ("other_v10_project_files_preserved", "PASS", "14/14"),
        ("last_completion_source_occurrence_one", "PASS", "1"),
        ("state_snapshot_guard_before_property", "PASS", "guard offsets precede local read"),
        ("completion_local_read_per_call_one", "PASS", "single local property read"),
        ("completion_post_freeze_reaccess_blocked", "PASS", "state/snapshot guard"),
        ("entry_disabled_after_success", "PASS", "behavior probe"),
        ("handoff_second_consume_zero", "PASS", "0"),
        ("false_positive_assertion_repaired", "PASS", "source occurrence and guard order"),
        ("mastered_branch", "PASS", probe_doc["values"]["MASTERED_3_NEEDS_0"]),
        ("needs_practice_branch", "PASS", probe_doc["values"]["MASTERED_2_NEEDS_1"]),
        ("one_shot_controller_probe", "PASS", probe_doc["values"]["ONE_SHOT_CONTROLLER"]),
        ("existing_test_names_120_preserved", "PASS", "120/120"),
        ("step10c_fixture_methods_18_preserved", "PASS", "18/18"),
        ("combined_static_inventory", "PASS", "138"),
        ("reference_asset_pairs", "PASS", "4/4 raw+meta"),
        ("reference_guid_graph", "PASS", "12/12 resolved"),
        ("reference_guid_duplicates", "PASS", "0"),
        ("reference_guid_unresolved", "PASS", "0"),
        ("scene_prefab_build_unchanged", "PASS", "v1.0 byte-identical"),
        ("offline_production_compile", "PASS", "error 0"),
        ("offline_test_compile", "PASS", "error 0"),
        ("offline_probe_compile", "PASS", "error 0"),
        ("step10a_project_preserved", "PASS", "30/30"),
        ("protected_files_preserved", "PASS", "106/106"),
        ("step10b_baseline_preserved", "PASS", "21 unchanged + approved Build append"),
        ("candidate_profile_preserved", "PASS", "22/22"),
        ("prototype_threshold_preserved", "PASS", "9/9 + 3/3"),
        ("candidate_repository_preserved", "PASS", "8 candidates / 25 files"),
        ("git_staged_zero", "PASS", "0"),
        ("transaction_failure_injection", "PASS", "4/4"),
        ("never_overwrite", "PASS", "second commit blocked"),
        ("unity_gui_compile", "UNVERIFIED", "source stage prohibited"),
        ("unity_editmode", "UNVERIFIED", "138 tests not executed"),
        ("unity_playmode", "UNVERIFIED", "portfolio quest not executed"),
        ("webcam_live_capture", "NOT_APPLICABLE", "source stage prohibited"),
    ])
    primary = {"schema_version": SCHEMA + ".primary_verdict", "status": "PASS",
               "verdict": "MINIMAL_ONE_SHOT_AND_REFERENCE_EVIDENCE_REPAIR_SOURCE_AND_PACKAGE_READY",
               "last_completion_source_occurrence_count": 1,
               "unity_gui_compile": "UNVERIFIED", "unity_editmode": "UNVERIFIED",
               "unity_playmode": "UNVERIFIED", "webcam_live_capture": "NOT_APPLICABLE",
               "status_strings": [
                   "STEP10C_PORTFOLIO_VERTICAL_SLICE_QUEST_PRESENTATION_SHELL_V1_1_SOURCE_AND_PACKAGE_READY",
                   "STEP10C_ONE_SHOT_COMPLETION_READ_REPAIR_IMPLEMENTED",
                   "STEP10C_REFERENCED_ASSET_GUID_EVIDENCE_COMPLETE",
                   "STEP10C_V1_0_APPROVED_SCOPE_PRESERVED",
                   "STEP10C_UNITY_GUI_EDITMODE_PLAYMODE_UNVERIFIED",
                   "BLOCKED_PENDING_EXTERNAL_REVIEW"]}
    report = """# Step 10C v1.1 — One-shot Completion Read and GUID Evidence Repair

## Verdict

PASS — the minimal source repair and deterministic Delivery are ready for independent external review.

## Repair

- `LastCompletion` has one source occurrence and is read once into a local variable per eligible polling call.
- state/snapshot guards run before the property read, so a frozen completion is never read again.
- the first completed snapshot remains immutable; the entry controller is disabled only after successful freeze.
- the existing 18-test fixture now checks source occurrence/guard ordering and calls the controller twice.

## Reference evidence

- Player, pet robot, education NPC and NotoSansKR raw asset/meta pairs are byte-identified.
- Their parsed `.meta` GUIDs match the Prefab references.
- All 12 non-builtin Scene/Prefab GUIDs resolve uniquely; duplicate and unresolved counts are zero.
- Unity import is not claimed at this source stage.

## Verification

- Existing 120 tests and Step 10C 18 method names are preserved; static total is 138.
- Offline Unity Mono Roslyn production/test/probe errors: 0/0/0.
- Mastered 3/NeedsPractice 0, Mastered 2/NeedsPractice 1 and one-shot controller probes pass.
- Step 10A 30/30, protected 106/106, Step 10B baseline, profile 22/22, prototypes 9/9, thresholds 3/3 and repository 8/25 are preserved.

## Execution boundary

- Unity GUI compile/EditMode/PlayMode: UNVERIFIED
- Webcam/live capture: NOT_APPLICABLE
"""

    files: dict[str, bytes] = {
        "input_provenance.json": jbytes({"schema_version": SCHEMA + ".input_provenance",
                                           "fixed_v10": fixed, "implementation_owner": "Codex",
                                           "unity_version": "6000.3.14f1", "status": "PASS"}),
        "implementation_delta.json": jbytes(delta),
        "one_shot_completion_source_contract.json": jbytes(contract),
        "reference_asset_guid_evidence.json": jbytes(guid),
        "test_inventory.json": jbytes(tests),
        "offline_compile_evidence.json": jbytes(compile_doc),
        "offline_behavior_probe.json": jbytes(probe_doc),
        "offline_production_compile_output.txt": compile_outputs["production"].encode("utf-8"),
        "offline_test_compile_output.txt": compile_outputs["test"].encode("utf-8"),
        "offline_probe_compile_output.txt": compile_outputs["probe_compile"].encode("utf-8"),
        "offline_behavior_probe_output.txt": compile_outputs["probe"].encode("utf-8"),
        "v10_project_files_before_after.json": jbytes(preservation),
        "step10a_project_files_before_after.json": jbytes(project),
        "protected_files_before_after.json": jbytes(protected),
        "step10b_files_before_after.json": jbytes(step10b),
        "active_candidate_profile_before_after.json": jbytes(profile),
        "candidate_repository_before_after.json": jbytes(repository),
        "git_staged_before_after.json": jbytes({"before_count": 0, "after_count": 0,
                                                  "before_after_unchanged": True, "status": "PASS"}),
        "packaging_transaction_failure_injection_regression.json": jbytes(transaction),
        "primary_verdict.json": jbytes(primary), "ASSERTIONS.jsonl": assertion_blob,
        "VALIDATION_REPORT.md": report.encode("utf-8"),
        "changes/before/PetRobotPortfolioQuestController.cs": before_controller,
        "changes/after/PetRobotPortfolioQuestController.cs": after_controller,
        "changes/before/PetRobotPortfolioQuestPresentationV1Tests.cs": before_test,
        "changes/after/PetRobotPortfolioQuestPresentationV1Tests.cs": after_test,
        "changes/unified_diff.patch": diff.encode("utf-8"),
        "analysis/step10c_v1_1_packaging_tool.py": Path(__file__).read_bytes(),
    }
    for archive_name, data in v10_files.items():
        if archive_name.startswith("project_files/"):
            relative = archive_name[len("project_files/"):]
            files[archive_name] = (root / relative).read_bytes()
    files.update(meta_subset)
    complete = manifest(files)
    zip_blob = tx.deterministic_zip(complete)
    detached_blob = (sha(zip_blob) + "  " + delivery.name + "\n").encode("ascii")
    pre = base.validate_zip_bytes(zip_blob, complete)
    require(pre["duplicate_count"] == 0 and pre["unsafe_path_count"] == 0 and
            pre["encrypted_entry_count"] == 0 and pre["symlink_entry_count"] == 0 and
            pre["crc_failure"] is None and pre["membership_match"] and
            pre["payload_mismatch_count"] == 0, "pre-commit ZIP validation failed")

    def validator(final_result: Path, final_zip: Path, final_detached: Path) -> None:
        disk = {path.relative_to(final_result).as_posix(): path.read_bytes()
                for path in final_result.rglob("*") if path.is_file()}
        require(disk == complete, "committed result map mismatch")
        require(tx.deterministic_zip(disk) == final_zip.read_bytes(), "deterministic ZIP mismatch")
        require(final_detached.read_bytes() == detached_blob, "detached checksum mismatch")
        post = base.validate_zip_bytes(final_zip.read_bytes(), disk)
        require(post["duplicate_count"] == 0 and post["unsafe_path_count"] == 0 and
                post["encrypted_entry_count"] == 0 and post["symlink_entry_count"] == 0 and
                post["crc_failure"] is None and post["membership_match"] and
                post["payload_mismatch_count"] == 0, "post-commit ZIP validation failed")

    paths = tx.commit_outputs(validation, complete,
                              names=(NAME, NAME + "_delivery.zip", NAME + "_delivery.zip.sha256"),
                              validator=validator)
    require(paths == (result, delivery, detached), "unexpected transaction targets")
    require(staged_files(root) == [], "Git staged files changed")
    require(all((root / relative).read_bytes() == data
                for relative, data in reference_snapshots.items()), "reference assets changed after packaging")
    print(json.dumps({"status": "PASS", "result_directory": str(result),
                      "result_file_count": len(complete), "manifest_target_count": len(complete) - 1,
                      "delivery_zip": str(delivery), "delivery_size_bytes": delivery.stat().st_size,
                      "delivery_sha256": sha(delivery.read_bytes()),
                      "detached_checksum": str(detached), "detached_size_bytes": detached.stat().st_size,
                      "detached_file_sha256": sha(detached.read_bytes()),
                      "controller_before": record(CONTROLLER, before_controller),
                      "controller_after": record(CONTROLLER, after_controller),
                      "test_before": record(TEST, before_test), "test_after": record(TEST, after_test),
                      "last_completion_occurrences": 1, "reference_guids": "12/12",
                      "tests": "120 + 18 = 138", "offline_compile_errors": "0/0/0",
                      "quest_branches": "2/2 PASS", "candidate_repository": "8/25 unchanged",
                      "git_staged": 0, "unity_gui_editmode_playmode": "UNVERIFIED",
                      "webcam_live": "NOT_APPLICABLE"}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    require(args.build, "use --build")
    build(args.root.resolve())


if __name__ == "__main__":
    main()
