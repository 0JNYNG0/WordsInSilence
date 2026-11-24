#!/usr/bin/env python3
"""Step 10D v1.4 scene wiring and Unity GUI evidence validator.

The preflight and static-verification modes never start Unity.  They only write
under Validation and never write a project asset.  GUI evidence packaging is
performed only after the user supplies the single-run Unity artifacts.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import io
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import shutil
import subprocess
import tempfile
from typing import Any
import xml.etree.ElementTree as ET
import zipfile


SCHEMA = "WordsInSilence.Step10D.ScenePlacedPlayerMoaRobotWiringGuiGate.v1.4"
NAME = "WordsInSilence_step10d_scene_placed_player_moa_robot_wiring_and_gui_gate_v1_4"
SCENE = "Assets/_Project/Scenes/VerticalSlice/PetRobotPortfolioQuestVerticalSlice.unity"
WORLD_PREFAB = "Assets/_Project/Resources/VerticalSlice/PetRobotWorldEntryReturn.prefab"
PRESENTATION_PREFAB = "Assets/_Project/Resources/VerticalSlice/PetRobotPortfolioQuestPresentation.prefab"
WRAPPER = "Assets/_Project/Resources/VerticalSlice/PetRobotAnimatedVisualWrapper.prefab"
PLACEHOLDER_SOURCE = "Assets/_Project/Scripts/Practice/PetRobotWorldIntegration/PetRobotWorldPlaceholderVisuals.cs"
TARGET_TEST = "Assets/_Project/Tests/EditMode/PetRobotFloatingCompanionPresentationV1Tests.cs"
VISUAL_FACTORY = "Assets/_Project/Scripts/Practice/PetRobotPortfolioQuest/PetRobotPortfolioQuestVisuals.cs"
COMPANION_SOURCE = "Assets/_Project/Scripts/Practice/PetRobotPortfolioQuest/PetRobotFloatingCompanionController.cs"

PLAYER_GUID = "f44a2ab6c6a52dd41b44e781d7f0c07a"
ROBOT_GUID = "50573fbff0bc00c418eb7319a40093d3"
ROBOT_CONTROLLER_GUID = "a414e66cfd19823419a0f9a1954db8d3"
WORLD_GUID = "10b00000000000000000000000000010"
PRESENTATION_GUID = "10c00000000000000000000000000010"
PLAYER_INSTANCE = "1006113214"
ROBOT_INSTANCE = "1275187441"
PLAYER_ROOT_TRANSFORM = "100100010"
ROBOT_ANCHOR_TRANSFORM = "100100011"
PLAYER_STRIPPED_TRANSFORM = "1006113215"
ROBOT_STRIPPED_TRANSFORM = "1275187442"

APPROVED = {
    PLACEHOLDER_SOURCE: (2032, "2f973da73f619add63bb5543eb0a9018640efe3af063de542d930427b5e29ab4"),
    TARGET_TEST: (12154, "3726968c9e8113ed8f08f184aae4b41862595ac6d6452cdb367737e684472ef7"),
    VISUAL_FACTORY: (1906, "76d8df6b1a52354495d00cff26194c80e98e178a00cab706fd61fb1bb7465543"),
    WRAPPER: (2706, "fa906ebbeebd5a4b309d5b741d4126d2000c1818a75b68d1c5b54ff22b2197a8"),
    PRESENTATION_PREFAB: (4315, "4fd26a3ed6b1b3aacdfc1262eb73e1b136b1bcbd63849c59dcccd9e58dfde602"),
    WORLD_PREFAB: (9861, "d035ba31f6b23eaa9ee97d7403d70b5759c53687e6a499bbcf4e1de60577603b"),
}

CHECKPOINT = (
    "Validation/WordsInSilence_step10d_existing_fbx_animator_floating_pet_robot_"
    "companion_presentation_layer_v1_0/asset_identity_and_guid_graph.json"
)
BASELINE_SCENE_SIZE = 14750
BASELINE_SCENE_SHA = "1a8eb3162be8fbf5cd55aa0ee32d8f364a0d8408d5cebab2009ac5fed2245894"

RESULT_NAME = NAME + "_evidence"
DELIVERY_NAME = NAME + "_delivery.zip"
V13_DELIVERY = "Validation/WordsInSilence_step10d_disabled_placeholder_awake_guard_repair_v1_3_delivery.zip"
V13_DELIVERY_SIZE = 64173
V13_DELIVERY_SHA = "49d297909be339fb111695e9fbec878ada8f921c9ac182868771120dafee3723"
V13_DETACHED_SIZE = 147
V13_DETACHED_SHA = "c7263173e95cc52f02b6d6d6e806e942847bcaf0f91421df33691af31976e085"

REQUIRED_FAILURE_EVIDENCE = [
    "02_test_runner_targeted_14.png",
    "03_scene_open.png",
    "04_player_hierarchy.png",
    "05_robot_hierarchy.png",
    "06_robot_animator_missing.png",
    "Step10D_v1_4_Targeted14_Editor.log",
    "Step10D_v1_4_Targeted14_TestResults.xml",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def record(relative: str, data: bytes) -> dict[str, Any]:
    return {"relative_path": relative, "size_bytes": len(data), "sha256": sha(data)}


def jbytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def write_new(path: Path, data: bytes) -> None:
    require(not path.exists(), "never-overwrite target exists: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def normalized(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def fixed_identities(root: Path) -> dict[str, Any]:
    rows = []
    for relative, (size, digest) in APPROVED.items():
        data = (root / relative).read_bytes()
        match = len(data) == size and sha(data) == digest
        require(match, "approved identity mismatch: " + relative)
        rows.append({
            **record(relative, data),
            "approved_size_bytes": size,
            "approved_sha256": digest,
            "match": match,
            "status": "PASS",
        })
    return {"target_count": len(rows), "match_count": len(rows), "records": rows, "status": "PASS"}


def git_staged(root: Path) -> dict[str, Any]:
    run = subprocess.run(
        ["git", "diff", "--cached", "--name-only"], cwd=root,
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
    )
    require(run.returncode == 0, "Git staged query failed")
    paths = [line for line in run.stdout.splitlines() if line.strip()]
    require(not paths, "Git staged files are nonzero")
    return {"staged_file_count": 0, "staged_paths": paths, "status": "PASS"}


def checkpoint_compare(root: Path) -> dict[str, Any]:
    source = root / CHECKPOINT
    checkpoint = json.loads(source.read_text(encoding="utf-8"))
    rows = []
    mismatch = []
    for prior in checkpoint["gameassets_dependencies"]:
        relative = prior["relative_path"]
        path = root / relative
        if not path.is_file():
            row = {"relative_path": relative, "status": "MISSING"}
            mismatch.append(row)
        else:
            data = path.read_bytes()
            same = len(data) == prior["size_bytes"] and sha(data) == prior["sha256"]
            row = {
                "relative_path": relative,
                "checkpoint_size_bytes": prior["size_bytes"],
                "checkpoint_sha256": prior["sha256"],
                "current_size_bytes": len(data),
                "current_sha256": sha(data),
                "byte_identical_to_post_step10d_checkpoint": same,
                "status": "PASS" if same else "FAIL",
            }
            if not same:
                mismatch.append(row)
        rows.append(row)
    require(len(rows) == 242 and not mismatch, "GameAssets post-Step10D checkpoint mismatch")
    return {
        "semantics": "post-Step10D checkpoint change detection only; not retroactive preimplementation preservation",
        "target_count": 242,
        "match_count": 242,
        "mismatch_count": 0,
        "retroactive_preimplementation_preservation": "UNVERIFIED_NONBLOCKING",
        "records": rows,
        "status": "PASS_POST_STEP10D_CHECKPOINT_UNCHANGED",
    }


def prefab_blocks(scene: str) -> dict[str, str]:
    return {
        match.group("id"): match.group("body")
        for match in re.finditer(
            r"--- !u!1001 &(?P<id>\d+)\nPrefabInstance:\n(?P<body>.*?)(?=\n--- !u!|\Z)",
            scene,
            flags=re.DOTALL,
        )
    }


def modification_value(body: str, guid: str, target: str, property_path: str) -> str | None:
    pattern = (
        r"- target: \{fileID: " + re.escape(target) + r", guid: " + re.escape(guid) +
        r", type: 3\}\n\s+propertyPath: " + re.escape(property_path) +
        r"\n\s+value:[ \t]*([^\n]*)"
    )
    match = re.search(pattern, body)
    return match.group(1).strip() if match else None


def transform_parent(body: str) -> str:
    match = re.search(r"m_TransformParent: \{fileID: ([-0-9]+)\}", body)
    require(match is not None, "PrefabInstance transform parent missing")
    return match.group(1)


def float_mod(body: str, guid: str, target: str, property_path: str) -> float:
    value = modification_value(body, guid, target, property_path)
    require(value is not None, "transform modification missing: " + property_path)
    return float(value)


def vector(body: str, guid: str, target: str, prefix: str) -> list[float]:
    return [float_mod(body, guid, target, prefix + suffix) for suffix in (".x", ".y", ".z")]


def find_candidate_scenes(root: Path) -> list[str]:
    candidates = []
    for path in sorted((root / "Assets/_Project/Scenes").rglob("*.unity")):
        text = normalized(path)
        conditions = [
            text.count("value: Player\n") == 1,
            text.count("value: MOA_robot\n") == 1,
            WORLD_GUID in text,
            PRESENTATION_GUID in text,
            "PetRobotPortfolioWorldBase" in text,
        ]
        if all(conditions):
            candidates.append(path.relative_to(root).as_posix())
    return candidates


def scene_before_inventory(root: Path, scene_data: bytes) -> dict[str, Any]:
    scene = scene_data.decode("utf-8").replace("\r\n", "\n")
    blocks = prefab_blocks(scene)
    require(PLAYER_INSTANCE in blocks and ROBOT_INSTANCE in blocks, "manual object PrefabInstances missing")
    player = blocks[PLAYER_INSTANCE]
    robot = blocks[ROBOT_INSTANCE]
    require(transform_parent(player) == "0" and transform_parent(robot) == "0", "manual objects are not baseline roots")
    require(modification_value(player, PLAYER_GUID, "919132149155446097", "m_Name") == "Player", "Player name mismatch")
    require(modification_value(robot, ROBOT_GUID, "919132149155446097", "m_Name") == "MOA_robot", "MOA_robot name mismatch")
    player_position = vector(player, PLAYER_GUID, "-8679921383154817045", "m_LocalPosition")
    robot_position = vector(robot, ROBOT_GUID, "-8679921383154817045", "m_LocalPosition")
    offset = [robot_position[index] - player_position[index] for index in range(3)]
    player_rotation = [
        float_mod(player, PLAYER_GUID, "-8679921383154817045", "m_LocalRotation." + suffix)
        for suffix in ("x", "y", "z", "w")
    ]
    robot_rotation = [
        float_mod(robot, ROBOT_GUID, "-8679921383154817045", "m_LocalRotation." + suffix)
        for suffix in ("x", "y", "z", "w")
    ]
    robot_scale = vector(robot, ROBOT_GUID, "-8679921383154817045", "m_LocalScale")
    return {
        "scene": record(SCENE, scene_data),
        "scene_candidate_count": 1,
        "scene_candidates": [SCENE],
        "manual_objects": {
            "Player": {
                "prefab_instance_file_id": int(PLAYER_INSTANCE),
                "source_guid": PLAYER_GUID,
                "source_path": "Assets/GameAssets/Player_model_animation_meshy/Meshy_AI_OriginalVer_Player_biped_Animation_Wave_One_Hand_withSkin.fbx",
                "parent_before": "SceneRoot",
                "world_position": player_position,
                "world_rotation_quaternion_xyzw": player_rotation,
                "world_scale": [1.0, 1.0, 1.0],
                "scale_evidence": "no instance scale override; ModelImporter globalScale=1",
                "imported_component_inventory": ["Transform", "Animator", "SkinnedMeshRenderer/model hierarchy"],
                "character_controller": "owned by PlayerPlaceholder gameplay root, not the visual",
                "movement_component": "owned by PlayerPlaceholder gameplay root, not the visual",
                "collider_import_setting": "addColliders=0",
            },
            "MOA_robot": {
                "prefab_instance_file_id": int(ROBOT_INSTANCE),
                "source_guid": ROBOT_GUID,
                "source_path": "Assets/GameAssets/petrobot_model_animation/MOA_robot.fbx",
                "parent_before": "SceneRoot",
                "world_position": robot_position,
                "world_rotation_quaternion_xyzw": robot_rotation,
                "world_scale": robot_scale,
                "imported_component_inventory": ["Transform", "Animator", "SkinnedMeshRenderer/model hierarchy"],
                "character_controller": False,
                "movement_component": False,
                "collider_import_setting": "addColliders=0",
            },
        },
        "robot_minus_player_position_xyz": offset,
        "existing_hierarchy": {
            "world": "PetRobotPortfolioWorldBase",
            "player_gameplay_root": "PlayerPlaceholder",
            "pet_robot_interaction_anchor": "PetRobotInteractionAnchor",
            "presentation": "PetRobotPortfolioQuestPresentation",
            "companion_controller_owner": "PetRobotPortfolioQuestPresentation",
        },
        "status": "PASS_UNIQUE_SAVED_SCENE_AND_MANUAL_OBJECTS_FIXED",
    }


def test_inventory(root: Path) -> dict[str, Any]:
    baseline_path = root / "Validation/WordsInSilence_step10d_existing_fbx_animator_floating_pet_robot_companion_presentation_layer_v1_0/test_inventory.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    approved = list(baseline["approved_rows"]) + list(baseline["new_rows"])
    current = []
    for relative in sorted({row["source_path"] for row in approved}):
        path = Path(relative)
        if not path.is_absolute():
            path = root / path
        text = path.read_text(encoding="utf-8")
        fixture = re.search(r"public sealed class\s+(\w+)", text)
        require(fixture is not None, "fixture class missing: " + relative)
        for method in re.findall(r"\[Test\]\s*public void\s+(\w+)\s*\(", text):
            current.append({"fixture": fixture.group(1), "method": method, "source_path": relative})
    key = lambda row: (row["fixture"], row["method"], row["source_path"])
    require(len(approved) == 152 and len(current) == 152, "test inventory count mismatch")
    require({key(row) for row in approved} == {key(row) for row in current}, "testcase name set mismatch")
    targeted = sorted(row["method"] for row in current if row["fixture"] == "PetRobotFloatingCompanionPresentationV1Tests")
    require(len(targeted) == 14, "targeted fixture count mismatch")
    return {
        "approved_testcase_count": 152,
        "current_testcase_count": 152,
        "name_set_preserved": True,
        "targeted_fixture_count": 14,
        "targeted_fixture_names": targeted,
        "status": "PASS",
    }


def static_wiring_contract(root: Path, before: dict[str, Any]) -> dict[str, Any]:
    scene_data = (root / SCENE).read_bytes()
    scene = scene_data.decode("utf-8").replace("\r\n", "\n")
    blocks = prefab_blocks(scene)
    world = blocks["100100000"]
    presentation = blocks["100200000"]
    player = blocks[PLAYER_INSTANCE]
    robot = blocks[ROBOT_INSTANCE]
    original = before["manual_objects"]
    original_player = original["Player"]["world_position"]
    original_robot = original["MOA_robot"]["world_position"]
    original_offset = before["robot_minus_player_position_xyz"]

    require(transform_parent(player) == PLAYER_ROOT_TRANSFORM, "Player is not parented to gameplay root")
    require(transform_parent(robot) == ROBOT_ANCHOR_TRANSFORM, "MOA_robot is not parented to interaction anchor")
    require(vector(player, PLAYER_GUID, "-8679921383154817045", "m_LocalPosition") == [0.0, 0.0, 0.0], "Player local position mismatch")
    require(vector(robot, ROBOT_GUID, "-8679921383154817045", "m_LocalPosition") == [0.0, 0.0, 0.0], "Robot local position mismatch")
    player_root_position = vector(world, WORLD_GUID, "400001", "m_LocalPosition")
    robot_anchor_position = vector(world, WORLD_GUID, "400002", "m_LocalPosition")
    require(player_root_position == original_player, "Player gameplay root did not preserve manual world position")
    require(robot_anchor_position == original_robot, "Robot anchor did not preserve manual world position")
    offsets = [
        float_mod(presentation, PRESENTATION_GUID, "11400003", property_name)
        for property_name in ("sideOffset", "heightOffset", "forwardOffset")
    ]
    require(all(abs(offsets[i] - original_offset[i]) <= 1e-7 for i in range(3)), "companion offset mismatch")

    require(scene.count(f"--- !u!4 &{PLAYER_STRIPPED_TRANSFORM} stripped\n") == 1, "Player stripped Transform missing")
    require(scene.count(f"m_PrefabInstance: {{fileID: {PLAYER_INSTANCE}}}\n") >= 1, "Player stripped pointer missing")
    require(scene.count(f"--- !u!4 &{ROBOT_STRIPPED_TRANSFORM} stripped\n") == 1, "Robot stripped Transform missing")
    require(scene.count(f"m_PrefabInstance: {{fileID: {ROBOT_INSTANCE}}}\n") >= 1, "Robot stripped pointer missing")
    require(scene.count(f"addedObject: {{fileID: {PLAYER_STRIPPED_TRANSFORM}}}") == 1, "Player added-object link missing")
    require(scene.count(f"addedObject: {{fileID: {ROBOT_STRIPPED_TRANSFORM}}}") == 1, "Robot added-object link missing")

    scene_roots = re.search(r"--- !u!1660057539 &9223372036854775807\nSceneRoots:\n(?P<body>.*)\Z", scene, re.DOTALL)
    require(scene_roots is not None, "SceneRoots missing")
    roots = scene_roots.group("body")
    require(f"{{fileID: {PLAYER_INSTANCE}}}" not in roots and f"{{fileID: {ROBOT_INSTANCE}}}" not in roots, "nested visuals remain Scene roots")

    require(modification_value(robot, ROBOT_GUID, "9500000", "m_Controller") == "", "Robot controller value field changed")
    controller_ref = (
        f"target: {{fileID: 9500000, guid: {ROBOT_GUID}, type: 3}}\n"
        "      propertyPath: m_Controller\n"
        "      value:\n"
        f"      objectReference: {{fileID: 9100000, guid: {ROBOT_CONTROLLER_GUID}, type: 2}}"
    )
    require(scene.count(controller_ref) == 1, "Robot Animator Controller link mismatch")
    root_motion = (
        f"target: {{fileID: 9500000, guid: {ROBOT_GUID}, type: 3}}\n"
        "      propertyPath: m_ApplyRootMotion\n"
        "      value: 0"
    )
    require(scene.count(root_motion) == 1, "Robot Apply Root Motion false mismatch")

    require(scene.count("value: Player\n") == 1 and scene.count("value: MOA_robot\n") == 1, "actual visual uniqueness mismatch")
    require("PetRobotBody" not in scene and "PetRobotHead" not in scene, "primitive robot visuals serialized")
    require("PortfolioPlayerVisual" not in scene and "PortfolioPetRobotVisual" not in scene, "auto-generated visual names serialized")
    placeholder = normalized(root / PLACEHOLDER_SOURCE)
    visual = normalized(root / VISUAL_FACTORY)
    companion = normalized(root / COMPANION_SOURCE)
    require(placeholder.count("if (!enabled) return;") == 1, "v1.3 Awake guard mismatch")
    require(visual.count("anchor.childCount != 0") == 1, "visual factory guard mismatch")
    require("Mathf.Sin" not in companion and "hover" not in companion.lower() and "bob" not in companion.lower(), "procedural hover source detected")

    return {
        "scene_after": record(SCENE, scene_data),
        "hierarchy": {
            "PetRobotPortfolioWorldBase/PlayerPlaceholder/Player": True,
            "PetRobotPortfolioWorldBase/PetRobotInteractionAnchor/MOA_robot": True,
        },
        "manual_world_transform_preserved": {
            "Player_position": player_root_position,
            "MOA_robot_position": robot_anchor_position,
            "MOA_robot_rotation_quaternion_xyzw": original["MOA_robot"]["world_rotation_quaternion_xyzw"],
            "MOA_robot_scale": original["MOA_robot"]["world_scale"],
        },
        "companion_offsets": {"sideOffset": offsets[0], "heightOffset": offsets[1], "forwardOffset": offsets[2]},
        "placeholder_component_disabled_override_count": scene.count(
            "- target: {fileID: 11400002, guid: 10b00000000000000000000000000010, type: 3}\n"
            "      propertyPath: m_Enabled\n      value: 0"
        ),
        "awake_guard_count": placeholder.count("if (!enabled) return;"),
        "visual_factory_child_guard_count": visual.count("anchor.childCount != 0"),
        "actual_player_count": scene.count("value: Player\n"),
        "actual_robot_count": scene.count("value: MOA_robot\n"),
        "robot_animator_controller_link_count": scene.count(controller_ref),
        "robot_apply_root_motion_false_count": scene.count(root_motion),
        "procedural_hover_present": False,
        "status": "PASS",
    }


def capture_before(root: Path) -> None:
    candidates = find_candidate_scenes(root)
    require(candidates == [SCENE], "saved target Scene candidate mismatch: " + repr(candidates))
    scene_data = (root / SCENE).read_bytes()
    require(len(scene_data) == BASELINE_SCENE_SIZE and sha(scene_data) == BASELINE_SCENE_SHA,
            "manual Scene baseline identity mismatch")
    inventory = scene_before_inventory(root, scene_data)
    base = root / "Validation" / NAME / "preflight"
    require(not base.exists(), "preflight already exists")
    base.mkdir(parents=True)
    write_new(base / "scene_before.unity", scene_data)
    write_new(base / "scene_object_inventory_before.json", jbytes(inventory))
    write_new(base / "approved_project_before.json", jbytes(fixed_identities(root)))
    write_new(base / "gameassets_checkpoint_before.json", jbytes(checkpoint_compare(root)))
    write_new(base / "git_staged_before.json", jbytes(git_staged(root)))
    summary = {
        "schema_version": SCHEMA + ".preflight",
        "scene": record(SCENE, scene_data),
        "player_position": inventory["manual_objects"]["Player"]["world_position"],
        "robot_position": inventory["manual_objects"]["MOA_robot"]["world_position"],
        "robot_minus_player_position_xyz": inventory["robot_minus_player_position_xyz"],
        "project_files_modified_by_preflight": 0,
        "status": "PASS_READY_FOR_SCENE_WIRING",
    }
    write_new(base / "preflight_before_snapshot.json", jbytes(summary))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def verify_wiring(root: Path) -> None:
    base = root / "Validation" / NAME
    preflight = base / "preflight"
    require(preflight.is_dir(), "preflight missing")
    before_scene = (preflight / "scene_before.unity").read_bytes()
    before = json.loads((preflight / "scene_object_inventory_before.json").read_text(encoding="utf-8"))
    after_scene = (root / SCENE).read_bytes()
    contract = static_wiring_contract(root, before)
    static = base / "static_after"
    require(not static.exists(), "static-after evidence already exists")
    static.mkdir(parents=True)
    write_new(static / "scene_after.unity", after_scene)
    diff = "\n".join(difflib.unified_diff(
        before_scene.decode("utf-8").splitlines(),
        after_scene.decode("utf-8").splitlines(),
        fromfile="scene_before.unity", tofile="scene_after.unity", lineterm="",
    )) + "\n"
    write_new(static / "scene_wiring_unified_diff.patch", diff.encode("utf-8"))
    write_new(static / "scene_wiring_contract.json", jbytes(contract))
    write_new(static / "approved_project_after_wiring.json", jbytes(fixed_identities(root)))
    write_new(static / "gameassets_checkpoint_after_wiring.json", jbytes(checkpoint_compare(root)))
    write_new(static / "test_inventory.json", jbytes(test_inventory(root)))
    write_new(static / "git_staged_after_wiring.json", jbytes(git_staged(root)))
    summary = {
        "schema_version": SCHEMA + ".static_after",
        "scene_before": record(SCENE, before_scene),
        "scene_after": record(SCENE, after_scene),
        "project_changed_file_count": 1,
        "project_changed_files": [SCENE],
        "companion_offsets": contract["companion_offsets"],
        "test_inventory": "152/152",
        "unity_gui": "UNVERIFIED",
        "webcam_live_capture": "NOT_APPLICABLE",
        "status": "PASS_READY_FOR_USER_UNITY_GUI_GATE",
    }
    write_new(static / "static_verification_summary.json", jbytes(summary))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def fixed_v13_delivery(root: Path) -> dict[str, Any]:
    path = root / V13_DELIVERY
    detached = Path(str(path) + ".sha256")
    blob = path.read_bytes()
    check = detached.read_bytes()
    require(len(blob) == V13_DELIVERY_SIZE and sha(blob) == V13_DELIVERY_SHA,
            "fixed v1.3 Delivery identity mismatch")
    require(len(check) == V13_DETACHED_SIZE and sha(check) == V13_DETACHED_SHA,
            "fixed v1.3 detached identity mismatch")
    words = check.decode("ascii").strip().split()
    require(words == [V13_DELIVERY_SHA, path.name], "fixed v1.3 detached syntax mismatch")
    return {
        "delivery": record(V13_DELIVERY, blob),
        "detached": record(V13_DELIVERY + ".sha256", check),
        "detached_digest_and_filename_match": True,
        "status": "PASS",
    }


def parse_targeted_xml(path: Path) -> dict[str, Any]:
    test_run = ET.parse(path).getroot()
    cases = test_run.findall(".//test-case")
    names = [case.attrib.get("fullname", case.attrib.get("name", "")) for case in cases]
    results = [case.attrib.get("result", "") for case in cases]
    prefix = "WordsInSilence.Tests.EditMode.PetRobotFloatingCompanionPresentationV1Tests."
    retry_repeat = []
    for element in test_run.iter():
        for key, value in element.attrib.items():
            if key.lower() in {"retry", "repeat", "retry-count", "repeat-count"} and value not in {"", "0"}:
                retry_repeat.append({"element": element.tag, "attribute": key, "value": value})
    summary = {
        "schema_version": SCHEMA + ".targeted_test_results",
        "platform": "EditMode",
        "fixture": "WordsInSilence.Tests.EditMode.PetRobotFloatingCompanionPresentationV1Tests",
        "test_run_id": test_run.attrib.get("id"),
        "start_time_utc": test_run.attrib.get("start-time"),
        "end_time_utc": test_run.attrib.get("end-time"),
        "duration_seconds": float(test_run.attrib.get("duration", "0")),
        "testcase_count": len(cases),
        "unique_testcase_count": len(set(names)),
        "passed": results.count("Passed"),
        "failed": results.count("Failed"),
        "skipped": results.count("Skipped"),
        "inconclusive": results.count("Inconclusive"),
        "retry_repeat_nonzero_count": len(retry_repeat),
        "retry_repeat_nonzero_rows": retry_repeat,
        "testcase_names": sorted(names),
    }
    summary["status"] = "PASS" if (
        len(cases) == 14 and len(set(names)) == 14 and all(name.startswith(prefix) for name in names)
        and summary["passed"] == 14 and summary["failed"] == 0 and summary["skipped"] == 0
        and summary["inconclusive"] == 0 and not retry_repeat
    ) else "FAIL"
    require(summary["status"] == "PASS", "targeted XML result mismatch")
    return summary


def analyze_editor_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    prebuild = text.count("Executing IPrebuildSetup for: Unity.PerformanceTesting.Editor.TestRunBuilder.")
    postbuild = text.count("Executing IPostBuildCleanup for: Unity.PerformanceTesting.Editor.TestRunBuilder.")
    automatic_save = len(re.findall(
        r"Saving results to: .*LocalLow[/\\]DefaultCompany[/\\]WordsInSilence[/\\]TestResults\.xml", text
    ))
    manual_export = text.count("Step10D_v1_4_Targeted14_TestResults.xml")
    compiler_errors = len(re.findall(r"\berror CS\d+\b|Compilation failed", text, flags=re.IGNORECASE))
    pptr = text.count("PPtr cast failed when dereferencing!")
    cast = text.count("Casting from PrefabInstance to Prefab at FileID 100100000!")
    scene_open = text.count("Opening scene 'Assets/_Project/Scenes/VerticalSlice/PetRobotPortfolioQuestVerticalSlice.unity'")
    playmode = text.count("Reloading assemblies for play mode.")
    exactly_once = prebuild == 1 and postbuild == 1 and automatic_save == 1
    summary = {
        "schema_version": SCHEMA + ".editor_log",
        "compiler_error_count": compiler_errors,
        "prebuild_cycle_count": prebuild,
        "automatic_result_save_count": automatic_save,
        "manual_xml_export_count": manual_export,
        "all_result_save_log_count": text.count("Saving results to:"),
        "postbuild_cycle_count": postbuild,
        "targeted_execution_exactly_once": exactly_once,
        "scene_open_count": scene_open,
        "reserved_pptr_error_count": pptr,
        "reserved_cast_error_count": cast,
        "playmode_execution_count": playmode,
        "status": "PASS" if (
            compiler_errors == 0 and exactly_once and manual_export == 1 and scene_open == 1
            and pptr == 0 and cast == 0 and playmode == 0
        ) else "FAIL",
    }
    require(summary["status"] == "PASS", "Editor log contract mismatch")
    return summary


def evidence_inventory(staging: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED_FAILURE_EVIDENCE if not (staging / name).is_file()]
    require(not missing, "required GUI evidence missing: " + repr(missing))
    rows = []
    for name in REQUIRED_FAILURE_EVIDENCE:
        data = (staging / name).read_bytes()
        rows.append(record(name, data))
    duplicate_hierarchy = sha((staging / "04_player_hierarchy.png").read_bytes()) == sha(
        (staging / "05_robot_hierarchy.png").read_bytes()
    )
    return {
        "required_count": len(REQUIRED_FAILURE_EVIDENCE),
        "present_count": len(rows),
        "records": rows,
        "compiler_error_zero_screen": {
            "status": "PASS_SHARED_SCREEN",
            "path": "02_test_runner_targeted_14.png",
            "detail": "Test Runner screen includes Console error badge 0; compiler result is cross-checked with the full Editor.log.",
        },
        "hierarchy_images_byte_identical": duplicate_hierarchy,
        "hierarchy_evidence_note": (
            "04 and 05 are the same full Hierarchy capture; 06 independently shows both Player and MOA_robot nested under their functional roots."
        ),
        "robot_animator_visual_review": {
            "path": "06_robot_animator_missing.png",
            "selected_object": "MOA_robot",
            "visible_components": ["Transform", "Mesh Filter", "Mesh Renderer"],
            "animator_component_visible": False,
            "classification": "ROBOT_ANIMATOR_COMPONENT_MISSING",
            "status": "FAIL_CONFIRMED",
        },
        "status": "PASS_EVIDENCE_COMPLETE_FOR_STOPPED_FAILURE_BRANCH",
    }


def transaction_commit(
    temporary_result: Path,
    temporary_zip: Path,
    temporary_detached: Path,
    final_result: Path,
    final_zip: Path,
    final_detached: Path,
    inject: str | None = None,
) -> None:
    finals = [final_result, final_zip, final_detached]
    require(not any(path.exists() for path in finals), "never-overwrite final target exists")
    created_finals: list[Path] = []
    committed = False
    try:
        if inject == "BEFORE_FINAL_RENAME":
            raise RuntimeError("injected failure: BEFORE_FINAL_RENAME")
        temporary_result.replace(final_result)
        created_finals.append(final_result)
        if inject == "AFTER_RESULT_RENAME":
            raise RuntimeError("injected failure: AFTER_RESULT_RENAME")
        temporary_zip.replace(final_zip)
        created_finals.append(final_zip)
        if inject == "AFTER_DELIVERY_RENAME":
            raise RuntimeError("injected failure: AFTER_DELIVERY_RENAME")
        temporary_detached.replace(final_detached)
        created_finals.append(final_detached)
        if inject == "AFTER_DETACHED_RENAME_BEFORE_COMMIT":
            raise RuntimeError("injected failure: AFTER_DETACHED_RENAME_BEFORE_COMMIT")
        committed = True
    finally:
        if not committed:
            for path in reversed(created_finals):
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()
            for path in (temporary_result, temporary_zip, temporary_detached):
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()


def transaction_regression() -> dict[str, Any]:
    points = [
        "BEFORE_FINAL_RENAME",
        "AFTER_RESULT_RENAME",
        "AFTER_DELIVERY_RENAME",
        "AFTER_DETACHED_RENAME_BEFORE_COMMIT",
    ]
    rows = []
    for point in points:
        with tempfile.TemporaryDirectory(prefix="step10d_v14_tx_") as temporary:
            base = Path(temporary)
            temp_result = base / "temp_result"
            temp_zip = base / "temp.zip.tmp"
            temp_detached = base / "temp.sha256.tmp"
            final_result = base / "final_result"
            final_zip = base / "final.zip"
            final_detached = base / "final.zip.sha256"
            sentinel = base / "preexisting_sentinel.txt"
            temp_result.mkdir()
            (temp_result / "payload.txt").write_text("payload", encoding="utf-8")
            temp_zip.write_bytes(b"zip")
            temp_detached.write_bytes(b"digest")
            sentinel.write_bytes(b"unchanged")
            observed = False
            try:
                transaction_commit(
                    temp_result, temp_zip, temp_detached,
                    final_result, final_zip, final_detached, inject=point,
                )
            except RuntimeError as error:
                observed = str(error) == "injected failure: " + point
            remaining_finals = sum(path.exists() for path in (final_result, final_zip, final_detached))
            remaining_temps = sum(path.exists() for path in (temp_result, temp_zip, temp_detached))
            sentinel_unchanged = sentinel.read_bytes() == b"unchanged"
            passed = observed and remaining_finals == 0 and remaining_temps == 0 and sentinel_unchanged
            require(passed, "transaction regression failed: " + point)
            rows.append({
                "injection_point": point,
                "expected_exception_observed": observed,
                "remaining_final_outputs": remaining_finals,
                "remaining_temporary_outputs": remaining_temps,
                "preexisting_sentinel_changed": not sentinel_unchanged,
                "status": "PASS",
            })
    return {"production_commit_function_used": True, "row_count": len(rows), "rows": rows, "status": "PASS"}


def never_overwrite_regression() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="step10d_v14_never_") as temporary:
        base = Path(temporary)
        temp_result = base / "temp_result"
        temp_zip = base / "temp.zip.tmp"
        temp_detached = base / "temp.sha.tmp"
        final_result = base / "final_result"
        final_zip = base / "final.zip"
        final_detached = base / "final.zip.sha256"
        temp_result.mkdir()
        (temp_result / "payload").write_bytes(b"payload")
        temp_zip.write_bytes(b"zip")
        temp_detached.write_bytes(b"checksum")
        final_zip.write_bytes(b"preexisting")
        refused = False
        try:
            transaction_commit(temp_result, temp_zip, temp_detached, final_result, final_zip, final_detached)
        except RuntimeError as error:
            refused = "never-overwrite" in str(error)
        sentinel_unchanged = final_zip.read_bytes() == b"preexisting"
        for path in (temp_result, temp_zip, temp_detached):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        require(refused and sentinel_unchanged and not final_result.exists() and not final_detached.exists(),
                "never-overwrite regression failed")
        return {
            "existing_final_refused": refused,
            "preexisting_sentinel_unchanged": sentinel_unchanged,
            "unexpected_final_outputs": 0,
            "status": "PASS",
        }


def deterministic_zip(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9, strict_timestamps=True
    ) as archive:
        for name in sorted(files):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.flag_bits = 0x800
            archive.writestr(info, files[name], compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return output.getvalue()


def validate_delivery(blob: bytes, files: dict[str, bytes]) -> dict[str, Any]:
    with zipfile.ZipFile(io.BytesIO(blob), "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        require(archive.testzip() is None, "Delivery CRC failure")
        require(len(names) == len(set(names)), "Delivery duplicate entry")
        require(set(names) == set(files), "Delivery membership mismatch")
        require(not any(name.startswith(("/", "\\")) or ".." in PurePosixPath(name).parts or "\\" in name for name in names),
                "Delivery unsafe path")
        require(not any(info.flag_bits & 1 for info in infos), "Delivery encryption detected")
        require(not any(((info.external_attr >> 16) & 0o170000) == 0o120000 for info in infos), "Delivery symlink detected")
        for name in names:
            require(archive.read(name) == files[name], "Delivery payload mismatch: " + name)
    manifest = json.loads(files["output_file_manifest.json"])
    rows = manifest["files"]
    require({row["relative_path"] for row in rows} == set(files) - {"output_file_manifest.json"},
            "manifest membership mismatch")
    for row in rows:
        data = files[row["relative_path"]]
        require(len(data) == row["size_bytes"] and sha(data) == row["sha256"],
                "manifest payload mismatch: " + row["relative_path"])
    return {
        "entry_count": len(files),
        "manifest_target_count": len(rows),
        "crc_duplicate_path_symlink_encryption_membership": "PASS",
        "manifest_size_sha256": "PASS",
        "deterministic_memory_recreation_byte_identical": deterministic_zip(files) == blob,
        "status": "PASS",
    }


def copy_tree_files(source: Path, prefix: str, files: dict[str, bytes]) -> None:
    for path in sorted(source.rglob("*")):
        if path.is_file():
            relative = path.relative_to(source).as_posix()
            files[f"{prefix}/{relative}"] = path.read_bytes()


def package_evidence(root: Path) -> None:
    validation = root / "Validation"
    staging = validation / NAME
    preflight = staging / "preflight"
    static = staging / "static_after"
    require(preflight.is_dir() and static.is_dir(), "preflight/static evidence missing")
    result = validation / RESULT_NAME
    delivery = validation / DELIVERY_NAME
    detached = Path(str(delivery) + ".sha256")
    require(not result.exists() and not delivery.exists() and not detached.exists(), "final output already exists")

    before = json.loads((preflight / "scene_object_inventory_before.json").read_text(encoding="utf-8"))
    contract = static_wiring_contract(root, before)
    xml_summary = parse_targeted_xml(staging / "Step10D_v1_4_Targeted14_TestResults.xml")
    log_summary = analyze_editor_log(staging / "Step10D_v1_4_Targeted14_Editor.log")
    evidence = evidence_inventory(staging)
    approved = fixed_identities(root)
    checkpoint = checkpoint_compare(root)
    staged = git_staged(root)
    inventory = test_inventory(root)
    fixed_v13 = fixed_v13_delivery(root)
    transaction = transaction_regression()
    never_overwrite = never_overwrite_regression()

    failure = {
        "schema_version": SCHEMA + ".failure_classification",
        "gate_stage": "SCENE_INSPECTOR_BEFORE_PLAYMODE",
        "classification": "ROBOT_ANIMATOR_COMPONENT_MISSING_ON_SCENE_PLACED_MOA_ROBOT",
        "learner_or_user_error": False,
        "scene_reference_integrity": "PASS",
        "reserved_pptr_error_recurrence_count": 0,
        "selected_object": "PetRobotPortfolioWorldBase/PetRobotInteractionAnchor/MOA_robot",
        "visible_components": ["Transform", "Mesh Filter", "Mesh Renderer"],
        "animator_component_visible": False,
        "manual_animator_addition_performed": False,
        "playmode_started": False,
        "project_fix_performed_after_failure": False,
        "status": "FAIL_CONFIRMED_STOPPED_PER_CONTRACT",
    }
    verdict = {
        "schema_version": SCHEMA + ".primary_verdict",
        "unity_version": "6000.3.14f1",
        "unity_compile": "PASS_COMPILER_ERROR_0",
        "targeted_editmode": "PASS_14_OF_14_EXACTLY_ONCE",
        "scene_reference_integrity": "PASS",
        "companion_presentation_gate": "FAIL_ROBOT_ANIMATOR_COMPONENT_MISSING_BEFORE_PLAYMODE",
        "playmode_execution_count": 0,
        "companion_follow": "NOT_APPLICABLE",
        "robot_animator_animation": "FAIL_NOT_AVAILABLE",
        "procedural_hover_absence": "PASS_STATIC_SOURCE",
        "prompt_follow": "NOT_APPLICABLE",
        "practice_entry": "NOT_APPLICABLE",
        "gameassets_retroactive_preimplementation_preservation": "UNVERIFIED_NONBLOCKING",
        "webcam_live_capture": "NOT_APPLICABLE",
        "overall": "FAIL",
        "state": [
            "STEP10D_V1_4_GUI_GATE_FAILURE_EVIDENCE_READY",
            "STEP10D_V1_4_TARGETED_EDITMODE_PASS_14_OF_14_EXACTLY_ONCE",
            "STEP10D_V1_4_SCENE_REFERENCE_INTEGRITY_PASS",
            "STEP10D_V1_4_COMPANION_PRESENTATION_BLOCKED_ROBOT_ANIMATOR_MISSING",
            "BLOCKED_PENDING_EXTERNAL_REVIEW",
        ],
    }

    assertions = [
        ("fixed_v13_delivery_identity", "PASS", fixed_v13["status"]),
        ("manual_scene_preflight_preserved", "PASS", before["status"]),
        ("scene_placed_visual_wiring_static_contract", "PASS", contract["status"]),
        ("approved_v13_source_test_identity", "PASS", approved["status"]),
        ("targeted_xml_14_unique_pass", "PASS", xml_summary["status"]),
        ("targeted_test_cycle_exactly_once", "PASS", str(log_summary["targeted_execution_exactly_once"])),
        ("unity_compiler_error_zero", "PASS", str(log_summary["compiler_error_count"])),
        ("scene_reference_integrity", "PASS", "PPtr/cast count 0"),
        ("scene_placed_robot_animator_available", "FAIL", "Animator absent in 06_robot_animator_missing.png"),
        ("playmode_execution", "NOT_APPLICABLE", "blocked before PlayMode"),
        ("companion_follow", "NOT_APPLICABLE", "PlayMode not entered"),
        ("robot_animation_playback", "NOT_APPLICABLE", "Animator absent; PlayMode not entered"),
        ("procedural_hover_absent", "PASS", "static source"),
        ("prompt_follow", "NOT_APPLICABLE", "PlayMode not entered"),
        ("practice_entry", "NOT_APPLICABLE", "PlayMode not entered"),
        ("gameassets_checkpoint_unchanged", "PASS", checkpoint["status"]),
        ("gameassets_retroactive_preservation", "UNVERIFIED_NONBLOCKING", "post-Step10D checkpoint only"),
        ("webcam_live_capture", "NOT_APPLICABLE", "out of scope"),
        ("static_test_inventory_152", "PASS", inventory["status"]),
        ("git_staged_zero", "PASS", staged["status"]),
        ("packaging_transaction_failure_injection", "PASS", transaction["status"]),
        ("never_overwrite", "PASS", never_overwrite["status"]),
        ("output_manifest", "PASS", "verified before commit"),
        ("deterministic_zip", "PASS", "memory recreation byte-identical before commit"),
    ]
    require(len({key for key, _, _ in assertions}) == len(assertions), "duplicate assertion key")
    assertion_rows = [
        {"assertion_key": key, "status": status, "detail": detail}
        for key, status, detail in assertions
    ]
    assertion_summary = {
        "total": len(assertion_rows),
        "pass": sum(row["status"] == "PASS" for row in assertion_rows),
        "fail": sum(row["status"] == "FAIL" for row in assertion_rows),
        "not_applicable": sum(row["status"] == "NOT_APPLICABLE" for row in assertion_rows),
        "unverified_nonblocking": sum(row["status"] == "UNVERIFIED_NONBLOCKING" for row in assertion_rows),
        "unique_keys": len({row["assertion_key"] for row in assertion_rows}),
        "status": "FAIL_EXPECTED_GATE_BRANCH",
    }

    files: dict[str, bytes] = {}
    for name in REQUIRED_FAILURE_EVIDENCE:
        files["unity_evidence/" + name] = (staging / name).read_bytes()
    files["unity_evidence/USER_GUI_GATE_INSTRUCTIONS.md"] = (staging / "USER_GUI_GATE_INSTRUCTIONS.md").read_bytes()
    copy_tree_files(preflight, "preflight", files)
    copy_tree_files(static, "static_after", files)
    files["project_files/after/PetRobotPortfolioQuestVerticalSlice.unity"] = (root / SCENE).read_bytes()
    files["project_files/approved/PetRobotWorldPlaceholderVisuals.cs"] = (root / PLACEHOLDER_SOURCE).read_bytes()
    files["project_files/approved/PetRobotFloatingCompanionPresentationV1Tests.cs"] = (root / TARGET_TEST).read_bytes()
    files["analysis/" + Path(__file__).name] = Path(__file__).read_bytes()
    reports = {
        "input_provenance.json": {"schema_version": SCHEMA + ".input_provenance", "fixed_v13": fixed_v13, "status": "PASS"},
        "unity_test_results_summary.json": xml_summary,
        "editor_log_analysis.json": log_summary,
        "gui_evidence_inventory.json": evidence,
        "companion_presentation_failure.json": failure,
        "primary_verdict.json": verdict,
        "approved_project_identity_after_gui.json": approved,
        "gameassets_post_step10d_checkpoint_after_gui.json": checkpoint,
        "test_inventory.json": inventory,
        "git_staged_after_gui.json": staged,
        "packaging_transaction_failure_injection_regression.json": transaction,
        "never_overwrite_regression.json": never_overwrite,
        "assertion_summary.json": assertion_summary,
    }
    for name, value in reports.items():
        files[name] = jbytes(value)
    files["ASSERTIONS.jsonl"] = b"".join(jbytes(row) for row in assertion_rows)
    report = (
        "# Step 10D v1.4 Scene-Placed Companion GUI Gate\n\n"
        "- Unity compile: **PASS** (compiler error 0).\n"
        "- Targeted EditMode: **PASS**, 14/14 exactly once.\n"
        "- Scene reference integrity: **PASS**; PPtr/FileID recurrence 0.\n"
        "- Scene wiring static contract: **PASS**; actual Player and MOA_robot are children of the functional roots.\n"
        "- Companion presentation: **FAIL** before PlayMode. The selected MOA_robot exposes Transform, Mesh Filter, and Mesh Renderer, but no Animator component.\n"
        "- User correctly did not add an Animator, did not save an ad-hoc fix, and did not enter PlayMode.\n"
        "- Companion follow, animated hover, prompt follow, and practice entry: **NOT_APPLICABLE**.\n"
        "- GameAssets comparison: post-Step10D checkpoint unchanged; retroactive preimplementation preservation remains **UNVERIFIED_NONBLOCKING**.\n"
        "- Webcam/live capture: **NOT_APPLICABLE**.\n"
        "- Overall gate: **FAIL**, fixed failure evidence ready for external review.\n"
    )
    files["VALIDATION_REPORT.md"] = report.encode("utf-8")
    files["packaging_verification.json"] = jbytes({
        "deterministic_zip": True,
        "manifest_self_excluded": True,
        "strict_detached_checksum": True,
        "transaction_rollback": transaction["status"],
        "never_overwrite": never_overwrite["status"],
        "status": "PASS",
    })

    manifest_rows = [record(name, data) for name, data in sorted(files.items())]
    files["output_file_manifest.json"] = jbytes({
        "schema_version": SCHEMA + ".output_manifest",
        "manifest_self_excluded": True,
        "target_count": len(manifest_rows),
        "files": manifest_rows,
        "status": "PASS",
    })

    blob = deterministic_zip(files)
    verification = validate_delivery(blob, files)
    require(verification["deterministic_memory_recreation_byte_identical"], "deterministic ZIP mismatch")
    detached_blob = f"{sha(blob)} {DELIVERY_NAME}\n".encode("ascii")

    temp_base = Path(tempfile.mkdtemp(prefix="." + NAME + "_", dir=validation))
    temp_result = temp_base / "result"
    temp_zip = temp_base / "delivery.zip"
    temp_detached = temp_base / "delivery.zip.sha256"
    try:
        temp_result.mkdir()
        for name, data in files.items():
            target = temp_result / Path(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        temp_zip.write_bytes(blob)
        temp_detached.write_bytes(detached_blob)
        transaction_commit(temp_result, temp_zip, temp_detached, result, delivery, detached)
    finally:
        if temp_base.exists():
            shutil.rmtree(temp_base)

    final_blob = delivery.read_bytes()
    require(final_blob == blob and deterministic_zip(files) == final_blob, "post-commit ZIP mismatch")
    require(detached.read_bytes() == detached_blob, "post-commit detached mismatch")
    summary = {
        "schema_version": SCHEMA + ".package_summary",
        "overall_gate": "FAIL_ROBOT_ANIMATOR_COMPONENT_MISSING_BEFORE_PLAYMODE",
        "result_directory": result.relative_to(root).as_posix(),
        "delivery": record(delivery.relative_to(root).as_posix(), final_blob),
        "detached": record(detached.relative_to(root).as_posix(), detached_blob),
        "entry_count": verification["entry_count"],
        "manifest_target_count": verification["manifest_target_count"],
        "assertions": assertion_summary,
        "status": "FAIL_EVIDENCE_PACKAGE_READY_PENDING_EXTERNAL_REVIEW",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--capture-before", action="store_true")
    parser.add_argument("--verify-wiring", action="store_true")
    parser.add_argument("--package-evidence", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    modes = [args.capture_before, args.verify_wiring, args.package_evidence]
    require(sum(bool(mode) for mode in modes) == 1, "choose exactly one mode")
    if args.capture_before:
        capture_before(root)
    elif args.verify_wiring:
        verify_wiring(root)
    else:
        package_evidence(root)


if __name__ == "__main__":
    main()
