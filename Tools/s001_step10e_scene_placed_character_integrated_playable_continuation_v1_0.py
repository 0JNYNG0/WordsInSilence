#!/usr/bin/env python3
"""Step 10E scene-placed character integrated playable validation helper.

This tool never starts Unity.  Preflight/static modes write only Validation
evidence; project edits are performed separately and minimally.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any


SCHEMA = "WordsInSilence.Step10E.ScenePlacedCharacterIntegratedPlayable.v1.0"
NAME = "WordsInSilence_step10e_scene_placed_character_integrated_playable_continuation_v1_0"

SCENE = "Assets/_Project/Scenes/VerticalSlice/PetRobotPortfolioQuestVerticalSlice.unity"
PLAYER_SOURCE = "Assets/_Project/Scripts/Practice/PetRobotWorldIntegration/PetRobotWorldPlayerController.cs"
PLAYER_SOURCE_META = PLAYER_SOURCE + ".meta"
PLACEHOLDER_SOURCE = "Assets/_Project/Scripts/Practice/PetRobotWorldIntegration/PetRobotWorldPlaceholderVisuals.cs"
VISUAL_FACTORY = "Assets/_Project/Scripts/Practice/PetRobotPortfolioQuest/PetRobotPortfolioQuestVisuals.cs"
COMPANION_SOURCE = "Assets/_Project/Scripts/Practice/PetRobotPortfolioQuest/PetRobotFloatingCompanionController.cs"
ENTRY_SOURCE = "Assets/_Project/Scripts/Practice/PetRobotWorldIntegration/PetRobotWorldEntryController.cs"
CAMERA_SOURCE = "Assets/_Project/Scripts/Practice/PetRobotWorldIntegration/PetRobotWorldCameraFollow.cs"
PLAYER_CONTROLLER = "Assets/GameAssets/Player_model_animation_meshy/Player.controller"
ROBOT_CONTROLLER = "Assets/GameAssets/petrobot_model_animation/Terrain.controller"

PLAYER_GUID = "f44a2ab6c6a52dd41b44e781d7f0c07a"
PLAYER_CONTROLLER_GUID = "5824cec5f72de9e459e864865536c381"
ROBOT_GUID = "50573fbff0bc00c418eb7319a40093d3"
ROBOT_CONTROLLER_GUID = "a414e66cfd19823419a0f9a1954db8d3"

BASELINE = {
    SCENE: (21702, "2bbc0a28973c8d3b4fb53d4d35e65dfb605c10d220b28ab1cc05be6e239535ff"),
    PLAYER_SOURCE: (1449, "bda4709941f5b0ba017679bd6aacfe8ec94ae64b718440ca3df4daf3244c68a2"),
    PLAYER_SOURCE_META: (240, "369fabeae24e74f64af1a7fb3f5b4f1dfb513932d00c04fc48d4235b05677251"),
    PLAYER_CONTROLLER: (4311, "2004ca8ca7ff56098562b433ce041b2b69c549aed85217b72c5987191c30ac8b"),
    ROBOT_CONTROLLER: (2055, "1c8421289d7612bcd59b3a9c022605ea78dc5284a23f287662a2b05e025f41ca"),
    PLACEHOLDER_SOURCE: (2032, "2f973da73f619add63bb5543eb0a9018640efe3af063de542d930427b5e29ab4"),
    VISUAL_FACTORY: (1906, "76d8df6b1a52354495d00cff26194c80e98e178a00cab706fd61fb1bb7465543"),
}

KEY_PROTECTED = [
    "Assets/_Project/Scripts/Recognition/Core/MaskAwareDtw.cs",
    "Assets/_Project/Scripts/Recognition/Core/RecognitionCore.cs",
    "Assets/_Project/Scripts/Recognition/Runtime/PersonalRuntimeOptInProfile.cs",
    "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PetRobotSignPracticeSession.cs",
    "Assets/_Project/Scripts/Practice/PetRobotWorldIntegration/PetRobotWorldIntegrationContracts.cs",
    "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PersonalizationCandidateRepository.cs",
    "ProjectSettings/EditorBuildSettings.asset",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rec(relative: str, data: bytes) -> dict[str, Any]:
    return {"relative_path": relative, "size_bytes": len(data), "sha256": sha(data)}


def jbytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def write_new(path: Path, data: bytes) -> None:
    require(not path.exists(), "never-overwrite target exists: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def read_text(root: Path, relative: str) -> str:
    return (root / relative).read_text(encoding="utf-8").replace("\r\n", "\n")


def identity(root: Path, relative: str) -> dict[str, Any]:
    data = (root / relative).read_bytes()
    return rec(relative, data)


def git_staged(root: Path) -> dict[str, Any]:
    run = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=root,
                         capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    require(run.returncode == 0, "git staged query failed")
    paths = [line for line in run.stdout.splitlines() if line.strip()]
    require(not paths, "Git staged files are nonzero")
    return {"staged_file_count": 0, "staged_paths": paths, "status": "PASS"}


def fixed_baseline(root: Path) -> dict[str, Any]:
    rows = []
    for relative, (size, digest) in BASELINE.items():
        data = (root / relative).read_bytes()
        match = len(data) == size and sha(data) == digest
        require(match, "Step 10E baseline mismatch: " + relative)
        rows.append({**rec(relative, data), "expected_size_bytes": size,
                     "expected_sha256": digest, "match": True, "status": "PASS"})
    return {"target_count": len(rows), "match_count": len(rows), "records": rows, "status": "PASS"}


def scene_contract(root: Path) -> dict[str, Any]:
    scene = read_text(root, SCENE)
    player_controller = read_text(root, PLAYER_CONTROLLER)
    robot_controller = read_text(root, ROBOT_CONTROLLER)
    checks = {
        "player_name_count": scene.count("value: Player\n"),
        "robot_name_count": scene.count("value: MOA_robot\n"),
        "player_parented_to_gameplay_root": "--- !u!1001 &1006113214" in scene and
            "m_TransformParent: {fileID: 100100010}" in scene,
        "robot_parented_to_interaction_anchor": "--- !u!1001 &1275187441" in scene and
            "m_TransformParent: {fileID: 100100011}" in scene,
        "offset_side_count": scene.count("propertyPath: sideOffset\n      value: 0.7029896"),
        "offset_height_count": scene.count("propertyPath: heightOffset\n      value: 1.63"),
        "offset_forward_count": scene.count("propertyPath: forwardOffset\n      value: -0.2899656"),
        "player_controller_reference_count": scene.count(
            "objectReference: {fileID: 9100000, guid: " + PLAYER_CONTROLLER_GUID + ", type: 2}"
        ),
        "player_root_motion_false_count": scene.count(
            "target: {fileID: 5866666021909216657, guid: " + PLAYER_GUID + ", type: 3}\n"
            "      propertyPath: m_ApplyRootMotion\n      value: 0"
        ),
        "robot_added_animator_count": scene.count("--- !u!95 &1275187443\nAnimator:"),
        "robot_avatar_null_count": scene.count("m_Avatar: {fileID: 0}"),
        "robot_controller_reference_count": scene.count(
            "m_Controller: {fileID: 9100000, guid: " + ROBOT_CONTROLLER_GUID + ", type: 2}"
        ),
        "robot_added_root_motion_false_count": scene.count(
            "--- !u!95 &1275187443\nAnimator:") and scene.count("  m_ApplyRootMotion: 0\n"),
        "placeholder_disabled_override_count": scene.count(
            "- target: {fileID: 11400002, guid: 10b00000000000000000000000000010, type: 3}\n"
            "      propertyPath: m_Enabled\n      value: 0"
        ),
        "player_isrun_bool_parameter_count": player_controller.count(
            "- m_Name: IsRun\n    m_Type: 4"
        ),
        "player_isrun_true_transition_count": player_controller.count(
            "m_ConditionMode: 1\n    m_ConditionEvent: IsRun"
        ),
        "player_isrun_false_transition_count": player_controller.count(
            "m_ConditionMode: 2\n    m_ConditionEvent: IsRun"
        ),
        "robot_default_clip_reference_count": robot_controller.count(
            "m_Motion: {fileID: 7400000, guid: 8a56dbc85c0c9b74381607d7d13f5e52, type: 2}"
        ),
        "serialized_primitive_or_duplicate_visual_names": sum(scene.count(token) for token in (
            "PetRobotBody", "PetRobotHead", "PortfolioPlayerVisual", "PortfolioPetRobotVisual"
        )),
    }
    require(checks["player_name_count"] == 1 and checks["robot_name_count"] == 1,
            "scene-placed visual count mismatch")
    require(checks["player_parented_to_gameplay_root"] and checks["robot_parented_to_interaction_anchor"],
            "scene hierarchy wiring mismatch")
    require(checks["offset_side_count"] == checks["offset_height_count"] == checks["offset_forward_count"] == 1,
            "companion offset mismatch")
    require(checks["player_controller_reference_count"] == 1 and checks["player_root_motion_false_count"] == 1,
            "Player Animator/controller contract mismatch")
    require(checks["robot_added_animator_count"] == 1 and checks["robot_controller_reference_count"] >= 1,
            "MOA_robot Animator/controller contract mismatch")
    require(checks["robot_added_root_motion_false_count"] >= 1, "MOA_robot root motion mismatch")
    require(checks["placeholder_disabled_override_count"] == 1, "placeholder disabled override mismatch")
    require(checks["player_isrun_bool_parameter_count"] == 1 and
            checks["player_isrun_true_transition_count"] == 1 and
            checks["player_isrun_false_transition_count"] == 1, "Player IsRun controller contract mismatch")
    require(checks["robot_default_clip_reference_count"] == 1, "Robot default animation mismatch")
    require(checks["serialized_primitive_or_duplicate_visual_names"] == 0, "serialized duplicate visual detected")
    return {
        "scene": identity(root, SCENE),
        "player": {
            "hierarchy": "PetRobotPortfolioWorldBase/PlayerPlaceholder/Player",
            "animator_controller_guid": PLAYER_CONTROLLER_GUID,
            "avatar": "preserved imported Humanoid avatar",
            "apply_root_motion": False,
            "run_parameter": "IsRun",
        },
        "robot": {
            "hierarchy": "PetRobotPortfolioWorldBase/PetRobotInteractionAnchor/MOA_robot",
            "added_animator_file_id": 1275187443,
            "animator_controller_guid": ROBOT_CONTROLLER_GUID,
            "avatar": None,
            "avatar_null_is_expected": True,
            "apply_root_motion": False,
        },
        "companion_offset": [0.7029896, 1.63, -0.2899656],
        "checks": checks,
        "status": "PASS_USER_AUTHORED_SCENE_BASELINE",
    }


def protected_snapshot(root: Path) -> dict[str, Any]:
    rows = []
    for relative in KEY_PROTECTED:
        path = root / relative
        if path.is_file():
            rows.append(identity(root, relative))
    require(len(rows) >= 5, "protected key file inventory unexpectedly small")
    return {"target_count": len(rows), "records": rows, "status": "PASS"}


def capture_before(root: Path) -> None:
    base = root / "Validation" / NAME / "preflight"
    require(not base.exists(), "Step 10E preflight already exists")
    base.mkdir(parents=True)
    fixed = fixed_baseline(root)
    scene = scene_contract(root)
    protected = protected_snapshot(root)
    write_new(base / "scene_before.unity", (root / SCENE).read_bytes())
    write_new(base / "player_controller_before.cs", (root / PLAYER_SOURCE).read_bytes())
    write_new(base / "fixed_baseline_identity.json", jbytes(fixed))
    write_new(base / "user_authored_scene_contract.json", jbytes(scene))
    write_new(base / "protected_key_files_before.json", jbytes(protected))
    write_new(base / "git_staged_before.json", jbytes(git_staged(root)))
    summary = {
        "schema_version": SCHEMA + ".preflight",
        "scene": scene["scene"],
        "player_source": identity(root, PLAYER_SOURCE),
        "player_controller_isrun": "PASS",
        "robot_avatar_null_expected": True,
        "project_files_changed_by_preflight": 0,
        "status": "PASS_READY_FOR_MINIMAL_ISRUNTIME_WIRING",
    }
    write_new(base / "preflight_summary.json", jbytes(summary))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def expected_player_source(before: bytes) -> bytes:
    text = before.decode("utf-8")
    text = text.replace(
        "        [SerializeField] float moveSpeed = 4f;\n        CharacterController characterController;\n",
        "        [SerializeField] float moveSpeed = 4f;\n"
        "        [SerializeField] Animator playerAnimator;\n"
        "        CharacterController characterController;\n",
        1,
    )
    text = text.replace(
        "            characterController = GetComponent<CharacterController>();\n",
        "            characterController = GetComponent<CharacterController>();\n"
        "            if (playerAnimator == null)\n"
        "                playerAnimator = GetComponentInChildren<Animator>(true);\n"
        "            SetRunning(false);\n",
        1,
    )
    text = text.replace(
        "            if (characterController == null) characterController = GetComponent<CharacterController>();\n"
        "            if (characterController == null || input.sqrMagnitude <= 0f) return;\n",
        "            if (characterController == null) characterController = GetComponent<CharacterController>();\n"
        "            bool isRunning = characterController != null && input.sqrMagnitude > 0f;\n"
        "            SetRunning(isRunning);\n"
        "            if (!isRunning) return;\n",
        1,
    )
    text = text.replace(
        "            transform.forward = direction;\n        }\n",
        "            transform.forward = direction;\n        }\n\n"
        "        void OnDisable()\n"
        "        {\n"
        "            SetRunning(false);\n"
        "        }\n\n"
        "        void SetRunning(bool value)\n"
        "        {\n"
        "            if (playerAnimator != null) playerAnimator.SetBool(\"IsRun\", value);\n"
        "        }\n",
        1,
    )
    return text.encode("utf-8")


def verify_static(root: Path) -> None:
    base = root / "Validation" / NAME
    preflight = base / "preflight"
    require(preflight.is_dir(), "Step 10E preflight missing")
    before_source = (preflight / "player_controller_before.cs").read_bytes()
    before_scene = (preflight / "scene_before.unity").read_bytes()
    current_source = (root / PLAYER_SOURCE).read_bytes()
    current_scene = (root / SCENE).read_bytes()
    require(current_source == expected_player_source(before_source), "Player controller delta is not exact")
    require(current_scene == before_scene, "user-authored Scene changed during Step 10E implementation")
    source = current_source.decode("utf-8")
    checks = {
        "serialized_player_animator_field": source.count("[SerializeField] Animator playerAnimator;") == 1,
        "child_animator_resolution": source.count("GetComponentInChildren<Animator>(true)") == 1,
        "isrun_setter": source.count('playerAnimator.SetBool("IsRun", value)') == 1,
        "moving_true_stationary_false": "bool isRunning = characterController != null && input.sqrMagnitude > 0f;" in source,
        "disable_resets_false": source.count("void OnDisable()") == 1 and source.count("SetRunning(false);") == 2,
        "recognition_or_capture_reference_count": sum(source.count(token) for token in (
            "RecognitionCore", "MaskAwareDtw", "MediaPipe", "WebCam", "CandidateRepository"
        )),
    }
    require(all(value is True for key, value in checks.items() if key != "recognition_or_capture_reference_count"),
            "minimal Animator wiring source contract mismatch")
    require(checks["recognition_or_capture_reference_count"] == 0, "Animator source coupled to recognition/capture")
    scene = scene_contract(root)
    protected_before = json.loads((preflight / "protected_key_files_before.json").read_text(encoding="utf-8"))
    protected_after = protected_snapshot(root)
    require(protected_before["records"] == protected_after["records"], "protected key file changed")
    require((root / PLAYER_SOURCE_META).read_bytes() == bytes((root / PLAYER_SOURCE_META).read_bytes()), "meta read failed")

    static = base / "static_after"
    require(not static.exists(), "static-after evidence already exists")
    static.mkdir(parents=True)
    write_new(static / "player_controller_after.cs", current_source)
    diff = "\n".join(difflib.unified_diff(
        before_source.decode("utf-8").splitlines(), current_source.decode("utf-8").splitlines(),
        fromfile="player_controller_before.cs", tofile="player_controller_after.cs", lineterm="",
    )) + "\n"
    write_new(static / "player_controller_unified_diff.patch", diff.encode("utf-8"))
    write_new(static / "minimal_isrun_wiring_contract.json", jbytes({"checks": checks, "status": "PASS"}))
    write_new(static / "user_authored_scene_after_static.json", jbytes(scene))
    write_new(static / "protected_key_files_after.json", jbytes(protected_after))
    write_new(static / "git_staged_after_static.json", jbytes(git_staged(root)))
    summary = {
        "schema_version": SCHEMA + ".static_after",
        "changed_project_file_count": 1,
        "changed_project_files": [PLAYER_SOURCE],
        "scene_before_after_byte_identical": True,
        "scene": identity(root, SCENE),
        "player_source_before": rec(PLAYER_SOURCE, before_source),
        "player_source_after": identity(root, PLAYER_SOURCE),
        "unity_compile": "UNVERIFIED_PENDING_USER_GUI",
        "integrated_playmode_smoke": "UNVERIFIED_PENDING_USER_GUI",
        "status": "PASS_READY_FOR_SINGLE_USER_INTEGRATED_PLAYMODE_GATE",
    }
    write_new(static / "static_summary.json", jbytes(summary))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def candidate_repository_snapshot() -> dict[str, Any]:
    local_app_data = Path(os.environ["LOCALAPPDATA"])
    repository = (
        local_app_data.parent / "LocalLow" / "DefaultCompany" / "WordsInSilence" /
        "WordsInSilence" / "PersonalizationCandidates" / "v1" / "local_learner"
    )
    require(repository.is_dir(), "candidate repository missing: " + str(repository))
    files = []
    for path in sorted(repository.rglob("*")):
        if path.is_file():
            data = path.read_bytes()
            files.append(rec(path.relative_to(repository).as_posix(), data))
    index_path = repository / "candidate_index.json"
    require(index_path.is_file(), "candidate index missing")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    entries = index.get("entries", [])
    require(len(entries) == 8 and len(files) == 25, "candidate baseline count mismatch")
    return {
        "repository": str(repository),
        "candidate_count": len(entries),
        "file_count": len(files),
        "index_entry_capture_ids": [row.get("capture_id") for row in entries],
        "files": files,
        "status": "PASS_BASELINE_8_CANDIDATES_25_FILES",
    }


def capture_gui_before(root: Path) -> None:
    gate = root / "Validation" / NAME / "gui_gate"
    before = gate / "preflight"
    require(not before.exists(), "GUI preflight already exists")
    before.mkdir(parents=True)
    candidate = candidate_repository_snapshot()
    project = {
        "scene": identity(root, SCENE),
        "player_source": identity(root, PLAYER_SOURCE),
        "player_source_meta": identity(root, PLAYER_SOURCE_META),
        "protected": protected_snapshot(root),
        "git_staged": git_staged(root),
        "status": "PASS",
    }
    write_new(before / "candidate_repository_before.json", jbytes(candidate))
    write_new(before / "project_before_gui.json", jbytes(project))
    summary = {
        "schema_version": SCHEMA + ".gui_preflight",
        "candidate_count_before": candidate["candidate_count"],
        "candidate_file_count_before": candidate["file_count"],
        "scene": project["scene"],
        "player_source": project["player_source"],
        "git_staged_file_count": 0,
        "status": "PASS_READY_FOR_SINGLE_INTEGRATED_PLAYMODE_SMOKE",
    }
    write_new(before / "gui_preflight_summary.json", jbytes(summary))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--capture-before", action="store_true")
    parser.add_argument("--verify-static", action="store_true")
    parser.add_argument("--capture-gui-before", action="store_true")
    args = parser.parse_args()
    require(sum((args.capture_before, args.verify_static, args.capture_gui_before)) == 1,
            "choose exactly one mode")
    root = args.root.resolve()
    if args.capture_before:
        capture_before(root)
    elif args.verify_static:
        verify_static(root)
    else:
        capture_gui_before(root)


if __name__ == "__main__":
    main()
