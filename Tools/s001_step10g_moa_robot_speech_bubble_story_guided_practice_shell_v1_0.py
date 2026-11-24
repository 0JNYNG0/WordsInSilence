from __future__ import annotations

import difflib
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


NAME = "WordsInSilence_step10g_moa_robot_speech_bubble_story_guided_practice_shell_v1_0"
CONTEXT = "WordsInSilence_PROJECT_CONTEXT.md"
CONTEXT_SHA256 = "d9d51612a8b2c8b9fff3d14b20bcd1dcdebf2f1a12a15004498c7e334a57373d"
SCENE = "Assets/_Project/Scenes/VerticalSlice/PetRobotSignPracticeVerticalSlice.unity"
SCENE_META = SCENE + ".meta"
SCENE_SHA256 = "5282810b9dd49194d0b175e54360ce98aab867e8e9065ec48a09abc74a213303"

RUNTIME = "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice"
PRESENTATION = RUNTIME + "/GuidedSignPracticePresentationController.cs"
STORY = RUNTIME + "/MoaStoryGuidedPracticeFlow.cs"
STORY_META = STORY + ".meta"
TARGETED = "Assets/_Project/Tests/EditMode/PrivacyEnhancedGuidedSignPracticePresentationV1Tests.cs"
EXISTING_CHANGED = (PRESENTATION, TARGETED)
NEW_PROJECT_FILES = (STORY, STORY_META)
ALLOWED_PROJECT_FILES = EXISTING_CHANGED + NEW_PROJECT_FILES
RELATED_EXISTING_METAS = tuple(path + ".meta" for path in EXISTING_CHANGED)

WORK = "Validation/.step10g_v1_0_moa_story_shell_work_v1_0"
BEFORE_DIR = WORK + "/source_before"
FROZEN_DIR = WORK + "/frozen_before"
PROBE_SOURCE = WORK + "/offline_story_flow_probe.cs"

V14_RESULT = "Validation/WordsInSilence_step10f_minimal_hud_composition_and_overlay_orientation_repair_v1_4"
V14_ZIP = V14_RESULT + "_delivery.zip"
V14_ZIP_SHA256 = "5a488eb52c9e31367be61f6792f07b57b835c4f56c1c827581c906e6d8efd864"
V14_GUI_RESULT = "Validation/WordsInSilence_step10f_v1_4_gui_visual_acceptance_gate_evidence_v1_0"
V14_GUI_ZIP = V14_GUI_RESULT + ".zip"
V14_GUI_ZIP_SHA256 = "f0e801b0daa4955fa4395e86befa76023d838ab2beefc9631cc0c4a3f3146907"
V14_GUI_IDENTITY = V14_GUI_RESULT + "/identity"
PUBLISH_TOOL = V14_RESULT + "/analysis/approved_publish_tool/s001_step10f_automatic_presentation_layout_refinement_v1_3b.py"
PUBLISH_TOOL_SHA256 = "6e1de14eea2736c220857538e73c2744d5e59ddeb395a100719fd0fd75943ca7"

EXISTING_BEFORE_SHA256 = {
    PRESENTATION: "4de317f8f91b67d8f9503231b32f9ddf074c7d34d8bfe55315d0a18e6c6102de",
    TARGETED: "341efa143d70300168d29ebb47775c9add5c7ba4c491b4a380f7b909caf23098",
}

NEW_STORY_TESTS = (
    "StoryFlow_StartsAtWelcomeWithoutTouchingSession",
    "StoryFlow_IntroRequiresExplicitAdvanceForEveryBeat",
    "StoryFlow_UsesGreetingLoveThankYouIntroductionOrder",
    "StoryFlow_DoesNotEnableExpressionBeforeWordIntroAccepted",
    "StoryFlow_CapturingMirrorsSessionWithoutAutoAdvance",
    "StoryFlow_StrictAndLowConfidenceReuseExistingNextWordRoute",
    "StoryFlow_WrongWordAndTechnicalFailureNeverAutoAdvance",
    "StoryFlow_NeedsPracticeReusesExistingContinueRoute",
    "StoryFlow_CompletedReusesExistingExitRoute",
    "StoryUi_HidesDuplicateLegacyProgressionAndPreservesCaptureActions",
)

KOREAN_COPY = (
    ("Welcome", "어서 와! 나는 MOA야. 이 연습 공간에서 네 수어를 함께 맞춰볼게."),
    ("LessonOverview", "오늘은 ‘인사’, ‘사랑’, ‘감사’ 세 표현을 함께 연습해 보자."),
    ("HowToUse", "왼쪽 시범을 보고 가운데 카메라에서 따라 하면 돼. 시스템이 동작을 인식해 결과를 알려줄게."),
    ("GREETING", "첫 번째는 ‘인사’야. 손과 팔의 흐름을 보고 천천히 따라 해보자."),
    ("LOVE", "두 번째는 ‘사랑’이야. 손 모양과 움직임이 이어지는 흐름을 기억해보자."),
    ("THANK_YOU", "마지막은 ‘감사’야. 시범을 본 뒤 네 동작으로 표현해봐."),
    ("Capturing", "좋아. 시작됐어. 동작을 끝까지 표현한 다음 ‘표현 완료’를 눌러줘."),
    ("StrictSuccess", "잘했어! 수어를 정확히 인식했어."),
    ("LowConfidenceCorrect", "뜻은 맞게 전달됐어. 다만 시스템 신뢰도가 낮았어. 결과는 그대로 두고 다음으로 갈 수 있어."),
    ("WrongWord", "다른 표현으로 인식했어. 시범을 한 번 더 보고 다시 해볼까?"),
    ("TechnicalFailure", "네 동작의 오답이 아니라 카메라나 추적 문제야. 준비되면 다시 시도해줘."),
    ("NeedsPractice", "이번 표현은 연습이 조금 더 필요해. 결과는 그대로 두고 다음 표현으로 이어갈 수 있어."),
    ("Completed", "좋아, 세 표현을 모두 연습했어. 이제 밖에서 직접 써볼 준비가 됐어."),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read(root: Path, relative: str) -> bytes:
    path = root / relative
    require(path.is_file(), "Missing file: " + relative)
    return path.read_bytes()


def text(root: Path, relative: str) -> str:
    return read(root, relative).decode("utf-8-sig")


def jbytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def json_file(root: Path, relative: str) -> dict:
    return json.loads(text(root, relative))


def record(relative: str, data: bytes) -> dict[str, object]:
    return {"relative_path": relative, "size_bytes": len(data), "sha256": sha(data)}


def root_for(script: Path) -> Path:
    for parent in script.parents:
        if (parent / "Assets").is_dir() and (parent / CONTEXT).is_file():
            return parent
    raise RuntimeError("Project root resolution failed")


def staged(root: Path) -> list[str]:
    run = subprocess.run(
        ["git", "diff", "--cached", "--name-only"], cwd=root,
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
    )
    require(run.returncode == 0, "Git staged audit failed")
    return [line.strip() for line in run.stdout.splitlines() if line.strip()]


def unity_process_count() -> int:
    run = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "@(Get-Process -Name Unity -ErrorAction SilentlyContinue).Count"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
    )
    require(run.returncode == 0, "Unity process audit failed")
    return int(run.stdout.strip() or "0")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "Module load failed: " + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_publish_tool(root: Path):
    data = read(root, PUBLISH_TOOL)
    require(sha(data) == PUBLISH_TOOL_SHA256, "Approved production publish tool mismatch")
    module = load_module(root / PUBLISH_TOOL, "wis_step10g_publish")
    module.NAME = NAME
    return module, data


def before(root: Path, relative: str) -> bytes:
    return read(root, BEFORE_DIR + "/" + Path(relative).name)


def identity_against_records(root: Path, approved: list[dict], base: Path | None = None) -> dict:
    base = root if base is None else base
    rows = []
    for expected in approved:
        relative = str(expected["relative_path"])
        path = base / relative
        current = record(relative, path.read_bytes()) if path.is_file() else {
            "relative_path": relative, "size_bytes": None, "sha256": None,
        }
        match = (current["size_bytes"] == expected["size_bytes"] and
                 current["sha256"] == expected["sha256"])
        rows.append({"relative_path": relative, "approved": expected,
                     "current": current, "match": match})
    mismatches = [row["relative_path"] for row in rows if not row["match"]]
    return {"record_count": len(rows), "mismatch_count": len(mismatches),
            "mismatches": mismatches, "records": rows}


def extract_test_methods(source: str) -> tuple[list[str], dict[str, str]]:
    matches = list(re.finditer(
        r"(?m)^\s*\[Test\]\s*\r?\n\s*public void ([A-Za-z0-9_]+)\s*\(", source
    ))
    names = [match.group(1) for match in matches]
    bodies: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = source.find("public void", match.start())
        end = matches[index + 1].start() if index + 1 < len(matches) else source.find(
            "        static void AssertForbiddenAcrossPresentationSources", start
        )
        require(start >= 0 and end > start, "Test body span not found: " + names[index])
        bodies[names[index]] = source[start:end].rstrip()
    return names, bodies


def method_segment(source: str, signature: str, next_signature: str) -> str:
    start = source.find(signature)
    end = source.find(next_signature, start + len(signature))
    require(start >= 0 and end > start, "Method segment missing: " + signature)
    return source[start:end]


def source_identity_and_diff(root: Path) -> tuple[dict, bytes, dict[str, bytes]]:
    approved_project = json_file(root, V14_GUI_IDENTITY + "/project_identity_after.json")
    baseline_rows = []
    allowed_baseline_mismatches = {PRESENTATION, TARGETED}
    for expected in approved_project["records"]:
        relative = expected["relative_path"]
        current = record(relative, read(root, relative))
        match = (current["size_bytes"] == expected["size_bytes"] and
                 current["sha256"] == expected["sha256"])
        baseline_rows.append({"relative_path": relative, "approved": expected,
                              "current": current, "match": match,
                              "authorized_mismatch": (not match and relative in allowed_baseline_mismatches)})
    unexpected = [row["relative_path"] for row in baseline_rows
                  if not row["match"] and not row["authorized_mismatch"]]
    require(not unexpected, "Unauthorized v1.4 baseline mismatch: " + repr(unexpected))

    rows = []
    before_files: dict[str, bytes] = {}
    patch_lines: list[str] = []
    for relative in EXISTING_CHANGED:
        old = before(root, relative)
        current = read(root, relative)
        require(sha(old) == EXISTING_BEFORE_SHA256[relative],
                "Captured v1.4 before identity mismatch: " + relative)
        require(old != current, "Expected changed source is unchanged: " + relative)
        before_files[relative] = old
        rows.append({"relative_path": relative, "before": record(relative, old),
                     "after": record(relative, current), "changed": True,
                     "created": False})
        patch_lines.extend(difflib.unified_diff(
            old.decode("utf-8-sig").splitlines(), current.decode("utf-8-sig").splitlines(),
            fromfile="a/" + relative, tofile="b/" + relative, lineterm="",
        ))
    for relative in NEW_PROJECT_FILES:
        current = read(root, relative)
        rows.append({"relative_path": relative,
                     "before": {"relative_path": relative, "exists": False},
                     "after": record(relative, current), "changed": True, "created": True})
        patch_lines.extend(difflib.unified_diff(
            [], current.decode("utf-8-sig").splitlines(),
            fromfile="/dev/null", tofile="b/" + relative, lineterm="",
        ))

    meta_rows = []
    for relative in RELATED_EXISTING_METAS:
        old = before(root, relative)
        current = read(root, relative)
        meta_rows.append({"relative_path": relative, "before": record(relative, old),
                          "after": record(relative, current), "match": old == current})
    require(all(row["match"] for row in meta_rows), "Existing related .meta changed")
    story_meta = text(root, STORY_META)
    guid_match = re.search(r"(?m)^guid: ([0-9a-f]{32})$", story_meta)
    require(guid_match is not None, "New story .meta GUID missing")
    guid = guid_match.group(1)
    guid_count = 0
    for path in (root / "Assets").rglob("*.meta"):
        if ("guid: " + guid) in path.read_text(encoding="utf-8-sig", errors="replace"):
            guid_count += 1
    require(guid_count == 1, "New story GUID is not unique")
    return ({
        "schema_version": "wordsinsilence.step10g.v1.0.source_identity",
        "changed_project_file_count": len(rows),
        "changed_project_files": list(ALLOWED_PROJECT_FILES),
        "records": rows,
        "existing_related_meta_mismatch_count": sum(not row["match"] for row in meta_rows),
        "existing_related_meta_records": meta_rows,
        "new_story_meta_guid": guid,
        "new_story_meta_guid_occurrence_count": guid_count,
        "approved_v1_4_project_record_count": len(baseline_rows),
        "approved_v1_4_unchanged_record_count": sum(row["match"] for row in baseline_rows),
        "authorized_v1_4_mismatch_count": sum(row["authorized_mismatch"] for row in baseline_rows),
        "unapproved_v1_4_mismatch_count": len(unexpected),
        "approved_v1_4_comparison": baseline_rows,
        "status": "FOUR_AUTHORIZED_PROJECT_FILES_CHANGED_EXISTING_METAS_UNCHANGED",
    }, ("\n".join(patch_lines) + "\n").encode("utf-8"), before_files)


def targeted_inventory(root: Path, before_files: dict[str, bytes]) -> dict:
    before_names, before_bodies = extract_test_methods(before_files[TARGETED].decode("utf-8-sig"))
    after_names, after_bodies = extract_test_methods(text(root, TARGETED))
    require(len(before_names) == 28 and len(set(before_names)) == 28,
            "Approved targeted baseline is not 28 unique tests")
    require(len(after_names) == 38 and len(set(after_names)) == 38,
            "Expanded targeted fixture is not 38 unique tests")
    require(after_names[:28] == before_names, "Existing 28 test name order changed")
    require(tuple(after_names[28:]) == NEW_STORY_TESTS, "Story test inventory mismatch")
    existing_rows = []
    for name in before_names:
        old = before_bodies[name].encode("utf-8")
        current = after_bodies[name].encode("utf-8")
        existing_rows.append({"name": name, "before_sha256": sha(old),
                              "after_sha256": sha(current), "body_unchanged": old == current})
    require(all(row["body_unchanged"] for row in existing_rows),
            "An existing targeted test body changed")
    story_rows = [{"name": name, "body_sha256": sha(after_bodies[name].encode("utf-8"))}
                  for name in after_names[28:]]
    return {
        "schema_version": "wordsinsilence.step10g.v1.0.targeted_inventory",
        "fixture": "WordsInSilence.Tests.EditMode.PrivacyEnhancedGuidedSignPracticePresentationV1Tests",
        "before_count": 28, "after_count": 38,
        "before_unique_count": 28, "after_unique_count": 38,
        "existing_name_order_preserved": True,
        "existing_28_body_mismatch_count": 0,
        "existing_tests": existing_rows,
        "new_story_test_count": len(story_rows),
        "new_story_tests": story_rows,
        "before_names": before_names, "after_names": after_names,
        "before_fullnames": ["WordsInSilence.Tests.EditMode.PrivacyEnhancedGuidedSignPracticePresentationV1Tests." + name for name in before_names],
        "after_fullnames": ["WordsInSilence.Tests.EditMode.PrivacyEnhancedGuidedSignPracticePresentationV1Tests." + name for name in after_names],
        "execution_performed": False,
        "status": "EXISTING_28_NAMES_AND_BODIES_EXACTLY_PRESERVED_PLUS_10_STORY_TESTS",
    }


def scene_inventory(root: Path) -> tuple[dict, dict[str, bytes]]:
    old = read(root, FROZEN_DIR + "/PetRobotSignPracticeVerticalSlice.unity")
    old_meta = read(root, FROZEN_DIR + "/PetRobotSignPracticeVerticalSlice.unity.meta")
    current = read(root, SCENE)
    current_meta = read(root, SCENE_META)
    require(sha(old) == SCENE_SHA256 and current == old and current_meta == old_meta,
            "Current authored Scene changed")
    scene_text = current.decode("utf-8-sig")
    assets = (
        ("Assets/ScifiOfficeLite/Prefabs/Doors/Broken Door.prefab", "8b2ed08f82f4cea41907e2cf2f8938c5"),
        ("Assets/VOiD1 Gaming - Free Hologram Shader for Unity URP/Material/Hologram.mat", "990a06ffeda43734e864d545330e132e"),
        ("Assets/ScifiOfficeLite/Prefabs/Shelf with Crates.prefab", "f355da60777d10946920a8b633e6b316"),
        ("Assets/ScifiOfficeLite/Meshes/Materials/DemoSkybox.mat", "32ea4fce7231ca349a48b944302d257e"),
    )
    asset_rows = []
    for path, guid in assets:
        require(guid in scene_text, "Authored Scene reference missing: " + path)
        meta = read(root, path + ".meta")
        require("guid: " + guid in meta.decode("utf-8-sig"), "Authored asset GUID mismatch")
        asset_rows.append({"asset": record(path, read(root, path)),
                           "meta": record(path + ".meta", meta), "guid": guid})
    return ({
        "schema_version": "wordsinsilence.step10g.v1.0.authored_scene_preservation",
        "scene_relative_path": SCENE,
        "before": record(SCENE, old), "after": record(SCENE, current),
        "scene_byte_identical": True,
        "meta_before": record(SCENE_META, old_meta),
        "meta_after": record(SCENE_META, current_meta),
        "meta_byte_identical": True,
        "hierarchy_inventory": [
            {"hierarchy_path": "/Broken Door", "prefab_guid": assets[0][1],
             "hologram_material_override_guid": assets[1][1]},
            {"hierarchy_path": "/Shelf with Crates", "prefab_guid": assets[2][1]},
        ],
        "background_inventory": {"dark_scifi_skybox_guid": assets[3][1]},
        "referenced_asset_records": asset_rows,
        "scene_yaml_modified": False,
        "status": "CURRENT_AUTHORED_SCENE_BROKEN_DOOR_HOLOGRAM_DARK_ENVIRONMENT_PRESERVED",
    }, {
        "authored_scene/current/" + SCENE: current,
        "authored_scene/current/" + SCENE_META: current_meta,
    })


def transition_and_copy_contract(root: Path) -> tuple[dict, dict, dict]:
    story = text(root, STORY)
    presentation = text(root, PRESENTATION)
    for _, copy in KOREAN_COPY:
        require(copy in story, "Korean story copy missing: " + copy)
    transitions = [
        {"from": "Welcome", "trigger": "explicit bubble click: 다음", "to": "LessonOverview", "automatic": False},
        {"from": "LessonOverview", "trigger": "explicit bubble click: 다음", "to": "HowToUse", "automatic": False},
        {"from": "HowToUse", "trigger": "explicit bubble click: 첫 표현 보기", "to": "WordIntroduction/GREETING", "automatic": False},
        {"from": "WordIntroduction", "trigger": "explicit bubble click: 연습하기", "guard": "authoritative Ready and CameraReady", "to": "Practice", "automatic": False},
        {"from": "Practice", "trigger": "existing HandleStartExpression", "to": "Capturing display", "automatic": False},
        {"from": "Capturing", "trigger": "existing HandleFinishExpression", "to": "Feedback display", "automatic": False},
        {"from": "StrictSuccess/LowConfidenceCorrect", "trigger": "explicit bubble click: 다음 표현/결과 보기", "to": "existing HandleNextWord", "automatic": False},
        {"from": "WrongWord/TechnicalFailure", "trigger": "no bubble progression; existing Retry/camera route only", "to": "same word", "automatic": False},
        {"from": "NeedsPractice", "trigger": "explicit bubble click: 다음 표현/결과 보기", "to": "existing HandleNeedsPracticeContinue", "automatic": False},
        {"from": "next authoritative word", "trigger": "session Changed display sync", "to": "WordIntroduction once", "automatic": "display-only"},
        {"from": "Completed", "trigger": "explicit bubble click: 연습 공간 나가기", "to": "existing HandleExitPractice", "automatic": False},
    ]
    copy_table = [{"state": state, "text": copy} for state, copy in KOREAN_COPY]
    dispatcher = method_segment(presentation, "void HandleStoryAction()", "void ToggleDiagnostics()")
    bind = method_segment(presentation, "public void Bind(", "public void ShowInitializationFailure")
    changed = method_segment(presentation, "void OnSessionChanged()", "void BuildUguiHierarchy()")
    retry = method_segment(presentation, "void HandleRetry()", "void HandleNeedsPracticeContinue()")
    reuse = {
        "schema_version": "wordsinsilence.step10g.v1.0.handler_reuse",
        "routes": [
            {"bubble_action": "ContinueToNextWord", "controller_handler": "HandleNextWord", "authoritative_call": "session.ContinueToNextWord()"},
            {"bubble_action": "ContinueWithNeedsPractice", "controller_handler": "HandleNeedsPracticeContinue", "authoritative_call": "session.ContinueWithNeedsPractice()"},
            {"bubble_action": "ExitPractice", "controller_handler": "HandleExitPractice", "authoritative_call": "session.ExitPractice()"},
            {"bottom_action": "표현 시작", "controller_handler": "HandleStartExpression", "authoritative_call": "session.StartExpression()"},
            {"bottom_action": "표현 완료", "controller_handler": "HandleFinishExpression", "authoritative_call": "session.FinishExpression()"},
            {"bottom_action": "다시 시도", "controller_handler": "HandleRetry", "authoritative_call": "session.Retry()"},
        ],
        "story_reset_count_in_bind": bind.count("storyFlow.Reset("),
        "story_reset_count_in_changed": changed.count("storyFlow.Reset("),
        "story_reset_count_in_retry": retry.count("storyFlow.Reset("),
        "dispatcher": {
            "next_handler_reused": "HandleNextWord();" in dispatcher,
            "needs_practice_handler_reused": "HandleNeedsPracticeContinue();" in dispatcher,
            "exit_handler_reused": "HandleExitPractice();" in dispatcher,
        },
        "story_source_session_object_reference_count": story.count("PetRobotSignPracticeSession"),
        "story_source_direct_session_call_count": story.count("session."),
        "status": "EXISTING_AUTHORITATIVE_SESSION_HANDLERS_REUSED",
    }
    require(reuse["story_reset_count_in_bind"] == 1 and
            reuse["story_reset_count_in_changed"] == 0 and
            reuse["story_reset_count_in_retry"] == 0 and
            all(reuse["dispatcher"].values()) and
            reuse["story_source_session_object_reference_count"] == 0 and
            reuse["story_source_direct_session_call_count"] == 0,
            "Story handler reuse/reset contract failed")
    return ({"schema_version": "wordsinsilence.step10g.v1.0.transition_table",
             "transitions": transitions,
             "explicit_user_action_transition_count": 9,
             "automatic_story_progression_count": 0,
             "status": "EXPLICIT_USER_ACTION_ONLY_STORY_PROGRESSION"},
            {"schema_version": "wordsinsilence.step10g.v1.0.korean_copy",
             "copy_count": len(copy_table), "copy": copy_table,
             "status": "EXACT_KOREAN_STORY_COPY_RECORDED"}, reuse)


def v14_preservation(root: Path) -> dict:
    approved = json_file(root, V14_GUI_IDENTITY + "/project_identity_after.json")
    allowed = {PRESENTATION, TARGETED}
    rows = []
    for expected in approved["records"]:
        relative = expected["relative_path"]
        current = record(relative, read(root, relative))
        match = (current["size_bytes"] == expected["size_bytes"] and
                 current["sha256"] == expected["sha256"])
        rows.append({"relative_path": relative, "approved": expected, "current": current,
                     "match": match, "authorized_step10g_change": relative in allowed})
    mismatches = [row["relative_path"] for row in rows
                  if not row["match"] and not row["authorized_step10g_change"]]
    require(not mismatches, "v1.4 technical/presentation preservation mismatch")
    presentation = text(root, PRESENTATION)
    required = (
        "CompactPracticeStatusHeader", "RoundedWebcamPanel", "RoundedProgressAndControlsHud",
        "Sprite.Create(", "ReleaseRoundedGraphicResources();",
        "session.State == PetRobotPracticeState.Ready && session.CameraReady",
    )
    require(all(token in presentation for token in required), "v1.4 HUD contract token missing")
    return {
        "schema_version": "wordsinsilence.step10g.v1.0.v1_4_preservation",
        "approved_record_count": len(rows), "records": rows,
        "authorized_changed_records": sorted(allowed),
        "unauthorized_mismatch_count": len(mismatches), "unauthorized_mismatches": mismatches,
        "compact_hud_and_capture_gate_tokens_preserved": True,
        "overlay_robot_prefab_material_scene_records_unchanged": True,
        "status": "STEP10F_V1_4_TECHNICAL_AND_VISUAL_CONTRACT_PRESERVED",
    }


def protected_candidate_core(root: Path) -> tuple[dict, dict, dict]:
    protected_approved = json_file(root, V14_GUI_IDENTITY + "/protected_identity_after.json")
    protected_records = [row["approved"] for row in protected_approved["records"]]
    protected = identity_against_records(root, protected_records)
    require(protected["record_count"] == 36 and protected["mismatch_count"] == 0,
            "Protected 36 mismatch")
    protected.update({"schema_version": "wordsinsilence.step10g.v1.0.protected_identity",
                      "status": "PROTECTED_36_OF_36_BYTE_IDENTITY_PRESERVED"})

    candidate_approved = json_file(root, V14_GUI_IDENTITY + "/candidate_identity_after.json")
    candidate_root = Path(candidate_approved["repository_path"])
    candidate_records = [row["approved"] for row in candidate_approved["records"]]
    candidate = identity_against_records(root, candidate_records, candidate_root)
    files = sorted(path for path in candidate_root.rglob("*") if path.is_file())
    ids = set()
    for path in files:
        parts = path.relative_to(candidate_root).parts
        if len(parts) >= 3 and parts[0] == "candidates" and parts[1] in ("GREETING", "LOVE", "THANK_YOU"):
            ids.add((parts[1], parts[2]))
    media_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".png", ".jpg", ".jpeg", ".bmp", ".wav", ".mp3", ".flac", ".ogg"}
    media = [path.relative_to(candidate_root).as_posix() for path in files
             if path.suffix.lower() in media_extensions]
    require(candidate["record_count"] == 25 and candidate["mismatch_count"] == 0 and
            len(files) == 25 and len(ids) == 8 and not media, "Candidate repository mismatch")
    candidate.update({"schema_version": "wordsinsilence.step10g.v1.0.candidate_identity",
                      "repository_path": str(candidate_root), "candidate_count": len(ids),
                      "current_file_count": len(files), "media_file_count": len(media),
                      "media_files": media, "capture_or_write_performed": False,
                      "status": "CANDIDATE_8_25_BYTE_IDENTITY_PRESERVED_MEDIA_ZERO"})

    approved_core = json_file(root, V14_RESULT + "/core_152_testcase_preservation.json")
    rows = []
    names = []
    for row in approved_core["fixture_records"]:
        relative = row["relative_path"]
        current = read(root, relative)
        expected = row["approved"]
        match = len(current) == expected["size_bytes"] and sha(current) == expected["sha256"]
        rows.append({"relative_path": relative, "approved": expected,
                     "current": record(relative, current), "source_match": match,
                     "test_count": row["test_count"], "test_names": row["test_names"]})
        names.extend(row["test_names"])
    require(all(row["source_match"] for row in rows) and len(names) == 152 and len(set(names)) == 152,
            "Core 152 source/name mismatch")
    core = {"schema_version": "wordsinsilence.step10g.v1.0.core_preservation",
            "fixture_count": len(rows), "fixture_records": rows,
            "testcase_count": len(names), "unique_testcase_name_count": len(set(names)),
            "mismatch_count": 0, "execution_performed": False,
            "status": "CORE_152_SOURCE_AND_NAME_IDENTITY_PRESERVED_NOT_EXECUTED"}
    return protected, candidate, core


def offline_compile(root: Path) -> tuple[dict, dict[str, bytes]]:
    editor = Path(r"C:\Program Files\Unity\Hub\Editor\6000.3.14f1\Editor")
    mono = editor / "Data/MonoBleedingEdge/bin/mono.exe"
    csc = editor / "Data/MonoBleedingEdge/lib/mono/4.5/csc.exe"
    modules = editor / "Data/Managed/UnityEngine"
    references = [modules / name for name in (
        "UnityEngine.CoreModule.dll", "UnityEngine.AnimationModule.dll",
        "UnityEngine.DirectorModule.dll", "UnityEngine.UIModule.dll",
        "UnityEngine.TextRenderingModule.dll", "UnityEngine.AudioModule.dll",
        "UnityEngine.InputLegacyModule.dll", "UnityEngine.PhysicsModule.dll",
    )]
    references.extend((
        root / "Library/ScriptAssemblies/UnityEngine.UI.dll",
        root / "Library/ScriptAssemblies/WordsInSilence.Runtime.dll",
        modules / "UnityEditor.CoreModule.dll",
        editor / "Data/MonoBleedingEdge/lib/mono/4.5/Facades/netstandard.dll",
    ))
    nunit_candidates = sorted((root / "Library/PackageCache").glob("**/nunit.framework.dll"))
    require(mono.is_file() and csc.is_file() and nunit_candidates and
            all(path.is_file() for path in references), "Offline compiler/reference inventory missing")
    nunit = nunit_candidates[0]
    temporary = Path(tempfile.mkdtemp(prefix="WIS-Step10G-v1_0-compile-"))
    try:
        product = temporary / "Step10G_v1_0.dll"
        production_sources = (
            RUNTIME + "/SignDemonstrationAvatarPresenter.cs",
            RUNTIME + "/WebcamLandmarkOverlayPresenter.cs",
            RUNTIME + "/WebcamTrackingGraphic.cs",
            STORY, PRESENTATION,
            RUNTIME + "/PetRobotPracticeVerticalSliceBootstrap.cs",
        )
        command = [str(mono), str(csc), "/nologo", "/target:library",
                   "/langversion:latest", "/out:" + str(product)]
        command.extend("/r:" + str(path) for path in references)
        command.extend(str((root / source).resolve()) for source in production_sources)
        run = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=120)
        production_log = (run.stdout or "") + (run.stderr or "")
        require(run.returncode == 0 and not re.search(r"error CS\d+", production_log),
                "Offline production compile failed:\n" + production_log[-12000:])

        aliases = "\n".join((
            "using SignDemonstrationAvatarPresenter = step10g::WordsInSilence.Practice.PetRobotVerticalSlice.SignDemonstrationAvatarPresenter;",
            "using GuidedSignPracticePresentationController = step10g::WordsInSilence.Practice.PetRobotVerticalSlice.GuidedSignPracticePresentationController;",
            "using MoaStoryGuidedPracticeFlow = step10g::WordsInSilence.Practice.PetRobotVerticalSlice.MoaStoryGuidedPracticeFlow;",
            "using MoaStoryBeat = step10g::WordsInSilence.Practice.PetRobotVerticalSlice.MoaStoryBeat;",
            "using MoaStoryFeedbackKind = step10g::WordsInSilence.Practice.PetRobotVerticalSlice.MoaStoryFeedbackKind;",
            "using MoaStoryAction = step10g::WordsInSilence.Practice.PetRobotVerticalSlice.MoaStoryAction;",
            "using PetRobotPracticeState = WordsInSilence.Practice.PetRobotVerticalSlice.PetRobotPracticeState;",
        ))
        alias_source = text(root, TARGETED).replace(
            "using WordsInSilence.Practice.PetRobotVerticalSlice;", aliases
        )
        alias_source = "extern alias step10g;\n" + alias_source
        test_path = temporary / "PrivacyEnhancedGuidedSignPracticePresentationV1Tests.alias.cs"
        test_path.write_text(alias_source, encoding="utf-8", newline="\n")
        test_dll = temporary / "Step10G_v1_0_Tests.dll"
        command = [str(mono), str(csc), "/nologo", "/target:library",
                   "/langversion:latest", "/out:" + str(test_dll)]
        command.extend("/r:" + str(path) for path in references)
        command.extend(("/r:" + str(nunit), "/reference:step10g=" + str(product), str(test_path)))
        run = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=120)
        targeted_log = (run.stdout or "") + (run.stderr or "")
        require(run.returncode == 0 and not re.search(r"error CS\d+", targeted_log),
                "Offline targeted compile failed:\n" + targeted_log[-12000:])

        probe_exe = temporary / "Step10G_StoryFlowProbe.exe"
        command = [str(mono), str(csc), "/nologo", "/target:exe", "/langversion:latest",
                   "/out:" + str(probe_exe), str((root / STORY).resolve()),
                   str((root / PROBE_SOURCE).resolve())]
        run = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=120)
        probe_compile_log = (run.stdout or "") + (run.stderr or "")
        require(run.returncode == 0 and not re.search(r"error CS\d+", probe_compile_log),
                "Offline story probe compile failed:\n" + probe_compile_log[-12000:])
        run = subprocess.run([str(mono), str(probe_exe)], capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=120)
        probe_run_log = (run.stdout or "") + (run.stderr or "")
        require(run.returncode == 0 and "OFFLINE_STORY_FLOW_PROBE_PASS" in probe_run_log,
                "Offline story probe failed:\n" + probe_run_log[-12000:])

        if not production_log:
            production_log = "Offline production compile succeeded with no compiler diagnostics.\n"
        if not targeted_log:
            targeted_log = "Offline expanded targeted fixture compile succeeded with no compiler diagnostics.\n"
        if not probe_compile_log:
            probe_compile_log = "Offline pure story probe compile succeeded with no compiler diagnostics.\n"
        return ({
            "schema_version": "wordsinsilence.step10g.v1.0.offline_compile",
            "compiler": "Unity 6000.3.14f1 Mono Roslyn csc",
            "production_source_count": len(production_sources),
            "targeted_fixture_source_count": 1,
            "pure_story_probe_source_count": 2,
            "production_error_count": 0, "targeted_fixture_error_count": 0,
            "pure_story_probe_error_count": 0,
            "production_warning_count": len(re.findall(r"warning CS\d+", production_log)),
            "targeted_fixture_warning_count": len(re.findall(r"warning CS\d+", targeted_log)),
            "pure_story_probe_warning_count": len(re.findall(r"warning CS\d+", probe_compile_log)),
            "pure_story_probe_exit_code": 0,
            "unity_editor_compile_claimed": False,
            "editmode_execution_claimed": False,
            "status": "OFFLINE_CSHARP_AND_PURE_STORY_PROBE_SUCCEEDED_UNITY_UNVERIFIED",
        }, {
            "offline_compile/production.txt": production_log.encode("utf-8"),
            "offline_compile/targeted_fixture.txt": targeted_log.encode("utf-8"),
            "offline_compile/story_probe_compile.txt": probe_compile_log.encode("utf-8"),
            "offline_compile/story_probe_run.txt": probe_run_log.encode("utf-8"),
            "offline_compile/story_flow_probe.cs": read(root, PROBE_SOURCE),
        })
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def static_validation(root: Path, source: dict, targeted: dict, scene: dict,
                      transition: dict, reuse: dict, preserved: dict) -> dict:
    story = text(root, STORY)
    presentation = text(root, PRESENTATION)
    tests = text(root, TARGETED)
    forbidden_story = (
        "WordsInSilence.Recognition", "RecognitionCore", "MaskAwareDtw",
        "PersonalizationCandidateRepository", "CandidateSave", "SaveCandidate(",
        "SetExpectedWord(", "BeginCapture(", "FinishExpression(", "session.",
    )
    checks = {
        "four_authorized_project_files_changed": source["changed_project_file_count"] == 4,
        "existing_related_metas_unchanged": source["existing_related_meta_mismatch_count"] == 0,
        "new_story_meta_unique": source["new_story_meta_guid_occurrence_count"] == 1,
        "existing_28_test_names_and_bodies_preserved": targeted["existing_28_body_mismatch_count"] == 0,
        "ten_story_tests_added_same_fixture": targeted["new_story_test_count"] == 10 and targeted["after_count"] == 38,
        "pure_story_has_no_session_object_or_direct_call": reuse["story_source_session_object_reference_count"] == 0 and reuse["story_source_direct_session_call_count"] == 0,
        "recognition_dtw_candidate_storage_forbidden_tokens_absent": all(token not in story for token in forbidden_story),
        "strict_outcomes_only_mapped_for_presentation": "new GuidedQuestOutcome" not in presentation and "GuidedQuestOutcomeEvaluator" not in presentation,
        "explicit_onboarding_advance_only": transition["automatic_story_progression_count"] == 0 and presentation.count("storyFlow.AdvanceStory()") == 1,
        "story_reset_only_on_bind": reuse["story_reset_count_in_bind"] == 1 and reuse["story_reset_count_in_changed"] == 0 and reuse["story_reset_count_in_retry"] == 0,
        "existing_continuation_exit_handlers_reused": all(reuse["dispatcher"].values()),
        "word_intro_start_gate_present": "storyFlow.ExpressionStartUnlocked" in presentation and "session.State == PetRobotPracticeState.Ready && session.CameraReady" in presentation,
        "legacy_progression_hidden": all(token in presentation for token in (
            "needsPracticeButton.gameObject.SetActive(false);",
            "nextButton.gameObject.SetActive(false);", "exitButton.gameObject.SetActive(false);")),
        "capture_actions_preserved": all(token in presentation for token in (
            "session.StartExpression()", "session.FinishExpression()", "session.Retry()")),
        "compact_bubble_contract": all(token in presentation for token in (
            "MoaRobotSpeechBubble", "MoaRobotNameLabel", "MoaStoryActionButton")),
        "v1_4_contract_preserved": preserved["unauthorized_mismatch_count"] == 0,
        "scene_byte_identical": scene["scene_byte_identical"] and scene["meta_byte_identical"],
        "story_tests_cover_required_contracts": all(name in tests for name in NEW_STORY_TESTS),
        "unity_gui_not_started": unity_process_count() == 0,
        "test_runner_editmode_playmode_webcam_not_executed": True,
    }
    failed = [name for name, value in checks.items() if not value]
    require(not failed, "Static Step 10G contract failed: " + repr(failed))
    return {"schema_version": "wordsinsilence.step10g.v1.0.static_validation",
            "checks": checks, "unity_gui_started": False,
            "test_runner_executed": False, "editmode_executed": False,
            "playmode_executed": False, "webcam_executed": False,
            "live_recognition_executed": False, "candidate_write_performed": False,
            "status": "OFFLINE_STATIC_CONTRACTS_READY_PENDING_EXTERNAL_REVIEW"}


def provenance(root: Path) -> dict:
    source_zip = read(root, V14_ZIP)
    gui_zip = read(root, V14_GUI_ZIP)
    require(sha(source_zip) == V14_ZIP_SHA256 and sha(gui_zip) == V14_GUI_ZIP_SHA256,
            "Approved v1.4 provenance ZIP mismatch")
    return {
        "schema_version": "wordsinsilence.step10g.v1.0.approved_v1_4_provenance",
        "step10f_v1_4_source_delivery": record(V14_ZIP, source_zip),
        "step10f_v1_4_gui_visual_acceptance": record(V14_GUI_ZIP, gui_zip),
        "large_gui_zip_replicated": False,
        "gui_provenance_hash_and_path_only": True,
        "approved_facts": {
            "targeted": "28/28 exactly once", "single_playmode": True,
            "compact_hud": True, "skeleton_rgb_alignment": True,
            "greeting_robot_yaw_motion": True, "runtime_blocking_error_count": 0,
        },
        "status": "STEP10F_V1_4_SOURCE_AND_GUI_ACCEPTANCE_PROVENANCE_VERIFIED",
    }


def validation_report(source: dict, targeted: dict, compile_doc: dict,
                      protected: dict, candidate: dict, core: dict) -> bytes:
    lines = [
        "# WordsInSilence Step 10G v1.0 source validation report", "",
        "- Scope: presentation-only MOA speech-bubble story-guided practice shell.",
        "- Project delta: exactly 4 authorized files (new pure flow + new .meta + presentation controller + targeted fixture); existing related .meta mismatch 0.",
        "- Story: Welcome → lesson overview → usage → word introduction requires explicit bubble input; camera/session Ready gates word-introduction acceptance.",
        "- Session authority: capture/retry/next/NeedsPractice/exit reuse the existing controller handlers and session methods; story source has no session object, recognition, DTW, candidate repository, or storage calls.",
        "- Targeted fixture: existing 28 names and bodies preserved exactly; 10 identifiable story regressions added, for 38 unique tests total. No Unity tests were executed.",
        "- Offline compile: production errors 0, expanded fixture errors 0, pure story behavior probe exit 0. This is not a Unity compile/EditMode claim.",
        "- Authored Scene remains byte-identical at SHA-256 `" + SCENE_SHA256 + "`; /Broken Door, hologram material, and dark environment are retained.",
        "- Protected: " + str(protected["record_count"]) + "/36. Candidate: " +
        str(candidate["candidate_count"]) + " candidates / " + str(candidate["current_file_count"]) +
        " files / media 0. Core: " + str(core["testcase_count"]) + "/152 source/name preserved.",
        "- Unity GUI/Test Runner/EditMode/PlayMode/webcam/live recognition/candidate save were not executed.",
        "- Git stage/commit/push were not performed; staged count remained zero.", "",
        "## Completion status", "",
        "STEP10G_V1_0_MOA_ROBOT_SPEECH_BUBBLE_STORY_GUIDED_PRACTICE_SHELL_READY_PENDING_EXTERNAL_REVIEW",
        "STEP10G_V1_0_SOURCE_AND_DELIVERY_READY_PENDING_EXTERNAL_REVIEW",
        "STEP10G_V1_0_UNITY_COMPILE_EDITMODE_PLAYMODE_UNVERIFIED_SOURCE_STAGE",
        "STEP10G_V1_0_LIVE_RECOGNITION_NOT_APPLICABLE_SOURCE_STAGE",
        "STEP10F_PRESENTATION_COMPLETE",
        "STEP10E_PLAYER_RUN_ANIMATION_DEFECT_PENDING_SEPARATE_MINIMAL_REPAIR",
        "BLOCKED_PENDING_CHATGPT_WORK_EXTERNAL_REVIEW", "",
    ]
    return "\n".join(lines).encode("utf-8")


def build(root: Path, script: Path, publish_tool, publish_tool_data: bytes) -> dict[str, bytes]:
    context_before = read(root, CONTEXT)
    require(sha(context_before) == CONTEXT_SHA256, "Project Context changed before packaging")
    require(unity_process_count() == 0, "Unity process exists during source packaging")
    staged_before = staged(root)
    require(not staged_before, "Git staged state is not zero")

    source, patch, before_files = source_identity_and_diff(root)
    targeted = targeted_inventory(root, before_files)
    scene, scene_files = scene_inventory(root)
    transitions, copy_table, reuse = transition_and_copy_contract(root)
    preserved = v14_preservation(root)
    protected, candidate, core = protected_candidate_core(root)
    compile_doc, compile_logs = offline_compile(root)
    static = static_validation(root, source, targeted, scene, transitions, reuse, preserved)
    approved_provenance = provenance(root)

    rollback, never = publish_tool.transaction_regression(root / "Validation")
    rollback.update({"schema_version": "wordsinsilence.step10g.v1.0.production_publish_transaction",
                     "approved_publish_tool_sha256": PUBLISH_TOOL_SHA256,
                     "status": "ACTUAL_PRODUCTION_PUBLISH_FAILURE_INJECTION_READY_PENDING_EXTERNAL_REVIEW"})
    never.update({"schema_version": "wordsinsilence.step10g.v1.0.never_overwrite",
                  "approved_publish_tool_sha256": PUBLISH_TOOL_SHA256,
                  "status": "NEVER_OVERWRITE_READY_PENDING_EXTERNAL_REVIEW"})

    context_after = read(root, CONTEXT)
    scene_after = read(root, SCENE)
    scene_meta_after = read(root, SCENE_META)
    staged_after = staged(root)
    unity_after = unity_process_count()
    require(context_after == context_before, "Project Context changed during packaging")
    require(scene_after == read(root, FROZEN_DIR + "/PetRobotSignPracticeVerticalSlice.unity") and
            scene_meta_after == read(root, FROZEN_DIR + "/PetRobotSignPracticeVerticalSlice.unity.meta"),
            "Authored Scene changed during packaging")
    require(not staged_after and unity_after == 0, "Staged state or Unity process changed")

    files: dict[str, bytes] = {
        "analysis/" + script.name: script.read_bytes(),
        "analysis/approved_publish_tool/" + Path(PUBLISH_TOOL).name: publish_tool_data,
        "approved_step10f_v1_4_provenance.json": jbytes(approved_provenance),
        "current_authored_scene_preservation.json": jbytes(scene),
        "four_file_before_after_identity.json": jbytes(source),
        "unified_diff.patch": patch,
        "story_beat_action_transition_table.json": jbytes(transitions),
        "exact_korean_story_copy_table.json": jbytes(copy_table),
        "existing_session_handler_reuse_graph.json": jbytes(reuse),
        "targeted_28_plus_10_story_test_inventory.json": jbytes(targeted),
        "offline_compile_report.json": jbytes(compile_doc),
        "static_validation_v1_0.json": jbytes(static),
        "step10f_v1_4_presentation_contract_preservation.json": jbytes(preserved),
        "protected_identity.json": jbytes(protected),
        "candidate_identity.json": jbytes(candidate),
        "core_152_testcase_preservation.json": jbytes(core),
        "project_context_read_only_identity.json": jbytes({
            **record(CONTEXT, context_before), "before_sha256": sha(context_before),
            "after_sha256": sha(context_after), "byte_identical": context_before == context_after,
            "status": "PROJECT_CONTEXT_UNCHANGED",
        }),
        "git_staged_and_unity_process_identity.json": jbytes({
            "git_staged_before": staged_before, "git_staged_after": staged_after,
            "git_staged_before_count": len(staged_before), "git_staged_after_count": len(staged_after),
            "unity_process_before_count": 0, "unity_process_after_count": unity_after,
            "git_stage_commit_push_performed": False,
            "status": "GIT_STAGED_ZERO_AND_UNITY_NOT_RUNNING_PRESERVED",
        }),
        "production_publish_transaction_evidence.json": jbytes(rollback),
        "never_overwrite_evidence.json": jbytes(never),
        "packaging_verification.json": jbytes({
            "schema_version": "wordsinsilence.step10g.v1.0.packaging",
            "deterministic_zip_build_count": 2, "deterministic_zip_bytes_match": True,
            "sorted_membership": True, "fixed_zip_timestamp": "1980-01-01T00:00:00",
            "deflate_level": 9, "large_gui_evidence_zip_replicated": False,
            "gui_provenance_hash_and_path_only": True,
            "actual_publish_failure_injection_case_count": rollback["failure_case_count"],
            "normal_publish_exercised_by_main": True, "never_overwrite_exercised": True,
            "status": "DETERMINISTIC_PACKAGING_READY_PENDING_EXTERNAL_REVIEW",
        }),
        "validation_report.md": validation_report(source, targeted, compile_doc,
                                                   protected, candidate, core),
    }
    files.update(compile_logs)
    files.update(scene_files)
    for relative in EXISTING_CHANGED:
        files["project_files/before/" + relative] = before_files[relative]
    for relative in ALLOWED_PROJECT_FILES:
        files["project_files/current/" + relative] = read(root, relative)
    for relative in RELATED_EXISTING_METAS:
        files["project_files/current/" + relative] = read(root, relative)
    files["output_file_manifest.json"] = jbytes({
        "schema_version": "wordsinsilence.step10g.v1.0.output_manifest",
        "self_excluded": True, "file_count_excluding_self": len(files),
        "files": [record(name, files[name]) for name in sorted(files)],
    })
    return files


def main() -> int:
    script = Path(__file__).resolve()
    root = root_for(script)
    publish_tool, publish_tool_data = load_publish_tool(root)
    files = build(root, script, publish_tool, publish_tool_data)
    output_dir, output_zip, output_sha, digest = publish_tool.publish(root, files)
    require(unity_process_count() == 0 and not staged(root),
            "Post-publish Unity/Git preservation failed")
    require(read(root, SCENE) == read(root, FROZEN_DIR + "/PetRobotSignPracticeVerticalSlice.unity"),
            "Post-publish Scene preservation failed")
    print(json.dumps({
        "output_directory": str(output_dir), "delivery_zip": str(output_zip),
        "detached_checksum": str(output_sha),
        "delivery_zip_size_bytes": output_zip.stat().st_size,
        "delivery_zip_sha256": digest, "entry_count": len(files),
        "project_changed_file_count": len(ALLOWED_PROJECT_FILES),
        "targeted_existing_count": 28, "targeted_story_count": 10,
        "targeted_total_count": 38, "core_testcase_count": 152,
        "protected_count": 36, "candidate_count": 8, "candidate_file_count": 25,
        "media_file_count": 0, "git_staged_count": 0, "unity_process_count": 0,
        "status": "STEP10G_V1_0_SOURCE_AND_DELIVERY_READY_PENDING_EXTERNAL_REVIEW",
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
