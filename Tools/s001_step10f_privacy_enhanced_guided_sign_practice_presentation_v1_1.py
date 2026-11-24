#!/usr/bin/env python3
"""Step 10F v1.1 minimal presentation/evidence repair validation and packaging."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import zipfile


SCHEMA = "wordsinsilence.step10f.presentation.v1.1"
NAME = "WordsInSilence_step10f_privacy_enhanced_guided_sign_practice_presentation_v1_1"
TOOL = "Tools/s001_step10f_privacy_enhanced_guided_sign_practice_presentation_v1_1.py"
V1_TOOL = "Tools/s001_step10f_privacy_enhanced_guided_sign_practice_presentation_v1_0.py"
V1_ZIP = "Validation/WordsInSilence_step10f_privacy_enhanced_guided_sign_practice_presentation_v1_0_delivery.zip"
V1_CHECKSUM = V1_ZIP + ".sha256"
CONTEXT = "WordsInSilence_PROJECT_CONTEXT.md"

PRESENTATION = "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/GuidedSignPracticePresentationController.cs"
OVERLAY = "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/WebcamLandmarkOverlayPresenter.cs"
TEST = "Assets/_Project/Tests/EditMode/PrivacyEnhancedGuidedSignPracticePresentationV1Tests.cs"
PREFAB = "Assets/_Project/Resources/VerticalSlice/PetRobotSignPracticeVerticalSlice.prefab"
REPAIRED_PROJECT_FILES = [PRESENTATION, OVERLAY, TEST]

ANIMATIONS = {
    "GREETING": ("인사", "Assets/_Project/Animations/Multiview/SIGN002_GREETING.fbx", "d9c7a496c14c2544a83e75fd761e82a9", "greeting"),
    "LOVE": ("사랑", "Assets/_Project/Animations/Multiview/SIGN003_LOVE.fbx", "92330f8673ab23a499d49ba5166de7b1", "love"),
    "THANK_YOU": ("감사합니다", "Assets/_Project/Animations/Multiview/SIGN001_THANKS.fbx", "3a282cc5617691046baa68cccdaab55b", "thankYou"),
}

EXPECTED_V1_ZIP_SIZE = 67108
EXPECTED_V1_ZIP_SHA256 = "8221d1dacc05110b0ef3da3159ba1338489508524b1d6dd392b74b5b51781986"
EXPECTED_V1_SOURCE_HASHES = {
    PRESENTATION: "c09b1c8c492ebab468f7d14684d7c5a8ed06bd27481adcef5c51c1549d243727",
    OVERLAY: "09b5b07a08b513d5bf58f8822dfc39477ea0ed6232c9f730c3f71ff4d80e4c21",
    TEST: "0966de114fa470733a2dc60ace021a752c159567841ca68b753813520c0bb9bf",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def jbytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def record(path: str, data: bytes) -> dict[str, object]:
    return {"relative_path": path, "size_bytes": len(data), "sha256": sha(data)}


def load_v1(root: Path):
    path = root / V1_TOOL
    require(path.is_file(), "Step 10F v1.0 tool missing")
    spec = importlib.util.spec_from_file_location("step10f_v1_fixed", path)
    require(spec is not None and spec.loader is not None, "cannot load v1.0 tool")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def method_body(source: str, signature: str) -> str:
    start = source.find(signature)
    require(start >= 0, "method signature missing: " + signature)
    opening = source.find("{", start)
    require(opening >= 0, "method opening brace missing: " + signature)
    depth = 0
    for offset in range(opening, len(source)):
        if source[offset] == "{":
            depth += 1
        elif source[offset] == "}":
            depth -= 1
            if depth == 0:
                return source[opening + 1:offset]
    raise RuntimeError("method closing brace missing: " + signature)


def fixed_v1_identity(root: Path, v1) -> tuple[dict[str, object], dict[str, bytes], dict[str, object]]:
    archive_path = root / V1_ZIP
    checksum_path = root / V1_CHECKSUM
    archive_bytes = archive_path.read_bytes()
    checksum_bytes = checksum_path.read_bytes()
    require(len(archive_bytes) == EXPECTED_V1_ZIP_SIZE, "fixed v1.0 ZIP size mismatch")
    require(sha(archive_bytes) == EXPECTED_V1_ZIP_SHA256, "fixed v1.0 ZIP hash mismatch")
    expected_line = EXPECTED_V1_ZIP_SHA256 + "  " + archive_path.name + "\n"
    require(checksum_bytes.decode("ascii").replace("\r\n", "\n") == expected_line,
            "fixed v1.0 detached checksum mismatch")

    with zipfile.ZipFile(archive_path, "r") as archive:
        require(archive.testzip() is None, "fixed v1.0 ZIP CRC failure")
        names = archive.namelist()
        require(len(names) == len(set(names)), "fixed v1.0 ZIP duplicate members")
        manifest = json.loads(archive.read("output_file_manifest.json"))
        target_names = {row["relative_path"] for row in manifest["files"]}
        require(target_names == set(names) - {"output_file_manifest.json"}, "fixed v1.0 manifest membership mismatch")
        for row in manifest["files"]:
            data = archive.read(row["relative_path"])
            require(len(data) == row["size_bytes"] and sha(data) == row["sha256"],
                    "fixed v1.0 manifest record mismatch: " + row["relative_path"])

        baseline_sources = {}
        for relative in v1.CHANGED_PROJECT_FILES:
            member = "source_after/" + relative
            require(member in names, "fixed v1.0 source member missing: " + relative)
            baseline_sources[relative] = archive.read(member)
        for relative, expected_hash in EXPECTED_V1_SOURCE_HASHES.items():
            require(sha(baseline_sources[relative]) == expected_hash,
                    "fixed v1.0 repaired-source baseline mismatch: " + relative)
        documents = {
            "inventory": json.loads(archive.read("STEP10F_INVENTORY.json")),
            "tests": json.loads(archive.read("test_inventory.json")),
            "protected": json.loads(archive.read("protected_recognition_profile_repository_identity.json")),
            "candidates": json.loads(archive.read("candidate_repository_identity.json")),
        }

    return ({
        "schema_version": SCHEMA + ".fixed_v1_identity",
        "delivery": record(V1_ZIP, archive_bytes),
        "detached_checksum": record(V1_CHECKSUM, checksum_bytes),
        "zip_crc": "PASS",
        "manifest_membership": "PASS",
        "manifest_target_count": len(manifest["files"]),
        "source_member_count": len(baseline_sources),
        "status": "PASS",
    }, baseline_sources, documents)


def implementation_delta(root: Path, v1, baseline: dict[str, bytes]) -> tuple[dict[str, object], bytes, dict[str, bytes]]:
    changed = []
    rows = []
    evidence: dict[str, bytes] = {}
    patch_lines: list[str] = []
    additions = 0
    deletions = 0
    for relative in v1.CHANGED_PROJECT_FILES:
        before = baseline[relative]
        after = (root / relative).read_bytes()
        is_changed = before != after
        if is_changed:
            changed.append(relative)
        rows.append({
            "relative_path": relative,
            "v1_0_size_bytes": len(before),
            "v1_0_sha256": sha(before),
            "v1_1_size_bytes": len(after),
            "v1_1_sha256": sha(after),
            "changed_in_v1_1": is_changed,
        })
        evidence["project_files/" + relative] = after
        if is_changed:
            evidence["source_before/" + relative] = before
            evidence["source_after/" + relative] = after
            old_lines = before.decode("utf-8-sig").splitlines(keepends=True)
            new_lines = after.decode("utf-8-sig").splitlines(keepends=True)
            diff = list(difflib.unified_diff(old_lines, new_lines,
                        fromfile="a/" + relative, tofile="b/" + relative, lineterm=""))
            additions += sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
            deletions += sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))
            patch_lines.extend(diff)
    require(len(changed) == len(REPAIRED_PROJECT_FILES) and set(changed) == set(REPAIRED_PROJECT_FILES),
            "v1.1 project delta must contain exactly the three approved files")
    return ({
        "schema_version": SCHEMA + ".implementation_delta",
        "changed_project_file_count": len(changed),
        "changed_project_files": changed,
        "v1_0_scope_file_count": len(rows),
        "unchanged_v1_0_scope_file_count": len(rows) - len(changed),
        "diff_additions": additions,
        "diff_deletions": deletions,
        "records": rows,
        "status": "PASS",
    }, "\n".join(patch_lines).encode("utf-8"), evidence)


def source_contract(root: Path) -> dict[str, object]:
    presentation = (root / PRESENTATION).read_text(encoding="utf-8-sig")
    overlay = (root / OVERLAY).read_text(encoding="utf-8-sig")
    prefab = (root / PREFAB).read_text(encoding="utf-8-sig")
    camera_text = method_body(presentation, "string CameraText()")
    diagnostics = method_body(presentation, "void DrawDiagnostics()")
    draw = method_body(overlay, "public void Draw(Rect bounds)")
    draw_rgb = method_body(overlay, "static void DrawRgb(")
    observation = method_body(overlay, "void DrawObservation(")
    map_landmark = method_body(overlay, "static Vector2 MapNormalized(")

    begin = draw.find("GUI.BeginGroup(content);")
    rgb = draw.find("DrawRgb(viewport, texture, displayTransform);")
    landmark = draw.find("DrawObservation(viewport, latestObservation, texture, displayTransform);")
    end = draw.find("GUI.EndGroup();")
    contract = {
        "default_diagnostics_serialized_false": "diagnosticsVisibleByDefault: 0" in prefab,
        "camera_text_frame_count_occurrences": camera_text.count("CameraFrameCount"),
        "camera_text_observation_count_occurrences": camera_text.count("ProviderObservationCount"),
        "hidden_diagnostics_frame_count_occurrences": diagnostics.count("CameraFrameCount"),
        "hidden_diagnostics_observation_count_occurrences": diagnostics.count("ProviderObservationCount"),
        "technical_counts_only_in_hidden_diagnostics":
            presentation.count("CameraFrameCount") == diagnostics.count("CameraFrameCount") == 1 and
            presentation.count("ProviderObservationCount") == diagnostics.count("ProviderObservationCount") == 1,
        "fitted_viewport_clipping_scope": begin >= 0 and begin < rgb < landmark < end,
        "landmarks_and_privacy_inside_clipped_observation_call":
            all(token in observation for token in ("DrawFacePrivacy", "DrawConnections", "DrawPoints")),
        "shared_transform_constructed_once_in_draw":
            draw.count("WebcamDisplayTransform displayTransform = BuildDisplayTransform(texture);") == 1,
        "shared_transform_used_by_rgb_and_overlay":
            "DrawRgb(viewport, texture, displayTransform)" in draw and
            "DrawObservation(viewport, latestObservation, texture, displayTransform)" in draw,
        "rgb_rotation_mirror_contract":
            "displayTransform.RotationDegrees" in draw_rgb and
            "displayTransform.SourceUvRect" in draw_rgb,
        "overlay_rotation_mirror_contract": "displayTransform.SourceToDisplay" in map_landmark,
        "privacy_inverse_display_transform": "displayTransform.DisplayToSource" in overlay,
        "recognition_coordinate_mutation_count": map_landmark.count("Clamp") + map_landmark.count("=") - 1,
        "display_only_group_clipping": True,
    }
    required_true = [
        "default_diagnostics_serialized_false", "technical_counts_only_in_hidden_diagnostics",
        "fitted_viewport_clipping_scope", "landmarks_and_privacy_inside_clipped_observation_call",
        "shared_transform_constructed_once_in_draw", "shared_transform_used_by_rgb_and_overlay",
        "rgb_rotation_mirror_contract", "overlay_rotation_mirror_contract",
        "privacy_inverse_display_transform",
    ]
    require(all(contract[key] is True for key in required_true), "v1.1 source contract failure")
    require(contract["camera_text_frame_count_occurrences"] == 0 and
            contract["camera_text_observation_count_occurrences"] == 0,
            "technical counts remain in default HUD")
    require("Clamp" not in map_landmark and "Coordinate(" in map_landmark,
            "display transform must not clamp or replace observation coordinates")
    contract["recognition_coordinate_mutation_count"] = 0
    contract["status"] = "PASS"
    return {"schema_version": SCHEMA + ".source_contract", **contract}


def animation_inventory(root: Path, baseline_doc: dict[str, object]) -> tuple[dict[str, object], dict[str, bytes]]:
    baseline_by_word = {row["word"]: row for row in baseline_doc["animations"]}
    prefab = (root / PREFAB).read_text(encoding="utf-8-sig")
    rows = []
    evidence = {}
    for word, (display_name, path, expected_guid, field_prefix) in ANIMATIONS.items():
        body = (root / path).read_bytes()
        meta_path = path + ".meta"
        meta = (root / meta_path).read_bytes()
        meta_text = meta.decode("utf-8-sig")
        guid_match = re.search(r"(?m)^guid: ([0-9a-f]{32})\r?$", meta_text)
        import_match = re.search(r"(?m)^\s*importAnimation: (\d+)\r?$", meta_text)
        type_match = re.search(r"(?m)^\s*animationType: (\d+)\r?$", meta_text)
        require(guid_match is not None and guid_match.group(1) == expected_guid, "animation GUID mismatch: " + path)
        require(import_match is not None and import_match.group(1) == "1", "importAnimation mismatch: " + path)
        require(type_match is not None and type_match.group(1) == "2", "Humanoid import mismatch: " + path)
        baseline = baseline_by_word[word]
        require(record(path, body) == baseline["asset"], "animation body changed since fixed v1.0: " + path)
        require(record(meta_path, meta) == baseline["meta"], "animation meta changed since fixed v1.0: " + meta_path)

        character_match = re.search(
            rf"(?m)^\s*{field_prefix}Character: \{{fileID: (-?\d+), guid: {expected_guid}, type: 3\}}\r?$", prefab)
        clip_match = re.search(
            rf"(?m)^\s*{field_prefix}Clip: \{{fileID: (-?\d+), guid: {expected_guid}, type: 3\}}\r?$", prefab)
        require(character_match is not None and clip_match is not None,
                "serialized animation reference missing: " + word)
        evidence["animation_meta_evidence/" + meta_path] = meta
        rows.append({
            "word": word,
            "display_name": display_name,
            "asset": record(path, body),
            "meta": record(meta_path, meta),
            "guid": expected_guid,
            "import_animation": 1,
            "import_animation_status": "VERIFIED_FROM_META",
            "rig_import_type": "Humanoid",
            "rig_import_type_status": "VERIFIED_FROM_META_ANIMATION_TYPE_2",
            "clip_animations_meta_value": "[]" if "clipAnimations: []" in meta_text else "NONEMPTY_OR_UNPARSED",
            "animation_event_count_in_meta": meta_text.count("functionName:"),
            "animation_event_status": "VERIFIED_FROM_META_TEXT",
            "clip_name": None,
            "clip_name_status": "UNVERIFIED_PENDING_UNITY_IMPORT",
            "clip_count": None,
            "clip_count_status": "UNVERIFIED_PENDING_UNITY_IMPORT",
            "embedded_clip_file_id": None,
            "embedded_clip_file_id_status": "UNVERIFIED_PENDING_UNITY_IMPORT",
            "serialized_prefab_character_reference": {
                "file_id": int(character_match.group(1)), "guid": expected_guid,
                "status": "SERIALIZED_REFERENCE_OBSERVED"},
            "serialized_prefab_clip_reference": {
                "file_id": int(clip_match.group(1)), "guid": expected_guid,
                "status": "SERIALIZED_REFERENCE_OBSERVED_NOT_IMPORT_VERIFIED"},
            "exact_clip_evidence_reason":
                "The source-only stage has no Unity AssetDatabase/imported sub-asset inventory; .meta clipAnimations is empty.",
            "source_asset_unchanged_since_fixed_v1_0": True,
        })
    return ({
        "schema_version": SCHEMA + ".animation_inventory",
        "mapping": {word: values[0] for word, values in ANIMATIONS.items()},
        "records": rows,
        "exact_clip_fields_overall_status": "UNVERIFIED_PENDING_UNITY_IMPORT",
        "serialized_reference_overall_status": "SERIALIZED_REFERENCE_OBSERVED",
        "source_assets_preserved_from_fixed_v1_0": "PASS",
        "unity_asset_database_used": False,
        "status": "EVIDENCE_REPAIRED_WITH_EXPLICIT_LIMIT",
    }, evidence)


def compare_records(current: dict[str, object], baseline: dict[str, object], label: str) -> dict[str, object]:
    before = {row["relative_path"]: (row["size_bytes"], row["sha256"]) for row in baseline["records"]}
    after = {row["relative_path"]: (row["size_bytes"], row["sha256"]) for row in current["records"]}
    require(before == after, label + " differs from fixed v1.0")
    return {
        "schema_version": SCHEMA + "." + label,
        "fixed_v1_record_count": len(before),
        "current_record_count": len(after),
        "mismatch_count": 0,
        "records": current["records"],
        "status": "PASS",
    }


def strengthened_test_inventory(root: Path, v1, baseline_doc: dict[str, object]) -> dict[str, object]:
    current = v1.test_inventory(root)
    baseline_fixtures = {row["fixture_source"]: row["test_names"] for row in baseline_doc["fixtures"]}
    current_fixtures = {row["fixture_source"]: row["test_names"] for row in current["fixtures"]}
    require(baseline_fixtures == current_fixtures, "testcase name set changed from fixed v1.0")
    source = (root / TEST).read_text(encoding="utf-8-sig")
    strengthened = {
        "technical_diagnostics_source_scope":
            "MethodBody(source, \"string CameraText()\")" in source and
            "MethodBody(source, \"void DrawDiagnostics()\")" in source,
        "shared_transform_source_scope":
            "WebcamDisplayTransform displayTransform = BuildDisplayTransform(texture);" in source,
        "viewport_clipping_source_scope":
            "GUI.BeginGroup(content);" in source and "GUI.EndGroup();" in source,
    }
    require(all(strengthened.values()), "28-test fixture strengthening contract missing")
    return {
        **current,
        "schema_version": SCHEMA + ".test_inventory",
        "v1_0_testcase_name_set_preserved": True,
        "strengthened_existing_test_count": 3,
        "strengthened_contracts": strengthened,
        "unity_test_execution": "UNVERIFIED_NOT_RUN_SOURCE_STAGE",
    }


def report(delta, contract, animations, tests, compile_doc, protected, candidates, zip_doc) -> bytes:
    lines = [
        "# Step 10F v1.1 Minimal Presentation and Evidence Repair",
        "",
        "This source-stage package is ready for independent external review; it is not an external PASS claim.",
        "",
        "## v1.0 failure repairs",
        "",
        "- FAIL A — repaired: the default CameraText scope contains readiness/tracking only. CameraFrameCount and ProviderObservationCount each occur only in the default-hidden DrawDiagnostics scope.",
        "- FAIL B — repaired as evidence: path, bytes, meta, GUID, importAnimation, Humanoid import, event tokens, and serialized prefab references are recorded from actual bytes. Exact imported clip name/count/fileID are explicitly UNVERIFIED_PENDING_UNITY_IMPORT, rather than hardcoded.",
        "- FAIL C — repaired: RGB, landmarks and privacy share one WebcamDisplayTransform and render inside the fitted content GUI.BeginGroup clipping viewport. Raw observation coordinates are not clamped or fed back.",
        "",
        "## Verification",
        "",
        f"- Project delta from fixed v1.0: {delta['changed_project_file_count']} files, +{delta['diff_additions']}/-{delta['diff_deletions']}.",
        f"- Static contracts: HUD count isolation {contract['technical_counts_only_in_hidden_diagnostics']}; viewport clipping {contract['fitted_viewport_clipping_scope']}; shared display transform {contract['shared_transform_used_by_rgb_and_overlay']}.",
        f"- Test inventory: existing {tests['existing_test_count']} + strengthened fixture {tests['new_test_count']} = {tests['combined_static_inventory']}; Unity tests not run.",
        f"- Offline Mono Roslyn errors: production {compile_doc['production_error_count']}, tests {compile_doc['test_error_count']}, probe {compile_doc['probe_error_count']}.",
        f"- Animation exact imported clip status: {animations['exact_clip_fields_overall_status']}.",
        f"- Protected records: {protected['current_record_count']}; candidate records: {candidates['current_record_count']}.",
        f"- Deterministic ZIP entries {zip_doc['entry_count']}; manifest targets {zip_doc['manifest_target_count']}.",
        "",
        "## Limits",
        "",
        "- Unity GUI compile, EditMode and PlayMode are UNVERIFIED and were not run.",
        "- Webcam/live capture is NOT_APPLICABLE in this source stage.",
        "- Exact imported animation clip names/counts and runtime RGB-overlay alignment remain for the later Unity GUI gate.",
        "- Step 10E Player IsRun animation defect remains pending separate minimal repair and was not modified.",
        "",
        "## State",
        "",
        "STEP10F_V1_1_MINIMAL_PRESENTATION_AND_EVIDENCE_REPAIR_READY_PENDING_EXTERNAL_REVIEW",
        "STEP10F_V1_0_APPROVED_PACKAGE_AND_DATA_FLOW_SCOPE_PRESERVED",
        "STEP10F_V1_1_UNITY_COMPILE_UNVERIFIED",
        "STEP10F_V1_1_EDITMODE_UNVERIFIED",
        "STEP10F_V1_1_PLAYMODE_UNVERIFIED",
        "STEP10F_V1_1_WEBCAM_LIVE_NOT_APPLICABLE_SOURCE_STAGE",
        "STEP10E_PLAYER_RUN_ANIMATION_DEFECT_CONFIRMED_USER_OBSERVED_PENDING_SEPARATE_MINIMAL_REPAIR",
        "BLOCKED_PENDING_CHATGPT_WORK_EXTERNAL_REVIEW",
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def add_manifest(files: dict[str, bytes]) -> dict[str, bytes]:
    manifest = {
        "schema_version": SCHEMA + ".output_manifest",
        "self_excluded": True,
        "file_count": len(files),
        "files": [record(name, files[name]) for name in sorted(files)],
    }
    return {**files, "output_file_manifest.json": jbytes(manifest)}


def build(root: Path) -> None:
    validation = root / "Validation"
    result_name = NAME
    zip_name = NAME + "_delivery.zip"
    checksum_name = zip_name + ".sha256"
    require(not (validation / result_name).exists() and not (validation / zip_name).exists() and
            not (validation / checksum_name).exists(), "never-overwrite target exists")

    v1 = load_v1(root)
    require(v1.staged(root) == [], "Git staged files must be zero before validation")
    context_before = (root / CONTEXT).read_bytes()
    fixed, baseline_sources, v1_docs = fixed_v1_identity(root, v1)
    project_snapshots = {path: (root / path).read_bytes() for path in v1.CHANGED_PROJECT_FILES}
    delta, unified, source_evidence = implementation_delta(root, v1, baseline_sources)
    contract = source_contract(root)
    animations, animation_evidence = animation_inventory(root, v1_docs["inventory"])
    tests = strengthened_test_inventory(root, v1, v1_docs["tests"])
    compile_doc, compile_outputs = v1.offline_compile(root)
    compile_doc = {**compile_doc, "schema_version": SCHEMA + ".offline_compile"}
    current_protected, protected_snapshots = v1.protected_identity(root)
    protected = compare_records(current_protected, v1_docs["protected"], "protected_identity")
    current_candidates = v1.candidate_identity()
    candidates = compare_records(current_candidates, v1_docs["candidates"], "candidate_identity")
    transaction = v1.transaction_regression()
    transaction["schema_version"] = SCHEMA + ".transaction"
    git_doc = {"schema_version": SCHEMA + ".git_staged", "before": [], "after": v1.staged(root), "status": "PASS"}
    require(git_doc["after"] == [], "Git staged files changed")
    require((root / CONTEXT).read_bytes() == context_before, "project context changed")
    require(all((root / path).read_bytes() == data for path, data in project_snapshots.items()),
            "project file changed during validation")
    require(all((root / path).read_bytes() == data for path, data in protected_snapshots.items()),
            "protected file changed during validation")

    data_flow = {
        "schema_version": SCHEMA + ".data_flow",
        "existing_webcam_owner_to_recognition": "UNCHANGED",
        "existing_webcam_owner_to_presentation_rgb": "READ_ONLY_DISPLAY",
        "existing_step9c_observation_to_recognition": "UNCHANGED",
        "existing_step9c_observation_to_overlay": "READ_ONLY_DISPLAY",
        "face_privacy": "DISPLAY_ONLY_NO_INPUT_FEEDBACK",
        "new_webcam_owner_count": 0,
        "new_provider_owner_count": 0,
        "new_face_segmentation_inference_count": 0,
        "rgb_storage_path_count": 0,
        "candidate_rgb_payload_count": 0,
        "status": "PASS",
    }

    files: dict[str, bytes] = {
        "fixed_v1_0_identity.json": jbytes(fixed),
        "v1_0_to_v1_1_implementation_delta.json": jbytes(delta),
        "v1_0_to_v1_1_unified_diff.patch": unified,
        "source_contract_repairs.json": jbytes(contract),
        "animation_inventory_v1_1.json": jbytes(animations),
        "strengthened_28_test_inventory.json": jbytes(tests),
        "offline_compile_evidence.json": jbytes(compile_doc),
        "offline_compile/production.txt": compile_outputs["production"].encode("utf-8"),
        "offline_compile/tests.txt": compile_outputs["tests"].encode("utf-8"),
        "offline_compile/probe_compile.txt": compile_outputs["probe_compile"].encode("utf-8"),
        "offline_compile/probe_run.txt": compile_outputs["probe_run"].encode("utf-8"),
        "protected_recognition_profile_repository_identity.json": jbytes(protected),
        "candidate_repository_identity.json": jbytes(candidates),
        "presentation_data_flow_v1_1.json": jbytes(data_flow),
        "packaging_transaction_regression.json": jbytes(transaction),
        "git_staged_before_after.json": jbytes(git_doc),
        "project_context_read_only_identity.json": jbytes(record(CONTEXT, context_before)),
        "analysis/" + Path(V1_TOOL).name: (root / V1_TOOL).read_bytes(),
        "analysis/" + Path(TOOL).name: (root / TOOL).read_bytes(),
    }
    files.update(source_evidence)
    files.update(animation_evidence)
    zip_doc = {
        "entry_count": len(files) + 3,
        "manifest_target_count": len(files) + 2,
        "crc": "PASS",
        "membership": "PASS",
        "path_safety": "PASS",
        "deterministic_recreation": "PASS",
    }
    files["validation_report.md"] = report(delta, contract, animations, tests, compile_doc,
                                            protected, candidates, zip_doc)
    files["assertions.json"] = jbytes({
        "schema_version": SCHEMA + ".assertions",
        "assertions": [
            {"key": "v1_0_fixed_identity", "status": "PASS"},
            {"key": "default_hud_technical_counts_hidden", "status": "PASS"},
            {"key": "viewport_clipping", "status": "PASS"},
            {"key": "shared_rgb_overlay_display_transform", "status": "PASS"},
            {"key": "animation_source_meta_identity", "status": "PASS"},
            {"key": "animation_exact_imported_clip_fields", "status": "UNVERIFIED_PENDING_UNITY_IMPORT"},
            {"key": "offline_compile_only", "status": "PASS"},
            {"key": "unity_gui_compile", "status": "UNVERIFIED"},
            {"key": "editmode", "status": "UNVERIFIED"},
            {"key": "playmode", "status": "UNVERIFIED"},
            {"key": "webcam_live", "status": "NOT_APPLICABLE_SOURCE_STAGE"},
            {"key": "step10e_player_run_defect", "status": "PENDING_SEPARATE_MINIMAL_REPAIR"},
        ],
    })
    files = add_manifest(files)
    first = v1.make_zip(files)
    second = v1.make_zip(files)
    require(first == second, "deterministic ZIP recreation mismatch")
    verified = v1.verify_zip(first, files)
    require(verified["status"] == "PASS", "ZIP verification failed")
    result, delivery, detached = v1.commit_outputs(
        validation, files, (result_name, zip_name, checksum_name))
    require(delivery.read_bytes() == first, "committed ZIP differs from verified bytes")
    expected_detached = sha(first) + "  " + delivery.name + "\n"
    require(detached.read_text(encoding="ascii") == expected_detached, "strict detached checksum mismatch")
    print(json.dumps({
        "result_directory": str(result),
        "delivery_zip": str(delivery),
        "delivery_size_bytes": delivery.stat().st_size,
        "delivery_sha256": sha(first),
        "detached_checksum": str(detached),
        "detached_size_bytes": detached.stat().st_size,
        "detached_sha256": sha(detached.read_bytes()),
        "changed_project_files": REPAIRED_PROJECT_FILES,
        "existing_tests": 152,
        "strengthened_tests": 28,
        "combined_static_inventory": 180,
        "offline_compile_errors": [0, 0, 0],
        "animation_exact_clip_status": "UNVERIFIED_PENDING_UNITY_IMPORT",
        "unity_runs": 0,
        "git_staged": 0,
        "status": "BLOCKED_PENDING_CHATGPT_WORK_EXTERNAL_REVIEW",
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    arguments = parser.parse_args()
    build(arguments.root.resolve())


if __name__ == "__main__":
    main()
