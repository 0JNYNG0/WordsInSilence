#!/usr/bin/env python3
"""Package the first Step 10F v1.1 GUI gate evidence without changing Unity project files."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile


NAME = "WordsInSilence_step10f_privacy_enhanced_guided_sign_practice_presentation_v1_1_gui_unity_gate"
INPUT = "Validation/" + NAME
RESULT = "Validation/" + NAME + "_evidence_v1_0"
DELIVERY = "Validation/" + NAME + "_delivery.zip"
DETACHED = DELIVERY + ".sha256"
V1_DELIVERY = "Validation/WordsInSilence_step10f_privacy_enhanced_guided_sign_practice_presentation_v1_1_delivery.zip"
V1_TOOL = "Tools/s001_step10f_privacy_enhanced_guided_sign_practice_presentation_v1_0.py"
TOOL = "Tools/s001_step10f_v1_1_gui_gate_evidence_v1_0.py"
CONTEXT = "WordsInSilence_PROJECT_CONTEXT.md"
DEMO = "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/SignDemonstrationAvatarPresenter.cs"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def record(path: str, data: bytes) -> dict[str, object]:
    return {"relative_path": path, "size_bytes": len(data), "sha256": digest(data)}


def jbytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def load_v1(root: Path):
    spec = importlib.util.spec_from_file_location("step10f_gate_v1", root / V1_TOOL)
    require(spec is not None and spec.loader is not None, "v1 helper unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def zip_bytes(files: dict[str, bytes]) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temporary:
        path = Path(temporary.name)
    try:
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name in sorted(files):
                require(not name.startswith(("/", "\\")) and ".." not in PurePosixPath(name).parts,
                        "unsafe path")
                info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, files[name], compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        return path.read_bytes()
    finally:
        path.unlink(missing_ok=True)


def verify_zip(data: bytes, files: dict[str, bytes]) -> None:
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temporary:
        path = Path(temporary.name)
        temporary.write(data)
    try:
        with zipfile.ZipFile(path) as archive:
            require(archive.testzip() is None, "CRC failure")
            names = archive.namelist()
            require(names == sorted(files) and len(names) == len(set(names)), "membership failure")
            require(all(archive.read(name) == files[name] for name in names), "payload failure")
    finally:
        path.unlink(missing_ok=True)


def add_manifest(files: dict[str, bytes]) -> dict[str, bytes]:
    manifest = {
        "schema_version": "wordsinsilence.step10f.v1.1.gui_gate.output_manifest.v1.0",
        "self_excluded": True,
        "file_count": len(files),
        "files": [record(name, files[name]) for name in sorted(files)],
    }
    return {**files, "output_file_manifest.json": jbytes(manifest)}


def commit(root: Path, files: dict[str, bytes]) -> tuple[Path, Path, Path]:
    result, delivery, detached = root / RESULT, root / DELIVERY, root / DETACHED
    require(not result.exists() and not delivery.exists() and not detached.exists(), "never-overwrite target exists")
    parent = result.parent
    token = os.urandom(8).hex()
    temp_result = parent / (".step10f-gate-result-" + token)
    temp_zip = parent / (".step10f-gate-delivery-" + token + ".zip")
    temp_sha = parent / (".step10f-gate-checksum-" + token)
    created: list[Path] = []
    try:
        temp_result.mkdir(parents=True)
        for name, data in files.items():
            target = temp_result / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        os.replace(temp_result, result); created.append(result)
        data = zip_bytes(files)
        temp_zip.write_bytes(data)
        os.replace(temp_zip, delivery); created.append(delivery)
        temp_sha.write_text(digest(data) + "  " + delivery.name + "\n", encoding="ascii")
        os.replace(temp_sha, detached); created.append(detached)
        return result, delivery, detached
    except Exception:
        for path in reversed(created):
            if path.is_dir(): shutil.rmtree(path, ignore_errors=True)
            else: path.unlink(missing_ok=True)
        raise
    finally:
        if temp_result.exists(): shutil.rmtree(temp_result, ignore_errors=True)
        temp_zip.unlink(missing_ok=True)
        temp_sha.unlink(missing_ok=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    input_root = root / INPUT
    require(input_root.is_dir(), "GUI input evidence missing")
    required = [
        "01_compiler_error_0.png",
        "02_test_runner_28_pass.png",
        "03_presentation_initial_and_greeting_clip.png",
        "04_webcam_overlay_privacy_and_animation.png",
        "04_webcam_failed.png",
        "Editor.log",
        "TestResults_Step10F_v1_1_28.xml",
    ]
    require(all((input_root / name).is_file() for name in required), "required GUI evidence missing")
    require(not (root / RESULT).exists() and not (root / DELIVERY).exists() and not (root / DETACHED).exists(),
            "never-overwrite target exists")

    staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=root,
                            capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    staged_names = [line for line in staged.stdout.splitlines() if line.strip()]
    require(not staged_names, "Git staged files must be zero")
    context = (root / CONTEXT).read_bytes()

    xml_path = input_root / "TestResults_Step10F_v1_1_28.xml"
    xml_root = ET.parse(xml_path).getroot()
    cases = list(xml_root.iter("test-case"))
    names = [case.attrib.get("fullname") or case.attrib.get("name") for case in cases]
    require(xml_root.attrib.get("total") == "28" and xml_root.attrib.get("passed") == "28" and
            xml_root.attrib.get("failed") == "0" and xml_root.attrib.get("skipped") == "0" and
            xml_root.attrib.get("inconclusive") == "0", "targeted XML result mismatch")
    require(len(cases) == len(set(names)) == 28 and all(case.attrib.get("result") == "Passed" for case in cases),
            "targeted testcase identity mismatch")

    log = (input_root / "Editor.log").read_text(encoding="utf-8-sig", errors="replace")
    prebuild = log.count("Executing IPrebuildSetup for: Unity.PerformanceTesting.Editor.TestRunBuilder.")
    postbuild = log.count("Executing IPostBuildCleanup for: Unity.PerformanceTesting.Editor.TestRunBuilder.")
    require(prebuild == postbuild == 1, "targeted execution cycle is not exactly once")
    require(len(re.findall(r"error CS\d+", log)) == 0, "post-repair Editor log contains compiler errors")

    v1_zip_bytes = (root / V1_DELIVERY).read_bytes()
    require(len(v1_zip_bytes) == 92636 and
            digest(v1_zip_bytes) == "535dc6f8c5d213397a0a12dfa1a9229f31de0f2dc6d19a9d917dc3ff051c1aaf",
            "fixed v1.1 delivery mismatch")
    with zipfile.ZipFile(root / V1_DELIVERY) as archive:
        delta = json.loads(archive.read("v1_0_to_v1_1_implementation_delta.json"))
        protected_baseline = json.loads(archive.read("protected_recognition_profile_repository_identity.json"))
        candidate_baseline = json.loads(archive.read("candidate_repository_identity.json"))
        before_demo = archive.read("project_files/" + DEMO)
        scope = []
        for row in delta["records"]:
            relative = row["relative_path"]
            current = (root / relative).read_bytes()
            approved = archive.read("project_files/" + relative)
            expected_repair = relative == DEMO
            require((current != approved) == expected_repair, "unexpected v1.1 scope change: " + relative)
            scope.append({**record(relative, current), "matches_fixed_v1_1": current == approved,
                          "classification": "minimal_camera_namespace_compile_repair" if expected_repair else "preserved"})

    v1 = load_v1(root)
    protected_current, protected_snapshots = v1.protected_identity(root)
    candidate_current = v1.candidate_identity()
    key = lambda doc: {row["relative_path"]: (row["size_bytes"], row["sha256"]) for row in doc["records"]}
    require(key(protected_current) == key(protected_baseline), "protected identity mismatch")
    require(key(candidate_current) == key(candidate_baseline), "candidate identity mismatch")

    after_demo = (root / DEMO).read_bytes()
    before_lines = before_demo.decode("utf-8-sig").splitlines(keepends=True)
    after_lines = after_demo.decode("utf-8-sig").splitlines(keepends=True)
    import difflib
    repair_diff = "".join(difflib.unified_diff(before_lines, after_lines,
                          fromfile="a/" + DEMO, tofile="b/" + DEMO))
    require("UnityEngine.Camera" in repair_diff and "Camera CreateCamera" in repair_diff,
            "compile repair diff contract missing")

    verdict = {
        "schema_version": "wordsinsilence.step10f.v1.1.gui_gate.verdict.v1.0",
        "unity_version": "6000.3.14f1",
        "initial_compile": "FAIL_CS0118_CAMERA_NAMESPACE_COLLISION",
        "minimal_compile_repair": "UnityEngine.Camera fully-qualified in SignDemonstrationAvatarPresenter.cs",
        "post_repair_compiler_error_count": 0,
        "targeted_editmode_execution_count": 1,
        "targeted_editmode": {"total": 28, "passed": 28, "failed": 0, "skipped": 0, "inconclusive": 0},
        "playmode_execution_count_user_reported": 1,
        "webcam_rgb": "OBSERVED",
        "pose_hand_overlay": "OBSERVED",
        "viewport_clipping": "OBSERVED_NO_BLEED_IN_PROVIDED_SCREEN",
        "greeting_demonstration_character": "FAIL_TYPE_MISMATCH_BLANK_SLOT",
        "greeting_animation": "NOT_APPLICABLE_CHARACTER_NOT_RENDERED",
        "robot_visual": "OBSERVED_RENDER_TEXTURE_PRESENTATION_USER_REJECTED",
        "presentation_design_acceptance": "FAIL_USER_REQUESTED_3D_WORLD_PRESENTATION_REWORK",
        "overall_presentation_smoke": "FAIL",
        "candidate_storage": "NOT_APPLICABLE_NO_CAPTURE",
        "player_isrun_defect": "PENDING_SEPARATE_MINIMAL_REPAIR_UNCHANGED",
        "external_review_claimed": False,
    }
    identity = {
        "schema_version": "wordsinsilence.step10f.v1.1.gui_gate.identity.v1.0",
        "fixed_v1_1_delivery": record(V1_DELIVERY, v1_zip_bytes),
        "project_scope": scope,
        "project_scope_count": len(scope),
        "fixed_scope_match_count": sum(row["matches_fixed_v1_1"] for row in scope),
        "minimal_compile_repair_count": 1,
        "protected_record_count": len(protected_current["records"]),
        "protected_mismatch_count": 0,
        "candidate_record_count": len(candidate_current["records"]),
        "candidate_mismatch_count": 0,
        "git_staged_count": 0,
        "context": record(CONTEXT, context),
    }
    report = """# Step 10F v1.1 Unity GUI Gate Evidence\n\n""" \
        "The targeted test gate passed after a one-file Camera namespace compile repair, but the presentation smoke failed. " \
        "The left GREETING demonstration slot is blank and the Inspector shows Type mismatch. The user also rejected the flat render-texture/UI composition and requested a later 3D-world presentation redesign.\n\n" \
        "## Results\n\n- Post-repair compiler errors: 0\n- Targeted EditMode: 28/28 PASS, exactly one prebuild/postbuild cycle\n" \
        "- Webcam RGB and pose/hand overlay: observed\n- GREETING demonstration character/animation: FAIL / NOT_APPLICABLE (Type mismatch, blank slot)\n" \
        "- Overall presentation smoke: FAIL\n- Candidate storage: NOT_APPLICABLE; capture was not started\n\n" \
        "## Preservation\n\n- Nine of the fixed v1.1 ten-file project scope remain byte-identical. One source has the minimal UnityEngine.Camera qualification repair.\n" \
        "- Protected 36 records and candidate 25 records match the fixed v1.1 package. Git staged files: 0.\n\n" \
        "This is self-evidence pending independent external review. Step 10E Player IsRun remains a separate pending defect.\n"

    files: dict[str, bytes] = {
        "gate_verdict.json": jbytes(verdict),
        "before_after_identity.json": jbytes(identity),
        "camera_namespace_compile_repair.patch": repair_diff.encode("utf-8"),
        "source_before/" + DEMO: before_demo,
        "source_after/" + DEMO: after_demo,
        "validation_report.md": report.encode("utf-8"),
        "analysis/" + Path(TOOL).name: (root / TOOL).read_bytes(),
    }
    for name in required:
        files["gui_evidence/" + name] = (input_root / name).read_bytes()
    files = add_manifest(files)
    first = zip_bytes(files)
    second = zip_bytes(files)
    require(first == second, "deterministic recreation failure")
    verify_zip(first, files)
    result, delivery, detached = commit(root, files)
    require(delivery.read_bytes() == first, "committed ZIP mismatch")
    print(json.dumps({
        "result": str(result), "delivery": str(delivery),
        "delivery_size_bytes": delivery.stat().st_size,
        "delivery_sha256": digest(first),
        "detached": str(detached), "detached_sha256": digest(detached.read_bytes()),
        "entries": len(files), "manifest_targets": len(files) - 1,
        "targeted_tests": "28/28", "presentation_smoke": "FAIL",
        "git_staged": 0,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
