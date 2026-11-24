#!/usr/bin/env python3
"""Capture and package Step 10F v1.2 Unity GUI-gate evidence only.

This helper never starts Unity and never writes below Assets/.  It records a
fresh pre/post byte identity, an append-only execution ledger, and creates a
transactional, never-overwrite deterministic Delivery from collected evidence.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
import zipfile


NAME = "WordsInSilence_step10f_v1_2_unity_gui_gate_evidence_v1_0"
WORK = "Validation/.step10f_v1_2_gui_gate_work_v1_0"
SOURCE_DELIVERY = (
    "Validation/WordsInSilence_step10f_3d_world_integrated_ugui_guided_sign_"
    "practice_presentation_v1_2_delivery.zip"
)
SOURCE_DELIVERY_SIZE = 139129
SOURCE_DELIVERY_SHA256 = "e027454a4b9af306b6238e7c272a832629659f466be2097469d75ea68e668bbb"
CONTEXT = "WordsInSilence_PROJECT_CONTEXT.md"
SCENE = "Assets/_Project/Scenes/VerticalSlice/PetRobotSignPracticeVerticalSlice.unity"

PROJECT_FILES = [
    SCENE,
    "Assets/_Project/Resources/VerticalSlice/PetRobotSignPracticeVerticalSlice.prefab",
    "Assets/_Project/Resources/VerticalSlice/SignDemonstrationGreetingWrapper.prefab",
    "Assets/_Project/Resources/VerticalSlice/SignDemonstrationLoveWrapper.prefab",
    "Assets/_Project/Resources/VerticalSlice/SignDemonstrationThankYouWrapper.prefab",
    "Assets/_Project/Resources/VerticalSlice/DisplayOnlyMildBlur.shader",
    "Assets/_Project/Resources/VerticalSlice/DisplayOnlyMildBlur.mat",
    "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PetRobotPracticeVerticalSliceBootstrap.cs",
    "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/SignDemonstrationAvatarPresenter.cs",
    "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/WebcamLandmarkOverlayPresenter.cs",
    "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/WebcamTrackingGraphic.cs",
    "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/GuidedSignPracticePresentationController.cs",
    "Assets/_Project/Tests/EditMode/PrivacyEnhancedGuidedSignPracticePresentationV1Tests.cs",
]

MEDIA_SUFFIXES = {".png", ".jpg", ".jpeg", ".webm", ".mp4", ".mov", ".wav", ".mp3"}
ALLOWED_EVENTS = ["unity_gui_launch", "targeted_run_selected", "playmode_enter"]


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def jbytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(jbytes(value))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def record(relative: str, data: bytes) -> dict[str, object]:
    return {"relative_path": relative, "size_bytes": len(data), "sha256": sha(data)}


def file_record(root: Path, relative: str) -> dict[str, object]:
    path = root / relative
    require(path.is_file(), "required file missing: " + relative)
    return record(relative, path.read_bytes())


def records_map(document: dict) -> dict[str, tuple[int, str]]:
    return {
        row["relative_path"]: (row["size_bytes"], row["sha256"])
        for row in document["records"]
    }


def staged(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"], cwd=root,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    require(result.returncode == 0, "git staged query failed")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def read_source_baselines(root: Path) -> tuple[dict, dict]:
    delivery = root / SOURCE_DELIVERY
    require(delivery.is_file(), "approved v1.2 source Delivery missing")
    data = delivery.read_bytes()
    require(len(data) == SOURCE_DELIVERY_SIZE and sha(data) == SOURCE_DELIVERY_SHA256,
            "approved v1.2 source Delivery identity mismatch")
    with zipfile.ZipFile(delivery, "r") as archive:
        protected = json.loads(archive.read(
            "protected_recognition_profile_repository_identity.json").decode("utf-8-sig"))
        candidates = json.loads(archive.read(
            "candidate_repository_identity.json").decode("utf-8-sig"))
    require(len(protected["records"]) == 36, "approved protected record count mismatch")
    require(len(candidates["records"]) == 25, "approved candidate record count mismatch")
    return protected, candidates


def project_identity(root: Path, phase: str) -> dict:
    exact = [file_record(root, relative) for relative in PROJECT_FILES]
    metas = []
    for relative in PROJECT_FILES:
        meta = relative + ".meta"
        if (root / meta).is_file():
            metas.append(file_record(root, meta))
    return {
        "schema_version": "wordsinsilence.step10f.v1.2.gui_gate.project_identity.v1.0",
        "phase": phase,
        "captured_utc": utc_now(),
        "exact_required_file_count": len(exact),
        "supporting_meta_count": len(metas),
        "scene": next(row for row in exact if row["relative_path"] == SCENE),
        "records": exact + metas,
    }


def protected_identity(root: Path, baseline: dict, phase: str) -> dict:
    rows = []
    mismatch = []
    for approved in baseline["records"]:
        current = file_record(root, approved["relative_path"])
        match = (current["size_bytes"] == approved["size_bytes"] and
                 current["sha256"] == approved["sha256"])
        rows.append({**current, "approved_v1_2_size_bytes": approved["size_bytes"],
                     "approved_v1_2_sha256": approved["sha256"], "match": match})
        if not match:
            mismatch.append(current["relative_path"])
    return {
        "schema_version": "wordsinsilence.step10f.v1.2.gui_gate.protected_identity.v1.0",
        "phase": phase, "captured_utc": utc_now(), "record_count": len(rows),
        "mismatch_count": len(mismatch), "mismatches": mismatch, "records": rows,
    }


def candidate_root() -> Path:
    return (Path.home() / "AppData/LocalLow/DefaultCompany/WordsInSilence/WordsInSilence/"
            "PersonalizationCandidates/v1/local_learner")


def candidate_identity(baseline: dict, phase: str) -> dict:
    root = candidate_root()
    require(root.is_dir(), "candidate repository missing")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    rows = [record(path.relative_to(root).as_posix(), path.read_bytes()) for path in files]
    approved = records_map(baseline)
    current = records_map({"records": rows})
    index = json.loads((root / "candidate_index.json").read_text(encoding="utf-8-sig"))
    media = [row["relative_path"] for row in rows if Path(row["relative_path"]).suffix.lower() in MEDIA_SUFFIXES]
    mismatches = sorted(set(approved) | set(current))
    mismatches = [name for name in mismatches if approved.get(name) != current.get(name)]
    return {
        "schema_version": "wordsinsilence.step10f.v1.2.gui_gate.candidate_identity.v1.0",
        "phase": phase, "captured_utc": utc_now(), "repository_path": str(root),
        "candidate_count": len(index.get("entries", [])), "record_count": len(rows),
        "media_file_count": len(media), "media_files": media,
        "approved_v1_2_mismatch_count": len(mismatches), "mismatches": mismatches,
        "records": rows,
    }


def other_identity(root: Path, phase: str) -> dict:
    names = staged(root)
    return {
        "schema_version": "wordsinsilence.step10f.v1.2.gui_gate.other_identity.v1.0",
        "phase": phase, "captured_utc": utc_now(), "git_staged_count": len(names),
        "git_staged_paths": names, "context": file_record(root, CONTEXT),
    }


def command_pre(root: Path) -> None:
    work = root / WORK
    require(not work.exists(), "fresh GUI-gate work directory already exists")
    protected_baseline, candidate_baseline = read_source_baselines(root)
    work.mkdir(parents=True)
    try:
        project = project_identity(root, "BEFORE_UNITY_GUI")
        protected = protected_identity(root, protected_baseline, "BEFORE_UNITY_GUI")
        candidates = candidate_identity(candidate_baseline, "BEFORE_UNITY_GUI")
        other = other_identity(root, "BEFORE_UNITY_GUI")
        require(protected["mismatch_count"] == 0, "protected source baseline already differs")
        require(candidates["candidate_count"] == 8 and candidates["record_count"] == 25,
                "candidate baseline is not 8 candidates / 25 files")
        require(candidates["approved_v1_2_mismatch_count"] == 0 and
                candidates["media_file_count"] == 0, "candidate baseline already differs")
        require(other["git_staged_count"] == 0, "Git staged baseline is not zero")
        write_json(work / "fresh_project_identity_before.json", project)
        write_json(work / "protected_identity_before.json", protected)
        write_json(work / "candidate_repository_before.json", candidates)
        write_json(work / "git_context_identity_before.json", other)
        write_json(work / "execution_count_ledger.json", {
            "schema_version": "wordsinsilence.step10f.v1.2.gui_gate.execution_ledger.v1.0",
            "allowed_events": ALLOWED_EVENTS, "events": [],
            "counts": {name: 0 for name in ALLOWED_EVENTS},
        })
        (work / "gui_evidence").mkdir()
    except Exception:
        shutil.rmtree(work, ignore_errors=True)
        raise
    print(json.dumps({
        "work_directory": str(work), "project_record_count": len(project["records"]),
        "scene_sha256": project["scene"]["sha256"],
        "protected_record_count": protected["record_count"],
        "candidate_count": candidates["candidate_count"],
        "candidate_record_count": candidates["record_count"],
        "git_staged_count": other["git_staged_count"],
    }, ensure_ascii=False, sort_keys=True, indent=2))


def command_event(root: Path, name: str) -> None:
    require(name in ALLOWED_EVENTS, "unsupported event name")
    ledger_path = root / WORK / "execution_count_ledger.json"
    require(ledger_path.is_file(), "pre-run ledger missing")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8-sig"))
    require(ledger["counts"][name] == 0, "event already recorded: " + name)
    if name == "targeted_run_selected":
        require(ledger["counts"]["unity_gui_launch"] == 1, "Unity GUI launch not recorded")
    if name == "playmode_enter":
        require(ledger["counts"]["targeted_run_selected"] == 1, "targeted run not recorded")
    ledger["events"].append({"sequence": len(ledger["events"]) + 1,
                             "name": name, "recorded_utc": utc_now()})
    ledger["counts"][name] = 1
    write_json(ledger_path, ledger)
    print(json.dumps(ledger, ensure_ascii=False, sort_keys=True, indent=2))


def changed_rows(before: dict, after: dict) -> list[dict]:
    old = records_map(before)
    new = records_map(after)
    return [
        {"relative_path": name, "before": old.get(name), "after": new.get(name)}
        for name in sorted(set(old) | set(new)) if old.get(name) != new.get(name)
    ]


def command_post(root: Path) -> None:
    work = root / WORK
    require(work.is_dir(), "pre-run work directory missing")
    require(not (work / "fresh_project_identity_after.json").exists(), "post snapshot already exists")
    protected_baseline, candidate_baseline = read_source_baselines(root)
    project = project_identity(root, "AFTER_UNITY_GUI")
    protected = protected_identity(root, protected_baseline, "AFTER_UNITY_GUI")
    candidates = candidate_identity(candidate_baseline, "AFTER_UNITY_GUI")
    other = other_identity(root, "AFTER_UNITY_GUI")
    before_project = json.loads((work / "fresh_project_identity_before.json").read_text(encoding="utf-8-sig"))
    before_protected = json.loads((work / "protected_identity_before.json").read_text(encoding="utf-8-sig"))
    before_candidates = json.loads((work / "candidate_repository_before.json").read_text(encoding="utf-8-sig"))
    before_other = json.loads((work / "git_context_identity_before.json").read_text(encoding="utf-8-sig"))
    project_changes = changed_rows(before_project, project)
    protected_changes = changed_rows(before_protected, protected)
    candidate_changes = changed_rows(before_candidates, candidates)
    scene_match = before_project["scene"]["sha256"] == project["scene"]["sha256"] and before_project["scene"]["size_bytes"] == project["scene"]["size_bytes"]
    context_match = before_other["context"]["sha256"] == other["context"]["sha256"] and before_other["context"]["size_bytes"] == other["context"]["size_bytes"]
    comparison = {
        "schema_version": "wordsinsilence.step10f.v1.2.gui_gate.pre_post_comparison.v1.0",
        "fresh_real_scene_pre_post_match": scene_match,
        "project_change_count": len(project_changes), "project_changes": project_changes,
        "protected_change_count": len(protected_changes), "protected_changes": protected_changes,
        "candidate_change_count": len(candidate_changes), "candidate_changes": candidate_changes,
        "candidate_count_before": before_candidates["candidate_count"],
        "candidate_count_after": candidates["candidate_count"],
        "candidate_record_count_before": before_candidates["record_count"],
        "candidate_record_count_after": candidates["record_count"],
        "candidate_media_file_count_after": candidates["media_file_count"],
        "git_staged_count_before": before_other["git_staged_count"],
        "git_staged_count_after": other["git_staged_count"],
        "project_context_pre_post_match": context_match,
        "all_required_preserved": (
            scene_match and not project_changes and not protected_changes and not candidate_changes and
            candidates["candidate_count"] == 8 and candidates["record_count"] == 25 and
            candidates["media_file_count"] == 0 and other["git_staged_count"] == 0 and context_match
        ),
    }
    write_json(work / "fresh_project_identity_after.json", project)
    write_json(work / "protected_identity_after.json", protected)
    write_json(work / "candidate_repository_after.json", candidates)
    write_json(work / "git_context_identity_after.json", other)
    write_json(work / "fresh_pre_post_preservation.json", comparison)
    print(json.dumps(comparison, ensure_ascii=False, sort_keys=True, indent=2))


def make_zip(files: dict[str, bytes]) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temporary:
        path = Path(temporary.name)
    try:
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name in sorted(files):
                require(not name.startswith(("/", "\\")) and ".." not in PurePosixPath(name).parts,
                        "unsafe archive path")
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
        with zipfile.ZipFile(path, "r") as archive:
            require(archive.testzip() is None, "ZIP CRC verification failed")
            require(archive.namelist() == sorted(files), "ZIP membership/order mismatch")
            require(all(archive.read(name) == files[name] for name in sorted(files)),
                    "ZIP payload verification failed")
    finally:
        path.unlink(missing_ok=True)


def commit(root: Path, files: dict[str, bytes], zip_data: bytes) -> tuple[Path, Path, Path]:
    result = root / "Validation" / NAME
    delivery = root / "Validation" / (NAME + "_delivery.zip")
    detached = root / "Validation" / (NAME + "_delivery.zip.sha256")
    require(not result.exists() and not delivery.exists() and not detached.exists(),
            "never-overwrite output target exists")
    token = os.urandom(8).hex()
    temp_result = result.parent / (".step10f-v12-gate-result-" + token)
    temp_zip = result.parent / (".step10f-v12-gate-delivery-" + token + ".zip")
    temp_sha = result.parent / (".step10f-v12-gate-checksum-" + token)
    committed: list[Path] = []
    try:
        temp_result.mkdir(parents=True)
        for name, data in files.items():
            target = temp_result / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        temp_zip.write_bytes(zip_data)
        temp_sha.write_text(sha(zip_data) + "  " + delivery.name + "\n", encoding="ascii")
        os.replace(temp_result, result); committed.append(result)
        os.replace(temp_zip, delivery); committed.append(delivery)
        os.replace(temp_sha, detached); committed.append(detached)
        return result, delivery, detached
    except Exception:
        for path in reversed(committed):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        raise
    finally:
        if temp_result.exists():
            shutil.rmtree(temp_result, ignore_errors=True)
        temp_zip.unlink(missing_ok=True)
        temp_sha.unlink(missing_ok=True)


def command_package(root: Path) -> None:
    work = root / WORK
    require(work.is_dir(), "GUI-gate work directory missing")
    required = [
        "fresh_project_identity_before.json", "fresh_project_identity_after.json",
        "protected_identity_before.json", "protected_identity_after.json",
        "candidate_repository_before.json", "candidate_repository_after.json",
        "git_context_identity_before.json", "git_context_identity_after.json",
        "fresh_pre_post_preservation.json", "execution_count_ledger.json",
        "compiler_import_summary.json", "gate_verdict.json", "validation_report.md",
        "gui_evidence/Editor.log", "gui_evidence/TestResults.xml",
    ]
    require(all((work / name).is_file() for name in required), "required evidence is incomplete")
    files: dict[str, bytes] = {}
    for path in sorted(work.rglob("*")):
        if path.is_file():
            files[path.relative_to(work).as_posix()] = path.read_bytes()
    tool_relative = "Tools/s001_step10f_v1_2_unity_gui_gate_evidence_v1_0.py"
    files["analysis/" + Path(tool_relative).name] = (root / tool_relative).read_bytes()
    manifest = {
        "schema_version": "wordsinsilence.step10f.v1.2.gui_gate.output_manifest.v1.0",
        "self_excluded": True, "file_count": len(files),
        "files": [record(name, files[name]) for name in sorted(files)],
    }
    files["output_file_manifest.json"] = jbytes(manifest)
    first = make_zip(files)
    second = make_zip(files)
    require(first == second, "deterministic ZIP rebuild mismatch")
    verify_zip(first, files)
    packaging = {
        "schema_version": "wordsinsilence.step10f.v1.2.gui_gate.packaging.v1.0",
        "rebuilt_twice_byte_identical": True, "crc_verified": True,
        "membership_verified": True, "path_safety_verified": True,
        "payload_sha256_before_self_record": sha(first), "entry_count_before_self_record": len(files),
    }
    files["deterministic_zip_verification.json"] = jbytes(packaging)
    manifest = {
        "schema_version": "wordsinsilence.step10f.v1.2.gui_gate.output_manifest.v1.0",
        "self_excluded": True, "file_count": len(files) - 1,
        "files": [record(name, files[name]) for name in sorted(files) if name != "output_file_manifest.json"],
    }
    files["output_file_manifest.json"] = jbytes(manifest)
    final_one = make_zip(files)
    final_two = make_zip(files)
    require(final_one == final_two, "final deterministic ZIP rebuild mismatch")
    verify_zip(final_one, files)
    result, delivery, detached = commit(root, files, final_one)
    require(staged(root) == [], "Git staged state changed during packaging")
    print(json.dumps({
        "result_directory": str(result), "delivery_zip": str(delivery),
        "detached_checksum": str(detached), "delivery_size_bytes": len(final_one),
        "delivery_sha256": sha(final_one), "entry_count": len(files),
        "deterministic_rebuild_byte_identical": True,
    }, ensure_ascii=False, sort_keys=True, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["pre", "event", "post", "package"])
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--name", choices=ALLOWED_EVENTS)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if args.command == "pre":
        command_pre(root)
    elif args.command == "event":
        require(args.name is not None, "event name required")
        command_event(root, args.name)
    elif args.command == "post":
        command_post(root)
    else:
        command_package(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
