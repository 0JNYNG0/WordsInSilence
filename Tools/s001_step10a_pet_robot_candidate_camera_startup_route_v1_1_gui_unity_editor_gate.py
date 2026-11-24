from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import stat
import subprocess
import tempfile
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "Validation" / (
    "WordsInSilence_step10a_pet_robot_candidate_camera_startup_route_v1_1_"
    "gui_unity_editor_gate"
)
ZIP_NAME = (
    "WordsInSilence_step10a_pet_robot_candidate_camera_startup_route_v1_1_"
    "gui_unity_editor_gate_delivery.zip"
)
DELIVERY = ROOT / "Validation" / ZIP_NAME
DETACHED = DELIVERY.with_name(DELIVERY.name + ".sha256")
CANDIDATE_REPOSITORY = Path(
    r"C:\Users\local-user\AppData\LocalLow\DefaultCompany\WordsInSilence\WordsInSilence"
    r"\PersonalizationCandidates\v1\local_learner"
)

EXPECTED_FIXTURES = {
    "PetRobotSignPracticeVerticalSliceV1Tests": 23,
    "GuidedQuestLowConfidenceOutcomeLayerV1Tests": 10,
    "CandidateTransientAmbiguousHandIdentityToleranceV1Tests": 8,
    "PersonalLiveRecognitionIntegrationV1Tests": 22,
    "PersonalMultiPrototypeRuntimeIntegrationV1Tests": 8,
    "PersonalMultiPrototypeLiveOptInRoutingV1Tests": 8,
    "PetRobotCandidateCameraStartupRouteV1Tests": 8,
}
EXPECTED_FAILURE = (
    "WordsInSilence.Tests.EditMode.PetRobotCandidateCameraStartupRouteV1Tests."
    "PreviewAndHandPoseProviders_UseTheSameSerializedFrameSource"
)

GENERATED_NAMES = {
    "test_results_summary.json",
    "editor_log_summary.json",
    "editmode_failure_diagnosis.json",
    "project_files_after.json",
    "protected_files_after.json",
    "active_candidate_profile_after.json",
    "candidate_repository_after.json",
    "git_staged_after.json",
    "playmode_live_gate_evaluation.json",
    "packaging_transaction_failure_injection_regression.json",
    "packaging_verification.json",
    "primary_verdict.json",
    "ASSERTIONS.jsonl",
    "VALIDATION_REPORT.md",
    "output_file_manifest.json",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def jbytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def write_json(path: Path, value: object) -> None:
    path.write_bytes(jbytes(value))


def file_record(path: Path, base: Path = ROOT) -> dict[str, object]:
    data = path.read_bytes()
    try:
        relative = path.relative_to(base).as_posix()
    except ValueError:
        relative = path.as_posix()
    return {
        "relative_path": relative,
        "size_bytes": len(data),
        "sha256": sha(data),
    }


def snapshot_tree(path: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    if path.is_dir():
        for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
            data = item.read_bytes()
            rows.append(
                {
                    "relative_path": item.relative_to(path).as_posix(),
                    "size_bytes": len(data),
                    "sha256": sha(data),
                }
            )
    count = 0
    index = path / "candidate_index.json"
    if index.is_file():
        payload = json.loads(index.read_text(encoding="utf-8"))
        candidates = payload.get("candidates", payload.get("entries", []))
        count = len(candidates)
    return {
        "path": str(path),
        "exists": path.exists(),
        "file_count": len(rows),
        "candidate_count": count,
        "files": rows,
    }


def deterministic_zip(files: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(files):
            pure = PurePosixPath(name)
            require(not pure.is_absolute() and ".." not in pure.parts and "\\" not in name,
                    "unsafe archive path: " + name)
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (0o100644 & 0xFFFF) << 16
            info.flag_bits = 0x800
            archive.writestr(
                info,
                files[name],
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
    return stream.getvalue()


def validate_zip(blob: bytes, expected: dict[str, bytes]) -> dict[str, object]:
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        infos = archive.infolist()
        names = [item.filename for item in infos]
        duplicate_count = len(names) - len(set(names))
        unsafe = [
            name for name in names
            if PurePosixPath(name).is_absolute()
            or ".." in PurePosixPath(name).parts
            or "\\" in name
        ]
        encrypted = [item.filename for item in infos if item.flag_bits & 1]
        symlinks = [
            item.filename for item in infos
            if stat.S_IFMT(item.external_attr >> 16) == stat.S_IFLNK
        ]
        crc_failure = archive.testzip()
        mismatch = [
            name for name, data in expected.items()
            if name not in names or archive.read(name) != data
        ]
        return {
            "entry_count": len(names),
            "duplicate_count": duplicate_count,
            "unsafe_path_count": len(unsafe),
            "encrypted_entry_count": len(encrypted),
            "symlink_entry_count": len(symlinks),
            "crc_failure": crc_failure,
            "membership_match": set(names) == set(expected),
            "payload_mismatch_count": len(mismatch),
            "fixed_timestamp_count": sum(
                item.date_time == (1980, 1, 1, 0, 0, 0) for item in infos
            ),
        }


def commit_pair(
    zip_blob: bytes,
    detached_blob: bytes,
    delivery: Path,
    detached: Path,
    inject: str | None = None,
) -> None:
    if delivery.exists() or detached.exists():
        raise FileExistsError("never-overwrite final target exists")
    temp_dir = Path(tempfile.mkdtemp(prefix="step10a_v11_gate_commit_", dir=delivery.parent))
    temp_zip = temp_dir / delivery.name
    temp_detached = temp_dir / detached.name
    created: list[Path] = []
    committed = False
    try:
        temp_zip.write_bytes(zip_blob)
        temp_detached.write_bytes(detached_blob)
        if inject == "BEFORE_FINAL_RENAME":
            raise RuntimeError(inject)
        temp_zip.replace(delivery)
        created.append(delivery)
        if inject == "AFTER_DELIVERY_RENAME":
            raise RuntimeError(inject)
        temp_detached.replace(detached)
        created.append(detached)
        if inject == "AFTER_DETACHED_RENAME_BEFORE_COMMIT":
            raise RuntimeError(inject)
        committed = True
    finally:
        if not committed:
            for path in reversed(created):
                if path.exists():
                    path.unlink()
        shutil.rmtree(temp_dir, ignore_errors=True)


def transaction_regression() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for point in (
        "BEFORE_FINAL_RENAME",
        "AFTER_DELIVERY_RENAME",
        "AFTER_DETACHED_RENAME_BEFORE_COMMIT",
    ):
        root = Path(tempfile.mkdtemp(prefix="step10a_v11_gate_injection_", dir=ROOT / "Validation"))
        delivery = root / "gate.zip"
        detached = root / "gate.zip.sha256"
        sentinel = root / "sentinel.txt"
        sentinel.write_bytes(b"pre-existing\n")
        sentinel_sha = sha(sentinel.read_bytes())
        observed = False
        try:
            commit_pair(b"zip", b"checksum", delivery, detached, point)
        except RuntimeError as error:
            observed = str(error) == point
        row = {
            "injection_point": point,
            "expected_exception_observed": observed,
            "remaining_final_outputs": int(delivery.exists()) + int(detached.exists()),
            "remaining_temporary_outputs": len(
                [p for p in root.iterdir() if p.is_dir() and p.name.startswith("step10a_v11_gate_commit_")]
            ),
            "sentinel_unchanged": sentinel.exists() and sha(sentinel.read_bytes()) == sentinel_sha,
        }
        row["status"] = "PASS" if (
            row["expected_exception_observed"]
            and row["remaining_final_outputs"] == 0
            and row["remaining_temporary_outputs"] == 0
            and row["sentinel_unchanged"]
        ) else "FAIL"
        rows.append(row)
        shutil.rmtree(root)

    root = Path(tempfile.mkdtemp(prefix="step10a_v11_gate_never_overwrite_", dir=ROOT / "Validation"))
    delivery = root / "gate.zip"
    detached = root / "gate.zip.sha256"
    delivery.write_bytes(b"existing")
    before = sha(delivery.read_bytes())
    rejected = False
    try:
        commit_pair(b"zip", b"checksum", delivery, detached)
    except FileExistsError:
        rejected = True
    never_overwrite = rejected and sha(delivery.read_bytes()) == before and not detached.exists()
    shutil.rmtree(root)
    return {
        "production_commit_function": "commit_pair",
        "rows": rows,
        "failure_injection_pass": all(row["status"] == "PASS" for row in rows),
        "never_overwrite_pass": never_overwrite,
        "status": "PASS" if all(row["status"] == "PASS" for row in rows) and never_overwrite else "FAIL",
    }


def property_value(case: ET.Element, name: str) -> str:
    for item in case.findall("./properties/property"):
        if item.attrib.get("name") == name:
            return item.attrib.get("value", "")
    return ""


def parse_tests() -> dict[str, object]:
    path = RESULT / "TestResults.xml"
    root = ET.parse(path).getroot()
    cases = root.findall(".//test-case")
    selected = [case for case in cases if case.attrib.get("classname", "").split(".")[-1] in EXPECTED_FIXTURES]
    fixture_rows: list[dict[str, object]] = []
    for fixture, expected in EXPECTED_FIXTURES.items():
        rows = [case for case in selected if case.attrib.get("classname", "").endswith("." + fixture)]
        fixture_rows.append({
            "fixture": fixture,
            "expected": expected,
            "actual": len(rows),
            "passed": sum(case.attrib.get("result") == "Passed" for case in rows),
            "failed": sum(case.attrib.get("result") == "Failed" for case in rows),
            "skipped": sum(case.attrib.get("result") == "Skipped" for case in rows),
            "inconclusive": sum(case.attrib.get("result") == "Inconclusive" for case in rows),
        })
    counts = Counter(case.attrib.get("result", "") for case in selected)
    names = [case.attrib.get("fullname", "") for case in selected]
    retries = sum(int(property_value(case, "retryIteration") or "0") != 0 for case in selected)
    repeats = sum(int(property_value(case, "repeatIteration") or "0") != 0 for case in selected)
    failures = []
    for case in selected:
        if case.attrib.get("result") != "Failed":
            continue
        message = case.findtext("./failure/message", default="").strip()
        stack = case.findtext("./failure/stack-trace", default="").strip()
        failures.append({
            "fullname": case.attrib.get("fullname"),
            "message": message,
            "stack_trace": stack,
            "retry_iteration": int(property_value(case, "retryIteration") or "0"),
            "repeat_iteration": int(property_value(case, "repeatIteration") or "0"),
        })
    expected_first_run = (
        len(selected) == 87
        and len(set(names)) == 87
        and counts["Passed"] == 86
        and counts["Failed"] == 1
        and counts["Skipped"] == 0
        and counts["Inconclusive"] == 0
        and retries == 0
        and repeats == 0
        and len(failures) == 1
        and failures[0]["fullname"] == EXPECTED_FAILURE
        and "Expected: 1" in failures[0]["message"]
        and "But was:  2" in failures[0]["message"]
    )
    return {
        "schema_version": "WordsInSilence.Step10A.v1.1.GuiGate.TestResults.v1",
        "gate_status": "FAIL",
        "evidence_consistency": "PASS" if expected_first_run else "FAIL",
        "platform": "EditMode",
        "execution_count": 1,
        "testcase_count": len(selected),
        "unique_testcase_count": len(set(names)),
        "passed": counts["Passed"],
        "failed": counts["Failed"],
        "skipped": counts["Skipped"],
        "inconclusive": counts["Inconclusive"],
        "retry_nonzero": retries,
        "repeat_nonzero": repeats,
        "start_time_utc": root.attrib.get("start-time"),
        "end_time_utc": root.attrib.get("end-time"),
        "duration_seconds": root.attrib.get("duration"),
        "fixtures": fixture_rows,
        "failures": failures,
        "xml": file_record(path, RESULT),
        "screen": file_record(RESULT / "TestRunner_FIRST_FAIL.png", RESULT),
    }


def parse_editor_log() -> dict[str, object]:
    path = RESULT / "Editor.log"
    text = path.read_text(encoding="utf-8", errors="replace")
    counts = {
        "unity_6000_3_14f1": text.count("6000.3.14f1"),
        "tundra_build_success": text.count("Tundra build success"),
        "compiler_error_cs": len(re.findall(r"\berror CS\d+", text)),
        "scripts_have_compiler_errors": text.count("Scripts have compiler errors"),
        "compilation_failed": text.count("Compilation failed"),
        "prebuild": text.count("Executing IPrebuildSetup"),
        "postbuild": text.count("Executing IPostBuildCleanup"),
        "default_test_result_save": len(re.findall(r"Saving results to: .*TestResults\.xml", text)),
        "enter_playmode": len(re.findall(r"Entering PlayMode|EnterPlayMode", text)),
        "exit_playmode": len(re.findall(r"Exiting PlayMode|ExitPlayMode", text)),
    }
    compiler_errors = (
        counts["compiler_error_cs"]
        + counts["scripts_have_compiler_errors"]
        + counts["compilation_failed"]
    )
    consistent = (
        counts["unity_6000_3_14f1"] > 0
        and counts["tundra_build_success"] >= 1
        and compiler_errors == 0
        and counts["prebuild"] == 1
        and counts["postbuild"] == 1
        and counts["default_test_result_save"] == 1
        and counts["enter_playmode"] == 0
        and counts["exit_playmode"] == 0
    )
    return {
        "schema_version": "WordsInSilence.Step10A.v1.1.GuiGate.EditorLog.v1",
        "status": "PASS" if consistent else "FAIL",
        "unity_version": "6000.3.14f1",
        "compiler_error_count": compiler_errors,
        "counts": counts,
        "same_test_execution_link": "default TestResults.xml save appears exactly once",
        "playmode_execution_count": 0,
        "log": file_record(path, RESULT),
    }


def snapshot_expected(before_name: str, after_name: str) -> dict[str, object]:
    before = json.loads((RESULT / before_name).read_text(encoding="utf-8"))
    rows = []
    for expected in before["records"]:
        path = ROOT / expected["relative_path"]
        current = file_record(path)
        current["before_size_bytes"] = expected["size_bytes"]
        current["before_sha256"] = expected["sha256"]
        current["before_after_byte_identical"] = (
            current["size_bytes"] == expected["size_bytes"]
            and current["sha256"] == expected["sha256"]
        )
        current["status"] = "PASS" if current["before_after_byte_identical"] else "FAIL"
        rows.append(current)
    result = {
        "schema_version": before["schema_version"].replace("Before", "After"),
        "file_count": len(rows),
        "pass_count": sum(row["status"] == "PASS" for row in rows),
        "mismatch_count": sum(row["status"] != "PASS" for row in rows),
        "records": rows,
    }
    result["status"] = "PASS" if result["mismatch_count"] == 0 else "FAIL"
    write_json(RESULT / after_name, result)
    return result


def snapshot_profile_after() -> dict[str, object]:
    before = json.loads((RESULT / "active_candidate_profile_before.json").read_text(encoding="utf-8"))
    rows = []
    for expected in before["records"]:
        current = file_record(ROOT / expected["relative_path"])
        current["before_size_bytes"] = expected["size_bytes"]
        current["before_sha256"] = expected["sha256"]
        current["before_after_byte_identical"] = (
            current["size_bytes"] == expected["size_bytes"]
            and current["sha256"] == expected["sha256"]
        )
        current["status"] = "PASS" if current["before_after_byte_identical"] else "FAIL"
        rows.append(current)
    result = {
        "schema_version": "WordsInSilence.Step10A.v1.1.GuiGate.ActiveProfileAfter.v1",
        "profile_id": before["profile_id"],
        "file_count": len(rows),
        "prototype_count": before["prototype_count"],
        "threshold_count": before["threshold_count"],
        "thresholds": before["thresholds"],
        "pass_count": sum(row["status"] == "PASS" for row in rows),
        "mismatch_count": sum(row["status"] != "PASS" for row in rows),
        "records": rows,
    }
    result["status"] = "PASS" if result["mismatch_count"] == 0 else "FAIL"
    write_json(RESULT / "active_candidate_profile_after.json", result)
    return result


def snapshot_git() -> dict[str, object]:
    completed = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    files = [line for line in completed.stdout.splitlines() if line]
    result = {
        "schema_version": "WordsInSilence.Step10A.v1.1.GuiGate.GitStagedAfter.v1",
        "staged_count": len(files),
        "staged_files": files,
        "status": "PASS" if not files else "FAIL",
    }
    write_json(RESULT / "git_staged_after.json", result)
    return result


def add_assertion(rows: list[dict[str, object]], key: str, status: str, detail: str) -> None:
    rows.append({"assertion_key": key, "status": status, "detail": detail})


def build_report(
    tests: dict[str, object],
    log: dict[str, object],
    project: dict[str, object],
    protected: dict[str, object],
    profile: dict[str, object],
    repository_before: dict[str, object],
    repository_after: dict[str, object],
    git_after: dict[str, object],
) -> str:
    return f"""# WordsInSilence Step 10A v1.1 GUI Unity Editor Gate Validation Report

## Outcome

- Unity version: `6000.3.14f1`
- Compiler errors: `{log['compiler_error_count']}` — **PASS**
- EditMode executions: `1`
- EditMode result: `{tests['passed']} PASS / {tests['failed']} FAIL / {tests['skipped']} skipped / {tests['inconclusive']} inconclusive` — **FAIL**
- Retry/repeat nonzero: `{tests['retry_nonzero']} / {tests['repeat_nonzero']}`
- PlayMode executions: `0` — **NOT_APPLICABLE after EditMode failure**
- Live camera, word outcomes, and candidate storage: **NOT_APPLICABLE**

## Confirmed first failure

`{EXPECTED_FAILURE}` failed at `PetRobotCandidateCameraStartupRouteV1Tests.cs:61` with `Expected: 1 / But was: 2`.

The assertion counts the substring `cameraDisplay: {{fileID: 447812208}}`. The fixed Scene contains one exact `cameraDisplay:` field and one `_cameraDisplay:` field, so the substring helper counts both. This is a test-only token-count collision. No production source, Scene, Prefab, profile, prototype, threshold, or candidate data was changed, and the test was not rerun.

## Preservation

- Step 10A project files: `{project['pass_count']}/{project['file_count']}` byte-identical before/after.
- Protected files: `{protected['pass_count']}/{protected['file_count']}` byte-identical before/after.
- Active candidate profile files: `{profile['pass_count']}/{profile['file_count']}` byte-identical; 9 prototypes and 3 thresholds preserved.
- Candidate repository: `{repository_before['candidate_count']} -> {repository_after['candidate_count']}` candidates, `{repository_after['file_count']}` files.
- Git staged files: `{git_after['staged_count']}`.

## Status

```text
STEP10A_V1_1_GUI_UNITY_EDITOR_COMPILE_PASS
STEP10A_V1_1_EDITMODE_FIRST_RUN_FAIL_86_OF_87
STEP10A_V1_1_SHARED_CAMERA_SOURCE_TEST_SUBSTRING_COUNT_COLLISION_CONFIRMED
STEP10A_V1_1_PLAYMODE_AND_LIVE_CANDIDATE_STORAGE_NOT_APPLICABLE
BLOCKED_PENDING_EXTERNAL_REVIEW
```
"""


def main() -> None:
    require(RESULT.is_dir(), "gate evidence directory missing")
    require(not DELIVERY.exists() and not DETACHED.exists(), "never-overwrite delivery target exists")
    required = [
        "fixed_input_preflight.json",
        "project_files_before.json",
        "protected_files_before.json",
        "active_candidate_profile_before.json",
        "candidate_repository_before.json",
        "git_staged_before.json",
        "preflight_before_snapshot.json",
        "contract_and_legacy_before.json",
        "TestRunner_FIRST_FAIL.png",
        "TestResults.xml",
        "Editor.log",
    ]
    for name in required:
        require((RESULT / name).is_file(), "required evidence missing: " + name)

    tests = parse_tests()
    log = parse_editor_log()
    require(tests["evidence_consistency"] == "PASS", "first-run XML is not the expected 86/87 evidence")
    require(log["status"] == "PASS", "Editor.log does not match the first-run gate")
    write_json(RESULT / "test_results_summary.json", tests)
    write_json(RESULT / "editor_log_summary.json", log)

    scene_path = ROOT / "Assets/_Project/Scenes/VerticalSlice/PetRobotSignPracticeVerticalSlice.unity"
    scene = scene_path.read_text(encoding="utf-8")
    exact_candidate = len(re.findall(r"(?m)^  cameraDisplay: \{fileID: 447812208\}$", scene))
    legacy_underscore = len(re.findall(r"(?m)^  _cameraDisplay: \{fileID: 447812208\}$", scene))
    substring_count = scene.count("cameraDisplay: {fileID: 447812208}")
    diagnosis = {
        "schema_version": "WordsInSilence.Step10A.v1.1.GuiGate.FailureDiagnosis.v1",
        "failed_test": EXPECTED_FAILURE,
        "source_line": "Assets/_Project/Tests/EditMode/PetRobotCandidateCameraStartupRouteV1Tests.cs:61",
        "expected": 1,
        "actual": 2,
        "assertion_token": "cameraDisplay: {fileID: 447812208}",
        "exact_candidate_field_count": exact_candidate,
        "legacy_underscore_field_count": legacy_underscore,
        "substring_count": substring_count,
        "cause": "test-only substring count matches both cameraDisplay and _cameraDisplay",
        "production_files_modified_for_diagnosis": 0,
        "test_rerun_count": 0,
        "status": "CONFIRMED",
    }
    require((exact_candidate, legacy_underscore, substring_count) == (1, 1, 2), "failure cause mismatch")
    write_json(RESULT / "editmode_failure_diagnosis.json", diagnosis)

    project = snapshot_expected("project_files_before.json", "project_files_after.json")
    protected = snapshot_expected("protected_files_before.json", "protected_files_after.json")
    profile = snapshot_profile_after()
    repository_before = json.loads((RESULT / "candidate_repository_before.json").read_text(encoding="utf-8"))
    repository_after = snapshot_tree(CANDIDATE_REPOSITORY)
    repository_after["schema_version"] = "WordsInSilence.Step10A.v1.1.GuiGate.CandidateRepositoryAfter.v1"
    repository_after["before_after_unchanged"] = repository_after["files"] == repository_before["files"]
    repository_after["status"] = "PASS" if repository_after["before_after_unchanged"] else "FAIL"
    write_json(RESULT / "candidate_repository_after.json", repository_after)
    git_after = snapshot_git()

    playmode = {
        "schema_version": "WordsInSilence.Step10A.v1.1.GuiGate.PlayModeLiveEvaluation.v1",
        "playmode_execution_count": 0,
        "camera_state": "NOT_APPLICABLE",
        "camera_frame_count": None,
        "provider_observation_count": None,
        "word_outcomes": {
            word: {"outcome": "NOT_APPLICABLE", "retry_count": 0, "storage": "NOT_APPLICABLE"}
            for word in ("GREETING", "LOVE", "THANK_YOU")
        },
        "candidate_count_before": repository_before["candidate_count"],
        "candidate_count_after": repository_after["candidate_count"],
        "reason": "EditMode first run failed; contract prohibits PlayMode entry",
        "status": "NOT_APPLICABLE",
    }
    write_json(RESULT / "playmode_live_gate_evaluation.json", playmode)

    transaction = transaction_regression()
    write_json(RESULT / "packaging_transaction_failure_injection_regression.json", transaction)

    assertions: list[dict[str, object]] = []
    fixed = json.loads((RESULT / "fixed_input_preflight.json").read_text(encoding="utf-8"))
    add_assertion(assertions, "fixed_inputs_preflight", fixed["status"], "v1.0, prior GUI evidence, and v1.1 ZIP fixed")
    add_assertion(assertions, "project_files_before_30", "PASS" if json.loads((RESULT / "project_files_before.json").read_text(encoding="utf-8"))["pass_count"] == 30 else "FAIL", "30/30 before")
    add_assertion(assertions, "unity_compiler_errors_zero", "PASS" if log["compiler_error_count"] == 0 else "FAIL", "compiler errors 0")
    add_assertion(assertions, "editmode_execution_exactly_one", "PASS" if tests["execution_count"] == 1 and log["counts"]["prebuild"] == 1 else "FAIL", "single first run")
    add_assertion(assertions, "editmode_unique_inventory_87", "PASS" if tests["unique_testcase_count"] == 87 else "FAIL", "87 unique testcases")
    add_assertion(assertions, "editmode_required_pass_87", "FAIL", "actual 86 PASS / 1 FAIL")
    add_assertion(assertions, "retry_repeat_zero", "PASS" if tests["retry_nonzero"] == tests["repeat_nonzero"] == 0 else "FAIL", "retry/repeat 0")
    add_assertion(assertions, "first_failure_identity", "PASS" if tests["failures"][0]["fullname"] == EXPECTED_FAILURE else "FAIL", EXPECTED_FAILURE)
    add_assertion(assertions, "failure_substring_collision_confirmed", "PASS" if diagnosis["status"] == "CONFIRMED" else "FAIL", "exact candidate 1 + underscore legacy 1 = substring 2")
    add_assertion(assertions, "test_runner_screen_preserved", "PASS", "manual visual review: 86 green, 1 red; full Test Runner visible")
    add_assertion(assertions, "full_editor_log_preserved", log["status"], "same-run result save, prebuild/postbuild 1")
    add_assertion(assertions, "playmode_not_entered_after_failure", "PASS" if log["playmode_execution_count"] == 0 else "FAIL", "PlayMode count 0")
    add_assertion(assertions, "camera_live_word_outcomes", "NOT_APPLICABLE", "EditMode gate failed")
    add_assertion(assertions, "candidate_storage_live_validation", "NOT_APPLICABLE", "EditMode gate failed")
    add_assertion(assertions, "candidate_repository_unchanged", repository_after["status"], f"{repository_before['candidate_count']} -> {repository_after['candidate_count']}")
    add_assertion(assertions, "project_files_after_30", project["status"], f"{project['pass_count']}/{project['file_count']}")
    add_assertion(assertions, "protected_files_after_106", protected["status"], f"{protected['pass_count']}/{protected['file_count']}")
    add_assertion(assertions, "active_profile_9_prototypes_3_thresholds", profile["status"], f"{profile['pass_count']}/{profile['file_count']}")
    add_assertion(assertions, "git_staged_zero", git_after["status"], f"staged {git_after['staged_count']}")
    add_assertion(assertions, "packaging_failure_injection", transaction["status"], "same commit_pair rollback 3/3 and never-overwrite")
    require(len({row["assertion_key"] for row in assertions}) == len(assertions), "duplicate assertion key")
    (RESULT / "ASSERTIONS.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in assertions),
        encoding="utf-8",
        newline="\n",
    )

    verdict = {
        "schema_version": "WordsInSilence.Step10A.v1.1.GuiGate.PrimaryVerdict.v1",
        "compiler_status": "PASS",
        "editmode_status": "FAIL",
        "test_execution_count": 1,
        "passed": 86,
        "failed": 1,
        "playmode_status": "NOT_APPLICABLE",
        "live_candidate_storage_status": "NOT_APPLICABLE",
        "candidate_count_before": repository_before["candidate_count"],
        "candidate_count_after": repository_after["candidate_count"],
        "primary_failure": EXPECTED_FAILURE,
        "primary_failure_cause": diagnosis["cause"],
        "status_strings": [
            "STEP10A_V1_1_GUI_UNITY_EDITOR_COMPILE_PASS",
            "STEP10A_V1_1_EDITMODE_FIRST_RUN_FAIL_86_OF_87",
            "STEP10A_V1_1_SHARED_CAMERA_SOURCE_TEST_SUBSTRING_COUNT_COLLISION_CONFIRMED",
            "STEP10A_V1_1_PLAYMODE_AND_LIVE_CANDIDATE_STORAGE_NOT_APPLICABLE",
            "BLOCKED_PENDING_EXTERNAL_REVIEW",
        ],
        "status": "FAIL",
    }
    write_json(RESULT / "primary_verdict.json", verdict)
    (RESULT / "VALIDATION_REPORT.md").write_text(
        build_report(tests, log, project, protected, profile, repository_before, repository_after, git_after),
        encoding="utf-8",
        newline="\n",
    )

    analysis = RESULT / "analysis"
    analysis.mkdir(exist_ok=True)
    shutil.copyfile(Path(__file__), analysis / Path(__file__).name)

    # Pre-commit deterministic and transaction evidence is included in the archive.
    package_probe_files = {
        path.relative_to(RESULT).as_posix(): path.read_bytes()
        for path in sorted(RESULT.rglob("*"))
        if path.is_file() and path.name != "output_file_manifest.json"
    }
    probe_a = deterministic_zip(package_probe_files)
    probe_b = deterministic_zip(package_probe_files)
    probe_validation = validate_zip(probe_a, package_probe_files)
    packaging = {
        "schema_version": "WordsInSilence.Step10A.v1.1.GuiGate.PackagingVerification.v1",
        "deterministic_memory_rebuild_byte_identical": probe_a == probe_b,
        "precommit_zip_validation": probe_validation,
        "transaction_failure_injection_status": transaction["status"],
        "never_overwrite": transaction["never_overwrite_pass"],
        "status": "PASS",
    }
    require(
        probe_a == probe_b
        and probe_validation["duplicate_count"] == 0
        and probe_validation["unsafe_path_count"] == 0
        and probe_validation["encrypted_entry_count"] == 0
        and probe_validation["symlink_entry_count"] == 0
        and probe_validation["crc_failure"] is None
        and transaction["status"] == "PASS",
        "precommit packaging verification failed",
    )
    write_json(RESULT / "packaging_verification.json", packaging)

    manifest_rows = []
    for path in sorted(candidate for candidate in RESULT.rglob("*") if candidate.is_file()):
        relative = path.relative_to(RESULT).as_posix()
        if relative == "output_file_manifest.json":
            continue
        data = path.read_bytes()
        manifest_rows.append({"relative_path": relative, "size_bytes": len(data), "sha256": sha(data)})
    manifest = {
        "schema_version": "WordsInSilence.Step10A.v1.1.GuiGate.OutputManifest.v1",
        "self_excluded": "output_file_manifest.json",
        "file_count": len(manifest_rows),
        "files": manifest_rows,
    }
    write_json(RESULT / "output_file_manifest.json", manifest)

    file_map = {
        path.relative_to(RESULT).as_posix(): path.read_bytes()
        for path in sorted(RESULT.rglob("*"))
        if path.is_file()
    }
    require(
        {row["relative_path"] for row in manifest_rows}
        == set(file_map) - {"output_file_manifest.json"},
        "manifest membership mismatch",
    )
    for row in manifest_rows:
        data = file_map[row["relative_path"]]
        require(len(data) == row["size_bytes"] and sha(data) == row["sha256"],
                "manifest identity mismatch: " + row["relative_path"])

    zip_a = deterministic_zip(file_map)
    zip_b = deterministic_zip(file_map)
    require(zip_a == zip_b, "deterministic ZIP memory rebuild mismatch")
    final_validation = validate_zip(zip_a, file_map)
    require(
        final_validation["duplicate_count"] == 0
        and final_validation["unsafe_path_count"] == 0
        and final_validation["encrypted_entry_count"] == 0
        and final_validation["symlink_entry_count"] == 0
        and final_validation["crc_failure"] is None
        and final_validation["membership_match"]
        and final_validation["payload_mismatch_count"] == 0
        and final_validation["fixed_timestamp_count"] == len(file_map),
        "final ZIP validation failed",
    )
    digest = sha(zip_a)
    detached_blob = f"{digest}  {DELIVERY.name}\n".encode("ascii")
    commit_pair(zip_a, detached_blob, DELIVERY, DETACHED)
    require(DELIVERY.read_bytes() == zip_a, "committed ZIP mismatch")
    require(DETACHED.read_bytes() == detached_blob, "committed detached checksum mismatch")

    print(json.dumps({
        "result_directory": str(RESULT),
        "result_file_count": len(file_map),
        "manifest_target_count": len(manifest_rows),
        "delivery": file_record(DELIVERY),
        "detached": file_record(DETACHED),
        "zip_validation": final_validation,
        "test_result": {
            "passed": tests["passed"],
            "failed": tests["failed"],
            "skipped": tests["skipped"],
            "inconclusive": tests["inconclusive"],
        },
        "primary_verdict": verdict["status"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
