#!/usr/bin/env python3
"""WordsInSilence Step 9I packaging transaction rollback repair v1.1.

This tool treats the externally reviewed Step 9I v1.0 Delivery and every
installed project file contained in its project_files/ subtree as immutable.
It repairs only the result-directory/Delivery/checksum commit transaction and
executes real failure injection against the same commit function used for the
production v1.1 package.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import tempfile
from typing import Any, Callable
import uuid
import zipfile


SCHEMA = "words_in_silence.step9i.personal_runtime_opt_in_integration.v1_1"
V1_0_REL = Path("Validation/WordsInSilence_step9i_personal_runtime_opt_in_integration_v1_0_delivery.zip")
V1_0_SIZE = 456_746
V1_0_SHA256 = "e126f42232756890ee1b7c4457f38694e5d43b8190e7bb6aaeaa94243a9967a7"
V1_0_DETACHED_SIZE = 142
V1_0_DETACHED_FILE_SHA256 = "18b85988a09ef61d0692184fae3a377643b5fe9173765aa42d8c0db614729f61"
V1_0_PROJECT_FILE_COUNT = 44

INJECTION_POINTS = (
    "BEFORE_FINAL_RENAME",
    "AFTER_RESULT_RENAME",
    "AFTER_DELIVERY_RENAME",
    "AFTER_DETACHED_RENAME_BEFORE_COMMIT",
)


class Step9IRepairError(RuntimeError):
    """A validation or packaging repair contract failure."""


class InjectedPackagingFailure(Step9IRepairError):
    """The expected exception raised by a real transaction injection point."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Step9IRepairError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def json_bytes(value: Any, *, pretty: bool = True) -> bytes:
    text = json.dumps(
        value,
        ensure_ascii=False,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
        allow_nan=False,
    )
    return (text + "\n").encode("utf-8")


def safe_zip_name(name: str) -> bool:
    pure = PurePosixPath(name)
    return (
        bool(name)
        and not name.startswith(("/", "\\"))
        and "\\" not in name
        and ":" not in pure.parts[0]
        and all(part not in ("", ".", "..") for part in pure.parts)
    )


def deterministic_zip_bytes(files: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(
        stream,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name in sorted(files):
            require(safe_zip_name(name), f"unsafe output path: {name}")
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


def inspect_zip(data: bytes, expected_names: set[str]) -> dict[str, Any]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        require(len(names) == len(set(names)), "ZIP duplicate entry")
        require(archive.testzip() is None, "ZIP CRC failure")
        require(set(names) == expected_names, "ZIP membership mismatch")
        require(all(safe_zip_name(name) for name in names), "ZIP unsafe path")
        require(not any(info.flag_bits & 1 for info in infos), "ZIP encrypted entry")
        require(
            not any(stat.S_ISLNK((info.external_attr >> 16) & 0xFFFF) for info in infos),
            "ZIP symlink entry",
        )
        return {
            "entry_count": len(names),
            "crc": "PASS",
            "duplicates": 0,
            "unsafe": 0,
            "symlinks": 0,
            "encrypted": 0,
            "membership": "PASS",
        }


def strict_detached_bytes(zip_bytes: bytes, delivery_name: str) -> bytes:
    return (sha256_bytes(zip_bytes) + "  " + delivery_name + "\n").encode("ascii")


def verify_strict_detached(data: bytes, zip_bytes: bytes, delivery_name: str) -> None:
    require(data == strict_detached_bytes(zip_bytes, delivery_name), "detached checksum mismatch")


def verify_v1_0_delivery(root: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    delivery = root / V1_0_REL
    detached = Path(str(delivery) + ".sha256")
    require(delivery.is_file(), f"fixed v1.0 Delivery missing: {delivery}")
    require(detached.is_file(), f"fixed v1.0 detached checksum missing: {detached}")
    delivery_bytes = delivery.read_bytes()
    detached_bytes = detached.read_bytes()
    require(len(delivery_bytes) == V1_0_SIZE, "fixed v1.0 Delivery size mismatch")
    require(sha256_bytes(delivery_bytes) == V1_0_SHA256, "fixed v1.0 Delivery SHA-256 mismatch")
    require(len(detached_bytes) == V1_0_DETACHED_SIZE, "fixed v1.0 detached size mismatch")
    require(
        sha256_bytes(detached_bytes) == V1_0_DETACHED_FILE_SHA256,
        "fixed v1.0 detached file SHA-256 mismatch",
    )
    verify_strict_detached(detached_bytes, delivery_bytes, delivery.name)

    with zipfile.ZipFile(io.BytesIO(delivery_bytes)) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        file_map = {name: archive.read(name) for name in names}
    require("output_file_manifest.json" in file_map, "v1.0 output manifest missing")
    manifest = json.loads(file_map["output_file_manifest.json"])
    records = manifest["outputs"]
    require(manifest["self_excluded"] == "output_file_manifest.json", "v1.0 manifest self exclusion mismatch")
    require(len(records) == 63, "v1.0 manifest target count mismatch")
    expected_names = {record["relative_path"] for record in records} | {"output_file_manifest.json"}
    safety = inspect_zip(delivery_bytes, expected_names)
    require(
        all(
            record["relative_path"] in file_map
            and len(file_map[record["relative_path"]]) == record["size_bytes"]
            and sha256_bytes(file_map[record["relative_path"]]) == record["sha256"]
            for record in records
        ),
        "v1.0 manifest size/hash mismatch",
    )
    regenerated = deterministic_zip_bytes(file_map)
    require(regenerated == delivery_bytes, "v1.0 deterministic ZIP regeneration mismatch")
    return (
        {
            "delivery": {
                "relative_path": V1_0_REL.as_posix(),
                "size_bytes": len(delivery_bytes),
                "sha256": sha256_bytes(delivery_bytes),
            },
            "detached_checksum": {
                "relative_path": V1_0_REL.as_posix() + ".sha256",
                "size_bytes": len(detached_bytes),
                "file_sha256": sha256_bytes(detached_bytes),
                "strict_syntax_filename_digest": True,
            },
            "archive": {
                **safety,
                "manifest_target_count": len(records),
                "manifest_size_hash_verified": True,
                "deterministic_regeneration_byte_identical": True,
            },
        },
        file_map,
    )


def current_project_path(root: Path, packaged_name: str) -> Path:
    prefix = "project_files/"
    require(packaged_name.startswith(prefix), f"not a project file: {packaged_name}")
    relative = packaged_name[len(prefix) :]
    if relative.startswith("production_source/"):
        return root / "Assets/_Project/Scripts/Recognition/Runtime" / relative[len("production_source/") :]
    if relative.startswith("test_source/"):
        return root / "Assets/_Project/Tests/EditMode" / relative[len("test_source/") :]
    if relative.startswith("personal_profile/"):
        item = relative[len("personal_profile/") :]
        parent = root / "Assets/StreamingAssets/RecognitionCoreV1/PersonalProfiles"
        if item == "PersonalProfiles_folder.meta":
            return Path(str(parent) + ".meta")
        profile = parent / "step9h_current_user_v1_0"
        if item == "profile_folder.meta":
            return Path(str(profile) + ".meta")
        return profile / item
    if relative.startswith("test_fixture/"):
        item = relative[len("test_fixture/") :]
        fixture = root / "Assets/_Project/Tests/EditMode/Fixtures/PersonalRuntimeOptInIntegrationV1"
        if item == "fixture_folder.meta":
            return Path(str(fixture) + ".meta")
        if item == "Queries_folder.meta":
            return Path(str(fixture / "Queries") + ".meta")
        return fixture / item
    raise Step9IRepairError(f"unmapped v1.0 project file: {packaged_name}")


def verify_preserved_project_files(
    root: Path,
    file_map: dict[str, bytes],
) -> tuple[dict[str, Any], dict[str, str]]:
    names = sorted(name for name in file_map if name.startswith("project_files/"))
    require(len(names) == V1_0_PROJECT_FILE_COUNT, "v1.0 project file count mismatch")
    records: list[dict[str, Any]] = []
    snapshot: dict[str, str] = {}
    for packaged_name in names:
        current = current_project_path(root, packaged_name)
        require(current.is_file(), f"preserved project file missing: {current}")
        current_bytes = current.read_bytes()
        approved_bytes = file_map[packaged_name]
        require(current_bytes == approved_bytes, f"preserved project file changed: {current}")
        relative = current.relative_to(root).as_posix()
        digest = sha256_bytes(current_bytes)
        snapshot[relative] = digest
        records.append(
            {
                "packaged_v1_0_path": packaged_name,
                "project_relative_path": relative,
                "size_bytes": len(current_bytes),
                "sha256": digest,
                "byte_identical": True,
            }
        )
    return (
        {
            "schema_version": SCHEMA + ".preserved_project_file_identity",
            "approved_v1_0_project_file_count": len(records),
            "current_project_file_count": len(records),
            "byte_identical_count": len(records),
            "mismatch_count": 0,
            "records": records,
        },
        snapshot,
    )


def project_snapshot(root: Path, identity: dict[str, Any]) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for record in identity["records"]:
        relative = record["project_relative_path"]
        path = root / PurePosixPath(relative)
        require(path.is_file(), f"project snapshot file missing: {path}")
        snapshot[relative] = sha256_file(path)
    return snapshot


def path_is_within(path: Path, allowed_root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(allowed_root.resolve(strict=True))
        return True
    except ValueError:
        return False


def remove_owned_path(path: Path, allowed_root: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    require(path_is_within(path, allowed_root), f"rollback target escapes allowed root: {path}")
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def commit_packaging_transaction(
    *,
    result_files: dict[str, bytes],
    delivery_bytes: bytes,
    detached_bytes: bytes,
    final_result: Path,
    final_delivery: Path,
    final_detached: Path,
    allowed_root: Path,
    injection_point: str | None = None,
    post_rename_validator: Callable[[Path, Path, Path], None] | None = None,
    transaction_token: str | None = None,
) -> None:
    """Atomically expose three outputs or recover every output owned by this call.

    This exact function is called both by production packaging and the four
    isolated failure-injection cases. Pre-existing final targets are rejected
    before any rename and are never added to the owned-final set.
    """

    if injection_point is not None:
        require(injection_point in INJECTION_POINTS, f"unknown injection point: {injection_point}")
    final_paths = (final_result, final_delivery, final_detached)
    for path in final_paths:
        require(path_is_within(path, allowed_root), f"transaction path escapes allowed root: {path}")
    existing = [str(path) for path in final_paths if path.exists() or path.is_symlink()]
    require(not existing, f"never-overwrite final target exists: {existing}")

    token = transaction_token or uuid.uuid4().hex
    require(token and all(character.isalnum() or character == "_" for character in token), "invalid transaction token")
    staged_result = allowed_root / (".step9i_v1_1_result_" + token)
    staged_delivery = allowed_root / (".step9i_v1_1_delivery_" + token + ".tmp")
    staged_detached = allowed_root / (".step9i_v1_1_detached_" + token + ".tmp")
    staged_paths = (staged_result, staged_delivery, staged_detached)
    for path in staged_paths:
        require(path_is_within(path, allowed_root), f"transaction temp path escapes allowed root: {path}")
        require(not path.exists() and not path.is_symlink(), f"transaction temp target exists: {path}")

    owned_finals: list[Path] = []
    owned_temporaries = list(staged_paths)
    committed = False

    def inject(point: str) -> None:
        if injection_point == point:
            raise InjectedPackagingFailure(point)

    try:
        staged_result.mkdir()
        for name, data in result_files.items():
            require(safe_zip_name(name), f"unsafe staged result path: {name}")
            path = staged_result / PurePosixPath(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        staged_delivery.write_bytes(delivery_bytes)
        staged_detached.write_bytes(detached_bytes)
        inject("BEFORE_FINAL_RENAME")
        staged_result.replace(final_result)
        owned_finals.append(final_result)
        inject("AFTER_RESULT_RENAME")
        staged_delivery.replace(final_delivery)
        owned_finals.append(final_delivery)
        inject("AFTER_DELIVERY_RENAME")
        staged_detached.replace(final_detached)
        owned_finals.append(final_detached)
        inject("AFTER_DETACHED_RENAME_BEFORE_COMMIT")
        if post_rename_validator is not None:
            post_rename_validator(final_result, final_delivery, final_detached)
        committed = True
    finally:
        if not committed:
            for path in reversed(owned_finals):
                remove_owned_path(path, allowed_root)
            for path in owned_temporaries:
                remove_owned_path(path, allowed_root)


def run_failure_injection_regression(
    root: Path,
    validation_root: Path,
    identity: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for injection_point in INJECTION_POINTS:
        with tempfile.TemporaryDirectory(prefix="step9i_v1_1_injection_", dir=validation_root) as temporary:
            sandbox = Path(temporary)
            sentinel = sandbox / "pre_existing_sentinel.bin"
            sentinel.write_bytes(("sentinel:" + injection_point).encode("ascii"))
            sentinel_before = sha256_file(sentinel)
            project_before = project_snapshot(root, identity)
            transaction_token = "injection_" + str(len(rows))
            staged = (
                sandbox / (".step9i_v1_1_result_" + transaction_token),
                sandbox / (".step9i_v1_1_delivery_" + transaction_token + ".tmp"),
                sandbox / (".step9i_v1_1_detached_" + transaction_token + ".tmp"),
            )
            finals = (
                sandbox / "final_result",
                sandbox / "final_delivery.zip",
                sandbox / "final_delivery.zip.sha256",
            )
            expected_exception_observed = False
            observed_exception_type: str | None = None
            try:
                commit_packaging_transaction(
                    result_files={"marker.json": b"{}\n"},
                    delivery_bytes=b"synthetic-delivery",
                    detached_bytes=b"synthetic-detached",
                    final_result=finals[0],
                    final_delivery=finals[1],
                    final_detached=finals[2],
                    allowed_root=sandbox,
                    injection_point=injection_point,
                    transaction_token=transaction_token,
                    post_rename_validator=lambda result, delivery, detached: require(
                        result.is_dir() and delivery.is_file() and detached.is_file(),
                        "synthetic post-rename validator missing final",
                    ),
                )
            except InjectedPackagingFailure as error:
                expected_exception_observed = str(error) == injection_point
                observed_exception_type = type(error).__name__
            project_after = project_snapshot(root, identity)
            changed_project_files = sorted(
                key for key in project_before if project_before[key] != project_after[key]
            )
            remaining_finals = sum(path.exists() or path.is_symlink() for path in finals)
            remaining_temporaries = sum(path.exists() or path.is_symlink() for path in staged)
            sentinel_changed = not sentinel.is_file() or sha256_file(sentinel) != sentinel_before
            passed = (
                expected_exception_observed
                and remaining_finals == 0
                and remaining_temporaries == 0
                and not sentinel_changed
                and len(changed_project_files) == 0
            )
            rows.append(
                {
                    "injection_point": injection_point,
                    "transaction_function": commit_packaging_transaction.__name__,
                    "expected_exception_observed": expected_exception_observed,
                    "observed_exception_type": observed_exception_type,
                    "remaining_final_outputs": remaining_finals,
                    "remaining_temporary_outputs": remaining_temporaries,
                    "pre_existing_sentinel_changed": sentinel_changed,
                    "project_files_changed": len(changed_project_files),
                    "changed_project_file_paths": changed_project_files,
                    "verdict": "PASS" if passed else "FAIL",
                }
            )
    pass_count = sum(row["verdict"] == "PASS" for row in rows)
    return {
        "schema_version": SCHEMA + ".packaging_transaction_failure_injection_regression",
        "production_transaction_function": commit_packaging_transaction.__name__,
        "synthetic_transaction_function": commit_packaging_transaction.__name__,
        "same_function_used": True,
        "injection_point_count": len(rows),
        "pass_count": pass_count,
        "fail_count": len(rows) - pass_count,
        "expected_exception_count": sum(row["expected_exception_observed"] for row in rows),
        "remaining_final_output_count": sum(row["remaining_final_outputs"] for row in rows),
        "remaining_temporary_output_count": sum(row["remaining_temporary_outputs"] for row in rows),
        "sentinel_change_count": sum(row["pre_existing_sentinel_changed"] for row in rows),
        "project_file_change_count": sum(row["project_files_changed"] for row in rows),
        "rows": rows,
        "verdict": "PASS" if pass_count == len(INJECTION_POINTS) else "FAIL",
    }


def run_never_overwrite_regression(validation_root: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="step9i_v1_1_never_overwrite_", dir=validation_root) as temporary:
        sandbox = Path(temporary)
        final_result = sandbox / "final_result"
        final_result.mkdir()
        marker = final_result / "pre_existing.bin"
        marker.write_bytes(b"pre-existing-must-remain")
        before = sha256_file(marker)
        final_delivery = sandbox / "final_delivery.zip"
        final_detached = sandbox / "final_delivery.zip.sha256"
        expected_exception_observed = False
        try:
            commit_packaging_transaction(
                result_files={"marker.json": b"{}\n"},
                delivery_bytes=b"synthetic-delivery",
                detached_bytes=b"synthetic-detached",
                final_result=final_result,
                final_delivery=final_delivery,
                final_detached=final_detached,
                allowed_root=sandbox,
                transaction_token="never_overwrite",
            )
        except Step9IRepairError as error:
            expected_exception_observed = "never-overwrite" in str(error)
        unchanged = marker.is_file() and sha256_file(marker) == before
        additional_final_count = sum(path.exists() for path in (final_delivery, final_detached))
        temporary_output_count = sum(1 for _ in sandbox.glob(".step9i_v1_1_*"))
        passed = expected_exception_observed and unchanged and additional_final_count == 0 and temporary_output_count == 0
        return {
            "expected_exception_observed": expected_exception_observed,
            "pre_existing_final_unchanged": unchanged,
            "additional_final_output_count": additional_final_count,
            "temporary_output_count": temporary_output_count,
            "verdict": "PASS" if passed else "FAIL",
        }


def assertion(key: str, status: str, subject: str, evidence: Any) -> dict[str, Any]:
    return {
        "assertion_key": key,
        "status": status,
        "subject": subject,
        "evidence": evidence,
    }


def output_category(name: str) -> str:
    if name == "ASSERTIONS.jsonl":
        return "assertions"
    if name == "VALIDATION_REPORT.md":
        return "validation_report"
    if name.startswith("analysis/"):
        return "analysis_source"
    if name.startswith("input_subset/"):
        return "input_subset"
    if name.startswith("preserved_project_files/"):
        return "preserved_project_file"
    return "validation_result"


def report_text(
    regression: dict[str, Any],
    identity: dict[str, Any],
    assertions: dict[str, int],
) -> str:
    return "\n".join(
        [
            "# WordsInSilence Step 9I Packaging Transaction Rollback Repair v1.1",
            "",
            "## Outcome",
            "",
            "- v1.0 runtime source, personal assets, fixtures and calculation outputs: preserved byte-identical",
            "- Production packaging and synthetic injection function: `commit_packaging_transaction`",
            "- Unity Editor and EditMode: not executed; prior status remains unverified",
            "",
            "## Failure injection",
            "",
            f"- Injection points: `{regression['injection_point_count']}`",
            f"- PASS: `{regression['pass_count']}`",
            f"- Expected exceptions: `{regression['expected_exception_count']}`",
            f"- Remaining final outputs: `{regression['remaining_final_output_count']}`",
            f"- Remaining temporary outputs: `{regression['remaining_temporary_output_count']}`",
            f"- Sentinel changes: `{regression['sentinel_change_count']}`",
            f"- Project-file changes: `{regression['project_file_change_count']}`",
            "",
            "## Preservation",
            "",
            f"- v1.0 project files checked: `{identity['current_project_file_count']}`",
            f"- Byte-identical: `{identity['byte_identical_count']}`",
            f"- Mismatches: `{identity['mismatch_count']}`",
            "",
            "## Assertions",
            "",
            f"- PASS: `{assertions['PASS']}`",
            f"- FAIL: `{assertions['FAIL']}`",
            f"- UNVERIFIED: `{assertions['UNVERIFIED']}`",
            f"- NOT_APPLICABLE: `{assertions['NOT_APPLICABLE']}`",
            "",
        ]
    )


def build(root: Path, output: Path, delivery: Path) -> dict[str, Any]:
    detached = Path(str(delivery) + ".sha256")
    validation_root = (root / "Validation").resolve()
    require(output.parent.resolve() == validation_root, "v1.1 output must be directly under Validation")
    require(delivery.parent.resolve() == validation_root, "v1.1 Delivery must be directly under Validation")
    for target in (output, delivery, detached):
        require(not target.exists(), f"never-overwrite output target exists: {target}")
    staged_git = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    require(not staged_git, f"Git staged files present: {staged_git}")

    fixed, v1_file_map = verify_v1_0_delivery(root)
    identity, original_project_snapshot = verify_preserved_project_files(root, v1_file_map)
    fixed_before = {
        V1_0_REL.as_posix(): sha256_file(root / V1_0_REL),
        V1_0_REL.as_posix() + ".sha256": sha256_file(Path(str(root / V1_0_REL) + ".sha256")),
    }
    regression = run_failure_injection_regression(root, validation_root, identity)
    require(regression["verdict"] == "PASS", "failure-injection regression failed")
    never_overwrite = run_never_overwrite_regression(validation_root)
    require(never_overwrite["verdict"] == "PASS", "never-overwrite regression failed")
    require(project_snapshot(root, identity) == original_project_snapshot, "project files changed during regression")

    contract = {
        "schema_version": SCHEMA + ".packaging_transaction_contract",
        "transaction_function": commit_packaging_transaction.__name__,
        "production_and_synthetic_share_same_function": True,
        "final_targets": ["result_directory", "delivery_zip", "detached_checksum"],
        "precondition": "all three final targets absent",
        "owned_temporary_and_final_paths_explicitly_tracked": True,
        "rollback_before_commit_removes_owned_final_and_temporary_paths": True,
        "pre_existing_paths_never_added_to_owned_paths": True,
        "committed_only_after_all_renames_and_post_rename_validation": True,
        "injection_points": list(INJECTION_POINTS),
    }
    primary = {
        "schema_version": SCHEMA + ".primary_verdict",
        "actual_primary_verdict": "SOURCE_AND_PACKAGE_READY",
        "v1_0_runtime_source_and_calculation_preserved": True,
        "packaging_transaction_failure_injection": regression["verdict"],
        "unity_compile_and_editmode_tests": "UNVERIFIED",
        "status_strings": [
            "STEP9I_PERSONAL_RUNTIME_OPT_IN_INTEGRATION_V1_1_SOURCE_AND_PACKAGE_READY",
            "STEP9I_V1_0_RUNTIME_SOURCE_AND_CALCULATION_PRESERVED",
            "STEP9I_PACKAGING_TRANSACTION_FAILURE_INJECTION_PASS",
            "UNITY_COMPILE_AND_EDITMODE_TESTS_UNVERIFIED",
            "BLOCKED_PENDING_EXTERNAL_REVIEW",
        ],
    }
    input_provenance = {
        "schema_version": SCHEMA + ".input_provenance",
        "fixed_step9i_v1_0": fixed,
        "v1_0_external_review_scope": {
            "runtime_source_and_personal_calculation": "PASS",
            "packaging_transaction_rollback": "FAIL_REPAIRED_BY_V1_1",
            "unity_compile_and_editmode": "UNVERIFIED",
        },
        "v1_0_delivery_and_detached_before_after_unchanged": True,
    }

    rollback_rows_pass = all(
        row["expected_exception_observed"]
        and row["remaining_final_outputs"] == 0
        and row["remaining_temporary_outputs"] == 0
        and not row["pre_existing_sentinel_changed"]
        and row["project_files_changed"] == 0
        and row["verdict"] == "PASS"
        for row in regression["rows"]
    )
    assertions = [
        assertion("fixed_v1_0_delivery_identity", "PASS", V1_0_REL.as_posix(), fixed["delivery"]),
        assertion("fixed_v1_0_detached_strict_checksum", "PASS", V1_0_REL.as_posix() + ".sha256", fixed["detached_checksum"]),
        assertion("fixed_v1_0_archive_safety_manifest_determinism", "PASS", "v1.0 Delivery", fixed["archive"]),
        assertion("v1_0_project_files_byte_identical", "PASS", "44 preserved files", {"checked": 44, "mismatch_count": 0}),
        assertion("production_test_csharp_preserved", "PASS", "approved C# source", True),
        assertion("personal_profile_assets_and_meta_preserved", "PASS", "approved profile", True),
        assertion("query_and_expected_fixtures_preserved", "PASS", "approved EditMode fixture", True),
        assertion("shared_production_injection_transaction_function", "PASS" if regression["same_function_used"] else "FAIL", commit_packaging_transaction.__name__, regression["same_function_used"]),
        assertion("failure_injection_all_four_expected_exceptions", "PASS" if regression["expected_exception_count"] == len(INJECTION_POINTS) else "FAIL", "four isolated transactions", regression["expected_exception_count"]),
        assertion("injected_failure_partial_output_zero", "PASS" if rollback_rows_pass else "FAIL", "actual failure-injection rows", {"derived_from_rows": rollback_rows_pass, "remaining_final": regression["remaining_final_output_count"], "remaining_temp": regression["remaining_temporary_output_count"]}),
        assertion("failure_injection_sentinels_unchanged", "PASS" if regression["sentinel_change_count"] == 0 else "FAIL", "four pre-existing sentinels", regression["sentinel_change_count"]),
        assertion("failure_injection_project_files_unchanged", "PASS" if regression["project_file_change_count"] == 0 else "FAIL", "44 project files x four injections", regression["project_file_change_count"]),
        assertion("never_overwrite_real_regression", "PASS" if never_overwrite["verdict"] == "PASS" else "FAIL", "pre-existing final target", never_overwrite),
        assertion("unity_compile", "UNVERIFIED", "Unity Editor", "not executed in v1.1 repair scope"),
        assertion("editmode_personal_runtime_integration", "UNVERIFIED", "PersonalRuntimeOptInIntegrationV1Tests", "not executed in v1.1 repair scope"),
        assertion("git_staged_zero", "PASS", "repository", 0),
    ]
    require(rollback_rows_pass, "injected failure assertion evidence is not fully PASS")
    require(len({record["assertion_key"] for record in assertions}) == len(assertions), "duplicate assertion key")
    assertion_summary = {
        status: sum(record["status"] == status for record in assertions)
        for status in ("PASS", "FAIL", "UNVERIFIED", "NOT_APPLICABLE")
    }
    require(assertion_summary["FAIL"] == 0 and assertion_summary["UNVERIFIED"] == 2, "assertion branch mismatch")

    result_files: dict[str, bytes] = {
        "input_provenance.json": json_bytes(input_provenance),
        "preserved_project_file_identity.json": json_bytes(identity),
        "packaging_transaction_contract.json": json_bytes(contract),
        "packaging_transaction_failure_injection_regression.json": json_bytes(regression),
        "never_overwrite_regression.json": json_bytes(never_overwrite),
        "primary_verdict.json": json_bytes(primary),
        "ASSERTIONS.jsonl": b"".join(json_bytes(record, pretty=False) for record in assertions),
        "VALIDATION_REPORT.md": report_text(regression, identity, assertion_summary).encode("utf-8"),
        "analysis/s001_step9i_personal_runtime_opt_in_integration_v1_1.py": Path(__file__).read_bytes(),
        "input_subset/v1_0_provenance/output_file_manifest.json": v1_file_map["output_file_manifest.json"],
        "input_subset/v1_0_provenance/source_change_inventory.json": v1_file_map["source_change_inventory.json"],
        "input_subset/v1_0_provenance/primary_verdict.json": v1_file_map["primary_verdict.json"],
        "input_subset/v1_0_provenance/personal_runtime_expected_scores.json": v1_file_map["personal_runtime_expected_scores.json"],
        "input_subset/v1_0_provenance/personal_runtime_expected_predictions.json": v1_file_map["personal_runtime_expected_predictions.json"],
        "input_subset/v1_0_provenance/global_regression_status.json": v1_file_map["global_regression_status.json"],
        "input_subset/v1_0_provenance/offline_csharp_compile_check.json": v1_file_map["offline_csharp_compile_check.json"],
        "input_subset/v1_0_provenance/unity_compile_editmode_status.json": v1_file_map["unity_compile_editmode_status.json"],
    }
    for name, data in sorted(v1_file_map.items()):
        if name.startswith("project_files/"):
            preserved_name = "preserved_project_files/" + name[len("project_files/") :]
            require(preserved_name not in result_files, f"preserved project file collision: {preserved_name}")
            result_files[preserved_name] = data

    manifest_records = [
        {
            "relative_path": name,
            "category": output_category(name),
            "size_bytes": len(data),
            "sha256": sha256_bytes(data),
        }
        for name, data in sorted(result_files.items())
    ]
    result_files["output_file_manifest.json"] = json_bytes(
        {
            "schema_version": SCHEMA + ".output_file_manifest",
            "self_excluded": "output_file_manifest.json",
            "output_count_excluding_manifest": len(manifest_records),
            "outputs": manifest_records,
        }
    )
    for name, data in result_files.items():
        if name.endswith(".json"):
            json.loads(data, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    zip_one = deterministic_zip_bytes(result_files)
    zip_two = deterministic_zip_bytes(result_files)
    require(zip_one == zip_two, "v1.1 deterministic ZIP memory regeneration mismatch")
    zip_inspection = inspect_zip(zip_one, set(result_files))
    detached_bytes = strict_detached_bytes(zip_one, delivery.name)

    def post_rename_validator(final_result: Path, final_delivery: Path, final_detached: Path) -> None:
        actual_names = {
            path.relative_to(final_result).as_posix()
            for path in final_result.rglob("*")
            if path.is_file()
        }
        require(actual_names == set(result_files), "post-rename result membership mismatch")
        require(
            all((final_result / PurePosixPath(name)).read_bytes() == data for name, data in result_files.items()),
            "post-rename result byte mismatch",
        )
        written_zip = final_delivery.read_bytes()
        require(written_zip == zip_one, "post-rename Delivery byte mismatch")
        inspect_zip(written_zip, set(result_files))
        verify_strict_detached(final_detached.read_bytes(), written_zip, final_delivery.name)
        require(project_snapshot(root, identity) == original_project_snapshot, "project files changed during production commit")
        require(
            fixed_before
            == {
                V1_0_REL.as_posix(): sha256_file(root / V1_0_REL),
                V1_0_REL.as_posix() + ".sha256": sha256_file(Path(str(root / V1_0_REL) + ".sha256")),
            },
            "v1.0 input changed during production commit",
        )
        staged_during_commit = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        require(not staged_during_commit, "Git staged files changed during production commit")

    commit_packaging_transaction(
        result_files=result_files,
        delivery_bytes=zip_one,
        detached_bytes=detached_bytes,
        final_result=output,
        final_delivery=delivery,
        final_detached=detached,
        allowed_root=validation_root,
        post_rename_validator=post_rename_validator,
    )
    return {
        "status": "PASS",
        "actual_primary_verdict": primary["actual_primary_verdict"],
        "preserved_project_file_count": identity["byte_identical_count"],
        "failure_injection": {
            "points": regression["injection_point_count"],
            "pass_count": regression["pass_count"],
            "remaining_final_outputs": regression["remaining_final_output_count"],
            "remaining_temporary_outputs": regression["remaining_temporary_output_count"],
            "sentinel_changes": regression["sentinel_change_count"],
            "project_file_changes": regression["project_file_change_count"],
        },
        "assertion_summary": assertion_summary,
        "unity_status": "UNITY_COMPILE_AND_EDITMODE_TESTS_UNVERIFIED",
        "output_directory": str(output),
        "output_file_count": len(result_files),
        "delivery_zip": str(delivery),
        "delivery_size_bytes": len(zip_one),
        "delivery_sha256": sha256_bytes(zip_one),
        "detached_checksum": str(detached),
        "detached_size_bytes": len(detached_bytes),
        "detached_file_sha256": sha256_bytes(detached_bytes),
        "zip_inspection": zip_inspection,
        "git_staged_count": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--delivery", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (
        args.output
        or root / "Validation/WordsInSilence_step9i_personal_runtime_opt_in_integration_v1_1"
    ).resolve()
    delivery = (
        args.delivery
        or root / "Validation/WordsInSilence_step9i_personal_runtime_opt_in_integration_v1_1_delivery.zip"
    ).resolve()
    result = build(root, output, delivery)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
