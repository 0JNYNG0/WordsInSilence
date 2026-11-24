#!/usr/bin/env python3
"""Install the externally approved Step 9L v1.1 project mapping once.

The fixed Delivery supplies exactly 54 mappings: one approved replacement and
53 never-overwrite files.  A failed transaction removes every owned new file,
restores the original runtime source bytes, and removes only directories that
this invocation created and left empty.
"""

from __future__ import annotations

import argparse
from collections import Counter
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


SCHEMA = "words_in_silence.step9l.personal_multi_prototype_runtime_candidate.install.v1_0"
DELIVERY_REL = Path("Validation/WordsInSilence_step9l_personal_multi_prototype_runtime_candidate_v1_1_delivery.zip")
DELIVERY_SIZE = 765_962
DELIVERY_SHA256 = "d3beede4797c4c70581b4a1f3f88af2dac25df57018f5da87cd4300b0e69bca6"
DETACHED_SIZE = 149
DETACHED_FILE_SHA256 = "73003baa37771891a6ef2acb2fcbb17b64ccd8547b81d7bc469c88ae62bef732"
DELIVERY_ENTRY_COUNT = 94
MANIFEST_TARGET_COUNT = 93

SOURCE_REL = Path("Assets/_Project/Scripts/Recognition/Runtime/PersonalRuntimeOptInProfile.cs")
SOURCE_BEFORE_SIZE = 54_894
SOURCE_BEFORE_SHA256 = "3e4579c153ad23ee85c00b01967956f0777a1092384e5e8ade8b57b9fa5798bc"
SOURCE_AFTER_SIZE = 55_677
SOURCE_AFTER_SHA256 = "468613729c0299d11a4c72343c17df0c4fc7ed5b36367c8ce86ea9d81b6b62cf"
PROFILE_ID = "step9l_current_user_multi_prototype_v1_0"
PROFILE_REL = Path("Assets/StreamingAssets/RecognitionCoreV1/PersonalProfiles") / PROFILE_ID
FIXTURE_REL = Path("Assets/_Project/Tests/EditMode/Fixtures/PersonalMultiPrototypeRuntimeIntegrationV1")
MULTI_SOURCE_REL = Path("Assets/_Project/Scripts/Recognition/Runtime/PersonalMultiPrototypeRuntimeProfile.cs")
TEST_REL = Path("Assets/_Project/Tests/EditMode/PersonalMultiPrototypeRuntimeIntegrationV1Tests.cs")

FIXED_ARCHIVES = (
    ("Validation/WordsInSilence_step9h_personalized_class_conditional_threshold_v1_0_delivery.zip", 3_214_867, "4827e81e029e63e7939e7ed91cb33ac5f90dd6822fa13722da01f642aed2ddb9"),
    ("Validation/WordsInSilence_step9h_personalized_class_conditional_threshold_v1_0_delivery.zip.sha256", 147, "d564d21cd5e6cf0dc1b3eda47cb88a80014710fd182d3889827b743961829c4b"),
    ("Validation/WordsInSilence_step9l_personal_multi_prototype_runtime_candidate_v1_0_delivery.zip", 719_243, "806ebed55c34590b8c7ed1579ecfff8d72192f20d668b7ccfba5daa79b6f09cf"),
    ("Validation/WordsInSilence_step9l_personal_multi_prototype_runtime_candidate_v1_0_delivery.zip.sha256", 149, "6292bcb9a5eb9d629d15d7e4bc9f9cd0332b5561a6a893d839d15229cdb70a24"),
    (DELIVERY_REL.as_posix(), DELIVERY_SIZE, DELIVERY_SHA256),
    (DELIVERY_REL.as_posix() + ".sha256", DETACHED_SIZE, DETACHED_FILE_SHA256),
)

INJECTION_POINTS = (
    "BEFORE_SOURCE_REPLACE",
    "AFTER_SOURCE_REPLACE",
    "AFTER_NEW_FILES",
    "AFTER_VALIDATION_BEFORE_COMMIT",
)


class InstallError(RuntimeError):
    """A fixed-input, precondition, transaction, or preservation failure."""


class InjectedInstallFailure(InstallError):
    """Expected synthetic failure raised by the production installer."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise InstallError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def safe_zip_name(name: str) -> bool:
    pure = PurePosixPath(name)
    return bool(name) and not name.startswith(("/", "\\")) and "\\" not in name and ":" not in pure.parts[0] and all(part not in ("", ".", "..") for part in pure.parts)


def file_record(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    require(path.is_file(), "required file missing: " + relative)
    payload = path.read_bytes()
    return {"relative_path": relative, "size_bytes": len(payload), "sha256": sha256_bytes(payload)}


def expected_records(items: tuple[tuple[str, int, str], ...]) -> list[dict[str, Any]]:
    return [{"relative_path": path, "size_bytes": size, "sha256": digest} for path, size, digest in items]


def verify_expected_records(root: Path, records: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    require(records, label + " baseline is empty")
    actual = [file_record(root, record["relative_path"]) for record in records]
    require(actual == records, label + " identity mismatch")
    return actual


def verify_fixed_delivery(root: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    path = root / DELIVERY_REL
    detached = Path(str(path) + ".sha256")
    require(path.is_file() and detached.is_file(), "fixed Step 9L v1.1 input missing")
    data = path.read_bytes()
    checksum = detached.read_bytes()
    require(len(data) == DELIVERY_SIZE and sha256_bytes(data) == DELIVERY_SHA256, "fixed Delivery identity mismatch")
    require(len(checksum) == DETACHED_SIZE and sha256_bytes(checksum) == DETACHED_FILE_SHA256, "fixed detached identity mismatch")
    require(checksum == (DELIVERY_SHA256 + "  " + path.name + "\n").encode("ascii"), "fixed detached syntax or digest mismatch")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        require(len(names) == len(set(names)) == DELIVERY_ENTRY_COUNT, "fixed ZIP entry or duplicate mismatch")
        require(archive.testzip() is None, "fixed ZIP CRC failure")
        require(all(safe_zip_name(name) for name in names), "fixed ZIP unsafe path")
        require(not any(info.flag_bits & 1 for info in infos), "fixed ZIP encrypted entry")
        require(not any(stat.S_ISLNK((info.external_attr >> 16) & 0xFFFF) for info in infos), "fixed ZIP symlink entry")
        file_map = {name: archive.read(name) for name in names}
    manifest = json.loads(file_map["output_file_manifest.json"])
    outputs = manifest["outputs"]
    require(len(outputs) == MANIFEST_TARGET_COUNT, "fixed manifest target count mismatch")
    require(set(file_map) == {record["relative_path"] for record in outputs} | {"output_file_manifest.json"}, "fixed manifest membership mismatch")
    for record in outputs:
        payload = file_map[record["relative_path"]]
        require(len(payload) == record["size_bytes"] and sha256_bytes(payload) == record["sha256"], "fixed manifest payload mismatch: " + record["relative_path"])
    return file_map, {
        "relative_path": DELIVERY_REL.as_posix(), "size_bytes": len(data), "sha256": sha256_bytes(data),
        "detached_size_bytes": len(checksum), "detached_file_sha256": sha256_bytes(checksum),
        "entry_count": len(file_map), "manifest_target_count": len(outputs),
        "crc": "PASS", "duplicate_count": 0, "unsafe_path_count": 0, "symlink_count": 0, "encrypted_entry_count": 0,
    }


def package_target(package_path: str) -> Path:
    if package_path == "project_files/source_after/PersonalRuntimeOptInProfile.cs":
        return SOURCE_REL
    if package_path.startswith("project_files/new_runtime_source/"):
        return SOURCE_REL.parent / Path(package_path).name
    if package_path.startswith("project_files/new_test_source/"):
        return TEST_REL.parent / Path(package_path).name
    if package_path == "project_files/candidate_profile/profile_folder.meta":
        return Path(str(PROFILE_REL) + ".meta")
    if package_path.startswith("project_files/candidate_profile/"):
        return PROFILE_REL / package_path.removeprefix("project_files/candidate_profile/")
    if package_path == "project_files/test_fixture/fixture_folder.meta":
        return Path(str(FIXTURE_REL) + ".meta")
    if package_path == "project_files/test_fixture/Queries_folder.meta":
        return Path(str(FIXTURE_REL / "Queries") + ".meta")
    if package_path.startswith("project_files/test_fixture/"):
        return FIXTURE_REL / package_path.removeprefix("project_files/test_fixture/")
    raise InstallError("unmapped approved package path: " + package_path)


def build_mappings(file_map: dict[str, bytes]) -> tuple[dict[str, bytes], list[dict[str, Any]]]:
    names = sorted(name for name in file_map if name.startswith("project_files/"))
    require(len(names) == 54, "approved project mapping count mismatch")
    mapped: dict[str, bytes] = {}
    records = []
    for package_path in names:
        target = package_target(package_path).as_posix()
        require(target not in mapped, "duplicate project target: " + target)
        payload = file_map[package_path]
        mapped[target] = payload
        records.append({
            "package_path": package_path, "project_relative_path": target,
            "operation": "replace_approved_source" if target == SOURCE_REL.as_posix() else "create_never_overwrite",
            "size_bytes": len(payload), "sha256": sha256_bytes(payload),
        })
    require(Counter(record["operation"] for record in records) == Counter({"replace_approved_source": 1, "create_never_overwrite": 53}), "approved mapping operation counts mismatch")
    require(len(mapped[SOURCE_REL.as_posix()]) == SOURCE_AFTER_SIZE and sha256_bytes(mapped[SOURCE_REL.as_posix()]) == SOURCE_AFTER_SHA256, "approved replacement source mismatch")
    return mapped, records


def validate_profile_contract(mapped: dict[str, bytes]) -> dict[str, Any]:
    manifest_path = (PROFILE_REL / "personal_runtime_profile_manifest.json").as_posix()
    config_path = (PROFILE_REL / "personal_multi_prototype_threshold_config.json").as_posix()
    manifest = json.loads(mapped[manifest_path])
    config = json.loads(mapped[config_path])
    require(manifest["profile_id"] == PROFILE_ID, "candidate profile ID mismatch")
    require(manifest["opt_in_only"] is True and manifest["default_active"] is False, "candidate manifest must be opt-in and default inactive")
    require(config["opt_in_only"] is True and config["runtime_activated"] is False, "candidate config must be opt-in and runtime inactive")
    return {
        "profile_id": PROFILE_ID, "opt_in_only": True, "default_active": False,
        "runtime_activated": False, "prototype_wisrt_count": sum(path.startswith(PROFILE_REL.as_posix() + "/") and path.endswith(".wisrt") for path in mapped),
    }


def load_protection_baselines(file_map: dict[str, bytes]) -> dict[str, list[dict[str, Any]]]:
    protection = json.loads(file_map["protected_files_before_after.json"])
    groups = {
        "global_runtime_assets": protection["global_runtime_assets"]["baseline"],
        "existing_personal_profile": protection["existing_personal_profile"]["baseline"],
        "approved_csharp": protection["approved_csharp"]["baseline"],
        "fixed_archives": expected_records(FIXED_ARCHIVES),
    }
    require(len(groups["global_runtime_assets"]) == 11, "global runtime baseline count mismatch")
    require(len(groups["existing_personal_profile"]) == 10, "existing personal profile baseline count mismatch")
    require(len(groups["approved_csharp"]) == 4, "approved C# baseline count mismatch")
    require(len(groups["fixed_archives"]) == 6, "fixed archive baseline count mismatch")
    return groups


def snapshot_protected(root: Path, baselines: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    groups = {}
    for name, expected in baselines.items():
        actual = verify_expected_records(root, expected, name)
        groups[name] = {"required_count": len(expected), "actual_count": len(actual), "mismatch_count": 0, "records": actual}
    return {"schema_version": SCHEMA + ".protected_snapshot", "groups": groups, "total_file_count": sum(group["actual_count"] for group in groups.values()), "total_mismatch_count": 0}


def ensure_preconditions(root: Path, mapped: dict[str, bytes]) -> bytes:
    source = root / SOURCE_REL
    require(source.is_file(), "approved source before file missing")
    before = source.read_bytes()
    require(len(before) == SOURCE_BEFORE_SIZE and sha256_bytes(before) == SOURCE_BEFORE_SHA256, "approved source before identity mismatch")
    collisions = [relative for relative in mapped if relative != SOURCE_REL.as_posix() and ((root / relative).exists() or (root / relative).is_symlink())]
    require(not collisions, "never-overwrite new target exists: " + repr(collisions))
    return before


def install_transaction(
    root: Path,
    mapped: dict[str, bytes],
    before_source: bytes,
    *,
    injection_point: str | None = None,
    after_validator: Callable[[], None] | None = None,
) -> dict[str, Any]:
    if injection_point is not None:
        require(injection_point in INJECTION_POINTS, "unknown install injection point")
    ensure_preconditions(root, mapped)
    source = root / SOURCE_REL
    new_relatives = sorted(relative for relative in mapped if relative != SOURCE_REL.as_posix())
    created_files: list[Path] = []
    created_dirs: list[Path] = []
    source_replaced = False
    committed = False

    def inject(point: str) -> None:
        if injection_point == point:
            raise InjectedInstallFailure(point)

    try:
        inject("BEFORE_SOURCE_REPLACE")
        source.write_bytes(mapped[SOURCE_REL.as_posix()])
        source_replaced = True
        inject("AFTER_SOURCE_REPLACE")
        for relative in new_relatives:
            path = root / relative
            missing = []
            parent = path.parent
            while parent != root and not parent.exists():
                missing.append(parent)
                parent = parent.parent
            for directory in reversed(missing):
                directory.mkdir()
                created_dirs.append(directory)
            require(not path.exists() and not path.is_symlink(), "new target appeared during transaction: " + relative)
            path.write_bytes(mapped[relative])
            created_files.append(path)
        inject("AFTER_NEW_FILES")
        for relative, expected in mapped.items():
            path = root / relative
            require(path.is_file() and path.read_bytes() == expected, "installed bytes mismatch: " + relative)
        if after_validator is not None:
            after_validator()
        inject("AFTER_VALIDATION_BEFORE_COMMIT")
        committed = True
        return {"replacement_count": 1, "new_file_count": len(created_files), "created_directory_count": len(created_dirs), "installed_file_count": len(mapped)}
    finally:
        if not committed:
            for path in reversed(created_files):
                if path.exists() or path.is_symlink():
                    path.unlink()
            if source_replaced:
                source.write_bytes(before_source)
            for directory in reversed(created_dirs):
                if directory.exists() and not any(directory.iterdir()):
                    directory.rmdir()


def transaction_regression(mapped: dict[str, bytes], before_source: bytes, temporary_parent: Path) -> dict[str, Any]:
    rows = []
    for point in INJECTION_POINTS:
        with tempfile.TemporaryDirectory(prefix=".s9ltx_", dir=temporary_parent) as temporary:
            root = Path(temporary)
            source = root / SOURCE_REL
            source.parent.mkdir(parents=True)
            source.write_bytes(before_source)
            failure = False
            try:
                install_transaction(root, mapped, before_source, injection_point=point, after_validator=lambda: None)
            except InjectedInstallFailure as error:
                failure = str(error) == point
            new_residue = [relative for relative in mapped if relative != SOURCE_REL.as_posix() and (root / relative).exists()]
            source_restored = source.read_bytes() == before_source
            temp_residue = [path.as_posix() for path in root.rglob("*.tmp")]
            passed = failure and not new_residue and source_restored and not temp_residue
            rows.append({"injection_point": point, "expected_exception_observed": failure, "remaining_new_files": len(new_residue), "original_source_restored": source_restored, "remaining_temporary_outputs": len(temp_residue), "verdict": "PASS" if passed else "FAIL"})
    require(all(row["verdict"] == "PASS" for row in rows), "install transaction failure regression failed")
    return {"schema_version": SCHEMA + ".transaction_failure_injection_regression", "same_production_function_used": True, "injection_count": len(rows), "pass_count": sum(row["verdict"] == "PASS" for row in rows), "fail_count": sum(row["verdict"] == "FAIL" for row in rows), "rows": rows}


def never_overwrite_regression(mapped: dict[str, bytes], before_source: bytes, temporary_parent: Path) -> dict[str, Any]:
    target = next(relative for relative in sorted(mapped) if relative != SOURCE_REL.as_posix())
    with tempfile.TemporaryDirectory(prefix=".s9lnw_", dir=temporary_parent) as temporary:
        root = Path(temporary)
        source = root / SOURCE_REL
        source.parent.mkdir(parents=True)
        source.write_bytes(before_source)
        collision = root / target
        collision.parent.mkdir(parents=True, exist_ok=True)
        sentinel = b"STEP9L_INSTALL_NEVER_OVERWRITE\n"
        collision.write_bytes(sentinel)
        rejected = False
        try:
            install_transaction(root, mapped, before_source)
        except InstallError as error:
            rejected = "never-overwrite" in str(error)
        unchanged = collision.read_bytes() == sentinel and source.read_bytes() == before_source
        require(rejected and unchanged, "install never-overwrite regression failed")
        return {"collision_target": target, "expected_rejection_observed": rejected, "pre_existing_bytes_unchanged": unchanged, "status": "PASS"}


def git_staged(root: Path) -> dict[str, Any]:
    result = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=root, capture_output=True, text=True, check=True)
    paths = [line for line in result.stdout.splitlines() if line]
    require(not paths, "Git staged files present: " + repr(paths))
    return {"staged_count": 0, "staged_paths": [], "status": "PASS"}


def write_evidence_transaction(output: Path, files: dict[str, bytes]) -> None:
    require(not output.exists() and not output.is_symlink(), "install evidence output already exists")
    token = uuid.uuid4().hex
    temporary = output.parent / ("." + output.name + ".tmp-" + token)
    require(not temporary.exists(), "install evidence temporary path exists")
    committed = False
    try:
        temporary.mkdir()
        for relative, payload in sorted(files.items()):
            require(safe_zip_name(relative), "unsafe evidence path")
            path = temporary / PurePosixPath(relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        temporary.replace(output)
        committed = True
    finally:
        if not committed and temporary.exists():
            shutil.rmtree(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = root / "Validation/WordsInSilence_step9l_personal_multi_prototype_runtime_candidate_v1_1_unity_install"
    require(not output.exists(), "install evidence output already exists")
    git_before = git_staged(root)
    file_map, fixed_input = verify_fixed_delivery(root)
    mapped, mapping_records = build_mappings(file_map)
    profile_contract = validate_profile_contract(mapped)
    baselines = load_protection_baselines(file_map)
    protected_before = snapshot_protected(root, baselines)
    before_source = ensure_preconditions(root, mapped)
    regression = transaction_regression(mapped, before_source, root)
    never_overwrite = never_overwrite_regression(mapped, before_source, root)

    protected_after_holder: dict[str, Any] = {}
    def validate_after() -> None:
        protected_after_holder.update(snapshot_protected(root, baselines))

    install_result = install_transaction(root, mapped, before_source, after_validator=validate_after)
    protected_after = protected_after_holder
    require(protected_before["groups"] == protected_after["groups"], "protected before/after mismatch")
    require((root / SOURCE_REL).read_bytes() == mapped[SOURCE_REL.as_posix()], "replacement source post-install mismatch")
    git_after = git_staged(root)
    assertions = [
        {"assertion_key": "fixed_step9l_v1_1_delivery", "status": "PASS", "details": fixed_input},
        {"assertion_key": "approved_project_mapping_54", "status": "PASS", "details": {"replacement": 1, "new": 53}},
        {"assertion_key": "approved_source_before_to_after", "status": "PASS", "details": {"before_sha256": SOURCE_BEFORE_SHA256, "after_sha256": SOURCE_AFTER_SHA256}},
        {"assertion_key": "transaction_failure_rollback_four_of_four", "status": "PASS", "details": regression},
        {"assertion_key": "new_target_never_overwrite", "status": "PASS", "details": never_overwrite},
        {"assertion_key": "protected_global_11_personal_10_csharp_4_archives_6", "status": "PASS", "details": {"global": 11, "personal": 10, "csharp": 4, "archives": 6}},
        {"assertion_key": "candidate_profile_default_inactive", "status": "PASS", "details": profile_contract},
        {"assertion_key": "git_staged_zero", "status": "PASS", "details": {"before": git_before, "after": git_after}},
        {"assertion_key": "unity_compile_editmode_eight", "status": "UNVERIFIED", "details": {"fixture": "PersonalMultiPrototypeRuntimeIntegrationV1Tests", "executed": False}},
    ]
    require(len({row["assertion_key"] for row in assertions}) == len(assertions), "assertion key duplicate")
    installed_records = []
    for record in mapping_records:
        actual = file_record(root, record["project_relative_path"])
        require(actual["size_bytes"] == record["size_bytes"] and actual["sha256"] == record["sha256"], "installed manifest mismatch")
        installed_records.append({**record, "installed_size_bytes": actual["size_bytes"], "installed_sha256": actual["sha256"], "byte_identical": True})
    evidence = {
        "fixed_input_verification.json": json_bytes({"schema_version": SCHEMA + ".fixed_input", **fixed_input}),
        "project_mapping_manifest.json": json_bytes({"schema_version": SCHEMA + ".project_mapping_manifest", "mapping_count": 54, "replacement_count": 1, "new_file_count": 53, "mappings": installed_records}),
        "protected_before.json": json_bytes(protected_before),
        "protected_after_install.json": json_bytes(protected_after),
        "install_transaction_failure_injection_regression.json": json_bytes(regression),
        "install_never_overwrite_regression.json": json_bytes(never_overwrite),
        "install_result.json": json_bytes({"schema_version": SCHEMA + ".install_result", **install_result, "source_before_sha256": SOURCE_BEFORE_SHA256, "source_after_sha256": SOURCE_AFTER_SHA256, "profile": profile_contract, "protected_mismatch_count": 0, "git_staged_count": 0, "status": "INSTALL_PASS_PENDING_UNITY_EDITOR_GATE"}),
        "unity_gate_status.json": json_bytes({"schema_version": SCHEMA + ".unity_gate_status", "fixture": "PersonalMultiPrototypeRuntimeIntegrationV1Tests", "expected_test_count": 8, "compiler_errors": None, "executed": False, "status": "UNVERIFIED_PENDING_NORMAL_GUI_EDITOR"}),
        "ASSERTIONS.jsonl": b"".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n" for row in assertions),
    }
    manifest_records = [{"relative_path": name, "size_bytes": len(payload), "sha256": sha256_bytes(payload)} for name, payload in sorted(evidence.items())]
    evidence["output_file_manifest.json"] = json_bytes({"schema_version": SCHEMA + ".output_file_manifest", "self_excluded": "output_file_manifest.json", "output_count_excluding_manifest": len(manifest_records), "outputs": manifest_records})
    write_evidence_transaction(output, evidence)
    print(json.dumps({
        "status": "INSTALL_PASS_PENDING_UNITY_EDITOR_GATE",
        "installed_file_count": 54, "replacement_count": 1, "new_file_count": 53,
        "source_before_sha256": SOURCE_BEFORE_SHA256, "source_after_sha256": SOURCE_AFTER_SHA256,
        "protected_mismatch_count": 0, "transaction_regression": "4/4 PASS",
        "evidence_directory": output.as_posix(), "evidence_file_count": len(evidence),
        "unity_fixture": "PersonalMultiPrototypeRuntimeIntegrationV1Tests", "unity_executed": False,
        "git_staged_count": 0,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InstallError as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, ensure_ascii=False, indent=2))
        raise SystemExit(1)
