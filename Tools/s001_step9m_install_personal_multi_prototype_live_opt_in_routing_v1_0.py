#!/usr/bin/env python3
"""Transactional installer contract for the Step 9M live opt-in routing payload."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any


SCHEMA = "words_in_silence.step9m.personal_multi_prototype_live_opt_in_routing.install.v1_0"
INJECTION_POINTS = (
    "BEFORE_FIRST_RENAME",
    "AFTER_REPLACEMENT_RENAME",
    "AFTER_FIRST_NEW_RENAME",
    "AFTER_ALL_RENAMES_BEFORE_COMMIT",
)


class InstallError(RuntimeError):
    pass


class InjectedInstallFailure(InstallError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise InstallError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _remove_owned_file(path: Path, root: Path) -> None:
    resolved = path.resolve()
    allowed = root.resolve()
    require(resolved == allowed or allowed in resolved.parents, f"cleanup escaped target root: {path}")
    if path.exists():
        require(path.is_file(), f"cleanup target is not a file: {path}")
        path.unlink()


def install_transaction(
    target_root: Path,
    payload: dict[str, bytes],
    mappings: list[dict[str, Any]],
    *,
    inject_failure_at: str | None = None,
    transaction_token: str | None = None,
) -> dict[str, Any]:
    """Install one approved replacement and never-overwrite new files atomically."""
    target_root = target_root.resolve()
    require(target_root.is_dir(), f"target root missing: {target_root}")
    require(inject_failure_at is None or inject_failure_at in INJECTION_POINTS,
            f"unknown injection point: {inject_failure_at}")
    token = transaction_token or uuid.uuid4().hex
    require(len(mappings) == len(payload), "mapping/payload count mismatch")

    finals: dict[str, Path] = {}
    temps: dict[str, Path] = {}
    backups: dict[str, Path] = {}
    replacement_keys: list[str] = []
    new_keys: list[str] = []

    for row in mappings:
        key = row["project_relative_path"]
        require(key in payload, f"payload missing mapping: {key}")
        require(key not in finals, f"duplicate mapping: {key}")
        final = target_root / Path(key)
        require(target_root in final.resolve().parents, f"unsafe project path: {key}")
        operation = row["operation"]
        if operation == "replace_approved_source":
            require(final.is_file(), f"approved replacement target missing: {key}")
            before = final.read_bytes()
            require(len(before) == row["before_size_bytes"], f"replacement before size mismatch: {key}")
            require(sha256_bytes(before) == row["before_sha256"], f"replacement before hash mismatch: {key}")
            replacement_keys.append(key)
        elif operation == "create_never_overwrite":
            require(not final.exists(), f"never-overwrite target exists: {key}")
            new_keys.append(key)
        else:
            raise InstallError(f"unsupported install operation: {operation}")
        body = payload[key]
        require(len(body) == row["size_bytes"] and sha256_bytes(body) == row["sha256"],
                f"payload identity mismatch: {key}")
        final.parent.mkdir(parents=True, exist_ok=True)
        temp = final.with_name(f".{final.name}.step9m-{token}.tmp")
        backup = final.with_name(f".{final.name}.step9m-{token}.bak")
        require(not temp.exists() and not backup.exists(), f"transaction temp collision: {key}")
        finals[key] = final
        temps[key] = temp
        backups[key] = backup

    require(bool(replacement_keys), "approved replacement mapping missing")
    created_finals: list[str] = []
    moved_replacements: list[str] = []
    committed = False
    try:
        for key, temp in temps.items():
            with temp.open("xb") as handle:
                handle.write(payload[key])
        if inject_failure_at == "BEFORE_FIRST_RENAME":
            raise InjectedInstallFailure(inject_failure_at)

        for replacement_key in replacement_keys:
            os.replace(finals[replacement_key], backups[replacement_key])
            moved_replacements.append(replacement_key)
            os.replace(temps[replacement_key], finals[replacement_key])
        if inject_failure_at == "AFTER_REPLACEMENT_RENAME":
            raise InjectedInstallFailure(inject_failure_at)

        for index, key in enumerate(new_keys):
            os.replace(temps[key], finals[key])
            created_finals.append(key)
            if index == 0 and inject_failure_at == "AFTER_FIRST_NEW_RENAME":
                raise InjectedInstallFailure(inject_failure_at)
        if inject_failure_at == "AFTER_ALL_RENAMES_BEFORE_COMMIT":
            raise InjectedInstallFailure(inject_failure_at)

        for row in mappings:
            key = row["project_relative_path"]
            body = finals[key].read_bytes()
            require(len(body) == row["size_bytes"] and sha256_bytes(body) == row["sha256"],
                    f"installed identity mismatch: {key}")
        for replacement_key in replacement_keys:
            _remove_owned_file(backups[replacement_key], target_root)
        moved_replacements.clear()
        committed = True
        return {
            "schema_version": SCHEMA + ".commit",
            "mapping_count": len(mappings),
            "replacement_count": len(replacement_keys),
            "new_file_count": len(new_keys),
            "committed": True,
        }
    finally:
        if not committed:
            for key in reversed(created_finals):
                _remove_owned_file(finals[key], target_root)
            for replacement_key in reversed(moved_replacements):
                _remove_owned_file(finals[replacement_key], target_root)
                if backups[replacement_key].exists():
                    os.replace(backups[replacement_key], finals[replacement_key])
            moved_replacements.clear()
        for path in list(temps.values()) + list(backups.values()):
            _remove_owned_file(path, target_root)


def run_failure_injection_regression(
    payload: dict[str, bytes],
    mappings: list[dict[str, Any]],
    temporary_parent: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    replacements = [row for row in mappings if row["operation"] == "replace_approved_source"]
    for point in INJECTION_POINTS:
        with tempfile.TemporaryDirectory(prefix="s9m_tx_", dir=temporary_parent) as temporary:
            root = Path(temporary)
            replacement_before: dict[str, bytes] = {}
            for replacement in replacements:
                replacement_path = root / replacement["project_relative_path"]
                replacement_path.parent.mkdir(parents=True, exist_ok=True)
                before = bytes.fromhex(replacement["before_hex"])
                replacement_path.write_bytes(before)
                replacement_before[replacement["project_relative_path"]] = before
            sentinel = root / "pre_existing_sentinel.bin"
            sentinel_body = b"STEP9M_PRE_EXISTING_SENTINEL"
            sentinel.write_bytes(sentinel_body)
            observed = False
            try:
                install_transaction(
                    root,
                    payload,
                    mappings,
                    inject_failure_at=point,
                    transaction_token="injection",
                )
            except InjectedInstallFailure:
                observed = True
            residual_temps = list(root.rglob("*.tmp")) + list(root.rglob("*.bak"))
            new_remaining = sum(
                (root / row["project_relative_path"]).exists()
                for row in mappings if row["operation"] == "create_never_overwrite"
            )
            restored = all(
                (root / relative).read_bytes() == before
                for relative, before in replacement_before.items()
            )
            sentinel_same = sentinel.read_bytes() == sentinel_body
            passed = observed and not residual_temps and new_remaining == 0 and restored and sentinel_same
            rows.append({
                "injection_point": point,
                "expected_exception_observed": observed,
                "remaining_new_final_outputs": new_remaining,
                "remaining_temporary_outputs": len(residual_temps),
                "replacement_restored": restored,
                "pre_existing_sentinel_changed": not sentinel_same,
                "verdict": "PASS" if passed else "FAIL",
            })
    return {
        "schema_version": SCHEMA + ".failure_injection_regression",
        "production_transaction_function": install_transaction.__name__,
        "same_function_used": True,
        "injection_count": len(rows),
        "pass_count": sum(row["verdict"] == "PASS" for row in rows),
        "fail_count": sum(row["verdict"] == "FAIL" for row in rows),
        "rows": rows,
        "status": "PASS" if all(row["verdict"] == "PASS" for row in rows) else "FAIL",
    }


def run_never_overwrite_regression(
    payload: dict[str, bytes],
    mappings: list[dict[str, Any]],
    temporary_parent: Path,
) -> dict[str, Any]:
    replacements = [row for row in mappings if row["operation"] == "replace_approved_source"]
    first_new = next(row for row in mappings if row["operation"] == "create_never_overwrite")
    with tempfile.TemporaryDirectory(prefix="s9m_no_", dir=temporary_parent) as temporary:
        root = Path(temporary)
        replacement_before: dict[str, bytes] = {}
        for replacement in replacements:
            replacement_path = root / replacement["project_relative_path"]
            replacement_path.parent.mkdir(parents=True, exist_ok=True)
            before = bytes.fromhex(replacement["before_hex"])
            replacement_path.write_bytes(before)
            replacement_before[replacement["project_relative_path"]] = before
        occupied = root / first_new["project_relative_path"]
        occupied.parent.mkdir(parents=True, exist_ok=True)
        sentinel = b"DO_NOT_OVERWRITE"
        occupied.write_bytes(sentinel)
        rejected = False
        try:
            install_transaction(root, payload, mappings, transaction_token="never-overwrite")
        except InstallError:
            rejected = True
        replacements_same = all(
            (root / relative).read_bytes() == before
            for relative, before in replacement_before.items()
        )
        passed = rejected and occupied.read_bytes() == sentinel and replacements_same
        return {
            "schema_version": SCHEMA + ".never_overwrite_regression",
            "existing_target_rejected": rejected,
            "existing_target_changed": occupied.read_bytes() != sentinel,
            "replacement_changed": not replacements_same,
            "status": "PASS" if passed else "FAIL",
        }


def _load_payload(payload_root: Path, manifest_path: Path) -> tuple[dict[str, bytes], list[dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mappings = manifest["mappings"]
    payload: dict[str, bytes] = {}
    for row in mappings:
        relative = row["project_relative_path"]
        payload[relative] = (payload_root / relative).read_bytes()
    return payload, mappings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload-root", type=Path, required=True)
    parser.add_argument("--mapping-manifest", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    payload, mappings = _load_payload(args.payload_root, args.mapping_manifest)
    if args.apply:
        result = install_transaction(args.target_root, payload, mappings)
    else:
        result = run_failure_injection_regression(payload, mappings, args.target_root)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
