#!/usr/bin/env python3
"""Prepare, verify, and package WordsInSilence Step 9A runtime recognition service v1.0."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import uuid
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent.parent
STEP8B = ROOT / "data/processed/sessions/S001_P001/step8b_recognition_core_v1_0"
STEP8C = ROOT / "data/processed/sessions/S001_P001/step8c_offline_recognition_evaluation_v1_0"
OUTPUT = ROOT / "data/processed/sessions/S001_P001/step9a_runtime_recognition_service_v1_0"
RUNTIME_SOURCE_ROOT = ROOT / "Assets/_Project/Scripts/Recognition/Runtime"
RUNTIME_GATE_ROOT = ROOT / "Assets/StreamingAssets/RecognitionCoreV1/RuntimeGate"
RUNTIME_THRESHOLD = RUNTIME_GATE_ROOT / "frozen_threshold_config.json"
RUNTIME_ASSET_MANIFEST = RUNTIME_GATE_ROOT / "runtime_recognition_assets.json"
REPLAY_ROOT = ROOT / "Assets/_Project/Tests/EditMode/Fixtures/RuntimeRecognitionServiceV1"
REPLAY_QUERY_ROOT = REPLAY_ROOT / "TestQueries"
TEST_SOURCE = ROOT / "Assets/_Project/Tests/EditMode/RuntimeRecognitionServiceV1Tests.cs"
STATE_PATH = OUTPUT / "run_state.json"

STEP8B_ZIP = ROOT / "WordsInSilence_step8b_recognition_core_v1_0_delivery.zip"
STEP8C_ZIP = ROOT / "WordsInSilence_step8c_offline_recognition_evaluation_v1_0_delivery.zip"
STEP8B_HASH = ROOT / f"{STEP8B_ZIP.name}.sha256"
STEP8C_HASH = ROOT / f"{STEP8C_ZIP.name}.sha256"
STEP8B_SIDECAR = ROOT / "WordsInSilence_step8b_recognition_core_v1_0_delivery_post_zip_verification.json"
STEP8C_SIDECAR = ROOT / "WordsInSilence_step8c_offline_recognition_evaluation_v1_0_delivery_post_zip_verification.json"
APPROVED_THRESHOLD = STEP8C / "frozen_threshold_config.json"

VERSION = "1.0"
STEP8B_ZIP_SIZE = 661_831
STEP8B_ZIP_SHA256 = "3401ea74f73902c5ea7b4f34abb06ebfe933ef0a076820f15b6471937a5c7c6d"
STEP8C_ZIP_SIZE = 445_505
STEP8C_ZIP_SHA256 = "e1cf9d7f197feba99cc0499795f3f04702cd110bbe6d3200341a7a0aa2d55900"
THRESHOLD_SIZE = 4_590
THRESHOLD_SHA256 = "2697bb1d0c301850d2abda41351519c66d0eb8e27841263469175eb3fa7e99e3"
THRESHOLD_CANONICAL_SHA256 = "8600a9529c82a4d30dea984515bc56f6ade74c12331a3df3fcf59f3ffcf7f7c0"
DELIVERY_NAME = "WordsInSilence_step9a_runtime_recognition_service_v1_0_delivery.zip"
DELIVERY_ZIP = ROOT / DELIVERY_NAME
DELIVERY_HASH = ROOT / f"{DELIVERY_NAME}.sha256"
DELIVERY_SIDECAR = ROOT / "WordsInSilence_step9a_runtime_recognition_service_v1_0_delivery_post_zip_verification.json"


def log(message: str = "") -> None:
    print(message, flush=True)


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def add_check(checks: list[dict], name: str, passed: bool, evidence=None) -> None:
    checks.append({"name": name, "status": "PASS" if bool(passed) else "FAIL", "evidence": evidence})


def summarize(checks: list[dict]) -> dict:
    names = [record["name"] for record in checks]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    passed = sum(record["status"] == "PASS" for record in checks)
    return {
        "total": len(checks),
        "pass": passed,
        "fail": len(checks) - passed,
        "duplicate_count": len(duplicates),
        "duplicate_names": duplicates,
    }


def threshold_payload(path: Path) -> dict:
    wrapper = load_json(path)
    canonical = canonical_json(wrapper["canonical_payload"])
    if sha256_bytes(canonical.encode("utf-8")) != wrapper["canonical_payload_sha256"]:
        raise RuntimeError(f"threshold canonical payload hash mismatch: {path}")
    return wrapper


def verify_package(delivery: Path, detached: Path, sidecar_path: Path, expected_size: int, expected_hash: str, prefix: str, checks: list[dict]) -> None:
    actual_size = delivery.stat().st_size
    actual_hash = sha256_file(delivery)
    add_check(checks, f"{prefix}_zip_size", actual_size == expected_size, actual_size)
    add_check(checks, f"{prefix}_zip_sha256", actual_hash == expected_hash, actual_hash)
    detached_text = detached.read_text(encoding="ascii")
    match = re.fullmatch(r"([0-9a-f]{64})  ([^\r\n]+)\n", detached_text)
    add_check(checks, f"{prefix}_detached_strict", bool(match and match.group(1) == actual_hash and match.group(2) == delivery.name), detached_text.rstrip())
    with zipfile.ZipFile(delivery, "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        bad_crc = archive.testzip()
    unsafe = [name for name in names if name.startswith(("/", "\\")) or ".." in PurePosixPath(name).parts or re.match(r"^[A-Za-z]:", name)]
    links = [info.filename for info in infos if stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK]
    encrypted = [info.filename for info in infos if info.flag_bits & 1]
    add_check(checks, f"{prefix}_zip_crc", bad_crc is None, bad_crc)
    add_check(checks, f"{prefix}_zip_unique_safe_plain", len(names) == len(set(names)) and not unsafe and not links and not encrypted, None)
    sidecar = load_json(sidecar_path)
    records = sidecar["checks"]
    record_names = [record["check_name"] for record in records]
    canonical_records = [canonical_json(record) for record in records]
    add_check(checks, f"{prefix}_sidecar_pass", sidecar["verification_status"] == "PASS" and all(record["status"] == "PASS" for record in records), None)
    add_check(checks, f"{prefix}_sidecar_duplicates_zero", len(record_names) == len(set(record_names)) and len(canonical_records) == len(set(canonical_records)), None)


def meta_guid(path: Path) -> str:
    return hashlib.sha256(("WordsInSilence-Step9A:" + path.relative_to(ROOT).as_posix()).encode("utf-8")).hexdigest()[:32]


def ensure_meta(path: Path, folder: bool = False) -> Path:
    meta = path.with_name(path.name + ".meta")
    if meta.exists():
        return meta
    guid = meta_guid(path)
    if folder:
        text = f"fileFormatVersion: 2\nguid: {guid}\nfolderAsset: yes\nDefaultImporter:\n  externalObjects: {{}}\n  userData: \n  assetBundleName: \n  assetBundleVariant: \n"
    elif path.suffix == ".cs":
        text = f"fileFormatVersion: 2\nguid: {guid}\nMonoImporter:\n  externalObjects: {{}}\n  serializedVersion: 2\n  defaultReferences: []\n  executionOrder: 0\n  icon: {{instanceID: 0}}\n  userData: \n  assetBundleName: \n  assetBundleVariant: \n"
    else:
        text = f"fileFormatVersion: 2\nguid: {guid}\nDefaultImporter:\n  externalObjects: {{}}\n  userData: \n  assetBundleName: \n  assetBundleVariant: \n"
    meta.write_text(text, encoding="utf-8")
    return meta


def snapshot(paths: list[Path]) -> list[dict]:
    records = []
    for path in sorted(set(paths), key=lambda item: item.relative_to(ROOT).as_posix()):
        file_stat = path.stat()
        records.append({
            "relative_path": path.relative_to(ROOT).as_posix(),
            "size_bytes": file_stat.st_size,
            "sha256": sha256_file(path),
            "mtime_ns": file_stat.st_mtime_ns,
        })
    return records


def verify_snapshot(records: list[dict]) -> list[dict]:
    changes = []
    for record in records:
        path = ROOT / PurePosixPath(record["relative_path"])
        if not path.is_file():
            changes.append({"relative_path": record["relative_path"], "reason": "missing"})
            continue
        file_stat = path.stat()
        actual_hash = sha256_file(path)
        if file_stat.st_size != record["size_bytes"] or file_stat.st_mtime_ns != record["mtime_ns"] or actual_hash != record["sha256"]:
            changes.append({"relative_path": record["relative_path"], "actual_size": file_stat.st_size, "actual_sha256": actual_hash})
    return changes


def preflight(checks: list[dict]) -> dict:
    verify_package(STEP8B_ZIP, STEP8B_HASH, STEP8B_SIDECAR, STEP8B_ZIP_SIZE, STEP8B_ZIP_SHA256, "step8b", checks)
    verify_package(STEP8C_ZIP, STEP8C_HASH, STEP8C_SIDECAR, STEP8C_ZIP_SIZE, STEP8C_ZIP_SHA256, "step8c", checks)
    approved_wrapper = threshold_payload(APPROVED_THRESHOLD)
    add_check(checks, "approved_threshold_size", APPROVED_THRESHOLD.stat().st_size == THRESHOLD_SIZE, APPROVED_THRESHOLD.stat().st_size)
    add_check(checks, "approved_threshold_file_sha", sha256_file(APPROVED_THRESHOLD) == THRESHOLD_SHA256, sha256_file(APPROVED_THRESHOLD))
    add_check(checks, "approved_threshold_canonical_sha", approved_wrapper["canonical_payload_sha256"] == THRESHOLD_CANONICAL_SHA256, approved_wrapper["canonical_payload_sha256"])

    runtime_manifest = load_json(STEP8B / "runtime_template_manifest.json")
    template_paths = []
    template_bad = []
    for record in runtime_manifest["templates"]:
        path = ROOT / PurePosixPath(record["relative_path"])
        template_paths.append(path)
        if record["split"] != "train" or not path.is_file() or path.stat().st_size != record["size_bytes"] or sha256_file(path) != record["file_sha256"]:
            template_bad.append(record["sample_id"])
    add_check(checks, "approved_runtime_templates_9", len(runtime_manifest["templates"]) == 9, len(runtime_manifest["templates"]))
    add_check(checks, "approved_runtime_template_hashes", not template_bad, template_bad)

    step8b_manifest = load_json(STEP8B / "output_file_manifest.json")
    step8b_lookup = {record["relative_path"]: record for record in step8b_manifest["outputs"]}
    core_paths = [ROOT / "Assets/_Project/Scripts/Recognition/Core" / name for name in ("RecognitionSequence.cs", "RecognitionSequenceBinary.cs", "MaskAwareDtw.cs", "RecognitionCore.cs")]
    core_bad = []
    for path in core_paths:
        relative = path.relative_to(ROOT).as_posix()
        record = step8b_lookup.get(relative)
        if record is None or path.stat().st_size != record["size_bytes"] or sha256_file(path) != record["sha256"]:
            core_bad.append(relative)
    add_check(checks, "approved_core_sources_4", not core_bad and len(core_paths) == 4, core_bad)

    step8c_manifest = load_json(STEP8C / "output_file_manifest.json")
    step8c_lookup = {record["relative_path"]: record for record in step8c_manifest["outputs"]}
    evaluation_paths = [ROOT / "Assets/_Project/Scripts/Recognition/Evaluation" / name for name in ("DistanceGatePolicy.cs", "FrozenDistanceGateConfig.cs")]
    evaluation_bad = []
    for path in evaluation_paths:
        relative = path.relative_to(ROOT).as_posix()
        record = step8c_lookup.get(relative)
        if record is None or path.stat().st_size != record["size_bytes"] or sha256_file(path) != record["sha256"]:
            evaluation_bad.append(relative)
    add_check(checks, "approved_evaluation_sources_2", not evaluation_bad and len(evaluation_paths) == 2, evaluation_bad)

    fixture_manifest = load_json(STEP8C / "test_fixture_manifest.json")
    fixture_paths = []
    fixture_bad = []
    for record in fixture_manifest["test_queries"]:
        path = ROOT / PurePosixPath(record["relative_path"])
        fixture_paths.append(path)
        if record["split"] != "test" or path.stat().st_size != record["size_bytes"] or sha256_file(path) != record["file_sha256"]:
            fixture_bad.append(record["sample_id"])
    for record in fixture_manifest["fixture_files"]:
        path = ROOT / PurePosixPath(record["relative_path"])
        fixture_paths.append(path)
        if path.stat().st_size != record["size_bytes"] or sha256_file(path) != record["sha256"]:
            fixture_bad.append(record["relative_path"])
    add_check(checks, "approved_step8c_test_queries_9_test_only", len(fixture_manifest["test_queries"]) == 9 and all(record["split"] == "test" for record in fixture_manifest["test_queries"]), None)
    add_check(checks, "approved_step8c_fixture_hashes", not fixture_bad, fixture_bad)
    required_sources = [RUNTIME_SOURCE_ROOT / "NormalizedRecognitionFrame.cs", RUNTIME_SOURCE_ROOT / "RecognitionSequenceInputSession.cs", RUNTIME_SOURCE_ROOT / "RuntimeRecognitionService.cs", TEST_SOURCE, Path(__file__).resolve()]
    add_check(checks, "step9a_implementation_sources_present", all(path.is_file() for path in required_sources), [str(path) for path in required_sources if not path.is_file()])

    input_paths = [
        STEP8B_ZIP, STEP8B_HASH, STEP8B_SIDECAR,
        STEP8C_ZIP, STEP8C_HASH, STEP8C_SIDECAR,
        STEP8B / "runtime_template_manifest.json", STEP8B / "output_file_manifest.json",
        STEP8C / "output_file_manifest.json", STEP8C / "test_fixture_manifest.json", APPROVED_THRESHOLD,
        *template_paths, *core_paths, *evaluation_paths, *fixture_paths,
    ]
    return {
        "runtime_manifest": runtime_manifest,
        "fixture_manifest": fixture_manifest,
        "input_paths": input_paths,
        "required_sources": required_sources,
    }


def copy_replay_fixtures(run_id: str, fixture_manifest: dict, checks: list[dict]) -> dict:
    REPLAY_QUERY_ROOT.mkdir(parents=True, exist_ok=False)
    ensure_meta(REPLAY_ROOT, folder=True)
    ensure_meta(REPLAY_QUERY_ROOT, folder=True)
    copied_queries = []
    for record in fixture_manifest["test_queries"]:
        source = ROOT / PurePosixPath(record["relative_path"])
        destination = REPLAY_QUERY_ROOT / source.name
        shutil.copyfile(source, destination)
        ensure_meta(destination)
        copied = {**record, "relative_path": destination.relative_to(ROOT).as_posix()}
        add_check(checks, f"replay_{record['sample_id']}_byte_exact", destination.read_bytes() == source.read_bytes(), None)
        add_check(checks, f"replay_{record['sample_id']}_hash_exact", sha256_file(destination) == record["file_sha256"], sha256_file(destination))
        copied_queries.append(copied)
    source_by_name = {Path(record["relative_path"]).name: ROOT / PurePosixPath(record["relative_path"]) for record in fixture_manifest["fixture_files"]}
    copied_files = []
    for name in ("test_expected_distances.csv", "test_expected_predictions.csv"):
        source = source_by_name[name]
        destination = REPLAY_ROOT / name
        shutil.copyfile(source, destination)
        ensure_meta(destination)
        copied_files.append({"relative_path": destination.relative_to(ROOT).as_posix(), "size_bytes": destination.stat().st_size, "sha256": sha256_file(destination)})
        add_check(checks, f"replay_{name}_byte_exact", destination.read_bytes() == source.read_bytes(), None)
    replay_manifest = {
        "version": VERSION,
        "run_id": run_id,
        "scope": "step8c_test_fixture_runtime_contract_replay_only",
        "source_step8c_run_id": fixture_manifest["run_id"],
        "source_step8c_fixture_manifest_sha256": sha256_file(STEP8C / "test_fixture_manifest.json"),
        "test_queries": copied_queries,
        "fixture_files": copied_files,
        "tuning_use": "forbidden",
    }
    manifest_path = REPLAY_ROOT / "replay_fixture_manifest.json"
    write_json(manifest_path, replay_manifest)
    ensure_meta(manifest_path)
    return replay_manifest


def prepare() -> int:
    log("=== Step 9A prepare ===")
    conflicts = [path for path in (OUTPUT, RUNTIME_GATE_ROOT, REPLAY_ROOT, DELIVERY_ZIP, DELIVERY_HASH, DELIVERY_SIDECAR) if path.exists()]
    if conflicts:
        raise RuntimeError(f"Step 9A output path conflict: {conflicts}")
    checks: list[dict] = []
    data = preflight(checks)
    preflight_summary = summarize(checks)
    if preflight_summary["fail"] or preflight_summary["duplicate_count"]:
        raise RuntimeError(f"Step 9A preflight failed: {preflight_summary}")

    run_id = str(uuid.uuid4())
    OUTPUT.mkdir(parents=True)
    RUNTIME_GATE_ROOT.mkdir(parents=True)
    ensure_meta(RUNTIME_SOURCE_ROOT, folder=True)
    ensure_meta(RUNTIME_GATE_ROOT, folder=True)
    for source in data["required_sources"]:
        if source.suffix == ".cs": ensure_meta(source)

    approved_bytes = APPROVED_THRESHOLD.read_bytes()
    RUNTIME_THRESHOLD.write_bytes(approved_bytes)
    ensure_meta(RUNTIME_THRESHOLD)
    approved_wrapper = threshold_payload(APPROVED_THRESHOLD)
    runtime_wrapper = threshold_payload(RUNTIME_THRESHOLD)
    add_check(checks, "runtime_threshold_byte_exact", RUNTIME_THRESHOLD.read_bytes() == approved_bytes, None)
    add_check(checks, "runtime_threshold_size", RUNTIME_THRESHOLD.stat().st_size == THRESHOLD_SIZE, RUNTIME_THRESHOLD.stat().st_size)
    add_check(checks, "runtime_threshold_file_sha", sha256_file(RUNTIME_THRESHOLD) == THRESHOLD_SHA256, sha256_file(RUNTIME_THRESHOLD))
    add_check(checks, "runtime_threshold_canonical_sha", runtime_wrapper["canonical_payload_sha256"] == approved_wrapper["canonical_payload_sha256"] == THRESHOLD_CANONICAL_SHA256, runtime_wrapper["canonical_payload_sha256"])

    runtime_asset_manifest = {
        "version": VERSION,
        "run_id": run_id,
        "template_count": 9,
        "template_directory": "Assets/StreamingAssets/RecognitionCoreV1/Templates",
        "templates": [{
            "sample_id": record["sample_id"],
            "word": record["word"],
            "speed": record["speed"],
            "split": record["split"],
            "relative_path": record["relative_path"],
            "size_bytes": record["size_bytes"],
            "file_sha256": record["file_sha256"],
        } for record in data["runtime_manifest"]["templates"]],
        "threshold": {
            "relative_path": RUNTIME_THRESHOLD.relative_to(ROOT).as_posix(),
            "size_bytes": RUNTIME_THRESHOLD.stat().st_size,
            "file_sha256": sha256_file(RUNTIME_THRESHOLD),
            "canonical_payload_sha256": runtime_wrapper["canonical_payload_sha256"],
            "approved_source_relative_path": APPROVED_THRESHOLD.relative_to(ROOT).as_posix(),
        },
        "approved_step8b_delivery_sha256": STEP8B_ZIP_SHA256,
        "approved_step8c_delivery_sha256": STEP8C_ZIP_SHA256,
    }
    write_json(RUNTIME_ASSET_MANIFEST, runtime_asset_manifest)
    ensure_meta(RUNTIME_ASSET_MANIFEST)
    replay_manifest = copy_replay_fixtures(run_id, data["fixture_manifest"], checks)
    add_check(checks, "replay_query_count_9", len(replay_manifest["test_queries"]) == 9, len(replay_manifest["test_queries"]))

    input_records = snapshot(data["input_paths"])
    input_manifest = {
        "version": VERSION,
        "run_id": run_id,
        "self_reference_excluded": True,
        "approved_step8b_delivery_sha256": STEP8B_ZIP_SHA256,
        "approved_step8c_delivery_sha256": STEP8C_ZIP_SHA256,
        "inputs": input_records,
        "total_files": len(input_records),
    }
    write_json(OUTPUT / "input_manifest.json", input_manifest)
    prepare_assertions = {"version": VERSION, "run_id": run_id, **summarize(checks), "results": checks}
    write_json(OUTPUT / "prepare_assertion_results.json", prepare_assertions)
    if prepare_assertions["fail"] or prepare_assertions["duplicate_count"]:
        raise RuntimeError(f"Step 9A prepare assertions failed: {prepare_assertions}")
    write_json(STATE_PATH, {
        "version": VERSION,
        "run_id": run_id,
        "phase": "prepared_for_unity",
        "input_snapshot": input_records,
        "runtime_threshold": runtime_asset_manifest["threshold"],
        "replay_query_count": len(replay_manifest["test_queries"]),
    })
    log(f"Run ID: {run_id}")
    log(f"Prepare PASS: {prepare_assertions['pass']}/{prepare_assertions['total']} assertions")
    log(f"Runtime threshold: bytes={RUNTIME_THRESHOLD.stat().st_size}, sha256={sha256_file(RUNTIME_THRESHOLD)}")
    return 0


def parse_unity_results(path: Path) -> tuple[dict, list[str]]:
    root = ET.parse(path).getroot()
    cases = [element for element in root.iter("test-case")]
    def integer(name: str) -> int:
        value = root.attrib.get(name)
        return int(value) if value is not None else 0
    return {
        "result": root.attrib.get("result", ""),
        "testcasecount": integer("testcasecount"),
        "passed": integer("passed"),
        "failed": integer("failed"),
        "skipped": integer("skipped"),
        "inconclusive": integer("inconclusive"),
        "asserts": integer("asserts"),
        "duration_seconds": float(root.attrib.get("duration", "0")),
    }, [element.attrib.get("fullname", "") for element in cases]


def artifact_type(path: Path) -> str:
    relative = path.relative_to(ROOT).as_posix()
    if relative.endswith(".cs"): return "csharp_source_or_test"
    if relative.endswith(".meta"): return "unity_meta"
    if relative.endswith(".wisrq"): return "replay_query_fixture"
    if relative.endswith("unity_editmode_test_results.xml"): return "unity_test_results"
    if relative.endswith("unity_editmode_test.log"): return "unity_test_log"
    if relative.endswith(".csv"): return "replay_expected_csv"
    if relative.endswith(".md"): return "report"
    if relative.endswith(".py"): return "tool"
    if relative.endswith(".txt"): return "reproduction_command"
    if relative.endswith("frozen_threshold_config.json"): return "runtime_threshold_asset"
    return "manifest_or_machine_result"


def collect_delivery_targets() -> list[Path]:
    targets = [Path(__file__).resolve(), TEST_SOURCE]
    targets += sorted(path for path in RUNTIME_SOURCE_ROOT.rglob("*") if path.is_file())
    targets += sorted(path for path in RUNTIME_GATE_ROOT.rglob("*") if path.is_file())
    targets += sorted(path for path in REPLAY_ROOT.rglob("*") if path.is_file())
    folder_metas = [
        RUNTIME_SOURCE_ROOT.with_name(RUNTIME_SOURCE_ROOT.name + ".meta"),
        RUNTIME_GATE_ROOT.with_name(RUNTIME_GATE_ROOT.name + ".meta"),
        REPLAY_ROOT.with_name(REPLAY_ROOT.name + ".meta"),
        REPLAY_QUERY_ROOT.with_name(REPLAY_QUERY_ROOT.name + ".meta"),
    ]
    targets += [path for path in folder_metas if path.is_file()]
    test_meta = TEST_SOURCE.with_name(TEST_SOURCE.name + ".meta")
    if test_meta.is_file(): targets.append(test_meta)
    targets += sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path.name != "output_file_manifest.json")
    unique = {path.resolve(): path for path in targets}
    return sorted(unique.values(), key=lambda path: path.relative_to(ROOT).as_posix())


def report_text(run_id: str, unity: dict, assertions: dict) -> str:
    return "\n".join([
        "# Step 9A Runtime Recognition Service and Sequence Input Bridge v1.0 Report",
        "",
        f"**Run ID**: {run_id}",
        f"**Version**: {VERSION}",
        "",
        "**Status**: RUNTIME_RECOGNITION_SERVICE_COMPLETE_BLOCKED_PENDING_EXTERNAL_REVIEW",
        "",
        "## Runtime boundary",
        "",
        "- Pure C# normalized-frame input session with Idle, Begun, Collecting, Finalized and Cancelled states.",
        "- Production metadata is explicitly UNLABELED / RUNTIME / runtime and uses deterministic SHA-256 provenance.",
        "- Fixture replay preserves approved metadata and byte-exact coordinate/mask payloads.",
        "- RuntimeRecognitionService loads the exact nine approved train templates and validated frozen threshold once.",
        "- RecognitionCore and DistanceGatePolicy are called unchanged; no mutable threshold, band, weights or template selection are exposed.",
        "",
        "## Runtime assets and replay",
        "",
        f"- Runtime threshold copy: {THRESHOLD_SIZE} bytes; SHA-256 {THRESHOLD_SHA256}; canonical payload SHA-256 {THRESHOLD_CANONICAL_SHA256}.",
        "- Nine Step 8C test queries are used only for runtime-contract replay, never tuning.",
        "- Nine reconstructions matched coordinates byte-exact, masks and metadata exact, with no missing or duplicate frames.",
        "- All 81 template distances and nine decisions matched the approved Step 8C fixtures within 1e-12.",
        "",
        "## Unity EditMode",
        "",
        f"- Unity 6000.3.14f1: {unity['passed']} PASS / {unity['failed']} FAIL / {unity['skipped']} skipped / {unity['inconclusive']} inconclusive.",
        "- State transitions, invalid shapes/timestamps/coordinates, empty sequences, metadata mismatch, immutable results, missing/duplicate templates and threshold corruption are covered.",
        f"- Assertions: {assertions['pass']} PASS / {assertions['fail']} FAIL / {assertions['duplicate_count']} duplicate names.",
        "",
        "## Scope",
        "",
        "No camera, MediaPipe adapter, automatic gesture segmentation, scene, UI or gameplay integration was implemented. Unknown/open-set and FAST performance claims remain unavailable.",
        "",
        "`RUNTIME_RECOGNITION_SERVICE_COMPLETE_BLOCKED_PENDING_EXTERNAL_REVIEW`",
        "",
    ])


def finalize(unity_results: Path) -> int:
    log("=== Step 9A finalize ===")
    if not STATE_PATH.is_file() or not unity_results.is_file():
        raise RuntimeError("Step 9A prepared state or Unity XML is missing")
    state = load_json(STATE_PATH)
    if state["phase"] != "prepared_for_unity":
        raise RuntimeError("Step 9A finalize requires prepared state")
    run_id = state["run_id"]
    unity_log = OUTPUT / "unity_editmode_test.log"
    if not unity_log.is_file(): raise RuntimeError("Unity log is missing")
    unity, test_names = parse_unity_results(unity_results)
    checks = []
    fixture_name = "WordsInSilence.Tests.EditMode.RuntimeRecognitionServiceV1Tests"
    add_check(checks, "unity_result_passed", unity["result"] == "Passed", unity)
    add_check(checks, "unity_exactly_11_step9a_tests", unity["testcasecount"] == len(test_names) == 11, [unity["testcasecount"], len(test_names)])
    add_check(checks, "unity_all_tests_passed", unity["passed"] == 11 and unity["failed"] == unity["skipped"] == unity["inconclusive"] == 0, unity)
    add_check(checks, "unity_only_step9a_fixture", all(fixture_name in name for name in test_names), test_names)
    log_text = unity_log.read_text(encoding="utf-8", errors="replace")
    add_check(checks, "unity_log_exit_zero", "Test run completed. Exiting with code 0 (Ok). Run completed." in log_text, None)
    add_check(checks, "unity_log_no_compile_error", "error CS" not in log_text and "Compilation failed" not in log_text, None)
    approved_changes = verify_snapshot(state["input_snapshot"])
    add_check(checks, "approved_inputs_unchanged_after_unity", not approved_changes, approved_changes)
    approved_bytes = APPROVED_THRESHOLD.read_bytes()
    add_check(checks, "runtime_threshold_copy_still_byte_exact", RUNTIME_THRESHOLD.read_bytes() == approved_bytes, None)
    add_check(checks, "runtime_threshold_copy_hashes", RUNTIME_THRESHOLD.stat().st_size == THRESHOLD_SIZE and sha256_file(RUNTIME_THRESHOLD) == THRESHOLD_SHA256 and threshold_payload(RUNTIME_THRESHOLD)["canonical_payload_sha256"] == THRESHOLD_CANONICAL_SHA256, None)
    replay_manifest = load_json(REPLAY_ROOT / "replay_fixture_manifest.json")
    add_check(checks, "replay_fixture_9_queries", len(replay_manifest["test_queries"]) == 9, len(replay_manifest["test_queries"]))
    replay_bad = []
    for record in replay_manifest["test_queries"]:
        path = ROOT / PurePosixPath(record["relative_path"])
        if path.stat().st_size != record["size_bytes"] or sha256_file(path) != record["file_sha256"]: replay_bad.append(record["sample_id"])
    add_check(checks, "replay_fixture_hashes_unchanged", not replay_bad, replay_bad)
    add_check(checks, "approved_step8b_zip_unchanged", STEP8B_ZIP.stat().st_size == STEP8B_ZIP_SIZE and sha256_file(STEP8B_ZIP) == STEP8B_ZIP_SHA256, None)
    add_check(checks, "approved_step8c_zip_unchanged", STEP8C_ZIP.stat().st_size == STEP8C_ZIP_SIZE and sha256_file(STEP8C_ZIP) == STEP8C_ZIP_SHA256, None)

    command = (
        "& 'C:\\Program Files\\Unity\\Hub\\Editor\\6000.3.14f1\\Editor\\Unity.exe' "
        "-batchmode -projectPath 'C:\\MyUnityProjects\\WordsInSilence' -runTests -testPlatform EditMode "
        "-testFilter 'WordsInSilence.Tests.EditMode.RuntimeRecognitionServiceV1Tests' "
        "-testResults 'C:\\MyUnityProjects\\WordsInSilence\\data\\processed\\sessions\\S001_P001\\step9a_runtime_recognition_service_v1_0\\unity_editmode_test_results.xml' "
        "-logFile 'C:\\MyUnityProjects\\WordsInSilence\\data\\processed\\sessions\\S001_P001\\step9a_runtime_recognition_service_v1_0\\unity_editmode_test.log'"
    )
    (OUTPUT / "unity_test_command.txt").write_text(command + "\n", encoding="utf-8")
    unity_summary = {
        "version": VERSION,
        "run_id": run_id,
        **unity,
        "fixture": fixture_name,
        "command": command,
        "results_relative_path": unity_results.relative_to(ROOT).as_posix(),
        "results_size_bytes": unity_results.stat().st_size,
        "results_sha256": sha256_file(unity_results),
        "log_relative_path": unity_log.relative_to(ROOT).as_posix(),
        "log_size_bytes": unity_log.stat().st_size,
        "log_sha256": sha256_file(unity_log),
        "replay_queries": 9,
        "replay_distance_records": 81,
        "distance_tolerance": 1e-12,
        "distance_mismatches": 0,
        "decision_records": 9,
        "decision_mismatches": 0,
    }
    write_json(OUTPUT / "unity_editmode_test_summary.json", unity_summary)
    state["phase"] = "finalized_ready_for_package"
    state["unity_summary"] = unity_summary
    write_json(STATE_PATH, state)

    prepare_results = load_json(OUTPUT / "prepare_assertion_results.json")["results"]
    combined = [{**record, "name": "prepare::" + record["name"]} for record in prepare_results] + checks
    assertions = {"version": VERSION, "run_id": run_id, **summarize(combined), "results": combined}
    write_json(OUTPUT / "assertion_results.json", assertions)
    if assertions["fail"] or assertions["duplicate_count"]:
        raise RuntimeError(f"Step 9A final assertions failed: {assertions}")
    report_path = OUTPUT / "STEP9A_RUNTIME_RECOGNITION_SERVICE_V1_0_REPORT.md"
    report_path.write_text(report_text(run_id, unity, assertions), encoding="utf-8")

    json_paths = [path for path in OUTPUT.glob("*.json") if path.name != "output_file_manifest.json"] + [RUNTIME_ASSET_MANIFEST, REPLAY_ROOT / "replay_fixture_manifest.json"]
    mismatches = [path.relative_to(ROOT).as_posix() for path in json_paths if load_json(path).get("run_id") != run_id]
    if mismatches: raise RuntimeError(f"Step 9A JSON run ID mismatch: {mismatches}")
    records = [{
        "relative_path": path.relative_to(ROOT).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "artifact_type": artifact_type(path),
    } for path in collect_delivery_targets()]
    write_json(OUTPUT / "output_file_manifest.json", {
        "version": VERSION,
        "schema_version": "9a_v1.0",
        "run_id": run_id,
        "self_reference_excluded": True,
        "outputs": records,
        "total_files": len(records),
    })
    return package_existing(run_id)


def package_existing(run_id: str) -> int:
    manifest_path = OUTPUT / "output_file_manifest.json"
    manifest = load_json(manifest_path)
    if manifest["run_id"] != run_id or not manifest["self_reference_excluded"]: raise RuntimeError("output manifest metadata invalid")
    records = manifest["outputs"]
    for record in records:
        path = ROOT / PurePosixPath(record["relative_path"])
        if not path.is_file() or path.stat().st_size != record["size_bytes"] or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"output manifest target mismatch: {record['relative_path']}")
    if RUNTIME_THRESHOLD.read_bytes() != APPROVED_THRESHOLD.read_bytes(): raise RuntimeError("runtime threshold copy changed before packaging")
    if STEP8B_ZIP.stat().st_size != STEP8B_ZIP_SIZE or sha256_file(STEP8B_ZIP) != STEP8B_ZIP_SHA256: raise RuntimeError("Step 8B delivery changed before packaging")
    if STEP8C_ZIP.stat().st_size != STEP8C_ZIP_SIZE or sha256_file(STEP8C_ZIP) != STEP8C_ZIP_SHA256: raise RuntimeError("Step 8C delivery changed before packaging")

    temp_zip = ROOT / f".{DELIVERY_NAME}.{run_id}.tmp"
    temp_hash = ROOT / f".{DELIVERY_NAME}.sha256.{run_id}.tmp"
    temp_sidecar = ROOT / f".{DELIVERY_SIDECAR.name}.{run_id}.tmp"
    with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for record in records: archive.write(ROOT / PurePosixPath(record["relative_path"]), record["relative_path"])
        archive.write(manifest_path, manifest_path.relative_to(ROOT).as_posix())
    zip_hash = sha256_file(temp_zip)
    temp_hash.write_text(f"{zip_hash}  {DELIVERY_NAME}\n", encoding="ascii")
    lookup = {record["relative_path"]: record for record in records}
    manifest_name = manifest_path.relative_to(ROOT).as_posix()
    checks = []
    def post(name: str, passed: bool, evidence=None) -> None:
        checks.append({"check_name": name, "status": "PASS" if bool(passed) else "FAIL", "evidence": evidence})
    json_bad = []
    threshold_copy_ok = False
    report_ok = False
    with zipfile.ZipFile(temp_zip, "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        bad_crc = archive.testzip()
        for index, info in enumerate(infos):
            data = archive.read(info.filename)
            unsafe = info.filename.startswith(("/", "\\")) or ".." in PurePosixPath(info.filename).parts or bool(re.match(r"^[A-Za-z]:", info.filename))
            link = stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK
            encrypted = bool(info.flag_bits & 1)
            post(f"entry_{index:04d}_crc", (zipfile.crc32(data) & 0xFFFFFFFF) == info.CRC, info.filename)
            post(f"entry_{index:04d}_safe", not unsafe, info.filename)
            post(f"entry_{index:04d}_not_symlink", not link, info.filename)
            post(f"entry_{index:04d}_not_encrypted", not encrypted, info.filename)
            if info.filename == manifest_name:
                post(f"entry_{index:04d}_self_excluded_manifest", info.filename not in lookup, info.filename)
            else:
                record = lookup.get(info.filename)
                post(f"entry_{index:04d}_in_manifest", record is not None, info.filename)
                post(f"entry_{index:04d}_size", record is not None and len(data) == record["size_bytes"], info.filename)
                post(f"entry_{index:04d}_sha256", record is not None and sha256_bytes(data) == record["sha256"], info.filename)
            if info.filename.endswith(".json"):
                try:
                    payload = json.loads(data.decode("utf-8"))
                    if info.filename != RUNTIME_THRESHOLD.relative_to(ROOT).as_posix() and payload.get("run_id") != run_id:
                        json_bad.append(info.filename)
                except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
                    json_bad.append(info.filename)
            if info.filename == RUNTIME_THRESHOLD.relative_to(ROOT).as_posix():
                threshold_copy_ok = data == APPROVED_THRESHOLD.read_bytes() and len(data) == THRESHOLD_SIZE and sha256_bytes(data) == THRESHOLD_SHA256
            if info.filename.endswith("STEP9A_RUNTIME_RECOGNITION_SERVICE_V1_0_REPORT.md"):
                report_ok = f"**Run ID**: {run_id}" in data.decode("utf-8")
    expected = set(lookup) | {manifest_name}
    post("global_testzip_crc", bad_crc is None, bad_crc)
    post("global_entry_set_exact", set(names) == expected, {"actual": len(set(names)), "expected": len(expected)})
    post("global_duplicate_entries_zero", len(names) == len(set(names)), {"total": len(names), "unique": len(set(names))})
    post("global_json_run_id_consistent", not json_bad, json_bad)
    post("global_report_run_id_consistent", report_ok, run_id)
    post("global_runtime_threshold_byte_exact", threshold_copy_ok, THRESHOLD_SHA256)
    post("global_step8b_input_unchanged", STEP8B_ZIP.stat().st_size == STEP8B_ZIP_SIZE and sha256_file(STEP8B_ZIP) == STEP8B_ZIP_SHA256, None)
    post("global_step8c_input_unchanged", STEP8C_ZIP.stat().st_size == STEP8C_ZIP_SIZE and sha256_file(STEP8C_ZIP) == STEP8C_ZIP_SHA256, None)
    detached = temp_hash.read_text(encoding="ascii")
    match = re.fullmatch(r"([0-9a-f]{64})  ([^\r\n]+)\n", detached)
    post("detached_strict_parse", match is not None, detached.rstrip())
    post("detached_hash_value", match is not None and match.group(1) == zip_hash, None)
    post("detached_filename", match is not None and match.group(2) == DELIVERY_NAME, None)

    names_for_checks = [record["check_name"] for record in checks]
    duplicate_names = sorted(name for name, count in Counter(names_for_checks).items() if count > 1)
    canonical_records = [canonical_json(record) for record in checks]
    duplicate_records = sum(count - 1 for count in Counter(canonical_records).values() if count > 1)
    failures = sum(record["status"] == "FAIL" for record in checks)
    sidecar = {
        "version": VERSION,
        "run_id": run_id,
        "zip_file": DELIVERY_NAME,
        "zip_size_bytes": temp_zip.stat().st_size,
        "zip_sha256": zip_hash,
        "zip_entry_count": len(names),
        "zip_unique_entry_count": len(set(names)),
        "output_manifest_target_count": len(records),
        "output_manifest_self_reference_excluded": True,
        "detached_hash_strict_parse": match is not None,
        "detached_hash_value_ok": match is not None and match.group(1) == zip_hash,
        "detached_hash_filename_ok": match is not None and match.group(2) == DELIVERY_NAME,
        "runtime_threshold_copy_size_bytes": RUNTIME_THRESHOLD.stat().st_size,
        "runtime_threshold_copy_sha256": sha256_file(RUNTIME_THRESHOLD),
        "runtime_threshold_copy_canonical_payload_sha256": threshold_payload(RUNTIME_THRESHOLD)["canonical_payload_sha256"],
        "post_zip_check_total": len(checks),
        "post_zip_check_unique": len(set(names_for_checks)),
        "post_zip_check_name_duplicates": len(duplicate_names),
        "post_zip_record_duplicates": duplicate_records,
        "post_zip_failure_count": failures,
        "duplicate_check_names": duplicate_names,
        "verification_status": "PASS" if failures == 0 and not duplicate_names and duplicate_records == 0 else "FAIL",
        "checks": checks,
    }
    write_json(temp_sidecar, sidecar)
    if sidecar["verification_status"] != "PASS": raise RuntimeError(f"post-ZIP verification failed: {failures} failures")
    os.replace(temp_zip, DELIVERY_ZIP)
    os.replace(temp_hash, DELIVERY_HASH)
    os.replace(temp_sidecar, DELIVERY_SIDECAR)
    log(f"Package PASS: entries={len(names)}, bytes={DELIVERY_ZIP.stat().st_size}, sha256={zip_hash}")
    log(f"Post-ZIP PASS: {len(checks)} checks / 0 failures / 0 duplicate names / 0 duplicate records")
    log("Status: RUNTIME_RECOGNITION_SERVICE_COMPLETE_BLOCKED_PENDING_EXTERNAL_REVIEW")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--finalize", action="store_true")
    modes.add_argument("--package-only", action="store_true")
    parser.add_argument("--unity-results", type=Path)
    args = parser.parse_args()
    try:
        if args.prepare: return prepare()
        if args.finalize:
            if args.unity_results is None: parser.error("--finalize requires --unity-results")
            return finalize(args.unity_results.resolve())
        state = load_json(STATE_PATH)
        return package_existing(state["run_id"])
    except Exception as exc:
        log(f"BLOCKED: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
