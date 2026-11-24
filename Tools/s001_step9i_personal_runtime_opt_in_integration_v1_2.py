#!/usr/bin/env python3
"""WordsInSilence Step 9I Unity first-run two-failure repair v1.2.

The tool validates that exactly the approved production loader and EditMode
fixture changed relative to Step 9I v1.1, executes static and offline runtime
regressions for the two confirmed failures, and packages the evidence through
the externally approved v1.1 transaction implementation. It never invokes
Unity Editor, batchmode, or the NUnit fixture.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tempfile
from typing import Any
import zipfile


SCHEMA = "words_in_silence.step9i.personal_runtime_opt_in_integration.v1_2"
V1_1_REL = Path("Validation/WordsInSilence_step9i_personal_runtime_opt_in_integration_v1_1_delivery.zip")
V1_1_SIZE = 447_805
V1_1_SHA256 = "eb63427300a341da2ef52be4521adfd742c4bc0d30ca20a760121af8c2b7fbbf"
V1_1_DETACHED_SIZE = 142
V1_1_DETACHED_FILE_SHA256 = "aa905e18a5cca651b0e335cdd69f2e8c776d6b5e7eef07abf6dc428d35792069"
V1_0_REL = Path("Validation/WordsInSilence_step9i_personal_runtime_opt_in_integration_v1_0_delivery.zip")
V1_0_SIZE = 456_746
V1_0_SHA256 = "e126f42232756890ee1b7c4457f38694e5d43b8190e7bb6aaeaa94243a9967a7"
V1_0_DETACHED_SIZE = 142
V1_0_DETACHED_FILE_SHA256 = "18b85988a09ef61d0692184fae3a377643b5fe9173765aa42d8c0db614729f61"
V1_1_TOOL_REL = Path("Tools/s001_step9i_personal_runtime_opt_in_integration_v1_1.py")
V1_1_TOOL_SIZE = 37_722
V1_1_TOOL_SHA256 = "30471dc6725bdbdd0cc80c345b238485d1cb0d8d7a5933aac7477d536d714caf"

PRODUCTION_REL = Path("Assets/_Project/Scripts/Recognition/Runtime/PersonalRuntimeOptInProfile.cs")
TEST_REL = Path("Assets/_Project/Tests/EditMode/PersonalRuntimeOptInIntegrationV1Tests.cs")
SCORES_REL = Path("Assets/_Project/Tests/EditMode/Fixtures/PersonalRuntimeOptInIntegrationV1/personal_expected_scores.csv")
QUERY_MANIFEST_REL = Path("Assets/_Project/Tests/EditMode/Fixtures/PersonalRuntimeOptInIntegrationV1/personal_query_manifest.json")
PROFILE_REL = Path("Assets/StreamingAssets/RecognitionCoreV1/PersonalProfiles/step9h_current_user_v1_0")

PRODUCTION_BEFORE_SHA256 = "2670cf9531133fb97ff534b3b9b592c87bc172ca21c25fb88320620526422aae"
TEST_BEFORE_SHA256 = "b8fd35e089a588db54da11e6c6234931d3ebec721da8f9ee82ee8d649393feb7"
SCORES_SHA256 = "c3f98a8c49a52acd018e3366d9fbe88453e529256c51a77d92e1c3e8069a97aa"
WORDS = ("GREETING", "LOVE", "THANK_YOU")

PACKAGED_PRODUCTION = "preserved_project_files/production_source/PersonalRuntimeOptInProfile.cs"
PACKAGED_TEST = "preserved_project_files/test_source/PersonalRuntimeOptInIntegrationV1Tests.cs"

RUNTIME_SOURCES = (
    "Assets/_Project/Scripts/Recognition/Core/RecognitionSequence.cs",
    "Assets/_Project/Scripts/Recognition/Core/RecognitionSequenceBinary.cs",
    "Assets/_Project/Scripts/Recognition/Core/MaskAwareDtw.cs",
    "Assets/_Project/Scripts/Recognition/Core/RecognitionCore.cs",
    "Assets/_Project/Scripts/Recognition/Evaluation/DistanceGatePolicy.cs",
    "Assets/_Project/Scripts/Recognition/Evaluation/FrozenDistanceGateConfig.cs",
    "Assets/_Project/Scripts/Recognition/Runtime/RuntimeRecognitionService.cs",
    PRODUCTION_REL.as_posix(),
)

NUNIT_UNITY_STUB = """using System;
using System.Collections.Generic;
namespace NUnit.Framework {
  [AttributeUsage(AttributeTargets.Method)] public sealed class TestAttribute : Attribute {}
  [AttributeUsage(AttributeTargets.Method)] public sealed class OneTimeSetUpAttribute : Attribute {}
  public delegate void TestDelegate();
  public static class Assert {
    public static void AreEqual<T>(T expected, T actual, string message = null) {}
    public static void AreEqual(double expected, double actual, double tolerance, string message = null) {}
    public static void AreNotEqual<T>(T expected, T actual) {}
    public static void IsTrue(bool value, string message = null) {}
    public static void IsFalse(bool value, string message = null) {}
    public static void IsNull(object value, string message = null) {}
    public static void LessOrEqual(double actual, double expected) {}
    public static T Throws<T>(TestDelegate code) where T : Exception { return null; }
  }
  public static class CollectionAssert {
    public static void AreEqual<T>(IEnumerable<T> expected, IEnumerable<T> actual) {}
    public static void AreEquivalent<T>(IEnumerable<T> expected, IEnumerable<T> actual) {}
  }
}
namespace UnityEngine {
  public static class Application {
    public static string dataPath { get { return string.Empty; } }
    public static string streamingAssetsPath { get { return string.Empty; } }
  }
  public static class JsonUtility {
    public static T FromJson<T>(string value) { return default(T); }
  }
}
"""

TYPED_FAILURE_PROBE = r"""using System;
using System.IO;
using WordsInSilence.Recognition.Runtime;
class TypedFailureProbe {
  static string CopyProfile(string source) {
    string destination = Path.Combine(Path.GetTempPath(), "WIS-Step9I-v12-probe-" + Guid.NewGuid().ToString("N"));
    Directory.CreateDirectory(destination);
    foreach (string file in Directory.GetFiles(source, "*", SearchOption.TopDirectoryOnly))
      if (!file.EndsWith(".meta", StringComparison.OrdinalIgnoreCase))
        File.Copy(file, Path.Combine(destination, Path.GetFileName(file)));
    return destination;
  }
  static PersonalProfileFailureCode Run(string source, bool missing) {
    string copy = CopyProfile(source);
    try {
      if (missing)
        File.Delete(Path.Combine(copy, "personal_template_LOVE.wisrt"));
      else
        File.Copy(Path.Combine(copy, "personal_template_GREETING.wisrt"), Path.Combine(copy, "unexpected_extra.wisrt"));
      try {
        PersonalRuntimeProfileLoader.Load(copy);
        throw new Exception("loader unexpectedly succeeded");
      } catch (PersonalProfileLoadException error) {
        return error.FailureCode;
      }
    } finally {
      Directory.Delete(copy, true);
    }
  }
  static int Main(string[] args) {
    PersonalProfileFailureCode missing = Run(args[0], true);
    PersonalProfileFailureCode extra = Run(args[0], false);
    Console.WriteLine("MISSING=" + missing);
    Console.WriteLine("EXTRA=" + extra);
    return missing == PersonalProfileFailureCode.AssetMissing &&
           extra == PersonalProfileFailureCode.WordSetMismatch ? 0 : 1;
  }
}
"""


class Step9IRepairError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Step9IRepairError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def json_bytes(value: Any, *, pretty: bool = True) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def safe_zip_name(name: str) -> bool:
    pure = PurePosixPath(name)
    return (
        bool(name)
        and not name.startswith(("/", "\\"))
        and "\\" not in name
        and ":" not in pure.parts[0]
        and all(part not in ("", ".", "..") for part in pure.parts)
    )


def inspect_archive(data: bytes, expected_names: set[str]) -> dict[str, Any]:
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


def load_fixed_archive(
    root: Path,
    relative: Path,
    size: int,
    digest: str,
    detached_size: int,
    detached_digest: str,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    path = root / relative
    detached = Path(str(path) + ".sha256")
    require(path.is_file() and detached.is_file(), f"fixed archive missing: {relative}")
    data = path.read_bytes()
    checksum = detached.read_bytes()
    require(len(data) == size and sha256_bytes(data) == digest, f"fixed archive identity mismatch: {relative}")
    require(
        len(checksum) == detached_size and sha256_bytes(checksum) == detached_digest,
        f"fixed detached identity mismatch: {relative}",
    )
    require(
        checksum == (digest + "  " + path.name + "\n").encode("ascii"),
        f"fixed detached strict syntax mismatch: {relative}",
    )
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        file_map = {name: archive.read(name) for name in names}
    manifest = json.loads(file_map["output_file_manifest.json"])
    records = manifest["outputs"]
    expected = {record["relative_path"] for record in records} | {"output_file_manifest.json"}
    safety = inspect_archive(data, expected)
    require(
        all(
            len(file_map[record["relative_path"]]) == record["size_bytes"]
            and sha256_bytes(file_map[record["relative_path"]]) == record["sha256"]
            for record in records
        ),
        f"fixed manifest size/hash mismatch: {relative}",
    )
    return (
        {
            "relative_path": relative.as_posix(),
            "size_bytes": len(data),
            "sha256": sha256_bytes(data),
            "detached_size_bytes": len(checksum),
            "detached_file_sha256": sha256_bytes(checksum),
            "manifest_target_count": len(records),
            "archive_safety": safety,
        },
        file_map,
    )


def load_v1_1_tool(root: Path):
    path = root / V1_1_TOOL_REL
    require(path.is_file(), "approved v1.1 transaction tool missing")
    require(path.stat().st_size == V1_1_TOOL_SIZE, "approved v1.1 transaction tool size mismatch")
    require(sha256_file(path) == V1_1_TOOL_SHA256, "approved v1.1 transaction tool SHA-256 mismatch")
    spec = importlib.util.spec_from_file_location("step9i_v1_1_approved", path)
    require(spec is not None and spec.loader is not None, "cannot load approved v1.1 transaction tool")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(hasattr(module, "commit_packaging_transaction"), "approved transaction function missing")
    return module


def current_project_path(root: Path, packaged_name: str, v1_1: Any) -> Path:
    prefix = "preserved_project_files/"
    require(packaged_name.startswith(prefix), f"not a v1.1 preserved project file: {packaged_name}")
    return v1_1.current_project_path(root, "project_files/" + packaged_name[len(prefix) :])


def verify_project_preservation(
    root: Path,
    file_map: dict[str, bytes],
    v1_1: Any,
) -> dict[str, Any]:
    names = sorted(name for name in file_map if name.startswith("preserved_project_files/"))
    require(len(names) == 44, "v1.1 preserved project file count mismatch")
    allowed = {PACKAGED_PRODUCTION, PACKAGED_TEST}
    records: list[dict[str, Any]] = []
    changed: list[str] = []
    for name in names:
        path = current_project_path(root, name, v1_1)
        require(path.is_file(), f"project file missing: {path}")
        before = file_map[name]
        after = path.read_bytes()
        different = before != after
        if different:
            changed.append(name)
        records.append(
            {
                "v1_1_packaged_path": name,
                "project_relative_path": path.relative_to(root).as_posix(),
                "before_size_bytes": len(before),
                "before_sha256": sha256_bytes(before),
                "after_size_bytes": len(after),
                "after_sha256": sha256_bytes(after),
                "changed": different,
                "change_allowed": name in allowed,
            }
        )
    require(set(changed) == allowed, f"project changes are not exactly the two approved files: {changed}")
    require(sha256_bytes(file_map[PACKAGED_PRODUCTION]) == PRODUCTION_BEFORE_SHA256, "production pre-fix hash mismatch")
    require(sha256_bytes(file_map[PACKAGED_TEST]) == TEST_BEFORE_SHA256, "test pre-fix hash mismatch")
    return {
        "schema_version": SCHEMA + ".project_preservation",
        "v1_1_project_file_count": len(records),
        "changed_project_file_count": len(changed),
        "changed_project_files": sorted(changed),
        "unchanged_project_file_count": len(records) - len(changed),
        "unchanged_byte_identical_count": sum(not record["changed"] for record in records),
        "unexpected_change_count": sum(record["changed"] and not record["change_allowed"] for record in records),
        "records": records,
    }


def expected_score_key_regression(root: Path) -> dict[str, Any]:
    with (root / SCORES_REL).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    require(rows, "expected score CSV empty")
    header, data = rows[0], rows[1:]
    require(header[1] == "query_id" and header[3] == "target_word", "expected score CSV column contract mismatch")
    keys = [row[1] + "|" + row[3] for row in data]
    manifest = json.loads((root / QUERY_MANIFEST_REL).read_text(encoding="utf-8"))
    runtime_keys = [query["query_id"] + "|" + word for query in manifest["queries"] for word in WORDS]
    key_set = set(keys)
    test_text = (root / TEST_REL).read_text(encoding="utf-8")
    result = {
        "schema_version": SCHEMA + ".expected_score_key_regression",
        "csv_relative_path": SCORES_REL.as_posix(),
        "csv_size_bytes": (root / SCORES_REL).stat().st_size,
        "csv_sha256": sha256_file(root / SCORES_REL),
        "csv_data_rows": len(data),
        "constructed_keys": len(keys),
        "unique_keys": len(key_set),
        "queries": len(manifest["queries"]),
        "target_words": sorted({row[3] for row in data}),
        "runtime_lookup_key_count": len(runtime_keys),
        "runtime_lookup_keys_present": sum(key in key_set for key in runtime_keys),
        "missing_runtime_lookup_keys": sorted(set(runtime_keys) - key_set),
        "source_uses_query_id_and_target_word_columns": 'row[1] + "|" + row[3]' in test_text,
        "obsolete_key_expression_absent": 'row[0] + "|" + row[2]' not in test_text,
    }
    passed = (
        result["csv_sha256"] == SCORES_SHA256
        and result["csv_data_rows"] == 27
        and result["constructed_keys"] == 27
        and result["unique_keys"] == 27
        and result["queries"] == 9
        and result["target_words"] == list(WORDS)
        and result["runtime_lookup_keys_present"] == 27
        and not result["missing_runtime_lookup_keys"]
        and result["source_uses_query_id_and_target_word_columns"]
        and result["obsolete_key_expression_absent"]
    )
    result["verdict"] = "PASS" if passed else "FAIL"
    require(passed, "expected-score key regression failed")
    return result


def static_typed_failure_control_flow(root: Path) -> dict[str, Any]:
    production = (root / PRODUCTION_REL).read_text(encoding="utf-8")
    test = (root / TEST_REL).read_text(encoding="utf-8")
    read_start = production.index("static byte[] ReadAsset(")
    read_end = production.index("static void ValidateAssetSha", read_start)
    read_asset = production[read_start:read_end]
    count_anchor = 'Directory.GetFiles(profileDirectory, "*.wisrt", SearchOption.TopDirectoryOnly).Length != 3'
    count_index = production.index(count_anchor)
    template_load_index = production.rindex("byte[] bytes = ReadAsset(assetPath, asset)", 0, count_index)
    return_index = production.index("return new PersonalRuntimeProfile(", count_index)
    missing_index = test.index('File.Delete(Path.Combine(Copy, "personal_template_LOVE.wisrt"))')
    missing_excerpt = test[missing_index : missing_index + 240]
    extra_index = test.index('Path.Combine(Copy, "unexpected_extra.wisrt")')
    extra_excerpt = test[extra_index : extra_index + 240]
    result = {
        "schema_version": SCHEMA + ".typed_failure_static_control_flow",
        "read_asset_checks_current_file_exists": "if (!File.Exists(path))" in read_asset,
        "read_asset_checks_current_file_size": "info.Length != asset.SizeBytes" in read_asset,
        "read_asset_has_no_directory_wisrt_count_check": "Directory.GetFiles" not in read_asset,
        "required_template_load_precedes_directory_count_check": template_load_index < count_index,
        "directory_count_check_precedes_profile_return": count_index < return_index,
        "directory_count_failure_code": "WordSetMismatch",
        "missing_love_test_expected_code": "AssetMissing" if "PersonalProfileFailureCode.AssetMissing" in missing_excerpt else "MISSING",
        "additional_wisrt_test_expected_code": "WordSetMismatch" if "PersonalProfileFailureCode.WordSetMismatch" in extra_excerpt else "MISSING",
        "unexpected_extra_fixture_name_present": "unexpected_extra.wisrt" in test,
        "nunit_test_count": len(re.findall(r"\[Test\]", test)),
    }
    passed = (
        all(
            result[key]
            for key in (
                "read_asset_checks_current_file_exists",
                "read_asset_checks_current_file_size",
                "read_asset_has_no_directory_wisrt_count_check",
                "required_template_load_precedes_directory_count_check",
                "directory_count_check_precedes_profile_return",
                "unexpected_extra_fixture_name_present",
            )
        )
        and result["missing_love_test_expected_code"] == "AssetMissing"
        and result["additional_wisrt_test_expected_code"] == "WordSetMismatch"
        and result["nunit_test_count"] == 18
    )
    result["verdict"] = "PASS" if passed else "FAIL"
    require(passed, "typed-failure static control-flow regression failed")
    return result


def compiler_error_count(text: str) -> int:
    return len(re.findall(r"\berror CS[0-9]+\b", text))


def compiler_warning_count(text: str) -> int:
    return len(re.findall(r"\bwarning CS[0-9]+\b", text))


def offline_compile_and_probe(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    mono = Path(r"C:\Program Files\Unity\Hub\Editor\6000.3.14f1\Editor\Data\MonoBleedingEdge\bin\mono.exe")
    csc = Path(r"C:\Program Files\Unity\Hub\Editor\6000.3.14f1\Editor\Data\MonoBleedingEdge\lib\mono\4.5\csc.exe")
    require(mono.is_file() and csc.is_file(), "Unity Mono Roslyn compiler unavailable")
    sources = [str((root / relative).resolve()) for relative in RUNTIME_SOURCES]
    with tempfile.TemporaryDirectory(prefix="WIS-Step9I-v12-offline-") as temporary:
        temp = Path(temporary)
        runtime_dll = temp / "Step9IRuntime.dll"
        production_command = [
            str(mono), str(csc), "/nologo", "/target:library", "/langversion:latest",
            "/out:" + str(runtime_dll), *sources,
        ]
        production = subprocess.run(
            production_command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120
        )
        production_output = (production.stdout or "") + (production.stderr or "")
        require(production.returncode == 0, "offline production compile failed:\n" + production_output[-4000:])

        stub_path = temp / "NUnitUnityCompileStubs.cs"
        stub_path.write_text(NUNIT_UNITY_STUB, encoding="utf-8")
        test_dll = temp / "Step9ITestFixture.dll"
        test_command = [
            str(mono), str(csc), "/nologo", "/target:library", "/langversion:latest",
            "/out:" + str(test_dll), "/r:" + str(runtime_dll),
            str(stub_path), str((root / TEST_REL).resolve()),
        ]
        test = subprocess.run(
            test_command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120
        )
        test_output = (test.stdout or "") + (test.stderr or "")
        require(test.returncode == 0, "offline test fixture compile failed:\n" + test_output[-4000:])

        probe_path = temp / "TypedFailureProbe.cs"
        probe_path.write_text(TYPED_FAILURE_PROBE, encoding="utf-8")
        probe_exe = temp / "TypedFailureProbe.exe"
        probe_compile_command = [
            str(mono), str(csc), "/nologo", "/target:exe", "/langversion:latest",
            "/out:" + str(probe_exe), "/r:" + str(runtime_dll), str(probe_path),
        ]
        probe_compile = subprocess.run(
            probe_compile_command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120
        )
        probe_compile_output = (probe_compile.stdout or "") + (probe_compile.stderr or "")
        require(probe_compile.returncode == 0, "typed-failure probe compile failed:\n" + probe_compile_output[-4000:])
        probe = subprocess.run(
            [str(mono), str(probe_exe), str((root / PROFILE_REL).resolve())],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        probe_output = ((probe.stdout or "") + (probe.stderr or "")).lstrip("\ufeff")
        require(probe.returncode == 0, "typed-failure runtime probe failed:\n" + probe_output[-4000:])

    compile_evidence = {
        "schema_version": SCHEMA + ".offline_csharp_compile",
        "compiler": "unity_mono_roslyn_csc",
        "target_framework": "Unity Mono 4.5 reference profile",
        "production_source_count": len(RUNTIME_SOURCES),
        "production_compiler_error_count": compiler_error_count(production_output),
        "production_compiler_warning_count": compiler_warning_count(production_output),
        "test_fixture_source_count": 1,
        "test_fixture_compile_stub_count": 1,
        "test_fixture_compiler_error_count": compiler_error_count(test_output),
        "test_fixture_compiler_warning_count": compiler_warning_count(test_output),
        "probe_compiler_error_count": compiler_error_count(probe_compile_output),
        "unity_editor_invoked": False,
        "unity_compile_pass_claimed": False,
        "editmode_pass_claimed": False,
        "status": "PASS",
    }
    require(
        compile_evidence["production_compiler_error_count"] == 0
        and compile_evidence["test_fixture_compiler_error_count"] == 0
        and compile_evidence["probe_compiler_error_count"] == 0,
        "offline compiler error count is nonzero",
    )
    typed_probe = {
        "schema_version": SCHEMA + ".typed_failure_runtime_probe",
        "runtime_under_test": "PersonalRuntimeProfileLoader.Load",
        "profile_copy_isolated_in_system_temp": True,
        "missing_required_template": {
            "mutation": "delete personal_template_LOVE.wisrt",
            "expected_failure_code": "AssetMissing",
            "actual_failure_code": "AssetMissing" if "MISSING=AssetMissing" in probe_output else "UNEXPECTED",
        },
        "additional_wisrt": {
            "mutation": "copy GREETING to unexpected_extra.wisrt",
            "expected_failure_code": "WordSetMismatch",
            "actual_failure_code": "WordSetMismatch" if "EXTRA=WordSetMismatch" in probe_output else "UNEXPECTED",
        },
        "probe_exit_code": probe.returncode,
        "project_files_modified": 0,
    }
    probe_pass = (
        typed_probe["missing_required_template"]["actual_failure_code"] == "AssetMissing"
        and typed_probe["additional_wisrt"]["actual_failure_code"] == "WordSetMismatch"
        and typed_probe["probe_exit_code"] == 0
    )
    typed_probe["verdict"] = "PASS" if probe_pass else "FAIL"
    require(probe_pass, "typed failure runtime probe result mismatch")
    return compile_evidence, typed_probe


def unified_diff(before: bytes, after: bytes, before_name: str, after_name: str) -> bytes:
    text = "".join(
        difflib.unified_diff(
            before.decode("utf-8").splitlines(keepends=True),
            after.decode("utf-8").splitlines(keepends=True),
            fromfile=before_name,
            tofile=after_name,
            lineterm="\n",
        )
    )
    return text.encode("utf-8")


def global_asset_preservation(root: Path, v1_0_map: dict[str, bytes]) -> dict[str, Any]:
    snapshot = json.loads(v1_0_map["global_runtime_assets_before_after.json"])["before"]
    records = []
    for expected in snapshot["assets"]:
        path = root / PurePosixPath(expected["relative_path"])
        actual_size = path.stat().st_size if path.is_file() else None
        actual_sha = sha256_file(path) if path.is_file() else None
        records.append(
            {
                **expected,
                "actual_size_bytes": actual_size,
                "actual_sha256": actual_sha,
                "byte_identical": actual_size == expected["size_bytes"] and actual_sha == expected["sha256"],
            }
        )
    mismatch = sum(not record["byte_identical"] for record in records)
    require(len(records) == 11 and mismatch == 0, "global runtime asset preservation mismatch")
    return {
        "schema_version": SCHEMA + ".global_asset_preservation",
        "asset_count": len(records),
        "byte_identical_count": len(records) - mismatch,
        "mismatch_count": mismatch,
        "records": records,
    }


def assertion(key: str, status: str, subject: str, evidence: Any) -> dict[str, Any]:
    return {"assertion_key": key, "status": status, "subject": subject, "evidence": evidence}


def output_category(name: str) -> str:
    if name == "ASSERTIONS.jsonl":
        return "assertions"
    if name == "VALIDATION_REPORT.md":
        return "validation_report"
    if name.startswith("analysis/"):
        return "analysis_source"
    if name.startswith("source_before/"):
        return "source_before"
    if name.startswith("source_after/"):
        return "source_after"
    if name.startswith("unified_diff/"):
        return "unified_diff"
    if name.startswith("input_subset/"):
        return "input_subset"
    return "validation_result"


def report_text(
    key_regression: dict[str, Any],
    typed_probe: dict[str, Any],
    preservation: dict[str, Any],
    compile_evidence: dict[str, Any],
    summary: dict[str, int],
) -> str:
    return "\n".join(
        [
            "# WordsInSilence Step 9I Unity First-Run Failure Repair v1.2",
            "",
            "## Repairs",
            "",
            "- expected-score key: CSV query_id column 1 + target_word column 3",
            "- typed failure precedence: required template reads precede the top-level WISRT count check",
            "- unexpected extra WISRT regression remains WordSetMismatch",
            "",
            "## Static and offline verification",
            "",
            f"- CSV keys: `{key_regression['runtime_lookup_keys_present']}/27`",
            f"- Missing required template runtime probe: `{typed_probe['missing_required_template']['actual_failure_code']}`",
            f"- Additional WISRT runtime probe: `{typed_probe['additional_wisrt']['actual_failure_code']}`",
            f"- Changed project files: `{preservation['changed_project_file_count']}`",
            f"- Unchanged v1.1 project files: `{preservation['unchanged_byte_identical_count']}/42`",
            f"- Offline production compiler errors: `{compile_evidence['production_compiler_error_count']}`",
            f"- Offline fixture compiler errors: `{compile_evidence['test_fixture_compiler_error_count']}`",
            "",
            "## Assertions",
            "",
            f"- PASS: `{summary['PASS']}`",
            f"- FAIL: `{summary['FAIL']}`",
            f"- UNVERIFIED: `{summary['UNVERIFIED']}`",
            f"- NOT_APPLICABLE: `{summary['NOT_APPLICABLE']}`",
            "",
            "## Unity status",
            "",
            "Unity Editor, batchmode, and EditMode were not run. The v1.2 fixture retest remains unverified pending external source/package review.",
            "",
        ]
    )


def build(root: Path, output: Path, delivery: Path) -> dict[str, Any]:
    detached = Path(str(delivery) + ".sha256")
    validation_root = (root / "Validation").resolve()
    require(output.parent.resolve() == validation_root and delivery.parent.resolve() == validation_root, "v1.2 targets must be under Validation")
    for target in (output, delivery, detached):
        require(not target.exists(), f"never-overwrite output target exists: {target}")
    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    require(not staged, f"Git staged files present: {staged}")

    v1_1 = load_v1_1_tool(root)
    fixed_v1_1, v1_1_map = load_fixed_archive(
        root, V1_1_REL, V1_1_SIZE, V1_1_SHA256, V1_1_DETACHED_SIZE, V1_1_DETACHED_FILE_SHA256
    )
    fixed_v1_0, v1_0_map = load_fixed_archive(
        root, V1_0_REL, V1_0_SIZE, V1_0_SHA256, V1_0_DETACHED_SIZE, V1_0_DETACHED_FILE_SHA256
    )
    require(
        v1_1_map["analysis/s001_step9i_personal_runtime_opt_in_integration_v1_1.py"]
        == (root / V1_1_TOOL_REL).read_bytes(),
        "approved v1.1 transaction source differs from Delivery",
    )
    preservation = verify_project_preservation(root, v1_1_map, v1_1)
    key_regression = expected_score_key_regression(root)
    static_control_flow = static_typed_failure_control_flow(root)
    compile_evidence, typed_probe = offline_compile_and_probe(root)
    global_preservation = global_asset_preservation(root, v1_0_map)

    production_before = v1_1_map[PACKAGED_PRODUCTION]
    test_before = v1_1_map[PACKAGED_TEST]
    production_after = (root / PRODUCTION_REL).read_bytes()
    test_after = (root / TEST_REL).read_bytes()
    production_diff = unified_diff(production_before, production_after, "a/" + PRODUCTION_REL.as_posix(), "b/" + PRODUCTION_REL.as_posix())
    test_diff = unified_diff(test_before, test_after, "a/" + TEST_REL.as_posix(), "b/" + TEST_REL.as_posix())
    require(production_diff and test_diff, "required source diff missing")

    first_run = {
        "schema_version": SCHEMA + ".first_run_failure_evidence_summary",
        "unity_version": "6000.3.14f1",
        "fixture": "PersonalRuntimeOptInIntegrationV1Tests",
        "platform": "EditMode",
        "total": 18,
        "passed": 16,
        "failed": 2,
        "compiler_errors": 0,
        "start_utc": "2026-08-09T10:03:06Z",
        "end_utc": "2026-08-09T10:03:07Z",
        "failures": [
            {
                "test": "Loader_RejectsMissingMalformedDuplicateAdditionalAndUnsafeProfilesWithTypedFailures",
                "expected": "AssetMissing",
                "actual": "WordSetMismatch",
                "confirmed_cause": "directory-wide WISRT count check executed while reading the first required template",
            },
            {
                "test": "RuntimeScores_AllTwentySevenMatchApprovedPythonWithinTolerance",
                "exception": "KeyNotFoundException",
                "missing_key": "step9c-take-01-greeting-f7875c16-ad18-425c-8ce5-39e9a6cc13f7|GREETING",
                "confirmed_cause": "expected-score dictionary used score_cell_key and expected_word instead of query_id and target_word",
            },
        ],
    }
    change_inventory = {
        "schema_version": SCHEMA + ".source_change_inventory",
        "changed_project_file_count": 2,
        "changed_project_files": [PRODUCTION_REL.as_posix(), TEST_REL.as_posix()],
        "unchanged_v1_1_project_files": 42,
        "production_loader_change": "relocate directory-wide WISRT count check after three required template loads",
        "test_fixture_changes": [
            "construct expected score key from CSV row[1] query_id and row[3] target_word",
            "add unexpected_extra.wisrt WordSetMismatch regression inside existing test method",
        ],
        "test_count_before": 18,
        "test_count_after": static_control_flow["nunit_test_count"],
        "asset_fixture_threshold_dtw_changes": 0,
    }
    primary = {
        "schema_version": SCHEMA + ".primary_verdict",
        "actual_primary_verdict": "SOURCE_AND_PACKAGE_READY",
        "typed_failure_precedence_repair": "IMPLEMENTED",
        "expected_score_key_repair": "IMPLEMENTED",
        "v1_1_approved_scope_preserved": True,
        "unity_editmode_retest": "UNVERIFIED",
        "status_strings": [
            "STEP9I_PERSONAL_RUNTIME_OPT_IN_INTEGRATION_V1_2_SOURCE_AND_PACKAGE_READY",
            "STEP9I_EDITOR_FIRST_RUN_TYPED_FAILURE_PRECEDENCE_REPAIR_IMPLEMENTED",
            "STEP9I_EDITOR_FIRST_RUN_EXPECTED_SCORE_KEY_REPAIR_IMPLEMENTED",
            "STEP9I_V1_1_APPROVED_SCOPE_PRESERVED",
            "STEP9I_UNITY_EDITMODE_RETEST_UNVERIFIED",
            "BLOCKED_PENDING_EXTERNAL_REVIEW",
        ],
    }
    input_provenance = {
        "schema_version": SCHEMA + ".input_provenance",
        "fixed_step9i_v1_1": fixed_v1_1,
        "fixed_step9i_v1_0_global_asset_provenance": fixed_v1_0,
        "approved_v1_1_transaction_source": {
            "relative_path": V1_1_TOOL_REL.as_posix(),
            "size_bytes": (root / V1_1_TOOL_REL).stat().st_size,
            "sha256": sha256_file(root / V1_1_TOOL_REL),
            "function": "commit_packaging_transaction",
        },
        "fixed_inputs_before_after_unchanged": True,
    }

    assertions = [
        assertion("fixed_v1_1_delivery_identity_archive_manifest", "PASS", V1_1_REL.as_posix(), fixed_v1_1),
        assertion("pre_fix_source_hashes", "PASS", "two approved sources", {"production": PRODUCTION_BEFORE_SHA256, "test": TEST_BEFORE_SHA256}),
        assertion("changed_project_files_exactly_two", "PASS" if preservation["changed_project_file_count"] == 2 else "FAIL", "v1.1 44-file baseline", preservation["changed_project_files"]),
        assertion("expected_score_key_twenty_seven_of_twenty_seven", "PASS" if key_regression["verdict"] == "PASS" else "FAIL", SCORES_REL.as_posix(), key_regression),
        assertion("typed_failure_static_control_flow", "PASS" if static_control_flow["verdict"] == "PASS" else "FAIL", PRODUCTION_REL.as_posix(), static_control_flow),
        assertion("missing_required_template_asset_missing_runtime_probe", "PASS" if typed_probe["missing_required_template"]["actual_failure_code"] == "AssetMissing" else "FAIL", "isolated profile copy", typed_probe["missing_required_template"]),
        assertion("additional_wisrt_word_set_mismatch_runtime_probe", "PASS" if typed_probe["additional_wisrt"]["actual_failure_code"] == "WordSetMismatch" else "FAIL", "isolated profile copy", typed_probe["additional_wisrt"]),
        assertion("unchanged_preserved_project_files_forty_two", "PASS" if preservation["unchanged_byte_identical_count"] == 42 else "FAIL", "v1.1 44-file baseline", {"unchanged": preservation["unchanged_byte_identical_count"], "unexpected": preservation["unexpected_change_count"]}),
        assertion("personal_assets_queries_fixtures_byte_identity", "PASS", "unchanged v1.1 project-file subset", {"changed_outside_two_sources": preservation["unexpected_change_count"]}),
        assertion("global_runtime_assets_eleven_byte_identity", "PASS" if global_preservation["mismatch_count"] == 0 else "FAIL", "global runtime", {"count": 11, "mismatch": global_preservation["mismatch_count"]}),
        assertion("offline_production_compile_error_zero", "PASS" if compile_evidence["production_compiler_error_count"] == 0 else "FAIL", "Unity Mono Roslyn", compile_evidence),
        assertion("offline_test_fixture_compile_error_zero", "PASS" if compile_evidence["test_fixture_compiler_error_count"] == 0 else "FAIL", "Unity Mono Roslyn with compile-only API stubs", compile_evidence),
        assertion("approved_v1_1_transaction_reused", "PASS", V1_1_TOOL_REL.as_posix(), "commit_packaging_transaction"),
        assertion("unity_editor_not_run", "PASS", "v1.2 repair scope", True),
        assertion("unity_editmode_retest", "UNVERIFIED", "PersonalRuntimeOptInIntegrationV1Tests", "pending external source/package review"),
        assertion("git_staged_zero", "PASS", "repository", 0),
        assertion("never_overwrite", "PASS", "v1.2 final targets", "all absent before transaction"),
    ]
    require(len({record["assertion_key"] for record in assertions}) == len(assertions), "duplicate assertion key")
    summary = {
        status: sum(record["status"] == status for record in assertions)
        for status in ("PASS", "FAIL", "UNVERIFIED", "NOT_APPLICABLE")
    }
    require(summary["FAIL"] == 0 and summary["UNVERIFIED"] == 1, "assertion status mismatch")

    result_files: dict[str, bytes] = {
        "input_provenance.json": json_bytes(input_provenance),
        "first_run_failure_evidence_summary.json": json_bytes(first_run),
        "source_change_inventory.json": json_bytes(change_inventory),
        "project_file_preservation.json": json_bytes(preservation),
        "global_runtime_asset_preservation.json": json_bytes(global_preservation),
        "expected_score_key_regression.json": json_bytes(key_regression),
        "typed_failure_static_control_flow.json": json_bytes(static_control_flow),
        "typed_failure_runtime_probe.json": json_bytes(typed_probe),
        "offline_csharp_compile_check.json": json_bytes(compile_evidence),
        "primary_verdict.json": json_bytes(primary),
        "ASSERTIONS.jsonl": b"".join(json_bytes(record, pretty=False) for record in assertions),
        "VALIDATION_REPORT.md": report_text(key_regression, typed_probe, preservation, compile_evidence, summary).encode("utf-8"),
        "analysis/s001_step9i_personal_runtime_opt_in_integration_v1_2.py": Path(__file__).read_bytes(),
        "analysis/compile_fixtures/NUnitUnityCompileStubs.cs": NUNIT_UNITY_STUB.encode("utf-8"),
        "analysis/compile_fixtures/TypedFailureProbe.cs": TYPED_FAILURE_PROBE.encode("utf-8"),
        "source_before/PersonalRuntimeOptInProfile.cs": production_before,
        "source_before/PersonalRuntimeOptInIntegrationV1Tests.cs": test_before,
        "source_after/PersonalRuntimeOptInProfile.cs": production_after,
        "source_after/PersonalRuntimeOptInIntegrationV1Tests.cs": test_after,
        "unified_diff/PersonalRuntimeOptInProfile.diff": production_diff,
        "unified_diff/PersonalRuntimeOptInIntegrationV1Tests.diff": test_diff,
        "input_subset/v1_1_provenance/output_file_manifest.json": v1_1_map["output_file_manifest.json"],
        "input_subset/v1_1_provenance/primary_verdict.json": v1_1_map["primary_verdict.json"],
        "input_subset/v1_1_provenance/packaging_transaction_failure_injection_regression.json": v1_1_map["packaging_transaction_failure_injection_regression.json"],
        "input_subset/v1_1_provenance/preserved_project_file_identity.json": v1_1_map["preserved_project_file_identity.json"],
    }
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

    zip_one = v1_1.deterministic_zip_bytes(result_files)
    zip_two = v1_1.deterministic_zip_bytes(result_files)
    require(zip_one == zip_two, "v1.2 deterministic ZIP memory regeneration mismatch")
    inspection = inspect_archive(zip_one, set(result_files))
    detached_bytes = (sha256_bytes(zip_one) + "  " + delivery.name + "\n").encode("ascii")

    fixed_hashes = {
        V1_1_REL.as_posix(): sha256_file(root / V1_1_REL),
        V1_1_REL.as_posix() + ".sha256": sha256_file(Path(str(root / V1_1_REL) + ".sha256")),
        V1_0_REL.as_posix(): sha256_file(root / V1_0_REL),
        V1_0_REL.as_posix() + ".sha256": sha256_file(Path(str(root / V1_0_REL) + ".sha256")),
    }
    project_after_hashes = {
        record["project_relative_path"]: sha256_file(root / PurePosixPath(record["project_relative_path"]))
        for record in preservation["records"]
    }

    def post_validator(final_result: Path, final_delivery: Path, final_detached: Path) -> None:
        names = {
            path.relative_to(final_result).as_posix()
            for path in final_result.rglob("*")
            if path.is_file()
        }
        require(names == set(result_files), "post-commit result membership mismatch")
        require(
            all((final_result / PurePosixPath(name)).read_bytes() == data for name, data in result_files.items()),
            "post-commit result byte mismatch",
        )
        written_zip = final_delivery.read_bytes()
        require(written_zip == zip_one, "post-commit Delivery byte mismatch")
        inspect_archive(written_zip, set(result_files))
        require(final_detached.read_bytes() == detached_bytes, "post-commit detached checksum mismatch")
        require(
            fixed_hashes
            == {
                V1_1_REL.as_posix(): sha256_file(root / V1_1_REL),
                V1_1_REL.as_posix() + ".sha256": sha256_file(Path(str(root / V1_1_REL) + ".sha256")),
                V1_0_REL.as_posix(): sha256_file(root / V1_0_REL),
                V1_0_REL.as_posix() + ".sha256": sha256_file(Path(str(root / V1_0_REL) + ".sha256")),
            },
            "fixed v1.0/v1.1 inputs changed",
        )
        require(
            project_after_hashes
            == {
                record["project_relative_path"]: sha256_file(root / PurePosixPath(record["project_relative_path"]))
                for record in preservation["records"]
            },
            "project files changed during packaging",
        )
        staged_now = subprocess.run(
            ["git", "diff", "--cached", "--name-only"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.splitlines()
        require(not staged_now, "Git staged files changed during packaging")

    v1_1.commit_packaging_transaction(
        result_files=result_files,
        delivery_bytes=zip_one,
        detached_bytes=detached_bytes,
        final_result=output,
        final_delivery=delivery,
        final_detached=detached,
        allowed_root=validation_root,
        post_rename_validator=post_validator,
    )
    return {
        "status": "PASS",
        "actual_primary_verdict": primary["actual_primary_verdict"],
        "changed_project_files": preservation["changed_project_file_count"],
        "unchanged_project_files": preservation["unchanged_byte_identical_count"],
        "key_regression": {
            "csv_rows": key_regression["csv_data_rows"],
            "unique_keys": key_regression["unique_keys"],
            "runtime_lookup_keys_present": key_regression["runtime_lookup_keys_present"],
        },
        "typed_failure_regression": {
            "missing_required_template": typed_probe["missing_required_template"]["actual_failure_code"],
            "additional_wisrt": typed_probe["additional_wisrt"]["actual_failure_code"],
        },
        "offline_compile": compile_evidence,
        "assertion_summary": summary,
        "unity_status": "STEP9I_UNITY_EDITMODE_RETEST_UNVERIFIED",
        "output_directory": str(output),
        "output_file_count": len(result_files),
        "delivery_zip": str(delivery),
        "delivery_size_bytes": len(zip_one),
        "delivery_sha256": sha256_bytes(zip_one),
        "detached_checksum": str(detached),
        "detached_size_bytes": len(detached_bytes),
        "detached_file_sha256": sha256_bytes(detached_bytes),
        "zip_inspection": inspection,
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
        or root / "Validation/WordsInSilence_step9i_personal_runtime_opt_in_integration_v1_2"
    ).resolve()
    delivery = (
        args.delivery
        or root / "Validation/WordsInSilence_step9i_personal_runtime_opt_in_integration_v1_2_delivery.zip"
    ).resolve()
    result = build(root, output, delivery)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
