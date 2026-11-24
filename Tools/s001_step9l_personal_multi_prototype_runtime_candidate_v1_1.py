#!/usr/bin/env python3
"""WordsInSilence Step 9L multi-prototype candidate evidence repair v1.1.

This tool does not install or regenerate the externally approved v1.0 C#,
profile, WISRT, DTW, or strict-LOO artifacts.  It preserves those bytes and
repairs only the packaging transaction and the missing comparison/protection
evidence required by the project context.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import tempfile
from typing import Any, Callable
import zipfile


SCHEMA = "words_in_silence.step9l.personal_multi_prototype_runtime_candidate.v1_1"
WORDS = ("GREETING", "LOVE", "THANK_YOU")
TOLERANCE = 1e-12

V1_NAME = "WordsInSilence_step9l_personal_multi_prototype_runtime_candidate_v1_0_delivery.zip"
V1_REL = Path("Validation") / V1_NAME
V1_SIZE = 719_243
V1_SHA256 = "806ebed55c34590b8c7ed1579ecfff8d72192f20d668b7ccfba5daa79b6f09cf"
V1_DETACHED_SIZE = 149
V1_DETACHED_FILE_SHA256 = "6292bcb9a5eb9d629d15d7e4bc9f9cd0332b5561a6a893d839d15229cdb70a24"
V1_ENTRY_COUNT = 82
V1_MANIFEST_TARGET_COUNT = 81
V1_ANALYSIS_PATH = "analysis/s001_step9l_personal_multi_prototype_runtime_candidate_v1_0.py"
V1_ANALYSIS_SIZE = 117_500
V1_ANALYSIS_SHA256 = "f0ea352d5a906f4365f98bab97c42ae2d477e4171a95032c3ff20e7bf1c6c79c"

STEP9H_NAME = "WordsInSilence_step9h_personalized_class_conditional_threshold_v1_0_delivery.zip"
STEP9H_REL = Path("Validation") / STEP9H_NAME
STEP9H_SIZE = 3_214_867
STEP9H_SHA256 = "4827e81e029e63e7939e7ed91cb33ac5f90dd6822fa13722da01f642aed2ddb9"
STEP9H_DETACHED_SIZE = 147
STEP9H_DETACHED_FILE_SHA256 = "d564d21cd5e6cf0dc1b3eda47cb88a80014710fd182d3889827b743961829c4b"
STEP9H_ENTRY_COUNT = 53
STEP9H_MANIFEST_TARGET_COUNT = 52

TRANSACTION_TOOL_REL = Path("Tools/s001_step9i_personal_runtime_opt_in_integration_v1_1.py")
TRANSACTION_TOOL_SIZE = 37_722
TRANSACTION_TOOL_SHA256 = "30471dc6725bdbdd0cc80c345b238485d1cb0d8d7a5933aac7477d536d714caf"

EXPECTED_SINGLE = {
    "GREETING": {
        "genuine_upper": 0.5172946711028763,
        "impostor_lower": 0.523833132325901,
        "separation_gap": 0.006538461223024705,
        "threshold": 0.5205639017143886,
    },
    "LOVE": {
        "genuine_upper": 0.36818101372824913,
        "impostor_lower": 0.4231019848491888,
        "separation_gap": 0.05492097112093969,
        "threshold": 0.39564149928871895,
    },
    "THANK_YOU": {
        "genuine_upper": 0.33790469372483245,
        "impostor_lower": 0.42459261430230166,
        "separation_gap": 0.08668792057746921,
        "threshold": 0.38124865401356706,
    },
}

EXPECTED_MULTI = {
    "GREETING": {
        "genuine_upper": 0.49877876241825186,
        "impostor_lower": 0.523833132325901,
        "separation_gap": 0.025054369907649143,
        "threshold": 0.5113059473720765,
    },
    "LOVE": {
        "genuine_upper": 0.36625719200585466,
        "impostor_lower": 0.4162948549149993,
        "separation_gap": 0.05003766290914463,
        "threshold": 0.391276023460427,
    },
    "THANK_YOU": {
        "genuine_upper": 0.33790469372483245,
        "impostor_lower": 0.4162948549149993,
        "separation_gap": 0.07839016119016684,
        "threshold": 0.37709977431991587,
    },
}

ORIGINAL_RUNTIME_SOURCE = {
    "relative_path": "Assets/_Project/Scripts/Recognition/Runtime/PersonalRuntimeOptInProfile.cs",
    "size_bytes": 54_894,
    "sha256": "3e4579c153ad23ee85c00b01967956f0777a1092384e5e8ade8b57b9fa5798bc",
}

APPROVED_CSHARP4 = (
    ("Assets/_Project/Tests/EditMode/PersonalRuntimeOptInIntegrationV1Tests.cs", 26_880, "0db5cd7afbacadf54fb2cdb53af0211dfb3ef48758d41a0d9de9df05ca7cc959"),
    ("Assets/_Project/Scripts/Recognition/LiveInput/PersonalLiveRecognitionController.cs", 31_702, "1fdea854b2368676a544a975acc8021533b17500dfc794519b04df130645db5b"),
    ("Assets/_Project/Scripts/Recognition/LiveInput/PersonalLiveRecognitionHarness.cs", 10_105, "4cb82dddbb682f465ef0905cae2a34418b7a64f3c66a3f19707d6dcd6a317097"),
    ("Assets/_Project/Tests/EditMode/PersonalLiveRecognitionIntegrationV1Tests.cs", 36_212, "6b3589de3d778d66117cdbcfae29e14a416d49332a8bfc3a429d6412a6821be3"),
)

PERSONAL_PROFILE10 = (
    ("Assets/StreamingAssets/RecognitionCoreV1/PersonalProfiles/step9h_current_user_v1_0/personal_class_threshold_config.json", 4_811, "e84edd1f598c2d94f8412767f3125031e0be330886d3a315ed429798fe49726e"),
    ("Assets/StreamingAssets/RecognitionCoreV1/PersonalProfiles/step9h_current_user_v1_0/personal_class_threshold_config.json.meta", 155, "7eb331af807803ee3e7b3330fae0cd164847e531ae811e6a02b1286f5090aea9"),
    ("Assets/StreamingAssets/RecognitionCoreV1/PersonalProfiles/step9h_current_user_v1_0/personal_runtime_profile_manifest.json", 3_768, "2f9b4589c0a6130913cca6a5d9e3260aeaa6e9d4a5cba76aaafa61692fe46eb9"),
    ("Assets/StreamingAssets/RecognitionCoreV1/PersonalProfiles/step9h_current_user_v1_0/personal_runtime_profile_manifest.json.meta", 155, "652d35b4fb4340c13f09ac9908e1ac5b8a53811e2a301af1bf8272fe351ed681"),
    ("Assets/StreamingAssets/RecognitionCoreV1/PersonalProfiles/step9h_current_user_v1_0/personal_template_GREETING.wisrt", 53_564, "d8d5713334b22d9427b9b589bccf820cdbb4849357bae3dce1c368eaf0049681"),
    ("Assets/StreamingAssets/RecognitionCoreV1/PersonalProfiles/step9h_current_user_v1_0/personal_template_GREETING.wisrt.meta", 155, "c00df8a98e1fc7058cd4611379d6da75ccfe4d49f1b217afc1efe1174fb32485"),
    ("Assets/StreamingAssets/RecognitionCoreV1/PersonalProfiles/step9h_current_user_v1_0/personal_template_LOVE.wisrt", 56_262, "013bf5515eb8af33f300400893da64c853668b2fa0442e19203da62b2777ad4b"),
    ("Assets/StreamingAssets/RecognitionCoreV1/PersonalProfiles/step9h_current_user_v1_0/personal_template_LOVE.wisrt.meta", 155, "c1593384d12f4f939af44f8bfe5ee2cfcfb68d7a2b363019ee6773eec4dc87e8"),
    ("Assets/StreamingAssets/RecognitionCoreV1/PersonalProfiles/step9h_current_user_v1_0/personal_template_THANK_YOU.wisrt", 49_319, "9bb9a7681d55a292a21d8b7f458b4b150c10555bc0ec163da692a780ed5280e5"),
    ("Assets/StreamingAssets/RecognitionCoreV1/PersonalProfiles/step9h_current_user_v1_0/personal_template_THANK_YOU.wisrt.meta", 155, "d87903fd222e28228e278daef98231b90cd829151b45801d534d6a7711925c57"),
)

GLOBAL_RUNTIME11 = (
    ("Assets/StreamingAssets/RecognitionCoreV1/Templates/template_GREETING__FAST.wisrt", 35_012, "a9111bd2733cb8a2d2c5984309607979cee49c2516346668228e4fb51cf88277"),
    ("Assets/StreamingAssets/RecognitionCoreV1/Templates/template_GREETING__NORMAL.wisrt", 50_842, "8a2859403e0db736d213f59c82bd9d9a3f7a44dc748b6c65dca5869bc4cfa4b9"),
    ("Assets/StreamingAssets/RecognitionCoreV1/Templates/template_GREETING__SLOW.wisrt", 60_102, "0646436e35e20d13e3c00fd0aef85c7a77829d3729eb7a08e2c9ffa3dc989680"),
    ("Assets/StreamingAssets/RecognitionCoreV1/Templates/template_LOVE__FAST.wisrt", 19_950, "1ab0ccee87c749cb961f6d8a8135062f14c8affc8ec15b40472240e3e936967c"),
    ("Assets/StreamingAssets/RecognitionCoreV1/Templates/template_LOVE__NORMAL.wisrt", 29_218, "74b42959641fd1ce182f4f72427c02307f14ddc564e06e6f24044694bca5a3c3"),
    ("Assets/StreamingAssets/RecognitionCoreV1/Templates/template_LOVE__SLOW.wisrt", 34_232, "509cfb87126d18285745085810fc4a5afcc82886a8869d02f375166f71819a28"),
    ("Assets/StreamingAssets/RecognitionCoreV1/Templates/template_THANK_YOU__FAST.wisrt", 29_610, "e53455dff4448865b0df7d9d859888213a31e33738512b7ed7af0d16230e3e6f"),
    ("Assets/StreamingAssets/RecognitionCoreV1/Templates/template_THANK_YOU__NORMAL.wisrt", 41_966, "7da43c9f770ac79e1a67eb487469cef35467b72f4ab3abbb41faa26f0396c82e"),
    ("Assets/StreamingAssets/RecognitionCoreV1/Templates/template_THANK_YOU__SLOW.wisrt", 46_208, "b83309ac244a8f92fd83ac143da9e2307768125fe3379fd0ff9345bf6b5bb26f"),
    ("Assets/StreamingAssets/RecognitionCoreV1/RuntimeGate/frozen_threshold_config.json", 4_590, "2697bb1d0c301850d2abda41351519c66d0eb8e27841263469175eb3fa7e99e3"),
    ("Assets/StreamingAssets/RecognitionCoreV1/RuntimeGate/runtime_recognition_assets.json", 4_202, "261aa173a9643c21677d66c16ef3448c2808b3db2ffa5f9e48c671af6e9bd5e7"),
)


class Step9LRepairError(RuntimeError):
    """A fixed-input, calculation, preservation, or packaging failure."""


class PostValidationInjectedFailure(Step9LRepairError):
    """Expected failure after the synthetic post-rename validation completes."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Step9LRepairError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def safe_zip_name(name: str) -> bool:
    pure = PurePosixPath(name)
    return (
        bool(name)
        and not name.startswith(("/", "\\"))
        and "\\" not in name
        and ":" not in pure.parts[0]
        and all(part not in ("", ".", "..") for part in pure.parts)
    )


def read_zip_map(data: bytes) -> tuple[dict[str, bytes], dict[str, Any]]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        require(len(names) == len(set(names)), "ZIP duplicate entry")
        require(archive.testzip() is None, "ZIP CRC failure")
        require(all(safe_zip_name(name) for name in names), "ZIP unsafe path")
        require(not any(info.flag_bits & 1 for info in infos), "ZIP encrypted entry")
        require(not any(stat.S_ISLNK((info.external_attr >> 16) & 0xFFFF) for info in infos), "ZIP symlink entry")
        return ({name: archive.read(name) for name in names}, {
            "entry_count": len(names), "crc": "PASS", "duplicate_count": 0,
            "unsafe_path_count": 0, "symlink_count": 0, "encrypted_entry_count": 0,
        })


def verify_manifest(file_map: dict[str, bytes], manifest_name: str, expected_targets: int) -> dict[str, Any]:
    require(manifest_name in file_map, f"manifest missing: {manifest_name}")
    manifest = json.loads(file_map[manifest_name])
    records = manifest["outputs"]
    require(len(records) == expected_targets, "manifest target count mismatch")
    record_names = [record["relative_path"] for record in records]
    require(len(record_names) == len(set(record_names)), "manifest duplicate path")
    require(set(file_map) == set(record_names) | {manifest_name}, "manifest membership mismatch")
    for record in records:
        payload = file_map[record["relative_path"]]
        require(len(payload) == record["size_bytes"], "manifest size mismatch: " + record["relative_path"])
        require(sha256_bytes(payload) == record["sha256"], "manifest hash mismatch: " + record["relative_path"])
    return {"manifest_target_count": len(records), "membership": "PASS", "size_sha256": "PASS"}


def verify_detached(data: bytes, zip_data: bytes, delivery_name: str) -> None:
    expected = (sha256_bytes(zip_data) + "  " + delivery_name + "\n").encode("ascii")
    require(data == expected, "detached checksum syntax, filename, or digest mismatch")


def load_transaction_module(root: Path) -> tuple[Any, dict[str, Any]]:
    path = root / TRANSACTION_TOOL_REL
    require(path.is_file(), "approved Step 9I v1.1 transaction source missing")
    payload = path.read_bytes()
    require(len(payload) == TRANSACTION_TOOL_SIZE, "approved transaction source size mismatch")
    require(sha256_bytes(payload) == TRANSACTION_TOOL_SHA256, "approved transaction source hash mismatch")
    spec = importlib.util.spec_from_file_location("step9i_transaction_v1_1", path)
    require(spec is not None and spec.loader is not None, "cannot load approved transaction source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(callable(module.commit_packaging_transaction), "approved transaction function missing")
    return module, {
        "relative_path": TRANSACTION_TOOL_REL.as_posix(),
        "size_bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "function": "commit_packaging_transaction",
    }


def verify_fixed_delivery(
    root: Path,
    relative: Path,
    size: int,
    digest: str,
    detached_size: int,
    detached_digest: str,
    entry_count: int,
    manifest_targets: int,
) -> tuple[dict[str, bytes], dict[str, Any], bytes, bytes]:
    path = root / relative
    detached = Path(str(path) + ".sha256")
    require(path.is_file(), f"fixed Delivery missing: {path}")
    require(detached.is_file(), f"fixed detached checksum missing: {detached}")
    zip_data = path.read_bytes()
    detached_data = detached.read_bytes()
    require(len(zip_data) == size, f"fixed Delivery size mismatch: {path.name}")
    require(sha256_bytes(zip_data) == digest, f"fixed Delivery hash mismatch: {path.name}")
    require(len(detached_data) == detached_size, f"fixed detached size mismatch: {detached.name}")
    require(sha256_bytes(detached_data) == detached_digest, f"fixed detached file hash mismatch: {detached.name}")
    verify_detached(detached_data, zip_data, path.name)
    file_map, safety = read_zip_map(zip_data)
    require(len(file_map) == entry_count, f"fixed Delivery entry count mismatch: {path.name}")
    manifest = verify_manifest(file_map, "output_file_manifest.json", manifest_targets)
    return file_map, {
        "relative_path": relative.as_posix(), "size_bytes": len(zip_data), "sha256": sha256_bytes(zip_data),
        "detached_relative_path": (relative.as_posix() + ".sha256"),
        "detached_size_bytes": len(detached_data), "detached_file_sha256": sha256_bytes(detached_data),
        "archive_safety": safety, "manifest": manifest,
    }, zip_data, detached_data


def fixed_records(items: tuple[tuple[str, int, str], ...]) -> list[dict[str, Any]]:
    return [{"relative_path": path, "size_bytes": size, "sha256": digest} for path, size, digest in items]


def snapshot_group(root: Path, records: list[dict[str, Any]], group: str) -> dict[str, Any]:
    require(len(records) > 0, f"{group} baseline must not be empty")
    require(len({r['relative_path'] for r in records}) == len(records), f"{group} baseline duplicate path")
    actual: list[dict[str, Any]] = []
    mismatch: list[dict[str, Any]] = []
    for expected in records:
        path = root / expected["relative_path"]
        if not path.is_file():
            actual_record = {"relative_path": expected["relative_path"], "exists": False, "size_bytes": None, "sha256": None}
        else:
            payload = path.read_bytes()
            actual_record = {"relative_path": expected["relative_path"], "exists": True, "size_bytes": len(payload), "sha256": sha256_bytes(payload)}
        actual.append(actual_record)
        if not actual_record["exists"] or actual_record["size_bytes"] != expected["size_bytes"] or actual_record["sha256"] != expected["sha256"]:
            mismatch.append({"expected": expected, "actual": actual_record})
    require(not mismatch, f"{group} baseline mismatch")
    return {
        "group": group, "required_count": len(records), "actual_count": len(actual),
        "mismatch_count": len(mismatch), "byte_identical_count": len(records) - len(mismatch),
        "baseline": records, "actual": actual,
    }


def require_exact_group_membership(root: Path, group: str, records: list[dict[str, Any]]) -> None:
    expected = {record["relative_path"] for record in records}
    if group == "existing_personal_profile":
        directory = root / "Assets/StreamingAssets/RecognitionCoreV1/PersonalProfiles/step9h_current_user_v1_0"
        actual = {path.relative_to(root).as_posix() for path in directory.iterdir() if path.is_file()}
    elif group == "global_runtime_assets":
        templates = root / "Assets/StreamingAssets/RecognitionCoreV1/Templates"
        runtime_gate = root / "Assets/StreamingAssets/RecognitionCoreV1/RuntimeGate"
        actual = (
            {path.relative_to(root).as_posix() for path in templates.glob("*.wisrt") if path.is_file()}
            | {path.relative_to(root).as_posix() for path in runtime_gate.glob("*.json") if path.is_file()}
        )
    else:
        raise Step9LRepairError("unsupported exact-membership group: " + group)
    require(actual == expected, f"{group} exact project membership mismatch")


def compact_snapshot(groups: list[dict[str, Any]], original_source: dict[str, Any]) -> dict[str, tuple[int, str]]:
    result: dict[str, tuple[int, str]] = {}
    for group in groups:
        for record in group["actual"]:
            result[record["relative_path"]] = (record["size_bytes"], record["sha256"])
    result[original_source["relative_path"]] = (original_source["size_bytes"], original_source["sha256"])
    return result


def verify_original_source(root: Path) -> dict[str, Any]:
    path = root / ORIGINAL_RUNTIME_SOURCE["relative_path"]
    require(path.is_file(), "original project runtime source missing")
    payload = path.read_bytes()
    actual = {"relative_path": ORIGINAL_RUNTIME_SOURCE["relative_path"], "size_bytes": len(payload), "sha256": sha256_bytes(payload)}
    require(actual == ORIGINAL_RUNTIME_SOURCE, "original project runtime source identity mismatch")
    return actual


def git_staged(root: Path) -> dict[str, Any]:
    probe = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=root, capture_output=True, text=True)
    if probe.returncode != 0 or probe.stdout.strip() != "true":
        return {"repository": False, "staged_count": 0, "staged_paths": [], "status": "NOT_APPLICABLE"}
    result = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=root, capture_output=True, text=True, check=True)
    paths = [line for line in result.stdout.splitlines() if line]
    require(not paths, f"Git staged files present: {paths}")
    return {"repository": True, "staged_count": 0, "staged_paths": [], "status": "PASS"}


def parse_csv(data: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"))))


def build_distance_inputs(v1_map: dict[str, bytes]) -> tuple[list[dict[str, Any]], dict[tuple[str, str], float]]:
    identity = json.loads(v1_map["capture_identity.json"])
    captures = identity["captures"]
    require(len(captures) == 9, "capture identity count mismatch")
    require(Counter(row["word"] for row in captures) == Counter({word: 3 for word in WORDS}), "capture count per word mismatch")
    require(len({row["capture_id"] for row in captures}) == 9, "capture ID duplicate")
    require(len({row["capture_key"] for row in captures}) == 9, "capture key duplicate")
    pairwise = parse_csv(v1_map["input_subset/provenance/step9h_pairwise_dtw.csv"])
    require(len(pairwise) == 81, "pairwise DTW count mismatch")
    distances: dict[tuple[str, str], float] = {}
    for row in pairwise:
        key = (row["left_capture_key"], row["right_capture_key"])
        require(key not in distances, "pairwise DTW duplicate key")
        value = float(row["distance"])
        require(abs(value - float(row["approved_runtime_contract_distance"])) <= TOLERANCE, "pairwise runtime parity mismatch")
        distances[key] = value
    keys = {row["capture_key"] for row in captures}
    require(set(distances) == {(left, right) for left in keys for right in keys}, "pairwise DTW membership mismatch")
    for left in keys:
        require(abs(distances[(left, left)]) <= TOLERANCE, "pairwise diagonal mismatch")
        for right in keys:
            require(abs(distances[(left, right)] - distances[(right, left)]) <= TOLERANCE, "pairwise symmetry mismatch")
    return captures, distances


def select_medoid(candidates: list[dict[str, Any]], distances: dict[tuple[str, str], float]) -> dict[str, Any]:
    require(len(candidates) >= 2, "medoid needs at least two candidates")
    scored = []
    for candidate in candidates:
        others = [other for other in candidates if other["capture_id"] != candidate["capture_id"]]
        mean = sum(distances[(candidate["capture_key"], other["capture_key"])] for other in others) / len(others)
        scored.append((mean, candidate["capture_id"], candidate))
    return min(scored, key=lambda item: (item[0], item[1]))[2]


def calculate_route(
    route: str,
    captures: list[dict[str, Any]],
    distances: dict[tuple[str, str], float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    by_query: dict[str, dict[str, dict[str, Any]]] = {}
    for fold_index, query in enumerate(captures):
        training = [row for row in captures if row["capture_id"] != query["capture_id"]]
        by_query[query["capture_id"]] = {}
        for target in WORDS:
            candidates = [row for row in training if row["word"] == target]
            if route == "single_medoid":
                winner = select_medoid(candidates, distances)
                role = "medoid"
            elif route == "multi_prototype":
                winner = min(candidates, key=lambda row: (distances[(query["capture_key"], row["capture_key"])], row["capture_id"]))
                role = "nearest_prototype"
            else:
                raise Step9LRepairError("unknown route")
            score = distances[(query["capture_key"], winner["capture_key"])]
            row = {
                "comparison_cell_key": route + "||" + query["capture_id"] + "||" + target,
                "route": route,
                "fold_index": fold_index,
                "query_capture_id": query["capture_id"],
                "query_capture_key": query["capture_key"],
                "expected_word": query["word"],
                "target_word": target,
                "training_candidate_count": len(candidates),
                "query_excluded_from_training": True,
                "winner_role": role,
                "winning_capture_id": winner["capture_id"],
                "winning_capture_key": winner["capture_key"],
                "score": score,
                "score_kind": "genuine" if query["word"] == target else "impostor",
            }
            cells.append(row)
            by_query[query["capture_id"]][target] = row

    per_word: dict[str, Any] = {}
    for word in WORDS:
        genuine = [row["score"] for row in cells if row["target_word"] == word and row["score_kind"] == "genuine"]
        impostor = [row["score"] for row in cells if row["target_word"] == word and row["score_kind"] == "impostor"]
        require(len(genuine) == 3 and len(impostor) == 6, "route score kind count mismatch")
        upper = max(genuine)
        lower = min(impostor)
        per_word[word] = {
            "genuine_scores": genuine, "impostor_scores": impostor,
            "genuine_score_count": len(genuine), "impostor_score_count": len(impostor),
            "genuine_upper": upper, "impostor_lower": lower,
            "separation_gap": lower - upper, "threshold": (upper + lower) / 2,
            "strict_separation": upper < lower,
        }

    queries: list[dict[str, Any]] = []
    for fold_index, query in enumerate(captures):
        scores = by_query[query["capture_id"]]
        predicted = min(WORDS, key=lambda word: (scores[word]["score"], word))
        expected_score = scores[query["word"]]["score"]
        best_impostor = min(scores[word]["score"] for word in WORDS if word != query["word"])
        expected_accept = expected_score <= per_word[query["word"]]["threshold"]
        predicted_accept = scores[predicted]["score"] <= per_word[predicted]["threshold"]
        query_row = {
            "comparison_query_key": route + "||" + query["capture_id"],
            "route": route, "fold_index": fold_index,
            "query_capture_id": query["capture_id"], "query_capture_key": query["capture_key"],
            "expected_word": query["word"], "predicted_word": predicted,
            "prediction_correct": predicted == query["word"],
            "expected_class_score": expected_score, "best_impostor_score": best_impostor,
            "classification_margin": best_impostor - expected_score,
            "expected_class_genuine_upper": per_word[query["word"]]["genuine_upper"],
            "expected_class_impostor_lower": per_word[query["word"]]["impostor_lower"],
            "expected_class_separation_gap": per_word[query["word"]]["separation_gap"],
            "expected_class_threshold": per_word[query["word"]]["threshold"],
            "expected_class_accept": expected_accept,
            "predicted_class_threshold": per_word[predicted]["threshold"],
            "predicted_class_accept": predicted_accept,
            "quest_success": predicted == query["word"] and expected_accept and predicted_accept,
            "winning_greeting_capture_id": scores["GREETING"]["winning_capture_id"],
            "winning_love_capture_id": scores["LOVE"]["winning_capture_id"],
            "winning_thank_you_capture_id": scores["THANK_YOU"]["winning_capture_id"],
        }
        queries.append(query_row)
        for target in WORDS:
            scores[target].update({
                "predicted_word": predicted,
                "prediction_correct": predicted == query["word"],
                "target_genuine_upper": per_word[target]["genuine_upper"],
                "target_impostor_lower": per_word[target]["impostor_lower"],
                "target_separation_gap": per_word[target]["separation_gap"],
                "target_threshold": per_word[target]["threshold"],
                "target_class_accept": scores[target]["score"] <= per_word[target]["threshold"],
                "expected_class_margin": best_impostor - expected_score,
            })
    summary = {
        "route": route, "query_count": len(queries), "score_cell_count": len(cells),
        "query_leak_free_fold_count": sum(all(cell["query_capture_id"] != cell["winning_capture_id"] for cell in cells if cell["fold_index"] == row["fold_index"]) for row in queries),
        "prediction_correct_count": sum(row["prediction_correct"] for row in queries),
        "quest_success_count": sum(row["quest_success"] for row in queries),
        "per_word": per_word,
    }
    return cells, queries, summary


def close(left: float, right: float) -> bool:
    return abs(left - right) <= TOLERANCE


def verify_route_against_saved(
    route: str,
    cells: list[dict[str, Any]],
    summary: dict[str, Any],
    saved_cells_data: bytes,
    saved_threshold_data: bytes,
) -> dict[str, Any]:
    saved_rows = parse_csv(saved_cells_data)
    require(len(saved_rows) == 27, f"{route} saved score cell count mismatch")
    saved_by_key = {(row.get("held_out_capture_id") or row.get("query_capture_id"), row["target_word"]): row for row in saved_rows}
    require(len(saved_by_key) == 27, f"{route} saved score key duplicate")
    mismatches = []
    max_error = 0.0
    for row in cells:
        saved = saved_by_key[(row["query_capture_id"], row["target_word"])]
        error = abs(row["score"] - float(saved["score"]))
        max_error = max(max_error, error)
        saved_winner = saved.get("target_medoid_capture_id") or saved.get("winning_prototype_capture_id")
        if error > TOLERANCE or saved_winner != row["winning_capture_id"]:
            mismatches.append({"cell": row["comparison_cell_key"], "score_error": error, "saved_winner": saved_winner, "calculated_winner": row["winning_capture_id"]})
    saved_thresholds = json.loads(saved_threshold_data)["per_word"]
    threshold_mismatches = []
    for word in WORDS:
        saved = saved_thresholds[word]
        saved_threshold = saved.get("threshold", saved.get("midpoint_threshold"))
        for field, saved_value in (
            ("genuine_upper", saved["genuine_upper"]),
            ("impostor_lower", saved["impostor_lower"]),
            ("separation_gap", saved["separation_gap"]),
            ("threshold", saved_threshold),
        ):
            if not close(summary["per_word"][word][field], float(saved_value)):
                threshold_mismatches.append({"word": word, "field": field, "saved": saved_value, "calculated": summary["per_word"][word][field]})
    require(not mismatches and not threshold_mismatches, f"{route} independent recalculation parity mismatch")
    return {
        "route": route, "saved_score_cell_count": len(saved_rows), "calculated_score_cell_count": len(cells),
        "score_mismatch_count": len(mismatches), "winning_identity_mismatch_count": 0,
        "threshold_mismatch_count": len(threshold_mismatches), "max_absolute_score_error": max_error,
        "tolerance": TOLERANCE, "status": "PASS",
    }


def verify_expected(summary: dict[str, Any], expected: dict[str, dict[str, float]], label: str) -> None:
    for word in WORDS:
        for field in ("genuine_upper", "impostor_lower", "separation_gap", "threshold"):
            require(close(summary["per_word"][word][field], expected[word][field]), f"{label} expected {word} {field} mismatch")


def comparison_evidence(v1_map: dict[str, bytes], step9h_map: dict[str, bytes]) -> tuple[dict[str, Any], bytes, bytes]:
    captures, distances = build_distance_inputs(v1_map)
    single_cells, single_queries, single_summary = calculate_route("single_medoid", captures, distances)
    multi_cells, multi_queries, multi_summary = calculate_route("multi_prototype", captures, distances)
    single_parity = verify_route_against_saved(
        "single_medoid", single_cells, single_summary,
        step9h_map["class_conditional_fold_scores.csv"], step9h_map["class_conditional_threshold_derivation.json"],
    )
    multi_parity = verify_route_against_saved(
        "multi_prototype", multi_cells, multi_summary,
        v1_map["strict_loo_scores.csv"], v1_map["strict_loo_threshold_derivation.json"],
    )
    verify_expected(single_summary, EXPECTED_SINGLE, "single-medoid")
    verify_expected(multi_summary, EXPECTED_MULTI, "multi-prototype")
    improvement = multi_summary["per_word"]["GREETING"]["separation_gap"] - single_summary["per_word"]["GREETING"]["separation_gap"]
    conditions = {
        "single_query_leak_free_9_of_9": single_summary["query_leak_free_fold_count"] == 9,
        "multi_query_leak_free_9_of_9": multi_summary["query_leak_free_fold_count"] == 9,
        "single_prediction_9_of_9": single_summary["prediction_correct_count"] == 9,
        "multi_prediction_9_of_9": multi_summary["prediction_correct_count"] == 9,
        "multi_all_three_gaps_positive": all(multi_summary["per_word"][word]["separation_gap"] > 0 for word in WORDS),
        "greeting_gap_improved": multi_summary["per_word"]["GREETING"]["separation_gap"] > single_summary["per_word"]["GREETING"]["separation_gap"],
        "single_saved_parity": single_parity["status"] == "PASS",
        "multi_saved_parity": multi_parity["status"] == "PASS",
    }
    require(all(conditions.values()), "baseline comparison condition failed")
    evidence = {
        "schema_version": SCHEMA + ".single_medoid_multi_prototype_comparison",
        "calculation_source": "independent reconstruction from fixed capture identity and 81 ordered approved DTW cells",
        "pass_not_hardcoded": True,
        "query_count": 9, "score_cell_count_per_route": 27, "combined_score_cell_count": 54,
        "single_medoid": single_summary, "multi_prototype": multi_summary,
        "single_saved_parity": single_parity, "multi_saved_parity": multi_parity,
        "greeting_gap": {
            "single_medoid": single_summary["per_word"]["GREETING"]["separation_gap"],
            "multi_prototype": multi_summary["per_word"]["GREETING"]["separation_gap"],
            "improvement": improvement,
        },
        "conditions": conditions,
        "status": "PASS" if all(conditions.values()) else "FAIL",
    }
    cells = single_cells + multi_cells
    queries = single_queries + multi_queries
    cell_fields = [
        "comparison_cell_key", "route", "fold_index", "query_capture_id", "query_capture_key", "expected_word",
        "target_word", "training_candidate_count", "query_excluded_from_training", "winner_role", "winning_capture_id",
        "winning_capture_key", "score", "score_kind", "predicted_word", "prediction_correct", "target_genuine_upper",
        "target_impostor_lower", "target_separation_gap", "target_threshold", "target_class_accept", "expected_class_margin",
    ]
    query_fields = [
        "comparison_query_key", "route", "fold_index", "query_capture_id", "query_capture_key", "expected_word",
        "predicted_word", "prediction_correct", "expected_class_score", "best_impostor_score", "classification_margin",
        "expected_class_genuine_upper", "expected_class_impostor_lower", "expected_class_separation_gap",
        "expected_class_threshold", "expected_class_accept", "predicted_class_threshold", "predicted_class_accept",
        "quest_success", "winning_greeting_capture_id", "winning_love_capture_id", "winning_thank_you_capture_id",
    ]
    require(len({row["comparison_cell_key"] for row in cells}) == 54, "comparison cell duplicate")
    require(len({row["comparison_query_key"] for row in queries}) == 18, "comparison query duplicate")
    return evidence, csv_bytes(cells, cell_fields), csv_bytes(queries, query_fields)


def v1_preservation(v1_map: dict[str, bytes]) -> tuple[dict[str, bytes], dict[str, Any]]:
    excluded = {
        "ASSERTIONS.jsonl", "VALIDATION_REPORT.md", "output_file_manifest.json", "primary_verdict.json",
        "protected_files_before_after.json", "packaging_transaction_failure_injection_regression.json",
        "never_overwrite_regression.json", V1_ANALYSIS_PATH, "input_provenance.json",
    }
    preserved = {name: payload for name, payload in v1_map.items() if name not in excluded}
    records = [{"relative_path": name, "size_bytes": len(payload), "sha256": sha256_bytes(payload)} for name, payload in sorted(preserved.items())]
    explicit = {
        "candidate_wisrt_count": sum(name.startswith("project_files/candidate_profile/") and name.endswith(".wisrt") for name in preserved),
        "ordered_dtw_cell_count": len(parse_csv(preserved["input_subset/provenance/step9h_pairwise_dtw.csv"])),
        "strict_loo_score_cell_count": len(parse_csv(preserved["strict_loo_scores.csv"])),
        "candidate_runtime_source_count": sum(name in preserved for name in (
            "project_files/source_after/PersonalRuntimeOptInProfile.cs",
            "project_files/new_runtime_source/PersonalMultiPrototypeRuntimeProfile.cs",
        )),
        "candidate_test_source_count": int("project_files/new_test_source/PersonalMultiPrototypeRuntimeIntegrationV1Tests.cs" in preserved),
    }
    require(explicit == {"candidate_wisrt_count": 9, "ordered_dtw_cell_count": 81, "strict_loo_score_cell_count": 27, "candidate_runtime_source_count": 2, "candidate_test_source_count": 1}, "approved v1.0 artifact membership mismatch")
    return preserved, {
        "schema_version": SCHEMA + ".v1_0_approved_artifact_preservation",
        "preserved_file_count": len(records), "mismatch_count": 0, "byte_identical_count": len(records),
        "explicit_approved_scope": explicit, "records": records, "status": "PASS",
    }


def tree_identity(path: Path) -> dict[str, Any]:
    records = []
    for file in (item for item in path.rglob("*") if item.is_file()):
        relative = file.relative_to(path).as_posix()
        payload = file.read_bytes()
        records.append({"relative_path": relative, "size_bytes": len(payload), "sha256": sha256_bytes(payload)})
    records.sort(key=lambda record: record["relative_path"])
    return {"file_count": len(records), "records_sha256": sha256_bytes(json_bytes(records)), "records": records}


def validate_committed_outputs(
    transaction: Any,
    result_files: dict[str, bytes],
    zip_data: bytes,
    detached_data: bytes,
    result: Path,
    delivery: Path,
    detached: Path,
) -> None:
    require(result.is_dir() and delivery.is_file() and detached.is_file(), "committed output missing")
    tree = tree_identity(result)
    expected_records = [{"relative_path": name, "size_bytes": len(payload), "sha256": sha256_bytes(payload)} for name, payload in sorted(result_files.items())]
    require(tree["records"] == expected_records, "committed result tree mismatch")
    require(delivery.read_bytes() == zip_data, "committed Delivery bytes mismatch")
    require(detached.read_bytes() == detached_data, "committed detached bytes mismatch")
    transaction.verify_strict_detached(detached_data, zip_data, delivery.name)
    transaction.inspect_zip(zip_data, set(result_files))
    if "output_file_manifest.json" in result_files:
        verify_manifest(result_files, "output_file_manifest.json", len(result_files) - 1)


def transaction_failure_regression(
    root: Path,
    validation_root: Path,
    transaction: Any,
    project_snapshot: dict[str, tuple[int, str]],
) -> dict[str, Any]:
    points = list(transaction.INJECTION_POINTS) + ["POST_VALIDATION_FAILURE"]
    rows = []
    for index, point in enumerate(points):
        with tempfile.TemporaryDirectory(prefix="step9l_v1_1_injection_", dir=validation_root) as temporary:
            sandbox = Path(temporary)
            sentinel = sandbox / "pre_existing_sentinel.bin"
            sentinel.write_bytes(("sentinel:" + point).encode("ascii"))
            sentinel_hash = sha256_file(sentinel)
            finals = (sandbox / "final_result", sandbox / "final_delivery.zip", sandbox / "final_delivery.zip.sha256")
            token = "step9l_injection_" + str(index)
            staged = (
                sandbox / (".step9i_v1_1_result_" + token),
                sandbox / (".step9i_v1_1_delivery_" + token + ".tmp"),
                sandbox / (".step9i_v1_1_detached_" + token + ".tmp"),
            )
            expected = False
            validation_completed = False

            def validator(result: Path, delivery: Path, detached: Path) -> None:
                nonlocal validation_completed
                require(result.is_dir() and delivery.is_file() and detached.is_file(), "synthetic final missing")
                require((result / "marker.json").read_bytes() == b"{}\n", "synthetic result mismatch")
                validation_completed = True
                if point == "POST_VALIDATION_FAILURE":
                    raise PostValidationInjectedFailure(point)

            try:
                transaction.commit_packaging_transaction(
                    result_files={"marker.json": b"{}\n"}, delivery_bytes=b"synthetic-delivery",
                    detached_bytes=b"synthetic-detached", final_result=finals[0], final_delivery=finals[1],
                    final_detached=finals[2], allowed_root=sandbox,
                    injection_point=None if point == "POST_VALIDATION_FAILURE" else point,
                    post_rename_validator=validator, transaction_token=token,
                )
            except (transaction.InjectedPackagingFailure, PostValidationInjectedFailure) as error:
                expected = str(error) == point
            current_snapshot = {}
            for relative in project_snapshot:
                path = root / relative
                current_snapshot[relative] = (path.stat().st_size, sha256_file(path))
            final_count = sum(path.exists() or path.is_symlink() for path in finals)
            temp_count = sum(path.exists() or path.is_symlink() for path in staged)
            sentinel_changed = not sentinel.is_file() or sha256_file(sentinel) != sentinel_hash
            changed = sorted(path for path in project_snapshot if project_snapshot[path] != current_snapshot[path])
            passed = expected and final_count == 0 and temp_count == 0 and not sentinel_changed and not changed
            rows.append({
                "injection_point": point, "same_production_transaction_function": "commit_packaging_transaction",
                "expected_exception_observed": expected, "post_validation_completed": validation_completed,
                "remaining_final_outputs": final_count, "remaining_temporary_outputs": temp_count,
                "pre_existing_sentinel_changed": sentinel_changed,
                "original_project_source_restored_or_unchanged": ORIGINAL_RUNTIME_SOURCE["relative_path"] not in changed,
                "project_files_changed": len(changed), "changed_project_paths": changed,
                "verdict": "PASS" if passed else "FAIL",
            })
    passed = sum(row["verdict"] == "PASS" for row in rows)
    require(passed == len(rows), "packaging transaction failure injection regression failed")
    return {
        "schema_version": SCHEMA + ".packaging_transaction_failure_injection_regression",
        "approved_source": TRANSACTION_TOOL_REL.as_posix(),
        "production_transaction_function": "commit_packaging_transaction",
        "synthetic_transaction_function": "commit_packaging_transaction", "same_function_used": True,
        "injection_count": len(rows), "pass_count": passed, "fail_count": len(rows) - passed,
        "remaining_final_output_count": sum(row["remaining_final_outputs"] for row in rows),
        "remaining_temporary_output_count": sum(row["remaining_temporary_outputs"] for row in rows),
        "project_file_change_count": sum(row["project_files_changed"] for row in rows),
        "rows": rows, "status": "PASS",
    }


def never_overwrite_regression(validation_root: Path, transaction: Any) -> dict[str, Any]:
    rows = []
    for index, collision in enumerate(("result", "delivery", "detached")):
        with tempfile.TemporaryDirectory(prefix="step9l_v1_1_never_overwrite_", dir=validation_root) as temporary:
            sandbox = Path(temporary)
            finals = (sandbox / "final_result", sandbox / "final_delivery.zip", sandbox / "final_delivery.zip.sha256")
            selected = finals[index]
            if collision == "result":
                selected.mkdir()
                marker = selected / "sentinel.bin"
                marker.write_bytes(b"pre-existing-result")
                before = sha256_file(marker)
            else:
                selected.write_bytes(("pre-existing-" + collision).encode("ascii"))
                marker = selected
                before = sha256_file(marker)
            rejected = False
            try:
                transaction.commit_packaging_transaction(
                    result_files={"marker.json": b"{}\n"}, delivery_bytes=b"delivery", detached_bytes=b"detached",
                    final_result=finals[0], final_delivery=finals[1], final_detached=finals[2],
                    allowed_root=sandbox, transaction_token="never_overwrite_" + str(index),
                )
            except transaction.Step9IRepairError as error:
                rejected = "never-overwrite" in str(error)
            unchanged = marker.is_file() and sha256_file(marker) == before
            other_count = sum(path.exists() for idx, path in enumerate(finals) if idx != index)
            temp_count = sum(1 for _ in sandbox.glob(".step9i_v1_1_*"))
            passed = rejected and unchanged and other_count == 0 and temp_count == 0
            rows.append({"collision_target": collision, "expected_rejection_observed": rejected, "pre_existing_bytes_unchanged": unchanged, "additional_final_outputs": other_count, "temporary_outputs": temp_count, "verdict": "PASS" if passed else "FAIL"})
    require(all(row["verdict"] == "PASS" for row in rows), "never-overwrite regression failed")
    return {"schema_version": SCHEMA + ".never_overwrite_regression", "case_count": 3, "pass_count": 3, "fail_count": 0, "rows": rows, "status": "PASS"}


def clean_run_determinism(
    validation_root: Path,
    transaction: Any,
    result_files: dict[str, bytes],
    delivery_name: str,
) -> dict[str, Any]:
    zip_data = transaction.deterministic_zip_bytes(result_files)
    detached_data = transaction.strict_detached_bytes(zip_data, delivery_name)
    runs = []
    with tempfile.TemporaryDirectory(prefix="step9l_v1_1_clean_runs_", dir=validation_root) as temporary:
        base = Path(temporary)
        for index in range(2):
            sandbox = base / ("run_" + str(index + 1))
            sandbox.mkdir()
            result = sandbox / "result"
            delivery = sandbox / delivery_name
            detached = sandbox / (delivery_name + ".sha256")
            transaction.commit_packaging_transaction(
                result_files=result_files, delivery_bytes=zip_data, detached_bytes=detached_data,
                final_result=result, final_delivery=delivery, final_detached=detached,
                allowed_root=sandbox, transaction_token="clean_run_" + str(index + 1),
                post_rename_validator=lambda r, d, c: validate_committed_outputs(transaction, result_files, zip_data, detached_data, r, d, c),
            )
            tree = tree_identity(result)
            runs.append({
                "run_index": index + 1, "file_count": tree["file_count"], "output_tree_records_sha256": tree["records_sha256"],
                "delivery_size_bytes": delivery.stat().st_size, "delivery_sha256": sha256_file(delivery),
                "detached_size_bytes": detached.stat().st_size, "detached_file_sha256": sha256_file(detached),
            })
    conditions = {
        "output_tree_byte_identical": runs[0]["output_tree_records_sha256"] == runs[1]["output_tree_records_sha256"],
        "zip_byte_identical": runs[0]["delivery_sha256"] == runs[1]["delivery_sha256"],
        "checksum_byte_identical": runs[0]["detached_file_sha256"] == runs[1]["detached_file_sha256"],
    }
    require(all(conditions.values()), "two clean run determinism mismatch")
    return {
        "schema_version": SCHEMA + ".independent_clean_run_determinism",
        "production_transaction_function": "commit_packaging_transaction", "clean_run_count": 2,
        "runs": runs, "conditions": conditions, "status": "PASS",
        "scope_note": "The two clean runs cover the complete deterministic core file map; this evidence file and the final manifest are then added deterministically.",
    }


def assertion(key: str, passed: bool | None, subject: str, evidence: Any) -> dict[str, Any]:
    status = "UNVERIFIED" if passed is None else ("PASS" if passed else "FAIL")
    return {"assertion_key": key, "status": status, "subject": subject, "evidence": evidence}


def output_category(name: str) -> str:
    if name.startswith("project_files/"): return "approved_v1_0_project_artifact"
    if name.startswith("input_subset/"): return "fixed_input_provenance"
    if name.startswith("analysis/"): return "analysis_source"
    if name.startswith("source_before/"): return "approved_v1_0_source_before"
    if name == "ASSERTIONS.jsonl": return "assertions"
    if name == "VALIDATION_REPORT.md": return "validation_report"
    return "validation_output"


def report_text(comparison: dict[str, Any], protection: dict[str, Any], transaction: dict[str, Any], assertions: dict[str, int]) -> str:
    single = comparison["single_medoid"]
    multi = comparison["multi_prototype"]
    return "\n".join([
        "# WordsInSilence Step 9L v1.1 Validation Report", "",
        "## Outcome", "",
        "- v1.0-approved C#, test, candidate profile/config, 9 WISRT, 81 DTW and strict LOO artifacts: byte-identical",
        "- Repair scope: packaging transaction plus missing comparison/protection evidence only",
        "- Unity install, compile, EditMode and live recognition: not executed", "",
        "## Single-medoid vs multi-prototype", "",
        f"- Query-leak-free folds: `{single['query_leak_free_fold_count']}/9` and `{multi['query_leak_free_fold_count']}/9`",
        f"- Predictions: `{single['prediction_correct_count']}/9` and `{multi['prediction_correct_count']}/9`",
        f"- GREETING gap: `{single['per_word']['GREETING']['separation_gap']}` -> `{multi['per_word']['GREETING']['separation_gap']}`",
        f"- Multi gaps: GREETING `{multi['per_word']['GREETING']['separation_gap']}`, LOVE `{multi['per_word']['LOVE']['separation_gap']}`, THANK_YOU `{multi['per_word']['THANK_YOU']['separation_gap']}`", "",
        "## Protection and packaging", "",
        f"- Global runtime assets: `{protection['global_runtime_assets']['byte_identical_count']}/11`, mismatches `{protection['global_runtime_assets']['mismatch_count']}`",
        f"- Existing personal profile: `{protection['existing_personal_profile']['byte_identical_count']}/10`",
        f"- Approved C# files: `{protection['approved_csharp']['byte_identical_count']}/4`",
        f"- Failure injection: `{transaction['pass_count']}/{transaction['injection_count']}`, residual final/temp `0/0`", "",
        "## Assertions", "",
        f"- PASS: `{assertions['PASS']}`", f"- FAIL: `{assertions['FAIL']}`",
        f"- UNVERIFIED: `{assertions['UNVERIFIED']}`", f"- NOT_APPLICABLE: `{assertions['NOT_APPLICABLE']}`", "",
        "## Gate", "",
        "The v1.1 outputs are frozen pending independent external review. Unity and live validation remain outside this repair.", "",
    ])


def manifest_for(files: dict[str, bytes]) -> dict[str, Any]:
    records = [{
        "relative_path": name, "category": output_category(name),
        "size_bytes": len(payload), "sha256": sha256_bytes(payload),
    } for name, payload in sorted(files.items())]
    return {"schema_version": SCHEMA + ".output_file_manifest", "self_excluded": "output_file_manifest.json", "output_count_excluding_manifest": len(records), "outputs": records}


def build(root: Path, output: Path, delivery: Path) -> dict[str, Any]:
    validation_root = output.parent
    detached = Path(str(delivery) + ".sha256")
    for target in (output, delivery, detached):
        require(not target.exists() and not target.is_symlink(), f"never-overwrite final target exists: {target}")

    transaction_module, transaction_source = load_transaction_module(root)
    v1_map, v1_fixed, v1_zip_before, v1_detached_before = verify_fixed_delivery(
        root, V1_REL, V1_SIZE, V1_SHA256, V1_DETACHED_SIZE, V1_DETACHED_FILE_SHA256,
        V1_ENTRY_COUNT, V1_MANIFEST_TARGET_COUNT,
    )
    step9h_map, step9h_fixed, step9h_zip_before, step9h_detached_before = verify_fixed_delivery(
        root, STEP9H_REL, STEP9H_SIZE, STEP9H_SHA256, STEP9H_DETACHED_SIZE,
        STEP9H_DETACHED_FILE_SHA256, STEP9H_ENTRY_COUNT, STEP9H_MANIFEST_TARGET_COUNT,
    )
    require(len(v1_map[V1_ANALYSIS_PATH]) == V1_ANALYSIS_SIZE and sha256_bytes(v1_map[V1_ANALYSIS_PATH]) == V1_ANALYSIS_SHA256, "approved v1.0 analysis source mismatch")
    require(transaction_module.deterministic_zip_bytes(v1_map) == v1_zip_before, "v1.0 deterministic ZIP regeneration mismatch")
    require(transaction_module.deterministic_zip_bytes(step9h_map) == step9h_zip_before, "Step 9H deterministic ZIP regeneration mismatch")

    git_before = git_staged(root)
    original_source_before = verify_original_source(root)
    csharp_before = snapshot_group(root, fixed_records(APPROVED_CSHARP4), "approved_csharp")
    personal_before = snapshot_group(root, fixed_records(PERSONAL_PROFILE10), "existing_personal_profile")
    global_before = snapshot_group(root, fixed_records(GLOBAL_RUNTIME11), "global_runtime_assets")
    require_exact_group_membership(root, "existing_personal_profile", fixed_records(PERSONAL_PROFILE10))
    require_exact_group_membership(root, "global_runtime_assets", fixed_records(GLOBAL_RUNTIME11))
    require(global_before["required_count"] == 11 and global_before["actual_count"] == 11, "global runtime baseline must contain exactly 11 files")

    preserved, preservation = v1_preservation(v1_map)
    comparison, comparison_cells_csv, comparison_queries_csv = comparison_evidence(v1_map, step9h_map)
    project_snapshot = compact_snapshot([csharp_before, personal_before, global_before], original_source_before)
    transaction_regression = transaction_failure_regression(root, validation_root, transaction_module, project_snapshot)
    never_overwrite = never_overwrite_regression(validation_root, transaction_module)

    original_source_after = verify_original_source(root)
    csharp_after = snapshot_group(root, fixed_records(APPROVED_CSHARP4), "approved_csharp")
    personal_after = snapshot_group(root, fixed_records(PERSONAL_PROFILE10), "existing_personal_profile")
    global_after = snapshot_group(root, fixed_records(GLOBAL_RUNTIME11), "global_runtime_assets")
    require_exact_group_membership(root, "existing_personal_profile", fixed_records(PERSONAL_PROFILE10))
    require_exact_group_membership(root, "global_runtime_assets", fixed_records(GLOBAL_RUNTIME11))
    protection = {
        "schema_version": SCHEMA + ".protected_files_before_after",
        "global_runtime_assets": {**global_after, "before": global_before["actual"], "after": global_after["actual"], "empty_baseline_rejected": True, "byte_identical_11_of_11": global_before["actual"] == global_after["actual"]},
        "existing_personal_profile": {**personal_after, "before": personal_before["actual"], "after": personal_after["actual"], "byte_identical_10_of_10": personal_before["actual"] == personal_after["actual"]},
        "approved_csharp": {**csharp_after, "before": csharp_before["actual"], "after": csharp_after["actual"], "byte_identical_4_of_4": csharp_before["actual"] == csharp_after["actual"]},
        "original_project_runtime_source": {"before": original_source_before, "after": original_source_after, "byte_identical": original_source_before == original_source_after},
        "total_mismatch_count": global_after["mismatch_count"] + personal_after["mismatch_count"] + csharp_after["mismatch_count"] + int(original_source_before != original_source_after),
        "project_install_or_activation_performed": False,
    }
    require(protection["total_mismatch_count"] == 0, "protected project file changed")
    git_after = git_staged(root)

    input_after = {
        "v1_0_zip_unchanged": (root / V1_REL).read_bytes() == v1_zip_before,
        "v1_0_detached_unchanged": Path(str(root / V1_REL) + ".sha256").read_bytes() == v1_detached_before,
        "step9h_zip_unchanged": (root / STEP9H_REL).read_bytes() == step9h_zip_before,
        "step9h_detached_unchanged": Path(str(root / STEP9H_REL) + ".sha256").read_bytes() == step9h_detached_before,
        "transaction_source_unchanged": sha256_file(root / TRANSACTION_TOOL_REL) == TRANSACTION_TOOL_SHA256,
    }
    require(all(input_after.values()), "fixed input changed during execution")

    files = dict(preserved)
    files.update({
        "analysis/s001_step9l_personal_multi_prototype_runtime_candidate_v1_1.py": Path(__file__).read_bytes(),
        "input_subset/provenance/s001_step9l_personal_multi_prototype_runtime_candidate_v1_0.py": v1_map[V1_ANALYSIS_PATH],
        "input_subset/provenance/s001_step9i_personal_runtime_opt_in_integration_v1_1.py": (root / TRANSACTION_TOOL_REL).read_bytes(),
        "input_subset/provenance/step9l_v1_0_output_file_manifest.json": v1_map["output_file_manifest.json"],
        "input_subset/provenance/step9h_class_conditional_fold_scores.csv": step9h_map["class_conditional_fold_scores.csv"],
        "input_subset/provenance/step9h_class_conditional_threshold_derivation.json": step9h_map["class_conditional_threshold_derivation.json"],
        "input_subset/provenance/step9h_output_file_manifest.json": step9h_map["output_file_manifest.json"],
        "input_provenance.json": json_bytes({
            "schema_version": SCHEMA + ".input_provenance", "fixed_v1_0": v1_fixed,
            "fixed_step9h": step9h_fixed, "approved_transaction_source": transaction_source,
            "fixed_inputs_unchanged_after": input_after,
        }),
        "v1_0_approved_artifact_preservation.json": json_bytes(preservation),
        "single_medoid_multi_prototype_comparison.json": json_bytes(comparison),
        "single_medoid_multi_prototype_score_cells.csv": comparison_cells_csv,
        "single_medoid_multi_prototype_queries.csv": comparison_queries_csv,
        "protected_files_before_after.json": json_bytes(protection),
        "global_runtime_asset_protection.json": json_bytes(protection["global_runtime_assets"]),
        "packaging_transaction_failure_injection_regression.json": json_bytes(transaction_regression),
        "never_overwrite_regression.json": json_bytes(never_overwrite),
    })

    clean_runs = clean_run_determinism(validation_root, transaction_module, files, delivery.name)
    files["independent_clean_run_determinism.json"] = json_bytes(clean_runs)

    primary = {
        "schema_version": SCHEMA + ".primary_verdict",
        "actual_primary_verdict": "MINIMAL_PACKAGE_EVIDENCE_REPAIR_READY_PENDING_EXTERNAL_REVIEW",
        "v1_0_calculation_wisrt_strict_loo_preserved": True,
        "single_medoid_baseline_comparison_complete": True,
        "global_runtime_asset_protection_11_of_11": True,
        "packaging_transaction_failure_injection_pass": True,
        "unity_install_compile_editmode_live_executed": False,
        "status_strings": [
            "STEP9L_PERSONAL_MULTI_PROTOTYPE_RUNTIME_CANDIDATE_V1_1_SOURCE_AND_PACKAGE_READY",
            "STEP9L_V1_0_APPROVED_CALCULATION_SOURCE_AND_ASSETS_PRESERVED",
            "STEP9L_SINGLE_MEDOID_MULTI_PROTOTYPE_BASELINE_COMPARISON_PASS",
            "STEP9L_GLOBAL_RUNTIME_ASSETS_PROTECTED_11_OF_11",
            "STEP9L_PACKAGING_TRANSACTION_FAILURE_INJECTION_PASS",
            "STEP9L_UNITY_COMPILE_EDITMODE_LIVE_UNVERIFIED",
            "BLOCKED_PENDING_EXTERNAL_REVIEW",
        ],
    }
    files["primary_verdict.json"] = json_bytes(primary)

    assertion_rows = [
        assertion("fixed_step9l_v1_0_delivery_identity_archive_manifest_determinism", v1_fixed["manifest"]["membership"] == "PASS", V1_NAME, v1_fixed),
        assertion("fixed_step9h_delivery_identity_archive_manifest_determinism", step9h_fixed["manifest"]["membership"] == "PASS", STEP9H_NAME, step9h_fixed),
        assertion("approved_step9i_v1_1_transaction_source_identity", transaction_source["sha256"] == TRANSACTION_TOOL_SHA256, TRANSACTION_TOOL_REL.as_posix(), transaction_source),
        assertion("v1_0_approved_artifacts_byte_identical", preservation["mismatch_count"] == 0, "v1.0 approved scope", preservation["explicit_approved_scope"]),
        assertion("single_medoid_score_and_threshold_independent_parity", comparison["single_saved_parity"]["status"] == "PASS", "Step 9H baseline", comparison["single_saved_parity"]),
        assertion("multi_prototype_score_and_threshold_independent_parity", comparison["multi_saved_parity"]["status"] == "PASS", "Step 9L v1.0", comparison["multi_saved_parity"]),
        assertion("both_routes_query_leak_free_nine_of_nine", comparison["conditions"]["single_query_leak_free_9_of_9"] and comparison["conditions"]["multi_query_leak_free_9_of_9"], "18 folds", comparison["conditions"]),
        assertion("both_routes_prediction_nine_of_nine", comparison["conditions"]["single_prediction_9_of_9"] and comparison["conditions"]["multi_prediction_9_of_9"], "18 predictions", comparison["conditions"]),
        assertion("candidate_three_class_gaps_positive", comparison["conditions"]["multi_all_three_gaps_positive"], "multi-prototype thresholds", {word: comparison["multi_prototype"]["per_word"][word]["separation_gap"] for word in WORDS}),
        assertion("greeting_gap_improves_from_single_to_multi", comparison["conditions"]["greeting_gap_improved"], "GREETING", comparison["greeting_gap"]),
        assertion("global_runtime_assets_exact_11_byte_identical", protection["global_runtime_assets"]["byte_identical_11_of_11"] and protection["global_runtime_assets"]["actual_count"] == 11, "approved Step 9I global assets", protection["global_runtime_assets"]),
        assertion("existing_personal_profile_exact_10_byte_identical", protection["existing_personal_profile"]["byte_identical_10_of_10"], "Step 9H personal profile", {"count": 10}),
        assertion("approved_csharp_exact_4_byte_identical", protection["approved_csharp"]["byte_identical_4_of_4"], "approved C#", {"count": 4}),
        assertion("original_project_runtime_source_unchanged", protection["original_project_runtime_source"]["byte_identical"], ORIGINAL_RUNTIME_SOURCE["relative_path"], protection["original_project_runtime_source"]),
        assertion("packaging_transaction_failure_injection_five_of_five", transaction_regression["pass_count"] == 5, "production commit_packaging_transaction", transaction_regression),
        assertion("failure_injection_partial_final_and_temp_zero", transaction_regression["remaining_final_output_count"] == 0 and transaction_regression["remaining_temporary_output_count"] == 0, "five injected failures", {"final": 0, "temp": 0}),
        assertion("never_overwrite_three_targets_preserve_bytes", never_overwrite["pass_count"] == 3, "result/Delivery/detached", never_overwrite),
        assertion("two_independent_clean_runs_byte_identical", all(clean_runs["conditions"].values()), "output tree, ZIP, checksum", clean_runs),
        assertion("fixed_inputs_before_after_unchanged", all(input_after.values()), "fixed input bytes", input_after),
        assertion("git_staged_zero_before_after", git_before["staged_count"] == 0 and git_after["staged_count"] == 0, "Git", {"before": git_before, "after": git_after}),
        assertion("unity_compile_editmode_live_not_executed", None, "Unity and live recognition", {"executed": False, "reason": "prohibited in Step 9L v1.1 repair"}),
    ]
    require(len({row["assertion_key"] for row in assertion_rows}) == len(assertion_rows), "assertion key duplicate")
    counts = Counter(row["status"] for row in assertion_rows)
    for status in ("PASS", "FAIL", "UNVERIFIED", "NOT_APPLICABLE"):
        counts.setdefault(status, 0)
    require(counts["FAIL"] == 0, "assertion failure present")
    files["ASSERTIONS.jsonl"] = b"".join(json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n" for row in assertion_rows)
    files["VALIDATION_REPORT.md"] = report_text(comparison, protection, transaction_regression, dict(counts)).encode("utf-8")
    expected_manifest_target_count = len(files)
    files["output_file_manifest.json"] = json_bytes(manifest_for(files))
    require(len(json.loads(files["output_file_manifest.json"])["outputs"]) == expected_manifest_target_count, "output manifest target count mismatch")
    require(set(files) == {record["relative_path"] for record in json.loads(files["output_file_manifest.json"])["outputs"]} | {"output_file_manifest.json"}, "output manifest membership mismatch")

    zip_data = transaction_module.deterministic_zip_bytes(files)
    require(transaction_module.deterministic_zip_bytes(files) == zip_data, "final deterministic ZIP regeneration mismatch")
    zip_check = transaction_module.inspect_zip(zip_data, set(files))
    detached_data = transaction_module.strict_detached_bytes(zip_data, delivery.name)

    transaction_module.commit_packaging_transaction(
        result_files=files, delivery_bytes=zip_data, detached_bytes=detached_data,
        final_result=output, final_delivery=delivery, final_detached=detached,
        allowed_root=validation_root, transaction_token="step9l_v1_1_production",
        post_rename_validator=lambda r, d, c: validate_committed_outputs(transaction_module, files, zip_data, detached_data, r, d, c),
    )

    require(verify_original_source(root) == original_source_before, "original project source changed after production commit")
    require(git_staged(root)["staged_count"] == 0, "Git staged files changed after production commit")
    return {
        "output_directory": output.as_posix(), "output_file_count": len(files),
        "delivery_zip": delivery.as_posix(), "delivery_size_bytes": len(zip_data), "delivery_sha256": sha256_bytes(zip_data),
        "detached_checksum": detached.as_posix(), "detached_size_bytes": len(detached_data), "detached_file_sha256": sha256_bytes(detached_data),
        "zip_verification": zip_check, "assertions": dict(counts),
        "single_medoid_prediction": comparison["single_medoid"]["prediction_correct_count"],
        "multi_prototype_prediction": comparison["multi_prototype"]["prediction_correct_count"],
        "greeting_gap_before": comparison["greeting_gap"]["single_medoid"],
        "greeting_gap_after": comparison["greeting_gap"]["multi_prototype"],
        "global_runtime_assets_byte_identical": protection["global_runtime_assets"]["byte_identical_count"],
        "unity": "UNVERIFIED_NOT_EXECUTED", "status": primary["actual_primary_verdict"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.project_root.resolve()
    validation = root / "Validation"
    validation.mkdir(parents=True, exist_ok=True)
    name = "WordsInSilence_step9l_personal_multi_prototype_runtime_candidate_v1_1"
    result = build(root, validation / name, validation / (name + "_delivery.zip"))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Step9LRepairError as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, ensure_ascii=False, indent=2))
        raise SystemExit(1)
