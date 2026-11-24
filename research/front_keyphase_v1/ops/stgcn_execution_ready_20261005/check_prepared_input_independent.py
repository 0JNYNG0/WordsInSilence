"""Bounded independent verification; no production IO/model helpers imported.

Reads the new prepared cache and metadata, plus exactly six representative
source NPZs. Reconstructs the frozen input expression directly, without
training, model forward, augmentation, or final evaluation score access.
"""
from pathlib import Path
import hashlib
import io
import json

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
PREP = ROOT / "prepared"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def tensor_sha(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(f"{array.dtype.str}|{array.shape}|".encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def main():
    split_path = ROOT / "TRAIN_SPLIT.json"
    frozen = read_json(ROOT / "FROZEN_EXPERIMENT.json")
    split = read_json(split_path)
    meta = read_json(PREP / "TRAINING_META.json")
    diags = read_json(PREP / "ROW_DIAGNOSTICS.json")
    manifest_path = ROOT.parent / "stgcn_transfer_compare_20261005/INPUT_MANIFEST.json"
    manifest = read_json(manifest_path)
    raw_cache = (PREP / "training.npz").read_bytes()
    with np.load(io.BytesIO(raw_cache), allow_pickle=False) as archive:
        cache = {key: np.array(archive[key]) for key in archive.files}
    checks = {}

    def check(name, condition):
        checks[name] = bool(condition)
        if not checks[name]:
            raise AssertionError(name)

    check("split_pin", sha(split_path.read_bytes()) == frozen["pins"]["train_split"] == meta["source_split_sha256"])
    check("manifest_pin", sha(manifest_path.read_bytes()) == frozen["pins"]["input_manifest"])
    check("cache_sha", sha(raw_cache) == meta["cache_sha256"])
    check("policy", meta["input_policy"] == frozen["input_policy"] == "provided_xy_isotropic_v1")
    check("class_order", meta["classes"] == split["classes"] == manifest["classes"] and len(meta["classes"]) == 93)
    check("cache_keys", set(cache) == {"x", "y", "weight", "fit", "internal_val", "primary"})
    check("shape", cache["x"].shape == (2346, 6, 2, 10, 27))
    check("dtype", cache["x"].dtype == np.float32 and cache["y"].dtype == np.int64 and cache["weight"].dtype == np.float64)
    check("finite", np.isfinite(cache["x"]).all() and np.isfinite(cache["weight"]).all())
    check("row_counts", len(meta["rows"]) == len(meta["ids"]) == len(meta["origins"]) == len(diags) == len(split["rows"]) == 2346)
    expected_specs = {
        "native67": {"xy_key": "xy_native", "mask_key": "coordinate_present", "time_key": "timestamp_nominal_us", "vertices": 67, "hand_slices": {"left": [25, 46], "right": [46, 67]}},
        "mp75": {"xy_key": "xy_px", "mask_key": "observed", "time_key": "timestamp_us", "vertices": 75, "hand_slices": {"left": [33, 54], "right": [54, 75]}},
    }
    check("layout_specs", meta["layout_specs"] == expected_specs)
    expected_ids = [f"{r['group']}:{r['manifest_index']}:{r['id']}" for r in split["rows"]]
    check("unique_row_ids", meta["ids"] == expected_ids and len(set(expected_ids)) == 2346)
    check("origins", meta["origins"] == [r["origin"] for r in split["rows"]])
    fit = np.array([r["partition"] == "fit" for r in split["rows"]])
    val = np.array([r["partition"] == "internal_val" for r in split["rows"]])
    primary = np.array([r["primary_validation"] for r in split["rows"]])
    check("partitions", np.array_equal(cache["fit"], fit) and np.array_equal(cache["internal_val"], val) and np.array_equal(cache["primary"], primary))
    check("partition_coverage", (fit ^ val).all() and fit.sum() == 2089 and val.sum() == 257 and primary.sum() == 233)
    check("labels_weights", np.array_equal(cache["y"], [r["y"] for r in split["rows"]]) and np.array_equal(cache["weight"], [r["weight_count"] for r in split["rows"]]))
    check("zero_weight_preserved", (cache["weight"] == 0).sum() == 19 and (cache["weight"][fit] == 0).sum() == 16 and (cache["weight"][val] == 0).sum() == 3)
    check("origin_disjoint", not (set(np.array(meta["origins"])[fit]) & set(np.array(meta["origins"])[val])))
    check("source_hash_disjoint", not ({r["sha256"] for r in split["rows"] if r["partition"] == "fit"} & {r["sha256"] for r in split["rows"] if r["partition"] == "internal_val"}))
    check("only_original_training_rows", {r["manifest_index"] for r in split["rows"]} == {i for i, r in enumerate(manifest["rows"]) if r["role"] == "train"})
    for i, (source, row, diag) in enumerate(zip(split["rows"], meta["rows"], diags)):
        expected_row = {**source, "id": expected_ids[i]}
        check(f"row_{i}", row == expected_row and diag["id"] == expected_ids[i])
        original = manifest["rows"][source["manifest_index"]]
        check(f"source_{i}", all(source[k] == original[k] for k in ("group", "id", "label", "y", "signer_id", "path", "sha256", "layout", "start", "end_exclusive", "weight_count")))
        check(f"diag_{i}", diag["source_sha256"] == source["sha256"] and diag["slice"] == [source["start"], source["end_exclusive"]] and diag["layout"] == source["layout"] and diag["mask_hand"] == original["mask_hand"] is None and diag["hand_slice"] is None)
        frames = source["end_exclusive"] - source["start"]
        indices = [-1] * (60 - frames) + list(range(frames)) if frames <= 60 else [j * (frames - 1) // 59 for j in range(60)]
        check(f"time_{i}", diag["prepare_diag"]["T"] == frames and diag["prepare_diag"]["indices"] == indices and diag["prepare_diag"]["shape"] == [1, 6, 2, 10, 27])
        check(f"tensor_sha_{i}", tensor_sha(cache["x"][i]) == diag["input_tensor_sha256"])

    hand = [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]
    selections = {
        "native67": [0, 16, 15, 5, 2, 6, 3] + [25 + j for j in hand] + [46 + j for j in hand],
        "mp75": [0, 2, 5, 11, 12, 13, 14] + [33 + j for j in hand] + [54 + j for j in hand],
    }
    representative_positions = [("native_short", 0), ("native_long", 910), ("native_apology_base", 2113), ("native_same_origin_context", 2140), ("NIKL_nonzero_slice", 2095), ("NIKL_study", 2098)]
    representatives = []
    for kind, position in representative_positions:
        row = meta["rows"][position]
        source_path = Path(row["path"])
        before = source_path.stat()
        raw_source = source_path.read_bytes()
        check(f"representative_sha_{position}", sha(raw_source) == row["sha256"])
        spec = expected_specs[row["layout"]]
        with np.load(io.BytesIO(raw_source), allow_pickle=False) as z:
            xy = np.array(z[spec["xy_key"]])[row["start"]:row["end_exclusive"]].astype(np.float32)
            present = np.array(z[spec["mask_key"]])[row["start"]:row["end_exclusive"]].astype(bool)
            timestamps = np.array(z[spec["time_key"]]).astype(np.float64)
            observed_not_usable = int((present & ~np.array(z["usable"])[row["start"]:row["end_exclusive"]].astype(bool)).sum()) if row["layout"] == "mp75" and "usable" in z.files else None
        check(f"representative_times_{position}", np.isfinite(timestamps).all() and (np.diff(timestamps) > 0).all())
        check(f"representative_observed_finite_{position}", np.isfinite(xy[present]).all())
        xy[~present] = 0
        joints = selections[row["layout"]]
        selected = xy[:, joints]
        # Independent direct tensor expression, matching the frozen arithmetic order.
        clip = torch.tensor(selected, dtype=torch.float32)
        center = ((clip[:, 3] + clip[:, 4]) / 2).mean(dim=0)
        distance = torch.sqrt(((clip[:, 3] - clip[:, 4]) ** 2).sum(-1)).mean()
        check(f"representative_scale_{position}", torch.isfinite(distance) and distance > 0)
        normalized = (clip - center) * (1.0 / distance)
        frames = len(clip)
        chosen = list(range(frames)) if frames <= 60 else [j * (frames - 1) // 59 for j in range(60)]
        packed = torch.zeros((60, 27, 2), dtype=torch.float32)
        if frames <= 60:
            packed[60 - frames:] = normalized
        else:
            packed[:] = normalized[chosen]
        expected = packed.reshape(6, 10, 27, 2).permute(0, 3, 1, 2).numpy()
        error = float(np.abs(expected - cache["x"][position]).max())
        check(f"representative_math_{position}", np.allclose(expected, cache["x"][position], atol=2e-6, rtol=2e-6))
        all_t_distance64 = float(np.linalg.norm(selected[:, 3].astype(np.float64) - selected[:, 4].astype(np.float64), axis=1).mean())
        check(f"representative_diag_scale_{position}", all_t_distance64 == diags[position]["prepare_diag"]["shoulder_distance"])
        after = source_path.stat()
        check(f"representative_source_unchanged_{position}", (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns))
        representatives.append({"kind": kind, "row": position, "id": row["id"], "path": row["path"], "source_sha256": sha(raw_source), "layout": row["layout"], "mask_key": spec["mask_key"], "slice": [row["start"], row["end_exclusive"]], "T": frames, "selected_missing_points": int((~present[:, joints]).sum()), "observed_not_usable_points": observed_not_usable, "max_abs_error": error, "all_T_shoulder_distance": all_t_distance64, "source_stat_unchanged": True})

    output = {
        "status": "PASS",
        "review": "Independent new prepared-input wiring verification; no production IO/normalizer/model helpers called",
        "cache_sha256": sha(raw_cache),
        "frozen_experiment_sha256": sha((ROOT / "FROZEN_EXPERIMENT.json").read_bytes()),
        "verifier_sha256": sha(Path(__file__).read_bytes()),
        "checks": len(checks),
        "failed_checks": [],
        "all_row_metadata_tensor_digest_checks": 2346,
        "source_npz_read_count": len(representatives),
        "representatives": representatives,
        "counts": {"refit": 2346, "fit": 2089, "internal_val": 257, "primary_validation": 233, "context_validation": 24, "zero_weight_refit": 19, "zero_weight_fit": 16, "zero_weight_validation": 3},
        "unresolved_or_not_exercised": ["All training mask_hand values are None; final masked conditions were not loaded here.", "NIKL actor identities remain unknown; origin/native-signer disjointness is not a claim of person independence across datasets.", "Only six raw source NPZs independently reconstructed; all-row digest/metadata verification does not assert new all-source content recomputation.", "Native W/H and original OpenHands coordinate equivalence remain UNVERIFIED; this validates provided_xy_isotropic_v1.", "No model forward, training, augmentation execution or final evaluation score access."],
    }
    (ROOT / "PREPARED_INPUT_INDEPENDENT_CHECK.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "checks": output["checks"], "rows": 2346, "source_npz_read_count": len(representatives), "max_abs_error": max(r["max_abs_error"] for r in representatives)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
