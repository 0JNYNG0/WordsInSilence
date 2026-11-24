"""Independent audit helpers; no camera, decoder, extractor, or learner imports.

Reads saved public result arrays only. Eight-forward independent verification
requires --score; no path imports a decoder, extractor, or camera module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np


FIELDS = ("xy_px", "timestamp_us", "frame_id", "observed", "usable", "conf", "hand_score")
GROUPS = {"all": slice(0, 75), "pose": slice(0, 33),
          "left": slice(33, 54), "right": slice(54, 75)}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while block := fh.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def exact(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if np.issubdtype(a.dtype, np.inexact):
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def validate_observation(obs):
    count = len(obs["timestamp_us"])
    shapes = {"xy_px": ((count, 75, 2), np.dtype("float32")),
              "timestamp_us": ((count,), np.dtype("int64")),
              "frame_id": ((count,), np.dtype("int64")),
              "observed": ((count, 75), np.dtype("bool")),
              "usable": ((count, 75), np.dtype("bool")),
              "conf": ((count, 75), np.dtype("float32")),
              "hand_score": ((count, 2), np.dtype("float32"))}
    for key, (shape, dtype) in shapes.items():
        if obs[key].shape != shape or obs[key].dtype != dtype:
            raise AssertionError((key, obs[key].shape, obs[key].dtype, shape, dtype))
    if count < 1 or not np.all(np.diff(obs["timestamp_us"]) > 0):
        raise AssertionError("empty or nonincreasing timestamp sequence")
    if not np.all(np.diff(obs["frame_id"]) > 0):
        raise AssertionError("nonincreasing original frame IDs")
    if np.any(obs["usable"] & ~obs["observed"]):
        raise AssertionError("usable without observed")
    if np.isfinite(obs["xy_px"][~obs["observed"]]).any():
        raise AssertionError("unobserved coordinate has finite value")
    if not np.isfinite(obs["xy_px"][obs["usable"]]).all():
        raise AssertionError("usable coordinate has nonfinite value")
    return count


def compare_observations_exact(a, b):
    validate_observation(a)
    validate_observation(b)
    checks = {key: exact(a[key], b[key]) for key in FIELDS}
    if not all(checks.values()):
        raise AssertionError({"observation_mismatch": checks})
    return checks


def select_even(obs):
    validate_observation(obs)
    keep = obs["frame_id"] % 2 == 0
    return {key: obs[key][keep].copy() for key in FIELDS}


def longest_true_run(mask):
    best = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def mask_summary(a, b):
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    return {"total": int(a.size), "same": int(np.sum(a == b)),
            "both_true": int(np.sum(a & b)), "both_false": int(np.sum(~a & ~b)),
            "a_only": int(np.sum(a & ~b)), "b_only": int(np.sum(~a & b)),
            "a_true": int(np.sum(a)), "b_true": int(np.sum(b))}


def hand_summary(usable):
    mask = np.asarray(usable, dtype=bool)
    if mask.ndim != 2 or mask.shape[1] != 21 or not len(mask):
        raise AssertionError("expected nonempty [T,21] hand mask")
    any_valid, all_valid = mask.any(axis=1), mask.all(axis=1)
    return {"frames": len(mask), "any_usable_frames": int(any_valid.sum()),
            "all21_usable_frames": int(all_valid.sum()),
            "any_usable_rate": float(any_valid.mean()),
            "all21_usable_rate": float(all_valid.mean()),
            "all21_absent_frames": int((~any_valid).sum()),
            "longest_all21_absent_run": longest_true_run(~any_valid)}


def coordinate_summary(a_xy, a_use, b_xy, b_use):
    common = (a_use & b_use & np.isfinite(a_xy).all(axis=2)
              & np.isfinite(b_xy).all(axis=2))
    count = int(common.sum())
    if not count:
        return {"common_points": 0, "mean_px": None, "median_px": None,
                "p95_px": None, "max_px": None}
    delta = a_xy[common].astype(np.float64) - b_xy[common].astype(np.float64)
    distances = np.sqrt(np.sum(delta * delta, axis=1))
    return {"common_points": count, "mean_px": float(np.mean(distances)),
            "median_px": float(np.median(distances)),
            "p95_px": float(np.percentile(distances, 95)), "max_px": float(np.max(distances))}


def paired_metrics(a, b):
    validate_observation(a)
    validate_observation(b)
    if not exact(a["frame_id"], b["frame_id"]) or not exact(a["timestamp_us"], b["timestamp_us"]):
        raise AssertionError("paired coordinates do not use identical IDs and PTS")
    result = {}
    for name, sl in GROUPS.items():
        row = {key: mask_summary(a[key][:, sl], b[key][:, sl]) for key in ("observed", "usable")}
        row["coordinate"] = coordinate_summary(a["xy_px"][:, sl], a["usable"][:, sl],
                                                b["xy_px"][:, sl], b["usable"][:, sl])
        if name in ("left", "right"):
            row["a_hand"] = hand_summary(a["usable"][:, sl])
            row["b_hand"] = hand_summary(b["usable"][:, sl])
        result[name] = row
    return result


def window_indices(obs, start_us, end_us):
    if not start_us < end_us:
        raise AssertionError("invalid half-open window")
    ts = obs["timestamp_us"]
    indices = np.flatnonzero((ts >= start_us) & (ts < end_us))
    if not indices.size:
        raise AssertionError("empty frozen window")
    return indices


def required(value, message):
    if not value:
        raise AssertionError(message)


def close_float(a, b, *, coordinate=False):
    if a is None or b is None:
        return a is None and b is None
    if coordinate:
        return bool(np.isclose(a, b, atol=0.0001, rtol=0.000001))
    return float(a) == float(b)


def array_hash(a):
    a = np.ascontiguousarray(a)
    prefix = a.dtype.str + "|" + "x".join(map(str, a.shape)) + "|"
    return hashlib.sha256(prefix.encode("ascii") + a.tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_result_file(result_dir, name):
    p = Path(name)
    required(not p.is_absolute() and p.name == name and name not in (".", ".."),
             "result entry is not a leaf filename")
    target = result_dir / name
    required(target.is_file() and target.resolve().parent == result_dir.resolve(),
             "result entry missing or redirects outside result directory")
    return target


def read_arm(result_dir, entry, clip, arm):
    npz_path = safe_result_file(result_dir, entry["npz"])
    meta = read_json(safe_result_file(result_dir, entry["meta"]))
    required(sha256_file(npz_path) == meta["npz_sha256"], "raw NPZ hash mismatch")
    required(meta["clip_id"] == clip["clip_id"], "arm clip ID mismatch")
    expected_arm = "A_full_then_even" if arm != "B" else "B_even_rgb_only"
    required(meta["arm"] == expected_arm and meta["tag"] == arm, "arm metadata mismatch")
    with np.load(npz_path, allow_pickle=False) as z:
        obs = {key: z["obs_" + key].copy() for key in FIELDS}
        decoded = {key: z["decoded_" + key].copy() for key in
                   ("frame_id", "timestamp_us", "pts", "rgb_sha256", "rgb_shape", "processed")}
        selected = z["selected_frame_id"].copy()
    n = clip["expected_frames_from_prior"]
    required(exact(decoded["frame_id"], np.arange(n, dtype=np.int64)), "decoded IDs changed")
    even = np.arange(0, n, 2, dtype=np.int64)
    required(exact(selected, even), "selected IDs not fixed phase0/stride2")
    required(decoded["pts"].dtype == np.int64 and decoded["pts"].shape == (n,), "PTS schema")
    required(np.all(np.diff(decoded["pts"]) > 0), "PTS does not increase")
    time_base = Fraction(meta["time_base"])
    required(time_base > 0, "invalid time base")
    pts0 = int(decoded["pts"][0])
    expected_ts = np.asarray([int(round(float((int(p) - pts0) * time_base) * 1_000_000))
                              for p in decoded["pts"]], dtype=np.int64)
    required(exact(expected_ts, decoded["timestamp_us"]), "stored timestamps differ from PTS equation")
    expected_ids = np.arange(n, dtype=np.int64) if arm != "B" else even
    expected_mask = np.ones(n, dtype=bool) if arm != "B" else np.arange(n) % 2 == 0
    required(exact(expected_mask, decoded["processed"]), "processed RGB set differs from assigned arm")
    required(exact(obs["frame_id"], expected_ids), "stored observation IDs differ from processed IDs")
    required(exact(obs["timestamp_us"], expected_ts[expected_ids]), "observation PTS changed")
    validate_observation(obs)
    required(decoded["rgb_shape"].shape == (n, 3), "RGB shape metadata length")
    required(np.all(decoded["rgb_shape"] > 0) and np.all(decoded["rgb_shape"][:, 2] == 3),
             "RGB shape invalid")
    hashes = decoded["rgb_sha256"].tolist()
    required(len(hashes) == n and all(isinstance(h, str) and len(h) == 64
             and set(h) <= set("0123456789abcdef") for h in hashes), "RGB hash metadata invalid")
    required(meta["counts"]["decoded_frames"] == n, "decoded count mismatch")
    required(meta["counts"]["process_calls"] == len(expected_ids), "process count mismatch")
    return {"obs": obs, "selected": select_even(obs), "decoded": decoded, "meta": meta}


def compare_reported_metrics(reference, reported):
    mapping = {"all": "all", "pose": "pose", "left": "left_hand", "right": "right_hand"}
    for independent_name, reported_name in mapping.items():
        ref = reference[independent_name]
        for mask in ("observed", "usable"):
            row = reported[mask][reported_name]
            map_counts = {"entries": "total", "A_true": "a_true", "B_true": "b_true",
                          "A_true_B_false": "a_only", "A_false_B_true": "b_only"}
            for rk, ik in map_counts.items():
                required(row[rk] == ref[mask][ik], f"{reported_name}/{mask}/{rk} count differs")
            required(row["differing"] == ref[mask]["a_only"] + ref[mask]["b_only"], "mask diff count")
        row = reported["coordinate_distance_px"][reported_name]
        required(row["count"] == ref["coordinate"]["common_points"], "common-point count")
        for k in ("mean", "median", "p95", "max"):
            required(close_float(row[k], ref["coordinate"][k + "_px"], coordinate=True),
                     f"{reported_name} coordinate {k} differs")
        if independent_name not in ("left", "right"):
            continue
        for arm, key in (("A", "a_hand"), ("B", "b_hand")):
            row, truth = reported["hand_rates"][arm][reported_name], ref[key]
            required(row["frames"] == truth["frames"], "hand frame count")
            required(row["any_usable_fraction"] == truth["any_usable_rate"], "hand any usable rate")
            required(row["all21_usable_fraction"] == truth["all21_usable_rate"], "hand all usable rate")
            required(row["all21_absent_frames"] == truth["all21_absent_frames"], "whole hand absent count")
            required(row["all21_absent_max_run"] == truth["longest_all21_absent_run"], "absent max run")


def independent_scores(config, result, arms):
    # Import only when --score is explicitly set after parent permits result verification.
    import torch
    pilot_src = Path(config["workspace_root"]) / "research/sign_model_pilot_v1/src"
    sys.path.insert(0, str(pilot_src))
    from pilot.data import build_channels, collate
    from pilot.web_input import to_native_like
    from pilot.model_masked_v1 import MaskedSignTCN
    torch.set_num_threads(2)
    checkpoint = Path(config["scoring"]["checkpoint_path"])
    required(sha256_file(checkpoint) == config["scoring"]["checkpoint_sha256"], "checkpoint pin changed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    classes = list(payload["classes"])
    required(len(classes) == len(set(classes)) == payload["n_classes"] == 93, "model classes invalid")
    required(classes == result["scoring"]["classes"], "saved class order mismatch")
    model = MaskedSignTCN(width=payload["width"], dilations=tuple(payload["dilations"]),
                          kernel_size=payload["kernel_size"], dropout=payload["dropout"], n_classes=93)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    output, forwards = [], 0
    windows = {w["clip_id"]: w for w in result["scoring"]["windows"]}
    for clip in config["clips"]:
        score = clip["score"]
        if score is None:
            continue
        saved = windows[clip["clip_id"]]
        required(saved["label"] == score["label"] and saved["start_us"] == score["start_us"]
                 and saved["end_us_exclusive"] == score["end_us_exclusive"], "frozen score window changed")
        for arm in ("A", "B"):
            obs = arms[clip["clip_id"]][arm]["selected"]
            ix = window_indices(obs, score["start_us"], score["end_us_exclusive"])
            required(saved["window_frame_ids"] == obs["frame_id"][ix].tolist(), "scored frame IDs")
            required(saved["window_timestamp_us"] == obs["timestamp_us"][ix].tolist(), "scored PTS")
            xy, present = to_native_like(obs["xy_px"][ix], obs["usable"][ix])
            feature = build_channels(xy, present, obs["timestamp_us"][ix])
            x, pad, y = collate([dict(feature, label=0)])
            required(np.all(y == 0) and np.all(pad == 1), "unexpected label or pad mask")
            stored_input = saved[arm]["input"]
            required(array_hash(x) == stored_input["x_sha256"], "independent input hash mismatch")
            required(list(x.shape) == stored_input["x_shape"], "independent input shape mismatch")
            required(feature["body_scale"] == stored_input["body_scale"], "body scale mismatch")
            for side, scale in feature["hand_scales"].items():
                other = stored_input["hand_scales"][side]
                required((not np.isfinite(scale) and other is None) or scale == other, "hand scale mismatch")
            with torch.inference_mode():
                logits = model(torch.from_numpy(x), torch.from_numpy(pad))[0].numpy().astype(np.float64)
            forwards += 1
            reference = np.asarray(saved[arm]["logits93"], dtype=np.float64)
            required(logits.shape == reference.shape == (93,), "logit count mismatch")
            delta = float(np.max(np.abs(logits - reference)))
            required(delta == 0.0, f"independent model logits differ by {delta}")
            # lexsort chooses lower index in a tie, independently from runner's sorted list.
            order = np.lexsort((np.arange(93), -logits))
            truth = classes.index(score["label"])
            best = int(order[0])
            rank = int(np.flatnonzero(order == truth)[0]) + 1
            rival = int(order[order != truth][0])
            margin = float(logits[truth] - logits[rival])
            sr = saved[arm]["score"]
            required(sr["top1"] == classes[best] and sr["top1_index"] == best, "top1 mismatch")
            required(sr["true_rank"] == rank and sr["true_vs_best_other_margin"] == margin, "rank/margin mismatch")
            required(sr["correct"] == (best == truth), "correctness mismatch")
            output.append({"clip_id": clip["clip_id"], "arm": arm, "x_exact": True,
                           "logits_max_abs_diff": delta, "top1": classes[best], "true_rank": rank})
    required(forwards == 8, "independent inference count must be exactly 8")
    required(all(torch.equal(before[k], v) for k, v in model.state_dict().items()), "model state mutated")
    return {"status": "PASS", "model_loads": 1, "forwards": forwards, "all_state_unchanged": True,
            "windows": output, "max_abs_logit_difference": max(r["logits_max_abs_diff"] for r in output)}


def audit_result(config_path, result_dir, score=False):
    config_path, result_dir = Path(config_path).resolve(), Path(result_dir).resolve()
    config = read_json(config_path)
    required(result_dir == Path(config["output_dir"]).resolve(), "not the frozen result directory")
    required(config_path.parent == result_dir.parent, "config outside this unit")
    for filename, expected in config["pins"].items():
        required(sha256_file(filename) == expected, f"input pin changed: {filename}")
    result = read_json(result_dir / "RESULT.json")
    required(result["status"] == "COMPLETE", "technical failure is not a performance result")
    required(result["config"]["sha256_before"] == result["config"]["sha256_after"]
             == sha256_file(config_path), "configuration changed")
    hashes = read_json(result_dir / "FILE_HASHES.json")
    for name, digest in hashes.items():
        required(sha256_file(safe_result_file(result_dir, name)) == digest, "result file hash mismatch")
    entries = {entry["clip_id"]: entry for entry in result["clips"]}
    required(set(entries) == {c["clip_id"] for c in config["clips"]}, "result clip set mismatch")
    arms, reports = {}, []
    for clip in config["clips"]:
        cid, entry = clip["clip_id"], entries[clip["clip_id"]]
        required(entry["source_sha256"] == clip["sha256"], "source digest differs")
        pair = {tag: read_arm(result_dir, entry["files"][tag], clip, tag) for tag in ("A", "B")}
        for key in ("frame_id", "timestamp_us", "pts", "rgb_sha256", "rgb_shape"):
            required(exact(pair["A"]["decoded"][key], pair["B"]["decoded"][key]), "A/B decoded metadata differs: " + key)
        required(pair["A"]["meta"]["time_base"] == pair["B"]["meta"]["time_base"], "A/B time base differs")
        metric = paired_metrics(pair["A"]["selected"], pair["B"]["selected"])
        compare_reported_metrics(metric, entry["metrics"])
        if cid == config["clips"][0]["clip_id"]:
            repeated = read_arm(result_dir, entry["files"]["A_repeat"], clip, "A_repeat")
            compare_observations_exact(pair["A"]["obs"], repeated["obs"])
            for key in pair["A"]["decoded"]:
                required(exact(pair["A"]["decoded"][key], repeated["decoded"][key]), "A repeat metadata differs")
        arms[cid] = pair
        reports.append({"clip_id": cid, "rows": len(pair["A"]["selected"]["frame_id"]),
                        "same_rgb_ids_pts": True, "metrics_match": True, "independent_metrics": metric})
    full = sum(c["expected_frames_from_prior"] for c in config["clips"])
    selected = sum(c["expected_selected_even_frames"] for c in config["clips"])
    extra = config["clips"][0]["expected_frames_from_prior"]
    expected = {"decoded_frames": 2 * full + extra, "landmarker_process_calls": full + selected + extra,
                "landmarker_constructed": 13, "landmarker_closed": 13, "decoder_opened": 13,
                "model_loads": 1, "tcn_forwards": 8, "camera_calls": 0, "training_runs": 0, "downloads": 0}
    required(result["execution_counts"]["actual"] == result["execution_counts"]["expected"] == expected,
             "runtime counters differ from fixed arithmetic")
    scored = independent_scores(config, result, arms) if score else {"status": "NOT_RUN", "model_loads": 0, "forwards": 0}
    for name, digest in hashes.items():
        required(sha256_file(safe_result_file(result_dir, name)) == digest, "result mutated during verification")
    for filename, digest in config["pins"].items():
        required(sha256_file(filename) == digest, "input mutated during verification")
    return {"status": "PASS", "config_sha256": sha256_file(config_path),
            "result_sha256": sha256_file(result_dir / "RESULT.json"), "pin_count": len(config["pins"]),
            "full_A_repeat_exact": True, "clips": reports, "execution_counts": expected,
            "independent_model": scored, "video_decodes": 0, "mediapipe_calls": 0,
            "camera_calls": 0, "training_runs": 0, "input_and_result_files_unchanged": True}


def _synthetic_observation(count=6):
    mask = np.ones((count, 75), dtype=bool)
    return {"xy_px": np.ones((count, 75, 2), dtype=np.float32),
            "timestamp_us": np.arange(count, dtype=np.int64) * 33333,
            "frame_id": np.arange(count, dtype=np.int64),
            "observed": mask.copy(), "usable": mask.copy(),
            "conf": np.ones((count, 75), dtype=np.float32),
            "hand_score": np.full((count, 2), np.nan, dtype=np.float32)}


def self_test():
    passed = []
    a = _synthetic_observation()
    validate_observation(a)
    compare_observations_exact(a, {key: value.copy() for key, value in a.items()})
    passed.append("NaN fields exact without treating NaN as unequal")
    b = {key: value.copy() for key, value in a.items()}
    b["hand_score"][0, 0] = 1
    assert not exact(a["hand_score"], b["hand_score"])
    passed.append("different NaN locations rejected")
    selected = select_even(a)
    assert selected["frame_id"].tolist() == [0, 2, 4]
    assert selected["timestamp_us"].tolist() == [0, 66666, 133332]
    passed.append("even original IDs without append-last or timestamp regeneration")
    assert window_indices(a, 33333, 99999).tolist() == [1, 2]
    passed.append("half-open fixed window")
    m = np.ones((6, 21), dtype=bool)
    m[1, 2] = False
    m[2:4] = False
    h = hand_summary(m)
    assert h["all21_absent_frames"] == 2 and h["longest_all21_absent_run"] == 2
    assert h["any_usable_frames"] == 4 and h["all21_usable_frames"] == 3
    passed.append("whole-hand absent differs from partly missing")
    n = mask_summary(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))
    assert n == {"total": 4, "same": 2, "both_true": 1, "both_false": 1,
                 "a_only": 1, "b_only": 1, "a_true": 2, "b_true": 2}
    passed.append("mask changes count both directions")
    b = {key: value.copy() for key, value in a.items()}
    b["xy_px"] += np.array([3, 4], dtype=np.float32)
    p = paired_metrics(a, b)
    assert all(row["coordinate"]["mean_px"] == 5 for row in p.values())
    passed.append("coordinate distance is Euclidean px over common usable points")
    b["usable"][:] = False
    p = paired_metrics(a, b)
    assert p["all"]["coordinate"]["mean_px"] is None
    assert p["all"]["coordinate"]["common_points"] == 0
    passed.append("zero common points reported N/A")
    return {"status": "PASS", "checks": passed, "model_loads": 0, "forwards": 0,
            "video_decodes": 0, "mediapipe_calls": 0, "camera_calls": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--score", action="store_true", help="explicitly allow eight independent forwards after parent approval")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        if args.result or args.score or args.config or args.output:
            parser.error("--self-test cannot be combined with actual-result arguments")
        print(json.dumps(self_test(), indent=2))
    else:
        if not args.config or not args.result:
            parser.error("--config and --result are required; no implicit live paths")
        report = audit_result(args.config, args.result, args.score)
        encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output:
            destination = args.output.resolve()
            if destination.parent != args.config.resolve().parent:
                parser.error("independent output must be a new file in this unit, outside raw result")
            with destination.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(encoded)
        print(json.dumps({"status": report["status"], "clips": len(report["clips"]),
                          "independent_forwards": report["independent_model"]["forwards"],
                          "max_abs_logit_difference": report["independent_model"].get("max_abs_logit_difference")},
                         ensure_ascii=False, indent=2))
