"""Independent array verifier for the frozen mechanism diagnostic.

This is not the experimental runner. It never modifies observations or builds
deployment features. Only synthetic self-tests are exposed before result schema
completion; any later model verification must be explicitly gated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


FIELDS = ("xy_px", "timestamp_us", "frame_id", "observed", "usable", "conf", "hand_score")
VARIANTS = ("common_A", "common_B", "pose_A_hands_B", "pose_B_hands_A")
HAND_SLICES = (slice(33, 54), slice(54, 75))


def must(condition, explanation):
    if not condition:
        raise AssertionError(explanation)


def exact(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(np.array_equal(a, b, equal_nan=True)) if a.dtype.kind in "fc" else bool(np.array_equal(a, b))


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def typed_hash(array):
    a = np.ascontiguousarray(array)
    header = a.dtype.str + "|" + "x".join(map(str, a.shape)) + "|"
    return hashlib.sha256(header.encode("ascii") + a.tobytes()).hexdigest()


def validate(obs):
    n = len(obs["frame_id"])
    requirements = {
        "xy_px": ((n, 75, 2), "float32"), "timestamp_us": ((n,), "int64"),
        "frame_id": ((n,), "int64"), "observed": ((n, 75), "bool"),
        "usable": ((n, 75), "bool"), "conf": ((n, 75), "float32"),
        "hand_score": ((n, 2), "float32")}
    for key, (shape, dtype) in requirements.items():
        must(obs[key].shape == shape and obs[key].dtype == np.dtype(dtype), f"bad {key} schema")
    must(n > 0 and (np.diff(obs["frame_id"]) > 0).all(), "empty/unordered original IDs")
    must((np.diff(obs["timestamp_us"]) > 0).all(), "nonincreasing original timestamps")
    must(not np.any(obs["usable"] & ~obs["observed"]), "usable without observation")
    must(np.isfinite(obs["xy_px"][obs["usable"]]).all(), "nonfinite usable xy")
    must(not np.isfinite(obs["xy_px"][~obs["observed"]]).any(), "finite unobserved xy")


def expected_variant(a, b, name):
    validate(a)
    validate(b)
    must(name in VARIANTS, "unknown variant")
    for key in ("frame_id", "timestamp_us"):
        must(exact(a[key], b[key]), "paired IDs/timestamps differ")
    if name.startswith("common_"):
        source = a if name == "common_A" else b
        shared = np.logical_and(a["usable"], b["usable"])
        out = {key: source[key].copy() for key in FIELDS}
        out["xy_px"][~shared] = np.nan
        out["conf"][~shared] = np.nan
        out["observed"] = shared.copy()
        out["usable"] = shared.copy()
        for side, sl in enumerate(HAND_SLICES):
            out["hand_score"][~shared[:, sl].any(axis=1), side] = np.nan
    else:
        pose, hand = (a, b) if name == "pose_A_hands_B" else (b, a)
        out = {key: pose[key].copy() for key in FIELDS}
        for key in ("xy_px", "observed", "usable", "conf"):
            out[key][:, 33:] = hand[key][:, 33:]
        out["hand_score"] = hand["hand_score"].copy()
    validate(out)
    return out


def equal_observations(a, b):
    validate(a)
    validate(b)
    checks = {key: exact(a[key], b[key]) for key in FIELDS}
    must(all(checks.values()), {"unequal_observation": checks})
    return checks


def select_window(obs, start, end):
    must(start < end, "invalid frozen window")
    keep = (obs["frame_id"] % 2 == 0) & (obs["timestamp_us"] >= start) & (obs["timestamp_us"] < end)
    must(keep.any(), "empty frozen window")
    return {key: obs[key][keep].copy() for key in FIELDS}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def pinned_inputs(config_path):
    config_path = Path(config_path).resolve()
    cfg = read_json(config_path)
    must(cfg["schema"] == "wis_capture_cadence_mechanism_config_v1", "config schema")
    must(tuple(cfg["variants"]) == VARIANTS and len(cfg["clips"]) == 4, "frozen conditions")
    must(len(cfg["pins"]) == 27, "pin count")
    for path, expected in cfg["pins"].items():
        must(file_hash(path) == expected, "changed pin: " + path)
    source_path = next(e["path"] for e in cfg["pin_inventory"] if e["role"] == "source_result")
    source = read_json(source_path)
    must(source["status"] == "COMPLETE" and file_hash(source_path) == cfg["source_result_sha256"], "source result")
    classes = cfg["model"]["classes"]
    must(len(classes) == len(set(classes)) == 93 and source["scoring"]["classes"] == classes, "class order")
    source_windows = {w["clip_id"]: w for w in source["scoring"]["windows"]}
    pairs = []
    for clip in cfg["clips"]:
        saved = source_windows[clip["clip_id"]]
        for key in ("clip_id", "label", "start_us", "end_us_exclusive", "window_frame_ids", "window_timestamp_us"):
            must(saved[key] == clip[key], "window differs: " + key)
        pair, decoded = {}, {}
        for arm in ("A", "B"):
            spec = clip["arms"][arm]
            meta = read_json(spec["meta_json"])
            must(meta["npz_sha256"] == cfg["pins"][spec["raw_npz"]], "meta npz hash")
            must(meta["clip_id"] == clip["clip_id"] and meta["tag"] == arm, "meta identity")
            must(meta["arm"] == ("A_full_then_even" if arm == "A" else "B_even_rgb_only"), "meta arm")
            with np.load(spec["raw_npz"], allow_pickle=False) as z:
                raw = {key: z["obs_" + key].copy() for key in FIELDS}
                decoded[arm] = {key: z["decoded_" + key].copy() for key in
                                ("frame_id", "timestamp_us", "pts", "rgb_sha256", "rgb_shape")}
                selected_ids = z["selected_frame_id"].copy()
            validate(raw)
            if arm == "B":
                must(np.all(raw["frame_id"] % 2 == 0), "B contains odd IDs")
            must(exact(raw["frame_id"][raw["frame_id"] % 2 == 0], selected_ids), "source selected IDs")
            selected = select_window(raw, clip["start_us"], clip["end_us_exclusive"])
            must(selected["frame_id"].tolist() == clip["window_frame_ids"], "window IDs")
            must(selected["timestamp_us"].tolist() == clip["window_timestamp_us"], "window timestamps")
            must(len(selected["frame_id"]) == clip["expected_window_rows"], "window rows")
            pair[arm] = selected
            for key, source_key in (("baseline_input", "input"), ("baseline_score", "score"),
                                    ("baseline_logits93", "logits93")):
                must(spec[key] == saved[arm][source_key], "baseline source/config mismatch")
        for key in decoded["A"]:
            must(exact(decoded["A"][key], decoded["B"][key]), "selected image source differs: " + key)
        for key in ("frame_id", "timestamp_us"):
            must(exact(pair["A"][key], pair["B"][key]), "unaligned pair")
        pairs.append(pair)
    return cfg, pairs


def official_builder(cfg):
    roles = {e["role"]: Path(e["path"]) for e in cfg["pin_inventory"]}
    pkg = roles["pilot_data"].parent
    sys.path.insert(0, str(pkg.parent))
    from pilot import data, web_input
    for module, role in ((data, "pilot_data"), (web_input, "pilot_mapping")):
        must(Path(module.__file__).resolve() == roles[role].resolve(), "official module path")
        must(file_hash(module.__file__) == cfg["pins"][str(roles[role])], "official module hash")

    def build(obs):
        xy, present = web_input.to_native_like(obs["xy_px"], obs["usable"])
        ch = data.build_channels(xy, present, obs["timestamp_us"])
        x, pad, y = data.collate([dict(ch, label=0)])
        must(np.all(y == 0) and np.all(pad == 1) and np.isfinite(x).all(), "model input invalid")
        return dict(ch, x=x, pad=pad)
    return build


def normalized_number(x):
    return float(x) if np.isfinite(x) else None


def input_metadata(inp):
    return {"x_sha256": typed_hash(inp["x"]), "x_shape": list(inp["x"].shape),
            "x_dtype": inp["x"].dtype.name, "frames": inp["frames"],
            "body_scale": normalized_number(inp["body_scale"]),
            "hand_scales": {k: normalized_number(v) for k, v in inp["hand_scales"].items()},
            "body_valid_sum": float(inp["body_valid"].sum()),
            "hand_valid_sum": float(inp["hand_valid"].sum())}


def feature_hashes(inp):
    return {"pad_mask_sha256": typed_hash(inp["pad"]),
            "body_valid_sha256": typed_hash(inp["body_valid"]),
            "hand_valid_sha256": typed_hash(inp["hand_valid"]),
            "x_valid_channels_sha256": typed_hash(inp["x"][..., 186:279])}


def input_precheck(config_path):
    cfg, pairs = pinned_inputs(config_path)
    build = official_builder(cfg)
    references, inputs, checks = [], [], []
    for ci, (clip, pair) in enumerate(zip(cfg["clips"], pairs)):
        out, feats = {}, {}
        for arm in ("A", "B"):
            feats[arm] = build(pair[arm])
            must(input_metadata(feats[arm]) == clip["arms"][arm]["baseline_input"], "original input differs")
        for name in VARIANTS:
            out[name] = expected_variant(pair["A"], pair["B"], name)
            feats[name] = build(out[name])
        for key in ("body_valid", "hand_valid"):
            must(exact(feats["common_A"][key], feats["common_B"][key]), "common final validity differs")
        must(exact(feats["common_A"]["x"][..., 186:279], feats["common_B"]["x"][..., 186:279]),
             "common model valid channels differ")
        references.append(out)
        inputs.append(feats)
        checks.append({"clip_index": ci, "clip_id": clip["clip_id"], "rows": clip["expected_window_rows"],
                       "original_A_B_inputs_exact": True, "common_final_validity_exact": True,
                       "input_hashes": {k: typed_hash(v["x"]) for k, v in feats.items()}})
    must(not any(m in sys.modules for m in ("torch", "av", "cv2", "mediapipe")), "precheck imported model/media")
    report = {"status": "PASS_INPUT_PRECHECK", "config_sha256": file_hash(config_path),
              "pinned_files": len(cfg["pins"]), "baseline_input_checks": 8, "new_input_builds": 16,
              "model_loads": 0, "forwards": 0, "video_decodes": 0, "mediapipe_calls": 0,
              "camera_calls": 0, "windows": checks}
    return cfg, pairs, references, inputs, report


def expected_score(logits, classes, label):
    x = np.asarray(logits, dtype=np.float64)
    must(x.shape == (93,) and np.isfinite(x).all(), "invalid logits")
    ti = classes.index(label)
    ordered = np.lexsort((np.arange(len(x)), -x))
    top = int(ordered[0])
    rival = next(int(i) for i in ordered if i != ti)
    return {"true_label": label, "top1": classes[top], "top1_index": top,
            "correct": top == ti, "true_rank": int(np.flatnonzero(ordered == ti)[0]) + 1,
            "best_other": classes[rival], "true_vs_best_other_margin": float(x[ti] - x[rival])}


def compare_input_blocks(x, baseline):
    ranges = {"body_xy": (0, 102), "hand_xy": (102, 186), "body_valid": (186, 237),
              "hand_valid": (237, 279), "dt": (279, 280)}
    result = {}
    for key, (lo, hi) in ranges.items():
        delta = x[..., lo:hi].astype(np.float64) - baseline[..., lo:hi].astype(np.float64)
        result[key] = {"changed_values": int(np.count_nonzero(delta)),
                       "max_abs_diff": float(np.max(np.abs(delta)))}
    return result


def result_check(config_path, result_dir, score=False):
    cfg, pairs, expected, inputs, report = input_precheck(config_path)
    result_dir = Path(result_dir).resolve()
    must(result_dir == Path(cfg["output_dir"]).resolve(), "not frozen output directory")
    result = read_json(result_dir / "RESULT.json")
    must(result["schema"] == "wis_capture_cadence_mechanism_result_v1", "result schema")
    must(result["status"] == "RUN_COMPLETE_NOT_INDEPENDENTLY_VERIFIED", "run not complete")
    must(result["config"]["sha256"] == file_hash(config_path), "config digest differs")
    must(result["pins"] == cfg["pins"] and result["source_result_sha256"] == cfg["source_result_sha256"], "source pin set")
    must(result["classes"] == cfg["model"]["classes"], "result class order")
    must(result["execution_counts"] == {"baseline_input_rechecks": 8, "baseline_forwards": 0,
         "new_input_builds": 16, "new_forward_calls": 16}, "execution count mismatch")
    hashes = read_json(result_dir / "FILE_HASHES.json")
    actual = {p.name for p in result_dir.iterdir() if p.is_file()} - {"FILE_HASHES.json"}
    must(set(hashes) == actual, "unmanifested or missing result artifact")
    for name, entry in hashes.items():
        must(Path(name).name == name, "nonlocal artifact name")
        p = result_dir / name
        must(file_hash(p) == entry["sha256"] and p.stat().st_size == entry["bytes"], "artifact hash: " + name)
    auth = result["resume_authorization"]
    must(auth["content"] == read_json(auth["path"]) and auth["sha256"] == file_hash(auth["path"]), "resume record")
    must(auth["content"]["prepared_config_sha256"] == file_hash(config_path), "resume config linkage")
    must(auth["content"]["supersedes_deadline_only"] == cfg["deadline_kst"], "resume deadline linkage")
    receipt = result["test_receipt"]
    must(receipt["content"] == read_json(receipt["path"]) and receipt["sha256"] == file_hash(receipt["path"]), "test receipt")
    for item in ("runner", "test"):
        must(file_hash(receipt[item]) == receipt["content"][item + "_sha256"], "runtime code changed")
    must(result["post_audit"]["state_equals_checkpoint"] and not result["post_audit"]["model_training"], "post model audit")
    must(result["runtime"]["forbidden_modules_loaded"] == [], "forbidden runtime imports")
    must(len(result["windows"]) == 4 and result["score_record_count"] == 24, "result record count")
    records, summaries = {}, []
    for ci, (clip, window, pair, variants, feats) in enumerate(zip(cfg["clips"], result["windows"], pairs, expected, inputs)):
        for key in ("clip_id", "label", "start_us", "end_us_exclusive", "window_frame_ids", "window_timestamp_us"):
            must(window[key] == clip[key], "record window: " + key)
        must(window["clip_index"] == ci and window["rows"] == clip["expected_window_rows"], "record clip position")
        common = pair["A"]["usable"] & pair["B"]["usable"]
        must(window["common_mask_sha256"] == typed_hash(common) and window["common_mask_true"] == int(common.sum())
             and window["common_mask_size"] == common.size, "common mask metadata")
        recs = window["records"]
        must([r["condition"] for r in recs] == ["A", "B", *VARIANTS], "condition set/order")
        scores = {}
        for rec in recs:
            name = rec["condition"]
            must(rec["input"] == input_metadata(feats[name]), "saved input differs: " + name)
            must(rec["hashes"] == feature_hashes(feats[name]), "saved feature hashes: " + name)
            if name in ("A", "B"):
                must(rec["status"] == "REUSED" and rec["logits93"] == clip["arms"][name]["baseline_logits93"], "baseline reuse")
            else:
                must(rec["status"] == "INFERRED", "variant status")
                with np.load(result_dir / f"clip{ci}_{name}.npz", allow_pickle=False) as z:
                    equal_observations({key: z[key].copy() for key in FIELDS}, variants[name])
                    if name.startswith("common_"):
                        must(exact(z["common_mask"], common), "stored common mask differs")
                for arm in ("A", "B"):
                    must(rec["block_changes_vs_" + arm] == compare_input_blocks(feats[name]["x"], feats[arm]["x"]),
                         "input block deltas differ")
            expected_s = expected_score(rec["logits93"], result["classes"], clip["label"])
            must(rec["score"] == expected_s, "score aggregation differs")
            scores[name] = expected_s
            records[ci, name] = rec
        summaries.append({"clip_id": clip["clip_id"], "scores": scores, "all_inputs_and_arrays_exact": True})
    independent = {"status": "NOT_RUN", "model_loads": 0, "forwards": 0}
    if score:
        independent = model_check(cfg, inputs, records)
    for path, expected_hash in cfg["pins"].items():
        must(file_hash(path) == expected_hash, "pin changed during independent audit")
    for name, entry in hashes.items():
        must(file_hash(result_dir / name) == entry["sha256"], "result changed during audit")
    return {"status": "PASS" if score else "PASS_ARTIFACTS_NO_MODEL_REINFERENCE", "config_sha256": file_hash(config_path),
            "result_sha256": file_hash(result_dir / "RESULT.json"), "pinned_files": len(cfg["pins"]),
            "result_files_checked": len(hashes), "baseline_input_checks": 8, "new_input_checks": 16,
            "new_observation_npz_exact": 16, "model_verification": independent, "windows": summaries,
            "video_decodes": 0, "mediapipe_calls": 0, "camera_calls": 0,
            "scope": "Oracle input-sensitivity diagnostic, not an improvement or physical-cause proof."}


def model_check(cfg, inputs, records):
    """Only called with explicit --score after the parent authorizes sixteen forwards."""
    import torch
    from pilot import model_masked_v1
    role = next(e for e in cfg["pin_inventory"] if e["role"] == "pilot_model")
    must(Path(model_masked_v1.__file__).resolve() == Path(role["path"]).resolve(), "official model module path")
    must(file_hash(model_masked_v1.__file__) == role["sha256"], "official model module hash")
    MaskedSignTCN = model_masked_v1.MaskedSignTCN
    torch.set_num_threads(2)
    path = cfg["model"]["checkpoint_path"]
    must(file_hash(path) == cfg["model"]["checkpoint_sha256"], "checkpoint hash")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    must(list(payload["classes"]) == cfg["model"]["classes"] and payload["n_classes"] == 93, "checkpoint classes")
    model = MaskedSignTCN(width=payload["width"], dilations=tuple(payload["dilations"]),
                          kernel_size=payload["kernel_size"], dropout=payload["dropout"], n_classes=93)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    calls, differences = 0, []
    for ci, feats in enumerate(inputs):
        for name in VARIANTS:
            with torch.inference_mode():
                logits = model(torch.from_numpy(feats[name]["x"]), torch.from_numpy(feats[name]["pad"]))[0].numpy()
            calls += 1
            saved = np.asarray(records[ci, name]["logits93"], dtype=np.float64)
            error = float(np.max(np.abs(logits.astype(np.float64) - saved)))
            must(error == 0.0, f"model output differs: clip{ci}/{name}: {error}")
            differences.append(error)
    must(calls == 16, "independent forward count")
    must(all(torch.equal(v, before[k]) for k, v in model.state_dict().items()), "state changed")
    return {"status": "PASS", "model_loads": 1, "forwards": calls, "baseline_forwards": 0,
            "max_abs_logit_difference": max(differences), "all_state_unchanged": True}


def fake_observation(offset):
    n = 4
    xy = np.arange(n * 75 * 2, dtype=np.float32).reshape(n, 75, 2) + offset
    return {"xy_px": xy, "timestamp_us": np.arange(n, dtype=np.int64) * 33333,
            "frame_id": np.arange(n, dtype=np.int64),
            "observed": np.ones((n, 75), bool), "usable": np.ones((n, 75), bool),
            "conf": np.full((n, 75), offset + 0.125, np.float32),
            "hand_score": np.full((n, 2), offset + 0.75, np.float32)}


def self_test():
    a, b = fake_observation(0), fake_observation(100)
    b["usable"][0, 33:54] = b["observed"][0, 33:54] = False
    b["xy_px"][0, 33:54] = b["conf"][0, 33:54] = np.nan
    b["hand_score"][0, 0] = np.nan
    a["usable"][1, 74] = False  # observed but unusable; common view must discard it.
    before_a = {key: value.copy() for key, value in a.items()}
    before_b = {key: value.copy() for key, value in b.items()}
    variants = {name: expected_variant(a, b, name) for name in VARIANTS}
    shared = a["usable"] & b["usable"]
    checks = []
    for name, src in (("common_A", a), ("common_B", b)):
        v = variants[name]
        must(exact(v["xy_px"][shared], src["xy_px"][shared]), "common xy changed")
        must(exact(v["conf"][shared], src["conf"][shared]), "common conf changed")
        must(np.isnan(v["xy_px"][~shared]).all() and np.isnan(v["conf"][~shared]).all(), "excluded values")
        must(exact(v["observed"], shared) and exact(v["usable"], shared), "common mask")
        must(np.isnan(v["hand_score"][0, 0]) and v["hand_score"][0, 1] == src["hand_score"][0, 1], "hand scores")
        checks.append(name + " exact provenance and absent rules")
    for name, pose, hand in (("pose_A_hands_B", a, b), ("pose_B_hands_A", b, a)):
        v = variants[name]
        for key in ("xy_px", "observed", "usable", "conf"):
            must(exact(v[key][:, :33], pose[key][:, :33]), "pose donor")
            must(exact(v[key][:, 33:], hand[key][:, 33:]), "hand donor")
        must(exact(v["hand_score"], hand["hand_score"]), "hand score donor")
        checks.append(name + " donor blocks exact")
    for name, v in variants.items():
        for key in FIELDS:
            must(not np.shares_memory(v[key], a[key]) and not np.shares_memory(v[key], b[key]), "shared source memory")
        must(exact(v["frame_id"], a["frame_id"]) and exact(v["timestamp_us"], a["timestamp_us"]), "clock changed")
    checks.append("independent copies and original clocks")
    equal_observations(a, before_a)
    equal_observations(b, before_b)
    checks.append("original arrays unchanged")
    w = select_window(a, 33333, 99999)
    must(w["frame_id"].tolist() == [2] and w["timestamp_us"].tolist() == [66666], "window/even selector")
    checks.append("even original ID and half-open interval")
    altered = {key: v.copy() for key, v in variants["common_A"].items()}
    altered["xy_px"][0, 0, 0] += 1
    must(not exact(altered["xy_px"], variants["common_A"]["xy_px"]), "changed value not detected")
    must(exact(np.array([np.nan], np.float32), np.array([np.nan], np.float32)), "NaN equality")
    must(not exact(np.array([np.nan], np.float32), np.array([0], np.float32)), "NaN placement")
    checks.append("NaN and finite provenance comparison")
    return {"status": "PASS", "checks": checks, "forwards": 0, "model_loads": 0,
            "video_decodes": 0, "mediapipe_calls": 0, "camera_calls": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        report = self_test()
    elif args.config:
        if args.score and not args.result:
            parser.error("--score requires completed --result and parent execution authorization")
        report = result_check(args.config, args.result, args.score) if args.result else input_precheck(args.config)[-1]
    else:
        parser.error("provide --self-test or --config for model-free input verification")
    if args.output:
        with args.output.open("x", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
