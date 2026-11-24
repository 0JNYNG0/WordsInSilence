"""Independent audit only: run after root confirms final freeze and saved evaluation.

Reuses the existing STGCNEncoder with its reference-einsum switch. Does not import
the experiment runner, its preprocessing functions, or any training code.
"""
import argparse
import datetime
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PLAN_SHA = "210bb69eaf320ce95791341728fcbfd9a04d3b1f9f326c967c9dbbb0f919bf77"
ARMS = ("head_only", "full_encoder")
ATOL, RTOL = 1e-4, 1e-5
SELECT = {
    "native67": [0, 16, 15, 5, 2, 6, 3, 25, 29, 30, 33, 34, 37, 38, 41,
                 42, 45, 46, 50, 51, 54, 55, 58, 59, 62, 63, 66],
    "mp75": [0, 2, 5, 11, 12, 13, 14, 33, 37, 38, 41, 42, 45, 46, 49,
             50, 53, 54, 58, 59, 62, 63, 66, 67, 70, 71, 74],
}


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    return sha_bytes(Path(path).read_bytes())


def pinned_bytes(path, expected):
    raw = Path(path).read_bytes()
    require(sha_bytes(raw) == expected, "file SHA mismatch: " + str(path))
    return raw


def npz_bytes(raw):
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def array_digest(array):
    a = np.ascontiguousarray(array)
    return sha_bytes(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        t = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(t.dtype).encode())
        digest.update(str(tuple(t.shape)).encode())
        digest.update(t.numpy().tobytes())
    return digest.hexdigest()


def independently_normalize(row):
    """Metadata-fixed interval/mask; direct per-frame normalization from raw XY."""
    archive = npz_bytes(pinned_bytes(row["path"], row["source_sha256"]))
    layout = row["layout"]
    if layout == "native67":
        xy_key, mask_key, time_key = "xy_native", "coordinate_present", "timestamp_nominal_us"
        n_points, left, right, shoulder = 67, (25, 46), (46, 67), (5, 2)
    else:
        require(layout == "mp75", "unknown layout")
        xy_key, mask_key, time_key = "xy_px", "usable", "timestamp_us"
        n_points, left, right, shoulder = 75, (33, 54), (54, 75), (11, 12)
    xy, valid, timestamp = archive[xy_key], archive[mask_key], archive[time_key]
    require(xy.ndim == 3 and xy.shape[1:] == (n_points, 2), "raw shape")
    require(valid.shape == xy.shape[:2] and timestamp.shape == (len(xy),), "mask/time shape")
    require(valid.dtype == np.bool_ or np.isin(valid, (0, 1)).all(), "mask values")
    lo, hi = row["start"], row["end_exclusive"]
    require(0 <= lo < hi <= len(xy), "fixed interval out of bounds")
    xy = xy[lo:hi].astype(np.float64)
    valid = valid[lo:hi].astype(bool, copy=True)
    timestamp = timestamp[lo:hi]
    require(len(xy) == row["frames"], "fixed frame count changed")
    require(np.isfinite(xy[valid]).all(), "nonfinite raw mask-true coordinate")
    require(np.isfinite(timestamp).all() and (np.diff(timestamp) > 0).all(), "invalid timestamp")
    if row["mask_hand"] is not None:
        require(row["mask_hand"] in ("left", "right"), "mask side")
        first, last = left if row["mask_hand"] == "left" else right
        valid[:, first:last] = False

    ls, rs = shoulder
    shoulder_valid = valid[:, ls] & valid[:, rs]
    require(shoulder_valid.any(), "no valid shoulder pair")
    delta = xy[shoulder_valid, ls] - xy[shoulder_valid, rs]
    scale = float(np.median(np.sqrt(delta[:, 0] ** 2 + delta[:, 1] ** 2)))
    require(np.isfinite(scale) and scale > 0, "invalid shoulder median scale")
    points = SELECT[layout]
    normalized = np.zeros((len(xy), 27, 2), np.float64)
    for frame in range(len(xy)):
        if not shoulder_valid[frame]:
            continue
        center = (xy[frame, ls] + xy[frame, rs]) * 0.5
        for dest, source in enumerate(points):
            if valid[frame, source]:
                normalized[frame, dest] = (xy[frame, source] - center) / scale
    x = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None], dtype=np.float32)
    require(np.isfinite(x).all(), "nonfinite reconstructed input")
    return x, scale, int((~shoulder_valid).sum())


def audit(args, report):
    run = Path(args.run_dir).resolve()
    plan = json.loads(pinned_bytes(HERE / "POINT_CHECK_PLAN.json", PLAN_SHA))
    require(plan["sample_count"] == 10 and len(plan["samples"]) == 10, "sample plan changed")
    require(plan["model_arms"] == list(ARMS), "planned arms")
    require(plan["verification_after_final_freeze"]["logits_atol"] == ATOL
            and plan["verification_after_final_freeze"]["logits_rtol"] == RTOL, "tolerance changed")
    manifest = json.loads(pinned_bytes(plan["manifest_path"], plan["manifest_sha256"]))
    frozen_path, scores_path = run / "FROZEN_MODELS.json", run / "eval_scores.npz"
    frozen = json.loads(pinned_bytes(frozen_path, args.frozen_sha))
    scores = npz_bytes(pinned_bytes(scores_path, args.scores_sha))
    metrics = json.loads((run / "eval_metrics.json").read_text(encoding="utf-8"))
    require(frozen.get("complete") is True and set(frozen["arms"]) == set(ARMS), "both final models not frozen")
    require(metrics["frozen_models_sha256"] == args.frozen_sha
            and metrics["eval_scores_sha256"] == args.scores_sha, "evaluation is not bound to frozen models")
    classes = manifest["classes"]
    require(len(classes) == 93 and classes == frozen["classes"] == scores["classes"].tolist(), "class identity")
    prior_head_pin = frozen["identity"]["pins"]["head"]
    parent_head = npz_bytes(pinned_bytes(prior_head_pin["path"], prior_head_pin["sha256"]))
    core_pin = frozen["identity"]["pins"]["prior_core"]
    pinned_bytes(core_pin["path"], core_pin["sha256"])
    require(core_pin["sha256"] == frozen["identity"]["code_sha256"]["stgcn_core"], "core identity")
    spec = importlib.util.spec_from_file_location("existing_point_check_stgcn_core", core_pin["path"])
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    torch.set_num_threads(2)
    torch.manual_seed(20261005)
    torch.use_deterministic_algorithms(True)

    positions = {int(value): i for i, value in enumerate(scores["row_index"])}
    require(len(positions) == 855, "saved evaluation row identity")
    models, heads, checksums = {}, {}, {
        str(frozen_path): args.frozen_sha,
        str(scores_path): args.scores_sha,
        str(HERE / "POINT_CHECK_PLAN.json"): PLAN_SHA,
        str(core_pin["path"]): core_pin["sha256"],
    }
    for arm in ARMS:
        record = frozen["arms"][arm]
        for kind in ("encoder_state", "head", "training"):
            entry = record[kind]
            require(Path(entry["file"]).name == entry["file"], "unexpected frozen artifact path")
            checksums[str(run / entry["file"])] = entry["sha256"]
            pinned_bytes(run / entry["file"], entry["sha256"])
        state = npz_bytes(pinned_bytes(run / record["encoder_state"]["file"], record["encoder_state"]["sha256"]))
        model = core.STGCNEncoder(state["A"])
        expected = model.state_dict()
        require(set(state) == set(expected) and len(state) == 170, "encoder full state key contract")
        loaded = {}
        for name, ref in expected.items():
            value = state[name]
            require(value.shape == tuple(ref.shape) and value.dtype == ref.numpy().dtype, "state shape/dtype: " + name)
            require(np.isfinite(value).all(), "nonfinite state: " + name)
            loaded[name] = torch.from_numpy(value.copy())
        model.load_state_dict(loaded, strict=True)
        model.set_reference_einsum(True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        require(all(block.gcn.reference_einsum for block in model.st_gcn_networks), "einsum not enabled")
        require(state_digest(model) == record["state_hash"], "loaded encoder state digest")
        h = npz_bytes(pinned_bytes(run / record["head"]["file"], record["head"]["sha256"]))
        for name, shape in (("W", (93, 256)), ("b", (93,)), ("mean", (256,)), ("std", (256,))):
            require(h[name].shape == shape and h[name].dtype == np.float64
                    and np.isfinite(h[name]).all(), "head shape/dtype/finite: " + name)
        require((h["std"] >= 1e-6).all(), "head std floor")
        require(h["classes"].tolist() == classes, "head class order")
        require(np.array_equal(h["mean"], parent_head["mean"])
                and np.array_equal(h["std"], parent_head["std"]), "parent fixed standardization changed")
        require(scores["scores_" + arm].shape == (855, 93), "saved score shape")
        models[arm], heads[arm] = model, h

    report.update({"frozen_models_sha256": args.frozen_sha, "eval_scores_sha256": args.scores_sha,
                   "core_sha256": core_pin["sha256"], "arms": list(ARMS), "atol": ATOL, "rtol": RTOL,
                   "planned_samples": 10, "model_forwards": 0, "checks": []})
    for sample in plan["samples"]:
        ri = sample["global_row_index"]
        require(ri in positions, "planned row absent")
        pos, row = positions[ri], manifest["rows"][ri]
        for name in ("group", "id", "condition", "label", "start", "end_exclusive", "layout", "path", "mask_hand", "exposed"):
            require(sample[name] == row[name], "planned manifest identity: " + name)
        for name, key in (("group", "groups"), ("id", "ids"), ("condition", "conditions"), ("layout", "layouts"), ("exposed", "exposed")):
            require(row[name] == scores[key][pos], "saved metadata mismatch: " + key)
        require(int(scores["y"][pos]) == row["y"] and sample["source_sha256"] == row["sha256"], "label/source identity")
        x, scale, zero_frames = independently_normalize(sample)
        digest = array_digest(x)
        require(digest == str(scores["input_array_sha256"][pos]), "independent input hash mismatch: " + sample["sample_key"])
        require(sample["source_sha256"] == str(scores["source_sha256"][pos]), "source digest differs")
        require(x.shape[2] == int(scores["frame_count"][pos]) and zero_frames == int(scores["zero_frame_count"][pos]), "input diagnostic counts")
        require(scale == float(scores["scale"][pos]), "independent normalization scale not exact")
        checksums[sample["path"]] = sample["source_sha256"]
        for arm in ARMS:
            with torch.no_grad():
                feature = models[arm](torch.from_numpy(x)).detach().cpu().numpy()
            report["model_forwards"] += 1
            require(feature.shape == (1, 256) and np.isfinite(feature).all(), "reference feature shape/finite")
            h = heads[arm]
            logits = ((feature.astype(np.float64) - h["mean"]) / h["std"]) @ h["W"].T + h["b"]
            logits = logits[0]
            saved = scores["scores_" + arm][pos].astype(np.float64)
            require(np.isfinite(logits).all() and np.isfinite(saved).all(), "nonfinite comparison logits")
            error = np.abs(logits - saved)
            close = bool(np.allclose(logits, saved, atol=ATOL, rtol=RTOL))
            own_top, saved_top = int(np.argmax(logits)), int(np.argmax(saved))
            report["checks"].append({"sample_key": sample["sample_key"], "global_row_index": ri,
                                     "id": row["id"], "condition": row["condition"], "arm": arm,
                                     "input_sha256_exact": True, "frame_count": int(x.shape[2]), "scale_exact": True,
                                     "max_abs_logit_difference": float(error.max()),
                                     "max_tolerance_ratio": float(np.max(error / (ATOL + RTOL * np.abs(saved)))),
                                     "all_logits_within_tolerance": close, "own_top1_index": own_top,
                                     "saved_top1_index": saved_top, "top1_equal": own_top == saved_top})
    for arm in ARMS:
        require(state_digest(models[arm]) == frozen["arms"][arm]["state_hash"], "encoder state changed in point verification")
    for path, expected in checksums.items():
        require(sha_file(path) == expected, "protected artifact changed: " + path)
    require(report["model_forwards"] == 20 and len(report["checks"]) == 20, "expected exactly20 forwards")
    require(all(x["all_logits_within_tolerance"] and x["top1_equal"] for x in report["checks"]), "numerical point check failed; preserve all observed differences")
    report["protected_hashes_unchanged"] = checksums
    report["status"] = "PASS"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--frozen-sha", required=True)
    parser.add_argument("--scores-sha", required=True)
    args = parser.parse_args()
    output = HERE / "POINT_CHECK.json"
    require(not output.exists(), "preserve previous point-check output")
    report = {"status": "STARTED", "plan_sha256": PLAN_SHA, "script_sha256": sha_file(__file__),
              "started_kst": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
              "scope": "Independent numerical spot audit. No training or candidate implementation edits.",
              "preprocessing_source": "direct independent calculation in this audit; no transfer_runner functions",
              "model_source": "existing pinned core STGCNEncoder, reference_einsum=True"}
    try:
        audit(args, report)
    except BaseException as exc:
        report["status"] = "FAIL"
        report["error"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        report["finished_kst"] = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
        with output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        print(json.dumps({"status": report["status"], "output": str(output),
                          "model_forwards": report.get("model_forwards", 0)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
