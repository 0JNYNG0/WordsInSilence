"""Frozen RGB descriptors from body-wrist crops; no labels or hand targets.

Only consumes the verified B0 observation adapter and original public RGB.
It must be run after the parent freezes this pilot's protocol. No network use.
The hand-keypoint target, raw/native hand stream and Gloss never place crops.
"""
from pathlib import Path
from datetime import datetime
import argparse
import hashlib
import importlib.util
import json
import time
import numpy as np

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent / "word100_temporal_public_bundle_20261025"
WEIGHTS = HERE.parents[1] / "assets/rgb_encoder/mobilenet_v3_small-047dcff4.pth"
WEIGHT_SHA = "047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def adapter():
    spec = importlib.util.spec_from_file_location("public_masked_adapter", BUNDLE / "load_inputs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def crop_plan(observation):
    """Use only body landmarks 11/12 and 15/16 and fixed ROI metadata.

    The whole-clip shoulder median is allowed in this offline pilot. No test
    scoring or target hand coordinate enters this deterministic transform.
    Missing shoulder scale or wrist evidence selects the fixed upper-body ROI.
    """
    xy = observation["xy_px"][:, :33]
    usable = observation["usable"][:, :33]
    confidence_available = observation["conf_available"][:, :33]
    shoulders = (usable[:, 11] & usable[:, 12]
                 & confidence_available[:, 11] & confidence_available[:, 12])
    widths = np.linalg.norm(xy[:, 11] - xy[:, 12], axis=1)
    shoulders &= np.isfinite(widths) & (widths >= 1.0)
    diameter = float(2.0 * np.median(widths[shoulders])) if shoulders.any() else None
    roi = np.asarray(observation["roi_box"], dtype=np.float64)
    assert roi.shape == (4,) and roi[2] > roi[0] and roi[3] > roi[1]
    boxes = np.broadcast_to(roi, (len(xy), 2, 4)).copy()
    valid = np.zeros((len(xy), 2), dtype=bool)
    if diameter is not None:
        for side, wrist in ((0, 15), (1, 16)):
            eligible = usable[:, wrist] & confidence_available[:, wrist] & np.isfinite(xy[:, wrist]).all(axis=1)
            for frame in np.flatnonzero(eligible):
                x, y = xy[frame, wrist]
                half = diameter / 2.0
                boxes[frame, side] = (x - half, y - half, x + half, y + half)
            valid[:, side] = eligible
    return {"crop_box": boxes.astype(np.float32), "crop_valid": valid,
            "crop_fallback": ~valid, "crop_side_px": np.float32(diameter or 0.0)}


def prepare_patch(rgb, box):
    """Square body-wrist patch with black outside-image padding, to 224px.

    Fallback is the fixed ROI rectangle resized to 224 square. The output is
    original colour RGB, ImageNet mean/std, no stochastic colour transforms.
    """
    import cv2
    x0, y0 = np.floor(box[:2]).astype(int)
    x1, y1 = np.ceil(box[2:]).astype(int)
    assert x1 > x0 and y1 > y0
    patch = np.zeros((y1-y0, x1-x0, 3), dtype=np.uint8)
    height, width = rgb.shape[:2]
    a, b, c, d = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
    if a < c and b < d:
        patch[b-y0:d-y0, a-x0:c-x0] = rgb[b:d, a:c]
    patch = cv2.resize(patch, (224, 224), interpolation=cv2.INTER_LINEAR)
    value = patch.astype(np.float32) / 255.0
    value = (value - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    return np.ascontiguousarray(value.transpose(2, 0, 1))


def verify_crop_independence(obs):
    original = crop_plan(obs)
    altered = {k: np.array(v, copy=True) for k, v in obs.items()}
    for key in ("xy_px", "usable", "observed", "conf", "conf_available", "point_origin"):
        if key in altered:
            altered[key][:, 33:] = 0
    altered["hand_score"][:] = 0
    altered["hand_score_available"][:] = False
    removed = crop_plan(altered)
    for key in original:
        assert np.array_equal(original[key], removed[key]), ("hand target crop leakage", key)
    return original


def run(protocol, readiness, batch_size=32, threads=4):
    assert Path(protocol).resolve().parent == HERE and Path(protocol).is_file(), "Local frozen protocol required"
    contract = read(protocol)
    assert isinstance(contract, dict) and contract, "Empty protocol"
    assert Path(readiness).resolve().parent == HERE and Path(readiness).is_file(), "Visible-evidence readiness required"
    gate = read(readiness)
    assert gate.get("ready_for_feature_extraction") is True, "Visible-evidence gate did not authorize features"
    assert not (HERE / "RGB_INDEX.json").exists(), "Do not overwrite completed features"
    assert sha(WEIGHTS) == WEIGHT_SHA, "Local official weight changed"
    manifest_path = BUNDLE / "MANIFEST.json"
    manifest_sha = sha(manifest_path)
    rows = read(manifest_path)
    assert len(rows) == 61
    assert all(r["classifier_prior_role"] == "train_candidate" and r["signer_id"] in range(1, 10) for r in rows)
    # Importing the local adapter does not import detector/model inference.
    loader = adapter()
    import torch
    from torchvision.models import mobilenet_v3_small
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    model = mobilenet_v3_small(weights=None)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True), strict=True)
    model.eval().requires_grad_(False)
    encoder = torch.nn.Sequential(model.features, model.avgpool, torch.nn.Flatten()).eval()
    assert not any(p.requires_grad for p in encoder.parameters())
    output = HERE / "features"
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    records = []
    images_processed = 0
    for ordinal, row in enumerate(rows):
        assert sha(row["arrays"]) == row["arrays_sha256"]
        assert sha(row["rgb_path"]) == row["rgb_sha256"]
        obs = loader.load_observations(ordinal, "B0")
        plan = verify_crop_independence(obs)
        features = np.empty((row["frames"], 2, 576), dtype=np.float32)
        batch, slots = [], []

        def flush():
            nonlocal images_processed
            if not batch:
                return
            with torch.inference_mode():
                values = encoder(torch.from_numpy(np.stack(batch))).cpu().numpy()
            assert values.shape == (len(batch), 576) and np.isfinite(values).all()
            for (frame, side), value in zip(slots, values):
                features[frame, side] = value
            images_processed += len(batch)
            batch.clear()
            slots.clear()

        decoded = 0
        for frame, timestamp, rgb in loader.iter_rgb_frames(ordinal):
            assert frame == decoded and timestamp == int(obs["timestamp_us"][frame])
            for side in (0, 1):
                batch.append(prepare_patch(rgb, plan["crop_box"][frame, side]))
                slots.append((frame, side))
                if len(batch) == batch_size:
                    flush()
            decoded += 1
        flush()
        assert decoded == row["frames"] and np.isfinite(features).all()
        dest = output / (row["clip_id"] + ".npz")
        np.savez_compressed(dest, features=features, frame_id=obs["frame_id"], timestamp_us=obs["timestamp_us"], **plan)
        records.append({"clip_id": row["clip_id"], "path": str(dest), "sha256": sha(dest),
                        "frames": decoded, "shape": list(features.shape), "crop_valid": int(plan["crop_valid"].sum()),
                        "crop_fallback": int(plan["crop_fallback"].sum()), "crop_side_px": float(plan["crop_side_px"]),
                        "source_arrays_sha256": row["arrays_sha256"], "source_rgb_sha256": row["rgb_sha256"]})
        print(json.dumps({"feature_progress": len(records), "total": len(rows), "images": images_processed,
                          "elapsed_seconds": round(time.perf_counter()-started, 2)}), flush=True)
    assert sha(manifest_path) == manifest_sha and sha(WEIGHTS) == WEIGHT_SHA
    (HERE / "RGB_INDEX.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    result = {"status": "FROZEN_RGB_FEATURES_PREPARED_NOT_TRAINED", "at": datetime.now().astimezone().isoformat(),
              "clips": len(records), "frames": sum(r["frames"] for r in records), "feature_images": images_processed,
              "feature_dim_per_side": 576, "elapsed_seconds": time.perf_counter()-started,
              "protocol_sha256": sha(protocol), "visible_readiness_sha256": sha(readiness), "script_sha256": sha(__file__), "source_manifest_sha256": manifest_sha,
              "loader_sha256": sha(BUNDLE / "load_inputs.py"), "weight_sha256": WEIGHT_SHA,
              "torch": torch.__version__, "cpu_threads": threads, "batch_size": batch_size,
              "crop_rule": "BODY pose wrists 15/16; square side twice clip median usable shoulder distance; fixed source ROI fallback",
              "crop_hand_target_independence_checks": len(records), "class_or_gloss_in_input": False,
              "geometry_note": "Original full-source RGB replay; not physical-camera outside-view simulation",
              "new_training": 0, "user_recording_access": 0}
    (HERE / "RGB_FEATURE_RESULT.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--readiness", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    assert 1 <= args.batch_size <= 64 and 1 <= args.threads <= 8
    run(args.protocol, args.readiness, args.batch_size, args.threads)
