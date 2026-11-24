"""One-time IMAGE-mode observations on the 3,359 external, publicly labelled crops.

This is not video tracking, a correction to the operating hand assignment, or
ground-truth annotation. Source classes/people never enter the detector.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

U = Path(__file__).resolve().parent
ROOT = U.parents[3]
DATA = U / "data"
INVENTORY = DATA / "DEEPHAND_INVENTORY.json"
MODEL = ROOT / "Tools/models/hand_landmarker.task"
PROTOCOL = DATA / "PUBLIC_JOINTS_PROTOCOL.json"
SOURCE = DATA / "PUBLIC_JOINTS_SOURCE.jsonl"
OUTPUT = DATA / "PUBLIC_JOINTS.npz"
META = DATA / "PUBLIC_JOINTS_META.json"
STATUS = DATA / "PUBLIC_JOINTS_STATUS.json"
LOCK = DATA / "PUBLIC_JOINTS_RUNNING.json"
FIRST_ERROR = DATA / "PUBLIC_JOINTS_FIRST_ERROR.json"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now().astimezone().isoformat()


def write_json(path, value):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def protected_hashes():
    frozen = json.loads((U / "FROZEN_PREPARATION.json").read_text(encoding="utf-8"))
    result = {}
    for path, expected in frozen["protected"].items():
        actual = sha(Path(path))
        if expected != actual:
            raise RuntimeError(f"Protected source changed: {path}")
        result[path] = actual
    return result


def prepare():
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    rows = inventory["rows"]
    if len(rows) != 3359 or len({r["id"] for r in rows}) != 3359:
        raise ValueError("Expected the frozen 3359 unique public image IDs")
    image_root = (DATA / "deephand_images").resolve()
    for i, row in enumerate(rows):
        image_path = Path(row["image"]).resolve()
        if image_path.parent != image_root or image_path.suffix.lower() != ".png":
            raise ValueError("Input outside approved public image folder")
        if sha(image_path) != row["image_sha256"]:
            raise ValueError(f"Source hash mismatch at {i}: {image_path}")
    pins = protected_hashes()
    protocol = {
        "schema": "wis_external_public_image_joints_v1",
        "image_count": 3359,
        "inventory_sha256": sha(INVENTORY),
        "ordered_id_sha256": hashlib.sha256("\n".join(r["id"] for r in rows).encode()).hexdigest(),
        "source_model": str(MODEL),
        "model_sha256": sha(MODEL),
        "running_mode": "IMAGE",
        "num_hands": 2,
        "min_hand_detection_confidence": 0.5,
        "min_hand_presence_confidence": 0.5,
        "min_tracking_confidence": 0.5,
        "tracking_used": False,
        "input": "full original RGB hand crop; preserve original 92x132 aspect and pixels",
        "selection": "minimum Euclidean distance of mean palm points[0,5,9,13,17] xy to crop center[0.5,0.5]; stable candidate-index tie break",
        "identity_uncertain": "flag only when two center distances differ by <=0.02 normalized crop units; never exclude/change selected candidate based on flag",
        "selected_output": "raw normalized crop21x3; all point masks true for a finite detected hand, zeros and all-false for no detection",
        "confidence": "handedness category score only; not per-landmark visibility/accuracy confidence",
        "all_candidates_retained": True,
        "no_semantic_input": True,
        "no_class_label_input": True,
        "no_handedness_correction": True,
        "denominator": "all3359 including no-hand observations; no score-based selection or sample exclusion",
        "protected_before": pins,
        "code_sha256": sha(Path(__file__)),
    }
    if PROTOCOL.exists():
        old = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        if old != protocol:
            raise RuntimeError("Frozen extraction protocol/input changed; do not overwrite")
    else:
        write_json(PROTOCOL, protocol)
    return rows, protocol


def read_saved(rows):
    saved = []
    if SOURCE.exists():
        for line_no, line in enumerate(SOURCE.read_text(encoding="utf-8").splitlines()):
            try:
                item = json.loads(line)
            except Exception as e:
                raise RuntimeError(f"Incomplete checkpoint line {line_no + 1}; preserve before repair") from e
            if len(saved) >= len(rows):
                raise RuntimeError("Checkpoint contains extra rows")
            row = rows[len(saved)]
            if item["image_id"] != row["id"] or item["image_sha256"] != row["image_sha256"]:
                raise RuntimeError("Checkpoint source ordering/hash mismatch")
            saved.append(item)
    return saved


def extract(rows, saved):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    with vision.HandLandmarker.create_from_options(options) as detector, SOURCE.open("a", encoding="utf-8", newline="\n") as out:
        for index in range(len(saved), len(rows)):
            row = rows[index]
            with Image.open(row["image"]) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if list(rgb.shape[:2]) != [row["height"], row["width"]]:
                raise ValueError("Image dimensions changed")
            t0 = time.perf_counter()
            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
            elapsed = time.perf_counter() - t0
            candidates = []
            if len(result.hand_landmarks) > 2:
                raise RuntimeError("Unexpected candidate count")
            for j, landmarks in enumerate(result.hand_landmarks):
                xyz = np.asarray([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
                if xyz.shape != (21, 3) or not np.isfinite(xyz).all():
                    raise ValueError("Non-finite or wrong-shape detected hand; preserved failure")
                center = xyz[[0, 5, 9, 13, 17], :2].mean(axis=0)
                categories = [{"index": c.index, "score": float(c.score), "category_name": c.category_name, "display_name": c.display_name} for c in result.handedness[j]]
                candidates.append({"raw_xyz": xyz.tolist(), "palm_xy": center.tolist(), "center_distance": float(np.linalg.norm(center - 0.5)), "handedness": categories})
            order = sorted(range(len(candidates)), key=lambda j: (candidates[j]["center_distance"], j))
            selected = order[0] if order else -1
            gap = candidates[order[1]]["center_distance"] - candidates[order[0]]["center_distance"] if len(order) == 2 else None
            record = {
                "index": index, "image_id": row["id"], "image": row["image"],
                "image_sha256": row["image_sha256"], "image_size_wh": [row["width"], row["height"]],
                "candidate_count": len(candidates), "candidates": candidates,
                "selected_candidate": selected, "center_distance_gap": gap,
                "identity_uncertain": gap is not None and gap <= 0.02,
                "inference_seconds": elapsed,
            }
            out.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            out.flush()
            saved.append(record)
            if (index + 1) % 100 == 0 or index + 1 == len(rows):
                write_json(STATUS, {"state": "EXTRACTING", "completed": index + 1, "total": len(rows), "pid": os.getpid(), "at": now()})
                print(json.dumps({"completed": index + 1, "total": len(rows)}), flush=True)
    return saved, mp.__version__


def pack(rows, saved, protocol, version):
    n = len(rows)
    xyz = np.zeros((n, 21, 3), np.float32)
    mask = np.zeros((n, 21), bool)
    candidates = np.zeros((n, 2, 21, 3), np.float32)
    candidate_mask = np.zeros((n, 2, 21), bool)
    distances = np.zeros((n, 2), np.float32)
    handedness = np.full((n, 2), "", dtype="<U8")
    handedness_score = np.zeros((n, 2), np.float32)
    count = np.array([r["candidate_count"] for r in saved], np.int8)
    selected = np.array([r["selected_candidate"] for r in saved], np.int8)
    uncertain = np.array([r["identity_uncertain"] for r in saved], bool)
    for i, record in enumerate(saved):
        for j, c in enumerate(record["candidates"]):
            candidates[i, j] = c["raw_xyz"]
            candidate_mask[i, j] = True
            distances[i, j] = c["center_distance"]
            if c["handedness"]:
                handedness[i, j] = c["handedness"][0]["category_name"] or ""
                handedness_score[i, j] = c["handedness"][0]["score"]
        if selected[i] >= 0:
            xyz[i] = candidates[i, selected[i]]
            mask[i] = True
    arrays = dict(
        schema=np.array("wis_public_image_joints_v1"), image_ids=np.array([r["id"] for r in rows]),
        raw_xyz=xyz, mask=mask, candidate_raw_xyz=candidates, candidate_mask=candidate_mask,
        candidate_count=count, selected_candidate=selected, identity_uncertain=uncertain,
        candidate_center_distance=distances, candidate_handedness=handedness,
        candidate_handedness_confidence=handedness_score,
        image_size_wh=np.asarray([[r["width"], r["height"]] for r in rows], np.int16),
        inference_seconds=np.asarray([r["inference_seconds"] for r in saved], np.float64),
    )
    if OUTPUT.exists():
        raise RuntimeError("Output exists without completed META; preserve before recovery")
    tmp = OUTPUT.with_name(OUTPUT.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, OUTPUT)
    with np.load(OUTPUT, allow_pickle=False) as check:
        for name, value in arrays.items():
            if not np.array_equal(check[name], value):
                raise RuntimeError(f"Saved array mismatch {name}")
    pins_after = protected_hashes()
    assert pins_after == protocol["protected_before"]
    assert len(saved) == n and np.isfinite(xyz).all() and not xyz[~mask].any()
    assert np.array_equal(mask.any(axis=1), count > 0)
    write_json(META, {
        "status": "COMPLETE_VERIFIED_EXTERNAL_STATIC_IMAGE_OBSERVATIONS",
        "at": now(), "rows": n, "no_hand": int((count == 0).sum()),
        "one_candidate": int((count == 1).sum()), "two_candidates": int((count == 2).sum()),
        "identity_uncertain": int(uncertain.sum()), "all_rows_preserved": True,
        "source_jsonl_sha256": sha(SOURCE), "npz_sha256": sha(OUTPUT),
        "protocol_sha256": sha(PROTOCOL), "inventory_sha256": sha(INVENTORY),
        "model_sha256": sha(MODEL), "python": sys.version, "platform": platform.platform(),
        "mediapipe": version, "numpy": np.__version__, "selection_frozen_before_detection": True,
        "per_landmark_confidence_available": False, "video_tracking": False,
        "new_operating_boundary_or_left_right_correction": False,
        "source_classes_used_by_detector": False,
        "latency_seconds": {"sum": float(arrays["inference_seconds"].sum()), "p50": float(np.median(arrays["inference_seconds"])), "p95": float(np.percentile(arrays["inference_seconds"], 95))},
        "protected_before": protocol["protected_before"], "protected_after": pins_after,
        "limitations": ["Hand crops are external source data, not full-body webcam videos.", "Class labels are separate manual targets; these joint outputs are automatic observations.", "No-hand rows remain in denominator with zero coordinates and false masks.", "Ambiguity flag does not remove or replace any candidate.", "Handedness confidence is not landmark visibility or correctness."],
    })
    write_json(STATUS, {"state": "COMPLETE", "completed": n, "total": n, "at": now()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    rows, protocol = prepare()
    if args.prepare_only:
        print(json.dumps({"status": "PREPARED", "rows": len(rows), "protocol": str(PROTOCOL)}))
        return
    if META.exists():
        meta = json.loads(META.read_text(encoding="utf-8"))
        assert sha(OUTPUT) == meta["npz_sha256"] and sha(SOURCE) == meta["source_jsonl_sha256"]
        print(json.dumps({"status": "REUSED_COMPLETE_NO_NEW_INFERENCE", "rows": meta["rows"]}))
        return
    with LOCK.open("x", encoding="utf-8") as lock:
        json.dump({"pid": os.getpid(), "at": now(), "script": str(Path(__file__))}, lock)
    try:
        saved = read_saved(rows)
        saved, version = extract(rows, saved)
        pack(rows, saved, protocol, version)
    except Exception:
        error = {"at": now(), "traceback": traceback.format_exc(), "pid": os.getpid()}
        if not FIRST_ERROR.exists():
            write_json(FIRST_ERROR, error)
        write_json(STATUS, {"state": "ERROR", **error})
        raise
    finally:
        if LOCK.exists() and json.loads(LOCK.read_text(encoding="utf-8"))["pid"] == os.getpid():
            LOCK.unlink()


if __name__ == "__main__":
    main()
