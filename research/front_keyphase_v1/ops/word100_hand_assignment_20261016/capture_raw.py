"""Immutable, TRAIN-only raw detector capture for hand-side replay.

Usage (only after root freezes SCOPE.json and PLAN.json):
    python capture_raw.py prepare
    python capture_raw.py run

This harness calls the existing decoder and original Landmarker unchanged.
The subclass records hand results before original side assignment and records
zero-hand frames as well. Each clip/mode gets newly constructed detector models;
the 24 conditions run sequentially in this dedicated Python process. No frame
is interpolated, omitted due to absence, resized, or sent to a classifier.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

UNIT = Path(__file__).resolve().parent
LAB = UNIT.parents[1]
ROOT = LAB.parents[1]
PRIOR = LAB / "ops/word100_upperbody_preflight_20261014"
CAPTURE = UNIT / "raw_capture"
CONTEXT = UNIT / "RAW_CAPTURE_CONTEXT.json"
sys.path.insert(0, str(LAB / "src"))

from wis_lab.capture.landmarker import Landmarker
from wis_lab.capture.video_source import extract
from wis_lab.core.observation import Observation, sha256_file


def read(path):
    return json.loads(Path(path).read_text("utf-8-sig"))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def check_pins(pins):
    for name, expected in pins.items():
        actual = sha256_file(name)
        if actual != expected:
            raise RuntimeError(f"Pinned input changed: {name}")


def log_event(**event):
    event.update(time_utc=utc_now(), pid=os.getpid())
    with (UNIT / "RAW_CAPTURE_PROGRESS.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)


def prepare():
    for filename in ("SCOPE.json", "PLAN.json"):
        if not (UNIT / filename).is_file():
            raise RuntimeError(f"Root must freeze {filename} before preparing capture")
    prior = read(PRIOR / "PLAN.json")
    check_pins(prior["pins"])
    items = prior["items"]
    assert len(items) == 12 and prior["conditions"] == ["native", "crop"]
    for item in items:
        assert item["signer_id"] == 1 and item["source_role"] == "train_candidate"
        assert sha256_file(item["video_path"]) == item["source_sha256"]
    files = [Path(__file__), UNIT / "SCOPE.json", UNIT / "PLAN.json", PRIOR / "PLAN.json"]
    files += [Path(p) for p in prior["pins"]]
    for item in items:
        files.append(Path(item["video_path"]))
        for mode in prior["conditions"]:
            files += [PRIOR / item["clip_id"] / f"{mode}.npz",
                      PRIOR / item["clip_id"] / f"{mode}.sidecar.json"]
    pins = {str(p.resolve()): sha256_file(p) for p in files}
    context = {
        "status": "FROZEN_BEFORE_RAW_CAPTURE", "created_utc": utc_now(),
        "pins": pins, "prior_plan_sha256": sha256_file(PRIOR / "PLAN.json"),
        "scope": "Exactly 12 prior TRAIN REAL01 clips x native/crop; same fixed boxes",
        "conditions": ["native", "crop"], "items": items,
        "output_root": str(CAPTURE),
        "isolation": "Dedicated capture process; new Landmarker/hand/pose per clip and mode; no shared VIDEO detector state",
        "schema": "wis_raw_hand_detections_v1",
        "label_mapping": {"Left": 0, "Right": 1, "unknown_or_padding": -1},
        "coordinates": "raw hand pixel XY and handedness scores retain float64; baseline observation and pose retain original float32; no coordinate repair",
        "resume": "Exact context/code/source hashes plus final SUCCESS.json and all output hashes; partial output is not silently overwritten",
        "training": 0, "classifier_inference": 0, "user_data": 0,
    }
    write_new(CONTEXT, context)
    print(f"Frozen raw capture context with {len(pins)} pinned files", flush=True)


class RawCaptureLandmarker(Landmarker):
    """Record raw hands without changing the superclass assignment or masks."""

    def __init__(self, contract):
        super().__init__(contract)
        self.raw_rows = []
        self._raw_frame = None
        self.instance_token = f"{os.getpid()}:{time.time_ns()}"

    def process(self, rgb, timestamp_us, out, row):
        h, w = rgb.shape[:2]
        self._raw_frame = {
            "raw_count": np.int16(0),
            "hand_xy_px": np.full((2, 21, 2), np.nan, np.float64),
            "raw_label": np.full(2, -1, np.int8),
            "raw_label_name": np.full(2, "", dtype="<U32"),
            "raw_score": np.full(2, np.nan, np.float64),
            "baseline_side": np.full(2, -1, np.int8),
            "baseline_score": np.full(2, np.nan, np.float64),
            "width": np.int32(w), "height": np.int32(h),
        }
        super().process(rgb, timestamp_us, out, row)
        capture = self._raw_frame
        capture.update(
            timestamp_us=np.int64(timestamp_us),
            frame_id=np.int64(out["frame_id"][row]),
            detector_timestamp_ms=np.int64(self._last_ms),
            pose_xy_px=out["xy_px"][row, :33].copy(),
            pose_conf=out["conf"][row, :33].copy(),
            pose_observed=out["observed"][row, :33].copy(),
            pose_usable=out["usable"][row, :33].copy(),
        )
        capture["hand_finite"] = np.isfinite(capture["hand_xy_px"]).all(-1)
        capture["raw_high_score"] = np.isfinite(capture["raw_score"]) & (
            capture["raw_score"] >= self.hand_score_thr)
        self.raw_rows.append(capture)
        self._raw_frame = None

    def _assign_sides(self, hand_res, pose_xy, pose_usable, w, h):
        capture = self._raw_frame
        assert capture is not None
        count = len(hand_res.hand_landmarks)
        assert 0 < count <= 2
        capture["raw_count"] = np.int16(count)
        for index, points in enumerate(hand_res.hand_landmarks):
            assert len(points) == 21
            category = hand_res.handedness[index][0]
            label = category.category_name
            capture["raw_label"][index] = {"Left": 0, "Right": 1}.get(label, -1)
            capture["raw_label_name"][index] = label
            capture["raw_score"][index] = float(category.score)
            capture["hand_xy_px"][index] = [(point.x * w, point.y * h) for point in points]
        assigned = super()._assign_sides(hand_res, pose_xy, pose_usable, w, h)
        for index, (side, score) in assigned.items():
            capture["baseline_side"][index] = -1 if side is None else side
            capture["baseline_score"][index] = score
        return assigned


class CropProxy:
    def __init__(self, base, box):
        self.base, self.box = base, box

    def __getattr__(self, key):
        return getattr(self.base, key)

    def process(self, rgb, timestamp_us, out, row):
        x0, y0, x1, y1 = self.box
        self.base.process(rgb[y0:y1, x0:x1], timestamp_us, out, row)


def equal_array(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if np.issubdtype(a.dtype, np.floating):
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def verify_capture(raw, ob, contract):
    """Assert recorded raw/assignment snapshots exactly reconstruct baseline hands."""
    assert len(raw["raw_count"]) == ob.n_frames
    for name in ("timestamp_us", "frame_id"):
        assert equal_array(raw[name], getattr(ob, name))
    assert (np.diff(raw["detector_timestamp_ms"]) > 0).all()
    for source, target in (("pose_xy_px", "xy_px"), ("pose_conf", "conf"),
                           ("pose_observed", "observed"), ("pose_usable", "usable")):
        assert equal_array(raw[source], getattr(ob, target)[:, :33])
    xy = np.full((ob.n_frames, 42, 2), np.nan, np.float32)
    observed = np.zeros((ob.n_frames, 42), bool)
    usable = observed.copy()
    scores = np.full((ob.n_frames, 2), np.nan, np.float32)
    margin = contract["usable_rule"]["bounds_margin_ratio"]
    score_min = contract["usable_rule"]["hand_score_threshold"]
    for frame in range(ob.n_frames):
        sides = []
        for index in range(int(raw["raw_count"][frame])):
            side = int(raw["baseline_side"][frame, index])
            if side < 0:
                continue
            assert side not in sides
            sides.append(side)
            # The original process casts pixel XY to float32 before _in_bounds.
            points = raw["hand_xy_px"][frame, index].astype(np.float32)
            score = raw["baseline_score"][frame, index]
            sl = slice(side * 21, (side + 1) * 21)
            xy[frame, sl] = points
            observed[frame, sl] = True
            w, h = int(raw["width"][frame]), int(raw["height"][frame])
            mx, my = float(margin) * w, float(margin) * h
            bounded = np.isfinite(points).all(1) & (points[:, 0] >= -mx) & (
                points[:, 0] <= w + mx) & (points[:, 1] >= -my) & (
                points[:, 1] <= h + my)
            usable[frame, sl] = bounded & (score >= score_min)
            scores[frame, side] = score
    for a, b in ((xy, ob.xy_px[:, 33:]), (observed, ob.observed[:, 33:]),
                 (usable, ob.usable[:, 33:]), (scores, ob.hand_score)):
        assert equal_array(a, b), "Recorded raw/assignment cannot exactly reconstruct original baseline"


def run():
    context = read(CONTEXT)
    context_hash = sha256_file(CONTEXT)
    check_pins(context["pins"])
    contract = read(LAB / "config/contract.json")
    if (UNIT / "RAW_CAPTURE_RESULT.json").exists():
        raise RuntimeError("Raw capture already completed; reuse its saved outputs")
    CAPTURE.mkdir(exist_ok=True)
    rows = []
    start = time.perf_counter()
    log_event(event="capture_start", total_conditions=24, context_sha256=context_hash,
              isolation=context["isolation"])
    for item in context["items"]:
        cid = item["clip_id"]
        pair = {}
        for mode in context["conditions"]:
            directory = CAPTURE / cid / mode
            success_path = directory / "SUCCESS.json"
            if success_path.is_file():
                result = read(success_path)
                assert result["context_sha256"] == context_hash
                check_pins(result["output_sha256"])
                pair[mode] = Observation.load(str(directory / "baseline.npz"))
                rows.append(result)
                log_event(event="reuse_exact_success", clip_id=cid, mode=mode, completed=len(rows))
                continue
            if directory.exists():
                raise RuntimeError(f"Partial output preserved; verified recovery required: {directory}")
            directory.mkdir(parents=True)
            assert sha256_file(item["video_path"]) == item["source_sha256"]
            log_event(event="item_start", clip_id=cid, label=item["label"], mode=mode)
            item_start = time.perf_counter()
            with RawCaptureLandmarker(contract) as landmarker:
                assert landmarker._last_ms == -1 and not landmarker.raw_rows
                token = landmarker.instance_token
                interface = CropProxy(landmarker, item["box"]) if mode == "crop" else landmarker
                baseline = extract(item["video_path"], interface, cid)
                assert landmarker.raw_rows
                raw = {key: np.stack([row[key] for row in landmarker.raw_rows])
                       for key in landmarker.raw_rows[0]}
            if mode == "crop":
                x0, y0, x1, y1 = item["box"]
                baseline.sidecar.update(width=x1-x0, height=y1-y0)
            verify_capture(raw, baseline, contract)
            baseline.sidecar.update(raw_capture_context_sha256=context_hash,
                raw_capture_mode=mode, source_bbox_px=item["box"] if mode == "crop"
                else [0, 0, *item["original_size"]], original_size=item["original_size"],
                detector_instance_token=token, detector_state="fresh at beginning of clip/mode")
            baseline.sidecar["quality"] = baseline.quality_summary()
            baseline.save(str(directory / "baseline.npz"))
            raw_path = directory / "raw.npz"
            with raw_path.open("xb") as handle:
                np.savez_compressed(handle, schema=np.array("wis_raw_hand_detections_v1"), **raw)
            prior = Observation.load(str(PRIOR / cid / f"{mode}.npz"))
            prior_equal = {key: equal_array(getattr(baseline, key), getattr(prior, key))
                for key in ("xy_px", "observed", "usable", "conf", "hand_score", "timestamp_us", "frame_id")}
            sidecar = dict(schema="wis_raw_hand_detections_v1", clip_id=cid, label=item["label"],
                mode=mode, context_sha256=context_hash, raw_sha256=sha256_file(raw_path),
                source_path=item["video_path"], source_sha256=item["source_sha256"],
                box=baseline.sidecar["source_bbox_px"], frames=baseline.n_frames,
                width=baseline.sidecar["width"], height=baseline.sidecar["height"],
                label_mapping=context["label_mapping"],
                baseline_side="raw hand index to original anatomical slot; -1 means unassigned or no detection",
                score_note="MediaPipe raw handedness score, not per-landmark visibility or independent detection confidence",
                detector_instance_token=token, process_id=os.getpid(),
                original_baseline_reconstruction_exact=True, prior_observation_equal=prior_equal,
                time_axis="original decoded frame_id and PTS microseconds; detector integer ms also recorded",
                zero_hand_frames=int((raw["raw_count"] == 0).sum()),
                raw_detections=int(raw["raw_count"].sum()))
            write_new(directory / "raw.sidecar.json", sidecar)
            files = [directory / name for name in ("raw.npz", "raw.sidecar.json", "baseline.npz", "baseline.sidecar.json")]
            result = dict(status="COMPLETE", clip_id=cid, label=item["label"], mode=mode,
                context_sha256=context_hash, frames=baseline.n_frames,
                output_sha256={str(path): sha256_file(path) for path in files},
                seconds=time.perf_counter()-item_start, raw_detections=sidecar["raw_detections"],
                zero_hand_frames=sidecar["zero_hand_frames"],
                original_baseline_reconstruction_exact=True, prior_observation_equal=prior_equal,
                detector_instance_token=token, process_id=os.getpid())
            write_new(success_path, result)
            rows.append(result)
            pair[mode] = baseline
            log_event(event="item_complete", clip_id=cid, mode=mode, frames=baseline.n_frames,
                      completed=len(rows), total_conditions=24, seconds=result["seconds"])
        for key in ("timestamp_us", "frame_id"):
            assert equal_array(getattr(pair["native"], key), getattr(pair["crop"], key))
    check_pins(context["pins"])
    assert sha256_file(CONTEXT) == context_hash
    write_new(UNIT / "RAW_CAPTURE_RESULT.json", dict(status="COMPLETE_PENDING_INDEPENDENT_REVIEW",
        context_sha256=context_hash, conditions=len(rows), source_clips=12,
        seconds=time.perf_counter()-start, rows=rows, source_pins_unchanged=True,
        training=0, classifier_inference=0, production_changed=False))
    log_event(event="capture_complete", completed=24, seconds=time.perf_counter()-start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    arguments = parser.parse_args()
    try:
        prepare() if arguments.action == "prepare" else run()
    except Exception as error:
        failure = dict(time_utc=utc_now(), pid=os.getpid(), error=repr(error), traceback=traceback.format_exc())
        failure_path = UNIT / "RAW_CAPTURE_FIRST_FAILURE.json"
        if not failure_path.exists():
            write_new(failure_path, failure)
        raise
