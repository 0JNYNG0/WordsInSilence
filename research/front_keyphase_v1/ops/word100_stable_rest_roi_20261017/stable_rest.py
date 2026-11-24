"""Frozen research-only stable preparation ROI and causal resting-wrist guard.

Calibration is an offline preparation step using full-source observations from
the first second only. The guard sees only a pointwise ROI-masked observation
and the fixed preparation result; it never recovers cropped points, interpolates,
backdates a decision, or alters the detector's pose wrists.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
_LAB = Path(__file__).resolve().parents[2]
if str(_LAB / "src") not in sys.path:
    sys.path.insert(0, str(_LAB / "src"))

from wis_lab.core.observation import HAND_SLICES, Observation


SCHEMA = "stable_preparation_rest_wrist_v1"
FIRST_US = 1_000_000
DWELL_US = 300_000
MAX_GAP_US = 100_000
MIN_FRAMES = 8
STABILITY_RATIO = .05
BOTTOM_MARGIN_RATIO = .02
BOTTOM_BAND_RATIO = .03
ANCHOR_RADIUS = .08
RUN_MOVEMENT_RADIUS = .02
MIN_ARM_DROP_RATIO = .20
MAX_FOREARM_SLOPE = .5
_REQUIRED = (11, 12, 13, 14, 33, 54)
_STABLE = (11, 12, 33, 54)
_SIDES = (("left", 11, 13, 33), ("right", 12, 14, 54))


def _parameters() -> dict:
    return {
        "initial_horizon_us": FIRST_US, "minimum_duration_us": DWELL_US,
        "minimum_calibration_frames": MIN_FRAMES, "maximum_gap_us": MAX_GAP_US,
        "calibration_stability_shoulder_width_ratio": STABILITY_RATIO,
        "bottom_margin_shoulder_width_ratio": BOTTOM_MARGIN_RATIO,
        "bottom_band_shoulder_width_ratio": BOTTOM_BAND_RATIO,
        "anchor_radius_normalized": ANCHOR_RADIUS,
        "run_movement_radius_normalized": RUN_MOVEMENT_RADIUS,
        "minimum_arm_drop_shoulder_width_ratio": MIN_ARM_DROP_RATIO,
        "maximum_forearm_horizontal_to_vertical_ratio": MAX_FOREARM_SLOPE,
    }


def _arm_down(shoulder, elbow, wrist, scale):
    """Apply the same image-y-down geometry to scalar frames or frame arrays."""
    upper_drop = elbow[..., 1] - shoulder[..., 1]
    forearm_drop = wrist[..., 1] - elbow[..., 1]
    return ((upper_drop >= MIN_ARM_DROP_RATIO * scale)
            & (forearm_drop >= MIN_ARM_DROP_RATIO * scale)
            & (np.abs(wrist[..., 0] - elbow[..., 0])
               <= MAX_FOREARM_SLOPE * forearm_drop))


def calibrate_rest(obs: Observation, width: int, height: int) -> dict:
    """Choose the first causally confirmable stable preparation window.

    End timestamps are considered in order, then starts in order. Thus the
    earliest confirmation wins; ties prefer the earliest start. All frames in
    a window must qualify; missing frames cannot be skipped. Times are relative
    to the first source timestamp, not absolute PTS. No window may extend past
    the first second, and no fallback boundary is supplied on failure. Duplicate
    or decreasing timestamps are rejected by the Observation contract.
    """
    obs.validate()
    if isinstance(width, bool) or isinstance(height, bool):
        raise ValueError("source dimensions must be positive integers")
    if int(width) != width or int(height) != height or width <= 0 or height <= 0:
        raise ValueError("source dimensions must be positive integers")
    if "roi_mask" in obs.sidecar:
        raise ValueError("calibration requires full-source observations before apply_roi")
    base = {"schema": SCHEMA, "status": "NOT_CALIBRATED",
            "reason": "NO_QUALIFYING_INITIAL_WINDOW", "source_width": int(width),
            "source_height": int(height), "parameters": _parameters(),
            "selected_frame_indices": [], "selected_frame_ids": [],
            "selected_timestamp_us": [],
            "selection_order": "earliest_end_then_earliest_start",
            "input_scope": "offline_full_source_first_second_only"}
    if obs.n_frames == 0:
        base["reason"] = "EMPTY_OBSERVATION"
        return base

    first = int(obs.timestamp_us[0])
    limit = int(np.searchsorted(obs.timestamp_us, first + FIRST_US, side="right"))
    xy = obs.xy_px[:limit].astype(np.float64)
    valid = (obs.observed[:limit, _REQUIRED]
             & obs.usable[:limit, _REQUIRED]
             & np.isfinite(xy[:, _REQUIRED]).all(axis=-1)).all(axis=1)
    gaps = np.diff(obs.timestamp_us[:limit])

    for end in range(MIN_FRAMES - 1, limit):
        for start in range(end - MIN_FRAMES + 2):
            if int(obs.timestamp_us[end]) - int(obs.timestamp_us[start]) < DWELL_US:
                continue
            if not valid[start:end + 1].all() or (gaps[start:end] > MAX_GAP_US).any():
                continue
            window = xy[start:end + 1]
            scale = float(np.median(np.linalg.norm(window[:, 11] - window[:, 12], axis=1)))
            if not math.isfinite(scale) or scale <= 0:
                continue
            if any(not _arm_down(window[:, shoulder], window[:, elbow],
                                 window[:, wrist], scale).all()
                   for _, shoulder, elbow, wrist in _SIDES):
                continue
            median_points = np.median(window[:, _STABLE], axis=0)
            movement = np.linalg.norm(window[:, _STABLE] - median_points, axis=-1)
            if float(movement.max()) > STABILITY_RATIO * scale:
                continue
            wrist_medians = {name: np.median(window[:, wrist], axis=0)
                             for name, _, _, wrist in _SIDES}
            center = np.median((window[:, 11] + window[:, 12]) * .5, axis=0)
            bottom = math.floor(min(v[1] for v in wrist_medians.values())
                                - BOTTOM_MARGIN_RATIO * scale)
            if not 0 < bottom <= height:
                continue
            selected = list(range(start, end + 1))
            return {**base, "status": "CALIBRATED", "reason": "STABLE_ARMS_DOWN_WINDOW",
                    "selected_frame_indices": selected,
                    "selected_frame_ids": obs.frame_id[selected].tolist(),
                    "selected_timestamp_us": obs.timestamp_us[selected].tolist(),
                    "start_timestamp_us": int(obs.timestamp_us[start]),
                    "end_timestamp_us": int(obs.timestamp_us[end]),
                    "duration_us": int(obs.timestamp_us[end]) - int(obs.timestamp_us[start]),
                    "shoulder_width": scale, "median_shoulder_center": center.tolist(),
                    "wrist_medians_source_pixels": {k: v.tolist() for k, v in wrist_medians.items()},
                    "wrist_anchors_normalized": {k: ((v - center) / scale).tolist()
                                                 for k, v in wrist_medians.items()},
                    "maximum_stability_distance_pixels": float(movement.max()),
                    "bottom": bottom, "box_source_pixels": [0, 0, int(width), bottom]}
    return base


def _validate_masked(masked: Observation) -> np.ndarray:
    masked.validate()
    meta = masked.sidecar.get("roi_mask", {})
    if (meta.get("schema") != "source_pixel_pointwise_roi_v1"
            or meta.get("coordinate_space") != "source_pixels"):
        raise ValueError("rest guard requires the output of frozen apply_roi")
    box = np.asarray(meta.get("box_source_pixels"), dtype=np.float64)
    if (box.shape != (4,) or not np.isfinite(box).all()
            or box[2] <= box[0] or box[3] <= box[1]):
        raise ValueError("invalid ROI geometry")
    xy = masked.xy_px
    inside = (np.isfinite(xy).all(axis=-1) & (xy[..., 0] >= box[0])
              & (xy[..., 0] < box[2]) & (xy[..., 1] >= box[1]) & (xy[..., 1] < box[3]))
    if (masked.observed & ~inside).any():
        raise ValueError("observed point outside masked ROI")
    return box


def suppress_rest_fragments(masked: Observation, calibration: dict) -> tuple[Observation, dict]:
    """Causally suppress only confirmed, isolated hand-wrist points.

    Every qualification and run update reads the original masked input, never
    the partially suppressed output. Runs reset on absent evidence, fingers,
    geometry failure, a >100ms gap, or >.02 normalized displacement from the
    run's first wrist OR elbow. A displacement starts a fresh run now. A dwell
    of at least 300ms suppresses the current point only; earlier points stay.
    Duplicate or decreasing timestamps are rejected by Observation.validate,
    rather than interpreted as a new stream or repaired.
    """
    box = _validate_masked(masked)
    if calibration.get("schema") != SCHEMA:
        raise ValueError("unknown calibration schema")
    status = calibration.get("status")
    if status not in ("CALIBRATED", "NOT_CALIBRATED"):
        raise ValueError("unknown calibration status")
    scale_fixed = None
    anchors = {}
    if status == "CALIBRATED":
        expected_box = np.asarray(calibration["box_source_pixels"], dtype=np.float64)
        if expected_box.shape != (4,) or not np.array_equal(box, expected_box):
            raise ValueError("ROI differs from fixed calibration")
        scale_fixed = float(calibration["shoulder_width"])
        if not math.isfinite(scale_fixed) or scale_fixed <= 0:
            raise ValueError("invalid calibrated shoulder width")
        for name, _, _, _ in _SIDES:
            anchor = np.asarray(calibration["wrist_anchors_normalized"][name], dtype=np.float64)
            if anchor.shape != (2,) or not np.isfinite(anchor).all():
                raise ValueError("invalid calibrated wrist anchor")
            anchors[name] = anchor

    out = copy.deepcopy(masked)
    trace = {"schema": SCHEMA, "calibration_status": status, "parameters": _parameters(),
             "frame_policy": "original_frames_and_pts_preserved_no_backdating",
             "evidence_scope": "ROI_visible_current_and_history_only_plus_fixed_calibration",
             "normalization": "current_visible_shoulder_center_and_width",
             "band_scale": "fixed_calibrated_shoulder_width",
             "suppressed_frame_counts": {"left": 0, "right": 0}, "frames": []}
    runs = {"left": None, "right": None}

    for f in range(masked.n_frames):
        now = int(masked.timestamp_us[f])
        entry = {"frame_index": f, "frame_id": int(masked.frame_id[f]), "timestamp_us": now}
        for side, (name, shoulder, elbow, wrist) in enumerate(_SIDES):
            item = {"reason": "NOT_CALIBRATED", "suppressed": False,
                    "dwell_us": 0, "run_start_frame_index": None}
            entry[name] = item
            if status != "CALIBRATED":
                continue
            hand_slice = HAND_SLICES[side]
            reason = None
            if masked.observed[f, hand_slice.start + 1:hand_slice.stop].any():
                reason = "OBSERVED_FINGER_PROTECTS_HAND"
            elif not masked.observed[f, wrist]:
                reason = "WRIST_NOT_OBSERVED"
            elif not masked.usable[f, wrist] or not np.isfinite(masked.xy_px[f, wrist]).all():
                reason = "WRIST_NOT_USABLE_FINITE"
            else:
                required = (11, 12, elbow)
                evidence = (masked.observed[f, required] & masked.usable[f, required]
                            & np.isfinite(masked.xy_px[f, required]).all(axis=-1))
                if not evidence.all():
                    reason = "VISIBLE_BODY_EVIDENCE_MISSING"
            if reason:
                item["reason"] = reason
                runs[name] = None
                continue

            xy = masked.xy_px[f].astype(np.float64)
            scale = float(np.linalg.norm(xy[11] - xy[12]))
            if not math.isfinite(scale) or scale <= 0:
                item["reason"] = "CURRENT_SHOULDER_SCALE_INVALID"
                runs[name] = None
                continue
            center = (xy[11] + xy[12]) * .5
            wrist_rel = (xy[wrist] - center) / scale
            elbow_rel = (xy[elbow] - center) / scale
            distance = float(np.linalg.norm(wrist_rel - anchors[name]))
            item["anchor_distance_normalized"] = distance
            item["bottom_distance_pixels"] = float(box[3] - xy[wrist, 1])
            if not 0 < box[3] - xy[wrist, 1] <= BOTTOM_BAND_RATIO * scale_fixed:
                reason = "OUTSIDE_BOTTOM_BAND"
            elif distance > ANCHOR_RADIUS:
                reason = "OUTSIDE_REST_ANCHOR"
            elif not _arm_down(xy[shoulder], xy[elbow], xy[wrist], scale):
                reason = "ARM_NOT_DOWN"
            if reason:
                item["reason"] = reason
                runs[name] = None
                continue

            run = runs[name]
            reset_reason = None
            if run is not None and now - run["last_us"] > MAX_GAP_US:
                reset_reason = "TIME_GAP_RESET"
            elif run is not None and (
                    np.linalg.norm(wrist_rel - run["wrist"]) > RUN_MOVEMENT_RADIUS
                    or np.linalg.norm(elbow_rel - run["elbow"]) > RUN_MOVEMENT_RADIUS):
                reset_reason = "MOVEMENT_RESET"
            if run is None or reset_reason:
                run = {"first_us": now, "last_us": now, "first_index": f,
                       "wrist": wrist_rel.copy(), "elbow": elbow_rel.copy()}
                runs[name] = run
            run["last_us"] = now
            dwell = now - run["first_us"]
            item["dwell_us"] = dwell
            item["run_start_frame_index"] = run["first_index"]
            item["reason"] = reset_reason or "DWELL_PENDING"
            if dwell >= DWELL_US:
                item["reason"] = "CONFIRMED_REST_WRIST_SUPPRESSED"
                item["suppressed"] = True
                out.observed[f, wrist] = False
                out.usable[f, wrist] = False
                out.xy_px[f, wrist] = np.nan
                out.conf[f, wrist] = np.nan
                out.hand_score[f, side] = np.nan
                trace["suppressed_frame_counts"][name] += 1
        trace["frames"].append(entry)

    out.sidecar["rest_wrist_guard"] = {
        "schema": SCHEMA, "calibration_status": status,
        "suppressed_frame_counts": copy.deepcopy(trace["suppressed_frame_counts"]),
        "policy": "causal_hand_wrist_only_no_backdating_no_interpolation",
    }
    out.validate()
    return out, trace
