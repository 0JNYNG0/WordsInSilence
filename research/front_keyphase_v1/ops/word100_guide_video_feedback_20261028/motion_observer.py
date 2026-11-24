"""Read-only, guide-blind measurements from saved public MediaPipe observations.

This is a geometric observation adapter, NOT a learned action or sign judge.
Source layout: Pose33 + anatomical left Hand21 + anatomical right Hand21.
Missing points remain missing. A finite/usable estimate is not verified truth.
No downloads, model calls, source writes, target labels or text are accepted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np

SCHEMA = "wis_guide_blind_motion_features_v1"
HAND_OFFSETS = {"L": 33, "R": 54}
ARM_IDS = {"L": (11, 13, 15), "R": (12, 14, 16)}
FINGER_ANGLES = {
    "thumb_mcp": (1, 2, 3), "thumb_ip": (2, 3, 4),
    "index_mcp": (0, 5, 6), "index_pip": (5, 6, 7), "index_dip": (6, 7, 8),
    "middle_mcp": (0, 9, 10), "middle_pip": (9, 10, 11), "middle_dip": (10, 11, 12),
    "ring_mcp": (0, 13, 14), "ring_pip": (13, 14, 15), "ring_dip": (14, 15, 16),
    "little_mcp": (0, 17, 18), "little_pip": (17, 18, 19), "little_dip": (18, 19, 20),
}


def read_observation(path: str | Path) -> dict:
    """Load numeric fields only, without pickle or side-effectful model imports."""
    required = ("xy_px", "observed", "usable", "timestamp_us", "frame_id")
    with np.load(Path(path), allow_pickle=False) as z:
        missing = set(required) - set(z.files)
        if missing:
            raise ValueError(f"Missing observation fields: {sorted(missing)}")
        out = {k: np.array(z[k], copy=True) for k in required}
    return out


def _validate(arrays: Mapping) -> tuple:
    xy = np.asarray(arrays["xy_px"], dtype=np.float64)
    observed = np.asarray(arrays["observed"], dtype=bool)
    usable = np.asarray(arrays["usable"], dtype=bool)
    ts = np.asarray(arrays["timestamp_us"], dtype=np.int64)
    frame = np.asarray(arrays["frame_id"], dtype=np.int64)
    if xy.ndim != 3 or xy.shape[1:] != (75, 2):
        raise ValueError("Expected T x 75 x 2 source-pixel coordinates")
    n = len(xy)
    if observed.shape != (n, 75) or usable.shape != (n, 75):
        raise ValueError("Observation masks must be T x 75")
    if ts.shape != (n,) or frame.shape != (n,) or n == 0:
        raise ValueError("Nonempty timestamps and frame IDs must match T")
    if np.any(np.diff(ts) <= 0) or np.any(np.diff(frame) <= 0):
        raise ValueError("Timestamps and original frame IDs must strictly increase")
    # The adapter deliberately refuses inferred-but-unobserved usable positions.
    valid = observed & usable & np.isfinite(xy).all(axis=2)
    return xy, valid, ts, frame


def _angle(xy: np.ndarray, valid: np.ndarray, ids: tuple[int, int, int]):
    a, b, c = ids
    u, v = xy[:, a] - xy[:, b], xy[:, c] - xy[:, b]
    norms = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
    good = valid[:, [a, b, c]].all(axis=1) & np.isfinite(norms) & (norms > 1e-12)
    angle = np.full(len(xy), np.nan)
    cosine = np.zeros(len(xy))
    np.divide(np.einsum("ij,ij->i", u, v), norms, out=cosine, where=good)
    angle[good] = np.arccos(np.clip(cosine[good], -1, 1)) / np.pi
    return angle, good


def extract_features(arrays: Mapping, *, max_motion_gap_seconds: float = 0.15) -> dict:
    """Return full-length x/mask; zero is a value, never a missing-value sentinel.

    All angles are projected 2D internal angles / 180 degrees. Positions and
    distances are in contemporaneous shoulder-width units around shoulder center.
    Pixel-space points are not changed. No interpolation or temporal smoothing.
    Feature-group indices let callers separate pose, shape, direction and speed.
    """
    if not np.isfinite(max_motion_gap_seconds) or max_motion_gap_seconds <= 0:
        raise ValueError("max_motion_gap_seconds must be positive and finite")
    xy, valid, ts, frame = _validate(arrays)
    n = len(xy)
    center = (xy[:, 11] + xy[:, 12]) / 2
    scale = np.linalg.norm(xy[:, 11] - xy[:, 12], axis=1)
    body_valid = valid[:, 11] & valid[:, 12] & np.isfinite(scale) & (scale > 1e-8)
    norm = np.full(xy.shape, np.nan, dtype=np.float64)
    norm[body_valid] = (xy[body_valid] - center[body_valid, None]) / scale[body_valid, None, None]
    norm_valid = valid & body_valid[:, None]
    values, masks, names, units = [], [], [], []
    groups: dict[str, list[int]] = {}
    index: dict[str, int] = {}
    dependencies: dict[str, list[int]] = {}

    def add(name, val, mask, group, unit, ids):
        val = np.asarray(val, dtype=np.float64)
        mask = np.asarray(mask, dtype=bool) & np.isfinite(val)
        if val.shape != (n,) or mask.shape != (n,):
            raise ValueError(f"Invalid feature shape: {name}")
        clean = np.where(mask, val, np.nan)
        index[name] = len(names)
        groups.setdefault(group, []).append(len(names))
        names.append(name); units.append(unit); values.append(clean); masks.append(mask)
        dependencies[name] = sorted(set(ids))

    for side in ("L", "R"):
        off = HAND_OFFSETS[side]
        sh, el, pw = ARM_IDS[side]
        # Hand-detector wrist and body-detector wrist remain separate estimates.
        for item, j in (("hand_wrist", off), ("pose_wrist", pw), ("elbow", el), ("index_tip", off + 8)):
            for dim, axis in enumerate(("x", "y")):
                add(f"{side}_{item}_{axis}", norm[:, j, dim], norm_valid[:, j],
                    f"{side}_position", "shoulder_width", [11, 12, j])
        palm_ids = [off + j for j in (0, 5, 9, 13, 17)]
        palm = np.mean(norm[:, palm_ids], axis=1)
        palm_valid = norm_valid[:, palm_ids].all(axis=1)
        for dim, axis in enumerate(("x", "y")):
            add(f"{side}_palm_center_{axis}", palm[:, dim], palm_valid,
                f"{side}_position", "shoulder_width", [11, 12, *palm_ids])
        ang, ok = _angle(xy, valid, (sh, el, pw))
        add(f"{side}_elbow_angle", ang, ok, f"{side}_arm_angle", "half_turn", [sh, el, pw])
        for name, triplet in FINGER_ANGLES.items():
            ids = tuple(off + j for j in triplet)
            ang, ok = _angle(xy, valid, ids)
            add(f"{side}_{name}_angle", ang, ok, f"{side}_finger_angle", "half_turn", ids)
        axis = xy[:, off + 9] - xy[:, off]
        axis_len = np.linalg.norm(axis, axis=1)
        axis_valid = valid[:, [off, off + 9]].all(axis=1) & np.isfinite(axis_len) & (axis_len > 1e-8)
        for dim, dimname in enumerate(("x", "y")):
            direction = np.full(n, np.nan)
            np.divide(axis[:, dim], axis_len, out=direction, where=axis_valid)
            add(f"{side}_palm_axis_{dimname}", direction, axis_valid,
                f"{side}_orientation_2d", "unit_direction", [off, off + 9])
        width = np.linalg.norm(norm[:, off + 5] - norm[:, off + 17], axis=1)
        add(f"{side}_palm_width", width, norm_valid[:, [off + 5, off + 17]].all(axis=1),
            f"{side}_position", "shoulder_width", [11, 12, off + 5, off + 17])

    pair_mask = norm_valid[:, [33, 54]].all(axis=1)
    delta = norm[:, 54] - norm[:, 33]
    for dim, axis in enumerate(("x", "y")):
        add(f"hands_wrist_delta_{axis}", delta[:, dim], pair_mask,
            "hand_relation", "shoulder_width", [11, 12, 33, 54])
    add("hands_wrist_distance", np.linalg.norm(delta, axis=1), pair_mask,
        "hand_relation", "shoulder_width", [11, 12, 33, 54])
    # Disagreement between two wrist estimators is diagnostic, not an identity fix.
    for side in ("L", "R"):
        off, pw = HAND_OFFSETS[side], ARM_IDS[side][2]
        distance = np.linalg.norm(norm[:, off] - norm[:, pw], axis=1)
        add(f"{side}_wrist_estimator_distance", distance, norm_valid[:, [off, pw]].all(axis=1),
            f"{side}_diagnostic", "shoulder_width", [11, 12, off, pw])

    dt = np.diff(ts).astype(np.float64) / 1e6
    consecutive = (np.diff(frame) == 1) & (dt > 0) & (dt <= max_motion_gap_seconds)
    for side in ("L", "R"):
        off = HAND_OFFSETS[side]
        step_valid = norm_valid[1:, off] & norm_valid[:-1, off] & consecutive
        velocity = np.full((n, 2), np.nan)
        if n > 1:
            velocity[1:][step_valid] = (norm[1:, off] - norm[:-1, off])[step_valid] / dt[step_valid, None]
        vmask = np.r_[False, step_valid]
        for dim, axis in enumerate(("x", "y")):
            add(f"{side}_wrist_velocity_{axis}", velocity[:, dim], vmask,
                f"{side}_speed", "shoulder_width_per_second", [11, 12, off])
        speed = np.linalg.norm(velocity, axis=1)
        moving = vmask & np.isfinite(speed) & (speed > 1e-8)
        for dim, axis in enumerate(("x", "y")):
            direction = np.full(n, np.nan)
            np.divide(velocity[:, dim], speed, out=direction, where=moving)
            add(f"{side}_wrist_motion_direction_{axis}", direction, moving,
                f"{side}_direction", "unit_direction", [11, 12, off])

    x, mask = np.stack(values, axis=1), np.stack(masks, axis=1)
    assert np.array_equal(mask, np.isfinite(x))
    return {
        "schema": SCHEMA, "x": x, "mask": mask, "feature_names": names,
        "groups": groups, "units": units, "dependencies": dependencies,
        "timestamp_us": ts.copy(), "frame_id": frame.copy(),
        "joint_observed_usable": valid.copy(), "body_normalizer_valid": body_valid.copy(),
        "normalization": "Per-frame shoulder center and width; source xy unchanged",
        "angle_interpretation": "2D projected internal angle / 180; no flexed/extended correctness threshold",
        "motion_interpretation": "Consecutive original frames only; image y grows downward; speed is separate from direction",
        "estimated_not_truth": True, "learned_model": False,
        "semantic_correctness": None, "contact": None, "hidden_shape": None,
    }


def _runs(mask: np.ndarray):
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    return list(zip(starts.tolist(), ends.tolist()))


def summarize_motion(features: Mapping, *, display_displacement: float = 0.03) -> dict:
    """Lossless availability summary plus coarse display-only movement intervals.

    `display_displacement` only suppresses tiny direction-run captions. All x/mask
    samples remain available. It is NOT a rest, stage-boundary or KSL threshold.
    No statement about handshape identity, contact, intention or repetitions.
    """
    if not np.isfinite(display_displacement) or display_displacement < 0:
        raise ValueError("display_displacement must be nonnegative and finite")
    x, mask = np.asarray(features["x"]), np.asarray(features["mask"], dtype=bool)
    ts, frame = np.asarray(features["timestamp_us"]), np.asarray(features["frame_id"])
    names = list(features["feature_names"])
    n = len(ts)
    out = {"frames_retained": n, "from_seconds": float(ts[0]) / 1e6,
           "to_seconds": float(ts[-1]) / 1e6, "sides": {},
           "semantic_stage_boundaries": None, "repeat_count": None,
           "correctness": None, "note": "Saved detector estimates; availability is not accuracy."}
    for side in ("L", "R"):
        wx, wy = names.index(f"{side}_hand_wrist_x"), names.index(f"{side}_hand_wrist_y")
        dx, dy = names.index(f"{side}_wrist_velocity_x"), names.index(f"{side}_wrist_velocity_y")
        observed = mask[:, wx] & mask[:, wy]
        missing = [{"from_frame": int(frame[a]), "to_frame": int(frame[b]),
                    "from_seconds": float(ts[a]) / 1e6, "to_seconds": float(ts[b]) / 1e6,
                    "frames": b - a + 1} for a, b in _runs(~observed)]
        intervals = []
        for vel_col, pos_col, positive, negative in (
            (dy, wy, "down_in_image", "up_in_image"),
            (dx, wx, "right_in_image", "left_in_image"),
        ):
            for polarity, label in ((1, positive), (-1, negative)):
                selected = mask[:, vel_col] & (np.nan_to_num(x[:, vel_col], nan=0) * polarity > 1e-8)
                for a, b in _runs(selected):
                    start = a - 1
                    if start < 0 or not mask[start, pos_col] or not mask[b, pos_col]:
                        continue
                    displacement = float(x[b, pos_col] - x[start, pos_col])
                    if abs(displacement) < display_displacement:
                        continue
                    intervals.append({"direction": label, "from_frame": int(frame[start]),
                        "to_frame": int(frame[b]), "from_seconds": float(ts[start]) / 1e6,
                        "to_seconds": float(ts[b]) / 1e6, "displacement_shoulder_width": displacement,
                        "scope": "2D body-relative estimate, not a linguistic action stage"})
        fingers = features["groups"][f"{side}_finger_angle"]
        out["sides"][side] = {"wrist_observed_frames": int(observed.sum()),
            "wrist_total_frames": n, "wrist_observation_ratio": float(observed.mean()),
            "unobserved_intervals": missing,
            "finger_observed_cells": int(mask[:, fingers].sum()),
            "finger_expected_cells": int(n * len(fingers)),
            "motion_intervals": sorted(intervals, key=lambda item: (item["from_seconds"], item["direction"]))}
    return out


def _self_test():
    checks = []
    rng = np.random.default_rng(12)
    xy = rng.normal(size=(8, 75, 2)) * 10 + 300
    xy[:, 11] = [250, 250]; xy[:, 12] = [350, 250]
    xy[:, 33] = np.c_[np.full(8, 230), 340 - np.arange(8) * 4]
    xy[:, 54] = np.c_[np.full(8, 360), 340 + np.arange(8) * 4]
    a = dict(xy_px=xy, observed=np.ones((8, 75), bool), usable=np.ones((8, 75), bool),
             timestamp_us=np.arange(8, dtype=np.int64) * 33333, frame_id=np.arange(8))
    before = {k: v.copy() for k, v in a.items()}
    f = extract_features(a)

    def check(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        checks.append(name)

    check("all_frames_retained", len(f["x"]) == len(xy))
    check("input_not_mutated", all(np.array_equal(a[k], v) for k, v in before.items()))
    check("nan_exactly_matches_mask", np.array_equal(np.isfinite(f["x"]), f["mask"]))
    b = {k: v.copy() for k, v in a.items()}; b["xy_px"] = b["xy_px"] * 2.7 + [120, -80]
    g = extract_features(b)
    check("translation_positive_scale_invariance", np.allclose(f["x"], g["x"], atol=1e-11, equal_nan=True))
    b = {k: v.copy() for k, v in a.items()}; b["usable"][3, 54:75] = False
    g = extract_features(b)
    columns = [i for i, name in enumerate(g["feature_names"]) if 54 in g["dependencies"][name]]
    check("missing_hand_not_filled", not g["mask"][3, columns].any())
    check("missing_hand_does_not_erase_other_arm", g["mask"][3, g["feature_names"].index("L_elbow_angle")])
    b = {k: v.copy() for k, v in a.items()}; b["observed"][3, 54] = False
    g = extract_features(b)
    check("unobserved_even_if_usable_not_adopted", not g["mask"][3, g["feature_names"].index("R_hand_wrist_x")])
    b = {k: v.copy() for k, v in a.items()}; b["usable"][3, 54 + 8] = False
    g = extract_features(b)
    check("one_finger_point_keeps_neighbor_angle", g["mask"][3, g["feature_names"].index("R_index_pip_angle")])
    check("affected_dip_angle_unknown", not g["mask"][3, g["feature_names"].index("R_index_dip_angle")])
    b = {k: v.copy() for k, v in a.items()}; b["frame_id"][4:] += 4
    g = extract_features(b)
    check("no_velocity_across_omitted_frames", not g["mask"][4, g["feature_names"].index("R_wrist_velocity_y")])
    b = {k: v.copy() for k, v in a.items()}; b["timestamp_us"][4:] += 300000
    g = extract_features(b)
    check("no_velocity_across_long_time_gap", not g["mask"][4, g["feature_names"].index("R_wrist_velocity_y")])
    b = {k: v.copy() for k, v in a.items()}; b["usable"][3, 11] = False
    g = extract_features(b)
    check("missing_body_normalizer_keeps_local_finger_angle", g["mask"][3, g["feature_names"].index("R_index_pip_angle")])
    check("missing_body_normalizer_blocks_position", not g["mask"][3, g["feature_names"].index("R_hand_wrist_x")])
    s = summarize_motion(f)
    check("image_direction_signs", any(v["direction"] == "up_in_image" for v in s["sides"]["L"]["motion_intervals"]))
    check("no_correctness_or_repeat_assertion", s["correctness"] is None and s["repeat_count"] is None)
    try:
        extract_features(a, guide="expected action")
        rejected = False
    except TypeError:
        rejected = True
    check("guide_label_not_accepted", rejected)
    return dict(status="PASS", checks=checks, count=len(checks), scope="synthetic geometry and interface checks; not accuracy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.self_test:
        parser.error("Only --self-test is an executable action; import for public-observation use")
    result = _self_test()
    if args.output:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
