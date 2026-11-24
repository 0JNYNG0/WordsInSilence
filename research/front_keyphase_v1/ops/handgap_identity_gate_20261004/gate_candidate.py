"""Frozen, experiment-only eligibility veto around the original handgap.restore.

The caller must put partial_hand_reconstruction_20261003/implementation on
sys.path so that ``handgap`` is the frozen experiment implementation. This module
does not import the operational service, read data, or perform model inference.
"""
from __future__ import annotations

import numpy as np

import handgap as HG


SIDES = ("left", "right")
POSE_WRISTS = {"left": 7, "right": 4}
PALM_LOCAL = np.array([0, 5, 9, 13, 17], dtype=np.int64)


def _mean_distance(a, b):
    """Mean corresponding-point Euclidean distance; float64, JSON-safe result."""
    with np.errstate(over="ignore", invalid="ignore"):
        delta = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
        distances = np.hypot(delta[:, 0], delta[:, 1])
        # Divide before summation to avoid overflow solely in the reduction.
        value = float(np.sum(distances / len(distances), dtype=np.float64))
    return value if np.isfinite(value) else None


def _palm(xy, pb, frame, side):
    hand = HG.HANDS[side]
    if not bool(pb[frame, hand].any()):
        return "absent", None
    indices = hand.start + PALM_LOCAL
    if not bool(pb[frame, indices].all()):
        return "incomplete", None
    points = xy[frame, indices].astype(np.float64)
    if not bool(np.isfinite(points).all()):
        return "incomplete", None
    return "complete", points


def _nearer(label, query, same, other, metrics, reasons):
    d_same = _mean_distance(query, same)
    d_other = _mean_distance(query, other)
    metrics[label] = {"same": d_same, "other": d_other}
    if d_same is None or d_other is None:
        reasons.append("insufficient_nonfinite_distance_" + label)
    elif not d_same < d_other:
        reasons.append("mismatch_" + label)


def _decision(xy, pb, frame, side):
    """Inspect only received observations, never base or accepted estimates."""
    opposite = "right" if side == "left" else "left"
    reasons, metrics = [], {}
    targets, others = {}, {}
    for name, row in (("before", frame - 1), ("after", frame + 1)):
        # Base eligibility already guarantees all 21 target points are present.
        _, targets[name] = _palm(xy, pb, row, side)
        state, others[name] = _palm(xy, pb, row, opposite)
        metrics["opposite_" + name + "_state"] = state
        if state == "incomplete":
            reasons.append("insufficient_opposite_palm_" + name)

        pose_ids = [POSE_WRISTS[side], POSE_WRISTS[opposite]]
        pose_ok = bool(pb[row, pose_ids].all()) and bool(
            np.isfinite(xy[row, pose_ids]).all())
        metrics["pose_" + name + "_available"] = pose_ok
        if not pose_ok:
            reasons.append("insufficient_pose_wrists_" + name)
        else:
            wrist = targets[name][0:1]
            _nearer("pose_ownership_" + name, wrist,
                    xy[row, pose_ids[0]:pose_ids[0] + 1],
                    xy[row, pose_ids[1]:pose_ids[1] + 1], metrics, reasons)

    metrics["target_temporal_distance"] = _mean_distance(
        targets["before"], targets["after"])
    if metrics["target_temporal_distance"] is None:
        reasons.append("insufficient_nonfinite_distance_target_temporal")
    if others["after"] is not None:
        _nearer("temporal_forward", targets["before"], targets["after"],
                others["after"], metrics, reasons)
    if others["before"] is not None:
        _nearer("temporal_backward", targets["after"], targets["before"],
                others["before"], metrics, reasons)

    middle_state, middle = _palm(xy, pb, frame, opposite)
    metrics["opposite_gap_state"] = middle_state
    if middle_state == "incomplete":
        reasons.append("insufficient_gap_opposite_palm")
    elif middle_state == "complete":
        for name in ("before", "after"):
            if others[name] is None:
                reasons.append("insufficient_gap_witness_endpoint_" + name)
            else:
                _nearer("gap_witness_" + name, middle, others[name],
                        targets[name], metrics, reasons)

    return {"frame": int(frame), "side": side, "accepted": not reasons,
            "reasons": reasons, "metrics": metrics}


def restore(xy, present, ts) -> dict:
    """Return the old API plus decisions for old-rule eligible hand-frame events.

    Decisions are frame/left/right ordered. A reason starting with
    ``insufficient_`` denotes missing or unusable evidence; ``mismatch_`` denotes
    a failed strict association comparison, including a tie. Multiple reasons
    may describe one rejection. Original values, including absent sentinels and
    NaN payloads outside accepted fills, are copied without arithmetic.
    """
    xy, present, ts, pb, _ = HG._validate(xy, present, ts)
    base = HG.restore(xy, present, ts)  # Exactly once, on the received input.
    out_xy, out_present = xy.copy(), present.copy()
    estimated = np.zeros(pb.shape, dtype=bool)
    decisions = []
    for frame in range(len(ts)):
        for side in SIDES:
            hand = HG.HANDS[side]
            if not bool(base["estimated"][frame, hand].any()):
                continue
            decision = _decision(xy, pb, frame, side)
            decisions.append(decision)
            if decision["accepted"]:
                out_xy[frame, hand] = base["xy"][frame, hand]
                out_present[frame, hand] = base["present"][frame, hand]
                estimated[frame, hand] = True
    return {"xy": out_xy, "present": out_present, "ts": ts.copy(),
            "observed": pb.copy(), "estimated": estimated,
            "identity": not bool(estimated.any()), "decisions": decisions}
