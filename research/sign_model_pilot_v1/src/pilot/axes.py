"""Split one DTW distance into handshape / location / movement, and measure each separately.

Why
  angle_variant_check.py showed the hand representation is not the bottleneck: swapping
  wrist-relative coordinates for rotation-invariant angles cut the within-word spread exactly where
  predicted, yet bought no headroom, because the nearest-neighbour distance fell just as far. A
  single scalar distance mixes every axis into one number, so no hand encoding can help when the
  axis that separates two words is a different one. NA and ALDA are the clearest case: the same
  palm at the same place, held still versus moved up and down. Only MOVEMENT tells them apart.

Axes
  handshape  flexion angles per hand (rotation invariant)               38 channels
  location   where each hand sits, in body frame and relative to the face   12
  movement   how fast the wrists and elbows travel, in body frame           12

Deviation from research/front_keyphase_v1, on purpose: there, `movement` also carries the static
arm positions (arm_L/R). Here movement is velocity only, so the three axes do not overlap and a
result can be attributed to one of them. front_keyphase_v1 itself is read, never modified.

Velocity uses each clip's own timestamps through the contract's delta_time channel, and a central
difference, so it is a real rate rather than a per-frame displacement that a different capture rate
would change.
"""
from __future__ import annotations

import numpy as np

from . import data as D
from .angles import hand_angles
from .dtw_baseline import BAND_RATIO, dtw_from_cost

AXES = ("handshape", "location", "movement")
MIN_SHARED = {"handshape": 6, "location": 4, "movement": 4}

# indices into the 51-point body-relative block
NOSE, L_ELBOW, R_ELBOW = 0, 3, 4
HAND_ROOT = (D.N_BODY, D.N_BODY + D.N_HAND)        # each hand's wrist, in body frame
EPS = 1e-6


def _blocks(clip: dict):
    x = clip["x"]
    t = len(x)
    body = x[:, :D.CH_BODY_XY].reshape(t, D.N_POINTS, 2)
    bval = x[:, D.CH_BODY_XY + D.CH_HAND_XY:
             D.CH_BODY_XY + D.CH_HAND_XY + D.N_POINTS] > 0.5
    hv_start = D.CH_BODY_XY + D.CH_HAND_XY + D.N_POINTS
    dt = x[:, -1]
    return body, bval, hv_start, dt


VELOCITY_SPAN = 1        # half-width in frames of the central difference (1 = frame to frame)


def _velocity(p: np.ndarray, ok: np.ndarray, dt: np.ndarray, span: int = None):
    """Central difference in the clip's own time. [T,K,2],[T,K] -> [T,K,2],[T,K]."""
    k = VELOCITY_SPAN if span is None else span
    t = len(p)
    v = np.zeros_like(p)
    g = np.zeros(ok.shape, dtype=bool)
    if t < 2 * k + 1:
        return v, g
    # elapsed seconds across the whole window, from the clip's own per-frame intervals
    csum = np.concatenate([[0.0], np.cumsum(dt[1:])])
    dur = csum[2 * k:] - csum[:-2 * k]
    good = dur > EPS
    v[k:-k] = np.where(good[:, None, None],
                       (p[2 * k:] - p[:-2 * k]) / np.maximum(dur, EPS)[:, None, None], 0.0)
    g[k:-k] = ok[2 * k:] & ok[:-2 * k] & good[:, None]
    # one-sided at the ends, otherwise the first and last frame carry no velocity at all and the
    # warping path has no admissible corner to start or finish in
    for i, j in [(a, a + k) for a in range(k)] + [(t - 1 - a, t - 1 - a - k) for a in range(k)]:
        d = abs(csum[max(i, j)] - csum[min(i, j)])
        if d > EPS:
            v[i] = (p[j] - p[i]) / d * (1.0 if j > i else -1.0)
            g[i] = ok[i] & ok[j]
    v[~g] = 0.0
    return v, g


def axis_features(clip: dict, axis: str):
    """(values [T,K], mask [T,K]) for one axis. Absent points are masked, never interpolated."""
    body, bval, hv_start, dt = _blocks(clip)
    t = len(body)
    x = clip["x"]

    if axis == "handshape":
        vals, masks = [], []
        for h in range(2):
            pts = x[:, D.CH_BODY_XY + h * D.N_HAND * 2:
                    D.CH_BODY_XY + (h + 1) * D.N_HAND * 2].reshape(t, D.N_HAND, 2)
            hv = x[:, hv_start + h * D.N_HAND:hv_start + (h + 1) * D.N_HAND] > 0.5
            a, ok = hand_angles(pts, hv)
            vals.append(a)
            masks.append(ok)
        return np.concatenate(vals, 1), np.concatenate(masks, 1)

    roots = list(HAND_ROOT)
    if axis == "location":
        pos = body[:, roots]                                   # hand in body frame
        ok = bval[:, roots]
        face = pos - body[:, NOSE][:, None, :]                  # and relative to the face
        okf = ok & bval[:, NOSE][:, None]
        elb = body[:, [L_ELBOW, R_ELBOW]]
        oke = bval[:, [L_ELBOW, R_ELBOW]]
        v = np.concatenate([pos.reshape(t, -1), face.reshape(t, -1), elb.reshape(t, -1)], 1)
        m = np.concatenate([np.repeat(ok, 2, 1), np.repeat(okf, 2, 1), np.repeat(oke, 2, 1)], 1)
        return np.where(m, v, 0.0), m

    if axis == "movement":
        pts = body[:, roots + [L_ELBOW, R_ELBOW]]
        ok = bval[:, roots + [L_ELBOW, R_ELBOW]]
        vel, g = _velocity(pts, ok, dt)
        speed = np.linalg.norm(vel, axis=-1)                    # rate, direction-free
        v = np.concatenate([vel.reshape(t, -1), speed], 1)
        m = np.concatenate([np.repeat(g, 2, 1), g], 1)
        return np.where(m, v, 0.0), m

    raise ValueError(axis)


def distance(a, b, axis: str, band_ratio: float = BAND_RATIO):
    """a, b are (values, mask) pairs from axis_features for the SAME axis."""
    va, ma = a
    vb, mb = b
    if len(va) < 2 or len(vb) < 2:
        return None
    # Compare on the channels BOTH clips actually use. A one-handed word has no left-hand angles at
    # all, so against a two-handed clip the shared set is empty in exactly the frames where the
    # other clip shows only its other hand - which leaves the warping path with no admissible
    # corner. Restricting to the shared support answers "how do these two differ where they are
    # comparable" instead of silently charging infinity or, worse, charging zero.
    support = (ma.mean(axis=0) >= 0.5) & (mb.mean(axis=0) >= 0.5)
    if support.sum() < MIN_SHARED[axis]:
        return None
    va, ma, vb, mb = va[:, support], ma[:, support], vb[:, support], mb[:, support]
    ka, kb = ma.any(axis=1), mb.any(axis=1)
    if ka.sum() < 2 or kb.sum() < 2:
        return None
    va, ma, vb, mb = va[ka], ma[ka], vb[kb], mb[kb]
    both = ma[:, None, :] & mb[None, :, :]
    diff = va[:, None, :] - vb[None, :, :]
    shared = both.sum(axis=2)
    total = np.where(both, np.square(diff), 0.0).sum(axis=2)
    cost = np.full(shared.shape, float("inf"))
    ok = shared >= min(MIN_SHARED[axis], int(support.sum()))
    cost[ok] = np.sqrt(total[ok] / shared[ok])
    return dtw_from_cost(cost, band_ratio)


# --- time-warp-free movement structure -------------------------------------------------------
#
# The movement AXIS above is compared with DTW, and DTW warps time to find the best alignment.
# That is exactly the property it should not have here: a clip held still and a clip moved up and
# down both pass through low-speed moments, and the warping path is free to stretch one onto the
# other. Measured, the movement axis separates nothing (1.0x to 1.7x on every word in both data
# sets, and widening the differentiation window from 1 to 5 frames does not change it).
#
# A structural count does not warp. How many times the wrist reverses direction is a property of
# the whole clip, and on AIHub it separates ALDA (median 3) from NA (median 0) far better than any
# DTW number does. DEADBAND is in shoulder-width units and has to be stated with any result: the
# separation holds from 0.01 to 0.04 and collapses by 0.08.

# VERSION NOTE. direction_reversals() below is v1: the function whose numbers appear in
# evidence/result_movement_structure.json and in the README. External review reproduced those
# numbers and then showed the function is not a repetition count at all - its deadband applies to a
# SINGLE FRAME's step, so the same trajectory returns 3 turns at 1s/30fps and 0 at either 4s/30fps
# or 1s/120fps. It is kept unchanged so the reported results stay reproducible, and superseded by
# movement_events() at the bottom of this file. Do not use v1 for new work.

DEADBAND = 0.02


def direction_reversals(clip: dict, deadband: float = None, axis_index: int = 1) -> int:
    """How many times the more active wrist reverses along one image axis (1 = vertical).

    Steps smaller than the deadband are ignored, so landmark jitter does not read as a reversal.
    Returns the larger count of the two hands, since a one-handed sign leaves the other resting.
    """
    dead = DEADBAND if deadband is None else deadband
    x = clip["x"]
    t = len(x)
    body = x[:, :D.CH_BODY_XY].reshape(t, D.N_POINTS, 2)
    bval = x[:, D.CH_BODY_XY + D.CH_HAND_XY:D.CH_BODY_XY + D.CH_HAND_XY + D.N_POINTS] > 0.5
    best = 0
    for root in HAND_ROOT:
        ok = bval[:, root]
        if ok.sum() < 5:
            continue
        step = np.diff(body[ok, root][:, axis_index])
        big = np.abs(step) > dead
        if big.sum() < 2:
            continue
        sign = np.sign(step[big])
        best = max(best, int(np.sum(sign[1:] != sign[:-1])))
    return best


# --- movement events, v2 ----------------------------------------------------------------------
#
# v1 asked "did the wrist move more than the deadband between these two frames". That is a speed
# test wearing a repetition test's clothes: halve the speed or quadruple the frame rate and the
# same path stops counting. External review demonstrated exactly that.
#
# v2 asks "did the wrist come back by more than TURN_AMPLITUDE from the furthest point it had
# reached". That is a property of the PATH, so it does not change when the same path is walked
# slowly or sampled densely. No speed is estimated anywhere, so no timing assumption is needed to
# produce the count; time is used only to decide whether two observations are close enough to be
# treated as consecutive, and is reported alongside the count.
#
# Three things are kept apart on purpose, because collapsing them is how a tracking failure turns
# into a judgement about a person's performance:
#   OK with turns == 0        the hand was observed and genuinely held still
#   INSUFFICIENT_OBSERVATION  too little was seen to say anything; turns is None, never 0
#   per-hand records          a missing hand is reported missing, never rewritten as a resting one
#
# TURN_AMPLITUDE is a round value in shoulder-width units chosen before looking at any word's
# result. It is NOT tuned to make a word pass; callers should report the sweep, not a single pick.

TURN_AMPLITUDE = 0.05          # retracement, in shoulder widths, that confirms a direction change
MAX_GAP_SECONDS = 0.25         # observations further apart than this are not joined
MIN_OBSERVED_FRAMES = 8
MIN_OBSERVED_FRACTION = 0.5


def _turns(y: np.ndarray, amp: float) -> int:
    """Direction changes confirmed by retracement amplitude. Independent of sampling and speed."""
    if len(y) < 2:
        return 0
    turns, direction = 0, 0
    lo = hi = ext = float(y[0])
    for v in map(float, y[1:]):
        if direction == 0:
            lo, hi = min(lo, v), max(hi, v)
            if hi - v >= amp:
                direction, ext = -1, v
            elif v - lo >= amp:
                direction, ext = 1, v
        elif direction == 1:
            if v > ext:
                ext = v
            elif ext - v >= amp:
                turns, direction, ext = turns + 1, -1, v
        else:
            if v < ext:
                ext = v
            elif v - ext >= amp:
                turns, direction, ext = turns + 1, 1, v
    return turns


def movement_events(clip: dict, turn_amplitude: float = None, max_gap_seconds: float = None,
                    axis_index: int = 1) -> dict:
    """Per-hand reversal count with an explicit observation status.

    Missing frames are NOT deleted and their neighbours are NOT joined: a gap longer than
    max_gap_seconds splits the series, and turns are counted inside each observed run only.
    """
    amp = TURN_AMPLITUDE if turn_amplitude is None else turn_amplitude
    gap = MAX_GAP_SECONDS if max_gap_seconds is None else max_gap_seconds
    x = clip["x"]
    t = len(x)
    body = x[:, :D.CH_BODY_XY].reshape(t, D.N_POINTS, 2)
    bval = x[:, D.CH_BODY_XY + D.CH_HAND_XY:D.CH_BODY_XY + D.CH_HAND_XY + D.N_POINTS] > 0.5
    clock = np.cumsum(x[:, -1].astype(np.float64))

    out = {"settings": {"turn_amplitude": amp, "max_gap_seconds": gap,
                        "axis": "vertical" if axis_index == 1 else "horizontal",
                        "min_observed_frames": MIN_OBSERVED_FRAMES,
                        "min_observed_fraction": MIN_OBSERVED_FRACTION},
           "total_frames": t, "clip_seconds": round(float(clock[-1] - clock[0]), 4) if t else 0.0}
    for name, root in zip(("left", "right"), HAND_ROOT):
        idx = np.flatnonzero(bval[:, root])
        rec = {"observed_frames": int(idx.size), "total_frames": t,
               "observed_fraction": round(float(idx.size / t), 4) if t else 0.0}
        if idx.size < MIN_OBSERVED_FRAMES or rec["observed_fraction"] < MIN_OBSERVED_FRACTION:
            rec.update(status="INSUFFICIENT_OBSERVATION", turns=None, segments=0,
                       reason=(f"observed {idx.size} frames "
                               f"({rec['observed_fraction']:.2f} of the clip); need at least "
                               f"{MIN_OBSERVED_FRAMES} and {MIN_OBSERVED_FRACTION:.2f}"))
            out[name] = rec
            continue
        y = body[idx, root, axis_index]
        times = clock[idx]
        breaks = np.flatnonzero(np.diff(times) > gap) + 1
        runs = np.split(np.arange(idx.size), breaks)
        turns = sum(_turns(y[r], amp) for r in runs if r.size >= 2)
        rec.update(status="OK", turns=int(turns), segments=int(len(runs)),
                   observed_seconds=round(float(times[-1] - times[0]), 4),
                   longest_gap_seconds=(round(float(np.diff(times).max()), 4)
                                        if idx.size > 1 else 0.0),
                   excursion=round(float(y.max() - y.min()), 4),
                   turns_span_gaps=False, reason=None)
        out[name] = rec
    return out
