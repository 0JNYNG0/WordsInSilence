"""Normalization and per-frame feature groups. Clips and webcam use this identically.

Every group declares the joints it needs. A group is either comparable in a frame or it is not;
nothing is silently filled in, so "only the visible points matched" cannot buy a good score.
"""
from __future__ import annotations

import numpy as np

from .observation import (
    LEFT_HAND_SLICE, POSE_L_SHOULDER, POSE_R_SHOULDER, RIGHT_HAND_SLICE,
)

POSE_NOSE = 0
POSE_L_ELBOW, POSE_R_ELBOW = 13, 14
POSE_L_WRIST_IDX, POSE_R_WRIST_IDX = 15, 16
HAND_MCP = (5, 9, 17)  # index / middle / pinky MCP, used for hand scale
HAND_DIR_TIP = 9       # middle MCP: hand direction proxy, wrist(0) -> 9

# group -> (width, weight). Weights are fixed here and frozen by config hash.
GROUPS = {
    "arm_L": (4, 1.0), "arm_R": (4, 1.0),
    "shape_L": (40, 2.0), "shape_R": (40, 2.0),
    "orient_L": (2, 1.0), "orient_R": (2, 1.0),
    "handpos_L": (2, 1.5), "handpos_R": (2, 1.5),
    "facerel_L": (2, 1.0), "facerel_R": (2, 1.0),
    "bimanual": (2, 1.0),
    # Movement. Always computed so the feature layout is fixed; weight 0 in the A baseline,
    # so enabling it is a pure weight switch and the A/B compares one thing.
    "motion_L": (3, 0.0), "motion_R": (3, 0.0),
    # Joint flexion angles. The coordinate-based shape groups turned out to correlate +0.958 with
    # palm orientation, so they measure viewing angle as much as finger configuration. An angle
    # between two bone segments is unchanged when the whole hand rotates in the image plane.
    # Width 19 = 3 interior angles x 5 fingers + 4 adjacent-finger spreads. Weight 0 until tested.
    "bend_L": (19, 0.0), "bend_R": (19, 0.0),
}
FINGER_CHAINS = ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
                 (0, 13, 14, 15, 16), (0, 17, 18, 19, 20))
SPREAD_PAIRS = ((1, 2), (2, 3), (3, 4), (0, 1))   # index-middle, middle-ring, ring-pinky, thumb-index
DYNAMICS_GROUPS = ("motion_L", "motion_R")
GROUP_ORDER = list(GROUPS)
FEATURE_WIDTH = sum(w for w, _ in GROUPS.values())
TOTAL_WEIGHT = sum(w for _, w in GROUPS.values())


class FeatureError(Exception):
    pass


def body_frame(obs) -> tuple[np.ndarray, float, np.ndarray]:
    """Per-frame shoulder midpoint, one robust shoulder-width scale, and a validity mask.

    The scale is a single robust summary for the whole sequence: a per-frame scale would let a
    detector wobble rescale the signer. Torso tilt is deliberately NOT removed.
    """
    ok = obs.usable[:, POSE_L_SHOULDER] & obs.usable[:, POSE_R_SHOULDER]
    if not ok.any():
        raise FeatureError("no frame has both shoulders usable")
    ls = obs.xy_px[:, POSE_L_SHOULDER, :]
    rs = obs.xy_px[:, POSE_R_SHOULDER, :]
    origin = (ls + rs) / 2.0
    widths = np.linalg.norm(ls[ok] - rs[ok], axis=1)
    scale = float(np.median(widths))
    if not np.isfinite(scale) or scale <= 1e-6:
        raise FeatureError(f"degenerate shoulder width {scale}")
    return origin, scale, ok


def active_span(obs) -> tuple[int, int]:
    """First..last frame with a usable hand. Applied identically to clips and attempts."""
    any_hand = (obs.usable[:, LEFT_HAND_SLICE].any(axis=1)
                | obs.usable[:, RIGHT_HAND_SLICE].any(axis=1))
    idx = np.flatnonzero(any_hand)
    if idx.size == 0:
        raise FeatureError("no frame has a usable hand")
    return int(idx[0]), int(idx[-1])


def _hand_scale(pts: np.ndarray, ok: np.ndarray) -> float:
    """Effective hand size from wrist-to-MCP lengths; NaN when it cannot be measured."""
    lens = [float(np.linalg.norm(pts[m] - pts[0])) for m in HAND_MCP if ok[m] and ok[0]]
    if not lens:
        return float("nan")
    s = float(np.median(lens))
    return s if s > 1e-6 else float("nan")


def _angle(u: np.ndarray, v: np.ndarray) -> float:
    """Angle between two 2D vectors, normalised to [0,1]. NaN when either is degenerate."""
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-6 or nv < 1e-6:
        return float("nan")
    c = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
    return float(np.arccos(c) / np.pi)


def bend_angles(pts: np.ndarray) -> np.ndarray:
    """21 hand landmarks -> 19 rotation-invariant angles (in the image plane)."""
    out = np.empty(19, dtype=np.float32)
    k = 0
    for chain in FINGER_CHAINS:
        for i in range(1, 4):
            a, b, c = pts[chain[i - 1]], pts[chain[i]], pts[chain[i + 1]]
            out[k] = _angle(a - b, c - b)
            k += 1
    dirs = [pts[ch[1]] - pts[ch[4]] for ch in FINGER_CHAINS]   # mcp -> tip per finger
    for i, j in SPREAD_PAIRS:
        out[k] = _angle(dirs[i], dirs[j])
        k += 1
    return out


def frame_features(obs, t: int, origin: np.ndarray, scale: float) -> tuple[np.ndarray, dict]:
    """One frame -> (FEATURE_WIDTH vector with NaN in absent groups, {group: bool comparable})."""
    xy = obs.xy_px[t]
    ok = obs.usable[t]
    vec = np.full(FEATURE_WIDTH, np.nan, dtype=np.float32)
    present = {g: False for g in GROUP_ORDER}
    off = {}
    cursor = 0
    for g in GROUP_ORDER:
        off[g] = cursor
        cursor += GROUPS[g][0]

    body_ok = ok[POSE_L_SHOULDER] and ok[POSE_R_SHOULDER]
    if not body_ok:
        return vec, present

    def body(p):
        return (xy[p] - origin[t]) / scale

    for side, (elbow, wrist, hand_sl) in (
            ("L", (POSE_L_ELBOW, POSE_L_WRIST_IDX, LEFT_HAND_SLICE)),
            ("R", (POSE_R_ELBOW, POSE_R_WRIST_IDX, RIGHT_HAND_SLICE))):
        if ok[elbow] and ok[wrist]:
            vec[off[f"arm_{side}"]:off[f"arm_{side}"] + 4] = np.concatenate([body(elbow), body(wrist)])
            present[f"arm_{side}"] = True

        h_ok = ok[hand_sl]
        h_pts = xy[hand_sl]
        if not h_ok[0]:
            continue
        hs = _hand_scale(h_pts, h_ok)

        vec[off[f"handpos_{side}"]:off[f"handpos_{side}"] + 2] = (h_pts[0] - origin[t]) / scale
        present[f"handpos_{side}"] = True

        if ok[POSE_NOSE]:
            vec[off[f"facerel_{side}"]:off[f"facerel_{side}"] + 2] = (h_pts[0] - xy[POSE_NOSE]) / scale
            present[f"facerel_{side}"] = True

        if np.isfinite(hs):
            if h_ok[HAND_DIR_TIP]:
                d = h_pts[HAND_DIR_TIP] - h_pts[0]
                n = np.linalg.norm(d)
                if n > 1e-6:
                    vec[off[f"orient_{side}"]:off[f"orient_{side}"] + 2] = d / n
                    present[f"orient_{side}"] = True
            if h_ok[1:21].all():
                rel = ((h_pts[1:21] - h_pts[0]) / hs).reshape(-1)
                vec[off[f"shape_{side}"]:off[f"shape_{side}"] + 40] = rel
                present[f"shape_{side}"] = True
                ang = bend_angles(h_pts)
                if np.isfinite(ang).all():
                    vec[off[f"bend_{side}"]:off[f"bend_{side}"] + 19] = ang
                    present[f"bend_{side}"] = True

    lo, ro = ok[LEFT_HAND_SLICE][0], ok[RIGHT_HAND_SLICE][0]
    if lo and ro:
        d = (xy[RIGHT_HAND_SLICE][0] - xy[LEFT_HAND_SLICE][0]) / scale
        vec[off["bimanual"]:off["bimanual"] + 2] = d
        present["bimanual"] = True

    return vec, present


# A hand hanging at the side and not moving is not performing the sign; it merely happens to be
# inside the instructor's frame. Measured over the 18 references: resting hands peak at amplitude
# 0.067, performing hands start at 0.170, with nothing between - so 0.12 sits in an empty gap
# rather than on a tuned edge.
REST_AMPLITUDE = 0.12
REST_BELOW_SHOULDERS = 1.0   # in shoulder widths

# Velocity in shoulder-widths per second lands around 14.4 between different words while the
# positional groups land around 0.61, so an unscaled movement group would decide every comparison
# by unit choice alone. Derived from the 18x18 reference matrix only - no human recording.
# evidence/R2/motion_scale_derivation.json
MOTION_SCALE = 0.04222


def hand_participation(obs, origin, scale, lo, hi) -> dict:
    """Which hands actually take part in the sign, as opposed to resting in frame."""
    out = {}
    for side, wrist in (("L", POSE_L_WRIST_IDX), ("R", POSE_R_WRIST_IDX)):
        sel = obs.usable[lo:hi + 1, wrist]
        if sel.sum() < 5 or sel.mean() < 0.3:
            out[side] = False
            continue
        y = obs.xy_px[lo:hi + 1, wrist, 1][sel]
        rel_y = float(np.median((y - origin[lo:hi + 1, 1][sel]) / scale))
        amp = float(np.std(y / scale))
        out[side] = bool(amp >= REST_AMPLITUDE or rel_y <= REST_BELOW_SHOULDERS)
    return out


FEEDBACK_ROLE_SCHEMA = "observed_hand_role_v1"
ROLE_MIN_FRAMES = 5
ROLE_MIN_FRACTION = 0.3


def feedback_hand_roles(obs, origin, scale, lo, hi) -> dict:
    """Evidence for feedback only; NEVER replaces lexical participation or feature values.

    The existing vertical wrist-height/amplitude heuristic is retained, but a wrist and
    BOTH shoulders must be usable and finite in the same frame. Insufficient evidence is
    UNKNOWN, not resting. This is an observed-motion heuristic, not a semantic assertion
    that a hand is irrelevant to a sign; its exclusion policy remains experimental.
    Statistics cover the original visibility span, before uniform sampling.
    """
    sl = slice(lo, hi + 1)
    n = hi - lo + 1
    shoulder_ids = [POSE_L_SHOULDER, POSE_R_SHOULDER]
    shoulders = (obs.usable[sl][:, shoulder_ids].all(axis=1)
                 & np.isfinite(obs.xy_px[sl][:, shoulder_ids]).all(axis=(1, 2))
                 & np.isfinite(origin[sl]).all(axis=1))
    scale_ok = bool(np.isfinite(scale) and scale > 1e-6)
    hands = {}
    for side, wrist in (("L", POSE_L_WRIST_IDX), ("R", POSE_R_WRIST_IDX)):
        sel = (shoulders & obs.usable[sl, wrist]
               & np.isfinite(obs.xy_px[sl, wrist]).all(axis=1))
        count = int(sel.sum())
        fraction = count / n if n else 0.0
        enough = scale_ok and count >= ROLE_MIN_FRAMES and fraction >= ROLE_MIN_FRACTION
        detail = {"state": "UNKNOWN", "reason": "insufficient_paired_observation",
                  "paired_frames": count, "span_frames": n, "paired_fraction": fraction,
                  "median_height_sw": None, "vertical_amplitude_sw": None}
        if enough:
            y = obs.xy_px[sl, wrist, 1][sel]
            height = float(np.median((y - origin[sl, 1][sel]) / scale))
            amplitude = float(np.std(y / scale))
            active = amplitude >= REST_AMPLITUDE or height <= REST_BELOW_SHOULDERS
            detail.update(state="ACTIVE" if active else "RESTING",
                          reason="observed_vertical_wrist_heuristic",
                          median_height_sw=height, vertical_amplitude_sw=amplitude)
        elif not scale_ok:
            detail["reason"] = "invalid_body_scale"
        hands[side] = detail
    return {"schema": FEEDBACK_ROLE_SCHEMA, "span": [int(lo), int(hi)],
            "basis": "paired_pose_wrist_and_shoulders_before_sampling",
            "minimum_frames": ROLE_MIN_FRAMES, "minimum_fraction": ROLE_MIN_FRACTION,
            "rest_amplitude_sw": REST_AMPLITUDE,
            "rest_below_shoulders_sw": REST_BELOW_SHOULDERS, "hands": hands}


def observed_resting_hand(evidence, side) -> bool:
    """Require measured resting evidence, never a legacy False or a state string alone."""
    if not isinstance(evidence, dict) or evidence.get("schema") != FEEDBACK_ROLE_SCHEMA:
        return False
    if evidence.get("basis") != "paired_pose_wrist_and_shoulders_before_sampling":
        return False
    hands = evidence.get("hands")
    d = hands.get(side) if isinstance(hands, dict) else None
    if not isinstance(d, dict) or d.get("state") != "RESTING":
        return False
    if d.get("reason") != "observed_vertical_wrist_heuristic":
        return False
    count, n = d.get("paired_frames"), d.get("span_frames")
    if type(count) is not int or type(n) is not int or not ROLE_MIN_FRAMES <= count <= n:
        return False
    values = [d.get(k) for k in
              ("paired_fraction", "median_height_sw", "vertical_amplitude_sw")]
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
               and np.isfinite(v) for v in values):
        return False
    fraction, height, amplitude = values
    return bool(fraction >= ROLE_MIN_FRACTION and abs(fraction - count / n) < 1e-9
                and height > REST_BELOW_SHOULDERS and 0 <= amplitude < REST_AMPLITUDE)


def _fill_motion(rows, masks, times, offsets) -> None:
    """Movement groups: change per second against the previous frame that had the same hand.

    Positional features alone cannot tell a performed sign from its end pose held still, so the
    rate of change is carried as its own group rather than inferred from the warping path.
    """
    for side in ("L", "R"):
        gi = GROUP_ORDER.index(f"motion_{side}")
        pos_off = offsets[f"handpos_{side}"]
        shape_off = offsets[f"shape_{side}"]
        pos_gi = GROUP_ORDER.index(f"handpos_{side}")
        shape_gi = GROUP_ORDER.index(f"shape_{side}")
        prev = None
        for t in range(len(rows)):
            if not masks[t][pos_gi]:
                prev = None
                continue
            cur = (times[t], rows[t][pos_off:pos_off + 2].copy(),
                   rows[t][shape_off:shape_off + 40].copy() if masks[t][shape_gi] else None)
            if prev is not None:
                dt = (cur[0] - prev[0]) / 1e6
                if 0 < dt <= 0.2:          # a longer gap is not a velocity, it is a hole
                    vel = (cur[1] - prev[1]) / dt
                    if cur[2] is not None and prev[2] is not None:
                        shape_rate = float(np.linalg.norm(cur[2] - prev[2]) / dt)
                    else:
                        shape_rate = 0.0
                    rows[t][offsets[f"motion_{side}"]:offsets[f"motion_{side}"] + 3] = (
                        vel[0] * MOTION_SCALE, vel[1] * MOTION_SCALE, shape_rate * MOTION_SCALE)
                    masks[t][gi] = True
            prev = cur


def sequence_features(obs, trim: bool = True) -> dict:
    """Whole observation -> features + presence masks + real timestamps."""
    origin, scale, _ = body_frame(obs)
    lo, hi = active_span(obs) if trim else (0, obs.n_frames - 1)
    rows, masks = [], []
    for t in range(lo, hi + 1):
        v, p = frame_features(obs, t, origin, scale)
        rows.append(v)
        masks.append([p[g] for g in GROUP_ORDER])
    offsets, cursor = {}, 0
    for g in GROUP_ORDER:
        offsets[g] = cursor
        cursor += GROUPS[g][0]
    _fill_motion(rows, masks, obs.timestamp_us[lo:hi + 1], offsets)
    participating = hand_participation(obs, origin, scale, lo, hi)
    from .movement_feedback import movement_source
    from .phase_feedback import body_context
    rows_array = np.asarray(rows, dtype=np.float32)
    mask_array = np.asarray(masks, dtype=np.bool_)
    return {
        "participating": participating,
        "feedback_hand_roles": feedback_hand_roles(obs, origin, scale, lo, hi),
        "feedback_body_context": body_context(obs, origin, scale, lo, hi),
        "features": rows_array,
        "present": mask_array,
        "movement_observation": movement_source(rows_array, mask_array, obs.timestamp_us[lo:hi + 1]),
        "timestamp_us": obs.timestamp_us[lo:hi + 1].copy(),
        "frame_id": obs.frame_id[lo:hi + 1].copy(),
        "span": (lo, hi),
        "shoulder_width_px": scale,
        "group_order": GROUP_ORDER,
    }


def uniform_time_sample(seq: dict, budget: int) -> dict:
    """Method A: even samples over REAL elapsed time, nearest observation per target."""
    from .phase_feedback import sample_body_context
    ts = seq["timestamp_us"]
    n = len(ts)
    if n == 0:
        raise FeatureError("empty sequence")
    if n == 1:
        picks = np.zeros(1, dtype=int)
    else:
        targets = np.linspace(ts[0], ts[-1], budget)
        picks = np.unique(np.searchsorted(ts, targets).clip(0, n - 1))
        # searchsorted lands on the right neighbour; keep whichever is actually nearer
        picks = np.unique(np.array([
            i if i == 0 or abs(ts[i] - t) <= abs(ts[i - 1] - t) else i - 1
            for i, t in zip(np.searchsorted(ts, targets).clip(0, n - 1), targets)
        ]))
    return {
        "participating": seq.get("participating", {"L": True, "R": True}),
        "feedback_hand_roles": seq.get("feedback_hand_roles"),
        "feedback_body_context": sample_body_context(seq.get("feedback_body_context"), picks),
        "movement_observation": seq.get("movement_observation"),
        "features": seq["features"][picks],
        "present": seq["present"][picks],
        "timestamp_us": ts[picks],
        "frame_id": seq["frame_id"][picks],
        "picked_rows": picks,
        "selector": f"uniform_time:{budget}",
    }
