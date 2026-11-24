"""Where the hands went, along the same warping path the distance already uses.

The hand-shape panels say what the fingers did; this says where the hands travelled. They are
different units and never share a drawing: shape_* is hand-local (wrist origin, hand scale), while
handpos_* is body-normalised - (wrist - body origin) / shoulder width - so both hands share one
origin and one scale and belong in one panel.

Two different things are stated apart. The COORDINATES are the front-view original video's: x
right, y down, nothing mirrored. The L / R hand LABEL is the signer's own anatomical side -
capture/landmarker.py _assign_sides binds each detected hand to an anatomical side from the pose
model's wrists, so "L" is the signer's left hand wherever it appears in the frame. No label is
swapped and no coordinate is mirrored here.

Time is each input's own elapsed time, its first timestamp subtracted from its own timestamps. The
two clocks are never subtracted from each other, and neither number is the real end of the
movement - it is elapsed time inside the trimmed, sampled sequence.

Each step also carries the hand shape of that same pair, so a drawing and a position can move
together under one slider. Those coordinates are hand-local and are never mixed with the
body-normalised positions here; each hand's panel gets ONE scale for the whole comparison, so
moving the slider never rescales the view. The path is the one already walked - no DTW is
recomputed and no model is asked again.

Gaps are never drawn across. uniform_time_sample keeps only the picked rows, so a dropout that
fell BETWEEN two picks leaves no trace in the sample; every segment is therefore checked against
the original row range as well as its two endpoints. Where a side's original timeline is not
available the payload says so instead of implying continuity.

This is a difference report. Nothing here says a direction was corrected, passed or was right.
"""
from __future__ import annotations

import numpy as np

from .dtw import _SLICES, _optimal_path, frame_cost_matrix
from .features import GROUP_ORDER

SIDES = ("L", "R")
SPACE = "body_normalised_shoulder_width"
SHAPE_SPACE = "hand_local_wrist_normalised"
SHAPE_JOINTS = 20                      # the finger points; the wrist IS this space's origin


def source_observation(seq: dict) -> dict:
    """The per-row hand-position presence of a FULL sequence, for the continuity check.

    Kept beside a reference so a gap that fell between two sampled rows can still be seen. It adds
    a key; the features, the presence masks and the timestamps it was built from are untouched.
    """
    gi = {g: i for i, g in enumerate(GROUP_ORDER)}
    out = {}
    for s in SIDES:
        # present alone is not enough: a row can be marked present and still carry a non-finite
        # coordinate, which is a break in observation exactly like an absent one. The continuity
        # flag is the AND of the two, matching what _side_points will actually plot.
        finite = np.isfinite(seq["features"][:, _SLICES[f"handpos_{s}"]]).all(axis=1)
        out[s] = (seq["present"][:, gi[f"handpos_{s}"]] & finite).tolist()
    return {"rows": int(len(seq["present"])), "handpos_present": out,
            "rule": "present AND every handpos component finite, the same test the plotted "
                    "points use"}


def _rel_seconds(ts) -> list[float]:
    """Elapsed time inside THIS input, from its own first sample. Never a cross-clock difference."""
    if len(ts) == 0:
        return []
    t0 = int(ts[0])
    return [round((int(t) - t0) / 1e6, 3) for t in ts]


def _side_points(sample: dict, side: str) -> list:
    """Body-normalised hand position per sampled frame, or None where it was not observed."""
    gi = GROUP_ORDER.index(f"handpos_{side}")
    sl = _SLICES[f"handpos_{side}"]
    out = []
    for t in range(len(sample["features"])):
        xy = sample["features"][t, sl]
        if not sample["present"][t, gi] or not np.isfinite(xy).all():
            out.append(None)
        else:
            out.append([round(float(xy[0]), 4), round(float(xy[1]), 4)])
    return out


def _shape_points(sample: dict, side: str, t: int):
    """The 20 finger points of one hand at one frame, in hand-local units, or None.

    shape_* presence is a whole-group contract, so the hand is either fully comparable here or not
    at all. The wrist is not among these points: it is the origin of this space by construction,
    and its BODY coordinate (handpos_*) belongs to the trajectory panel, never to this one.
    """
    gi = GROUP_ORDER.index(f"shape_{side}")
    sl = _SLICES[f"shape_{side}"]
    if t >= len(sample["features"]) or not sample["present"][t, gi]:
        return None
    v = sample["features"][t, sl]
    if not np.isfinite(v).all():
        return None
    return [[round(float(v[2 * k]), 4), round(float(v[2 * k + 1]), 4)]
            for k in range(SHAPE_JOINTS)]


def _segments(sample: dict, points: list, source: dict | None, side: str) -> dict:
    """Which consecutive sampled frames may be joined by a line."""
    n = len(points)
    picks = sample.get("picked_rows")
    present = (source or {}).get("handpos_present", {}).get(side)
    checked = picks is not None and present is not None and len(present) >= (
        int(max(picks)) + 1 if len(picks) else 0)
    connect, reasons = [], []
    for k in range(max(0, n - 1)):
        if points[k] is None or points[k + 1] is None:
            connect.append(False)
            reasons.append("ENDPOINT_NOT_OBSERVED")
            continue
        if not checked:
            # not knowing is not the same as being continuous, and it is never drawn as one
            connect.append(False)
            reasons.append("ORIGINAL_NOT_AVAILABLE")
            continue
        a, b = int(picks[k]), int(picks[k + 1])
        gap = not all(present[r] for r in range(a, b + 1))
        connect.append(not gap)
        reasons.append("GAP_IN_ORIGINAL" if gap else None)
    return {"connect": connect, "reason": reasons, "checked": bool(checked),
            "observed_frames": sum(1 for p in points if p is not None), "frames": n,
            "drawn_segments": sum(1 for c in connect if c), "possible_segments": len(connect)}


def location_path(query: dict, ref: dict, frozen_config: dict,
                  query_source: dict | None = None) -> dict:
    """Hand positions along the shared warping path, with time, observation and continuity."""
    cost_cfg = {k: v for k, v in (frozen_config.get("cost") or {}).items()
                if k in ("one_sided_penalty", "one_sided_mode", "group_weights",
                         "hand_participation")}
    cost, _ = frame_cost_matrix(query, ref, **cost_cfg)
    path = _optimal_path(cost, 0.2)
    if not path:
        return {"status": "UNAVAILABLE", "reason": "no admissible warping path"}

    q_pts = {s: _side_points(query, s) for s in SIDES}
    r_pts = {s: _side_points(ref, s) for s in SIDES}
    q_seg = {s: _segments(query, q_pts[s], query_source, s) for s in SIDES}
    r_seg = {s: _segments(ref, r_pts[s], ref.get("source_observation"), s) for s in SIDES}
    q_time = _rel_seconds(query.get("timestamp_us", []))
    r_time = _rel_seconds(ref.get("timestamp_us", []))

    steps = []
    for i, j in path:
        step = {"i": int(i), "j": int(j),
                "relative_time_s": {"query": q_time[i] if i < len(q_time) else None,
                                    "reference": r_time[j] if j < len(r_time) else None},
                "difference": {}, "distance": {}, "observed": {}}
        for s in SIDES:
            a, b = q_pts[s][i], r_pts[s][j]
            both = a is not None and b is not None
            step["observed"][s] = {"query": a is not None, "reference": b is not None,
                                   "both": both}
            step["difference"][s] = [round(a[0] - b[0], 4), round(a[1] - b[1], 4)] if both else None
            step["distance"][s] = round(float(np.hypot(a[0] - b[0], a[1] - b[1])), 4) if both \
                else None
        steps.append(step)

    # The hand shape of the SAME pair, on the steps already walked above. No DTW is recomputed
    # and no model is asked again: this reads the two feature rows the step already names.
    scales = ((frozen_config.get("handshape_check") or {}).get("joint_scales") or {})
    view = {s: {"min": None, "max": None, "steps_observed": 0} for s in SIDES}
    for st in steps:
        st["shape"] = {}
        for s in SIDES:
            a = _shape_points(query, s, st["i"])
            b = _shape_points(ref, s, st["j"])
            both = a is not None and b is not None
            block = {"observed": {"query": a is not None, "reference": b is not None,
                                  "both": both},
                     "query": a, "reference": b, "difference": None}
            if both:
                sc = scales.get(f"shape_{s}")
                if sc:
                    block["difference"] = [
                        round(float(np.hypot(a[k][0] - b[k][0], a[k][1] - b[k][1])) / sc, 3)
                        for k in range(SHAPE_JOINTS)]
                view[s]["steps_observed"] += 1
                for pts in (a, b):
                    xs = [q[0] for q in pts]
                    ys = [q[1] for q in pts]
                    lo = [min(xs), min(ys)]
                    hi = [max(xs), max(ys)]
                    if view[s]["min"] is None:
                        view[s]["min"], view[s]["max"] = lo, hi
                    else:
                        view[s]["min"] = [min(view[s]["min"][0], lo[0]),
                                          min(view[s]["min"][1], lo[1])]
                        view[s]["max"] = [max(view[s]["max"][0], hi[0]),
                                          max(view[s]["max"][1], hi[1])]
            st["shape"][s] = block
    for s in SIDES:
        view[s]["steps_total"] = len(steps)

    compared = {s: sum(1 for st in steps if st["observed"][s]["both"]) for s in SIDES}
    extra = {}
    if (frozen_config.get("feedback") or {}).get("selected_interval_review") is True:
        from .phase_feedback import inspect_path
        definition = ((frozen_config.get("feedback") or {}).get("reference_review_windows") or {}).get(ref.get("clip_id"))
        extra["interval_review"] = inspect_path(query, ref, path, review_window=definition)
    return {
        **extra,
        "step_handshape": True,
        "shape_space": SHAPE_SPACE,
        "shape_view": view,
        "shape_scale_rule": "ONE scale per hand for the whole comparison, taken from the bounds "
                            "over every observed step of the input and the reference together. "
                            "The slider never refits the view, and both sides share the scale.",
        "shape_unit": "hand-local: (hand point - wrist) / hand scale. The wrist is this space's "
                      "origin and its body coordinate (handpos_*) is never mixed in. Per-joint "
                      "differences are divided by the shape group's own scale.",
        "shape_observation": "shape_* presence is a whole-group contract: a hand is either fully "
                             "comparable on a step or not at all. A hand unobserved on the "
                             "selected step is not drawn and not connected. This is the SELECTED "
                             "STEP's observation and is separate from the whole-path rate.",
        "status": "OK",
        "space": SPACE,
        "unit": "shoulder widths; both hands share one origin and one scale",
        "origin": "the body frame origin used by the features, identical for both hands",
        "axes": "x right, y down, in the FRONT-VIEW ORIGINAL VIDEO's coordinates, not mirrored",
        "side_labels": "L and R are the SIGNER's anatomical sides, bound from the pose model's "
                       "wrists in capture/landmarker.py _assign_sides - not the left and right of "
                       "the video frame. Nothing here swaps a label or mirrors a coordinate.",
        "not_mixed_with": "shape_* is hand-local (wrist origin, hand scale) and is never plotted "
                          "in this panel",
        "points": {"query": q_pts, "reference": r_pts},
        "relative_time_s": {"query": q_time, "reference": r_time},
        "time_rule": "each input's own first timestamp subtracted from its own timestamps. The "
                     "two clocks are never subtracted from each other, and this is not the real "
                     "end of the movement - it is elapsed time inside the trimmed sample.",
        "segments": {"query": q_seg, "reference": r_seg},
        "continuity_rule": "a segment is drawn only when the hand is observed at both endpoints "
                           "AND the original rows between the two picked frames carry it "
                           "throughout. An unchecked side is not drawn as continuous.",
        "steps": steps,
        "path_pairs": len(path),
        "steps_compared": compared,
        "is_a_verdict": False,
        "note": "a difference report. It shows where the two paths differ, never whether the "
                "movement was correct - no boundary between right and wrong has been measured.",
    }
