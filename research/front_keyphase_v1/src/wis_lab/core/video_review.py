"""Finger-angle differences ON THE ACTUAL SAVED RGB FRAMES, for the video review only.

Why this exists. The saved JPEG samples and the uniform-time DTW query are two INDEPENDENT
selections: diagnostic_capture keeps at most one RGB sample per 100 ms, uniform_time_sample picks
evenly over elapsed time inside the trimmed span. The two lists usually intersect in only a few
rows, so painting every JPEG with the colour of "some nearby sampled frame" would put one moment's
numbers on another moment's picture. The frames the person can actually see are therefore aligned
in their own right: this module builds a query sequence out of the rows the SAVED FRAMES name and
warps that sequence onto the already-prepared reference sample.

What this is NOT.

* Not recognition. One extra DTW against ONE reference, the one the trial is already bound to.
  Nothing here re-ranks, re-scores, re-thresholds, retrains or touches the classifier path, and
  the number it produces is never the trial's distance.
* Not a correctness verdict. The output is a raw difference in degrees between two PROJECTED 2D
  angles - the participant's and the instructor's - with no participant-derived scale, no band, no
  pass line and no probability. A large difference can be a different handshape or a different
  camera angle; the 2D projection cannot tell those apart, and this module does not pretend to.
* Not interpolation. A saved frame outside the trimmed analysis span stays outside, and a saved
  frame whose identity does not match the observation is rejected rather than matched to a
  neighbour.

Normalization is the existing one: sequence_features(obs, trim=True), whose bend_L / bend_R groups
hold 19 rotation-invariant angles each, stored as arccos/pi. The first 15 of those are the interior
finger angles (3 per finger) reported here; the last 4 are adjacent-finger spreads, which belong to
no single joint and are listed as unmapped rather than painted somewhere plausible.

The motion_* groups in these rows were filled from each row's ORIGINAL neighbour inside the trimmed
sequence, not from the sparse RGB selection; they take part in the cost exactly as prepared, and
this module adds no new value to them.
"""
from __future__ import annotations

import math

import numpy as np

from . import features as F
from .dtw import _SLICES, _optimal_path, frame_cost_matrix
from .features import GROUP_ORDER
from .observation import LEFT_HAND_SLICE, RIGHT_HAND_SLICE

SCHEMA = "wis_video_angle_review_v1"
# diagnostic_capture.SCHEMA, repeated as a literal on purpose: core must not import the service
# layer (and with it cv2) just to name a string.
CAPTURE_SCHEMA = "wis_local_capture_review_v1"
CAPTURE_USABLE_STATES = ("AVAILABLE", "PARTIAL")

DEFAULT_BAND_RATIO = 0.2
COST_KEYS = ("one_sided_penalty", "one_sided_mode", "group_weights", "hand_participation")
DEGREES_PER_UNIT = 180.0        # bend_* stores arccos/pi, so 1.0 == 180 degrees
INTERIOR_ANGLE_COUNT = 15       # 3 interior angles x 5 fingers; indices 15..18 are spreads
FINGERS = ("thumb", "index", "middle", "ring", "little")
THUMB_JOINTS = ("cmc", "mcp", "ip")
FINGER_JOINTS = ("mcp", "pip", "dip")
HAND_BASE = {"L": LEFT_HAND_SLICE.start, "R": RIGHT_HAND_SLICE.start}

FRAME_STATES = ("COMPARED", "NO_ANGLE_OBSERVED", "UNALIGNED", "OUTSIDE_SPAN", "IDENTITY_REJECTED")


def _angle_map() -> list[dict]:
    """Angle index -> the joint it is measured at and the bones that meet there.

    bend_angles() walks FINGER_CHAINS and, for i in 1..3, measures the angle at chain[i] between
    the segments chain[i-1]->chain[i] and chain[i]->chain[i+1]. The vertex is therefore a single
    landmark and belongs to exactly one angle. The two segments do not: chain[i]->chain[i+1] is the
    proximal segment of the NEXT angle as well. Only the distal segment is nominated for painting,
    because that assignment is one-to-one - no bone ever has to choose between two angle values -
    and it leaves the wrist-to-MCP bone uncoloured, which is correct: a wrist is not a finger
    angle.
    """
    out = []
    for finger, chain in zip(FINGERS, F.FINGER_CHAINS):
        names = THUMB_JOINTS if finger == "thumb" else FINGER_JOINTS
        for i in range(1, 4):
            u, v, w = chain[i - 1], chain[i], chain[i + 1]
            out.append({
                "index": len(out),
                "id": f"{finger}_{names[i - 1]}",
                "finger": finger,
                "joint": names[i - 1],
                "vertex_landmark": v,
                "segments": [[u, v], [v, w]],
                "colored_segment": [v, w],
                "measured": "angle at vertex_landmark between its two segments, in the image plane",
            })
    return out


ANGLE_MAP = _angle_map()
NEUTRAL_LANDMARKS = {
    "hand": [0],
    "meaning": "the hand wrist carries no interior finger angle. Pose/body landmarks carry none "
               "either and are not listed per hand; neither may be coloured with a finger value.",
}
UNMAPPED_GROUP_INDICES = {
    "indices": list(range(INTERIOR_ANGLE_COUNT, F.GROUPS["bend_L"][0])),
    "meaning": "adjacent-finger spreads. They are part of the bend_* group and of its all-or-"
               "nothing presence rule, but they are measured between two fingers and belong to no "
               "single joint, so they are reported as unmapped instead of being drawn.",
}


class ReviewError(Exception):
    """The comparison cannot be built. Reported as a reason, never as an empty result."""


def band_ratio(frozen_config: dict) -> float:
    """The band the classifier path uses, read exactly as recognize.predict reads it."""
    dtw_cfg = frozen_config["dtw"]
    return (float(dtw_cfg["band"].split("ratio")[-1].strip(" ,:"))
            if isinstance(dtw_cfg.get("band"), str) else DEFAULT_BAND_RATIO)


def cost_config(frozen_config: dict) -> dict:
    """The frozen cost keys, the same subset every other comparison in this project passes."""
    return {k: v for k, v in (frozen_config.get("cost") or {}).items() if k in COST_KEYS}


def _finite(x) -> float:
    v = float(x)
    if not math.isfinite(v):
        raise ReviewError("non-finite value reached the payload")
    return v


def _deg(x, digits: int = 2) -> float:
    return round(_finite(x), digits)


def _identity_rows(obs) -> tuple[dict, set]:
    """frame_id -> observation row, plus the ids that appear more than once."""
    rows, duplicates = {}, set()
    for i, fid in enumerate(obs.frame_id):
        key = int(fid)
        if key in rows:
            duplicates.add(key)
        else:
            rows[key] = i
    return rows, duplicates


def _int_or_none(value):
    return value if type(value) is int else None


def _classify(entry: dict, rows: dict, duplicates: set, obs, lo: int, hi: int,
              seen: dict) -> tuple[str, str | None, int | None]:
    """(state, reason, observation row) for one saved RGB frame. Never guesses an identity."""
    fid = _int_or_none(entry.get("frame_id"))
    ts = _int_or_none(entry.get("timestamp_us"))
    claimed = _int_or_none(entry.get("observation_row"))
    if fid is None or ts is None or claimed is None:
        return "IDENTITY_REJECTED", "MANIFEST_FIELD_MISSING_OR_NOT_AN_INTEGER", None
    if seen.get(fid, 0) > 1:
        return "IDENTITY_REJECTED", "DUPLICATE_FRAME_ID_IN_MANIFEST", None
    if fid in duplicates:
        return "IDENTITY_REJECTED", "DUPLICATE_FRAME_ID_IN_OBSERVATION", None
    row = rows.get(fid)
    if row is None:
        return "IDENTITY_REJECTED", "FRAME_ID_NOT_IN_OBSERVATION", None
    if claimed != row:
        return "IDENTITY_REJECTED", "OBSERVATION_ROW_DISAGREES_WITH_FRAME_ID", None
    if int(obs.timestamp_us[row]) != ts:
        return "IDENTITY_REJECTED", "TIMESTAMP_DISAGREES_WITH_OBSERVATION", None
    if row < lo or row > hi:
        return "OUTSIDE_SPAN", "FRAME_OUTSIDE_THE_TRIMMED_ANALYSIS_SPAN", row
    return "ELIGIBLE", None, row


def _hand_comparable(features: np.ndarray, present: np.ndarray, gi: int, group: str,
                     row: int) -> bool:
    """The whole-group contract, unchanged: flagged present AND every component finite."""
    return bool(present[row, gi]) and bool(np.isfinite(features[row, _SLICES[group]]).all())


def _pair_angles(q_row: np.ndarray, r_row: np.ndarray, group: str) -> list[float]:
    start = _SLICES[group].start
    return [abs(float(q_row[start + k]) - float(r_row[start + k])) * DEGREES_PER_UNIT
            for k in range(INTERIOR_ANGLE_COUNT)]


def _aggregate(side: str, pairs: list[tuple[int, list[float], list[float], list[float]]],
               matched: int, missing: dict) -> dict:
    """Aggregate ACROSS every reference row this query frame was actually matched to.

    A query frame with several matches is not resolved by taking the first pair or the smallest
    difference. Every pair contributes, the mean is the reported value, the spread and the pair
    list travel with it, and `support` says how many pairs are behind each number.
    """
    out = {
        "hand": side,
        "group": f"bend_{side}",
        "matched_pairs": matched,
        "compared_pairs": len(pairs),
        "reference_rows_compared": [p[0] for p in pairs],
        "pair_status": dict(missing),
        "aggregation": "mean over every matched reference row that was comparable; min/max and "
                       "the per-pair list are reported whenever support > 1",
    }
    if not pairs:
        out["status"] = "NOT_COMPARABLE"
        out["angles"] = []
        out["reason"] = ("QUERY_GROUP_NOT_OBSERVED" if missing["query_group_missing"] == matched
                         else "REFERENCE_GROUP_NOT_OBSERVED"
                         if missing["reference_group_missing"] == matched
                         else "NO_PAIR_HAD_THIS_GROUP_ON_BOTH_SIDES")
        return out
    out["status"] = "COMPARED"
    angles = []
    for k in range(INTERIOR_ANGLE_COUNT):
        diffs = [p[1][k] for p in pairs]
        entry = {
            "index": k,
            "id": ANGLE_MAP[k]["id"],
            "support": len(diffs),
            "mean_abs_diff_deg": _deg(sum(diffs) / len(diffs)),
            # every pair here shares ONE query row, so the observed angle is the same in all of
            # them. This is not a pick between differing values.
            "query_deg": _deg(pairs[0][2][k]),
            "reference_deg_mean": _deg(sum(p[3][k] for p in pairs) / len(pairs)),
        }
        if len(diffs) > 1:
            entry["min_abs_diff_deg"] = _deg(min(diffs))
            entry["max_abs_diff_deg"] = _deg(max(diffs))
            entry["per_pair_abs_diff_deg"] = [_deg(d) for d in diffs]
        angles.append(entry)
    out["angles"] = angles
    return out


def frame_angle_comparison(obs, manifest: dict, reference: dict, frozen_config: dict) -> dict:
    """VIDEO-REVIEW-ONLY comparison for the frames that were actually saved as JPEG.

    obs             the saved Observation for this trial
    manifest        the validated diagnostic capture manifest of the SAME trial
    reference       the already-prepared reference sample the trial is bound to. Used as given;
                    it is read, never rebuilt and never written to.
    frozen_config   the contract. Its cost keys and DTW band are used unchanged.

    Returns a strict-JSON-safe dict. Raises ReviewError when the inputs do not belong together.
    """
    if manifest.get("schema") != CAPTURE_SCHEMA:
        raise ReviewError(f"unexpected capture schema {manifest.get('schema')!r}")
    if manifest.get("status") not in CAPTURE_USABLE_STATES:
        raise ReviewError(f"capture status {manifest.get('status')!r} carries no frames")
    saved = manifest.get("frames") or []
    if not saved:
        raise ReviewError("capture manifest has no frames")
    obs_sha = (obs.sidecar or {}).get("observation_sha256")
    if not obs_sha or manifest.get("observation_sha256") != obs_sha:
        raise ReviewError("capture manifest does not belong to this observation")

    try:
        seq = F.sequence_features(obs, trim=True)
    except F.FeatureError as exc:
        return {"schema": SCHEMA, "status": "UNAVAILABLE", "reason": type(exc).__name__,
                "detail": str(exc), "frames": [], "is_a_verdict": False}

    lo, hi = int(seq["span"][0]), int(seq["span"][1])
    rows, duplicates = _identity_rows(obs)
    seen: dict[int, int] = {}
    for entry in saved:
        fid = _int_or_none(entry.get("frame_id"))
        if fid is not None:
            seen[fid] = seen.get(fid, 0) + 1

    # pass 1: decide what each saved frame IS, before anything is aligned
    decided = []
    for position, entry in enumerate(saved):
        state, reason, row = _classify(entry, rows, duplicates, obs, lo, hi, seen)
        decided.append({"position": position, "entry": entry, "state": state,
                        "reason": reason, "observation_row": row})

    eligible = [d for d in decided if d["state"] == "ELIGIBLE"]
    # ascending observation row: the saved frames are a time sequence, and DTW needs it to be one
    eligible.sort(key=lambda d: d["observation_row"])
    picked = [d["observation_row"] - lo for d in eligible]
    for slot, d in enumerate(eligible):
        d["query_row"] = slot
    picked_idx = np.asarray(picked, dtype=int)      # explicit: an empty list must index as empty

    query = {
        # every array is a fresh copy; seq and reference are inputs and stay untouched
        "features": seq["features"][picked_idx].copy(),
        "present": seq["present"][picked_idx].copy(),
        "timestamp_us": seq["timestamp_us"][picked_idx].copy(),
        "frame_id": seq["frame_id"][picked_idx].copy(),
        "picked_rows": picked_idx,
        "participating": dict(seq.get("participating") or {}),
        "feedback_hand_roles": seq.get("feedback_hand_roles"),
        "movement_observation": seq.get("movement_observation"),
        "selector": "saved_rgb_frames",
    }

    cost_cfg = cost_config(frozen_config)
    band = band_ratio(frozen_config)
    path: list[tuple[int, int]] = []
    align_reason = None
    if len(picked) >= 2:
        cost, _evidence = frame_cost_matrix(query, reference, **cost_cfg)
        path = _optimal_path(cost, band)
        if not path:
            align_reason = "NO_ADMISSIBLE_WARPING_PATH"
    else:
        align_reason = "FEWER_THAN_TWO_SAVED_FRAMES_INSIDE_THE_SPAN"

    matches: dict[int, list[int]] = {}
    for i, j in path:
        matches.setdefault(i, []).append(j)

    gi = {g: n for n, g in enumerate(GROUP_ORDER)}
    qf, qp = query["features"], query["present"]
    rf, rp = reference["features"], reference["present"]

    for d in eligible:
        i = d["query_row"]
        js = matches.get(i, [])
        if not js:
            d["state"] = "UNALIGNED"
            d["reason"] = align_reason or "QUERY_FRAME_NOT_ON_THE_WARPING_PATH"
            continue
        hands, compared_any = {}, False
        for side in ("L", "R"):
            group = f"bend_{side}"
            index = gi[group]
            q_ok = _hand_comparable(qf, qp, index, group, i)
            pairs, missing = [], {"compared": 0, "query_group_missing": 0,
                                  "reference_group_missing": 0}
            for j in js:
                r_ok = _hand_comparable(rf, rp, index, group, j)
                if not q_ok:
                    missing["query_group_missing"] += 1
                    continue
                if not r_ok:
                    missing["reference_group_missing"] += 1
                    continue
                start = _SLICES[group].start
                q_deg = [float(qf[i, start + k]) * DEGREES_PER_UNIT
                         for k in range(INTERIOR_ANGLE_COUNT)]
                r_deg = [float(rf[j, start + k]) * DEGREES_PER_UNIT
                         for k in range(INTERIOR_ANGLE_COUNT)]
                pairs.append((int(j), _pair_angles(qf[i], rf[j], group), q_deg, r_deg))
                missing["compared"] += 1
            hands[side] = _aggregate(side, pairs, len(js), missing)
            compared_any = compared_any or bool(pairs)
        d["hands"] = hands
        d["reference_rows"] = [int(j) for j in js]
        d["state"] = "COMPARED" if compared_any else "NO_ANGLE_OBSERVED"
        d["reason"] = None if compared_any else "NO_MATCHED_PAIR_CARRIED_A_HAND_ANGLE_GROUP"

    frames = []
    for d in decided:
        entry = d["entry"]
        row = {
            "index": d["position"],
            "frame_id": _int_or_none(entry.get("frame_id")),
            "timestamp_us": _int_or_none(entry.get("timestamp_us")),
            "manifest_observation_row": _int_or_none(entry.get("observation_row")),
            "observation_row": d["observation_row"],
            "query_row": d.get("query_row"),
            "reference_rows": d.get("reference_rows", []),
            "state": d["state"],
            "reason": d["reason"],
            "comparison_available": d["state"] == "COMPARED",
            "hands": d.get("hands", {}),
        }
        frames.append(row)

    counts = {s: sum(1 for r in frames if r["state"] == s) for s in FRAME_STATES}
    return {
        "schema": SCHEMA,
        "status": "OK",
        "is_a_verdict": False,
        "unit": "degrees",
        "value_meaning": "absolute difference between the participant's and the reference's "
                         "PROJECTED 2D angle at that finger joint. Not a score, not a probability, "
                         "not a correctness judgement, and not scaled by any participant statistic.",
        "value_range": {"min": 0.0, "max": 180.0,
                        "note": "the arithmetic range of a difference of two angles; no display "
                                "saturation is decided here"},
        "observation": {
            "sha256": obs_sha,
            "frames": int(obs.n_frames),
            "analysis_span": [lo, hi],
            "span_rule": "features.active_span on the trimmed sequence; a saved frame outside it "
                         "stays outside and is never matched to a neighbouring frame",
        },
        "reference": {
            "clip_id": reference.get("clip_id"),
            "word_id": reference.get("word_id"),
            "selector": reference.get("selector"),
            "sampled_rows": int(len(rf)),
            "prepared": "the reference sample already prepared for this trial; it was used as "
                        "given and was not rebuilt, resampled or modified",
        },
        "alignment": {
            "method": "frame_cost_matrix + _optimal_path, the existing policy",
            "band_ratio": band,
            "cost_config": cost_cfg,
            "query_selector": "saved_rgb_frames",
            "query_rows": len(picked),
            "path_length": len(path),
            "unaligned_reason": align_reason,
            "scope": "one additional one-reference DTW for drawing only. The trial's ranking, "
                     "distance, thresholds and any neural reading are untouched.",
            "not_the_classifier_path": "the classifier compares a uniform-time sample against "
                                       "every reference; this compares the saved RGB frames "
                                       "against one reference",
        },
        "angle_map": ANGLE_MAP,
        "angles_per_hand": INTERIOR_ANGLE_COUNT,
        "neutral_landmarks": NEUTRAL_LANDMARKS,
        "unmapped_group_indices": UNMAPPED_GROUP_INDICES,
        "hand_point_base": dict(HAND_BASE),
        "group_rule": "bend_* is all-or-nothing: a hand contributes only when the whole group is "
                      "present and every one of its 19 components is finite on BOTH sides, the "
                      "same rule the cost path uses. A missing hand is missing, never a zero.",
        "state_meaning": {
            "COMPARED": "this saved frame is inside the span, was aligned, and at least one hand "
                        "angle group was comparable on a matched pair",
            "NO_ANGLE_OBSERVED": "aligned, but no matched pair carried a hand angle group on both "
                                 "sides",
            "UNALIGNED": "inside the span but no warping-path pair reached it",
            "OUTSIDE_SPAN": "outside the trimmed analysis span; deliberately not compared",
            "IDENTITY_REJECTED": "the saved frame's identity does not agree with the observation; "
                                 "it is never matched to a different row",
        },
        "counts": {**counts, "saved_frames": len(frames)},
        "frames": frames,
    }
