"""Inspect an explicit interval on the existing DTW path, without refitting or judging it.

The interval is a reviewer's selection, NOT an automatically identified linguistic keyphase.
Body coordinates still use shoulder scale. Visibility is not physical body calibration.
"""
from __future__ import annotations

import hashlib
import numpy as np

# Default-path policy for the observational detail blocks: the selected-interval report on
# location_path and the full-rate movement observation beside axis_deviation. Both are additive
# fields read from data the features already prepared; neither feeds a distance, rank or verdict.
DETAIL_POLICY = "observational_detail_v1"

ANCHORS = {"nose": (0,), "mouth": (9, 10), "shoulder_L": (11,),
           "shoulder_R": (12,), "elbow_L": (13,), "elbow_R": (14,)}
FINGERS = ("thumb", "index", "middle", "ring", "little")


def body_context(obs, origin, scale, lo, hi):
    """Observed upper-body anchors, in the SAME coordinate frame as handpos, with no imputation."""
    ids = slice(lo, hi + 1)
    xy, usable = obs.xy_px[ids], obs.usable[ids]
    shoulder_ok = (usable[:, [11, 12]].all(axis=1)
                   & np.isfinite(xy[:, [11, 12]]).all(axis=(1, 2)))
    scale_ok = bool(np.isfinite(scale) and scale > 1e-6)
    torso_ok = (shoulder_ok & usable[:, [23, 24]].all(axis=1)
                & np.isfinite(xy[:, [23, 24]]).all(axis=(1, 2)))
    points = {}
    for name, joints in ANCHORS.items():
        valid = (shoulder_ok & usable[:, joints].all(axis=1)
                 & np.isfinite(xy[:, joints]).all(axis=(1, 2)) & scale_ok)
        pts = (xy[:, joints].mean(axis=1) - origin[ids]) / scale if scale_ok else None
        points[name] = [pts[t].astype(float).tolist() if valid[t] else None
                        for t in range(hi - lo + 1)]
    return {"schema": "observed_upper_body_context_v1", "points": points,
            "normalization": {"method": "shoulder_midpoint_and_clip_median_width",
                "span_frames": hi - lo + 1, "paired_shoulders_frames": int(shoulder_ok.sum()),
                "paired_torso_frames": int(torso_ok.sum()),
                "arm_length_calibrated": False, "torso_length_calibrated": False,
                "tilt_removed": False,
                "meaning": "2D projected body relations; shoulder scale only. Observed hips "
                           "do not imply that torso length was calibrated."}}


def sample_body_context(context, picks):
    if not context:
        return None
    return {"schema": context["schema"], "normalization": dict(context["normalization"]),
            "points": {k: [v[int(i)] for i in picks] for k, v in context["points"].items()}}


def _group(sample, row, group):
    from .dtw import _SLICES
    from .features import GROUP_ORDER
    value = sample["features"][row, _SLICES[group]]
    if not sample["present"][row, GROUP_ORDER.index(group)] or not np.isfinite(value).all():
        return None
    return value.astype(float)


def _pair(a, b, multiplier=1.0):
    return {"query": (a * multiplier).tolist() if a is not None else None,
            "reference": (b * multiplier).tolist() if b is not None else None,
            "delta": ((a - b) * multiplier).tolist() if a is not None and b is not None else None}


def bind_review_window(definition, reference_frames, clip_id, word_id):
    """Bind an authored visual-review window to this exact reference representation.

    Never use another word's interval or silently turn a stale definition into a full-sign claim.
    """
    if definition is None:
        return {"status": "NOT_CONFIGURED"}
    if not isinstance(definition, dict):
        return {"status": "UNAVAILABLE", "reason": "invalid_definition"}
    if definition.get("status") != "VISUALLY_INSPECTED_CENTRAL_WINDOW_NOT_EXPERT_ANNOTATION":
        return {"status": "UNAVAILABLE", "reason": "unsupported_annotation_status"}
    if definition.get("clip_id") != clip_id or definition.get("word_id") != word_id:
        return {"status": "UNAVAILABLE", "reason": "reference_identity_mismatch"}
    digest = definition.get("sampled_feature_sha256")
    if not digest or digest != reference_frames.get("sampled_feature_sha256"):
        return {"status": "UNAVAILABLE", "reason": "reference_features_changed"}
    bounds = definition.get("frame_id_range")
    if (not isinstance(bounds, list) or len(bounds) != 2
            or any(type(x) is not int for x in bounds) or bounds[0] < 0 or bounds[1] <= bounds[0]):
        return {"status": "UNAVAILABLE", "reason": "invalid_frame_range"}
    ids = reference_frames.get("frame_ids", [])
    if not ids or bounds[0] < ids[0] or bounds[1] > ids[-1]:
        return {"status": "UNAVAILABLE", "reason": "range_outside_reference"}
    selected = [i for i, frame in enumerate(ids) if bounds[0] <= frame <= bounds[1]]
    if len(selected) < 2:
        return {"status": "UNAVAILABLE", "reason": "too_few_reference_samples"}
    return {"status": "AVAILABLE", "reference_sample_range": [selected[0], selected[-1]],
            "authored_frame_id_range": bounds.copy(),
            "actual_sampled_frame_ids": [ids[i] for i in selected],
            "annotation_status": definition.get("status"),
            "scope": "central visual review only; not an expert-labelled complete sign",
            "is_a_verdict": False}


def inspect_path(query, reference, path, review_window=None):
    """Uses the supplied path exactly once; no DTW/NN call, threshold or word-specific rule."""
    frames = {}
    for who, sample in (("query", query), ("reference", reference)):
        ctx = sample.get("feedback_body_context") or {}
        fingerprint = hashlib.sha256()
        for key in ("features", "present", "timestamp_us", "frame_id"):
            arr = np.ascontiguousarray(sample.get(key, []))
            fingerprint.update(f"{key}:{arr.dtype.str}:{arr.shape}".encode())
            fingerprint.update(arr.tobytes())
        frames[who] = {"body_points": ctx.get("points", {}),
                       "normalization": ctx.get("normalization"),
                       "sampled_feature_sha256": fingerprint.hexdigest(),
                       "hash_scope": "sampled features, masks, timestamps, frame IDs; not raw video or NPZ bytes",
                       "frame_ids": [int(v) for v in sample.get("frame_id", [])],
                       # identity metadata only: the sampled rows' own observation timestamps,
                       # NOT relative_time_s (which subtracts each sequence's first sample)
                       "timestamp_us": [int(v) for v in sample.get("timestamp_us", [])],
                       "rows": len(sample["features"])}
    pairs = []
    for i, j in path:
        hands = {}
        for side in ("L", "R"):
            bend = [_group(s, t, "bend_" + side) for s, t in ((query, i), (reference, j))]
            fingers = {name: _pair(*(v[n * 3:n * 3 + 3] if v is not None else None
                                     for v in bend), multiplier=180.0)
                       for n, name in enumerate(FINGERS)}
            hands[side] = {
                "shoulder_position": _pair(_group(query, i, "handpos_" + side),
                                           _group(reference, j, "handpos_" + side)),
                "nose_position": _pair(_group(query, i, "facerel_" + side),
                                       _group(reference, j, "facerel_" + side)),
                "finger_angles_deg": fingers,
                "spread_angles_deg": _pair(*(v[15:19] if v is not None else None for v in bend),
                                           multiplier=180.0),
            }
        pairs.append({"i": int(i), "j": int(j), "hands": hands})
    return {"schema": "selected_interval_feedback_v1", "status": "AVAILABLE", "steps": pairs,
            "review_window": bind_review_window(review_window, frames["reference"],
                                                 reference.get("clip_id"), reference.get("word_id")),
            "inputs": frames, "reference_clip_id": reference.get("clip_id"),
            "reference_word_id": reference.get("word_id"),
            "angle_unit": "degrees of 2D projected interior angles, not anatomical 3D flexion",
            "location_unit": "shoulder widths; x video-right, y down; anatomical side labels",
            "selection_role": "exploratory reference-sample interval; not an annotated sign phase",
            "aggregation": "Within each reference sample, mean over comparable query matches; "
                           "then equal mean across observed reference samples. Not frame-time "
                           "weighted; missing samples excluded, never filled as zero.",
            "accepted": None, "execution_assessment": "UNCONFIGURED"}
