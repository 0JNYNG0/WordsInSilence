"""Adapter: recorded MediaPipe 2D frames -> approved full-hand normalizer input.

Pure stdlib, no I/O. Emits one OpenPose-style frame in PIXEL XY plus
diagnostics. No quality assessment and no anatomical verification;
declared sides are copied through unverified.
"""
import math
from collections.abc import Mapping, Sequence

SCHEMA = "wis_multiview_review_adapter_v1"
POSE_MAP = {11: 5, 12: 2, 13: 6, 14: 3, 15: 7, 16: 4, 23: 12, 24: 9}
POSE_LEN = 33
BODY25_LEN = 25
HAND_LEN = 21
SIDES = ("left", "right")
RAW_LABELS = {"Left": "left", "Right": "right"}


def _is_num(value):
    ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    return ok and math.isfinite(value)


def _is_seq(value):
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _check_resolution(resolution):
    if not _is_seq(resolution) or len(resolution) != 2:
        raise ValueError("resolution must be [width, height]")
    width, height = resolution
    for value in (width, height):
        if not _is_num(value) or value <= 0:
            raise ValueError("resolution values must be finite and positive")
    return float(width), float(height)


def _triple(x, y, conf, width, height):
    """Return a pixel triple, or zeros when the observation is unusable."""
    if not (_is_num(x) and _is_num(y) and _is_num(conf)) or conf <= 0:
        return (0.0, 0.0, 0.0)
    return (float(x) * width, float(y) * height, float(conf))


def _raw_landmark(entry, length):
    if not _is_seq(entry) or len(entry) != length:
        raise ValueError("malformed landmark entry")
    return entry


def _pose_rows(pose, format_kind):
    """Yield (x, y, visibility) per MediaPipe pose index, validating shape."""
    if not _is_seq(pose) or len(pose) != POSE_LEN:
        raise ValueError("pose must hold %d landmarks" % POSE_LEN)
    rows = []
    for entry in pose:
        if format_kind == "raw":
            entry = _raw_landmark(entry, 4)
            rows.append((entry[0], entry[1], entry[3]))
        else:
            if not isinstance(entry, Mapping):
                raise ValueError("enriched pose landmarks must be mappings")
            rows.append((entry.get("norm_x"), entry.get("norm_y"),
                         entry.get("visibility")))
    return rows


def _hand_rows(landmarks, score, format_kind):
    """Yield (x, y, score) per hand joint; score is shared by all 21 joints."""
    if not _is_seq(landmarks) or len(landmarks) != HAND_LEN:
        raise ValueError("hand must hold %d landmarks" % HAND_LEN)
    rows = []
    for entry in landmarks:
        if format_kind == "raw":
            entry = _raw_landmark(entry, 3)
            rows.append((entry[0], entry[1], score))
        else:
            if not isinstance(entry, Mapping):
                raise ValueError("enriched hand landmarks must be mappings")
            rows.append((entry.get("norm_x"), entry.get("norm_y"), score))
    return rows


def _collect_raw_hands(hands):
    """Group raw hands by declared handedness without any score-based choice."""
    if not _is_seq(hands):
        raise ValueError("raw hands must be a list or None")
    grouped = {"left": [], "right": []}
    for hand in hands:
        if not isinstance(hand, Mapping):
            raise ValueError("raw hand entries must be mappings")
        label = hand.get("handedness")
        if label not in RAW_LABELS:
            raise ValueError("unknown handedness label: %r" % (label,))
        rows = _hand_rows(hand.get("norm"), hand.get("score"), "raw")
        grouped[RAW_LABELS[label]].append(rows)
    return grouped


def _collect_enriched_hands(hands):
    if not isinstance(hands, Mapping):
        raise ValueError("enriched hands must be a mapping or None")
    unknown = set(hands) - set(SIDES)
    if unknown:
        raise ValueError("unknown hand side keys: %s" % sorted(unknown))
    grouped = {"left": [], "right": []}
    for side in SIDES:
        hand = hands.get(side)
        if hand is None:
            continue
        if not isinstance(hand, Mapping):
            raise ValueError("enriched hand must be a mapping or None")
        declared = hand.get("anatomical_side")
        if declared not in SIDES:
            raise ValueError("unknown anatomical_side: %r" % (declared,))
        if declared != side:
            raise ValueError("anatomical_side %r under key %r" % (declared, side))
        grouped[side].append(
            _hand_rows(hand.get("landmarks"), hand.get("mp_score"), "enriched")
        )
    return grouped


def adapt_frame(frame, resolution, format_kind):
    """Adapt one recorded MediaPipe frame to normalizer input + diagnostics."""
    width, height = _check_resolution(resolution)
    if format_kind not in ("raw", "enriched"):
        raise ValueError("format_kind must be 'raw' or 'enriched'")
    if not isinstance(frame, Mapping):
        raise ValueError("frame must be a mapping")

    pose = frame.get("pose_norm") if format_kind == "raw" else frame.get("pose")
    hands = frame.get("hands")

    pose_rows = None if pose is None else _pose_rows(pose, format_kind)
    if hands is None:
        grouped = {"left": [], "right": []}
    elif format_kind == "raw":
        grouped = _collect_raw_hands(hands)
    else:
        grouped = _collect_enriched_hands(hands)

    ambiguous = sorted(s for s in SIDES if len(grouped[s]) > 1)

    diagnostics = {
        "schema": SCHEMA,
        "quality": "UNASSESSED",
        "enrollment_eligible": False,
        "coordinate_space": "pixel_xy",
        "depth_used": False,
        "side_policy": ("raw_declared_unverified" if format_kind == "raw"
                        else "enriched_declared_anatomical_unverified"),
        "ambiguous_sides": ambiguous,
        "hand_confidence_basis": "detector_score_shared_across_21_joints",
        "copied_pose_joints": 0,
        "copied_hand_joints": {"left": 0, "right": 0},
    }
    detected = bool(grouped["left"] or grouped["right"])
    if pose is None and not detected:
        return {"people": []}, diagnostics

    person = {"pose_keypoints_2d": [0.0] * (BODY25_LEN * 3)}
    if pose_rows is not None:
        for mp_index, body_index in POSE_MAP.items():
            triple = _triple(*pose_rows[mp_index], width=width, height=height)
            if triple[2] > 0:
                diagnostics["copied_pose_joints"] += 1
            person["pose_keypoints_2d"][body_index * 3:body_index * 3 + 3] = triple

    for side in SIDES:
        values = [0.0] * (HAND_LEN * 3)
        if len(grouped[side]) == 1:
            for index, row in enumerate(grouped[side][0]):
                triple = _triple(*row, width=width, height=height)
                if triple[2] > 0:
                    diagnostics["copied_hand_joints"][side] += 1
                values[index * 3:index * 3 + 3] = triple
        person["hand_%s_keypoints_2d" % side] = values

    return {"people": [person]}, diagnostics
