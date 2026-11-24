#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F9 candidate: per-finger pair descriptors added behind the unchanged F3R1 prefix.

The approved F3R1 module (``full_hand_pair_features_v1``) is imported unmodified and
supplies every geometric decision: the 86 feature points, the 5 groups, the equal-weight
grouped cost, the banded DTW with its pinned tie order, and the descriptor definition.
This module only SLICES the local hand-shape distances that F3R1 has already computed,
on the SAME single path, into 10 four-joint finger groups.

    fixed_time_features   15 F3R1 values + 30 finger values = 45   (DTW called 0 times)
    aligned_pair_features 18 F3R1 values + 30 finger values = 48   (DTW called 1 time)

The first 15 / 18 values are produced by the baseline functions themselves, so they match
the baseline output element for element, NaN included. Nothing is renormalized, reweighted
or thresholded here.

The 30 added values are ordered left hand then right hand, and within a hand
thumb, index, middle, ring, pinky. Each finger is the 4 wrist-relative joints of that
finger inside the existing local group (left local 46..65, right local 66..85); the wrist
is not duplicated into any finger group and the global points are untouched. Per finger:
residual mean, residual p90 and shared-observation fraction over the path, the fraction
having denominator ``len(path) * 4`` and lying in 0..1.

Missingness is never turned into agreement: a finger with no jointly valid observation on
the path reports NaN residuals and coverage 0.0, exactly like an unavailable F3R1 group.

SCHEMA_ID is new because the vector width and layout are new. It does NOT change the
FeatureSequence schema, the F3R1 vector, any score boundary or any trained model, and any
model consuming this width has to be fitted separately. No recognition or teaching-quality
improvement is claimed or measured here.
"""

from __future__ import annotations

import os
import sys
from typing import List, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_INPUT_DIR = os.path.join(os.path.dirname(_HERE), "input")
if _INPUT_DIR not in sys.path:
    sys.path.insert(0, _INPUT_DIR)

import full_hand_pair_features_v1 as baseline   # noqa: E402  (unmodified F3R1 baseline)

# Re-exported so a caller needs one import only. These objects stay the baseline's.
FeatureError = baseline.FeatureError
FeatureSequence = baseline.FeatureSequence
PairResult = baseline.PairResult
build_feature_sequence = baseline.build_feature_sequence

SCHEMA_ID = "wis_finger_pair_v2"          # the PAIR FEATURE VECTOR schema, new width
SEQUENCE_SCHEMA_ID = baseline.SCHEMA_ID   # the accepted FeatureSequence schema, unchanged
BASELINE_SCHEMA_ID = baseline.SCHEMA_ID

HAND_ORDER: Tuple[str, ...] = ("left", "right")
FINGER_ORDER: Tuple[str, ...] = ("thumb", "index", "middle", "ring", "pinky")
FINGER_JOINT_COUNT = 4
FINGER_COUNT = len(HAND_ORDER) * len(FINGER_ORDER)

# (group index in baseline.GROUPS, first column inside that group).
FINGER_SLICES: Tuple[Tuple[int, int], ...] = tuple(
    (group_index, finger_index * FINGER_JOINT_COUNT)
    for group_index in baseline.LOCAL_GROUP_INDICES
    for finger_index in range(len(FINGER_ORDER)))

# The same slices as absolute feature-point ranges, for documentation and cross-checking.
FINGER_POINT_RANGES: Tuple[Tuple[str, int, int], ...] = tuple(
    ("%s_%s" % (hand, finger),
     baseline.GROUPS[group_index][1] + finger_index * FINGER_JOINT_COUNT,
     baseline.GROUPS[group_index][1] + (finger_index + 1) * FINGER_JOINT_COUNT)
    for hand, group_index in zip(HAND_ORDER, baseline.LOCAL_GROUP_INDICES)
    for finger_index, finger in enumerate(FINGER_ORDER))

FINGER_FEATURE_NAMES: Tuple[str, ...] = tuple(
    "finger_%s_%s_%s" % (hand, finger, statistic)
    for hand in HAND_ORDER
    for finger in FINGER_ORDER
    for statistic in ("residual_mean", "residual_p90", "joint_valid_fraction"))
FINGER_FEATURE_COUNT = len(FINGER_FEATURE_NAMES)                       # 30

FIXED_FEATURE_NAMES = baseline.FIXED_FEATURE_NAMES + FINGER_FEATURE_NAMES
ALIGNED_FEATURE_NAMES = baseline.ALIGNED_FEATURE_NAMES + FINGER_FEATURE_NAMES
FIXED_FEATURE_COUNT = len(FIXED_FEATURE_NAMES)                         # 45
ALIGNED_FEATURE_COUNT = len(ALIGNED_FEATURE_NAMES)                     # 48
BASELINE_FIXED_PREFIX = baseline.FIXED_FEATURE_COUNT                   # 15
BASELINE_ALIGNED_PREFIX = baseline.ALIGNED_FEATURE_COUNT               # 18


class InvalidFeatureSequence(FeatureError):
    """A structurally unusable input. Raised, so it is never confused with a pair failure
    such as NO_SHARED_HAND_SHAPE or NO_VALID_DTW_PATH, which stay ordinary results."""


def validate_sequence(sequence, label: str = "sequence") -> None:
    """Schema, shape and the masked-2D contract. Mask-false slots are never read and may
    hold NaN or anything else."""
    if not isinstance(sequence, baseline.FeatureSequence):
        raise InvalidFeatureSequence("%s is not a FeatureSequence" % label)
    if sequence.schema != SEQUENCE_SCHEMA_ID:
        raise InvalidFeatureSequence(
            "%s has schema %r, expected %r" % (label, sequence.schema, SEQUENCE_SCHEMA_ID))
    points, mask = sequence.points, sequence.mask
    if not isinstance(points, np.ndarray) or not isinstance(mask, np.ndarray):
        raise InvalidFeatureSequence("%s points and mask must be numpy arrays" % label)
    shape = (baseline.RESAMPLE_T, baseline.FEATURE_POINT_COUNT, baseline.COORD_DIMS)
    if points.shape != shape:
        raise InvalidFeatureSequence(
            "%s points must be %s, got %s" % (label, shape, points.shape))
    if mask.shape != shape[:2]:
        raise InvalidFeatureSequence(
            "%s mask must be %s, got %s" % (label, shape[:2], mask.shape))
    if mask.dtype != np.bool_:
        raise InvalidFeatureSequence("%s mask must be boolean" % label)
    if not np.issubdtype(points.dtype, np.floating):
        raise InvalidFeatureSequence("%s points must be floating point" % label)
    observed = points[mask]
    if not np.all(np.isfinite(observed[:, :2])):
        raise InvalidFeatureSequence("%s has a non-finite xy at a valid slot" % label)
    if np.any(observed[:, 2] != 0.0):
        raise InvalidFeatureSequence(
            "%s has a non-zero z at a valid slot; this contract is masked 2D" % label)


def finger_descriptors(geometry, path: Sequence[Tuple[int, int]]) -> np.ndarray:
    """30 values from the distances the baseline geometry already holds, on ONE path.

    No DTW is run here and no per-finger alignment exists: every finger reads the rows of
    the single path it is given."""
    rows = np.asarray(path, dtype=np.int64)
    length = int(rows.shape[0])
    values: List[float] = []
    for group_index, start in FINGER_SLICES:
        stop = start + FINGER_JOINT_COUNT
        distance = geometry.distances[group_index][rows[:, 0], rows[:, 1], start:stop]
        together = geometry.joint[group_index][rows[:, 0], rows[:, 1], start:stop]
        pool = distance[together]
        if pool.size:
            values.extend([float(pool.mean()), float(np.percentile(pool, 90))])
        else:
            values.extend([float("nan"), float("nan")])   # absent, never agreement
        values.append(float(together.sum()) / float(length * FINGER_JOINT_COUNT))
    return np.asarray(values, dtype=np.float64)


def fixed_time_features(a: FeatureSequence, b: FeatureSequence) -> PairResult:
    """ML-only: the baseline 15 on the fixed diagonal plus the 30 finger values. DTW is
    called 0 times and the sequences are never aligned."""
    validate_sequence(a, "a")
    validate_sequence(b, "b")
    geometry = baseline.pair_geometry(a, b)
    path = baseline.diagonal_path(baseline.RESAMPLE_T)
    if baseline.shared_local_hand_points(geometry, path) < 1:
        return PairResult(False, baseline.NO_SHARED_HAND_SHAPE)
    features = np.concatenate([baseline.descriptors_on_path(geometry, path),
                               finger_descriptors(geometry, path)])
    if features.size != FIXED_FEATURE_COUNT:
        return PairResult(False, "FIXED_FEATURE_WIDTH")
    return PairResult(True, None, features, float("nan"), baseline.RESAMPLE_T)


def aligned_pair_features(a: FeatureSequence, b: FeatureSequence) -> PairResult:
    """Hybrid: the baseline 18 plus the 30 finger values. The ONE shared DTW path is
    computed once, by the baseline rule, and reused by all 10 fingers."""
    validate_sequence(a, "a")
    validate_sequence(b, "b")
    geometry = baseline.pair_geometry(a, b)
    path, normalized = baseline.dtw_path(geometry.cost)          # the only DTW call
    if path is None or len(path) < 2:
        return PairResult(False, "NO_VALID_DTW_PATH")
    if baseline.shared_local_hand_points(geometry, path) < 1:
        return PairResult(False, baseline.NO_SHARED_HAND_SHAPE)
    features = np.concatenate([baseline.descriptors_on_path(geometry, path),
                               baseline.path_statistics(geometry, path),
                               finger_descriptors(geometry, path)])
    if features.size != ALIGNED_FEATURE_COUNT:
        return PairResult(False, "ALIGNED_FEATURE_WIDTH")
    return PairResult(True, None, features, normalized, len(path))
