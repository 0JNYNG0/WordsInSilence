#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F3 full-hand pair representation: 86 feature points, 5 groups, one shared DTW path.

Fixed before any result. Reusable by later work.

POINTS (86) = global 46 (F1 body-normalized, movement and location preserved)
              + left  local 20 (left  joints 1..20 relative to left  wrist, size-scaled)
              + right local 20 (right joints 1..20 relative to right wrist, size-scaled)

The local block is hand SHAPE with neither hand rotated. Its per-hand scalar is the median
of the valid positive wrist->MCP(5,9,13,17) distances over the source frames. That is a
per-capture geometric size normalization, NOT a learned personal profile and not evidence
of any personal improvement. Everything is 2D; source depth is untouched and no 3D bend
angle is inferred, so two shapes differing only out of plane can still look alike.

GROUPS, fixed order and equal weight, no tuning grid:
    0 global_pose        points 0..3     4
    1 global_left_hand   points 4..24    21
    2 global_right_hand  points 25..45   21
    3 local_left_hand    points 46..65   20
    4 local_right_hand   points 66..85   20
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SCHEMA_ID = "wis_full_hand_pair_v1"
F1_SCHEMA_ID = "wis_full_hand_common_v1"
SOURCE_POINT_COUNT = 46
FEATURE_POINT_COUNT = 86
COORD_DIMS = 3
HAND_JOINT_COUNT = 21
HAND_MCP_JOINTS = (5, 9, 13, 17)
MIN_HAND_SCALE = 1e-6
RESAMPLE_T = 32
DTW_BAND = 0.10
DTW_BAND_EPS = 1e-15

SOURCE_POSE = (0, 4)
SOURCE_LEFT = (4, 25)
SOURCE_RIGHT = (25, 46)

GROUPS: Tuple[Tuple[str, int, int], ...] = (
    ("global_pose", 0, 4),
    ("global_left_hand", 4, 25),
    ("global_right_hand", 25, 46),
    ("local_left_hand", 46, 66),
    ("local_right_hand", 66, 86),
)
GROUP_COUNT = len(GROUPS)

DESCRIPTOR_NAMES: Tuple[str, ...] = tuple(
    "%s_%s" % (name, statistic)
    for name, _, _ in GROUPS
    for statistic in ("residual_mean", "residual_p90", "joint_valid_fraction"))
DTW_DESCRIPTOR_NAMES = ("dtw_mean_path_cost", "dtw_path_length_ratio",
                        "dtw_non_diagonal_fraction")
FIXED_FEATURE_NAMES = DESCRIPTOR_NAMES
ALIGNED_FEATURE_NAMES = DESCRIPTOR_NAMES + DTW_DESCRIPTOR_NAMES
FIXED_FEATURE_COUNT = len(FIXED_FEATURE_NAMES)      # 15
ALIGNED_FEATURE_COUNT = len(ALIGNED_FEATURE_NAMES)  # 18


class FeatureError(ValueError):
    """A refused input. Never substituted with a usable value."""


def load_f1_schema(path: Optional[str] = None):
    """Optional read-only import of the F1 module, for schema cross-checking only."""
    candidates = []
    if path:
        candidates.append(path)
    if os.environ.get("WIS_F1_MODULE"):
        candidates.append(os.environ["WIS_F1_MODULE"])
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(os.path.dirname(here), "f1", "full_hand_common_v1.py"))
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            import importlib.util
            spec = importlib.util.spec_from_file_location("f1_full_hand_common", candidate)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            try:
                spec.loader.exec_module(module)
                if module.POINT_COUNT != SOURCE_POINT_COUNT or \
                        module.HAND_JOINT_COUNT != HAND_JOINT_COUNT or \
                        module.SCHEMA_ID != F1_SCHEMA_ID:
                    raise FeatureError("the F1 module does not match the pinned F1 schema")
            except BaseException:
                sys.modules.pop(spec.name, None)
                raise
            return module
    return None


# ---- representation --------------------------------------------------------

@dataclass
class FeatureSequence:
    schema: str
    points: np.ndarray            # [32, 86, 3], NaN where masked
    mask: np.ndarray              # [32, 86] bool
    group_available: Tuple[bool, ...]
    hand_scales: Dict[str, float]
    source_frames: int
    usable: bool
    failure: Optional[str] = None
    detail: Dict = field(default_factory=dict)


def hand_scale(coords: np.ndarray, mask: np.ndarray, base: int) -> Tuple[float, int]:
    """Median of the valid positive wrist->MCP distances over the source frames."""
    wrist = base
    distances: List[float] = []
    for joint in HAND_MCP_JOINTS:
        both = mask[:, wrist] & mask[:, base + joint]
        if not both.any():
            continue
        delta = coords[both, base + joint, :2] - coords[both, wrist, :2]
        values = np.sqrt((delta * delta).sum(axis=1))
        values = values[np.isfinite(values) & (values > MIN_HAND_SCALE)]
        distances.extend(float(value) for value in values)
    if not distances:
        return float("nan"), 0
    return float(np.median(np.asarray(distances, dtype=np.float64))), len(distances)


def resample_to_fixed(points: np.ndarray, mask: np.ndarray, t_target: int = RESAMPLE_T):
    """The approved mask-aware linear rule: both endpoints valid interpolates, one valid
    carries, neither stays missing. Masked slots hold NaN and are never read."""
    t_src = int(points.shape[0])
    count = int(points.shape[1])
    out = np.full((t_target, count, COORD_DIMS), np.nan, dtype=np.float64)
    out_mask = np.zeros((t_target, count), dtype=bool)
    for t in range(t_target):
        target = t * (t_src - 1) / max(t_target - 1, 1)
        low = int(math.floor(target))
        high = min(low + 1, t_src - 1)
        alpha = target - low
        valid_low, valid_high = mask[low], mask[high]
        both = valid_low & valid_high
        out[t][both] = (1.0 - alpha) * points[low][both] + alpha * points[high][both]
        only_low = valid_low & ~valid_high
        out[t][only_low] = points[low][only_low]
        only_high = ~valid_low & valid_high
        out[t][only_high] = points[high][only_high]
        out_mask[t] = valid_low | valid_high
    return out, out_mask


def build_feature_sequence(coordinates, mask) -> FeatureSequence:
    """F1 [T,46,3] masked 2D -> the 86-point feature sequence resampled to T=32."""
    coords = np.asarray(coordinates, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if coords.ndim != 3 or coords.shape[1] != SOURCE_POINT_COUNT or \
            coords.shape[2] != COORD_DIMS:
        raise FeatureError("coordinates must be [T, 46, 3]")
    if valid.shape != coords.shape[:2]:
        raise FeatureError("the mask shape does not match the coordinates")
    if coords.shape[0] == 0:
        raise FeatureError("an empty sequence carries no observation")
    if not np.all(np.isfinite(coords[valid][:, :2])):
        raise FeatureError("a valid point has a non-finite xy")
    if np.any(coords[valid][:, 2] != 0.0):
        raise FeatureError("a valid point has a non-zero z; this contract is masked 2D")

    frames = int(coords.shape[0])
    points = np.full((frames, FEATURE_POINT_COUNT, COORD_DIMS), np.nan, dtype=np.float64)
    built = np.zeros((frames, FEATURE_POINT_COUNT), dtype=bool)
    points[:, :SOURCE_POINT_COUNT, :] = np.where(valid[:, :, None], coords, np.nan)
    built[:, :SOURCE_POINT_COUNT] = valid

    scales: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for hand, base, local_start in (("left", SOURCE_LEFT[0], 46),
                                    ("right", SOURCE_RIGHT[0], 66)):
        scale, count = hand_scale(coords, valid, base)
        scales[hand] = scale
        counts[hand] = count
        if not math.isfinite(scale) or scale <= MIN_HAND_SCALE:
            continue          # the shape group stays missing; nothing is invented
        for joint in range(1, HAND_JOINT_COUNT):
            target = local_start + joint - 1
            pair = valid[:, base] & valid[:, base + joint]
            if not pair.any():
                continue
            relative = (coords[:, base + joint] - coords[:, base]) / scale
            points[pair, target] = relative[pair]
            built[pair, target] = True

    resampled, resampled_mask = resample_to_fixed(points, built)
    available = tuple(bool(resampled_mask[:, low:high].any()) for _, low, high in GROUPS)
    failure = None
    if not any(available):
        failure = "ALL_POINTS_MISSING"
    elif not (available[3] or available[4]):
        failure = "NO_LOCAL_HAND_SHAPE"
    return FeatureSequence(
        schema=SCHEMA_ID, points=resampled, mask=resampled_mask,
        group_available=available, hand_scales=scales, source_frames=frames,
        usable=failure is None, failure=failure,
        detail={"hand_scale_samples": counts,
                "group_valid_fraction": [float(resampled_mask[:, low:high].mean())
                                         for _, low, high in GROUPS]})


# ---- pair geometry, one shared time axis -----------------------------------

@dataclass
class PairGeometry:
    distances: List[np.ndarray]      # per group [32, 32, P]
    joint: List[np.ndarray]          # per group [32, 32, P] bool
    cost: np.ndarray                 # [32, 32], inf where no group overlaps
    available_groups: np.ndarray     # [32, 32] int, groups contributing to that cell


def pair_geometry(a: FeatureSequence, b: FeatureSequence) -> PairGeometry:
    """Equal-weight mean over the AVAILABLE groups. No group is given a fake zero."""
    distances, joints, means = [], [], []
    for _, low, high in GROUPS:
        pa, pb = a.points[:, low:high, :2], b.points[:, low:high, :2]
        ma, mb = a.mask[:, low:high], b.mask[:, low:high]
        delta = pa[:, None, :, :] - pb[None, :, :, :]
        distance = np.sqrt((delta * delta).sum(axis=3))
        together = ma[:, None, :] & mb[None, :, :]
        distance = np.where(together, distance, np.nan)
        count = together.sum(axis=2)
        total = np.where(together, np.where(together, distance, 0.0), 0.0).sum(axis=2)
        mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
        distances.append(distance)
        joints.append(together)
        means.append(mean)
    stacked = np.stack(means, axis=0)
    contributing = np.isfinite(stacked).sum(axis=0)
    cost = np.where(contributing > 0,
                    np.where(np.isfinite(stacked), stacked, 0.0).sum(axis=0)
                    / np.maximum(contributing, 1),
                    np.inf)
    return PairGeometry(distances, joints, cost, contributing)


def diagonal_path(length: int = RESAMPLE_T) -> List[Tuple[int, int]]:
    return [(index, index) for index in range(length)]


def dtw_path(cost: np.ndarray):
    """Banded DTW with a pinned tie order: diagonal, then vertical, then horizontal."""
    rows, columns = cost.shape
    accumulated = np.full((rows + 1, columns + 1), np.inf, dtype=np.float64)
    accumulated[0, 0] = 0.0
    for i in range(1, rows + 1):
        ri = (i - 1) / max(rows - 1, 1)
        for j in range(1, columns + 1):
            rj = (j - 1) / max(columns - 1, 1)
            if abs(ri - rj) > DTW_BAND + DTW_BAND_EPS:
                continue
            local = cost[i - 1, j - 1]
            if not np.isfinite(local):
                continue                       # unusable cell, never a zero distance
            best = accumulated[i - 1, j - 1]
            if accumulated[i - 1, j] < best:
                best = accumulated[i - 1, j]
            if accumulated[i, j - 1] < best:
                best = accumulated[i, j - 1]
            accumulated[i, j] = local + best
    if not np.isfinite(accumulated[rows, columns]):
        return None, float("nan")
    i, j = rows, columns
    path: List[Tuple[int, int]] = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        diagonal = accumulated[i - 1, j - 1]
        vertical = accumulated[i - 1, j]
        horizontal = accumulated[i, j - 1]
        best, bi, bj = diagonal, i - 1, j - 1
        if vertical < best:
            best, bi, bj = vertical, i - 1, j
        if horizontal < best:
            best, bi, bj = horizontal, i, j - 1
        i, j = bi, bj
    path.reverse()
    return path, float(accumulated[rows, columns] / max(len(path), 1))


def descriptors_on_path(geometry: PairGeometry, path: Sequence[Tuple[int, int]]):
    """15 values: per group residual mean, residual p90 and joint valid fraction.

    A group with no jointly valid comparison anywhere on the path is UNAVAILABLE: NaN
    residuals and coverage 0. It is never reported as agreement."""
    rows = np.asarray(path, dtype=np.int64)
    length = int(rows.shape[0])
    values: List[float] = []
    for index, (_, low, high) in enumerate(GROUPS):
        width = high - low
        distance = geometry.distances[index][rows[:, 0], rows[:, 1], :]
        together = geometry.joint[index][rows[:, 0], rows[:, 1], :]
        pool = distance[together]
        if pool.size:
            values.extend([float(pool.mean()), float(np.percentile(pool, 90))])
        else:
            values.extend([float("nan"), float("nan")])
        values.append(float(together.sum()) / float(length * width))
    return np.asarray(values, dtype=np.float64)


def path_statistics(geometry: PairGeometry, path: Sequence[Tuple[int, int]]):
    rows = np.asarray(path, dtype=np.int64)
    local = geometry.cost[rows[:, 0], rows[:, 1]]
    steps = np.diff(rows, axis=0)
    non_diagonal = int(np.sum(~((steps[:, 0] == 1) & (steps[:, 1] == 1)))) \
        if steps.size else 0
    denominator = max(len(path) - 1, 1)
    return np.asarray([float(np.mean(local)), len(path) / float(RESAMPLE_T),
                       non_diagonal / float(denominator)], dtype=np.float64)


@dataclass
class PairResult:
    ok: bool
    failure: Optional[str] = None
    features: Optional[np.ndarray] = None
    dtw_cost: float = float("nan")
    path_length: int = 0


LOCAL_GROUP_INDICES = (3, 4)
NO_SHARED_HAND_SHAPE = "NO_SHARED_HAND_SHAPE"


def shared_local_hand_points(geometry: PairGeometry,
                             path: Sequence[Tuple[int, int]]) -> int:
    """Jointly valid local hand-shape comparisons on the given path, either hand."""
    rows = np.asarray(path, dtype=np.int64)
    total = 0
    for index in LOCAL_GROUP_INDICES:
        total += int(geometry.joint[index][rows[:, 0], rows[:, 1], :].sum())
    return total


def fixed_time_features(a: FeatureSequence, b: FeatureSequence) -> PairResult:
    """ML-only: the 15 descriptors on the fixed diagonal. It never aligns."""
    geometry = pair_geometry(a, b)
    path = diagonal_path(RESAMPLE_T)
    if shared_local_hand_points(geometry, path) < 1:
        return PairResult(False, NO_SHARED_HAND_SHAPE)
    features = descriptors_on_path(geometry, path)
    if features.size != FIXED_FEATURE_COUNT:
        return PairResult(False, "FIXED_FEATURE_WIDTH")
    return PairResult(True, None, features, float("nan"), RESAMPLE_T)


def aligned_pair_features(a: FeatureSequence, b: FeatureSequence) -> PairResult:
    """Hybrid: the same 15 descriptors on the ONE shared DTW path, plus 3 path statistics."""
    geometry = pair_geometry(a, b)
    path, normalized = dtw_path(geometry.cost)
    if path is None or len(path) < 2:
        return PairResult(False, "NO_VALID_DTW_PATH")
    if shared_local_hand_points(geometry, path) < 1:
        return PairResult(False, NO_SHARED_HAND_SHAPE)
    features = np.concatenate([descriptors_on_path(geometry, path),
                               path_statistics(geometry, path)])
    if features.size != ALIGNED_FEATURE_COUNT:
        return PairResult(False, "ALIGNED_FEATURE_WIDTH")
    return PairResult(True, None, features, normalized, len(path))


def dtw_distance(a: FeatureSequence, b: FeatureSequence) -> PairResult:
    """DTW-only: the normalized grouped path cost, no learner involved."""
    geometry = pair_geometry(a, b)
    path, normalized = dtw_path(geometry.cost)
    if path is None:
        return PairResult(False, "NO_VALID_DTW_PATH")
    if shared_local_hand_points(geometry, path) < 1:
        return PairResult(False, NO_SHARED_HAND_SHAPE)
    return PairResult(True, None, None, normalized, len(path))
