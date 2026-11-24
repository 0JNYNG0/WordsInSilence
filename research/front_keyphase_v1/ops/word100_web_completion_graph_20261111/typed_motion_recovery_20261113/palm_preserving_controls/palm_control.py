"""Pure NumPy synthetic attribute intervention; no models, files, or camera.

Only PIP/DIP/TIP of the four non-thumb fingers may change. This does not
generate real incorrect KSL or certify anatomical validity/visibility. A caller
must supply a separately reviewed donor and retain the synthetic_point mask.
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

MCPS = (5, 9, 13, 17)
CHAINS = tuple((m, m + 1, m + 2, m + 3) for m in MCPS)
DISTAL = tuple(j for chain in CHAINS for j in chain[1:])
PROTECTED = (0, 1, 2, 3, 4, *MCPS)
REQUIRED = (0, *range(5, 21))
HAND_START = {"L": 91, "R": 112}
# Numerical degeneracy checks, not learned/semantic/word-dependent thresholds.
EPS = 1e-8
MIN_PALM_SINGULAR_RATIO = 1e-6


def _side(value: str) -> str:
    if value not in HAND_START:
        raise ValueError("side must be L or R")
    return value


def _hand(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value)
    if value.shape != (21, 2) or not np.issubdtype(value.dtype, np.floating):
        raise ValueError(f"{name} must be floating [21,2]")
    return value


def _observed(hand: np.ndarray, mask: np.ndarray | None, name: str) -> None:
    if not np.isfinite(hand[list(REQUIRED)]).all():
        raise ValueError(f"{name}: nonfinite required point")
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != (21,) or mask.dtype != np.bool_:
            raise ValueError(f"{name}: availability must be boolean [21]")
        if not mask[list(REQUIRED)].all():
            raise ValueError(f"{name}: unavailable required point")


def _palm_vectors(hand: np.ndarray, name: str) -> np.ndarray:
    palm = hand[list(MCPS)].astype(np.float64) - hand[0].astype(np.float64)
    singular = np.linalg.svd(palm, compute_uv=False)
    if singular[0] <= EPS or singular[1] / singular[0] <= MIN_PALM_SINGULAR_RATIO:
        raise ValueError(f"{name}: degenerate or edge-on projected palm")
    return palm


def replace_distal_four_fingers(
    donor: np.ndarray,
    target: np.ndarray,
    *,
    donor_side: str,
    target_side: str,
    donor_available: np.ndarray | None = None,
    target_available: np.ndarray | None = None,
) -> np.ndarray:
    """Copy donor projected bone directions onto target's original bone lengths.

    The donor is mirrored in its wrist-centred x basis for opposite handedness,
    then rotated by a proper 2-D palm Procrustes fit. Each finger starts at its
    OWN original MCP; only its next three vertices are assigned. Consequently
    wrist, MCP spread, thumb, and all target palm measurements are unchanged.
    Projected segment lengths are preserved to target dtype rounding, not 3-D
    bone anatomy. Invalid/edge-on geometry raises rather than inventing points.
    """
    donor = _hand(donor, "donor")
    target = _hand(target, "target")
    _side(donor_side)
    _side(target_side)
    _observed(donor, donor_available, "donor")
    _observed(target, target_available, "target")
    local = donor.astype(np.float64) - donor[0].astype(np.float64)
    if donor_side != target_side:
        local[:, 0] *= -1
    src_palm = _palm_vectors(local, "donor")
    dst_palm = _palm_vectors(target, "target")
    u, _, vt = np.linalg.svd(src_palm.T @ dst_palm)
    correction = np.eye(2)
    correction[-1, -1] = 1 if np.linalg.det(u @ vt) > 0 else -1
    rotation = u @ correction @ vt
    out = target.copy()
    for chain in CHAINS:
        points = list(chain)
        donor_bones = np.diff(local[points], axis=0)
        target_bones = np.diff(target[points].astype(np.float64), axis=0)
        donor_lengths = np.linalg.norm(donor_bones, axis=1)
        target_lengths = np.linalg.norm(target_bones, axis=1)
        if (donor_lengths <= EPS).any() or (target_lengths <= EPS).any():
            raise ValueError("degenerate projected finger segment")
        directions = donor_bones / donor_lengths[:, None] @ rotation
        chain_points = target[chain[0]].astype(np.float64) + np.cumsum(
            directions * target_lengths[:, None], axis=0
        )
        out[list(chain[1:])] = chain_points
    if not np.isfinite(out[list(REQUIRED)]).all():
        raise ValueError("replacement overflowed target dtype")
    return out


def point_available(xy: np.ndarray, width: int, height: int) -> np.ndarray:
    """Adapter-compatible retained-coordinate availability, NOT visibility."""
    return (np.isfinite(xy).all(-1) & (xy[..., 0] >= 0) & (xy[..., 0] < width)
            & (xy[..., 1] >= 0) & (xy[..., 1] < height))


def make_control(
    xy: np.ndarray,
    score: np.ndarray,
    timestamp_us: np.ndarray,
    frame_id: np.ndarray,
    donor: np.ndarray,
    *,
    side: str,
    donor_side: str,
    frame_indices: Iterable[int],
    width: int,
    height: int,
    donor_available: np.ndarray | None = None,
    source_available: np.ndarray | None = None,
) -> dict:
    """Return a new RTMW133 pack, leaving ineligible requested frames unchanged.

    Explicit frame_indices are array indices, never timestamps/frame IDs. No
    semantic interval is selected here. Caller masks may REMOVE availability,
    never fill missing coordinates. All selected fingers must be observed.
    The original scores are carried as metadata; they do not describe generated
    points. The synthetic_point mask is the intervention provenance, even when
    a replacement coordinate coincidentally equals its original coordinate.
    """
    xy, score = np.asarray(xy), np.asarray(score)
    ts, fid = np.asarray(timestamp_us), np.asarray(frame_id)
    n = len(xy)
    if n == 0 or xy.shape != (n, 133, 2) or not np.issubdtype(xy.dtype, np.floating):
        raise ValueError("xy must be nonempty floating [T,133,2]")
    if score.shape != (n, 133) or ts.shape != (n,) or fid.shape != (n,):
        raise ValueError("score/time/frame shape mismatch")
    if not np.issubdtype(ts.dtype, np.integer) or not np.issubdtype(fid.dtype, np.integer):
        raise ValueError("source timestamp_us and frame_id must be integer")
    if not (ts[1:] > ts[:-1]).all() or not (fid[1:] > fid[:-1]).all():
        raise ValueError("source times and frame IDs must strictly increase")
    if not np.isfinite([width, height]).all() or width <= 0 or height <= 0:
        raise ValueError("positive image dimensions required")
    _side(side)
    _side(donor_side)
    donor = _hand(donor, "donor")
    _observed(donor, donor_available, "donor")
    _palm_vectors(donor, "donor")
    indices = list(frame_indices)
    if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
           or i < 0 or i >= n for i in indices) or len(set(indices)) != len(indices):
        raise ValueError("frame_indices must contain unique in-range integers")
    available = point_available(xy, width, height)
    if source_available is not None:
        mask = np.asarray(source_available)
        if mask.shape != (n, 133) or mask.dtype != np.bool_:
            raise ValueError("source_available must be boolean [T,133]")
        available &= mask
    out = xy.copy()
    synthetic = np.zeros((n, 133), dtype=bool)
    frame_report = []
    start = HAND_START[side]
    for index in sorted(indices):
        index = int(index)
        target = xy[index, start:start + 21]
        try:
            replacement = replace_distal_four_fingers(
                donor, target, donor_side=donor_side, target_side=side,
                donor_available=donor_available,
                target_available=available[index, start:start + 21],
            )
            if not point_available(replacement, width, height)[list(REQUIRED)].all():
                raise ValueError("replacement would leave image bounds")
        except ValueError as error:
            frame_report.append({"index": index, "applied": False, "reason": str(error)})
            continue
        # Do not round-trip the protected coordinates through any calculation.
        positions = start + np.asarray(DISTAL)
        out[index, positions] = replacement[list(DISTAL)]
        synthetic[index, positions] = True
        frame_report.append({"index": index, "applied": True, "reason": None})
    return dict(
        xy_px=out, score=score.copy(), timestamp_us=ts.copy(), frame_id=fid.copy(),
        synthetic_point=synthetic, frame_report=frame_report, side=side,
        source_width=width, source_height=height,
        requested_frame_count=len(indices),
        applied_frame_count=sum(r["applied"] for r in frame_report),
        synthetic_attribute_control=True, real_incorrect_ksl=False,
        carried_score_is_synthetic_visibility_confidence=False,
    )


def rebuild_channel_pair(
    original_xy: np.ndarray,
    control: dict,
    *,
    convert: Callable,
) -> dict:
    """Recompute both views with the canonical adapter.convert supplied by caller.

    This function does not import a model or run quarantine. If a frozen source
    is already quarantined, supply that SAME source to make_control and here.
    A later guard that changes anchors invalidates the invariant and must be
    audited separately; it cannot be silently treated as a pure shape control.
    """
    before_xy = np.asarray(original_xy)
    after_xy = control["xy_px"]
    synthetic = control["synthetic_point"]
    if before_xy.shape != after_xy.shape or synthetic.shape != before_xy.shape[:-1]:
        raise ValueError("original/control shape mismatch")
    if before_xy[~synthetic].tobytes() != after_xy[~synthetic].tobytes():
        raise ValueError("control changed protected original coordinates")
    def build(xy):
        return convert(xy.copy(), control["score"].copy(), control["timestamp_us"].copy(),
                       control["frame_id"].copy(), control["source_width"], control["source_height"])
    baseline, changed = build(before_xy), build(after_xy)
    for key in ("point_available", "visibility_evidence", "body_relative_available",
                "hand_local_available", "native_available", "timestamp_us", "frame_id", "raw_score"):
        if baseline[key].tobytes() != changed[key].tobytes():
            raise ValueError(f"canonical rebuild changed invariant {key}")
    allowed = np.zeros_like(baseline["tcn280"], dtype=bool)
    start = HAND_START[control["side"]]
    common_start = 9 if control["side"] == "L" else 30
    local_start = 102 if control["side"] == "L" else 144
    for j in DISTAL:
        rows = synthetic[:, start + j]
        for offset in (0, 1):
            allowed[rows, 2 * (common_start + j) + offset] = True
            allowed[rows, local_start + 2 * j + offset] = True
    if baseline["tcn280"][~allowed].tobytes() != changed["tcn280"][~allowed].tobytes():
        raise ValueError("canonical rebuild changed protected body/palm/thumb/time channels")
    if (baseline["body_relative_xy"][~synthetic].tobytes()
            != changed["body_relative_xy"][~synthetic].tobytes()):
        raise ValueError("body-relative protected coordinates changed")
    return dict(baseline=baseline, control=changed, allowed_tcn_channel_change=allowed,
                protected_channels_identical=True, model_inference=False)
