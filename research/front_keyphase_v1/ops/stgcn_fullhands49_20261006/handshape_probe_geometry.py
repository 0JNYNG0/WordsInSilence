"""Frozen handshape-probe geometry; no data loading, model calls, or file writes.

Coordinates have MP75 layout, shape (T, 75, 2), and float32/float64 dtype.
Observed masks have shape (T, 75) and boolean dtype. Runs use [start, stop).
The synthetic self-test can be run with ``python -B FILE --self-test``.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np


FINGERS = {
    "thumb": {"chain": (1, 2, 3, 4), "pivot": 2, "distal": (3, 4)},
    "index": {"chain": (5, 6, 7, 8), "pivot": 6, "distal": (7, 8)},
    "middle": {"chain": (9, 10, 11, 12), "pivot": 10, "distal": (11, 12)},
    "ring": {"chain": (13, 14, 15, 16), "pivot": 14, "distal": (15, 16)},
    "pinky": {"chain": (17, 18, 19, 20), "pivot": 18, "distal": (19, 20)},
}
HAND_OFFSETS = {"left": 33, "right": 54}
BODY7 = (0, 2, 5, 11, 12, 13, 14)
RETAINED_HAND10 = (0, 4, 5, 8, 9, 12, 13, 16, 17, 20)
FULL49 = BODY7 + tuple(range(33, 75))
RETAINED27 = BODY7 + tuple(
    offset + joint for offset in (33, 54) for joint in RETAINED_HAND10
)
WORK_RTOL = 1e-10
WORK_ATOL = 1e-10


def _validate(xy, observed, hand, finger):
    if not isinstance(xy, np.ndarray) or xy.ndim != 3 or xy.shape[1:] != (75, 2):
        raise ValueError("xy must be a numpy array with shape (T, 75, 2)")
    if xy.dtype.kind != "f" or xy.dtype.itemsize not in (4, 8):
        raise ValueError("xy dtype must be float32 or float64")
    if not isinstance(observed, np.ndarray) or observed.shape != xy.shape[:2]:
        raise ValueError("observed must be a numpy array with shape (T, 75)")
    if observed.dtype != np.dtype(bool):
        raise ValueError("observed dtype must be bool")
    if hand not in HAND_OFFSETS:
        raise ValueError("hand must be 'left' or 'right'")
    if finger not in FINGERS:
        raise ValueError("finger must be one of " + ", ".join(FINGERS))


def _runs(mask):
    padded = np.concatenate(([False], mask, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [[int(a), int(b)] for a, b in zip(changes[::2], changes[1::2])]


def eligibility(xy, observed, hand, finger):
    """Return raw-frame eligibility and JSON-serializable exclusion diagnostics.

    Each eligible frame has an observed finite wrist and complete finger chain,
    with strictly positive wrist-to-first and all successive chain lengths.
    Exclusion reasons can overlap; no interpolation or mask mutation occurs.
    """
    _validate(xy, observed, hand, finger)
    offset = HAND_OFFSETS[hand]
    chain = [offset + joint for joint in FINGERS[finger]["chain"]]
    wrist = offset
    finite = np.isfinite(xy).all(axis=2)
    reasons = {
        "unobserved_wrist": ~observed[:, wrist],
        "unobserved_chain": ~observed[:, chain].all(axis=1),
        "nonfinite_wrist": ~finite[:, wrist],
        "nonfinite_chain": ~finite[:, chain].all(axis=1),
    }
    ready = observed[:, [wrist] + chain].all(axis=1) & finite[:, [wrist] + chain].all(axis=1)
    nonpositive = np.zeros(len(xy), dtype=bool)
    ready_indices = np.flatnonzero(ready)
    if len(ready_indices):
        points = xy[ready_indices][:, [wrist] + chain].astype(np.float64)
        delta = np.diff(points, axis=1)
        lengths = np.hypot(delta[:, :, 0], delta[:, :, 1])
        nonpositive[ready_indices] = (lengths <= 0.0).any(axis=1)
    reasons["nonpositive_bone_length"] = nonpositive
    eligible = ready & ~nonpositive
    runs = _runs(eligible)
    count = int(eligible.sum())
    diagnostic = {
        "frame_count": int(len(xy)),
        "eligible_count": count,
        "eligible_fraction": float(count / len(xy)) if len(xy) else None,
        "ineligible_count": int(len(xy) - count),
        "eligible_runs": runs,
        "eligible_run_count": len(runs),
        "run_index_convention": "start_inclusive_stop_exclusive",
        "reason_counts": {name: int(mask.sum()) for name, mask in reasons.items()},
        "frame_reasons": [[name for name, mask in reasons.items() if mask[i]] for i in range(len(xy))],
    }
    return eligible, diagnostic


def _rotate_in_place(working, point_mask, pivot, degrees):
    rows, joints = np.nonzero(point_mask)
    if not len(rows):
        return
    origin = working[rows, pivot].copy()
    delta = working[rows, joints] - origin
    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    rotated = np.empty_like(delta)
    rotated[:, 0] = cosine * delta[:, 0] - sine * delta[:, 1]
    rotated[:, 1] = sine * delta[:, 0] + cosine * delta[:, 1]
    working[rows, joints] = origin + rotated


def _max_abs(array):
    return float(np.max(np.abs(array))) if array.size else 0.0


def _displacement(original, changed, observed, indices):
    before = original[:, indices].astype(np.float64)
    after = changed[:, indices].astype(np.float64)
    selected_observed = observed[:, indices]
    valid = selected_observed & np.isfinite(before).all(axis=2) & np.isfinite(after).all(axis=2)
    delta = after[valid] - before[valid]
    distance = np.hypot(delta[:, 0], delta[:, 1])
    point_count = int(valid.sum())
    # This RMS is Euclidean point displacement over every finite observed point,
    # including unchanged points and ineligible frames, not coordinate-axis RMS.
    rms = float(np.linalg.norm(distance) / math.sqrt(point_count)) if point_count else None
    return {
        "joint_count": len(indices),
        "total_frame_joint_count": int(len(original) * len(indices)),
        "observed_point_count": int(selected_observed.sum()),
        "finite_observed_point_count": point_count,
        "nonfinite_observed_point_count": int(selected_observed.sum()) - point_count,
        "coordinate_count": point_count * 2,
        "changed_point_count": int(np.count_nonzero(distance)),
        "rms": rms,
        "max": float(np.max(distance)) if point_count else None,
        "rms_definition": "sqrt(sum(dx^2+dy^2)/finite_observed_frame_joint_count)",
        "includes_unchanged_points_and_ineligible_frames": True,
    }


def _length_audit(before, after, observed, eligible, hand, finger, kind):
    offset = HAND_OFFSETS[hand]
    checked = 0
    max_error = 0.0
    passed = True
    for frame in np.flatnonzero(eligible):
        if kind == "internal":
            ids = [offset] + [offset + joint for joint in FINGERS[finger]["chain"]]
            a_delta = np.diff(before[frame, ids], axis=0)
            b_delta = np.diff(after[frame, ids], axis=0)
        else:
            ids = np.arange(offset, offset + 21)[observed[frame, offset:offset + 21]]
            ia, ib = np.triu_indices(len(ids), k=1)
            a_delta = before[frame, ids[ia]] - before[frame, ids[ib]]
            b_delta = after[frame, ids[ia]] - after[frame, ids[ib]]
        a_length = np.hypot(a_delta[:, 0], a_delta[:, 1])
        b_length = np.hypot(b_delta[:, 0], b_delta[:, 1])
        checked += int(len(a_length))
        max_error = max(max_error, _max_abs(b_length - a_length))
        passed = passed and bool(np.allclose(a_length, b_length, rtol=WORK_RTOL, atol=WORK_ATOL))
    return {
        "kind": "wrist_and_finger_chain_bones" if kind == "internal" else "all_observed_hand_pairwise_distances",
        "distance_count": checked,
        "max_abs_error": max_error,
        "rtol": WORK_RTOL,
        "atol": WORK_ATOL,
        "passed": passed,
    }


def transform(xy, observed, hand, finger, degrees, kind):
    """Rotate the frozen target and return (new_xy, eligible_frames, diagnostics).

    ``kind`` is 'internal' (the two distal points about their finger pivot) or
    'rigid' (all observed nonwrist points about the wrist). Both use exactly the
    same finger eligibility. Only degrees == -15 or +15 is accepted. Coordinates
    retain their dtype, and protected points retain their original bytes.
    Observed nonfinite rigid targets fail closed instead of being silently skipped.
    """
    _validate(xy, observed, hand, finger)
    if isinstance(degrees, (bool, np.bool_)) or not isinstance(degrees, (int, float, np.integer, np.floating)) or degrees not in (-15, 15):
        raise ValueError("degrees must be exactly -15 or +15")
    if kind not in ("internal", "rigid"):
        raise ValueError("kind must be 'internal' or 'rigid'")
    eligible, eligible_diagnostic = eligibility(xy, observed, hand, finger)
    source_bytes, mask_bytes = xy.tobytes(), observed.tobytes()
    offset = HAND_OFFSETS[hand]
    point_mask = np.zeros(observed.shape, dtype=bool)
    if kind == "internal":
        pivot = offset + FINGERS[finger]["pivot"]
        ids = [offset + joint for joint in FINGERS[finger]["distal"]]
        point_mask[:, ids] = eligible[:, None]
    else:
        pivot = offset
        same_hand_bad = observed[:, offset:offset + 21] & ~np.isfinite(xy[:, offset:offset + 21]).all(axis=2)
        if np.any(same_hand_bad & eligible[:, None]):
            raise ValueError("rigid eligible frame contains a nonfinite observed same-hand coordinate")
        point_mask[:, offset + 1:offset + 21] = eligible[:, None] & observed[:, offset + 1:offset + 21]
    before = xy.astype(np.float64, copy=True)
    working = before.copy()
    _rotate_in_place(working, point_mask, pivot, float(degrees))
    if not np.isfinite(working[point_mask]).all():
        raise ArithmeticError("rotation produced a nonfinite coordinate")
    result = xy.copy()
    result[point_mask] = working[point_mask].astype(xy.dtype)
    if not np.isfinite(result[point_mask]).all():
        raise ArithmeticError("dtype cast produced a nonfinite coordinate")

    protected_exact = result[~point_mask].tobytes() == xy[~point_mask].tobytes()
    inverse = working.copy()
    _rotate_in_place(inverse, point_mask, pivot, -float(degrees))
    inverse_error = _max_abs(inverse[point_mask] - before[point_mask])
    inverse_passed = bool(np.allclose(inverse[point_mask], before[point_mask], rtol=WORK_RTOL, atol=WORK_ATOL))
    cast_inverse = result.astype(np.float64)
    _rotate_in_place(cast_inverse, point_mask, pivot, -float(degrees))
    cast_inverse_error = _max_abs(cast_inverse[point_mask] - before[point_mask])
    finite_observed = observed & np.isfinite(xy).all(axis=2)
    original_scale = max(1.0, _max_abs(before[finite_observed]))
    cast_tolerance = float(8 * np.finfo(np.float32).eps * original_scale)
    cast_inverse_passed = bool(cast_inverse_error <= cast_tolerance)
    length_audit = _length_audit(before, working, observed, eligible, hand, finger, kind)
    inputs_exact = source_bytes == xy.tobytes() and mask_bytes == observed.tobytes()
    changed_points = np.zeros(observed.shape, dtype=bool)
    changed_points[point_mask] = np.any(result[point_mask] != xy[point_mask], axis=1)
    changed_frames = changed_points.any(axis=1)
    audits = {
        "protected_coordinates_byte_exact": protected_exact,
        "input_coordinates_and_mask_byte_exact": inputs_exact,
        "lengths_float64": length_audit,
        "inverse_float64": {"passed": inverse_passed, "max_abs_error": inverse_error, "rtol": WORK_RTOL, "atol": WORK_ATOL},
        "inverse_after_dtype_cast": {"passed": cast_inverse_passed, "max_abs_error": cast_inverse_error, "absolute_tolerance": cast_tolerance, "tolerance_formula": "8*eps(float32)*max(1,max_abs_original_finite_observed_coordinate)"},
    }
    audit_passed = protected_exact and inputs_exact and length_audit["passed"] and inverse_passed and cast_inverse_passed
    diagnostic = {
        "hand": hand,
        "finger": finger,
        "degrees": float(degrees),
        "kind": kind,
        "dtype": str(xy.dtype),
        "eligibility": eligible_diagnostic,
        "eligible_count": eligible_diagnostic["eligible_count"],
        "eligible_run_count": eligible_diagnostic["eligible_run_count"],
        "eligible_runs": eligible_diagnostic["eligible_runs"],
        "reason_counts": eligible_diagnostic["reason_counts"],
        "scheduled_point_count": int(point_mask.sum()),
        "changed_point_count": int(changed_points.sum()),
        "changed_frame_count": int(changed_frames.sum()),
        "changed_frames": changed_frames.tolist(),
        "displacement": {
            "full49": _displacement(xy, result, observed, FULL49),
            "retained27": _displacement(xy, result, observed, RETAINED27),
        },
        "audits": audits,
        "audit_passed": bool(audit_passed),
    }
    if not audit_passed:
        raise ArithmeticError("handshape geometry invariant failed: " + json.dumps(audits, allow_nan=False))
    # Also reject nonfinite diagnostic scalars; None describes absent denominators.
    json.dumps(diagnostic, allow_nan=False)
    return result, eligible, diagnostic


def _synthetic_fixture(dtype):
    xy = np.zeros((6, 75, 2), dtype=dtype)
    observed = np.ones((6, 75), dtype=bool)
    for frame in range(len(xy)):
        xy[frame] = np.stack((np.arange(75) * 0.013 + frame * 0.05, np.arange(75) * 0.019 - frame * 0.03), axis=1)
        for offset in HAND_OFFSETS.values():
            wrist = np.array([offset * 0.04 + frame * 0.02, 0.9 + frame * 0.03])
            xy[frame, offset] = wrist
            for finger_index, spec in enumerate(FINGERS.values()):
                for joint_index, joint in enumerate(spec["chain"], 1):
                    xy[frame, offset + joint] = wrist + [0.2 * (finger_index - 2) + 0.025 * joint_index ** 2, -0.13 * joint_index]
    return xy, observed


def self_test():
    """Exercise synthetic geometry only; return JSON data and perform no writes."""
    cases = []
    checks = []
    for dtype in (np.float32, np.float64):
        for hand, offset in HAND_OFFSETS.items():
            for finger, spec in FINGERS.items():
                xy, observed = _synthetic_fixture(dtype)
                # Eligible runs [0,2), [3,4), [5,6); separately exercise a
                # missing joint and a degenerate wrist-to-first bone.
                observed[2, offset + spec["chain"][2]] = False
                xy[4, offset + spec["chain"][0]] = xy[4, offset]
                # Unobserved unusual bytes, including a non-default NaN payload.
                observed[:, 1] = False
                payload = 0x7FC12345 if dtype == np.float32 else 0x7FF8123456789ABC
                uint = np.uint32 if dtype == np.float32 else np.uint64
                xy[:, 1, 0] = np.array([payload], dtype=uint).view(dtype)[0]
                xy[:, 1, 1] = -0.0
                original_bytes, observed_bytes = xy.tobytes(), observed.tobytes()
                for degrees in (-15, 15):
                    matched_eligibility = None
                    for kind in ("internal", "rigid"):
                        result, eligible, diag = transform(xy, observed, hand, finger, degrees, kind)
                        assert result.dtype == xy.dtype and result.shape == xy.shape
                        assert not np.shares_memory(result, xy)
                        assert eligible.tolist() == [True, True, False, True, False, True]
                        assert diag["eligible_runs"] == [[0, 2], [3, 4], [5, 6]]
                        assert diag["reason_counts"]["unobserved_chain"] == 1
                        assert diag["reason_counts"]["nonpositive_bone_length"] == 1
                        assert diag["changed_frames"] == eligible.tolist()
                        assert diag["audit_passed"]
                        assert result[~eligible].tobytes() == xy[~eligible].tobytes()
                        assert result[:, 1].tobytes() == xy[:, 1].tobytes()
                        assert result[:, offset].tobytes() == xy[:, offset].tobytes()
                        tip = offset + spec["distal"][-1]
                        assert np.all(np.any(result[eligible, tip] != xy[eligible, tip], axis=1))
                        assert tip in RETAINED27
                        if matched_eligibility is None:
                            matched_eligibility = eligible
                        else:
                            assert np.array_equal(eligible, matched_eligibility)
                        assert xy.tobytes() == original_bytes and observed.tobytes() == observed_bytes
                        json.dumps(diag, allow_nan=False)
                        cases.append({"dtype": np.dtype(dtype).name, "hand": hand, "finger": finger, "degrees": degrees, "kind": kind, "passed": True})
    checks.append("80 dtype/hand/finger/sign/kind cases: lengths, inverse, common tip, bytes, eligibility, JSON")

    xy, observed = _synthetic_fixture(np.float32)
    observed[:] = False
    result, eligible, diag = transform(xy, observed, "left", "index", 15, "rigid")
    assert not eligible.any() and result.tobytes() == xy.tobytes()
    assert diag["displacement"]["full49"]["rms"] is None
    checks.append("no observations: no changes and undefined displacement denominator")

    xy, observed = _synthetic_fixture(np.float64)
    xy[1, 33, 0] = np.nan
    xy[2, 40, 0] = np.inf
    observed[3, 33] = False
    xy[4, 39] = xy[4, 38]
    eligible, diag = eligibility(xy, observed, "left", "index")
    assert eligible.tolist() == [True, False, False, False, False, True]
    assert diag["reason_counts"]["nonfinite_wrist"] == 1
    assert diag["reason_counts"]["nonfinite_chain"] == 1
    assert diag["reason_counts"]["unobserved_wrist"] == 1
    assert diag["reason_counts"]["nonpositive_bone_length"] == 1
    checks.append("nonfinite, missing wrist and intermediate zero-length bones are ineligible")

    xy, observed = _synthetic_fixture(np.float32)
    observed[:, 50] = False
    xy[:, 50] = np.nan
    result, _, diag = transform(xy, observed, "left", "index", 15, "rigid")
    assert result[:, 50].tobytes() == xy[:, 50].tobytes()
    assert diag["scheduled_point_count"] == 6 * 19
    assert diag["audits"]["lengths_float64"]["distance_count"] == 6 * (20 * 19 // 2)
    observed[0, 50] = True
    try:
        transform(xy, observed, "left", "index", 15, "rigid")
    except ValueError as error:
        assert "nonfinite observed" in str(error)
    else:
        raise AssertionError("observed nonfinite rigid target must fail closed")
    checks.append("rigid preserves unobserved NaN; observed nonfinite target fails closed")

    xy, observed = _synthetic_fixture(np.float32)
    for degrees, kind in ((0, "internal"), (30, "rigid"), (True, "internal"), (15, "other")):
        try:
            transform(xy, observed, "right", "thumb", degrees, kind)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid transform parameters were accepted")
    checks.append("only exact +/-15 degrees and the two frozen kinds accepted")

    result, eligible, diag = transform(xy[:0], observed[:0], "left", "thumb", -15, "internal")
    assert result.shape == (0, 75, 2) and not len(eligible)
    assert diag["eligible_count"] == 0 and diag["eligibility"]["eligible_fraction"] is None
    checks.append("empty input produces zero counts without fabricated displacement")
    return {"status": "PASS", "synthetic_only": True, "writes": 0, "case_count": len(cases), "checks": checks, "cases": cases}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(self_test(), ensure_ascii=False, allow_nan=False))
