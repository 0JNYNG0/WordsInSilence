"""Independent mathematical reference. Does not import the candidate implementation."""
from __future__ import annotations

import hashlib
from fractions import Fraction

import numpy as np

HANDS = (slice(25, 46), slice(46, 67))


def checked(xy, pr, ts):
    xy, pr, ts = np.asarray(xy), np.asarray(pr), np.asarray(ts)
    if xy.ndim != 3 or xy.shape[1:] != (67, 2) or len(xy) < 1:
        raise ValueError("xy must be T,67,2 with T>=1")
    if not np.issubdtype(xy.dtype, np.floating):
        raise ValueError("xy floating dtype required")
    if pr.shape != xy.shape[:2] or ts.shape != (len(xy),):
        raise ValueError("mask/time shape")
    if not (pr.dtype == np.bool_ or np.issubdtype(pr.dtype, np.number)) or not np.all((pr == 0) | (pr == 1)):
        raise ValueError("binary mask required")
    if not np.issubdtype(ts.dtype, np.integer) or ts.dtype == np.bool_:
        raise ValueError("integer timestamps required")
    times = [int(t) for t in ts]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("strictly increasing timestamps required")
    valid = pr.astype(bool)
    if not np.isfinite(xy[valid]).all():
        raise ValueError("present coordinates must be finite")
    return xy, pr, ts, valid, times


def reference_mask(xy, pr, ts, sample_id):
    xy, pr, ts, valid, times = checked(xy, pr, ts)
    digest = hashlib.sha256(("WIS_HAND_GAP_V1|" + sample_id).encode("utf-8")).digest()
    out, mask = xy.copy(), pr.copy()
    injected = np.zeros(pr.shape, dtype=bool)
    events = []
    positions = range(2 + digest[0] % 6, len(ts) - 2, 6)
    for sequence_index, frame in enumerate(positions):
        side = (digest[1] % 2 + sequence_index) % 2
        hand = HANDS[side]
        if valid[frame - 1:frame + 2, hand].all() and times[frame + 1] - times[frame - 1] <= 66668:
            out[frame, hand] = 0
            mask[frame, hand] = 0
            injected[frame, hand] = True
            events.append({"frame": frame, "side": side, "scheduled_index": sequence_index})
    return {"xy": out, "present": mask, "ts": ts.copy(), "injected": injected, "events": events}


def reference_restore(xy, pr, ts):
    xy, pr, ts, valid, times = checked(xy, pr, ts)
    out, mask = xy.copy(), pr.copy()
    estimated = np.zeros(pr.shape, dtype=bool)
    events = []
    for side, hand in enumerate(HANDS):
        missing = ~valid[:, hand].any(axis=1)
        boundaries = np.flatnonzero(np.diff(np.r_[False, missing, False].astype(np.int8)))
        for start, stop in zip(boundaries[::2], boundaries[1::2]):
            i = int(start)
            if stop - start != 1 or i == 0 or i + 1 == len(ts):
                continue
            if not valid[i - 1, hand].all() or not valid[i + 1, hand].all():
                continue
            elapsed = times[i + 1] - times[i - 1]
            if elapsed > 66668:
                continue
            ratio = Fraction(times[i] - times[i - 1], elapsed)
            left = xy[i - 1, hand].astype(np.float64)
            right = xy[i + 1, hand].astype(np.float64)
            out[i, hand] = (left + float(ratio) * (right - left)).astype(xy.dtype)
            mask[i, hand] = 1
            estimated[i, hand] = True
            events.append({"frame": i, "side": side})
    return {"xy": out, "present": mask, "ts": ts.copy(), "observed": valid.copy(),
            "estimated": estimated, "identity": not bool(estimated.any()), "events": events}


def array_hash(arr):
    arr = np.asarray(arr)
    return hashlib.sha256(f"{arr.shape}|{arr.dtype}|".encode() + arr.tobytes()).hexdigest()


def raw_hash(xy, pr, ts):
    return {"xy": array_hash(xy), "present": array_hash(pr), "ts": array_hash(ts)}


def four_arms(xy, pr, ts, sample_id):
    # Original hidden values are available to the evaluator only. The estimator
    # receives precisely the degraded arrays, not the original evaluator copy.
    hidden = reference_mask(xy, pr, ts, sample_id)
    repaired = reference_restore(hidden["xy"], hidden["present"], hidden["ts"])
    original_repair = reference_restore(xy, pr, ts)
    assert np.all(repaired["estimated"][hidden["injected"]])
    arms = {"original": {"xy": xy.copy(), "present": pr.copy(), "ts": ts.copy()},
            "original_repair": original_repair, "masked": hidden, "masked_repair": repaired}
    return arms, hidden["injected"]
