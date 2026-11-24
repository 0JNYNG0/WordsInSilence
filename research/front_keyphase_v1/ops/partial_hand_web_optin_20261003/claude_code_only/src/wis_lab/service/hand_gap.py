"""Isolated whole-hand dropout: deterministic masking schedule and same-channel linear restoration.

Pure NumPy, no external state. Input xy [T,67,2] floating, present [T,67] bool or exactly 0/1,
ts [T] strictly increasing integer microseconds. Hand slots: left 25:46, right 46:67.
Frames and timestamps are never added, removed or moved.

mask_one_hand(xy, present, ts, sample_id)   (FROZEN_PLAN.mask)
  digest = sha256(('WIS_HAND_GAP_V1|' + sample_id).encode('utf-8')).digest()
  offset = digest[0] % 6, first_side = digest[1] % 2 (0 left, 1 right)
  for k, i in enumerate(range(2 + offset, T - 2, 6)): side = (first_side + k) % 2 (skips still advance)
  eligible iff that hand has all 21 joints present in the RECEIVED input at i-1, i, i+1 and
  ts[i+1] - ts[i-1] <= 66668 us. Eligible: xy[i, hand] = 0, present[i, hand] = 0. Nothing else changes.
  No label, score or coordinate geometry enters the choice. No eligible position is a valid identity.

restore(xy, present, ts)                    (FROZEN_PLAN.candidate)
  per side, per interior i: iff all 21 joints of that hand are absent at i, all 21 present at i-1 and
  i+1 in the RECEIVED input, and ts[i+1] - ts[i-1] <= 66668 us: float64 time-weighted linear
  interpolation w = (ts[i]-ts[i-1]) / (ts[i+1]-ts[i-1]), cast to the xy dtype, present = 1, estimated.
  Endpoints are read from the received arrays only, so estimates never feed other estimates.
  No partial-hand, body, first/last frame, 2+ consecutive absence or long-gap fill; no identity fix.
  The filled hand is the same channel; this is not evidence that it is the same physical hand.
"""
from __future__ import annotations

import hashlib

import numpy as np

N_JOINTS = 67
HAND_JOINTS = 21
HANDS = {"left": slice(25, 46), "right": slice(46, 67)}
SIDES = ("left", "right")
SALT = "WIS_HAND_GAP_V1|"
PERIOD = 6
FIRST = 2
MAX_SPAN_US = 66668


def _validate(xy, present, ts):
    xy, present, ts = np.asarray(xy), np.asarray(present), np.asarray(ts)
    if xy.ndim != 3 or xy.shape[0] < 1 or xy.shape[1:] != (N_JOINTS, 2):
        raise ValueError(f"xy must be [T>=1,{N_JOINTS},2], got {xy.shape}")
    if not np.issubdtype(xy.dtype, np.floating):
        raise ValueError(f"xy must be floating, got {xy.dtype}")
    t = xy.shape[0]
    if present.shape != (t, N_JOINTS):
        raise ValueError(f"present must be [{t},{N_JOINTS}], got {present.shape}")
    if present.dtype == np.bool_:
        pb = present.copy()
    elif np.issubdtype(present.dtype, np.integer) or np.issubdtype(present.dtype, np.floating):
        if not np.all((present == 0) | (present == 1)):
            raise ValueError("present must be bool or exactly 0/1")
        pb = present == 1
    else:
        raise ValueError(f"present dtype {present.dtype} not allowed")
    if ts.shape != (t,):
        raise ValueError(f"ts must be [{t}], got {ts.shape}")
    if ts.dtype == np.bool_ or not np.issubdtype(ts.dtype, np.integer):
        raise ValueError(f"ts must be integer microseconds, got {ts.dtype}")
    ti = [int(v) for v in ts]                                   # Python ints: no overflow
    if any(b <= a for a, b in zip(ti, ti[1:])):
        raise ValueError("ts must be strictly increasing")
    if not np.isfinite(xy[pb]).all():
        raise ValueError("non-finite coordinate where present")
    return xy, present, ts, pb, ti


def schedule(sample_id, t):
    """-> (offset, first_side_index, [(frame, side), ...]) of scheduled positions; eligibility not applied."""
    if not isinstance(sample_id, str):
        raise TypeError(f"sample_id must be str, got {type(sample_id).__name__}")
    d = hashlib.sha256((SALT + sample_id).encode("utf-8")).digest()
    offset, first = d[0] % PERIOD, d[1] % 2
    return offset, first, [(i, SIDES[(first + k) % 2]) for k, i in enumerate(range(FIRST + offset, t - 2, PERIOD))]


def _all(pb, i, side):
    return bool(pb[i, HANDS[side]].all())


def mask_one_hand(xy, present, ts, sample_id) -> dict:
    xy, present, ts, pb, ti = _validate(xy, present, ts)
    offset, first, sched = schedule(sample_id, len(ti))
    out_xy, out_pr = xy.copy(), present.copy()
    injected = np.zeros(pb.shape, dtype=bool)
    rows = []
    for i, side in sched:
        ok = _all(pb, i - 1, side) and _all(pb, i, side) and _all(pb, i + 1, side) \
            and ti[i + 1] - ti[i - 1] <= MAX_SPAN_US
        rows.append({"frame": i, "side": side, "eligible": ok})
        if ok:
            s = HANDS[side]
            out_xy[i, s] = 0
            out_pr[i, s] = 0
            injected[i, s] = True
    return {"xy": out_xy, "present": out_pr, "ts": ts.copy(), "injected": injected,
            "events": [{"frame": e["frame"], "side": e["side"]} for e in rows if e["eligible"]],
            "schedule": rows, "offset": offset, "first_side": SIDES[first]}


def restore(xy, present, ts) -> dict:
    xy, present, ts, pb, ti = _validate(xy, present, ts)
    out_xy, out_pr = xy.copy(), present.copy()
    estimated = np.zeros(pb.shape, dtype=bool)
    for side in SIDES:
        s = HANDS[side]
        for i in range(1, len(ti) - 1):
            span = ti[i + 1] - ti[i - 1]
            if pb[i, s].any() or not (_all(pb, i - 1, side) and _all(pb, i + 1, side)) or span > MAX_SPAN_US:
                continue
            w = (ti[i] - ti[i - 1]) / span                      # exact int / int -> float64
            lo, hi = xy[i - 1, s].astype(np.float64), xy[i + 1, s].astype(np.float64)
            out_xy[i, s] = ((1.0 - w) * lo + w * hi).astype(xy.dtype)
            out_pr[i, s] = 1
            estimated[i, s] = True
    return {"xy": out_xy, "present": out_pr, "ts": ts.copy(), "observed": pb.copy(), "estimated": estimated,
            "identity": not bool(estimated.any())}
