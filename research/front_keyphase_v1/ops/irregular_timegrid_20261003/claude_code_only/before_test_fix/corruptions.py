"""Predeclared deterministic irregular-time corruptions of ONE clean native-layout clip.

Pure NumPy + hashlib. Selection depends only on (sample_id, case, frame index) through a fixed
domain-separated SHA-256; no label, user, session or score can enter (there is no parameter for
them). Coordinates and masks are only ever indexed from source frames - never interpolated,
filled or edited - and the inputs are never mutated.

  isolated_drop  step = 1 + (D(cur)[0] & 1); first and last kept; max source-index gap 2
  burst_drop     step = 3 if D(cur)[0] % 4 == 0 else 1; max gap 3 (spans of ~100 ms that the
                 restorer's long-gap rule must leave unsupported)
  clock_noise    every frame kept; interior ts[i] += int.from_bytes(D(i)[:2], 'big') % 2001 - 1000;
                 endpoints fixed. Timestamp measurement noise only, NOT motion resampling or a
                 speed change. A result that is not strictly increasing is refused, never clamped.

D(i) = sha256(f"WIS_TIMEGRID_IRREG_V1|{sample_id}|{case}|{i}".encode('utf-8')).digest()
"""
from __future__ import annotations

import hashlib

import numpy as np

CASES = ["isolated_drop", "burst_drop", "clock_noise"]
DOMAIN = "WIS_TIMEGRID_IRREG_V1"
N_JOINTS = 67
MAX_SOURCE_GAP = {"isolated_drop": 2, "burst_drop": 3, "clock_noise": 1}
MAX_OFFSET_US = 1000


class CorruptionError(ValueError):
    pass


def digest(sample_id: str, case: str, i: int) -> bytes:
    return hashlib.sha256(f"{DOMAIN}|{sample_id}|{case}|{i}".encode("utf-8")).digest()


def _check_case(case):
    if case not in CASES:
        raise CorruptionError(f"case must be one of {CASES}, got {case!r}")


def select_indices(t: int, sample_id: str, case: str) -> list:
    """Retained source indices; always starts at 0 and ends at t-1."""
    _check_case(case)
    if type(t) is not int or t < 1:
        raise CorruptionError(f"t must be an int >= 1, got {t!r}")
    if case == "clock_noise":
        return list(range(t))
    idx, cur = [0], 0
    while cur != t - 1:
        b = digest(sample_id, case, cur)[0]
        step = 1 + (b & 1) if case == "isolated_drop" else (3 if b % 4 == 0 else 1)
        cur = min(cur + step, t - 1)
        idx.append(cur)
    return idx


def clock_offsets(t: int, sample_id: str) -> list:
    """Per-frame microsecond offsets of clock_noise; 0 at both endpoints."""
    return [0 if i in (0, t - 1) else
            int.from_bytes(digest(sample_id, "clock_noise", i)[:2], "big") % 2001 - MAX_OFFSET_US
            for i in range(t)]


def _validate(xy, present, ts, sample_id, case):
    _check_case(case)
    if not isinstance(sample_id, str) or not sample_id:
        raise CorruptionError("sample_id must be a non-empty string")
    xy, present, ts = np.asarray(xy), np.asarray(present), np.asarray(ts)
    if xy.ndim != 3 or xy.shape[0] < 1 or xy.shape[1:] != (N_JOINTS, 2):
        raise CorruptionError(f"xy must be [T>=1,{N_JOINTS},2], got {xy.shape}")
    t = xy.shape[0]
    if present.shape != (t, N_JOINTS):
        raise CorruptionError(f"present must be [{t},{N_JOINTS}], got {present.shape}")
    if ts.shape != (t,) or ts.dtype == np.bool_ or not np.issubdtype(ts.dtype, np.integer):
        raise CorruptionError(f"ts must be integer [{t}], got {ts.dtype} {ts.shape}")
    ti = [int(v) for v in ts]
    if any(b <= a for a, b in zip(ti, ti[1:])):
        raise CorruptionError("source ts must be strictly increasing")
    return xy, present, ts, ti


def corrupt(xy, present, ts, sample_id, case) -> dict:
    """-> dict xy, present, ts, indices, clock_offset_us, case. Inputs are not modified."""
    xy, present, ts, ti = _validate(xy, present, ts, sample_id, case)
    t = len(ti)
    idx = np.asarray(select_indices(t, sample_id, case), dtype=np.int64)
    off = clock_offsets(t, sample_id) if case == "clock_noise" else [0] * len(idx)
    new = [ti[i] + o for i, o in zip(idx.tolist(), off)]
    if any(b <= a for a, b in zip(new, new[1:])):
        raise CorruptionError(f"{sample_id}/{case}: corrupted ts not strictly increasing (refused, not clamped)")
    try:
        out_ts = np.asarray(new, dtype=ts.dtype)
    except OverflowError as exc:
        raise CorruptionError(f"{sample_id}/{case}: ts overflow in {ts.dtype}") from exc
    if [int(v) for v in out_ts] != new:
        raise CorruptionError(f"{sample_id}/{case}: ts overflow in {ts.dtype}")
    return {"xy": xy[idx], "present": present[idx], "ts": out_ts, "indices": idx,
            "clock_offset_us": np.asarray(off, dtype=np.int64), "case": case}
