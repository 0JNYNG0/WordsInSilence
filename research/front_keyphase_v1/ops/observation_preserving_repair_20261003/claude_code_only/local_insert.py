"""Observation-preserving local insertion for ONE native-layout clip (offline, one-frame lookahead).

Pure NumPy, no external state. restore() receives only the arrays to repair (xy, present, ts):
there is no parameter through which a dense original, a dropped frame, a label, a target length,
a model or a score could enter.

Rules (fixed before any result; no tuning, no word-specific exception):
  * every input frame is copied exactly once, in order, with its timestamp, present and xy verbatim
    (absent values incl. NaN are kept); there is no global grid, no frame replacement, no extrapolation;
  * for each pair of ADJACENT input frames with gap g = t1 - t0 (Python ints) exactly ONE midpoint is
    inserted iff 1.5P <= g <= 2P + 2 us (integer test 1_500_000 <= 30*g <= 2_000_060, P = 1e6/30) AND
    at least one joint is present at both ends; i.e. the gap's nearest nominal interval count is 2,
    capped by the prior long-gap rule. Longer gaps and ordinary jitter stay unchanged. Inserted frames
    are never used as endpoints (no recursion);
  * midpoint timestamp = round((t0 + t1) / 2), exact rational ties-to-even on the absolute time;
  * per jointly-present joint: float64 linear interpolation with w = (mid - t0) / g, cast back to the
    source xy dtype; every other slot of the inserted frame is xy 0, present 0;
  * estimates are never marked observed; identity = no insertion (still copies).
"""
from __future__ import annotations

import numpy as np

N_JOINTS = 67
MIN_GAP_30 = 1_500_000          # 30*gap >= 1.5e6       <=>  gap >= 1.5P
MAX_GAP_30 = 2 * 1_000_000 + 60  # 30*gap <= 2e6 + 60   <=>  gap <= 2P + 2 us


def _validate(xy, present, ts):
    xy, present, ts = np.asarray(xy), np.asarray(present), np.asarray(ts)
    if xy.ndim != 3 or xy.shape[1:] != (N_JOINTS, 2) or xy.shape[0] < 1:
        raise ValueError(f"xy must be [T>=1,{N_JOINTS},2], got {xy.shape}")
    if not np.issubdtype(xy.dtype, np.floating):
        raise ValueError(f"xy must be floating, got {xy.dtype}")
    t = xy.shape[0]
    if present.shape != (t, N_JOINTS):
        raise ValueError(f"present must be [{t},{N_JOINTS}], got {present.shape}")
    if present.dtype == np.bool_:
        pb = present.astype(bool)
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
    ti = [int(v) for v in ts]
    if any(b <= a for a, b in zip(ti, ti[1:])):
        raise ValueError("ts must be strictly increasing")
    if not np.isfinite(xy[pb]).all():
        raise ValueError("non-finite coordinate where present")
    return xy, present, ts, pb, ti


def _round_half_even(numerator: int, denominator: int) -> int:
    """Exact round(numerator/denominator), ties to even; denominator > 0, any-sign numerator."""
    q, r = divmod(numerator, denominator)                # floor division: 0 <= r < denominator
    return q + int(2 * r > denominator or (2 * r == denominator and q % 2 != 0))


def midpoint_us(t0: int, t1: int) -> int:
    return _round_half_even(t0 + t1, 2)


def restore(xy, present, ts) -> dict:
    xy, present, ts, pb, ti = _validate(xy, present, ts)
    t = len(ti)
    plan = {}                                            # left source index -> (mid, joint mask)
    for k in range(t - 1):
        gap = ti[k + 1] - ti[k]
        if MIN_GAP_30 <= 30 * gap <= MAX_GAP_30:
            valid = pb[k] & pb[k + 1]
            if valid.any():
                plan[k] = (midpoint_us(ti[k], ti[k + 1]), valid)
    if not plan:
        return {"xy": xy.copy(), "present": present.copy(), "ts": ts.copy(),
                "observed": pb.copy(), "estimated": np.zeros_like(pb),
                "exact_source_index": np.arange(t, dtype=np.int64), "identity": True}

    n = t + len(plan)
    out_xy = np.zeros((n, N_JOINTS, 2), dtype=xy.dtype)
    out_pr = np.zeros((n, N_JOINTS), dtype=present.dtype)
    observed = np.zeros((n, N_JOINTS), dtype=bool)
    estimated = np.zeros((n, N_JOINTS), dtype=bool)
    src = np.full(n, -1, dtype=np.int64)
    out_ts = []
    p = 0
    for k in range(t):
        out_xy[p], out_pr[p], observed[p], src[p] = xy[k], present[k], pb[k], k
        out_ts.append(ti[k])
        p += 1
        if k not in plan:
            continue
        mid, valid = plan[k]
        w = (mid - ti[k]) / (ti[k + 1] - ti[k])          # exact int division to nearest float
        left = xy[k][valid].astype(np.float64)
        right = xy[k + 1][valid].astype(np.float64)
        out_xy[p][valid] = ((1.0 - w) * left + w * right).astype(xy.dtype)
        out_pr[p][valid] = 1
        estimated[p] = valid
        out_ts.append(mid)
        p += 1
    ts_out = np.asarray(out_ts, dtype=ts.dtype)
    if [int(v) for v in ts_out] != out_ts:
        raise ValueError(f"timestamps do not fit {ts.dtype}")
    return {"xy": out_xy, "present": out_pr, "ts": ts_out, "observed": observed, "estimated": estimated,
            "exact_source_index": src, "identity": False}
