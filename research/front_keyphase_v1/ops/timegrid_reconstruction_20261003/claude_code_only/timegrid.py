"""Restore the nominal 30 Hz time grid of a RETAINED native sequence (e.g. raw frames[::2]).

Pure NumPy, no external state. restore() only receives the retained arrays: there is no
parameter through which an original dense sequence, a dropped frame or a target length could
enter. The grid is derived from the retained first/last timestamps alone.

Rules (fixed before any result):
  * already-nominal clocks (every interval within 2 us of 1e6/30) and one-frame clips are returned
    as unchanged copies (identity) - no opportunistic repair;
  * otherwise nout = max(2, round(span / P) + 1) points spanning first..last, both ends exact;
  * a grid point within 1 us of a source timestamp copies that source frame verbatim (observed);
  * an interior point is estimated per joint only where BOTH bracketing source frames have the
    joint present and their gap is <= 2P + 2 us; never across missing observations or long gaps;
  * 'previous' holds the left value, 'linear' interpolates; both share the same validity mask;
  * unsupported points are zeros with present False. Estimates are never marked observed.

Integer arithmetic is used for every clock comparison and rounding, so the result does not depend
on float rounding of microsecond timestamps. Rounding is exact rational ties-to-even (Python round
semantics), applied to nout and to the ABSOLUTE grid timestamp first + span*k/(nout-1), not to the
offset alone.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right

import numpy as np

N_JOINTS = 67
NOMINAL_STEP_US = 1e6 / 30.0
EXACT_TOL_US = 1
CLOCK_TOL_US = 2
METHODS = ("linear", "previous")
# integer forms of the rules: P = 1e6/30  ->  compare 30*d against 1e6
_P30 = 1_000_000


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


def is_nominal_clock(ti) -> bool:
    """Every interval within CLOCK_TOL_US of 1e6/30 (integer test: |30d - 1e6| <= 30*tol)."""
    return all(abs(30 * (b - a) - _P30) <= 30 * CLOCK_TOL_US for a, b in zip(ti, ti[1:]))


def _round_half_even(numerator: int, denominator: int) -> int:
    """Exact round(numerator/denominator), ties to even; denominator > 0, any-sign numerator."""
    q, r = divmod(numerator, denominator)                # floor division: 0 <= r < denominator
    return q + int(2 * r > denominator or (2 * r == denominator and q % 2 != 0))


def grid_times(first: int, last: int) -> list:
    """nout = max(2, round(span*30/1e6)+1); grid[k] = round(first + span*k/(nout-1)), ties to even."""
    span = last - first
    nout = max(2, _round_half_even(30 * span, _P30) + 1)
    den = nout - 1
    return [_round_half_even(first * den + span * k, den) for k in range(nout)]


def restore(xy, present, ts, method="linear") -> dict:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    xy, present, ts, pb, ti = _validate(xy, present, ts)
    t = len(ti)
    if t == 1 or is_nominal_clock(ti):
        return {"xy": xy.copy(), "present": present.copy(), "ts": ts.copy(),
                "observed": pb.copy(), "estimated": np.zeros_like(pb),
                "exact_source_index": np.arange(t, dtype=np.int64), "identity": True}

    grid = grid_times(ti[0], ti[-1])
    if any(b <= a for a, b in zip(grid, grid[1:])):
        raise ValueError("duplicate grid times")
    nout = len(grid)
    out_xy = np.zeros((nout, N_JOINTS, 2), dtype=xy.dtype)
    out_pr = np.zeros((nout, N_JOINTS), dtype=present.dtype)
    observed = np.zeros((nout, N_JOINTS), dtype=bool)
    estimated = np.zeros((nout, N_JOINTS), dtype=bool)
    src = np.full(nout, -1, dtype=np.int64)
    for k, g in enumerate(grid):
        j = bisect_left(ti, g)
        best = None
        for i in (j - 1, j):                             # lower index first: ties keep it
            if 0 <= i < t and (best is None or abs(ti[i] - g) < best[0]):
                best = (abs(ti[i] - g), i)
        if best[0] <= EXACT_TOL_US:
            i = best[1]
            out_xy[k], out_pr[k], observed[k], src[k] = xy[i], present[i], pb[i], i
            continue
        lo = bisect_right(ti, g) - 1
        hi = lo + 1
        if lo < 0 or hi >= t:                            # no extrapolation (cannot occur: ends exact)
            continue
        gap = ti[hi] - ti[lo]
        if 30 * gap > 2 * _P30 + 30 * CLOCK_TOL_US:      # gap > 2P + 2 us
            continue
        valid = pb[lo] & pb[hi]
        if not valid.any():
            continue
        left = xy[lo][valid].astype(np.float64)
        if method == "previous":
            est = left
        else:
            w = (g - ti[lo]) / float(gap)
            est = (1.0 - w) * left + w * xy[hi][valid].astype(np.float64)
        out_xy[k][valid] = est.astype(np.float32)
        out_pr[k][valid] = 1
        estimated[k] = valid
    return {"xy": out_xy, "present": out_pr, "ts": np.asarray(grid, dtype=ts.dtype),
            "observed": observed, "estimated": estimated, "exact_source_index": src,
            "identity": False}
