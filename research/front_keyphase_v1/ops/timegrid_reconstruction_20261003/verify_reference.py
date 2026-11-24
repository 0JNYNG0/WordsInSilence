"""Codex independent reference for verification only; not the evaluated implementation."""
import numpy as np


def round_rational_even(num, den):
    q, r = divmod(int(num), int(den))
    return q + int(2*r > den or (2*r == den and q % 2 != 0))


def independent_restore(xy, mask, timestamps, method):
    xy = np.asarray(xy)
    mask = np.asarray(mask, dtype=bool)
    ts = np.asarray(timestamps, dtype=np.int64)
    if len(ts) == 1 or np.all(np.abs(np.diff(ts) - 1e6/30) <= 2):
        return dict(xy=xy.copy(), present=mask.copy(), ts=ts.copy(), observed=mask.copy(),
                    estimated=np.zeros_like(mask), exact_source_index=np.arange(len(ts)), identity=True)
    n = max(2, round((int(ts[-1])-int(ts[0]))*30/1e6)+1)
    span = int(ts[-1])-int(ts[0])
    times = np.array([round_rational_even(int(ts[0])*(n-1)+span*k, n-1) for k in range(n)], dtype=np.int64)
    out = np.zeros((n, 67, 2), dtype=np.float32)
    valid = np.zeros((n, 67), dtype=bool)
    observed = np.zeros_like(valid)
    estimated = np.zeros_like(valid)
    exact = np.full(n, -1, dtype=np.int64)
    for k, t in enumerate(times):
        differences = np.abs(ts-t)
        closest = int(differences.argmin())
        if differences[closest] <= 1:
            out[k], valid[k], observed[k], exact[k] = xy[closest], mask[closest], mask[closest], closest
            continue
        right = int(np.searchsorted(ts, t))
        left = right-1
        if left < 0 or right >= len(ts) or ts[right]-ts[left] > 2e6/30+2:
            continue
        support = mask[left] & mask[right]
        valid[k] = estimated[k] = support
        if method == 'previous':
            out[k, support] = xy[left, support]
        else:
            alpha = float(t-ts[left])/float(ts[right]-ts[left])
            out[k, support] = ((1-alpha)*xy[left,support].astype(np.float64) + alpha*xy[right,support].astype(np.float64)).astype(np.float32)
    return dict(xy=out, present=valid, ts=times, observed=observed, estimated=estimated,
                exact_source_index=exact, identity=False)
