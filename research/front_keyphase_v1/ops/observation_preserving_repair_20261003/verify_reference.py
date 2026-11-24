"""Independent Codex reference for verification, never used by candidate scoring.

Uses Fraction for timestamps and builds output by sorted time/source records.
The candidate module is deliberately not imported here.
"""
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np


def reference(xy, present, ts):
    records = [(int(t), i, i) for i, t in enumerate(ts)]
    for i in range(len(ts) - 1):
        gap = int(ts[i + 1]) - int(ts[i])
        if Fraction(50_000) <= gap <= Fraction(2_000_060, 30):
            common = present[i].astype(bool) & present[i + 1].astype(bool)
            if common.any():
                records.append((round(Fraction(int(ts[i]) + int(ts[i + 1]), 2)), i, i + 1))
    records.sort()
    n = len(records)
    out = np.zeros((n, 67, 2), dtype=xy.dtype)
    mask = np.zeros((n, 67), dtype=present.dtype)
    obs = np.zeros((n, 67), dtype=bool)
    est = np.zeros_like(obs)
    src = np.full(n, -1, dtype=np.int64)
    for k, (t, lo, hi) in enumerate(records):
        if lo == hi:
            out[k] = xy[lo]
            mask[k] = present[lo]
            obs[k] = present[lo].astype(bool)
            src[k] = lo
        else:
            valid = present[lo].astype(bool) & present[hi].astype(bool)
            alpha = float(Fraction(t - int(ts[lo]), int(ts[hi]) - int(ts[lo])))
            out[k, valid] = ((1 - alpha) * xy[lo, valid].astype(np.float64)
                             + alpha * xy[hi, valid].astype(np.float64)).astype(xy.dtype)
            mask[k, valid] = 1
            est[k, valid] = True
    return dict(xy=out, present=mask, ts=np.array([r[0] for r in records], dtype=ts.dtype),
                observed=obs, estimated=est, exact_source_index=src, identity=n == len(ts))


def assert_contract(out, xy, present, ts):
    s = out['exact_source_index']
    source = s >= 0
    assert np.array_equal(s[source], np.arange(len(ts)))
    assert np.array_equal(out['xy'][source], xy, equal_nan=True)
    assert np.array_equal(out['present'][source], present)
    assert np.array_equal(out['ts'][source], ts)
    assert out['xy'].dtype == xy.dtype and out['present'].dtype == present.dtype
    assert out['ts'].dtype == ts.dtype
    assert np.all(np.diff(out['ts']) > 0)
    assert out['ts'][0] == ts[0] and out['ts'][-1] == ts[-1]
    assert not (out['observed'] & out['estimated']).any()
    assert np.array_equal(out['present'].astype(bool), out['observed'] | out['estimated'])
    assert not out['observed'][~source].any()
    assert not out['estimated'][source].any()
    assert out['present'][~source].astype(bool).any(axis=1).all()
    assert np.all(out['xy'][~source][~out['present'][~source].astype(bool)] == 0)
    assert out['identity'] == bool(source.all())


def filehash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))
