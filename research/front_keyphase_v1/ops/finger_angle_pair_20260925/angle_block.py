"""Finger-joint geometry blocks computed from the EXISTING 280-channel input.

Two width-matched blocks are produced from the same joints, the same validity rule and the
same vector pair (u, v):

  angle block   arccos(u.v / |u||v|) / pi      scale invariant
  chord block   |u - v|                        the same relation in coordinate units

Both are 38 values (19 per hand) + 38 validity flags = 76 channels.

What these are NOT
  * not real 3-D flexion angles. The source is a 2-D projection; a finger pointing at the camera
    and a bent finger can produce the same planar angle. Nothing here recovers depth.
  * not an expert handshape ground truth. No linguistic judgement is attached to any value.
  * not a new observation. Every number is a function of coordinates the model already receives.

Rules kept from the existing contract: an invalid value is 0 AND reported through its mask, never
interpolated, never filled from a neighbour, never clipped to hide an outlier. A value is valid
only when every point it is built from is valid and neither vector is degenerate.

The 280-channel layout (pilot/data.py): 102 body XY | 84 hand XY | 51 body valid | 42 hand valid | 1 dt
Hand points are 42 = 21 left + 21 right, in wrist-relative hand-scale-normalised units. Angles are
invariant to that normalisation; chords are expressed in it, which is the unit the model already uses.
"""
from __future__ import annotations

import numpy as np

FINGER_CHAINS = ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
                 (0, 13, 14, 15, 16), (0, 17, 18, 19, 20))
SPREAD_PAIRS = ((1, 2), (2, 3), (3, 4), (0, 1))
N_ANGLES = 19
EPS = 1e-6

CH_BODY_XY, CH_HAND_XY = 102, 84
CH_BODY_VALID, CH_HAND_VALID = 51, 42
HAND_XY0 = CH_BODY_XY                                   # 102
HAND_VALID0 = CH_BODY_XY + CH_HAND_XY + CH_BODY_VALID   # 237
TOTAL_CHANNELS = 280
N_HAND = 21
EXTRA_VALUES = 2 * N_ANGLES     # 38
EXTRA_WIDTH = 2 * EXTRA_VALUES  # 76


def _triplets():
    """The (a, b, c) definitions, in the fixed order the 19 channels use."""
    out = []
    for ch in FINGER_CHAINS:
        for i in range(1, 4):
            out.append(('interior', ch[i - 1], ch[i], ch[i + 1]))
    for i, j in SPREAD_PAIRS:
        out.append(('spread', i, j, None))
    assert len(out) == N_ANGLES
    return tuple(out)


TRIPLETS = _triplets()


def hand_points(x: np.ndarray):
    """[T,280] -> (pts[T,2,21,2], valid[T,2,21]) for the left and right hand."""
    if x.ndim != 2 or x.shape[1] != TOTAL_CHANNELS:
        raise ValueError(f"x must be [T,{TOTAL_CHANNELS}], got {x.shape}")
    t = x.shape[0]
    xy = x[:, HAND_XY0:HAND_XY0 + CH_HAND_XY].reshape(t, 2, N_HAND, 2)
    ok = x[:, HAND_VALID0:HAND_VALID0 + CH_HAND_VALID].reshape(t, 2, N_HAND) > 0.5
    return np.ascontiguousarray(xy, dtype=np.float64), np.ascontiguousarray(ok)


def _vectors(pts: np.ndarray, valid: np.ndarray, kind, a, b, c):
    """Return (u, v, ok) for one channel definition. pts [T,21,2], valid [T,21]."""
    if kind == 'interior':
        u = pts[:, a] - pts[:, b]
        v = pts[:, c] - pts[:, b]
        ok = valid[:, a] & valid[:, b] & valid[:, c]
    else:                                             # spread between two finger directions
        ci, cj = FINGER_CHAINS[a], FINGER_CHAINS[b]
        u = pts[:, ci[1]] - pts[:, ci[4]]
        v = pts[:, cj[1]] - pts[:, cj[4]]
        ok = (valid[:, ci[1]] & valid[:, ci[4]] & valid[:, cj[1]] & valid[:, cj[4]])
    return u, v, ok


def blocks(x: np.ndarray):
    """[T,280] -> (angle[T,76], chord[T,76]) float32.

    Channel order in each block: 38 values (left 19 then right 19) followed by their 38 masks.
    The two blocks share the validity mask exactly.
    """
    pts, ok = hand_points(x)
    t = x.shape[0]
    ang = np.zeros((t, EXTRA_VALUES), dtype=np.float64)
    cho = np.zeros((t, EXTRA_VALUES), dtype=np.float64)
    msk = np.zeros((t, EXTRA_VALUES), dtype=bool)
    for h in range(2):
        for k, (kind, a, b, c) in enumerate(TRIPLETS):
            u, v, joint_ok = _vectors(pts[:, h], ok[:, h], kind, a, b, c)
            nu = np.linalg.norm(u, axis=-1)
            nv = np.linalg.norm(v, axis=-1)
            good = joint_ok & (nu > EPS) & (nv > EPS)
            cos = np.clip((u * v).sum(axis=-1) / np.where(good, nu * nv, 1.0), -1.0, 1.0)
            idx = h * N_ANGLES + k
            ang[good, idx] = np.arccos(cos[good]) / np.pi
            cho[good, idx] = np.linalg.norm(u - v, axis=-1)[good]
            msk[:, idx] = good
    angle = np.concatenate([ang * msk, msk.astype(np.float64)], axis=1)
    chord = np.concatenate([cho * msk, msk.astype(np.float64)], axis=1)
    for name, arr in (('angle', angle), ('chord', chord)):
        if not np.isfinite(arr).all():
            raise ValueError(f"non-finite value in the {name} block")
    return angle.astype(np.float32), chord.astype(np.float32)
