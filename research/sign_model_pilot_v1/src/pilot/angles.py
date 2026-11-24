"""Rotation-invariant hand features, as an alternative to wrist-relative hand coordinates.

Why this module exists
  The 280-channel contract stores each hand as 21 wrist-relative (x, y) pairs. Those coordinates
  describe WHERE the fingers are, which means they also encode which way the hand is pointing: keep
  the hand shape identical and rotate the wrist, and all 42 numbers change. Measured on the
  participant's 100 recordings, the words that hold a fixed hand orientation are tight (ALDA 0.082,
  JADA 0.112) while the words that rotate the hand are loose (SARANG 0.267, INSA 0.361), and
  removing the hand channels cuts the INSA spread by 84.5 percent.

  An angle between two bones does not change when the whole hand rotates, so flexion angles
  separate "what shape is the hand making" from "which way is it pointing". Orientation is not
  discarded: it is kept as an explicit unit vector per hand, two channels instead of eighty-four,
  so the two can be weighted separately instead of one drowning the other.

The angle definition is the one research/front_keyphase_v1 already uses (features.bend_angles):
19 per hand = 3 interior angles for each of 5 fingers, plus 4 adjacent-finger spreads, each
normalised to [0, 1]. That code is read, not modified.

This module ADDS a variant; it does not change the contract. build_channels() and the trained
checkpoint keep the 280-channel layout they were built with.
"""
from __future__ import annotations

import numpy as np

from . import data as D
from .dtw_baseline import BAND_RATIO, MIN_SHARED_CHANNELS, dtw_from_cost

FINGER_CHAINS = ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
                 (0, 13, 14, 15, 16), (0, 17, 18, 19, 20))
SPREAD_PAIRS = ((1, 2), (2, 3), (3, 4), (0, 1))   # index-middle, middle-ring, ring-pinky, thumb-index
N_ANGLES = 19
MID_MCP = 9                                        # wrist -> middle MCP defines hand orientation
EPS = 1e-6


def _angle(u: np.ndarray, v: np.ndarray):
    """Angle between two 2D vectors per frame, normalised to [0,1]. Also returns non-degeneracy."""
    nu = np.linalg.norm(u, axis=-1)
    nv = np.linalg.norm(v, axis=-1)
    ok = (nu > EPS) & (nv > EPS)
    c = np.clip((u * v).sum(axis=-1) / np.where(ok, nu * nv, 1.0), -1.0, 1.0)
    return np.arccos(c) / np.pi, ok


def hand_angles(pts: np.ndarray, valid: np.ndarray):
    """[T,21,2] points and [T,21] validity -> ([T,19] angles in [0,1], [T,19] validity).

    An angle is valid only when every point it is built from is valid and neither vector is
    degenerate. Invalid angles are left at 0 and reported through the mask, never interpolated.
    """
    t = len(pts)
    out = np.zeros((t, N_ANGLES), dtype=np.float32)
    ok = np.zeros((t, N_ANGLES), dtype=bool)
    k = 0
    for ch in FINGER_CHAINS:
        for i in range(1, 4):
            a, b, c = ch[i - 1], ch[i], ch[i + 1]
            v, g = _angle(pts[:, a] - pts[:, b], pts[:, c] - pts[:, b])
            out[:, k] = v
            ok[:, k] = g & valid[:, a] & valid[:, b] & valid[:, c]
            k += 1
    dirs = [pts[:, ch[1]] - pts[:, ch[4]] for ch in FINGER_CHAINS]      # same sense as front_keyphase
    dok = [valid[:, ch[1]] & valid[:, ch[4]] for ch in FINGER_CHAINS]
    for i, j in SPREAD_PAIRS:
        v, g = _angle(dirs[i], dirs[j])
        out[:, k] = v
        ok[:, k] = g & dok[i] & dok[j]
        k += 1
    out[~ok] = 0.0
    return out, ok


def variant(clip: dict, hand_xy: bool = False, use_angles: bool = True,
            orientation: bool = True):
    """A clip's 280 channels -> (values, mask) for one feature choice.

    The body block is always kept: it carries where each hand IS, which is a different axis from
    what shape it makes. Only the 84 wrist-relative hand coordinates are up for replacement.
    """
    x = clip["x"]
    t = len(x)
    vb = D.CH_BODY_XY + D.CH_HAND_XY + D.N_POINTS      # start of the handshape validity block
    vals = [x[:, :D.CH_BODY_XY]]
    masks = [np.repeat(x[:, D.CH_BODY_XY + D.CH_HAND_XY:vb], 2, axis=1)]
    for h in range(2):
        pts = x[:, D.CH_BODY_XY + h * D.N_HAND * 2:
                D.CH_BODY_XY + (h + 1) * D.N_HAND * 2].reshape(t, D.N_HAND, 2)
        hv = x[:, vb + h * D.N_HAND:vb + (h + 1) * D.N_HAND] > 0.5
        if hand_xy:
            vals.append(pts.reshape(t, -1))
            masks.append(np.repeat(hv, 2, axis=1))
        if use_angles:
            a, ok = hand_angles(pts, hv)
            vals.append(a)
            masks.append(ok)
        if orientation:
            d = pts[:, MID_MCP] - pts[:, 0]
            n = np.linalg.norm(d, axis=-1, keepdims=True)
            g = (n[:, 0] > EPS) & hv[:, 0] & hv[:, MID_MCP]
            u = np.where(n > EPS, d / np.maximum(n, EPS), 0.0)
            u[~g] = 0.0
            vals.append(u)
            masks.append(np.repeat(g[:, None], 2, axis=1))
    return (np.concatenate(vals, axis=1).astype(np.float32),
            np.concatenate(masks, axis=1).astype(np.float32) > 0.5)


def frame_cost(va, ma, vb_, mb):
    """Same rule as dtw_baseline._frame_cost, but for any (values, mask) layout."""
    both = ma[:, None, :] & mb[None, :, :]
    diff = va[:, None, :] - vb_[None, :, :]
    shared = both.sum(axis=2)
    total = np.where(both, np.square(diff), 0.0).sum(axis=2)
    cost = np.full(shared.shape, float("inf"))
    ok = shared >= MIN_SHARED_CHANNELS
    cost[ok] = np.sqrt(total[ok] / shared[ok])
    return cost


def distance(a, b, band_ratio: float = BAND_RATIO):
    """a, b are (values, mask) pairs from variant()."""
    if len(a[0]) < 2 or len(b[0]) < 2:
        return None
    return dtw_from_cost(frame_cost(a[0], a[1], b[0], b[1]), band_ratio)
