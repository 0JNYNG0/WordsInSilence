"""Nearest-template DTW over exactly the same channels and the same training clips.

The comparison is only fair if nothing differs but the decision rule, so this baseline reads the
identical feature tensor the classifier sees, uses the identical train/validation split, and
predicts the class of whichever training clip is nearest.

Distance is computed on the valid part of each frame: a channel marked invalid on either side is
excluded from that frame's cost, and the cost is the mean over the channels that remained. A
frame pair with nothing in common is inadmissible rather than free.
"""
from __future__ import annotations

import numpy as np

from .data import CH_BODY_XY, CH_HAND_XY, N_HAND, N_POINTS

INF = float("inf")
BAND_RATIO = 0.2                 # same Sakoe-Chiba ratio the production web path uses
MIN_SHARED_CHANNELS = 8


def _frame_cost(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """[Ta,C],[Tb,C] -> [Ta,Tb] mean squared difference over jointly valid coordinates."""
    ta, tb = len(a), len(b)
    xy_a, xy_b = a[:, :CH_BODY_XY + CH_HAND_XY], b[:, :CH_BODY_XY + CH_HAND_XY]
    v_start = CH_BODY_XY + CH_HAND_XY
    va = a[:, v_start:v_start + N_POINTS + N_HAND * 2]
    vb = b[:, v_start:v_start + N_POINTS + N_HAND * 2]
    # each validity flag covers one point, i.e. two coordinate channels
    va2 = np.repeat(va, 2, axis=1)
    vb2 = np.repeat(vb, 2, axis=1)

    both = (va2[:, None, :] > 0) & (vb2[None, :, :] > 0)
    diff = xy_a[:, None, :] - xy_b[None, :, :]
    shared = both.sum(axis=2)
    total = np.where(both, np.square(diff), 0.0).sum(axis=2)
    cost = np.full((ta, tb), INF)
    ok = shared >= MIN_SHARED_CHANNELS
    cost[ok] = np.sqrt(total[ok] / shared[ok])
    return cost


def dtw_distance(a: np.ndarray, b: np.ndarray, band_ratio: float = BAND_RATIO):
    return dtw_from_cost(_frame_cost(a, b), band_ratio)


def dtw_from_cost(cost: np.ndarray, band_ratio: float = BAND_RATIO):
    """Banded DTW over a precomputed frame-cost matrix, so another feature layout can reuse it."""
    na, nb = cost.shape
    if na < 2 or nb < 2:
        return None
    band = max(1, int(round(band_ratio * max(na, nb))))
    acc = np.full((na + 1, nb + 1), INF)
    steps = np.zeros((na + 1, nb + 1), dtype=np.int32)
    acc[0, 0] = 0.0
    for i in range(1, na + 1):
        centre = (i - 1) * (nb - 1) / (na - 1)
        jlo = max(1, int(np.floor(centre - band)) + 1)
        jhi = min(nb, int(np.ceil(centre + band)) + 1)
        for j in range(jlo, jhi + 1):
            c = cost[i - 1, j - 1]
            if not np.isfinite(c):
                continue
            prev = (acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1])
            k = int(np.argmin(prev))
            if not np.isfinite(prev[k]):
                continue
            pi, pj = ((i - 1, j - 1), (i - 1, j), (i, j - 1))[k]
            acc[i, j] = prev[k] + c
            steps[i, j] = steps[pi, pj] + 1
    if not np.isfinite(acc[na, nb]) or steps[na, nb] == 0:
        return None
    return float(acc[na, nb] / steps[na, nb])


def predict(train_clips: list, query_clip: dict) -> dict:
    """Nearest training template. Returns the predicted class and every distance."""
    rows = []
    for t in train_clips:
        d = dtw_distance(query_clip["x"], t["x"])
        rows.append({"sample_id": t["sample_id"], "label": t["label"], "distance": d})
    scored = [r for r in rows if r["distance"] is not None]
    if not scored:
        return {"status": "INSUFFICIENT", "predicted": None,
                "reason": "no training template produced an admissible warping path"}
    scored.sort(key=lambda r: r["distance"])
    best = scored[0]
    second = next((r for r in scored if r["label"] != best["label"]), None)
    return {
        "status": "OK",
        "predicted": best["label"],
        "nearest_sample_id": best["sample_id"],
        "distance": round(best["distance"], 6),
        "margin_to_other_class": (round(second["distance"] - best["distance"], 6)
                                  if second else None),
        "compared": len(scored),
    }
