"""Constrained DTW over grouped features.

Frame cost is a weighted mean over the feature groups that took part. Two rules keep a cheap
distance from being a meaningless one:

* A group present on one side only is charged a fixed mismatch penalty rather than skipped.
  Otherwise a query that simply loses a hand would score better than one that gets it wrong.
* A group absent on BOTH sides is a genuine agreement (both are resting) and costs nothing, but
  the evidence weight behind each cell is carried along the warping path. A match built on
  almost no evidence is reported as INSUFFICIENT instead of being sold as a small distance.

Version note (runtime v2, 2026-09-13). The two rules above used to share one denominator: a group
present on one side only was charged its penalty AND counted as evidence that a comparison had
happened. Independent review reproduced the consequence - eight frames with every group present on
one side and every group absent on the other returned status OK, mean_evidence_fraction 1.0 and
distance 1.5, with zero groups actually observed on both sides. Cost and evidence are now separate:
the penalty still charges, but only a group measured on BOTH sides counts as evidence. A group
flagged present whose features are NaN or Inf is treated as unobserved rather than as a distance of
zero. Frozen experiment results produced before this change were computed under runtime v1 and are
left untouched.
"""
from __future__ import annotations

import numpy as np

from .features import GROUP_ORDER, GROUPS, TOTAL_WEIGHT

ONE_SIDED_PENALTY = 1.5           # same units as a normalized group distance
MIN_PATH_EVIDENCE_FRACTION = 0.35  # mean fraction of TOTAL_WEIGHT compared along the path
INF = float("inf")

_SLICES = {}
_c = 0
for _g in GROUP_ORDER:
    _w = GROUPS[_g][0]
    _SLICES[_g] = slice(_c, _c + _w)
    _c += _w
_WEIGHTS = np.array([GROUPS[g][1] for g in GROUP_ORDER], dtype=np.float64)


def _nanmean(x, axis):
    """np.nanmean without the all-NaN warning: those slices are masked out by the caller."""
    n = np.sum(~np.isnan(x), axis=axis)
    s = np.nansum(x, axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def penalty_vector(one_sided_penalty) -> np.ndarray:
    """Accept a scalar (variant A) or a per-group mapping (variant B)."""
    if one_sided_penalty is None:
        return np.full(len(GROUP_ORDER), ONE_SIDED_PENALTY, dtype=np.float64)
    if isinstance(one_sided_penalty, dict):
        return np.array([float(one_sided_penalty[g]) for g in GROUP_ORDER], dtype=np.float64)
    return np.full(len(GROUP_ORDER), float(one_sided_penalty), dtype=np.float64)


SYSTEMATIC_USE_RATE = 0.5   # a group present in at least this share of frames counts as "used"
RUNTIME_VERSION = "dtw_runtime_v2_observation_support"


def weight_vector(group_weights=None) -> np.ndarray:
    if not group_weights:
        return _WEIGHTS
    return np.array([float(group_weights.get(g, GROUPS[g][1])) for g in GROUP_ORDER],
                    dtype=np.float64)


def _transient_mask(a: dict, b: dict) -> np.ndarray:
    """Which groups are used by BOTH sequences overall.

    A group both sides use, missing from one frame, is a detection dropout - a camera fact, not
    a sign difference. A group one side uses throughout and the other never uses (one hand vs
    two) is a real difference and must still be charged.
    """
    ra = a["present"].mean(axis=0)
    rb = b["present"].mean(axis=0)
    return (ra >= SYSTEMATIC_USE_RATE) & (rb >= SYSTEMATIC_USE_RATE)


SIDE_GROUPS = {"L": ("arm_L", "shape_L", "orient_L", "handpos_L", "facerel_L", "motion_L"),
               "R": ("arm_R", "shape_R", "orient_R", "handpos_R", "facerel_R", "motion_R")}


def applied_weights(a, b, group_weights=None, hand_participation: bool = False) -> np.ndarray:
    """The weight vector actually used for a comparison, after participation is applied.

    Exposed so a diagnostic reports the weights that were used rather than the module defaults.
    """
    w = weight_vector(group_weights)
    if hand_participation:
        w = _participation_weights(a, b, w)
    return w


def _participation_weights(a, b, weights):
    """Drop a hand that is resting in BOTH sequences; keep charging when only one side uses it.

    The NIKL instructor's non-dominant hand hangs in frame on 8 of the 18 words. Requiring an
    attempt to match a hand that is not performing the sign charged a constant penalty whenever
    the tester's resting hand sat outside their own camera frame.
    """
    pa = a.get("participating", {"L": True, "R": True})
    pb = b.get("participating", {"L": True, "R": True})
    w = weights.copy()
    for side, groups in SIDE_GROUPS.items():
        if not pa.get(side, True) and not pb.get(side, True):
            for g in groups:
                w[GROUP_ORDER.index(g)] = 0.0
    if not (pa.get("L", True) and pa.get("R", True)
            and pb.get("L", True) and pb.get("R", True)):
        w[GROUP_ORDER.index("bimanual")] = 0.0   # a two-hand relation needs two working hands
    return w


def frame_cost_matrix(a: dict, b: dict, one_sided_penalty=None, one_sided_mode: str = "flat",
                      group_weights=None, hand_participation: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(cost[na,nb], evidence_fraction[na,nb]) between two sampled sequences."""
    pen = penalty_vector(one_sided_penalty)
    weights = applied_weights(a, b, group_weights, hand_participation)
    total_weight = float(weights.sum())
    fa, fb = a["features"], b["features"]
    pa, pb = a["present"], b["present"]
    na, nb = len(fa), len(fb)

    per_group = np.empty((len(GROUP_ORDER), na, nb), dtype=np.float64)
    for gi, g in enumerate(GROUP_ORDER):
        sl = _SLICES[g]
        d = fa[:, None, sl] - fb[None, :, sl]
        with np.errstate(invalid="ignore"):
            per_group[gi] = np.sqrt(_nanmean(np.square(d), axis=2))
    per_group = np.moveaxis(per_group, 0, 2)        # [na,nb,G]

    both = pa[:, None, :] & pb[None, :, :]
    one = pa[:, None, :] ^ pb[None, :, :]
    if one_sided_mode == "systematic_only":
        # both sides use this group overall -> a one-sided frame is dropout, so drop the cell
        one = one & ~_transient_mask(a, b)[None, None, :]

    # present on both sides but the group cannot actually be compared: the flag says observed and
    # the numbers say otherwise. Treating that as a distance of zero would score it a perfect
    # match, so it is moved out of the observed set entirely - no cost, and no evidence either.
    #
    # Presence is a GROUP-level contract, so the group counts only when EVERY component it needs
    # is finite on both sides. Checking the group distance alone was not enough: _nanmean skips a
    # NaN component and averages the rest, so a single NaN per group still returned a finite
    # distance of 0 and full evidence. Independent review reproduced exactly that.
    comp_a = np.empty((na, len(GROUP_ORDER)), dtype=bool)
    comp_b = np.empty((nb, len(GROUP_ORDER)), dtype=bool)
    for gi, g in enumerate(GROUP_ORDER):
        sl = _SLICES[g]
        comp_a[:, gi] = np.isfinite(fa[:, sl]).all(axis=1)
        comp_b[:, gi] = np.isfinite(fb[:, sl]).all(axis=1)
    complete = comp_a[:, None, :] & comp_b[None, :, :]
    measurable = complete & np.isfinite(per_group)
    observed = both & measurable
    unmeasurable = both & ~measurable
    counted = observed | one

    contrib = np.where(observed, np.nan_to_num(per_group, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    contrib = np.where(one, pen, contrib)
    w = np.where(counted, weights, 0.0)
    weight = w.sum(axis=2)
    total = (contrib * w).sum(axis=2)

    # weight == 0 means neither side offered anything here: no disagreement, so no cost.
    with np.errstate(invalid="ignore", divide="ignore"):
        cost = np.where(weight > 0, total / np.where(weight > 0, weight, 1.0), 0.0)

    # EVIDENCE IS NOT COST. Only a group measured on both sides is evidence that a comparison
    # happened; a one-sided group is charged but proves nothing about agreement, and an
    # unmeasurable one proves nothing either. The path-level guard then refuses a match assembled
    # out of such cells.
    evidence_weight = np.where(observed, weights, 0.0).sum(axis=2)
    _ = unmeasurable       # named for readability; its exclusion is the point
    return cost, evidence_weight / max(total_weight, 1e-9)


def group_residuals(a: dict, b: dict, band_ratio: float = 0.2, one_sided_penalty=None,
                    one_sided_mode: str = "flat", group_weights=None,
                    hand_participation: bool = False) -> dict:
    """Diagnostic: split one comparison into its feature groups along the optimal path.

    Answers the question a single distance cannot: was a group's own distance large and merely
    outweighed, or was it small to begin with?
    """
    cost, evidence = frame_cost_matrix(a, b, one_sided_penalty, one_sided_mode, group_weights,
                                       hand_participation)
    path = _optimal_path(cost, band_ratio)
    if not path:
        return {"status": "INSUFFICIENT", "groups": {}}

    fa, fb, pa, pb = a["features"], b["features"], a["present"], b["present"]
    weights = applied_weights(a, b, group_weights, hand_participation)
    total_applied = float(weights.sum())
    steps = len(path)
    out = {}
    for gi, g in enumerate(GROUP_ORDER):
        sl = _SLICES[g]
        vals, one_sided, unmeasurable, unobserved = [], 0, 0, 0
        for i, j in path:
            if pa[i, gi] and pb[j, gi]:
                # every component the group needs must be finite on both sides, for the same
                # reason as in frame_cost_matrix: one NaN component must not read as a match
                whole = (np.isfinite(fa[i, sl]).all() and np.isfinite(fb[j, sl]).all())
                v = (float(np.sqrt(_nanmean(np.square(fa[i, sl] - fb[j, sl]), axis=0)))
                     if whole else float("nan"))
                if whole and np.isfinite(v):
                    vals.append(v)
                else:
                    unmeasurable += 1
            elif pa[i, gi] ^ pb[j, gi]:
                one_sided += 1
            else:
                unobserved += 1
        w = float(weights[gi])
        mean = float(np.mean(vals)) if vals else None
        out[g] = {
            "weight_applied": w,
            "weight_default": GROUPS[g][1],
            "compared_steps": len(vals),
            "one_sided_steps": one_sided,
            "unmeasurable_steps": unmeasurable,
            "unobserved_steps": unobserved,
            "support_fraction": round(len(vals) / steps, 4) if steps else 0.0,
            "mean_distance": round(mean, 5) if mean is not None else None,
            "weighted_share": (round(mean * w / total_applied, 5)
                               if mean is not None and total_applied > 0 else None),
        }
    measured_weight = sum(out[g]["weight_applied"] for g in GROUP_ORDER
                          if out[g]["mean_distance"] is not None)
    return {
        "status": "OK", "path_length": steps, "groups": out,
        "runtime_version": RUNTIME_VERSION,
        "total_weight": TOTAL_WEIGHT,
        "total_weight_applied": round(total_applied, 5),
        "applied_weights": {g: float(weights[i]) for i, g in enumerate(GROUP_ORDER)},
        "support": {
            "measured_weight": round(measured_weight, 5),
            "measured_weight_fraction": (round(measured_weight / total_applied, 4)
                                         if total_applied > 0 else 0.0),
            "groups_measured": sum(1 for g in GROUP_ORDER
                                   if out[g]["mean_distance"] is not None),
            "groups_with_nonzero_weight": sum(1 for g in GROUP_ORDER
                                              if out[g]["weight_applied"] > 0),
            "meaning": "measured means observed on BOTH sides with finite features; a one-sided "
                       "or unmeasurable group is charged in the cost but is not evidence",
        },
    }


def _optimal_path(cost: np.ndarray, band_ratio: float) -> list[tuple[int, int]]:
    na, nb = cost.shape
    if na < 2 or nb < 2:
        return []
    band = max(1, int(round(band_ratio * max(na, nb))))
    acc = np.full((na + 1, nb + 1), INF)
    back = np.full((na + 1, nb + 1), -1, dtype=np.int8)
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
            acc[i, j] = prev[k] + c
            back[i, j] = k
    if back[na, nb] < 0:
        return []
    path, i, j = [], na, nb
    while i > 0 and j > 0 and back[i, j] >= 0:
        path.append((i - 1, j - 1))
        k = back[i, j]
        i, j = ((i - 1, j - 1), (i - 1, j), (i, j - 1))[k]
    return list(reversed(path))


def dtw(a: dict, b: dict, band_ratio: float = 0.2, one_sided_penalty=None,
        one_sided_mode: str = "flat", group_weights=None, hand_participation: bool = False) -> dict:
    """Sakoe-Chiba banded DTW on normalized progress; distance / path length."""
    cost, evidence = frame_cost_matrix(a, b, one_sided_penalty, one_sided_mode, group_weights,
                                       hand_participation)
    na, nb = cost.shape
    if na < 2 or nb < 2:
        return {"status": "INSUFFICIENT", "distance": None, "path_length": 0,
                "reason": f"sequence too short ({na},{nb})"}

    band = max(1, int(round(band_ratio * max(na, nb))))
    acc = np.full((na + 1, nb + 1), INF)
    steps = np.zeros((na + 1, nb + 1), dtype=np.int32)
    ev = np.zeros((na + 1, nb + 1), dtype=np.float64)
    acc[0, 0] = 0.0
    for i in range(1, na + 1):
        # band on normalized progress, so sequences of different length stay comparable
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
            ev[i, j] = ev[pi, pj] + evidence[i - 1, j - 1]

    if not np.isfinite(acc[na, nb]) or steps[na, nb] == 0:
        return {"status": "INSUFFICIENT", "distance": None, "path_length": 0,
                "reason": "no admissible warping path"}

    n = int(steps[na, nb])
    mean_evidence = float(ev[na, nb] / n)
    if mean_evidence < MIN_PATH_EVIDENCE_FRACTION:
        return {"status": "INSUFFICIENT", "distance": None, "path_length": n,
                "mean_evidence_fraction": round(mean_evidence, 4),
                "reason": f"matched on {mean_evidence:.2f} of the feature weight, "
                          f"minimum is {MIN_PATH_EVIDENCE_FRACTION}"}
    return {
        "status": "OK",
        "distance": float(acc[na, nb] / n),
        "path_length": n,
        "mean_evidence_fraction": round(mean_evidence, 4),
        "band_ratio": band_ratio,
    }
