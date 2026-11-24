"""Axis decomposition: score handshape / location / movement / orientation separately.

Five attempts to tune the blended scalar all failed the same way - one knob was being asked to
do two opposite jobs at once, so every setting that rescued the one-handed words degraded the
two-handed ones. This scores each linguistic axis on its own and combines RANKS, not distances,
so no axis can win on unit choice and no scale constant needs tuning.

The time alignment stays shared: a sign is one coordinated performance, so matching the handshape
at one moment and the location at another would not be a match at all.
"""
from __future__ import annotations

import numpy as np

from .dtw import group_residuals

# axis -> {group: weight within that axis}. Weights only balance groups inside one axis; they
# never trade one axis against another, because axes are combined by rank.
AXES = {
    "handshape": {"shape_L": 1.0, "shape_R": 1.0},
    "location": {"handpos_L": 1.5, "handpos_R": 1.5, "facerel_L": 1.0, "facerel_R": 1.0,
                 "bimanual": 1.0},
    "movement": {"arm_L": 1.0, "arm_R": 1.0, "motion_L": 1.0, "motion_R": 1.0},
    "orientation": {"orient_L": 1.0, "orient_R": 1.0},
}
AXIS_ORDER = list(AXES)


def axis_distances(query: dict, ref: dict, cost_cfg: dict) -> dict:
    """Per-axis distance between one attempt and one reference, on a shared alignment."""
    res = group_residuals(query, ref, **cost_cfg)
    if res["status"] != "OK":
        return {a: None for a in AXIS_ORDER}
    groups = res["groups"]
    out = {}
    for axis, members in AXES.items():
        num = den = 0.0
        for g, w in members.items():
            v = groups.get(g)
            if v and v["mean_distance"] is not None:
                num += v["mean_distance"] * w
                den += w
        out[axis] = (num / den) if den > 0 else None
    return out


def _ranks(values: list, worst: int) -> list:
    """Competition ranking; an unmeasurable axis is placed last rather than skipped."""
    order = sorted((v, i) for i, v in enumerate(values) if v is not None)
    out = [worst] * len(values)
    for pos, (_v, i) in enumerate(order):
        out[i] = pos + 1
    return out


def predict_axes(observation, reference_set: list[dict], frozen_config: dict) -> dict:
    """Same contract as predict(): never receives the requested or ground-truth word."""
    from . import features as F

    out = {"schema_version": "wis_result_axes_v1", "observation_status": "OK",
           "recognition": "UNCONFIGURED", "execution_assessment": "UNCONFIGURED",
           "accepted": None, "ranking": [], "insufficient_reason": None}
    try:
        seq = F.sequence_features(observation)
        query = F.uniform_time_sample(seq, int(frozen_config["sampling_A"]["budget"]))
    except F.FeatureError as exc:
        out["observation_status"] = "INSUFFICIENT"
        out["insufficient_reason"] = str(exc)
        return out

    cost_cfg = {k: v for k, v in frozen_config.get("cost", {}).items()
                if k in ("one_sided_penalty", "one_sided_mode", "group_weights",
                         "hand_participation")}
    per_ref = [axis_distances(query, r, cost_cfg) for r in reference_set]
    n = len(reference_set)
    worst = n
    rank_by_axis = {a: _ranks([d[a] for d in per_ref], worst) for a in AXIS_ORDER}

    rows = []
    for i, ref in enumerate(reference_set):
        ranks = {a: rank_by_axis[a][i] for a in AXIS_ORDER}
        measurable = [a for a in AXIS_ORDER if per_ref[i][a] is not None]
        rows.append({
            "clip_id": ref["clip_id"], "word_id": ref["word_id"],
            "axis_rank": ranks,
            "axis_distance": {a: (round(per_ref[i][a], 5) if per_ref[i][a] is not None else None)
                              for a in AXIS_ORDER},
            "measurable_axes": measurable,
            "combined": float(np.mean([ranks[a] for a in AXIS_ORDER])),
        })
    rows.sort(key=lambda r: (r["combined"], r["axis_rank"]["handshape"]))
    for i, r in enumerate(rows):
        r["rank"] = i + 1
    out["ranking"] = rows
    if not rows:
        out["observation_status"] = "INSUFFICIENT"
        out["insufficient_reason"] = "no reference produced an admissible alignment"
        return out

    best = rows[0]
    out["top1_word_id"] = best["word_id"]
    out["top3_word_ids"] = [r["word_id"] for r in rows[:3]]
    out["top1_axis_rank"] = best["axis_rank"]
    out["margin_to_second"] = (round(rows[1]["combined"] - best["combined"], 4)
                               if len(rows) > 1 else None)
    out["recognition_note"] = (
        "combined rank over four axes. A rank is not a distance and not a score: it says only "
        "where this reference sat among the candidates on that axis.")
    out["execution_assessment_note"] = (
        "UNCONFIGURED: no pass line. Per-axis ranks show WHICH axis disagreed, not whether the "
        "performance was correct.")
    return out
