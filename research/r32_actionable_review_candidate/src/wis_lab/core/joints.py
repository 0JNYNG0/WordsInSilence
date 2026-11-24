"""Per-joint difference, for colouring a skeleton.

The axis bars say WHICH part differs; this says WHERE inside that part. Both are difference
reports, never verdicts: the colour encodes how far a joint sits from the reference, not whether
the performance was right. Nobody has told this system where "wrong" begins.

Observation support travels with every number here. A joint compared on ONE step of the warping
path used to arrive with a mean, a status of OK and nothing to say it rested on a single pair, so
a barely-seen joint read exactly like a fully tracked one. Each joint now reports how many of the
path's pairs actually compared it, and a joint that was never comparable is listed as unmeasured
rather than filled in with a zero.

The drawn pair carries both sides: the query frame, the reference frame it was aligned to, the
per-joint difference on that one pair, and which coordinate space every landmark lives in. Those
are additions - every field that existed keeps its shape.

Comparability follows the same rule as the DTW runtime: presence is a GROUP-level contract, so a
pair counts only when every component that group needs is finite on BOTH sides, and then the
joint's own two dimensions must be finite too. Aligning this with the cost path is why some joints
that previously produced a number now report no comparable pair - the distances, the scales and
the thresholds are unchanged.
"""
from __future__ import annotations

import numpy as np

from .dtw import _SLICES, _optimal_path, frame_cost_matrix
from .features import GROUP_ORDER

# landmark index -> (feature group, offset of its 2 dims inside that group)
LANDMARK_MAP: dict[int, tuple[str, int]] = {}
for _side, _hand_base, _pose_elbow, _pose_wrist in (("L", 33, 13, 15), ("R", 54, 14, 16)):
    LANDMARK_MAP[_hand_base] = (f"handpos_{_side}", 0)          # the wrist itself
    for _j in range(1, 21):                                      # the 20 finger points
        LANDMARK_MAP[_hand_base + _j] = (f"shape_{_side}", (_j - 1) * 2)
    LANDMARK_MAP[_pose_elbow] = (f"arm_{_side}", 0)
    LANDMARK_MAP[_pose_wrist] = (f"arm_{_side}", 2)

_USED_GROUPS = sorted({g for g, _ in LANDMARK_MAP.values()})

# Which coordinate space each group lives in. shape_* is (hand point - wrist) / hand scale, so its
# origin IS the wrist and its unit is that hand. handpos_* and arm_* are body-normalised. Drawing
# one next to the other would put a distance on screen that nothing measured, so the two spaces
# never share a panel.
GROUP_SPACE = {"shape_L": "hand_local_wrist_normalised", "shape_R": "hand_local_wrist_normalised",
               "handpos_L": "body_normalised", "handpos_R": "body_normalised",
               "arm_L": "body_normalised", "arm_R": "body_normalised"}
PANEL_GROUPS = {"L": "shape_L", "R": "shape_R"}


def _whole_group(features: np.ndarray, present: np.ndarray, gi: dict) -> dict:
    """(group, frame) -> the group is present AND every component it needs is finite."""
    out = {}
    for g in _USED_GROUPS:
        sl = _SLICES[g]
        out[g] = present[:, gi[g]] & np.isfinite(features[:, sl]).all(axis=1)
    return out


def joint_distances(query: dict, ref: dict, cost_cfg: dict) -> dict:
    """Per-landmark distance along the shared path, with the pairs each one was measured on."""
    cost, _ = frame_cost_matrix(query, ref, **cost_cfg)
    path = _optimal_path(cost, 0.2)
    if not path:
        return {}
    fa, fb = query["features"], ref["features"]
    gi = {g: i for i, g in enumerate(GROUP_ORDER)}
    ok_a = _whole_group(fa, query["present"], gi)
    ok_b = _whole_group(fb, ref["present"], gi)

    per = {lm: [] for lm in LANDMARK_MAP}
    worst_pair, worst_total = None, -1.0
    for i, j in path:
        total, n = 0.0, 0
        for lm, (group, off) in LANDMARK_MAP.items():
            if not (ok_a[group][i] and ok_b[group][j]):
                continue
            s = _SLICES[group].start + off
            d = float(np.linalg.norm(fa[i, s:s + 2] - fb[j, s:s + 2]))
            if not np.isfinite(d):
                continue
            per[lm].append(d)
            total += d
            n += 1
        if n and total / n > worst_total:
            worst_total, worst_pair = total / n, (i, j)
    return {"per_joint": {lm: float(np.mean(v)) for lm, v in per.items() if v},
            "compared_pairs": {lm: len(v) for lm, v in per.items()},
            "path_pairs": len(path),
            "worst_pair": worst_pair,
            "worst_frame": worst_pair[0] if worst_pair else None}


def joint_deviation(query: dict, ref: dict, frozen_config: dict) -> dict:
    """Per-joint deviation in reference-derived units, plus one frame to draw."""
    scales = ((frozen_config.get("handshape_check") or {}).get("joint_scales") or {})
    if not scales:
        return {"status": "UNAVAILABLE", "reason": "no joint scales in the frozen config"}
    cost_cfg = {k: v for k, v in frozen_config.get("cost", {}).items()
                if k in ("one_sided_penalty", "one_sided_mode", "group_weights",
                         "hand_participation")}
    res = joint_distances(query, ref, cost_cfg)
    if not res:
        return {"status": "UNAVAILABLE", "reason": "no admissible warping path"}

    # Per-JOINT denominators blow up on joints that barely move between words: the left elbow has a
    # median different-word distance of 0.09, so dividing by it made correct attempts look worse
    # than deliberate errors. Each joint is scaled by its GROUP instead - inside a group the units
    # are already the same, so no tiny denominator can appear.
    pairs = res["path_pairs"]
    dev, support = {}, {}
    for lm, (group, _) in LANDMARK_MAP.items():
        compared = int(res["compared_pairs"].get(lm, 0))
        entry = {"group": group, "compared_pairs": compared, "path_pairs": pairs,
                 "time_fraction": round(compared / pairs, 4) if pairs else 0.0}
        d = res["per_joint"].get(lm)
        scale = scales.get(group)
        if d is None:
            entry.update(measured=False, reason="NO_COMPARABLE_PAIR")
        elif not scale:
            entry.update(measured=False, reason="NO_SCALE_FOR_GROUP")
        else:
            entry.update(measured=True, reason=None)
            dev[lm] = round(d / scale, 3)
        support[lm] = entry
    if not dev:
        return {"status": "UNAVAILABLE", "reason": "no joint could be scaled",
                "support": {"path_pairs": pairs, "per_joint": support}}

    measured = [lm for lm in support if support[lm]["measured"]]
    partial_time = [lm for lm in measured if support[lm]["time_fraction"] < 1.0]
    times = [support[lm]["time_fraction"] for lm in measured]
    partial = bool(measured) and (len(measured) < len(LANDMARK_MAP) or bool(partial_time))

    i, j = res["worst_pair"]
    gi = {g: i2 for i2, g in enumerate(GROUP_ORDER)}
    ok_a = _whole_group(query["features"], query["present"], gi)
    ok_b = _whole_group(ref["features"], ref["present"], gi)
    pose, ref_pose, frame_cmp, frame_dev = {}, {}, {}, {}
    for lm, (group, off) in LANDMARK_MAP.items():
        s = _SLICES[group].start + off
        qxy, rxy = query["features"][i, s:s + 2], ref["features"][j, s:s + 2]
        q_ok = bool(ok_a[group][i]) and bool(np.isfinite(qxy).all())
        r_ok = bool(ok_b[group][j]) and bool(np.isfinite(rxy).all())
        # frame_features keeps its original [x, y] shape so existing consumers do not break.
        # Drawing a joint is not the same as having compared it - the reference side may not have
        # carried that group on this pair - so that distinction lives in its own map.
        if q_ok:
            pose[lm] = [round(float(qxy[0]), 4), round(float(qxy[1]), 4)]
            frame_cmp[lm] = r_ok
        if r_ok:
            ref_pose[lm] = [round(float(rxy[0]), 4), round(float(rxy[1]), 4)]
        if q_ok and r_ok and scales.get(group):
            # the difference ON THIS PAIR, in the same group unit the whole-path deviation uses.
            # It is a per-frame number and must not be read as the attempt's deviation.
            frame_dev[lm] = round(float(np.linalg.norm(qxy - rxy)) / scales[group], 3)
    return {
        "status": "OK",
        "deviation": dev,
        "support": {
            "path_pairs": pairs,
            "joints_expected": len(LANDMARK_MAP), "joints_measured": len(measured),
            "joints_unmeasured": sorted(lm for lm in support if not support[lm]["measured"]),
            "time_fraction_min": round(min(times), 4) if times else 0.0,
            "time_fraction_mean": round(sum(times) / len(times), 4) if times else 0.0,
            "joints_with_partial_time": sorted(partial_time),
            "partial": partial,
            "partial_reason": ([] if not partial else
                               (["unmeasured_joints"] if len(measured) < len(LANDMARK_MAP) else [])
                               + (["partial_time_coverage"] if partial_time else [])),
            "per_joint": support,
            "comparability_rule": "a pair counts only when the joint's whole feature group is "
                                  "finite on BOTH sides and the joint's own two dimensions are "
                                  "finite; unmeasured joints are listed, never filled with zero",
            "no_pass_line": "partial is reported with its numbers; no threshold decides whether "
                            "the comparison was sufficient",
        },
        "frame_features": pose,
        "frame_comparison": {
            "per_joint": frame_cmp,
            "meaning": "true when this joint was actually compared on the drawn pair; a joint can "
                       "have coordinates to draw and still have no comparison behind it",
            "compared": sum(1 for v in frame_cmp.values() if v),
            "drawn": len(frame_cmp)},
        "reference_frame_features": ref_pose,
        "frame_joint_deviation": frame_dev,
        "frame_spaces": {
            "per_group": GROUP_SPACE,
            "per_landmark": {lm: GROUP_SPACE[g] for lm, (g, _) in LANDMARK_MAP.items()},
            "panel_groups": PANEL_GROUPS,
            "rule": "hand-unit and body-unit coordinates are never plotted in one panel. shape_* "
                    "is (hand point - wrist) / hand scale, so its wrist is the origin by "
                    "construction; handpos_* and arm_* are body-normalised and belong to the axis "
                    "bars, not to a hand panel.",
            "wrist_landmarks_are_not_panel_points": [33, 54]},
        "frame_index": int(i),
        "frame_pair": [int(i), int(j)],
        "unit": "the median different-word difference for that joint's GROUP, measured over the "
                "reference set. Per-joint denominators were not used: a joint that barely moves "
                "between words has a tiny one and would exaggerate every difference.",
        "note": "a difference map, not a verdict. Colour by magnitude; never colour "
                "correct/incorrect, because no measured boundary exists.",
    }
